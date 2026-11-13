# Ported from scikit-learn 1.9.1 (BSD-3-Clause)

# HDBSCAN: core distances, Prim's minimum spanning tree over the mutual
# reachability graph (mst_from_data_matrix, also used for precomputed
# distances via the brute mutual-reachability MST), single-linkage tree,
# condensed tree, cluster stability, eom/leaf selection with
# cluster_selection_epsilon and max_cluster_size, labels/probabilities,
# centroids/medoids and dbscan_clustering(cut).

from ..base import estimator_repr as _estimator_repr
import numpy as np
from ._kmeans import _as2d
from ._optics import _pdist_row

class _UF:
    parent: List[int]
    size: List[int]
    next_label: int

    def __init__(self, n: int):
        self.parent = [-1] * (2 * n - 1)
        self.size = [1] * n + [0] * (n - 1)
        self.next_label = n

    def union(self, m: int, n: int):
        self.parent[m] = self.next_label
        self.parent[n] = self.next_label
        self.size[self.next_label] = self.size[m] + self.size[n]
        self.next_label += 1

    def fast_find(self, n: int) -> int:
        p = n
        while self.parent[n] != -1:
            n = self.parent[n]
        while self.parent[p] != n and p != n:
            q = self.parent[p]
            self.parent[p] = n
            p = q
        return n

def _mst_prims(D: np.ndarray[float, 2], core: np.ndarray[float, 1], alpha: float):
    n = D.shape[0]
    in_tree = [False] * n
    minr = [np.inf] * n
    src = [1] * n
    cur = 0
    ea = List[int]()
    eb = List[int]()
    ed = List[float]()
    for _ in range(n - 1):
        in_tree[cur] = True
        ccd = core[cur]
        newr = 1.7976931348623157e308
        sn = 0
        nn = 0
        for j in range(n):
            if in_tree[j]:
                continue
            nmr = minr[j]
            nsrc = src[j]
            pdist = D[cur, j] / alpha
            mr = max(ccd, core[j], pdist)
            if mr < nmr:
                minr[j] = mr
                src[j] = cur
                if mr < newr:
                    newr = mr
                    sn = cur
                    nn = j
            elif nmr < newr:
                newr = nmr
                sn = nsrc
                nn = j
        ea.append(sn)
        eb.append(nn)
        ed.append(newr)
        cur = nn
    return ea, eb, ed

def _mst_brute(M: np.ndarray[float, 2]):
    n = M.shape[0]
    labels = list(range(n))
    minr = [np.inf] * n
    cur = 0
    ea = List[int]()
    eb = List[int]()
    ed = List[float]()
    for _ in range(n - 1):
        nl = List[int]()
        nm = List[float]()
        for q in range(len(labels)):
            if labels[q] != cur:
                nl.append(labels[q])
                nm.append(minr[q])
        labels = nl
        minr = [min(nm[q], M[cur, labels[q]]) for q in range(len(labels))]
        bi = 0
        for q in range(1, len(minr)):
            if minr[q] < minr[bi]:
                bi = q
        nnode = labels[bi]
        ea.append(cur)
        eb.append(nnode)
        ed.append(minr[bi])
        cur = nnode
    return ea, eb, ed

def _single_linkage(ea: List[int], eb: List[int], ed: List[float]):
    m = len(ed)
    edv = np.empty(m)
    for i in range(m):
        edv[i] = ed[i]
    order = np.argsort(edv)
    n = m + 1
    U = _UF(n)
    L = np.empty((m, 4))
    for i in range(m):
        k = int(order[i])
        a = U.fast_find(ea[k])
        b = U.fast_find(eb[k])
        L[i, 0] = float(a)
        L[i, 1] = float(b)
        L[i, 2] = ed[k]
        L[i, 3] = float(U.size[a] + U.size[b])
        U.union(a, b)
    return L

def _bfs_hier(H: np.ndarray[float, 2], root: int) -> List[int]:
    n = H.shape[0] + 1
    res = List[int]()
    q = [root]
    while len(q) > 0:
        res.extend(q)
        nq = List[int]()
        for x in q:
            if x >= n:
                nq.append(int(H[x - n, 0]))
                nq.append(int(H[x - n, 1]))
        q = nq
    return res

class _Condensed:
    parent: List[int]
    child: List[int]
    value: List[float]
    size: List[int]

    def __init__(self):
        self.parent = List[int]()
        self.child = List[int]()
        self.value = List[float]()
        self.size = List[int]()

    def add(self, p: int, c: int, v: float, s: int):
        self.parent.append(p)
        self.child.append(c)
        self.value.append(v)
        self.size.append(s)

    def __len__(self):
        return len(self.parent)

def _condense(H: np.ndarray[float, 2], mcs: int) -> _Condensed:
    root = 2 * H.shape[0]
    n = H.shape[0] + 1
    nxt = n + 1
    nodes = _bfs_hier(H, root)
    relabel = [0] * (root + 1)
    relabel[root] = n
    ignore = [False] * len(nodes)
    out = _Condensed()
    for node in nodes:
        if ignore[node] or node < n:
            continue
        left = int(H[node - n, 0])
        right = int(H[node - n, 1])
        dist = H[node - n, 2]
        lam = 1.0 / dist if dist > 0.0 else np.inf
        lc = int(H[left - n, 3]) if left >= n else 1
        rc = int(H[right - n, 3]) if right >= n else 1
        if lc >= mcs and rc >= mcs:
            relabel[left] = nxt
            nxt += 1
            out.add(relabel[node], relabel[left], lam, lc)
            relabel[right] = nxt
            nxt += 1
            out.add(relabel[node], relabel[right], lam, rc)
        elif lc < mcs and rc < mcs:
            for s in _bfs_hier(H, left):
                if s < n:
                    out.add(relabel[node], s, lam, 1)
                ignore[s] = True
            for s in _bfs_hier(H, right):
                if s < n:
                    out.add(relabel[node], s, lam, 1)
                ignore[s] = True
        elif lc < mcs:
            relabel[right] = relabel[node]
            for s in _bfs_hier(H, left):
                if s < n:
                    out.add(relabel[node], s, lam, 1)
                ignore[s] = True
        else:
            relabel[left] = relabel[node]
            for s in _bfs_hier(H, right):
                if s < n:
                    out.add(relabel[node], s, lam, 1)
                ignore[s] = True
    return out

def _stability(T: _Condensed) -> Dict[int, float]:
    largest_child = max(T.child)
    smallest = min(T.parent)
    nclust = max(T.parent) - smallest + 1
    largest_child = max(largest_child, smallest)
    births = [np.nan] * (largest_child + 1)
    for i in range(len(T)):
        births[T.child[i]] = T.value[i]
    births[smallest] = 0.0
    res = [0.0] * nclust
    for i in range(len(T)):
        p = T.parent[i]
        res[p - smallest] += (T.value[i] - births[p]) * T.size[i]
    out = Dict[int, float]()
    for i in range(nclust):
        out[i + smallest] = res[i]
    return out

def _subtree(T: _Condensed, keep: List[bool]) -> _Condensed:
    o = _Condensed()
    for i in range(len(T)):
        if keep[i]:
            o.add(T.parent[i], T.child[i], T.value[i], T.size[i])
    return o

def _bfs_cluster(T: _Condensed, root: int) -> List[int]:
    res = List[int]()
    q = [root]
    while len(q) > 0:
        res.extend(q)
        qs = set(q)
        nq = List[int]()
        for i in range(len(T)):
            if T.parent[i] in qs:
                nq.append(T.child[i])
        q = nq
    return res

def _leaves(T: _Condensed, node: int) -> List[int]:
    ch = [T.child[i] for i in range(len(T)) if T.parent[i] == node]
    if len(ch) == 0:
        return [node]
    out = List[int]()
    for c in ch:
        out.extend(_leaves(T, c))
    return out

def _traverse_upwards(T: _Condensed, eps: float, leaf: int, allow_single: bool) -> int:
    root = min(T.parent)
    parent = -1
    for i in range(len(T)):
        if T.child[i] == leaf:
            parent = T.parent[i]
    if parent == root:
        return parent if allow_single else leaf
    pe = 0.0
    for i in range(len(T)):
        if T.child[i] == parent:
            pe = 1.0 / T.value[i]
    if pe > eps:
        return parent
    return _traverse_upwards(T, eps, parent, allow_single)

def _epsilon_search(leaves: List[int], T: _Condensed, eps: float, allow_single: bool) -> Set[int]:
    sel = List[int]()
    processed = List[int]()
    for leaf in leaves:
        e = 0.0
        for i in range(len(T)):
            if T.child[i] == leaf:
                e = 1.0 / T.value[i]
                break
        if e < eps:
            if leaf not in processed:
                ec = _traverse_upwards(T, eps, leaf, allow_single)
                sel.append(ec)
                for s in _bfs_cluster(T, ec):
                    if s != ec:
                        processed.append(s)
        else:
            sel.append(leaf)
    return set(sel)

class _TUF:
    data0: List[int]
    data1: List[int]

    def __init__(self, n: int):
        self.data0 = list(range(n))
        self.data1 = [0] * n

    def find(self, x: int) -> int:
        if self.data0[x] != x:
            self.data0[x] = self.find(self.data0[x])
        return self.data0[x]

    def union(self, x: int, y: int):
        xr = self.find(x)
        yr = self.find(y)
        if self.data1[xr] < self.data1[yr]:
            self.data0[xr] = yr
        elif self.data1[xr] > self.data1[yr]:
            self.data0[yr] = xr
        else:
            self.data0[yr] = xr
            self.data1[xr] += 1

def _get_clusters(T: _Condensed, stab: Dict[int, float], method: str, allow_single: bool,
                  eps: float, max_cluster_size: Optional[int]):
    keys = sorted(list(stab.keys()), reverse=True)
    node_list = keys if allow_single else keys[:-1]
    ct = _subtree(T, [T.size[i] > 1 for i in range(len(T))])
    is_cluster = Dict[int, bool]()
    for c in node_list:
        is_cluster[c] = True
    n_samples = 0
    for i in range(len(T)):
        if T.size[i] == 1 and T.child[i] + 1 > n_samples:
            n_samples = T.child[i] + 1
    mcs = n_samples + 1 if max_cluster_size is None else max_cluster_size.__val__()
    sizes = Dict[int, int]()
    for i in range(len(ct)):
        sizes[ct.child[i]] = ct.size[i]
    if allow_single and len(node_list) > 0:
        s = 0
        for i in range(len(ct)):
            if ct.parent[i] == node_list[-1]:
                s += ct.size[i]
        sizes[node_list[-1]] = s
    if method == "eom":
        for node in node_list:
            sub = 0.0
            for i in range(len(ct)):
                if ct.parent[i] == node:
                    sub += stab[ct.child[i]]
            if sub > stab[node] or sizes.get(node, 0) > mcs:
                is_cluster[node] = False
                stab[node] = sub
            else:
                for s in _bfs_cluster(ct, node):
                    if s != node:
                        is_cluster[s] = False
        if eps != 0.0 and len(ct) > 0:
            eom = [c for c in is_cluster if is_cluster[c]]
            sel = Set[int]()
            if len(eom) == 1 and eom[0] == min(ct.parent):
                if allow_single:
                    sel = set(eom)
            else:
                sel = _epsilon_search(eom, ct, eps, allow_single)
            for c in is_cluster:
                is_cluster[c] = c in sel
    else:
        leaves = List[int]()
        if len(ct) > 0:
            leaves = _leaves(ct, min(ct.parent))
        if len(leaves) == 0:
            for c in is_cluster:
                is_cluster[c] = False
            is_cluster[min(T.parent)] = True
        sel = set(leaves)
        if eps != 0.0:
            sel = _epsilon_search(leaves, ct, eps, allow_single)
        for c in is_cluster:
            is_cluster[c] = c in sel
    clusters = sorted([c for c in is_cluster if is_cluster[c]])
    cmap = Dict[int, int]()
    rmap = Dict[int, int]()
    for q in range(len(clusters)):
        cmap[clusters[q]] = q
        rmap[q] = clusters[q]
    cset = set(clusters)
    root = min(T.parent)
    U = _TUF(max(T.parent) + 1)
    for i in range(len(T)):
        if T.child[i] not in cset:
            U.union(T.parent[i], T.child[i])
    labels = np.empty(root, dtype=int)
    for n in range(root):
        c = U.find(n)
        lab = -1
        if c != root:
            lab = cmap[c]
        elif len(cset) == 1 and allow_single:
            pl = 0.0
            for i in range(len(T)):
                if T.child[i] == n:
                    pl = T.value[i]
            th = 0.0
            if eps != 0.0:
                th = 1.0 / eps
            else:
                th = -np.inf
                for i in range(len(T)):
                    if T.parent[i] == c and T.value[i] > th:
                        th = T.value[i]
            if pl >= th:
                lab = cmap[c]
        labels[n] = lab
    # probabilities
    deaths = [0.0] * (max(T.parent) + 1)
    cp = T.parent[0]
    ml = T.value[0]
    for i in range(1, len(T)):
        if T.parent[i] == cp:
            ml = max(ml, T.value[i])
        else:
            deaths[cp] = ml
            cp = T.parent[i]
            ml = T.value[i]
    deaths[cp] = ml
    probs = np.zeros(root)
    for i in range(len(T)):
        pt = T.child[i]
        if pt >= root:
            continue
        cn = labels[pt]
        if cn == -1:
            continue
        mx = deaths[rmap[cn]]
        if mx == 0.0 or T.value[i] == np.inf:
            probs[pt] = 1.0
        else:
            probs[pt] = min(T.value[i], mx) / mx
    return labels, probs

class HDBSCAN:
    def __repr__(self) -> str:
        return _estimator_repr(self)

    min_cluster_size: int
    min_samples: Optional[int]
    cluster_selection_epsilon: float
    max_cluster_size: Optional[int]
    metric: str
    alpha: float
    cluster_selection_method: str
    allow_single_cluster: bool
    store_centers: Optional[str]
    labels_: np.ndarray[int, 1]
    probabilities_: np.ndarray[float, 1]
    centroids_: np.ndarray[float, 2]
    medoids_: np.ndarray[float, 2]
    _single_linkage_tree_: np.ndarray[float, 2]
    n_features_in_: int

    def __init__(self, min_cluster_size: int = 5, min_samples: Optional[int] = None,
                 cluster_selection_epsilon: float = 0.0, max_cluster_size: Optional[int] = None,
                 metric: str = "euclidean", metric_params = None, alpha: float = 1.0,
                 algorithm: str = "auto", leaf_size: int = 40, n_jobs = None,
                 cluster_selection_method: str = "eom", allow_single_cluster: bool = False,
                 store_centers: Optional[str] = None, copy = "warn"):
        if min_cluster_size < 2:
            raise ValueError("min_cluster_size must be >= 2")
        if cluster_selection_method != "eom" and cluster_selection_method != "leaf":
            raise ValueError(f"cluster_selection_method must be 'eom' or 'leaf', got {cluster_selection_method}")
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.cluster_selection_epsilon = cluster_selection_epsilon
        self.max_cluster_size = max_cluster_size
        self.metric = metric
        self.alpha = alpha
        self.cluster_selection_method = cluster_selection_method
        self.allow_single_cluster = allow_single_cluster
        self.store_centers = store_centers
        self.labels_ = np.empty(0, dtype=int)
        self.probabilities_ = np.empty(0)
        self.centroids_ = np.empty((0, 0))
        self.medoids_ = np.empty((0, 0))
        self._single_linkage_tree_ = np.empty((0, 4))
        self.n_features_in_ = 0

    def fit(self, X, y = None):
        Xa = _as2d(X)
        n = Xa.shape[0]
        if n == 1:
            raise ValueError("n_samples=1 while HDBSCAN requires more than one sample")
        ms = self.min_cluster_size if self.min_samples is None else self.min_samples.__val__()
        if ms > n:
            raise ValueError(f"min_samples ({ms}) must be at most the number of samples in X ({n})")
        D = np.empty((n, n))
        for i in range(n):
            D[i] = _pdist_row(Xa, i, "euclidean" if self.metric == "l2" else self.metric, 2.0)
        if self.metric == "precomputed":
            M = D / self.alpha
            core = np.empty(n)
            for i in range(n):
                core[i] = np.sort(M[i])[ms - 1]
            for i in range(n):
                for j in range(n):
                    M[i, j] = max(core[i], core[j], M[i, j])
            ea, eb, ed = _mst_brute(M)
        else:
            core = np.empty(n)
            for i in range(n):
                core[i] = np.sort(D[i])[ms - 1]
            ea, eb, ed = _mst_prims(D, core, self.alpha)
        H = _single_linkage(ea, eb, ed)
        self._single_linkage_tree_ = H
        T = _condense(H, self.min_cluster_size)
        lab, prob = _get_clusters(T, _stability(T), self.cluster_selection_method,
                                  self.allow_single_cluster, self.cluster_selection_epsilon,
                                  self.max_cluster_size)
        self.labels_ = lab
        self.probabilities_ = prob
        self.n_features_in_ = Xa.shape[1]
        if self.store_centers is not None:
            sc = self.store_centers.__val__()
            nc = 0
            for v in lab:
                if v + 1 > nc:
                    nc = v + 1
            self.centroids_ = np.empty((nc, Xa.shape[1]))
            self.medoids_ = np.empty((nc, Xa.shape[1]))
            for c in range(nc):
                idx = [i for i in range(n) if lab[i] == c]
                w = np.array([prob[i] for i in idx])
                sub = np.ascontiguousarray(Xa[np.array(idx)])
                if sc == "centroid" or sc == "both":
                    self.centroids_[c] = (sub * w[:, None]).sum(axis=0) / w.sum()
                if sc == "medoid" or sc == "both":
                    best = 0
                    bv = np.inf
                    for a in range(len(idx)):
                        s = 0.0
                        for b in range(len(idx)):
                            s += D[idx[a], idx[b]] * w[b]
                        if s < bv:
                            bv = s
                            best = a
                    self.medoids_[c] = sub[best]
        return self

    def fit_predict(self, X, y = None):
        return self.fit(X).labels_

    def dbscan_clustering(self, cut_distance: float, min_cluster_size: int = 5):
        H = self._single_linkage_tree_
        root = 2 * H.shape[0]
        n = root // 2 + 1
        U = _TUF(root + 1)
        cl = n
        for i in range(H.shape[0]):
            if H[i, 2] < cut_distance:
                U.union(int(H[i, 0]), cl)
                U.union(int(H[i, 1]), cl)
            cl += 1
        size = [0] * cl
        res = [0] * n
        for i in range(n):
            c = U.find(i)
            size[c] += 1
            res[i] = c
        cmap = Dict[int, int]()
        lab = 0
        for c in sorted(set(res)):
            if size[c] < min_cluster_size:
                cmap[c] = -1
            else:
                cmap[c] = lab
                lab += 1
        out = np.empty(n, dtype=int)
        for i in range(n):
            out[i] = cmap[res[i]]
        return out
