# Ported from scikit-learn 1.9.1 (BSD-3-Clause)

# AgglomerativeClustering without connectivity: the full tree is built as
# scikit-learn does through scipy.cluster.hierarchy (nearest-neighbor chain
# for ward/complete/average, then stable sort + union-find relabel) and the
# Prim MST + single_linkage_label path for single linkage.

import numpy as np
from heapq import heappush, heappushpop
from ._kmeans import _as2d
from numpy.random.legacy import fmuladd

def _pdist(X: np.ndarray[float, 2], metric: str) -> np.ndarray[float, 2]:
    n, d = X.shape
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            r = 0.0
            if metric == "euclidean" or metric == "l2":
                for k in range(d):
                    # compiled C accumulates with a fused multiply-add
                    a = X[i, k] - X[j, k]
                    r = fmuladd(a, a, r)
                r = np.sqrt(r)
            elif metric == "manhattan" or metric == "l1" or metric == "cityblock":
                for k in range(d):
                    r += abs(X[i, k] - X[j, k])
            elif metric == "chebyshev":
                for k in range(d):
                    a = abs(X[i, k] - X[j, k])
                    if a > r:
                        r = a
            elif metric == "cosine":
                uv = 0.0; uu = 0.0; vv = 0.0
                for k in range(d):
                    uv += X[i, k] * X[j, k]
                    uu += X[i, k] * X[i, k]
                    vv += X[j, k] * X[j, k]
                r = 1.0 - uv / np.sqrt(uu * vv)
                if r < 0.0:
                    r = 0.0
                if r > 2.0:
                    r = 2.0
            elif metric == "precomputed":
                r = X[i, j]
            else:
                raise ValueError(f"Unknown metric {metric}")
            D[i, j] = r
            D[j, i] = r
    return D

def _relabel(Z: np.ndarray[float, 2], n: int):
    # scipy _hierarchy.label: union-find relabel of a sorted linkage
    parent = np.arange(2 * n - 1)
    size = np.ones(2 * n - 1, dtype=int)
    nxt = n
    def find(parent, x: int) -> int:
        p = x
        while parent[x] != x:
            x = int(parent[x])
        while parent[p] != x:
            q = int(parent[p])
            parent[p] = x
            p = q
        return x
    for i in range(n - 1):
        xr = find(parent, int(Z[i, 0]))
        yr = find(parent, int(Z[i, 1]))
        if xr < yr:
            Z[i, 0] = float(xr); Z[i, 1] = float(yr)
        else:
            Z[i, 0] = float(yr); Z[i, 1] = float(xr)
        parent[xr] = nxt
        parent[yr] = nxt
        size[nxt] = size[xr] + size[yr]
        Z[i, 3] = float(size[nxt])
        nxt += 1

def _nn_chain(D: np.ndarray[float, 2], n: int, method: str) -> np.ndarray[float, 2]:
    D = D.copy()
    Z = np.empty((n - 1, 4))
    size = np.ones(n, dtype=int)
    chain = np.empty(n, dtype=int)
    clen = 0
    for k in range(n - 1):
        if clen == 0:
            clen = 1
            for i in range(n):
                if size[i] > 0:
                    chain[0] = i
                    break
        x = 0
        y = 0
        cur = 0.0
        while True:
            x = int(chain[clen - 1])
            if clen > 1:
                y = int(chain[clen - 2])
                cur = D[x, y]
            else:
                cur = np.inf
            for i in range(n):
                if size[i] == 0 or x == i:
                    continue
                dd = D[x, i]
                if dd < cur:
                    cur = dd
                    y = i
            if clen > 1 and y == chain[clen - 2]:
                break
            chain[clen] = y
            clen += 1
        clen -= 2
        if x > y:
            x, y = y, x
        nx = size[x]
        ny = size[y]
        Z[k, 0] = float(x); Z[k, 1] = float(y); Z[k, 2] = cur; Z[k, 3] = float(nx + ny)
        size[x] = 0
        size[y] = nx + ny
        for i in range(n):
            ni = size[i]
            if ni == 0 or i == y:
                continue
            dx = D[i, x]
            dy = D[i, y]
            v = 0.0
            if method == "complete":
                v = dx if dx > dy else dy
            elif method == "average":
                v = (nx * dx + ny * dy) / float(nx + ny)
            else:
                t = 1.0 / float(nx + ny + ni)
                v = np.sqrt((ni + nx) * t * dx * dx + (ni + ny) * t * dy * dy - ni * t * cur * cur)
            D[i, y] = v
            D[y, i] = v
    order = np.argsort(Z[:, 2], kind="stable")
    Z = np.ascontiguousarray(Z[order])
    _relabel(Z, n)
    return Z

def _single(D: np.ndarray[float, 2], n: int) -> np.ndarray[float, 2]:
    # sklearn mst_linkage_core + mergesort + _single_linkage_label
    in_tree = [False] * n
    mst = np.zeros((n - 1, 3))
    cur_d = np.full(n, np.inf)
    node = 0
    for i in range(n - 1):
        in_tree[node] = True
        best = np.inf
        new_node = 0
        for j in range(n):
            if in_tree[j]:
                continue
            lv = D[node, j]
            if lv < cur_d[j]:
                cur_d[j] = lv
            if cur_d[j] < best:
                best = cur_d[j]
                new_node = j
        mst[i, 0] = float(node); mst[i, 1] = float(new_node); mst[i, 2] = best
        node = new_node
    order = np.argsort(mst[:, 2], kind="stable")
    mst = mst[order]
    Z = np.zeros((n - 1, 4))
    parent = np.full(2 * n - 1, -1, dtype=int)
    size = np.zeros(2 * n - 1, dtype=int)
    for i in range(n):
        size[i] = 1
    nxt = n
    def ffind(parent, x: int) -> int:
        p = x
        while parent[x] != -1:
            x = int(parent[x])
        while parent[p] != x and p != x:
            q = int(parent[p])
            parent[p] = x
            p = q
        return x
    for i in range(n - 1):
        lc = ffind(parent, int(mst[i, 0]))
        rc = ffind(parent, int(mst[i, 1]))
        Z[i, 0] = float(lc); Z[i, 1] = float(rc); Z[i, 2] = mst[i, 2]
        Z[i, 3] = float(size[lc] + size[rc])
        parent[lc] = nxt
        parent[rc] = nxt
        size[nxt] = size[lc] + size[rc]
        nxt += 1
    return Z

def _hc_cut(n_clusters: int, children: np.ndarray[int, 2], n_leaves: int) -> np.ndarray[int, 1]:
    if n_clusters > n_leaves:
        raise ValueError(f"Cannot extract more clusters than samples: {n_clusters} clusters were given for a tree with {n_leaves} leaves.")
    last = children.shape[0] - 1
    m = children[last, 0] if children[last, 0] > children[last, 1] else children[last, 1]
    nodes = [-(int(m) + 1)]
    for _ in range(n_clusters - 1):
        these = children[-nodes[0] - n_leaves]
        heappush(nodes, -int(these[0]))
        heappushpop(nodes, -int(these[1]))
    label = np.zeros(n_leaves, dtype=int)
    for i in range(len(nodes)):
        stack = [-nodes[i]]
        while len(stack) > 0:
            nd = stack.pop()
            if nd < n_leaves:
                label[nd] = i
            else:
                stack.append(int(children[nd - n_leaves, 0]))
                stack.append(int(children[nd - n_leaves, 1]))
    return label

def _tree(X: np.ndarray[float, 2], linkage: str, metric: str) -> np.ndarray[float, 2]:
    n = X.shape[0]
    if linkage == "ward":
        return _nn_chain(_pdist(X, "euclidean"), n, "ward")
    D = _pdist(X, metric)
    if linkage == "single":
        return _single(D, n)
    return _nn_chain(D, n, linkage)

class AgglomerativeClustering:
    n_clusters: Optional[int]
    metric: str
    linkage: str
    distance_threshold: Optional[float]
    compute_distances: bool
    has_conn: bool
    connectivity_: np.ndarray[float, 2]
    full_tree: int
    n_clusters_: int
    labels_: np.ndarray[int, 1]
    children_: np.ndarray[int, 2]
    distances_: np.ndarray[float, 1]
    n_leaves_: int
    n_connected_components_: int
    n_features_in_: int

    def __init__(self, n_clusters: Optional[int] = 2, *, metric: str = "euclidean", memory = None,
                 connectivity = None, compute_full_tree = "auto", linkage: str = "ward",
                 distance_threshold: Optional[float] = None, compute_distances: bool = False):
        if linkage != "ward" and linkage != "complete" and linkage != "average" and linkage != "single":
            raise ValueError(f"Unknown linkage type {linkage}")
        self.n_clusters = n_clusters
        self.metric = metric
        self.linkage = linkage
        self.distance_threshold = distance_threshold
        self.compute_distances = compute_distances
        self.has_conn = connectivity is not None
        self.connectivity_ = np.empty((0, 0))
        if connectivity is not None:
            self.connectivity_ = np.asarray(connectivity, dtype=float)
        self.full_tree = -1
        if isinstance(compute_full_tree, bool):
            self.full_tree = 1 if compute_full_tree else 0
        self.n_clusters_ = 0
        self.labels_ = np.empty(0, dtype=int)
        self.children_ = np.empty((0, 2), dtype=int)
        self.distances_ = np.empty(0)
        self.n_leaves_ = 0
        self.n_connected_components_ = 1
        self.n_features_in_ = 0

    def fit(self, X, y = None):
        Xa = _as2d(X)
        if not ((self.n_clusters is None) ^ (self.distance_threshold is None)):
            raise ValueError("Exactly one of n_clusters and distance_threshold has to be set, and the other needs to be None.")
        if self.linkage == "ward" and self.metric != "euclidean" and self.metric != "l2":
            raise ValueError(f"{self.metric} was provided as metric. Ward can only work with euclidean distances (i.e. 'euclidean' and 'l2').")
        n = Xa.shape[0]
        if self.has_conn:
            return self._fit_structured(Xa)
        Z = _tree(Xa, self.linkage, self.metric)
        ch = np.empty((n - 1, 2), dtype=int)
        dist = np.empty(n - 1)
        for i in range(n - 1):
            ch[i, 0] = int(Z[i, 0]); ch[i, 1] = int(Z[i, 1]); dist[i] = Z[i, 2]
        self.children_ = ch
        self.n_leaves_ = n
        if self.distance_threshold is not None or self.compute_distances:
            self.distances_ = dist
        if self.distance_threshold is not None:
            thr = self.distance_threshold
            c = 0
            for i in range(n - 1):
                if dist[i] >= thr:
                    c += 1
            self.n_clusters_ = c + 1
        else:
            self.n_clusters_ = self.n_clusters
        self.labels_ = _hc_cut(self.n_clusters_, ch, n)
        self.n_features_in_ = Xa.shape[1]
        return self

    def _fit_structured(self, Xa: np.ndarray[float, 2]):
        from ._structured import ward_tree, linkage_tree, _hc_get_heads
        n = Xa.shape[0]
        if self.distance_threshold is not None and self.full_tree == 0:
            raise ValueError("compute_full_tree must be True if distance_threshold is set.")
        full = True
        if self.full_tree == 1:
            full = True
        elif self.full_tree == 0:
            full = False
        else:
            if self.distance_threshold is not None:
                full = True
            else:
                full = float(self.n_clusters.__val__()) < max(100.0, 0.02 * n)
        ncl: Optional[int] = None
        if not full:
            ncl = self.n_clusters
        if self.linkage == "ward":
            ch, ncc, nl, parents, dist = ward_tree(Xa, connectivity=self.connectivity_, n_clusters=ncl)
        else:
            ch, ncc, nl, parents, dist = linkage_tree(Xa, self.connectivity_, ncl, self.linkage, self.metric)
        self.children_ = ch
        self.n_connected_components_ = ncc
        self.n_leaves_ = nl
        if self.distance_threshold is not None or self.compute_distances:
            self.distances_ = dist
        if self.distance_threshold is not None:
            c = 0
            for i in range(dist.shape[0]):
                if dist[i] >= self.distance_threshold.__val__():
                    c += 1
            self.n_clusters_ = c + 1
        else:
            self.n_clusters_ = self.n_clusters.__val__()
        if full:
            self.labels_ = _hc_cut(self.n_clusters_, ch, nl)
        else:
            heads = _hc_get_heads(parents)[:n].copy()
            self.labels_ = np.searchsorted(np.unique(heads), heads)
        self.n_features_in_ = Xa.shape[1]
        return self

    def fit_predict(self, X, y = None):
        return self.fit(X).labels_

class FeatureAgglomeration:
    _agg: AgglomerativeClustering
    labels_: np.ndarray[int, 1]
    n_clusters_: int
    children_: np.ndarray[int, 2]
    distances_: np.ndarray[float, 1]
    n_leaves_: int
    n_connected_components_: int
    n_features_in_: int

    def __init__(self, n_clusters: Optional[int] = 2, *, metric: str = "euclidean", memory = None,
                 connectivity = None, compute_full_tree = "auto", linkage: str = "ward",
                 pooling_func = None, distance_threshold: Optional[float] = None,
                 compute_distances: bool = False):
        self._agg = AgglomerativeClustering(n_clusters, metric=metric, connectivity=connectivity,
                                            compute_full_tree=compute_full_tree, linkage=linkage,
                                            distance_threshold=distance_threshold,
                                            compute_distances=compute_distances)
        self.labels_ = np.empty(0, dtype=int)
        self.n_clusters_ = 0
        self.children_ = np.empty((0, 2), dtype=int)
        self.distances_ = np.empty(0)
        self.n_leaves_ = 0
        self.n_connected_components_ = 1
        self.n_features_in_ = 0

    def fit(self, X, y = None):
        Xa = _as2d(X)
        if Xa.shape[1] < 2:
            raise ValueError("FeatureAgglomeration requires at least 2 features")
        self._agg.fit(np.ascontiguousarray(Xa.T))
        self.labels_ = self._agg.labels_
        self.n_clusters_ = self._agg.n_clusters_
        self.children_ = self._agg.children_
        self.distances_ = self._agg.distances_
        self.n_leaves_ = self._agg.n_leaves_
        self.n_connected_components_ = self._agg.n_connected_components_
        self.n_features_in_ = Xa.shape[1]
        return self

    def transform(self, X):
        Xa = _as2d(X)
        k = 0
        for v in self.labels_:
            if v + 1 > k:
                k = v + 1
        size = np.zeros(k)
        for v in self.labels_:
            size[v] += 1.0
        out = np.zeros((Xa.shape[0], k))
        for i in range(Xa.shape[0]):
            for f in range(Xa.shape[1]):
                out[i, self.labels_[f]] += Xa[i, f]
            for c in range(k):
                out[i, c] /= size[c]
        return out

    def fit_transform(self, X, y = None):
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        Xa = np.asarray(X, dtype=float)
        uniq = np.unique(self.labels_)
        inv = np.searchsorted(uniq, self.labels_)
        return Xa[..., inv]
