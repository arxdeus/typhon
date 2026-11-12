# Ported from scikit-learn 1.9.1 (BSD-3-Clause)

# Structured (connectivity-constrained) hierarchical clustering: ward_tree
# and linkage_tree (complete/average via max/average merges, single via
# MST + single_linkage_label) on a dense connectivity matrix, with
# _fix_connectivity completing disconnected graphs; hc_get_heads early
# stopping; FeatureAgglomeration (mean pooling).

import numpy as np
from heapq import heappush, heappop, heapify
from ._kmeans import _as2d

class _WEdge:
    weight: float
    a: int
    b: int

    def __init__(self, w: float, a: int, b: int):
        self.weight = w
        self.a = a
        self.b = b

    # WeightedEdge compares on weight only
    def __lt__(self, o: _WEdge) -> bool:
        return self.weight < o.weight
    def __le__(self, o: _WEdge) -> bool:
        return self.weight <= o.weight
    def __gt__(self, o: _WEdge) -> bool:
        return self.weight > o.weight
    def __ge__(self, o: _WEdge) -> bool:
        return self.weight >= o.weight
    def __eq__(self, o: _WEdge) -> bool:
        return self.weight == o.weight
    def __ne__(self, o: _WEdge) -> bool:
        return self.weight != o.weight

def _pair_d(X: np.ndarray[float, 2], i: int, j: int, metric: str) -> float:
    r = 0.0
    d = X.shape[1]
    if metric == "precomputed":
        return X[i, j]
    if metric == "manhattan" or metric == "cityblock" or metric == "l1":
        for k in range(d):
            r += abs(X[i, k] - X[j, k])
        return r
    elif metric == "cosine":
        a = 0.0
        b = 0.0
        for k in range(d):
            r += X[i, k] * X[j, k]
            a += X[i, k] * X[i, k]
            b += X[j, k] * X[j, k]
        return 1.0 - r / (np.sqrt(a) * np.sqrt(b))
    for k in range(d):
        t = X[i, k] - X[j, k]
        r += t * t
    return np.sqrt(r)

def _components(C: np.ndarray[bool, 2]):
    n = C.shape[0]
    lab = [-1] * n
    nc = 0
    for s in range(n):
        if lab[s] >= 0:
            continue
        stack = [s]
        lab[s] = nc
        while len(stack) > 0:
            u = stack.pop()
            for v in range(n):
                if C[u, v] and lab[v] < 0:
                    lab[v] = nc
                    stack.append(v)
        nc += 1
    return nc, lab

def _fix_connectivity(X: np.ndarray[float, 2], conn, metric: str):
    n = X.shape[0]
    c0 = np.asarray(conn)
    if c0.shape[0] != n or c0.shape[1] != n:
        raise ValueError(f"Wrong shape for connectivity matrix: {c0.shape} when X is {X.shape}")
    C = (c0.astype(float) + c0.astype(float).T) != 0.0
    W = c0.astype(float) + c0.astype(float).T
    nc, lab = _components(C)
    if nc > 1:
        for i in range(nc):
            ii = [q for q in range(n) if lab[q] == i]
            for j in range(i):
                jj = [q for q in range(n) if lab[q] == j]
                best = np.inf
                bi = 0
                bj = 0
                for a in range(len(ii)):
                    for b in range(len(jj)):
                        dv = _pair_d(X, ii[a], jj[b], metric)
                        if dv < best:
                            best = dv
                            bi = ii[a]
                            bj = jj[b]
                C[bi, bj] = True
                C[bj, bi] = True
                W[bi, bj] = 1.0
                W[bj, bi] = 1.0
    return C, nc

def _get_parents(nodes: List[int], heads: List[int], parent: np.ndarray[int, 1],
                 not_visited: np.ndarray[bool, 1]):
    for node0 in nodes:
        node = node0
        p = parent[node]
        while p != node:
            node = p
            p = parent[node]
        if not_visited[node]:
            not_visited[node] = False
            heads.append(node)

def ward_tree(X, *, connectivity = None, n_clusters: Optional[int] = None):
    Xa = _as2d(X)
    n, nf = Xa.shape
    C, ncc = _fix_connectivity(Xa, connectivity, "euclidean")
    n_nodes = 2 * n - 1 if n_clusters is None else 2 * n - n_clusters.__val__()
    A = List[List[int]]()
    crow = List[int]()
    ccol = List[int]()
    for ind in range(n):
        row = [q for q in range(n) if C[ind, q]]
        A.append(row)
        for q in row:
            if q < ind:
                crow.append(ind)
                ccol.append(q)
    m1 = np.zeros(n_nodes)
    m1[:n] = 1.0
    m2 = np.zeros((n_nodes, nf))
    m2[:n] = Xa

    def wdist(r: int, c: int) -> float:
        nn = (m1[r] * m1[c]) / (m1[r] + m1[c])
        pa = 0.0
        for j in range(nf):
            pa += (m2[r, j] / m1[r] - m2[c, j] / m1[c]) ** 2
        return pa * nn

    heap = [(wdist(crow[q], ccol[q]), crow[q], ccol[q]) for q in range(len(crow))]
    heapify(heap)
    parent = np.arange(n_nodes)
    used = np.ones(n_nodes, dtype=bool)
    children = List[Tuple[int, int]]()
    dist = np.empty(n_nodes - n)
    nv = np.empty(n_nodes, dtype=bool)
    for k in range(n, n_nodes):
        inert = 0.0
        i = 0
        j = 0
        while True:
            inert, i, j = heappop(heap)
            if used[i] and used[j]:
                break
        parent[i] = k
        parent[j] = k
        children.append((i, j))
        used[i] = False
        used[j] = False
        dist[k - n] = inert
        m1[k] = m1[i] + m1[j]
        m2[k] = m2[i] + m2[j]
        cc = List[int]()
        nv[:] = True
        nv[k] = False
        _get_parents(A[i], cc, parent, nv)
        _get_parents(A[j], cc, parent, nv)
        for col in cc:
            A[col].append(k)
        A.append(cc)
        for col in cc:
            heappush(heap, (wdist(k, col), k, col))
    ch = np.empty((len(children), 2), dtype=int)
    for q in range(len(children)):
        ch[q, 0] = children[q][1]
        ch[q, 1] = children[q][0]
    return ch, ncc, n, parent, np.sqrt(2.0 * dist)

def _single_linkage_label(L: np.ndarray[float, 2]):
    m = L.shape[0]
    n = m + 1
    parent = [-1] * (2 * n - 1)
    size = [1] * n + [0] * (n - 1)
    nxt = n
    out = np.empty((m, 4))

    def find(x: int) -> int:
        p = x
        while parent[x] != -1:
            x = parent[x]
        while parent[p] != x and p != x:
            q = parent[p]
            parent[p] = x
            p = q
        return x

    for i in range(m):
        a = int(L[i, 0])
        b = int(L[i, 1])
        ra = find(a)
        rb = find(b)
        out[i, 0] = float(ra)
        out[i, 1] = float(rb)
        out[i, 2] = L[i, 2]
        out[i, 3] = float(size[ra] + size[rb])
        parent[ra] = nxt
        parent[rb] = nxt
        size[nxt] = size[ra] + size[rb]
        nxt += 1
    return out

def _mst_sparse(W: np.ndarray[float, 2], C: np.ndarray[bool, 2]):
    # scipy minimum_spanning_tree (Kruskal on edges sorted by weight, stable)
    n = W.shape[0]
    er = List[int]()
    ec = List[int]()
    ew = List[float]()
    for i in range(n):
        for j in range(n):
            if C[i, j] and i != j:
                er.append(i)
                ec.append(j)
                ew.append(W[i, j])
    m = len(ew)
    wv = np.empty(m)
    for q in range(m):
        wv[q] = ew[q]
    order = np.argsort(wv, kind="stable")
    par = list(range(n))

    def f(x: int) -> int:
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x

    rows = List[int]()
    cols = List[int]()
    ws = List[float]()
    for o in order:
        q = int(o)
        a = f(er[q])
        b = f(ec[q])
        if a != b:
            par[a] = b
            rows.append(er[q])
            cols.append(ec[q])
            ws.append(ew[q])
    # csr->coo ordering: by row then col
    idx = list(range(len(rows)))
    idx.sort(key=lambda t: (rows[t], cols[t]))
    M = np.empty((len(idx), 3))
    for t in range(len(idx)):
        M[t, 0] = float(rows[idx[t]])
        M[t, 1] = float(cols[idx[t]])
        M[t, 2] = ws[idx[t]]
    return M

def linkage_tree(X, connectivity = None, n_clusters: Optional[int] = None, linkage: str = "complete",
                 affinity: str = "euclidean"):
    Xa = _as2d(X)
    n = Xa.shape[0]
    C, ncc = _fix_connectivity(Xa, connectivity, affinity)
    for i in range(n):
        C[i, i] = False
    W = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if C[i, j]:
                W[i, j] = _pair_d(Xa, i, j, affinity)
    n_nodes = 2 * n - 1 if n_clusters is None else 2 * n - n_clusters.__val__()
    if linkage == "single":
        eps = np.finfo(float).eps
        W2 = W.copy()
        for i in range(n):
            for j in range(n):
                if C[i, j] and W2[i, j] == 0.0:
                    W2[i, j] = eps
        M = _mst_sparse(W2, C)
        for t in range(M.shape[0]):
            if M[t, 2] == eps:
                M[t, 2] = 0.0
        order = np.argsort(M[:, 2], kind="mergesort")
        M = np.ascontiguousarray(M[order])
        Z = _single_linkage_label(M)
        ch = np.empty((Z.shape[0], 2), dtype=int)
        for t in range(Z.shape[0]):
            ch[t, 0] = int(Z[t, 0])
            ch[t, 1] = int(Z[t, 1])
        parent = np.arange(n_nodes)
        for t in range(Z.shape[0]):
            i = n + t
            if n_clusters is not None and i >= n_nodes:
                break
            if ch[t, 0] < n_nodes:
                parent[ch[t, 0]] = i
            if ch[t, 1] < n_nodes:
                parent[ch[t, 1]] = i
        return ch, ncc, n, parent, Z[:, 2].copy()
    A = List[Dict[int, float]]()
    heap = List[_WEdge]()
    for ind in range(n):
        d = Dict[int, float]()
        for r in range(n):
            if C[ind, r]:
                d[r] = W[ind, r]
                if r < ind:
                    heap.append(_WEdge(W[ind, r], ind, r))
        A.append(d)
    heapify(heap)
    parent = np.arange(n_nodes)
    used = np.ones(n_nodes, dtype=int)
    children = List[Tuple[int, int]]()
    dist = np.empty(n_nodes - n)
    for k in range(n, n_nodes):
        e = heap[0]
        while True:
            e = heappop(heap)
            if used[e.a] != 0 and used[e.b] != 0:
                break
        i = e.a
        j = e.b
        dist[k - n] = e.weight
        parent[i] = k
        parent[j] = k
        children.append((i, j))
        n_i = used[i]
        n_j = used[j]
        used[k] = n_i + n_j
        used[i] = 0
        used[j] = 0
        out = Dict[int, float]()
        for key in sorted(A[i].keys()):
            if used[key] != 0:
                out[key] = A[i][key]
        for key in sorted(A[j].keys()):
            if used[key] != 0:
                v = A[j][key]
                if key in out:
                    if linkage == "complete":
                        out[key] = max(out[key], v)
                    else:
                        out[key] = (n_i * out[key] + n_j * v) / float(n_i + n_j)
                else:
                    out[key] = v
        for col in sorted(out.keys()):
            A[col][k] = out[col]
            heappush(heap, _WEdge(out[col], k, col))
        A.append(out)
        A[i] = Dict[int, float]()
        A[j] = Dict[int, float]()
    ch = np.empty((len(children), 2), dtype=int)
    for q in range(len(children)):
        ch[q, 0] = children[q][1]
        ch[q, 1] = children[q][0]
    return ch, ncc, n, parent, dist

def _hc_get_heads(parents: np.ndarray[int, 1]) -> np.ndarray[int, 1]:
    p = parents.copy()
    size = p.shape[0]
    for node0 in range(size - 1, -1, -1):
        node = node0
        par = p[node]
        while par != node:
            p[node0] = par
            node = par
            par = p[node]
    return p
