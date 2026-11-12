# Copyright (C) 2026 the Typhon authors
# Ported from scipy 1.18.1 (BSD-3-Clause)

# Shortest paths (scipy/sparse/csgraph/_shortest_path.pyx): Dijkstra,
# Bellman-Ford, Johnson, Floyd-Warshall and the shortest_path dispatcher.
# Loop / relaxation orders follow the Cython code exactly. Dijkstra's queue
# reproduces std::priority_queue<pair<-dist, node>>: pops the smallest
# distance, ties broken by the larger node index.

import math
import numpy as np
from ._validation import _csr_graph, _dense_graph, _transpose_csr, _index_list, \
    _squeeze, NegativeCycleError, NULL_IDX, _issparse

_INF = math.inf

class _Heap:
    key: List[float]
    node: List[int]

    def __init__(self):
        self.key = []
        self.node = []

    def __len__(self):
        return len(self.key)

    def _less(self, a: int, b: int) -> bool:
        # "a pops before b": max-heap on (-dist, node)
        if self.key[a] != self.key[b]:
            return self.key[a] < self.key[b]
        return self.node[a] > self.node[b]

    def _swap(self, a: int, b: int):
        self.key[a], self.key[b] = self.key[b], self.key[a]
        self.node[a], self.node[b] = self.node[b], self.node[a]

    def push(self, k: float, v: int):
        self.key.append(k)
        self.node.append(v)
        i = len(self.key) - 1
        while i > 0:
            p = (i - 1) // 2
            if self._less(i, p):
                self._swap(i, p)
                i = p
            else:
                break

    def pop(self):
        k = self.key[0]
        v = self.node[0]
        last = len(self.key) - 1
        self._swap(0, last)
        self.key.pop()
        self.node.pop()
        n = len(self.key)
        i = 0
        while True:
            l = 2 * i + 1
            r = l + 1
            s = i
            if l < n and self._less(l, s):
                s = l
            if r < n and self._less(r, s):
                s = r
            if s == i:
                break
            self._swap(i, s)
            i = s
        return k, v

def _scan(h, d: float, v: int, indptr, indices, data, dist, pred, srcs,
          row: int, ret_pred: bool, ret_src: bool, limit: float):
    for e in range(indptr[v], indptr[v + 1]):
        w = indices[e]
        nv = d + data[e]
        if nv <= limit:
            if dist[row, w] > nv:
                dist[row, w] = nv
                h.push(nv, w)
                if ret_pred:
                    pred[row, w] = v
                if ret_src:
                    srcs[row, w] = srcs[row, v]

def _dijkstra_core(G, GT, directed: bool, sources: List[int], limit: float,
                   N: int, dist, pred, srcs, row: int, ret_src: bool,
                   ret_pred: bool = True):
    # dist / pred / srcs: 2-D, row `row` is filled (min_only uses one row)
    h = _Heap()
    for s in sources:
        dist[row, s] = 0.0
        if ret_src:
            srcs[row, s] = s
        h.push(0.0, s)
    while len(h) > 0:
        d, v = h.pop()
        if dist[row, v] < d:
            continue
        _scan(h, d, v, G.indptr, G.indices, G.data, dist, pred, srcs, row,
              ret_pred, ret_src, limit)
        if not directed:
            _scan(h, d, v, GT.indptr, GT.indices, GT.data, dist, pred, srcs, row,
                  ret_pred, ret_src, limit)

def _unit(G):
    H = G.copy()
    for i in range(len(H.data)):
        H.data[i] = 1.0
    return H

def _check_nonneg(G):
    for v in G.data:
        if v < 0:
            return False
    return True

def dijkstra(csgraph, directed: bool = True, indices = None,
             return_predecessors: Literal[bool] = False,
             unweighted: bool = False, limit: float = _INF,
             min_only: Literal[bool] = False):
    G = _csr_graph(csgraph)
    if unweighted:
        G = _unit(G)
    N = G.shape[0]
    GT = _transpose_csr(G)
    idx = _index_list(indices, N)
    if limit < 0:
        raise ValueError("limit must be >= 0")
    if min_only:
        dist = np.full((1, N), _INF)
        pred = np.full((1, N), NULL_IDX, dtype=int)
        srcs = np.full((1, N), NULL_IDX, dtype=int)
        _dijkstra_core(G, GT, directed, idx, limit, N, dist, pred, srcs, 0,
                       return_predecessors, return_predecessors)
        if return_predecessors:
            return dist[0].copy(), pred[0].copy(), srcs[0].copy()
        else:
            return dist[0].copy()
    else:
        D = np.full((len(idx), N), _INF)
        P = np.full((len(idx), N), NULL_IDX, dtype=int)
        dummy = np.zeros((1, 1), dtype=int)
        for r in range(len(idx)):
            _dijkstra_core(G, GT, directed, [idx[r]], limit, N, D, P, dummy, r, False)
        if return_predecessors:
            return _squeeze(indices, D), _squeeze(indices, P)
        else:
            return _squeeze(indices, D)

_DTYPE_EPS = 1e-6

def _bf_run(G, directed: bool, N: int, i: int, src: int, D, P) -> bool:
    # returns True when a negative cycle is detected
    D[i, src] = 0.0
    for count in range(N - 1):
        for j in range(N):
            d1 = D[i, j]
            for k in range(G.indptr[j], G.indptr[j + 1]):
                w12 = G.data[k]
                ik = G.indices[k]
                d2 = D[i, ik]
                if directed:
                    if d1 + w12 < d2:
                        D[i, ik] = d1 + w12
                        P[i, ik] = j
                else:
                    if d1 + w12 < d2:
                        d2 = d1 + w12
                        D[i, ik] = d2
                        P[i, ik] = j
                    if d2 + w12 < d1:
                        d1 = d2 + w12
                        D[i, j] = d1
                        P[i, j] = ik
    for j in range(N):
        d1 = D[i, j]
        for k in range(G.indptr[j], G.indptr[j + 1]):
            w12 = G.data[k]
            d2 = D[i, G.indices[k]]
            if directed:
                if d1 + w12 + _DTYPE_EPS < d2:
                    return True
            else:
                if abs(d2 - d1) > w12 + _DTYPE_EPS:
                    return True
    return False

def bellman_ford(csgraph, directed: bool = True, indices = None,
                 return_predecessors: Literal[bool] = False,
                 unweighted: bool = False):
    G = _csr_graph(csgraph)
    if unweighted:
        G = _unit(G)
    N = G.shape[0]
    idx = _index_list(indices, N)
    D = np.full((len(idx), N), _INF)
    P = np.full((len(idx), N), NULL_IDX, dtype=int)
    for r in range(len(idx)):
        if _bf_run(G, directed, N, r, idx[r], D, P):
            raise NegativeCycleError(f"Negative cycle detected on node {idx[r]}")
    if return_predecessors:
        return _squeeze(indices, D), _squeeze(indices, P)
    else:
        return _squeeze(indices, D)

def johnson(csgraph, directed: bool = True, indices = None,
            return_predecessors: Literal[bool] = False,
            unweighted: bool = False):
    if unweighted:
        return dijkstra(csgraph, directed, indices, return_predecessors, True)
    G = _csr_graph(csgraph)
    N = G.shape[0]
    idx = _index_list(indices, N)
    pot = np.zeros(N)
    neg = -1
    for count in range(N):
        for j in range(N):
            d1 = pot[j]
            for k in range(G.indptr[j], G.indptr[j + 1]):
                w12 = G.data[k]
                ik = G.indices[k]
                d2 = pot[ik]
                if d1 + w12 < d2:
                    pot[ik] = d1 + w12
                if not directed:
                    if d2 + w12 < d1:
                        d1 = d2 + w12
                        pot[j] = d1
    for j in range(N):
        d1 = pot[j]
        for k in range(G.indptr[j], G.indptr[j + 1]):
            w12 = G.data[k]
            d2 = pot[G.indices[k]]
            if directed:
                if d1 + w12 + _DTYPE_EPS < d2 and neg < 0:
                    neg = j
            else:
                if abs(d2 - d1) > w12 + _DTYPE_EPS and neg < 0:
                    neg = j
    if neg >= 0:
        raise NegativeCycleError(f"Negative cycle detected on node {neg}")
    for j in range(N):
        for k in range(G.indptr[j], G.indptr[j + 1]):
            G.data[k] += pot[j]
            G.data[k] -= pot[G.indices[k]]
    GT = _transpose_csr(G)
    if not directed:
        for j in range(N):
            for k in range(GT.indptr[j], GT.indptr[j + 1]):
                GT.data[k] += pot[j]
                GT.data[k] -= pot[GT.indices[k]]
    D = np.full((len(idx), N), _INF)
    P = np.full((len(idx), N), NULL_IDX, dtype=int)
    dummy = np.zeros((1, 1), dtype=int)
    for r in range(len(idx)):
        _dijkstra_core(G, GT, directed, [idx[r]], _INF, N, D, P, dummy, r, False)
    for r in range(len(idx)):
        for j in range(N):
            D[r, j] += pot[j]
            D[r, j] -= pot[idx[r]]
    if return_predecessors:
        return _squeeze(indices, D), _squeeze(indices, P)
    else:
        return _squeeze(indices, D)

def floyd_warshall(csgraph, directed: bool = True,
                   return_predecessors: Literal[bool] = False,
                   unweighted: bool = False, overwrite: bool = False):
    D = _dense_graph(csgraph)
    N = D.shape[0]
    if unweighted:
        for i in range(N):
            for j in range(N):
                if not math.isinf(D[i, j]):
                    D[i, j] = 1.0
    for i in range(N):
        D[i, i] = 0.0
    if not directed:
        for i in range(N):
            for j in range(i + 1, N):
                if D[j, i] <= D[i, j]:
                    D[i, j] = D[j, i]
                else:
                    D[j, i] = D[i, j]
    P = np.full((N, N), NULL_IDX, dtype=int)
    for i in range(N):
        for j in range(N):
            if i != j and not math.isinf(D[i, j]):
                P[i, j] = i
    for k in range(N):
        for i in range(N):
            if D[i, k] == _INF:
                continue
            for j in range(N):
                d = D[i, k] + D[k, j]
                if d < D[i, j]:
                    D[i, j] = d
                    P[i, j] = P[k, j]
    for i in range(N):
        if D[i, i] < 0:
            raise NegativeCycleError("Negative cycle in nodes")
    if return_predecessors:
        return D, P
    else:
        return D

def _auto_method(csgraph, indices) -> str:
    neg = False
    Nk = 0
    if _issparse(csgraph):
        c = csgraph.tocoo()
        Nk = c.nnz
        for v in c.data:
            if v < 0:
                neg = True
        N = c.shape[0]
    else:
        G = np.asarray(csgraph).astype(float)
        N = G.shape[0]
        for v in G.flat:
            if v != 0 and not math.isinf(v) and not math.isnan(v):
                Nk += 1
                if v < 0:
                    neg = True
    if indices is not None or Nk < N * N / 4:
        return "J" if neg else "D"
    else:
        return "FW"

def shortest_path(csgraph, method: str = "auto", directed: bool = True,
                  return_predecessors: Literal[bool] = False,
                  unweighted: bool = False, overwrite: bool = False,
                  indices = None):
    m = _auto_method(csgraph, indices) if method == "auto" else method
    if m == "FW":
        if indices is not None:
            raise ValueError("Cannot specify indices with method == 'FW'.")
        return floyd_warshall(csgraph, directed, return_predecessors, unweighted)
    elif m == "D":
        return dijkstra(csgraph, directed, indices, return_predecessors, unweighted)
    elif m == "BF":
        return bellman_ford(csgraph, directed, indices, return_predecessors, unweighted)
    elif m == "J":
        return johnson(csgraph, directed, indices, return_predecessors, unweighted)
    else:
        raise ValueError(f"unrecognized method '{m}'")
        return dijkstra(csgraph, directed, indices, return_predecessors, unweighted)

def construct_dist_matrix(graph, predecessors, directed: bool = True,
                          null_value: float = _INF):
    G = _dense_graph(graph)
    P = np.asarray(predecessors)
    N = G.shape[0]
    if P.shape[0] != N or P.shape[1] != N:
        raise ValueError("graph and predecessors must have the same shape")
    if not directed:
        for i in range(N):
            for j in range(i + 1, N):
                if G[j, i] <= G[i, j]:
                    G[i, j] = G[j, i]
                else:
                    G[j, i] = G[i, j]
    D = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            null_path = True
            k2 = j
            while k2 != i:
                k1 = P[i, k2]
                if k1 == NULL_IDX:
                    break
                D[i, j] += G[k1, k2]
                null_path = False
                k2 = k1
            if null_path and i != j:
                D[i, j] = null_value
    return D

# ----------------------------------------------------------------- yen

def _johnson_potentials(data, G, directed: bool, N: int):
    # Bellman-Ford on a virtual source (scipy _johnson_directed/_undirected)
    pot = np.zeros(N)
    for count in range(N):
        for j in range(N):
            d1 = pot[j]
            for k in range(G.indptr[j], G.indptr[j + 1]):
                w12 = data[k]
                ik = G.indices[k]
                d2 = pot[ik]
                if d1 + w12 < d2:
                    pot[ik] = d1 + w12
                if not directed:
                    if d2 + w12 < d1:
                        d1 = d2 + w12
                        pot[j] = d1
    for j in range(N):
        d1 = pot[j]
        for k in range(G.indptr[j], G.indptr[j + 1]):
            w12 = data[k]
            d2 = pot[G.indices[k]]
            if directed:
                if d1 + w12 + _DTYPE_EPS < d2:
                    raise NegativeCycleError(f"Negative cycle detected on node {j}")
            else:
                if abs(d2 - d1) > w12 + _DTYPE_EPS:
                    raise NegativeCycleError(f"Negative cycle detected on node {j}")
    return pot

class _YenCand:
    dist: List[float]
    path: List[List[int]]
    spur: List[int]
    required: int

    def __init__(self, K: int):
        self.dist = List[float]()
        self.path = List[List[int]]()
        self.spur = List[int]()
        self.required = K

    def insert(self, distance: float, src_pred, spur_pred, spur_node: int, sink: int):
        if len(self.dist) >= self.required and distance >= self.dist[len(self.dist) - 1]:
            return
        p = List[int]()
        idx = sink
        while idx != spur_node:
            p.append(idx)
            idx = spur_pred[idx]
        while idx != NULL_IDX:
            p.append(idx)
            idx = src_pred[idx]
        # std::lower_bound on distance
        pos = 0
        while pos < len(self.dist) and self.dist[pos] < distance:
            pos += 1
        self.dist.insert(pos, distance)
        self.path.insert(pos, p)
        self.spur.insert(pos, spur_node)
        while len(self.dist) > self.required:
            self.dist.pop()
            self.path.pop()
            self.spur.pop()

def _yen_dijkstra(src: int, source: int, w, G, wT, GT, directed: bool, N: int):
    D = np.full((1, N), _INF)
    P = np.full((1, N), NULL_IDX, dtype=int)
    dummy = np.zeros((1, 1), dtype=int)
    D[0, source] = 0.0
    Gw = G.copy()
    Gw.data = w
    GTw = GT.copy()
    if not directed:
        GTw.data = wT
    _dijkstra_core(Gw, GTw, directed, [src], _INF, N, D, P, dummy, 0, False)
    return D, P

def yen(csgraph, source: int, sink: int, K: int, directed: bool = True,
        return_predecessors: Literal[bool] = False, unweighted: bool = False):
    G = _csr_graph(csgraph)
    N = G.shape[0]
    if source < 0 or source >= N or sink < 0 or sink >= N:
        raise ValueError(f"For csgraph with shape (N, N), must have 0 <= source < N and 0 <= sink < N. Got N={N}, source={source}, sink={sink}.")
    dist_array = np.full(K, _INF)
    preds = np.full((K, N), NULL_IDX, dtype=int)
    neg = False
    if unweighted:
        data = np.ones(len(G.data))
    else:
        data = G.data.copy()
        for v in data:
            if v < 0:
                neg = True
    pot = np.zeros(N)
    if neg:
        pot = _johnson_potentials(data, G, directed, N)
        for j in range(N):
            for k in range(G.indptr[j], G.indptr[j + 1]):
                data[k] += pot[j]
                data[k] -= pot[G.indices[k]]
    GT = _transpose_csr(G)
    if directed:
        dataT = np.empty(0)
    else:
        if unweighted:
            dataT = data
        else:
            dataT = GT.data.copy()
            if neg:
                for j in range(N):
                    for k in range(GT.indptr[j], GT.indptr[j + 1]):
                        dataT[k] += pot[j]
                        dataT[k] -= pot[GT.indices[k]]
    # --- _yen ---
    D, P = _yen_dijkstra(source, source, data, G, dataT, GT, directed, N)
    dist_array[0] = D[0, sink]
    if dist_array[0] != _INF:
        w = data.copy()
        wT = dataT.copy()
        node = sink
        while node != NULL_IDX:
            preds[0, node] = P[0, node]
            node = P[0, node]
        cand = _YenCand(K)
        spur_km1 = source
        for k in range(1, K):
            spur_node = sink
            root_dist = dist_array[k - 1]
            while spur_node != spur_km1:
                tmp_i = preds[k - 1, spur_node]
                tmp_d = _INF
                for i in range(G.indptr[tmp_i], G.indptr[tmp_i + 1]):
                    if G.indices[i] == spur_node:
                        tmp_d = w[i]
                        break
                if not directed:
                    for i in range(GT.indptr[tmp_i], GT.indptr[tmp_i + 1]):
                        if GT.indices[i] == spur_node and wT[i] < tmp_d:
                            tmp_d = wT[i]
                            break
                if tmp_d == _INF:
                    raise ValueError(f"No edge between nodes {tmp_i} and {spur_node}")
                root_dist -= tmp_d
                spur_node = tmp_i
                for sp_i in range(k):
                    node = spur_node
                    while preds[sp_i, node] == preds[k - 1, node]:
                        if node == source:
                            for i in range(G.indptr[spur_node], G.indptr[spur_node + 1]):
                                if spur_node == preds[sp_i, G.indices[i]]:
                                    w[i] = _INF
                            if not directed:
                                for i in range(GT.indptr[spur_node], GT.indptr[spur_node + 1]):
                                    if spur_node == preds[sp_i, GT.indices[i]]:
                                        wT[i] = _INF
                            break
                        node = preds[sp_i, node]
                node = preds[k - 1, spur_node]
                while node != NULL_IDX:
                    for i in range(G.indptr[node], G.indptr[node + 1]):
                        w[i] = _INF
                    if not directed:
                        for i in range(GT.indptr[node], GT.indptr[node + 1]):
                            wT[i] = _INF
                    node = preds[k - 1, node]
                D2, P2 = _yen_dijkstra(spur_node, source, w, G, wT, GT, directed, N)
                total = D2[0, sink] + root_dist
                if total != _INF:
                    cand.insert(total, preds[k - 1], P2[0], spur_node, sink)
                node = spur_node
                while node != NULL_IDX:
                    for i in range(G.indptr[node], G.indptr[node + 1]):
                        w[i] = data[i]
                    if not directed:
                        for i in range(GT.indptr[node], GT.indptr[node + 1]):
                            wT[i] = dataT[i]
                    node = preds[k - 1, node]
            if len(cand.dist) == 0:
                break
            dist_array[k] = cand.dist[0]
            p = cand.path[0]
            spur_km1 = cand.spur[0]
            cand.dist.pop(0)
            cand.path.pop(0)
            cand.spur.pop(0)
            for q in range(len(p) - 1):
                preds[k, p[q]] = p[q + 1]
            cand.required -= 1
    if neg:
        for k in range(K):
            dist_array[k] += pot[sink] - pot[source]
    nf = 0
    for k in range(K):
        if dist_array[k] < _INF:
            nf += 1
    if return_predecessors:
        return dist_array[:nf].copy(), np.ascontiguousarray(preds[:nf])
    else:
        return dist_array[:nf].copy()
