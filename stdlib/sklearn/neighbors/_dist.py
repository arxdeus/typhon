# Ported from scikit-learn 1.9.1 (BSD-3-Clause)

# Exact brute-force distance backend shared by every neighbors estimator.
# algorithm='kd_tree' / 'ball_tree' give identical results to 'brute' in
# scikit-learn, so all of them are served by this exact backend.

import numpy as np
from ._tree_util import heap_push, simultaneous_sort_range

def _metric_code(metric: str, p: float) -> int:
    # 0 euclidean, 1 manhattan, 2 chebyshev, 3 general minkowski, 4 cosine,
    # 5 sqeuclidean
    if metric == "euclidean" or metric == "l2":
        return 0
    if metric == "manhattan" or metric == "cityblock" or metric == "l1":
        return 1
    if metric == "chebyshev" or metric == "infinity":
        return 2
    if metric == "minkowski":
        if p == 2.0:
            return 0
        if p == 1.0:
            return 1
        if p == float("inf"):
            return 2
        if p <= 0.0:
            raise ValueError("p must be greater than 0")
        return 3
    if metric == "cosine":
        return 4
    if metric == "sqeuclidean":
        return 5
    # tree-only metrics (haversine, hamming, ...) have no brute kernel here
    return -1

def _pairwise(Xq: np.ndarray[float, 2], Xf: np.ndarray[float, 2], code: int,
              p: float) -> np.ndarray[float, 2]:
    nq = Xq.shape[0]
    nf = Xf.shape[0]
    d = Xq.shape[1]
    D = np.empty((nq, nf))
    nq_norm = np.zeros(nq)
    nf_norm = np.zeros(nf)
    if code == 4:
        for i in range(nq):
            s = 0.0
            for k in range(d):
                s += Xq[i, k] * Xq[i, k]
            nq_norm[i] = np.sqrt(s)
        for j in range(nf):
            s = 0.0
            for k in range(d):
                s += Xf[j, k] * Xf[j, k]
            nf_norm[j] = np.sqrt(s)
    for i in range(nq):
        for j in range(nf):
            r = 0.0
            if code == 0 or code == 5:
                for k in range(d):
                    t = Xq[i, k] - Xf[j, k]
                    r += t * t
                if code == 0:
                    r = np.sqrt(r)
            elif code == 1:
                for k in range(d):
                    r += abs(Xq[i, k] - Xf[j, k])
            elif code == 2:
                for k in range(d):
                    t = abs(Xq[i, k] - Xf[j, k])
                    if t > r:
                        r = t
            elif code == 3:
                for k in range(d):
                    r += abs(Xq[i, k] - Xf[j, k]) ** p
                r = r ** (1.0 / p)
            else:
                dot = 0.0
                for k in range(d):
                    dot += Xq[i, k] * Xf[j, k]
                den = nq_norm[i] * nf_norm[j]
                r = 1.0 - (dot / den if den != 0.0 else 0.0)
                if r < 0.0:
                    r = 0.0
            D[i, j] = r
    return D

def _kneighbors(D: np.ndarray[float, 2], k: int, skip_self: bool):
    """Indices of the k smallest distances per row (ties: lower index first).
    skip_self drops the query point itself (row i, column i)."""
    nq = D.shape[0]
    nf = D.shape[1]
    dist = np.empty((nq, k))
    ind = np.empty((nq, k), dtype=int)
    for i in range(nq):
        row = D[i]
        order = np.argsort(row, kind="stable")
        c = 0
        for t in range(nf):
            j = int(order[t])
            if skip_self and j == i:
                continue
            if c < k:
                dist[i, c] = row[j]
                ind[i, c] = j
                c += 1
    return dist, ind

def _as2d(X) -> np.ndarray[float, 2]:
    a = np.asarray(X, dtype=float)
    if a.ndim == 1:
        return np.ascontiguousarray(a.reshape((1, -1)))
    return np.ascontiguousarray(a)


def _kneighbors_heap(Xq: np.ndarray[float, 2], Xf: np.ndarray[float, 2], D: np.ndarray[float, 2],
                     code: int, k: int):
    """ArgKmin (pairwise_distances_reduction) selection: a size-k max-heap fed in
    index order with the surrogate key (euclidean: -2 x.y + |y|^2), then the
    introsort of simultaneous_sort.  Returns (distances, indices)."""
    nq = Xq.shape[0]
    nf = Xf.shape[0]
    d = Xq.shape[1]
    HD = np.full((nq, k), float("inf"))
    HI = np.zeros((nq, k), dtype=int)
    ynorm = np.zeros(nf)
    if code == 0 or code == 5:
        for j in range(nf):
            s = 0.0
            for c in range(d):
                s += Xf[j, c] * Xf[j, c]
            ynorm[j] = s
    chunk = 256
    nchunks = (nf + chunk - 1) // chunk
    # parallel_on_Y (used while nq < 256 * n_threads): one heap per Y chunk,
    # merged into the result heap in chunk order, entries in heap-array order
    LD = np.empty(k)
    LI = np.zeros(k, dtype=int)
    L2D = np.empty((1, k))
    L2I = np.zeros((1, k), dtype=int)
    G = Xq @ Xf.T if (code == 0 or code == 5) else np.empty((0, 0))
    for i in range(nq):
        for ch in range(nchunks):
            for c in range(k):
                L2D[0, c] = float("inf")
                L2I[0, c] = 0
            lo = ch * chunk
            hi = min(nf, lo + chunk)
            for j in range(lo, hi):
                key = D[i, j]
                if code == 0 or code == 5:
                    key = -2.0 * G[i, j] + ynorm[j]
                heap_push(L2D, L2I, 0, k, key, j)
            for c in range(k):
                heap_push(HD, HI, i, k, L2D[0, c], L2I[0, c])
        rv = np.ascontiguousarray(HD[i])
        ri = np.ascontiguousarray(HI[i])
        simultaneous_sort_range(rv, ri, 0, k)
        for c in range(k):
            HI[i, c] = ri[c]
            HD[i, c] = D[i, ri[c]]
    return HD, HI
