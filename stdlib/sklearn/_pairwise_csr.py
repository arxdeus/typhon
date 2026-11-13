# Copyright (C) 2026 the Typhon authors
# Ported from scikit-learn 1.9.1 (BSD-3-Clause)

# Native CSR kernels for pairwise distances (sklearn.metrics.pairwise with
# scipy.sparse input; sklearn/metrics/_pairwise_distances_reduction and
# sklearn/metrics/pairwise.py::_sparse_manhattan). Rows are merged over their
# sorted column indices, so no densification happens.

import math
import numpy as np

def _csr_parts(X):
    # (data, indices, indptr, nrows, ncols) of a CSR matrix with sorted unique indices
    C = X.tocsr()
    C.sum_duplicates()
    C.sort_indices()
    return C.data, C.indices, C.indptr, C.shape[0], C.shape[1]

def csr_row_sq_norms(X) -> np.ndarray[float, 1]:
    d, ind, ptr, n, m = _csr_parts(X)
    out = np.zeros(n)
    for i in range(n):
        s = 0.0
        for k in range(ptr[i], ptr[i + 1]):
            s += d[k] * d[k]
        out[i] = s
    return out

def _row_dot(d1, i1, p1, a: int, d2, i2, p2, b: int) -> float:
    ka = p1[a]
    ea = p1[a + 1]
    kb = p2[b]
    eb = p2[b + 1]
    s = 0.0
    while ka < ea and kb < eb:
        ca = i1[ka]
        cb = i2[kb]
        if ca == cb:
            s += d1[ka] * d2[kb]
            ka += 1
            kb += 1
        elif ca < cb:
            ka += 1
        else:
            kb += 1
    return s

def csr_dot_T(X, Y) -> np.ndarray[float, 2]:
    """X @ Y.T for CSR inputs (dense result)."""
    d1, i1, p1, n, m1 = _csr_parts(X)
    d2, i2, p2, q, m2 = _csr_parts(Y)
    if m1 != m2:
        raise ValueError(f"Incompatible dimension for X and Y matrices: X.shape[1] == {m1} while Y.shape[1] == {m2}")
    out = np.empty((n, q))
    for i in range(n):
        for j in range(q):
            out[i, j] = _row_dot(d1, i1, p1, i, d2, i2, p2, j)
    return out

def csr_euclidean(X, Y, same: bool, squared: bool) -> np.ndarray[float, 2]:
    """Euclidean distances via -2 X.Y^T + |x|^2 + |y|^2 (clipped), like sklearn."""
    XX = csr_row_sq_norms(X)
    YY = XX if same else csr_row_sq_norms(Y)
    D = csr_dot_T(X, Y)
    n, q = D.shape
    for i in range(n):
        for j in range(q):
            v = -2 * D[i, j]
            v += XX[i]
            v += YY[j]
            if v < 0.0:
                v = 0.0
            D[i, j] = v
    if same:
        for i in range(min(n, q)):
            D[i, i] = 0.0
    if not squared:
        for i in range(n):
            for j in range(q):
                D[i, j] = math.sqrt(D[i, j])
    return D

def csr_manhattan(X, Y) -> np.ndarray[float, 2]:
    """L1 distances between CSR rows (merge over sorted indices)."""
    d1, i1, p1, n, m1 = _csr_parts(X)
    d2, i2, p2, q, m2 = _csr_parts(Y)
    if m1 != m2:
        raise ValueError(f"Incompatible dimension for X and Y matrices: X.shape[1] == {m1} while Y.shape[1] == {m2}")
    out = np.empty((n, q))
    for i in range(n):
        for j in range(q):
            ka = p1[i]
            ea = p1[i + 1]
            kb = p2[j]
            eb = p2[j + 1]
            s = 0.0
            while ka < ea and kb < eb:
                ca = i1[ka]
                cb = i2[kb]
                if ca == cb:
                    s += abs(d1[ka] - d2[kb])
                    ka += 1
                    kb += 1
                elif ca < cb:
                    s += abs(d1[ka])
                    ka += 1
                else:
                    s += abs(d2[kb])
                    kb += 1
            while ka < ea:
                s += abs(d1[ka])
                ka += 1
            while kb < eb:
                s += abs(d2[kb])
                kb += 1
            out[i, j] = s
    return out

def csr_cosine_distances(X, Y, same: bool) -> np.ndarray[float, 2]:
    """1 - cosine similarity (rows normalised by their L2 norm), clipped to [0, 2]."""
    XX = csr_row_sq_norms(X)
    YY = XX if same else csr_row_sq_norms(Y)
    D = csr_dot_T(X, Y)
    n, q = D.shape
    for i in range(n):
        nx = math.sqrt(XX[i])
        for j in range(q):
            ny = math.sqrt(YY[j])
            sx = 1.0 if nx == 0.0 else nx
            sy = 1.0 if ny == 0.0 else ny
            v = 1.0 - D[i, j] / sx / sy
            if v < 0.0:
                v = 0.0
            elif v > 2.0:
                v = 2.0
            D[i, j] = v
    if same:
        for i in range(min(n, q)):
            D[i, i] = 0.0
    return D
