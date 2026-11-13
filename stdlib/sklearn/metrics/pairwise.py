# Copyright (C) 2026 the Typhon authors
# Ported from scikit-learn 1.9.1 (BSD-3-Clause)

# Pairwise distances and kernels on dense float64 arrays. Operation order
# follows sklearn (euclidean = -2 X.Y^T + |x|^2 + |y|^2, clipped, sqrt), so
# results are bit-identical to the reference for float64 input.

import numpy as np
from ..decomposition._extmath import row_norms
from .._pairwise_k import manhattan_distances as _manhattan_k, additive_chi2_kernel as _chi2_k

def _is_sp(a) -> Literal[bool]:
    return hasattr(a, "tocsr") and hasattr(a, "indptr")

def _dn(X):
    if _is_sp(X):
        return np.ascontiguousarray(np.asarray(X.toarray(), dtype=float))
    else:
        return X

def _f2(X):
    # scipy.sparse CSR/CSC input is densified (results match the sparse kernels)
    if _is_sp(X):
        return np.ascontiguousarray(np.asarray(X.toarray(), dtype=float))
    a = np.asarray(X, dtype=float)
    if static.len(a.shape) == 1:
        compile_error("Expected 2D array, got 1D array instead")
    return np.ascontiguousarray(a)

def check_pairwise_arrays(X, Y = None):
    Xa = _f2(X)
    if isinstance(Y, NoneType):
        return Xa, Xa
    else:
        Ya = _f2(Y)
        if Xa.shape[1] != Ya.shape[1]:
            raise ValueError(f"Incompatible dimension for X and Y matrices: X.shape[1] == {Xa.shape[1]} while Y.shape[1] == {Ya.shape[1]}")
        return Xa, Ya

def check_paired_arrays(X, Y):
    Xa, Ya = check_pairwise_arrays(X, Y)
    if Xa.shape[0] != Ya.shape[0] or Xa.shape[1] != Ya.shape[1]:
        raise ValueError(f"X and Y should be of same shape. They were respectively {Xa.shape} and {Ya.shape} long.")
    return Xa, Ya

def _gamma(gamma, p: int) -> float:
    if isinstance(gamma, NoneType):
        return 1.0 / float(p)
    else:
        return float(gamma)

def _euclid(X, Y, same: bool, squared: bool, XX, YY):
    D = -2 * (X @ Y.T)
    D += XX[:, None]
    D += YY[None, :]
    np.maximum(D, 0.0, out=D)
    if same:
        for i in range(min(D.shape[0], D.shape[1])):
            D[i, i] = 0.0
    if not squared:
        np.sqrt(D, out=D)
    return D

def euclidean_distances(X, Y = None, *, Y_norm_squared = None, squared: bool = False,
                        X_norm_squared = None):
    Xa, Ya = check_pairwise_arrays(X, Y)
    same = isinstance(Y, NoneType)
    if isinstance(X_norm_squared, NoneType):
        XX = row_norms(Xa, squared=True)
    else:
        XX = np.asarray(X_norm_squared, dtype=float).reshape(-1)
        if XX.shape[0] != Xa.shape[0]:
            raise ValueError(f"Incompatible dimensions for X of shape {Xa.shape} and X_norm_squared of shape {np.asarray(X_norm_squared).shape}.")
    if same:
        YY = XX
    elif isinstance(Y_norm_squared, NoneType):
        YY = row_norms(Ya, squared=True)
    else:
        YY = np.asarray(Y_norm_squared, dtype=float).reshape(-1)
        if YY.shape[0] != Ya.shape[0]:
            raise ValueError(f"Incompatible dimensions for Y of shape {Ya.shape} and Y_norm_squared of shape {np.asarray(Y_norm_squared).shape}.")
    return _euclid(Xa, Ya, same, squared, XX, YY)

def nan_euclidean_distances(X, Y = None, *, squared: bool = False, missing_values: float = np.nan,
                            copy: bool = True):
    Xa = np.array(np.asarray(X, dtype=float))
    same = isinstance(Y, NoneType)
    Ya = Xa
    if not isinstance(Y, NoneType):
        Ya = np.array(np.asarray(Y, dtype=float))
    if Xa.shape[1] != Ya.shape[1]:
        raise ValueError(f"Incompatible dimension for X and Y matrices: X.shape[1] == {Xa.shape[1]} while Y.shape[1] == {Ya.shape[1]}")
    use_nan = np.isnan(missing_values)
    mX = np.isnan(Xa) if use_nan else (Xa == missing_values)
    mY = mX if same else (np.isnan(Ya) if use_nan else (Ya == missing_values))
    for v in Xa.reshape(-1):
        if np.isinf(v):
            raise ValueError("Input contains infinity or a value too large for dtype('float64').")
    Xa[mX] = 0.0
    if not same:
        Ya[mY] = 0.0
    mXf = mX.astype(float)
    mYf = mY.astype(float)
    XX = row_norms(Xa, squared=True)
    YY = XX if same else row_norms(Ya, squared=True)
    D = _euclid(Xa, Ya, same, True, XX, YY)
    D -= (Xa * Xa) @ mYf.T
    D -= mXf @ (Ya * Ya).T
    np.clip(D, 0.0, None, out=D)
    if same:
        for i in range(D.shape[0]):
            D[i, i] = 0.0
    pres = (1 - mXf) @ (1 - mYf).T
    for i in range(D.shape[0]):
        for j in range(D.shape[1]):
            if pres[i, j] == 0:
                D[i, j] = np.nan
            else:
                D[i, j] = D[i, j] / pres[i, j] * Xa.shape[1]
            if not squared:
                D[i, j] = np.sqrt(D[i, j])
    return D

def manhattan_distances(X, Y = None):
    Xa, Ya = check_pairwise_arrays(X, Y)
    return _manhattan_k(Xa, Ya)

def haversine_distances(X, Y = None):
    Xa, Ya = check_pairwise_arrays(X, Y)
    if Xa.shape[1] != 2:
        raise ValueError("Haversine distance only valid in 2 dimensions")
    n = Xa.shape[0]; m = Ya.shape[0]
    D = np.empty((n, m))
    for i in range(n):
        for j in range(m):
            s0 = np.sin(0.5 * (Xa[i, 0] - Ya[j, 0]))
            s1 = np.sin(0.5 * (Xa[i, 1] - Ya[j, 1]))
            v = s0 * s0 + np.cos(Xa[i, 0]) * np.cos(Ya[j, 0]) * s1 * s1
            D[i, j] = 2 * np.arcsin(np.sqrt(v))
    return D

def _normalize_rows(X):
    nr = row_norms(X)
    for i in range(nr.shape[0]):
        if nr[i] == 0:
            nr[i] = 1.0
    return X / nr[:, None]

def cosine_similarity(X, Y = None, dense_output: bool = True):
    Xa, Ya = check_pairwise_arrays(X, Y)
    Xn = _normalize_rows(Xa)
    Yn = Xn
    if not isinstance(Y, NoneType):
        Yn = _normalize_rows(Ya)
    return Xn @ Yn.T

def cosine_distances(X, Y = None):
    S = cosine_similarity(X, Y)
    S *= -1
    S += 1
    S = np.clip(S, 0.0, 2.0)
    if isinstance(Y, NoneType):
        for i in range(min(S.shape[0], S.shape[1])):
            S[i, i] = 0.0
    return S

def linear_kernel(X, Y = None, dense_output: bool = True):
    Xa, Ya = check_pairwise_arrays(X, Y)
    return Xa @ Ya.T

def polynomial_kernel(X, Y = None, degree: float = 3, gamma = None, coef0: float = 1):
    Xa, Ya = check_pairwise_arrays(X, Y)
    K = Xa @ Ya.T
    K *= _gamma(gamma, Xa.shape[1])
    K += float(coef0)
    K **= float(degree)
    return K

def sigmoid_kernel(X, Y = None, gamma = None, coef0: float = 1):
    Xa, Ya = check_pairwise_arrays(X, Y)
    K = Xa @ Ya.T
    K *= _gamma(gamma, Xa.shape[1])
    K += float(coef0)
    np.tanh(K, out=K)
    return K

def rbf_kernel(X, Y = None, gamma = None):
    Xa, Ya = check_pairwise_arrays(X, Y)
    K = _rbf_d(Xa, Ya, isinstance(Y, NoneType))
    K *= -_gamma(gamma, Xa.shape[1])
    np.exp(K, out=K)
    return K

def _rbf_d(Xa, Ya, same: bool):
    XX = row_norms(Xa, squared=True)
    YY = XX if same else row_norms(Ya, squared=True)
    return _euclid(Xa, Ya, same, True, XX, YY)

def laplacian_kernel(X, Y = None, gamma = None):
    Xa, Ya = check_pairwise_arrays(X, Y)
    K = -_gamma(gamma, Xa.shape[1]) * _manhattan_k(Xa, Ya)
    np.exp(K, out=K)
    return K

def additive_chi2_kernel(X, Y = None):
    Xa, Ya = check_pairwise_arrays(X, Y)
    if np.any(Xa < 0):
        raise ValueError("X contains negative values.")
    if not isinstance(Y, NoneType):
        if np.any(Ya < 0):
            raise ValueError("Y contains negative values.")
    return _chi2_k(Xa, Ya)

def chi2_kernel(X, Y = None, gamma: float = 1.0):
    K = additive_chi2_kernel(X, Y)
    K *= gamma
    np.exp(K, out=K)
    return K

# ------------------------------------------------------------ paired

def paired_euclidean_distances(X, Y):
    Xa, Ya = check_paired_arrays(X, Y)
    return row_norms(Xa - Ya)

def paired_manhattan_distances(X, Y):
    Xa, Ya = check_paired_arrays(X, Y)
    return np.sum(np.abs(Xa - Ya), axis=1)

def paired_cosine_distances(X, Y):
    Xa, Ya = check_paired_arrays(X, Y)
    return 0.5 * row_norms(_normalize_rows(Xa) - _normalize_rows(Ya), squared=True)

def paired_distances(X, Y, *, metric = "euclidean", **kwds):
    if isinstance(metric, str):
        if metric == "euclidean" or metric == "l2":
            return paired_euclidean_distances(X, Y)
        elif metric == "manhattan" or metric == "l1" or metric == "cityblock":
            return paired_manhattan_distances(X, Y)
        elif metric == "cosine":
            return paired_cosine_distances(X, Y)
        raise ValueError(f"Unknown distance {metric}")
    else:
        Xa, Ya = check_paired_arrays(X, Y)
        out = np.zeros(Xa.shape[0])
        for i in range(Xa.shape[0]):
            out[i] = float(metric(Xa[i], Ya[i]))
        return out

# ------------------------------------------------------------ dispatch

PAIRWISE_DISTANCE_FUNCTIONS = ["cityblock", "cosine", "euclidean", "haversine", "l2", "l1",
                               "manhattan", "precomputed", "nan_euclidean"]
PAIRWISE_KERNEL_FUNCTIONS = ["additive_chi2", "chi2", "linear", "polynomial", "poly", "rbf",
                             "laplacian", "sigmoid", "cosine"]
_SCIPY_METRICS = ["braycurtis", "canberra", "chebyshev", "correlation", "dice", "hamming",
                  "jaccard", "mahalanobis", "minkowski", "rogerstanimoto", "russellrao",
                  "seuclidean", "sokalsneath", "sqeuclidean", "yule", "sokalmichener"]
_BOOL_METRICS = ["dice", "jaccard", "rogerstanimoto", "russellrao", "sokalmichener",
                 "sokalsneath", "yule"]

def distance_metrics():
    return PAIRWISE_DISTANCE_FUNCTIONS

def kernel_metrics():
    return PAIRWISE_KERNEL_FUNCTIONS

def _scipy_pd(Xa, Ya, same: bool, metric: str, p: float, V, VI):
    from scipy.spatial.distance import cdist, pdist, squareform
    A = Xa
    B = Ya
    if metric in _BOOL_METRICS:
        A = (Xa != 0).astype(float)
        B = (Ya != 0).astype(float)
    if same:
        return squareform(pdist(A, metric, p=p, V=V, VI=VI))
    return cdist(A, B, metric, p=p, V=V, VI=VI)

def pairwise_distances(X, Y = None, metric = "euclidean", *, n_jobs = None,
                       force_all_finite = None, ensure_all_finite = None, p: float = 2.0,
                       V = None, VI = None, w = None, squared: bool = False, **kwds):
    """Distance matrix between rows of X (and Y). metric: a sklearn metric
    name, a scipy.spatial.distance metric name, "precomputed" or a
    callable f(x, y) -> float."""
    return _pairwise_distances(_dn(X), _dn(Y), metric, p, V, VI, squared)

def _pairwise_distances(X, Y, metric, p: float, V, VI, squared: bool):
    same = isinstance(Y, NoneType)
    if isinstance(metric, str):
        if metric == "precomputed":
            Xa = _f2(X)
            if np.any(Xa < 0):
                raise ValueError("Negative values in data passed to precomputed distance matrix.")
            return Xa.copy()
        if metric == "nan_euclidean":
            return nan_euclidean_distances(X, Y, squared=squared)
        Xa, Ya = check_pairwise_arrays(X, Y)
        for v in Xa.reshape(-1):
            if not np.isfinite(v):
                raise ValueError("Input contains NaN.")
        if not same:
            for v in Ya.reshape(-1):
                if not np.isfinite(v):
                    raise ValueError("Input contains NaN.")
        if metric == "euclidean" or metric == "l2":
            XX = row_norms(Xa, squared=True)
            YY = XX if same else row_norms(Ya, squared=True)
            return _euclid(Xa, Ya, same, squared, XX, YY)
        elif metric == "manhattan" or metric == "cityblock" or metric == "l1":
            return _manhattan_k(Xa, Ya)
        elif metric == "cosine":
            S = cosine_distances(Xa, Ya)
            if same:
                for i in range(min(S.shape[0], S.shape[1])):
                    S[i, i] = 0.0
            return S
        elif metric == "haversine":
            return haversine_distances(Xa, Ya)
        elif metric in _SCIPY_METRICS:
            return _scipy_pd(Xa, Ya, same, metric, p, V, VI)
        raise ValueError(f"Unknown metric {metric}. Valid metrics are {PAIRWISE_DISTANCE_FUNCTIONS + _SCIPY_METRICS}, or 'precomputed', or a callable")
    else:
        Xa, Ya = check_pairwise_arrays(X, Y)
        n = Xa.shape[0]; m = Ya.shape[0]
        D = np.zeros((n, m))
        if same:
            for i in range(n):
                for j in range(i + 1, n):
                    d = float(metric(Xa[i], Xa[j]))
                    D[i, j] = d
                    D[j, i] = d
            for i in range(n):
                D[i, i] = float(metric(Xa[i], Xa[i]))
        else:
            for i in range(n):
                for j in range(m):
                    D[i, j] = float(metric(Xa[i], Ya[j]))
        return D

def pairwise_kernels(X, Y = None, metric = "linear", *, filter_params: bool = False,
                     n_jobs = None, gamma = None, degree: float = 3, coef0: float = 1, **kwds):
    if isinstance(metric, str):
        if metric == "precomputed":
            return _f2(X).copy()
        Yn = Y
        if metric == "linear":
            return linear_kernel(X, Yn)
        elif metric == "poly" or metric == "polynomial":
            return polynomial_kernel(X, Yn, degree=degree, gamma=gamma, coef0=coef0)
        elif metric == "rbf":
            return rbf_kernel(X, Yn, gamma=gamma)
        elif metric == "laplacian":
            return laplacian_kernel(X, Yn, gamma=gamma)
        elif metric == "sigmoid":
            return sigmoid_kernel(X, Yn, gamma=gamma, coef0=coef0)
        elif metric == "cosine":
            return cosine_similarity(X, Yn)
        elif metric == "chi2":
            return chi2_kernel(X, Yn, gamma=1.0 if isinstance(gamma, NoneType) else float(gamma))
        elif metric == "additive_chi2":
            return additive_chi2_kernel(X, Yn)
        raise ValueError(f"Unknown kernel {metric!r}")
    else:
        Xa, Ya = check_pairwise_arrays(X, Y)
        n = Xa.shape[0]; m = Ya.shape[0]
        K = np.zeros((n, m))
        for i in range(n):
            for j in range(m):
                K[i, j] = float(metric(Xa[i], Ya[j]))
        return K

def pairwise_distances_argmin_min(X, Y, *, axis: int = 1, metric = "euclidean", metric_kwargs = None):
    Xa, Ya = check_pairwise_arrays(X, Y)
    if axis == 0:
        Xa, Ya = Ya, Xa
    D = pairwise_distances(Xa, Ya, metric=metric)
    n = D.shape[0]
    idx = np.empty(n, dtype=int)
    val = np.empty(n)
    for i in range(n):
        b = 0
        for j in range(1, D.shape[1]):
            if D[i, j] < D[i, b]:
                b = j
        idx[i] = b
        val[i] = D[i, b]
    return idx, val

def pairwise_distances_argmin(X, Y, *, axis: int = 1, metric = "euclidean", metric_kwargs = None):
    idx, _ = pairwise_distances_argmin_min(X, Y, axis=axis, metric=metric)
    return idx

def _chunk_n_rows(row_bytes: int, max_n_rows: int, working_memory: float) -> int:
    n = int(working_memory * (2 ** 20) // row_bytes)
    if n > max_n_rows:
        n = max_n_rows
    if n < 1:
        n = 1
    return n

def pairwise_distances_chunked(X, Y = None, *, reduce_func = None, metric = "euclidean",
                               n_jobs = None, working_memory = None, **kwds):
    """List of the distance-matrix row chunks (reduce_func(D_chunk, start)
    applied when given), chunked by working_memory MiB (default 1024)."""
    Xa = _f2(X)
    full = pairwise_distances(Xa, Y, metric=metric)
    nY = full.shape[1]
    wm = 1024.0
    if not isinstance(working_memory, NoneType):
        wm = float(working_memory)
    rows = _chunk_n_rows(8 * nY, Xa.shape[0], wm)
    if isinstance(reduce_func, NoneType):
        out = List[type(full)]()
        s = 0
        while s < Xa.shape[0]:
            e = min(s + rows, Xa.shape[0])
            out.append(full[s:e].copy())
            s = e
        return out
    else:
        s0 = 0
        e0 = min(rows, Xa.shape[0])
        first = reduce_func(full[s0:e0].copy(), s0)
        res = [first]
        s = e0
        while s < Xa.shape[0]:
            e = min(s + rows, Xa.shape[0])
            res.append(reduce_func(full[s:e].copy(), s))
            s = e
        return res
