# Copyright (C) 2026 the Typhon authors
# Ported from scikit-learn 1.9.1 (BSD-3-Clause)

# sklearn.utils.extmath. The randomized SVD, row_norms (bit-exact with
# numpy's einsum kernel) and fast_logdet live in decomposition/_extmath.py
# and are re-exported here.

import numpy as np
from numpy.random.legacy import RandomState, check_random_state
from ..decomposition._extmath import (squared_norm, row_norms, fast_logdet, svd_flip_u, svd_flip_v,
                                      randomized_range_finder as _rrf, randomized_svd as _rsvd)

def _is_sparse(a) -> Literal[bool]:
    return hasattr(a, "tocsr") and hasattr(a, "indptr")

def safe_sparse_dot(a, b, *, dense_output: bool = False):
    """Dot product that handles the sparse matrix case."""
    if _is_sparse(a) or _is_sparse(b):
        r = a @ b
        if dense_output and hasattr(r, "toarray"):
            return r.toarray()
        return r
    else:
        nda: Literal[int] = static.len(a.shape)
        ndb: Literal[int] = static.len(b.shape)
        if nda > 2 or ndb > 2:
            return np.tensordot(a, b, axes=([-1], [-1 if ndb == 1 else -2]))
        return a @ b

def density(w) -> float:
    if _is_sparse(w):
        return float(w.nnz) / float(w.shape[0] * w.shape[1])
    else:
        a = np.asarray(w)
        return float(np.count_nonzero(a)) / float(a.size)

def randomized_range_finder(A, *, size: int, n_iter: int, power_iteration_normalizer: str = "auto",
                            random_state = None):
    rs = check_random_state(random_state)
    return _rrf(np.asarray(A, dtype=float), size, n_iter, power_iteration_normalizer, rs)

def randomized_svd(M, n_components: int, *, n_oversamples: int = 10, n_iter = "auto",
                   power_iteration_normalizer: str = "auto", transpose = "auto",
                   flip_sign: bool = True, random_state = None, svd_lapack_driver: str = "gesdd"):
    rs = check_random_state(random_state)
    it = -1
    if not isinstance(n_iter, str):
        it = int(n_iter)
    return _rsvd(M, n_components, rs, n_oversamples, it, power_iteration_normalizer, flip_sign)

def svd_flip(u, v, u_based_decision: bool = True):
    """Sign correction so the largest-|.| entry of each singular vector
    (columns of u, or rows of v) is positive. In place; returns (u, v)."""
    if isinstance(v, NoneType):
        e = np.empty((0, 0))
        svd_flip_u(u, e)
        return u, v
    elif isinstance(u, NoneType):
        e = np.empty((0, 0))
        svd_flip_v(e, v)
        return u, v
    else:
        if u_based_decision:
            # argmax over |u.T| rows: first maximal entry wins (np.argmax)
            for r in range(u.shape[1]):
                best = 0
                for i in range(u.shape[0]):
                    if abs(u[i, r]) > abs(u[best, r]):
                        best = i
                s = np.sign(u[best, r])
                for i in range(u.shape[0]):
                    u[i, r] *= s
                if r < v.shape[0]:
                    for c in range(v.shape[1]):
                        v[r, c] *= s
        else:
            for r in range(v.shape[0]):
                best = 0
                for c in range(v.shape[1]):
                    if abs(v[r, c]) > abs(v[r, best]):
                        best = c
                s = np.sign(v[r, best])
                for c in range(v.shape[1]):
                    v[r, c] *= s
                if r < u.shape[1]:
                    for i in range(u.shape[0]):
                        u[i, r] *= s
        return u, v

def softmax(X, copy: bool = True):
    """Row-wise softmax (in place when copy=False)."""
    A = np.array(X, dtype=float) if copy else X
    mx = np.max(A, axis=1).reshape(-1, 1)
    A -= mx
    np.exp(A, out=A)
    s = np.sum(A, axis=1).reshape(-1, 1)
    A /= s
    return A

def _log_logistic_1(x: float) -> float:
    if x > 0:
        return -np.log1p(np.exp(-x))
    else:
        return x - np.log1p(np.exp(x))

def log_logistic(X, out = None):
    """log(1 / (1 + e ** -x)) computed stably (elementwise)."""
    A = np.asarray(X, dtype=float)
    if isinstance(out, NoneType):
        r = np.empty_like(A)
        rf = r.ravel()
        af = A.ravel()
        for i in range(af.shape[0]):
            rf[i] = _log_logistic_1(af[i])
        return rf.reshape(A.shape)
    else:
        of = out.reshape(-1)
        af = A.ravel()
        for i in range(af.shape[0]):
            of[i] = _log_logistic_1(af[i])
        return out

def stable_cumsum(arr, axis = None, rtol: float = 1e-05, atol: float = 1e-08):
    """Cumulative sum in float64; warns (not raised here) when unstable."""
    a = np.asarray(arr, dtype=float)
    if isinstance(axis, NoneType):
        return np.cumsum(a.ravel())
    else:
        return np.cumsum(a, axis=axis)

def weighted_mode(a, w, *, axis = 0):
    """Weighted modal value(s) and their weighted counts along `axis`
    (smallest value wins ties, as in sklearn)."""
    if isinstance(axis, NoneType):
        A = np.asarray(a).ravel()
        W = np.asarray(w, dtype=float).ravel()
        return weighted_mode(A, W, axis=0)
    else:
        A = np.asarray(a)
        W0 = np.asarray(w, dtype=float)
        W = W0 if W0.shape == A.shape else np.full(A.shape, W0.item() if W0.size == 1 else 0.0)
        scores = np.unique(A.ravel())
        counts_shape = list(A.shape)
        counts_shape[axis] = 1
        oldmostfreq = np.zeros(A.shape).sum(axis=axis, keepdims=True)
        oldcounts = np.zeros(A.shape).sum(axis=axis, keepdims=True)
        mostfrequent = oldmostfreq.copy()
        for score in scores:
            template = np.where(A == score, W, 0.0)
            counts = np.sum(template, axis=axis, keepdims=True)
            mostfrequent = np.where(counts > oldcounts, float(score), oldmostfreq)
            oldcounts = np.maximum(counts, oldcounts)
            oldmostfreq = mostfrequent
        return mostfrequent, oldcounts

def cartesian(arrays):
    """Cartesian product of 1-D arrays (list of arrays of one dtype)."""
    arrs = [np.asarray(x) for x in arrays]
    k = len(arrs)
    n = 1
    for x in arrs:
        n *= x.shape[0]
    out = np.empty((n, k), dtype=arrs[0].dtype)
    for r in range(n):
        rem = r
        for c in range(k - 1, -1, -1):
            m = arrs[c].shape[0]
            out[r, c] = arrs[c][rem % m]
            rem //= m
    return out

def make_nonnegative(X, min_value: float = 0.0):
    A = np.asarray(X, dtype=float)
    mn = float(np.min(A))
    if mn < min_value:
        return A + (min_value - mn)
    return A

def safe_sqr(X, *, copy: bool = True):
    if _is_sparse(X):
        Y = X.copy() if copy else X
        Y.data **= 2
        return Y
    else:
        A = np.asarray(X)
        if copy:
            return A ** 2
        A **= 2
        return A

def _deterministic_vector_sign_flip(u):
    for r in range(u.shape[0]):
        best = 0
        for c in range(u.shape[1]):
            if abs(u[r, c]) > abs(u[r, best]):
                best = c
        s = np.sign(u[r, best])
        for c in range(u.shape[1]):
            u[r, c] *= s
    return u

def _nanaverage(a, weights = None) -> float:
    """Average ignoring NaN entries (nan when empty or all-NaN)."""
    A = np.asarray(a, dtype=float)
    if A.shape[0] == 0:
        return np.nan
    if isinstance(weights, NoneType):
        s = 0.0
        c = 0
        for v in A:
            if not np.isnan(v):
                s += v
                c += 1
        return np.nan if c == 0 else float(np.nanmean(A))
    else:
        W = np.asarray(weights, dtype=float)
        mask = ~np.isnan(A)
        if not np.any(mask):
            return np.nan
        a2 = A[mask]
        w2 = W[mask]
        if float(np.sum(w2)) == 0.0:
            return float(np.mean(a2))
        return float(np.average(a2, weights=w2))

def _incremental_mean_and_var(X, last_mean, last_variance, last_sample_count):
    """Chan et al. update of column mean/variance (NaNs ignored)."""
    A = np.asarray(X, dtype=float)
    lm = np.asarray(last_mean, dtype=float)
    lv = np.asarray(last_variance, dtype=float)
    lc = np.asarray(last_sample_count, dtype=float) * np.ones(A.shape[1])
    nanm = np.isnan(A)
    Az = np.where(nanm, 0.0, A)
    last_sum = lm * lc
    new_sum = np.sum(Az, axis=0)
    new_count = A.shape[0] - np.sum(nanm.astype(float), axis=0)
    upd_count = lc + new_count
    upd_mean = (last_sum + new_sum) / upd_count
    T = new_sum / new_count
    temp = np.where(nanm, 0.0, A - T)
    correction = np.sum(temp, axis=0)
    new_unnorm = np.sum(temp ** 2, axis=0)
    new_unnorm -= correction ** 2 / new_count
    last_unnorm = lv * lc
    upd_unnorm = np.empty(A.shape[1])
    for j in range(A.shape[1]):
        if lc[j] == 0:
            upd_unnorm[j] = new_unnorm[j]
        else:
            r = lc[j] / new_count[j]
            upd_unnorm[j] = last_unnorm[j] + new_unnorm[j] + r / upd_count[j] * (last_sum[j] / r - new_sum[j]) ** 2
    return upd_mean, upd_unnorm / upd_count, upd_count
