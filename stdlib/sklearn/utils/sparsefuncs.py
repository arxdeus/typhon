# Copyright (C) 2026 the Typhon authors
# Ported from scikit-learn 1.9.1 (BSD-3-Clause)

# sklearn.utils.sparsefuncs on stdlib scipy.sparse CSR/CSC matrices
# (sparsefuncs_fast loops written out).

import numpy as np

def _check_fmt(X):
    if not (hasattr(X, "indptr") and hasattr(X, "format")):
        compile_error("Expected a CSR or CSC sparse matrix")

def _mv_major(X, n_major: int, n_minor: int, weights):
    """Mean/variance along the compressed (row-for-CSR) direction, i.e.
    per minor index (column of CSR) over the major axis; port of
    _csr_mean_variance_axis0 (NaNs ignored)."""
    w = np.ones(n_major) if isinstance(weights, NoneType) else np.asarray(weights, dtype=float)
    total_w = float(np.sum(w))
    means = np.zeros(n_minor)
    variances = np.zeros(n_minor)
    sum_w = np.full(n_minor, total_w)
    sum_w_nz = np.zeros(n_minor)
    correction = np.zeros(n_minor)
    counts = np.full(n_minor, n_major)
    counts_nz = np.zeros(n_minor, dtype=int)
    for r in range(n_major):
        for i in range(X.indptr[r], X.indptr[r + 1]):
            c = X.indices[i]
            v = float(X.data[i])
            if not np.isnan(v):
                means[c] += v * w[r]
                sum_w_nz[c] += w[r]
                counts_nz[c] += 1
            else:
                sum_w[c] -= w[r]
                counts[c] -= 1
    for c in range(n_minor):
        means[c] /= sum_w[c]
    for r in range(n_major):
        for i in range(X.indptr[r], X.indptr[r + 1]):
            c = X.indices[i]
            v = float(X.data[i])
            if not np.isnan(v):
                d = v - means[c]
                correction[c] += d * w[r]
                variances[c] += d * d * w[r]
    for c in range(n_minor):
        if counts[c] != counts_nz[c]:
            correction[c] -= (sum_w[c] - sum_w_nz[c]) * means[c]
        correction[c] = correction[c] ** 2 / sum_w[c]
        if counts[c] != counts_nz[c]:
            variances[c] += (sum_w[c] - sum_w_nz[c]) * means[c] ** 2
        variances[c] = (variances[c] - correction[c]) / sum_w[c]
    return means, variances, sum_w

def _mv_minor(X, n_major: int, n_minor: int, weights):
    """Mean/variance per major index (row of CSR), over minor entries."""
    w = np.ones(n_minor) if isinstance(weights, NoneType) else np.asarray(weights, dtype=float)
    means = np.zeros(n_major)
    variances = np.zeros(n_major)
    sum_w = np.zeros(n_major)
    for r in range(n_major):
        tw = float(np.sum(w))
        s = 0.0
        for i in range(X.indptr[r], X.indptr[r + 1]):
            v = float(X.data[i])
            if np.isnan(v):
                tw -= w[X.indices[i]]
            else:
                s += v * w[X.indices[i]]
        m = s / tw
        var = 0.0
        nzw = 0.0
        for i in range(X.indptr[r], X.indptr[r + 1]):
            v = float(X.data[i])
            if not np.isnan(v):
                var += w[X.indices[i]] * (v - m) ** 2
                nzw += w[X.indices[i]]
        var += (tw - nzw) * m * m
        means[r] = m
        variances[r] = var / tw
        sum_w[r] = tw
    return means, variances, sum_w

def mean_variance_axis(X, axis: int, weights = None, return_sum_weights: Literal[bool] = False):
    """Column (axis=0) or row (axis=1) means and variances of a CSR/CSC
    matrix (implicit zeros included, explicit NaNs ignored)."""
    _check_fmt(X)
    if axis != 0 and axis != 1:
        raise ValueError(f"Unknown axis value: {axis}. Use 0 for rows, or 1 for columns")
    n, m = X.shape
    is_csr = X.format == "csr"
    if not is_csr and X.format != "csc":
        raise TypeError("Expected a CSR or CSC sparse matrix, got " + X.format + ".")
    if (axis == 0) == is_csr:
        r = _mv_major(X, n if is_csr else m, m if is_csr else n, weights)
    else:
        r = _mv_minor(X, n if is_csr else m, m if is_csr else n, weights)
    if return_sum_weights:
        return r
    else:
        return r[0], r[1]

def inplace_csr_column_scale(X, scale):
    s = np.asarray(scale, dtype=float)
    for i in range(X.indices.shape[0]):
        X.data[i] *= s[X.indices[i]]

def inplace_csr_row_scale(X, scale):
    s = np.asarray(scale, dtype=float)
    for r in range(X.shape[0]):
        for i in range(X.indptr[r], X.indptr[r + 1]):
            X.data[i] *= s[r]

def inplace_column_scale(X, scale):
    _check_fmt(X)
    if X.format == "csr":
        inplace_csr_column_scale(X, scale)
    else:
        inplace_csr_row_scale_csc(X, scale)

def inplace_csr_row_scale_csc(X, scale):
    s = np.asarray(scale, dtype=float)
    for c in range(X.shape[1]):
        for i in range(X.indptr[c], X.indptr[c + 1]):
            X.data[i] *= s[c]

def inplace_row_scale(X, scale):
    _check_fmt(X)
    if X.format == "csr":
        inplace_csr_row_scale(X, scale)
    else:
        s = np.asarray(scale, dtype=float)
        for i in range(X.indices.shape[0]):
            X.data[i] *= s[X.indices[i]]

def min_max_axis(X, axis: int, ignore_nan: bool = False):
    """Min and max along an axis of a sparse matrix (implicit zeros count)."""
    D = X.toarray()
    if ignore_nan:
        return np.nanmin(D, axis=axis), np.nanmax(D, axis=axis)
    return np.min(D, axis=axis), np.max(D, axis=axis)

def count_nonzero(X, axis = None, sample_weight = None):
    D = X.toarray()
    nz = (D != 0).astype(float)
    if isinstance(sample_weight, NoneType):
        if isinstance(axis, NoneType):
            return int(np.sum(nz))
        else:
            return np.sum(nz, axis=axis).astype(int)
    else:
        w = np.asarray(sample_weight, dtype=float)
        if isinstance(axis, NoneType):
            return float(np.sum(nz * w[:, None]))
        elif axis == 1:
            return np.sum(nz, axis=1) * w
        else:
            return w @ nz

def csc_median_axis_0(X):
    D = X.toarray()
    return np.median(D, axis=0)

def incr_mean_variance_axis(X, *, axis: int, last_mean, last_var, last_n, weights = None):
    """Incremental mean and variance of a CSR/CSC matrix (implicit zeros
    counted). last_n: array or scalar sample count."""
    _check_fmt(X)
    if axis != 0 and axis != 1:
        raise ValueError(f"Unknown axis value: {axis}. Use 0 for rows, or 1 for columns")
    lm = np.asarray(last_mean, dtype=float).copy()
    lv = np.asarray(last_var, dtype=float).copy()
    k = lm.shape[0]
    ln = np.full(k, 0.0)
    if static.len(np.asarray(last_n).shape) == 0:
        ln = np.full(k, float(last_n))
    else:
        ln = np.asarray(last_n, dtype=float).copy()
    if lv.shape[0] != k or ln.shape[0] != k:
        raise ValueError("last_mean, last_var, last_n do not have the same shapes.")
    n, m = X.shape
    if (axis == 1 and k != n) or (axis == 0 and k != m):
        raise ValueError(f"If axis={axis}, then last_mean, last_n, last_var should be of size {'n_samples ' + str(n) if axis == 1 else 'n_features ' + str(m)} (Got {k}).")
    r = mean_variance_axis(X, axis, weights, True)
    nm = r[0]
    nv = r[1]
    nn = r[2]
    first = True
    for i in range(k):
        if ln[i] > 0:
            first = False
    if first:
        return nm, nv, nn
    um = np.empty(k)
    uv = np.empty(k)
    un = np.empty(k)
    for i in range(k):
        un[i] = ln[i] + nn[i]
        if nn[i] > 0:
            lon = ln[i] / nn[i]
            lmi = lm[i] * ln[i]
            lvi = lv[i] * ln[i]
            nmi = nm[i] * nn[i]
            nvi = nv[i] * nn[i]
            uv[i] = lvi + nvi + lon / un[i] * (lmi / lon - nmi) ** 2
            um[i] = (lmi + nmi) / un[i]
            uv[i] /= un[i]
        else:
            uv[i] = lv[i]
            um[i] = lm[i]
            un[i] = ln[i]
    return um, uv, un

def csr_row_norms(X):
    """Squared L2 norm of each row of a CSR matrix."""
    _check_fmt(X)
    n = X.shape[0]
    out = np.zeros(n)
    for r in range(n):
        s = 0.0
        for i in range(X.indptr[r], X.indptr[r + 1]):
            v = float(X.data[i])
            s += v * v
        out[r] = s
    return out
