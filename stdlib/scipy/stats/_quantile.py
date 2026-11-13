# Copyright (C) 2026 the Typhon authors
# Ported from scipy 1.18.1 (BSD-3-Clause)

# scipy.stats.quantile and estimated_cdf (1-D or N-D x along ``axis``; p a
# scalar or 1-D array; optional weights for the Hyndman-Fan methods).

import numpy as np
import scipy.special._kernels as K
from ._common import _f64

NAN = K.NAN

def _ms(method: str, p: float) -> float:
    if method == 'inverted_cdf' or method == 'averaged_inverted_cdf' or method == 'interpolated_inverted_cdf':
        return 0.0
    elif method == 'closest_observation':
        return -0.5
    elif method == 'hazen':
        return 0.5
    elif method == 'weibull':
        return p
    elif method == 'linear':
        return 1 - p
    elif method == 'median_unbiased':
        return p / 3 + 1 / 3
    else:
        return p / 4 + 3 / 8

def _isnan_any(y: np.ndarray[float, 1]) -> bool:
    for i in range(y.shape[0]):
        if y[i] != y[i]:
            return True
    return False

def _round_half_even(v: float) -> float:
    f = K.floor(v)
    d = v - f
    if d < 0.5:
        return f
    if d > 0.5:
        return f + 1
    return f if int(f) % 2 == 0 else f + 1

def _clipf(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)

def _quantile1(y: np.ndarray[float, 1], w: np.ndarray[float, 1], has_w: bool, p: float,
               method: str, nan_policy: str) -> float:
    if not (p >= 0 and p <= 1):
        return NAN
    # drop zero-weight entries, sort
    vals = []
    wts = []
    for i in range(y.shape[0]):
        if has_w and w[i] == 0:
            continue
        vals.append(y[i])
        wts.append(w[i] if has_w else 1.0)
    yy = np.array(vals, dtype=float)
    ww = np.array(wts, dtype=float)
    if nan_policy == 'omit':
        kv = []
        kw = []
        for i in range(yy.shape[0]):
            if yy[i] == yy[i]:
                kv.append(yy[i])
                kw.append(ww[i])
        yy = np.array(kv, dtype=float)
        ww = np.array(kw, dtype=float)
    elif _isnan_any(yy):
        return NAN
    if yy.shape[0] == 0:
        return NAN
    order = np.argsort(yy)
    yy = yy[order]
    ww = ww[order]
    n = yy.shape[0]
    nf = float(n)
    if method == 'harrell-davis':
        a = p * (nf + 1)
        b = (1 - p) * (nf + 1)
        s = 0.0
        prev = 0.0
        for i in range(1, n + 1):
            cur = K.betainc(a, b, i / nf)
            wi = cur - prev
            prev = cur
            if wi == wi:
                s += wi * yy[i - 1]
        return s
    if method == '_lower' or method == '_midpoint' or method == '_higher' or method == '_nearest':
        ij = p * (nf - 1)
        if method == '_midpoint':
            return (yy[int(K.floor(ij))] + yy[int(K.ceil(ij))]) / 2
        k = K.floor(ij) if method == '_lower' else (K.ceil(ij) if method == '_higher' else _round_half_even(ij))
        return yy[int(k)]
    if method == 'round_outward' or method == 'round_inward' or method == 'round_nearest':
        v = p * nf if p < 0.5 else nf * p - 1
        if method == 'round_nearest':
            j = _round_half_even(v)
        elif method == 'round_outward':
            j = K.floor(v) if p < 0.5 else K.ceil(v)
        else:
            j = K.ceil(v) if p < 0.5 else K.floor(v)
        return yy[int(j)]
    m = _ms(method, p)
    jg = 0.0
    jp1 = 0.0
    j = 0.0
    if not has_w:
        jg = p * nf + m
        jp1 = K.floor(jg)
        j = jp1 - 1
    else:
        cw = np.cumsum(ww)
        total = cw[n - 1]
        jg = p * total + m
        a = 0
        while a < n and cw[a] <= jg:
            a += 1
        b = 0
        while b < n and cw[b] <= jg - 1:
            b += 1
        jp1 = float(a)
        j = float(b)
    g = jg - K.floor(jg)
    if method == 'inverted_cdf':
        g = 1.0 if g > 0 else 0.0
    elif method == 'averaged_inverted_cdf':
        g = (1 + (1.0 if g > 0 else 0.0)) / 2
    elif method == 'closest_observation':
        g = 1 - (1.0 if (g == 0 and int(j) % 2 == 1) else 0.0)
    if method == 'inverted_cdf' or method == 'averaged_inverted_cdf' or method == 'closest_observation':
        if jg < 0:
            g = 0.0
    if j < 0:
        g = 0.0
    j = _clipf(j, 0.0, nf - 1)
    jp1 = _clipf(jp1, 0.0, nf - 1)
    return (1 - g) * yy[int(j)] + g * yy[int(jp1)]

def _check_method(method: str):
    ok = method == 'inverted_cdf' or method == 'averaged_inverted_cdf' or method == 'closest_observation' \
        or method == 'hazen' or method == 'interpolated_inverted_cdf' or method == 'linear' \
        or method == 'median_unbiased' or method == 'normal_unbiased' or method == 'weibull' \
        or method == 'harrell-davis' or method == '_lower' or method == '_midpoint' \
        or method == '_higher' or method == '_nearest' or method == 'round_outward' \
        or method == 'round_inward' or method == 'round_nearest'
    if not ok:
        raise ValueError("`method` must be one of {'inverted_cdf', 'averaged_inverted_cdf', 'closest_observation', 'hazen', 'interpolated_inverted_cdf', 'linear', 'median_unbiased', 'normal_unbiased', 'weibull', 'harrell-davis', '_lower', '_midpoint', '_higher', '_nearest', 'round_outward', 'round_inward', 'round_nearest'}")

def quantile(x, p, *, method: str = 'linear', axis = 0, nan_policy: str = 'propagate',
             keepdims = None, weights = None):
    _check_method(method)
    if nan_policy != 'propagate' and nan_policy != 'omit':
        raise ValueError("nan_policy must be one of {'propagate', 'omit'}")
    xa = _f64(x)
    pa = _f64(p)
    has_w = weights is not None
    if has_w:
        if method == '_lower' or method == '_midpoint' or method == '_higher' or method == '_nearest' \
           or method == 'harrell-davis' or method == 'round_nearest' or method == 'round_inward' \
           or method == 'round_outward':
            raise ValueError(f"`method='{method}'` does not support `weights`.")
    wa = _f64(weights) if has_w else np.zeros(xa.shape)
    if has_w:
        wa = np.broadcast_to(wa, xa.shape)
    p_scalar = pa.ndim == 0
    pv = pa.ravel()
    k = pv.shape[0]
    if axis is None:
        xa = xa.ravel()
        wa = wa.ravel()
        ax = 0
    else:
        ax = int(axis)
        if ax < 0:
            ax += xa.ndim
    xm = np.moveaxis(xa, ax, -1)
    wm = np.moveaxis(wa, ax, -1)
    if keepdims is not None and not keepdims and k != 1:
        raise ValueError("`keepdims` may be False only if the length of `p` along `axis` is 1.")
    keep = (k != 1) if keepdims is None else bool(keepdims)
    bshape = xm.shape[:-1]
    out = np.empty(bshape + (k,), dtype=float)
    for idx in np.ndindex(bshape):
        row = np.ascontiguousarray(xm[idx])
        wrow = np.ascontiguousarray(wm[idx])
        for j in range(k):
            out[idx + (j,)] = _quantile1(row, wrow, has_w, float(pv[j]), method, nan_policy)
    if keep:
        res = np.moveaxis(out, -1, ax)
        return res
    return out.reshape(bshape)

def _ecdf1(x: np.ndarray[float, 1], yv: float, method: str, nan_policy: str) -> float:
    if yv != yv:
        return NAN
    xs = []
    for i in range(x.shape[0]):
        if x[i] == x[i]:
            xs.append(x[i])
        elif nan_policy == 'propagate':
            return NAN
    s = np.sort(np.array(xs, dtype=float))
    n = s.shape[0]
    if n == 0:
        return NAN
    jp1 = 0
    while jp1 < n and s[jp1] <= yv:
        jp1 += 1
    p = 0.0
    if method == 'inverted_cdf' or method == 'averaged_inverted_cdf' or method == 'closest_observation':
        dp = 0.5 if method == 'closest_observation' else 0.0
        p = (jp1 + dp) / n
    else:
        a = 0.0
        b = 0.0
        if method == 'interpolated_inverted_cdf':
            a = 0.0
            b = 1.0
        elif method == 'hazen':
            a = 0.5
            b = 0.5
        elif method == 'weibull':
            a = 0.0
            b = 0.0
        elif method == 'linear':
            a = 1.0
            b = 1.0
        elif method == 'median_unbiased':
            a = 1 / 3
            b = 1 / 3
        else:
            a = 3 / 8
            b = 3 / 8
        j_max = n - 1
        j_min = min(j_max, 1)
        j1 = min(max(jp1, j_min), j_max)
        j0 = max(j1 - 1, 0)
        xj = s[j0]
        xj1 = s[j1]
        delta = (yv - xj) / (xj1 - xj) if (xj1 > xj and K.isfinite(xj)) else 1.0
        p = (j1 + delta - a) / (n + 1 - a - b)
    if yv < s[0]:
        p = 0.0
    if yv > s[n - 1]:
        p = 1.0
    return _clipf(p, 0.0, 1.0)

def estimated_cdf(x, y, *, method: str = 'linear', axis = 0, nan_policy: str = 'propagate',
                  keepdims = None):
    xa = _f64(x)
    ya = _f64(y)
    ax = 0 if axis is None else (int(axis) + xa.ndim if int(axis) < 0 else int(axis))
    if axis is None:
        xa = xa.ravel()
    xm = np.moveaxis(xa, ax, -1)
    yv = ya.ravel()
    k = yv.shape[0]
    bshape = xm.shape[:-1]
    out = np.empty(bshape + (k,), dtype=float)
    for idx in np.ndindex(bshape):
        row = np.ascontiguousarray(xm[idx])
        for j in range(k):
            out[idx + (j,)] = _ecdf1(row, float(yv[j]), method, nan_policy)
    keep = (k != 1) if keepdims is None else bool(keepdims)
    if keep:
        return np.moveaxis(out, -1, ax)
    return out.reshape(bshape)
