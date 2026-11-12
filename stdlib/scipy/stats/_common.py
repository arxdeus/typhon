# Ported from scipy 1.18.1 (BSD-3-Clause)

# Private helpers for scipy.stats: axis handling and 1-D kernels.

import numpy as np
import internal.static as static
import scipy.special._kernels as K

def _is_arr(x) -> Literal[bool]:
    return isinstance(x, np.ndarray) or isinstance(x, List) or isinstance(x, Tuple)

def _anyarr(*args) -> Literal[bool]:
    if static.len(args) == 0:
        return False
    else:
        return _is_arr(args[0]) or _anyarr(*args[1:])

def _vmap(f, *args):
    # elementwise scalar function, broadcasting over array arguments
    if _anyarr(*args):
        arrs = tuple(np.asarray(a) for a in args)
        shp = np.broadcast_shapes(*tuple(a.shape for a in arrs))
        bs = tuple(np.broadcast_to(a, shp) for a in arrs)
        out = np.empty(shp, dtype=float)
        for idx in np.ndindex(shp):
            out[idx] = f(*tuple(float(b[idx]) for b in bs))
        return out
    else:
        return f(*tuple(float(a) for a in args))

def _f64(a):
    return np.asarray(a).astype(float)

def _slices(a, axis):
    """Returns (array with ``axis`` moved last, result shape)."""
    if axis is None:
        r = _f64(a).ravel()
        return r, ()
    else:
        x = _f64(a)
        ax = int(axis)
        if ax < 0:
            ax += x.ndim
        m = np.moveaxis(x, ax, -1)
        return m, m.shape[:-1]

def _reduce(a, axis, f):
    """Apply ``f(1-D float array) -> float`` along ``axis``; scalar for 1-D
    input or ``axis=None``."""
    m, shp = _slices(a, axis)
    if static.len(shp) == 0:
        return f(np.ascontiguousarray(m))
    else:
        out = np.empty(shp, dtype=float)
        for idx in np.ndindex(shp):
            out[idx] = f(np.ascontiguousarray(m[idx]))
        return out

def _reduce2(a, b, axis, f):
    ma, shp = _slices(a, axis)
    mb, shpb = _slices(b, axis)
    if static.len(shp) == 0:
        return f(np.ascontiguousarray(ma), np.ascontiguousarray(mb))
    else:
        out = np.empty(shp, dtype=float)
        for idx in np.ndindex(shp):
            out[idx] = f(np.ascontiguousarray(ma[idx]), np.ascontiguousarray(mb[idx]))
        return out

# 1-D kernels (numpy pairwise summation via np.mean/np.sum for bitwise parity)

def _mean(x) -> float:
    return float(np.mean(x))

def _moment(x, order: int) -> float:
    c = np.mean(x)
    d = x - c
    if order == 2:
        return float(np.mean(d * d))
    elif order == 3:
        return float(np.mean(d ** 3))
    elif order == 4:
        return float(np.mean(d ** 4))
    else:
        return float(np.mean(d ** order))

def _var(x, ddof: int) -> float:
    n = float(x.shape[0])
    v = _moment(x, 2)
    if ddof != 0:
        v *= n / (n - ddof)
    return v

def _rank_avg(x):
    """Average ranks (1-based) and tie counts of a 1-D float array."""
    n = x.shape[0]
    order = np.argsort(x, kind='mergesort')
    ranks = np.empty(n, dtype=float)
    ties = []
    i = 0
    while i < n:
        j = i
        while j + 1 < n and x[order[j + 1]] == x[order[i]]:
            j += 1
        r = 0.5 * (i + j) + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = r
        ties.append(float(j - i + 1))
        i = j + 1
    return ranks, ties

def _tie_term(ties) -> float:
    s = 0.0
    for t in ties:
        s += t * t * t - t
    return s

def _pvalue_t(t: float, df: float, alternative: str) -> float:
    if alternative == 'less':
        return K.stdtr(df, t)
    elif alternative == 'greater':
        return K.stdtr(df, -t)
    elif alternative == 'two-sided':
        return 2 * K.stdtr(df, -abs(t))
    else:
        raise ValueError("`alternative` must be 'less', 'greater', or 'two-sided'.")

def _pvalue_norm(z: float, alternative: str) -> float:
    if alternative == 'less':
        return K.ndtr(z)
    elif alternative == 'greater':
        return K.ndtr(-z)
    elif alternative == 'two-sided':
        return 2 * K.ndtr(-abs(z))
    else:
        raise ValueError("`alternative` must be 'less', 'greater', or 'two-sided'.")

def _div(a: float, b: float) -> float:
    return a.cdiv(b)
