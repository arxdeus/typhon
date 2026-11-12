# Ported from numpy 2.5.4 (BSD-3-Clause)
#
# numpy.ma.extras: average, median, cov/corrcoef, edges and clumps.

import numpy as np
from numpy.util import tuple_set
from core import MaskedArray, default_fill_value, _dm, _binop, _unop
from _red import lanes, flat2, lane

def _ma(a):
    if isinstance(a, MaskedArray):
        return a
    d = np.array(a)
    return MaskedArray(d, np.zeros(d.shape, bool), True, default_fill_value(d.dtype), False)

def average(a, axis = None, weights = None, returned: Literal[bool] = False):
    x = _ma(a)
    if weights is None:
        avg = x.mean(axis)
        if axis is None:
            scl = float(x.count())
        else:
            scl = x.count(axis).astype(float)
        if returned:
            return avg, scl
        return avg
    else:
        N: Literal[int] = static.len(x.shape)
        w0 = np.asarray(weights).astype(float)
        if static.len(w0.shape) == N:
            if w0.shape != x.shape:
                raise ValueError("Shape of weights must be consistent with shape of a along specified axis.")
            w = w0
        else:
            if axis is None:
                raise TypeError("Axis must be specified when shapes of a and weights differ.")
            ax = axis if axis >= 0 else axis + N
            if static.len(w0.shape) != 1 or w0.shape[0] != x.shape[ax]:
                raise ValueError("Shape of weights must be consistent with shape of a along specified axis.")
            w = np.broadcast_to(w0.reshape(tuple_set((1,) * N, ax, w0.shape[0])), x.shape).copy()
        wm = MaskedArray(w * (~x._mask), x._mask.copy(), x._nomask, 1e20, False)
        prod = MaskedArray(x._data.astype(float) * w, x._mask.copy(), x._nomask, 1e20, False)
        if axis is None:
            scl = wm.sum()
            s = prod.sum()
            if s is None or scl is None:
                avg = Optional[float]()
            else:
                avg = Optional[float](s.__val__() / scl.__val__())
            if returned:
                return avg, scl
            return avg
        else:
            scl = wm.sum(axis)
            avg = prod.sum(axis) / scl
            if returned:
                return avg, scl
            return avg

def median(a, axis = None):
    x = _ma(a)
    return x._red("median", axis, 0.0, float)

def count_masked(arr, axis = None):
    m = _ma(arr)._mask
    if axis is None:
        return int(m.sum())
    return m.sum(axis=axis)

def flatnotmasked_edges(a):
    x = _ma(a)
    m = x._mask.ravel()
    first = -1
    last = -1
    for i in range(m.size):
        if not m[i]:
            if first < 0:
                first = i
            last = i
    if first < 0:
        return Optional[np.ndarray[int, 1]]()
    return Optional[np.ndarray[int, 1]](np.array([first, last]))

def notmasked_edges(a, axis = None):
    x = _ma(a)
    if axis is not None and x.ndim != 1:
        compile_error("notmasked_edges: only axis=None or 1-d input supported")
    return flatnotmasked_edges(x)

def _ezclump(mask) -> List[slice]:
    r = List[slice]()
    n = mask.size
    i = 0
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            r.append(slice(i, j))
            i = j
        else:
            i += 1
    return r

def clump_masked(a) -> List[slice]:
    x = _ma(a)
    if x._nomask:
        return List[slice]()
    return _ezclump(x._mask.ravel())

def clump_unmasked(a) -> List[slice]:
    x = _ma(a)
    return _ezclump(~x._mask.ravel())

def flatnotmasked_contiguous(a) -> List[slice]:
    return clump_unmasked(a)

def mask_rowcols(a, axis = None):
    x = _ma(a)
    m = x._mask
    r = m.any(axis=1)
    c = m.any(axis=0)
    nm = m.copy()
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            rr = r[i] and (axis is None or axis == 0)
            cc = c[j] and (axis is None or axis == 1 or axis == -1)
            if rr or cc:
                nm[i, j] = True
    return MaskedArray(x._data.copy(), nm, False, x._fill, False)

def mask_rows(a):
    return mask_rowcols(a, 0)

def mask_cols(a):
    return mask_rowcols(a, 1)

def compress_rows(a):
    x = _ma(a)
    keep = ~x._mask.any(axis=1)
    return x._data[keep]

def compress_cols(a):
    x = _ma(a)
    keep = ~x._mask.any(axis=0)
    return x._data[:, keep]

def compress_rowcols(x, axis = None):
    a = _ma(x)
    rk = ~a._mask.any(axis=1)
    ck = ~a._mask.any(axis=0)
    if axis is None:
        return a._data[rk][:, ck]
    elif axis == 0:
        return a._data[rk]
    else:
        return a._data[:, ck]

def _covhelper(x, y, rowvar: bool, allow_masked: bool):
    xa = _ma(x)
    xd2 = np.atleast_2d(xa._data.astype(float))
    xm2 = np.atleast_2d(xa._mask)
    if not rowvar:
        xd2 = xd2.T
        xm2 = xm2.T
    if xd2.shape[0] == 1 and False:
        pass
    if y is not None:
        ya = _ma(y)
        yd2 = np.atleast_2d(ya._data.astype(float))
        ym2 = np.atleast_2d(ya._mask)
        if not rowvar:
            yd2 = yd2.T
            ym2 = ym2.T
        if not allow_masked and (xm2.any() or ym2.any()):
            raise ValueError("Cannot process masked data.")
        if xd2.shape == yd2.shape:
            cm = xm2 | ym2
            xm2 = cm
            ym2 = cm
        xd2 = np.concatenate((xd2, yd2), axis=0)
        xm2 = np.concatenate((xm2, ym2), axis=0)
    elif not allow_masked and xm2.any():
        raise ValueError("Cannot process masked data.")
    # rows are variables, columns observations; center by masked row means
    V = xd2.shape[0]
    n = xd2.shape[1]
    xc = xd2.copy()
    for i in range(V):
        s = 0.0
        c = 0
        for j in range(n):
            if not xm2[i, j]:
                s += xd2[i, j]
                c += 1
        mu = s / c if c > 0 else 0.0
        for j in range(n):
            xc[i, j] = 0.0 if xm2[i, j] else xd2[i, j] - mu
    notm = (~xm2).astype(float)
    return xc, notm

def cov(x, y = None, rowvar: bool = True, bias: bool = False, allow_masked: bool = True,
        ddof: Optional[int] = None):
    dd = 0
    if ddof is None:
        dd = 0 if bias else 1
    else:
        dd = ddof.__val__()
    xc, notm = _covhelper(x, y, rowvar, allow_masked)
    fact = np.dot(notm, notm.T) - float(dd)
    mask = fact <= 0.0
    data = np.dot(xc, xc.T) / np.where(mask, 1.0, fact)
    if data.shape[0] == 1:
        return MaskedArray(data, mask, False, 1e20, False)
    return MaskedArray(data, mask, False, 1e20, False)

def corrcoef(x, y = None, rowvar: bool = True, allow_masked: bool = True):
    c = cov(x, y, rowvar, False, allow_masked, None)
    V = c._data.shape[0]
    d = c._data.copy()
    m = c._mask.copy()
    sd = np.empty(V)
    sm = np.zeros(V, bool)
    for i in range(V):
        if m[i, i] or d[i, i] < 0.0:
            sm[i] = True
            sd[i] = 1.0
        else:
            sd[i] = np.sqrt(d[i, i])
    for i in range(V):
        for j in range(V):
            den = sd[i] * sd[j]
            if sm[i] or sm[j] or den == 0.0:
                m[i, j] = True
            else:
                d[i, j] = d[i, j] / den
    return MaskedArray(d, m, False, 1e20, False)
