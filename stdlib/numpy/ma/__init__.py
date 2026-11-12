# Ported from numpy 2.5.4 (BSD-3-Clause)
#
# numpy.ma module-level constructors, predicates, masked math functions and
# array routines.

import numpy as np
from core import MaskedArray, default_fill_value, _dm, _binop, _unop, masked, nomask

def _mk(d: np.ndarray[X, N], m, nm: bool, fill_value, hard: bool, X: type, N: Literal[int]):
    mm = np.broadcast_to(np.asarray(m, dtype=bool), d.shape).copy()
    if fill_value is None:
        return MaskedArray(d, mm, nm, default_fill_value(X), hard)
    else:
        return MaskedArray(d, mm, nm, X(fill_value), hard)

def masked_array(data, mask = None, dtype: type = NoneType, copy: bool = False,
                 fill_value = None, keep_mask: bool = True, hard_mask: bool = False,
                 shrink: bool = True):
    if isinstance(data, MaskedArray):
        d = data._data.copy() if copy else data._data
        if dtype is not NoneType:
            d = d.astype(dtype)
        if mask is None:
            fv = fill_value
            if fv is None and dtype is NoneType:
                fv = data._fill
            return _mk(d, data._mask.copy(), data._nomask, fv, hard_mask)
        else:
            m = np.asarray(mask, dtype=bool)
            if keep_mask:
                return _mk(d, np.broadcast_to(m, d.shape) | data._mask, False, fill_value, hard_mask)
            return _mk(d, m, False, fill_value, hard_mask)
    else:
        if dtype is NoneType:
            d = np.array(data)
        else:
            d = np.array(data, dtype=dtype)
        if mask is None:
            return _mk(d, False, True, fill_value, hard_mask)
        elif isinstance(mask, bool):
            return _mk(d, mask, not mask and shrink, fill_value, hard_mask)
        else:
            return _mk(d, np.asarray(mask, dtype=bool), False, fill_value, hard_mask)

def array(data, mask = None, dtype: type = NoneType, copy: bool = False,
          fill_value = None, keep_mask: bool = True, hard_mask: bool = False,
          shrink: bool = True):
    return masked_array(data, mask=mask, dtype=dtype, copy=copy, fill_value=fill_value,
                        keep_mask=keep_mask, hard_mask=hard_mask, shrink=shrink)

asarray = masked_array
asanyarray = masked_array

def getdata(a):
    if isinstance(a, MaskedArray):
        return a._data
    return np.asarray(a)

def getmaskarray(a):
    if isinstance(a, MaskedArray):
        return a._mask
    return np.zeros(np.asarray(a).shape, bool)

def getmask(a):
    # nomask is reported as the scalar False
    return getmaskarray(a)

def is_masked(x) -> bool:
    if isinstance(x, MaskedArray):
        return bool(x._mask.any())
    elif x is None:
        return True
    return False

def isMaskedArray(x) -> bool:
    return isinstance(x, MaskedArray)

isMA = isMaskedArray
isarray = isMaskedArray

def make_mask(m, shrink: bool = True):
    return np.asarray(m, dtype=bool).copy()

def make_mask_none(shape):
    return np.zeros(shape, bool)

def mask_or(m1, m2):
    return np.logical_or(np.asarray(m1, dtype=bool), np.asarray(m2, dtype=bool))

def filled(a, fill_value = None):
    if isinstance(a, MaskedArray):
        return a.filled(fill_value)
    return np.asarray(a)

def compressed(x):
    if isinstance(x, MaskedArray):
        return x.compressed()
    return np.asarray(x).ravel()

def count(a, axis = None):
    return masked_array(a).count(axis)

def masked_where(condition, a, copy: bool = True):
    c = np.asarray(condition, dtype=bool)
    if isinstance(a, MaskedArray):
        d = a._data.copy() if copy else a._data
        return MaskedArray(d, np.broadcast_to(c, d.shape) | a._mask, False, a._fill, a._hard)
    d = np.array(a) if copy else np.asarray(a)
    return _mk(d, c, False, None, False)

def masked_invalid(a, copy: bool = True):
    d = getdata(a)
    return masked_where(~np.isfinite(d), a, copy)

def masked_equal(x, value, copy: bool = True):
    r = masked_where(getdata(x) == value, x, copy)
    r.fill_value = value
    return r

def masked_not_equal(x, value, copy: bool = True):
    return masked_where(getdata(x) != value, x, copy)

def masked_greater(x, value, copy: bool = True):
    return masked_where(getdata(x) > value, x, copy)

def masked_greater_equal(x, value, copy: bool = True):
    return masked_where(getdata(x) >= value, x, copy)

def masked_less(x, value, copy: bool = True):
    return masked_where(getdata(x) < value, x, copy)

def masked_less_equal(x, value, copy: bool = True):
    return masked_where(getdata(x) <= value, x, copy)

def masked_inside(x, v1, v2, copy: bool = True):
    lo = v1 if v1 <= v2 else v2
    hi = v2 if v1 <= v2 else v1
    d = getdata(x)
    return masked_where((d >= lo) & (d <= hi), x, copy)

def masked_outside(x, v1, v2, copy: bool = True):
    lo = v1 if v1 <= v2 else v2
    hi = v2 if v1 <= v2 else v1
    d = getdata(x)
    return masked_where((d < lo) | (d > hi), x, copy)

def masked_values(x, value, rtol: float = 1e-05, atol: float = 1e-08, copy: bool = True,
                  shrink: bool = True):
    d = getdata(x)
    if d.dtype is float or d.dtype is float32:
        c = np.isclose(d, value, rtol=rtol, atol=atol)
    else:
        c = d == value
    r = masked_where(c, x, copy)
    r.fill_value = value
    if shrink and not c.any():
        r._nomask = True
    return r

def fix_invalid(a, fill_value = None):
    r = masked_invalid(a, True)
    d = r._data
    if fill_value is None:
        d[r._mask] = r._fill
    else:
        d[r._mask] = fill_value
    return r

# ---- masked math ----------------------------------------------------
def add(a, b):
    return _binop(a, b, "add")
def subtract(a, b):
    return _binop(a, b, "sub")
def multiply(a, b):
    return _binop(a, b, "mul")
def divide(a, b):
    return _binop(a, b, "div")
true_divide = divide
def floor_divide(a, b):
    return _binop(a, b, "floordiv")
def remainder(a, b):
    return _binop(a, b, "mod")
mod = remainder
def power(a, b):
    return _binop(a, b, "pow")
def maximum(a, b):
    return _binop(a, b, "maximum")
def minimum(a, b):
    return _binop(a, b, "minimum")
def arctan2(a, b):
    return _binop(a, b, "arctan2")
def hypot(a, b):
    return _binop(a, b, "hypot")
def equal(a, b):
    return _binop(a, b, "eq")
def not_equal(a, b):
    return _binop(a, b, "ne")
def less(a, b):
    return _binop(a, b, "lt")
def less_equal(a, b):
    return _binop(a, b, "le")
def greater(a, b):
    return _binop(a, b, "gt")
def greater_equal(a, b):
    return _binop(a, b, "ge")
def logical_and(a, b):
    return _binop(np.asarray(getdata(a), dtype=bool) if not isinstance(a, MaskedArray) else a.astype(bool),
                  np.asarray(getdata(b), dtype=bool) if not isinstance(b, MaskedArray) else b.astype(bool), "and")
def logical_or(a, b):
    return _binop(np.asarray(getdata(a), dtype=bool) if not isinstance(a, MaskedArray) else a.astype(bool),
                  np.asarray(getdata(b), dtype=bool) if not isinstance(b, MaskedArray) else b.astype(bool), "or")

def sqrt(a):
    return _unop(a, np.sqrt, "sqrt")
def log(a):
    return _unop(a, np.log, "log")
def log2(a):
    return _unop(a, np.log2, "log")
def log10(a):
    return _unop(a, np.log10, "log")
def arcsin(a):
    return _unop(a, np.arcsin, "arc")
def arccos(a):
    return _unop(a, np.arccos, "arc")
def tan(a):
    return _unop(a, np.tan, "tan")
def exp(a):
    return _unop(a, np.exp, "")
def sin(a):
    return _unop(a, np.sin, "")
def cos(a):
    return _unop(a, np.cos, "")
def arctan(a):
    return _unop(a, np.arctan, "")
def sinh(a):
    return _unop(a, np.sinh, "")
def cosh(a):
    return _unop(a, np.cosh, "")
def tanh(a):
    return _unop(a, np.tanh, "")
def absolute(a):
    return _unop(a, np.abs, "")
abs = absolute
def negative(a):
    return _unop(a, np.negative, "")
def floor(a):
    return _unop(a, np.floor, "")
def ceil(a):
    return _unop(a, np.ceil, "")
def logical_not(a):
    return _unop(a, np.logical_not, "")

# ---- reductions -----------------------------------------------------
def sum(a, axis = None):
    return masked_array(a).sum(axis)
def prod(a, axis = None):
    return masked_array(a).prod(axis)
product = prod
def mean(a, axis = None):
    return masked_array(a).mean(axis)
def var(a, axis = None, ddof: int = 0):
    return masked_array(a).var(axis, ddof)
def std(a, axis = None, ddof: int = 0):
    return masked_array(a).std(axis, ddof)
def min(a, axis = None):
    return masked_array(a).min(axis)
def max(a, axis = None):
    return masked_array(a).max(axis)
def ptp(a, axis = None):
    return masked_array(a).ptp(axis)
def argmin(a, axis = None):
    return masked_array(a).argmin(axis)
def argmax(a, axis = None):
    return masked_array(a).argmax(axis)
def any(a, axis = None):
    return masked_array(a).any(axis)
def all(a, axis = None):
    return masked_array(a).all(axis)
def cumsum(a, axis = None):
    return masked_array(a).cumsum(axis)

# ---- shape routines -------------------------------------------------
def _parts(arrays):
    ds = [getdata(x) for x in arrays]
    ms = [getmaskarray(x) for x in arrays]
    nm = True
    for x in arrays:
        if isinstance(x, MaskedArray):
            if not x._nomask:
                nm = False
    return ds, ms, nm

def concatenate(arrays, axis: int = 0):
    ds, ms, nm = _parts(arrays)
    d = np.concatenate(ds, axis=axis)
    m = np.concatenate(ms, axis=axis)
    return MaskedArray(d, m, nm and not m.any(), default_fill_value(d.dtype), False)

def stack(arrays, axis: int = 0):
    ds, ms, nm = _parts(arrays)
    d = np.stack(ds, axis=axis)
    m = np.stack(ms, axis=axis)
    return MaskedArray(d, m, nm and not m.any(), default_fill_value(d.dtype), False)

def vstack(arrays):
    ds, ms, nm = _parts(arrays)
    d = np.vstack(ds)
    m = np.vstack(ms)
    return MaskedArray(d, m, nm and not m.any(), default_fill_value(d.dtype), False)

row_stack = vstack

def hstack(arrays):
    ds, ms, nm = _parts(arrays)
    d = np.hstack(ds)
    m = np.hstack(ms)
    return MaskedArray(d, m, nm and not m.any(), default_fill_value(d.dtype), False)

def where(condition, x, y):
    cd = np.asarray(filled(condition, False) if isinstance(condition, MaskedArray) else condition, dtype=bool)
    cm = getmaskarray(condition) if isinstance(condition, MaskedArray) else np.zeros(cd.shape, bool)
    xd, xm, xn = _dm(x)
    yd, ym, yn = _dm(y)
    d = np.where(cd, xd, yd)
    m = np.where(cd, xm, ym) | cm
    m = np.broadcast_to(m, d.shape).copy()
    return MaskedArray(d, m, False, default_fill_value(d.dtype), False)

def transpose(a):
    return masked_array(a).transpose()

def ravel(a):
    return masked_array(a).ravel()

def reshape(a, shape):
    return masked_array(a).reshape(shape)

def diagonal(a):
    x = masked_array(a)
    return MaskedArray(np.diagonal(x._data).copy(), np.diagonal(x._mask).copy(), x._nomask, x._fill, False)

def dot(a, b, strict: bool = False):
    ad, am, an = _dm(a)
    bd, bm, bn = _dm(b)
    d = np.dot(np.where(am, type(ad.dtype())(0), ad), np.where(bm, type(bd.dtype())(0), bd))
    na = (~am).astype(int)
    nb = (~bm).astype(int)
    m = ~(np.dot(na, nb) > 0)
    if strict:
        m = (np.dot(am.astype(int), np.ones(bm.shape, int)) > 0) | m
    return MaskedArray(d, m, False, default_fill_value(d.dtype), False)

def outer(a, b):
    ad, am, an = _dm(a)
    bd, bm, bn = _dm(b)
    d = np.outer(ad.ravel(), bd.ravel())
    m = (am.ravel().reshape((-1, 1)) | bm.ravel().reshape((1, -1)))
    return MaskedArray(d, np.broadcast_to(m, d.shape).copy(), an and bn, default_fill_value(d.dtype), False)

outerproduct = outer

def sort(a, axis: int = -1, endwith: bool = True):
    x = masked_array(a)
    if x.ndim != 1:
        compile_error("numpy.ma.sort: only 1-d arrays are supported")
    n = x.size
    idx = np.argsort(x.filled(), kind="stable")
    d = x._data[idx]
    m = x._mask[idx]
    return MaskedArray(d, m, x._nomask, x._fill, False)

from extras import *
