# Ported from numpy 2.5.4 (BSD-3-Clause)
#
# numpy.ma: set operations, sorting, fill-value helpers, polyfit,
# apply_along_axis, notmasked_contiguous and friends.

import numpy as np
from numpy.util import normalize_axis_index
from core import MaskedArray, default_fill_value, _dm, _binop, _unop, nomask, masked
from more1 import _ma1, _from, around

def set_fill_value(a, fill_value):
    a.set_fill_value(fill_value)

def common_fill_value(a, b):
    x = _ma1(a)
    y = _ma1(b)
    if x.dtype is y.dtype and x._fill == y._fill:
        return Optional[type(x._fill)](x._fill)
    return Optional[type(x._fill)]()

def maximum_fill_value(obj):
    return _ext_fill(_ma1(obj).dtype, True)

def minimum_fill_value(obj):
    return _ext_fill(_ma1(obj).dtype, False)

def _ext_fill(T: type, mx: bool):
    if T is float or T is float32:
        return T('-inf') if mx else T('inf')
    elif T is bool:
        return False if mx else True
    elif T is int:
        return -9223372036854775808 if mx else 9223372036854775807
    elif T is Int[32]:
        return Int[32](-2147483648) if mx else Int[32](2147483647)
    elif T is Int[16]:
        return Int[16](-32768) if mx else Int[16](32767)
    elif T is Int[8]:
        return Int[8](-128) if mx else Int[8](127)
    elif T is UInt[8] or T is UInt[16] or T is UInt[32] or T is UInt[64]:
        return T(0) if mx else ~T(0)
    else:
        compile_error("fill value extremum is not defined for this dtype")

def harden_mask(a):
    a._hard = True
    return a

def soften_mask(a):
    a._hard = False
    return a

def is_mask(m) -> bool:
    return isinstance(m, np.ndarray) and m.dtype is bool

def flatten_mask(mask):
    return np.asarray(mask, dtype=bool).ravel()

def make_mask_descr(ndtype):
    return bool

# ---- sorting ----
def argsort(a, axis: int = -1, kind: str = 'stable', endwith: bool = True, fill_value = None):
    x = _ma1(a)
    N: Literal[int] = static.len(x.shape)
    ax = normalize_axis_index(axis, N)
    # masked entries sort to the end (endwith) or to the start
    key = np.where(x._mask, np.zeros(1, x._data.dtype)[0], x._data)
    # two-key stable sort: data first, then mask flag as the dominant key
    flag = x._mask.astype(int)
    if not endwith:
        flag = 1 - flag
    return _lex_axis(key, flag, ax)

def _lex_axis(key, flag, ax: int):
    N: Literal[int] = static.len(key.shape)
    k2 = np.moveaxis(key, ax, -1)
    f2 = np.moveaxis(flag, ax, -1)
    out = np.empty(k2.shape, int)
    n = k2.shape[-1]
    L = k2.size // n if n > 0 else 0
    kk = np.ascontiguousarray(k2).reshape((L, n))
    ff = np.ascontiguousarray(f2).reshape((L, n))
    oo = np.empty((L, n), int)
    for r in range(L):
        o1 = np.argsort(kk[r], kind='stable')
        o2 = np.argsort(ff[r][o1], kind='stable')
        oo[r] = o1[o2]
    return np.moveaxis(oo.reshape(k2.shape), -1, ax)

# ---- sets ----
def _set_data(a):
    x = _ma1(a)
    return x.compressed() if True else x._data

def unique(ar1, return_index: Literal[bool] = False, return_inverse: Literal[bool] = False):
    x = _ma1(ar1).ravel()
    d = x._data
    m = x._mask
    vals = np.unique(d[~m])
    has_masked = bool(m.any())
    if has_masked:
        # numpy keeps one masked entry at the end
        out = np.append(vals, np.zeros(1, vals.dtype))
        om = np.zeros(out.size, bool)
        om[out.size - 1] = True
        return MaskedArray(out, om, False, x._fill, False)
    return MaskedArray(vals, np.zeros(vals.size, bool), True, x._fill, False)

def intersect1d(ar1, ar2, assume_unique: bool = False):
    x = _ma1(ar1)
    y = _ma1(ar2)
    r = np.intersect1d(x.compressed(), y.compressed())
    if bool(x._mask.any()) and bool(y._mask.any()):
        return MaskedArray(np.append(r, np.zeros(1, r.dtype)), np.append(np.zeros(r.size, bool), np.ones(1, bool)),
                           False, default_fill_value(r.dtype), False)
    return _from(r)

def setdiff1d(ar1, ar2, assume_unique: bool = False):
    r = np.setdiff1d(_ma1(ar1).compressed(), _ma1(ar2).compressed())
    return _from(r)

def setxor1d(ar1, ar2, assume_unique: bool = False):
    r = np.setxor1d(_ma1(ar1).compressed(), _ma1(ar2).compressed())
    return _from(r)

def union1d(ar1, ar2):
    x = _ma1(ar1)
    y = _ma1(ar2)
    r = np.union1d(x.compressed(), y.compressed())
    if bool(x._mask.any()) or bool(y._mask.any()):
        return MaskedArray(np.append(r, np.zeros(1, r.dtype)), np.append(np.zeros(r.size, bool), np.ones(1, bool)),
                           False, default_fill_value(r.dtype), False)
    return _from(r)

def isin(element, test_elements, assume_unique: bool = False, invert: bool = False):
    x = _ma1(element)
    r = np.isin(x.filled(0), _ma1(test_elements).compressed(), invert=invert) & ~x._mask
    return _from(r)

# ---- compress / put / choose ----
def compress(condition, a, axis = None):
    x = _ma1(a)
    c = np.asarray(condition, dtype=bool)
    return MaskedArray(np.compress(c, x._data, axis=axis), np.compress(c, x._mask, axis=axis), False, x._fill, False)

def compress_nd(x, axis = None):
    a = _ma1(x)
    d = a._data
    m = a._mask
    N: Literal[int] = static.len(a.shape)
    if axis is None:
        res = d
        mm = m
        for k in static.range(N):
            keep = ~(np.moveaxis(m, k, 0).reshape((m.shape[k], -1)).any(axis=1))
            res = np.compress(keep, res, axis=k)
        return res
    ax = normalize_axis_index(axis, N)
    keep = ~(np.moveaxis(m, ax, 0).reshape((m.shape[ax], -1)).any(axis=1))
    return np.compress(keep, d, axis=ax)

def put(a, indices, values, mode: str = 'raise'):
    a.__setitem__(np.asarray(indices), values)

def putmask(a, mask, values):
    mk = np.asarray(mask, dtype=bool)
    v = _ma1(values)
    n = v._data.size
    j = 0
    flat = a._data.reshape(-1)
    fm = a._mask.reshape(-1)
    mf = mk.reshape(-1)
    vd = v._data.ravel()
    vm = v._mask.ravel()
    for i in range(mf.size):
        if mf[i]:
            flat[i] = type(flat[0])(vd[i % n])
            fm[i] = vm[i % n]
    a._data = flat.reshape(a._data.shape)
    a._mask = fm.reshape(a._mask.shape)

def choose(indices, choices, out = None, mode: str = 'raise'):
    idx = np.asarray(indices)
    ds = [_ma1(c)._data for c in choices]
    ms = [_ma1(c)._mask for c in choices]
    d = np.choose(idx, ds)
    m = np.choose(idx, ms)
    if isinstance(indices, MaskedArray):
        m = m | indices._mask
    return MaskedArray(d, m, False, default_fill_value(d.dtype), False)

def inner(a, b):
    x = _ma1(a)
    y = _ma1(b)
    zx = np.where(x._mask, type(x._data.dtype())(0), x._data)
    zy = np.where(y._mask, type(y._data.dtype())(0), y._data)
    return np.inner(zx, zy)

innerproduct = inner

def convolve(a, v, mode: str = 'full', propagate_mask: bool = True):
    x = _ma1(a)
    y = _ma1(v)
    d = np.convolve(np.where(x._mask, type(x._data.dtype())(0), x._data),
                    np.where(y._mask, type(y._data.dtype())(0), y._data), mode)
    if propagate_mask:
        m = np.convolve(x._mask.astype(float), np.ones(y._data.size, float), mode) > 0.0
        m2 = np.convolve(np.ones(x._data.size, float), y._mask.astype(float), mode) > 0.0
        return MaskedArray(d, m | m2, False, x._fill, False)
    return _from(d)

def correlate(a, v, mode: str = 'valid', propagate_mask: bool = True):
    y = _ma1(v)
    rev = MaskedArray(y._data[::-1].copy(), y._mask[::-1].copy(), y._nomask, y._fill, False)
    return convolve(a, rev, mode, propagate_mask)

# ---- polyfit / apply_along_axis ----
def polyfit(x, y, deg: int, rcond = None, full: Literal[bool] = False, w = None, cov: Literal[bool] = False):
    xa = _ma1(x)
    ya = _ma1(y)
    m = xa._mask | ya._mask
    keep = ~m
    xd = xa._data[keep].astype(float)
    yd = ya._data[keep]
    if w is None:
        return np.polyfit(xd, yd, deg, rcond, full, None, cov)
    wd = np.asarray(w)[keep]
    return np.polyfit(xd, yd, deg, rcond, full, wd, cov)

def apply_along_axis(func1d, axis: int, arr):
    x = _ma1(arr)
    r = np.apply_along_axis(func1d, axis, x.filled(0) if True else x._data)
    return _from(r)

def apply_over_axes(func, a, axes):
    return np.apply_over_axes(func, _ma1(a).filled(0), axes)

def notmasked_contiguous(a, axis = None):
    from extras import flatnotmasked_contiguous
    x = _ma1(a)
    if axis is None or x.ndim == 1:
        return flatnotmasked_contiguous(x)
    compile_error("notmasked_contiguous: only axis=None or 1-d input supported")

def fromfunction(function, shape, dtype: type = float):
    return _from(np.fromfunction(function, shape, dtype=dtype))

def frombuffer(buffer, dtype: type = float, count: int = -1, offset: int = 0):
    return _from(np.frombuffer(buffer, dtype=dtype, count=count, offset=offset))

def ndenumerate(a, compressed: bool = True):
    x = _ma1(a)
    out = []
    for idx in np.ndindex(x.shape):
        if compressed and x._mask[idx]:
            continue
        out.append((idx, x._data[idx]))
    return out

# ---- aliases and leftovers ----
def append(a, b, axis = None):
    if axis is None:
        return concatenate_flat(a, b)
    return _concat2(a, b, axis)

def _concat2(a, b, axis: int):
    x = _ma1(a)
    y = _ma1(b)
    return MaskedArray(np.concatenate((x._data, y._data), axis=axis),
                       np.concatenate((x._mask, y._mask), axis=axis),
                       False, x._fill, False)

def concatenate_flat(a, b):
    x = _ma1(a)
    y = _ma1(b)
    return MaskedArray(np.concatenate((x._data.ravel(), y._data.ravel())),
                       np.concatenate((x._mask.ravel(), y._mask.ravel())),
                       False, x._fill, False)

def in1d(ar1, ar2, assume_unique: bool = False, invert: bool = False):
    return isin(ar1, ar2, assume_unique, invert)

def round_(a, decimals: int = 0):
    return around(a, decimals)

def alltrue(a, axis = None):
    return _ma1(a).all(axis)

def sometrue(a, axis = None):
    return _ma1(a).any(axis)

masked_print_option = '--'
