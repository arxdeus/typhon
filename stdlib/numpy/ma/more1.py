# Ported from numpy 2.5.4 (BSD-3-Clause)
#
# numpy.ma: elementwise functions, creators and shape routines.

import numpy as np
from numpy.util import tuple_delete, normalize_axis_index
from core import MaskedArray, default_fill_value, _dm, _binop, _unop, nomask, masked
from _red import lanes

MaskType = bool
bool_ = bool
masked_singleton = None

def _from(d, m = None):
    dd = np.asarray(d)
    if m is None:
        return MaskedArray(dd, np.zeros(dd.shape, bool), True, default_fill_value(dd.dtype), False)
    return MaskedArray(dd, np.broadcast_to(np.asarray(m, dtype=bool), dd.shape).copy(), False,
                       default_fill_value(dd.dtype), False)

def _ma1(a):
    if isinstance(a, MaskedArray):
        return a
    return _from(np.array(a))

# ---- elementwise ----
def arcsinh(a):
    return _unop(a, np.arcsinh, "")
def arccosh(a):
    return _unop(a, np.arccosh, "inv1")
def arctanh(a):
    return _unop(a, np.arctanh, "inv2")
def fabs(a):
    return _unop(a, np.fabs, "")
def conjugate(a):
    return _unop(a, np.conjugate, "")
def angle(a):
    return _unop(a, np.angle, "")

def around(a, decimals: int = 0):
    x = _ma1(a)
    return MaskedArray(np.round(x._data, decimals), x._mask.copy(), x._nomask, x._fill, False)

round = around

def clip(a, a_min, a_max):
    x = _ma1(a)
    return MaskedArray(np.clip(x._data, a_min, a_max), x._mask.copy(), x._nomask, x._fill, False)

def bitwise_and(a, b):
    return _binop(a, b, "and")
def bitwise_or(a, b):
    return _binop(a, b, "or")
def bitwise_xor(a, b):
    return _binop(a, b, "xor")
def left_shift(a, n):
    return _binop(a, n, "lshift")
def right_shift(a, n):
    return _binop(a, n, "rshift")
def fmod(a, b):
    return _binop(a, b, "fmod")
def logical_xor(a, b):
    return _binop(_ma1(a).astype(bool), _ma1(b).astype(bool), "ne")

def amax(a, axis = None):
    return _ma1(a).max(axis)
def amin(a, axis = None):
    return _ma1(a).min(axis)

def cumprod(a, axis = None):
    x = _ma1(a)
    d = np.where(x._mask, type(x._data.dtype())(1), x._data)
    if axis is None:
        r = np.cumprod(d.ravel())
        return MaskedArray(r, x._mask.ravel().copy(), x._nomask, x._fill, False)
    r = np.cumprod(d, axis=axis)
    return MaskedArray(r, x._mask.copy(), x._nomask, x._fill, False)

def diff(a, n: int = 1, axis: int = -1):
    x = _ma1(a)
    d = x._data
    m = x._mask
    for _ in range(n):
        d = np.diff(d, axis=axis)
        m = np.diff(m.astype(int), axis=axis) != 0
        m = m | _lag_or(x._mask, axis, n)
    return MaskedArray(d, m, False, x._fill, False)

def _lag_or(m, axis: int, n: int):
    # a difference is masked when any of the n+1 entries it uses is masked
    r = m
    for _ in range(n):
        N: Literal[int] = static.len(r.shape)
        ax = normalize_axis_index(axis, N)
        k = r.shape[ax]
        hi = np.take(r, np.arange(1, k), axis=ax)
        lo = np.take(r, np.arange(0, k - 1), axis=ax)
        r = hi | lo
    return r

def ediff1d(arr, to_end = None, to_begin = None):
    x = _ma1(arr).ravel()
    d = np.diff(x._data)
    m = x._mask[1:] | x._mask[:-1]
    return MaskedArray(d, m, False, x._fill, False)

def allclose(a, b, masked_equal: bool = True, rtol: float = 1e-5, atol: float = 1e-8):
    x = _ma1(a)
    y = _ma1(b)
    xd = x._data.astype(float)
    yd = y._data.astype(float)
    xm = np.broadcast_to(x._mask, np.broadcast(xd, yd).shape)
    ym = np.broadcast_to(y._mask, np.broadcast(xd, yd).shape)
    close = np.abs(xd - yd) <= atol + rtol * np.abs(yd)
    both = xm & ym
    either = xm | ym
    if masked_equal:
        return bool(np.where(either, both, close).all())
    return bool(np.where(either, False, close).all())

def allequal(a, b, fill_value: bool = True):
    x = _ma1(a)
    y = _ma1(b)
    eq = x._data == y._data
    either = x._mask | y._mask
    if fill_value:
        return bool(np.where(either, True, eq).all())
    return bool(np.where(either, False, eq).all())

def anom(a, axis = None, dtype: type = NoneType):
    x = _ma1(a).astype(float)
    mu = x.mean(axis)
    if axis is None:
        m0 = mu if mu is not None else 0.0
        return MaskedArray(x._data - m0, x._mask.copy(), x._nomask, x._fill, False)
    N: Literal[int] = static.len(x.shape)
    ax = normalize_axis_index(axis, N)
    mm = np.expand_dims(mu._data, ax)
    return MaskedArray(x._data - mm, x._mask.copy(), x._nomask, x._fill, False)

anomalies = anom

# ---- creators ----
def zeros(shape, dtype: type = float):
    return _from(np.zeros(shape, dtype))
def ones(shape, dtype: type = float):
    return _from(np.ones(shape, dtype))
def empty(shape, dtype: type = float):
    return _from(np.empty(shape, dtype))
def zeros_like(a, dtype: type = NoneType):
    return _from(np.zeros_like(_ma1(a)._data) if dtype is NoneType else np.zeros_like(_ma1(a)._data, dtype=dtype))
def ones_like(a, dtype: type = NoneType):
    return _from(np.ones_like(_ma1(a)._data) if dtype is NoneType else np.ones_like(_ma1(a)._data, dtype=dtype))
def empty_like(a, dtype: type = NoneType):
    return _from(np.empty_like(_ma1(a)._data) if dtype is NoneType else np.empty_like(_ma1(a)._data, dtype=dtype))
def arange(*args):
    return _from(np.arange(*args))
def identity(n: int, dtype: type = float):
    return _from(np.identity(n, dtype))
def indices(dimensions, dtype: type = int):
    return _from(np.indices(dimensions, dtype))
def masked_all(shape, dtype: type = float):
    d = np.empty(shape, dtype)
    return MaskedArray(d, np.ones(d.shape, bool), False, default_fill_value(dtype), False)
def masked_all_like(arr):
    d = np.empty_like(_ma1(arr)._data)
    return MaskedArray(d, np.ones(d.shape, bool), False, default_fill_value(d.dtype), False)
def masked_object(x, value, copy: bool = True, shrink: bool = True):
    xa = np.array(x)
    return MaskedArray(xa, xa == value, False, default_fill_value(xa.dtype), False)
def diag(v, k: int = 0):
    x = _ma1(v)
    return MaskedArray(np.diag(x._data, k), np.diag(x._mask, k), False, x._fill, False)
def diagflat(v, k: int = 0):
    x = _ma1(v)
    return MaskedArray(np.diagflat(x._data, k), np.diagflat(x._mask, k) | ~np.diagflat(np.ones(x._data.size, bool), k),
                       False, x._fill, False)
def vander(x, n = None):
    xa = _ma1(x)
    d = np.vander(xa._data) if n is None else np.vander(xa._data, n)
    cols = d.shape[1]
    m = np.broadcast_to(xa._mask.reshape((-1, 1)), d.shape).copy()
    return MaskedArray(d, m, False, xa._fill, False)
def copy(a):
    return _ma1(a).copy()

# ---- shape ----
def ndim(a) -> int:
    return _ma1(a).ndim
def shape(a):
    return _ma1(a).shape
def size(a, axis = None) -> int:
    x = _ma1(a)
    if axis is None:
        return x.size
    return x.shape[axis]
def squeeze(a):
    return _ma1(a).squeeze()
def swapaxes(a, axis1: int, axis2: int):
    x = _ma1(a)
    return MaskedArray(np.swapaxes(x._data, axis1, axis2), np.swapaxes(x._mask, axis1, axis2), x._nomask, x._fill, False)
def expand_dims(a, axis: int):
    x = _ma1(a)
    return MaskedArray(np.expand_dims(x._data, axis), np.expand_dims(x._mask, axis), x._nomask, x._fill, False)
def atleast_1d(a):
    x = _ma1(a)
    return MaskedArray(np.atleast_1d(x._data), np.atleast_1d(x._mask), x._nomask, x._fill, False)
def atleast_2d(a):
    x = _ma1(a)
    return MaskedArray(np.atleast_2d(x._data), np.atleast_2d(x._mask), x._nomask, x._fill, False)
def atleast_3d(a):
    x = _ma1(a)
    return MaskedArray(np.atleast_3d(x._data), np.atleast_3d(x._mask), x._nomask, x._fill, False)
def repeat(a, repeats: int, axis = None):
    x = _ma1(a)
    return MaskedArray(np.repeat(x._data, repeats, axis=axis), np.repeat(x._mask, repeats, axis=axis), x._nomask, x._fill, False)
def resize(x, new_shape):
    xa = _ma1(x)
    return MaskedArray(np.resize(xa._data, new_shape), np.resize(xa._mask, new_shape), False, xa._fill, False)
def take(a, indices, axis = None):
    x = _ma1(a)
    return MaskedArray(np.take(x._data, indices, axis=axis), np.take(x._mask, indices, axis=axis), False, x._fill, False)
def trace(a, offset: int = 0):
    return _trace(_ma1(a), offset)
def _trace(x, offset: int):
    d = np.diagonal(x._data, offset)
    m = np.diagonal(x._mask, offset)
    return MaskedArray(d.copy(), m.copy(), False, x._fill, False).sum()
def nonzero(a):
    x = _ma1(a)
    return np.nonzero(np.where(x._mask, False, x._data != type(x._data.dtype())(0)))
def ids(a):
    return (id(a), id(a))
def column_stack(tup):
    return _stack_with(tup, "col")
def dstack(tup):
    return _stack_with(tup, "d")
def _stack_with(tup, kind: Literal[str]):
    ds = [np.asarray(_ma1(t)._data) for t in tup]
    ms = [np.asarray(_ma1(t)._mask) for t in tup]
    if kind == "col":
        d = np.column_stack(ds)
        m = np.column_stack(ms)
    else:
        d = np.dstack(ds)
        m = np.dstack(ms)
    return MaskedArray(d, m, False, default_fill_value(d.dtype), False)
def hsplit(ary, indices_or_sections):
    x = _ma1(ary)
    ds = np.hsplit(x._data, indices_or_sections)
    ms = np.hsplit(x._mask, indices_or_sections)
    return [MaskedArray(ds[i], ms[i], False, x._fill, False) for i in range(len(ds))]
