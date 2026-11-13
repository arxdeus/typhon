# Ported from numpy 2.5.4 (BSD-3-Clause)
#
# Assorted numpy 2.x top-level functions: unique_* (array API), index
# helpers (s_, index_exp, ogrid, mgrid), binary_repr/base_repr, fix,
# float_power, divmod, bitwise_count, rollaxis, unstack, permute_dims,
# concat, cumulative_prod, vecdot/matvec/vecmat, histogram2d/histogramdd.

import util
import internal.static as static

from .ndarray import ndarray
from .routines import asarray, array, arange, linspace, concatenate, \
    moveaxis, transpose, empty, zeros, broadcast_arrays, swapaxes, atleast_1d
from .lib.arraysetops import unique
from .ndmath import cumprod, trunc, floor_divide, remainder, power
from .statistics import histogram_bin_edges, histogram

# ---- unique_* ----

@tuple
class UniqueAllResult[V, I, J, C]:
    values: V
    indices: I
    inverse_indices: J
    counts: C

@tuple
class UniqueCountsResult[V, C]:
    values: V
    counts: C

@tuple
class UniqueInverseResult[V, J]:
    values: V
    inverse_indices: J

def unique_values(x):
    xa = asarray(x)
    return unique(xa.ravel(), equal_nan=False)

def unique_counts(x):
    v, c = unique(asarray(x).ravel(), return_counts=True, equal_nan=False)
    return UniqueCountsResult(v, c)

def unique_inverse(x):
    xa = asarray(x)
    v, inv = unique(xa.ravel(), return_inverse=True, equal_nan=False)
    return UniqueInverseResult(v, inv.reshape(xa.shape))

def unique_all(x):
    xa = asarray(x)
    v, idx, inv, c = unique(xa.ravel(), return_index=True, return_inverse=True,
                            return_counts=True, equal_nan=False)
    return UniqueAllResult(v, idx, inv.reshape(xa.shape), c)

# ---- index helpers ----

@tuple
class SliceExpression:
    def __getitem__(self, item):
        return item

@tuple
class IndexExpression:
    def __getitem__(self, item):
        if isinstance(item, Tuple):
            return item
        return (item,)

s_ = SliceExpression()
index_exp = IndexExpression()

def _grid_len(s: Slice):
    if s.V is complex:
        return int(abs(s.step))
    start = s.start if s.start is not None else 0
    step = s.step if s.step is not None else 1
    return len(range(start, s.stop, step))

def _grid_axis(s: Slice):
    if s.V is complex:
        start = float(s.start) if s.start is not None else 0.0
        return linspace(start, float(s.stop), num=int(abs(s.step)))
    start = s.start if s.start is not None else 0
    step = s.step if s.step is not None else 1
    return arange(start, s.stop, step)

def _shape_at(i: Literal[int], n: Literal[int], m: int):
    t = (1,) * n
    p = Ptr[int](__ptr__(t).as_byte())
    p[i] = m
    return t

@tuple
class OGridClass:
    def __getitem__(self, key):
        if isinstance(key, Slice):
            return _grid_axis(key)
        n: Literal[int] = static.len(key)
        return tuple(_grid_axis(key[i]).reshape(_shape_at(i, n, -1)) for i in static.range(n))

@tuple
class MGridClass:
    def __getitem__(self, key):
        if isinstance(key, Slice):
            return _grid_axis(key)
        n: Literal[int] = static.len(key)
        parts = tuple(_grid_axis(key[i]).reshape(_shape_at(i, n, -1)) for i in static.range(n))
        bs = broadcast_arrays(*parts)
        first = bs[0]
        out = empty((n, *first.shape), dtype=first.dtype)
        for i in static.range(n):
            out[i] = bs[i]
        return out

ogrid = OGridClass()
mgrid = MGridClass()

# ---- number formatting ----

def binary_repr(num: int, width: Optional[int] = None) -> str:
    if num == 0:
        return '0' * (width if width is not None and width > 0 else 1)
    if num > 0:
        b = bin(num)[2:]
        if width is not None and len(b) < width:
            b = '0' * (width - len(b)) + b
        return b
    if width is None:
        return '-' + bin(-num)[2:]
    pl = len(bin(-num)[2:])
    w = width
    t = bin((1 << w) + num)[2:] if (1 << w) + num >= 0 else bin(-num)[2:]
    if len(t) < w:
        t = '0' * (w - len(t)) + t
    if pl > w:
        t = bin((1 << (pl + 1)) + num)[2:]
    return t

def base_repr(number: int, base: int = 2, padding: int = 0) -> str:
    digits = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    if base > len(digits):
        raise ValueError("Bases greater than 36 not handled in base_repr.")
    elif base < 2:
        raise ValueError("Bases less than 2 not handled in base_repr.")
    num = abs(number)
    res = ''
    while num:
        res += digits[num % base]
        num //= base
    if padding:
        res += '0' * padding
    if number < 0:
        res += '-'
    if res == '':
        res = '0'
    return res[::-1]

# ---- elementwise ----

def fix(x):
    return trunc(x)

def float_power(x1, x2):
    return power(asarray(x1).astype(float), asarray(x2).astype(float))

def divmod(x1, x2):
    return (floor_divide(x1, x2), remainder(x1, x2))

def _popcount_val(v):
    if isinstance(v, bool):
        return u8(1) if v else u8(0)
    elif isinstance(v, int):
        return u8((v if v >= 0 else -v).popcnt())
    else:
        a = int(v)
        return u8((a if a >= 0 else -a).popcnt())

def bitwise_count(x):
    xa = asarray(x)
    out = empty(xa.shape, dtype=u8)
    fl = xa.ravel()
    for i in range(fl.size):
        out.data[i] = _popcount_val(fl[i])
    return out

# ---- shape manipulation ----

def rollaxis(a, axis: int, start: int = 0):
    xa = asarray(a)
    n = xa.ndim
    ax = util.normalize_axis_index(axis, n)
    st = start
    if st < 0:
        st += n
    if not (0 <= st < n + 1):
        raise util.AxisError("'start' arg requires " + str(-n) + " <= start < " + str(n + 1) + ", but " + str(start) + " was passed in")
    if ax < st:
        st -= 1
    if ax == st:
        return xa
    return moveaxis(xa, ax, st)

def permute_dims(a, axes = None):
    return transpose(a, axes)

def concat(arrays, axis = 0):
    return concatenate(arrays, axis=axis)

def unstack(x, axis: int = 0):
    xa = asarray(x)
    m = moveaxis(xa, axis, 0)
    return [m[i] for i in range(m.shape[0])]

def cumulative_prod(x, axis = None, dtype: type = NoneType, include_initial: bool = False):
    xa = asarray(x)
    if axis is None:
        if xa.ndim != 1:
            raise ValueError("axis must be specified for multi-dimensional arrays")
        ax = 0
    else:
        ax = axis
    c = cumprod(xa, axis=ax, dtype=dtype)
    if include_initial:
        ax2 = util.normalize_axis_index(ax, xa.ndim)
        oshape = util.tuple_set(c.shape, ax2, 1)
        o = empty(oshape, dtype=c.dtype)
        o.fill(c.dtype(1))
        c = concatenate((o, c), axis=ax2)
    return c

# ---- vector products (gufuncs) ----

def vecdot(x1, x2, axis: int = -1):
    a = asarray(x1)
    b = asarray(x2)
    if a.dtype is complex or a.dtype is complex64:
        a = a.conj()
    a, b = broadcast_arrays(a, b)
    am = moveaxis(a, axis, -1)
    bm = moveaxis(b, axis, -1)
    return (am * bm).sum(axis=-1)

def matvec(x1, x2):
    a = asarray(x1)
    b = asarray(x2)
    bb = b.reshape((*b.shape[:-1], 1, b.shape[-1]))
    return (a * bb).sum(axis=-1)

def vecmat(x1, x2):
    a = asarray(x1)
    b = asarray(x2)
    if a.dtype is complex or a.dtype is complex64:
        a = a.conj()
    aa = a.reshape((*a.shape[:-1], a.shape[-1], 1))
    return (aa * b).sum(axis=-2)

# ---- histograms ----

def _hist_bin(e, v: float) -> int:
    # index of bin in [0, m-2], -1 if outside; last edge inclusive
    m = e.size
    if v < e[0] or v > e[m - 1]:
        return -1
    if v == e[m - 1]:
        return m - 2
    return int(e.searchsorted(v, side='right')) - 1

def histogram2d(x, y, bins = 10, range = None, density: bool = False, weights = None):
    xa = asarray(x).astype(float).ravel()
    ya = asarray(y).astype(float).ravel()
    if isinstance(bins, int):
        bx = bins
        by = bins
    else:
        bx = bins[0]
        by = bins[1]
    if range is None:
        ex = histogram_bin_edges(xa, bx)
        ey = histogram_bin_edges(ya, by)
    else:
        ex = histogram_bin_edges(xa, bx, range=range[0])
        ey = histogram_bin_edges(ya, by, range=range[1])
    nx = ex.size - 1
    ny = ey.size - 1
    h = zeros((nx, ny), dtype=float)
    for i in _brange(xa.size):
        p = _hist_bin(ex, xa[i])
        q = _hist_bin(ey, ya[i])
        if p >= 0 and q >= 0:
            if weights is None:
                h[p, q] += 1.0
            else:
                h[p, q] += float(asarray(weights).ravel()[i])
    if density:
        tot = h.sum()
        for p in _brange(nx):
            for q in _brange(ny):
                h[p, q] = h[p, q] / tot / ((ex[p + 1] - ex[p]) * (ey[q + 1] - ey[q]))
    return h, ex, ey

def _brange(n: int):
    return range(n)

def histogramdd(sample, bins = 10, range = None, density: bool = False, weights = None):
    # 2-column samples (the common case); returns (hist, [edges...])
    s = asarray(sample)
    h, ex, ey = histogram2d(s[:, 0], s[:, 1], bins, range, density, weights)
    return h, [ex, ey]
