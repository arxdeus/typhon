# Ported from scipy 1.18.1 (BSD-3-Clause)

# Batch support ("linalg_batch"): arrays with ndim > 2 are treated as stacks
# of matrices over the leading dimensions. Each public function keeps the
# 2-D fast path and otherwise applies the 2-D kernel per slice and assembles
# the results (scalars, arrays, and result pairs/triples of them).

import numpy as np
import internal.static as static

def _zeros_idx(shape):
    return tuple(0 for _ in static.range(static.len(shape)))
from ._util import _Pair, _Triple

def _is_pair(r) -> Literal[bool]:
    return isinstance(r, _Pair)

def _is_triple(r) -> Literal[bool]:
    return isinstance(r, _Triple)

def _alloc(r, bshape):
    if _is_pair(r):
        return _Pair(_alloc(r.a, bshape), _alloc(r.b, bshape))
    elif _is_triple(r):
        return _Triple(_alloc(r.a, bshape), _alloc(r.b, bshape), _alloc(r.c, bshape))
    elif isinstance(r, np.ndarray):
        return np.empty(bshape + r.shape, dtype=r.dtype)
    else:
        return np.empty(bshape, dtype=type(r))

def _store(out, idx, r):
    if _is_pair(r):
        _store(out.a, idx, r.a)
        _store(out.b, idx, r.b)
    elif _is_triple(r):
        _store(out.a, idx, r.a)
        _store(out.b, idx, r.b)
        _store(out.c, idx, r.c)
    elif isinstance(r, np.ndarray):
        out[idx] = r
    else:
        out[idx] = r

def _batch_shape(x, core: int):
    if core == 2:
        return x.shape[:-2]
    else:
        return x.shape[:-1]

def batch1(f, a):
    """Apply ``f`` (2-D kernel) over the leading dimensions of ``a``."""
    x = np.asarray(a)
    bshape = x.shape[:-2]
    size = 1
    for s in bshape:
        size *= s
    if size == 0:
        raise ValueError("empty batch is not supported")
    out = _alloc(f(np.ascontiguousarray(x[_zeros_idx(bshape)])), bshape)
    for idx in np.ndindex(bshape):
        _store(out, idx, f(np.ascontiguousarray(x[idx])))
    return out

def batch2(f, a, b, bcore2: Literal[bool] = True):
    """Like ``batch1`` for two operands with broadcast batch dimensions.
    ``b`` has 2 core dims (or 1 when ``bcore2`` is False)."""
    x = np.asarray(a)
    y = np.asarray(b)
    if bcore2:
        bs = np.broadcast_shapes(x.shape[:-2], y.shape[:-2])
        xb = np.broadcast_to(x, bs + x.shape[-2:])
        yb = np.broadcast_to(y, bs + y.shape[-2:])
    else:
        bs = np.broadcast_shapes(x.shape[:-2], y.shape[:-1])
        xb = np.broadcast_to(x, bs + x.shape[-2:])
        yb = np.broadcast_to(y, bs + y.shape[-1:])
    size = 1
    for s in bs:
        size *= s
    if size == 0:
        raise ValueError("empty batch is not supported")
    z = _zeros_idx(bs)
    out = _alloc(f(np.ascontiguousarray(xb[z]), np.ascontiguousarray(yb[z])), bs)
    for idx in np.ndindex(bs):
        _store(out, idx, f(np.ascontiguousarray(xb[idx]), np.ascontiguousarray(yb[idx])))
    return out
