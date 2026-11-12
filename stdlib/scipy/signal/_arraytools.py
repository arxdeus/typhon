# Ported from scipy 1.18.1 (BSD-3-Clause)

import numpy as np
from numpy.util import tuple_set, normalize_axis_index

def _as_float(x):
    a = np.asarray(x)
    if a.dtype is float:
        return a
    else:
        return a.astype(float)

def _to_rows(x, axis: int):
    # Move `axis` last and flatten the rest: returns (rows (M, N), moved shape, axis).
    ax = normalize_axis_index(axis, x.ndim)
    xm = np.ascontiguousarray(np.moveaxis(x, ax, -1))
    n = xm.shape[-1]
    return xm.reshape(-1, n), xm.shape, ax

def _from_rows(y2, mshape, ax: int):
    n = y2.shape[1]
    sh = tuple_set(mshape, static_len_minus_one(mshape), n)
    y = y2.reshape(sh)
    return np.ascontiguousarray(np.moveaxis(y, -1, ax))

def static_len_minus_one(t) -> int:
    return len(t) - 1

def axis_slice(a, start: Optional[int] = None, stop: Optional[int] = None,
               step: Optional[int] = None, axis: int = -1):
    x2, msh, ax = _to_rows(np.asarray(a), axis)
    n = x2.shape[1]
    idx = list(range(n))[slice(start, stop, step)]
    out = np.empty((x2.shape[0], len(idx)), dtype=x2.dtype)
    for r in range(x2.shape[0]):
        for j in range(len(idx)):
            out[r, j] = x2[r, idx[j]]
    return _from_rows(out, msh, ax)

def axis_reverse(a, axis: int = -1):
    return axis_slice(a, step=-1, axis=axis)

def _ext_rows(x2, n: int, kind: str):
    M = x2.shape[0]
    N = x2.shape[1]
    if n < 1:
        return x2
    if kind != 'constant' and n > N - 1:
        raise ValueError(f'The extension length n ({n}) is too big. It must not exceed x.shape[axis]-1, which is {N - 1}.')
    out = np.empty((M, N + 2 * n), dtype=x2.dtype)
    for r in range(M):
        l0 = x2[r, 0]
        r0 = x2[r, N - 1]
        for j in range(N):
            out[r, n + j] = x2[r, j]
        for i in range(n):
            # left position i corresponds to x[n - i]
            if kind == 'odd':
                out[r, i] = 2 * l0 - x2[r, n - i]
                out[r, n + N + i] = 2 * r0 - x2[r, N - 2 - i]
            elif kind == 'even':
                out[r, i] = x2[r, n - i]
                out[r, n + N + i] = x2[r, N - 2 - i]
            elif kind == 'constant':
                out[r, i] = l0
                out[r, n + N + i] = r0
            else:  # zeros
                out[r, i] = type(l0)()
                out[r, n + N + i] = type(l0)()
    return out

def _ext(x, n: int, axis: int, kind: str):
    x2, msh, ax = _to_rows(np.asarray(x), axis)
    return _from_rows(_ext_rows(x2, n, kind), msh, ax)

def odd_ext(x, n: int, axis: int = -1):
    return _ext(x, n, axis, 'odd')

def even_ext(x, n: int, axis: int = -1):
    return _ext(x, n, axis, 'even')

def const_ext(x, n: int, axis: int = -1):
    return _ext(x, n, axis, 'constant')

def zero_ext(x, n: int, axis: int = -1):
    return _ext(x, n, axis, 'zeros')
