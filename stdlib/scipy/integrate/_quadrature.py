# Ported from scipy 1.18.1 (BSD-3-Clause)
#
# Sampled-data integration rules: trapezoid, simpson, cumulative_trapezoid,
# cumulative_simpson, romb. Arrays are processed as rows along `axis`.
# 1-D input gives a float; N-D input gives an array (shape without `axis`).
# Deviations: romb `show` is ignored; cumulative_simpson `dx` must be scalar.

import numpy as np
from numpy.util import tuple_set, normalize_axis_index

def _as_float(x):
    a = np.asarray(x)
    if a.dtype is float:
        return a
    else:
        return a.astype(float)

def _to_rows(x, axis: int):
    ax = normalize_axis_index(axis, x.ndim)
    xm = np.ascontiguousarray(np.moveaxis(x, ax, -1))
    n = xm.shape[-1]
    return xm.reshape(-1, n), xm.shape, ax

def _lm1(t) -> int:
    return len(t) - 1

def _from_rows(y2, mshape, ax: int):
    n = y2.shape[1]
    sh = tuple_set(mshape, _lm1(mshape), n)
    y = y2.reshape(sh)
    return np.ascontiguousarray(np.moveaxis(y, -1, ax))

def _reduced(vals, mshape, ya):
    # vals: 1-D array of per-row results
    if ya.ndim == 1:
        return float(vals[0])
    else:
        return vals.reshape(mshape[:-1])

def _x_rows(x, ya, axis: int, nrows: int, n: int):
    # returns x as (nrows, n) float array, or raises
    xa = _as_float(x)
    if xa.ndim == 1:
        if xa.shape[0] != n:
            raise ValueError("If given, length of x along axis must be the same as y.")
        out = np.empty((nrows, n))
        for r in range(nrows):
            for i in range(n):
                out[r, i] = xa[i]
        return out
    else:
        if xa.ndim != ya.ndim:
            raise ValueError("If given, shape of x must be 1-D or the same as y.")
        x2, _m, _a = _to_rows(xa, axis)
        if x2.shape[1] != n or x2.shape[0] != nrows:
            raise ValueError("If given, length of x along axis must be the same as y.")
        return x2

def trapezoid(y, x=None, dx: float = 1.0, axis: int = -1):
    ya = _as_float(y)
    y2, msh, ax = _to_rows(ya, axis)
    m = y2.shape[0]
    n = y2.shape[1]
    out = np.zeros(m)
    if x is None:
        for r in range(m):
            s = 0.0
            for i in range(n - 1):
                s += dx * (y2[r, i + 1] + y2[r, i]) / 2.0
            out[r] = s
    else:
        x2 = _x_rows(x, ya, axis, m, n)
        for r in range(m):
            s = 0.0
            for i in range(n - 1):
                s += (x2[r, i + 1] - x2[r, i]) * (y2[r, i + 1] + y2[r, i]) / 2.0
            out[r] = s
    return _reduced(out, msh, ya)

def _div0(a: float, b: float) -> float:
    return a / b if b != 0.0 else 0.0

def _basic_simpson_row(y2, r: int, start: int, stop: int, xr, has_x: bool, dx: float) -> float:
    total = 0.0
    if not has_x:
        acc = 0.0
        for i in range(start, stop, 2):
            acc += y2[r, i] + 4.0 * y2[r, i + 1] + y2[r, i + 2]
        return dx / 3.0 * acc
    for i in range(start, stop, 2):
        h0 = xr[r, i + 1] - xr[r, i]
        h1 = xr[r, i + 2] - xr[r, i + 1]
        hsum = h0 + h1
        hprod = h0 * h1
        h0divh1 = _div0(h0, h1)
        tmp = hsum / 6.0 * (y2[r, i] * (2.0 - _div0(1.0, h0divh1))
                            + y2[r, i + 1] * (hsum * _div0(hsum, hprod))
                            + y2[r, i + 2] * (2.0 - h0divh1))
        total += tmp
    return total

def simpson(y, x=None, dx: float = 1.0, axis: int = -1):
    ya = _as_float(y)
    y2, msh, ax = _to_rows(ya, axis)
    m = y2.shape[0]
    N = y2.shape[1]
    has_x = x is not None
    xr = np.zeros((m, N))
    if x is not None:
        xr = _x_rows(x, ya, axis, m, N)
    out = np.zeros(m)
    for r in range(m):
        if N % 2 == 0:
            if N == 2:
                last_dx = dx
                if has_x:
                    last_dx = xr[r, 1] - xr[r, 0]
                out[r] = 0.5 * last_dx * (y2[r, 1] + y2[r, 0])
            else:
                res = _basic_simpson_row(y2, r, 0, N - 3, xr, has_x, dx)
                h0 = dx
                h1 = dx
                if has_x:
                    h0 = xr[r, N - 2] - xr[r, N - 3]
                    h1 = xr[r, N - 1] - xr[r, N - 2]
                alpha = _div0(2 * h1 ** 2 + 3 * h0 * h1, 6 * (h1 + h0))
                beta = _div0(h1 ** 2 + 3.0 * h0 * h1, 6 * h0)
                eta = _div0(1 * h1 ** 3, 6 * h0 * (h0 + h1))
                res += alpha * y2[r, N - 1] + beta * y2[r, N - 2] - eta * y2[r, N - 3]
                out[r] = res
        else:
            out[r] = _basic_simpson_row(y2, r, 0, N - 2, xr, has_x, dx)
    return _reduced(out, msh, ya)

def cumulative_trapezoid(y, x=None, dx: float = 1.0, axis: int = -1, initial=None):
    ya = _as_float(y)
    y2, msh, ax = _to_rows(ya, axis)
    m = y2.shape[0]
    n = y2.shape[1]
    xr = np.zeros((m, n))
    has_x = x is not None
    if x is not None:
        xr = _x_rows(x, ya, axis, m, n)
    off = 0
    init = 0.0
    if initial is not None:
        if float(initial) != 0.0:
            raise ValueError("`initial` must be `None` or `0`.")
        off = 1
        init = float(initial)
    nout = n - 1 + off if n >= 1 else off
    out = np.empty((m, nout))
    for r in range(m):
        s = 0.0
        if off == 1:
            out[r, 0] = init
        for i in range(n - 1):
            d = (xr[r, i + 1] - xr[r, i]) if has_x else dx
            s += d * (y2[r, i + 1] + y2[r, i]) / 2.0
            out[r, i + off] = s
    return _from_rows(out, msh, ax)

def _cumsimp_row(y2, r: int, d: List[float], equal: bool) -> List[float]:
    n = len(d) + 1
    def sub(yv: List[float], dv: List[float]) -> List[float]:
        res: List[float] = []
        for i in range(len(yv) - 2):
            f1 = yv[i]
            f2 = yv[i + 1]
            f3 = yv[i + 2]
            if equal:
                dd = dv[i]
                res.append(dd / 3 * (5 * f1 / 4 + 2 * f2 - f3 / 4))
            else:
                x21 = dv[i]
                x32 = dv[i + 1]
                x31 = x21 + x32
                x21_x31 = x21 / x31
                x21_x32 = x21 / x32
                x21x21_x31x32 = x21_x31 * x21_x32
                c1 = 3 - x21_x31
                c2 = 3 + x21x21_x31x32 + x21_x31
                c3 = -x21x21_x31x32
                res.append(x21 / 6 * (c1 * f1 + c2 * f2 + c3 * f3))
        return res
    yv = [y2[r, i] for i in range(n)]
    h1 = sub(yv, d)
    h2r = sub(yv[::-1], d[::-1])
    h2 = h2r[::-1]
    nsub = len(h1) + 1
    subs = [0.0] * nsub
    for i in range(0, nsub - 1, 2):
        subs[i] = h1[i]
    for i in range(1, nsub, 2):
        subs[i] = h2[i - 1]
    subs[nsub - 1] = h2[len(h2) - 1]
    out: List[float] = []
    s = 0.0
    for v in subs:
        s += v
        out.append(s)
    return out

def cumulative_simpson(y, x=None, dx: float = 1.0, axis: int = -1, initial=None):
    ya = _as_float(y)
    y2, msh, ax = _to_rows(ya, axis)
    m = y2.shape[0]
    n = y2.shape[1]
    if n < 3:
        res = cumulative_trapezoid(ya, x, dx=dx, axis=axis, initial=None)
        r2, _m, _a = _to_rows(_as_float(res), axis)
        return _finish_cs(r2, msh, ax, initial)
    xr = np.zeros((m, n))
    has_x = x is not None
    if x is not None:
        xa = _as_float(x)
        if not (xa.shape == ya.shape or (xa.ndim == 1 and xa.shape[0] == n)):
            raise ValueError("If given, shape of `x` must be the same as `y` or 1-D with "
                             "the same length as `y` along `axis`.")
        xr = _x_rows(xa, ya, axis, m, n)
    out = np.empty((m, n - 1))
    for r in range(m):
        d = [0.0] * (n - 1)
        for i in range(n - 1):
            if has_x:
                d[i] = xr[r, i + 1] - xr[r, i]
                if d[i] <= 0:
                    raise ValueError("Input x must be strictly increasing.")
            else:
                d[i] = dx
        row = _cumsimp_row(y2, r, d, not has_x)
        for i in range(n - 1):
            out[r, i] = row[i]
    return _finish_cs(out, msh, ax, initial)

def _finish_cs(out, msh, ax: int, initial):
    if initial is None:
        return _from_rows(out, msh, ax)
    init = float(initial)
    m = out.shape[0]
    k = out.shape[1]
    o2 = np.empty((m, k + 1))
    for r in range(m):
        o2[r, 0] = init
        for i in range(k):
            o2[r, i + 1] = out[r, i] + init
    return _from_rows(o2, msh, ax)

def romb(y, dx: float = 1.0, axis: int = -1, show: bool = False):
    ya = _as_float(y)
    y2, msh, ax = _to_rows(ya, axis)
    m = y2.shape[0]
    Nsamps = y2.shape[1]
    Ninterv = Nsamps - 1
    n = 1
    k = 0
    while n < Ninterv:
        n <<= 1
        k += 1
    if n != Ninterv:
        raise ValueError("Number of samples must be one plus a non-negative power of 2.")
    out = np.zeros(m)
    for r in range(m):
        R: Dict[Tuple[int, int], float] = {}
        h = Ninterv * dx
        R[(0, 0)] = (y2[r, 0] + y2[r, Nsamps - 1]) / 2.0 * h
        start = Ninterv
        stop = Ninterv
        step = Ninterv
        for i in range(1, k + 1):
            start >>= 1
            s = 0.0
            for j in range(start, stop, step):
                s += y2[r, j]
            step >>= 1
            R[(i, 0)] = 0.5 * (R[(i - 1, 0)] + h * s)
            for j in range(1, i + 1):
                prev = R[(i, j - 1)]
                R[(i, j)] = prev + (prev - R[(i - 1, j - 1)]) / ((1 << (2 * j)) - 1)
            h /= 2.0
        out[r] = R[(k, k)]
    return _reduced(out, msh, ya)
