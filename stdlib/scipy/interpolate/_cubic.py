# Ported from scipy 1.18.1 (BSD-3-Clause)
#
# Cubic interpolators.  Each constructor returns a PPoly (the scipy classes
# are PPoly subclasses; here they are factory functions returning PPoly).

import numpy as np
import internal.static as static
from _common import _band_solve, _vals, _ext_mode
from _ppoly import PPoly

def _prepare(x, y, axis: int):
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    if static.len(xa.shape) != 1:
        compile_error("`x` must be 1-dimensional.")
    if static.len(ya.shape) == 0:
        compile_error("`y` must be at least 1-dimensional.")
    ax = axis % ya.ndim
    if xa.size < 2:
        raise ValueError("`x` must contain at least 2 elements.")
    if xa.size != ya.shape[ax]:
        raise ValueError(f"The length of `y` along `axis`={ax} doesn't match the length of `x`")
    for i in range(xa.size):
        if not np.isfinite(xa[i]):
            raise ValueError("`x` must contain only finite values.")
    for v in ya.flat:
        if not np.isfinite(v):
            raise ValueError("`y` must contain only finite values.")
    for i in range(xa.size - 1):
        if xa[i + 1] - xa[i] <= 0.0:
            raise ValueError("`x` must be strictly increasing sequence.")
    ym = np.moveaxis(ya, ax, 0)
    tsh = ym.shape[1:]
    y2 = np.ascontiguousarray(ym).reshape((xa.size, -1)).copy()
    return xa, y2, tsh, ax

def _hermite(xa, y2, s2, tsh, ax: int, ext: int):
    n = xa.size
    R = y2.shape[1]
    c = np.empty((4, n - 1, R))
    for i in range(n - 1):
        dx = xa[i + 1] - xa[i]
        for j in range(R):
            slope = (y2[i + 1, j] - y2[i, j]) / dx
            t = (s2[i, j] + s2[i + 1, j] - 2.0 * slope) / dx
            c[0, i, j] = t / dx
            c[1, i, j] = (slope - s2[i, j]) / dx - t
            c[2, i, j] = s2[i, j]
            c[3, i, j] = y2[i, j]
    p = PPoly(c.reshape((4, n - 1) + tsh), xa, _ext=ext)
    p.axis = ax
    return p

def CubicHermiteSpline(x, y, dydx, axis: int = 0, extrapolate = None):
    xa, y2, tsh, ax = _prepare(x, y, axis)
    d = np.asarray(dydx, dtype=float)
    if d.shape != np.asarray(y).shape:
        raise ValueError("The shapes of `y` and `dydx` must be identical.")
    d2 = np.ascontiguousarray(np.moveaxis(d, ax, 0)).reshape((xa.size, -1)).copy()
    return _hermite(xa, y2, d2, tsh, ax, _ext_mode(extrapolate, 1))

def _sign(v: float) -> float:
    return 1.0 if v > 0.0 else (-1.0 if v < 0.0 else 0.0)

def _pchip_edge(h0: float, h1: float, m0: float, m1: float) -> float:
    d = ((2.0 * h0 + h1) * m0 - h0 * m1) / (h0 + h1)
    if _sign(d) != _sign(m0):
        return 0.0
    if _sign(m0) != _sign(m1) and abs(d) > 3.0 * abs(m0):
        return 3.0 * m0
    return d

def PchipInterpolator(x, y, axis: int = 0, extrapolate = None):
    xa, y2, tsh, ax = _prepare(x, y, axis)
    n = xa.size
    R = y2.shape[1]
    dk = np.zeros((n, R))
    for j in range(R):
        hk = [xa[i + 1] - xa[i] for i in range(n - 1)]
        mk = [(y2[i + 1, j] - y2[i, j]) / hk[i] for i in range(n - 1)]
        if n == 2:
            dk[0, j] = mk[0]
            dk[1, j] = mk[0]
            continue
        for i in range(n - 2):
            if _sign(mk[i + 1]) != _sign(mk[i]) or mk[i + 1] == 0.0 or mk[i] == 0.0:
                dk[i + 1, j] = 0.0
            else:
                w1 = 2.0 * hk[i + 1] + hk[i]
                w2 = hk[i + 1] + 2.0 * hk[i]
                whmean = (w1 / mk[i] + w2 / mk[i + 1]) / (w1 + w2)
                dk[i + 1, j] = 1.0 / whmean
        dk[0, j] = _pchip_edge(hk[0], hk[1], mk[0], mk[1])
        dk[n - 1, j] = _pchip_edge(hk[n - 2], hk[n - 3], mk[n - 2], mk[n - 3])
    return _hermite(xa, y2, dk, tsh, ax, _ext_mode(extrapolate, 1))

def pchip_interpolate(xi, yi, x, der: int = 0, axis: int = 0):
    P = PchipInterpolator(xi, yi, axis=axis)
    if der == 0:
        return P(x)
    return P.derivative(der)(x)

def Akima1DInterpolator(x, y, axis: int = 0, method: str = "akima", extrapolate = None):
    if method != "akima" and method != "makima":
        raise NotImplementedError(f"`method`={method} is unsupported.")
    xa, y2, tsh, ax = _prepare(x, y, axis)
    n = xa.size
    R = y2.shape[1]
    t = np.zeros((n, R))
    if n == 2:
        for j in range(R):
            mk = (y2[1, j] - y2[0, j]) / (xa[1] - xa[0])
            t[0, j] = mk
            t[1, j] = mk
    else:
        m = np.empty((n + 3, R))
        for j in range(R):
            for i in range(n - 1):
                m[i + 2, j] = (y2[i + 1, j] - y2[i, j]) / (xa[i + 1] - xa[i])
            m[1, j] = 2.0 * m[2, j] - m[3, j]
            m[0, j] = 2.0 * m[1, j] - m[2, j]
            m[n + 1, j] = 2.0 * m[n, j] - m[n - 1, j]
            m[n + 2, j] = 2.0 * m[n + 1, j] - m[n, j]
        f1 = np.empty((n, R))
        f2 = np.empty((n, R))
        mmax = -float('inf')
        for i in range(n):
            for j in range(R):
                t[i, j] = 0.5 * (m[i + 3, j] + m[i, j])
                a1 = abs(m[i + 3, j] - m[i + 2, j])
                a2 = abs(m[i + 1, j] - m[i, j])
                if method == "makima":
                    a1 += 0.5 * abs(m[i + 3, j] + m[i + 2, j])
                    a2 += 0.5 * abs(m[i + 1, j] + m[i, j])
                f1[i, j] = a1
                f2[i, j] = a2
                if a1 + a2 > mmax:
                    mmax = a1 + a2
        for i in range(n):
            for j in range(R):
                f12 = f1[i, j] + f2[i, j]
                if f12 > 1e-9 * mmax:
                    t[i, j] = m[i + 1, j] + (f2[i, j] / f12) * (m[i + 2, j] - m[i + 1, j])
    return _hermite(xa, y2, t, tsh, ax, _ext_mode(extrapolate, 0))

def _solve_tri(A, b):
    # A: (3, n) scipy banded (upper, diag, lower); b: (n, R)
    n = A.shape[1]
    ab = np.zeros((n, 4))
    for i in range(n):
        ab[i, 1] = A[1, i]
        if i + 1 < n:
            ab[i, 2] = A[0, i + 1]
        if i > 0:
            ab[i, 0] = A[2, i - 1]
    return _band_solve(ab, n, 1, 1, b)

def _bc_parse(bc, R: int):
    # returns (kind, values) where kind: 0 not-a-knot, 1/2 derivative order, 3 periodic
    if isinstance(bc, str):
        if bc == 'clamped':
            return 1, [0.0] * R
        elif bc == 'natural':
            return 2, [0.0] * R
        elif bc == 'not-a-knot':
            return 0, [0.0] * R
        elif bc == 'periodic':
            return 3, [0.0] * R
        raise ValueError(f"bc_type={bc} is not allowed.")
    else:
        order = int(bc[0])
        if order != 1 and order != 2:
            raise ValueError("The specified derivative order must be 1 or 2.")
        return order, _vals(bc[1], R)

def CubicSpline(x, y, axis: int = 0, bc_type = 'not-a-knot', extrapolate = None):
    xa, y2, tsh, ax = _prepare(x, y, axis)
    n = xa.size
    R = y2.shape[1]
    if isinstance(bc_type, str):
        if bc_type == 'periodic':
            for j in range(R):
                if abs(y2[0, j] - y2[n - 1, j]) > 1e-15 * (1.0 + abs(y2[n - 1, j])):
                    raise ValueError(f"The first and last `y` point along axis {ax} must be identical (within machine precision) when bc_type='periodic'.")
        b0, v0 = _bc_parse(bc_type, R)
        b1, v1 = _bc_parse(bc_type, R)
    else:
        if len(bc_type) != 2:
            raise ValueError("`bc_type` must contain 2 elements to specify start and end conditions.")
        b0, v0 = _bc_parse(bc_type[0], R)
        b1, v1 = _bc_parse(bc_type[1], R)
        if b0 == 3 or b1 == 3:
            raise ValueError("'periodic' `bc_type` is defined for both curve ends and cannot be used with other boundary conditions.")
    ext = _ext_mode(extrapolate, 2 if b0 == 3 else 1)
    dx = [xa[i + 1] - xa[i] for i in range(n - 1)]
    slope = np.empty((n - 1, R))
    for i in range(n - 1):
        for j in range(R):
            slope[i, j] = (y2[i + 1, j] - y2[i, j]) / dx[i]
    s = np.empty((n, R))
    if n == 2:
        if b0 == 0 or b0 == 3:
            b0 = 1
            v0 = [slope[0, j] for j in range(R)]
        if b1 == 0 or b1 == 3:
            b1 = 1
            v1 = [slope[0, j] for j in range(R)]
    if n == 3 and b0 == 0 and b1 == 0:
        A = np.zeros((3, 3))
        A[0, 0] = 1.0
        A[0, 1] = 1.0
        A[1, 0] = dx[1]
        A[1, 1] = 2.0 * (dx[0] + dx[1])
        A[1, 2] = dx[0]
        A[2, 1] = 1.0
        A[2, 2] = 1.0
        b = np.empty((3, R))
        for j in range(R):
            b[0, j] = 2.0 * slope[0, j]
            b[1, j] = 3.0 * (dx[0] * slope[1, j] + dx[1] * slope[0, j])
            b[2, j] = 2.0 * slope[1, j]
        s = np.linalg.solve(A, b)
    elif n == 3 and b0 == 3:
        for j in range(R):
            num = slope[0, j] / dx[0] + slope[1, j] / dx[1]
            den = 1.0 / dx[0] + 1.0 / dx[1]
            for i in range(3):
                s[i, j] = num / den
    elif b0 == 3:
        s = _periodic_slopes(xa, dx, slope, n, R)
    else:
        A = np.zeros((3, n))
        b = np.empty((n, R))
        for i in range(1, n - 1):
            A[1, i] = 2.0 * (dx[i - 1] + dx[i])
            A[0, i + 1] = dx[i - 1]
            A[2, i - 1] = dx[i]
            for j in range(R):
                b[i, j] = 3.0 * (dx[i] * slope[i - 1, j] + dx[i - 1] * slope[i, j])
        if b0 == 0:
            A[1, 0] = dx[1]
            d = xa[2] - xa[0]
            A[0, 1] = d
            for j in range(R):
                b[0, j] = ((dx[0] + 2.0 * d) * dx[1] * slope[0, j] + dx[0] * dx[0] * slope[1, j]) / d
        elif b0 == 1:
            A[1, 0] = 1.0
            A[0, 1] = 0.0
            for j in range(R):
                b[0, j] = v0[j]
        else:
            A[1, 0] = 2.0 * dx[0]
            A[0, 1] = dx[0]
            for j in range(R):
                b[0, j] = -0.5 * v0[j] * dx[0] * dx[0] + 3.0 * (y2[1, j] - y2[0, j])
        if b1 == 0:
            A[1, n - 1] = dx[n - 3]
            d = xa[n - 1] - xa[n - 3]
            A[2, n - 2] = d
            for j in range(R):
                b[n - 1, j] = (dx[n - 2] * dx[n - 2] * slope[n - 3, j] + (2.0 * d + dx[n - 2]) * dx[n - 3] * slope[n - 2, j]) / d
        elif b1 == 1:
            A[1, n - 1] = 1.0
            A[2, n - 2] = 0.0
            for j in range(R):
                b[n - 1, j] = v1[j]
        else:
            A[1, n - 1] = 2.0 * dx[n - 2]
            A[2, n - 2] = dx[n - 2]
            for j in range(R):
                b[n - 1, j] = 0.5 * v1[j] * dx[n - 2] * dx[n - 2] + 3.0 * (y2[n - 1, j] - y2[n - 2, j])
        s = _solve_tri(A, b)
    return _hermite(xa, y2, s, tsh, ax, ext)

def _periodic_slopes(xa, dx: List[float], slope, n: int, R: int):
    # Sherman-Morrison style solve of the cyclic system (as in scipy)
    m = n - 1
    A = np.zeros((3, m))
    b = np.empty((m, R))
    for i in range(1, n - 1):
        A[1, i] = 2.0 * (dx[i - 1] + dx[i])
        if i + 1 < m:
            A[0, i + 1] = dx[i - 1]
        A[2, i - 1] = dx[i]
        if i < m:
            for j in range(R):
                b[i, j] = 3.0 * (dx[i] * slope[i - 1, j] + dx[i - 1] * slope[i, j])
    A[1, 0] = 2.0 * (dx[n - 2] + dx[0])
    A[0, 1] = dx[n - 2]
    a_m1_0 = dx[n - 3]
    a_m1_m2 = dx[n - 2]
    a_m1_m1 = 2.0 * (dx[n - 2] + dx[n - 3])
    a_m2_m1 = dx[n - 4] if n >= 4 else dx[n - 3]
    a_0_m1 = dx[0]
    for j in range(R):
        b[0, j] = 3.0 * (dx[0] * slope[n - 2, j] + dx[n - 2] * slope[0, j])
        b[m - 1, j] = 3.0 * (dx[n - 2] * slope[n - 3, j] + dx[n - 3] * slope[n - 2, j])
    mm = m - 1
    Ac = np.zeros((3, mm))
    for r in range(3):
        for i in range(mm):
            Ac[r, i] = A[r, i]
    b1 = np.empty((mm, R))
    b2 = np.zeros((mm, R))
    for i in range(mm):
        for j in range(R):
            b1[i, j] = b[i, j]
    for j in range(R):
        b2[0, j] = -a_0_m1
        b2[mm - 1, j] = -a_m2_m1
    s1 = _solve_tri(Ac, b1)
    s2 = _solve_tri(Ac, b2)
    s = np.empty((n, R))
    for j in range(R):
        s_m1 = (b[m - 1, j] - a_m1_0 * s1[0, j] - a_m1_m2 * s1[mm - 1, j]) / \
               (a_m1_m1 + a_m1_0 * s2[0, j] + a_m1_m2 * s2[mm - 1, j])
        for i in range(mm):
            s[i, j] = s1[i, j] + s_m1 * s2[i, j]
        s[n - 2, j] = s_m1
        s[n - 1, j] = s[0, j]
    return s
