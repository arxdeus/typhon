# Ported from scipy 1.18.1 (BSD-3-Clause)

# Spline filters: gauss_spline, cspline1d/qspline1d (+ _eval), cspline2d,
# qspline2d, spline_filter, sepfir2d, symiirorder1, symiirorder2.
# Deviations: real float64 data only (float32 input is processed in double,
# complex is not supported); symiirorder1's c0/z1 and symiirorder2's r/omega
# are real scalars; symiirorder1/2 accept 1-D or 2-D input along the last
# axis.

import math
import numpy as np

def _hc(k: int, cs: float, r: float, omega: float) -> float:
    if k < 0:
        return 0.0
    if omega == 0.0:
        return cs * math.pow(r, float(k)) * (k + 1)
    elif omega == math.pi:
        return cs * math.pow(r, float(k)) * (k + 1) * (1 - 2 * (k % 2))
    return cs * math.pow(r, float(k)) * math.sin(omega * (k + 1)) / math.sin(omega)

def _hs(k0: int, cs: float, rsq: float, omega: float) -> float:
    cssq = cs * cs
    k = abs(k0)
    rsupk = math.pow(rsq, float(k) / 2.0)
    if omega == 0.0:
        c0 = (1 + rsq) / ((1 - rsq) * (1 - rsq) * (1 - rsq)) * cssq
        gamma = (1 - rsq) / (1 + rsq)
        return c0 * rsupk * (1 + gamma * k)
    if omega == math.pi:
        c0 = (1 + rsq) / ((1 - rsq) * (1 - rsq) * (1 - rsq)) * cssq
        gamma = (1 - rsq) / (1 + rsq) * (1 - 2 * (k % 2))
        return c0 * rsupk * (1 + gamma * k)
    c0 = cssq * (1.0 + rsq) / (1.0 - rsq) / (1 - 2 * rsq * math.cos(2 * omega) + rsq * rsq)
    gamma = (1.0 - rsq) / (1.0 + rsq) / math.tan(omega)
    return c0 * rsupk * (math.cos(omega * k) + gamma * math.sin(omega * k))

def _prec(precision: float) -> float:
    if precision <= 0.0 or precision > 1.0:
        return 1e-11
    return precision

def _iir1_ic(x: np.ndarray[float, 1], z1: float, precision: float) -> float:
    N = x.size
    if abs(z1) >= 1.0:
        raise ValueError('|z1| must be less than 1.0')
    y0 = x[0]
    powz1 = 1.0
    k = 0
    prec = precision * precision
    while True:
        powz1 *= z1
        y0 += powz1 * x[k]
        err = powz1 * powz1
        k += 1
        if not (err > prec and k < N):
            break
    if k >= N:
        raise ValueError('Sum to find symmetric boundary conditions did not converge.')
    return y0

def _sym1_row(x: np.ndarray[float, 1], c0: float, z1: float, precision: float) -> np.ndarray[float, 1]:
    N = x.size
    y = np.empty(N)
    y[0] = _iir1_ic(x, z1, precision)
    for n in range(1, N):
        y[n] = x[n] + z1 * y[n - 1]
    out = np.empty(N)
    out[N - 1] = -c0 / (z1 - 1.0) * y[N - 1]
    for n in range(N - 2, -1, -1):
        out[n] = c0 * y[n] + z1 * out[n + 1]
    return out

def _iir2_fwd_ic(x: np.ndarray[float, 1], r: float, omega: float, precision: float):
    N = x.size
    cs = 1 - 2 * r * math.cos(omega) + r * r
    y0 = _hc(0, cs, r, omega) * x[0]
    k = 0
    prec = precision * precision
    while True:
        diff = _hc(k + 1, cs, r, omega)
        y0 += diff * x[k]
        err = diff * diff
        k += 1
        if not (err > prec and k < N):
            break
    if k >= N:
        raise ValueError('Sum to find symmetric boundary conditions did not converge.')
    y1 = _hc(0, cs, r, omega) * x[1] + _hc(1, cs, r, omega) * x[0]
    k = 0
    while True:
        diff = _hc(k + 2, cs, r, omega)
        y1 += diff * x[k]
        err = diff * diff
        k += 1
        if not (err > prec and k < N):
            break
    if k >= N:
        raise ValueError('Sum to find symmetric boundary conditions did not converge.')
    return y0, y1

def _iir2_bwd_ic(x: np.ndarray[float, 1], r: float, omega: float, precision: float):
    N = x.size
    rsq = r * r
    cs = 1 - 2 * r * math.cos(omega) + rsq
    b0 = 0.0
    b1 = 0.0
    k = 0
    while True:
        diff = _hs(k, cs, rsq, omega) + _hs(k + 1, cs, rsq, omega)
        b0 += diff * x[N - 1 - k]
        err = diff * diff
        k += 1
        if not (err > precision and k < N):
            break
    if k >= N:
        raise ValueError('Sum to find symmetric boundary conditions did not converge.')
    k = 0
    while True:
        diff = _hs(k - 1, cs, rsq, omega) + _hs(k + 2, cs, rsq, omega)
        b1 += diff * x[N - 1 - k]
        err = diff * diff
        k += 1
        if not (err > precision and k < N):
            break
    if k >= N:
        raise ValueError('Sum to find symmetric boundary conditions did not converge.')
    return b0, b1

def _sym2_row(x: np.ndarray[float, 1], r: float, omega: float, precision: float) -> np.ndarray[float, 1]:
    N = x.size
    rsq = r * r
    a2 = 2 * r * math.cos(omega)
    a3 = -rsq
    cs = 1 - 2 * r * math.cos(omega) + rsq
    yf = np.empty(N)
    y0, y1 = _iir2_fwd_ic(x, r, omega, precision)
    yf[0] = y0
    yf[1] = y1
    for n in range(2, N):
        yf[n] = cs * x[n] + a2 * yf[n - 1] + a3 * yf[n - 2]
    b0, b1 = _iir2_bwd_ic(x, r, omega, precision)
    out = np.empty(N)
    out[N - 1] = b0
    out[N - 2] = b1
    for t in range(2, N):
        out[N - 1 - t] = cs * yf[N - 1 - t] + a2 * out[N - t] + a3 * out[N - t + 1]
    return out

def _rows2d(x) -> np.ndarray[float, 2]:
    a = np.asarray(x, float)
    if a.ndim == 1:
        return a.reshape((1, a.size)).copy()
    return np.ascontiguousarray(a)

def symiirorder1(signal, c0: float, z1: float, precision: float = -1.0):
    a = np.asarray(signal, float)
    if abs(z1) >= 1:
        raise ValueError('|z1| must be less than 1.0')
    if a.ndim > 2:
        raise ValueError('Input must be 1D or 2D')
    s = _rows2d(a)
    out = np.empty(s.shape)
    p = _prec(precision)
    for i in range(s.shape[0]):
        o = _sym1_row(np.ascontiguousarray(s[i]), c0, z1, p)
        for j in range(o.size):
            out[i, j] = o[j]
    if a.ndim == 1:
        return out[0]
    return out

def symiirorder2(input, r: float, omega: float, precision: float = -1.0):
    a = np.asarray(input, float)
    if r >= 1.0:
        raise ValueError('r must be less than 1.0')
    if a.ndim > 2:
        raise ValueError('Input must be 1D or 2D')
    s = _rows2d(a)
    out = np.empty(s.shape)
    p = _prec(precision)
    for i in range(s.shape[0]):
        o = _sym2_row(np.ascontiguousarray(s[i]), r, omega, p)
        for j in range(o.size):
            out[i, j] = o[j]
    if a.ndim == 1:
        return out[0]
    return out

def _along(a: np.ndarray[float, 2], axis0: bool, kind: int, p1: float, p2: float,
           precision: float) -> np.ndarray[float, 2]:
    # apply a symiirorder row filter along axis 0 (columns) or axis 1 (rows)
    M = a.shape[0]
    N = a.shape[1]
    out = np.empty((M, N))
    if axis0:
        for j in range(N):
            col = np.ascontiguousarray(a[:, j])
            o = _sym1_row(col, p1, p2, precision) if kind == 1 else _sym2_row(col, p1, p2, precision)
            for i in range(M):
                out[i, j] = o[i]
    else:
        for i in range(M):
            row = np.ascontiguousarray(a[i])
            o = _sym1_row(row, p1, p2, precision) if kind == 1 else _sym2_row(row, p1, p2, precision)
            for j in range(N):
                out[i, j] = o[j]
    return out

def _compute_root_from_lambda(lamb: float):
    tmp = math.sqrt(3 + 144 * lamb)
    xi = 1 - 96 * lamb + 24 * lamb * tmp
    omega = math.atan(math.sqrt((144 * lamb - 1.0) / xi))
    tmp2 = math.sqrt(xi)
    r = ((24 * lamb - 1 - tmp2) / (24 * lamb) * math.sqrt(48 * lamb + 24 * lamb * tmp) / tmp2)
    return r, omega

def cspline2d(signal, lamb: float = 0.0, precision: float = -1.0):
    a = np.asarray(signal, float)
    if a.ndim != 2:
        raise ValueError('Input must be 2D')
    p = precision
    if p < 0.0 or p >= 1.0:
        p = 1e-6
    if lamb <= 1 / 144.0:
        r = -2 + math.sqrt(3.0)
        o = _along(np.ascontiguousarray(a), False, 1, -r * 6.0, r, p)
        return _along(o, True, 1, -r * 6.0, r, p)
    r, omega = _compute_root_from_lambda(lamb)
    o = _along(np.ascontiguousarray(a), False, 2, r, omega, p)
    return _along(o, True, 2, r, omega, p)

def qspline2d(signal, lamb: float = 0.0, precision: float = -1.0):
    a = np.asarray(signal, float)
    if a.ndim != 2:
        raise ValueError('Input must be 2D')
    p = precision
    if p < 0.0 or p >= 1.0:
        p = 1e-6
    if lamb > 0:
        raise ValueError('lambda must be negative or zero')
    r = -3 + 2 * math.sqrt(2.0)
    o = _along(np.ascontiguousarray(a), False, 1, -r * 8.0, r, p)
    return _along(o, True, 1, -r * 8.0, r, p)

# ------------------------------------------------------------- sepfir2d

def _refl(i: int, N: int) -> int:
    if i < 0:
        return -i - 1
    if i >= N:
        return 2 * N - 1 - i
    return i

def _fir_mirror(x: np.ndarray[float, 1], h: np.ndarray[float, 1]) -> np.ndarray[float, 1]:
    N = x.size
    Nh = h.size
    N2 = Nh >> 1
    out = np.empty(N)
    for n in range(N):
        s = 0.0
        for j in range(Nh):
            s += h[j] * x[_refl(n + N2 - j, N)]
        out[n] = s
    return out

def sepfir2d(input, hrow, hcol):
    a = np.asarray(input, float)
    if a.ndim != 2:
        raise ValueError('Input must be 2D')
    hr = np.asarray(hrow, float).flatten()
    hc = np.asarray(hcol, float).flatten()
    M = a.shape[0]
    N = a.shape[1]
    tmp = np.empty((M, N))
    for m in range(M):
        o = _fir_mirror(np.ascontiguousarray(a[m]), hr)
        for n in range(N):
            tmp[m, n] = o[n]
    out = np.empty((M, N))
    for n in range(N):
        o = _fir_mirror(np.ascontiguousarray(tmp[:, n]), hc)
        for m in range(M):
            out[m, n] = o[m]
    return out

def spline_filter(Iin, lmbda: float = 5.0):
    a = np.asarray(Iin, float)
    h0 = float(np.float32(1.0 / 6.0))
    h1 = float(np.float32(4.0 / 6.0))
    hcol = np.array([h0, h1, h0])
    ck = cspline2d(a, lmbda)
    return sepfir2d(ck, hcol, hcol)

# --------------------------------------------------------------- 1-D

def gauss_spline(x, n):
    xa = np.asarray(x, float)
    signsq = (n + 1) / 12.0
    return 1 / math.sqrt(2 * math.pi * signsq) * np.exp(-xa ** 2 / 2 / signsq)

def _coeff_smooth(lam: float):
    xi = 1 - 96 * lam + 24 * lam * math.sqrt(3 + 144 * lam)
    omeg = math.atan2(math.sqrt(144 * lam - 1), math.sqrt(xi))
    rho = (24 * lam - 1 - math.sqrt(xi)) / (24 * lam)
    rho = rho * math.sqrt((48 * lam + 24 * lam * math.sqrt(3 + 144 * lam)) / xi)
    return rho, omeg

def _hc1(k: int, cs: float, rho: float, omega: float) -> float:
    if k > -1:
        return cs / math.sin(omega) * math.pow(rho, float(k)) * math.sin(omega * (k + 1))
    return 0.0

def _hs1(k: int, cs: float, rho: float, omega: float) -> float:
    c0 = (cs * cs * (1 + rho * rho) / (1 - rho * rho) /
          (1 - 2 * rho * rho * math.cos(2 * omega) + math.pow(rho, 4.0)))
    gamma = (1 - rho * rho) / (1 + rho * rho) / math.tan(omega)
    ak = abs(k)
    return c0 * math.pow(rho, float(ak)) * (math.cos(omega * ak) + gamma * math.sin(omega * ak))

def _cubic_smooth_coeff(signal: np.ndarray[float, 1], lamb: float) -> np.ndarray[float, 1]:
    rho, omega = _coeff_smooth(lamb)
    cs = 1 - 2 * rho * math.cos(omega) + rho * rho
    K = signal.size
    zi_2 = _hc1(0, cs, rho, omega) * signal[0]
    zi_1 = _hc1(0, cs, rho, omega) * signal[0] + _hc1(1, cs, rho, omega) * signal[1]
    for k in range(K):
        zi_2 += _hc1(k + 1, cs, rho, omega) * signal[k]
        zi_1 += _hc1(k + 2, cs, rho, omega) * signal[k]
    a1 = -2 * rho * math.cos(omega)
    a2 = rho * rho
    yp = np.empty(K)
    yp[0] = zi_2
    yp[1] = zi_1
    for n in range(2, K):
        yp[n] = cs * signal[n] - a1 * yp[n - 1] - a2 * yp[n - 2]
    zb2 = 0.0
    zb1 = 0.0
    for k in range(K):
        zb2 += (_hs1(k, cs, rho, omega) + _hs1(k + 1, cs, rho, omega)) * signal[K - 1 - k]
        zb1 += (_hs1(k - 1, cs, rho, omega) + _hs1(k + 2, cs, rho, omega)) * signal[K - 1 - k]
    y = np.empty(K)
    y[K - 1] = zb2
    y[K - 2] = zb1
    for t in range(2, K):
        y[K - 1 - t] = cs * yp[K - 1 - t] - a1 * y[K - t] - a2 * y[K - t + 1]
    return y

def _first_order_coeff(signal: np.ndarray[float, 1], zi: float, scale: float) -> np.ndarray[float, 1]:
    K = signal.size
    if K == 1:
        powers = 1.0
        s = signal[0]
        yplus = signal[0] + zi * (powers * s)
        out = np.empty(1)
        out[0] = zi / (zi - 1) * yplus
        return out
    st = 0.0
    p = 1.0
    for i in range(K):
        st += p * signal[i]
        p *= zi
    yplus = np.empty(K)
    yplus[0] = signal[0] + zi * st
    for n in range(1, K):
        yplus[n] = signal[n] + zi * yplus[n - 1]
    out = np.empty(K)
    out[K - 1] = zi / (zi - 1) * yplus[K - 1]
    for n in range(K - 2, -1, -1):
        out[n] = -zi * yplus[n] + zi * out[n + 1]
    for n in range(K):
        out[n] *= scale
    return out

def cspline1d(signal, lamb: float = 0.0):
    s = np.ascontiguousarray(np.atleast_1d(np.asarray(signal, float)))
    if lamb != 0.0:
        return _cubic_smooth_coeff(s, lamb)
    return _first_order_coeff(s, -2 + math.sqrt(3), 6.0)

def qspline1d(signal, lamb: float = 0.0):
    if lamb != 0.0:
        raise ValueError('Smoothing quadratic splines not supported yet.')
    s = np.ascontiguousarray(np.atleast_1d(np.asarray(signal, float)))
    return _first_order_coeff(s, -3 + 2 * math.sqrt(2.0), 8.0)

def _cubic_b(x: float) -> float:
    ax = abs(x)
    if ax >= 2.0:
        return 0.0
    if ax < 1.0:
        return 2.0 / 3 - ax * ax * (1 - ax / 2.0)
    t = 2.0 - ax
    return t * t * t / 6.0

def _quad_b(x: float) -> float:
    ax = abs(x)
    if ax >= 1.5:
        return 0.0
    if ax <= 0.5:
        return 0.75 - ax * ax
    t = 1.5 - ax
    return t * t / 2.0

def cspline1d_eval(cj, newx, dx: float = 1.0, x0: float = 0.0):
    c = np.asarray(cj, float)
    if c.size == 0:
        raise ValueError("Spline coefficients 'cj' must not be empty.")
    xa = np.asarray(newx, float)
    xf = xa.flatten()
    N = c.size
    out = np.empty(xf.size)
    for q in range(xf.size):
        v = (xf[q] - x0) / dx
        if v < 0:
            v = -v
        if v > N - 1:
            v = 2 * (N - 1) - v
        # reflect repeatedly like scipy's recursion
        while v < 0 or v > N - 1:
            if v < 0:
                v = -v
            if v > N - 1:
                v = 2 * (N - 1) - v
        jl = int(math.floor(v - 2)) + 1
        s = 0.0
        for i in range(4):
            tj = jl + i
            ind = min(max(tj, 0), N - 1)
            s += c[ind] * _cubic_b(v - tj)
        out[q] = s
    return out.reshape(xa.shape)

def qspline1d_eval(cj, newx, dx: float = 1.0, x0: float = 0.0):
    c = np.asarray(cj, float)
    if c.size == 0:
        raise ValueError("Spline coefficients 'cj' must not be empty.")
    xa = np.asarray(newx, float)
    xf = xa.flatten()
    N = c.size
    out = np.empty(xf.size)
    for q in range(xf.size):
        v = (xf[q] - x0) / dx
        while v < 0 or v > N - 1:
            if v < 0:
                v = -v
            if v > N - 1:
                v = 2 * (N - 1) - v
        jl = int(math.floor(v - 1.5)) + 1
        s = 0.0
        for i in range(3):
            tj = jl + i
            ind = min(max(tj, 0), N - 1)
            s += c[ind] * _quad_b(v - tj)
        out[q] = s
    return out.reshape(xa.shape)
