# Ported from scipy 1.18.1 (BSD-3-Clause)
# Algorithms from xsf (BSD-3) lambertw.h (Halley iteration, Corless et al.)
# and scipy wright.cc (wrightomega_real, Fritsch-Shafer-Crowley).

import cmath
from ._kernels import exp, log, fabs, pow
from C import fma(float, float, float) -> float

NAN = float('nan')
INF = float('inf')
PI = 3.14159265358979323846
M_E = 2.71828182845904523536
EXPN1 = 0.36787944117144232159553
OMEGA = 0.56714329040978387299997
TWOITERTOL = 2.220446049250313e-16

def _cevalpoly(c0: float, c1: float, c2: float, z: complex) -> complex:
    a = c0
    b = c1
    r = 2 * z.real
    s = z.real * z.real + z.imag * z.imag
    tmp = b
    b = fma(-s, a, c2)
    a = fma(r, a, tmp)
    return z * a + b

def _branchpt(z: complex) -> complex:
    p = cmath.sqrt(2.0 * (M_E * z + 1.0))
    return _cevalpoly(-1.0 / 3.0, 1.0, -1.0, p)

def _pade0(z: complex) -> complex:
    return z * _cevalpoly(12.85106382978723404255, 12.34042553191489361902, 1.0, z) / \
        _cevalpoly(32.53191489361702127660, 14.34042553191489361702, 1.0, z)

def _asy(z: complex, k: int) -> complex:
    w = cmath.log(z) + complex(0.0, 2.0 * PI * k)
    return w - cmath.log(w)

def lambertw(z: complex, k: int, tol: float) -> complex:
    if z.real != z.real or z.imag != z.imag:
        return z
    if z.real == INF:
        return z + complex(0.0, 2.0 * PI * k)
    if z.real == -INF:
        return -z + complex(0.0, 2.0 * PI * k + PI)
    if z.real == 0.0 and z.imag == 0.0:
        if k == 0:
            return z
        return complex(-INF, 0.0)
    if z.real == 1.0 and z.imag == 0.0 and k == 0:
        return complex(OMEGA, 0.0)
    absz = abs(z)
    w = complex(0.0, 0.0)
    if k == 0:
        if abs(z + EXPN1) < 0.3:
            w = _branchpt(z)
        elif -1.0 < z.real and z.real < 1.5 and fabs(z.imag) < 1.0 and -2.5 * fabs(z.imag) - 0.2 < z.real:
            w = _pade0(z)
        else:
            w = _asy(z, k)
    elif k == -1:
        if absz <= EXPN1 and z.imag == 0.0 and z.real < 0.0:
            w = complex(log(-z.real), 0.0)
        else:
            w = _asy(z, k)
    else:
        w = _asy(z, k)
    if w.real >= 0:
        for i in range(100):
            ew = cmath.exp(-w)
            wewz = w - z * ew
            wn = w - wewz / (w + 1.0 - (w + 2.0) * wewz / (2.0 * w + 2.0))
            if abs(wn - w) <= tol * abs(wn):
                return wn
            w = wn
    else:
        for i in range(100):
            ew = cmath.exp(w)
            wew = w * ew
            wewz = wew - z
            wn = w - wewz / (wew + ew - (w + 2.0) * wewz / (2.0 * w + 2.0))
            if abs(wn - w) <= tol * abs(wn):
                return wn
            w = wn
    return complex(NAN, NAN)

def wrightomega_real(x: float) -> float:
    if x != x:
        return x
    if x == INF:
        return x
    if x == -INF:
        return 0.0
    if x < -50.0:
        return exp(x)
    if x > 1e20:
        return x
    w = 0.0
    if x < -2.0:
        w = exp(x)
    elif x < 1:
        w = exp(2.0 * (x - 1.0) / 3.0)
    else:
        w = log(x)
        w = x - w + w / x
    r = x - w - log(w)
    wp1 = w + 1.0
    e = r / wp1 * (2.0 * wp1 * (wp1 + 2.0 / 3.0 * r) - r) / (2.0 * wp1 * (wp1 + 2.0 / 3.0 * r) - 2.0 * r)
    w = w * (1.0 + e)
    if fabs((2.0 * w * w - 8.0 * w - 1.0) * pow(fabs(r), 4.0)) >= TWOITERTOL * 72.0 * pow(fabs(wp1), 6.0):
        r = x - w - log(w)
        wp1 = w + 1.0
        e = r / wp1 * (2.0 * wp1 * (wp1 + 2.0 / 3.0 * r) - r) / (2.0 * wp1 * (wp1 + 2.0 / 3.0 * r) - 2.0 * r)
        w = w * (1.0 + e)
    return w
