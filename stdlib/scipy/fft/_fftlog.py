# Ported from scipy 1.18.1 (BSD-3-Clause)
#
# Fast Hankel transform (FFTLog, Hamilton 2000): fht, ifht, fhtoffset.

import numpy as np
import numpy.fft as _f
import cmath
import math

_LN_2 = 0.6931471805599453
_LOGPI = 1.1447298858494002

def _lg_stirling(z: complex) -> complex:
    # (z-1/2) log z - z + log(2 pi)/2 + sum B_2k / (2k(2k-1) z^(2k-1))
    c = (8.33333333333333333e-2, -2.77777777777777778e-3, 7.93650793650793651e-4,
         -5.95238095238095238e-4, 8.41750841750841751e-4, -1.91752691752691753e-3,
         6.41025641025641026e-3, -2.95506535947712418e-2)
    rz = 1.0 / z
    rzz = rz / z
    s = complex(c[7])
    for i in range(6, -1, -1):
        s = s * rzz + c[i]
    return (z - 0.5) * cmath.log(z) - z + 0.9189385332046727 + rz * s

def _sinpi(z: complex) -> complex:
    x = z.real
    piy = math.pi * z.imag
    absx = abs(x)
    r = absx - 2.0 * math.floor(absx / 2.0)
    sx = math.sin(math.pi * r) if r != 1.0 and r != 0.0 else 0.0
    if r == 0.5:
        sx = 1.0
    elif r == 1.5:
        sx = -1.0
    if x < 0:
        sx = -sx
    cx = math.cos(math.pi * r) if r != 0.5 and r != 1.5 else 0.0
    if r == 0.0:
        cx = 1.0
    elif r == 1.0:
        cx = -1.0
    return complex(sx * math.cosh(piy), cx * math.sinh(piy))

def loggamma_c(z: complex) -> complex:
    # Principal branch of log(Gamma(z)) (scipy.special.loggamma for complex z)
    if math.isnan(z.real) or math.isnan(z.imag):
        return complex(math.nan, math.nan)
    if z.real <= 0 and z.imag == 0.0 and z.real == math.floor(z.real):
        return complex(math.nan, math.nan)
    if z.real > 7.0 or abs(z.imag) > 7.0:
        return _lg_stirling(z)
    if z.real < 0.1:
        tmp = math.copysign(2.0 * math.pi, z.imag) * math.floor(0.5 * z.real + 0.25)
        return complex(_LOGPI, tmp) - cmath.log(_sinpi(z)) - loggamma_c(1.0 - z)
    acc = complex(0.0)
    w = z
    while w.real <= 7.0:
        acc += cmath.log(w)
        w += 1.0
    return _lg_stirling(w) - acc

from scipy.special._gamma2 import poch as _poch

def fhtcoeff(n: int, dln: float, mu: float, offset: float = 0.0, bias: float = 0.0,
             inverse: bool = False):
    lnkr = offset
    q = bias
    xp = (mu + 1 + q) / 2
    xm = (mu + 1 - q) / 2
    m = n // 2 + 1
    y = np.linspace(0.0, math.pi * (n // 2) / (n * dln), m)
    u = np.empty(m, dtype=complex)
    for i in range(m):
        v = loggamma_c(complex(xm, y[i]))
        w = loggamma_c(complex(xp, y[i]))
        yy = y[i] * (2 * (_LN_2 - lnkr))
        re = w.real - v.real + _LN_2 * q
        im = w.imag + v.imag + yy
        u[i] = cmath.exp(complex(re, im))
    if n % 2 == 0:
        u[m - 1] = complex(u[m - 1].real, 0.0)
    u0 = u[0]
    if not (math.isfinite(u0.real) and math.isfinite(u0.imag)):
        u[0] = complex(2.0 ** q * _poch(xm, xp - xm))
    u0 = u[0]
    if (math.isinf(u0.real) or math.isinf(u0.imag)) and not inverse:
        u[0] = complex(0.0)
    elif u0 == complex(0.0) and inverse:
        u[0] = complex(math.inf)
    return u

def _fhtq(a, u, inverse: bool):
    n = a.shape[a.ndim - 1]
    A = _f.rfft(a, axis=-1)
    if not inverse:
        A = A * u
    else:
        A = A / np.conj(u)
    r = _f.irfft(A, n, axis=-1)
    return np.flip(r, axis=-1).copy()

def fht(a, dln: float, mu: float, offset: float = 0.0, bias: float = 0.0):
    a = np.asarray(a).astype(float)
    n = a.shape[a.ndim - 1]
    j_c = (n - 1) / 2
    j = np.arange(n, dtype=float)
    if bias != 0:
        a = a * np.exp(-bias * (j - j_c) * dln)
    u = fhtcoeff(n, dln, mu, offset=offset, bias=bias)
    A = _fhtq(a, u, False)
    if bias != 0:
        A = A * np.exp(-bias * ((j - j_c) * dln + offset))
    return A

def ifht(A, dln: float, mu: float, offset: float = 0.0, bias: float = 0.0):
    A = np.asarray(A).astype(float)
    n = A.shape[A.ndim - 1]
    j_c = (n - 1) / 2
    j = np.arange(n, dtype=float)
    if bias != 0:
        A = A * np.exp(bias * ((j - j_c) * dln + offset))
    u = fhtcoeff(n, dln, mu, offset=offset, bias=bias, inverse=True)
    a = _fhtq(A, u, True)
    if bias != 0:
        a = a / np.exp(-bias * (j - j_c) * dln)
    return a

def fhtoffset(dln: float, mu: float, initial: float = 0.0, bias: float = 0.0) -> float:
    lnkr = initial
    q = bias
    xp = (mu + 1 + q) / 2
    xm = (mu + 1 - q) / 2
    y = math.pi / (2 * dln)
    zp = loggamma_c(complex(xp, y))
    zm = loggamma_c(complex(xm, y))
    arg = (_LN_2 - lnkr) / dln + (zp.imag + zm.imag) / math.pi
    return lnkr + (arg - float(np.round(arg))) * dln
