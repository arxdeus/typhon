# Ported from scipy 1.18.1 (BSD-3-Clause): scipy.fftpack (legacy interface).
#
# rfft/irfft use the FFTPACK real "packed" layout
# [y0, Re y1, Im y1, ..., (Re y_{n/2})]; the pseudo-differential operators
# (diff, tilbert, hilbert, cs_diff, ..., shift) follow scipy's convolve.pyx.

import numpy as np
import numpy.fft as _f
from math import pi, sin, cos, sinh, cosh, tanh
from numpy.util import normalize_axis_index
from scipy.fft import fft, ifft, fftn, ifftn, fft2, ifft2, fftshift, ifftshift, fftfreq, \
    dct, dst, dctn, dstn
import scipy.fft as _sf

def _inorm(norm: Optional[str]) -> str:
    # fftpack's inverse r2r transforms are unnormalised unless norm='ortho'
    if norm is None:
        return 'forward'
    return norm

def idct(x, type: int = 2, n: Optional[int] = None, axis: int = -1,
         norm: Optional[str] = None, overwrite_x: bool = False):
    return _sf.idct(x, type, n, axis, _inorm(norm))

def idst(x, type: int = 2, n: Optional[int] = None, axis: int = -1,
         norm: Optional[str] = None, overwrite_x: bool = False):
    return _sf.idst(x, type, n, axis, _inorm(norm))

def idctn(x, type: int = 2, shape = None, axes = None, norm: Optional[str] = None,
          overwrite_x: bool = False):
    return _sf.idctn(x, type, shape, axes, _inorm(norm))

def idstn(x, type: int = 2, shape = None, axes = None, norm: Optional[str] = None,
          overwrite_x: bool = False):
    return _sf.idstn(x, type, shape, axes, _inorm(norm))

def _pack(c, n: int):
    # complex rfft output (n//2+1,) -> packed real (n,)
    out = np.empty(n)
    out[0] = c[0].real
    k = 1
    for j in range(1, n - 1, 2):
        out[j] = c[k].real
        out[j + 1] = c[k].imag
        k += 1
    if n % 2 == 0 and n > 1:
        out[n - 1] = c[n // 2].real
    return out

def _unpack(x, n: int):
    m = n // 2 + 1
    c = np.zeros(m, dtype=complex)
    c[0] = complex(x[0], 0.0)
    k = 1
    for j in range(1, n - 1, 2):
        c[k] = complex(x[j], x[j + 1])
        k += 1
    if n % 2 == 0 and n > 1:
        c[n // 2] = complex(x[n - 1], 0.0)
    return c

def _apply_rows(x, n: Optional[int], axis: int, fwd: bool):
    a = np.asarray(x, dtype=float)
    ax = normalize_axis_index(axis, a.ndim)
    b = np.moveaxis(a, ax, -1)
    N0 = b.shape[b.ndim - 1]
    N = N0 if n is None else n
    if N < 1:
        raise ValueError(f"invalid number of data points ({N}) specified")
    rows = b.size // N0 if N0 > 0 else 0
    B = np.ascontiguousarray(b).reshape((rows, N0))
    out = np.empty((rows, N))
    for r in range(rows):
        v = np.zeros(N)
        for i in range(min(N, N0)):
            v[i] = B[r, i]
        if fwd:
            o = _pack(_f.rfft(v), N)
        else:
            o = _f.irfft(_unpack(v, N), N)
        for i in range(N):
            out[r, i] = o[i]
    shp = b.shape[:-1] + (N,)
    return np.moveaxis(out.reshape(shp), -1, ax).copy()

def rfft(x, n: Optional[int] = None, axis: int = -1, overwrite_x: bool = False):
    return _apply_rows(x, n, axis, True)

def irfft(x, n: Optional[int] = None, axis: int = -1, overwrite_x: bool = False):
    return _apply_rows(x, n, axis, False)

def rfftfreq(n: int, d: float = 1.0):
    if n < 0:
        raise ValueError(f"n = {n} is not valid. n must be a nonnegative integer.")
    out = np.empty(n)
    for i in range(n):
        out[i] = float((i + 1) // 2) / float(n * d)
    return out

def next_fast_len(target: int) -> int:
    return _sf.next_fast_len(target, True)

# ------------------------------------------------------------ pseudo diffs

_K_DIFF = 0
_K_TILB = 1
_K_ITILB = 2
_K_HILB = 3
_K_CS = 4
_K_SC = 5
_K_SS = 6
_K_CC = 7
_K_COS = 8
_K_SIN = 9

def _kern(kind: int, k: int, a: float, b: float, order: int) -> float:
    if kind == _K_DIFF:
        return (a * k) ** order if k != 0 else 0.0
    if kind == _K_TILB:
        return 1.0 / tanh(a * k) if k != 0 else 0.0
    if kind == _K_ITILB:
        return -tanh(a * k) if k != 0 else 0.0
    if kind == _K_HILB:
        return 1.0 if k > 0 else (-1.0 if k < 0 else 0.0)
    if kind == _K_CS:
        return -cosh(a * k) / sinh(b * k) if k != 0 else 0.0
    if kind == _K_SC:
        return sinh(a * k) / cosh(b * k) if k != 0 else 0.0
    if kind == _K_SS:
        return sinh(a * k) / sinh(b * k) if k != 0 else a / b
    if kind == _K_CC:
        return cosh(a * k) / cosh(b * k)
    if kind == _K_COS:
        return cos(a * k)
    return sin(a * k)

def _omega(n: int, kind: int, a: float, b: float, order: int, d: int, zero_nyquist: int):
    zn = (d % 2 != 0) if zero_nyquist < 0 else zero_nyquist != 0
    om = np.empty(n)
    l = n if n % 2 != 0 else n - 1
    om[0] = _kern(kind, 0, a, b, order) / n
    dd = d % 4
    sr = 1.0 / n if dd == 0 or dd == 1 else -1.0 / n
    si = 1.0 / n if dd == 0 or dd == 3 else -1.0 / n
    k = 1
    for j in range(1, l, 2):
        x = _kern(kind, k, a, b, order)
        om[j] = sr * x
        om[j + 1] = si * x
        k += 1
    if n % 2 == 0:
        if zn:
            om[n - 1] = 0.0
        else:
            om[n - 1] = sr * _kern(kind, k, a, b, order)
    return om

def _convolve(x, w, swap: bool):
    n = x.shape[0]
    X = _pack(_f.rfft(x), n)
    if swap:
        X[0] *= w[0]
        for i in range(1, n - 1, 2):
            c = X[i] * w[i]
            X[i] = X[i + 1] * w[i + 1]
            X[i + 1] = c
        if n % 2 == 0:
            X[n - 1] *= w[n - 1]
    else:
        X = X * w
    return _f.irfft(_unpack(X, n), n) * n

def _convolve_z(x, wr, wi):
    n = x.shape[0]
    X = _pack(_f.rfft(x), n)
    X[0] *= wr[0] + wi[0]
    if n % 2 == 0:
        X[n - 1] *= wr[n - 1] + wi[n - 1]
    for i in range(1, n - 1, 2):
        c = X[i + 1] * wr[i + 1] + X[i] * wi[i]
        X[i] = X[i] * wr[i] + X[i + 1] * wi[i + 1]
        X[i + 1] = c
    return _f.irfft(_unpack(X, n), n) * n

def _real_op(x, kind: int, a: float, b: float, order: int, d: int, zn: int, swap: bool):
    t = np.asarray(x)
    if t.dtype is complex:
        re = _real_op(t.real.astype(float), kind, a, b, order, d, zn, swap)
        im = _real_op(t.imag.astype(float), kind, a, b, order, d, zn, swap)
        return re + 1j * im
    else:
        xa = np.ascontiguousarray(t.astype(float))
        w = _omega(xa.shape[0], kind, a, b, order, d, zn)
        return _convolve(xa, w, swap)

def diff(x, order: int = 1, period: Optional[float] = None):
    if order == 0:
        t = np.asarray(x)
        if t.dtype is complex:
            return t.copy()
        else:
            return t.astype(float)
    c = 1.0 if period is None else 2 * pi / period
    return _real_op(x, _K_DIFF, c, 0.0, order, order, 1, order % 2 != 0)

def tilbert(x, h: float, period: Optional[float] = None):
    hh = h if period is None else h * 2 * pi / period
    return _real_op(x, _K_TILB, hh, 0.0, 0, 1, -1, True)

def itilbert(x, h: float, period: Optional[float] = None):
    hh = h if period is None else h * 2 * pi / period
    return _real_op(x, _K_ITILB, hh, 0.0, 0, 1, -1, True)

def hilbert(x):
    return _real_op(x, _K_HILB, 0.0, 0.0, 0, 1, -1, True)

def ihilbert(x):
    return -hilbert(x)

def _ab(a: float, b: float, period: Optional[float]):
    if period is None:
        return a, b
    return a * 2 * pi / period, b * 2 * pi / period

def cs_diff(x, a: float, b: float, period: Optional[float] = None):
    aa, bb = _ab(a, b, period)
    return _real_op(x, _K_CS, aa, bb, 0, 1, -1, True)

def sc_diff(x, a: float, b: float, period: Optional[float] = None):
    aa, bb = _ab(a, b, period)
    return _real_op(x, _K_SC, aa, bb, 0, 1, -1, True)

def ss_diff(x, a: float, b: float, period: Optional[float] = None):
    aa, bb = _ab(a, b, period)
    return _real_op(x, _K_SS, aa, bb, 0, 0, -1, False)

def cc_diff(x, a: float, b: float, period: Optional[float] = None):
    aa, bb = _ab(a, b, period)
    return _real_op(x, _K_CC, aa, bb, 0, 0, -1, False)

def _shift_real(x, a: float):
    xa = np.ascontiguousarray(x.astype(float))
    n = xa.shape[0]
    wr = _omega(n, _K_COS, a, 0.0, 0, 0, 0)
    wi = _omega(n, _K_SIN, a, 0.0, 0, 1, 0)
    return _convolve_z(xa, wr, wi)

def shift(x, a: float, period: Optional[float] = None):
    aa = a if period is None else a * 2 * pi / period
    t = np.asarray(x)
    if t.dtype is complex:
        return _shift_real(t.real, aa) + 1j * _shift_real(t.imag, aa)
    else:
        return _shift_real(t, aa)
