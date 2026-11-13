# Ported from scipy 1.18.1 (BSD-3-Clause)

# Fourier-domain filters (scipy.ndimage._fourier): fourier_gaussian,
# fourier_uniform and fourier_shift (ports of NI_FourierFilter and
# NI_FourierShift). Inputs are float or complex arrays holding a DFT.

import numpy as np
import internal.static as static
from math import exp, sin, cos, fabs, sqrt, pi as _PI
from ._ni_support import _seq_float, _shape, _strides, _prod, _next_coord

def _tables(sh: List[int], params: List[float], n: int, axis: int, kind: int) -> List[Optional[List[float]]]:
    nd = len(sh)
    out = List[Optional[List[float]]]()
    for hh in range(nd):
        dim = sh[hh]
        if dim <= 1:
            out.append(None)
            continue
        shape = (dim if n < 0 else n) if hh == axis else dim
        t = [0.0] * dim
        ks = List[int]()
        if hh == axis and n >= 0:
            ks = list(range(dim))
        else:
            ks = list(range((dim + 1) // 2)) + list(range(-(dim // 2), 0))
        if kind == 0:
            p = params[hh] * _PI / float(shape)
            p = -2.0 * p * p
            for j in range(dim):
                kk = ks[j]
                tmp = p * kk * kk
                t[j] = 0.0 if fabs(tmp) > 50.0 else exp(tmp)
        elif kind == 1:
            tmp = _PI * params[hh] / (n if (hh == axis and n >= 0) else dim)
            t[0] = 1.0
            for j in range(1, dim):
                kk = ks[j]
                t[j] = sin(tmp * kk) / (tmp * kk) if tmp > 0.0 else 0.0
        else:
            s = -2.0 * _PI * params[hh] / float(shape)
            for j in range(dim):
                t[j] = s * ks[j]
        out.append(t)
    return out

def _norm_axis(axis: int, nd: int) -> int:
    if axis < -nd or axis >= nd:
        raise ValueError(f"axis {axis} is out of bounds for array of dimension {nd}")
    return axis % nd

def _filter(input, params, n: int, axis: int, kind: int):
    a = np.ascontiguousarray(np.asarray(input))
    sh = _shape(a)
    nd = len(sh)
    ax = _norm_axis(axis, nd)
    tabs = _tables(sh, _seq_float(params, nd), n, ax, kind)
    T = a.dtype
    if T is complex or T is complex64 or T is float32:
        out = np.empty(a.shape, dtype=T)
    else:
        out = np.empty(a.shape, dtype=float)
    coord = [0] * nd
    for h in range(a.size):
        tmp = 1.0
        for k in range(nd):
            t = tabs[k]
            if t is not None:
                tmp *= t[coord[k]]
        if T is complex or T is complex64:
            v = a.data[h]
            out.data[h] = type(v)(v.real * tmp, v.imag * tmp)
        else:
            out.data[h] = type(out.data[0])(tmp * float(a.data[h]))
        _next_coord(coord, sh)
    return out

def fourier_gaussian(input, sigma, n: int = -1, axis: int = -1, output = None):
    return _filter(input, sigma, n, axis, 0)

def fourier_uniform(input, size, n: int = -1, axis: int = -1, output = None):
    return _filter(input, size, n, axis, 1)

def fourier_shift(input, shift, n: int = -1, axis: int = -1, output = None):
    a = np.ascontiguousarray(np.asarray(input))
    sh = _shape(a)
    nd = len(sh)
    ax = _norm_axis(axis, nd)
    tabs = _tables(sh, _seq_float(shift, nd), n, ax, 2)
    T = a.dtype
    out = np.empty(a.shape, dtype=complex64 if T is complex64 else complex)
    coord = [0] * nd
    for h in range(a.size):
        tmp = 0.0
        for k in range(nd):
            t = tabs[k]
            if t is not None:
                tmp += t[coord[k]]
        s = sin(tmp)
        c = cos(tmp)
        if T is complex or T is complex64:
            v = a.data[h]
            r = v.real * c - v.imag * s
            i = v.real * s + v.imag * c
            out.data[h] = type(out.data[0])(r, i)
        else:
            x = float(a.data[h])
            out.data[h] = type(out.data[0])(x * c, x * s)
        _next_coord(coord, sh)
    return out

# --- fourier_ellipsoid (NI_FourierFilter, _NI_ELLIPSOID) -------------------

def _polevl(x: float, coef: List[float]) -> float:
    ans = coef[0]
    for i in range(1, len(coef)):
        ans = ans * x + coef[i]
    return ans

def _p1evl(x: float, coef: List[float]) -> float:
    ans = x + coef[0]
    for i in range(1, len(coef)):
        ans = ans * x + coef[i]
    return ans

def _bessel_j1(x: float) -> float:
    RP = [-8.99971225705559398224E8, 4.52228297998194034323E11,
          -7.27494245221818276015E13, 3.68295732863852883286E15]
    RQ = [6.20836478118054335476E2, 2.56987256757748830383E5, 8.35146791431949253037E7,
          2.21511595479792499675E10, 4.74914122079991414898E12, 7.84369607876235854894E14,
          8.95222336184627338078E16, 5.32278620332680085395E18]
    PP = [7.62125616208173112003E-4, 7.31397056940917570436E-2, 1.12719608129684925192E0,
          5.11207951146807644818E0, 8.42404590141772420927E0, 5.21451598682361504063E0,
          1.00000000000000000254E0]
    PQ = [5.71323128072548699714E-4, 6.88455908754495404082E-2, 1.10514232634061696926E0,
          5.07386386128601488557E0, 8.39985554327604159757E0, 5.20982848682361821619E0,
          9.99999999999999997461E-1]
    QP = [5.10862594750176621635E-2, 4.98213872951233449420E0, 7.58238284132545283818E1,
          3.66779609360150777800E2, 7.10856304998926107277E2, 5.97489612400613639965E2,
          2.11688757100572135698E2, 2.52070205858023719784E1]
    QQ = [7.42373277035675149943E1, 1.05644886038262816351E3, 4.98641058337653607651E3,
          9.56231892404756170795E3, 7.99704160447350683650E3, 2.82619278517639096600E3,
          3.36093607810698293419E2]
    THPIO4 = 2.35619449019234492885
    SQ2OPI = .79788456080286535588
    Z1 = 1.46819706421238932572E1
    Z2 = 4.92184563216946036703E1
    w = x
    if x < 0:
        w = -x
    if w <= 5.0:
        z = x * x
        w = _polevl(z, RP) / _p1evl(z, RQ)
        w = w * x * (z - Z1) * (z - Z2)
        return w
    w = 5.0 / x
    z = w * w
    p = _polevl(z, PP) / _polevl(z, PQ)
    q = _polevl(z, QP) / _p1evl(z, QQ)
    xn = x - THPIO4
    p = p * cos(xn) - w * q * sin(xn)
    return p * SQ2OPI / sqrt(x)

def fourier_ellipsoid(input, size, n: int = -1, axis: int = -1, output = None):
    a = np.ascontiguousarray(np.asarray(input))
    sh = _shape(a)
    nd = len(sh)
    ax = _norm_axis(axis, nd)
    if nd < 1 or nd > 3:
        raise NotImplementedError("Only 1d, 2d and 3d inputs are supported")
    prm = _seq_float(size, nd)
    tabs = List[List[float]]()
    for hh in range(nd):
        dim = sh[hh]
        t = [0.0] * dim
        if dim > 0:
            t[0] = 1.0
        if hh == ax and n >= 0:
            tmp = _PI * prm[hh] / float(n)
            for kk in range(dim):
                t[kk] = float(kk) * tmp
        else:
            tmp = _PI * prm[hh] / float(dim)
            jj = 0
            for kk in range((dim + 1) // 2):
                t[jj] = float(kk) * tmp
                jj += 1
            for kk in range(-(dim // 2), 0):
                t[jj] = float(kk) * tmp
                jj += 1
        tabs.append(t)
    if nd > 1:
        for hh in range(nd):
            for kk in range(sh[hh]):
                tabs[hh][kk] = tabs[hh][kk] * tabs[hh][kk]
    T = a.dtype
    if T is complex or T is complex64 or T is float32:
        out = np.empty(a.shape, dtype=T)
    else:
        out = np.empty(a.shape, dtype=float)
    coord = [0] * nd
    for h in range(a.size):
        tmp = 1.0
        if nd == 1:
            tmp = tabs[0][coord[0]]
            tmp = sin(tmp) / tmp if tmp != 0.0 else 1.0
        elif nd == 2:
            tmp = 0.0
            for k in range(2):
                tmp += tabs[k][coord[k]]
            tmp = sqrt(tmp)
            tmp = 2.0 * _bessel_j1(tmp) / tmp if tmp > 0.0 else 1.0
        else:
            r = 0.0
            for k in range(3):
                r += tabs[k][coord[k]]
            r = sqrt(r)
            if r > 0.0:
                tmp = 3.0 * (sin(r) - r * cos(r))
                tmp /= r * r * r
            else:
                tmp = 1.0
        if T is complex or T is complex64:
            v = a.data[h]
            out.data[h] = type(v)(v.real * tmp, v.imag * tmp)
        else:
            out.data[h] = type(out.data[0])(tmp * float(a.data[h]))
        _next_coord(coord, sh)
    return out
