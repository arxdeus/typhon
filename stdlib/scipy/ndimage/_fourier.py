# Ported from scipy 1.18.1 (BSD-3-Clause)

# Fourier-domain filters (scipy.ndimage._fourier): fourier_gaussian,
# fourier_uniform and fourier_shift (ports of NI_FourierFilter and
# NI_FourierShift). Inputs are float or complex arrays holding a DFT.

import numpy as np
import internal.static as static
from math import exp, sin, cos, fabs, pi as _PI
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
