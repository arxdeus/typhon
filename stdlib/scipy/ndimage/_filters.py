# Ported from scipy 1.18.1 (BSD-3-Clause)

# Multidimensional image filters (scipy.ndimage._filters). Line filters
# reproduce NI_Correlate1D / NI_UniformFilter1D / NI_MinOrMaxFilter1D on
# extended line buffers; N-d filters reproduce NI_Correlate,
# NI_MinOrMaxFilter and NI_RankFilter, including the accumulation order.

import numpy as np
import internal.static as static
from math import fabs, sqrt, exp
from ._ni_support import _extend_mode_to_code, _seq_int, _seq_float, _seq_str, \
    _seq_opt_int, _check_axes, _shape, _strides, _prod, _fbuf, _farray, \
    _line_bases, _extend_line, _Footprint, _gather_offsets, _next_coord, \
    _proto, _quantize, _emit, _tuple_shape

_DBL_EPS = 2.220446049250313e-16

def _norm_axis(axis: int, nd: int) -> int:
    if axis < -nd or axis >= nd:
        raise ValueError(f"axis {axis} is out of bounds for array of dimension {nd}")
    return axis % nd

def _invalid_origin(origin: int, lenw: int) -> bool:
    return origin < -(lenw // 2) or origin > (lenw - 1) // 2

# ---- line kernels ------------------------------------------------------------

def _corr1d_buf(src: List[float], sh: List[int], axis: int, fw: List[float],
                mode: int, cval: float, origin: int) -> List[float]:
    fs = len(fw)
    size1 = fs // 2
    size2 = fs - size1 - 1
    symmetric = 0
    if fs & 1:
        symmetric = 1
        for ii in range(1, fs // 2 + 1):
            if fabs(fw[ii + size1] - fw[size1 - ii]) > _DBL_EPS:
                symmetric = 0
                break
        if symmetric == 0:
            symmetric = -1
            for ii in range(1, fs // 2 + 1):
                if fabs(fw[size1 + ii] + fw[size1 - ii]) > _DBL_EPS:
                    symmetric = 0
                    break
    out = [0.0] * len(src)
    L = sh[axis]
    st = _strides(sh)[axis]
    before = size1 + origin
    after = size2 - origin
    line = [0.0] * (before + L + after)
    for base in _line_bases(sh, axis):
        for k in range(L):
            line[before + k] = src[base + k * st]
        _extend_line(line, before, L, after, mode, cval)
        # as in NI_Correlate1D, iline[jj] == line[ll + size1 + jj]
        for ll in range(L):
            c = ll + size1
            if symmetric > 0:
                v = line[c] * fw[size1]
                for jj in range(-size1, 0):
                    v += (line[c + jj] + line[c - jj]) * fw[size1 + jj]
            elif symmetric < 0:
                v = line[c] * fw[size1]
                for jj in range(-size1, 0):
                    v += (line[c + jj] - line[c - jj]) * fw[size1 + jj]
            else:
                v = line[c + size2] * fw[size1 + size2]
                for jj in range(-size1, size2):
                    v += line[c + jj] * fw[size1 + jj]
            out[base + ll * st] = v
    return out

def _uniform1d_buf(src: List[float], sh: List[int], axis: int, size: int,
                   mode: int, cval: float, origin: int) -> List[float]:
    size1 = size // 2
    size2 = size - size1 - 1
    out = [0.0] * len(src)
    L = sh[axis]
    st = _strides(sh)[axis]
    before = size1 + origin
    after = size2 - origin
    line = [0.0] * (before + L + after)
    for base in _line_bases(sh, axis):
        for k in range(L):
            line[before + k] = src[base + k * st]
        _extend_line(line, before, L, after, mode, cval)
        tmp = 0.0
        for ll in range(size):
            tmp += line[ll]
        out[base] = tmp / size
        for ll in range(1, L):
            tmp += line[ll - 1 + size] - line[ll - 1]
            out[base + ll * st] = tmp / size
    return out

def _minmax1d_buf(src: List[float], sh: List[int], axis: int, size: int,
                  mode: int, cval: float, origin: int, minimum: bool) -> List[float]:
    size1 = size // 2
    size2 = size - size1 - 1
    out = [0.0] * len(src)
    L = sh[axis]
    st = _strides(sh)[axis]
    before = size1 + origin
    after = size2 - origin
    line = [0.0] * (before + L + after)
    for base in _line_bases(sh, axis):
        for k in range(L):
            line[before + k] = src[base + k * st]
        _extend_line(line, before, L, after, mode, cval)
        for ll in range(L):
            v = line[ll]
            for j in range(1, size):
                w = line[ll + j]
                if minimum:
                    if w < v:
                        v = w
                else:
                    if w > v:
                        v = w
            out[base + ll * st] = v
    return out

# ---- argument helpers --------------------------------------------------------

def _arr(input):
    a = np.asarray(input)
    if a.dtype is complex or a.dtype is complex64:
        compile_error("Complex type not supported")
    return a

def _weights1d(weights) -> List[float]:
    w = np.asarray(weights, dtype=float)
    if static.len(w.shape) != 1:
        raise RuntimeError('no filter weights given')
    if w.shape[0] < 1:
        raise RuntimeError('no filter weights given')
    return _fbuf(w)

def correlate1d(input, weights, axis: int = -1, output = None, mode: str = 'reflect',
                cval: float = 0.0, origin: int = 0):
    a = _arr(input)
    fw = _weights1d(weights)
    sh = _shape(a)
    ax = _norm_axis(axis, len(sh))
    if _invalid_origin(origin, len(fw)):
        raise ValueError('Invalid origin; origin must satisfy -(len(weights) // 2) <= origin <= (len(weights)-1) // 2')
    m = _extend_mode_to_code(mode)
    r = _corr1d_buf(_fbuf(a), sh, ax, fw, m, cval, origin)
    return _emit(r, a.shape, _proto(a, output), output)

def convolve1d(input, weights, axis: int = -1, output = None, mode: str = 'reflect',
               cval: float = 0.0, origin: int = 0):
    fw = _weights1d(weights)
    fw.reverse()
    org = -origin
    if not (len(fw) & 1):
        org -= 1
    return correlate1d(input, fw, axis, output, mode, cval, org)

def _gaussian_kernel1d(sigma: float, order: int, radius: int) -> List[float]:
    if order < 0:
        raise ValueError('order must be non-negative')
    sigma2 = sigma * sigma
    x = np.arange(-radius, radius + 1).astype(float)
    phi_x = np.exp(-0.5 / sigma2 * x ** 2)
    phi_x = phi_x / phi_x.sum()
    if order == 0:
        return _fbuf(phi_x)
    q = np.zeros(order + 1)
    q[0] = 1.0
    Q = np.zeros((order + 1, order + 1))
    for i in range(order):
        Q[i, i + 1] = float(i + 1)
        Q[i + 1, i] = 1.0 / -sigma2
    for _ in range(order):
        q = Q.dot(q)
    n = 2 * radius + 1
    P = np.empty((n, order + 1))
    for i in range(n):
        for j in range(order + 1):
            P[i, j] = x[i] ** j
    qq = P.dot(q)
    return _fbuf(qq * phi_x)

def _gauss1d_buf(src: List[float], sh: List[int], axis: int, sigma: float, order: int,
                 mode: int, cval: float, truncate: float, radius: Optional[int]) -> List[float]:
    lw = int(truncate * sigma + 0.5)
    if radius is not None:
        lw = radius
    if lw < 0:
        raise ValueError('Radius must be a nonnegative integer.')
    w = _gaussian_kernel1d(sigma, order, lw)
    w.reverse()
    return _corr1d_buf(src, sh, axis, w, mode, cval, 0)

def gaussian_filter1d(input, sigma: float, axis: int = -1, order: int = 0, output = None,
                      mode: str = 'reflect', cval: float = 0.0, truncate: float = 4.0,
                      radius: Optional[int] = None):
    a = _arr(input)
    sh = _shape(a)
    ax = _norm_axis(axis, len(sh))
    r = _gauss1d_buf(_fbuf(a), sh, ax, float(sigma), order, _extend_mode_to_code(mode),
                     cval, truncate, radius)
    return _emit(r, a.shape, _proto(a, output), output)

def _gaussian_buf(a, sigma, order, proto, mode, cval: float, truncate: float, radius, axes) -> List[float]:
    sh = _shape(a)
    ax = _check_axes(axes, len(sh))
    na = len(ax)
    orders = _seq_int(order, na)
    sigmas = _seq_float(sigma, na)
    modes = _seq_str(mode, na)
    radii = _seq_opt_int(radius, na)
    buf = _fbuf(a)
    did = False
    for ii in range(na):
        if sigmas[ii] > 1e-15:
            buf = _gauss1d_buf(buf, sh, ax[ii], sigmas[ii], orders[ii],
                               _extend_mode_to_code(modes[ii]), cval, truncate, radii[ii])
            buf = _quantize(buf, proto)
            did = True
    if not did:
        buf = _quantize(buf, proto)
    return buf

def gaussian_filter(input, sigma, order = 0, output = None, mode = 'reflect',
                    cval: float = 0.0, truncate: float = 4.0, radius = None, axes = None):
    a = _arr(input)
    p = _proto(a, output)
    buf = _gaussian_buf(a, sigma, order, p, mode, cval, truncate, radius, axes)
    return _emit(buf, a.shape, p, output)

def _sep_deriv(a, axis: int, w1: List[float], w2: List[float], output, mode, cval: float):
    sh = _shape(a)
    nd = len(sh)
    ax = _norm_axis(axis, nd)
    p = _proto(a, output)
    modes = _seq_str(mode, nd)
    buf = _corr1d_buf(_fbuf(a), sh, ax, w1, _extend_mode_to_code(modes[ax]), cval, 0)
    buf = _quantize(buf, p)
    for ii in range(nd):
        if ii != ax:
            buf = _corr1d_buf(buf, sh, ii, w2, _extend_mode_to_code(modes[ii]), cval, 0)
            buf = _quantize(buf, p)
    return _emit(buf, a.shape, p, output)

def prewitt(input, axis: int = -1, output = None, mode = 'reflect', cval: float = 0.0):
    return _sep_deriv(_arr(input), axis, [-1.0, 0.0, 1.0], [1.0, 1.0, 1.0], output, mode, cval)

def sobel(input, axis: int = -1, output = None, mode = 'reflect', cval: float = 0.0):
    return _sep_deriv(_arr(input), axis, [-1.0, 0.0, 1.0], [1.0, 2.0, 1.0], output, mode, cval)

def laplace(input, output = None, mode = 'reflect', cval: float = 0.0, axes = None):
    a = _arr(input)
    sh = _shape(a)
    ax = _check_axes(axes, len(sh))
    p = _proto(a, output)
    if len(ax) == 0:
        return _emit(_quantize(_fbuf(a), p), a.shape, p, output)
    modes = _seq_str(mode, len(ax))
    src = _fbuf(a)
    w = [1.0, -2.0, 1.0]
    acc = _quantize(_corr1d_buf(src, sh, ax[0], w, _extend_mode_to_code(modes[0]), cval, 0), p)
    for ii in range(1, len(ax)):
        t = _quantize(_corr1d_buf(src, sh, ax[ii], w, _extend_mode_to_code(modes[ii]), cval, 0), p)
        for k in range(len(acc)):
            acc[k] += t[k]
        acc = _quantize(acc, p)
    return _emit(acc, a.shape, p, output)

def _order_for(nd: int, axis: int, k: int) -> List[int]:
    o = [0] * nd
    o[axis] = k
    return o

def _sigma_full(sigma, ax: List[int], nd: int) -> List[float]:
    s = _seq_float(sigma, len(ax))
    if len(ax) < nd:
        full = [0.0] * nd
        for i in range(len(ax)):
            full[ax[i]] = s[i]
        return full
    return s

def gaussian_laplace(input, sigma, output = None, mode = 'reflect', cval: float = 0.0,
                     axes = None, truncate: float = 4.0, radius = None):
    a = _arr(input)
    sh = _shape(a)
    nd = len(sh)
    ax = _check_axes(axes, nd)
    p = _proto(a, output)
    if len(ax) == 0:
        return _emit(_quantize(_fbuf(a), p), a.shape, p, output)
    sig = _sigma_full(sigma, ax, nd)
    modes = _seq_str(mode, len(ax))
    acc = _gaussian_buf(a, sig, _order_for(nd, ax[0], 2), p, modes[0], cval, truncate, radius, None)
    for ii in range(1, len(ax)):
        t = _gaussian_buf(a, sig, _order_for(nd, ax[ii], 2), p, modes[ii], cval, truncate, radius, None)
        for k in range(len(acc)):
            acc[k] += t[k]
        acc = _quantize(acc, p)
    return _emit(acc, a.shape, p, output)

def gaussian_gradient_magnitude(input, sigma, output = None, mode = 'reflect', cval: float = 0.0,
                                axes = None, truncate: float = 4.0, radius = None):
    a = _arr(input)
    sh = _shape(a)
    nd = len(sh)
    ax = _check_axes(axes, nd)
    p = _proto(a, output)
    if len(ax) == 0:
        return _emit(_quantize(_fbuf(a), p), a.shape, p, output)
    modes = _seq_str(mode, len(ax))
    acc = _gaussian_buf(a, sigma, _order_for(nd, ax[0], 1), p, modes[0], cval, truncate, radius, None)
    for k in range(len(acc)):
        acc[k] = acc[k] * acc[k]
    acc = _quantize(acc, p)
    for ii in range(1, len(ax)):
        t = _gaussian_buf(a, sigma, _order_for(nd, ax[ii], 1), p, modes[ii], cval, truncate, radius, None)
        for k in range(len(acc)):
            acc[k] += _quantize([t[k] * t[k]], p)[0]
        acc = _quantize(acc, p)
    for k in range(len(acc)):
        acc[k] = sqrt(acc[k])
    return _emit(acc, a.shape, p, output)

# ---- N-d footprint filters ------------------------------------------------------

def _bool_buf(fp) -> List[bool]:
    A = np.ascontiguousarray(np.asarray(fp))
    out = List[bool](A.size)
    for i in range(A.size):
        out.append(bool(A.data[i]))
    return out

def _expand_fp(nd: int, ax: List[int], fsh: List[int], what: str) -> List[int]:
    # inserts unit dimensions for axes not filtered
    if len(ax) < nd:
        if len(fsh) != len(ax):
            raise RuntimeError(f'{what}.ndim ({len(fsh)}) must match len(axes) ({len(ax)})')
        full = [1] * nd
        for i in range(len(ax)):
            full[ax[i]] = fsh[i]
        return full
    if len(fsh) != nd:
        raise RuntimeError(f'{what}.ndim ({len(fsh)}) must match len(axes) ({len(ax)})')
    return fsh

def _expand_origin(nd: int, ax: List[int], origin) -> List[int]:
    o = _seq_int(origin, len(ax))
    if len(ax) < nd:
        full = [0] * nd
        for i in range(len(ax)):
            full[ax[i]] = o[i]
        return full
    return o

def _check_origins(origins: List[int], fsh: List[int]):
    for i in range(len(fsh)):
        if fsh[i] // 2 + origins[i] < 0 or fsh[i] // 2 + origins[i] >= fsh[i]:
            raise ValueError('invalid origin')

def _correlate_nd(src: List[float], sh: List[int], fsh: List[int], w: List[float],
                  origins: List[int], mode: int, cval: float) -> List[float]:
    mask = [fabs(v) > _DBL_EPS for v in w]
    fp = _Footprint(fsh, mask, origins)
    ww = [w[k] for k in fp.flat]
    st = _strides(sh)
    n = _prod(sh)
    out = [0.0] * n
    offs = [0] * len(fp)
    coord = [0] * len(sh)
    for jj in range(n):
        _gather_offsets(fp, sh, st, coord, mode, offs)
        tmp = 0.0
        for i in range(len(offs)):
            o = offs[i]
            if o < 0:
                tmp += ww[i] * cval
            else:
                tmp += ww[i] * src[o]
        out[jj] = tmp
        _next_coord(coord, sh)
    return out

def _correlate_or_convolve(input, weights, output, mode, cval: float, origin, convolution: bool, axes):
    a = _arr(input)
    sh = _shape(a)
    nd = len(sh)
    ax = _check_axes(axes, nd)
    W = np.asarray(weights, dtype=float)
    wsh = _expand_fp(nd, ax, _shape(W), 'weights')
    origins = _expand_origin(nd, ax, origin)
    w = _fbuf(W)
    if convolution:
        w.reverse()
        for ii in range(nd):
            origins[ii] = -origins[ii]
            if not (wsh[ii] & 1):
                origins[ii] -= 1
    for ii in range(nd):
        if _invalid_origin(origins[ii], wsh[ii]):
            raise ValueError('Invalid origin; origin must satisfy -(weights.shape[k] // 2) <= origin[k] <= (weights.shape[k]-1) // 2')
    if not isinstance(mode, str):
        raise RuntimeError('A sequence of modes is not supported')
    r = _correlate_nd(_fbuf(a), sh, wsh, w, origins, _extend_mode_to_code(mode), cval)
    return _emit(r, a.shape, _proto(a, output), output)

def correlate(input, weights, output = None, mode = 'reflect', cval: float = 0.0, origin = 0, axes = None):
    return _correlate_or_convolve(input, weights, output, mode, cval, origin, False, axes)

def convolve(input, weights, output = None, mode = 'reflect', cval: float = 0.0, origin = 0, axes = None):
    return _correlate_or_convolve(input, weights, output, mode, cval, origin, True, axes)

def uniform_filter1d(input, size: int, axis: int = -1, output = None, mode: str = 'reflect',
                     cval: float = 0.0, origin: int = 0):
    a = _arr(input)
    sh = _shape(a)
    ax = _norm_axis(axis, len(sh))
    if size < 1:
        raise RuntimeError('incorrect filter size')
    if size // 2 + origin < 0 or size // 2 + origin >= size:
        raise ValueError('invalid origin')
    r = _uniform1d_buf(_fbuf(a), sh, ax, size, _extend_mode_to_code(mode), cval, origin)
    return _emit(r, a.shape, _proto(a, output), output)

def uniform_filter(input, size = 3, output = None, mode = 'reflect', cval: float = 0.0,
                   origin = 0, axes = None):
    a = _arr(input)
    sh = _shape(a)
    ax = _check_axes(axes, len(sh))
    na = len(ax)
    sizes = _seq_int(size, na)
    origins = _seq_int(origin, na)
    modes = _seq_str(mode, na)
    p = _proto(a, output)
    buf = _fbuf(a)
    for ii in range(na):
        if sizes[ii] > 1:
            if sizes[ii] // 2 + origins[ii] < 0 or sizes[ii] // 2 + origins[ii] >= sizes[ii]:
                raise ValueError('invalid origin')
            buf = _quantize(_uniform1d_buf(buf, sh, ax[ii], sizes[ii],
                                           _extend_mode_to_code(modes[ii]), cval, origins[ii]), p)
    return _emit(_quantize(buf, p), a.shape, p, output)

def _minmax1d(input, size: int, axis: int, output, mode: str, cval: float, origin: int, minimum: bool):
    a = _arr(input)
    sh = _shape(a)
    ax = _norm_axis(axis, len(sh))
    if size < 1:
        raise RuntimeError('incorrect filter size')
    if size // 2 + origin < 0 or size // 2 + origin >= size:
        raise ValueError('invalid origin')
    r = _minmax1d_buf(_fbuf(a), sh, ax, size, _extend_mode_to_code(mode), cval, origin, minimum)
    return _emit(r, a.shape, _proto(a, output), output)

def minimum_filter1d(input, size: int, axis: int = -1, output = None, mode: str = 'reflect',
                     cval: float = 0.0, origin: int = 0):
    return _minmax1d(input, size, axis, output, mode, cval, origin, True)

def maximum_filter1d(input, size: int, axis: int = -1, output = None, mode: str = 'reflect',
                     cval: float = 0.0, origin: int = 0):
    return _minmax1d(input, size, axis, output, mode, cval, origin, False)

def _minmax_nd(src: List[float], sh: List[int], fsh: List[int], mask: List[bool],
               ss: Optional[List[float]], origins: List[int], mode: int, cval: float,
               minimum: bool, proto) -> List[float]:
    # NI_MinOrMaxFilter: values are compared in the input type
    fp = _Footprint(fsh, mask, origins)
    sv = List[float]()
    if ss is not None:
        for k in fp.flat:
            sv.append(-ss[k] if minimum else ss[k])
    cv = _quantize([cval], proto)[0]
    st = _strides(sh)
    n = _prod(sh)
    out = [0.0] * n
    offs = [0] * len(fp)
    coord = [0] * len(sh)
    for jj in range(n):
        _gather_offsets(fp, sh, st, coord, mode, offs)
        res = cv if offs[0] < 0 else src[offs[0]]
        if ss is not None:
            res += sv[0]
        for i in range(1, len(offs)):
            t = cv if offs[i] < 0 else src[offs[i]]
            if ss is not None:
                t += _quantize([sv[i]], proto)[0]
            if minimum:
                if t < res:
                    res = t
            else:
                if t > res:
                    res = t
        out[jj] = res
        _next_coord(coord, sh)
    return _quantize(out, proto)

def _min_or_max_filter(input, size, footprint, structure, output, mode, cval: float,
                       origin, minimum: bool, axes):
    a = _arr(input)
    sh = _shape(a)
    nd = len(sh)
    ax = _check_axes(axes, nd)
    na = len(ax)
    p = _proto(a, output)
    separable = False
    fsh = List[int]()
    mask = List[bool]()
    ss: Optional[List[float]] = None
    sizes = List[int]()
    if structure is None:
        if footprint is None:
            if size is None:
                raise RuntimeError('no footprint provided')
            else:
                sizes = _seq_int(size, na)
            separable = True
        else:
            F = np.asarray(footprint)
            mask = _bool_buf(F)
            fsh = _shape(F)
            anyt = False
            allt = True
            for b in mask:
                anyt = anyt or b
                allt = allt and b
            if not anyt:
                raise ValueError('All-zero footprint is not supported.')
            if allt:
                sizes = list(fsh)
                separable = True
    else:
        S = np.asarray(structure, dtype=float)
        ss = _fbuf(S)
        if footprint is None:
            fsh = _shape(S)
            mask = [True] * len(ss)
        else:
            F = np.asarray(footprint)
            mask = _bool_buf(F)
            fsh = _shape(F)
    if separable:
        origins = _seq_int(origin, na)
        modes = _seq_str(mode, na)
        if len(sizes) != na:
            raise RuntimeError('sequence argument must have length equal to input rank')
        buf = _fbuf(a)
        for ii in range(na):
            if sizes[ii] > 1:
                if sizes[ii] // 2 + origins[ii] < 0 or sizes[ii] // 2 + origins[ii] >= sizes[ii]:
                    raise ValueError('invalid origin')
                buf = _quantize(_minmax1d_buf(buf, sh, ax[ii], sizes[ii],
                                              _extend_mode_to_code(modes[ii]), cval,
                                              origins[ii], minimum), p)
        return _emit(_quantize(buf, p), a.shape, p, output)
    fsh2 = _expand_fp(nd, ax, fsh, 'footprint')
    origins = _expand_origin(nd, ax, origin)
    _check_origins(origins, fsh2)
    if not isinstance(mode, str):
        raise RuntimeError('A sequence of modes is not supported for non-separable footprints')
    src = _quantize(_fbuf(a), p)
    r = _minmax_nd(src, sh, fsh2, mask, ss, origins, _extend_mode_to_code(mode), cval, minimum, p)
    return _emit(r, a.shape, p, output)

def minimum_filter(input, size = None, footprint = None, output = None, mode = 'reflect',
                   cval: float = 0.0, origin = 0, axes = None):
    return _min_or_max_filter(input, size, footprint, None, output, mode, cval, origin, True, axes)

def maximum_filter(input, size = None, footprint = None, output = None, mode = 'reflect',
                   cval: float = 0.0, origin = 0, axes = None):
    return _min_or_max_filter(input, size, footprint, None, output, mode, cval, origin, False, axes)

def _select_rank(buf: List[float], lo: int, hi: int, rank: int) -> float:
    # NI_Select (Hoare partition quickselect)
    while True:
        if lo == hi:
            return buf[lo]
        x = buf[lo]
        ii = lo - 1
        jj = hi + 1
        while True:
            jj -= 1
            while buf[jj] > x:
                jj -= 1
            ii += 1
            while buf[ii] < x:
                ii += 1
            if ii < jj:
                t = buf[ii]
                buf[ii] = buf[jj]
                buf[jj] = t
            else:
                break
        k = jj - lo + 1
        if rank < k:
            hi = jj
        else:
            lo = jj + 1
            rank -= k

def _rank_nd(src: List[float], sh: List[int], fsh: List[int], mask: List[bool],
             origins: List[int], mode: int, cval: float, rank: int) -> List[float]:
    fp = _Footprint(fsh, mask, origins)
    st = _strides(sh)
    n = _prod(sh)
    out = [0.0] * n
    offs = [0] * len(fp)
    buf = [0.0] * len(fp)
    coord = [0] * len(sh)
    for jj in range(n):
        _gather_offsets(fp, sh, st, coord, mode, offs)
        for i in range(len(offs)):
            buf[i] = cval if offs[i] < 0 else src[offs[i]]
        out[jj] = _select_rank(buf, 0, len(buf) - 1, rank)
        _next_coord(coord, sh)
    return out

def _rank_filter(input, rank, size, footprint, output, mode, cval: float, origin,
                 operation: str, axes):
    a = _arr(input)
    sh = _shape(a)
    nd = len(sh)
    ax = _check_axes(axes, nd)
    na = len(ax)
    fsh = List[int]()
    mask = List[bool]()
    if footprint is None:
        if size is None:
            raise RuntimeError('no footprint or filter size provided')
        else:
            fsh = _seq_int(size, na)
            mask = [True] * _prod(fsh)
    else:
        F = np.asarray(footprint)
        fsh = _shape(F)
        mask = _bool_buf(F)
    fsh2 = _expand_fp(nd, ax, fsh, 'footprint')
    origins = _expand_origin(nd, ax, origin)
    _check_origins(origins, fsh2)
    filter_size = 0
    for b in mask:
        if b:
            filter_size += 1
    rk = 0
    if operation == 'median':
        rk = filter_size // 2
    elif operation == 'percentile':
        pc = float(rank)
        if pc < 0.0:
            pc += 100.0
        if pc < 0 or pc > 100:
            raise RuntimeError('invalid percentile')
        if pc == 100.0:
            rk = filter_size - 1
        else:
            rk = int(float(filter_size) * pc / 100.0)
    else:
        rk = int(rank)
    if rk < 0:
        rk += filter_size
    if rk < 0 or rk >= filter_size:
        raise RuntimeError('rank not within filter footprint size')
    if not isinstance(mode, str):
        raise RuntimeError('A sequence of modes is not supported by non-separable rank filters')
    p = _proto(a, output)
    if rk == 0 or rk == filter_size - 1:
        src = _quantize(_fbuf(a), p)
        r = _minmax_nd(src, sh, fsh2, mask, None, origins, _extend_mode_to_code(mode),
                       cval, rk == 0, p)
        return _emit(r, a.shape, p, output)
    m = _extend_mode_to_code(mode, True)
    cv = _quantize([cval], a.dtype())[0]
    r = _rank_nd(_fbuf(a), sh, fsh2, mask, origins, m, cv, rk)
    return _emit(_quantize(r, a.dtype()), a.shape, p, output)

def rank_filter(input, rank: int, size = None, footprint = None, output = None,
                mode = 'reflect', cval: float = 0.0, origin = 0, axes = None):
    return _rank_filter(input, rank, size, footprint, output, mode, cval, origin, 'rank', axes)

def median_filter(input, size = None, footprint = None, output = None, mode = 'reflect',
                  cval: float = 0.0, origin = 0, axes = None):
    return _rank_filter(input, 0, size, footprint, output, mode, cval, origin, 'median', axes)

def percentile_filter(input, percentile: float, size = None, footprint = None, output = None,
                      mode = 'reflect', cval: float = 0.0, origin = 0, axes = None):
    return _rank_filter(input, percentile, size, footprint, output, mode, cval, origin,
                        'percentile', axes)

def generic_filter1d(input, function, filter_size: int, axis: int = -1, output = None,
                     mode: str = 'reflect', cval: float = 0.0, origin: int = 0,
                     extra_arguments = ()):
    # `function(input_line, output_line, *extra_arguments)` fills output_line in place
    a = _arr(input)
    sh = _shape(a)
    ax = _norm_axis(axis, len(sh))
    if filter_size < 1:
        raise RuntimeError('invalid filter size')
    if filter_size // 2 + origin < 0 or filter_size // 2 + origin >= filter_size:
        raise ValueError('invalid origin')
    size1 = filter_size // 2
    size2 = filter_size - size1 - 1
    before = size1 + origin
    after = size2 - origin
    src = _fbuf(a)
    out = [0.0] * len(src)
    L = sh[ax]
    st = _strides(sh)[ax]
    line = [0.0] * (before + L + after)
    m = _extend_mode_to_code(mode)
    for base in _line_bases(sh, ax):
        for k in range(L):
            line[before + k] = src[base + k * st]
        _extend_line(line, before, L, after, m, cval)
        iline = np.empty(len(line), dtype=float)
        for k in range(len(line)):
            iline[k] = line[k]
        oline = np.zeros(L, dtype=float)
        function(iline, oline, *extra_arguments)
        for k in range(L):
            out[base + k * st] = float(oline[k])
    return _emit(out, a.shape, _proto(a, output), output)

def generic_filter(input, function, size = None, footprint = None, output = None,
                   mode: str = 'reflect', cval: float = 0.0, origin = 0,
                   extra_arguments = (), axes = None):
    # `function(values, *extra_arguments) -> float` receives a 1-D float
    # array of the footprint values in C order
    a = _arr(input)
    sh = _shape(a)
    nd = len(sh)
    ax = _check_axes(axes, nd)
    fsh = List[int]()
    mask = List[bool]()
    if footprint is None:
        if size is None:
            raise RuntimeError('no footprint or filter size provided')
        else:
            fsh = _seq_int(size, len(ax))
            mask = [True] * _prod(fsh)
    else:
        F = np.asarray(footprint)
        fsh = _shape(F)
        mask = _bool_buf(F)
    fsh2 = _expand_fp(nd, ax, fsh, 'footprint')
    origins = _expand_origin(nd, ax, origin)
    _check_origins(origins, fsh2)
    fp = _Footprint(fsh2, mask, origins)
    src = _fbuf(a)
    st = _strides(sh)
    n = _prod(sh)
    out = [0.0] * n
    offs = [0] * len(fp)
    coord = [0] * nd
    m = _extend_mode_to_code(mode)
    for jj in range(n):
        _gather_offsets(fp, sh, st, coord, m, offs)
        vals = np.empty(len(offs), dtype=float)
        for i in range(len(offs)):
            vals[i] = cval if offs[i] < 0 else src[offs[i]]
        out[jj] = float(function(vals, *extra_arguments))
        _next_coord(coord, sh)
    return _emit(out, a.shape, _proto(a, output), output)

def generic_laplace(input, derivative2, output = None, mode = 'reflect', cval: float = 0.0,
                    extra_arguments = (), extra_keywords = None, *, axes = None):
    # derivative2(input, axis, output, mode, cval, *extra_arguments) -> array
    a = _arr(input)
    sh = _shape(a)
    ax = _check_axes(axes, len(sh))
    p = _proto(a, output)
    if len(ax) == 0:
        return _emit(_quantize(_fbuf(a), p), a.shape, p, output)
    modes = _seq_str(mode, len(ax))
    acc = _quantize(_fbuf(np.asarray(derivative2(a, ax[0], None, modes[0], cval, *extra_arguments))), p)
    for ii in range(1, len(ax)):
        t = _quantize(_fbuf(np.asarray(derivative2(a, ax[ii], None, modes[ii], cval, *extra_arguments))), p)
        for k in range(len(acc)):
            acc[k] += t[k]
        acc = _quantize(acc, p)
    return _emit(acc, a.shape, p, output)

def generic_gradient_magnitude(input, derivative, output = None, mode = 'reflect',
                               cval: float = 0.0, extra_arguments = (), extra_keywords = None,
                               *, axes = None):
    a = _arr(input)
    sh = _shape(a)
    ax = _check_axes(axes, len(sh))
    p = _proto(a, output)
    if len(ax) == 0:
        return _emit(_quantize(_fbuf(a), p), a.shape, p, output)
    modes = _seq_str(mode, len(ax))
    acc = _quantize(_fbuf(np.asarray(derivative(a, ax[0], None, modes[0], cval, *extra_arguments))), p)
    for k in range(len(acc)):
        acc[k] = acc[k] * acc[k]
    acc = _quantize(acc, p)
    for ii in range(1, len(ax)):
        t = _quantize(_fbuf(np.asarray(derivative(a, ax[ii], None, modes[ii], cval, *extra_arguments))), p)
        for k in range(len(acc)):
            acc[k] += _quantize([t[k] * t[k]], p)[0]
        acc = _quantize(acc, p)
    for k in range(len(acc)):
        acc[k] = sqrt(acc[k])
    return _emit(acc, a.shape, p, output)

def vectorized_filter(input, function, *, size = None, footprint = None, output = None,
                      mode = 'reflect', cval = None, origin = None, axes = None,
                      batch_memory = 1 << 30):
    """Apply `function(window_values, axis=-1)` over every neighbourhood
    (flattened footprint values, C order). Equivalent to scipy's
    sliding-window formulation for reducing functions."""
    a = _arr(input)
    sh = _shape(a)
    nd = len(sh)
    ax = _check_axes(axes, nd)
    fsh = List[int]()
    mask = List[bool]()
    if footprint is None:
        if size is None:
            raise ValueError('Either `size` or `footprint` must be provided.')
        fsh = _seq_int(size, len(ax))
        mask = [True] * _prod(fsh)
    else:
        F = np.asarray(footprint)
        fsh = _shape(F)
        mask = _bool_buf(F)
    fsh2 = _expand_fp(nd, ax, fsh, 'footprint')
    origins = _expand_origin(nd, ax, 0 if origin is None else origin)
    _check_origins(origins, fsh2)
    # vectorized_filter centres even-sized windows one step to the left of
    # generic_filter (borders i//2 + o before, (i-1)//2 - o after)
    org2 = [0] * nd
    for i in range(nd):
        org2[i] = origins[i] + (0 if fsh2[i] % 2 == 1 else 0)
    cv = 0.0 if cval is None else float(cval)
    fp = _Footprint(fsh2, mask, org2)
    src = _fbuf(a)
    st = _strides(sh)
    n = _prod(sh)
    out = [0.0] * n
    offs = [0] * len(fp)
    coord = [0] * nd
    m = _extend_mode_to_code(mode)
    for jj in range(n):
        _gather_offsets(fp, sh, st, coord, m, offs)
        vals = np.empty(len(offs), dtype=float)
        for i in range(len(offs)):
            vals[i] = cv if offs[i] < 0 else src[offs[i]]
        out[jj] = float(function(vals, axis=-1))
        _next_coord(coord, sh)
    return _emit(out, a.shape, _proto(a, output), output)
