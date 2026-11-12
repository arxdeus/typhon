# Ported from scipy 1.18.1 (BSD-3-Clause)

# Spline interpolation (scipy.ndimage._interpolation): spline prefilters
# (ni_splines.c), NI_GeometricTransform / NI_ZoomShift with exact boundary
# mapping, and zoom/shift/rotate/affine_transform/map_coordinates.

import numpy as np
import internal.static as static
from math import floor, pow, sin, cos, pi as _PI, fabs
from ._ni_support import _extend_mode_to_code, _seq_float, _shape, _strides, _prod, \
    _fbuf, _line_bases, _next_coord, _proto, _emit, _tuple_shape, \
    NEAREST, WRAP, REFLECT, MIRROR, CONSTANT, GRID_WRAP, GRID_CONSTANT

# ---- spline prefilter ------------------------------------------------------------

def _poles(order: int) -> List[float]:
    if order == 2:
        return [-0.171572875253809902396622551580603843]
    elif order == 3:
        return [-0.267949192431122706472553658494127633]
    elif order == 4:
        return [-0.361341225900220177092212841325675255, -0.013725429297339121360331226939128204]
    elif order == 5:
        return [-0.430575347099973791851434783493520110, -0.043096288203264653822712376822550182]
    return List[float]()

def _apply_filter(c: List[float], n: int, z: float, kind: int):
    # kind 0: mirror, 1: wrap, 2: reflect
    if kind == 0:
        z_i = z
        z_n_1 = pow(z, float(n - 1))
        c[0] = c[0] + z_n_1 * c[n - 1]
        for i in range(1, n - 1):
            c[0] += z_i * (c[i] + z_n_1 * c[n - 1 - i])
            z_i *= z
        c[0] /= 1 - z_n_1 * z_n_1
    elif kind == 1:
        z_i = z
        for i in range(1, n):
            c[0] += z_i * c[n - i]
            z_i *= z
        c[0] /= 1 - z_i
    else:
        z_i = z
        z_n = pow(z, float(n))
        s = c[0] + z_n * c[n - 1]
        for i in range(1, n):
            s += z_i * (c[i] + z_n * c[n - 1 - i])
            z_i *= z
        c[0] += s * z / (1 - z_n * z_n)
    for i in range(1, n):
        c[i] += z * c[i - 1]
    if kind == 0:
        c[n - 1] = (z * c[n - 2] + c[n - 1]) * z / (z * z - 1)
    elif kind == 1:
        z_i = z
        for i in range(n - 1):
            c[n - 1] += z_i * c[i]
            z_i *= z
        c[n - 1] *= z / (z_i - 1)
    else:
        c[n - 1] *= z / (z - 1)
    for i in range(n - 2, -1, -1):
        c[i] = z * (c[i + 1] - c[i])

def _spline_line(c: List[float], order: int, mode: int):
    n = len(c)
    kind = 0
    if mode == GRID_WRAP:
        kind = 1
    elif mode == NEAREST or mode == REFLECT:
        kind = 2
    poles = _poles(order)
    gain = 1.0
    for z in poles:
        gain *= (1.0 - z) * (1.0 - 1.0 / z)
    for i in range(n):
        c[i] *= gain
    for z in poles:
        _apply_filter(c, n, z, kind)

def _spline_filter1d_buf(src: List[float], sh: List[int], axis: int, order: int, mode: int) -> List[float]:
    out = list(src)
    L = sh[axis]
    if L <= 1:
        return out
    st = _strides(sh)[axis]
    line = [0.0] * L
    for base in _line_bases(sh, axis):
        for k in range(L):
            line[k] = out[base + k * st]
        _spline_line(line, order, mode)
        for k in range(L):
            out[base + k * st] = line[k]
    return out

def spline_filter1d(input, order: int = 3, axis: int = -1, output = float, mode: str = 'mirror'):
    if order < 0 or order > 5:
        raise RuntimeError('spline order not supported')
    a = np.asarray(input)
    sh = _shape(a)
    nd = len(sh)
    if axis < -nd or axis >= nd:
        raise ValueError(f"axis {axis} is out of bounds for array of dimension {nd}")
    buf = _fbuf(a)
    if order > 1:
        buf = _spline_filter1d_buf(buf, sh, axis % nd, order, _extend_mode_to_code(mode))
    return _emit(buf, a.shape, _proto(a, output), output)

def _spline_filter_buf(buf: List[float], sh: List[int], order: int, mode: int) -> List[float]:
    for ax in range(len(sh)):
        buf = _spline_filter1d_buf(buf, sh, ax, order, mode)
    return buf

def spline_filter(input, order: int = 3, output = float, mode: str = 'mirror'):
    if order < 2 or order > 5:
        raise RuntimeError('spline order not supported')
    a = np.asarray(input)
    sh = _shape(a)
    buf = _spline_filter_buf(_fbuf(a), sh, order, _extend_mode_to_code(mode))
    return _emit(buf, a.shape, _proto(a, output), output)

# ---- interpolation core ------------------------------------------------------------

def _map_coordinate(x: float, n: int, mode: int) -> float:
    inn = x
    if inn < 0:
        if mode == MIRROR:
            if n <= 1:
                inn = 0.0
            else:
                sz2 = 2 * n - 2
                inn = sz2 * int(-inn / sz2) + inn
                inn = inn + sz2 if inn <= 1 - n else -inn
        elif mode == REFLECT:
            if n <= 1:
                inn = 0.0
            else:
                sz2 = 2 * n
                if inn < -sz2:
                    inn = sz2 * int(-inn / sz2) + inn
                if inn < -n:
                    inn = inn + sz2
                else:
                    inn = (1e-15 if inn > -1e-15 else -inn) - 1
        elif mode == WRAP:
            if n <= 1:
                inn = 0.0
            else:
                sz = n - 1
                inn += sz * (int(-inn / sz) + 1)
        elif mode == GRID_WRAP:
            if n <= 1:
                inn = 0.0
            else:
                inn += n * (int((-1 - inn) / n) + 1)
        elif mode == NEAREST:
            inn = 0.0
        elif mode == CONSTANT:
            inn = -1.0
    elif inn > n - 1:
        if mode == MIRROR:
            if n <= 1:
                inn = 0.0
            else:
                sz2 = 2 * n - 2
                inn -= sz2 * int(inn / sz2)
                if inn >= n:
                    inn = sz2 - inn
        elif mode == REFLECT:
            if n <= 1:
                inn = 0.0
            else:
                sz2 = 2 * n
                inn -= sz2 * int(inn / sz2)
                if inn >= n:
                    inn = sz2 - inn - 1
        elif mode == WRAP:
            if n <= 1:
                inn = 0.0
            else:
                sz = n - 1
                inn -= sz * int(inn / sz)
        elif mode == GRID_WRAP:
            if n <= 1:
                inn = 0.0
            else:
                inn -= n * int(inn / n)
        elif mode == NEAREST:
            inn = float(n - 1)
        elif mode == CONSTANT:
            inn = -1.0
    return inn

def _spline_weights(x: float, order: int, w: List[float]):
    x -= floor(x if (order & 1) else x + 0.5)
    y = x
    z = 1.0 - x
    if order == 1:
        w[0] = 1.0 - x
    elif order == 2:
        w[1] = 0.75 - x * x
        y = 0.5 - x
        w[0] = 0.5 * y * y
    elif order == 3:
        w[1] = (y * y * (y - 2.0) * 3.0 + 4.0) / 6.0
        w[2] = (z * z * (z - 2.0) * 3.0 + 4.0) / 6.0
        w[0] = z * z * z / 6.0
    elif order == 4:
        t = x * x
        w[2] = t * (t * 0.25 - 0.625) + 115.0 / 192.0
        y = 1.0 + x
        w[1] = y * (y * (y * (5.0 - y) / 6.0 - 1.25) + 5.0 / 24.0) + 55.0 / 96.0
        w[3] = z * (z * (z * (5.0 - z) / 6.0 - 1.25) + 5.0 / 24.0) + 55.0 / 96.0
        y = 0.5 - x
        t = y * y
        w[0] = t * t / 24.0
    elif order == 5:
        t = y * y
        w[2] = t * (t * (0.25 - y / 12.0) - 0.5) + 0.55
        t = z * z
        w[3] = t * (t * (0.25 - z / 12.0) - 0.5) + 0.55
        y += 1.0
        w[1] = y * (y * (y * (y * (y / 24.0 - 0.375) + 1.25) - 1.75) + 0.625) + 0.425
        z += 1.0
        w[4] = z * (z * (z * (z * (z / 24.0 - 0.375) + 1.25) - 1.75) + 0.625) + 0.425
        y = 1.0 - x
        t = y * y
        w[0] = y * t * t / 120.0
    w[order] = 1.0
    for i in range(order):
        w[order] -= w[i]

def _spline_mode(mode: int) -> int:
    if mode == CONSTANT or mode == WRAP:
        return MIRROR
    return mode

class _AxisPlan:
    # per-axis interpolation data for one input coordinate
    const: bool
    start: int
    idx: List[int]       # input index per filter tap (already mapped)
    out: List[bool]      # grid-constant: tap outside
    w: List[float]

    def __init__(self, order: int):
        self.const = False
        self.start = 0
        self.idx = [0] * (order + 1)
        self.out = [False] * (order + 1)
        self.w = [1.0] * (order + 1)

def _plan(p: _AxisPlan, cc0: float, n: int, order: int, mode: int, nprepad: int, smode: int):
    cc = cc0 + nprepad
    if mode != GRID_CONSTANT and mode != NEAREST:
        cc = _map_coordinate(cc, n, mode)
    if cc > -1.0 or mode == GRID_CONSTANT or mode == NEAREST:
        p.const = False
        start = 0
        if order & 1:
            start = int(floor(cc)) - order // 2
        else:
            start = int(floor(cc + 0.5)) - order // 2
        p.start = start
        edge = start < 0 or start + order >= n
        for ll in range(order + 1):
            ix = start + ll
            p.out[ll] = False
            if mode == GRID_CONSTANT:
                p.out[ll] = ix < 0 or ix >= n
                p.idx[ll] = ix
            elif edge:
                p.idx[ll] = int(_map_coordinate(float(ix), n, smode))
            else:
                p.idx[ll] = ix
        if order != 0:
            _spline_weights(cc, order, p.w)
    else:
        p.const = True

def _interp_point(data: List[float], ist: List[int], plans: List[_AxisPlan], order: int,
                  mode: int, cval: float, tap: List[int]) -> float:
    rank = len(plans)
    for h in range(rank):
        if plans[h].const:
            return cval
    fsize = 1
    for h in range(rank):
        fsize *= order + 1
    for h in range(rank):
        tap[h] = 0
    t = 0.0
    for hh in range(fsize):
        is_cval = False
        if mode == GRID_CONSTANT:
            for ll in range(rank):
                if plans[ll].out[tap[ll]]:
                    is_cval = True
        coeff = 0.0
        if is_cval:
            coeff = cval
        else:
            idx = 0
            for ll in range(rank):
                idx += plans[ll].idx[tap[ll]] * ist[ll]
            coeff = data[idx]
        if order > 0:
            for ll in range(rank):
                coeff *= plans[ll].w[tap[ll]]
        t += coeff
        for j in range(rank - 1, -1, -1):
            if tap[j] < order:
                tap[j] += 1
                break
            tap[j] = 0
    return t

def _prefiltered(a, order: int, mode: str, cval: float, prefilter: bool) -> Tuple[List[float], List[int], int]:
    sh = _shape(a)
    if prefilter and order > 1:
        npad = 0
        buf = _fbuf(a)
        if mode == 'nearest' or mode == 'grid-constant':
            npad = 12
            psh = [s + 2 * npad for s in sh]
            pst = _strides(psh)
            pb = [cval] * _prod(psh)
            n = len(buf)
            if mode == 'nearest':
                idx = [0] * len(sh)
                for q in range(_prod(psh)):
                    off = 0
                    st = _strides(sh)
                    for d in range(len(sh)):
                        c = idx[d] - npad
                        if c < 0:
                            c = 0
                        if c >= sh[d]:
                            c = sh[d] - 1
                        off += c * st[d]
                    pb[q] = buf[off]
                    _next_coord(idx, psh)
            else:
                idx = [0] * len(sh)
                st = _strides(sh)
                for q in range(n):
                    off = 0
                    for d in range(len(sh)):
                        off += (idx[d] + npad) * pst[d]
                    pb[off] = buf[q]
                    _next_coord(idx, sh)
            buf = pb
            sh = psh
        buf = _spline_filter_buf(buf, sh, order, _extend_mode_to_code(mode))
        return (buf, sh, npad)
    return (_fbuf(a), sh, 0)

def _geometric(data: List[float], ish: List[int], osh: List[int], coords_fn, order: int,
               mode: int, cval: float, nprepad: int) -> List[float]:
    rank = len(ish)
    ist = _strides(ish)
    n = _prod(osh)
    out = [0.0] * n
    plans = [_AxisPlan(order) for _ in range(rank)]
    smode = _spline_mode(mode)
    oc = [0] * len(osh)
    ic = [0.0] * rank
    tap = [0] * rank
    for k in range(n):
        coords_fn(k, oc, ic)
        for h in range(rank):
            _plan(plans[h], ic[h], ish[h], order, mode, nprepad, smode)
        out[k] = _interp_point(data, ist, plans, order, mode, cval, tap)
        _next_coord(oc, osh)
    return out

def _check_order(order: int):
    if order < 0 or order > 5:
        raise RuntimeError('spline order not supported')

def map_coordinates(input, coordinates, output = None, order: int = 3, mode: str = 'constant',
                    cval: float = 0.0, prefilter: bool = True):
    _check_order(order)
    a = np.asarray(input)
    C = np.ascontiguousarray(np.asarray(coordinates, dtype=float))
    csh = _shape(C)
    rank = static.len(a.shape)
    if csh[0] != rank:
        raise RuntimeError('invalid shape for coordinate array')
    osh = csh[1:]
    no = _prod(osh)
    cbuf = _fbuf(C)
    pf = _prefiltered(a, order, mode, cval, prefilter)
    def cfn(k: int, oc: List[int], ic: List[float]):
        for h in range(len(ic)):
            ic[h] = cbuf[h * no + k]
    r = _geometric(pf[0], pf[1], osh, cfn, order, _extend_mode_to_code(mode), cval, pf[2])
    return _emit(r, C.shape[1:], _proto(a, output), output, True)

def geometric_transform(input, mapping, output_shape = None, output = None, order: int = 3,
                        mode: str = 'constant', cval: float = 0.0, prefilter: bool = True,
                        extra_arguments = ()):
    # mapping(output_coordinate_tuple, *extra_arguments) -> input coordinate tuple
    _check_order(order)
    a = np.asarray(input)
    pf = _prefiltered(a, order, mode, cval, prefilter)
    osh = _shape(a)
    if output_shape is not None:
        osh = [int(s) for s in output_shape]
    rank = static.len(a.shape)
    def cfn(k: int, oc: List[int], ic: List[float]):
        t = _tuple_shape(oc, a.shape)
        r = mapping(t, *extra_arguments)
        h = 0
        for v in r:
            ic[h] = float(v)
            h += 1
    r = _geometric(pf[0], pf[1], osh, cfn, order, _extend_mode_to_code(mode), cval, pf[2])
    return _emit(r, _tuple_shape(osh, a.shape), _proto(a, output), output, True)

def _zoom_shift(data: List[float], ish: List[int], osh: List[int], zooms: Optional[List[float]],
                shifts: Optional[List[float]], order: int, mode: int, cval: float,
                nprepad: int, grid_mode: bool) -> List[float]:
    rank = len(ish)
    smode = _spline_mode(mode)
    # precompute per-axis plans for every output index
    plans = List[List[_AxisPlan]]()
    for jj in range(rank):
        pl = List[_AxisPlan]()
        for kk in range(osh[jj]):
            cc = float(kk)
            if shifts is not None:
                cc += shifts[jj]
            if zooms is not None:
                if grid_mode:
                    cc += 0.5
                    cc *= zooms[jj]
                    cc -= 0.5
                else:
                    cc *= zooms[jj]
            p = _AxisPlan(order)
            if mode == CONSTANT:
                # NI_ZoomShift flags whole output rows as constant
                c2 = cc + nprepad
                c2 = _map_coordinate(c2, ish[jj], mode)
                _plan(p, cc, ish[jj], order, mode, nprepad, smode)
                if not (c2 > -1.0):
                    p.const = True
            else:
                _plan(p, cc, ish[jj], order, mode, nprepad, smode)
            pl.append(p)
        plans.append(pl)
    ist = _strides(ish)
    n = _prod(osh)
    out = [0.0] * n
    oc = [0] * rank
    cur = List[_AxisPlan](rank)
    for jj in range(rank):
        cur.append(plans[jj][0])
    tap = [0] * rank
    for k in range(n):
        for jj in range(rank):
            cur[jj] = plans[jj][oc[jj]]
        out[k] = _interp_point(data, ist, cur, order, mode, cval, tap)
        _next_coord(oc, osh)
    return out

def shift(input, shift, output = None, order: int = 3, mode: str = 'constant',
          cval: float = 0.0, prefilter: bool = True):
    _check_order(order)
    a = np.asarray(input)
    rank = static.len(a.shape)
    pf = _prefiltered(a, order, mode, cval, prefilter)
    sh = [-v for v in _seq_float(shift, rank)]
    r = _zoom_shift(pf[0], pf[1], _shape(a), None, sh, order, _extend_mode_to_code(mode),
                    cval, pf[2], False)
    return _emit(r, a.shape, _proto(a, output), output, True)

def zoom(input, zoom, output = None, order: int = 3, mode: str = 'constant', cval: float = 0.0,
         prefilter: bool = True, grid_mode: bool = False):
    _check_order(order)
    a = np.asarray(input)
    ish = _shape(a)
    rank = len(ish)
    zs = _seq_float(zoom, rank)
    osh = [int(round(ish[i] * zs[i])) for i in range(rank)]
    allone = True
    for z in zs:
        if z != 1.0:
            allone = False
    if allone and prefilter:
        return _emit(_fbuf(a), a.shape, _proto(a, output), output, True)
    pf = _prefiltered(a, order, mode, cval, prefilter)
    zr = [1.0] * rank
    for i in range(rank):
        num = ish[i] if grid_mode else ish[i] - 1
        div = osh[i] if grid_mode else osh[i] - 1
        if div != 0:
            zr[i] = num / div
    r = _zoom_shift(pf[0], pf[1], osh, zr, None, order, _extend_mode_to_code(mode), cval,
                    pf[2], grid_mode)
    return _emit(r, _tuple_shape(osh, a.shape), _proto(a, output), output, True)

def _affine_core(a, M: List[List[float]], off: List[float], osh: List[int], output, order: int,
                 mode: str, cval: float, prefilter: bool):
    pf = _prefiltered(a, order, mode, cval, prefilter)
    irank = len(M)
    orank = len(osh)
    def cfn(k: int, oc: List[int], ic: List[float]):
        for hh in range(irank):
            tmp = off[hh]
            row = M[hh]
            ll = 0
            while ll + 1 < orank:
                tmp += oc[ll] * row[ll]
                tmp += oc[ll + 1] * row[ll + 1]
                ll += 2
            if ll < orank:
                tmp += oc[ll] * row[ll]
            ic[hh] = tmp
    r = _geometric(pf[0], pf[1], osh, cfn, order, _extend_mode_to_code(mode), cval, pf[2])
    return r

def affine_transform(input, matrix, offset = 0.0, output_shape = None, output = None,
                     order: int = 3, mode: str = 'constant', cval: float = 0.0,
                     prefilter: bool = True):
    _check_order(order)
    a = np.asarray(input)
    ish = _shape(a)
    rank = len(ish)
    osh = list(ish)
    if output_shape is not None:
        osh = [int(s) for s in output_shape]
    Mt = np.asarray(matrix, dtype=float)
    if static.len(Mt.shape) == 1:
        if Mt.shape[0] != rank:
            raise RuntimeError('affine matrix has wrong number of rows')
        offs = _seq_float(offset, rank)
        zs = [float(Mt[i]) for i in range(rank)]
        sh = [offs[i] / zs[i] for i in range(rank)]
        pf = _prefiltered(a, order, mode, cval, prefilter)
        r = _zoom_shift(pf[0], pf[1], osh, zs, sh, order, _extend_mode_to_code(mode), cval,
                        pf[2], False)
        return _emit(r, _tuple_shape(osh, a.shape), _proto(a, output), output, True)
    else:
        nr = Mt.shape[0]
        nc = Mt.shape[1]
        offs = List[float]()
        M = List[List[float]]()
        if nc == rank + 1 and (nr == rank or nr == rank + 1):
            if nr == rank + 1:
                for j in range(rank + 1):
                    exp = 1.0 if j == rank else 0.0
                    if Mt[rank, j] != exp:
                        raise ValueError('Expected homogeneous transformation matrix')
            for i in range(rank):
                offs.append(float(Mt[i, rank]))
                M.append([float(Mt[i, j]) for j in range(rank)])
        else:
            if nr != rank:
                raise RuntimeError('affine matrix has wrong number of rows')
            if nc != len(osh):
                raise RuntimeError('affine matrix has wrong number of columns')
            offs = _seq_float(offset, rank)
            for i in range(rank):
                M.append([float(Mt[i, j]) for j in range(nc)])
        r = _affine_core(a, M, offs, osh, output, order, mode, cval, prefilter)
        return _emit(r, _tuple_shape(osh, a.shape), _proto(a, output), output, True)

def _cosdg(x: float) -> float:
    # degree-argument cosine with exact values at multiples of 90 (cephes cosdg)
    r = x % 360.0
    if r == 0.0:
        return 1.0
    if r == 90.0 or r == 270.0:
        return 0.0
    if r == 180.0:
        return -1.0
    return cos(x * (_PI / 180.0))

def _sindg(x: float) -> float:
    r = x % 360.0
    if r == 0.0 or r == 180.0:
        return 0.0 if x >= 0 else -0.0
    if r == 90.0:
        return 1.0
    if r == 270.0:
        return -1.0
    return sin(x * (_PI / 180.0))

def rotate(input, angle: float, axes = (1, 0), reshape: bool = True, output = None,
           order: int = 3, mode: str = 'constant', cval: float = 0.0, prefilter: bool = True):
    a = np.asarray(input)
    ish = _shape(a)
    nd = len(ish)
    if nd < 2:
        raise ValueError('input array should be at least 2D')
    ax = [int(v) for v in axes]
    if len(ax) != 2:
        raise ValueError('axes should contain exactly two values')
    for i in range(2):
        if ax[i] < 0:
            ax[i] += nd
    if ax[0] < 0 or ax[1] < 0 or ax[0] >= nd or ax[1] >= nd:
        raise ValueError('invalid rotation plane specified')
    ax.sort()
    c = _cosdg(angle)
    s = _sindg(angle)
    rm = [[c, s], [-s, c]]
    iy = ish[ax[0]]
    ix = ish[ax[1]]
    oy = iy
    ox = ix
    if reshape:
        xs = [0.0, 0.0, float(iy), float(iy)]
        ys = [0.0, float(ix), 0.0, float(ix)]
        b0 = [rm[0][0] * xs[k] + rm[0][1] * ys[k] for k in range(4)]
        b1 = [rm[1][0] * xs[k] + rm[1][1] * ys[k] for k in range(4)]
        oy = int((max(b0) - min(b0)) + 0.5)
        ox = int((max(b1) - min(b1)) + 0.5)
    oc0 = (oy - 1) / 2
    oc1 = (ox - 1) / 2
    out_c = [rm[0][0] * oc0 + rm[0][1] * oc1, rm[1][0] * oc0 + rm[1][1] * oc1]
    in_c = [(iy - 1) / 2, (ix - 1) / 2]
    offs = [in_c[0] - out_c[0], in_c[1] - out_c[1]]
    osh = list(ish)
    osh[ax[0]] = oy
    osh[ax[1]] = ox
    p = _proto(a, output)
    if nd == 2:
        r = _affine_core(a, rm, offs, osh, output, order, mode, cval, prefilter)
        return _emit(r, _tuple_shape(osh, a.shape), p, output, True)
    # rotate each plane spanned by `ax`
    buf = _fbuf(a)
    ist = _strides(ish)
    ost = _strides(osh)
    res = [0.0] * _prod(osh)
    others = [d for d in range(nd) if d != ax[0] and d != ax[1]]
    osub = [ish[d] for d in others]
    idx = [0] * len(others)
    for q in range(_prod(osub)):
        ib = 0
        ob = 0
        for k in range(len(others)):
            ib += idx[k] * ist[others[k]]
            ob += idx[k] * ost[others[k]]
        plane = np.empty((iy, ix), dtype=float)
        for y in range(iy):
            for x in range(ix):
                plane[y, x] = buf[ib + y * ist[ax[0]] + x * ist[ax[1]]]
        r = _affine_core(plane, rm, offs, [oy, ox], None, order, mode, cval, prefilter)
        for y in range(oy):
            for x in range(ox):
                res[ob + y * ost[ax[0]] + x * ost[ax[1]]] = r[y * ox + x]
        _next_coord(idx, osub)
    return _emit(res, _tuple_shape(osh, a.shape), p, output, True)
