# Ported from scipy 1.18.1 (BSD-3-Clause)

# Binary and grey-scale morphology and distance transforms
# (scipy.ndimage._morphology). Binary erosion follows NI_BinaryErosion;
# iterated operations are repeated full passes (same result as scipy's
# coordinate-list acceleration). distance_transform_edt is the
# Maurer/Qi/Raghavan feature transform of NI_EuclideanFeatureTransform.

import numpy as np
import internal.static as static
from math import sqrt
from ._ni_support import _seq_int, _check_axes, _shape, _strides, _prod, \
    _Footprint, _gather_offsets, _next_coord, _tuple_shape, _map_index
from ._filters import _min_or_max_filter, _bool_buf, _expand_fp, _expand_origin

def generate_binary_structure(rank: Literal[int], connectivity: int):
    t = _bstruct(rank, connectivity)
    out = np.empty((3,) * rank, dtype=bool)
    for k in range(len(t[1])):
        out.data[k] = t[1][k]
    return out

def _bstruct(rank: int, connectivity: int) -> Tuple[List[int], List[bool]]:
    sh = [3] * rank
    n = _prod(sh)
    m = List[bool](n)
    idx = [0] * rank
    for k in range(n):
        s = 0
        for d in range(rank):
            s += abs(idx[d] - 1)
        m.append(s <= connectivity)
        _next_coord(idx, sh)
    return (sh, m)

def _bool_out(buf: List[bool], shape, output):
    res = np.empty(shape, dtype=bool)
    for i in range(len(buf)):
        res.data[i] = buf[i]
    if output is None:
        return res
    elif isinstance(output, type):
        if output is bool:
            return res
        else:
            return res.astype(output)
    else:
        output[...] = res
        return output

def _erode_pass(src: List[bool], sh: List[int], fp: _Footprint, maskv: Optional[List[bool]],
                border_value: int, invert: bool, cit: bool) -> Tuple[List[bool], bool]:
    st = _strides(sh)
    n = _prod(sh)
    out = List[bool](n)
    offs = [0] * len(fp)
    coord = [0] * len(sh)
    bdr = 0
    tr = True
    fl = False
    if invert:
        bdr = 0 if border_value else 1
        tr = False
        fl = True
    else:
        bdr = 1 if border_value else 0
    changed = False
    for jj in range(n):
        inn = src[jj]
        mv = True
        if maskv is not None:
            mv = maskv[jj]
        o = inn
        if mv:
            if cit and inn == fl:
                o = inn
            else:
                _gather_offsets(fp, sh, st, coord, 4, offs)
                o = tr
                for i in range(len(offs)):
                    if offs[i] < 0:
                        if not bdr:
                            o = fl
                            break
                    else:
                        nn = tr if src[offs[i]] else fl
                        if not nn:
                            o = fl
                            break
                if o != inn:
                    changed = True
        out.append(o)
        _next_coord(coord, sh)
    return (out, changed)

def _binary_erosion(input, structure, iterations: int, mask, output, border_value: int,
                    origin, invert: bool, axes, reverse: bool):
    a = np.asarray(input)
    sh = _shape(a)
    nd = len(sh)
    ax = _check_axes(axes, nd)
    na = len(ax)
    ssh = List[int]()
    sm = List[bool]()
    if structure is None:
        t = _bstruct(na, 1)
        ssh = t[0]
        sm = t[1]
    else:
        S = np.asarray(structure)
        ssh = _shape(S)
        sm = _bool_buf(S)
    if len(sm) < 1:
        raise RuntimeError('structure must not be empty')
    org = _seq_int(origin, na)
    if reverse:
        sm.reverse()
        for ii in range(na):
            org[ii] = -org[ii]
            if not (ssh[ii] & 1):
                org[ii] -= 1
    fsh = _expand_fp(nd, ax, ssh, 'structure')
    origins = _expand_origin(nd, ax, org)
    # center_is_true
    cidx = 0
    st = _strides(fsh)
    for d in range(nd):
        cidx += (origins[d] + fsh[d] // 2) * st[d]
    cit = sm[cidx]
    maskv: Optional[List[bool]] = None
    if mask is not None:
        M = np.asarray(mask)
        if _shape(M) != sh:
            raise RuntimeError('mask and input must have equal sizes')
        maskv = _bool_buf(M)
    fp = _Footprint(fsh, sm, origins)
    buf = _bool_buf(a)
    if iterations == 1:
        buf = _erode_pass(buf, sh, fp, maskv, border_value, invert, cit)[0]
    else:
        r = _erode_pass(buf, sh, fp, maskv, border_value, invert, cit)
        buf = r[0]
        changed = r[1]
        ii = 1
        while ii < iterations or (iterations < 1 and changed):
            r = _erode_pass(buf, sh, fp, maskv, border_value, invert, cit)
            buf = r[0]
            changed = r[1]
            ii += 1
    return _bool_out(buf, a.shape, output)

def binary_erosion(input, structure = None, iterations: int = 1, mask = None, output = None,
                   border_value: int = 0, origin = 0, brute_force: bool = False, axes = None):
    return _binary_erosion(input, structure, iterations, mask, output, border_value, origin,
                           False, axes, False)

def binary_dilation(input, structure = None, iterations: int = 1, mask = None, output = None,
                    border_value: int = 0, origin = 0, brute_force: bool = False, axes = None):
    return _binary_erosion(input, structure, iterations, mask, output, border_value, origin,
                           True, axes, True)

def binary_opening(input, structure = None, iterations: int = 1, output = None, origin = 0,
                   mask = None, border_value: int = 0, brute_force: bool = False, axes = None):
    tmp = binary_erosion(input, structure, iterations, mask, None, border_value, origin, brute_force, axes)
    return binary_dilation(tmp, structure, iterations, mask, output, border_value, origin, brute_force, axes)

def binary_closing(input, structure = None, iterations: int = 1, output = None, origin = 0,
                   mask = None, border_value: int = 0, brute_force: bool = False, axes = None):
    tmp = binary_dilation(input, structure, iterations, mask, None, border_value, origin, brute_force, axes)
    return binary_erosion(tmp, structure, iterations, mask, output, border_value, origin, brute_force, axes)

def binary_propagation(input, structure = None, mask = None, output = None, border_value: int = 0,
                       origin = 0, axes = None):
    return binary_dilation(input, structure, -1, mask, output, border_value, origin, False, axes)

def binary_fill_holes(input, structure = None, output = None, origin = 0, axes = None):
    a = np.asarray(input)
    m = np.logical_not(a)
    tmp = np.zeros(a.shape, dtype=bool)
    r = binary_dilation(tmp, structure, -1, m, None, 1, origin, False, axes)
    r2 = np.logical_not(r)
    if output is None:
        return r2
    elif isinstance(output, type):
        return r2.astype(output)
    else:
        output[...] = r2
        return output

def binary_hit_or_miss(input, structure1 = None, structure2 = None, output = None,
                       origin1 = 0, origin2 = None, axes = None):
    a = np.asarray(input)
    sh = _shape(a)
    na = len(_check_axes(axes, len(sh)))
    s1sh = List[int]()
    s1 = List[bool]()
    if structure1 is None:
        t = _bstruct(na, 1)
        s1sh = t[0]
        s1 = t[1]
    else:
        S = np.asarray(structure1)
        s1sh = _shape(S)
        s1 = _bool_buf(S)
    s2 = List[bool]()
    s2sh = List[int]()
    if structure2 is None:
        s2sh = s1sh
        s2 = [not b for b in s1]
    else:
        S2 = np.asarray(structure2)
        s2sh = _shape(S2)
        s2 = _bool_buf(S2)
    S1a = np.empty(_prod(s1sh), dtype=bool)
    for i in range(len(s1)):
        S1a[i] = s1[i]
    S2a = np.empty(_prod(s2sh), dtype=bool)
    for i in range(len(s2)):
        S2a[i] = s2[i]
    o1 = _seq_int(origin1, na)
    o2 = o1 if origin2 is None else _seq_int(origin2, na)
    t1 = _binary_erosion(a, S1a.reshape(_tuple_shape(s1sh, a.shape)), 1, None, None, 0, o1, False, axes, False)
    t2 = _binary_erosion(a, S2a.reshape(_tuple_shape(s2sh, a.shape)), 1, None, None, 0, o2, True, axes, False)
    r = np.logical_and(t1, np.logical_not(t2))
    if output is None:
        return r
    elif isinstance(output, type):
        return r.astype(output)
    else:
        output[...] = r
        return output

def iterate_structure(structure, iterations: int, origin = None):
    S = np.asarray(structure)
    if iterations < 2:
        if origin is None:
            return S.copy() != 0
        else:
            compile_error("origin is not supported")
    ni = iterations - 1
    ssh = _shape(S)
    shape = [s + ni * (s - 1) for s in ssh]
    pos = [ni * (s // 2) for s in ssh]
    out = np.zeros(_tuple_shape(shape, S.shape), dtype=bool)
    sb = _bool_buf(S)
    st = _strides(shape)
    idx = [0] * len(ssh)
    for k in range(len(sb)):
        off = 0
        for d in range(len(ssh)):
            off += (idx[d] + pos[d]) * st[d]
        out.data[off] = sb[k]
        _next_coord(idx, ssh)
    return binary_dilation(out, S, ni)

# ---- grey morphology ---------------------------------------------------------

def grey_erosion(input, size = None, footprint = None, structure = None, output = None,
                 mode = 'reflect', cval: float = 0.0, origin = 0, axes = None):
    if size is None and footprint is None and structure is None:
        raise ValueError('size, footprint, or structure must be specified')
    return _min_or_max_filter(input, size, footprint, structure, output, mode, cval, origin, True, axes)

def _flip_nd(x):
    X = np.ascontiguousarray(np.asarray(x))
    out = np.empty(X.shape, dtype=X.dtype)
    n = X.size
    for i in range(n):
        out.data[i] = X.data[n - 1 - i]
    return out

def grey_dilation(input, size = None, footprint = None, structure = None, output = None,
                  mode = 'reflect', cval: float = 0.0, origin = 0, axes = None):
    if size is None and footprint is None and structure is None:
        raise ValueError('size, footprint, or structure must be specified')
    a = np.asarray(input)
    nd = static.len(a.shape)
    ax = _check_axes(axes, nd)
    org = _seq_int(origin, len(ax))
    szs = List[int]()
    if footprint is not None:
        szs = _shape(np.asarray(footprint))
    elif structure is not None:
        szs = _shape(np.asarray(structure))
    elif size is not None:
        szs = _seq_int(size, len(ax))
    for ii in range(len(org)):
        org[ii] = -org[ii]
        if not (szs[ii] & 1):
            org[ii] -= 1
    if structure is not None:
        if footprint is not None:
            return _min_or_max_filter(a, size, _flip_nd(footprint), _flip_nd(structure), output,
                                      mode, cval, org, False, axes)
        else:
            return _min_or_max_filter(a, size, None, _flip_nd(structure), output,
                                      mode, cval, org, False, axes)
    elif footprint is not None:
        return _min_or_max_filter(a, size, _flip_nd(footprint), None, output,
                                  mode, cval, org, False, axes)
    else:
        return _min_or_max_filter(a, size, None, None, output, mode, cval, org, False, axes)

def grey_opening(input, size = None, footprint = None, structure = None, output = None,
                 mode = 'reflect', cval: float = 0.0, origin = 0, axes = None):
    tmp = grey_erosion(input, size, footprint, structure, None, mode, cval, origin, axes)
    return grey_dilation(tmp, size, footprint, structure, output, mode, cval, origin, axes)

def grey_closing(input, size = None, footprint = None, structure = None, output = None,
                 mode = 'reflect', cval: float = 0.0, origin = 0, axes = None):
    tmp = grey_dilation(input, size, footprint, structure, None, mode, cval, origin, axes)
    return grey_erosion(tmp, size, footprint, structure, output, mode, cval, origin, axes)

def morphological_gradient(input, size = None, footprint = None, structure = None, output = None,
                           mode = 'reflect', cval: float = 0.0, origin = 0, axes = None):
    tmp = grey_dilation(input, size, footprint, structure, None, mode, cval, origin, axes)
    return tmp - grey_erosion(input, size, footprint, structure, None, mode, cval, origin, axes)

def morphological_laplace(input, size = None, footprint = None, structure = None, output = None,
                          mode = 'reflect', cval: float = 0.0, origin = 0, axes = None):
    a = np.asarray(input)
    t1 = grey_dilation(a, size, footprint, structure, None, mode, cval, origin, axes)
    t2 = grey_erosion(a, size, footprint, structure, None, mode, cval, origin, axes)
    return t1 + t2 - a - a

def white_tophat(input, size = None, footprint = None, structure = None, output = None,
                 mode = 'reflect', cval: float = 0.0, origin = 0, axes = None):
    a = np.asarray(input)
    tmp = grey_erosion(a, size, footprint, structure, None, mode, cval, origin, axes)
    tmp = grey_dilation(tmp, size, footprint, structure, None, mode, cval, origin, axes)
    if a.dtype is bool:
        return np.logical_xor(a, tmp)
    return a - tmp

def black_tophat(input, size = None, footprint = None, structure = None, output = None,
                 mode = 'reflect', cval: float = 0.0, origin = 0, axes = None):
    a = np.asarray(input)
    tmp = grey_dilation(a, size, footprint, structure, None, mode, cval, origin, axes)
    tmp = grey_erosion(tmp, size, footprint, structure, None, mode, cval, origin, axes)
    if a.dtype is bool:
        return np.logical_xor(tmp, a)
    return tmp - a

# ---- distance transforms -------------------------------------------------------

def _dt_arg_check(dt_inplace: bool, ft_inplace: bool, return_distances: bool, return_indices: bool):
    msgs = List[str]()
    if not return_distances and not return_indices:
        msgs.append('at least one of return_distances/return_indices must be True')
    if dt_inplace and not return_distances:
        msgs.append('return_distances must be True if distances is supplied')
    if ft_inplace and not return_indices:
        msgs.append('return_indices must be True if indices is supplied')
    if len(msgs) > 0:
        raise RuntimeError(', '.join(msgs))

def _voronoi_ft(F: List[int], base: int, stride: int, cstride: int, length: int,
                coor: List[int], rank: int, d: int, f: List[List[int]], g: List[int],
                sampling: Optional[List[float]]):
    for ii in range(length):
        for jj in range(rank):
            f[ii][jj] = F[base + ii * stride + cstride * jj]
    l = -1
    for ii in range(length):
        if F[base + ii * stride] >= 0:
            fd = float(f[ii][d])
            wR = 0.0
            for jj in range(rank):
                if jj != d:
                    tw = float(f[ii][jj] - coor[jj])
                    if sampling is not None:
                        tw *= sampling[jj]
                    wR += tw * tw
            while l >= 1:
                idx1 = g[l]
                f1 = float(f[idx1][d])
                idx2 = g[l - 1]
                a = f1 - f[idx2][d]
                b = fd - f1
                if sampling is not None:
                    a *= sampling[d]
                    b *= sampling[d]
                c = a + b
                uR = 0.0
                vR = 0.0
                for jj in range(rank):
                    if jj != d:
                        cc = coor[jj]
                        tu = float(f[idx2][jj] - cc)
                        tv = float(f[idx1][jj] - cc)
                        if sampling is not None:
                            tu *= sampling[jj]
                            tv *= sampling[jj]
                        uR += tu * tu
                        vR += tv * tv
                if c * vR - b * uR - a * wR - a * b * c <= 0.0:
                    break
                l -= 1
            l += 1
            g[l] = ii
    maxl = l
    if maxl >= 0:
        l = 0
        for ii in range(length):
            delta1 = 0.0
            for jj in range(rank):
                t = float(f[g[l]][jj] - ii) if jj == d else float(f[g[l]][jj] - coor[jj])
                if sampling is not None:
                    t *= sampling[jj]
                delta1 += t * t
            while l < maxl:
                delta2 = 0.0
                for jj in range(rank):
                    t = float(f[g[l + 1]][jj] - ii) if jj == d else float(f[g[l + 1]][jj] - coor[jj])
                    if sampling is not None:
                        t *= sampling[jj]
                    delta2 += t * t
                if delta1 <= delta2:
                    break
                delta1 = delta2
                l += 1
            idx1 = g[l]
            for jj in range(rank):
                F[base + ii * stride + jj * cstride] = f[idx1][jj]

def _compute_ft(I: List[bool], ibase: int, F: List[int], fbase: int, ish: List[int],
                ist: List[int], fst: List[int], rank: int, d: int, coor: List[int],
                f: List[List[int]], g: List[int], sampling: Optional[List[float]]):
    # F has shape (rank,) + ish; fst are its strides
    if d == 0:
        pi = ibase
        tf1 = fbase
        for jj in range(ish[0]):
            if I[pi]:
                F[tf1] = -1
            else:
                F[tf1] = jj
                tf2 = tf1
                for kk in range(1, rank):
                    tf2 += fst[0]
                    F[tf2] = coor[kk]
            pi += ist[0]
            tf1 += fst[1]
        _voronoi_ft(F, fbase, fst[1], fst[0], ish[0], coor, rank, 0, f, g, sampling)
    else:
        pi = ibase
        tf = fbase
        for jj in range(ish[d]):
            coor[d] = jj
            _compute_ft(I, pi, F, tf, ish, ist, fst, rank, d - 1, coor, f, g, sampling)
            pi += ist[d]
            tf += fst[d + 1]
        # iterate over the subspace of axes 0..d-1 (C order)
        sub = [ish[k] for k in range(d)]
        size = _prod(sub)
        idx = [0] * d
        for jj in range(size):
            off = fbase
            for kk in range(d):
                coor[kk] = idx[kk]
                off += idx[kk] * fst[kk + 1]
            _voronoi_ft(F, off, fst[d + 1], fst[0], ish[d], coor, rank, d, f, g, sampling)
            _next_coord(idx, sub)
        for kk in range(d):
            coor[kk] = 0

def _edt_features(I: List[bool], ish: List[int], sampling: Optional[List[float]]) -> List[int]:
    rank = len(ish)
    fsh = [rank] + ish
    F = [0] * _prod(fsh)
    ist = _strides(ish)
    fst = _strides(fsh)
    mx = 0
    for s in ish:
        if s > mx:
            mx = s
    f = [[0] * rank for _ in range(mx)]
    g = [0] * mx
    coor = [0] * rank
    _compute_ft(I, 0, F, 0, ish, ist, fst, rank, rank - 1, coor, f, g, sampling)
    return F

def _sampling_list(sampling, nd: int) -> Optional[List[float]]:
    if sampling is None:
        return None
    else:
        out = List[float]()
        if isinstance(sampling, int) or isinstance(sampling, float):
            for _ in range(nd):
                out.append(float(sampling))
        else:
            for v in sampling:
                out.append(float(v))
        if len(out) != nd:
            raise RuntimeError('sequence argument must have length equal to input rank')
        return out

def _indices_shape(a):
    return (static.len(a.shape),) + a.shape

def distance_transform_edt(input, sampling = None, return_distances: Literal[bool] = True,
                           return_indices: Literal[bool] = False, distances = None, indices = None):
    _dt_arg_check(distances is not None, indices is not None, return_distances, return_indices)
    a = np.asarray(input)
    ish = _shape(a)
    nd = len(ish)
    I = _bool_buf(a)
    samp = _sampling_list(sampling, nd)
    F = _edt_features(I, ish, samp)
    n = _prod(ish)
    if return_distances:
        dt = np.empty(a.shape, dtype=float)
        idx = [0] * nd
        for p in range(n):
            s = 0.0
            for d in range(nd):
                t = float(F[d * n + p] - idx[d])
                if samp is not None:
                    t *= samp[d]
                s += t * t
            dt.data[p] = sqrt(s)
            _next_coord(idx, ish)
        if return_indices:
            ft = np.empty(_indices_shape(a), dtype=np.int32)
            for p in range(len(F)):
                ft.data[p] = np.int32(F[p])
            return (dt, ft)
        else:
            return dt
    else:
        ft = np.empty(_indices_shape(a), dtype=np.int32)
        for p in range(len(F)):
            ft.data[p] = np.int32(F[p])
        return ft

def _dt_one_pass(msh: List[int], mstruct: List[bool], D: List[int], dsh: List[int],
                 Fe: Optional[List[int]], rev: bool):
    # NI_DistanceTransformOnePass on the (possibly reversed) view of D
    ssize = len(mstruct)
    fpm = [False] * ssize
    for jj in range(ssize // 2):
        fpm[jj] = mstruct[jj]
    fp = _Footprint(msh, fpm, [0] * len(msh))
    n = len(D)
    st = _strides(dsh)
    offs = [0] * len(fp)
    coord = [0] * len(dsh)
    for jj in range(n):
        p = n - 1 - jj if rev else jj
        value = D[p]
        if value != 0:
            _gather_offsets(fp, dsh, st, coord, 4, offs)
            mn = value
            mo = -1
            for ii in range(len(offs)):
                o = offs[ii]
                tt = -1
                if o >= 0:
                    q = n - 1 - o if rev else o
                    tt = D[q]
                if tt >= 0:
                    if mn < 0 or tt + 1 < mn:
                        mn = tt + 1
                        mo = o
            D[p] = mn
            if Fe is not None:
                if mo >= 0:
                    q = n - 1 - mo if rev else mo
                    Fe[p] = Fe[q]
        _next_coord(coord, dsh)

def distance_transform_cdt(input, metric = 'chessboard', return_distances: Literal[bool] = True,
                           return_indices: Literal[bool] = False, distances = None, indices = None):
    _dt_arg_check(distances is not None, indices is not None, return_distances, return_indices)
    a = np.asarray(input)
    ish = _shape(a)
    nd = len(ish)
    msh = List[int]()
    ms = List[bool]()
    if isinstance(metric, str):
        if metric in ('taxicab', 'cityblock', 'manhattan'):
            t = _bstruct(nd, 1)
            msh = t[0]
            ms = t[1]
        elif metric == 'chessboard':
            t = _bstruct(nd, nd)
            msh = t[0]
            ms = t[1]
        else:
            raise ValueError('invalid metric provided')
    else:
        M = np.asarray(metric)
        msh = _shape(M)
        ms = _bool_buf(M)
        for s in msh:
            if s != 3:
                raise ValueError('metric sizes must be equal to 3')
    I = _bool_buf(a)
    n = len(I)
    D = [(-1 if b else 0) for b in I]
    Fe: Optional[List[int]] = None
    if return_indices:
        Fe = list(range(n))
    _dt_one_pass(msh, ms, D, ish, Fe, False)
    _dt_one_pass(msh, ms, D, ish, Fe, True)
    dt = np.empty(a.shape, dtype=np.int32)
    for p in range(n):
        dt.data[p] = np.int32(D[p])
    if return_indices:
        ft = np.empty(_indices_shape(a), dtype=np.int32)
        ist = _strides(ish)
        fe = Fe
        if fe is not None:
            for p in range(n):
                r = fe[p]
                for d in range(nd):
                    ft.data[d * n + p] = np.int32(r // ist[d])
                    r %= ist[d]
        if return_distances:
            return (dt, ft)
        else:
            return ft
    else:
        return dt

def distance_transform_bf(input, metric: str = 'euclidean', sampling = None,
                          return_distances: Literal[bool] = True,
                          return_indices: Literal[bool] = False, distances = None, indices = None):
    # brute force: distance from every foreground point to the nearest background point
    _dt_arg_check(distances is not None, indices is not None, return_distances, return_indices)
    a = np.asarray(input)
    ish = _shape(a)
    nd = len(ish)
    I = _bool_buf(a)
    n = len(I)
    samp = _sampling_list(sampling, nd)
    m = metric.lower()
    code = 0
    if m == 'euclidean':
        code = 1
    elif m in ('taxicab', 'cityblock', 'manhattan'):
        code = 2
    elif m == 'chessboard':
        code = 3
    else:
        raise RuntimeError('distance metric not supported')
    ist = _strides(ish)
    # border points: background points adjacent (full connectivity) to foreground
    bg = List[int]()
    for p in range(n):
        if not I[p]:
            bg.append(p)
    dist = [0.0] * n
    feat = list(range(n))
    for p in range(n):
        if I[p]:
            best = 1e300
            bi = -1
            for q in bg:
                s = 0.0
                for d in range(nd):
                    t = float(abs(p // ist[d] % ish[d] - q // ist[d] % ish[d]))
                    if samp is not None:
                        t *= samp[d]
                    if code == 1:
                        s += t * t
                    elif code == 2:
                        s += t
                    else:
                        if t > s:
                            s = t
                if s < best:
                    best = s
                    bi = q
            dist[p] = sqrt(best) if code == 1 else best
            feat[p] = bi
    if return_distances:
        dt = np.empty(a.shape, dtype=float)
        for p in range(n):
            dt.data[p] = dist[p]
        if return_indices:
            ft = np.empty(_indices_shape(a), dtype=np.int32)
            for p in range(n):
                r = feat[p]
                for d in range(nd):
                    ft.data[d * n + p] = np.int32(r // ist[d])
                    r %= ist[d]
            return (dt, ft)
        else:
            return dt
    else:
        ft = np.empty(_indices_shape(a), dtype=np.int32)
        for p in range(n):
            r = feat[p]
            for d in range(nd):
                ft.data[d * n + p] = np.int32(r // ist[d])
                r %= ist[d]
        return ft
