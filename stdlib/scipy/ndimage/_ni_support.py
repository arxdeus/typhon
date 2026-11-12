# Ported from scipy 1.18.1 (BSD-3-Clause)

# Shared helpers for scipy.ndimage: boundary-mode codes, line extension
# (NI_ExtendLine), coordinate mapping (NI_InitFilterOffsets), output
# casting and N-d index bookkeeping. Images are processed as C-contiguous
# float64 buffers.

import numpy as np
import internal.static as static

NEAREST = 0
WRAP = 1
REFLECT = 2
MIRROR = 3
CONSTANT = 4
GRID_WRAP = 5
GRID_CONSTANT = 6

def _extend_mode_to_code(mode: str, is_filter: bool = False) -> int:
    if mode == 'nearest':
        return 0
    elif mode == 'wrap':
        return 1
    elif mode == 'reflect' or mode == 'grid-mirror':
        return 2
    elif mode == 'mirror':
        return 3
    elif mode == 'constant':
        return 4
    elif mode == 'grid-wrap' and is_filter:
        return 1
    elif mode == 'grid-wrap':
        return 5
    elif mode == 'grid-constant' and is_filter:
        return 4
    elif mode == 'grid-constant':
        return 6
    else:
        raise RuntimeError('boundary mode not supported')

def _seq_int(x, n: int) -> List[int]:
    if isinstance(x, int):
        return [x] * n
    else:
        out = List[int]()
        for v in x:
            out.append(int(v))
        if len(out) != n:
            raise RuntimeError('sequence argument must have length equal to input rank')
        return out

def _seq_float(x, n: int) -> List[float]:
    if isinstance(x, int) or isinstance(x, float):
        return [float(x)] * n
    else:
        out = List[float]()
        for v in x:
            out.append(float(v))
        if len(out) != n:
            raise RuntimeError('sequence argument must have length equal to input rank')
        return out

def _seq_str(x, n: int) -> List[str]:
    if isinstance(x, str):
        return [x] * n
    else:
        out = List[str]()
        for v in x:
            out.append(v)
        if len(out) != n:
            raise RuntimeError('sequence argument must have length equal to input rank')
        return out

def _seq_opt_int(x, n: int) -> List[Optional[int]]:
    if x is None:
        return [None] * n
    elif isinstance(x, int):
        return [Optional[int](x)] * n
    else:
        out = List[Optional[int]]()
        for v in x:
            out.append(Optional[int](int(v)))
        if len(out) != n:
            raise RuntimeError('sequence argument must have length equal to input rank')
        return out

def _check_axes(axes, ndim: int) -> List[int]:
    if axes is None:
        return list(range(ndim))
    elif isinstance(axes, int):
        a = axes
        if a < -ndim or a > ndim - 1:
            raise ValueError(f'specified axis: {a} is out of range')
        return [a % ndim]
    else:
        out = List[int]()
        for ax in axes:
            a = int(ax)
            if a < -ndim or a > ndim - 1:
                raise ValueError(f'specified axis: {a} is out of range')
            a = a % ndim
            if a in out:
                raise ValueError('axes must be unique')
            out.append(a)
        return out

def _shape(a) -> List[int]:
    return [a.shape[i] for i in static.range(static.len(a.shape))]

def _strides(sh: List[int]) -> List[int]:
    st = [1] * len(sh)
    for d in range(len(sh) - 2, -1, -1):
        st[d] = st[d + 1] * sh[d + 1]
    return st

def _prod(sh: List[int]) -> int:
    p = 1
    for v in sh:
        p *= v
    return p

def _tuple_shape(lst: List[int], like):
    return Ptr[type(like)](lst._ptr.as_byte())[0]

def _fbuf(a) -> List[float]:
    # C-ordered float64 copy of an array
    A = np.ascontiguousarray(a)
    out = List[float](A.size)
    for i in range(A.size):
        out.append(float(A.data[i]))
    return out

def _farray(buf: List[float], shape):
    out = np.empty(shape, dtype=float)
    for i in range(len(buf)):
        out.data[i] = buf[i]
    return out

def _line_bases(sh: List[int], axis: int) -> List[int]:
    # base offsets of all 1-D lines along `axis` (C order)
    st = _strides(sh)
    n = _prod(sh)
    L = sh[axis]
    out = List[int]()
    if L == 0:
        return out
    nl = n // L
    idx = [0] * len(sh)
    for k in range(nl):
        off = 0
        for d in range(len(sh)):
            off += idx[d] * st[d]
        out.append(off)
        for d in range(len(sh) - 1, -1, -1):
            if d == axis:
                continue
            idx[d] += 1
            if idx[d] < sh[d]:
                break
            idx[d] = 0
    return out

def _extend_line(buf: List[float], size_before: int, line_length: int,
                 size_after: int, mode: int, cval: float):
    # in-place: buf holds size_before + line_length + size_after values,
    # the line itself at [size_before, size_before + line_length)
    first = size_before
    last = first + line_length
    if line_length == 1 and mode == MIRROR:
        mode = NEAREST
    if mode == NEAREST:
        v = buf[first]
        for i in range(size_before):
            buf[i] = v
        v = buf[last - 1]
        for i in range(size_after):
            buf[last + i] = v
    elif mode == WRAP or mode == GRID_WRAP:
        src = last - 1
        dst = first - 1
        sb = size_before
        while sb > 0:
            buf[dst] = buf[src]
            dst -= 1
            src -= 1
            sb -= 1
        src = first
        dst = last
        sa = size_after
        while sa > 0:
            buf[dst] = buf[src]
            dst += 1
            src += 1
            sa -= 1
    elif mode == REFLECT:
        src = first
        dst = first - 1
        sb = size_before
        while sb > 0 and src < last:
            buf[dst] = buf[src]
            dst -= 1
            src += 1
            sb -= 1
        src = last - 1
        while sb > 0:
            buf[dst] = buf[src]
            dst -= 1
            src -= 1
            sb -= 1
        src = last - 1
        dst = last
        sa = size_after
        while sa > 0 and src >= first:
            buf[dst] = buf[src]
            dst += 1
            src -= 1
            sa -= 1
        src = first
        while sa > 0:
            buf[dst] = buf[src]
            dst += 1
            src += 1
            sa -= 1
    elif mode == MIRROR:
        src = first + 1
        dst = first - 1
        sb = size_before
        while sb > 0 and src < last:
            buf[dst] = buf[src]
            dst -= 1
            src += 1
            sb -= 1
        src = last - 2
        while sb > 0:
            buf[dst] = buf[src]
            dst -= 1
            src -= 1
            sb -= 1
        src = last - 2
        dst = last
        sa = size_after
        while sa > 0 and src >= first:
            buf[dst] = buf[src]
            dst += 1
            src -= 1
            sa -= 1
        src = first + 1
        while sa > 0:
            buf[dst] = buf[src]
            dst += 1
            src += 1
            sa -= 1
    else:
        for i in range(size_before):
            buf[i] = cval
        for i in range(size_after):
            buf[last + i] = cval

def _map_index(cc: int, n: int, mode: int) -> int:
    # NI_InitFilterOffsets boundary handling; -1 flags "outside" (constant)
    if mode == MIRROR:
        if cc < 0:
            if n <= 1:
                return 0
            sz2 = 2 * n - 2
            cc = sz2 * (-cc // sz2) + cc
            return cc + sz2 if cc <= 1 - n else -cc
        elif cc >= n:
            if n <= 1:
                return 0
            sz2 = 2 * n - 2
            cc -= sz2 * (cc // sz2)
            if cc >= n:
                cc = sz2 - cc
        return cc
    elif mode == REFLECT:
        if cc < 0:
            if n <= 1:
                return 0
            sz2 = 2 * n
            if cc < -sz2:
                cc = sz2 * (-cc // sz2) + cc
            return cc + sz2 if cc < -n else -cc - 1
        elif cc >= n:
            if n <= 1:
                return 0
            sz2 = 2 * n
            cc -= sz2 * (cc // sz2)
            if cc >= n:
                cc = sz2 - cc - 1
        return cc
    elif mode == WRAP or mode == GRID_WRAP:
        if cc < 0:
            if n <= 1:
                return 0
            cc += n * (-cc // n)
            if cc < 0:
                cc += n
        elif cc >= n:
            if n <= 1:
                return 0
            cc -= n * (cc // n)
        return cc
    elif mode == NEAREST:
        if cc < 0:
            return 0
        if cc >= n:
            return n - 1
        return cc
    else:
        if cc < 0 or cc >= n:
            return -1
        return cc

class _Footprint:
    # relative offsets (per axis) of the true footprint elements, in C order
    rel: List[List[int]]
    flat: List[int]      # flat index into the footprint array
    fshape: List[int]

    def __init__(self, fshape: List[int], mask: List[bool], origins: List[int]):
        self.fshape = fshape
        self.rel = List[List[int]]()
        self.flat = List[int]()
        r = len(fshape)
        n = _prod(fshape)
        idx = [0] * r
        for k in range(n):
            if mask[k]:
                rr = [0] * r
                for d in range(r):
                    rr[d] = idx[d] - (fshape[d] // 2 + origins[d])
                self.rel.append(rr)
                self.flat.append(k)
            for d in range(r - 1, -1, -1):
                idx[d] += 1
                if idx[d] < fshape[d]:
                    break
                idx[d] = 0

    def __len__(self) -> int:
        return len(self.rel)

def _gather_offsets(fp: _Footprint, sh: List[int], st: List[int], coord: List[int],
                    mode: int, out: List[int]):
    # fills out[i] with the flat input offset of footprint element i at
    # position `coord` (-1 if outside with constant mode)
    r = len(sh)
    for i in range(len(fp.rel)):
        rr = fp.rel[i]
        off = 0
        for d in range(r):
            c = _map_index(coord[d] + rr[d], sh[d], mode)
            if c < 0:
                off = -1
                break
            off += c * st[d]
        out[i] = off

def _next_coord(coord: List[int], sh: List[int]):
    for d in range(len(sh) - 1, -1, -1):
        coord[d] += 1
        if coord[d] < sh[d]:
            return
        coord[d] = 0

# ---- output handling -------------------------------------------------------

def _proto(input, output):
    if output is None:
        return input.dtype()
    elif isinstance(output, type):
        return output()
    else:
        return output.dtype()

def _cast1(v: float, T: type):
    if T is float:
        return v
    elif T is bool:
        return int(v) != 0
    elif T is float32:
        return float32(v)
    elif T is u8 or T is u16 or T is u32 or T is u64 or T is np.uint8 or T is np.uint16 or T is np.uint32 or T is np.uint64:
        return T(int(v)) if v > -1.0 else T(-int(-v))
    else:
        return T(int(v))

def _cast1_round(v: float, T: type):
    if T is float:
        return v
    elif T is float32:
        return float32(v)
    elif T is bool:
        return int(v + 0.5 if v > 0 else v - 0.5) != 0
    elif T is u8 or T is u16 or T is u32 or T is u64 or T is np.uint8 or T is np.uint16 or T is np.uint32 or T is np.uint64:
        t = v + 0.5 if v > 0 else 0.0
        return T(int(t))
    else:
        t = v + 0.5 if v > 0 else v - 0.5
        return T(int(t))

def _quantize(buf: List[float], proto) -> List[float]:
    T = type(proto)
    if T is float:
        return buf
    else:
        out = List[float](len(buf))
        for v in buf:
            out.append(float(_cast1(v, T)))
        return out

def _emit(buf: List[float], shape, proto, output, rnd: bool = False):
    T = type(proto)
    res = np.empty(shape, dtype=T)
    for i in range(len(buf)):
        if rnd:
            res.data[i] = _cast1_round(buf[i], T)
        else:
            res.data[i] = _cast1(buf[i], T)
    if output is None:
        return res
    elif isinstance(output, type):
        return res
    else:
        if _shape(output) != _shape(res):
            raise RuntimeError('output shape not correct')
        output[...] = res
        return output
