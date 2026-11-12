# Ported from torch 2.15.0 (BSD-3-Clause)
#
# Fast 2-D convolution kernels (im2col + BLAS gemm), as used by torch's CPU
# slow_conv2d / thnn_conv2d path. Tensors must be contiguous NCHW of the same
# float dtype. Exposed for torch.nn.functional:
#   conv2d_forward(x, w, b, stride, padding, dilation, groups) -> out
#   conv2d_backward(x, w, gy, stride, padding, dilation, groups,
#                   need_x, need_w, need_b) -> (gx, gw, gb)

from numpy.linalg.blas import cblas_sgemm, cblas_dgemm, fint
from torch._tensor import Tensor, _new, _F32, _F64, _contig_as, _is_float

_ROW = 101
_NT = 111
_TR = 112

def _gemm(T: type, ta: bool, tb: bool, M: int, N: int, K: int, A: Ptr[T], lda: int, B: Ptr[T], ldb: int, beta: float, C: Ptr[T], ldc: int):
    if M == 0 or N == 0:
        return
    if T is float32:
        cblas_sgemm(fint(_ROW), fint(_TR if ta else _NT), fint(_TR if tb else _NT), fint(M), fint(N), fint(K), float32(1.0), A, fint(lda), B, fint(ldb), float32(beta), C, fint(ldc))
    else:
        cblas_dgemm(fint(_ROW), fint(_TR if ta else _NT), fint(_TR if tb else _NT), fint(M), fint(N), fint(K), 1.0, A, fint(lda), B, fint(ldb), beta, C, fint(ldc))

@tuple
class _Geo:
    N: int
    C: int
    H: int
    W: int
    O: int
    KH: int
    KW: int
    OH: int
    OW: int
    sh: int
    sw: int
    ph: int
    pw: int
    dh: int
    dw: int
    G: int

def _geo(x: Tensor, w: Tensor, stride: Tuple[int, int], padding: Tuple[int, int], dilation: Tuple[int, int], groups: int) -> _Geo:
    N = x._shape[0]
    C = x._shape[1]
    H = x._shape[2]
    W = x._shape[3]
    O = w._shape[0]
    KH = w._shape[2]
    KW = w._shape[3]
    OH = (H + 2 * padding[0] - dilation[0] * (KH - 1) - 1) // stride[0] + 1
    OW = (W + 2 * padding[1] - dilation[1] * (KW - 1) - 1) // stride[1] + 1
    return _Geo(N, C, H, W, O, KH, KW, OH, OW, stride[0], stride[1], padding[0], padding[1], dilation[0], dilation[1], groups)

# cols[(c*KH+ky)*KW+kx, oy*OW+ox] = x[c, oy*sh-ph+ky*dh, ox*sw-pw+kx*dw]
def _im2col(T: type, xp: Ptr[T], g: _Geo, c0: int, Cg: int, cols: Ptr[T]):
    L = g.OH * g.OW
    for c in range(Cg):
        xc = xp + (c0 + c) * g.H * g.W
        for ky in range(g.KH):
            for kx in range(g.KW):
                row = cols + ((c * g.KH + ky) * g.KW + kx) * L
                for oy in range(g.OH):
                    iy = oy * g.sh - g.ph + ky * g.dh
                    dst = row + oy * g.OW
                    if iy < 0 or iy >= g.H:
                        for ox in range(g.OW):
                            dst[ox] = T()
                        continue
                    src = xc + iy * g.W
                    off = kx * g.dw - g.pw
                    if g.sw == 1:
                        lo = 0 if off >= 0 else -off
                        hi = g.OW if g.W - off >= g.OW else g.W - off
                        if hi < lo:
                            hi = lo
                        for ox in range(lo):
                            dst[ox] = T()
                        s2 = src + off
                        for ox in range(lo, hi):
                            dst[ox] = s2[ox]
                        for ox in range(hi, g.OW):
                            dst[ox] = T()
                    else:
                        for ox in range(g.OW):
                            ix = ox * g.sw + off
                            dst[ox] = src[ix] if (ix >= 0 and ix < g.W) else T()

def _col2im(T: type, cols: Ptr[T], g: _Geo, c0: int, Cg: int, xp: Ptr[T]):
    L = g.OH * g.OW
    for c in range(Cg):
        xc = xp + (c0 + c) * g.H * g.W
        for ky in range(g.KH):
            for kx in range(g.KW):
                row = cols + ((c * g.KH + ky) * g.KW + kx) * L
                for oy in range(g.OH):
                    iy = oy * g.sh - g.ph + ky * g.dh
                    if iy < 0 or iy >= g.H:
                        continue
                    src = row + oy * g.OW
                    dst = xc + iy * g.W
                    off = kx * g.dw - g.pw
                    for ox in range(g.OW):
                        ix = ox * g.sw + off
                        if ix >= 0 and ix < g.W:
                            dst[ix] += src[ox]

def _is_1x1_simple(g: _Geo) -> bool:
    return g.KH == 1 and g.KW == 1 and g.sh == 1 and g.sw == 1 and g.ph == 0 and g.pw == 0

def _fwd(T: type, x: Tensor, w: Tensor, b: Optional[Tensor], g: _Geo, out: Tensor):
    xp = Ptr[T](x._st.ptr) + x._off
    wp = Ptr[T](w._st.ptr) + w._off
    op = Ptr[T](out._st.ptr) + out._off
    Cg = g.C // g.G
    Og = g.O // g.G
    K = Cg * g.KH * g.KW
    L = g.OH * g.OW
    simple = _is_1x1_simple(g)
    cols = Ptr[T](1 if simple else K * L)
    for n in range(g.N):
        xn = xp + n * g.C * g.H * g.W
        on = op + n * g.O * L
        for gi in range(g.G):
            if simple:
                cp = xn + gi * Cg * L
            else:
                _im2col(T, xn, g, gi * Cg, Cg, cols)
                cp = cols
            _gemm(T, False, False, Og, L, K, wp + gi * Og * K, K, cp, L, 0.0, on + gi * Og * L, L)
        if b is not None:
            bp = Ptr[T](b._st.ptr) + b._off
            for o in range(g.O):
                bv = bp[o]
                q = on + o * L
                for i in range(L):
                    q[i] += bv

def conv2d_forward(x: Tensor, w: Tensor, b: Optional[Tensor], stride: Tuple[int, int], padding: Tuple[int, int], dilation: Tuple[int, int], groups: int) -> Tensor:
    dt = x._dt
    xc = _contig_as(x, dt)
    wc = _contig_as(w, dt)
    bc: Optional[Tensor] = None
    if b is not None:
        bc = _contig_as(b, dt)
    g = _geo(xc, wc, stride, padding, dilation, groups)
    out = _new([g.N, g.O, g.OH, g.OW], dt)
    if dt == _F64:
        _fwd(float, xc, wc, bc, g, out)
    else:
        _fwd(float32, xc, wc, bc, g, out)
    return out

def _bwd(T: type, x: Tensor, w: Tensor, gy: Tensor, g: _Geo, gx: Tensor, gw: Tensor, gb: Tensor, need_x: bool, need_w: bool, need_b: bool):
    xp = Ptr[T](x._st.ptr) + x._off
    wp = Ptr[T](w._st.ptr) + w._off
    gp = Ptr[T](gy._st.ptr) + gy._off
    Cg = g.C // g.G
    Og = g.O // g.G
    K = Cg * g.KH * g.KW
    L = g.OH * g.OW
    simple = _is_1x1_simple(g)
    cols = Ptr[T](1 if simple else K * L)
    gxp = Ptr[T](gx._st.ptr)
    gwp = Ptr[T](gw._st.ptr)
    if need_x:
        for i in range(gx.numel()):
            gxp[i] = T()
    if need_w:
        for i in range(gw.numel()):
            gwp[i] = T()
    for n in range(g.N):
        xn = xp + n * g.C * g.H * g.W
        gn = gp + n * g.O * L
        for gi in range(g.G):
            gyg = gn + gi * Og * L
            if need_w:
                if simple:
                    cp = xn + gi * Cg * L
                else:
                    _im2col(T, xn, g, gi * Cg, Cg, cols)
                    cp = cols
                # gw[Og, K] += gy[Og, L] @ cols[K, L]^T
                _gemm(T, False, True, Og, K, L, gyg, L, cp, L, 1.0, gwp + gi * Og * K, K)
            if need_x:
                gxn = gxp + n * g.C * g.H * g.W
                if simple:
                    # gx[Cg, L] = W^T[Cg, Og] @ gy[Og, L]
                    _gemm(T, True, False, K, L, Og, wp + gi * Og * K, K, gyg, L, 0.0, gxn + gi * Cg * L, L)
                else:
                    _gemm(T, True, False, K, L, Og, wp + gi * Og * K, K, gyg, L, 0.0, cols, L)
                    _col2im(T, cols, g, gi * Cg, Cg, gxn)
    if need_b:
        bp = Ptr[T](gb._st.ptr)
        for o in range(g.O):
            s = T()
            for n in range(g.N):
                q = gp + (n * g.O + o) * L
                for i in range(L):
                    s += q[i]
            bp[o] = s

def conv2d_backward(x: Tensor, w: Tensor, gy: Tensor, stride: Tuple[int, int], padding: Tuple[int, int], dilation: Tuple[int, int], groups: int, need_x: bool, need_w: bool, need_b: bool) -> Tuple[Tensor, Tensor, Tensor]:
    dt = x._dt
    xc = _contig_as(x, dt)
    wc = _contig_as(w, dt)
    gc = _contig_as(gy, dt)
    g = _geo(xc, wc, stride, padding, dilation, groups)
    gx = _new(xc._shape.copy() if need_x else [0], dt)
    gw = _new(wc._shape.copy() if need_w else [0], dt)
    gb = _new([g.O] if need_b else [0], dt)
    if dt == _F64:
        _bwd(float, xc, wc, gc, g, gx, gw, gb, need_x, need_w, need_b)
    else:
        _bwd(float32, xc, wc, gc, g, gx, gw, gb, need_x, need_w, need_b)
    return (gx, gw, gb)

# ---------------------------------------------------------------- 3-D

@tuple
class _Geo3:
    N: int
    C: int
    D: int
    H: int
    W: int
    O: int
    KD: int
    KH: int
    KW: int
    OD: int
    OH: int
    OW: int
    sd: int
    sh: int
    sw: int
    pd: int
    ph: int
    pw: int
    dd: int
    dh: int
    dw: int
    G: int

def _geo3(x: Tensor, w: Tensor, stride: List[int], padding: List[int], dilation: List[int], groups: int) -> _Geo3:
    D = x._shape[2]
    H = x._shape[3]
    W = x._shape[4]
    KD = w._shape[2]
    KH = w._shape[3]
    KW = w._shape[4]
    OD = (D + 2 * padding[0] - dilation[0] * (KD - 1) - 1) // stride[0] + 1
    OH = (H + 2 * padding[1] - dilation[1] * (KH - 1) - 1) // stride[1] + 1
    OW = (W + 2 * padding[2] - dilation[2] * (KW - 1) - 1) // stride[2] + 1
    return _Geo3(x._shape[0], x._shape[1], D, H, W, w._shape[0], KD, KH, KW, OD, OH, OW,
                 stride[0], stride[1], stride[2], padding[0], padding[1], padding[2],
                 dilation[0], dilation[1], dilation[2], groups)

# cols[((c*KD+kd)*KH+ky)*KW+kx, (od*OH+oy)*OW+ox]; add=True does col2vol
def _vol2col(T: type, xp: Ptr[T], g: _Geo3, c0: int, Cg: int, cols: Ptr[T], add: bool):
    L = g.OD * g.OH * g.OW
    HW = g.H * g.W
    for c in range(Cg):
        xc = xp + (c0 + c) * g.D * HW
        for kd in range(g.KD):
            for ky in range(g.KH):
                for kx in range(g.KW):
                    row = cols + (((c * g.KD + kd) * g.KH + ky) * g.KW + kx) * L
                    for od in range(g.OD):
                        iz = od * g.sd - g.pd + kd * g.dd
                        for oy in range(g.OH):
                            iy = oy * g.sh - g.ph + ky * g.dh
                            dst = row + (od * g.OH + oy) * g.OW
                            inside = iz >= 0 and iz < g.D and iy >= 0 and iy < g.H
                            if not inside:
                                if not add:
                                    for ox in range(g.OW):
                                        dst[ox] = T()
                                continue
                            src = xc + iz * HW + iy * g.W
                            off = kx * g.dw - g.pw
                            for ox in range(g.OW):
                                ix = ox * g.sw + off
                                if ix >= 0 and ix < g.W:
                                    if add:
                                        src[ix] += dst[ox]
                                    else:
                                        dst[ox] = src[ix]
                                elif not add:
                                    dst[ox] = T()

def _fwd3(T: type, x: Tensor, w: Tensor, b: Optional[Tensor], g: _Geo3, out: Tensor):
    xp = Ptr[T](x._st.ptr) + x._off
    wp = Ptr[T](w._st.ptr) + w._off
    op = Ptr[T](out._st.ptr)
    Cg = g.C // g.G
    Og = g.O // g.G
    K = Cg * g.KD * g.KH * g.KW
    L = g.OD * g.OH * g.OW
    cols = Ptr[T](K * L if K * L > 0 else 1)
    vol = g.C * g.D * g.H * g.W
    for n in range(g.N):
        on = op + n * g.O * L
        for gi in range(g.G):
            _vol2col(T, xp + n * vol, g, gi * Cg, Cg, cols, False)
            _gemm(T, False, False, Og, L, K, wp + gi * Og * K, K, cols, L, 0.0, on + gi * Og * L, L)
        if b is not None:
            bp = Ptr[T](b._st.ptr) + b._off
            for o in range(g.O):
                bv = bp[o]
                q = on + o * L
                for i in range(L):
                    q[i] += bv

def conv3d_forward(x: Tensor, w: Tensor, b: Optional[Tensor], stride: List[int], padding: List[int], dilation: List[int], groups: int) -> Tensor:
    dt = x._dt
    xc = _contig_as(x, dt)
    wc = _contig_as(w, dt)
    bc: Optional[Tensor] = None
    if b is not None:
        bc = _contig_as(b, dt)
    g = _geo3(xc, wc, stride, padding, dilation, groups)
    out = _new([g.N, g.O, g.OD, g.OH, g.OW], dt)
    if dt == _F64:
        _fwd3(float, xc, wc, bc, g, out)
    else:
        _fwd3(float32, xc, wc, bc, g, out)
    return out

def _bwd3(T: type, x: Tensor, w: Tensor, gy: Tensor, g: _Geo3, gx: Tensor, gw: Tensor, gb: Tensor, need_x: bool, need_w: bool, need_b: bool):
    xp = Ptr[T](x._st.ptr) + x._off
    wp = Ptr[T](w._st.ptr) + w._off
    gp = Ptr[T](gy._st.ptr) + gy._off
    Cg = g.C // g.G
    Og = g.O // g.G
    K = Cg * g.KD * g.KH * g.KW
    L = g.OD * g.OH * g.OW
    cols = Ptr[T](K * L if K * L > 0 else 1)
    vol = g.C * g.D * g.H * g.W
    gxp = Ptr[T](gx._st.ptr)
    gwp = Ptr[T](gw._st.ptr)
    if need_x:
        for i in range(gx.numel()):
            gxp[i] = T()
    if need_w:
        for i in range(gw.numel()):
            gwp[i] = T()
    for n in range(g.N):
        gn = gp + n * g.O * L
        for gi in range(g.G):
            gyg = gn + gi * Og * L
            if need_w:
                _vol2col(T, xp + n * vol, g, gi * Cg, Cg, cols, False)
                _gemm(T, False, True, Og, K, L, gyg, L, cols, L, 1.0, gwp + gi * Og * K, K)
            if need_x:
                _gemm(T, True, False, K, L, Og, wp + gi * Og * K, K, gyg, L, 0.0, cols, L)
                _vol2col(T, gxp + n * vol, g, gi * Cg, Cg, cols, True)
    if need_b:
        bp = Ptr[T](gb._st.ptr)
        for o in range(g.O):
            s = T()
            for n in range(g.N):
                q = gp + (n * g.O + o) * L
                for i in range(L):
                    s += q[i]
            bp[o] = s

def conv3d_backward(x: Tensor, w: Tensor, gy: Tensor, stride: List[int], padding: List[int], dilation: List[int], groups: int, need_x: bool, need_w: bool, need_b: bool) -> Tuple[Tensor, Tensor, Tensor]:
    dt = x._dt
    xc = _contig_as(x, dt)
    wc = _contig_as(w, dt)
    gc = _contig_as(gy, dt)
    g = _geo3(xc, wc, stride, padding, dilation, groups)
    gx = _new(xc._shape.copy() if need_x else [0], dt)
    gw = _new(wc._shape.copy() if need_w else [0], dt)
    gb = _new([g.O] if need_b else [0], dt)
    if dt == _F64:
        _bwd3(float, xc, wc, gc, g, gx, gw, gb, need_x, need_w, need_b)
    else:
        _bwd3(float32, xc, wc, gc, g, gx, gw, gb, need_x, need_w, need_b)
    return (gx, gw, gb)
