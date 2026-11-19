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
from simd import Vec

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

def _dw_row_vec4(xr: Ptr[float32], wr: Ptr[float32], q: Ptr[float32], bv: float32, i: int, hi: int, pw: int, KW: int) -> int:
    bvv = Vec[float32, 4](bv)
    while i + 4 <= hi:
        accv = Vec[float32, 4](float32(0.0))
        s = i - pw
        for k in range(KW):
            accv = Vec[float32, 4].load(xr + s + k)._fma(Vec[float32, 4](wr[k]), accv)
        (accv + bvv).store(q + i)
        i += 4
    return i

def _fwd_depthwise_row(T: type, x: Tensor, w: Tensor, b: Optional[Tensor], g: _Geo, out: Tensor):
    # Depthwise single-row conv (conv1d with groups == in_channels, stride 1,
    # dilation 1), as ATen's CPU path on arm64: per output a sequential fused
    # multiply-add over the kernel taps starting from 0 (zero-padding taps
    # included as fma(0, w, acc)), then + bias.
    xp = Ptr[T](x._st.ptr) + x._off
    wp = Ptr[T](w._st.ptr) + w._off
    op = Ptr[T](out._st.ptr) + out._off
    Og = g.O // g.G
    L = g.OW
    KW = g.KW
    for n in range(g.N):
        for o in range(g.O):
            xr = xp + (n * g.C + o // Og) * g.W
            wr = wp + o * KW
            q = op + (n * g.O + o) * L
            bv = T(0.0)
            if b is not None:
                bv = (Ptr[T](b._st.ptr) + b._off)[o]
            # interior outputs [lo, hi) (no padding taps), float32: 4 outputs per
            # step, each lane the same sequential fma chain as the scalar loop
            lo = min(max(g.pw, 0), L)
            hi = max(min(g.W - KW + g.pw + 1, L), lo)
            i = 0
            while i < L:
                if i == lo:
                    if T is float32:
                        i = _dw_row_vec4(xr, wr, q, bv, i, hi, g.pw, KW)
                    if i >= L:
                        break
                acc = T(0.0)
                s = i - g.pw
                if s >= 0 and s + KW <= g.W:
                    for k in range(KW):
                        acc = T._fma(xr[s + k], wr[k], acc)
                else:
                    for k in range(KW):
                        j = s + k
                        xv = xr[j] if 0 <= j < g.W else T(0.0)
                        acc = T._fma(xv, wr[k], acc)
                q[i] = acc + bv
                i += 1

def _nnpack_ft8x8_eligible(g: _Geo) -> bool:
    # ATen ConvParams::use_nnpack (torch 2.15 CPU float, not mobile): batch >= 16, stride 1
    # (else per-sample nnp_convolution_inference), no dilation, kernel < 17, padding < kernel;
    # NNPACK auto algorithm must pick ft8x8 (wt8x8 for 3x3 and ft16x16 are not ported).
    if g.N < 16 or g.sh != 1 or g.sw != 1 or g.dh != 1 or g.dw != 1:
        return False
    if g.KH >= 17 or g.KW >= 17 or g.ph >= g.KH or g.pw >= g.KW:
        return False
    if g.KH > 8 or g.KW > 8 or (g.KH == 3 and g.KW == 3):
        return False
    t8 = ((g.OH + (8 - g.KH)) // (8 - g.KH + 1)) * ((g.OW + (8 - g.KW)) // (8 - g.KW + 1))
    t16 = ((g.OH + (16 - g.KH)) // (16 - g.KH + 1)) * ((g.OW + (16 - g.KW)) // (16 - g.KW + 1))
    return t8 <= 4 * t16

def _tmp(T: type, n: int) -> Ptr[T]:
    # per-call scratch (callers may run on several threads), released with _rel
    return Ptr[T](max(n, 1))

def _rel(p: Ptr[T], T: type):
    from internal.gc import free
    free(p.as_byte())

def _fwd_nnpack(x: Tensor, w: Tensor, b: Optional[Tensor], g: _Geo, out: Tensor):
    # ATen convolution(): backends without native groups loop over groups on contiguous
    # sub-tensors (input/weight/bias split along dim 1/0) and concatenate the outputs.
    from torch._nnpack import conv_ft8x8
    Cg = g.C // g.G
    Og = g.O // g.G
    xp = Ptr[float32](x._st.ptr) + x._off
    wp = Ptr[float32](w._st.ptr) + w._off
    op = Ptr[float32](out._st.ptr) + out._off
    HW = g.H * g.W
    L = g.OH * g.OW
    bg = _tmp(float32, Og)
    if g.G == 1:
        # one group: the sub-tensors are the tensors themselves, no copies
        bp = bg
        if b is None:
            for o in range(Og):
                bg[o] = float32(0.0)
        else:
            bp = Ptr[float32](b._st.ptr) + b._off
        conv_ft8x8(g.N, Cg, Og, g.H, g.W, g.KH, g.KW, g.ph, g.pw, xp, wp, bp, op)
        _rel(bg)
        return
    xg = _tmp(float32, g.N * Cg * HW)
    yg = _tmp(float32, g.N * Og * L)
    for gi in range(g.G):
        for n in range(g.N):
            src = xp + (n * g.C + gi * Cg) * HW
            dst = xg + n * Cg * HW
            for i in range(Cg * HW):
                dst[i] = src[i]
        for o in range(Og):
            bg[o] = float32(0.0) if b is None else (Ptr[float32](b._st.ptr) + b._off)[gi * Og + o]
        conv_ft8x8(g.N, Cg, Og, g.H, g.W, g.KH, g.KW, g.ph, g.pw, xg, wp + gi * Og * Cg * g.KH * g.KW, bg, yg)
        for n in range(g.N):
            src = yg + n * Og * L
            dst = op + (n * g.O + gi * Og) * L
            for i in range(Og * L):
                dst[i] = src[i]
    _rel(xg)
    _rel(yg)
    _rel(bg)

def _fwd_slow2d(x: Tensor, w: Tensor, b: Optional[Tensor], g: _Geo, out: Tensor):
    # ATen Slow2d (thnn_conv2d, slow_conv2d_update_output_frame) per sample:
    #   output = bias broadcast (or 0); output(O x L) += weight(O x K) * columns(K x L)
    # as cpublas::gemm(N, N, L, O, K, 1, columns, L, weight, K, 1 or 0, output, L).
    xp = Ptr[float32](x._st.ptr) + x._off
    wp = Ptr[float32](w._st.ptr) + w._off
    op = Ptr[float32](out._st.ptr) + out._off
    K = g.C * g.KH * g.KW
    L = g.OH * g.OW
    simple = _is_1x1_simple(g)
    cols = _tmp(float32, 1 if simple else K * L)
    for n in range(g.N):
        xn = xp + n * g.C * g.H * g.W
        on = op + n * g.O * L
        cp = xn
        if not simple:
            _im2col(float32, xn, g, 0, g.C, cols)
            cp = cols
        beta = 0.0
        if b is not None:
            bp = Ptr[float32](b._st.ptr) + b._off
            for o in range(g.O):
                bv = bp[o]
                q = on + o * L
                for i in range(L):
                    q[i] = bv
            beta = 1.0
        _gemm_cm(False, False, L, g.O, K, cp, L, wp, K, beta, on, L)
    _rel(cols)

def _fwd(T: type, x: Tensor, w: Tensor, b: Optional[Tensor], g: _Geo, out: Tensor):
    if g.H == 1 and g.KH == 1 and g.ph == 0 and g.sw == 1 and g.dw == 1 and g.G > 1 and g.C == g.G and g.N < 16:
        _fwd_depthwise_row(T, x, w, b, g, out)
        return
    if T is float32:
        if _nnpack_ft8x8_eligible(g):
            _fwd_nnpack(x, w, b, g, out)
            return
        if g.G == 1:
            _fwd_slow2d(x, w, b, g, out)
            return
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
    _rel(cols)

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

def _gemm_cm(ta: bool, tb: bool, M: int, N: int, K: int, A: Ptr[float32], lda: int, B: Ptr[float32], ldb: int, beta: float, C: Ptr[float32], ldc: int):
    # column-major sgemm, exactly as ATen cpublas::gemm hands it to BLAS
    if M == 0 or N == 0:
        return
    cblas_sgemm(fint(102), fint(_TR if ta else _NT), fint(_TR if tb else _NT), fint(M), fint(N), fint(K), float32(1.0), A, fint(lda), B, fint(ldb), float32(beta), C, fint(ldc))

def _bwd_dilated(x: Tensor, w: Tensor, gy: Tensor, g: _Geo, gx: Tensor, gw: Tensor, gb: Tensor, need_x: bool, need_w: bool, need_b: bool):
    # ATen backward of NnpackSpatial = slow_conv_dilated2d_backward (NaiveDilatedConvolution.cpp)
    # per group (convolution_backward's nogroup loop on subtensors), per sample:
    #   grad_input:  columns^T = grad_output_n^T * weight          (gemm N,T; col2im)
    #   grad_weight: grad_weight^T += columns * grad_output_n^T    (gemm T,N, beta=1)
    #   grad_bias  += grad_output_n.sum(spatial)
    from torch._tensor import _cascade_sum_f32
    xp = Ptr[float32](x._st.ptr) + x._off
    wp = Ptr[float32](w._st.ptr) + w._off
    gp = Ptr[float32](gy._st.ptr) + gy._off
    Cg = g.C // g.G
    Og = g.O // g.G
    K = Cg * g.KH * g.KW
    L = g.OH * g.OW
    HW = g.H * g.W
    cols = Ptr[float32](K * L)
    gyn = Ptr[float32](Og * L)
    gxp = Ptr[float32](gx._st.ptr)
    gwp = Ptr[float32](gw._st.ptr)
    gbp = Ptr[float32](gb._st.ptr)
    if need_w:
        for i in range(gw.numel()):
            gwp[i] = float32(0.0)
    if need_b:
        for i in range(g.O):
            gbp[i] = float32(0.0)
    for gi in range(g.G):
        for n in range(g.N):
            src = gp + (n * g.O + gi * Og) * L
            for i in range(Og * L):
                gyn[i] = src[i]
            if need_x:
                _gemm_cm(False, True, L, K, Og, gyn, L, wp + gi * Og * K, K, 0.0, cols, L)
                gxn = gxp + n * g.C * HW
                for i in range(Cg * HW):
                    gxn[gi * Cg * HW + i] = float32(0.0)
                _col2im(float32, cols, g, gi * Cg, Cg, gxn)
            if need_w:
                _im2col(float32, xp + n * g.C * HW, g, gi * Cg, Cg, cols)
                _gemm_cm(True, False, K, Og, L, cols, L, gyn, L, 1.0, gwp + gi * Og * K, K)
            if need_b:
                for o in range(Og):
                    gbp[gi * Og + o] = gbp[gi * Og + o] + _cascade_sum_f32(gyn + o * L, L)
    _rel(cols)
    _rel(gyn)

def _bwd(T: type, x: Tensor, w: Tensor, gy: Tensor, g: _Geo, gx: Tensor, gw: Tensor, gb: Tensor, need_x: bool, need_w: bool, need_b: bool):
    if T is float32:
        if _nnpack_ft8x8_eligible(g):
            _bwd_dilated(x, w, gy, g, gx, gw, gb, need_x, need_w, need_b)
            return
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
    _rel(cols)
    if need_b:
        bp = Ptr[T](gb._st.ptr)
        if T is float32:
            # ATen Slow2d backward: grad_bias = grad_output.sum({0, 2, 3}); SumKernel's
            # cascade_sum runs the contiguous inner reduction per sample and adds it to
            # the output (out += cascade(row)), samples in order
            from torch._tensor import _cascade_sum_f32
            gpf = Ptr[float32](gy._st.ptr) + gy._off
            for o in range(g.O):
                s = float32(0.0)
                for n in range(g.N):
                    s = s + _cascade_sum_f32(gpf + (n * g.O + o) * L, L)
                bp[o] = s
            return
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
