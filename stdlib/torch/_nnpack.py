# Copyright (C) 2026 the Typhon authors
# Ported from NNPACK c07e3a0 (BSD-2-Clause), as built into torch 2.15.0 (BSD-3-Clause)
#
# Bit-exact float32 port of NNPACK's ft8x8 `nnp_convolution_output` (psimd 8x8 real FFT
# transforms + aarch64 NEON complex tuple GEMM with conj(b)), the kernel ATen's
# NnpackSpatial backend runs for CPU float convolutions with batch >= 16 on arm64.
# The 4-lane psimd vectors are unrolled to scalars; every fused multiply-add that clang
# (-ffp-contract=on) or the NEON intrinsics produce is an explicit float32._fma, all other
# products and sums are separately rounded, exactly as in the shipped torch binary.
#
#   conv_ft8x8(N, C, O, H, W, KH, KW, ph, pw, x, w, bias, y)
#     x: N x C x H x W, w: O x C x KH x KW, y: N x O x OH x OW (stride 1, no dilation).

from simd import Vec

_S2_2 = float32(0.7071067690849304)   # 0x1.6A09E6p-1f
_S2_4 = float32(0.3535533845424652)   # 0x1.6A09E6p-2f

def _doz(a: int, b: int) -> int:
    return a - b if a > b else 0

def _fma(a: float32, b: float32, c: float32) -> float32:
    return float32._fma(a, b, c)

# A psimd vector is 4 consecutive float32 at a pointer; helpers operate on Vec[float32, 4]
# lanes (IEEE per-lane add/sub/mul/fma = same bits as the scalar form).
@inline
def _ld(p: Ptr[float32]) -> Vec[float32, 4]:
    return Vec[float32, 4].load(p)

@inline
def _bfly(a: Ptr[float32], b: Ptr[float32]):
    x = _ld(a)
    y = _ld(b)
    (x + y).store(a)
    (x - y).store(b)

@inline
def _bfly_negb(a: Ptr[float32], b: Ptr[float32]):
    # psimd_butterfly_with_negated_b_f32: a, b = a - b, a + b
    x = _ld(a)
    y = _ld(b)
    (x - y).store(a)
    (x + y).store(b)

@inline
def _swp(a: Ptr[float32], b: Ptr[float32]):
    x = _ld(a)
    y = _ld(b)
    y.store(a)
    x.store(b)

@inline
def _cpy(d: Ptr[float32], s: Ptr[float32]):
    _ld(s).store(d)

@inline
def _shuf(d: Ptr[float32], a: Ptr[float32], b: Ptr[float32], i0: int, i1: int, i2: int, i3: int):
    # __builtin_shufflevector(a, b, i0..i3); d must not alias a or b
    d[0] = a[i0] if i0 < 4 else b[i0 - 4]
    d[1] = a[i1] if i1 < 4 else b[i1 - 4]
    d[2] = a[i2] if i2 < 4 else b[i2 - 4]
    d[3] = a[i3] if i3 < 4 else b[i3 - 4]

@inline
def _cmulc(re: Ptr[float32], im: Ptr[float32], cr: Ptr[float32], ci: Ptr[float32]):
    # xr*cr + xi*ci -> fma(xr, cr, xi*ci); xi*cr - xr*ci -> fma(xi, cr, -(xr*ci))
    xr = _ld(re)
    xi = _ld(im)
    vcr = _ld(cr)
    vci = _ld(ci)
    xr.fma(vcr, xi * vci).store(re)
    xi.fma(vcr, -(xr * vci)).store(im)

@inline
def _cmul(re: Ptr[float32], im: Ptr[float32], cr: Ptr[float32], ci: Ptr[float32]):
    xr = _ld(re)
    xi = _ld(im)
    vcr = _ld(cr)
    vci = _ld(ci)
    xr.fma(vcr, -(xi * vci)).store(re)
    xi.fma(vcr, xr * vci).store(im)

class _Consts:
    c4: Ptr[float32]      # FFT8 twiddles cos/sin(k pi/4)
    s4: Ptr[float32]
    c2: Ptr[float32]      # FFT4 twiddles
    s2: Ptr[float32]
    c4s: Ptr[float32]     # IFFT8 twiddles * 1/8
    s4s: Ptr[float32]

    def __init__(self):
        self.c4 = Ptr[float32](4)
        self.s4 = Ptr[float32](4)
        self.c2 = Ptr[float32](4)
        self.s2 = Ptr[float32](4)
        self.c4s = Ptr[float32](4)
        self.s4s = Ptr[float32](4)
        cv = (float32(1.0), _S2_2, float32(0.0), -_S2_2)
        sv = (float32(0.0), _S2_2, float32(1.0), _S2_2)
        for l in range(4):
            self.c4[l] = cv[l]
            self.s4[l] = sv[l]
            self.c4s[l] = cv[l] * float32(0.125)
            self.s4s[l] = sv[l] * float32(0.125)
        self.c2[0] = float32(1.0); self.c2[1] = float32(0.0); self.c2[2] = float32(1.0); self.c2[3] = float32(0.0)
        self.s2[0] = float32(0.0); self.s2[1] = float32(1.0); self.s2[2] = float32(0.0); self.s2[3] = float32(1.0)

_K = _Consts()

# ------------------------------------------------------------------ 8-point transforms
# Vector registers live in a scratch area `v` (slots of 4 float32): v + 4*k.

def _fft4_aos(tlo: Ptr[float32], thi: Ptr[float32], st: int, rs: int, rc0: int, v: Ptr[float32]):
    # outputs w0r w0i w1r w1i w2r w2i w3r w3i in slots 0..7 (psimd_fft4_aos_f32)
    w0r = v; w0i = v + 4; w1r = v + 8; w1i = v + 12
    w2r = v + 16; w2i = v + 20; w3r = v + 24; w3i = v + 28
    for k in range(32):
        v[k] = float32(0.0)
    re = rs + rc0
    rc = rc0
    lo = tlo
    hi = thi
    while True:
        if rs <= 0:
            _cpy(w0r, lo); _cpy(w2r, lo); lo += st
            rc -= 1
            if rc == 0: break
        if rs <= 4 and re > 4:
            _cpy(w2r, hi); hi += st; _bfly(w0r, w2r)
            rc -= 1
            if rc == 0: break
        if rs <= 1:
            _cpy(w0i, lo); _cpy(w2i, lo); lo += st
            rc -= 1
            if rc == 0: break
        if rs <= 5 and re > 5:
            _cpy(w2i, hi); hi += st; _bfly(w0i, w2i)
            rc -= 1
            if rc == 0: break
        if rs <= 2:
            _cpy(w1r, lo); _cpy(w3r, lo); lo += st
            rc -= 1
            if rc == 0: break
        if rs <= 6 and re > 6:
            _cpy(w3r, hi); hi += st; _bfly(w1r, w3r)
            rc -= 1
            if rc == 0: break
        if rs <= 3:
            _cpy(w1i, lo); _cpy(w3i, lo); lo += st
            rc -= 1
            if rc == 0: break
        if rs <= 7 and re > 7:
            _cpy(w3i, hi); hi += st; _bfly(w1i, w3i)
            rc -= 1
        break
    _swp(w3r, w3i)
    _bfly(w0r, w1r); _bfly(w0i, w1i); _bfly(w2r, w3r); _bfly_negb(w2i, w3i)
    _swp(w1r, w2r); _swp(w1i, w2i)

def _fft8_real(t0: Ptr[float32], t4: Ptr[float32], st: int, ro: int, rc: int, f: Ptr[float32], sf: int, v: Ptr[float32]):
    _fft4_aos(t0, t4, st, ro, rc, v)
    w0r = _ld(v); w0i = _ld(v + 4); w1r = _ld(v + 8); w1i = _ld(v + 12)
    w2r = _ld(v + 16); w2i = _ld(v + 20); w3r = _ld(v + 24); w3i = _ld(v + 28)
    half = Vec[float32, 4](float32(0.5))
    s24 = Vec[float32, 4](_S2_4)
    g1r = half * (w1r + w3r)
    g1i = half * (w1i - w3i)
    two_h1r = w1i + w3i
    two_h1i = w3r - w1r
    h1p = s24 * (two_h1i + two_h1r)
    h1m = s24 * (two_h1i - two_h1r)
    (w0r + w0i).store(f)
    (w0r - w0i).store(f + sf)
    (g1r + h1p).store(f + 2 * sf)
    (h1m + g1i).store(f + 3 * sf)
    w2r.store(f + 4 * sf)
    (-w2i).store(f + 5 * sf)
    (g1r - h1p).store(f + 6 * sf)
    (h1m - g1i).store(f + 7 * sf)

def _ifft8_real(b: Ptr[float32], col: int, t0: Ptr[float32], t4: Ptr[float32], st: int, v: Ptr[float32]):
    # inputs: rows 0..7 of block b at column group `col` (f0r f4r f1r f1i f2r f2i f3r f3i)
    sc = Vec[float32, 4](float32(0.5))
    bb = b + col * 4
    f0r = _ld(bb) * sc
    f4r = _ld(bb + 8) * sc
    f1r = _ld(bb + 16) * sc
    f1i = _ld(bb + 24) * sc
    f2r = _ld(bb + 32)
    f2i = _ld(bb + 40)
    f3r = _ld(bb + 48) * sc
    f3i = _ld(bb + 56) * sc
    w0r = f0r + f4r
    w0i = f0r - f4r
    w2r = f2r
    w2i = -f2i
    g1r = f1r + f3r
    g1i = f1i - f3i
    h1r = f1r - f3r
    h1i = f1i + f3i
    hp = h1r + h1i
    hm = h1r - h1i
    s22 = Vec[float32, 4](_S2_2)
    ns22 = Vec[float32, 4](-_S2_2)
    w1r = ns22.fma(hp, g1r)
    w1i = s22.fma(hm, g1i)
    w3r = s22.fma(hp, g1r)
    w3i = s22.fma(hm, -g1i)
    # psimd_ifft4_aos_f32
    w1r, w2r = w2r, w1r
    w1i, w2i = w2i, w1i
    w0r, w1r = w0r + w1r, w0r - w1r
    w0i, w1i = w0i + w1i, w0i - w1i
    w2r, w3r = w2r + w3r, w2r - w3r
    w2i, w3i = w2i + w3i, w2i - w3i
    w3r, w3i = w3i, w3r
    q = Vec[float32, 4](float32(0.25))
    w0r = w0r * q; w0i = w0i * q; w1r = w1r * q; w1i = w1i * q
    w2r = w2r * q; w2i = w2i * q; w3r = w3r * q; w3i = w3i * q
    (w0r + w2r).store(t0); (w0r - w2r).store(t4)
    (w0i + w2i).store(t0 + st); (w0i - w2i).store(t4 + st)
    (w1r - w3r).store(t0 + 2 * st); (w1r + w3r).store(t4 + 2 * st)
    (w1i + w3i).store(t0 + 3 * st); (w1i - w3i).store(t4 + 3 * st)

@pure
@llvm
def _s0145(a: Vec[float32, 4], b: Vec[float32, 4]) -> Vec[float32, 4]:
    %r = shufflevector <4 x float> %a, <4 x float> %b, <4 x i32> <i32 0, i32 1, i32 4, i32 5>
    ret <4 x float> %r

@pure
@llvm
def _s2367(a: Vec[float32, 4], b: Vec[float32, 4]) -> Vec[float32, 4]:
    %r = shufflevector <4 x float> %a, <4 x float> %b, <4 x i32> <i32 2, i32 3, i32 6, i32 7>
    ret <4 x float> %r

@pure
@llvm
def _s0246(a: Vec[float32, 4], b: Vec[float32, 4]) -> Vec[float32, 4]:
    %r = shufflevector <4 x float> %a, <4 x float> %b, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
    ret <4 x float> %r

@pure
@llvm
def _s1357(a: Vec[float32, 4], b: Vec[float32, 4]) -> Vec[float32, 4]:
    %r = shufflevector <4 x float> %a, <4 x float> %b, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
    ret <4 x float> %r

@pure
@llvm
def _s0415(a: Vec[float32, 4], b: Vec[float32, 4]) -> Vec[float32, 4]:
    %r = shufflevector <4 x float> %a, <4 x float> %b, <4 x i32> <i32 0, i32 4, i32 1, i32 5>
    ret <4 x float> %r

@pure
@llvm
def _s2637(a: Vec[float32, 4], b: Vec[float32, 4]) -> Vec[float32, 4]:
    %r = shufflevector <4 x float> %a, <4 x float> %b, <4 x i32> <i32 2, i32 6, i32 3, i32 7>
    ret <4 x float> %r

@pure
@llvm
def _s4321(a: Vec[float32, 4], b: Vec[float32, 4]) -> Vec[float32, 4]:
    %r = shufflevector <4 x float> %a, <4 x float> %b, <4 x i32> <i32 4, i32 3, i32 2, i32 1>
    ret <4 x float> %r

@pure
@llvm
def _s4123(a: Vec[float32, 4], b: Vec[float32, 4]) -> Vec[float32, 4]:
    %r = shufflevector <4 x float> %a, <4 x float> %b, <4 x i32> <i32 4, i32 1, i32 2, i32 3>
    ret <4 x float> %r

@inline
def _cmulc_v(xr: Vec[float32, 4], xi: Vec[float32, 4], cr: Vec[float32, 4], ci: Vec[float32, 4]) -> Tuple[Vec[float32, 4], Vec[float32, 4]]:
    return (xr.fma(cr, xi * ci), xi.fma(cr, -(xr * ci)))

@inline
def _cmul_v(xr: Vec[float32, 4], xi: Vec[float32, 4], cr: Vec[float32, 4], ci: Vec[float32, 4]) -> Tuple[Vec[float32, 4], Vec[float32, 4]]:
    return (xr.fma(cr, -(xi * ci)), xi.fma(cr, xr * ci))

@inline
def _fft8_soa_v(w0r: Vec[float32, 4], w4r: Vec[float32, 4], w0i: Vec[float32, 4], w4i: Vec[float32, 4]):
    # psimd_fft8_soa_f32
    w0r, w4r = w0r + w4r, w0r - w4r
    w0i, w4i = w0i + w4i, w0i - w4i
    w4r, w4i = _cmulc_v(w4r, w4i, _ld(_K.c4), _ld(_K.s4))
    a = _s0145(w0r, w4r); bb = _s2367(w0r, w4r)
    a, bb = a + bb, a - bb
    c = _s0145(w0i, w4i); d = _s2367(w0i, w4i)
    c, d = c + d, c - d
    bb, d = _cmulc_v(bb, d, _ld(_K.c2), _ld(_K.s2))
    r0 = _s0246(a, bb); r4 = _s1357(a, bb)
    r0, r4 = r0 + r4, r0 - r4
    i0 = _s0246(c, d); i4 = _s1357(c, d)
    i0, i4 = i0 + i4, i0 - i4
    return (r0, r4, i0, i4)

@inline
def _ifft8_soa_v(r0: Vec[float32, 4], r4: Vec[float32, 4], i0: Vec[float32, 4], i4: Vec[float32, 4]):
    # psimd_ifft8_soa_f32
    r0, r4 = r0 + r4, r0 - r4
    a = _s0415(r0, r4); bb = _s2637(r0, r4)
    i0, i4 = i0 + i4, i0 - i4
    c = _s0415(i0, i4); d = _s2637(i0, i4)
    bb, d = _cmul_v(bb, d, _ld(_K.c2), _ld(_K.s2))
    a, bb = a + bb, a - bb
    r0 = _s0145(a, bb); r4 = _s2367(a, bb)
    c, d = c + d, c - d
    i0 = _s0145(c, d); i4 = _s2367(c, d)
    r4, i4 = _cmul_v(r4, i4, _ld(_K.c4s), _ld(_K.s4s))
    sc = Vec[float32, 4](float32(0.125))
    r0 = r0 * sc
    i0 = i0 * sc
    r0, r4 = r0 + r4, r0 - r4
    i0, i4 = i0 + i4, i0 - i4
    return (r0, r4, i0, i4)

def _fft8_soa(r0: Ptr[float32], r4: Ptr[float32], i0: Ptr[float32], i4: Ptr[float32], v: Ptr[float32]):
    a, b, c, d = _fft8_soa_v(_ld(r0), _ld(r4), _ld(i0), _ld(i4))
    a.store(r0); b.store(r4); c.store(i0); d.store(i4)

def _ifft8_soa(r0: Ptr[float32], r4: Ptr[float32], i0: Ptr[float32], i4: Ptr[float32], v: Ptr[float32]):
    a, b, c, d = _ifft8_soa_v(_ld(r0), _ld(r4), _ld(i0), _ld(i4))
    a.store(r0); b.store(r4); c.store(i0); d.store(i4)

def _fft8_dualreal(s0p: Ptr[float32], s4p: Ptr[float32], h0p: Ptr[float32], h4p: Ptr[float32], v: Ptr[float32]):
    s0, s4, h0, h4 = _fft8_soa_v(_ld(s0p), _ld(s4p), _ld(h0p), _ld(h4p))
    x0765 = _s4321(s4, s0)
    y0765 = _s4321(h4, h0)
    half = Vec[float32, 4](float32(0.5))
    r0246 = half * (s0 + x0765)
    r1357 = half * (h0 + y0765)
    i_246 = half * (h0 - y0765)
    i_357 = half * (x0765 - s0)
    i0246 = _s4123(i_246, s4)
    i1357 = _s4123(i_357, h4)
    _s0415(r0246, r1357).store(s0p); _s2637(r0246, r1357).store(s4p)
    _s0415(i0246, i1357).store(h0p); _s2637(i0246, i1357).store(h4p)

def _ifft8_dualreal(s0p: Ptr[float32], s4p: Ptr[float32], h0p: Ptr[float32], h4p: Ptr[float32], v: Ptr[float32]):
    s0 = _ld(s0p); s4 = _ld(s4p); h0 = _ld(h0p); h4 = _ld(h4p)
    r0246 = _s0246(s0, s4); r1357 = _s1357(s0, s4)
    i0246 = _s0246(h0, h4); i1357 = _s1357(h0, h4)
    a = _s4123(r0246 - i1357, r0246)
    b = _s4321(r0246 + i1357, i0246)
    c = _s4123(r1357 + i0246, r1357)
    d = _s4321(r1357 - i0246, i1357)
    a, b, c, d = _ifft8_soa_v(a, b, c, d)
    a.store(s0p); b.store(s4p); c.store(h0p); d.store(h4p)

# ------------------------------------------------------------------ 2-D 8x8 transforms
# Block b: 8 rows x 8 float32 (row-major), vector (row, col) at b + row*8 + col*4.

def _fft8x8(data: Ptr[float32], tr: Ptr[float32], dstride: int, tstride: int,
            row_count: int, column_count0: int, row_offset: int, column_offset: int,
            b: Ptr[float32], v: Ptr[float32]):
    # nnp_fft8x8_with_offset__psimd (tstride in floats)
    for k in range(64):
        b[k] = float32(0.0)
    column_count = column_count0
    if column_count >= 4:
        row0 = data
        row4 = data + _doz(4, row_offset) * dstride
        out = b + column_offset
        while True:
            cb = column_count if column_count < 4 else 4
            row0 += cb; row4 += cb; out += cb
            _fft8_real(row0 - 4, row4 - 4, dstride, row_offset, row_count, out - 4, 8, v)
            column_count -= cb
            if column_count == 0:
                break
    else:
        for r in range(row_count):
            for c in range(column_count):
                b[(row_offset + r) * 8 + column_offset + c] = data[r * dstride + c]
        col = column_offset if column_offset < 4 else 4
        tmp = v + 64       # 8x4 copy (loads precede stores)
        res = v + 96
        for r in range(8):
            for c in range(4):
                tmp[r * 4 + c] = b[r * 8 + col + c]
        hi = row_offset if row_offset > 4 else 4
        _fft8_real(tmp + row_offset * 4, tmp + hi * 4, 4, row_offset, row_count, res, 4, v)
        for r in range(8):
            for c in range(4):
                b[r * 8 + col + c] = res[r * 4 + c]
    _fft8_dualreal(b, b + 4, b + 8, b + 12, v)
    for row in range(2, 8, 2):
        _fft8_soa(b + row * 8, b + row * 8 + 4, b + row * 8 + 8, b + row * 8 + 12, v)
    t = tr
    for row in range(0, 8, 2):
        for col in range(2):
            _cpy(t, b + row * 8 + col * 4)
            _cpy(t + 4, b + (row + 1) * 8 + col * 4)
            t += tstride

def _ifft8x8_bias(tr: Ptr[float32], data: Ptr[float32], bias: float32, tstride: int, dstride: int,
                  row_count: int, column_count: int, b: Ptr[float32], v: Ptr[float32]):
    # nnp_ifft8x8_with_bias__psimd (tstride in floats)
    t = tr
    for row in range(0, 8, 2):
        for col in range(2):
            _cpy(b + row * 8 + col * 4, t)
            _cpy(b + (row + 1) * 8 + col * 4, t + 4)
            t += tstride
    b[0] = _fma(bias, float32(64.0), b[0])
    _ifft8_dualreal(b, b + 4, b + 8, b + 12, v)
    for row in range(2, 8, 2):
        _ifft8_soa(b + row * 8, b + row * 8 + 4, b + row * 8 + 8, b + row * 8 + 12, v)
    blk = v + 64
    for k in range(64):
        blk[k] = b[k]
    _ifft8_real(blk, 0, b, b + 32, 8, v)
    if column_count > 4:
        _ifft8_real(blk, 1, b + 4, b + 36, 8, v)
    for r in range(row_count):
        for c in range(column_count):
            data[r * dstride + c] = b[r * 8 + c]

# ------------------------------------------------------------------ convolution_output

# nnp_hwinfo on Apple arm64 (init_static_ios_hwinfo): L1 32 KiB; tuple = 8 floats.
_IC_BLOCK_MAX = (32 * 1024 // (8 * 4)) // (2 + 2) // 2 * 2

@pure
@llvm
def _blend_lo2(lo: Vec[float32, 4], hi: Vec[float32, 4]) -> Vec[float32, 4]:
    # lanes 0,1 from lo, lanes 2,3 from hi
    %r = shufflevector <4 x float> %lo, <4 x float> %hi, <4 x i32> <i32 0, i32 1, i32 6, i32 7>
    ret <4 x float> %r

def _cgemm_tuple(acc: Ptr[float32], a: Ptr[float32], b: Ptr[float32], first_tuple: bool):
    # one k-step of nnp_{s4c2,c4}gemm_conjb_*_2x2__neon for a single (row, col):
    # acc += a * conj(b) with fused multiply-adds; tuple 0 holds two real "dual" lanes.
    for l in range(4):
        ar = a[l]
        ai = a[4 + l]
        br = b[l]
        bi = b[4 + l]
        if first_tuple and l < 2:
            acc[l] = _fma(ar, br, acc[l])
            acc[4 + l] = _fma(ai, bi, acc[4 + l])
        else:
            r = _fma(ar, br, acc[l])
            i = _fma(ai, br, acc[4 + l])
            acc[l] = _fma(ai, bi, r)
            acc[4 + l] = _fma(-ar, bi, i)

@inline
def _cstep(a: Ptr[float32], b: Ptr[float32], ar: Vec[float32, 4], ai: Vec[float32, 4], dual: bool) -> Tuple[Vec[float32, 4], Vec[float32, 4]]:
    var = Vec[float32, 4].load(a)
    vai = Vec[float32, 4].load(a + 4)
    vbr = Vec[float32, 4].load(b)
    vbi = Vec[float32, 4].load(b + 4)
    if dual:
        # s4c2 kernels: lanes 0,1 = two real channels, lanes 2,3 complex
        r1 = var.fma(vbr, ar)
        i_dual = vai.fma(vbi, ai)
        i_cplx = vai.fma(vbr, ai)
        r2 = vai.fma(vbi, r1)
        i2 = (-var).fma(vbi, i_cplx)
        return (_blend_lo2(r1, r2), _blend_lo2(i_dual, i2))
    r = var.fma(vbr, ar)
    i = vai.fma(vbr, ai)
    return (vai.fma(vbi, r), (-var).fma(vbi, i))

def _gemm_tile(it: Ptr[float32], kt: Ptr[float32], C: int, n: int, n2: int, o: int, o2: int, t: int, otb: Ptr[float32]):
    # out tuple t for samples n..n+n2-1, channels o..o+o2-1: sum_c a(n,c) * conj(b(o,c)),
    # input-channel blocks of _IC_BLOCK_MAX summed afterwards (compute_matrix_multiplication)
    dual = t == 0
    z = Vec[float32, 4](float32(0.0))
    a0 = it + n * C * 64 + t * 8
    a1 = it + (n + n2 - 1) * C * 64 + t * 8
    b0 = kt + o * C * 64 + t * 8
    b1 = kt + (o + o2 - 1) * C * 64 + t * 8
    c0 = 0
    while c0 < C:
        cb = min(C - c0, _IC_BLOCK_MAX)
        r00 = z; i00 = z; r01 = z; i01 = z; r10 = z; i10 = z; r11 = z; i11 = z
        for c in range(c0, c0 + cb):
            off = c * 64
            r00, i00 = _cstep(a0 + off, b0 + off, r00, i00, dual)
            r01, i01 = _cstep(a0 + off, b1 + off, r01, i01, dual)
            r10, i10 = _cstep(a1 + off, b0 + off, r10, i10, dual)
            r11, i11 = _cstep(a1 + off, b1 + off, r11, i11, dual)
        q = otb + t * 8
        if c0 == 0:
            r00.store(q); i00.store(q + 4)
            r01.store(q + 64); i01.store(q + 68)
            r10.store(q + 128); i10.store(q + 132)
            r11.store(q + 192); i11.store(q + 196)
        else:
            (Vec[float32, 4].load(q) + r00).store(q); (Vec[float32, 4].load(q + 4) + i00).store(q + 4)
            (Vec[float32, 4].load(q + 64) + r01).store(q + 64); (Vec[float32, 4].load(q + 68) + i01).store(q + 68)
            (Vec[float32, 4].load(q + 128) + r10).store(q + 128); (Vec[float32, 4].load(q + 132) + i10).store(q + 132)
            (Vec[float32, 4].load(q + 192) + r11).store(q + 192); (Vec[float32, 4].load(q + 196) + i11).store(q + 196)
        c0 += cb

def conv_ft8x8(N: int, C: int, O: int, H: int, W: int, KH: int, KW: int, ph: int, pw: int,
               x: Ptr[float32], w: Ptr[float32], bias: Ptr[float32], y: Ptr[float32]):
    if __aarch64__:   # native NNPACK in libtyphonrt (pinned torch 2.15 commit)
        # Worker threads never change results (NNPACK per-tile order is fixed); layers
        # with one input channel stay on 1 thread (thread overhead dominates there).
        from C import typhon_nnp_conv_output_mt(int, int, int, int, int, int, int, int, int, Ptr[float32], Ptr[float32], Ptr[float32], Ptr[float32], int) -> int
        from torch._misc import _num_workers
        nt = 1 if C == 1 else _num_workers()
        if typhon_nnp_conv_output_mt(N, C, O, H, W, KH, KW, ph, pw, x, w, bias, y, nt) == 0:
            return
    from torch._misc import _num_workers
    if _num_workers() > 1 and N > 2:
        conv_ft8x8_par(N, C, O, H, W, KH, KW, ph, pw, x, w, bias, y)
        return
    _conv_ft8x8_py(N, C, O, H, W, KH, KW, ph, pw, x, w, bias, y)

def _conv_ft8x8_py(N: int, C: int, O: int, H: int, W: int, KH: int, KW: int, ph: int, pw: int,
                   x: Ptr[float32], w: Ptr[float32], bias: Ptr[float32], y: Ptr[float32]):
    OH = ph + H + ph - KH + 1
    OW = pw + W + pw - KW + 1
    th = 8 - KH + 1
    tw = 8 - KW + 1
    kt = Ptr[float32](O * C * 64)
    it = Ptr[float32](N * C * 64)
    ot = Ptr[float32](64)
    acc = Ptr[float32](8)
    blk = Ptr[float32](64)
    v = Ptr[float32](192)
    for o in range(O):
        for c in range(C):
            _fft8x8(w + (o * C + c) * KH * KW, kt + (o * C + c) * 64, KW, 8, KH, KW, 0, 0, blk, v)
    y0 = 0
    while y0 < OH:
        iy = min(_doz(y0, ph), H)
        x0 = 0
        while x0 < OW:
            ix = min(_doz(x0, pw), W)
            ro = _doz(ph, y0)
            co = _doz(pw, x0)
            rc = min(H - iy, 8 - ro)
            cc = min(W - ix, 8 - co)
            for n in range(N):
                for c in range(C):
                    _fft8x8(x + ((n * C + c) * H + iy) * W + ix, it + (n * C + c) * 64, W, 8, rc, cc, ro, co, blk, v)
            # 2 samples x 2 output channels per inner loop (independent accumulators; each
            # element keeps NNPACK's per-k accumulation order exactly).
            otb = Ptr[float32](4 * 64)
            n = 0
            while n < N:
                n2 = 2 if n + 1 < N else 1
                o = 0
                while o < O:
                    o2 = 2 if o + 1 < O else 1
                    for t in range(8):
                        _gemm_tile(it, kt, C, n, n2, o, o2, t, otb)
                    for dn in range(n2):
                        for do in range(o2):
                            _ifft8x8_bias(otb + (dn * 2 + do) * 64, y + ((((n + dn) * O + o + do) * OH) + y0) * OW + x0,
                                          bias[o + do], 8, OW, min(th, OH - y0), min(tw, OW - x0), blk, v)
                    o += o2
                n += n2
            x0 += tw
        y0 += th

def _ft8x8_pair(n: int, n2: int, C: int, O: int, H: int, W: int, KH: int, KW: int, ph: int, pw: int,
                x: Ptr[float32], kt: Ptr[float32], bias: Ptr[float32], y: Ptr[float32], scr: Ptr[float32]):
    OH = ph + H + ph - KH + 1
    OW = pw + W + pw - KW + 1
    th = 8 - KH + 1
    tw = 8 - KW + 1
    it = scr
    blk = scr + 2 * C * 64
    v = blk + 64
    otb = v + 192
    y0 = 0
    while y0 < OH:
        iy = min(_doz(y0, ph), H)
        x0 = 0
        while x0 < OW:
            ix = min(_doz(x0, pw), W)
            ro = _doz(ph, y0)
            co = _doz(pw, x0)
            rc = min(H - iy, 8 - ro)
            cc = min(W - ix, 8 - co)
            for dn in range(n2):
                for c in range(C):
                    _fft8x8(x + (((n + dn) * C + c) * H + iy) * W + ix, it + (dn * C + c) * 64, W, 8, rc, cc, ro, co, blk, v)
            o = 0
            while o < O:
                o2 = 2 if o + 1 < O else 1
                for t in range(8):
                    _gemm_tile(it, kt, C, 0, n2, o, o2, t, otb)
                for dn in range(n2):
                    for do in range(o2):
                        _ifft8x8_bias(otb + (dn * 2 + do) * 64, y + ((((n + dn) * O + o + do) * OH) + y0) * OW + x0,
                                      bias[o + do], 8, OW, min(th, OH - y0), min(tw, OW - x0), blk, v)
                o += o2
            x0 += tw
        y0 += th

def conv_ft8x8_par(N: int, C: int, O: int, H: int, W: int, KH: int, KW: int, ph: int, pw: int,
                   x: Ptr[float32], w: Ptr[float32], bias: Ptr[float32], y: Ptr[float32]):
    # Scratch is allocated here (main thread) only: worker threads never touch the GC.
    from torch._misc import _num_workers
    nw = _num_workers()
    kt = Ptr[float32](O * C * 64)
    npairs = (N + 1) // 2
    per = 2 * C * 64 + 64 + 192 + 256
    scr = Ptr[float32](npairs * per)
    kscr = Ptr[float32](O * C * 256)
    @par(num_threads=nw, schedule='static', chunk_size=1)
    for oc in range(O * C):
        _fft8x8(w + oc * KH * KW, kt + oc * 64, KW, 8, KH, KW, 0, 0, kscr + oc * 256, kscr + oc * 256 + 64)
    @par(num_threads=nw, schedule='dynamic', chunk_size=1)
    for p in range(npairs):
        n = 2 * p
        n2 = 2 if n + 1 < N else 1
        _ft8x8_pair(n, n2, C, O, H, W, KH, KW, ph, pw, x, kt, bias, y, scr + p * per)
