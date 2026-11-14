# Copyright (C) 2026 the Typhon authors
# Ported from SLEEF 5a1d179 (BSL-1.0), as built into torch 2.15.0 (BSD-3-Clause)
#
# Scalar, bit-exact ports of the AdvSIMD (FMA) float kernels torch's Vectorized<float>
# uses on aarch64: Sleef_expf4_u10 and Sleef_expm1f4_u10 (sleefsimdsp.c xexpf, xexpm1f
# with expk2f and the double-float helpers of common/df.h, ENABLE_FMA_SP).

_L2Uf = float32(0.693145751953125)
_L2Lf = float32(1.428606765330187045e-06)
_R_LN2f = float32(1.442695040888963407359924681001892137426645954152985934135449406931)

def _fma(x: float32, y: float32, z: float32) -> float32:
    return float32._fma(x, y, z)

def _rint_i(d: float32) -> int:
    # vcvtq_s32_f32(vrndnq_f32(d)): round half to even
    from C import rintf(float32) -> float32
    return int(rintf(d))

def _pow2i(q: int) -> float32:
    bits = u32((q + 0x7f) << 23)
    return Ptr[float32](__ptr__(bits).as_byte())[0]

def _ldexp2(d: float32, e: int) -> float32:
    h = e >> 1
    return (d * _pow2i(h)) * _pow2i(e - h)

def expf(d: float32) -> float32:
    q = _rint_i(d * _R_LN2f)
    qf = float32(q)
    s = _fma(qf, -_L2Uf, d)
    s = _fma(qf, -_L2Lf, s)
    u = float32(0.000198527617612853646278381)
    u = _fma(u, s, float32(0.00139304355252534151077271))
    u = _fma(u, s, float32(0.00833336077630519866943359))
    u = _fma(u, s, float32(0.0416664853692054748535156))
    u = _fma(u, s, float32(0.166666671633720397949219))
    u = _fma(u, s, float32(0.5))
    u = float32(1.0) + _fma(s * s, u, s)
    u = _ldexp2(u, q)
    if d < float32(-104.0):
        u = float32(0.0)
    if float32(100.0) < d:
        u = float32(1e300 * 1e300)
    return u

# double-float (hi, lo) helpers, FMA flavour
@inline
def _dfadd2_ff(xh: float32, xl: float32, y: float32) -> Tuple[float32, float32]:
    s = xh + y
    v = s - xh
    t = (xh - (s - v)) + (y - v)
    return (s, t + xl)

@inline
def _dfadd2_f2f2(xh: float32, xl: float32, yh: float32, yl: float32) -> Tuple[float32, float32]:
    s = xh + yh
    v = s - xh
    t = (xh - (s - v)) + (yh - v)
    return (s, t + (xl + yl))

@inline
def _dfadd_f_f2(x: float32, yh: float32, yl: float32) -> Tuple[float32, float32]:
    s = x + yh
    return (s, ((x - s) + yh) + yl)

@inline
def _dfmul_f2f(xh: float32, xl: float32, y: float32) -> Tuple[float32, float32]:
    s = xh * y
    return (s, _fma(xl, y, _fma(xh, y, -s)))

@inline
def _dfmul_f2f2(xh: float32, xl: float32, yh: float32, yl: float32) -> Tuple[float32, float32]:
    s = xh * yh
    return (s, _fma(xh, yl, _fma(xl, yh, _fma(xh, yh, -s))))

@inline
def _dfsqu(xh: float32, xl: float32) -> Tuple[float32, float32]:
    s = xh * xh
    return (s, _fma(xh + xh, xl, _fma(xh, xh, -s)))

def _expk2f(dh: float32, dl: float32) -> Tuple[float32, float32]:
    u = (dh + dl) * _R_LN2f
    q = _rint_i(u)
    qf = float32(q)
    sh, sl = _dfadd2_ff(dh, dl, qf * -_L2Uf)
    sh, sl = _dfadd2_ff(sh, sl, qf * -_L2Lf)
    u = float32(+0.1980960224e-3)
    u = _fma(u, sh, float32(+0.1394256484e-2))
    u = _fma(u, sh, float32(+0.8333456703e-2))
    u = _fma(u, sh, float32(+0.4166637361e-1))
    th, tl = _dfmul_f2f(sh, sl, u)
    th, tl = _dfadd2_ff(th, tl, float32(+0.166666659414234244790680580464e+0))
    ah, al = _dfmul_f2f2(sh, sl, th, tl)
    th, tl = _dfadd2_ff(ah, al, float32(0.5))
    qh, ql = _dfsqu(sh, sl)
    mh, ml = _dfmul_f2f2(qh, ql, th, tl)
    th, tl = _dfadd2_f2f2(sh, sl, mh, ml)
    th, tl = _dfadd_f_f2(float32(1.0), th, tl)
    th = _ldexp2(th, q)
    tl = _ldexp2(tl, q)
    if dh < float32(-104.0):
        th = float32(0.0)
        tl = float32(0.0)
    return (th, tl)

def expm1f(a: float32) -> float32:
    eh, el = _expk2f(a, float32(0.0))
    dh, dl = _dfadd2_ff(eh, el, float32(-1.0))
    x = dh + dl
    if a > float32(88.72283172607421875):
        x = float32(1e300 * 1e300)
    if a < float32(-16.635532333438687426013570):
        x = float32(-1.0)
    if a == float32(0.0) and (Ptr[u32](__ptr__(a).as_byte())[0] >> u32(31)) == u32(1):
        x = float32(-0.0)
    return x

def _bits(x: float32) -> int:
    return int(Ptr[u32](__ptr__(x).as_byte())[0])

def _from_bits(b: int) -> float32:
    u = u32(b & 0xffffffff)
    return Ptr[float32](__ptr__(u).as_byte())[0]

def logf(d0: float32) -> float32:
    # Sleef_logf4_u10 = xlogf_u1 (AdvSIMD, FMA)
    d = d0
    o = d < float32(1.1754943508222875e-38)   # SLEEF_FLT_MIN
    if o:
        d = d * float32(18446744073709551616.0)
    e = ((_bits(d * float32(1.0 / 0.75)) >> 23) & 0xff) - 0x7f
    m = _from_bits(_bits(d) + ((-e) << 23))
    if o:
        e = e - 64
    ef = float32(e)
    # s = dfmul_vf2_vf2_vf((ln2 hi, lo), e)
    s_h = float32(0.69314718246459960938) * ef
    s_l = _fma(float32(-1.904654323148236017e-09), ef, _fma(float32(0.69314718246459960938), ef, -s_h))
    # x = dfdiv(dfadd2(-1, m), dfadd2(1, m))
    nh = float32(-1.0) + m
    v = nh - float32(-1.0)
    nl = (float32(-1.0) - (nh - v)) + (m - v)
    dh = float32(1.0) + m
    v2 = dh - float32(1.0)
    dl = (float32(1.0) - (dh - v2)) + (m - v2)
    t = float32(1.0) / dh
    sx = nh * t
    u = _fma(t, nh, -sx)
    vv = _fma(-dl, t, _fma(-dh, t, float32(1.0)))
    x_h = sx
    x_l = _fma(sx, vv, _fma(nl, t, u))
    x2 = x_h * x_h
    tt = float32(+0.3027294874e+0)
    tt = _fma(tt, x2, float32(+0.3996108174e+0))
    tt = _fma(tt, x2, float32(+0.6666694880e+0))
    # s = dfadd_vf2_vf2_vf2(s, dfscale(x, 2))
    yh = x_h * float32(2.0)
    yl = x_l * float32(2.0)
    s2 = s_h + yh
    s2l = (((s_h - s2) + yh) + s_l) + yl
    # s = dfadd_vf2_vf2_vf(s, x2 * x_h * t)
    z = (x2 * x_h) * tt
    s3 = s2 + z
    s3l = ((s2 - s3) + z) + s2l
    r = s3 + s3l
    if d0 == float32(1e300 * 1e300):
        r = float32(1e300 * 1e300)
    if d0 < float32(0.0) or d0 != d0:
        r = float32(0.0) / float32(0.0)
    if d0 == float32(0.0):
        r = -float32(1e300 * 1e300)
    return r

# ------------------------------------------------------------------ 4-lane (AdvSIMD) versions
# Same operation sequence as the scalar ports above, lane-wise; branches become selects.

from simd import Vec

V4 = Vec[float32, 4]

# Native SLEEF from libtyphonrt (aarch64 builds; 6dbce2b8): process floor(n/4)*4
# elements with Sleef_*f4_u10 and return that count. Elsewhere: the 4-lane .py ports.
_NATIVE = __aarch64__

def expm1f_block(p: Ptr[float32], q: Ptr[float32], n: int) -> int:
    if _NATIVE:
        from C import typhon_sleef_expm1f_v(Ptr[float32], Ptr[float32], int) -> int
        return typhon_sleef_expm1f_v(p, q, n)
    full = n - n % 4
    for i in range(0, full, 4):
        expm1f_v(Vec[float32, 4].load(p + i)).store(q + i)
    return full

def expf_block(p: Ptr[float32], q: Ptr[float32], n: int) -> int:
    if _NATIVE:
        from C import typhon_sleef_expf_v(Ptr[float32], Ptr[float32], int) -> int
        return typhon_sleef_expf_v(p, q, n)
    full = n - n % 4
    for i in range(full):
        q[i] = expf(p[i])
    return full

def logf_block(p: Ptr[float32], q: Ptr[float32], n: int) -> int:
    if _NATIVE:
        from C import typhon_sleef_logf_v(Ptr[float32], Ptr[float32], int) -> int
        return typhon_sleef_logf_v(p, q, n)
    full = n - n % 4
    for i in range(full):
        q[i] = logf(p[i])
    return full

@pure
@llvm
def _v_rndn_i(d: Vec[float32, 4]) -> Vec[i32, 4]:
    declare <4 x float> @llvm.roundeven.v4f32(<4 x float>)
    %r = call <4 x float> @llvm.roundeven.v4f32(<4 x float> %d)
    %q = fptosi <4 x float> %r to <4 x i32>
    ret <4 x i32> %q

@pure
@llvm
def _v_i2f(q: Vec[i32, 4]) -> Vec[float32, 4]:
    %r = sitofp <4 x i32> %q to <4 x float>
    ret <4 x float> %r

@pure
@llvm
def _v_pow2i(q: Vec[i32, 4]) -> Vec[float32, 4]:
    %a = add <4 x i32> %q, <i32 127, i32 127, i32 127, i32 127>
    %b = shl <4 x i32> %a, <i32 23, i32 23, i32 23, i32 23>
    %r = bitcast <4 x i32> %b to <4 x float>
    ret <4 x float> %r

@pure
@llvm
def _v_ldexp2(d: Vec[float32, 4], e: Vec[i32, 4]) -> Vec[float32, 4]:
    %h = ashr <4 x i32> %e, <i32 1, i32 1, i32 1, i32 1>
    %l = sub <4 x i32> %e, %h
    %ha = add <4 x i32> %h, <i32 127, i32 127, i32 127, i32 127>
    %hb = shl <4 x i32> %ha, <i32 23, i32 23, i32 23, i32 23>
    %hf = bitcast <4 x i32> %hb to <4 x float>
    %la = add <4 x i32> %l, <i32 127, i32 127, i32 127, i32 127>
    %lb = shl <4 x i32> %la, <i32 23, i32 23, i32 23, i32 23>
    %lf = bitcast <4 x i32> %lb to <4 x float>
    %m = fmul <4 x float> %d, %hf
    %r = fmul <4 x float> %m, %lf
    ret <4 x float> %r

@pure
@llvm
def _v_sel_lt(a: Vec[float32, 4], b: Vec[float32, 4], x: Vec[float32, 4], y: Vec[float32, 4]) -> Vec[float32, 4]:
    # a < b ? x : y
    %c = fcmp olt <4 x float> %a, %b
    %r = select <4 x i1> %c, <4 x float> %x, <4 x float> %y
    ret <4 x float> %r

@pure
@llvm
def _v_sel_negzero(a: Vec[float32, 4], x: Vec[float32, 4]) -> Vec[float32, 4]:
    # a is -0.0 (bitwise) ? -0.0 : x
    %ai = bitcast <4 x float> %a to <4 x i32>
    %c = icmp eq <4 x i32> %ai, <i32 -2147483648, i32 -2147483648, i32 -2147483648, i32 -2147483648>
    %r = select <4 x i1> %c, <4 x float> <float -0.0, float -0.0, float -0.0, float -0.0>, <4 x float> %x
    ret <4 x float> %r

@inline
def _vf(x: float32) -> Vec[float32, 4]:
    return Vec[float32, 4](x)

@inline
def _v_dfadd2_ff(xh: V4, xl: V4, y: V4):
    s = xh + y
    v = s - xh
    t = (xh - (s - v)) + (y - v)
    return (s, t + xl)

@inline
def _v_dfmul_f2f(xh: V4, xl: V4, y: V4):
    s = xh * y
    return (s, xl.fma(y, xh.fma(y, -s)))

@inline
def _v_dfmul_f2f2(xh: V4, xl: V4, yh: V4, yl: V4):
    s = xh * yh
    return (s, xh.fma(yl, xl.fma(yh, xh.fma(yh, -s))))

@inline
def expm1f_v(a: V4) -> V4:
    # Sleef_expm1f4_u10
    q = _v_rndn_i(a * _vf(_R_LN2f))
    qf = _v_i2f(q)
    sh, sl = _v_dfadd2_ff(a, _vf(float32(0.0)), qf * _vf(-_L2Uf))
    sh, sl = _v_dfadd2_ff(sh, sl, qf * _vf(-_L2Lf))
    u = _vf(float32(+0.1980960224e-3))
    u = u.fma(sh, _vf(float32(+0.1394256484e-2)))
    u = u.fma(sh, _vf(float32(+0.8333456703e-2)))
    u = u.fma(sh, _vf(float32(+0.4166637361e-1)))
    th, tl = _v_dfmul_f2f(sh, sl, u)
    th, tl = _v_dfadd2_ff(th, tl, _vf(float32(+0.166666659414234244790680580464e+0)))
    ah, al = _v_dfmul_f2f2(sh, sl, th, tl)
    th, tl = _v_dfadd2_ff(ah, al, _vf(float32(0.5)))
    s2 = sh * sh
    qh = s2
    ql = (sh + sh).fma(sl, sh.fma(sh, -s2))
    mh, ml = _v_dfmul_f2f2(qh, ql, th, tl)
    s = sh + mh
    v = s - sh
    t = (sh - (s - v)) + (mh - v)
    th = s
    tl = t + (sl + ml)
    one = _vf(float32(1.0))
    s = one + th
    tl = ((one - s) + th) + tl
    th = s
    th = _v_ldexp2(th, q)
    tl = _v_ldexp2(tl, q)
    zero = _vf(float32(0.0))
    lim = _vf(float32(-104.0))
    th = _v_sel_lt(a, lim, zero, th)
    tl = _v_sel_lt(a, lim, zero, tl)
    dh, dl = _v_dfadd2_ff(th, tl, _vf(float32(-1.0)))
    x = dh + dl
    x = _v_sel_lt(_vf(float32(88.72283172607421875)), a, _vf(float32(1e300 * 1e300)), x)
    x = _v_sel_lt(a, _vf(float32(-16.635532333438687426013570)), _vf(float32(-1.0)), x)
    x = _v_sel_negzero(a, x)
    return x
