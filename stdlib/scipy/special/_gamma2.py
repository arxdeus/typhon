# Ported from scipy 1.18.1 (BSD-3-Clause)
# Algorithms from Cephes (BSD-3) / xsf (BSD-3) / specfun (Zhang & Jin)
#
# Gamma-family extras (float -> float): zeta, zetac, expn, exp1, expi,
# spence, poch, owens_t, polygamma.

from . import _kernels as K
from ._kernels import exp, log, sqrt, fabs, sin, cos, pow
from . import _gamma_const as GC
from . import _bessel as B

NAN = float('nan')
INF = float('inf')
PI = 3.14159265358979323846
M_E = 2.71828182845904523536
MACHEP = 1.1102230246251565404236316680908203125E-16
MAXLOG = 7.097827128933839730962063185871E2
SQRT2OPI = 7.978845608028653558798921198687637369517E-1
EULER = 0.577215664901532860606512090082402431

from C import fmod(float, float) -> float
from C import atan(float) -> float
from C import atan2(float, float) -> float

def _isnan(x: float) -> bool:
    return x != x

def _isfinite(x: float) -> bool:
    return x == x and x != INF and x != -INF

def _polevl(x: float, c) -> float:
    r = 0.0
    for v in c:
        r = r * x + v
    return r

def _p1evl(x: float, c) -> float:
    r = 1.0
    for v in c:
        r = r * x + v
    return r

# ------------------------------------------------------------------ zeta
def zeta(x: float, q: float) -> float:
    """Hurwitz zeta (cephes zeta)."""
    if x == 1.0:
        return INF
    if x < 1.0:
        return NAN
    if q <= 0.0:
        if q == K.floor(q):
            return INF
        if x != K.floor(x):
            return NAN
    if q > 1e8:
        return (1 / (x - 1) + 1 / (2 * q)) * pow(q, 1 - x)
    s = pow(q, -x)
    a = q
    i = 0
    b = 0.0
    while i < 9 or a <= 9.0:
        i += 1
        a += 1.0
        b = pow(a, -x)
        s += b
        if fabs(b / s) < MACHEP:
            return s
    w = a
    s += b * w / (x - 1.0)
    s -= 0.5 * b
    a = 1.0
    k = 0.0
    for i in range(12):
        a *= x + k
        b /= w
        t = a * b / GC._zeta_A[i]
        s = s + t
        t = fabs(t / s)
        if t < MACHEP:
            return s
        k += 1.0
        a *= x + k
        b /= w
        k += 1.0
    return s

def _zetac_positive(x: float) -> float:
    if x == 1.0:
        return INF
    if x >= 127:
        return 0.0
    w = K.floor(x)
    if w == x:
        i = int(x)
        if i < 31:
            return GC._azetac[i]
    if x < 1.0:
        w = 1.0 - x
        return _polevl(x, GC._zetac_R) / (w * _p1evl(x, GC._zetac_S))
    if x <= 10.0:
        b = pow(2.0, x) * (x - 1.0)
        w = 1.0 / x
        return (x * _polevl(w, GC._zetac_P)) / (b * _p1evl(w, GC._zetac_Q))
    if x <= 50.0:
        b = pow(2.0, -x)
        w = _polevl(x, GC._zetac_A) / _p1evl(x, GC._zetac_B)
        return exp(w) + b
    s = 0.0
    a = 1.0
    while True:
        a += 2.0
        b = pow(a, -x)
        s += b
        if not (b / s > MACHEP):
            break
    b = pow(2.0, -x)
    return (s + b) / (1.0 - b)

def _zetac_smallneg(x: float) -> float:
    return _polevl(x, GC._zetac_TAYLOR0)

def _zeta_reflection(x: float) -> float:
    hx = x / 2
    if hx == K.floor(hx):
        return 0.0
    x_shift = fmod(x, 4.0)
    small_term = -SQRT2OPI * sin(0.5 * PI * x_shift)
    small_term *= K._lanczos_sum_expg_scaled(x + 1) * zeta(x + 1, 1.0)
    base = (x + K._LANCZOS_G + 0.5) / (2 * PI * M_E)
    large_term = pow(base, x + 0.5)
    if _isfinite(large_term):
        return large_term * small_term
    large_term = pow(base, 0.5 * x + 0.25)
    return (large_term * small_term) * large_term

def zetac(x: float) -> float:
    if _isnan(x):
        return x
    elif x == -INF:
        return NAN
    elif x < 0.0 and x > -0.01:
        return _zetac_smallneg(x)
    elif x < 0.0:
        return _zeta_reflection(-x) - 1
    return _zetac_positive(x)

def riemann_zeta(x: float) -> float:
    if _isnan(x):
        return x
    elif x == -INF:
        return NAN
    elif x < 0.0 and x > -0.01:
        return 1 + _zetac_smallneg(x)
    elif x < 0.0:
        return _zeta_reflection(-x)
    return 1 + _zetac_positive(x)

# ------------------------------------------------------------------ exponential integrals
def exp1(x: float) -> float:
    if x == 0.0:
        return INF
    if x <= 1.0:
        e1 = 1.0
        r = 1.0
        for k in range(1, 26):
            r = -r * k * x / pow(k + 1.0, 2.0)
            e1 += r
            if fabs(r) <= fabs(e1) * 1e-15:
                break
        return -EULER - log(x) + x * e1
    m = 20 + int(80.0 / x)
    t0 = 0.0
    k = m
    while k > 0:
        t0 = k / (1.0 + k / (x + t0))
        k -= 1
    t = 1.0 / (x + t0)
    return exp(-x) * t

def expi(x: float) -> float:
    if x == 0.0:
        return -INF
    elif x < 0:
        return -exp1(-x)
    elif fabs(x) <= 40.0:
        ei = 1.0
        r = 1.0
        for k in range(1, 101):
            r = r * k * x / ((k + 1.0) * (k + 1.0))
            ei += r
            if fabs(r / ei) <= 1.0e-15:
                break
        return EULER + log(x) + x * ei
    elif x == INF:
        return INF
    elif _isnan(x):
        return x
    ei = 1.0
    r = 1.0
    for k in range(1, 21):
        r = r * k / x
        ei += r
    return exp(x) / x * ei

def _expint1_factor_cont_frac(x: float) -> float:
    m = 20 + int(80.0 / x)
    t0 = 0.0
    k = m
    while k > 0:
        t0 = k / (x + k / (1 + t0))
        k -= 1
    return 1 / (1 + t0)

def scaled_exp1(x: float) -> float:
    if x < 0:
        return NAN
    if x == 0:
        return 0.0
    if x <= 1:
        return x * exp(x) * exp1(x)
    if x <= 1250:
        return _expint1_factor_cont_frac(x)
    return 1 + (-1 + (2 + (-6 + (24 - 120 / x) / x) / x) / x) / x

def _expn_large_n(n: int, x: float) -> float:
    p = float(n)
    lam = x / p
    multiplier = 1 / p / (lam + 1) / (lam + 1)
    fac = 1.0
    res = 1.0
    expfac = exp(-lam * p) / (lam + 1) / p
    if expfac == 0:
        return 0.0
    fac *= multiplier
    res += fac
    for k in range(2, 13):
        fac *= multiplier
        r = 0.0
        for j in range(GC._expn_A_off[k], GC._expn_A_off[k + 1]):
            r = r * lam + GC._expn_A_flat[j]
        term = fac * r
        res += term
        if fabs(term) < MACHEP * fabs(res):
            break
    return expfac * res

def expn(nf: float, x: float) -> float:
    if _isnan(nf) or _isnan(x):
        return NAN
    n = int(nf)
    big = 1.44115188075855872E+17
    if n < 0 or x < 0:
        return NAN
    if x > MAXLOG:
        return 0.0
    if x == 0.0:
        if n < 2:
            return INF
        return 1.0 / (n - 1.0)
    if n == 0:
        return exp(-x) / x
    if n > 50:
        return _expn_large_n(n, x)
    if x > 1.0:
        k = 1
        pkm2 = 1.0
        qkm2 = x
        pkm1 = 1.0
        qkm1 = x + n
        ans = pkm1 / qkm1
        while True:
            k += 1
            yk = 0.0
            xk = 0.0
            if k & 1:
                yk = 1.0
                xk = float(n + (k - 1) // 2)
            else:
                yk = x
                xk = float(k // 2)
            pk = pkm1 * yk + pkm2 * xk
            qk = qkm1 * yk + qkm2 * xk
            t = 1.0
            if qk != 0:
                r = pk / qk
                t = fabs((ans - r) / r)
                ans = r
            pkm2 = pkm1
            pkm1 = pk
            qkm2 = qkm1
            qkm1 = qk
            if fabs(pk) > big:
                pkm2 /= big
                pkm1 /= big
                qkm2 /= big
                qkm1 /= big
            if not (t > MACHEP):
                break
        return ans * exp(-x)
    psi = -EULER - log(x)
    for i in range(1, n):
        psi = psi + 1.0 / i
    z = -x
    xk = 0.0
    yk = 1.0
    pk = 1.0 - n
    ans = 0.0 if n == 1 else 1.0 / pk
    while True:
        xk += 1.0
        yk *= z / xk
        pk += 1.0
        if pk != 0.0:
            ans += yk / pk
        t = fabs(yk / ans) if ans != 0.0 else 1.0
        if not (t > MACHEP):
            break
    return (pow(z, float(n - 1)) * psi * B.rgamma(float(n))) - ans

# ------------------------------------------------------------------ spence
def spence(x: float) -> float:
    if x < 0.0:
        return NAN
    if x == 1.0:
        return 0.0
    if x == 0.0:
        return PI * PI / 6.0
    flag = 0
    if x > 2.0:
        x = 1.0 / x
        flag |= 2
    w = 0.0
    if x > 1.5:
        w = (1.0 / x) - 1.0
        flag |= 2
    elif x < 0.5:
        w = -x
        flag |= 1
    else:
        w = x - 1.0
    y = -w * _polevl(w, GC._spence_A) / _polevl(w, GC._spence_B)
    if flag & 1:
        y = (PI * PI) / 6.0 - log(x) * log(1.0 - x) - y
    if flag & 2:
        z = log(x)
        y = -0.5 * z * z - y
    return y

# ------------------------------------------------------------------ poch
def _is_nonpos_int(x: float) -> bool:
    return x <= 0 and x == K.ceil(x) and fabs(x) < 1e13

def poch(a: float, m: float) -> float:
    r = 1.0
    while m >= 1.0:
        if a + m == 1:
            break
        m -= 1.0
        r *= (a + m)
        if not _isfinite(r) or r == 0:
            break
    while m <= -1.0:
        if a + m == 0:
            break
        r /= (a + m)
        m += 1.0
        if not _isfinite(r) or r == 0:
            break
    if m == 0:
        return r
    elif a > 1e4 and fabs(m) <= 1:
        return r * pow(a, m) * (1 + m * (m - 1) / (2 * a) + m * (m - 1) * (m - 2) * (3 * m - 1) / (24 * a * a) +
                                m * m * (m - 1) * (m - 1) * (m - 2) * (m - 3) / (48 * a * a * a))
    if _is_nonpos_int(a + m) and not _is_nonpos_int(a) and a + m != m:
        return INF
    if not _is_nonpos_int(a + m) and _is_nonpos_int(a):
        return 0.0
    return r * exp(K.gammaln(a + m) - K.gammaln(a)) * K.gammasgn(a + m) * K.gammasgn(a)

# ------------------------------------------------------------------ polygamma
def polygamma(n: float, x: float) -> float:
    if n == 0:
        return K.digamma(x)
    s = 1.0 if (n + 1) % 2.0 == 0 else -1.0
    if _isnan(n):
        return NAN
    return s * K.gamma(n + 1.0) * zeta(n + 1, x)

# ------------------------------------------------------------------ owens_t (cephes owens_t.h)
def _ot_get_method(h: float, a: float) -> int:
    ihint = 14
    iaint = 7
    for i in range(14):
        if h <= GC._owens_t_HRANGE[i]:
            ihint = i
            break
    for i in range(7):
        if a <= GC._owens_t_ARANGE[i]:
            iaint = i
            break
    return GC._owens_t_SELECT_METHOD[iaint * 15 + ihint]

def _ot_norm1(x: float) -> float:
    return K.erf(x / sqrt(2.0)) / 2

def _ot_norm2(x: float) -> float:
    return K.erfc(x / sqrt(2.0)) / 2

def _owensT1(h: float, a: float, m: float) -> float:
    j = 1
    jj = 1
    hs = -0.5 * h * h
    dhs = exp(hs)
    as_ = a * a
    aj = a / (2 * PI)
    dj = K.expm1(hs)
    gj = hs * dhs
    val = atan(a) / (2 * PI)
    while True:
        val += dj * aj / jj
        if m <= j:
            break
        j += 1
        jj += 2
        aj *= as_
        dj = gj - dj
        gj *= hs / j
    return val

def _owensT2(h: float, a: float, ah: float, m: float) -> float:
    i = 1
    maxi = 2 * m + 1
    hs = h * h
    as_ = -a * a
    y = 1.0 / hs
    val = 0.0
    vi = a * exp(-0.5 * ah * ah) / sqrt(2 * PI)
    z = (K.ndtr(ah) - 0.5) / h
    while True:
        val += z
        if maxi <= i:
            break
        z = y * (vi - i * z)
        vi *= as_
        i += 2
    val *= exp(-0.5 * hs) / sqrt(2 * PI)
    return val

def _owensT3(h: float, a: float, ah: float) -> float:
    aa = a * a
    hh = h * h
    y = 1 / hh
    vi = a * exp(-ah * ah / 2) / sqrt(2 * PI)
    zi = _ot_norm1(ah) / h
    result = 0.0
    for i in range(31):
        result += zi * GC._owens_t_C[i]
        zi = y * ((2 * i + 1) * zi - vi)
        vi *= aa
    result *= exp(-hh / 2) / sqrt(2 * PI)
    return result

def _owensT4(h: float, a: float, m: float) -> float:
    maxi = 2 * m + 1
    hh = h * h
    naa = -a * a
    i = 1
    ai = a * exp(-hh * (1 - naa) / 2) / (2 * PI)
    yi = 1.0
    result = 0.0
    while True:
        result += ai * yi
        if maxi <= i:
            break
        i += 2
        yi = (1 - hh * yi) / i
        ai *= naa
    return result

def _owensT5(h: float, a: float) -> float:
    result = 0.0
    aa = a * a
    nhh = -0.5 * h * h
    for i in range(1, 14):
        r = 1 + aa * GC._owens_t_PTS[i - 1]
        result += GC._owens_t_WTS[i - 1] * exp(nhh * r) / r
    return result * a

def _owensT6(h: float, a: float) -> float:
    normh = _ot_norm2(h)
    y = 1 - a
    r = atan2(y, (1 + a))
    result = normh * (1 - normh) / 2
    if r != 0:
        result -= r * exp(-y * h * h / (2 * r)) / (2 * PI)
    return result

def _owens_t_dispatch(h: float, a: float, ah: float) -> float:
    if h == 0:
        return atan(a) / (2 * PI)
    if a == 0:
        return 0.0
    if a == 1:
        return _ot_norm2(-h) * _ot_norm2(h) / 2
    index = _ot_get_method(h, a)
    m = GC._owens_t_ORD[index]
    meth = GC._owens_t_METHODS[index]
    if meth == 1:
        return _owensT1(h, a, m)
    if meth == 2:
        return _owensT2(h, a, ah, m)
    if meth == 3:
        return _owensT3(h, a, ah)
    if meth == 4:
        return _owensT4(h, a, m)
    if meth == 5:
        return _owensT5(h, a)
    if meth == 6:
        return _owensT6(h, a)
    return NAN

def owens_t(h: float, a: float) -> float:
    if _isnan(h) or _isnan(a):
        return NAN
    h = fabs(h)
    fabs_a = fabs(a)
    fabs_ah = fabs_a * h
    result = 0.0
    if fabs_a == INF:
        result = 0.5 * _ot_norm2(h)
    elif h == INF:
        result = 0.0
    elif fabs_a <= 1:
        result = _owens_t_dispatch(h, fabs_a, fabs_ah)
    else:
        if fabs_ah <= 0.67:
            normh = _ot_norm1(h)
            normah = _ot_norm1(fabs_ah)
            result = 0.25 - normh * normah - _owens_t_dispatch(fabs_ah, (1 / fabs_a), h)
        else:
            normh = _ot_norm2(h)
            normah = _ot_norm2(fabs_ah)
            result = (normh + normah) / 2 - normh * normah - _owens_t_dispatch(fabs_ah, (1 / fabs_a), h)
    if a < 0:
        return -result
    return result

# ------------------------------------------------------------------ sici / shichi / fresnel (cephes)
def _hyp3f0(a1: float, a2: float, a3: float, z: float) -> float:
    m = pow(z, -1.0 / 3)
    maxiter = int(m) if m < 50 else 50
    term = 1.0
    s = term
    for n in range(maxiter):
        term *= (a1 + n) * (a2 + n) * (a3 + n) * z / (n + 1)
        s += term
        if fabs(term) < 1e-13 * fabs(s) or term == 0:
            break
    err = fabs(term)
    if err > 1e-13 * fabs(s):
        return NAN
    return s

def shichi(x: float):
    """Returns (shi, chi)."""
    from C import sinh(float) -> float
    from C import cosh(float) -> float
    sign = False
    if x < 0.0:
        sign = True
        x = -x
    if x == 0.0:
        return (0.0, -INF)
    s = 0.0
    c = 0.0
    if x >= 8.0:
        if x < 18.0:
            a = (576.0 / x - 52.0) / 10.0
            k = exp(x) / x
            s = k * K._chbevl(a, GC._shichi_S1)
            c = k * K._chbevl(a, GC._shichi_C1)
        elif x <= 88.0:
            a = (6336.0 / x - 212.0) / 70.0
            k = exp(x) / x
            s = k * K._chbevl(a, GC._shichi_S2)
            c = k * K._chbevl(a, GC._shichi_C2)
        else:
            si = 0.0
            ci = 0.0
            if x > 1000:
                si = INF
                ci = INF
            else:
                a = _hyp3f0(0.5, 1., 1., 4.0 / (x * x))
                b = _hyp3f0(1., 1., 1.5, 4.0 / (x * x))
                si = cosh(x) / x * a + sinh(x) / (x * x) * b
                ci = sinh(x) / x * a + cosh(x) / (x * x) * b
            if sign:
                si = -si
            return (si, ci)
    else:
        z = x * x
        a = 1.0
        s = 1.0
        c = 0.0
        k = 2.0
        while True:
            a *= z / k
            c += a / k
            k += 1.0
            a /= k
            s += a / k
            k += 1.0
            if not (fabs(a / s) > MACHEP):
                break
        s *= x
    if sign:
        s = -s
    return (s, EULER + log(x) + c)

def sici(x: float):
    """Returns (si, ci)."""
    PI_2 = 1.57079632679489661923
    sign = False
    if x < 0.0:
        sign = True
        x = -x
    if x == 0.0:
        return (0.0, -INF)
    if x > 1.0e9:
        if x == INF:
            if sign:
                return (-PI_2, NAN)
            return (PI_2, 0.0)
    if x > 4.0:
        s = sin(x)
        c = cos(x)
        z = 1.0 / (x * x)
        f = 0.0
        g = 0.0
        if x < 8.0:
            f = _polevl(z, GC._sici_FN4) / (x * _p1evl(z, GC._sici_FD4))
            g = z * _polevl(z, GC._sici_GN4) / _p1evl(z, GC._sici_GD4)
        else:
            f = _polevl(z, GC._sici_FN8) / (x * _p1evl(z, GC._sici_FD8))
            g = z * _polevl(z, GC._sici_GN8) / _p1evl(z, GC._sici_GD8)
        si = PI_2 - f * c - g * s
        if sign:
            si = -si
        return (si, f * s - g * c)
    z = x * x
    s = x * _polevl(z, GC._sici_SN) / _polevl(z, GC._sici_SD)
    c = z * _polevl(z, GC._sici_CN) / _polevl(z, GC._sici_CD)
    if sign:
        s = -s
    return (s, EULER + log(x) + c)

def fresnel(xxa: float):
    """Returns (S, C)."""
    cc = 0.0
    ss = 0.0
    if xxa == INF or xxa == -INF:
        cc = 0.5
        ss = 0.5
    else:
        x = fabs(xxa)
        x2 = x * x
        if x2 < 2.5625:
            t = x2 * x2
            ss = x * x2 * _polevl(t, GC._fresnl_sn) / _p1evl(t, GC._fresnl_sd)
            cc = x * _polevl(t, GC._fresnl_cn) / _polevl(t, GC._fresnl_cd)
        elif x > 36974.0:
            cc = 0.5 + 1 / (PI * x) * B.sinpi(x * x / 2)
            ss = 0.5 - 1 / (PI * x) * B.cospi(x * x / 2)
        else:
            t = PI * x2
            u = 1.0 / (t * t)
            t = 1.0 / t
            f = 1.0 - u * _polevl(u, GC._fresnl_fn) / _p1evl(u, GC._fresnl_fd)
            g = t * _polevl(u, GC._fresnl_gn) / _p1evl(u, GC._fresnl_gd)
            c = B.cospi(x2 / 2)
            s = B.sinpi(x2 / 2)
            t = PI * x
            cc = 0.5 + (f * s - g * c) / t
            ss = 0.5 - (f * c + g * s) / t
    if xxa < 0.0:
        cc = -cc
        ss = -ss
    return (ss, cc)
