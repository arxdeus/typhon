# Ported from scipy 1.18.1 (BSD-3-Clause)
# Algorithms from Cephes (BSD-3) / xsf (BSD-3)
#
# Real-argument Bessel kernels (float -> float): j0/j1/y0/y1/k0/k1(+e),
# jn/yn/jv/yv (cephes), iv (cephes scipy_iv: Temme + uniform asymptotics),
# kv (Temme's method), airy, rgamma.

from . import _kernels as K
from ._kernels import exp, log, sqrt, fabs, sin, cos, pow, cbrt, acos
from . import _bessel_const as BC

NAN = float('nan')
INF = float('inf')
PI = 3.14159265358979323846
PI_4 = 0.78539816339744830962
PI_2 = 1.57079632679489661923
M_2_PI = 0.63661977236758134308
SQRT1_2 = 0.70710678118654752440
MACHEP = 1.1102230246251565404236316680908203125E-16
MAXLOG = 7.097827128933839730962063185871E2
SQRT1OPI = 5.641895835477562869480794515607725858441E-1
SQRT2OPI = 7.978845608028653558798921198687637369517E-1
THPIO4 = 2.356194490192344928846982537459627163148
SQRT3 = 1.732050807568877293527446341505872366943
EULER = 0.577215664901532860606512090082402431
MAXGAM = 171.624376956302725
MAXITER = 500
DBL_MAX = 1.7976931348623157e308

from C import fmod(float, float) -> float
from C import sinh(float) -> float
from C import cosh(float) -> float
from C import frexp(float, Ptr[i32]) -> float
from C import round(float) -> float as c_round

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

def _isnan(x: float) -> bool:
    return x != x

def _isinf(x: float) -> bool:
    return x == INF or x == -INF

def sinpi(x: float) -> float:
    s = 1.0
    if K._signbit(x):
        x = -x
        s = -1.0
    r = fmod(x, 2.0)
    if r < 0.5:
        return s * sin(PI * r)
    elif r > 1.5:
        return s * sin(PI * (r - 2.0))
    return -s * sin(PI * (r - 1.0))

def cospi(x: float) -> float:
    if x < 0.0:
        x = -x
    r = fmod(x, 2.0)
    if r == 0.5:
        return 0.0
    if r < 1.0:
        return -sin(PI * (r - 0.5))
    return sin(PI * (r - 1.5))

def rgamma(x: float) -> float:
    if x == 0:
        return x
    if x < 0 and x == K.floor(x):
        return 0.0
    if fabs(x) > 4.0:
        return 1.0 / K.gamma(x)
    z = 1.0
    w = x
    while w > 1.0:
        w -= 1.0
        z *= w
    while w < 0.0:
        z /= w
        w += 1.0
    if w == 0.0:
        return 0.0
    if w == 1.0:
        return 1.0 / z
    return w * (1.0 + K._chbevl(4.0 * w - 2.0, BC._rgamma_R)) / z

# ------------------------------------------------------------------ j0 / y0
def j0(x: float) -> float:
    if x < 0:
        x = -x
    if x <= 5.0:
        z = x * x
        if x < 1.0e-5:
            return 1.0 - z / 4.0
        p = (z - BC._j0_DR1) * (z - BC._j0_DR2)
        return p * _polevl(z, BC._j0_RP) / _p1evl(z, BC._j0_RQ)
    w = 5.0 / x
    q = 25.0 / (x * x)
    p = _polevl(q, BC._j0_PP) / _polevl(q, BC._j0_PQ)
    q = _polevl(q, BC._j0_QP) / _p1evl(q, BC._j0_QQ)
    if x < 10.0:
        xn = x - PI_4
        p = p * cos(xn) - w * q * sin(xn)
        return p * SQRT2OPI / sqrt(x)
    p = (p + w * q) * cos(x) + (p - w * q) * sin(x)
    return p * SQRT1OPI / sqrt(x)

def y0(x: float) -> float:
    if x <= 5.0:
        if x == 0.0:
            return -INF
        elif x < 0.0:
            return NAN
        z = x * x
        w = _polevl(z, BC._j0_YP) / _p1evl(z, BC._j0_YQ)
        w += M_2_PI * log(x) * j0(x)
        return w
    w = 5.0 / x
    z = 25.0 / (x * x)
    p = _polevl(z, BC._j0_PP) / _polevl(z, BC._j0_PQ)
    q = _polevl(z, BC._j0_QP) / _p1evl(z, BC._j0_QQ)
    if x < 10.0:
        xn = x - PI_4
        p = p * sin(xn) + w * q * cos(xn)
        return p * SQRT2OPI / sqrt(x)
    p = (w * q - p) * cos(x) + (p + w * q) * sin(x)
    return p * SQRT1OPI / sqrt(x)

# ------------------------------------------------------------------ j1 / y1
def j1(x: float) -> float:
    if x < 0:
        return -j1(-x)
    if x <= 5.0:
        z = x * x
        w = _polevl(z, BC._j1_RP) / _p1evl(z, BC._j1_RQ)
        return w * x * (z - BC._j1_Z1) * (z - BC._j1_Z2)
    w = 5.0 / x
    z = w * w
    p = _polevl(z, BC._j1_PP) / _polevl(z, BC._j1_PQ)
    q = _polevl(z, BC._j1_QP) / _p1evl(z, BC._j1_QQ)
    if x < 10.0:
        xn = x - THPIO4
        p = p * cos(xn) - w * q * sin(xn)
        return p * SQRT2OPI / sqrt(x)
    a = SQRT1_2 * (w * q - p)
    b = SQRT1_2 * (p + w * q)
    return (a * cos(x) + b * sin(x)) * SQRT2OPI / sqrt(x)

def y1(x: float) -> float:
    if x <= 5.0:
        if x == 0.0:
            return -INF
        elif x <= 0.0:
            return NAN
        z = x * x
        w = x * (_polevl(z, BC._j1_YP) / _p1evl(z, BC._j1_YQ))
        w += M_2_PI * (j1(x) * log(x) - 1.0 / x)
        return w
    w = 5.0 / x
    z = w * w
    p = _polevl(z, BC._j1_PP) / _polevl(z, BC._j1_PQ)
    q = _polevl(z, BC._j1_QP) / _p1evl(z, BC._j1_QQ)
    xn = x - THPIO4
    p = p * sin(xn) + w * q * cos(xn)
    return p * SQRT2OPI / sqrt(x)

# ------------------------------------------------------------------ k0 / k1
def k0(x: float) -> float:
    if x == 0.0:
        return INF
    elif x < 0.0:
        return NAN
    if x <= 2.0:
        y = x * x - 2.0
        return K._chbevl(y, BC._k0_A) - log(0.5 * x) * K.i0(x)
    z = 8.0 / x - 2.0
    return exp(-x) * K._chbevl(z, BC._k0_B) / sqrt(x)

def k0e(x: float) -> float:
    if x == 0.0:
        return INF
    elif x < 0.0:
        return NAN
    if x <= 2.0:
        y = x * x - 2.0
        y = K._chbevl(y, BC._k0_A) - log(0.5 * x) * K.i0(x)
        return y * exp(x)
    return K._chbevl(8.0 / x - 2.0, BC._k0_B) / sqrt(x)

def k1(x: float) -> float:
    if x == 0.0:
        return INF
    elif x < 0.0:
        return NAN
    z = 0.5 * x
    if x <= 2.0:
        y = x * x - 2.0
        return log(z) * K.i1(x) + K._chbevl(y, BC._k1_A) / x
    return exp(-x) * K._chbevl(8.0 / x - 2.0, BC._k1_B) / sqrt(x)

def k1e(x: float) -> float:
    if x == 0.0:
        return INF
    elif x < 0.0:
        return NAN
    if x <= 2.0:
        y = x * x - 2.0
        y = log(0.5 * x) * K.i1(x) + K._chbevl(y, BC._k1_A) / x
        return y * exp(x)
    return K._chbevl(8.0 / x - 2.0, BC._k1_B) / sqrt(x)

# ------------------------------------------------------------------ yn
def yn(nf: float, x: float) -> float:
    if _isnan(nf):
        return nf
    n = int(nf)
    sign = 1.0
    if n < 0:
        n = -n
        if (n & 1) != 0:
            sign = -1.0
    if n == 0:
        return sign * y0(x)
    if n == 1:
        return sign * y1(x)
    if x == 0.0:
        return -INF * sign
    elif x < 0.0:
        return NAN
    anm2 = y0(x)
    anm1 = y1(x)
    k = 1
    r = 2.0 * k
    an = 0.0
    while True:
        an = r * anm1 / x - anm2
        anm2 = anm1
        anm1 = an
        r += 2.0
        k += 1
        if not (k < n and not _isinf(an) and not _isnan(an)):
            break
    return sign * an

# ------------------------------------------------------------------ airy
def airy(x: float):
    """Returns (ai, aip, bi, bip) (cephes airy)."""
    ai = 0.0
    aip = 0.0
    bi = 0.0
    bip = 0.0
    domflg = 0
    if x > BC._MAXAIRY:
        return (0.0, 0.0, INF, INF)
    if x < -2.09:
        t = sqrt(-x)
        zeta = -2.0 * x * t / 3.0
        t = sqrt(t)
        k = SQRT1OPI / t
        z = 1.0 / zeta
        zz = z * z
        uf = 1.0 + zz * _polevl(zz, BC._airy_AFN) / _p1evl(zz, BC._airy_AFD)
        ug = z * _polevl(zz, BC._airy_AGN) / _p1evl(zz, BC._airy_AGD)
        theta = zeta + 0.25 * PI
        f = sin(theta)
        g = cos(theta)
        ai = k * (f * uf - g * ug)
        bi = k * (g * uf + f * ug)
        uf = 1.0 + zz * _polevl(zz, BC._airy_APFN) / _p1evl(zz, BC._airy_APFD)
        ug = z * _polevl(zz, BC._airy_APGN) / _p1evl(zz, BC._airy_APGD)
        k = SQRT1OPI * t
        aip = -k * (g * uf + f * ug)
        bip = k * (f * uf - g * ug)
        return (ai, aip, bi, bip)
    if x >= 2.09:
        domflg = 5
        t = sqrt(x)
        zeta = 2.0 * x * t / 3.0
        g = exp(zeta)
        t = sqrt(t)
        k = 2.0 * t * g
        z = 1.0 / zeta
        f = _polevl(z, BC._airy_AN) / _polevl(z, BC._airy_AD)
        ai = SQRT1OPI * f / k
        k = -0.5 * SQRT1OPI * t / g
        f = _polevl(z, BC._airy_APN) / _polevl(z, BC._airy_APD)
        aip = f * k
        if x > 8.3203353:
            f = z * _polevl(z, BC._airy_BN16) / _p1evl(z, BC._airy_BD16)
            k = SQRT1OPI * g
            bi = k * (1.0 + f) / t
            f = z * _polevl(z, BC._airy_BPPN) / _p1evl(z, BC._airy_BPPD)
            bip = k * t * (1.0 + f)
            return (ai, aip, bi, bip)
    f = 1.0
    g = x
    t = 1.0
    uf = 1.0
    ug = x
    k = 1.0
    z = x * x * x
    while t > MACHEP:
        uf *= z
        k += 1.0
        uf /= k
        ug *= z
        k += 1.0
        ug /= k
        uf /= k
        f += uf
        k += 1.0
        ug /= k
        g += ug
        t = fabs(uf / f)
    uf = BC._airy_c1 * f
    ug = BC._airy_c2 * g
    if (domflg & 1) == 0:
        ai = uf - ug
    if (domflg & 2) == 0:
        bi = SQRT3 * (uf + ug)
    k = 4.0
    uf = x * x / 2.0
    ug = z / 3.0
    f = uf
    g = 1.0 + ug
    uf /= 3.0
    t = 1.0
    while t > MACHEP:
        uf *= z
        ug /= k
        k += 1.0
        ug *= z
        uf /= k
        f += uf
        k += 1.0
        ug /= k
        uf /= k
        g += ug
        k += 1.0
        t = fabs(ug / g)
    uf = BC._airy_c1 * f
    ug = BC._airy_c2 * g
    if (domflg & 4) == 0:
        aip = uf - ug
    if (domflg & 8) == 0:
        bip = SQRT3 * (uf + ug)
    return (ai, aip, bi, bip)

# ------------------------------------------------------------------ iv / kv (scipy_iv.h)
def _iv_asymptotic(v: float, x: float) -> float:
    prefactor = exp(x) / sqrt(2 * PI * x)
    if prefactor == INF:
        return prefactor
    mu = 4 * v * v
    s = 1.0
    term = 1.0
    k = 1
    while True:
        factor = (mu - (2 * k - 1) * (2 * k - 1)) / (8 * x) / k
        if k > 100:
            break
        term *= -factor
        s += term
        k += 1
        if not (fabs(term) > MACHEP * fabs(s)):
            break
    return s * prefactor

def _ikv_asymptotic_uniform(v: float, x: float):
    """Returns (i_value, k_value)."""
    sign = 1
    if v < 0:
        sign = -1
        v = -v
    z = x / v
    t = 1 / sqrt(1 + z * z)
    t2 = t * t
    eta = sqrt(1 + z * z) + log(z / (1 + 1 / t))
    i_prefactor = sqrt(t / (2 * PI * v)) * exp(v * eta)
    i_sum = 1.0
    k_prefactor = sqrt(PI * t / (2 * v)) * exp(-v * eta)
    k_sum = 1.0
    divisor = v
    term = 0.0
    for n in range(1, 11):
        term = 0.0
        k = 31 - 1 - 3 * n
        row = BC._iv_ufactors[n]
        while k < 31 - n:
            term *= t2
            term += row[k]
            k += 2
        k = 1
        while k < n:
            term *= t2
            k += 2
        if n % 2 == 1:
            term *= t
        term /= divisor
        i_sum += term
        k_sum += term if n % 2 == 0 else -term
        if fabs(term) < MACHEP:
            break
        divisor *= v
    kv = k_prefactor * k_sum
    iv = 0.0
    if sign == 1:
        iv = i_prefactor * i_sum
    else:
        iv = i_prefactor * i_sum + (2 / PI) * sinpi(v) * k_prefactor * k_sum
    return (iv, kv)

def _temme_ik_series(v: float, x: float):
    gp = K.gamma(v + 1) - 1
    gm = K.gamma(-v + 1) - 1
    a = log(x / 2)
    b = exp(v * a)
    sigma = -a * v
    c = 1.0 if fabs(v) < MACHEP else sinpi(v) / (v * PI)
    d = 1.0 if fabs(sigma) < MACHEP else sinh(sigma) / sigma
    gamma1 = -EULER if fabs(v) < MACHEP else (0.5 / v) * (gp - gm) * c
    gamma2 = (2 + gp + gm) * c / 2
    p = (gp + 1) / (2 * b)
    q = (1 + gm) * b / 2
    f = (cosh(sigma) * gamma1 + d * (-a) * gamma2) / c
    h = p
    coef = 1.0
    s = coef * f
    s1 = coef * h
    for kk in range(1, MAXITER):
        k = float(kk)
        f = (k * f + p + q) / (k * k - v * v)
        p /= k - v
        q /= k + v
        h = p - k * f
        coef *= x * x / (4 * k)
        s += coef * f
        s1 += coef * h
        if fabs(coef * f) < fabs(s) * MACHEP:
            break
    return (s, 2 * s1 / x)

def _cf1_ik(v: float, x: float) -> float:
    tolerance = 2 * MACHEP
    tiny = 1 / sqrt(DBL_MAX)
    C = tiny
    f = tiny
    D = 0.0
    for kk in range(1, MAXITER):
        a = 1.0
        b = 2 * (v + kk) / x
        C = b + a / C
        D = b + a * D
        if C == 0:
            C = tiny
        if D == 0:
            D = tiny
        D = 1 / D
        delta = C * D
        f *= delta
        if fabs(delta - 1) <= tolerance:
            break
    return f

def _cf2_ik(v: float, x: float):
    tolerance = MACHEP
    a = v * v - 0.25
    b = 2 * (x + 1)
    D = 1 / b
    f = D
    delta = D
    prev = 0.0
    current = 1.0
    Q = -a
    C = -a
    S = 1 + Q * delta
    for kk in range(2, MAXITER):
        k = float(kk)
        a -= 2 * (k - 1)
        b += 2
        D = 1 / (b + a * D)
        delta *= b * D - 1
        f += delta
        q = (prev - (b - 2) * current) / a
        prev = current
        current = q
        C *= -a / k
        Q += C * q
        S += Q * delta
        if fabs(Q * delta) < fabs(S) * tolerance:
            break
    Kv = sqrt(PI / (2 * x)) * exp(-x) / S
    Kv1 = Kv * (0.5 + v + x + (v * v - 0.25) * f) / x
    return (Kv, Kv1)

def _ikv_temme(v: float, x: float, need_i: bool, need_k: bool):
    """Returns (Iv, Kv) (xsf ikv_temme)."""
    reflect = False
    if v < 0:
        reflect = True
        v = -v
        need_k = True
    n = int(c_round(v))
    u = v - n
    if x < 0:
        return (NAN, NAN)
    if x == 0:
        Iv = 1.0 if v == 0 else 0.0
        Kv = INF if need_k else NAN
        if reflect and need_i:
            z = u + n % 2
            Iv = Iv if sinpi(z) == 0 else INF
        return (Iv, Kv)
    W = 1 / x
    Ku = 0.0
    Ku1 = 0.0
    if x <= 2:
        r = _temme_ik_series(u, x)
        Ku = r[0]
        Ku1 = r[1]
    else:
        r = _cf2_ik(u, x)
        Ku = r[0]
        Ku1 = r[1]
    prev = Ku
    current = Ku1
    for k in range(1, n + 1):
        nxt = 2 * (u + k) * current / x + prev
        prev = current
        current = nxt
    Kv = prev
    Kv1 = current
    Iv = NAN
    if need_i:
        lim = (4 * v * v + 10) / (8 * x)
        lim *= lim
        lim *= lim
        lim /= 24
        if lim < MACHEP * 10 and x > 100:
            Iv = _iv_asymptotic(v, x)
        else:
            fv = _cf1_ik(v, x)
            Iv = W / (Kv * fv + Kv1)
    if reflect:
        z = u + n % 2
        return (Iv + (2 / PI) * sinpi(z) * Kv, Kv)
    return (Iv, Kv)

def iv(v: float, x: float) -> float:
    if _isnan(v) or _isnan(x):
        return NAN
    t = K.floor(v)
    if v < 0.0:
        if t == v:
            v = -v
            t = -t
    sign = 1.0
    if x < 0.0:
        if t != v:
            return NAN
        if v != 2.0 * K.floor(v / 2.0):
            sign = -1.0
    if x == 0.0:
        if v == 0.0:
            return 1.0
        if v < 0.0:
            return INF
        return 0.0
    ax = fabs(x)
    res = 0.0
    if fabs(v) > 50:
        res = _ikv_asymptotic_uniform(v, ax)[0]
    else:
        res = _ikv_temme(v, ax, True, False)[0]
    return res * sign

# ------------------------------------------------------------------ jv (cephes jv.h)
def _jv_recur(n: float, x: float, newn: float, cancel: bool):
    """Returns (value, n_out, newn_out)."""
    big = BC._jv_BIG
    maxiter = 22000
    miniter = int(fabs(x) - fabs(n))
    if miniter < 1:
        miniter = 1
    nflag = 1 if n < 0.0 else 0
    ans = 0.0
    while True:
        pkm2 = 0.0
        qkm2 = 1.0
        pkm1 = x
        qkm1 = n + n
        xk = -x * x
        yk = qkm1
        ans = 0.0
        ctr = 0
        while True:
            yk += 2.0
            pk = pkm1 * yk + pkm2 * xk
            qk = qkm1 * yk + qkm2 * xk
            pkm2 = pkm1
            pkm1 = pk
            qkm2 = qkm1
            qkm1 = qk
            r = 0.0
            if qk != 0 and ctr > miniter:
                r = pk / qk
            t = 1.0
            if r != 0:
                t = fabs((ans - r) / r)
                ans = r
            ctr += 1
            if ctr > maxiter:
                break
            if t < MACHEP:
                break
            if fabs(pk) > big:
                pkm2 /= big
                pkm1 /= big
                qkm2 /= big
                qkm1 /= big
            if not (t > MACHEP):
                break
        if ans == 0:
            ans = 1.0
        if nflag > 0 and fabs(ans) < 0.125:
            nflag = -1
            n = n - 1.0
            continue
        break
    kf = newn
    pk = 1.0
    pkm1 = 1.0 / ans
    k = n - 1.0
    r = 2 * k
    pkm2 = 0.0
    while True:
        pkm2 = (pkm1 * r - pk * x) / x
        pk = pkm1
        pkm1 = pkm2
        r -= 2.0
        k -= 1.0
        if not (k > (kf + 0.5)):
            break
    if cancel:
        if kf >= 0.0 and fabs(pk) > fabs(pkm1):
            k += 1.0
            pkm2 = pk
    return (pkm2, n, k)

def _jv_jvs(n: float, x: float) -> float:
    z = -x * x / 4.0
    u = 1.0
    y = u
    k = 1.0
    t = 1.0
    while t > MACHEP:
        u *= z / (k * (n + k))
        y += u
        k += 1.0
        if y != 0:
            t = fabs(u / y)
    ex = i32(0)
    t = frexp(0.5 * x, __ptr__(ex))
    exn = int(int(ex) * n)
    if exn > -1023 and exn < 1023 and n > 0.0 and n < (MAXGAM - 1.0):
        t = pow(0.5 * x, n) * rgamma(n + 1.0)
        y *= t
    else:
        t = n * log(0.5 * x) - K.gammaln(n + 1.0)
        sgngam = K.gammasgn(n + 1.0)
        if y < 0:
            sgngam = -sgngam
            y = -y
        t += log(y)
        if t < -MAXLOG:
            return 0.0
        if t > MAXLOG:
            return INF
        y = sgngam * exp(t)
    return y

def _jv_hankel(n: float, x: float) -> float:
    m = 4.0 * n * n
    j = 1.0
    z = 8.0 * x
    k = 1.0
    p = 1.0
    u = (m - 1.0) / z
    q = u
    sign = 1.0
    conv = 1.0
    flag = 0
    t = 1.0
    pp = 1.0e38
    qq = 1.0e38
    while t > MACHEP:
        k += 2.0
        j += 1.0
        sign = -sign
        u *= (m - k * k) / (j * z)
        p += sign * u
        k += 2.0
        j += 1.0
        u *= (m - k * k) / (j * z)
        q += sign * u
        t = fabs(u / p)
        if t < conv:
            conv = t
            qq = q
            pp = p
            flag = 1
        if flag != 0 and t > conv:
            break
    u = x - (0.5 * n + 0.25) * PI
    return sqrt(2.0 / (PI * x)) * (pp * cos(u) - qq * sin(u))

def _jv_jnt(n: float, x: float) -> float:
    cbn = cbrt(n)
    z = (x - n) / cbn
    cbtwo = cbrt(2.0)
    zz = -cbtwo * z
    a = airy(zz)
    ai = a[0]
    aip = a[1]
    zz = z * z
    z3 = zz * z
    F = (1.0, -z / 5.0, _polevl(z3, BC._jv_PF2) * zz, _polevl(z3, BC._jv_PF3),
         _polevl(z3, BC._jv_PF4) * z)
    G = (0.3 * zz, _polevl(z3, BC._jv_PG1), _polevl(z3, BC._jv_PG2) * z,
         _polevl(z3, BC._jv_PG3) * zz)
    pp = 0.0
    qq = 0.0
    nk = 1.0
    n23 = cbrt(n * n)
    for k in range(5):
        fk = F[k] * nk
        pp += fk
        if k != 4:
            gk = G[k] * nk
            qq += gk
        nk /= n23
    return cbtwo * ai * pp / cbn + cbrt(4.0) * aip * qq / n

def _jv_jnx(n: float, x: float) -> float:
    cbn = cbrt(n)
    z = (x - n) / cbn
    if fabs(z) <= 0.7:
        return _jv_jnt(n, x)
    z = x / n
    zz = 1.0 - z * z
    if zz == 0.0:
        return 0.0
    sz = 0.0
    t = 0.0
    zeta = 0.0
    nflg = 1
    if zz > 0.0:
        sz = sqrt(zz)
        t = 1.5 * (log((1.0 + sz) / z) - sz)
        zeta = cbrt(t * t)
        nflg = 1
    else:
        sz = sqrt(-zz)
        t = 1.5 * (sz - acos(1.0 / z))
        zeta = -cbrt(t * t)
        nflg = -1
    z32i = fabs(1.0 / t)
    sqz = cbrt(t)
    n23 = cbrt(n * n)
    t = n23 * zeta
    a = airy(t)
    ai = a[0]
    aip = a[1]
    u = [0.0] * 8
    u[0] = 1.0
    zzi = 1.0 / zz
    u[1] = _polevl(zzi, BC._jv_P1) / sz
    u[2] = _polevl(zzi, BC._jv_P2) / zz
    u[3] = _polevl(zzi, BC._jv_P3) / (sz * zz)
    pp = zz * zz
    u[4] = _polevl(zzi, BC._jv_P4) / pp
    u[5] = _polevl(zzi, BC._jv_P5) / (pp * sz)
    pp *= zz
    u[6] = _polevl(zzi, BC._jv_P6) / pp
    u[7] = _polevl(zzi, BC._jv_P7) / (pp * sz)
    pp = 0.0
    qq = 0.0
    np_ = 1.0
    doa = True
    dob = True
    akl = INF
    bkl = INF
    for k in range(4):
        tk = 2 * k
        tkp1 = tk + 1
        zp = 1.0
        ak = 0.0
        bk = 0.0
        for s in range(tk + 1):
            if doa:
                sign = float(nflg) if (s & 3) > 1 else 1.0
                ak += sign * BC._jv_mu[s] * zp * u[tk - s]
            if dob:
                m = tkp1 - s
                sign = float(nflg) if ((m + 1) & 3) > 1 else 1.0
                bk += sign * BC._jv_lambda[s] * zp * u[m]
            zp *= z32i
        if doa:
            ak *= np_
            t = fabs(ak)
            if t < akl:
                akl = t
                pp += ak
            else:
                doa = False
        if dob:
            bk += BC._jv_lambda[tkp1] * zp * u[0]
            bk *= -np_ / sqz
            t = fabs(bk)
            if t < bkl:
                bkl = t
                qq += bk
            else:
                dob = False
        if np_ < MACHEP:
            break
        np_ /= n * n
    t = 4.0 * zeta / zz
    t = sqrt(sqrt(t))
    t *= ai * pp / cbrt(n) + aip * qq / (n23 * n)
    return t

def jv_cephes(n: float, x: float) -> float:
    nint = False
    sign = 1.0
    an = fabs(n)
    y = K.floor(an)
    if y == an:
        nint = True
        i = int(an - 16384.0 * K.floor(an / 16384.0))
        if n < 0.0:
            if i & 1:
                sign = -sign
            n = an
        if x < 0.0:
            if i & 1:
                sign = -sign
            x = -x
        if n == 0.0:
            return j0(x)
        if n == 1.0:
            return sign * j1(x)
    if x < 0.0 and y != an:
        return NAN
    if x == 0 and n < 0 and not nint:
        return INF * rgamma(n + 1)
    y = fabs(x)
    if y * y < fabs(n + 1) * MACHEP:
        return pow(0.5 * x, n) * rgamma(n + 1)
    k = 3.6 * sqrt(y)
    t = 3.6 * sqrt(an)
    if y < t and an > 21.0:
        return sign * _jv_jvs(n, x)
    if an < k and y > 21.0:
        return sign * _jv_hankel(n, x)
    if an < 500.0:
        if nint:
            r = _jv_recur(n, x, 0.0, True)
            q = r[0]
            n = r[1]
            k = r[2]
            if k == 0.0:
                return sign * (j0(x) / q)
            if k == 1.0:
                return sign * (j1(x) / q)
        if an > 2.0 * y or (n >= 0.0 and n < 20.0 and y > 6.0 and y < 20.0):
            k = n
            y = y + an + 1.0
            if y < 30.0:
                y = 30.0
            y = n + K.floor(y - n)
            r = _jv_recur(y, x, k, False)
            q = r[0]
            y = r[1]
            return sign * (_jv_jvs(y, x) * q)
        if k <= 30.0:
            k = 2.0
        elif k < 90.0:
            k = (3 * k) / 4
        q = 1.0
        if an > (k + 3.0):
            if n < 0.0:
                k = -k
            q = n - K.floor(n)
            k = K.floor(k) + q
            if n > 0.0:
                r = _jv_recur(n, x, k, True)
                q = r[0]
                n = r[1]
                k = r[2]
            else:
                tt = k
                k = n
                r = _jv_recur(tt, x, k, True)
                q = r[0]
                k = r[1]
            if q == 0.0:
                return sign * 0.0
        else:
            k = n
            q = 1.0
        y = fabs(k)
        if y < 26.0:
            t = (0.0083 * y + 0.09) * y + 12.9
        else:
            t = 0.9 * y
        if x > t:
            y = _jv_hankel(k, x)
        else:
            y = _jv_jvs(k, x)
        if n > 0.0:
            y /= q
        else:
            y *= q
    else:
        if n < 0.0:
            return NAN
        t = x / n
        t /= n
        if t > 0.3:
            y = _jv_hankel(n, x)
        else:
            y = _jv_jnx(n, x)
    return sign * y

def yv_cephes(v: float, x: float) -> float:
    n = int(v) if not _isnan(v) and fabs(v) < 9.2e18 else 0
    if float(n) == v:
        return yn(float(n), x)
    elif v == K.floor(v):
        return NAN
    t = PI * v
    y = (cos(t) * jv_cephes(v, x) - jv_cephes(-v, x)) / sin(t)
    if _isinf(y):
        if v > 0:
            return -INF
        elif v < -1e10:
            return NAN
    return y

# ------------------------------------------------------------------ public real-argument wrappers
# Real-argument kv/kve/ive use Temme's method (the same algorithm Boost/AMOS
# reduce to for real x); jv/yv use the cephes kernels scipy falls back on.

def _kv_scaled(v: float, x: float, scaled: bool) -> float:
    if _isnan(v) or _isnan(x):
        return NAN
    if x < 0:
        return NAN
    if x == 0:
        return INF
    if v < 0:
        v = -v
    if not scaled and x > 710 * (1 + fabs(v)):
        return 0.0
    n = int(c_round(v))
    u = v - n
    Ku = 0.0
    Ku1 = 0.0
    if x <= 2:
        r = _temme_ik_series(u, x)
        Ku = r[0]
        Ku1 = r[1]
        if scaled:
            e = exp(x)
            Ku *= e
            Ku1 *= e
    else:
        r = _cf2_ik_scaled(u, x)
        Ku = r[0]
        Ku1 = r[1]
        if not scaled:
            e = exp(-x)
            Ku *= e
            Ku1 *= e
    prev = Ku
    current = Ku1
    for k in range(1, n + 1):
        nxt = 2 * (u + k) * current / x + prev
        prev = current
        current = nxt
        if _isinf(current):
            if k == n:
                break
            return INF
    return prev

def _cf2_ik_scaled(v: float, x: float):
    # CF2_ik without the exp(-x) factor
    r = _cf2_ik(v, x)
    e = exp(x)
    if r[0] == 0.0 or e == INF:
        # recompute: Kv = sqrt(pi/2x)/S
        return _cf2_ik_noexp(v, x)
    return (r[0] * e, r[1] * e)

def _cf2_ik_noexp(v: float, x: float):
    tolerance = MACHEP
    a = v * v - 0.25
    b = 2 * (x + 1)
    D = 1 / b
    f = D
    delta = D
    prev = 0.0
    current = 1.0
    Q = -a
    C = -a
    S = 1 + Q * delta
    for kk in range(2, MAXITER):
        k = float(kk)
        a -= 2 * (k - 1)
        b += 2
        D = 1 / (b + a * D)
        delta *= b * D - 1
        f += delta
        q = (prev - (b - 2) * current) / a
        prev = current
        current = q
        C *= -a / k
        Q += C * q
        S += Q * delta
        if fabs(Q * delta) < fabs(S) * tolerance:
            break
    Kv = sqrt(PI / (2 * x)) / S
    Kv1 = Kv * (0.5 + v + x + (v * v - 0.25) * f) / x
    return (Kv, Kv1)

def kv(v: float, x: float) -> float:
    return _kv_scaled(v, x, False)

def kve(v: float, x: float) -> float:
    return _kv_scaled(v, x, True)

def kn(n: float, x: float) -> float:
    if _isnan(n):
        return n
    return _kv_scaled(float(int(n)), x, False)

def ive(v: float, x: float) -> float:
    if _isnan(v) or _isnan(x):
        return NAN
    if v != K.floor(v) and x < 0:
        return NAN
    if x == 0.0:
        if v == 0.0:
            return 1.0
        if v < 0.0 and v != K.floor(v):
            return INF
        return 0.0
    sign = 1.0
    ax = fabs(x)
    if x < 0 and fabs(v) % 2.0 == 1.0:
        sign = -1.0
    if ax <= 2.0 or fabs(v) > 50:
        r = iv(v, ax)
        return sign * r * exp(-ax)
    # Temme with scaled K: Iv*exp(-x) = (1/x) / (Ks*fv + Ks1)
    vv = fabs(v)
    n = int(c_round(vv))
    u = vv - n
    r2 = _cf2_ik_noexp(u, ax)
    prev = r2[0]
    current = r2[1]
    for k in range(1, n + 1):
        nxt = 2 * (u + k) * current / ax + prev
        prev = current
        current = nxt
    Kv = prev
    Kv1 = current
    Iv = 0.0
    lim = (4 * vv * vv + 10) / (8 * ax)
    lim *= lim
    lim *= lim
    lim /= 24
    if lim < MACHEP * 10 and ax > 100:
        Iv = _iv_asymptotic_scaled(vv, ax)
    else:
        fv = _cf1_ik(vv, ax)
        Iv = (1 / ax) / (Kv * fv + Kv1)
    if v < 0 and v != K.floor(v):
        z = u + n % 2
        Iv = Iv + (2 / PI) * sinpi(z) * Kv * exp(-2 * ax)
    return sign * Iv

def _iv_asymptotic_scaled(v: float, x: float) -> float:
    prefactor = 1.0 / sqrt(2 * PI * x)
    mu = 4 * v * v
    s = 1.0
    term = 1.0
    k = 1
    while True:
        factor = (mu - (2 * k - 1) * (2 * k - 1)) / (8 * x) / k
        if k > 100:
            break
        term *= -factor
        s += term
        k += 1
        if not (fabs(term) > MACHEP * fabs(s)):
            break
    return s * prefactor

# ------------------------------------------------------------------ spherical (xsf sph_bessel.h)
def spherical_jn(nf: float, x: float) -> float:
    if _isnan(x):
        return x
    n = int(nf)
    if n < 0:
        return NAN
    if _isinf(x):
        return 0.0
    if x == 0:
        return 1.0 if n == 0 else 0.0
    if n > 0 and n >= x:
        return sqrt(PI_2 / x) * jv(n + 0.5, x)
    s0 = sin(x) / x
    if n == 0:
        return s0
    s1 = (s0 - cos(x)) / x
    if n == 1:
        return s1
    sn = 0.0
    for i in range(n - 1):
        sn = (2 * i + 3) * s1 / x - s0
        s0 = s1
        s1 = sn
        if _isinf(sn):
            return sn
    return sn

def spherical_yn(nf: float, x: float) -> float:
    if _isnan(x):
        return x
    n = int(nf)
    if n < 0:
        return NAN
    if x < 0:
        return (1.0 if (n + 1) % 2 == 0 else -1.0) * spherical_yn(nf, -x)
    if _isinf(x):
        return 0.0
    if x == 0:
        return -INF
    s0 = -cos(x) / x
    if n == 0:
        return s0
    s1 = (s0 - sin(x)) / x
    if n == 1:
        return s1
    sn = 0.0
    for idx in range(n - 1):
        sn = (2 * idx + 3) * s1 / x - s0
        s0 = s1
        s1 = sn
        if _isinf(sn):
            return sn
    return sn

def spherical_in(nf: float, x: float) -> float:
    if _isnan(x):
        return x
    n = int(nf)
    if n < 0:
        return NAN
    if x == 0:
        return 1.0 if n == 0 else 0.0
    if _isinf(x):
        if x == -INF:
            return (1.0 if n % 2 == 0 else -1.0) * INF
        return INF
    return sqrt(PI_2 / x) * iv(n + 0.5, x)

def spherical_kn(nf: float, x: float) -> float:
    if _isnan(x):
        return x
    n = int(nf)
    if n < 0:
        return NAN
    if x == 0:
        return INF
    if _isinf(x):
        return 0.0 if x == INF else -INF
    return sqrt(PI_2 / x) * kv(n + 0.5, x)

def spherical_jn_d(nf: float, z: float) -> float:
    n = int(nf)
    if n == 0:
        return -spherical_jn(1.0, z)
    if z == 0.0:
        return 1.0 / 3.0 if n == 1 else 0.0
    return spherical_jn(nf - 1.0, z) - (n + 1) * spherical_jn(nf, z) / z

def spherical_yn_d(nf: float, x: float) -> float:
    n = int(nf)
    if n == 0:
        return -spherical_yn(1.0, x)
    return spherical_yn(nf - 1.0, x) - K._ieee_div((n + 1) * spherical_yn(nf, x), x)

def spherical_in_d(nf: float, z: float) -> float:
    n = int(nf)
    if n == 0:
        return spherical_in(1.0, z)
    if z == 0.0:
        return 1.0 / 3.0 if n == 1 else 0.0
    return spherical_in(nf - 1.0, z) - (n + 1) * spherical_in(nf, z) / z

def spherical_kn_d(nf: float, z: float) -> float:
    n = int(nf)
    if n == 0:
        return -spherical_kn(1.0, z)
    return -spherical_kn(nf - 1.0, z) - K._ieee_div((n + 1) * spherical_kn(nf, z), z)

# ------------------------------------------------------------------ J_v/Y_v by Temme's method (x <= 2) and
# Steed's CF2 (x > 2) with CF1 for J'/J (Temme 1976; Barnett et al.), used for
# non-integer Y_v and the negative-order reflections (scipy uses AMOS there).
def _temme_jy(v: float, x: float):
    """Returns (Y_v, Y_{v+1}) for |v| <= 1/2, 0 < x <= 2."""
    gp = K.gamma(v + 1) - 1
    gm = K.gamma(-v + 1) - 1
    spv = sinpi(v)
    spv2 = sinpi(v / 2)
    xp = pow(x / 2, v)
    a = log(x / 2)
    sigma = -a * v
    d = 1.0 if fabs(sigma) < 2.220446049250313e-16 else sinh(sigma) / sigma
    e = 2 * spv2 * spv2 / v if fabs(v) >= 2.220446049250313e-16 else v * PI * PI / 2
    g1 = -EULER if v == 0 else (gp - gm) / ((1 + gp) * (1 + gm) * 2 * v)
    g2 = (2 + gp + gm) / ((1 + gp) * (1 + gm) * 2)
    vspv = 1.0 / PI if fabs(v) < 2.220446049250313e-16 else v / spv
    f = (g1 * cosh(sigma) - g2 * a * d) * 2 * vspv
    p = vspv / (xp * (1 + gm))
    q = vspv * xp / (1 + gp)
    g = f + e * q
    h = p
    coef = 1.0
    s = coef * g
    s1 = coef * h
    k = 1
    while k < MAXITER * 4:
        f = (k * f + p + q) / (k * k - v * v)
        p /= k - v
        q /= k + v
        g = f + e * q
        h = p - k * g
        coef *= -x * x / (4 * k)
        s += coef * g
        s1 += coef * h
        if fabs(coef * g) < fabs(s) * 2.220446049250313e-16:
            break
        k += 1
    return (-s, -2 * s1 / x)

def _cf1_jy(v: float, x: float):
    """Returns (J'_v/J_v, sign) by modified Lentz."""
    tiny = sqrt(2.2250738585072014e-308)
    C = tiny
    f = tiny
    D = 0.0
    s = 1
    k = 1
    while k < 100000000:
        a = -1.0
        b = 2 * (v + k) / x
        C = b + a / C
        D = b + a * D
        if C == 0:
            C = tiny
        if D == 0:
            D = tiny
        D = 1 / D
        delta = C * D
        f *= delta
        if D < 0:
            s = -s
        if fabs(delta - 1) < 2 * 2.220446049250313e-16:
            break
        k += 1
    return (-f, s)

def _cf2_jy(v: float, x: float):
    """Steed's CF2 (Boost CF2_jy): returns (p, q)."""
    tiny = sqrt(2.2250738585072014e-308)
    tolerance = 2 * 2.220446049250313e-16
    Cr = -0.5 / x
    fr = Cr
    Ci = 1.0
    fi = 1.0
    v2 = v * v
    a = (0.25 - v2) / x
    br = 2 * x
    bi = 2.0
    temp = Cr * Cr + 1
    Ci = bi + a * Cr / temp
    Cr = br + a / temp
    Dr = br
    Di = bi
    if fabs(Cr) + fabs(Ci) < tiny:
        Cr = tiny
    if fabs(Dr) + fabs(Di) < tiny:
        Dr = tiny
    temp = Dr * Dr + Di * Di
    Dr = Dr / temp
    Di = -Di / temp
    delta_r = Cr * Dr - Ci * Di
    delta_i = Ci * Dr + Cr * Di
    temp = fr
    fr = temp * delta_r - fi * delta_i
    fi = temp * delta_i + fi * delta_r
    for k in range(2, 1000000):
        a = float(k) - 0.5
        a *= a
        a -= v2
        bi += 2
        temp = Cr * Cr + Ci * Ci
        Cr = br + a * Cr / temp
        Ci = bi - a * Ci / temp
        Dr = br + a * Dr
        Di = bi + a * Di
        if fabs(Cr) + fabs(Ci) < tiny:
            Cr = tiny
        if fabs(Dr) + fabs(Di) < tiny:
            Dr = tiny
        temp = Dr * Dr + Di * Di
        Dr = Dr / temp
        Di = -Di / temp
        delta_r = Cr * Dr - Ci * Di
        delta_i = Ci * Dr + Cr * Di
        temp = fr
        fr = temp * delta_r - fi * delta_i
        fi = temp * delta_i + fi * delta_r
        if fabs(delta_r - 1) + fabs(delta_i) < tolerance:
            break
    return (fr, fi)

def bessel_jy(v: float, x: float):
    """(J_v(x), Y_v(x)) for v >= 0, x > 0 via Temme (x<=2) or CF1+CF2
    (x>2), the core of Boost's bessel_jy."""
    n = int(c_round(v))
    u = v - n
    W = 2.0 / (x * PI)
    if x <= 2:
        r = _temme_jy(u, x)
        prev = r[0]
        current = r[1]
        for k in range(1, n + 1):
            nxt = 2 * (u + k) / x * current - prev
            prev = current
            current = nxt
        Yv = prev
        Yv1 = current
        fv = _cf1_jy(v, x)[0]
        Jv = W / (Yv * fv - Yv1)
        return (Jv, Yv)
    c1 = _cf1_jy(v, x)
    fv = c1[0]
    s = float(c1[1])
    init = sqrt(2.2250738585072014e-308)
    prev = fv * s * init
    current = s * init
    for kk in range(n, 0, -1):
        nxt = 2 * (u + kk) * current / x - prev
        if nxt == 0:
            nxt = prev * 2.220446049250313e-16 / 2
        prev = current
        current = nxt
    ratio = (s * init) / current
    fu = prev / current
    pq = _cf2_jy(u, x)
    p = pq[0]
    q = pq[1]
    t = u / x - fu
    gamma = (p - t) / q
    if gamma == 0:
        gamma = u * 2.220446049250313e-16 / x
    Ju = sqrt(W / (q + gamma * (p - t)))
    if current < 0:
        Ju = -Ju
    Jv = Ju * ratio
    Yu = gamma * Ju
    Yu1 = Yu * (u / x - p - q / gamma)
    prev = Yu
    current = Yu1
    for k in range(1, n + 1):
        nxt = 2 * (u + k) / x * current - prev
        prev = current
        current = nxt
    return (Jv, prev)

def _jy_nonneg(v: float, x: float):
    # J_v, Y_v for v >= 0 non-integer, x > 0
    return bessel_jy(v, x)

def jv(v: float, x: float) -> float:
    if _isnan(v) or _isnan(x):
        return NAN
    if v != K.floor(v) and x < 0:
        return NAN
    if v >= 0 or v == K.floor(v) or x == 0 or fabs(v) > 60 or x > 1e4:
        return jv_cephes(v, x)
    if v < 0:
        av = -v
        r = _jy_nonneg(av, x)
        return cospi(av) * r[0] - sinpi(av) * r[1]
    return _jy_nonneg(v, x)[0]

def yv(v: float, x: float) -> float:
    if _isnan(v) or _isnan(x):
        return NAN
    if x < 0:
        return NAN
    if x == 0:
        return -INF
    if v == K.floor(v) or fabs(v) > 60 or x > 1e4:
        return yv_cephes(v, x)
    if v < 0:
        av = -v
        r = _jy_nonneg(av, x)
        # rotate_jy(y, j, -v) for Y_{-v}: cos(pi v) Y + sin(pi v) J
        return cospi(av) * r[1] + sinpi(av) * r[0]
    return _jy_nonneg(v, x)[1]

def jn(n: float, x: float) -> float:
    return jv(n, x)

def jve(v: float, x: float) -> float:
    return jv(v, x)

def yve(v: float, x: float) -> float:
    return yv(v, x)


def airye(x: float):
    """Exponentially scaled Airy functions for real x (eAi, eAip, eBi, eBip).
    Real x < 0 gives nan for the Ai pair (complex scale factor), as scipy."""
    if x != x:
        return (x, x, x, x)
    if x < 0:
        r = airy(x)
        return (NAN, NAN, r[2], r[3])
    if x <= 8.3203353:
        r = airy(x)
        zt = 2.0 * x * sqrt(x) / 3.0
        g = exp(zt)
        return (r[0] * g, r[1] * g, r[2] / g, r[3] / g)
    t = sqrt(x)
    zeta = 2.0 * x * t / 3.0
    t = sqrt(t)
    z = 1.0 / zeta
    f = _polevl(z, BC._airy_AN) / _polevl(z, BC._airy_AD)
    eai = SQRT1OPI * f / (2.0 * t)
    f = _polevl(z, BC._airy_APN) / _polevl(z, BC._airy_APD)
    eaip = f * (-0.5 * SQRT1OPI * t)
    f = z * _polevl(z, BC._airy_BN16) / _p1evl(z, BC._airy_BD16)
    ebi = SQRT1OPI * (1.0 + f) / t
    f = z * _polevl(z, BC._airy_BPPN) / _p1evl(z, BC._airy_BPPD)
    ebip = SQRT1OPI * t * (1.0 + f)
    return (eai, eaip, ebi, ebip)

