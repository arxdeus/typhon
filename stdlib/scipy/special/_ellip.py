# Ported from scipy 1.18.1 (BSD-3-Clause)
# Algorithms from Cephes (BSD-3) / xsf (BSD-3) / scipy ellint_carlson_cpp_lite
# (BSD-3, Carlson symmetric integrals with compensated arithmetic).
#
# ellipk, ellipkm1, ellipe, ellipkinc, ellipeinc, ellipj, agm and the real
# Carlson integrals elliprc, elliprd, elliprf, elliprg, elliprj.

from . import _kernels as K
from ._kernels import exp, log, sqrt, fabs, sin, cos, tan, pow
from . import _misc as M

from C import fmod(float, float) -> float
from C import atan(float) -> float
from C import asin(float) -> float
from C import asinh(float) -> float
from C import atanh(float) -> float
from C import sinh(float) -> float
from C import cosh(float) -> float
from C import tanh(float) -> float
from C import fma(float, float, float) -> float

NAN = float('nan')
INF = float('inf')
PI = 3.14159265358979323846
PI_2 = 1.57079632679489661923
MACHEP = 1.1102230246251565404236316680908203125E-16
DBL_MIN = 2.2250738585072014e-308

def _isnan(x: float) -> bool:
    return x != x

def _isinf(x: float) -> bool:
    return x == INF or x == -INF

def _polevl(x: float, c) -> float:
    r = 0.0
    for v in c:
        r = r * x + v
    return r

# ------------------------------------------------------------------ complete integrals
_ELLPK_P = (1.37982864606273237150E-4, 2.28025724005875567385E-3, 7.97404013220415179367E-3,
            9.85821379021226008714E-3, 6.87489687449949877925E-3, 6.18901033637687613229E-3,
            8.79078273952743772254E-3, 1.49380448916805252718E-2, 3.08851465246711995998E-2,
            9.65735902811690126535E-2, 1.38629436111989062502E0)
_ELLPK_Q = (2.94078955048598507511E-5, 9.14184723865917226571E-4, 5.94058303753167793257E-3,
            1.54850516649762399335E-2, 2.39089602715924892727E-2, 3.01204715227604046988E-2,
            3.73774314173823228969E-2, 4.88280347570998239232E-2, 7.03124996963957469739E-2,
            1.24999999999870820058E-1, 4.99999999999999999821E-1)
_ELLPK_C1 = 1.3862943611198906188E0

def ellpk(x: float) -> float:
    if x < 0.0:
        return NAN
    if x > 1.0:
        if _isinf(x):
            return 0.0
        return ellpk(1 / x) / sqrt(x)
    if x > MACHEP:
        return _polevl(x, _ELLPK_P) - log(x) * _polevl(x, _ELLPK_Q)
    if x == 0.0:
        return INF
    return _ELLPK_C1 - 0.5 * log(x)

_ELLPE_P = (1.53552577301013293365E-4, 2.50888492163602060990E-3, 8.68786816565889628429E-3,
            1.07350949056076193403E-2, 7.77395492516787092951E-3, 7.58395289413514708519E-3,
            1.15688436810574127319E-2, 2.18317996015557253103E-2, 5.68051945617860553470E-2,
            4.43147180560990850618E-1, 1.00000000000000000299E0)
_ELLPE_Q = (3.27954898576485872656E-5, 1.00962792679356715133E-3, 6.50609489976927491433E-3,
            1.68862163993311317300E-2, 2.61769742454493659583E-2, 3.34833904888224918614E-2,
            4.27180926518931511717E-2, 5.85936634471101055642E-2, 9.37499997197644278445E-2,
            2.49999999999888314361E-1)

def ellpe(x: float) -> float:
    x = 1.0 - x
    if x <= 0.0:
        if x == 0.0:
            return 1.0
        return NAN
    if x > 1.0:
        return ellpe(1.0 - 1 / x) * sqrt(x)
    return _polevl(x, _ELLPE_P) - log(x) * (x * _polevl(x, _ELLPE_Q))

def ellipk(m: float) -> float:
    return ellpk(1.0 - m)

def ellipkm1(p: float) -> float:
    return ellpk(p)

def ellipe(m: float) -> float:
    return ellpe(m)

# ------------------------------------------------------------------ incomplete K
def _ellik_neg_m(phi: float, m: float) -> float:
    n = 0
    mpp = (m * phi) * phi
    if -mpp < 1e-6 and phi < -m:
        return phi + (-mpp * phi * phi / 30.0 + 3.0 * mpp * mpp / 40.0 + mpp / 6.0) * phi
    if -mpp > 4e7:
        sm = sqrt(-m)
        sp = sin(phi)
        cp = cos(phi)
        a = log(4 * sp * sm / (1 + cp))
        b = -(1 + cp / sp / sp - a) / 4 / m
        return (a + b) / sm
    x = 0.0
    y = 0.0
    z = 0.0
    scale = 1.0
    if phi > 1e-153 and m > -1e305:
        s = sin(phi)
        csc2 = 1.0 / (s * s)
        scale = 1.0
        x = 1.0 / (tan(phi) * tan(phi))
        y = csc2 - m
        z = csc2
    else:
        scale = phi
        x = 1.0
        y = 1 - m * scale * scale
        z = 1.0
    if x == y and x == z:
        return scale / sqrt(x)
    A0 = (x + y + z) / 3.0
    A = A0
    x1 = x
    y1 = y
    z1 = z
    Q = 400.0 * max(fabs(A0 - x), max(fabs(A0 - y), fabs(A0 - z)))
    while Q > fabs(A) and n <= 100:
        sx = sqrt(x1)
        sy = sqrt(y1)
        sz = sqrt(z1)
        lam = sx * sy + sx * sz + sy * sz
        x1 = (x1 + lam) / 4.0
        y1 = (y1 + lam) / 4.0
        z1 = (z1 + lam) / 4.0
        A = (x1 + y1 + z1) / 3.0
        n += 1
        Q /= 4
    p4 = float(1 << (2 * n))
    X = (A0 - x) / A / p4
    Y = (A0 - y) / A / p4
    Z = -(X + Y)
    E2 = X * Y - Z * Z
    E3 = X * Y * Z
    return scale * (1.0 - E2 / 10.0 + E3 / 14.0 + E2 * E2 / 24.0 - 3.0 * E2 * E3 / 44.0) / sqrt(A)

def ellik(phi: float, m: float) -> float:
    if _isnan(phi) or _isnan(m):
        return NAN
    if m > 1.0:
        return NAN
    if _isinf(phi) or _isinf(m):
        if _isinf(m) and not _isinf(phi):
            return 0.0
        elif _isinf(phi) and not _isinf(m):
            return phi
        return NAN
    if m == 0.0:
        return phi
    a = 1.0 - m
    if a == 0.0:
        if fabs(phi) >= PI_2:
            return INF
        return asinh(tan(phi))
    npio2 = K.floor(phi / PI_2)
    if fmod(fabs(npio2), 2.0) == 1.0:
        npio2 += 1
    Kc = 0.0
    if npio2 != 0.0:
        Kc = ellpk(a)
        phi = phi - npio2 * PI_2
    sign = 0
    if phi < 0.0:
        phi = -phi
        sign = -1
    temp = 0.0
    if a > 1.0:
        temp = _ellik_neg_m(phi, m)
    else:
        b = sqrt(a)
        t = tan(phi)
        done = False
        if fabs(t) > 10.0:
            e = 1.0 / (b * t)
            if fabs(e) < 10.0:
                e = atan(e)
                if npio2 == 0:
                    Kc = ellpk(a)
                temp = Kc - ellik(e, m)
                done = True
        if not done:
            a = 1.0
            c = sqrt(m)
            d = 1
            mod = 0
            while fabs(c / a) > MACHEP:
                temp = b / a
                phi = phi + atan(t * temp) + mod * PI
                denom = 1.0 - temp * t * t
                if fabs(denom) > 10 * MACHEP:
                    t = t * (1.0 + temp) / denom
                    mod = int((phi + PI_2) / PI)
                else:
                    t = tan(phi)
                    mod = int(K.floor((phi - atan(t)) / PI))
                c = (a - b) / 2.0
                temp = sqrt(a * b)
                a = (a + b) / 2.0
                b = temp
                d += d
            temp = (atan(t) + mod * PI) / (d * a)
    if sign < 0:
        temp = -temp
    temp += npio2 * Kc
    return temp

def ellipkinc(phi: float, m: float) -> float:
    return ellik(phi, m)

# ------------------------------------------------------------------ incomplete E
def _ellie_neg_m(phi: float, m: float) -> float:
    n = 0
    mpp = (m * phi) * phi
    if -mpp < 1e-6 and phi < -m:
        return phi + (mpp * phi * phi / 30.0 - mpp * mpp / 40.0 - mpp / 6.0) * phi
    if -mpp > 1e6:
        sm = sqrt(-m)
        sp = sin(phi)
        cp = cos(phi)
        a = -M.cosm1(phi)
        b1 = log(4 * sp * sm / (1 + cp))
        b = -(0.5 + b1) / 2.0 / m
        c = (0.75 + cp / sp / sp - b1) / 16.0 / m / m
        return (a + b + c) * sm
    x = 0.0
    y = 0.0
    z = 0.0
    scalef = 0.0
    scaled = 0.0
    if phi > 1e-153 and m > -1e200:
        s = sin(phi)
        csc2 = 1.0 / s / s
        scalef = 1.0
        scaled = m / 3.0
        x = 1.0 / tan(phi) / tan(phi)
        y = csc2 - m
        z = csc2
    else:
        scalef = phi
        scaled = mpp * phi / 3.0
        x = 1.0
        y = 1 - mpp
        z = 1.0
    if x == y and x == z:
        return (scalef + scaled / x) / sqrt(x)
    A0f = (x + y + z) / 3.0
    Af = A0f
    A0d = (x + y + 3.0 * z) / 5.0
    Ad = A0d
    x1 = x
    y1 = y
    z1 = z
    seriesd = 0.0
    seriesn = 1.0
    Q = 400.0 * max(fabs(A0f - x), max(fabs(A0f - y), fabs(A0f - z)))
    while Q > fabs(Af) and Q > fabs(Ad) and n <= 100:
        sx = sqrt(x1)
        sy = sqrt(y1)
        sz = sqrt(z1)
        lam = sx * sy + sx * sz + sy * sz
        seriesd += seriesn / (sz * (z1 + lam))
        x1 = (x1 + lam) / 4.0
        y1 = (y1 + lam) / 4.0
        z1 = (z1 + lam) / 4.0
        Af = (x1 + y1 + z1) / 3.0
        Ad = (Ad + lam) / 4.0
        n += 1
        Q /= 4.0
        seriesn /= 4.0
    p4 = float(1 << (2 * n))
    Xf = (A0f - x) / Af / p4
    Yf = (A0f - y) / Af / p4
    Zf = -(Xf + Yf)
    E2f = Xf * Yf - Zf * Zf
    E3f = Xf * Yf * Zf
    ret = scalef * (1.0 - E2f / 10.0 + E3f / 14.0 + E2f * E2f / 24.0 - 3.0 * E2f * E3f / 44.0) / sqrt(Af)
    Xd = (A0d - x) / Ad / p4
    Yd = (A0d - y) / Ad / p4
    Zd = -(Xd + Yd) / 3.0
    E2d = Xd * Yd - 6.0 * Zd * Zd
    E3d = (3 * Xd * Yd - 8.0 * Zd * Zd) * Zd
    E4d = 3.0 * (Xd * Yd - Zd * Zd) * Zd * Zd
    E5d = Xd * Yd * Zd * Zd * Zd
    ret -= scaled * (1.0 - 3.0 * E2d / 14.0 + E3d / 6.0 + 9.0 * E2d * E2d / 88.0 - 3.0 * E4d / 22.0 -
                     9.0 * E2d * E3d / 52.0 + 3.0 * E5d / 26.0) / p4 / Ad / sqrt(Ad)
    ret -= 3.0 * scaled * seriesd
    return ret

def ellie(phi: float, m: float) -> float:
    if _isnan(phi) or _isnan(m):
        return NAN
    if m > 1.0:
        return NAN
    if _isinf(phi):
        return phi
    if _isinf(m):
        return -m
    if m == 0.0:
        return phi
    lphi = phi
    npio2 = K.floor(lphi / PI_2)
    if fmod(fabs(npio2), 2.0) == 1.0:
        npio2 += 1
    lphi = lphi - npio2 * PI_2
    sign = 1
    if lphi < 0.0:
        lphi = -lphi
        sign = -1
    a = 1.0 - m
    E = ellpe(m)
    temp = 0.0
    if a == 0.0:
        temp = sin(lphi)
    elif a > 1.0:
        temp = _ellie_neg_m(lphi, m)
    elif lphi < 0.135:
        m11 = (((((-7.0 / 2816.0) * m + (5.0 / 1056.0)) * m - (7.0 / 2640.0)) * m + (17.0 / 41580.0)) * m - (1.0 / 155925.0)) * m
        m9 = ((((-5.0 / 1152.0) * m + (1.0 / 144.0)) * m - (1.0 / 360.0)) * m + (1.0 / 5670.0)) * m
        m7 = ((-m / 112.0 + (1.0 / 84.0)) * m - (1.0 / 315.0)) * m
        m5 = (-m / 40.0 + (1.0 / 30)) * m
        m3 = -m / 6.0
        p2 = lphi * lphi
        temp = ((((m11 * p2 + m9) * p2 + m7) * p2 + m5) * p2 + m3) * p2 * lphi + lphi
    else:
        t = tan(lphi)
        b = sqrt(a)
        done = False
        if fabs(t) > 10.0:
            e = 1.0 / (b * t)
            if fabs(e) < 10.0:
                e = atan(e)
                temp = E + m * sin(lphi) * sin(e) - ellie(e, m)
                done = True
        if not done:
            c = sqrt(m)
            a = 1.0
            d = 1
            e = 0.0
            mod = 0
            while fabs(c / a) > MACHEP:
                temp = b / a
                lphi = lphi + atan(t * temp) + mod * PI
                denom = 1 - temp * t * t
                if fabs(denom) > 10 * MACHEP:
                    t = t * (1.0 + temp) / denom
                    mod = int((lphi + PI_2) / PI)
                else:
                    t = tan(lphi)
                    mod = int(K.floor((lphi - atan(t)) / PI))
                c = (a - b) / 2.0
                temp = sqrt(a * b)
                a = (a + b) / 2.0
                b = temp
                d += d
                e += c * sin(lphi)
            temp = E / ellpk(1.0 - m)
            temp *= (atan(t) + mod * PI) / (d * a)
            temp += e
    if sign < 0:
        temp = -temp
    temp += npio2 * E
    return temp

def ellipeinc(phi: float, m: float) -> float:
    return ellie(phi, m)

# ------------------------------------------------------------------ Jacobi elliptic
def ellipj(u: float, m: float):
    """Returns (sn, cn, dn, ph) (cephes ellpj)."""
    if m < 0.0 or m > 1.0 or _isnan(m):
        return (NAN, NAN, NAN, NAN)
    if m < 1.0e-9:
        t = sin(u)
        b = cos(u)
        ai = 0.25 * m * (u - t * b)
        return (t - ai * b, b + ai * t, 1.0 - 0.5 * m * t * t, u - ai)
    if m >= 0.9999999999:
        ai = 0.25 * (1.0 - m)
        b = cosh(u)
        t = tanh(u)
        phi = 1.0 / b
        twon = b * sinh(u)
        sn = t + ai * (twon - u) / (b * b)
        ph = 2.0 * atan(exp(u)) - PI_2 + ai * (twon - u) / b
        ai *= t * phi
        return (sn, phi - ai * (twon - u), phi + ai * (twon + u), ph)
    a = [0.0] * 9
    c = [0.0] * 9
    a[0] = 1.0
    b = sqrt(1.0 - m)
    c[0] = sqrt(m)
    twon = 1.0
    i = 0
    while fabs(c[i] / a[i]) > MACHEP:
        if i > 7:
            break
        ai = a[i]
        i += 1
        c[i] = (ai - b) / 2.0
        t = sqrt(ai * b)
        a[i] = (ai + b) / 2.0
        b = t
        twon *= 2.0
    phi = twon * a[i] * u
    t = 0.0
    while True:
        t = c[i] * sin(phi) / a[i]
        b = phi
        phi = (asin(t) + phi) / 2.0
        i -= 1
        if i == 0:
            break
    sn = sin(phi)
    t = cos(phi)
    cn = t
    dnfac = cos(phi - b)
    dn = 0.0
    if fabs(dnfac) < 0.1:
        dn = sqrt(1 - m * sn * sn)
    else:
        dn = t / dnfac
    return (sn, cn, dn, phi)

# ------------------------------------------------------------------ agm (scipy _agm.pxd)
def _agm_iter(a: float, b: float) -> float:
    count = 20
    amean = 0.5 * a + 0.5 * b
    while count > 0 and (amean != a and amean != b):
        gmean = sqrt(a) * sqrt(b)
        a = amean
        b = gmean
        amean = 0.5 * a + 0.5 * b
        count -= 1
    return amean

def agm(a: float, b: float) -> float:
    sqrthalfmax = 9.480751908109176e+153
    invsqrthalfmax = 1.0547686614863e-154
    if _isnan(a) or _isnan(b):
        return NAN
    if (a < 0 and b > 0) or (a > 0 and b < 0):
        return NAN
    if (_isinf(a) or _isinf(b)) and (a == 0 or b == 0):
        return NAN
    if a == 0 or b == 0:
        return 0.0
    if a == b:
        return a
    sgn = 1.0
    if a < 0:
        sgn = -1.0
        a = -a
        b = -b
    if invsqrthalfmax < a and a < sqrthalfmax and invsqrthalfmax < b and b < sqrthalfmax:
        e = 4 * a * b / ((a + b) * (a + b))
        return sgn * (PI / 4) * (a + b) / ellpk(e)
    return sgn * _agm_iter(a, b)

# ------------------------------------------------------------------ Carlson (real), ellint_carlson_cpp_lite
RERR = 5e-16
MAX_ITER = 1000
RC_C = (80080.0, 0.0, 24024.0, 11440.0, 30030.0, 32760.0, 61215.0, 90090.0)
RF_C1 = (0.0, -24024.0, 10010.0, -5775.0)
RF_C2 = (17160.0, -16380.0, 15015.0)
RF_c33 = 6930.0
RF_DENOM = 240240.0
RDJ_C1 = (0.0, -875160.0, 417690.0, -255255.0)
RDJ_C2 = (0.0, 680680.0, 306306.0)
RDJ_C3 = (0.0, -706860.0, 675675.0)
RDJ_C4 = (-556920.0, 612612.0)
RDJ_C5 = (471240.0, -540540.0)
RDJ_DENOM = 4084080.0

def _too_small(x: float) -> bool:
    return x == 0.0 or fabs(x) < DBL_MIN

def _ph_good(x: float) -> bool:
    return x >= 0.0

def _ocrt(x: float) -> float:
    return sqrt(sqrt(sqrt(x)))

def _sum2(xs) -> float:
    # compensated sum (TwoSum accumulator)
    p = 0.0
    s = 0.0
    for v in xs:
        t = v + p
        z = t - v
        e = (v - (t - z)) + (p - z)
        p = t
        s += e
    return p + s

def _dot2(xs, ys) -> float:
    p = 0.0
    s = 0.0
    for i in range(len(xs)):
        h = xs[i] * ys[i]
        r = fma(xs[i], ys[i], -h)
        t = p + h
        z = t - p
        q = (p - (t - z)) + (h - z)
        p = t
        s += q + r
    return p + s

def _comp_horner(x: float, poly) -> float:
    # coefficients in ascending order of power
    degree = len(poly) - 1
    s = poly[degree]
    r = 0.0
    i = degree - 1
    while i >= 0:
        ptmp = s * x
        pp = fma(s, x, -ptmp)
        sn = ptmp + poly[i]
        z = sn - ptmp
        ps = (ptmp - (sn - z)) + (poly[i] - z)
        s = sn
        r = r * x + (pp + ps)
        i -= 1
    return s + r

def _sort3(a: float, b: float, c: float):
    # ascending by absolute value (stable as std::sort for distinct keys)
    if fabs(b) < fabs(a):
        t = a
        a = b
        b = t
    if fabs(c) < fabs(b):
        t = b
        b = c
        c = t
        if fabs(b) < fabs(a):
            t = a
            a = b
            b = t
    return (a, b, c)

def _sort3r(a: float, b: float, c: float):
    if b < a:
        t = a
        a = b
        b = t
    if c < b:
        t = b
        b = c
        c = t
        if b < a:
            t = a
            a = b
            b = t
    return (a, b, c)

def _rc(x: float, y: float, rerr: float) -> float:
    if y < 0.0:
        tmp = _rc(x - y, -y, rerr)
        return tmp * sqrt(x / (x - y))
    elif _too_small(y) or not _ph_good(x):
        return NAN
    elif _isinf(x) or _isinf(y):
        return 0.0
    Am = (x + 2.0 * y) / 3.0
    fterm = fabs(Am - x) / _ocrt(3.0 * rerr)
    xm = x
    ym = y
    sm = y - Am
    m = 0
    while max(fabs(xm - ym), fterm) >= fabs(Am):
        if m > MAX_ITER:
            break
        lam = 2.0 * sqrt(xm) * sqrt(ym) + ym
        Am = (Am + lam) * 0.25
        xm = (xm + lam) * 0.25
        ym = (ym + lam) * 0.25
        sm *= 0.25
        fterm *= 0.25
        m += 1
    Am = (xm + ym + ym) / 3.0
    sm /= Am
    return _comp_horner(sm, RC_C) / (sqrt(Am) * RC_C[0])

def elliprc(x: float, y: float) -> float:
    if _isnan(x) or _isnan(y):
        return NAN
    return _rc(x, y, RERR)

def _rf0(x: float, y: float, rerr: float) -> float:
    rsq = 2.0 * sqrt(rerr)
    xm = sqrt(x)
    ym = sqrt(y)
    m = 0
    while fabs(xm - ym) >= rsq * min(fabs(xm), fabs(ym)):
        if m > MAX_ITER:
            break
        xn = (xm + ym) * 0.5
        yn = sqrt(xm * ym)
        xm = xn
        ym = yn
        m += 1
    return PI / (xm + ym)

def _rf(x: float, y: float, z: float, rerr: float) -> float:
    if _ph_good(x) and _ph_good(y) and _ph_good(z):
        if _isinf(x) or _isinf(y) or _isinf(z):
            return 0.0
    else:
        return NAN
    s3 = _sort3(x, y, z)
    xm = s3[0]
    ym = s3[1]
    zm = s3[2]
    if _too_small(xm):
        if _too_small(ym):
            return INF
        tmp = _rf0(ym, zm, rerr * 0.5)
        return tmp - sqrt(xm / (ym * zm))
    Am = _sum2((xm, ym, zm)) / 3.0
    xxm = Am - xm
    yym = Am - ym
    fterm = max(fabs(xxm), max(fabs(yym), fabs(Am - zm))) / _ocrt(3.0 * rerr)
    m = 0
    while True:
        aAm = fabs(Am)
        if not (aAm <= fterm or aAm <= max(fabs(xxm), max(fabs(yym), fabs(Am - zm)))):
            break
        if m > MAX_ITER:
            break
        sx = sqrt(xm)
        sy = sqrt(ym)
        sz = sqrt(zm)
        lam = _dot2((sx, sy, sz), (sy, sz, sx))
        Am = (Am + lam) * 0.25
        xm = (xm + lam) * 0.25
        ym = (ym + lam) * 0.25
        zm = (zm + lam) * 0.25
        xxm *= 0.25
        yym *= 0.25
        fterm *= 0.25
        m += 1
    Am = _sum2((xm, ym, zm)) / 3.0
    xxm /= Am
    yym /= Am
    zzm = -(xxm + yym)
    e2 = xxm * yym - zzm * zzm
    e3 = xxm * (yym * zzm)
    s = _comp_horner(e2, RF_C1)
    s += e3 * (_comp_horner(e2, RF_C2) + e3 * RF_c33)
    s /= RF_DENOM
    s += 1.0
    return s / sqrt(Am)

def elliprf(x: float, y: float, z: float) -> float:
    if _isnan(x) or _isnan(y) or _isnan(z):
        return NAN
    return _rf(x, y, z, RERR)

def _rd(x: float, y: float, z: float, rerr: float) -> float:
    if not (_ph_good(x) and _ph_good(y) and _ph_good(z)):
        return NAN
    if _too_small(z):
        return INF
    if _isinf(x) or _isinf(y) or _isinf(z):
        return 0.0
    if _too_small(x) and _too_small(y):
        return INF
    Am = _sum2((x, y, z, z, z)) / 5.0
    xm = x
    ym = y
    zm = z
    xxm = Am - xm
    yym = Am - ym
    fterm = max(fabs(xxm), max(fabs(yym), fabs(Am - z))) / _ocrt(rerr / 5.0)
    d4m = 1.0
    adt = 0.0
    ade = 0.0
    m = 0
    while True:
        aAm = fabs(Am)
        if not (aAm <= fterm or aAm <= max(fabs(xxm), max(fabs(yym), fabs(Am - zm)))):
            break
        if m > MAX_ITER:
            break
        sx = sqrt(xm)
        sy = sqrt(ym)
        sz = sqrt(zm)
        lam = _dot2((sx, sy, sz), (sy, sz, sx))
        tmp = d4m / (sz * (zm + lam))
        t = tmp + adt
        zz = t - tmp
        e = (tmp - (t - zz)) + (adt - zz)
        adt = t
        ade += e
        Am = (Am + lam) * 0.25
        xm = (xm + lam) * 0.25
        ym = (ym + lam) * 0.25
        zm = (zm + lam) * 0.25
        xxm *= 0.25
        yym *= 0.25
        fterm *= 0.25
        d4m *= 0.25
        m += 1
    Am = _sum2((xm, ym, zm, zm, zm)) / 5.0
    xxm /= Am
    yym /= Am
    zzm = (xxm + yym) / (-3.0)
    xy = xxm * yym
    zz2 = zzm * zzm
    e2 = xy - zz2 * 6.0
    e3 = (xy * 3.0 - zz2 * 8.0) * zzm
    e4 = (xy - zz2) * zz2 * 3.0
    e5 = xy * zz2 * zzm
    t = sqrt(Am)
    tmp = d4m / (t * t * t)
    c = (_comp_horner(e2, RDJ_C1), _comp_horner(e3, RDJ_C2), _comp_horner(e2, RDJ_C3),
         _comp_horner(e2, RDJ_C4), _comp_horner(e2, RDJ_C5), e3 * RDJ_C5[1])
    t = _dot2(c, (1.0, 1.0, e3, e4, e5, e4)) / RDJ_DENOM + 1.0
    return _dot2((tmp, 3.0, 3.0), (t, adt, ade))

def elliprd(x: float, y: float, z: float) -> float:
    if _isnan(x) or _isnan(y) or _isnan(z):
        return NAN
    return _rd(x, y, z, RERR)

def _rg0(x: float, y: float, rerr: float) -> float:
    rsq = 2.0 * sqrt(rerr)
    fac = 0.25
    xm = sqrt(x)
    ym = sqrt(y)
    dm = (xm + ym) * 0.5
    sm = -dm * dm
    cor = 0.0
    dm = xm - ym
    m = 0
    while fabs(dm) >= rsq * min(fabs(xm), fabs(ym)):
        if m > MAX_ITER:
            break
        xn = (xm + ym) * 0.5
        yn = sqrt(xm * ym)
        xm = xn
        ym = yn
        fac *= 2.0
        dm = xm - ym
        term = dm * dm
        h = fac * term
        r = fma(fac, term, -h)
        t = sm + h
        z = t - sm
        q = (sm - (t - z)) + (h - z)
        sm = t
        cor += q + r
        m += 1
    res = PI / (xm + ym)
    return res * (-0.5 * (sm + cor))

def _rg(x: float, y: float, z: float, rerr: float) -> float:
    s3 = _sort3(x, y, z)
    if (_isinf(s3[0]) or _isinf(s3[1]) or _isinf(s3[2])) and _ph_good(s3[0]) and _ph_good(s3[1]) and _ph_good(s3[2]):
        return INF
    if _too_small(s3[0]):
        if _too_small(s3[1]):
            return sqrt(s3[2]) * 0.5
        return _rg0(s3[1], s3[2], rerr)
    r0 = _rd(y, z, x, rerr)
    r1 = _rd(z, x, y, rerr)
    r2 = _rd(x, y, z, rerr)
    if _isnan(r0) or _isnan(r1) or _isnan(r2):
        return NAN
    c0 = _dot2((x, x), (y, z))
    c1 = _dot2((y, y), (x, z))
    c2 = _dot2((z, z), (x, y))
    return _dot2((c0, c1, c2), (r0, r1, r2)) / 6.0

def elliprg(x: float, y: float, z: float) -> float:
    if _isnan(x) or _isnan(y) or _isnan(z):
        return NAN
    return _rg(x, y, z, RERR)

def _safe_atan_sqrt_div(x: float) -> float:
    if _too_small(x):
        return 1.0
    if x < 0.0:
        s = sqrt(-x)
        return atanh(s) / s
    s = sqrt(x)
    return atan(s) / s

def _asymp_zero(r: float) -> bool:
    return 0.0 < r and r <= 5.0e-14

def _abs_close_zero(r: float) -> bool:
    return 0.0 < r and r <= 1.0e-9

def _rj(x: float, y: float, z: float, p: float, rerr: float, noasymp: bool) -> float:
    s3 = _sort3r(x, y, z)
    xm = s3[0]
    ym = s3[1]
    zm = s3[2]
    # good_args (real)
    if _too_small(xm) and _too_small(ym) and _ph_good(zm) and not _too_small(p):
        return INF
    if (_isinf(xm) or _isinf(ym) or _isinf(zm) or _isinf(p)) and (_ph_good(xm) and _ph_good(ym) and _ph_good(zm)):
        return 0.0
    maybe_asymp = False
    good = False
    nonneg = xm >= 0.0 and ym > 0.0
    if nonneg:
        if p < 0.0:
            return _rj_cpv(xm, ym, zm, p, rerr)
        if p > 0.0:
            maybe_asymp = True
            good = True
    if not good:
        x0 = _too_small(xm)
        y0 = _too_small(ym)
        z0 = _too_small(zm)
        if p > 0.0 and xm >= 0.0 and not ((y0 and (x0 or z0)) or (x0 and z0)):
            good = True
        elif (not _too_small(p)) and _ph_good(p):
            good = nonneg
    if not good:
        return NAN
    if maybe_asymp and not noasymp:
        r = _rj_asym(xm, ym, zm, p, rerr)
        if r[1]:
            return r[0]
    Am = _sum2((xm, ym, zm, p, p)) / 5.0
    delta = (p - xm) * (p - ym) * (p - zm)
    xxm = Am - xm
    yym = Am - ym
    zzm = Am - zm
    fterm = max(max(fabs(xxm), fabs(yym)), max(fabs(zzm), fabs(Am - p))) / _ocrt(rerr / 5.0)
    d4m = 1.0
    pm = p
    # CALC
    prm = sqrt(pm)
    c0 = sqrt(xm)
    c1 = sqrt(ym)
    c2 = sqrt(zm)
    lam = _dot2((c0, c1, c2), (c2, c0, c1))
    dm = (prm + c0) * (prm + c1) * (prm + c2)
    sm = dm * 0.5
    # UPDT
    Am = (Am + lam) * 0.25
    xm = (xm + lam) * 0.25
    ym = (ym + lam) * 0.25
    zm = (zm + lam) * 0.25
    pm = (pm + lam) * 0.25
    xxm *= 0.25
    yym *= 0.25
    zzm *= 0.25
    d4m *= 0.25
    fterm *= 0.25
    m = 1
    while True:
        aAm = fabs(Am)
        if not (aAm <= fterm or aAm <= max(max(fabs(xxm), fabs(yym)), max(fabs(zzm), fabs(Am - pm)))):
            break
        if m > MAX_ITER:
            break
        rm = sm * (sqrt((delta * d4m) / (sm * sm) + 1.0) + 1.0)
        prm = sqrt(pm)
        c0 = sqrt(xm)
        c1 = sqrt(ym)
        c2 = sqrt(zm)
        lam = _dot2((c0, c1, c2), (c2, c0, c1))
        dm = (prm + c0) * (prm + c1) * (prm + c2)
        sm = (rm * dm - delta * (d4m * d4m)) * 0.5 / (dm + rm * d4m)
        Am = (Am + lam) * 0.25
        xm = (xm + lam) * 0.25
        ym = (ym + lam) * 0.25
        zm = (zm + lam) * 0.25
        pm = (pm + lam) * 0.25
        xxm *= 0.25
        yym *= 0.25
        zzm *= 0.25
        d4m *= 0.25
        fterm *= 0.25
        m += 1
    Am = _sum2((xm, ym, zm, pm, pm)) / 5.0
    xxm /= Am
    yym /= Am
    zzm /= Am
    pp = _sum2((xxm, yym, zzm)) * (-0.5)
    pp2 = pp * pp
    xyz = yym * zzm * xxm
    e2 = _dot2((xxm, yym, zzm, pp * (-3.0)), (yym, zzm, xxm, pp))
    e3 = xyz + pp * 2.0 * (e2 + pp2 * 2.0)
    e4 = (2.0 * xyz + (e2 + 3.0 * pp2) * pp) * pp
    e5 = xyz * pp2
    t = sqrt(Am)
    tmp = d4m / (t * t * t)
    c = (_comp_horner(e2, RDJ_C1), _comp_horner(e3, RDJ_C2), _comp_horner(e2, RDJ_C3),
         _comp_horner(e2, RDJ_C4), _comp_horner(e2, RDJ_C5), e3 * RDJ_C5[1])
    t = _dot2(c, (1.0, 1.0, e3, e4, e5, e4)) / RDJ_DENOM + 1.0
    tmp *= t
    t = delta * d4m / (sm * sm)
    tmp += _safe_atan_sqrt_div(t) * 3.0 / sm
    return tmp

def _rj_cpv(x: float, y: float, z: float, p: float, rerr: float) -> float:
    xy = x * y
    q3 = 1.0 - p / z
    pn = (_sum2((x, y, -p)) - xy / z) / q3
    r0 = _rj(x, y, z, pn, rerr, False)
    r1 = _rf(x, y, z, rerr)
    pq = pn * (-p)
    xypq = xy + pq
    r2 = _rc(xypq, pq, rerr)
    tmpres = _dot2((pn - z, -3.0, 3.0 * sqrt(xy * z / xypq)), (r0, r1, r2))
    return tmpres / (z - p)

def _rj_asym(x: float, y: float, z: float, p: float, rerr: float):
    """Returns (value, used)."""
    t = z / p
    if _asymp_zero(t):
        tmp = _rf(x, y, z, rerr * 0.5)
        return (3.0 * (tmp - 0.5 * PI / sqrt(p)) / p, True)
    if _abs_close_zero(p) or (x != 0.0 and _asymp_zero(p / x)):
        f = sqrt(x * y * z)
        r = rerr * 0.5
        a1 = sqrt(z)
        a2 = sqrt(x)
        a3 = sqrt(y)
        lamt = _dot2((a1, a2, a3), (a2, a3, a1))
        alpha = _dot2((a1, a2, a3), (p, p, p)) + f
        alpha = alpha * alpha
        beta = p + lamt
        beta = beta * beta * p
        v0 = _rc(alpha, beta, r)
        v1 = _rj(x + lamt, y + lamt, z + lamt, p + lamt, r, False)
        return (_dot2((3.0, 2.0), (v0, v1)), True)
    t = y / min(z, p)
    if (0.0 < y and y <= 1e-26) or _asymp_zero(t):
        a = 0.5 * (x + y)
        g = sqrt(x * y)
        if (a / z + a / p) * fabs(log(p / a)) <= 1.0:
            tx = _rc(1.0, p / z, rerr)
            tmpres = log(8.0 * z / (a + g)) - 2.0 * tx
            tx = log(2.0 * p / (a + g)) / p
            rt_h = tx * a * (1.0 + 0.5 * p / z) / (1.0 - a / p)
            tx = 1.5 / (sqrt(z) * p)
            tmpres *= tx
            tmpres += rt_h * tx
            return (tmpres, True)
    if x != 0.0 and _asymp_zero(max(z, p) / x):
        r = rerr / 3.0
        tmpres = 1.0 / sqrt(y * z)
        t1 = _rc(x, p, r)
        t2 = _rg(0.0, y, z, r)
        tmpres *= (3.0 * t1 - 2.0 * t2 * tmpres)
        return (tmpres, True)
    if z != 0.0 and _asymp_zero(max(y, p) / z):
        b = 0.5 * (x + y)
        h = sqrt(x * y)
        if fabs(log(z / (b + h))) <= sqrt(z):
            tt = h + p
            tt *= tt
            tm = b + h
            tmpres = _rc(tt, 2.0 * tm * p, rerr * 0.5)
            tt = log(2.0 * z / tm)
            rt_h = 0.75 * (2.0 * p * tmpres / z - tt / (z - h))
            tt = 3.0 / sqrt(z)
            tmpres *= tt
            tmp_direct = _rj(x, y, z, p, rerr * 0.5, True)
            if tmp_direct > tmpres or tmp_direct < tmpres + rt_h * tt:
                tmpres += tt * 0.3 * rt_h
            else:
                tmpres = tmp_direct
            return (tmpres, True)
    return (0.0, False)

def elliprj(x: float, y: float, z: float, p: float) -> float:
    if _isnan(x) or _isnan(y) or _isnan(z) or _isnan(p):
        return NAN
    return _rj(x, y, z, p, RERR, False)
