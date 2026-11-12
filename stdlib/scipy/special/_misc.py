# Ported from scipy 1.18.1 (BSD-3-Clause)
# Algorithms from Cephes (BSD-3) / xsf (BSD-3) / scipy Cython (_boxcox.pxd,
# _convex_analysis.pxd, _agm.pxd)
#
# Miscellaneous scalar kernels.

from . import _kernels as K
from ._kernels import exp, log, sqrt, fabs, sin, cos, tan, pow, log1p, expm1, copysign
from . import _tools as T

from C import ldexp(float, i32) -> float as c_ldexp
from C import exp2(float) -> float as c_exp2
from C import cbrt(float) -> float as c_cbrt

NAN = float('nan')
INF = float('inf')
PI = 3.14159265358979323846
PI180 = 1.745329251994329576923690768488612713443E-2
P64800 = 4.848136811095359935899141023579479759563533023727e-6

def _isnan(x: float) -> bool:
    return x != x

# ------------------------------------------------------------------ boxcox family
def boxcox1p(x: float, lmbda: float) -> float:
    lgx = log1p(x)
    if fabs(lmbda) < 1e-19 or (fabs(lgx) < 1e-289 and fabs(lmbda) < 1e273):
        return lgx
    elif lmbda * lgx < 709.78:
        return expm1(lmbda * lgx) / lmbda
    return copysign(1., lmbda) * exp(lmbda * lgx - log(fabs(lmbda))) - 1 / lmbda

def inv_boxcox(x: float, lmbda: float) -> float:
    if lmbda == 0:
        return exp(x)
    elif lmbda * x < 1.79e308:
        return exp(log1p(lmbda * x) / lmbda)
    return exp((log(copysign(1., lmbda) * (x + 1 / lmbda)) + log(fabs(lmbda))) / lmbda)

def inv_boxcox1p(x: float, lmbda: float) -> float:
    if lmbda == 0:
        return expm1(x)
    elif fabs(lmbda * x) < 1e-154:
        return x
    elif lmbda * x < 1.79e308:
        return expm1(log1p(lmbda * x) / lmbda)
    return expm1((log(copysign(1., lmbda) * (x + 1 / lmbda)) + log(fabs(lmbda))) / lmbda)

# ------------------------------------------------------------------ betainccinv
class _BcRoot:
    a: float
    b: float
    y: float
    def __init__(self, a: float, b: float, y: float):
        self.a = a
        self.b = b
        self.y = y
    def __call__(self, x: float) -> float:
        # increasing in x
        return self.y - K.betaincc(self.a, self.b, x)

def betainccinv(a: float, b: float, y: float) -> float:
    """Inverse of the regularized upper incomplete beta (Boost ibetac_inv)."""
    if _isnan(a) or _isnan(b) or _isnan(y):
        return NAN
    if a <= 0 or b <= 0 or y < 0 or y > 1:
        return NAN
    if y == 0:
        return 1.0
    if y == 1:
        return 0.0
    if y >= 0.5:
        return K.betaincinv(a, b, 1.0 - y)
    # small upper tail: x = 1 - I^{-1}_{1-.}(b, a; y) with refinement
    x0 = 1.0 - K.betaincinv(b, a, y)
    f = _BcRoot(a, b, y)
    fx = f(x0)
    if fx == 0:
        return x0
    lo = x0
    hi = x0
    flo = fx
    fhi = fx
    step = max((1.0 - x0) * 1e-6, 1e-300)
    for _ in range(2000):
        if fx < 0:
            hi = min(1.0, hi + step)
            fhi = f(hi)
            if fhi >= 0 or hi == 1.0:
                break
            lo = hi
            flo = fhi
        else:
            lo = max(0.0, lo - step)
            flo = f(lo)
            if flo <= 0 or lo == 0.0:
                break
            hi = lo
            fhi = flo
        step *= 2
    if flo > 0 or fhi < 0:
        return x0
    r = T.chandrupatla(f, lo, hi, flo, fhi, T.EPS, 1e-300, 200)
    return r[0]

# ------------------------------------------------------------------ binom (xsf binom.h)
def binom(n: float, k: float) -> float:
    if n < 0:
        nx = K.floor(n)
        if n == nx:
            return NAN
    kx = K.floor(k)
    if k == kx and (fabs(n) > 1E-8 or n == 0):
        nx = K.floor(n)
        if nx == n and kx > nx / 2 and nx > 0:
            kx = nx - kx
        if kx >= 0 and kx < 20:
            num = 1.0
            den = 1.0
            for i in range(1, 1 + int(kx)):
                num *= i + n - kx
                den *= i
                if fabs(num) > 1E50:
                    num /= den
                    den = 1.0
            return num / den
    if n >= 1E10 * k and k > 0:
        return exp(-K.betaln(1 + n - k, 1 + k) - log(n + 1))
    if k > 1E8 * fabs(n):
        num = K.gamma(1 + n) / fabs(k) + K.gamma(1 + n) * n / (2 * k * k)
        num /= PI * pow(fabs(k), n)
        if k > 0:
            kx = K.floor(k)
            dk = 0.0
            sgn = 1.0
            if fabs(kx) < 2147483647.0:
                dk = k - kx
                sgn = 1.0 if int(kx) % 2 == 0 else -1.0
            else:
                dk = k
            return num * sin((dk - n) * PI) * sgn
        kx = K.floor(k)
        if fabs(kx) < 2147483647.0:
            return 0.0
        return num * sin(k * PI)
    return 1 / (n + 1) / K.beta(1 + n - k, 1 + k)

# ------------------------------------------------------------------ cephes trig in degrees
_SINCOF = (1.58962301572218447952E-10, -2.50507477628503540135E-8, 2.75573136213856773549E-6,
           -1.98412698295895384658E-4, 8.33333333332211858862E-3, -1.66666666666666307295E-1)
_COSCOF = (1.13678171382044553091E-11, -2.08758833757683644217E-9, 2.75573155429816611547E-7,
           -2.48015872936186303776E-5, 1.38888888888806666760E-3, -4.16666666666666348141E-2,
           4.99999999999999999798E-1)

def _polevl(x: float, c) -> float:
    r = 0.0
    for v in c:
        r = r * x + v
    return r

def sindg(x: float) -> float:
    sign = 1
    if K._signbit(x):
        x = -x
        sign = -1
    if x > 1.0e14:
        return 0.0
    y = K.floor(x / 45.0)
    z = c_ldexp(y, i32(-4))
    z = K.floor(z)
    z = y - c_ldexp(z, i32(4))
    j = int(z)
    if j & 1:
        j += 1
        y += 1.0
    j = j & 7
    if j > 3:
        sign = -sign
        j -= 4
    z = x - y * 45.0
    z *= PI180
    zz = z * z
    if j == 1 or j == 2:
        y = 1.0 - zz * _polevl(zz, _COSCOF)
    else:
        y = z + z * (zz * _polevl(zz, _SINCOF))
    if sign < 0:
        y = -y
    return y

def cosdg(x: float) -> float:
    sign = 1
    if x < 0:
        x = -x
    if x > 1.0e14:
        return 0.0
    y = K.floor(x / 45.0)
    z = c_ldexp(y, i32(-4))
    z = K.floor(z)
    z = y - c_ldexp(z, i32(4))
    j = int(z)
    if j & 1:
        j += 1
        y += 1.0
    j = j & 7
    if j > 3:
        j -= 4
        sign = -sign
    if j > 1:
        sign = -sign
    z = x - y * 45.0
    z *= PI180
    zz = z * z
    if j == 1 or j == 2:
        y = z + z * (zz * _polevl(zz, _SINCOF))
    else:
        y = 1.0 - zz * _polevl(zz, _COSCOF)
    if sign < 0:
        y = -y
    return y

def _tancot(xx: float, cotflg: bool) -> float:
    x = xx
    sign = 1.0
    if K._signbit(xx):
        x = -xx
        sign = -1.0
    if x > 1.0e14:
        return 0.0
    x = x - 180.0 * K.floor(x / 180.0)
    if cotflg:
        if x <= 90.0:
            x = 90.0 - x
        else:
            x = x - 90.0
            sign *= -1
    else:
        if x > 90.0:
            x = 180.0 - x
            sign *= -1
    if x == 0.0:
        return sign * 0.0
    elif x == 45.0:
        return sign * 1.0
    elif x == 90.0:
        return INF
    return sign * tan(x * PI180)

def tandg(x: float) -> float:
    return _tancot(x, False)

def cotdg(x: float) -> float:
    return _tancot(x, True)

def radian(d: float, m: float, s: float) -> float:
    return ((d * 60.0 + m) * 60.0 + s) * P64800

_UNITY_COSCOF = (4.7377507964246204691685E-14, -1.1470284843425359765671E-11, 2.0876754287081521758361E-9,
                 -2.7557319214999787979814E-7, 2.4801587301570552304991E-5, -1.3888888888888872993737E-3,
                 4.1666666666666666609054E-2)

def cosm1(x: float) -> float:
    if x < -0.78539816339744830962 or x > 0.78539816339744830962:
        return cos(x) - 1.0
    xx = x * x
    return -0.5 * xx + xx * xx * _polevl(xx, _UNITY_COSCOF)

def cround(x: float) -> float:
    y = K.floor(x)
    r = x - y
    if r > 0.5:
        return y + 1.0
    if r == 0.5:
        r = y - 2.0 * K.floor(0.5 * y)
        if r == 1.0:
            y += 1.0
    return y

def exp2(x: float) -> float:
    return c_exp2(x)

def exp10(x: float) -> float:
    return pow(10.0, x)

def cbrt(x: float) -> float:
    return c_cbrt(x)

# ------------------------------------------------------------------ convex analysis
def huber(delta: float, r: float) -> float:
    if delta < 0:
        return INF
    elif fabs(r) <= delta:
        return 0.5 * r * r
    return delta * (fabs(r) - 0.5 * delta)

def pseudo_huber(delta: float, r: float) -> float:
    if delta < 0:
        return INF
    elif delta == 0 or r == 0:
        return 0.0
    u = delta
    v = r / delta
    return u * u * expm1(0.5 * log1p(v * v))
