# Ported from scipy 1.18.1 (BSD-3-Clause)
# Algorithms from Cephes (BSD-3) / xsf (BSD-3) / cdflib (public domain)
#
# Scalar distribution functions (float -> float).

from . import _kernels as K

NAN = float('nan')
INF = float('inf')

def _isnan(x: float) -> bool:
    return x != x

# ---------------------------------------------------------------- Poisson
def pdtrc(k: float, m: float) -> float:
    if k < 0.0 or m < 0.0:
        return NAN
    if m == 0.0:
        return 0.0
    return K.gammainc(K.floor(k) + 1.0, m)

def pdtr(k: float, m: float) -> float:
    if k < 0.0 or m < 0.0:
        return NAN
    if m == 0.0:
        return 1.0
    return K.gammaincc(K.floor(k) + 1.0, m)

def pdtri(k: float, y: float) -> float:
    if _isnan(k) or _isnan(y):
        return NAN
    kk = int(k)
    if kk < 0 or y < 0.0 or y >= 1.0:
        return NAN
    return K.gammainccinv(float(kk + 1), y)

# ---------------------------------------------------------------- binomial
def bdtrc(k: float, n: float, p: float) -> float:
    if _isnan(p) or _isnan(k) or _isnan(n):
        return NAN
    nn = float(int(n))
    fk = K.floor(k)
    if p < 0.0 or p > 1.0 or nn < fk:
        return NAN
    if fk < 0:
        return 1.0
    if fk == nn:
        return 0.0
    dn = nn - fk
    if k == 0:
        if p < .01:
            return -K.expm1(dn * K.log1p(-p))
        return 1.0 - K.pow(1.0 - p, dn)
    return K.betainc(fk + 1.0, dn, p)

def bdtr(k: float, n: float, p: float) -> float:
    if _isnan(p) or _isnan(k) or _isnan(n):
        return NAN
    nn = float(int(n))
    fk = K.floor(k)
    if p < 0.0 or p > 1.0 or fk < 0 or nn < fk:
        return NAN
    if fk == nn:
        return 1.0
    dn = nn - fk
    if fk == 0:
        return K.pow(1.0 - p, dn)
    return K.betainc(dn, fk + 1.0, 1.0 - p)

def bdtri(k: float, n: float, y: float) -> float:
    if _isnan(k) or _isnan(n) or _isnan(y):
        return NAN
    nn = float(int(n))
    fk = K.floor(k)
    if y < 0.0 or y > 1.0 or fk < 0.0 or nn <= fk:
        return NAN
    dn = nn - fk
    if fk == 0:
        if y > 0.8:
            return -K.expm1(K.log1p(y - 1.0) / dn)
        return 1.0 - K.pow(y, 1.0 / dn)
    dk = fk + 1.0
    if y >= 0.5:
        return K.betaincinv(dk, dn, 1.0 - y)
    return 1.0 - K.betaincinv(dn, dk, y)

# ---------------------------------------------------------------- negative binomial
def nbdtrc(k: float, n: float, p: float) -> float:
    if _isnan(k) or _isnan(n) or _isnan(p):
        return NAN
    kk = int(k)
    if p < 0.0 or p > 1.0 or kk < 0:
        return NAN
    return K.betainc(float(kk + 1), float(int(n)), 1.0 - p)

def nbdtr(k: float, n: float, p: float) -> float:
    if _isnan(k) or _isnan(n) or _isnan(p):
        return NAN
    kk = int(k)
    if p < 0.0 or p > 1.0 or kk < 0:
        return NAN
    return K.betainc(float(int(n)), float(kk + 1), p)

def nbdtri(k: float, n: float, p: float) -> float:
    if _isnan(k) or _isnan(n) or _isnan(p):
        return NAN
    kk = int(k)
    if p < 0.0 or p > 1.0 or kk < 0:
        return NAN
    return K.betaincinv(float(int(n)), float(kk + 1), p)

# ---------------------------------------------------------------- gamma
def gdtr(a: float, b: float, x: float) -> float:
    if x < 0.0:
        return NAN
    return K.gammainc(b, a * x)

def gdtrc(a: float, b: float, x: float) -> float:
    if x < 0.0:
        return NAN
    return K.gammaincc(b, a * x)

def gdtri(a: float, b: float, y: float) -> float:
    if y < 0.0 or y > 1.0 or a <= 0.0 or b < 0.0:
        return NAN
    return K.gammainccinv(b, 1.0 - y) / a

# ---------------------------------------------------------------- misc
def _ndtri_exp_small_y(y: float) -> float:
    x = 0.0
    if y >= -K.MAXNUM * 0.5:
        x = K.sqrt(-2.0 * y)
    else:
        x = K.SQRT2 * K.sqrt(-y)
    x0 = x - K.log(x) / x
    z = 1.0 / x
    x1 = 0.0
    if x < 8.0:
        x1 = z * K._polevl(z, K._P1) / K._p1evl(z, K._Q1)
    else:
        x1 = z * K._polevl(z, K._P2) / K._p1evl(z, K._Q2)
    return x1 - x0

def ndtri_exp(y: float) -> float:
    """Inverse of log_ndtr (xsf ndtri_exp)."""
    if y < -K.MAXNUM:
        return -INF
    if y < -2.0:
        return _ndtri_exp_small_y(y)
    if y > -0.14541345786885906:
        return -K.ndtri(-K.expm1(y))
    return K.ndtri(K.exp(y))

# ---------------------------------------------------------------- inverses (xsf cdflib.h)
from . import _tools as T

def gdtria(p: float, b: float, x: float) -> float:
    if x == 0:
        return NAN
    if b == 0 and p == 0:
        if K.isinf(x) and x > 0:
            return NAN
        return 0.0
    return K.gammaincinv(b, p) / x

class _GdtribF:
    a: float
    p: float
    q: float
    x: float
    def __init__(self, a: float, p: float, x: float):
        self.a = a
        self.p = p
        self.q = 1.0 - p
        self.x = x
    def __call__(self, b: float) -> float:
        if self.p <= self.q:
            return K.gammainc(b, self.a * self.x) - self.p
        return self.q - K.gammaincc(b, self.a * self.x)

def gdtrib(a: float, p: float, x: float) -> float:
    if _isnan(p) or _isnan(a) or _isnan(x):
        return NAN
    if not (0 <= p and p <= 1):
        return NAN
    if not (a > 0) or K.isinf(a):
        return NAN
    if not (x >= 0) or K.isinf(x):
        return NAN
    if x == 0.0:
        return NAN if p == 0.0 else 0.0
    if p == 0.0:
        return INF
    if p == 1.0:
        return 0.0
    return T.invert_pos(_GdtribF(a, p, x), False)

def gdtrix(a: float, b: float, p: float) -> float:
    if a == 0 and b == 0:
        return NAN
    if (K.isinf(a) or K.isinf(b)) and (a >= 0 and b >= 0):
        return NAN
    return K.gammaincinv(b, p) / a

class _QinvaF:
    # Q(a, x) - p, decreasing in a... Q(a,x) increases with a
    x: float
    p: float
    def __init__(self, x: float, p: float):
        self.x = x
        self.p = p
    def __call__(self, a: float) -> float:
        if self.p < 0.5:
            return K.gammaincc(a, self.x) - self.p
        return (1.0 - self.p) - K.gammainc(a, self.x)

def _gamma_q_inva(x: float, p: float) -> float:
    # boost::math::gamma_q_inva: a such that Q(a, x) = p
    return T.invert_pos(_QinvaF(x, p), True)

def pdtrik(p: float, n: float) -> float:
    if _isnan(p) or _isnan(n):
        return NAN
    if n < 0 or p < 0 or p > 1:
        return NAN
    if p == 1:
        return NAN
    if p == 0 or n == 0:
        return 0.0
    y = _gamma_q_inva(n, p) - 1.0
    if y < -1:
        return NAN
    if y < 0:
        return 0.0
    return y
