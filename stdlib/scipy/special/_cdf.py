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

# ------------------------------------------------------------------ parameter inversions
# Boost ibeta_inva/ibeta_invb/gamma_p_inva and the binomial/negative-binomial
# real-valued quantiles (scipy SpecialPolicy: discrete_quantile<real>), and
# cdflib's df searches, solved here with xsf bracketing + Chandrupatla.

class _ParamRoot:
    kind: int
    p1: float
    p2: float
    target: float
    def __init__(self, kind: int, p1: float, p2: float, target: float):
        self.kind = kind
        self.p1 = p1
        self.p2 = p2
        self.target = target
    def __call__(self, t: float) -> float:
        k = self.kind
        q = self.target
        if k == 0:   # btdtria: I_x(a, b) = p in a   (decreasing in a)
            return K.betainc(t, self.p1, self.p2) - q
        elif k == 1:  # btdtrib: I_x(a, b) = p in b  (increasing in b)
            return K.betainc(self.p1, t, self.p2) - q
        elif k == 2:  # chdtriv: P(v/2, x/2) = p in v (decreasing)
            return K.gammainc(t / 2, self.p1 / 2) - q
        elif k == 3:  # bdtrik: ibetac(k+1, n-k, p) = y in k (increasing)
            if t >= self.p1:
                return 1.0 - q
            return K.betaincc(t + 1, self.p1 - t, self.p2) - q
        elif k == 4:  # nbdtrik: ibeta(n, k+1, p) = y in k (increasing)
            return K.betainc(self.p1, t + 1, self.p2) - q
        elif k == 5:  # nbdtrin: ibeta(n, k+1, p) = y in n (increasing)
            return K.betainc(t, self.p1 + 1, self.p2) - q
        elif k == 6:  # bdtrin: ibetac(k+1, n-k, p) = y in n (decreasing)
            return K.betaincc(self.p1 + 1, t - self.p1, self.p2) - q
        elif k == 7:  # stdtridf: stdtr(df, t) = p in df
            return K.stdtr(t, self.p1) - q
        else:         # fdtridfd: fdtr(dfn, dfd, f) = p in dfd
            return K.fdtr(self.p1, t, self.p2) - q

def _root_in(f: _ParamRoot, lo: float, hi: float, increasing: bool) -> float:
    # bracket by power-of-two steps from 1 within [lo, hi]
    r = T.bracket_root(f, 1.0 if (lo < 1.0 and hi > 1.0) else 0.5 * (lo + hi), lo, hi,
                       -0.875, 7.0, 0.125, 8.0, increasing, 342)
    st = r[4]
    if st == 1:
        return lo
    if st == 2:
        return hi
    if st >= 3:
        return NAN
    res = T.chandrupatla(f, r[0], r[1], r[2], r[3], T.EPS, 1e-300, 200)
    return res[0] if res[1] == 0 else NAN

def btdtria(p: float, b: float, x: float) -> float:
    if _isnan(p) or _isnan(b) or _isnan(x):
        return NAN
    if b <= 0 or x <= 0 or p < 0 or p > 1:
        return NAN
    if x >= 1:
        return NAN if p != 1 else 0.0
    if p == 0:
        return INF
    if p == 1:
        return 0.0
    return _root_in(_ParamRoot(0, b, x, p), T.DBL_MIN, T.DBL_MAX, False)

def btdtrib(a: float, p: float, x: float) -> float:
    if _isnan(p) or _isnan(a) or _isnan(x):
        return NAN
    if a <= 0 or x <= 0 or p < 0 or p > 1:
        return NAN
    if x >= 1:
        return NAN if p != 0 else 0.0
    if p == 0:
        return 0.0
    if p == 1:
        return INF
    return _root_in(_ParamRoot(1, a, x, p), T.DBL_MIN, T.DBL_MAX, True)

def chdtriv(p: float, x: float) -> float:
    if _isnan(p) or _isnan(x):
        return NAN
    if x <= 0 or p < 0 or p > 1:
        return NAN
    if p == 1:
        return 0.0
    if p == 0:
        return INF
    return _root_in(_ParamRoot(2, x, 0.0, p), T.DBL_MIN, T.DBL_MAX, False)

def _solve_box(f: _ParamRoot, lo: float, hi: float) -> float:
    # root of an increasing f on [lo, hi] (clamped to the ends)
    flo = f(lo)
    if flo >= 0:
        return lo
    fhi = f(hi)
    if fhi <= 0:
        return hi
    res = T.chandrupatla(f, lo, hi, flo, fhi, T.EPS, 1e-300, 400)
    return res[0]

def bdtrik(y: float, n: float, p: float) -> float:
    if _isnan(y) or _isnan(n) or _isnan(p):
        return NAN
    if n < 0 or p < 0 or p > 1 or y < 0 or y > 1:
        return NAN
    if p == 0 or y == 0:
        return 0.0
    if p == 1 or y == 1:
        return n
    return _solve_box(_ParamRoot(3, n, p, y), 0.0, n)

def nbdtrik(y: float, n: float, p: float) -> float:
    if _isnan(y) or _isnan(n) or _isnan(p):
        return NAN
    if n < 0 or p < 0 or p > 1 or y < 0 or y > 1:
        return NAN
    if y == 0 or p == 1:
        return 0.0
    if y == 1 or p == 0:
        return INF
    f0 = _ParamRoot(4, n, p, y)
    if f0(0.0) >= 0:
        return 0.0
    hi = 1.0
    while f0(hi) < 0:
        hi *= 2
        if hi > 1e300:
            return INF
    return _solve_box(f0, 0.0, hi)

def nbdtrin(k: float, y: float, p: float) -> float:
    if _isnan(k) or _isnan(y) or _isnan(p):
        return NAN
    if k < 0 or y < 0 or y > 1 or p < 0 or p > 1:
        return NAN
    return _root_in(_ParamRoot(5, k, p, y), T.DBL_MIN, T.DBL_MAX, False)

def bdtrin(k: float, y: float, p: float) -> float:
    if _isnan(k) or _isnan(y) or _isnan(p):
        return NAN
    if k < 0 or y < 0 or y > 1 or p < 0 or p > 1:
        return NAN
    f0 = _ParamRoot(6, k, p, y)
    r = T.bracket_root(f0, k + 1.0, k, T.DBL_MAX, -0.875 * (k + 1.0) / (k + 1.0), 7.0, 0.125, 8.0, False, 342)
    if r[4] != 0:
        return k if r[4] == 1 else INF
    res = T.chandrupatla(f0, r[0], r[1], r[2], r[3], T.EPS, 1e-300, 200)
    return res[0]

def stdtridf(p: float, t: float) -> float:
    if _isnan(p) or _isnan(t):
        return NAN
    if p <= 0 or p >= 1:
        return NAN
    inc = t > 0
    return _root_in(_ParamRoot(7, t, 0.0, p), 1e-100, 1e10, inc)

def fdtridfd(dfn: float, p: float, f: float) -> float:
    if _isnan(dfn) or _isnan(p) or _isnan(f):
        return NAN
    if p <= 0 or p >= 1 or f <= 0 or dfn <= 0:
        return NAN
    # fdtr is increasing in dfd when f > 1-ish; detect direction numerically
    g = _ParamRoot(8, dfn, f, p)
    inc = g(2.0) > g(1.0)
    return _root_in(g, 1e-100, 1e10, inc)
