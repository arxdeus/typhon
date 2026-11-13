# Copyright (C) 2026 the Typhon authors
# Ported from scipy 1.18.1 (BSD-3-Clause)

# Continuous distribution kernels, part 5 (scipy.stats._continuous_distns):
# kstwo, geninvgauss, genhyperbolic, jf_skew_t, landau, vonmises_line,
# dpareto_lognorm. See _distn_infrastructure for the kernel protocol.

import numpy as np
import scipy.special._kernels as K
import scipy.special._bessel as B
import scipy.special._kolmogorov as KS
from scipy.integrate import quad
from scipy.optimize import brentq
from ._distn_infrastructure import rv_continuous, NAN, INF, _NA, _map_ppf
from ._continuous_distns import _norm_pdf, _norm_logpdf, _norm_cdf, _norm_logcdf, \
    _norm_logsf, _norm_sf
from ._continuous_distns2 import _VonmisesK
from ._morestats import _kolmogn, _log_nfactorial_div_n_pow_n
from ._landau import landau_pdf, landau_cdf, landau_sf, landau_ppf, landau_isf

from C import fabs(float) -> float
from C import cos(float) -> float
from C import acos(float) -> float
from C import tan(float) -> float
from C import cosh(float) -> float
from C import sinh(float) -> float
from C import log(float) -> float
from C import exp(float) -> float

_PI = 3.141592653589793

def _nsamples(size) -> int:
    if size is None:
        return 1
    elif isinstance(size, int):
        return size
    else:
        n = 1
        for s in size:
            n *= int(s)
        return n

def _shape_out(x: np.ndarray[float, 1], size):
    if size is None:
        return x[0]
    elif isinstance(size, int):
        return x
    else:
        return x.reshape(size)

# ------------------------------------------------------------------- kstwo

def _kolmogn_p(n: int, x: float) -> float:
    if n <= 0:
        return NAN
    if x >= 1.0 or x <= 0:
        return 0.0
    t = n * x
    if t <= 1.0:
        if t <= 0.5:
            return 0.0
        prd = 0.0
        if n <= 140:
            prd = 1.0
            for i in range(1, n):
                prd *= i * (1.0 / n) * (2 * t - 1)
        else:
            prd = K.exp(_log_nfactorial_div_n_pow_n(float(n)) + (n - 1) * K.log(2 * t - 1))
        return prd * 2 * n ** 2
    if t >= n - 1:
        return 2 * (1.0 - x) ** (n - 1) * n
    if x >= 0.5:
        return -2 * KS.smirnovp(float(n), x)
    # _derivative(_kk, x, dx=delta, order=5)
    delta = x / 2.0 ** 16
    delta = min(delta, x - 1.0 / n)
    delta = min(delta, 0.5 - x)
    f1 = _kolmogn(n, x - 2 * delta, True)
    f2 = _kolmogn(n, x - delta, True)
    f3 = _kolmogn(n, x + delta, True)
    f4 = _kolmogn(n, x + 2 * delta, True)
    return (f1 - 8.0 * f2 + 8.0 * f3 - f4) / (12.0 * delta)

class _KolmogniF:
    n: int
    p: float
    def __init__(self, n: int, p: float):
        self.n = n
        self.p = p
    def __call__(self, x: float) -> float:
        return _kolmogn(self.n, x, True) - self.p

def _kolmogni(n: int, p: float, q: float) -> float:
    if n <= 0:
        return NAN
    if p <= 0:
        return 1.0 / n
    if q <= 0:
        return 1.0
    delta = K.exp((K.log(p) - K.gammaln(n + 1.0)) / n)
    if delta <= 1.0 / n:
        return (delta + 1.0 / n) / 2
    x = -K.expm1(K.log(q / 2.0) / n)
    if x >= 1 - 1.0 / n:
        return x
    x1 = KS.kolmogci(p) / K.sqrt(float(n))
    x1 = min(x1, 1.0 - 1.0 / n)
    return brentq(_KolmogniF(n, p), 1.0 / n, x1, xtol=1e-14)

@tuple
class _KstwoK:
    @staticmethod
    def nshape() -> Literal[int]: return 1
    @staticmethod
    def check(sh) -> bool: return sh[0] >= 1 and sh[0] == K.floor(sh[0] + 0.5)
    @staticmethod
    def a(sh) -> float: return 0.5 / sh[0]
    @staticmethod
    def b(sh) -> float: return 1.0
    @staticmethod
    def momtype0() -> Literal[bool]: return True
    @staticmethod
    def pdf(x: float, sh) -> float: return _kolmogn_p(int(sh[0]), x)
    @staticmethod
    def cdf(x: float, sh) -> float: return _kolmogn(int(sh[0]), x, True)
    @staticmethod
    def sf(x: float, sh) -> float: return _kolmogn(int(sh[0]), x, False)
    @staticmethod
    def ppf(q: float, sh) -> float: return _kolmogni(int(sh[0]), q, 1 - q)
    @staticmethod
    def isf(q: float, sh) -> float: return _kolmogni(int(sh[0]), 1 - q, q)

# ------------------------------------------------------------- geninvgauss

def _gig_logpdf(x: float, p: float, b: float) -> float:
    if x <= 0:
        return -INF
    z = B.kve(p, b)
    if K.isinf(z):
        return NAN
    c = -K.log(2.0) - K.log(z) + b
    return c + (p - 1) * K.log(x) - b * (x + 1 / x) / 2

def _gig_logquasipdf(x: float, p: float, b: float) -> float:
    if x > 0:
        return (p - 1) * K.log(x) - b * (x + 1 / x) / 2
    return -INF

def _gig_mode(p: float, b: float) -> float:
    if p < 1:
        return b / (K.sqrt((p - 1) ** 2 + b ** 2) + 1 - p)
    return (K.sqrt((1 - p) ** 2 + b ** 2) - (1 - p)) / b

class _GigPdf:
    p: float
    b: float
    def __init__(self, p: float, b: float):
        self.p = p
        self.b = b
    def __call__(self, x: float) -> float:
        return K.exp(_gig_logpdf(x, self.p, self.b))

def _gig_rvs_scalar(rs, p0: float, b: float, N: int) -> np.ndarray[float, 1]:
    p = p0
    invert = False
    if p < 0:
        p = -p
        invert = True
    m = _gig_mode(p, b)
    ratio_unif = True
    mode_shift = True
    if p >= 1 or b > 1:
        mode_shift = True
    elif b >= min(0.5, 2 * K.sqrt(1 - p) / 3):
        mode_shift = False
    else:
        ratio_unif = False
    x = np.zeros(N)
    simulated = 0
    if ratio_unif:
        umax = 1.0
        vmin = 0.0
        vmax = 0.0
        c = 0.0
        lm = 0.0
        if mode_shift:
            a2 = -2 * (p + 1) / b - m
            a1 = 2 * m * (p - 1) / b - 1
            p1 = a1 - a2 ** 2 / 3
            q1 = 2 * a2 ** 3 / 27 - a2 * a1 / 3 + m
            phi = acos(-q1 * K.sqrt(-27 / p1 ** 3) / 2)
            s1 = -K.sqrt(-4 * p1 / 3)
            root1 = s1 * cos(phi / 3 + _PI / 3) - a2 / 3
            root2 = -s1 * cos(phi / 3) - a2 / 3
            lm = _gig_logquasipdf(m, p, b)
            d1 = _gig_logquasipdf(root1, p, b) - lm
            d2 = _gig_logquasipdf(root2, p, b) - lm
            vmin = (root1 - m) * K.exp(0.5 * d1)
            vmax = (root2 - m) * K.exp(0.5 * d2)
            umax = 1.0
            c = m
        else:
            umax = K.exp(0.5 * _gig_logquasipdf(m, p, b))
            xplus = ((1 + p) + K.sqrt((1 + p) ** 2 + b ** 2)) / b
            vmin = 0.0
            vmax = xplus * K.exp(0.5 * _gig_logquasipdf(xplus, p, b))
            c = 0.0
            lm = 0.0
        if vmin >= vmax:
            raise ValueError("vmin must be smaller than vmax.")
        if umax <= 0:
            raise ValueError("umax must be positive.")
        i = 1
        while simulated < N:
            k = N - simulated
            u = umax * rs.uniform(size=k)
            v = rs.uniform(size=k)
            v = vmin + (vmax - vmin) * v
            for j in range(k):
                r = v[j] / u[j] + c
                if 2 * K.log(u[j]) <= _gig_logquasipdf(r, p, b) - lm:
                    x[simulated] = r
                    simulated += 1
            if simulated == 0 and i * N >= 50000:
                raise RuntimeError("Not a single random variate could be generated in "
                                   f"{i * N} attempts. Sampling does not appear to "
                                   "work for the provided parameters.")
            i += 1
    else:
        x0 = b / (1 - p)
        xs = max(x0, 2 / b)
        k1 = K.exp(_gig_logquasipdf(m, p, b))
        A1 = k1 * x0
        k2 = 0.0
        A2 = 0.0
        if x0 < 2 / b:
            k2 = K.exp(-b)
            if p > 0:
                A2 = k2 * ((2 / b) ** p - x0 ** p) / p
            else:
                A2 = k2 * K.log(2 / b ** 2)
        k3 = xs ** (p - 1)
        A3 = 2 * k3 * K.exp(-xs * b / 2) / b
        A = A1 + A2 + A3
        while simulated < N:
            k = N - simulated
            u = rs.uniform(size=k)
            v = A * rs.uniform(size=k)
            for j in range(k):
                vj = v[j]
                h = 0.0
                r = 0.0
                if vj <= A1:
                    r = x0 * vj / A1
                    h = k1
                elif vj <= A1 + A2:
                    if p > 0:
                        r = (x0 ** p + (vj - A1) * p / k2) ** (1 / p)
                    else:
                        r = b * K.exp((vj - A1) * K.exp(b))
                    h = k2 * r ** (p - 1)
                else:
                    z = K.exp(-xs * b / 2) - b * (vj - A1 - A2) / (2 * k3)
                    r = -2 / b * K.log(z)
                    h = k3 * K.exp(-r * b / 2)
                if K.log(u[j] * h) <= _gig_logquasipdf(r, p, b):
                    x[simulated] = r
                    simulated += 1
    if invert:
        for i in range(N):
            x[i] = 1 / x[i]
    return x

@tuple
class _GeninvgaussK:
    @staticmethod
    def nshape() -> Literal[int]: return 2
    @staticmethod
    def check(sh) -> bool: return sh[0] == sh[0] and sh[1] > 0
    @staticmethod
    def a(sh) -> float: return 0.0
    @staticmethod
    def rvs(rs, sh, size):
        n = _nsamples(size)
        return _shape_out(_gig_rvs_scalar(rs, sh[0], sh[1], n), size)
    @staticmethod
    def logpdf(x: float, sh) -> float: return _gig_logpdf(x, sh[0], sh[1])
    @staticmethod
    def pdf(x: float, sh) -> float: return K.exp(_gig_logpdf(x, sh[0], sh[1]))
    @staticmethod
    def cdf(x: float, sh) -> float:
        return quad(_GigPdf(sh[0], sh[1]), 0.0, x)[0]
    @staticmethod
    def munp(n: float, sh) -> float:
        num = B.kve(sh[0] + n, sh[1])
        denom = B.kve(sh[0], sh[1])
        if K.isinf(num) or K.isinf(denom):
            return NAN
        return num / denom

# ----------------------------------------------------------- genhyperbolic

def _gh_log_norming(p: float, a: float, b: float) -> float:
    t1 = (a + b) * (a - b)
    t2 = p * 0.5 * K.log(t1)
    t3 = 0.5 * K.log(2 * _PI)
    t4 = (p - 0.5) * K.log(a)
    t5 = K.sqrt(t1)
    t6 = K.log(B.kve(p, t5)) - t5
    return t2 - t3 - t4 - t6

def _gh_logpdf(x: float, p: float, a: float, b: float) -> float:
    t1 = _gh_log_norming(p, a, b)
    t2 = K.sqrt(1.0 + x * x)
    t3 = (p - 0.5) * K.log(t2)
    t4 = K.log(B.kve(p - 0.5, a * t2)) - a * t2
    t5 = b * x
    return t1 + t3 + t4 + t5

class _GhPdf:
    p: float
    a: float
    b: float
    def __init__(self, p: float, a: float, b: float):
        self.p = p
        self.a = a
        self.b = b
    def __call__(self, x: float) -> float:
        return K.exp(_gh_logpdf(x, self.p, self.a, self.b))

def _gh_integrate(x0: float, x1: float, p: float, a: float, b: float) -> float:
    f = _GhPdf(p, a, b)
    d = K.sqrt((a + b) * (a - b))
    mean = b / d * B.kv(p + 1, d) / B.kv(p, d)
    epsrel = 1e-10
    epsabs = 0.0
    intgrl = 0.0
    if x0 < mean and mean < x1:
        intgrl = (quad(f, x0, mean, epsrel=epsrel, epsabs=epsabs)[0]
                  + quad(f, mean, x1, epsrel=epsrel, epsabs=epsabs)[0])
    else:
        intgrl = quad(f, x0, x1, epsrel=epsrel, epsabs=epsabs)[0]
    return max(0.0, min(1.0, intgrl))

@tuple
class _GenhyperbolicK:
    @staticmethod
    def nshape() -> Literal[int]: return 3
    @staticmethod
    def check(sh) -> bool:
        p = sh[0]
        a = sh[1]
        b = sh[2]
        return (fabs(b) < a and p >= 0) or (fabs(b) <= a and p < 0)
    @staticmethod
    def fitstart(data: np.ndarray[float, 1]) -> List[float]: return [1.0, 1.0, 0.5]
    @staticmethod
    def logpdf(x: float, sh) -> float: return _gh_logpdf(x, sh[0], sh[1], sh[2])
    @staticmethod
    def pdf(x: float, sh) -> float: return K.exp(_gh_logpdf(x, sh[0], sh[1], sh[2]))
    @staticmethod
    def cdf(x: float, sh) -> float: return _gh_integrate(-INF, x, sh[0], sh[1], sh[2])
    @staticmethod
    def sf(x: float, sh) -> float: return _gh_integrate(x, INF, sh[0], sh[1], sh[2])
    @staticmethod
    def rvs(rs, sh, size):
        p = sh[0]
        a = sh[1]
        b = sh[2]
        t1 = a ** 2 - b ** 2
        t2 = t1 ** 0.5
        t3 = t1 ** -0.5
        n = _nsamples(size)
        gig = _gig_rvs_scalar(rs, p, t2, n) * t3
        normst = rs.standard_normal(size=n)
        out = b * gig + np.sqrt(gig) * normst
        return _shape_out(out, size)
    @staticmethod
    def stats(sh):
        p = sh[0]
        a = sh[1]
        b = sh[2]
        t1 = (a ** 2 - b ** 2) ** 0.5
        t2 = 1.0 / t1
        b0 = B.kv(p, t1)
        b1 = B.kv(p + 1, t1)
        b2 = B.kv(p + 2, t1)
        b3 = B.kv(p + 3, t1)
        b4 = B.kv(p + 4, t1)
        r1 = b1 / b0
        r2 = b2 / b0
        r3 = b3 / b0
        r4 = b4 / b0
        m = b * t2 * r1
        v = t2 * r1 + b ** 2 * t2 ** 2 * (r2 - r1 ** 2)
        m3e = (b ** 3 * t2 ** 3 * (r3 - 3 * b2 * b1 * b0 ** -2 + 2 * r1 ** 3)
               + 3 * b * t2 ** 2 * (r2 - r1 ** 2))
        s = m3e * v ** (-3 / 2)
        m4e = (b ** 4 * t2 ** 4 * (r4 - 4 * b3 * b1 * b0 ** -2
                                   + 6 * b2 * b1 ** 2 * b0 ** -3 - 3 * r1 ** 4)
               + b ** 2 * t2 ** 3 * (6 * r3 - 12 * b2 * b1 * b0 ** -2 + 6 * r1 ** 3)
               + 3 * t2 ** 2 * r2)
        k = m4e * v ** -2 - 3
        return (m, v, s, k)

# --------------------------------------------------------------- jf_skew_t

def _jf_nth_moment(n: int, a: float, b: float) -> float:
    num = (a + b) ** (0.5 * n)
    denom = 2.0 ** n * K.beta(a, b)
    s = 0.0
    for i in range(n + 1):
        sgn = -1.0 if i % 2 > 0 else 1.0
        d = K.beta(a + 0.5 * n - i, b - 0.5 * n + i)
        c = 1.0
        for j in range(i):
            c = c * (n - j) / (j + 1)
        s += c * sgn * d
    return num / denom * s

@tuple
class _JfSkewTK:
    @staticmethod
    def nshape() -> Literal[int]: return 2
    @staticmethod
    def rvs(rs, sh, size):
        a = sh[0]
        b = sh[1]
        d1 = rs.beta(a, b, size)
        d2 = (2 * d1 - 1) * K.sqrt(a + b)
        d3 = 2 * np.sqrt(d1 * (1 - d1))
        return d2 / d3
    @staticmethod
    def pdf(x: float, sh) -> float:
        a = sh[0]
        b = sh[1]
        c = 2 ** (a + b - 1) * K.beta(a, b) * K.sqrt(a + b)
        d1 = (1 + x / K.sqrt(a + b + x ** 2)) ** (a + 0.5)
        d2 = (1 - x / K.sqrt(a + b + x ** 2)) ** (b + 0.5)
        return d1 * d2 / c
    @staticmethod
    def cdf(x: float, sh) -> float:
        y = (1 + x / K.sqrt(sh[0] + sh[1] + x ** 2)) * 0.5
        return K.betainc(sh[0], sh[1], y)
    @staticmethod
    def sf(x: float, sh) -> float:
        y = (1 + x / K.sqrt(sh[0] + sh[1] + x ** 2)) * 0.5
        return K.betaincc(sh[0], sh[1], y)
    @staticmethod
    def ppf(q: float, sh) -> float:
        a = sh[0]
        b = sh[1]
        d1 = K.betaincinv(a, b, q)
        d2 = (2 * d1 - 1) * K.sqrt(a + b)
        d3 = 2 * K.sqrt(d1 * (1 - d1))
        return d2 / d3
    @staticmethod
    def munp(n: float, sh) -> float:
        a = sh[0]
        b = sh[1]
        if a > 0.5 * n and b > 0.5 * n and n >= 0:
            return _jf_nth_moment(int(n), a, b)
        return NAN

# ------------------------------------------------------------------ landau

@tuple
class _LandauK:
    @staticmethod
    def nshape() -> Literal[int]: return 0
    @staticmethod
    def entropy(sh) -> float: return 2.37263644000448182
    @staticmethod
    def rvs(rs, sh, size):
        pi_2 = _PI / 2
        U = rs.uniform(-_PI / 2, _PI / 2, size=size)
        W = rs.standard_exponential(size=size)
        return 2 / _PI * ((pi_2 + U) * np.tan(U) - np.log((pi_2 * W * np.cos(U)) / (pi_2 + U)))
    @staticmethod
    def pdf(x: float, sh) -> float: return landau_pdf(x)
    @staticmethod
    def cdf(x: float, sh) -> float: return landau_cdf(x)
    @staticmethod
    def sf(x: float, sh) -> float: return landau_sf(x)
    @staticmethod
    def ppf(q: float, sh) -> float: return landau_ppf(q)
    @staticmethod
    def isf(q: float, sh) -> float: return landau_isf(q)
    @staticmethod
    def stats(sh): return (NAN, NAN, NAN, NAN)
    @staticmethod
    def munp(n: float, sh) -> float: return NAN if n > 0 else 1.0

# ------------------------------------------------------------ vonmises_line

@tuple
class _VonmisesLineK:
    @staticmethod
    def nshape() -> Literal[int]: return 1
    @staticmethod
    def check(sh) -> bool: return sh[0] >= 0
    @staticmethod
    def a(sh) -> float: return -_PI
    @staticmethod
    def b(sh) -> float: return _PI
    @staticmethod
    def rvs(rs, sh, size): return rs.vonmises(0.0, sh[0], size=size)
    @staticmethod
    def pdf(x: float, sh) -> float: return _VonmisesK.pdf(x, sh)
    @staticmethod
    def logpdf(x: float, sh) -> float: return _VonmisesK.logpdf(x, sh)
    @staticmethod
    def cdf(x: float, sh) -> float: return _VonmisesK.cdf(x, sh)
    @staticmethod
    def entropy(sh) -> float: return _VonmisesK.entropy(sh)

# ---------------------------------------------------------- dpareto_lognorm

def _dp_logR(z: float) -> float:
    return _norm_logsf(z) - _norm_logpdf(z)

def _logaddexp(a: float, b: float) -> float:
    if a == -INF and b == -INF:
        return -INF
    m = max(a, b)
    return m + K.log1p(K.exp(-fabs(a - b))) if not K.isinf(m) else m

def _dp_logpdf(x: float, u: float, s: float, a: float, b: float) -> float:
    if x == 0 or K.isinf(x):
        return -INF
    log_y = K.log(x)
    z = (log_y - u) / s
    x1 = a * s - z
    x2 = b * s + z
    out = K.log(a) + K.log(b) - K.log(a + b) - log_y
    out += _norm_logpdf(z)
    out += _logaddexp(_dp_logR(x1), _dp_logR(x2))
    return out

def _dp_logcdf(x: float, u: float, s: float, a: float, b: float) -> float:
    if x == 0:
        return -INF
    log_y = K.log(x)
    z = (log_y - u) / s
    x1 = a * s - z
    x2 = b * s + z
    t1 = _norm_logcdf(z)
    t2 = _norm_logpdf(z)
    t3 = K.log(b) + _dp_logR(x1)
    t4 = K.log(a) + _dp_logR(x2)
    # logsumexp([t3, t4], b=[1, -1], return_sign=True)
    m = max(t3, t4)
    d = K.exp(t3 - m) - K.exp(t4 - m)
    sign = 1.0 if d >= 0 else -1.0
    t5 = m + K.log(fabs(d))
    # logsumexp([t1, t2 + t5 - log(a + b)], b=[1, -sign])
    u2 = t2 + t5 - K.log(a + b)
    mm = max(t1, u2)
    return mm + K.log(K.exp(t1 - mm) - sign * K.exp(u2 - mm))

def _log1mexp(x: float) -> float:
    if x > -0.6931471805599453:
        return K.log(-K.expm1(x))
    return K.log1p(-K.exp(x))

@tuple
class _DparetoLognormK:
    @staticmethod
    def nshape() -> Literal[int]: return 4
    @staticmethod
    def check(sh) -> bool: return sh[1] > 0 and sh[2] > 0 and sh[3] > 0
    @staticmethod
    def a(sh) -> float: return 0.0
    @staticmethod
    def rvs(rs, sh, size):
        u = sh[0]
        s = sh[1]
        a = sh[2]
        b = sh[3]
        Z = rs.normal(u, s, size=size)
        E1 = rs.standard_exponential(size=size)
        E2 = rs.standard_exponential(size=size)
        return np.exp(Z + E1 / a - E2 / b)
    @staticmethod
    def logpdf(x: float, sh) -> float: return _dp_logpdf(x, sh[0], sh[1], sh[2], sh[3])
    @staticmethod
    def pdf(x: float, sh) -> float: return K.exp(_dp_logpdf(x, sh[0], sh[1], sh[2], sh[3]))
    @staticmethod
    def logcdf(x: float, sh) -> float: return _dp_logcdf(x, sh[0], sh[1], sh[2], sh[3])
    @staticmethod
    def cdf(x: float, sh) -> float: return K.exp(_dp_logcdf(x, sh[0], sh[1], sh[2], sh[3]))
    @staticmethod
    def logsf(x: float, sh) -> float: return _log1mexp(_dp_logcdf(x, sh[0], sh[1], sh[2], sh[3]))
    @staticmethod
    def sf(x: float, sh) -> float: return K.exp(_log1mexp(_dp_logcdf(x, sh[0], sh[1], sh[2], sh[3])))
    @staticmethod
    def munp(n: float, sh) -> float:
        m = sh[0]
        s = sh[1]
        a = sh[2]
        b = sh[3]
        if a <= n:
            return NAN
        return (a * b) / ((a - n) * (b + n)) * K.exp(n * m + n ** 2 * s ** 2 / 2)

kstwo = rv_continuous[_KstwoK]('kstwo')
geninvgauss = rv_continuous[_GeninvgaussK]('geninvgauss')
genhyperbolic = rv_continuous[_GenhyperbolicK]('genhyperbolic')
jf_skew_t = rv_continuous[_JfSkewTK]('jf_skew_t')
landau = rv_continuous[_LandauK]('landau')
vonmises_line = rv_continuous[_VonmisesLineK]('vonmises_line')
dpareto_lognorm = rv_continuous[_DparetoLognormK]('dpareto_lognorm')
