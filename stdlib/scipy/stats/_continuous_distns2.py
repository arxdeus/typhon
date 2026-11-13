# Copyright (C) 2026 the Typhon authors
# Ported from scipy 1.18.1 (BSD-3-Clause)

# Continuous distribution kernels, part 2 (scipy.stats._continuous_distns):
# erlang, loguniform/reciprocal, johnsonsb/su, skewnorm, exponnorm, gennorm,
# foldnorm, foldcauchy, nakagami, loggamma, alpha, dgamma, betaprime,
# recipinvgauss, fatiguelife, wrapcauchy, cosine, vonmises, truncnorm,
# nct, ncx2, ncf, pearson3. See _distn_infrastructure for the kernel protocol.

import numpy as np
import scipy.special._kernels as K
import scipy.special._cdf as C
import scipy.special._gamma2 as G2
import scipy.special._misc as M
import scipy.special._noncentral as NC
import scipy.special._bessel as B
import scipy.special._hyp as H
from ._distn_infrastructure import rv_continuous, NAN, INF, _NA, _generic_munp, _map_ppf
from ._continuous_distns import _GammaK, _BetaK, _Chi2K, _TK, _CauchyK, _skew1, _norm_pdf, \
    _norm_logpdf, _norm_cdf, _norm_sf, _norm_logcdf, _norm_logsf, _norm_ppf, _norm_isf, \
    _PI, _LOG2, _SQRT_2_OVER_PI, _SQRT_PI, _EULER

from C import atan(float) -> float
from C import atan2(float, float) -> float
from C import tan(float) -> float
from C import sinh(float) -> float
from C import cosh(float) -> float
from C import asinh(float) -> float
from C import fabs(float) -> float
from C import fmod(float, float) -> float
from C import round(float) -> float

_PI64 = 3.141592653589793

# ----------------------------------------------------------------- erlang

@tuple
class _ErlangK:
    @staticmethod
    def nshape() -> Literal[int]: return 1
    @staticmethod
    def a(sh) -> float: return 0.0
    @staticmethod
    def rvs(rs, sh, size): return rs.standard_gamma(sh[0], size)
    @staticmethod
    def logpdf(z: float, sh) -> float: return _GammaK.logpdf(z, sh)
    @staticmethod
    def pdf(z: float, sh) -> float: return _GammaK.pdf(z, sh)
    @staticmethod
    def cdf(z: float, sh) -> float: return _GammaK.cdf(z, sh)
    @staticmethod
    def sf(z: float, sh) -> float: return _GammaK.sf(z, sh)
    @staticmethod
    def ppf(q: float, sh) -> float: return _GammaK.ppf(q, sh)
    @staticmethod
    def isf(q: float, sh) -> float: return _GammaK.isf(q, sh)
    @staticmethod
    def stats(sh): return _GammaK.stats(sh)
    @staticmethod
    def munp(n: float, sh) -> float: return _GammaK.munp(n, sh)
    @staticmethod
    def entropy(sh) -> float: return _GammaK.entropy(sh)
    @staticmethod
    def fitstart(data: np.ndarray[float, 1]) -> List[float]:
        return [float(int(4.0 / (1e-8 + _skew1(data) ** 2)))]

# --------------------------------------------------- loguniform/reciprocal

@tuple
class _ReciprocalK:
    @staticmethod
    def nshape() -> Literal[int]: return 2
    @staticmethod
    def check(sh) -> bool: return sh[0] > 0 and sh[1] > sh[0]
    @staticmethod
    def a(sh) -> float: return sh[0]
    @staticmethod
    def b(sh) -> float: return sh[1]
    @staticmethod
    def logpdf(x: float, sh) -> float:
        return -K.log(x) - K.log(K.log(sh[1]) - K.log(sh[0]))
    @staticmethod
    def pdf(x: float, sh) -> float: return K.exp(_ReciprocalK.logpdf(x, sh))
    @staticmethod
    def cdf(x: float, sh) -> float:
        a = sh[0]
        b = sh[1]
        return (K.log(x) - K.log(a)) / (K.log(b) - K.log(a))
    @staticmethod
    def ppf(q: float, sh) -> float:
        a = sh[0]
        b = sh[1]
        return K.exp(K.log(a) + q * (K.log(b) - K.log(a)))
    @staticmethod
    def munp(n: float, sh) -> float:
        a = sh[0]
        b = sh[1]
        if n == 0:
            return 1.0
        t1 = 1 / (K.log(b) - K.log(a)) / n
        # exp(_log_diff(n log b, n log a)) = b^n - a^n
        lp = n * K.log(b)
        lq = n * K.log(a)
        t2 = K.exp(lp + K.log1p(-K.exp(lq - lp)))
        return t1 * t2
    @staticmethod
    def entropy(sh) -> float:
        a = sh[0]
        b = sh[1]
        return 0.5 * (K.log(a) + K.log(b)) + K.log(K.log(b) - K.log(a))

# ---------------------------------------------------------- johnsonsb / su

@tuple
class _JohnsonsbK:
    @staticmethod
    def nshape() -> Literal[int]: return 2
    @staticmethod
    def check(sh) -> bool: return sh[1] > 0 and sh[0] == sh[0]
    @staticmethod
    def a(sh) -> float: return 0.0
    @staticmethod
    def b(sh) -> float: return 1.0
    @staticmethod
    def open_support() -> Literal[bool]: return True
    @staticmethod
    def pdf(x: float, sh) -> float:
        trm = _norm_pdf(sh[0] + sh[1] * K.logit(x))
        return sh[1] * 1.0 / (x * (1 - x)) * trm
    @staticmethod
    def cdf(x: float, sh) -> float: return _norm_cdf(sh[0] + sh[1] * K.logit(x))
    @staticmethod
    def ppf(q: float, sh) -> float: return K.expit(1.0 / sh[1] * (_norm_ppf(q) - sh[0]))
    @staticmethod
    def sf(x: float, sh) -> float: return _norm_sf(sh[0] + sh[1] * K.logit(x))
    @staticmethod
    def isf(q: float, sh) -> float: return K.expit(1.0 / sh[1] * (_norm_isf(q) - sh[0]))

@tuple
class _JohnsonsuK:
    @staticmethod
    def nshape() -> Literal[int]: return 2
    @staticmethod
    def check(sh) -> bool: return sh[1] > 0 and sh[0] == sh[0]
    @staticmethod
    def pdf(x: float, sh) -> float:
        trm = _norm_pdf(sh[0] + sh[1] * asinh(x))
        return sh[1] * 1.0 / K.sqrt(x * x + 1.0) * trm
    @staticmethod
    def cdf(x: float, sh) -> float: return _norm_cdf(sh[0] + sh[1] * asinh(x))
    @staticmethod
    def ppf(q: float, sh) -> float: return sinh((_norm_ppf(q) - sh[0]) / sh[1])
    @staticmethod
    def sf(x: float, sh) -> float: return _norm_sf(sh[0] + sh[1] * asinh(x))
    @staticmethod
    def isf(q: float, sh) -> float: return sinh((_norm_isf(q) - sh[0]) / sh[1])
    @staticmethod
    def stats(sh):
        a = sh[0]
        b = sh[1]
        bn2 = b ** -2.
        expbn2 = K.exp(bn2)
        a_b = a / b
        mu = -expbn2 ** 0.5 * sinh(a_b)
        mu2 = 0.5 * K.expm1(bn2) * (expbn2 * cosh(2 * a_b) + 1)
        t1 = expbn2 ** .5 * K.expm1(bn2) ** 0.5
        t2 = 3 * sinh(a_b)
        t3 = expbn2 * (expbn2 + 2) * sinh(3 * a_b)
        denom = K.sqrt(2.0) * (1 + expbn2 * cosh(2 * a_b)) ** (3 / 2)
        g1 = -t1 * (t2 + t3) / denom
        t1 = 3 + 6 * expbn2
        t2 = 4 * expbn2 ** 2 * (expbn2 + 2) * cosh(2 * a_b)
        t3 = expbn2 ** 2 * cosh(4 * a_b)
        t4 = -3 + 3 * expbn2 ** 2 + 2 * expbn2 ** 3 + expbn2 ** 4
        denom = 2 * (1 + expbn2 * cosh(2 * a_b)) ** 2
        g2 = (t1 + t2 + t3 * t4) / denom - 3
        return (mu, mu2, g1, g2)

# ---------------------------------------------------------------- skewnorm

def _skewnorm_cdf(x: float, a: float) -> float:
    # boost skew_normal cdf (loc 0, scale 1)
    return K.ndtr(x) - 2 * G2.owens_t(x, a)

class _SkewCdfRoot:
    a: float
    q: float
    comp: bool
    def __init__(self, a: float, q: float, comp: bool):
        self.a = a
        self.q = q
        self.comp = comp
    def __call__(self, x: float) -> float:
        if self.comp:
            return (1.0 - _skewnorm_cdf(x, self.a)) - self.q
        return _skewnorm_cdf(x, self.a) - self.q

def _skewnorm_quantile(p: float, a: float, comp: bool) -> float:
    # boost skew_normal quantile: Cornish-Fisher start, then root find on cdf
    if a == 0:
        return -_norm_ppf(p) if comp else _norm_ppf(p)
    from scipy.optimize import brentq
    pp = 1.0 - p if comp else p
    x = _norm_ppf(pp)
    d = a / K.sqrt(1 + a * a)
    const = K.sqrt(2 / _PI) * d
    mean = const
    var = 1 - const * const
    skew = ((4 - _PI) / 2) * (const / K.sqrt(1 - const * const)) ** 3
    x2 = x * x
    guess = mean + K.sqrt(var) * (x + (x2 - 1) * skew / 6)
    f = _SkewCdfRoot(a, p, comp)
    lo = guess - 1.0
    hi = guess + 1.0
    step = 1.0
    while (f(lo) > 0) == (f(hi) > 0):
        step *= 2
        lo = guess - step
        hi = guess + step
    return brentq(f, lo, hi, xtol=1e-300, rtol=8.881784197001252e-16, maxiter=200)

@tuple
class _SkewnormK:
    @staticmethod
    def nshape() -> Literal[int]: return 1
    @staticmethod
    def check(sh) -> bool: return sh[0] == sh[0] and fabs(sh[0]) != INF
    @staticmethod
    def rvs(rs, sh, size):
        a = sh[0]
        u0 = rs.normal(size=size)
        v = rs.normal(size=size)
        d = a / K.sqrt(1 + a * a)
        u1 = d * u0 + v * K.sqrt(1 - d * d)
        return np.where(u0 >= 0, u1, -u1)
    @staticmethod
    def pdf(x: float, sh) -> float:
        a = sh[0]
        if a == 0:
            return _norm_pdf(x)
        return 2. * _norm_pdf(x) * _norm_cdf(a * x)
    @staticmethod
    def logpdf(x: float, sh) -> float:
        a = sh[0]
        if a == 0:
            return _norm_logpdf(x)
        return K.log(2.0) + _norm_logpdf(x) + _norm_logcdf(a * x)
    @staticmethod
    def cdf(x: float, sh) -> float:
        a = sh[0]
        c = _skewnorm_cdf(x, a)
        if c < 1e-6 and a > 0:
            from scipy.integrate import quad
            c = quad(_SkewPdf(a), -INF, x)[0]
        return min(max(c, 0.0), 1.0)
    @staticmethod
    def sf(x: float, sh) -> float: return _SkewnormK.cdf(-x, (-sh[0],))
    @staticmethod
    def ppf(q: float, sh) -> float: return _skewnorm_quantile(q, sh[0], False)
    @staticmethod
    def isf(q: float, sh) -> float: return _skewnorm_quantile(q, sh[0], True)
    @staticmethod
    def stats(sh):
        a = sh[0]
        const = K.sqrt(2 / _PI) * a / K.sqrt(1 + a * a)
        return (const, 1 - const * const,
                ((4 - _PI) / 2) * (const / K.sqrt(1 - const * const)) ** 3,
                (2 * (_PI - 3)) * (const ** 4 / (1 - const * const) ** 2))
    @staticmethod
    def fitstart_full(data: np.ndarray[float, 1], fx: List[float]) -> List[float]:
        # skewnorm_gen.fit (method='mle'): method-of-moments start
        fa = fx[0]
        floc = fx[1]
        fscale = fx[2]
        a = fa
        d = 0.0
        if a != a:
            s = _skew1(data)
            s = min(max(s, -0.99), 0.99)
            s_23 = K.pow(fabs(s), 2 / 3)
            sg = 1.0 if s > 0 else (-1.0 if s < 0 else 0.0)
            d = sg * K.sqrt(_PI / 2 * s_23 / (s_23 + K.pow((4 - _PI) / 2, 2 / 3)))
            a = K.sqrt(d * d / (1 - d * d)) * sg
        else:
            d = a / K.sqrt(1 + a * a)
        scale = fscale
        if scale != scale:
            v = float(np.var(data))
            scale = K.sqrt(v / (1 - 2 * d * d / _PI))
        loc = floc
        if loc != loc:
            loc = float(np.mean(data)) - scale * d * K.sqrt(2 / _PI)
        return [a, loc, scale]

class _SkewPdf:
    a: float
    def __init__(self, a: float):
        self.a = a
    def __call__(self, x: float) -> float:
        return _SkewnormK.pdf(x, (self.a,))

# --------------------------------------------------------------- exponnorm

@tuple
class _ExponnormK:
    @staticmethod
    def nshape() -> Literal[int]: return 1
    @staticmethod
    def rvs(rs, sh, size):
        expval = rs.standard_exponential(size) * sh[0]
        gval = rs.standard_normal(size)
        return expval + gval
    @staticmethod
    def logpdf(x: float, sh) -> float:
        k = sh[0]
        u = (-x + 1.0 / k) / K.sqrt(2.0)
        if u >= 0 and k < 1:
            return -K.log(2.0) - 0.5 * (x ** 2) - K.log(k) + K.log(K.erfcx(u))
        invK = 1.0 / k
        exparg = invK * (-x + 0.5 * invK)
        return exparg + _norm_logcdf(x - invK) - K.log(k)
    @staticmethod
    def pdf(x: float, sh) -> float: return K.exp(_ExponnormK.logpdf(x, sh))
    @staticmethod
    def cdf(x: float, sh) -> float:
        k = sh[0]
        u = (-x + 1.0 / k) / K.sqrt(2.0)
        if u >= 0 and k < 1:
            return _norm_cdf(x) - 0.5 * K.exp(-0.5 * x ** 2) * K.erfcx(u)
        invK = 1.0 / k
        expval = invK * (0.5 * invK - x)
        logprod = expval + _norm_logcdf(x - invK)
        return _norm_cdf(x) - K.exp(logprod)
    @staticmethod
    def sf(x: float, sh) -> float:
        k = sh[0]
        u = (-x + 1.0 / k) / K.sqrt(2.0)
        if u >= 0 and k < 1:
            return _norm_cdf(-x) + 0.5 * K.exp(-0.5 * x ** 2) * K.erfcx(u)
        invK = 1.0 / k
        expval = invK * (0.5 * invK - x)
        logprod = expval + _norm_logcdf(x - invK)
        return _norm_cdf(-x) + K.exp(logprod)
    @staticmethod
    def stats(sh):
        k = sh[0]
        K2 = k * k
        opK2 = 1.0 + K2
        return (k, opK2, 2 * k ** 3 * opK2 ** (-1.5), 6.0 * K2 * K2 * opK2 ** (-2))

# ----------------------------------------------------------------- gennorm

@tuple
class _GennormK:
    @staticmethod
    def nshape() -> Literal[int]: return 1
    @staticmethod
    def rvs(rs, sh, size):
        beta = sh[0]
        z = rs.gamma(1 / beta, size=size)
        y = np.asarray(z ** (1 / beta))
        mask = rs.random(size=y.shape) < 0.5
        y[mask] = -y[mask]
        return y
    @staticmethod
    def logpdf(x: float, sh) -> float:
        beta = sh[0]
        return K.log(0.5 * beta) - K.gammaln(1.0 / beta) - K.pow(fabs(x), beta)
    @staticmethod
    def pdf(x: float, sh) -> float: return K.exp(_GennormK.logpdf(x, sh))
    @staticmethod
    def cdf(x: float, sh) -> float:
        beta = sh[0]
        c = 0.5 * (1.0 if x > 0 else (-1.0 if x < 0 else 0.0))
        return (0.5 + c) - c * K.gammaincc(1.0 / beta, K.pow(fabs(x), beta))
    @staticmethod
    def ppf(x: float, sh) -> float:
        beta = sh[0]
        d = x - 0.5
        c = 1.0 if d > 0 else (-1.0 if d < 0 else 0.0)
        return c * K.pow(K.gammainccinv(1.0 / beta, (1.0 + c) - 2.0 * c * x), 1.0 / beta)
    @staticmethod
    def sf(x: float, sh) -> float: return _GennormK.cdf(-x, sh)
    @staticmethod
    def isf(x: float, sh) -> float: return -_GennormK.ppf(x, sh)
    @staticmethod
    def munp(n: float, sh) -> float:
        beta = sh[0]
        if n == 0:
            return 1.
        if int(n) % 2 == 0:
            return K.exp(K.gammaln((n + 1.0) / beta) - K.gammaln(1.0 / beta))
        return 0.
    @staticmethod
    def stats(sh):
        beta = sh[0]
        c1 = K.gammaln(1.0 / beta)
        c3 = K.gammaln(3.0 / beta)
        c5 = K.gammaln(5.0 / beta)
        return (0., K.exp(c3 - c1), 0., K.exp(c5 + c1 - 2.0 * c3) - 3.)
    @staticmethod
    def entropy(sh) -> float:
        beta = sh[0]
        return 1. / beta - K.log(.5 * beta) + K.gammaln(1. / beta)

# ----------------------------------------------------- foldnorm/foldcauchy

@tuple
class _FoldnormK:
    @staticmethod
    def nshape() -> Literal[int]: return 1
    @staticmethod
    def check(sh) -> bool: return sh[0] >= 0
    @staticmethod
    def a(sh) -> float: return 0.0
    @staticmethod
    def rvs(rs, sh, size): return np.abs(rs.standard_normal(size) + sh[0])
    @staticmethod
    def pdf(x: float, sh) -> float: return _norm_pdf(x + sh[0]) + _norm_pdf(x - sh[0])
    @staticmethod
    def cdf(x: float, sh) -> float:
        c = sh[0]
        s2 = K.sqrt(2.0)
        return 0.5 * (K.erf((x - c) / s2) + K.erf((x + c) / s2))
    @staticmethod
    def sf(x: float, sh) -> float: return _norm_sf(x - sh[0]) + _norm_sf(x + sh[0])
    @staticmethod
    def stats(sh):
        c = sh[0]
        c2 = c * c
        expfac = K.exp(-0.5 * c2) / K.sqrt(2. * _PI)
        mu = 2. * expfac + c * K.erf(c / K.sqrt(2.0))
        mu2 = c2 + 1 - mu * mu
        g1 = 2. * (mu * mu * mu - c2 * mu - expfac)
        g1 /= K.pow(mu2, 1.5)
        g2 = c2 * (c2 + 6.) + 3 + 8. * expfac * mu
        g2 += (2. * (c2 - 3.) - 3. * mu ** 2) * mu ** 2
        g2 = g2 / mu2 ** 2.0 - 3.
        return (mu, mu2, g1, g2)

@tuple
class _FoldcauchyK:
    @staticmethod
    def nshape() -> Literal[int]: return 1
    @staticmethod
    def check(sh) -> bool: return sh[0] >= 0
    @staticmethod
    def a(sh) -> float: return 0.0
    @staticmethod
    def rvs(rs, sh, size):
        # abs(cauchy.rvs(loc=c)): cauchy uses the generic ppf(uniform) path
        return np.abs(_map_ppf(_CauchyK, rs.uniform(size=size), ()) + sh[0])
    @staticmethod
    def pdf(x: float, sh) -> float:
        c = sh[0]
        return 1.0 / _PI * (1.0 / (1 + (x - c) ** 2) + 1.0 / (1 + (x + c) ** 2))
    @staticmethod
    def cdf(x: float, sh) -> float:
        c = sh[0]
        return 1.0 / _PI * (atan(x - c) + atan(x + c))
    @staticmethod
    def sf(x: float, sh) -> float:
        c = sh[0]
        return (atan2(1.0, x - c) + atan2(1.0, x + c)) / _PI
    @staticmethod
    def stats(sh): return (INF, INF, NAN, NAN)

# ---------------------------------------------------------------- nakagami

@tuple
class _NakagamiK:
    @staticmethod
    def nshape() -> Literal[int]: return 1
    @staticmethod
    def a(sh) -> float: return 0.0
    @staticmethod
    def rvs(rs, sh, size): return np.sqrt(rs.standard_gamma(sh[0], size=size) / sh[0])
    @staticmethod
    def logpdf(x: float, sh) -> float:
        nu = sh[0]
        return (K.log(2.0) + K.xlogy(nu, nu) - K.gammaln(nu)
                + K.xlogy(2 * nu - 1, x) - nu * x ** 2)
    @staticmethod
    def pdf(x: float, sh) -> float: return K.exp(_NakagamiK.logpdf(x, sh))
    @staticmethod
    def cdf(x: float, sh) -> float: return K.gammainc(sh[0], sh[0] * x * x)
    @staticmethod
    def ppf(q: float, sh) -> float: return K.sqrt(1.0 / sh[0] * K.gammaincinv(sh[0], q))
    @staticmethod
    def sf(x: float, sh) -> float: return K.gammaincc(sh[0], sh[0] * x * x)
    @staticmethod
    def isf(p: float, sh) -> float: return K.sqrt(1 / sh[0] * K.gammainccinv(sh[0], p))
    @staticmethod
    def stats(sh):
        nu = sh[0]
        mu = G2.poch(nu, 0.5) / K.sqrt(nu)
        mu2 = 1.0 - mu * mu
        g1 = mu * (1 - 4 * nu * mu2) / 2.0 / nu / K.pow(mu2, 1.5)
        g2 = -6 * mu ** 4 * nu + (8 * nu - 2) * mu ** 2 - 2 * nu + 1
        g2 /= nu * mu2 ** 2.0
        return (mu, mu2, g1, g2)
    @staticmethod
    def entropy(sh) -> float:
        nu = sh[0]
        A = K.gammaln(nu)
        Bv = nu - (nu - 0.5) * K.digamma(nu)
        Cv = -0.5 * K.log(nu) - K.log(2.0)
        if nu > 5e4:
            return Cv + (0.5 * K.log(2 * _PI) + 0.5) - 1 / (12 * nu)
        return A + Bv + Cv
    @staticmethod
    def fitstart_full(data: np.ndarray[float, 1], fx: List[float]) -> List[float]:
        # nakagami_gen._fitstart: (1.0, min(data), sqrt(sum((data - loc)**2) / n))
        loc = float(np.min(data))
        scale = K.sqrt(float(np.sum((data - loc) ** 2)) / len(data))
        return [1.0, loc, scale]

# ---------------------------------------------------------------- loggamma

_XMIN = 2.2250738585072014e-308
_LOGXMIN = -708.3964185322641

@tuple
class _LoggammaK:
    @staticmethod
    def nshape() -> Literal[int]: return 1
    @staticmethod
    def rvs(rs, sh, size):
        c = sh[0]
        return np.log(rs.gamma(c + 1, size=size)) + np.log(rs.uniform(size=size)) / c
    @staticmethod
    def logpdf(x: float, sh) -> float:
        c = sh[0]
        return c * x - K.exp(x) - K.gammaln(c)
    @staticmethod
    def pdf(x: float, sh) -> float:
        c = sh[0]
        return K.exp(c * x - K.exp(x) - K.gammaln(c))
    @staticmethod
    def cdf(x: float, sh) -> float:
        c = sh[0]
        if x < _LOGXMIN:
            return K.exp(c * x - K.gammaln(c + 1))
        return K.gammainc(c, K.exp(x))
    @staticmethod
    def ppf(q: float, sh) -> float:
        c = sh[0]
        g = K.gammaincinv(c, q)
        if g < _XMIN:
            return (K.log(q) + K.gammaln(c + 1)) / c
        return K.log(g)
    @staticmethod
    def sf(x: float, sh) -> float:
        c = sh[0]
        if x < _LOGXMIN:
            return -K.expm1(c * x - K.gammaln(c + 1))
        return K.gammaincc(c, K.exp(x))
    @staticmethod
    def isf(q: float, sh) -> float:
        c = sh[0]
        g = K.gammainccinv(c, q)
        if g < _XMIN:
            return (K.log1p(-q) + K.gammaln(c + 1)) / c
        return K.log(g)
    @staticmethod
    def stats(sh):
        c = sh[0]
        var = G2.polygamma(1.0, c)
        return (K.digamma(c), var, G2.polygamma(2.0, c) / K.pow(var, 1.5),
                G2.polygamma(3.0, c) / (var * var))
    @staticmethod
    def entropy(sh) -> float:
        c = sh[0]
        if c >= 45:
            term = -0.5 * K.log(c) + c ** -1. / 6 - c ** -3. / 90 + c ** -5. / 210
            return (0.5 * K.log(2 * _PI) + 0.5) + term
        return K.gammaln(c) - c * K.digamma(c) + c

# -------------------------------------------------------------- alpha/dgamma

@tuple
class _AlphaK:
    @staticmethod
    def nshape() -> Literal[int]: return 1
    @staticmethod
    def a(sh) -> float: return 0.0
    @staticmethod
    def open_support() -> Literal[bool]: return True
    @staticmethod
    def pdf(x: float, sh) -> float:
        a = sh[0]
        return 1.0 / (x ** 2) / _norm_cdf(a) * _norm_pdf(a - 1.0 / x)
    @staticmethod
    def logpdf(x: float, sh) -> float:
        a = sh[0]
        return -2 * K.log(x) + _norm_logpdf(a - 1.0 / x) - K.log(_norm_cdf(a))
    @staticmethod
    def cdf(x: float, sh) -> float:
        a = sh[0]
        return _norm_cdf(a - 1.0 / x) / _norm_cdf(a)
    @staticmethod
    def ppf(q: float, sh) -> float:
        a = sh[0]
        return 1.0 / (a - _norm_ppf(q * _norm_cdf(a)))
    @staticmethod
    def stats(sh): return (INF, INF, NAN, NAN)

@tuple
class _DgammaK:
    @staticmethod
    def nshape() -> Literal[int]: return 1
    @staticmethod
    def rvs(rs, sh, size):
        u = rs.uniform(size=size)
        gm = rs.standard_gamma(sh[0], size)
        return gm * np.where(u >= 0.5, 1.0, -1.0)
    @staticmethod
    def pdf(x: float, sh) -> float:
        a = sh[0]
        ax = fabs(x)
        return 1.0 / (2 * K.gamma(a)) * K.pow(ax, a - 1.0) * K.exp(-ax)
    @staticmethod
    def logpdf(x: float, sh) -> float:
        a = sh[0]
        ax = fabs(x)
        return K.xlogy(a - 1.0, ax) - ax - K.log(2.0) - K.gammaln(a)
    @staticmethod
    def cdf(x: float, sh) -> float:
        a = sh[0]
        return 0.5 + 0.5 * K.gammainc(a, x) if x > 0 else 0.5 * K.gammaincc(a, -x)
    @staticmethod
    def sf(x: float, sh) -> float:
        a = sh[0]
        return 0.5 * K.gammaincc(a, x) if x > 0 else 0.5 + 0.5 * K.gammainc(a, -x)
    @staticmethod
    def entropy(sh) -> float: return _GammaK.entropy(sh) - K.log(0.5)
    @staticmethod
    def ppf(q: float, sh) -> float:
        a = sh[0]
        return K.gammaincinv(a, 2 * q - 1) if q > 0.5 else -K.gammainccinv(a, 2 * q)
    @staticmethod
    def isf(q: float, sh) -> float:
        a = sh[0]
        return -K.gammaincinv(a, 2 * q - 1) if q > 0.5 else K.gammainccinv(a, 2 * q)
    @staticmethod
    def stats(sh):
        a = sh[0]
        mu2 = a * (a + 1.0)
        return (0.0, mu2, 0.0, (a + 2.0) * (a + 3.0) / mu2 - 3.0)

# -------------------------------------------------------------- betaprime

@tuple
class _BetaprimeK:
    @staticmethod
    def nshape() -> Literal[int]: return 2
    @staticmethod
    def a(sh) -> float: return 0.0
    @staticmethod
    def open_support() -> Literal[bool]: return True
    @staticmethod
    def rvs(rs, sh, size):
        u1 = rs.standard_gamma(sh[0], size)
        u2 = rs.standard_gamma(sh[1], size)
        return u1 / u2
    @staticmethod
    def logpdf(x: float, sh) -> float:
        a = sh[0]
        b = sh[1]
        return K.xlogy(a - 1.0, x) - K.xlog1py(a + b, x) - K.betaln(a, b)
    @staticmethod
    def pdf(x: float, sh) -> float: return K.exp(_BetaprimeK.logpdf(x, sh))
    @staticmethod
    def cdf(x: float, sh) -> float:
        a = sh[0]
        b = sh[1]
        if x > 1:
            return _BetaK.sf(1 / (1 + x), (b, a))
        return _BetaK.cdf(x / (1 + x), (a, b))
    @staticmethod
    def sf(x: float, sh) -> float:
        a = sh[0]
        b = sh[1]
        if x > 1:
            return _BetaK.cdf(1 / (1 + x), (b, a))
        return _BetaK.sf(x / (1 + x), (a, b))
    @staticmethod
    def ppf(p: float, sh) -> float:
        a = sh[0]
        b = sh[1]
        r = _BetaK.ppf(p, (a, b))
        if r > 0.9999:
            return 1 / _BetaK.isf(p, (b, a)) - 1
        return r / (1 - r)
    @staticmethod
    def munp(n: float, sh) -> float:
        a = sh[0]
        b = sh[1]
        if b > n:
            r = 1.0
            for i in range(1, int(n) + 1):
                r *= (a + i - 1) / (b - i)
            return r
        return INF

# ----------------------------------------- recipinvgauss / fatiguelife

@tuple
class _RecipinvgaussK:
    @staticmethod
    def nshape() -> Literal[int]: return 1
    @staticmethod
    def a(sh) -> float: return 0.0
    @staticmethod
    def rvs(rs, sh, size): return 1.0 / rs.wald(sh[0], 1.0, size=size)
    @staticmethod
    def logpdf(x: float, sh) -> float:
        mu = sh[0]
        if not (x > 0):
            return -INF
        return -(1 - mu * x) ** 2.0 / (2 * x * mu ** 2.0) - 0.5 * K.log(2 * _PI * x)
    @staticmethod
    def pdf(x: float, sh) -> float: return K.exp(_RecipinvgaussK.logpdf(x, sh))
    @staticmethod
    def cdf(x: float, sh) -> float:
        mu = sh[0]
        if not (x > 0):
            return 0.0
        trm1 = 1.0 / mu - x
        trm2 = 1.0 / mu + x
        isqx = 1.0 / K.sqrt(x)
        return _norm_cdf(-isqx * trm1) - K.exp(2.0 / mu) * _norm_cdf(-isqx * trm2)
    @staticmethod
    def sf(x: float, sh) -> float:
        mu = sh[0]
        if not (x > 0):
            return 1.0
        trm1 = 1.0 / mu - x
        trm2 = 1.0 / mu + x
        isqx = 1.0 / K.sqrt(x)
        return _norm_cdf(isqx * trm1) + K.exp(2.0 / mu) * _norm_cdf(-isqx * trm2)

@tuple
class _FatiguelifeK:
    @staticmethod
    def nshape() -> Literal[int]: return 1
    @staticmethod
    def a(sh) -> float: return 0.0
    @staticmethod
    def open_support() -> Literal[bool]: return True
    @staticmethod
    def rvs(rs, sh, size):
        z = rs.standard_normal(size)
        x = 0.5 * sh[0] * z
        x2 = x * x
        return 1.0 + 2 * x2 + 2 * x * np.sqrt(1 + x2)
    @staticmethod
    def logpdf(x: float, sh) -> float:
        c = sh[0]
        return (K.log(x + 1) - (x - 1) ** 2 / (2.0 * x * c ** 2) - K.log(2 * c)
                - 0.5 * (K.log(2 * _PI) + 3 * K.log(x)))
    @staticmethod
    def pdf(x: float, sh) -> float: return K.exp(_FatiguelifeK.logpdf(x, sh))
    @staticmethod
    def cdf(x: float, sh) -> float:
        return _norm_cdf(1.0 / sh[0] * (K.sqrt(x) - 1.0 / K.sqrt(x)))
    @staticmethod
    def ppf(q: float, sh) -> float:
        tmp = sh[0] * _norm_ppf(q)
        return 0.25 * (tmp + K.sqrt(tmp ** 2 + 4)) ** 2
    @staticmethod
    def sf(x: float, sh) -> float:
        return _norm_sf(1.0 / sh[0] * (K.sqrt(x) - 1.0 / K.sqrt(x)))
    @staticmethod
    def isf(q: float, sh) -> float:
        tmp = -sh[0] * _norm_ppf(q)
        return 0.25 * (tmp + K.sqrt(tmp ** 2 + 4)) ** 2
    @staticmethod
    def stats(sh):
        c = sh[0]
        c2 = c * c
        den = 5.0 * c2 + 4.0
        return (c2 / 2.0 + 1.0, c2 * den / 4.0, 4 * c * (11 * c2 + 6.0) / K.pow(den, 1.5),
                6 * c2 * (93 * c2 + 40.0) / den ** 2.0)

# -------------------------------------------------------------- wrapcauchy

@tuple
class _WrapcauchyK:
    @staticmethod
    def nshape() -> Literal[int]: return 1
    @staticmethod
    def check(sh) -> bool: return sh[0] > 0 and sh[0] < 1
    @staticmethod
    def a(sh) -> float: return 0.0
    @staticmethod
    def b(sh) -> float: return 2 * _PI
    @staticmethod
    def rvs_post(v): return np.mod(v, 2 * _PI)
    @staticmethod
    def pdf(x: float, sh) -> float:
        c = sh[0]
        return (1.0 - c * c) / (2 * _PI * (1 + c * c - 2 * c * K.cos(x)))
    @staticmethod
    def cdf(x: float, sh) -> float:
        c = sh[0]
        cr = (1 + c) / (1 - c)
        if x < _PI:
            return 1 / _PI * atan(cr * tan(x / 2))
        return 1 - 1 / _PI * atan(cr * tan((2 * _PI - x) / 2))
    @staticmethod
    def ppf(q: float, sh) -> float:
        c = sh[0]
        val = (1.0 - c) / (1.0 + c)
        if q < 1.0 / 2:
            return 2 * atan(val * tan(_PI * q))
        return 2 * _PI - 2 * atan(val * tan(_PI * (1 - q)))
    @staticmethod
    def entropy(sh) -> float:
        c = sh[0]
        return K.log(2 * _PI * (1 - c * c))

# ----------------------------------------------------------------- cosine

def _cosine_cdf(x: float) -> float:
    if x >= _PI64:
        return 1.0
    if x < -_PI64:
        return 0.0
    if x < -1.6:
        h = (x + _PI64) + 1.2246467991473532e-16
        h2 = h * h
        h3 = h2 * h
        numer = h3 * K._polevl(h2, (-3.8360369451359084e-08, 1.0235408442872927e-05,
                                    -0.0007883197097740538, 0.026525823848649224))
        denom = K._polevl(h2, (1.6955280904096042e-11, 6.498171564823105e-09,
                               1.4162345851873058e-06, 0.00020944197182753272,
                               0.020281047093125535, 1.0))
        return numer / denom
    return 0.5 + (x + K.sin(x)) / (2 * _PI64)

def _cosine_invcdf(p: float) -> float:
    if p < 0 or p > 1:
        return NAN
    if p <= 1e-48:
        return -_PI64
    if p == 1:
        return _PI64
    sgn = 1.0
    if p > 0.5:
        p = 1.0 - p
        sgn = -1.0
    x = 0.0
    if p < 0.0925:
        s = K.cbrt(12 * _PI64 * p)
        s2 = s * s
        x = s * K._polevl(s2, (1.1911667949082915e-08, 1.683039183039183e-07, 43.0 / 17248000,
                               1.0 / 25200, 1.0 / 1400, 1.0 / 60, 1.0)) - _PI64
    else:
        y = _PI64 * (2 * p - 1)
        y2 = y * y
        x = y * K._polevl(y2, (-6.8448463845552725e-09, 3.4900934227012284e-06,
                               -0.00030539712907115167, 0.009350454384541677,
                               -0.11602142940208726, 0.5)) / K._polevl(
            y2, (-5.579679571562129e-08, 1.3728570152788793e-05, -0.0008916919927321117,
                 0.022927496105281435, -0.25287619213750784, 1.0))
    if 0.0018 < p and p < 0.42:
        f0 = _PI64 + x + K.sin(x) - 2 * _PI64 * p
        f1 = 1 + K.cos(x)
        f2 = -K.sin(x)
        x = x - 2 * f0 * f1 / (2 * f1 * f1 - f0 * f2)
    return sgn * x

@tuple
class _CosineK:
    @staticmethod
    def nshape() -> Literal[int]: return 0
    @staticmethod
    def a(sh) -> float: return -_PI
    @staticmethod
    def b(sh) -> float: return _PI
    @staticmethod
    def pdf(x: float, sh) -> float: return 1.0 / 2 / _PI * (1 + K.cos(x))
    @staticmethod
    def logpdf(x: float, sh) -> float:
        c = K.cos(x)
        if c == -1:
            return -INF
        return K.log1p(c) - K.log(2 * _PI)
    @staticmethod
    def cdf(x: float, sh) -> float: return _cosine_cdf(x)
    @staticmethod
    def sf(x: float, sh) -> float: return _cosine_cdf(-x)
    @staticmethod
    def ppf(p: float, sh) -> float: return _cosine_invcdf(p)
    @staticmethod
    def isf(p: float, sh) -> float: return -_cosine_invcdf(p)
    @staticmethod
    def stats(sh):
        v = (_PI * _PI / 3.0) - 2.0
        k = -6.0 * (_PI ** 4 - 90) / (5.0 * (_PI * _PI - 6) ** 2)
        return (0.0, v, 0.0, k)
    @staticmethod
    def entropy(sh) -> float: return K.log(4 * _PI) - 1.0

# ---------------------------------------------------------------- vonmises

def _von_mises_cdf_series(k: float, x: float, p: int) -> float:
    s = K.sin(x)
    c = K.cos(x)
    sn = K.sin(p * x)
    cn = K.cos(p * x)
    R = 0.0
    V = 0.0
    n = p - 1
    while n >= 1:
        sn, cn = sn * c - cn * s, cn * c + sn * s
        R = k / (2 * n + k * R)
        V = R * (sn / n + V)
        n -= 1
    return 0.5 + x / (2 * _PI) + V / _PI

def _von_mises_cdf(k: float, x: float) -> float:
    ix = round(x / (2 * _PI))
    x = x - ix * (2 * _PI)
    r = 0.0
    if k < 50:
        p = int(1 + 28. + 0.5 * k - 100. / (k + 5.))
        r = _von_mises_cdf_series(k, x, p)
        r = 0.0 if r < 0 else (1.0 if r > 1 else r)
    else:
        b = 0.79788456080286535588 / K.i0e(k)
        r = _norm_cdf(b * K.sin(x / 2.))
    return r + ix

@tuple
class _VonmisesK:
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
    def rvs_post(v): return np.mod(v + _PI, 2 * _PI) - _PI
    @staticmethod
    def pdf(x: float, sh) -> float:
        k = sh[0]
        return K.exp(k * M.cosm1(x)) / (2 * _PI * K.i0e(k))
    @staticmethod
    def logpdf(x: float, sh) -> float:
        k = sh[0]
        return k * M.cosm1(x) - K.log(2 * _PI) - K.log(K.i0e(k))
    @staticmethod
    def cdf(x: float, sh) -> float: return _von_mises_cdf(sh[0], x)
    @staticmethod
    def entropy(sh) -> float:
        k = sh[0]
        return -k * K.i1e(k) / K.i0e(k) + K.log(2 * _PI * K.i0e(k)) + k

# --------------------------------------------------------------- truncnorm

def _log_sum(lp: float, lq: float) -> float:
    m = max(lp, lq)
    if m == -INF:
        return -INF
    return m + K.log1p(K.exp(min(lp, lq) - m))

def _log_diff(lp: float, lq: float) -> float:
    # real part of logsumexp([lp, lq + pi*j]) = log(exp(lp) - exp(lq)), lp >= lq
    if lq == -INF:
        return lp
    return lp + K.log1p(-K.exp(lq - lp))

def _log_gauss_mass(a: float, b: float) -> float:
    if b <= 0:
        return _log_diff(_norm_logcdf(b), _norm_logcdf(a))
    if a > 0:
        return _log_diff(_norm_logcdf(-a), _norm_logcdf(-b))
    return K.log1p(-_norm_cdf(a) - _norm_cdf(-b))

@tuple
class _TruncnormK:
    @staticmethod
    def nshape() -> Literal[int]: return 2
    @staticmethod
    def check(sh) -> bool: return sh[0] < sh[1]
    @staticmethod
    def a(sh) -> float: return sh[0]
    @staticmethod
    def b(sh) -> float: return sh[1]
    @staticmethod
    def logpdf(x: float, sh) -> float: return _norm_logpdf(x) - _log_gauss_mass(sh[0], sh[1])
    @staticmethod
    def pdf(x: float, sh) -> float: return K.exp(_TruncnormK.logpdf(x, sh))
    @staticmethod
    def logcdf(x: float, sh) -> float:
        a = sh[0]
        b = sh[1]
        lc = _log_gauss_mass(a, x) - _log_gauss_mass(a, b)
        if lc > -0.1:
            ls = _log_gauss_mass(x, b) - _log_gauss_mass(a, b)
            return K.log1p(-K.exp(ls))
        return lc
    @staticmethod
    def logsf(x: float, sh) -> float:
        a = sh[0]
        b = sh[1]
        ls = _log_gauss_mass(x, b) - _log_gauss_mass(a, b)
        if ls > -0.1:
            lc = _log_gauss_mass(a, x) - _log_gauss_mass(a, b)
            return K.log1p(-K.exp(lc))
        return ls
    @staticmethod
    def cdf(x: float, sh) -> float: return K.exp(_TruncnormK.logcdf(x, sh))
    @staticmethod
    def sf(x: float, sh) -> float: return K.exp(_TruncnormK.logsf(x, sh))
    @staticmethod
    def ppf(q: float, sh) -> float:
        a = sh[0]
        b = sh[1]
        if a < 0:
            return C.ndtri_exp(_log_sum(_norm_logcdf(a), K.log(q) + _log_gauss_mass(a, b)))
        return -C.ndtri_exp(_log_sum(_norm_logcdf(-b), K.log1p(-q) + _log_gauss_mass(a, b)))
    @staticmethod
    def isf(q: float, sh) -> float:
        a = sh[0]
        b = sh[1]
        if b < 0:
            return C.ndtri_exp(_log_diff(_norm_logcdf(b), K.log(q) + _log_gauss_mass(a, b)))
        return -C.ndtri_exp(_log_diff(_norm_logcdf(-a), K.log1p(-q) + _log_gauss_mass(a, b)))
    @staticmethod
    def stats(sh):
        a = sh[0]
        b = sh[1]
        pA = K.exp(_TruncnormK.logpdf(a, sh))
        pB = K.exp(_TruncnormK.logpdf(b, sh))
        m1 = pA - pB
        mu = m1
        def term(p: float, y: float, k: int) -> float:
            return p * y ** float(k) if p != 0 else 0.0
        m2 = 1 + (term(pA, a, 1) + term(-pB, b, 1))
        mu2 = 1 + (term(pA, a - mu, 1) + term(-pB, b - mu, 1))
        m3 = 2 * m1 + (term(pA, a, 2) + term(-pB, b, 2))
        m4 = 3 * m2 + (term(pA, a, 3) + term(-pB, b, 3))
        mu3 = m3 + m1 * (-3 * m2 + 2 * m1 ** 2)
        g1 = mu3 / K.pow(mu2, 1.5)
        mu4 = m4 + m1 * (-4 * m3 + 3 * m1 * (2 * m2 - m1 ** 2))
        g2 = mu4 / mu2 ** 2 - 3
        return (mu, mu2, g1, g2)
    @staticmethod
    def entropy(sh) -> float:
        a = sh[0]
        b = sh[1]
        A = _norm_cdf(a)
        Bv = _norm_cdf(b)
        Z = Bv - A
        Cv = K.log(K.sqrt(2 * _PI * 2.718281828459045) * Z)
        D = (a * _norm_pdf(a) - b * _norm_pdf(b)) / (2 * Z)
        return Cv + D
    @staticmethod
    def fitstart(data: np.ndarray[float, 1]) -> List[float]:
        return [float(np.min(data)), float(np.max(data))]

# ------------------------------------------------------------------- ncx2

def _ncx2_pdf_series(x: float, n: float, lam: float) -> float:
    errtol = 2.220446049250313e-16
    x2 = x / 2
    n2 = n / 2
    l2 = lam / 2
    s = 0.0
    km = (K.sqrt((n2 - 1) * (n2 - 1) + 4 * l2 * x2) - n2 - 1) / 2
    k = 0 if km < 0 else int(km if km < l2 else l2)
    pois = NC._gamma_p_derivative(float(k + 1), l2) * NC._gamma_p_derivative(n2 + k, x2)
    if pois == 0:
        return 0.0
    poisb = pois
    i = k
    while True:
        s += pois
        if pois / s < errtol:
            break
        if i - k >= 1000000:
            break
        pois *= l2 * x2 / ((i + 1) * (n2 + i))
        i += 1
    i = k - 1
    while i >= 0:
        poisb *= (i + 1) * (n2 + i) / (l2 * x2)
        s += poisb
        if poisb / s < errtol:
            break
        i -= 1
    return s / 2

def _ncx2_pdf(x: float, k: float, l: float) -> float:
    if l == 0:
        return _Chi2K.pdf(x, (k,))
    if x == 0:
        return K.exp(-l / 2) / 2 if k == 2 else 0.0
    if l > 50:
        return _ncx2_pdf_series(x, k, l)
    r = K.log(x / l) * (k / 4 - 0.5) - (x + l) / 2
    if fabs(r) >= 709.782712893384 / 4:
        return _ncx2_pdf_series(x, k, l)
    return 0.5 * K.exp(r) * B.iv(k / 2 - 1, K.sqrt(l * x))

class _Ncx2Root:
    df: float
    nc: float
    q: float
    comp: bool
    def __init__(self, df: float, nc: float, q: float, comp: bool):
        self.df = df
        self.nc = nc
        self.q = q
        self.comp = comp
    def __call__(self, x: float) -> float:
        if self.comp:
            return NC.chndtrc(x, self.df, self.nc) - self.q
        return NC.chndtr(x, self.df, self.nc) - self.q

@tuple
class _Ncx2K:
    @staticmethod
    def nshape() -> Literal[int]: return 2
    @staticmethod
    def check(sh) -> bool: return sh[0] > 0 and sh[0] != INF and sh[1] >= 0
    @staticmethod
    def a(sh) -> float: return 0.0
    @staticmethod
    def rvs(rs, sh, size): return rs.noncentral_chisquare(sh[0], sh[1], size)
    @staticmethod
    def pdf(x: float, sh) -> float: return _ncx2_pdf(x, sh[0], sh[1])
    @staticmethod
    def logpdf(x: float, sh) -> float:
        if sh[1] == 0:
            return _Chi2K.logpdf(x, (sh[0],))
        return K.log(_ncx2_pdf(x, sh[0], sh[1]))
    @staticmethod
    def cdf(x: float, sh) -> float:
        if sh[1] == 0:
            return _Chi2K.cdf(x, (sh[0],))
        return NC.chndtr(x, sh[0], sh[1])
    @staticmethod
    def sf(x: float, sh) -> float:
        if sh[1] == 0:
            return _Chi2K.sf(x, (sh[0],))
        return NC.chndtrc(x, sh[0], sh[1])
    @staticmethod
    def ppf(q: float, sh) -> float:
        if sh[1] == 0:
            return _Chi2K.ppf(q, (sh[0],))
        return NC.chndtrix(q, sh[0], sh[1])
    @staticmethod
    def stats(sh):
        df = sh[0]
        nc = sh[1]
        return (df + nc, 2.0 * (df + 2.0 * nc),
                K.sqrt(8.0) * (df + 3 * nc) / K.sqrt((df + 2.0 * nc) ** 3),
                12.0 * (df + 4.0 * nc) / (df + 2.0 * nc) ** 2)

# -------------------------------------------------------------------- ncf

def _nc_beta_pdf(a: float, b: float, lam: float, x: float, y: float) -> float:
    if x == 0:
        return b * K.exp(-lam / 2) if a == 1 else 0.0
    if y == 0:
        return a + lam / 2 if b == 1 else 0.0
    errtol = 2.220446049250313e-16
    l2 = lam / 2
    k = int(l2)
    pois = NC._gamma_p_derivative(float(k + 1), l2)
    beta = (NC._ibeta_deriv_term(a + k, b, x, y) if x < y
            else NC._ibeta_deriv_term(b, a + k, y, x))
    while fabs(beta * pois) < 2.2250738585072014e-308:
        if k == 0 or pois == 0:
            return 0.0
        k //= 2
        pois = NC._gamma_p_derivative(float(k + 1), l2)
        beta = (NC._ibeta_deriv_term(a + k, b, x, y) if x < y
                else NC._ibeta_deriv_term(b, a + k, y, x))
    s = 0.0
    poisf = pois
    betaf = beta
    count = k
    ratio = 0.0
    old_ratio = 0.0
    i = k
    while i >= 0:
        term = beta * pois
        s += term
        ratio = fabs(term / s)
        if (ratio < errtol and ratio < old_ratio) or term == 0:
            count = k - i
            break
        ratio = old_ratio
        pois *= i / l2
        if a + b + i != 1:
            beta *= (a + i - 1) / (x * (a + i + b - 1))
        i -= 1
    old_ratio = 0.0
    i = k + 1
    while True:
        poisf *= l2 / i
        betaf *= x * (a + b + i - 1) / (a + i - 1)
        term = poisf * betaf
        s += term
        ratio = fabs(term / s)
        if (ratio < errtol and ratio < old_ratio) or term == 0:
            break
        old_ratio = ratio
        if count + i - k > 1000000:
            break
        i += 1
    return s

def _ncf_pdf(x: float, v1: float, v2: float, l: float) -> float:
    alpha = v1 / 2
    beta = v2 / 2
    y = x * alpha / beta
    t = y / (1 + y)
    r = 0.0
    if l == 0:
        r = _BetaK.pdf(t, (alpha, beta))
    else:
        r = _nc_beta_pdf(alpha, beta, l, t, 1 - t)
    return r * (v1 / v2) / ((1 + y) * (1 + y))

@tuple
class _NcfK:
    @staticmethod
    def nshape() -> Literal[int]: return 3
    @staticmethod
    def check(sh) -> bool: return sh[0] > 0 and sh[1] > 0 and sh[2] >= 0
    @staticmethod
    def a(sh) -> float: return 0.0
    @staticmethod
    def rvs(rs, sh, size): return rs.noncentral_f(sh[0], sh[1], sh[2], size)
    @staticmethod
    def pdf(x: float, sh) -> float: return _ncf_pdf(x, sh[0], sh[1], sh[2])
    @staticmethod
    def cdf(x: float, sh) -> float: return NC.ncfdtr(sh[0], sh[1], sh[2], x)
    @staticmethod
    def ppf(q: float, sh) -> float: return NC.ncfdtri(sh[0], sh[1], sh[2], q)
    @staticmethod
    def sf(x: float, sh) -> float: return NC._ncf_cdf(x, sh[0], sh[1], sh[2], True)
    @staticmethod
    def stats(sh):
        n = sh[0]
        m = sh[1]
        l = sh[2]
        mean = m * (n + l) / (n * (m - 2)) if m > 2 else NAN
        var = NAN
        if m > 4:
            var = 2 * m * m * ((n + l) * (n + l) + (m - 2) * (n + 2 * l))
            var /= (m - 4) * (m - 2) * (m - 2) * n * n
        sk = NAN
        if m > 6:
            sk = 2 * K.sqrt(2.0) * K.sqrt(m - 4)
            sk *= (n * (m + n - 2) * (m + 2 * n - 2) + 3 * (m + n - 2) * (m + 2 * n - 2) * l
                   + 6 * (m + n - 2) * l * l + 2 * l * l * l)
            sk /= (m - 6) * K.pow(n * (m + n - 2) + 2 * (m + n - 2) * l + l * l, 1.5)
        ku = NAN
        if m > 8:
            l2 = l * l
            l3 = l2 * l
            l4 = l2 * l2
            ku = (3 * (m - 4) * (n * (m + n - 2) * (4 * (m - 2) * (m - 2) + (m - 2) * (m + 10) * n
                                                    + (10 + m) * n * n)
                                 + 4 * (m + n - 2) * (4 * (m - 2) * (m - 2) + (m - 2) * (10 + m) * n
                                                      + (10 + m) * n * n) * l
                                 + 2 * (10 + m) * (m + n - 2) * (2 * m + 3 * n - 4) * l2
                                 + 4 * (10 + m) * (-2 + m + n) * l3 + (10 + m) * l4)
                  / ((-8 + m) * (-6 + m) * (n * (-2 + m + n) + 2 * (-2 + m + n) * l + l2) ** 2))
            # boost kurtosis_excess; scipy's _ncf_kurtosis_excess is the full
            # kurtosis (excess + 3) and ncf._stats subtracts 3 -> same value
            ku = ku - 3
        return (mean, var, sk, ku)

# -------------------------------------------------------------------- nct

def _nct_mean(v: float, delta: float) -> float:
    if v == INF or v > 1 / 2.220446049250313e-16:
        return delta
    return delta * K.sqrt(v / 2) * K._tgamma_delta_ratio_half((v - 1) * 0.5)

def _nct_pdf(x: float, v: float, mu: float) -> float:
    # scipy _nct_pdf (scipy/special/boost_special_functions.h): boost
    # non_central_t pdf
    if v == INF:
        return _norm_pdf(x - mu)
    if x * x < 2.220446049250313e-16:
        return (K.exp(-K._log_gamma_ratio(v / 2 + 0.5, 0.5)) * K.sqrt(v / _PI)
                * K.exp(-mu * mu / 2) / 2)
    if fabs(mu / (4 * v)) < 2.220446049250313e-16:
        return _TK.pdf(x - mu, (v,))
    a = (v + 1) / 2
    z = mu * mu * x * x / (2 * (x * x + v))
    summit = (K.sqrt(z * (4 * a + z)) + z) / 2
    if summit < 40:
        s1, sc1 = H._1f1_generic_series((v + 1) / 2, 0.5, z)
        Av = s1 * K.exp(float(sc1))
        s2, sc2 = H._1f1_generic_series(v / 2 + 1.0, 1.5, z)
        Bv = s2 * K.exp(float(sc2))
        Bv *= K.exp(-K._log_gamma_ratio(v / 2 + 1.0, -0.5))
        Bv *= K.sqrt(2.0) * mu * x / K.sqrt(x * x + v)
        tol = 1.4901161193847656e-08 * Av * 4
        Av += Bv
        if not (Av < tol):
            Av *= (K.exp(-mu * mu / 2) * K.pow(1 + x * x / v, -(v + 1) / 2)
                   * K.exp(-K._log_gamma_ratio(v / 2 + 0.5, -0.5)))
            Av /= K.sqrt(v) * _SQRT_PI
            return Av
    from scipy.integrate import quad
    return _nct_pdf_integral(x, v, mu)

class _NctInteg:
    x: float
    v: float
    mu: float
    def __init__(self, x: float, v: float, mu: float):
        self.x = x
        self.v = v
        self.mu = mu
    def __call__(self, y: float) -> float:
        x = self.x
        v = self.v
        c = (y - self.mu * x / K.sqrt(x * x + v)) ** 2
        if v * K.log(y) < 709.782712893384:
            return K.pow(y, v) * K.exp(c / -2)
        return K.exp(K.log(y) * v + c / -2)

def _nct_pdf_integral(x: float, v: float, mu: float) -> float:
    from scipy.integrate import quad
    integral = K.pow(v, v / 2) * K.exp(-v * mu * mu / (2 * (x * x + v)))
    if integral != 0:
        integral *= quad(_NctInteg(x, v, mu), 0.0, INF, epsabs=0.0, epsrel=1e-13)[0]
    integral /= (_SQRT_PI * K.gamma(v / 2) * K.pow(2.0, (v - 1) / 2)
                 * K.pow(x * x + v, (v + 1) / 2))
    return integral

@tuple
class _NctK:
    @staticmethod
    def nshape() -> Literal[int]: return 2
    @staticmethod
    def check(sh) -> bool: return sh[0] > 0 and sh[1] == sh[1]
    @staticmethod
    def rvs(rs, sh, size):
        n = rs.normal(loc=sh[1], size=size)
        c2 = rs.chisquare(sh[0], size)
        return n * K.sqrt(sh[0]) / np.sqrt(c2)
    @staticmethod
    def pdf(x: float, sh) -> float: return _nct_pdf(x, sh[0], sh[1])
    @staticmethod
    def cdf(x: float, sh) -> float: return NC.nctdtr(sh[0], sh[1], x)
    @staticmethod
    def ppf(q: float, sh) -> float: return NC.nctdtrit(sh[0], sh[1], q)
    @staticmethod
    def sf(x: float, sh) -> float:
        return min(max(NC._nct_cdf(sh[0], sh[1], x, True), 0.0), 1.0)
    @staticmethod
    def stats(sh):
        v = sh[0]
        delta = sh[1]
        mean = _nct_mean(v, delta) if v > 1 else NAN
        var = NAN
        sk = NAN
        ku = NAN
        if v > 2:
            if v == INF:
                var = 1.0
            elif delta == 0:
                var = v / (v - 2)
            else:
                var = ((delta * delta + 1) * v) / (v - 2) - mean * mean
        if v > 3:
            if v == INF or delta == 0:
                sk = 0.0
            else:
                l2 = delta * delta
                vv = ((l2 + 1) * v) / (v - 2) - mean * mean
                r = -2 * vv
                r += v * (l2 + 2 * v - 3) / ((v - 3) * (v - 2))
                r *= mean
                sk = r / K.pow(vv, 1.5)
        if v > 4:
            if v == INF or delta == 0:
                ku = 1.0
            else:
                l2 = delta * delta
                vv = ((l2 + 1) * v) / (v - 2) - mean * mean
                r = -3 * vv
                r += v * (l2 * (v + 1) + 3 * (3 * v - 5)) / ((v - 3) * (v - 2))
                r *= -mean * mean
                r += v * v * (l2 * l2 + 6 * l2 + 3) / ((v - 4) * (v - 2))
                r /= vv * vv
                ku = r - 3
        return (mean, var, sk, ku)

# ---------------------------------------------------------------- pearson3

@tuple
class _Pearson3K:
    @staticmethod
    def nshape() -> Literal[int]: return 1
    @staticmethod
    def check(sh) -> bool: return sh[0] == sh[0] and fabs(sh[0]) != INF
    @staticmethod
    def stats(sh): return (0.0, 1.0, sh[0], 1.5 * sh[0] ** 2)
    @staticmethod
    def logpdf(x: float, sh) -> float:
        skew = sh[0]
        if fabs(skew) < 0.000016:
            return K.log(_norm_pdf(x))
        beta = 2.0 / skew
        alpha = beta * beta
        zeta = -alpha / beta
        transx = beta * (x - zeta)
        if not (transx > 0):
            return -INF if transx == transx else NAN
        return K.log(fabs(beta)) + _GammaK.logpdf(transx, (alpha,))
    @staticmethod
    def pdf(x: float, sh) -> float:
        r = K.exp(_Pearson3K.logpdf(x, sh))
        return 0.0 if r != r else r
    @staticmethod
    def cdf(x: float, sh) -> float:
        skew = sh[0]
        if fabs(skew) < 0.000016:
            return _norm_cdf(x)
        beta = 2.0 / skew
        alpha = beta * beta
        zeta = -alpha / beta
        transx = beta * (x - zeta)
        if skew > 0:
            return 0.0 if transx <= 0 else _GammaK.cdf(transx, (alpha,))
        return 1.0 if transx <= 0 else _GammaK.sf(transx, (alpha,))
    @staticmethod
    def sf(x: float, sh) -> float:
        skew = sh[0]
        if fabs(skew) < 0.000016:
            return _norm_sf(x)
        beta = 2.0 / skew
        alpha = beta * beta
        zeta = -alpha / beta
        transx = beta * (x - zeta)
        if skew > 0:
            return 1.0 if transx <= 0 else _GammaK.sf(transx, (alpha,))
        return 0.0 if transx <= 0 else _GammaK.cdf(transx, (alpha,))
    @staticmethod
    def ppf(q: float, sh) -> float:
        skew = sh[0]
        if fabs(skew) < 0.000016:
            return _norm_ppf(q)
        beta = 2.0 / skew
        alpha = beta * beta
        zeta = -alpha / beta
        if beta < 0:
            q = 1 - q
        return K.gammaincinv(alpha, q) / beta + zeta
    @staticmethod
    def rvs(rs, sh, size):
        skew = sh[0]
        if fabs(skew) < 0.000016:
            return rs.standard_normal(size)
        beta = 2.0 / skew
        alpha = beta * beta
        zeta = -alpha / beta
        return rs.standard_gamma(alpha, size) / beta + zeta

erlang = rv_continuous[_ErlangK]('erlang')
reciprocal = rv_continuous[_ReciprocalK]('reciprocal')
loguniform = rv_continuous[_ReciprocalK]('loguniform')
johnsonsb = rv_continuous[_JohnsonsbK]('johnsonsb')
johnsonsu = rv_continuous[_JohnsonsuK]('johnsonsu')
skewnorm = rv_continuous[_SkewnormK]('skewnorm')
exponnorm = rv_continuous[_ExponnormK]('exponnorm')
gennorm = rv_continuous[_GennormK]('gennorm')
foldnorm = rv_continuous[_FoldnormK]('foldnorm')
foldcauchy = rv_continuous[_FoldcauchyK]('foldcauchy')
nakagami = rv_continuous[_NakagamiK]('nakagami')
loggamma = rv_continuous[_LoggammaK]('loggamma')
alpha = rv_continuous[_AlphaK]('alpha')
dgamma = rv_continuous[_DgammaK]('dgamma')
betaprime = rv_continuous[_BetaprimeK]('betaprime')
recipinvgauss = rv_continuous[_RecipinvgaussK]('recipinvgauss')
fatiguelife = rv_continuous[_FatiguelifeK]('fatiguelife')
wrapcauchy = rv_continuous[_WrapcauchyK]('wrapcauchy')
cosine = rv_continuous[_CosineK]('cosine')
vonmises = rv_continuous[_VonmisesK]('vonmises')
truncnorm = rv_continuous[_TruncnormK]('truncnorm')
ncx2 = rv_continuous[_Ncx2K]('ncx2')
ncf = rv_continuous[_NcfK]('ncf')
nct = rv_continuous[_NctK]('nct')
pearson3 = rv_continuous[_Pearson3K]('pearson3')
