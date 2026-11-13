# Copyright (C) 2026 the Typhon authors
# Ported from scipy 1.18.1 (BSD-3-Clause)

# Continuous distribution kernels, part 3 (scipy.stats._continuous_distns):
# halflogistic, genhalflogistic, halfgennorm, invweibull, exponpow, powernorm,
# powerlognorm, laplace_asymmetric, trapezoid, truncweibull_min, truncpareto,
# gengamma, genexpon, kappa4, crystalball, argus, tukeylambda, kstwobign,
# rel_breitwigner. See _distn_infrastructure for the kernel protocol.

import numpy as np
import scipy.special._kernels as K
import scipy.special._gamma2 as G2
import scipy.special._misc as M
import scipy.special._kolmogorov as KS
import scipy.special._lambertw as LW
import scipy.special._legendre as LG
from ._distn_infrastructure import rv_continuous, NAN, INF, _NA, _generic_munp
from ._continuous_distns import _norm_pdf, _norm_logpdf, _norm_cdf, _norm_sf, \
    _norm_logcdf, _norm_ppf, _PI, _EULER, _ZETA3

from C import tanh(float) -> float
from C import atanh(float) -> float
from C import atan(float) -> float
from C import fabs(float) -> float

_NORM_PDF_C = 2.5066282746310002
_NORM_PDF_LOGC = 0.9189385332046727

# ---------------------------------------------------------- halflogistic

@tuple
class _HalflogisticK:
    @staticmethod
    def nshape() -> Literal[int]: return 0
    @staticmethod
    def a(sh) -> float: return 0.0
    @staticmethod
    def logpdf(x: float, sh) -> float: return K.log(2.0) - x - 2. * K.log1p(K.exp(-x))
    @staticmethod
    def pdf(x: float, sh) -> float: return K.exp(_HalflogisticK.logpdf(x, sh))
    @staticmethod
    def cdf(x: float, sh) -> float: return tanh(x / 2.0)
    @staticmethod
    def ppf(q: float, sh) -> float: return 2 * atanh(q)
    @staticmethod
    def sf(x: float, sh) -> float: return 2 * K.expit(-x)
    @staticmethod
    def isf(q: float, sh) -> float:
        if q < 0.5:
            return -K.logit(0.5 * q)
        return 2 * atanh(1 - q)
    @staticmethod
    def munp(n: float, sh) -> float:
        if n == 0:
            return 1.0
        if n == 1:
            return 2 * K.log(2.0)
        if n == 2:
            return _PI * _PI / 3.0
        if n == 3:
            return 9 * _ZETA3
        if n == 4:
            return 7 * _PI ** 4 / 15.0
        return 2 * (1 - K.pow(2.0, 1 - n)) * K.gamma(n + 1) * G2.zeta(n, 1.0)
    @staticmethod
    def entropy(sh) -> float: return 2 - K.log(2.0)

@tuple
class _GenhalflogisticK:
    @staticmethod
    def nshape() -> Literal[int]: return 1
    @staticmethod
    def a(sh) -> float: return 0.0
    @staticmethod
    def b(sh) -> float: return 1.0 / sh[0]
    @staticmethod
    def pdf(x: float, sh) -> float:
        c = sh[0]
        limit = 1.0 / c
        tmp = 1 - c * x
        tmp0 = K.pow(tmp, limit - 1)
        tmp2 = tmp0 * tmp
        return 2 * tmp0 / (1 + tmp2) ** 2
    @staticmethod
    def cdf(x: float, sh) -> float:
        c = sh[0]
        tmp2 = K.pow(1 - c * x, 1.0 / c)
        return (1.0 - tmp2) / (1 + tmp2)
    @staticmethod
    def ppf(q: float, sh) -> float:
        c = sh[0]
        return 1.0 / c * (1 - K.pow((1.0 - q) / (1.0 + q), c))
    @staticmethod
    def entropy(sh) -> float: return 2 - (2 * sh[0] + 1) * K.log(2.0)

@tuple
class _HalfgennormK:
    @staticmethod
    def nshape() -> Literal[int]: return 1
    @staticmethod
    def a(sh) -> float: return 0.0
    @staticmethod
    def logpdf(x: float, sh) -> float:
        beta = sh[0]
        return K.log(beta) - K.gammaln(1.0 / beta) - K.pow(x, beta)
    @staticmethod
    def pdf(x: float, sh) -> float: return K.exp(_HalfgennormK.logpdf(x, sh))
    @staticmethod
    def cdf(x: float, sh) -> float: return K.gammainc(1.0 / sh[0], K.pow(x, sh[0]))
    @staticmethod
    def ppf(x: float, sh) -> float: return K.pow(K.gammaincinv(1.0 / sh[0], x), 1.0 / sh[0])
    @staticmethod
    def sf(x: float, sh) -> float: return K.gammaincc(1.0 / sh[0], K.pow(x, sh[0]))
    @staticmethod
    def isf(x: float, sh) -> float: return K.pow(K.gammainccinv(1.0 / sh[0], x), 1.0 / sh[0])
    @staticmethod
    def munp(n: float, sh) -> float: return G2.poch(1 / sh[0], n / sh[0])
    @staticmethod
    def entropy(sh) -> float:
        beta = sh[0]
        return 1.0 / beta - K.log(beta) + K.gammaln(1.0 / beta)

# -------------------------------------------------- invweibull / exponpow

@tuple
class _InvweibullK:
    @staticmethod
    def nshape() -> Literal[int]: return 1
    @staticmethod
    def a(sh) -> float: return 0.0
    @staticmethod
    def open_support() -> Literal[bool]: return True
    @staticmethod
    def pdf(x: float, sh) -> float:
        c = sh[0]
        return c * K.pow(x, -c - 1.0) * K.exp(-K.pow(x, -c))
    @staticmethod
    def cdf(x: float, sh) -> float: return K.exp(-K.pow(x, -sh[0]))
    @staticmethod
    def sf(x: float, sh) -> float: return -K.expm1(-K.pow(x, -sh[0]))
    @staticmethod
    def ppf(q: float, sh) -> float: return K.pow(-K.log(q), -1.0 / sh[0])
    @staticmethod
    def isf(p: float, sh) -> float: return K.pow(-K.log1p(-p), -1 / sh[0])
    @staticmethod
    def munp(n: float, sh) -> float: return K.gamma(1 - n / sh[0])
    @staticmethod
    def entropy(sh) -> float:
        c = sh[0]
        return 1 + _EULER + _EULER / c - K.log(c)
    @staticmethod
    def fitstart(data: np.ndarray[float, 1]) -> List[float]: return [2.0]

@tuple
class _ExponpowK:
    @staticmethod
    def nshape() -> Literal[int]: return 1
    @staticmethod
    def a(sh) -> float: return 0.0
    @staticmethod
    def logpdf(x: float, sh) -> float:
        b = sh[0]
        xb = K.pow(x, b)
        return 1 + K.log(b) + K.xlogy(b - 1.0, x) + xb - K.exp(xb)
    @staticmethod
    def pdf(x: float, sh) -> float: return K.exp(_ExponpowK.logpdf(x, sh))
    @staticmethod
    def cdf(x: float, sh) -> float: return -K.expm1(-K.expm1(K.pow(x, sh[0])))
    @staticmethod
    def sf(x: float, sh) -> float: return K.exp(-K.expm1(K.pow(x, sh[0])))
    @staticmethod
    def isf(x: float, sh) -> float: return K.pow(K.log1p(-K.log(x)), 1. / sh[0])
    @staticmethod
    def ppf(q: float, sh) -> float: return K.pow(K.log1p(-K.log1p(-q)), 1.0 / sh[0])

# ------------------------------------------------ powernorm / powerlognorm

@tuple
class _PowernormK:
    @staticmethod
    def nshape() -> Literal[int]: return 1
    @staticmethod
    def pdf(x: float, sh) -> float:
        c = sh[0]
        return c * _norm_pdf(x) * K.pow(_norm_cdf(-x), c - 1.0)
    @staticmethod
    def logpdf(x: float, sh) -> float:
        c = sh[0]
        return K.log(c) + _norm_logpdf(x) + (c - 1) * _norm_logcdf(-x)
    @staticmethod
    def logsf(x: float, sh) -> float: return sh[0] * _norm_logcdf(-x)
    @staticmethod
    def cdf(x: float, sh) -> float: return -K.expm1(_PowernormK.logsf(x, sh))
    @staticmethod
    def sf(x: float, sh) -> float: return K.exp(_PowernormK.logsf(x, sh))
    @staticmethod
    def ppf(q: float, sh) -> float: return -_norm_ppf(K.pow(1.0 - q, 1.0 / sh[0]))
    @staticmethod
    def isf(q: float, sh) -> float: return -_norm_ppf(K.exp(K.log(q) / sh[0]))

@tuple
class _PowerlognormK:
    @staticmethod
    def nshape() -> Literal[int]: return 2
    @staticmethod
    def a(sh) -> float: return 0.0
    @staticmethod
    def open_support() -> Literal[bool]: return True
    @staticmethod
    def logpdf(x: float, sh) -> float:
        c = sh[0]
        s = sh[1]
        return (K.log(c) - K.log(x) - K.log(s) + _norm_logpdf(K.log(x) / s)
                + _norm_logcdf(-K.log(x) / s) * (c - 1.))
    @staticmethod
    def pdf(x: float, sh) -> float: return K.exp(_PowerlognormK.logpdf(x, sh))
    @staticmethod
    def logsf(x: float, sh) -> float: return _norm_logcdf(-K.log(x) / sh[1]) * sh[0]
    @staticmethod
    def cdf(x: float, sh) -> float: return -K.expm1(_PowerlognormK.logsf(x, sh))
    @staticmethod
    def sf(x: float, sh) -> float: return K.exp(_PowerlognormK.logsf(x, sh))
    @staticmethod
    def isf(q: float, sh) -> float:
        return K.exp(-_norm_ppf(K.pow(q, 1 / sh[0])) * sh[1])
    @staticmethod
    def ppf(q: float, sh) -> float: return _PowerlognormK.isf(1 - q, sh)

# --------------------------------------------------- laplace_asymmetric

@tuple
class _LaplaceAsymK:
    @staticmethod
    def nshape() -> Literal[int]: return 1
    @staticmethod
    def logpdf(x: float, sh) -> float:
        kappa = sh[0]
        kapinv = 1 / kappa
        l = x * (-kappa if x >= 0 else kapinv)
        return l - K.log(kappa + kapinv)
    @staticmethod
    def pdf(x: float, sh) -> float: return K.exp(_LaplaceAsymK.logpdf(x, sh))
    @staticmethod
    def cdf(x: float, sh) -> float:
        kappa = sh[0]
        kapinv = 1 / kappa
        kk = kappa + kapinv
        if x >= 0:
            return 1 - K.exp(-x * kappa) * (kapinv / kk)
        return K.exp(x * kapinv) * (kappa / kk)
    @staticmethod
    def sf(x: float, sh) -> float:
        kappa = sh[0]
        kapinv = 1 / kappa
        kk = kappa + kapinv
        if x >= 0:
            return K.exp(-x * kappa) * (kapinv / kk)
        return 1 - K.exp(x * kapinv) * (kappa / kk)
    @staticmethod
    def ppf(q: float, sh) -> float:
        kappa = sh[0]
        kapinv = 1 / kappa
        kk = kappa + kapinv
        if q >= kappa / kk:
            return -K.log((1 - q) * kk * kappa) * kapinv
        return K.log(q * kk / kappa) * kappa
    @staticmethod
    def isf(q: float, sh) -> float:
        kappa = sh[0]
        kapinv = 1 / kappa
        kk = kappa + kapinv
        if q <= kapinv / kk:
            return -K.log(q * kk * kappa) * kapinv
        return K.log((1 - q) * kk / kappa) * kappa
    @staticmethod
    def stats(sh):
        kappa = sh[0]
        kapinv = 1 / kappa
        return (kapinv - kappa, kapinv * kapinv + kappa * kappa,
                2.0 * (1 - K.pow(kappa, 6.0)) / K.pow(1 + K.pow(kappa, 4.0), 1.5),
                6.0 * (1 + K.pow(kappa, 8.0)) / K.pow(1 + K.pow(kappa, 4.0), 2.0))
    @staticmethod
    def entropy(sh) -> float: return 1 + K.log(sh[0] + 1 / sh[0])

# --------------------------------------------------------------- trapezoid

@tuple
class _TrapezoidK:
    @staticmethod
    def nshape() -> Literal[int]: return 2
    @staticmethod
    def check(sh) -> bool:
        c = sh[0]
        d = sh[1]
        return c >= 0 and c <= 1 and d >= 0 and d <= 1 and d >= c
    @staticmethod
    def a(sh) -> float: return 0.0
    @staticmethod
    def b(sh) -> float: return 1.0
    @staticmethod
    def pdf(x: float, sh) -> float:
        c = sh[0]
        d = sh[1]
        u = 2 / (d - c + 1)
        if x < c:
            return u * x / c
        if x <= d:
            return u
        return u * (1 - x) / (1 - d)
    @staticmethod
    def cdf(x: float, sh) -> float:
        c = sh[0]
        d = sh[1]
        if x < c:
            return x ** 2 / c / (d - c + 1)
        if x <= d:
            return (c + 2 * (x - c)) / (d - c + 1)
        return 1 - ((1 - x) ** 2 / (d - c + 1) / (1 - d))
    @staticmethod
    def ppf(q: float, sh) -> float:
        c = sh[0]
        d = sh[1]
        qc = _TrapezoidK.cdf(c, sh)
        qd = _TrapezoidK.cdf(d, sh)
        if q < qc:
            return K.sqrt(q * c * (1 + d - c))
        if q <= qd:
            return 0.5 * q * (1 + d - c) + 0.5 * c
        return 1 - K.sqrt((1 - q) * (d - c + 1) * (1 - d))
    @staticmethod
    def munp(n: float, sh) -> float:
        c = sh[0]
        d = sh[1]
        ab_term = K.pow(c, n + 1)
        dc_term = 1.0
        if d == 0.0:
            dc_term = 1.0
        elif d < 1.0:
            dc_term = K.expm1((n + 2) * K.log(d)) / (d - 1.0)
        else:
            dc_term = n + 2
        return 2.0 / (1.0 + d - c) * (dc_term - ab_term) / ((n + 1) * (n + 2))
    @staticmethod
    def entropy(sh) -> float:
        c = sh[0]
        d = sh[1]
        return 0.5 * (1.0 - d + c) / (1.0 + d - c) + K.log(0.5 * (1.0 + d - c))
    @staticmethod
    def fitstart(data: np.ndarray[float, 1]) -> List[float]: return [0.33, 0.66]

# -------------------------------------------------------- truncweibull_min

@tuple
class _TruncweibullMinK:
    @staticmethod
    def nshape() -> Literal[int]: return 3
    @staticmethod
    def check(sh) -> bool: return sh[1] >= 0. and sh[2] > sh[1] and sh[0] > 0.
    @staticmethod
    def a(sh) -> float: return sh[1]
    @staticmethod
    def b(sh) -> float: return sh[2]
    @staticmethod
    def pdf(x: float, sh) -> float:
        c = sh[0]
        a = sh[1]
        b = sh[2]
        den = K.exp(-K.pow(a, c)) - K.exp(-K.pow(b, c))
        return (c * K.pow(x, c - 1) * K.exp(-K.pow(x, c))) / den
    @staticmethod
    def logpdf(x: float, sh) -> float:
        c = sh[0]
        a = sh[1]
        b = sh[2]
        logden = K.log(K.exp(-K.pow(a, c)) - K.exp(-K.pow(b, c)))
        return K.log(c) + K.xlogy(c - 1, x) - K.pow(x, c) - logden
    @staticmethod
    def cdf(x: float, sh) -> float:
        c = sh[0]
        a = sh[1]
        b = sh[2]
        return ((K.exp(-K.pow(a, c)) - K.exp(-K.pow(x, c)))
                / (K.exp(-K.pow(a, c)) - K.exp(-K.pow(b, c))))
    @staticmethod
    def logcdf(x: float, sh) -> float:
        c = sh[0]
        a = sh[1]
        b = sh[2]
        return (K.log(K.exp(-K.pow(a, c)) - K.exp(-K.pow(x, c)))
                - K.log(K.exp(-K.pow(a, c)) - K.exp(-K.pow(b, c))))
    @staticmethod
    def sf(x: float, sh) -> float:
        c = sh[0]
        a = sh[1]
        b = sh[2]
        return ((K.exp(-K.pow(x, c)) - K.exp(-K.pow(b, c)))
                / (K.exp(-K.pow(a, c)) - K.exp(-K.pow(b, c))))
    @staticmethod
    def logsf(x: float, sh) -> float:
        c = sh[0]
        a = sh[1]
        b = sh[2]
        return (K.log(K.exp(-K.pow(x, c)) - K.exp(-K.pow(b, c)))
                - K.log(K.exp(-K.pow(a, c)) - K.exp(-K.pow(b, c))))
    @staticmethod
    def isf(q: float, sh) -> float:
        c = sh[0]
        a = sh[1]
        b = sh[2]
        return K.pow(-K.log((1 - q) * K.exp(-K.pow(b, c)) + q * K.exp(-K.pow(a, c))), 1 / c)
    @staticmethod
    def ppf(q: float, sh) -> float:
        c = sh[0]
        a = sh[1]
        b = sh[2]
        return K.pow(-K.log((1 - q) * K.exp(-K.pow(a, c)) + q * K.exp(-K.pow(b, c))), 1 / c)
    @staticmethod
    def munp(n: float, sh) -> float:
        c = sh[0]
        a = sh[1]
        b = sh[2]
        g = K.gamma(n / c + 1.) * (K.gammainc(n / c + 1., K.pow(b, c))
                                   - K.gammainc(n / c + 1., K.pow(a, c)))
        return g / (K.exp(-K.pow(a, c)) - K.exp(-K.pow(b, c)))
    @staticmethod
    def fitstart(data: np.ndarray[float, 1]) -> List[float]: return [1.0, 0.0, 1.0]

# ------------------------------------------------------------- truncpareto

@tuple
class _TruncparetoK:
    @staticmethod
    def nshape() -> Literal[int]: return 2
    @staticmethod
    def check(sh) -> bool: return sh[0] != 0. and sh[1] > 1.
    @staticmethod
    def a(sh) -> float: return 1.0
    @staticmethod
    def b(sh) -> float: return sh[1]
    @staticmethod
    def pdf(x: float, sh) -> float:
        b = sh[0]
        c = sh[1]
        return b * K.pow(x, -(b + 1)) / (1 - 1 / K.pow(c, b))
    @staticmethod
    def logpdf(x: float, sh) -> float:
        b = sh[0]
        c = sh[1]
        if b > 0:
            return K.log(b) - K.log(-K.expm1(-b * K.log(c))) - (b + 1) * K.log(x)
        return K.log(_TruncparetoK.pdf(x, sh))
    @staticmethod
    def cdf(x: float, sh) -> float:
        b = sh[0]
        c = sh[1]
        return (1 - K.pow(x, -b)) / (1 - 1 / K.pow(c, b))
    @staticmethod
    def logcdf(x: float, sh) -> float:
        b = sh[0]
        c = sh[1]
        if b > 0:
            return K.log1p(-K.pow(x, -b)) - K.log1p(-1 / K.pow(c, b))
        return K.log(_TruncparetoK.cdf(x, sh))
    @staticmethod
    def ppf(q: float, sh) -> float:
        b = sh[0]
        c = sh[1]
        return K.pow(1 - (1 - 1 / K.pow(c, b)) * q, -1 / b)
    @staticmethod
    def sf(x: float, sh) -> float:
        b = sh[0]
        c = sh[1]
        return (K.pow(x, -b) - 1 / K.pow(c, b)) / (1 - 1 / K.pow(c, b))
    @staticmethod
    def logsf(x: float, sh) -> float:
        b = sh[0]
        c = sh[1]
        if b > 0:
            return K.log(K.pow(x, -b) - 1 / K.pow(c, b)) - K.log1p(-1 / K.pow(c, b))
        return K.log(_TruncparetoK.sf(x, sh))
    @staticmethod
    def isf(q: float, sh) -> float:
        b = sh[0]
        c = sh[1]
        return K.pow(1 / K.pow(c, b) + (1 - 1 / K.pow(c, b)) * q, -1 / b)
    @staticmethod
    def entropy(sh) -> float:
        b = sh[0]
        c = sh[1]
        return -(K.log(b / (1 - 1 / K.pow(c, b))) + (b + 1) * (K.log(c) / (K.pow(c, b) - 1) - 1 / b))
    @staticmethod
    def munp(n: float, sh) -> float:
        b = sh[0]
        c = sh[1]
        if n == b:
            return b * K.log(c) / (1 - 1 / K.pow(c, b))
        return b / (b - n) * (K.pow(c, b) - K.pow(c, n)) / (K.pow(c, b) - 1)

# ---------------------------------------------------------------- gengamma

@tuple
class _GengammaK:
    @staticmethod
    def nshape() -> Literal[int]: return 2
    @staticmethod
    def check(sh) -> bool: return sh[0] > 0 and sh[1] != 0
    @staticmethod
    def a(sh) -> float: return 0.0
    @staticmethod
    def rvs(rs, sh, size): return rs.standard_gamma(sh[0], size=size) ** (1. / sh[1])
    @staticmethod
    def logpdf(x: float, sh) -> float:
        a = sh[0]
        c = sh[1]
        if x != 0 or c > 0:
            return K.log(fabs(c)) + K.xlogy(c * a - 1, x) - K.pow(x, c) - K.gammaln(a)
        return -INF
    @staticmethod
    def pdf(x: float, sh) -> float: return K.exp(_GengammaK.logpdf(x, sh))
    @staticmethod
    def cdf(x: float, sh) -> float:
        a = sh[0]
        c = sh[1]
        xc = K.pow(x, c)
        return K.gammainc(a, xc) if c > 0 else K.gammaincc(a, xc)
    @staticmethod
    def sf(x: float, sh) -> float:
        a = sh[0]
        c = sh[1]
        xc = K.pow(x, c)
        return K.gammaincc(a, xc) if c > 0 else K.gammainc(a, xc)
    @staticmethod
    def ppf(q: float, sh) -> float:
        a = sh[0]
        c = sh[1]
        v = K.gammaincinv(a, q) if c > 0 else K.gammainccinv(a, q)
        return K.pow(v, 1.0 / c)
    @staticmethod
    def isf(q: float, sh) -> float:
        a = sh[0]
        c = sh[1]
        v = K.gammainccinv(a, q) if c > 0 else K.gammaincinv(a, q)
        return K.pow(v, 1.0 / c)
    @staticmethod
    def munp(n: float, sh) -> float: return G2.poch(sh[0], n * 1.0 / sh[1])
    @staticmethod
    def entropy(sh) -> float:
        a = sh[0]
        c = sh[1]
        if a >= 200:
            return ((0.5 * K.log(2 * _PI) + 0.5) - K.log(a) / 2 - K.log(fabs(c)) + (a ** -1.) / 6
                    - (a ** -3.) / 90 + (K.log(a) - (a ** -1.) / 2 - (a ** -2.) / 12
                                        + (a ** -4.) / 120) / c)
        val = K.digamma(a)
        return a * (1 - val) + val / c + K.gammaln(a) - K.log(fabs(c))

# ----------------------------------------------------------------- genexpon

@tuple
class _GenexponK:
    @staticmethod
    def nshape() -> Literal[int]: return 3
    @staticmethod
    def a(sh) -> float: return 0.0
    @staticmethod
    def pdf(x: float, sh) -> float:
        a = sh[0]
        b = sh[1]
        c = sh[2]
        return (a + b * (-K.expm1(-c * x))) * K.exp((-a - b) * x + b * (-K.expm1(-c * x)) / c)
    @staticmethod
    def logpdf(x: float, sh) -> float:
        a = sh[0]
        b = sh[1]
        c = sh[2]
        return K.log(a + b * (-K.expm1(-c * x))) + (-a - b) * x + b * (-K.expm1(-c * x)) / c
    @staticmethod
    def cdf(x: float, sh) -> float:
        a = sh[0]
        b = sh[1]
        c = sh[2]
        return -K.expm1((-a - b) * x + b * (-K.expm1(-c * x)) / c)
    @staticmethod
    def sf(x: float, sh) -> float:
        a = sh[0]
        b = sh[1]
        c = sh[2]
        return K.exp((-a - b) * x + b * (-K.expm1(-c * x)) / c)
    @staticmethod
    def ppf(p: float, sh) -> float:
        a = sh[0]
        b = sh[1]
        c = sh[2]
        s = a + b
        t = (b - c * K.log1p(-p)) / s
        return (t + LW.lambertw(complex(-b / s * K.exp(-t)), 0, 1e-8).real) / c
    @staticmethod
    def isf(p: float, sh) -> float:
        a = sh[0]
        b = sh[1]
        c = sh[2]
        s = a + b
        t = (b - c * K.log(p)) / s
        return (t + LW.lambertw(complex(-b / s * K.exp(-t)), 0, 1e-8).real) / c

# ------------------------------------------------------------------ kappa4

@tuple
class _Kappa4K:
    @staticmethod
    def nshape() -> Literal[int]: return 2
    @staticmethod
    def check(sh) -> bool: return True
    @staticmethod
    def a(sh) -> float:
        h = sh[0]
        k = sh[1]
        if h > 0:
            if k == 0:
                return K.log(h)
            return (1.0 - K.pow(h, -k)) / k
        if k < 0:
            return 1.0 / k
        return -INF
    @staticmethod
    def b(sh) -> float:
        k = sh[1]
        return 1.0 / k if k > 0 else INF
    @staticmethod
    def logpdf(x: float, sh) -> float:
        h = sh[0]
        k = sh[1]
        if h != 0 and k != 0:
            return (K.xlog1py(1.0 / k - 1.0, -k * x)
                    + K.xlog1py(1.0 / h - 1.0, -h * K.pow(1.0 - k * x, 1.0 / k)))
        if h == 0 and k != 0:
            return K.xlog1py(1.0 / k - 1.0, -k * x) - K.pow(1.0 - k * x, 1.0 / k)
        if h != 0 and k == 0:
            return -x + K.xlog1py(1.0 / h - 1.0, -h * K.exp(-x))
        return -x - K.exp(-x)
    @staticmethod
    def pdf(x: float, sh) -> float: return K.exp(_Kappa4K.logpdf(x, sh))
    @staticmethod
    def logcdf(x: float, sh) -> float:
        h = sh[0]
        k = sh[1]
        if h != 0 and k != 0:
            return (1.0 / h) * K.log1p(-h * K.pow(1.0 - k * x, 1.0 / k))
        if h == 0 and k != 0:
            return -K.pow(1.0 - k * x, 1.0 / k)
        if h != 0 and k == 0:
            return (1.0 / h) * K.log1p(-h * K.exp(-x))
        return -K.exp(-x)
    @staticmethod
    def cdf(x: float, sh) -> float: return K.exp(_Kappa4K.logcdf(x, sh))
    @staticmethod
    def ppf(q: float, sh) -> float:
        h = sh[0]
        k = sh[1]
        if h != 0 and k != 0:
            return 1.0 / k * (1.0 - K.pow((1.0 - K.pow(q, h)) / h, k))
        if h == 0 and k != 0:
            return 1.0 / k * (1.0 - K.pow(-K.log(q), k))
        if h != 0 and k == 0:
            return -K.log1p(-K.pow(q, h)) + K.log(h)
        return -K.log(-K.log(q))
    @staticmethod
    def stats(sh):
        h = sh[0]
        k = sh[1]
        maxr = 5
        if h < 0 and k >= 0:
            maxr = int(-1.0 / h * k)
        elif k < 0:
            maxr = int(-1.0 / k)
        return (_NA if 1 < maxr else NAN, _NA if 2 < maxr else NAN,
                _NA if 3 < maxr else NAN, _NA if 4 < maxr else NAN)
    @staticmethod
    def munp(n: float, sh) -> float:
        h = sh[0]
        k = sh[1]
        maxr = 5
        if h < 0 and k >= 0:
            maxr = int(-1.0 / h * k)
        elif k < 0:
            maxr = int(-1.0 / k)
        if n >= maxr:
            return NAN
        return _generic_munp(_Kappa4K, n, sh)

# ------------------------------------------------------------- crystalball

def _cb_N(beta: float, m: float) -> float:
    return 1.0 / (m / beta / (m - 1) * K.exp(-beta ** 2 / 2.0) + _NORM_PDF_C * _norm_cdf(beta))

@tuple
class _CrystalballK:
    @staticmethod
    def nshape() -> Literal[int]: return 2
    @staticmethod
    def check(sh) -> bool: return sh[1] > 1 and sh[0] > 0
    @staticmethod
    def pdf(x: float, sh) -> float:
        beta = sh[0]
        m = sh[1]
        N = _cb_N(beta, m)
        if x > -beta:
            return N * K.exp(-x ** 2 / 2)
        return N * (K.pow(m / beta, m) * K.exp(-beta ** 2 / 2.0) * K.pow(m / beta - beta - x, -m))
    @staticmethod
    def logpdf(x: float, sh) -> float:
        beta = sh[0]
        m = sh[1]
        N = _cb_N(beta, m)
        if x > -beta:
            return K.log(N) + (-x ** 2 / 2)
        return K.log(N) + (m * K.log(m / beta) - beta ** 2 / 2 - m * K.log(m / beta - beta - x))
    @staticmethod
    def cdf(x: float, sh) -> float:
        beta = sh[0]
        m = sh[1]
        N = _cb_N(beta, m)
        if x > -beta:
            return N * ((m / beta) * K.exp(-beta ** 2 / 2.0) / (m - 1)
                        + _NORM_PDF_C * (_norm_cdf(x) - _norm_cdf(-beta)))
        return N * (K.pow(m / beta, m) * K.exp(-beta ** 2 / 2.0)
                    * K.pow(m / beta - beta - x, -m + 1) / (m - 1))
    @staticmethod
    def sf(x: float, sh) -> float:
        beta = sh[0]
        m = sh[1]
        if x > -beta:
            Mv = m / beta / (m - 1) * K.exp(-beta ** 2 / 2) + _NORM_PDF_C * _norm_cdf(beta)
            return _NORM_PDF_C * _norm_sf(x) / Mv
        return 1 - _CrystalballK.cdf(x, sh)
    @staticmethod
    def ppf(p: float, sh) -> float:
        beta = sh[0]
        m = sh[1]
        N = _cb_N(beta, m)
        pbeta = N * (m / beta) * K.exp(-beta ** 2 / 2) / (m - 1)
        eb2 = K.exp(-beta ** 2 / 2)
        Cv = (m / beta) * eb2 / (m - 1)
        N2 = 1 / (Cv + _NORM_PDF_C * _norm_cdf(beta))
        if p < pbeta:
            return m / beta - beta - K.pow((m - 1) * K.pow(m / beta, -m) / eb2 * p / N2, 1 / (1 - m))
        return _norm_ppf(_norm_cdf(-beta) + (1 / _NORM_PDF_C) * (p / N2 - Cv))
    @staticmethod
    def munp(n: float, sh) -> float:
        beta = sh[0]
        m = sh[1]
        N = _cb_N(beta, m)
        if not (n + 1 < m):
            return INF
        A = K.pow(m / beta, m) * K.exp(-beta ** 2 / 2.0)
        Bv = m / beta - beta
        sgn = 1.0 if int(n) % 2 == 0 else -1.0
        rhs = (K.pow(2.0, (n - 1) / 2.0) * K.gamma((n + 1) / 2)
               * (1.0 + sgn * K.gammainc((n + 1) / 2, beta ** 2 / 2)))
        lhs = 0.0
        for k in range(int(n) + 1):
            sk = 1.0 if k % 2 == 0 else -1.0
            lhs += (M.binom(n, float(k)) * K.pow(Bv, n - k) * sk / (m - k - 1)
                    * K.pow(m / beta, -m + k + 1))
        return N * (A * lhs + rhs)
    @staticmethod
    def fitstart(data: np.ndarray[float, 1]) -> List[float]: return [1.0, 1.5]

# ------------------------------------------------------------------- argus

def _argus_phi(chi: float) -> float: return K.gammainc(1.5, chi ** 2 / 2) / 2

@tuple
class _ArgusK:
    @staticmethod
    def nshape() -> Literal[int]: return 1
    @staticmethod
    def a(sh) -> float: return 0.0
    @staticmethod
    def b(sh) -> float: return 1.0
    @staticmethod
    def logpdf(x: float, sh) -> float:
        chi = sh[0]
        y = 1.0 - x * x
        A = 3 * K.log(chi) - _NORM_PDF_LOGC - K.log(_argus_phi(chi))
        return A + K.log(x) + 0.5 * K.log1p(-x * x) - chi ** 2 * y / 2
    @staticmethod
    def pdf(x: float, sh) -> float: return K.exp(_ArgusK.logpdf(x, sh))
    @staticmethod
    def sf(x: float, sh) -> float:
        chi = sh[0]
        return _argus_phi(chi * K.sqrt((1 - x) * (1 + x))) / _argus_phi(chi)
    @staticmethod
    def cdf(x: float, sh) -> float: return 1.0 - _ArgusK.sf(x, sh)
    @staticmethod
    def rvs(rs, sh, size):
        chi = sh[0]
        shp = () if size is None else ((size,) if isinstance(size, int) else size)
        N = 1
        for s in shp:
            N *= s
        x = np.zeros(N)
        simulated = 0
        chi2 = chi * chi
        if chi <= 0.5:
            d = -chi2 / 2
            while simulated < N:
                k = N - simulated
                u = rs.uniform(size=k)
                v = rs.uniform(size=k)
                for i in range(k):
                    z = v[i] ** (2 / 3)
                    if K.log(u[i]) <= d * z:
                        x[simulated] = K.sqrt(1 - z)
                        simulated += 1
        elif chi <= 1.8:
            echi = K.exp(-chi2 / 2)
            while simulated < N:
                k = N - simulated
                u = rs.uniform(size=k)
                v = rs.uniform(size=k)
                for i in range(k):
                    z = 2 * K.log(echi * (1 - v[i]) + v[i]) / chi2
                    if u[i] ** 2 + z <= 0:
                        x[simulated] = K.sqrt(1 + z)
                        simulated += 1
        else:
            while simulated < N:
                k = N - simulated
                g = rs.standard_gamma(1.5, size=k)
                for i in range(k):
                    if g[i] <= chi2 / 2:
                        x[simulated] = g[i]
                        simulated += 1
            x = np.sqrt(1 - 2 * x / chi2)
        if size is None:
            return x
        return x.reshape(shp)
    @staticmethod
    def stats(sh):
        chi = sh[0]
        phi = _argus_phi(chi)
        import scipy.special._bessel as B
        m = K.sqrt(_PI / 8) * chi * B.ive(1.0, chi ** 2 / 4) / phi
        mu2 = 0.0
        if chi > 0.1:
            mu2 = 1 - 3 / chi ** 2 + chi * _norm_pdf(chi) / phi
        else:
            mu2 = K._polevl(chi, (-358 / 65690625, 0.0, -94 / 1010625, 0.0, 2 / 2625, 0.0,
                                  6 / 175, 0.0, 0.0, 0.4))
        return (m, mu2 - m ** 2, _NA, _NA)

# ------------------------------------------------------------- tukeylambda

def _tlvar(lam: float) -> float:
    if lam < -0.5:
        return NAN
    if lam == -0.5:
        return INF
    if fabs(lam) < 0.075:
        p = K._polevl(lam, (-0.02371146284628187, 0.17292046290190008, -0.5370742306855439,
                            0.7306125098871127, 3.289868133696453))
        q = K._polevl(lam, (0.2643989311168465, 1.7660926747377275, 4.184152498888124,
                            3.683605511659861, 1.0))
        return p / q
    return (2.0 / lam ** 2) * (1.0 / (1.0 + 2 * lam) - K.beta(lam + 1, lam + 1))

def _tlkurt(lam: float) -> float:
    if lam < -0.25:
        return NAN
    if lam == -0.25:
        return INF
    if fabs(lam) < 0.055:
        p = K._polevl(lam, (4.59796302262789, 0.20601184383406815, -22.653447381131077,
                            -5.853465139719495, 1.2))
        q = K._polevl(lam, (-2.789746758009912, 0.43075235247853005, 12.96663094361842,
                            7.171149192233599, 1.0))
        return p / q
    numer = (1.0 / (4 * lam + 1) - 4 * K.beta(3 * lam + 1, lam + 1)
             + 3 * K.beta(2 * lam + 1, 2 * lam + 1))
    denom = 2 * (1.0 / (2 * lam + 1) - K.beta(lam + 1, lam + 1)) ** 2
    return numer / denom - 3

class _TlEntr:
    lam: float
    def __init__(self, lam: float):
        self.lam = lam
    def __call__(self, p: float) -> float:
        return K.log(K.pow(p, self.lam - 1) + K.pow(1 - p, self.lam - 1))

@tuple
class _TukeylambdaK:
    @staticmethod
    def nshape() -> Literal[int]: return 1
    @staticmethod
    def check(sh) -> bool: return sh[0] == sh[0] and fabs(sh[0]) != INF
    @staticmethod
    def a(sh) -> float: return -1 / sh[0] if sh[0] > 0 else -INF
    @staticmethod
    def b(sh) -> float: return 1 / sh[0] if sh[0] > 0 else INF
    @staticmethod
    def open_support() -> Literal[bool]: return True
    @staticmethod
    def pdf(x: float, sh) -> float:
        lam = sh[0]
        Fx = LG.tklmbda(x, lam)
        Px = K.pow(Fx, lam - 1.0) + K.pow(1 - Fx, lam - 1.0)
        Px = 1.0 / Px
        if lam <= 0 or fabs(x) < 1.0 / lam:
            return Px
        return 0.0
    @staticmethod
    def cdf(x: float, sh) -> float: return LG.tklmbda(x, sh[0])
    @staticmethod
    def ppf(q: float, sh) -> float: return K.boxcox(q, sh[0]) - M.boxcox1p(-q, sh[0])
    @staticmethod
    def stats(sh): return (0.0, _tlvar(sh[0]), 0.0, _tlkurt(sh[0]))
    @staticmethod
    def entropy(sh) -> float:
        from scipy.integrate import quad
        return quad(_TlEntr(sh[0]), 0.0, 1.0)[0]

# --------------------------------------------------------------- kstwobign

@tuple
class _KstwobignK:
    @staticmethod
    def nshape() -> Literal[int]: return 0
    @staticmethod
    def a(sh) -> float: return 0.0
    @staticmethod
    def pdf(x: float, sh) -> float: return -KS.kolmogp(x)
    @staticmethod
    def cdf(x: float, sh) -> float: return KS.kolmogc(x)
    @staticmethod
    def sf(x: float, sh) -> float: return KS.kolmogorov(x)
    @staticmethod
    def ppf(q: float, sh) -> float: return KS.kolmogci(q)
    @staticmethod
    def isf(q: float, sh) -> float: return KS.kolmogi(q)

# --------------------------------------------------------- rel_breitwigner

@tuple
class _RelBreitwignerK:
    @staticmethod
    def nshape() -> Literal[int]: return 1
    @staticmethod
    def a(sh) -> float: return 0.0
    @staticmethod
    def pdf(x: float, sh) -> float:
        rho = sh[0]
        C = K.sqrt(2 * (1 + 1 / rho ** 2) / (1 + K.sqrt(1 + 1 / rho ** 2))) * 2 / _PI
        return C / (((x - rho) * (x + rho) / rho) ** 2 + 1)
    @staticmethod
    def cdf(x: float, sh) -> float:
        rho = sh[0]
        C = K.sqrt(2 / (1 + K.sqrt(1 + 1 / rho ** 2))) / _PI
        import cmath
        r = cmath.sqrt(complex(-1, 1 / rho)) * cmath.atan(x / cmath.sqrt(complex(-rho * rho, -rho)))
        return min(C * 2 * r.imag, 1.0)
    @staticmethod
    def munp(n: float, sh) -> float:
        rho = sh[0]
        if n == 0:
            return 1.
        if n == 1:
            C = K.sqrt(2 * (1 + 1 / rho ** 2) / (1 + K.sqrt(1 + 1 / rho ** 2))) / _PI * rho
            return C * (_PI / 2 + atan(rho))
        if n == 2:
            import cmath
            C = K.sqrt((1 + 1 / rho ** 2) / (2 * (1 + K.sqrt(1 + 1 / rho ** 2)))) * rho
            r = complex(1, -rho) / cmath.sqrt(complex(-1, -1 / rho))
            return 2 * C * r.real
        return INF
    @staticmethod
    def stats(sh): return (_NA, _NA, NAN, NAN)

halflogistic = rv_continuous[_HalflogisticK]('halflogistic')
genhalflogistic = rv_continuous[_GenhalflogisticK]('genhalflogistic')
halfgennorm = rv_continuous[_HalfgennormK]('halfgennorm')
invweibull = rv_continuous[_InvweibullK]('invweibull')
exponpow = rv_continuous[_ExponpowK]('exponpow')
powernorm = rv_continuous[_PowernormK]('powernorm')
powerlognorm = rv_continuous[_PowerlognormK]('powerlognorm')
laplace_asymmetric = rv_continuous[_LaplaceAsymK]('laplace_asymmetric')
trapezoid = rv_continuous[_TrapezoidK]('trapezoid')
truncweibull_min = rv_continuous[_TruncweibullMinK]('truncweibull_min')
truncpareto = rv_continuous[_TruncparetoK]('truncpareto')
gengamma = rv_continuous[_GengammaK]('gengamma')
genexpon = rv_continuous[_GenexponK]('genexpon')
kappa4 = rv_continuous[_Kappa4K]('kappa4')
crystalball = rv_continuous[_CrystalballK]('crystalball')
argus = rv_continuous[_ArgusK]('argus')
tukeylambda = rv_continuous[_TukeylambdaK]('tukeylambda')
kstwobign = rv_continuous[_KstwobignK]('kstwobign')
rel_breitwigner = rv_continuous[_RelBreitwignerK]('rel_breitwigner')
