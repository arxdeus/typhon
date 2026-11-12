# Copyright (C) 2026 the Typhon authors
# Ported from scipy 1.18.1 (BSD-3-Clause)

# Continuous distribution kernels (scipy.stats._continuous_distns).
# See _distn_infrastructure for the kernel protocol.

import numpy as np
import scipy.special._kernels as K
import scipy.special._cdf as C
import scipy.special._gamma2 as G2
import scipy.special._misc as M
import scipy.special._noncentral as NC
import scipy.special._hyp as H
from ._distn_infrastructure import rv_continuous, rv_frozen, NAN, INF, _NA, \
    _generic_munp, _k_ppf

_EULER = 0.577215664901532860606512090082402431042
_ZETA3 = 1.202056903159594285399738161511449990765
_SQRT_PI = 1.772453850905516027298167483341145182798
_SQRT_2_OVER_PI = 0.7978845608028654
_LOG_PI = 1.1447298858494002
_LOG_SQRT_2_OVER_PI = -0.22579135264472744
_LOG_SQRT_2PI = 0.91893853320467274178
_LOG2 = 0.6931471805599453
_PI = 3.141592653589793

from C import atan(float) -> float
from C import atan2(float, float) -> float
from C import tan(float) -> float
from C import tanh(float) -> float
from C import cosh(float) -> float
from C import sinh(float) -> float
from C import asinh(float) -> float
from C import fabs(float) -> float

def _tsize(size):
    if size is None:
        return ()
    elif isinstance(size, int):
        return (size,)
    else:
        return size

def _norm_pdf(x: float) -> float: return K.exp(-x * x / 2.0) / 2.5066282746310002
def _norm_logpdf(x: float) -> float: return -x * x / 2.0 - _LOG_SQRT_2PI
def _norm_cdf(x: float) -> float: return K.ndtr(x)
def _norm_sf(x: float) -> float: return K.ndtr(-x)
def _norm_logcdf(x: float) -> float: return K.log_ndtr(x)
def _norm_logsf(x: float) -> float: return K.log_ndtr(-x)
def _norm_ppf(q: float) -> float: return K.ndtri(q)
def _norm_isf(q: float) -> float: return -K.ndtri(q)

# ------------------------------------------------------------------- norm

@tuple
class _NormK:
    @staticmethod
    def nshape() -> Literal[int]: return 0
    @staticmethod
    def rvs(rs, sh, size): return rs.standard_normal(size)
    @staticmethod
    def pdf(z: float, sh) -> float: return _norm_pdf(z)
    @staticmethod
    def logpdf(z: float, sh) -> float: return _norm_logpdf(z)
    @staticmethod
    def cdf(z: float, sh) -> float: return _norm_cdf(z)
    @staticmethod
    def sf(z: float, sh) -> float: return _norm_sf(z)
    @staticmethod
    def logcdf(z: float, sh) -> float: return _norm_logcdf(z)
    @staticmethod
    def logsf(z: float, sh) -> float: return _norm_logsf(z)
    @staticmethod
    def ppf(q: float, sh) -> float: return _norm_ppf(q)
    @staticmethod
    def isf(q: float, sh) -> float: return _norm_isf(q)
    @staticmethod
    def stats(sh): return (0.0, 1.0, 0.0, 0.0)
    @staticmethod
    def entropy(sh) -> float: return 0.5 * (K.log(2 * _PI) + 1)
    @staticmethod
    def munp(n: float, sh) -> float:
        if n == 0:
            return 1.
        if int(n) % 2 == 0:
            r = 1.0
            k = int(n) - 1
            while k > 1:
                r *= k
                k -= 2
            return r
        return 0.
    @staticmethod
    def fitc(data: np.ndarray[float, 1], fx: List[float]) -> List[float]:
        loc = fx[0]
        if loc != loc:
            loc = float(data.mean())
        scale = fx[1]
        if scale != scale:
            scale = K.sqrt(float(((data - loc) ** 2).mean()))
        return [loc, scale]

# ---------------------------------------------------------------------- t

@tuple
class _TK:
    @staticmethod
    def nshape() -> Literal[int]: return 1
    @staticmethod
    def rvs(rs, sh, size): return rs.standard_t(sh[0], size=size)
    @staticmethod
    def logpdf(z: float, sh) -> float:
        df = sh[0]
        if K.isinf(df):
            return _norm_logpdf(z)
        return (K.log(G2.poch(0.5 * df, 0.5))
                - 0.5 * (K.log(df) + K.log(_PI))
                - (df + 1) / 2 * K.log1p(z * z / df))
    @staticmethod
    def pdf(z: float, sh) -> float:
        if K.isinf(sh[0]):
            return _norm_pdf(z)
        return K.exp(_TK.logpdf(z, sh))
    @staticmethod
    def cdf(z: float, sh) -> float: return K.stdtr(sh[0], z)
    @staticmethod
    def sf(z: float, sh) -> float: return K.stdtr(sh[0], -z)
    @staticmethod
    def ppf(q: float, sh) -> float: return K.stdtrit(sh[0], q)
    @staticmethod
    def isf(q: float, sh) -> float: return -K.stdtrit(sh[0], q)
    @staticmethod
    def stats(sh):
        df = sh[0]
        mu = 0.0 if df > 1 else INF
        mu2 = NAN
        if df > 1 and df <= 2:
            mu2 = INF
        elif df > 2 and not K.isinf(df):
            mu2 = df / (df - 2.0)
        elif K.isinf(df):
            mu2 = 1.0
        g1 = 0.0 if df > 3 else NAN
        g2 = NAN
        if df > 2 and df <= 4:
            g2 = INF
        elif df > 4 and not K.isinf(df):
            g2 = 6.0 / (df - 4.0)
        elif K.isinf(df):
            g2 = 0.0
        return (mu, mu2, g1, g2)
    @staticmethod
    def entropy(sh) -> float:
        df = sh[0]
        if df >= 100:
            return (0.5 * (K.log(2 * _PI) + 1) + 1 / df + (df ** -2.) / 4 - (df ** -3.) / 6
                    - (df ** -4.) / 8 + 3 / 10 * (df ** -5.) + (df ** -6.) / 4)
        half = df / 2
        half1 = (df + 1) / 2
        return (half1 * (K.digamma(half1) - K.digamma(half))
                + K.log(K.sqrt(df) * K.beta(half, 0.5)))

# ------------------------------------------------------------------- chi2

@tuple
class _Chi2K:
    @staticmethod
    def nshape() -> Literal[int]: return 1
    @staticmethod
    def a(sh) -> float: return 0.0
    @staticmethod
    def rvs(rs, sh, size): return rs.chisquare(sh[0], size)
    @staticmethod
    def logpdf(z: float, sh) -> float:
        df = sh[0]
        return K.xlogy(df / 2. - 1, z) - z / 2. - K.gammaln(df / 2.) - (K.log(2.0) * df) / 2.
    @staticmethod
    def pdf(z: float, sh) -> float: return K.exp(_Chi2K.logpdf(z, sh))
    @staticmethod
    def cdf(z: float, sh) -> float: return K.chdtr(sh[0], z)
    @staticmethod
    def sf(z: float, sh) -> float: return K.chdtrc(sh[0], z)
    @staticmethod
    def ppf(q: float, sh) -> float: return 2 * K.gammaincinv(sh[0] / 2, q)
    @staticmethod
    def isf(q: float, sh) -> float: return K.chdtri(sh[0], q)
    @staticmethod
    def stats(sh):
        df = sh[0]
        return (df, 2 * df, 2 * K.sqrt(2.0 / df), 12.0 / df)
    @staticmethod
    def entropy(sh) -> float:
        h = 0.5 * sh[0]
        if h < 125:
            return h + K.log(2.0) + K.gammaln(h) + (1 - h) * K.digamma(h)
        c = K.log(2.0) + 0.5 * (1 + K.log(2 * _PI))
        hh = 0.5 / h
        return hh * (-2 / 3 + hh * (-1 / 3 + hh * (-4 / 45 + hh / 7.5))) + 0.5 * K.log(h) + c

# ---------------------------------------------------------------------- f

@tuple
class _FK:
    @staticmethod
    def nshape() -> Literal[int]: return 2
    @staticmethod
    def a(sh) -> float: return 0.0
    @staticmethod
    def rvs(rs, sh, size): return rs.f(sh[0], sh[1], size)
    @staticmethod
    def logpdf(z: float, sh) -> float:
        n = sh[0]
        m = sh[1]
        return (m / 2 * K.log(m) + n / 2 * K.log(n) + K.xlogy(n / 2 - 1, z)
                - (((n + m) / 2) * K.log(m + n * z) + K.betaln(n / 2, m / 2)))
    @staticmethod
    def pdf(z: float, sh) -> float: return K.exp(_FK.logpdf(z, sh))
    @staticmethod
    def cdf(z: float, sh) -> float: return K.fdtr(sh[0], sh[1], z)
    @staticmethod
    def sf(z: float, sh) -> float: return K.fdtrc(sh[0], sh[1], z)
    @staticmethod
    def ppf(q: float, sh) -> float: return K.fdtri(sh[0], sh[1], q)
    @staticmethod
    def stats(sh):
        v1 = sh[0]
        v2 = sh[1]
        v2_2, v2_4, v2_6, v2_8 = v2 - 2., v2 - 4., v2 - 6., v2 - 8.
        mu = v2 / v2_2 if v2 > 2 else INF
        mu2 = 2 * v2 * v2 * (v1 + v2_2) / (v1 * v2_2 ** 2 * v2_4) if v2 > 4 else INF
        g1 = (2 * v1 + v2_2) / v2_6 * K.sqrt(v2_4 / (v1 * (v1 + v2_2))) if v2 > 6 else NAN
        g1 *= K.sqrt(8.)
        g2 = (8 + g1 * g1 * v2_6) / v2_8 if v2 > 8 else NAN
        g2 *= 3. / 2.
        return (mu, mu2, g1, g2)
    @staticmethod
    def entropy(sh) -> float:
        dfn = sh[0]
        dfd = sh[1]
        hn = 0.5 * dfn
        hd = 0.5 * dfd
        hs = 0.5 * (dfn + dfd)
        return (K.log(dfd) - K.log(dfn) + K.betaln(hn, hd) + (1 - hn) * K.digamma(hn)
                - (1 + hd) * K.digamma(hd) + hs * K.digamma(hs))

# ---------------------------------------------------------------- uniform

@tuple
class _UniformK:
    @staticmethod
    def nshape() -> Literal[int]: return 0
    @staticmethod
    def a(sh) -> float: return 0.0
    @staticmethod
    def b(sh) -> float: return 1.0
    @staticmethod
    def rvs(rs, sh, size): return rs.uniform(0.0, 1.0, size)
    @staticmethod
    def pdf(z: float, sh) -> float: return 1.0
    @staticmethod
    def cdf(z: float, sh) -> float: return z
    @staticmethod
    def ppf(q: float, sh) -> float: return q
    @staticmethod
    def stats(sh): return (0.5, 1.0 / 12, 0.0, -1.2)
    @staticmethod
    def entropy(sh) -> float: return 0.0
    @staticmethod
    def fitc(data: np.ndarray[float, 1], fx: List[float]) -> List[float]:
        floc = fx[0]
        fscale = fx[1]
        mn = float(data.min())
        mx = float(data.max())
        if fscale != fscale:
            if floc != floc:
                return [mn, mx - mn]
            if mn < floc:
                raise ValueError("Invalid values in `data`.  Maximum likelihood estimation "
                                 "with 'uniform' requires that floc <= data.min().")
            return [floc, mx - floc]
        ptp = mx - mn
        if ptp > fscale:
            raise ValueError("Invalid values in `data`. Maximum likelihood estimation "
                             "with 'uniform' requires that data.ptp() <= fscale.")
        return [mn - 0.5 * (fscale - ptp), fscale]

# ------------------------------------------------------------------ expon

@tuple
class _ExponK:
    @staticmethod
    def nshape() -> Literal[int]: return 0
    @staticmethod
    def a(sh) -> float: return 0.0
    @staticmethod
    def rvs(rs, sh, size): return rs.standard_exponential(size)
    @staticmethod
    def pdf(z: float, sh) -> float: return K.exp(-z)
    @staticmethod
    def logpdf(z: float, sh) -> float: return -z
    @staticmethod
    def cdf(z: float, sh) -> float: return -K.expm1(-z)
    @staticmethod
    def ppf(q: float, sh) -> float: return -K.log1p(-q)
    @staticmethod
    def sf(z: float, sh) -> float: return K.exp(-z)
    @staticmethod
    def logsf(z: float, sh) -> float: return -z
    @staticmethod
    def isf(q: float, sh) -> float: return -K.log(q)
    @staticmethod
    def stats(sh): return (1.0, 1.0, 2.0, 6.0)
    @staticmethod
    def entropy(sh) -> float: return 1.0
    @staticmethod
    def fitc(data: np.ndarray[float, 1], fx: List[float]) -> List[float]:
        mn = float(data.min())
        loc = fx[0]
        if loc != loc:
            loc = mn
        elif mn < loc:
            raise ValueError("Invalid values in `data`.  Maximum likelihood estimation "
                             "with 'expon' requires that floc <= data.min().")
        scale = fx[1]
        if scale != scale:
            scale = float(data.mean()) - loc
        return [loc, scale]

# ------------------------------------------------------------------ gamma

def _digammainv(y: float) -> float:
    # scipy.stats._continuous_distns._digammainv (Newton on digamma(x) - y)
    if y > -0.125:
        x = K.exp(y) + 0.5
    else:
        x = -1.0 / (y - K.digamma(1.0))
    for _ in range(100):
        # scipy uses optimize.fsolve with fprime=polygamma(1, x)
        dx = (K.digamma(x) - y) / G2.polygamma(1.0, x)
        x -= dx
        if fabs(dx) < 1e-11 * fabs(x):
            break
    return x

class _GammaShapeEq:
    s: float
    def __init__(self, s: float):
        self.s = s
    def __call__(self, a: float) -> float:
        return K.log(a) - K.digamma(a) - self.s

def _skew1(data: np.ndarray[float, 1]) -> float:
    mu = float(data.mean())
    m2 = float(((data - mu) ** 2).mean())
    m3 = float(((data - mu) ** 3).mean())
    return m3 / m2 ** 1.5

def _kurt1(data: np.ndarray[float, 1]) -> float:
    mu = float(data.mean())
    m2 = float(((data - mu) ** 2).mean())
    m4 = float(((data - mu) ** 4).mean())
    return m4 / m2 ** 2 - 3

@tuple
class _GammaK:
    @staticmethod
    def nshape() -> Literal[int]: return 1
    @staticmethod
    def a(sh) -> float: return 0.0
    @staticmethod
    def rvs(rs, sh, size): return rs.standard_gamma(sh[0], size)
    @staticmethod
    def logpdf(z: float, sh) -> float:
        a = sh[0]
        return K.xlogy(a - 1.0, z) - z - K.gammaln(a)
    @staticmethod
    def pdf(z: float, sh) -> float: return K.exp(_GammaK.logpdf(z, sh))
    @staticmethod
    def cdf(z: float, sh) -> float: return K.gammainc(sh[0], z)
    @staticmethod
    def sf(z: float, sh) -> float: return K.gammaincc(sh[0], z)
    @staticmethod
    def ppf(q: float, sh) -> float: return K.gammaincinv(sh[0], q)
    @staticmethod
    def isf(q: float, sh) -> float: return K.gammainccinv(sh[0], q)
    @staticmethod
    def stats(sh):
        a = sh[0]
        return (a, a, 2.0 / K.sqrt(a), 6.0 / a)
    @staticmethod
    def munp(n: float, sh) -> float: return G2.poch(sh[0], n)
    @staticmethod
    def entropy(sh) -> float:
        a = sh[0]
        if a < 250:
            return K.digamma(a) * (1 - a) + a + K.gammaln(a)
        return (0.5 * (1. + K.log(2 * _PI) + K.log(a)) - 1 / (3 * a)
                - (a ** -2.) / 12 - (a ** -3.) / 90 + (a ** -4.) / 120)
    @staticmethod
    def fitstart(data: np.ndarray[float, 1]) -> List[float]:
        sk = _skew1(data)
        return [4 / (1e-8 + sk ** 2)]
    @staticmethod
    def fitc(data: np.ndarray[float, 1], fx: List[float]) -> List[float]:
        f0 = fx[0]
        floc = fx[1]
        fscale = fx[2]
        if floc != floc:
            return List[float]()
        for v in data:
            if v <= floc:
                raise ValueError("Invalid values in `data`.  Maximum likelihood estimation "
                                 "with 'gamma' requires that floc < data.min().")
        d = data - floc
        xbar = float(d.mean())
        lmean = float(np.log(d).mean())
        a = 0.0
        scale = 0.0
        if fscale != fscale:
            if f0 == f0:
                a = f0
            else:
                s = K.log(xbar) - lmean
                aest = (3 - s + K.sqrt((s - 3) ** 2 + 24 * s)) / (12 * s)
                from scipy.optimize import brentq
                a = brentq(_GammaShapeEq(s), aest * (1 - 0.4), aest * (1 + 0.4), disp=False)
            scale = xbar / a
        else:
            a = _digammainv(lmean - K.log(fscale))
            scale = fscale
        return [a, floc, scale]

# ------------------------------------------------------------------- beta

@tuple
class _BetaK:
    @staticmethod
    def nshape() -> Literal[int]: return 2
    @staticmethod
    def a(sh) -> float: return 0.0
    @staticmethod
    def b(sh) -> float: return 1.0
    @staticmethod
    def rvs(rs, sh, size): return rs.beta(sh[0], sh[1], size)
    @staticmethod
    def logpdf(z: float, sh) -> float:
        a = sh[0]
        b = sh[1]
        return K.xlog1py(b - 1.0, -z) + K.xlogy(a - 1.0, z) - K.betaln(a, b)
    @staticmethod
    def pdf(z: float, sh) -> float:
        a = sh[0]
        b = sh[1]
        if z == 0.0:
            if a < 1:
                return INF
            return b if a == 1 else 0.0
        if z == 1.0:
            if b < 1:
                return INF
            return a if b == 1 else 0.0
        return K.exp(_BetaK.logpdf(z, sh))
    @staticmethod
    def cdf(z: float, sh) -> float: return K.betainc(sh[0], sh[1], z)
    @staticmethod
    def sf(z: float, sh) -> float: return K.betaincc(sh[0], sh[1], z)
    @staticmethod
    def ppf(q: float, sh) -> float: return K.betaincinv(sh[0], sh[1], q)
    @staticmethod
    def isf(q: float, sh) -> float:
        return M.betainccinv(sh[0], sh[1], q)
    @staticmethod
    def stats(sh):
        a = sh[0]
        b = sh[1]
        s = a + b
        mean = a / s
        var = a * b / (s ** 2 * (s + 1))
        skew = (2 * (b - a) * K.sqrt(s + 1)) / ((s + 2) * K.sqrt(a * b))
        kn = 6 * ((a - b) ** 2 * (s + 1) - a * b * (s + 2))
        kd = a * b * (s + 2) * (s + 3)
        return (mean, var, skew, kn / kd)
    @staticmethod
    def entropy(sh) -> float:
        a = sh[0]
        b = sh[1]
        if a < 4.9e6 and b < 4.9e6:
            return (K.betaln(a, b) - (a - 1) * K.digamma(a) -
                    (b - 1) * K.digamma(b) + (a + b - 2) * K.digamma(a + b))
        s = a + b
        lt = 0.5 * (K.log(2 * _PI) + K.log(a) + K.log(b) - 3 * K.log(s) + 1)
        t1 = 110 / s + 20 * s ** -2.0 + s ** -3.0 - 2 * s ** -4.0
        t2 = -50 / a - 10 * a ** -2.0 - a ** -3.0 + a ** -4.0
        t3 = -50 / b - 10 * b ** -2.0 - b ** -3.0 + b ** -4.0
        return lt + (t1 + t2 + t3) / 120

# ---------------------------------------------------------------- lognorm

@tuple
class _LognormK:
    @staticmethod
    def nshape() -> Literal[int]: return 1
    @staticmethod
    def a(sh) -> float: return 0.0
    @staticmethod
    def open_support() -> Literal[bool]: return True
    @staticmethod
    def rvs(rs, sh, size): return np.exp(sh[0] * rs.standard_normal(size))
    @staticmethod
    def logpdf(z: float, sh) -> float:
        s = sh[0]
        if z == 0:
            return -INF
        return -K.log(z) ** 2 / (2 * s ** 2) - K.log(s * z * K.sqrt(2 * _PI))
    @staticmethod
    def pdf(z: float, sh) -> float: return K.exp(_LognormK.logpdf(z, sh))
    @staticmethod
    def cdf(z: float, sh) -> float: return _norm_cdf(K.log(z) / sh[0])
    @staticmethod
    def logcdf(z: float, sh) -> float: return _norm_logcdf(K.log(z) / sh[0])
    @staticmethod
    def ppf(q: float, sh) -> float: return K.exp(sh[0] * _norm_ppf(q))
    @staticmethod
    def sf(z: float, sh) -> float: return _norm_sf(K.log(z) / sh[0])
    @staticmethod
    def logsf(z: float, sh) -> float: return _norm_logsf(K.log(z) / sh[0])
    @staticmethod
    def isf(q: float, sh) -> float: return K.exp(sh[0] * _norm_isf(q))
    @staticmethod
    def stats(sh):
        s = sh[0]
        p = K.exp(s * s)
        mu = K.sqrt(p)
        mu2 = p * (p - 1)
        g1 = K.sqrt(p - 1) * (2 + p)
        g2 = (((p + 2.0) * p + 3.0) * p + 0.0) * p - 6.0
        return (mu, mu2, g1, g2)
    @staticmethod
    def entropy(sh) -> float: return 0.5 * (1 + K.log(2 * _PI) + 2 * K.log(sh[0]))
    @staticmethod
    def fitc(data: np.ndarray[float, 1], fx: List[float]) -> List[float]:
        fshape = fx[0]
        floc = fx[1]
        fscale = fx[2]
        if floc != floc:
            return List[float]()
        if floc >= float(data.min()):
            raise ValueError("Invalid values in `data`.  Maximum likelihood estimation "
                             "with 'lognorm' requires that 0.0 < (x - loc)/scale  < inf "
                             "for each x in `data`.")
        ld = np.log(data - floc)
        scale = fscale if fscale == fscale and fscale != 0 else K.exp(float(ld.mean()))
        shape = fshape if fshape == fshape and fshape != 0 else K.sqrt(float(((ld - K.log(scale)) ** 2).mean()))
        return [shape, floc, scale]

# ---------------------------------------------------------------- laplace

@tuple
class _LaplaceK:
    @staticmethod
    def nshape() -> Literal[int]: return 0
    @staticmethod
    def rvs(rs, sh, size): return rs.laplace(0.0, 1.0, size=size)
    @staticmethod
    def pdf(z: float, sh) -> float: return 0.5 * K.exp(-fabs(z))
    @staticmethod
    def cdf(z: float, sh) -> float:
        return 1.0 - 0.5 * K.exp(-z) if z > 0 else 0.5 * K.exp(z)
    @staticmethod
    def sf(z: float, sh) -> float: return _LaplaceK.cdf(-z, sh)
    @staticmethod
    def ppf(q: float, sh) -> float:
        return -K.log(2 * (1 - q)) if q > 0.5 else K.log(2 * q)
    @staticmethod
    def isf(q: float, sh) -> float: return -_LaplaceK.ppf(q, sh)
    @staticmethod
    def stats(sh): return (0.0, 2.0, 0.0, 3.0)
    @staticmethod
    def entropy(sh) -> float: return K.log(2.0) + 1
    @staticmethod
    def fitc(data: np.ndarray[float, 1], fx: List[float]) -> List[float]:
        floc = fx[0]
        fscale = fx[1]
        if floc != floc:
            floc = float(np.median(data))
        if fscale != fscale:
            fscale = float(np.sum(np.abs(data - floc))) / len(data)
        return [floc, fscale]

# --------------------------------------------------------------- logistic

@tuple
class _LogisticK:
    @staticmethod
    def nshape() -> Literal[int]: return 0
    @staticmethod
    def rvs(rs, sh, size): return rs.logistic(size=size)
    @staticmethod
    def logpdf(z: float, sh) -> float:
        y = -fabs(z)
        return y - 2. * K.log1p(K.exp(y))
    @staticmethod
    def pdf(z: float, sh) -> float: return K.exp(_LogisticK.logpdf(z, sh))
    @staticmethod
    def cdf(z: float, sh) -> float: return K.expit(z)
    @staticmethod
    def logcdf(z: float, sh) -> float: return K.log_expit(z)
    @staticmethod
    def ppf(q: float, sh) -> float: return K.logit(q)
    @staticmethod
    def sf(z: float, sh) -> float: return K.expit(-z)
    @staticmethod
    def logsf(z: float, sh) -> float: return K.log_expit(-z)
    @staticmethod
    def isf(q: float, sh) -> float: return -K.logit(q)
    @staticmethod
    def stats(sh): return (0.0, _PI * _PI / 3.0, 0.0, 6.0 / 5.0)
    @staticmethod
    def entropy(sh) -> float: return 2.0

# ----------------------------------------------------------------- cauchy

@tuple
class _CauchyK:
    @staticmethod
    def nshape() -> Literal[int]: return 0
    @staticmethod
    def pdf(z: float, sh) -> float: return 1.0 / _PI / (1.0 + z * z)
    @staticmethod
    def logpdf(z: float, sh) -> float:
        ax = fabs(z)
        if ax < 1:
            return -_LOG_PI - K.log1p(ax ** 2)
        return -_LOG_PI - (2 * K.log(ax) + K.log1p((1 / ax) ** 2))
    @staticmethod
    def cdf(z: float, sh) -> float: return atan2(1.0, -z) / _PI
    @staticmethod
    def sf(z: float, sh) -> float: return atan2(1.0, z) / _PI
    @staticmethod
    def ppf(q: float, sh) -> float:
        # boost cauchy quantile
        if q == 0.5:
            return 0.0
        p = q
        if p > 0.5:
            p = p - 1
        r = -1.0 / tan(_PI * p)
        return r
    @staticmethod
    def isf(q: float, sh) -> float: return -_CauchyK.ppf(q, sh)
    @staticmethod
    def stats(sh): return (NAN, NAN, NAN, NAN)
    @staticmethod
    def entropy(sh) -> float: return K.log(4 * _PI)

_XMIN = 2.2250738585072014e-308
_LOGXMIN = -708.3964185322641

def _powm1(x: float, y: float) -> float:
    # boost::math::powm1
    if x > 0:
        l = y * K.log(x)
        if fabs(l) < 0.5:
            return K.expm1(l)
    return K.pow(x, y) - 1.0

# ------------------------------------------------------ weibull_min / max

@tuple
class _WeibullMinK:
    @staticmethod
    def nshape() -> Literal[int]: return 1
    @staticmethod
    def a(sh) -> float: return 0.0
    @staticmethod
    def pdf(z: float, sh) -> float:
        c = sh[0]
        return c * K.pow(z, c - 1) * K.exp(-K.pow(z, c))
    @staticmethod
    def logpdf(z: float, sh) -> float:
        c = sh[0]
        return K.log(c) + K.xlogy(c - 1, z) - K.pow(z, c)
    @staticmethod
    def cdf(z: float, sh) -> float: return -K.expm1(-K.pow(z, sh[0]))
    @staticmethod
    def ppf(q: float, sh) -> float: return K.pow(-K.log1p(-q), 1.0 / sh[0])
    @staticmethod
    def sf(z: float, sh) -> float: return K.exp(-K.pow(z, sh[0]))
    @staticmethod
    def logsf(z: float, sh) -> float: return -K.pow(z, sh[0])
    @staticmethod
    def isf(q: float, sh) -> float: return K.pow(-K.log(q), 1 / sh[0])
    @staticmethod
    def munp(n: float, sh) -> float: return K.gamma(1.0 + n * 1.0 / sh[0])
    @staticmethod
    def entropy(sh) -> float:
        c = sh[0]
        return -_EULER / c - K.log(c) + _EULER + 1

@tuple
class _WeibullMaxK:
    @staticmethod
    def nshape() -> Literal[int]: return 1
    @staticmethod
    def b(sh) -> float: return 0.0
    @staticmethod
    def pdf(z: float, sh) -> float:
        c = sh[0]
        return c * K.pow(-z, c - 1) * K.exp(-K.pow(-z, c))
    @staticmethod
    def logpdf(z: float, sh) -> float:
        c = sh[0]
        return K.log(c) + K.xlogy(c - 1, -z) - K.pow(-z, c)
    @staticmethod
    def cdf(z: float, sh) -> float: return K.exp(-K.pow(-z, sh[0]))
    @staticmethod
    def logcdf(z: float, sh) -> float: return -K.pow(-z, sh[0])
    @staticmethod
    def sf(z: float, sh) -> float: return -K.expm1(-K.pow(-z, sh[0]))
    @staticmethod
    def ppf(q: float, sh) -> float: return -K.pow(-K.log(q), 1.0 / sh[0])
    @staticmethod
    def munp(n: float, sh) -> float:
        val = K.gamma(1.0 + n * 1.0 / sh[0])
        return -val if int(n) % 2 else val
    @staticmethod
    def entropy(sh) -> float:
        c = sh[0]
        return -_EULER / c - K.log(c) + _EULER + 1

# ------------------------------------------------------------ gumbel_r/l

@tuple
class _GumbelRK:
    @staticmethod
    def nshape() -> Literal[int]: return 0
    @staticmethod
    def logpdf(z: float, sh) -> float: return -z - K.exp(-z)
    @staticmethod
    def pdf(z: float, sh) -> float: return K.exp(-z - K.exp(-z))
    @staticmethod
    def cdf(z: float, sh) -> float: return K.exp(-K.exp(-z))
    @staticmethod
    def logcdf(z: float, sh) -> float: return -K.exp(-z)
    @staticmethod
    def ppf(q: float, sh) -> float: return -K.log(-K.log(q))
    @staticmethod
    def sf(z: float, sh) -> float: return -K.expm1(-K.exp(-z))
    @staticmethod
    def isf(q: float, sh) -> float: return -K.log(-K.log1p(-q))
    @staticmethod
    def stats(sh): return (_EULER, _PI * _PI / 6.0, 12 * K.sqrt(6.0) / _PI ** 3 * _ZETA3, 12.0 / 5)
    @staticmethod
    def entropy(sh) -> float: return _EULER + 1.

@tuple
class _GumbelLK:
    @staticmethod
    def nshape() -> Literal[int]: return 0
    @staticmethod
    def logpdf(z: float, sh) -> float: return z - K.exp(z)
    @staticmethod
    def pdf(z: float, sh) -> float: return K.exp(z - K.exp(z))
    @staticmethod
    def cdf(z: float, sh) -> float: return -K.expm1(-K.exp(z))
    @staticmethod
    def ppf(q: float, sh) -> float: return K.log(-K.log1p(-q))
    @staticmethod
    def logsf(z: float, sh) -> float: return -K.exp(z)
    @staticmethod
    def sf(z: float, sh) -> float: return K.exp(-K.exp(z))
    @staticmethod
    def isf(q: float, sh) -> float: return K.log(-K.log(q))
    @staticmethod
    def stats(sh): return (-_EULER, _PI * _PI / 6.0, -12 * K.sqrt(6.0) / _PI ** 3 * _ZETA3, 12.0 / 5)
    @staticmethod
    def entropy(sh) -> float: return _EULER + 1.

# ------------------------------------------------------- pareto / lomax

def _pareto_stats(b: float):
    mu = b / (b - 1.0) if b > 1 else INF
    mu2 = b / (b - 2.0) / (b - 1.0) ** 2 if b > 2 else INF
    g1 = 2 * (b + 1.0) * K.sqrt(b - 2.0) / ((b - 3.0) * K.sqrt(b)) if b > 3 else NAN
    g2 = NAN
    if b > 4:
        g2 = (6.0 * (((b + 1.0) * b - 6) * b - 2) / (((b - 7.0) * b + 12.0) * b + 0.0))
    return (mu, mu2, g1, g2)

@tuple
class _ParetoK:
    @staticmethod
    def nshape() -> Literal[int]: return 1
    @staticmethod
    def a(sh) -> float: return 1.0
    @staticmethod
    def pdf(z: float, sh) -> float: return sh[0] * z ** (-sh[0] - 1)
    @staticmethod
    def cdf(z: float, sh) -> float: return 1 - z ** (-sh[0])
    @staticmethod
    def ppf(q: float, sh) -> float: return K.pow(1 - q, -1.0 / sh[0])
    @staticmethod
    def sf(z: float, sh) -> float: return z ** (-sh[0])
    @staticmethod
    def isf(q: float, sh) -> float: return K.pow(q, -1.0 / sh[0])
    @staticmethod
    def stats(sh): return _pareto_stats(sh[0])
    @staticmethod
    def entropy(sh) -> float:
        b = sh[0]
        return 1 + 1.0 / b - K.log(b)

@tuple
class _LomaxK:
    @staticmethod
    def nshape() -> Literal[int]: return 1
    @staticmethod
    def a(sh) -> float: return 0.0
    @staticmethod
    def pdf(z: float, sh) -> float:
        c = sh[0]
        return c * 1.0 / (1.0 + z) ** (c + 1.0)
    @staticmethod
    def logpdf(z: float, sh) -> float:
        c = sh[0]
        return K.log(c) - (c + 1) * K.log1p(z)
    @staticmethod
    def cdf(z: float, sh) -> float: return -K.expm1(-sh[0] * K.log1p(z))
    @staticmethod
    def sf(z: float, sh) -> float: return K.exp(-sh[0] * K.log1p(z))
    @staticmethod
    def logsf(z: float, sh) -> float: return -sh[0] * K.log1p(z)
    @staticmethod
    def ppf(q: float, sh) -> float: return K.expm1(-K.log1p(-q) / sh[0])
    @staticmethod
    def isf(q: float, sh) -> float: return q ** (-1.0 / sh[0]) - 1
    @staticmethod
    def stats(sh):
        # pareto.stats(c, loc=-1.0, moments='mvsk')
        mu, mu2, g1, g2 = _pareto_stats(sh[0])
        return (mu - 1.0, mu2, g1, g2)
    @staticmethod
    def entropy(sh) -> float:
        c = sh[0]
        return 1 + 1.0 / c - K.log(c)

# ------------------------------------------------- rayleigh / maxwell / chi

@tuple
class _ChiK:
    @staticmethod
    def nshape() -> Literal[int]: return 1
    @staticmethod
    def a(sh) -> float: return 0.0
    @staticmethod
    def rvs(rs, sh, size): return np.sqrt(rs.chisquare(sh[0], size))
    @staticmethod
    def logpdf(z: float, sh) -> float:
        df = sh[0]
        l = K.log(2.0) - .5 * K.log(2.0) * df - K.gammaln(.5 * df)
        return l + K.xlogy(df - 1., z) - .5 * z ** 2
    @staticmethod
    def pdf(z: float, sh) -> float: return K.exp(_ChiK.logpdf(z, sh))
    @staticmethod
    def cdf(z: float, sh) -> float: return K.gammainc(.5 * sh[0], .5 * z ** 2)
    @staticmethod
    def sf(z: float, sh) -> float: return K.gammaincc(.5 * sh[0], .5 * z ** 2)
    @staticmethod
    def ppf(q: float, sh) -> float: return K.sqrt(2 * K.gammaincinv(.5 * sh[0], q))
    @staticmethod
    def isf(q: float, sh) -> float: return K.sqrt(2 * K.gammainccinv(.5 * sh[0], q))
    @staticmethod
    def stats(sh):
        df = sh[0]
        mu = K.sqrt(2.0) * G2.poch(0.5 * df, 0.5)
        mu2 = df - mu * mu
        g1 = (2 * mu ** 3.0 + mu * (1 - 2 * df)) / K.pow(mu2, 1.5)
        g2 = 2 * df * (1.0 - df) - 6 * mu ** 4 + 4 * mu ** 2 * (2 * df - 1)
        g2 /= mu2 ** 2.0
        return (mu, mu2, g1, g2)
    @staticmethod
    def entropy(sh) -> float:
        df = sh[0]
        if df < 300:
            return K.gammaln(.5 * df) + 0.5 * (df - K.log(2.0) - (df - 1) * K.digamma(0.5 * df))
        return (0.5 + K.log(_PI) / 2 - (df ** -1) / 6 - (df ** -2) / 6
                - 4 / 45 * (df ** -3) + (df ** -4) / 15)

@tuple
class _RayleighK:
    @staticmethod
    def nshape() -> Literal[int]: return 0
    @staticmethod
    def a(sh) -> float: return 0.0
    @staticmethod
    def open_support() -> Literal[bool]: return True
    @staticmethod
    def rvs(rs, sh, size): return np.sqrt(rs.chisquare(2.0, size))
    @staticmethod
    def logpdf(z: float, sh) -> float: return K.log(z) - 0.5 * z * z
    @staticmethod
    def pdf(z: float, sh) -> float: return K.exp(K.log(z) - 0.5 * z * z)
    @staticmethod
    def cdf(z: float, sh) -> float: return -K.expm1(-0.5 * z ** 2)
    @staticmethod
    def ppf(q: float, sh) -> float: return K.sqrt(-2 * K.log1p(-q))
    @staticmethod
    def sf(z: float, sh) -> float: return K.exp(-0.5 * z * z)
    @staticmethod
    def logsf(z: float, sh) -> float: return -0.5 * z * z
    @staticmethod
    def isf(q: float, sh) -> float: return K.sqrt(-2 * K.log(q))
    @staticmethod
    def stats(sh):
        val = 4 - _PI
        return (K.sqrt(_PI / 2), val / 2, 2 * (_PI - 3) * K.sqrt(_PI) / val ** 1.5,
                6 * _PI / val - 16 / val ** 2)
    @staticmethod
    def entropy(sh) -> float: return _EULER / 2.0 + 1 - 0.5 * K.log(2.0)

@tuple
class _MaxwellK:
    @staticmethod
    def nshape() -> Literal[int]: return 0
    @staticmethod
    def a(sh) -> float: return 0.0
    @staticmethod
    def rvs(rs, sh, size): return np.sqrt(rs.chisquare(3.0, size))
    @staticmethod
    def pdf(z: float, sh) -> float: return _SQRT_2_OVER_PI * z * z * K.exp(-z * z / 2.0)
    @staticmethod
    def logpdf(z: float, sh) -> float: return _LOG_SQRT_2_OVER_PI + 2 * K.log(z) - 0.5 * z * z
    @staticmethod
    def cdf(z: float, sh) -> float: return K.gammainc(1.5, z * z / 2.0)
    @staticmethod
    def ppf(q: float, sh) -> float: return K.sqrt(2 * K.gammaincinv(1.5, q))
    @staticmethod
    def sf(z: float, sh) -> float: return K.gammaincc(1.5, z * z / 2.0)
    @staticmethod
    def isf(q: float, sh) -> float: return K.sqrt(2 * K.gammainccinv(1.5, q))
    @staticmethod
    def stats(sh):
        val = 3 * _PI - 8
        return (2 * K.sqrt(2.0 / _PI), 3 - 8 / _PI, K.sqrt(2.0) * (32 - 10 * _PI) / val ** 1.5,
                (-12 * _PI * _PI + 160 * _PI - 384) / val ** 2.0)
    @staticmethod
    def entropy(sh) -> float: return _EULER + 0.5 * K.log(2 * _PI) - 0.5

# -------------------------------------------------- halfnorm / halfcauchy

@tuple
class _HalfnormK:
    @staticmethod
    def nshape() -> Literal[int]: return 0
    @staticmethod
    def a(sh) -> float: return 0.0
    @staticmethod
    def rvs(rs, sh, size): return np.abs(rs.standard_normal(size=size))
    @staticmethod
    def pdf(z: float, sh) -> float: return K.sqrt(2.0 / _PI) * K.exp(-z * z / 2.0)
    @staticmethod
    def logpdf(z: float, sh) -> float: return 0.5 * K.log(2.0 / _PI) - z * z / 2.0
    @staticmethod
    def cdf(z: float, sh) -> float: return K.erf(z / K.sqrt(2.0))
    @staticmethod
    def ppf(q: float, sh) -> float: return _norm_ppf((1 + q) / 2.0)
    @staticmethod
    def sf(z: float, sh) -> float: return 2 * _norm_sf(z)
    @staticmethod
    def isf(q: float, sh) -> float: return _norm_isf(q / 2)
    @staticmethod
    def stats(sh):
        return (K.sqrt(2.0 / _PI), 1 - 2.0 / _PI, K.sqrt(2.0) * (4 - _PI) / (_PI - 2) ** 1.5,
                8 * (_PI - 3) / (_PI - 2) ** 2)
    @staticmethod
    def entropy(sh) -> float: return 0.5 * K.log(_PI / 2.0) + 0.5
    @staticmethod
    def fitc(data: np.ndarray[float, 1], fx: List[float]) -> List[float]:
        mn = float(data.min())
        loc = fx[0]
        if loc != loc:
            loc = mn
        elif mn < loc:
            raise ValueError("Invalid values in `data`.  Maximum likelihood estimation "
                             "with 'halfnorm' requires that floc <= data.min().")
        scale = fx[1]
        if scale != scale:
            scale = float(((data - loc) ** 2).mean()) ** 0.5
        return [loc, scale]

@tuple
class _HalfcauchyK:
    @staticmethod
    def nshape() -> Literal[int]: return 0
    @staticmethod
    def a(sh) -> float: return 0.0
    @staticmethod
    def pdf(z: float, sh) -> float: return 2.0 / _PI / (1.0 + z * z)
    @staticmethod
    def logpdf(z: float, sh) -> float: return K.log(2.0 / _PI) - K.log1p(z * z)
    @staticmethod
    def cdf(z: float, sh) -> float: return 2.0 / _PI * atan(z)
    @staticmethod
    def ppf(q: float, sh) -> float: return tan(_PI / 2 * q)
    @staticmethod
    def sf(z: float, sh) -> float: return 2.0 / _PI * atan2(1.0, z)
    @staticmethod
    def isf(q: float, sh) -> float: return 1.0 / tan(_PI * q / 2)
    @staticmethod
    def stats(sh): return (INF, INF, NAN, NAN)
    @staticmethod
    def entropy(sh) -> float: return K.log(2 * _PI)

# --------------------------------------------------------------- invgamma

@tuple
class _InvgammaK:
    @staticmethod
    def nshape() -> Literal[int]: return 1
    @staticmethod
    def a(sh) -> float: return 0.0
    @staticmethod
    def open_support() -> Literal[bool]: return True
    @staticmethod
    def logpdf(z: float, sh) -> float:
        a = sh[0]
        return -(a + 1) * K.log(z) - K.gammaln(a) - 1.0 / z
    @staticmethod
    def pdf(z: float, sh) -> float: return K.exp(_InvgammaK.logpdf(z, sh))
    @staticmethod
    def cdf(z: float, sh) -> float: return K.gammaincc(sh[0], 1.0 / z)
    @staticmethod
    def ppf(q: float, sh) -> float: return 1.0 / K.gammainccinv(sh[0], q)
    @staticmethod
    def sf(z: float, sh) -> float: return K.gammainc(sh[0], 1.0 / z)
    @staticmethod
    def isf(q: float, sh) -> float: return 1.0 / K.gammaincinv(sh[0], q)
    @staticmethod
    def stats(sh):
        a = sh[0]
        m1 = 1. / (a - 1.) if a > 1 else INF
        m2 = 1. / (a - 1.) ** 2 / (a - 2.) if a > 2 else INF
        g1 = 4. * K.sqrt(a - 2.) / (a - 3.) if a > 3 else NAN
        g2 = 6. * (5. * a - 11.) / (a - 3.) / (a - 4.) if a > 4 else NAN
        return (m1, m2, g1, g2)
    @staticmethod
    def entropy(sh) -> float:
        a = sh[0]
        if a >= 200:
            return ((1 - 3 * K.log(a) + K.log(2.0) + K.log(_PI)) / 2
                    + 2 / 3 * a ** -1. + a ** -2. / 12 - a ** -3. / 90 - a ** -4. / 120)
        return a - (a + 1.0) * K.digamma(a) + K.gammaln(a)

# -------------------------------------------------------- invgauss / wald

def _invgauss_logcdf(x: float, mu: float) -> float:
    if x <= 0.0:
        return -INF
    fac = 1 / K.sqrt(x)
    a = _norm_logcdf(fac * (x / mu - 1))
    b = 2 / mu + _norm_logcdf(-fac * (x / mu + 1))
    return a + K.log1p(K.exp(b - a))

def _invgauss_logsf(x: float, mu: float) -> float:
    if x <= 0.0:
        return 0.0
    fac = 1 / K.sqrt(x)
    a = _norm_logsf(fac * (x / mu - 1))
    b = 2 / mu + _norm_logcdf(-fac * (x / mu + 1))
    return a + K.log1p(-K.exp(b - a))

@tuple
class _InvgaussK:
    @staticmethod
    def nshape() -> Literal[int]: return 1
    @staticmethod
    def a(sh) -> float: return 0.0
    @staticmethod
    def open_support() -> Literal[bool]: return True
    @staticmethod
    def rvs(rs, sh, size): return rs.wald(sh[0], 1.0, size=size)
    @staticmethod
    def pdf(z: float, sh) -> float:
        mu = sh[0]
        if z <= 0.0:
            return 0.0
        return 1.0 / K.sqrt(2 * _PI * z ** 3.0) * K.exp(-1.0 / (2 * z) * (z / mu - 1) ** 2)
    @staticmethod
    def logpdf(z: float, sh) -> float:
        mu = sh[0]
        return -0.5 * K.log(2 * _PI) - 1.5 * K.log(z) - (z / mu - 1) ** 2 / (2 * z)
    @staticmethod
    def logcdf(z: float, sh) -> float: return _invgauss_logcdf(z, sh[0])
    @staticmethod
    def logsf(z: float, sh) -> float: return _invgauss_logsf(z, sh[0])
    @staticmethod
    def cdf(z: float, sh) -> float: return K.exp(_invgauss_logcdf(z, sh[0]))
    @staticmethod
    def sf(z: float, sh) -> float: return K.exp(_invgauss_logsf(z, sh[0]))
    @staticmethod
    def stats(sh):
        mu = sh[0]
        return (mu, mu ** 3.0, 3 * K.sqrt(mu), 15 * mu)
    @staticmethod
    def entropy(sh) -> float:
        mu = sh[0]
        a = 1. + K.log(2 * _PI) + 3 * K.log(mu)
        r = 2 / mu
        return 0.5 * a - 1.5 * G2.scaled_exp1(r) / r
    @staticmethod
    def fitc(data: np.ndarray[float, 1], fx: List[float]) -> List[float]:
        fmu = fx[0]
        floc = fx[1]
        fscale = fx[2]
        if floc != floc or fmu == fmu:
            return List[float]()
        if float(data.min()) - floc < 0:
            raise ValueError("Invalid values in `data`.  Maximum likelihood estimation "
                             "with 'invgauss' requires that 0 < (x - loc)/scale  < inf "
                             "for each x in `data`.")
        d = data - floc
        n = float(d.mean())
        if fscale != fscale:
            fscale = len(d) / float(np.sum(d ** -1.0 - n ** -1.0))
        return [n / fscale, floc, fscale]

@tuple
class _WaldK:
    @staticmethod
    def nshape() -> Literal[int]: return 0
    @staticmethod
    def a(sh) -> float: return 0.0
    @staticmethod
    def open_support() -> Literal[bool]: return True
    @staticmethod
    def rvs(rs, sh, size): return rs.wald(1.0, 1.0, size=size)
    @staticmethod
    def pdf(z: float, sh) -> float: return _InvgaussK.pdf(z, (1.0,))
    @staticmethod
    def logpdf(z: float, sh) -> float: return _InvgaussK.logpdf(z, (1.0,))
    @staticmethod
    def cdf(z: float, sh) -> float: return K.exp(_invgauss_logcdf(z, 1.0))
    @staticmethod
    def sf(z: float, sh) -> float: return K.exp(_invgauss_logsf(z, 1.0))
    @staticmethod
    def logcdf(z: float, sh) -> float: return _invgauss_logcdf(z, 1.0)
    @staticmethod
    def logsf(z: float, sh) -> float: return _invgauss_logsf(z, 1.0)
    @staticmethod
    def stats(sh): return (1.0, 1.0, 3.0, 15.0)
    @staticmethod
    def entropy(sh) -> float: return _InvgaussK.entropy((1.0,))

# ---------------------------------------------------------------- rice

@tuple
class _RiceK:
    @staticmethod
    def nshape() -> Literal[int]: return 1
    @staticmethod
    def a(sh) -> float: return 0.0
    @staticmethod
    def check(sh) -> bool: return sh[0] >= 0
    @staticmethod
    def rvs(rs, sh, size):
        sz = _tsize(size)
        t = sh[0] / K.sqrt(2.0) + rs.standard_normal(size=(2,) + sz)
        return np.sqrt((t * t).sum(axis=0))
    @staticmethod
    def cdf(z: float, sh) -> float: return NC.chndtr(z * z, 2.0, sh[0] * sh[0])
    @staticmethod
    def ppf(q: float, sh) -> float: return K.sqrt(NC.chndtrix(q, 2.0, sh[0] * sh[0]))
    @staticmethod
    def pdf(z: float, sh) -> float:
        b = sh[0]
        return z * K.exp(-(z - b) * (z - b) / 2.0) * K.i0e(z * b)
    @staticmethod
    def munp(n: float, sh) -> float:
        b = sh[0]
        nd2 = n / 2.0
        n1 = 1 + nd2
        b2 = b * b / 2.0
        return 2.0 ** nd2 * K.exp(-b2) * K.gamma(n1) * H.hyp1f1(n1, 1.0, b2)

# --------------------------------------------------------------- truncexpon

@tuple
class _TruncexponK:
    @staticmethod
    def nshape() -> Literal[int]: return 1
    @staticmethod
    def a(sh) -> float: return 0.0
    @staticmethod
    def b(sh) -> float: return sh[0]
    @staticmethod
    def pdf(z: float, sh) -> float: return K.exp(-z) / (-K.expm1(-sh[0]))
    @staticmethod
    def logpdf(z: float, sh) -> float: return -z - K.log(-K.expm1(-sh[0]))
    @staticmethod
    def cdf(z: float, sh) -> float: return K.expm1(-z) / K.expm1(-sh[0])
    @staticmethod
    def ppf(q: float, sh) -> float: return -K.log1p(q * K.expm1(-sh[0]))
    @staticmethod
    def sf(z: float, sh) -> float: return (K.exp(-sh[0]) - K.exp(-z)) / K.expm1(-sh[0])
    @staticmethod
    def isf(q: float, sh) -> float: return -K.log(K.exp(-sh[0]) - q * K.expm1(-sh[0]))
    @staticmethod
    def munp(n: float, sh) -> float:
        b = sh[0]
        if n == 1:
            return (1 - (b + 1) * K.exp(-b)) / (-K.expm1(-b))
        elif n == 2:
            return 2 * (1 - 0.5 * (b * b + 2 * b + 2) * K.exp(-b)) / (-K.expm1(-b))
        return _generic_munp(_TruncexponK, n, sh)
    @staticmethod
    def entropy(sh) -> float:
        b = sh[0]
        eB = K.exp(b)
        return K.log(eB - 1) + (1 + eB * (b - 1.0)) / (1.0 - eB)

# ------------------------------------------------------------------ triang

@tuple
class _TriangK:
    @staticmethod
    def nshape() -> Literal[int]: return 1
    @staticmethod
    def a(sh) -> float: return 0.0
    @staticmethod
    def b(sh) -> float: return 1.0
    @staticmethod
    def check(sh) -> bool: return sh[0] >= 0 and sh[0] <= 1
    @staticmethod
    def rvs(rs, sh, size): return rs.triangular(0.0, sh[0], 1.0, size)
    @staticmethod
    def pdf(x: float, sh) -> float:
        c = sh[0]
        if c == 0:
            return 2 - 2 * x
        if x < c:
            return 2 * x / c
        if c != 1:
            return 2 * (1 - x) / (1 - c)
        return 2 * x
    @staticmethod
    def cdf(x: float, sh) -> float:
        c = sh[0]
        if c == 0:
            return 2 * x - x * x
        if x < c:
            return x * x / c
        if c != 1:
            return (x * x - 2 * x + c) / (c - 1)
        return x * x
    @staticmethod
    def ppf(q: float, sh) -> float:
        c = sh[0]
        return K.sqrt(c * q) if q < c else 1 - K.sqrt((1 - c) * (1 - q))
    @staticmethod
    def stats(sh):
        c = sh[0]
        return ((c + 1.0) / 3.0, (1.0 - c + c * c) / 18,
                K.sqrt(2.0) * (2 * c - 1) * (c + 1) * (c - 2) / (5 * K.pow(1.0 - c + c * c, 1.5)),
                -3.0 / 5.0)
    @staticmethod
    def entropy(sh) -> float: return 0.5 - K.log(2.0)

norm = rv_continuous[_NormK]('norm')
t = rv_continuous[_TK]('t')
chi2 = rv_continuous[_Chi2K]('chi2')
f = rv_continuous[_FK]('f')
uniform = rv_continuous[_UniformK]('uniform')
expon = rv_continuous[_ExponK]('expon')
gamma = rv_continuous[_GammaK]('gamma')
beta = rv_continuous[_BetaK]('beta')
lognorm = rv_continuous[_LognormK]('lognorm')
laplace = rv_continuous[_LaplaceK]('laplace')
logistic = rv_continuous[_LogisticK]('logistic')
cauchy = rv_continuous[_CauchyK]('cauchy')
weibull_min = rv_continuous[_WeibullMinK]('weibull_min')
weibull_max = rv_continuous[_WeibullMaxK]('weibull_max')
gumbel_r = rv_continuous[_GumbelRK]('gumbel_r')
gumbel_l = rv_continuous[_GumbelLK]('gumbel_l')
pareto = rv_continuous[_ParetoK]('pareto')
lomax = rv_continuous[_LomaxK]('lomax')
chi = rv_continuous[_ChiK]('chi')
rayleigh = rv_continuous[_RayleighK]('rayleigh')
maxwell = rv_continuous[_MaxwellK]('maxwell')
halfnorm = rv_continuous[_HalfnormK]('halfnorm')
halfcauchy = rv_continuous[_HalfcauchyK]('halfcauchy')
invgamma = rv_continuous[_InvgammaK]('invgamma')
invgauss = rv_continuous[_InvgaussK]('invgauss')
wald = rv_continuous[_WaldK]('wald')
rice = rv_continuous[_RiceK]('rice')
truncexpon = rv_continuous[_TruncexponK]('truncexpon')
triang = rv_continuous[_TriangK]('triang')
