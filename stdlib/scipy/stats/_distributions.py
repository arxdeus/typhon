# Ported from scipy 1.18.1 (BSD-3-Clause)

# Continuous distributions: norm, t, f, chi2, uniform.
#
# Both styles are supported:
#     norm.cdf(x, loc, scale)       t.ppf(q, df, loc=0, scale=1)
#     rv = t(df, loc, scale); rv.cdf(x)
# Scalar arguments give a float, array arguments a broadcast float64 ndarray.

import numpy as np
import internal.static as static
import scipy.special._kernels as K
from ._common import _is_arr, _anyarr

NAN = K.NAN
INF = K.INF
_LOG_SQRT_2PI = 0.91893853320467274178

@tuple
class _NormK:
    @staticmethod
    def nshape() -> Literal[int]: return 0
    @staticmethod
    def a(sh) -> float: return -INF
    @staticmethod
    def b(sh) -> float: return INF
    @staticmethod
    def check(sh) -> bool: return True
    @staticmethod
    def pdf(z: float, sh) -> float: return K.exp(-z * z / 2.0) / 2.5066282746310002
    @staticmethod
    def logpdf(z: float, sh) -> float: return -z * z / 2.0 - _LOG_SQRT_2PI
    @staticmethod
    def cdf(z: float, sh) -> float: return K.ndtr(z)
    @staticmethod
    def sf(z: float, sh) -> float: return K.ndtr(-z)
    @staticmethod
    def logcdf(z: float, sh) -> float: return K.log_ndtr(z)
    @staticmethod
    def logsf(z: float, sh) -> float: return K.log_ndtr(-z)
    @staticmethod
    def ppf(q: float, sh) -> float: return K.ndtri(q)
    @staticmethod
    def isf(q: float, sh) -> float: return -K.ndtri(q)
    @staticmethod
    def mean(sh) -> float: return 0.0
    @staticmethod
    def var(sh) -> float: return 1.0

@tuple
class _TK:
    @staticmethod
    def nshape() -> Literal[int]: return 1
    @staticmethod
    def a(sh) -> float: return -INF
    @staticmethod
    def b(sh) -> float: return INF
    @staticmethod
    def check(sh) -> bool: return sh[0] > 0
    @staticmethod
    def logpdf(z: float, sh) -> float:
        df = sh[0]
        if K.isinf(df):
            return _NormK.logpdf(z, ())
        return (K.gammaln(0.5 * df + 0.5) - K.gammaln(0.5 * df)
                - 0.5 * (K.log(df) + K.log(K.pi))
                - (df + 1) / 2 * K.log1p(z * z / df))
    @staticmethod
    def pdf(z: float, sh) -> float: return K.exp(_TK.logpdf(z, sh))
    @staticmethod
    def cdf(z: float, sh) -> float: return K.stdtr(sh[0], z)
    @staticmethod
    def sf(z: float, sh) -> float: return K.stdtr(sh[0], -z)
    @staticmethod
    def logcdf(z: float, sh) -> float: return K.log(_TK.cdf(z, sh))
    @staticmethod
    def logsf(z: float, sh) -> float: return K.log(_TK.sf(z, sh))
    @staticmethod
    def ppf(q: float, sh) -> float: return K.stdtrit(sh[0], q)
    @staticmethod
    def isf(q: float, sh) -> float: return -K.stdtrit(sh[0], q)
    @staticmethod
    def mean(sh) -> float: return 0.0 if sh[0] > 1 else (INF if False else NAN) if sh[0] <= 1 else 0.0
    @staticmethod
    def var(sh) -> float:
        df = sh[0]
        if K.isinf(df):
            return 1.0
        if df > 2:
            return df / (df - 2.0)
        if df > 1:
            return INF
        return NAN

@tuple
class _Chi2K:
    @staticmethod
    def nshape() -> Literal[int]: return 1
    @staticmethod
    def a(sh) -> float: return 0.0
    @staticmethod
    def b(sh) -> float: return INF
    @staticmethod
    def check(sh) -> bool: return sh[0] > 0
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
    def logcdf(z: float, sh) -> float: return K.log(_Chi2K.cdf(z, sh))
    @staticmethod
    def logsf(z: float, sh) -> float: return K.log(_Chi2K.sf(z, sh))
    @staticmethod
    def ppf(q: float, sh) -> float: return 2 * K.gammaincinv(sh[0] / 2, q)
    @staticmethod
    def isf(q: float, sh) -> float: return K.chdtri(sh[0], q)
    @staticmethod
    def mean(sh) -> float: return sh[0]
    @staticmethod
    def var(sh) -> float: return 2 * sh[0]

@tuple
class _FK:
    @staticmethod
    def nshape() -> Literal[int]: return 2
    @staticmethod
    def a(sh) -> float: return 0.0
    @staticmethod
    def b(sh) -> float: return INF
    @staticmethod
    def check(sh) -> bool: return sh[0] > 0 and sh[1] > 0
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
    def logcdf(z: float, sh) -> float: return K.log(_FK.cdf(z, sh))
    @staticmethod
    def logsf(z: float, sh) -> float: return K.log(_FK.sf(z, sh))
    @staticmethod
    def ppf(q: float, sh) -> float: return K.fdtri(sh[0], sh[1], q)
    @staticmethod
    def isf(q: float, sh) -> float: return K.fdtri(sh[0], sh[1], 1.0 - q)
    @staticmethod
    def mean(sh) -> float:
        v2 = sh[1]
        return v2 / (v2 - 2.0) if v2 > 2 else INF
    @staticmethod
    def var(sh) -> float:
        v1 = sh[0]
        v2 = sh[1]
        if v2 > 4:
            return 2 * v2 * v2 * (v1 + v2 - 2) / (v1 * (v2 - 2) ** 2 * (v2 - 4))
        return INF

@tuple
class _UniformK:
    @staticmethod
    def nshape() -> Literal[int]: return 0
    @staticmethod
    def a(sh) -> float: return 0.0
    @staticmethod
    def b(sh) -> float: return 1.0
    @staticmethod
    def check(sh) -> bool: return True
    @staticmethod
    def pdf(z: float, sh) -> float: return 1.0
    @staticmethod
    def logpdf(z: float, sh) -> float: return 0.0
    @staticmethod
    def cdf(z: float, sh) -> float: return z
    @staticmethod
    def sf(z: float, sh) -> float: return 1.0 - z
    @staticmethod
    def logcdf(z: float, sh) -> float: return K.log(z)
    @staticmethod
    def logsf(z: float, sh) -> float: return K.log1p(-z)
    @staticmethod
    def ppf(q: float, sh) -> float: return q
    @staticmethod
    def isf(q: float, sh) -> float: return 1.0 - q
    @staticmethod
    def mean(sh) -> float: return 0.5
    @staticmethod
    def var(sh) -> float: return 1.0 / 12.0

def _eval(Kr: type, op: Literal[str], x: float, loc: float, scale: float, sh) -> float:
    if not (scale > 0) or not Kr.check(sh):
        return NAN
    if op == 'ppf' or op == 'isf':
        q = x
        if not (q >= 0 and q <= 1):
            return NAN
        lo = Kr.a(sh) * scale + loc
        hi = Kr.b(sh) * scale + loc
        if op == 'ppf':
            if q == 0:
                return lo
            if q == 1:
                return hi
            return Kr.ppf(q, sh) * scale + loc
        else:
            if q == 0:
                return hi
            if q == 1:
                return lo
            return Kr.isf(q, sh) * scale + loc
    if x != x:
        return NAN
    z = (x - loc) / scale
    a = Kr.a(sh)
    b = Kr.b(sh)
    if op == 'pdf':
        if z < a or z > b:
            return 0.0
        return Kr.pdf(z, sh) / scale
    elif op == 'logpdf':
        if z < a or z > b:
            return -INF
        return Kr.logpdf(z, sh) - K.log(scale)
    elif op == 'cdf':
        if z <= a:
            return 0.0 if (z < a or a == -INF) else Kr.cdf(z, sh)
        if z >= b:
            return 1.0
        return Kr.cdf(z, sh)
    elif op == 'sf':
        if z < a:
            return 1.0
        if z >= b:
            return 0.0
        return Kr.sf(z, sh)
    elif op == 'logcdf':
        if z < a:
            return -INF
        if z >= b:
            return 0.0
        return Kr.logcdf(z, sh)
    elif op == 'logsf':
        if z < a:
            return 0.0
        if z >= b:
            return -INF
        return Kr.logsf(z, sh)
    else:
        compile_error("unknown distribution method")

def _bcast_eval(Kr: type, op: Literal[str], x, loc, scale, sh):
    if _anyarr(x, loc, scale, *sh):
        arrs = (np.asarray(x), np.asarray(loc), np.asarray(scale)) + tuple(np.asarray(s) for s in sh)
        shp = np.broadcast_shapes(*tuple(a.shape for a in arrs))
        bs = tuple(np.broadcast_to(a, shp) for a in arrs)
        out = np.empty(shp, dtype=float)
        for idx in np.ndindex(shp):
            vals = tuple(float(b[idx]) for b in bs)
            out[idx] = _eval(Kr, op, vals[0], vals[1], vals[2], vals[3:])
        return out
    else:
        return _eval(Kr, op, float(x), float(loc), float(scale), tuple(float(s) for s in sh))

def _split(Kr: type, args, loc, scale):
    NS: Literal[int] = Kr.nshape()
    if static.len(args) < NS:
        compile_error("missing shape parameter(s) for distribution")
    sh = args[:NS]
    extra = args[NS:]
    if static.len(extra) > 2:
        compile_error("too many positional arguments")
    if static.len(extra) >= 1:
        l = extra[0]
    else:
        l = loc
    if static.len(extra) == 2:
        s = extra[1]
    else:
        s = scale
    return sh, l, s

def _mom1(Kr: type, which: Literal[str], l: float, s: float, shv) -> float:
    if not (s > 0) or not Kr.check(shv):
        return NAN
    if which == 'mean':
        return Kr.mean(shv) * s + l
    elif which == 'var':
        return Kr.var(shv) * s * s
    else:
        return K.sqrt(Kr.var(shv)) * s

def _moment_eval(Kr: type, which: Literal[str], loc, scale, sh):
    if _anyarr(loc, scale, *sh):
        arrs = (np.asarray(loc), np.asarray(scale)) + tuple(np.asarray(v) for v in sh)
        shp = np.broadcast_shapes(*tuple(a.shape for a in arrs))
        bs = tuple(np.broadcast_to(a, shp) for a in arrs)
        out = np.empty(shp, dtype=float)
        for idx in np.ndindex(shp):
            vals = tuple(float(b[idx]) for b in bs)
            out[idx] = _mom1(Kr, which, vals[0], vals[1], vals[2:])
        return out
    else:
        return _mom1(Kr, which, float(loc), float(scale), tuple(float(v) for v in sh))

class rv_frozen[Kr, S, L, C]:
    """Frozen distribution: ``norm(loc, scale)``, ``t(df)``, ..."""
    sh: S
    loc: L
    scale: C

    def __init__(self, sh: S, loc: L, scale: C):
        self.sh = sh
        self.loc = loc
        self.scale = scale

    def pdf(self, x): return _bcast_eval(Kr, 'pdf', x, self.loc, self.scale, self.sh)
    def logpdf(self, x): return _bcast_eval(Kr, 'logpdf', x, self.loc, self.scale, self.sh)
    def cdf(self, x): return _bcast_eval(Kr, 'cdf', x, self.loc, self.scale, self.sh)
    def logcdf(self, x): return _bcast_eval(Kr, 'logcdf', x, self.loc, self.scale, self.sh)
    def sf(self, x): return _bcast_eval(Kr, 'sf', x, self.loc, self.scale, self.sh)
    def logsf(self, x): return _bcast_eval(Kr, 'logsf', x, self.loc, self.scale, self.sh)
    def ppf(self, q): return _bcast_eval(Kr, 'ppf', q, self.loc, self.scale, self.sh)
    def isf(self, q): return _bcast_eval(Kr, 'isf', q, self.loc, self.scale, self.sh)
    def mean(self): return _moment_eval(Kr, 'mean', self.loc, self.scale, self.sh)
    def var(self): return _moment_eval(Kr, 'var', self.loc, self.scale, self.sh)
    def std(self): return _moment_eval(Kr, 'std', self.loc, self.scale, self.sh)
    def median(self): return self.ppf(0.5)
    def interval(self, confidence):
        c = confidence
        return (self.ppf((1.0 - c) / 2.0), self.ppf((1.0 + c) / 2.0))

class rv_continuous[Kr]:
    name: str

    def __init__(self, name: str):
        self.name = name

    def __call__(self, *args, loc = 0.0, scale = 1.0):
        sh, l, s = _split(Kr, args, loc, scale)
        return rv_frozen[Kr, type(sh), type(l), type(s)](sh, l, s)

    def pdf(self, x, *args, loc = 0.0, scale = 1.0):
        sh, l, s = _split(Kr, args, loc, scale)
        return _bcast_eval(Kr, 'pdf', x, l, s, sh)
    def logpdf(self, x, *args, loc = 0.0, scale = 1.0):
        sh, l, s = _split(Kr, args, loc, scale)
        return _bcast_eval(Kr, 'logpdf', x, l, s, sh)
    def cdf(self, x, *args, loc = 0.0, scale = 1.0):
        sh, l, s = _split(Kr, args, loc, scale)
        return _bcast_eval(Kr, 'cdf', x, l, s, sh)
    def logcdf(self, x, *args, loc = 0.0, scale = 1.0):
        sh, l, s = _split(Kr, args, loc, scale)
        return _bcast_eval(Kr, 'logcdf', x, l, s, sh)
    def sf(self, x, *args, loc = 0.0, scale = 1.0):
        sh, l, s = _split(Kr, args, loc, scale)
        return _bcast_eval(Kr, 'sf', x, l, s, sh)
    def logsf(self, x, *args, loc = 0.0, scale = 1.0):
        sh, l, s = _split(Kr, args, loc, scale)
        return _bcast_eval(Kr, 'logsf', x, l, s, sh)
    def ppf(self, q, *args, loc = 0.0, scale = 1.0):
        sh, l, s = _split(Kr, args, loc, scale)
        return _bcast_eval(Kr, 'ppf', q, l, s, sh)
    def isf(self, q, *args, loc = 0.0, scale = 1.0):
        sh, l, s = _split(Kr, args, loc, scale)
        return _bcast_eval(Kr, 'isf', q, l, s, sh)
    def mean(self, *args, loc = 0.0, scale = 1.0):
        sh, l, s = _split(Kr, args, loc, scale)
        return _moment_eval(Kr, 'mean', l, s, sh)
    def var(self, *args, loc = 0.0, scale = 1.0):
        sh, l, s = _split(Kr, args, loc, scale)
        return _moment_eval(Kr, 'var', l, s, sh)
    def std(self, *args, loc = 0.0, scale = 1.0):
        sh, l, s = _split(Kr, args, loc, scale)
        return _moment_eval(Kr, 'std', l, s, sh)
    def median(self, *args, loc = 0.0, scale = 1.0):
        return self.ppf(0.5, *args, loc=loc, scale=scale)
    def interval(self, confidence, *args, loc = 0.0, scale = 1.0):
        c = confidence
        return (self.ppf((1.0 - c) / 2.0, *args, loc=loc, scale=scale),
                self.ppf((1.0 + c) / 2.0, *args, loc=loc, scale=scale))

norm = rv_continuous[_NormK]('norm')
t = rv_continuous[_TK]('t')
chi2 = rv_continuous[_Chi2K]('chi2')
f = rv_continuous[_FK]('f')
uniform = rv_continuous[_UniformK]('uniform')
