# Copyright (C) 2026 the Typhon authors
# Ported from scipy 1.18.1 (BSD-3-Clause)

# New-style distribution objects (scipy.stats._distribution_infrastructure) for
# scalar parameters: Normal, Uniform, Logistic, Binomial, Mixture,
# order_statistic, truncate and the abs/exp/log transforms with methods
# pdf/pmf/logpdf/cdf/logcdf/ccdf/logccdf/icdf/iccdf/mean/variance/
# standard_deviation/median/support/sample. One tagged class stands for every
# distribution so that the transformations compose; make_distribution wraps an
# old-style rv_continuous (not composable).

import numpy as np
import scipy.special._kernels as K
from scipy.integrate import quad
from scipy.optimize import brentq
from ._distn_infrastructure import check_random_state, NAN, INF
from ._discrete_distns import binom

# kinds
_NORMAL = 0
_UNIFORM = 1
_LOGISTIC = 2
_BINOMIAL = 3
_MIXTURE = 4
_ORDER = 5
_TRUNC = 6
_ABS = 7
_EXP = 8
_LOG = 9

class _Fn:
    d: Optional[object]
    def __init__(self):
        self.d = None

class Distribution:
    kind: int
    p: List[float]
    kids: List[Distribution]
    discrete: bool

    def __init__(self, kind: int, p: List[float], kids: List[Distribution], discrete: bool = False):
        self.kind = kind
        self.p = p
        self.kids = kids
        self.discrete = discrete

    # --------------------------------------------------------- support
    def _lo(self) -> float:
        k = self.kind
        if k == _NORMAL or k == _LOGISTIC:
            return -INF
        elif k == _UNIFORM:
            return self.p[0]
        elif k == _BINOMIAL:
            return 0.0
        elif k == _MIXTURE:
            lo = INF
            for c in self.kids:
                lo = min(lo, c._lo())
            return lo
        elif k == _ORDER:
            return self.kids[0]._lo()
        elif k == _TRUNC:
            return max(self.p[0], self.kids[0]._lo())
        elif k == _ABS:
            return 0.0
        elif k == _EXP:
            return K.exp(self.kids[0]._lo())
        else:
            return K.log(max(self.kids[0]._lo(), 0.0)) if self.kids[0]._lo() > 0 else -INF

    def _hi(self) -> float:
        k = self.kind
        if k == _NORMAL or k == _LOGISTIC:
            return INF
        elif k == _UNIFORM:
            return self.p[1]
        elif k == _BINOMIAL:
            return self.p[0]
        elif k == _MIXTURE:
            hi = -INF
            for c in self.kids:
                hi = max(hi, c._hi())
            return hi
        elif k == _ORDER:
            return self.kids[0]._hi()
        elif k == _TRUNC:
            return min(self.p[1], self.kids[0]._hi())
        elif k == _ABS:
            return max(abs(self.kids[0]._lo()), abs(self.kids[0]._hi()))
        elif k == _EXP:
            return K.exp(self.kids[0]._hi())
        else:
            return K.log(self.kids[0]._hi())

    def support(self):
        return (self._lo(), self._hi())

    # ------------------------------------------------------------ pdf
    def _pdf(self, x: float) -> float:
        k = self.kind
        if k == _NORMAL:
            z = (x - self.p[0]) / self.p[1]
            return K.exp(-0.5 * z * z) / (2.5066282746310002 * self.p[1])
        elif k == _UNIFORM:
            a = self.p[0]
            b = self.p[1]
            return 1 / (b - a) if (x >= a and x <= b) else 0.0
        elif k == _LOGISTIC:
            z = (x - self.p[0]) / self.p[1]
            e = K.exp(-abs(z))
            return e / (self.p[1] * (1 + e) ** 2)
        elif k == _BINOMIAL:
            if K.floor(x) != x or x < 0 or x > self.p[0]:
                return 0.0
            return float(binom.pmf(x, self.p[0], self.p[1]))
        elif k == _MIXTURE:
            s = 0.0
            for i in range(len(self.kids)):
                s += self.p[i] * self.kids[i]._pdf(x)
            return s
        elif k == _ORDER:
            X = self.kids[0]
            r = self.p[0]
            n = self.p[1]
            if x < X._lo() or x > X._hi():
                return 0.0
            F = X._cdf(x)
            lc = K.gammaln(n + 1) - K.gammaln(r) - K.gammaln(n - r + 1)
            return K.exp(lc) * F ** (r - 1) * (1 - F) ** (n - r) * X._pdf(x)
        elif k == _TRUNC:
            X = self.kids[0]
            if x < self.p[0] or x > self.p[1]:
                return 0.0
            return X._pdf(x) / (X._cdf(self.p[1]) - X._cdf(self.p[0]))
        elif k == _ABS:
            X = self.kids[0]
            if x < 0:
                return 0.0
            return X._pdf(x) + X._pdf(-x)
        elif k == _EXP:
            if x <= 0:
                return 0.0
            return self.kids[0]._pdf(K.log(x)) / x
        else:
            return self.kids[0]._pdf(K.exp(x)) * K.exp(x)

    def pdf(self, x): return self._apply(0, x)
    def pmf(self, x): return self._apply(0, x)
    def logpdf(self, x): return self._apply(1, x)
    def logpmf(self, x): return self._apply(1, x)

    # ------------------------------------------------------------ cdf
    def _cdf(self, x: float) -> float:
        k = self.kind
        if k == _NORMAL:
            return K.ndtr((x - self.p[0]) / self.p[1])
        elif k == _UNIFORM:
            a = self.p[0]
            b = self.p[1]
            return 0.0 if x < a else (1.0 if x > b else (x - a) / (b - a))
        elif k == _LOGISTIC:
            return K.expit((x - self.p[0]) / self.p[1])
        elif k == _BINOMIAL:
            return float(binom.cdf(x, self.p[0], self.p[1]))
        elif k == _MIXTURE:
            s = 0.0
            for i in range(len(self.kids)):
                s += self.p[i] * self.kids[i]._cdf(x)
            return s
        elif k == _ORDER:
            X = self.kids[0]
            return K.betainc(self.p[0], self.p[1] - self.p[0] + 1, X._cdf(x))
        elif k == _TRUNC:
            X = self.kids[0]
            if x < self.p[0]:
                return 0.0
            if x > self.p[1]:
                return 1.0
            lo = X._cdf(self.p[0])
            return (X._cdf(x) - lo) / (X._cdf(self.p[1]) - lo)
        elif k == _ABS:
            X = self.kids[0]
            if x < 0:
                return 0.0
            return X._cdf(x) - X._cdf(-x) + X._pdf_atom(-x)
        elif k == _EXP:
            return 0.0 if x <= 0 else self.kids[0]._cdf(K.log(x))
        else:
            return self.kids[0]._cdf(K.exp(x))

    def _pdf_atom(self, x: float) -> float:
        # point mass at x (only discrete distributions have one)
        if self.discrete:
            return self._pdf(x)
        return 0.0

    def _ccdf(self, x: float) -> float:
        k = self.kind
        if k == _NORMAL:
            return K.ndtr(-(x - self.p[0]) / self.p[1])
        elif k == _LOGISTIC:
            return K.expit(-(x - self.p[0]) / self.p[1])
        elif k == _BINOMIAL:
            return float(binom.sf(x, self.p[0], self.p[1]))
        elif k == _ORDER:
            return K.betaincc(self.p[0], self.p[1] - self.p[0] + 1, self.kids[0]._cdf(x))
        return 1.0 - self._cdf(x)

    def cdf(self, x): return self._apply(2, x)
    def logcdf(self, x): return self._apply(3, x)
    def ccdf(self, x): return self._apply(4, x)
    def logccdf(self, x): return self._apply(5, x)

    # ----------------------------------------------------------- icdf
    def _icdf(self, q: float) -> float:
        k = self.kind
        if not (q >= 0 and q <= 1):
            return NAN
        if k == _NORMAL:
            return self.p[0] + self.p[1] * K.ndtri(q)
        elif k == _UNIFORM:
            return self.p[0] + q * (self.p[1] - self.p[0])
        elif k == _LOGISTIC:
            return self.p[0] + self.p[1] * K.logit(q)
        elif k == _BINOMIAL:
            return float(binom.ppf(q, self.p[0], self.p[1]))
        elif k == _ORDER:
            return self.kids[0]._icdf(K.betaincinv(self.p[0], self.p[1] - self.p[0] + 1, q))
        elif k == _TRUNC:
            X = self.kids[0]
            lo = X._cdf(self.p[0])
            hi = X._cdf(self.p[1])
            return X._icdf(lo + q * (hi - lo))
        elif k == _EXP:
            return K.exp(self.kids[0]._icdf(q))
        elif k == _LOG:
            return K.log(self.kids[0]._icdf(q))
        return self._solve(q)

    def _solve(self, q: float) -> float:
        # numerical inversion of the cdf over the support (mixture, abs)
        lo = self._lo()
        hi = self._hi()
        if q == 0:
            return lo
        if q == 1:
            return hi
        a = lo
        b = hi
        if K.isinf(a):
            a = -1.0
            while self._cdf(a) > q:
                a *= 2
        if K.isinf(b):
            b = 1.0
            while self._cdf(b) < q:
                b *= 2
        if self.discrete:
            x = K.ceil(a)
            while x < b and self._cdf(x) < q:
                x += 1
            return x
        return brentq(_CdfMinus(self, q), a, b, xtol=1e-14)

    def icdf(self, p): return self._apply(6, p)
    def iccdf(self, p): return self._apply(7, p)

    # -------------------------------------------------------- moments
    def mean(self) -> float:
        k = self.kind
        if k == _NORMAL:
            return self.p[0]
        elif k == _UNIFORM:
            return 0.5 * (self.p[0] + self.p[1])
        elif k == _LOGISTIC:
            return self.p[0]
        elif k == _BINOMIAL:
            return self.p[0] * self.p[1]
        elif k == _MIXTURE:
            s = 0.0
            for i in range(len(self.kids)):
                s += self.p[i] * self.kids[i].mean()
            return s
        return self._raw(1)

    def variance(self) -> float:
        k = self.kind
        if k == _NORMAL:
            return self.p[1] ** 2
        elif k == _UNIFORM:
            return (self.p[1] - self.p[0]) ** 2 / 12
        elif k == _LOGISTIC:
            return self.p[1] ** 2 * 3.141592653589793 ** 2 / 3
        elif k == _BINOMIAL:
            return self.p[0] * self.p[1] * (1 - self.p[1])
        elif k == _MIXTURE:
            m = self.mean()
            s = 0.0
            for i in range(len(self.kids)):
                s += self.p[i] * (self.kids[i].variance() + self.kids[i].mean() ** 2)
            return s - m * m
        m = self._raw(1)
        return self._central(2, m)

    def standard_deviation(self) -> float:
        return K.sqrt(self.variance())

    def median(self) -> float:
        return self._icdf(0.5)

    def _raw(self, order: int) -> float:
        return quad(_Moment(self, 0.0, order), self._lo(), self._hi())[0]

    def _central(self, order: int, c: float) -> float:
        return quad(_Moment(self, c, order), self._lo(), self._hi())[0]

    def _apply(self, op: int, x):
        if isinstance(x, np.ndarray) or isinstance(x, List) or isinstance(x, Tuple):
            a = np.asarray(x).astype(float)
            out = np.empty(a.shape, dtype=float)
            for idx in np.ndindex(a.shape):
                out[idx] = self._eval(op, float(a[idx]))
            return out
        return self._eval(op, float(x))

    def _eval(self, op: int, x: float) -> float:
        if x != x:
            return NAN
        if op == 0:
            return self._pdf(x)
        elif op == 1:
            return K.log(self._pdf(x))
        elif op == 2:
            return self._cdf(x)
        elif op == 3:
            return K.log(self._cdf(x))
        elif op == 4:
            return self._ccdf(x)
        elif op == 5:
            return K.log(self._ccdf(x))
        elif op == 6:
            return self._icdf(x)
        else:
            return self._icdf(1 - x)

    def sample(self, shape = (), rng = None):
        rs = check_random_state(rng)
        if isinstance(shape, int):
            n = shape
        else:
            n = 1
            for s in shape:
                n *= int(s)
        u = rs.uniform(size=n)
        out = np.empty(n, dtype=float)
        for i in range(n):
            out[i] = self._icdf(float(u[i]))
        if isinstance(shape, int):
            return out
        if len(shape) == 0:
            return out[0]
        return out.reshape(shape)

class _CdfMinus:
    d: Distribution
    q: float
    def __init__(self, d: Distribution, q: float):
        self.d = d
        self.q = q
    def __call__(self, x: float) -> float:
        return self.d._cdf(x) - self.q

class _Moment:
    d: Distribution
    c: float
    order: int
    def __init__(self, d: Distribution, c: float, order: int):
        self.d = d
        self.c = c
        self.order = order
    def __call__(self, x: float) -> float:
        return (x - self.c) ** self.order * self.d._pdf(x)

# ---------------------------------------------------------- constructors

def Normal(mu = 0.0, sigma = 1.0):
    if not (float(sigma) > 0):
        raise ValueError("Parameter `sigma` must be positive.")
    return Distribution(_NORMAL, [float(mu), float(sigma)], List[Distribution]())

def Uniform(*, a = 0.0, b = 1.0):
    if not (float(b) > float(a)):
        raise ValueError("Parameter `b` must be greater than `a`.")
    return Distribution(_UNIFORM, [float(a), float(b)], List[Distribution]())

def Logistic(*, mu = 0.0, s = 1.0):
    if not (float(s) > 0):
        raise ValueError("Parameter `s` must be positive.")
    return Distribution(_LOGISTIC, [float(mu), float(s)], List[Distribution]())

def Binomial(*, n, p):
    if not (float(n) >= 0 and float(p) >= 0 and float(p) <= 1):
        raise ValueError("Invalid parameters for Binomial.")
    return Distribution(_BINOMIAL, [float(n), float(p)], List[Distribution](), True)

def Mixture(components, *, weights = None):
    kids = List[Distribution]()
    for c in components:
        kids.append(c)
    n = len(kids)
    if n == 0:
        raise ValueError("`components` must contain at least one distribution.")
    w = [1.0 / n for _ in range(n)]
    if weights is not None:
        ww = np.asarray(weights).astype(float).ravel()
        if ww.shape[0] != n:
            raise ValueError("`components` and `weights` must have the same length.")
        s = float(np.sum(ww))
        for i in range(n):
            w[i] = float(ww[i]) / s
    disc = kids[0].discrete
    for c in kids:
        if c.discrete != disc:
            raise ValueError("All components must be either continuous or discrete.")
    return Distribution(_MIXTURE, w, kids, disc)

def order_statistic(X, /, *, r, n):
    if not (int(r) >= 1 and int(r) <= int(n)):
        raise ValueError("`r` must satisfy 1 <= r <= n.")
    kids = List[Distribution]()
    kids.append(X)
    return Distribution(_ORDER, [float(r), float(n)], kids)

def truncate(X, lb = -INF, ub = INF):
    if not (float(lb) < float(ub)):
        raise ValueError("Parameter `lb` must be less than `ub`.")
    kids = List[Distribution]()
    kids.append(X)
    return Distribution(_TRUNC, [float(lb), float(ub)], kids, X.discrete)

def abs(X, /):
    kids = List[Distribution]()
    kids.append(X)
    return Distribution(_ABS, List[float](), kids, X.discrete)

def exp(X, /):
    kids = List[Distribution]()
    kids.append(X)
    return Distribution(_EXP, List[float](), kids)

def log(X, /):
    kids = List[Distribution]()
    kids.append(X)
    return Distribution(_LOG, List[float](), kids)

# ------------------------------------------------------- make_distribution

class ParamDistribution[D]:
    # wraps an old-style rv_continuous / rv_discrete instance (shape args fixed
    # at construction)
    dist: D
    args: Tuple
    def __init__(self, dist: D, args):
        self.dist = dist
        self.args = args
    def pdf(self, x): return self.dist.pdf(x, *self.args)
    def logpdf(self, x): return self.dist.logpdf(x, *self.args)
    def cdf(self, x): return self.dist.cdf(x, *self.args)
    def logcdf(self, x): return self.dist.logcdf(x, *self.args)
    def ccdf(self, x): return self.dist.sf(x, *self.args)
    def logccdf(self, x): return self.dist.logsf(x, *self.args)
    def icdf(self, p): return self.dist.ppf(p, *self.args)
    def iccdf(self, p): return self.dist.isf(p, *self.args)
    def mean(self): return self.dist.mean(*self.args)
    def variance(self): return self.dist.var(*self.args)
    def standard_deviation(self): return self.dist.std(*self.args)
    def median(self): return self.dist.median(*self.args)
    def support(self): return self.dist.support(*self.args)
    def sample(self, shape = (), rng = None):
        return self.dist.rvs(*self.args, size=shape if len(shape) > 0 else None, random_state=rng)

class _MadeDistribution[D]:
    dist: D
    def __init__(self, dist: D):
        self.dist = dist
    def __call__(self, *args):
        return ParamDistribution(self.dist, args)

def make_distribution(dist):
    return _MadeDistribution(dist)
