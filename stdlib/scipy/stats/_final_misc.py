# Copyright (C) 2026 the Typhon authors
# Ported from scipy 1.18.1 (BSD-3-Clause)

# Warning/error classes, kstat, kstatvar, gstd, bayes_mvs and the
# PermutationMethod / MonteCarloMethod / BootstrapMethod option holders.

import numpy as np
import scipy.special._kernels as K
from ._common import _f64
from ._continuous_distns import t as _t_dist, norm as _norm_dist, invgamma as _invgamma

class DegenerateDataWarning(RuntimeWarning):
    def __init__(self, msg: str = "Degenerate data encountered; results may not be reliable."):
        super().__init__(msg)

class ConstantInputWarning(DegenerateDataWarning):
    def __init__(self, msg: str = "All values in data are exactly equal; results may not be reliable."):
        super().__init__(msg)

class NearConstantInputWarning(DegenerateDataWarning):
    def __init__(self, msg: str = "All values in data are nearly equal; results may not be reliable."):
        super().__init__(msg)

class FitError(RuntimeError):
    def __init__(self, msg: str = "An error occurred when fitting a distribution to data."):
        super().__init__(msg)

class PermutationMethod:
    n_resamples: int
    batch: Optional[int]
    rng: Optional[int]
    def __init__(self, n_resamples: int = 9999, batch: Optional[int] = None, rng: Optional[int] = None):
        self.n_resamples = n_resamples
        self.batch = batch
        self.rng = rng

class MonteCarloMethod:
    n_resamples: int
    batch: Optional[int]
    rvs: Optional[object]
    rng: Optional[int]
    def __init__(self, n_resamples: int = 9999, batch: Optional[int] = None, rvs: Optional[object] = None,
                 rng: Optional[int] = None):
        self.n_resamples = n_resamples
        self.batch = batch
        self.rvs = rvs
        self.rng = rng

class BootstrapMethod:
    n_resamples: int
    batch: Optional[int]
    method: str
    rng: Optional[int]
    def __init__(self, method: str = 'BCa', n_resamples: int = 9999, batch: Optional[int] = None,
                 rng: Optional[int] = None):
        self.n_resamples = n_resamples
        self.batch = batch
        self.method = method
        self.rng = rng

def _kstat1(x: np.ndarray[float, 1], n: int) -> float:
    N = float(x.shape[0])
    s1 = float(np.sum(x))
    if n == 1:
        return s1 * 1.0 / N
    s2 = float(np.sum(x ** 2))
    if n == 2:
        return (N * s2 - s1 ** 2.0) / (N * (N - 1.0))
    s3 = float(np.sum(x ** 3))
    if n == 3:
        return (2 * s1 ** 3 - 3 * N * s1 * s2 + N * N * s3) / (N * (N - 1.0) * (N - 2.0))
    s4 = float(np.sum(x ** 4))
    return ((-6 * s1 ** 4 + 12 * N * s1 ** 2 * s2 - 3 * N * (N - 1.0) * s2 ** 2
             - 4 * N * (N + 1) * s1 * s3 + N * N * (N + 1) * s4)
            / (N * (N - 1.0) * (N - 2.0) * (N - 3.0)))

def kstat(data, n: int = 2) -> float:
    if n > 4 or n < 1:
        raise ValueError("k-statistics only supported for 1<=n<=4")
    return _kstat1(_f64(data).ravel(), n)

def kstatvar(data, n: int = 2) -> float:
    x = _f64(data).ravel()
    N = float(x.shape[0])
    if n == 1:
        return _kstat1(x, 2) * 1.0 / N
    elif n == 2:
        k2 = _kstat1(x, 2)
        k4 = _kstat1(x, 4)
        return (2 * N * k2 ** 2 + (N - 1) * k4) / (N * (N + 1))
    raise ValueError("Only n=1 or n=2 supported.")

def gstd(a, ddof: int = 1) -> float:
    x = np.log(_f64(a).ravel())
    n = x.shape[0]
    m = float(np.mean(x))
    v = float(np.sum((x - m) ** 2)) / (n - ddof)
    return K.exp(K.sqrt(v))

class BayesMvs:
    statistic: float
    low: float
    high: float
    def __init__(self, statistic: float, low: float, high: float):
        self.statistic = statistic
        self.low = low
        self.high = high
    def __getitem__(self, i: Literal[int]):
        if i == 0 or i == -2:
            return self.statistic
        elif i == 1 or i == -1:
            return (self.low, self.high)
        else:
            compile_error("index out of range")
    def __iter__(self):
        yield self.statistic
        yield (self.low, self.high)

def bayes_mvs(data, alpha: float = 0.90):
    if alpha >= 1 or alpha <= 0:
        raise ValueError(f"0 < alpha < 1 is required, but alpha={alpha} was given.")
    x = _f64(data).ravel()
    n = x.shape[0]
    if n < 2:
        raise ValueError("Need at least 2 data-points.")
    xbar = float(np.mean(x))
    C = float(np.var(x))
    lo = (1 - alpha) / 2
    hi = (1 + alpha) / 2
    if n > 1000:
        sc = K.sqrt(C / n)
        m = BayesMvs(xbar, _norm_dist.ppf(lo, loc=xbar, scale=sc), _norm_dist.ppf(hi, loc=xbar, scale=sc))
        sv = K.sqrt(2.0 / n) * C
        v = BayesMvs(C, _norm_dist.ppf(lo, loc=C, scale=sv), _norm_dist.ppf(hi, loc=C, scale=sv))
        ss = K.sqrt(C / (2. * n))
        s = BayesMvs(K.sqrt(C), _norm_dist.ppf(lo, loc=K.sqrt(C), scale=ss),
                     _norm_dist.ppf(hi, loc=K.sqrt(C), scale=ss))
        return (m, v, s)
    nm1 = n - 1
    fac = n * C / 2.
    val = nm1 / 2.
    sc = K.sqrt(C / nm1)
    m = BayesMvs(xbar, _t_dist.ppf(lo, nm1, loc=xbar, scale=sc), _t_dist.ppf(hi, nm1, loc=xbar, scale=sc))
    v = BayesMvs(float(_invgamma.mean(val, scale=fac)), _invgamma.ppf(lo, val, scale=fac),
                 _invgamma.ppf(hi, val, scale=fac))
    from ._continuous_distns3 import gengamma as _gengamma
    gs = K.sqrt(fac)
    s = BayesMvs(float(_gengamma.mean(val, -2.0, scale=gs)), _gengamma.ppf(lo, val, -2.0, scale=gs),
                 _gengamma.ppf(hi, val, -2.0, scale=gs))
    return (m, v, s)

def mvsdist(data):
    x = _f64(data).ravel()
    n = x.shape[0]
    if n < 2:
        raise ValueError("Need at least 2 data-points.")
    xbar = float(np.mean(x))
    C = float(np.var(x))
    from ._continuous_distns3 import gengamma as _gengamma
    if n > 1000:
        return (_norm_dist(loc=xbar, scale=K.sqrt(C / n)),
                _norm_dist(loc=C, scale=K.sqrt(2.0 / n) * C),
                _norm_dist(loc=K.sqrt(C), scale=K.sqrt(C / (2. * n))))
    nm1 = n - 1
    fac = n * C / 2.
    val = nm1 / 2.
    return (_t_dist(nm1, loc=xbar, scale=K.sqrt(C / nm1)),
            _invgamma(val, scale=fac),
            _gengamma(val, -2.0, scale=K.sqrt(fac)))
