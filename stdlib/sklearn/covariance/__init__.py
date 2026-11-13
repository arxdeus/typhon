# Ported from scikit-learn 1.9.1 (BSD-3-Clause)

import numpy as np
import scipy.linalg as sl

def empirical_covariance(X, *, assume_centered: bool = False):
    Xa = np.asarray(X, dtype=float)
    if assume_centered:
        return (Xa.T @ Xa) / float(Xa.shape[0])
    Xc = Xa - Xa.mean(0)
    return (Xc.T @ Xc) / float(Xa.shape[0])

def shrunk_covariance(emp_cov, shrinkage: float = 0.1):
    e = np.asarray(emp_cov, dtype=float)
    p = e.shape[0]
    mu = np.trace(e) / float(p)
    out = (1.0 - shrinkage) * e
    for i in range(p):
        out[i, i] += shrinkage * mu
    return out

def ledoit_wolf_shrinkage(X, assume_centered: bool = False, block_size: int = 1000) -> float:
    Xa = np.asarray(X, dtype=float)
    n, p = Xa.shape
    if p == 1:
        return 0.0
    Xc = Xa if assume_centered else Xa - Xa.mean(0)
    X2 = Xc ** 2
    emp_cov_trace = np.sum(X2, axis=0) / float(n)
    mu = np.sum(emp_cov_trace) / float(p)
    if p > block_size:
        raise NotImplementedError("ledoit_wolf_shrinkage: n_features > block_size")
    delta_ = float(np.sum((Xc.T @ Xc) ** 2)) / float(n) ** 2
    beta_ = float(np.sum(X2.T @ X2))
    beta = 1.0 / float(p * n) * (beta_ / float(n) - delta_)
    delta = delta_ - 2.0 * mu * emp_cov_trace.sum() + float(p) * mu ** 2
    delta /= float(p)
    beta = min(beta, delta)
    return 0.0 if beta == 0 else beta / delta

def ledoit_wolf(X, *, assume_centered: bool = False, block_size: int = 1000):
    Xa = np.asarray(X, dtype=float)
    p = Xa.shape[1]
    if p == 1:
        Xc = Xa if assume_centered else Xa - Xa.mean()
        return np.full((1, 1), float((Xc ** 2).mean())), 0.0
    sh = ledoit_wolf_shrinkage(Xa, assume_centered=assume_centered, block_size=block_size)
    emp = empirical_covariance(Xa, assume_centered=assume_centered)
    mu = np.sum(np.trace(emp)) / float(p)
    out = (1.0 - sh) * emp
    for i in range(p):
        out[i, i] += sh * mu
    return out, sh

def oas(X, *, assume_centered: bool = False):
    Xa = np.asarray(X, dtype=float)
    n, p = Xa.shape
    if p == 1:
        Xc = Xa if assume_centered else Xa - Xa.mean()
        return np.full((1, 1), float((Xc ** 2).mean())), 0.0
    emp = empirical_covariance(Xa, assume_centered=assume_centered)
    alpha = float(np.mean(emp ** 2))
    mu = np.trace(emp) / float(p)
    mu2 = mu ** 2
    num = alpha + mu2
    den = float(n + 1) * (alpha - mu2 / float(p))
    sh = 1.0 if den == 0 else min(num / den, 1.0)
    out = (1.0 - sh) * emp
    for i in range(p):
        out[i, i] += sh * mu
    return out, sh

class EmpiricalCovariance:
    store_precision: bool
    assume_centered: bool
    covariance_: np.ndarray[float, 2]
    precision_: np.ndarray[float, 2]
    location_: np.ndarray[float, 1]

    def __init__(self, *, store_precision: bool = True, assume_centered: bool = False):
        self.store_precision = store_precision
        self.assume_centered = assume_centered
        self.covariance_ = np.empty((0, 0))
        self.precision_ = np.empty((0, 0))
        self.location_ = np.empty(0)

    def _loc(self, Xa):
        self.location_ = np.zeros(Xa.shape[1]) if self.assume_centered else Xa.mean(0)

    def _set_covariance(self, c):
        self.covariance_ = c
        if self.store_precision:
            self.precision_ = sl.pinvh(c)

    def fit(self, X, y = None):
        Xa = np.asarray(X, dtype=float)
        self._loc(Xa)
        self._set_covariance(empirical_covariance(Xa, assume_centered=self.assume_centered))
        return self

    def get_precision(self):
        return self.precision_ if self.store_precision else sl.pinvh(self.covariance_)

    def mahalanobis(self, X):
        Xc = np.asarray(X, dtype=float) - self.location_
        P = self.get_precision()
        return np.sum((Xc @ P) * Xc, axis=1)

    def score(self, X, y = None) -> float:
        # Gaussian log-likelihood of X under the fitted model
        Xa = np.asarray(X, dtype=float)
        p = Xa.shape[1]
        test_cov = empirical_covariance(Xa - self.location_, assume_centered=True)
        P = self.get_precision()
        _, logdet = np.linalg.slogdet(P)
        return float(-0.5 * (np.sum(test_cov * P) - logdet + p * np.log(2 * np.pi)))

class ShrunkCovariance(EmpiricalCovariance):
    shrinkage: float

    def __init__(self, *, store_precision: bool = True, assume_centered: bool = False,
                 shrinkage: float = 0.1):
        super().__init__(store_precision=store_precision, assume_centered=assume_centered)
        self.shrinkage = shrinkage

    def fit(self, X, y = None):
        Xa = np.asarray(X, dtype=float)
        self._loc(Xa)
        emp = empirical_covariance(Xa, assume_centered=self.assume_centered)
        self._set_covariance(shrunk_covariance(emp, self.shrinkage))
        return self

class LedoitWolf(EmpiricalCovariance):
    block_size: int
    shrinkage_: float

    def __init__(self, *, store_precision: bool = True, assume_centered: bool = False,
                 block_size: int = 1000):
        super().__init__(store_precision=store_precision, assume_centered=assume_centered)
        self.block_size = block_size
        self.shrinkage_ = 0.0

    def fit(self, X, y = None):
        Xa = np.asarray(X, dtype=float)
        self._loc(Xa)
        c, s = ledoit_wolf(Xa - self.location_, assume_centered=True, block_size=self.block_size)
        self.shrinkage_ = s
        self._set_covariance(c)
        return self

class OAS(EmpiricalCovariance):
    shrinkage_: float

    def __init__(self, *, store_precision: bool = True, assume_centered: bool = False):
        super().__init__(store_precision=store_precision, assume_centered=assume_centered)
        self.shrinkage_ = 0.0

    def fit(self, X, y = None):
        Xa = np.asarray(X, dtype=float)
        self._loc(Xa)
        c, s = oas(Xa - self.location_, assume_centered=True)
        self.shrinkage_ = s
        self._set_covariance(c)
        return self

def log_likelihood(emp_cov, precision) -> float:
    from ..decomposition._extmath import fast_logdet
    p = precision.shape[0]
    ll = -float(np.sum(emp_cov * precision)) + fast_logdet(precision)
    ll -= float(p) * np.log(2 * np.pi)
    ll /= 2.0
    return ll

from ._robust_covariance import MinCovDet, EllipticEnvelope, fast_mcd
from ._graph_lasso import GraphicalLasso, GraphicalLassoCV, graphical_lasso, graphical_lasso_path
