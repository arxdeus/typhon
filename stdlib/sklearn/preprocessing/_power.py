# Copyright (C) 2026 the Typhon authors
# Ported from scikit-learn 1.9.1 (BSD-3-Clause)

# PowerTransformer / power_transform. Lambdas are found as scipy.stats does:
# Box-Cox by optimize.brent(brack=(-2, 2)) on -boxcox_llf, Yeo-Johnson by
# optimize.fminbound on -yeojohnson_llf with scipy's overflow-safe bounds.

from ..base import estimator_repr as _estimator_repr
import numpy as np
import math
from ._data import _as2df, _is_constant_feature, StandardScaler

_EPS = 2.220446049250313e-16
_TINY = 2.2250738585072014e-308
_FMAX = 1.7976931348623157e308

def _lse_pairs(a: List[float], b: List[float]) -> Tuple[float, float]:
    # scipy.special._logsumexp._logsumexp for a 1-d vector (with weights b)
    m_ = -math.inf
    for v in a:
        if v > m_:
            m_ = v
    amax = m_
    na = len(a)
    mt = np.zeros(na)
    st = np.zeros(na)
    for i in range(na):
        if a[i] == amax:
            mt[i] = b[i]
        else:
            st[i] = b[i] * math.exp(a[i] - amax)
    m = float(np.sum(mt))
    s = float(np.sum(st))
    if s != 0.0:
        s = s / m
    sgn = (1.0 if s + 1.0 > 0 else (-1.0 if s + 1.0 < 0 else 0.0)) * (1.0 if m > 0 else (-1.0 if m < 0 else 0.0))
    if s < -1.0:
        s = -s - 2.0
    m = abs(m)
    return math.log1p(s) + (math.log(m) if m > 0.0 else -math.inf) + amax, sgn

def _log_var(logx: List[float]) -> float:
    n = len(logx)
    ones = [1.0] * n
    logmean = _lse_pairs(logx, ones)[0] - math.log(float(n))
    lm = List[float]()
    for v in logx:
        lm.append(2.0 * _lse_pairs([v, logmean], [1.0, -1.0])[0])
    return _lse_pairs(lm, ones)[0] - math.log(float(n))

def _var(x: List[float]) -> float:
    return float(np.var(np.array(x)))

def _yj_scalar(x: float, lmbda: float) -> float:
    if x >= 0:
        if abs(lmbda) < _EPS:
            return math.log1p(x)
        return math.expm1(lmbda * math.log1p(x)) / lmbda
    if abs(lmbda - 2.0) > _EPS:
        return -math.expm1((2.0 - lmbda) * math.log1p(-x)) / (2.0 - lmbda)
    return -math.log1p(-x)

def _yeojohnson_llf(lmb: float, data: np.ndarray[float, 1]) -> float:
    n = data.shape[0]
    allpos = True
    allneg = True
    for v in data:
        if v >= 0:
            allneg = False
        else:
            allpos = False
    logvar = 0.0
    if allpos:
        if abs(lmb) < _EPS:
            logvar = math.log(_var([math.log1p(v) for v in data]))
        else:
            logvar = _log_var([lmb * math.log1p(v) for v in data]) - 2.0 * math.log(abs(lmb))
    elif allneg:
        if abs(lmb - 2.0) < _EPS:
            logvar = math.log(_var([math.log1p(-v) for v in data]))
        else:
            logvar = _log_var([(2.0 - lmb) * math.log1p(-v) for v in data]) - 2.0 * math.log(abs(2.0 - lmb))
    else:
        sig = _var([_yj_scalar(v, lmb) for v in data])
        logvar = math.log(sig) if sig >= _TINY else -math.inf
    acc = np.empty(n)
    for i in range(n):
        v = data[i]
        sg = 1.0 if v > 0 else (-1.0 if v < 0 else 0.0)
        acc[i] = sg * math.log1p(abs(v))
    return -float(n) / 2.0 * logvar + (lmb - 1.0) * float(np.sum(acc))

def _neg_yj(lmb: float, data: np.ndarray[float, 1]) -> float:
    llf = _yeojohnson_llf(lmb, data)
    if math.isinf(llf):
        llf = -math.inf
    return -llf

def _boxcox_llf(lmb: float, data: np.ndarray[float, 1]) -> float:
    n = data.shape[0]
    logdata = np.log(data)
    logvar = 0.0
    if lmb == 0.0:
        logvar = math.log(float(np.var(logdata)))
    else:
        logvar = _log_var([lmb * v for v in logdata]) - 2.0 * math.log(abs(lmb))
    return (lmb - 1.0) * float(np.sum(logdata)) - float(n) / 2.0 * logvar

def _neg_bc(lmb: float, data: np.ndarray[float, 1]) -> float:
    return -_boxcox_llf(lmb, data)

def _yeojohnson_normmax(x: np.ndarray[float, 1]) -> float:
    from scipy.optimize import fminbound
    if np.all(x == 0.0):
        return 1.0
    log1p_max_x = math.log1p(20.0 * float(np.max(np.abs(x))))
    log_eps = math.log(_EPS)
    log_tiny = (math.log(_TINY) - log_eps) / 2.0
    log_max = (math.log(_FMAX) + log_eps) / 2.0
    lb = log_tiny / log1p_max_x
    ub = log_max / log1p_max_x
    if np.all(x < 0.0):
        lb, ub = 2.0 - ub, 2.0 - lb
    elif np.any(x < 0.0):
        lb, ub = max(2.0 - ub, lb), min(2.0 - lb, ub)
    return float(fminbound(_neg_yj, lb, ub, args=(x,), xtol=1.48e-08))

def _boxcox_normmax(x: np.ndarray[float, 1]) -> float:
    """method='mle'; the ymax overflow constraint (lambertw) is not applied."""
    from scipy.optimize import brent
    return float(brent(_neg_bc, args=(x,), brack=[-2.0, 2.0]))

def _nonnan(col):
    return col[~np.isnan(col)]

class PowerTransformer:
    def __repr__(self) -> str:
        return _estimator_repr(self)

    method: str
    standardize: bool
    copy: bool
    lambdas_: np.ndarray[float, 1]
    _scaler: StandardScaler
    n_features_in_: int

    def __init__(self, method: str = "yeo-johnson", *, standardize: bool = True, copy: bool = True):
        self.method = method
        self.standardize = standardize
        self.copy = copy
        self.lambdas_ = np.empty(0)
        self._scaler = StandardScaler()
        self.n_features_in_ = 0

    def _check(self, Xa, in_fit: bool):
        if self.method != "box-cox" and self.method != "yeo-johnson":
            raise ValueError(f"The 'method' parameter must be 'box-cox' or 'yeo-johnson', got {self.method!r}")
        if in_fit and self.method == "box-cox" and float(np.nanmin(Xa)) <= 0.0:
            raise ValueError("The Box-Cox transformation can only be applied to strictly positive data")
        if not in_fit and Xa.shape[1] != self.n_features_in_:
            raise ValueError(f"Input data has a different number of features than fitting data. Should have {self.n_features_in_}, data has {Xa.shape[1]}")

    def _tf_col(self, col, lmbda: float):
        out = col.copy()
        from scipy.special import boxcox
        for i in range(col.shape[0]):
            v = col[i]
            if v != v:
                continue
            out[i] = float(boxcox(v, lmbda)) if self.method == "box-cox" else _yj_scalar(v, lmbda)
        return out

    def _fit(self, X, force_transform: bool):
        Xa = _as2df(X).copy()
        self._check(Xa, True)
        n, p = Xa.shape
        self.n_features_in_ = p
        mean = np.mean(Xa, axis=0)
        var = np.var(Xa, axis=0)
        self.lambdas_ = np.empty(p)
        for i in range(p):
            col = np.ascontiguousarray(Xa[:, i])
            if self.method == "yeo-johnson" and _is_constant_feature(var[i], mean[i], float(n)):
                self.lambdas_[i] = 1.0
                continue
            if self.method == "box-cox":
                if np.all(np.isnan(col)):
                    raise ValueError("Column must not be all nan.")
                self.lambdas_[i] = _boxcox_normmax(_nonnan(col))
            else:
                self.lambdas_[i] = _yeojohnson_normmax(_nonnan(col))
            if self.standardize or force_transform:
                Xa[:, i] = self._tf_col(col, self.lambdas_[i])
        if self.standardize:
            self._scaler = StandardScaler()
            if force_transform:
                return self._scaler.fit_transform(Xa)
            self._scaler.fit(Xa)
        return Xa

    def fit(self, X, y = None):
        self._fit(X, False)
        return self

    def fit_transform(self, X, y = None):
        return self._fit(X, True)

    def transform(self, X):
        Xa = _as2df(X).copy()
        self._check(Xa, False)
        for i in range(Xa.shape[1]):
            Xa[:, i] = self._tf_col(np.ascontiguousarray(Xa[:, i]), self.lambdas_[i])
        if self.standardize:
            return self._scaler.transform(Xa)
        return Xa

    def inverse_transform(self, X):
        from scipy.special import inv_boxcox
        Xa = _as2df(X).copy()
        if self.standardize:
            Xa = self._scaler.inverse_transform(Xa)
        for i in range(Xa.shape[1]):
            lm = self.lambdas_[i]
            for r in range(Xa.shape[0]):
                v = Xa[r, i]
                if self.method == "box-cox":
                    Xa[r, i] = float(inv_boxcox(v, lm))
                elif v >= 0:
                    Xa[r, i] = math.exp(v) - 1.0 if abs(lm) < _EPS else math.pow(v * lm + 1.0, 1.0 / lm) - 1.0
                else:
                    Xa[r, i] = 1.0 - math.pow(-(2.0 - lm) * v + 1.0, 1.0 / (2.0 - lm)) if abs(lm - 2.0) > _EPS else 1.0 - math.exp(-v)
        return Xa

def power_transform(X, method: str = "yeo-johnson", *, standardize: bool = True, copy: bool = True):
    return PowerTransformer(method, standardize=standardize).fit_transform(X)
