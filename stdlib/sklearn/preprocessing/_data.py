# Copyright (C) 2026 the Typhon authors
# Ported from scikit-learn 1.9.1 (BSD-3-Clause)

# Dense scalers: StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler and
# the matching functions. NaNs are ignored in fit and kept in transform.

import numpy as np
from ..base import estimator_repr as _estimator_repr

_EPS = 2.220446049250313e-16

def _as2df(X):
    Xa = np.asarray(X, dtype=float)
    if Xa.ndim != 2:
        raise ValueError(f"Expected 2D array, got {Xa.ndim}D array instead")
    return Xa

def _handle_zeros_in_scale(scale):
    out = scale.copy()
    for j in range(out.shape[0]):
        if out[j] < 10.0 * _EPS:
            out[j] = 1.0
    return out

def _handle_zeros_mask(scale, mask):
    out = scale.copy()
    for j in range(out.shape[0]):
        if mask[j]:
            out[j] = 1.0
    return out

def _is_constant_feature(var: float, mean: float, n: float) -> bool:
    ub = n * _EPS * var + (n * mean * _EPS) ** 2
    return var <= ub

def _check_nfeat(Xa, p: int, name: str):
    if Xa.shape[1] != p:
        raise ValueError(f"X has {Xa.shape[1]} features, but {name} is expecting {p} features as input.")

def _incremental_mean_and_var(X, last_mean, last_var, last_n, sample_weight = None):
    # utils.extmath._incremental_mean_and_var for dense float64 X
    p = X.shape[1]
    nanmask = np.isnan(X)
    has_nan = bool(np.any(nanmask))
    Xz = np.where(nanmask, 0.0, X) if has_nan else X
    last_sum = last_mean * last_n
    new_sum = np.empty(p)
    new_n = np.empty(p)
    if sample_weight is None:
        new_sum = np.sum(Xz, axis=0)
        new_n = float(X.shape[0]) - np.sum(nanmask.astype(float), axis=0)
    else:
        w = np.asarray(sample_weight, dtype=float)
        new_sum = w @ Xz
        new_n = np.sum(w[:, None] * (~nanmask).astype(float), axis=0)
    upd_n = last_n + new_n
    upd_mean = (last_sum + new_sum) / upd_n
    T = new_sum / new_n
    temp = X - T
    if has_nan:
        temp = np.where(nanmask, 0.0, temp)
    correction = np.empty(p)
    nuv = np.empty(p)
    if sample_weight is None:
        correction = np.sum(temp, axis=0)
        temp = temp ** 2
        nuv = np.sum(temp, axis=0)
    else:
        w = np.asarray(sample_weight, dtype=float)
        correction = w @ temp
        temp = temp ** 2
        nuv = w @ temp
    nuv -= correction ** 2 / new_n
    luv = last_var * last_n
    upd_uv = np.empty(p)
    for j in range(p):
        if last_n[j] == 0.0:
            upd_uv[j] = nuv[j]
        else:
            lo = last_n[j] / new_n[j]
            d = last_sum[j] / lo - new_sum[j]
            upd_uv[j] = luv[j] + nuv[j] + lo / upd_n[j] * d * d
    return upd_mean, upd_uv / upd_n, upd_n

class StandardScaler:
    def __repr__(self) -> str:
        return _estimator_repr(self)

    copy: bool
    with_mean: bool
    with_std: bool
    mean_: np.ndarray[float, 1]
    var_: np.ndarray[float, 1]
    scale_: np.ndarray[float, 1]
    n_samples_seen_: int
    _n_seen: np.ndarray[float, 1]
    n_features_in_: int

    def __init__(self, *, copy: bool = True, with_mean: bool = True, with_std: bool = True):
        self.copy = copy
        self.with_mean = with_mean
        self.with_std = with_std
        self._reset()

    def _reset(self):
        self.mean_ = np.empty(0)
        self.var_ = np.empty(0)
        self.scale_ = np.empty(0)
        self.n_samples_seen_ = 0
        self._n_seen = np.empty(0)
        self.n_features_in_ = 0

    def fit(self, X, y = None, sample_weight = None):
        self._reset()
        return self.partial_fit(X, y, sample_weight)

    def partial_fit(self, X, y = None, sample_weight = None):
        Xa = _as2df(X)
        p = Xa.shape[1]
        if self.n_features_in_ == 0:
            self.n_features_in_ = p
            self.mean_ = np.zeros(p)
            self.var_ = np.zeros(p)
            self._n_seen = np.zeros(p)
        else:
            _check_nfeat(Xa, self.n_features_in_, "StandardScaler")
        if not self.with_mean and not self.with_std:
            self._n_seen = self._n_seen + (float(Xa.shape[0]) - np.sum(np.isnan(Xa).astype(float), axis=0))
            self.mean_ = np.empty(0)
            self.var_ = np.empty(0)
        else:
            m, v, n = _incremental_mean_and_var(Xa, self.mean_, self.var_, self._n_seen, sample_weight)
            self.mean_ = m
            self.var_ = v
            self._n_seen = n
        self.n_samples_seen_ = int(self._n_seen[0])
        if self.with_std:
            cm = np.zeros(p, dtype=bool)
            for j in range(p):
                cm[j] = _is_constant_feature(self.var_[j], self.mean_[j], self._n_seen[j])
            self.scale_ = _handle_zeros_mask(np.sqrt(self.var_), cm)
        else:
            self.scale_ = np.empty(0)
        if not self.with_mean and self.with_std:
            pass
        return self

    def transform(self, X, copy = None):
        Xa = _as2df(X)
        _check_nfeat(Xa, self.n_features_in_, "StandardScaler")
        out = Xa.copy()
        if self.with_mean:
            out -= self.mean_
        if self.with_std:
            out /= self.scale_
        return out

    def inverse_transform(self, X, copy = None):
        out = np.asarray(X, dtype=float).copy()
        if self.with_std:
            out *= self.scale_
        if self.with_mean:
            out += self.mean_
        return out

    def fit_transform(self, X, y = None, sample_weight = None):
        return self.fit(X, y, sample_weight).transform(X)

class MinMaxScaler:
    def __repr__(self) -> str:
        return _estimator_repr(self)

    feature_range: Tuple[float, float]
    copy: bool
    clip: bool
    data_min_: np.ndarray[float, 1]
    data_max_: np.ndarray[float, 1]
    data_range_: np.ndarray[float, 1]
    scale_: np.ndarray[float, 1]
    min_: np.ndarray[float, 1]
    n_samples_seen_: int
    n_features_in_: int

    def __init__(self, feature_range = (0.0, 1.0), *, copy: bool = True, clip: bool = False):
        fr = (float(feature_range[0]), float(feature_range[1]))
        self.feature_range = fr
        self.copy = copy
        self.clip = clip
        self._reset()

    def _reset(self):
        self.data_min_ = np.empty(0); self.data_max_ = np.empty(0); self.data_range_ = np.empty(0)
        self.scale_ = np.empty(0); self.min_ = np.empty(0)
        self.n_samples_seen_ = 0
        self.n_features_in_ = 0

    def fit(self, X, y = None):
        self._reset()
        return self.partial_fit(X, y)

    def partial_fit(self, X, y = None):
        fr = self.feature_range
        if fr[0] >= fr[1]:
            raise ValueError(f"Minimum of desired feature range must be smaller than maximum. Got {fr}.")
        Xa = _as2df(X)
        dmin = np.nanmin(Xa, axis=0)
        dmax = np.nanmax(Xa, axis=0)
        if self.n_samples_seen_ == 0:
            self.n_samples_seen_ = Xa.shape[0]
            self.n_features_in_ = Xa.shape[1]
        else:
            _check_nfeat(Xa, self.n_features_in_, "MinMaxScaler")
            dmin = np.minimum(self.data_min_, dmin)
            dmax = np.maximum(self.data_max_, dmax)
            self.n_samples_seen_ += Xa.shape[0]
        rng = dmax - dmin
        self.scale_ = (fr[1] - fr[0]) / _handle_zeros_in_scale(rng)
        self.min_ = fr[0] - dmin * self.scale_
        self.data_min_ = dmin
        self.data_max_ = dmax
        self.data_range_ = rng
        return self

    def transform(self, X):
        Xa = _as2df(X)
        _check_nfeat(Xa, self.n_features_in_, "MinMaxScaler")
        out = Xa * self.scale_
        out += self.min_
        if self.clip:
            out = np.clip(out, self.feature_range[0], self.feature_range[1])
        return out

    def inverse_transform(self, X):
        out = np.asarray(X, dtype=float) - self.min_
        out /= self.scale_
        return out

    def fit_transform(self, X, y = None):
        return self.fit(X, y).transform(X)

    def __repr__(self) -> str:
        return "MinMaxScaler()"
