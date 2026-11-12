# Copyright (C) 2026 the Typhon authors
# Ported from scikit-learn 1.9.1 (BSD-3-Clause)

import numpy as np
from ._data import (_as2df, _handle_zeros_in_scale, _check_nfeat, StandardScaler,
                    MinMaxScaler)

class MaxAbsScaler:
    copy: bool
    clip: bool
    max_abs_: np.ndarray[float, 1]
    scale_: np.ndarray[float, 1]
    n_samples_seen_: int
    n_features_in_: int

    def __init__(self, *, copy: bool = True, clip: bool = False):
        self.copy = copy
        self.clip = clip
        self._reset()

    def _reset(self):
        self.max_abs_ = np.empty(0)
        self.scale_ = np.empty(0)
        self.n_samples_seen_ = 0
        self.n_features_in_ = 0

    def fit(self, X, y = None):
        self._reset()
        return self.partial_fit(X, y)

    def partial_fit(self, X, y = None):
        Xa = _as2df(X)
        ma = np.nanmax(np.abs(Xa), axis=0)
        if self.n_samples_seen_ == 0:
            self.n_samples_seen_ = Xa.shape[0]
            self.n_features_in_ = Xa.shape[1]
        else:
            _check_nfeat(Xa, self.n_features_in_, "MaxAbsScaler")
            ma = np.maximum(self.max_abs_, ma)
            self.n_samples_seen_ += Xa.shape[0]
        self.max_abs_ = ma
        self.scale_ = _handle_zeros_in_scale(ma)
        return self

    def transform(self, X):
        Xa = _as2df(X)
        _check_nfeat(Xa, self.n_features_in_, "MaxAbsScaler")
        out = Xa / self.scale_
        if self.clip:
            out = np.clip(out, -1.0, 1.0)
        return out

    def inverse_transform(self, X):
        return np.asarray(X, dtype=float) * self.scale_

    def fit_transform(self, X, y = None):
        return self.fit(X, y).transform(X)

class RobustScaler:
    with_centering: bool
    with_scaling: bool
    quantile_range: Tuple[float, float]
    copy: bool
    unit_variance: bool
    center_: np.ndarray[float, 1]
    scale_: np.ndarray[float, 1]
    n_features_in_: int

    def __init__(self, *, with_centering: bool = True, with_scaling: bool = True,
                 quantile_range = (25.0, 75.0), copy: bool = True,
                 unit_variance: bool = False):
        self.with_centering = with_centering
        self.with_scaling = with_scaling
        self.quantile_range = (float(quantile_range[0]), float(quantile_range[1]))
        self.copy = copy
        self.unit_variance = unit_variance
        self.center_ = np.empty(0)
        self.scale_ = np.empty(0)
        self.n_features_in_ = 0

    def fit(self, X, y = None):
        Xa = _as2df(X)
        q0 = self.quantile_range[0]
        q1 = self.quantile_range[1]
        if not (0.0 <= q0 and q0 <= q1 and q1 <= 100.0):
            raise ValueError(f"Invalid quantile range: {self.quantile_range}")
        p = Xa.shape[1]
        self.n_features_in_ = p
        if self.with_centering:
            self.center_ = np.nanmedian(Xa, axis=0)
        if self.with_scaling:
            sc = np.empty(p)
            for j in range(p):
                col = np.ascontiguousarray(Xa[:, j])
                sc[j] = float(np.nanpercentile(col, q1)) - float(np.nanpercentile(col, q0))
            sc = _handle_zeros_in_scale(sc)
            if self.unit_variance:
                from scipy.special import ndtri
                adjust = ndtri(q1 / 100.0) - ndtri(q0 / 100.0)
                sc = sc / adjust
            self.scale_ = sc
        return self

    def transform(self, X):
        Xa = _as2df(X)
        _check_nfeat(Xa, self.n_features_in_, "RobustScaler")
        out = Xa.copy()
        if self.with_centering:
            out -= self.center_
        if self.with_scaling:
            out /= self.scale_
        return out

    def inverse_transform(self, X):
        out = np.asarray(X, dtype=float).copy()
        if self.with_scaling:
            out *= self.scale_
        if self.with_centering:
            out += self.center_
        return out

    def fit_transform(self, X, y = None):
        return self.fit(X, y).transform(X)

def _row_norms_l2(X):
    from ..decomposition._extmath import row_norms
    return row_norms(X)

def normalize(X, norm: str = "l2", *, axis: int = 1, copy: bool = True,
              return_norm: Literal[bool] = False):
    Xa = _as2df(X)
    if axis != 0 and axis != 1:
        raise ValueError(f"'{axis}' is not a supported axis")
    W = np.ascontiguousarray(Xa.T) if axis == 0 else Xa.copy()
    norms = np.empty(W.shape[0])
    if norm == "l1":
        norms = np.sum(np.abs(W), axis=1)
    elif norm == "l2":
        norms = _row_norms_l2(W)
    elif norm == "max":
        norms = np.max(np.abs(W), axis=1)
    else:
        raise ValueError(f"'{norm}' is not a supported norm")
    norms = _handle_zeros_in_scale(norms)
    W /= norms[:, None]
    out = np.ascontiguousarray(W.T) if axis == 0 else W
    if return_norm:
        return out, norms
    else:
        return out

class Normalizer:
    norm: str
    copy: bool
    n_features_in_: int

    def __init__(self, norm: str = "l2", *, copy: bool = True):
        self.norm = norm
        self.copy = copy
        self.n_features_in_ = 0

    def fit(self, X, y = None):
        self.n_features_in_ = _as2df(X).shape[1]
        return self

    def transform(self, X, copy = None):
        return normalize(X, self.norm, axis=1)

    def fit_transform(self, X, y = None):
        return self.fit(X).transform(X)

def binarize(X, *, threshold: float = 0.0, copy: bool = True):
    Xa = np.asarray(X, dtype=float)
    return np.where(Xa > threshold, 1.0, 0.0)

class Binarizer:
    threshold: float
    copy: bool
    n_features_in_: int

    def __init__(self, *, threshold: float = 0.0, copy: bool = True):
        self.threshold = float(threshold)
        self.copy = copy
        self.n_features_in_ = 0

    def fit(self, X, y = None):
        self.n_features_in_ = _as2df(X).shape[1]
        return self

    def transform(self, X, copy = None):
        return binarize(X, threshold=self.threshold)

    def fit_transform(self, X, y = None):
        return self.fit(X).transform(X)

class KernelCenterer:
    K_fit_rows_: np.ndarray[float, 1]
    K_fit_all_: float

    def __init__(self):
        self.K_fit_rows_ = np.empty(0)
        self.K_fit_all_ = 0.0

    def fit(self, K, y = None):
        Ka = _as2df(K)
        if Ka.shape[0] != Ka.shape[1]:
            raise ValueError(f"Kernel matrix must be a square matrix. Input is a {Ka.shape[0]}x{Ka.shape[1]} matrix.")
        n = Ka.shape[0]
        self.K_fit_rows_ = np.sum(Ka, axis=0) / float(n)
        self.K_fit_all_ = float(np.sum(self.K_fit_rows_)) / float(n)
        return self

    def transform(self, K, copy: bool = True):
        Ka = _as2df(K).copy()
        cols = (np.sum(Ka, axis=1) / float(self.K_fit_rows_.shape[0]))[:, None]
        Ka -= self.K_fit_rows_
        Ka -= cols
        Ka += self.K_fit_all_
        return Ka

    def fit_transform(self, K, y = None):
        return self.fit(K).transform(K)

def add_dummy_feature(X, value: float = 1.0):
    Xa = _as2df(X)
    return np.hstack((np.full((Xa.shape[0], 1), value), Xa))

def _apply_axis(s, X, axis: int):
    # shared driver of the *_scale functions (1d input allowed)
    Xa = np.asarray(X, dtype=float)
    if static.len(Xa.shape) == 1:
        X2 = Xa.reshape(Xa.shape[0], 1)
        if axis == 0:
            return np.ascontiguousarray(s.fit_transform(X2)).ravel()
        return np.ascontiguousarray(s.fit_transform(np.ascontiguousarray(X2.T)).T).ravel()
    else:
        if axis == 0:
            return s.fit_transform(Xa)
        return np.ascontiguousarray(s.fit_transform(np.ascontiguousarray(Xa.T)).T)

def scale(X, *, axis: int = 0, with_mean: bool = True, with_std: bool = True, copy: bool = True):
    Xa = np.asarray(X, dtype=float).copy()
    if static.len(Xa.shape) == 1:
        return np.ascontiguousarray(scale(Xa.reshape(Xa.shape[0], 1), axis=0, with_mean=with_mean, with_std=with_std)).ravel()
    else:
        sd = _handle_zeros_in_scale(np.nanstd(Xa, axis=axis))
        Xr = Xa if axis == 0 else np.ascontiguousarray(Xa.T)
        if with_mean:
            Xr -= np.nanmean(Xr, axis=0)
            m1 = np.nanmean(Xr, axis=0)
            if not np.allclose(m1, 0.0):
                Xr -= m1
        if with_std:
            Xr /= sd
            if with_mean:
                m2 = np.nanmean(Xr, axis=0)
                if not np.allclose(m2, 0.0):
                    Xr -= m2
        return Xr if axis == 0 else np.ascontiguousarray(Xr.T)

def minmax_scale(X, feature_range = (0.0, 1.0), *, axis: int = 0, copy: bool = True):
    return _apply_axis(MinMaxScaler(feature_range), X, axis)

def maxabs_scale(X, *, axis: int = 0, copy: bool = True):
    return _apply_axis(MaxAbsScaler(), X, axis)

def robust_scale(X, *, axis: int = 0, with_centering: bool = True, with_scaling: bool = True,
                 quantile_range = (25.0, 75.0), copy: bool = True, unit_variance: bool = False):
    return _apply_axis(RobustScaler(with_centering=with_centering, with_scaling=with_scaling,
                                    quantile_range=quantile_range, unit_variance=unit_variance), X, axis)
