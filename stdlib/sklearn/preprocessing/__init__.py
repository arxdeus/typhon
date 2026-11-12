# Ported from scikit-learn 1.9.1 (BSD-3-Clause)

import numpy as np
from ..base import _shallow_copy

def _handle_zeros_in_scale(scale):
    out = scale.copy()
    eps = 2.220446049250313e-16
    for j in range(out.shape[0]):
        if out[j] < 10.0 * eps:
            out[j] = 1.0
    return out

class StandardScaler:
    copy: bool
    with_mean: bool
    with_std: bool
    mean_: np.ndarray[float, 1]
    var_: np.ndarray[float, 1]
    scale_: np.ndarray[float, 1]
    n_samples_seen_: int
    n_features_in_: int

    def __init__(self, *, copy: bool = True, with_mean: bool = True, with_std: bool = True):
        self.copy = copy
        self.with_mean = with_mean
        self.with_std = with_std
        self.mean_ = np.empty(0)
        self.var_ = np.empty(0)
        self.scale_ = np.empty(0)
        self.n_samples_seen_ = 0
        self.n_features_in_ = 0

    def fit(self, X, y = None, sample_weight = None):
        # _incremental_mean_and_var from a fresh state (last_sample_count = 0)
        Xa = np.asarray(X, dtype=float)
        if Xa.ndim != 2:
            raise ValueError("Expected 2D array")
        n, p = Xa.shape
        new_sum = np.sum(Xa, axis=0)
        cnt = float(n)
        mean = new_sum / cnt
        T = new_sum / cnt
        temp = Xa - T
        correction = np.sum(temp, axis=0)
        temp = temp ** 2
        nuv = np.sum(temp, axis=0)
        nuv -= correction ** 2 / cnt
        var = nuv / cnt
        self.mean_ = mean
        self.var_ = var
        self.scale_ = _handle_zeros_in_scale(np.sqrt(var)) if self.with_std else np.ones(p)
        self.n_samples_seen_ = n
        self.n_features_in_ = p
        return self

    def transform(self, X, copy = None):
        Xa = np.asarray(X, dtype=float)
        if Xa.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {Xa.shape[1]} features, but StandardScaler is expecting {self.n_features_in_} features as input.")
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

    def fit_transform(self, X, y = None):
        return self.fit(X, y).transform(X)

    def __repr__(self) -> str:
        return "StandardScaler()"

class MinMaxScaler:
    feature_range: Tuple[float, float]
    clip: bool
    data_min_: np.ndarray[float, 1]
    data_max_: np.ndarray[float, 1]
    data_range_: np.ndarray[float, 1]
    scale_: np.ndarray[float, 1]
    min_: np.ndarray[float, 1]

    def __init__(self, feature_range: Tuple[float, float] = (0.0, 1.0), *, copy: bool = True, clip: bool = False):
        if feature_range[0] >= feature_range[1]:
            raise ValueError("Minimum of desired feature range must be smaller than maximum.")
        self.feature_range = feature_range
        self.clip = clip
        self.data_min_ = np.empty(0); self.data_max_ = np.empty(0); self.data_range_ = np.empty(0)
        self.scale_ = np.empty(0); self.min_ = np.empty(0)

    def fit(self, X, y = None):
        Xa = np.asarray(X, dtype=float)
        self.data_min_ = np.min(Xa, axis=0)
        self.data_max_ = np.max(Xa, axis=0)
        self.data_range_ = self.data_max_ - self.data_min_
        self.scale_ = (self.feature_range[1] - self.feature_range[0]) / _handle_zeros_in_scale(self.data_range_)
        self.min_ = self.feature_range[0] - self.data_min_ * self.scale_
        return self

    def transform(self, X):
        out = np.asarray(X, dtype=float) * self.scale_
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

class RobustScaler:
    with_centering: bool
    with_scaling: bool
    quantile_range: Tuple[float, float]
    unit_variance: bool
    center_: np.ndarray[float, 1]
    scale_: np.ndarray[float, 1]

    def __init__(self, *, with_centering: bool = True, with_scaling: bool = True,
                 quantile_range: Tuple[float, float] = (25.0, 75.0), copy: bool = True,
                 unit_variance: bool = False):
        self.with_centering = with_centering
        self.with_scaling = with_scaling
        self.quantile_range = quantile_range
        self.unit_variance = unit_variance
        self.center_ = np.empty(0)
        self.scale_ = np.empty(0)

    def fit(self, X, y = None):
        Xa = np.asarray(X, dtype=float)
        p = Xa.shape[1]
        if self.with_centering:
            self.center_ = np.nanmedian(Xa, axis=0)
        if self.with_scaling:
            sc = np.empty(p)
            for j in range(p):
                col = np.ascontiguousarray(Xa[:, j])
                q0 = float(np.nanpercentile(col, self.quantile_range[0]))
                q1 = float(np.nanpercentile(col, self.quantile_range[1]))
                sc[j] = q1 - q0
            sc = _handle_zeros_in_scale(sc)
            if self.unit_variance:
                from scipy.special import ndtri
                adjust = ndtri(self.quantile_range[1] / 100.0) - ndtri(self.quantile_range[0] / 100.0)
                sc = sc / adjust
            self.scale_ = sc
        return self

    def transform(self, X):
        out = np.asarray(X, dtype=float).copy()
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

class FunctionTransformer[F, G]:
    """func / inverse_func are Typhon callables (functions or lambdas)."""
    func: F
    inverse_func: G
    validate: bool

    def __init__(self, func: F = None, inverse_func: G = None, *, validate: bool = False,
                 accept_sparse: bool = False, check_inverse: bool = True,
                 feature_names_out = None, kw_args = None, inv_kw_args = None):
        self.func = func
        self.inverse_func = inverse_func
        self.validate = validate

    def fit(self, X, y = None):
        return self

    def transform(self, X):
        if self.func is None:
            return X
        else:
            return self.func(X)

    def inverse_transform(self, X):
        if self.inverse_func is None:
            return X
        else:
            return self.inverse_func(X)

    def fit_transform(self, X, y = None):
        return self.transform(X)
