# Ported from scikit-learn 1.9.1 (BSD-3-Clause)
# Numeric (float) imputers. missing_values is a float (default nan).

from ..base import estimator_repr as _estimator_repr
import numpy as np
from ..base import clone

def _get_mask(X, missing_values: float):
    if np.isnan(missing_values):
        return np.isnan(X)
    else:
        return X == missing_values

class MissingIndicator:
    def __repr__(self) -> str:
        return _estimator_repr(self)

    missing_values: float
    features: str
    error_on_new: bool
    features_: np.ndarray[int, 1]
    _n_features: int

    def __init__(self, *, missing_values: float = np.nan, features: str = "missing-only",
                 sparse = "auto", error_on_new: bool = True):
        if features != "missing-only" and features != "all":
            raise ValueError("features must be 'missing-only' or 'all'")
        self.missing_values = missing_values
        self.features = features
        self.error_on_new = error_on_new
        self.features_ = np.empty(0, dtype=int)
        self._n_features = 0

    def _fit_mask(self, mask):
        self._n_features = mask.shape[1]
        if self.features == "all":
            self.features_ = np.arange(mask.shape[1])
        else:
            self.features_ = np.flatnonzero(np.any(mask, axis=0))
        return self

    def fit(self, X, y = None):
        return self._fit_mask(_get_mask(np.asarray(X, dtype=float), self.missing_values))

    def _transform_mask(self, mask):
        if self.features == "missing-only" and self.error_on_new:
            have = np.any(mask, axis=0)
            fit = np.zeros(mask.shape[1], dtype=bool)
            fit[self.features_] = True
            for j in range(mask.shape[1]):
                if have[j] and not fit[j]:
                    raise ValueError("The features have missing values in transform but have no missing values in fit.")
        return mask[:, self.features_]

    def transform(self, X):
        return self._transform_mask(_get_mask(np.asarray(X, dtype=float), self.missing_values))

    def fit_transform(self, X, y = None):
        return self.fit(X).transform(X)

def _most_frequent(col) -> float:
    if col.shape[0] == 0:
        return np.nan
    s = np.sort(col)
    best = s[0]
    best_c = 0
    i = 0
    while i < s.shape[0]:
        j = i
        while j < s.shape[0] and s[j] == s[i]:
            j += 1
        if j - i > best_c:
            best_c = j - i
            best = s[i]
        i = j
    return best

class SimpleImputer:
    def __repr__(self) -> str:
        return _estimator_repr(self)

    missing_values: float
    strategy: str
    fill_value: Optional[float]
    add_indicator: bool
    keep_empty_features: bool
    statistics_: np.ndarray[float, 1]
    indicator_: MissingIndicator

    def __init__(self, *, missing_values: float = np.nan, strategy: str = "mean",
                 fill_value: Optional[float] = None, copy: bool = True, add_indicator: bool = False,
                 keep_empty_features: bool = False):
        if strategy not in ("mean", "median", "most_frequent", "constant"):
            raise ValueError(f"Can only use these strategies: mean, median, most_frequent, constant got strategy={strategy}")
        self.missing_values = missing_values
        self.strategy = strategy
        self.fill_value = fill_value
        self.add_indicator = add_indicator
        self.keep_empty_features = keep_empty_features
        self.statistics_ = np.empty(0)
        self.indicator_ = MissingIndicator(missing_values=missing_values, error_on_new=False)

    def fit(self, X, y = None):
        Xa = np.asarray(X, dtype=float)
        mask = _get_mask(Xa, self.missing_values)
        if self.add_indicator:
            self.indicator_._fit_mask(mask)
        p = Xa.shape[1]
        st = np.empty(p)
        empty_val = 0.0 if self.keep_empty_features else np.nan
        for j in range(p):
            col = Xa[:, j][~mask[:, j]]
            if self.strategy == "constant":
                fv = 0.0 if self.fill_value is None else self.fill_value.__val__()
                st[j] = fv if (col.shape[0] > 0 or self.keep_empty_features) else np.nan
            elif col.shape[0] == 0:
                st[j] = empty_val
            elif self.strategy == "mean":
                st[j] = float(np.mean(col))
            elif self.strategy == "median":
                st[j] = float(np.median(col))
            else:
                st[j] = _most_frequent(col)
        self.statistics_ = st
        return self

    def transform(self, X):
        Xa = np.array(X, dtype=float)
        mask = _get_mask(Xa, self.missing_values)
        p = Xa.shape[1]
        if p != self.statistics_.shape[0]:
            raise ValueError(f"X has {p} features per sample, expected {self.statistics_.shape[0]}")
        valid = ~np.isnan(self.statistics_) if not self.keep_empty_features else np.ones(p, dtype=bool)
        for i in range(Xa.shape[0]):
            for j in range(p):
                if mask[i, j] and valid[j]:
                    Xa[i, j] = self.statistics_[j]
        out = Xa[:, valid]
        if self.add_indicator:
            ind = self.indicator_._transform_mask(mask).astype(float)
            return np.hstack((out, ind))
        return out

    def fit_transform(self, X, y = None):
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        if not self.add_indicator:
            raise ValueError("'inverse_transform' works only when 'SimpleImputer' is instantiated with 'add_indicator=True'.")
        Xa = np.asarray(X, dtype=float)
        p = self.statistics_.shape[0]
        nf = self.indicator_.features_.shape[0]
        valid = np.flatnonzero(~np.isnan(self.statistics_))
        out = np.zeros((Xa.shape[0], p))
        nv = valid.shape[0]
        for k in range(nv):
            out[:, valid[k]] = Xa[:, k]
        for k in range(nf):
            f = self.indicator_.features_[k]
            for i in range(Xa.shape[0]):
                if Xa[i, nv + k] != 0:
                    out[i, f] = self.missing_values
        return out

def _nan_euclidean(X, Y, missing_values: float):
    mx = _get_mask(X, missing_values)
    my = _get_mask(Y, missing_values)
    Xz = np.where(mx, 0.0, X)
    Yz = np.where(my, 0.0, Y)
    XX = np.sum(Xz * Xz, axis=1)
    YY = np.sum(Yz * Yz, axis=1)
    D = -2.0 * (Xz @ Yz.T)
    D += XX[:, None]
    D += YY[None, :]
    np.maximum(D, 0.0, out=D)
    mxf = mx.astype(float)
    myf = my.astype(float)
    D -= (Xz * Xz) @ myf.T
    D -= mxf @ (Yz * Yz).T
    np.maximum(D, 0.0, out=D)
    pc = (1.0 - mxf) @ (1.0 - myf).T
    for i in range(D.shape[0]):
        for j in range(D.shape[1]):
            if pc[i, j] == 0:
                D[i, j] = np.nan
            else:
                D[i, j] = np.sqrt(D[i, j] / pc[i, j] * X.shape[1])
    return D

class KNNImputer:
    def __repr__(self) -> str:
        return _estimator_repr(self)

    missing_values: float
    n_neighbors: int
    weights: str
    add_indicator: bool
    keep_empty_features: bool
    _fit_X: np.ndarray[float, 2]
    _mask_fit_X: np.ndarray[bool, 2]
    _valid_mask: np.ndarray[bool, 1]
    indicator_: MissingIndicator

    def __init__(self, *, missing_values: float = np.nan, n_neighbors: int = 5,
                 weights: str = "uniform", metric: str = "nan_euclidean", copy: bool = True,
                 add_indicator: bool = False, keep_empty_features: bool = False):
        if weights != "uniform" and weights != "distance":
            raise ValueError("weights must be 'uniform' or 'distance'")
        if metric != "nan_euclidean":
            raise NotImplementedError("KNNImputer: only metric='nan_euclidean' is ported")
        self.missing_values = missing_values
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.add_indicator = add_indicator
        self.keep_empty_features = keep_empty_features
        self._fit_X = np.empty((0, 0))
        self._mask_fit_X = np.empty((0, 0), dtype=bool)
        self._valid_mask = np.empty(0, dtype=bool)
        self.indicator_ = MissingIndicator(missing_values=missing_values, error_on_new=False)

    def fit(self, X, y = None):
        self._fit_X = np.array(X, dtype=float)
        self._mask_fit_X = _get_mask(self._fit_X, self.missing_values)
        self._valid_mask = ~np.all(self._mask_fit_X, axis=0)
        if self.add_indicator:
            self.indicator_._fit_mask(self._mask_fit_X)
        return self

    def _impute(self, dist, k: int, col):
        # dist: (n_receivers, n_donors) without all-nan rows
        r = dist.shape[0]
        out = np.empty(r)
        for i in range(r):
            order = np.argsort(np.where(np.isnan(dist[i]), np.inf, dist[i]), kind="stable")
            idx = order[:k]
            dd = dist[i][idx]
            w = np.ones(k)
            if self.weights == "distance":
                has0 = False
                for t in range(k):
                    if dd[t] == 0.0:
                        has0 = True
                for t in range(k):
                    if has0:
                        w[t] = 1.0 if dd[t] == 0.0 else 0.0
                    else:
                        w[t] = 1.0 / dd[t]
            for t in range(k):
                if np.isnan(dd[t]) or np.isnan(w[t]):
                    w[t] = 0.0
            sw = float(np.sum(w))
            out[i] = float(np.sum(w * col[idx])) / sw
        return out

    def transform(self, X):
        Xa = np.array(X, dtype=float)
        mask = _get_mask(Xa, self.missing_values)
        p = Xa.shape[1]
        rows = np.flatnonzero(np.any(mask[:, self._valid_mask], axis=1))
        if rows.shape[0] > 0:
            D = _nan_euclidean(Xa[rows], self._fit_X, self.missing_values)
            non_missing_fit = ~self._mask_fit_X
            for col in range(p):
                if not self._valid_mask[col]:
                    continue
                cm = mask[rows, col]
                if not np.any(cm):
                    continue
                donors = np.flatnonzero(non_missing_fit[:, col])
                recv_local = np.flatnonzero(cm)
                sub = D[recv_local][:, donors]
                allnan = np.all(np.isnan(sub), axis=1)
                if np.any(allnan):
                    cmean = float(np.mean(self._fit_X[donors, col]))
                    for t in range(recv_local.shape[0]):
                        if allnan[t]:
                            Xa[rows[recv_local[t]], col] = cmean
                    recv_local = recv_local[~allnan]
                    if recv_local.shape[0] == 0:
                        continue
                    sub = D[recv_local][:, donors]
                k = min(self.n_neighbors, donors.shape[0])
                vals = self._impute(sub, k, self._fit_X[donors, col])
                for t in range(recv_local.shape[0]):
                    Xa[rows[recv_local[t]], col] = vals[t]
        out = np.empty((0, 0))
        if self.keep_empty_features:
            out = Xa
            out[:, ~self._valid_mask] = 0.0
        else:
            out = Xa[:, self._valid_mask]
        if self.add_indicator:
            return np.hstack((out, self.indicator_._transform_mask(mask).astype(float)))
        return out

    def fit_transform(self, X, y = None):
        return self.fit(X).transform(X)
