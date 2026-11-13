# Ported from scikit-learn 1.9.1 (BSD-3-Clause)

from ..base import estimator_repr as _estimator_repr
import numpy as np
from numpy.random.legacy import RandomState
from ..base import clone
from ._base import SimpleImputer, MissingIndicator, _get_mask

_IntArr = np.ndarray[int, 1]

def _default_estimator(estimator):
    if estimator is None:
        from ..linear_model import BayesianRidge
        return BayesianRidge()
    else:
        return estimator

class _IterativeImputer[E]:
    def __repr__(self) -> str:
        return _estimator_repr(self)

    """Round-robin regression imputation. `estimator` is any regressor with
    fit/predict (default BayesianRidge). sample_posterior is not ported."""
    estimator: E
    missing_values: float
    max_iter: int
    tol: float
    n_nearest_features: Optional[int]
    initial_strategy: str
    fill_value: Optional[float]
    imputation_order: str
    skip_complete: bool
    min_value: float
    max_value: float
    random_state: Optional[int]
    add_indicator: bool
    keep_empty_features: bool
    random_state_: RandomState
    initial_imputer_: SimpleImputer
    indicator_: MissingIndicator
    imputation_sequence_: List[Tuple[int, _IntArr, E]]
    n_iter_: int
    n_features_with_missing_: int
    _is_empty_feature: np.ndarray[bool, 1]

    def __init__(self, estimator: E, missing_values: float = np.nan,
                 sample_posterior: bool = False, max_iter: int = 10, tol: float = 1e-3,
                 n_nearest_features: Optional[int] = None, initial_strategy: str = "mean",
                 fill_value: Optional[float] = None, imputation_order: str = "ascending",
                 skip_complete: bool = False, min_value: float = -np.inf, max_value: float = np.inf,
                 verbose: int = 0, random_state: Optional[int] = None, add_indicator: bool = False,
                 keep_empty_features: bool = False):
        if sample_posterior:
            raise NotImplementedError("IterativeImputer: sample_posterior=True is not ported")
        if imputation_order not in ("ascending", "descending", "roman", "arabic", "random"):
            raise ValueError(f"invalid imputation_order {imputation_order}")
        self.estimator = estimator
        self.missing_values = missing_values
        self.max_iter = max_iter
        self.tol = tol
        self.n_nearest_features = n_nearest_features
        self.initial_strategy = initial_strategy
        self.fill_value = fill_value
        self.imputation_order = imputation_order
        self.skip_complete = skip_complete
        self.min_value = min_value
        self.max_value = max_value
        self.random_state = random_state
        self.add_indicator = add_indicator
        self.keep_empty_features = keep_empty_features
        self.random_state_ = np.random.RandomState(random_state)
        self.initial_imputer_ = SimpleImputer(missing_values=missing_values, strategy=initial_strategy,
                                              fill_value=fill_value, keep_empty_features=keep_empty_features)
        self.indicator_ = MissingIndicator(missing_values=missing_values, error_on_new=False)
        self.imputation_sequence_ = []
        self.n_iter_ = 0
        self.n_features_with_missing_ = 0
        self._is_empty_feature = np.empty(0, dtype=bool)

    def _initial_imputation(self, X, in_fit: bool):
        Xm = _get_mask(X, self.missing_values)
        mask = Xm.copy()
        X_filled = np.empty((0, 0))
        if in_fit:
            X_filled = self.initial_imputer_.fit_transform(X)
            self._is_empty_feature = np.all(mask, axis=0)
        else:
            X_filled = self.initial_imputer_.transform(X)
        Xt = X
        if not self.keep_empty_features:
            Xt = X[:, ~self._is_empty_feature]
            mask = mask[:, ~self._is_empty_feature]
        else:
            mask[:, self._is_empty_feature] = False
        return Xt, X_filled, mask, Xm

    def _ordered_idx(self, mask):
        frac = mask.mean(axis=0)
        idx = np.flatnonzero(frac) if self.skip_complete else np.arange(frac.shape[0])
        o = self.imputation_order
        if o == "roman":
            return idx
        elif o == "arabic":
            return np.ascontiguousarray(idx[::-1])
        elif o == "ascending":
            n = frac.shape[0] - idx.shape[0]
            return np.ascontiguousarray(np.argsort(frac, kind="mergesort")[n:])
        elif o == "descending":
            n = frac.shape[0] - idx.shape[0]
            return np.ascontiguousarray(np.argsort(frac, kind="mergesort")[n:][::-1])
        else:
            r = idx.copy()
            self.random_state_.shuffle(r)
            return r

    def _abs_corr(self, X):
        n_features = X.shape[1]
        C = np.abs(np.corrcoef(X.T))
        tol = 1e-6
        for i in range(n_features):
            for j in range(n_features):
                if np.isnan(C[i, j]) or C[i, j] < tol:
                    C[i, j] = tol
            C[i, i] = 0.0
        C /= np.sum(np.abs(C), axis=0)[None, :]
        return C

    def _neighbors(self, n_features: int, feat: int, corr):
        if self.n_nearest_features is not None and self.n_nearest_features.__val__() < n_features:
            p = np.ascontiguousarray(corr[:, feat])
            return self.random_state_.choice(np.arange(n_features), self.n_nearest_features.__val__(),
                                             replace=False, p=p)
        else:
            out = np.empty(n_features - 1, dtype=int)
            k = 0
            for j in range(n_features):
                if j != feat:
                    out[k] = j
                    k += 1
            return out

    def _impute_one(self, Xf, mask, feat: int, nb, est, fit_mode: bool):
        miss = mask[:, feat]
        if fit_mode:
            obs = ~miss
            est.fit(Xf[obs][:, nb], Xf[obs, feat])
        if int(np.sum(miss)) == 0:
            return
        pred = np.asarray(est.predict(Xf[miss][:, nb]), dtype=float)
        pred = np.clip(pred, self.min_value, self.max_value)
        rows = np.flatnonzero(miss)
        for t in range(rows.shape[0]):
            Xf[rows[t], feat] = pred[t]

    def fit_transform(self, X, y = None):
        Xa = np.array(X, dtype=float)
        base = self.estimator
        self.random_state_ = np.random.RandomState(self.random_state)
        self.imputation_sequence_ = []
        Xo, Xt, mask, complete = self._initial_imputation(Xa, True)
        if self.add_indicator:
            self.indicator_._fit_mask(complete)
        if self.max_iter == 0 or bool(np.all(mask)) or Xt.shape[1] == 1:
            self.n_iter_ = 0
            return self._concat(Xt, complete)
        if not self.max_value > self.min_value:
            raise ValueError("One (or more) features have min_value >= max_value.")
        ordered = self._ordered_idx(mask)
        self.n_features_with_missing_ = ordered.shape[0]
        n_features = Xt.shape[1]
        corr = np.empty((0, 0))
        if self.n_nearest_features is not None and self.n_nearest_features.__val__() < n_features:
            corr = self._abs_corr(Xt)
        prev = Xt.copy()
        norm_tol = self.tol * float(np.max(np.abs(Xo[~mask])))
        for it in range(1, self.max_iter + 1):
            self.n_iter_ = it
            if self.imputation_order == "random":
                ordered = self._ordered_idx(mask)
            for feat in ordered:
                nb = self._neighbors(n_features, int(feat), corr)
                est = clone(base)
                self._impute_one(Xt, mask, int(feat), nb, est, True)
                self.imputation_sequence_.append((int(feat), nb, est))
            inf_norm = float(np.max(np.abs(Xt - prev)))
            if inf_norm < norm_tol:
                break
            prev = Xt.copy()
        for i in range(Xt.shape[0]):
            for j in range(Xt.shape[1]):
                if not mask[i, j]:
                    Xt[i, j] = Xo[i, j]
        return self._concat(Xt, complete)

    def _concat(self, Xt, complete):
        if self.add_indicator:
            return np.hstack((Xt, self.indicator_._transform_mask(complete).astype(float)))
        return Xt

    def fit(self, X, y = None):
        self.fit_transform(X)
        return self

    def transform(self, X):
        Xa = np.array(X, dtype=float)
        Xo, Xt, mask, complete = self._initial_imputation(Xa, False)
        if self.n_iter_ == 0 or bool(np.all(mask)):
            return self._concat(Xt, complete)
        for t in self.imputation_sequence_:
            self._impute_one(Xt, mask, t[0], t[1], t[2], False)
        for i in range(Xt.shape[0]):
            for j in range(Xt.shape[1]):
                if not mask[i, j]:
                    Xt[i, j] = Xo[i, j]
        return self._concat(Xt, complete)

def IterativeImputer(estimator = None, *, missing_values: float = np.nan,
                     sample_posterior: bool = False, max_iter: int = 10, tol: float = 1e-3,
                     n_nearest_features: Optional[int] = None, initial_strategy: str = "mean",
                     fill_value: Optional[float] = None, imputation_order: str = "ascending",
                     skip_complete: bool = False, min_value: float = -np.inf, max_value: float = np.inf,
                     verbose: int = 0, random_state: Optional[int] = None, add_indicator: bool = False,
                     keep_empty_features: bool = False):
    est = _default_estimator(estimator)
    return _IterativeImputer(est, missing_values, sample_posterior, max_iter, tol, n_nearest_features,
                             initial_strategy, fill_value, imputation_order, skip_complete, min_value,
                             max_value, verbose, random_state, add_indicator, keep_empty_features)
