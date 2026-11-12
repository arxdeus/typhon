# Ported from scikit-learn 1.9.1 (BSD-3-Clause)
#
# Shared machinery of the linear_model port.
#
# Typing convention: estimators that accept 1-d or 2-d targets are generic in
# D (the ndim of y, default 1). Internally coefficients are always stored as
# `_coef` (n_targets, n_features) and `_intercept` (n_targets,); the public
# `coef_` / `intercept_` properties return sklearn's shapes for D. fit()
# called with a target of a different ndim re-types the estimator
# (`_retype`) and returns the D=ndim(y) specialization.

import numpy as np
import scipy.linalg as sl

def _retype(est, T: type):
    n = T.__new__()
    for k, v in static.vars(est):
        setattr(n, k, v)
    return n

def _ndim_of(a: np.ndarray[T, N], T: type, N: Literal[int]) -> Literal[int]:
    return N

def _as2d_y(y) -> np.ndarray[float, 2]:
    Y = np.asarray(y, dtype=float)
    if Y.ndim == 1:
        return np.ascontiguousarray(Y.reshape((Y.shape[0], 1)))
    else:
        return np.ascontiguousarray(Y.reshape((Y.shape[0], -1)))

def _check_sw(sample_weight, n: int) -> np.ndarray[float, 1]:
    if sample_weight is None:
        return np.ones(n)
    elif isinstance(sample_weight, float) or isinstance(sample_weight, int):
        return np.full(n, float(sample_weight))
    else:
        sw = np.asarray(sample_weight, dtype=float).ravel()
        if sw.shape[0] != n:
            raise ValueError(f"sample_weight.shape == {(sw.shape[0],)}, expected {(n,)}!")
        return sw.copy()

def _weighted_mean_cols(X: np.ndarray[float, 2], sw: np.ndarray[float, 1], has_sw: bool) -> np.ndarray[float, 1]:
    if has_sw:
        return np.average(X, axis=0, weights=sw)
    else:
        return np.mean(X, axis=0)

def _preprocess_data(X, Y: np.ndarray[float, 2], fit_intercept: bool,
                     sw: np.ndarray[float, 1], has_sw: bool, rescale_with_sw: bool = True):
    """Dense _preprocess_data: returns (X, Y, X_offset, y_offset, sqrt_sw)."""
    Xa = np.array(X, dtype=float)
    Ya = Y.copy()
    n, p = Xa.shape
    X_offset = np.zeros(p)
    y_offset = np.zeros(Ya.shape[1])
    if fit_intercept:
        X_offset = _weighted_mean_cols(Xa, sw, has_sw)
        Xa -= X_offset
        y_offset = _weighted_mean_cols(Ya, sw, has_sw)
        Ya -= y_offset
    sqrt_sw = np.ones(n)
    if has_sw and rescale_with_sw:
        sqrt_sw = np.sqrt(sw)
        Xa *= sqrt_sw[:, None]
        Ya *= sqrt_sw[:, None]
    return Xa, Ya, X_offset, y_offset, sqrt_sw

def _set_intercept(coef: np.ndarray[float, 2], X_offset: np.ndarray[float, 1],
                   y_offset: np.ndarray[float, 1], fit_intercept: bool) -> np.ndarray[float, 1]:
    if fit_intercept:
        return y_offset - X_offset @ coef.T
    else:
        return np.zeros(coef.shape[0])

def _r2(y_true, y_pred, sw: np.ndarray[float, 1], has_sw: bool) -> float:
    # r2_score with uniform_average over outputs
    Yt = _as2d_y(y_true)
    Yp = _as2d_y(y_pred)
    n, k = Yt.shape
    tot = 0.0
    for j in range(k):
        wsum = 0.0
        mean = 0.0
        for i in range(n):
            w = sw[i] if has_sw else 1.0
            wsum += w
            mean += w * Yt[i, j]
        mean /= wsum
        num = 0.0
        den = 0.0
        for i in range(n):
            w = sw[i] if has_sw else 1.0
            num += w * (Yt[i, j] - Yp[i, j]) ** 2
            den += w * (Yt[i, j] - mean) ** 2
        if den != 0.0:
            tot += 1.0 - num / den
        elif num == 0.0:
            tot += 1.0
        else:
            tot += 0.0
    return tot / float(k)

class _LinearRegressorMixin:
    """coef_/intercept_/predict/score for estimators storing _coef/_intercept."""
    @property
    def coef_(self):
        if self._nd() == 1:
            return self._coef[0].copy()
        else:
            return self._coef.copy()

    @property
    def intercept_(self):
        if self._nd() == 1:
            return float(self._intercept[0])
        else:
            return self._intercept.copy()

    def _decision(self, X) -> np.ndarray[float, 2]:
        Xa = np.asarray(X, dtype=float)
        return Xa @ self._coef.T + self._intercept

    def predict(self, X):
        P = self._decision(X)
        if self._nd() == 1:
            return np.ascontiguousarray(P[:, 0])
        else:
            return P

    def score(self, X, y, sample_weight = None) -> float:
        n = np.asarray(X).shape[0]
        return _r2(y, self._decision(X), _check_sw(sample_weight, n), sample_weight is not None)

def _encode_labels(y):
    ya = np.asarray(y)
    classes, yi = np.unique(ya, return_inverse=True)
    return classes, yi

def _class_weight_vec(class_weight, classes, yi: np.ndarray[int, 1]) -> np.ndarray[float, 1]:
    """compute_class_weight; class_weight None, 'balanced' or dict."""
    K = classes.shape[0]
    cw = np.ones(K)
    if class_weight is None:
        pass
    elif isinstance(class_weight, str):
        if class_weight != "balanced":
            raise ValueError("class_weight must be dict, 'balanced', or None")
        cnt = np.bincount(yi, minlength=K)
        n = yi.shape[0]
        for k in range(K):
            cw[k] = float(n) / (float(K) * float(cnt[k]))
    else:
        for k in range(K):
            key = classes[k]
            if key in class_weight:
                cw[k] = float(class_weight[key])
    return cw

def _expit(x: float) -> float:
    if x >= 0.0:
        return 1.0 / (1.0 + np.exp(-x))
    else:
        e = np.exp(x)
        return e / (1.0 + e)

class _LinearClassifierMixin:
    """decision_function/predict/score for classifiers with 2-d _coef."""
    @property
    def coef_(self):
        return self._coef.copy()

    @property
    def intercept_(self):
        return self._intercept.copy()

    def _scores(self, X) -> np.ndarray[float, 2]:
        Xa = np.asarray(X, dtype=float)
        return Xa @ self._coef.T + self._intercept

    def decision_function(self, X):
        s = self._scores(X)
        if s.shape[1] == 1:
            return np.ascontiguousarray(s[:, 0])
        else:
            return s

    def predict(self, X):
        s = self._scores(X)
        n = s.shape[0]
        idx = np.zeros(n, dtype=int)
        if s.shape[1] == 1:
            for i in range(n):
                idx[i] = 1 if s[i, 0] > 0.0 else 0
        else:
            idx = np.argmax(s, axis=1)
        return self.classes_[idx]

    def score(self, X, y, sample_weight = None) -> float:
        ya = np.asarray(y)
        pr = self.predict(X)
        n = ya.shape[0]
        sw = _check_sw(sample_weight, n)
        num = 0.0
        den = 0.0
        for i in range(n):
            den += sw[i]
            if ya[i] == pr[i]:
                num += sw[i]
        return num / den

    def _predict_proba_lr(self, X):
        s = self._scores(X)
        n = s.shape[0]
        if s.shape[1] == 1:
            pr = np.empty((n, 2))
            for i in range(n):
                e = _expit(s[i, 0])
                pr[i, 1] = e
                pr[i, 0] = 1.0 - e
            return pr
        else:
            K = s.shape[1]
            pr = np.empty((n, K))
            for i in range(n):
                tot = 0.0
                for k in range(K):
                    pr[i, k] = _expit(s[i, k])
                    tot += pr[i, k]
                if tot == 0.0:
                    for k in range(K):
                        pr[i, k] = 1.0 / float(K)
                else:
                    for k in range(K):
                        pr[i, k] /= tot
            return pr

    def densify(self):
        return self

    def sparsify(self):
        return self

def nnls(A, b, maxiter: Optional[int] = None):
    """scipy.optimize.nnls (Lawson-Hanson, scipy's C kernel)."""
    from scipy.optimize._slsqp_core import _nnls
    Aa = np.asarray(A, dtype=float)
    ba = np.asarray(b, dtype=float).ravel()
    m, n = Aa.shape
    if n == 0:
        return np.empty(0), float(np.linalg.norm(ba))
    a = Ptr[float](m * n)
    for j in range(n):
        for i in range(m):
            a[i + j * m] = Aa[i, j]
    bb = Ptr[float](m)
    for i in range(m):
        bb[i] = ba[i]
    x = Ptr[float](n)
    w = Ptr[float](n)
    zz = Ptr[float](m)
    ind = Ptr[int](n)
    mi = 3 * n if maxiter is None else maxiter
    rnorm, info = _nnls(m, n, a, bb, x, w, zz, ind, mi)
    if info == 3:
        raise RuntimeError("Maximum number of iterations reached.")
    out = np.empty(n)
    for i in range(n):
        out[i] = x[i]
    return out, rnorm

class LinearRegression[D: Literal[int] = 1](_LinearRegressorMixin):
    fit_intercept: bool
    copy_X: bool
    tol: float
    n_jobs: Optional[int]
    positive: bool
    _coef: np.ndarray[float, 2]
    _intercept: np.ndarray[float, 1]
    rank_: int
    singular_: np.ndarray[float, 1]
    n_features_in_: int

    def __init__(self, *, fit_intercept: bool = True, copy_X: bool = True, tol: float = 1e-6,
                 n_jobs: Optional[int] = None, positive: bool = False):
        self.fit_intercept = fit_intercept
        self.copy_X = copy_X
        self.tol = tol
        self.n_jobs = n_jobs
        self.positive = positive
        self._coef = np.zeros((0, 0))
        self._intercept = np.zeros(0)
        self.rank_ = 0
        self.singular_ = np.zeros(0)
        self.n_features_in_ = 0

    def _nd(self) -> Literal[int]:
        return D

    def fit(self, X, y, sample_weight = None):
        ya = np.asarray(y, dtype=float)
        if _ndim_of(ya) != D:
            return _retype(self, LinearRegression[_ndim_of(ya)]).fit(X, ya, sample_weight)
        Y = _as2d_y(ya)
        n = Y.shape[0]
        sw = _check_sw(sample_weight, n)
        has_sw = sample_weight is not None
        Xa, Yc, X_offset, y_offset, _ = _preprocess_data(X, Y, self.fit_intercept, sw, has_sw)
        p = Xa.shape[1]
        self.n_features_in_ = p
        k = Yc.shape[1]
        coef = np.zeros((k, p))
        if self.positive:
            for j in range(k):
                c, _ = nnls(Xa, Yc[:, j])
                coef[j] = c
        else:
            sol, _, rank, sing = sl.lstsq(Xa, Yc, cond=self.tol)
            coef = np.ascontiguousarray(sol.T)
            self.rank_ = rank
            self.singular_ = sing
        self._coef = coef
        self._intercept = _set_intercept(coef, X_offset, y_offset, self.fit_intercept)
        return self
