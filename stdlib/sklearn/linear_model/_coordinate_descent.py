# Ported from scikit-learn 1.9.1 (BSD-3-Clause)
#
# ElasticNet, Lasso, MultiTaskElasticNet, MultiTaskLasso, the *CV variants,
# enet_path / lasso_path and _alpha_grid (dense inputs).
# DEVIATIONS: `alphas` for paths / CV is an int (grid size) or a list of
# floats; MultiTask*CV and *CV accept cv=int (KFold, as sklearn's check_cv for
# regressors) or a splitter object; n_jobs/verbose are ignored.

import math
import numpy as np
from ._base import (_retype, _ndim_of, _as2d_y, _check_sw, _preprocess_data,
                    _set_intercept, _LinearRegressorMixin)
from ._cd_fast import (enet_coordinate_descent, enet_coordinate_descent_gram,
                       enet_coordinate_descent_multi_task)
from ..decomposition._extmath import check_random_state

_RESOLUTION = 1e-15

def _alpha_grid(X, y, Xy = None, l1_ratio: float = 1.0, fit_intercept: bool = True,
                eps: float = 1e-3, n_alphas: int = 100, sample_weight = None,
                positive: bool = False) -> np.ndarray[float, 1]:
    if l1_ratio == 0.0:
        raise ValueError("Automatic alpha grid generation is not supported for l1_ratio=0. Please supply a grid by providing your estimator with the appropriate `alphas=` argument.")
    Xa = np.asarray(X, dtype=float)
    Y = _as2d_y(y)
    n = Xa.shape[0]
    Xyw = np.zeros((0, 0))
    if Xy is not None:
        Xyw = _as2d_y(Xy)
    else:
        has_sw = sample_weight is not None
        sw = _check_sw(sample_weight, n)
        Yc = Y
        X_mean = np.zeros(Xa.shape[1])
        if fit_intercept:
            if has_sw:
                Yc = Y - np.average(Y, axis=0, weights=sw)
                X_mean = np.average(Xa, axis=0, weights=sw)
            else:
                Yc = Y - np.mean(Y, axis=0)
                X_mean = np.mean(Xa, axis=0)
        yw = Yc * sw[:, None] if has_sw else Yc
        if fit_intercept:
            Xyw = Xa.T @ yw - X_mean[:, None] * np.sum(yw, axis=0)
        else:
            Xyw = Xa.T @ yw
    ns = float(n)
    if sample_weight is not None:
        ns = float(np.sum(_check_sw(sample_weight, n)))
    alpha_max = 0.0
    if not positive:
        alpha_max = math.sqrt(float(np.max(np.sum(Xyw ** 2, axis=1)))) / (ns * l1_ratio)
    else:
        alpha_max = max(0.0, float(np.max(Xyw))) / (ns * l1_ratio)
    if alpha_max <= _RESOLUTION:
        return np.full(n_alphas, _RESOLUTION)
    return np.geomspace(alpha_max, alpha_max * eps, num=n_alphas)

def _alphas_desc(alphas) -> np.ndarray[float, 1]:
    a = np.sort(np.atleast_1d(np.asarray(alphas, dtype=float)).ravel())
    m = a.shape[0]
    out = np.empty(m)
    for i in range(m):
        out[i] = a[m - 1 - i]
    return out

def _enet_path_1d(X: np.ndarray[float, 2], y: np.ndarray[float, 1], l1_ratio: float,
                  alphas: np.ndarray[float, 1], use_gram: bool, Gram: np.ndarray[float, 2],
                  Xy: np.ndarray[float, 1], coef_init: np.ndarray[float, 1], tol: float,
                  max_iter: int, rng, random: bool, positive: bool):
    """Path on already preprocessed (centered, rescaled) data, single target."""
    n, p = X.shape
    na = alphas.shape[0]
    coefs = np.empty((p, na))
    gaps = np.empty(na)
    iters = List[int]()
    coef_ = coef_init.copy()
    for i in range(na):
        l1 = alphas[i] * l1_ratio * float(n)
        l2 = alphas[i] * (1.0 - l1_ratio) * float(n)
        gap = 0.0
        it = 0
        if use_gram:
            _, gap, _, it = enet_coordinate_descent_gram(coef_, l1, l2, Gram, Xy, y, max_iter,
                                                         tol, rng, random, positive)
        else:
            _, gap, _, it = enet_coordinate_descent(coef_, l1, l2, X, y, max_iter, tol, rng,
                                                    random, positive)
        coefs[:, i] = coef_
        gaps[i] = gap / float(n)
        iters.append(it)
    return coefs, gaps, iters

def _enet_path_mt(X: np.ndarray[float, 2], Y: np.ndarray[float, 2], l1_ratio: float,
                  alphas: np.ndarray[float, 1], coef_init: np.ndarray[float, 2], tol: float,
                  max_iter: int, rng, random: bool):
    n, p = X.shape
    k = Y.shape[1]
    na = alphas.shape[0]
    coefs = np.empty((k, p, na))
    gaps = np.empty(na)
    iters = List[int]()
    W = coef_init.copy()
    for i in range(na):
        l1 = alphas[i] * l1_ratio * float(n)
        l2 = alphas[i] * (1.0 - l1_ratio) * float(n)
        _, gap, _, it = enet_coordinate_descent_multi_task(W, l1, l2, X, Y, max_iter, tol, rng,
                                                           random)
        coefs[:, :, i] = W
        gaps[i] = gap / float(n)
        iters.append(it)
    return coefs, gaps, iters

def _resolve_precompute(precompute, n: int, p: int) -> bool:
    if isinstance(precompute, str):
        if precompute != "auto":
            raise ValueError(f"Precompute should be one of True, False, 'auto' or array-like. Got {precompute!r}")
        return n > p
    elif isinstance(precompute, bool):
        return precompute
    else:
        return True

def enet_path(X, y, *, l1_ratio: float = 0.5, eps: float = 1e-3, alphas = 100,
              precompute = "auto", Xy = None, copy_X: bool = True, coef_init = None,
              verbose = False, return_n_iter: Literal[bool] = False, positive: bool = False,
              check_input: bool = True, tol: float = 1e-4, max_iter: int = 1000,
              random_state = None, selection: str = "cyclic"):
    Xa = np.array(X, dtype=float)
    ya = np.asarray(y, dtype=float)
    n, p = Xa.shape
    if _ndim_of(ya) != 1 and positive:
        raise ValueError("positive=True is not allowed for multi-output (y.ndim != 1)")
    al = np.zeros(0)
    if isinstance(alphas, int):
        al = _alpha_grid(Xa, ya, l1_ratio=l1_ratio, fit_intercept=False, positive=positive,
                         eps=eps, n_alphas=alphas)
    else:
        al = _alphas_desc(alphas)
    if selection != "random" and selection != "cyclic":
        raise ValueError("selection should be either random or cyclic.")
    rng = check_random_state(random_state)
    random = selection == "random"
    if _ndim_of(ya) == 1:
        use_gram = _resolve_precompute(precompute, n, p)
        G = np.zeros((0, 0))
        xy = np.zeros(0)
        if use_gram:
            if isinstance(precompute, str) or isinstance(precompute, bool):
                G = Xa.T @ Xa
            else:
                G = np.asarray(precompute, dtype=float)
            if Xy is None:
                xy = Xa.T @ ya
            else:
                xy = np.asarray(Xy, dtype=float).ravel()
        c0 = np.zeros(p) if coef_init is None else np.array(coef_init, dtype=float).ravel()
        coefs, gaps, iters = _enet_path_1d(Xa, ya, l1_ratio, al, use_gram, G, xy, c0, tol,
                                           max_iter, rng, random, positive)
        if return_n_iter:
            return al, coefs, gaps, iters
        else:
            return al, coefs, gaps
    else:
        Y = _as2d_y(ya)
        c0 = np.zeros((Y.shape[1], p)) if coef_init is None else np.array(coef_init, dtype=float)
        coefs, gaps, iters = _enet_path_mt(Xa, Y, l1_ratio, al, c0, tol, max_iter, rng, random)
        if return_n_iter:
            return al, coefs, gaps, iters
        else:
            return al, coefs, gaps

def lasso_path(X, y, *, eps: float = 1e-3, alphas = 100, precompute = "auto", Xy = None,
               copy_X: bool = True, coef_init = None, verbose = False,
               return_n_iter: Literal[bool] = False, positive: bool = False,
               tol: float = 1e-4, max_iter: int = 1000, random_state = None,
               selection: str = "cyclic"):
    return enet_path(X, y, l1_ratio=1.0, eps=eps, alphas=alphas, precompute=precompute, Xy=Xy,
                     copy_X=copy_X, coef_init=coef_init, verbose=verbose,
                     return_n_iter=return_n_iter, positive=positive, tol=tol,
                     max_iter=max_iter, random_state=random_state, selection=selection)

def _sw_or_none(sample_weight, n: int):
    """sklearn treats a scalar sample_weight as None in the enet models."""
    if sample_weight is None:
        return np.ones(n), False
    elif isinstance(sample_weight, Optional):
        return _sw_or_none(sample_weight.__val__(), n)
    elif isinstance(sample_weight, float) or isinstance(sample_weight, int):
        return np.ones(n), False
    else:
        return _check_sw(sample_weight, n), True

def _pre_fit(X, Y: np.ndarray[float, 2], precompute, fit_intercept: bool, sw: np.ndarray[float, 1],
             has_sw: bool):
    """Dense _pre_fit: returns (X, Y, X_offset, y_offset, use_gram, Gram, Xy (p, k))."""
    Xc, Yc, X_offset, y_offset, _ = _preprocess_data(X, Y, fit_intercept, sw, has_sw, True)
    n, p = Xc.shape
    use_gram = False
    G = np.zeros((0, 0))
    if isinstance(precompute, str):
        use_gram = n > p
        if use_gram:
            G = Xc.T @ Xc
    elif isinstance(precompute, bool):
        use_gram = precompute
        if use_gram:
            G = Xc.T @ Xc
    else:
        use_gram = True
        G = np.asarray(precompute, dtype=float)
        if fit_intercept and not np.allclose(X_offset, np.zeros(p)):
            G = Xc.T @ Xc
    Xy = np.zeros((0, 0))
    if use_gram:
        Xy = Xc.T @ Yc
    return Xc, Yc, X_offset, y_offset, use_gram, G, Xy

class _EnetMixin:
    def _nd(self) -> Literal[int]:
        return self._D()

    @property
    def n_iter_(self):
        if self._D() == 1:
            return self._n_iter[0]
        else:
            return self._n_iter.copy()

    @property
    def dual_gap_(self):
        if self._D() == 1:
            return float(self._dual_gap[0])
        else:
            return self._dual_gap.copy()

    @property
    def sparse_coef_(self):
        return self.coef_

    def _fit_enet(self, X, ya, sample_weight, l1_ratio: float):
        Y = _as2d_y(ya)
        n = Y.shape[0]
        sw, has_sw = _sw_or_none(sample_weight, n)
        if has_sw:
            sw = sw * (float(n) / float(np.sum(sw)))
        Xc, Yc, X_offset, y_offset, use_gram, G, Xy = _pre_fit(X, Y, self.precompute,
                                                               self.fit_intercept, sw, has_sw)
        p = Xc.shape[1]
        k = Yc.shape[1]
        self.n_features_in_ = p
        coef = np.zeros((k, p))
        if self.warm_start and self._coef.shape[0] == k and self._coef.shape[1] == p:
            coef = self._coef.copy()
        gaps = np.zeros(k)
        iters = List[int]()
        rng = check_random_state(self.random_state)
        if self.selection != "random" and self.selection != "cyclic":
            raise ValueError("selection should be either random or cyclic.")
        al = np.array([self.alpha])
        for t in range(k):
            yt = np.ascontiguousarray(Yc[:, t])
            xyt = np.ascontiguousarray(Xy[:, t]) if use_gram else np.zeros(0)
            c, g, it = _enet_path_1d(Xc, yt, l1_ratio, al, use_gram, G, xyt,
                                     np.ascontiguousarray(coef[t]), self.tol, self.max_iter,
                                     rng, self.selection == "random", self.positive)
            coef[t] = c[:, 0]
            gaps[t] = g[0]
            iters.append(it[0])
        self._coef = coef
        self._dual_gap = gaps
        self._n_iter = iters
        self._intercept = _set_intercept(coef, X_offset, y_offset, self.fit_intercept)
        if not (np.all(np.isfinite(self._coef)) and np.all(np.isfinite(self._intercept))):
            raise ValueError("Coordinate descent iterations resulted in non-finite parameter values. The input data may contain large values and need to be preprocessed.")
        return self

class ElasticNet[D: Literal[int] = 1, PC = bool](_EnetMixin, _LinearRegressorMixin):
    alpha: float
    l1_ratio: float
    fit_intercept: bool
    precompute: PC
    max_iter: int
    copy_X: bool
    tol: float
    warm_start: bool
    positive: bool
    random_state: Optional[int]
    selection: str
    _coef: np.ndarray[float, 2]
    _intercept: np.ndarray[float, 1]
    _dual_gap: np.ndarray[float, 1]
    _n_iter: List[int]
    n_features_in_: int

    def __init__(self, alpha: float = 1.0, *, l1_ratio: float = 0.5, fit_intercept: bool = True,
                 precompute: PC = False, max_iter: int = 1000, copy_X: bool = True,
                 tol: float = 1e-4, warm_start: bool = False, positive: bool = False,
                 random_state: Optional[int] = None, selection: str = "cyclic"):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept
        self.precompute = precompute
        self.max_iter = max_iter
        self.copy_X = copy_X
        self.tol = tol
        self.warm_start = warm_start
        self.positive = positive
        self.random_state = random_state
        self.selection = selection
        self._coef = np.zeros((0, 0))
        self._intercept = np.zeros(0)
        self._dual_gap = np.zeros(0)
        self._n_iter = List[int]()
        self.n_features_in_ = 0

    def _D(self) -> Literal[int]:
        return D

    def path(self, X, y, **kw):
        return enet_path(X, y, **kw)

    def fit(self, X, y, sample_weight = None, check_input: bool = True):
        ya = np.asarray(y, dtype=float)
        if _ndim_of(ya) != D:
            return _retype(self, ElasticNet[_ndim_of(ya), PC]).fit(X, ya, sample_weight)
        return self._fit_enet(X, ya, sample_weight, self.l1_ratio)

class Lasso[D: Literal[int] = 1, PC = bool](_EnetMixin, _LinearRegressorMixin):
    alpha: float
    fit_intercept: bool
    precompute: PC
    max_iter: int
    copy_X: bool
    tol: float
    warm_start: bool
    positive: bool
    random_state: Optional[int]
    selection: str
    _coef: np.ndarray[float, 2]
    _intercept: np.ndarray[float, 1]
    _dual_gap: np.ndarray[float, 1]
    _n_iter: List[int]
    n_features_in_: int

    def __init__(self, alpha: float = 1.0, *, fit_intercept: bool = True, precompute: PC = False,
                 copy_X: bool = True, max_iter: int = 1000, tol: float = 1e-4,
                 warm_start: bool = False, positive: bool = False,
                 random_state: Optional[int] = None, selection: str = "cyclic"):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.precompute = precompute
        self.max_iter = max_iter
        self.copy_X = copy_X
        self.tol = tol
        self.warm_start = warm_start
        self.positive = positive
        self.random_state = random_state
        self.selection = selection
        self._coef = np.zeros((0, 0))
        self._intercept = np.zeros(0)
        self._dual_gap = np.zeros(0)
        self._n_iter = List[int]()
        self.n_features_in_ = 0

    @property
    def l1_ratio(self) -> float:
        return 1.0

    def _D(self) -> Literal[int]:
        return D

    def path(self, X, y, **kw):
        return lasso_path(X, y, **kw)

    def fit(self, X, y, sample_weight = None, check_input: bool = True):
        ya = np.asarray(y, dtype=float)
        if _ndim_of(ya) != D:
            return _retype(self, Lasso[_ndim_of(ya), PC]).fit(X, ya, sample_weight)
        return self._fit_enet(X, ya, sample_weight, 1.0)

class _MultiTaskMixin:
    @property
    def coef_(self):
        return self._coef.copy()

    @property
    def intercept_(self):
        return self._intercept.copy()

    def _nd(self) -> Literal[int]:
        return 2

    @property
    def sparse_coef_(self):
        return self._coef.copy()

    def _fit_mt(self, X, y, sample_weight, l1_ratio: float):
        ya = np.asarray(y, dtype=float)
        if _ndim_of(ya) == 1:
            raise ValueError("For mono-task outputs, use ElasticNet")
        Y = _as2d_y(ya)
        n = Y.shape[0]
        sw, has_sw = _sw_or_none(sample_weight, n)
        if has_sw:
            sw = sw * (float(n) / float(np.sum(sw)))
        Xc, Yc, X_offset, y_offset, _ = _preprocess_data(X, Y, self.fit_intercept, sw, has_sw, True)
        p = Xc.shape[1]
        k = Yc.shape[1]
        self.n_features_in_ = p
        W = np.zeros((k, p))
        if self.warm_start and self._coef.shape[0] == k and self._coef.shape[1] == p:
            W = self._coef.copy()
        _, gap, eps, it = enet_coordinate_descent_multi_task(
            W, self.alpha * l1_ratio * float(n), self.alpha * (1.0 - l1_ratio) * float(n),
            Xc, Yc, self.max_iter, self.tol, check_random_state(self.random_state),
            self.selection == "random")
        self._coef = W
        self.dual_gap_ = gap / float(n)
        self.eps_ = eps
        self.n_iter_ = it
        self._intercept = _set_intercept(W, X_offset, y_offset, self.fit_intercept)
        return self

class MultiTaskElasticNet(_MultiTaskMixin, _LinearRegressorMixin):
    alpha: float
    l1_ratio: float
    fit_intercept: bool
    copy_X: bool
    max_iter: int
    tol: float
    warm_start: bool
    random_state: Optional[int]
    selection: str
    _coef: np.ndarray[float, 2]
    _intercept: np.ndarray[float, 1]
    dual_gap_: float
    eps_: float
    n_iter_: int
    n_features_in_: int

    def __init__(self, alpha: float = 1.0, *, l1_ratio: float = 0.5, fit_intercept: bool = True,
                 copy_X: bool = True, max_iter: int = 1000, tol: float = 1e-4,
                 warm_start: bool = False, random_state: Optional[int] = None,
                 selection: str = "cyclic"):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept
        self.copy_X = copy_X
        self.max_iter = max_iter
        self.tol = tol
        self.warm_start = warm_start
        self.random_state = random_state
        self.selection = selection
        self._coef = np.zeros((0, 0))
        self._intercept = np.zeros(0)
        self.dual_gap_ = 0.0
        self.eps_ = 0.0
        self.n_iter_ = 0
        self.n_features_in_ = 0

    def fit(self, X, y, sample_weight = None):
        return self._fit_mt(X, y, sample_weight, self.l1_ratio)

class MultiTaskLasso(_MultiTaskMixin, _LinearRegressorMixin):
    alpha: float
    fit_intercept: bool
    copy_X: bool
    max_iter: int
    tol: float
    warm_start: bool
    random_state: Optional[int]
    selection: str
    _coef: np.ndarray[float, 2]
    _intercept: np.ndarray[float, 1]
    dual_gap_: float
    eps_: float
    n_iter_: int
    n_features_in_: int

    def __init__(self, alpha: float = 1.0, *, fit_intercept: bool = True, copy_X: bool = True,
                 max_iter: int = 1000, tol: float = 1e-4, warm_start: bool = False,
                 random_state: Optional[int] = None, selection: str = "cyclic"):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.copy_X = copy_X
        self.max_iter = max_iter
        self.tol = tol
        self.warm_start = warm_start
        self.random_state = random_state
        self.selection = selection
        self._coef = np.zeros((0, 0))
        self._intercept = np.zeros(0)
        self.dual_gap_ = 0.0
        self.eps_ = 0.0
        self.n_iter_ = 0
        self.n_features_in_ = 0

    @property
    def l1_ratio(self) -> float:
        return 1.0

    def fit(self, X, y, sample_weight = None):
        return self._fit_mt(X, y, sample_weight, 1.0)

# ------------------------------------------------------------------ CV

def _regressor_folds(cv, n: int, X, y):
    from ..model_selection import KFold
    if cv is None:
        return [(tr, te) for tr, te in KFold(5).split(X, y)]
    elif isinstance(cv, int):
        return [(tr, te) for tr, te in KFold(cv).split(X, y)]
    else:
        return [(tr, te) for tr, te in cv.split(X, y)]

def _path_mse(X: np.ndarray[float, 2], Y: np.ndarray[float, 2], sw: np.ndarray[float, 1],
              has_sw: bool, tr: np.ndarray[int, 1], te: np.ndarray[int, 1], fit_intercept: bool,
              precompute, multitask: bool, alphas: np.ndarray[float, 1], l1_ratio: float,
              tol: float, max_iter: int, random_state, selection: str,
              positive: bool) -> np.ndarray[float, 1]:
    """_path_residuals: mean (over targets) test MSE for each alpha."""
    Xtr = X[tr]
    Ytr = Y[tr]
    Xte = X[te]
    Yte = Y[te]
    swtr = np.ones(tr.shape[0])
    swte = np.ones(te.shape[0])
    if has_sw:
        swtr = sw[tr]
        swte = sw[te]
        swtr = swtr * (float(tr.shape[0]) / float(np.sum(swtr)))
    rng = check_random_state(random_state)
    random = selection == "random"
    na = alphas.shape[0]
    k = Y.shape[1]
    p = X.shape[1]
    coefs = np.zeros((k, p, na))
    X_offset = np.zeros(p)
    y_offset = np.zeros(k)
    if multitask:
        Xc, Yc, X_offset, y_offset, _ = _preprocess_data(Xtr, Ytr, fit_intercept, swtr, has_sw, True)
        c, _, _ = _enet_path_mt(Xc, Yc, l1_ratio, alphas, np.zeros((k, p)), tol, max_iter, rng,
                                random)
        coefs = c
    else:
        Xc, Yc, X_offset, y_offset, use_gram, G, Xy = _pre_fit(Xtr, Ytr, precompute,
                                                               fit_intercept, swtr, has_sw)
        xy = np.ascontiguousarray(Xy[:, 0]) if use_gram else np.zeros(0)
        c, _, _ = _enet_path_1d(Xc, np.ascontiguousarray(Yc[:, 0]), l1_ratio, alphas, use_gram,
                                G, xy, np.zeros(p), tol, max_iter, rng, random, positive)
        coefs[0] = c
    nte = te.shape[0]
    mse = np.zeros(na)
    for a in range(na):
        tot = 0.0
        for t in range(k):
            inter = y_offset[t]
            for j in range(p):
                inter -= X_offset[j] * coefs[t, j, a]
            num = 0.0
            den = 0.0
            for i in range(nte):
                pr = inter
                for j in range(p):
                    pr += Xte[i, j] * coefs[t, j, a]
                r = pr - Yte[i, t]
                num += swte[i] * r * r
                den += swte[i]
            tot += num / den
        mse[a] = tot / float(k)
    return mse

class _CVMixin:
    @property
    def mse_path_(self):
        return self._mse_path.copy()

    @property
    def alphas_(self):
        return self._alphas.copy()

    def _fit_cv(self, X, y, sample_weight, l1_ratios: np.ndarray[float, 1], multitask: bool,
                positive: bool, precompute):
        Xa = np.asarray(X, dtype=float)
        ya = np.asarray(y, dtype=float)
        Y = _as2d_y(ya)
        n, p = Xa.shape
        sw, has_sw = _sw_or_none(sample_weight, n)
        self.n_features_in_ = p
        nl = l1_ratios.shape[0]
        grids = List[np.ndarray[float, 1]]()
        for li in range(nl):
            if isinstance(self.alphas, int):
                grids.append(_alpha_grid(Xa, Y, l1_ratio=l1_ratios[li],
                                         fit_intercept=self.fit_intercept, positive=positive,
                                         eps=self.eps, n_alphas=self.alphas,
                                         sample_weight=sw if has_sw else None))
            else:
                al = _alphas_desc(self.alphas)
                for i in range(al.shape[0]):
                    if al[i] < 0.0:
                        raise ValueError(f"alphas[{i}] == {al[i]}, must be >= 0.0.")
                grids.append(al)
        folds = _regressor_folds(self.cv, n, Xa, ya)
        nf = len(folds)
        na = grids[0].shape[0]
        mse = np.zeros((nl, nf, na))
        for li in range(nl):
            for fi in range(nf):
                tr, te = folds[fi]
                mse[li, fi] = _path_mse(Xa, Y, sw, has_sw, tr, te, self.fit_intercept, precompute,
                                        multitask, grids[li], l1_ratios[li], self.tol,
                                        self.max_iter, self.random_state, self.selection,
                                        positive)
        best_mse = math.inf
        best_alpha = 0.0
        best_l1 = 1.0
        for li in range(nl):
            ib = 0
            for a in range(na):
                m = 0.0
                for fi in range(nf):
                    m += mse[li, fi, a]
                m /= float(nf)
                mb = 0.0
                for fi in range(nf):
                    mb += mse[li, fi, ib]
                mb /= float(nf)
                if m < mb:
                    ib = a
            this = 0.0
            for fi in range(nf):
                this += mse[li, fi, ib]
            this /= float(nf)
            if this < best_mse:
                best_mse = this
                best_alpha = grids[li][ib]
                best_l1 = l1_ratios[li]
        self.alpha_ = best_alpha
        self.l1_ratio_ = best_l1
        # mse_path_: squeeze(moveaxis(mse, 2, 1)) -> (n_l1, n_alphas, n_folds)
        mp = np.zeros((nl * na, nf))
        for li in range(nl):
            for a in range(na):
                for fi in range(nf):
                    mp[li * na + a, fi] = mse[li, fi, a]
        self._mse_path = mp
        if isinstance(self.alphas, int):
            ag = np.zeros((nl, na))
            for li in range(nl):
                ag[li] = grids[li]
            self._alphas = ag
        else:
            self._alphas = grids[0].reshape((1, na))
        return Y, sw, has_sw, best_alpha, best_l1

class LassoCV[CV = NoneType, AL = int, PC = str](_CVMixin, _LinearRegressorMixin):
    eps: float
    alphas: AL
    fit_intercept: bool
    precompute: PC
    max_iter: int
    tol: float
    copy_X: bool
    cv: CV
    positive: bool
    random_state: Optional[int]
    selection: str
    alpha_: float
    l1_ratio_: float
    _alphas: np.ndarray[float, 2]
    _mse_path: np.ndarray[float, 2]
    _coef: np.ndarray[float, 2]
    _intercept: np.ndarray[float, 1]
    dual_gap_: float
    n_iter_: int
    n_features_in_: int

    def __init__(self, *, eps: float = 1e-3, alphas: AL = 100, fit_intercept: bool = True,
                 precompute: PC = "auto", max_iter: int = 1000, tol: float = 1e-4,
                 copy_X: bool = True, cv: CV = None, verbose = False, n_jobs = None,
                 positive: bool = False, random_state: Optional[int] = None,
                 selection: str = "cyclic"):
        self.eps = eps
        self.alphas = alphas
        self.fit_intercept = fit_intercept
        self.precompute = precompute
        self.max_iter = max_iter
        self.tol = tol
        self.copy_X = copy_X
        self.cv = cv
        self.positive = positive
        self.random_state = random_state
        self.selection = selection
        self.alpha_ = 0.0
        self.l1_ratio_ = 1.0
        self._alphas = np.zeros((0, 0))
        self._mse_path = np.zeros((0, 0))
        self._coef = np.zeros((0, 0))
        self._intercept = np.zeros(0)
        self.dual_gap_ = 0.0
        self.n_iter_ = 0
        self.n_features_in_ = 0

    def _nd(self) -> Literal[int]:
        return 1

    @property
    def alphas_(self):
        return self._alphas[0].copy()

    def fit(self, X, y, sample_weight = None):
        Y, sw, has_sw, a, l1 = self._fit_cv(X, y, sample_weight, np.array([1.0]), False,
                                            self.positive, self.precompute)
        pc = False if isinstance(self.precompute, str) else self.precompute
        m = Lasso[1, type(pc)](a, fit_intercept=self.fit_intercept, precompute=pc,
                               max_iter=self.max_iter, tol=self.tol, positive=self.positive,
                               random_state=self.random_state, selection=self.selection)
        m.fit(X, np.ascontiguousarray(Y[:, 0]), sw if has_sw else None)
        self._coef = m._coef
        self._intercept = m._intercept
        self.dual_gap_ = m.dual_gap_
        self.n_iter_ = m.n_iter_
        return self

class ElasticNetCV[CV = NoneType, AL = int, PC = str, LR = float](_CVMixin, _LinearRegressorMixin):
    l1_ratio: LR
    eps: float
    alphas: AL
    fit_intercept: bool
    precompute: PC
    max_iter: int
    tol: float
    copy_X: bool
    cv: CV
    positive: bool
    random_state: Optional[int]
    selection: str
    alpha_: float
    l1_ratio_: float
    _alphas: np.ndarray[float, 2]
    _mse_path: np.ndarray[float, 2]
    _coef: np.ndarray[float, 2]
    _intercept: np.ndarray[float, 1]
    dual_gap_: float
    n_iter_: int
    n_features_in_: int

    def __init__(self, *, l1_ratio: LR = 0.5, eps: float = 1e-3, alphas: AL = 100,
                 fit_intercept: bool = True, precompute: PC = "auto", max_iter: int = 1000,
                 tol: float = 1e-4, cv: CV = None, copy_X: bool = True, verbose = 0,
                 n_jobs = None, positive: bool = False, random_state: Optional[int] = None,
                 selection: str = "cyclic"):
        self.l1_ratio = l1_ratio
        self.eps = eps
        self.alphas = alphas
        self.fit_intercept = fit_intercept
        self.precompute = precompute
        self.max_iter = max_iter
        self.tol = tol
        self.copy_X = copy_X
        self.cv = cv
        self.positive = positive
        self.random_state = random_state
        self.selection = selection
        self.alpha_ = 0.0
        self.l1_ratio_ = 0.0
        self._alphas = np.zeros((0, 0))
        self._mse_path = np.zeros((0, 0))
        self._coef = np.zeros((0, 0))
        self._intercept = np.zeros(0)
        self.dual_gap_ = 0.0
        self.n_iter_ = 0
        self.n_features_in_ = 0

    def _nd(self) -> Literal[int]:
        return 1

    @property
    def alphas_(self):
        if self._alphas.shape[0] == 1:
            return self._alphas[0].copy()
        return self._alphas.copy()

    def fit(self, X, y, sample_weight = None):
        lrs = np.atleast_1d(np.asarray(self.l1_ratio, dtype=float)).ravel()
        Y, sw, has_sw, a, l1 = self._fit_cv(X, y, sample_weight, lrs, False, self.positive,
                                            self.precompute)
        pc = False if isinstance(self.precompute, str) else self.precompute
        m = ElasticNet[1, type(pc)](a, l1_ratio=l1, fit_intercept=self.fit_intercept,
                                    precompute=pc, max_iter=self.max_iter, tol=self.tol,
                                    positive=self.positive, random_state=self.random_state,
                                    selection=self.selection)
        m.fit(X, np.ascontiguousarray(Y[:, 0]), sw if has_sw else None)
        self._coef = m._coef
        self._intercept = m._intercept
        self.dual_gap_ = m.dual_gap_
        self.n_iter_ = m.n_iter_
        return self

class MultiTaskElasticNetCV[CV = NoneType, AL = int, LR = float](_CVMixin, _MultiTaskMixin, _LinearRegressorMixin):
    l1_ratio: LR
    eps: float
    alphas: AL
    fit_intercept: bool
    max_iter: int
    tol: float
    copy_X: bool
    cv: CV
    random_state: Optional[int]
    selection: str
    alpha_: float
    l1_ratio_: float
    _alphas: np.ndarray[float, 2]
    _mse_path: np.ndarray[float, 2]
    _coef: np.ndarray[float, 2]
    _intercept: np.ndarray[float, 1]
    dual_gap_: float
    n_iter_: int
    n_features_in_: int

    def __init__(self, *, l1_ratio: LR = 0.5, eps: float = 1e-3, alphas: AL = 100,
                 fit_intercept: bool = True, max_iter: int = 1000, tol: float = 1e-4,
                 cv: CV = None, copy_X: bool = True, verbose = 0, n_jobs = None,
                 random_state: Optional[int] = None, selection: str = "cyclic"):
        self.l1_ratio = l1_ratio
        self.eps = eps
        self.alphas = alphas
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.copy_X = copy_X
        self.cv = cv
        self.random_state = random_state
        self.selection = selection
        self.alpha_ = 0.0
        self.l1_ratio_ = 0.0
        self._alphas = np.zeros((0, 0))
        self._mse_path = np.zeros((0, 0))
        self._coef = np.zeros((0, 0))
        self._intercept = np.zeros(0)
        self.dual_gap_ = 0.0
        self.n_iter_ = 0
        self.n_features_in_ = 0

    @property
    def alphas_(self):
        if self._alphas.shape[0] == 1:
            return self._alphas[0].copy()
        return self._alphas.copy()

    def fit(self, X, y, sample_weight = None):
        lrs = np.atleast_1d(np.asarray(self.l1_ratio, dtype=float)).ravel()
        Y, sw, has_sw, a, l1 = self._fit_cv(X, y, sample_weight, lrs, True, False, False)
        m = MultiTaskElasticNet(a, l1_ratio=l1, fit_intercept=self.fit_intercept,
                                max_iter=self.max_iter, tol=self.tol,
                                random_state=self.random_state, selection=self.selection)
        m.fit(X, Y, sw if has_sw else None)
        self._coef = m._coef
        self._intercept = m._intercept
        self.dual_gap_ = m.dual_gap_
        self.n_iter_ = m.n_iter_
        return self

class MultiTaskLassoCV[CV = NoneType, AL = int](_CVMixin, _MultiTaskMixin, _LinearRegressorMixin):
    eps: float
    alphas: AL
    fit_intercept: bool
    max_iter: int
    tol: float
    copy_X: bool
    cv: CV
    random_state: Optional[int]
    selection: str
    alpha_: float
    l1_ratio_: float
    _alphas: np.ndarray[float, 2]
    _mse_path: np.ndarray[float, 2]
    _coef: np.ndarray[float, 2]
    _intercept: np.ndarray[float, 1]
    dual_gap_: float
    n_iter_: int
    n_features_in_: int

    def __init__(self, *, eps: float = 1e-3, alphas: AL = 100, fit_intercept: bool = True,
                 max_iter: int = 1000, tol: float = 1e-4, copy_X: bool = True, cv: CV = None,
                 verbose = False, n_jobs = None, random_state: Optional[int] = None,
                 selection: str = "cyclic"):
        self.eps = eps
        self.alphas = alphas
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.copy_X = copy_X
        self.cv = cv
        self.random_state = random_state
        self.selection = selection
        self.alpha_ = 0.0
        self.l1_ratio_ = 1.0
        self._alphas = np.zeros((0, 0))
        self._mse_path = np.zeros((0, 0))
        self._coef = np.zeros((0, 0))
        self._intercept = np.zeros(0)
        self.dual_gap_ = 0.0
        self.n_iter_ = 0
        self.n_features_in_ = 0

    @property
    def alphas_(self):
        return self._alphas[0].copy()

    def fit(self, X, y, sample_weight = None):
        Y, sw, has_sw, a, l1 = self._fit_cv(X, y, sample_weight, np.array([1.0]), True, False,
                                            False)
        m = MultiTaskLasso(a, fit_intercept=self.fit_intercept, max_iter=self.max_iter,
                           tol=self.tol, random_state=self.random_state,
                           selection=self.selection)
        m.fit(X, Y, sw if has_sw else None)
        self._coef = m._coef
        self._intercept = m._intercept
        self.dual_gap_ = m.dual_gap_
        self.n_iter_ = m.n_iter_
        return self
