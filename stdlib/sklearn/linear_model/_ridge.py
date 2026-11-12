# Ported from scikit-learn 1.9.1 (BSD-3-Clause)
#
# Ridge, RidgeClassifier, RidgeCV, RidgeClassifierCV, ridge_regression.
# Dense inputs only. Internally targets are always 2-d (n_samples, n_targets).
# DEVIATIONS: Ridge fitted on a 2-d y with one column returns 2-d coef_
# (sklearn ravels it); RidgeClassifier* always expose a 2-d coef_
# (1, n_features) for binary problems (sklearn returns 1-d there).
# RidgeCV: scoring must be None (GCV MSE, or the estimator's score for cv!=None).

import math
import numpy as np
import scipy.linalg as sl
from ._base import (_retype, _ndim_of, _as2d_y, _check_sw, _preprocess_data,
                    _set_intercept, _LinearRegressorMixin, _LinearClassifierMixin,
                    _class_weight_vec)
from ._sag import sag_solver

def _solve_cholesky(X, Y, alpha: np.ndarray[float, 1]):
    p = X.shape[1]
    k = Y.shape[1]
    A = X.T @ X
    Xy = X.T @ Y
    one_alpha = True
    for i in range(alpha.shape[0]):
        if alpha[i] != alpha[0]:
            one_alpha = False
    if one_alpha:
        for i in range(p):
            A[i, i] += alpha[0]
        return np.ascontiguousarray(sl.solve(A, Xy, assume_a="pos").T)
    else:
        coefs = np.empty((k, p))
        for t in range(k):
            for i in range(p):
                A[i, i] += alpha[t]
            coefs[t] = sl.solve(A, np.ascontiguousarray(Xy[:, t]), assume_a="pos")
            for i in range(p):
                A[i, i] -= alpha[t]
        return coefs

def _solve_cholesky_kernel(K, Y, alpha: np.ndarray[float, 1]):
    n = K.shape[0]
    k = Y.shape[1]
    one_alpha = True
    for i in range(alpha.shape[0]):
        if alpha[i] != alpha[0]:
            one_alpha = False
    Kc = K.copy()
    if one_alpha:
        for i in range(n):
            Kc[i, i] += alpha[0]
        return sl.solve(Kc, Y, assume_a="pos")
    else:
        dual = np.empty((k, n))
        for t in range(k):
            for i in range(n):
                Kc[i, i] += alpha[t]
            dual[t] = sl.solve(Kc, np.ascontiguousarray(Y[:, t]), assume_a="pos")
            for i in range(n):
                Kc[i, i] -= alpha[t]
        return np.ascontiguousarray(dual.T)

def _solve_svd(X, Y, alpha: np.ndarray[float, 1]):
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    m = s.shape[0]
    UTy = U.T @ Y
    d = np.zeros((m, alpha.shape[0]))
    for i in range(m):
        if s[i] > 1e-15:
            for t in range(alpha.shape[0]):
                d[i, t] = s[i] / (s[i] ** 2 + alpha[t])
    return np.ascontiguousarray((Vt.T @ (d * UTy)).T)

class _GramMV:
    X: np.ndarray[float, 2]
    a: float
    def __init__(self, X: np.ndarray[float, 2], a: float):
        self.X = X
        self.a = a
    def __call__(self, x):
        return self.X @ (self.X.T @ x) + self.a * x

class _CovMV:
    X: np.ndarray[float, 2]
    a: float
    def __init__(self, X: np.ndarray[float, 2], a: float):
        self.X = X
        self.a = a
    def __call__(self, x):
        return self.X.T @ (self.X @ x) + self.a * x

def _solve_sparse_cg(X, Y, alpha: np.ndarray[float, 1], max_iter: Optional[int], tol: float):
    from scipy.sparse.linalg import LinearOperator, cg
    n, p = X.shape
    k = Y.shape[1]
    coefs = np.empty((k, p))
    for t in range(k):
        yc = np.ascontiguousarray(Y[:, t])
        if p > n:
            C = LinearOperator((n, n), matvec=_GramMV(X, alpha[t]))
            c, info = cg(C, yc, rtol=tol)
            if info < 0:
                raise ValueError(f"Failed with error code {info}")
            coefs[t] = X.T @ c
        else:
            C = LinearOperator((p, p), matvec=_CovMV(X, alpha[t]))
            c, info = cg(C, X.T @ yc, maxiter=max_iter, rtol=tol)
            if info < 0:
                raise ValueError(f"Failed with error code {info}")
            coefs[t] = c
    return coefs

def _solve_lsqr(X, Y, alpha: np.ndarray[float, 1], max_iter: Optional[int], tol: float):
    from scipy.sparse.linalg import lsqr
    n, p = X.shape
    k = Y.shape[1]
    coefs = np.empty((k, p))
    n_iter = np.empty(k, dtype=int)
    for t in range(k):
        res = lsqr(X, np.ascontiguousarray(Y[:, t]), damp=math.sqrt(alpha[t]), atol=tol,
                   btol=tol, iter_lim=max_iter)
        coefs[t] = res[0]
        n_iter[t] = res[2]
    return coefs, n_iter

class _RidgeLbfgsFun:
    X: np.ndarray[float, 2]
    y: np.ndarray[float, 1]
    a: float
    def __init__(self, X: np.ndarray[float, 2], y: np.ndarray[float, 1], a: float):
        self.X = X
        self.y = y
        self.a = a
    def fun(self, w):
        r = self.X @ w - self.y
        return 0.5 * float(np.dot(r, r)) + 0.5 * self.a * float(np.dot(w, w))
    def grad(self, w):
        r = self.X @ w - self.y
        return self.X.T @ r + self.a * w

def _solve_lbfgs(X, Y, alpha: np.ndarray[float, 1], positive: bool, max_iter: Optional[int], tol: float):
    from scipy.optimize import minimize
    n, p = X.shape
    k = Y.shape[1]
    coefs = np.empty((k, p))
    bounds = [(0.0, math.inf) for _ in range(p)] if positive else [(-math.inf, math.inf) for _ in range(p)]
    for t in range(k):
        f = _RidgeLbfgsFun(X, np.ascontiguousarray(Y[:, t]), alpha[t])
        opts = {"maxiter": float(max_iter.__val__())} if max_iter is not None else Dict[str, float]()
        res = minimize(f.fun, np.zeros(p), method="L-BFGS-B", jac=f.grad, bounds=bounds,
                       tol=tol, options=opts)
        coefs[t] = res.x
    return coefs

def _resolve_solver(solver: str, positive: bool, return_intercept: bool) -> str:
    if solver != "auto":
        return solver
    if positive:
        return "lbfgs"
    if return_intercept:
        return "sag"
    return "cholesky"

def _alpha_vec(alpha, k: int) -> np.ndarray[float, 1]:
    a = np.atleast_1d(np.asarray(alpha, dtype=float)).ravel()
    if a.shape[0] != 1 and a.shape[0] != k:
        raise ValueError(f"Number of targets and number of penalties do not correspond: {a.shape[0]} != {k}")
    if a.shape[0] == 1 and k > 1:
        return np.full(k, a[0])
    return a.copy()

def _ridge_core(X, Y: np.ndarray[float, 2], alpha, sw: np.ndarray[float, 1], has_sw: bool,
                solver: str, max_iter: Optional[int], tol: float, positive: bool,
                random_state, return_intercept: bool):
    """_ridge_regression on dense 2-d Y; returns (coef (k,p), n_iter or None,
    intercept (k,), solver)."""
    solver = _resolve_solver(solver, positive, return_intercept)
    if positive and solver != "lbfgs":
        raise ValueError(f"When positive=True, only 'lbfgs' solver can be used. Please change solver {solver} to 'lbfgs' or set positive=False.")
    if solver == "lbfgs" and not positive:
        raise ValueError("'lbfgs' solver can be used only when positive=True. Please use another solver.")
    if return_intercept and solver != "sag":
        raise ValueError("In Ridge, only 'sag' solver can directly fit the intercept. Please change solver to 'sag' or set return_intercept=False.")
    Xa = np.array(X, dtype=float)
    Ya = Y.copy()
    n, p = Xa.shape
    k = Ya.shape[1]
    if has_sw and solver != "sag" and solver != "saga":
        sq = np.sqrt(sw)
        Xa *= sq[:, None]
        Ya *= sq[:, None]
    a = _alpha_vec(alpha, k)
    n_iter: Optional[np.ndarray[int, 1]] = None
    intercept = np.zeros(k)
    coef = np.zeros((k, p))
    if solver == "sparse_cg":
        coef = _solve_sparse_cg(Xa, Ya, a, max_iter, tol)
    elif solver == "lsqr":
        coef, ni = _solve_lsqr(Xa, Ya, a, max_iter, tol)
        n_iter = ni
    elif solver == "cholesky":
        try:
            if p > n:
                K = Xa @ Xa.T
                dual = _solve_cholesky_kernel(K, Ya, a)
                coef = np.ascontiguousarray((Xa.T @ dual).T)
            else:
                coef = _solve_cholesky(Xa, Ya, a)
        except np.linalg.LinAlgError:
            solver = "svd"
    elif solver == "sag" or solver == "saga":
        mss = 0.0
        for i in range(n):
            s = 0.0
            for j in range(p):
                s += Xa[i, j] * Xa[i, j]
            mss = max(mss, s)
        ni = np.empty(k, dtype=int)
        mi = 1000 if max_iter is None else max_iter.__val__()
        swv = sw if has_sw else np.ones(n)
        for t in range(k):
            ci = np.zeros((p, 1))
            ii = np.zeros(1)
            ni[t] = sag_solver(Xa, np.ascontiguousarray(Ya[:, t]), swv, "squared", a[t], 0.0, mi,
                               tol, random_state, ci, ii, return_intercept, solver == "saga", mss)
            coef[t] = ci[:, 0]
            intercept[t] = ii[0]
        n_iter = ni
    elif solver == "lbfgs":
        coef = _solve_lbfgs(Xa, Ya, a, positive, max_iter, tol)
    elif solver != "svd":
        raise ValueError(f"Known solvers are 'sparse_cg', 'cholesky', 'svd' 'lsqr', 'sag', 'saga' or 'lbfgs'. Got {solver}.")
    if solver == "svd":
        coef = _solve_svd(Xa, Ya, a)
    return coef, n_iter, intercept, solver

def ridge_regression(X, y, alpha, *, sample_weight = None, solver: str = "auto",
                     max_iter: Optional[int] = None, tol: float = 1e-4, verbose: int = 0,
                     positive: bool = False, random_state = None,
                     return_n_iter: Literal[bool] = False,
                     return_intercept: Literal[bool] = False, check_input: bool = True):
    ya = np.asarray(y, dtype=float)
    Y = _as2d_y(ya)
    n = Y.shape[0]
    sw = _check_sw(sample_weight, n)
    coef, n_iter, intercept, _ = _ridge_core(X, Y, alpha, sw, sample_weight is not None, solver,
                                             max_iter, tol, positive, random_state, return_intercept)
    if _ndim_of(ya) == 1:
        c1 = np.ascontiguousarray(coef[0])
        if return_n_iter and return_intercept:
            return c1, n_iter, float(intercept[0])
        elif return_intercept:
            return c1, float(intercept[0])
        elif return_n_iter:
            return c1, n_iter
        else:
            return c1
    else:
        if return_n_iter and return_intercept:
            return coef, n_iter, intercept
        elif return_intercept:
            return coef, intercept
        elif return_n_iter:
            return coef, n_iter
        else:
            return coef

class Ridge[D: Literal[int] = 1](_LinearRegressorMixin):
    alpha: np.ndarray[float, 1]
    fit_intercept: bool
    copy_X: bool
    max_iter: Optional[int]
    tol: float
    solver: str
    positive: bool
    random_state: Optional[int]
    _coef: np.ndarray[float, 2]
    _intercept: np.ndarray[float, 1]
    n_iter_: Optional[np.ndarray[int, 1]]
    solver_: str
    n_features_in_: int

    def __init__(self, alpha = 1.0, *, fit_intercept: bool = True, copy_X: bool = True,
                 max_iter: Optional[int] = None, tol: float = 1e-4, solver: str = "auto",
                 positive: bool = False, random_state: Optional[int] = None):
        self.alpha = np.atleast_1d(np.asarray(alpha, dtype=float)).ravel()
        self.fit_intercept = fit_intercept
        self.copy_X = copy_X
        self.max_iter = max_iter
        self.tol = tol
        self.solver = solver
        self.positive = positive
        self.random_state = random_state
        self._coef = np.zeros((0, 0))
        self._intercept = np.zeros(0)
        self.n_iter_ = None
        self.solver_ = ""
        self.n_features_in_ = 0

    def _nd(self) -> Literal[int]:
        return D

    def _fit2d(self, X, Y: np.ndarray[float, 2], sample_weight):
        if self.solver == "lbfgs" and not self.positive:
            raise ValueError("'lbfgs' solver can be used only when positive=True. Please use another solver.")
        if self.positive and self.solver != "auto" and self.solver != "lbfgs":
            raise ValueError(f"solver='{self.solver}' does not support positive fitting. Please set the solver to 'auto' or 'lbfgs', or set `positive=False`")
        n = Y.shape[0]
        sw = _check_sw(sample_weight, n)
        has_sw = sample_weight is not None
        Xc, Yc, X_offset, y_offset, _ = _preprocess_data(X, Y, self.fit_intercept, sw, has_sw, False)
        self.n_features_in_ = Xc.shape[1]
        coef, n_iter, _, solver = _ridge_core(Xc, Yc, self.alpha, sw, has_sw, self.solver,
                                              self.max_iter, self.tol, self.positive,
                                              self.random_state, False)
        self._coef = coef
        self.n_iter_ = n_iter
        self.solver_ = solver
        self._intercept = _set_intercept(coef, X_offset, y_offset, self.fit_intercept)
        return self

    def fit(self, X, y, sample_weight = None):
        ya = np.asarray(y, dtype=float)
        if _ndim_of(ya) != D:
            return _retype(self, Ridge[_ndim_of(ya)]).fit(X, ya, sample_weight)
        return self._fit2d(X, _as2d_y(ya), sample_weight)

def _label_binarize_pm1(y):
    """LabelBinarizer(pos_label=1, neg_label=-1) on 1-d labels:
    returns (classes, Y) with Y (n, 1) for binary and (n, K) otherwise."""
    ya = np.asarray(y)
    classes, yi = np.unique(ya, return_inverse=True)
    K = classes.shape[0]
    n = ya.shape[0]
    if K == 2:
        Y = np.full((n, 1), -1.0)
        for i in range(n):
            if yi[i] == 1:
                Y[i, 0] = 1.0
        return classes, yi, Y
    else:
        Y = np.full((n, K), -1.0)
        for i in range(n):
            Y[i, yi[i]] = 1.0
        return classes, yi, Y

class RidgeClassifier[CW](_LinearClassifierMixin):
    alpha: np.ndarray[float, 1]
    fit_intercept: bool
    copy_X: bool
    max_iter: Optional[int]
    tol: float
    class_weight: CW
    solver: str
    positive: bool
    random_state: Optional[int]
    classes_: np.ndarray[int, 1]
    _coef: np.ndarray[float, 2]
    _intercept: np.ndarray[float, 1]
    n_iter_: Optional[np.ndarray[int, 1]]
    solver_: str
    n_features_in_: int

    def __init__(self, alpha = 1.0, *, fit_intercept: bool = True, copy_X: bool = True,
                 max_iter: Optional[int] = None, tol: float = 1e-4, class_weight: CW = None,
                 solver: str = "auto", positive: bool = False, random_state: Optional[int] = None):
        self.alpha = np.atleast_1d(np.asarray(alpha, dtype=float)).ravel()
        self.fit_intercept = fit_intercept
        self.copy_X = copy_X
        self.max_iter = max_iter
        self.tol = tol
        self.class_weight = class_weight
        self.solver = solver
        self.positive = positive
        self.random_state = random_state
        self.classes_ = np.zeros(0, dtype=int)
        self._coef = np.zeros((0, 0))
        self._intercept = np.zeros(0)
        self.n_iter_ = None
        self.solver_ = ""
        self.n_features_in_ = 0

    def fit(self, X, y, sample_weight = None):
        classes, yi, Y = _label_binarize_pm1(y)
        self.classes_ = classes
        n = Y.shape[0]
        sw = _check_sw(sample_weight, n)
        cw = _class_weight_vec(self.class_weight, classes, yi)
        for i in range(n):
            sw[i] *= cw[yi[i]]
        r = Ridge[2](self.alpha, fit_intercept=self.fit_intercept, max_iter=self.max_iter,
                     tol=self.tol, solver=self.solver, positive=self.positive,
                     random_state=self.random_state)
        r._fit2d(X, Y, sw)
        self._coef = r._coef
        self._intercept = r._intercept
        self.n_iter_ = r.n_iter_
        self.solver_ = r.solver_
        self.n_features_in_ = r.n_features_in_
        return self

def _gcv_solve(mode: str, alpha: float, Y: np.ndarray[float, 2], sqrt_sw: np.ndarray[float, 1],
               fit_intercept: bool, X: np.ndarray[float, 2], ev: np.ndarray[float, 1],
               Q: np.ndarray[float, 2], QT_y: np.ndarray[float, 2],
               QT_sw: np.ndarray[float, 1]):
    """looe (n, k) and coef (p, k) for one alpha (dense _RidgeGCV)."""
    n = Y.shape[0]
    if mode == "gram":
        Dv = 1.0 / (ev + alpha)
        c = Q @ (Dv[:, None] * QT_y)
        d = np.sum(Dv * Q ** 2, axis=1)
        if fit_intercept:
            sw_sum = float(sqrt_sw @ sqrt_sw)
            G = Q @ (Dv * QT_sw)
            d -= G * sqrt_sw / sw_sum
        looe = c / d[:, None]
        return looe, X.T @ c
    elif mode == "cov":
        # here Q = V (p, p), QT_y = X^T y (p, k), QT_sw = X^T sqrt_sw
        Dv = 1.0 / (ev + alpha)
        Hinv = (Q * Dv) @ Q.T
        Hinv_XT_y = Hinv @ QT_y
        Hinv_XT_sw = Hinv @ QT_sw
        X_Hinv_XT_y = X @ Hinv_XT_y
        X_Hinv_XT_sw = X @ Hinv_XT_sw
        alpha_c = Y - X_Hinv_XT_y
        XA = X @ Hinv
        alpha_d = 1.0 - np.sum(XA * X, axis=1)
        if fit_intercept:
            sw_sum = float(sqrt_sw @ sqrt_sw)
            agw = sqrt_sw - X_Hinv_XT_sw
            alpha_d -= agw * sqrt_sw / sw_sum
        return alpha_c / alpha_d[:, None], Hinv_XT_y
    else:
        # svd: ev = singvals, Q = U (n, r), QT_y = U^T y, QT_sw = U^T sqrt_sw; X holds V (p, r)
        V = X
        s = ev
        p = V.shape[0]
        coef = V @ ((s / (s ** 2 + alpha))[:, None] * QT_y)
        if n <= p:
            aD = alpha / (s ** 2 + alpha)
            alpha_c = Q @ (aD[:, None] * QT_y)
            alpha_d = np.sum(aD * Q ** 2, axis=1)
            if fit_intercept:
                sw_sum = float(sqrt_sw @ sqrt_sw)
                agw = Q @ (aD * QT_sw)
                alpha_d -= agw * sqrt_sw / sw_sum
            return alpha_c / alpha_d[:, None], coef
        else:
            M = alpha / (s ** 2 + alpha) - 1.0
            alpha_c = Q @ (M[:, None] * QT_y) + Y
            alpha_d = np.sum(M * Q ** 2, axis=1) + 1.0
            if fit_intercept:
                sw_sum = float(sqrt_sw @ sqrt_sw)
                agw = Q @ (M * QT_sw) + sqrt_sw
                alpha_d -= agw * sqrt_sw / sw_sum
            return alpha_c / alpha_d[:, None], coef

def _ridge_gcv(X, Y: np.ndarray[float, 2], alphas: np.ndarray[float, 1], sample_weight,
               fit_intercept: bool, gcv_mode: Optional[str], alpha_per_target: bool,
               store_cv_results: bool):
    """Returns (coef (k,p), intercept (k,), alpha_ (k,), best_score_ (k,), cv_results)."""
    n = Y.shape[0]
    sw = _check_sw(sample_weight, n)
    has_sw = sample_weight is not None
    Xc, Yc, X_offset, y_offset, sqrt_sw = _preprocess_data(X, Y, fit_intercept, sw, has_sw, True)
    p = Xc.shape[1]
    k = Yc.shape[1]
    mode = "svd" if (gcv_mode is not None and gcv_mode == "svd") else ("gram" if n <= p else "cov")
    ev = np.zeros(0)
    Q = np.zeros((0, 0))
    QT_y = np.zeros((0, 0))
    QT_sw = np.zeros(0)
    Xs = Xc
    if mode == "gram":
        K = Xc @ Xc.T
        ev, Q = np.linalg.eigh(K)
        QT_y = Q.T @ Yc
        QT_sw = Q.T @ sqrt_sw
    elif mode == "cov":
        C = Xc.T @ Xc
        ev, Q = np.linalg.eigh(C)
        QT_y = Xc.T @ Yc
        QT_sw = Xc.T @ sqrt_sw
    else:
        U, s, VT = np.linalg.svd(Xc, full_matrices=False)
        ev = s
        Q = U
        QT_y = U.T @ Yc
        QT_sw = U.T @ sqrt_sw
        Xs = np.ascontiguousarray(VT.T)
    na = alphas.shape[0]
    cvr = np.zeros((n * k, na))
    best_coef = np.zeros((p, k))
    best_score = np.zeros(k)
    best_alpha = np.zeros(k)
    for i in range(na):
        a = float(alphas[i])
        looe, coef = _gcv_solve(mode, a, Yc, sqrt_sw, fit_intercept, Xs, ev, Q, QT_y, QT_sw)
        se = looe ** 2
        if store_cv_results:
            fl = se.ravel()
            for r in range(n * k):
                cvr[r, i] = fl[r]
        if alpha_per_target and k > 1:
            sc = np.mean(-se, axis=0)
            for t in range(k):
                if i == 0 or sc[t] > best_score[t]:
                    best_score[t] = sc[t]
                    best_alpha[t] = a
                    for j in range(p):
                        best_coef[j, t] = coef[j, t]
        else:
            sc0 = float(np.mean(-se))
            if i == 0 or sc0 > best_score[0]:
                best_coef = coef
                for t in range(k):
                    best_score[t] = sc0
                    best_alpha[t] = a
    coefT = np.ascontiguousarray(best_coef.T)
    inter = _set_intercept(coefT, X_offset, y_offset, fit_intercept)
    return coefT, inter, best_alpha, best_score, cvr

class RidgeCV[D: Literal[int] = 1, CV = NoneType](_LinearRegressorMixin):
    alphas: np.ndarray[float, 1]
    fit_intercept: bool
    cv: CV
    gcv_mode: Optional[str]
    store_cv_results: bool
    alpha_per_target: bool
    _coef: np.ndarray[float, 2]
    _intercept: np.ndarray[float, 1]
    _alpha: np.ndarray[float, 1]
    _best_score: np.ndarray[float, 1]
    cv_results_: np.ndarray[float, 2]
    n_features_in_: int

    def __init__(self, alphas = (0.1, 1.0, 10.0), *, fit_intercept: bool = True, scoring = None,
                 cv: CV = None, gcv_mode: Optional[str] = None, store_cv_results: bool = False,
                 alpha_per_target: bool = False):
        if scoring is not None:
            compile_error("RidgeCV: only scoring=None is supported")
        self.alphas = np.atleast_1d(np.asarray(alphas, dtype=float)).ravel()
        self.fit_intercept = fit_intercept
        self.cv = cv
        self.gcv_mode = gcv_mode
        self.store_cv_results = store_cv_results
        self.alpha_per_target = alpha_per_target
        self._coef = np.zeros((0, 0))
        self._intercept = np.zeros(0)
        self._alpha = np.zeros(0)
        self._best_score = np.zeros(0)
        self.cv_results_ = np.zeros((0, 0))
        self.n_features_in_ = 0

    def _nd(self) -> Literal[int]:
        return D

    @property
    def alpha_(self):
        if D == 1:
            return float(self._alpha[0])
        else:
            if self.alpha_per_target:
                return self._alpha.copy()
            else:
                return self._alpha[:1].copy()

    @property
    def best_score_(self):
        if D == 1:
            return float(self._best_score[0])
        else:
            return self._best_score.copy()

    def fit(self, X, y, sample_weight = None):
        ya = np.asarray(y, dtype=float)
        if _ndim_of(ya) != D:
            return _retype(self, RidgeCV[_ndim_of(ya), CV]).fit(X, ya, sample_weight)
        Y = _as2d_y(ya)
        self.n_features_in_ = np.asarray(X).shape[1]
        if self.cv is None:
            for i in range(self.alphas.shape[0]):
                if not self.alphas[i] > 0.0:
                    raise ValueError(f"alphas[{i}] == {self.alphas[i]}, must be > 0.0.")
            coef, inter, al, bs, cvr = _ridge_gcv(X, Y, self.alphas, sample_weight,
                                                  self.fit_intercept, self.gcv_mode,
                                                  self.alpha_per_target, self.store_cv_results)
            self._coef = coef
            self._intercept = inter
            self._alpha = al
            self._best_score = bs
            if self.store_cv_results:
                self.cv_results_ = cvr
        else:
            if self.store_cv_results:
                raise ValueError("cv!=None and store_cv_results=True are incompatible")
            if self.alpha_per_target:
                raise ValueError("cv!=None and alpha_per_target=True are incompatible")
            best, bs = _ridge_cv_search(X, ya, self.alphas, self.cv, self.fit_intercept, False)
            r = Ridge[D](best, fit_intercept=self.fit_intercept)
            r.fit(X, ya, sample_weight)
            self._coef = r._coef
            self._intercept = r._intercept
            self._alpha = np.array([best])
            self._best_score = np.array([bs])
        return self

def _ridge_cv_search(X, y, alphas: np.ndarray[float, 1], cv, fit_intercept: bool, is_clf: bool):
    """GridSearchCV(Ridge/RidgeClassifier, {'alpha': alphas}, cv) with the
    estimator's default score (r2 / accuracy). Returns (best_alpha, best_score)."""
    from ..model_selection import check_cv
    Xa = np.asarray(X, dtype=float)
    ya = np.asarray(y)
    cvo = check_cv(cv, ya, classifier=is_clf)
    folds = [(tr, te) for tr, te in cvo.split(Xa, ya)]
    best = 0
    best_score = -math.inf
    for ai in range(alphas.shape[0]):
        tot = 0.0
        for f in folds:
            tr, te = f
            if is_clf:
                m = RidgeClassifier(float(alphas[ai]), fit_intercept=fit_intercept)
                m.fit(Xa[tr], ya[tr])
                tot += m.score(Xa[te], ya[te])
            else:
                m2 = Ridge[_ndim_of(ya)](float(alphas[ai]), fit_intercept=fit_intercept)
                m2.fit(Xa[tr], ya[tr])
                tot += m2.score(Xa[te], ya[te])
        mean = tot / float(len(folds))
        if mean > best_score:
            best_score = mean
            best = ai
    return float(alphas[best]), best_score

class RidgeClassifierCV[CW, CV = NoneType](_LinearClassifierMixin):
    alphas: np.ndarray[float, 1]
    fit_intercept: bool
    cv: CV
    class_weight: CW
    store_cv_results: bool
    classes_: np.ndarray[int, 1]
    _coef: np.ndarray[float, 2]
    _intercept: np.ndarray[float, 1]
    alpha_: float
    best_score_: float
    cv_results_: np.ndarray[float, 2]
    n_features_in_: int

    def __init__(self, alphas = (0.1, 1.0, 10.0), *, fit_intercept: bool = True, scoring = None,
                 cv: CV = None, class_weight: CW = None, store_cv_results: bool = False):
        if scoring is not None:
            compile_error("RidgeClassifierCV: only scoring=None is supported")
        self.alphas = np.atleast_1d(np.asarray(alphas, dtype=float)).ravel()
        self.fit_intercept = fit_intercept
        self.cv = cv
        self.class_weight = class_weight
        self.store_cv_results = store_cv_results
        self.classes_ = np.zeros(0, dtype=int)
        self._coef = np.zeros((0, 0))
        self._intercept = np.zeros(0)
        self.alpha_ = 0.0
        self.best_score_ = 0.0
        self.cv_results_ = np.zeros((0, 0))
        self.n_features_in_ = 0

    def fit(self, X, y, sample_weight = None):
        classes, yi, Y = _label_binarize_pm1(y)
        self.classes_ = classes
        n = Y.shape[0]
        sw = _check_sw(sample_weight, n)
        cw = _class_weight_vec(self.class_weight, classes, yi)
        for i in range(n):
            sw[i] *= cw[yi[i]]
        self.n_features_in_ = np.asarray(X).shape[1]
        if self.cv is None:
            coef, inter, al, bs, cvr = _ridge_gcv(X, Y, self.alphas, sw, self.fit_intercept,
                                                  None, False, self.store_cv_results)
            self._coef = coef
            self._intercept = inter
            self.alpha_ = float(al[0])
            self.best_score_ = float(bs[0])
            if self.store_cv_results:
                self.cv_results_ = cvr
        else:
            best, bs = _ridge_cv_search(X, y, self.alphas, self.cv, self.fit_intercept, True)
            m = RidgeClassifier(best, fit_intercept=self.fit_intercept, class_weight=self.class_weight)
            m.fit(X, y, sample_weight)
            self._coef = m._coef
            self._intercept = m._intercept
            self.alpha_ = best
            self.best_score_ = bs
        return self
