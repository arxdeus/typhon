# Ported from scikit-learn 1.9.1 (BSD-3-Clause)

from ..base import estimator_repr as _estimator_repr
import math
import numpy as np
import scipy.linalg as sl

class BayesianRidge:
    def __repr__(self) -> str:
        return _estimator_repr(self)

    """Bayesian ridge regression (evidence maximisation, SVD based)."""
    max_iter: int
    tol: float
    alpha_1: float
    alpha_2: float
    lambda_1: float
    lambda_2: float
    alpha_init: Optional[float]
    lambda_init: Optional[float]
    compute_score: bool
    fit_intercept: bool
    coef_: np.ndarray[float, 1]
    intercept_: float
    alpha_: float
    lambda_: float
    sigma_: np.ndarray[float, 2]
    scores_: List[float]
    n_iter_: int
    X_offset_: np.ndarray[float, 1]
    X_scale_: np.ndarray[float, 1]

    def __init__(self, *, max_iter: int = 300, tol: float = 1e-3, alpha_1: float = 1e-6,
                 alpha_2: float = 1e-6, lambda_1: float = 1e-6, lambda_2: float = 1e-6,
                 alpha_init: Optional[float] = None, lambda_init: Optional[float] = None,
                 compute_score: bool = False, fit_intercept: bool = True, copy_X: bool = True,
                 verbose: bool = False):
        self.max_iter = max_iter
        self.tol = tol
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.alpha_init = alpha_init
        self.lambda_init = lambda_init
        self.compute_score = compute_score
        self.fit_intercept = fit_intercept
        self.coef_ = np.empty(0)
        self.intercept_ = 0.0
        self.alpha_ = 0.0
        self.lambda_ = 0.0
        self.sigma_ = np.empty((0, 0))
        self.scores_ = []
        self.n_iter_ = 0
        self.X_offset_ = np.empty(0)
        self.X_scale_ = np.empty(0)

    def _update_coef(self, X, y, n: int, p: int, XT_y, U, Vh, ev, alpha_: float, lambda_: float):
        coef = np.empty(p)
        if n > p:
            coef = Vh.T @ ((Vh / (ev + lambda_ / alpha_)[:, None]) @ XT_y)
        else:
            coef = X.T @ ((U / (ev + lambda_ / alpha_)[None, :]) @ (U.T @ y))
        r = y - X @ coef
        return coef, float(np.sum(r * r))

    def _lml(self, n: int, p: int, ev, alpha_: float, lambda_: float, coef, sse: float) -> float:
        logdet = 0.0
        if n > p:
            logdet = -float(np.sum(np.log(lambda_ + alpha_ * ev)))
        else:
            ld = np.full(p, lambda_)
            ld[:n] += alpha_ * ev
            logdet = -float(np.sum(np.log(ld)))
        s = self.lambda_1 * np.log(lambda_) - self.lambda_2 * lambda_
        s += self.alpha_1 * np.log(alpha_) - self.alpha_2 * alpha_
        s += 0.5 * (p * np.log(lambda_) + n * np.log(alpha_) - alpha_ * sse
                    - lambda_ * float(np.sum(coef ** 2)) + logdet - n * np.log(2 * np.pi))
        return s

    def fit(self, X, y, sample_weight = None):
        Xa = np.array(X, dtype=float)
        ya = np.array(y, dtype=float)
        n, p = Xa.shape
        y_var = float(ya.var())
        y_off = 0.0
        if self.fit_intercept:
            self.X_offset_ = Xa.mean(axis=0)
            y_off = float(ya.mean())
            Xa -= self.X_offset_
            ya -= y_off
        else:
            self.X_offset_ = np.zeros(p)
        self.X_scale_ = np.ones(p)
        eps = 2.220446049250313e-16
        alpha_ = 1.0 / (y_var + eps) if self.alpha_init is None else self.alpha_init.__val__()
        lambda_ = 1.0 if self.lambda_init is None else self.lambda_init.__val__()
        self.scores_ = []
        XT_y = Xa.T @ ya
        U, S, Vh_full = sl.svd(Xa, full_matrices=n < p)
        K = S.shape[0]
        ev = S ** 2
        ev_full = np.zeros(p)
        ev_full[:K] = ev
        Vh = np.ascontiguousarray(Vh_full[:K, :])
        if n < p:
            U = np.ascontiguousarray(U[:, :K])
        coef_old = np.zeros(p)
        coef = np.zeros(p)
        sse = 0.0
        it = 0
        for i in range(self.max_iter):
            it = i
            coef, sse = self._update_coef(Xa, ya, n, p, XT_y, U, Vh, ev, alpha_, lambda_)
            if self.compute_score:
                self.scores_.append(self._lml(n, p, ev, alpha_, lambda_, coef, sse))
            gamma_ = float(np.sum(alpha_ * ev / (lambda_ + alpha_ * ev)))
            lambda_ = (gamma_ + 2 * self.lambda_1) / (float(np.sum(coef ** 2)) + 2 * self.lambda_2)
            alpha_ = (n - gamma_ + 2 * self.alpha_1) / (sse + 2 * self.alpha_2)
            if i != 0 and float(np.sum(np.abs(coef_old - coef))) < self.tol:
                break
            coef_old = coef.copy()
        self.n_iter_ = it + 1
        self.alpha_ = alpha_
        self.lambda_ = lambda_
        self.coef_, sse = self._update_coef(Xa, ya, n, p, XT_y, U, Vh, ev, alpha_, lambda_)
        if self.compute_score:
            self.scores_.append(self._lml(n, p, ev, alpha_, lambda_, coef, sse))
        self.sigma_ = Vh_full.T @ (Vh_full / (alpha_ * ev_full + lambda_)[:, None])
        if self.fit_intercept:
            self.intercept_ = y_off - float(self.X_offset_ @ self.coef_)
        else:
            self.intercept_ = 0.0
        return self

    def predict(self, X, return_std: Literal[bool] = False):
        Xa = np.asarray(X, dtype=float)
        mean = Xa @ self.coef_ + self.intercept_
        if return_std:
            Xc = Xa - self.X_offset_
            s2 = ((Xc @ self.sigma_) * Xc).sum(axis=1)
            return mean, np.sqrt(s2 + 1.0 / self.alpha_)
        else:
            return mean

    def score(self, X, y, sample_weight = None) -> float:
        ya = np.asarray(y, dtype=float)
        r = ya - self.predict(X)
        t = ya - ya.mean()
        return 1.0 - float(np.sum(r * r)) / float(np.sum(t * t))

class ARDRegression:
    def __repr__(self) -> str:
        return _estimator_repr(self)

    """ARDRegression (sklearn/linear_model/_bayes.py), dense float64."""
    max_iter: int
    tol: float
    fit_intercept: bool
    alpha_1: float
    alpha_2: float
    lambda_1: float
    lambda_2: float
    compute_score: bool
    threshold_lambda: float
    coef_: np.ndarray[float, 1]
    intercept_: float
    alpha_: float
    lambda_: np.ndarray[float, 1]
    sigma_: np.ndarray[float, 2]
    scores_: List[float]
    n_iter_: int
    X_offset_: np.ndarray[float, 1]
    X_scale_: np.ndarray[float, 1]
    n_features_in_: int

    def __init__(self, *, max_iter: int = 300, tol: float = 1e-3, alpha_1: float = 1e-6,
                 alpha_2: float = 1e-6, lambda_1: float = 1e-6, lambda_2: float = 1e-6,
                 compute_score: bool = False, threshold_lambda: float = 1.0e4,
                 fit_intercept: bool = True, copy_X: bool = True, verbose: bool = False):
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.compute_score = compute_score
        self.threshold_lambda = threshold_lambda
        self.coef_ = np.empty(0)
        self.intercept_ = 0.0
        self.alpha_ = 0.0
        self.lambda_ = np.empty(0)
        self.sigma_ = np.empty((0, 0))
        self.scores_ = []
        self.n_iter_ = 0
        self.X_offset_ = np.empty(0)
        self.X_scale_ = np.empty(0)
        self.n_features_in_ = 0

    def _keep_idx(self, keep: np.ndarray[bool, 1]) -> np.ndarray[int, 1]:
        r = List[int]()
        for j in range(keep.shape[0]):
            if keep[j]:
                r.append(j)
        return np.array(r, dtype=int)

    def _update_sigma(self, X: np.ndarray[float, 2], alpha_: float, lambda_: np.ndarray[float, 1],
                      ki: np.ndarray[int, 1], woodbury: bool) -> np.ndarray[float, 2]:
        Xk = np.ascontiguousarray(X[:, ki])
        lk = lambda_[ki]
        if not woodbury:
            gram = Xk.T @ Xk
            m = gram.shape[0]
            sinv = alpha_ * gram
            for a in range(m):
                for b in range(m):
                    sinv[a, b] = (lk[b] if a == b else 0.0) + alpha_ * gram[a, b]
            return sl.pinvh(sinv)
        n = X.shape[0]
        inv_l = 1.0 / lk
        XL = Xk * inv_l[None, :]
        M = np.eye(n) / alpha_ + XL @ Xk.T
        S = sl.pinvh(M)
        S = S @ XL
        S = -((inv_l[:, None] * Xk.T) @ S)
        for a in range(S.shape[1]):
            S[a, a] += 1.0 / lk[a]
        return S

    def fit(self, X, y):
        from ..decomposition._extmath import fast_logdet
        Xa = np.array(X, dtype=float)
        ya = np.array(y, dtype=float)
        n, p = Xa.shape
        if n < 2:
            raise ValueError(f"Found array with {n} sample(s) while a minimum of 2 is required by ARDRegression.")
        coef_ = np.zeros(p)
        y_off = 0.0
        if self.fit_intercept:
            self.X_offset_ = Xa.mean(axis=0)
            y_off = float(ya.mean())
            Xa -= self.X_offset_
            ya -= y_off
        else:
            self.X_offset_ = np.zeros(p)
        self.X_scale_ = np.ones(p)
        keep = np.ones(p, dtype=bool)
        eps = 2.220446049250313e-16
        alpha_ = 1.0 / (float(np.var(ya)) + eps)
        lambda_ = np.ones(p)
        self.scores_ = []
        coef_old = np.zeros(p)
        woodbury = n < p
        it = 0
        sigma_ = np.zeros((0, 0))
        for iter_ in range(self.max_iter):
            it = iter_
            ki = self._keep_idx(keep)
            sigma_ = self._update_sigma(Xa, alpha_, lambda_, ki, woodbury)
            ck = alpha_ * (sigma_ @ (np.ascontiguousarray(Xa[:, ki]).T @ ya))
            for a in range(ki.shape[0]):
                coef_[ki[a]] = ck[a]
            r = ya - Xa @ coef_
            sse_ = float(np.sum(r ** 2))
            gamma_ = np.empty(ki.shape[0])
            for a in range(ki.shape[0]):
                gamma_[a] = 1.0 - lambda_[ki[a]] * sigma_[a, a]
            for a in range(ki.shape[0]):
                j = ki[a]
                lambda_[j] = (gamma_[a] + 2.0 * self.lambda_1) / (coef_[j] ** 2 + 2.0 * self.lambda_2)
            alpha_ = (float(n) - float(gamma_.sum()) + 2.0 * self.alpha_1) / (sse_ + 2.0 * self.alpha_2)
            anyk = False
            for j in range(p):
                keep[j] = lambda_[j] < self.threshold_lambda
                if not keep[j]:
                    coef_[j] = 0.0
                else:
                    anyk = True
            if self.compute_score:
                s = float((self.lambda_1 * np.log(lambda_) - self.lambda_2 * lambda_).sum())
                s += self.alpha_1 * math.log(alpha_) - self.alpha_2 * alpha_
                s += 0.5 * (fast_logdet(sigma_) + n * math.log(alpha_) + float(np.sum(np.log(lambda_))))
                s -= 0.5 * (alpha_ * sse_ + float((lambda_ * coef_ ** 2).sum()))
                self.scores_.append(s)
            if iter_ > 0 and float(np.sum(np.abs(coef_old - coef_))) < self.tol:
                break
            coef_old = coef_.copy()
            if not anyk:
                break
        self.n_iter_ = it + 1
        anyk = False
        for j in range(p):
            if keep[j]:
                anyk = True
        if anyk:
            ki = self._keep_idx(keep)
            sigma_ = self._update_sigma(Xa, alpha_, lambda_, ki, woodbury)
            ck = alpha_ * (sigma_ @ (np.ascontiguousarray(Xa[:, ki]).T @ ya))
            for a in range(ki.shape[0]):
                coef_[ki[a]] = ck[a]
        else:
            sigma_ = np.zeros((0, 0))
        self.coef_ = coef_
        self.alpha_ = alpha_
        self.sigma_ = sigma_
        self.lambda_ = lambda_
        if self.fit_intercept:
            self.intercept_ = y_off - float(self.X_offset_ @ self.coef_)
        else:
            self.intercept_ = 0.0
        self.n_features_in_ = p
        return self

    def predict(self, X, return_std: Literal[bool] = False):
        Xa = np.asarray(X, dtype=float)
        mean = Xa @ self.coef_ + self.intercept_
        if return_std:
            ki = self._keep_idx(self.lambda_ < self.threshold_lambda)
            Xc = np.ascontiguousarray((Xa - self.X_offset_)[:, ki])
            s2 = ((Xc @ self.sigma_) * Xc).sum(axis=1)
            return mean, np.sqrt(s2 + 1.0 / self.alpha_)
        else:
            return mean

    def score(self, X, y, sample_weight = None) -> float:
        ya = np.asarray(y, dtype=float)
        r = ya - self.predict(X)
        t = ya - ya.mean()
        return 1.0 - float(np.sum(r * r)) / float(np.sum(t * t))
