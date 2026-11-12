# Ported from scikit-learn 1.9.1 (BSD-3-Clause)

import numpy as np
import scipy.linalg as sl

class BayesianRidge:
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
