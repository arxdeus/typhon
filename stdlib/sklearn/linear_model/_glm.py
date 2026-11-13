# Ported from scikit-learn 1.9.1 (BSD-3-Clause)
#
# PoissonRegressor / GammaRegressor / TweedieRegressor (sklearn/linear_model/
# _glm/glm.py; solvers 'lbfgs' and 'newton-cholesky'), HuberRegressor
# (_huber.py) and QuantileRegressor (_quantile.py, scipy linprog 'highs*').

from ..base import estimator_repr as _estimator_repr
import math
import numpy as np
from ._base import _check_sw, _r2
from ._linear_loss import LinearModelLoss, lbfgs_fit, TWEEDIE_LOG, TWEEDIE_ID, _closs
from ._newton import NewtonCholeskySolver

def _xlogy(x: float, y: float) -> float:
    if x == 0.0 and not math.isnan(y):
        return 0.0
    return x * math.log(y)

class _GeneralizedLinearRegressor:
    def __repr__(self) -> str:
        return _estimator_repr(self)

    alpha: float
    fit_intercept: bool
    solver: str
    max_iter: int
    tol: float
    warm_start: bool
    verbose: int
    power: float
    link: str
    coef_: np.ndarray[float, 1]
    intercept_: float
    n_iter_: int
    n_features_in_: int
    _fitted: bool

    def _kind(self) -> int:
        # HalfTweedieLossIdentity for link='identity' or (auto and power <= 0)
        if self.link == "identity" or (self.link == "auto" and self.power <= 0.0):
            return TWEEDIE_ID
        return TWEEDIE_LOG

    def _in_range(self, y: np.ndarray[float, 1]) -> bool:
        p = self.power
        for v in y:
            if p <= 0.0:
                if not math.isfinite(v):
                    return False
            elif p < 2.0:
                if not (v >= 0.0):
                    return False
            else:
                if not (v > 0.0):
                    return False
        return True

    def _link(self, mu: float) -> float:
        return mu if self._kind() == TWEEDIE_ID else math.log(mu)

    def _inverse(self, r: float) -> float:
        return r if self._kind() == TWEEDIE_ID else math.exp(r)

    def _loss_name(self) -> str:
        if self._kind() == TWEEDIE_ID:
            return "HalfTweedieLossIdentity"
        if self.power == 1.0:
            return "HalfPoissonLoss"
        if self.power == 2.0:
            return "HalfGammaLoss"
        return "HalfTweedieLoss"

    def fit(self, X, y, sample_weight = None):
        Xa = np.ascontiguousarray(np.asarray(X, dtype=float))
        ya = np.ascontiguousarray(np.asarray(y, dtype=float))
        n, p = Xa.shape
        sw = _check_sw(sample_weight, n)
        if self.solver != "lbfgs" and self.solver != "newton-cholesky":
            raise ValueError(f"Invalid solver={self.solver}.")
        if not self._in_range(ya):
            raise ValueError(f"Some value(s) of y are out of the valid range of the loss {self._loss_name()!r}.")
        lm = LinearModelLoss(Xa, ya, sw, self._kind(), self.power, 1, self.fit_intercept)
        nd = p + (1 if self.fit_intercept else 0)
        coef = np.zeros(nd)
        if self.warm_start and self._fitted:
            for j in range(p):
                coef[j] = self.coef_[j]
            if self.fit_intercept:
                coef[p] = self.intercept_
        elif self.fit_intercept:
            coef[p] = self._link(float(np.average(ya, weights=sw)))
        l2 = self.alpha
        if self.solver == "lbfgs":
            x, nit, st = lbfgs_fit(lm, coef, l2, self.max_iter, self.tol)
            coef = x
            self.n_iter_ = nit
        else:
            sol = NewtonCholeskySolver(lm, coef, l2, self.tol, self.max_iter)
            coef = sol.solve()
            self.n_iter_ = sol.iteration
        self.coef_ = coef[:p].copy()
        self.intercept_ = coef[p] if self.fit_intercept else 0.0
        self.n_features_in_ = p
        self._fitted = True
        return self

    def _linear_predictor(self, X) -> np.ndarray[float, 1]:
        Xa = np.asarray(X, dtype=float)
        return Xa @ self.coef_ + self.intercept_

    def predict(self, X):
        r = self._linear_predictor(X)
        out = np.empty(r.shape[0])
        for i in range(r.shape[0]):
            out[i] = self._inverse(r[i])
        return out

    def _constant(self, y: float) -> float:
        # constant_to_optimal_zero
        p = self.power
        if self._kind() == TWEEDIE_ID:
            if p == 0.0:
                return 0.0
            elif p == 1.0:
                return 0.0
            elif p == 2.0:
                return 0.0
            return 0.0
        if p == 0.0:
            return 0.0
        elif p == 1.0:
            return _xlogy(y, y) - y
        elif p == 2.0:
            return -math.log(y) - 1.0
        return math.pow(max(y, 0.0), 2.0 - p) / (1.0 - p) / (2.0 - p)

    def score(self, X, y, sample_weight = None) -> float:
        r = self._linear_predictor(X)
        ya = np.asarray(y, dtype=float)
        n = ya.shape[0]
        sw = _check_sw(sample_weight, n)
        if not self._in_range(ya):
            raise ValueError(f"Some value(s) of y are out of the valid range of the loss {self._loss_name()}.")
        kind = self._kind()
        swsum = float(np.sum(sw))
        const = 0.0
        dev = 0.0
        ymean = 0.0
        for i in range(n):
            ymean += sw[i] * ya[i]
        ymean /= swsum
        ym = self._link(ymean)
        devn = 0.0
        for i in range(n):
            const += sw[i] * self._constant(ya[i])
            dev += sw[i] * _closs(kind, self.power, ya[i], r[i])
            devn += sw[i] * _closs(kind, self.power, ya[i], ym)
        const /= swsum
        dev /= swsum
        devn /= swsum
        return float(1.0 - (dev + const) / (devn + const))

def _glm_init(m, alpha: float, fit_intercept: bool, solver: str, max_iter: int, tol: float,
              warm_start: bool, verbose: int, power: float, link: str):
    m.alpha = alpha
    m.fit_intercept = fit_intercept
    m.solver = solver
    m.max_iter = max_iter
    m.tol = tol
    m.warm_start = warm_start
    m.verbose = verbose
    m.power = power
    m.link = link
    m.coef_ = np.zeros(0)
    m.intercept_ = 0.0
    m.n_iter_ = 0
    m.n_features_in_ = 0
    m._fitted = False

class PoissonRegressor(_GeneralizedLinearRegressor):
    def __init__(self, *, alpha: float = 1.0, fit_intercept: bool = True, solver: str = "lbfgs",
                 max_iter: int = 100, tol: float = 1e-4, warm_start: bool = False, verbose: int = 0):
        _glm_init(self, alpha, fit_intercept, solver, max_iter, tol, warm_start, verbose, 1.0, "log")

class GammaRegressor(_GeneralizedLinearRegressor):
    def __init__(self, *, alpha: float = 1.0, fit_intercept: bool = True, solver: str = "lbfgs",
                 max_iter: int = 100, tol: float = 1e-4, warm_start: bool = False, verbose: int = 0):
        _glm_init(self, alpha, fit_intercept, solver, max_iter, tol, warm_start, verbose, 2.0, "log")

class TweedieRegressor(_GeneralizedLinearRegressor):
    def __init__(self, *, power: float = 0.0, alpha: float = 1.0, fit_intercept: bool = True,
                 link: str = "auto", solver: str = "lbfgs", max_iter: int = 100, tol: float = 1e-4,
                 warm_start: bool = False, verbose: int = 0):
        if link != "auto" and link != "log" and link != "identity":
            raise ValueError(f"The 'link' parameter of TweedieRegressor must be a str among {{'auto', 'identity', 'log'}}. Got {link!r} instead.")
        if 0.0 < power and power < 1.0:
            raise ValueError("Tweedie power must not be in (0, 1).")
        _glm_init(self, alpha, fit_intercept, solver, max_iter, tol, warm_start, verbose, power, link)

class _HuberFG:
    X: np.ndarray[float, 2]
    y: np.ndarray[float, 1]
    eps: float
    alpha: float
    sw: np.ndarray[float, 1]
    _x: np.ndarray[float, 1]
    _g: np.ndarray[float, 1]
    _has: bool

    def __init__(self, X: np.ndarray[float, 2], y: np.ndarray[float, 1], eps: float, alpha: float,
                 sw: np.ndarray[float, 1]):
        self.X = X
        self.y = y
        self.eps = eps
        self.alpha = alpha
        self.sw = sw
        self._x = np.zeros(0)
        self._g = np.zeros(0)
        self._has = False

    def loss_grad(self, wv: np.ndarray[float, 1]) -> Tuple[float, np.ndarray[float, 1]]:
        X = self.X
        n, p = X.shape
        fit_intercept = p + 2 == wv.shape[0]
        intercept = wv[p] if fit_intercept else 0.0
        sigma = wv[wv.shape[0] - 1]
        w = wv[:p]
        epsilon = self.eps
        n_samples = float(np.sum(self.sw))
        lin = self.y - X @ w
        if fit_intercept:
            lin -= intercept
        absl = np.abs(lin)
        mask = absl > epsilon * sigma
        out_idx = List[int]()
        in_idx = List[int]()
        for i in range(n):
            if mask[i]:
                out_idx.append(i)
            else:
                in_idx.append(i)
        oi = np.array(out_idx, dtype=int)
        ii = np.array(in_idx, dtype=int)
        outliers = absl[oi]
        outliers_sw = self.sw[oi]
        n_sw_outliers = float(np.sum(outliers_sw))
        outlier_loss = 2.0 * epsilon * float(np.sum(outliers_sw * outliers)) - sigma * n_sw_outliers * epsilon * epsilon
        non_out = lin[ii]
        wno = self.sw[ii] * non_out
        weighted_loss = float(np.dot(wno, non_out))
        squared_loss = weighted_loss / sigma
        grad = np.zeros(wv.shape[0])
        Xno = -X[ii]
        g1 = 2.0 / sigma * (wno @ Xno)
        signed = np.ones(oi.shape[0])
        lo = lin[oi]
        for k in range(oi.shape[0]):
            if lo[k] < 0.0:
                signed[k] = -1.0
        Xo = X[oi]
        sw_out = self.sw[oi] * signed
        g1 -= 2.0 * epsilon * (sw_out @ Xo)
        g1 += self.alpha * 2.0 * w
        for j in range(p):
            grad[j] = g1[j]
        last = wv.shape[0] - 1
        grad[last] = n_samples
        grad[last] -= n_sw_outliers * epsilon * epsilon
        grad[last] -= squared_loss / sigma
        if fit_intercept:
            grad[p] = -2.0 * float(np.sum(wno)) / sigma
            grad[p] -= 2.0 * epsilon * float(np.sum(sw_out))
        loss = n_samples * sigma + squared_loss + outlier_loss
        loss += self.alpha * float(np.dot(w, w))
        return loss, grad

    def _same(self, x: np.ndarray[float, 1]) -> bool:
        if not self._has or self._x.shape[0] != x.shape[0]:
            return False
        for i in range(x.shape[0]):
            if self._x[i] != x[i]:
                return False
        return True

    def fun(self, x: np.ndarray[float, 1]) -> float:
        f, g = self.loss_grad(x)
        self._x = x.copy()
        self._g = g
        self._has = True
        return f

    def grad(self, x: np.ndarray[float, 1]) -> np.ndarray[float, 1]:
        if not self._same(x):
            self.fun(x)
        return self._g.copy()

class HuberRegressor:
    def __repr__(self) -> str:
        return _estimator_repr(self)

    epsilon: float
    max_iter: int
    alpha: float
    warm_start: bool
    fit_intercept: bool
    tol: float
    coef_: np.ndarray[float, 1]
    intercept_: float
    scale_: float
    n_iter_: int
    outliers_: np.ndarray[bool, 1]
    n_features_in_: int
    _fitted: bool

    def __init__(self, *, epsilon: float = 1.35, max_iter: int = 100, alpha: float = 0.0001,
                 warm_start: bool = False, fit_intercept: bool = True, tol: float = 1e-05):
        if epsilon < 1.0:
            raise ValueError(f"The 'epsilon' parameter of HuberRegressor must be a float in the range [1.0, inf). Got {epsilon} instead.")
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.alpha = alpha
        self.warm_start = warm_start
        self.fit_intercept = fit_intercept
        self.tol = tol
        self.coef_ = np.zeros(0)
        self.intercept_ = 0.0
        self.scale_ = 0.0
        self.n_iter_ = 0
        self.outliers_ = np.zeros(0, dtype=bool)
        self.n_features_in_ = 0
        self._fitted = False

    def fit(self, X, y, sample_weight = None):
        from scipy.optimize import minimize
        Xa = np.ascontiguousarray(np.asarray(X, dtype=float))
        ya = np.asarray(y, dtype=float)
        n, p = Xa.shape
        sw = _check_sw(sample_weight, n)
        if self.warm_start and self._fitted:
            params = np.empty(p + 2)
            for j in range(p):
                params[j] = self.coef_[j]
            params[p] = self.intercept_
            params[p + 1] = self.scale_
        else:
            params = np.zeros(p + 2 if self.fit_intercept else p + 1)
            params[params.shape[0] - 1] = 1.0
        m = params.shape[0]
        lb = np.full(m, -math.inf)
        ub = np.full(m, math.inf)
        lb[m - 1] = 2.220446049250313e-16 * 10.0
        bounds = [(lb[i], ub[i]) for i in range(m)]
        fg = _HuberFG(Xa, ya, self.epsilon, self.alpha, sw)
        opts = {"maxiter": float(self.max_iter), "gtol": self.tol}
        res = minimize(fg.fun, params, method="L-BFGS-B", jac=fg.grad, bounds=bounds, options=opts)
        params = res.x
        if res.status == 2:
            raise ValueError(f"HuberRegressor convergence failed: l-BFGS-b solver terminated with {res.message}")
        self.n_iter_ = min(res.nit, self.max_iter)
        self.scale_ = params[m - 1]
        self.intercept_ = params[p] if self.fit_intercept else 0.0
        self.coef_ = params[:p].copy()
        resid = np.abs(ya - Xa @ self.coef_ - self.intercept_)
        self.outliers_ = resid > self.scale_ * self.epsilon
        self.n_features_in_ = p
        self._fitted = True
        return self

    def predict(self, X):
        return np.asarray(X, dtype=float) @ self.coef_ + self.intercept_

    def score(self, X, y, sample_weight = None) -> float:
        sw = _check_sw(sample_weight, np.asarray(y).shape[0])
        return _r2(np.asarray(y, dtype=float), self.predict(X), sw, sample_weight is not None)

class QuantileRegressor:
    def __repr__(self) -> str:
        return _estimator_repr(self)

    quantile: float
    alpha: float
    fit_intercept: bool
    solver: str
    coef_: np.ndarray[float, 1]
    intercept_: float
    n_iter_: int
    n_features_in_: int

    def __init__(self, *, quantile: float = 0.5, alpha: float = 1.0, fit_intercept: bool = True,
                 solver: str = "highs", solver_options = None):
        if not (0.0 < quantile and quantile < 1.0):
            raise ValueError(f"The 'quantile' parameter of QuantileRegressor must be a float in the range (0.0, 1.0). Got {quantile} instead.")
        if solver == "interior-point":
            raise ValueError(f"Solver {solver} is not anymore available in SciPy >= 1.11.0.")
        self.quantile = quantile
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.solver = solver
        self.coef_ = np.zeros(0)
        self.intercept_ = 0.0
        self.n_iter_ = 0
        self.n_features_in_ = 0

    def fit(self, X, y, sample_weight = None):
        from scipy.optimize import linprog
        Xa = np.asarray(X, dtype=float)
        ya = np.asarray(y, dtype=float)
        n, p = Xa.shape
        sw = _check_sw(sample_weight, n)
        npar = p + (1 if self.fit_intercept else 0)
        alpha = float(np.sum(sw)) * self.alpha
        idx = List[int]()
        for i in range(n):
            if sw[i] != 0.0:
                idx.append(i)
        ia = np.array(idx, dtype=int)
        ni = ia.shape[0]
        if ni < n:
            sw = sw[ia]
            Xa = Xa[ia]
            ya = ya[ia]
        nv = 2 * npar + 2 * ni
        c = np.empty(nv)
        for k in range(2 * npar):
            c[k] = alpha
        for i in range(ni):
            c[2 * npar + i] = sw[i] * self.quantile
            c[2 * npar + ni + i] = sw[i] * (1.0 - self.quantile)
        if self.fit_intercept:
            c[0] = 0.0
            c[npar] = 0.0
        A = np.zeros((ni, nv))
        off = 1 if self.fit_intercept else 0
        for i in range(ni):
            if self.fit_intercept:
                A[i, 0] = 1.0
                A[i, npar] = -1.0
            for j in range(p):
                A[i, off + j] = Xa[i, j]
                A[i, npar + off + j] = -Xa[i, j]
            A[i, 2 * npar + i] = 1.0
            A[i, 2 * npar + ni + i] = -1.0
        res = linprog(c, A_eq=A, b_eq=ya, method=self.solver)
        sol = res.x
        params = np.empty(npar)
        for k in range(npar):
            params[k] = sol[k] - sol[npar + k]
        self.n_iter_ = res.nit
        if self.fit_intercept:
            self.coef_ = params[1:].copy()
            self.intercept_ = params[0]
        else:
            self.coef_ = params
            self.intercept_ = 0.0
        self.n_features_in_ = p
        return self

    def predict(self, X):
        return np.asarray(X, dtype=float) @ self.coef_ + self.intercept_

    def score(self, X, y, sample_weight = None) -> float:
        sw = _check_sw(sample_weight, np.asarray(y).shape[0])
        return _r2(np.asarray(y, dtype=float), self.predict(X), sw, sample_weight is not None)
