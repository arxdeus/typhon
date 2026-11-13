# Ported from scikit-learn 1.9.1 (BSD-3-Clause)

# GaussianProcessRegressor (Algorithm 2.1 of Rasmussen & Williams): L-BFGS-B
# on the negative log marginal likelihood with analytic gradients, restarts
# drawn from the legacy RandomState. Kernel Km is a static type (default
# C(1.0) * RBF(1.0)); 1-D or 2-D targets.

import math
import numpy as np
import scipy.linalg as sl
from numpy.random.legacy import check_random_state
from ..base import clone
from .kernels import ConstantKernel, RBF, Product

class _LmlFG[G]:
    gp: G
    _x: np.ndarray[float, 1]
    _g: np.ndarray[float, 1]
    _f: float
    _has: bool

    def __init__(self, gp: G):
        self.gp = gp
        self._x = np.empty(0)
        self._g = np.empty(0)
        self._f = 0.0
        self._has = False

    def _eval(self, x):
        xa = np.asarray(x, dtype=float).copy()
        if self._has and self._x.shape[0] == xa.shape[0]:
            same = True
            for i in range(xa.shape[0]):
                if self._x[i] != xa[i]:
                    same = False
            if same:
                return
        lml, g = self.gp._lml_grad(xa)
        self._x = xa
        self._f = -lml
        self._g = -g
        self._has = True

    def fun(self, x) -> float:
        self._eval(x)
        return self._f

    def grad(self, x):
        self._eval(x)
        return self._g.copy()

def _optimize(obj, theta0, bounds):
    from scipy.optimize import minimize
    bl = [(float(bounds[i, 0]), float(bounds[i, 1])) for i in range(bounds.shape[0])]
    res = minimize(obj.fun, theta0, method="L-BFGS-B", jac=obj.grad, bounds=bl)
    return res.x.copy(), float(res.fun)

class GaussianProcessRegressor[K, D: Literal[int] = 1]:
    kernel: K
    alpha: float
    optimizer: str
    n_restarts_optimizer: int
    normalize_y: bool
    random_state: Optional[int]
    kernel_: K
    X_train_: np.ndarray[float, 2]
    y_train_: np.ndarray[float, 2]
    _y1d: bool
    _y_train_mean: np.ndarray[float, 1]
    _y_train_std: np.ndarray[float, 1]
    L_: np.ndarray[float, 2]
    alpha_: np.ndarray[float, 2]
    log_marginal_likelihood_value_: float

    def __init__(self, kernel: K = ConstantKernel(1.0) * RBF(1.0), *, alpha: float = 1e-10,
                 optimizer = "fmin_l_bfgs_b", n_restarts_optimizer: int = 0,
                 normalize_y: bool = False, copy_X_train: bool = True, n_targets = None,
                 random_state: Optional[int] = None):
        self.kernel = kernel
        self.alpha = alpha
        self.optimizer = "" if isinstance(optimizer, NoneType) else str(optimizer)
        self.n_restarts_optimizer = n_restarts_optimizer
        self.normalize_y = normalize_y
        self.random_state = random_state
        self.kernel_ = kernel
        self.X_train_ = np.empty((0, 0))
        self.y_train_ = np.empty((0, 0))
        self._y1d = True
        self._y_train_mean = np.empty(0)
        self._y_train_std = np.empty(0)
        self.L_ = np.empty((0, 0))
        self.alpha_ = np.empty((0, 0))
        self.log_marginal_likelihood_value_ = 0.0

    def _lml_grad(self, theta):
        k = self.kernel_
        k.theta = theta
        Km, KG = k.grad(self.X_train_)
        n = Km.shape[0]
        for i in range(n):
            Km[i, i] += self.alpha
        try:
            L = sl.cholesky(Km, lower=True, check_finite=False)
        except np.linalg.LinAlgError:
            return -np.inf, np.zeros(theta.shape[0])
        Y = self.y_train_
        A = sl.cho_solve((L, True), Y, check_finite=False)
        T = Y.shape[1]
        lml = 0.0
        ld = float(np.log(np.diag(L)).sum())
        for t in range(T):
            v = -0.5 * float(np.dot(Y[:, t], A[:, t]))
            v -= ld
            v -= n / 2 * np.log(2 * np.pi)
            lml += v
        m = KG.shape[2]
        grad = np.zeros(m)
        Kinv = sl.cho_solve((L, True), np.eye(n), check_finite=False)
        for kk in range(m):
            for t in range(T):
                s = 0.0
                for i in range(n):
                    for j in range(n):
                        s += (A[i, t] * A[j, t] - Kinv[i, j]) * KG[j, i, kk]
                grad[kk] += 0.5 * s
        return lml, grad

    def log_marginal_likelihood(self, theta = None, eval_gradient: Literal[bool] = False):
        if isinstance(theta, NoneType):
            return self.log_marginal_likelihood_value_
        else:
            saved = self.kernel_.theta
            r = self._lml_grad(np.asarray(theta, dtype=float))
            self.kernel_.theta = saved
            if eval_gradient:
                return r
            else:
                return r[0]

    def fit(self, X, y):
        if static.len(np.asarray(y).shape) != D:
            from ..linear_model._base import _retype
            return _retype(self, GaussianProcessRegressor[K, static.len(np.asarray(y).shape)]).fit(X, y)
        self.kernel_ = clone(self.kernel)
        rng = check_random_state(self.random_state)
        Xa = np.ascontiguousarray(np.asarray(X, dtype=float))
        ya = np.asarray(y, dtype=float)
        self._y1d = ya.ndim == 1
        Y = np.array(ya.reshape(ya.shape[0], -1))
        T = Y.shape[1]
        if self.normalize_y:
            self._y_train_mean = np.mean(Y, axis=0)
            sd = np.std(Y, axis=0)
            for i in range(T):
                if sd[i] == 0.0:
                    sd[i] = 1.0
            self._y_train_std = sd
            Y = (Y - self._y_train_mean) / self._y_train_std
        else:
            self._y_train_mean = np.zeros(T)
            self._y_train_std = np.ones(T)
        self.X_train_ = Xa
        self.y_train_ = np.ascontiguousarray(Y)
        if self.optimizer != "" and self.kernel_.n_dims > 0:
            obj = _LmlFG(self)
            th0 = self.kernel_.theta.copy()
            bounds = self.kernel_.bounds
            ths = List[np.ndarray[float, 1]]()
            fs = List[float]()
            t, f = _optimize(obj, th0, bounds)
            ths.append(t); fs.append(f)
            for _ in range(self.n_restarts_optimizer):
                if not np.isfinite(bounds).all():
                    raise ValueError("Multiple optimizer restarts (n_restarts_optimizer>0) requires that all bounds are finite.")
                ti = rng.uniform(bounds[:, 0], bounds[:, 1])
                t, f = _optimize(obj, ti, bounds)
                ths.append(t); fs.append(f)
            best = int(np.argmin(np.array(fs)))
            self.kernel_.theta = ths[best]
            self.log_marginal_likelihood_value_ = -fs[best]
        else:
            self.log_marginal_likelihood_value_ = self._lml_grad(self.kernel_.theta)[0]
        Km = self.kernel_(Xa)
        for i in range(Km.shape[0]):
            Km[i, i] += self.alpha
        self.L_ = sl.cholesky(Km, lower=True, check_finite=False)
        self.alpha_ = sl.cho_solve((self.L_, True), self.y_train_, check_finite=False)
        return self

    def _mean(self, X):
        Kt = self.kernel_(X, self.X_train_)
        return Kt, (Kt @ self.alpha_) * self._y_train_std + self._y_train_mean

    def predict(self, X):
        _, M = self._mean(X)
        if D == 1:
            return np.ascontiguousarray(M[:, 0])
        else:
            return M

    def predict_std(self, X):
        """(mean, std) as predict(X, return_std=True), single-target."""
        Xa = np.asarray(X, dtype=float)
        Kt, M = self._mean(Xa)
        V = sl.solve_triangular(self.L_, np.ascontiguousarray(Kt.T), lower=True, check_finite=False)
        var = self.kernel_.diag(Xa).copy()
        for i in range(var.shape[0]):
            s = 0.0
            for k in range(V.shape[0]):
                s += V[k, i] * V[k, i]
            var[i] -= s
            if var[i] < 0:
                var[i] = 0.0
        sd = np.sqrt(var * self._y_train_std[0] ** 2)
        return np.ascontiguousarray(M[:, 0]), sd

    def predict_cov(self, X):
        Xa = np.asarray(X, dtype=float)
        Kt, M = self._mean(Xa)
        V = sl.solve_triangular(self.L_, np.ascontiguousarray(Kt.T), lower=True, check_finite=False)
        C = self.kernel_(Xa) - V.T @ V
        return np.ascontiguousarray(M[:, 0]), C * self._y_train_std[0] ** 2

    def sample_y(self, X, n_samples: int = 1, random_state = 0):
        rng = check_random_state(random_state)
        m, C = self.predict_cov(X)
        return rng.multivariate_normal(m, C, n_samples).T

    def score(self, X, y, sample_weight = None) -> float:
        from ..metrics import r2_score
        return float(r2_score(np.asarray(y, dtype=float), self.predict(X)))
