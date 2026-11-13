# Ported from scikit-learn 1.9.1 (BSD-3-Clause)

# GaussianProcessClassifier: binary Laplace approximation (Algorithms 3.1,
# 3.2, 5.1 of Rasmussen & Williams) with L-BFGS-B hyperparameter search;
# multiclass one_vs_rest (default) or one_vs_one over binary models.

import math
import numpy as np
import scipy.linalg as sl
from numpy.random.legacy import check_random_state
from ..base import clone, _accuracy
from .kernels import ConstantKernel, RBF
from ._gpr import _optimize

_LAMBDAS = [0.41, 0.4, 0.37, 0.44, 0.39]
_COEFS = [-1854.8214151, 3516.89893646, 221.29346712, 128.12323805, -2010.49422654]

def _expit(x: float) -> float:
    from scipy.special import expit
    return float(expit(x))

class _Binary[K]:
    kernel_: K
    X_train_: np.ndarray[float, 2]
    y_train_: np.ndarray[float, 1]
    classes_: np.ndarray[int, 1]
    max_iter_predict: int
    pi_: np.ndarray[float, 1]
    W_sr_: np.ndarray[float, 1]
    L_: np.ndarray[float, 2]
    lml_: float

    def __init__(self, kernel: K, max_iter_predict: int):
        self.kernel_ = kernel
        self.max_iter_predict = max_iter_predict
        self.X_train_ = np.empty((0, 0))
        self.y_train_ = np.empty(0)
        self.classes_ = np.empty(0, dtype=int)
        self.pi_ = np.empty(0); self.W_sr_ = np.empty(0); self.L_ = np.empty((0, 0))
        self.lml_ = 0.0

    def _posterior_mode(self, Km):
        n = Km.shape[0]
        f = np.zeros(n)
        lml = -np.inf
        pi = np.empty(n); W_sr = np.empty(n); L = np.empty((0, 0))
        b = np.empty(n); a = np.empty(n)
        y = self.y_train_
        for _ in range(self.max_iter_predict):
            pi = np.empty(n)
            for i in range(n):
                pi[i] = _expit(f[i])
            W = pi * (1 - pi)
            W_sr = np.sqrt(W)
            W_sr_K = W_sr[:, None] * Km
            B = np.eye(n) + W_sr_K * W_sr
            L = sl.cholesky(B, lower=True)
            b = W * f + (y - pi)
            a = b - W_sr * sl.cho_solve((L, True), W_sr_K.dot(b))
            f = Km.dot(a)
            v = -0.5 * float(a.dot(f)) - float(np.log1p(np.exp(-(y * 2 - 1) * f)).sum()) - float(np.log(np.diag(L)).sum())
            if v - lml < 1e-10:
                break
            lml = v
        return lml, pi, W_sr, L, b, a

    def _lml_grad(self, theta):
        k = self.kernel_
        k.theta = theta
        Km, KG = k.grad(self.X_train_)
        Z, pi, W_sr, L, b, a = self._posterior_mode(Km)
        m = theta.shape[0]
        d = np.empty(m)
        R = W_sr[:, None] * sl.cho_solve((L, True), np.diag(W_sr))
        Cm = sl.solve(L, W_sr[:, None] * Km)
        s2 = -0.5 * (np.diag(Km) - np.einsum("ij,ij->j", Cm, Cm)) * (pi * (1 - pi) * (1 - 2 * pi))
        for j in range(m):
            C = np.ascontiguousarray(KG[:, :, j])
            s1 = 0.5 * float(a.dot(C).dot(a)) - 0.5 * float(R.T.ravel().dot(C.ravel()))
            bb = C.dot(self.y_train_ - pi)
            s3 = bb - Km.dot(R.dot(bb))
            d[j] = s1 + float(s2.dot(s3))
        return Z, d

    def fit(self, X, y01: np.ndarray[float, 1], optimize: bool, n_restarts: int, rng):
        self.X_train_ = X
        self.y_train_ = y01
        if optimize and self.kernel_.n_dims > 0:
            from ._gpr import _LmlFG
            obj = _LmlFG(self)
            bounds = self.kernel_.bounds
            ths = List[np.ndarray[float, 1]]()
            fs = List[float]()
            t, f = _optimize(obj, self.kernel_.theta.copy(), bounds)
            ths.append(t); fs.append(f)
            for _ in range(n_restarts):
                ti = np.exp(rng.uniform(bounds[:, 0], bounds[:, 1]))
                t, f = _optimize(obj, ti, bounds)
                ths.append(t); fs.append(f)
            best = int(np.argmin(np.array(fs)))
            self.kernel_.theta = ths[best]
            self.lml_ = -fs[best]
        else:
            self.lml_ = self._lml_grad(self.kernel_.theta)[0]
        Km = self.kernel_(X)
        _, self.pi_, self.W_sr_, self.L_, _, _ = self._posterior_mode(Km)
        return self

    def latent(self, X):
        Ks = self.kernel_(self.X_train_, X)
        mean = Ks.T.dot(self.y_train_ - self.pi_)
        v = sl.solve(self.L_, self.W_sr_[:, None] * Ks)
        var = self.kernel_.diag(X) - np.einsum("ij,ij->j", v, v)
        return mean, var

    def proba1(self, X) -> np.ndarray[float, 1]:
        from scipy.special import erf
        mean, var = self.latent(X)
        n = mean.shape[0]
        out = np.empty(n)
        cs = 0.0
        for c in _COEFS:
            cs += c
        for i in range(n):
            alpha = 1 / (2 * var[i])
            s = 0.0
            for t in range(5):
                lam = _LAMBDAS[t]
                g = lam * mean[i]
                integ = math.sqrt(math.pi / alpha) * float(erf(g * math.sqrt(alpha / (alpha + lam ** 2)))) / (2 * math.sqrt(var[i] * 2 * math.pi))
                s += _COEFS[t] * integ
            out[i] = s + 0.5 * cs
        return out

    def predict01(self, X) -> np.ndarray[int, 1]:
        Ks = self.kernel_(self.X_train_, X)
        fs = Ks.T.dot(self.y_train_ - self.pi_)
        return (fs > 0).astype(int)

class GaussianProcessClassifier[K]:
    kernel: K
    optimizer: str
    n_restarts_optimizer: int
    max_iter_predict: int
    multi_class: str
    random_state: Optional[int]
    classes_: np.ndarray[int, 1]
    estimators_: List[_Binary[K]]
    pairs_: List[Tuple[int, int]]
    log_marginal_likelihood_value_: float
    _estimator_type: ClassVar[str] = "classifier"

    def __init__(self, kernel: K = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(1.0, length_scale_bounds="fixed"),
                 *, optimizer = "fmin_l_bfgs_b", n_restarts_optimizer: int = 0,
                 max_iter_predict: int = 100, warm_start: bool = False, copy_X_train: bool = True,
                 random_state: Optional[int] = None, multi_class: str = "one_vs_rest", n_jobs = None):
        self.kernel = kernel
        self.optimizer = "" if isinstance(optimizer, NoneType) else str(optimizer)
        self.n_restarts_optimizer = n_restarts_optimizer
        self.max_iter_predict = max_iter_predict
        self.multi_class = multi_class
        self.random_state = random_state
        self.classes_ = np.empty(0, dtype=int)
        self.estimators_ = []
        self.pairs_ = []
        self.log_marginal_likelihood_value_ = 0.0

    @property
    def n_classes_(self) -> int:
        return self.classes_.shape[0]

    @property
    def kernel_(self):
        return self.estimators_[0].kernel_

    def _fit_bin(self, X, y01, rng) -> _Binary[K]:
        b = _Binary[K](clone(self.kernel), self.max_iter_predict)
        b.fit(np.ascontiguousarray(X), y01, self.optimizer != "", self.n_restarts_optimizer, rng)
        return b

    def fit(self, X, y):
        Xa = np.ascontiguousarray(np.asarray(X, dtype=float))
        ya = np.asarray(y, dtype=int)
        self.classes_ = np.unique(ya)
        k = self.classes_.shape[0]
        if k == 1:
            raise ValueError(f"GaussianProcessClassifier requires 2 or more distinct classes; got 1 class (only class {self.classes_[0]} is present)")
        self.estimators_ = []
        self.pairs_ = []
        if k == 2 or self.multi_class == "one_vs_rest":
            cols = [1] if k == 2 else list(range(k))
            for c in cols:
                rng = check_random_state(self.random_state)
                y01 = np.array([1.0 if v == self.classes_[c] else 0.0 for v in ya])
                self.estimators_.append(self._fit_bin(Xa, y01, rng))
        elif self.multi_class == "one_vs_one":
            for i in range(k):
                for j in range(i + 1, k):
                    rng = check_random_state(self.random_state)
                    idx = [t for t in range(ya.shape[0]) if ya[t] == self.classes_[i] or ya[t] == self.classes_[j]]
                    y01 = np.array([0.0 if ya[t] == self.classes_[i] else 1.0 for t in idx])
                    self.estimators_.append(self._fit_bin(Xa[np.array(idx)], y01, rng))
                    self.pairs_.append((i, j))
        else:
            raise ValueError(f"Unknown multi-class mode {self.multi_class}")
        s = 0.0
        for e in self.estimators_:
            s += e.lml_
        self.log_marginal_likelihood_value_ = s / float(len(self.estimators_))
        return self

    def predict_proba(self, X):
        Xa = np.ascontiguousarray(np.asarray(X, dtype=float))
        n = Xa.shape[0]
        k = self.classes_.shape[0]
        if k > 2 and self.multi_class == "one_vs_one":
            raise ValueError("one_vs_one multi-class mode does not support predicting probability estimates. Use one_vs_rest mode instead.")
        if k == 2:
            p = self.estimators_[0].proba1(Xa)
            out = np.empty((n, 2))
            for i in range(n):
                out[i, 0] = 1 - p[i]; out[i, 1] = p[i]
            return out
        P = np.empty((n, k))
        for c in range(k):
            p = self.estimators_[c].proba1(Xa)
            for i in range(n):
                P[i, c] = p[i]
        for i in range(n):
            s = 0.0
            for c in range(k):
                s += P[i, c]
            if s != 0:
                for c in range(k):
                    P[i, c] /= s
        return P

    def predict(self, X):
        Xa = np.ascontiguousarray(np.asarray(X, dtype=float))
        n = Xa.shape[0]
        k = self.classes_.shape[0]
        out = np.empty(n, dtype=int)
        if k == 2:
            p = self.estimators_[0].predict01(Xa)
            for i in range(n):
                out[i] = self.classes_[p[i]]
            return out
        if self.multi_class == "one_vs_rest":
            # OneVsRestClassifier.predict: max of predict_proba[:, 1] per class
            maxima = np.full(n, -np.inf)
            arg = np.zeros(n, dtype=int)
            for t in range(k):
                c = k - 1 - t
                pred = self.estimators_[c].proba1(Xa)
                for i in range(n):
                    if pred[i] > maxima[i]:
                        maxima[i] = pred[i]
                    if maxima[i] == pred[i]:
                        arg[i] = c
            for i in range(n):
                out[i] = self.classes_[arg[i]]
            return out
        from ..utils.multiclass import _ovr_decision_function
        m = len(self.estimators_)
        Pm = np.empty((n, m)); Cm = np.empty((n, m))
        for t in range(m):
            p01 = self.estimators_[t].predict01(Xa)
            pr = self.estimators_[t].proba1(Xa)
            for i in range(n):
                Pm[i, t] = float(p01[i]); Cm[i, t] = pr[i]
        Y = _ovr_decision_function(Pm, Cm, k)
        idx = np.argmax(Y, axis=1)
        for i in range(n):
            out[i] = self.classes_[idx[i]]
        return out

    def log_marginal_likelihood(self) -> float:
        return self.log_marginal_likelihood_value_

    def score(self, X, y, sample_weight = None) -> float:
        return _accuracy(y, self.predict(X))
