# Ported from scikit-learn 1.9.1 (BSD-3-Clause)

# LogisticRegression, L2 penalty, same objective as sklearn's lbfgs path:
#   mean_i loss_i(w) + 0.5 / (C * n) * ||coef||^2   (intercept not penalized)
# binary: HalfBinomialLoss, multiclass: HalfMultinomialLoss (symmetric
# n_classes x n_features parameterization).
# DEVIATION: the minimizer is found with damped Newton iterations to machine
# precision instead of scipy L-BFGS-B stopped at gtol=tol. The objective is
# strictly convex in coef, so both converge to the same optimum; sklearn's own
# coefficients carry an error of order `tol` (default 1e-4), ours are exact.

import numpy as np
import scipy.linalg as sl
from ..base import _accuracy

def _log1pexp(x: float) -> float:
    if x <= -37.0:
        return np.exp(x)
    elif x <= -2.0:
        return np.log1p(np.exp(x))
    elif x <= 18.0:
        return x + np.log1p(np.exp(-x))
    else:
        return x + np.exp(-x)

def _expit(x: float) -> float:
    if x >= 0.0:
        return 1.0 / (1.0 + np.exp(-x))
    else:
        e = np.exp(x)
        return e / (1.0 + e)

class LogisticRegression:
    penalty: str
    C: float
    tol: float
    fit_intercept: bool
    intercept_scaling: float
    max_iter: int
    random_state: Optional[int]
    solver: str
    verbose: int
    warm_start: bool
    n_jobs: Optional[int]
    l1_ratio: Optional[float]
    classes_: np.ndarray[int, 1]
    coef_: np.ndarray[float, 2]
    intercept_: np.ndarray[float, 1]
    n_iter_: np.ndarray[int, 1]
    n_features_in_: int

    def __init__(self, penalty: str = "l2", *, dual: bool = False, tol: float = 1e-4,
                 C: float = 1.0, fit_intercept: bool = True, intercept_scaling: float = 1.0,
                 class_weight = None, random_state: Optional[int] = None,
                 solver: str = "lbfgs", max_iter: int = 100, verbose: int = 0,
                 warm_start: bool = False, n_jobs: Optional[int] = None,
                 l1_ratio: Optional[float] = None):
        if penalty != "l2":
            raise NotImplementedError("LogisticRegression: only penalty='l2' is ported")
        if C <= 0.0:
            raise ValueError("Penalty term must be positive")
        self.penalty = penalty
        self.C = C
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling
        self.max_iter = max_iter
        self.random_state = random_state
        self.solver = solver
        self.verbose = verbose
        self.warm_start = warm_start
        self.n_jobs = n_jobs
        self.l1_ratio = l1_ratio
        self.classes_ = np.empty(0, dtype=int)
        self.coef_ = np.empty((0, 0))
        self.intercept_ = np.empty(0)
        self.n_iter_ = np.zeros(1, dtype=int)
        self.n_features_in_ = 0

    # objective in the flat parameter vector w (K blocks of p [+1])
    def _obj(self, w, Xa, Y, K: int, binary: bool, l2: float) -> float:
        n, p = Xa.shape
        d = p + (1 if self.fit_intercept else 0)
        tot = 0.0
        if binary:
            for i in range(n):
                z = w[p] if self.fit_intercept else 0.0
                for j in range(p):
                    z += Xa[i, j] * w[j]
                tot += _log1pexp(z) - Y[i, 0] * z
        else:
            z = np.empty(K)
            for i in range(n):
                for k in range(K):
                    s = w[k * d + p] if self.fit_intercept else 0.0
                    for j in range(p):
                        s += Xa[i, j] * w[k * d + j]
                    z[k] = s
                m = np.max(z)
                se = 0.0
                for k in range(K):
                    se += np.exp(z[k] - m)
                lse = m + np.log(se)
                for k in range(K):
                    if Y[i, k] > 0.5:
                        tot += lse - z[k]
        reg = 0.0
        nb = 1 if binary else K
        for k in range(nb):
            for j in range(p):
                reg += w[k * d + j] * w[k * d + j]
        return tot / float(n) + 0.5 * l2 * reg

    def _grad_hess(self, w, Xa, Y, K: int, binary: bool, l2: float):
        n, p = Xa.shape
        d = p + (1 if self.fit_intercept else 0)
        nb = 1 if binary else K
        D = nb * d
        g = np.zeros(D)
        H = np.zeros((D, D))
        xr = np.empty(d)
        prob = np.empty(nb)
        z = np.empty(K)
        for i in range(n):
            for j in range(p):
                xr[j] = Xa[i, j]
            if self.fit_intercept:
                xr[p] = 1.0
            if binary:
                s = 0.0
                for j in range(d):
                    s += xr[j] * w[j]
                prob[0] = _expit(s)
                r = prob[0] - Y[i, 0]
                h = prob[0] * (1.0 - prob[0])
                for a in range(d):
                    g[a] += r * xr[a]
                    for b in range(d):
                        H[a, b] += h * xr[a] * xr[b]
            else:
                for k in range(K):
                    s = 0.0
                    for j in range(d):
                        s += xr[j] * w[k * d + j]
                    z[k] = s
                m = np.max(z)
                se = 0.0
                for k in range(K):
                    se += np.exp(z[k] - m)
                for k in range(K):
                    prob[k] = np.exp(z[k] - m) / se
                for k in range(K):
                    r = prob[k] - Y[i, k]
                    for a in range(d):
                        g[k * d + a] += r * xr[a]
                    for l in range(K):
                        c = (prob[k] if k == l else 0.0) - prob[k] * prob[l]
                        for a in range(d):
                            for b in range(d):
                                H[k * d + a, l * d + b] += c * xr[a] * xr[b]
        g /= float(n)
        H /= float(n)
        for k in range(nb):
            for j in range(p):
                g[k * d + j] += l2 * w[k * d + j]
                H[k * d + j, k * d + j] += l2
        return g, H

    def fit(self, X, y, sample_weight = None):
        Xa = np.asarray(X, dtype=float)
        ya = np.asarray(y, dtype=int)
        n, p = Xa.shape
        classes, yi = np.unique(ya, return_inverse=True)
        K = classes.shape[0]
        if K < 2:
            raise ValueError("This solver needs samples of at least 2 classes in the data")
        self.classes_ = classes
        self.n_features_in_ = p
        binary = K == 2
        nb = 1 if binary else K
        Y = np.zeros((n, nb if binary else K))
        for i in range(n):
            if binary:
                Y[i, 0] = 1.0 if yi[i] == 1 else 0.0
            else:
                Y[i, yi[i]] = 1.0
        d = p + (1 if self.fit_intercept else 0)
        l2 = 1.0 / (self.C * float(n))
        w = np.zeros(nb * d)
        f = self._obj(w, Xa, Y, K, binary, l2)
        it = 0
        while it < max(self.max_iter, 100):
            it += 1
            g, H = self._grad_hess(w, Xa, Y, K, binary, l2)
            gmax = np.max(np.abs(g))
            if gmax < 1e-13:
                break
            # multinomial intercepts: H is singular along sum(intercepts);
            # the gradient has no component there, lstsq gives the min-norm step
            if binary:
                step = sl.solve(H, g, assume_a="pos")
            else:
                step = np.linalg.lstsq(H, g, rcond=None)[0]
            t = 1.0
            wn = w - step
            fn = self._obj(wn, Xa, Y, K, binary, l2)
            dec = float(np.dot(g, step))
            while fn > f - 1e-4 * t * dec and t > 1e-10:
                t *= 0.5
                wn = w - t * step
                fn = self._obj(wn, Xa, Y, K, binary, l2)
            if f - fn <= 1e-16 * max(1.0, abs(f)) and t < 1.0:
                w = wn if fn <= f else w
                break
            w = wn
            f = fn
        self.n_iter_ = np.array([it])
        coef = np.empty((nb, p))
        inter = np.zeros(nb)
        for k in range(nb):
            for j in range(p):
                coef[k, j] = w[k * d + j]
            if self.fit_intercept:
                inter[k] = w[k * d + p]
        self.coef_ = coef
        self.intercept_ = inter
        return self

    def decision_function(self, X):
        Xa = np.asarray(X, dtype=float)
        return Xa @ self.coef_.T + self.intercept_

    def predict(self, X):
        s = self.decision_function(X)
        n = s.shape[0]
        out = np.empty(n, dtype=int)
        if s.shape[1] == 1:
            for i in range(n):
                out[i] = self.classes_[1] if s[i, 0] > 0 else self.classes_[0]
        else:
            idx = np.argmax(s, axis=1)
            for i in range(n):
                out[i] = self.classes_[idx[i]]
        return out

    def predict_proba(self, X):
        s = self.decision_function(X)
        n = s.shape[0]
        if s.shape[1] == 1:
            pr = np.empty((n, 2))
            for i in range(n):
                e = _expit(s[i, 0])
                pr[i, 1] = e
                pr[i, 0] = 1.0 - e
            return pr
        else:
            m = np.max(s, axis=1, keepdims=True)
            E = np.exp(s - m)
            return E / np.sum(E, axis=1, keepdims=True)

    def predict_log_proba(self, X):
        return np.log(self.predict_proba(X))

    def score(self, X, y, sample_weight = None):
        return _accuracy(y, self.predict(X))
from ._bayes import BayesianRidge
