# Ported from scikit-learn 1.9.1 (BSD-3-Clause)

import numpy as np
import scipy.linalg as sl

class PCA:
    """svd_solver 'auto' / 'full' / 'covariance_eigh' (the paths sklearn's
    'auto' picks for dense data with <= 500 rows or tall data); integer or
    None n_components; whiten."""
    n_components: Optional[int]
    whiten: bool
    svd_solver: str
    components_: np.ndarray[float, 2]
    explained_variance_: np.ndarray[float, 1]
    explained_variance_ratio_: np.ndarray[float, 1]
    singular_values_: np.ndarray[float, 1]
    mean_: np.ndarray[float, 1]
    n_components_: int
    noise_variance_: float
    n_samples_: int
    random_state: Optional[int]
    iterated_power: int
    n_oversamples: int
    power_iteration_normalizer: str

    def __init__(self, n_components: Optional[int] = None, *, copy: bool = True,
                 whiten: bool = False, svd_solver: str = "auto", tol: float = 0.0,
                 iterated_power = "auto", n_oversamples: int = 10,
                 power_iteration_normalizer: str = "auto", random_state: Optional[int] = None):
        self.n_components = n_components
        self.random_state = random_state
        self.iterated_power = -1 if isinstance(iterated_power, str) else iterated_power
        self.n_oversamples = n_oversamples
        self.power_iteration_normalizer = power_iteration_normalizer
        self.whiten = whiten
        self.svd_solver = svd_solver
        self.components_ = np.empty((0, 0))
        self.explained_variance_ = np.empty(0)
        self.explained_variance_ratio_ = np.empty(0)
        self.singular_values_ = np.empty(0)
        self.mean_ = np.empty(0)
        self.n_components_ = 0
        self.noise_variance_ = 0.0
        self.n_samples_ = 0

    def fit(self, X, y = None):
        Xa = np.asarray(X, dtype=float)
        n, p = Xa.shape
        k = min(n, p) if self.n_components is None else self.n_components
        if not (0 <= k <= min(n, p)):
            raise ValueError(f"n_components={k} must be between 0 and min(n_samples, n_features)={min(n, p)}")
        solver = self.svd_solver
        if solver == "auto":
            if p <= 1000 and n >= 10 * p:
                solver = "covariance_eigh"
            elif max(n, p) <= 500:
                solver = "full"
            elif 1 <= k < 0.8 * min(n, p):
                solver = "randomized"
            else:
                solver = "full"
        self.mean_ = np.mean(Xa, axis=0)
        Vt = np.empty((0, 0))
        S = np.empty(0)
        ev = np.empty(0)
        if solver == "full":
            Xc = Xa - self.mean_
            _, S0, Vt0 = sl.svd(Xc, full_matrices=False)
            S = S0
            Vt = Vt0
            ev = (S ** 2) / float(n - 1)
        elif solver == "covariance_eigh":
            C = Xa.T @ Xa
            C -= float(n) * self.mean_.reshape((-1, 1)) * self.mean_.reshape((1, -1))
            C /= float(n - 1)
            w, V = np.linalg.eigh(C)
            w = np.ascontiguousarray(w[::-1])
            V = np.ascontiguousarray(V[:, ::-1])
            for i in range(w.shape[0]):
                if w[i] < 0.0:
                    w[i] = 0.0
            ev = w
            S = np.sqrt(w * float(n - 1))
            Vt = np.ascontiguousarray(V.T)
        elif solver == "randomized":
            from ._extmath import check_random_state, randomized_svd, svd_flip_v
            if not 1 <= k <= min(n, p):
                raise ValueError(f"n_components={k} must be between 1 and min(n_samples, n_features)")
            Xc = Xa - self.mean_
            U, S0, Vt0 = randomized_svd(Xc, k, check_random_state(self.random_state),
                                        n_oversamples=self.n_oversamples, n_iter=self.iterated_power,
                                        normalizer=self.power_iteration_normalizer, flip_sign=False)
            svd_flip_v(U, Vt0)
            total_var = float(np.sum(Xc * Xc)) / float(n - 1)
            self.n_samples_ = n
            self.n_components_ = k
            self.components_ = Vt0
            self.explained_variance_ = S0 ** 2 / float(n - 1)
            self.explained_variance_ratio_ = self.explained_variance_ / total_var
            self.singular_values_ = S0
            if k < min(p, n):
                self.noise_variance_ = (total_var - float(np.sum(self.explained_variance_))) / float(min(p, n) - k)
            else:
                self.noise_variance_ = 0.0
            return self
        else:
            raise NotImplementedError(f"PCA svd_solver={solver}")
        # svd_flip(u_based_decision=False): largest |v| in each row positive
        for r in range(Vt.shape[0]):
            best = 0
            for c in range(Vt.shape[1]):
                if abs(Vt[r, c]) > abs(Vt[r, best]):
                    best = c
            if Vt[r, best] < 0:
                for c in range(Vt.shape[1]):
                    Vt[r, c] = -Vt[r, c]
        total = np.sum(ev)
        evr = ev / total
        if k < min(p, n):
            self.noise_variance_ = float(np.mean(ev[k:]))
        else:
            self.noise_variance_ = 0.0
        self.n_samples_ = n
        self.n_components_ = k
        self.components_ = np.ascontiguousarray(Vt[:k])
        self.explained_variance_ = ev[:k].copy()
        self.explained_variance_ratio_ = evr[:k].copy()
        self.singular_values_ = S[:k].copy()
        return self

    def transform(self, X):
        Xa = np.asarray(X, dtype=float)
        Xt = (Xa - self.mean_) @ self.components_.T
        if self.whiten:
            Xt /= np.sqrt(self.explained_variance_)
        return Xt

    def fit_transform(self, X, y = None):
        return self.fit(X, y).transform(X)

    def inverse_transform(self, X):
        Xa = np.asarray(X, dtype=float)
        if self.whiten:
            return (Xa * np.sqrt(self.explained_variance_)) @ self.components_ + self.mean_
        return Xa @ self.components_ + self.mean_

    def get_covariance(self):
        C = self.components_
        ev = self.explained_variance_
        if self.whiten:
            C = C * np.sqrt(ev[:, None])
        d = np.where(ev > self.noise_variance_, ev - self.noise_variance_, 0.0)
        cov = (C.T * d) @ C
        for i in range(cov.shape[0]):
            cov[i, i] += self.noise_variance_
        return cov

    def get_precision(self):
        nf = self.components_.shape[1]
        if self.n_components_ == 0:
            return np.eye(nf) / self.noise_variance_
        if self.noise_variance_ == 0.0:
            return sl.inv(self.get_covariance())
        C = self.components_
        ev = self.explained_variance_
        if self.whiten:
            C = C * np.sqrt(ev[:, None])
        d = np.where(ev > self.noise_variance_, ev - self.noise_variance_, 0.0)
        P = (C @ C.T) / self.noise_variance_
        for i in range(P.shape[0]):
            P[i, i] += 1.0 / d[i]
        P = C.T @ sl.inv(P) @ C
        P /= -(self.noise_variance_ ** 2)
        for i in range(nf):
            P[i, i] += 1.0 / self.noise_variance_
        return P

    def score_samples(self, X):
        from ._extmath import fast_logdet
        Xr = np.asarray(X, dtype=float) - self.mean_
        nf = Xr.shape[1]
        P = self.get_precision()
        ll = -0.5 * np.sum(Xr * (Xr @ P), axis=1)
        ll -= 0.5 * (nf * np.log(2.0 * np.pi) - fast_logdet(P))
        return ll

    def score(self, X, y = None) -> float:
        return float(np.mean(self.score_samples(X)))

# ------------------------------------------------------------ FastICA

def _sym_decorrelation(W):
    s, u = sl.eigh(W @ W.T)
    tiny = 2.2250738585072014e-308
    for i in range(s.shape[0]):
        if s[i] < tiny:
            s[i] = tiny
    return ((u * (1.0 / np.sqrt(s))) @ u.T) @ W

def _g(fun: str, x, alpha: float):
    # returns (g(x), mean over samples of g'(x)) per component row
    n = x.shape[0]
    gp = np.empty(n)
    if fun == "logcosh":
        gx = np.tanh(x * alpha)
        for i in range(n):
            gp[i] = np.mean(alpha * (1 - gx[i] ** 2))
        return gx, gp
    elif fun == "exp":
        e = np.exp(-(x ** 2) / 2)
        gx = x * e
        g_x = (1 - x ** 2) * e
        for i in range(n):
            gp[i] = np.mean(g_x[i])
        return gx, gp
    elif fun == "cube":
        gx = x ** 3
        for i in range(n):
            gp[i] = np.mean(3 * x[i] ** 2)
        return gx, gp
    else:
        raise ValueError(f"unknown fun {fun}")

class FastICA:
    """algorithm 'parallel' / 'deflation', fun logcosh/exp/cube, whiten
    'unit-variance' / 'arbitrary-variance' / False, whiten_solver svd/eigh.
    w_init drawn from numpy's legacy RandomState(random_state).normal, so an
    integer random_state reproduces sklearn exactly."""
    n_components: Optional[int]
    algorithm: str
    whiten: str
    fun: str
    alpha: float
    max_iter: int
    tol: float
    whiten_solver: str
    random_state: Optional[int]
    components_: np.ndarray[float, 2]
    mixing_: np.ndarray[float, 2]
    mean_: np.ndarray[float, 1]
    whitening_: np.ndarray[float, 2]
    n_iter_: int

    def __init__(self, n_components: Optional[int] = None, *, algorithm: str = "parallel",
                 whiten = "unit-variance", fun: str = "logcosh", fun_args: Optional[Dict[str, float]] = None,
                 max_iter: int = 200, tol: float = 1e-4, w_init = None,
                 whiten_solver: str = "svd", random_state: Optional[int] = None):
        self.n_components = n_components
        self.algorithm = algorithm
        if isinstance(whiten, bool):
            if whiten:
                raise ValueError("whiten=True: use 'unit-variance' or 'arbitrary-variance'")
            self.whiten = ""
        else:
            self.whiten = whiten
        self.fun = fun
        self.alpha = 1.0 if fun_args is None else fun_args.__val__().get("alpha", 1.0)
        self.max_iter = max_iter
        self.tol = tol
        self.whiten_solver = whiten_solver
        self.random_state = random_state
        self.components_ = np.empty((0, 0))
        self.mixing_ = np.empty((0, 0))
        self.mean_ = np.empty(0)
        self.whitening_ = np.empty((0, 0))
        self.n_iter_ = 0

    def _fit(self, X):
        from ..utils import _LegacyGauss
        XT = np.ascontiguousarray(np.asarray(X, dtype=float).T.copy())
        nf, ns = XT.shape
        k = min(ns, nf)
        if self.whiten and self.n_components is not None:
            k = min(self.n_components.__val__(), k)
        K = np.empty((0, 0))
        X1 = XT
        if self.whiten:
            mean = XT.mean(axis=-1)
            XT -= mean[:, None]
            self.mean_ = mean
            u = np.empty((0, 0))
            d = np.empty(0)
            if self.whiten_solver == "eigh":
                dd, uu = sl.eigh(XT @ XT.T)
                order = np.argsort(dd)[::-1]
                eps = 2.220446049250313e-16 * 10
                for i in range(dd.shape[0]):
                    if dd[i] < eps:
                        dd[i] = eps
                d = np.sqrt(dd)[order]
                u = np.ascontiguousarray(uu[:, order])
            else:
                U, S, _ = sl.svd(XT, full_matrices=False)
                u = U
                d = S
            sg = np.sign(u[0])
            u = u * sg
            K = np.ascontiguousarray((u / d).T[:k])
            X1 = (K @ XT) * np.sqrt(float(ns))
        if self.random_state is None:
            raise ValueError("FastICA needs an integer random_state in Typhon")
        w_init = _LegacyGauss(self.random_state.__val__()).normal(k, k)
        W = np.empty((0, 0))
        if self.algorithm == "parallel":
            W = _sym_decorrelation(w_init)
            p_ = float(X1.shape[1])
            it = 0
            while it < self.max_iter:
                gwtx, g_wtx = _g(self.fun, W @ X1, self.alpha)
                W1 = _sym_decorrelation((gwtx @ X1.T) / p_ - g_wtx[:, None] * W)
                lim = float(np.max(np.abs(np.abs(np.sum(W1 * W, axis=1)) - 1)))
                W = W1
                it += 1
                if lim < self.tol:
                    break
            self.n_iter_ = it
        else:
            # _ica_def: one component at a time, Gram-Schmidt deflation
            W = np.zeros((k, k))
            mx = 0
            for j in range(k):
                w = w_init[j, :].copy()
                w /= np.sqrt((w ** 2).sum())
                it = 0
                while it < self.max_iter:
                    gwtx, g_wtx = _g(self.fun, (w @ X1).reshape((1, -1)), self.alpha)
                    w1 = (X1 * gwtx[0]).mean(axis=1) - g_wtx[0] * w
                    if j > 0:
                        w1 -= (w1 @ W[:j].T) @ W[:j]
                    w1 /= np.sqrt((w1 ** 2).sum())
                    lim = abs(abs((w1 * w).sum()) - 1)
                    w = w1
                    it += 1
                    if lim < self.tol:
                        break
                mx = max(mx, it)
                W[j, :] = w
            self.n_iter_ = mx
        if self.whiten:
            if self.whiten == "unit-variance":
                S = ((W @ K) @ XT).T
                Sstd = np.std(S, axis=0)
                W = W / Sstd[:, None]
            self.components_ = W @ K
            self.whitening_ = K
        else:
            self.components_ = W
            self.mean_ = np.zeros(nf)
        self.mixing_ = sl.pinv(self.components_)
        return self

    def fit(self, X, y = None):
        return self._fit(X)

    def transform(self, X):
        Xa = np.asarray(X, dtype=float)
        if self.whiten:
            Xa = Xa - self.mean_
        return Xa @ self.components_.T

    def fit_transform(self, X, y = None):
        return self._fit(X).transform(X)

    def inverse_transform(self, X):
        out = np.asarray(X, dtype=float) @ self.mixing_.T
        if self.whiten:
            out += self.mean_
        return out

from ._truncated_svd import TruncatedSVD
from ._nmf import NMF
from ._kernel_pca import KernelPCA
from ._incremental_fa import IncrementalPCA, FactorAnalysis
