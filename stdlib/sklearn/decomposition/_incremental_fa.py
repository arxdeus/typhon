# Ported from scikit-learn 1.9.1 (BSD-3-Clause)

from ..base import estimator_repr as _estimator_repr
import numpy as np
import scipy.linalg as sl
from ._extmath import svd_flip_v, squared_norm, fast_logdet

class IncrementalPCA:
    def __repr__(self) -> str:
        return _estimator_repr(self)

    n_components: Optional[int]
    whiten: bool
    batch_size: Optional[int]
    batch_size_: int
    components_: np.ndarray[float, 2]
    singular_values_: np.ndarray[float, 1]
    explained_variance_: np.ndarray[float, 1]
    explained_variance_ratio_: np.ndarray[float, 1]
    mean_: np.ndarray[float, 1]
    var_: np.ndarray[float, 1]
    noise_variance_: float
    n_samples_seen_: int
    n_components_: int
    _fitted: bool

    def __init__(self, n_components: Optional[int] = None, *, whiten: bool = False,
                 copy: bool = True, batch_size: Optional[int] = None):
        self.n_components = n_components
        self.whiten = whiten
        self.batch_size = batch_size
        self._reset()

    def _reset(self):
        self.batch_size_ = 0
        self.components_ = np.empty((0, 0))
        self.singular_values_ = np.empty(0)
        self.explained_variance_ = np.empty(0)
        self.explained_variance_ratio_ = np.empty(0)
        self.mean_ = np.empty(0)
        self.var_ = np.empty(0)
        self.noise_variance_ = 0.0
        self.n_samples_seen_ = 0
        self.n_components_ = 0
        self._fitted = False

    def fit(self, X, y = None):
        self._reset()
        Xa = np.asarray(X, dtype=float)
        n_samples, n_features = Xa.shape
        self.batch_size_ = 5 * n_features if self.batch_size is None else self.batch_size.__val__()
        min_bs = 0 if self.n_components is None else self.n_components.__val__()
        # gen_batches(n, batch_size, min_batch_size)
        start = 0
        for _ in range(n_samples // self.batch_size_):
            end = start + self.batch_size_
            if end + min_bs > n_samples:
                continue
            self.partial_fit(Xa[start:end])
            start = end
        if start < n_samples:
            self.partial_fit(Xa[start:])
        return self

    def partial_fit(self, X, y = None):
        Xb = np.array(X, dtype=float)
        n_samples, n_features = Xb.shape
        first = not self._fitted
        if self.n_components is None:
            self.n_components_ = min(n_samples, n_features) if first else self.components_.shape[0]
        else:
            k = self.n_components.__val__()
            if not k <= n_features:
                raise ValueError(f"n_components={k} invalid for n_features={n_features}")
            if k > n_samples and first:
                raise ValueError(f"n_components={k} must be less or equal to the batch number of samples {n_samples} for the first partial_fit call.")
            self.n_components_ = k
        if not first and self.components_.shape[0] != self.n_components_:
            raise ValueError("Number of input features has changed between calls to partial_fit")
        if first:
            self.mean_ = np.zeros(n_features)
            self.var_ = np.zeros(n_features)
        # _incremental_mean_and_var
        last_n = float(self.n_samples_seen_)
        last_sum = self.mean_ * last_n
        new_sum = np.sum(Xb, axis=0)
        nn = float(n_samples)
        upd_n = last_n + nn
        col_mean = (last_sum + new_sum) / upd_n
        T = new_sum / nn
        temp = Xb - T
        correction = np.sum(temp, axis=0)
        new_unnorm = np.sum(temp * temp, axis=0)
        new_unnorm -= correction ** 2 / nn
        col_var = np.empty(n_features)
        if last_n == 0:
            col_var = new_unnorm / upd_n
        else:
            last_unnorm = self.var_ * last_n
            lon = last_n / nn
            col_var = (last_unnorm + new_unnorm + lon / upd_n * (last_sum / lon - new_sum) ** 2) / upd_n
        n_total = int(upd_n)
        Xs = Xb
        if self.n_samples_seen_ == 0:
            Xs = Xb - col_mean
        else:
            bmean = np.mean(Xb, axis=0)
            Xc = Xb - bmean
            mc = np.sqrt(float(self.n_samples_seen_) / upd_n * nn) * (self.mean_ - bmean)
            Xs = np.vstack((self.singular_values_.reshape((-1, 1)) * self.components_, Xc,
                            mc.reshape((1, -1))))
        U, S, Vt = sl.svd(Xs, full_matrices=False)
        svd_flip_v(U, Vt)
        ev = S ** 2 / float(n_total - 1)
        evr = S ** 2 / float(np.sum(col_var * upd_n))
        k = self.n_components_
        self.n_samples_seen_ = n_total
        self.components_ = np.ascontiguousarray(Vt[:k])
        self.singular_values_ = S[:k].copy()
        self.mean_ = col_mean
        self.var_ = col_var
        self.explained_variance_ = ev[:k].copy()
        self.explained_variance_ratio_ = evr[:k].copy()
        if k != n_samples and k != n_features:
            self.noise_variance_ = float(ev[k:].mean())
        else:
            self.noise_variance_ = 0.0
        self._fitted = True
        return self

    def transform(self, X):
        Xt = (np.asarray(X, dtype=float) - self.mean_) @ self.components_.T
        if self.whiten:
            Xt /= np.sqrt(self.explained_variance_)
        return Xt

    def fit_transform(self, X, y = None):
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        Xa = np.asarray(X, dtype=float)
        if self.whiten:
            return (Xa * np.sqrt(self.explained_variance_)) @ self.components_ + self.mean_
        return Xa @ self.components_ + self.mean_


def _ortho_rotation(components, method: str, tol: float = 1e-6, max_iter: int = 100):
    nrow, ncol = components.shape
    R = np.eye(ncol)
    var = 0.0
    for _ in range(max_iter):
        comp_rot = components @ R
        tmp = np.zeros_like(comp_rot)
        if method == "varimax":
            tmp = comp_rot * ((comp_rot ** 2).sum(axis=0) / nrow)
        u, s, v = np.linalg.svd(components.T @ (comp_rot ** 3 - tmp))
        R = u @ v
        var_new = float(np.sum(s))
        if var != 0 and var_new < var * (1 + tol):
            break
        var = var_new
    return (components @ R).T

class FactorAnalysis:
    def __repr__(self) -> str:
        return _estimator_repr(self)

    """svd_method 'lapack'; rotation None/'varimax'/'quartimax'."""
    n_components: Optional[int]
    tol: float
    max_iter: int
    rotation: str
    noise_variance_init: Optional[np.ndarray[float, 1]]
    components_: np.ndarray[float, 2]
    noise_variance_: np.ndarray[float, 1]
    mean_: np.ndarray[float, 1]
    loglike_: List[float]
    n_iter_: int

    def __init__(self, n_components: Optional[int] = None, *, tol: float = 1e-2, copy: bool = True,
                 max_iter: int = 1000, noise_variance_init: Optional[np.ndarray[float, 1]] = None,
                 svd_method: str = "lapack", iterated_power: int = 3,
                 rotation: Optional[str] = None, random_state = 0):
        if svd_method != "lapack":
            raise NotImplementedError("FactorAnalysis: only svd_method='lapack' is ported")
        self.n_components = n_components
        self.tol = tol
        self.max_iter = max_iter
        self.rotation = "" if rotation is None else rotation.__val__()
        self.noise_variance_init = noise_variance_init
        self.components_ = np.empty((0, 0))
        self.noise_variance_ = np.empty(0)
        self.mean_ = np.empty(0)
        self.loglike_ = []
        self.n_iter_ = 0

    def fit(self, X, y = None):
        Xa = np.array(X, dtype=float)
        n_samples, n_features = Xa.shape
        k = n_features if self.n_components is None else self.n_components.__val__()
        self.mean_ = np.mean(Xa, axis=0)
        Xa -= self.mean_
        nsqrt = np.sqrt(float(n_samples))
        llconst = n_features * np.log(2.0 * np.pi) + k
        var = np.var(Xa, axis=0)
        psi = np.ones(n_features)
        if self.noise_variance_init is not None:
            psi = np.array(self.noise_variance_init.__val__())
        loglike: List[float] = []
        old_ll = -np.inf
        SMALL = 1e-12
        W = np.empty((0, 0))
        it = 0
        for i in range(self.max_iter):
            it = i
            sqrt_psi = np.sqrt(psi) + SMALL
            _, s0, Vt0 = sl.svd(Xa / (sqrt_psi * nsqrt), full_matrices=False)
            s = s0[:k] ** 2
            Vt = Vt0[:k]
            unexp = squared_norm(s0[k:])
            W = np.sqrt(np.maximum(s - 1.0, 0.0))[:, None] * Vt
            W *= sqrt_psi
            ll = llconst + float(np.sum(np.log(s)))
            ll += unexp + float(np.sum(np.log(psi)))
            ll *= -n_samples / 2.0
            loglike.append(ll)
            if ll - old_ll < self.tol:
                break
            old_ll = ll
            psi = np.maximum(var - np.sum(W ** 2, axis=0), SMALL)
        self.components_ = W
        if self.rotation != "":
            self.components_ = np.ascontiguousarray(_ortho_rotation(W.T, self.rotation)[:k])
        self.noise_variance_ = psi
        self.loglike_ = loglike
        self.n_iter_ = it + 1
        return self

    def transform(self, X):
        Ih = np.eye(self.components_.shape[0])
        Xt = np.asarray(X, dtype=float) - self.mean_
        Wpsi = self.components_ / self.noise_variance_
        cov_z = sl.inv(Ih + Wpsi @ self.components_.T)
        return (Xt @ Wpsi.T) @ cov_z

    def fit_transform(self, X, y = None):
        return self.fit(X).transform(X)

    def get_covariance(self):
        cov = self.components_.T @ self.components_
        for i in range(cov.shape[0]):
            cov[i, i] += self.noise_variance_[i]
        return cov

    def get_precision(self):
        nf = self.components_.shape[1]
        k = self.components_.shape[0]
        if k == 0:
            return np.diag(1.0 / self.noise_variance_)
        if k == nf:
            return sl.inv(self.get_covariance())
        C = self.components_
        P = (C / self.noise_variance_) @ C.T
        for i in range(k):
            P[i, i] += 1.0
        P = C.T @ (sl.inv(P) @ C)
        P /= self.noise_variance_[:, None]
        P /= -self.noise_variance_[None, :]
        for i in range(nf):
            P[i, i] += 1.0 / self.noise_variance_[i]
        return P

    def score_samples(self, X):
        Xr = np.asarray(X, dtype=float) - self.mean_
        P = self.get_precision()
        nf = Xr.shape[1]
        ll = -0.5 * (Xr * (Xr @ P)).sum(axis=1)
        ll -= 0.5 * (nf * np.log(2.0 * np.pi) - fast_logdet(P))
        return ll

    def score(self, X, y = None) -> float:
        return float(np.mean(self.score_samples(X)))
