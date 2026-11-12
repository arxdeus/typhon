# Ported from scikit-learn 1.9.1 (BSD-3-Clause)

import numpy as np
import scipy.linalg as sl
from ._extmath import check_random_state, randomized_svd, svd_flip_v

class TruncatedSVD:
    """algorithm 'arpack' (computed with a dense LAPACK SVD: same subspace,
    signs fixed by svd_flip) or 'randomized' (sklearn's random stream)."""
    n_components: int
    algorithm: str
    n_iter: int
    n_oversamples: int
    power_iteration_normalizer: str
    random_state: Optional[int]
    tol: float
    components_: np.ndarray[float, 2]
    explained_variance_: np.ndarray[float, 1]
    explained_variance_ratio_: np.ndarray[float, 1]
    singular_values_: np.ndarray[float, 1]

    def __init__(self, n_components: int = 2, *, algorithm: str = "randomized", n_iter: int = 5,
                 n_oversamples: int = 10, power_iteration_normalizer: str = "auto",
                 random_state: Optional[int] = None, tol: float = 0.0):
        self.n_components = n_components
        self.algorithm = algorithm
        self.n_iter = n_iter
        self.n_oversamples = n_oversamples
        self.power_iteration_normalizer = power_iteration_normalizer
        self.random_state = random_state
        self.tol = tol
        self.components_ = np.empty((0, 0))
        self.explained_variance_ = np.empty(0)
        self.explained_variance_ratio_ = np.empty(0)
        self.singular_values_ = np.empty(0)

    def fit(self, X, y = None):
        self.fit_transform(X)
        return self

    def fit_transform(self, X, y = None):
        Xa = np.asarray(X, dtype=float)
        k = self.n_components
        Xt = np.empty((0, 0))
        if self.algorithm == "arpack":
            if not (0 < k < min(Xa.shape[0], Xa.shape[1])):
                raise ValueError("arpack: n_components must be < min(X.shape)")
            U0, S0, Vt0 = sl.svd(Xa, full_matrices=False)
            U = np.ascontiguousarray(U0[:, :k])
            Vt = np.ascontiguousarray(Vt0[:k])
            S = S0[:k].copy()
            svd_flip_v(U, Vt)
            self.components_ = Vt
            self.singular_values_ = S
            Xt = U * S if self.tol == 0.0 else Xa @ Vt.T
        elif self.algorithm == "randomized":
            if k > Xa.shape[1]:
                raise ValueError(f"n_components({k}) must be <= n_features({Xa.shape[1]}).")
            rs = check_random_state(self.random_state)
            U, S, Vt = randomized_svd(Xa, k, rs, n_oversamples=self.n_oversamples,
                                      n_iter=self.n_iter, normalizer=self.power_iteration_normalizer,
                                      flip_sign=False)
            svd_flip_v(U, Vt)
            self.components_ = Vt
            self.singular_values_ = S
            Xt = Xa @ Vt.T
        else:
            raise ValueError(f"unknown algorithm {self.algorithm}")
        self.explained_variance_ = np.var(Xt, axis=0)
        full_var = float(np.sum(np.var(Xa, axis=0)))
        self.explained_variance_ratio_ = self.explained_variance_ / full_var
        return Xt

    def transform(self, X):
        return np.asarray(X, dtype=float) @ self.components_.T

    def inverse_transform(self, X):
        return np.asarray(X, dtype=float) @ self.components_
