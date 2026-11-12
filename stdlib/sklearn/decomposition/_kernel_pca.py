# Ported from scikit-learn 1.9.1 (BSD-3-Clause)

import numpy as np
import scipy.linalg as sl
from ._extmath import pairwise_kernel, svd_flip_u_only

class KernelPCA:
    """kernels linear/poly/rbf/sigmoid/cosine/precomputed, dense eigen
    solver, optional fit_inverse_transform (kernel ridge pre-image)."""
    n_components: Optional[int]
    kernel: str
    gamma: Optional[float]
    degree: float
    coef0: float
    alpha: float
    fit_inverse_transform: bool
    remove_zero_eig: bool
    gamma_: float
    eigenvalues_: np.ndarray[float, 1]
    eigenvectors_: np.ndarray[float, 2]
    X_fit_: np.ndarray[float, 2]
    K_fit_rows_: np.ndarray[float, 1]
    K_fit_all_: float
    dual_coef_: np.ndarray[float, 2]
    X_transformed_fit_: np.ndarray[float, 2]

    def __init__(self, n_components: Optional[int] = None, *, kernel: str = "linear",
                 gamma: Optional[float] = None, degree: float = 3, coef0: float = 1,
                 kernel_params = None, alpha: float = 1.0, fit_inverse_transform: bool = False,
                 eigen_solver: str = "auto", tol: float = 0, max_iter = None,
                 iterated_power = "auto", remove_zero_eig: bool = False,
                 random_state = None, copy_X: bool = True, n_jobs = None):
        if fit_inverse_transform and kernel == "precomputed":
            raise ValueError("Cannot fit_inverse_transform with a precomputed kernel.")
        self.n_components = n_components
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.alpha = alpha
        self.fit_inverse_transform = fit_inverse_transform
        self.remove_zero_eig = remove_zero_eig
        self.gamma_ = 0.0
        self.eigenvalues_ = np.empty(0)
        self.eigenvectors_ = np.empty((0, 0))
        self.X_fit_ = np.empty((0, 0))
        self.K_fit_rows_ = np.empty(0)
        self.K_fit_all_ = 0.0
        self.dual_coef_ = np.empty((0, 0))
        self.X_transformed_fit_ = np.empty((0, 0))

    def _get_kernel(self, X, Y, same: bool = False):
        if self.kernel == "precomputed":
            return X.copy()
        return pairwise_kernel(X, Y, self.kernel, self.gamma_, self.degree, self.coef0, same)

    def _center(self, K):
        pred_cols = np.sum(K, axis=1) / float(self.K_fit_rows_.shape[0])
        K -= self.K_fit_rows_[None, :]
        K -= pred_cols[:, None]
        K += self.K_fit_all_
        return K

    def fit(self, X, y = None):
        Xa = np.array(X, dtype=float)
        self.gamma_ = 1.0 / Xa.shape[1] if self.gamma is None else self.gamma.__val__()
        K = self._get_kernel(Xa, Xa, True)
        n = K.shape[0]
        self.K_fit_rows_ = np.sum(K, axis=0) / float(n)
        self.K_fit_all_ = float(self.K_fit_rows_.sum()) / float(n)
        K = self._center(K)
        k = n if self.n_components is None else min(n, self.n_components.__val__())
        w, V = sl.eigh(K, subset_by_index=[n - k, n - 1])
        w = np.array(w)
        V = np.ascontiguousarray(V)
        mx = float(w.max())
        if mx < 0:
            raise ValueError("All eigenvalues are negative")
        mn = float(w.min())
        if mn < -1e-5 * mx and mn < -1e-10:
            raise ValueError("There are significant negative eigenvalues")
        for i in range(k):
            if w[i] < 0:
                w[i] = 0.0
            if 0 < w[i] < 1e-12 * mx:
                w[i] = 0.0
        svd_flip_u_only(V)
        idx = np.argsort(w, kind="stable")[::-1]
        w = w[idx]
        V = np.ascontiguousarray(V[:, idx])
        if self.remove_zero_eig or self.n_components is None:
            keep = w > 0
            V = np.ascontiguousarray(V[:, keep])
            w = w[keep]
        self.eigenvalues_ = w
        self.eigenvectors_ = V
        if self.fit_inverse_transform:
            self._fit_inverse(V * np.sqrt(w), Xa)
        self.X_fit_ = Xa
        return self

    def _fit_inverse(self, Xt, X):
        n = Xt.shape[0]
        K = self._get_kernel(Xt, Xt, True)
        for i in range(n):
            K[i, i] += self.alpha
        self.dual_coef_ = sl.solve(K, X, assume_a='pos')
        self.X_transformed_fit_ = Xt

    def fit_transform(self, X, y = None):
        self.fit(X)
        return self.eigenvectors_ * np.sqrt(self.eigenvalues_)

    def transform(self, X):
        Xa = np.asarray(X, dtype=float)
        K = self._center(self._get_kernel(Xa, self.X_fit_))
        sa = np.zeros_like(self.eigenvectors_)
        for j in range(self.eigenvalues_.shape[0]):
            if self.eigenvalues_[j] != 0:
                sa[:, j] = self.eigenvectors_[:, j] / np.sqrt(self.eigenvalues_[j])
        return K @ sa

    def inverse_transform(self, X):
        if not self.fit_inverse_transform:
            raise ValueError("The fit_inverse_transform parameter was not set to True")
        K = self._get_kernel(np.asarray(X, dtype=float), self.X_transformed_fit_)
        return K @ self.dual_coef_
