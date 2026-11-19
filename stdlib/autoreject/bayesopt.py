# Copyright (C) 2026 the Typhon authors
# Ported from autoreject 0.5.2 (BSD-3-Clause)

"""Bayesian optimization (Brochu, Cora and de Freitas tutorial).

The ``debug`` plotting branch of ``bayes_opt`` is a no-op in Typhon.
"""

import numpy as np
from scipy.linalg import cho_solve, cholesky, solve_triangular
from scipy.special import ndtr


class _GaussianProcess:
    """1D Gaussian process regression with the fixed kernel ``1.0 * RBF(1.0)``
    and ``alpha=1e-10`` (same as sklearn GaussianProcessRegressor(optimizer=None))."""
    alpha: float
    X_train_: np.ndarray[float, 2]
    L_: np.ndarray[float, 2]
    alpha_: np.ndarray[float, 1]

    def __init__(self):
        self.alpha = 1e-10
        self.X_train_ = np.empty((0, 1))
        self.L_ = np.empty((0, 0))
        self.alpha_ = np.empty(0)

    @staticmethod
    def _kernel(a: np.ndarray[float, 2], b: np.ndarray[float, 2]) -> np.ndarray[float, 2]:
        return np.exp(-0.5 * (a - b.T) ** 2)

    def fit(self, X: np.ndarray[float, 2], y: np.ndarray[float, 1]):
        self.X_train_ = X
        K = _GaussianProcess._kernel(X, X)
        for i in range(K.shape[0]):
            K[i, i] += self.alpha
        self.L_ = cholesky(K, lower=True, check_finite=False)
        self.alpha_ = cho_solve((self.L_, True), y, check_finite=False)
        return self

    def predict(self, X: np.ndarray[float, 2]) -> np.ndarray[float, 1]:
        K_trans = _GaussianProcess._kernel(X, self.X_train_)
        return K_trans @ self.alpha_

    def predict_std(self, X: np.ndarray[float, 2]) -> Tuple[np.ndarray[float, 1], np.ndarray[float, 1]]:
        """predict(X, return_std=True)."""
        K_trans = _GaussianProcess._kernel(X, self.X_train_)
        y_mean = K_trans @ self.alpha_
        V = solve_triangular(self.L_, K_trans.T, lower=True, check_finite=False)
        y_var = np.clip(1. - np.einsum('ij,ij->j', V, V), 0., None)
        return y_mean, np.sqrt(y_var)


def expected_improvement(gp: _GaussianProcess, best_y: float, x: np.ndarray[float, 1]) -> np.ndarray[float, 1]:
    """The expected improvement acquisition function (Eq. 3 of the tutorial)."""
    y, y_std = gp.predict_std(x[:, None])
    Z = (y - best_y) / (y_std + 1e-12)
    pdf = np.exp(-Z ** 2 / 2.0) / np.sqrt(2 * np.pi)
    return (y - best_y) * ndtr(Z) + y_std * pdf


def bayes_opt(f, initial_x, all_x, acquisition, max_iter: int = 100,
              debug: bool = False) -> Tuple[float, float]:
    """Minimize f over all_x starting from initial_x. Returns (best_x, best_f)."""
    X = List[float]()
    y = List[float]()
    for x in initial_x:
        if not np.isinf(f(x)):
            y.append(f(x))
            X.append(float(x))

    k = int(np.argmin(np.array(y)))
    best_x = X[k]
    best_f = y[k]
    gp = _GaussianProcess()
    ax = np.asarray(all_x, dtype=float)

    for i in range(max_iter):
        gp.fit(np.array(X)[:, None], np.array(y))
        new_x = float(ax[int(acquisition(gp, best_f, ax).argmin())])
        new_f = f(new_x)
        if not np.isinf(new_f):
            X.append(new_x)
            y.append(new_f)
            if new_f < best_f:
                best_f = new_f
                best_x = new_x
        if debug:
            print("iter", i, "best_x", best_x, best_f)
    return best_x, best_f
