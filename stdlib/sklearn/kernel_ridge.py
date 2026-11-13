# Ported from scikit-learn 1.9.1 (BSD-3-Clause)

# KernelRidge: dual coefficients by a Cholesky solve of (K + alpha I) as
# linear_model._ridge._solve_cholesky_kernel.

import numpy as np
import scipy.linalg as sl
from ._pairwise_k import kernel_matrix

class KernelRidge[D: Literal[int] = 1]:
    alpha: List[float]
    kernel: str
    gamma: Optional[float]
    degree: float
    coef0: float
    _dual: np.ndarray[float, 2]
    X_fit_: np.ndarray[float, 2]
    n_features_in_: int

    def __init__(self, alpha = 1.0, *, kernel: str = "linear", gamma: Optional[float] = None,
                 degree: float = 3.0, coef0: float = 1.0, kernel_params = None):
        if isinstance(alpha, float) or isinstance(alpha, int):
            self.alpha = [float(alpha)]
        else:
            self.alpha = [float(a) for a in alpha]
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self._dual = np.empty((0, 0))
        self.X_fit_ = np.empty((0, 0))
        self.n_features_in_ = 0

    def _get_kernel(self, X, Y, same: bool = False):
        if self.kernel == "precomputed":
            return np.asarray(X, dtype=float).copy()
        return kernel_matrix(X, Y, self.kernel, self.gamma, self.degree, self.coef0, same)

    def fit(self, X, y, sample_weight = None):
        ya = np.asarray(y, dtype=float)
        if static.len(ya.shape) != D:
            from .linear_model._base import _retype
            return _retype(self, KernelRidge[static.len(ya.shape)]).fit(X, ya, sample_weight)
        Xa = np.ascontiguousarray(np.asarray(X, dtype=float))
        K = self._get_kernel(Xa, Xa, True)
        Y = ya.reshape(ya.shape[0], -1).copy()
        n = K.shape[0]
        k = Y.shape[1]
        has_sw = not isinstance(sample_weight, NoneType)
        sw = np.ones(n)
        if not isinstance(sample_weight, NoneType):
            sw = np.sqrt(np.asarray(sample_weight, dtype=float) * np.ones(n))
            for i in range(n):
                for t in range(k):
                    Y[i, t] *= sw[i]
            K *= np.outer(sw, sw)
        one_alpha = True
        for a in self.alpha:
            if a != self.alpha[0]:
                one_alpha = False
        dual = np.empty((n, k))
        if one_alpha:
            for i in range(n):
                K[i, i] += self.alpha[0]
            dual = sl.solve(K, Y, assume_a="pos")
            if has_sw:
                for i in range(n):
                    for t in range(k):
                        dual[i, t] *= sw[i]
        else:
            if len(self.alpha) != k:
                raise ValueError("Number of targets and number of penalties do not correspond")
            for t in range(k):
                Kc = K.copy()
                for i in range(n):
                    Kc[i, i] += self.alpha[t]
                d = sl.solve(Kc, np.ascontiguousarray(Y[:, t]), assume_a="pos")
                for i in range(n):
                    dual[i, t] = d[i] * (sw[i] if has_sw else 1.0)
        self._dual = dual
        self.X_fit_ = Xa
        self.n_features_in_ = Xa.shape[1]
        return self

    @property
    def dual_coef_(self):
        if D == 1:
            return np.ascontiguousarray(self._dual[:, 0])
        else:
            return self._dual

    def predict(self, X):
        K = self._get_kernel(X, self.X_fit_)
        return np.dot(K, self.dual_coef_)

    def score(self, X, y, sample_weight = None) -> float:
        from .metrics import r2_score
        return r2_score(y, self.predict(X))
