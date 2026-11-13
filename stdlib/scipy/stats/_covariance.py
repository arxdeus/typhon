# Copyright (C) 2026 the Typhon authors
# Ported from scipy 1.18.1 (BSD-3-Clause)

# scipy.stats.Covariance for a single (non-batched) matrix. One concrete
# class holds the representation (precision, diagonal, Cholesky or
# eigendecomposition); the subclass hierarchy of scipy is flattened into a
# ``kind`` tag.

import numpy as np
import scipy.special._kernels as K
import scipy.linalg as LA

def _square(A, name: str) -> np.ndarray[float, 2]:
    a = np.asarray(A).astype(float)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"The input `{name}` must be a square, two-dimensional array of real numbers.")
    return a

def _vector(A, name: str) -> np.ndarray[float, 1]:
    a = np.asarray(A).astype(float)
    if a.ndim != 1:
        raise ValueError(f"The input `{name}` must be a one-dimensional array of real numbers.")
    return a

class Covariance:
    kind: int   # 0 precision, 1 diagonal, 2 cholesky, 3 eigendecomposition
    _chol_P: np.ndarray[float, 2]
    _factor: np.ndarray[float, 2]
    _diag: np.ndarray[float, 1]
    _w: np.ndarray[float, 1]
    _v: np.ndarray[float, 2]
    _LP: np.ndarray[float, 1]
    _cov_matrix: np.ndarray[float, 2]
    _has_cov: bool
    _log_pdet: float
    _rank: int
    _n: int
    _allow_singular: bool

    def __init__(self, kind: int, n: int):
        self.kind = kind
        self._n = n
        self._chol_P = np.zeros((0, 0))
        self._factor = np.zeros((0, 0))
        self._diag = np.zeros(0)
        self._w = np.zeros(0)
        self._v = np.zeros((0, 0))
        self._LP = np.zeros(0)
        self._cov_matrix = np.zeros((0, 0))
        self._has_cov = False
        self._log_pdet = 0.0
        self._rank = n
        self._allow_singular = kind == 1 or kind == 3

    @staticmethod
    def from_diagonal(diagonal):
        d = _vector(diagonal, 'diagonal')
        n = d.shape[0]
        c = Covariance(1, n)
        c._diag = d
        lp = np.zeros(n)
        s = 0.0
        zero = 0
        for i in range(n):
            if d[i] <= 0:
                zero += 1
            s += K.log(d[i]) if d[i] > 0 else (0.0 if False else K.log(d[i]) if d[i] == d[i] else d[i])
            lp[i] = 1 / K.sqrt(d[i]) if d[i] > 0 else (K.INF if d[i] == 0 else K.NAN)
        c._log_pdet = s
        c._LP = lp
        c._rank = n - zero
        return c

    @staticmethod
    def from_precision(precision, covariance = None):
        P = _square(precision, 'precision')
        c = Covariance(0, P.shape[0])
        if covariance is not None:
            C = _square(covariance, 'covariance')
            if C.shape != P.shape:
                raise ValueError("`precision.shape` must equal `covariance.shape`.")
            c._cov_matrix = C
            c._has_cov = True
        c._chol_P = np.linalg.cholesky(P)
        s = 0.0
        for i in range(P.shape[0]):
            s += K.log(c._chol_P[i, i])
        c._log_pdet = -2 * s
        return c

    @staticmethod
    def from_cholesky(cholesky):
        L = _square(cholesky, 'cholesky')
        c = Covariance(2, L.shape[0])
        c._factor = L
        s = 0.0
        for i in range(L.shape[0]):
            s += K.log(L[i, i])
        c._log_pdet = 2 * s
        return c

    @staticmethod
    def from_eigendecomposition(eigendecomposition):
        w = _vector(eigendecomposition[0], 'eigenvalues')
        v = _square(eigendecomposition[1], 'eigenvectors')
        if w.shape[0] != v.shape[0]:
            raise ValueError("The shapes of `eigenvalues` and `eigenvectors` must be compatible.")
        n = w.shape[0]
        c = Covariance(3, n)
        c._w = w
        c._v = v
        zero = 0
        s = 0.0
        for i in range(n):
            if w[i] <= 0:
                zero += 1
            s += K.log(w[i])
        c._log_pdet = s
        c._rank = n - zero
        return c

    def _whiten_mat(self) -> np.ndarray[float, 2]:
        # matrix W such that whiten(x) = x @ W
        n = self._n
        if self.kind == 0:
            return self._chol_P
        elif self.kind == 1:
            return np.diag(self._LP)
        elif self.kind == 2:
            return LA.solve_triangular(self._factor, np.eye(n), lower=True).T
        else:
            m = np.zeros((n, n))
            for j in range(n):
                f = 1 / K.sqrt(self._w[j]) if self._w[j] > 0 else (K.INF if self._w[j] == 0 else K.NAN)
                for i in range(n):
                    m[i, j] = self._v[i, j] * f
            return m

    def _colorize_mat(self) -> np.ndarray[float, 2]:
        # matrix C such that colorize(x) = x @ C
        n = self._n
        if self.kind == 0:
            return LA.solve_triangular(self._chol_P.T, np.eye(n), lower=False).T
        elif self.kind == 1:
            return np.diag(np.sqrt(self._diag))
        elif self.kind == 2:
            return self._factor.T
        else:
            m = np.zeros((n, n))
            for j in range(n):
                f = K.sqrt(self._w[j])
                for i in range(n):
                    m[i, j] = self._v[i, j] * f
            return m.T

    def whiten(self, x):
        xa = np.asarray(x).astype(float)
        return xa @ self._whiten_mat()

    def colorize(self, x):
        xa = np.asarray(x).astype(float)
        return xa @ self._colorize_mat()

    def log_pdet(self) -> float:
        return self._log_pdet

    def rank(self) -> int:
        return self._rank

    def covariance(self) -> np.ndarray[float, 2]:
        n = self._n
        if self.kind == 0:
            if self._has_cov:
                return self._cov_matrix
            return LA.cho_solve((self._chol_P, True), np.eye(n))
        elif self.kind == 1:
            return np.diag(self._diag)
        elif self.kind == 2:
            return self._factor @ self._factor.T
        else:
            return (self._v * self._w) @ self._v.T

    def shape(self):
        return (self._n, self._n)
