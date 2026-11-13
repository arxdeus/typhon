# Ported from pyriemann 0.12 (BSD-3-Clause)

import numpy as np
from .geometry.mean import gmean
from .geometry.tangentspace import tangent_space, untangent_space

class TangentSpace:
    metric: str
    tsupdate: bool
    reference_: np.ndarray[float, 2]
    _fitted: bool

    def __init__(self, metric: str = "riemann", tsupdate: bool = False):
        self.metric = metric
        self.tsupdate = tsupdate
        self._fitted = False
        self.reference_ = np.empty((0, 0))

    def fit(self, X, y = None, sample_weight = None):
        self.reference_ = gmean(np.asarray(X, dtype=float), metric=self.metric,
                                sample_weight=sample_weight)
        self._fitted = True
        return self

    def _check_data_dim(self, X) -> int:
        if X.ndim == 2:
            m = X.shape[1]
            nc = ((1 + 8 * m) ** 0.5 - 1) / 2
            if nc != float(int(nc)):
                raise ValueError("Shape of tangent space vector does not "
                                 "correspond to a square matrix.")
            return int(nc)
        elif X.ndim == 3:
            if X.shape[1] != X.shape[2]:
                raise ValueError("Matrices must be square")
            return X.shape[1]
        else:
            compile_error("Shape must be of len 2 or 3.")

    def _check_reference_points(self, X):
        n = self._check_data_dim(X)
        if not self._fitted:
            self.reference_ = np.eye(n)
            self._fitted = True
        elif self.reference_.shape[0] != n:
            raise ValueError("Data must be same size of reference matrix.")

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        self._check_reference_points(X)
        if self.tsupdate:
            Cr = gmean(X, metric=self.metric)
            return tangent_space(X, Cr, metric=self.metric)
        else:
            return tangent_space(X, self.reference_, metric=self.metric)

    def fit_transform(self, X, y = None, sample_weight = None):
        self.fit(X, y, sample_weight=sample_weight)
        return tangent_space(np.asarray(X, dtype=float), self.reference_,
                             metric=self.metric)

    def inverse_transform(self, X):
        X = np.asarray(X, dtype=float)
        self._check_reference_points(X)
        return untangent_space(X, self.reference_, metric=self.metric)

class FGDA:
    metric: str
    tsupdate: bool
    _ts: TangentSpace
    _W: np.ndarray[float, 2]

    def __init__(self, metric: str = "riemann", tsupdate: bool = False):
        self.metric = metric
        self.tsupdate = tsupdate
        self._ts = TangentSpace(metric, tsupdate)
        self._W = np.empty((0, 0))

    def _fit_lda(self, X, y, sample_weight = None):
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        y = np.asarray(y, dtype=int)
        classes = np.unique(y)
        lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto",
                                         n_components=classes.shape[0] - 1)
        ts = self._ts.fit_transform(X, sample_weight=sample_weight)
        lda.fit(ts, y)
        W = lda.coef_.copy()
        self._W = W.T @ np.linalg.pinv(W @ W.T) @ W
        return ts

    def _retro_project(self, ts):
        return self._ts.inverse_transform(ts @ self._W)

    def fit(self, X, y = None, sample_weight = None):
        self._ts = TangentSpace(self.metric, self.tsupdate)
        self._fit_lda(X, y, sample_weight=sample_weight)
        return self

    def transform(self, X):
        return self._retro_project(self._ts.transform(X))

    def fit_transform(self, X, y = None, sample_weight = None):
        self._ts = TangentSpace(self.metric, self.tsupdate)
        ts = self._fit_lda(X, y, sample_weight=sample_weight)
        return self._retro_project(ts)
