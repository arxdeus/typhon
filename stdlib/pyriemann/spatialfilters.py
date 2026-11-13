# Copyright (C) 2026 the Typhon authors
# Ported from pyriemann 0.12 (BSD-3-Clause)

import numpy as np
from scipy.linalg import eigh
from .geometry.ajd import ajd
from .geometry.covariance import covariances
from .geometry.mean import gmean

def _sort_desc(v):
    return np.argsort(v)[::-1]

class Xdawn:
    nfilter: int
    estimator: str
    classes_: np.ndarray[int, 1]
    evokeds_: np.ndarray[float, 2]
    filters_: np.ndarray[float, 2]
    patterns_: np.ndarray[float, 2]
    has_baseline: bool
    baseline_cov: np.ndarray[float, 2]

    def __init__(self, nfilter: int = 4, estimator: str = "scm", baseline_cov = None):
        self.nfilter = nfilter
        self.estimator = estimator
        self.classes_ = np.empty(0, dtype=int)
        self.evokeds_ = np.empty((0, 0))
        self.filters_ = np.empty((0, 0))
        self.patterns_ = np.empty((0, 0))
        self.has_baseline = baseline_cov is not None
        self.baseline_cov = np.empty((0, 0)) if baseline_cov is None else np.asarray(baseline_cov, dtype=float)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        n_trials = X.shape[0]
        n_ch = X.shape[1]
        n_t = X.shape[2]
        self.classes_ = np.unique(y)
        if self.has_baseline:
            Cx = self.baseline_cov
        else:
            tmp = np.ascontiguousarray(X.transpose((1, 2, 0)))
            Cx = covariances(tmp.reshape(n_ch, n_t * n_trials), self.estimator)
        ev = []
        fl = []
        pt = []
        for c in self.classes_:
            P = np.mean(X[y == c], axis=0)
            C = covariances(P, self.estimator)
            evals, evecs = eigh(C, Cx)
            evecs = evecs[:, _sort_desc(evals)]
            for j in range(evecs.shape[1]):
                evecs[:, j] = evecs[:, j] / np.linalg.norm(evecs[:, j])
            A = np.linalg.pinv(evecs.T)
            k = min(self.nfilter, n_ch)
            F = evecs[:, 0:k].T
            fl.append(F)
            pt.append(A[:, 0:k].T)
            ev.append(F @ P)
        self.evokeds_ = np.concatenate(ev, axis=0)
        self.filters_ = np.concatenate(fl, axis=0)
        self.patterns_ = np.concatenate(pt, axis=0)
        return self

    def transform(self, X):
        return self.filters_ @ np.asarray(X, dtype=float)

    def fit_transform(self, X, y = None):
        return self.fit(X, np.asarray(y, dtype=int)).transform(X)

class BilinearFilter:
    filters_: np.ndarray[float, 2]
    log: bool

    def __init__(self, filters, log: bool = False):
        self.filters_ = np.asarray(filters, dtype=float)
        self.log = log

    def fit(self, X, y = None):
        return self

    def transform_cov(self, X):
        # Spatially filtered covariance matrices (log=False behavior).
        X = np.asarray(X, dtype=float)
        if X.shape[2] != self.filters_.shape[1]:
            raise ValueError("Input and filters dimension must be compatible.")
        return self.filters_ @ X @ self.filters_.T

    def transform(self, X):
        # Log-variance (log=True behavior); use transform_cov for log=False.
        Xn = self.transform_cov(X)
        out = np.zeros((Xn.shape[0], Xn.shape[1]))
        for i in range(Xn.shape[0]):
            out[i] = np.log(np.diagonal(Xn[i]))
        return out

    def fit_transform(self, X, y = None):
        return self.fit(X, y).transform(X)

class CSP:
    nfilter: int
    metric: str
    log: bool
    ajd_method: str
    filters_: np.ndarray[float, 2]
    patterns_: np.ndarray[float, 2]

    def __init__(self, nfilter: int = 4, metric: str = "euclid", log: bool = True,
                 ajd_method: str = "ajd_pham"):
        self.nfilter = nfilter
        self.metric = metric
        self.log = log
        self.ajd_method = ajd_method
        self.filters_ = np.empty((0, 0))
        self.patterns_ = np.empty((0, 0))

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        if X.ndim != 3:
            raise ValueError("X must be n_trials * n_channels * n_channels")
        if y.shape[0] != X.shape[0]:
            raise ValueError("X and y must have the same length.")
        n_ch = X.shape[1]
        classes = np.unique(y)
        nc = classes.shape[0]
        C = np.empty((nc, n_ch, n_ch))
        for i in range(nc):
            C[i] = gmean(X[y == classes[i]], metric=self.metric)
        if nc == 2:
            evals, evecs = eigh(C[1], C[0] + C[1])
            ix = _sort_desc(np.abs(evals - 0.5))
        elif nc > 2:
            V, D = ajd(C, method=self.ajd_method)
            Ctot = gmean(C, metric=self.metric)
            evecs = np.ascontiguousarray(V.T)
            for i in range(evecs.shape[1]):
                tmp = evecs[:, i] @ Ctot @ evecs[:, i]
                evecs[:, i] = evecs[:, i] / np.sqrt(tmp)
            Pc = np.empty(nc)
            for i in range(nc):
                Pc[i] = np.mean(np.asarray(y == classes[i], dtype=float))
            mi = np.empty(evecs.shape[1])
            for j in range(evecs.shape[1]):
                a = 0.0
                b = 0.0
                for i in range(nc):
                    tmp = evecs[:, j] @ C[i] @ evecs[:, j]
                    a += Pc[i] * np.log(np.sqrt(tmp))
                    b += Pc[i] * (tmp ** 2 - 1.0)
                mi[j] = -(a + (3.0 / 16.0) * (b ** 2))
            ix = _sort_desc(mi)
        else:
            raise ValueError("Number of classes must be >= 2.")
        evecs = evecs[:, ix]
        A = np.linalg.pinv(evecs.T)
        k = min(self.nfilter, evecs.shape[1])
        self.filters_ = evecs[:, 0:k].T
        self.patterns_ = A[:, 0:k].T
        return self

    def transform(self, X):
        return BilinearFilter(self.filters_, True).transform(X)

    def transform_cov(self, X):
        return BilinearFilter(self.filters_, False).transform_cov(X)

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)

class SPoC(CSP):
    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        target = np.asarray(y, dtype=float)
        target = target - np.mean(target)
        target = target / np.std(target)
        C = gmean(X, metric=self.metric)
        Ce = np.zeros_like(X)
        for i in range(Ce.shape[0]):
            Ce[i] = X[i] * target[i]
        Cz = gmean(Ce, metric=self.metric)
        evals, evecs = eigh(Cz, C)
        ix = _sort_desc(np.abs(evals))
        evecs = evecs[:, ix]
        A = np.linalg.pinv(evecs.T)
        k = min(self.nfilter, evecs.shape[1])
        self.filters_ = evecs[:, 0:k].T
        self.patterns_ = A[:, 0:k].T
        return self
