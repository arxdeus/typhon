# Copyright (C) 2026 the Typhon authors
# Ported from pyriemann 0.12 (BSD-3-Clause)

import numpy as np
from .classification import MDM
from .geometry.distance import distance

class FlatChannelRemover:
    channels_: np.ndarray[int, 1]

    def __init__(self):
        self.channels_ = np.empty(0, dtype=int)

    def fit(self, X, y = None):
        X = np.asarray(X, dtype=float)
        s = np.std(X, axis=2)
        std = np.mean(s * s, axis=0)
        self.channels_ = np.nonzero(std)[0]
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return X[:, self.channels_, :]

    def fit_transform(self, X, y = None):
        return self.fit(X, y).transform(X)

class ElectrodeSelection:
    nelec: int
    metric: str
    subelec_: List[int]
    dist_: List[float]
    covmeans_: np.ndarray[float, 3]

    def __init__(self, nelec: int = 16, metric: str = "riemann", n_jobs: int = 1):
        self.nelec = nelec
        self.metric = metric
        self.subelec_ = List[int]()
        self.dist_ = List[float]()
        self.covmeans_ = np.empty((0, 0, 0))

    def fit(self, X, y = None, sample_weight = None):
        X = np.asarray(X, dtype=float)
        yy = np.ones(X.shape[0], dtype=int) if y is None else np.asarray(y, dtype=int)
        mdm = MDM(self.metric)
        mdm.fit(X, yy, sample_weight=sample_weight)
        self.covmeans_ = mdm.covmeans_
        nc = self.covmeans_.shape[0]
        n_ch = self.covmeans_.shape[1]
        self.dist_ = List[float]()
        self.subelec_ = [i for i in range(n_ch)]
        while len(self.subelec_) > self.nelec:
            m = len(self.subelec_)
            di = np.zeros(m)
            for idx in range(m):
                sub = [self.subelec_[t] for t in range(m) if t != idx]
                for i in range(nc):
                    for j in range(i + 1, nc):
                        A = self.covmeans_[i][sub][:, sub]
                        B = self.covmeans_[j][sub][:, sub]
                        di[idx] += float(distance(A, B, metric=self.metric))
            torm = int(np.argmax(di))
            self.dist_.append(float(np.max(di)))
            self.subelec_.pop(torm)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        out = X[:, self.subelec_, :]
        return out[:, :, self.subelec_]

    def fit_transform(self, X, y = None, sample_weight = None):
        return self.fit(X, y, sample_weight=sample_weight).transform(X)
