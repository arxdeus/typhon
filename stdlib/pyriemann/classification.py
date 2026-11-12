# Ported from pyriemann 0.12 (BSD-3-Clause)

import numpy as np
from .geometry.mean import gmean
from .geometry.distance import distance

def _softmax(X):
    # sklearn.utils.extmath.softmax
    m = np.max(X, axis=1, keepdims=True)
    E = np.exp(X - m)
    return E / np.sum(E, axis=1, keepdims=True)

class MDM:
    # Labels are integers (cast with int()).
    metric: str
    n_jobs: int
    classes_: np.ndarray[int, 1]
    covmeans_: np.ndarray[float, 3]

    def __init__(self, metric: str = "riemann", n_jobs: int = 1):
        self.metric = metric
        self.n_jobs = n_jobs
        self.classes_ = np.empty(0, dtype=int)
        self.covmeans_ = np.empty((0, 0, 0))

    def fit(self, X, y, sample_weight = None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        self.classes_ = np.unique(y)
        sw = np.ones(X.shape[0]) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        nc = self.classes_.shape[0]
        n = X.shape[1]
        self.covmeans_ = np.empty((nc, n, n))
        for k in range(nc):
            mask = y == self.classes_[k]
            self.covmeans_[k] = gmean(X[mask], metric=self.metric,
                                      sample_weight=sw[mask])
        return self

    def _predict_distances(self, X):
        X = np.asarray(X, dtype=float)
        nc = self.classes_.shape[0]
        dist = np.empty((X.shape[0], nc))
        for k in range(nc):
            dist[:, k] = distance(X, self.covmeans_[k], metric=self.metric)[:, 0]
        return dist

    def predict(self, X):
        dist = self._predict_distances(X)
        return self.classes_[np.argmin(dist, axis=1)]

    def transform(self, X):
        return self._predict_distances(X)

    def fit_transform(self, X, y, sample_weight = None):
        return self.fit(X, y, sample_weight=sample_weight).transform(X)

    def fit_predict(self, X, y):
        return self.fit(X, y).predict(X)

    def predict_proba(self, X):
        d = self._predict_distances(X)
        return _softmax(-(d * d))

    def score(self, X, y, sample_weight = None):
        pred = self.predict(X)
        y = np.asarray(y, dtype=int)
        correct = np.asarray(pred == y, dtype=float)
        if sample_weight is None:
            return float(np.mean(correct))
        else:
            w = np.asarray(sample_weight, dtype=float)
            return float(np.sum(correct * w) / np.sum(w))
