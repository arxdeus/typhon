# Copyright (C) 2026 the Typhon authors
# Ported from pyriemann 0.12 (BSD-3-Clause)

import numpy as np
from numpy.random import RandomState
from .classification import MDM
from .geometry.distance import distance

class Kmeans:
    # Random initialisation through the sklearn KMeans centroid initialiser
    # (init='random' or 'k-means++'), seeds drawn like pyriemann.
    n_clusters: int
    max_iter: int
    metric: str
    random_state: int
    init: str
    n_init: int
    tol: float
    labels_: np.ndarray[int, 1]
    inertia_: float
    mdm_: MDM

    def __init__(self, n_clusters: int = 2, max_iter: int = 100, metric: str = "riemann",
                 random_state: int = 0, init: str = "random", n_init: int = 10,
                 n_jobs: int = 1, tol: float = 1e-4):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.metric = metric
        self.random_state = random_state
        self.init = init
        self.n_init = n_init
        self.tol = tol
        self.labels_ = np.empty(0, dtype=int)
        self.inertia_ = 0.0
        self.mdm_ = MDM(metric)

    def _fit_single(self, X, seed: int):
        n = X.shape[0]
        nn = X.shape[1]
        rs = RandomState(seed)
        p = np.ones(n) / n
        pick = rs.choice(n, size=self.n_clusters, replace=False, p=p / p.sum())
        mdm = MDM(self.metric)
        mdm.classes_ = np.arange(self.n_clusters)
        mdm.covmeans_ = np.ascontiguousarray(X[pick])
        labels = mdm.predict(X)
        k = 0
        dist = mdm._predict_distances(X)
        while True:
            old = labels.copy()
            mdm.fit(X, old)
            dist = mdm._predict_distances(X)
            labels = mdm.classes_[np.argmin(dist, axis=1)]
            k += 1
            if k > self.max_iter or float(np.mean(np.asarray(labels == old, dtype=float))) > (1.0 - self.tol):
                break
        inertia = 0.0
        for i in range(mdm.classes_.shape[0]):
            for r in range(n):
                if labels[r] == mdm.classes_[i]:
                    inertia += dist[r, i]
        return labels, inertia, mdm

    def fit(self, X, y = None):
        X = np.asarray(X, dtype=float)
        np.random.seed(self.random_state)
        seeds = np.random.randint(2147483647, size=self.n_init)
        have = False
        best_in = 0.0
        for s in range(self.n_init):
            labels, inertia, mdm = self._fit_single(X, int(seeds[s]))
            if not have or inertia < best_in:
                have = True
                best_in = inertia
                self.labels_ = labels
                self.mdm_ = mdm
        self.inertia_ = best_in
        return self

    def predict(self, X):
        return self.mdm_.predict(X)

    def transform(self, X):
        return self.mdm_.transform(X)

    def fit_predict(self, X, y = None):
        return self.fit(X, y).labels_

    def centroids(self):
        return self.mdm_.covmeans_
