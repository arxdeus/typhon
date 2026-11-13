# Copyright (C) 2026 the Typhon authors
# Ported from pyriemann 0.12 (BSD-3-Clause)

import numpy as np
from numpy.random import RandomState
from .classification import MDM
from .geometry.distance import distance, pairwise_distance, distance_mahalanobis
from .geometry.tangentspace import log_map, exp_map, tangent_space
from .geometry.mean import gmean, mean_riemann, check_weights
from .geometry.covariance import covariance_scm

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


class MeanShift:
    kernel: str
    bandwidth: float
    has_bandwidth: bool
    metric: str
    tol: float
    max_iter: int
    modes_: np.ndarray[float, 3]
    labels_: np.ndarray[int, 1]
    _bw: float

    def __init__(self, kernel: str = "uniform", bandwidth = None, metric: str = "riemann",
                 tol: float = 1e-3, max_iter: int = 100, n_jobs: int = 1):
        self.kernel = kernel
        self.has_bandwidth = bandwidth is not None
        self.bandwidth = 0.0 if bandwidth is None else float(bandwidth)
        self.metric = metric
        self.tol = tol
        self.max_iter = max_iter
        self.modes_ = np.empty((0, 0, 0))
        self.labels_ = np.empty(0, dtype=int)
        self._bw = 0.0

    def _kern(self, d2):
        x = d2
        if self.kernel == "normal":
            return np.exp(-(x ** 2))
        elif self.kernel == "uniform":
            return np.asarray(np.abs(x) <= 1.0, dtype=float)
        raise ValueError("Unknown kernel")

    def _estimate_bandwidth(self, X, quantile: float):
        dist = pairwise_distance(X, None, metric=self.metric)
        n = dist.shape[0]
        vals = List[float]()
        for i in range(n):
            for j in range(i + 1, n):
                if dist[i, j] > 0:
                    vals.append(dist[i, j])
        arr = np.sort(np.array(vals))
        return float(arr[int(np.floor(quantile * arr.shape[0]))])

    def _seek_mode(self, X, mean0):
        mean = mean0.copy()
        bw2 = self._bw * self._bw
        for _ in range(self.max_iter):
            T = log_map(X, mean, metric=self.metric)
            dist2 = distance(X, mean, metric=self.metric, squared=True)
            weights = self._kern(dist2[:, 0] / bw2)
            ms = np.zeros(mean.shape)
            for i in range(X.shape[0]):
                ms = ms + weights[i] * T[i]
            ms = ms / np.sum(weights)
            mean = exp_map(ms, mean, metric=self.metric)
            if np.linalg.norm(ms) <= self.tol:
                break
        return mean

    def fit(self, X, y = None):
        X = np.asarray(X, dtype=float)
        if self.has_bandwidth:
            self._bw = self.bandwidth
        else:
            self._bw = self._estimate_bandwidth(X, 0.3)
        n = X.shape[0]
        modes = List[np.ndarray[float, 2]]()
        for i in range(n):
            modes.append(self._seek_mode(X, X[i]))
        out = List[np.ndarray[float, 2]]()
        for m in modes:
            out.append(m)
        inm = np.stack(modes, axis=0)
        dist = pairwise_distance(inm, None, metric=self.metric)
        # Row i is removed when a still-alive mode other than itself is within
        # the bandwidth (columns of removed modes are ignored afterwards).
        dead = np.zeros(n, dtype=bool)
        i = n - 1
        while i >= 0:
            close = False
            for j in range(n):
                if j != i and not dead[j] and dist[i, j] < self._bw:
                    close = True
            if close:
                out.pop(i)
                dead[i] = True
            i -= 1
        if len(out) == 0:
            raise ValueError("No mode found, try other parameters")
        self.modes_ = np.stack(out, axis=0)
        self.labels_ = self.predict(X)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        nm = self.modes_.shape[0]
        dist = np.empty((X.shape[0], nm))
        for k in range(nm):
            dist[:, k] = distance(X, self.modes_[k], metric=self.metric)[:, 0]
        return np.argmin(dist, axis=1)

    def fit_predict(self, X, y = None):
        return self.fit(X).labels_

class Gaussian:
    n: int
    mu: np.ndarray[float, 2]
    sigma: np.ndarray[float, 2]
    metric: str

    def __init__(self, n: int, mu, sigma = None, metric: str = "riemann"):
        self.n = n
        self.mu = np.asarray(mu, dtype=float)
        if sigma is None:
            self.sigma = np.eye(n * (n + 1) // 2)
        else:
            self.sigma = np.asarray(sigma, dtype=float)
        self.metric = metric

    def pdf(self, X, reg: float = 1e-16, use_pi: bool = True):
        tv = tangent_space(X, self.mu, metric=self.metric)
        dist = distance_mahalanobis(tv.T, self.sigma, squared=True)
        num = np.exp(-0.5 * dist)
        det = np.linalg.det(self.sigma)
        if use_pi:
            denom = np.sqrt(((2.0 * np.pi) ** self.n) * det)
        else:
            denom = np.sqrt(det)
        return num / (denom + reg)

    def update_mean(self, X, sample_weight):
        if self.metric == "riemann":
            self.mu = mean_riemann(X, init=self.mu, sample_weight=sample_weight)
        else:
            self.mu = gmean(X, metric=self.metric, sample_weight=sample_weight)

    def update_covariance(self, X, sample_weight):
        tv = tangent_space(X, self.mu, metric=self.metric)
        self.sigma = covariance_scm(tv.T, assume_centered=True, weights=sample_weight)

class GaussianMixture:
    n_components: int
    metric: str
    tol: float
    maxiter: int
    random_state: int
    weights_: np.ndarray[float, 1]
    _weights_init: np.ndarray[float, 1]
    _means_init: np.ndarray[float, 3]
    _components: List[Gaussian]

    def __init__(self, n_components: int = 1, metric: str = "riemann", weights_init = None,
                 means_init = None, tol: float = 1e-5, maxiter: int = 100,
                 random_state: int = 0, verbose: bool = False):
        self.n_components = n_components
        self.metric = metric
        self.tol = tol
        self.maxiter = maxiter
        self.random_state = random_state
        self._weights_init = np.empty(0) if weights_init is None else np.asarray(weights_init, dtype=float)
        self._means_init = np.empty((0, 0, 0)) if means_init is None else np.asarray(means_init, dtype=float)
        self.weights_ = np.empty(0)
        self._components = List[Gaussian]()

    @property
    def means_(self):
        return np.stack([c.mu for c in self._components], axis=0)

    @property
    def covariances_(self):
        return np.stack([c.sigma for c in self._components], axis=0)

    def _get_wlik(self, X, use_pi: bool = True):
        wlik = np.zeros((X.shape[0], self.n_components))
        for k in range(self.n_components):
            lik = self._components[k].pdf(X, use_pi=use_pi)
            wlik[:, k] = self.weights_[k] * lik
        return wlik

    def _get_proba(self, X, reg: float = 1e-16):
        num = self._get_wlik(X, use_pi=False)
        return num / (np.sum(num, axis=1, keepdims=True) + reg)

    def _log(self, X):
        return np.log(np.clip(X, 1e-10, 1e300))

    def fit(self, X, y = None):
        X = np.asarray(X, dtype=float)
        nm = X.shape[0]
        nc = X.shape[1]
        if nc * (nc + 1) // 2 > nm:
            raise ValueError("Not enough matrices for training GMM.")
        rs = RandomState(self.random_state)
        if self._means_init.shape == (self.n_components, nc, nc):
            means_init = self._means_init
        else:
            inds = rs.randint(nm, size=(self.n_components,))
            means_init = X[inds]
        self._components = List[Gaussian]()
        for k in range(self.n_components):
            self._components.append(Gaussian(nc, means_init[k], None, self.metric))
        if self._weights_init.shape[0] == 0:
            self.weights_ = check_weights(None, self.n_components)
        else:
            self.weights_ = check_weights(self._weights_init, self.n_components)
        crit = 0.0
        for _ in range(self.maxiter):
            prob = self._get_proba(X)
            self.weights_ = np.sum(prob, axis=0) / nm
            self.weights_ = self.weights_ / np.sum(self.weights_)
            for k in range(self.n_components):
                self._components[k].update_mean(X, prob[:, k])
                self._components[k].update_covariance(X, prob[:, k])
            crit_new = -float(np.sum(self._log(np.sum(self._get_wlik(X), axis=1))))
            if abs(crit - crit_new) < self.tol:
                break
            crit = crit_new
        return self

    def predict_proba(self, X):
        return self._get_proba(np.asarray(X, dtype=float))

    def predict(self, X):
        return np.argmax(self._get_proba(np.asarray(X, dtype=float)), axis=1)

    def fit_predict(self, X, y = None):
        return self.fit(X).predict(X)

    def score(self, X, y = None):
        lik = np.sum(self._get_wlik(np.asarray(X, dtype=float)), axis=1)
        return float(np.mean(self._log(lik)))

from .artifact_detection import Potato, PotatoField
