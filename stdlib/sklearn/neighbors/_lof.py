# Ported from scikit-learn 1.9.1 (BSD-3-Clause)

# sklearn.neighbors.LocalOutlierFactor

import numpy as np
from ._unsupervised import NearestNeighbors

class LocalOutlierFactor:
    n_neighbors: int
    algorithm: str
    leaf_size: int
    metric: str
    p: float
    contamination_auto: bool
    contamination_value: float
    novelty: bool
    _nn: NearestNeighbors
    n_neighbors_: int
    negative_outlier_factor_: np.ndarray[float, 1]
    offset_: float
    n_samples_fit_: int
    n_features_in_: int
    effective_metric_: str
    _distances_fit_X_: np.ndarray[float, 2]
    _lrd: np.ndarray[float, 1]
    _fit_X: np.ndarray[float, 2]

    def __init__(self, n_neighbors: int = 20, *, algorithm: str = "auto", leaf_size: int = 30,
                 metric: str = "minkowski", p: float = 2.0, metric_params = None,
                 contamination = "auto", novelty: bool = False, n_jobs = None):
        self.n_neighbors = n_neighbors
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.metric = metric
        self.p = float(p)
        if isinstance(contamination, str):
            if contamination != "auto":
                raise ValueError("contamination must be 'auto' or a float in (0, 0.5]")
            self.contamination_auto = True
            self.contamination_value = 0.0
        else:
            c = float(contamination)
            if not (0.0 < c <= 0.5):
                raise ValueError("contamination must be in (0, 0.5]")
            self.contamination_auto = False
            self.contamination_value = c
        self.novelty = novelty
        self._nn = NearestNeighbors(n_neighbors=n_neighbors, algorithm=algorithm,
                                    leaf_size=leaf_size, metric=metric, p=float(p))
        self.n_neighbors_ = 0
        self.negative_outlier_factor_ = np.empty(0)
        self.offset_ = 0.0
        self.n_samples_fit_ = 0
        self.n_features_in_ = 0
        self.effective_metric_ = metric
        self._distances_fit_X_ = np.empty((0, 0))
        self._lrd = np.empty(0)
        self._fit_X = np.empty((0, 0))

    @property
    def contamination(self):
        if self.contamination_auto:
            return "auto"
        return str(self.contamination_value)

    def _lrd_of(self, dist_X: np.ndarray[float, 2], ind: np.ndarray[int, 2]) -> np.ndarray[float, 1]:
        nq, k = dist_X.shape
        out = np.empty(nq)
        for i in range(nq):
            s = 0.0
            for j in range(k):
                dk = self._distances_fit_X_[ind[i, j], self.n_neighbors_ - 1]
                s += max(dist_X[i, j], dk)
            out[i] = 1.0 / (s / float(k) + 1e-10)
        return out

    def fit(self, X, y = None):
        self._nn = NearestNeighbors(n_neighbors=self.n_neighbors, algorithm=self.algorithm,
                                    leaf_size=self.leaf_size, metric=self.metric, p=self.p).fit(X)
        n = self._nn.n_samples_fit_
        self.n_samples_fit_ = n
        self.n_features_in_ = self._nn.n_features_in_
        self.effective_metric_ = self._nn.effective_metric_
        self._fit_X = self._nn._fit_X
        self.n_neighbors_ = max(1, min(self.n_neighbors, n - 1))
        dist, ind = self._nn.kneighbors(None, self.n_neighbors_, True)
        self._distances_fit_X_ = dist
        self._lrd = self._lrd_of(dist, ind)
        k = self.n_neighbors_
        nof = np.empty(n)
        for i in range(n):
            s = 0.0
            for j in range(k):
                s += self._lrd[ind[i, j]] / self._lrd[i]
            nof[i] = -(s / float(k))
        self.negative_outlier_factor_ = nof
        if self.contamination_auto:
            self.offset_ = -1.5
        else:
            self.offset_ = float(np.percentile(nof, 100.0 * self.contamination_value))
        return self

    def _predict(self, X = None):
        if X is not None:
            s = self.decision_function(X)
            out = np.ones(s.shape[0], dtype=int)
            for i in range(s.shape[0]):
                if s[i] < 0.0:
                    out[i] = -1
            return out
        out = np.ones(self.n_samples_fit_, dtype=int)
        for i in range(self.n_samples_fit_):
            if self.negative_outlier_factor_[i] < self.offset_:
                out[i] = -1
        return out

    def fit_predict(self, X, y = None):
        if self.novelty:
            raise AttributeError("fit_predict is not available when novelty=True. Use novelty=False if you want to predict on the training set.")
        return self.fit(X)._predict()

    def predict(self, X = None):
        if not self.novelty:
            raise AttributeError("predict is not available when novelty=False, use fit_predict if you want to predict on training data. Use novelty=True if you want to use LOF for novelty detection and predict on new unseen data.")
        return self._predict(X)

    def score_samples(self, X):
        if not self.novelty:
            raise AttributeError("score_samples is not available when novelty=False. The scores of the training samples are always available through the negative_outlier_factor_ attribute. Use novelty=True if you want to use LOF for novelty detection and compute score_samples for new unseen data.")
        dist, ind = self._nn.kneighbors(X, self.n_neighbors_, True)
        x_lrd = self._lrd_of(dist, ind)
        nq, k = ind.shape
        out = np.empty(nq)
        for i in range(nq):
            s = 0.0
            for j in range(k):
                s += self._lrd[ind[i, j]] / x_lrd[i]
            out[i] = -(s / float(k))
        return out

    def decision_function(self, X):
        if not self.novelty:
            raise AttributeError("decision_function is not available when novelty=False. Use novelty=True if you want to use LOF for novelty detection and compute decision_function for new unseen data. Note that the opposite LOF of the training samples is always available by considering the negative_outlier_factor_ attribute.")
        return self.score_samples(X) - self.offset_

    def kneighbors(self, X = None, n_neighbors: Optional[int] = None,
                   return_distance: Literal[bool] = True):
        return self._nn.kneighbors(X, n_neighbors, return_distance)

    def kneighbors_graph(self, X = None, n_neighbors: Optional[int] = None, mode: str = "connectivity"):
        return self._nn.kneighbors_graph(X, n_neighbors, mode)
