# Ported from scikit-learn 1.9.1 (BSD-3-Clause)

from ..base import estimator_repr as _estimator_repr
import numpy as np
from ._dist import _as2d
from ._unsupervised import NearestNeighbors, _nn, _class_index, _get_weights, _check_algorithm

def _check_weights(weights: str):
    if weights != "uniform" and weights != "distance":
        raise ValueError(f"weights not recognized: should be 'uniform' or 'distance', got {weights}")

def _r2(y_true, y_pred) -> float:
    a = np.asarray(y_true, dtype=float).ravel()
    b = np.asarray(y_pred, dtype=float).ravel()
    m = a.mean()
    ss_res = 0.0
    ss_tot = 0.0
    for i in range(a.shape[0]):
        ss_res += (a[i] - b[i]) ** 2
        ss_tot += (a[i] - m) ** 2
    if ss_tot == 0.0:
        return 1.0 if ss_res == 0.0 else 0.0
    return 1.0 - ss_res / ss_tot

def _acc(y_true, y_pred) -> float:
    a = np.asarray(y_true, dtype=int).ravel()
    b = np.asarray(y_pred, dtype=int).ravel()
    c = 0
    for i in range(a.shape[0]):
        if a[i] == b[i]:
            c += 1
    return float(c) / float(a.shape[0]) if a.shape[0] > 0 else 0.0

class KNeighborsClassifier:
    def __repr__(self) -> str:
        return _estimator_repr(self)

    n_neighbors: int
    weights: str
    algorithm: str
    leaf_size: int
    p: float
    metric: str
    _nn: NearestNeighbors
    classes_: np.ndarray[int, 1]
    _y: np.ndarray[int, 1]
    _estimator_type: Literal[str] = "classifier"

    def __init__(self, n_neighbors: int = 5, weights: str = "uniform",
                 algorithm: str = "auto", leaf_size: int = 30, p: float = 2.0,
                 metric: str = "minkowski", metric_params = None, n_jobs = None):
        _check_weights(weights)
        _check_algorithm(algorithm)
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.p = float(p)
        self.metric = metric
        self._nn = _nn(n_neighbors, 1.0, algorithm, leaf_size, metric, self.p)
        self.classes_ = np.empty(0, dtype=int)
        self._y = np.empty(0, dtype=int)

    def fit(self, X, y):
        self._nn = _nn(self.n_neighbors, 1.0, self.algorithm, self.leaf_size, self.metric, self.p)
        self._nn.fit(X)
        self.classes_, self._y = _class_index(y)
        return self

    @property
    def n_samples_fit_(self) -> int:
        return self._nn.n_samples_fit_

    @property
    def n_features_in_(self) -> int:
        return self._nn.n_features_in_

    @property
    def effective_metric_(self) -> str:
        return self._nn.effective_metric_

    def kneighbors(self, X = None, n_neighbors: Optional[int] = None,
                   return_distance: Literal[bool] = True):
        k = self.n_neighbors if n_neighbors is None else n_neighbors
        return self._nn.kneighbors(X, k, return_distance)

    def kneighbors_graph(self, X = None, n_neighbors: Optional[int] = None,
                         mode: str = "connectivity"):
        k = self.n_neighbors if n_neighbors is None else n_neighbors
        return self._nn.kneighbors_graph(X, k, mode)

    def predict_proba(self, X = None):
        dist, ind = self._nn.kneighbors(X, self.n_neighbors, True)
        nq, k = ind.shape
        nc = self.classes_.shape[0]
        w = np.ones((nq, k)) if self.weights == "uniform" else _get_weights(dist)
        P = np.zeros((nq, nc))
        for i in range(nq):
            for j in range(k):
                P[i, self._y[ind[i, j]]] += w[i, j]
            s = 0.0
            for c in range(nc):
                s += P[i, c]
            if s == 0.0:
                s = 1.0
            for c in range(nc):
                P[i, c] /= s
        return P

    def predict(self, X = None):
        P = self.predict_proba(X)
        idx = np.argmax(P, axis=1)
        out = np.empty(P.shape[0], dtype=int)
        for i in range(P.shape[0]):
            out[i] = self.classes_[idx[i]]
        return out

    def score(self, X, y, sample_weight = None):
        return _acc(y, self.predict(X))

class KNeighborsRegressor:
    def __repr__(self) -> str:
        return _estimator_repr(self)

    n_neighbors: int
    weights: str
    algorithm: str
    leaf_size: int
    p: float
    metric: str
    _nn: NearestNeighbors
    _y: np.ndarray[float, 1]
    _estimator_type: Literal[str] = "regressor"

    def __init__(self, n_neighbors: int = 5, weights: str = "uniform",
                 algorithm: str = "auto", leaf_size: int = 30, p: float = 2.0,
                 metric: str = "minkowski", metric_params = None, n_jobs = None):
        _check_weights(weights)
        _check_algorithm(algorithm)
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.p = float(p)
        self.metric = metric
        self._nn = _nn(n_neighbors, 1.0, algorithm, leaf_size, metric, self.p)
        self._y = np.empty(0)

    def fit(self, X, y):
        self._nn = _nn(self.n_neighbors, 1.0, self.algorithm, self.leaf_size, self.metric, self.p)
        self._nn.fit(X)
        self._y = np.ascontiguousarray(np.asarray(y, dtype=float).ravel())
        return self

    @property
    def n_samples_fit_(self) -> int:
        return self._nn.n_samples_fit_

    def kneighbors(self, X = None, n_neighbors: Optional[int] = None,
                   return_distance: Literal[bool] = True):
        k = self.n_neighbors if n_neighbors is None else n_neighbors
        return self._nn.kneighbors(X, k, return_distance)

    def predict(self, X = None):
        dist, ind = self._nn.kneighbors(X, self.n_neighbors, True)
        nq, k = ind.shape
        out = np.empty(nq)
        if self.weights == "uniform":
            for i in range(nq):
                s = 0.0
                for j in range(k):
                    s += self._y[ind[i, j]]
                out[i] = s / float(k)
        else:
            w = _get_weights(dist)
            for i in range(nq):
                s = 0.0
                sw = 0.0
                for j in range(k):
                    s += self._y[ind[i, j]] * w[i, j]
                    sw += w[i, j]
                out[i] = s / sw
        return out

    def score(self, X, y, sample_weight = None):
        return _r2(y, self.predict(X))

def _radius_weights(d: np.ndarray[float, 1], weights: str) -> np.ndarray[float, 1]:
    n = d.shape[0]
    w = np.ones(n)
    if weights == "distance":
        has0 = False
        for j in range(n):
            if d[j] == 0.0:
                has0 = True
        for j in range(n):
            w[j] = (1.0 if d[j] == 0.0 else 0.0) if has0 else 1.0 / d[j]
    return w

class RadiusNeighborsClassifier:
    def __repr__(self) -> str:
        return _estimator_repr(self)

    """outlier_label: None (raise on a query without neighbors), an int
    label, or 'most_frequent' (via outlier_label_most_frequent=True)."""
    radius: float
    weights: str
    algorithm: str
    leaf_size: int
    p: float
    metric: str
    outlier_label: Optional[int]
    outlier_most_frequent: bool
    _nn: NearestNeighbors
    classes_: np.ndarray[int, 1]
    _y: np.ndarray[int, 1]
    outlier_label_: int
    _estimator_type: Literal[str] = "classifier"

    def __init__(self, radius: float = 1.0, weights: str = "uniform",
                 algorithm: str = "auto", leaf_size: int = 30, p: float = 2.0,
                 metric: str = "minkowski", outlier_label: Optional[int] = None,
                 metric_params = None, n_jobs = None,
                 outlier_label_most_frequent: bool = False):
        _check_weights(weights)
        _check_algorithm(algorithm)
        self.radius = radius
        self.weights = weights
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.p = float(p)
        self.metric = metric
        self.outlier_label = outlier_label
        self.outlier_most_frequent = outlier_label_most_frequent
        self._nn = _nn(0, radius, algorithm, leaf_size, metric, self.p)
        self.classes_ = np.empty(0, dtype=int)
        self._y = np.empty(0, dtype=int)
        self.outlier_label_ = 0

    def fit(self, X, y):
        self._nn = _nn(0, self.radius, self.algorithm, self.leaf_size, self.metric, self.p)
        self._nn.fit(X)
        self.classes_, self._y = _class_index(y)
        if self.outlier_most_frequent:
            cnt = np.bincount(self._y)
            self.outlier_label_ = int(self.classes_[int(np.argmax(cnt))])
        elif self.outlier_label is not None:
            self.outlier_label_ = self.outlier_label
        return self

    def radius_neighbors(self, X = None, radius: Optional[float] = None,
                         return_distance: Literal[bool] = True,
                         sort_results: bool = False):
        return self._nn.radius_neighbors(X, radius, return_distance, sort_results)

    def predict_proba(self, X = None):
        dists, inds = self._nn.radius_neighbors(X, self.radius, True)
        nq = len(inds)
        nc = self.classes_.shape[0]
        has_outlier_label = self.outlier_most_frequent or self.outlier_label is not None
        P = np.zeros((nq, nc))
        for i in range(nq):
            if inds[i].shape[0] == 0:
                if not has_outlier_label:
                    raise ValueError("No neighbors found for test samples, you can try using larger radius, giving a label for outliers, or considering removing them from your dataset.")
                for c in range(nc):
                    if self.classes_[c] == self.outlier_label_:
                        P[i, c] = 1.0
                continue
            w = _radius_weights(dists[i], self.weights)
            for t in range(inds[i].shape[0]):
                P[i, self._y[inds[i][t]]] += w[t]
            s = 0.0
            for c in range(nc):
                s += P[i, c]
            if s == 0.0:
                s = 1.0
            for c in range(nc):
                P[i, c] /= s
        return P

    def predict(self, X = None):
        dists, inds = self._nn.radius_neighbors(X, self.radius, True)
        P = self.predict_proba(X)
        out = np.empty(P.shape[0], dtype=int)
        for i in range(P.shape[0]):
            if inds[i].shape[0] == 0:
                out[i] = self.outlier_label_
            else:
                out[i] = self.classes_[int(np.argmax(P[i]))]
        return out

    def score(self, X, y, sample_weight = None):
        return _acc(y, self.predict(X))

class RadiusNeighborsRegressor:
    def __repr__(self) -> str:
        return _estimator_repr(self)

    radius: float
    weights: str
    algorithm: str
    leaf_size: int
    p: float
    metric: str
    _nn: NearestNeighbors
    _y: np.ndarray[float, 1]
    _estimator_type: Literal[str] = "regressor"

    def __init__(self, radius: float = 1.0, weights: str = "uniform",
                 algorithm: str = "auto", leaf_size: int = 30, p: float = 2.0,
                 metric: str = "minkowski", metric_params = None, n_jobs = None):
        _check_weights(weights)
        _check_algorithm(algorithm)
        self.radius = radius
        self.weights = weights
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.p = float(p)
        self.metric = metric
        self._nn = _nn(0, radius, algorithm, leaf_size, metric, self.p)
        self._y = np.empty(0)

    def fit(self, X, y):
        self._nn = _nn(0, self.radius, self.algorithm, self.leaf_size, self.metric, self.p)
        self._nn.fit(X)
        self._y = np.ascontiguousarray(np.asarray(y, dtype=float).ravel())
        return self

    def radius_neighbors(self, X = None, radius: Optional[float] = None,
                         return_distance: Literal[bool] = True,
                         sort_results: bool = False):
        return self._nn.radius_neighbors(X, radius, return_distance, sort_results)

    def predict(self, X = None):
        """Queries without neighbors predict NaN (as scikit-learn)."""
        dists, inds = self._nn.radius_neighbors(X, self.radius, True)
        nq = len(inds)
        out = np.empty(nq)
        for i in range(nq):
            n = inds[i].shape[0]
            if n == 0:
                out[i] = float("nan")
                continue
            w = _radius_weights(dists[i], self.weights)
            s = 0.0
            sw = 0.0
            for t in range(n):
                s += self._y[inds[i][t]] * w[t]
                sw += w[t]
            out[i] = s / sw
        return out

    def score(self, X, y, sample_weight = None):
        return _r2(y, self.predict(X))
