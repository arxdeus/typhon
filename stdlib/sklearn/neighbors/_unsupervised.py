# Ported from scikit-learn 1.9.1 (BSD-3-Clause)

# sklearn.neighbors: NearestNeighbors, KNeighborsClassifier/Regressor,
# RadiusNeighborsClassifier/Regressor, NearestCentroid, KernelDensity.
# All algorithms ('auto', 'brute', 'kd_tree', 'ball_tree') use one exact
# brute-force backend (identical results to scikit-learn's trees).
# Classification targets are integer labels.

import numpy as np
from ._dist import _metric_code, _pairwise, _kneighbors, _as2d

def _check_algorithm(algorithm: str):
    if (algorithm != "auto" and algorithm != "brute" and algorithm != "kd_tree"
            and algorithm != "ball_tree"):
        raise ValueError(f"unknown algorithm {algorithm}")

def _get_weights(dist: np.ndarray[float, 2]) -> np.ndarray[float, 2]:
    # weights='distance': 1/d, rows with an exact match weight only those
    n, k = dist.shape
    w = np.empty((n, k))
    for i in range(n):
        has0 = False
        for j in range(k):
            if dist[i, j] == 0.0:
                has0 = True
        for j in range(k):
            if has0:
                w[i, j] = 1.0 if dist[i, j] == 0.0 else 0.0
            else:
                w[i, j] = 1.0 / dist[i, j]
    return w

class _NeighborsBase:
    """Shared fitted state and queries (fields are on concrete classes)."""
    pass

class NearestNeighbors:
    n_neighbors: int
    radius: float
    algorithm: str
    leaf_size: int
    metric: str
    p: float
    _fit_X: np.ndarray[float, 2]
    _code: int
    n_samples_fit_: int
    n_features_in_: int
    effective_metric_: str

    def __init__(self, n_neighbors: int = 5, radius: float = 1.0,
                 algorithm: str = "auto", leaf_size: int = 30,
                 metric: str = "minkowski", p: float = 2.0,
                 metric_params = None, n_jobs = None):
        _check_algorithm(algorithm)
        self.n_neighbors = n_neighbors
        self.radius = radius
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.metric = metric
        self.p = float(p)
        self._fit_X = np.empty((0, 0))
        self._code = _metric_code(metric, self.p)
        self.n_samples_fit_ = 0
        self.n_features_in_ = 0
        self.effective_metric_ = metric

    def _set_fit(self, X):
        self._code = _metric_code(self.metric, self.p)
        self._fit_X = _as2d(X)
        self.n_samples_fit_ = self._fit_X.shape[0]
        self.n_features_in_ = self._fit_X.shape[1]
        em = self.metric
        if em == "minkowski":
            if self.p == 2.0:
                em = "euclidean"
            elif self.p == 1.0:
                em = "manhattan"
            elif self.p == float("inf"):
                em = "chebyshev"
        self.effective_metric_ = em

    def fit(self, X, y = None):
        self._set_fit(X)
        return self

    def _dist(self, X) -> np.ndarray[float, 2]:
        return _pairwise(_as2d(X), self._fit_X, self._code, self.p)

    def kneighbors(self, X = None, n_neighbors: Optional[int] = None,
                   return_distance: Literal[bool] = True):
        k = self.n_neighbors if n_neighbors is None else n_neighbors
        if k <= 0:
            raise ValueError(f"Expected n_neighbors > 0. Got {k}")
        if X is None:
            if k >= self.n_samples_fit_:
                raise ValueError(f"Expected n_neighbors < n_samples_fit, but n_neighbors = {k}, n_samples_fit = {self.n_samples_fit_}, n_samples = {self.n_samples_fit_}")
            D = _pairwise(self._fit_X, self._fit_X, self._code, self.p)
            dist, ind = _kneighbors(D, k, True)
        else:
            if k > self.n_samples_fit_:
                raise ValueError(f"Expected n_neighbors <= n_samples_fit, but n_neighbors = {k}, n_samples_fit = {self.n_samples_fit_}")
            dist, ind = _kneighbors(self._dist(X), k, False)
        if return_distance:
            return dist, ind
        else:
            return ind

    def kneighbors_graph(self, X = None, n_neighbors: Optional[int] = None,
                         mode: str = "connectivity"):
        """Dense (n_queries, n_samples_fit) matrix (sparse in scikit-learn)."""
        dist, ind = self.kneighbors(X, n_neighbors, True)
        nq = ind.shape[0]
        G = np.zeros((nq, self.n_samples_fit_))
        for i in range(nq):
            for j in range(ind.shape[1]):
                G[i, ind[i, j]] = 1.0 if mode == "connectivity" else dist[i, j]
        return G

    def radius_neighbors(self, X = None, radius: Optional[float] = None,
                         return_distance: Literal[bool] = True,
                         sort_results: bool = False):
        r = self.radius if radius is None else radius
        skip = X is None
        D = _pairwise(self._fit_X, self._fit_X, self._code, self.p) if X is None else self._dist(X)
        dists = List[np.ndarray[float, 1]]()
        inds = List[np.ndarray[int, 1]]()
        for i in range(D.shape[0]):
            li = List[int]()
            for j in range(D.shape[1]):
                if skip and j == i:
                    continue
                if D[i, j] <= r:
                    li.append(j)
            if sort_results:
                li.sort(key=lambda j: (D[i, j], j))
            ia = np.empty(len(li), dtype=int)
            da = np.empty(len(li))
            for t in range(len(li)):
                ia[t] = li[t]
                da[t] = D[i, li[t]]
            dists.append(da)
            inds.append(ia)
        if return_distance:
            return dists, inds
        else:
            return inds

    def radius_neighbors_graph(self, X = None, radius: Optional[float] = None,
                               mode: str = "connectivity",
                               sort_results: bool = False):
        dists, inds = self.radius_neighbors(X, radius, True)
        G = np.zeros((len(inds), self.n_samples_fit_))
        for i in range(len(inds)):
            for t in range(inds[i].shape[0]):
                G[i, inds[i][t]] = 1.0 if mode == "connectivity" else dists[i][t]
        return G

def _nn(n_neighbors: int, radius: float, algorithm: str, leaf_size: int,
        metric: str, p: float) -> NearestNeighbors:
    return NearestNeighbors(n_neighbors=n_neighbors, radius=radius,
                            algorithm=algorithm, leaf_size=leaf_size,
                            metric=metric, p=p)

def _class_index(y):
    ya = np.asarray(y, dtype=int)
    classes, yi = np.unique(ya, return_inverse=True)
    return classes, np.asarray(yi, dtype=int).ravel()
