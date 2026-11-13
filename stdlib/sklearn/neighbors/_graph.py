# Ported from scikit-learn 1.9.1 (BSD-3-Clause)

# sklearn.neighbors._graph (kneighbors_graph, radius_neighbors_graph,
# KNeighborsTransformer, RadiusNeighborsTransformer) and
# sort_graph_by_row_values from _base.py; graphs are scipy.sparse CSR.

import numpy as np
from scipy.sparse import csr_matrix
from ._unsupervised import NearestNeighbors

def _include_self(include_self, mode: str) -> bool:
    if isinstance(include_self, str):
        if include_self != "auto":
            raise ValueError("include_self must be a boolean or 'auto'")
        return mode == "connectivity"
    return bool(include_self)

def kneighbors_graph(X, n_neighbors: int, *, mode: str = "connectivity",
                     metric: str = "minkowski", p: float = 2.0, metric_params = None,
                     include_self = False, n_jobs = None):
    if isinstance(X, NearestNeighbors):
        nn = X
        inc = _include_self(include_self, mode)
        if inc:
            return nn.kneighbors_graph(nn._fit_X, n_neighbors, mode)
        return nn.kneighbors_graph(None, n_neighbors, mode)
    else:
        nn = NearestNeighbors(n_neighbors=n_neighbors, metric=metric, p=p).fit(X)
        inc = _include_self(include_self, mode)
        if inc:
            return nn.kneighbors_graph(nn._fit_X, n_neighbors, mode)
        return nn.kneighbors_graph(None, n_neighbors, mode)

def radius_neighbors_graph(X, radius: float, *, mode: str = "connectivity",
                           metric: str = "minkowski", p: float = 2.0, metric_params = None,
                           include_self = False, n_jobs = None):
    if isinstance(X, NearestNeighbors):
        nn = X
        inc = _include_self(include_self, mode)
        if inc:
            return nn.radius_neighbors_graph(nn._fit_X, radius, mode)
        return nn.radius_neighbors_graph(None, radius, mode)
    else:
        nn = NearestNeighbors(radius=radius, metric=metric, p=p).fit(X)
        inc = _include_self(include_self, mode)
        if inc:
            return nn.radius_neighbors_graph(nn._fit_X, radius, mode)
        return nn.radius_neighbors_graph(None, radius, mode)

def _is_sorted_by_data(indptr: np.ndarray[int, 1], data: np.ndarray[float, 1]) -> bool:
    for r in range(indptr.shape[0] - 1):
        for t in range(indptr[r] + 1, indptr[r + 1]):
            if data[t - 1] > data[t]:
                return False
    return True

def sort_graph_by_row_values(graph, copy: bool = False, warn_when_not_sorted: bool = True):
    """Sort each CSR row by data (stable, mergesort order); unsorted input is
    sorted in place unless copy=True."""
    g = graph.tocsr() if copy else graph
    if copy:
        g = g.copy()
    if _is_sorted_by_data(g.indptr, g.data):
        return g
    nr = g.indptr.shape[0] - 1
    for r in range(nr):
        s = int(g.indptr[r])
        e = int(g.indptr[r + 1])
        if e - s < 2:
            continue
        order = np.argsort(np.ascontiguousarray(g.data[s:e]), kind="stable")
        d2 = np.ascontiguousarray(g.data[s:e]).copy()
        i2 = np.ascontiguousarray(g.indices[s:e]).copy()
        for t in range(e - s):
            g.data[s + t] = d2[int(order[t])]
            g.indices[s + t] = i2[int(order[t])]
    return g

class KNeighborsTransformer:
    mode: str
    n_neighbors: int
    algorithm: str
    leaf_size: int
    metric: str
    p: float
    _nn: NearestNeighbors
    n_samples_fit_: int
    n_features_in_: int
    effective_metric_: str

    def __init__(self, *, mode: str = "distance", n_neighbors: int = 5, algorithm: str = "auto",
                 leaf_size: int = 30, metric: str = "minkowski", p: float = 2.0,
                 metric_params = None, n_jobs = None):
        self.mode = mode
        self.n_neighbors = n_neighbors
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.metric = metric
        self.p = float(p)
        self._nn = NearestNeighbors(n_neighbors=n_neighbors, algorithm=algorithm, leaf_size=leaf_size,
                                    metric=metric, p=float(p))
        self.n_samples_fit_ = 0
        self.n_features_in_ = 0
        self.effective_metric_ = metric

    def fit(self, X, y = None):
        self._nn = NearestNeighbors(n_neighbors=self.n_neighbors, algorithm=self.algorithm,
                                    leaf_size=self.leaf_size, metric=self.metric, p=self.p).fit(X)
        self.n_samples_fit_ = self._nn.n_samples_fit_
        self.n_features_in_ = self._nn.n_features_in_
        self.effective_metric_ = self._nn.effective_metric_
        return self

    def kneighbors(self, X = None, n_neighbors: Optional[int] = None,
                   return_distance: Literal[bool] = True):
        return self._nn.kneighbors(X, n_neighbors, return_distance)

    def kneighbors_graph(self, X = None, n_neighbors: Optional[int] = None, mode: str = "connectivity"):
        return self._nn.kneighbors_graph(X, n_neighbors, mode)

    def transform(self, X):
        add_one = 1 if self.mode == "distance" else 0
        return self._nn.kneighbors_graph(X, self.n_neighbors + add_one, self.mode)

    def fit_transform(self, X, y = None):
        return self.fit(X).transform(X)

    def get_feature_names_out(self) -> List[str]:
        return [f"kneighborstransformer{i}" for i in range(self.n_samples_fit_)]

class RadiusNeighborsTransformer:
    mode: str
    radius: float
    algorithm: str
    leaf_size: int
    metric: str
    p: float
    _nn: NearestNeighbors
    n_samples_fit_: int
    n_features_in_: int
    effective_metric_: str

    def __init__(self, *, mode: str = "distance", radius: float = 1.0, algorithm: str = "auto",
                 leaf_size: int = 30, metric: str = "minkowski", p: float = 2.0,
                 metric_params = None, n_jobs = None):
        self.mode = mode
        self.radius = radius
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.metric = metric
        self.p = float(p)
        self._nn = NearestNeighbors(n_neighbors=0, radius=radius, algorithm=algorithm,
                                    leaf_size=leaf_size, metric=metric, p=float(p))
        self.n_samples_fit_ = 0
        self.n_features_in_ = 0
        self.effective_metric_ = metric

    def fit(self, X, y = None):
        self._nn = NearestNeighbors(n_neighbors=0, radius=self.radius, algorithm=self.algorithm,
                                    leaf_size=self.leaf_size, metric=self.metric, p=self.p).fit(X)
        self.n_samples_fit_ = self._nn.n_samples_fit_
        self.n_features_in_ = self._nn.n_features_in_
        self.effective_metric_ = self._nn.effective_metric_
        return self

    def radius_neighbors(self, X = None, radius: Optional[float] = None,
                         return_distance: Literal[bool] = True, sort_results: bool = False):
        return self._nn.radius_neighbors(X, radius, return_distance, sort_results)

    def radius_neighbors_graph(self, X = None, radius: Optional[float] = None,
                               mode: str = "connectivity", sort_results: bool = False):
        return self._nn.radius_neighbors_graph(X, radius, mode, sort_results)

    def transform(self, X):
        return self._nn.radius_neighbors_graph(X, None, self.mode, True)

    def fit_transform(self, X, y = None):
        return self.fit(X).transform(X)

    def get_feature_names_out(self) -> List[str]:
        return [f"radiusneighborstransformer{i}" for i in range(self.n_samples_fit_)]
