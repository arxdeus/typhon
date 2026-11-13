# Ported from scikit-learn 1.9.1 (BSD-3-Clause)

# sklearn.neighbors: NearestNeighbors, KNeighborsClassifier/Regressor,
# RadiusNeighborsClassifier/Regressor, NearestCentroid, KernelDensity.
# All algorithms ('auto', 'brute', 'kd_tree', 'ball_tree') use one exact
# brute-force backend (identical results to scikit-learn's trees).
# Classification targets are integer labels.

import numpy as np
from ._dist import _metric_code, _pairwise, _kneighbors, _as2d, _kneighbors_heap
from ._binary_tree import _BinaryTree, _make_tree, _tree_kind_for
from ._tree_util import simultaneous_sort_range

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
    _fit_method: str
    _tree: Optional[_BinaryTree]

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
        self._fit_method = "brute"
        self._tree = None

    def _resolve_method(self, em: str):
        m = self.algorithm
        nf = self._fit_X.shape[1]
        if m == "auto":
            if nf > 15 or (self.n_neighbors > 0 and self.n_neighbors >= self.n_samples_fit_ // 2):
                m = "brute"
            elif self.metric == "minkowski" and self.p < 1.0:
                m = "brute"
            else:
                kind = _tree_kind_for(em, self.p)
                m = "kd_tree" if kind == 0 else ("ball_tree" if kind == 1 else "brute")
        if self.metric == "minkowski" and self.p < 1.0 and m != "brute":
            raise ValueError(f'algorithm="{m}" does not support 0 < p < 1 for the Minkowski metric. To resolve this problem either set p >= 1 or algorithm="brute".')
        if m != "brute" and _tree_kind_for(em, self.p) < 0 and self._code >= 0:
            m = "brute"   # metrics without a tree kernel (cosine, sqeuclidean) use the exact backend
        self._fit_method = m
        if m == "brute":
            if self._code < 0:
                raise ValueError(f"Metric '{self.metric}' not supported by algorithm='brute' in this port")
            self._tree = None
        else:
            if m == "kd_tree" and _tree_kind_for(em, self.p) != 0:
                raise ValueError(f"metric {em} is not valid for KDTree")
            self._tree = _make_tree(0 if m == "kd_tree" else 1, self._fit_X, self.leaf_size, em, self.p)

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
        self._resolve_method(em)

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
        if self._fit_method != "brute":
            return self._kneighbors_tree(X, k, return_distance)
        if X is None:
            if k >= self.n_samples_fit_:
                raise ValueError(f"Expected n_neighbors < n_samples_fit, but n_neighbors = {k}, n_samples_fit = {self.n_samples_fit_}, n_samples = {self.n_samples_fit_}")
            D = _pairwise(self._fit_X, self._fit_X, self._code, self.p)
            if self._code != 4:
                d0, i0 = _kneighbors_heap(self._fit_X, self._fit_X, D, self._code, k + 1)
                dist = np.empty((D.shape[0], k))
                ind = np.empty((D.shape[0], k), dtype=int)
                for i in range(D.shape[0]):
                    skip = 0
                    for j in range(k + 1):
                        if i0[i, j] == i:
                            skip = j
                            break
                    c = 0
                    for j in range(k + 1):
                        if j != skip:
                            dist[i, c] = d0[i, j]
                            ind[i, c] = i0[i, j]
                            c += 1
            else:
                dist, ind = _kneighbors(D, k, True)
        else:
            if k > self.n_samples_fit_:
                raise ValueError(f"Expected n_neighbors <= n_samples_fit, but n_neighbors = {k}, n_samples_fit = {self.n_samples_fit_}")
            Dq = self._dist(X)
            if self._code != 4:
                dist, ind = _kneighbors_heap(_as2d(X), self._fit_X, Dq, self._code, k)
            else:
                dist, ind = _kneighbors(Dq, k, False)
        if return_distance:
            return dist, ind
        else:
            return ind

    def _kneighbors_tree(self, X, k: int, return_distance: Literal[bool]):
        t = self._tree.__val__()
        n_fit = self.n_samples_fit_
        if X is None:
            if k + 1 > n_fit:
                raise ValueError(f"Expected n_neighbors < n_samples_fit, but n_neighbors = {k}, n_samples_fit = {n_fit}, n_samples = {n_fit}")
            dist0, ind0 = t.query(self._fit_X, k + 1, True)
            nq = ind0.shape[0]
            dist = np.empty((nq, k))
            ind = np.empty((nq, k), dtype=int)
            for i in range(nq):
                skip = 0
                for j in range(k + 1):
                    if ind0[i, j] == i:
                        skip = j
                        break
                c = 0
                for j in range(k + 1):
                    if j != skip:
                        dist[i, c] = dist0[i, j]
                        ind[i, c] = ind0[i, j]
                        c += 1
            if return_distance:
                return dist, ind
            else:
                return ind
        if k > n_fit:
            raise ValueError(f"Expected n_neighbors <= n_samples_fit, but n_neighbors = {k}, n_samples_fit = {n_fit}, n_samples = {_as2d(X).shape[0]}")
        dist, ind = t.query(_as2d(X), k, True)
        if return_distance:
            return dist, ind
        else:
            return ind

    def kneighbors_graph(self, X = None, n_neighbors: Optional[int] = None,
                         mode: str = "connectivity"):
        """CSR (n_queries, n_samples_fit) matrix, rows in neighbour order."""
        if mode != "connectivity" and mode != "distance":
            raise ValueError(f'Unsupported mode, must be one of "connectivity", or "distance" but got "{mode}" instead')
        k = self.n_neighbors if n_neighbors is None else n_neighbors
        dist, ind = self.kneighbors(X, k, True)
        nq = ind.shape[0]
        data = np.ones(nq * k)
        if mode == "distance":
            data = np.ascontiguousarray(dist.reshape((-1,)).copy())
        indices = np.ascontiguousarray(ind.reshape((-1,)).copy())
        indptr = np.arange(0, nq * k + 1, k)
        from scipy.sparse import csr_matrix
        return csr_matrix((data, indices, indptr), shape=(nq, self.n_samples_fit_))

    def radius_neighbors(self, X = None, radius: Optional[float] = None,
                         return_distance: Literal[bool] = True,
                         sort_results: bool = False):
        r = self.radius if radius is None else radius
        skip = X is None
        if self._fit_method != "brute":
            t = self._tree.__val__()
            Xq = self._fit_X if X is None else _as2d(X)
            ra = np.full(Xq.shape[0], r)
            _c, ind0, dst0 = t._radius_core(Xq, ra, True, False, sort_results)
            dl = List[np.ndarray[float, 1]]()
            il = List[np.ndarray[int, 1]]()
            for i in range(len(ind0)):
                a = ind0[i]
                b = dst0[i]
                if skip:
                    keep = List[int]()
                    for j in range(a.shape[0]):
                        if a[j] != i:
                            keep.append(j)
                    a2 = np.empty(len(keep), dtype=int)
                    b2 = np.empty(len(keep))
                    for j in range(len(keep)):
                        a2[j] = a[keep[j]]
                        b2[j] = b[keep[j]]
                    il.append(a2)
                    dl.append(b2)
                else:
                    il.append(a)
                    dl.append(b)
            if return_distance:
                return dl, il
            else:
                return il
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
                tv = np.empty(len(li))
                ti = np.empty(len(li), dtype=int)
                for t in range(len(li)):
                    tv[t] = D[i, li[t]]
                    ti[t] = li[t]
                simultaneous_sort_range(tv, ti, 0, len(li))
                for t in range(len(li)):
                    li[t] = int(ti[t])
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
        if mode != "connectivity" and mode != "distance":
            raise ValueError(f'Unsupported mode, must be one of "connectivity", or "distance" but got "{mode}" instead')
        dists, inds = self.radius_neighbors(X, radius, True, sort_results)
        nq = len(inds)
        tot = 0
        for i in range(nq):
            tot += inds[i].shape[0]
        data = np.ones(tot)
        indices = np.empty(tot, dtype=int)
        indptr = np.zeros(nq + 1, dtype=int)
        pos = 0
        for i in range(nq):
            for t in range(inds[i].shape[0]):
                indices[pos] = inds[i][t]
                if mode == "distance":
                    data[pos] = dists[i][t]
                pos += 1
            indptr[i + 1] = pos
        from scipy.sparse import csr_matrix
        return csr_matrix((data, indices, indptr), shape=(nq, self.n_samples_fit_))

def _nn(n_neighbors: int, radius: float, algorithm: str, leaf_size: int,
        metric: str, p: float) -> NearestNeighbors:
    return NearestNeighbors(n_neighbors=n_neighbors, radius=radius,
                            algorithm=algorithm, leaf_size=leaf_size,
                            metric=metric, p=p)

def _class_index(y):
    ya = np.asarray(y, dtype=int)
    classes, yi = np.unique(ya, return_inverse=True)
    return classes, np.asarray(yi, dtype=int).ravel()
