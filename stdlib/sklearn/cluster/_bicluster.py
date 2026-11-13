# Ported from scikit-learn 1.9.1 (BSD-3-Clause)

# SpectralCoclustering and SpectralBiclustering (svd_method 'randomized').

import numpy as np
from numpy.random.legacy import check_random_state
from ._kmeans import KMeans
from ._minibatch_kmeans import MiniBatchKMeans
from ..decomposition._extmath import randomized_svd as _rsvd

def _nonneg(X: np.ndarray[float, 2], min_value: float) -> np.ndarray[float, 2]:
    mn = float(np.min(X))
    if mn < min_value:
        return X + (min_value - mn)
    return X.copy()

def _scale_normalize(X0: np.ndarray[float, 2]):
    X = _nonneg(X0, 0.0)
    n, m = X.shape
    row_diag = 1.0 / np.sqrt(X.sum(axis=1))
    col_diag = 1.0 / np.sqrt(X.sum(axis=0))
    for i in range(n):
        if np.isnan(row_diag[i]):
            row_diag[i] = 0.0
    for j in range(m):
        if np.isnan(col_diag[j]):
            col_diag[j] = 0.0
    an = np.empty((n, m))
    for i in range(n):
        for j in range(m):
            an[i, j] = row_diag[i] * X[i, j] * col_diag[j]
    return an, row_diag, col_diag

def _bistochastic_normalize(X0: np.ndarray[float, 2], max_iter: int = 1000, tol: float = 1e-5):
    X = _nonneg(X0, 0.0)
    Xs = X
    for _ in range(max_iter):
        Xn, _r, _c = _scale_normalize(Xs)
        D = Xs - Xn
        dist = float(np.sqrt(np.sum(D * D)))
        Xs = Xn
        if dist < tol:
            break
    return Xs

def _log_normalize(X0: np.ndarray[float, 2]) -> np.ndarray[float, 2]:
    X = _nonneg(X0, 1.0)
    L = np.log(X)
    n, m = L.shape
    row_avg = L.mean(axis=1)
    col_avg = L.mean(axis=0)
    avg = float(L.mean())
    out = np.empty((n, m))
    for i in range(n):
        for j in range(m):
            out[i, j] = L[i, j] - row_avg[i] - col_avg[j] + avg
    return out

def _km(data: np.ndarray[float, 2], k: int, mini_batch: bool, init: str, n_init: int,
        random_state: Optional[int]):
    if mini_batch:
        mb = MiniBatchKMeans(k, init=init, n_init=n_init, random_state=random_state)
        mb.fit(data)
        return mb.cluster_centers_, mb.labels_
    km = KMeans(k, init=init, n_init=n_init, random_state=random_state)
    km.fit(data)
    return km.cluster_centers_, km.labels_

def _svd_uv(array: np.ndarray[float, 2], n_components: int, n_discard: int,
            svd_method: str, n_svd_vecs: int, random_state: Optional[int]):
    rs = check_random_state(random_state)
    u0 = np.empty((0, 0))
    vt0 = np.empty((0, 0))
    if svd_method == "randomized":
        nover = 10
        if n_svd_vecs > 0:
            nover = n_svd_vecs
        u0, _s, vt0 = _rsvd(array, n_components, rs, nover, -1, "auto", True)
    else:
        import scipy.linalg as sl
        uu, ss, vv = sl.svd(array, full_matrices=False)
        u0 = np.ascontiguousarray(uu[:, :n_components])
        vt0 = np.ascontiguousarray(vv[:n_components])
    u = np.ascontiguousarray(u0[:, n_discard:])
    v = np.ascontiguousarray(vt0[n_discard:].T)
    return u, v

class SpectralCoclustering:
    n_clusters: int
    svd_method: str
    n_svd_vecs: int
    mini_batch: bool
    init: str
    n_init: int
    random_state: Optional[int]
    row_labels_: np.ndarray[int, 1]
    column_labels_: np.ndarray[int, 1]
    rows_: np.ndarray[bool, 2]
    columns_: np.ndarray[bool, 2]

    def __init__(self, n_clusters: int = 3, *, svd_method: str = "randomized", n_svd_vecs = None,
                 mini_batch: bool = False, init: str = "k-means++", n_init: int = 10,
                 random_state = None):
        self.n_clusters = n_clusters
        self.svd_method = svd_method
        self.n_svd_vecs = 0 if n_svd_vecs is None else int(n_svd_vecs)
        self.mini_batch = mini_batch
        self.init = init
        self.n_init = n_init
        self.random_state = None if random_state is None else int(random_state)
        self.row_labels_ = np.empty(0, dtype=int)
        self.column_labels_ = np.empty(0, dtype=int)
        self.rows_ = np.empty((0, 0), dtype=bool)
        self.columns_ = np.empty((0, 0), dtype=bool)

    def fit(self, X, y = None):
        A = np.asarray(X, dtype=float)
        if self.n_clusters > A.shape[0]:
            raise ValueError(f"n_clusters should be <= n_samples={A.shape[0]}. Got {self.n_clusters} instead.")
        nd, row_diag, col_diag = _scale_normalize(A)
        n_sv = 1 + int(np.ceil(np.log2(self.n_clusters)))
        u, v = _svd_uv(nd, n_sv, 1, self.svd_method, self.n_svd_vecs, self.random_state)
        nr = A.shape[0]
        nc = A.shape[1]
        z = np.empty((nr + nc, n_sv - 1))
        for i in range(nr):
            for j in range(n_sv - 1):
                z[i, j] = row_diag[i] * u[i, j]
        for i in range(nc):
            for j in range(n_sv - 1):
                z[nr + i, j] = col_diag[i] * v[i, j]
        _c, labels = _km(z, self.n_clusters, self.mini_batch, self.init, self.n_init, self.random_state)
        self.row_labels_ = labels[:nr].copy()
        self.column_labels_ = labels[nr:].copy()
        k = self.n_clusters
        self.rows_ = np.zeros((k, nr), dtype=bool)
        self.columns_ = np.zeros((k, nc), dtype=bool)
        for c in range(k):
            for i in range(nr):
                self.rows_[c, i] = self.row_labels_[i] == c
            for j in range(nc):
                self.columns_[c, j] = self.column_labels_[j] == c
        return self

class SpectralBiclustering:
    n_row_clusters: int
    n_col_clusters: int
    method: str
    n_components: int
    n_best: int
    svd_method: str
    n_svd_vecs: int
    mini_batch: bool
    init: str
    n_init: int
    random_state: Optional[int]
    row_labels_: np.ndarray[int, 1]
    column_labels_: np.ndarray[int, 1]
    rows_: np.ndarray[bool, 2]
    columns_: np.ndarray[bool, 2]

    def __init__(self, n_clusters = 3, *, method: str = "bistochastic", n_components: int = 6,
                 n_best: int = 3, svd_method: str = "randomized", n_svd_vecs = None,
                 mini_batch: bool = False, init: str = "k-means++", n_init: int = 10,
                 random_state = None):
        if isinstance(n_clusters, int):
            self.n_row_clusters = n_clusters
            self.n_col_clusters = n_clusters
        else:
            self.n_row_clusters = int(n_clusters[0])
            self.n_col_clusters = int(n_clusters[1])
        if method != "bistochastic" and method != "scale" and method != "log":
            raise ValueError("Unknown method")
        self.method = method
        self.n_components = n_components
        self.n_best = n_best
        self.svd_method = svd_method
        self.n_svd_vecs = 0 if n_svd_vecs is None else int(n_svd_vecs)
        self.mini_batch = mini_batch
        self.init = init
        self.n_init = n_init
        self.random_state = None if random_state is None else int(random_state)
        self.row_labels_ = np.empty(0, dtype=int)
        self.column_labels_ = np.empty(0, dtype=int)
        self.rows_ = np.empty((0, 0), dtype=bool)
        self.columns_ = np.empty((0, 0), dtype=bool)

    def _best_piecewise(self, vectors: np.ndarray[float, 2], n_best: int, k: int) -> np.ndarray[float, 2]:
        nv, m = vectors.shape
        dists = np.empty(nv)
        for t in range(nv):
            v = np.ascontiguousarray(vectors[t].reshape(-1, 1))
            cen, lab = _km(v, k, self.mini_batch, self.init, self.n_init, self.random_state)
            s = 0.0
            for i in range(m):
                d = vectors[t, i] - cen[lab[i], 0]
                s += d * d
            dists[t] = np.sqrt(s)
        order = np.argsort(dists)
        out = np.empty((n_best, m))
        for i in range(n_best):
            out[i] = vectors[order[i]]
        return out

    def fit(self, X, y = None):
        A = np.asarray(X, dtype=float)
        ns = A.shape[0]
        if self.n_row_clusters > ns or self.n_col_clusters > ns or self.n_row_clusters < 1 or self.n_col_clusters < 1:
            raise ValueError("Incorrect parameter n_clusters")
        if self.n_best > self.n_components:
            raise ValueError(f"n_best={self.n_best} must be <= n_components={self.n_components}.")
        n_sv = self.n_components
        nd = np.empty((0, 0))
        if self.method == "bistochastic":
            nd = _bistochastic_normalize(A)
            n_sv += 1
        elif self.method == "scale":
            nd, _r, _c = _scale_normalize(A)
            n_sv += 1
        else:
            nd = _log_normalize(A)
        n_discard = 0 if self.method == "log" else 1
        u, v = _svd_uv(nd, n_sv, n_discard, self.svd_method, self.n_svd_vecs, self.random_state)
        ut = np.ascontiguousarray(u.T)
        vt = np.ascontiguousarray(v.T)
        best_ut = self._best_piecewise(ut, self.n_best, self.n_row_clusters)
        best_vt = self._best_piecewise(vt, self.n_best, self.n_col_clusters)
        proj_r = A @ np.ascontiguousarray(best_vt.T)
        _c1, self.row_labels_ = _km(proj_r, self.n_row_clusters, self.mini_batch, self.init,
                                    self.n_init, self.random_state)
        proj_c = np.ascontiguousarray(A.T) @ np.ascontiguousarray(best_ut.T)
        _c2, self.column_labels_ = _km(proj_c, self.n_col_clusters, self.mini_batch, self.init,
                                       self.n_init, self.random_state)
        R = self.n_row_clusters
        C = self.n_col_clusters
        nr = A.shape[0]
        nc = A.shape[1]
        self.rows_ = np.zeros((R * C, nr), dtype=bool)
        self.columns_ = np.zeros((R * C, nc), dtype=bool)
        for a in range(R):
            for b in range(C):
                for i in range(nr):
                    self.rows_[a * C + b, i] = self.row_labels_[i] == a
                for j in range(nc):
                    self.columns_[a * C + b, j] = self.column_labels_[j] == b
        return self
