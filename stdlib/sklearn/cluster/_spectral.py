# Ported from scikit-learn 1.9.1 (BSD-3-Clause)

# SpectralClustering with dense affinities (rbf, nearest_neighbors,
# precomputed, precomputed_nearest_neighbors), a dense spectral embedding
# (normalized Laplacian eigenvectors, deterministic sign flip; the ARPACK
# start vector is still drawn so the RandomState stream matches sklearn) and
# label assignment by kmeans, discretize or cluster_qr.

import numpy as np
import scipy.linalg as sl
from numpy.random.legacy import RandomState, check_random_state
from ._kmeans import _as2d, KMeans

def _rbf(X: np.ndarray[float, 2], gamma: float) -> np.ndarray[float, 2]:
    n = X.shape[0]
    G = X @ X.T
    K = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            d = G[i, i] - 2.0 * G[i, j] + G[j, j]
            if d < 0.0 or i == j:
                d = 0.0 if i == j else max(d, 0.0)
            K[i, j] = np.exp(-gamma * d)
    return K

def _knn_connectivity(D: np.ndarray[float, 2], k: int) -> np.ndarray[float, 2]:
    n = D.shape[0]
    C = np.zeros((n, n))
    for i in range(n):
        order = np.argsort(D[i], kind="stable")
        for t in range(k):
            C[i, order[t]] = 1.0
    return 0.5 * (C + C.T)

def _euclid(X: np.ndarray[float, 2]) -> np.ndarray[float, 2]:
    n = X.shape[0]
    G = X @ X.T
    D = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            v = G[i, i] - 2.0 * G[i, j] + G[j, j]
            D[i, j] = np.sqrt(v) if v > 0.0 else 0.0
        D[i, i] = 0.0
    return D

def _sign_flip(u: np.ndarray[float, 2]):
    for r in range(u.shape[0]):
        best = 0
        for c in range(u.shape[1]):
            if abs(u[r, c]) > abs(u[r, best]):
                best = c
        s = 1.0 if u[r, best] >= 0.0 else -1.0
        for c in range(u.shape[1]):
            u[r, c] *= s

def _spectral_embedding(A0: np.ndarray[float, 2], n_components: int, rs: RandomState,
                        norm_laplacian: bool = True, drop_first: bool = True):
    n = A0.shape[0]
    nc = n_components + 1 if drop_first else n_components
    A = 0.5 * (A0 + A0.T)
    m = A.copy()
    for i in range(n):
        m[i, i] = 0.0
    w = m.sum(axis=0)
    zero = np.zeros(n, dtype=bool)
    if norm_laplacian:
        for i in range(n):
            w[i] = np.sqrt(w[i])
            if w[i] == 0.0:
                zero[i] = True
                w[i] = 1.0
        L = np.empty((n, n))
        for i in range(n):
            for j in range(n):
                L[i, j] = -m[i, j] / w[j] / w[i]
        for i in range(n):
            L[i, i] = 1.0
    else:
        L = -m
        for i in range(n):
            L[i, i] = w[i]
    # ARPACK start vector (consumed for RandomState stream parity)
    rs.uniform(-1.0, 1.0, n)
    vals, vecs = sl.eigh(L)
    emb = np.ascontiguousarray(vecs.T[:nc])
    if norm_laplacian:
        emb = emb / w
    _sign_flip(emb)
    if drop_first:
        return np.ascontiguousarray(emb[1:nc].T)
    return np.ascontiguousarray(emb[:nc].T)

def spectral_embedding(adjacency, *, n_components: int = 8, eigen_solver = None,
                       random_state = None, eigen_tol = "auto", norm_laplacian: bool = True,
                       drop_first: bool = True):
    return _spectral_embedding(_as2d(adjacency), n_components, check_random_state(random_state),
                               norm_laplacian, drop_first)

def discretize(vectors, *, copy: bool = True, max_svd_restarts: int = 30, n_iter_max: int = 20,
               random_state = None):
    rs = random_state if isinstance(random_state, RandomState) else check_random_state(random_state)
    V = _as2d(vectors).copy()
    eps = np.finfo(float).eps
    n, k = V.shape
    norm_ones = np.sqrt(float(n))
    for i in range(k):
        nv = np.sqrt(float((V[:, i] ** 2).sum()))
        V[:, i] = (V[:, i] / nv) * norm_ones
        if V[0, i] != 0.0:
            V[:, i] = -1.0 * V[:, i] * np.sign(V[0, i])
    V = V / np.sqrt((V ** 2).sum(axis=1))[:, None]
    restarts = 0
    converged = False
    labels = np.zeros(n, dtype=int)
    while restarts < max_svd_restarts and not converged:
        R = np.zeros((k, k))
        R[:, 0] = V[rs.randint(n), :]
        c = np.zeros(n)
        for j in range(1, k):
            c += np.abs(V @ R[:, j - 1])
            R[:, j] = V[int(c.argmin()), :]
        last = 0.0
        it = 0
        while not converged:
            it += 1
            T = V @ R
            labels = T.argmax(axis=1)
            tsvd = np.zeros((k, k))
            for s in range(n):
                tsvd[labels[s]] += V[s]
            U, S, Vh = np.linalg.svd(tsvd)
            ncut = 2.0 * (n - float(S.sum()))
            if abs(ncut - last) < eps or it > n_iter_max:
                converged = True
            else:
                last = ncut
                R = Vh.T @ U.T
    if not converged:
        raise np.linalg.LinAlgError("SVD did not converge")
    return labels

def cluster_qr(vectors):
    V = _as2d(vectors)
    k = V.shape[1]
    _, _, piv = sl.qr(np.ascontiguousarray(V.T), pivoting=True)
    sub = np.ascontiguousarray(V[piv[:k], :].T)
    ut, _, v = sl.svd(sub)
    W = np.abs(V @ (ut @ v))
    return W.argmax(axis=1)

class SpectralClustering:
    n_clusters: int
    eigen_solver: Optional[str]
    n_components: Optional[int]
    random_state: Optional[int]
    n_init: int
    gamma: float
    affinity: str
    n_neighbors: int
    assign_labels: str
    affinity_matrix_: np.ndarray[float, 2]
    labels_: np.ndarray[int, 1]
    n_features_in_: int

    def __init__(self, n_clusters: int = 8, *, eigen_solver: Optional[str] = None,
                 n_components: Optional[int] = None, random_state: Optional[int] = None,
                 n_init: int = 10, gamma: float = 1.0, affinity: str = "rbf", n_neighbors: int = 10,
                 eigen_tol = "auto", assign_labels: str = "kmeans", degree: float = 3,
                 coef0: float = 1, kernel_params = None, n_jobs = None, verbose: bool = False):
        if affinity not in ("rbf", "nearest_neighbors", "precomputed", "precomputed_nearest_neighbors"):
            raise ValueError(f"affinity {affinity!r} is not supported in Typhon (use rbf, nearest_neighbors, precomputed, precomputed_nearest_neighbors)")
        if assign_labels not in ("kmeans", "discretize", "cluster_qr"):
            raise ValueError(f"The 'assign_labels' parameter of SpectralClustering must be a str among {{'kmeans', 'discretize', 'cluster_qr'}}. Got {assign_labels!r} instead.")
        if eigen_solver is not None:
            if eigen_solver.__val__() == "amg":
                raise ValueError("The eigen_solver was set to 'amg', but pyamg is not available.")
        self.n_clusters = n_clusters
        self.eigen_solver = eigen_solver
        self.n_components = n_components
        self.random_state = random_state
        self.n_init = n_init
        self.gamma = gamma
        self.affinity = affinity
        self.n_neighbors = n_neighbors
        self.assign_labels = assign_labels
        self.affinity_matrix_ = np.empty((0, 0))
        self.labels_ = np.empty(0, dtype=int)
        self.n_features_in_ = 0

    def fit(self, X, y = None):
        Xa = _as2d(X)
        if self.affinity == "nearest_neighbors":
            self.affinity_matrix_ = _knn_connectivity(_euclid(Xa), self.n_neighbors)
        elif self.affinity == "precomputed_nearest_neighbors":
            self.affinity_matrix_ = _knn_connectivity(Xa, self.n_neighbors)
        elif self.affinity == "precomputed":
            self.affinity_matrix_ = Xa.copy()
        else:
            self.affinity_matrix_ = _rbf(Xa, self.gamma)
        rs = check_random_state(self.random_state)
        nc = self.n_clusters if self.n_components is None else self.n_components.__val__()
        maps = _spectral_embedding(self.affinity_matrix_, nc, rs, True, False)
        if self.assign_labels == "kmeans":
            km = KMeans(self.n_clusters, random_state=rs, n_init=self.n_init)
            km.fit(maps)
            self.labels_ = km.labels_
        elif self.assign_labels == "cluster_qr":
            self.labels_ = cluster_qr(maps)
        else:
            self.labels_ = discretize(maps, random_state=rs)
        self.n_features_in_ = Xa.shape[1]
        return self

    def fit_predict(self, X, y = None):
        return self.fit(X).labels_
