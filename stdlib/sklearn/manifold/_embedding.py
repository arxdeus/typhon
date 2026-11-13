# Ported from scikit-learn 1.9.1 (BSD-3-Clause)
# MDS (SMACOF metric / non-metric), ClassicalMDS, Isomap,
# LocallyLinearEmbedding, SpectralEmbedding and trustworthiness.

import numpy as np
import scipy.linalg as sl
from ._utils import (_knn, _use_brute, _euclidean_distances, _components, _dijkstra_all,
                     _isotonic_fit_transform)
from ..decomposition._extmath import svd_flip_u_only, euclidean_distances_sq

# ------------------------------------------------------------------ MDS

def _smacof_single(D, metric: bool, n_components: int, init, has_init: bool, max_iter: int,
                   eps: float, rs, normalized_stress: bool):
    n = D.shape[0]
    tri_mask = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(i + 1, n):
            tri_mask[i, j] = True
    dflat = np.where(tri_mask, D, 0.0).ravel()
    nz = dflat != 0
    dflat_w = dflat[nz]
    X = np.empty((n, n_components))
    if has_init:
        X = init.copy()
    else:
        X = rs.uniform(size=n * n_components).reshape((n, n_components))
    dist = _euclidean_distances(X, X, True)
    old_stress = 0.0
    have_old = False
    stress = 0.0
    it = 0
    disp = D
    for i in range(max_iter):
        it = i
        if metric:
            disp = D
        else:
            dfl = dist.ravel()
            dfl_w = dfl[nz]
            df = dflat_w if i < 1 else _isotonic_fit_transform(dflat_w, dfl_w)
            dd = np.zeros(n * n)
            dd[nz] = df
            dm = dd.reshape((n, n))
            dm *= np.sqrt(n * (n - 1) / 2 / float((dm ** 2).sum()))
            disp = dm + dm.T
        dist[dist == 0] = 1e-5
        ratio = disp / dist
        B = -ratio
        rsum = ratio.sum(axis=1)
        for a in range(n):
            B[a, a] += rsum[a]
        X = 1.0 / n * (B @ X)
        dist = _euclidean_distances(X, X, True)
        stress = float(((dist.ravel() - disp.ravel()) ** 2).sum()) / 2
        if have_old:
            ssd = float((dist.ravel() ** 2).sum())
            if (old_stress - stress) / (ssd / 2) < eps:
                break
        old_stress = stress
        have_old = True
    if normalized_stress:
        ssd = float((dist.ravel() ** 2).sum())
        stress = float(np.sqrt(stress / (ssd / 2)))
    return X, stress, it + 1

class ClassicalMDS:
    n_components: int
    metric: str
    embedding_: np.ndarray[float, 2]
    eigenvalues_: np.ndarray[float, 1]
    dissimilarity_matrix_: np.ndarray[float, 2]

    def __init__(self, n_components: int = 2, *, metric: str = "euclidean", metric_params = None):
        self.n_components = n_components
        self.metric = metric
        self.embedding_ = np.empty((0, 0))
        self.eigenvalues_ = np.empty(0)
        self.dissimilarity_matrix_ = np.empty((0, 0))

    def fit_transform(self, X, y = None):
        Xa = np.asarray(X, dtype=float)
        if self.metric == "precomputed":
            self.dissimilarity_matrix_ = Xa.copy()
        else:
            self.dissimilarity_matrix_ = _euclidean_distances(Xa, Xa, True)
        B = self.dissimilarity_matrix_ ** 2
        B -= np.mean(B, axis=0)
        B -= np.mean(B, axis=1)[:, None]
        B *= -0.5
        w, U = sl.eigh(B)
        k = self.n_components
        w = np.ascontiguousarray(w[::-1][:k])
        U = np.ascontiguousarray(U[:, ::-1][:, :k])
        svd_flip_u_only(U)
        self.embedding_ = np.sqrt(w) * U
        self.eigenvalues_ = w
        return self.embedding_

    def fit(self, X, y = None):
        self.fit_transform(X)
        return self

class MDS:
    """SMACOF metric / non-metric MDS (sequential n_init). init 'random' or
    'classical_mds'; metric 'euclidean' or 'precomputed'."""
    n_components: int
    metric_mds: bool
    n_init: int
    init: str
    max_iter: int
    eps: float
    random_state: Optional[int]
    metric: str
    normalized_stress: str
    embedding_: np.ndarray[float, 2]
    stress_: float
    n_iter_: int
    dissimilarity_matrix_: np.ndarray[float, 2]

    def __init__(self, n_components: int = 2, *, metric_mds: bool = True, n_init: int = 1,
                 init: str = "warn", max_iter: int = 300, verbose: int = 0, eps: float = 1e-6,
                 n_jobs = None, random_state: Optional[int] = None, dissimilarity: str = "deprecated",
                 metric = "euclidean", metric_params = None, normalized_stress = "auto"):
        self.n_components = n_components
        self.metric_mds = metric_mds
        self.metric = "euclidean"
        if isinstance(metric, bool):
            self.metric_mds = metric
        else:
            self.metric = metric
        if dissimilarity != "deprecated":
            self.metric = dissimilarity
        if self.metric != "euclidean" and self.metric != "precomputed":
            raise NotImplementedError("MDS: only metric 'euclidean' / 'precomputed' are ported")
        self.n_init = n_init
        self.init = "random" if init == "warn" else init
        self.max_iter = max_iter
        self.eps = eps
        self.random_state = random_state
        if isinstance(normalized_stress, str):
            self.normalized_stress = normalized_stress
        else:
            self.normalized_stress = "true" if normalized_stress else "false"
        self.embedding_ = np.empty((0, 0))
        self.stress_ = 0.0
        self.n_iter_ = 0
        self.dissimilarity_matrix_ = np.empty((0, 0))

    def fit_transform(self, X, y = None, init = None):
        Xa = np.asarray(X, dtype=float)
        if self.metric == "precomputed":
            self.dissimilarity_matrix_ = Xa.copy()
        else:
            self.dissimilarity_matrix_ = _euclidean_distances(Xa, Xa, True)
        has_init = False
        init_arr = np.empty((0, 0))
        if init is not None:
            has_init = True
            init_arr = np.array(init, dtype=float)
        elif self.init == "classical_mds":
            has_init = True
            init_arr = ClassicalMDS(self.n_components, metric="precomputed").fit_transform(self.dissimilarity_matrix_)
        ns = (not self.metric_mds) if self.normalized_stress == "auto" else self.normalized_stress == "true"
        rs = np.random.RandomState(self.random_state)
        n_init = 1 if has_init else self.n_init
        best_stress = 0.0
        best_pos = np.empty((0, 0))
        best_iter = 0
        for t in range(n_init):
            pos, stress, nit = _smacof_single(self.dissimilarity_matrix_, self.metric_mds,
                                              self.n_components, init_arr, has_init, self.max_iter,
                                              self.eps, rs, ns)
            if t == 0 or stress < best_stress:
                best_stress = stress
                best_pos = pos.copy()
                best_iter = nit
        self.embedding_ = best_pos
        self.stress_ = best_stress
        self.n_iter_ = best_iter
        return self.embedding_

    def fit(self, X, y = None, init = None):
        self.fit_transform(X, init=init)
        return self

# ------------------------------------------------------------------ Isomap

class Isomap:
    """kNN graph + Dijkstra geodesics + KernelPCA(precomputed, dense)."""
    n_neighbors: int
    n_components: int
    _brute: bool
    embedding_: np.ndarray[float, 2]
    dist_matrix_: np.ndarray[float, 2]
    _fit_X: np.ndarray[float, 2]
    _evals: np.ndarray[float, 1]
    _evecs: np.ndarray[float, 2]
    _K_rows: np.ndarray[float, 1]
    _K_all: float

    def __init__(self, *, n_neighbors: int = 5, radius = None, n_components: int = 2,
                 eigen_solver: str = "auto", tol: float = 0, max_iter = None, path_method: str = "auto",
                 neighbors_algorithm: str = "auto", n_jobs = None, metric: str = "minkowski",
                 p: float = 2, metric_params = None):
        if radius is not None:
            raise NotImplementedError("Isomap: radius neighbourhoods are not ported")
        if metric != "minkowski" and metric != "euclidean":
            raise NotImplementedError("Isomap: only the euclidean metric is ported")
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self._brute = False
        self.embedding_ = np.empty((0, 0))
        self.dist_matrix_ = np.empty((0, 0))
        self._fit_X = np.empty((0, 0))
        self._evals = np.empty(0)
        self._evecs = np.empty((0, 0))
        self._K_rows = np.empty(0)
        self._K_all = 0.0

    def _kpca_fit(self, K):
        n = K.shape[0]
        self._K_rows = np.sum(K, axis=0) / float(n)
        self._K_all = float(self._K_rows.sum()) / float(n)
        Kc = K - self._K_rows[None, :] - (np.sum(K, axis=1) / float(n))[:, None] + self._K_all
        k = min(n, self.n_components)
        w, V = sl.eigh(Kc, subset_by_index=[n - k, n - 1])
        w = np.array(w)
        V = np.ascontiguousarray(V)
        mx = float(w.max())
        for i in range(k):
            if w[i] < 0 or (0 < w[i] < 1e-12 * mx):
                w[i] = 0.0
        svd_flip_u_only(V)
        idx = np.argsort(w, kind="stable")[::-1]
        self._evals = w[idx]
        self._evecs = np.ascontiguousarray(V[:, idx])
        return self._evecs * np.sqrt(self._evals)

    def fit_transform(self, X, y = None):
        Xa = np.asarray(X, dtype=float)
        n = Xa.shape[0]
        self._fit_X = Xa
        self._brute = _use_brute(Xa, self.n_neighbors)
        dist, ind = _knn(Xa, Xa, self.n_neighbors, True, self._brute)
        adj = np.full((n, n), np.inf)
        for i in range(n):
            for t in range(self.n_neighbors):
                j = ind[i, t]
                d = dist[i, t]
                # undirected: the csgraph symmetrises with the minimum weight
                if d < adj[i, j]:
                    adj[i, j] = d
                if d < adj[j, i]:
                    adj[j, i] = d
        nc, lab = _components(adj)
        if nc > 1:
            # _fix_connected_components: connect each pair of components by
            # their closest pair of points
            D = _euclidean_distances(Xa, Xa, True)
            for a in range(nc):
                for b in range(a + 1, nc):
                    best = np.inf
                    bi = -1
                    bj = -1
                    for i in range(n):
                        if lab[i] != a:
                            continue
                        for j in range(n):
                            if lab[j] == b and D[i, j] < best:
                                best = D[i, j]
                                bi = i
                                bj = j
                    adj[bi, bj] = best
                    adj[bj, bi] = best
        self.dist_matrix_ = _dijkstra_all(adj)
        G = self.dist_matrix_ ** 2
        G *= -0.5
        self.embedding_ = self._kpca_fit(G)
        return self.embedding_

    def fit(self, X, y = None):
        self.fit_transform(X)
        return self

    def transform(self, X):
        Xq = np.asarray(X, dtype=float)
        dist, ind = _knn(self._fit_X, Xq, self.n_neighbors, False, self._brute)
        q = Xq.shape[0]
        n = self._fit_X.shape[0]
        G = np.empty((q, n))
        for i in range(q):
            for j in range(n):
                best = np.inf
                for t in range(self.n_neighbors):
                    v = self.dist_matrix_[ind[i, t], j] + dist[i, t]
                    if v < best:
                        best = v
                G[i, j] = best
        G = G ** 2
        G *= -0.5
        Kc = G - self._K_rows[None, :] - (np.sum(G, axis=1) / float(n))[:, None] + self._K_all
        sa = np.zeros_like(self._evecs)
        for j in range(self._evals.shape[0]):
            if self._evals[j] != 0:
                sa[:, j] = self._evecs[:, j] / np.sqrt(self._evals[j])
        return Kc @ sa

    def reconstruction_error(self) -> float:
        G = -0.5 * self.dist_matrix_ ** 2
        n = G.shape[0]
        Gc = G - np.mean(G, axis=0)[None, :] - np.mean(G, axis=1)[:, None] + float(np.mean(G))
        return float(np.sqrt(float(np.sum(Gc ** 2)) - float(np.sum(self._evals ** 2)))) / n

# ------------------------------------------------------------------ LLE

def _barycenter_weights(X, Y, ind, reg: float):
    n, k = ind.shape
    B = np.empty((n, k))
    v = np.ones(k)
    for i in range(n):
        A = Y[ind[i]]
        C = A - X[i]
        G = C @ C.T
        tr = float(np.trace(G))
        R = reg * tr if tr > 0 else reg
        for a in range(k):
            G[a, a] += R
        w = sl.solve(G, v, assume_a='pos')
        B[i, :] = w / float(np.sum(w))
    return B

def _null_space_dense(M, k: int, k_skip: int):
    w, V = sl.eigh(M, subset_by_index=[k_skip, k + k_skip - 1])
    idx = np.argsort(np.abs(w), kind="stable")
    return np.ascontiguousarray(V[:, idx]), float(np.sum(w))

class LocallyLinearEmbedding:
    """methods standard / modified / ltsa / hessian with the dense eigen
    solver (eigen_solver 'auto' / 'arpack' are computed densely too)."""
    n_neighbors: int
    n_components: int
    reg: float
    method: str
    hessian_tol: float
    modified_tol: float
    embedding_: np.ndarray[float, 2]
    reconstruction_error_: float
    _fit_X: np.ndarray[float, 2]
    _brute: bool

    def __init__(self, *, n_neighbors: int = 5, n_components: int = 2, reg: float = 1e-3,
                 eigen_solver: str = "auto", tol: float = 1e-6, max_iter: int = 100,
                 method: str = "standard", hessian_tol: float = 1e-4, modified_tol: float = 1e-12,
                 neighbors_algorithm: str = "auto", random_state = None, n_jobs = None):
        if method not in ("standard", "hessian", "modified", "ltsa"):
            raise ValueError(f"unrecognized method '{method}'")
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.reg = reg
        self.method = method
        self.hessian_tol = hessian_tol
        self.modified_tol = modified_tol
        self.embedding_ = np.empty((0, 0))
        self.reconstruction_error_ = 0.0
        self._fit_X = np.empty((0, 0))
        self._brute = False

    def fit(self, X, y = None):
        Xa = np.asarray(X, dtype=float)
        N, d_in = Xa.shape
        k = self.n_neighbors
        nc = self.n_components
        if nc > d_in:
            raise ValueError("output dimension must be less than or equal to input dimension")
        if k >= N:
            raise ValueError(f"Expected n_neighbors < n_samples, but n_samples = {N}, n_neighbors = {k}")
        self._fit_X = Xa
        self._brute = _use_brute(Xa, k + 1)
        _, nb1 = _knn(Xa, Xa, k + 1, False, self._brute)
        nb = np.ascontiguousarray(nb1[:, 1:])   # kneighbors(X)[:, 1:] as sklearn (self assumed first)
        M = np.zeros((N, N))
        if self.method == "standard":
            W = _barycenter_weights(Xa, Xa, nb, self.reg)
            Wd = np.zeros((N, N))
            for i in range(N):
                for t in range(k):
                    Wd[i, nb[i, t]] = W[i, t]
            M = Wd.T @ Wd - Wd.T - Wd
            for i in range(N):
                M[i, i] += 1.0
        elif self.method == "hessian":
            dp = nc * (nc + 1) // 2
            if k <= nc + dp:
                raise ValueError("for method='hessian', n_neighbors must be greater than [n_components * (n_components + 3) / 2]")
            Yi = np.empty((k, 1 + nc + dp))
            Yi[:, 0] = 1.0
            use_svd = k > d_in
            for i in range(N):
                Gi = Xa[nb[i]]
                Gi = Gi - Gi.mean(axis=0)
                U = np.empty((0, 0))
                if use_svd:
                    U = sl.svd(Gi, full_matrices=False)[0]
                else:
                    U = np.ascontiguousarray(sl.eigh(Gi @ Gi.T)[1][:, ::-1])
                Yi[:, 1:1 + nc] = U[:, :nc]
                j = 1 + nc
                for a in range(nc):
                    Yi[:, j:j + nc - a] = U[:, a:a + 1] * U[:, a:nc]
                    j += nc - a
                Q, R = sl.qr(Yi)
                w = np.ascontiguousarray(Q[:, nc + 1:])
                S = w.sum(axis=0)
                for t in range(S.shape[0]):
                    if abs(S[t]) < self.hessian_tol:
                        S[t] = 1.0
                w /= S
                WW = w @ w.T
                for a in range(k):
                    for b in range(k):
                        M[nb[i, b], nb[i, a]] += WW[a, b]
        elif self.method == "modified":
            if k < nc:
                raise ValueError("modified LLE requires n_neighbors >= n_components")
            V = np.zeros((N, k, k))
            nev = min(d_in, k)
            evals = np.zeros((N, nev))
            use_svd = k > d_in
            for i in range(N):
                Xn = Xa[nb[i]] - Xa[i]
                if use_svd:
                    Ui, si, _ = sl.svd(Xn, full_matrices=True)
                    V[i] = Ui
                    evals[i] = si ** 2
                else:
                    evi, vi = sl.eigh(Xn @ Xn.T)
                    evals[i] = evi[::-1]
                    V[i] = vi[:, ::-1]
            reg = 1e-3 * evals.sum(axis=1)
            w_reg = np.zeros((N, k))
            for i in range(N):
                tmp = V[i].T @ np.ones(k)
                for t in range(k):
                    if t < nev:
                        tmp[t] /= evals[i, t] + reg[i]
                    else:
                        tmp[t] /= reg[i]
                w_reg[i] = V[i] @ tmp
            w_reg /= w_reg.sum(axis=1)[:, None]
            rho = evals[:, nc:].sum(axis=1) / evals[:, :nc].sum(axis=1)
            eta = float(np.median(rho))
            s_range = np.zeros(N, dtype=int)
            ecs = np.cumsum(evals, axis=1)
            for i in range(N):
                er = ecs[i, -1] / ecs[i, :-1] - 1
                s_range[i] = int(np.searchsorted(np.ascontiguousarray(er[::-1]), eta))
            s_range += k - nev
            for i in range(N):
                s_i = int(s_range[i])
                Vi = np.ascontiguousarray(V[i, :, k - s_i:])
                alpha_i = float(np.linalg.norm(Vi.sum(axis=0))) / np.sqrt(float(s_i))
                h = np.full(s_i, alpha_i) - Vi.T @ np.ones(k)
                nh = float(np.linalg.norm(h))
                if nh < self.modified_tol:
                    h *= 0.0
                else:
                    h /= nh
                Wi = Vi - 2.0 * np.outer(Vi @ h, h) + (1 - alpha_i) * w_reg[i][:, None]
                WW = Wi @ Wi.T
                for a in range(k):
                    for b in range(k):
                        M[nb[i, b], nb[i, a]] += WW[a, b]
                ws = Wi.sum(axis=1)
                for t in range(k):
                    M[i, nb[i, t]] -= ws[t]
                for t in range(k):
                    M[nb[i, t], i] -= ws[t]
                M[i, i] += s_i
        else:
            use_svd = k > d_in
            for i in range(N):
                Xi = Xa[nb[i]]
                Xi = Xi - Xi.mean(axis=0)
                v = np.empty((0, 0))
                if use_svd:
                    v = sl.svd(Xi, full_matrices=True)[0]
                else:
                    v = np.ascontiguousarray(sl.eigh(Xi @ Xi.T)[1][:, ::-1])
                Gi = np.zeros((k, nc + 1))
                Gi[:, 1:] = v[:, :nc]
                Gi[:, 0] = 1.0 / np.sqrt(float(k))
                GG = Gi @ Gi.T
                for a in range(k):
                    for b in range(k):
                        M[nb[i, b], nb[i, a]] -= GG[a, b]
                for t in range(k):
                    M[nb[i, t], nb[i, t]] += 1.0
        self.embedding_, self.reconstruction_error_ = _null_space_dense(M, nc, 1)
        return self

    def fit_transform(self, X, y = None):
        return self.fit(X).embedding_

    def transform(self, X):
        Xq = np.asarray(X, dtype=float)
        _, ind = _knn(self._fit_X, Xq, self.n_neighbors, False, self._brute)
        W = _barycenter_weights(Xq, self._fit_X, ind, self.reg)
        out = np.empty((Xq.shape[0], self.n_components))
        for i in range(Xq.shape[0]):
            out[i] = W[i] @ self.embedding_[ind[i]]
        return out

# ------------------------------------------------------------------ Spectral

class SpectralEmbedding:
    """affinity 'nearest_neighbors' / 'rbf' / 'precomputed'; normalised
    Laplacian eigenmaps solved densely (sklearn's arpack shift-invert gives
    the same eigenvectors; signs fixed by _deterministic_vector_sign_flip)."""
    n_components: int
    affinity: str
    gamma: Optional[float]
    n_neighbors: Optional[int]
    embedding_: np.ndarray[float, 2]
    affinity_matrix_: np.ndarray[float, 2]
    n_neighbors_: int
    gamma_: float

    def __init__(self, n_components: int = 2, *, affinity: str = "nearest_neighbors",
                 gamma: Optional[float] = None, random_state = None, eigen_solver = None,
                 eigen_tol = "auto", n_neighbors: Optional[int] = None, n_jobs = None):
        if affinity not in ("nearest_neighbors", "rbf", "precomputed"):
            raise NotImplementedError(f"SpectralEmbedding: affinity '{affinity}' is not ported")
        self.n_components = n_components
        self.affinity = affinity
        self.gamma = gamma
        self.n_neighbors = n_neighbors
        self.embedding_ = np.empty((0, 0))
        self.affinity_matrix_ = np.empty((0, 0))
        self.n_neighbors_ = 0
        self.gamma_ = 0.0

    def _affinity(self, Xa):
        n = Xa.shape[0]
        if self.affinity == "precomputed":
            return Xa.copy()
        elif self.affinity == "rbf":
            self.gamma_ = 1.0 / Xa.shape[1] if self.gamma is None else self.gamma.__val__()
            K = euclidean_distances_sq(Xa, Xa, True)
            K *= -self.gamma_
            return np.exp(K)
        else:
            self.n_neighbors_ = max(int(n / 10), 1) if self.n_neighbors is None else self.n_neighbors.__val__()
            k = self.n_neighbors_
            # kneighbors_graph(X, k, include_self=True): connectivity
            _, ind = _knn(Xa, Xa, k, False, _use_brute(Xa, k))
            A = np.zeros((n, n))
            for i in range(n):
                for t in range(k):
                    A[i, ind[i, t]] = 1.0
            return 0.5 * (A + A.T)

    def fit(self, X, y = None):
        Xa = np.asarray(X, dtype=float)
        A = self._affinity(Xa)
        self.affinity_matrix_ = A
        self.embedding_ = spectral_embedding(A, n_components=self.n_components)
        return self

    def fit_transform(self, X, y = None):
        return self.fit(X).embedding_

def spectral_embedding(adjacency, *, n_components: int = 8, eigen_solver = None, random_state = None,
                       eigen_tol = "auto", norm_laplacian: bool = True, drop_first: bool = True):
    """Laplacian eigenmaps of a dense adjacency matrix (smallest eigenpairs of
    the (normalised) graph Laplacian computed densely)."""
    A = np.asarray(adjacency, dtype=float)
    n = A.shape[0]
    if A.ndim != 2 or A.shape[1] != n:
        raise ValueError("adjacency must be a square matrix")
    nc = n_components + 1 if drop_first else n_components
    m = A.copy()
    for i in range(n):
        m[i, i] = 0.0
    w = m.sum(axis=0)
    dd = np.ones(n)
    L = np.empty((n, n))
    if norm_laplacian:
        for i in range(n):
            dd[i] = 1.0 if w[i] == 0 else np.sqrt(w[i])
        L = -(m / dd) / dd[:, None]
        for i in range(n):
            L[i, i] = 0.0 if w[i] == 0 else 1.0
    else:
        L = -m
        for i in range(n):
            L[i, i] = w[i]
    ev, V = sl.eigh(L, subset_by_index=[0, nc - 1])
    emb = np.ascontiguousarray(V.T[:nc])
    if norm_laplacian:
        emb = emb / dd
    for r in range(nc):
        best = 0
        for c in range(n):
            if abs(emb[r, c]) > abs(emb[r, best]):
                best = c
        if emb[r, best] < 0:
            emb[r] = -emb[r]
    if drop_first:
        return np.ascontiguousarray(emb[1:nc].T)
    return np.ascontiguousarray(emb[:nc].T)

def smacof(dissimilarities, *, metric: bool = True, n_components: int = 2, init = None,
           n_init: int = 8, n_jobs = None, max_iter: int = 300, verbose: int = 0,
           eps: float = 1e-6, random_state: Optional[int] = None,
           return_n_iter: Literal[bool] = False, normalized_stress = "auto"):
    D = np.asarray(dissimilarities, dtype=float)
    ns = False
    if isinstance(normalized_stress, str):
        ns = not metric
    else:
        ns = bool(normalized_stress)
    has_init = False
    init_arr = np.empty((0, 0))
    if init is not None:
        has_init = True
        init_arr = np.array(init, dtype=float)
    rs = np.random.RandomState(random_state)
    ni = 1 if has_init else n_init
    best_stress = 0.0
    best_pos = np.empty((0, 0))
    best_iter = 0
    for t in range(ni):
        pos, stress, nit = _smacof_single(D, metric, n_components, init_arr, has_init, max_iter,
                                          eps, rs, ns)
        if t == 0 or stress < best_stress:
            best_stress = stress
            best_pos = pos.copy()
            best_iter = nit
    if return_n_iter:
        return best_pos, best_stress, best_iter
    return best_pos, best_stress

def locally_linear_embedding(X, *, n_neighbors: int, n_components: int, reg: float = 1e-3,
                             eigen_solver: str = "auto", tol: float = 1e-6, max_iter: int = 100,
                             method: str = "standard", hessian_tol: float = 1e-4,
                             modified_tol: float = 1e-12, random_state = None, n_jobs = None):
    est = LocallyLinearEmbedding(n_neighbors=n_neighbors, n_components=n_components, reg=reg,
                                 eigen_solver=eigen_solver, tol=tol, max_iter=max_iter, method=method,
                                 hessian_tol=hessian_tol, modified_tol=modified_tol)
    est.fit(X)
    return est.embedding_, est.reconstruction_error_

# ------------------------------------------------------------------ trustworthiness

def trustworthiness(X, X_embedded, *, n_neighbors: int = 5, metric: str = "euclidean") -> float:
    Xa = np.asarray(X, dtype=float)
    Xe = np.asarray(X_embedded, dtype=float)
    n = Xa.shape[0]
    if n_neighbors >= n / 2:
        raise ValueError(f"n_neighbors ({n_neighbors}) should be less than n_samples / 2 ({n / 2})")
    D = np.empty((0, 0))
    if metric == "precomputed":
        D = Xa.copy()
    elif metric == "euclidean":
        D = _euclidean_distances(Xa, Xa, True)
    else:
        raise NotImplementedError("trustworthiness: metric must be 'euclidean' or 'precomputed'")
    for i in range(n):
        D[i, i] = np.inf
    _, ind_e = _knn(Xe, Xe, n_neighbors, True, _use_brute(Xe, n_neighbors))
    inv = np.zeros((n, n), dtype=int)
    for i in range(n):
        order = np.argsort(D[i])
        for r in range(n):
            inv[i, order[r]] = r + 1
    t = 0
    for i in range(n):
        for j in range(n_neighbors):
            rk = inv[i, ind_e[i, j]] - n_neighbors
            if rk > 0:
                t += rk
    return 1.0 - t * (2.0 / (n * n_neighbors * (2.0 * n - 3.0 * n_neighbors - 1.0)))
