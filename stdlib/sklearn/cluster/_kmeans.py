# Ported from scikit-learn 1.9.1 (BSD-3-Clause)

# KMeans (Lloyd; 'elkan' is accepted and runs the exact same Lloyd
# iterations), k-means++ seeding and kmeans_plusplus. Random streams follow
# numpy's legacy RandomState exactly as scikit-learn consumes them.

from ..base import estimator_repr as _estimator_repr
import numpy as np
from numpy.random.legacy import RandomState, check_random_state

def _as2d(X) -> np.ndarray[float, 2]:
    A = np.ascontiguousarray(np.asarray(X, dtype=float))
    if static.len(A.shape) != 2:
        compile_error("Expected a 2D array")
    return A

def _row_norms_sq(X: np.ndarray[float, 2]) -> np.ndarray[float, 1]:
    n, p = X.shape
    out = np.empty(n)
    for i in range(n):
        s = 0.0
        for k in range(p):
            s += X[i, k] * X[i, k]
        out[i] = s
    return out

def _sqdist_rows(A: np.ndarray[float, 2], An: np.ndarray[float, 1],
                 B: np.ndarray[float, 2], Bn: np.ndarray[float, 1]) -> np.ndarray[float, 2]:
    # sklearn _euclidean_distances(squared=True): -2 A.B^T + |a|^2 + |b|^2, >= 0
    D = A @ B.T
    na, nb = D.shape
    for i in range(na):
        for j in range(nb):
            v = -2.0 * D[i, j] + An[i] + Bn[j]
            D[i, j] = v if v > 0.0 else 0.0
    return D

def _sample_weight(sample_weight, n: int) -> np.ndarray[float, 1]:
    if sample_weight is None:
        return np.ones(n)
    else:
        w = np.ascontiguousarray(np.asarray(sample_weight, dtype=float)).copy()
        if w.shape[0] != n:
            raise ValueError("sample_weight.shape == {}, expected {}!".format(w.shape[0], n))
        return w

def _kmeans_plusplus(X: np.ndarray[float, 2], n_clusters: int, x_squared_norms: np.ndarray[float, 1],
                     sample_weight: np.ndarray[float, 1], random_state: RandomState,
                     n_local_trials: Optional[int] = None):
    n_samples, n_features = X.shape
    centers = np.empty((n_clusters, n_features))
    trials = 2 + int(np.log(n_clusters))
    if n_local_trials is not None:
        trials = n_local_trials.__val__()
    center_id = random_state.choice(n_samples, p=sample_weight / sample_weight.sum())
    indices = np.full(n_clusters, -1, dtype=int)
    centers[0] = X[center_id]
    indices[0] = center_id
    c0 = X[center_id:center_id + 1]
    closest = _sqdist_rows(c0, _row_norms_sq(c0), X, x_squared_norms)[0].copy()
    current_pot = float(np.dot(closest, sample_weight))
    for c in range(1, n_clusters):
        rand_vals = random_state.uniform(size=trials) * current_pot
        cand = np.searchsorted(np.cumsum(sample_weight * closest), rand_vals)
        for t in range(trials):
            if cand[t] > n_samples - 1:
                cand[t] = n_samples - 1
        Xc = np.ascontiguousarray(X[cand])
        dc = _sqdist_rows(Xc, x_squared_norms[cand], X, x_squared_norms)
        best = 0
        best_pot = 0.0
        for t in range(trials):
            for j in range(n_samples):
                if closest[j] < dc[t, j]:
                    dc[t, j] = closest[j]
            pot = float(np.dot(dc[t], sample_weight))
            if t == 0 or pot < best_pot:
                best_pot = pot
                best = t
        current_pot = best_pot
        closest = dc[best].copy()
        bc = int(cand[best])
        centers[c] = X[bc]
        indices[c] = bc
    return centers, indices

def kmeans_plusplus(X, n_clusters: int, *, sample_weight = None, x_squared_norms = None,
                    random_state = None, n_local_trials: Optional[int] = None):
    Xa = _as2d(X)
    if Xa.shape[0] < n_clusters:
        raise ValueError(f"n_samples={Xa.shape[0]} should be >= n_clusters={n_clusters}.")
    sw = _sample_weight(sample_weight, Xa.shape[0])
    xn = _row_norms_sq(Xa)
    if x_squared_norms is not None:
        xn = np.asarray(x_squared_norms, dtype=float)
    return _kmeans_plusplus(Xa, n_clusters, xn, sw, check_random_state(random_state), n_local_trials)

def _assign(X: np.ndarray[float, 2], centers: np.ndarray[float, 2], labels: np.ndarray[int, 1]):
    # labels only (lloyd_iter with update_centers=False)
    n = X.shape[0]
    k = centers.shape[0]
    cn = _row_norms_sq(centers)
    G = X @ centers.T
    for i in range(n):
        lab = 0
        m = cn[0] - 2.0 * G[i, 0]
        for j in range(1, k):
            d = cn[j] - 2.0 * G[i, j]
            if d < m:
                m = d
                lab = j
        labels[i] = lab

def _euclid_sq(X: np.ndarray[float, 2], i: int, C: np.ndarray[float, 2], j: int) -> float:
    p = X.shape[1]
    r = 0.0
    q = p // 4
    k = 0
    for _ in range(q):
        a0 = X[i, k] - C[j, k]; a1 = X[i, k + 1] - C[j, k + 1]
        a2 = X[i, k + 2] - C[j, k + 2]; a3 = X[i, k + 3] - C[j, k + 3]
        r += a0 * a0 + a1 * a1 + a2 * a2 + a3 * a3
        k += 4
    while k < p:
        a = X[i, k] - C[j, k]
        r += a * a
        k += 1
    return r

def _inertia(X: np.ndarray[float, 2], sw: np.ndarray[float, 1], centers: np.ndarray[float, 2],
             labels: np.ndarray[int, 1]) -> float:
    s = 0.0
    for i in range(X.shape[0]):
        s += _euclid_sq(X, i, centers, labels[i]) * sw[i]
    return s

def _lloyd_iter(X: np.ndarray[float, 2], sw: np.ndarray[float, 1], centers_old: np.ndarray[float, 2],
                centers_new: np.ndarray[float, 2], weight: np.ndarray[float, 1],
                labels: np.ndarray[int, 1], center_shift: np.ndarray[float, 1]):
    n, p = X.shape
    k = centers_old.shape[0]
    _assign(X, centers_old, labels)
    for j in range(k):
        weight[j] = 0.0
        for f in range(p):
            centers_new[j, f] = 0.0
    for i in range(n):
        lab = labels[i]
        weight[lab] += sw[i]
        for f in range(p):
            centers_new[lab, f] += X[i, f] * sw[i]
    # _relocate_empty_clusters_dense
    empty = List[int]()
    for j in range(k):
        if weight[j] == 0.0:
            empty.append(j)
    if len(empty) > 0:
        dist = np.empty(n)
        mx = 0.0
        for i in range(n):
            dist[i] = _euclid_sq(X, i, centers_old, labels[i])
            if dist[i] > mx:
                mx = dist[i]
        if mx != 0.0:
            far = np.argsort(-dist, kind="stable")
            for idx in range(len(empty)):
                new_id = empty[idx]
                fi = int(far[idx])
                w = sw[fi]
                old_id = labels[fi]
                for f in range(p):
                    centers_new[old_id, f] -= X[fi, f] * w
                    centers_new[new_id, f] = X[fi, f] * w
                weight[new_id] = w
                weight[old_id] -= w
    # _average_centers
    amax = 0
    for j in range(1, k):
        if weight[j] > weight[amax]:
            amax = j
    for j in range(k):
        if weight[j] > 0:
            alpha = 1.0 / weight[j]
            for f in range(p):
                centers_new[j, f] *= alpha
        else:
            for f in range(p):
                centers_new[j, f] = centers_new[amax, f]
    for j in range(k):
        center_shift[j] = np.sqrt(_euclid_sq(centers_new, j, centers_old, j))

def _kmeans_single_lloyd(X: np.ndarray[float, 2], sw: np.ndarray[float, 1],
                         centers_init: np.ndarray[float, 2], max_iter: int, tol: float):
    k = centers_init.shape[0]
    centers = centers_init.copy()
    centers_new = np.zeros_like(centers)
    labels = np.full(X.shape[0], -1, dtype=int)
    labels_old = labels.copy()
    weight = np.zeros(k)
    shift = np.zeros(k)
    strict = False
    it = 0
    for i in range(max_iter):
        it = i
        _lloyd_iter(X, sw, centers, centers_new, weight, labels, shift)
        centers, centers_new = centers_new, centers
        same = True
        for t in range(labels.shape[0]):
            if labels[t] != labels_old[t]:
                same = False
                break
        if same:
            strict = True
            break
        tot = 0.0
        for j in range(k):
            tot += shift[j] * shift[j]
        if tot <= tol:
            break
        for t in range(labels.shape[0]):
            labels_old[t] = labels[t]
    if not strict:
        _assign(X, centers, labels)
    inertia = _inertia(X, sw, centers, labels)
    return labels, inertia, centers, it + 1

def _is_same_clustering(l1: np.ndarray[int, 1], l2: np.ndarray[int, 1], k: int) -> bool:
    mapping = [-1] * k
    for i in range(l1.shape[0]):
        if mapping[l1[i]] == -1:
            mapping[l1[i]] = l2[i]
        elif mapping[l1[i]] != l2[i]:
            return False
    return True

def _tolerance(X: np.ndarray[float, 2], tol: float) -> float:
    if tol == 0:
        return 0.0
    return float(np.mean(np.var(X, axis=0))) * tol

class KMeans:
    def __repr__(self) -> str:
        return _estimator_repr(self)

    """init: 'k-means++', 'random' or an (n_clusters, n_features) array."""
    n_clusters: int
    init: str
    init_array: np.ndarray[float, 2]
    n_init: str
    n_init_value: int
    max_iter: int
    tol: float
    random_state: Optional[int]
    algorithm: str
    cluster_centers_: np.ndarray[float, 2]
    labels_: np.ndarray[int, 1]
    inertia_: float
    n_iter_: int
    n_features_in_: int
    _rng: Optional[RandomState]

    def __init__(self, n_clusters: int = 8, *, init = "k-means++", n_init = "auto",
                 max_iter: int = 300, tol: float = 1e-4, verbose: int = 0,
                 random_state = None, copy_x: bool = True, algorithm: str = "lloyd"):
        self.n_clusters = n_clusters
        if isinstance(init, str):
            if init != "k-means++" and init != "random":
                raise ValueError(f"init should be 'k-means++', 'random' or an array, got {init}")
            self.init = init
            self.init_array = np.empty((0, 0))
        else:
            self.init = "array"
            self.init_array = _as2d(init).copy()
        if isinstance(n_init, str):
            if n_init != "auto":
                raise ValueError("n_init should be 'auto' or a positive int")
            self.n_init = "auto"
            self.n_init_value = 0
        else:
            self.n_init = ""
            self.n_init_value = int(n_init)
        if algorithm != "lloyd" and algorithm != "elkan":
            raise ValueError(f"algorithm should be 'lloyd' or 'elkan', got {algorithm}")
        self.max_iter = max_iter
        self.tol = tol
        self._rng = None
        if isinstance(random_state, RandomState):
            self._rng = random_state
            self.random_state = None
        elif isinstance(random_state, NoneType):
            self.random_state = None
        else:
            self.random_state = random_state
        self.algorithm = algorithm
        self.cluster_centers_ = np.empty((0, 0))
        self.labels_ = np.empty(0, dtype=int)
        self.inertia_ = 0.0
        self.n_iter_ = 0
        self.n_features_in_ = 0

    def _get_rng(self) -> RandomState:
        if self._rng is not None:
            return self._rng
        return check_random_state(self.random_state)

    def _init_centroids(self, X: np.ndarray[float, 2], xn: np.ndarray[float, 1],
                        rs: RandomState, sw: np.ndarray[float, 1], X_mean: np.ndarray[float, 1]):
        if self.init == "k-means++":
            c, _ = _kmeans_plusplus(X, self.n_clusters, xn, sw, rs)
            return c
        elif self.init == "random":
            seeds = rs.choice(X.shape[0], size=self.n_clusters, replace=False, p=sw / sw.sum())
            return np.ascontiguousarray(X[seeds])
        else:
            return self.init_array - X_mean

    def fit(self, X, y = None, sample_weight = None):
        Xa = _as2d(X).copy()
        n, p = Xa.shape
        if n < self.n_clusters:
            raise ValueError(f"n_samples={n} should be >= n_clusters={self.n_clusters}.")
        if self.init == "array" and (self.init_array.shape[0] != self.n_clusters or self.init_array.shape[1] != p):
            raise ValueError("The shape of the initial centers does not match the number of clusters / features")
        tol = _tolerance(Xa, self.tol)
        n_init = self.n_init_value
        if self.n_init == "auto":
            n_init = 10 if self.init == "random" else 1
        if self.init == "array":
            n_init = 1
        rs = self._get_rng()
        sw = _sample_weight(sample_weight, n)
        X_mean = np.mean(Xa, axis=0)
        Xa -= X_mean
        xn = _row_norms_sq(Xa)
        have = False
        best_labels = np.empty(0, dtype=int)
        best_centers = np.empty((0, 0))
        best_inertia = 0.0
        best_iter = 0
        for _ in range(n_init):
            ci = self._init_centroids(Xa, xn, rs, sw, X_mean)
            labels, inertia, centers, it = _kmeans_single_lloyd(Xa, sw, ci, self.max_iter, tol)
            if not have or (inertia < best_inertia and not _is_same_clustering(labels, best_labels, self.n_clusters)):
                have = True
                best_labels = labels
                best_centers = centers
                best_inertia = inertia
                best_iter = it
        self.cluster_centers_ = best_centers + X_mean
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        self.n_iter_ = best_iter
        self.n_features_in_ = p
        return self

    def predict(self, X):
        Xa = _as2d(X)
        labels = np.empty(Xa.shape[0], dtype=int)
        _assign(Xa, self.cluster_centers_, labels)
        return labels

    def fit_predict(self, X, y = None, sample_weight = None):
        return self.fit(X, sample_weight=sample_weight).labels_

    def transform(self, X):
        Xa = _as2d(X)
        D = _sqdist_rows(Xa, _row_norms_sq(Xa), self.cluster_centers_, _row_norms_sq(self.cluster_centers_))
        return np.sqrt(D)

    def fit_transform(self, X, y = None, sample_weight = None):
        return self.fit(X, sample_weight=sample_weight).transform(X)

    def score(self, X, y = None, sample_weight = None) -> float:
        Xa = _as2d(X)
        labels = np.empty(Xa.shape[0], dtype=int)
        _assign(Xa, self.cluster_centers_, labels)
        return -_inertia(Xa, _sample_weight(sample_weight, Xa.shape[0]), self.cluster_centers_, labels)
