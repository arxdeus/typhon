# Ported from scikit-learn 1.9.1 (BSD-3-Clause)

# MiniBatchKMeans: k-means++ / random init on an init_size subsample, best
# of n_init by validation inertia, weighted minibatch sampling with
# replacement, per-center learning-rate updates, low-count reassignment and
# EWA-inertia early stopping. The RandomState stream follows sklearn's.

from ..base import estimator_repr as _estimator_repr
import numpy as np
from numpy.random.legacy import RandomState, check_random_state
from ._kmeans import (_as2d, _row_norms_sq, _sample_weight, _kmeans_plusplus, _assign,
                      _inertia, _tolerance)

def _labels_inertia(X: np.ndarray[float, 2], sw: np.ndarray[float, 1], centers: np.ndarray[float, 2]):
    labels = np.empty(X.shape[0], dtype=int)
    _assign(X, centers, labels)
    return labels, _inertia(X, sw, centers, labels)

def _minibatch_update_dense(X: np.ndarray[float, 2], sw: np.ndarray[float, 1],
                            centers_old: np.ndarray[float, 2], centers_new: np.ndarray[float, 2],
                            weight_sums: np.ndarray[float, 1], labels: np.ndarray[int, 1]):
    n = X.shape[0]
    k, p = centers_old.shape
    for c in range(k):
        wsum = 0.0
        for i in range(n):
            if labels[i] == c:
                wsum += sw[i]
        if wsum > 0.0:
            for f in range(p):
                centers_new[c, f] = centers_old[c, f] * weight_sums[c]
            for i in range(n):
                if labels[i] == c:
                    for f in range(p):
                        centers_new[c, f] += X[i, f] * sw[i]
            weight_sums[c] += wsum
            alpha = 1.0 / weight_sums[c]
            for f in range(p):
                centers_new[c, f] *= alpha
        else:
            for f in range(p):
                centers_new[c, f] = centers_old[c, f]

def _mini_batch_step(X: np.ndarray[float, 2], sw: np.ndarray[float, 1], centers: np.ndarray[float, 2],
                     centers_new: np.ndarray[float, 2], weight_sums: np.ndarray[float, 1],
                     rs: RandomState, random_reassign: bool, reassignment_ratio: float) -> float:
    labels, inertia = _labels_inertia(X, sw, centers)
    _minibatch_update_dense(X, sw, centers, centers_new, weight_sums, labels)
    if random_reassign and reassignment_ratio > 0.0:
        k = weight_sums.shape[0]
        mx = float(weight_sums.max())
        to_re = weight_sums < reassignment_ratio * mx
        if float(to_re.sum()) > 0.5 * X.shape[0]:
            order = np.argsort(weight_sums)
            for t in range(int(0.5 * X.shape[0]), k):
                to_re[order[t]] = False
        nre = int(to_re.sum())
        if nre > 0:
            new_c = rs.choice(X.shape[0], replace=False, size=nre)
            q = 0
            for c in range(k):
                if to_re[c]:
                    centers_new[c] = X[new_c[q]]
                    q += 1
        mn = np.inf
        for c in range(k):
            if not to_re[c] and weight_sums[c] < mn:
                mn = weight_sums[c]
        for c in range(k):
            if to_re[c]:
                weight_sums[c] = mn
    return inertia

class MiniBatchKMeans:
    def __repr__(self) -> str:
        return _estimator_repr(self)

    n_clusters: int
    init: str
    init_array: np.ndarray[float, 2]
    max_iter: int
    batch_size: int
    compute_labels: bool
    random_state: Optional[int]
    tol: float
    max_no_improvement: Optional[int]
    init_size: Optional[int]
    n_init: int
    reassignment_ratio: float
    cluster_centers_: np.ndarray[float, 2]
    labels_: np.ndarray[int, 1]
    inertia_: float
    n_iter_: int
    n_steps_: int
    n_features_in_: int
    _counts: np.ndarray[float, 1]
    _ewa_inertia: Optional[float]
    _ewa_inertia_min: Optional[float]
    _no_improvement: int
    _n_since_last_reassign: int
    _batch_size: int
    _tol: float
    _rng: Optional[RandomState]

    def __init__(self, n_clusters: int = 8, *, init = "k-means++", max_iter: int = 100,
                 batch_size: int = 1024, verbose: int = 0, compute_labels: bool = True,
                 random_state = None, tol: float = 0.0, max_no_improvement: Optional[int] = 10,
                 init_size: Optional[int] = None, n_init = "auto", reassignment_ratio: float = 0.01):
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
            self.n_init = -1
        else:
            self.n_init = int(n_init)
        if reassignment_ratio < 0.0:
            raise ValueError(f"reassignment_ratio should be >= 0, got {reassignment_ratio} instead.")
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.compute_labels = compute_labels
        self._rng = None
        if isinstance(random_state, RandomState):
            self._rng = random_state
            self.random_state = None
        elif isinstance(random_state, NoneType):
            self.random_state = None
        else:
            self.random_state = random_state
        self.tol = tol
        self.max_no_improvement = max_no_improvement
        self.init_size = init_size
        self.reassignment_ratio = reassignment_ratio
        self.cluster_centers_ = np.empty((0, 0))
        self.labels_ = np.empty(0, dtype=int)
        self.inertia_ = 0.0
        self.n_iter_ = 0
        self.n_steps_ = 0
        self.n_features_in_ = 0
        self._counts = np.empty(0)
        self._ewa_inertia = None
        self._ewa_inertia_min = None
        self._no_improvement = 0
        self._n_since_last_reassign = 0
        self._batch_size = 0
        self._tol = 0.0

    def _init_centroids(self, X: np.ndarray[float, 2], xn: np.ndarray[float, 1], rs: RandomState,
                        init_size: int, sw: np.ndarray[float, 1]):
        n = X.shape[0]
        Xs = X
        xs = xn
        ws = sw
        if init_size < n:
            idx = rs.randint(0, n, init_size)
            Xs = np.ascontiguousarray(X[idx])
            xs = xn[idx]
            ws = sw[idx]
        if self.init == "k-means++":
            c, _ = _kmeans_plusplus(Xs, self.n_clusters, xs, ws, rs)
            return c
        elif self.init == "random":
            seeds = rs.choice(Xs.shape[0], size=self.n_clusters, replace=False, p=ws / ws.sum())
            return np.ascontiguousarray(Xs[seeds])
        else:
            return self.init_array.copy()

    def _convergence(self, step: int, n_samples: int, csd: float, batch_inertia: float) -> bool:
        bi = batch_inertia / self._batch_size
        step = step + 1
        if step == 1:
            return False
        if self._ewa_inertia is None:
            self._ewa_inertia = bi
        else:
            alpha = min(self._batch_size * 2.0 / (n_samples + 1), 1.0)
            self._ewa_inertia = self._ewa_inertia.__val__() * (1 - alpha) + bi * alpha
        if self._tol > 0.0 and csd <= self._tol:
            return True
        ewa = self._ewa_inertia.__val__()
        if self._ewa_inertia_min is None:
            self._no_improvement = 0
            self._ewa_inertia_min = ewa
        elif ewa < self._ewa_inertia_min.__val__():
            self._no_improvement = 0
            self._ewa_inertia_min = ewa
        else:
            self._no_improvement += 1
        if self.max_no_improvement is not None:
            if self._no_improvement >= self.max_no_improvement.__val__():
                return True
        return False

    def _random_reassign(self) -> bool:
        self._n_since_last_reassign += self._batch_size
        if (self._counts == 0.0).any() or self._n_since_last_reassign >= 10 * self.n_clusters:
            self._n_since_last_reassign = 0
            return True
        return False

    def fit(self, X, y = None, sample_weight = None):
        Xa = _as2d(X)
        n, p = Xa.shape
        if n < self.n_clusters:
            raise ValueError(f"n_samples={n} should be >= n_clusters={self.n_clusters}.")
        self._tol = _tolerance(Xa, self.tol)
        n_init = self.n_init
        if n_init < 0:
            n_init = 3 if self.init == "random" else 1
        if self.init == "array":
            n_init = 1
        self._batch_size = min(self.batch_size, n)
        isz = 0
        if self.init_size is None:
            isz = 3 * self._batch_size
            if isz < self.n_clusters:
                isz = 3 * self.n_clusters
        else:
            isz = self.init_size.__val__()
            if isz < self.n_clusters:
                isz = 3 * self.n_clusters
        isz = min(isz, n)
        rs = self._rng.__val__() if self._rng is not None else check_random_state(self.random_state)
        sw = _sample_weight(sample_weight, n)
        xn = _row_norms_sq(Xa)
        vidx = rs.randint(0, n, isz)
        Xv = np.ascontiguousarray(Xa[vidx])
        swv = sw[vidx]
        have = False
        best = 0.0
        centers = np.empty((0, 0))
        for _ in range(n_init):
            cc = self._init_centroids(Xa, xn, rs, isz, sw)
            _, inert = _labels_inertia(Xv, swv, cc)
            if not have or inert < best:
                centers = cc
                best = inert
                have = True
        centers_new = np.empty(centers.shape)
        self._counts = np.zeros(self.n_clusters)
        self._ewa_inertia = None
        self._ewa_inertia_min = None
        self._no_improvement = 0
        self._n_since_last_reassign = 0
        sum_w = float(sw.sum())
        n_steps = (self.max_iter * n) // self._batch_size
        nsw = sw / sum_w
        unit = np.ones(self._batch_size)
        i = 0
        step_done = 0
        while i < n_steps:
            mb = rs.choice(n, self._batch_size, p=nsw, replace=True)
            Xb = np.ascontiguousarray(Xa[mb])
            bi = _mini_batch_step(Xb, unit, centers, centers_new, self._counts, rs,
                                  self._random_reassign(), self.reassignment_ratio)
            csd = 0.0
            if self._tol > 0.0:
                csd = float(((centers_new - centers) ** 2).sum())
            tmp = centers
            centers = centers_new
            centers_new = tmp
            step_done = i
            if self._convergence(i, n, csd, bi):
                break
            i += 1
        self.cluster_centers_ = centers
        self.n_steps_ = step_done + 1
        self.n_iter_ = int(np.ceil(((step_done + 1) * self._batch_size) / n))
        self.n_features_in_ = p
        if self.compute_labels:
            self.labels_, self.inertia_ = _labels_inertia(Xa, sw, centers)
        else:
            self.inertia_ = self._ewa_inertia.__val__() * sum_w
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
        D = np.empty((Xa.shape[0], self.cluster_centers_.shape[0]))
        for i in range(Xa.shape[0]):
            for j in range(self.cluster_centers_.shape[0]):
                D[i, j] = np.sqrt(float(((Xa[i] - self.cluster_centers_[j]) ** 2).sum()))
        return D

    def score(self, X, y = None, sample_weight = None) -> float:
        Xa = _as2d(X)
        _, inert = _labels_inertia(Xa, _sample_weight(sample_weight, Xa.shape[0]), self.cluster_centers_)
        return -inert
