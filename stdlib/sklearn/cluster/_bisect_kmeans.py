# Ported from scikit-learn 1.9.1 (BSD-3-Clause)

# BisectingKMeans: the bisecting tree is stored in flat lists (node arrays)
# and leaves are visited depth-first, left before right, like the reference.

from ..base import estimator_repr as _estimator_repr
import numpy as np
from numpy.random.legacy import RandomState, check_random_state
from ._kmeans import (_as2d, _row_norms_sq, _sample_weight, _kmeans_plusplus, _kmeans_single_lloyd,
                      _euclid_sq, _inertia, _assign, _tolerance, _sqdist_rows)

class BisectingKMeans:
    def __repr__(self) -> str:
        return _estimator_repr(self)

    n_clusters: int
    init: str
    n_init: int
    max_iter: int
    tol: float
    random_state: Optional[int]
    algorithm: str
    bisecting_strategy: str
    cluster_centers_: np.ndarray[float, 2]
    labels_: np.ndarray[int, 1]
    inertia_: float
    n_features_in_: int
    X_mean_: np.ndarray[float, 1]
    # flat tree
    t_center: List[np.ndarray[float, 1]]
    t_indices: List[np.ndarray[int, 1]]
    t_score: List[float]
    t_left: List[int]
    t_right: List[int]
    t_label: List[int]

    def __init__(self, n_clusters: int = 8, *, init: str = "random", n_init: int = 1,
                 random_state = None, max_iter: int = 300, verbose: int = 0, tol: float = 1e-4,
                 copy_x: bool = True, algorithm: str = "lloyd",
                 bisecting_strategy: str = "biggest_inertia"):
        if init != "k-means++" and init != "random":
            raise ValueError(f"init should be 'k-means++' or 'random', got {init}")
        if algorithm != "lloyd" and algorithm != "elkan":
            raise ValueError(f"algorithm should be 'lloyd' or 'elkan', got {algorithm}")
        if bisecting_strategy != "biggest_inertia" and bisecting_strategy != "largest_cluster":
            raise ValueError("bisecting_strategy should be 'biggest_inertia' or 'largest_cluster'")
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        if random_state is None:
            self.random_state = None
        else:
            self.random_state = int(random_state)
        self.algorithm = algorithm
        self.bisecting_strategy = bisecting_strategy
        self.cluster_centers_ = np.empty((0, 0))
        self.labels_ = np.empty(0, dtype=int)
        self.inertia_ = 0.0
        self.n_features_in_ = 0
        self.X_mean_ = np.empty(0)
        self.t_center = []
        self.t_indices = []
        self.t_score = []
        self.t_left = []
        self.t_right = []
        self.t_label = []

    def _add_node(self, center: np.ndarray[float, 1], indices: np.ndarray[int, 1], score: float) -> int:
        self.t_center.append(center)
        self.t_indices.append(indices)
        self.t_score.append(score)
        self.t_left.append(-1)
        self.t_right.append(-1)
        self.t_label.append(-1)
        return len(self.t_score) - 1

    def _leaves(self, node: int, out: List[int]):
        if self.t_left[node] < 0:
            out.append(node)
        else:
            self._leaves(self.t_left[node], out)
            self._leaves(self.t_right[node], out)

    def _to_bisect(self) -> int:
        leaves = List[int]()
        self._leaves(0, leaves)
        best = leaves[0]
        mx = self.t_score[best]
        for i in range(1, len(leaves)):
            if self.t_score[leaves[i]] > mx:
                mx = self.t_score[leaves[i]]
                best = leaves[i]
        return best

    def _bisect(self, X: np.ndarray[float, 2], xn: np.ndarray[float, 1], sw: np.ndarray[float, 1],
                rs: RandomState, tol: float, node: int):
        idx = self.t_indices[node]
        Xs = np.ascontiguousarray(X[idx])
        xns = xn[idx]
        sws = sw[idx]
        have = False
        best_labels = np.empty(0, dtype=int)
        best_centers = np.empty((0, 0))
        best_inertia = 0.0
        for _ in range(self.n_init):
            ci = np.empty((0, 0))
            if self.init == "k-means++":
                c, _i = _kmeans_plusplus(Xs, 2, xns, sws, rs)
                ci = c
            else:
                seeds = rs.choice(Xs.shape[0], size=2, replace=False, p=sws / sws.sum())
                ci = np.ascontiguousarray(Xs[seeds])
            labels, inertia, centers, _it = _kmeans_single_lloyd(Xs, sws, ci, self.max_iter, tol)
            if not have or inertia < best_inertia * (1.0 - 1e-6):
                have = True
                best_labels = labels
                best_centers = centers
                best_inertia = inertia
        s0 = 0.0
        s1 = 0.0
        n0 = 0
        n1 = 0
        for i in range(Xs.shape[0]):
            if best_labels[i] == 0:
                s0 += _euclid_sq(Xs, i, best_centers, 0) * sws[i]
                n0 += 1
            else:
                s1 += _euclid_sq(Xs, i, best_centers, 1) * sws[i]
                n1 += 1
        if self.bisecting_strategy == "largest_cluster":
            s0 = float(n0)
            s1 = float(n1)
        i0 = np.empty(n0, dtype=int)
        i1 = np.empty(n1, dtype=int)
        a = 0
        b = 0
        for i in range(Xs.shape[0]):
            if best_labels[i] == 0:
                i0[a] = idx[i]
                a += 1
            else:
                i1[b] = idx[i]
                b += 1
        l = self._add_node(best_centers[0].copy(), i0, s0)
        r = self._add_node(best_centers[1].copy(), i1, s1)
        self.t_left[node] = l
        self.t_right[node] = r
        self.t_indices[node] = np.empty(0, dtype=int)

    def fit(self, X, y = None, sample_weight = None):
        Xa = _as2d(X).copy()
        n, p = Xa.shape
        if n < self.n_clusters:
            raise ValueError(f"n_samples={n} should be >= n_clusters={self.n_clusters}.")
        rs = check_random_state(self.random_state)
        sw = _sample_weight(sample_weight, n)
        tol = _tolerance(Xa, self.tol)
        X_mean = np.mean(Xa, axis=0)
        Xa -= X_mean
        self.t_center = []
        self.t_indices = []
        self.t_score = []
        self.t_left = []
        self.t_right = []
        self.t_label = []
        self._add_node(np.mean(Xa, axis=0), np.arange(n), 0.0)
        xn = _row_norms_sq(Xa)
        for _ in range(self.n_clusters - 1):
            node = self._to_bisect()
            self._bisect(Xa, xn, sw, rs, tol, node)
        leaves = List[int]()
        self._leaves(0, leaves)
        labels = np.full(n, -1, dtype=int)
        centers = np.empty((self.n_clusters, p))
        for i in range(len(leaves)):
            nd = leaves[i]
            ind = self.t_indices[nd]
            for t in range(ind.shape[0]):
                labels[ind[t]] = i
            centers[i] = self.t_center[nd]
            self.t_label[nd] = i
        Xa += X_mean
        centers += X_mean
        self.labels_ = labels
        self.cluster_centers_ = centers
        self.X_mean_ = X_mean
        self.n_features_in_ = p
        self.inertia_ = _inertia(Xa, sw, centers, labels)
        return self

    def _predict_rec(self, X: np.ndarray[float, 2], rows: np.ndarray[int, 1], node: int,
                     out: np.ndarray[int, 1]):
        if self.t_left[node] < 0:
            for t in range(rows.shape[0]):
                out[rows[t]] = self.t_label[node]
            return
        c = np.empty((2, X.shape[1]))
        c[0] = self.t_center[self.t_left[node]] + self.X_mean_
        c[1] = self.t_center[self.t_right[node]] + self.X_mean_
        Xs = np.ascontiguousarray(X[rows])
        lab = np.empty(Xs.shape[0], dtype=int)
        _assign(Xs, c, lab)
        m0 = 0
        for t in range(lab.shape[0]):
            if lab[t] == 0:
                m0 += 1
        r0 = np.empty(m0, dtype=int)
        r1 = np.empty(lab.shape[0] - m0, dtype=int)
        a = 0
        b = 0
        for t in range(lab.shape[0]):
            if lab[t] == 0:
                r0[a] = rows[t]
                a += 1
            else:
                r1[b] = rows[t]
                b += 1
        self._predict_rec(X, r0, self.t_left[node], out)
        self._predict_rec(X, r1, self.t_right[node], out)

    def predict(self, X):
        Xa = _as2d(X)
        out = np.full(Xa.shape[0], -1, dtype=int)
        self._predict_rec(Xa, np.arange(Xa.shape[0]), 0, out)
        return out

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
        lab = self.predict(Xa)
        return -_inertia(Xa, _sample_weight(sample_weight, Xa.shape[0]), self.cluster_centers_, lab)
