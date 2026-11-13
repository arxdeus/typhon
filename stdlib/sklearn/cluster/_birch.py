# Ported from scikit-learn 1.9.1 (BSD-3-Clause)

# Birch: CF-tree construction (insert / merge / split with farthest-pair
# redistribution, leaf linked list), subcluster centers, global clustering
# with AgglomerativeClustering(n_clusters) (or none), predict / transform /
# partial_fit.

from ..base import estimator_repr as _estimator_repr
import numpy as np
from ._kmeans import _as2d
from ._agglomerative import AgglomerativeClustering

class _CFSub:
    n_samples_: int
    squared_sum_: float
    linear_sum_: np.ndarray[float, 1]
    centroid_: np.ndarray[float, 1]
    sq_norm_: float
    child_: int

    def __init__(self, nf: int, ls: Optional[np.ndarray[float, 1]] = None):
        if ls is None:
            self.n_samples_ = 0
            self.squared_sum_ = 0.0
            self.linear_sum_ = np.zeros(nf)
            self.centroid_ = np.zeros(nf)
            self.sq_norm_ = 0.0
        else:
            v = ls.__val__()
            self.n_samples_ = 1
            self.linear_sum_ = v.copy()
            self.centroid_ = v.copy()
            self.squared_sum_ = float(np.dot(v, v))
            self.sq_norm_ = self.squared_sum_
        self.child_ = -1

    def update(self, o: _CFSub):
        self.n_samples_ += o.n_samples_
        self.linear_sum_ = self.linear_sum_ + o.linear_sum_
        self.squared_sum_ += o.squared_sum_
        self.centroid_ = self.linear_sum_ / self.n_samples_
        self.sq_norm_ = float(np.dot(self.centroid_, self.centroid_))

    def merge_subcluster(self, o: _CFSub, threshold: float) -> bool:
        new_ss = self.squared_sum_ + o.squared_sum_
        new_ls = self.linear_sum_ + o.linear_sum_
        new_n = self.n_samples_ + o.n_samples_
        new_c = (1 / new_n) * new_ls
        new_sq = float(np.dot(new_c, new_c))
        sq_radius = new_ss / new_n - new_sq
        if sq_radius <= threshold ** 2:
            self.n_samples_ = new_n
            self.linear_sum_ = new_ls
            self.squared_sum_ = new_ss
            self.centroid_ = new_c
            self.sq_norm_ = new_sq
            return True
        return False

    @property
    def radius(self) -> float:
        return np.sqrt(max(0.0, self.squared_sum_ / self.n_samples_ - self.sq_norm_))

class _CFNode:
    threshold: float
    branching_factor: int
    is_leaf: bool
    n_features: int
    subclusters_: List[_CFSub]
    init_centroids_: np.ndarray[float, 2]
    init_sq_norm_: np.ndarray[float, 1]
    prev_leaf_: int
    next_leaf_: int
    pool: List[_CFNode]
    idx: int

    def __init__(self, threshold: float, branching_factor: int, is_leaf: bool, n_features: int,
                 pool: List[_CFNode]):
        self.pool = pool
        self.idx = len(pool)
        self.threshold = threshold
        self.branching_factor = branching_factor
        self.is_leaf = is_leaf
        self.n_features = n_features
        self.subclusters_ = List[_CFSub]()
        self.init_centroids_ = np.zeros((branching_factor + 1, n_features))
        self.init_sq_norm_ = np.zeros(branching_factor + 1)
        self.prev_leaf_ = -1
        self.next_leaf_ = -1
        pool.append(self)

    def centroids(self):
        return self.init_centroids_[:len(self.subclusters_)]

    def sq_norms(self):
        return self.init_sq_norm_[:len(self.subclusters_)]

    def append_subcluster(self, s: _CFSub):
        k = len(self.subclusters_)
        self.subclusters_.append(s)
        self.init_centroids_[k] = s.centroid_
        self.init_sq_norm_[k] = s.sq_norm_

    def update_split_subclusters(self, s: _CFSub, n1: _CFSub, n2: _CFSub):
        ind = 0
        for q in range(len(self.subclusters_)):
            if self.subclusters_[q] is s:
                ind = q
                break
        self.subclusters_[ind] = n1
        self.init_centroids_[ind] = n1.centroid_
        self.init_sq_norm_[ind] = n1.sq_norm_
        self.append_subcluster(n2)

    def insert_cf_subcluster(self, s: _CFSub) -> bool:
        if len(self.subclusters_) == 0:
            self.append_subcluster(s)
            return False
        C = self.centroids()
        dm = (C @ s.centroid_) * -2.0
        dm += self.sq_norms()
        ci = int(np.argmin(dm))
        cs = self.subclusters_[ci]
        if cs.child_ >= 0:
            child = self.pool[cs.child_]
            split_child = child.insert_cf_subcluster(s)
            if not split_child:
                cs.update(s)
                self.init_centroids_[ci] = self.subclusters_[ci].centroid_
                self.init_sq_norm_[ci] = self.subclusters_[ci].sq_norm_
                return False
            else:
                n1, n2 = _split_node(child, self.threshold, self.branching_factor)
                self.update_split_subclusters(cs, n1, n2)
                return len(self.subclusters_) > self.branching_factor
        else:
            if cs.merge_subcluster(s, self.threshold):
                self.init_centroids_[ci] = cs.centroid_
                self.init_sq_norm_[ci] = cs.sq_norm_
                return False
            elif len(self.subclusters_) < self.branching_factor:
                self.append_subcluster(s)
                return False
            else:
                self.append_subcluster(s)
                return True

def _split_node(node: _CFNode, threshold: float, bf: int):
    nf = node.n_features
    s1 = _CFSub(nf)
    s2 = _CFSub(nf)
    pool = node.pool
    n1 = _CFNode(threshold, bf, node.is_leaf, nf, pool)
    n2 = _CFNode(threshold, bf, node.is_leaf, nf, pool)
    s1.child_ = n1.idx
    s2.child_ = n2.idx
    if node.is_leaf:
        if node.prev_leaf_ >= 0:
            pool[node.prev_leaf_].next_leaf_ = n1.idx
        n1.prev_leaf_ = node.prev_leaf_
        n1.next_leaf_ = n2.idx
        n2.prev_leaf_ = n1.idx
        n2.next_leaf_ = node.next_leaf_
        if node.next_leaf_ >= 0:
            pool[node.next_leaf_].prev_leaf_ = n2.idx
    C = node.centroids()
    sq = node.sq_norms()
    k = C.shape[0]
    G = C @ C.T
    D = np.empty((k, k))
    for i in range(k):
        for j in range(k):
            v = -2.0 * G[i, j] + sq[i] + sq[j]
            D[i, j] = v if v > 0.0 else 0.0
        D[i, i] = 0.0
    fi = 0
    fj = 0
    best = -np.inf
    for i in range(k):
        for j in range(k):
            if D[i, j] > best:
                best = D[i, j]
                fi = i
                fj = j
    for idx in range(k):
        closer = D[fi, idx] < D[fj, idx] or idx == fi
        sub = node.subclusters_[idx]
        if closer:
            n1.append_subcluster(sub)
            s1.update(sub)
        else:
            n2.append_subcluster(sub)
            s2.update(sub)
    return s1, s2

class Birch:
    def __repr__(self) -> str:
        return _estimator_repr(self)

    threshold: float
    branching_factor: int
    n_clusters: Optional[int]
    compute_labels: bool
    root_: _CFNode
    dummy_leaf_: _CFNode
    has_root: bool
    pool: List[_CFNode]
    subcluster_centers_: np.ndarray[float, 2]
    subcluster_labels_: np.ndarray[int, 1]
    labels_: np.ndarray[int, 1]
    _subcluster_norms: np.ndarray[float, 1]
    n_features_in_: int

    def __init__(self, *, threshold: float = 0.5, branching_factor: int = 50,
                 n_clusters: Optional[int] = 3, compute_labels: bool = True):
        if threshold <= 0.0:
            raise ValueError("threshold must be > 0")
        if branching_factor <= 1:
            raise ValueError("branching_factor must be > 1")
        self.threshold = threshold
        self.branching_factor = branching_factor
        self.n_clusters = n_clusters
        self.compute_labels = compute_labels
        self.pool = List[_CFNode]()
        self.root_ = _CFNode(threshold, branching_factor, True, 0, self.pool)
        self.dummy_leaf_ = _CFNode(threshold, branching_factor, True, 0, self.pool)
        self.has_root = False
        self.subcluster_centers_ = np.empty((0, 0))
        self.subcluster_labels_ = np.empty(0, dtype=int)
        self.labels_ = np.empty(0, dtype=int)
        self._subcluster_norms = np.empty(0)
        self.n_features_in_ = 0

    def _fit(self, Xa: np.ndarray[float, 2], partial: bool):
        n, nf = Xa.shape
        if not (partial and self.has_root):
            self.pool = List[_CFNode]()
            self.root_ = _CFNode(self.threshold, self.branching_factor, True, nf, self.pool)
            self.dummy_leaf_ = _CFNode(self.threshold, self.branching_factor, True, nf, self.pool)
            self.dummy_leaf_.next_leaf_ = self.root_.idx
            self.root_.prev_leaf_ = self.dummy_leaf_.idx
            self.has_root = True
            self.n_features_in_ = nf
        for i in range(n):
            sub = _CFSub(nf, Xa[i].copy())
            if self.root_.insert_cf_subcluster(sub):
                s1, s2 = _split_node(self.root_, self.threshold, self.branching_factor)
                self.root_ = _CFNode(self.threshold, self.branching_factor, False, nf, self.pool)
                self.root_.append_subcluster(s1)
                self.root_.append_subcluster(s2)
        parts = List[np.ndarray[float, 2]]()
        lp = self.dummy_leaf_.next_leaf_
        while lp >= 0:
            leaf = self.pool[lp]
            parts.append(leaf.centroids().copy())
            lp = leaf.next_leaf_
        self.subcluster_centers_ = np.concatenate(parts, axis=0)
        self._global_clustering(Xa, True)
        return self

    def fit(self, X, y = None):
        return self._fit(_as2d(X), False)

    def partial_fit(self, X = None, y = None):
        if X is None:
            self._global_clustering(np.empty((0, 0)), False)
            return self
        return self._fit(_as2d(X), True)

    def _global_clustering(self, Xa: np.ndarray[float, 2], has_x: bool):
        C = self.subcluster_centers_
        k = C.shape[0]
        self._subcluster_norms = (C * C).sum(axis=1)
        not_enough = False
        if self.n_clusters is not None and k < self.n_clusters.__val__():
            not_enough = True
        if self.n_clusters is None or not_enough:
            self.subcluster_labels_ = np.arange(k)
        else:
            self.subcluster_labels_ = AgglomerativeClustering(n_clusters=self.n_clusters).fit(C).labels_
        if has_x and self.compute_labels:
            self.labels_ = self._predict(Xa)

    def _predict(self, Xa: np.ndarray[float, 2]):
        C = self.subcluster_centers_
        xn = (Xa * Xa).sum(axis=1)
        G = Xa @ C.T
        out = np.empty(Xa.shape[0], dtype=int)
        for i in range(Xa.shape[0]):
            best = 0
            bv = np.inf
            for j in range(C.shape[0]):
                v = -2.0 * G[i, j] + xn[i] + self._subcluster_norms[j]
                v = v if v > 0.0 else 0.0
                if v < bv:
                    bv = v
                    best = j
            out[i] = self.subcluster_labels_[best]
        return out

    def predict(self, X):
        return self._predict(_as2d(X))

    def fit_predict(self, X, y = None):
        return self.fit(X).labels_

    def transform(self, X):
        Xa = _as2d(X)
        C = self.subcluster_centers_
        xn = (Xa * Xa).sum(axis=1)
        G = Xa @ C.T
        D = np.empty((Xa.shape[0], C.shape[0]))
        for i in range(Xa.shape[0]):
            for j in range(C.shape[0]):
                v = -2.0 * G[i, j] + xn[i] + self._subcluster_norms[j]
                D[i, j] = np.sqrt(v) if v > 0.0 else 0.0
        return D

    def fit_transform(self, X, y = None):
        return self.fit(X).transform(X)
