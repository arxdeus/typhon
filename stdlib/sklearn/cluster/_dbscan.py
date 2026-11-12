# Ported from scikit-learn 1.9.1 (BSD-3-Clause)

# DBSCAN with exact brute-force radius neighborhoods (neighbors listed in
# index order) and the _dbscan_inner depth-first cluster expansion.

import numpy as np
from ._kmeans import _as2d

def _pair_dist(X: np.ndarray[float, 2], i: int, j: int, metric: str, p: float) -> float:
    d = X.shape[1]
    r = 0.0
    if metric == "euclidean" or metric == "l2" or (metric == "minkowski" and p == 2.0):
        for k in range(d):
            a = X[i, k] - X[j, k]
            r += a * a
        return np.sqrt(r)
    elif metric == "manhattan" or metric == "cityblock" or metric == "l1" or (metric == "minkowski" and p == 1.0):
        for k in range(d):
            r += abs(X[i, k] - X[j, k])
        return r
    elif metric == "chebyshev" or (metric == "minkowski" and p == np.inf):
        for k in range(d):
            a = abs(X[i, k] - X[j, k])
            if a > r:
                r = a
        return r
    elif metric == "minkowski":
        for k in range(d):
            r += abs(X[i, k] - X[j, k]) ** p
        return r ** (1.0 / p)
    elif metric == "precomputed":
        return X[i, j]
    else:
        raise ValueError(f"DBSCAN: unsupported metric {metric}")

class DBSCAN:
    eps: float
    min_samples: int
    metric: str
    p: float
    labels_: np.ndarray[int, 1]
    core_sample_indices_: np.ndarray[int, 1]
    components_: np.ndarray[float, 2]
    n_features_in_: int

    def __init__(self, eps: float = 0.5, *, min_samples: int = 5, metric: str = "euclidean",
                 metric_params = None, algorithm: str = "auto", leaf_size: int = 30,
                 p: Optional[float] = None, n_jobs = None):
        if eps <= 0.0:
            raise ValueError("eps must be > 0")
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.p = 2.0 if p is None else p
        self.labels_ = np.empty(0, dtype=int)
        self.core_sample_indices_ = np.empty(0, dtype=int)
        self.components_ = np.empty((0, 0))
        self.n_features_in_ = 0

    def fit(self, X, y = None, sample_weight = None):
        Xa = _as2d(X)
        n = Xa.shape[0]
        sw = np.ones(n)
        weighted = False
        if sample_weight is not None:
            sw = np.asarray(sample_weight, dtype=float)
            weighted = True
        neigh = List[List[int]](n)
        for i in range(n):
            neigh.append(List[int]())
        for i in range(n):
            for j in range(n):
                if _pair_dist(Xa, i, j, self.metric, self.p) <= self.eps:
                    neigh[i].append(j)
        core = [False] * n
        for i in range(n):
            if weighted:
                s = 0.0
                for j in neigh[i]:
                    s += sw[j]
                core[i] = s >= float(self.min_samples)
            else:
                core[i] = len(neigh[i]) >= self.min_samples
        labels = np.full(n, -1, dtype=int)
        label_num = 0
        stack = List[int]()
        for i0 in range(n):
            if labels[i0] != -1 or not core[i0]:
                continue
            i = i0
            while True:
                if labels[i] == -1:
                    labels[i] = label_num
                    if core[i]:
                        for v in neigh[i]:
                            if labels[v] == -1:
                                stack.append(v)
                if len(stack) == 0:
                    break
                i = stack.pop()
            label_num += 1
        nc = 0
        for i in range(n):
            if core[i]:
                nc += 1
        csi = np.empty(nc, dtype=int)
        t = 0
        for i in range(n):
            if core[i]:
                csi[t] = i
                t += 1
        self.core_sample_indices_ = csi
        self.labels_ = labels
        if self.metric == "precomputed":
            self.components_ = np.empty((0, Xa.shape[1]))
        else:
            self.components_ = np.ascontiguousarray(Xa[csi]) if nc > 0 else np.empty((0, Xa.shape[1]))
        self.n_features_in_ = Xa.shape[1]
        return self

    def fit_predict(self, X, y = None, sample_weight = None):
        return self.fit(X, sample_weight=sample_weight).labels_

def dbscan(X, eps: float = 0.5, *, min_samples: int = 5, metric: str = "euclidean",
           p: Optional[float] = None, sample_weight = None):
    est = DBSCAN(eps, min_samples=min_samples, metric=metric, p=p).fit(X, sample_weight=sample_weight)
    return est.core_sample_indices_, est.labels_
