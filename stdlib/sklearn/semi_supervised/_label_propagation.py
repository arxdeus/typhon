# Ported from scikit-learn 1.9.1 (BSD-3-Clause)

# LabelPropagation / LabelSpreading with kernel 'rbf' or 'knn' (dense
# graphs; scipy.sparse.csgraph.laplacian(normed=True) dense path). Integer
# labels, -1 marks unlabeled samples.

import numpy as np
from .._pairwise_k import kernel_matrix

class _BaseLabelPropagation:
    kernel: str
    gamma: float
    n_neighbors: int
    alpha: float
    max_iter: int
    tol: float
    X_: np.ndarray[float, 2]
    classes_: np.ndarray[int, 1]
    label_distributions_: np.ndarray[float, 2]
    transduction_: np.ndarray[int, 1]
    n_iter_: int
    _spreading: bool

    def _init_lp(self, kernel: str, gamma: float, n_neighbors: int, alpha: float, max_iter: int,
                 tol: float, spreading: bool):
        if kernel != "rbf" and kernel != "knn":
            raise ValueError(f"kernel must be 'rbf' or 'knn', got {kernel!r}")
        self.kernel = kernel
        self.gamma = gamma
        self.n_neighbors = n_neighbors
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self._spreading = spreading
        self.X_ = np.empty((0, 0))
        self.classes_ = np.empty(0, dtype=int)
        self.label_distributions_ = np.empty((0, 0))
        self.transduction_ = np.empty(0, dtype=int)
        self.n_iter_ = 0

    def _affinity(self) -> np.ndarray[float, 2]:
        if self.kernel == "rbf":
            return kernel_matrix(self.X_, self.X_, "rbf", self.gamma, 3.0, 1.0, True)
        from ..neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=self.n_neighbors).fit(self.X_)
        return nn.kneighbors_graph(self.X_, self.n_neighbors, mode="connectivity")

    def _build_graph(self) -> np.ndarray[float, 2]:
        A = self._affinity()
        n = A.shape[0]
        if not self._spreading:
            norm = A.sum(axis=1)
            A /= norm[:, None]
            return A
        # -laplacian(A, normed=True) with a zero diagonal (dense path)
        m = A.copy()
        for i in range(n):
            m[i, i] = 0.0
        d = m.sum(axis=0)
        for i in range(n):
            d[i] = 1.0 if d[i] == 0 else np.sqrt(d[i])
        m /= d
        m /= d[:, None]
        m *= -1
        m = -m
        for i in range(n):
            m[i, i] = 0.0
        return m

    def fit(self, X, y):
        Xa = np.ascontiguousarray(np.asarray(X, dtype=float))
        ya = np.asarray(y, dtype=int)
        self.X_ = Xa
        G = self._build_graph()
        u = np.unique(ya)
        cl = List[int]()
        for c in u:
            if c != -1:
                cl.append(int(c))
        self.classes_ = np.array(cl, dtype=int)
        n = ya.shape[0]
        k = len(cl)
        LD = np.zeros((n, k))
        for i in range(n):
            for c in range(k):
                if ya[i] == cl[c]:
                    LD[i, c] = 1.0
        ys = LD.copy()
        if not self._spreading:
            for i in range(n):
                if ya[i] == -1:
                    for c in range(k):
                        ys[i, c] = 0.0
        else:
            ys *= 1 - self.alpha
        prev = np.zeros((n, k))
        it = 0
        done = False
        while it < self.max_iter:
            if float(np.abs(LD - prev).sum()) < self.tol:
                done = True
                break
            prev = LD
            LD = G @ LD
            if not self._spreading:
                nrm = np.sum(LD, axis=1)
                for i in range(n):
                    if nrm[i] == 0:
                        nrm[i] = 1.0
                LD = LD / nrm[:, None]
                for i in range(n):
                    if ya[i] != -1:
                        for c in range(k):
                            LD[i, c] = ys[i, c]
            else:
                LD = self.alpha * LD + ys
            it += 1
        if done:
            self.n_iter_ = it
        else:
            self.n_iter_ = self.max_iter
        nrm = np.sum(LD, axis=1)
        for i in range(n):
            if nrm[i] == 0:
                nrm[i] = 1.0
        LD = LD / nrm[:, None]
        self.label_distributions_ = LD
        idx = np.argmax(LD, axis=1)
        tr = np.empty(n, dtype=int)
        for i in range(n):
            tr[i] = self.classes_[idx[i]]
        self.transduction_ = tr
        return self

    def predict_proba(self, X):
        Xa = np.ascontiguousarray(np.asarray(X, dtype=float))
        P = np.empty((0, 0))
        if self.kernel == "rbf":
            W = kernel_matrix(self.X_, Xa, "rbf", self.gamma, 3.0, 1.0, False)
            P = np.ascontiguousarray(W.T) @ self.label_distributions_
        else:
            from ..neighbors import NearestNeighbors
            nn = NearestNeighbors(n_neighbors=self.n_neighbors).fit(self.X_)
            ind = nn.kneighbors(Xa, return_distance=False)
            k = self.classes_.shape[0]
            P = np.zeros((Xa.shape[0], k))
            for i in range(Xa.shape[0]):
                for t in range(ind.shape[1]):
                    for c in range(k):
                        P[i, c] += self.label_distributions_[ind[i, t], c]
        nr = np.sum(P, axis=1)
        return P / nr[:, None]

    def predict(self, X):
        P = self.predict_proba(X)
        idx = np.argmax(P, axis=1)
        out = np.empty(P.shape[0], dtype=int)
        for i in range(P.shape[0]):
            out[i] = self.classes_[idx[i]]
        return out

    def score(self, X, y, sample_weight = None) -> float:
        from ..base import _accuracy
        return _accuracy(y, self.predict(X))

class LabelPropagation(_BaseLabelPropagation):
    def __init__(self, kernel: str = "rbf", *, gamma: float = 20.0, n_neighbors: int = 7,
                 max_iter: int = 1000, tol: float = 1e-3, n_jobs = None):
        self._init_lp(kernel, gamma, n_neighbors, 0.0, max_iter, tol, False)

class LabelSpreading(_BaseLabelPropagation):
    def __init__(self, kernel: str = "rbf", *, gamma: float = 20.0, n_neighbors: int = 7,
                 alpha: float = 0.2, max_iter: int = 30, tol: float = 1e-3, n_jobs = None):
        self._init_lp(kernel, gamma, n_neighbors, alpha, max_iter, tol, True)
