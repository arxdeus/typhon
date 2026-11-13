# Ported from scikit-learn 1.9.1 (BSD-3-Clause)

# sklearn.neighbors.NeighborhoodComponentsAnalysis (L-BFGS-B from scipy.optimize)

from ..base import estimator_repr as _estimator_repr
import numpy as np
import math
from ._dist import _as2d

class _NCAObjective:
    X: np.ndarray[float, 2]
    same: np.ndarray[float, 2]
    nf: int
    n_iter: int
    last_x: np.ndarray[float, 1]
    last_g: np.ndarray[float, 1]
    last_f: float
    have: bool
    callback_n: int

    def __init__(self, X: np.ndarray[float, 2], same: np.ndarray[float, 2]):
        self.X = X
        self.same = same
        self.nf = X.shape[1]
        self.n_iter = 0
        self.last_x = np.empty(0)
        self.last_g = np.empty(0)
        self.last_f = 0.0
        self.have = False
        self.callback_n = 0

    def _eval(self, t: np.ndarray[float, 1]):
        if self.have and self.last_x.shape[0] == t.shape[0]:
            same_x = True
            for i in range(t.shape[0]):
                if self.last_x[i] != t[i]:
                    same_x = False
                    break
            if same_x:
                return
        if self.n_iter == 0:
            self.n_iter += 1
        n = self.X.shape[0]
        T = np.ascontiguousarray(t.reshape((-1, self.nf)))
        E = self.X @ T.T
        sq = np.sum(E * E, axis=1)
        P = -2.0 * (E @ E.T)
        P += sq.reshape((-1, 1))
        P += sq.reshape((1, -1))
        for i in range(n):
            for j in range(n):
                if P[i, j] < 0.0:
                    P[i, j] = 0.0
            P[i, i] = 0.0
        for i in range(n):
            P[i, i] = float("inf")
        # softmax(-P) row-wise
        W = np.empty((n, n))
        for i in range(n):
            mx = -float("inf")
            for j in range(n):
                v = -P[i, j]
                if v > mx:
                    mx = v
            s = 0.0
            for j in range(n):
                W[i, j] = math.exp(-P[i, j] - mx)
                s += W[i, j]
            for j in range(n):
                W[i, j] /= s
        M = W * self.same
        p = np.sum(M, axis=1)
        loss = float(np.sum(p))
        Wd = M - W * p.reshape((-1, 1))
        Ws = Wd + Wd.T
        col = np.sum(Wd, axis=0)
        for i in range(n):
            Ws[i, i] = -col[i]
        G = 2.0 * ((E.T @ Ws) @ self.X)
        self.last_x = t.copy()
        self.last_g = np.ascontiguousarray(G.reshape((-1,)))
        self.last_f = loss
        self.have = True

    def fun(self, t):
        self._eval(t)
        return -self.last_f

    def grad(self, t):
        self._eval(t)
        return -self.last_g

    def cb(self, t):
        self.n_iter += 1

class NeighborhoodComponentsAnalysis:
    def __repr__(self) -> str:
        return _estimator_repr(self)

    n_components: Optional[int]
    init: str
    init_matrix: np.ndarray[float, 2]
    has_init_matrix: bool
    warm_start: bool
    max_iter: int
    tol: float
    random_state: Optional[int]
    components_: np.ndarray[float, 2]
    n_iter_: int
    n_features_in_: int
    classes_: np.ndarray[int, 1]
    _fitted: bool

    def __init__(self, n_components: Optional[int] = None, *, init = "auto", warm_start: bool = False,
                 max_iter: int = 50, tol: float = 1e-5, callback = None, verbose: int = 0,
                 random_state: Optional[int] = None):
        self.n_components = n_components
        self.has_init_matrix = False
        self.init_matrix = np.empty((0, 0))
        if isinstance(init, str):
            if init != "auto" and init != "pca" and init != "lda" and init != "identity" and init != "random":
                raise ValueError(f"init must be one of auto, pca, lda, identity, random, got {init}")
            self.init = init
        else:
            self.init = ""
            self.has_init_matrix = True
            self.init_matrix = np.ascontiguousarray(np.asarray(init, dtype=float))
        self.warm_start = warm_start
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.components_ = np.empty((0, 0))
        self.n_iter_ = 0
        self.n_features_in_ = 0
        self.classes_ = np.empty(0, dtype=int)
        self._fitted = False

    def _initialize(self, X: np.ndarray[float, 2], yi: np.ndarray[int, 1], ncls: int) -> np.ndarray[float, 2]:
        if self.warm_start and self._fitted:
            return self.components_.copy()
        if self.has_init_matrix:
            return self.init_matrix.copy()
        n, nf = X.shape
        nc = nf if self.n_components is None else self.n_components.__val__()
        init = self.init
        if init == "auto":
            if nc <= min(nf, ncls - 1):
                init = "lda"
            elif nc < min(nf, n):
                init = "pca"
            else:
                init = "identity"
        if init == "identity":
            return np.eye(nc, nf)
        if init == "random":
            if self.random_state is None:
                raise ValueError("NeighborhoodComponentsAnalysis(init='random') needs an integer random_state in Typhon")
            from ..utils import _LegacyGauss
            return _LegacyGauss(self.random_state.__val__()).normal(nc, nf)
        if init == "pca":
            from ..decomposition import PCA
            pca = PCA(n_components=nc, random_state=self.random_state)
            pca.fit(X)
            return pca.components_.copy()
        from ..discriminant_analysis import LinearDiscriminantAnalysis
        lda = LinearDiscriminantAnalysis(n_components=nc)
        lda.fit(X, yi)
        return np.ascontiguousarray(lda.scalings_.T[:nc])

    def fit(self, X, y):
        Xa = _as2d(X)
        n, nf = Xa.shape
        ya = np.asarray(y, dtype=int).ravel()
        if ya.shape[0] != n:
            raise ValueError("Found input variables with inconsistent numbers of samples")
        classes, yi = np.unique(ya, return_inverse=True)
        yi = np.asarray(yi, dtype=int).ravel()
        self.classes_ = classes
        if self.n_components is not None and self.n_components.__val__() > nf:
            raise ValueError(f"The preferred dimensionality of the projected space `n_components` ({self.n_components.__val__()}) cannot be greater than the given data dimensionality ({nf})!")
        if self.warm_start and self._fitted and self.components_.shape[1] != nf:
            raise ValueError(f"The new inputs dimensionality ({nf}) does not match the input dimensionality of the previously learned transformation ({self.components_.shape[1]}).")
        if self.has_init_matrix:
            im = self.init_matrix
            if im.shape[1] != nf:
                raise ValueError(f"The input dimensionality ({im.shape[1]}) of the given linear transformation `init` must match the dimensionality of the given inputs `X` ({nf}).")
            if im.shape[0] > im.shape[1]:
                raise ValueError(f"The output dimensionality ({im.shape[0]}) of the given linear transformation `init` cannot be greater than its input dimensionality ({im.shape[1]}).")
            if self.n_components is not None and self.n_components.__val__() != im.shape[0]:
                raise ValueError(f"The preferred dimensionality of the projected space `n_components` ({self.n_components.__val__()}) does not match the output dimensionality of the given linear transformation `init` ({im.shape[0]})!")
        same = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if yi[i] == yi[j]:
                    same[i, j] = 1.0
        T0 = self._initialize(Xa, yi, classes.shape[0])
        x0 = np.ascontiguousarray(T0.reshape((-1,)))
        obj = _NCAObjective(Xa, same)
        from scipy.optimize import minimize
        opts = {"maxiter": float(self.max_iter)}
        res = minimize(obj.fun, x0, method="L-BFGS-B", jac=obj.grad, tol=self.tol,
                       options=opts, callback=obj.cb)
        self.n_iter_ = obj.n_iter
        self.components_ = np.ascontiguousarray(res.x.reshape((-1, nf)))
        self.n_features_in_ = nf
        self._fitted = True
        return self

    def transform(self, X):
        return _as2d(X) @ self.components_.T

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)
