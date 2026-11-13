# Ported from scikit-learn 1.9.1 (BSD-3-Clause)

# PLSRegression / PLSCanonical / CCA (NIPALS power method or SVD) and PLSSVD.
# predict() returns 2-D for 2-D y and 1-D for 1-D y (static dispatch via D).

from ..base import estimator_repr as _estimator_repr
import numpy as np
import scipy.linalg as sl

_EPS = 2.220446049250313e-16

def _pinv2_old(a):
    u, s, vh = sl.svd(a, full_matrices=False, check_finite=False)
    cond = float(np.max(s)) * 1000000.0 * _EPS
    rank = 0
    for v in s:
        if v > cond:
            rank += 1
    uu = np.ascontiguousarray(u[:, :rank]) / s[:rank]
    return np.ascontiguousarray(np.dot(uu, vh[:rank]).T)

def _first_sv_power(X, y, mode: str, max_iter: int, tol: float, norm_y_weights: bool):
    eps = _EPS
    q = y.shape[1]
    ys = np.empty(0)
    found = False
    for j in range(q):
        col = np.ascontiguousarray(y[:, j])
        if np.any(np.abs(col) > eps):
            ys = col
            found = True
            break
    if not found:
        raise StopIteration("y residual is constant")
    xw_old = np.full(X.shape[1], 100.0)
    Xp = np.empty((0, 0)); yp = np.empty((0, 0))
    if mode == "B":
        Xp = _pinv2_old(X)
        yp = _pinv2_old(y)
    xw = np.empty(0); yw = np.empty(0)
    it = 0
    for i in range(max_iter):
        it = i
        if mode == "B":
            xw = np.dot(Xp, ys)
        else:
            xw = np.dot(X.T, ys) / np.dot(ys, ys)
        xw /= np.sqrt(np.dot(xw, xw)) + eps
        xs = np.dot(X, xw)
        if mode == "B":
            yw = np.dot(yp, xs)
        else:
            yw = np.dot(y.T, xs) / np.dot(xs, xs)
        if norm_y_weights:
            yw /= np.sqrt(np.dot(yw, yw)) + eps
        ys = np.dot(y, yw) / (np.dot(yw, yw) + eps)
        d = xw - xw_old
        if np.dot(d, d) < tol or q == 1:
            break
        xw_old = xw
    return xw, yw, it + 1

def _center_scale_xy(X, y, scale: bool):
    xm = X.mean(axis=0)
    X -= xm
    ym = y.mean(axis=0)
    y -= ym
    xs = np.ones(X.shape[1])
    ysd = np.ones(y.shape[1])
    if scale:
        xs = X.std(axis=0, ddof=1)
        for i in range(xs.shape[0]):
            if xs[i] == 0.0:
                xs[i] = 1.0
        X /= xs
        ysd = y.std(axis=0, ddof=1)
        for i in range(ysd.shape[0]):
            if ysd[i] == 0.0:
                ysd[i] = 1.0
        y /= ysd
    return X, y, xm, ym, xs, ysd

def _svd_flip_1d(u, v):
    b = int(np.argmax(np.abs(u)))
    s = np.sign(u[b])
    u *= s
    v *= s

class _PLS[D: Literal[int] = 1]:
    n_components: int
    scale: bool
    deflation_mode: str
    mode: str
    algorithm: str
    max_iter: int
    tol: float
    x_weights_: np.ndarray[float, 2]
    y_weights_: np.ndarray[float, 2]
    _x_scores: np.ndarray[float, 2]
    _y_scores: np.ndarray[float, 2]
    x_loadings_: np.ndarray[float, 2]
    y_loadings_: np.ndarray[float, 2]
    x_rotations_: np.ndarray[float, 2]
    y_rotations_: np.ndarray[float, 2]
    coef_: np.ndarray[float, 2]
    intercept_: np.ndarray[float, 1]
    n_iter_: List[int]
    _x_mean: np.ndarray[float, 1]
    _y_mean: np.ndarray[float, 1]
    _x_std: np.ndarray[float, 1]
    _y_std: np.ndarray[float, 1]

    def _init_pls(self, n_components: int, scale: bool, deflation_mode: str, mode: str,
                  algorithm: str, max_iter: int, tol: float):
        self.n_components = n_components
        self.scale = scale
        self.deflation_mode = deflation_mode
        self.mode = mode
        self.algorithm = algorithm
        self.max_iter = max_iter
        self.tol = tol
        e2 = np.empty((0, 0)); e1 = np.empty(0)
        self.x_weights_ = e2; self.y_weights_ = e2; self._x_scores = e2; self._y_scores = e2
        self.x_loadings_ = e2; self.y_loadings_ = e2; self.x_rotations_ = e2; self.y_rotations_ = e2
        self.coef_ = e2; self.intercept_ = e1; self.n_iter_ = []
        self._x_mean = e1; self._y_mean = e1; self._x_std = e1; self._y_std = e1

    def _fit(self, X, y):
        Xa = np.array(np.asarray(X, dtype=float))
        ya = np.asarray(y, dtype=float)
        Y = np.array(ya.reshape(ya.shape[0], -1))
        n, p = Xa.shape
        q = Y.shape[1]
        nc = self.n_components
        rub = min(n, p) if self.deflation_mode == "regression" else min(n, p, q)
        if nc > rub:
            raise ValueError(f"`n_components` upper bound is {rub}. Got {nc} instead. Reduce `n_components`.")
        norm_y = self.deflation_mode == "canonical"
        Xk, yk, self._x_mean, self._y_mean, self._x_std, self._y_std = _center_scale_xy(Xa, Y, self.scale)
        self.x_weights_ = np.zeros((p, nc)); self.y_weights_ = np.zeros((q, nc))
        self._x_scores = np.zeros((n, nc)); self._y_scores = np.zeros((n, nc))
        self.x_loadings_ = np.zeros((p, nc)); self.y_loadings_ = np.zeros((q, nc))
        self.n_iter_ = []
        y_eps = _EPS
        for k in range(nc):
            xw = np.empty(0); yw = np.empty(0)
            if self.algorithm == "nipals":
                for j in range(q):
                    allsmall = True
                    for i in range(n):
                        if not abs(yk[i, j]) < 10 * y_eps:
                            allsmall = False
                    if allsmall:
                        for i in range(n):
                            yk[i, j] = 0.0
                try:
                    xw, yw, it = _first_sv_power(Xk, yk, self.mode, self.max_iter, self.tol, norm_y)
                except StopIteration:
                    break
                self.n_iter_.append(it)
            else:
                U, _, Vt = sl.svd(np.dot(Xk.T, yk), full_matrices=False)
                xw = np.ascontiguousarray(U[:, 0])
                yw = np.ascontiguousarray(Vt[0, :])
            _svd_flip_1d(xw, yw)
            xs = np.dot(Xk, xw)
            y_ss = 1.0 if norm_y else float(np.dot(yw, yw))
            ys = np.dot(yk, yw) / y_ss
            xl = np.dot(xs, Xk) / np.dot(xs, xs)
            Xk -= np.outer(xs, xl)
            yl = np.empty(0)
            if self.deflation_mode == "canonical":
                yl = np.dot(ys, yk) / np.dot(ys, ys)
                yk -= np.outer(ys, yl)
            else:
                yl = np.dot(xs, yk) / np.dot(xs, xs)
                yk -= np.outer(xs, yl)
            self.x_weights_[:, k] = xw
            self.y_weights_[:, k] = yw
            self._x_scores[:, k] = xs
            self._y_scores[:, k] = ys
            self.x_loadings_[:, k] = xl
            self.y_loadings_[:, k] = yl
        self.x_rotations_ = np.dot(self.x_weights_, sl.pinv(np.dot(self.x_loadings_.T, self.x_weights_), check_finite=False))
        self.y_rotations_ = np.dot(self.y_weights_, sl.pinv(np.dot(self.y_loadings_.T, self.y_weights_), check_finite=False))
        c = np.dot(self.x_rotations_, self.y_loadings_.T)
        self.coef_ = np.ascontiguousarray((c * self._y_std).T / self._x_std)
        self.intercept_ = self._y_mean
        return self

    def transform(self, X):
        Xa = (np.asarray(X, dtype=float) - self._x_mean) / self._x_std
        return np.dot(Xa, self.x_rotations_)

    def transform_xy(self, X, y):
        ya = np.asarray(y, dtype=float)
        Y = (ya.reshape(ya.shape[0], -1) - self._y_mean) / self._y_std
        return self.transform(X), np.dot(Y, self.y_rotations_)

    def inverse_transform(self, X):
        R = np.matmul(np.asarray(X, dtype=float), self.x_loadings_.T)
        R *= self._x_std
        R += self._x_mean
        return R

    def predict(self, X):
        Xa = np.asarray(X, dtype=float) - self._x_mean
        yp = Xa @ self.coef_.T + self.intercept_
        if D == 1:
            return np.ascontiguousarray(yp.ravel())
        else:
            return yp

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)

    def score(self, X, y, sample_weight = None) -> float:
        from ..metrics import r2_score
        return float(r2_score(y, self.predict(X)))

def _retype(est, T: type):
    n = T.__new__()
    for k, v in static.vars(est):
        setattr(n, k, v)
    return n

class PLSRegression[D: Literal[int] = 1](_PLS[D]):
    def __repr__(self) -> str:
        return _estimator_repr(self)

    x_scores_: np.ndarray[float, 2]
    y_scores_: np.ndarray[float, 2]

    def __init__(self, n_components: int = 2, *, scale: bool = True, max_iter: int = 500,
                 tol: float = 1e-6, copy: bool = True):
        self._init_pls(n_components, scale, "regression", "A", "nipals", max_iter, tol)
        self.x_scores_ = np.empty((0, 0))
        self.y_scores_ = np.empty((0, 0))

    def fit(self, X, y):
        ya = np.asarray(y, dtype=float)
        if static.len(ya.shape) != D:
            return _retype(self, PLSRegression[static.len(ya.shape)]).fit(X, ya)
        self._fit(X, ya)
        self.x_scores_ = self._x_scores
        self.y_scores_ = self._y_scores
        return self

class PLSCanonical[D: Literal[int] = 1](_PLS[D]):
    def __repr__(self) -> str:
        return _estimator_repr(self)

    def __init__(self, n_components: int = 2, *, scale: bool = True, algorithm: str = "nipals",
                 max_iter: int = 500, tol: float = 1e-6, copy: bool = True):
        self._init_pls(n_components, scale, "canonical", "A", algorithm, max_iter, tol)

    def fit(self, X, y):
        ya = np.asarray(y, dtype=float)
        if static.len(ya.shape) != D:
            return _retype(self, PLSCanonical[static.len(ya.shape)]).fit(X, ya)
        return self._fit(X, ya)

class CCA[D: Literal[int] = 1](_PLS[D]):
    def __repr__(self) -> str:
        return _estimator_repr(self)

    def __init__(self, n_components: int = 2, *, scale: bool = True, max_iter: int = 500,
                 tol: float = 1e-6, copy: bool = True):
        self._init_pls(n_components, scale, "canonical", "B", "nipals", max_iter, tol)

    def fit(self, X, y):
        ya = np.asarray(y, dtype=float)
        if static.len(ya.shape) != D:
            return _retype(self, CCA[static.len(ya.shape)]).fit(X, ya)
        return self._fit(X, ya)

class PLSSVD:
    def __repr__(self) -> str:
        return _estimator_repr(self)

    n_components: int
    scale: bool
    x_weights_: np.ndarray[float, 2]
    y_weights_: np.ndarray[float, 2]
    _x_mean: np.ndarray[float, 1]
    _y_mean: np.ndarray[float, 1]
    _x_std: np.ndarray[float, 1]
    _y_std: np.ndarray[float, 1]

    def __init__(self, n_components: int = 2, *, scale: bool = True, copy: bool = True):
        self.n_components = n_components
        self.scale = scale
        e2 = np.empty((0, 0)); e1 = np.empty(0)
        self.x_weights_ = e2; self.y_weights_ = e2
        self._x_mean = e1; self._y_mean = e1; self._x_std = e1; self._y_std = e1

    def fit(self, X, y):
        from ..utils.extmath import svd_flip
        Xa = np.array(np.asarray(X, dtype=float))
        ya = np.asarray(y, dtype=float)
        Y = np.array(ya.reshape(ya.shape[0], -1))
        nc = self.n_components
        rub = min(Xa.shape[0], Xa.shape[1], Y.shape[1])
        if nc > rub:
            raise ValueError(f"`n_components` upper bound is {rub}. Got {nc} instead. Reduce `n_components`.")
        Xc, Yc, self._x_mean, self._y_mean, self._x_std, self._y_std = _center_scale_xy(Xa, Y, self.scale)
        U, s, Vt = sl.svd(np.dot(Xc.T, Yc), full_matrices=False)
        U = np.ascontiguousarray(U[:, :nc])
        Vt = np.ascontiguousarray(Vt[:nc])
        svd_flip(U, Vt)
        self.x_weights_ = U
        self.y_weights_ = np.ascontiguousarray(Vt.T)
        return self

    def transform(self, X):
        return np.dot((np.asarray(X, dtype=float) - self._x_mean) / self._x_std, self.x_weights_)

    def transform_xy(self, X, y):
        ya = np.asarray(y, dtype=float)
        Y = (ya.reshape(ya.shape[0], -1) - self._y_mean) / self._y_std
        return self.transform(X), np.dot(Y, self.y_weights_)

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)
