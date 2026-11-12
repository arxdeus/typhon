# Ported from scikit-learn 1.9.1 (BSD-3-Clause)
#
# Orthogonal Matching Pursuit: orthogonal_mp, orthogonal_mp_gram,
# OrthogonalMatchingPursuit, OrthogonalMatchingPursuitCV (dense).
# Internally every target's solution path is returned as a dense
# (n_features, n_steps) matrix in original feature order.
# DEVIATION: orthogonal_mp(..., return_path=True) returns a 2-d
# (n_features, n_steps) path for one target / 3-d for several (sklearn squeezes).

import math
import numpy as np
from ._base import (_retype, _ndim_of, _as2d_y, _preprocess_data, _set_intercept,
                    _LinearRegressorMixin)
from ._least_angle import _solve_tri_lower, _potrs

_EPS = 2.220446049250313e-16

def _cholesky_omp(X0: np.ndarray[float, 2], y: np.ndarray[float, 1], n_nonzero_coefs: int,
                  tol: Optional[float]):
    """Returns (path (n_features, n_active), n_active)."""
    X = np.array(X0)
    n, p = X.shape
    alpha = X.T @ y
    residual = y.copy()
    indices = np.arange(p)
    max_features = p if tol is not None else n_nonzero_coefs
    L = np.zeros((max_features, max_features))
    path = np.zeros((p, max_features))
    n_active = 0
    while True:
        corr = np.abs(X.T @ residual)
        lam = int(np.argmax(corr))
        if lam < n_active or alpha[lam] ** 2 < _EPS:
            break
        if n_active > 0:
            row = np.zeros(n_active)
            for a in range(n_active):
                s = 0.0
                for i in range(n):
                    s += X[i, a] * X[i, lam]
                row[a] = s
            _solve_tri_lower(L, n_active, row)
            v = 0.0
            for a in range(n_active):
                L[n_active, a] = row[a]
            v = math.sqrt(float(np.dot(row, row))) ** 2
            nl = 0.0
            for i in range(n):
                nl += X[i, lam] * X[i, lam]
            Lkk = math.sqrt(nl) ** 2 - v
            if Lkk <= _EPS:
                break
            L[n_active, n_active] = math.sqrt(Lkk)
        else:
            nl = 0.0
            for i in range(n):
                nl += X[i, lam] * X[i, lam]
            L[0, 0] = math.sqrt(nl)
        for i in range(n):
            t = X[i, n_active]
            X[i, n_active] = X[i, lam]
            X[i, lam] = t
        ta = alpha[n_active]
        alpha[n_active] = alpha[lam]
        alpha[lam] = ta
        ti = indices[n_active]
        indices[n_active] = indices[lam]
        indices[lam] = ti
        n_active += 1
        rhs = np.zeros(n_active)
        for a in range(n_active):
            rhs[a] = alpha[a]
        gamma = _potrs(L, n_active, rhs)
        for a in range(n_active):
            path[indices[a], n_active - 1] = gamma[a]
        residual = y - X[:, :n_active] @ gamma
        if tol is not None:
            rn = math.sqrt(float(np.dot(residual, residual)))
            if rn ** 2 <= tol.__val__():
                break
        if n_active == max_features:
            break
    return np.ascontiguousarray(path[:, :n_active]), n_active

def _gram_omp(Gram0: np.ndarray[float, 2], Xy0: np.ndarray[float, 1], n_nonzero_coefs: int,
              tol_0: float, tol: Optional[float]):
    Gram = np.array(Gram0)
    Xy = Xy0.copy()
    p = Gram.shape[0]
    indices = np.arange(p)
    alpha = Xy.copy()
    tol_curr = tol_0
    delta = 0.0
    max_features = p if tol is not None else n_nonzero_coefs
    L = np.zeros((max_features, max_features))
    L[0, 0] = 1.0
    path = np.zeros((p, max_features))
    n_active = 0
    while True:
        lam = int(np.argmax(np.abs(alpha)))
        if lam < n_active or alpha[lam] ** 2 < _EPS:
            break
        if n_active > 0:
            row = np.zeros(n_active)
            for a in range(n_active):
                row[a] = Gram[lam, a]
            _solve_tri_lower(L, n_active, row)
            for a in range(n_active):
                L[n_active, a] = row[a]
            v = math.sqrt(float(np.dot(row, row))) ** 2
            Lkk = Gram[lam, lam] - v
            if Lkk <= _EPS:
                break
            L[n_active, n_active] = math.sqrt(Lkk)
        else:
            L[0, 0] = math.sqrt(Gram[lam, lam])
        for k in range(p):
            t = Gram[n_active, k]
            Gram[n_active, k] = Gram[lam, k]
            Gram[lam, k] = t
        for k in range(p):
            t = Gram[k, n_active]
            Gram[k, n_active] = Gram[k, lam]
            Gram[k, lam] = t
        ti = indices[n_active]
        indices[n_active] = indices[lam]
        indices[lam] = ti
        tx = Xy[n_active]
        Xy[n_active] = Xy[lam]
        Xy[lam] = tx
        n_active += 1
        rhs = np.zeros(n_active)
        for a in range(n_active):
            rhs[a] = Xy[a]
        gamma = _potrs(L, n_active, rhs)
        for a in range(n_active):
            path[indices[a], n_active - 1] = gamma[a]
        beta = Gram[:, :n_active] @ gamma
        alpha = Xy - beta
        if tol is not None:
            tol_curr += delta
            delta = 0.0
            for a in range(n_active):
                delta += gamma[a] * beta[a]
            tol_curr -= delta
            if abs(tol_curr) <= tol.__val__():
                break
        elif n_active == max_features:
            break
    return np.ascontiguousarray(path[:, :n_active]), n_active

def _omp_all(X: np.ndarray[float, 2], Y: np.ndarray[float, 2], n_nonzero_coefs: Optional[int],
             tol: Optional[float], use_gram: bool):
    """Returns (list of paths per target, n_iters)."""
    p = X.shape[1]
    nnz = 0
    if n_nonzero_coefs is None and tol is None:
        nnz = max(int(0.1 * float(p)), 1)
    elif n_nonzero_coefs is not None:
        nnz = n_nonzero_coefs.__val__()
    if tol is not None and tol.__val__() < 0.0:
        raise ValueError("Epsilon cannot be negative")
    if tol is None and nnz <= 0:
        raise ValueError("The number of atoms must be positive")
    if tol is None and nnz > p:
        raise ValueError("The number of atoms cannot be more than the number of features")
    paths = List[np.ndarray[float, 2]]()
    iters = List[int]()
    if use_gram:
        G = X.T @ X
        XY = X.T @ Y
        for k in range(Y.shape[1]):
            ns = 0.0
            if tol is not None:
                for i in range(Y.shape[0]):
                    ns += Y[i, k] ** 2
            pth, it = _gram_omp(G, np.ascontiguousarray(XY[:, k]), nnz, ns, tol)
            paths.append(pth)
            iters.append(it)
    else:
        for k in range(Y.shape[1]):
            pth, it = _cholesky_omp(X, np.ascontiguousarray(Y[:, k]), nnz, tol)
            paths.append(pth)
            iters.append(it)
    return paths, iters

def _final(paths: List[np.ndarray[float, 2]], p: int) -> np.ndarray[float, 2]:
    coef = np.zeros((len(paths), p))
    for k in range(len(paths)):
        pth = paths[k]
        if pth.shape[1] > 0:
            coef[k] = pth[:, pth.shape[1] - 1]
    return coef

def orthogonal_mp(X, y, *, n_nonzero_coefs: Optional[int] = None, tol: Optional[float] = None,
                  precompute = False, copy_X: bool = True, return_path: Literal[bool] = False,
                  return_n_iter: Literal[bool] = False):
    Xa = np.asarray(X, dtype=float)
    ya = np.asarray(y, dtype=float)
    Y = _as2d_y(ya)
    use_gram = False
    if isinstance(precompute, str):
        use_gram = Xa.shape[0] > Xa.shape[1]
    else:
        use_gram = bool(precompute)
    paths, iters = _omp_all(Xa, Y, n_nonzero_coefs, tol, use_gram)
    p = Xa.shape[1]
    if return_path:
        if _ndim_of(ya) == 1:
            if return_n_iter:
                return paths[0], iters[0]
            else:
                return paths[0]
        else:
            ns = paths[0].shape[1]
            for q in paths:
                ns = min(ns, q.shape[1])
            out = np.zeros((p, Y.shape[1], ns))
            for k in range(Y.shape[1]):
                for s in range(ns):
                    out[:, k, s] = paths[k][:, s]
            if return_n_iter:
                return out, iters
            else:
                return out
    else:
        coef = _final(paths, p)
        if _ndim_of(ya) == 1:
            if return_n_iter:
                return np.ascontiguousarray(coef[0]), iters[0]
            else:
                return np.ascontiguousarray(coef[0])
        else:
            if return_n_iter:
                return np.ascontiguousarray(coef.T), iters
            else:
                return np.ascontiguousarray(coef.T)

def orthogonal_mp_gram(Gram, Xy, *, n_nonzero_coefs: Optional[int] = None,
                       tol: Optional[float] = None, norms_squared = None,
                       copy_Gram: bool = True, copy_Xy: bool = True,
                       return_path: Literal[bool] = False, return_n_iter: Literal[bool] = False):
    G = np.asarray(Gram, dtype=float)
    xya = np.asarray(Xy, dtype=float)
    XY = _as2d_y(xya)
    p = G.shape[0]
    nnz = max(int(0.1 * float(p)), 1) if (n_nonzero_coefs is None and tol is None) else (0 if n_nonzero_coefs is None else n_nonzero_coefs.__val__())
    if tol is not None and norms_squared is None:
        raise ValueError("Gram OMP needs the precomputed norms in order to evaluate the error sum of squares.")
    nsq = np.zeros(XY.shape[1]) if norms_squared is None else np.atleast_1d(np.asarray(norms_squared, dtype=float))
    paths = List[np.ndarray[float, 2]]()
    iters = List[int]()
    for k in range(XY.shape[1]):
        pth, it = _gram_omp(G, np.ascontiguousarray(XY[:, k]), nnz, nsq[k], tol)
        paths.append(pth)
        iters.append(it)
    coef = _final(paths, p)
    if _ndim_of(xya) == 1:
        if return_n_iter:
            return np.ascontiguousarray(coef[0]), iters[0]
        else:
            return np.ascontiguousarray(coef[0])
    else:
        if return_n_iter:
            return np.ascontiguousarray(coef.T), iters
        else:
            return np.ascontiguousarray(coef.T)

class OrthogonalMatchingPursuit[D: Literal[int] = 1, PC = str](_LinearRegressorMixin):
    n_nonzero_coefs: Optional[int]
    tol: Optional[float]
    fit_intercept: bool
    precompute: PC
    n_nonzero_coefs_: Optional[int]
    _coef: np.ndarray[float, 2]
    _intercept: np.ndarray[float, 1]
    _n_iter: List[int]
    n_features_in_: int

    def __init__(self, *, n_nonzero_coefs: Optional[int] = None, tol: Optional[float] = None,
                 fit_intercept: bool = True, precompute: PC = "auto"):
        self.n_nonzero_coefs = n_nonzero_coefs
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.precompute = precompute
        self.n_nonzero_coefs_ = None
        self._coef = np.zeros((0, 0))
        self._intercept = np.zeros(0)
        self._n_iter = List[int]()
        self.n_features_in_ = 0

    def _nd(self) -> Literal[int]:
        return D

    @property
    def n_iter_(self):
        if D == 1:
            return self._n_iter[0]
        else:
            return self._n_iter

    def fit(self, X, y):
        ya = np.asarray(y, dtype=float)
        if _ndim_of(ya) != D:
            return _retype(self, OrthogonalMatchingPursuit[_ndim_of(ya), PC]).fit(X, ya)
        Y = _as2d_y(ya)
        n = Y.shape[0]
        Xc, Yc, X_offset, y_offset, _ = _preprocess_data(X, Y, self.fit_intercept, np.ones(n),
                                                         False, False)
        p = Xc.shape[1]
        self.n_features_in_ = p
        if self.n_nonzero_coefs is None and self.tol is None:
            self.n_nonzero_coefs_ = max(int(0.1 * float(p)), 1)
        elif self.tol is not None:
            self.n_nonzero_coefs_ = None
        else:
            self.n_nonzero_coefs_ = self.n_nonzero_coefs
        use_gram = False
        pc = self.precompute
        if isinstance(pc, str):
            use_gram = n > p
        elif isinstance(pc, bool):
            use_gram = pc
        else:
            use_gram = True
        paths, iters = _omp_all(Xc, Yc, self.n_nonzero_coefs_, self.tol, use_gram)
        self._n_iter = iters
        coef = _final(paths, p)
        self._coef = coef
        self._intercept = _set_intercept(coef, X_offset, y_offset, self.fit_intercept)
        return self

class OrthogonalMatchingPursuitCV[CV = NoneType](_LinearRegressorMixin):
    copy: bool
    fit_intercept: bool
    max_iter: Optional[int]
    cv: CV
    n_nonzero_coefs_: int
    n_iter_: int
    _coef: np.ndarray[float, 2]
    _intercept: np.ndarray[float, 1]
    n_features_in_: int

    def __init__(self, *, copy: bool = True, fit_intercept: bool = True,
                 max_iter: Optional[int] = None, cv: CV = None, n_jobs = None, verbose = False):
        self.copy = copy
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.cv = cv
        self.n_nonzero_coefs_ = 0
        self.n_iter_ = 0
        self._coef = np.zeros((0, 0))
        self._intercept = np.zeros(0)
        self.n_features_in_ = 0

    def _nd(self) -> Literal[int]:
        return 1

    def fit(self, X, y):
        from ._coordinate_descent import _regressor_folds
        Xa = np.asarray(X, dtype=float)
        ya = np.asarray(y, dtype=float).ravel()
        n, p = Xa.shape
        mi = min(max(int(0.1 * float(p)), 5), p) if (self.max_iter is None or self.max_iter.__val__() == 0) else self.max_iter.__val__()
        folds = _regressor_folds(self.cv, n, Xa, ya)
        mses = List[np.ndarray[float, 1]]()
        for f in folds:
            tr, te = f
            Xtr = Xa[tr]
            ytr = ya[tr]
            Xte = Xa[te]
            yte = ya[te]
            if self.fit_intercept:
                Xm = Xtr.mean(axis=0)
                Xtr = Xtr - Xm
                Xte = Xte - Xm
                ym = float(ytr.mean())
                ytr = ytr - ym
                yte = yte - ym
            pth, _ = _cholesky_omp(Xtr, np.ascontiguousarray(ytr), mi, None)
            R = pth.T @ Xte.T - yte
            mses.append((R ** 2).mean(axis=1))
        ms = mses[0].shape[0]
        for m in mses:
            ms = min(ms, m.shape[0])
        avg = np.zeros(ms)
        for m in mses:
            for i in range(ms):
                avg[i] += m[i]
        for i in range(ms):
            avg[i] /= float(len(mses))
        best = int(np.argmin(avg)) + 1
        self.n_nonzero_coefs_ = best
        omp = OrthogonalMatchingPursuit(n_nonzero_coefs=best, fit_intercept=self.fit_intercept)
        omp.fit(Xa, ya)
        self._coef = omp._coef
        self._intercept = omp._intercept
        self.n_iter_ = omp.n_iter_
        self.n_features_in_ = p
        return self
