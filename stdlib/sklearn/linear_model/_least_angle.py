# Ported from scikit-learn 1.9.1 (BSD-3-Clause)
#
# Least Angle Regression: lars_path, lars_path_gram, Lars, LassoLars, LarsCV,
# LassoLarsCV, LassoLarsIC (dense inputs).
# DEVIATIONS: Lars/LassoLars keep 1-d targets only through the D=1 path
# (multi-target via D=2 as elsewhere); `jitter` uses RandomState.uniform.

from ..base import estimator_repr as _estimator_repr
import math
import numpy as np
import scipy.linalg as sl
from ._base import (_retype, _ndim_of, _as2d_y, _check_sw, _preprocess_data,
                    _set_intercept, _LinearRegressorMixin, LinearRegression)

_EPS = 2.220446049250313e-16
_TINY32 = 1.1754943508222875e-38
_F32EPS = 1.1920928955078125e-07
_DBL_MAX = 1.7976931348623157e308

def _min_pos(x) -> float:
    m = _DBL_MAX
    for i in range(x.shape[0]):
        if 0.0 < x[i] < m:
            m = x[i]
    return m

def _rotg(a: float, b: float):
    # BLAS drotg (reference / OpenBLAS semantics): returns (r, z, c, s)
    roe = b
    if abs(a) > abs(b):
        roe = a
    scale = abs(a) + abs(b)
    if scale == 0.0:
        return 0.0, 0.0, 1.0, 0.0
    r = scale * math.sqrt((a / scale) ** 2 + (b / scale) ** 2)
    if roe < 0.0:
        r = -r
    c = a / r
    s = b / r
    z = 1.0
    if abs(a) > abs(b):
        z = s
    if abs(b) >= abs(a) and c != 0.0:
        z = 1.0 / c
    return r, z, c, s

def _cholesky_delete(L: np.ndarray[float, 2], n: int, go_out: int):
    # arrayfuncs.cholesky_delete on L[:n, :n] (row-major, row stride L.shape[1])
    for i in range(go_out, n - 1):
        for k in range(i + 2):
            L[i, k] = L[i + 1, k]
    for i in range(go_out, n - 1):
        r, z, c, s = _rotg(L[i, i], L[i, i + 1])
        L[i, i] = r
        L[i, i + 1] = z
        if L[i, i] < 0.0:
            L[i, i] = abs(L[i, i])
            c = -c
            s = -s
        L[i, i + 1] = 0.0
        # rot(n - i - 2, L[i+1:, i], L[i+1:, i+1])
        for k in range(i + 1, n - 1):
            x = L[k, i]
            y = L[k, i + 1]
            L[k, i] = c * x + s * y
            L[k, i + 1] = c * y - s * x

def _solve_tri_lower(L: np.ndarray[float, 2], m: int, b: np.ndarray[float, 1]):
    # forward substitution L[:m,:m] x = b (in place)
    for i in range(m):
        s = b[i]
        for k in range(i):
            s -= L[i, k] * b[k]
        b[i] = s / L[i, i]

def _potrs(L: np.ndarray[float, 2], m: int, rhs: np.ndarray[float, 1]) -> np.ndarray[float, 1]:
    x = rhs.copy()
    _solve_tri_lower(L, m, x)
    for i in range(m - 1, -1, -1):
        s = x[i]
        for k in range(i + 1, m):
            s -= L[k, i] * x[k]
        x[i] = s / L[i, i]
    return x

def _round15(a: np.ndarray[float, 1]) -> np.ndarray[float, 1]:
    return np.round(a, 15)

def _lars_path_solver(X: np.ndarray[float, 2], y: np.ndarray[float, 1], has_X: bool,
                      Xy: Optional[np.ndarray[float, 1]], Gram0: Optional[np.ndarray[float, 2]],
                      n_samples: int, max_iter: int, alpha_min: float, method: str, eps: float,
                      return_path: bool, positive: bool):
    """Returns (alphas, active, coefs (n_features, n_steps) or (n_features,1), n_iter)."""
    if method == "lar" and positive:
        raise ValueError("Positive constraint not supported for 'lar' coding method.")
    Cov = np.zeros(0)
    if Xy is None:
        Cov = X.T @ y
    else:
        Cov = Xy.copy()
    use_gram = Gram0 is not None
    Gram = np.zeros((0, 0))
    XT = np.zeros((0, 0))
    n_features = 0
    if use_gram:
        Gram = Gram0.__val__().copy()
        n_features = Cov.shape[0]
        if Gram.shape[0] != n_features or Gram.shape[1] != n_features:
            raise ValueError("The shapes of the inputs Gram and Xy do not match.")
    else:
        XT = np.ascontiguousarray(X.T)
        n_features = XT.shape[0]
    max_features = min(max_iter, n_features)
    cap = max_features + 1
    coefs = np.zeros((cap, n_features))
    alphas = np.zeros(cap)
    coef = np.zeros(n_features)
    prev_coef = np.zeros(n_features)
    alpha0 = 0.0
    prev_alpha0 = 0.0
    n_iter = 0
    n_active = 0
    active = List[int]()
    indices = np.arange(n_features)
    sign_active = np.zeros(max_features)
    drop = False
    L = np.zeros((max(max_features, 1), max(max_features, 1)))
    Gram_copy = Gram.copy()
    Cov_copy = Cov.copy()
    # Cov is a view Cov_full[cov_off:]
    Cov_full = Cov
    cov_off = 0
    least_squares = np.zeros(0)
    idx = List[int]()
    while True:
        C = 0.0
        C_ = 0.0
        C_idx = 0
        nc = Cov_full.shape[0] - cov_off
        if nc > 0:
            best = cov_off
            for t in range(cov_off, Cov_full.shape[0]):
                v = Cov_full[t] if positive else abs(Cov_full[t])
                bv = Cov_full[best] if positive else abs(Cov_full[best])
                if v > bv:
                    best = t
            C_idx = best - cov_off
            C_ = Cov_full[best]
            C = C_ if positive else abs(C_)
        if return_path:
            alpha0 = alphas[n_iter]
            coef = coefs[n_iter].copy()
            prev_alpha0 = alphas[n_iter - 1] if n_iter > 0 else alphas[cap - 1]
            prev_coef = coefs[n_iter - 1].copy() if n_iter > 0 else coefs[cap - 1].copy()
        alpha0 = C / float(n_samples)
        if return_path:
            alphas[n_iter] = alpha0
        if alpha0 <= alpha_min + _F32EPS:
            if abs(alpha0 - alpha_min) > _F32EPS:
                if n_iter > 0:
                    ss = (prev_alpha0 - alpha_min) / (prev_alpha0 - alpha0)
                    coef = prev_coef + ss * (coef - prev_coef)
                alpha0 = alpha_min
                if return_path:
                    alphas[n_iter] = alpha0
            if return_path:
                coefs[n_iter] = coef
            break
        if n_iter >= max_iter or n_active >= n_features:
            break
        if not drop:
            sign_active[n_active] = 1.0 if positive else (1.0 if C_ > 0.0 else (-1.0 if C_ < 0.0 else 0.0))
            m = n_active
            n = C_idx + n_active
            t0 = Cov_full[cov_off + C_idx]
            Cov_full[cov_off + C_idx] = Cov_full[cov_off]
            Cov_full[cov_off] = t0
            ti = indices[n]
            indices[n] = indices[m]
            indices[m] = ti
            saved_off = cov_off
            cov_off += 1
            c = 0.0
            if not use_gram:
                for k in range(XT.shape[1]):
                    tx = XT[n, k]
                    XT[n, k] = XT[m, k]
                    XT[m, k] = tx
                nr = 0.0
                for k in range(XT.shape[1]):
                    nr += XT[n_active, k] * XT[n_active, k]
                c = math.sqrt(nr) ** 2
                for a in range(n_active):
                    s = 0.0
                    for k in range(XT.shape[1]):
                        s += XT[n_active, k] * XT[a, k]
                    L[n_active, a] = s
            else:
                for k in range(n_features):
                    tg = Gram[m, k]
                    Gram[m, k] = Gram[n, k]
                    Gram[n, k] = tg
                for k in range(n_features):
                    tg = Gram[k, m]
                    Gram[k, m] = Gram[k, n]
                    Gram[k, n] = tg
                c = Gram[n_active, n_active]
                for a in range(n_active):
                    L[n_active, a] = Gram[n_active, a]
            if n_active > 0:
                row = np.zeros(n_active)
                for a in range(n_active):
                    row[a] = L[n_active, a]
                _solve_tri_lower(L, n_active, row)
                for a in range(n_active):
                    L[n_active, a] = row[a]
            v = 0.0
            for a in range(n_active):
                v += L[n_active, a] * L[n_active, a]
            diag = max(math.sqrt(abs(c - v)), eps)
            L[n_active, n_active] = diag
            if diag < 1e-7:
                cov_off = saved_off
                Cov_full[cov_off] = 0.0
                t0 = Cov_full[cov_off + C_idx]
                Cov_full[cov_off + C_idx] = Cov_full[cov_off]
                Cov_full[cov_off] = t0
                continue
            active.append(indices[n_active])
            n_active += 1
        if method == "lasso" and n_iter > 0 and prev_alpha0 < alpha0:
            break
        sa = np.zeros(n_active)
        for a in range(n_active):
            sa[a] = sign_active[a]
        least_squares = _potrs(L, n_active, sa)
        AA = 0.0
        if n_active == 1 and least_squares[0] == 0.0:
            least_squares[0] = 1.0
            AA = 1.0
        else:
            AA = 1.0 / math.sqrt(float(np.sum(least_squares * sa)))
            if not math.isfinite(AA):
                i = 0
                L_ = np.zeros((n_active, n_active))
                for a in range(n_active):
                    for b in range(n_active):
                        L_[a, b] = L[a, b]
                while not math.isfinite(AA):
                    for a in range(n_active):
                        L_[a, a] += (2.0 ** i) * eps
                    least_squares = _potrs(L_, n_active, sa)
                    tmp = max(float(np.sum(least_squares * sa)), eps)
                    AA = 1.0 / math.sqrt(tmp)
                    i += 1
            least_squares *= AA
        nrest = n_features - n_active
        corr_eq_dir = np.zeros(nrest)
        if not use_gram:
            nsmp = XT.shape[1]
            eq_dir = np.zeros(nsmp)
            for a in range(n_active):
                for k in range(nsmp):
                    eq_dir[k] += XT[a, k] * least_squares[a]
            for r in range(nrest):
                s = 0.0
                for k in range(nsmp):
                    s += XT[n_active + r, k] * eq_dir[k]
                corr_eq_dir[r] = s
        else:
            for r in range(nrest):
                s = 0.0
                for a in range(n_active):
                    s += Gram[a, n_active + r] * least_squares[a]
                corr_eq_dir[r] = s
        corr_eq_dir = _round15(corr_eq_dir)
        Covv = np.zeros(nrest)
        for r in range(nrest):
            Covv[r] = Cov_full[cov_off + r]
        g1 = _min_pos((C - Covv) / (AA - corr_eq_dir + _TINY32))
        gamma_ = 0.0
        if positive:
            gamma_ = min(g1, C / AA)
        else:
            g2 = _min_pos((C + Covv) / (AA + corr_eq_dir + _TINY32))
            gamma_ = min(g1, g2, C / AA)
        drop = False
        z = np.zeros(n_active)
        for a in range(n_active):
            z[a] = -coef[active[a]] / (least_squares[a] + _TINY32)
        z_pos = _min_pos(z)
        idx = List[int]()
        if z_pos < gamma_:
            for a in range(n_active - 1, -1, -1):
                if z[a] == z_pos:
                    idx.append(a)
            for a in idx:
                sign_active[a] = -sign_active[a]
            if method == "lasso":
                gamma_ = z_pos
            drop = True
        n_iter += 1
        if return_path:
            if n_iter >= coefs.shape[0]:
                add = 2 * max(1, max_features - n_active)
                nc2 = np.zeros((n_iter + add, n_features))
                na2 = np.zeros(n_iter + add)
                # np.resize repeats the flattened data
                flat = coefs.ravel()
                nf = nc2.ravel()
                for t in range(nf.shape[0]):
                    nf[t] = flat[t % flat.shape[0]]
                nc2 = nf.reshape((n_iter + add, n_features))
                for t in range(na2.shape[0]):
                    na2[t] = alphas[t % alphas.shape[0]]
                for t in range(n_iter + add - add, n_iter + add):
                    for k in range(n_features):
                        nc2[t, k] = 0.0
                    na2[t] = 0.0
                coefs = nc2
                alphas = na2
                cap = coefs.shape[0]
            prev_coef = coefs[n_iter - 1].copy()
            coef = coefs[n_iter].copy()
        else:
            prev_coef = coef
            prev_alpha0 = alpha0
            coef = np.zeros(n_features)
        for a in range(n_active):
            coef[active[a]] = prev_coef[active[a]] + gamma_ * least_squares[a]
        for r in range(nrest):
            Cov_full[cov_off + r] -= gamma_ * corr_eq_dir[r]
        if return_path:
            coefs[n_iter] = coef
        if drop and method == "lasso":
            for ii in idx:
                _cholesky_delete(L, n_active, ii)
            n_active -= 1
            drop_idx = List[int]()
            for ii in idx:
                drop_idx.append(active.pop(ii))
            temp = np.zeros(len(idx))
            if not use_gram:
                for ii in idx:
                    for i in range(ii, n_active):
                        for k in range(XT.shape[1]):
                            tx = XT[i, k]
                            XT[i, k] = XT[i + 1, k]
                            XT[i + 1, k] = tx
                        ti2 = indices[i]
                        indices[i] = indices[i + 1]
                        indices[i + 1] = ti2
                nsmp = XT.shape[1]
                residual = y.copy()
                for a in range(n_active):
                    for k in range(nsmp):
                        residual[k] -= XT[a, k] * coef[active[a]]
                s = 0.0
                for k in range(nsmp):
                    s += XT[n_active, k] * residual[k]
                temp = np.array([s])
            else:
                for ii in idx:
                    for i in range(ii, n_active):
                        ti2 = indices[i]
                        indices[i] = indices[i + 1]
                        indices[i + 1] = ti2
                        for k in range(n_features):
                            tg = Gram[i, k]
                            Gram[i, k] = Gram[i + 1, k]
                            Gram[i + 1, k] = tg
                        for k in range(n_features):
                            tg = Gram[k, i]
                            Gram[k, i] = Gram[k, i + 1]
                            Gram[k, i + 1] = tg
                for q in range(len(drop_idx)):
                    s = Cov_copy[drop_idx[q]]
                    for k in range(n_features):
                        s -= Gram_copy[drop_idx[q], k] * coef[k]
                    temp[q] = s
            # Cov = np.r_[temp, Cov]
            rest = Cov_full[cov_off:].copy()
            newc = np.zeros(temp.shape[0] + rest.shape[0])
            for q in range(temp.shape[0]):
                newc[q] = temp[q]
            for q in range(rest.shape[0]):
                newc[temp.shape[0] + q] = rest[q]
            Cov_full = newc
            cov_off = 0
            # sign_active = append(delete(sign_active, idx), 0)
            keep = List[float]()
            for a in range(sign_active.shape[0]):
                if a not in idx:
                    keep.append(sign_active[a])
            while len(keep) < sign_active.shape[0]:
                keep.append(0.0)
            sign_active = np.array(keep)
    if return_path:
        al = alphas[:n_iter + 1].copy()
        cf = np.ascontiguousarray(coefs[:n_iter + 1].T)
        return al, active, cf, n_iter
    else:
        return np.array([alpha0]), active, coef.reshape((n_features, 1)), n_iter

def _opt_arr1(a) -> Optional[np.ndarray[float, 1]]:
    if a is None:
        return None
    else:
        return np.ascontiguousarray(np.asarray(a, dtype=float).ravel())

def _resolve_gram(Gram, X: np.ndarray[float, 2]) -> Optional[np.ndarray[float, 2]]:
    if Gram is None:
        return None
    elif isinstance(Gram, bool):
        if Gram:
            return X.T @ X
        return None
    elif isinstance(Gram, str):
        if Gram != "auto":
            raise ValueError("Gram must be 'auto', a bool or an array")
        if X.shape[0] > X.shape[1]:
            return X.T @ X
        return None
    else:
        return np.ascontiguousarray(np.asarray(Gram, dtype=float))

def lars_path(X, y, Xy = None, *, Gram = None, max_iter: int = 500, alpha_min: float = 0.0,
              method: str = "lar", copy_X: bool = True, eps: float = _EPS,
              copy_Gram: bool = True, verbose = 0, return_path: Literal[bool] = True,
              return_n_iter: Literal[bool] = False, positive: bool = False):
    Xa = np.array(X, dtype=float)
    ya = np.ascontiguousarray(np.asarray(y, dtype=float).ravel())
    al, active, cf, it = _lars_path_solver(Xa, ya, True, _opt_arr1(Xy), _resolve_gram(Gram, Xa),
                                           ya.shape[0], max_iter, alpha_min, method, eps,
                                           return_path, positive)
    if return_path:
        if return_n_iter:
            return al, active, cf, it
        else:
            return al, active, cf
    else:
        if return_n_iter:
            return al, active, np.ascontiguousarray(cf[:, 0]), it
        else:
            return al, active, np.ascontiguousarray(cf[:, 0])

def lars_path_gram(Xy, Gram, *, n_samples: int, max_iter: int = 500, alpha_min: float = 0.0,
                   method: str = "lar", copy_X: bool = True, eps: float = _EPS,
                   copy_Gram: bool = True, verbose = 0, return_path: Literal[bool] = True,
                   return_n_iter: Literal[bool] = False, positive: bool = False):
    xy = _opt_arr1(Xy)
    G = np.ascontiguousarray(np.asarray(Gram, dtype=float))
    al, active, cf, it = _lars_path_solver(np.zeros((0, 0)), np.zeros(0), False, xy, G,
                                           n_samples, max_iter, alpha_min, method, eps,
                                           return_path, positive)
    if return_path:
        if return_n_iter:
            return al, active, cf, it
        else:
            return al, active, cf
    else:
        if return_n_iter:
            return al, active, np.ascontiguousarray(cf[:, 0]), it
        else:
            return al, active, np.ascontiguousarray(cf[:, 0])

class _LarsMixin:
    @property
    def alphas_(self):
        if self._nd() == 1:
            return self._alphas[0].copy()
        else:
            return self._alphas

    @property
    def active_(self):
        if self._nd() == 1:
            return self._active[0]
        else:
            return self._active

    @property
    def coef_path_(self):
        if self._nd() == 1:
            return self._coef_path[0].copy()
        else:
            return self._coef_path

    @property
    def n_iter_(self):
        if self._nd() == 1:
            return self._n_iter[0]
        else:
            return self._n_iter

    def _fit_lars(self, X, Y: np.ndarray[float, 2], max_iter: int, alpha: float, fit_path: bool,
                  method: str, positive: bool, Xy = None):
        n = Y.shape[0]
        Xc, Yc, X_offset, y_offset, _ = _preprocess_data(X, Y, self.fit_intercept, np.ones(n),
                                                         False, False)
        p = Xc.shape[1]
        k = Yc.shape[1]
        self.n_features_in_ = p
        G: Optional[np.ndarray[float, 2]] = None
        pc = self.precompute
        if isinstance(pc, str):
            if pc == "auto" and (Xc.shape[0] > p or k > 1):
                G = Xc.T @ Xc
        elif isinstance(pc, bool):
            if pc:
                G = Xc.T @ Xc
        else:
            G = np.ascontiguousarray(np.asarray(pc, dtype=float))
        XY = None if Xy is None else _as2d_y(Xy)
        self._alphas = List[np.ndarray[float, 1]]()
        self._active = List[List[int]]()
        self._coef_path = List[np.ndarray[float, 2]]()
        self._n_iter = List[int]()
        coef = np.zeros((k, p))
        for t in range(k):
            xyt: Optional[np.ndarray[float, 1]] = None
            if XY is not None:
                xyt = np.ascontiguousarray(XY.__val__()[:, t])
            yt = np.ascontiguousarray(Yc[:, t])
            al, act, cf, it = _lars_path_solver(Xc, yt, True, xyt, G, n, max_iter, alpha,
                                                method, self.eps, fit_path, positive)
            self._alphas.append(al)
            self._n_iter.append(it)
            if fit_path:
                self._active.append(act)
                self._coef_path.append(cf)
            coef[t] = cf[:, cf.shape[1] - 1]
        self._coef = coef
        self._intercept = _set_intercept(coef, X_offset, y_offset, self.fit_intercept)
        return self

    def _jitter_y(self, ya):
        if self.jitter is None:
            return ya
        else:
            rng = np.random.RandomState(0 if self.random_state is None else self.random_state.__val__())
            noise = rng.uniform(high=float(self.jitter.__val__()), size=ya.shape[0])
            if _ndim_of(ya) == 1:
                return ya + noise
            else:
                return ya + noise[:, None]

class Lars[D: Literal[int] = 1, PC = str](_LarsMixin, _LinearRegressorMixin):
    def __repr__(self) -> str:
        return _estimator_repr(self)

    fit_intercept: bool
    verbose: int
    precompute: PC
    n_nonzero_coefs: int
    eps: float
    copy_X: bool
    fit_path: bool
    jitter: Optional[float]
    random_state: Optional[int]
    _coef: np.ndarray[float, 2]
    _intercept: np.ndarray[float, 1]
    _alphas: List[np.ndarray[float, 1]]
    _active: List[List[int]]
    _coef_path: List[np.ndarray[float, 2]]
    _n_iter: List[int]
    n_features_in_: int

    def __init__(self, *, fit_intercept: bool = True, verbose = False, precompute: PC = "auto",
                 n_nonzero_coefs: int = 500, eps: float = _EPS, copy_X: bool = True,
                 fit_path: bool = True, jitter: Optional[float] = None,
                 random_state: Optional[int] = None):
        self.fit_intercept = fit_intercept
        self.verbose = int(verbose)
        self.precompute = precompute
        self.n_nonzero_coefs = n_nonzero_coefs
        self.eps = eps
        self.copy_X = copy_X
        self.fit_path = fit_path
        self.jitter = jitter
        self.random_state = random_state
        self._coef = np.zeros((0, 0))
        self._intercept = np.zeros(0)
        self._alphas = List[np.ndarray[float, 1]]()
        self._active = List[List[int]]()
        self._coef_path = List[np.ndarray[float, 2]]()
        self._n_iter = List[int]()
        self.n_features_in_ = 0

    def _nd(self) -> Literal[int]:
        return D

    def fit(self, X, y, Xy = None):
        ya = np.asarray(y, dtype=float)
        if _ndim_of(ya) != D:
            return _retype(self, Lars[_ndim_of(ya), PC]).fit(X, ya, Xy)
        ya = self._jitter_y(ya)
        return self._fit_lars(X, _as2d_y(ya), self.n_nonzero_coefs, 0.0, self.fit_path, "lar",
                              False, Xy)

class LassoLars[D: Literal[int] = 1, PC = str](_LarsMixin, _LinearRegressorMixin):
    def __repr__(self) -> str:
        return _estimator_repr(self)

    alpha: float
    fit_intercept: bool
    verbose: int
    precompute: PC
    max_iter: int
    eps: float
    copy_X: bool
    fit_path: bool
    positive: bool
    jitter: Optional[float]
    random_state: Optional[int]
    _coef: np.ndarray[float, 2]
    _intercept: np.ndarray[float, 1]
    _alphas: List[np.ndarray[float, 1]]
    _active: List[List[int]]
    _coef_path: List[np.ndarray[float, 2]]
    _n_iter: List[int]
    n_features_in_: int

    def __init__(self, alpha: float = 1.0, *, fit_intercept: bool = True, verbose = False,
                 precompute: PC = "auto", max_iter: int = 500, eps: float = _EPS,
                 copy_X: bool = True, fit_path: bool = True, positive: bool = False,
                 jitter: Optional[float] = None, random_state: Optional[int] = None):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.verbose = int(verbose)
        self.precompute = precompute
        self.max_iter = max_iter
        self.eps = eps
        self.copy_X = copy_X
        self.fit_path = fit_path
        self.positive = positive
        self.jitter = jitter
        self.random_state = random_state
        self._coef = np.zeros((0, 0))
        self._intercept = np.zeros(0)
        self._alphas = List[np.ndarray[float, 1]]()
        self._active = List[List[int]]()
        self._coef_path = List[np.ndarray[float, 2]]()
        self._n_iter = List[int]()
        self.n_features_in_ = 0

    def _nd(self) -> Literal[int]:
        return D

    def fit(self, X, y, Xy = None):
        ya = np.asarray(y, dtype=float)
        if _ndim_of(ya) != D:
            return _retype(self, LassoLars[_ndim_of(ya), PC]).fit(X, ya, Xy)
        ya = self._jitter_y(ya)
        return self._fit_lars(X, _as2d_y(ya), self.max_iter, self.alpha, self.fit_path,
                              "lasso", self.positive, Xy)

def _interp_linear(x: np.ndarray[float, 1], Y: np.ndarray[float, 2],
                   xn: np.ndarray[float, 1]) -> np.ndarray[float, 2]:
    # scipy.interpolate.interp1d(x, Y, axis=0)(xn), linear; x sorted stably first
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    Ys = Y[order]
    m = xs.shape[0]
    out = np.zeros((xn.shape[0], Y.shape[1]))
    for i in range(xn.shape[0]):
        if xn[i] < xs[0] or xn[i] > xs[m - 1]:
            raise ValueError("A value in x_new is out of the interpolation range.")
        hi = int(np.searchsorted(xs, xn[i]))
        if hi < 1:
            hi = 1
        if hi > m - 1:
            hi = m - 1
        lo = hi - 1
        a = (xn[i] - xs[lo]) / (xs[hi] - xs[lo])
        b = (xs[hi] - xn[i]) / (xs[hi] - xs[lo])
        for j in range(Y.shape[1]):
            out[i, j] = a * Ys[hi, j] + b * Ys[lo, j]
    return out

def _lars_cv_fit(est, X, y, method: str, positive: bool):
    from ._coordinate_descent import _regressor_folds
    Xa = np.array(X, dtype=float)
    ya = np.array(y, dtype=float).ravel()
    n, p = Xa.shape
    folds = _regressor_folds(est.cv, n, Xa, ya)
    paths_a = List[np.ndarray[float, 1]]()
    paths_r = List[np.ndarray[float, 2]]()
    for f in folds:
        tr, te = f
        Xtr = Xa[tr]
        ytr = ya[tr]
        Xte = Xa[te]
        yte = ya[te]
        if est.fit_intercept:
            Xm = Xtr.mean(axis=0)
            Xtr = Xtr - Xm
            Xte = Xte - Xm
            ym = float(ytr.mean())
            ytr = ytr - ym
            yte = yte - ym
        G = _resolve_gram(est.precompute if not _is_array(est.precompute) else "auto", Xtr)
        al, _, cf, _ = _lars_path_solver(Xtr, np.ascontiguousarray(ytr), True, None, G,
                                         ytr.shape[0], est.max_iter, 0.0, method, est.eps, True,
                                         positive)
        res = Xte @ cf - yte[:, None]
        paths_a.append(al)
        paths_r.append(np.ascontiguousarray(res.T))
    tot = 0
    for a in paths_a:
        tot += a.shape[0]
    cat = np.zeros(tot)
    q = 0
    for a in paths_a:
        for i in range(a.shape[0]):
            cat[q] = a[i]
            q += 1
    all_alphas = np.unique(cat)
    stride = int(max(1, int(float(all_alphas.shape[0]) / float(est.max_n_alphas))))
    all_alphas = np.ascontiguousarray(all_alphas[::stride])
    na = all_alphas.shape[0]
    nf = len(folds)
    mse = np.zeros((na, nf))
    last = all_alphas[na - 1]
    for fi in range(nf):
        al0 = paths_a[fi]
        r0 = paths_r[fi]
        m = al0.shape[0]
        lst_a = List[float]()
        lst_r = List[np.ndarray[float, 1]]()
        for i in range(m - 1, -1, -1):
            lst_a.append(al0[i])
            lst_r.append(r0[i].copy())
        if lst_a[0] != 0.0:
            lst_a.insert(0, 0.0)
            lst_r.insert(0, lst_r[0].copy())
        if lst_a[len(lst_a) - 1] != last:
            lst_a.append(last)
            lst_r.append(lst_r[len(lst_r) - 1].copy())
        aa = np.array(lst_a)
        RR = np.zeros((len(lst_r), r0.shape[1]))
        for i in range(len(lst_r)):
            RR[i] = lst_r[i]
        tr2 = _interp_linear(aa, RR, all_alphas) ** 2
        mse[:, fi] = np.mean(tr2, axis=-1)
    keep = List[int]()
    for i in range(na):
        ok = True
        for fi in range(nf):
            if not math.isfinite(mse[i, fi]):
                ok = False
        if ok:
            keep.append(i)
    ki = np.array(keep)
    all_alphas = all_alphas[ki]
    mse = mse[ki]
    best = int(np.argmin(mse.mean(axis=-1)))
    est.alpha_ = float(all_alphas[best])
    est.cv_alphas_ = all_alphas
    est.mse_path_ = mse
    return est.alpha_

def _is_array(x) -> bool:
    if isinstance(x, str) or isinstance(x, bool):
        return False
    return True

class LarsCV[CV = NoneType, PC = str](_LarsMixin, _LinearRegressorMixin):
    def __repr__(self) -> str:
        return _estimator_repr(self)

    fit_intercept: bool
    verbose: int
    max_iter: int
    precompute: PC
    cv: CV
    max_n_alphas: int
    eps: float
    copy_X: bool
    alpha_: float
    cv_alphas_: np.ndarray[float, 1]
    mse_path_: np.ndarray[float, 2]
    _coef: np.ndarray[float, 2]
    _intercept: np.ndarray[float, 1]
    _alphas: List[np.ndarray[float, 1]]
    _active: List[List[int]]
    _coef_path: List[np.ndarray[float, 2]]
    _n_iter: List[int]
    n_features_in_: int

    def __init__(self, *, fit_intercept: bool = True, verbose = False, max_iter: int = 500,
                 precompute: PC = "auto", cv: CV = None, max_n_alphas: int = 1000,
                 n_jobs = None, eps: float = _EPS, copy_X: bool = True):
        self.fit_intercept = fit_intercept
        self.verbose = int(verbose)
        self.max_iter = max_iter
        self.precompute = precompute
        self.cv = cv
        self.max_n_alphas = max_n_alphas
        self.eps = eps
        self.copy_X = copy_X
        self.alpha_ = 0.0
        self.cv_alphas_ = np.zeros(0)
        self.mse_path_ = np.zeros((0, 0))
        self._coef = np.zeros((0, 0))
        self._intercept = np.zeros(0)
        self._alphas = List[np.ndarray[float, 1]]()
        self._active = List[List[int]]()
        self._coef_path = List[np.ndarray[float, 2]]()
        self._n_iter = List[int]()
        self.n_features_in_ = 0

    def _nd(self) -> Literal[int]:
        return 1

    def fit(self, X, y):
        a = _lars_cv_fit(self, X, y, "lar", False)
        return self._fit_lars(X, _as2d_y(y), self.max_iter, a, True, "lar", False)

class LassoLarsCV[CV = NoneType, PC = str](_LarsMixin, _LinearRegressorMixin):
    def __repr__(self) -> str:
        return _estimator_repr(self)

    fit_intercept: bool
    verbose: int
    max_iter: int
    precompute: PC
    cv: CV
    max_n_alphas: int
    eps: float
    copy_X: bool
    positive: bool
    alpha_: float
    cv_alphas_: np.ndarray[float, 1]
    mse_path_: np.ndarray[float, 2]
    _coef: np.ndarray[float, 2]
    _intercept: np.ndarray[float, 1]
    _alphas: List[np.ndarray[float, 1]]
    _active: List[List[int]]
    _coef_path: List[np.ndarray[float, 2]]
    _n_iter: List[int]
    n_features_in_: int

    def __init__(self, *, fit_intercept: bool = True, verbose = False, max_iter: int = 500,
                 precompute: PC = "auto", cv: CV = None, max_n_alphas: int = 1000,
                 n_jobs = None, eps: float = _EPS, copy_X: bool = True, positive: bool = False):
        self.fit_intercept = fit_intercept
        self.verbose = int(verbose)
        self.max_iter = max_iter
        self.precompute = precompute
        self.cv = cv
        self.max_n_alphas = max_n_alphas
        self.eps = eps
        self.copy_X = copy_X
        self.positive = positive
        self.alpha_ = 0.0
        self.cv_alphas_ = np.zeros(0)
        self.mse_path_ = np.zeros((0, 0))
        self._coef = np.zeros((0, 0))
        self._intercept = np.zeros(0)
        self._alphas = List[np.ndarray[float, 1]]()
        self._active = List[List[int]]()
        self._coef_path = List[np.ndarray[float, 2]]()
        self._n_iter = List[int]()
        self.n_features_in_ = 0

    def _nd(self) -> Literal[int]:
        return 1

    def fit(self, X, y):
        a = _lars_cv_fit(self, X, y, "lasso", self.positive)
        return self._fit_lars(X, _as2d_y(y), self.max_iter, a, True, "lasso", self.positive)

class LassoLarsIC[PC = str](_LinearRegressorMixin):
    def __repr__(self) -> str:
        return _estimator_repr(self)

    criterion: str
    fit_intercept: bool
    verbose: int
    precompute: PC
    max_iter: int
    eps: float
    copy_X: bool
    positive: bool
    noise_variance: Optional[float]
    alpha_: float
    alphas_: np.ndarray[float, 1]
    criterion_: np.ndarray[float, 1]
    noise_variance_: float
    n_iter_: int
    _coef: np.ndarray[float, 2]
    _intercept: np.ndarray[float, 1]
    n_features_in_: int

    def __init__(self, criterion: str = "aic", *, fit_intercept: bool = True, verbose = False,
                 precompute: PC = "auto", max_iter: int = 500, eps: float = _EPS,
                 copy_X: bool = True, positive: bool = False,
                 noise_variance: Optional[float] = None):
        self.criterion = criterion
        self.fit_intercept = fit_intercept
        self.verbose = int(verbose)
        self.precompute = precompute
        self.max_iter = max_iter
        self.eps = eps
        self.copy_X = copy_X
        self.positive = positive
        self.noise_variance = noise_variance
        self.alpha_ = 0.0
        self.alphas_ = np.zeros(0)
        self.criterion_ = np.zeros(0)
        self.noise_variance_ = 0.0
        self.n_iter_ = 0
        self._coef = np.zeros((0, 0))
        self._intercept = np.zeros(0)
        self.n_features_in_ = 0

    def _nd(self) -> Literal[int]:
        return 1

    def fit(self, X, y, copy_X = None):
        Y = _as2d_y(y)
        n = Y.shape[0]
        Xc, Yc, X_offset, y_offset, _ = _preprocess_data(X, Y, self.fit_intercept, np.ones(n),
                                                         False, False)
        p = Xc.shape[1]
        self.n_features_in_ = p
        yc = np.ascontiguousarray(Yc[:, 0])
        G = _resolve_gram(self.precompute, Xc) if _is_array(self.precompute) else None
        if not _is_array(self.precompute):
            pc = self.precompute
            if isinstance(pc, bool):
                if pc:
                    G = Xc.T @ Xc
        al, _, cf, it = _lars_path_solver(Xc, yc, True, None, G, n, self.max_iter, 0.0, "lasso",
                                          self.eps, True, self.positive)
        self.n_iter_ = it
        factor = 0.0
        if self.criterion == "aic":
            factor = 2.0
        elif self.criterion == "bic":
            factor = math.log(float(n))
        else:
            raise ValueError(f"criterion should be either bic or aic, got {self.criterion!r}")
        res = yc[:, None] - Xc @ cf
        rss = np.sum(res ** 2, axis=0)
        ns = cf.shape[1]
        dof = np.zeros(ns)
        for k in range(ns):
            c = 0
            for j in range(p):
                if abs(cf[j, k]) > _EPS:
                    c += 1
            dof[k] = float(c)
        self.alphas_ = al
        if self.noise_variance is None:
            if n <= p + int(self.fit_intercept):
                raise ValueError("You are using LassoLarsIC in the case where the number of samples is smaller than the number of features. In this setting, getting a good estimate for the variance of the noise is not possible. Provide an estimate of the noise variance in the constructor.")
            ols = LinearRegression(positive=self.positive, fit_intercept=False).fit(Xc, yc)
            r = yc - ols.predict(Xc)
            self.noise_variance_ = float(np.sum(r ** 2)) / float(n - p - int(self.fit_intercept))
        else:
            self.noise_variance_ = self.noise_variance.__val__()
        self.criterion_ = (float(n) * math.log(2.0 * math.pi * self.noise_variance_)
                           + rss / self.noise_variance_ + factor * dof)
        nb = int(np.argmin(self.criterion_))
        self.alpha_ = float(al[nb])
        coef = np.zeros((1, p))
        coef[0] = cf[:, nb]
        self._coef = coef
        self._intercept = _set_intercept(coef, X_offset, y_offset, self.fit_intercept)
        return self
