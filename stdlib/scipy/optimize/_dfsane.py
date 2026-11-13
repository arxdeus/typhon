# Ported from scipy 1.18.1 (BSD-3-Clause)

# df-sane (derivative-free spectral residual method), fixed_point and the
# public line_search.
# Deviations: real problems only, merit exponent 2, no user eta_strategy / fnorm.

import math
import numpy as np
from ._common import OptimizeResult, _e1, _e2, _norm2, _vec, INF
from ._linesearch import _line_search_wolfe12
from ._nonlin import NoConvergence, _vdot, nonlin_solve, _nl_result, _ArgsFun, \
    BroydenFirst, BroydenSecond, Anderson, LinearMixing, DiagBroyden, \
    ExcitingMixing, KrylovJacobian

class _NoConv(Exception):
    pass

class _DFWrap[F, A]:
    func: F
    args: A
    shape: List[int]
    nfev: int
    maxfev: int

    def __init__(self, func: F, args: A, maxfev: int):
        self.func = func
        self.args = args
        self.shape = List[int]()
        self.nfev = 1
        self.maxfev = maxfev

    def raw(self, x: np.ndarray[float, 1]) -> np.ndarray[float, 1]:
        return np.atleast_1d(np.asarray(self.func(x.copy(), *self.args), float)).flatten()

    def __call__(self, x: np.ndarray[float, 1]):
        if self.nfev >= self.maxfev:
            raise _NoConv()
        self.nfev += 1
        F = self.raw(x)
        n = _norm2(F)
        return n * n, F

def _search_cruz(f, x_k, d, prev_fs: List[float], eta: float):
    gamma = 1e-4
    tau_min = 0.1
    tau_max = 0.5
    f_k = prev_fs[len(prev_fs) - 1]
    f_bar = prev_fs[0]
    for v in prev_fs:
        if v > f_bar:
            f_bar = v
    alpha_p = 1.0
    alpha_m = 1.0
    alpha = 1.0
    xp = x_k
    fp = 0.0
    Fp = x_k
    while True:
        xp = x_k + alpha_p * d
        fp, Fp = f(xp)
        if fp <= f_bar + eta - gamma * alpha_p ** 2 * f_k:
            alpha = alpha_p
            break
        alpha_tp = alpha_p ** 2 * f_k / (fp + (2 * alpha_p - 1) * f_k)
        xp = x_k - alpha_m * d
        fp, Fp = f(xp)
        if fp <= f_bar + eta - gamma * alpha_m ** 2 * f_k:
            alpha = -alpha_m
            break
        alpha_tm = alpha_m ** 2 * f_k / (fp + (2 * alpha_m - 1) * f_k)
        alpha_p = min(max(alpha_tp, tau_min * alpha_p), tau_max * alpha_p)
        alpha_m = min(max(alpha_tm, tau_min * alpha_m), tau_max * alpha_m)
    return alpha, xp, fp, Fp

def _search_cheng(f, x_k, d, f_k: float, C: float, Q: float, eta: float):
    gamma = 1e-4
    tau_min = 0.1
    tau_max = 0.5
    nu = 0.85
    alpha_p = 1.0
    alpha_m = 1.0
    alpha = 1.0
    xp = x_k
    fp = 0.0
    Fp = x_k
    while True:
        xp = x_k + alpha_p * d
        fp, Fp = f(xp)
        if fp <= C + eta - gamma * alpha_p ** 2 * f_k:
            alpha = alpha_p
            break
        alpha_tp = alpha_p ** 2 * f_k / (fp + (2 * alpha_p - 1) * f_k)
        xp = x_k - alpha_m * d
        fp, Fp = f(xp)
        if fp <= C + eta - gamma * alpha_m ** 2 * f_k:
            alpha = -alpha_m
            break
        alpha_tm = alpha_m ** 2 * f_k / (fp + (2 * alpha_m - 1) * f_k)
        alpha_p = min(max(alpha_tp, tau_min * alpha_p), tau_max * alpha_p)
        alpha_m = min(max(alpha_tm, tau_min * alpha_m), tau_max * alpha_m)
    Q_next = nu * Q + 1
    C = (nu * Q * (C + eta) + fp) / Q_next
    Q = Q_next
    return alpha, xp, fp, Fp, C, Q

def _root_df_sane(func, x0, args = (), ftol: float = 1e-8, fatol: float = 1e-300,
                  maxfev: int = 1000, fnorm = None, callback = None,
                  disp: bool = False, M: int = 10, eta_strategy = None,
                  sigma_eps: float = 1e-10, sigma_0: float = 1.0,
                  line_search: str = 'cruz'):
    if line_search != 'cheng' and line_search != 'cruz':
        raise ValueError(f"Invalid value '{line_search}' for 'line_search'")
    w = _DFWrap(func, args, maxfev)
    x_k = _vec(x0)
    F_k = w.raw(x_k)
    n0 = _norm2(F_k)
    f_k = n0 * n0
    k = 0
    f_0 = f_k
    sigma_k = sigma_0
    F_0_norm = math.sqrt(f_0)
    prev_fs = [f_k]
    Q = 1.0
    C = f_0
    converged = False
    message = 'too many function evaluations required'
    while True:
        F_k_norm = math.sqrt(f_k)
        if disp:
            print(f'iter {k}: ||F|| = {F_k_norm:g}, sigma = {sigma_k:g}')
        if callback is not None:
            callback(x_k, F_k)
        if F_k_norm < ftol * F_0_norm + fatol:
            message = 'successful convergence'
            converged = True
            break
        if abs(sigma_k) > 1 / sigma_eps:
            sigma_k = 1 / sigma_eps * (1.0 if sigma_k > 0 else (-1.0 if sigma_k < 0 else 0.0))
        elif abs(sigma_k) < sigma_eps:
            sigma_k = sigma_eps
        d = (-sigma_k) * F_k
        eta = f_0 / (1 + k) ** 2
        try:
            if line_search == 'cruz':
                alpha, xp, fp, Fp = _search_cruz(w, x_k, d, prev_fs, eta)
            else:
                alpha, xp, fp, Fp, C, Q = _search_cheng(w, x_k, d, f_k, C, Q, eta)
        except _NoConv:
            break
        s_k = xp - x_k
        y_k = Fp - F_k
        sigma_k = _vdot(s_k, s_k) / _vdot(s_k, y_k)
        x_k = xp
        F_k = Fp
        f_k = fp
        if line_search == 'cruz':
            prev_fs.append(fp)
            if len(prev_fs) > M:
                prev_fs.pop(0)
        k += 1
    return OptimizeResult(x_k, F_k, _e2(), nit=k, nfev=w.nfev, success=converged,
                          message=message, method='df-sane')

# ------------------------------------------------------------ fixed_point

def _fp_relerr_ok(p: np.ndarray[float, 1], p0: np.ndarray[float, 1], xtol: float) -> bool:
    for i in range(p.size):
        r = p[i]
        if p0[i] != 0:
            r = (p[i] - p0[i]) / p0[i]
        if not (abs(r) < xtol):
            return False
    return True

def fixed_point(func, x0, args = (), xtol: float = 1e-8, maxiter: int = 500,
                method: str = 'del2'):
    if not isinstance(args, Tuple):
        return fixed_point(func, x0, (args,), xtol, maxiter, method)
    r = _fixed_point(func, x0, args, xtol, maxiter, method)
    if isinstance(x0, float) or isinstance(x0, int):
        return r[0]
    else:
        return r

def _fixed_point(func, x0, args, xtol: float, maxiter: int, method: str):
    use_accel = True
    if method == 'iteration':
        use_accel = False
    elif method != 'del2':
        raise KeyError(method)
    p0 = _vec(x0)
    p = p0
    for i in range(maxiter):
        p1 = np.atleast_1d(np.asarray(func(p0.copy(), *args), float)).flatten()
        if use_accel:
            p2 = np.atleast_1d(np.asarray(func(p1.copy(), *args), float)).flatten()
            d = p2 - 2.0 * p1 + p0
            p = p2.copy()
            for j in range(p.size):
                if d[j] != 0:
                    p[j] = p0[j] - (p1[j] - p0[j]) ** 2 / d[j]
        else:
            p = p1
        if _fp_relerr_ok(p, p0, xtol):
            return p
        p0 = p
    raise RuntimeError(f'Failed to converge after {maxiter} iterations, value is {p}')

# ------------------------------------------------------------ line_search

class _LS[F, G, A]:
    f: F
    g: G
    args: A

    def __init__(self, f: F, g: G, args: A):
        self.f = f
        self.g = g
        self.args = args

    def fun(self, x: np.ndarray[float, 1]) -> float:
        return float(self.f(x.copy(), *self.args))

    def grad(self, x: np.ndarray[float, 1]) -> np.ndarray[float, 1]:
        return _vec(self.g(x.copy(), *self.args))

def line_search(f, myfprime, xk, pk, gfk = None, old_fval: Optional[float] = None,
                old_old_fval: Optional[float] = None, args = (), c1: float = 1e-4,
                c2: float = 0.9, amax: Optional[float] = None, extra_condition = None,
                maxiter: int = 10):
    # scipy.optimize.line_search (strong Wolfe). Returns
    # (alpha, fc, gc, new_fval, old_fval, new_slope); alpha is nan on failure.
    # fc/gc are not tracked (0).
    if not isinstance(args, Tuple):
        return line_search(f, myfprime, xk, pk, gfk, old_fval, old_old_fval, (args,),
                           c1, c2, amax, extra_condition, maxiter)
    x = _vec(xk)
    p = _vec(pk)
    sf = _LS(f, myfprime, args)
    g0 = sf.grad(x)
    if gfk is not None:
        g0 = _vec(gfk)
    of = 0.0
    if old_fval is None:
        of = sf.fun(x)
    else:
        of = old_fval
    oof = math.nan
    if old_old_fval is not None:
        oof = old_old_fval
    am = INF if amax is None else amax
    ok, alpha, fval, oldf, gnew, has = _line_search_wolfe12(
        sf, x, p, g0, of, oof, c1, c2, 1e-100, am, _noextra)
    if not ok:
        return math.nan, 0, 0, math.nan, oldf, g0
    return alpha, 0, 0, fval, oldf, gnew

def _noextra(a: float, xk: np.ndarray[float, 1], fv: float, gv: np.ndarray[float, 1]) -> bool:
    return True

# ------------------------------------------------------------ root dispatch

def _optstr(options, key: str, default: str) -> str:
    if options is None:
        return default
    else:
        if key in options:
            return str(options[key])
        return default

def _optopt(options, key: str) -> Optional[float]:
    if options is None:
        return None
    else:
        if key in options:
            return float(options[key])
        return None

def _root_nonlin(fun, x0, args, meth: str, tol: Optional[float], options,
                 callback):
    # options: nit, disp, maxiter, ftol, fatol, xtol, xatol, line_search plus
    # the Jacobian parameters (alpha, reduction_method, max_rank, w0, M,
    # alphamax, rdiff, method, inner_maxiter, outer_k) given directly
    # (scipy nests the latter in options['jac_options']).
    f = _ArgsFun(fun, args)
    f_tol = _optopt(options, 'fatol')
    f_rtol = _optopt(options, 'ftol')
    x_tol = _optopt(options, 'xatol')
    x_rtol = _optopt(options, 'xtol')
    if tol is not None:
        if x_rtol is None:
            x_rtol = tol
        if x_tol is None:
            x_tol = INF
        if f_rtol is None:
            f_rtol = INF
        if f_tol is None:
            f_tol = INF
    nit: Optional[int] = None
    if options is not None:
        if 'nit' in options:
            nit = int(options['nit'])
    maxiter: Optional[int] = None
    if options is not None:
        if 'maxiter' in options:
            maxiter = int(options['maxiter'])
    verbose = False
    if options is not None:
        if 'disp' in options:
            verbose = bool(options['disp'])
    ls: Optional[str] = _optstr(options, 'line_search', 'armijo')
    if ls == 'none':
        ls = None
    alpha = _optopt(options, 'alpha')
    mr: Optional[int] = None
    if options is not None:
        if 'max_rank' in options:
            mr = int(options['max_rank'])
    if meth == 'broyden1':
        x, Fx, st, nit2 = nonlin_solve(f, x0, BroydenFirst(alpha, _optstr(options, 'reduction_method', 'restart'), mr),
                                       nit, verbose, maxiter, f_tol, f_rtol, x_tol, x_rtol, ls, callback, False)
    elif meth == 'broyden2':
        x, Fx, st, nit2 = nonlin_solve(f, x0, BroydenSecond(alpha, _optstr(options, 'reduction_method', 'restart'), mr),
                                       nit, verbose, maxiter, f_tol, f_rtol, x_tol, x_rtol, ls, callback, False)
    elif meth == 'anderson':
        M = 5
        if options is not None:
            if 'M' in options:
                M = int(options['M'])
        w0 = _optopt(options, 'w0')
        x, Fx, st, nit2 = nonlin_solve(f, x0, Anderson(alpha, 0.01 if w0 is None else w0, M),
                                       nit, verbose, maxiter, f_tol, f_rtol, x_tol, x_rtol, ls, callback, False)
    elif meth == 'linearmixing':
        x, Fx, st, nit2 = nonlin_solve(f, x0, LinearMixing(alpha),
                                       nit, verbose, maxiter, f_tol, f_rtol, x_tol, x_rtol, ls, callback, False)
    elif meth == 'diagbroyden':
        x, Fx, st, nit2 = nonlin_solve(f, x0, DiagBroyden(alpha),
                                       nit, verbose, maxiter, f_tol, f_rtol, x_tol, x_rtol, ls, callback, False)
    elif meth == 'excitingmixing':
        am = _optopt(options, 'alphamax')
        x, Fx, st, nit2 = nonlin_solve(f, x0, ExcitingMixing(alpha, 1.0 if am is None else am),
                                       nit, verbose, maxiter, f_tol, f_rtol, x_tol, x_rtol, ls, callback, False)
    else:
        im = 20
        ok = 10
        if options is not None:
            if 'inner_maxiter' in options:
                im = int(options['inner_maxiter'])
            if 'outer_k' in options:
                ok = int(options['outer_k'])
        x, Fx, st, nit2 = nonlin_solve(f, x0, KrylovJacobian(_optopt(options, 'rdiff'), _optstr(options, 'method', 'lgmres'), im, ok),
                                       nit, verbose, maxiter, f_tol, f_rtol, x_tol, x_rtol, ls, callback, False)
    return _nl_result(x, Fx, st, nit2, meth, f.nfev)
