# Ported from scipy 1.18.1 (BSD-3-Clause)

# Unconstrained multivariate minimizers: Nelder-Mead, BFGS, CG, Powell.

import math
import numpy as np
from ._common import *
from ._common import (_ScalarFunction, _approx_grad, _approx_jac, _bound_val, _call_scalar, _call_vec, _e1, _e2, _eps_for_method, _epsilon, _fd_steps, _has_opt, _norm2, _norm_inf, _old_bound_to_new, _opt_bool, _opt_float, _opt_int, _prepare_bounds_pair, _scalar_eps, _sign, _sqrt, _standardize_bounds, _status_message_maxfev, _status_message_maxiter, _status_message_nan, _status_message_oob, _status_message_pr_loss, _status_message_success, _vec)
from ._linesearch import _line_search_wolfe12
from ._callback import _cb_halt, _STOP_MSG
from ._scalar import _minimize_scalar_brent_impl, _minimize_scalar_bounded_impl

class _MaxFuncCallError(RuntimeError):
    def __init__(self, message: str = ''):
        super().__init__(message)

class _CountedFun[FN, A]:
    fun: FN
    args: A
    ncalls: int
    maxfun: float

    def __init__(self, fun: FN, args: A, maxfun: float):
        self.fun = fun
        self.args = args
        self.ncalls = 0
        self.maxfun = maxfun

    def __call__(self, x) -> float:
        if float(self.ncalls) >= self.maxfun:
            raise _MaxFuncCallError('Too many function calls')
        self.ncalls += 1
        return _call_scalar(self.fun, x.copy(), self.args)

def _no_bounds():
    return Optional[Tuple[np.ndarray[float, 1], np.ndarray[float, 1]]]()

# -------------------------------------------------------------- Nelder-Mead

def _minimize_neldermead(func, x0: np.ndarray[float, 1], args, options,
                         bnds: Optional[Tuple[np.ndarray[float, 1], np.ndarray[float, 1]]],
                         callback = None,
                         initial_simplex: np.ndarray[float, 2] = np.empty((0, 0), float)):
    xatol = _opt_float(options, 'xatol', 1e-4)
    fatol = _opt_float(options, 'fatol', 1e-4)
    adaptive = _opt_bool(options, 'adaptive', False)
    N = len(x0)
    rho = 1.0
    chi = 2.0
    psi = 0.5
    sigma = 0.5
    if adaptive:
        dim = float(N)
        chi = 1 + 2 / dim
        psi = 0.75 - 1 / (2 * dim)
        sigma = 1 - 1 / dim
    nonzdelt = 0.05
    zdelt = 0.00025
    has_b = bnds is not None
    lower_bound = np.full(N, -INF)
    upper_bound = np.full(N, INF)
    if bnds is not None:
        lower_bound, upper_bound = bnds
        for i in range(N):
            if lower_bound[i] > upper_bound[i]:
                raise ValueError("Nelder Mead - one of the lower bounds is greater than an upper bound.")
        x0 = np.clip(x0, lower_bound, upper_bound)

    sim = np.empty((N + 1, N), float)
    if initial_simplex.size == 0:
        sim[0] = x0
        for k in range(N):
            y = x0.copy()
            if y[k] != 0:
                y[k] = (1 + nonzdelt) * y[k]
            else:
                y[k] = zdelt
            sim[k + 1] = y
    else:
        s0 = initial_simplex
        if s0.shape[0] != s0.shape[1] + 1:
            raise ValueError('`initial_simplex` should be an array of shape (N+1,N)')
        if N != s0.shape[1]:
            raise ValueError('Size of `initial_simplex` is not consistent with `x0`')
        sim = s0.copy()

    has_mi = _has_opt(options, 'maxiter')
    has_mf = _has_opt(options, 'maxfev')
    maxiter = INF
    maxfun = INF
    if not has_mi and not has_mf:
        maxiter = float(N * 200)
        maxfun = float(N * 200)
    elif not has_mi:
        maxfun = _opt_float(options, 'maxfev', 0.0)
        maxiter = float(N * 200) if maxfun == INF else INF
    elif not has_mf:
        maxiter = _opt_float(options, 'maxiter', 0.0)
        maxfun = float(N * 200) if maxiter == INF else INF
    else:
        maxiter = _opt_float(options, 'maxiter', 0.0)
        maxfun = _opt_float(options, 'maxfev', 0.0)

    if has_b:
        for k in range(N + 1):
            for j in range(N):
                if sim[k, j] > upper_bound[j]:
                    sim[k, j] = 2 * upper_bound[j] - sim[k, j]
                sim[k, j] = min(max(sim[k, j], lower_bound[j]), upper_bound[j])

    fsim = np.full(N + 1, INF)
    fc = _CountedFun(func, args, maxfun)
    try:
        for k in range(N + 1):
            fsim[k] = fc(sim[k])
    except _MaxFuncCallError:
        pass
    ind = np.argsort(fsim, kind='stable')
    sim = np.take(sim, ind, 0)
    fsim = np.take(fsim, ind, 0)
    ind = np.argsort(fsim, kind='stable')
    fsim = np.take(fsim, ind, 0)
    sim = np.take(sim, ind, 0)

    iterations = 1
    stopped = False
    while float(fc.ncalls) < maxfun and float(iterations) < maxiter:
        try:
            if (np.max(np.ravel(np.abs(sim[1:] - sim[0]))) <= xatol and
                    np.max(np.abs(fsim[0] - fsim[1:])) <= fatol):
                break
            xbar = np.add.reduce(sim[:-1], 0) / N
            xr = (1 + rho) * xbar - rho * sim[-1]
            if has_b:
                xr = np.clip(xr, lower_bound, upper_bound)
            fxr = fc(xr)
            doshrink = False
            if fxr < fsim[0]:
                xe = (1 + rho * chi) * xbar - rho * chi * sim[-1]
                if has_b:
                    xe = np.clip(xe, lower_bound, upper_bound)
                fxe = fc(xe)
                if fxe < fxr:
                    sim[-1] = xe
                    fsim[-1] = fxe
                else:
                    sim[-1] = xr
                    fsim[-1] = fxr
            else:
                if fxr < fsim[-2]:
                    sim[-1] = xr
                    fsim[-1] = fxr
                else:
                    if fxr < fsim[-1]:
                        xc = (1 + psi * rho) * xbar - psi * rho * sim[-1]
                        if has_b:
                            xc = np.clip(xc, lower_bound, upper_bound)
                        fxc = fc(xc)
                        if fxc <= fxr:
                            sim[-1] = xc
                            fsim[-1] = fxc
                        else:
                            doshrink = True
                    else:
                        xcc = (1 - psi) * xbar + psi * sim[-1]
                        if has_b:
                            xcc = np.clip(xcc, lower_bound, upper_bound)
                        fxcc = fc(xcc)
                        if fxcc < fsim[-1]:
                            sim[-1] = xcc
                            fsim[-1] = fxcc
                        else:
                            doshrink = True
                    if doshrink:
                        for j in range(1, N + 1):
                            sim[j] = sim[0] + sigma * (sim[j] - sim[0])
                            if has_b:
                                sim[j] = np.clip(sim[j], lower_bound, upper_bound)
                            fsim[j] = fc(sim[j])
            iterations += 1
        except _MaxFuncCallError:
            pass
        ind = np.argsort(fsim, kind='stable')
        sim = np.take(sim, ind, 0)
        fsim = np.take(fsim, ind, 0)
        if _cb_halt(callback, sim[0], fsim[0]):
            stopped = True
            break

    x = sim[0].copy()
    fval = float(np.min(fsim))
    warnflag = 0
    msg = _status_message_success
    if float(fc.ncalls) >= maxfun:
        warnflag = 1
        msg = _status_message_maxfev
    elif float(iterations) >= maxiter:
        warnflag = 2
        msg = _status_message_maxiter
    r = OptimizeResult(x, fval, _e1(), nit=iterations, nfev=fc.ncalls,
                       status=warnflag, success=(warnflag == 0),
                       message=msg)
    if stopped:
        r.success = False
        r.status = 99
        r.message = _STOP_MSG
    return r

# --------------------------------------------------------------------- BFGS

def _true4(a: float, x: np.ndarray[float, 1], f: float, g: np.ndarray[float, 1]) -> bool:
    return True

def _minimize_bfgs(fun, x0: np.ndarray[float, 1], args, jac, options, callback = None):
    gtol = _opt_float(options, 'gtol', 1e-05)
    norm = _opt_float(options, 'norm', INF)
    eps = _opt_float(options, 'eps', _epsilon)
    maxiter = _opt_int(options, 'maxiter', len(x0) * 200)
    xrtol = _opt_float(options, 'xrtol', 0.0)
    c1 = _opt_float(options, 'c1', 1e-4)
    c2 = _opt_float(options, 'c2', 0.9)
    frs: Optional[float] = None
    if _has_opt(options, 'finite_diff_rel_step'):
        frs = _opt_float(options, 'finite_diff_rel_step', 0.0)
    sf = _ScalarFunction(fun, x0, args, jac, eps, frs)
    old_fval = sf.fun(x0)
    gfk = sf.grad(x0)
    k = 0
    N = len(x0)
    I = np.eye(N)
    Hk = np.eye(N)
    old_old_fval = old_fval + _norm2(gfk) / 2
    xk = x0
    warnflag = 0
    gnorm = vecnorm(gfk, norm)
    while gnorm > gtol and k < maxiter:
        pk = -np.dot(Hk, gfk)
        ok, alpha_k, new_f, prev_f, gfkp1, has_g = _line_search_wolfe12(
            sf, xk, pk, gfk, old_fval, old_old_fval, c1, c2, 1e-100, 1e100,
            _true4)
        if not ok:
            warnflag = 2
            break
        old_fval = new_f
        old_old_fval = prev_f
        sk = alpha_k * pk
        xkp1 = xk + sk
        xk = xkp1
        if not has_g:
            gfkp1 = sf.grad(xkp1)
        yk = gfkp1 - gfk
        gfk = gfkp1
        k += 1
        if _cb_halt(callback, xk, old_fval):
            warnflag = 99
            break
        gnorm = vecnorm(gfk, norm)
        if gnorm <= gtol:
            break
        if alpha_k * vecnorm(pk) <= xrtol * (xrtol + vecnorm(xk)):
            break
        if not (old_fval == old_fval and abs(old_fval) != INF):
            warnflag = 2
            break
        rhok_inv = float(np.dot(yk, sk))
        rhok = 1000.0
        if rhok_inv != 0.0:
            rhok = 1.0 / rhok_inv
        A1 = I - sk[:, None] * yk[None, :] * rhok
        A2 = I - yk[:, None] * sk[None, :] * rhok
        Hk = np.dot(A1, np.dot(Hk, A2)) + (rhok * sk[:, None] * sk[None, :])
    fval = old_fval
    msg = _status_message_success
    if warnflag == 99:
        msg = _STOP_MSG
    elif warnflag == 2:
        msg = _status_message_pr_loss
    elif k >= maxiter:
        warnflag = 1
        msg = _status_message_maxiter
    elif gnorm != gnorm or fval != fval or np.isnan(xk).any():
        warnflag = 3
        msg = _status_message_nan
    return OptimizeResult(xk, fval, gfk, nit=k, nfev=sf.nfev, njev=sf.ngev,
                          status=warnflag, success=(warnflag == 0),
                          message=msg, hess_inv=Hk)

# ----------------------------------------------------------------------- CG

class _PRPStep:
    # Polak-Ribiere+ step with the descent-condition cache used by CG
    xk: np.ndarray[float, 1]
    pk: np.ndarray[float, 1]
    gfk: np.ndarray[float, 1]
    deltak: float
    norm: float
    gtol: float
    sigma_3: float
    valid: bool
    c_alpha: float
    c_x: np.ndarray[float, 1]
    c_p: np.ndarray[float, 1]
    c_g: np.ndarray[float, 1]
    c_n: float

    def __init__(self, xk: np.ndarray[float, 1], pk: np.ndarray[float, 1],
                 gfk: np.ndarray[float, 1], deltak: float, norm: float,
                 gtol: float, sigma_3: float):
        self.xk = xk
        self.pk = pk
        self.gfk = gfk
        self.deltak = deltak
        self.norm = norm
        self.gtol = gtol
        self.sigma_3 = sigma_3
        self.valid = False
        self.c_alpha = 0.0
        self.c_x = xk
        self.c_p = pk
        self.c_g = gfk
        self.c_n = 0.0

    def step(self, alpha: float, gfkp1: np.ndarray[float, 1]):
        xkp1 = self.xk + alpha * self.pk
        yk = gfkp1 - self.gfk
        beta_k = max(0.0, float(np.dot(yk, gfkp1)) / self.deltak)
        pkp1 = -gfkp1 + beta_k * self.pk
        gn = vecnorm(gfkp1, self.norm)
        return alpha, xkp1, pkp1, gfkp1, gn

    def __call__(self, alpha: float, xkp1: np.ndarray[float, 1], fp1: float,
                 gfkp1: np.ndarray[float, 1]) -> bool:
        a, xx, pp, gg, gn = self.step(alpha, gfkp1)
        self.valid = True
        self.c_alpha = a
        self.c_x = xx
        self.c_p = pp
        self.c_g = gg
        self.c_n = gn
        if gn <= self.gtol:
            return True
        return float(np.dot(pp, gg)) <= -self.sigma_3 * float(np.dot(gg, gg))

def _minimize_cg(fun, x0: np.ndarray[float, 1], args, jac, options, callback = None):
    gtol = _opt_float(options, 'gtol', 1e-05)
    norm = _opt_float(options, 'norm', INF)
    eps = _opt_float(options, 'eps', _epsilon)
    maxiter = _opt_int(options, 'maxiter', len(x0) * 200)
    c1 = _opt_float(options, 'c1', 1e-4)
    c2 = _opt_float(options, 'c2', 0.4)
    frs: Optional[float] = None
    if _has_opt(options, 'finite_diff_rel_step'):
        frs = _opt_float(options, 'finite_diff_rel_step', 0.0)
    sf = _ScalarFunction(fun, x0, args, jac, eps, frs)
    old_fval = sf.fun(x0)
    gfk = sf.grad(x0)
    k = 0
    xk = x0
    old_old_fval = old_fval + _norm2(gfk) / 2
    warnflag = 0
    pk = -gfk
    gnorm = vecnorm(gfk, norm)
    sigma_3 = 0.01
    while gnorm > gtol and k < maxiter:
        cs = _PRPStep(xk, pk, gfk, float(np.dot(gfk, gfk)), norm, gtol, sigma_3)
        ok, alpha_k, new_f, prev_f, gfkp1, has_g = _line_search_wolfe12(
            sf, xk, pk, gfk, old_fval, old_old_fval, c1, c2, 1e-100, 1e100,
            cs)
        if not ok:
            warnflag = 2
            break
        old_fval = new_f
        old_old_fval = prev_f
        if cs.valid and alpha_k == cs.c_alpha:
            xk = cs.c_x
            pk = cs.c_p
            gfk = cs.c_g
            gnorm = cs.c_n
        else:
            if not has_g:
                gfkp1 = sf.grad(xk + alpha_k * pk)
            a, xx, pp, gg, gn = cs.step(alpha_k, gfkp1)
            xk = xx
            pk = pp
            gfk = gg
            gnorm = gn
        k += 1
        if _cb_halt(callback, xk, old_fval):
            warnflag = 99
            break
    fval = old_fval
    msg = _status_message_success
    if warnflag == 99:
        msg = _STOP_MSG
    elif warnflag == 2:
        msg = _status_message_pr_loss
    elif k >= maxiter:
        warnflag = 1
        msg = _status_message_maxiter
    elif gnorm != gnorm or fval != fval or np.isnan(xk).any():
        warnflag = 3
        msg = _status_message_nan
    return OptimizeResult(xk, fval, gfk, nit=k, nfev=sf.nfev, njev=sf.ngev,
                          status=warnflag, success=(warnflag == 0),
                          message=msg)

# ------------------------------------------------------------------- Powell

def _line_for_search(x0: np.ndarray[float, 1], alpha: np.ndarray[float, 1],
                     lower_bound: np.ndarray[float, 1],
                     upper_bound: np.ndarray[float, 1]):
    lmin = -INF
    lmax = INF
    for i in range(x0.size):
        if alpha[i] != 0.0:
            low = (lower_bound[i] - x0[i]) / alpha[i]
            high = (upper_bound[i] - x0[i]) / alpha[i]
            if alpha[i] > 0:
                lmin = max(lmin, low)
                lmax = min(lmax, high)
            else:
                lmin = max(lmin, high)
                lmax = min(lmax, low)
    if lmax >= lmin:
        return lmin, lmax
    return 0.0, 0.0

class _PowellLine[F]:
    func: F
    p: np.ndarray[float, 1]
    xi: np.ndarray[float, 1]
    tan: bool

    def __init__(self, func: F, p: np.ndarray[float, 1], xi: np.ndarray[float, 1],
                 tan: bool):
        self.func = func
        self.p = p
        self.xi = xi
        self.tan = tan

    def __call__(self, alpha: float) -> float:
        if self.tan:
            alpha = math.tan(alpha)
        return self.func(self.p + alpha * self.xi)

def _linesearch_powell(func, p: np.ndarray[float, 1], xi: np.ndarray[float, 1],
                       tol: float, has_b: bool, lower_bound: np.ndarray[float, 1],
                       upper_bound: np.ndarray[float, 1], fval: float):
    myfunc = _PowellLine(func, p, xi, False)
    anyx = False
    for v in xi:
        if v != 0.0:
            anyx = True
    if not anyx:
        return fval, p, xi
    bnd = (-INF, INF)
    if has_b:
        bnd = _line_for_search(p, xi, lower_bound, upper_bound)
    if bnd[0] == -INF and bnd[1] == INF:
        from ._scalar import _recover_brent
        res = _recover_brent(myfunc, None, (), tol, 500, False)
        xi2 = res.x * xi
        return res.fun, p + xi2, xi2
    elif bnd[0] != -INF and bnd[1] != INF:
        res = _minimize_scalar_bounded_impl(myfunc, bnd[0], bnd[1], (),
                                            tol / 100)
        xi2 = res.x * xi
        return res.fun, p + xi2, xi2
    else:
        tfunc = _PowellLine(func, p, xi, True)
        res = _minimize_scalar_bounded_impl(tfunc, math.atan(bnd[0]),
                                            math.atan(bnd[1]), (), tol / 100)
        xi2 = math.tan(res.x) * xi
        return res.fun, p + xi2, xi2

def _minimize_powell(func, x0: np.ndarray[float, 1], args, options,
                     bnds: Optional[Tuple[np.ndarray[float, 1], np.ndarray[float, 1]]],
                     callback = None):
    xtol = _opt_float(options, 'xtol', 1e-4)
    ftol = _opt_float(options, 'ftol', 1e-4)
    x = x0.copy()
    N = len(x)
    has_mi = _has_opt(options, 'maxiter')
    has_mf = _has_opt(options, 'maxfev')
    maxiter = INF
    maxfun = INF
    if not has_mi and not has_mf:
        maxiter = float(N * 1000)
        maxfun = float(N * 1000)
    elif not has_mi:
        maxfun = _opt_float(options, 'maxfev', 0.0)
        maxiter = float(N * 1000) if maxfun == INF else INF
    elif not has_mf:
        maxiter = _opt_float(options, 'maxiter', 0.0)
        maxfun = float(N * 1000) if maxiter == INF else INF
    else:
        maxiter = _opt_float(options, 'maxiter', 0.0)
        maxfun = _opt_float(options, 'maxfev', 0.0)
    fc = _CountedFun(func, args, maxfun)
    f = fc

    direc = np.eye(N)
    has_b = bnds is not None
    lower_bound = np.full(N, -INF)
    upper_bound = np.full(N, INF)
    if bnds is not None:
        lower_bound, upper_bound = bnds
    fval = f(x)
    x1 = x.copy()
    iter = 0
    stopped = False
    while True:
        try:
            fx = fval
            bigind = 0
            delta = 0.0
            for i in range(N):
                direc1 = direc[i].copy()
                fx2 = fval
                fval, x, direc1 = _linesearch_powell(f, x, direc1, xtol * 100,
                                                     has_b, lower_bound,
                                                     upper_bound, fval)
                if (fx2 - fval) > delta:
                    delta = fx2 - fval
                    bigind = i
            iter += 1
            if _cb_halt(callback, x, fval):
                stopped = True
                break
            bnd = ftol * (abs(fx) + abs(fval)) + 1e-20
            if 2.0 * (fx - fval) <= bnd:
                break
            if float(fc.ncalls) >= maxfun:
                break
            if float(iter) >= maxiter:
                break
            if fx != fx and fval != fval:
                break
            direc1 = x - x1
            x1 = x.copy()
            lmax = 1.0
            if has_b:
                _, lmax = _line_for_search(x, direc1, lower_bound, upper_bound)
            x2 = x + min(lmax, 1.0) * direc1
            fx2 = f(x2)
            if fx > fx2:
                t = 2.0 * (fx + fx2 - 2.0 * fval)
                temp = fx - fval - delta
                t *= temp * temp
                temp = fx - fx2
                t -= delta * temp * temp
                if t < 0.0:
                    fval, x, direc1 = _linesearch_powell(f, x, direc1, xtol * 100,
                                                         has_b, lower_bound,
                                                         upper_bound, fval)
                    anyd = False
                    for v in direc1:
                        if v != 0.0:
                            anyd = True
                    if anyd:
                        direc[bigind] = direc[-1]
                        direc[-1] = direc1
        except _MaxFuncCallError:
            break
    warnflag = 0
    msg = _status_message_success
    oob = False
    if has_b:
        for i in range(N):
            if lower_bound[i] > x[i] or x[i] > upper_bound[i]:
                oob = True
    if oob:
        warnflag = 4
        msg = _status_message_oob
    elif float(fc.ncalls) >= maxfun:
        warnflag = 1
        msg = _status_message_maxfev
    elif float(iter) >= maxiter:
        warnflag = 2
        msg = _status_message_maxiter
    elif fval != fval or np.isnan(x).any():
        warnflag = 3
        msg = _status_message_nan
    r = OptimizeResult(x, fval, _e1(), nit=iter, nfev=fc.ncalls,
                       status=warnflag, success=(warnflag == 0),
                       message=msg, hess_inv=direc)
    if stopped:
        r.success = False
        r.status = 99
        r.message = _STOP_MSG
    return r
