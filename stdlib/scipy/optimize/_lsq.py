# Ported from scipy 1.18.1 (BSD-3-Clause)

# Nonlinear least squares (least_squares with 'trf' and 'lm'), curve_fit,
# fsolve and root(method='hybr').

import math
import numpy as np
from ._common import *
from ._common import (_ScalarFunction, _approx_grad, _approx_jac, _bound_val, _call_scalar, _call_vec, _e1, _e2, _eps_for_method, _epsilon, _fd_steps, _has_opt, _norm2, _norm_inf, _old_bound_to_new, _opt_bool, _opt_float, _opt_int, _prepare_bounds_pair, _scalar_eps, _sign, _sqrt, _standardize_bounds, _status_message_maxfev, _status_message_maxiter, _status_message_nan, _status_message_oob, _status_message_pr_loss, _status_message_success, _vec)
from ._minpack import _hybrd, _lmdif, _leastsq_cov

_TERMINATION_MESSAGES = ["Improper input parameters status returned from `leastsq`",
                         "The maximum number of function evaluations is exceeded.",
                         "`gtol` termination condition is satisfied.",
                         "`ftol` termination condition is satisfied.",
                         "`xtol` termination condition is satisfied.",
                         "Both `ftol` and `xtol` termination conditions are satisfied."]

def _term_msg(status: int) -> str:
    if status == -2:
        return "Stopped because `callback` function raised `StopIteration` or returned `True`"
    return _TERMINATION_MESSAGES[status + 1]

# ------------------------------------------------------------- trf helpers

def _in_bounds(x, lb, ub) -> bool:
    for i in range(x.size):
        if not (x[i] >= lb[i] and x[i] <= ub[i]):
            return False
    return True

def _find_active(x, lb, ub, rtol: float) -> np.ndarray[int, 1]:
    n = x.size
    active = np.zeros(n, int)
    for i in range(n):
        if rtol == 0:
            if x[i] <= lb[i]:
                active[i] = -1
            if x[i] >= ub[i]:
                active[i] = 1
            continue
        lower_dist = x[i] - lb[i]
        upper_dist = ub[i] - x[i]
        lt = rtol * max(1.0, abs(lb[i]))
        ut = rtol * max(1.0, abs(ub[i]))
        if math.isfinite(lb[i]) and lower_dist <= min(upper_dist, lt):
            active[i] = -1
        if math.isfinite(ub[i]) and upper_dist <= min(lower_dist, ut):
            active[i] = 1
    return active

def _nextafter(a: float, b: float) -> float:
    return float(np.nextafter(a, b))

def _make_strictly_feasible(x, lb, ub, rstep: float = 1e-10):
    x_new = x.copy()
    active = _find_active(x, lb, ub, rstep)
    for i in range(x.size):
        if active[i] == -1:
            if rstep == 0:
                x_new[i] = _nextafter(lb[i], ub[i])
            else:
                x_new[i] = lb[i] + rstep * max(1.0, abs(lb[i]))
        elif active[i] == 1:
            if rstep == 0:
                x_new[i] = _nextafter(ub[i], lb[i])
            else:
                x_new[i] = ub[i] - rstep * max(1.0, abs(ub[i]))
        if x_new[i] < lb[i] or x_new[i] > ub[i]:
            x_new[i] = 0.5 * (lb[i] + ub[i])
    return x_new

def _cl_scaling(x, g, lb, ub):
    n = x.size
    v = np.ones(n)
    dv = np.zeros(n)
    for i in range(n):
        if g[i] < 0 and math.isfinite(ub[i]):
            v[i] = ub[i] - x[i]
            dv[i] = -1
        if g[i] > 0 and math.isfinite(lb[i]):
            v[i] = x[i] - lb[i]
            dv[i] = 1
    return v, dv

def _step_size_to_bound(x, s, lb, ub):
    n = x.size
    steps = np.full(n, INF)
    for i in range(n):
        if s[i] != 0:
            steps[i] = max((lb[i] - x[i]) / s[i], (ub[i] - x[i]) / s[i])
    min_step = float(np.min(steps))
    hits = np.zeros(n, int)
    for i in range(n):
        if steps[i] == min_step:
            hits[i] = 1 if s[i] > 0 else (-1 if s[i] < 0 else 0)
    return min_step, hits

def _intersect_trust_region(x, s, Delta: float):
    a = float(np.dot(s, s))
    if a == 0:
        raise ValueError('`s` is zero.')
    b = float(np.dot(x, s))
    c = float(np.dot(x, x)) - Delta ** 2
    if c > 0:
        raise ValueError('`x` is not within the trust region.')
    d = math.sqrt(b * b - a * c)
    q = -(b + math.copysign(d, b))
    t1 = q / a
    t2 = c / q
    if t1 < t2:
        return t1, t2
    else:
        return t2, t1

def _evaluate_quadratic(J, g, s, diag) -> float:
    Js = J @ s
    q = float(np.dot(Js, Js))
    if diag is not None:
        q += float(np.dot(s * diag, s))
    l = float(np.dot(s, g))
    return 0.5 * q + l

def _build_quadratic_1d(J, g, s, diag, s0):
    v = J @ s
    a = float(np.dot(v, v))
    if diag is not None:
        a += float(np.dot(s * diag, s))
    a *= 0.5
    b = float(np.dot(g, s))
    c = 0.0
    if s0 is not None:
        u = J @ s0
        b += float(np.dot(u, v))
        c = 0.5 * float(np.dot(u, u)) + float(np.dot(g, s0))
        if diag is not None:
            b += float(np.dot(s0 * diag, s))
            c += 0.5 * float(np.dot(s0 * diag, s0))
    return a, b, c

def _minimize_quadratic_1d(a: float, b: float, lb: float, ub: float,
                           c: float = 0.0):
    ts = [lb, ub]
    if a != 0:
        extremum = -0.5 * b / a
        if lb < extremum < ub:
            ts.append(extremum)
    best = 0
    bestv = INF
    for i in range(len(ts)):
        t = ts[i]
        y = t * (a * t + b) + c
        if y < bestv or (i == 0):
            if i == 0 or y < bestv:
                best = i
                bestv = y
    return ts[best], bestv

def _solve_lsq_trust_region(n: int, m: int, uf, s, V, Delta: float,
                            initial_alpha: float, rtol: float = 0.01,
                            max_iter: int = 10):
    suf = s * uf
    full_rank = False
    if m >= n:
        threshold = EPS * m * s[0]
        full_rank = s[s.size - 1] > threshold
    if full_rank:
        p = -(V @ (uf / s))
        if _norm2(p) <= Delta:
            return p, 0.0, 0

    def phi_and_derivative(alpha: float):
        denom = s ** 2 + alpha
        p_norm = _norm2(suf / denom)
        phi = p_norm - Delta
        phi_prime = -float(np.sum(suf ** 2 / denom ** 3)) / p_norm
        return phi, phi_prime

    alpha_upper = _norm2(suf) / Delta
    alpha_lower = 0.0
    if full_rank:
        phi, phi_prime = phi_and_derivative(0.0)
        alpha_lower = -phi / phi_prime
    alpha = initial_alpha
    if not full_rank and initial_alpha == 0:
        alpha = max(0.001 * alpha_upper, (alpha_lower * alpha_upper) ** 0.5)
    it = 0
    for it in range(max_iter):
        if alpha < alpha_lower or alpha > alpha_upper:
            alpha = max(0.001 * alpha_upper, (alpha_lower * alpha_upper) ** 0.5)
        phi, phi_prime = phi_and_derivative(alpha)
        if phi < 0:
            alpha_upper = alpha
        ratio = phi / phi_prime
        alpha_lower = max(alpha_lower, alpha - ratio)
        alpha -= (phi + Delta) * ratio / Delta
        if abs(phi) < rtol * Delta:
            break
    p = -(V @ (suf / (s ** 2 + alpha)))
    p *= Delta / _norm2(p)
    return p, alpha, it + 1

def _update_tr_radius(Delta: float, actual_reduction: float,
                      predicted_reduction: float, step_norm: float,
                      bound_hit: bool):
    ratio = 0.0
    if predicted_reduction > 0:
        ratio = actual_reduction / predicted_reduction
    elif predicted_reduction == actual_reduction == 0:
        ratio = 1.0
    else:
        ratio = 0.0
    if ratio < 0.25:
        Delta = 0.25 * step_norm
    elif ratio > 0.75 and bound_hit:
        Delta *= 2.0
    return Delta, ratio

def _check_termination(dF: float, F: float, dx_norm: float, x_norm: float,
                       ratio: float, ftol: float, xtol: float) -> int:
    ftol_satisfied = dF < ftol * F and ratio > 0.25
    xtol_satisfied = dx_norm < xtol * (xtol + x_norm)
    if ftol_satisfied and xtol_satisfied:
        return 4
    elif ftol_satisfied:
        return 2
    elif xtol_satisfied:
        return 3
    return -100

# ------------------------------------------------------------------- trf

def _select_step(x, J_h, diag_h, g_h, p, p_h, d, Delta: float, lb, ub,
                 theta: float):
    if _in_bounds(x + p, lb, ub):
        p_value = _evaluate_quadratic(J_h, g_h, p_h, diag_h)
        return p, p_h, -p_value
    p_stride, hits = _step_size_to_bound(x, p, lb, ub)
    r_h = p_h.copy()
    for i in range(r_h.size):
        if hits[i] != 0:
            r_h[i] *= -1
    r = d * r_h
    p = p * p_stride
    p_h = p_h * p_stride
    x_on_bound = x + p
    _, to_tr = _intersect_trust_region(p_h, r_h, Delta)
    to_bound, _ = _step_size_to_bound(x_on_bound, r, lb, ub)
    r_stride = min(to_bound, to_tr)
    r_stride_l = 0.0
    r_stride_u = -1.0
    if r_stride > 0:
        r_stride_l = (1 - theta) * p_stride / r_stride
        if r_stride == to_bound:
            r_stride_u = theta * to_bound
        else:
            r_stride_u = to_tr
    r_value = INF
    if r_stride_l <= r_stride_u:
        a, b, c = _build_quadratic_1d(J_h, g_h, r_h, diag_h, p_h)
        rs, rv = _minimize_quadratic_1d(a, b, r_stride_l, r_stride_u, c)
        r_value = rv
        r_h = r_h * rs + p_h
        r = r_h * d
    p = p * theta
    p_h = p_h * theta
    p_value = _evaluate_quadratic(J_h, g_h, p_h, diag_h)
    ag_h = -g_h
    ag = d * ag_h
    to_tr2 = Delta / _norm2(ag_h)
    to_bound2, _ = _step_size_to_bound(x, ag, lb, ub)
    ag_stride = to_tr2
    if to_bound2 < to_tr2:
        ag_stride = theta * to_bound2
    a2, b2, c2 = _build_quadratic_1d(J_h, g_h, ag_h, diag_h, None)
    ag_stride, ag_value = _minimize_quadratic_1d(a2, b2, 0.0, ag_stride)
    ag_h = ag_h * ag_stride
    ag = ag * ag_stride
    if p_value < r_value and p_value < ag_value:
        return p, p_h, -p_value
    elif r_value < p_value and r_value < ag_value:
        return r, r_h, -r_value
    else:
        return ag, ag_h, -ag_value

def _trf(fun, jac, x0, f0, J0, lb, ub, ftol: float, xtol: float, gtol: float,
         max_nfev: int, x_scale, jac_scale: bool, bounded: bool):
    x = x0.copy()
    f = f0
    f_true = f.copy()
    nfev = 1
    J = J0
    njev = 1
    m = J.shape[0]
    n = J.shape[1]
    cost = 0.5 * float(np.dot(f, f))
    g = J.T @ f
    scale = x_scale
    scale_inv = 1 / x_scale
    if jac_scale:
        scale_inv = np.sqrt(np.sum(J ** 2, axis=0))
        for i in range(n):
            if scale_inv[i] == 0:
                scale_inv[i] = 1.0
        scale = 1 / scale_inv
    Delta = 0.0
    if bounded:
        v, dv = _cl_scaling(x, g, lb, ub)
        for i in range(n):
            if dv[i] != 0:
                v[i] *= scale_inv[i]
        Delta = _norm2(x0 * scale_inv / v ** 0.5)
    else:
        Delta = _norm2(x0 * scale_inv)
    if Delta == 0:
        Delta = 1.0
    g_norm = 0.0
    alpha = 0.0
    termination_status = -100
    iteration = 0
    x_new = x
    f_new = f
    cost_new = cost
    while True:
        v = np.ones(n)
        dv = np.zeros(n)
        if bounded:
            v, dv = _cl_scaling(x, g, lb, ub)
            g_norm = _norm_inf(g * v)
        else:
            g_norm = _norm_inf(g)
        if g_norm < gtol:
            termination_status = 1
        if termination_status != -100 or nfev == max_nfev:
            break
        d = scale
        diag_h = np.zeros(n)
        if bounded:
            for i in range(n):
                if dv[i] != 0:
                    v[i] *= scale_inv[i]
            d = v ** 0.5 * scale
            diag_h = g * dv * scale
        g_h = d * g
        J_h = J * d
        U = J_h
        s = g
        V = J_h
        uf = g
        if bounded:
            J_aug = np.zeros((m + n, n))
            J_aug[:m] = J_h
            for i in range(n):
                J_aug[m + i, i] = diag_h[i] ** 0.5
            f_aug = np.zeros(m + n)
            f_aug[:m] = f
            U, s, Vt = np.linalg.svd(J_aug, full_matrices=False)
            V = Vt.T
            uf = U.T @ f_aug
        else:
            U, s, Vt = np.linalg.svd(J_h, full_matrices=False)
            V = Vt.T
            uf = U.T @ f
        theta = max(0.995, 1 - g_norm)
        actual_reduction = -1.0
        step_norm = 0.0
        while actual_reduction <= 0 and nfev < max_nfev:
            p_h, alpha, n_iter = _solve_lsq_trust_region(n, m, uf, s, V, Delta, alpha)
            step = p_h
            step_h = p_h
            predicted_reduction = 0.0
            if bounded:
                p = d * p_h
                step, step_h, predicted_reduction = _select_step(
                    x, J_h, diag_h, g_h, p, p_h, d, Delta, lb, ub, theta)
                x_new = _make_strictly_feasible(x + step, lb, ub, 0.0)
            else:
                step_h = p_h
                predicted_reduction = -_evaluate_quadratic(J_h, g_h, step_h, None)
                step = d * step_h
                x_new = x + step
            f_new = fun(x_new)
            nfev += 1
            step_h_norm = _norm2(step_h)
            finite = True
            for vv in f_new:
                if not math.isfinite(vv):
                    finite = False
            if not finite:
                Delta = 0.25 * step_h_norm
                continue
            cost_new = 0.5 * float(np.dot(f_new, f_new))
            actual_reduction = cost - cost_new
            Delta_new, ratio = _update_tr_radius(Delta, actual_reduction,
                                                 predicted_reduction, step_h_norm,
                                                 step_h_norm > 0.95 * Delta)
            step_norm = _norm2(step)
            termination_status = _check_termination(actual_reduction, cost,
                                                     step_norm, _norm2(x), ratio,
                                                     ftol, xtol)
            if termination_status != -100:
                break
            alpha *= Delta / Delta_new
            Delta = Delta_new
        if actual_reduction > 0:
            x = x_new
            f = f_new
            f_true = f.copy()
            cost = cost_new
            J = jac(x, f)
            njev += 1
            g = J.T @ f
            if jac_scale:
                scale_inv = np.maximum(np.sqrt(np.sum(J ** 2, axis=0)), scale_inv)
                scale = 1 / scale_inv
        iteration += 1
    if termination_status == -100:
        termination_status = 0
    active_mask = np.zeros(n, int)
    if bounded:
        active_mask = _find_active(x, lb, ub, xtol)
    return x, cost, f_true, J, g, g_norm, active_mask, nfev, njev, termination_status

# ----------------------------------------------------------- least_squares

class _VecFun[F, A]:
    fun: F
    args: A
    nfev: int

    def __init__(self, fun: F, args: A):
        self.fun = fun
        self.args = args
        self.nfev = 0

    def __call__(self, x) -> np.ndarray[float, 1]:
        self.nfev += 1
        return _call_vec(self.fun, x.copy(), self.args)

def _from_minpack(info: int) -> int:
    if info == 0:
        return -1
    elif info == 1:
        return 2
    elif info == 2:
        return 3
    elif info == 3:
        return 4
    elif info == 4:
        return 1
    elif info == 5:
        return 0
    return -1

class _JacFun[F, J, A]:
    # Jacobian of a residual function: user callable or finite differences
    fun: F
    jac: J
    args: A
    diff_step: Optional[float]
    lb: np.ndarray[float, 1]
    ub: np.ndarray[float, 1]

    def __init__(self, fun: F, jac: J, args: A, diff_step: Optional[float],
                 lb: np.ndarray[float, 1], ub: np.ndarray[float, 1]):
        self.fun = fun
        self.jac = jac
        self.args = args
        self.diff_step = diff_step
        self.lb = lb
        self.ub = ub

    def _f(self, y: np.ndarray[float, 1]) -> np.ndarray[float, 1]:
        return _call_vec(self.fun, y, self.args)

    def __call__(self, x: np.ndarray[float, 1], fx: np.ndarray[float, 1]) -> np.ndarray[float, 2]:
        if isinstance(self.jac, str):
            J, nf = _approx_jac(self._f, x, fx, self.jac, self.diff_step, None,
                                self.lb, self.ub)
            return J
        else:
            return np.atleast_2d(np.asarray(self.jac(x.copy(), *self.args), float))

class _JacAt[JF]:
    jf: JF

    def __init__(self, jf: JF):
        self.jf = jf

    def __call__(self, x: np.ndarray[float, 1]) -> np.ndarray[float, 2]:
        return self.jf(x, self.jf._f(x.copy()))

def least_squares(fun, x0, jac = '2-point', bounds = (-float('inf'), float('inf')),
                  method: str = 'trf', ftol: Optional[float] = 1e-08,
                  xtol: Optional[float] = 1e-08, gtol: Optional[float] = 1e-08,
                  x_scale = None, loss: str = 'linear', f_scale: float = 1.0,
                  diff_step: Optional[float] = None,
                  max_nfev: Optional[int] = None, verbose: int = 0, args = ()):
    if not isinstance(args, Tuple):
        return least_squares(fun, x0, jac, bounds, method, ftol, xtol, gtol,
                             x_scale, loss, f_scale, diff_step, max_nfev,
                             verbose, (args,))
    if method != 'trf' and method != 'dogbox' and method != 'lm':
        raise ValueError("`method` must be 'trf', 'dogbox' or 'lm'.")
    if method == 'dogbox':
        raise ValueError("method='dogbox' is not supported by this port; use 'trf' or 'lm'.")
    if isinstance(jac, str):
        if jac != '2-point' and jac != '3-point' and jac != 'cs':
            raise ValueError("`jac` must be '2-point', '3-point', 'cs' or callable.")
        if jac == 'cs':
            raise ValueError("jac='cs' is not supported by this port.")
    if loss != 'linear':
        if method == 'lm':
            raise ValueError("method='lm' supports only 'linear' loss function.")
        raise ValueError("only loss='linear' is supported by this port.")
    if verbose != 0 and verbose != 1 and verbose != 2:
        raise ValueError('`verbose` must be in [0, 1, 2].')
    if max_nfev is not None and max_nfev <= 0:
        raise ValueError('`max_nfev` must be None or positive integer.')
    x0v = _vec(x0)
    n = x0v.size
    lb, ub = _prepare_bounds_pair(bounds, n)
    unb = True
    for i in range(n):
        if lb[i] != -INF or ub[i] != INF:
            unb = False
    if method == 'lm' and not unb:
        raise ValueError("Method 'lm' doesn't support bounds.")
    if lb.size != n or ub.size != n:
        raise ValueError('Inconsistent shapes between bounds and `x0`.')
    for i in range(n):
        if lb[i] >= ub[i]:
            raise ValueError('Each lower bound must be strictly less than each upper bound.')
    if not _in_bounds(x0v, lb, ub):
        raise ValueError('Initial guess is outside of provided bounds')
    # x_scale
    jac_scale = False
    xs = np.ones(n)
    if x_scale is None:
        jac_scale = method == 'lm'
    elif isinstance(x_scale, str):
        if x_scale != 'jac':
            raise ValueError("`x_scale` must be 'jac' or array_like with positive numbers.")
        jac_scale = True
    else:
        xs1 = np.atleast_1d(np.asarray(x_scale, float)).flatten()
        for v in xs1:
            if not (math.isfinite(v) and v > 0):
                raise ValueError("`x_scale` must be 'jac' or array_like with positive numbers.")
        if xs1.size == 1:
            xs = np.full(n, xs1[0])
        elif xs1.size == n:
            xs = xs1
        else:
            raise ValueError('Inconsistent shapes between `x_scale` and `x0`.')
    ft = 0.0 if ftol is None else ftol
    xt = 0.0 if xtol is None else xtol
    gt = 0.0 if gtol is None else gtol
    if method == 'lm' and (ft < EPS or xt < EPS or gt < EPS):
        raise ValueError(f"All tolerances must be higher than machine epsilon ({EPS:.2e}) for method 'lm'.")
    elif ft < EPS and xt < EPS and gt < EPS:
        raise ValueError(f"At least one of the tolerances must be higher than machine epsilon ({EPS:.2e}).")
    if method == 'trf':
        x0v = _make_strictly_feasible(x0v, lb, ub)
    vf = _VecFun(fun, args)
    jacf = _JacFun(fun, jac, args, diff_step, lb, ub)
    f0 = vf(x0v)
    m = f0.size
    for v in f0:
        if not math.isfinite(v):
            raise ValueError('Residuals are not finite in the initial point.')
    if method == 'lm' and m < n:
        raise ValueError("Method 'lm' doesn't work when the number of residuals is less than the number of variables.")
    J0 = jacf(x0v, f0)
    if J0.shape[0] != m or J0.shape[1] != n:
        raise ValueError(f'The return value of `jac` has wrong shape: expected {(m, n)}, actual {J0.shape}.')
    if method == 'lm':
        mfev = 100 * n if max_nfev is None else max_nfev
        diag = np.ones(n)
        mode = 1
        if not jac_scale:
            diag = 1 / xs
            mode = 2

        lmj = _JacAt(jacf)
        xx = x0v.copy()
        fvec, fjac, ipvt, qtf, info, nfev, njev = _lmdif(
            vf, lmj, xx, m, ft, xt, gt, mfev, 0.0, diag, mode, 100.0, True)
        J = lmj(xx)
        cost = 0.5 * float(np.dot(fvec, fvec))
        g = J.T @ fvec
        status = _from_minpack(info)
        return OptimizeResult(xx, fvec, J, nfev=nfev,
                              njev=(njev if not isinstance(jac, str) else 0),
                              status=status, success=status > 0,
                              message=_term_msg(status), cost=cost, grad=g,
                              optimality=_norm_inf(g),
                              active_mask=np.zeros(n, int))
    mfev = 100 * n if max_nfev is None else max_nfev
    x, cost, f, J, g, gn, am, nfev, njev, status = _trf(
        vf, jacf, x0v, f0, J0, lb, ub, ft, xt, gt, mfev, xs, jac_scale, not unb)
    return OptimizeResult(x, f, J, nfev=nfev, njev=njev, status=status,
                          success=status > 0, message=_term_msg(status),
                          cost=cost, grad=g, optimality=gn, active_mask=am)

# ------------------------------------------------------------- curve_fit

def _nparams(f) -> Literal[int]:
    return static.len(static.function.args(f)) - 1

def _splat(p, N: Literal[int]):
    if N == 0:
        return ()
    else:
        return _splat(p, N - 1) + (p[N - 1],)

class _CFModel[F, X]:
    f: F
    xd: X
    yd: np.ndarray[float, 1]
    tr: np.ndarray[float, 1]

    def __init__(self, f: F, xd: X, yd: np.ndarray[float, 1],
                 tr: np.ndarray[float, 1]):
        self.f = f
        self.xd = xd
        self.yd = yd
        self.tr = tr

    def __call__(self, p: np.ndarray[float, 1]) -> np.ndarray[float, 1]:
        N: Literal[int] = _nparams(self.f)
        model = np.asarray(self.f(self.xd, *_splat(p, N)), float).flatten()
        return self.tr * (model - self.yd)

class _CFJac[F, J, X]:
    f: F
    jac: J
    xd: X
    tr: np.ndarray[float, 1]

    def __init__(self, f: F, jac: J, xd: X, tr: np.ndarray[float, 1]):
        self.f = f
        self.jac = jac
        self.xd = xd
        self.tr = tr

    def __call__(self, p: np.ndarray[float, 1]) -> np.ndarray[float, 2]:
        N: Literal[int] = _nparams(self.f)
        if self.jac is None:
            return np.zeros((0, 0))
        else:
            J = np.atleast_2d(np.asarray(self.jac(self.xd, *_splat(p, N)), float))
            return J * self.tr[:, None]

def _cf_trf(func, jac, jobj, p0v, lb, ub, meth: str, ft: float, xt: float,
            gt: float, mfev: Optional[int]):
    if jac is None:
        return least_squares(func, p0v, '2-point', (lb, ub), meth, ft, xt, gt,
                             None, 'linear', 1.0, None, mfev)
    else:
        return least_squares(func, p0v, jobj, (lb, ub), meth, ft, xt, gt,
                             None, 'linear', 1.0, None, mfev)

def curve_fit(f, xdata, ydata, p0 = None, sigma = None,
              absolute_sigma: bool = False, check_finite: bool = True,
              bounds = (-float('inf'), float('inf')), method: Optional[str] = None,
              jac = None, full_output: Literal[bool] = False,
              maxfev: Optional[int] = None, ftol: Optional[float] = None,
              xtol: Optional[float] = None, gtol: Optional[float] = None):
    N: Literal[int] = _nparams(f)
    n = N
    p0v = np.ones(n)
    if p0 is not None:
        p0v = _vec(p0)
        n = p0v.size
        if n != N:
            raise ValueError(f'p0 has {n} parameters, but the model function takes {N}.')
    lb, ub = _prepare_bounds_pair(bounds, n)
    if p0 is None:
        # _initialize_feasible
        for i in range(n):
            lf = math.isfinite(lb[i])
            uf = math.isfinite(ub[i])
            if lf and uf:
                p0v[i] = 0.5 * (lb[i] + ub[i])
            elif lf:
                p0v[i] = lb[i] + 1
            elif uf:
                p0v[i] = ub[i] - 1
    bounded = False
    for i in range(n):
        if lb[i] > -INF or ub[i] < INF:
            bounded = True
    meth = ''
    if method is None:
        meth = 'trf' if bounded else 'lm'
    else:
        meth = method
    if meth == 'lm' and bounded:
        raise ValueError("Method 'lm' only works for unconstrained problems. Use 'trf' or 'dogbox' instead.")
    yd = np.asarray(ydata, float).flatten()
    xd = np.asarray(xdata, float)
    if check_finite:
        for v in yd:
            if not math.isfinite(v):
                raise ValueError("array must not contain infs or NaNs")
        for v in xd.flat:
            if not math.isfinite(v):
                raise ValueError("array must not contain infs or NaNs")
    if yd.size == 0:
        raise ValueError('`ydata` must not be empty!')
    tr = np.ones(yd.size)
    if sigma is not None:
        sg = np.atleast_1d(np.asarray(sigma, float)).flatten()
        if sg.size == 1:
            tr = np.full(yd.size, 1.0 / sg[0])
        elif sg.size == yd.size:
            tr = 1.0 / sg
        else:
            raise ValueError('`sigma` has incorrect shape.')

    func = _CFModel(f, xd, yd, tr)

    popt = p0v
    pcov = np.zeros((n, n))
    ysize = yd.size
    cost = 0.0
    nfev = 0
    fvec = np.zeros(0)
    ier = 0
    errmsg = ''
    if meth == 'lm':
        if ysize != 1 and n > ysize:
            raise TypeError(f'The number of func parameters={n} must not exceed the number of data points={ysize}')
        mfev = 200 * (n + 1) if maxfev is None else maxfev
        ft = 1.49012e-08 if ftol is None else ftol
        xt = 1.49012e-08 if xtol is None else xtol
        gt = 0.0 if gtol is None else gtol
        xx = p0v.copy()
        fv0 = func(xx)
        m = fv0.size
        if n > m:
            raise TypeError(f'Improper input: func input vector length N={n} must not exceed func output vector length M={m}')

        jf = _CFJac(f, jac, xd, tr)
        fvec, fjac, ipvt, qtf, info, nfev, njev = _lmdif(
            func, jf, xx, m, ft, xt, gt, mfev, EPS, np.ones(n), 1, 100.0,
            jac is not None)
        ier = info
        popt = xx
        cost = float(np.sum(fvec ** 2))
        if ier == 5:
            errmsg = f'Number of calls to function has reached maxfev = {mfev}.'
        if ier not in (1, 2, 3, 4):
            raise RuntimeError('Optimal parameters not found: ' + errmsg)
        ok, cov = _leastsq_cov(fjac, ipvt, m, n)
        if ok:
            pcov = cov
        else:
            pcov = np.full((n, n), NAN)
    else:
        mfev2: Optional[int] = maxfev
        ft2 = 1e-08 if ftol is None else ftol
        xt2 = 1e-08 if xtol is None else xtol
        gt2 = 1e-08 if gtol is None else gtol
        res = _cf_trf(func, jac, _CFJac(f, jac, xd, tr), p0v, lb, ub, meth,
                      ft2, xt2, gt2, mfev2)
        if not res.success:
            raise RuntimeError('Optimal parameters not found: ' + res.message)
        fvec = res.fun
        nfev = res.nfev
        ier = res.status
        errmsg = res.message
        ysize = res.fun.size
        cost = 2 * res.cost
        popt = res.x
        U, s, VT = np.linalg.svd(res.jac, full_matrices=False)
        threshold = EPS * max(res.jac.shape[0], res.jac.shape[1]) * s[0]
        k = 0
        for v in s:
            if v > threshold:
                k += 1
        sk = s[:k]
        VTk = VT[:k]
        pcov = np.dot(VTk.T / sk ** 2, VTk)
    nanp = False
    for v in pcov.flat:
        if v != v:
            nanp = True
    if nanp:
        pcov = np.full((n, n), INF)
    elif not absolute_sigma:
        if ysize > n:
            s_sq = cost / (ysize - n)
            pcov = pcov * s_sq
        else:
            pcov = np.full((n, n), INF)
    if full_output:
        return popt, pcov, fvec, errmsg, ier
    else:
        return popt, pcov

# ------------------------------------------------------- fsolve / root

_HYBR_ERRORS = ['Improper input parameters were entered.',
                'The solution converged.',
                '', '',
                'The iteration is not making good progress, as measured by the \n improvement from the last five Jacobian evaluations.',
                'The iteration is not making good progress, as measured by the \n improvement from the last ten iterations.']

class _HFun[F, A]:
    func: F
    args: A
    n: int
    nf: int

    def __init__(self, func: F, args: A, n: int):
        self.func = func
        self.args = args
        self.n = n
        self.nf = 0

    def __call__(self, y: np.ndarray[float, 1]) -> np.ndarray[float, 1]:
        self.nf += 1
        r = _call_vec(self.func, y.copy(), self.args)
        if r.size != self.n:
            raise TypeError(f"fsolve: there is a mismatch between the input and output shape of the 'func' argument.Shape should be ({self.n},) but it is ({r.size},).")
        return r

class _HJac[J, A]:
    jac: J
    args: A

    def __init__(self, jac: J, args: A):
        self.jac = jac
        self.args = args

    def __call__(self, y: np.ndarray[float, 1]) -> np.ndarray[float, 2]:
        if self.jac is None:
            return np.zeros((0, 0))
        else:
            return np.atleast_2d(np.asarray(self.jac(y.copy(), *self.args), float))

def _root_hybr(func, x0, args, jac, xtol: float = 1.49012e-08, maxfev: int = 0,
               band: Optional[Tuple[int, int]] = None,
               eps: Optional[float] = None, factor: float = 100,
               diag = None):
    x = _vec(x0)
    n = x.size
    epsfcn = EPS if eps is None else eps
    fcn = _HFun(func, args, n)
    jf = _HJac(jac, args)

    dg = np.ones(n)
    mode = 1
    if diag is not None:
        dg = _vec(diag)
        mode = 2
    mf = maxfev
    ml = 0
    mu = 0
    if jac is None:
        if band is None:
            ml = n - 1
            mu = n - 1
        else:
            ml, mu = band
            if ml < 0:
                ml = n - 1
            if mu < 0:
                mu = n - 1
        if mf == 0:
            mf = 200 * (n + 1)
    else:
        if mf == 0:
            mf = 100 * (n + 1)
    fvec, fjac, r, qtf, info, nfev, njev = _hybrd(
        fcn, jf, x, xtol, mf, ml, mu, epsfcn, dg, mode, factor, jac is not None)
    msg = ''
    if info == 2:
        msg = f'The number of calls to function has reached maxfev = {mf}.'
    elif info == 3:
        msg = f'xtol={xtol:f} is too small, no further improvement in the approximate\n solution is possible.'
    elif info >= 0 and info <= 5:
        msg = _HYBR_ERRORS[info]
    else:
        msg = 'An error occurred.'
    FJ = np.zeros((n, n))
    for j in range(n):
        for i in range(n):
            FJ[i, j] = fjac[i + n * j]
    # scipy counts two extra calls: _check_func's shape probe and the C
    # wrapper's initial evaluation
    return OptimizeResult(x, fvec, _e2(), nfev=fcn.nf + 2, njev=njev,
                          status=info, success=(info == 1), message=msg,
                          fjac=FJ, qtf=qtf, method='hybr')

def fsolve(func, x0, args = (), fprime = None, full_output: Literal[bool] = False,
           col_deriv: bool = False, xtol: float = 1.49012e-08, maxfev: int = 0,
           band: Optional[Tuple[int, int]] = None,
           epsfcn: Optional[float] = None, factor: float = 100, diag = None):
    if not isinstance(args, Tuple):
        return fsolve(func, x0, (args,), fprime, full_output, col_deriv, xtol,
                      maxfev, band, epsfcn, factor, diag)
    res = _root_hybr(func, x0, args, fprime, xtol, maxfev, band, epsfcn,
                     factor, diag)
    if full_output:
        return res.x, res, res.status, res.message
    else:
        if res.status == 0:
            raise TypeError(res.message)
        elif res.status != 1 and not (2 <= res.status <= 5):
            raise TypeError(res.message)
        return res.x

def root(fun, x0, args = (), method: str = 'hybr', jac = None,
         tol: Optional[float] = None, options = None):
    if not isinstance(args, Tuple):
        return root(fun, x0, (args,), method, jac, tol, options)
    meth = method.lower()
    if meth == 'hybr':
        xtol = _opt_float(options, 'xtol', 1.49012e-08 if tol is None else tol)
        maxfev = _opt_int(options, 'maxfev', 0)
        factor = _opt_float(options, 'factor', 100.0)
        eps: Optional[float] = None
        if _has_opt(options, 'eps'):
            eps = _opt_float(options, 'eps', 0.0)
        return _root_hybr(fun, x0, args, jac, xtol, maxfev, None, eps, factor)
    elif meth == 'lm':
        # scipy's _root_leastsq: MINPACK lmdif with leastsq defaults
        x = _vec(x0)
        n = x.size
        xtol = _opt_float(options, 'xtol', 1.49012e-08 if tol is None else tol)
        ftol = _opt_float(options, 'ftol', 1.49012e-08)
        gtol = _opt_float(options, 'gtol', 0.0)
        maxiter = _opt_int(options, 'maxiter', 0)
        mf = 200 * (n + 1) if maxiter == 0 else maxiter

        fl = _HFun(fun, args, -1)
        jl = _HJac(None, args)
        m = _call_vec(fun, x.copy(), args).size
        fl.n = m
        fvec, fjac, ipvt, qtf, info, nfev, njev = _lmdif(
            fl, jl, x, m, ftol, xtol, gtol, mf, EPS, np.ones(n), 1, 100.0,
            False)
        # +2: leastsq's _check_func probe and the C wrapper's first call
        return OptimizeResult(x, fvec, _e2(), nfev=nfev + 2, status=info,
                              success=(1 <= info <= 4), message='',
                              qtf=qtf, method='lm')
    else:
        raise ValueError(f'Unknown solver {method}')
    return _root_hybr(fun, x0, args, jac)
