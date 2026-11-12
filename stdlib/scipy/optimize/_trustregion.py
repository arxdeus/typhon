# Ported from scipy 1.18.1 (BSD-3-Clause)

# Trust-region minimizers for unconstrained problems (_trustregion.py,
# _trustregion_ncg.py, _trustregion_dogleg.py, _trustregion_exact.py,
# _trustregion_krylov.py) and Newton-CG (_optimize._minimize_newtoncg).
# The quadratic subproblems are one model class with a `kind` switch.

import math
import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from numpy.linalg import LinAlgError
from scipy.linalg._basic import _potrf
from scipy.linalg._util import _to_f_as, _from_f
from ._common import OptimizeResult, _opt_float, _opt_int, _opt_bool, \
                     _has_opt, _epsilon, _vec, _norm2, \
                     _status_message_success, _status_message_maxiter, \
                     _status_message_pr_loss, _status_message_nan
from ._differentiable import _ScalarFunctionH
from ._hessian_update_strategy import HessianUpdateStrategy
from ._callback import _cb_halt, _STOP_MSG
from ._linesearch import _line_search_wolfe12
from ._trlib import _TrlibState, _trlib_solve

_TR_NCG = 0
_TR_DOGLEG = 1
_TR_EXACT = 2
_TR_KRYLOV = 3

EPS = 2.220446049250313e-16

class _HessP[SF, HP, A]:
    # hessp(x, p) of _minimize_trust_region: the user hessp (counted) or
    # sf.hess(x).dot(p) for finite-difference / quasi-Newton Hessians
    sf: SF
    hessp: HP
    args: A
    ncalls: int
    use_user: bool

    def __init__(self, sf: SF, hessp: HP, args: A, use_user: bool):
        self.sf = sf
        self.hessp = hessp
        self.args = args
        self.ncalls = 0
        self.use_user = use_user

    def __call__(self, x: np.ndarray[float, 1], p: np.ndarray[float, 1]) -> np.ndarray[float, 1]:
        self.ncalls += 1
        if self.hessp is None:
            return self.sf.hess_dot(x, p)
        else:
            if self.use_user:
                return _vec(self.hessp(x, p, *self.args))
            return self.sf.hess_dot(x, p)

def _cholesky_info(A: np.ndarray[float, 2]):
    # LAPACK potrf (upper, clean=True): returns (U, info)
    n = A.shape[0]
    P = _to_f_as(A, float)
    info = _potrf(P, n, False)
    U = _from_f(P, n, n, n)
    for i in range(n):
        for j in range(i):
            U[i, j] = 0.0
    return U, info

def _boundaries_intersections(z: np.ndarray[float, 1], d: np.ndarray[float, 1],
                              trust_radius: float):
    a = float(np.dot(d, d))
    b = 2 * float(np.dot(z, d))
    c = float(np.dot(z, z)) - trust_radius ** 2
    sqrt_discriminant = math.sqrt(b * b - 4 * a * c)
    aux = b + math.copysign(sqrt_discriminant, b)
    ta = -aux / (2 * a)
    tb = -2 * c / aux
    if ta <= tb:
        return ta, tb
    return tb, ta

def estimate_smallest_singular_value(U: np.ndarray[float, 2]):
    m, n = U.shape
    if m != n:
        raise ValueError('A square triangular matrix should be provided.')
    p = np.zeros(n)
    w = np.empty(n)
    for k in range(n):
        wp = (1 - p[k]) / U[k, k]
        wm = (-1 - p[k]) / U[k, k]
        pp = p[k + 1:] + U[k, k + 1:] * wp
        pm = p[k + 1:] + U[k, k + 1:] * wm
        if abs(wp) + float(np.sum(np.abs(pp))) >= abs(wm) + float(np.sum(np.abs(pm))):
            w[k] = wp
            p[k + 1:] = pp
        else:
            w[k] = wm
            p[k + 1:] = pm
    v = solve_triangular(U, w)
    v_norm = _norm2(v)
    w_norm = _norm2(w)
    s_min = w_norm / v_norm
    z_min = v / v_norm
    return s_min, z_min

def gershgorin_bounds(H: np.ndarray[float, 2]):
    H_diag = np.diag(H)
    H_diag_abs = np.abs(H_diag)
    H_row_sums = np.sum(np.abs(H), axis=1)
    lb = float(np.min(H_diag + H_diag_abs - H_row_sums))
    ub = float(np.max(H_diag - H_diag_abs + H_row_sums))
    return lb, ub

def singular_leading_submatrix(A: np.ndarray[float, 2], U: np.ndarray[float, 2], k: int):
    delta = float(np.sum(U[:k - 1, k - 1] ** 2)) - A[k - 1, k - 1]
    n = A.shape[0]
    v = np.zeros(n)
    v[k - 1] = 1.0
    if k != 1:
        v[:k - 1] = solve_triangular(U[:k - 1, :k - 1], -U[:k - 1, k - 1])
    return delta, v

def _norm_inf_mat(H: np.ndarray[float, 2]) -> float:
    return float(np.max(np.sum(np.abs(H), axis=1)))

def _norm_fro(H: np.ndarray[float, 2]) -> float:
    return math.sqrt(float(np.sum(H * H)))

class _QuadModel[SF, HP]:
    # BaseQuadraticSubproblem + the four subproblem solvers. f, g and H are
    # evaluated lazily, exactly as the properties of the scipy class.
    kind: int
    sf: SF
    hp: HP
    x: np.ndarray[float, 1]
    has_f: bool
    f_: float
    has_g: bool
    g_: np.ndarray[float, 1]
    has_h: bool
    h_: np.ndarray[float, 2]
    has_gmag: bool
    gmag_: float
    has_hess: bool          # dense Hessian available (callable hess)
    # dogleg
    has_cauchy: bool
    cauchy_: np.ndarray[float, 1]
    has_newton: bool
    newton_: np.ndarray[float, 1]
    # exact
    previous_tr_radius: float
    lambda_lb_: float
    niter: int
    k_easy: float
    k_hard: float
    maxiter: int
    exact_ready: bool
    hess_gershgorin_lb: float
    hess_gershgorin_ub: float
    hess_inf: float
    hess_fro: float
    CLOSE_TO_ZERO: float
    lambda_current: float
    # krylov
    tol_rel_i: float
    tol_rel_b: float
    kstate: _TrlibState

    def __init__(self, kind: int, x: np.ndarray[float, 1], sf: SF, hp: HP,
                 has_hess: bool, maxiter: int, tol_rel_i: float = -2.0,
                 tol_rel_b: float = -3.0):
        self.kind = kind
        self.sf = sf
        self.hp = hp
        self.x = x
        self.has_f = False
        self.f_ = 0.0
        self.has_g = False
        self.g_ = np.zeros(0)
        self.has_h = False
        self.h_ = np.zeros((0, 0))
        self.has_gmag = False
        self.gmag_ = 0.0
        self.has_hess = has_hess
        self.has_cauchy = False
        self.cauchy_ = np.zeros(0)
        self.has_newton = False
        self.newton_ = np.zeros(0)
        self.previous_tr_radius = -1.0
        self.lambda_lb_ = 0.0
        self.niter = 0
        self.k_easy = 0.1
        self.k_hard = 0.2
        self.maxiter = maxiter
        self.exact_ready = False
        self.hess_gershgorin_lb = 0.0
        self.hess_gershgorin_ub = 0.0
        self.hess_inf = 0.0
        self.hess_fro = 0.0
        self.CLOSE_TO_ZERO = 0.0
        self.lambda_current = 0.0
        self.tol_rel_i = tol_rel_i
        self.tol_rel_b = tol_rel_b
        self.kstate = _TrlibState(x.size)
        if kind == _TR_EXACT:
            if maxiter < 0:
                raise ValueError('maxiter must not be set to a negative number, use np.inf to mean infinite.')
            H = self.hess()
            self.hess_gershgorin_lb, self.hess_gershgorin_ub = gershgorin_bounds(H)
            self.hess_inf = _norm_inf_mat(H)
            self.hess_fro = _norm_fro(H)
            self.CLOSE_TO_ZERO = H.shape[0] * EPS * self.hess_inf

    def fun(self) -> float:
        if not self.has_f:
            self.f_ = self.sf.fun(self.x)
            self.has_f = True
        return self.f_

    def jac(self) -> np.ndarray[float, 1]:
        if not self.has_g:
            self.g_ = self.sf.grad(self.x)
            self.has_g = True
        return self.g_

    def hess(self) -> np.ndarray[float, 2]:
        if not self.has_h:
            self.h_ = self.sf.hess(self.x)
            self.has_h = True
        return self.h_

    def hessp(self, p: np.ndarray[float, 1]) -> np.ndarray[float, 1]:
        if self.has_hess:
            return np.dot(self.hess(), p)
        else:
            return self.hp(self.x, p)

    def jac_mag(self) -> float:
        if not self.has_gmag:
            self.gmag_ = _norm2(self.jac())
            self.has_gmag = True
        return self.gmag_

    def __call__(self, p: np.ndarray[float, 1]) -> float:
        return self.fun() + float(np.dot(self.jac(), p)) + \
            0.5 * float(np.dot(p, self.hessp(p)))

    def solve(self, trust_radius: float):
        if self.kind == _TR_NCG:
            return self._solve_ncg(trust_radius)
        elif self.kind == _TR_DOGLEG:
            return self._solve_dogleg(trust_radius)
        elif self.kind == _TR_EXACT:
            return self._solve_exact(trust_radius)
        else:
            return _trlib_solve(self, trust_radius)

    # ------------------------------------------------------ CG-Steihaug
    def _solve_ncg(self, trust_radius: float):
        jac = self.jac()
        p_origin = np.zeros_like(jac)
        jm = self.jac_mag()
        tolerance = min(0.5, math.sqrt(jm)) * jm
        if jm < tolerance:
            return p_origin, False
        z = p_origin
        r = jac
        d = -r
        while True:
            Bd = self.hessp(d)
            dBd = float(np.dot(d, Bd))
            if dBd <= 0:
                ta, tb = _boundaries_intersections(z, d, trust_radius)
                pa = z + ta * d
                pb = z + tb * d
                if self(pa) < self(pb):
                    return pa, True
                return pb, True
            r_squared = float(np.dot(r, r))
            alpha = r_squared / dBd
            z_next = z + alpha * d
            if _norm2(z_next) >= trust_radius:
                ta, tb = _boundaries_intersections(z, d, trust_radius)
                return z + tb * d, True
            r_next = r + alpha * Bd
            r_next_squared = float(np.dot(r_next, r_next))
            if math.sqrt(r_next_squared) < tolerance:
                return z_next, False
            beta_next = r_next_squared / r_squared
            d_next = -r_next + beta_next * d
            z = z_next
            r = r_next
            d = d_next
        return p_origin, False

    # ------------------------------------------------------------ dogleg
    def cauchy_point(self) -> np.ndarray[float, 1]:
        if not self.has_cauchy:
            g = self.jac()
            Bg = self.hessp(g)
            self.cauchy_ = -(float(np.dot(g, g)) / float(np.dot(g, Bg))) * g
            self.has_cauchy = True
        return self.cauchy_

    def newton_point(self) -> np.ndarray[float, 1]:
        if not self.has_newton:
            g = self.jac()
            B = self.hess()
            cho_info = cho_factor(B)
            self.newton_ = -cho_solve(cho_info, g)
            self.has_newton = True
        return self.newton_

    def _solve_dogleg(self, trust_radius: float):
        p_best = self.newton_point()
        if _norm2(p_best) < trust_radius:
            return p_best, False
        p_u = self.cauchy_point()
        p_u_norm = _norm2(p_u)
        if p_u_norm >= trust_radius:
            return p_u * (trust_radius / p_u_norm), True
        _, tb = _boundaries_intersections(p_u, p_best - p_u, trust_radius)
        return p_u + tb * (p_best - p_u), True

    # ------------------------------------------------------------- exact
    def _initial_values(self, tr_radius: float):
        H = self.hess()
        jm = self.jac_mag()
        lambda_ub = max(0.0, jm / tr_radius + min(-self.hess_gershgorin_lb,
                                                 self.hess_fro, self.hess_inf))
        lambda_lb = max(0.0, max(-float(np.min(np.diag(H))),
                                 jm / tr_radius - min(self.hess_gershgorin_ub,
                                                      self.hess_fro, self.hess_inf)))
        if tr_radius < self.previous_tr_radius:
            lambda_lb = max(self.lambda_lb_, lambda_lb)
        lambda_initial = 0.0
        if lambda_lb != 0:
            lambda_initial = max(math.sqrt(lambda_lb * lambda_ub),
                                 lambda_lb + 0.01 * (lambda_ub - lambda_lb))
        return lambda_initial, lambda_lb, lambda_ub

    def _solve_exact(self, tr_radius: float):
        lambda_current, lambda_lb, lambda_ub = self._initial_values(tr_radius)
        Hs = self.hess()
        jac = self.jac()
        jm = self.jac_mag()
        n = Hs.shape[0]
        hits_boundary = True
        already_factorized = False
        self.niter = 0
        p = np.zeros(n)
        U = np.zeros((n, n))
        H = Hs
        info = 0
        while self.niter < self.maxiter:
            if already_factorized:
                already_factorized = False
            else:
                H = Hs + lambda_current * np.eye(n)
                U, info = _cholesky_info(H)
            self.niter += 1
            if info == 0 and jm > self.CLOSE_TO_ZERO:
                p = cho_solve((U, False), -jac)
                p_norm = _norm2(p)
                if p_norm <= tr_radius and lambda_current == 0:
                    hits_boundary = False
                    break
                w = solve_triangular(U, p, trans='T')
                w_norm = _norm2(w)
                delta_lambda = (p_norm / w_norm) ** 2 * (p_norm - tr_radius) / tr_radius
                lambda_new = lambda_current + delta_lambda
                if p_norm < tr_radius:
                    s_min, z_min = estimate_smallest_singular_value(U)
                    ta, tb = _boundaries_intersections(p, z_min, tr_radius)
                    step_len = ta if abs(ta) <= abs(tb) else tb
                    quadratic_term = float(np.dot(p, np.dot(H, p)))
                    relative_error = step_len ** 2 * s_min ** 2 / \
                        (quadratic_term + lambda_current * tr_radius ** 2)
                    if relative_error <= self.k_hard:
                        p += step_len * z_min
                        break
                    lambda_ub = lambda_current
                    lambda_lb = max(lambda_lb, lambda_current - s_min ** 2)
                    H = Hs + lambda_new * np.eye(n)
                    c, info2 = _cholesky_info(H)
                    info = info2
                    if info == 0:
                        lambda_current = lambda_new
                        already_factorized = True
                        U = c
                    else:
                        lambda_lb = max(lambda_lb, lambda_new)
                        lambda_current = max(math.sqrt(abs(lambda_lb * lambda_ub)),
                                             lambda_lb + 0.01 * (lambda_ub - lambda_lb))
                else:
                    relative_error = abs(p_norm - tr_radius) / tr_radius
                    if relative_error <= self.k_easy:
                        break
                    lambda_lb = lambda_current
                    lambda_current = lambda_new
            elif info == 0 and jm <= self.CLOSE_TO_ZERO:
                if lambda_current == 0:
                    p = np.zeros(n)
                    hits_boundary = False
                    break
                s_min, z_min = estimate_smallest_singular_value(U)
                step_len = tr_radius
                p = step_len * z_min
                if step_len ** 2 * s_min ** 2 <= self.k_hard * lambda_current * tr_radius ** 2:
                    break
                lambda_ub = lambda_current
                lambda_lb = max(lambda_lb, lambda_current - s_min ** 2)
                lambda_current = max(math.sqrt(lambda_lb * lambda_ub),
                                     lambda_lb + 0.01 * (lambda_ub - lambda_lb))
            else:
                delta, v = singular_leading_submatrix(H, U, info)
                v_norm = _norm2(v)
                lambda_lb = max(lambda_lb, lambda_current + delta / v_norm ** 2)
                lambda_current = max(math.sqrt(abs(lambda_lb * lambda_ub)),
                                     lambda_lb + 0.01 * (lambda_ub - lambda_lb))
        self.lambda_lb_ = lambda_lb
        self.lambda_current = lambda_current
        self.previous_tr_radius = tr_radius
        return p, hits_boundary

_TR_MESSAGES = (_status_message_success, _status_message_maxiter,
                'A bad approximation caused failure to predict improvement.',
                'A linalg error occurred, such as a non-psd Hessian.')

def _hess_is_callable(hess) -> Literal[bool]:
    if hess is None:
        return False
    elif isinstance(hess, str):
        return False
    elif isinstance(hess, HessianUpdateStrategy):
        return False
    else:
        return True

def _minimize_trust_region(kind: int, fun, x0: np.ndarray[float, 1], args, jac,
                           hess, hessp, callback, options):
    initial_trust_radius = _opt_float(options, 'initial_trust_radius', 1.0)
    max_trust_radius = _opt_float(options, 'max_trust_radius', 1000.0)
    eta = _opt_float(options, 'eta', 0.15)
    gtol = _opt_float(options, 'gtol', 1e-4)
    maxiter = _opt_int(options, 'maxiter', len(x0) * 200)
    inexact = _opt_bool(options, 'inexact', True)
    sub_maxiter = _opt_int(options, 'subproblem_maxiter', 25)
    if jac is None:
        raise ValueError('Jacobian is currently required for trust-region methods')
    if hess is None and hessp is None:
        raise ValueError('Either the Hessian or the Hessian-vector product is currently required for trust-region methods')
    if not (0 <= eta < 0.25):
        raise ValueError('invalid acceptance stringency')
    if max_trust_radius <= 0:
        raise ValueError('the max trust radius must be positive')
    if initial_trust_radius <= 0:
        raise ValueError('the initial trust radius must be positive')
    if initial_trust_radius >= max_trust_radius:
        raise ValueError('the initial trust radius must be less than the max trust radius')
    sf = _ScalarFunctionH(fun, x0, args, jac, hess)
    has_hess = _hess_is_callable(hess)
    use_user = not has_hess and hessp is not None
    if not has_hess and hessp is None and hess is None:
        raise ValueError('Either the Hessian or the Hessian-vector product is currently required for trust-region methods')
    hp = _HessP(sf, hessp, args, use_user)
    tri = -2.0 if inexact else 1e-8
    trb = -3.0 if inexact else 1e-6
    warnflag = 0
    trust_radius = initial_trust_radius
    x = x0
    m = _QuadModel(kind, x, sf, hp, has_hess, sub_maxiter, tri, trb)
    k = 0
    stopped = False
    while m.jac_mag() >= gtol:
        p = np.zeros(0)
        hits_boundary = False
        try:
            p, hits_boundary = m.solve(trust_radius)
        except LinAlgError:
            warnflag = 3
            break
        predicted_value = m(p)
        x_proposed = x + p
        m_proposed = _QuadModel(kind, x_proposed, sf, hp, has_hess,
                                sub_maxiter, tri, trb)
        actual_reduction = m.fun() - m_proposed.fun()
        predicted_reduction = m.fun() - predicted_value
        if predicted_reduction <= 0:
            warnflag = 2
            break
        rho = actual_reduction / predicted_reduction
        if rho < 0.25:
            trust_radius *= 0.25
        elif rho > 0.75 and hits_boundary:
            trust_radius = min(2 * trust_radius, max_trust_radius)
        if rho > eta:
            x = x_proposed
            m = m_proposed
        k += 1
        if _cb_halt(callback, x, m.fun()):
            stopped = True
            break
        if m.jac_mag() < gtol:
            warnflag = 0
            break
        if k >= maxiter:
            warnflag = 1
            break
    nh = sf.nhev + hp.ncalls
    res = OptimizeResult(x, m.fun(), m.jac(), nit=k, nfev=sf.nfev,
                         njev=sf.ngev, success=(warnflag == 0),
                         status=warnflag, message=_TR_MESSAGES[warnflag])
    res.nhev = nh
    if has_hess:
        res.hess = m.hess()
    if stopped:
        res.success = False
        res.status = 99
        res.message = _STOP_MSG
    return res

# --------------------------------------------------------------- Newton-CG

def _minimize_newtoncg(fun, x0: np.ndarray[float, 1], args, jac, hess, hessp,
                       callback, options):
    if jac is None:
        raise ValueError('Jacobian is required for Newton-CG method')
    avextol = _opt_float(options, 'xtol', 1e-05)
    epsilon = _opt_float(options, 'eps', _epsilon)
    maxiter = _opt_int(options, 'maxiter', len(x0) * 200)
    c1 = _opt_float(options, 'c1', 1e-4)
    c2 = _opt_float(options, 'c2', 0.9)
    sf = _ScalarFunctionH(fun, x0, args, jac, hess, epsilon)
    _h = sf.hess(x0)
    has_hess = _hess_is_callable(hess)
    fd_hess = isinstance(hess, str) or isinstance(hess, HessianUpdateStrategy)
    has_hessp = hessp is not None or fd_hess
    hcalls = 0
    cg_maxiter = 20 * len(x0)
    xtol = len(x0) * avextol
    update_l1norm = 1.7976931348623157e308
    xk = x0.copy()
    k = 0
    gfk = np.zeros(0)
    old_fval = sf.fun(x0)
    old_old_fval: Optional[float] = None
    float64eps = EPS
    warnflag = 0
    msg = ''
    done = False
    while update_l1norm > xtol:
        if k >= maxiter:
            warnflag = 1
            msg = 'Warning: ' + _status_message_maxiter
            done = True
            break
        b = -sf.grad(xk)
        maggrad = float(np.sum(np.abs(b)))
        eta = min(0.5, math.sqrt(maggrad))
        termcond = eta * maggrad
        xsupi = np.zeros(len(x0))
        ri = -b
        psupi = -ri
        i = 0
        dri0 = float(np.dot(ri, ri))
        A = np.zeros((0, 0))
        if has_hess:
            A = sf.hess(xk)
            hcalls += 1
        converged = False
        for k2 in range(cg_maxiter):
            if float(np.add.reduce(np.abs(ri))) <= termcond:
                converged = True
                break
            Ap = np.zeros(0)
            if not has_hess:
                if not has_hessp:
                    g1 = sf.grad(xk)
                    Ap = (sf.grad(xk + epsilon * psupi) - g1) / epsilon
                else:
                    if fd_hess:
                        Ap = sf.hess_dot(xk, psupi)
                        hcalls += 1
                    else:
                        if hessp is not None:
                            Ap = _vec(hessp(xk, psupi, *args))
                        hcalls += 1
            else:
                Ap = A.dot(psupi)
            curv = float(np.dot(psupi, Ap))
            if 0 <= curv <= 3 * float64eps:
                converged = True
                break
            elif curv < 0:
                if i > 0:
                    converged = True
                    break
                else:
                    xsupi = dri0 / -curv * b
                    converged = True
                    break
            alphai = dri0 / curv
            xsupi += alphai * psupi
            ri += alphai * Ap
            dri1 = float(np.dot(ri, ri))
            betai = dri1 / dri0
            psupi = -ri + betai * psupi
            i += 1
            dri0 = dri1
        if not converged:
            warnflag = 3
            msg = "Warning: CG iterations didn't converge. The Hessian is not positive definite."
            done = True
            break
        pk = xsupi
        gfk = -b
        # old_old_fval None -> NaN makes the initial step 1.0 as in scipy
        oof = math.nan if old_old_fval is None else old_old_fval
        ok, alphak, new_f, prev_f, gfkp1, has_g = _line_search_wolfe12(
            sf, xk, pk, gfk, old_fval, oof, c1, c2, 1e-100, 1e100, _true4n)
        if not ok:
            warnflag = 2
            msg = 'Warning: ' + _status_message_pr_loss
            done = True
            break
        old_fval = new_f
        old_old_fval = prev_f
        update = alphak * pk
        xk += update
        k += 1
        if _cb_halt(callback, xk, old_fval):
            warnflag = 99
            msg = _STOP_MSG
            done = True
            break
        update_l1norm = float(np.sum(np.abs(update)))
    if not done:
        if old_fval != old_fval or update_l1norm != update_l1norm:
            warnflag = 3
            msg = _status_message_nan
        else:
            warnflag = 0
            msg = _status_message_success
    res = OptimizeResult(xk, old_fval, gfk, nit=k, nfev=sf.nfev, njev=sf.ngev,
                         status=warnflag, success=(warnflag == 0), message=msg)
    res.nhev = hcalls
    return res

def _true4n(a: float, x: np.ndarray[float, 1], f: float, g: np.ndarray[float, 1]) -> bool:
    return True
