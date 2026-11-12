# Ported from scipy 1.18.1 (BSD-3-Clause)

# Bounded linear least squares: lsq_linear ('trf' and 'bvls',
# _lsq/lsq_linear.py, _lsq/trf_linear.py, _lsq/bvls.py,
# _lsq/givens_elimination.pyx) and nnls (_nnls.py + src/nnls.c).

import math
import numpy as np
from scipy.linalg import qr, solve_triangular
from scipy.sparse.linalg import LinearOperator, lsmr
from ._common import OptimizeResult, Bounds, INF, _norm2, _norm_inf, _vec, \
    _prepare_bounds_pair
from ._lsq import _in_bounds, _find_active, _make_strictly_feasible, \
    _cl_scaling, _step_size_to_bound, _minimize_quadratic_1d
from ._lsq_ext import _RegMV, _RegRMV
from ._slsqp_core import _nnls
from ._blaslapack import dlartg

EPS = 2.220446049250313e-16

_LIN_MESSAGES = ("The algorithm was not able to make progress on the last iteration.",
                 "The maximum number of iterations is exceeded.",
                 "The first-order optimality measure is less than `tol`.",
                 "The relative change of the cost function is less than `tol`.",
                 "The unconstrained solution is optimal.")

def _lin_msg(status: int) -> str:
    return _LIN_MESSAGES[status + 1]

def _reflective_transformation(y: np.ndarray[float, 1], lb: np.ndarray[float, 1],
                               ub: np.ndarray[float, 1]):
    n = y.size
    if _in_bounds(y, lb, ub):
        return y, np.ones(n)
    x = y.copy()
    g = np.ones(n)
    for i in range(n):
        lf = math.isfinite(lb[i])
        uf = math.isfinite(ub[i])
        if lf and not uf:
            x[i] = max(y[i], 2 * lb[i] - y[i])
            if y[i] < lb[i]:
                g[i] = -1.0
        elif (not lf) and uf:
            x[i] = min(y[i], 2 * ub[i] - y[i])
            if y[i] > ub[i]:
                g[i] = -1.0
        elif lf and uf:
            d = ub[i] - lb[i]
            t = float(np.remainder(y[i] - lb[i], 2 * d))
            x[i] = lb[i] + min(t, 2 * d - t)
            if t > d:
                g[i] = -1.0
    return x, g

def _eval_q(A: np.ndarray[float, 2], d: np.ndarray[float, 1], g: np.ndarray[float, 1],
            s: np.ndarray[float, 1], diag: Optional[np.ndarray[float, 1]]) -> float:
    # evaluate_quadratic with J = A * d (right multiplied operator)
    Js = A @ (s * d)
    q = float(np.dot(Js, Js))
    if diag is not None:
        dg: np.ndarray[float, 1] = diag
        q += float(np.dot(s * dg, s))
    return 0.5 * q + float(np.dot(s, g))

def _bq1d(A: np.ndarray[float, 2], d: np.ndarray[float, 1], g: np.ndarray[float, 1],
          s: np.ndarray[float, 1], diag: np.ndarray[float, 1],
          s0: Optional[np.ndarray[float, 1]]):
    v = A @ (s * d)
    a = float(np.dot(v, v))
    a += float(np.dot(s * diag, s))
    a *= 0.5
    b = float(np.dot(g, s))
    c = 0.0
    if s0 is not None:
        t0: np.ndarray[float, 1] = s0
        u = A @ (t0 * d)
        b += float(np.dot(u, v))
        c = 0.5 * float(np.dot(u, u)) + float(np.dot(g, t0))
        b += float(np.dot(t0 * diag, s))
        c += 0.5 * float(np.dot(t0 * diag, t0))
    return a, b, c

def _givens_elimination(S: np.ndarray[float, 2], v: np.ndarray[float, 1],
                        diag: np.ndarray[float, 1]):
    n = diag.size
    diag_row = np.empty(n)
    for i in range(n):
        if diag[i] == 0:
            continue
        for j in range(i + 1, n):
            diag_row[j] = 0.0
        diag_row[i] = diag[i]
        u = 0.0
        for j in range(i, n):
            if diag_row[j] != 0:
                cs, sn, r = dlartg(S[j, j], diag_row[j])
                S[j, j] = r
                for k in range(j + 1, n):
                    sx = S[j, k]
                    dy = diag_row[k]
                    S[j, k] = cs * sx + sn * dy
                    diag_row[k] = cs * dy - sn * sx
                fv = v[j]
                v[j] = cs * fv + sn * u
                u = -sn * fv + cs * u

def _regularized_lsq_with_qr(m: int, n: int, R: np.ndarray[float, 2],
                             QTb: np.ndarray[float, 1], perm: np.ndarray[int, 1],
                             diag: np.ndarray[float, 1]) -> np.ndarray[float, 1]:
    v = QTb.copy()
    dp = np.empty(n)
    for i in range(n):
        dp[i] = diag[perm[i]]
    _givens_elimination(R, v, dp)
    mx = 0.0
    for i in range(n):
        mx = max(mx, abs(R[i, i]))
    threshold = EPS * max(m, n) * mx
    nns = List[int]()
    for i in range(n):
        if abs(R[i, i]) > threshold:
            nns.append(i)
    k = len(nns)
    Rs = np.empty((k, k))
    vs = np.empty(k)
    for a in range(k):
        vs[a] = v[nns[a]]
        for b in range(k):
            Rs[a, b] = R[nns[a], nns[b]]
    x = np.zeros(n)
    if k > 0:
        sol = solve_triangular(Rs, vs)
        for a in range(k):
            x[perm[nns[a]]] = sol[a]
    return x

# ------------------------------------------------------------ trf_linear

def _select_step_lin(x, A: np.ndarray[float, 2], g_h, c_h, p, p_h, d, lb, ub,
                     theta: float) -> np.ndarray[float, 1]:
    if _in_bounds(x + p, lb, ub):
        return p
    p_stride, hits = _step_size_to_bound(x, p, lb, ub)
    r_h = p_h.copy()
    for i in range(r_h.size):
        if hits[i] != 0:
            r_h[i] *= -1
    r = d * r_h
    p = p * p_stride
    p_h = p_h * p_stride
    x_on_bound = x + p
    r_stride_u, _ = _step_size_to_bound(x_on_bound, r, lb, ub)
    r_stride_l = (1 - theta) * r_stride_u
    r_stride_u *= theta
    r_value = INF
    if r_stride_u > 0:
        a, b, c = _bq1d(A, d, g_h, r_h, c_h, p_h)
        r_stride, rv = _minimize_quadratic_1d(a, b, r_stride_l, r_stride_u, c)
        r_value = rv
        r_h = p_h + r_h * r_stride
        r = d * r_h
    p_h = p_h * theta
    p = p * theta
    p_value = _eval_q(A, d, g_h, p_h, c_h)
    ag_h = -g_h
    ag = d * ag_h
    ag_stride_u, _ = _step_size_to_bound(x, ag, lb, ub)
    ag_stride_u *= theta
    a2, b2, _c2 = _bq1d(A, d, g_h, ag_h, c_h, None)
    ag_stride, ag_value = _minimize_quadratic_1d(a2, b2, 0.0, ag_stride_u)
    ag = ag * ag_stride
    if p_value < r_value and p_value < ag_value:
        return p
    elif r_value < p_value and r_value < ag_value:
        return r
    else:
        return ag

def _backtracking(A: np.ndarray[float, 2], g, x, p, theta: float, p_dot_g: float,
                  lb, ub):
    alpha = 1.0
    n = x.size
    ones = np.ones(n)
    x_new = x
    step = x
    cost_change = 0.0
    while True:
        x_new, _ = _reflective_transformation(x + alpha * p, lb, ub)
        step = x_new - x
        cost_change = -_eval_q(A, ones, g, step, None)
        if cost_change > -0.1 * alpha * p_dot_g:
            break
        alpha *= 0.5
    active = _find_active(x_new, lb, ub, 1e-10)
    anyact = False
    for i in range(n):
        if active[i] != 0:
            anyact = True
    if anyact:
        x_new, _ = _reflective_transformation(x + theta * alpha * p, lb, ub)
        x_new = _make_strictly_feasible(x_new, lb, ub, 0.0)
        step = x_new - x
        cost_change = -_eval_q(A, ones, g, step, None)
    return x, step, cost_change

def _trf_linear(A: np.ndarray[float, 2], b: np.ndarray[float, 1],
                x_lsq: np.ndarray[float, 1], lb, ub, tol: float, lsq_solver: str,
                lsmr_tol: float, auto_lsmr_tol: bool, max_iter: int,
                lsmr_maxiter: int):
    m, n = A.shape
    x, _ = _reflective_transformation(x_lsq, lb, ub)
    x = _make_strictly_feasible(x, lb, ub, 0.1)
    QT = np.zeros((0, 0))
    R = np.zeros((0, 0))
    perm = np.zeros(0, int)
    QTr = np.zeros(n)
    k = min(m, n)
    if lsq_solver == 'exact':
        Q, R0, perm0 = qr(A, mode='economic', pivoting=True)
        QT = Q.T
        R = R0
        perm = perm0
        if m < n:
            R = np.concatenate((R, np.zeros((n - m, n))), axis=0)
    r_aug = np.zeros(m + n)
    r = A @ x - b
    g = A.T @ r
    cost = 0.5 * float(np.dot(r, r))
    initial_cost = cost
    termination_status = -100
    g_norm = 0.0
    iteration = 0
    ones = np.ones(n)
    for it in range(max_iter):
        iteration = it
        v, dv = _cl_scaling(x, g, lb, ub)
        g_scaled = g * v
        g_norm = _norm_inf(g_scaled)
        if g_norm < tol:
            termination_status = 1
        if termination_status != -100:
            break
        diag_h = g * dv
        diag_root_h = diag_h ** 0.5
        d = v ** 0.5
        g_h = d * g
        p_h = np.zeros(n)
        if lsq_solver == 'exact':
            QTr[:k] = QT @ r
            Rd = R.copy()
            for j in range(n):
                dj = d[perm[j]]
                for i in range(R.shape[0]):
                    Rd[i, j] = R[i, j] * dj
            p_h = -_regularized_lsq_with_qr(m, n, Rd, QTr, perm, diag_root_h)
        else:
            op = LinearOperator((m + n, n), _RegMV(A, d, diag_root_h),
                                _RegRMV(A, d, diag_root_h))
            r_aug[:m] = r
            lt = lsmr_tol
            if auto_lsmr_tol:
                eta = 0.01 * min(0.5, g_norm)
                lt = max(EPS, min(0.1, eta * g_norm))
            if lsmr_maxiter > 0:
                p_h = -np.asarray(lsmr(op, r_aug, 0.0, lt, lt, 1e8, lsmr_maxiter)[0], float)
            else:
                p_h = -np.asarray(lsmr(op, r_aug, 0.0, lt, lt)[0], float)
        p = d * p_h
        p_dot_g = float(np.dot(p, g))
        if p_dot_g > 0:
            termination_status = -1
        theta = 1 - min(0.005, g_norm)
        step = _select_step_lin(x, A, g_h, diag_h, p, p_h, d, lb, ub, theta)
        cost_change = -_eval_q(A, ones, g, step, None)
        if cost_change < 0:
            x, step, cost_change = _backtracking(A, g, x, p, theta, p_dot_g, lb, ub)
        else:
            x = _make_strictly_feasible(x + step, lb, ub, 0.0)
        r = A @ x - b
        g = A.T @ r
        if cost_change < tol * cost:
            termination_status = 2
        cost = 0.5 * float(np.dot(r, r))
        iteration = it
    if termination_status == -100:
        termination_status = 0
    active_mask = _find_active(x, lb, ub, tol)
    return x, r, cost, g_norm, active_mask, iteration + 1, termination_status

# ------------------------------------------------------------------ bvls

def _kkt_opt(g: np.ndarray[float, 1], on_bound: np.ndarray[float, 1]) -> float:
    mx = -INF
    for i in range(g.size):
        v = abs(g[i]) if on_bound[i] == 0 else g[i] * on_bound[i]
        if v > mx:
            mx = v
    return mx

def _cols(A: np.ndarray[float, 2], idx: List[int]) -> np.ndarray[float, 2]:
    m = A.shape[0]
    R = np.empty((m, len(idx)))
    for k in range(len(idx)):
        for i in range(m):
            R[i, k] = A[i, idx[k]]
    return R

def _lstsq(A: np.ndarray[float, 2], b: np.ndarray[float, 1]) -> np.ndarray[float, 1]:
    if A.shape[1] == 0:
        return np.zeros(0)
    return np.asarray(np.linalg.lstsq(A, b, rcond=None)[0], float)

def _bvls(A: np.ndarray[float, 2], b: np.ndarray[float, 1], x_lsq: np.ndarray[float, 1],
          lb, ub, tol: float, max_iter: int):
    m, n = A.shape
    x = x_lsq.copy()
    on_bound = np.zeros(n)
    for i in range(n):
        if x[i] <= lb[i]:
            x[i] = lb[i]
            on_bound[i] = -1.0
    for i in range(n):
        if x[i] >= ub[i]:
            x[i] = ub[i]
            on_bound[i] = 1.0
    active_set = np.zeros(n, bool)
    free_set = List[int]()
    for i in range(n):
        if on_bound[i] == 0:
            free_set.append(i)
        else:
            active_set[i] = True
    r = A @ x - b
    cost = 0.5 * float(np.dot(r, r))
    g = A.T @ r
    iteration = 0
    while len(free_set) > 0:
        iteration += 1
        A_free = _cols(A, free_set)
        xa = x.copy()
        for i in range(n):
            if not active_set[i]:
                xa[i] = 0.0
        b_free = b - A @ xa
        z = _lstsq(A_free, b_free)
        anyv = False
        keep = List[int]()
        for k in range(len(free_set)):
            i = free_set[k]
            if z[k] < lb[i]:
                x[i] = lb[i]
                active_set[i] = True
                on_bound[i] = -1.0
                anyv = True
            elif z[k] > ub[i]:
                x[i] = ub[i]
                active_set[i] = True
                on_bound[i] = 1.0
                anyv = True
            else:
                x[i] = z[k]
                keep.append(i)
        r = A @ x - b
        cost = 0.5 * float(np.dot(r, r))
        g = A.T @ r
        if anyv:
            free_set = keep
        else:
            break
    max_iter += iteration
    termination_status = -100
    optimality = _kkt_opt(g, on_bound)
    it = iteration
    for it2 in range(iteration, max_iter):
        it = it2
        if optimality < tol:
            termination_status = 1
        if termination_status != -100:
            break
        best = 0
        bv = -INF
        for i in range(n):
            v = g[i] * on_bound[i]
            if v > bv:
                bv = v
                best = i
        on_bound[best] = 0.0
        x_free = np.zeros(0)
        x_free_old = np.zeros(0)
        while True:
            fs = List[int]()
            act = np.zeros(n, bool)
            for i in range(n):
                if on_bound[i] == 0:
                    fs.append(i)
                else:
                    act[i] = True
            x_free = np.empty(len(fs))
            for k in range(len(fs)):
                x_free[k] = x[fs[k]]
            x_free_old = x_free.copy()
            A_free = _cols(A, fs)
            xa = x.copy()
            for i in range(n):
                if not act[i]:
                    xa[i] = 0.0
            b_free = b - A @ xa
            z = _lstsq(A_free, b_free)
            lbv = List[int]()
            ubv = List[int]()
            for k in range(len(fs)):
                if z[k] < lb[fs[k]]:
                    lbv.append(k)
            for k in range(len(fs)):
                if z[k] > ub[fs[k]]:
                    ubv.append(k)
            nv = len(lbv) + len(ubv)
            if nv > 0:
                bi = 0
                ba = INF
                for q in range(nv):
                    k = lbv[q] if q < len(lbv) else ubv[q - len(lbv)]
                    num = (lb[fs[k]] - x_free[k]) if q < len(lbv) else (ub[fs[k]] - x_free[k])
                    al = num / (z[k] - x_free[k])
                    if al < ba:
                        ba = al
                        bi = q
                i_free = lbv[bi] if bi < len(lbv) else ubv[bi - len(lbv)]
                x_free *= 1 - ba
                x_free += ba * z
                for k in range(len(fs)):
                    x[fs[k]] = x_free[k]
                # scipy: on_bound[free_set[i_free]] with `i` < lbv.size test
                if bi < len(lbv):
                    on_bound[fs[i_free]] = -1.0
                else:
                    on_bound[fs[i_free]] = 1.0
            else:
                x_free = z
                for k in range(len(fs)):
                    x[fs[k]] = x_free[k]
                break
        r = A @ x - b
        cost_new = 0.5 * float(np.dot(r, r))
        cost_change = cost - cost_new
        if cost_change < tol * cost:
            termination_status = 2
        cost = cost_new
        g = A.T @ r
        optimality = _kkt_opt(g, on_bound)
        it = it2
    if termination_status == -100:
        termination_status = 0
    return x, r, cost, optimality, on_bound, it + 1, termination_status

# ------------------------------------------------------------ lsq_linear

def lsq_linear(A, b, bounds = (-float('inf'), float('inf')), method: str = 'trf',
               tol: float = 1e-10, lsq_solver: Optional[str] = None,
               lsmr_tol = None, max_iter: Optional[int] = None,
               verbose: int = 0, lsmr_maxiter: Optional[int] = None):
    if method != 'trf' and method != 'bvls':
        raise ValueError("`method` must be 'trf' or 'bvls'")
    solver = 'exact'
    if lsq_solver is not None:
        if lsq_solver != 'exact' and lsq_solver != 'lsmr':
            raise ValueError("`solver` must be None, 'exact' or 'lsmr'.")
        solver = lsq_solver
    if method == 'bvls' and solver == 'lsmr':
        raise ValueError("method='bvls' can't be used with lsq_solver='lsmr'")
    if verbose != 0 and verbose != 1 and verbose != 2:
        raise ValueError('`verbose` must be in [0, 1, 2].')
    if max_iter is not None and max_iter <= 0:
        raise ValueError('`max_iter` must be None or positive integer.')
    Am = np.atleast_2d(np.asarray(A, float))
    if Am.ndim != 2:
        raise ValueError('`A` must have at most 2 dimensions.')
    m, n = Am.shape
    bv = np.atleast_1d(np.asarray(b, float)).flatten()
    if bv.size != m:
        raise ValueError('Inconsistent shapes between `A` and `b`.')
    lb, ub = _prepare_bounds_pair(bounds, n)
    for i in range(n):
        if lb[i] >= ub[i]:
            raise ValueError('Each lower bound must be strictly less than each upper bound.')
    if lsmr_maxiter is not None and lsmr_maxiter < 1:
        raise ValueError('`lsmr_maxiter` must be None or positive integer.')
    lmi = -1 if lsmr_maxiter is None else lsmr_maxiter
    ltol = 0.01 * tol
    auto = False
    if lsmr_tol is not None:
        if isinstance(lsmr_tol, str):
            if lsmr_tol != 'auto':
                raise ValueError("`lsmr_tol` must be None, 'auto', or positive float.")
            auto = True
        else:
            ltol = float(lsmr_tol)
            if not ltol > 0:
                raise ValueError("`lsmr_tol` must be None, 'auto', or positive float.")
    x_lsq = np.zeros(n)
    if solver == 'exact':
        x_lsq = np.asarray(np.linalg.lstsq(Am, bv, rcond=-1)[0], float)
    else:
        fl = 0.01 * tol
        if lsmr_tol is not None and not auto:
            fl = ltol
        if lmi > 0:
            x_lsq = np.asarray(lsmr(Am, bv, 0.0, fl, fl, 1e8, lmi)[0], float)
        else:
            x_lsq = np.asarray(lsmr(Am, bv, 0.0, fl, fl)[0], float)
    if _in_bounds(x_lsq, lb, ub):
        r = Am @ x_lsq - bv
        cost = 0.5 * float(np.dot(r, r))
        g = Am.T @ r
        res = OptimizeResult(x_lsq, r, np.empty((0, 0)), nit=0, status=3,
                             message=_lin_msg(3), success=True, cost=cost,
                             optimality=_norm_inf(g))
        res.active_mask = np.zeros(n, int)
        return res
    if method == 'trf':
        mi = 100 if max_iter is None else max_iter
        x, r, cost, opt, am, nit, status = _trf_linear(Am, bv, x_lsq, lb, ub, tol,
                                                      solver, ltol, auto, mi, lmi)
        res = OptimizeResult(x, r, np.empty((0, 0)), nit=nit, status=status,
                             message=_lin_msg(status), success=status > 0,
                             cost=cost, optimality=opt)
        res.active_mask = am.astype(int)
        return res
    else:
        mi = n if max_iter is None else max_iter
        x, r, cost, opt, am, nit, status = _bvls(Am, bv, x_lsq, lb, ub, tol, mi)
        res = OptimizeResult(x, r, np.empty((0, 0)), nit=nit, status=status,
                             message=_lin_msg(status), success=status > 0,
                             cost=cost, optimality=opt)
        res.active_mask = am.astype(int)
        return res

# ------------------------------------------------------------------ nnls

def nnls(A, b, maxiter: Optional[int] = None, atol = None):
    Am = np.asarray(A, float)
    bv = np.asarray(b, float)
    if Am.ndim != 2:
        raise ValueError(f'Expected a two-dimensional array, but the shape of A is {Am.shape}')
    m, n = Am.shape
    if bv.ndim != 1:
        raise ValueError('Expected a one-dimensional array, but the shape of b is ' + str(bv.shape))
    if m != bv.shape[0]:
        raise ValueError('Incompatible dimensions. The first dimension of A is ' +
                         f'{m}, while the shape of b is {(bv.shape[0],)}')
    if n == 0:
        return np.empty(0), _norm2(bv)
    mi = 3 * n if (maxiter is None or maxiter == 0) else maxiter
    a = Ptr[float](max(m * n, 1))
    for j in range(n):
        for i in range(m):
            a[i + j * m] = Am[i, j]
    bp = Ptr[float](max(m, 1))
    for i in range(m):
        bp[i] = bv[i]
    x = Ptr[float](max(n, 1))
    w = Ptr[float](max(n, 1))
    zz = Ptr[float](max(m, 1))
    idx = Ptr[int](max(n, 1))
    rnorm, info = _nnls(m, n, a, bp, x, w, zz, idx, mi)
    if info == 3:
        raise RuntimeError('Maximum number of iterations reached.')
    xs = np.empty(n)
    for i in range(n):
        xs[i] = x[i]
    return xs, rnorm
