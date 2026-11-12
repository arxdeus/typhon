# Ported from scipy 1.18.1 (BSD-3-Clause)

# linprog (methods 'highs', 'highs-ds', 'highs-ipm') and milp.
# scipy delegates both to the HiGHS C++ library; this port solves the same
# problem formulation (rows rl <= A x <= ru built from A_ub/b_ub and
# A_eq/b_eq, column bounds) with the dense bounded primal simplex of
# _linprog_simplex.py and, for integer variables, a best-bound
# branch-and-bound on top of it.
# Results: x, fun, slack, con, status and message match HiGHS for problems
# with a unique optimum; marginals (ineqlin/eqlin/lower/upper duals) match
# when the dual solution is unique. Degenerate problems or problems with
# several optimal vertices may return a different optimal vertex / duals.
# `nit` and `mip_node_count` count this solver's iterations / nodes and do
# not match HiGHS. When HiGHS returns no solution (x=None, fun=None) this
# port returns an empty `x` and fun=nan. All three method names use the
# same solver. Unbounded problems report HiGHS status 10 (HiGHS presolve
# sometimes reports 9, "unbounded or infeasible").

import math
import time
import numpy as np
from ._common import Bounds, _bound_val
from ._constraints import LinearConstraint
from ._linprog_simplex import solve_lp, LPSolution, LP_OPTIMAL, \
    LP_ITER_LIMIT, LP_INFEASIBLE, LP_UNBOUNDED, LP_TIME_LIMIT

class _Sens:
    residual: np.ndarray[float, 1]
    marginals: np.ndarray[float, 1]
    def __init__(self, residual: np.ndarray[float, 1],
                 marginals: np.ndarray[float, 1]):
        self.residual = residual
        self.marginals = marginals
    def __repr__(self):
        return f" residual: {self.residual}\n marginals: {self.marginals}"

class LinprogResult:
    x: np.ndarray[float, 1]
    fun: float
    slack: np.ndarray[float, 1]
    con: np.ndarray[float, 1]
    success: bool
    status: int
    message: str
    nit: int
    crossover_nit: int
    ineqlin: _Sens
    eqlin: _Sens
    lower: _Sens
    upper: _Sens
    mip_node_count: int
    mip_dual_bound: float
    mip_gap: float

    def __init__(self, x: np.ndarray[float, 1], fun: float,
                 slack: np.ndarray[float, 1], con: np.ndarray[float, 1],
                 status: int, message: str, nit: int, ineqlin: _Sens,
                 eqlin: _Sens, lower: _Sens, upper: _Sens,
                 mip_node_count: int = 0, mip_dual_bound: float = 0.0,
                 mip_gap: float = 0.0):
        self.x = x
        self.fun = fun
        self.slack = slack
        self.con = con
        self.status = status
        self.success = status == 0
        self.message = message
        self.nit = nit
        self.crossover_nit = 0
        self.ineqlin = ineqlin
        self.eqlin = eqlin
        self.lower = lower
        self.upper = upper
        self.mip_node_count = mip_node_count
        self.mip_dual_bound = mip_dual_bound
        self.mip_gap = mip_gap

    def __repr__(self):
        return (f" message: {self.message}\n success: {self.success}\n"
                f"  status: {self.status}\n     fun: {self.fun}\n"
                f"       x: {self.x}\n     nit: {self.nit}")

def _e() -> np.ndarray[float, 1]:
    return np.empty(0, float)

_MSG_OPT = 'Optimization terminated successfully. (HiGHS Status 7: Optimal)'
_MSG_INF = 'The problem is infeasible. (HiGHS Status 8: model_status is Infeasible; primal_status is None)'
_MSG_UNB = 'The problem is unbounded. (HiGHS Status 10: model_status is Unbounded; primal_status is Feasible)'
_MSG_UNB_INF = 'The problem is unbounded or infeasible. (HiGHS Status 9: model_status is Primal infeasible or unbounded; primal_status is None)'

def _msg_iter(feasible: bool) -> str:
    ps = 'Feasible' if feasible else 'Infeasible'
    return f'Iteration limit reached. (HiGHS Status 14: model_status is Iteration limit reached; primal_status is {ps})'

def _msg_time(feasible: bool) -> str:
    ps = 'Feasible' if feasible else 'None'
    return f'Time limit reached. (HiGHS Status 13: model_status is Time limit reached; primal_status is {ps})'

def _msg_nodes(feasible: bool) -> str:
    ps = 'Feasible' if feasible else 'None'
    return f'The HiGHS status code was not recognized. (HiGHS Status 16: model_status is Solution limit reached; primal_status is {ps})'

# ------------------------------------------------------------ input parsing

def _mat(A, ncol: int) -> np.ndarray[float, 2]:
    if isinstance(A, NoneType):
        return np.zeros((0, ncol))
    else:
        a = np.asarray(A, float)
        if a.ndim == 1:
            if a.size == 0:
                return np.zeros((0, ncol))
            return a.reshape((1, a.size))
        return np.atleast_2d(a).astype(float)

def _vec1(b) -> np.ndarray[float, 1]:
    if isinstance(b, NoneType):
        return np.zeros(0)
    else:
        return np.atleast_1d(np.asarray(b, float)).flatten()

def _is_scalar_like(v) -> Literal[bool]:
    if isinstance(v, NoneType) or isinstance(v, int) or isinstance(v, float):
        return True
    else:
        if isinstance(v, Optional):
            return True
        return False

def _lp_bounds(bounds, n: int):
    lb = np.zeros(n)
    ub = np.full(n, math.inf)
    if isinstance(bounds, NoneType):
        return lb, ub
    elif isinstance(bounds, Bounds):
        return (np.broadcast_to(bounds.lb, (n,)).astype(float),
                np.broadcast_to(bounds.ub, (n,)).astype(float))
    elif isinstance(bounds, Tuple):
        if static.len(bounds) == 2 and _is_scalar_like(bounds[0]) and _is_scalar_like(bounds[1]):
            l = _bound_val(bounds[0], -math.inf)
            u = _bound_val(bounds[1], math.inf)
            return np.full(n, l), np.full(n, u)
        else:
            return _lp_bounds_seq(bounds, n)
    else:
        return _lp_bounds_seq(bounds, n)

def _lp_bounds_seq(bounds, n: int):
    k = len(bounds)
    lb = np.zeros(n)
    ub = np.full(n, math.inf)
    if k == 1:
        for b in bounds:
            l = _bound_val(b[0], -math.inf)
            u = _bound_val(b[1], math.inf)
            lb = np.full(n, l)
            ub = np.full(n, u)
        return lb, ub
    if k != n:
        raise ValueError(f"Invalid input for linprog: unable to interpret bounds with this dimension tuple: ({k}, 2).")
    i = 0
    for b in bounds:
        lb[i] = _bound_val(b[0], -math.inf)
        ub[i] = _bound_val(b[1], math.inf)
        i += 1
    return lb, ub

# ----------------------------------------------------------- MIP solver

def _cut(best_f: float) -> float:
    if best_f == math.inf:
        return math.inf
    return best_f - 1e-9 * max(1.0, abs(best_f))

class _MipOut:
    status: int          # LP_* codes, plus 5 = node limit
    sol: LPSolution
    has_sol: bool
    nodes: int
    nit: int
    bound: float
    def __init__(self, status: int, sol: LPSolution, has_sol: bool,
                 nodes: int, nit: int, bound: float):
        self.status = status
        self.sol = sol
        self.has_sol = has_sol
        self.nodes = nodes
        self.nit = nit
        self.bound = bound

def _solve_mip(c: np.ndarray[float, 1], A: np.ndarray[float, 2],
               rl: np.ndarray[float, 1], ru: np.ndarray[float, 1],
               lo: np.ndarray[float, 1], hi: np.ndarray[float, 1],
               integ: np.ndarray[bool, 1], maxiter: int, time_limit: float,
               node_limit: int) -> _MipOut:
    n = c.size
    t0 = time.time()
    lo = lo.copy()
    hi = hi.copy()
    for j in range(n):
        if integ[j]:
            lo[j] = math.ceil(lo[j] - 1e-9) if lo[j] > -math.inf else lo[j]
            hi[j] = math.floor(hi[j] + 1e-9) if hi[j] < math.inf else hi[j]
    root = solve_lp(c, A, rl, ru, lo, hi, maxiter)
    nit = root.nit
    if root.status != LP_OPTIMAL:
        return _MipOut(root.status, root, False, 1, nit, math.nan)
    # best-first branch and bound; nodes are (bound, lo, hi)
    open_b = [root.fun]
    open_lo = [lo]
    open_hi = [hi]
    open_sol = [root]
    best = root
    has_best = False
    best_f = math.inf
    nodes = 0
    tol = 1e-6
    while len(open_b) > 0:
        if time.time() - t0 > time_limit:
            bnd = min(open_b)
            if not has_best:
                return _MipOut(LP_TIME_LIMIT, root, False, nodes, nit, bnd)
            return _MipOut(LP_TIME_LIMIT, best, True, nodes, nit, bnd)
        if nodes >= node_limit:
            bnd = min(open_b)
            if not has_best:
                return _MipOut(5, root, False, nodes, nit, bnd)
            return _MipOut(5, best, True, nodes, nit, bnd)
        k = 0
        for i in range(1, len(open_b)):
            if open_b[i] < open_b[k]:
                k = i
        fb = open_b.pop(k)
        nlo = open_lo.pop(k)
        nhi = open_hi.pop(k)
        s = open_sol.pop(k)
        nodes += 1
        if fb >= _cut(best_f):
            continue
        # most fractional integer variable (lowest index on ties)
        bj = -1
        bfrac = 0.0
        for j in range(n):
            if integ[j]:
                v = s.x[j]
                fr = abs(v - math.floor(v + 0.5))
                if fr > tol and fr > bfrac + 1e-12:
                    bfrac = fr
                    bj = j
        if bj < 0:
            if s.fun < best_f:
                xr = s.x.copy()
                for j in range(n):
                    if integ[j]:
                        xr[j] = float(math.floor(xr[j] + 0.5))
                best = LPSolution(s.status, xr, A @ xr if A.shape[0] > 0 else np.zeros(0),
                                  s.y, s.d, s.at_hi, s.basic,
                                  float(np.dot(c, xr)), s.nit, True)
                best_f = best.fun
                has_best = True
            continue
        v = s.x[bj]
        for side in range(2):
            clo = nlo.copy()
            chi = nhi.copy()
            if side == 0:
                chi[bj] = math.floor(v)
            else:
                clo[bj] = math.ceil(v)
            if clo[bj] > chi[bj]:
                continue
            ch = solve_lp(c, A, rl, ru, clo, chi, maxiter)
            nit += ch.nit
            if ch.status == LP_OPTIMAL and ch.fun < _cut(best_f):
                open_b.append(ch.fun)
                open_lo.append(clo)
                open_hi.append(chi)
                open_sol.append(ch)
    if not has_best:
        return _MipOut(LP_INFEASIBLE, root, False, nodes, nit, math.nan)
    else:
        return _MipOut(LP_OPTIMAL, best, True, nodes, nit, best_f)

# ------------------------------------------------------------- linprog

def _build_result(sol: LPSolution, status: int, message: str, nit: int,
                  has_x: bool, n: int, mub: int, meq: int,
                  b_ub: np.ndarray[float, 1], b_eq: np.ndarray[float, 1],
                  lb: np.ndarray[float, 1], ub: np.ndarray[float, 1],
                  duals: bool, mip: bool, nodes: int, dbound: float) -> LinprogResult:
    if not has_x:
        return LinprogResult(_e(), math.nan, _e(), _e(), status, message,
                             nit, _Sens(_e(), _e()), _Sens(_e(), _e()),
                             _Sens(_e(), _e()), _Sens(_e(), _e()), nodes,
                             dbound, 0.0)
    x = sol.x
    slack = b_ub - sol.act[:mub]
    con = b_eq - sol.act[mub:mub + meq]
    mi = np.zeros(mub)
    me = np.zeros(meq)
    ml = np.zeros(n)
    mu = np.zeros(n)
    if duals:
        for i in range(mub):
            mi[i] = sol.y[i]
        for i in range(meq):
            me[i] = sol.y[mub + i]
        for j in range(n):
            if not sol.basic[j]:
                if sol.at_hi[j]:
                    mu[j] = sol.d[j]
                else:
                    ml[j] = sol.d[j]
    gap = 0.0
    return LinprogResult(x, sol.fun, slack, con, status, message, nit,
                         _Sens(slack, mi), _Sens(con, me), _Sens(x - lb, ml),
                         _Sens(ub - x, mu), nodes, dbound, gap)

def _opt_num(options, key: str, default: float) -> float:
    if isinstance(options, NoneType):
        return default
    else:
        if key in options:
            return float(options[key])
        return default

def linprog(c, A_ub = None, b_ub = None, A_eq = None, b_eq = None,
            bounds = (0, None), method: str = 'highs', callback = None,
            options = None, x0 = None, integrality = None):
    meth = method.lower()
    if meth != 'highs' and meth != 'highs-ds' and meth != 'highs-ipm':
        if meth == 'simplex' or meth == 'revised simplex' or meth == 'interior-point':
            raise NotImplementedError(f"method='{meth}' was removed from scipy; use 'highs'")
        raise ValueError(f"Unknown solver '{method}'")
    if not isinstance(callback, NoneType):
        raise NotImplementedError("HiGHS solvers do not support the callback interface.")
    cv = _vec1(c)
    n = cv.size
    Aub = _mat(A_ub, n)
    bub = _vec1(b_ub)
    Aeq = _mat(A_eq, n)
    beq = _vec1(b_eq)
    if Aub.shape[0] != bub.size:
        raise ValueError("Invalid input for linprog: b_ub must be a 1-D array of values, not a scalar; the number of rows in A_ub must be equal to the number of values in b_ub")
    if Aeq.shape[0] != beq.size:
        raise ValueError("Invalid input for linprog: b_eq must be a 1-D array of values, not a scalar; the number of rows in A_eq must be equal to the number of values in b_eq")
    if (Aub.shape[0] > 0 and Aub.shape[1] != n) or (Aeq.shape[0] > 0 and Aeq.shape[1] != n):
        raise ValueError("Invalid input for linprog: the number of columns in A_ub/A_eq must be equal to the size of c")
    lb, ub = _lp_bounds(bounds, n)
    mub = Aub.shape[0]
    meq = Aeq.shape[0]
    A = np.concatenate((Aub, Aeq), axis=0)
    rl = np.concatenate((np.full(mub, -math.inf), beq))
    ru = np.concatenate((bub, beq))
    maxiter = int(_opt_num(options, 'maxiter', 4.6e18))
    time_limit = _opt_num(options, 'time_limit', math.inf)
    integ = np.zeros(n, bool)
    any_int = False
    if not isinstance(integrality, NoneType) and meth == 'highs':
        ia = np.broadcast_to(np.asarray(integrality, int), (n,))
        for j in range(n):
            if ia[j] != 0:
                integ[j] = True
                any_int = True
    if any_int:
        mo = _solve_mip(cv, A, rl, ru, lb, ub, integ, maxiter, time_limit,
                        int(_opt_num(options, 'mip_max_nodes', 4.6e18)))
        st, msg = _mip_status(mo)
        return _build_result(mo.sol, st, msg, -1, mo.has_sol, n, mub, meq,
                             bub, beq, lb, ub, False, True, mo.nodes,
                             mo.bound)
    if time_limit <= 0.0:
        s0 = solve_lp(cv, A, rl, ru, lb, ub, 0)
        return _build_result(s0, 1, _msg_time(False), 0, False, n, mub, meq,
                             bub, beq, lb, ub, False, False, 0, 0.0)
    sol = solve_lp(cv, A, rl, ru, lb, ub, maxiter)
    if sol.status == LP_OPTIMAL:
        return _build_result(sol, 0, _MSG_OPT, sol.nit, True, n, mub, meq,
                             bub, beq, lb, ub, True, False, 0, 0.0)
    elif sol.status == LP_INFEASIBLE:
        return _build_result(sol, 2, _MSG_INF, sol.nit, False, n, mub, meq,
                             bub, beq, lb, ub, False, False, 0, 0.0)
    elif sol.status == LP_UNBOUNDED:
        return _build_result(sol, 3, _MSG_UNB, sol.nit, False, n, mub, meq,
                             bub, beq, lb, ub, False, False, 0, 0.0)
    else:
        return _build_result(sol, 1, _msg_iter(sol.feasible), sol.nit,
                             False, n, mub, meq, bub, beq, lb, ub, False,
                             False, 0, 0.0)

def _mip_status(mo: _MipOut) -> Tuple[int, str]:
    if mo.status == LP_OPTIMAL:
        return 0, _MSG_OPT
    elif mo.status == LP_INFEASIBLE:
        return 2, _MSG_INF
    elif mo.status == LP_UNBOUNDED:
        return 3, _MSG_UNB
    elif mo.status == LP_TIME_LIMIT:
        return 1, _msg_time(mo.has_sol)
    elif mo.status == 5:
        return 4, _msg_nodes(mo.has_sol)
    else:
        return 1, _msg_iter(mo.has_sol)

# ----------------------------------------------------------------- milp

def _add_constraint(con, rows: List[np.ndarray[float, 1]], rl: List[float],
                    ru: List[float], n: int):
    if isinstance(con, LinearConstraint):
        for i in range(con.A.shape[0]):
            rows.append(con.A[i].astype(float))
            rl.append(float(con.lb[i]))
            ru.append(float(con.ub[i]))
    else:
        lc = LinearConstraint(con[0], con[1], con[2])
        _add_constraint(lc, rows, rl, ru, n)

def milp(c, integrality = None, bounds = None, constraints = None,
         options = None):
    cv = _vec1(c)
    n = cv.size
    integ = np.zeros(n, bool)
    if not isinstance(integrality, NoneType):
        ia = np.broadcast_to(np.asarray(integrality, int), (n,))
        for j in range(n):
            if ia[j] < 0 or ia[j] > 3:
                raise ValueError("`integrality` must contain integers 0-3 and be broadcastable to `c.shape`.")
            if ia[j] == 2 or ia[j] == 3:
                raise ValueError("semi-continuous / semi-integer variables are not supported by this port")
            integ[j] = ia[j] == 1
    lb = np.zeros(n)
    ub = np.full(n, math.inf)
    if not isinstance(bounds, NoneType):
        if isinstance(bounds, Bounds):
            lb = np.broadcast_to(bounds.lb, (n,)).astype(float)
            ub = np.broadcast_to(bounds.ub, (n,)).astype(float)
        else:
            bb = Bounds(bounds[0], bounds[1])
            lb = np.broadcast_to(bb.lb, (n,)).astype(float)
            ub = np.broadcast_to(bb.ub, (n,)).astype(float)
    rows = List[np.ndarray[float, 1]]()
    rlo = List[float]()
    rup = List[float]()
    if isinstance(constraints, NoneType):
        pass
    elif isinstance(constraints, LinearConstraint):
        _add_constraint(constraints, rows, rlo, rup, n)
    elif isinstance(constraints, Tuple) and static.len(constraints) == 3 and not isinstance(constraints[0], LinearConstraint):
        _add_constraint(constraints, rows, rlo, rup, n)
    else:
        for con in constraints:
            _add_constraint(con, rows, rlo, rup, n)
    m = len(rows)
    A = np.zeros((m, n))
    for i in range(m):
        if rows[i].size != n:
            raise ValueError("The shape of `A` must be (len(lb), len(c)).")
        for j in range(n):
            A[i, j] = rows[i][j]
    rl = np.array(rlo, float) if m > 0 else np.zeros(0)
    ru = np.array(rup, float) if m > 0 else np.zeros(0)
    time_limit = _opt_num(options, 'time_limit', math.inf)
    node_limit = int(_opt_num(options, 'node_limit', 4.6e18))
    if node_limit <= 0 and time_limit > 0.0:
        return LinprogResult(_e(), math.nan, _e(), _e(), 4, _msg_nodes(False),
                             0, _Sens(_e(), _e()), _Sens(_e(), _e()),
                             _Sens(_e(), _e()), _Sens(_e(), _e()))
    if time_limit <= 0.0:
        return LinprogResult(_e(), math.nan, _e(), _e(), 1, _msg_time(False),
                             0, _Sens(_e(), _e()), _Sens(_e(), _e()),
                             _Sens(_e(), _e()), _Sens(_e(), _e()))
    mo = _solve_mip(cv, A, rl, ru, lb, ub, integ, 1 << 62, time_limit,
                    node_limit)
    st, msg = _mip_status(mo)
    if st == 3 and _any_int(integ):
        # HiGHS's MIP solver reports "unbounded or infeasible"
        st = 4
        msg = _MSG_UNB_INF
    res = _build_result(mo.sol, st, msg, -1, mo.has_sol, n, 0, 0, _e(), _e(),
                        lb, ub, False, True, mo.nodes, mo.bound)
    if mo.has_sol and st == 0:
        res.mip_dual_bound = res.fun
    return res

def _any_int(integ: np.ndarray[bool, 1]) -> bool:
    for v in integ:
        if v:
            return True
    return False
