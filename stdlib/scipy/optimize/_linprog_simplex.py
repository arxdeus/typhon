# Ported from scipy 1.18.1 (BSD-3-Clause)

# Dense bounded primal simplex used by linprog / milp in place of HiGHS.
#
# Problem form (the same one HiGHS receives from scipy):
#     min c @ x   s.t.  rl <= A @ x <= ru,   lo <= x <= hi
# Logical variables s = A @ x with bounds [rl, ru] are added so that the
# constraint matrix is [A, -I] with zero right-hand side; the initial basis
# is all-logical. A composite phase 1 (minimize the sum of bound violations
# of basic variables) is followed by phase 2 on the true costs. Bounded
# (box) variables, free variables and bound flips are handled directly.
# Dantzig pricing with Bland's rule after long degenerate runs prevents
# cycling. The basis inverse is kept explicitly (product-form updates with
# periodic refactorization).
#
# For problems with a unique optimum the returned x / fun / row activities
# equal HiGHS's; dual values are those of the final optimal basis (equal to
# HiGHS's when the dual solution is unique, i.e. the optimum is primal
# nondegenerate).

import math
import numpy as np

_FEAS_TOL = 1e-7
_OPT_TOL = 1e-7
_PIV_TOL = 1e-9

# internal status codes
LP_OPTIMAL = 0
LP_ITER_LIMIT = 1
LP_INFEASIBLE = 2
LP_UNBOUNDED = 3
LP_TIME_LIMIT = 4

class LPSolution:
    status: int
    x: np.ndarray[float, 1]
    act: np.ndarray[float, 1]       # row activities A @ x
    y: np.ndarray[float, 1]         # row duals (d fun / d row bound)
    d: np.ndarray[float, 1]         # column reduced costs
    at_hi: np.ndarray[bool, 1]      # nonbasic structural at its upper bound
    basic: np.ndarray[bool, 1]      # structural column is basic
    fun: float
    nit: int
    feasible: bool                  # primal feasible point at exit

    def __init__(self, status: int, x: np.ndarray[float, 1],
                 act: np.ndarray[float, 1], y: np.ndarray[float, 1],
                 d: np.ndarray[float, 1], at_hi: np.ndarray[bool, 1],
                 basic: np.ndarray[bool, 1], fun: float, nit: int,
                 feasible: bool):
        self.status = status
        self.x = x
        self.act = act
        self.y = y
        self.d = d
        self.at_hi = at_hi
        self.basic = basic
        self.fun = fun
        self.nit = nit
        self.feasible = feasible

class _Simplex:
    m: int
    n: int
    A: np.ndarray[float, 2]
    c: np.ndarray[float, 1]
    lo: np.ndarray[float, 1]        # n + m
    hi: np.ndarray[float, 1]
    val: np.ndarray[float, 1]       # values of all n + m variables
    basis: List[int]                # basic variable of each row
    pos: List[int]                  # row of a basic variable or -1
    Binv: np.ndarray[float, 2]
    since_refactor: int

    def __init__(self, c: np.ndarray[float, 1], A: np.ndarray[float, 2],
                 rl: np.ndarray[float, 1], ru: np.ndarray[float, 1],
                 lo: np.ndarray[float, 1], hi: np.ndarray[float, 1]):
        m = A.shape[0]
        n = A.shape[1]
        self.m = m
        self.n = n
        self.A = A
        self.c = c
        self.lo = np.concatenate((lo, rl))
        self.hi = np.concatenate((hi, ru))
        self.val = np.zeros(n + m)
        self.basis = [n + i for i in range(m)]
        self.pos = [-1] * (n + m)
        for i in range(m):
            self.pos[n + i] = i
        for j in range(n):
            if lo[j] > -math.inf:
                self.val[j] = lo[j]
            elif hi[j] < math.inf:
                self.val[j] = hi[j]
            else:
                self.val[j] = 0.0
        self.Binv = np.zeros((m, m))
        self.since_refactor = 0
        self.refactor()

    def col(self, j: int) -> np.ndarray[float, 1]:
        if j < self.n:
            return self.A[:, j].copy()
        e = np.zeros(self.m)
        e[j - self.n] = -1.0
        return e

    def cost(self, j: int) -> float:
        return self.c[j] if j < self.n else 0.0

    def refactor(self):
        m = self.m
        if m == 0:
            return
        B = np.empty((m, m))
        for i in range(m):
            cj = self.col(self.basis[i])
            for k in range(m):
                B[k, i] = cj[k]
        self.Binv = np.linalg.inv(B)
        self.since_refactor = 0
        self.recompute_basic()

    def recompute_basic(self):
        # xB = Binv @ (-(sum of nonbasic columns * values)) since [A, -I] v = 0
        m = self.m
        n = self.n
        r = np.zeros(m)
        for j in range(n):
            if self.pos[j] < 0 and self.val[j] != 0.0:
                for k in range(m):
                    r[k] -= self.A[k, j] * self.val[j]
        for i in range(m):
            j = n + i
            if self.pos[j] < 0 and self.val[j] != 0.0:
                r[i] += self.val[j]
        xb = self.Binv @ r
        for i in range(m):
            self.val[self.basis[i]] = xb[i]

    def infeas(self, j: int) -> int:
        v = self.val[j]
        if v < self.lo[j] - _FEAS_TOL:
            return -1
        if v > self.hi[j] + _FEAS_TOL:
            return 1
        return 0

    def duals(self, phase1: bool) -> np.ndarray[float, 1]:
        m = self.m
        cb = np.zeros(m)
        for i in range(m):
            j = self.basis[i]
            if phase1:
                cb[i] = float(self.infeas(j))
            else:
                cb[i] = self.cost(j)
        return cb @ self.Binv

    def reduced(self, j: int, y: np.ndarray[float, 1], phase1: bool) -> float:
        cj = 0.0 if phase1 else self.cost(j)
        if j < self.n:
            s = 0.0
            for k in range(self.m):
                s += y[k] * self.A[k, j]
            return cj - s
        return cj + y[j - self.n]

    def pivot(self, r: int, alpha: np.ndarray[float, 1], j: int):
        m = self.m
        piv = alpha[r]
        Bi = self.Binv
        for k in range(m):
            Bi[r, k] /= piv
        for i in range(m):
            if i != r:
                a = alpha[i]
                if a != 0.0:
                    for k in range(m):
                        Bi[i, k] -= a * Bi[r, k]
        old = self.basis[r]
        self.pos[old] = -1
        self.basis[r] = j
        self.pos[j] = r
        self.since_refactor += 1
        if self.since_refactor >= 50:
            self.refactor()

    def snap_nonbasic(self, j: int):
        # put a leaving variable exactly on its nearest finite bound
        v = self.val[j]
        lo = self.lo[j]
        hi = self.hi[j]
        if lo > -math.inf and hi < math.inf:
            self.val[j] = lo if abs(v - lo) <= abs(v - hi) else hi
        elif lo > -math.inf:
            self.val[j] = lo
        elif hi < math.inf:
            self.val[j] = hi
        else:
            self.val[j] = 0.0

    def iterate(self, phase1: bool, maxiter: int, nit: int) -> Tuple[int, int]:
        # returns (status, nit); status LP_OPTIMAL means "no improving
        # direction" (phase 1: caller checks feasibility)
        m = self.m
        nt = self.n + m
        degenerate = 0
        while True:
            if phase1:
                anyinf = False
                for i in range(m):
                    if self.infeas(self.basis[i]) != 0:
                        anyinf = True
                        break
                if not anyinf:
                    return LP_OPTIMAL, nit
            y = self.duals(phase1)
            bland = degenerate > 50
            enter = -1
            edir = 0
            best = 0.0
            for j in range(nt):
                if self.pos[j] >= 0:
                    continue
                lo = self.lo[j]
                hi = self.hi[j]
                if lo == hi:
                    continue
                dj = self.reduced(j, y, phase1)
                v = self.val[j]
                if dj < -_OPT_TOL and v < hi:
                    score = -dj
                    if enter < 0 or (not bland and score > best):
                        enter = j
                        edir = 1
                        best = score
                elif dj > _OPT_TOL and v > lo:
                    score = dj
                    if enter < 0 or (not bland and score > best):
                        enter = j
                        edir = -1
                        best = score
            if enter < 0:
                return LP_OPTIMAL, nit
            if nit >= maxiter:
                return LP_ITER_LIMIT, nit
            alpha = self.Binv @ self.col(enter)
            # ratio test
            tmax = self.hi[enter] - self.lo[enter]
            leave = -1
            for i in range(m):
                dlt = -alpha[i] * edir
                if abs(dlt) <= _PIV_TOL:
                    continue
                b = self.basis[i]
                v = self.val[b]
                lo = self.lo[b]
                hi = self.hi[b]
                t = math.inf
                if phase1 and v < lo - _FEAS_TOL:
                    if dlt > 0:
                        t = (lo - v) / dlt
                elif phase1 and v > hi + _FEAS_TOL:
                    if dlt < 0:
                        t = (hi - v) / dlt
                else:
                    if dlt > 0:
                        if hi < math.inf:
                            t = max(hi - v, 0.0) / dlt
                    else:
                        if lo > -math.inf:
                            t = max(v - lo, 0.0) / -dlt
                if t < tmax or (t == tmax and leave >= 0 and
                                (abs(dlt) > abs(alpha[leave]) if not bland
                                 else b < self.basis[leave])):
                    tmax = t
                    leave = i
            if tmax == math.inf:
                if phase1:
                    # cannot happen for a decreasing phase-1 direction
                    return LP_OPTIMAL, nit
                return LP_UNBOUNDED, nit
            nit += 1
            if tmax <= 1e-12:
                degenerate += 1
            else:
                degenerate = 0
            self.val[enter] += edir * tmax
            for i in range(m):
                self.val[self.basis[i]] -= alpha[i] * edir * tmax
            if leave < 0:
                # bound flip
                self.val[enter] = self.hi[enter] if edir > 0 else self.lo[enter]
                continue
            lb = self.basis[leave]
            self.pivot(leave, alpha, enter)
            self.snap_nonbasic(lb)
            if self.since_refactor == 0:
                continue
            # keep basic values consistent with the snapped variable
            if self.since_refactor % 10 == 0:
                self.recompute_basic()

def solve_lp(c: np.ndarray[float, 1], A: np.ndarray[float, 2],
             rl: np.ndarray[float, 1], ru: np.ndarray[float, 1],
             lo: np.ndarray[float, 1], hi: np.ndarray[float, 1],
             maxiter: int = 1 << 62) -> LPSolution:
    n = c.size
    m = A.shape[0]
    for j in range(n):
        if lo[j] > hi[j]:
            return _fail(LP_INFEASIBLE, n, m, 0)
    for i in range(m):
        if rl[i] > ru[i]:
            return _fail(LP_INFEASIBLE, n, m, 0)
    sx = _Simplex(c, A, rl, ru, lo, hi)
    st, nit = sx.iterate(True, maxiter, 0)
    sx.recompute_basic()
    if st == LP_ITER_LIMIT:
        return _result(sx, LP_ITER_LIMIT, nit, False)
    for i in range(m):
        if sx.infeas(sx.basis[i]) != 0:
            return _result(sx, LP_INFEASIBLE, nit, False)
    st, nit = sx.iterate(False, maxiter, nit)
    sx.refactor()
    if st == LP_OPTIMAL:
        # clean up drift after refactorization
        st2, nit = sx.iterate(False, maxiter, nit)
        if st2 != LP_OPTIMAL:
            st = st2
    return _result(sx, st, nit, True)

def _fail(status: int, n: int, m: int, nit: int) -> LPSolution:
    return LPSolution(status, np.full(n, math.nan), np.full(m, math.nan),
                      np.zeros(m), np.zeros(n), np.zeros(n, bool),
                      np.zeros(n, bool), math.nan, nit, False)

def _result(sx: _Simplex, status: int, nit: int, feasible: bool) -> LPSolution:
    n = sx.n
    m = sx.m
    x = sx.val[:n].copy()
    act = np.zeros(m)
    if m > 0:
        act = sx.A @ x
    y = sx.duals(False) if m > 0 else np.zeros(0, float)
    d = np.zeros(n)
    at_hi = np.zeros(n, bool)
    basic = np.zeros(n, bool)
    for j in range(n):
        if sx.pos[j] >= 0:
            basic[j] = True
        else:
            d[j] = sx.reduced(j, y, False)
            at_hi[j] = sx.hi[j] < math.inf and sx.val[j] == sx.hi[j] and \
                       not (sx.lo[j] == sx.hi[j] and d[j] > 0)
    fun = float(np.dot(sx.c, x))
    return LPSolution(status, x, act, y, d, at_hi, basic, fun, nit, feasible)
