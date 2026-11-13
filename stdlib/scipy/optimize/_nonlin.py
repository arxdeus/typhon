# Ported from scipy 1.18.1 (BSD-3-Clause)

# scipy.optimize._nonlin: Newton-Krylov / quasi-Newton nonlinear solvers
# (broyden1, broyden2, anderson, linearmixing, diagbroyden, excitingmixing,
# newton_krylov) and the df-sane spectral solver.
# Deviations: real 1-D problems only (x0 is flattened, the solution is a 1-D
# array), tol_norm is the max-norm, no preconditioner for newton_krylov
# (inner_M), no user eta_strategy / fnorm for df-sane.

import math
import numpy as np
from numpy.linalg import LinAlgError
from ._common import OptimizeResult, _e1, _e2, _norm2, _vec, INF
from ._linesearch import _wolfe1
from scipy.sparse.linalg import LinearOperator, lgmres, gmres, bicgstab, cgs, \
                                minres, tfqmr

class NoConvergence(Exception):
    x: np.ndarray[float, 1]

    def __init__(self, x: np.ndarray[float, 1] = _e1()):
        super().__init__('')
        self.x = x

def maxnorm(x: np.ndarray[float, 1]) -> float:
    m = 0.0
    for i in range(x.size):
        a = abs(x[i])
        if a > m or a != a:
            m = a
            if a != a:
                break
    return m

def _fdiv(a: float, b: float) -> float:
    # IEEE division (no ZeroDivisionError)
    if b == 0.0:
        if a == 0.0 or a != a:
            return math.nan
        neg = (a < 0) != (math.copysign(1.0, b) < 0)
        return -INF if neg else INF
    return a / b

def _safe_norm(v: np.ndarray[float, 1]) -> float:
    for i in range(v.size):
        if not math.isfinite(v[i]):
            return INF
    return _norm2(v)

def _vdot(a: np.ndarray[float, 1], b: np.ndarray[float, 1]) -> float:
    return float(np.dot(a, b))

class _Fun[F]:
    # flattened float view of the user function
    f: F

    def __init__(self, f: F):
        self.f = f

    def __call__(self, z: np.ndarray[float, 1]) -> np.ndarray[float, 1]:
        return np.atleast_1d(np.asarray(self.f(z.copy()), float)).flatten()

class _ArgsFun[F, A]:
    f: F
    args: A
    nfev: int

    def __init__(self, f: F, args: A):
        self.f = f
        self.args = args
        self.nfev = 0

    def __call__(self, z: np.ndarray[float, 1]):
        self.nfev += 1
        return self.f(z, *self.args)

# ---------------------------------------------------------------- Jacobians

def _autoscale(alpha: Optional[float], x0: np.ndarray[float, 1],
               f0: np.ndarray[float, 1]) -> float:
    if alpha is None:
        normf0 = _norm2(f0)
        if normf0 != 0.0:
            return 0.5 * max(_norm2(x0), 1.0) / normf0
        return 1.0
    return float(alpha)

class LowRankMatrix:
    alpha: float
    cs: List[np.ndarray[float, 1]]
    ds: List[np.ndarray[float, 1]]
    n: int
    collapsed: np.ndarray[float, 2]
    is_collapsed: bool

    def __init__(self, alpha: float, n: int):
        self.alpha = alpha
        self.cs = List[np.ndarray[float, 1]]()
        self.ds = List[np.ndarray[float, 1]]()
        self.n = n
        self.collapsed = _e2()
        self.is_collapsed = False

    def _matvec(self, v: np.ndarray[float, 1], alpha: float, cs, ds) -> np.ndarray[float, 1]:
        w = alpha * v
        for k in range(len(cs)):
            a = _vdot(ds[k], v)
            w = w + a * cs[k]
        return w

    def _solve(self, v: np.ndarray[float, 1], alpha: float, cs, ds) -> np.ndarray[float, 1]:
        m = len(cs)
        if m == 0:
            return v / alpha
        A = alpha * np.identity(m)
        for i in range(m):
            for j in range(m):
                A[i, j] += _vdot(ds[i], cs[j])
        q = np.zeros(m)
        for j in range(m):
            q[j] = _vdot(ds[j], v)
        q = q / alpha
        q = np.linalg.solve(A, q)
        w = v / alpha
        for j in range(m):
            w = w + (-q[j]) * cs[j]
        return w

    def matvec(self, v: np.ndarray[float, 1]) -> np.ndarray[float, 1]:
        if self.is_collapsed:
            return np.dot(self.collapsed, v)
        return self._matvec(v, self.alpha, self.cs, self.ds)

    def rmatvec(self, v: np.ndarray[float, 1]) -> np.ndarray[float, 1]:
        if self.is_collapsed:
            return np.dot(self.collapsed.T, v)
        return self._matvec(v, self.alpha, self.ds, self.cs)

    def solve(self, v: np.ndarray[float, 1]) -> np.ndarray[float, 1]:
        if self.is_collapsed:
            return np.linalg.solve(self.collapsed, v)
        return self._solve(v, self.alpha, self.cs, self.ds)

    def rsolve(self, v: np.ndarray[float, 1]) -> np.ndarray[float, 1]:
        if self.is_collapsed:
            return np.linalg.solve(self.collapsed.T, v)
        return self._solve(v, self.alpha, self.ds, self.cs)

    def todense(self) -> np.ndarray[float, 2]:
        if self.is_collapsed:
            return self.collapsed
        G = self.alpha * np.identity(self.n)
        for k in range(len(self.cs)):
            c = self.cs[k]
            d = self.ds[k]
            for i in range(self.n):
                for j in range(self.n):
                    G[i, j] += c[i] * d[j]
        return G

    def append(self, c: np.ndarray[float, 1], d: np.ndarray[float, 1]):
        if self.is_collapsed:
            for i in range(self.n):
                for j in range(self.n):
                    self.collapsed[i, j] += c[i] * d[j]
            return
        self.cs.append(c)
        self.ds.append(d)
        if len(self.cs) > c.size:
            self.collapse()

    def collapse(self):
        self.collapsed = self.todense()
        self.is_collapsed = True
        self.cs = List[np.ndarray[float, 1]]()
        self.ds = List[np.ndarray[float, 1]]()
        self.alpha = 0.0

    def restart_reduce(self, rank: int):
        if self.is_collapsed:
            return
        if len(self.cs) > rank:
            self.cs = List[np.ndarray[float, 1]]()
            self.ds = List[np.ndarray[float, 1]]()

    def simple_reduce(self, rank: int):
        if self.is_collapsed:
            return
        while len(self.cs) > rank:
            self.cs.pop(0)
            self.ds.pop(0)

    def svd_reduce(self, max_rank: int, to_retain: Optional[int] = None):
        if self.is_collapsed:
            return
        p = max_rank
        q = p - 2
        if to_retain is not None:
            q = to_retain
        if len(self.cs) > 0:
            p = min(p, self.cs[0].size)
        q = max(0, min(q, p - 1))
        m = len(self.cs)
        if m < p:
            return
        n = self.n
        C = np.zeros((n, m))
        D = np.zeros((n, m))
        for k in range(m):
            for i in range(n):
                C[i, k] = self.cs[k][i]
                D[i, k] = self.ds[k][i]
        Q, R = np.linalg.qr(D, mode='reduced')
        D = Q
        C = np.dot(C, R.T)
        U, S, WH = np.linalg.svd(C, full_matrices=False)
        C = np.dot(C, np.linalg.inv(WH))
        D = np.dot(D, WH.T)
        for k in range(q):
            self.cs[k] = C[:, k].copy()
            self.ds[k] = D[:, k].copy()
        while len(self.cs) > q:
            self.cs.pop()
            self.ds.pop()

class BroydenFirst:
    alpha: Optional[float]
    a: float
    reduction_method: str
    max_rank: int
    to_retain: Optional[int]
    second: bool
    Gm: LowRankMatrix
    last_x: np.ndarray[float, 1]
    last_f: np.ndarray[float, 1]
    n: int

    def __init__(self, alpha: Optional[float] = None,
                 reduction_method: str = 'restart',
                 max_rank: Optional[int] = None,
                 to_retain: Optional[int] = None):
        self.alpha = alpha
        self.a = 0.0
        self.max_rank = (1 << 60) if max_rank is None else max_rank
        if reduction_method != 'svd' and reduction_method != 'simple' and \
           reduction_method != 'restart':
            raise ValueError(f"Unknown rank reduction method '{reduction_method}'")
        self.reduction_method = reduction_method
        self.to_retain = to_retain
        self.second = False
        self.Gm = LowRankMatrix(0.0, 0)
        self.last_x = _e1()
        self.last_f = _e1()
        self.n = 0

    def _reduce(self):
        if self.reduction_method == 'svd':
            self.Gm.svd_reduce(self.max_rank - 1, self.to_retain)
        elif self.reduction_method == 'simple':
            self.Gm.simple_reduce(self.max_rank - 1)
        else:
            self.Gm.restart_reduce(self.max_rank - 1)

    def setup(self, x: np.ndarray[float, 1], F: np.ndarray[float, 1]):
        self.last_f = F
        self.last_x = x
        self.n = F.size
        self.a = _autoscale(self.alpha, x, F)
        self.alpha = self.a
        self.Gm = LowRankMatrix(-self.a, F.size)

    def todense(self) -> np.ndarray[float, 2]:
        return np.linalg.inv(self.Gm.todense())

    def solve(self, f: np.ndarray[float, 1], tol: float, func) -> np.ndarray[float, 1]:
        r = self.Gm.matvec(f)
        for i in range(r.size):
            if not math.isfinite(r[i]):
                self.setup(self.last_x, self.last_f)
                return self.Gm.matvec(f)
        return r

    def matvec(self, f: np.ndarray[float, 1]) -> np.ndarray[float, 1]:
        return self.Gm.solve(f)

    def rsolve(self, f: np.ndarray[float, 1]) -> np.ndarray[float, 1]:
        return self.Gm.rmatvec(f)

    def rmatvec(self, f: np.ndarray[float, 1]) -> np.ndarray[float, 1]:
        return self.Gm.rsolve(f)

    def update(self, x: np.ndarray[float, 1], f: np.ndarray[float, 1]):
        df = f - self.last_f
        dx = x - self.last_x
        self._reduce()
        if self.second:
            dfn = _norm2(df)
            c = dx - self.Gm.matvec(df)
            d = df / (dfn * dfn)
            self.Gm.append(c, d)
        else:
            v = self.Gm.rmatvec(dx)
            c = dx - self.Gm.matvec(df)
            d = v / _vdot(df, v)
            self.Gm.append(c, d)
        self.last_f = f
        self.last_x = x

class BroydenSecond(BroydenFirst):
    def __init__(self, alpha: Optional[float] = None,
                 reduction_method: str = 'restart',
                 max_rank: Optional[int] = None,
                 to_retain: Optional[int] = None):
        super().__init__(alpha, reduction_method, max_rank, to_retain)
        self.second = True

class Anderson:
    alpha: Optional[float]
    a_: float
    w0: float
    M: int
    dxs: List[np.ndarray[float, 1]]
    dfs: List[np.ndarray[float, 1]]
    a: np.ndarray[float, 2]
    last_x: np.ndarray[float, 1]
    last_f: np.ndarray[float, 1]

    def __init__(self, alpha: Optional[float] = None, w0: float = 0.01, M: int = 5):
        self.alpha = alpha
        self.a_ = 0.0
        self.M = M
        self.w0 = w0
        self.dxs = List[np.ndarray[float, 1]]()
        self.dfs = List[np.ndarray[float, 1]]()
        self.a = _e2()
        self.last_x = _e1()
        self.last_f = _e1()

    def setup(self, x: np.ndarray[float, 1], F: np.ndarray[float, 1]):
        self.last_f = F
        self.last_x = x
        self.a_ = _autoscale(self.alpha, x, F)
        self.alpha = self.a_

    def solve(self, f: np.ndarray[float, 1], tol: float, func) -> np.ndarray[float, 1]:
        dx = (-self.a_) * f
        n = len(self.dxs)
        if n == 0:
            return dx
        df_f = np.empty(n)
        for k in range(n):
            df_f[k] = _vdot(self.dfs[k], f)
        gamma = np.zeros(n)
        try:
            gamma = np.linalg.solve(self.a, df_f)
        except LinAlgError:
            self.dxs = List[np.ndarray[float, 1]]()
            self.dfs = List[np.ndarray[float, 1]]()
            return dx
        for m in range(n):
            dx = dx + gamma[m] * (self.dxs[m] + self.a_ * self.dfs[m])
        return dx

    def matvec(self, f: np.ndarray[float, 1]) -> np.ndarray[float, 1]:
        dx = (-f) / self.a_
        n = len(self.dxs)
        if n == 0:
            return dx
        df_f = np.empty(n)
        for k in range(n):
            df_f[k] = _vdot(self.dfs[k], f)
        b = np.empty((n, n))
        for i in range(n):
            for j in range(n):
                b[i, j] = _vdot(self.dfs[i], self.dxs[j])
                if i == j and self.w0 != 0:
                    b[i, j] -= _vdot(self.dfs[i], self.dfs[i]) * self.w0 ** 2 * self.a_
        gamma = np.linalg.solve(b, df_f)
        for m in range(n):
            dx = dx + gamma[m] * (self.dfs[m] + self.dxs[m] / self.a_)
        return dx

    def update(self, x: np.ndarray[float, 1], f: np.ndarray[float, 1]):
        df = f - self.last_f
        dx = x - self.last_x
        self.last_f = f
        self.last_x = x
        if self.M == 0:
            return
        self.dxs.append(dx)
        self.dfs.append(df)
        while len(self.dxs) > self.M:
            self.dxs.pop(0)
            self.dfs.pop(0)
        n = len(self.dxs)
        a = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                wd = self.w0 ** 2 if i == j else 0.0
                a[i, j] = (1 + wd) * _vdot(self.dfs[i], self.dfs[j])
        for i in range(n):
            for j in range(i + 1, n):
                a[j, i] = a[i, j]
        self.a = a

class DiagBroyden:
    alpha: Optional[float]
    d: np.ndarray[float, 1]
    last_x: np.ndarray[float, 1]
    last_f: np.ndarray[float, 1]

    def __init__(self, alpha: Optional[float] = None):
        self.alpha = alpha
        self.d = _e1()
        self.last_x = _e1()
        self.last_f = _e1()

    def setup(self, x: np.ndarray[float, 1], F: np.ndarray[float, 1]):
        self.last_f = F
        self.last_x = x
        a = _autoscale(self.alpha, x, F)
        self.alpha = a
        self.d = np.full(F.size, 1.0 / a)

    def solve(self, f: np.ndarray[float, 1], tol: float, func) -> np.ndarray[float, 1]:
        return (-f) / self.d

    def matvec(self, f: np.ndarray[float, 1]) -> np.ndarray[float, 1]:
        return (-f) * self.d

    def todense(self) -> np.ndarray[float, 2]:
        return np.diag(-self.d)

    def update(self, x: np.ndarray[float, 1], f: np.ndarray[float, 1]):
        df = f - self.last_f
        dx = x - self.last_x
        dxn = _norm2(dx)
        self.d = self.d - (df + self.d * dx) * dx / (dxn * dxn)
        self.last_f = f
        self.last_x = x

class LinearMixing:
    alpha: Optional[float]
    a: float
    n: int
    last_x: np.ndarray[float, 1]
    last_f: np.ndarray[float, 1]

    def __init__(self, alpha: Optional[float] = None):
        self.alpha = alpha
        self.a = 0.0
        self.n = 0
        self.last_x = _e1()
        self.last_f = _e1()

    def setup(self, x: np.ndarray[float, 1], F: np.ndarray[float, 1]):
        self.last_f = F
        self.last_x = x
        self.a = _autoscale(self.alpha, x, F)
        self.alpha = self.a
        self.n = F.size

    def solve(self, f: np.ndarray[float, 1], tol: float, func) -> np.ndarray[float, 1]:
        return (-f) * self.a

    def matvec(self, f: np.ndarray[float, 1]) -> np.ndarray[float, 1]:
        return (-f) / self.a

    def todense(self) -> np.ndarray[float, 2]:
        return np.diag(np.full(self.n, -1.0 / self.a))

    def update(self, x: np.ndarray[float, 1], f: np.ndarray[float, 1]):
        self.last_f = f
        self.last_x = x

class ExcitingMixing:
    alpha: Optional[float]
    a: float
    alphamax: float
    beta: np.ndarray[float, 1]
    last_x: np.ndarray[float, 1]
    last_f: np.ndarray[float, 1]

    def __init__(self, alpha: Optional[float] = None, alphamax: float = 1.0):
        self.alpha = alpha
        self.a = 0.0
        self.alphamax = alphamax
        self.beta = _e1()
        self.last_x = _e1()
        self.last_f = _e1()

    def setup(self, x: np.ndarray[float, 1], F: np.ndarray[float, 1]):
        self.last_f = F
        self.last_x = x
        self.a = _autoscale(self.alpha, x, F)
        self.alpha = self.a
        self.beta = np.full(F.size, self.a)

    def solve(self, f: np.ndarray[float, 1], tol: float, func) -> np.ndarray[float, 1]:
        return (-f) * self.beta

    def matvec(self, f: np.ndarray[float, 1]) -> np.ndarray[float, 1]:
        return (-f) / self.beta

    def todense(self) -> np.ndarray[float, 2]:
        return np.diag(-1.0 / self.beta)

    def update(self, x: np.ndarray[float, 1], f: np.ndarray[float, 1]):
        for i in range(f.size):
            if f[i] * self.last_f[i] > 0:
                self.beta[i] += self.a
            else:
                self.beta[i] = self.a
            if self.beta[i] < 0.0:
                self.beta[i] = 0.0
            elif self.beta[i] > self.alphamax:
                self.beta[i] = self.alphamax
        self.last_f = f
        self.last_x = x

class _KrylovMV[FN]:
    func: FN
    x0: np.ndarray[float, 1]
    f0: np.ndarray[float, 1]
    omega: float

    def __init__(self, func: FN, x0: np.ndarray[float, 1], f0: np.ndarray[float, 1],
                 omega: float):
        self.func = func
        self.x0 = x0
        self.f0 = f0
        self.omega = omega

    def __call__(self, v: np.ndarray[float, 1]) -> np.ndarray[float, 1]:
        nv = _norm2(v)
        if nv == 0:
            return 0.0 * v
        sc = self.omega / nv
        r = (self.func(self.x0 + sc * v) - self.f0) / sc
        okr = True
        for i in range(r.size):
            if not math.isfinite(r[i]):
                okr = False
        if not okr:
            okv = True
            for i in range(v.size):
                if not math.isfinite(v[i]):
                    okv = False
            if okv:
                raise ValueError('Function returned non-finite results')
        return r

class KrylovJacobian:
    rdiff: Optional[float]
    rd: float
    method: str
    inner_maxiter: int
    outer_k: int
    x0: np.ndarray[float, 1]
    f0: np.ndarray[float, 1]
    omega: float
    outer_v: List[Tuple[np.ndarray[float, 1], np.ndarray[float, 1], bool]]

    def __init__(self, rdiff: Optional[float] = None, method: str = 'lgmres',
                 inner_maxiter: int = 20, outer_k: int = 10):
        self.rdiff = rdiff
        self.rd = 0.0
        self.method = method
        self.inner_maxiter = inner_maxiter
        self.outer_k = outer_k
        self.x0 = _e1()
        self.f0 = _e1()
        self.omega = 0.0
        self.outer_v = List[Tuple[np.ndarray[float, 1], np.ndarray[float, 1], bool]]()
        if method != 'lgmres' and method != 'gmres' and method != 'bicgstab' and \
           method != 'cgs' and method != 'minres' and method != 'tfqmr':
            raise ValueError(f'Unknown inner method {method}')

    def _update_diff_step(self):
        mx = maxnorm(self.x0)
        mf = maxnorm(self.f0)
        self.omega = self.rd * max(1.0, mx) / max(1.0, mf)

    def setup(self, x: np.ndarray[float, 1], F: np.ndarray[float, 1]):
        self.x0 = x
        self.f0 = F
        if self.rdiff is None:
            self.rd = math.sqrt(2.220446049250313e-16)
        else:
            self.rd = float(self.rdiff)
        self._update_diff_step()

    def update(self, x: np.ndarray[float, 1], f: np.ndarray[float, 1]):
        self.x0 = x
        self.f0 = f
        self._update_diff_step()

    def solve(self, rhs: np.ndarray[float, 1], tol: float, func) -> np.ndarray[float, 1]:
        n = rhs.size
        mv = _KrylovMV(func, self.x0, self.f0, self.omega)
        op = LinearOperator((n, n), mv, dtype=float)
        m = self.method
        if m == 'lgmres':
            sol, info = lgmres(op, rhs, rtol=tol, atol=0.0, maxiter=1,
                               outer_k=self.outer_k, outer_v=self.outer_v,
                               prepend_outer_v=True, store_outer_Av=False)
            return sol
        elif m == 'gmres':
            sol, info = gmres(op, rhs, rtol=tol, atol=0.0,
                              restart=self.inner_maxiter, maxiter=1)
            return sol
        elif m == 'bicgstab':
            sol, info = bicgstab(op, rhs, rtol=tol, atol=0.0,
                                 maxiter=self.inner_maxiter)
            return sol
        elif m == 'cgs':
            sol, info = cgs(op, rhs, rtol=tol, atol=0.0,
                            maxiter=self.inner_maxiter)
            return sol
        elif m == 'minres':
            sol, info = minres(op, rhs, rtol=tol, maxiter=self.inner_maxiter)
            return sol
        else:
            sol, info = tfqmr(op, rhs, rtol=tol, maxiter=self.inner_maxiter)
            return sol

# ---------------------------------------------------------------- solver

class _TermCond:
    x_tol: float
    x_rtol: float
    f_tol: float
    f_rtol: float
    it: int
    has_iter: bool
    f0_norm: float
    has_f0: bool
    iteration: int

    def __init__(self, f_tol: Optional[float], f_rtol: Optional[float],
                 x_tol: Optional[float], x_rtol: Optional[float],
                 it: Optional[int]):
        self.f_tol = math.pow(2.220446049250313e-16, 1.0 / 3) if f_tol is None else f_tol
        self.f_rtol = INF if f_rtol is None else f_rtol
        self.x_tol = INF if x_tol is None else x_tol
        self.x_rtol = INF if x_rtol is None else x_rtol
        self.has_iter = it is not None
        self.it = 0 if it is None else it
        self.f0_norm = 0.0
        self.has_f0 = False
        self.iteration = 0

    def check(self, f: np.ndarray[float, 1], x: np.ndarray[float, 1],
              dx: np.ndarray[float, 1]) -> int:
        self.iteration += 1
        f_norm = maxnorm(f)
        x_norm = maxnorm(x)
        dx_norm = maxnorm(dx)
        if not self.has_f0:
            self.f0_norm = f_norm
            self.has_f0 = True
        if f_norm == 0:
            return 1
        if self.has_iter:
            return 2 if self.iteration > self.it else 0
        if f_norm <= self.f_tol and _fdiv(f_norm, self.f_rtol) <= self.f0_norm and \
           dx_norm <= self.x_tol and _fdiv(dx_norm, self.x_rtol) <= x_norm:
            return 1
        return 0

class _ArmijoLS[FN]:
    func: FN
    x: np.ndarray[float, 1]
    dx: np.ndarray[float, 1]
    tmp_s: float
    tmp_Fx: np.ndarray[float, 1]
    tmp_phi: float
    s_norm: float
    rdiff: float

    def __init__(self, func: FN, x: np.ndarray[float, 1], Fx: np.ndarray[float, 1],
                 dx: np.ndarray[float, 1]):
        self.func = func
        self.x = x
        self.dx = dx
        self.tmp_s = 0.0
        self.tmp_Fx = Fx
        n0 = _norm2(Fx)
        self.tmp_phi = n0 * n0
        self.s_norm = _fdiv(_norm2(x), _norm2(dx))
        self.rdiff = 1e-8

    def phi(self, s: float, store: bool = True) -> float:
        if s == self.tmp_s:
            return self.tmp_phi
        xt = self.x + s * self.dx
        v = self.func(xt)
        nv = _safe_norm(v)
        p = nv * nv
        if store:
            self.tmp_s = s
            self.tmp_phi = p
            self.tmp_Fx = v
        return p

    def derphi(self, s: float) -> float:
        ds = (abs(s) + self.s_norm + 1) * self.rdiff
        return (self.phi(s + ds, False) - self.phi(s)) / ds

def _scalar_search_armijo(ls, phi0: float, derphi0: float, c1: float,
                          alpha0: float, amin: float):
    # returns (ok, alpha, phi1)
    phi_a0 = ls.phi(alpha0)
    if phi_a0 <= phi0 + c1 * alpha0 * derphi0:
        return True, alpha0, phi_a0
    alpha1 = -(derphi0) * alpha0 ** 2 / 2.0 / (phi_a0 - phi0 - derphi0 * alpha0)
    phi_a1 = ls.phi(alpha1)
    if phi_a1 <= phi0 + c1 * alpha1 * derphi0:
        return True, alpha1, phi_a1
    while alpha1 > amin:
        factor = alpha0 ** 2 * alpha1 ** 2 * (alpha1 - alpha0)
        a = alpha0 ** 2 * (phi_a1 - phi0 - derphi0 * alpha1) - \
            alpha1 ** 2 * (phi_a0 - phi0 - derphi0 * alpha0)
        a = a / factor
        b = -alpha0 ** 3 * (phi_a1 - phi0 - derphi0 * alpha1) + \
            alpha1 ** 3 * (phi_a0 - phi0 - derphi0 * alpha0)
        b = b / factor
        alpha2 = (-b + math.sqrt(abs(b ** 2 - 3 * a * derphi0))) / (3.0 * a)
        phi_a2 = ls.phi(alpha2)
        if phi_a2 <= phi0 + c1 * alpha2 * derphi0:
            return True, alpha2, phi_a2
        if (alpha1 - alpha2) > alpha1 / 2.0 or (1 - alpha2 / alpha1) < 0.96:
            alpha2 = alpha1 / 2.0
        alpha0 = alpha1
        alpha1 = alpha2
        phi_a0 = phi_a1
        phi_a1 = phi_a2
    return False, 0.0, phi_a1

def _nonlin_line_search(func, x: np.ndarray[float, 1], Fx: np.ndarray[float, 1],
                        dx: np.ndarray[float, 1], search_type: str,
                        smin: float = 1e-2):
    ls = _ArmijoLS(func, x, Fx, dx)
    ok = False
    s = 1.0
    if search_type == 'wolfe':
        ph0 = ls.tmp_phi
        dph0 = ls.derphi(0.0)
        ok, s, phi1, phi0 = _wolfe1(ls, ph0, math.nan, dph0, 1e-4, 0.9, 50.0,
                                    smin, 1e-2)
    else:
        ok, s, phi1 = _scalar_search_armijo(ls, ls.tmp_phi, -ls.tmp_phi, 1e-4,
                                            1.0, smin)
    if not ok:
        s = 1.0
    xn = x + s * dx
    Fn = Fx
    if s == ls.tmp_s:
        Fn = ls.tmp_Fx
    else:
        Fn = func(xn)
    return s, xn, Fn, _norm2(Fn)

def nonlin_solve(F, x0, jacobian, it: Optional[int] = None,
                 verbose: bool = False, maxiter: Optional[int] = None,
                 f_tol: Optional[float] = None, f_rtol: Optional[float] = None,
                 x_tol: Optional[float] = None, x_rtol: Optional[float] = None,
                 line_search: Optional[str] = 'armijo', callback = None,
                 raise_exception: bool = True):
    # returns (x, Fx, status, nit); `jacobian` is one of the Jacobian classes
    func = _Fun(F)
    condition = _TermCond(f_tol, f_rtol, x_tol, x_rtol, it)
    x = _vec(x0)
    dx = np.full(x.size, INF)
    Fx = func(x)
    Fx_norm = _norm2(Fx)
    jacobian.setup(x.copy(), Fx)
    mi = 0
    if maxiter is None:
        if it is not None:
            mi = it + 1
        else:
            mi = 100 * (x.size + 1)
    else:
        mi = maxiter
    if line_search is not None and line_search != 'armijo' and line_search != 'wolfe':
        raise ValueError('Invalid line search')
    gamma = 0.9
    eta_max = 0.9999
    eta_treshold = 0.1
    eta = 1e-3
    status = 0
    for n in range(mi):
        status = condition.check(Fx, x, dx)
        if status != 0:
            break
        tol = min(eta, eta * Fx_norm)
        dx = -jacobian.solve(Fx, tol, func)
        if _norm2(dx) == 0:
            raise ValueError('Jacobian inversion yielded zero vector. '
                             'This indicates a bug in the Jacobian '
                             'approximation.')
        s = 1.0
        Fx_norm_new = 0.0
        if line_search is not None:
            s, x, Fx, Fx_norm_new = _nonlin_line_search(func, x, Fx, dx, line_search)
        else:
            x = x + dx
            Fx = func(x)
            Fx_norm_new = _norm2(Fx)
        jacobian.update(x.copy(), Fx)
        if callback is not None:
            callback(x, Fx)
        eta_A = gamma * Fx_norm_new ** 2 / Fx_norm ** 2
        if gamma * eta ** 2 < eta_treshold:
            eta = min(eta_max, eta_A)
        else:
            eta = min(eta_max, max(eta_A, gamma * eta ** 2))
        Fx_norm = Fx_norm_new
        if verbose:
            print(f'{n}:  |F(x)| = {maxnorm(Fx):g}; step {s:g}')
    else:
        if raise_exception:
            raise NoConvergence(x)
        else:
            status = 2
    return x, Fx, status, condition.iteration

def _nl_message(status: int) -> str:
    if status == 1:
        return 'A solution was found at the specified tolerance.'
    return 'The maximum number of iterations allowed has been reached.'

def _nl_result(x, Fx, status: int, nit: int, meth: str, nfev: int = 0):
    return OptimizeResult(x, Fx, _e2(), nit=nit, nfev=nfev, status=status,
                          success=status == 1, message=_nl_message(status),
                          method=meth)

def _nl_wrap(F, jac_obj, xin, it, verbose, maxiter, f_tol, f_rtol, x_tol, x_rtol,
             line_search, callback):
    x, Fx, status, nit = nonlin_solve(F, xin, jac_obj, it, verbose, maxiter,
                                      f_tol, f_rtol, x_tol, x_rtol, line_search,
                                      callback, True)
    return x

def broyden1(F, xin, iter: Optional[int] = None, alpha: Optional[float] = None,
             reduction_method: str = 'restart', max_rank: Optional[int] = None,
             verbose: bool = False, maxiter: Optional[int] = None,
             f_tol: Optional[float] = None, f_rtol: Optional[float] = None,
             x_tol: Optional[float] = None, x_rtol: Optional[float] = None,
             line_search: Optional[str] = 'armijo', callback = None):
    return _nl_wrap(F, BroydenFirst(alpha, reduction_method, max_rank), xin, iter,
                    verbose, maxiter, f_tol, f_rtol, x_tol, x_rtol, line_search,
                    callback)

def broyden2(F, xin, iter: Optional[int] = None, alpha: Optional[float] = None,
             reduction_method: str = 'restart', max_rank: Optional[int] = None,
             verbose: bool = False, maxiter: Optional[int] = None,
             f_tol: Optional[float] = None, f_rtol: Optional[float] = None,
             x_tol: Optional[float] = None, x_rtol: Optional[float] = None,
             line_search: Optional[str] = 'armijo', callback = None):
    return _nl_wrap(F, BroydenSecond(alpha, reduction_method, max_rank), xin, iter,
                    verbose, maxiter, f_tol, f_rtol, x_tol, x_rtol, line_search,
                    callback)

def anderson(F, xin, iter: Optional[int] = None, alpha: Optional[float] = None,
             w0: float = 0.01, M: int = 5, verbose: bool = False,
             maxiter: Optional[int] = None, f_tol: Optional[float] = None,
             f_rtol: Optional[float] = None, x_tol: Optional[float] = None,
             x_rtol: Optional[float] = None,
             line_search: Optional[str] = 'armijo', callback = None):
    return _nl_wrap(F, Anderson(alpha, w0, M), xin, iter, verbose, maxiter,
                    f_tol, f_rtol, x_tol, x_rtol, line_search, callback)

def linearmixing(F, xin, iter: Optional[int] = None, alpha: Optional[float] = None,
                 verbose: bool = False, maxiter: Optional[int] = None,
                 f_tol: Optional[float] = None, f_rtol: Optional[float] = None,
                 x_tol: Optional[float] = None, x_rtol: Optional[float] = None,
                 line_search: Optional[str] = 'armijo', callback = None):
    return _nl_wrap(F, LinearMixing(alpha), xin, iter, verbose, maxiter, f_tol,
                    f_rtol, x_tol, x_rtol, line_search, callback)

def diagbroyden(F, xin, iter: Optional[int] = None, alpha: Optional[float] = None,
                verbose: bool = False, maxiter: Optional[int] = None,
                f_tol: Optional[float] = None, f_rtol: Optional[float] = None,
                x_tol: Optional[float] = None, x_rtol: Optional[float] = None,
                line_search: Optional[str] = 'armijo', callback = None):
    return _nl_wrap(F, DiagBroyden(alpha), xin, iter, verbose, maxiter, f_tol,
                    f_rtol, x_tol, x_rtol, line_search, callback)

def excitingmixing(F, xin, iter: Optional[int] = None, alpha: Optional[float] = None,
                   alphamax: float = 1.0, verbose: bool = False,
                   maxiter: Optional[int] = None, f_tol: Optional[float] = None,
                   f_rtol: Optional[float] = None, x_tol: Optional[float] = None,
                   x_rtol: Optional[float] = None,
                   line_search: Optional[str] = 'armijo', callback = None):
    return _nl_wrap(F, ExcitingMixing(alpha, alphamax), xin, iter, verbose, maxiter,
                    f_tol, f_rtol, x_tol, x_rtol, line_search, callback)

def newton_krylov(F, xin, iter: Optional[int] = None, rdiff: Optional[float] = None,
                  method: str = 'lgmres', inner_maxiter: int = 20,
                  outer_k: int = 10, verbose: bool = False,
                  maxiter: Optional[int] = None, f_tol: Optional[float] = None,
                  f_rtol: Optional[float] = None, x_tol: Optional[float] = None,
                  x_rtol: Optional[float] = None,
                  line_search: Optional[str] = 'armijo', callback = None):
    return _nl_wrap(F, KrylovJacobian(rdiff, method, inner_maxiter, outer_k), xin,
                    iter, verbose, maxiter, f_tol, f_rtol, x_tol, x_rtol,
                    line_search, callback)
