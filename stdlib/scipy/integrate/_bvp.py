# Ported from scipy 1.18.1 (BSD-3-Clause)

# solve_bvp: boundary value problem solver (4th order collocation with the
# residual control of Kierzenka & Shampine).
# Deviations: real-valued problems only; `sol` is a piecewise cubic callable
# (BVPSol) with the PPoly evaluation `sol(x, nu=0)` returning an (n, len(x))
# array; `p` in the result is an empty array when there are no unknown
# parameters (scipy returns None); `verbose` output is not printed.

import math
import numpy as np
from scipy.sparse import csc_array
from scipy.sparse.linalg import splu
from scipy.linalg import pinv

EPS = 2.220446049250313e-16

TERMINATION_MESSAGES = [
    'The algorithm converged to the desired accuracy.',
    'The maximum number of mesh nodes is exceeded.',
    'A singular Jacobian encountered when solving the collocation system.',
    'The solver was unable to satisfy boundary conditions tolerance on iteration 10.',
]

class BVPSol:
    # piecewise cubic with coefficients c[4, n, m-1] on the mesh x
    c: np.ndarray[float, 3]
    x: np.ndarray[float, 1]

    def __init__(self, c: np.ndarray[float, 3], x: np.ndarray[float, 1]):
        self.c = c
        self.x = x

    def __call__(self, xq, nu: int = 0) -> np.ndarray[float, 2]:
        xs = np.atleast_1d(np.asarray(xq, float)).flatten()
        n = self.c.shape[1]
        m1 = self.c.shape[2]
        out = np.empty((n, xs.size))
        for q in range(xs.size):
            v = xs[q]
            idx = int(np.searchsorted(self.x, v, side='right')) - 1
            if idx < 0:
                idx = 0
            if idx > m1 - 1:
                idx = m1 - 1
            s = v - self.x[idx]
            for a in range(n):
                c0 = self.c[0, a, idx]
                c1 = self.c[1, a, idx]
                c2 = self.c[2, a, idx]
                c3 = self.c[3, a, idx]
                if nu == 0:
                    out[a, q] = ((c0 * s + c1) * s + c2) * s + c3
                elif nu == 1:
                    out[a, q] = (3 * c0 * s + 2 * c1) * s + c2
                elif nu == 2:
                    out[a, q] = 6 * c0 * s + 2 * c1
                elif nu == 3:
                    out[a, q] = 6 * c0
                else:
                    out[a, q] = 0.0
        return out

class BVPResult:
    sol: BVPSol
    p: np.ndarray[float, 1]
    x: np.ndarray[float, 1]
    y: np.ndarray[float, 2]
    yp: np.ndarray[float, 2]
    rms_residuals: np.ndarray[float, 1]
    niter: int
    status: int
    message: str
    success: bool

    def __init__(self, sol: BVPSol, p: np.ndarray[float, 1], x: np.ndarray[float, 1],
                 y: np.ndarray[float, 2], yp: np.ndarray[float, 2],
                 rms: np.ndarray[float, 1], niter: int, status: int):
        self.sol = sol
        self.p = p
        self.x = x
        self.y = y
        self.yp = yp
        self.rms_residuals = rms
        self.niter = niter
        self.status = status
        self.message = TERMINATION_MESSAGES[status]
        self.success = status == 0

    def __getitem__(self, key: Literal[str]):
        return getattr(self, key)

# ------------------------------------------------------------ user wrappers

class _FunW[F, K: Static[int]]:
    f: F

    def __init__(self, f: F):
        self.f = f

    def __call__(self, x: np.ndarray[float, 1], y: np.ndarray[float, 2],
                 p: np.ndarray[float, 1]) -> np.ndarray[float, 2]:
        if K == 0:
            return np.atleast_2d(np.asarray(self.f(x, y), float))
        else:
            return np.atleast_2d(np.asarray(self.f(x, y, p), float))

class _BcW[F, K: Static[int]]:
    f: F

    def __init__(self, f: F):
        self.f = f

    def __call__(self, ya: np.ndarray[float, 1], yb: np.ndarray[float, 1],
                 p: np.ndarray[float, 1]) -> np.ndarray[float, 1]:
        if K == 0:
            return np.atleast_1d(np.asarray(self.f(ya, yb), float)).flatten()
        else:
            return np.atleast_1d(np.asarray(self.f(ya, yb, p), float)).flatten()

class _FunJacW[F, K: Static[int]]:
    f: F

    def __init__(self, f: F):
        self.f = f

    def __call__(self, x: np.ndarray[float, 1], y: np.ndarray[float, 2],
                 p: np.ndarray[float, 1]):
        # returns (df_dy[n, n, m], df_dp[n, k, m])
        if K == 0:
            r = np.asarray(self.f(x, y), float)
            return r, np.zeros((r.shape[0], 0, r.shape[2]))
        else:
            r = self.f(x, y, p)
            return np.asarray(r[0], float), np.asarray(r[1], float)

class _BcJacW[F, K: Static[int]]:
    f: F

    def __init__(self, f: F):
        self.f = f

    def __call__(self, ya: np.ndarray[float, 1], yb: np.ndarray[float, 1],
                 p: np.ndarray[float, 1]):
        if K == 0:
            r = self.f(ya, yb)
            a = np.asarray(r[0], float)
            return a, np.asarray(r[1], float), np.zeros((0, 0))
        else:
            r = self.f(ya, yb, p)
            return np.asarray(r[0], float), np.asarray(r[1], float), np.asarray(r[2], float)

class _FunS[F]:
    # singular term: f(x, y) + S y / (x - a)
    f: F
    a: float
    S: np.ndarray[float, 2]
    D: np.ndarray[float, 2]

    def __init__(self, f: F, a: float, S: np.ndarray[float, 2], D: np.ndarray[float, 2]):
        self.f = f
        self.a = a
        self.S = S
        self.D = D

    def __call__(self, x: np.ndarray[float, 1], y: np.ndarray[float, 2],
                 p: np.ndarray[float, 1]) -> np.ndarray[float, 2]:
        f = self.f(x, y, p)
        n = f.shape[0]
        m = f.shape[1]
        if x[0] == self.a:
            f0 = np.dot(self.D, f[:, 0])
            for a in range(n):
                f[a, 0] = f0[a]
            for j in range(1, m):
                col = np.dot(self.S, y[:, j]) / (x[j] - self.a)
                for a in range(n):
                    f[a, j] += col[a]
        else:
            for j in range(m):
                col = np.dot(self.S, y[:, j]) / (x[j] - self.a)
                for a in range(n):
                    f[a, j] += col[a]
        return f

class _FunJacS[FJ]:
    f: FJ
    a: float
    S: np.ndarray[float, 2]
    D: np.ndarray[float, 2]

    def __init__(self, f: FJ, a: float, S: np.ndarray[float, 2], D: np.ndarray[float, 2]):
        self.f = f
        self.a = a
        self.S = S
        self.D = D

    def __call__(self, x: np.ndarray[float, 1], y: np.ndarray[float, 2],
                 p: np.ndarray[float, 1]):
        r = self.f(x, y, p)
        df_dy = r[0]
        n = df_dy.shape[0]
        m = df_dy.shape[2]
        j0 = 0
        if x[0] == self.a:
            d0 = np.dot(self.D, df_dy[:, :, 0])
            for a in range(n):
                for b in range(n):
                    df_dy[a, b, 0] = d0[a, b]
            j0 = 1
        for j in range(j0, m):
            for a in range(n):
                for b in range(n):
                    df_dy[a, b, j] += self.S[a, b] / (x[j] - self.a)
        return df_dy, r[1]

# ------------------------------------------------------- finite differences

def _fd_fun_jac(fun, x: np.ndarray[float, 1], y: np.ndarray[float, 2],
                p: np.ndarray[float, 1], f0: np.ndarray[float, 2]):
    n = y.shape[0]
    m = y.shape[1]
    df_dy = np.empty((n, n, m))
    sq = math.sqrt(EPS)
    for i in range(n):
        y_new = y.copy()
        h = np.empty(m)
        for j in range(m):
            h[j] = sq * (1 + abs(y[i, j]))
            y_new[i, j] += h[j]
            h[j] = y_new[i, j] - y[i, j]
        f_new = fun(x, y_new, p)
        for a in range(n):
            for j in range(m):
                df_dy[a, i, j] = (f_new[a, j] - f0[a, j]) / h[j]
    k = p.size
    df_dp = np.empty((n, k, m))
    for i in range(k):
        p_new = p.copy()
        hh = sq * (1 + abs(p[i]))
        p_new[i] += hh
        hh = p_new[i] - p[i]
        f_new = fun(x, y, p_new)
        for a in range(n):
            for j in range(m):
                df_dp[a, i, j] = (f_new[a, j] - f0[a, j]) / hh
    return df_dy, df_dp

def _fd_bc_jac(bc, ya: np.ndarray[float, 1], yb: np.ndarray[float, 1],
               p: np.ndarray[float, 1], bc0: np.ndarray[float, 1]):
    n = ya.size
    k = p.size
    sq = math.sqrt(EPS)
    dya = np.empty((n + k, n))
    for i in range(n):
        ya_new = ya.copy()
        h = sq * (1 + abs(ya[i]))
        ya_new[i] += h
        h = ya_new[i] - ya[i]
        b = bc(ya_new, yb, p)
        for r in range(n + k):
            dya[r, i] = (b[r] - bc0[r]) / h
    dyb = np.empty((n + k, n))
    for i in range(n):
        yb_new = yb.copy()
        h = sq * (1 + abs(yb[i]))
        yb_new[i] += h
        h = yb_new[i] - yb[i]
        b = bc(ya, yb_new, p)
        for r in range(n + k):
            dyb[r, i] = (b[r] - bc0[r]) / h
    dp = np.empty((n + k, k))
    for i in range(k):
        p_new = p.copy()
        h = sq * (1 + abs(p[i]))
        p_new[i] += h
        h = p_new[i] - p[i]
        b = bc(ya, yb, p_new)
        for r in range(n + k):
            dp[r, i] = (b[r] - bc0[r]) / h
    return dya, dyb, dp

# -------------------------------------------------------------- collocation

def _collocation_fun(fun, y: np.ndarray[float, 2], p: np.ndarray[float, 1],
                     x: np.ndarray[float, 1], h: np.ndarray[float, 1]):
    n = y.shape[0]
    m = y.shape[1]
    f = fun(x, y, p)
    y_middle = np.empty((n, m - 1))
    for a in range(n):
        for j in range(m - 1):
            y_middle[a, j] = 0.5 * (y[a, j + 1] + y[a, j]) - 0.125 * h[j] * (f[a, j + 1] - f[a, j])
    xm = np.empty(m - 1)
    for j in range(m - 1):
        xm[j] = x[j] + 0.5 * h[j]
    f_middle = fun(xm, y_middle, p)
    col_res = np.empty((n, m - 1))
    for a in range(n):
        for j in range(m - 1):
            col_res[a, j] = y[a, j + 1] - y[a, j] - h[j] / 6 * (f[a, j] + f[a, j + 1] + 4 * f_middle[a, j])
    return col_res, y_middle, f, f_middle

def _build_jac(n: int, m: int, k: int, h: np.ndarray[float, 1],
               df_dy: np.ndarray[float, 3], dym: np.ndarray[float, 3],
               df_dp: np.ndarray[float, 3], dpm: np.ndarray[float, 3],
               dya: np.ndarray[float, 2], dyb: np.ndarray[float, 2],
               dbp: np.ndarray[float, 2]):
    rows = List[int]()
    cols = List[int]()
    vals = List[float]()
    # dPhi_dy_0
    for b in range(m - 1):
        hb = h[b]
        T = np.zeros((n, n))
        for a in range(n):
            for c in range(n):
                s = 0.0
                for e in range(n):
                    s += dym[a, e, b] * df_dy[e, c, b]
                T[a, c] = s
        for a in range(n):
            for c in range(n):
                v = (-1.0 if a == c else 0.0)
                v -= hb / 6 * (df_dy[a, c, b] + 2 * dym[a, c, b])
                v -= hb ** 2 / 12 * T[a, c]
                rows.append(b * n + a)
                cols.append(b * n + c)
                vals.append(v)
    for b in range(m - 1):
        hb = h[b]
        T = np.zeros((n, n))
        for a in range(n):
            for c in range(n):
                s = 0.0
                for e in range(n):
                    s += dym[a, e, b] * df_dy[e, c, b + 1]
                T[a, c] = s
        for a in range(n):
            for c in range(n):
                v = (1.0 if a == c else 0.0)
                v -= hb / 6 * (df_dy[a, c, b + 1] + 2 * dym[a, c, b])
                v += hb ** 2 / 12 * T[a, c]
                rows.append(b * n + a)
                cols.append(b * n + n + c)
                vals.append(v)
    for r in range(n + k):
        for c in range(n):
            rows.append((m - 1) * n + r)
            cols.append(c)
            vals.append(dya[r, c])
    for r in range(n + k):
        for c in range(n):
            rows.append((m - 1) * n + r)
            cols.append((m - 1) * n + c)
            vals.append(dyb[r, c])
    if k > 0:
        for b in range(m - 1):
            hb = h[b]
            # df_dp_middle += 0.125 * h * dym @ (df_dp[b] - df_dp[b+1])
            for a in range(n):
                for q in range(k):
                    s = 0.0
                    for e in range(n):
                        s += dym[a, e, b] * (df_dp[e, q, b] - df_dp[e, q, b + 1])
                    dpm[a, q, b] += 0.125 * hb * s
            for a in range(n):
                for q in range(k):
                    rows.append(b * n + a)
                    cols.append(m * n + q)
                    vals.append(-hb / 6 * (df_dp[a, q, b] + df_dp[a, q, b + 1] + 4 * dpm[a, q, b]))
        for r in range(n + k):
            for q in range(k):
                rows.append((m - 1) * n + r)
                cols.append(m * n + q)
                vals.append(dbp[r, q])
    N = m * n + k
    return csc_array((np.array(vals), (np.array(rows), np.array(cols))), shape=(N, N))

def _bc_ok(bc_res: np.ndarray[float, 1], bc_tol: float) -> bool:
    for v in bc_res:
        if not (abs(v) < bc_tol):
            return False
    return True

def _res_vec(col_res: np.ndarray[float, 2], bc_res: np.ndarray[float, 1]) -> np.ndarray[float, 1]:
    n = col_res.shape[0]
    mm = col_res.shape[1]
    out = np.empty(n * mm + bc_res.size)
    for j in range(mm):
        for a in range(n):
            out[j * n + a] = col_res[a, j]
    for i in range(bc_res.size):
        out[n * mm + i] = bc_res[i]
    return out

def _solve_newton(fun, bc, fun_jac, bc_jac, n: int, m: int, k: int,
                  x: np.ndarray[float, 1], h: np.ndarray[float, 1],
                  y0: np.ndarray[float, 2], p0: np.ndarray[float, 1],
                  B: np.ndarray[float, 2], has_B: bool, bvp_tol: float,
                  bc_tol: float):
    tol_r = 2.0 / 3 * h * 5e-2 * bvp_tol
    max_njev = 4
    max_iter = 8
    sigma = 0.2
    tau = 0.5
    n_trial = 4
    y = y0
    p = p0
    col_res, y_middle, f, f_middle = _collocation_fun(fun, y, p, x, h)
    bc_res = bc(y[:, 0], y[:, m - 1], p)
    res = _res_vec(col_res, bc_res)
    njev = 0
    singular = False
    recompute_jac = True
    x_middle = np.empty(m - 1)
    for j in range(m - 1):
        x_middle[j] = x[j] + 0.5 * h[j]
    LU = None
    step = np.zeros(m * n + k)
    cost = 0.0
    for iteration in range(max_iter):
        if recompute_jac:
            if fun_jac is None:
                df_dy, df_dp = _fd_fun_jac(fun, x, y, p, f)
                dym, dpm = _fd_fun_jac(fun, x_middle, y_middle, p, f_middle)
            else:
                r1 = fun_jac(x, y, p)
                df_dy = r1[0]
                df_dp = r1[1]
                r2 = fun_jac(x_middle, y_middle, p)
                dym = r2[0]
                dpm = r2[1]
            dya = np.zeros((0, 0))
            dyb = np.zeros((0, 0))
            dbp = np.zeros((0, 0))
            if bc_jac is None:
                dya, dyb, dbp = _fd_bc_jac(bc, y[:, 0].copy(), y[:, m - 1].copy(), p, bc_res)
            else:
                r3 = bc_jac(y[:, 0].copy(), y[:, m - 1].copy(), p)
                dya = r3[0]
                dyb = r3[1]
                dbp = r3[2]
            J = _build_jac(n, m, k, h, df_dy, dym, df_dp, dpm, dya, dyb, dbp)
            njev += 1
            try:
                LU = splu(J)
            except RuntimeError:
                singular = True
                break
            step = LU.solve(res)
            cost = float(np.dot(step, step))
        y_step = np.empty((n, m))
        for j in range(m):
            for a in range(n):
                y_step[a, j] = step[j * n + a]
        p_step = step[m * n:].copy()
        alpha = 1.0
        y_new = y
        p_new = p
        step_new = step
        cost_new = cost
        for trial in range(n_trial + 1):
            y_new = y - alpha * y_step
            if has_B:
                c0 = np.dot(B, y_new[:, 0])
                for a in range(n):
                    y_new[a, 0] = c0[a]
            p_new = p - alpha * p_step
            col_res, y_middle, f, f_middle = _collocation_fun(fun, y_new, p_new, x, h)
            bc_res = bc(y_new[:, 0], y_new[:, m - 1], p_new)
            res = _res_vec(col_res, bc_res)
            step_new = LU.solve(res)
            cost_new = float(np.dot(step_new, step_new))
            if cost_new < (1 - 2 * alpha * sigma) * cost:
                break
            if trial < n_trial:
                alpha *= tau
        y = y_new
        p = p_new
        if njev == max_njev:
            break
        ok = True
        for a in range(n):
            for j in range(m - 1):
                if not (abs(col_res[a, j]) < tol_r[j] * (1 + abs(f_middle[a, j]))):
                    ok = False
        if ok and _bc_ok(bc_res, bc_tol):
            break
        if alpha == 1:
            step = step_new
            cost = cost_new
            recompute_jac = False
        else:
            recompute_jac = True
    return y, p, singular

def _create_spline(y: np.ndarray[float, 2], yp: np.ndarray[float, 2],
                   h: np.ndarray[float, 1], x: np.ndarray[float, 1]) -> BVPSol:
    n = y.shape[0]
    m = y.shape[1]
    c = np.empty((4, n, m - 1))
    for a in range(n):
        for j in range(m - 1):
            slope = (y[a, j + 1] - y[a, j]) / h[j]
            t = (yp[a, j] + yp[a, j + 1] - 2 * slope) / h[j]
            c[0, a, j] = t / h[j]
            c[1, a, j] = (slope - yp[a, j]) / h[j] - t
            c[2, a, j] = yp[a, j]
            c[3, a, j] = y[a, j]
    return BVPSol(c, x)

def _rms_residuals(fun, sol: BVPSol, x: np.ndarray[float, 1], h: np.ndarray[float, 1],
                   p: np.ndarray[float, 1], r_middle: np.ndarray[float, 2],
                   f_middle: np.ndarray[float, 2]) -> np.ndarray[float, 1]:
    n = r_middle.shape[0]
    mm = h.size
    x1 = np.empty(mm)
    x2 = np.empty(mm)
    for j in range(mm):
        xmid = x[j] + 0.5 * h[j]
        s = 0.5 * h[j] * math.sqrt(3.0 / 7)
        x1[j] = xmid + s
        x2[j] = xmid - s
    y1 = sol(x1)
    y2 = sol(x2)
    y1p = sol(x1, 1)
    y2p = sol(x2, 1)
    f1 = fun(x1, y1, p)
    f2 = fun(x2, y2, p)
    out = np.empty(mm)
    for j in range(mm):
        s1 = 0.0
        s2 = 0.0
        sm = 0.0
        for a in range(n):
            r1 = (y1p[a, j] - f1[a, j]) / (1 + abs(f1[a, j]))
            r2 = (y2p[a, j] - f2[a, j]) / (1 + abs(f2[a, j]))
            rm = r_middle[a, j] / (1 + abs(f_middle[a, j]))
            s1 += r1 * r1
            s2 += r2 * r2
            sm += rm * rm
        out[j] = (0.5 * (32.0 / 45 * sm + 49.0 / 90 * (s1 + s2))) ** 0.5
    return out

def _bvp_main(fun, bc, fun_jac, bc_jac, x0: np.ndarray[float, 1],
              y0: np.ndarray[float, 2], p0: np.ndarray[float, 1], B: np.ndarray[float, 2],
              has_B: bool, tol: float, max_nodes: int, bc_tol: float):
    x = x0
    y = y0
    p = p0
    n = y.shape[0]
    k = p.size
    h = np.diff(x)
    max_iteration = 10
    f = fun(x, y, p)
    if f.shape[0] != y.shape[0] or f.shape[1] != y.shape[1]:
        raise ValueError(f'`fun` return is expected to have shape {y.shape}, but actually has {f.shape}.')
    bc_res = bc(y[:, 0], y[:, y.shape[1] - 1], p)
    if bc_res.size != n + k:
        raise ValueError(f'`bc` return is expected to have shape {(n + k,)}, but actually has {bc_res.shape}.')
    status = 0
    iteration = 0
    sol = BVPSol(np.zeros((4, n, 1)), np.zeros(2))
    rms_res = np.zeros(0)
    while True:
        m = x.size
        y, p, singular = _solve_newton(fun, bc, fun_jac, bc_jac, n, m, k, x, h, y, p,
                                       B, has_B, tol, bc_tol)
        iteration += 1
        col_res, y_middle, f, f_middle = _collocation_fun(fun, y, p, x, h)
        bc_res = bc(y[:, 0], y[:, m - 1], p)
        max_bc_res = 0.0
        for v in bc_res:
            if abs(v) > max_bc_res:
                max_bc_res = abs(v)
        r_middle = np.empty((n, m - 1))
        for a in range(n):
            for j in range(m - 1):
                r_middle[a, j] = 1.5 * col_res[a, j] / h[j]
        sol = _create_spline(y, f, h, x)
        rms_res = _rms_residuals(fun, sol, x, h, p, r_middle, f_middle)
        if singular:
            status = 2
            break
        ins1 = List[int]()
        ins2 = List[int]()
        for j in range(m - 1):
            if rms_res[j] > tol and rms_res[j] < 100 * tol:
                ins1.append(j)
            if rms_res[j] >= 100 * tol:
                ins2.append(j)
        nodes_added = len(ins1) + 2 * len(ins2)
        if m + nodes_added > max_nodes:
            status = 1
            break
        if nodes_added > 0:
            nx = List[float]()
            for v in x:
                nx.append(v)
            for j in ins1:
                nx.append(0.5 * (x[j] + x[j + 1]))
            for j in ins2:
                nx.append((2 * x[j] + x[j + 1]) / 3)
                nx.append((x[j] + 2 * x[j + 1]) / 3)
            x = np.sort(np.array(nx))
            h = np.diff(x)
            y = sol(x)
        elif max_bc_res <= bc_tol:
            status = 0
            break
        elif iteration >= max_iteration:
            status = 3
            break
    return BVPResult(sol, p, x, y, f, rms_res, iteration, status)

def solve_bvp(fun, bc, x, y, p = None, S = None, fun_jac = None, bc_jac = None,
              tol: float = 1e-3, max_nodes: int = 1000, verbose: int = 0,
              bc_tol: Optional[float] = None):
    xa = np.asarray(x, float)
    if xa.ndim != 1:
        raise ValueError('`x` must be 1 dimensional.')
    h = np.diff(xa)
    for v in h:
        if v <= 0:
            raise ValueError('`x` must be strictly increasing.')
    ya = np.asarray(y, float)
    if ya.ndim != 2:
        raise ValueError('`y` must be 2 dimensional.')
    if ya.shape[1] != xa.shape[0]:
        raise ValueError(f'`y` is expected to have {xa.shape[0]} columns, but actually has {ya.shape[1]}.')
    yc = ya.copy()
    t = tol
    if t < 100 * EPS:
        t = 100 * EPS
    if verbose != 0 and verbose != 1 and verbose != 2:
        raise ValueError('`verbose` must be in [0, 1, 2].')
    btol = t if bc_tol is None else bc_tol
    if p is None:
        return _bvp_k0(fun, bc, xa, yc, S, fun_jac, bc_jac, t, max_nodes, btol)
    else:
        pa = np.asarray(p, float)
        if pa.ndim != 1:
            raise ValueError('`p` must be 1 dimensional.')
        return _bvp_k1(fun, bc, xa, yc, pa.copy(), S, fun_jac, bc_jac, t, max_nodes, btol)

def _bvp_k0(fun, bc, x, y, S, fun_jac, bc_jac, tol: float, max_nodes: int, bc_tol: float):
    return _bvp_s(_FunW[type(fun), 0](fun), _BcW[type(bc), 0](bc), _wrap_fj0(fun_jac),
                  _wrap_bj0(bc_jac), x, y, np.zeros(0), S, tol, max_nodes, bc_tol)

def _bvp_k1(fun, bc, x, y, p, S, fun_jac, bc_jac, tol: float, max_nodes: int, bc_tol: float):
    return _bvp_s(_FunW[type(fun), 1](fun), _BcW[type(bc), 1](bc), _wrap_fj1(fun_jac),
                  _wrap_bj1(bc_jac), x, y, p, S, tol, max_nodes, bc_tol)

def _wrap_fj0(fj):
    if fj is None:
        return None
    else:
        return _FunJacW[type(fj), 0](fj)

def _wrap_fj1(fj):
    if fj is None:
        return None
    else:
        return _FunJacW[type(fj), 1](fj)

def _wrap_bj0(bj):
    if bj is None:
        return None
    else:
        return _BcJacW[type(bj), 0](bj)

def _wrap_bj1(bj):
    if bj is None:
        return None
    else:
        return _BcJacW[type(bj), 1](bj)

def _bvp_s(fun, bc, fj, bj, x, y, p, S, tol: float, max_nodes: int, bc_tol: float):
    n = y.shape[0]
    if S is None:
        return _bvp_main(fun, bc, fj, bj, x, y, p, np.zeros((0, 0)), False, tol,
                         max_nodes, bc_tol)
    else:
        Sa = np.asarray(S, float)
        if Sa.shape[0] != n or Sa.shape[1] != n:
            raise ValueError(f'`S` is expected to have shape {(n, n)}, but actually has {Sa.shape}')
        B = np.identity(n) - np.dot(pinv(Sa), Sa)
        c0 = np.dot(B, y[:, 0])
        for a in range(n):
            y[a, 0] = c0[a]
        D = pinv(np.identity(n) - Sa)
        a0 = x[0]
        return _bvp_main(_FunS(fun, a0, Sa, D), bc, _wrap_fjs(fj, a0, Sa, D), bj, x, y,
                         p, B, True, tol, max_nodes, bc_tol)

def _wrap_fjs(fj, a0: float, S: np.ndarray[float, 2], D: np.ndarray[float, 2]):
    if fj is None:
        return None
    else:
        return _FunJacS(fj, a0, S, D)
