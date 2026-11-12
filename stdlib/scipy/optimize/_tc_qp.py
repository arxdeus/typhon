# Ported from scipy 1.18.1 (BSD-3-Clause)

# trust-constr building blocks: _trustregion_constr/projections.py and
# _trustregion_constr/qp_subproblem.py. Operators are dense; the sparse
# 'AugmentedSystem' factorization (used by scipy for sparse Jacobians, e.g.
# bound-only problems) is reproduced with a dense LU of the same KKT matrix.

import math
import numpy as np
from scipy.linalg import qr, solve_triangular, svd, lu_factor, lu_solve
from ._common import INF

def _norm(v: np.ndarray[float, 1]) -> float:
    if v.size == 0:
        return 0.0
    return float(np.linalg.norm(v))

def _norm_fro(A: np.ndarray[float, 2]) -> float:
    if A.size == 0:
        return 0.0
    return float(np.linalg.norm(A, ord='fro'))

def orthogonality(A: np.ndarray[float, 2], g: np.ndarray[float, 1]) -> float:
    norm_g = _norm(g)
    norm_A = _norm_fro(A)
    if norm_g == 0 or norm_A == 0:
        return 0.0
    norm_A_g = _norm(A @ g)
    return norm_A_g / (norm_A * norm_g)

_P_QR = 0
_P_AUG = 1
_P_SVD = 2

class _Proj:
    # projections(A): null_space (Z), least_squares (LS), row_space (Y)
    kind: int
    A: np.ndarray[float, 2]
    m: int
    n: int
    orth_tol: float
    max_refin: int
    Q: np.ndarray[float, 2]
    R: np.ndarray[float, 2]
    P: np.ndarray[int, 1]
    K: np.ndarray[float, 2]
    LU: np.ndarray[float, 2]
    piv: np.ndarray[int, 1]
    U: np.ndarray[float, 2]
    s: np.ndarray[float, 1]
    Vt: np.ndarray[float, 2]

    def __init__(self, A: np.ndarray[float, 2], sparse: bool,
                 orth_tol: float = 1e-12, max_refin: int = 3, tol: float = 1e-15):
        m, n = A.shape
        self.A = A
        self.m = m
        self.n = n
        self.orth_tol = orth_tol
        self.max_refin = max_refin
        self.Q = np.zeros((0, 0))
        self.R = np.zeros((0, 0))
        self.P = np.zeros(0, int)
        self.K = np.zeros((0, 0))
        self.LU = np.zeros((0, 0))
        self.piv = np.zeros(0, int)
        self.U = np.zeros((0, 0))
        self.s = np.zeros(0)
        self.Vt = np.zeros((0, 0))
        if m * n == 0 or sparse:
            self.kind = _P_AUG
            K = np.zeros((n + m, n + m))
            for i in range(n):
                K[i, i] = 1.0
            for i in range(m):
                for j in range(n):
                    K[j, n + i] = A[i, j]
                    K[n + i, j] = A[i, j]
            self.K = K
            singular = False
            try:
                lu, piv = lu_factor(K)
                self.LU = lu
                self.piv = np.asarray(piv, int)
                for i in range(n + m):
                    if lu[i, i] == 0.0:
                        singular = True
            except:
                singular = True
            if singular:
                self._init_svd(tol)
        else:
            self.kind = _P_QR
            Q, R, P = qr(A.T, mode='economic', pivoting=True)
            self.Q = Q
            self.R = R
            self.P = P
            last = 0.0
            for j in range(R.shape[1]):
                last = max(last, abs(R[R.shape[0] - 1, j]))
            if last < tol:
                self._init_svd(tol)

    def _init_svd(self, tol: float):
        self.kind = _P_SVD
        U, s, Vt = svd(self.A, full_matrices=False)
        keep = List[int]()
        for i in range(s.size):
            if s[i] > tol:
                keep.append(i)
        k = len(keep)
        self.U = np.empty((self.m, k))
        self.Vt = np.empty((k, self.n))
        self.s = np.empty(k)
        for a in range(k):
            self.s[a] = s[keep[a]]
            for i in range(self.m):
                self.U[i, a] = U[i, keep[a]]
            for j in range(self.n):
                self.Vt[a, j] = Vt[keep[a], j]

    def _solve(self, v: np.ndarray[float, 1]) -> np.ndarray[float, 1]:
        return np.asarray(lu_solve((self.LU, self.piv), v), float)

    def _qr_v(self, x: np.ndarray[float, 1]) -> np.ndarray[float, 1]:
        aux1 = self.Q.T @ x
        aux2 = solve_triangular(self.R, aux1, lower=False)
        v = np.zeros(self.m)
        for i in range(self.m):
            v[self.P[i]] = aux2[i]
        return v

    def _svd_v(self, x: np.ndarray[float, 1]) -> np.ndarray[float, 1]:
        aux1 = self.Vt @ x
        aux2 = 1 / self.s * aux1
        return self.U @ aux2

    def null_space(self, x: np.ndarray[float, 1]) -> np.ndarray[float, 1]:
        A = self.A
        if self.kind == _P_AUG:
            v = np.concatenate((x, np.zeros(self.m)))
            lu_sol = self._solve(v)
            z = lu_sol[:self.n].copy()
            k = 0
            while orthogonality(A, z) > self.orth_tol:
                if k >= self.max_refin:
                    break
                new_v = v - self.K @ lu_sol
                lu_update = self._solve(new_v)
                lu_sol += lu_update
                z = lu_sol[:self.n].copy()
                k += 1
            return z
        elif self.kind == _P_QR:
            v = self._qr_v(x)
            z = x - A.T @ v
            k = 0
            while orthogonality(A, z) > self.orth_tol:
                if k >= self.max_refin:
                    break
                v = self._qr_v(z)
                z = z - A.T @ v
                k += 1
            return z
        else:
            v = self._svd_v(x)
            z = x - A.T @ v
            k = 0
            while orthogonality(A, z) > self.orth_tol:
                if k >= self.max_refin:
                    break
                v = self._svd_v(z)
                z = z - A.T @ v
                k += 1
            return z

    def least_squares(self, x: np.ndarray[float, 1]) -> np.ndarray[float, 1]:
        if self.kind == _P_AUG:
            v = np.concatenate((x, np.zeros(self.m)))
            lu_sol = self._solve(v)
            return lu_sol[self.n:self.m + self.n].copy()
        elif self.kind == _P_QR:
            return self._qr_v(x)
        else:
            return self._svd_v(x)

    def row_space(self, x: np.ndarray[float, 1]) -> np.ndarray[float, 1]:
        if self.kind == _P_AUG:
            v = np.concatenate((np.zeros(self.n), x))
            lu_sol = self._solve(v)
            return lu_sol[:self.n].copy()
        elif self.kind == _P_QR:
            aux1 = np.empty(self.m)
            for i in range(self.m):
                aux1[i] = x[self.P[i]]
            aux2 = solve_triangular(self.R, aux1, lower=False, trans='T')
            return self.Q @ aux2
        else:
            aux1 = self.U.T @ x
            aux2 = 1 / self.s * aux1
            return self.Vt.T @ aux2

# ------------------------------------------------------- qp_subproblem

def sphere_intersections(z: np.ndarray[float, 1], d: np.ndarray[float, 1],
                         trust_radius: float, entire_line: bool = False):
    if _norm(d) == 0:
        return 0.0, 0.0, False
    if trust_radius == INF:
        if entire_line:
            return -INF, INF, True
        return 0.0, 1.0, True
    a = float(np.dot(d, d))
    b = 2 * float(np.dot(z, d))
    c = float(np.dot(z, z)) - trust_radius ** 2
    discriminant = b * b - 4 * a * c
    if discriminant < 0:
        return 0.0, 0.0, False
    sqrt_discriminant = math.sqrt(discriminant)
    aux = b + math.copysign(sqrt_discriminant, b)
    ta = -aux / (2 * a)
    tb = -2 * c / aux
    if tb < ta:
        t = ta
        ta = tb
        tb = t
    if entire_line:
        return ta, tb, True
    elif tb < 0 or ta > 1:
        return 0.0, 0.0, False
    return max(0.0, ta), min(1.0, tb), True

def box_intersections(z: np.ndarray[float, 1], d: np.ndarray[float, 1],
                      lb: np.ndarray[float, 1], ub: np.ndarray[float, 1],
                      entire_line: bool = False):
    if _norm(d) == 0:
        return 0.0, 0.0, False
    n = z.size
    for i in range(n):
        if d[i] == 0 and (z[i] < lb[i] or z[i] > ub[i]):
            return 0.0, 0.0, False
    ta = -INF
    tb = INF
    for i in range(n):
        if d[i] != 0:
            t_lb = (lb[i] - z[i]) / d[i]
            t_ub = (ub[i] - z[i]) / d[i]
            ta = max(ta, min(t_lb, t_ub))
            tb = min(tb, max(t_lb, t_ub))
    intersect = ta <= tb
    if not entire_line:
        if tb < 0 or ta > 1:
            return 0.0, 0.0, False
        ta = max(0.0, ta)
        tb = min(1.0, tb)
    return ta, tb, intersect

def box_sphere_intersections(z, d, lb, ub, trust_radius: float,
                             entire_line: bool = False):
    ta_b, tb_b, intersect_b = box_intersections(z, d, lb, ub, entire_line)
    ta_s, tb_s, intersect_s = sphere_intersections(z, d, trust_radius, entire_line)
    ta = max(ta_b, ta_s)
    tb = min(tb_b, tb_s)
    intersect = intersect_b and intersect_s and ta <= tb
    return ta, tb, intersect

def inside_box_boundaries(x, lb, ub) -> bool:
    for i in range(x.size):
        if not (lb[i] <= x[i]) or not (x[i] <= ub[i]):
            return False
    return True

def reinforce_box_boundaries(x, lb, ub):
    return np.minimum(np.maximum(x, lb), ub)

def modified_dogleg(A: np.ndarray[float, 2], Y: _Proj, b: np.ndarray[float, 1],
                    trust_radius: float, lb, ub) -> np.ndarray[float, 1]:
    newton_point = -Y.row_space(b)
    if inside_box_boundaries(newton_point, lb, ub) and _norm(newton_point) <= trust_radius:
        return newton_point
    g = A.T @ b
    A_g = A @ g
    cauchy_point = -float(np.dot(g, g)) / float(np.dot(A_g, A_g)) * g
    origin_point = np.zeros_like(cauchy_point)
    z = cauchy_point
    p = newton_point - cauchy_point
    _, alpha, intersect = box_sphere_intersections(z, p, lb, ub, trust_radius)
    x1 = z
    if intersect:
        x1 = z + alpha * p
    else:
        z = origin_point
        p = cauchy_point
        _, alpha, _ = box_sphere_intersections(z, p, lb, ub, trust_radius)
        x1 = z + alpha * p
    z = origin_point
    p = newton_point
    _, alpha, _ = box_sphere_intersections(z, p, lb, ub, trust_radius)
    x2 = z + alpha * p
    if _norm(A @ x1 + b) < _norm(A @ x2 + b):
        return x1
    return x2

def projected_cg(H, c: np.ndarray[float, 1], Z: _Proj, b: np.ndarray[float, 1],
                 trust_radius: float, lb: np.ndarray[float, 1], ub: np.ndarray[float, 1]):
    # returns (x, niter, stop_cond, hits_boundary)
    CLOSE_TO_ZERO = 1e-25
    n = c.size
    m = b.size
    x = Z.row_space(-b)
    r = Z.null_space(H.dot(x) + c)
    g = Z.null_space(r)
    p = -g
    H_p = H.dot(p)
    rt_g = _norm(g) ** 2
    tr_distance = trust_radius - _norm(x)
    if tr_distance < 0:
        raise ValueError('Trust region problem does not have a solution.')
    elif tr_distance < CLOSE_TO_ZERO:
        return x, 0, 2, True
    tol = max(min(0.01 * math.sqrt(rt_g), 0.1 * rt_g), CLOSE_TO_ZERO)
    max_iter = n - m
    max_infeasible_iter = n - m
    hits_boundary = False
    stop_cond = 1
    counter = 0
    last_feasible_x = np.zeros_like(x)
    k = 0
    for i in range(max_iter):
        if rt_g < tol:
            stop_cond = 4
            break
        k += 1
        pt_H_p = float(H_p.dot(p))
        if pt_H_p <= 0:
            if trust_radius == INF:
                raise ValueError('Negative curvature not allowed for unrestricted problems.')
            else:
                _, alpha, intersect = box_sphere_intersections(x, p, lb, ub, trust_radius, True)
                if intersect:
                    x = x + alpha * p
                x = reinforce_box_boundaries(x, lb, ub)
                stop_cond = 3
                hits_boundary = True
                break
        alpha = rt_g / pt_H_p
        x_next = x + alpha * p
        if _norm(x_next) >= trust_radius:
            _, theta, intersect = box_sphere_intersections(x, alpha * p, lb, ub, trust_radius)
            if intersect:
                x = x + theta * alpha * p
            x = reinforce_box_boundaries(x, lb, ub)
            stop_cond = 2
            hits_boundary = True
            break
        if inside_box_boundaries(x_next, lb, ub):
            counter = 0
        else:
            counter += 1
        if counter > 0:
            _, theta, intersect = box_sphere_intersections(x, alpha * p, lb, ub, trust_radius)
            if intersect:
                last_feasible_x = x + theta * alpha * p
                last_feasible_x = reinforce_box_boundaries(last_feasible_x, lb, ub)
                counter = 0
        if counter > max_infeasible_iter:
            break
        r_next = r + alpha * H_p
        g_next = Z.null_space(r_next)
        rt_g_next = _norm(g_next) ** 2
        beta = rt_g_next / rt_g
        p = -g_next + beta * p
        x = x_next
        g = g_next
        r = g_next
        rt_g = _norm(g) ** 2
        H_p = H.dot(p)
    if not inside_box_boundaries(x, lb, ub):
        x = last_feasible_x
        hits_boundary = True
    return x, k, stop_cond, hits_boundary
