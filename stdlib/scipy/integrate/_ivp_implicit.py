# Ported from scipy 1.18.1 (BSD-3-Clause)
#
# Implicit solve_ivp methods: Radau (scipy/integrate/_ivp/radau.py), BDF
# (bdf.py) and the LSODA OdeSolver wrapper (lsoda.py) plus `num_jac` from
# common.py. Dense linear algebra uses an unblocked LU with partial pivoting
# (dgetf2 order; complex pivots chosen by |re|+|im| like izamax) instead of
# LAPACK, so step counts can differ from scipy in rare near-tie cases.
# Deviations: `jac_sparsity` and sparse Jacobians are not supported
# (ValueError); BDF does not support complex y0.

import math
import numpy as np
from _lsoda import _Lsoda

_EPS = 2.220446049250313e-16
_TOO_SMALL_STEP = "Required step size is less than spacing between numbers."

def _nrm(x: np.ndarray[float, 1]) -> float:
    s = 0.0
    for i in range(x.shape[0]):
        s += x[i] * x[i]
    return math.sqrt(s) / math.sqrt(float(x.shape[0]))

def _pabs(v) -> float:
    if isinstance(v, complex):
        return abs(v.real) + abs(v.imag)
    else:
        return abs(v)

def _lu(a, n: int, piv: List[int]) -> int:
    info = 0
    for k in range(n):
        p = k
        amax = _pabs(a[k, k])
        for i in range(k + 1, n):
            v = _pabs(a[i, k])
            if v > amax:
                amax = v
                p = i
        piv[k] = p
        if amax == 0.0:
            if info == 0:
                info = k + 1
            continue
        if p != k:
            for j in range(n):
                tmp = a[k, j]
                a[k, j] = a[p, j]
                a[p, j] = tmp
        inv = 1.0 / a[k, k]
        for i in range(k + 1, n):
            a[i, k] *= inv
        for j in range(k + 1, n):
            akj = a[k, j]
            for i in range(k + 1, n):
                a[i, j] -= a[i, k] * akj
    return info

def _lus(a, n: int, piv: List[int], x):
    for k in range(n):
        p = piv[k]
        if p != k:
            tmp = x[k]
            x[k] = x[p]
            x[p] = tmp
    for j in range(n):
        xj = x[j]
        for i in range(j + 1, n):
            x[i] -= xj * a[i, j]
    for j in range(n - 1, -1, -1):
        x[j] /= a[j, j]
        xj = x[j]
        for i in range(j):
            x[i] -= xj * a[i, j]

_NJ_REJECT = _EPS ** 0.875
_NJ_SMALL = _EPS ** 0.75
_NJ_BIG = _EPS ** 0.25
_NJ_MIN_FACTOR = 1e3 * _EPS

def _num_jac(fv, t: float, y: np.ndarray[float, 1], f: np.ndarray[float, 1],
             threshold: np.ndarray[float, 1], factor: np.ndarray[float, 1]):
    # factor of size 0 means None. Returns (J, factor).
    n = y.shape[0]
    if n == 0:
        return np.empty((0, 0)), factor
    fac = np.empty(n)
    for i in range(n):
        fac[i] = math.sqrt(_EPS) if factor.shape[0] == 0 else factor[i]
    y_scale = np.empty(n)
    h = np.empty(n)
    for i in range(n):
        fs = 1.0 if f[i] >= 0 else -1.0
        y_scale[i] = fs * max(threshold[i], abs(y[i]))
        h[i] = (y[i] + fac[i] * y_scale[i]) - y[i]
        while h[i] == 0:
            fac[i] *= 10
            h[i] = (y[i] + fac[i] * y_scale[i]) - y[i]
    Y = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            Y[i, j] = y[i] + (h[j] if i == j else 0.0)
    f_new = fv(t, Y)
    diff = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            diff[i, j] = f_new[i, j] - f[i]
    max_ind = [0] * n
    max_diff = np.empty(n)
    scale = np.empty(n)
    for j in range(n):
        mi = 0
        mv = abs(diff[0, j])
        for i in range(1, n):
            if abs(diff[i, j]) > mv:
                mv = abs(diff[i, j])
                mi = i
        max_ind[j] = mi
        max_diff[j] = mv
        scale[j] = max(abs(f[mi]), abs(f_new[mi, j]))
    ind: List[int] = []
    for j in range(n):
        if max_diff[j] < _NJ_REJECT * scale[j]:
            ind.append(j)
    if len(ind) > 0:
        k = len(ind)
        new_factor = [10.0 * fac[j] for j in ind]
        h_new = [(y[ind[q]] + new_factor[q] * y_scale[ind[q]]) - y[ind[q]] for q in range(k)]
        Y2 = np.empty((n, k))
        for i in range(n):
            for q in range(k):
                Y2[i, q] = y[i] + (h_new[q] if i == ind[q] else 0.0)
        f_new2 = fv(t, Y2)
        for q in range(k):
            j = ind[q]
            mi = 0
            mv = abs(f_new2[0, q] - f[0])
            for i in range(1, n):
                dv = abs(f_new2[i, q] - f[i])
                if dv > mv:
                    mv = dv
                    mi = i
            scale_new = max(abs(f[mi]), abs(f_new2[mi, q]))
            if max_diff[j] * scale_new < mv * scale[j]:
                fac[j] = new_factor[q]
                h[j] = h_new[q]
                for i in range(n):
                    diff[i, j] = f_new2[i, q] - f[i]
                scale[j] = scale_new
                max_diff[j] = mv
    for i in range(n):
        for j in range(n):
            diff[i, j] /= h[j]
    for j in range(n):
        if max_diff[j] < _NJ_SMALL * scale[j]:
            fac[j] *= 10
        elif max_diff[j] > _NJ_BIG * scale[j]:
            fac[j] *= 0.1
        fac[j] = max(fac[j], _NJ_MIN_FACTOR)
    return diff, fac

def _select_initial_step(fn, t0: float, y0: np.ndarray[float, 1], t_bound: float,
                         max_step: float, f0: np.ndarray[float, 1], direction: float,
                         order: int, rtol: float, atol: np.ndarray[float, 1]) -> float:
    n = y0.shape[0]
    if n == 0:
        return float('inf')
    interval_length = abs(t_bound - t0)
    if interval_length == 0.0:
        return 0.0
    scale = np.empty(n)
    for i in range(n):
        scale[i] = atol[i] + abs(y0[i]) * rtol
    d0 = _nrm(y0 / scale)
    d1 = _nrm(f0 / scale)
    h0 = 1e-6 if (d0 < 1e-5 or d1 < 1e-5) else 0.01 * d0 / d1
    h0 = min(h0, interval_length)
    y1 = y0 + h0 * direction * f0
    f1 = fn(t0 + h0 * direction, y1)
    d2 = _nrm((f1 - f0) / scale) / h0
    h1 = 0.0
    if d1 <= 1e-15 and d2 <= 1e-15:
        h1 = max(1e-6, h0 * 1e-3)
    else:
        h1 = (0.01 / max(d1, d2)) ** (1.0 / (order + 1))
    return min(100 * h0, h1, interval_length, max_step)

# ---------------------------------------------------------------- Radau IIA
_S6 = 6 ** 0.5
_RC = [(4 - _S6) / 10, (4 + _S6) / 10, 1.0]
_RE = [(-13 - 7 * _S6) / 3, (-13 + 7 * _S6) / 3, -1.0 / 3]
_MU_REAL = 3 + 3 ** (2 / 3) - 3 ** (1 / 3)
_MU_COMPLEX = complex(3 + 0.5 * (3 ** (1 / 3) - 3 ** (2 / 3)), -0.5 * (3 ** (5 / 6) + 3 ** (7 / 6)))
_RT = [[0.09443876248897524, -0.14125529502095421, 0.03002919410514742],
       [0.25021312296533332, 0.20412935229379994, -0.38294211275726192],
       [1.0, 1.0, 0.0]]
_RTI = [[4.17871859155190428, 0.32768282076106237, 0.52337644549944951],
        [-4.17871859155190428, -0.32768282076106237, 0.47662355450055044],
        [0.50287263494578682, -2.57192694985560522, 0.59603920482822492]]
_RP = [[13 / 3 + 7 * _S6 / 3, -23 / 3 - 22 * _S6 / 3, 10 / 3 + 5 * _S6],
       [13 / 3 - 7 * _S6 / 3, -23 / 3 + 22 * _S6 / 3, 10 / 3 - 5 * _S6],
       [1 / 3, -8 / 3, 10 / 3]]
_RADAU_NEWTON_MAXITER = 6

class _Radau:
    n: int
    t: float
    y: np.ndarray[float, 1]
    f: np.ndarray[float, 1]
    t_old: float
    y_old: np.ndarray[float, 1]
    t_bound: float
    direction: float
    max_step: float
    rtol: float
    atol: np.ndarray[float, 1]
    h_abs: float
    h_abs_old: float        # < 0 means None
    error_norm_old: float   # < 0 means None
    newton_tol: float
    has_jac: bool           # False: finite differences
    const_jac: bool         # jac given as array
    jac_factor: np.ndarray[float, 1]
    J: np.ndarray[float, 2]
    LU_real: np.ndarray[float, 2]
    piv_real: List[int]
    LU_complex: np.ndarray[complex, 2]
    piv_complex: List[int]
    have_lu: bool
    current_jac: bool
    Z: np.ndarray[float, 2]
    has_sol: bool
    sol_Q: np.ndarray[float, 2]
    sol_t_old: float
    sol_y_old: np.ndarray[float, 1]
    status: str
    nfev: int
    njev: int
    nlu: int

    def __init__(self, t0: float, y0: np.ndarray[float, 1], t_bound: float, max_step: float,
                 rtol: float, atol: np.ndarray[float, 1], jac_mode: int):
        self.n = y0.shape[0]
        self.t = t0
        self.y = y0
        self.f = y0
        self.t_old = t0
        self.y_old = y0
        self.t_bound = t_bound
        self.direction = 1.0 if t_bound >= t0 else -1.0
        self.max_step = max_step
        self.rtol = rtol
        self.atol = atol
        self.h_abs = 0.0
        self.h_abs_old = -1.0
        self.error_norm_old = -1.0
        self.newton_tol = max(10 * _EPS / rtol, min(0.03, rtol ** 0.5))
        self.has_jac = jac_mode != 0
        self.const_jac = jac_mode == 2
        self.jac_factor = np.zeros(0)
        self.J = np.zeros((self.n, self.n))
        self.LU_real = np.zeros((self.n, self.n))
        self.piv_real = [0] * self.n
        self.LU_complex = np.zeros((self.n, self.n), dtype=complex)
        self.piv_complex = [0] * self.n
        self.have_lu = False
        self.current_jac = True
        self.Z = np.zeros((3, self.n))
        self.has_sol = False
        self.sol_Q = np.zeros((self.n, 3))
        self.sol_t_old = t0
        self.sol_y_old = y0
        self.status = 'running'
        self.nfev = 0
        self.njev = 0
        self.nlu = 0

    def fun(self, F, t: float, y: np.ndarray[float, 1]) -> np.ndarray[float, 1]:
        self.nfev += 1
        return F[0](t, y)

    def _jac(self, F, t: float, y: np.ndarray[float, 1], f: np.ndarray[float, 1]) -> np.ndarray[float, 2]:
        self.njev += 1
        if not self.has_jac:
            J, fac = _num_jac(F[1], t, y, f, self.atol, self.jac_factor)
            self.jac_factor = fac
            return J
        return F[2](t, y)

    def init_step(self, F, first_step: Optional[float]):
        self.f = self.fun(F, self.t, self.y)
        if first_step is None:
            fn = lambda tt, yy: self.fun(F, tt, yy)
            self.h_abs = _select_initial_step(fn, self.t, self.y, self.t_bound, self.max_step,
                                              self.f, self.direction, 3, self.rtol, self.atol)
        else:
            fs = first_step
            if fs <= 0:
                raise ValueError("`first_step` must be positive.")
            if fs > abs(self.t_bound - self.t):
                raise ValueError("`first_step` exceeds bounds.")
            self.h_abs = fs
        if not self.has_jac:
            self.J = self._jac(F, self.t, self.y, self.f)
        else:
            self.J = F[2](self.t, self.y)
            self.njev = 0 if self.const_jac else 1

    def _factor(self, h: float):
        n = self.n
        mr = _MU_REAL / h
        mc = _MU_COMPLEX / h
        for i in range(n):
            for j in range(n):
                d = 1.0 if i == j else 0.0
                self.LU_real[i, j] = mr * d - self.J[i, j]
                self.LU_complex[i, j] = mc * d - self.J[i, j]
        _lu(self.LU_real, n, self.piv_real)
        self.nlu += 1
        _lu(self.LU_complex, n, self.piv_complex)
        self.nlu += 1
        self.have_lu = True

    def _collocation(self, F, t: float, y: np.ndarray[float, 1], h: float,
                     Z0: np.ndarray[float, 2], scale: np.ndarray[float, 1]):
        # returns (converged, n_iter, Z, rate); rate < 0 means None
        n = self.n
        M_real = _MU_REAL / h
        M_complex = _MU_COMPLEX / h
        W = np.zeros((3, n))
        for r in range(3):
            for j in range(n):
                W[r, j] = _RTI[r][0] * Z0[0, j] + _RTI[r][1] * Z0[1, j] + _RTI[r][2] * Z0[2, j]
        Z = Z0.copy()
        F3 = np.empty((3, n))
        dW = np.empty((3, n))
        dW_norm_old = -1.0
        converged = False
        rate = -1.0
        k = 0
        fr = np.empty(n)
        fc = np.empty(n, dtype=complex)
        dws = np.empty(n)
        while k < _RADAU_NEWTON_MAXITER:
            finite = True
            for i in range(3):
                yi = y + Z[i]
                fi = self.fun(F, t + h * _RC[i], yi)
                for j in range(n):
                    F3[i, j] = fi[j]
                    if not math.isfinite(fi[j]):
                        finite = False
            if not finite:
                break
            for j in range(n):
                fr[j] = (F3[0, j] * _RTI[0][0] + F3[1, j] * _RTI[0][1] + F3[2, j] * _RTI[0][2]) - M_real * W[0, j]
                re = F3[0, j] * _RTI[1][0] + F3[1, j] * _RTI[1][1] + F3[2, j] * _RTI[1][2]
                im = F3[0, j] * _RTI[2][0] + F3[1, j] * _RTI[2][1] + F3[2, j] * _RTI[2][2]
                fc[j] = complex(re, im) - M_complex * complex(W[1, j], W[2, j])
            _lus(self.LU_real, n, self.piv_real, fr)
            _lus(self.LU_complex, n, self.piv_complex, fc)
            for j in range(n):
                dW[0, j] = fr[j]
                dW[1, j] = fc[j].real
                dW[2, j] = fc[j].imag
            s = 0.0
            for r in range(3):
                for j in range(n):
                    v = dW[r, j] / scale[j]
                    s += v * v
            dW_norm = math.sqrt(s) / math.sqrt(3.0 * n)
            if dW_norm_old >= 0:
                rate = dW_norm / dW_norm_old
            if rate >= 0 and (rate >= 1 or rate ** (_RADAU_NEWTON_MAXITER - k) / (1 - rate) * dW_norm > self.newton_tol):
                break
            for r in range(3):
                for j in range(n):
                    W[r, j] += dW[r, j]
            for r in range(3):
                for j in range(n):
                    Z[r, j] = _RT[r][0] * W[0, j] + _RT[r][1] * W[1, j] + _RT[r][2] * W[2, j]
            if dW_norm == 0 or (rate >= 0 and rate / (1 - rate) * dW_norm < self.newton_tol):
                converged = True
                break
            dW_norm_old = dW_norm
            k += 1
        if k == _RADAU_NEWTON_MAXITER:
            k -= 1
        return converged, k + 1, Z, rate

    def _predict_factor(self, h_abs: float, h_abs_old: float, error_norm: float,
                        error_norm_old: float) -> float:
        multiplier = 1.0
        if not (error_norm_old < 0 or h_abs_old < 0 or error_norm == 0):
            multiplier = h_abs / h_abs_old * (error_norm_old / error_norm) ** 0.25
        if error_norm == 0:
            return float('inf')
        return min(1.0, multiplier) * error_norm ** -0.25

    def _sol_at(self, t: float) -> np.ndarray[float, 1]:
        n = self.n
        x = (t - self.sol_t_old) / (self.t - self.sol_t_old)
        out = np.empty(n)
        p1 = x
        p2 = x * x
        p3 = p2 * x
        for j in range(n):
            out[j] = self.sol_Q[j, 0] * p1 + self.sol_Q[j, 1] * p2 + self.sol_Q[j, 2] * p3 + self.sol_y_old[j]
        return out

    def _step_impl(self, F):
        n = self.n
        t = self.t
        y = self.y
        f = self.f
        max_step = self.max_step
        rtol = self.rtol
        atol = self.atol
        min_step = 10 * abs(np.nextafter(t, self.direction * float('inf')) - t)
        h_abs = 0.0
        h_abs_old = -1.0
        error_norm_old = -1.0
        if self.h_abs > max_step:
            h_abs = max_step
        elif self.h_abs < min_step:
            h_abs = min_step
        else:
            h_abs = self.h_abs
            h_abs_old = self.h_abs_old
            error_norm_old = self.error_norm_old
        current_jac = self.current_jac
        rejected = False
        step_accepted = False
        h = 0.0
        t_new = t
        y_new = y
        Z = self.Z
        n_iter = 0
        rate = -1.0
        error_norm = 0.0
        safety = 0.0
        while not step_accepted:
            if h_abs < min_step:
                return False, _TOO_SMALL_STEP
            h = h_abs * self.direction
            t_new = t + h
            if self.direction * (t_new - self.t_bound) > 0:
                t_new = self.t_bound
            h = t_new - t
            h_abs = abs(h)
            Z0 = np.zeros((3, n))
            if self.has_sol:
                for i in range(3):
                    zi = self._sol_at(t + h * _RC[i])
                    for j in range(n):
                        Z0[i, j] = zi[j] - y[j]
            scale = np.empty(n)
            for j in range(n):
                scale[j] = atol[j] + abs(y[j]) * rtol
            converged = False
            while not converged:
                if not self.have_lu:
                    self._factor(h)
                converged, n_iter, Z, rate = self._collocation(F, t, y, h, Z0, scale)
                if not converged:
                    if current_jac:
                        break
                    self.J = self._jac(F, t, y, f)
                    current_jac = True
                    self.have_lu = False
            if not converged:
                h_abs *= 0.5
                self.have_lu = False
                continue
            y_new = np.empty(n)
            ZE = np.empty(n)
            for j in range(n):
                y_new[j] = y[j] + Z[2, j]
                ZE[j] = (Z[0, j] * _RE[0] + Z[1, j] * _RE[1] + Z[2, j] * _RE[2]) / h
            error = np.empty(n)
            for j in range(n):
                error[j] = f[j] + ZE[j]
            _lus(self.LU_real, n, self.piv_real, error)
            for j in range(n):
                scale[j] = atol[j] + max(abs(y[j]), abs(y_new[j])) * rtol
            error_norm = _nrm(error / scale)
            safety = 0.9 * (2 * _RADAU_NEWTON_MAXITER + 1) / (2 * _RADAU_NEWTON_MAXITER + n_iter)
            if rejected and error_norm > 1:
                fe = self.fun(F, t, y + error)
                for j in range(n):
                    error[j] = fe[j] + ZE[j]
                _lus(self.LU_real, n, self.piv_real, error)
                error_norm = _nrm(error / scale)
            if error_norm > 1:
                factor = self._predict_factor(h_abs, h_abs_old, error_norm, error_norm_old)
                h_abs *= max(0.2, safety * factor)
                self.have_lu = False
                rejected = True
            else:
                step_accepted = True
        recompute_jac = (not self.const_jac) and n_iter > 2 and rate > 1e-3
        factor = self._predict_factor(h_abs, h_abs_old, error_norm, error_norm_old)
        factor = min(10.0, safety * factor)
        if not recompute_jac and factor < 1.2:
            factor = 1.0
        else:
            self.have_lu = False
        f_new = self.fun(F, t_new, y_new)
        if recompute_jac:
            self.J = self._jac(F, t_new, y_new, f_new)
            current_jac = True
        elif not self.const_jac:
            current_jac = False
        self.h_abs_old = self.h_abs
        self.error_norm_old = error_norm
        self.h_abs = h_abs * factor
        self.y_old = y
        self.t = t_new
        self.y = y_new
        self.f = f_new
        self.Z = Z
        self.current_jac = current_jac
        self.t_old = t
        # dense output Q = Z.T @ P
        Q = np.empty((n, 3))
        for j in range(n):
            for c in range(3):
                Q[j, c] = Z[0, j] * _RP[0][c] + Z[1, j] * _RP[1][c] + Z[2, j] * _RP[2][c]
        self.sol_Q = Q
        self.sol_t_old = t
        self.sol_y_old = y
        self.has_sol = True
        return True, ''

# ---------------------------------------------------------------- BDF (NDF)
_BDF_MAX_ORDER = 5
_BDF_NEWTON_MAXITER = 4
_KAPPA = [0.0, -0.1850, -1 / 9, -0.0823, -0.0415, 0.0]

def _compute_R(order: int, factor: float) -> np.ndarray[float, 2]:
    M = np.zeros((order + 1, order + 1))
    for j in range(order + 1):
        M[0, j] = 1.0
    for i in range(1, order + 1):
        for j in range(1, order + 1):
            M[i, j] = (i - 1 - factor * j) / i
    for i in range(1, order + 1):
        for j in range(order + 1):
            M[i, j] *= M[i - 1, j]
    return M

def _change_D(D: np.ndarray[float, 2], order: int, factor: float):
    R = _compute_R(order, factor)
    U = _compute_R(order, 1.0)
    m = order + 1
    RU = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            s = 0.0
            for k in range(m):
                s += R[i, k] * U[k, j]
            RU[i, j] = s
    n = D.shape[1]
    tmp = np.zeros((m, n))
    for i in range(m):
        for c in range(n):
            s = 0.0
            for k in range(m):
                s += RU[k, i] * D[k, c]
            tmp[i, c] = s
    for i in range(m):
        for c in range(n):
            D[i, c] = tmp[i, c]

class _BDF:
    n: int
    t: float
    y: np.ndarray[float, 1]
    t_old: float
    t_bound: float
    direction: float
    max_step: float
    rtol: float
    atol: np.ndarray[float, 1]
    h_abs: float
    newton_tol: float
    has_jac: bool
    const_jac: bool
    jac_factor: np.ndarray[float, 1]
    J: np.ndarray[float, 2]
    LU: np.ndarray[float, 2]
    piv: List[int]
    have_lu: bool
    gamma: List[float]
    alpha: List[float]
    error_const: List[float]
    D: np.ndarray[float, 2]
    order: int
    n_equal_steps: int
    status: str
    nfev: int
    njev: int
    nlu: int

    def __init__(self, t0: float, y0: np.ndarray[float, 1], t_bound: float, max_step: float,
                 rtol: float, atol: np.ndarray[float, 1], jac_mode: int):
        n = y0.shape[0]
        self.n = n
        self.t = t0
        self.y = y0
        self.t_old = t0
        self.t_bound = t_bound
        self.direction = 1.0 if t_bound >= t0 else -1.0
        self.max_step = max_step
        self.rtol = rtol
        self.atol = atol
        self.h_abs = 0.0
        self.newton_tol = max(10 * _EPS / rtol, min(0.03, rtol ** 0.5))
        self.has_jac = jac_mode != 0
        self.const_jac = jac_mode == 2
        self.jac_factor = np.zeros(0)
        self.J = np.zeros((n, n))
        self.LU = np.zeros((n, n))
        self.piv = [0] * n
        self.have_lu = False
        g = [0.0]
        acc = 0.0
        for k in range(1, _BDF_MAX_ORDER + 1):
            acc += 1.0 / k
            g.append(acc)
        self.gamma = g
        self.alpha = [(1 - _KAPPA[k]) * g[k] for k in range(_BDF_MAX_ORDER + 1)]
        self.error_const = [_KAPPA[k] * g[k] + 1.0 / (k + 1) for k in range(_BDF_MAX_ORDER + 1)]
        self.D = np.zeros((_BDF_MAX_ORDER + 3, n))
        self.order = 1
        self.n_equal_steps = 0
        self.status = 'running'
        self.nfev = 0
        self.njev = 0
        self.nlu = 0

    def fun(self, F, t: float, y: np.ndarray[float, 1]) -> np.ndarray[float, 1]:
        self.nfev += 1
        return F[0](t, y)

    def _jac(self, F, t: float, y: np.ndarray[float, 1]) -> np.ndarray[float, 2]:
        self.njev += 1
        if not self.has_jac:
            f = F[0](t, y)
            J, fac = _num_jac(F[1], t, y, f, self.atol, self.jac_factor)
            self.jac_factor = fac
            return J
        return F[2](t, y)

    def init_step(self, F, first_step: Optional[float]):
        f = self.fun(F, self.t, self.y)
        if first_step is None:
            fn = lambda tt, yy: self.fun(F, tt, yy)
            self.h_abs = _select_initial_step(fn, self.t, self.y, self.t_bound, self.max_step,
                                              f, self.direction, 1, self.rtol, self.atol)
        else:
            fs = first_step
            if fs <= 0:
                raise ValueError("`first_step` must be positive.")
            if fs > abs(self.t_bound - self.t):
                raise ValueError("`first_step` exceeds bounds.")
            self.h_abs = fs
        if not self.has_jac:
            self.J = self._jac(F, self.t, self.y)
        else:
            self.J = F[2](self.t, self.y)
            self.njev = 0 if self.const_jac else 1
        for j in range(self.n):
            self.D[0, j] = self.y[j]
            self.D[1, j] = f[j] * self.h_abs * self.direction

    def _solve_system(self, F, t_new: float, y_predict: np.ndarray[float, 1], c: float,
                      psi: np.ndarray[float, 1], scale: np.ndarray[float, 1]):
        n = self.n
        d = np.zeros(n)
        y = y_predict.copy()
        dy_norm_old = -1.0
        converged = False
        k = 0
        dy = np.empty(n)
        while k < _BDF_NEWTON_MAXITER:
            f = self.fun(F, t_new, y)
            finite = True
            for j in range(n):
                if not math.isfinite(f[j]):
                    finite = False
            if not finite:
                break
            for j in range(n):
                dy[j] = c * f[j] - psi[j] - d[j]
            _lus(self.LU, n, self.piv, dy)
            dy_norm = _nrm(dy / scale)
            rate = -1.0
            if dy_norm_old >= 0:
                rate = dy_norm / dy_norm_old
            if rate >= 0 and (rate >= 1 or rate ** (_BDF_NEWTON_MAXITER - k) / (1 - rate) * dy_norm > self.newton_tol):
                break
            for j in range(n):
                y[j] += dy[j]
                d[j] += dy[j]
            if dy_norm == 0 or (rate >= 0 and rate / (1 - rate) * dy_norm < self.newton_tol):
                converged = True
                break
            dy_norm_old = dy_norm
            k += 1
        if k == _BDF_NEWTON_MAXITER:
            k -= 1
        return converged, k + 1, y, d

    def _factor(self, c: float):
        n = self.n
        for i in range(n):
            for j in range(n):
                self.LU[i, j] = (1.0 if i == j else 0.0) - c * self.J[i, j]
        _lu(self.LU, n, self.piv)
        self.nlu += 1
        self.have_lu = True

    def _step_impl(self, F):
        n = self.n
        t = self.t
        D = self.D
        max_step = self.max_step
        min_step = 10 * abs(np.nextafter(t, self.direction * float('inf')) - t)
        h_abs = 0.0
        if self.h_abs > max_step:
            h_abs = max_step
            _change_D(D, self.order, max_step / self.h_abs)
            self.n_equal_steps = 0
        elif self.h_abs < min_step:
            h_abs = min_step
            _change_D(D, self.order, min_step / self.h_abs)
            self.n_equal_steps = 0
        else:
            h_abs = self.h_abs
        atol = self.atol
        rtol = self.rtol
        order = self.order
        alpha = self.alpha
        gamma = self.gamma
        error_const = self.error_const
        current_jac = self.const_jac   # scipy: self.jac is None (constant array)
        step_accepted = False
        t_new = t
        y_new = self.y
        d = self.y
        n_iter = 0
        scale = np.empty(n)
        error_norm = 0.0
        safety = 0.0
        while not step_accepted:
            if h_abs < min_step:
                return False, _TOO_SMALL_STEP
            h = h_abs * self.direction
            t_new = t + h
            if self.direction * (t_new - self.t_bound) > 0:
                t_new = self.t_bound
                _change_D(D, order, abs(t_new - t) / h_abs)
                self.n_equal_steps = 0
                self.have_lu = False
            h = t_new - t
            h_abs = abs(h)
            y_predict = np.zeros(n)
            for k in range(order + 1):
                for j in range(n):
                    y_predict[j] += D[k, j]
            for j in range(n):
                scale[j] = atol[j] + rtol * abs(y_predict[j])
            psi = np.zeros(n)
            for k in range(1, order + 1):
                for j in range(n):
                    psi[j] += D[k, j] * gamma[k]
            for j in range(n):
                psi[j] /= alpha[order]
            converged = False
            c = h / alpha[order]
            while not converged:
                if not self.have_lu:
                    self._factor(c)
                converged, n_iter, y_new, d = self._solve_system(F, t_new, y_predict, c, psi, scale)
                if not converged:
                    if current_jac:
                        break
                    self.J = self._jac(F, t_new, y_predict)
                    self.have_lu = False
                    current_jac = True
            if not converged:
                factor = 0.5
                h_abs *= factor
                _change_D(D, order, factor)
                self.n_equal_steps = 0
                self.have_lu = False
                continue
            safety = 0.9 * (2 * _BDF_NEWTON_MAXITER + 1) / (2 * _BDF_NEWTON_MAXITER + n_iter)
            for j in range(n):
                scale[j] = atol[j] + rtol * abs(y_new[j])
            error = error_const[order] * d
            error_norm = _nrm(error / scale)
            if error_norm > 1:
                factor = max(0.2, safety * error_norm ** (-1.0 / (order + 1)))
                h_abs *= factor
                _change_D(D, order, factor)
                self.n_equal_steps = 0
            else:
                step_accepted = True
        self.n_equal_steps += 1
        self.t = t_new
        self.y = y_new
        self.h_abs = h_abs
        for j in range(n):
            D[order + 2, j] = d[j] - D[order + 1, j]
            D[order + 1, j] = d[j]
        for i in range(order, -1, -1):
            for j in range(n):
                D[i, j] += D[i + 1, j]
        if self.n_equal_steps < order + 1:
            return True, ''
        error_m_norm = float('inf')
        error_p_norm = float('inf')
        if order > 1:
            error_m_norm = _nrm(error_const[order - 1] * D[order] / scale)
        if order < _BDF_MAX_ORDER:
            error_p_norm = _nrm(error_const[order + 1] * D[order + 2] / scale)
        norms = [error_m_norm, error_norm, error_p_norm]
        factors = [0.0, 0.0, 0.0]
        for k in range(3):
            if norms[k] == 0:
                factors[k] = float('inf')
            else:
                factors[k] = norms[k] ** (-1.0 / (order + k))
        best = 0
        for k in range(1, 3):
            if factors[k] > factors[best]:
                best = k
        order += best - 1
        self.order = order
        factor = min(10.0, safety * max(factors[0], factors[1], factors[2]))
        self.h_abs *= factor
        _change_D(D, order, factor)
        self.n_equal_steps = 0
        self.have_lu = False
        return True, ''

    def dense_Q(self):
        # BdfDenseOutput data: (h, order, D[:order+1])
        m = self.order + 1
        Dc = np.empty((m, self.n))
        for i in range(m):
            for j in range(self.n):
                Dc[i, j] = self.D[i, j]
        return self.h_abs * self.direction, self.order, Dc

# ---------------------------------------------------------------- LSODA
class _LsodaSolver:
    n: int
    t: float
    y: np.ndarray[float, 1]
    t_old: float
    t_bound: float
    direction: float
    S: _Lsoda
    istate: int
    status: str
    nfev: int
    njev: int
    nlu: int

    def __init__(self, t0: float, y0: np.ndarray[float, 1], t_bound: float, first_step: Optional[float],
                 min_step: float, max_step: float, rtol: float, atol: np.ndarray[float, 1], has_jac: bool):
        n = y0.shape[0]
        self.n = n
        self.t = t0
        self.y = y0.copy()
        self.t_old = t0
        self.t_bound = t_bound
        self.direction = 1.0 if t_bound >= t0 else -1.0
        fs = 0.0
        if first_step is not None:
            fs = first_step
            if fs <= 0:
                raise ValueError("`first_step` must be positive.")
            if fs > abs(t_bound - t0):
                raise ValueError("`first_step` exceeds bounds.")
        fs *= self.direction
        ms = 0.0
        if max_step != float('inf'):
            if max_step <= 0:
                raise ValueError("`max_step` must be positive.")
            ms = max_step
        if min_step < 0:
            raise ValueError("`min_step` must be nonnegative.")
        S = _Lsoda()
        S.jt = 1 if has_jac else 2
        rv = np.empty(n)
        for i in range(n):
            rv[i] = rtol
        S.rtol = rv
        S.atol = atol
        S.iopt = 1
        S.opt_h0 = fs
        S.opt_hmax = ms
        S.opt_hmin = min_step
        S.opt_mxstep = 500
        S.opt_mxordn = 12
        S.opt_mxords = 5
        S.tcrit = t_bound
        self.S = S
        self.istate = 1
        self.status = 'running'
        self.nfev = 0
        self.njev = 0
        self.nlu = 0

    def _step_impl(self, F):
        fcount = lambda tt, yy: self._f(F, tt, yy)
        tn, ist = self.S.run(fcount, F[2], self.y, self.t, self.t_bound, 5, self.istate)
        if ist < 0:
            return False, 'Unexpected istate in LSODA.'
        self.istate = 2
        self.t = tn
        self.njev = self.S.nje
        self.nlu = self.S.nje
        return True, ''

    def _f(self, F, t: float, y: np.ndarray[float, 1]) -> np.ndarray[float, 1]:
        self.nfev += 1
        return F[0](t, y)

    def dense_data(self):
        S = self.S
        n = self.n
        order = S.nqu
        h = S.h
        yh = np.empty((n, order + 1))
        for j in range(order + 1):
            for i in range(n):
                yh[i, j] = S.yh[i + j * n]
        if S.nq < order:
            sc = (h / S.hu) ** order
            for i in range(n):
                yh[i, order] *= sc
        return h, yh

# ---------------------------------------------------------------- shared driver glue
def _base_step(solver, F) -> str:
    # OdeSolver.step
    message = ''
    if solver.n == 0 or solver.t == solver.t_bound:
        solver.t_old = solver.t
        solver.t = solver.t_bound
        solver.status = 'finished'
    else:
        t = solver.t
        success, message = solver._step_impl(F)
        if not success:
            solver.status = 'failed'
        else:
            solver.t_old = t
            if solver.direction * (solver.t - solver.t_bound) >= 0:
                solver.status = 'finished'
    return message
