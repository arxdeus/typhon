# Ported from scipy 1.18.1 (BSD-3-Clause)

# least_squares extras: robust loss functions (least_squares.py), the
# lsmr trust-region helpers (_lsq/common.py) and the dogbox solver
# (_lsq/dogbox.py).

import math
import numpy as np
from numpy.linalg import LinAlgError
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse.linalg import LinearOperator, lsmr
from ._common import INF, _norm2, _norm_inf

EPS = 2.220446049250313e-16

# ----------------------------------------------------------- loss functions

def _loss_rho(name: str, z: np.ndarray[float, 1], cost_only: bool) -> np.ndarray[float, 2]:
    m = z.size
    rho = np.empty((3, m))
    if name == 'huber':
        for i in range(m):
            if z[i] <= 1:
                rho[0, i] = z[i]
            else:
                rho[0, i] = 2 * z[i] ** 0.5 - 1
        if cost_only:
            return rho
        for i in range(m):
            if z[i] <= 1:
                rho[1, i] = 1.0
                rho[2, i] = 0.0
            else:
                rho[1, i] = z[i] ** (-0.5)
                rho[2, i] = -0.5 * z[i] ** (-1.5)
    elif name == 'soft_l1':
        t = 1 + z
        rho[0] = 2 * (t ** 0.5 - 1)
        if cost_only:
            return rho
        rho[1] = t ** (-0.5)
        rho[2] = -0.5 * t ** (-1.5)
    elif name == 'cauchy':
        rho[0] = np.log1p(z)
        if cost_only:
            return rho
        t = 1 + z
        rho[1] = 1 / t
        rho[2] = -1 / t ** 2
    elif name == 'arctan':
        rho[0] = np.arctan(z)
        if cost_only:
            return rho
        t = 1 + z ** 2
        rho[1] = 1 / t
        rho[2] = -2 * z / t ** 2
    else:
        raise ValueError(f"`loss` must be one of ['linear', 'huber', 'soft_l1', 'cauchy', 'arctan'] or a callable.")
    return rho

class _Loss[L]:
    # construct_loss_function; `loss` is a name or a callable z -> (3, m)
    loss: L
    f_scale: float

    def __init__(self, loss: L, f_scale: float):
        self.loss = loss
        self.f_scale = f_scale

    def _rho(self, z: np.ndarray[float, 1], cost_only: bool) -> np.ndarray[float, 2]:
        if isinstance(self.loss, str):
            return _loss_rho(self.loss, z, cost_only)
        else:
            return np.asarray(self.loss(z), float).reshape((3, z.size)).copy()

    def cost(self, f: np.ndarray[float, 1]) -> float:
        z = (f / self.f_scale) ** 2
        rho = self._rho(z, True)
        return 0.5 * self.f_scale ** 2 * float(np.sum(rho[0]))

    def __call__(self, f: np.ndarray[float, 1]) -> np.ndarray[float, 2]:
        z = (f / self.f_scale) ** 2
        rho = self._rho(z, False)
        rho[0] *= self.f_scale ** 2
        rho[2] /= self.f_scale ** 2
        return rho

def _scale_for_robust_loss_function(J: np.ndarray[float, 2], f: np.ndarray[float, 1],
                                    rho: np.ndarray[float, 2]):
    J_scale = rho[1] + 2 * rho[2] * f ** 2
    for i in range(J_scale.size):
        if J_scale[i] < EPS:
            J_scale[i] = EPS
    J_scale = J_scale ** 0.5
    f2 = f * (rho[1] / J_scale)
    return J * J_scale[:, None], f2

# ---------------------------------------------------- lsmr operators

class _RMulMV:
    J: np.ndarray[float, 2]
    d: np.ndarray[float, 1]

    def __init__(self, J: np.ndarray[float, 2], d: np.ndarray[float, 1]):
        self.J = J
        self.d = d

    def __call__(self, x) -> np.ndarray[float, 1]:
        return self.J @ (np.ravel(x) * self.d)

class _RMulRMV:
    J: np.ndarray[float, 2]
    d: np.ndarray[float, 1]

    def __init__(self, J: np.ndarray[float, 2], d: np.ndarray[float, 1]):
        self.J = J
        self.d = d

    def __call__(self, x) -> np.ndarray[float, 1]:
        return self.d * (self.J.T @ np.ravel(x))

class _RegMV:
    # regularized_lsq_operator(right_multiplied_operator(J, d), diag)
    J: np.ndarray[float, 2]
    d: np.ndarray[float, 1]
    diag: np.ndarray[float, 1]

    def __init__(self, J: np.ndarray[float, 2], d: np.ndarray[float, 1],
                 diag: np.ndarray[float, 1]):
        self.J = J
        self.d = d
        self.diag = diag

    def __call__(self, x) -> np.ndarray[float, 1]:
        xv = np.ravel(x)
        return np.concatenate((self.J @ (xv * self.d), self.diag * xv))

class _RegRMV:
    J: np.ndarray[float, 2]
    d: np.ndarray[float, 1]
    diag: np.ndarray[float, 1]
    m: int

    def __init__(self, J: np.ndarray[float, 2], d: np.ndarray[float, 1],
                 diag: np.ndarray[float, 1]):
        self.J = J
        self.d = d
        self.diag = diag
        self.m = J.shape[0]

    def __call__(self, x) -> np.ndarray[float, 1]:
        xv = np.ravel(x)
        x1 = xv[:self.m]
        x2 = xv[self.m:]
        return self.d * (self.J.T @ x1) + self.diag * x2

class _JhOp:
    # dense stand-in for right_multiplied_operator(J, d) when building
    # quadratics: dot(s) = J @ (s * d)
    J: np.ndarray[float, 2]
    d: np.ndarray[float, 1]

    def __init__(self, J: np.ndarray[float, 2], d: np.ndarray[float, 1]):
        self.J = J
        self.d = d

    def __matmul__(self, s: np.ndarray[float, 1]) -> np.ndarray[float, 1]:
        return self.J @ (s * self.d)

def _lsmr_tr(tr_options, key: str, default: float) -> float:
    if tr_options is None:
        return default
    else:
        if key in tr_options:
            return float(tr_options[key])
        return default

def _lsmr_solve(op, b: np.ndarray[float, 1], damp: float, tr_options) -> np.ndarray[float, 1]:
    atol = _lsmr_tr(tr_options, 'atol', 1e-6)
    btol = _lsmr_tr(tr_options, 'btol', 1e-6)
    conlim = _lsmr_tr(tr_options, 'conlim', 1e8)
    mi = int(_lsmr_tr(tr_options, 'maxiter', -1.0))
    if mi < 0:
        r = lsmr(op, b, damp, atol, btol, conlim)
        return np.asarray(r[0], float)
    else:
        r = lsmr(op, b, damp, atol, btol, conlim, mi)
        return np.asarray(r[0], float)

def _solve_trust_region_2d(B: np.ndarray[float, 2], g: np.ndarray[float, 1],
                           Delta: float):
    try:
        R, lower = cho_factor(B)
        p = -cho_solve((R, lower), g)
        if float(np.dot(p, p)) <= Delta ** 2:
            return p, True
    except LinAlgError:
        pass
    a = B[0, 0] * Delta ** 2
    b = B[0, 1] * Delta ** 2
    c = B[1, 1] * Delta ** 2
    d = g[0] * Delta
    f = g[1] * Delta
    coeffs = np.array([-b + d, 2 * (a - c + f), 6 * b, 2 * (-a + c + f), -b - d])
    t = np.roots(coeffs)
    best = INF
    bp = np.zeros(2)
    for tq in t:
        tc = complex(tq)
        if tc.imag == 0:
            tr = tc.real
            p = np.array([Delta * 2 * tr / (1 + tr ** 2), Delta * (1 - tr ** 2) / (1 + tr ** 2)])
            value = 0.5 * float(np.dot(p, B @ p)) + float(np.dot(g, p))
            if value < best:
                best = value
                bp = p
    return bp, False

def _qr2(A: np.ndarray[float, 2]) -> np.ndarray[float, 2]:
    Q, _ = np.linalg.qr(A)
    return Q
