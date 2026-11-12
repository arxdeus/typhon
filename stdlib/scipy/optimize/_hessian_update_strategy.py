# Ported from scipy 1.18.1 (BSD-3-Clause)

# Quasi-Newton Hessian approximations (_hessian_update_strategy.py).
# scipy keeps only the upper triangle (BLAS syr/syr2/symv); this port
# applies the same rank updates to the upper triangle and reads the
# matrix through it, so results agree with scipy to rounding.

import math
import numpy as np

def _syr(alpha: float, x: np.ndarray[float, 1], A: np.ndarray[float, 2]):
    # A := alpha*x*x' + A  (upper triangle)
    n = x.size
    for j in range(n):
        if x[j] != 0.0:
            t = alpha * x[j]
            for i in range(j + 1):
                A[i, j] += x[i] * t

def _syr2(alpha: float, x: np.ndarray[float, 1], y: np.ndarray[float, 1],
          A: np.ndarray[float, 2]):
    # A := alpha*x*y' + alpha*y*x' + A  (upper triangle)
    n = x.size
    for j in range(n):
        if x[j] != 0.0 or y[j] != 0.0:
            t1 = alpha * y[j]
            t2 = alpha * x[j]
            for i in range(j + 1):
                A[i, j] += x[i] * t1 + y[i] * t2

def _symv(A: np.ndarray[float, 2], x: np.ndarray[float, 1]) -> np.ndarray[float, 1]:
    # y := A*x with A symmetric stored in the upper triangle (reference BLAS)
    n = x.size
    y = np.zeros(n)
    for j in range(n):
        t1 = x[j]
        t2 = 0.0
        for i in range(j):
            y[i] += t1 * A[i, j]
            t2 += A[i, j] * x[i]
        y[j] += t1 * A[j, j] + t2
    return y

class HessianUpdateStrategy:
    # Shared state of FullHessianUpdateStrategy. `kind` selects the update
    # rule: 0 = BFGS, 1 = SR1.
    kind: int
    init_scale: float
    auto_scale: bool
    first_iteration: bool
    approx_type: str
    n: int
    B: np.ndarray[float, 2]
    H: np.ndarray[float, 2]
    # BFGS
    exception_strategy: str
    min_curvature: float
    # SR1
    min_denominator: float

    def __init__(self, kind: int = 0):
        self.kind = kind
        self.init_scale = 1.0
        self.auto_scale = True
        self.first_iteration = True
        self.approx_type = 'hess'
        self.n = 0
        self.B = np.empty((0, 0), float)
        self.H = np.empty((0, 0), float)
        self.exception_strategy = 'skip_update'
        self.min_curvature = 1e-08
        self.min_denominator = 1e-08

    def _set_scale(self, init_scale):
        if isinstance(init_scale, str):
            if init_scale != 'auto':
                raise ValueError("init_scale must be 'auto' or a number")
            self.auto_scale = True
        else:
            self.auto_scale = False
            self.init_scale = float(init_scale)

    def initialize(self, n: int, approx_type: str):
        self.first_iteration = True
        self.n = n
        self.approx_type = approx_type
        if approx_type != 'hess' and approx_type != 'inv_hess':
            raise ValueError("`approx_type` must be 'hess' or 'inv_hess'.")
        if approx_type == 'hess':
            self.B = np.eye(n, dtype=float)
        else:
            self.H = np.eye(n, dtype=float)

    def _auto_scale(self, delta_x, delta_grad) -> float:
        s_norm2 = float(np.dot(delta_x, delta_x))
        y_norm2 = float(np.dot(delta_grad, delta_grad))
        ys = abs(float(np.dot(delta_grad, delta_x)))
        if ys == 0.0 or y_norm2 == 0 or s_norm2 == 0:
            return 1.0
        if self.approx_type == 'hess':
            return y_norm2 / ys
        else:
            return ys / y_norm2

    def _M(self) -> np.ndarray[float, 2]:
        return self.B if self.approx_type == 'hess' else self.H

    def update(self, delta_x: np.ndarray[float, 1],
               delta_grad: np.ndarray[float, 1]):
        if np.all(delta_x == 0.0):
            return
        if np.all(delta_grad == 0.0):
            return
        if self.first_iteration:
            scale = self._auto_scale(delta_x, delta_grad) if self.auto_scale \
                else self.init_scale
            if self.approx_type == 'hess':
                self.B *= scale
            else:
                self.H *= scale
            self.first_iteration = False
        if self.kind == 0:
            self._update_bfgs(delta_x, delta_grad)
        else:
            self._update_sr1(delta_x, delta_grad)

    def dot(self, p) -> np.ndarray[float, 1]:
        return _symv(self._M(), np.asarray(p, float))

    def __matmul__(self, p) -> np.ndarray[float, 1]:
        return self.dot(p)

    def get_matrix(self) -> np.ndarray[float, 2]:
        M = self._M().copy()
        for i in range(self.n):
            for j in range(i):
                M[i, j] = M[j, i]
        return M

    def _update_bfgs(self, delta_x: np.ndarray[float, 1],
                     delta_grad: np.ndarray[float, 1]):
        hess = self.approx_type == 'hess'
        w = delta_x if hess else delta_grad
        z = delta_grad if hess else delta_x
        wz = float(np.dot(w, z))
        Mw = self.dot(w)
        wMw = float(Mw.dot(w))
        if wMw <= 0.0:
            scale = self._auto_scale(delta_x, delta_grad)
            if hess:
                self.B = scale * np.eye(self.n, dtype=float)
            else:
                self.H = scale * np.eye(self.n, dtype=float)
            Mw = self.dot(w)
            wMw = float(Mw.dot(w))
        if wz <= self.min_curvature * wMw:
            if self.exception_strategy == 'skip_update':
                return
            elif self.exception_strategy == 'damp_update':
                update_factor = (1 - self.min_curvature) / (1 - wz / wMw)
                z = update_factor * z + (1 - update_factor) * Mw
                wz = float(np.dot(w, z))
        if hess:
            _syr(1.0 / wz, z, self.B)
            _syr(-1.0 / wMw, Mw, self.B)
        else:
            _syr2(-1.0 / wz, z, Mw, self.H)
            _syr((wz + wMw) / wz ** 2, z, self.H)

    def _update_sr1(self, delta_x: np.ndarray[float, 1],
                    delta_grad: np.ndarray[float, 1]):
        hess = self.approx_type == 'hess'
        w = delta_x if hess else delta_grad
        z = delta_grad if hess else delta_x
        Mw = self.dot(w)
        z_minus_Mw = z - Mw
        denominator = float(np.dot(w, z_minus_Mw))
        if abs(denominator) <= self.min_denominator * \
                math.sqrt(float(np.dot(w, w))) * \
                math.sqrt(float(np.dot(z_minus_Mw, z_minus_Mw))):
            return
        if hess:
            _syr(1 / denominator, z_minus_Mw, self.B)
        else:
            _syr(1 / denominator, z_minus_Mw, self.H)

def BFGS(exception_strategy: str = 'skip_update',
         min_curvature: Optional[float] = None, init_scale = 'auto'):
    s = HessianUpdateStrategy(0)
    if exception_strategy == 'skip_update':
        s.min_curvature = min_curvature if min_curvature is not None else 1e-08
    elif exception_strategy == 'damp_update':
        s.min_curvature = min_curvature if min_curvature is not None else 0.2
    else:
        raise ValueError("`exception_strategy` must be 'skip_update' or 'damp_update'.")
    s.exception_strategy = exception_strategy
    s._set_scale(init_scale)
    return s

def SR1(min_denominator: float = 1e-08, init_scale = 'auto'):
    s = HessianUpdateStrategy(1)
    s.min_denominator = min_denominator
    s._set_scale(init_scale)
    return s
