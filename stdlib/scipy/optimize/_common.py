# Ported from scipy 1.18.1 (BSD-3-Clause)

# Shared pieces of scipy.optimize: the result class, exceptions, bounds
# handling, finite-difference derivatives and the cached scalar function
# wrapper used by the gradient-based minimizers.

import math
import numpy as np

EPS = 2.220446049250313e-16
INF = math.inf
NAN = math.nan
_epsilon = 1.4901161193847656e-08   # sqrt(np.finfo(float).eps)

_status_message_success = 'Optimization terminated successfully.'
_status_message_maxfev = 'Maximum number of function evaluations has been exceeded.'
_status_message_maxiter = 'Maximum number of iterations has been exceeded.'
_status_message_pr_loss = 'Desired error not necessarily achieved due to precision loss.'
_status_message_nan = 'NaN result encountered.'
_status_message_oob = 'The result is outside of the provided bounds.'

def _e1():
    return np.empty(0, float)

def _e2():
    return np.empty((0, 0), float)

class OptimizeResult[X, F, J]:
    # Concrete (statically typed) counterpart of scipy's dict-like
    # OptimizeResult. Fields that a solver does not produce keep their
    # defaults (0, empty arrays, '').
    x: X
    fun: F
    jac: J
    nit: int
    nfev: int
    njev: int
    success: bool
    status: int
    message: str
    hess_inv: np.ndarray[float, 2]
    # least_squares
    cost: float
    grad: np.ndarray[float, 1]
    optimality: float
    active_mask: np.ndarray[int, 1]
    # root (hybr)
    fjac: np.ndarray[float, 2]
    qtf: np.ndarray[float, 1]
    method: str

    def __init__(self, x: X, fun: F, jac: J, nit: int = 0, nfev: int = 0,
                 njev: int = 0, success: bool = False, status: int = 0,
                 message: str = '', hess_inv: np.ndarray[float, 2] = _e2(),
                 cost: float = 0.0, grad: np.ndarray[float, 1] = _e1(),
                 optimality: float = 0.0,
                 active_mask: np.ndarray[int, 1] = np.empty(0, int),
                 fjac: np.ndarray[float, 2] = _e2(),
                 qtf: np.ndarray[float, 1] = _e1(), method: str = ''):
        self.x = x
        self.fun = fun
        self.jac = jac
        self.nit = nit
        self.nfev = nfev
        self.njev = njev
        self.success = success
        self.status = status
        self.message = message
        self.hess_inv = hess_inv
        self.cost = cost
        self.grad = grad
        self.optimality = optimality
        self.active_mask = active_mask
        self.fjac = fjac
        self.qtf = qtf
        self.method = method

    def __repr__(self):
        return (f" message: {self.message}\n success: {self.success}\n"
                f"  status: {self.status}\n     fun: {self.fun}\n"
                f"       x: {self.x}\n     nit: {self.nit}\n"
                f"    nfev: {self.nfev}")

    def __str__(self):
        return self.__repr__()

class OptimizeWarning(Exception):
    def __init__(self, message: str = ''):
        super().__init__(message)

class BracketError(RuntimeError):
    data: Tuple[float, float, float, float, float, float, int]

    def __init__(self, message: str,
                 data: Tuple[float, float, float, float, float, float, int]):
        super().__init__(message)
        self.data = data

class Bounds:
    lb: np.ndarray[float, 1]
    ub: np.ndarray[float, 1]

    def __init__(self, lb = -float('inf'), ub = float('inf')):
        self.lb = np.atleast_1d(np.asarray(lb, float))
        self.ub = np.atleast_1d(np.asarray(ub, float))

    def __repr__(self):
        return f"Bounds({self.lb}, {self.ub})"

# ---------------------------------------------------------------- utilities

def _sqrt(v: float) -> float:
    # np.sqrt semantics (nan for negative input, no exception)
    if v < 0.0 or v != v:
        return NAN
    return math.sqrt(v)

def _sign(v: float) -> float:
    if v > 0.0:
        return 1.0
    elif v < 0.0:
        return -1.0
    elif v == 0.0:
        return 0.0
    return NAN

def _vec(x0) -> np.ndarray[float, 1]:
    a = np.atleast_1d(np.asarray(x0)).flatten()
    return a.astype(float)

def _norm2(x) -> float:
    # np.linalg.norm for 1-D real input: sqrt(dot(x, x))
    return math.sqrt(float(np.dot(x, x)))

def _norm_inf(x) -> float:
    s = 0.0
    for v in x:
        a = abs(v)
        if a > s or a != a:
            s = a
    return s

def vecnorm(x, ord: float = 2.0) -> float:
    if ord == INF:
        return float(np.amax(np.abs(x)))
    elif ord == -INF:
        return float(np.amin(np.abs(x)))
    else:
        return float(np.sum(np.abs(x) ** ord) ** (1.0 / ord))

def _opt_int(options, key: str, default: int) -> int:
    if options is None:
        return default
    else:
        if key in options:
            return int(options[key])
        return default

def _opt_float(options, key: str, default: float) -> float:
    if options is None:
        return default
    else:
        if key in options:
            return float(options[key])
        return default

def _opt_bool(options, key: str, default: bool) -> bool:
    if options is None:
        return default
    else:
        if key in options:
            return bool(options[key])
        return default

def _has_opt(options, key: str) -> bool:
    if options is None:
        return False
    else:
        return key in options

def _call_scalar(fun, x, args) -> float:
    v = fun(x, *args)
    if isinstance(v, np.ndarray):
        if v.size != 1:
            raise ValueError("The user-provided objective function must return a scalar value.")
        return float(v.flat[0])
    else:
        return float(v)

def _call_vec(fun, x, args) -> np.ndarray[float, 1]:
    return np.atleast_1d(np.asarray(fun(x, *args), float)).flatten()

# ------------------------------------------------------------------- bounds

def _bound_val(v, default: float) -> float:
    if v is None:
        return default
    else:
        if isinstance(v, Optional):
            if v is None:
                return default
            return float(v)
        else:
            return float(v)

def _old_bound_to_new(bounds, n: int):
    # sequence of (min, max) pairs, None meaning unbounded
    lb = np.full(n, -INF)
    ub = np.full(n, INF)
    i = 0
    for b in bounds:
        if i >= n:
            raise ValueError('length of x0 != length of bounds')
        lb[i] = _bound_val(b[0], -INF)
        ub[i] = _bound_val(b[1], INF)
        i += 1
    if i != n:
        raise ValueError('length of x0 != length of bounds')
    return lb, ub

def _standardize_bounds(bounds, n: int):
    if isinstance(bounds, Bounds):
        lb = np.resize(bounds.lb, n).astype(float)
        ub = np.resize(bounds.ub, n).astype(float)
        return lb, ub
    else:
        return _old_bound_to_new(bounds, n)

def _prepare_bounds_pair(bounds, n: int):
    # least_squares/curve_fit style: (lb, ub) with scalars or arrays
    if isinstance(bounds, Bounds):
        return (np.resize(bounds.lb, n).astype(float),
                np.resize(bounds.ub, n).astype(float))
    else:
        if len(bounds) != 2:
            raise ValueError('`bounds` must contain 2 elements.')
        lb = np.atleast_1d(np.asarray(bounds[0], float)).flatten()
        ub = np.atleast_1d(np.asarray(bounds[1], float)).flatten()
        if lb.size == 1:
            lb = np.full(n, lb[0])
        if ub.size == 1:
            ub = np.full(n, ub[0])
        return lb, ub

# ------------------------------------------------------ finite differences

def _eps_for_method(method: str) -> float:
    if method == '2-point' or method == 'cs':
        return 1.4901161193847656e-08
    elif method == '3-point':
        return 6.055454452393343e-06
    raise ValueError(f"Unknown method '{method}'. ")

def _ov(o: Optional[float], d: float) -> float:
    if o is None:
        return d
    else:
        return o

def _fd_steps(x0: np.ndarray[float, 1], method: str, rel_step: Optional[float],
              abs_step: Optional[float], lb: np.ndarray[float, 1],
              ub: np.ndarray[float, 1]):
    # _compute_absolute_step / abs_step handling + _adjust_scheme_to_bounds
    n = x0.size
    rstep = _eps_for_method(method)
    h = np.empty(n, float)
    has_abs = abs_step is not None
    has_rel = rel_step is not None
    absv = _ov(abs_step, 0.0)
    relv = _ov(rel_step, 0.0)
    for i in range(n):
        s = 1.0 if x0[i] >= 0 else -1.0
        default = rstep * s * max(1.0, abs(x0[i]))
        if has_abs:
            hi = absv
            if (x0[i] + hi) - x0[i] == 0.0:
                hi = default
            h[i] = hi
        elif has_rel:
            hi = relv * s * abs(x0[i])
            if (x0[i] + hi) - x0[i] == 0.0:
                hi = default
            h[i] = hi
        else:
            h[i] = default
    one_sided = np.ones(n, bool)
    if method == '3-point':
        h = np.abs(h)
        one_sided = np.zeros(n, bool)
    unbounded = True
    for i in range(n):
        if lb[i] != -INF or ub[i] != INF:
            unbounded = False
    if unbounded:
        return h, one_sided
    num_steps = 1.0
    if method == '2-point':
        for i in range(n):
            lower_dist = x0[i] - lb[i]
            upper_dist = ub[i] - x0[i]
            ht = h[i] * num_steps
            xx = x0[i] + ht
            violated = (xx < lb[i]) or (xx > ub[i])
            fitting = abs(ht) <= max(lower_dist, upper_dist)
            if violated and fitting:
                h[i] = -h[i]
            elif not fitting:
                if upper_dist >= lower_dist:
                    h[i] = upper_dist / num_steps
                else:
                    h[i] = -lower_dist / num_steps
    else:
        for i in range(n):
            lower_dist = x0[i] - lb[i]
            upper_dist = ub[i] - x0[i]
            ht = h[i] * num_steps
            central = (lower_dist >= ht) and (upper_dist >= ht)
            if not central:
                if upper_dist >= lower_dist:
                    h[i] = min(h[i], 0.5 * upper_dist / num_steps)
                else:
                    h[i] = -min(h[i], 0.5 * lower_dist / num_steps)
                one_sided[i] = True
                min_dist = min(upper_dist, lower_dist) / num_steps
                if abs(h[i]) <= min_dist:
                    h[i] = min_dist
                    one_sided[i] = False
    return h, one_sided

def _approx_grad(f, x0: np.ndarray[float, 1], f0: float, method: str = '2-point',
                 rel_step: Optional[float] = None, abs_step: Optional[float] = None,
                 lb: Optional[np.ndarray[float, 1]] = None,
                 ub: Optional[np.ndarray[float, 1]] = None):
    # gradient of a scalar function f(x) -> float; returns (g, nfev)
    n = x0.size
    lo = lb if lb is not None else np.full(n, -INF)
    hi = ub if ub is not None else np.full(n, INF)
    h, one_sided = _fd_steps(x0, method, rel_step, abs_step, lo, hi)
    g = np.empty(n, float)
    nfev = 0
    for i in range(n):
        if method == '2-point':
            x1 = x0.copy()
            x1[i] = x0[i] + h[i]
            dx = x1[i] - x0[i]
            g[i] = (f(x1) - f0) / dx
            nfev += 1
        else:
            x1 = x0.copy()
            x2 = x0.copy()
            if one_sided[i]:
                x1[i] = x0[i] + h[i]
                x2[i] = x0[i] + 2 * h[i]
                f1 = f(x1)
                f2 = f(x2)
                g[i] = (-3.0 * f0 + 4 * f1 - f2) / (x2[i] - x0[i])
            else:
                x1[i] = x0[i] - h[i]
                x2[i] = x0[i] + h[i]
                f1 = f(x1)
                f2 = f(x2)
                g[i] = (f2 - f1) / (x2[i] - x1[i])
            nfev += 2
    return g, nfev

def _approx_jac(f, x0: np.ndarray[float, 1], f0: np.ndarray[float, 1],
                method: str = '2-point', rel_step: Optional[float] = None,
                abs_step: Optional[float] = None,
                lb: Optional[np.ndarray[float, 1]] = None,
                ub: Optional[np.ndarray[float, 1]] = None):
    # Jacobian (m, n) of a vector function f(x) -> 1-D array; returns (J, nfev)
    n = x0.size
    m = f0.size
    lo = lb if lb is not None else np.full(n, -INF)
    hi = ub if ub is not None else np.full(n, INF)
    h, one_sided = _fd_steps(x0, method, rel_step, abs_step, lo, hi)
    J = np.empty((m, n), float)
    nfev = 0
    for i in range(n):
        if method == '2-point':
            x1 = x0.copy()
            x1[i] = x0[i] + h[i]
            dx = x1[i] - x0[i]
            f1 = f(x1)
            for k in range(m):
                J[k, i] = (f1[k] - f0[k]) / dx
            nfev += 1
        else:
            x1 = x0.copy()
            x2 = x0.copy()
            if one_sided[i]:
                x1[i] = x0[i] + h[i]
                x2[i] = x0[i] + 2 * h[i]
                f1 = f(x1)
                f2 = f(x2)
                dx = x2[i] - x0[i]
                for k in range(m):
                    J[k, i] = (-3.0 * f0[k] + 4 * f1[k] - f2[k]) / dx
            else:
                x1[i] = x0[i] - h[i]
                x2[i] = x0[i] + h[i]
                f1 = f(x1)
                f2 = f(x2)
                dx = x2[i] - x1[i]
                for k in range(m):
                    J[k, i] = (f2[k] - f1[k]) / dx
            nfev += 2
    return J, nfev

def approx_fprime(xk, f, epsilon = _epsilon, *args):
    x = _vec(xk)
    f0v = f(x, *args)
    if isinstance(f0v, np.ndarray):
        if f0v.ndim == 0 or f0v.size == 1:
            g, _ = _approx_grad(lambda y: _call_scalar(f, y, args), x,
                                _call_scalar(f, x, args), '2-point', None,
                                _scalar_eps(epsilon))
            return g
        else:
            J, _ = _approx_jac(lambda y: _call_vec(f, y, args), x,
                               _call_vec(f, x, args), '2-point', None,
                               _scalar_eps(epsilon))
            return J
    else:
        g, _ = _approx_grad(lambda y: _call_scalar(f, y, args), x,
                            float(f0v), '2-point', None, _scalar_eps(epsilon))
        return g

def _scalar_eps(epsilon) -> Optional[float]:
    if isinstance(epsilon, np.ndarray):
        return float(epsilon.flat[0])
    else:
        return float(epsilon)

def _rng(seed):
    if seed is None:
        return np.random.default_rng()
    else:
        return np.random.default_rng(int(seed))

def check_grad(func, grad, x0, *args, epsilon: float = _epsilon,
               direction: str = 'all', seed = None):
    x = _vec(x0)
    if direction == 'random':
        _grad = _vec(grad(x, *args))
        rng = _rng(seed)
        v = rng.standard_normal(size=x.shape)
        analytical = float(np.dot(_grad, v))
        g, _ = _approx_grad(lambda w: _call_scalar(func, x + w[0] * v, args),
                            np.zeros(1), _call_scalar(func, x, args),
                            '2-point', None, epsilon)
        d = analytical - g[0]
        return math.sqrt(d * d)
    elif direction == 'all':
        analytical = _vec(grad(x, *args))
        num = approx_fprime(x, func, epsilon, *args)
        return float(np.sqrt(np.sum(np.abs(analytical - num) ** 2)))
    else:
        raise ValueError(f'{direction} is not a valid string for ``direction`` argument')

# ------------------------------------------------ cached scalar function

class _ScalarFunction[FN, JC, A]:
    # scipy's ScalarFunction: caches f and g at the last x and counts
    # evaluations (nfev includes finite-difference evaluations).
    fun_: FN
    jac_: JC
    args: A
    x: np.ndarray[float, 1]
    f: float
    g: np.ndarray[float, 1]
    f_updated: bool
    g_updated: bool
    _nfev: int
    _gnfev: int
    ngev: int
    fd_method: str
    abs_step: Optional[float]
    rel_step: Optional[float]
    lb: Optional[np.ndarray[float, 1]]
    ub: Optional[np.ndarray[float, 1]]

    def __init__(self, fun: FN, x0: np.ndarray[float, 1], args: A, jac: JC,
                 epsilon: Optional[float] = None,
                 finite_diff_rel_step: Optional[float] = None,
                 lb: Optional[np.ndarray[float, 1]] = None,
                 ub: Optional[np.ndarray[float, 1]] = None):
        self.fun_ = fun
        self.jac_ = jac
        self.args = args
        self.x = x0.copy()
        self.f = 0.0
        self.g = np.zeros(x0.size)
        self.f_updated = False
        self.g_updated = False
        self._nfev = 0
        self._gnfev = 0
        self.ngev = 0
        self.lb = lb
        self.ub = ub
        self.rel_step = finite_diff_rel_step
        if jac is None:
            # default: 2-point differences with absolute step epsilon
            self.fd_method = '2-point'
            self.abs_step = epsilon
        elif isinstance(jac, str):
            if jac != '2-point' and jac != '3-point':
                raise ValueError("`grad` must be either callable or one of ('2-point', '3-point', 'cs').")
            self.fd_method = jac
            self.abs_step = None
        else:
            self.fd_method = ''
            self.abs_step = None
        self._update_fun()
        self._update_grad()

    @property
    def nfev(self) -> int:
        return self._nfev + self._gnfev

    def _eval(self, x) -> float:
        return _call_scalar(self.fun_, x.copy(), self.args)

    def _update_x(self, x):
        self.x = np.asarray(x, float).copy()
        self.f_updated = False
        self.g_updated = False

    def _update_fun(self):
        if not self.f_updated:
            self.f = self._eval(self.x)
            self._nfev += 1
            self.f_updated = True

    def _update_grad(self):
        if not self.g_updated:
            if self.jac_ is None or isinstance(self.jac_, str):
                self._update_fun()
                g, nf = _approx_grad(lambda y: self._eval(y), self.x, self.f,
                                     self.fd_method, self.rel_step,
                                     self.abs_step, self.lb, self.ub)
                self._gnfev += nf
                self.g = g
            else:
                self.g = _vec(self.jac_(self.x.copy(), *self.args))
            self.ngev += 1
            self.g_updated = True

    def _same(self, x) -> bool:
        if x.size != self.x.size:
            return False
        for i in range(x.size):
            if x[i] != self.x[i]:
                return False
        return True

    def fun(self, x) -> float:
        if not self._same(x):
            self._update_x(x)
        self._update_fun()
        return self.f

    def grad(self, x) -> np.ndarray[float, 1]:
        if not self._same(x):
            self._update_x(x)
        self._update_grad()
        return self.g

    def fun_and_grad(self, x):
        if not self._same(x):
            self._update_x(x)
        self._update_fun()
        self._update_grad()
        return self.f, self.g
