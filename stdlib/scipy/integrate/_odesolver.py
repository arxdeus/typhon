# Ported from scipy 1.18.1 (BSD-3-Clause)
#
# Public OdeSolver classes RK23, RK45, DOP853, Radau, BDF, LSODA with the
# step() / dense_output() API (scipy/integrate/_ivp/*.py), implemented over
# the steppers used by solve_ivp (_ivp.py, _ivp_implicit.py).
# Deviations: `fun` must return a real array; `jac_sparsity`, `lband`/`uband`
# are not supported; complex y0 is not supported.

import math
import numpy as np
from _ivp import _RK, _Interp, _jmat, _jac_mode, _EPS
from _ivp_implicit import _Radau, _BDF, _LsodaSolver, _base_step

def _f1(fun, vectorized: bool, n: int, t: float, y: np.ndarray[float, 1]) -> np.ndarray[float, 1]:
    if vectorized:
        Y = np.empty((n, 1))
        for i in range(n):
            Y[i, 0] = y[i]
        r = np.asarray(fun(t, Y)).astype(float)
        return r.reshape(n)
    r = np.asarray(fun(t, y)).astype(float)
    if r.size != n:
        raise ValueError("fun returned an array of wrong size")
    return r.reshape(n)

def _fv(fun, vectorized: bool, n: int, t: float, Y: np.ndarray[float, 2]) -> np.ndarray[float, 2]:
    k = Y.shape[1]
    if vectorized:
        r = np.asarray(fun(t, Y)).astype(float)
        return r.reshape(n, k)
    out = np.empty((n, k))
    for c in range(k):
        yc = np.empty(n)
        for i in range(n):
            yc[i] = Y[i, c]
        rc = np.asarray(fun(t, yc)).astype(float).reshape(n)
        for i in range(n):
            out[i, c] = rc[i]
    return out

def _prep(y0, rtol: float, atol, max_step: float):
    ya = np.asarray(y0).astype(float)
    if ya.ndim != 1:
        raise ValueError("`y0` must be 1-dimensional.")
    n = ya.shape[0]
    yv = ya.reshape(n).copy()
    for i in range(n):
        if not math.isfinite(yv[i]):
            raise ValueError("All components of the initial state `y0` must be finite.")
    if max_step <= 0:
        raise ValueError("`max_step` must be positive.")
    at = np.asarray(atol).astype(float)
    atv = np.empty(n)
    if at.ndim == 0:
        for i in range(n):
            atv[i] = at.item()
    else:
        if at.ndim != 1 or at.size != n:
            raise ValueError("`atol` has wrong shape.")
        af = at.reshape(n)
        for i in range(n):
            atv[i] = af[i]
    for i in range(n):
        if atv[i] < 0:
            raise ValueError("`atol` must be positive.")
    return yv, atv

class DenseOutput:
    t_old: float
    t: float
    t_min: float
    t_max: float
    _i: _Interp

    def __init__(self, it: _Interp):
        self._i = it
        self.t_old = it.t_old
        self.t = it.t
        self.t_min = min(it.t_old, it.t)
        self.t_max = max(it.t_old, it.t)

    def __call__(self, t):
        if isinstance(t, float) or isinstance(t, int):
            return self._i.at(float(t))
        ta = np.asarray(t).astype(float)
        if ta.ndim == 0:
            return self._i.at(ta.item())
        k = ta.shape[0]
        n = self._i.y_old.shape[0]
        out = np.empty((n, k))
        for i in range(k):
            v = self._i.at(ta[i])
            for j in range(n):
                out[j, i] = v[j]
        return out

class OdeSolver:
    pass


class RK23[F](OdeSolver):
    fun: F
    n: int
    t: float
    y: np.ndarray[float, 1]
    t_old: float
    t_bound: float
    direction: float
    status: str
    nfev: int
    njev: int
    nlu: int
    vectorized: bool
    _s: _RK

    def __init__(self, fun: F, t0: float, y0, t_bound: float, max_step: float = float('inf'),
                 rtol: float = 1e-3, atol = 1e-6, vectorized: bool = False, first_step: Optional[float] = None):
        yv, atv = _prep(y0, rtol, atol, max_step)
        rt = rtol
        if rt < 100 * _EPS:
            rt = 100 * _EPS
        self.fun = fun
        self.vectorized = vectorized
        self.n = yv.shape[0]
        self._s = _RK(0, t0, yv, t_bound, max_step, rt, atv)
        self._s.init_step(self._F()[0], first_step)
        self.t = t0
        self.y = self._s.y
        self.t_old = t0
        self.t_bound = t_bound
        self.direction = self._s.direction
        self.status = 'running'
        self.nfev = self._s.nfev
        self.njev = 0
        self.nlu = 0

    def _F(self):
        return (lambda t, y: _f1(self.fun, self.vectorized, self.n, t, y),
                lambda t, Y: _fv(self.fun, self.vectorized, self.n, t, Y),
                lambda t, y: _jmat(None, (), self.n, t, y))

    def _sync(self):
        self.t = self._s.t
        self.y = self._s.y.copy()
        self.t_old = self._s.t_old
        self.status = self._s.status
        self.nfev = self._s.nfev
        self.njev = self._s.njev
        self.nlu = self._s.nlu

    @property
    def step_size(self) -> float:
        return abs(self.t - self.t_old)

    def step(self) -> Optional[str]:
        if self.status != 'running':
            raise RuntimeError("Attempt to step on a failed or finished solver.")
        msg = self._s.step(self._F()[0])
        self._sync()
        if msg == '':
            return None
        return msg

    def dense_output(self) -> DenseOutput:
        if self.status == 'running' and self.t_old == self.t:
            raise RuntimeError("Dense output is available after a successful step was made.")
        return DenseOutput(self._s.dense_output(self._F()[0]))

class RK45[F](OdeSolver):
    fun: F
    n: int
    t: float
    y: np.ndarray[float, 1]
    t_old: float
    t_bound: float
    direction: float
    status: str
    nfev: int
    njev: int
    nlu: int
    vectorized: bool
    _s: _RK

    def __init__(self, fun: F, t0: float, y0, t_bound: float, max_step: float = float('inf'),
                 rtol: float = 1e-3, atol = 1e-6, vectorized: bool = False, first_step: Optional[float] = None):
        yv, atv = _prep(y0, rtol, atol, max_step)
        rt = rtol
        if rt < 100 * _EPS:
            rt = 100 * _EPS
        self.fun = fun
        self.vectorized = vectorized
        self.n = yv.shape[0]
        self._s = _RK(1, t0, yv, t_bound, max_step, rt, atv)
        self._s.init_step(self._F()[0], first_step)
        self.t = t0
        self.y = self._s.y
        self.t_old = t0
        self.t_bound = t_bound
        self.direction = self._s.direction
        self.status = 'running'
        self.nfev = self._s.nfev
        self.njev = 0
        self.nlu = 0

    def _F(self):
        return (lambda t, y: _f1(self.fun, self.vectorized, self.n, t, y),
                lambda t, Y: _fv(self.fun, self.vectorized, self.n, t, Y),
                lambda t, y: _jmat(None, (), self.n, t, y))

    def _sync(self):
        self.t = self._s.t
        self.y = self._s.y.copy()
        self.t_old = self._s.t_old
        self.status = self._s.status
        self.nfev = self._s.nfev
        self.njev = self._s.njev
        self.nlu = self._s.nlu

    @property
    def step_size(self) -> float:
        return abs(self.t - self.t_old)

    def step(self) -> Optional[str]:
        if self.status != 'running':
            raise RuntimeError("Attempt to step on a failed or finished solver.")
        msg = self._s.step(self._F()[0])
        self._sync()
        if msg == '':
            return None
        return msg

    def dense_output(self) -> DenseOutput:
        if self.status == 'running' and self.t_old == self.t:
            raise RuntimeError("Dense output is available after a successful step was made.")
        return DenseOutput(self._s.dense_output(self._F()[0]))

class DOP853[F](OdeSolver):
    fun: F
    n: int
    t: float
    y: np.ndarray[float, 1]
    t_old: float
    t_bound: float
    direction: float
    status: str
    nfev: int
    njev: int
    nlu: int
    vectorized: bool
    _s: _RK

    def __init__(self, fun: F, t0: float, y0, t_bound: float, max_step: float = float('inf'),
                 rtol: float = 1e-3, atol = 1e-6, vectorized: bool = False, first_step: Optional[float] = None):
        yv, atv = _prep(y0, rtol, atol, max_step)
        rt = rtol
        if rt < 100 * _EPS:
            rt = 100 * _EPS
        self.fun = fun
        self.vectorized = vectorized
        self.n = yv.shape[0]
        self._s = _RK(2, t0, yv, t_bound, max_step, rt, atv)
        self._s.init_step(self._F()[0], first_step)
        self.t = t0
        self.y = self._s.y
        self.t_old = t0
        self.t_bound = t_bound
        self.direction = self._s.direction
        self.status = 'running'
        self.nfev = self._s.nfev
        self.njev = 0
        self.nlu = 0

    def _F(self):
        return (lambda t, y: _f1(self.fun, self.vectorized, self.n, t, y),
                lambda t, Y: _fv(self.fun, self.vectorized, self.n, t, Y),
                lambda t, y: _jmat(None, (), self.n, t, y))

    def _sync(self):
        self.t = self._s.t
        self.y = self._s.y.copy()
        self.t_old = self._s.t_old
        self.status = self._s.status
        self.nfev = self._s.nfev
        self.njev = self._s.njev
        self.nlu = self._s.nlu

    @property
    def step_size(self) -> float:
        return abs(self.t - self.t_old)

    def step(self) -> Optional[str]:
        if self.status != 'running':
            raise RuntimeError("Attempt to step on a failed or finished solver.")
        msg = self._s.step(self._F()[0])
        self._sync()
        if msg == '':
            return None
        return msg

    def dense_output(self) -> DenseOutput:
        if self.status == 'running' and self.t_old == self.t:
            raise RuntimeError("Dense output is available after a successful step was made.")
        return DenseOutput(self._s.dense_output(self._F()[0]))

class Radau[F, J](OdeSolver):
    fun: F
    jac: J
    n: int
    t: float
    y: np.ndarray[float, 1]
    t_old: float
    t_bound: float
    direction: float
    status: str
    nfev: int
    njev: int
    nlu: int
    vectorized: bool
    _s: _Radau

    def __init__(self, fun: F, t0: float, y0, t_bound: float, max_step: float = float('inf'),
                 rtol: float = 1e-3, atol = 1e-6, vectorized: bool = False, first_step: Optional[float] = None, jac: J = None):
        yv, atv = _prep(y0, rtol, atol, max_step)
        rt = rtol
        if rt < 100 * _EPS:
            rt = 100 * _EPS
        self.fun = fun
        self.jac = jac
        self.vectorized = vectorized
        self.n = yv.shape[0]
        self._s = _Radau(t0, yv, t_bound, max_step, rt, atv, _jac_mode(jac))
        self._s.init_step(self._F(), first_step)
        self.t = t0
        self.y = self._s.y
        self.t_old = t0
        self.t_bound = t_bound
        self.direction = self._s.direction
        self.status = 'running'
        self.nfev = self._s.nfev
        self.njev = 0
        self.nlu = 0

    def _F(self):
        return (lambda t, y: _f1(self.fun, self.vectorized, self.n, t, y),
                lambda t, Y: _fv(self.fun, self.vectorized, self.n, t, Y),
                lambda t, y: _jmat(self.jac, (), self.n, t, y))

    def _sync(self):
        self.t = self._s.t
        self.y = self._s.y.copy()
        self.t_old = self._s.t_old
        self.status = self._s.status
        self.nfev = self._s.nfev
        self.njev = self._s.njev
        self.nlu = self._s.nlu

    @property
    def step_size(self) -> float:
        return abs(self.t - self.t_old)

    def step(self) -> Optional[str]:
        if self.status != 'running':
            raise RuntimeError("Attempt to step on a failed or finished solver.")
        msg = _base_step(self._s, self._F())
        self._sync()
        if msg == '':
            return None
        return msg

    def dense_output(self) -> DenseOutput:
        if self.status == 'running' and self.t_old == self.t:
            raise RuntimeError("Dense output is available after a successful step was made.")
        s = self._s
        if s.t == s.t_old:
            return DenseOutput(_Interp(s.t_old, s.t, s.y, np.zeros((0, 0)), 2))
        return DenseOutput(_Interp(s.t_old, s.t, s.sol_y_old, s.sol_Q, 3))

class BDF[F, J](OdeSolver):
    fun: F
    jac: J
    n: int
    t: float
    y: np.ndarray[float, 1]
    t_old: float
    t_bound: float
    direction: float
    status: str
    nfev: int
    njev: int
    nlu: int
    vectorized: bool
    _s: _BDF

    def __init__(self, fun: F, t0: float, y0, t_bound: float, max_step: float = float('inf'),
                 rtol: float = 1e-3, atol = 1e-6, vectorized: bool = False, first_step: Optional[float] = None, jac: J = None):
        yv, atv = _prep(y0, rtol, atol, max_step)
        rt = rtol
        if rt < 100 * _EPS:
            rt = 100 * _EPS
        self.fun = fun
        self.jac = jac
        self.vectorized = vectorized
        self.n = yv.shape[0]
        self._s = _BDF(t0, yv, t_bound, max_step, rt, atv, _jac_mode(jac))
        self._s.init_step(self._F(), first_step)
        self.t = t0
        self.y = self._s.y
        self.t_old = t0
        self.t_bound = t_bound
        self.direction = self._s.direction
        self.status = 'running'
        self.nfev = self._s.nfev
        self.njev = 0
        self.nlu = 0

    def _F(self):
        return (lambda t, y: _f1(self.fun, self.vectorized, self.n, t, y),
                lambda t, Y: _fv(self.fun, self.vectorized, self.n, t, Y),
                lambda t, y: _jmat(self.jac, (), self.n, t, y))

    def _sync(self):
        self.t = self._s.t
        self.y = self._s.y.copy()
        self.t_old = self._s.t_old
        self.status = self._s.status
        self.nfev = self._s.nfev
        self.njev = self._s.njev
        self.nlu = self._s.nlu

    @property
    def step_size(self) -> float:
        return abs(self.t - self.t_old)

    def step(self) -> Optional[str]:
        if self.status != 'running':
            raise RuntimeError("Attempt to step on a failed or finished solver.")
        msg = _base_step(self._s, self._F())
        self._sync()
        if msg == '':
            return None
        return msg

    def dense_output(self) -> DenseOutput:
        if self.status == 'running' and self.t_old == self.t:
            raise RuntimeError("Dense output is available after a successful step was made.")
        s = self._s
        if s.t == s.t_old:
            return DenseOutput(_Interp(s.t_old, s.t, s.y, np.zeros((0, 0)), 2))
        hh, order, Dc = s.dense_Q()
        it = _Interp(s.t_old, s.t, s.y, Dc, 4)
        it.hh = hh
        return DenseOutput(it)

class LSODA[F, J](OdeSolver):
    fun: F
    jac: J
    n: int
    t: float
    y: np.ndarray[float, 1]
    t_old: float
    t_bound: float
    direction: float
    status: str
    nfev: int
    njev: int
    nlu: int
    vectorized: bool
    _s: _LsodaSolver

    def __init__(self, fun: F, t0: float, y0, t_bound: float, first_step: Optional[float] = None,
                 min_step: float = 0.0, max_step: float = float('inf'), rtol: float = 1e-3, atol = 1e-6,
                 jac: J = None, vectorized: bool = False):
        yv, atv = _prep(y0, rtol, atol, max_step)
        rt = rtol
        if rt < 100 * _EPS:
            rt = 100 * _EPS
        self.fun = fun
        self.jac = jac
        self.vectorized = vectorized
        self._s = _LsodaSolver(t0, yv, t_bound, first_step, min_step, max_step, rt, atv, _jac_mode(jac) != 0)
        self.n = yv.shape[0]
        self.t = t0
        self.y = self._s.y.copy()
        self.t_old = t0
        self.t_bound = t_bound
        self.direction = self._s.direction
        self.status = 'running'
        self.nfev = 0
        self.njev = 0
        self.nlu = 0

    def _F(self):
        return (lambda t, y: _f1(self.fun, self.vectorized, self.n, t, y),
                lambda t, Y: _fv(self.fun, self.vectorized, self.n, t, Y),
                lambda t, y: _jmat(self.jac, (), self.n, t, y))

    @property
    def step_size(self) -> float:
        return abs(self.t - self.t_old)

    def step(self) -> Optional[str]:
        if self.status != 'running':
            raise RuntimeError("Attempt to step on a failed or finished solver.")
        msg = _base_step(self._s, self._F())
        self.t = self._s.t
        self.y = self._s.y.copy()
        self.t_old = self._s.t_old
        self.status = self._s.status
        self.nfev = self._s.nfev
        self.njev = self._s.njev
        self.nlu = self._s.nlu
        if msg == '':
            return None
        return msg

    def dense_output(self) -> DenseOutput:
        s = self._s
        if s.t == s.t_old:
            return DenseOutput(_Interp(s.t_old, s.t, s.y, np.zeros((0, 0)), 2))
        hh, yh = s.dense_data()
        it = _Interp(s.t_old, s.t, s.y, yh, 5)
        it.hh = hh
        return DenseOutput(it)
