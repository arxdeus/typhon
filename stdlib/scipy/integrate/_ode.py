# Ported from scipy 1.18.1 (BSD-3-Clause)
#
# Legacy `scipy.integrate.ode` interface (scipy/integrate/_ode.py).
# Integrators:
#   'lsoda'          LSODA port (_lsoda.py); results match scipy.
#   'vode'           mapped onto the LSODA engine with the method forced
#                    ('adams' -> Adams only, 'bdf' -> BDF only by limiting the
#                    other family's max order), not the VODE code itself, so
#                    step counts differ from scipy.
#   'dopri5'/'dop853' use the solve_ivp RK45/DOP853 steppers (scipy's
#                    Hairer codes differ in step-size control details).
# Deviations: f_params/jac_params must be numeric (stored as floats, at most
# 4) and f/jac must be plain functions (arity read statically);
# set_solout and complex_ode are not supported; integrator options are
# explicit keyword arguments of set_integrator.

import numpy as np
from _lsoda import _Lsoda, _tol_vec, _ODE_MSGS
from _ivp import _RK

def _callp(f, t: float, y: np.ndarray[float, 1], p: List[float]):
    # dispatch on the static arity of f (params are stored as floats)
    k: Literal[int] = static.len(static.function.args(f))
    if k <= 2:
        return f(t, y)
    elif k == 3:
        return f(t, y, p[0])
    elif k == 4:
        return f(t, y, p[0], p[1])
    elif k == 5:
        return f(t, y, p[0], p[1], p[2])
    else:
        return f(t, y, p[0], p[1], p[2], p[3])

def _call_fp(f, t: float, y: np.ndarray[float, 1], p: List[float], n: int) -> np.ndarray[float, 1]:
    return np.asarray(_callp(f, t, y, p)).astype(float).reshape(n)

def _call_jp(jac, t: float, y: np.ndarray[float, 1], p: List[float], n: int) -> np.ndarray[float, 2]:
    if jac is None:
        return np.zeros((n, n))
    else:
        return np.asarray(_callp(jac, t, y, p)).astype(float).reshape(n, n)

class ode[F, J]:
    f: F
    jac: J
    f_params: List[float]
    jac_params: List[float]
    _y: np.ndarray[float, 1]
    t: float
    name: str
    rtol: float
    atol: float
    nsteps: int
    max_step: float
    min_step: float
    first_step: float
    meth: int
    max_order_ns: int
    max_order_s: int
    S: _Lsoda
    istate: int
    success: int
    h_rk: float
    nfev: int

    def __init__(self, f: F, jac: J = None):
        self.f = f
        self.jac = jac
        self.f_params = List[float]()
        self.jac_params = List[float]()
        self._y = np.zeros(0)
        self.t = 0.0
        self.name = ''
        self.rtol = 1e-6
        self.atol = 1e-12
        self.nsteps = 500
        self.max_step = 0.0
        self.min_step = 0.0
        self.first_step = 0.0
        self.meth = 0
        self.max_order_ns = 12
        self.max_order_s = 5
        self.S = _Lsoda()
        self.istate = 1
        self.success = 1
        self.h_rk = 0.0
        self.nfev = 0

    @property
    def y(self):
        return self._y

    def set_integrator(self, name: str, rtol: Optional[float] = None, atol: Optional[float] = None,
                       nsteps: Optional[int] = None, max_step: float = 0.0, min_step: float = 0.0,
                       first_step: float = 0.0, method: str = 'adams', with_jacobian: bool = False,
                       max_order_ns: int = 12, max_order_s: int = 5, order: int = 12):
        if name not in ('lsoda', 'vode', 'dopri5', 'dop853'):
            raise ValueError(f"No integrator name match with '{name}' or is not available.")
        self.name = name
        rk = name == 'dopri5' or name == 'dop853'
        self.rtol = rtol if rtol is not None else (1e-6 if not rk else 1e-6)
        self.atol = atol if atol is not None else (1e-12 if not rk else 1e-12)
        self.nsteps = nsteps if nsteps is not None else (500 if not rk else 500)
        self.max_step = max_step
        self.min_step = min_step
        self.first_step = first_step
        self.max_order_ns = max_order_ns
        self.max_order_s = max_order_s
        self.meth = 0
        if name == 'vode':
            if method == 'adams':
                self.meth = 1
                self.max_order_ns = min(order, 12)
            elif method == 'bdf':
                self.meth = 2
                self.max_order_s = min(order, 5)
            else:
                raise ValueError(f"Unknown integration method {method}")
        if self._y.shape[0] == 0:
            self.t = 0.0
            self._y = np.zeros(1)
        self._reset()
        return self

    def _reset(self):
        n = self._y.shape[0]
        S = _Lsoda()
        S.jt = 2 if self.jac is None else 1
        rv, rvec = _tol_vec(self.rtol, n, 0.0)
        av, avec = _tol_vec(self.atol, n, 0.0)
        S.rtol = rv
        S.atol = av
        S.iopt = 1
        S.opt_h0 = self.first_step
        S.opt_hmax = self.max_step
        S.opt_hmin = self.min_step
        S.opt_mxstep = self.nsteps
        S.opt_mxordn = self.max_order_ns
        S.opt_mxords = self.max_order_s
        self.S = S
        self.istate = 1
        self.success = 1
        self.h_rk = self.first_step
        self.nfev = 0

    def set_initial_value(self, y, t: float = 0.0):
        if self.name == '':
            self.set_integrator('lsoda')
        self._y = np.asarray(y).astype(float).reshape(-1).copy()
        self.t = float(t)
        self._reset()
        return self

    def set_f_params(self, *args):
        self.f_params = [float(a) for a in args]
        return self

    def set_jac_params(self, *args):
        self.jac_params = [float(a) for a in args]
        return self

    def successful(self) -> bool:
        return self.success == 1

    def get_return_code(self) -> int:
        return self.istate

    def integrate(self, t: float, step: bool = False, relax: bool = False) -> np.ndarray[float, 1]:
        n = self._y.shape[0]
        fp = self.f_params
        jp = self.jac_params
        f = self.f
        jac = self.jac
        fw = lambda tt, yy: _call_fp(f, tt, yy, fp, n)
        jw = lambda tt, yy: _call_jp(jac, tt, yy, jp, n)
        if self.name == 'dopri5' or self.name == 'dop853':
            m = 1 if self.name == 'dopri5' else 2
            ms = float('inf') if self.max_step == 0.0 else self.max_step
            rv, rvec = _tol_vec(self.rtol, n, 0.0)
            av, avec = _tol_vec(self.atol, n, 0.0)
            solver = _RK(m, self.t, self._y.copy(), float(t), ms, rv[0], av)
            if self.h_rk > 0.0 and self.h_rk <= abs(float(t) - self.t):
                solver.f = solver.fun(fw, self.t, self._y)
                solver.h_abs = self.h_rk
            else:
                solver.init_step(fw, None)
            nst = 0
            while solver.status == 'running':
                solver.step(fw)
                nst += 1
                if nst > self.nsteps and solver.status == 'running':
                    solver.status = 'failed'
            self.nfev += solver.nfev
            if solver.status == 'failed':
                self.success = 0
                self.istate = -1
                print(f"UserWarning: {self.name}: larger nsteps is needed")
            else:
                self.success = 1
                self.istate = 1
                self._y = solver.y.copy()
                self.t = solver.t
                self.h_rk = solver.h_abs
            return self._y
        itask = 1
        if self.name == 'vode':
            # only vode supports step/run_relax; scipy falls back to run otherwise
            itask = 2 if step else (3 if relax else 1)
        S = self.S
        if self.meth != 0:
            S.fixed_meth = self.meth
            S.fixed_miter = S.jt
        y = self._y.copy()
        tn, ist = S.run(fw, jw, y, self.t, float(t), itask, self.istate)
        if ist < 0:
            self.istate = ist
            self.success = 0
            msg = _ODE_MSGS[ist] if ist in _ODE_MSGS else f"Unexpected istate={ist}"
            print(f"UserWarning: {self.name}: {msg}")
        else:
            self.istate = 2
            self.success = 1
        self._y = y
        self.t = tn
        return self._y
