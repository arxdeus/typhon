# Ported from scipy 1.18.1 (BSD-3-Clause)

# ScalarFunction with Hessian support (_differentiable_functions.py):
# hess may be None (scipy's dummy callable), a callable, a finite-difference
# method string ('2-point', '3-point') giving a Hessian-vector product, or a
# HessianUpdateStrategy. Evaluation counts follow scipy exactly.

import math
import numpy as np
from ._common import _approx_grad, _call_scalar, _vec, INF
from ._hessian_update_strategy import HessianUpdateStrategy

_HESS_NONE = 0
_HESS_CALLABLE = 1
_HESS_FD = 2
_HESS_HUS = 3

def _fd_rel_step(method: str) -> float:
    if method == '3-point':
        return 6.055454452393343e-06
    return 1.4901161193847656e-08

class _ScalarFunctionH[FN, JC, HS, A]:
    fun_: FN
    jac_: JC
    hess_: HS
    args: A
    x: np.ndarray[float, 1]
    f: float
    g: np.ndarray[float, 1]
    H: np.ndarray[float, 2]
    f_updated: bool
    g_updated: bool
    H_updated: bool
    _nfev: int
    _gnfev: int
    ngev: int
    nhev: int
    hmode: int
    fd_method: str
    hfd_method: str
    abs_step: Optional[float]
    rel_step: Optional[float]
    lb: Optional[np.ndarray[float, 1]]
    ub: Optional[np.ndarray[float, 1]]
    x_prev: np.ndarray[float, 1]
    g_prev: np.ndarray[float, 1]

    def __init__(self, fun: FN, x0: np.ndarray[float, 1], args: A, jac: JC,
                 hess: HS, epsilon: Optional[float] = None,
                 finite_diff_rel_step: Optional[float] = None,
                 lb: Optional[np.ndarray[float, 1]] = None,
                 ub: Optional[np.ndarray[float, 1]] = None):
        self.fun_ = fun
        self.jac_ = jac
        self.hess_ = hess
        self.args = args
        self.x = x0.copy()
        n = x0.size
        self.f = 0.0
        self.g = np.zeros(n)
        self.H = np.zeros((0, 0))
        self.f_updated = False
        self.g_updated = False
        self.H_updated = False
        self._nfev = 0
        self._gnfev = 0
        self.ngev = 0
        self.nhev = 0
        self.lb = lb
        self.ub = ub
        self.rel_step = finite_diff_rel_step
        self.x_prev = np.zeros(0)
        self.g_prev = np.zeros(0)
        self.hfd_method = ''
        if jac is None:
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
        if hess is None:
            self.hmode = _HESS_NONE
        elif isinstance(hess, str):
            if hess != '2-point' and hess != '3-point' and hess != 'cs':
                raise ValueError("`hess` must be either callable, HessianUpdateStrategy or one of ('2-point', '3-point', 'cs').")
            if self.fd_method != '':
                raise ValueError('Whenever the gradient is estimated via finite-differences, we require the Hessian to be estimated using one of the quasi-Newton strategies.')
            self.hmode = _HESS_FD
            self.hfd_method = hess if hess != 'cs' else '2-point'
        elif isinstance(hess, HessianUpdateStrategy):
            self.hmode = _HESS_HUS
        else:
            self.hmode = _HESS_CALLABLE
        self._update_fun()
        self._update_grad()
        if self.hmode == _HESS_HUS:
            if isinstance(hess, HessianUpdateStrategy):
                hess.initialize(n, 'hess')
            self.H_updated = True
        elif self.hmode == _HESS_CALLABLE:
            self.H = self._eval_hess(x0.copy())
            self.nhev += 1
            self.H_updated = True
        elif self.hmode == _HESS_NONE:
            # scipy substitutes `def hess(x, *args): return None`, which is
            # evaluated once on construction
            self.nhev += 1
            self.H_updated = True
        else:
            self._update_grad()
            self.H_updated = True

    def _eval_hess(self, x) -> np.ndarray[float, 2]:
        if self.hess_ is None or isinstance(self.hess_, str) or \
           isinstance(self.hess_, HessianUpdateStrategy):
            return np.zeros((0, 0))
        else:
            return np.atleast_2d(np.asarray(self.hess_(x, *self.args), float))

    @property
    def nfev(self) -> int:
        return self._nfev + self._gnfev

    def _eval(self, x) -> float:
        return _call_scalar(self.fun_, x.copy(), self.args)

    def _grad_at(self, x: np.ndarray[float, 1], f0: float) -> np.ndarray[float, 1]:
        # _ScalarGradWrapper.__call__ (counts ngev)
        if self.jac_ is None or isinstance(self.jac_, str):
            g, nf = _approx_grad(lambda y: self._eval(y), x, f0,
                                 self.fd_method, self.rel_step,
                                 self.abs_step, self.lb, self.ub)
            self._gnfev += nf
            self.ngev += 1
            return g
        else:
            self.ngev += 1
            return _vec(self.jac_(x.copy(), *self.args))

    def _update_x(self, x):
        if self.hmode == _HESS_HUS:
            self._update_grad()
            self.x_prev = self.x
            self.g_prev = self.g
            self.x = np.asarray(x, float).copy()
            self.f_updated = False
            self.g_updated = False
            self.H_updated = False
            self._update_hess()
        else:
            self.x = np.asarray(x, float).copy()
            self.f_updated = False
            self.g_updated = False
            self.H_updated = False

    def _update_fun(self):
        if not self.f_updated:
            self.f = self._eval(self.x)
            self._nfev += 1
            self.f_updated = True

    def _update_grad(self):
        if not self.g_updated:
            if self.fd_method != '':
                self._update_fun()
            self.g = self._grad_at(self.x, self.f)
            self.g_updated = True

    def _update_hess(self):
        if not self.H_updated:
            if self.hmode == _HESS_FD:
                self._update_grad()
            elif self.hmode == _HESS_HUS:
                self._update_grad()
                if isinstance(self.hess_, HessianUpdateStrategy):
                    self.hess_.update(self.x - self.x_prev, self.g - self.g_prev)
            elif self.hmode == _HESS_CALLABLE:
                self.H = self._eval_hess(self.x.copy())
                self.nhev += 1
            else:
                self.nhev += 1
            self.H_updated = True

    def _same(self, x) -> bool:
        if x.size != self.x.size:
            return False
        for i in range(x.size):
            if x[i] != self.x[i]:
                return False
        return True

    def _goto(self, x):
        if not self._same(x):
            self._update_x(x)

    def fun(self, x) -> float:
        self._goto(x)
        self._update_fun()
        return self.f

    def grad(self, x) -> np.ndarray[float, 1]:
        self._goto(x)
        self._update_grad()
        return self.g

    def fun_and_grad(self, x):
        self._goto(x)
        self._update_fun()
        self._update_grad()
        return self.f, self.g

    def hess(self, x) -> np.ndarray[float, 2]:
        # dense Hessian (callable hess or quasi-Newton matrix)
        self._goto(x)
        self._update_hess()
        if self.hmode == _HESS_HUS:
            if isinstance(self.hess_, HessianUpdateStrategy):
                return self.hess_.get_matrix()
        return self.H

    def hess_dot(self, x, p) -> np.ndarray[float, 1]:
        # sf.hess(x).dot(p) for every Hessian kind
        self._goto(x)
        self._update_hess()
        if self.hmode == _HESS_FD:
            return self._fd_hessp(np.asarray(p, float))
        elif self.hmode == _HESS_HUS:
            if isinstance(self.hess_, HessianUpdateStrategy):
                return self.hess_.dot(p)
            return np.zeros(0)
        else:
            return np.dot(self.H, np.asarray(p, float))

    def _fd_hessp(self, p: np.ndarray[float, 1]) -> np.ndarray[float, 1]:
        # _linear_operator_difference on the gradient with h = rel_step
        m = self.g.size
        pn = 0.0
        allzero = True
        for v in p:
            if v != 0.0:
                allzero = False
        if allzero:
            return np.zeros(m)
        pn = math.sqrt(float(np.dot(p, p)))
        h = _fd_rel_step(self.hfd_method)
        if self.rel_step is not None:
            h = self.rel_step
        x0 = self.x
        f0 = self.g
        if self.hfd_method == '2-point':
            dx = h / pn
            x = x0 + dx * p
            df = self._grad_at(x, 0.0) - f0
            return df / dx
        else:
            dx = 2 * h / pn
            x1 = x0 - dx / 2 * p
            x2 = x0 + dx / 2 * p
            f1 = self._grad_at(x1, 0.0)
            f2 = self._grad_at(x2, 0.0)
            return (f2 - f1) / dx
