# Ported from scipy 1.18.1 (BSD-3-Clause)

# Native Typhon implementation of a scipy.optimize subset: minimize
# (Nelder-Mead, Powell, CG, BFGS, L-BFGS-B), minimize_scalar, scalar root
# finders, fsolve/root (MINPACK hybrd), least_squares ('trf', 'lm'),
# curve_fit, linear_sum_assignment and finite-difference helpers.

import numpy as np
from ._common import OptimizeResult, OptimizeWarning, BracketError, Bounds, \
                     approx_fprime, check_grad, _vec, _standardize_bounds, \
                     _opt_float, INF
from ._minimize import _minimize_neldermead, _minimize_bfgs, _minimize_cg, \
                       _minimize_powell
from ._lbfgsb import _minimize_lbfgsb
from ._scalar import minimize_scalar, bracket, brent, golden, fminbound
from ._zeros import RootResults, bisect, brentq, brenth, ridder, newton, \
                    root_scalar
from ._lsq import least_squares, curve_fit, fsolve, root
from ._lsap import linear_sum_assignment

def rosen(x):
    x = np.asarray(x)
    r = np.sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0,
               axis=0)
    return r

def rosen_der(x):
    x = np.asarray(x, float)
    xm = x[1:-1]
    xm_m1 = x[:-2]
    xm_p1 = x[2:]
    der = np.zeros_like(x)
    der[1:-1] = (200 * (xm - xm_m1 ** 2) -
                 400 * (xm_p1 - xm ** 2) * xm - 2 * (1 - xm))
    der[0] = -400 * x[0] * (x[1] - x[0] ** 2) - 2 * (1 - x[0])
    der[-1] = 200 * (x[-1] - x[-2] ** 2)
    return der

def rosen_hess(x):
    x = np.atleast_1d(np.asarray(x, float))
    n = x.size
    H = np.zeros((n, n))
    for i in range(n - 1):
        H[i, i + 1] = -400 * x[i]
        H[i + 1, i] = -400 * x[i]
    diagonal = np.zeros(n)
    diagonal[0] = 1200 * x[0] ** 2 - 400 * x[1] + 2
    diagonal[-1] = 200
    for i in range(1, n - 1):
        diagonal[i] = 202 + 1200 * x[i] ** 2 - 400 * x[i + 1]
    for i in range(n):
        H[i, i] = diagonal[i]
    return H

def _apply_tol(meth: str, tol: Optional[float], options, key: str,
               default: float) -> float:
    # value of option `key`, defaulting to `tol` (when the method maps tol
    # to that option) and then to the solver default
    v = default
    if tol is not None:
        if ((key == 'xatol' or key == 'fatol') and meth == 'nelder-mead') or \
           (key == 'xtol' and meth == 'powell') or \
           (key == 'ftol' and (meth == 'powell' or meth == 'l-bfgs-b')) or \
           (key == 'gtol' and (meth == 'bfgs' or meth == 'cg' or meth == 'l-bfgs-b')):
            v = tol
    return _opt_float(options, key, v)

class _Opts[O]:
    # options view with `tol` folded in (scipy: options.setdefault(...))
    opts: O
    meth: str
    tol: Optional[float]

    def __init__(self, opts: O, meth: str, tol: Optional[float]):
        self.opts = opts
        self.meth = meth
        self.tol = tol

    def _tolkey(self, key: str) -> bool:
        if self.tol is None:
            return False
        meth = self.meth
        return (((key == 'xatol' or key == 'fatol') and meth == 'nelder-mead') or
                (key == 'xtol' and meth == 'powell') or
                (key == 'ftol' and (meth == 'powell' or meth == 'l-bfgs-b')) or
                (key == 'gtol' and (meth == 'bfgs' or meth == 'cg' or meth == 'l-bfgs-b')))

    def __contains__(self, key: str) -> bool:
        if self._tolkey(key):
            return True
        if self.opts is None:
            return False
        else:
            return key in self.opts

    def __getitem__(self, key: str) -> float:
        if self.opts is not None:
            if key in self.opts:
                return float(self.opts[key])
        if self._tolkey(key):
            if self.tol is not None:
                return self.tol
        raise KeyError(key)

def minimize(fun, x0, args = (), method: Optional[str] = None, jac = None,
             hess = None, hessp = None, bounds = None, constraints = (),
             tol: Optional[float] = None, callback = None, options = None):
    if not isinstance(args, Tuple):
        return minimize(fun, x0, (args,), method, jac, hess, hessp, bounds,
                        constraints, tol, callback, options)
    if hess is not None or hessp is not None:
        compile_error("hess/hessp are not supported by this scipy.optimize port")
    if callback is not None:
        compile_error("callback is not supported by this scipy.optimize port")
    if static.len(constraints) != 0:
        compile_error("constraints are not supported by this scipy.optimize port")
    xa = np.atleast_1d(np.asarray(x0))
    if xa.ndim != 1:
        raise ValueError("'x0' must only have one dimension.")
    x = _vec(x0)
    meth = ''
    if method is None:
        meth = 'bfgs' if bounds is None else 'l-bfgs-b'
    else:
        meth = method.lower()
    opts = _Opts(options, meth, tol)
    bnds: Optional[Tuple[np.ndarray[float, 1], np.ndarray[float, 1]]] = None
    if bounds is not None:
        lb, ub = _standardize_bounds(bounds, x.size)
        if lb.size != x.size:
            raise ValueError('length of x0 != length of bounds')
        for i in range(x.size):
            if lb[i] > ub[i]:
                raise ValueError("An upper bound is less than the corresponding lower bound.")
        bnds = (lb, ub)
    if meth == 'nelder-mead':
        return _minimize_neldermead(fun, x, args, opts, bnds)
    elif meth == 'powell':
        return _minimize_powell(fun, x, args, opts, bnds)
    elif meth == 'cg':
        return _minimize_cg(fun, x, args, jac, opts)
    elif meth == 'bfgs':
        return _minimize_bfgs(fun, x, args, jac, opts)
    elif meth == 'l-bfgs-b':
        return _minimize_lbfgsb(fun, x, args, jac, bnds, opts)
    else:
        raise ValueError(f'Unknown solver {method}')
    return _minimize_bfgs(fun, x, args, jac, opts)
