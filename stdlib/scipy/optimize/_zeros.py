# Ported from scipy 1.18.1 (BSD-3-Clause)

# Scalar root finding: bisect, brentq, brenth, ridder, newton (secant,
# Newton-Raphson, Halley) and root_scalar.

import math
import numpy as np
from ._common import *
from ._common import (_ScalarFunction, _approx_grad, _approx_jac, _bound_val, _call_scalar, _call_vec, _e1, _e2, _eps_for_method, _epsilon, _fd_steps, _has_opt, _norm2, _norm_inf, _old_bound_to_new, _opt_bool, _opt_float, _opt_int, _prepare_bounds_pair, _scalar_eps, _sign, _sqrt, _standardize_bounds, _status_message_maxfev, _status_message_maxiter, _status_message_nan, _status_message_oob, _status_message_pr_loss, _status_message_success, _vec)

_ECONVERGED = 0
_ESIGNERR = -1
_ECONVERR = -2
_EVALUEERR = -3
_iter = 100
_xtol = 2e-12
_rtol = 4 * EPS

class RootResults:
    root: float
    iterations: int
    function_calls: int
    converged: bool
    flag: str
    method: str

    def __init__(self, root: float, iterations: int, function_calls: int,
                 flag: int, method: str):
        self.root = root
        self.iterations = iterations
        self.function_calls = function_calls
        self.converged = flag == _ECONVERGED
        if flag == _ECONVERGED:
            self.flag = 'converged'
        elif flag == _ESIGNERR:
            self.flag = 'sign error'
        elif flag == _ECONVERR:
            self.flag = 'convergence error'
        elif flag == _EVALUEERR:
            self.flag = 'value error'
        else:
            self.flag = 'in progress'
        self.method = method

    def __repr__(self):
        return (f"      converged: {self.converged}\n           flag: {self.flag}\n"
                f" function_calls: {self.function_calls}\n     iterations: {self.iterations}\n"
                f"           root: {self.root}\n         method: {self.method}")

def _signbit(v: float) -> bool:
    return math.copysign(1.0, v) < 0

class _NanRaise[F, A]:
    f: F
    args: A
    calls: int

    def __init__(self, f: F, args: A):
        self.f = f
        self.args = args
        self.calls = 0

    def __call__(self, x: float) -> float:
        fx = float(self.f(x, *self.args))
        self.calls += 1
        if fx != fx:
            raise ValueError(f'The function value at x={x} is NaN; solver cannot continue.')
        return fx

def _check_tols(xtol: float, rtol: float):
    if xtol <= 0:
        raise ValueError(f'xtol too small ({xtol:g} <= 0)')
    if rtol < _rtol:
        raise ValueError(f'rtol too small ({rtol:g} < {_rtol:g})')

def _bracket_result(x: float, funcalls: int, iterations: int, flag: int,
                    disp: bool, full_output: Literal[bool], method: str):
    if flag == _ESIGNERR:
        raise ValueError("f(a) and f(b) must have different signs")
    if flag == _ECONVERR and disp:
        raise RuntimeError(f"Failed to converge after {iterations} iterations.")
    if full_output:
        return x, RootResults(x, iterations, funcalls, flag, method)
    else:
        return x, RootResults(x, iterations, funcalls, flag, method)

def _brent_core(f, xa: float, xb: float, xtol: float, rtol: float, iter: int,
                hyperbolic: bool):
    # returns (x, funcalls, iterations, flag)
    xpre = xa
    xcur = xb
    xblk = 0.0
    fblk = 0.0
    spre = 0.0
    scur = 0.0
    fpre = f(xpre)
    fcur = f(xcur)
    funcalls = 2
    if fpre == 0:
        return xpre, funcalls, 0, _ECONVERGED
    if fcur == 0:
        return xcur, funcalls, 0, _ECONVERGED
    if _signbit(fpre) == _signbit(fcur):
        return 0.0, funcalls, 0, _ESIGNERR
    iterations = 0
    for i in range(iter):
        iterations += 1
        if fpre != 0 and fcur != 0 and (_signbit(fpre) != _signbit(fcur)):
            xblk = xpre
            fblk = fpre
            spre = xcur - xpre
            scur = spre
        if abs(fblk) < abs(fcur):
            xpre = xcur
            xcur = xblk
            xblk = xpre
            fpre = fcur
            fcur = fblk
            fblk = fpre
        delta = (xtol + rtol * abs(xcur)) / 2
        sbis = (xblk - xcur) / 2
        if fcur == 0 or abs(sbis) < delta:
            return xcur, funcalls, iterations, _ECONVERGED
        if abs(spre) > delta and abs(fcur) < abs(fpre):
            stry = 0.0
            if xpre == xblk:
                stry = -fcur * (xcur - xpre) / (fcur - fpre)
            else:
                dpre = (fpre - fcur) / (xpre - xcur)
                dblk = (fblk - fcur) / (xblk - xcur)
                if hyperbolic:
                    stry = -fcur * (fblk - fpre) / (fblk * dpre - fpre * dblk)
                else:
                    stry = -fcur * (fblk * dblk - fpre * dpre) / (dblk * dpre * (fblk - fpre))
            if 2 * abs(stry) < min(abs(spre), 3 * abs(sbis) - delta):
                spre = scur
                scur = stry
            else:
                spre = sbis
                scur = sbis
        else:
            spre = sbis
            scur = sbis
        xpre = xcur
        fpre = fcur
        if abs(scur) > delta:
            xcur += scur
        else:
            xcur += (delta if sbis > 0 else -delta)
        fcur = f(xcur)
        funcalls += 1
    return xcur, funcalls, iterations, _ECONVERR

def _bisect_core(f, xa: float, xb: float, xtol: float, rtol: float, iter: int):
    fa = f(xa)
    fb = f(xb)
    funcalls = 2
    if fa == 0:
        return xa, funcalls, 0, _ECONVERGED
    if fb == 0:
        return xb, funcalls, 0, _ECONVERGED
    if _signbit(fa) == _signbit(fb):
        return 0.0, funcalls, 0, _ESIGNERR
    dm = xb - xa
    iterations = 0
    for i in range(iter):
        iterations += 1
        dm *= .5
        xm = xa + dm
        fm = f(xm)
        funcalls += 1
        if _signbit(fm) == _signbit(fa):
            xa = xm
        if fm == 0 or abs(dm) < xtol + rtol * abs(xm):
            return xm, funcalls, iterations, _ECONVERGED
    return xa, funcalls, iterations, _ECONVERR

def _ridder_core(f, xa: float, xb: float, xtol: float, rtol: float, iter: int):
    tol = xtol + rtol * min(abs(xa), abs(xb))
    fa = f(xa)
    fb = f(xb)
    funcalls = 2
    if fa == 0:
        return xa, funcalls, 0, _ECONVERGED
    if fb == 0:
        return xb, funcalls, 0, _ECONVERGED
    if _signbit(fa) == _signbit(fb):
        return 0.0, funcalls, 0, _ESIGNERR
    iterations = 0
    xn = 0.0
    for i in range(iter):
        iterations += 1
        dm = 0.5 * (xb - xa)
        xm = xa + dm
        funcalls += 1
        fm = f(xm)
        if fm == 0.0:
            return xm, funcalls, iterations, _ECONVERGED
        ratio = fm / fa
        dn = dm * ratio / math.sqrt(ratio * ratio - fb / fa)
        sg = 1.0 if dn > 0. else -1.0
        xn = xm + sg * min(abs(dn), abs(dm) - .5 * tol)
        funcalls += 1
        fn = f(xn)
        if _signbit(fn) != _signbit(fm):
            xa = xn
            fa = fn
            xb = xm
            fb = fm
        elif _signbit(fn) != _signbit(fa):
            xb = xn
            fb = fn
        else:
            xa = xn
            fa = fn
        tol = xtol + rtol * xn
        if fn == 0.0 or abs(xb - xa) < tol:
            return xn, funcalls, iterations, _ECONVERGED
    return xn, funcalls, iterations, _ECONVERR

def _bracket_solver(kind: int, f, a, b, args, xtol: float, rtol: float,
                    maxiter: int, full_output: Literal[bool], disp: bool,
                    method: str):
    if not isinstance(args, Tuple):
        return _bracket_solver(kind, f, a, b, (args,), xtol, rtol, maxiter,
                               full_output, disp, method)
    _check_tols(xtol, rtol)
    if maxiter < 0:
        raise ValueError("maxiter must be >= 0")
    fw = _NanRaise(f, args)
    x = 0.0
    funcalls = 0
    iterations = 0
    flag = 0
    if kind == 0:
        x, funcalls, iterations, flag = _bisect_core(fw, float(a), float(b), xtol, rtol, maxiter)
    elif kind == 1:
        x, funcalls, iterations, flag = _brent_core(fw, float(a), float(b), xtol, rtol, maxiter, False)
    elif kind == 2:
        x, funcalls, iterations, flag = _brent_core(fw, float(a), float(b), xtol, rtol, maxiter, True)
    else:
        x, funcalls, iterations, flag = _ridder_core(fw, float(a), float(b), xtol, rtol, maxiter)
    r, res = _bracket_result(x, funcalls, iterations, flag, disp, full_output, method)
    if full_output:
        return r, res
    else:
        return r

def bisect(f, a, b, args = (), xtol: float = _xtol, rtol: float = _rtol,
           maxiter: int = _iter, full_output: Literal[bool] = False,
           disp: bool = True):
    return _bracket_solver(0, f, a, b, args, xtol, rtol, maxiter, full_output,
                           disp, 'bisect')

def brentq(f, a, b, args = (), xtol: float = _xtol, rtol: float = _rtol,
           maxiter: int = _iter, full_output: Literal[bool] = False,
           disp: bool = True):
    return _bracket_solver(1, f, a, b, args, xtol, rtol, maxiter, full_output,
                           disp, 'brentq')

def brenth(f, a, b, args = (), xtol: float = _xtol, rtol: float = _rtol,
           maxiter: int = _iter, full_output: Literal[bool] = False,
           disp: bool = True):
    return _bracket_solver(2, f, a, b, args, xtol, rtol, maxiter, full_output,
                           disp, 'brenth')

def ridder(f, a, b, args = (), xtol: float = _xtol, rtol: float = _rtol,
           maxiter: int = _iter, full_output: Literal[bool] = False,
           disp: bool = True):
    return _bracket_solver(3, f, a, b, args, xtol, rtol, maxiter, full_output,
                           disp, 'ridder')

def _isclose(a: float, b: float, rtol: float, atol: float) -> bool:
    return abs(a - b) <= atol + rtol * abs(b)

def _newton_scalar(func, x0: float, fprime, args, tol: float, maxiter: int,
                   fprime2, x1: Optional[float], rtol: float, disp: bool):
    # returns (root, funcalls, iterations, flag, method)
    if tol <= 0:
        raise ValueError(f'tol too small ({tol:g} <= 0)')
    if maxiter < 1:
        raise ValueError('maxiter must be greater than 0')
    p0 = x0 * 1.0
    p = p0
    funcalls = 0
    itr = 0
    method = ''
    if fprime is not None:
        method = 'newton'
        for itr in range(maxiter):
            fval = float(func(p0, *args))
            funcalls += 1
            if fval == 0:
                return p0, funcalls, itr, _ECONVERGED, method
            fder = float(fprime(p0, *args))
            funcalls += 1
            if fder == 0:
                msg = 'Derivative was zero.'
                if disp:
                    msg += f' Failed to converge after {itr + 1} iterations, value is {p0}.'
                    raise RuntimeError(msg)
                return p0, funcalls, itr + 1, _ECONVERR, method
            newton_step = fval / fder
            if fprime2 is not None:
                fder2 = float(fprime2(p0, *args))
                funcalls += 1
                method = 'halley'
                adj = newton_step * fder2 / fder / 2
                if abs(adj) < 1:
                    newton_step /= 1.0 - adj
            p = p0 - newton_step
            if _isclose(p, p0, rtol, tol):
                return p, funcalls, itr + 1, _ECONVERGED, method
            p0 = p
    else:
        method = 'secant'
        p1 = 0.0
        if x1 is not None:
            if x1 == x0:
                raise ValueError('x1 and x0 must be different')
            p1 = x1
        else:
            eps = 1e-4
            p1 = x0 * (1 + eps)
            p1 += (eps if p1 >= 0 else -eps)
        q0 = float(func(p0, *args))
        funcalls += 1
        q1 = float(func(p1, *args))
        funcalls += 1
        if abs(q1) < abs(q0):
            p0, p1, q0, q1 = p1, p0, q1, q0
        for itr in range(maxiter):
            if q1 == q0:
                if p1 != p0:
                    msg = f'Tolerance of {p1 - p0} reached.'
                    if disp:
                        msg += f' Failed to converge after {itr + 1} iterations, value is {p1}.'
                        raise RuntimeError(msg)
                p = (p1 + p0) / 2.0
                return p, funcalls, itr + 1, _ECONVERR, method
            else:
                if abs(q1) > abs(q0):
                    p = (-q0 / q1 * p1 + p0) / (1 - q0 / q1)
                else:
                    p = (-q1 / q0 * p0 + p1) / (1 - q1 / q0)
            if _isclose(p, p1, rtol, tol):
                return p, funcalls, itr + 1, _ECONVERGED, method
            p0, q0 = p1, q1
            p1 = p
            q1 = float(func(p1, *args))
            funcalls += 1
    if disp:
        raise RuntimeError(f'Failed to converge after {itr + 1} iterations, value is {p}.')
    return p, funcalls, itr + 1, _ECONVERR, method

def _array_newton(func, x0, fprime, args, tol: float, maxiter: int, fprime2):
    p = np.array(x0, float).flatten()
    failures = np.ones(p.size, bool)
    nz_der = np.ones(p.size, bool)
    if fprime is not None:
        for iteration in range(maxiter):
            fval = np.asarray(func(p, *args), float)
            if not fval.any():
                failures = fval.astype(bool)
                break
            fder = np.asarray(fprime(p, *args), float)
            nz_der = fder != 0
            if not nz_der.any():
                break
            dp = fval[nz_der] / fder[nz_der]
            if fprime2 is not None:
                fder2 = np.asarray(fprime2(p, *args), float)
                dp = dp / (1.0 - 0.5 * dp * fder2[nz_der] / fder[nz_der])
            p = p.copy()
            p[nz_der] -= dp
            failures[nz_der] = np.abs(dp) >= tol
            if not failures[nz_der].any():
                break
    else:
        dx = np.finfo(float).eps ** 0.33
        p1 = p * (1 + dx) + np.where(p >= 0, dx, -dx)
        q0 = np.asarray(func(p, *args), float)
        q1 = np.asarray(func(p1, *args), float)
        active = np.ones(p.size, bool)
        for iteration in range(maxiter):
            nz_der = q1 != q0
            if not nz_der.any():
                p = (p1 + p) / 2.0
                break
            dp = (q1 * (p1 - p))[nz_der] / (q1 - q0)[nz_der]
            p = p.copy()
            p[nz_der] = p1[nz_der] - dp
            active_zero_der = ~nz_der & active
            p[active_zero_der] = (p1 + p)[active_zero_der] / 2.0
            active &= nz_der
            failures[nz_der] = np.abs(dp) >= tol
            if not failures[nz_der].any():
                break
            p1, p = p, p1
            q0 = q1
            q1 = np.asarray(func(p1, *args), float)
    if (failures & nz_der).any():
        all_or_some = 'all' if failures.all() else 'some'
        if fprime is not None and nz_der.all():
            pass
        raise RuntimeError(f'{all_or_some} failed to converge after {maxiter} iterations')
    if not nz_der.all() and fprime is not None:
        raise RuntimeError('some derivatives were zero')
    return p

def newton(func, x0, fprime = None, args = (), tol: float = 1.48e-8,
           maxiter: int = 50, fprime2 = None, x1: Optional[float] = None,
           rtol: float = 0.0, full_output: Literal[bool] = False,
           disp: bool = True):
    if not isinstance(args, Tuple):
        return newton(func, x0, fprime, (args,), tol, maxiter, fprime2, x1,
                      rtol, full_output, disp)
    if isinstance(x0, np.ndarray) or isinstance(x0, List):
        if tol <= 0:
            raise ValueError(f'tol too small ({tol:g} <= 0)')
        return _array_newton(func, x0, fprime, args, tol, maxiter, fprime2)
    else:
        r, fc, it, flag, method = _newton_scalar(func, float(x0), fprime, args,
                                                 tol, maxiter, fprime2, x1,
                                                 rtol, disp)
        if full_output:
            return r, RootResults(r, it, fc, flag, method)
        else:
            return r

class _CountF[F]:
    f: F
    n: int

    def __init__(self, f: F):
        self.f = f
        self.n = 0

    def __call__(self, x, *args):
        self.n += 1
        return self.f(x, *args)

def root_scalar(f, args = (), method: Optional[str] = None, bracket = None,
                fprime = None, fprime2 = None, x0: Optional[float] = None,
                x1: Optional[float] = None, xtol: Optional[float] = None,
                rtol: Optional[float] = None, maxiter: Optional[int] = None,
                options = None):
    if not isinstance(args, Tuple):
        return root_scalar(f, (args,), method, bracket, fprime, fprime2, x0,
                           x1, xtol, rtol, maxiter, options)
    meth = ''
    if method is not None:
        meth = method.lower()
    else:
        if bracket is not None:
            meth = 'brentq'
        elif x0 is not None:
            if fprime is not None:
                if fprime2 is not None:
                    meth = 'halley'
                else:
                    meth = 'newton'
            elif x1 is not None:
                meth = 'secant'
            else:
                meth = 'newton'
    if meth == '':
        raise ValueError('Unable to select a solver as neither bracket nor starting point provided.')
    xt = _opt_float(options, 'xtol', _xtol if xtol is None else xtol)
    rt = _opt_float(options, 'rtol', _rtol if rtol is None else rtol)
    mi = _opt_int(options, 'maxiter', _iter if maxiter is None else maxiter)
    if (meth == 'bisect' or meth == 'brentq' or meth == 'brenth' or
            meth == 'ridder'):
        if bracket is None:
            raise ValueError(f'Bracket needed for {meth}')
        else:
            a = float(bracket[0])
            b = float(bracket[1])
            kind = 0
            if meth == 'brentq':
                kind = 1
            elif meth == 'brenth':
                kind = 2
            elif meth == 'ridder':
                kind = 3
            r, sol = _bracket_solver(kind, f, a, b, args, xt, rt, mi, True,
                                     False, meth)
            return sol
    elif meth == 'secant' or meth == 'newton' or meth == 'halley':
        if x0 is None:
            raise ValueError(f'x0 must not be None for {meth}')
        tol = _opt_float(options, 'xtol', 1.48e-8 if xtol is None else xtol)
        rtol2 = _opt_float(options, 'rtol', 0.0 if rtol is None else rtol)
        mi2 = _opt_int(options, 'maxiter', 50 if maxiter is None else maxiter)
        if meth == 'secant':
            r, fc, it, flag, m2 = _newton_scalar(f, x0, None, args, tol, mi2,
                                                 None, x1, rtol2, False)
            return RootResults(r, it, fc, flag, 'secant')
        elif meth == 'newton':
            if fprime is None:
                def fp(x: float, *a) -> float:
                    fx = float(f(x, *a))
                    h = EPS ** 0.5 * (1.0 if x >= 0 else -1.0) * max(1.0, abs(x))
                    x1h = x + h
                    return (float(f(x1h, *a)) - fx) / (x1h - x)
                r, fc, it, flag, m2 = _newton_scalar(f, x0, fp, args, tol, mi2,
                                                     None, None, rtol2, False)
                return RootResults(r, it, fc, flag, 'newton')
            else:
                r, fc, it, flag, m2 = _newton_scalar(f, x0, fprime, args, tol,
                                                     mi2, None, None, rtol2,
                                                     False)
                return RootResults(r, it, fc, flag, 'newton')
        else:
            if fprime is None:
                raise ValueError(f'fprime must be specified for {meth}')
            if fprime2 is None:
                raise ValueError(f'fprime2 must be specified for {meth}')
            r, fc, it, flag, m2 = _newton_scalar(f, x0, fprime, args, tol, mi2,
                                                 fprime2, None, rtol2, False)
            return RootResults(r, it, fc, flag, 'halley')
    else:
        raise ValueError(f'Unknown solver {meth}')
    return RootResults(0.0, 0, 0, 0, '')
