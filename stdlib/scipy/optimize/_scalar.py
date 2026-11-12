# Ported from scipy 1.18.1 (BSD-3-Clause)

# Scalar minimization (bracket, Brent, golden section, bounded Brent) and
# minimize_scalar.

import math
import numpy as np
from ._common import *
from ._common import (_ScalarFunction, _approx_grad, _approx_jac, _bound_val, _call_scalar, _call_vec, _e1, _e2, _eps_for_method, _epsilon, _fd_steps, _has_opt, _norm2, _norm_inf, _old_bound_to_new, _opt_bool, _opt_float, _opt_int, _prepare_bounds_pair, _scalar_eps, _sign, _sqrt, _standardize_bounds, _status_message_maxfev, _status_message_maxiter, _status_message_nan, _status_message_oob, _status_message_pr_loss, _status_message_success, _vec)

def bracket(func, xa: float = 0.0, xb: float = 1.0, args = (),
            grow_limit: float = 110.0, maxiter: int = 1000):
    _gold = 1.618034
    _verysmall_num = 1e-21
    xa = float(xa)
    xb = float(xb)
    fa = float(func(xa, *args))
    fb = float(func(xb, *args))
    if fa < fb:
        xa, xb = xb, xa
        fa, fb = fb, fa
    xc = xb + _gold * (xb - xa)
    fc = float(func(xc, *args))
    funcalls = 3
    iter = 0
    while fc < fb:
        tmp1 = (xb - xa) * (fb - fc)
        tmp2 = (xb - xc) * (fb - fa)
        val = tmp2 - tmp1
        denom = 0.0
        if abs(val) < _verysmall_num:
            denom = 2.0 * _verysmall_num
        else:
            denom = 2.0 * val
        w = xb - ((xb - xc) * tmp2 - (xb - xa) * tmp1) / denom
        wlim = xb + grow_limit * (xc - xb)
        if iter > maxiter:
            raise RuntimeError("No valid bracket was found before the iteration limit was reached. Consider trying different initial points or increasing `maxiter`.")
        iter += 1
        fw = 0.0
        if (w - xc) * (xb - w) > 0.0:
            fw = float(func(w, *args))
            funcalls += 1
            if fw < fc:
                xa = xb
                xb = w
                fa = fb
                fb = fw
                break
            elif fw > fb:
                xc = w
                fc = fw
                break
            w = xc + _gold * (xc - xb)
            fw = float(func(w, *args))
            funcalls += 1
        elif (w - wlim) * (wlim - xc) >= 0.0:
            w = wlim
            fw = float(func(w, *args))
            funcalls += 1
        elif (w - wlim) * (xc - w) > 0.0:
            fw = float(func(w, *args))
            funcalls += 1
            if fw < fc:
                xb = xc
                xc = w
                w = xc + _gold * (xc - xb)
                fb = fc
                fc = fw
                fw = float(func(w, *args))
                funcalls += 1
        else:
            w = xc + _gold * (xc - xb)
            fw = float(func(w, *args))
            funcalls += 1
        xa = xb
        xb = xc
        xc = w
        fa = fb
        fb = fc
        fc = fw
    cond1 = (fb < fc and fb <= fa) or (fb < fa and fb <= fc)
    cond2 = (xa < xb < xc) or (xc < xb < xa)
    cond3 = (math.isfinite(xa) and math.isfinite(xb) and math.isfinite(xc))
    if not (cond1 and cond2 and cond3):
        raise BracketError("The algorithm terminated without finding a valid bracket. Consider trying different initial points.",
                           (xa, xb, xc, fa, fb, fc, funcalls))
    return xa, xb, xc, fa, fb, fc, funcalls

def _get_bracket(func, brack, args):
    if brack is None:
        return bracket(func, args=args)
    else:
        if len(brack) == 2:
            return bracket(func, xa=float(brack[0]), xb=float(brack[1]), args=args)
        elif len(brack) == 3:
            xa = float(brack[0])
            xb = float(brack[1])
            xc = float(brack[2])
            if xa > xc:
                xc, xa = xa, xc
            if not (xa < xb and xb < xc):
                raise ValueError("Bracketing values (xa, xb, xc) do not fulfill this requirement: (xa < xb) and (xb < xc)")
            fa = float(func(xa, *args))
            fb = float(func(xb, *args))
            fc = float(func(xc, *args))
            if not (fb < fa and fb < fc):
                raise ValueError("Bracketing values (xa, xb, xc) do not fulfill this requirement: (f(xb) < f(xa)) and (f(xb) < f(xc))")
            return xa, xb, xc, fa, fb, fc, 3
        else:
            raise ValueError('Bracketing interval must be length 2 or 3 sequence.')

def _success_msg(xtol: float) -> str:
    return ("\nOptimization terminated successfully;\nThe returned value satisfies "
            f"the termination criteria\n(using xtol = {xtol} )")

def _minimize_scalar_brent_impl(func, brack, args, xtol: float = 1.48e-08,
                                maxiter: int = 500):
    tol = xtol
    if tol < 0:
        raise ValueError(f'tolerance should be >= 0, got {tol}')
    xa, xb, xc, fa, fb, fc, funcalls = _get_bracket(func, brack, args)
    _mintol = 1.0e-11
    _cg = 0.3819660
    x = xb
    w = xb
    v = xb
    fw = fb
    fv = fb
    fx = fb
    a = 0.0
    b = 0.0
    if xa < xc:
        a = xa
        b = xc
    else:
        a = xc
        b = xa
    deltax = 0.0
    rat = 0.0
    iter = 0
    while iter < maxiter:
        tol1 = tol * abs(x) + _mintol
        tol2 = 2.0 * tol1
        xmid = 0.5 * (a + b)
        if abs(x - xmid) < (tol2 - 0.5 * (b - a)):
            break
        if abs(deltax) <= tol1:
            if x >= xmid:
                deltax = a - x
            else:
                deltax = b - x
            rat = _cg * deltax
        else:
            tmp1 = (x - w) * (fx - fv)
            tmp2 = (x - v) * (fx - fw)
            p = (x - v) * tmp2 - (x - w) * tmp1
            tmp2 = 2.0 * (tmp2 - tmp1)
            if tmp2 > 0.0:
                p = -p
            tmp2 = abs(tmp2)
            dx_temp = deltax
            deltax = rat
            if ((p > tmp2 * (a - x)) and (p < tmp2 * (b - x)) and
                    (abs(p) < abs(0.5 * tmp2 * dx_temp))):
                rat = p * 1.0 / tmp2
                u = x + rat
                if (u - a) < tol2 or (b - u) < tol2:
                    if xmid - x >= 0:
                        rat = tol1
                    else:
                        rat = -tol1
            else:
                if x >= xmid:
                    deltax = a - x
                else:
                    deltax = b - x
                rat = _cg * deltax
        u = 0.0
        if abs(rat) < tol1:
            if rat >= 0:
                u = x + tol1
            else:
                u = x - tol1
        else:
            u = x + rat
        fu = float(func(u, *args))
        funcalls += 1
        if fu > fx:
            if u < x:
                a = u
            else:
                b = u
            if fu <= fw or w == x:
                v = w
                w = u
                fv = fw
                fw = fu
            elif fu <= fv or v == x or v == w:
                v = u
                fv = fu
        else:
            if u >= x:
                a = x
            else:
                b = x
            v = w
            w = x
            x = u
            fv = fw
            fw = fx
            fx = fu
        iter += 1
    success = iter < maxiter and not (x != x or fx != fx)
    message = ''
    if success:
        message = _success_msg(xtol)
    else:
        if iter >= maxiter:
            message = "\nMaximum number of iterations exceeded"
        if x != x or fx != fx:
            message = _status_message_nan
    return OptimizeResult(x, fx, 0.0, nit=iter, nfev=funcalls,
                          success=success, message=message)

def _minimize_scalar_golden_impl(func, brack, args,
                                 xtol: float = _epsilon, maxiter: int = 5000):
    tol = xtol
    xa, xb, xc, fa, fb, fc, funcalls = _get_bracket(func, brack, args)
    _gR = 0.61803399
    _gC = 1.0 - _gR
    x3 = xc
    x0 = xa
    x1 = 0.0
    x2 = 0.0
    if abs(xc - xb) > abs(xb - xa):
        x1 = xb
        x2 = xb + _gC * (xc - xb)
    else:
        x2 = xb
        x1 = xb - _gC * (xb - xa)
    f1 = float(func(x1, *args))
    f2 = float(func(x2, *args))
    funcalls += 2
    nit = 0
    for i in range(maxiter):
        if abs(x3 - x0) <= tol * (abs(x1) + abs(x2)):
            break
        if f2 < f1:
            x0 = x1
            x1 = x2
            x2 = _gR * x1 + _gC * x3
            f1 = f2
            f2 = float(func(x2, *args))
        else:
            x3 = x2
            x2 = x1
            x1 = _gR * x2 + _gC * x0
            f2 = f1
            f1 = float(func(x1, *args))
        funcalls += 1
        nit += 1
    xmin = x2
    fval = f2
    if f1 < f2:
        xmin = x1
        fval = f1
    success = nit < maxiter and not (fval != fval or xmin != xmin)
    message = ''
    if success:
        message = _success_msg(xtol)
    else:
        if nit >= maxiter:
            message = "\nMaximum number of iterations exceeded"
        if xmin != xmin or fval != fval:
            message = _status_message_nan
    return OptimizeResult(xmin, fval, 0.0, nit=nit, nfev=funcalls,
                          success=success, message=message)

def _minimize_scalar_bounded_impl(func, x1: float, x2: float, args,
                                  xatol: float = 1e-05, maxiter: int = 500):
    maxfun = maxiter
    if not (math.isfinite(x1) and math.isfinite(x2)):
        raise ValueError('Optimization bounds must be finite scalars.')
    if x1 > x2:
        raise ValueError('The lower bound exceeds the upper bound.')
    flag = 0
    sqrt_eps = math.sqrt(2.2e-16)
    golden_mean = 0.5 * (3.0 - math.sqrt(5.0))
    a = x1
    b = x2
    fulc = a + golden_mean * (b - a)
    nfc = fulc
    xf = fulc
    rat = 0.0
    e = 0.0
    x = xf
    fx = float(func(x, *args))
    num = 1
    fu = INF
    ffulc = fx
    fnfc = fx
    xm = 0.5 * (a + b)
    tol1 = sqrt_eps * abs(xf) + xatol / 3.0
    tol2 = 2.0 * tol1
    while abs(xf - xm) > (tol2 - 0.5 * (b - a)):
        golden = True
        if abs(e) > tol1:
            golden = False
            r = (xf - nfc) * (fx - ffulc)
            q = (xf - fulc) * (fx - fnfc)
            p = (xf - fulc) * q - (xf - nfc) * r
            q = 2.0 * (q - r)
            if q > 0.0:
                p = -p
            q = abs(q)
            r = e
            e = rat
            if ((abs(p) < abs(0.5 * q * r)) and (p > q * (a - xf)) and
                    (p < q * (b - xf))):
                rat = (p + 0.0) / q
                x = xf + rat
                if ((x - a) < tol2) or ((b - x) < tol2):
                    si = _sign(xm - xf) + (1.0 if (xm - xf) == 0 else 0.0)
                    rat = tol1 * si
            else:
                golden = True
        if golden:
            if xf >= xm:
                e = a - xf
            else:
                e = b - xf
            rat = golden_mean * e
        si = _sign(rat) + (1.0 if rat == 0 else 0.0)
        x = xf + si * max(abs(rat), tol1)
        fu = float(func(x, *args))
        num += 1
        if fu <= fx:
            if x >= xf:
                a = xf
            else:
                b = xf
            fulc, ffulc = nfc, fnfc
            nfc, fnfc = xf, fx
            xf, fx = x, fu
        else:
            if x < xf:
                a = x
            else:
                b = x
            if (fu <= fnfc) or (nfc == xf):
                fulc, ffulc = nfc, fnfc
                nfc, fnfc = x, fu
            elif (fu <= ffulc) or (fulc == xf) or (fulc == nfc):
                fulc, ffulc = x, fu
        xm = 0.5 * (a + b)
        tol1 = sqrt_eps * abs(xf) + xatol / 3.0
        tol2 = 2.0 * tol1
        if num >= maxfun:
            flag = 1
            break
    if xf != xf or fx != fx or fu != fu:
        flag = 2
    msg = 'Solution found.'
    if flag == 1:
        msg = 'Maximum number of function calls reached.'
    elif flag == 2:
        msg = _status_message_nan
    return OptimizeResult(xf, fx, 0.0, nit=num, nfev=num, status=flag,
                          success=(flag == 0), message=msg)

def _recover_brent(func, brack, args, xtol: float, maxiter: int, golden: bool):
    try:
        if golden:
            return _minimize_scalar_golden_impl(func, brack, args, xtol, maxiter)
        else:
            return _minimize_scalar_brent_impl(func, brack, args, xtol, maxiter)
    except BracketError as e:
        xa, xb, xc, fa, fb, fc, funcalls = e.data
        x = NAN
        fun = NAN
        if not (xa != xa or xb != xb or xc != xc or fa != fa or fb != fb or fc != fc):
            x = xa
            fun = fa
            if fb < fun:
                x = xb
                fun = fb
            if fc < fun:
                x = xc
                fun = fc
        return OptimizeResult(x, fun, 0.0, nit=0, nfev=funcalls,
                              success=False, message=str(e))

def minimize_scalar(fun, bracket = None, bounds = None, args = (),
                    method: Optional[str] = None, tol: Optional[float] = None,
                    options = None):
    if not isinstance(args, Tuple):
        return minimize_scalar(fun, bracket, bounds, (args,), method, tol, options)
    meth = ''
    if method is None:
        meth = 'brent' if bounds is None else 'bounded'
    else:
        meth = method.lower()
    if bounds is not None and (meth == 'brent' or meth == 'golden'):
        raise ValueError(f"Use of `bounds` is incompatible with 'method={method}'.")
    if meth == 'brent':
        xtol = _opt_float(options, 'xtol', 1.48e-08 if tol is None else tol)
        maxiter = _opt_int(options, 'maxiter', 500)
        return _recover_brent(fun, bracket, args, xtol, maxiter, False)
    elif meth == 'golden':
        xtol = _opt_float(options, 'xtol', _epsilon if tol is None else tol)
        maxiter = _opt_int(options, 'maxiter', 5000)
        return _recover_brent(fun, bracket, args, xtol, maxiter, True)
    elif meth == 'bounded':
        if bounds is None:
            raise ValueError('The `bounds` parameter is mandatory for method `bounded`.')
        else:
            if len(bounds) != 2:
                raise ValueError('bounds must have two elements.')
            xatol = _opt_float(options, 'xatol', 1e-05 if tol is None else tol)
            maxiter = _opt_int(options, 'maxiter', 500)
            return _minimize_scalar_bounded_impl(fun, float(bounds[0]),
                                                 float(bounds[1]), args, xatol,
                                                 maxiter)
    else:
        raise ValueError(f'Unknown solver {method}')
    return _minimize_scalar_bounded_impl(fun, 0.0, 1.0, args)

def brent(func, args = (), brack = None, tol: float = 1.48e-8,
          full_output: Literal[bool] = False, maxiter: int = 500):
    res = _minimize_scalar_brent_impl(func, brack, args, tol, maxiter)
    if full_output:
        return res.x, res.fun, res.nit, res.nfev
    else:
        return res.x

def golden(func, args = (), brack = None, tol: float = _epsilon,
           full_output: Literal[bool] = False, maxiter: int = 5000):
    res = _minimize_scalar_golden_impl(func, brack, args, tol, maxiter)
    if full_output:
        return res.x, res.fun, res.nfev
    else:
        return res.x

def fminbound(func, x1: float, x2: float, args = (), xtol: float = 1e-5,
              maxfun: int = 500, full_output: Literal[bool] = False):
    res = _minimize_scalar_bounded_impl(func, float(x1), float(x2), args, xtol,
                                        maxfun)
    if full_output:
        return res.x, res.fun, res.status, res.nfev
    else:
        return res.x
