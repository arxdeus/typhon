# Ported from scipy 1.18.1 (BSD-3-Clause)

# Legacy fmin_* wrappers over minimize(), leastsq, isotonic_regression,
# rosen_hess_prod and toms748. `disp` messages are not printed.
# Deviation: retall is not supported; leastsq(full_output=True) returns
# (x, cov_x, fvec, nfev, ier) instead of (x, cov_x, infodict, mesg, ier);
# otherwise (x, ier).

import math
import numpy as np
from ._common import OptimizeResult, _e1, _e2, _vec, INF
from ._constraints import NonlinearConstraint
from ._minpack import _lmdif, _leastsq_cov
from ._lsq import _HFun, _HJac, _call_vec

def _od(items: List[Tuple[str, float]]) -> Dict[str, float]:
    d = Dict[str, float]()
    for k, v in items:
        d[k] = v
    return d

def fmin(func, x0, args = (), xtol: float = 1e-4, ftol: float = 1e-4,
         maxiter: Optional[int] = None, maxfun: Optional[int] = None,
         full_output: Literal[bool] = False, disp: bool = True,
         retall: bool = False, callback = None, initial_simplex = None):
    from . import minimize
    items = [('xatol', xtol), ('fatol', ftol)]
    if maxiter is not None:
        items.append(('maxiter', float(maxiter)))
    if maxfun is not None:
        items.append(('maxfev', float(maxfun)))
    opts = _od(items)
    if initial_simplex is not None:
        raise NotImplementedError('fmin: initial_simplex is not supported')
    res = minimize(func, x0, args, 'Nelder-Mead', None, None, None, None, (), None,
                   callback, opts)
    if full_output:
        return res.x, res.fun, res.nit, res.nfev, res.status
    return res.x

def fmin_powell(func, x0, args = (), xtol: float = 1e-4, ftol: float = 1e-4,
                maxiter: Optional[int] = None, maxfun: Optional[int] = None,
                full_output: Literal[bool] = False, disp: bool = True,
                retall: bool = False, callback = None):
    from . import minimize
    items = [('xtol', xtol), ('ftol', ftol)]
    if maxiter is not None:
        items.append(('maxiter', float(maxiter)))
    if maxfun is not None:
        items.append(('maxfev', float(maxfun)))
    res = minimize(func, x0, args, 'Powell', None, None, None, None, (), None,
                   callback, _od(items))
    if full_output:
        return res.x, res.fun, res.hess_inv, res.nit, res.nfev, res.status
    return res.x

def fmin_bfgs(f, x0, fprime = None, args = (), gtol: float = 1e-5,
              norm: float = INF, epsilon: float = 1.4901161193847656e-08,
              maxiter: Optional[int] = None, full_output: Literal[bool] = False,
              disp: bool = True, retall: bool = False, callback = None,
              xrtol: float = 0.0, c1: float = 1e-4, c2: float = 0.9):
    from . import minimize
    items = [('gtol', gtol), ('norm', norm), ('eps', epsilon)]
    if maxiter is not None:
        items.append(('maxiter', float(maxiter)))
    res = minimize(f, x0, args, 'BFGS', fprime, None, None, None, (), None,
                   callback, _od(items))
    if full_output:
        return res.x, res.fun, res.jac, res.hess_inv, res.nfev, res.njev, res.status
    return res.x

def fmin_cg(f, x0, fprime = None, args = (), gtol: float = 1e-5,
            norm: float = INF, epsilon: float = 1.4901161193847656e-08,
            maxiter: Optional[int] = None, full_output: Literal[bool] = False,
            disp: bool = True, retall: bool = False, callback = None,
            c1: float = 1e-4, c2: float = 0.4):
    from . import minimize
    items = [('gtol', gtol), ('norm', norm), ('eps', epsilon)]
    if maxiter is not None:
        items.append(('maxiter', float(maxiter)))
    res = minimize(f, x0, args, 'CG', fprime, None, None, None, (), None,
                   callback, _od(items))
    if full_output:
        return res.x, res.fun, res.nfev, res.njev, res.status
    return res.x

def fmin_ncg(f, x0, fprime, fhess_p = None, fhess = None, args = (),
             avextol: float = 1e-5, epsilon: float = 1.4901161193847656e-08,
             maxiter: Optional[int] = None, full_output: Literal[bool] = False,
             disp: bool = True, retall: bool = False, callback = None,
             c1: float = 1e-4, c2: float = 0.9):
    from . import minimize
    items = [('xtol', avextol), ('eps', epsilon)]
    if maxiter is not None:
        items.append(('maxiter', float(maxiter)))
    res = minimize(f, x0, args, 'Newton-CG', fprime, fhess, fhess_p, None, (), None,
                   callback, _od(items))
    if full_output:
        return res.x, res.fun, res.nfev, res.njev, res.nhev, res.status
    return res.x

def fmin_l_bfgs_b(func, x0, fprime = None, args = (), approx_grad: bool = False,
                  bounds = None, m: int = 10, factr: float = 1e7,
                  pgtol: float = 1e-5, epsilon: float = 1e-8, iprint: int = -1,
                  maxfun: int = 15000, maxiter: int = 15000, disp = None,
                  callback = None, maxls: int = 20):
    from . import minimize
    opts = _od([('maxcor', float(m)), ('ftol', factr * 2.220446049250313e-16),
                ('gtol', pgtol), ('eps', epsilon), ('maxfun', float(maxfun)),
                ('maxiter', float(maxiter)), ('maxls', float(maxls))])
    if approx_grad:
        res = minimize(func, x0, args, 'L-BFGS-B', '2-point', None, None, bounds, (),
                       None, callback, opts)
    else:
        res = minimize(func, x0, args, 'L-BFGS-B', fprime, None, None, bounds, (),
                       None, callback, opts)
    return res.x, res.fun, res.status

def fmin_tnc(func, x0, fprime = None, args = (), approx_grad: bool = False,
             bounds = None, epsilon: float = 1e-8, scale = None, offset = None,
             messages: int = 15, maxCGit: int = -1, maxfun: Optional[int] = None,
             eta: float = -1.0, stepmx: float = 0.0, accuracy: float = 0.0,
             fmin: float = 0.0, ftol: float = -1.0, xtol: float = -1.0,
             pgtol: float = -1.0, rescale: float = -1.0, disp = None,
             callback = None):
    from . import minimize
    items = [('eps', epsilon), ('maxCGit', float(maxCGit)), ('eta', eta),
             ('stepmx', stepmx), ('accuracy', accuracy), ('minfev', fmin),
             ('ftol', ftol), ('xtol', xtol), ('gtol', pgtol), ('rescale', rescale)]
    if maxfun is not None:
        items.append(('maxfun', float(maxfun)))
    opts = _od(items)
    if approx_grad:
        res = minimize(func, x0, args, 'TNC', '2-point', None, None, bounds, (), None,
                       callback, opts)
    else:
        res = minimize(func, x0, args, 'TNC', fprime, None, None, bounds, (), None,
                       callback, opts)
    return res.x, res.nfev, res.status

def _slsq(func, x0, args, fprime, bounds, cons, opts, callback):
    from . import minimize
    if static.len(bounds) == 0:
        return minimize(func, x0, args, 'SLSQP', fprime, None, None, None, cons,
                        None, callback, opts)
    else:
        return minimize(func, x0, args, 'SLSQP', fprime, None, None, bounds, cons,
                        None, callback, opts)

def fmin_slsqp(func, x0, eqcons = (), f_eqcons = None, ieqcons = (),
               f_ieqcons = None, bounds = (), fprime = None, fprime_eqcons = None,
               fprime_ieqcons = None, args = (), iter: int = 100,
               acc: float = 1e-6, iprint: int = 1, disp = None,
               full_output: Literal[bool] = False, epsilon: float = 1.4901161193847656e-08,
               callback = None):
    opts = _od([('maxiter', float(iter)), ('ftol', acc), ('eps', epsilon)])
    res = None
    if f_eqcons is None and f_ieqcons is None:
        res = _slsq(func, x0, args, fprime, bounds, (), opts, callback)
    elif f_ieqcons is None:
        res = _slsq(func, x0, args, fprime, bounds,
                    ({'type': 'eq', 'fun': f_eqcons},), opts, callback)
    elif f_eqcons is None:
        res = _slsq(func, x0, args, fprime, bounds,
                    ({'type': 'ineq', 'fun': f_ieqcons},), opts, callback)
    else:
        res = _slsq(func, x0, args, fprime, bounds,
                    ({'type': 'eq', 'fun': f_eqcons}, {'type': 'ineq', 'fun': f_ieqcons}),
                    opts, callback)
    if full_output:
        return res.x, res.fun, res.nit, res.status, res.message
    return res.x

class _ConsAll[L]:
    fs: L

    def __init__(self, fs: L):
        self.fs = fs

    def __call__(self, x) -> np.ndarray[float, 1]:
        out = List[float]()
        for c in self.fs:
            v = np.atleast_1d(np.asarray(c(x), float)).flatten()
            for t in v:
                out.append(t)
        return np.array(out)

def fmin_cobyla(func, x0, cons, args = (), consargs = None, rhobeg: float = 1.0,
                rhoend: float = 1e-4, maxfun: int = 1000, disp = None,
                catol: float = 2e-4, callback = None):
    from . import minimize
    opts = _od([('rhobeg', rhobeg), ('tol', rhoend), ('maxiter', float(maxfun)),
                ('catol', catol)])
    res = minimize(func, x0, args, 'COBYLA', None, None, None, None,
                   (NonlinearConstraint(_ConsAll(cons), 0.0, INF),), None, callback, opts)
    return res.x

# ------------------------------------------------------------------ leastsq

def leastsq(func, x0, args = (), Dfun = None, full_output: Literal[bool] = False,
            col_deriv: bool = False, ftol: float = 1.49012e-08,
            xtol: float = 1.49012e-08, gtol: float = 0.0, maxfev: int = 0,
            epsfcn: Optional[float] = None, factor: float = 100,
            diag = None):
    if not isinstance(args, Tuple):
        return leastsq(func, x0, (args,), Dfun, full_output, col_deriv, ftol, xtol,
                       gtol, maxfev, epsfcn, factor, diag)
    x = _vec(x0)
    n = x.size
    f0 = _call_vec(func, x.copy(), args)
    m = f0.size
    if n > m:
        raise TypeError(f'Improper input: func input vector length N={n} must not exceed func output vector length M={m}')
    mf = maxfev
    if mf == 0:
        mf = 100 * (n + 1) if Dfun is not None else 200 * (n + 1)
    fl = _HFun(func, args, m)
    jl = _HJac(Dfun, args)
    dg = np.ones(n)
    mode = 1
    if diag is not None:
        dg = _vec(diag)
        mode = 2
    ef = 2.220446049250313e-16 if epsfcn is None else epsfcn
    fvec, fjac, ipvt, qtf, info, nfev, njev = _lmdif(
        fl, jl, x, m, ftol, xtol, gtol, mf, ef, dg, mode, factor, Dfun is not None)
    if full_output:
        ok, cov = _leastsq_cov(fjac, ipvt, m, n)
        return x, cov, fvec, nfev, info
    return x, info

# --------------------------------------------------------------- isotonic

class IsotonicResult:
    x: np.ndarray[float, 1]
    weights: np.ndarray[float, 1]
    blocks: np.ndarray[int, 1]

    def __init__(self, x: np.ndarray[float, 1], weights: np.ndarray[float, 1],
                 blocks: np.ndarray[int, 1]):
        self.x = x
        self.weights = weights
        self.blocks = blocks

    def __getitem__(self, key: Literal[str]):
        return getattr(self, key)

def isotonic_regression(y, weights = None, increasing: bool = True):
    yv = _vec(y)
    n = yv.size
    w = np.ones(n)
    if weights is not None:
        w = _vec(weights)
        if w.size != n:
            raise ValueError('Input arrays y and w must have one dimension of equal length.')
        for i in range(n):
            if not (w[i] > 0):
                raise ValueError('Weights w must be strictly positive.')
    x = yv.copy()
    wx = w.copy()
    if not increasing:
        x = x[::-1].copy()
        wx = wx[::-1].copy()
    r = np.full(n + 1, -1, dtype=int)
    b = 0
    if n > 0:
        # Busing (2022) O(n) PAVA, as scipy's _pava_pybind
        r[0] = 0
        r[1] = 1
        xb_prev = x[0]
        wb_prev = wx[0]
        i = 1
        while i < n:
            b += 1
            xb = x[i]
            wb = wx[i]
            if xb_prev >= xb:
                b -= 1
                sb = wb_prev * xb_prev + wb * xb
                wb += wb_prev
                xb = sb / wb
                while i < n - 1 and xb >= x[i + 1]:
                    i += 1
                    sb += wx[i] * x[i]
                    wb += wx[i]
                    xb = sb / wb
                while b > 0 and x[b - 1] >= xb:
                    b -= 1
                    sb += wx[b] * x[b]
                    wb += wx[b]
                    xb = sb / wb
            x[b] = xb
            xb_prev = xb
            wx[b] = wb
            wb_prev = wb
            r[b + 1] = i + 1
            i += 1
        f = n - 1
        k = b
        while k >= 0:
            t = int(r[k])
            xk = x[k]
            j = f
            while j >= t:
                x[j] = xk
                j -= 1
            f = t - 1
            k -= 1
        b += 1
    rr = r[:b + 1].copy()
    wb2 = wx[:b].copy()
    if not increasing:
        x = x[::-1].copy()
        wb2 = wb2[::-1].copy()
        last = int(rr[rr.size - 1])
        rr = (last - rr[::-1]).copy()
    return IsotonicResult(x, wb2, rr)

# --------------------------------------------------------------------- misc

def rosen_hess_prod(x, p):
    xa = np.atleast_1d(np.asarray(x, float))
    pa = np.atleast_1d(np.asarray(p, float))
    n = xa.size
    Hp = np.zeros(n)
    Hp[0] = (1200 * xa[0] ** 2 - 400 * xa[1] + 2) * pa[0] - 400 * xa[0] * pa[1]
    for i in range(1, n - 1):
        Hp[i] = (-400 * xa[i - 1] * pa[i - 1] +
                 (202 + 1200 * xa[i] ** 2 - 400 * xa[i + 1]) * pa[i] -
                 400 * xa[i] * pa[i + 1])
    Hp[n - 1] = -400 * xa[n - 2] * pa[n - 2] + 200 * pa[n - 1]
    return Hp

# ------------------------------------------------------------------ toms748

class _Tom:
    ab: List[float]
    fab: List[float]
    d: float
    fd: float
    e: float
    fe: float
    has_e: bool
    calls: int
    iterations: int

    def __init__(self):
        self.ab = [math.nan, math.nan]
        self.fab = [math.nan, math.nan]
        self.d = 0.0
        self.fd = 0.0
        self.e = 0.0
        self.fe = 0.0
        self.has_e = False
        self.calls = 0
        self.iterations = 0

def _sgn(v: float) -> float:
    return 1.0 if v > 0 else (-1.0 if v < 0 else 0.0)

def _tcall[F, A](f: F, args: A, st: _Tom, x: float) -> float:
    fx = float(f(x, *args))
    st.calls += 1
    if not math.isfinite(fx):
        raise ValueError(f'Invalid function value: f({x:f}) -> {fx} ')
    return fx

def _tom_update_bracket(st: _Tom, c: float, fc: float):
    idx = 0 if _sgn(st.fab[0]) * _sgn(fc) > 0 else 1
    rx = st.ab[idx]
    rfx = st.fab[idx]
    st.fab[idx] = fc
    st.ab[idx] = c
    return rx, rfx

def _tom_secant(x0: float, x1: float, f0: float, f1: float) -> float:
    if f0 == f1:
        return math.nan
    if abs(f1) > abs(f0):
        return (-f0 / f1 * x1 + x0) / (1 - f0 / f1)
    return (-f1 / f0 * x0 + x1) / (1 - f1 / f0)

def _tom_notclose(fs: List[float], atol: float) -> bool:
    for v in fs:
        if v == 0.0 or not math.isfinite(v):
            return False
    for i in range(len(fs) - 1):
        for j in range(i + 1, len(fs)):
            if abs(fs[i] - fs[j]) <= atol:
                return False
    return True

def _tom_ipz(a: float, b: float, c: float, d: float, fa: float, fb: float,
             fc: float, fd: float) -> float:
    # Neville's algorithm on the inverse data: x as function of f, at 0
    xs = [fa, fb, fc, fd]
    ys = [a, b, c, d]
    N = 4
    Q = np.zeros((N, N))
    D = np.zeros((N, N))
    for i in range(N):
        Q[i, 0] = ys[i]
        D[i, 0] = ys[i]
    for k in range(1, N):
        for i in range(k, N):
            alpha = D[i, k - 1] - Q[i - 1, k - 1]
            diffik = xs[i - k] - xs[i]
            Q[i, k] = (xs[i] - 0.0) / diffik * alpha
            D[i, k] = (xs[i - k] - 0.0) / diffik * alpha
    s = 0.0
    for j in range(1, N):
        s += Q[N - 1, j]
    return s + Q[N - 1, 0]

def _tom_dd(xv: List[float], fv: List[float], forward: bool) -> List[float]:
    n = len(xv)
    row = [v for v in fv]
    dd = [v for v in fv]
    idx = 0 if forward else -1
    dd[0] = fv[0] if forward else fv[n - 1]
    for i in range(1, n):
        m = len(row) - 1
        nr = List[float]()
        for j in range(m):
            nr.append((row[j + 1] - row[j]) / (xv[i + j] - xv[j]))
        row = nr
        dd[i] = row[0] if forward else row[m - 1]
    return dd

def _tom_newton_quad(st: _Tom, d: float, fd: float, k: int) -> float:
    a = st.ab[0]
    b = st.ab[1]
    fa = st.fab[0]
    fb = st.fab[1]
    dd = _tom_dd([a, b, d], [fa, fb, fd], True)
    B = dd[1]
    A = dd[2]
    if A == 0:
        return a - fa / B
    r = a if _sgn(A) * _sgn(fa) > 0 else b
    for i in range(k):
        P = (A * (r - b) + B) * (r - a) + fa
        r1 = r - P / (B + A * (2 * r - a - b))
        if not (a < r1 < b):
            if a < r < b:
                return r
            r = (a + b) / 2.0
            break
        r = r1
    return r

def toms748(f, a: float, b: float, args = (), k: int = 1, xtol: float = 2e-12,
            rtol: float = 8.881784197001252e-16, maxiter: int = 100,
            full_output: Literal[bool] = False, disp: bool = True):
    if not isinstance(args, Tuple):
        return toms748(f, a, b, (args,), k, xtol, rtol, maxiter, full_output, disp)
    return _toms748(f, float(a), float(b), args, k, xtol, rtol, maxiter,
                    full_output, disp)

def _toms748(f, a: float, b: float, args, k: int, xtol: float, rtol: float,
             maxiter: int, full_output: Literal[bool], disp: bool):
    from ._zeros import RootResults
    if xtol <= 0:
        raise ValueError(f'xtol too small ({xtol:g} <= 0)')
    if rtol < 8.881784197001252e-16:
        raise ValueError(f'rtol too small ({rtol:g} < 8.881784197001252e-16)')
    kk = max(k, 1)
    if kk > 100:
        kk = 100
    st = _Tom()
    st.ab[0] = a
    st.ab[1] = b
    if not math.isfinite(a):
        raise ValueError(f'Invalid x value: {a} ')
    if not math.isfinite(b):
        raise ValueError(f'Invalid x value: {b} ')
    fa = _tcall(f, args, st, a)
    if fa == 0:
        return _tom_out(a, st, 0, full_output, RootResults)
    fb = _tcall(f, args, st, b)
    if fb == 0:
        return _tom_out(b, st, 0, full_output, RootResults)
    if _sgn(fb) * _sgn(fa) > 0:
        raise ValueError(f'f(a) and f(b) must have different signs, but f({a:e})={fa:e}, f({b:e})={fb:e} ')
    st.fab[0] = fa
    st.fab[1] = fb
    c = _tom_secant(st.ab[0], st.ab[1], st.fab[0], st.fab[1])
    if not (st.ab[0] < c < st.ab[1]):
        c = (st.ab[0] + st.ab[1]) / 2.0
    fc = _tcall(f, args, st, c)
    if fc == 0:
        return _tom_out(c, st, 0, full_output, RootResults)
    d, fd = _tom_update_bracket(st, c, fc)
    st.d = d
    st.fd = fd
    st.has_e = False
    st.iterations += 1
    eps = 2.220446049250313e-16
    while True:
        st.iterations += 1
        d = st.d
        fd = st.fd
        e = st.e
        fe = st.fe
        has_e = st.has_e
        ab_width = st.ab[1] - st.ab[0]
        c = math.nan
        has_c = False
        for nsteps in range(2, kk + 2):
            if has_e:
                fl = [st.fab[0], st.fab[1], fd, fe]
                if _tom_notclose(fl, 32 * eps):
                    c0 = _tom_ipz(st.ab[0], st.ab[1], d, e, st.fab[0], st.fab[1], fd, fe)
                    if st.ab[0] < c0 < st.ab[1]:
                        c = c0
                        has_c = True
            if not has_c:
                c = _tom_newton_quad(st, d, fd, nsteps)
                has_c = True
            fc = _tcall(f, args, st, c)
            if fc == 0:
                return _tom_out(c, st, 0, full_output, RootResults)
            e = d
            fe = fd
            has_e = True
            d, fd = _tom_update_bracket(st, c, fc)
        uix = 0 if abs(st.fab[0]) < abs(st.fab[1]) else 1
        u = st.ab[uix]
        fu = st.fab[uix]
        dd = _tom_dd([st.ab[0], st.ab[1]], [st.fab[0], st.fab[1]], uix == 0)
        A = dd[1]
        c = u - 2 * fu / A
        if abs(c - u) > 0.5 * (st.ab[1] - st.ab[0]):
            c = (st.ab[0] + st.ab[1]) / 2.0
        else:
            if abs(c - u) <= eps * abs(u):
                m0, e0 = math.frexp(st.fab[0])
                m1, e1 = math.frexp(st.fab[1])
                fr0 = e0
                fr1 = e1
                if (uix == 0 and fr0 < fr1 - 50) or (uix == 1 and fr1 < fr0 - 50):
                    c = (31 * st.ab[uix] + st.ab[1 - uix]) / 32
                else:
                    mm = 1.0 if uix == 0 else -1.0
                    adj = mm * abs(c) * rtol + mm * xtol
                    c = u + adj
                if not (st.ab[0] < c < st.ab[1]):
                    c = (st.ab[0] + st.ab[1]) / 2.0
        fc = _tcall(f, args, st, c)
        if fc == 0:
            return _tom_out(c, st, 0, full_output, RootResults)
        e = d
        fe = fd
        d, fd = _tom_update_bracket(st, c, fc)
        if st.ab[1] - st.ab[0] > 0.5 * ab_width:
            e = d
            fe = fd
            z = (st.ab[0] + st.ab[1]) / 2.0
            fz = _tcall(f, args, st, z)
            if fz == 0:
                return _tom_out(z, st, 0, full_output, RootResults)
            d, fd = _tom_update_bracket(st, z, fz)
        st.d = d
        st.fd = fd
        st.e = e
        st.fe = fe
        st.has_e = True
        a1 = st.ab[0]
        b1 = st.ab[1]
        xn = (a1 + b1) / 2.0
        if abs(a1 - b1) <= xtol + rtol * abs(b1):
            return _tom_out(xn, st, 0, full_output, RootResults)
        if st.iterations >= maxiter:
            if disp:
                raise RuntimeError(f'Failed to converge after {st.iterations + 1} iterations, bracket is [{st.ab[0]}, {st.ab[1]}]')
            return _tom_out(xn, st, -2, full_output, RootResults)

def _tom_out(x: float, st: _Tom, flag: int, full_output: Literal[bool], RR):
    if full_output:
        return x, RR(x, st.iterations, st.calls, flag, 'toms748')
    return x
