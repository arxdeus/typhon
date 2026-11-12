# Ported from scipy 1.18.1 (BSD-3-Clause)

# brute (grid search + optional polishing) from scipy/optimize/_optimize.py
# plus a minimal show_options.
# Deviations: `finish` defaults to the string 'fmin' (scipy.optimize.fmin,
# i.e. Nelder-Mead with fmin's defaults); None disables polishing; a
# callable finish must be called as finish(func, x0, args=args) and return
# an object with x, fun, success. The return value is always a 1-D x
# array (scipy returns a scalar for 1 variable without finish). With
# full_output=True the grid is returned flattened as an (N, prod(Ns))
# array and Jout as a 1-D array in C order (scipy keeps the N-D shapes).

import math
import numpy as np
from ._common import _call_scalar
from ._minimize import _minimize_neldermead

def _axis(r, Ns: int) -> np.ndarray[float, 1]:
    # np.mgrid semantics for slice(lo, hi, step) / (lo, hi) with complex(Ns)
    lo = 0.0
    hi = 0.0
    st = 1.0
    if isinstance(r, slice):
        lo = float(r.start)
        hi = float(r.stop)
        if r.step is not None:
            st = float(r.step)
    elif static.len(r) == 2:
        return np.linspace(float(r[0]), float(r[1]), Ns)
    else:
        lo = float(r[0])
        hi = float(r[1])
        st = float(r[2])
    if True:
        num = int(math.ceil((hi - lo) / st))
        out = np.empty(max(num, 0))
        for i in range(out.size):
            out[i] = lo + i * st
        return out

def brute(func, ranges, args = (), Ns: int = 20,
          full_output: Literal[bool] = False, finish = 'fmin',
          disp: bool = False, workers: int = 1):
    if not isinstance(args, Tuple):
        return brute(func, ranges, (args,), Ns, full_output, finish, disp,
                     workers)
    N = len(ranges)
    if N > 40:
        raise ValueError("Brute Force not possible with more than 40 variables.")
    axes = [_axis(r, Ns) for r in ranges]
    shape = [a.size for a in axes]
    total = 1
    for s in shape:
        total *= s
    grid = np.empty((N, total))
    Jout = np.empty(total)
    idx = [0] * N
    x = np.empty(N)
    best = 0
    for t in range(total):
        for k in range(N):
            x[k] = axes[k][idx[k]]
            grid[k, t] = x[k]
        Jout[t] = _call_scalar(func, x.copy(), args)
        # advance C-order multi-index
        k = N - 1
        while k >= 0:
            idx[k] += 1
            if idx[k] < shape[k]:
                break
            idx[k] = 0
            k -= 1
    # np.argmin: first minimum, NaN wins
    best = 0
    for t in range(total):
        if Jout[best] != Jout[best]:
            break
        if Jout[t] < Jout[best] or Jout[t] != Jout[t]:
            best = t
    xmin = grid[:, best].copy()
    Jmin = float(Jout[best])
    if isinstance(finish, NoneType):
        pass
    elif isinstance(finish, str):
        if finish != 'fmin':
            raise ValueError("finish must be None, 'fmin' or a callable")
        res = _minimize_neldermead(func, xmin, args, None, None)
        xmin = res.x
        Jmin = res.fun
        if not res.success and disp:
            print("Either final optimization did not succeed or `finish` does not return `statuscode` as its last argument.")
    else:
        res = finish(func, xmin, args=args)
        xmin = np.asarray(res.x, float).flatten()
        Jmin = float(res.fun)
    if full_output:
        return xmin, Jmin, grid, Jout
    else:
        return xmin

def show_options(solver: Optional[str] = None, method: Optional[str] = None,
                 disp: bool = True) -> str:
    # Minimal: lists the solvers / methods available in this port.
    txt = ('minimize: nelder-mead, powell, cg, bfgs, newton-cg, l-bfgs-b, tnc, '
           'cobyla, slsqp, trust-constr, dogleg, trust-ncg, trust-exact, '
           'trust-krylov\nroot: hybr, lm\nminimize_scalar: brent, bounded, '
           'golden\nroot_scalar: bisect, brentq, brenth, ridder, newton, '
           'secant, halley\nlinprog: highs, highs-ds, highs-ipm')
    if disp:
        print(txt)
        return ''
    return txt
