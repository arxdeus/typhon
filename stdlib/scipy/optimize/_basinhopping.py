# Ported from scipy 1.18.1 (BSD-3-Clause)

# basinhopping translated from scipy/optimize/_basinhopping.py.
# Deviations: the local minimizer is scipy.optimize.minimize called with the
# fields of a typed `MinimizerKwargs(method=, jac=, bounds=, args=, tol=,
# options=)` value (a plain dict may only carry 'method'); `accept_test` is called with keyword args f_new, x_new, f_old,
# x_old and must return bool (string 'force accept' is not supported);
# the result has no lowest_optimization_result field.

import math
import numpy as np
from ._common import OptimizeResult

class RandomDisplacement[R]:
    stepsize: float
    rng: R

    def __init__(self, stepsize: float = 0.5, rng: R = None):
        self.stepsize = stepsize
        self.rng = rng

    def __call__(self, x: np.ndarray[float, 1]) -> np.ndarray[float, 1]:
        u = np.asarray(self.rng.uniform(-self.stepsize, self.stepsize, x.shape), float)
        x += u
        return x

class AdaptiveStepsize[T]:
    takestep: T
    target_accept_rate: float
    interval: int
    factor: float
    verbose: bool
    nstep: int
    nstep_tot: int
    naccept: int

    def __init__(self, takestep: T, accept_rate: float = 0.5,
                 interval: int = 50, factor: float = 0.9, verbose: bool = True):
        self.takestep = takestep
        self.target_accept_rate = accept_rate
        self.interval = interval
        self.factor = factor
        self.verbose = verbose
        self.nstep = 0
        self.nstep_tot = 0
        self.naccept = 0

    def _adjust_step_size(self):
        old = self.takestep.stepsize
        accept_rate = float(self.naccept) / self.nstep
        if accept_rate > self.target_accept_rate:
            self.takestep.stepsize /= self.factor
        else:
            self.takestep.stepsize *= self.factor
        if self.verbose:
            print(f'adaptive stepsize: acceptance rate {accept_rate:f} target {self.target_accept_rate:f} new stepsize {self.takestep.stepsize:g} old stepsize {old:g}')

    def __call__(self, x: np.ndarray[float, 1]) -> np.ndarray[float, 1]:
        self.nstep += 1
        self.nstep_tot += 1
        if self.nstep % self.interval == 0:
            self._adjust_step_size()
        return np.asarray(self.takestep(x), float)

    def report(self, accept: bool):
        if accept:
            self.naccept += 1

class _PlainStep[T]:
    takestep: T
    def __init__(self, takestep: T):
        self.takestep = takestep
    def __call__(self, x: np.ndarray[float, 1]) -> np.ndarray[float, 1]:
        return np.asarray(self.takestep(x), float)
    def report(self, accept: bool):
        pass

class Metropolis[R]:
    beta: float
    rng: R

    def __init__(self, T: float, rng: R = None):
        self.beta = 1.0 / T if T != 0 else math.inf
        self.rng = rng

    def accept_reject(self, fun_new: float, ok_new: bool, fun_old: float,
                      ok_old: bool) -> bool:
        prod = -(fun_new - fun_old) * self.beta
        w = math.exp(min(0.0, prod)) if prod == prod else math.nan
        rand = float(self.rng.uniform())
        return w >= rand and (ok_new or not ok_old)

class _Lowest:
    x: np.ndarray[float, 1]
    fun: float
    success: bool
    def __init__(self, x: np.ndarray[float, 1], fun: float, success: bool):
        self.x = x.copy()
        self.fun = fun
        self.success = success
    def update(self, x: np.ndarray[float, 1], fun: float, success: bool) -> bool:
        if success and (fun < self.fun or not self.success):
            self.x = x.copy()
            self.fun = fun
            self.success = success
            return True
        return False

class MinimizerKwargs[A, M, J, B, O]:
    # Typed stand-in for basinhopping's `minimizer_kwargs` dict (Typhon dicts
    # are homogeneous). Plain dicts with only str values ({'method': ...})
    # are accepted too.
    args: A
    method: M
    jac: J
    bounds: B
    tol: Optional[float]
    options: O

    def __init__(self, method: M = None, jac: J = None, bounds: B = None,
                 args: A = (), tol: Optional[float] = None, options: O = None):
        self.args = args
        self.method = method
        self.jac = jac
        self.bounds = bounds
        self.tol = tol
        self.options = options

def _local_min(func, x0: np.ndarray[float, 1], mk):
    from . import minimize
    if isinstance(mk, NoneType):
        return minimize(func, x0)
    elif isinstance(mk, MinimizerKwargs):
        return minimize(func, x0, args=mk.args, method=mk.method, jac=mk.jac,
                        bounds=mk.bounds, tol=mk.tol, options=mk.options)
    else:
        m: Optional[str] = None
        if 'method' in mk:
            m = mk['method']
        return minimize(func, x0, method=m)

def basinhopping(func, x0, niter: int = 100, T: float = 1.0,
                 stepsize: float = 0.5, minimizer_kwargs = None,
                 take_step = None, accept_test = None, callback = None,
                 interval: int = 50, disp: bool = False,
                 niter_success: Optional[int] = None, rng = None,
                 target_accept_rate: float = 0.5,
                 stepwise_factor: float = 0.9, seed = None,
                 local_minimizer = None):
    from ._differentialevolution import _make_rng
    if target_accept_rate <= 0.0 or target_accept_rate >= 1.0:
        raise ValueError('target_accept_rate has to be in range (0, 1)')
    if stepwise_factor <= 0.0 or stepwise_factor >= 1.0:
        raise ValueError('stepwise_factor has to be in range (0, 1)')
    r = _make_rng(rng, seed)
    if isinstance(take_step, NoneType):
        ts = AdaptiveStepsize(RandomDisplacement(stepsize, r), interval=interval,
                              accept_rate=target_accept_rate,
                              factor=stepwise_factor, verbose=disp)
        return _bh_run(func, x0, niter, T, minimizer_kwargs, ts, accept_test,
                       callback, disp, niter_success, r, local_minimizer)
    elif hasattr(take_step, 'stepsize'):
        ts = AdaptiveStepsize(take_step, interval=interval,
                              accept_rate=target_accept_rate,
                              factor=stepwise_factor, verbose=disp)
        return _bh_run(func, x0, niter, T, minimizer_kwargs, ts, accept_test,
                       callback, disp, niter_success, r, local_minimizer)
    else:
        return _bh_run(func, x0, niter, T, minimizer_kwargs,
                       _PlainStep(take_step), accept_test, callback, disp,
                       niter_success, r, local_minimizer)

def _minimize_one(func, x: np.ndarray[float, 1], mk, local_minimizer):
    # returns (x, fun, success, nfev, njev)
    if isinstance(local_minimizer, NoneType):
        res = _local_min(func, x, mk)
        return np.asarray(res.x, float).copy(), float(res.fun), res.success, res.nfev, res.njev
    else:
        res = local_minimizer(func, x)
        return np.asarray(res.x, float).copy(), float(res.fun), res.success, res.nfev, res.njev

def _bh_run(func, x0, niter: int, T: float, mk, ts, accept_test, callback,
            disp: bool, niter_success: Optional[int], r, local_minimizer):
    x = np.array(x0, float).flatten()
    metropolis = Metropolis(T, r)
    ns = niter + 2
    if niter_success is not None:
        ns = niter_success
    failures = 0
    mx, mf, ms, nfev, njev = _minimize_one(func, x, mk, local_minimizer)
    if not ms:
        failures += 1
        if disp:
            print('warning: basinhopping: local minimization failure')
    x = mx.copy()
    energy = mf
    inc_fun = mf
    inc_ok = ms
    if disp:
        print(f'basinhopping step 0: f {energy:g}')
    store = _Lowest(mx, mf, ms)
    if not isinstance(callback, NoneType):
        callback(store.x, store.fun, True)
    count = 0
    i = 0
    message = 'requested number of basinhopping iterations completed successfully'
    for it in range(niter):
        i = it
        # one_cycle
        xs = ts(x.copy())
        tx, tf, tok, tn, tj = _minimize_one(func, xs, mk, local_minimizer)
        if not tok:
            failures += 1
            if disp:
                print('warning: basinhopping: local minimization failure')
        nfev += tn
        njev += tj
        accept = True
        if not isinstance(accept_test, NoneType):
            tr = accept_test(f_new=tf, x_new=tx, f_old=energy, x_old=x)
            if not bool(tr):
                accept = False
        if not metropolis.accept_reject(tf, tok, inc_fun, inc_ok):
            accept = False
        ts.report(accept)
        new_global_min = False
        if accept:
            energy = tf
            x = tx.copy()
            inc_fun = tf
            inc_ok = tok
            new_global_min = store.update(tx, tf, tok)
        if disp:
            print(f'basinhopping step {it + 1}: f {energy:g} trial_f {tf:g} accepted {accept} lowest_f {store.fun:g}')
            if new_global_min:
                print(f'found new global minimum on step {it + 1} with function value {energy:g}')
        if not isinstance(callback, NoneType):
            val = callback(tx, tf, accept)
            if not isinstance(val, NoneType):
                if val:
                    message = 'callback function requested stop early byreturning True'
                    break
        count += 1
        if new_global_min:
            count = 0
        elif count > ns:
            message = 'success condition satisfied'
            break
    res = OptimizeResult(store.x.copy(), store.fun, np.empty(0, float),
                         nit=i + 1, nfev=nfev, njev=njev,
                         success=store.success, message=message)
    res.status = failures   # minimization_failures
    return res
