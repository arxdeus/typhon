# Ported from scipy 1.18.1 (BSD-3-Clause)

# dual_annealing (generalized simulated annealing, Xiang et al.) translated
# from scipy/optimize/_dual_annealing.py. Same random stream as scipy.
# Deviations: `minimizer_kwargs` must be None or a MinimizerKwargs value
# (see _basinhopping.py); with None the local search is L-BFGS-B with
# maxiter = min(max(6n, 100), 1000) as in scipy. `message` is a str.

import math
import numpy as np
from ._common import OptimizeResult, Bounds, _standardize_bounds, _call_scalar
from ._lbfgsb import _minimize_lbfgsb
from ._basinhopping import MinimizerKwargs

_TAIL_LIMIT = 100000000.0
_MIN_VISIT_BOUND = 1e-10

class VisitingDistribution[R]:
    visiting_param: float
    rng: R
    lower: np.ndarray[float, 1]
    upper: np.ndarray[float, 1]
    bound_range: np.ndarray[float, 1]
    factor2: float
    factor3: float
    factor4_p: float
    factor5: float
    d1: float
    factor6: float

    def __init__(self, lb: np.ndarray[float, 1], ub: np.ndarray[float, 1],
                 visiting_param: float, rng: R):
        q = visiting_param
        self.visiting_param = q
        self.rng = rng
        self.lower = lb
        self.upper = ub
        self.bound_range = ub - lb
        self.factor2 = math.exp((4.0 - q) * math.log(q - 1.0))
        self.factor3 = math.exp((2.0 - q) * math.log(2.0) / (q - 1.0))
        self.factor4_p = math.sqrt(math.pi) * self.factor2 / (self.factor3 * (3.0 - q))
        self.factor5 = 1.0 / (q - 1.0) - 0.5
        self.d1 = 2.0 - self.factor5
        self.factor6 = math.pi * (1.0 - self.factor5) / math.sin(math.pi * (1.0 - self.factor5)) / math.exp(math.lgamma(self.d1))

    def visit_fn(self, temperature: float, dim: int) -> np.ndarray[float, 1]:
        q = self.visiting_param
        xy = np.asarray(self.rng.normal(size=(dim, 2)), float)
        factor1 = math.exp(math.log(temperature) / (q - 1.0))
        factor4 = self.factor4_p * factor1
        mult = math.exp(-(q - 1.0) * math.log(self.factor6 / factor4) / (3.0 - q))
        out = np.empty(dim)
        for i in range(dim):
            x = xy[i, 0] * mult
            den = math.exp((q - 1.0) * math.log(abs(xy[i, 1])) / (3.0 - q))
            out[i] = x / den
        return out

    def visiting(self, x: np.ndarray[float, 1], step: int,
                 temperature: float) -> np.ndarray[float, 1]:
        dim = x.size
        if step < dim:
            visits = self.visit_fn(temperature, dim)
            us = np.asarray(self.rng.uniform(size=2), float)
            xv = np.empty(dim)
            for i in range(dim):
                v = visits[i]
                if v > _TAIL_LIMIT:
                    v = _TAIL_LIMIT * us[0]
                elif v < -_TAIL_LIMIT:
                    v = -_TAIL_LIMIT * us[1]
                a = v + x[i] - self.lower[i]
                b = math.fmod(a, self.bound_range[i]) + self.bound_range[i]
                xv[i] = math.fmod(b, self.bound_range[i]) + self.lower[i]
                if abs(xv[i] - self.lower[i]) < _MIN_VISIT_BOUND:
                    xv[i] += 1e-10
            return xv
        else:
            xv = x.copy()
            visit = self.visit_fn(temperature, 1)[0]
            if visit > _TAIL_LIMIT:
                visit = _TAIL_LIMIT * float(self.rng.uniform())
            elif visit < -_TAIL_LIMIT:
                visit = -_TAIL_LIMIT * float(self.rng.uniform())
            index = step - dim
            xv[index] = visit + x[index]
            a = xv[index] - self.lower[index]
            b = math.fmod(a, self.bound_range[index]) + self.bound_range[index]
            xv[index] = math.fmod(b, self.bound_range[index]) + self.lower[index]
            if abs(xv[index] - self.lower[index]) < _MIN_VISIT_BOUND:
                xv[index] += _MIN_VISIT_BOUND
            return xv

class _DAFun[F, A]:
    # ObjectiveFunWrapper
    func: F
    args: A
    nfev: int
    ngev: int
    maxfun: float
    def __init__(self, func: F, args: A, maxfun: float):
        self.func = func
        self.args = args
        self.nfev = 0
        self.ngev = 0
        self.maxfun = maxfun
    def __call__(self, x) -> float:
        self.nfev += 1
        return _call_scalar(self.func, x, self.args)

class _DAState[FW, MK, R, CB]:
    fw: FW
    mk: MK
    rng: R
    callback: CB
    lower: np.ndarray[float, 1]
    upper: np.ndarray[float, 1]
    has_best: bool
    ebest: float
    xbest: np.ndarray[float, 1]
    current_energy: float
    current_location: np.ndarray[float, 1]
    # strategy chain
    emin: float
    xmin: np.ndarray[float, 1]
    acceptance_param: float
    not_improved_idx: int
    not_improved_max_idx: int
    temperature_step: float
    K: int
    improved: bool
    ls_maxiter: int

    def __init__(self, fw: FW, mk: MK, rng: R, callback: CB,
                 lower: np.ndarray[float, 1], upper: np.ndarray[float, 1],
                 accept: float):
        self.fw = fw
        self.mk = mk
        self.rng = rng
        self.callback = callback
        self.lower = lower
        self.upper = upper
        self.has_best = False
        self.ebest = 0.0
        self.xbest = np.empty(0, float)
        self.current_energy = 0.0
        self.current_location = np.empty(0, float)
        self.acceptance_param = accept
        self.emin = 0.0
        self.xmin = np.empty(0, float)
        self.not_improved_idx = 0
        self.not_improved_max_idx = 1000
        self.temperature_step = 0.0
        self.K = 100 * lower.size
        self.improved = False
        n = lower.size
        self.ls_maxiter = min(max(n * 6, 100), 1000)

    def _uniform_loc(self) -> np.ndarray[float, 1]:
        return np.asarray(self.rng.uniform(self.lower, self.upper,
                                           size=self.lower.size), float)

    def reset(self, x0: Optional[np.ndarray[float, 1]]):
        if x0 is None:
            self.current_location = self._uniform_loc()
        else:
            self.current_location = x0.copy()
        reinit = 0
        while True:
            self.current_energy = self.fw(self.current_location)
            if not math.isfinite(self.current_energy):
                if reinit >= 1000:
                    raise ValueError('Stopping algorithm because function create NaN or (+/-) infinity values even with trying new random parameters')
                self.current_location = self._uniform_loc()
                reinit += 1
                if not self.has_best:
                    self.has_best = True
                    self.ebest = self.current_energy
                    self.xbest = self.current_location.copy()
            else:
                if not self.has_best:
                    self.has_best = True
                    self.ebest = self.current_energy
                    self.xbest = self.current_location.copy()
                break

    def update_best(self, e: float, x: np.ndarray[float, 1], context: int) -> bool:
        self.ebest = e
        self.xbest = x.copy()
        if not isinstance(self.callback, NoneType):
            val = self.callback(x, e, context)
            if not isinstance(val, NoneType):
                if val:
                    return True
        return False

    def update_current(self, e: float, x: np.ndarray[float, 1]):
        self.current_energy = e
        self.current_location = x.copy()

    def chain_init(self):
        self.emin = self.current_energy
        self.xmin = self.current_location.copy()

    def accept_reject(self, j: int, e: float, xv: np.ndarray[float, 1]):
        r = float(self.rng.uniform())
        pqv_temp = 1.0 - (1.0 - self.acceptance_param) * (e - self.current_energy) / self.temperature_step
        pqv = 0.0
        if pqv_temp > 0.0:
            pqv = math.exp(math.log(pqv_temp) / (1.0 - self.acceptance_param))
        if r <= pqv:
            self.update_current(e, xv)
            self.xmin = self.current_location.copy()
        if self.not_improved_idx >= self.not_improved_max_idx:
            if j == 0 or self.current_energy < self.emin:
                self.emin = self.current_energy
                self.xmin = self.current_location.copy()

    def run(self, vd, step: int, temperature: float) -> str:
        self.temperature_step = temperature / float(step + 1)
        self.not_improved_idx += 1
        for j in range(self.current_location.size * 2):
            if j == 0:
                self.improved = step == 0
            xv = vd.visiting(self.current_location, j, temperature)
            e = self.fw(xv)
            if e < self.current_energy:
                self.update_current(e, xv)
                if e < self.ebest:
                    if self.update_best(e, xv, 0):
                        return 'Callback function requested to stop early by returning True'
                    self.improved = True
                    self.not_improved_idx = 0
            else:
                self.accept_reject(j, e, xv)
            if self.fw.nfev >= self.fw.maxfun:
                return 'Maximum number of function call reached during annealing'
        return ''

    def _ls(self, x: np.ndarray[float, 1], e: float):
        from . import minimize
        xt = x.copy()
        if isinstance(self.mk, NoneType):
            mres = _minimize_lbfgsb(self.fw, x.copy(), (), None,
                                    (self.lower.copy(), self.upper.copy()),
                                    {'maxiter': self.ls_maxiter})
            self.fw.ngev += mres.njev
            return self._ls_check(mres.x, mres.fun, xt, e)
        else:
            mres = minimize(self.fw, x.copy(), method=self.mk.method,
                            jac=self.mk.jac, bounds=self.mk.bounds,
                            tol=self.mk.tol, options=self.mk.options)
            self.fw.ngev += mres.njev
            return self._ls_check(np.asarray(mres.x, float), float(mres.fun), xt, e)

    def _ls_check(self, mx: np.ndarray[float, 1], mf: float,
                  xt: np.ndarray[float, 1], e: float):
        ok = math.isfinite(mf)
        for i in range(mx.size):
            if not math.isfinite(mx[i]) or mx[i] < self.lower[i] or mx[i] > self.upper[i]:
                ok = False
        if ok and mf < e:
            return mf, mx.copy()
        return e, xt

    def local_search(self) -> str:
        if self.improved:
            e, x = self._ls(self.xbest, self.ebest)
            if e < self.ebest:
                self.not_improved_idx = 0
                if self.update_best(e, x, 1):
                    return 'Callback function requested to stop early by returning True'
                self.update_current(e, x)
            if self.fw.nfev >= self.fw.maxfun:
                return 'Maximum number of function call reached during local search'
        do_ls = False
        n = self.current_location.size
        if self.K < 90 * n:
            arg = self.K * (self.ebest - self.current_energy) / self.temperature_step
            pls = math.inf if arg > 709.0 else math.exp(arg)   # np.exp overflow
            if pls >= float(self.rng.uniform()):
                do_ls = True
        if self.not_improved_idx >= self.not_improved_max_idx:
            do_ls = True
        if do_ls:
            e, x = self._ls(self.xmin, self.emin)
            self.xmin = x.copy()
            self.emin = e
            self.not_improved_idx = 0
            self.not_improved_max_idx = n
            if e < self.ebest:
                if self.update_best(self.emin, self.xmin, 2):
                    return 'Callback function requested to stop early by returning True'
                self.update_current(e, x)
            if self.fw.nfev >= self.fw.maxfun:
                return 'Maximum number of function call reached during dual annealing'
        return ''

def dual_annealing(func, bounds, args = (), maxiter: int = 1000,
                   minimizer_kwargs = None, initial_temp: float = 5230.0,
                   restart_temp_ratio: float = 2e-05, visit: float = 2.62,
                   accept: float = -5.0, maxfun: float = 10000000.0,
                   rng = None, no_local_search: bool = False,
                   callback = None, x0 = None, seed = None):
    from ._differentialevolution import _make_rng, _bounds_limits
    if not isinstance(args, Tuple):
        return dual_annealing(func, bounds, (args,), maxiter, minimizer_kwargs,
                              initial_temp, restart_temp_ratio, visit, accept,
                              maxfun, rng, no_local_search, callback, x0, seed)
    lower, upper = _bounds_limits(bounds)
    n = lower.size
    xs: Optional[np.ndarray[float, 1]] = None
    if not isinstance(x0, NoneType):
        xa = np.asarray(x0, float).flatten()
        if xa.size != n:
            raise ValueError('Bounds size does not match x0')
        xs = xa
    if restart_temp_ratio <= 0.0 or restart_temp_ratio >= 1.0:
        raise ValueError('Restart temperature ratio has to be in range (0, 1)')
    for i in range(n):
        if math.isinf(lower[i]) or math.isinf(upper[i]) or math.isnan(lower[i]) or math.isnan(upper[i]):
            raise ValueError('Some bounds values are inf values or nan values')
    for i in range(n):
        if not lower[i] < upper[i]:
            raise ValueError('Bounds are not consistent min < max')
    fw = _DAFun(func, args, maxfun)
    r = _make_rng(rng, seed)
    st = _DAState(fw, minimizer_kwargs, r, callback, lower, upper, accept)
    st.reset(xs)
    temperature_restart = initial_temp * restart_temp_ratio
    vd = VisitingDistribution(lower, upper, visit, r)
    st.chain_init()
    need_to_stop = False
    iteration = 0
    message = ''
    success = True
    t1 = math.exp((visit - 1) * math.log(2.0)) - 1.0
    while not need_to_stop:
        for i in range(maxiter):
            s = float(i) + 2.0
            t2 = math.exp((visit - 1) * math.log(s)) - 1.0
            temperature = initial_temp * t1 / t2
            if iteration >= maxiter:
                message = 'Maximum number of iteration reached'
                need_to_stop = True
                break
            if temperature < temperature_restart:
                st.reset(None)
                break
            val = st.run(vd, i, temperature)
            if val != '':
                message = val
                need_to_stop = True
                success = False
                break
            if not no_local_search:
                val = st.local_search()
                if val != '':
                    message = val
                    need_to_stop = True
                    success = False
                    break
            iteration += 1
    return OptimizeResult(st.xbest.copy(), st.ebest, np.empty(0, float),
                          nit=iteration, nfev=fw.nfev, njev=fw.ngev,
                          success=success, message=message)
