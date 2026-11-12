# Ported from scipy 1.18.1 (BSD-3-Clause)

# differential_evolution (Storn & Price) translated from
# scipy/optimize/_differentialevolution.py. The random stream (Generator via
# `rng`, RandomState via integer `seed`) is consumed in exactly the same order
# as scipy, so results match for a fixed seed.
# Deviations: workers must be 1; constraints are not supported (compile
# error); init='sobol'/'halton' need scipy.stats.qmc and are rejected; the
# result has no population/population_energies fields.

import math
import numpy as np
from ._common import OptimizeResult, Bounds, _standardize_bounds, _call_scalar
from ._lbfgsb import _minimize_lbfgsb

_MACHEPS = 2.220446049250313e-16
_msg_success = 'Optimization terminated successfully.'
_msg_maxiter = 'Maximum number of iterations has been exceeded.'

_BIN = ['best1bin', 'randtobest1bin', 'currenttobest1bin', 'best2bin',
        'rand2bin', 'rand1bin']
_EXP = ['best1exp', 'rand1exp', 'randtobest1exp', 'currenttobest1exp',
        'best2exp', 'rand2exp']

def _make_rng(rng, seed):
    if isinstance(rng, NoneType):
        if isinstance(seed, NoneType):
            return np.random.default_rng()
        elif isinstance(seed, int):
            return np.random.RandomState(seed)
        else:
            return seed
    else:
        if isinstance(rng, int):
            return np.random.default_rng(rng)
        else:
            return rng

def _rng_int(rng, high: int) -> int:
    # scipy._lib._util.rng_integers(rng, high)
    if isinstance(rng, np.random.RandomState):
        return int(rng.randint(0, high))
    else:
        return int(rng.integers(0, high))

def _rng_ints(rng, high: int, size: int) -> np.ndarray[int, 1]:
    if isinstance(rng, np.random.RandomState):
        return np.asarray(rng.randint(0, high, size=size), int)
    else:
        return np.asarray(rng.integers(0, high, size=size), int)

def _bounds_limits(bounds):
    if isinstance(bounds, Bounds):
        return bounds.lb.astype(float), bounds.ub.astype(float)
    else:
        return _standardize_bounds(bounds, len(bounds))

class _DEPolishFun[F, A]:
    func: F
    args: A
    def __init__(self, func: F, args: A):
        self.func = func
        self.args = args
    def __call__(self, x) -> float:
        return _call_scalar(self.func, x, self.args)

class _RemoveFixed[F, A]:
    # scipy.optimize._minimize._Remove_From_Func
    func: F
    args: A
    fixed: np.ndarray[bool, 1]
    xfix: np.ndarray[float, 1]
    def __init__(self, func: F, args: A, fixed: np.ndarray[bool, 1],
                 xfix: np.ndarray[float, 1]):
        self.func = func
        self.args = args
        self.fixed = fixed
        self.xfix = xfix
    def full(self, x) -> np.ndarray[float, 1]:
        out = self.xfix.copy()
        k = 0
        for i in range(out.size):
            if not self.fixed[i]:
                out[i] = x[k]
                k += 1
        return out
    def __call__(self, x) -> float:
        return _call_scalar(self.func, self.full(x), self.args)

def _polish_lbfgsb(func, args, x0: np.ndarray[float, 1],
                   lb: np.ndarray[float, 1], ub: np.ndarray[float, 1]):
    # minimize(func, x0, method='L-BFGS-B', bounds=Bounds(lb, ub)) including
    # scipy's removal of fixed (lb == ub) variables when using finite
    # differences. Returns (x, fun, jac, nfev, success).
    n = x0.size
    fixed = np.zeros(n, bool)
    nfix = 0
    for i in range(n):
        if lb[i] == ub[i]:
            fixed[i] = True
            nfix += 1
    if nfix == n:
        xr = lb.copy()
        return xr, _call_scalar(func, xr, args), np.full(n, math.nan), 1, True
    if nfix == 0:
        r = _minimize_lbfgsb(_DEPolishFun(func, args), x0.copy(), (), None,
                             (lb.copy(), ub.copy()), None)
        return r.x, r.fun, r.jac, r.nfev, r.success
    m = n - nfix
    xs = np.empty(m)
    ls = np.empty(m)
    us = np.empty(m)
    k = 0
    for i in range(n):
        if not fixed[i]:
            xs[k] = x0[i]
            ls[k] = lb[i]
            us[k] = ub[i]
            k += 1
    rf = _RemoveFixed(func, args, fixed, lb.copy())
    r = _minimize_lbfgsb(rf, xs, (), None, (ls, us), None)
    jac = np.full(n, math.nan)
    k = 0
    for i in range(n):
        if not fixed[i]:
            jac[i] = r.jac[k]
            k += 1
    return rf.full(r.x), r.fun, jac, r.nfev, r.success

class DifferentialEvolutionSolver[F, A, ST, R]:
    func: F
    args: A
    strategy: ST
    rng: R
    sname: str
    binomial: bool
    maxiter: int
    tol: float
    atol: float
    scale: float
    has_dither: bool
    dither0: float
    dither1: float
    cross_over_probability: float
    deferred: bool
    vectorized: bool
    polish: bool
    disp: bool
    lb: np.ndarray[float, 1]
    ub: np.ndarray[float, 1]
    scale_arg1: np.ndarray[float, 1]
    scale_arg2: np.ndarray[float, 1]
    recip_scale_arg2: np.ndarray[float, 1]
    integrality: np.ndarray[bool, 1]
    any_int: bool
    all_int: bool
    parameter_count: int
    num_population_members: int
    population: np.ndarray[float, 2]
    population_energies: np.ndarray[float, 1]
    nfev: int
    rpi: np.ndarray[int, 1]

    def __init__(self, func: F, bounds, args: A, strategy: ST, maxiter: int,
                 popsize: int, tol: float, mutation, recombination: float,
                 rng: R, polish: bool, disp: bool, init, atol: float,
                 updating: str, vectorized: bool, x0, integrality):
        self.func = func
        self.args = args
        self.strategy = strategy
        self.rng = rng
        if isinstance(strategy, str):
            if strategy not in _BIN and strategy not in _EXP:
                raise ValueError('Please select a valid mutation strategy')
            self.sname = strategy
            self.binomial = strategy in _BIN
        else:
            self.sname = ''
            self.binomial = True
        self.polish = polish
        self.disp = disp
        self.vectorized = vectorized
        self.deferred = updating == 'deferred' or vectorized
        self.tol = tol
        self.atol = atol
        self.maxiter = maxiter
        if isinstance(mutation, Tuple) or isinstance(mutation, List):
            m0 = float(mutation[0])
            m1 = float(mutation[1])
            if not (math.isfinite(m0) and math.isfinite(m1)) or m0 >= 2 or m1 >= 2 or m0 < 0 or m1 < 0:
                raise ValueError('The mutation constant must be a float in U[0, 2), or specified as a tuple(min, max) where min < max and min, max are in U[0, 2).')
            self.has_dither = True
            self.dither0 = min(m0, m1)
            self.dither1 = max(m0, m1)
            self.scale = m0
        else:
            m = float(mutation)
            if not math.isfinite(m) or m >= 2 or m < 0:
                raise ValueError('The mutation constant must be a float in U[0, 2), or specified as a tuple(min, max) where min < max and min, max are in U[0, 2).')
            self.has_dither = False
            self.dither0 = 0.0
            self.dither1 = 0.0
            self.scale = m
        self.cross_over_probability = recombination
        lb, ub = _bounds_limits(bounds)
        n = lb.size
        for i in range(n):
            if not (math.isfinite(lb[i]) and math.isfinite(ub[i])):
                raise ValueError('bounds should be a sequence containing finite real valued (min, max) pairs for each value in x')
        self.parameter_count = n
        # scipy computes the scaling before the integrality adjustment
        self.scale_arg1 = 0.5 * (lb + ub)
        self.scale_arg2 = np.fabs(lb - ub)
        self.integrality = np.zeros(n, bool)
        self.any_int = False
        if not isinstance(integrality, NoneType):
            ia = np.broadcast_to(np.asarray(integrality, bool), (n,))
            for i in range(n):
                self.integrality[i] = bool(ia[i])
        for i in range(n):
            if self.integrality[i]:
                self.any_int = True
        self.all_int = self.any_int
        for i in range(n):
            if not self.integrality[i]:
                self.all_int = False
        if self.any_int:
            for i in range(n):
                if self.integrality[i]:
                    l = math.ceil(lb[i])
                    u = math.floor(ub[i])
                    if not l <= u:
                        raise ValueError('One of the integrality constraints does not have any possible integer values between the lower/upper bounds.')
                    lb[i] = float(np.nextafter(float(l) - 0.5, math.inf))
                    ub[i] = float(np.nextafter(float(u) + 0.5, -math.inf))
        self.lb = lb
        self.ub = ub
        self.recip_scale_arg2 = np.zeros(n)
        eb = 0
        for i in range(n):
            if self.scale_arg2[i] != 0.0:
                self.recip_scale_arg2[i] = 1.0 / self.scale_arg2[i]
            if lb[i] == ub[i]:
                eb += 1
        self.num_population_members = max(5, popsize * max(1, n - eb))
        self.nfev = 0
        if isinstance(init, str):
            if init == 'latinhypercube':
                self._init_lhs()
            elif init == 'random':
                self.population = np.asarray(self.rng.uniform(size=(self.num_population_members, n)), float)
            else:
                raise ValueError("The population initialization method must be one of 'latinhypercube' or 'random', or an array of shape (S, N) where N is the number of parameters and S>5")
        else:
            popn = np.asarray(init, float)
            if popn.shape[0] < 5 or popn.shape[1] != n:
                raise ValueError('The population supplied needs to have shape (S, len(x)), where S > 4.')
            pop = np.empty((popn.shape[0], n))
            for k in range(popn.shape[0]):
                r = self._unscale(popn[k].copy())
                for j in range(n):
                    pop[k, j] = min(max(r[j], 0.0), 1.0)
            self.population = pop
            self.num_population_members = popn.shape[0]
        self.population_energies = np.full(self.num_population_members, math.inf)
        if not isinstance(x0, NoneType):
            xs = self._unscale(np.asarray(x0, float).flatten())
            for j in range(n):
                if xs[j] > 1.0 or xs[j] < 0.0:
                    raise ValueError('Some entries in x0 lay outside the specified bounds')
                self.population[0, j] = xs[j]
        self.rpi = np.arange(self.num_population_members)

    def _init_lhs(self):
        rng = self.rng
        npop = self.num_population_members
        n = self.parameter_count
        segsize = 1.0 / npop
        u = np.asarray(rng.uniform(size=(npop, n)), float)
        samples = np.empty((npop, n))
        for i in range(npop):
            off = float(i) * segsize   # np.linspace: i * step
            for j in range(n):
                samples[i, j] = segsize * u[i, j] + off
        self.population = np.zeros((npop, n))
        for j in range(n):
            order = rng.permutation(np.arange(npop))
            for i in range(npop):
                self.population[i, j] = samples[int(order[i]), j]

    def _scale(self, trial: np.ndarray[float, 1]) -> np.ndarray[float, 1]:
        s = self.scale_arg1 + (trial - 0.5) * self.scale_arg2
        if self.any_int:
            for i in range(s.size):
                if self.integrality[i]:
                    s[i] = float(np.round(s[i]))
        return s

    def _unscale(self, p: np.ndarray[float, 1]) -> np.ndarray[float, 1]:
        return (p - self.scale_arg1) * self.recip_scale_arg2 + 0.5

    def x(self) -> np.ndarray[float, 1]:
        return self._scale(self.population[0].copy())

    def _energy(self, x: np.ndarray[float, 1]) -> float:
        return _call_scalar(self.func, x, self.args)

    def _energies(self, pop: np.ndarray[float, 2]) -> np.ndarray[float, 1]:
        S = pop.shape[0]
        n = self.parameter_count
        if self.vectorized:
            P = np.empty((n, S))
            for k in range(S):
                xs = self._scale(pop[k].copy())
                for j in range(n):
                    P[j, k] = xs[j]
            e = np.atleast_1d(np.asarray(self.func(P, *self.args), float)).flatten()
            if e.size != S:
                raise RuntimeError('The vectorized function must return an array of shape (S,) when given an array of shape (len(x), S)')
            self.nfev += 1
            return e
        e = np.empty(S)
        for k in range(S):
            e[k] = self._energy(self._scale(pop[k].copy()))
        self.nfev += S
        return e

    def _promote_lowest_energy(self):
        e = self.population_energies
        l = 0
        for i in range(1, e.size):
            if e[i] < e[l]:
                l = i
        t = e[0]
        e[0] = e[l]
        e[l] = t
        for j in range(self.parameter_count):
            t = self.population[0, j]
            self.population[0, j] = self.population[l, j]
            self.population[l, j] = t

    def _std_ok(self) -> Tuple[bool, float, float]:
        e = self.population_energies
        for v in e:
            if math.isinf(v):
                return False, 0.0, 0.0
        n = float(e.size)
        mu = float(np.sum(e)) / n
        d = e - mu
        return True, math.sqrt(float(np.sum(d * d)) / n), mu

    def convergence(self) -> float:
        ok, sd, mu = self._std_ok()
        if not ok:
            return math.inf
        return sd / (abs(mu) + _MACHEPS)

    def converged(self) -> bool:
        ok, sd, mu = self._std_ok()
        if not ok:
            return False
        return sd <= self.atol + self.tol * abs(mu)

    # ------------------------------------------------------------ mutation
    def _select_samples(self, candidate: int) -> List[int]:
        self.rng.shuffle(self.rpi)
        out = List[int](5)
        for k in range(min(6, self.rpi.size)):
            v = int(self.rpi[k])
            if v != candidate and len(out) < 5:
                out.append(v)
        return out

    def _bprime(self, candidate: int, r: List[int]) -> np.ndarray[float, 1]:
        P = self.population
        s = self.scale
        st = self.sname
        if st == 'best1bin' or st == 'best1exp':
            return P[0] + s * (P[r[0]] - P[r[1]])
        elif st == 'rand1bin' or st == 'rand1exp':
            return P[r[0]] + s * (P[r[1]] - P[r[2]])
        elif st == 'randtobest1bin' or st == 'randtobest1exp':
            b = P[r[0]].copy()
            b += s * (P[0] - b)
            b += s * (P[r[1]] - P[r[2]])
            return b
        elif st == 'currenttobest1bin' or st == 'currenttobest1exp':
            return P[candidate] + s * (P[0] - P[candidate] + P[r[0]] - P[r[1]])
        elif st == 'best2bin' or st == 'best2exp':
            return P[0] + s * (P[r[0]] + P[r[1]] - P[r[2]] - P[r[3]])
        else:
            return P[r[0]] + s * (P[r[1]] + P[r[2]] - P[r[3]] - P[r[4]])

    def _custom(self, candidate: int) -> np.ndarray[float, 1]:
        npop = self.num_population_members
        n = self.parameter_count
        sp = np.empty((npop, n))
        for k in range(npop):
            xs = self._scale(self.population[k].copy())
            for j in range(n):
                sp[k, j] = xs[j]
        t = np.asarray(self.strategy(candidate, sp, rng=self.rng), float)
        if t.ndim != 1 or t.size != n:
            raise RuntimeError('strategy must have signature f(candidate: int, population: np.ndarray, rng=None) returning an array of shape (N,)')
        return self._unscale(t.flatten())

    def _cross(self, trial: np.ndarray[float, 1], bprime: np.ndarray[float, 1],
               fill_point: int, cr: np.ndarray[float, 1]):
        n = self.parameter_count
        if self.binomial:
            for j in range(n):
                if cr[j] < self.cross_over_probability or j == fill_point:
                    trial[j] = bprime[j]
        else:
            i = 0
            fp = fill_point
            while i < n and (i == 0 or cr[i] < self.cross_over_probability):
                trial[fp] = bprime[fp]
                fp = (fp + 1) % n
                i += 1

    def _mutate(self, candidate: int) -> np.ndarray[float, 1]:
        if not isinstance(self.strategy, str):
            return self._custom(candidate)
        n = self.parameter_count
        fill_point = _rng_int(self.rng, n)
        r = self._select_samples(candidate)
        trial = self.population[candidate].copy()
        bprime = self._bprime(candidate, r)
        cr = np.asarray(self.rng.uniform(size=n), float)
        self._cross(trial, bprime, fill_point, cr)
        return trial

    def _mutate_many(self) -> np.ndarray[float, 2]:
        S = self.num_population_members
        n = self.parameter_count
        trial = np.empty((S, n))
        if not isinstance(self.strategy, str):
            for c in range(S):
                t = self._custom(c)
                for j in range(n):
                    trial[c, j] = t[j]
            return trial
        samples = [self._select_samples(c) for c in range(S)]
        bps = [self._bprime(c, samples[c]) for c in range(S)]
        fill = _rng_ints(self.rng, n, S)
        cr = np.asarray(self.rng.uniform(size=(S, n)), float)
        for c in range(S):
            t = self.population[c].copy()
            self._cross(t, bps[c], int(fill[c]), cr[c].copy())
            for j in range(n):
                trial[c, j] = t[j]
        return trial

    def _ensure_constraint(self, trial: np.ndarray[float, 1]):
        oob = 0
        for v in trial:
            if v > 1.0 or v < 0.0:
                oob += 1
        if oob:
            u = np.asarray(self.rng.uniform(size=oob), float)
            k = 0
            for j in range(trial.size):
                if trial[j] > 1.0 or trial[j] < 0.0:
                    trial[j] = u[k]
                    k += 1

    def _ensure_constraint2(self, trial: np.ndarray[float, 2]):
        # boolean-mask assignment fills in C (row-major) order
        oob = 0
        for v in trial.flat:
            if v > 1.0 or v < 0.0:
                oob += 1
        if oob:
            u = np.asarray(self.rng.uniform(size=oob), float)
            k = 0
            for i in range(trial.shape[0]):
                for j in range(trial.shape[1]):
                    if trial[i, j] > 1.0 or trial[i, j] < 0.0:
                        trial[i, j] = u[k]
                        k += 1

    # ----------------------------------------------------------- iteration
    def _init_energies(self):
        all_inf = True
        for v in self.population_energies:
            if not math.isinf(v):
                all_inf = False
        if all_inf:
            self.population_energies = self._energies(self.population)
            self._promote_lowest_energy()

    def step(self):
        self._init_energies()
        if self.has_dither:
            self.scale = float(self.rng.uniform(self.dither0, self.dither1))
        npop = self.num_population_members
        if not self.deferred:
            for c in range(npop):
                trial = self._mutate(c)
                self._ensure_constraint(trial)
                e = self._energy(self._scale(trial.copy()))
                self.nfev += 1
                if e <= self.population_energies[c]:
                    for j in range(self.parameter_count):
                        self.population[c, j] = trial[j]
                    self.population_energies[c] = e
                    if e <= self.population_energies[0]:
                        self._promote_lowest_energy()
        else:
            tp = self._mutate_many()
            self._ensure_constraint2(tp)
            te = self._energies(tp)
            for c in range(npop):
                if te[c] <= self.population_energies[c]:
                    for j in range(self.parameter_count):
                        self.population[c, j] = tp[c, j]
                    self.population_energies[c] = te[c]
            self._promote_lowest_energy()

    def _result(self, nit: int, message: str, warning_flag: bool):
        return OptimizeResult(self.x(), float(self.population_energies[0]),
                              np.empty(0, float), nit=nit, nfev=self.nfev,
                              success=not warning_flag, message=message)

    def solve(self, callback):
        nit = 0
        warning_flag = False
        status_message = _msg_success
        self._init_energies()
        completed = False
        for it in range(1, self.maxiter + 1):
            nit = it
            self.step()
            if self.disp:
                print(f'differential_evolution step {nit}: f(x)= {self.population_energies[0]}')
            if not isinstance(callback, NoneType):
                res = self._result(nit, 'in progress', False)
                if bool(callback(res)):
                    warning_flag = True
                    status_message = 'callback function requested stop early'
            if warning_flag or self.converged():
                completed = True
                break
        if not completed:
            status_message = _msg_maxiter
            warning_flag = True
        DE = self._result(nit, status_message, warning_flag)
        if self.polish and not self.all_int:
            lb = self.lb.copy()
            ub = self.ub.copy()
            if self.any_int:
                for i in range(self.parameter_count):
                    if self.integrality[i]:
                        lb[i] = DE.x[i]
                        ub[i] = DE.x[i]
            if self.disp:
                print("Polishing solution with 'L-BFGS-B'")
            px, pf, pj, pn, ps = _polish_lbfgsb(self.func, self.args, DE.x, lb, ub)
            self.nfev += pn
            DE.nfev = self.nfev
            inb = True
            for i in range(self.parameter_count):
                if not (px[i] <= ub[i] and lb[i] <= px[i]):
                    inb = False
            if pf < DE.fun and ps and inb:
                DE.fun = pf
                DE.x = px
                DE.jac = pj
                self.population_energies[0] = pf
                p0 = self._unscale(px)
                for j in range(self.parameter_count):
                    self.population[0, j] = p0[j]
        return DE

def differential_evolution(func, bounds, args = (), strategy = 'best1bin',
                           maxiter: int = 1000, popsize: int = 15,
                           tol: float = 0.01, mutation = (0.5, 1.0),
                           recombination: float = 0.7, rng = None,
                           callback = None, disp: bool = False,
                           polish: bool = True, init = 'latinhypercube',
                           atol: float = 0.0, updating: str = 'immediate',
                           workers: int = 1, constraints = (), x0 = None,
                           integrality = None, vectorized: bool = False,
                           seed = None):
    if not isinstance(args, Tuple):
        return differential_evolution(func, bounds, (args,), strategy, maxiter,
                                      popsize, tol, mutation, recombination,
                                      rng, callback, disp, polish, init, atol,
                                      updating, workers, constraints, x0,
                                      integrality, vectorized, seed)
    if static.len(constraints) != 0:
        compile_error("differential_evolution: constraints are not supported by this port")
    if workers != 1:
        raise ValueError("differential_evolution: only workers=1 is supported")
    if updating != 'immediate' and updating != 'deferred':
        raise ValueError("updating must be 'immediate' or 'deferred'")
    r = _make_rng(rng, seed)
    solver = DifferentialEvolutionSolver(func, bounds, args, strategy, maxiter,
                                         popsize, tol, mutation, recombination,
                                         r, polish, disp, init, atol, updating,
                                         vectorized, x0, integrality)
    return solver.solve(callback)
