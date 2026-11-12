# Ported from scipy 1.18.1 (BSD-3-Clause)

# shgo (simplicial homology global optimization, Endres et al.) translated
# from scipy/optimize/_shgo.py with the simplicial complex of
# _shgo_complex.py. Same sampling, minimizer pool, local minimizations and
# stopping rules as scipy for sampling_method='simplicial'.
# Deviations: sampling_method 'sobol'/'halton'/callable (Delaunay
# triangulation via Qhull) and `constraints` are not supported; `options`
# must be a dict of numbers (keys maxiter, maxfev, maxev, maxtime, f_min,
# f_tol, minhgrd, minimize_every_iter, local_iter, ftol, disp);
# `minimizer_kwargs` must be None (SLSQP with ftol=1e-12 as in scipy) or a
# MinimizerKwargs value; callbacks are not passed to the local minimizer.

import math
import time
import numpy as np
from ._common import Bounds
from ._shgo_complex import Complex
from ._basinhopping import MinimizerKwargs

class ShgoResult:
    x: np.ndarray[float, 1]
    fun: float
    xl: np.ndarray[float, 2]
    funl: np.ndarray[float, 1]
    nit: int
    nfev: int
    nlfev: int
    nljev: int
    nlhev: int
    tnev: int
    success: bool
    message: str

    def __init__(self):
        self.x = np.empty(0, float)
        self.fun = math.nan
        self.xl = np.empty((0, 0), float)
        self.funl = np.empty(0, float)
        self.nit = 0
        self.nfev = 0
        self.nlfev = 0
        self.nljev = 0
        self.nlhev = 0
        self.tnev = 0
        self.success = False
        self.message = ''

    def __repr__(self):
        return (f" message: {self.message}\n success: {self.success}\n"
                f"     fun: {self.fun}\n       x: {self.x}\n     nit: {self.nit}\n"
                f"    nfev: {self.nfev}\n   nlfev: {self.nlfev}")

def _opt(options, key: str, default: float) -> float:
    if isinstance(options, NoneType):
        return default
    else:
        if key in options:
            return float(options[key])
        return default

def _has(options, key: str) -> bool:
    if isinstance(options, NoneType):
        return False
    else:
        return key in options

class _LMap:
    x0: List[float]
    has_res: bool
    x_l: np.ndarray[float, 1]
    f_min: float
    nfev: int
    def __init__(self, x0: List[float]):
        self.x0 = x0
        self.has_res = False
        self.x_l = np.empty(0, float)
        self.f_min = math.nan
        self.nfev = 0

class SHGO[F, A, MK]:
    func: F
    args: A
    mk: MK
    dim: int
    bounds: List[List[float]]
    HC: Complex
    ftol: float
    n: int                     # -1 means None
    iters: int                 # -1 means None
    maxiter: int
    maxfev: int
    maxev: int
    maxtime: float
    f_min_true: float
    has_fmin: bool
    f_tol: float
    minhgrd: int
    minimize_every_iter: bool
    local_iter: int
    disp: bool
    init_t: float
    stop_global: bool
    break_routine: bool
    stop_l_iter: bool
    iters_done: int
    n_sampled: int
    fn: int
    hgr: int
    # LMapCache
    lmc: List[_LMap]
    lmc_keys: Dict[Tuple[int, int], List[int]]
    xl_maps: List[np.ndarray[float, 1]]
    f_maps: List[float]
    X_min: List[List[float]]
    pool: List[int]
    f_lowest: float
    x_lowest: np.ndarray[float, 1]
    has_lowest: bool
    res: ShgoResult

    def __init__(self, func: F, bounds, args: A, n: Optional[int],
                 iters: Optional[int], mk: MK, options):
        self.func = func
        self.args = args
        self.mk = mk
        from ._differentialevolution import _bounds_limits
        lb, ub = _bounds_limits(bounds)
        self.dim = lb.size
        self.bounds = List[List[float]]()
        for i in range(self.dim):
            l = lb[i] if math.isfinite(lb[i]) else -1e+50
            u = ub[i] if math.isfinite(ub[i]) else 1e+50
            if l > u:
                raise ValueError("Error: lb > ub in bounds.")
            self.bounds.append([l, u])
        self.ftol = _opt(options, 'ftol', 1e-12)
        self.minimize_every_iter = bool(_opt(options, 'minimize_every_iter', 1.0))
        self.maxiter = int(_opt(options, 'maxiter', -1.0))
        self.maxfev = int(_opt(options, 'maxfev', -1.0))
        self.maxev = int(_opt(options, 'maxev', -1.0))
        self.maxtime = _opt(options, 'maxtime', -1.0)
        self.has_fmin = _has(options, 'f_min')
        self.f_min_true = _opt(options, 'f_min', 0.0)
        self.f_tol = _opt(options, 'f_tol', 0.0001)
        self.minhgrd = int(_opt(options, 'minhgrd', -1.0))
        self.local_iter = int(_opt(options, 'local_iter', 0.0))
        self.disp = bool(_opt(options, 'disp', 0.0))
        if _has(options, 'symmetry') or _has(options, 'infty_constraints'):
            pass
        self.init_t = time.time()
        self.stop_global = False
        self.break_routine = False
        self.stop_l_iter = False
        self.iters_done = 0
        self.n_sampled = 0
        self.fn = 0
        self.hgr = 0
        it = 1 if iters is None else iters
        nn = -1 if n is None else n
        if n is None and iters is None:
            nn = 2 ** self.dim + 1
        if nn == 100:
            nn = 2 ** self.dim + 1
        if self.maxiter >= 0 or self.maxfev >= 0 or self.maxev >= 0 or \
           self.minhgrd >= 0 or self.has_fmin:
            it = -1
        self.n = nn
        self.iters = it
        self.HC = Complex(self.dim, self.bounds)
        self.lmc = List[_LMap]()
        self.lmc_keys = Dict[Tuple[int, int], List[int]]()
        self.xl_maps = List[np.ndarray[float, 1]]()
        self.f_maps = List[float]()
        self.X_min = List[List[float]]()
        self.pool = List[int]()
        self.f_lowest = math.inf
        self.x_lowest = np.empty(0, float)
        self.has_lowest = False
        self.res = ShgoResult()

    # ------------------------------------------------------------ LMap
    def lmc_get(self, x: List[float]) -> int:
        from ._shgo_complex import _py_hash_tuple, _xeq
        h = _py_hash_tuple(x)
        key = (h, 0)
        if key in self.lmc_keys:
            for k in self.lmc_keys[key]:
                if _xeq(self.lmc[k].x0, x):
                    return k
        else:
            self.lmc_keys[key] = List[int]()
        k = len(self.lmc)
        self.lmc.append(_LMap(list(x)))
        self.lmc_keys[key].append(k)
        return k

    # ------------------------------------------------------- main loop
    def iterate_all(self):
        while not self.stop_global:
            if self.break_routine:
                break
            self.iterate()
            self.stopping_criteria()
        if not self.minimize_every_iter:
            if not self.break_routine:
                self.find_minima()
        self.res.nit = self.iters_done
        self.fn = self.HC.nfev

    def iterate(self):
        self.iterate_hypercube()
        if self.minimize_every_iter:
            if not self.break_routine:
                self.find_minima()
        self.iters_done += 1

    def iterate_hypercube(self):
        if self.n < 0:
            self.HC.refine(-1)
            self.n_sampled = self.HC.size()
        else:
            self.HC.refine(self.n)
            self.n_sampled += self.n
        if len(self.xl_maps) > 0:
            for lm in self.lmc:
                v = self.HC.V(lm.x0)
                self.HC.star(v)
        self.HC.process_pools(self.func, self.args)
        self.fn = self.HC.nfev

    def find_minima(self):
        self.minimizers()
        if len(self.X_min) != 0:
            self.minimise_pool(self.local_iter)
            self.sort_result()
            self.f_lowest = self.res.fun
            self.x_lowest = self.res.x
            self.has_lowest = True
        else:
            self.find_lowest_vertex()

    def find_lowest_vertex(self):
        self.f_lowest = math.inf
        self.has_lowest = False
        for k in range(self.HC.size()):
            if self.HC.f[k] < self.f_lowest:
                self.f_lowest = self.HC.f[k]
                self.x_lowest = np.array(self.HC.vx[k], float)
                self.has_lowest = True
        for lm in self.lmc:
            if lm.has_res and lm.f_min < self.f_lowest:
                self.f_lowest = lm.f_min
                self.x_lowest = lm.x_l
                self.has_lowest = True

    # ---------------------------------------------------- minimizer pool
    def minimizers(self):
        pool = List[int]()
        HC = self.HC
        for k in range(HC.size()):
            in_lmc = False
            for xl in self.xl_maps:
                same = True
                for i in range(self.dim):
                    if HC.vx[k][i] != xl[i]:
                        same = False
                if same:
                    in_lmc = True
            if in_lmc:
                continue
            if HC.minimiser(k):
                if k not in pool:
                    pool.append(k)
        # scipy sorts only the (unused for 'simplicial') pool arrays by f;
        # X_min stays in vertex-cache order
        self.pool = pool
        self.X_min = [list(HC.vx[k]) for k in pool]

    def trim(self, idx: int):
        self.X_min.pop(idx)
        self.pool.pop(idx)

    def minimise_pool(self, force_iter: int):
        lx, lf = self.minimize(0)
        self.trim(0)
        while not self.stop_l_iter:
            self.stopping_criteria()
            if force_iter:
                force_iter -= 1
                if force_iter == 0:
                    self.stop_l_iter = True
                    break
            if len(self.X_min) == 0:
                self.stop_l_iter = True
                break
            # g_topograph: order the pool by distance to the last minimizer
            m = len(self.X_min)
            d = np.empty(m)
            for q in range(m):
                s = 0.0
                for i in range(self.dim):
                    t = lx[i] - self.X_min[q][i]
                    s += t * t
                d[q] = math.sqrt(s)
            Z = np.argsort(d)
            far = int(Z[m - 1])
            # minimize the farthest pool point, then drop it
            lx, lf = self.minimize(far)
            self.trim(far)
        self.stop_l_iter = False

    def minimize(self, q: int):
        from . import minimize
        x0 = self.X_min[q]
        li = self.lmc_get(x0)
        lm = self.lmc[li]
        if lm.has_res:
            return lm.x_l, lm.f_min
        v = self.pool[q]
        HC = self.HC
        cb = [[b[0], b[1]] for b in self.bounds]
        xv = HC.vx[v]
        for vn in HC.nn[v].items():
            xn = HC.vx[vn]
            for i in range(self.dim):
                if xn[i] < xv[i] and xn[i] > cb[i][0]:
                    cb[i][0] = xn[i]
                if xn[i] > xv[i] and xn[i] < cb[i][1]:
                    cb[i][1] = xn[i]
        bl = [(c[0], c[1]) for c in cb]
        xs = np.array(x0, float)
        if isinstance(self.mk, NoneType):
            lres = minimize(self.func, xs, args=self.args, method='SLSQP',
                            bounds=bl, options={'ftol': self.ftol})
            return self._add(li, lres.x, float(lres.fun), lres.nfev, lres.njev)
        else:
            lres = minimize(self.func, xs, args=self.args,
                            method=self.mk.method, jac=self.mk.jac,
                            bounds=bl, tol=self.mk.tol,
                            options=self.mk.options)
            return self._add(li, np.asarray(lres.x, float), float(lres.fun),
                             lres.nfev, lres.njev)

    def _add(self, li: int, x: np.ndarray[float, 1], f: float, nfev: int,
             njev: int):
        self.res.nlfev += nfev
        self.res.nljev += njev
        lm = self.lmc[li]
        lm.x_l = x.copy()
        lm.f_min = f
        lm.has_res = True
        self.xl_maps.append(x.copy())
        self.f_maps.append(f)
        return lm.x_l, lm.f_min

    def sort_result(self):
        fm = np.array(self.f_maps, float)
        ind = np.argsort(fm)
        m = len(self.f_maps)
        xl = np.empty((m, self.dim))
        funl = np.empty(m)
        for r in range(m):
            k = int(ind[r])
            funl[r] = fm[k]
            for i in range(self.dim):
                xl[r, i] = self.xl_maps[k][i]
        self.res.xl = xl
        self.res.funl = funl
        self.res.x = xl[0].copy()
        self.res.fun = funl[0]
        self.res.nfev = self.fn + self.res.nlfev

    # ------------------------------------------------ stopping criteria
    def finite_iterations(self):
        if self.iters >= 0 and self.iters_done >= self.iters:
            self.stop_global = True
        if self.maxiter >= 0 and self.iters_done >= self.maxiter:
            self.stop_global = True

    def stopping_criteria(self) -> bool:
        if self.maxiter >= 0:
            self.finite_iterations()
        if self.iters >= 0:
            self.finite_iterations()
        if self.maxfev >= 0:
            if self.fn >= self.maxfev:
                self.stop_global = True
        if self.maxev >= 0:
            if self.n_sampled >= self.maxev:
                self.stop_global = True
        if self.maxtime >= 0.0:
            if time.time() - self.init_t >= self.maxtime:
                self.stop_global = True
        if self.has_fmin:
            self.find_lowest_vertex()
            if self.has_lowest:
                if self.f_min_true == 0.0:
                    if self.f_lowest <= self.f_tol:
                        self.stop_global = True
                else:
                    pe = (self.f_lowest - self.f_min_true) / abs(self.f_min_true)
                    if self.f_lowest <= self.f_min_true:
                        self.stop_global = True
                    if pe <= self.f_tol:
                        self.stop_global = True
        if self.minhgrd >= 0:
            if len(self.f_maps) != 0:
                hgrd = len(self.f_maps) - self.hgr
                self.hgr = len(self.f_maps)
                if hgrd <= self.minhgrd:
                    self.stop_global = True
        return self.stop_global

def shgo(func, bounds, args = (), constraints = None, n: Optional[int] = 100,
         iters: Optional[int] = 1, callback = None, minimizer_kwargs = None,
         options = None, sampling_method: str = 'simplicial',
         workers: int = 1):
    if not isinstance(args, Tuple):
        return shgo(func, bounds, (args,), constraints, n, iters, callback,
                    minimizer_kwargs, options, sampling_method, workers)
    if not isinstance(constraints, NoneType):
        compile_error("shgo: constraints are not supported by this port")
    if sampling_method != 'simplicial':
        if sampling_method == 'sobol' or sampling_method == 'halton':
            raise NotImplementedError("shgo: only sampling_method='simplicial' is supported by this port")
        raise ValueError('Unknown sampling_method specified. Valid methods: halton, sobol, simplicial')
    shc = SHGO(func, bounds, args, n, iters, minimizer_kwargs, options)
    shc.iterate_all()
    if len(shc.xl_maps) == 0:
        shc.find_lowest_vertex()
        shc.break_routine = True
        shc.res.success = False
        shc.res.message = f'Failed to find a feasible minimizer point. Lowest sampling point = {shc.f_lowest}'
        shc.res.fun = shc.f_lowest
        shc.res.x = shc.x_lowest
        shc.res.nfev = shc.fn
        shc.res.tnev = shc.n_sampled
    if not shc.break_routine:
        shc.res.message = 'Optimization terminated successfully.'
        shc.res.success = True
    return shc.res
