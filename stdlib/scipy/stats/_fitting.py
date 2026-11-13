# Copyright (C) 2026 the Typhon authors
# Ported from scipy 1.18.1 (BSD-3-Clause)

# scipy.stats.fit (generic MLE / MSE by differential evolution) and
# goodness_of_fit (Monte Carlo test with KS / AD / CvM / Filliben statistics)
# for continuous distributions.  Parameters are the distribution's shape
# parameters followed by ``loc`` and ``scale``; ``fixed`` / ``known_params``
# dictionaries accept the keys 'loc' and 'scale'.

import numpy as np
import internal.static as static
import scipy.special._kernels as K
from scipy.optimize import differential_evolution
from scipy.optimize._common import Bounds
from ._distn_infrastructure import check_random_state, NAN, INF

class FitParams:
    vals: List[float]
    nshape: int
    def __init__(self, vals: List[float], nshape: int):
        self.vals = vals
        self.nshape = nshape
    def __len__(self):
        return len(self.vals)
    def __getitem__(self, i: int):
        return self.vals[i]
    def __iter__(self):
        for v in self.vals:
            yield v
    @property
    def loc(self) -> float:
        return self.vals[self.nshape]
    @property
    def scale(self) -> float:
        return self.vals[self.nshape + 1]
    def shapes(self):
        return self.vals[:self.nshape]

class FitResult[D]:
    dist: D
    data: np.ndarray[float, 1]
    params: FitParams
    success: bool
    message: str
    def __init__(self, dist: D, data: np.ndarray[float, 1], params: FitParams, success: bool, message: str):
        self.dist = dist
        self.data = data
        self.params = params
        self.success = success
        self.message = message
    def nllf(self, params = None, data = None) -> float:
        th = self.params.vals
        if not isinstance(params, NoneType):
            th = [float(v) for v in params]
        d = self.data
        if not isinstance(data, NoneType):
            d = np.asarray(data).astype(float).ravel()
        return self.dist._nllf_full(th, d, False)

class _FitObj[D]:
    dist: D
    data: np.ndarray[float, 1]
    mm: bool
    def __call__(self, x) -> float:
        th = List[float](len(x))
        for v in x:
            th.append(float(v))
        return self.dist._nllf_full(th, self.data, self.mm)

def _lookup(d, key: str) -> float:
    if isinstance(d, NoneType):
        return NAN
    elif key in d:
        return float(d[key])
    else:
        return NAN

def fit(dist, data, bounds = None, *, guess = None, method: str = 'mle', rng = None):
    NS: Literal[int] = dist._nshape()
    x = np.asarray(data).astype(float)
    if x.ndim != 1:
        raise ValueError("`data` must be exactly one-dimensional.")
    if not bool(np.all(np.isfinite(x))):
        raise ValueError("All elements of `data` must be finite numbers.")
    meth = method.lower()
    if meth != 'mle' and meth != 'mse':
        raise ValueError("method must be 'mle' or 'mse'")
    if meth == 'mse':
        raise NotImplementedError("fit: method='mse' is not supported by this port")
    n_params = NS + 2
    lb = np.empty(n_params)
    ub = np.empty(n_params)
    lb[NS] = 0.0
    ub[NS] = 0.0
    lb[NS + 1] = 1.0
    ub[NS + 1] = 1.0
    if isinstance(bounds, NoneType):
        if NS > 0:
            raise ValueError("A `bounds` sequence must contain tuples for all shape parameters.")
    elif isinstance(bounds, Dict):
        for k in bounds:
            if k == 'loc':
                lb[NS] = float(bounds[k][0])
                ub[NS] = float(bounds[k][1])
            elif k == 'scale':
                lb[NS + 1] = float(bounds[k][0])
                ub[NS + 1] = float(bounds[k][1])
    else:
        nb = len(bounds)
        if nb < NS:
            raise ValueError(f"A `bounds` sequence must contain at least {NS} elements: tuples specifying the lower and upper bounds of all shape parameters.")
        if nb > n_params:
            raise ValueError(f"A `bounds` sequence may not contain more than {n_params} elements.")
        i = 0
        for b in bounds:
            lb[i] = float(b[0])
            ub[i] = float(b[1])
            if lb[i] > ub[i]:
                raise ValueError("There are no values for a parameter on the given interval.")
            i += 1
    for i in range(n_params):
        if not (lb[i] > -INF and ub[i] < INF):
            raise ValueError("The intersection of user-provided bounds and the domain of the distribution is not finite. Please provide finite bounds for the shape parameters in `bounds`.")
    obj = _FitObj(dist, x, False)
    r = None
    if isinstance(guess, NoneType):
        r = differential_evolution(obj, Bounds(lb, ub), rng=rng)
    else:
        g = np.mean(np.stack((lb, ub), axis=1), axis=1)
        gi = 0
        for v in guess:
            g[gi] = float(v)
            gi += 1
        for i in range(n_params):
            g[i] = min(max(g[i], lb[i]), ub[i])
        r = differential_evolution(obj, Bounds(lb, ub), rng=rng, x0=g)
    vals = List[float](n_params)
    for i in range(n_params):
        vals.append(float(r.x[i]))
    ok = bool(r.success)
    msg = r.message
    res = FitResult(dist, x, FitParams(vals, NS), ok, msg)
    if ok and not (res.nllf() < INF):
        res.success = False
        res.message = "Optimization converged to parameter values that are inconsistent with the data."
    return res

# ------------------------------------------------------------------ goodness_of_fit

class GoodnessOfFitResult[D, N]:
    fit_result: FitResult[D]
    statistic: float
    pvalue: float
    null_distribution: N
    def __init__(self, fit_result: FitResult[D], statistic: float, pvalue: float, null_distribution: N):
        self.fit_result = fit_result
        self.statistic = statistic
        self.pvalue = pvalue
        self.null_distribution = null_distribution
    def __getitem__(self, i: Literal[int]):
        if i == 0 or i == -4:
            return self.fit_result
        elif i == 1 or i == -3:
            return self.statistic
        elif i == 2 or i == -2:
            return self.pvalue
        elif i == 3 or i == -1:
            return self.null_distribution
        else:
            compile_error("index out of range")
    def __len__(self):
        return 4

def _fit_values(dist, x: np.ndarray[float, 1], fixed: List[float], guess: List[float]) -> List[float]:
    NS: Literal[int] = dist._nshape()
    if dist.name == 'norm':
        loc = fixed[NS]
        scale = fixed[NS + 1]
        if loc != loc and scale != scale:
            loc = float(np.mean(x))
            scale = float(np.std(x, ddof=1))
        elif loc != loc:
            loc = float(np.mean(x))
        elif scale != scale:
            scale = float(np.sqrt(np.mean((x - loc) ** 2)))
        return [loc, scale]
    r = dist._fit_vals(x, fixed, guess)
    out = List[float](NS + 2)
    for i in static.range(NS + 2):
        out.append(float(r[i]))
    return out

def _stat_value(name: str, fd, x: np.ndarray[float, 1]) -> float:
    n = x.shape[0]
    s = np.sort(x)
    if name == 'ks':
        c = fd.cdf(s)
        dp = -INF
        dm = -INF
        for i in range(n):
            a = (i + 1.0) / n - c[i]
            b = c[i] - i / n
            if a > dp:
                dp = a
            if b > dm:
                dm = b
        return max(dp, dm)
    elif name == 'ad':
        lc = fd.logcdf(s)
        rs = fd.logsf(s[::-1])
        tot = 0.0
        for i in range(n):
            tot += (2 * (i + 1) - 1) / n * (lc[i] + rs[i])
        return -n - tot
    elif name == 'cvm':
        c = fd.cdf(s)
        w = 0.0
        for i in range(n):
            u = (2 * (i + 1) - 1) / (2.0 * n)
            w += (u - c[i]) ** 2
        return 1 / (12.0 * n) + w
    else:
        m = np.empty(n)
        for k in range(n):
            m[k] = K.betaincinv(k + 1.0, n - k, 0.5)
        M = fd.ppf(m)
        xm = float(np.mean(s))
        mm = float(np.mean(M))
        num = 0.0
        d1 = 0.0
        d2 = 0.0
        for i in range(n):
            num += (s[i] - xm) * (M[i] - mm)
            d1 += (s[i] - xm) ** 2
            d2 += (M[i] - mm) ** 2
        return num / K.sqrt(d1 * d2)

def _pairs(d, NS: int, extra: Dict[str, float]) -> List[float]:
    out = List[float](NS + 2)
    for _ in range(NS + 2):
        out.append(NAN)
    out[NS] = _lookup(d, 'loc')
    out[NS + 1] = _lookup(d, 'scale')
    for k in extra:
        pass
    return out

def goodness_of_fit(dist, data, *, known_params = None, fit_params = None,
                    guessed_params = None, statistic: str = 'ad', n_mc_samples: int = 9999, rng = None):
    NS: Literal[int] = dist._nshape()
    x = np.asarray(data).astype(float)
    if x.ndim != 1:
        raise ValueError("`data` must be a one-dimensional array of numbers.")
    st = statistic.lower()
    if st != 'ad' and st != 'ks' and st != 'cvm' and st != 'filliben':
        raise ValueError("`statistic` must be one of {'ad', 'ks', 'cvm', 'filliben'}.")
    g = np.random.default_rng(rng)
    empty = Dict[str, float]()
    known = _pairs(known_params, NS, empty)
    fitp = _pairs(fit_params, NS, empty)
    guessed = _pairs(guessed_params, NS, empty)
    nhd_fixed = List[float](NS + 2)
    rfd_guess = List[float](NS + 2)
    for i in range(NS + 2):
        nhd_fixed.append(known[i] if known[i] == known[i] else fitp[i])
        rfd_guess.append(guessed[i] if guessed[i] == guessed[i] else fitp[i])
    no_guess = List[float](NS + 2)
    for _ in range(NS + 2):
        no_guess.append(NAN)
    nhd_vals = _fit_values(dist, x, nhd_fixed, guessed)
    nhd = dist._from_vals(nhd_vals)
    n = x.shape[0]
    samples = nhd.rvs(size=(n_mc_samples, n), random_state=g)
    obs_fd = dist._from_vals(_fit_values(dist, x, known, rfd_guess))
    observed = _stat_value(st, obs_fd, x)
    null = np.empty(n_mc_samples)
    for r in range(n_mc_samples):
        row = np.ascontiguousarray(samples[r])
        fd = dist._from_vals(_fit_values(dist, row, known, rfd_guess))
        null[r] = _stat_value(st, fd, row)
    eps = 2.220446049250313e-16 * 100
    gamma = abs(eps * observed)
    cnt = 0
    for r in range(n_mc_samples):
        if st == 'filliben':
            if null[r] <= observed + gamma:
                cnt += 1
        else:
            if null[r] >= observed - gamma:
                cnt += 1
    pv = min(max((cnt + 1.0) / (n_mc_samples + 1.0), 0.0), 1.0)
    fr = FitResult(dist, x, FitParams(nhd_vals, NS), True, "The fit was performed successfully.")
    return GoodnessOfFitResult(fr, observed, pv, null)
