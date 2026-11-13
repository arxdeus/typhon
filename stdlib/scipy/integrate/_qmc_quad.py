# Copyright (C) 2026 the Typhon authors
# Ported from scipy 1.18.1 (BSD-3-Clause)

# qmc_quad: integration of an N-d function by Quasi-Monte Carlo quadrature.

import math
import numpy as np
from scipy.special import logsumexp

class QMCQuadResult:
    integral: float
    standard_error: float

    def __init__(self, integral: float, standard_error: float):
        self.integral = integral
        self.standard_error = standard_error

def _rebuild(qrng, rng):
    # fresh, independently scrambled engine of the same type (type(qrng)(seed=rngs[i], **qrng._init_quad))
    return type(qrng)(qrng.d, scramble=qrng.scramble, rng=rng)

def _qmc_quad_run(func, a: np.ndarray[float, 1], b: np.ndarray[float, 1], n_points: int,
                  n_estimates: int, qrng, log: bool) -> QMCQuadResult:
    dim = a.shape[0]
    sign = 1.0
    for k in range(dim):
        if a[k] == b[k]:
            return QMCQuadResult(-math.inf if log else 0.0, 0.0)
    nswap = 0
    for k in range(dim):
        if b[k] < a[k]:
            t = a[k]
            a[k] = b[k]
            b[k] = t
            nswap += 1
    if nswap % 2 == 1:
        sign = -1.0
    A = 1.0
    for k in range(dim):
        A = A * (b[k] - a[k])
    dA = A / n_points
    estimates = np.zeros(n_estimates)
    rngs = qrng.rng.spawn(n_estimates)
    cur = qrng
    for i in range(n_estimates):
        sample = cur.random(n_points)
        x = np.empty((dim, n_points))
        for p in range(n_points):
            for k in range(dim):
                x[k, p] = sample[p, k] * (b[k] - a[k]) + a[k]
        integrands = np.asarray(func(x), dtype=float)
        if log:
            estimates[i] = float(logsumexp(integrands)) + math.log(dA)
        else:
            s = 0.0
            for p in range(integrands.shape[0]):
                s += integrands[p] * dA
            estimates[i] = s
        cur = _rebuild(qrng, rngs[i])
    integral = 0.0
    se = 0.0
    if log:
        integral = float(logsumexp(estimates)) - math.log(n_estimates)
        # std (ddof=1) of log-estimates, evaluated in the linear domain relative to the mean
        acc = 0.0
        for i in range(n_estimates):
            diff = math.exp(estimates[i] - integral) - 1.0
            acc += diff * diff
        s = 0.5 * (math.log(acc) - math.log(n_estimates - 1)) + integral if acc > 0 else -math.inf
        se = s - 0.5 * math.log(n_estimates)
    else:
        m = np.mean(estimates)
        integral = float(m)
        s = float(np.std(estimates, ddof=1))
        se = s / math.sqrt(n_estimates)
    return QMCQuadResult(integral * sign if not log else integral, se)

def qmc_quad(func, a, b, *, n_estimates: int = 8, n_points: int = 1024, qrng = None,
             log: bool = False):
    """Integrate func over the box [a, b] with randomized QMC. `func` takes an
    array of shape (d, n_points) and returns the integrand values. When qrng
    is None a scrambled Halton engine is used."""
    aa = np.atleast_1d(np.asarray(a, dtype=float)).copy()
    bb = np.atleast_1d(np.asarray(b, dtype=float)).copy()
    if aa.shape[0] != bb.shape[0]:
        raise ValueError("`a` and `b` must have the same dimension.")
    if n_points < 1 or n_estimates < 2:
        raise ValueError("`n_estimates` must be at least 2 and `n_points` positive.")
    if qrng is None:
        from scipy.stats.qmc import Halton
        return _qmc_quad_run(func, aa, bb, n_points, n_estimates, Halton(aa.shape[0]), log)
    if qrng.d != aa.shape[0]:
        raise ValueError("`qrng` must be an instance of scipy.stats.qmc.QMCEngine with `d` matching the dimension of `a` and `b`.")
    return _qmc_quad_run(func, aa, bb, n_points, n_estimates, qrng, log)
