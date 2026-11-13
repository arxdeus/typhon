# Copyright (C) 2026 the Typhon authors
# Ported from mne 1.13.2 (BSD-3-Clause)
#
# mne/stats/multi_comp.py, mne/stats/permutations.py

import numpy as np
from math import sqrt


def _ecdf(x):
    nobs = len(x)
    return np.arange(1, nobs + 1) / float(nobs)


def fdr_correction(pvals, alpha: float = 0.05, method: str = "indep"):
    """FDR correction (Benjamini/Hochberg or Benjamini/Yekutieli) (mne)."""
    p = np.asarray(pvals, dtype=float)
    shape_init = p.shape
    pv = p.ravel()
    pvals_sortind = np.argsort(pv)
    pvals_sorted = pv[pvals_sortind]
    sortrevind = pvals_sortind.argsort()
    if method in ["i", "indep", "p", "poscorr"]:
        ecdffactor = _ecdf(pvals_sorted)
    elif method in ["n", "negcorr"]:
        cm = float(np.sum(1.0 / np.arange(1, len(pvals_sorted) + 1)))
        ecdffactor = _ecdf(pvals_sorted) / cm
    else:
        raise ValueError("Method should be 'indep' and 'negcorr'")
    reject = pvals_sorted < (ecdffactor * alpha)
    rejectmax = 0
    for i in range(len(reject)):
        if reject[i]:
            rejectmax = i
    reject[:rejectmax] = True
    raw = pvals_sorted / ecdffactor
    n = len(raw)
    corrected = np.empty(n)
    if n > 0:
        cur = raw[n - 1]
        for i in range(n - 1, -1, -1):
            if raw[i] < cur:
                cur = raw[i]
            corrected[i] = cur if cur <= 1.0 else 1.0
    return reject[sortrevind].reshape(shape_init), corrected[sortrevind].reshape(shape_init)


def bonferroni_correction(pval, alpha: float = 0.05):
    """P-value correction with Bonferroni method (mne)."""
    p = np.asarray(pval, dtype=float)
    pc = np.minimum(p * float(p.size), 1.0)
    return pc < alpha, pc


def _check_rng(rng, seed):
    """mne _legacy_rng: seed= -> RandomState, rng= -> default_rng(rng)."""
    if not isinstance(seed, NoneType):
        if isinstance(seed, int):
            return np.random.RandomState(seed)
        else:
            return seed
    elif isinstance(rng, int) or isinstance(rng, NoneType):
        return np.random.default_rng(rng)
    else:
        return rng


def _max_stat(X, X2, perms, dof_scaling: float):
    n_samples = len(X)
    mus = np.dot(perms, X) / float(n_samples)
    stds = np.sqrt(X2[None, :] - mus * mus) * dof_scaling
    return np.max(np.abs(mus) / (stds / sqrt(n_samples)), axis=1)


def permutation_t_test(X, n_permutations = 10000, tail: int = 0, n_jobs = None,
                       verbose = None, *, rng = None, seed = None):
    """One sample/paired sample permutation test based on a t-statistic (mne)."""
    from .cluster_level import _get_1samp_orders
    r = _check_rng(rng, seed)
    Xf = np.asarray(X, dtype=float)
    n_samples = Xf.shape[0]
    X2 = np.mean(Xf ** 2, axis=0)
    mu0 = np.mean(Xf, axis=0)
    dof_scaling = sqrt(n_samples / (n_samples - 1.0))
    std0 = np.sqrt(X2 - mu0 ** 2) * dof_scaling
    T_obs = np.mean(Xf, axis=0) / (std0 / sqrt(n_samples))
    orders, _, extra = _get_1samp_orders(n_samples, n_permutations, tail, r)
    perms = 2 * orders - 1
    max_abs = _max_stat(Xf, X2, perms.astype(float), dof_scaling)
    max_abs = np.concatenate((max_abs, np.array([float(np.abs(T_obs).max())])))
    H0 = np.sort(max_abs)
    nt = T_obs.shape[0]
    p_values = np.empty(nt)
    nh = float(H0.shape[0])
    for i in range(nt):
        c = 0
        for h in H0:
            if tail == 0:
                c += int(h >= abs(T_obs[i]))
            elif tail == 1:
                c += int(h >= T_obs[i])
            else:
                c += int(-h <= T_obs[i])
        p_values[i] = c / nh
    return T_obs, p_values, H0


def bootstrap_confidence_interval(arr, ci: float = 0.95, n_bootstraps: int = 2000,
                                  stat_fun = "mean", *, rng = None, random_state = None):
    """Get confidence intervals from non-parametric bootstrap (mne)."""
    r = _check_rng(rng, random_state)
    n_trials = arr.shape[0]
    indices = np.arange(n_trials)
    boot_indices = r.choice(n_trials, replace=True, size=(n_bootstraps, n_trials))
    first = arr[boot_indices[0]]
    if isinstance(stat_fun, str):
        if stat_fun != "mean" and stat_fun != "median":
            raise ValueError("stat_fun must be 'mean', 'median' or callable.")
    s0 = _boot_stat(stat_fun, first)
    shp = (n_bootstraps,) + s0.shape
    stat = np.empty(shp)
    stat[0] = s0
    for i in range(1, n_bootstraps):
        stat[i] = _boot_stat(stat_fun, arr[boot_indices[i]])
    q = ((1 - ci) / 2 * 100, (1 - (1 - ci) / 2) * 100)
    return np.percentile(stat, q, axis=0)


def _boot_stat(stat_fun, x):
    if isinstance(stat_fun, str):
        if stat_fun == "mean":
            return np.asarray(x.mean(axis=0))
        else:
            return np.asarray(np.median(x, axis=0))
    else:
        return np.asarray(stat_fun(x))


def _ci(arr, ci: float = 0.95, method: str = "bootstrap", n_bootstraps: int = 2000, rng = None):
    if method == "bootstrap":
        return bootstrap_confidence_interval(arr, ci=ci, n_bootstraps=n_bootstraps, rng=rng)
    else:
        from .parametric import _parametric_ci
        lo, hi = _parametric_ci(arr, ci=ci)
        return np.array([lo, hi])
