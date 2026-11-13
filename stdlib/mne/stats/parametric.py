# Copyright (C) 2026 the Typhon authors
# Ported from mne 1.13.2 (BSD-3-Clause)
#
# mne/stats/parametric.py

import numpy as np
from scipy import stats as _sst
from scipy.signal import detrend as _detrend

_ASCII_UPPER = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _check_method(method: str):
    if method != "absolute" and method != "relative":
        raise ValueError(f"Invalid value for the 'method' parameter. Allowed values are 'absolute' and 'relative', but got '{method}' instead.")


def ttest_1samp_no_p(X, sigma: float = 0, method: str = "relative"):
    """One-sample t-test without p-value computation (mne)."""
    _check_method(method)
    var = np.var(X, axis=0, ddof=1)
    if sigma > 0:
        limit = sigma * float(np.max(var)) if method == "relative" else float(sigma)
        var += limit
    return np.mean(X, axis=0) / np.sqrt(var / X.shape[0])


def ttest_ind_no_p(a, b, equal_var: bool = True, sigma: float = 0.0):
    """Independent-sample t-test without p-value computation (mne)."""
    v1 = np.var(a, axis=0, ddof=1)
    v2 = np.var(b, axis=0, ddof=1)
    n1 = a.shape[0]
    n2 = b.shape[0]
    if equal_var:
        df = n1 + n2 - 2.0
        var = ((n1 - 1) * v1 + (n2 - 1) * v2) / df
        var = var * (1.0 / n1 + 1.0 / n2)
    else:
        vn1 = v1 / n1
        vn2 = v2 / n2
        var = vn1 + vn2
    if sigma > 0:
        var += sigma * float(np.max(var))
    denom = np.sqrt(var)
    d = np.mean(a, 0) - np.mean(b, 0)
    return d / denom


def _f_oneway_list(args, sigma: float = 0.0, method: str = "relative"):
    _check_method(method)
    if sigma < 0:
        raise ValueError(f"sigma must be >= 0, got {sigma}")
    n_classes = len(args)
    n_samples = 0
    for a in args:
        n_samples += len(a)
    ss_alldata = np.sum(args[0] ** 2, axis=0)
    sums0 = np.sum(args[0], axis=0)
    sum_all = sums0.copy()
    for k in range(1, n_classes):
        ss_alldata = ss_alldata + np.sum(args[k] ** 2, axis=0)
        sum_all = sum_all + np.sum(args[k], axis=0)
    square_of_sums_alldata = sum_all ** 2
    sstot = ss_alldata - square_of_sums_alldata / float(n_samples)
    ssbn = np.zeros_like(sstot)
    for k in range(n_classes):
        s = np.sum(args[k], axis=0)
        ssbn = ssbn + (s ** 2) / len(args[k])
    ssbn -= square_of_sums_alldata / float(n_samples)
    sswn = sstot - ssbn
    dfbn = n_classes - 1
    dfwn = n_samples - n_classes
    msb = ssbn / float(dfbn)
    msw = sswn / float(dfwn)
    if sigma > 0.0:
        sigma_sq = sigma * float(np.median(msw)) if method == "relative" else float(sigma)
        msw = (sswn + sigma_sq) / float(dfwn)
    return msb / msw


def f_oneway(*args, sigma: float = 0.0, method: str = "relative"):
    """Call scipy.stats.f_oneway, but return only f-value (mne)."""
    lst = [args[0]]
    for i in static.range(1, static.len(args)):
        lst.append(args[i])
    return _f_oneway_list(lst, sigma, method)


def _get_contrast_indices(effect_idx: int, n_factors: int) -> List[int]:
    out = [0] * n_factors
    v = effect_idx
    for i in range(n_factors):
        out[n_factors - 1 - i] = v & 1
        v >>= 1
    return out


def _map_effects(n_factors: int, effects) -> Tuple[List[int], List[str]]:
    if n_factors > 26:
        raise ValueError("Maximum number of factors supported is 26")
    factor_names = [_ASCII_UPPER[i] for i in range(n_factors)]
    eff_list: Optional[List[str]] = None
    star = ""
    if isinstance(effects, str):
        if "*" in effects and ":" in effects:
            raise ValueError('Not "*" and ":" permitted in effects')
        elif "+" in effects and ":" in effects:
            raise ValueError('Not "+" and ":" permitted in effects')
        elif effects == "all":
            eff_list = None
        elif len(effects) == 1 or ":" in effects:
            eff_list = [effects]
        elif "+" in effects:
            eff_list = effects.split("+")
        elif "*" in effects:
            star = effects
        else:
            raise ValueError(f'"{effects}" is not a valid option for "effects"')
    else:
        eff_list = [str(e) for e in effects]
    names = List[str]()
    for this_effect in range(2 ** n_factors - 1):
        ci = _get_contrast_indices(this_effect + 1, n_factors)
        this_name = List[str]()
        for j in range(n_factors):
            if ci[j] == 1:
                this_name.append(factor_names[n_factors - 1 - j])
        this_name.sort()
        names.append(":".join(this_name))
    effects_ = names if (eff_list is None) else eff_list
    selection = [names.index(sel) for sel in effects_]
    names2 = [names[sel] for sel in selection]
    if star != "":
        sel_ind = names2.index(star.replace("*", ":")) + 1
        names2 = names2[:sel_ind]
        selection = selection[:sel_ind]
    return selection, names2


def _iter_contrasts(n_subjects: int, factor_levels: List[int], effect_picks: List[int]):
    n_factors = len(factor_levels)
    sc = List[List[np.ndarray[float, 2]]]()
    for n_levels in factor_levels:
        sc.append([np.ones((n_levels, 1)), _detrend(np.eye(n_levels), type="constant")])
    out = List[Tuple[np.ndarray[float, 2], int, int]]()
    for this_effect in effect_picks:
        ci = _get_contrast_indices(this_effect + 1, n_factors)
        c_ = sc[0][ci[n_factors - 1]]
        for i_contrast in range(1, n_factors):
            this_contrast = ci[n_factors - 1 - i_contrast]
            c_ = np.kron(c_, sc[i_contrast][this_contrast])
        df1 = int(np.linalg.matrix_rank(c_))
        df2 = df1 * (n_subjects - 1)
        out.append((c_, df1, df2))
    return out


def f_threshold_mway_rm(n_subjects: int, factor_levels, effects = "A*B", pvalue: float = 0.05) -> List[float]:
    """Compute F-value thresholds for a mixed-effects ANOVA (mne).

    Always returns a list (mne returns a bare float for a single effect)."""
    fl = [int(v) for v in factor_levels]
    effect_picks, _ = _map_effects(len(fl), effects)
    F_threshold = List[float]()
    for c_, df1, df2 in _iter_contrasts(n_subjects, fl, effect_picks):
        F_threshold.append(float(_sst.f.isf(pvalue, df1, df2)))
    return F_threshold


def f_mway_rm(data, factor_levels, effects = "all", correction: bool = False,
              return_pvals: bool = True):
    """Compute M-way repeated measures ANOVA for fully balanced designs (mne).

    Returns (fvals, pvals) as 2D arrays (n_effects, n_tests) (mne squeezes)."""
    fl = [int(v) for v in factor_levels]
    d3 = np.asarray(data, dtype=float).reshape((data.shape[0], data.shape[1], -1))
    effect_picks, _ = _map_effects(len(fl), effects)
    n_obs = d3.shape[2]
    n_replications = d3.shape[0]
    dr = np.ascontiguousarray(np.moveaxis(d3, 2, 0))
    contrasts = _iter_contrasts(n_replications, fl, effect_picks)
    fvalues = np.zeros((len(contrasts), n_obs))
    pvalues = np.zeros((len(contrasts), n_obs) if return_pvals else (len(contrasts), 0))
    for k in range(len(contrasts)):
        c_, df1, df2 = contrasts[k]
        nc = c_.shape[1]
        y = np.empty((n_obs, n_replications, nc))
        for o in range(n_obs):
            y[o] = dr[o] @ c_
        b = np.mean(y, axis=1)
        fvals = np.empty(n_obs)
        eps = np.empty(n_obs)
        for o in range(n_obs):
            ss = 0.0
            sy = 0.0
            for r in range(n_replications):
                for c in range(nc):
                    ss += y[o, r, c] * b[o, c]
                    sy += y[o, r, c] * y[o, r, c]
            mse = (sy - ss) / (float(df2) / float(df1))
            fvals[o] = ss / mse
            if correction:
                v = y[o].T @ y[o]
                eps[o] = float(np.trace(v)) ** 2 / (df1 * float(np.sum(v * v)))
        fvalues[k] = fvals
        if return_pvals:
            for o in range(n_obs):
                d1 = float(df1)
                d2 = float(df2)
                if correction:
                    d1 = max(d1 * eps[o], 1.0)
                    d2 = max(d2 * eps[o], 1.0)
                pvalues[k, o] = float(_sst.f.sf(fvals[o], d1, d2))
    return fvalues, pvalues


def _parametric_ci(arr, ci: float = 0.95):
    mean = arr.mean(0)
    sigma = _sst.sem(arr, 0)
    return _sst.t.interval(ci, arr.shape[0], loc=mean, scale=sigma)
