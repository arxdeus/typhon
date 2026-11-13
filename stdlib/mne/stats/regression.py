# Copyright (C) 2026 the Typhon authors
# Ported from mne 1.13.2 (BSD-3-Clause)
#
# mne/stats/regression.py (linear_regression on Epochs)

import numpy as np
import scipy.linalg
from scipy import stats as _sst


@tuple
class lm:
    beta: T
    stderr: T
    t_val: T
    p_val: T
    mlog10_p_val: T
    T: type


def _fit_lm(data: np.ndarray[float, 3], design_matrix, names: List[str]):
    n_samples = data.shape[0]
    n_ch = data.shape[1]
    n_t = data.shape[2]
    n_features = n_ch * n_t
    dm = np.asarray(design_matrix, dtype=float)
    if dm.ndim != 2:
        raise ValueError("Design matrix must be a 2d array")
    n_rows = dm.shape[0]
    n_predictors = dm.shape[1]
    if n_samples != n_rows:
        raise ValueError("Number of rows in design matrix must be equal to number of observations")
    if n_predictors != len(names):
        raise ValueError("Number of regressor names must be equal to number of column in design matrix")
    y = np.ascontiguousarray(data.reshape((n_samples, n_features)))
    betas, resid_sum_squares, _, _ = scipy.linalg.lstsq(dm, y)
    df = n_rows - n_predictors
    sqrt_noise_var = np.sqrt(resid_sum_squares / df).reshape((n_ch, n_t))
    design_invcov = scipy.linalg.inv(dm.T @ dm)
    unscaled_stderrs = np.sqrt(np.diag(design_invcov))
    tiny = np.finfo(np.float64).tiny
    out = Dict[str, List[np.ndarray[float, 2]]]()
    for k in range(n_predictors):
        b = betas[k].reshape((n_ch, n_t)).copy()
        se = sqrt_noise_var * unscaled_stderrs[k]
        tv = np.empty((n_ch, n_t))
        pv = np.empty((n_ch, n_t))
        for i in range(n_ch):
            for j in range(n_t):
                if se[i, j] > 0:
                    tv[i, j] = b[i, j] / se[i, j]
                    cdf = float(_sst.t.cdf(abs(tv[i, j]), df))
                    pv[i, j] = min(max((1.0 - cdf) * 2.0, tiny), 1.0)
                elif b[i, j] > 0:
                    tv[i, j] = np.inf
                    pv[i, j] = tiny
                else:
                    tv[i, j] = 0.0
                    pv[i, j] = 1.0
        out[names[k]] = [b, se, tv, pv, -np.log10(pv)]
    return out


def linear_regression(inst, design_matrix, names = None):
    """Fit Ordinary Least Squares (OLS) regression to Epochs (mne).

    Returns dict name -> lm(beta, stderr, t_val, p_val, mlog10_p_val) of Evoked."""
    from .._containers import Evoked
    dm = np.asarray(design_matrix, dtype=float)
    nm = List[str]()
    if isinstance(names, NoneType):
        for i in range(dm.shape[1]):
            nm.append(f"x{i}")
    else:
        for n in names:
            nm.append(str(n))
    data = inst.get_data()
    params = _fit_lm(data, dm, nm)
    res = Dict[str, lm[Evoked]]()
    for name in nm:
        p = params[name]
        ev = [Evoked(p[i], inst.info.copy(), inst.times.copy(), len(inst)) for i in range(5)]
        res[name] = lm[Evoked](ev[0], ev[1], ev[2], ev[3], ev[4])
    return res
