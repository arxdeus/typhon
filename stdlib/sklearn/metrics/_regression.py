# Copyright (C) 2026 the Typhon authors
# Ported from scikit-learn 1.9.1 (BSD-3-Clause)

# Regression metrics. Targets are handled as (n_samples, n_outputs) float
# matrices internally; `multioutput` may be "uniform_average",
# "raw_values" (array result, selected statically for literal strings),
# "variance_weighted" (r2 / explained variance) or an array of weights.

import numpy as np

_EPS64 = 2.220446049250313e-16

def _as2d(y):
    a = np.asarray(y, dtype=float)
    if static.len(a.shape) == 1:
        return a.reshape(-1, 1)
    else:
        return a.copy()

def _check_reg(y_true, y_pred, sample_weight):
    T = _as2d(y_true)
    P = _as2d(y_pred)
    if T.shape[0] != P.shape[0]:
        raise ValueError(f"Found input variables with inconsistent numbers of samples: [{T.shape[0]}, {P.shape[0]}]")
    if T.shape[1] != P.shape[1]:
        raise ValueError(f"y_true and y_pred have different number of output ({T.shape[1]}!={P.shape[1]})")
    for v in T.reshape(-1):
        if not np.isfinite(v):
            raise ValueError("Input y_true contains NaN or infinity.")
    for v in P.reshape(-1):
        if not np.isfinite(v):
            raise ValueError("Input y_pred contains NaN or infinity.")
    if not isinstance(sample_weight, NoneType):
        sw = np.asarray(sample_weight, dtype=float)
        if sw.shape[0] != T.shape[0]:
            raise ValueError(f"Found input variables with inconsistent numbers of samples: [{T.shape[0]}, {sw.shape[0]}]")
    return T, P

def _colavg(E, sample_weight):
    if isinstance(sample_weight, NoneType):
        return np.average(E, axis=0)
    else:
        return np.average(E, axis=0, weights=np.asarray(sample_weight, dtype=float))

def _ndim1(y) -> Literal[bool]:
    return static.len(np.asarray(y).shape) == 1

def _finish(errs, multioutput):
    """Average per-output errors (array multioutput = weights)."""
    W = np.asarray(multioutput, dtype=float)
    if W.shape[0] != errs.shape[0]:
        raise ValueError(f"There must be equally many custom weights ({W.shape[0]}) as outputs ({errs.shape[0]}).")
    return float(np.average(errs, weights=W))

@overload
def _finish(errs, multioutput: str):
    if multioutput == "raw_values":
        raise ValueError("multioutput='raw_values' must be passed as a literal string")
    if multioutput != "uniform_average":
        raise ValueError(f"Allowed 'multioutput' string values are ('raw_values', 'uniform_average', 'variance_weighted'). You provided multioutput={multioutput!r}.")
    return float(np.average(errs))

@overload
def _finish(errs, multioutput: Literal[str]):
    if multioutput == "raw_values":
        return errs
    else:
        if multioutput != "uniform_average":
            raise ValueError(f"Allowed 'multioutput' string values are ('raw_values', 'uniform_average', 'variance_weighted'). You provided multioutput={multioutput!r}.")
        return float(np.average(errs))

def mean_absolute_error(y_true, y_pred, *, sample_weight = None, multioutput = "uniform_average"):
    T, P = _check_reg(y_true, y_pred, sample_weight)
    return _finish(_colavg(np.abs(P - T), sample_weight), multioutput)

@overload
def mean_absolute_error(y_true, y_pred, *, sample_weight = None, multioutput: Literal[str] = "uniform_average"):
    T, P = _check_reg(y_true, y_pred, sample_weight)
    return _finish(_colavg(np.abs(P - T), sample_weight), multioutput)

def mean_squared_error(y_true, y_pred, *, sample_weight = None, multioutput = "uniform_average"):
    T, P = _check_reg(y_true, y_pred, sample_weight)
    return _finish(_colavg((T - P) ** 2, sample_weight), multioutput)

@overload
def mean_squared_error(y_true, y_pred, *, sample_weight = None, multioutput: Literal[str] = "uniform_average"):
    T, P = _check_reg(y_true, y_pred, sample_weight)
    return _finish(_colavg((T - P) ** 2, sample_weight), multioutput)

def root_mean_squared_error(y_true, y_pred, *, sample_weight = None, multioutput = "uniform_average"):
    T, P = _check_reg(y_true, y_pred, sample_weight)
    return _finish(np.sqrt(_colavg((T - P) ** 2, sample_weight)), multioutput)

@overload
def root_mean_squared_error(y_true, y_pred, *, sample_weight = None, multioutput: Literal[str] = "uniform_average"):
    T, P = _check_reg(y_true, y_pred, sample_weight)
    return _finish(np.sqrt(_colavg((T - P) ** 2, sample_weight)), multioutput)

def _check_log(T, P, name: str):
    if np.any(T <= -1) or np.any(P <= -1):
        raise ValueError(f"{name} cannot be used when targets contain values less than or equal to -1.")

def mean_squared_log_error(y_true, y_pred, *, sample_weight = None, multioutput = "uniform_average"):
    T, P = _check_reg(y_true, y_pred, sample_weight)
    _check_log(T, P, "Mean Squared Logarithmic Error")
    return mean_squared_error(np.log1p(T), np.log1p(P), sample_weight=sample_weight, multioutput=multioutput)

@overload
def mean_squared_log_error(y_true, y_pred, *, sample_weight = None, multioutput: Literal[str] = "uniform_average"):
    T, P = _check_reg(y_true, y_pred, sample_weight)
    _check_log(T, P, "Mean Squared Logarithmic Error")
    return mean_squared_error(np.log1p(T), np.log1p(P), sample_weight=sample_weight, multioutput=multioutput)

def root_mean_squared_log_error(y_true, y_pred, *, sample_weight = None, multioutput = "uniform_average"):
    T, P = _check_reg(y_true, y_pred, sample_weight)
    _check_log(T, P, "Root Mean Squared Logarithmic Error")
    return root_mean_squared_error(np.log1p(T), np.log1p(P), sample_weight=sample_weight, multioutput=multioutput)

@overload
def root_mean_squared_log_error(y_true, y_pred, *, sample_weight = None, multioutput: Literal[str] = "uniform_average"):
    T, P = _check_reg(y_true, y_pred, sample_weight)
    _check_log(T, P, "Root Mean Squared Logarithmic Error")
    return root_mean_squared_error(np.log1p(T), np.log1p(P), sample_weight=sample_weight, multioutput=multioutput)

def mean_absolute_percentage_error(y_true, y_pred, *, sample_weight = None, multioutput = "uniform_average"):
    T, P = _check_reg(y_true, y_pred, sample_weight)
    mape = np.abs(P - T) / np.maximum(np.abs(T), _EPS64)
    return _finish(_colavg(mape, sample_weight), multioutput)

@overload
def mean_absolute_percentage_error(y_true, y_pred, *, sample_weight = None, multioutput: Literal[str] = "uniform_average"):
    T, P = _check_reg(y_true, y_pred, sample_weight)
    mape = np.abs(P - T) / np.maximum(np.abs(T), _EPS64)
    return _finish(_colavg(mape, sample_weight), multioutput)

def mean_pinball_loss(y_true, y_pred, *, sample_weight = None, alpha: float = 0.5,
                      multioutput = "uniform_average"):
    T, P = _check_reg(y_true, y_pred, sample_weight)
    return _finish(_pinball_cols(T, P, sample_weight, alpha), multioutput)

@overload
def mean_pinball_loss(y_true, y_pred, *, sample_weight = None, alpha: float = 0.5,
                      multioutput: Literal[str] = "uniform_average"):
    T, P = _check_reg(y_true, y_pred, sample_weight)
    return _finish(_pinball_cols(T, P, sample_weight, alpha), multioutput)

def _pinball_cols(T, P, sample_weight, alpha: float):
    diff = T - P
    sign = (diff >= 0).astype(float)
    loss = alpha * sign * diff - (1 - alpha) * (1 - sign) * diff
    return _colavg(loss, sample_weight)

def _weighted_percentile_col(a, w, percentile_rank: float, average: bool) -> float:
    """sklearn.utils.stats._weighted_percentile for one column."""
    n = a.shape[0]
    allzero = True
    for v in w:
        if v != 0:
            allzero = False
    if allzero:
        return np.nan
    sidx = np.argsort(a)
    sw = w[sidx].copy()
    if np.isnan(a[sidx[n - 1]]):
        for i in range(n):
            if np.isnan(a[sidx[i]]):
                sw[i] = 0.0
    cdf = np.cumsum(sw)
    adj = percentile_rank / 100 * cdf[n - 1]
    if adj == 0:
        adj = np.nextafter(adj, adj + 1)
    pi = int(np.searchsorted(cdf, adj))
    max_idx = n - 1
    pi = min(max(pi, 0), max_idx)
    val = a[sidx[pi]]
    if not average:
        return val
    frac = cdf[pi] - adj
    if frac > _EPS64:
        return val
    p1 = min(pi + 1, max_idx)
    p1s = sidx[p1]
    if w[p1s] == 0:
        nxt = int(np.searchsorted(cdf, cdf[pi], side="right"))
        if nxt > max_idx:
            nxt = pi
        p1s = sidx[nxt]
    return (val + a[p1s]) / 2

def median_absolute_error(y_true, y_pred, *, multioutput = "uniform_average", sample_weight = None):
    T, P = _check_reg(y_true, y_pred, sample_weight)
    E = np.abs(P - T)
    k = E.shape[1]
    if isinstance(sample_weight, NoneType):
        errs = np.median(E, axis=0)
    else:
        sw = np.asarray(sample_weight, dtype=float)
        errs = np.empty(k)
        for j in range(k):
            errs[j] = _weighted_percentile_col(np.ascontiguousarray(E[:, j]), sw, 50.0, True)
    return _finish(errs, multioutput)

@overload
def median_absolute_error(y_true, y_pred, *, multioutput: Literal[str] = "uniform_average", sample_weight = None):
    T, P = _check_reg(y_true, y_pred, sample_weight)
    E = np.abs(P - T)
    k = E.shape[1]
    if isinstance(sample_weight, NoneType):
        errs = np.median(E, axis=0)
    else:
        sw = np.asarray(sample_weight, dtype=float)
        errs = np.empty(k)
        for j in range(k):
            errs[j] = _weighted_percentile_col(np.ascontiguousarray(E[:, j]), sw, 50.0, True)
    return _finish(errs, multioutput)

def max_error(y_true, y_pred) -> float:
    T, P = _check_reg(y_true, y_pred, None)
    if T.shape[1] > 1:
        raise ValueError("Multioutput not supported in max_error")
    return float(np.max(np.abs(T - P)))

def _scores(num, den, force_finite: bool):
    k = num.shape[0]
    scores = np.ones(k)
    for j in range(k):
        if not force_finite:
            scores[j] = 1 - num[j] / den[j]
        elif den[j] != 0 and num[j] != 0:
            scores[j] = 1 - num[j] / den[j]
        elif num[j] != 0 and den[j] == 0:
            scores[j] = 0.0
    return scores

def _assemble(num, den, multioutput, force_finite: bool):
    return _finish_vw(_scores(num, den, force_finite), den, multioutput)

@overload
def _assemble(num, den, multioutput: Literal[str], force_finite: bool):
    return _finish_vw(_scores(num, den, force_finite), den, multioutput)

def _finish_vw(scores, den, multioutput):
    return _finish(scores, multioutput)

@overload
def _finish_vw(scores, den, multioutput: str):
    if multioutput == "variance_weighted":
        if not np.any(den != 0):
            return float(np.average(scores))
        return float(np.average(scores, weights=den))
    return _finish(scores, multioutput)

@overload
def _finish_vw(scores, den, multioutput: Literal[str]):
    if multioutput == "variance_weighted":
        if not np.any(den != 0):
            return float(np.average(scores))
        return float(np.average(scores, weights=den))
    else:
        return _finish(scores, multioutput)

def r2_score(y_true, y_pred, *, sample_weight = None, multioutput = "uniform_average",
             force_finite: bool = True):
    T, P = _check_reg(y_true, y_pred, sample_weight)
    k = T.shape[1]
    num = np.full(k, np.nan)
    den = np.ones(k)
    if T.shape[0] >= 2:
        if isinstance(sample_weight, NoneType):
            num = np.sum((T - P) ** 2, axis=0)
            den = np.sum((T - np.average(T, axis=0)) ** 2, axis=0)
        else:
            sw = np.asarray(sample_weight, dtype=float)
            w = sw[:, None]
            num = np.sum(w * (T - P) ** 2, axis=0)
            den = np.sum(w * (T - np.average(T, axis=0, weights=sw)) ** 2, axis=0)
    return _assemble(num, den, multioutput, force_finite and T.shape[0] >= 2)

@overload
def r2_score(y_true, y_pred, *, sample_weight = None, multioutput: Literal[str] = "uniform_average",
             force_finite: bool = True):
    T, P = _check_reg(y_true, y_pred, sample_weight)
    k = T.shape[1]
    num = np.full(k, np.nan)
    den = np.ones(k)
    if T.shape[0] >= 2:
        if isinstance(sample_weight, NoneType):
            num = np.sum((T - P) ** 2, axis=0)
            den = np.sum((T - np.average(T, axis=0)) ** 2, axis=0)
        else:
            sw = np.asarray(sample_weight, dtype=float)
            w = sw[:, None]
            num = np.sum(w * (T - P) ** 2, axis=0)
            den = np.sum(w * (T - np.average(T, axis=0, weights=sw)) ** 2, axis=0)
    return _assemble(num, den, multioutput, force_finite and T.shape[0] >= 2)

def explained_variance_score(y_true, y_pred, *, sample_weight = None,
                             multioutput = "uniform_average", force_finite: bool = True):
    T, P = _check_reg(y_true, y_pred, sample_weight)
    yda = _colavg(T - P, sample_weight)
    num = _colavg((T - P - yda) ** 2, sample_weight)
    yta = _colavg(T, sample_weight)
    den = _colavg((T - yta) ** 2, sample_weight)
    return _assemble(num, den, multioutput, force_finite)

@overload
def explained_variance_score(y_true, y_pred, *, sample_weight = None,
                             multioutput: Literal[str] = "uniform_average", force_finite: bool = True):
    T, P = _check_reg(y_true, y_pred, sample_weight)
    yda = _colavg(T - P, sample_weight)
    num = _colavg((T - P - yda) ** 2, sample_weight)
    yta = _colavg(T, sample_weight)
    den = _colavg((T - yta) ** 2, sample_weight)
    return _assemble(num, den, multioutput, force_finite)

def _xlogy_arr(x, y):
    out = np.empty(x.shape[0])
    for i in range(x.shape[0]):
        out[i] = 0.0 if x[i] == 0 else x[i] * np.log(y[i])
    return out

def _tweedie_dev(t, p, sample_weight, power: float) -> float:
    q = power
    if q < 0:
        dev = 2 * (np.power(np.where(t > 0, t, 0.0), 2 - q) / ((1 - q) * (2 - q))
                   - t * np.power(p, 1 - q) / (1 - q) + np.power(p, 2 - q) / (2 - q))
    elif q == 0:
        dev = (t - p) ** 2
    elif q == 1:
        dev = 2 * (_xlogy_arr(t, t / p) - t + p)
    elif q == 2:
        dev = 2 * (np.log(p / t) + t / p - 1)
    else:
        dev = 2 * (np.power(t, 2 - q) / ((1 - q) * (2 - q))
                   - t * np.power(p, 1 - q) / (1 - q) + np.power(p, 2 - q) / (2 - q))
    if isinstance(sample_weight, NoneType):
        return float(np.average(dev))
    else:
        return float(np.average(dev, weights=np.asarray(sample_weight, dtype=float)))

def _tweedie_1d(y_true, y_pred, sample_weight, name: str):
    T, P = _check_reg(y_true, y_pred, sample_weight)
    if T.shape[1] > 1:
        raise ValueError(f"Multioutput not supported in {name}")
    return np.ascontiguousarray(T[:, 0]), np.ascontiguousarray(P[:, 0])

def _check_tweedie(t, p, power: float):
    msg = f"Mean Tweedie deviance error with power={power} can only be used on "
    if power < 0:
        if np.any(p <= 0):
            raise ValueError(msg + "strictly positive y_pred.")
    elif power == 0:
        pass
    elif 1 <= power < 2:
        if np.any(t < 0) or np.any(p <= 0):
            raise ValueError(msg + "non-negative y and strictly positive y_pred.")
    elif power >= 2:
        if np.any(t <= 0) or np.any(p <= 0):
            raise ValueError(msg + "strictly positive y and y_pred.")
    else:
        raise ValueError(f"Tweedie deviance is only defined for power<=0 and power>=1, got power={power}")

def mean_tweedie_deviance(y_true, y_pred, *, sample_weight = None, power: float = 0) -> float:
    t, p = _tweedie_1d(y_true, y_pred, sample_weight, "mean_tweedie_deviance")
    _check_tweedie(t, p, float(power))
    return _tweedie_dev(t, p, sample_weight, float(power))

def mean_poisson_deviance(y_true, y_pred, *, sample_weight = None) -> float:
    return mean_tweedie_deviance(y_true, y_pred, sample_weight=sample_weight, power=1.0)

def mean_gamma_deviance(y_true, y_pred, *, sample_weight = None) -> float:
    return mean_tweedie_deviance(y_true, y_pred, sample_weight=sample_weight, power=2.0)

def d2_tweedie_score(y_true, y_pred, *, sample_weight = None, power: float = 0) -> float:
    t, p = _tweedie_1d(y_true, y_pred, sample_weight, "d2_tweedie_score")
    if t.shape[0] < 2:
        return np.nan
    num = mean_tweedie_deviance(t, p, sample_weight=sample_weight, power=power)
    if isinstance(sample_weight, NoneType):
        ya = float(np.average(t))
    else:
        ya = float(np.average(t, weights=np.asarray(sample_weight, dtype=float)))
    den = _tweedie_dev(t, np.full(t.shape[0], ya), sample_weight, float(power))
    return 1 - num / den

def d2_pinball_score(y_true, y_pred, *, sample_weight = None, alpha: float = 0.5,
                     multioutput = "uniform_average"):
    T, P = _check_reg(y_true, y_pred, sample_weight)
    k = T.shape[1]
    if T.shape[0] < 2:
        return _assemble(np.full(k, np.nan), np.ones(k), multioutput, False)
    num = _pinball_cols(T, P, sample_weight, alpha)
    sw = np.ones(T.shape[0]) if isinstance(sample_weight, NoneType) else np.asarray(sample_weight, dtype=float)
    Q = np.empty(T.shape)
    for j in range(k):
        qv = _weighted_percentile_col(np.ascontiguousarray(T[:, j]), sw, alpha * 100, True)
        for i in range(T.shape[0]):
            Q[i, j] = qv
    den = _pinball_cols(T, Q, sw, alpha)
    return _assemble(num, den, multioutput, True)

@overload
def d2_pinball_score(y_true, y_pred, *, sample_weight = None, alpha: float = 0.5,
                     multioutput: Literal[str] = "uniform_average"):
    T, P = _check_reg(y_true, y_pred, sample_weight)
    k = T.shape[1]
    if T.shape[0] < 2:
        return _assemble(np.full(k, np.nan), np.ones(k), multioutput, False)
    num = _pinball_cols(T, P, sample_weight, alpha)
    sw = np.ones(T.shape[0]) if isinstance(sample_weight, NoneType) else np.asarray(sample_weight, dtype=float)
    Q = np.empty(T.shape)
    for j in range(k):
        qv = _weighted_percentile_col(np.ascontiguousarray(T[:, j]), sw, alpha * 100, True)
        for i in range(T.shape[0]):
            Q[i, j] = qv
    den = _pinball_cols(T, Q, sw, alpha)
    return _assemble(num, den, multioutput, True)

def d2_absolute_error_score(y_true, y_pred, *, sample_weight = None, multioutput = "uniform_average"):
    return d2_pinball_score(y_true, y_pred, sample_weight=sample_weight, alpha=0.5, multioutput=multioutput)

@overload
def d2_absolute_error_score(y_true, y_pred, *, sample_weight = None, multioutput: Literal[str] = "uniform_average"):
    return d2_pinball_score(y_true, y_pred, sample_weight=sample_weight, alpha=0.5, multioutput=multioutput)

