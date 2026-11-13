# Copyright (C) 2026 the Typhon authors
# Ported from scipy 1.18.1 (BSD-3-Clause)

# tukey_hsd and dunnett multiple-comparison tests.

import numpy as np
import internal.static as static
import scipy.special._kernels as K
from scipy.optimize import minimize_scalar
from ._distn_infrastructure import check_random_state, NAN, INF
from ._studentized_range import _sr_cdf, studentized_range
from ._multivariate import _mvt_box

class ConfidenceIntervalArr[L, H]:
    low: L
    high: H
    def __init__(self, low: L, high: H):
        self.low = low
        self.high = high
    def __getitem__(self, i: Literal[int]):
        if i == 0 or i == -2:
            return self.low
        elif i == 1 or i == -1:
            return self.high
        else:
            compile_error("index out of range")
    def __len__(self):
        return 2
    def __iter__(self):
        yield self.low
        yield self.high

# ------------------------------------------------------------------ tukey_hsd

class TukeyHSDResult:
    statistic: np.ndarray[float, 2]
    pvalue: np.ndarray[float, 2]
    _nt: int
    _df: np.ndarray[float, 2]
    _se: np.ndarray[float, 2]

    def __init__(self, statistic: np.ndarray[float, 2], pvalue: np.ndarray[float, 2], nt: int,
                 df: np.ndarray[float, 2], se: np.ndarray[float, 2]):
        self.statistic = statistic
        self.pvalue = pvalue
        self._nt = nt
        self._df = df
        self._se = se

    def confidence_interval(self, confidence_level: float = .95):
        if not (0 < confidence_level < 1):
            raise ValueError("Confidence level must be between 0 and 1.")
        n = self._nt
        low = np.empty((n, n))
        high = np.empty((n, n))
        for i in range(n):
            for j in range(n):
                srd = studentized_range.ppf(confidence_level, float(n), self._df[i, j])
                rad = srd * self._se[i, j]
                low[i, j] = self.statistic[i, j] - rad
                high[i, j] = self.statistic[i, j] + rad
        return ConfidenceIntervalArr(low, high)

def _tmap_f(d):
    if static.len(d) == 0:
        return ()
    else:
        return (np.asarray(d[0]).astype(float).ravel(),) + _tmap_f(d[1:])

def tukey_hsd(*samples, equal_var: bool = True):
    if static.len(samples) < 2:
        raise ValueError("There must be more than 1 treatment.")
    args = _tmap_f(samples)
    ls = [a for a in args]
    return _tukey(ls, equal_var)

def _tukey(args: List[np.ndarray[float, 1]], equal_var: bool):
    k = len(args)
    means = np.empty(k)
    ns = np.empty(k)
    vs = np.empty(k)
    nobs = 0.0
    for i in range(k):
        a = args[i]
        if a.shape[0] <= 1:
            raise ValueError("Input sample size must be greater than one.")
        for v in a:
            if v == INF or v == -INF:
                raise ValueError("Input samples must be finite.")
        means[i] = float(np.mean(a))
        ns[i] = float(a.shape[0])
        vs[i] = float(np.var(a, ddof=1))
        nobs += ns[i]
    se = np.empty((k, k))
    df = np.empty((k, k))
    if equal_var:
        mse = float(np.sum(vs * (ns - 1))) / (nobs - k)
        same = True
        for i in range(k):
            if ns[i] != ns[0]:
                same = False
        for i in range(k):
            for j in range(k):
                nrm = 2 / ns[0] if same else 1 / ns[j] + 1 / ns[i]
                se[i, j] = K.sqrt(nrm * mse / 2)
                df[i, j] = nobs - k
    else:
        sj2 = vs / ns
        for i in range(k):
            for j in range(k):
                se[i, j] = K.sqrt(sj2[i] + sj2[j]) / K.sqrt(2.0)
                df[i, j] = (sj2[i] + sj2[j]) ** 2 / (sj2[i] ** 2 / (ns[i] - 1) + sj2[j] ** 2 / (ns[j] - 1))
    md = np.empty((k, k))
    pv = np.empty((k, k))
    for i in range(k):
        for j in range(k):
            md[i, j] = means[i] - means[j]
            t = abs(md[i, j]) / se[i, j]
            pv[i, j] = 1.0 - _sr_cdf(t, float(k), df[i, j]) if t == t else NAN
    return TukeyHSDResult(md, pv, k, df, se)

# ------------------------------------------------------------------ dunnett

def _dunnett_p(rho: np.ndarray[float, 2], df: float, stat: np.ndarray[float, 1], alt: str, rs):
    k = stat.shape[0]
    out = np.empty(k)
    for i in range(k):
        a = np.empty(k)
        b = np.empty(k)
        if alt == 'two-sided':
            s = abs(stat[i])
            for j in range(k):
                a[j] = -s
                b[j] = s
            out[i] = 1 - _mvt_box(rho, df, a, b, rs)
        elif alt == 'greater':
            s = stat[i]
            for j in range(k):
                a[j] = -INF
                b[j] = s
            out[i] = 1 - _mvt_box(rho, df, a, b, rs)
        else:
            s = stat[i]
            for j in range(k):
                a[j] = s
                b[j] = INF
            out[i] = 1 - _mvt_box(rho, df, a, b, rs)
    return out

class _DunnettObj[R]:
    rho: np.ndarray[float, 2]
    df: float
    alt: str
    rs: R
    alpha: float
    def __call__(self, x: float) -> float:
        # single-element statistic broadcast as in scipy: statistic is (1,)->1 row
        k = self.rho.shape[0]
        st = np.full(k, x)
        sf = _dunnett_p(self.rho, self.df, st[0:1], self.alt, self.rs) if False else _dunnett_p_one(self.rho, self.df, x, self.alt, self.rs)
        return abs(sf - self.alpha) / self.alpha

def _dunnett_p_one(rho: np.ndarray[float, 2], df: float, x: float, alt: str, rs):
    k = rho.shape[0]
    a = np.empty(k)
    b = np.empty(k)
    if alt == 'two-sided':
        s = abs(x)
        for j in range(k):
            a[j] = -s
            b[j] = s
    elif alt == 'greater':
        for j in range(k):
            a[j] = -INF
            b[j] = x
    else:
        for j in range(k):
            a[j] = x
            b[j] = INF
    return 1 - _mvt_box(rho, df, a, b, rs)

class DunnettResult[R]:
    statistic: np.ndarray[float, 1]
    pvalue: np.ndarray[float, 1]
    _alt: str
    _rho: np.ndarray[float, 2]
    _df: float
    _std: float
    _mean_samples: np.ndarray[float, 1]
    _mean_control: float
    _n_samples: np.ndarray[float, 1]
    _n_control: float
    _rs: R

    def __init__(self, statistic: np.ndarray[float, 1], pvalue: np.ndarray[float, 1], alt: str,
                 rho: np.ndarray[float, 2], df: float, std: float, mean_samples: np.ndarray[float, 1],
                 mean_control: float, n_samples: np.ndarray[float, 1], n_control: float, rs: R):
        self.statistic = statistic
        self.pvalue = pvalue
        self._alt = alt
        self._rho = rho
        self._df = df
        self._std = std
        self._mean_samples = mean_samples
        self._mean_control = mean_control
        self._n_samples = n_samples
        self._n_control = n_control
        self._rs = rs

    def confidence_interval(self, confidence_level: float = .95):
        if not (0 < confidence_level < 1):
            raise ValueError("Confidence level must be between 0 and 1.")
        tol = 1e-3
        alpha = 1 - confidence_level
        obj = _DunnettObj(self._rho, self._df, self._alt, self._rs, alpha)
        res = minimize_scalar(obj, method='brent', tol=tol)
        crit = res.x
        k = self.statistic.shape[0]
        low = np.empty(k)
        high = np.empty(k)
        for i in range(k):
            allowance = abs(crit * self._std * K.sqrt(1 / self._n_samples[i] + 1 / self._n_control))
            d = self._mean_samples[i] - self._mean_control
            low[i] = d - allowance
            high[i] = d + allowance
            if self._alt == 'greater':
                high[i] = INF
            elif self._alt == 'less':
                low[i] = -INF
        return ConfidenceIntervalArr(low, high)

def dunnett(*samples, control, alternative: str = "two-sided", rng = None):
    if alternative != 'two-sided' and alternative != 'less' and alternative != 'greater':
        raise ValueError("alternative must be 'less', 'greater' or 'two-sided'")
    rs = check_random_state(rng)
    ctl = np.asarray(control).astype(float).ravel()
    ss = [a for a in _tmap_f(samples)]
    if ctl.shape[0] < 1:
        raise ValueError("Control and samples groups must have at least 1 observation")
    k = len(ss)
    n_samples = np.empty(k)
    n_sample = 0.0
    for i in range(k):
        if ss[i].shape[0] < 1:
            raise ValueError("Control and samples groups must have at least 1 observation")
        n_samples[i] = float(ss[i].shape[0])
        n_sample += n_samples[i]
    n_control = float(ctl.shape[0])
    df = n_sample + n_control - k - 1
    rho = np.empty((k, k))
    for i in range(k):
        for j in range(k):
            a = n_control / n_samples[i] + 1
            b = n_control / n_samples[j] + 1
            rho[i, j] = 1 / K.sqrt(a * b) if i != j else 1.0
    mean_control = float(np.mean(ctl))
    mean_samples = np.empty(k)
    ssum = 0.0
    for i in range(k):
        mean_samples[i] = float(np.mean(ss[i]))
    ssum += float(np.mean((ctl - mean_control) ** 2)) * n_control
    for i in range(k):
        ssum += float(np.mean((ss[i] - mean_samples[i]) ** 2)) * n_samples[i]
    std = K.sqrt(ssum / df)
    stat = np.empty(k)
    for i in range(k):
        stat[i] = (mean_samples[i] - mean_control) / K.sqrt(1 / n_samples[i] + 1 / n_control) / std
    pv = _dunnett_p(rho, df, stat, alternative, rs)
    return DunnettResult(stat, pv, alternative, rho, df, std, mean_samples, mean_control,
                         n_samples, n_control, rs)
