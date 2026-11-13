# Copyright (C) 2026 the Typhon authors
# Ported from scipy 1.18.1 (BSD-3-Clause)

# barnard_exact and boschloo_exact.  scipy maximizes the p-value over the
# nuisance parameter with shgo(sampling_method="sobol"); here the (smooth,
# one-dimensional) log p-value is minimized with a dense grid scan followed by
# bounded Brent refinement of every local minimum, which finds the same global
# maximum.

import numpy as np
import scipy.special._kernels as K
from scipy.optimize import fminbound
from ._discrete_distns import hypergeom

NAN = K.NAN
INF = K.INF

class BarnardExactResult:
    statistic: float
    pvalue: float
    def __init__(self, statistic: float, pvalue: float):
        self.statistic = statistic
        self.pvalue = pvalue
    def __getitem__(self, i: Literal[int]):
        if i == 0 or i == -2:
            return self.statistic
        elif i == 1 or i == -1:
            return self.pvalue
        else:
            compile_error("index out of range")
    def __iter__(self):
        yield self.statistic
        yield self.pvalue

class BoschlooExactResult(BarnardExactResult):
    def __init__(self, statistic: float, pvalue: float):
        super().__init__(statistic, pvalue)

class _LogP:
    s: np.ndarray[int, 2]
    lc: np.ndarray[float, 2]
    mask: np.ndarray[bool, 2]
    n: int
    def __call__(self, pi: float) -> float:
        t1 = self.s.shape[0]
        t2 = self.s.shape[1]
        lp = K.log(pi) if pi > 0 else -INF
        l1p = K.log(1 - pi) if 1 - pi > 0 else -INF
        mx = -INF
        for i in range(t1):
            for j in range(t2):
                if self.mask[i, j]:
                    sv = self.s[i, j]
                    a = 0.0 if sv == 0 else lp * sv
                    b = 0.0 if sv == self.n else l1p * (self.n - sv)
                    v = self.lc[i, j] + a + b
                    if v > mx:
                        mx = v
        tot = 0.0
        for i in range(t1):
            for j in range(t2):
                if self.mask[i, j]:
                    sv = self.s[i, j]
                    a = 0.0 if sv == 0 else lp * sv
                    b = 0.0 if sv == self.n else l1p * (self.n - sv)
                    tot += K.exp(self.lc[i, j] + a + b - mx)
        lg = mx + K.log(tot) if tot > 0 else -INF
        return -lg

def _log_comb(n: int):
    out = np.empty(n + 1)
    for k in range(n + 1):
        out[k] = K.gammaln(n + 1.0) - K.gammaln(k + 1.0) - K.gammaln(n - k + 1.0)
    return out

def _maximize(f: _LogP, n_pts: int) -> float:
    G = 256
    xs = np.empty(G + 1)
    vs = np.empty(G + 1)
    best = INF
    for g in range(G + 1):
        x = g / G
        xs[g] = x
        vs[g] = f(x)
        if vs[g] < best:
            best = vs[g]
    for g in range(G + 1):
        left = vs[g - 1] if g > 0 else INF
        right = vs[g + 1] if g < G else INF
        if vs[g] <= left and vs[g] <= right:
            lo = xs[g - 1] if g > 0 else 0.0
            hi = xs[g + 1] if g < G else 1.0
            r = fminbound(f, lo, hi, xtol=1e-12)
            fv = f(float(r))
            if fv < best:
                best = fv
    return best

def _check_table(table):
    t = np.asarray(table).astype(int)
    if t.ndim != 2 or t.shape[0] != 2 or t.shape[1] != 2:
        raise ValueError("The input `table` must be of shape (2, 2).")
    for v in t.flat:
        if v < 0:
            raise ValueError("All values in `table` must be nonnegative.")
    return t

def barnard_exact(table, alternative: str = "two-sided", pooled: bool = True, n: int = 32):
    if n <= 0:
        raise ValueError("Number of points `n` must be strictly positive, found " + str(n))
    t = _check_table(table)
    c1 = int(t[0, 0] + t[1, 0])
    c2 = int(t[0, 1] + t[1, 1])
    if c1 == 0 or c2 == 0:
        return BarnardExactResult(NAN, 1.0)
    if alternative != 'two-sided' and alternative != 'less' and alternative != 'greater':
        raise ValueError("`alternative` should be one of {'two-sided', 'less', 'greater'}, found '" + alternative + "'")
    wald = np.empty((c1 + 1, c2 + 1))
    for i in range(c1 + 1):
        for j in range(c2 + 1):
            p1 = i / c1
            p2 = j / c2
            var = 0.0
            if pooled:
                p = (i + j) / (c1 + c2)
                var = p * (1 - p) * (1 / c1 + 1 / c2)
            else:
                var = p1 * (1 - p1) / c1 + p2 * (1 - p2) / c2
            w = (p1 - p2) / K.sqrt(var) if var > 0 else (NAN if p1 != p2 else 0.0)
            if p1 == p2:
                w = 0.0
            wald[i, j] = w
    obs = wald[int(t[0, 0]), int(t[0, 1])]
    mask = np.empty((c1 + 1, c2 + 1), dtype=bool)
    s = np.empty((c1 + 1, c2 + 1), dtype=int)
    l1 = _log_comb(c1)
    l2 = _log_comb(c2)
    lc = np.empty((c1 + 1, c2 + 1))
    for i in range(c1 + 1):
        for j in range(c2 + 1):
            w = wald[i, j]
            if alternative == 'two-sided':
                mask[i, j] = abs(w) >= abs(obs)
            elif alternative == 'less':
                mask[i, j] = w <= obs
            else:
                mask[i, j] = w >= obs
            s[i, j] = i + j
            lc[i, j] = l1[i] + l2[j]
    f = _LogP(s, lc, mask, c1 + c2)
    best = _maximize(f, n)
    return BarnardExactResult(obs, min(max(K.exp(-best), 0.0), 1.0))

def boschloo_exact(table, alternative: str = "two-sided", n: int = 32):
    if n <= 0:
        raise ValueError("Number of points `n` must be strictly positive, found " + str(n))
    t = _check_table(table)
    c1 = int(t[0, 0] + t[1, 0])
    c2 = int(t[0, 1] + t[1, 1])
    if c1 == 0 or c2 == 0:
        return BoschlooExactResult(NAN, NAN)
    if alternative == 'two-sided':
        a = boschloo_exact(t, 'less', n)
        b = boschloo_exact(t, 'greater', n)
        r = a if a.pvalue < b.pvalue else b
        return BoschlooExactResult(r.statistic, min(max(2 * r.pvalue, 0.0), 1.0))
    if alternative != 'less' and alternative != 'greater':
        raise ValueError("`alternative` should be one of ('two-sided', 'less', 'greater'), found '" + alternative + "'")
    total = c1 + c2
    pv = np.empty((c1 + 1, c2 + 1))
    for i in range(c1 + 1):
        for j in range(c2 + 1):
            if alternative == 'less':
                pv[i, j] = hypergeom.cdf(float(i), float(total), float(i + j), float(c1))
            else:
                pv[i, j] = hypergeom.cdf(float(j), float(total), float(i + j), float(c2))
    fs = pv[int(t[0, 0]), int(t[0, 1])]
    mask = np.empty((c1 + 1, c2 + 1), dtype=bool)
    s = np.empty((c1 + 1, c2 + 1), dtype=int)
    l1 = _log_comb(c1)
    l2 = _log_comb(c2)
    lc = np.empty((c1 + 1, c2 + 1))
    for i in range(c1 + 1):
        for j in range(c2 + 1):
            mask[i, j] = pv[i, j] <= fs * (1 + 1e-13)
            s[i, j] = i + j
            lc[i, j] = l1[i] + l2[j]
    f = _LogP(s, lc, mask, c1 + c2)
    best = _maximize(f, n)
    return BoschlooExactResult(fs, min(max(K.exp(-best), 0.0), 1.0))
