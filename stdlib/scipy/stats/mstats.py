# Copyright (C) 2026 the Typhon authors
# Ported from scipy 1.18.1 (BSD-3-Clause)

# scipy.stats.mstats basics for plain (unmasked) arrays: gmean, hmean,
# mquantiles, winsorize, trimmed_mean/var/std, trima/trimr helpers.  Inputs are
# ndarray-likes; results are plain ndarrays (no mask).

import numpy as np
import internal.static as static
import scipy.special._kernels as K
from C import rint(float) -> float
from ._common import _f64, _slices, _reduce
from ._descriptive import gmean as _gmean, hmean as _hmean

NAN = K.NAN

def gmean(a, axis = 0, dtype = None, weights = None):
    return _gmean(a, axis=axis, weights=weights)

def hmean(a, axis = 0, dtype = None, weights = None):
    return _hmean(a, axis=axis, weights=weights)

def _chk_lim(l, name: str):
    if l < 0.0 or l > 1.0:
        raise ValueError("The proportion to cut from the " + name + " should be between 0. and 1.")

# ------------------------------------------------------------------ mquantiles

class _MQ:
    p: np.ndarray[float, 1]
    m: np.ndarray[float, 1]
    def __call__(self, data):
        x = np.sort(data)
        n = x.shape[0]
        k = self.p.shape[0]
        out = np.empty(k, dtype=float)
        if n == 0:
            for i in range(k):
                out[i] = NAN
            return out
        if n == 1:
            for i in range(k):
                out[i] = x[0]
            return out
        for i in range(k):
            aleph = n * self.p[i] + self.m[i]
            kk = K.floor(min(max(aleph, 1.0), float(n - 1)))
            gamma = min(max(aleph - kk, 0.0), 1.0)
            ki = int(kk)
            out[i] = (1.0 - gamma) * x[ki - 1] + gamma * x[ki]
        return out

def mquantiles(a, prob = (.25, .5, .75), alphap: float = .4, betap: float = .4, axis = None, limit = ()):
    p = np.atleast_1d(np.asarray(prob).astype(float))
    m = alphap + p * (1. - alphap - betap)
    f = _MQ(p, m)
    x0 = _f64(a)
    if static.len(limit) == 2:
        lo = float(limit[0])
        hi = float(limit[1])
        x0 = np.where((x0 > lo) & (x0 < hi), x0, NAN)
    if axis is None:
        return f(_nonan(x0.ravel()))
    mv, shp = _slices(x0, axis)
    if static.len(shp) == 0:
        return f(_nonan(np.ascontiguousarray(mv)))
    out = np.empty(shp + (p.shape[0],), dtype=float)
    for idx in np.ndindex(shp):
        r = f(_nonan(np.ascontiguousarray(mv[idx])))
        for i in range(p.shape[0]):
            out[idx + (i,)] = r[i]
    ax = int(axis)
    if ax < 0:
        ax += x0.ndim
    return np.moveaxis(out, -1, ax)

def _nonan(x):
    n = x.shape[0]
    k = 0
    for i in range(n):
        if x[i] == x[i]:
            k += 1
    if k == n:
        return x
    o = np.empty(k, dtype=float)
    j = 0
    for i in range(n):
        if x[i] == x[i]:
            o[j] = x[i]
            j += 1
    return o

# ------------------------------------------------------------------ winsorize

def _lim_pair(limits):
    if isinstance(limits, float):
        return (limits, limits)
    else:
        return limits

class _Wins:
    lo: float
    hi: float
    has_lo: bool
    has_hi: bool
    loinc: bool
    upinc: bool
    def __call__(self, a0):
        a = a0.copy()
        n = a.shape[0]
        idx = np.argsort(a, kind='mergesort')
        if self.has_lo and self.lo != 0.0:
            lowidx = 0
            if self.loinc:
                lowidx = int(self.lo * n)
            else:
                lowidx = int(rint(self.lo * n))
            v = a[idx[lowidx]]
            for i in range(lowidx):
                a[idx[i]] = v
        if self.has_hi:
            upidx = 0
            if self.upinc:
                upidx = n - int(n * self.hi)
            else:
                upidx = n - int(rint(n * self.hi))
            v = a[idx[upidx - 1]]
            for i in range(upidx, n):
                a[idx[i]] = v
        return a

def winsorize(a, limits = None, inclusive = (True, True), inplace: bool = False, axis = None,
              nan_policy: str = 'propagate'):
    x = _f64(a).copy()
    if isinstance(limits, NoneType):
        return x
    lp = _lim_pair(limits)
    lo = lp[0]
    hi = lp[1]
    if not isinstance(lo, NoneType):
        _chk_lim(float(lo), "beginning")
    if not isinstance(hi, NoneType):
        _chk_lim(float(hi), "end")
    w = _Wins(_flt(lo), _flt(hi), not isinstance(lo, NoneType), not isinstance(hi, NoneType),
              bool(inclusive[0]), bool(inclusive[1]))
    if axis is None:
        shp = x.shape
        return np.reshape(w(x.ravel()), shp)
    mv, shp = _slices(x, axis)
    out = np.empty(mv.shape, dtype=float)
    for idx in np.ndindex(shp):
        out[idx] = w(np.ascontiguousarray(mv[idx]))
    ax = int(axis)
    if ax < 0:
        ax += x.ndim
    return np.moveaxis(out, -1, ax)

def _flt(v) -> float:
    if isinstance(v, NoneType):
        return 0.0
    else:
        return float(v)

# ------------------------------------------------------------------ trimming

class _TrimR:
    lo: float
    hi: float
    has_lo: bool
    has_hi: bool
    loinc: bool
    upinc: bool
    def __call__(self, x):
        # returns the kept values in original order
        n = x.shape[0]
        idx = np.argsort(x, kind='mergesort')
        keep = np.ones(n, dtype=bool)
        if self.has_lo and self.lo != 0.0:
            lowidx = int(self.lo * n) if self.loinc else int(rint(self.lo * n))
            for i in range(lowidx):
                keep[idx[i]] = False
        if self.has_hi:
            upidx = n - int(n * self.hi) if self.upinc else n - int(rint(n * self.hi))
            for i in range(upidx, n):
                keep[idx[i]] = False
        return x[keep]

class _TrimA:
    lo: float
    hi: float
    has_lo: bool
    has_hi: bool
    loinc: bool
    upinc: bool
    def __call__(self, x):
        n = x.shape[0]
        k = 0
        keep = np.ones(n, dtype=bool)
        for i in range(n):
            v = x[i]
            if self.has_lo:
                if self.loinc:
                    if v < self.lo:
                        keep[i] = False
                else:
                    if v <= self.lo:
                        keep[i] = False
            if self.has_hi:
                if self.upinc:
                    if v > self.hi:
                        keep[i] = False
                else:
                    if v >= self.hi:
                        keep[i] = False
        return x[keep]

class _TrimStat:
    kind: str
    ddof: int
    relative: bool
    tr: _TrimR
    ta: _TrimA
    def __call__(self, x):
        y = self.tr(x) if self.relative else self.ta(x)
        n = y.shape[0]
        if n == 0:
            return NAN
        if self.kind == 'mean':
            return float(np.sum(y)) / n
        elif self.kind == 'var':
            return float(np.var(y, ddof=self.ddof))
        else:
            return float(np.std(y, ddof=self.ddof))

def _trim_stat(a, limits, inclusive, relative: bool, axis, kind: str, ddof: int):
    lp = _lim_pair(limits)
    lo = lp[0]
    hi = lp[1]
    hl = not isinstance(lo, NoneType)
    hh = not isinstance(hi, NoneType)
    if relative:
        if hl:
            _chk_lim(float(lo), "beginning")
        if hh:
            _chk_lim(float(hi), "end")
    tr = _TrimR(_flt(lo), _flt(hi), hl, hh, bool(inclusive[0]), bool(inclusive[1]))
    ta = _TrimA(_flt(lo), _flt(hi), hl, hh, bool(inclusive[0]), bool(inclusive[1]))
    return _reduce(a, axis, _TrimStat(kind, ddof, relative, tr, ta))

def trimmed_mean(a, limits = (0.1, 0.1), inclusive = (1, 1), relative: bool = True, axis = None):
    return _trim_stat(a, limits, inclusive, relative, axis, 'mean', 0)

def trimmed_var(a, limits = (0.1, 0.1), inclusive = (1, 1), relative: bool = True, axis = None, ddof: int = 0):
    return _trim_stat(a, limits, inclusive, relative, axis, 'var', ddof)

def trimmed_std(a, limits = (0.1, 0.1), inclusive = (1, 1), relative: bool = True, axis = None, ddof: int = 0):
    return _trim_stat(a, limits, inclusive, relative, axis, 'std', ddof)
