# Ported from scipy 1.18.1 (BSD-3-Clause)
# Inverses of the noncentral distributions with respect to df / nc
# (cdflib cdfchn/cdft/cdff which=3,4), solved by xsf-style bracketing and
# Chandrupatla instead of cdflib's dzror.

from . import _noncentral as NC
from . import _tools as T

NAN = float('nan')
INF = float('inf')

class _NcRoot:
    kind: int
    a: float
    b: float
    c: float
    target: float
    def __init__(self, kind: int, a: float, b: float, c: float, target: float):
        self.kind = kind
        self.a = a
        self.b = b
        self.c = c
        self.target = target
    def __call__(self, t: float) -> float:
        k = self.kind
        if k == 0:    # chndtridf: chndtr(x=a, df=t, nc=b)
            return NC.chndtr(self.a, t, self.b) - self.target
        elif k == 1:  # chndtrinc: chndtr(x=a, df=b, nc=t)
            return NC.chndtr(self.a, self.b, t) - self.target
        elif k == 2:  # nctdtridf: nctdtr(df=t, nc=a, t=b)
            return NC.nctdtr(t, self.a, self.b) - self.target
        elif k == 3:  # nctdtrinc: nctdtr(df=a, nc=t, t=b)
            return NC.nctdtr(self.a, t, self.b) - self.target
        elif k == 4:  # ncfdtridfd: ncfdtr(dfn=a, dfd=t, nc=b, f=c)
            return NC.ncfdtr(self.a, t, self.b, self.c) - self.target
        elif k == 5:  # ncfdtridfn: ncfdtr(dfn=t, dfd=a, nc=b, f=c)
            return NC.ncfdtr(t, self.a, self.b, self.c) - self.target
        else:         # ncfdtrinc: ncfdtr(dfn=a, dfd=b, nc=t, f=c)
            return NC.ncfdtr(self.a, self.b, t, self.c) - self.target

def _isnan(x: float) -> bool:
    return x != x

def _solve(f: _NcRoot, lo: float, hi: float) -> float:
    g1 = f(1.0)
    g2 = f(2.0)
    g3 = f(0.5)
    if _isnan(g1) or _isnan(g2) or _isnan(g3):
        return NAN
    inc = g2 > g3
    if g2 == g3:
        inc = g1 > f(4.0)
        inc = not inc
    r = T.bracket_root(f, 1.0, lo, hi, -0.875, 7.0, 0.125, 8.0, inc, 342)
    st = r[4]
    if st == 1:
        return lo
    if st == 2:
        return hi
    if st >= 3:
        return NAN
    res = T.chandrupatla(f, r[0], r[1], r[2], r[3], T.EPS, 1e-100, 200)
    if res[1] != 0:
        return NAN
    return res[0]

def chndtridf(x: float, p: float, nc: float) -> float:
    if _isnan(x) or _isnan(p) or _isnan(nc):
        return NAN
    if x < 0 or p <= 0 or p >= 1 or nc < 0:
        return NAN
    return _solve(_NcRoot(0, x, nc, 0.0, p), 1e-300, 1e300)

def chndtrinc(x: float, df: float, p: float) -> float:
    if _isnan(x) or _isnan(df) or _isnan(p):
        return NAN
    if x < 0 or df <= 0 or p <= 0 or p >= 1:
        return NAN
    return _solve(_NcRoot(1, x, df, 0.0, p), 0.0, 1e4)

def nctdtridf(p: float, nc: float, t: float) -> float:
    if _isnan(p) or _isnan(nc) or _isnan(t):
        return NAN
    if p <= 0 or p >= 1:
        return NAN
    return _solve(_NcRoot(2, nc, t, 0.0, p), 1e-300, 1e10)

def nctdtrinc(df: float, p: float, t: float) -> float:
    if _isnan(p) or _isnan(df) or _isnan(t):
        return NAN
    if df <= 0 or p <= 0 or p >= 1:
        return NAN
    return _solve(_NcRoot(3, df, t, 0.0, p), -1e6, 1e6)

def ncfdtridfd(dfn: float, p: float, nc: float, f: float) -> float:
    if _isnan(dfn) or _isnan(p) or _isnan(nc) or _isnan(f):
        return NAN
    if dfn <= 0 or p <= 0 or p >= 1 or nc < 0 or f < 0:
        return NAN
    return _solve(_NcRoot(4, dfn, nc, f, p), 1e-300, 1e10)

def ncfdtridfn(p: float, dfd: float, nc: float, f: float) -> float:
    if _isnan(dfd) or _isnan(p) or _isnan(nc) or _isnan(f):
        return NAN
    if dfd <= 0 or p <= 0 or p >= 1 or nc < 0 or f < 0:
        return NAN
    return _solve(_NcRoot(5, dfd, nc, f, p), 1e-300, 1e10)

def ncfdtrinc(dfn: float, dfd: float, p: float, f: float) -> float:
    if _isnan(dfn) or _isnan(dfd) or _isnan(p) or _isnan(f):
        return NAN
    if dfn <= 0 or dfd <= 0 or p <= 0 or p >= 1 or f < 0:
        return NAN
    return _solve(_NcRoot(6, dfn, dfd, f, p), 0.0, 1e4)
