# Copyright (C) 2026 the Typhon authors
# Ported from scipy 1.18.1 (BSD-3-Clause)

# Continuous distribution kernels, part 4 (scipy.stats._continuous_distns):
# skewcauchy, irwinhall, gausshyper, ksone, norminvgauss.
# See _distn_infrastructure for the kernel protocol.

import numpy as np
import scipy.special._kernels as K
import scipy.special._hyp as H
import scipy.special._bessel as B
import scipy.special._kolmogorov as KS
from scipy.integrate import quad
from scipy.optimize import brentq
from ._distn_infrastructure import rv_continuous, NAN, INF, _NA, _map_ppf
from ._continuous_distns import _InvgaussK, _PI

from C import atan(float) -> float
from C import tan(float) -> float
from C import fabs(float) -> float
from C import hypot(float, float) -> float

# -------------------------------------------------------------- skewcauchy

@tuple
class _SkewcauchyK:
    @staticmethod
    def nshape() -> Literal[int]: return 1
    @staticmethod
    def check(sh) -> bool: return fabs(sh[0]) < 1
    @staticmethod
    def pdf(x: float, sh) -> float:
        a = sh[0]
        sgn = 1.0 if x > 0 else (-1.0 if x < 0 else 0.0)
        return 1 / (_PI * (x ** 2 / (a * sgn + 1) ** 2 + 1))
    @staticmethod
    def cdf(x: float, sh) -> float:
        a = sh[0]
        if x <= 0:
            return (1 - a) / 2 + (1 - a) / _PI * atan(x / (1 - a))
        return (1 - a) / 2 + (1 + a) / _PI * atan(x / (1 + a))
    @staticmethod
    def ppf(x: float, sh) -> float:
        a = sh[0]
        if x < _SkewcauchyK.cdf(0.0, sh):
            return tan(_PI / (1 - a) * (x - (1 - a) / 2)) * (1 - a)
        return tan(_PI / (1 + a) * (x - (1 - a) / 2)) * (1 + a)
    @staticmethod
    def stats(sh): return (NAN, NAN, NAN, NAN)

# --------------------------------------------------------------- irwinhall

def _binom_int(n: int, k: int) -> int:
    r = 1
    for i in range(k):
        r = r * (n - i) // (i + 1)
    return r

def _stirling2(m: int, n: int) -> int:
    # S(m, n) = 1/n! sum_j (-1)^j C(n, j) (n - j)^m (exact, small arguments)
    s = 0
    for j in range(n + 1):
        t = _binom_int(n, j) * (n - j) ** m
        s += -t if j % 2 else t
    f = 1
    for i in range(2, n + 1):
        f *= i
    return s // f

@tuple
class _IrwinhallK:
    @staticmethod
    def nshape() -> Literal[int]: return 1
    @staticmethod
    def check(sh) -> bool: return sh[0] > 0 and sh[0] == K.floor(sh[0])
    @staticmethod
    def a(sh) -> float: return 0.0
    @staticmethod
    def b(sh) -> float: return sh[0]
    @staticmethod
    def rvs(rs, sh, size):
        n = int(sh[0])
        if size is None:
            return rs.uniform(size=n).sum(axis=0)
        else:
            sz = (size,) if isinstance(size, int) else size
            return rs.uniform(size=(n,) + sz).sum(axis=0)
    @staticmethod
    def pdf(x: float, sh) -> float:
        # cardinal B-spline of degree n-1 on knots 0..n
        n = int(sh[0])
        if x < 0 or x > n:
            return 0.0
        s = 0.0
        fact = 1.0
        for i in range(2, n):
            fact *= i
        for k in range(int(K.floor(x)) + 1):
            t = _binom_int(n, k) * K.pow(x - k, n - 1.0)
            s += -t if k % 2 else t
        return s / fact
    @staticmethod
    def cdf(x: float, sh) -> float:
        n = int(sh[0])
        if x <= 0:
            return 0.0
        if x >= n:
            return 1.0
        s = 0.0
        fact = 1.0
        for i in range(2, n + 1):
            fact *= i
        for k in range(int(K.floor(x)) + 1):
            t = _binom_int(n, k) * K.pow(x - k, float(n))
            s += -t if k % 2 else t
        return s / fact
    @staticmethod
    def sf(x: float, sh) -> float: return _IrwinhallK.cdf(sh[0] - x, sh)
    @staticmethod
    def munp(order: float, sh) -> float:
        n = int(sh[0])
        o = int(order)
        return float(_stirling2(n + o, n)) / float(_binom_int(n + o, n))
    @staticmethod
    def stats(sh):
        n = sh[0]
        return (n / 2, n / 12, 0.0, -6 / (5 * n))

# -------------------------------------------------------------- gausshyper

@tuple
class _GausshyperK:
    @staticmethod
    def nshape() -> Literal[int]: return 4
    @staticmethod
    def check(sh) -> bool: return sh[0] > 0 and sh[1] > 0 and sh[2] == sh[2] and sh[3] > -1
    @staticmethod
    def a(sh) -> float: return 0.0
    @staticmethod
    def b(sh) -> float: return 1.0
    @staticmethod
    def pdf(x: float, sh) -> float:
        a = sh[0]
        b = sh[1]
        c = sh[2]
        z = sh[3]
        nc = K.beta(a, b) * H.hyp2f1(c, a, a + b, -z)
        return (1. / nc * K.pow(x, a - 1.) * K.pow(1. - x, b - 1.0) / K.pow(1.0 + z * x, c))
    @staticmethod
    def munp(n: float, sh) -> float:
        a = sh[0]
        b = sh[1]
        c = sh[2]
        z = sh[3]
        fac = K.beta(n + a, b) / K.beta(a, b)
        return fac * H.hyp2f1(c, a + n, a + b + n, -z) / H.hyp2f1(c, a, a + b, -z)

# ------------------------------------------------------------------- ksone

@tuple
class _KsoneK:
    @staticmethod
    def nshape() -> Literal[int]: return 1
    @staticmethod
    def check(sh) -> bool: return sh[0] >= 1 and sh[0] == K.floor(sh[0] + 0.5)
    @staticmethod
    def a(sh) -> float: return 0.0
    @staticmethod
    def b(sh) -> float: return 1.0
    @staticmethod
    def pdf(x: float, sh) -> float: return -KS.smirnovp(sh[0], x)
    @staticmethod
    def cdf(x: float, sh) -> float: return KS.smirnovc(sh[0], x)
    @staticmethod
    def sf(x: float, sh) -> float: return KS.smirnov(sh[0], x)
    @staticmethod
    def ppf(q: float, sh) -> float: return KS.smirnovci(sh[0], q)
    @staticmethod
    def isf(q: float, sh) -> float: return KS.smirnovi(sh[0], q)

# ------------------------------------------------------------ norminvgauss

def _nig_pdf(x: float, a: float, b: float) -> float:
    gamma = K.sqrt((a + b) * (a - b))
    fac1 = a / _PI
    sq = hypot(1.0, x)
    return fac1 * B.k1e(a * sq) * K.exp(b * x - a * sq + gamma) / sq

class _NigPdf:
    a: float
    b: float
    def __init__(self, a: float, b: float):
        self.a = a
        self.b = b
    def __call__(self, x: float) -> float:
        return _nig_pdf(x, self.a, self.b)

def _nig_quad(x: float, y: float, a: float, b: float) -> float:
    mean = b / K.sqrt((a + b) * (a - b))
    f = _NigPdf(a, b)
    if x == -INF and y > fabs(mean):
        return 1 - quad(f, y, INF)[0]
    if y == INF and x < -fabs(mean):
        return 1 - quad(f, -INF, x)[0]
    res = quad(f, x, y)[0]
    return min(max(res, 0.0), 1.0)

class _NigSfEq:
    a: float
    b: float
    q: float
    def __init__(self, a: float, b: float, q: float):
        self.a = a
        self.b = b
        self.q = q
    def __call__(self, x: float) -> float:
        return _nig_quad(x, INF, self.a, self.b) - self.q

@tuple
class _NorminvgaussK:
    @staticmethod
    def nshape() -> Literal[int]: return 2
    @staticmethod
    def check(sh) -> bool: return sh[0] > 0 and fabs(sh[1]) < sh[0]
    @staticmethod
    def open_support() -> Literal[bool]: return True
    @staticmethod
    def rvs(rs, sh, size):
        a = sh[0]
        b = sh[1]
        gamma = K.sqrt((a + b) * (a - b))
        ig = rs.wald(1 / gamma, 1.0, size=size)
        return b * ig + np.sqrt(ig) * rs.standard_normal(size)
    @staticmethod
    def pdf(x: float, sh) -> float: return _nig_pdf(x, sh[0], sh[1])
    @staticmethod
    def cdf(x: float, sh) -> float: return _nig_quad(-INF, x, sh[0], sh[1])
    @staticmethod
    def sf(x: float, sh) -> float: return _nig_quad(x, INF, sh[0], sh[1])
    @staticmethod
    def isf(q: float, sh) -> float:
        a = sh[0]
        b = sh[1]
        eq = _NigSfEq(a, b, q)
        xm = b / K.sqrt((a + b) * (a - b))
        em = eq(xm)
        if em == 0:
            return xm
        left = xm
        right = xm
        if em > 0:
            delta = 1.0
            right = xm + delta
            while eq(right) > 0:
                delta = 2 * delta
                right = xm + delta
        else:
            delta = 1.0
            left = xm - delta
            while eq(left) < 0:
                delta = 2 * delta
                left = xm - delta
        return brentq(eq, left, right, xtol=1e-14)
    @staticmethod
    def stats(sh):
        a = sh[0]
        b = sh[1]
        gamma = K.sqrt((a + b) * (a - b))
        return (b / gamma, a ** 2 / gamma ** 3, 3.0 * b / (a * K.sqrt(gamma)),
                3.0 * (1 + 4 * b ** 2 / a ** 2) / gamma)
    @staticmethod
    def fitstart(data: np.ndarray[float, 1]) -> List[float]: return [1.0, 0.5]

skewcauchy = rv_continuous[_SkewcauchyK]('skewcauchy')
irwinhall = rv_continuous[_IrwinhallK]('irwinhall')
gausshyper = rv_continuous[_GausshyperK]('gausshyper')
ksone = rv_continuous[_KsoneK]('ksone')
norminvgauss = rv_continuous[_NorminvgaussK]('norminvgauss')
