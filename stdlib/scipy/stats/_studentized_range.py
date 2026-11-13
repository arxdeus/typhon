# Copyright (C) 2026 the Typhon authors
# Ported from scipy 1.18.1 (BSD-3-Clause)

# studentized_range distribution (scipy.stats._continuous_distns +
# _stats.pyx kernels): cdf/pdf by nested adaptive quadrature.

import numpy as np
import scipy.special._kernels as K
from scipy.integrate import quad
from ._distn_infrastructure import rv_continuous, NAN, INF

_LOG2 = 0.6931471805599453

def _phi(z: float) -> float:
    return 0.3989422804014327 * K.exp(-0.5 * z * z)

def _logphi(z: float) -> float:
    return -0.9189385332046727 - 0.5 * z * z

def _Phi(z: float) -> float:
    return 0.5 * K.erfc(-z * 0.7071067811865475)

def _cdf_logconst(k: float, df: float) -> float:
    return K.log(k) + (df / 2) * K.log(df) - (K.gammaln(df / 2) + (df / 2 - 1) * _LOG2)

def _pdf_logconst(k: float, df: float) -> float:
    return K.log(k) + K.log(k - 1) + (df / 2) * K.log(df) - (K.gammaln(df / 2) + (df / 2 - 1) * _LOG2)

class _CdfInner:
    q: float
    k: float
    df: float
    lc: float
    s: float
    def __call__(self, z: float) -> float:
        s = self.s
        lt = self.lc + (self.df - 1) * K.log(s) - (self.df * s * s / 2) + _logphi(z)
        return K.exp(lt) * K.pow(_Phi(z + self.q * s) - _Phi(z), self.k - 1)

class _CdfOuter:
    q: float
    k: float
    df: float
    lc: float
    def __call__(self, s: float) -> float:
        f = _CdfInner(self.q, self.k, self.df, self.lc, s)
        return quad(f, -INF, INF, (), 0, 1e-11, 1e-12)[0]

class _CdfAsym:
    q: float
    k: float
    def __call__(self, z: float) -> float:
        return self.k * _phi(z) * K.pow(_Phi(z + self.q) - _Phi(z), self.k - 1)

class _PdfInner:
    q: float
    k: float
    df: float
    lc: float
    s: float
    def __call__(self, z: float) -> float:
        s = self.s
        lt = self.lc + self.df * K.log(s) - self.df * s * s / 2 + _logphi(z) + _logphi(s * self.q + z)
        return K.exp(lt) * K.pow(_Phi(s * self.q + z) - _Phi(z), self.k - 2)

class _PdfOuter:
    q: float
    k: float
    df: float
    lc: float
    def __call__(self, s: float) -> float:
        f = _PdfInner(self.q, self.k, self.df, self.lc, s)
        return quad(f, -INF, INF, (), 0, 1e-11, 1e-12)[0]

class _PdfAsym:
    q: float
    k: float
    def __call__(self, z: float) -> float:
        return self.k * (self.k - 1) * _phi(z) * _phi(z + self.q) * K.pow(_Phi(z + self.q) - _Phi(z), self.k - 2)

def _sr_cdf(q: float, k: float, df: float) -> float:
    if q != q or k != k or df != df:
        return NAN
    r = 0.0
    if df < 100000:
        r = quad(_CdfOuter(q, k, df, _cdf_logconst(k, df)), 0.0, INF, (), 0, 1e-11, 1e-12)[0]
    else:
        r = quad(_CdfAsym(q, k), -INF, INF, (), 0, 1e-11, 1e-12)[0]
    return min(max(r, 0.0), 1.0)

def _sr_pdf(q: float, k: float, df: float) -> float:
    if q != q or k != k or df != df:
        return NAN
    if df < 100000:
        return quad(_PdfOuter(q, k, df, _pdf_logconst(k, df)), 0.0, INF, (), 0, 1e-11, 1e-12)[0]
    else:
        return quad(_PdfAsym(q, k), -INF, INF, (), 0, 1e-11, 1e-12)[0]

@tuple
class _StudentizedRangeK:
    @staticmethod
    def nshape() -> Literal[int]: return 2
    @staticmethod
    def check(sh) -> bool: return sh[0] > 1 and sh[1] > 0
    @staticmethod
    def a(sh) -> float: return 0.0
    @staticmethod
    def cdf(z: float, sh) -> float: return _sr_cdf(z, sh[0], sh[1])
    @staticmethod
    def pdf(z: float, sh) -> float: return _sr_pdf(z, sh[0], sh[1])

studentized_range = rv_continuous[_StudentizedRangeK]('studentized_range')
