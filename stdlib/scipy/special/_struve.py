# Ported from scipy 1.18.1 (BSD-3-Clause)
# Algorithms from Cephes (BSD-3) / xsf (BSD-3) cephes/struve.h:
# struve (H_v) and modstruve (L_v).

from . import _kernels as K
from ._kernels import exp, log, sqrt, fabs
from . import _bessel as B
from . import _dd as D

NAN = float('nan')
INF = float('inf')
PI = 3.14159265358979323846
MAXITER = 10000
SUM_EPS = 1e-16
SUM_TINY = 1e-100
GOOD_EPS = 1e-12
ACC_EPS = 1e-7
ACC_ATOL = 1e-300

def _isfinite(x: float) -> bool:
    return x == x and x != INF and x != -INF

def _asymp_large_z(v: float, z: float, is_h: bool):
    sgn = -1.0 if is_h else 1.0
    m = z / 2
    maxiter = 0
    if m <= 0:
        maxiter = 0
    elif m > MAXITER:
        maxiter = MAXITER
    else:
        maxiter = int(m)
    if maxiter == 0:
        return (NAN, INF)
    if z < v:
        return (NAN, INF)
    term = -sgn / sqrt(PI) * exp(-K.gammaln(v + 0.5) + (v - 1) * log(z / 2)) * K.gammasgn(v + 0.5)
    s = term
    maxterm = 0.0
    for n in range(maxiter):
        term *= sgn * (1 + 2 * n) * (1 + 2 * n - 2 * v) / (z * z)
        s += term
        if fabs(term) > maxterm:
            maxterm = fabs(term)
        if fabs(term) < SUM_EPS * fabs(s) or term == 0 or not _isfinite(s):
            break
    if is_h:
        s += B.yv(v, z)
    else:
        s += B.iv(v, z)
    return (s, fabs(term) + fabs(maxterm) * SUM_EPS)

def _power_series(v: float, z: float, is_h: bool):
    sgn = -1.0 if is_h else 1.0
    tmp = -K.gammaln(v + 1.5) + (v + 1) * log(z / 2)
    scaleexp = 0.0
    if tmp < -600 or tmp > 600:
        scaleexp = tmp / 2
        tmp -= scaleexp
    term = 2 / sqrt(PI) * exp(tmp) * K.gammasgn(v + 1.5)
    s = term
    maxterm = 0.0
    cterm = (term, 0.0)
    csum = (s, 0.0)
    z2 = (sgn * z * z, 0.0)
    c2v = (2 * v, 0.0)
    for n in range(MAXITER):
        cdiv = (float(3 + 2 * n), 0.0)
        ctmp = D.add(cdiv, c2v)
        cdiv = D.mul(cdiv, ctmp)
        cterm = D.mul(cterm, z2)
        cterm = D.div(cterm, cdiv)
        csum = D.add(csum, cterm)
        term = cterm[0]
        s = csum[0]
        if fabs(term) > maxterm:
            maxterm = fabs(term)
        if fabs(term) < SUM_TINY * fabs(s) or term == 0 or not _isfinite(s):
            break
    err = fabs(term) + fabs(maxterm) * 1e-22
    if scaleexp != 0:
        s *= exp(scaleexp)
        err *= exp(scaleexp)
    if s == 0 and term == 0 and v < 0 and not is_h:
        return (NAN, INF)
    return (s, err)

def _bessel_series(v: float, z: float, is_h: bool):
    if is_h and v < 0:
        return (NAN, INF)
    s = 0.0
    maxterm = 0.0
    cterm = sqrt(z / (2 * PI))
    term = 0.0
    for n in range(MAXITER):
        if is_h:
            term = cterm * B.jv(n + v + 0.5, z) / (n + 0.5)
            cterm *= z / 2 / (n + 1)
        else:
            term = cterm * B.iv(n + v + 0.5, z) / (n + 0.5)
            cterm *= -z / 2 / (n + 1)
        s += term
        if fabs(term) > maxterm:
            maxterm = fabs(term)
        if fabs(term) < SUM_EPS * fabs(s) or term == 0 or not _isfinite(s):
            break
    err = fabs(term) + fabs(maxterm) * 1e-16
    err += 1e-300 * fabs(cterm)
    return (s, err)

def struve_hl(v: float, z: float, is_h: bool) -> float:
    if v != v or z != z:
        return NAN
    if z < 0:
        n = int(v)
        if v == float(n):
            tmp = -1.0 if n % 2 == 0 else 1.0
            return tmp * struve_hl(v, -z, is_h)
        return NAN
    elif z == 0:
        if v < -1:
            return K.gammasgn(v + 1.5) * INF
        elif v == -1:
            return 2 / sqrt(PI) * B.rgamma(0.5)
        return 0.0
    nf = -v - 0.5
    n = int(nf)
    if float(n) == nf and n > 0:
        if is_h:
            return (1.0 if n % 2 == 0 else -1.0) * B.jv(n + 0.5, z)
        return B.iv(n + 0.5, z)
    v0 = NAN
    e0 = INF
    if z >= 0.7 * v + 12:
        r = _asymp_large_z(v, z, is_h)
        v0 = r[0]
        e0 = r[1]
        if e0 < GOOD_EPS * fabs(v0):
            return v0
    r1 = _power_series(v, z, is_h)
    v1 = r1[0]
    e1 = r1[1]
    if e1 < GOOD_EPS * fabs(v1):
        return v1
    v2 = NAN
    e2 = INF
    if fabs(z) < fabs(v) + 20:
        r2 = _bessel_series(v, z, is_h)
        v2 = r2[0]
        e2 = r2[1]
        if e2 < GOOD_EPS * fabs(v2):
            return v2
    bv = v0
    be = e0
    if e1 < be:
        bv = v1
        be = e1
    if e2 < be:
        bv = v2
        be = e2
    if be < ACC_EPS * fabs(bv) or be < ACC_ATOL:
        return bv
    tmp = -K.gammaln(v + 1.5) + (v + 1) * log(z / 2)
    if not is_h:
        tmp = fabs(tmp)
    if tmp > 700:
        return INF * K.gammasgn(v + 1.5)
    return NAN

def struve(v: float, z: float) -> float:
    return struve_hl(v, z, True)

def modstruve(v: float, z: float) -> float:
    return struve_hl(v, z, False)
