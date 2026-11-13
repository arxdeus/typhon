# Copyright (C) 2026 the Typhon authors
# Ported from scipy 1.18.1 (BSD-3-Clause)

# levy_stable with Nolan's piecewise integration for pdf/cdf (scipy's default
# 'piecewise'/'best' methods) in the S1 (default) and S0 parameterizations.
# The characteristic-function quadrature and FFT pdf methods are not ported.

import numpy as np
import scipy.special._kernels as K
from scipy.integrate import quad
from scipy.optimize import bisect, minimize
from scipy.special import fresnel
from ._distn_infrastructure import rv_continuous, NAN, INF, check_random_state
from ._continuous_distns import _norm_pdf, _norm_cdf

from C import tan(float) -> float
from C import atan(float) -> float
from C import cos(float) -> float
from C import sin(float) -> float
from C import fabs(float) -> float
from C import pow(float, float) -> float
from C import exp(float) -> float

_PI = 3.141592653589793
_PI_2 = 1.5707963267948966
_QUAD_EPS = 1.2e-14
_XTOL_ZETA = 0.005
_ATOL_ONE = 0.005

class _Nolan:
    alpha: float
    zeta: float
    xi: float
    zeta_prefactor: float
    alpha_exp: float
    alpha_xi: float
    zeta_offset: float
    two_beta_div_pi: float
    pi_div_two_beta: float
    x0_div_term: float
    c1: float
    c2: float
    c3: float

    def __init__(self, alpha: float, beta: float, x0: float):
        self.alpha = alpha
        self.zeta = -beta * tan(_PI_2 * alpha)
        self.zeta_prefactor = 0.0
        self.alpha_exp = 0.0
        self.alpha_xi = 0.0
        self.zeta_offset = 0.0
        self.two_beta_div_pi = 0.0
        self.pi_div_two_beta = 0.0
        self.x0_div_term = 0.0
        if alpha != 1.:
            self.xi = atan(-self.zeta) / alpha
            self.zeta_prefactor = pow(pow(self.zeta, 2.) + 1., -1. / (2. * (alpha - 1.)))
            self.alpha_exp = alpha / (alpha - 1.)
            self.alpha_xi = atan(-self.zeta)
            self.zeta_offset = x0 - self.zeta
            if alpha < 1.:
                self.c1 = 0.5 - self.xi / _PI
                self.c3 = 1 / _PI
            else:
                self.c1 = 1.
                self.c3 = -1 / _PI
            self.c2 = alpha / _PI / fabs(alpha - 1.) / (x0 - self.zeta)
        else:
            self.xi = _PI_2
            self.two_beta_div_pi = beta * 2 / _PI
            self.pi_div_two_beta = _PI_2 / beta
            self.x0_div_term = x0 / self.two_beta_div_pi
            self.c1 = 0.
            self.c2 = .5 / fabs(beta)
            self.c3 = 1 / _PI

    def g(self, theta: float) -> float:
        if self.alpha != 1.:
            if theta == -self.xi:
                return 0.0 if self.alpha < 1 else INF
            if theta == _PI_2:
                return INF if self.alpha < 1 else 0.0
            ct = cos(theta)
            return (self.zeta_prefactor
                    * pow(ct / sin(self.alpha_xi + self.alpha * theta) * self.zeta_offset, self.alpha_exp)
                    * cos(self.alpha_xi + (self.alpha - 1) * theta) / ct)
        if theta == -self.xi:
            return 0.0
        if theta == _PI_2:
            return INF
        return ((1 + theta * self.two_beta_div_pi)
                * exp((self.pi_div_two_beta + theta) * tan(theta) - self.x0_div_term) / cos(theta))

class _GMinus:
    n: _Nolan
    h: float
    def __init__(self, n: _Nolan, h: float):
        self.n = n
        self.h = h
    def __call__(self, t: float) -> float:
        return self.n.g(t) - self.h

class _PdfInt:
    n: _Nolan
    def __init__(self, n: _Nolan):
        self.n = n
    def __call__(self, theta: float) -> float:
        g1 = self.n.g(theta)
        if not K.isfinite(g1) or g1 < 0:
            g1 = 0.0
        return g1 * exp(-g1)

class _CdfInt:
    n: _Nolan
    def __init__(self, n: _Nolan):
        self.n = n
    def __call__(self, theta: float) -> float:
        return exp(-self.n.g(theta))

class _CdfInt1:
    n: _Nolan
    def __init__(self, n: _Nolan):
        self.n = n
    def __call__(self, x: np.ndarray[float, 1]) -> float:
        return exp(-self.n.g(x[0]))

def _round_difficult(x0: float, alpha: float, zeta: float):
    a = alpha
    if fabs(a - 1) < _ATOL_ONE:
        a = 1.0
    x = x0
    if fabs(x - zeta) < _XTOL_ZETA * pow(a, 1 / a):
        x = zeta
    return (x, a)

def _pdf_post(x0: float, alpha: float, beta: float) -> float:
    nol = _Nolan(alpha, beta, x0)
    zeta = nol.zeta
    xi = nol.xi
    x = x0
    if fabs(x - zeta) < _XTOL_ZETA * pow(alpha, 1 / alpha):
        x = zeta
    if x == zeta:
        return K.gamma(1 + 1 / alpha) * cos(xi) / _PI / pow(1 + zeta ** 2, 1 / alpha / 2)
    elif x < zeta:
        return _pdf_post(-x, alpha, -beta)
    if fabs(-xi + _PI_2) <= 1e-14 + 1e-14 * _PI_2:
        return 0.0
    peak = bisect(_GMinus(nol, 1.0), -xi, _PI_2, xtol=_QUAD_EPS)
    p1 = bisect(_GMinus(nol, 100.0), -xi, _PI_2)
    p2 = bisect(_GMinus(nol, 10.0), -xi, _PI_2)
    p3 = bisect(_GMinus(nol, 5.0), -xi, _PI_2)
    f = _PdfInt(nol)
    if beta != -1:
        pts = [0.0, peak, p1, p2, p3]
        intg = quad(f, -xi, _PI_2, points=pts, limit=100, epsrel=_QUAD_EPS, epsabs=0.0)[0]
    else:
        intg = quad(f, -xi, _PI_2, limit=100, epsrel=_QUAD_EPS, epsabs=0.0)[0]
    return nol.c2 * intg

def _pdf_z0(x0: float, alpha0: float, beta: float) -> float:
    zeta = -beta * tan(_PI / 2.0 * alpha0)
    r = _round_difficult(x0, alpha0, zeta)
    x = r[0]
    alpha = r[1]
    if alpha == 2.0:
        return _norm_pdf(x / K.sqrt(2.0)) / K.sqrt(2.0)
    elif alpha == 0.5 and beta == 1.0:
        xx = x + 1
        if xx <= 0:
            return 0.0
        return 1 / K.sqrt(2 * _PI * xx) / xx * exp(-1 / (2 * xx))
    elif alpha == 0.5 and beta == 0.0 and x != 0:
        fr = fresnel(1 / K.sqrt(2 * _PI * fabs(x)))
        arg = 1 / (4 * fabs(x))
        return (sin(arg) * (0.5 - float(fr[0])) + cos(arg) * (0.5 - float(fr[1]))) / K.sqrt(2 * _PI * fabs(x) ** 3)
    elif alpha == 1.0 and beta == 0.0:
        return 1 / (1 + x ** 2) / _PI
    return _pdf_post(x, alpha, beta)

def _cdf_post(x0: float, alpha: float, beta: float) -> float:
    nol = _Nolan(alpha, beta, x0)
    zeta = nol.zeta
    xi = nol.xi
    x = x0
    if fabs(x - zeta) < _XTOL_ZETA * pow(alpha, 1 / alpha):
        x = zeta
    if (alpha == 1 and beta < 0) or x < zeta:
        return 1 - _cdf_post(-x, alpha, -beta)
    elif x == zeta:
        return 0.5 - xi / _PI
    if fabs(-xi + _PI_2) <= 1e-14 + 1e-14 * _PI_2:
        return nol.c1
    f = _CdfInt(nol)
    left = -xi
    right = _PI_2
    if alpha > 1:
        if f(-xi) != 0.0:
            res = minimize(_CdfInt1(nol), np.array([-xi]), method='L-BFGS-B', bounds=[(-xi, _PI_2)])
            left = float(res.x[0])
    else:
        if f(_PI_2) != 0.0:
            res = minimize(_CdfInt1(nol), np.array([_PI_2]), method='L-BFGS-B', bounds=[(-xi, _PI_2)])
            right = float(res.x[0])
    intg = quad(f, left, right, points=[left, right], limit=100, epsrel=_QUAD_EPS, epsabs=0.0)[0]
    return nol.c1 + nol.c3 * intg

def _cdf_z0(x0: float, alpha0: float, beta: float) -> float:
    zeta = -beta * tan(_PI / 2.0 * alpha0)
    r = _round_difficult(x0, alpha0, zeta)
    x = r[0]
    alpha = r[1]
    if alpha == 2.0:
        return _norm_cdf(x / K.sqrt(2.0))
    elif alpha == 0.5 and beta == 1.0:
        xx = x + 1
        if xx <= 0:
            return 0.0
        return K.erfc(K.sqrt(0.5 / xx))
    elif alpha == 1.0 and beta == 0.0:
        return 0.5 + atan(x) / _PI
    return _cdf_post(x, alpha, beta)

def _z1_shift(x: float, alpha: float, beta: float) -> float:
    zeta = -beta * tan(_PI * alpha / 2.0)
    return x + zeta if alpha != 1 else x

def _rvs_z1(rs, alpha: float, beta: float, size):
    n = 1
    if size is None:
        n = 1
    elif isinstance(size, int):
        n = size
    else:
        for s in size:
            n *= int(s)
    out = np.empty(n, dtype=float)
    TH = rs.uniform(-_PI / 2.0, _PI / 2.0, size=n)
    W = rs.standard_exponential(size=n)
    for i in range(n):
        th = TH[i]
        w = W[i]
        aTH = alpha * th
        bTH = beta * th
        cosTH = cos(th)
        tanTH = tan(th)
        v = 0.0
        if alpha == 1:
            v = 2 / _PI * ((_PI / 2 + bTH) * tanTH - beta * K.log((_PI / 2 * w * cosTH) / (_PI / 2 + bTH)))
        elif beta == 0:
            v = w / (cosTH / tan(aTH) + sin(th)) * ((cos(aTH) + sin(aTH) * tanTH) / w) ** (1.0 / alpha)
        else:
            val0 = beta * tan(_PI * alpha / 2)
            th0 = atan(val0) / alpha
            val3 = w / (cosTH / tan(alpha * (th0 + th)) + sin(th))
            v = val3 * ((cos(aTH) + sin(aTH) * tanTH - val0 * (sin(aTH) - cos(aTH) * tanTH)) / w) ** (1.0 / alpha)
        out[i] = v
    if size is None:
        return out[0]
    elif isinstance(size, int):
        return out
    else:
        return out.reshape(size)

@tuple
class _LevyS1K:
    @staticmethod
    def nshape() -> Literal[int]: return 2
    @staticmethod
    def check(sh) -> bool: return sh[0] > 0 and sh[0] <= 2 and sh[1] <= 1 and sh[1] >= -1
    @staticmethod
    def rvs(rs, sh, size): return _rvs_z1(rs, sh[0], sh[1], size)
    @staticmethod
    def pdf(x: float, sh) -> float: return _pdf_z0(_z1_shift(x, sh[0], sh[1]), sh[0], sh[1])
    @staticmethod
    def cdf(x: float, sh) -> float: return _cdf_z0(_z1_shift(x, sh[0], sh[1]), sh[0], sh[1])
    @staticmethod
    def stats(sh):
        a = sh[0]
        return (0.0 if a > 1 else NAN, 2.0 if a == 2 else INF, 0.0 if a == 2.0 else NAN,
                0.0 if a == 2.0 else NAN)
    @staticmethod
    def fitstart(data: np.ndarray[float, 1]) -> List[float]: return [1.5, 0.0]

@tuple
class _LevyS0K:
    @staticmethod
    def nshape() -> Literal[int]: return 2
    @staticmethod
    def check(sh) -> bool: return sh[0] > 0 and sh[0] <= 2 and sh[1] <= 1 and sh[1] >= -1
    @staticmethod
    def rvs(rs, sh, size):
        v = _rvs_z1(rs, sh[0], sh[1], size)
        if sh[0] == 1.0:
            return v
        return v - sh[1] * tan(_PI * sh[0] / 2.0)
    @staticmethod
    def pdf(x: float, sh) -> float: return _pdf_z0(x, sh[0], sh[1])
    @staticmethod
    def cdf(x: float, sh) -> float: return _cdf_z0(x, sh[0], sh[1])
    @staticmethod
    def stats(sh):
        a = sh[0]
        return (0.0 if a > 1 else NAN, 2.0 if a == 2 else INF, 0.0 if a == 2.0 else NAN,
                0.0 if a == 2.0 else NAN)
    @staticmethod
    def fitstart(data: np.ndarray[float, 1]) -> List[float]: return [1.5, 0.0]

_levy_s1 = rv_continuous[_LevyS1K]('levy_stable')
_levy_s0 = rv_continuous[_LevyS0K]('levy_stable')

def _loc_s1(alpha: float, beta: float, loc: float, scale: float) -> float:
    # pdf/cdf/rvs location correction of the S1 parameterization at alpha == 1
    if alpha == 1.0:
        return loc + 2 * beta * scale * K.log(scale) / _PI
    return loc

class levy_stable_gen:
    parameterization: str

    def __init__(self):
        self.parameterization = "S1"

    def _s0(self) -> bool:
        if self.parameterization == "S0":
            return True
        if self.parameterization == "S1":
            return False
        raise RuntimeError(f"Parameterization '{self.parameterization}' in supported list: ('S0', 'S1')")

    def pdf(self, x, alpha, beta, loc = 0.0, scale = 1.0):
        if self._s0():
            return _levy_s0.pdf(x, alpha, beta, loc=loc, scale=scale)
        return _levy_s1.pdf(x, alpha, beta, loc=_loc_s1(float(alpha), float(beta), float(loc), float(scale)),
                            scale=scale)

    def cdf(self, x, alpha, beta, loc = 0.0, scale = 1.0):
        if self._s0():
            return _levy_s0.cdf(x, alpha, beta, loc=loc, scale=scale)
        return _levy_s1.cdf(x, alpha, beta, loc=_loc_s1(float(alpha), float(beta), float(loc), float(scale)),
                            scale=scale)

    def logpdf(self, x, alpha, beta, loc = 0.0, scale = 1.0):
        return np.log(self.pdf(x, alpha, beta, loc, scale))

    def sf(self, x, alpha, beta, loc = 0.0, scale = 1.0):
        return 1.0 - self.cdf(x, alpha, beta, loc, scale)

    def ppf(self, q, alpha, beta, loc = 0.0, scale = 1.0):
        if self._s0():
            return _levy_s0.ppf(q, alpha, beta, loc=loc, scale=scale)
        return _levy_s1.ppf(q, alpha, beta, loc=_loc_s1(float(alpha), float(beta), float(loc), float(scale)),
                            scale=scale)

    def stats(self, alpha, beta, loc = 0.0, scale = 1.0, moments: Literal[str] = 'mv'):
        return _levy_s1.stats(alpha, beta, loc=loc, scale=scale, moments=moments)

    def rvs(self, alpha, beta, loc = 0.0, scale = 1.0, size = None, random_state = None):
        a = float(alpha)
        b = float(beta)
        rs = check_random_state(random_state)
        if not (a > 0 and a <= 2 and b <= 1 and b >= -1 and scale >= 0):
            raise ValueError("Domain error in arguments. The `scale` parameter must be positive for all distributions, and many distributions have restrictions on shape parameters.")
        x1 = _rvs_z1(rs, a, b, size) * scale + loc
        if a == 1.0:
            x1 = x1 + 2 * b * scale * K.log(scale) / _PI
        if self._s0():
            if a == 1.0:
                return x1 - (b * 2 * scale * K.log(scale) / _PI)
            return x1 - scale * b * tan(_PI * a / 2.0)
        return x1

levy_stable = levy_stable_gen()
