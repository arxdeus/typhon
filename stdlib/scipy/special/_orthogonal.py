# Copyright (C) 2026 the Typhon authors
# Ported from scipy 1.18.1 (BSD-3-Clause)
# Orthogonal polynomial factories (scipy.special._orthogonal): jacobi, legendre, ...

import math
import numpy as np
from . import _kernels as K
from . import _roots as R
from ._orthopoly import orthopoly1d

INF = float('inf')
PI = 3.141592653589793

def _g(x: float) -> float:
    return K.gamma(x)

def _empty() -> np.ndarray[float, 1]:
    return np.empty(0, dtype=float)

def _chk(n: int):
    if n < 0:
        raise ValueError("n must be nonnegative.")

def jacobi(n: int, alpha: float, beta: float, monic: bool = False):
    _chk(n)
    if n == 0:
        return orthopoly1d(_empty(), _empty(), 1.0, 1.0, 1, alpha, beta, (-1.0, 1.0), monic, -1, 0, 0.0, 0.0)
    t = R.roots_jacobi_mu(n, alpha, beta)
    ab1 = alpha + beta + 1.0
    hn = 2**ab1 / (2 * n + ab1) * _g(n + alpha + 1)
    hn *= _g(n + beta + 1.0) / _g(n + 1) / _g(n + ab1)
    kn = _g(2 * n + ab1) / 2.0**n / _g(n + 1) / _g(n + ab1)
    return orthopoly1d(t[0], t[1], hn, kn, 1, alpha, beta, (-1.0, 1.0), monic, 1, n, alpha, beta)

def sh_jacobi(n: int, p: float, q: float, monic: bool = False):
    _chk(n)
    if n == 0:
        return orthopoly1d(_empty(), _empty(), 1.0, 1.0, 2, p, q, (-1.0, 1.0), monic, -1, 0, 0.0, 0.0)
    if (p - q) <= -1 or q <= 0:
        raise ValueError("(p - q) must be greater than -1, and q must be greater than 0.")
    t = R.roots_jacobi_mu(n, p - q, q - 1.0)
    sc = 2.0 ** p
    x = (t[0] + 1) / 2
    w = t[1] / sc
    hn = _g(n + 1) * _g(n + q) * _g(n + p) * _g(n + p - q + 1)
    hn /= (2 * n + p) * (_g(2 * n + p)**2)
    return orthopoly1d(x, w, hn, 1.0, 2, p, q, (0.0, 1.0), monic, 2, n, p, q)

def genlaguerre(n: int, alpha: float, monic: bool = False):
    if alpha <= -1:
        raise ValueError("alpha must be > -1")
    _chk(n)
    hn = _g(n + alpha + 1) / _g(n + 1)
    kn = (-1.0)**n / _g(n + 1)
    if n == 0:
        return orthopoly1d(_empty(), _empty(), hn, kn, 3, alpha, 0.0, (0.0, INF), monic, 3, n, alpha, 0.0)
    t = R.roots_genlaguerre_mu(n, alpha)
    return orthopoly1d(t[0], t[1], hn, kn, 3, alpha, 0.0, (0.0, INF), monic, 3, n, alpha, 0.0)

def laguerre(n: int, monic: bool = False):
    _chk(n)
    kn = (-1.0)**n / _g(n + 1)
    if n == 0:
        return orthopoly1d(_empty(), _empty(), 1.0, kn, 3, 0.0, 0.0, (0.0, INF), monic, 4, n, 0.0, 0.0)
    t = R.roots_genlaguerre_mu(n, 0.0)
    return orthopoly1d(t[0], t[1], 1.0, kn, 3, 0.0, 0.0, (0.0, INF), monic, 4, n, 0.0, 0.0)

def hermite(n: int, monic: bool = False):
    _chk(n)
    hn = 2.0**n * _g(n + 1) * math.sqrt(PI)
    kn = 2.0**n
    if n == 0:
        return orthopoly1d(_empty(), _empty(), hn, kn, 5, 0.0, 0.0, (-INF, INF), monic, 5, n, 0.0, 0.0)
    t = R.roots_hermite_mu(n)
    return orthopoly1d(t[0], t[1], hn, kn, 5, 0.0, 0.0, (-INF, INF), monic, 5, n, 0.0, 0.0)

def hermitenorm(n: int, monic: bool = False):
    _chk(n)
    hn = math.sqrt(2 * PI) * _g(n + 1)
    if n == 0:
        return orthopoly1d(_empty(), _empty(), hn, 1.0, 6, 0.0, 0.0, (-INF, INF), monic, 6, n, 0.0, 0.0)
    t = R.roots_hermitenorm_mu(n)
    return orthopoly1d(t[0], t[1], hn, 1.0, 6, 0.0, 0.0, (-INF, INF), monic, 6, n, 0.0, 0.0)

def gegenbauer(n: int, alpha: float, monic: bool = False):
    if not math.isfinite(alpha) or alpha <= -0.5:
        raise ValueError("`alpha` must be a finite number greater than -1/2")
    base = jacobi(n, alpha - 0.5, alpha - 0.5, monic)
    if monic or n == 0:
        return base
    factor = (_g(2 * alpha + n) * _g(alpha + 0.5) / _g(2 * alpha) / _g(alpha + 0.5 + n))
    base._scale(factor)
    base.ek = 7
    base.ea = alpha
    base.escale = 1.0
    return base

def chebyt(n: int, monic: bool = False):
    _chk(n)
    if n == 0:
        return orthopoly1d(_empty(), _empty(), PI, 1.0, 8, 0.0, 0.0, (-1.0, 1.0), monic, 8, n, 0.0, 0.0)
    t = R.roots_chebyt_mu(n)
    return orthopoly1d(t[0], t[1], PI / 2, 2.0**(n - 1), 8, 0.0, 0.0, (-1.0, 1.0), monic, 8, n, 0.0, 0.0)

def chebyu(n: int, monic: bool = False):
    base = jacobi(n, 0.5, 0.5, monic)
    if monic:
        return base
    factor = math.sqrt(PI) / 2.0 * _g(n + 2) / _g(n + 1.5)
    base._scale(factor)
    return base

def chebyc(n: int, monic: bool = False):
    _chk(n)
    hn = 4 * PI * (2.0 if n == 0 else 1.0)
    if n == 0:
        p = orthopoly1d(_empty(), _empty(), hn, 1.0, 10, 0.0, 0.0, (-2.0, 2.0), monic, 0, n, 0.0, 0.0)
    else:
        t = R.roots_chebyt_mu(n)
        p = orthopoly1d(t[0] * 2.0, t[1] * 2.0, hn, 1.0, 10, 0.0, 0.0, (-2.0, 2.0), monic, 0, n, 0.0, 0.0)
    if not monic:
        p._scale(2.0 / float(p(2.0)))
        p.ek = 10
        p.en = n
        p.escale = 1.0
    return p

def chebys(n: int, monic: bool = False):
    _chk(n)
    if n == 0:
        p = orthopoly1d(_empty(), _empty(), PI, 1.0, 11, 0.0, 0.0, (-2.0, 2.0), monic, 0, n, 0.0, 0.0)
    else:
        t = R.roots_chebyu_mu(n)
        p = orthopoly1d(t[0] * 2.0, t[1] * 2.0, PI, 1.0, 11, 0.0, 0.0, (-2.0, 2.0), monic, 0, n, 0.0, 0.0)
    if not monic:
        p._scale((n + 1.0) / float(p(2.0)))
        p.ek = 11
        p.en = n
        p.escale = 1.0
    return p

def sh_chebyt(n: int, monic: bool = False):
    base = sh_jacobi(n, 0.0, 0.5, monic)
    if monic:
        return base
    base._scale(4.0**n / 2.0 if n > 0 else 1.0)
    return base

def sh_chebyu(n: int, monic: bool = False):
    base = sh_jacobi(n, 2.0, 1.5, monic)
    if monic:
        return base
    base._scale(4.0**n)
    return base

def legendre(n: int, monic: bool = False):
    _chk(n)
    hn = 2.0 / (2 * n + 1)
    kn = _g(2 * n + 1) / _g(n + 1)**2 / 2.0**n
    if n == 0:
        return orthopoly1d(_empty(), _empty(), hn, kn, 14, 0.0, 0.0, (-1.0, 1.0), monic, 14, n, 0.0, 0.0)
    t = R.roots_legendre_mu(n)
    return orthopoly1d(t[0], t[1], hn, kn, 14, 0.0, 0.0, (-1.0, 1.0), monic, 14, n, 0.0, 0.0)

def sh_legendre(n: int, monic: bool = False):
    _chk(n)
    hn = 1.0 / (2 * n + 1.0)
    kn = _g(2 * n + 1) / _g(n + 1)**2
    if n == 0:
        return orthopoly1d(_empty(), _empty(), hn, kn, 14, 0.0, 0.0, (0.0, 1.0), monic, 15, n, 0.0, 0.0)
    t = R.roots_legendre_mu(n)
    return orthopoly1d((t[0] + 1) / 2, t[1] / 2, hn, kn, 14, 0.0, 0.0, (0.0, 1.0), monic, 15, n, 0.0, 0.0)
