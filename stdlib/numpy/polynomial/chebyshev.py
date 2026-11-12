# Ported from numpy 2.5.4 (BSD-3-Clause)
#
# numpy.polynomial.chebyshev: thin wrappers over the generic routines in polyutils.

import numpy as np
from .polyutils import trimcoef, _add, _sub, _mul, _mulx, _div, _pow, _val, \
    _vander, _line, _fromroots, _der, _int, _fit, _companion, _roots, \
    _gauss, _to_poly, _from_poly, _weight, as_series
from ._polybase import ABCPolyBase

Chebyshev = ABCPolyBase[1]

chebtrim = trimcoef
chebdomain = np.array([-1.0, 1.0])
chebzero = np.array([0.0])
chebone = np.array([1.0])
chebx = _line(1, 0.0, 1.0)

def chebline(off: float, scl: float):
    return _line(1, off, scl)

def chebadd(c1, c2):
    return _add(c1, c2)

def chebsub(c1, c2):
    return _sub(c1, c2)

def chebmul(c1, c2):
    return _mul(1, c1, c2)

def chebmulx(c):
    return _mulx(1, c)

def chebdiv(c1, c2):
    return _div(1, c1, c2)

def chebpow(c, pow: int, maxpower: int = 16):
    return _pow(1, c, pow, maxpower)

def chebval(x, c):
    return _val(1, x, c)

def chebvander(x, deg: int):
    return _vander(1, x, deg)

def chebfromroots(roots):
    return _fromroots(1, roots)

def chebder(c, m: int = 1, scl: float = 1.0):
    return _der(1, c, m, scl)

def chebint(c, m: int = 1, k = None, lbnd: float = 0.0, scl: float = 1.0):
    return _int(1, c, m, k, lbnd, scl)

def chebfit(x, y, deg, rcond = None, full: Literal[bool] = False, w = None):
    return _fit(1, x, y, deg, rcond, full, w)

def chebcompanion(c):
    return _companion(1, c)

def chebroots(c):
    return _roots(1, c)

def chebgauss(deg: int):
    return _gauss(1, deg)

def chebweight(x):
    return _weight(1, x)

def cheb2poly(c):
    return _to_poly(1, c)

def poly2cheb(pol):
    return _from_poly(1, pol)

def chebpts1(npts: int):
    if npts < 1:
        raise ValueError("npts must be >= 1")
    x = 0.5 * np.pi / npts * np.arange(-npts + 1, npts + 1, 2)
    return np.sin(x)

def chebpts2(npts: int):
    if npts < 2:
        raise ValueError("npts must be >= 2")
    x = np.linspace(-np.pi, 0.0, npts)
    return np.cos(x)

def chebinterpolate(func, deg: int, args = ()):
    if deg < 0:
        raise ValueError("expected deg >= 0")
    order = deg + 1
    xcheb = chebpts1(order)
    yfunc = np.asarray(func(xcheb, *args), dtype=float)
    m = _vander(1, xcheb, deg)
    c = np.dot(m.T, yfunc)
    c[0] /= order
    for i in range(1, order):
        c[i] /= 0.5 * order
    return c
