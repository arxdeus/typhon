# Ported from numpy 2.5.4 (BSD-3-Clause)
#
# numpy.polynomial.legendre: thin wrappers over the generic routines in polyutils.

import numpy as np
from .polyutils import trimcoef, _add, _sub, _mul, _mulx, _div, _pow, _val, \
    _vander, _line, _fromroots, _der, _int, _fit, _companion, _roots, \
    _gauss, _to_poly, _from_poly, _weight, as_series
from ._polybase import ABCPolyBase

Legendre = ABCPolyBase[2]

legtrim = trimcoef
legdomain = np.array([-1.0, 1.0])
legzero = np.array([0.0])
legone = np.array([1.0])
legx = _line(2, 0.0, 1.0)

def legline(off: float, scl: float):
    return _line(2, off, scl)

def legadd(c1, c2):
    return _add(c1, c2)

def legsub(c1, c2):
    return _sub(c1, c2)

def legmul(c1, c2):
    return _mul(2, c1, c2)

def legmulx(c):
    return _mulx(2, c)

def legdiv(c1, c2):
    return _div(2, c1, c2)

def legpow(c, pow: int, maxpower: int = 16):
    return _pow(2, c, pow, maxpower)

def legval(x, c):
    return _val(2, x, c)

def legvander(x, deg: int):
    return _vander(2, x, deg)

def legfromroots(roots):
    return _fromroots(2, roots)

def legder(c, m: int = 1, scl: float = 1.0):
    return _der(2, c, m, scl)

def legint(c, m: int = 1, k = None, lbnd: float = 0.0, scl: float = 1.0):
    return _int(2, c, m, k, lbnd, scl)

def legfit(x, y, deg, rcond = None, full: Literal[bool] = False, w = None):
    return _fit(2, x, y, deg, rcond, full, w)

def legcompanion(c):
    return _companion(2, c)

def legroots(c):
    return _roots(2, c)

def leggauss(deg: int):
    return _gauss(2, deg)

def legweight(x):
    return _weight(2, x)

def leg2poly(c):
    return _to_poly(2, c)

def poly2leg(pol):
    return _from_poly(2, pol)
