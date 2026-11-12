# Ported from numpy 2.5.4 (BSD-3-Clause)
#
# numpy.polynomial.laguerre: thin wrappers over the generic routines in polyutils.

import numpy as np
from .polyutils import trimcoef, _add, _sub, _mul, _mulx, _div, _pow, _val, \
    _vander, _line, _fromroots, _der, _int, _fit, _companion, _roots, \
    _gauss, _to_poly, _from_poly, _weight, as_series
from ._polybase import ABCPolyBase

Laguerre = ABCPolyBase[5]

lagtrim = trimcoef
lagdomain = np.array([0.0, 1.0])
lagzero = np.array([0.0])
lagone = np.array([1.0])
lagx = _line(5, 0.0, 1.0)

def lagline(off: float, scl: float):
    return _line(5, off, scl)

def lagadd(c1, c2):
    return _add(c1, c2)

def lagsub(c1, c2):
    return _sub(c1, c2)

def lagmul(c1, c2):
    return _mul(5, c1, c2)

def lagmulx(c):
    return _mulx(5, c)

def lagdiv(c1, c2):
    return _div(5, c1, c2)

def lagpow(c, pow: int, maxpower: int = 16):
    return _pow(5, c, pow, maxpower)

def lagval(x, c):
    return _val(5, x, c)

def lagvander(x, deg: int):
    return _vander(5, x, deg)

def lagfromroots(roots):
    return _fromroots(5, roots)

def lagder(c, m: int = 1, scl: float = 1.0):
    return _der(5, c, m, scl)

def lagint(c, m: int = 1, k = None, lbnd: float = 0.0, scl: float = 1.0):
    return _int(5, c, m, k, lbnd, scl)

def lagfit(x, y, deg, rcond = None, full: Literal[bool] = False, w = None):
    return _fit(5, x, y, deg, rcond, full, w)

def lagcompanion(c):
    return _companion(5, c)

def lagroots(c):
    return _roots(5, c)

def laggauss(deg: int):
    return _gauss(5, deg)

def lagweight(x):
    return _weight(5, x)

def lag2poly(c):
    return _to_poly(5, c)

def poly2lag(pol):
    return _from_poly(5, pol)
