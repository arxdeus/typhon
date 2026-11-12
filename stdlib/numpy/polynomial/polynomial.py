# Ported from numpy 2.5.4 (BSD-3-Clause)
#
# numpy.polynomial.polynomial: thin wrappers over the generic routines in polyutils.

import numpy as np
from .polyutils import trimcoef, _add, _sub, _mul, _mulx, _div, _pow, _val, \
    _vander, _line, _fromroots, _der, _int, _fit, _companion, _roots, \
    _gauss, _to_poly, _from_poly, _weight, as_series
from ._polybase import ABCPolyBase

Polynomial = ABCPolyBase[0]

polytrim = trimcoef
polydomain = np.array([-1.0, 1.0])
polyzero = np.array([0.0])
polyone = np.array([1.0])
polyx = _line(0, 0.0, 1.0)

def polyline(off: float, scl: float):
    return _line(0, off, scl)

def polyadd(c1, c2):
    return _add(c1, c2)

def polysub(c1, c2):
    return _sub(c1, c2)

def polymul(c1, c2):
    return _mul(0, c1, c2)

def polymulx(c):
    return _mulx(0, c)

def polydiv(c1, c2):
    return _div(0, c1, c2)

def polypow(c, pow: int, maxpower: int = 16):
    return _pow(0, c, pow, maxpower)

def polyval(x, c):
    return _val(0, x, c)

def polyvander(x, deg: int):
    return _vander(0, x, deg)

def polyfromroots(roots):
    return _fromroots(0, roots)

def polyder(c, m: int = 1, scl: float = 1.0):
    return _der(0, c, m, scl)

def polyint(c, m: int = 1, k = None, lbnd: float = 0.0, scl: float = 1.0):
    return _int(0, c, m, k, lbnd, scl)

def polyfit(x, y, deg, rcond = None, full: Literal[bool] = False, w = None):
    return _fit(0, x, y, deg, rcond, full, w)

def polycompanion(c):
    return _companion(0, c)

def polyroots(c):
    return _roots(0, c)
