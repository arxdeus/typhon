# Ported from numpy 2.5.4 (BSD-3-Clause)
#
# numpy.polynomial.hermite: thin wrappers over the generic routines in polyutils.

import numpy as np
from .polyutils import trimcoef, _add, _sub, _mul, _mulx, _div, _pow, _val, \
    _vander, _line, _fromroots, _der, _int, _fit, _companion, _roots, \
    _gauss, _to_poly, _from_poly, _weight, as_series
from ._polybase import ABCPolyBase

Hermite = ABCPolyBase[3]

hermtrim = trimcoef
hermdomain = np.array([-1.0, 1.0])
hermzero = np.array([0.0])
hermone = np.array([1.0])
hermx = _line(3, 0.0, 1.0)

def hermline(off: float, scl: float):
    return _line(3, off, scl)

def hermadd(c1, c2):
    return _add(c1, c2)

def hermsub(c1, c2):
    return _sub(c1, c2)

def hermmul(c1, c2):
    return _mul(3, c1, c2)

def hermmulx(c):
    return _mulx(3, c)

def hermdiv(c1, c2):
    return _div(3, c1, c2)

def hermpow(c, pow: int, maxpower: int = 16):
    return _pow(3, c, pow, maxpower)

def hermval(x, c):
    return _val(3, x, c)

def hermvander(x, deg: int):
    return _vander(3, x, deg)

def hermfromroots(roots):
    return _fromroots(3, roots)

def hermder(c, m: int = 1, scl: float = 1.0):
    return _der(3, c, m, scl)

def hermint(c, m: int = 1, k = None, lbnd: float = 0.0, scl: float = 1.0):
    return _int(3, c, m, k, lbnd, scl)

def hermfit(x, y, deg, rcond = None, full: Literal[bool] = False, w = None):
    return _fit(3, x, y, deg, rcond, full, w)

def hermcompanion(c):
    return _companion(3, c)

def hermroots(c):
    return _roots(3, c)

def hermgauss(deg: int):
    return _gauss(3, deg)

def hermweight(x):
    return _weight(3, x)

def herm2poly(c):
    return _to_poly(3, c)

def poly2herm(pol):
    return _from_poly(3, pol)
