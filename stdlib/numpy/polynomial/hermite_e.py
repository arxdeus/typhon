# Ported from numpy 2.5.4 (BSD-3-Clause)
#
# numpy.polynomial.hermite_e: thin wrappers over the generic routines in polyutils.

import numpy as np
from .polyutils import trimcoef, _add, _sub, _mul, _mulx, _div, _pow, _val, \
    _vander, _line, _fromroots, _der, _int, _fit, _companion, _roots, \
    _gauss, _to_poly, _from_poly, _weight, as_series
from ._polybase import ABCPolyBase

HermiteE = ABCPolyBase[4]

hermetrim = trimcoef
hermedomain = np.array([-1.0, 1.0])
hermezero = np.array([0.0])
hermeone = np.array([1.0])
hermex = _line(4, 0.0, 1.0)

def hermeline(off: float, scl: float):
    return _line(4, off, scl)

def hermeadd(c1, c2):
    return _add(c1, c2)

def hermesub(c1, c2):
    return _sub(c1, c2)

def hermemul(c1, c2):
    return _mul(4, c1, c2)

def hermemulx(c):
    return _mulx(4, c)

def hermediv(c1, c2):
    return _div(4, c1, c2)

def hermepow(c, pow: int, maxpower: int = 16):
    return _pow(4, c, pow, maxpower)

def hermeval(x, c):
    return _val(4, x, c)

def hermevander(x, deg: int):
    return _vander(4, x, deg)

def hermefromroots(roots):
    return _fromroots(4, roots)

def hermeder(c, m: int = 1, scl: float = 1.0):
    return _der(4, c, m, scl)

def hermeint(c, m: int = 1, k = None, lbnd: float = 0.0, scl: float = 1.0):
    return _int(4, c, m, k, lbnd, scl)

def hermefit(x, y, deg, rcond = None, full: Literal[bool] = False, w = None):
    return _fit(4, x, y, deg, rcond, full, w)

def hermecompanion(c):
    return _companion(4, c)

def hermeroots(c):
    return _roots(4, c)

def hermegauss(deg: int):
    return _gauss(4, deg)

def hermeweight(x):
    return _weight(4, x)

def herme2poly(c):
    return _to_poly(4, c)

def poly2herme(pol):
    return _from_poly(4, pol)
