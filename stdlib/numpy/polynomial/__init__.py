# Ported from numpy 2.5.4 (BSD-3-Clause)
#
# numpy.polynomial: power, Chebyshev, Legendre, Hermite, HermiteE and
# Laguerre series (module functions and convenience classes).

import polyutils
import polynomial
import chebyshev
import legendre
import hermite
import hermite_e
import laguerre
from ._polybase import ABCPolyBase
from .polynomial import Polynomial
from .chebyshev import Chebyshev
from .legendre import Legendre
from .hermite import Hermite
from .hermite_e import HermiteE
from .laguerre import Laguerre

_default_printstyle = 'ascii'

def set_default_printstyle(style: str):
    global _default_printstyle
    if style != 'ascii' and style != 'unicode':
        raise ValueError("Unsupported format string '" + style + "'. Valid options are 'ascii' or 'unicode'")
    _default_printstyle = style
