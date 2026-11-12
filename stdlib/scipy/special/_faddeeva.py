# Ported from scipy 1.18.1 (BSD-3-Clause)
# Algorithms from the Faddeeva package (MIT, Steven G. Johnson) as vendored
# in xsf (BSD-3): w_im (Im w(x) for real x), erfi, dawsn, plus the complex
# Faddeeva w(z) used by wofz / voigt_profile.

from . import _kernels as K
from ._kernels import exp, sqrt, fabs
from . import _faddeeva_const as FC

NAN = float('nan')
INF = float('inf')
ISPI = 0.56418958354775628694807945156

def _w_im_y100(y100: float, x: float) -> float:
    if y100 != y100:
        return NAN
    k = int(y100)
    if k >= 99:
        x2 = x * x
        return x * (1.1283791670955125739 - x2 * (0.75225277806367504925 - x2 * (0.30090111122547001970 - x2 * 0.085971746064420005629)))
    t = 2 * y100 - FC._WIM_OFF[k]
    c = FC._WIM_C[k]
    r = 0.0
    i = len(c) - 1
    while i >= 0:
        r = r * t + c[i]
        i -= 1
    return r

def w_im(x: float) -> float:
    if x >= 0:
        if x > 45:
            if x > 5e7:
                return ISPI / x
            return ISPI * ((x * x) * (x * x - 4.5) + 2) / (x * ((x * x) * (x * x - 5) + 3.75))
        return _w_im_y100(100 / (1 + x), x)
    else:
        if x < -45:
            if x < -5e7:
                return ISPI / x
            return ISPI * ((x * x) * (x * x - 4.5) + 2) / (x * ((x * x) * (x * x - 5) + 3.75))
        return -_w_im_y100(100 / (1 - x), -x)

def erfi(x: float) -> float:
    if x * x > 720:
        return INF if x > 0 else -INF
    return exp(x * x) * w_im(x)

def dawsn(x: float) -> float:
    return 0.8862269254527580136490837416705725913990 * w_im(x)
