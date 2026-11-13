# Ported from scipy 1.18.1 (BSD-3-Clause)
# Algorithms from the Faddeeva package (MIT, Steven G. Johnson) as vendored
# in xsf (BSD-3): w_im (Im w(x) for real x), erfi, dawsn, plus the complex
# Faddeeva w(z) used by wofz / voigt_profile.

from . import _kernels as K
from ._kernels import exp, sqrt, fabs
from . import _faddeeva_const as FC
import cmath
from C import cos(float) -> float
from C import sin(float) -> float

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

def _sinc(x: float, sinx: float) -> float:
    return 1 - 0.1666666666666666666667 * x * x if fabs(x) < 1e-4 else sinx / x

def _sinh_taylor(x: float) -> float:
    return x * (1 + (x * x) * (0.1666666666666666666667 + 0.00833333333333333333333 * (x * x)))

def w(z: complex) -> complex:
    """Faddeeva w(z) = exp(-z^2) erfc(-iz) with relerr = machine epsilon."""
    if z.real == 0.0:
        return complex(K.erfcx(z.imag), z.real)
    elif z.imag == 0:
        return complex(exp(-z.real * z.real), w_im(z.real))
    relerr = 2.220446049250313e-16
    a = 0.518321480430085929872
    c = 0.329973702884629072537
    a2 = 0.268657157075235951582
    x = fabs(z.real)
    y = z.imag
    ya = fabs(y)
    ret = complex(0.0, 0.0)
    sum1 = 0.0
    sum2 = 0.0
    sum3 = 0.0
    sum4 = 0.0
    sum5 = 0.0
    if ya > 7 or (x > 6 and (ya > 0.1 or (x > 8 and ya > 1e-10) or x > 28)):
        xs = -z.real if y < 0 else z.real
        if x + ya > 4000:
            if x + ya > 1e7:
                if x > ya:
                    yax = ya / xs
                    denom = ISPI / (xs + yax * ya)
                    ret = complex(denom * yax, denom)
                elif ya == INF:
                    if x != x or y < 0:
                        return complex(NAN, NAN)
                    return complex(0.0, 0.0)
                else:
                    xya = xs / ya
                    denom = ISPI / (xya * xs + ya)
                    ret = complex(denom, denom * xya)
            else:
                dr = xs * xs - ya * ya - 0.5
                di = 2 * xs * ya
                denom = ISPI / (dr * dr + di * di)
                ret = complex(denom * (xs * di - ya * dr), denom * (xs * dr + ya * di))
        else:
            nu = K.floor(3.9 + 11.398 / (0.08254 * x + 0.1421 * ya + 0.2023))
            wr = xs
            wi = ya
            nu = 0.5 * (nu - 1)
            while nu > 0.4:
                denom = nu / (wr * wr + wi * wi)
                wr = xs - wr * denom
                wi = ya + wi * denom
                nu -= 0.5
            denom = ISPI / (wr * wr + wi * wi)
            ret = complex(denom * wi, denom * wr)
        if y < 0:
            e = cmath.exp(complex((ya - xs) * (xs + ya), 2 * xs * y))
            return 2.0 * e - ret
        return ret
    elif x < 10:
        prod2ax = 1.0
        prodm2ax = 1.0
        expx2 = 0.0
        if y != y:
            return complex(y, y)
        if x < 5e-4:
            x2 = x * x
            expx2 = 1 - x2 * (1 - 0.5 * x2)
            ax2 = 1.036642960860171859744 * x
            exp2ax = 1 + ax2 * (1 + ax2 * (0.5 + 0.166666666666666666667 * ax2))
            expm2ax = 1 - ax2 * (1 - ax2 * (0.5 - 0.166666666666666666667 * ax2))
            n = 1
            while True:
                coef = FC.EXPA2N2[n - 1] * expx2 / (a2 * float(n * n) + y * y)
                prod2ax *= exp2ax
                prodm2ax *= expm2ax
                sum1 += coef
                sum2 += coef * prodm2ax
                sum3 += coef * prod2ax
                sum5 += coef * (2 * a) * n * _sinh_taylor((2 * a) * n * x)
                if coef * prod2ax < relerr * sum3:
                    break
                n += 1
        else:
            expx2 = exp(-x * x)
            exp2ax = exp((2 * a) * x)
            expm2ax = 1 / exp2ax
            n = 1
            while True:
                coef = FC.EXPA2N2[n - 1] * expx2 / (a2 * float(n * n) + y * y)
                prod2ax *= exp2ax
                prodm2ax *= expm2ax
                sum1 += coef
                sum2 += coef * prodm2ax
                sum4 += (coef * prodm2ax) * (a * n)
                sum3 += coef * prod2ax
                sum5 += (coef * prod2ax) * (a * n)
                if (coef * prod2ax) * (a * n) < relerr * sum5:
                    break
                n += 1
        expx2erfcxy = expx2 * K.erfcx(y) if y > -6 else 2 * exp(y * y - x * x)
        if y > 5:
            sinxy = sin(x * y)
            ret = complex((expx2erfcxy - c * y * sum1) * cos(2 * x * y) + (c * x * expx2) * sinxy * _sinc(x * y, sinxy), 0.0)
        else:
            xs = z.real
            sinxy = sin(xs * y)
            sin2xy = sin(2 * xs * y)
            cos2xy = cos(2 * xs * y)
            coef1 = expx2erfcxy - c * y * sum1
            coef2 = c * xs * expx2
            ret = complex(coef1 * cos2xy + coef2 * sinxy * _sinc(xs * y, sinxy),
                          coef2 * _sinc(2 * xs * y, sin2xy) - coef1 * sin2xy)
    else:
        if x != x:
            return complex(x, x)
        if y != y:
            return complex(y, y)
        ret = complex(exp(-x * x), 0.0)
        n0 = K.floor(x / a + 0.5)
        dx = a * n0 - x
        sum3 = exp(-dx * dx) / (a2 * (n0 * n0) + y * y)
        sum5 = a * n0 * sum3
        exp1 = exp(4 * a * dx)
        exp1dn = 1.0
        dn = 1
        done = False
        while n0 - dn > 0:
            np_ = n0 + dn
            nm = n0 - dn
            tp = exp(-(a * dn + dx) * (a * dn + dx))
            exp1dn *= exp1
            tm = tp * exp1dn
            tp /= (a2 * (np_ * np_) + y * y)
            tm /= (a2 * (nm * nm) + y * y)
            sum3 += tp + tm
            sum5 += a * (np_ * tp + nm * tm)
            if a * (np_ * tp + nm * tm) < relerr * sum5:
                done = True
                break
            dn += 1
        while not done:
            np_ = n0 + dn
            dn += 1
            tp = exp(-(a * dn + dx) * (a * dn + dx)) / (a2 * (np_ * np_) + y * y)
            sum3 += tp
            sum5 += a * np_ * tp
            if a * np_ * tp < relerr * sum5:
                done = True
    return ret + complex((0.5 * c) * y * (sum2 + sum3), (0.5 * c) * K._copysign(sum5 - sum4, z.real))

def voigt_profile(x: float, sigma: float, gamma: float) -> float:
    INV_SQRT_2 = 0.707106781186547524401
    SQRT_2PI = 2.5066282746310002416123552393401042
    if sigma == 0:
        if gamma == 0:
            if x != x:
                return x
            if x == 0:
                return INF
            return 0.0
        return gamma / 3.14159265358979323846 / (x * x + gamma * gamma)
    if gamma == 0:
        return 1 / SQRT_2PI / sigma * exp(-(x / sigma) * (x / sigma) / 2)
    zz = complex(x / sigma * INV_SQRT_2, gamma / sigma * INV_SQRT_2)
    return w(zz).real / sigma / SQRT_2PI
