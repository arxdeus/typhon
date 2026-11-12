# Ported from scipy 1.18.1 (BSD-3-Clause); scalar kernels follow the Cephes
# Mathematical Library (Stephen L. Moshier, BSD-licensed in scipy) and the
# xsf/Boost algorithms scipy dispatches to.

# Scalar kernels operate on float. The public API (``__init__``) broadcasts
# them elementwise over ndarrays.

from math import pi, isnan, isinf

# libm directly: IEEE semantics (nan/inf) instead of Python domain errors
from C import exp(float) -> float
from C import log(float) -> float
from C import sqrt(float) -> float
from C import fabs(float) -> float
from C import sin(float) -> float
from C import cos(float) -> float
from C import tan(float) -> float
from C import log1p(float) -> float
from C import expm1(float) -> float
from C import pow(float, float) -> float
from C import copysign(float, float) -> float
from C import acos(float) -> float
from C import log2(float) -> float
from C import cbrt(float) -> float
from C import ldexp(float, i32) -> float
def _copysign(a: float, b: float) -> float:
    return copysign(a, b)

MACHEP = 1.11022302462515654042e-16
MAXLOG = 7.09782712893383996843e2
MINLOG = -7.08396418532264106224e2
MAXNUM = 1.79769313486231570815e308
EULER = 0.577215664901532860606512090082402431
NAN = float('nan')
INF = float('inf')
SQRT2 = 1.41421356237309504880
S2PI = 2.50662827463100050242

def floor(x: float) -> float:
    from math import floor as _fl
    return float(_fl(x))

def ceil(x: float) -> float:
    return -floor(-x)

def _polevl(x: float, c) -> float:
    r = 0.0
    for v in c:
        r = r * x + v
    return r

def _p1evl(x: float, c) -> float:
    r = 1.0
    for v in c:
        r = r * x + v
    return r

# --------------------------------------------------------------------- gamma

_GP = (1.60119522476751861407e-4, 1.19135147006586384913e-3, 1.04213797561761569935e-2,
       4.76367800457137231464e-2, 2.07448227648435975150e-1, 4.94214826801497100753e-1,
       9.99999999999999996796e-1)
_GQ = (-2.31581873324120129819e-5, 5.39605580493303397842e-4, -4.45641913851797240494e-3,
       1.18139785222060435552e-2, 3.58236398605498653373e-2, -2.34591795718243348568e-1,
       7.14304917030273074085e-2, 1.00000000000000000320e0)
_STIR = (7.87311395793093628397e-4, -2.29549961613378126380e-4, -2.68132617805781232825e-3,
         3.47222221605458667310e-3, 8.33333333333482257126e-2)
MAXSTIR = 143.01608

def _stirf(x: float) -> float:
    if x >= MAXGAM:
        return INF
    w = 1.0 / x
    w = 1.0 + w * _polevl(w, _STIR)
    y = exp(x)
    if x > MAXSTIR:
        v = x ** (0.5 * x - 0.25)
        y = v * (v / y)
    else:
        y = x ** (x - 0.5) / y
    return S2PI * y * w

MAXGAM = 171.624376956302725

def gamma(x: float) -> float:
    if not (x == x):
        return x
    if isinf(x):
        return x if x > 0 else NAN
    if x == 0.0:
        return -INF if _signbit(x) else INF
    if x < 0.0 and x == floor(x):
        return NAN
    sgngam = 1.0
    q = fabs(x)
    if q > 33.0:
        if x < 0.0:
            p = floor(q)
            if p == q:
                return INF   # scipy: pole -> inf
            i = int(p)
            if (i & 1) == 0:
                sgngam = -1.0
            z = q - p
            if z > 0.5:
                p += 1.0
                z = q - p
            z = q * sin(pi * z)
            if z == 0.0:
                return sgngam * INF
            z = fabs(z)
            z = pi / (z * _stirf(q))
        else:
            z = _stirf(x)
        return sgngam * z
    z = 1.0
    while x >= 3.0:
        x -= 1.0
        z *= x
    while x < 0.0:
        if x > -1.0e-9:
            return _gsmall(x, z)
        z /= x
        x += 1.0
    while x < 2.0:
        if x < 1.0e-9:
            return _gsmall(x, z)
        z /= x
        x += 1.0
    if x == 2.0:
        return z
    x -= 2.0
    p = _polevl(x, _GP)
    q = _polevl(x, _GQ)
    return z * p / q

def _gsmall(x: float, z: float) -> float:
    if x == 0.0:
        return INF  # scipy returns inf at 0 (and -0 -> -inf)
    return z / ((1.0 + 0.5772156649015329 * x) * x)

# log|gamma|
_LA = (8.11614167470508450300e-4, -5.95061904284301438324e-4, 7.93650340457716943945e-4,
       -2.77777777730099687205e-3, 8.33333333333331927722e-2)
_LB = (-1.37825152569120859100e3, -3.88016315134637840924e4, -3.31612992738871184744e5,
       -1.16237097492762307383e6, -1.72173700820839662146e6, -8.53555664245765465627e5)
_LC = (-3.51815701436523470549e2, -1.70642106651881159223e4, -2.20528590553854454839e5,
       -1.13933444367982507207e6, -2.53252307177582951285e6, -2.01889141433532773231e6)
LS2PI = 0.91893853320467274178
LOGPI = 1.14472988584940017414
MAXLGM = 2.556348e305

def gammaln(x: float) -> float:
    if not (x == x):
        return x
    if isinf(x):
        return INF
    if x < -34.0:
        q = -x
        w = gammaln(q)
        p = floor(q)
        if p == q:
            return INF
        z = q - p
        if z > 0.5:
            p += 1.0
            z = p - q
        z = q * sin(pi * z)
        if z == 0.0:
            return INF
        return LOGPI - log(fabs(z)) - w
    if x < 13.0:
        z = 1.0
        p = 0.0
        u = x
        while u >= 3.0:
            p -= 1.0
            u = x + p
            z *= u
        while u < 2.0:
            if u == 0.0:
                return INF
            z /= u
            p += 1.0
            u = x + p
        if z < 0.0:
            z = -z
        if u == 2.0:
            return log(z)
        p -= 2.0
        x = x + p
        p = x * _polevl(x, _LB) / _p1evl(x, _LC)
        return log(z) + p
    if x > MAXLGM:
        return INF
    q = (x - 0.5) * log(x) - x + LS2PI
    if x > 1.0e8:
        return q
    p = 1.0 / (x * x)
    if x >= 1000.0:
        q += ((7.9365079365079365079365e-4 * p - 2.7777777777777777777778e-3) * p
              + 0.0833333333333333333333) / x
    else:
        q += _polevl(p, _LA) / x
    return q

def gammasgn(x: float) -> float:
    if x > 0:
        return 1.0
    if x == 0:
        return 1.0 if not _signbit(x) else -1.0
    if not (x == x):
        return NAN
    fx = floor(x)
    if x - fx == 0.0:
        return NAN
    return -1.0 if (int(fx) % 2) else 1.0

def _signbit(x: float) -> bool:
    return _copysign(1.0, x) < 0

# ------------------------------------------------------------------- digamma

_PSI_A = (8.33333333333333333333e-2, -2.10927960927960927961e-2, 7.57575757575757575758e-3,
          -4.16666666666666666667e-3, 3.96825396825396825397e-3, -8.33333333333333333333e-3,
          8.33333333333333333333e-2)

def _psi_asy(x: float) -> float:
    if x < 1.0e17:
        z = 1.0 / (x * x)
        y = z * _polevl(z, _PSI_A)
    else:
        y = 0.0
    return log(x) - (0.5 / x) - y

# xsf digamma: series around the first positive root, Cephes elsewhere
_PSI_ROOT1 = 1.4616321449683622
_PSI_ROOT1_HI = 1.4616321449683622
_PSI_ROOT1_LO = 0.0  # placeholder; refined path uses zeta series below

def _zeta_hurwitz(s: float, a: float) -> float:
    # Hurwitz zeta via Euler-Maclaurin (Cephes zeta.c)
    A = (12.0, -720.0, 30240.0, -1209600.0, 47900160.0, -1.8924375803183791606e9,
         7.47242496e10, -2.950130727918164224e12, 1.1646782814350067249e14,
         -4.5979787224074726105e15, 1.8152105401943546773e17, -7.1661652561756670113e18)
    if s == 1.0:
        return INF
    if a <= 0.0:
        if a == floor(a):
            return INF
    sm = a ** (-s)
    aa = a
    i = 0
    b = 0.0
    while i < 9 or aa <= 9.0:
        i += 1
        aa += 1.0
        b = aa ** (-s)
        sm += b
        if fabs(b / sm) < MACHEP:
            return sm
    w = aa
    sm += b * w / (s - 1.0)
    sm -= 0.5 * b
    ac = 1.0
    k = 0.0
    for j in range(12):
        ac *= s + k
        b /= w
        t = ac * b / A[j]
        sm = sm + t
        t = fabs(t / sm)
        if t < MACHEP:
            return sm
        k += 1.0
        ac *= s + k
        b /= w
        k += 1.0
    return sm

def _digamma_near_root(x: float) -> float:
    # Taylor series of digamma around its positive root x0 (xsf)
    x0 = 1.4616321449683622
    x0lo = 1.2005468935802483e-17  # x0 = x0hi + x0lo (approx)
    xm = x - x0
    xm = xm - x0lo
    coeff = -1.0
    res = 0.0
    for n in range(1, 100):
        coeff *= -xm
        term = coeff * _zeta_hurwitz(n + 1.0, x0)
        res += term
        if fabs(term) < MACHEP * fabs(res):
            break
    return res

def digamma(x: float) -> float:
    if not (x == x):
        return x
    if x == 0.0:
        return -INF if not _signbit(x) else INF   # scipy: psi(0) = -inf? see note
    if x < 0.0:
        if x == floor(x):
            return NAN
        # reflection
        q = x
        p = floor(q)
        nz = q - p
        if nz == 0.5:
            nz = 0.0
        else:
            if nz > 0.5:
                p += 1.0
                nz = q - p
            nz = pi / tan(pi * nz)
        return digamma(1.0 - x) - nz
    if fabs(x - 1.4616321449683622) < 0.5:
        return _digamma_near_root(x)
    y = 0.0
    if x <= 10.0 and x == floor(x):
        n = int(x)
        for i in range(1, n):
            y += 1.0 / i
        return y - EULER
    w = 0.0
    while x < 10.0:
        w += 1.0 / x
        x += 1.0
    return _psi_asy(x) - w

# ---------------------------------------------------------------------- beta

MAXGAMB = 171.624376956302725
ASYMP_FACTOR = 1e6

def _lbeta_asymp(a: float, b: float, sgn: List[int]) -> float:
    r = gammaln(b)
    r -= b * log(a)
    r += b * (1 - b) / (2 * a)
    r += b * (1 - b) * (1 - 2 * b) / (12 * a * a)
    r += -b * b * (1 - b) * (1 - b) / (12 * a * a * a)
    return r

def beta(a: float, b: float) -> float:
    if not (a == a) or not (b == b):
        return NAN
    sign = 1.0
    if a <= 0.0:
        if a == floor(a):
            if a == float(int(a)):
                return _beta_negint(int(a), b)
            return INF
    if b <= 0.0:
        if b == floor(b):
            if b == float(int(b)):
                return _beta_negint(int(b), a)
            return INF
    if fabs(a) < fabs(b):
        a, b = b, a
    if fabs(a) > ASYMP_FACTOR * fabs(b) and a > ASYMP_FACTOR:
        y = _lbeta_asymp(a, b, [1])
        return sign * exp(y)
    y = a + b
    if fabs(y) > MAXGAMB or fabs(a) > MAXGAMB or fabs(b) > MAXGAMB:
        sgngam = gammasgn(y)
        y = gammaln(y)
        sign *= sgngam
        sgngam = gammasgn(b)
        y = gammaln(b) - y
        sign *= sgngam
        sgngam = gammasgn(a)
        y = gammaln(a) + y
        sign *= sgngam
        if y > MAXLOG:
            return sign * INF
        return sign * exp(y)
    y = gamma(y)
    a = gamma(a)
    b = gamma(b)
    if y == 0.0:
        return sign * INF
    if fabs(fabs(a) - fabs(y)) > fabs(fabs(b) - fabs(y)):
        y = b / y
        y *= a
    else:
        y = a / y
        y *= b
    return sign * y

def _beta_negint(a: int, b: float) -> float:
    if b == float(int(b)) and 1 - a - b > 0:
        sgn = 1.0 if (int(b) % 2 == 0) else -1.0
        return sgn * beta(1 - a - b, b)
    return INF

def betaln(a: float, b: float) -> float:
    if not (a == a) or not (b == b):
        return NAN
    if a <= 0.0 and a == floor(a):
        if a == float(int(a)):
            return _lbeta_negint(int(a), b)
        return INF
    if b <= 0.0 and b == floor(b):
        if b == float(int(b)):
            return _lbeta_negint(int(b), a)
        return INF
    if fabs(a) < fabs(b):
        a, b = b, a
    if fabs(a) > ASYMP_FACTOR * fabs(b) and a > ASYMP_FACTOR:
        return _lbeta_asymp(a, b, [1])
    y = a + b
    if fabs(y) > MAXGAMB or fabs(a) > MAXGAMB or fabs(b) > MAXGAMB:
        y = gammaln(y)
        y = gammaln(b) - y
        y = gammaln(a) + y
        return y
    y = gamma(y)
    a = gamma(a)
    b = gamma(b)
    if y == 0.0:
        return INF
    if fabs(fabs(a) - fabs(y)) > fabs(fabs(b) - fabs(y)):
        y = b / y
        y *= a
    else:
        y = a / y
        y *= b
    return log(fabs(y))

def _lbeta_negint(a: int, b: float) -> float:
    if b == float(int(b)) and 1 - a - b > 0:
        return betaln(1 - a - b, b)
    return INF

# ---------------------------------------------------------- erf / ndtr family

_NP = (2.46196981473530512524e-10, 5.64189564831068821977e-1, 7.46321056442269912687e0,
       4.86371970985681366614e1, 1.96520832956077098242e2, 5.26445194995477358631e2,
       9.34528527171957607540e2, 1.02755188689515710272e3, 5.57535335369399327526e2)
_NQ = (1.32281951154744992508e1, 8.67072140885989742329e1, 3.54937778887819891062e2,
       9.75708501743205489753e2, 1.82390916687909736289e3, 2.24633760818710981792e3,
       1.65666309194161350182e3, 5.57535340817727675546e2)
_NR = (5.64189583547755073984e-1, 1.27536670759978104416e0, 5.01905042251180477414e0,
       6.16021097993053585195e0, 7.40974269950448939160e0, 2.97886665372100240670e0)
_NS = (2.26052863220117276590e0, 9.39603524938001434673e0, 1.20489539808096656605e1,
       1.70814450747565897222e1, 9.60896809063285878198e0, 3.36907645100081516050e0)
_NT = (9.60497373987051638749e0, 9.00260197203842689217e1, 2.23200534594684319226e3,
       7.00332514112805075473e3, 5.55923013010394962768e4)
_NU = (3.35617141647503099647e1, 5.21357949780152679795e2, 4.59432382970980127987e3,
       2.26290000613890934246e4, 4.92673942608635921086e4)

def erf(x: float) -> float:
    if not (x == x):
        return x
    if x < 0.0:
        return -erf(-x)
    if fabs(x) > 1.0:
        return 1.0 - erfc(x)
    z = x * x
    return x * _polevl(z, _NT) / _p1evl(z, _NU)

def erfc(a: float) -> float:
    if not (a == a):
        return a
    x = fabs(a)
    if x < 1.0:
        return 1.0 - erf(a)
    z = -a * a
    if z < -MAXLOG:
        return 2.0 if a < 0 else 0.0
    z = exp(z)
    if x < 8.0:
        p = _polevl(x, _NP)
        q = _p1evl(x, _NQ)
    else:
        p = _polevl(x, _NR)
        q = _p1evl(x, _NS)
    y = (z * p) / q
    if a < 0:
        y = 2.0 - y
    if y == 0.0:
        return 2.0 if a < 0 else 0.0
    return y

def ndtr(a: float) -> float:
    if not (a == a):
        return a
    x = a * 0.70710678118654752440
    z = fabs(x)
    if z < 0.70710678118654752440:
        y = 0.5 + 0.5 * erf(x)
    else:
        y = 0.5 * erfc(z)
        if x > 0:
            y = 1.0 - y
    return y

def log_ndtr(a: float) -> float:
    # xsf log_ndtr: log(erfc) asymptotics for a < -1, log1p for a > 6
    t = a * 0.70710678118654752440
    if a < -1.0:
        return log(erfcx(-t) / 2) - t * t
    else:
        return log1p(-erfc(t) / 2)

def erfcx(x: float) -> float:
    # scaled complementary error function exp(x^2) erfc(x)
    if not (x == x):
        return x
    if x < 0:
        if x < -26.7:
            return INF
        elif x < -6.1:
            return 2 * exp(x * x)
        else:
            return 2 * exp(x * x) - erfcx(-x)
    if x < 0.5:
        return exp(x * x) * erfc(x)
    if x < 50.0:
        # continued fraction (Lentz) for erfc(x) * exp(x^2) * sqrt(pi)
        return _erfcx_cf(x)
    ispi = 0.56418958354775628694807945156
    if x > 5e7:
        return ispi / x
    x2 = x * x
    return ispi * ((x2) * (x2 + 4.5) + 2) / (x * ((x2) * (x2 + 5) + 3.75))

def _erfcx_cf(x: float) -> float:
    # erfc(x) exp(x^2) = (1/sqrt(pi)) * 1/(x + 1/2/(x + 1/(x + 3/2/(x + ...))))
    tiny = 1e-300
    f = x
    C = x
    D = 0.0
    for k in range(1, 5000):
        an = k * 0.5
        D = x + an * D
        if D == 0.0:
            D = tiny
        C = x + an / C
        if C == 0.0:
            C = tiny
        D = 1.0 / D
        delta = C * D
        f *= delta
        if fabs(delta - 1.0) < 1e-16:
            break
    return 0.56418958354775628694807945156 / f

# ndtri (Cephes)
_P0 = (-5.99633501014107895267e1, 9.80010754185999661536e1, -5.66762857469070293439e1,
       1.39312609387279679503e1, -1.23916583867381258016e0)
_Q0 = (1.95448858338141759834e0, 4.67627912898881538453e0, 8.63602421390890590575e1,
       -2.25462687854119370527e2, 2.00260212380060660359e2, -8.20372256168333339912e1,
       1.59056225126211695515e1, -1.18331621121330003142e0)
_P1 = (4.05544892305962419923e0, 3.15251094599893866154e1, 5.71628192246421288162e1,
       4.40805073893200834700e1, 1.46849561928858024014e1, 2.18663306850790267539e0,
       -1.40256079171354495875e-1, -3.50424626827848203418e-2, -8.57456785154685413611e-4)
_Q1 = (1.57799883256466749731e1, 4.53907635128879210584e1, 4.13172038254672030440e1,
       1.50425385692907503408e1, 2.50464946208309415979e0, -1.42182922854787788574e-1,
       -3.80806407691578277194e-2, -9.33259480895457427372e-4)
_P2 = (3.23774891776946035970e0, 6.91522889068984211695e0, 3.93881025292474443415e0,
       1.33303460815807542389e0, 2.01485389549179081538e-1, 1.23716634817820021358e-2,
       3.01581553508235416007e-4, 2.65806974686737550832e-6, 6.23974539184983293730e-9)
_Q2 = (6.02427039364742014255e0, 3.67983563856160859403e0, 1.37702099489081330271e0,
       2.16236993594496635890e-1, 1.34204006088543189037e-2, 3.28014464682127739104e-4,
       2.89247864745380683936e-6, 6.79019408009981274425e-9)

def ndtri(y0: float) -> float:
    if not (y0 == y0):
        return y0
    if y0 == 0.0:
        return -INF
    if y0 == 1.0:
        return INF
    if y0 < 0.0 or y0 > 1.0:
        return NAN
    code = 1
    y = y0
    if y > (1.0 - 0.13533528323661269189):
        y = 1.0 - y
        code = 0
    if y > 0.13533528323661269189:
        y = y - 0.5
        y2 = y * y
        x = y + y * (y2 * _polevl(y2, _P0) / _p1evl(y2, _Q0))
        x = x * S2PI
        return x
    x = sqrt(-2.0 * log(y))
    x0 = x - log(x) / x
    z = 1.0 / x
    if x < 8.0:
        x1 = z * _polevl(z, _P1) / _p1evl(z, _Q1)
    else:
        x1 = z * _polevl(z, _P2) / _p1evl(z, _Q2)
    x = x0 - x1
    if code != 0:
        x = -x
    return x

def erfinv(y: float) -> float:
    if not (y == y) or y < -1 or y > 1:
        return NAN
    if y == 0.0:
        return y
    if y == 1.0:
        return INF
    if y == -1.0:
        return -INF
    if -1e-7 < y < 1e-7:
        return y / 1.1283791670955126  # 2/sqrt(pi)
    # initial guess via ndtri, then Newton/Halley refinement on erf
    x = ndtri(0.5 * (y + 1.0)) * 0.70710678118654752440
    return _refine_erfinv(x, y)

def _refine_erfinv(x: float, y: float) -> float:
    for _ in range(3):
        if fabs(x) < 1.0:
            err = erf(x) - y
        else:
            err = (1.0 - y) - erfc(x) if y > 0 else -((1.0 + y) - erfc(-x))
        d = 1.1283791670955126 * exp(-x * x)
        if d == 0.0:
            break
        dx = err / d
        x = x - dx / (1.0 + x * dx)
    return x

def erfcinv(y: float) -> float:
    if not (y == y) or y < 0 or y > 2:
        return NAN
    if y == 0.0:
        return INF
    if y == 2.0:
        return -INF
    if y == 1.0:
        return 0.0
    x = -ndtri(0.5 * y) * 0.70710678118654752440
    for _ in range(3):
        if x > 0:
            err = erfc(x) - y
        else:
            err = (2.0 - erfc(-x)) - y
        d = -1.1283791670955126 * exp(-x * x)
        if d == 0.0:
            break
        dx = err / d
        x = x - dx / (1.0 - x * dx)
    return x

# ------------------------------------------------- incomplete gamma (Cephes)

BIG = 4.503599627370496e15
BIGINV = 2.22044604925031308085e-16

def _igam_fac(a: float, x: float) -> float:
    # x^a e^-x / Gamma(a), with Temme-like accuracy via lanczos (xsf uses
    # Lanczos; here gammaln-based with large-a correction)
    if fabs(a - x) > 0.4 * fabs(a):
        ax = a * log(x) - x - gammaln(a)
        if ax < -MAXLOG:
            return 0.0
        return exp(ax)
    return _lanczos_fac(a, x)

_LANCZOS_G = 6.024680040776729583740234375
_LZ_NUM = (2.506628274631000270164908177133837338626, 210.8242777515793458725097339207133627117,
           8071.672002365816210638002902272250613822, 186056.2653952234950402949897160456992822,
           2876370.628935372441225409051620849613599, 31426415.58540019438061423162831820536287,
           248874557.8620541565114603864132294232163, 1439720407.311721673663223072794912393972,
           6039542586.35202800506429164430729792107, 17921034426.03720969991975575445893111267,
           35711959237.35566804944018545154716670596, 42919803642.64909876895789904700198885093,
           23531376880.41075968857200767445163675473)
_LZ_DEN = (1.0, 66.0, 1925.0, 32670.0, 357423.0, 2637558.0, 13339535.0, 45995730.0,
           105258076.0, 150917976.0, 120543840.0, 39916800.0, 0.0)
_LZ_NUM_SCALED = (0.006061842346248906525783753964555936883222, 0.5098416655656676188125178644804694509993,
                  19.51992788247617482847860966235652136208, 449.9445569063168119446858607650988409623,
                  6955.999602515376140356310115515198987526, 75999.29304014542649875303443598909137092,
                  601859.6171681098786670226533699352302507, 3481712.15498064590882071018964774556468,
                  14605578.08768506808414169982791359218571, 43338889.32467613834773723740590533316085,
                  86363131.28813859145546927288977868422342, 103794043.1163445451906271053616070238554,
                  56906521.91347156388090791033559122686859)

def _ratevl_lz(x: float, num, den) -> float:
    # rational evaluation with highest power first in tables reversed (Boost order)
    # Cephes ratevl: coefficients highest power first; for |x| > 1 evaluate
    # the reversed polynomials in 1/x
    absx = fabs(x)
    if absx > 1:
        y = 1.0 / x
        nn = 0.0
        for i in range(12, -1, -1):
            nn = nn * y + num[i]
        dd = 0.0
        for i in range(12, -1, -1):
            dd = dd * y + den[i]
        return nn / dd
    else:
        nn = 0.0
        for i in range(13):
            nn = nn * x + num[i]
        dd = 0.0
        for i in range(13):
            dd = dd * x + den[i]
        return nn / dd

def _lanczos_sum_expg_scaled(x: float) -> float:
    return _ratevl_lz(x, _LZ_NUM_SCALED, _LZ_DEN)

def _lanczos_fac(a: float, x: float) -> float:
    fac = a + _LANCZOS_G - 0.5
    res = sqrt(fac / exp(1.0)) / _lanczos_sum_expg_scaled(a)
    if a < 200 and x < 200:
        res *= exp(a - x) * (x / fac) ** a
    else:
        num = x - a - _LANCZOS_G + 0.5
        res *= exp(a * log1pmx(num / fac) + x * (0.5 - _LANCZOS_G) / fac)
    return res

def log1pmx(x: float) -> float:
    # log(1+x) - x
    if fabs(x) < 0.5:
        xfac = x
        res = 0.0
        for n in range(2, 500):
            xfac *= -x
            term = xfac / n
            res += term
            if fabs(term) < MACHEP * fabs(res):
                break
        return res
    return log1p(x) - x

def _igam_series(a: float, x: float) -> float:
    ax = _igam_fac(a, x)
    if ax == 0.0:
        return 0.0
    r = a
    c = 1.0
    ans = 1.0
    for i in range(2000):
        r += 1.0
        c *= x / r
        ans += c
        if c <= MACHEP * ans:
            break
    return ans * ax / a

def _igamc_cf(a: float, x: float) -> float:
    ax = _igam_fac(a, x)
    if ax == 0.0:
        return 0.0
    y = 1.0 - a
    z = x + y + 1.0
    c = 0.0
    pkm2 = 1.0
    qkm2 = x
    pkm1 = x + 1.0
    qkm1 = z * x
    ans = pkm1 / qkm1
    for i in range(2000):
        c += 1.0
        y += 1.0
        z += 2.0
        yc = y * c
        pk = pkm1 * z - pkm2 * yc
        qk = qkm1 * z - qkm2 * yc
        if qk != 0:
            r = pk / qk
            t = fabs((ans - r) / r)
            ans = r
        else:
            t = 1.0
        pkm2 = pkm1
        pkm1 = pk
        qkm2 = qkm1
        qkm1 = qk
        if fabs(pk) > BIG:
            pkm2 *= BIGINV
            pkm1 *= BIGINV
            qkm2 *= BIGINV
            qkm1 *= BIGINV
        if t <= MACHEP:
            break
    return ans * ax

def _igamc_series(a: float, x: float) -> float:
    # Q(a, x) for small x via series of P complement (DLMF 8.7.3)
    fac = 1.0
    sm = 0.0
    for n in range(1, 2000):
        fac *= -x / n
        term = fac / (a + n)
        sm += term
        if fabs(term) <= MACHEP * fabs(sm):
            break
    logx = log(x)
    term = -expm1(a * logx - gammaln(a + 1.0))
    return term - exp(a * logx - gammaln(a)) * sm

SMALL = 20.0
LARGE = 200.0
SMALLRATIO = 0.3
LARGERATIO = 4.5

def gammainc(a: float, x: float) -> float:
    if x < 0 or a < 0 or not (a == a) or not (x == x):
        return NAN
    if a == 0:
        return 1.0 if x > 0 else NAN
    if x == 0:
        return 0.0
    if isinf(a):
        return NAN if isinf(x) else 0.0
    if isinf(x):
        return 1.0
    if x > 1.0 and x > a:
        return 1.0 - gammaincc(a, x)
    return _igam_series(a, x)

def gammaincc(a: float, x: float) -> float:
    if x < 0 or a < 0 or not (a == a) or not (x == x):
        return NAN
    if a == 0:
        return 0.0 if x > 0 else NAN
    if x == 0:
        return 1.0
    if isinf(a):
        return NAN if isinf(x) else 1.0
    if isinf(x):
        return 0.0
    if x > 1.1:
        if x < a:
            return 1.0 - _igam_series(a, x)
        else:
            return _igamc_cf(a, x)
    elif x <= 0.5:
        if -0.4 / log(x) < a:
            return 1.0 - _igam_series(a, x)
        else:
            return _igamc_series(a, x)
    else:
        if x * 1.1 < a:
            return 1.0 - _igam_series(a, x)
        else:
            return _igamc_series(a, x)

def gammaincinv(a: float, y: float) -> float:
    # inverse of regularized lower incomplete gamma: Newton/Halley on P
    if not (a == a) or not (y == y) or a < 0 or y < 0 or y > 1:
        return NAN
    if y == 0.0:
        return 0.0
    if y == 1.0:
        return INF
    if a == 0:
        return 0.0 if y < 1 else INF
    return _igami(a, y, False)

def gammainccinv(a: float, y: float) -> float:
    if not (a == a) or not (y == y) or a < 0 or y < 0 or y > 1:
        return NAN
    if y == 0.0:
        return INF
    if y == 1.0:
        return 0.0
    return _igami(a, y, True)

def _igami(a: float, p: float, upper: bool) -> float:
    # Wilson-Hilferty initial guess, then Halley steps safeguarded by a
    # bisection bracket (P is increasing in x).
    q = 1.0 - p if not upper else p
    pl = p if not upper else 1.0 - p
    if a == 1.0:
        if upper:
            return -log(q)
        return -log1p(-pl)
    d = 1.0 / (9.0 * a)
    z = -ndtri(q) if not upper else -ndtri(p)
    x = a * (1.0 - d + z * sqrt(d)) ** 3
    if not (x > 0) or isinf(x):
        x = exp((log(pl) + gammaln(a + 1.0)) / a) if pl > 0 else 1e-300
    if a < 1.0 and pl < 0.5:
        x = exp((log(pl) + gammaln(a + 1.0)) / a)
    lo = 0.0
    hi = INF
    for it in range(300):
        if upper:
            f = q - gammaincc(a, x)   # increasing in x like P - pl
        else:
            f = gammainc(a, x) - pl
        if f < 0:
            lo = x
        elif f > 0:
            hi = x
        else:
            break
        fp = _igam_fac(a, x) / x
        if fp == 0.0 or not (fp == fp):
            xn = 0.5 * (lo + hi) if hi < INF else 2.0 * x + 1.0
        else:
            dx = f / fp
            f2 = (a - 1.0) / x - 1.0
            corr = 0.5 * dx * f2
            if fabs(corr) < 1.0:
                dx = dx / (1.0 - corr)
            xn = x - dx
            if not (xn > lo and xn < hi):
                xn = 0.5 * (lo + hi) if hi < INF else 2.0 * x + 1.0
        if fabs(xn - x) <= 2e-16 * fabs(xn):
            x = xn
            break
        x = xn
    return x

# --------------------------------------------- incomplete beta (Cephes incbet)

def _incbcf(a: float, b: float, x: float) -> float:
    k1 = a; k2 = a + b; k3 = a; k4 = a + 1.0
    k5 = 1.0; k6 = b - 1.0; k7 = k4; k8 = a + 2.0
    pkm2 = 0.0; qkm2 = 1.0; pkm1 = 1.0; qkm1 = 1.0
    ans = 1.0; r = 1.0
    thresh = 3.0 * MACHEP
    for n in range(300):
        xk = -(x * k1 * k2) / (k3 * k4)
        pk = pkm1 + pkm2 * xk
        qk = qkm1 + qkm2 * xk
        pkm2 = pkm1; pkm1 = pk; qkm2 = qkm1; qkm1 = qk
        xk = (x * k5 * k6) / (k7 * k8)
        pk = pkm1 + pkm2 * xk
        qk = qkm1 + qkm2 * xk
        pkm2 = pkm1; pkm1 = pk; qkm2 = qkm1; qkm1 = qk
        if qk != 0:
            r = pk / qk
        if r != 0:
            t = fabs((ans - r) / r)
            ans = r
        else:
            t = 1.0
        if t < thresh:
            break
        k1 += 1.0; k2 += 1.0; k3 += 2.0; k4 += 2.0
        k5 += 1.0; k6 -= 1.0; k7 += 2.0; k8 += 2.0
        if fabs(qk) + fabs(pk) > BIG:
            pkm2 *= BIGINV; pkm1 *= BIGINV; qkm2 *= BIGINV; qkm1 *= BIGINV
        if fabs(qk) < BIGINV or fabs(pk) < BIGINV:
            pkm2 *= BIG; pkm1 *= BIG; qkm2 *= BIG; qkm1 *= BIG
    return ans

def _incbd(a: float, b: float, x: float) -> float:
    k1 = a; k2 = b - 1.0; k3 = a; k4 = a + 1.0
    k5 = 1.0; k6 = a + b; k7 = a + 1.0; k8 = a + 2.0
    pkm2 = 0.0; qkm2 = 1.0; pkm1 = 1.0; qkm1 = 1.0
    z = x / (1.0 - x)
    ans = 1.0; r = 1.0
    thresh = 3.0 * MACHEP
    for n in range(300):
        xk = -(z * k1 * k2) / (k3 * k4)
        pk = pkm1 + pkm2 * xk
        qk = qkm1 + qkm2 * xk
        pkm2 = pkm1; pkm1 = pk; qkm2 = qkm1; qkm1 = qk
        xk = (z * k5 * k6) / (k7 * k8)
        pk = pkm1 + pkm2 * xk
        qk = qkm1 + qkm2 * xk
        pkm2 = pkm1; pkm1 = pk; qkm2 = qkm1; qkm1 = qk
        if qk != 0:
            r = pk / qk
        if r != 0:
            t = fabs((ans - r) / r)
            ans = r
        else:
            t = 1.0
        if t < thresh:
            break
        k1 += 1.0; k2 -= 1.0; k3 += 2.0; k4 += 2.0
        k5 += 1.0; k6 += 1.0; k7 += 2.0; k8 += 2.0
        if fabs(qk) + fabs(pk) > BIG:
            pkm2 *= BIGINV; pkm1 *= BIGINV; qkm2 *= BIGINV; qkm1 *= BIGINV
        if fabs(qk) < BIGINV or fabs(pk) < BIGINV:
            pkm2 *= BIG; pkm1 *= BIG; qkm2 *= BIG; qkm1 *= BIG
    return ans

def _pseries(a: float, b: float, x: float) -> float:
    ai = 1.0 / a
    u = (1.0 - b) * x
    v = u / (a + 1.0)
    t1 = v
    t = u
    n = 2.0
    s = 0.0
    z = MACHEP * ai
    while fabs(v) > z:
        u = (n - b) * x / n
        t *= u
        v = t / (a + n)
        s += v
        n += 1.0
    s += t1
    s += ai
    u = a * log(x)
    if (a + b) < MAXGAM and fabs(u) < MAXLOG:
        t = 1.0 / beta(a, b)
        s = s * t * (x ** a)
    else:
        t = -betaln(a, b) + u + log(s)
        if t < MINLOG:
            s = 0.0
        else:
            s = exp(t)
    return s

def betainc(aa: float, bb: float, xx: float) -> float:
    if not (aa == aa) or not (bb == bb) or not (xx == xx):
        return NAN
    if aa < 0.0 or bb < 0.0:
        return NAN
    if xx < 0.0 or xx > 1.0:
        return NAN
    if aa == 0.0 and bb == 0.0:
        return NAN
    if aa == 0.0:
        return 1.0 if xx > 0 else 0.0 if False else 1.0
    if bb == 0.0:
        return 0.0 if xx < 1 else 1.0
    if isinf(aa) or isinf(bb):
        if isinf(aa) and isinf(bb):
            return NAN
        if isinf(aa):
            return 0.0 if xx < 1 else 1.0
        return 1.0 if xx > 0 else 0.0
    if xx == 0.0:
        return 0.0
    if xx == 1.0:
        return 1.0
    if (bb * xx) <= 1.0 and xx <= 0.95:
        return _pseries(aa, bb, xx)
    w = 1.0 - xx
    flag = False
    if xx > (aa / (aa + bb)):
        flag = True
        a = bb; b = aa; xc = xx; x = w
    else:
        a = aa; b = bb; xc = w; x = xx
    if flag and (b * x) <= 1.0 and x <= 0.95:
        t = _pseries(a, b, x)
        if t <= MACHEP:
            t = 1.0 - MACHEP
        else:
            t = 1.0 - t
        return t
    y = x * (a + b - 2.0) - (a - 1.0)
    if y < 0.0:
        w = _incbcf(a, b, x)
    else:
        w = _incbd(a, b, x) / xc
    y = a * log(x)
    t = b * log(xc)
    if (a + b) < MAXGAM and fabs(y) < MAXLOG and fabs(t) < MAXLOG:
        t = xc ** b
        t *= x ** a
        t /= a
        t *= w
        t *= 1.0 / beta(a, b)
    else:
        y += t - betaln(a, b)
        y += log(w / a)
        if y < MINLOG:
            t = 0.0
        else:
            t = exp(y)
    if flag:
        if t <= MACHEP:
            t = 1.0 - MACHEP
        else:
            t = 1.0 - t
    return t

def betaincc(a: float, b: float, x: float) -> float:
    if not (x == x):
        return NAN
    return betainc(b, a, 1.0 - x) if x != 0 else 1.0

def betaincinv(a: float, b: float, y: float) -> float:
    # inverse of regularized incomplete beta: bisection-safeguarded Newton
    if not (a == a) or not (b == b) or not (y == y):
        return NAN
    if a <= 0 or b <= 0 or y < 0 or y > 1:
        return NAN
    if y == 0.0:
        return 0.0
    if y == 1.0:
        return 1.0
    # initial approximation (Abramowitz & Stegun 26.5.22)
    if a > 1.0 and b > 1.0:
        yp = -ndtri(y)
        lmb = (yp * yp - 3.0) / 6.0
        h = 2.0 / (1.0 / (2.0 * a - 1.0) + 1.0 / (2.0 * b - 1.0))
        w = (yp * sqrt(h + lmb) / h
             - (1.0 / (2.0 * b - 1.0) - 1.0 / (2.0 * a - 1.0)) * (lmb + 5.0 / 6.0 - 2.0 / (3.0 * h)))
        x = a / (a + b * exp(2.0 * w))
    else:
        lna = log(a / (a + b))
        lnb = log(b / (a + b))
        t = exp(a * lna) / a
        u = exp(b * lnb) / b
        w = t + u
        if y < t / w:
            x = (a * w * y) ** (1.0 / a)
        else:
            x = 1.0 - (b * w * (1.0 - y)) ** (1.0 / b)
    lo = 0.0
    hi = 1.0
    lbeta = betaln(a, b)
    for it in range(300):
        if not (x > lo and x < hi):
            x = 0.5 * (lo + hi)
        f = betainc(a, b, x) - y
        if f < 0:
            lo = x
        else:
            hi = x
        if f == 0.0:
            break
        dens = exp((a - 1.0) * log(x) + (b - 1.0) * log1p(-x) - lbeta)
        if dens == 0.0 or not (dens == dens):
            x = 0.5 * (lo + hi)
            continue
        dx = f / dens
        f2 = dens * ((a - 1.0) / x - (b - 1.0) / (1.0 - x))
        corr = 0.5 * dx * f2 / dens
        if fabs(corr) < 1.0:
            dx = dx / (1.0 - corr)
        xn = x - dx
        if xn <= lo or xn >= hi:
            xn = 0.5 * (lo + hi)
        if fabs(xn - x) <= 4e-16 * fabs(xn) or hi - lo <= 4e-16 * fabs(x):
            x = xn
            break
        x = xn
    return x

# ------------------------------------------------- distributions (cdtr etc.)

def stdtr(df: float, t: float) -> float:
    if df <= 0 or not (df == df) or not (t == t):
        return NAN
    if isinf(t):
        return 1.0 if t > 0 else 0.0
    if isinf(df):
        return ndtr(t)
    if t == 0:
        return 0.5
    x = df / (df + t * t)
    p = 0.5 * betainc(0.5 * df, 0.5, x)
    return 1.0 - p if t > 0 else p

# Student t quantile: port of Boost.Math fast_students_t_quantile (the
# backend of scipy 1.18 stdtrit): exact forms for df 1, 2, 4, 6, Hill /
# body / tail series estimates otherwise, refined by up to 3 Halley steps.

def _st_hill(ndf: float, u: float) -> float:
    if ndf > 1e20:
        return ndtri(u)
    a = 1 / (ndf - 0.5)
    b = 48 / (a * a)
    c = ((20700 * a / b - 98) * a - 16) * a + 96.36
    d = ((94.5 / (b + c) - 3) / b + 1) * sqrt(a * pi / 2) * ndf
    y = pow(d * 2 * u, 2 / ndf)
    if y > (0.05 + a):
        x = ndtri(u)
        y = x * x
        if ndf < 5:
            c += 0.3 * (ndf - 4.5) * (x + 0.6)
        c += (((0.05 * d * x - 5) * x - 7) * x - 2) * x + b
        y = (((((0.4 * y + 6.3) * y + 36) * y + 94.5) / c - y - 3) / b + 1) * x
        y = expm1(a * y * y)
    else:
        y = (((1 / (((ndf + 6) / (ndf * y) - 0.089 * d - 0.822) * (ndf + 2) * 3) + 0.5 / (ndf + 4)) * y - 1) * (ndf + 1) / (ndf + 2) + 1 / y)
    return -sqrt(ndf * y)

def _tgamma_delta_ratio_half(z: float) -> float:
    # Gamma(z) / Gamma(z + 0.5)
    return exp(-_log_gamma_ratio(z, 0.5))

def _st_tail_series(df: float, v: float) -> float:
    w = _tgamma_delta_ratio_half(df / 2) * sqrt(df * pi) * v
    np2 = df + 2
    np4 = df + 4
    np6 = df + 6
    d = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    d[1] = -(df + 1) / (2 * np2)
    np2 *= (df + 2)
    d[2] = -df * (df + 1) * (df + 3) / (8 * np2 * np4)
    np2 *= df + 2
    d[3] = -df * (df + 1) * (df + 5) * (((3 * df) + 7) * df - 2) / (48 * np2 * np4 * np6)
    np2 *= (df + 2)
    np4 *= (df + 4)
    d[4] = -df * (df + 1) * (df + 7) * ((((((15 * df) + 154) * df + 465) * df + 286) * df - 336) * df + 64) / (384 * np2 * np4 * np6 * (df + 8))
    np2 *= (df + 2)
    d[5] = -df * (df + 1) * (df + 3) * (df + 9) * (((((((35 * df + 452) * df + 1573) * df + 600) * df - 2020) * df) + 928) * df - 128) / (1280 * np2 * np4 * np6 * (df + 8) * (df + 10))
    np2 *= (df + 2)
    np4 *= (df + 4)
    np6 *= (df + 6)
    d[6] = -df * (df + 1) * (df + 11) * ((((((((((((945 * df) + 31506) * df + 425858) * df + 2980236) * df + 11266745) * df + 20675018) * df + 7747124) * df - 22574632) * df - 8565600) * df + 18108416) * df - 7099392) * df + 884736) / (46080 * np2 * np4 * np6 * (df + 8) * (df + 10) * (df + 12))
    rn = sqrt(df)
    div = pow(rn * w, 1 / df)
    power = div * div
    r = d[6]
    for i in range(5, -1, -1):
        r = r * power + d[i]
    r *= rn
    r /= div
    return -r

def _st_body_series(df: float, u: float) -> float:
    v = _tgamma_delta_ratio_half(df / 2) * sqrt(df * pi) * (u - 0.5)
    i = 1 / df
    c = [0.0] * 11
    c[1] = 1.0
    c[2] = 0.16666666666666666667 + 0.16666666666666666667 * i
    c[3] = (0.0083333333333333333333 * i + 0.066666666666666666667) * i + 0.058333333333333333333
    c[4] = ((0.00019841269841269841270 * i + 0.0017857142857142857143) * i + 0.026785714285714285714) * i + 0.025198412698412698413
    c[5] = (((2.7557319223985890653e-6 * i + 0.00037477954144620811287) * i - 0.0011078042328042328042) * i + 0.010559964726631393298) * i + 0.012039792768959435626
    c[6] = ((((2.5052108385441718775e-8 * i - 0.000062705427288760622094) * i + 0.00059458674042007375341) * i - 0.0016095979637646304313) * i + 0.0061039211560044893378) * i + 0.0038370059724226390893
    c[7] = (((((1.6059043836821614599e-10 * i + 0.000015401265401265401265) * i - 0.00016376804137220803887) * i + 0.00069084207973096861986) * i - 0.0012579159844784844785) * i + 0.0010898206731540064873) * i + 0.0032177478835464946576
    c[8] = ((((((7.6471637318198164759e-13 * i - 3.9851014346715404916e-6) * i + 0.000049255746366361445727) * i - 0.00024947258047043099953) * i + 0.00064513046951456342991) * i - 0.00076245135440323932387) * i + 0.000033530976880017885309) * i + 0.0017438262298340009980
    c[9] = (((((((2.8114572543455207632e-15 * i + 1.0914179173496789432e-6) * i - 0.000015303004486655377567) * i + 0.000090867107935219902229) * i - 0.00029133414466938067350) * i + 0.00051406605788341121363) * i - 0.00036307660358786885787) * i - 0.00031101086326318780412) * i + 0.00096472747321388644237
    c[10] = ((((((((8.2206352466243297170e-18 * i - 3.1239569599829868045e-7) * i + 4.8903045291975346210e-6) * i - 0.000033202652391372058698) * i + 0.00012645437628698076975) * i - 0.00028690924218514613987) * i + 0.00035764655430568632777) * i - 0.00010230378073700412687) * i - 0.00036942667800009661203) * i + 0.00054229262813129686486
    # odd polynomial: v * sum c[2k+1] v^(2k) + sum c[2k] v^(2k)? Boost evaluate_odd_polynomial:
    # c[0] + v * P(v^2) with P coefficients c[1], c[2], ...
    x2 = v * v
    r = c[10]
    for k in range(9, 0, -1):
        r = r * x2 + c[k]
    return c[0] + v * r

def _inverse_students_t(df: float, u: float, v: float, exact: List[bool]) -> float:
    invert = False
    if u > v:
        u, v = v, u
        invert = True
    result = 0.0
    done = False
    if floor(df) == df and df < 20:
        idf = int(df)
        if idf == 1:
            result = 0.0 if u == 0.5 else -cos(pi * u) / sin(pi * u)
            exact[0] = True
            done = True
        elif idf == 2:
            result = (2 * u - 1) / sqrt(2 * u * v)
            exact[0] = True
            done = True
        elif idf == 4:
            alpha = 4 * u * v
            ra = sqrt(alpha)
            r = 4 * cos(acos(ra) / 3) / ra
            x = sqrt(r - 4)
            result = -x if u - 0.5 < 0 else x
            exact[0] = True
            done = True
        elif idf == 6:
            if u < 1e-150:
                return (-1.0 if invert else 1.0) * _st_hill(df, u)
            a = 4 * (u - u * u)
            b = cbrt(a)
            cc = 0.85498797333834849467655443627193
            pp = 6 * (1 + cc * (1 / b - 1))
            tol = ldexp(1.0, i32(-((2 * 53) // 3)))
            while True:
                p2 = pp * pp
                p4 = p2 * p2
                p5 = pp * p4
                p0 = pp
                pp = 2 * (8 * a * p5 - 270 * p2 + 2187) / (5 * (4 * a * p4 - 216 * pp - 243))
                if not (fabs((pp - p0) / pp) > tol):
                    break
            pp = sqrt(pp - df)
            result = -pp if (u - 0.5) < 0 else pp
            done = True
    if not done:
        if df > 268435456.0:
            result = ndtri(u)
            if df >= 1e20:
                exact[0] = True
        elif df < 3:
            crossover = 0.2742 - df * 0.0242143
            if u > crossover:
                result = _st_body_series(df, u)
            else:
                result = _st_tail_series(df, u)
        else:
            # frexp(u) exponent < df / 0.654  (Boost as bundled in scipy 1.18)
            ue = int(floor(log2(u))) + 1 if u > 0 else 0
            if u > 0 and ue < df / 0.654:
                result = _st_hill(df, u)
            else:
                result = _st_tail_series(df, u)
    return -result if invert else result

def _stirling_corr(x: float) -> float:
    # lgamma(x) - [(x - 0.5) log x - x + 0.5 log(2 pi)] for x >= 10
    x2 = 1.0 / (x * x)
    return (1.0 / 12 - x2 * (1.0 / 360 - x2 * (1.0 / 1260 - x2 * (1.0 / 1680 - x2 / 1188)))) / x

def _log_gamma_ratio(a: float, b: float) -> float:
    # log(Gamma(a + b) / Gamma(a)) without cancellation for large a
    if a < 10.0 or a + b < 10.0:
        return gammaln(a + b) - gammaln(a)
    return (a - 0.5) * log1p(b / a) + b * log(a + b) - b + _stirling_corr(a + b) - _stirling_corr(a)

def _bgrat(a: float, b: float, x: float, y: float) -> float:
    # Boost beta_small_b_large_a_series (normalised, s0=0, mult=1):
    # I_x(a, b) for large a and small b.
    bm1 = b - 1
    t = a + bm1 / 2
    lx = log1p(-y) if y < 0.35 else log(x)
    u = -t * lx
    lh = b * log(u) - u - gammaln(b)
    if lh < -708.0:
        return 0.0
    h = exp(lh)
    prefix = h * exp(_log_gamma_ratio(a, b))
    prefix /= pow(t, b)
    j = gammaincc(b, u) / h
    total = prefix * j
    comp = 0.0
    tnp1 = 1
    lx2 = lx / 2
    lx2 *= lx2
    lxp = 1.0
    t4 = 4 * t * t
    b2n = b
    pcoef = [1.0] + [0.0] * 29
    fact = [1.0] * 64
    for i in range(1, 64):
        fact[i] = fact[i - 1] * i
    for n in range(1, 30):
        tnp1 += 2
        pn = 0.0
        tmp1 = 3
        for m in range(1, n):
            mbn = m * b - n
            pn += mbn * pcoef[n - m] / fact[tmp1]
            tmp1 += 2
        pn /= n
        pn += bm1 / fact[tnp1]
        pcoef[n] = pn
        j = (b2n * (b2n + 1) * j + (u + b2n + 1) * lxp) / t4
        lxp *= lx2
        b2n += 2
        r = prefix * pn * j
        # Kahan-compensated sum
        yk = r - comp
        tk = total + yk
        comp = (tk - total) - yk
        total = tk
        if fabs(r / 2.220446049250313e-16) < fabs(total):
            break
    return total

def _ieee_div(a: float, b: float) -> float:
    if b == 0.0:
        if a == 0.0 or a != a:
            return NAN
        return copysign(INF, a) * copysign(1.0, b)
    return a / b

def stdtrit(df: float, p: float) -> float:
    r = _stdtrit_impl(df, p)
    if isinf(r) and p != 0.0 and p != 1.0:
        # Boost raises overflow_error; scipy's wrapper maps it to +inf
        return INF
    return r

def _stdtrit_impl(df: float, p: float) -> float:
    if not (df > 0) or not (p == p) or p < 0 or p > 1:
        return NAN
    if p == 0.0:
        return -INF
    if p == 1.0:
        return INF
    if isinf(df):
        return ndtri(p)
    if p == 0.5:
        return 0.0
    if df < 2 and floor(df) != df:
        # Boost: inverse incomplete beta path
        prob = 1 - p if p > 0.5 else p
        a0 = df / 2
        # small-x asymptote I_x(a, 1/2) ~ x^a / (a B(a, 1/2)); Boost's ibeta_inv
        # bottoms out at DBL_MIN in the deep tail, which scipy inherits.
        lx = (log(2 * prob) + log(a0) + betaln(a0, 0.5)) / a0
        x = 0.0
        if lx < -46.0:
            x = max(exp(lx), 2.2250738585072014e-308)
        else:
            x = betaincinv(a0, 0.5, 2 * prob)
        y = 1.0 - x
        t = 0.0
        if df * y > MAXNUM * x:
            t = INF
        else:
            t = sqrt(df * y / x)
        return -t if p < 0.5 else t
    invert = False
    if p > 0.5:
        p = 1 - p
        invert = True
    exact = [False]
    t = _inverse_students_t(df, p, 1 - p, exact)
    if t == 0 or exact[0]:
        return -t if invert else t
    # Single Halley step on the incomplete-beta formulation (Boost 1.8x as
    # bundled by scipy 1.18). Products are evaluated in the same order so
    # that underflow in the far tail (which makes scipy return inf) matches.
    t2 = t * t
    xb = df / (df + t2)
    y = t2 / (df + t2)
    a = df / 2
    if xb == 0:
        return -t if invert else t
    # log B(a, 1/2) = lgamma(1/2) - log(Gamma(a + 1/2) / Gamma(a))
    lb = 0.5723649429247001 - _log_gamma_ratio(a, 0.5)
    f0 = 0.0
    if a >= 15.0:
        # I_xb(a, 1/2) == 1 - I_y(1/2, a); cephes loses ~1e-9 here
        f0 = _bgrat(a, 0.5, xb, y)
    elif xb < y:
        f0 = betainc(a, 0.5, xb)
    else:
        f0 = 1.0 - betainc(0.5, a, y)
    # derivative of I_x(a, 1/2): x^(a-1) y^(-1/2) / B(a, 1/2), as power terms / (x y)
    pt = exp(a * log(xb) + 0.5 * log(y) - lb)
    f1 = _ieee_div(pt, y * xb)
    p0 = f0 / 2 - p
    p1 = f1 * sqrt(y * xb * xb * xb / df)
    p2 = t * (df + 1) / (t * t + df)
    t = fabs(t)
    t += _ieee_div(p0, p1 + p0 * p2 / 2)
    return t if invert else -t

def fdtr(dfn: float, dfd: float, x: float) -> float:
    if dfn <= 0 or dfd <= 0 or not (x == x):
        return NAN
    if x < 0:
        return 0.0
    if isinf(x):
        return 1.0
    w = dfn * x
    w = w / (dfd + w)
    return betainc(0.5 * dfn, 0.5 * dfd, w)

def fdtrc(dfn: float, dfd: float, x: float) -> float:
    if dfn <= 0 or dfd <= 0 or not (x == x):
        return NAN
    if x < 0:
        return 1.0
    if isinf(x):
        return 0.0
    w = dfd / (dfd + dfn * x)
    return betainc(0.5 * dfd, 0.5 * dfn, w)

def fdtri(dfn: float, dfd: float, y: float) -> float:
    if dfn <= 0 or dfd <= 0 or y < 0 or y > 1 or not (y == y):
        return NAN
    w = betaincinv(0.5 * dfn, 0.5 * dfd, y)
    return dfd * w / (dfn * (1.0 - w))

def chdtr(df: float, x: float) -> float:
    if x < 0:
        return 0.0
    return gammainc(df / 2.0, x / 2.0)

def chdtrc(df: float, x: float) -> float:
    if x < 0:
        return 1.0
    return gammaincc(df / 2.0, x / 2.0)

def chdtri(df: float, y: float) -> float:
    return 2.0 * gammainccinv(0.5 * df, y)

def bdtr(k: float, n: float, p: float) -> float:
    fk = floor(k)
    if fk < 0 or p < 0 or p > 1:
        return NAN
    if fk >= n:
        return 1.0
    if fk == 0:
        return (1.0 - p) ** (n - fk)
    return betainc(n - fk, fk + 1.0, 1.0 - p)

# ------------------------------------------------------- elementary helpers

def xlogy(x: float, y: float) -> float:
    if x == 0 and not isnan(y):
        return 0.0
    return x * log(y)

def xlog1py(x: float, y: float) -> float:
    if x == 0 and not isnan(y):
        return 0.0
    return x * log1p(y)

def entr(x: float) -> float:
    if isnan(x):
        return x
    if x > 0:
        return -x * log(x)
    if x == 0:
        return 0.0
    return -INF

def rel_entr(x: float, y: float) -> float:
    if isnan(x) or isnan(y):
        return NAN
    if x > 0 and y > 0:
        return x * log(x / y)
    if x == 0 and y >= 0:
        return 0.0
    return INF

def kl_div(x: float, y: float) -> float:
    if isnan(x) or isnan(y):
        return NAN
    if x > 0 and y > 0:
        return x * log(x / y) - x + y
    if x == 0 and y >= 0:
        return y
    return INF

def expit(x: float) -> float:
    # scipy: 1 / (1 + exp(-x)) evaluated stably
    if x >= 0:
        return 1.0 / (1.0 + exp(-x))
    e = exp(x)
    return e / (1.0 + e)

def logit(x: float) -> float:
    # scipy uses log(x / (1 - x)) with a log1p branch near 1/2
    if x < 0.3 or x > 0.65:
        return log(x.cdiv(1.0 - x))
    s = 2.0 * (x - 0.5)
    return log1p(s) - log1p(-s)

def log_expit(x: float) -> float:
    if x < 0:
        return x - log1p(exp(x))
    return -log1p(exp(-x))

def boxcox(x: float, lmbda: float) -> float:
    if fabs(lmbda) < 1e-19:
        return log(x)
    return expm1(lmbda * log(x)) / lmbda

# -------------------------------------------------------------------- Bessel

_I0A = (-4.41534164647933937950e-18, 3.33079451882223809783e-17, -2.43127984654795469359e-16,
        1.71539128555513303061e-15, -1.16853328779934516808e-14, 7.67618549860493561688e-14,
        -4.85644678311192946090e-13, 2.95505266312963983461e-12, -1.72682629144155570723e-11,
        9.67580903537323691224e-11, -5.18979560163526290666e-10, 2.65982372468238665035e-9,
        -1.30002500998624804212e-8, 6.04699502254191894932e-8, -2.67079385394061173391e-7,
        1.11738753912010371815e-6, -4.41673835845875056359e-6, 1.64484480707288970893e-5,
        -5.75419501008210370398e-5, 1.88502885095841655729e-4, -5.76375574538582365885e-4,
        1.63947561694133579842e-3, -4.32430999505057594430e-3, 1.05464603945949983183e-2,
        -2.37374148058994688156e-2, 4.93052842396707084878e-2, -9.49010970480476444210e-2,
        1.71620901522208775349e-1, -3.04682672343198398683e-1, 6.76795274409476084995e-1)
_I0B = (-7.23318048787475395456e-18, -4.83050448594418207126e-18, 4.46562142029675999901e-17,
        3.46122286769746109310e-17, -2.82762398051658348494e-16, -3.42548561967721913462e-16,
        1.77256013305652638360e-15, 3.81168066935262242075e-15, -9.55484669882830764870e-15,
        -4.15056934728722208663e-14, 1.54008621752140982691e-14, 3.85277838274214270114e-13,
        7.18012445138366623367e-13, -1.79417853150680611778e-12, -1.32158118404477131188e-11,
        -3.14991652796324136454e-11, 1.18891471078464383424e-11, 4.94060238822496958910e-10,
        3.39623202570838634515e-9, 2.26666899049817806459e-8, 2.04891858946906374183e-7,
        2.89137052083475648297e-6, 6.88975834691682398426e-5, 3.36911647825569408990e-3,
        8.04490411014108831608e-1)
_I1A = (2.77791411276104639959e-18, -2.11142121435816608115e-17, 1.55363195773620046921e-16,
        -1.10559694773538630805e-15, 7.60068429473540693410e-15, -5.04218550472791168711e-14,
        3.22379336594557470981e-13, -1.98397439776494371520e-12, 1.17361862988909016308e-11,
        -6.66348972350202774223e-11, 3.62559028155211703701e-10, -1.88724975172282928790e-9,
        9.38153738649577178388e-9, -4.44505912879632808065e-8, 2.00329475355213526229e-7,
        -8.56872026469545474066e-7, 3.47025130813767847674e-6, -1.32731636560394358279e-5,
        4.78156510755005422638e-5, -1.61760815825896745588e-4, 5.12285956168575772895e-4,
        -1.51357245063125314899e-3, 4.15642294431288815669e-3, -1.05640848946261981558e-2,
        2.47264490306265168283e-2, -5.29459812080949914269e-2, 1.02643658689847095384e-1,
        -1.76416518357834055153e-1, 2.52587186443633654823e-1)
_I1B = (7.51729631084210481353e-18, 4.41434832307170791151e-18, -4.65030536848935832153e-17,
        -3.20952592199342395980e-17, 2.96262899764595013876e-16, 3.30820231092092828324e-16,
        -1.88035477551078244854e-15, -3.81440307243700780478e-15, 1.04202769841288027642e-14,
        4.27244001671195135429e-14, -2.10154184277266431302e-14, -4.08355111109219731823e-13,
        -7.19855177624590851209e-13, 2.03562854414708950722e-12, 1.41258074366137813316e-11,
        3.25260358301548823856e-11, -1.89749581235054123450e-11, -5.58974346219658380687e-10,
        -3.83538038596423702205e-9, -2.63146884688951950684e-8, -2.51223623787020892529e-7,
        -3.88256480887769039346e-6, -1.10588938762623716291e-4, -9.76109749136146840777e-3,
        7.78576235018280120474e-1)

def _chbevl(x: float, c) -> float:
    b0 = c[0]
    b1 = 0.0
    b2 = 0.0
    for i in range(1, len(c)):
        b2 = b1
        b1 = b0
        b0 = x * b1 - b2 + c[i]
    return 0.5 * (b0 - b2)

def i0e(x: float) -> float:
    if x < 0:
        x = -x
    if x <= 8.0:
        y = (x / 2.0) - 2.0
        return _chbevl(y, _I0A)
    return _chbevl(32.0 / x - 2.0, _I0B) / sqrt(x)

def i0(x: float) -> float:
    if x < 0:
        x = -x
    if x <= 8.0:
        y = (x / 2.0) - 2.0
        return exp(x) * _chbevl(y, _I0A)
    return exp(x) * _chbevl(32.0 / x - 2.0, _I0B) / sqrt(x)

def i1e(x: float) -> float:
    z = fabs(x)
    if z <= 8.0:
        y = (z / 2.0) - 2.0
        z = _chbevl(y, _I1A) * z
    else:
        z = _chbevl(32.0 / z - 2.0, _I1B) / sqrt(z)
    if x < 0.0:
        z = -z
    return z

def i1(x: float) -> float:
    z = fabs(x)
    if z <= 8.0:
        y = (z / 2.0) - 2.0
        z = _chbevl(y, _I1A) * z * exp(z)
    else:
        z = exp(z) * _chbevl(32.0 / z - 2.0, _I1B) / sqrt(z)
    if x < 0.0:
        z = -z
    return z
