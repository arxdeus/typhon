# Copyright (C) 2026 the Typhon authors
# Ported from scipy 1.18.1 (BSD-3-Clause); algorithms by Agner Fog (stocc).

# scipy.stats._biasedurn: Fisher's and Wallenius' noncentral hypergeometric
# probability functions (fnchyppr.cpp, wnchyppr.cpp) and the StochasticLib3
# samplers (stoc1.cpp, stoc3.cpp).

import scipy.special._kernels as K
from ._biasedurn_tab import NUM_SDEV, ERF_RES, ERFRES_B, ERFRES_S, ERFRES_N, ERFRES_L

from C import ilogb(float) -> i32
from C import ldexp(float, i32) -> float
from C import fabs(float) -> float
from C import expm1(float) -> float
from C import log1p(float) -> float
from C import exp(float) -> float
from C import log(float) -> float
from C import sqrt(float) -> float
from C import pow(float, float) -> float

LN2 = 0.693147180559945309417
FAK_LEN = 1024
SHAT1 = 2.943035529371538573
SHAT2 = 0.8989161620588987408

def _mk_fac_table() -> List[float]:
    t = [0.0 for _ in range(FAK_LEN)]
    s = 0.0
    for i in range(1, FAK_LEN):
        s += log(float(i))
        t[i] = s
    return t

_FAC_TABLE = _mk_fac_table()

def LnFac(n: int) -> float:
    if n < FAK_LEN:
        if n <= 1:
            if n < 0:
                raise ValueError("Parameter negative in LnFac function")
            return 0.0
        return _FAC_TABLE[n]
    n1 = float(n)
    r = 1. / n1
    return (n1 + 0.5) * log(n1) - n1 + 0.918938533204672722 + r * (1. / 12. + r * r * (-1. / 360.))

def LnFacr(x0: float) -> float:
    x = x0
    ix = int(x)
    if x == float(ix):
        return LnFac(ix)
    D = 1.0
    if x < 6.:
        if x == 0 or x == 1:
            return 0.0
        while x < 6:
            x += 1.0
            D *= x
    r = 1. / x
    r2 = r * r
    f = (x + 0.5) * log(x) - x + 0.918938533204672722 + \
        r * (1. / 12. + r2 * (-1. / 360. + r2 * (1. / 1260. + r2 * (-1. / 1680.))))
    if D != 1.:
        f -= log(D)
    return f

def FallingFactorial(a0: float, b: float) -> float:
    a = a0
    if b < 30 and float(int(b)) == b and a < 1E10:
        f = 1.0
        i = 0
        while float(i) < b:
            f *= a
            a -= 1.0
            i += 1
        return log(f)
    if a > 100. * b and b > 1.:
        ar = 1. / a
        cr = 1. / (a - b)
        s = 0.0
        lasts = 0.0
        n = 1.0
        ba = b * ar
        f = ba
        while True:
            lasts = s
            s += f / n
            f *= ba
            n += 1
            if s == lasts:
                break
        return (a + 0.5) * s + b * log(a - b) - b + (1. / 12.) * (ar - cr)
    return LnFacr(a) - LnFacr(a - b)

def _erf(x: float) -> float:
    rsqrtpi = 0.564189583547756286948
    rsqrtpi2 = 1.12837916709551257390
    if x < 0.:
        return -_erf(-x)
    if x > 6.:
        return 1.0
    if x < 2.4:
        s = 0.0
        xx2 = x * x * 2.
        term = x
        j21 = 1.0
        for j in range(51):
            s += term
            if term <= 1.E-13:
                break
            j21 += 2.
            term *= xx2 / j21
        return exp(-x * x) * s * rsqrtpi2
    # float32 coefficients as in the C source
    n = int(2.25 * x * x - 23.399999618530273 * x + 60.84000015258789)
    if n < 1:
        n = 1
    a = 0.5 * n
    f = x
    while n > 0:
        f = x + a / f
        a -= 0.5
        n -= 1
    return 1. - exp(-x * x) * rsqrtpi / f

def _floor_log2_f32(x: float) -> int:
    # FloorLog2((float)x): exponent of the float32 value
    if x <= 0.:
        return -127
    return int(ilogb(float(f32(x))))

def NumSD(accuracy: float) -> int:
    fract = [2.699796e-03, 4.652582e-04, 6.334248e-05, 6.795346e-06, 5.733031e-07,
             3.797912e-08, 1.973175e-09, 8.032001e-11, 2.559625e-12, 6.381783e-14]
    i = 0
    while i < 10:
        if accuracy >= fract[i]:
            break
        i += 1
    return i + 6

def pow2_1(q0: float):
    # returns (1 - 2^q, 2^q)
    q = q0 * LN2
    y = 0.0
    y1 = 0.0
    if fabs(q) > 0.1:
        y = exp(q)
        y1 = 1. - y
    else:
        y1 = expm1(q)
        y = y1 + 1
        y1 = -y1
    return (y1, y)

def log1mx(x: float, x1: float) -> float:
    if fabs(x) > 0.03:
        return log(x1)
    return log1p(-x)

def log1pow(q: float, x: float) -> float:
    y = 0.0
    y1 = 0.0
    if fabs(q) > 0.1:
        y = exp(q)
        y1 = 1. - y
    else:
        y1 = expm1(q)
        y = y1 + 1
        y1 = -y1
    if y > 0.1:
        return x * log(y1)
    return x * log1p(-y)

# ------------------------------------------------------------- Fisher

class FishersNCHypergeometric:
    n: int
    m: int
    N: int
    odds: float
    accuracy: float
    logodds: float
    scale: float
    rsum: float
    xmin: int
    xmax: int
    parameters_changed: bool
    mFac: float
    xFac: float
    xLast: int

    def __init__(self, n: int, m: int, N: int, odds: float, accuracy: float = 1E-8):
        if n < 0 or m < 0 or N < 0 or odds < 0. or n > N or m > N:
            raise RuntimeError("Parameter out of range in class CFishersNCHypergeometric")
        self.n = n
        self.m = m
        self.N = N
        self.odds = odds
        acc = accuracy
        if acc < 0:
            acc = 0.0
        if acc > 1:
            acc = 1.0
        self.accuracy = acc
        self.logodds = log(odds) if odds > 0 else -K.INF
        self.scale = 0.0
        self.rsum = 0.0
        self.parameters_changed = True
        self.mFac = 0.0
        self.xFac = 0.0
        self.xLast = -99
        xmin = m + n - N
        if xmin < 0:
            xmin = 0
        xmax = n
        if xmax > m:
            xmax = m
        self.xmin = xmin
        self.xmax = xmax

    def mode(self) -> int:
        L = self.m + self.n - self.N
        m1 = self.m + 1
        n1 = self.n + 1
        x = 0.0
        if self.odds == 1.:
            x = (self.m + 1.) * (self.n + 1.) / (self.N + 2.)
        else:
            A = 1. - self.odds
            B = (m1 + n1) * self.odds - L
            C = -float(m1) * n1 * self.odds
            D = B * B - 4 * A * C
            D = sqrt(D) if D > 0. else 0.
            x = (D - B) / (A + A)
        return int(x)

    def mean(self) -> float:
        m = self.m
        n = self.n
        N = self.N
        odds = self.odds
        if odds == 1.:
            return float(m) * n / N
        a = (m + n) * odds + (N - m - n)
        b = a * a - 4. * odds * (odds - 1.) * m * n
        b = sqrt(b) if b > 0. else 0.
        return (a - b) / (2. * (odds - 1.))

    def variance(self) -> float:
        m = self.m
        n = self.n
        N = self.N
        my = self.mean()
        r1 = my * (m - my)
        r2 = (n - my) * (my + N - n - m)
        if r1 <= 0. or r2 <= 0.:
            return 0.0
        var = N * r1 * r2 / ((N - 1) * (m * r2 + (N - m) * r1))
        if var < 0.:
            var = 0.0
        return var

    def moments(self):
        sy = 0.0
        sxy = 0.0
        sxxy = 0.0
        accur = 0.1 * self.accuracy
        xm = int(self.mean())
        x = xm
        while x <= self.xmax:
            y = self.probability(x)
            x1 = float(x - xm)
            sy += y
            sxy += x1 * y
            sxxy += x1 * x1 * y
            if y < accur and x != xm:
                break
            x += 1
        x = xm - 1
        while x >= self.xmin:
            y = self.probability(x)
            x1 = float(x - xm)
            sy += y
            sxy += x1 * y
            sxxy += x1 * x1 * y
            if y < accur:
                break
            x -= 1
        me1 = sxy / sy
        mean = me1 + xm
        y = sxxy / sy - me1 * me1
        if y < 0:
            y = 0.0
        return (mean, y)

    def probability(self, x: int) -> float:
        accur = self.accuracy * 0.1
        n = self.n
        m = self.m
        N = self.N
        if x < self.xmin or x > self.xmax:
            return 0.0
        if n == 0:
            return 1.0
        if self.odds == 1.:
            return exp(LnFac(m) - LnFac(x) - LnFac(m - x) +
                       LnFac(N - m) - LnFac(n - x) - LnFac((N - m) - (n - x)) -
                       (LnFac(N) - LnFac(n) - LnFac(N - n)))
        if self.odds == 0.:
            if n > N - m:
                raise RuntimeError("Not enough items with nonzero weight in CFishersNCHypergeometric::probability")
            return 1.0 if x == 0 else 0.0
        if not self.rsum:
            x1 = int(self.mean())
            if x1 < self.xmin:
                x1 = self.xmin
            x2 = x1 + 1
            self.scale = 0.0
            self.scale = self.lng(x1)
            self.rsum = 1.0
            x1 -= 1
            while x1 >= self.xmin:
                y = exp(self.lng(x1))
                self.rsum += y
                if y < accur:
                    break
                x1 -= 1
            while x2 <= self.xmax:
                y = exp(self.lng(x2))
                self.rsum += y
                if y < accur:
                    break
                x2 += 1
            self.rsum = 1. / self.rsum
        return exp(self.lng(x)) * self.rsum

    def lng(self, x: int) -> float:
        n = self.n
        m = self.m
        x2 = n - x
        m2 = self.N - m
        if self.parameters_changed:
            self.mFac = LnFac(m) + LnFac(m2)
            self.xLast = -99
            self.parameters_changed = False
        d = x - self.xLast
        if m < FAK_LEN and m2 < FAK_LEN:
            self.xFac = LnFac(x) + LnFac(x2) + LnFac(m - x) + LnFac(m2 - x2)
        elif d == 0:
            pass
        elif d == 1:
            self.xFac += log(float(x) * (m2 - x2) / (float(x2 + 1) * (m - x + 1)))
        elif d == -1:
            self.xFac += log(float(x2) * (m - x) / (float(x + 1) * (m2 - x2 + 1)))
        else:
            self.xFac = LnFac(x) + LnFac(x2) + LnFac(m - x) + LnFac(m2 - x2)
        self.xLast = x
        return self.mFac - self.xFac + x * self.logodds - self.scale

# ----------------------------------------------------------- Wallenius

_XVAL8 = [-0.960289856498, -0.796666477414, -0.525532409916, -0.183434642496,
          0.183434642496, 0.525532409916, 0.796666477414, 0.960289856498]
_WEIGHTS8 = [0.10122853629, 0.222381034453, 0.313706645878, 0.362683783378,
             0.362683783378, 0.313706645878, 0.222381034453, 0.10122853629]

class WalleniusNCHypergeometric:
    n: int
    m: int
    N: int
    x: int
    omega: float
    accuracy: float
    xmin: int
    xmax: int
    xLastBico: int
    xLastFindpars: int
    bico: float
    mFac: float
    xFac: float
    r: float
    rd: float
    w: float
    wr: float
    E: float
    phi2d: float

    def __init__(self, n: int, m: int, N: int, odds: float, accuracy: float = 1E-8):
        self.accuracy = accuracy
        self.n = n
        self.m = m
        self.N = N
        self.omega = odds
        self.x = 0
        self.xmin = 0
        self.xmax = 0
        self.xLastBico = -99
        self.xLastFindpars = -99
        self.bico = 0.0
        self.mFac = 0.0
        self.xFac = 0.0
        self.r = 1.0
        self.rd = 0.0
        self.w = 0.0
        self.wr = 0.0
        self.E = 0.0
        self.phi2d = 0.0
        self.set_parameters(n, m, N, odds)

    def set_parameters(self, n: int, m: int, N: int, odds: float):
        if n < 0 or n > N or m < 0 or m > N or odds < 0:
            raise RuntimeError("Parameter out of range in CWalleniusNCHypergeometric")
        self.n = n
        self.m = m
        self.N = N
        self.omega = odds
        xmin = m + n - N
        if xmin < 0:
            xmin = 0
        xmax = n
        if xmax > m:
            xmax = m
        self.xmin = xmin
        self.xmax = xmax
        self.xLastBico = -99
        self.xLastFindpars = -99
        self.r = 1.0

    def mean(self) -> float:
        n = self.n
        m = self.m
        N = self.N
        omega = self.omega
        if omega == 1.:
            return float(m) * n / N
        if omega == 0.:
            if n > N - m:
                raise RuntimeError("Not enough items with nonzero weight in CWalleniusNCHypergeometric::mean")
            return 0.0
        if self.xmin == self.xmax:
            return float(self.xmin)
        a = (m + n) * omega + (N - m - n)
        b = a * a - 4. * omega * (omega - 1.) * m * n
        b = sqrt(b) if b > 0. else 0.
        mean = (a - b) / (2. * (omega - 1.))
        if mean < self.xmin:
            mean = float(self.xmin)
        if mean > self.xmax:
            mean = float(self.xmax)
        m1r = 1. / m
        m2r = 1. / (N - m)
        it = 0
        if omega > 1.:
            while True:
                mean1 = mean
                e1 = 1. - (n - mean) * m2r
                e2 = 0.0
                if e1 < 1E-14:
                    e2 = 0.0
                else:
                    e2 = pow(e1, omega - 1.)
                g = e2 * e1 + (mean - m) * m1r
                gd = e2 * omega * m2r + m1r
                mean -= g / gd
                if mean < self.xmin:
                    mean = float(self.xmin)
                if mean > self.xmax:
                    mean = float(self.xmax)
                it += 1
                if it > 40:
                    raise RuntimeError("Search for mean failed in function CWalleniusNCHypergeometric::mean")
                if not (fabs(mean1 - mean) > 2E-6):
                    break
        else:
            omegar = 1. / omega
            while True:
                mean1 = mean
                e1 = 1. - mean * m1r
                e2 = 0.0
                if e1 < 1E-14:
                    e2 = 0.0
                else:
                    e2 = pow(e1, omegar - 1.)
                g = 1. - (n - mean) * m2r - e2 * e1
                gd = e2 * omegar * m1r + m2r
                mean -= g / gd
                if mean < self.xmin:
                    mean = float(self.xmin)
                if mean > self.xmax:
                    mean = float(self.xmax)
                it += 1
                if it > 40:
                    raise RuntimeError("Search for mean failed in function CWalleniusNCHypergeometric::mean")
                if not (fabs(mean1 - mean) > 2E-6):
                    break
        return mean

    def variance(self) -> float:
        m = self.m
        n = self.n
        N = self.N
        my = self.mean()
        r1 = my * (m - my)
        r2 = (n - my) * (my + N - n - m)
        if r1 <= 0. or r2 <= 0.:
            return 0.0
        var = N * r1 * r2 / ((N - 1) * (m * r2 + (N - m) * r1))
        if var < 0.:
            var = 0.0
        return var

    def moments(self):
        sy = 0.0
        sxy = 0.0
        sxxy = 0.0
        accuracy = 1.000000013351432e-10
        xm = int(self.mean())
        x = xm
        while x <= self.xmax:
            y = self.probability(x)
            x1 = float(x - xm)
            sy += y
            sxy += x1 * y
            sxxy += x1 * x1 * y
            if y < accuracy and x != xm:
                break
            x += 1
        x = xm - 1
        while x >= self.xmin:
            y = self.probability(x)
            x1 = float(x - xm)
            sy += y
            sxy += x1 * y
            sxxy += x1 * x1 * y
            if y < accuracy:
                break
            x -= 1
        me1 = sxy / sy
        mean = me1 + xm
        y = sxxy / sy - me1 * me1
        if y < 0:
            y = 0.0
        return (mean, y)

    def mode(self) -> int:
        m = self.m
        n = self.n
        N = self.N
        omega = self.omega
        Mode = 0
        if omega == 1.:
            L = m + n - N
            m1 = m + 1
            n1 = n + 1
            Mode = int(float(m1) * n1 * omega / ((m1 + n1) * omega - L))
        else:
            f2 = 0.0
            xmin = m + n - N
            if xmin < 0:
                xmin = 0
            xmax = n
            if xmax > m:
                xmax = m
            Mode = int(self.mean())
            if omega < 1.:
                if Mode < xmax:
                    Mode += 1
                x2 = xmin
                if omega > 0.294 and N <= 10000000:
                    x2 = Mode - 1
                xi = Mode
                while xi >= x2:
                    f = self.probability(xi)
                    if f <= f2:
                        break
                    Mode = xi
                    f2 = f
                    xi -= 1
            else:
                if Mode < xmin:
                    Mode += 1
                x2 = xmax
                if omega < 3.4 and N <= 10000000:
                    x2 = Mode + 1
                xi = Mode
                while xi <= x2:
                    f = self.probability(xi)
                    if f <= f2:
                        break
                    Mode = xi
                    f2 = f
                    xi += 1
        return Mode

    def lnbico(self) -> float:
        n = self.n
        m = self.m
        x = self.x
        x2 = n - x
        m2 = self.N - m
        if self.xLastBico < 0:
            self.mFac = LnFac(m) + LnFac(m2)
        d = x - self.xLastBico
        if m < FAK_LEN and m2 < FAK_LEN:
            self.xFac = LnFac(x) + LnFac(x2) + LnFac(m - x) + LnFac(m2 - x2)
        elif d == 0:
            pass
        elif d == 1:
            self.xFac += log(float(x) * (m2 - x2) / (float(x2 + 1) * (m - x + 1)))
        elif d == -1:
            self.xFac += log(float(x2) * (m - x) / (float(x + 1) * (m2 - x2 + 1)))
        else:
            self.xFac = LnFac(x) + LnFac(x2) + LnFac(m - x) + LnFac(m2 - x2)
        self.xLastBico = x
        self.bico = self.mFac - self.xFac
        return self.bico

    def findpars(self):
        if self.x == self.xLastFindpars:
            return
        x = self.x
        n = self.n
        m = self.m
        N = self.N
        omega = self.omega
        xx0 = float(x)
        xx1 = float(n - x)
        oo0 = omega
        oo1 = 1.0
        if omega > 1.:
            oo0 = 1.0
            oo1 = 1. / omega
        dd = oo0 * (m - x) + oo1 * (N - m - xx1)
        d1 = 1. / dd
        self.E = (oo0 * m + oo1 * (N - m)) * d1
        rr = self.r
        if rr <= d1:
            rr = 1.2 * d1
        j = 0
        lastr = rr
        while True:
            lastr = rr
            rrc = 1. / rr
            z = dd - rrc
            zd = rrc * rrc
            for i in range(2):
                oi = oo0 if i == 0 else oo1
                xi = xx0 if i == 0 else xx1
                rt = rr * oi
                if rt < 100.:
                    p = pow2_1(rt)
                    r21 = p[0]
                    r2 = p[1]
                    a = oi / r21
                    b = xi * a
                    z += b
                    zd += b * a * LN2 * r2
            if zd == 0:
                raise RuntimeError("can't find r in function CWalleniusNCHypergeometric::findpars")
            rr -= z / zd
            if rr <= d1:
                rr = lastr * 0.125 + d1 * 0.875
            j += 1
            if j == 70:
                raise RuntimeError("convergence problem searching for r in function CWalleniusNCHypergeometric::findpars")
            if not (fabs(rr - lastr) > rr * 1.E-6):
                break
        if omega > 1:
            dd *= omega
            rr *= oo1
        self.r = rr
        self.rd = rr * dd
        ro = self.r * omega
        k1 = 0.0
        k2 = 0.0
        if ro < 300:
            k1 = pow2_1(ro)[0]
            k1 = -1. / k1
            k1 = omega * omega * (k1 + k1 * k1)
        if self.r < 300:
            k2 = pow2_1(self.r)[0]
            k2 = -1. / k2
            k2 = k2 + k2 * k2
        self.phi2d = -4. * self.r * self.r * (x * k1 + (n - x) * k2)
        if self.phi2d >= 0.:
            raise RuntimeError("peak width undefined in function CWalleniusNCHypergeometric::findpars")
        self.wr = sqrt(-self.phi2d)
        self.w = 1. / self.wr
        self.xLastFindpars = x

    def bernouilli_h(self, x_: int, h: float, rh: float, sto):
        rsqrt8 = 0.3535533905932737622
        sqrt2pi = 2.506628274631000454
        n = self.n
        omega = self.omega
        xi0 = float(x_)
        xi1 = float(n - x_)
        self.x = x_
        self.lnbico()
        self.findpars()
        k = 1.0
        if self.E > 0.:
            k = log(self.E)
            k = 1. + 0.0271 * (k * sqrt(k))
        k *= self.w
        rdm1 = self.rd - 1.
        phideri0 = -LN2 * rdm1
        for i in range(2):
            om = omega if i == 0 else 1.0
            xi = xi0 if i == 0 else xi1
            romegi = self.r * om
            qi = 0.0
            qi1 = 1.0
            if romegi > 40.:
                qi = 0.0
                qi1 = 1.0
            else:
                p = pow2_1(-romegi)
                qi1 = p[0]
                qi = p[1]
            phideri0 += xi * log1mx(qi, qi1)
        erfk = _erf(rsqrt8 / k)
        f0 = self.rd * exp(phideri0 + self.bico)
        G_integral = f0 * sqrt2pi * k * erfk
        if G_integral <= h:
            ts = 0.0
            while True:
                ts = sto.Normal(0.0, k)
                if not (fabs(ts) >= 0.5):
                    break
            ts += 0.5
            fts = 0.0
            for j in range(2):
                logts = log(ts)
                rlogts = self.r * logts
                fts += exp(log1pow(rlogts * omega, xi0) + log1pow(rlogts, xi1) + rdm1 * logts + self.bico)
                ts = 1. - ts
            fts *= 0.5
            t2 = (ts - 0.5) / k
            rgts = exp(-(phideri0 + self.bico - 0.5 * t2 * t2))
            return rh < G_integral * fts * rgts
        return rh < self.probability(x_)

    def recursive(self) -> float:
        BUFSIZE = 512
        n = self.n
        m = self.m
        N = self.N
        x = self.x
        omega = self.omega
        accuracya = 0.004999999888241291 * self.accuracy
        # two buffers play the role of p1/p2 with offsets (p array of BUFSIZE+2)
        p = [0.0 for _ in range(BUFSIZE + 2)]
        p1 = 1
        p2 = 1
        p[p1 - 1] = 0.0
        p[p1] = 1.0
        x1 = 0
        x2 = 0
        for nu in range(1, n + 1):
            if n - nu < x - x1 or p[p1 + x1] < accuracya:
                x1 += 1
                p2 -= 1
            y1 = 0.0
            if x2 < x and p[p1 + x2] >= accuracya:
                x2 += 1
                y1 = 0.0
            else:
                y1 = p[p1 + x2]
            if x1 > x2:
                return 0.0
            if p2 + x2 > BUFSIZE:
                raise RuntimeError("buffer overrun in function CWalleniusNCHypergeometric::recursive")
            mxo = (m - x2) * omega
            Nmnx = float(N - m - nu + x2 + 1)
            xi = x2
            while xi >= x1:
                d2 = mxo + Nmnx
                mxo += omega
                Nmnx -= 1
                d1 = mxo + Nmnx
                dcom = 1. / (d1 * d2)
                y = p[p1 + xi - 1] * mxo * d2 * dcom + y1 * (Nmnx + 1) * d1 * dcom
                y1 = p[p1 + xi - 1]
                p[p2 + xi] = y
                xi -= 1
            p1 = p2
        if x < x1 or x > x2:
            return 0.0
        return p[p1 + x]

    def binoexpand(self) -> float:
        x = self.x
        n = self.n
        x1 = 0
        m1 = 0
        m2 = 0
        o = 0.0
        if x > n // 2:
            x1 = n - x
            m1 = self.N - self.m
            m2 = self.m
            o = 1. / self.omega
        else:
            x1 = x
            m1 = self.m
            m2 = self.N - self.m
            o = self.omega
        if x1 == 0:
            return exp(FallingFactorial(float(m2), float(n)) - FallingFactorial(m2 + o * m1, float(n)))
        if x1 == 1:
            q = FallingFactorial(float(m2), float(n - 1))
            e = o * m1 + m2
            q1 = q - FallingFactorial(e, float(n))
            e -= o
            q0 = q - FallingFactorial(e, float(n))
            d = e - (n - 1)
            return m1 * d * (exp(q0) - exp(q1))
        raise RuntimeError("x > 1 not supported by function CWalleniusNCHypergeometric::binoexpand")

    def laplace(self) -> float:
        MAXDEG = 40
        sqrt8 = 2.828427124746190098
        x = self.x
        n = self.n
        omega = self.omega
        xi = [float(x), float(n - x)]
        omegai = [omega, 1.0]
        rho = [0.0, 0.0]
        qq = [0.0, 0.0]
        eta = [[0.0 for _ in range(MAXDEG + 1)] for _ in range(3)]
        phideri = [0.0 for _ in range(MAXDEG + 1)]
        PSIderi = [0.0 for _ in range(MAXDEG + 1)]
        for k in range(3):
            phideri[k] = 0.0
            PSIderi[k] = 0.0
        for i in range(2):
            rho[i] = self.r * omegai[i]
            qi = 0.0
            qi1 = 1.0
            if rho[i] > 40.:
                qi = 0.0
                qi1 = 1.0
            else:
                p = pow2_1(-rho[i])
                qi1 = p[0]
                qi = p[1]
            qq[i] = qi / qi1
            phideri[0] += xi[i] * log1mx(qi, qi1)
            eta[i][0] = 0.
            eta[i][1] = rho[i] * rho[i]
            eta[i][2] = rho[i] * rho[i]
        phideri[0] -= (self.rd - 1.) * LN2
        f0 = self.rd * exp(phideri[0] + self.lnbico())
        vr = sqrt8 * self.w
        phideri[2] = self.phi2d
        pv = -_floor_log2_f32(self.accuracy) - ERFRES_B + ERFRES_S - 1
        pi_ = -((-pv) // ERFRES_S) if pv < 0 else pv // ERFRES_S
        PrecisionIndex = pi_
        if PrecisionIndex < 0:
            PrecisionIndex = 0
        if PrecisionIndex > ERFRES_N - 1:
            PrecisionIndex = ERFRES_N - 1
        while self.w * NUM_SDEV[PrecisionIndex] > 0.3:
            if PrecisionIndex == 0:
                raise RuntimeError("Laplace method failed. Peak width too high in function CWalleniusNCHypergeometric::laplace")
            PrecisionIndex -= 1
        erfresp = ERF_RES[PrecisionIndex]
        degree = MAXDEG
        if degree >= ERFRES_L * 2:
            degree = ERFRES_L * 2 - 2
        v2m2 = 0.25 * vr * vr
        PSIderi[0] = 1.
        pow2k = 8.
        sum_ = 0.5 * vr * erfresp[0]
        v2mk1 = 0.5 * vr * v2m2 * v2m2
        accur = self.accuracy * sum_
        converg = 0
        for k in range(3, degree + 1):
            phideri[k] = 0.
            for i in range(2):
                eta[i][k] = 0.
                j = k
                while j > 0:
                    eta[i][j] = eta[i][j] * (j * rho[i] - (k - 2)) + eta[i][j - 1] * rho[i] * (j - 1)
                    j -= 1
                qqpow = 1.
                for j in range(1, k + 1):
                    qqpow *= qq[i]
                    phideri[k] += xi[i] * eta[i][j] * qqpow
            phideri[k] = -pow2k * phideri[k] + 2 * (1 - k) * phideri[k - 1]
            pow2k *= 2.
            PSIderi[k] = phideri[k]
            bino = 0.5 * (k - 1) * (k - 2)
            for j in range(3, k - 2):
                PSIderi[k] += PSIderi[k - j] * phideri[j] * bino
                bino *= float(k - j) / float(j)
            if (k & 1) == 0:
                ll = k // 2
                s = PSIderi[k] * v2mk1 * erfresp[ll]
                sum_ += s
                if fabs(s) < accur:
                    converg += 1
                else:
                    converg = 0
                if converg > 1:
                    break
                v2mk1 *= v2m2
        return f0 * sum_

    def integrate_step(self, ta: float, tb: float) -> float:
        delta = 0.5 * (tb - ta)
        ab = 0.5 * (ta + tb)
        rdm1 = self.rd - 1.
        s = 0.0
        for i in range(8):
            tau = ab + delta * _XVAL8[i]
            ltau = log(tau)
            taur = self.r * ltau
            y = log1pow(taur * self.omega, float(self.x)) + log1pow(taur, float(self.n - self.x)) + rdm1 * ltau + self.bico
            if y > -50.:
                s += _WEIGHTS8[i] * exp(y)
        return delta * s

    def search_inflect(self, t_from0: float, t_to0: float) -> float:
        t_from = t_from0
        t_to = t_to0
        rdm1 = self.rd - 1.
        if t_from == 0 and rdm1 <= 1.:
            return 0.0
        rho = [self.r * self.omega, self.r]
        xx = [float(self.x), float(self.n - self.x)]
        t = 0.5 * (t_from + t_to)
        # zeta[i][j][k]
        zeta = [[[0.0 for _ in range(4)] for _ in range(4)] for _ in range(2)]
        for i in range(2):
            zeta[i][1][1] = rho[i]
            zeta[i][1][2] = rho[i] * (rho[i] - 1.)
            zeta[i][2][2] = rho[i] * rho[i]
            zeta[i][1][3] = zeta[i][1][2] * (rho[i] - 2.)
            zeta[i][2][3] = zeta[i][1][2] * rho[i] * 3.
            zeta[i][3][3] = zeta[i][2][2] * rho[i] * 2.
        it = 0
        t1 = t
        while True:
            t1 = t
            tr = 1. / t
            log2t = log(t) * (1. / LN2)
            phi1 = 0.0
            phi2 = 0.0
            phi3 = 0.0
            for i in range(2):
                pp = pow2_1(rho[i] * log2t)
                q1 = pp[0]
                q = pp[1] / q1
                phi1 -= xx[i] * zeta[i][1][1] * q
                phi2 -= xx[i] * q * (zeta[i][1][2] + q * zeta[i][2][2])
                phi3 -= xx[i] * q * (zeta[i][1][3] + q * (zeta[i][2][3] + q * zeta[i][3][3]))
            phi1 += rdm1
            phi2 -= rdm1
            phi3 += 2. * rdm1
            phi1 *= tr
            phi2 *= tr * tr
            phi3 *= tr * tr * tr
            method = float((it & 2) >> 1)
            Z2 = phi1 * phi1 + phi2
            Zd = method * phi1 * phi1 * phi1 + (2. + method) * phi1 * phi2 + phi3
            if t < 0.5:
                if Z2 > 0:
                    t_from = t
                else:
                    t_to = t
                if Zd >= 0:
                    t = (0.5 if t_from != 0 else 0.2) * (t_from + t_to)
                else:
                    t -= Z2 / Zd
            else:
                if Z2 < 0:
                    t_from = t
                else:
                    t_to = t
                if Zd <= 0:
                    t = 0.5 * (t_from + t_to)
                else:
                    t -= Z2 / Zd
            if t >= t_to:
                t = (t1 + t_to) * 0.5
            if t <= t_from:
                t = (t1 + t_from) * 0.5
            it += 1
            if it > 20:
                raise RuntimeError("Search for inflection point failed in function CWalleniusNCHypergeometric::search_inflect")
            if not (fabs(t - t1) > 1E-5):
                break
        return t

    def integrate(self) -> float:
        self.lnbico()
        accuracy = self.accuracy
        w = self.w
        sum_ = 0.0
        ta = 0.0
        tb = 0.0
        if w < 0.02 or (w < 0.1 and (self.x == self.m or self.n - self.x == self.N - self.m) and accuracy > 1E-6):
            s1 = 0.5 if accuracy < 1E-9 else 1.0
            delta = s1 * w
            ta = 0.5 + 0.5 * delta
            sum_ = self.integrate_step(1. - ta, ta)
            while True:
                tb = ta + delta
                if tb > 1.:
                    tb = 1.0
                s = self.integrate_step(ta, tb)
                s += self.integrate_step(1. - tb, 1. - ta)
                sum_ += s
                if s < accuracy * sum_:
                    break
                ta = tb
                if tb > 0.5 + w:
                    delta *= 2.
                if not (tb < 1.):
                    break
        else:
            t1 = 0.0
            t2 = 0.5
            while t1 < 1.:
                tinf = self.search_inflect(t1, t2)
                delta = tinf - t1
                if delta > t2 - tinf:
                    delta = t2 - tinf
                delta *= 1. / 7.
                if delta < 1E-4:
                    delta = 1E-4
                delta1 = delta
                ta = tinf
                while True:
                    tb = ta + delta1
                    if tb > t2 - 0.25 * delta1:
                        tb = t2
                    s = self.integrate_step(ta, tb)
                    sum_ += s
                    delta1 *= 2
                    if s < sum_ * 1E-4:
                        delta1 *= 8.
                    ta = tb
                    if not (tb < t2):
                        break
                if tinf != 0:
                    tb = tinf
                    while True:
                        ta = tb - delta
                        if ta < t1 + 0.25 * delta:
                            ta = t1
                        s = self.integrate_step(ta, tb)
                        sum_ += s
                        delta *= 2
                        if s < sum_ * 1E-4:
                            delta *= 8.
                        tb = ta
                        if not (ta > t1):
                            break
                t1 += 0.5
                t2 += 0.5
        return sum_ * self.rd

    def probability(self, x_: int) -> float:
        self.x = x_
        x = x_
        n = self.n
        m = self.m
        N = self.N
        if x < self.xmin or x > self.xmax:
            return 0.0
        if self.xmin == self.xmax:
            return 1.0
        if self.omega == 1.:
            return exp(self.lnbico() + LnFac(n) + LnFac(N - n) - LnFac(N))
        if self.omega == 0.:
            if n > N - m:
                raise RuntimeError("Not enough items with nonzero weight in CWalleniusNCHypergeometric::probability")
            return 1.0 if x == 0 else 0.0
        x2 = n - x
        x0 = x if x < x2 else x2
        em = (x == m or x2 == N - m)
        if x0 == 0 and n > 500:
            return self.binoexpand()
        if float(n) * x0 < 1000 or (float(n) * x0 < 10000 and (N > 1000. * n or em)):
            return self.recursive()
        if x0 <= 1 and N - n <= 1:
            return self.binoexpand()
        self.findpars()
        if self.w < 0.04 and self.E < 10 and ((not em) or self.w > 0.004):
            return self.laplace()
        return self.integrate()

    def make_table(self, max_length: int):
        # CWalleniusNCHypergeometric::MakeTable (table method and one-by-one);
        # returns (success, xfirst, xlast, table)
        n = self.n
        m = self.m
        N = self.N
        omega = self.omega
        cutoff = 0.01 * self.accuracy
        table = [0.0 for _ in range(max_length + 2)]
        if n == 0 or m == 0:
            table[0] = 1.0
            return (1, 0, 0, table)
        if n == N:
            table[0] = 1.0
            return (1, m, m, table)
        if m == N:
            table[0] = 1.0
            return (1, n, n, table)
        if omega <= 0.:
            if n > N - m:
                raise RuntimeError("Not enough items with nonzero weight in  CWalleniusNCHypergeometric::MakeTable")
            table[0] = 1.0
            return (1, 0, 0, table)
        LengthNeeded = N - m
        if m < LengthNeeded:
            LengthNeeded = m
        if n < LengthNeeded:
            LengthNeeded = n
        area = float(n) * LengthNeeded
        UseTable = area < 5000. or (area < 10000. and N > 1000. * n)
        xmin = self.xmin
        xmax = self.xmax
        if UseTable and max_length > LengthNeeded:
            p1 = 1
            p2 = 1
            table[p1 - 1] = 0.0
            table[p1] = 1.0
            x1 = 0
            x2 = 0
            failed = False
            for nu in range(1, n + 1):
                if n - nu < xmin - x1 or table[p1 + x1] < cutoff:
                    x1 += 1
                    p2 -= 1
                y1 = 0.0
                if x2 < xmax and table[p1 + x2] >= cutoff:
                    x2 += 1
                    y1 = 0.0
                else:
                    y1 = table[p1 + x2]
                if p2 + x2 >= max_length or x1 > x2:
                    failed = True
                    break
                mxo = (m - x2) * omega
                Nmnx = float(N - m - nu + x2 + 1)
                xi = x2
                while xi >= x1:
                    d2 = mxo + Nmnx
                    mxo += omega
                    Nmnx -= 1
                    d1 = mxo + Nmnx
                    dcom = 1. / (d1 * d2)
                    y = table[p1 + xi - 1] * mxo * d2 * dcom + y1 * (Nmnx + 1) * d1 * dcom
                    y1 = table[p1 + xi - 1]
                    table[p2 + xi] = y
                    xi -= 1
                p1 = p2
            if not failed:
                i1 = x2 - x1 + 1
                i2 = i1
                if i2 > max_length:
                    i2 = max_length
                out = [0.0 for _ in range(max_length + 2)]
                for k in range(i2):
                    out[k] = table[1 + k]
                return (1 if i1 == i2 else 0, x1, x1 + i2 - 1, out)
        # one by one
        table = [0.0 for _ in range(max_length + 2)]
        x2b = int(self.mean())
        x1b = x2b + 1
        i1b = max_length
        while x1b > xmin:
            x1b -= 1
            i1b -= 1
            y = self.probability(x1b)
            table[i1b] = y
            if y < cutoff:
                break
            if i1b == 0:
                break
        xfirst = x1b
        i2b = x2b - x1b + 1
        if i1b > 0 and i2b > 0:
            for k in range(i2b):
                table[k] = table[i1b + k]
        i2b -= 1
        while x2b < xmax:
            if i2b == max_length - 1:
                return (0, xfirst, x2b, table)
            x2b += 1
            i2b += 1
            y = self.probability(x2b)
            table[i2b] = y
            if y < cutoff:
                break
        return (1, xfirst, x2b, table)

# ---------------------------------------------------------- sampler lib

def _unif(rs) -> float:
    # bit generator next_double
    if hasattr(rs, '_bitgen'):
        return rs._bitgen.next_double()
    else:
        return rs.bit_generator.next_double()

def _snormal(rs) -> float:
    if hasattr(rs, '_bitgen'):
        return rs._bitgen.random_standard_normal()
    else:
        return rs.bit_generator.random_standard_normal()

class StochasticLib3[R]:
    rs: R
    accuracy: float
    hyp_n_last: int
    hyp_m_last: int
    hyp_N_last: int
    hyp_mode: int
    hyp_mp: int
    hyp_bound: int
    hyp_a: float
    hyp_h: float
    hyp_fm: float
    fnc_n_last: int
    fnc_m_last: int
    fnc_N_last: int
    fnc_o_last: float
    fnc_bound: int
    fnc_f0: float
    fnc_scale: float
    fnc_a: float
    fnc_h: float
    fnc_lfm: float
    fnc_logb: float
    wnc_n_last: int
    wnc_m_last: int
    wnc_N_last: int
    wnc_o_last: float
    wnc_bound1: int
    wnc_bound2: int
    wnc_mode: int
    wnc_a: float
    wnc_h: float
    wnc_k: float
    use_chop_down: bool
    wall_tablen: int
    wall_x1: int
    wall_ytable: List[float]

    def __init__(self, rs: R):
        self.rs = rs
        self.accuracy = 1.E-8
        self.hyp_n_last = -1
        self.hyp_m_last = -1
        self.hyp_N_last = -1
        self.hyp_mode = 0
        self.hyp_mp = 0
        self.hyp_bound = 0
        self.hyp_a = 0.0
        self.hyp_h = 0.0
        self.hyp_fm = 0.0
        self.fnc_n_last = -1
        self.fnc_m_last = -1
        self.fnc_N_last = -1
        self.fnc_o_last = -1.
        self.fnc_bound = 0
        self.fnc_f0 = 0.0
        self.fnc_scale = 0.0
        self.fnc_a = 0.0
        self.fnc_h = 0.0
        self.fnc_lfm = 0.0
        self.fnc_logb = 0.0
        self.wnc_n_last = -1
        self.wnc_m_last = -1
        self.wnc_N_last = -1
        self.wnc_o_last = -1.
        self.wnc_bound1 = 0
        self.wnc_bound2 = 0
        self.wnc_mode = 0
        self.wnc_a = 0.0
        self.wnc_h = 0.0
        self.wnc_k = 0.0
        self.use_chop_down = False
        self.wall_tablen = 0
        self.wall_x1 = 0
        self.wall_ytable = [0.0 for _ in range(514)]

    def Random(self) -> float:
        return _unif(self.rs)

    def Normal(self, m: float, s: float) -> float:
        return m + s * _snormal(self.rs)

    def fc_lnpk(self, k: int, L: int, m: int, n: int) -> float:
        return LnFac(k) + LnFac(m - k) + LnFac(n - k) + LnFac(L + k)

    # ---- stoc1: hypergeometric
    def Hypergeometric(self, n0: int, m0: int, N: int) -> int:
        n = n0
        m = m0
        if n > N or m > N or n < 0 or m < 0:
            raise RuntimeError("Parameter out of range in hypergeometric function")
        fak = 1
        addd = 0
        if m > N // 2:
            m = N - m
            fak = -1
            addd = n
        if n > N // 2:
            n = N - n
            addd += fak * m
            fak = -fak
        if n > m:
            x = n
            n = m
            m = x
        if n == 0:
            return addd
        x = 0
        if N > 680 or n > 70:
            x = self.HypRatioOfUnifoms(n, m, N)
        else:
            x = self.HypInversionMod(n, m, N)
        return x * fak + addd

    def HypInversionMod(self, n: int, m: int, N: int) -> int:
        L = N - m - n
        L1 = float(L)
        Mp = float(m + 1)
        np_ = float(n + 1)
        if N != self.hyp_N_last or m != self.hyp_m_last or n != self.hyp_n_last:
            self.hyp_N_last = N
            self.hyp_m_last = m
            self.hyp_n_last = n
            p = Mp / (N + 2.)
            modef = np_ * p
            self.hyp_mode = int(modef)
            if float(self.hyp_mode) == modef and p == 0.5:
                self.hyp_mp = self.hyp_mode
                self.hyp_mode -= 1
            else:
                self.hyp_mp = self.hyp_mode + 1
            hm = self.hyp_mode
            self.hyp_fm = exp(LnFac(N - m) - LnFac(L + hm) - LnFac(n - hm)
                              + LnFac(m) - LnFac(m - hm) - LnFac(hm)
                              - LnFac(N) + LnFac(N - n) + LnFac(n))
            self.hyp_bound = int(modef + 11. * sqrt(modef * (1. - p) * (1. - n / float(N)) + 1.))
            if self.hyp_bound > n:
                self.hyp_bound = n
        while True:
            U = self.Random()
            U -= self.hyp_fm
            if U <= 0.:
                return self.hyp_mode
            c = self.hyp_fm
            d = self.hyp_fm
            k1 = float(self.hyp_mp - 1)
            k2 = float(self.hyp_mode + 1)
            I = 1
            while I <= self.hyp_mode:
                divisor = (np_ - k1) * (Mp - k1)
                U *= divisor
                d *= divisor
                c *= k1 * (L1 + k1)
                U -= c
                if U <= 0.:
                    return self.hyp_mp - I - 1
                divisor = k2 * (L1 + k2)
                U *= divisor
                c *= divisor
                d *= (np_ - k2) * (Mp - k2)
                U -= d
                if U <= 0.:
                    return self.hyp_mode + I
                I += 1
                k1 -= 1
                k2 += 1
            I = self.hyp_mp + self.hyp_mode
            k2 = float(I)
            while I <= self.hyp_bound:
                divisor = k2 * (L1 + k2)
                U *= divisor
                d *= (np_ - k2) * (Mp - k2)
                U -= d
                if U <= 0.:
                    return I
                I += 1
                k2 += 1

    def HypRatioOfUnifoms(self, n: int, m: int, N: int) -> int:
        L = N - m - n
        if self.hyp_N_last != N or self.hyp_m_last != m or self.hyp_n_last != n:
            self.hyp_N_last = N
            self.hyp_m_last = m
            self.hyp_n_last = n
            rNN = 1. / (float(N) * (N + 2))
            my = float(n) * m * rNN * (N + 2)
            mode = int(float(n + 1) * float(m + 1) * rNN * N)
            var = float(n) * m * (N - m) * (N - n) / (float(N) * N * (N - 1))
            self.hyp_h = sqrt(SHAT1 * (var + 0.5)) + SHAT2
            self.hyp_a = my + 0.5
            self.hyp_fm = self.fc_lnpk(mode, L, m, n)
            self.hyp_bound = int(self.hyp_a + 4.0 * self.hyp_h)
            if self.hyp_bound > n:
                self.hyp_bound = n
        k = 0
        while True:
            u = self.Random()
            if u == 0:
                continue
            x = self.hyp_a + self.hyp_h * (self.Random() - 0.5) / u
            if x < 0. or x > 2E9:
                continue
            k = int(x)
            if k > self.hyp_bound:
                continue
            lf = self.hyp_fm - self.fc_lnpk(k, L, m, n)
            if u * (4.0 - u) - 3.0 <= lf:
                break
            if u * (u - lf) > 1.0:
                continue
            if 2.0 * log(u) <= lf:
                break
        return k

    # ---- stoc3: Fisher
    def FishersNCHyp(self, n0: int, m0: int, N: int, odds0: float) -> int:
        n = n0
        m = m0
        odds = odds0
        if n > N or m > N or n < 0 or m < 0 or odds <= 0.:
            if odds == 0.:
                if n > N - m:
                    raise RuntimeError("Not enough items with nonzero weight in function FishersNCHyp")
                return 0
            raise RuntimeError("Parameter out of range in function FishersNCHyp")
        if odds == 1.:
            return self.Hypergeometric(n, m, N)
        fak = 1
        addd = 0
        if m > N // 2:
            m = N - m
            fak = -1
            addd = n
        if n > N // 2:
            n = N - n
            addd += fak * m
            fak = -fak
        if n > m:
            x = n
            n = m
            m = x
        if n == 0 or odds == 0.:
            return addd
        if fak == -1:
            odds = 1. / odds
        x = 0
        if n < 30 and N < 1024 and odds > 1.E-5 and odds < 1.E5:
            x = self.FishersNCHypInversion(n, m, N, odds)
        else:
            x = self.FishersNCHypRatioOfUnifoms(n, m, N, odds)
        return x * fak + addd

    def FishersNCHypInversion(self, n: int, m: int, N: int, odds: float) -> int:
        L = N - m - n
        if n != self.fnc_n_last or m != self.fnc_m_last or N != self.fnc_N_last or odds != self.fnc_o_last:
            self.fnc_n_last = n
            self.fnc_m_last = m
            self.fnc_N_last = N
            self.fnc_o_last = odds
            self.fnc_f0 = 1.E-100
            sm = self.fnc_f0
            f = self.fnc_f0
            self.fnc_scale = 1.
            a1 = float(m)
            a2 = float(n)
            b1 = 1.0
            b2 = float(L + 1)
            for x in range(1, n + 1):
                f1 = a1 * a2 * odds
                f2 = b1 * b2
                a1 -= 1
                a2 -= 1
                b1 += 1
                b2 += 1
                f *= f1
                sm *= f2
                self.fnc_scale *= f2
                sm += f
            self.fnc_f0 *= self.fnc_scale
            self.fnc_scale = sm
        u = self.Random() * self.fnc_scale
        f = self.fnc_f0
        x = 0
        a1 = float(m)
        a2 = float(n)
        b1 = 0.0
        b2 = float(L)
        while True:
            u -= f
            if u <= 0:
                break
            x += 1
            b1 += 1
            b2 += 1
            f *= a1 * a2 * odds
            u *= b1 * b2
            a1 -= 1
            a2 -= 1
            if not (x < n):
                break
        return x

    def FishersNCHypRatioOfUnifoms(self, n: int, m: int, N: int, odds: float) -> int:
        L = N - m - n
        if n != self.fnc_n_last or m != self.fnc_m_last or N != self.fnc_N_last or odds != self.fnc_o_last:
            self.fnc_n_last = n
            self.fnc_m_last = m
            self.fnc_N_last = N
            self.fnc_o_last = odds
            AA = (m + n) * odds + L
            BB = sqrt(AA * AA - 4 * odds * (odds - 1) * m * n)
            mean = (AA - BB) / (2 * (odds - 1))
            AA = mean * (m - mean)
            BB = (n - mean) * (mean + L)
            variance = N * AA * BB / ((N - 1) * (m * BB + (n + L) * AA))
            self.fnc_logb = log(odds)
            self.fnc_a = mean + 0.5
            self.fnc_h = 1.028 + 1.717 * sqrt(variance + 0.5) + 0.032 * fabs(self.fnc_logb)
            self.fnc_bound = int(mean + 4.0 * self.fnc_h)
            if self.fnc_bound > n:
                self.fnc_bound = n
            mode = int(mean)
            g1 = float(m - mode) * (n - mode) * odds
            g2 = float(mode + 1) * (L + mode + 1)
            if g1 > g2 and mode < n:
                mode += 1
            self.fnc_lfm = mode * self.fnc_logb - self.fc_lnpk(mode, L, m, n)
        k = 0
        while True:
            u = self.Random()
            if u == 0:
                continue
            x = self.fnc_a + self.fnc_h * (self.Random() - 0.5) / u
            if x < 0. or x > 2E9:
                continue
            k = int(x)
            if k > self.fnc_bound:
                continue
            lf = k * self.fnc_logb - self.fc_lnpk(k, L, m, n) - self.fnc_lfm
            if u * (4.0 - u) - 3.0 <= lf:
                break
            if u * (u - lf) > 1.0:
                continue
            if 2.0 * log(u) <= lf:
                break
        return k

    # ---- stoc3: Wallenius
    def WalleniusNCHyp(self, n: int, m: int, N: int, odds: float) -> int:
        if n >= N or m >= N or n <= 0 or m <= 0 or odds <= 0.:
            if n == 0 or m == 0:
                return 0
            if m == N:
                return n
            if n == N:
                return m
            if odds == 0.:
                if n > N - m:
                    raise RuntimeError("Not enough items with nonzero weight in function WalleniusNCHyp")
                return 0
            raise RuntimeError("Parameter out of range in function WalleniusNCHyp")
        if odds == 1.:
            return self.Hypergeometric(n, m, N)
        if n < 30:
            return self.WalleniusNCHypUrn(n, m, N, odds)
        if float(n) * N < 10000:
            return self.WalleniusNCHypTable(n, m, N, odds)
        return self.WalleniusNCHypRatioOfUnifoms(n, m, N, odds)

    def WalleniusNCHypUrn(self, n0: int, m0: int, N: int, odds: float) -> int:
        n = n0
        m = m0
        x = 0
        m2 = N - m
        mw1 = m * odds
        mw2 = float(m2)
        while True:
            if self.Random() * (mw1 + mw2) < mw1:
                x += 1
                m -= 1
                if m == 0:
                    break
                mw1 = m * odds
            else:
                m2 -= 1
                if m2 == 0:
                    x += n - 1
                    break
                mw2 = float(m2)
            n -= 1
            if n == 0:
                break
        return x

    def WalleniusNCHypTable(self, n: int, m: int, N: int, odds: float) -> int:
        if n != self.wnc_n_last or m != self.wnc_m_last or N != self.wnc_N_last or odds != self.wnc_o_last:
            self.wnc_n_last = n
            self.wnc_m_last = m
            self.wnc_N_last = N
            self.wnc_o_last = odds
            wnch = WalleniusNCHypergeometric(n, m, N, odds)
            res = wnch.make_table(512)
            if res[0] != 0:
                self.wall_x1 = res[1]
                self.wall_ytable = res[3]
                self.wall_tablen = res[2] - res[1] + 1
            else:
                self.wall_tablen = 0
        if self.wall_tablen == 0:
            return self.WalleniusNCHypRatioOfUnifoms(n, m, N, odds)
        while True:
            u = self.Random()
            for x in range(self.wall_tablen):
                u -= self.wall_ytable[x]
                if u < 0.:
                    return x + self.wall_x1

    def WalleniusNCHypRatioOfUnifoms(self, n: int, m: int, N: int, odds: float) -> int:
        rsqrt2pi = 0.3989422804014326857
        wnch = WalleniusNCHypergeometric(n, m, N, odds, self.accuracy)
        xmin = m + n - N
        if xmin < 0:
            xmin = 0
        xmax = n
        if xmax > m:
            xmax = m
        if n != self.wnc_n_last or m != self.wnc_m_last or N != self.wnc_N_last or odds != self.wnc_o_last:
            self.wnc_n_last = n
            self.wnc_m_last = m
            self.wnc_N_last = N
            self.wnc_o_last = odds
            mean = wnch.mean()
            r1 = mean * (m - mean)
            r2 = (n - mean) * (mean + N - n - m)
            variance = N * r1 * r2 / ((N - 1) * (m * r2 + (N - m) * r1))
            self.use_chop_down = variance < 4.
            if not self.use_chop_down:
                self.wnc_mode = int(mean)
                f2 = 0.0
                if odds < 1.:
                    if self.wnc_mode < xmax:
                        self.wnc_mode += 1
                    x2 = xmin
                    if odds > 0.294 and N <= 10000000:
                        x2 = self.wnc_mode - 1
                    xi = self.wnc_mode
                    while xi >= x2:
                        f = wnch.probability(xi)
                        if f <= f2:
                            break
                        self.wnc_mode = xi
                        f2 = f
                        xi -= 1
                else:
                    if self.wnc_mode < xmin:
                        self.wnc_mode += 1
                    x2 = xmax
                    if odds < 3.4 and N <= 10000000:
                        x2 = self.wnc_mode + 1
                    xi = self.wnc_mode
                    while xi <= x2:
                        f = wnch.probability(xi)
                        if f <= f2:
                            break
                        self.wnc_mode = xi
                        f2 = f
                        xi += 1
                self.wnc_k = f2
                variance = rsqrt2pi / self.wnc_k
                variance *= variance
                self.wnc_a = mean + 0.5
                s123 = 0.40 + 0.8579 * sqrt(variance + 0.5) + 0.4 * fabs(mean - self.wnc_mode)
                s4 = 0.
                r1 = xmax - mean - s123
                r2 = mean - s123 - xmin
                if r1 > r2:
                    r1 = r2
                if (odds > 5. or odds < 0.2) and r1 >= -0.5 and r1 <= 8.:
                    if r1 < 1.:
                        r1 = 1.
                    s4 = 0.029 * pow(float(N), 0.23) / (r1 * r1)
                self.wnc_h = 2. * (s123 + s4)
                self.wnc_bound1 = int(mean - 4. * self.wnc_h)
                if self.wnc_bound1 < xmin:
                    self.wnc_bound1 = xmin
                self.wnc_bound2 = int(mean + 4. * self.wnc_h)
                if self.wnc_bound2 > xmax:
                    self.wnc_bound2 = xmax
        if self.use_chop_down:
            return self.WalleniusNCHypInversion(n, m, N, odds)
        xi = 0
        while True:
            u = self.Random()
            if u == 0.:
                continue
            x = self.wnc_a + self.wnc_h * (self.Random() - 0.5) / u
            if x < 0. or x > 2E9:
                continue
            xi = int(x)
            if xi < self.wnc_bound1 or xi > self.wnc_bound2:
                continue
            s2 = self.wnc_h * 0.5
            s2 *= s2
            xma2 = xi - (self.wnc_a - 0.5)
            xma2 *= xma2
            hx = 1. if s2 >= xma2 else s2 / xma2
            if wnch.bernouilli_h(xi, hx * self.wnc_k * 1.01, u * u * self.wnc_k * 1.01, self):
                break
        return xi

    def WalleniusNCHypInversion(self, n: int, m: int, N: int, odds: float) -> int:
        wnch1 = WalleniusNCHypergeometric(n, m, N, odds, self.accuracy)
        wnch2 = WalleniusNCHypergeometric(n, m, N, odds, self.accuracy)
        accura = self.accuracy * 0.01
        if accura > 1E-7:
            accura = 1E-7
        wall_x1 = int(wnch1.mean())
        x2 = wall_x1 + 1
        xmin = m + n - N
        if xmin < 0:
            xmin = 0
        xmax = n
        if xmax > m:
            xmax = m
        updown = 3
        while True:
            u = self.Random()
            while updown != 0:
                if (updown & 1) != 0:
                    if wall_x1 < xmin:
                        updown &= ~1
                    else:
                        f = wnch1.probability(wall_x1)
                        u -= f
                        if u <= 0.:
                            return wall_x1
                        wall_x1 -= 1
                        if f < accura:
                            updown &= ~1
                if (updown & 2) != 0:
                    if x2 > xmax:
                        updown &= ~2
                    else:
                        f = wnch2.probability(x2)
                        u -= f
                        if u <= 0.:
                            return x2
                        x2 += 1
                        if f < accura:
                            updown &= ~2
