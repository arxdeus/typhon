# Ported from numpy 2.5 (BSD-3-Clause): numpy/random/mtrand.pyx,
# numpy/random/src/legacy/legacy-distributions.c and the legacy (masked)
# path of numpy/random/_bounded_integers.pyx.in.
#
# numpy.random.RandomState: the legacy MT19937 stream with the legacy
# distributions (polar Box-Muller gauss with a cached second value, masked
# rejection randint, NumPy 1.16 gamma/beta/binomial/...). The module level
# numpy.random.* functions are methods of a hidden global RandomState.

import internal.static as static

from ..routines import *
from ..util import PI, zext, itrunc, isnan, nan64, fabs, cmod, floor, ceil, log, log1p, exp, pow, sqrt, cos, acos
from .mt19937 import MT19937, N as _MT_N
from .seed import SeedSequence
from .bitgen import BitGenerator, Binomial, cont, convert_array_like, gather_arrays, \
                    check_array_constraint, kahan_sum, CONS_NONE, CONS_NON_NEGATIVE, \
                    CONS_POSITIVE, CONS_BOUNDED_0_1, CONS_BOUNDED_GT_0_1, \
                    CONS_BOUNDED_LT_0_1, CONS_GT_1, CONS_GTE_1, CONS_POISSON
from threading import Lock

_RAND_INT_MAX = 9.223372036854775807e18

# numpy's C sources are compiled with clang/gcc's default -ffp-contract, which
# fuses `a*b + c` inside one expression into llvm.fmuladd (an FMA on
# arm64/x86-64 with FMA). fmuladd() reproduces exactly those contractions so
# the legacy stream stays bit identical; each use mirrors one C expression.
@pure
@llvm
def fmuladd(a: float, b: float, c: float) -> float:
    declare double @llvm.fmuladd.f64(double, double, double)
    %r = call double @llvm.fmuladd.f64(double %a, double %b, double %c)
    ret double %r

@pure
@llvm
def _loggam_coef(idx: int) -> float:
    @data = private unnamed_addr constant [10 x double] [double 0x3FB5555555555555, double 0xBF66C16C16C16C17, double 0x3F4A01A01A01A01A, double 0xBF43813813813813, double 0x3F4B951E2B18FF24, double 0xBF5F6AB0D9993C7F, double 0x3F7A41A41A41A41A, double 0xBF9E4286CB0F5397, double 0x3FC6FE96381E0685, double 0xBFF6476701181F35], align 16
    %p = getelementptr inbounds [10 x double], ptr @data, i64 0, i64 %idx
    %x = load double, ptr %p, align 8
    ret double %x

def _loggam(x: float) -> float:
    # random_loggam (distributions.c)
    n = 0
    if x == 1.0 or x == 2.0:
        return 0.0
    elif x < 7.0:
        n = int(7 - x)
    x0 = x + n
    x2 = (1.0 / x0) * (1.0 / x0)
    lg2pi = 1.8378770664093453e+00
    gl0 = _loggam_coef(9)
    k = 8
    while k >= 0:
        gl0 *= x2
        gl0 += _loggam_coef(k)
        k -= 1
    gl = fmuladd(x0 - 0.5, log(x0), fmuladd(0.5, lg2pi, gl0 / x0)) - x0
    if x < 7.0:
        k = 1
        while k <= n:
            gl -= log(x0 - 1.0)
            x0 -= 1.0
            k += 1
    return gl

def _size_shape(size):
    if size is None:
        return ()
    elif isinstance(size, int):
        return (size,)
    else:
        return size

def _bounds_error(closed: bool, low_is_zero: bool):
    if low_is_zero:
        return 'high < 0' if closed else 'high <= 0'
    else:
        return 'low > high' if closed else 'low >= high'


class RandomState:
    _bitgen: BitGenerator[MT19937]
    _binomial: Binomial
    _has_binomial: bool
    has_gauss: bool
    gauss: float
    lock: Lock

    def __init__(self, seed = None):
        self._bitgen = BitGenerator[MT19937](MT19937(0))
        self._binomial = Binomial()
        self._has_binomial = False
        self.has_gauss = False
        self.gauss = 0.0
        self.lock = Lock()
        self.seed(seed)

    def __str__(self):
        return "RandomState(MT19937)"

    def __repr__(self):
        return "RandomState(MT19937)"

    # state

    def _reset_gauss(self):
        self.has_gauss = False
        self.gauss = 0.0

    def seed(self, seed = None):
        mt = self._bitgen.state
        if seed is None:
            val = SeedSequence().generate_state(_MT_N, u32)
            mt.init_array(val.data, _MT_N)
        elif isinstance(seed, int):
            if seed < 0 or seed > 0xffffffff:
                raise ValueError("Seed must be between 0 and 2**32 - 1")
            mt.seed_legacy(seed)
        elif isinstance(seed, Optional):
            self.seed(seed.__val__())
            return
        else:
            arr = asarray(seed)
            if static.len(arr.shape) == 0:
                self.seed(int(arr.item()))
                return
            n = arr.size
            if n == 0:
                raise ValueError("Seed must be non-empty")
            key = Ptr[u32](n)
            i = 0
            for v in arr.flat:
                iv = int(v)
                if iv < 0 or iv > 0xffffffff:
                    raise ValueError("Seed values must be between 0 and 2**32 - 1")
                key[i] = u32(iv)
                i += 1
            mt.init_array(key, n)
        self._bitgen.has_uint32 = False
        self._reset_gauss()

    def get_state(self, legacy: bool = True):
        mt = self._bitgen.state
        key = empty((_MT_N,), dtype=u32)
        for i in range(_MT_N):
            key.data[i] = mt.state[i]
        return ('MT19937', key, mt.pos, int(self.has_gauss), self.gauss)

    def set_state(self, state):
        if state[0] != 'MT19937':
            raise ValueError('set_state can only be used with legacy MT19937 state instances.')
        key = asarray(state[1])
        if key.size != _MT_N:
            raise ValueError("state must be 624 elements")
        mt = self._bitgen.state
        i = 0
        for v in key.flat:
            mt.state[i] = u32(int(v))
            i += 1
        mt.data[0] = u32(int(state[2]))
        if static.len(state) > 3:
            self.has_gauss = bool(state[3])
            self.gauss = float(state[4])
        else:
            self._reset_gauss()

    # legacy core (legacy-distributions.c)

    def _double(self) -> float:
        return self._bitgen.state.genrand_res53()

    def _gauss(self) -> float:
        if self.has_gauss:
            tmp = self.gauss
            self.has_gauss = False
            self.gauss = 0.0
            return tmp
        x1 = 0.0
        x2 = 0.0
        r2 = 0.0
        while True:
            x1 = 2.0 * self._double() - 1.0
            x2 = 2.0 * self._double() - 1.0
            r2 = fmuladd(x1, x1, x2 * x2)
            if not (r2 >= 1.0 or r2 == 0.0):
                break
        f = sqrt(-2.0 * log(r2) / r2)
        self.gauss = f * x1
        self.has_gauss = True
        return f * x2

    def _std_exp(self) -> float:
        return -log(1.0 - self._double())

    def _std_gamma(self, shape: float) -> float:
        if shape == 1.0:
            return self._std_exp()
        elif shape == 0.0:
            return 0.0
        elif shape < 1.0:
            while True:
                U = self._double()
                V = self._std_exp()
                if U <= 1.0 - shape:
                    X = pow(U, 1. / shape)
                    if X <= V:
                        return X
                else:
                    Y = -log((1 - U) / shape)
                    X = pow(fmuladd(shape, Y, 1.0 - shape), 1. / shape)
                    if X <= V + Y:
                        return X
            return 0.0
        else:
            b = shape - 1. / 3.
            c = 1. / sqrt(9 * b)
            while True:
                X = 0.0
                V = 0.0
                while True:
                    X = self._gauss()
                    V = fmuladd(c, X, 1.0)
                    if V > 0.0:
                        break
                V = V * V * V
                U = self._double()
                if U < fmuladd(-(0.0331 * (X * X)), X * X, 1.0):
                    return b * V
                if log(U) < fmuladd(0.5 * X, X, b * (1. - V + log(V))):
                    return b * V
            return 0.0

    def _l_normal(self, loc: float, scale: float) -> float:
        return fmuladd(scale, self._gauss(), loc)

    def _l_gamma(self, shape: float, scale: float) -> float:
        return scale * self._std_gamma(shape)

    def _l_exponential(self, scale: float) -> float:
        return scale * self._std_exp()

    def _l_pareto(self, a: float) -> float:
        return exp(self._std_exp() / a) - 1

    def _l_weibull(self, a: float) -> float:
        if a == 0.0:
            return 0.0
        return pow(self._std_exp(), 1. / a)

    def _l_power(self, a: float) -> float:
        return pow(1 - exp(-self._std_exp()), 1. / a)

    def _l_chisquare(self, df: float) -> float:
        return 2.0 * self._std_gamma(df / 2.0)

    def _l_rayleigh(self, mode: float) -> float:
        return mode * sqrt(-2.0 * log1p(-self._double()))

    def _l_noncentral_chisquare(self, df: float, nonc: float) -> float:
        if nonc == 0:
            return self._l_chisquare(df)
        if 1 < df:
            Chi2 = self._l_chisquare(df - 1)
            n = self._gauss() + sqrt(nonc)
            return fmuladd(n, n, Chi2)
        else:
            i = self._poisson(nonc / 2.0)
            out = self._l_chisquare(df + 2 * i)
            if isnan(nonc):
                return nan64()
            return out

    def _l_noncentral_f(self, dfnum: float, dfden: float, nonc: float) -> float:
        t = self._l_noncentral_chisquare(dfnum, nonc) * dfden
        return t / (self._l_chisquare(dfden) * dfnum)

    def _l_wald(self, mean: float, scale: float) -> float:
        mu_2l = mean / (2 * scale)
        Y = self._gauss()
        Y = mean * Y * Y
        X = fmuladd(mu_2l, Y - sqrt(fmuladd(4 * scale, Y, Y * Y)), mean)
        U = self._double()
        if U <= mean / (mean + X):
            return X
        else:
            return mean * mean / X

    def _l_lognormal(self, mean: float, sigma: float) -> float:
        return exp(self._l_normal(mean, sigma))

    def _l_standard_t(self, df: float) -> float:
        num = self._gauss()
        denom = self._std_gamma(df / 2)
        return sqrt(df / 2) * num / sqrt(denom)

    def _l_negative_binomial(self, n: float, p: float) -> int:
        Y = self._l_gamma(n, (1 - p) / p)
        return self._poisson(Y)

    def _l_standard_cauchy(self) -> float:
        return self._gauss() / self._gauss()

    def _l_beta(self, a: float, b: float) -> float:
        if a <= 1.0 and b <= 1.0:
            while True:
                U = self._double()
                V = self._double()
                X = pow(U, 1.0 / a)
                Y = pow(V, 1.0 / b)
                if X + Y <= 1.0:
                    if X + Y > 0:
                        return X / (X + Y)
                    else:
                        logX = log(U) / a
                        logY = log(V) / b
                        logM = logX if logX > logY else logY
                        logX -= logM
                        logY -= logM
                        return exp(logX - log(exp(logX) + exp(logY)))
            return 0.0
        else:
            Ga = self._std_gamma(a)
            Gb = self._std_gamma(b)
            return Ga / (Ga + Gb)

    def _l_f(self, dfnum: float, dfden: float) -> float:
        return (self._l_chisquare(dfnum) * dfden) / (self._l_chisquare(dfden) * dfnum)

    def _binomial_inversion(self, n: int, p: float) -> int:
        q = 0.0
        qn = 0.0
        np = 0.0
        bound = 0
        b = self._binomial
        if not self._has_binomial or b.nsave != n or b.psave != p:
            b.nsave = n
            b.psave = p
            self._has_binomial = True
            q = 1.0 - p
            qn = exp(n * log(q))
            np = n * p
            bound = int(min(float(n), fmuladd(10.0, sqrt(fmuladd(np, q, 1.0)), np)))
            b.q = q
            b.r = qn
            b.c = np
            b.m = bound
        else:
            q = b.q
            qn = b.r
            np = b.c
            bound = b.m
        X = 0
        px = qn
        U = self._double()
        while U > px:
            X += 1
            if X > bound:
                X = 0
                px = qn
                U = self._double()
            else:
                U -= px
                px = ((n - X + 1) * p * px) / (X * q)
        return X

    def _binomial_btpe(self, n: int, p: float) -> int:
        b = self._binomial
        if not self._has_binomial or b.nsave != n or b.psave != p:
            b.nsave = n
            b.psave = p
            self._has_binomial = True
            b.r = min(p, 1.0 - p)
            b.q = 1.0 - b.r
            b.fm = fmuladd(float(n), b.r, b.r)
            b.m = int(floor(b.fm))
            b.p1 = floor(fmuladd(2.195, sqrt(n * b.r * b.q), -(4.6 * b.q))) + 0.5
            b.xm = b.m + 0.5
            b.xl = b.xm - b.p1
            b.xr = b.xm + b.p1
            b.c = 0.134 + 20.5 / (15.3 + b.m)
            a = (b.fm - b.xl) / fmuladd(-b.xl, b.r, b.fm)
            b.laml = a * (1.0 + a / 2.0)
            a = (b.xr - b.fm) / (b.xr * b.q)
            b.lamr = a * (1.0 + a / 2.0)
            b.p2 = b.p1 * fmuladd(2.0, b.c, 1.0)
            b.p3 = b.p2 + b.c / b.laml
            b.p4 = b.p3 + b.c / b.lamr
        r = b.r
        q = b.q
        m = b.m
        p1 = b.p1
        xm = b.xm
        xl = b.xl
        xr = b.xr
        c = b.c
        laml = b.laml
        lamr = b.lamr
        p2 = b.p2
        p3 = b.p3
        p4 = b.p4

        y = 0
        while True:  # Step10
            nrq = n * r * q
            u = self._double() * p4
            v = self._double()
            if u <= p1:
                y = int(floor(fmuladd(-p1, v, xm) + u))
                break  # Step60
            if u <= p2:  # Step20
                x = xl + (u - p1) / c
                v = fmuladd(v, c, 1.0) - fabs(m - x + 0.5) / p1
                if v > 1.0:
                    continue
                y = int(floor(x))
            elif u <= p3:  # Step30
                y = int(floor(xl + log(v) / laml))
                if y < 0 or v == 0.0:
                    continue
                v = v * (u - p2) * laml
            else:  # Step40
                y = int(floor(xr - log(v) / lamr))
                if y > n or v == 0.0:
                    continue
                v = v * (u - p3) * lamr
            # Step50
            k = abs(y - m)
            if not (k > 20 and k < (nrq / 2.0 - 1)):
                s = r / q
                a = s * (n + 1)
                F = 1.0
                if m < y:
                    for i in range(m + 1, y + 1):
                        F *= (a / i - s)
                elif m > y:
                    for i in range(y + 1, m + 1):
                        F /= (a / i - s)
                if v > F:
                    continue
                break
            # Step52
            rho = (k / nrq) * (fmuladd(float(k), k / 3.0 + 0.625, 0.16666666666666666) / nrq + 0.5)
            t = -k * k / (2 * nrq)
            A = log(v)
            if A < (t - rho):
                break
            if A > (t + rho):
                continue
            x1 = float(y) + 1
            f1 = float(m) + 1
            z = float(n) + 1 - float(m)
            w = float(n) - float(y) + 1
            x2 = x1 * x1
            f2 = f1 * f1
            z2 = z * z
            w2 = w * w
            if (A > (fmuladd(float(y - m), log(w * r / (x1 * q)),
                             fmuladd(xm, log(f1 / x1), (n - m + 0.5) * log(z / w))) +
                     (13680. - (462. - (132. - (99. - 140. / f2) / f2) / f2) / f2) / f1 / 166320. +
                     (13680. - (462. - (132. - (99. - 140. / z2) / z2) / z2) / z2) / z / 166320. +
                     (13680. - (462. - (132. - (99. - 140. / x2) / x2) / x2) / x2) / x1 / 166320. +
                     (13680. - (462. - (132. - (99. - 140. / w2) / w2) / w2) / w2) / w / 166320.)):
                continue
            break
        # Step60
        if p > 0.5:
            y = n - y
        return y

    def _l_binomial(self, p: float, n: int) -> int:
        if p <= 0.5:
            if p * n <= 30.0:
                return self._binomial_inversion(n, p)
            else:
                return self._binomial_btpe(n, p)
        else:
            q = 1.0 - p
            if q * n <= 30.0:
                return n - self._binomial_inversion(n, q)
            else:
                return n - self._binomial_btpe(n, q)

    def _hyp(self, good: int, bad: int, sample: int) -> int:
        d1 = bad + good - sample
        d2 = float(min(bad, good))
        y = d2
        k = sample
        while y > 0.0:
            u = self._double()
            y -= int(floor(u + y / (d1 + k)))
            k -= 1
            if k == 0:
                break
        z = int(d2 - y)
        if good > bad:
            z = sample - z
        return z

    def _hrua(self, good: int, bad: int, sample: int) -> int:
        D1 = 1.7155277699214135
        D2 = 0.8989161620588988
        loggam = _loggam
        mingoodbad = min(good, bad)
        popsize = good + bad
        maxgoodbad = max(good, bad)
        m = min(sample, popsize - sample)
        d4 = float(mingoodbad) / popsize
        d5 = 1.0 - d4
        d6 = fmuladd(float(m), d4, 0.5)
        d7 = sqrt(float(popsize - m) * sample * d4 * d5 / (popsize - 1) + 0.5)
        d8 = fmuladd(D1, d7, D2)
        d9 = int(floor(float(m + 1) * (mingoodbad + 1) / (popsize + 2)))
        d10 = (loggam(float(d9 + 1)) + loggam(float(mingoodbad - d9 + 1)) +
               loggam(float(m - d9 + 1)) + loggam(float(maxgoodbad - m + d9 + 1)))
        d11 = min(min(m, mingoodbad) + 1.0, floor(fmuladd(16.0, d7, d6)))
        Z = 0
        while True:
            X = self._double()
            Y = self._double()
            W = d6 + d8 * (Y - 0.5) / X
            if W < 0.0 or W >= d11:
                continue
            Z = int(floor(W))
            T = d10 - (loggam(float(Z + 1)) + loggam(float(mingoodbad - Z + 1)) +
                       loggam(float(m - Z + 1)) + loggam(float(maxgoodbad - m + Z + 1)))
            if fmuladd(X, 4.0 - X, -3.0) <= T:
                break
            if X * (X - T) >= 1:
                continue
            if 2.0 * log(X) <= T:
                break
        if good > bad:
            Z = m - Z
        if m < sample:
            Z = good - Z
        return Z

    def _l_hypergeometric(self, good: int, bad: int, sample: int) -> int:
        if sample > 10:
            return self._hrua(good, bad, sample)
        elif sample > 0:
            return self._hyp(good, bad, sample)
        else:
            return 0

    def _poisson_ptrs(self, lam: float) -> int:
        slam = sqrt(lam)
        loglam = log(lam)
        b = fmuladd(2.53, slam, 0.931)
        a = fmuladd(0.02483, b, -0.059)
        invalpha = 1.1239 + 1.1328 / (b - 3.4)
        vr = 0.9277 - 3.6224 / (b - 2)
        while True:
            U = self._double() - 0.5
            V = self._double()
            us = 0.5 - fabs(U)
            k = int(floor(fmuladd(2 * a / us + b, U, lam) + 0.43))
            if us >= 0.07 and V <= vr:
                return k
            if k < 0 or (us < 0.013 and V > us):
                continue
            if ((log(V) + log(invalpha) - log(a / (us * us) + b)) <=
                (fmuladd(float(k), loglam, -lam) - _loggam(float(k) + 1))):
                return k
        return 0

    def _poisson(self, lam: float) -> int:
        if lam >= 10:
            return self._poisson_ptrs(lam)
        elif lam == 0:
            return 0
        else:
            enlam = exp(-lam)
            X = 0
            prod = 1.0
            while True:
                prod *= self._double()
                if prod > enlam:
                    X += 1
                else:
                    return X
            return 0

    def _l_poisson(self, lam: float) -> int:
        return self._poisson(lam)

    def _l_zipf(self, a: float) -> int:
        am1 = a - 1.0
        b = pow(2.0, am1)
        while True:
            U = 1.0 - self._double()
            V = self._double()
            X = floor(pow(U, -1.0 / am1))
            if X > _RAND_INT_MAX or X < 1.0:
                continue
            T = pow(1.0 + 1.0 / X, am1)
            if V * X * (T - 1.0) / (b - 1.0) <= T / b:
                return int(X)
        return 0

    def _l_geometric(self, p: float) -> int:
        if p >= 0.333333333333333333333333:
            return self._bitgen.random_geometric_search(p)
        else:
            return int(ceil(log1p(-self._double()) / log(1 - p)))

    def _l_vonmises(self, mu: float, kappa: float) -> float:
        if isnan(kappa):
            return nan64()
        if kappa < 1e-8:
            return PI * (2 * self._double() - 1)
        s = 0.0
        if kappa < 1e-5:
            s = (1. / kappa + kappa)
        else:
            r = 1 + sqrt(fmuladd(4 * kappa, kappa, 1.0))
            rho = (r - sqrt(2 * r)) / (2 * kappa)
            s = fmuladd(rho, rho, 1.0) / (2 * rho)
        W = 0.0
        while True:
            U = self._double()
            Z = cos(PI * U)
            W = fmuladd(s, Z, 1.0) / (s + Z)
            Y = kappa * (s - W)
            V = self._double()
            if (fmuladd(Y, 2 - Y, -V) >= 0) or (log(Y / V) + 1 - Y >= 0):
                break
        U = self._double()
        result = acos(W)
        if U < 0.5:
            result = -result
        result += mu
        neg = result < 0
        mod = fabs(result)
        mod = cmod(mod + PI, 2 * PI) - PI
        if neg:
            mod *= -1
        return mod

    def _l_logseries(self, p: float) -> int:
        r = log(1.0 - p)
        while True:
            V = self._double()
            if V >= p:
                return 1
            U = self._double()
            q = 1.0 - exp(r * U)
            if V <= q * q:
                result = int(floor(1 + log(V) / log(q)))
                if result < 1 or V == 0.0:
                    continue
                else:
                    return result
            if V >= q:
                return 1
            return 2
        return 0

    def _l_multinomial(self, n: int, mnix: Ptr[int], pix: Ptr[float], d: int):
        remaining_p = 1.0
        dn = n
        for j in range(d - 1):
            mnix[j] = self._l_binomial(pix[j] / remaining_p, dn)
            dn = dn - mnix[j]
            if dn <= 0:
                break
            remaining_p -= pix[j]
        if dn > 0:
            mnix[d - 1] = dn

    # shared (non-legacy) kernels from distributions.c

    def _uniform(self, low: float, high: float) -> float:
        rng = high - low
        if not util.isfinite(rng):
            raise OverflowError('Range exceeds valid bounds')
        return fmuladd(rng, self._double(), low)

    def _laplace(self, loc: float, scale: float) -> float:
        while True:
            U = self._double()
            if U > 0.0:
                lo = U + U
                hi = 2.0 - U - U
                sgn = -1.0 if U >= 0.5 else 1.0
                return fmuladd(sgn * scale, log(lo if lo < hi else hi), loc)
        return 0.0

    def _gumbel(self, loc: float, scale: float) -> float:
        while True:
            U = 1.0 - self._double()
            if U < 1.0:
                return fmuladd(-scale, log(-log(U)), loc)
        return 0.0

    def _logistic(self, loc: float, scale: float) -> float:
        while True:
            U = self._double()
            if U > 0.0:
                return fmuladd(scale, log(U / (1.0 - U)), loc)
        return 0.0

    def _triangular(self, left: float, mode: float, right: float) -> float:
        return self._bitgen.random_triangular(left, mode, right)

    def _interval(self, mx: int) -> int:
        return int(self._bitgen.random_interval(u64(mx)))

    # public API

    def random_sample(self, size = None):
        return cont(self._double, size, self.lock, (), (), ())

    def random(self, size = None):
        return self.random_sample(size)

    def ranf(self, size = None):
        return self.random_sample(size)

    def sample(self, size = None):
        return self.random_sample(size)

    def rand(self, *args):
        if static.len(args) == 0:
            return self.random_sample()
        else:
            return self.random_sample(size=args)

    def randn(self, *args):
        if static.len(args) == 0:
            return self.standard_normal()
        else:
            return self.standard_normal(size=args)

    def standard_normal(self, size = None):
        return cont(self._gauss, size, self.lock, (), (), ())

    def normal(self, loc = 0.0, scale = 1.0, size = None):
        return cont(self._l_normal, size, self.lock, (loc, scale), ('', 'scale'),
                    (CONS_NONE, CONS_NON_NEGATIVE))

    def uniform(self, low = 0.0, high = 1.0, size = None):
        return cont(self._uniform, size, self.lock, (low, high), ('', ''),
                    (CONS_NONE, CONS_NONE))

    def beta(self, a, b, size = None):
        return cont(self._l_beta, size, self.lock, (a, b), ('a', 'b'),
                    (CONS_POSITIVE, CONS_POSITIVE))

    def exponential(self, scale = 1.0, size = None):
        return cont(self._l_exponential, size, self.lock, (scale,), ('scale',),
                    (CONS_NON_NEGATIVE,))

    def standard_exponential(self, size = None):
        return cont(self._std_exp, size, self.lock, (), (), ())

    def standard_gamma(self, shape, size = None):
        return cont(self._std_gamma, size, self.lock, (shape,), ('shape',),
                    (CONS_NON_NEGATIVE,))

    def gamma(self, shape, scale = 1.0, size = None):
        return cont(self._l_gamma, size, self.lock, (shape, scale), ('shape', 'scale'),
                    (CONS_NON_NEGATIVE, CONS_NON_NEGATIVE))

    def f(self, dfnum, dfden, size = None):
        return cont(self._l_f, size, self.lock, (dfnum, dfden), ('dfnum', 'dfden'),
                    (CONS_POSITIVE, CONS_POSITIVE))

    def noncentral_f(self, dfnum, dfden, nonc, size = None):
        return cont(self._l_noncentral_f, size, self.lock, (dfnum, dfden, nonc),
                    ('dfnum', 'dfden', 'nonc'), (CONS_POSITIVE, CONS_POSITIVE, CONS_NON_NEGATIVE))

    def chisquare(self, df, size = None):
        return cont(self._l_chisquare, size, self.lock, (df,), ('df',), (CONS_POSITIVE,))

    def noncentral_chisquare(self, df, nonc, size = None):
        return cont(self._l_noncentral_chisquare, size, self.lock, (df, nonc), ('df', 'nonc'),
                    (CONS_POSITIVE, CONS_NON_NEGATIVE))

    def standard_cauchy(self, size = None):
        return cont(self._l_standard_cauchy, size, self.lock, (), (), ())

    def standard_t(self, df, size = None):
        return cont(self._l_standard_t, size, self.lock, (df,), ('df',), (CONS_POSITIVE,))

    def vonmises(self, mu, kappa, size = None):
        return cont(self._l_vonmises, size, self.lock, (mu, kappa), ('mu', 'kappa'),
                    (CONS_NONE, CONS_NON_NEGATIVE))

    def pareto(self, a, size = None):
        return cont(self._l_pareto, size, self.lock, (a,), ('a',), (CONS_POSITIVE,))

    def weibull(self, a, size = None):
        return cont(self._l_weibull, size, self.lock, (a,), ('a',), (CONS_NON_NEGATIVE,))

    def power(self, a, size = None):
        return cont(self._l_power, size, self.lock, (a,), ('a',), (CONS_POSITIVE,))

    def laplace(self, loc = 0.0, scale = 1.0, size = None):
        return cont(self._laplace, size, self.lock, (loc, scale), ('loc', 'scale'),
                    (CONS_NONE, CONS_NON_NEGATIVE))

    def gumbel(self, loc = 0.0, scale = 1.0, size = None):
        return cont(self._gumbel, size, self.lock, (loc, scale), ('loc', 'scale'),
                    (CONS_NONE, CONS_NON_NEGATIVE))

    def logistic(self, loc = 0.0, scale = 1.0, size = None):
        return cont(self._logistic, size, self.lock, (loc, scale), ('loc', 'scale'),
                    (CONS_NONE, CONS_NON_NEGATIVE))

    def lognormal(self, mean = 0.0, sigma = 1.0, size = None):
        return cont(self._l_lognormal, size, self.lock, (mean, sigma), ('mean', 'sigma'),
                    (CONS_NONE, CONS_NON_NEGATIVE))

    def rayleigh(self, scale = 1.0, size = None):
        return cont(self._l_rayleigh, size, self.lock, (scale,), ('scale',), (CONS_NON_NEGATIVE,))

    def wald(self, mean, scale, size = None):
        return cont(self._l_wald, size, self.lock, (mean, scale), ('mean', 'scale'),
                    (CONS_POSITIVE, CONS_POSITIVE))

    def triangular(self, left, mode, right, size = None):
        left = convert_array_like(left)
        mode = convert_array_like(mode)
        right = convert_array_like(right)
        if static.len(gather_arrays((left, mode, right))) == 0:
            fl = float(left)
            fm = float(mode)
            fr = float(right)
            if fl > fm:
                raise ValueError("left > mode")
            if fm > fr:
                raise ValueError("mode > right")
            if fl == fr:
                raise ValueError("left == right")
        else:
            oleft = asarray(left)
            omode = asarray(mode)
            oright = asarray(right)
            if (oleft > omode).any():
                raise ValueError("left > mode")
            if (omode > oright).any():
                raise ValueError("mode > right")
            if (oleft == oright).any():
                raise ValueError("left == right")
        return cont(self._triangular, size, self.lock, (left, mode, right), ('', '', ''),
                    (CONS_NONE, CONS_NONE, CONS_NONE))

    def binomial(self, n, p, size = None):
        n = convert_array_like(n)
        p = convert_array_like(p)
        if static.len(gather_arrays((n, p))) == 0:
            check_array_constraint(float(p), 'p', CONS_BOUNDED_0_1)
            check_array_constraint(int(n), 'n', CONS_NON_NEGATIVE)
        return cont(self._l_binomial, size, self.lock, (p, n), ('p', 'n'),
                    (CONS_BOUNDED_0_1, CONS_NON_NEGATIVE), dtype=int)

    def negative_binomial(self, n, p, size = None):
        return cont(self._l_negative_binomial, size, self.lock, (n, p), ('n', 'p'),
                    (CONS_POSITIVE, CONS_BOUNDED_0_1), dtype=int)

    def poisson(self, lam = 1.0, size = None):
        return cont(self._l_poisson, size, self.lock, (lam,), ('lam',), (CONS_POISSON,), dtype=int)

    def zipf(self, a, size = None):
        return cont(self._l_zipf, size, self.lock, (a,), ('a',), (CONS_GT_1,), dtype=int)

    def geometric(self, p, size = None):
        return cont(self._l_geometric, size, self.lock, (p,), ('p',), (CONS_BOUNDED_GT_0_1,), dtype=int)

    def logseries(self, p, size = None):
        return cont(self._l_logseries, size, self.lock, (p,), ('p',), (CONS_BOUNDED_LT_0_1,), dtype=int)

    def hypergeometric(self, ngood, nbad, nsample, size = None):
        ngood = convert_array_like(ngood)
        nbad = convert_array_like(nbad)
        nsample = convert_array_like(nsample)
        if static.len(gather_arrays((ngood, nbad, nsample))) == 0:
            if int(ngood) + int(nbad) < int(nsample):
                raise ValueError("ngood + nbad < nsample")
        else:
            if (asarray(ngood) + asarray(nbad) < asarray(nsample)).any():
                raise ValueError("ngood + nbad < nsample")
        return cont(self._l_hypergeometric, size, self.lock, (ngood, nbad, nsample),
                    ('ngood', 'nbad', 'nsample'),
                    (CONS_NON_NEGATIVE, CONS_NON_NEGATIVE, CONS_GTE_1), dtype=int)

    # integers (legacy masked path of _bounded_integers)

    def _bounded1(self, off: U, rng: U, bcnt: int, buf: u32, U: type):
        bg = self._bitgen
        if U is u64:
            mask = BitGenerator.gen_mask(rng)
            return bg.random_bounded_uint64(off, rng, mask, True), bcnt, buf
        elif U is u32:
            mask = itrunc(BitGenerator.gen_mask(zext(rng, u64)), u32)
            return bg.random_buffered_bounded_uint32(off, rng, mask, True, bcnt, buf)
        elif U is u16:
            mask = itrunc(BitGenerator.gen_mask(zext(rng, u64)), u16)
            return bg.random_buffered_bounded_uint16(off, rng, mask, True, bcnt, buf)
        elif U is u8:
            mask = itrunc(BitGenerator.gen_mask(zext(rng, u64)), u8)
            return bg.random_buffered_bounded_uint8(off, rng, mask, True, bcnt, buf)
        else:
            return bg.buffered_bounded_bool(off, rng, False, bcnt, buf)

    def _randint(self, low, high, size, dtype: type, closed: bool):
        if dtype is u64:
            lb = 0
            ub = 0x7FFFFFFFFFFFFFFF
            fn = self._bitgen.random_bounded_uint64_fill
            out_val = u64()
        elif dtype is u32:
            lb = 0
            ub = 0xFFFFFFFF
            fn = self._bitgen.random_bounded_uint32_fill
            out_val = u32()
        elif dtype is u16:
            lb = 0
            ub = 0xFFFF
            fn = self._bitgen.random_bounded_uint16_fill
            out_val = u16()
        elif dtype is u8:
            lb = 0
            ub = 0xFF
            fn = self._bitgen.random_bounded_uint8_fill
            out_val = u8()
        elif dtype is bool:
            lb = 0
            ub = 1
            fn = self._bitgen.random_bounded_bool_fill
            out_val = False
        elif dtype is i64 or dtype is int:
            lb = -9223372036854775808
            ub = 0x7FFFFFFFFFFFFFFF
            fn = self._bitgen.random_bounded_uint64_fill
            out_val = u64()
        elif dtype is i32:
            lb = -0x80000000
            ub = 0x7FFFFFFF
            fn = self._bitgen.random_bounded_uint32_fill
            out_val = u32()
        elif dtype is i16:
            lb = -0x8000
            ub = 0x7FFF
            fn = self._bitgen.random_bounded_uint16_fill
            out_val = u16()
        elif dtype is i8:
            lb = -0x80
            ub = 0x7F
            fn = self._bitgen.random_bounded_uint8_fill
            out_val = u8()
        else:
            compile_error("Unsupported dtype " + dtype.__name__ + " for randint")

        U = type(out_val)
        low = convert_array_like(low)
        high = convert_array_like(high)

        if static.len(gather_arrays((low, high))) == 0:
            ilow = int(low)
            ihigh = int(high)
            if not closed:
                ihigh -= 1
            if ilow < lb:
                raise ValueError("low is out of bounds for " + dtype.__name__)
            if ihigh > ub:
                raise ValueError("high is out of bounds for " + dtype.__name__)
            if ilow > ihigh:
                raise ValueError(_bounds_error(closed, ilow == 0))
            rng = U(ihigh - ilow)
            off = U(ilow)
            if size is None:
                with self.lock:
                    fn(off, rng, 1, True, __ptr__(out_val))
                return util.cast(out_val, dtype)
            else:
                out_arr = empty(_size_shape(size), dtype)
                with self.lock:
                    fn(off, rng, out_arr.size, True, Ptr[U](out_arr.data.as_byte()))
                return out_arr
        else:
            alow = asarray(low)
            ahigh = asarray(high)
            for v in alow.flat:
                if int(v) < lb:
                    raise ValueError("low is out of bounds for " + dtype.__name__)
            for v in ahigh.flat:
                if int(v) - (0 if closed else 1) > ub:
                    raise ValueError("high is out of bounds for " + dtype.__name__)
            if size is None:
                bshape = broadcast_shapes(alow.shape, ahigh.shape)
            else:
                bshape = broadcast_shapes(alow.shape, ahigh.shape, _size_shape(size))
            lowzero = True
            for v in alow.flat:
                if int(v) != 0:
                    lowzero = False
            for idx in util.multirange(bshape):
                e_low = int(alow._ptr(idx, broadcast=True)[0])
                e_high = int(ahigh._ptr(idx, broadcast=True)[0]) - (0 if closed else 1)
                if e_low > e_high:
                    raise ValueError(_bounds_error(closed, lowzero))
            if size is None:
                out_arr = empty(bshape, dtype)
            else:
                out_arr = empty(_size_shape(size), dtype)
            bcnt = 0
            buf = u32(0)
            with self.lock:
                for idx in util.multirange(out_arr.shape):
                    e_low = int(alow._ptr(idx, broadcast=True)[0])
                    e_high = int(ahigh._ptr(idx, broadcast=True)[0]) - (0 if closed else 1)
                    rng = U(e_high - e_low)
                    off = U(e_low)
                    val, bcnt, buf = self._bounded1(off, rng, bcnt, buf)
                    out_arr._ptr(idx)[0] = util.cast(val, dtype)
            return out_arr

    def randint(self, low, high = None, size = None, dtype: type = int):
        if high is None:
            return self._randint(0, low, size, dtype, False)
        else:
            return self._randint(low, high, size, dtype, False)

    def random_integers(self, low, high = None, size = None):
        if high is None:
            return self._randint(1, low, size, int, True)
        else:
            return self._randint(low, high, size, int, True)

    def _maxint(self) -> int:
        return int(self._bitgen.next64() >> u64(1))

    def tomaxint(self, size = None):
        return cont(self._maxint, size, self.lock, (), (), (), dtype=int)

    def bytes(self, length: int):
        if length < 0:
            raise ValueError("length must be non-negative")
        n_uint32 = (length - 1) // 4 + 1
        arr = self.randint(0, 4294967296, size=n_uint32, dtype=u32)
        return str(arr.data.as_byte(), length)

    # sequences

    def shuffle(self, x):
        n = len(x)
        if isinstance(x, ndarray):
            ndim: Literal[int] = static.len(x.shape)
            if x.size == 0:
                return
            if ndim == 1:
                stride = x.strides[0]
                itemsize = x.itemsize
                data = x.data.as_byte()
                buf = cobj(itemsize)
                with self.lock:
                    for i in range(n - 1, 0, -1):
                        j = self._interval(i)
                        str.memcpy(buf, data + j * stride, itemsize)
                        str.memcpy(data + j * stride, data + i * stride, itemsize)
                        str.memcpy(data + i * stride, buf, itemsize)
            else:
                buf = empty_like(x[0, ...])
                with self.lock:
                    for i in range(n - 1, 0, -1):
                        j = self._interval(i)
                        if i == j:
                            continue
                        buf[...] = x[j, ...]
                        x[j, ...] = x[i, ...]
                        x[i, ...] = buf
        else:
            with self.lock:
                for i in range(n - 1, 0, -1):
                    j = self._interval(i)
                    x[i], x[j] = x[j], x[i]

    def permutation(self, x):
        if isinstance(x, int):
            arr = arange(x)
            self.shuffle(arr)
            return arr
        else:
            a = asarray(x)
            if static.len(a.shape) == 0:
                compile_error("x must be an integer or at least 1-dimensional")
            arr = a.copy()
            if static.len(a.shape) == 1:
                self.shuffle(arr)
                return arr
            idx = arange(arr.shape[0])
            self.shuffle(idx)
            out = empty_like(arr)
            for i in range(arr.shape[0]):
                out[i, ...] = arr[idx[i], ...]
            return out

    def choice(self, a, size = None, replace: bool = True, p = None):
        pop_size = 0
        if isinstance(a, int):
            pop_size = a
        else:
            arr0 = asarray(a)
            if static.len(arr0.shape) != 1:
                compile_error("a must be 1-dimensional or an integer")
            pop_size = arr0.shape[0]

        shape = _size_shape(size)
        n = util.count(shape)
        if isinstance(a, int):
            if pop_size <= 0 and n != 0:
                raise ValueError("a must be greater than 0 unless no samples are taken")
        else:
            if pop_size == 0 and n != 0:
                raise ValueError("'a' cannot be empty unless no samples are taken")

        pv = List[float]()
        if p is not None:
            parr = asarray(p).astype(float)
            if static.len(parr.shape) != 1:
                compile_error("'p' must be 1-dimensional")
            if parr.size != pop_size:
                raise ValueError("'a' and 'p' must have same size")
            for v in parr:
                pv.append(v)
            p_sum = kahan_sum(parr.data, len(pv)) if parr._contig[0] else kahan_sum(ascontiguousarray(parr).data, len(pv))
            if isnan(p_sum):
                raise ValueError("probabilities contain NaN")
            for v in pv:
                if v < 0:
                    raise ValueError("probabilities are not non-negative")
            if abs(p_sum - 1.) > sqrt(2.220446049250313e-16):
                raise ValueError("probabilities do not sum to 1")

        def cdf_of(pv: List[float]):
            cdf = List[float](len(pv))
            acc = 0.0
            for v in pv:
                acc += v
                cdf.append(acc)
            last = cdf[-1]
            for i in range(len(cdf)):
                cdf[i] /= last
            return cdf

        def search_right(cdf: List[float], u: float):
            lo = 0
            hi = len(cdf)
            while lo < hi:
                mid = (lo + hi) >> 1
                if cdf[mid] <= u:
                    lo = mid + 1
                else:
                    hi = mid
            return lo

        idx = List[int](n)
        if replace:
            if p is not None:
                cdf = cdf_of(pv)
                if size is None:
                    idx.append(search_right(cdf, self.random_sample()))
                else:
                    us = self.random_sample(shape)
                    for i in range(n):
                        idx.append(search_right(cdf, us.data[i]))
            else:
                if size is None:
                    idx.append(self.randint(0, pop_size))
                else:
                    r = self.randint(0, pop_size, size=shape)
                    for i in range(n):
                        idx.append(r.data[i])
        else:
            if n > pop_size:
                raise ValueError("Cannot take a larger sample than population when 'replace=False'")
            if p is not None:
                nz = 0
                for v in pv:
                    if v > 0:
                        nz += 1
                if nz < n:
                    raise ValueError("Fewer non-zero entries in p than size")
                pc = pv.copy()
                n_uniq = 0
                while n_uniq < n:
                    x = self.random_sample((n - n_uniq,))
                    if n_uniq > 0:
                        for j in range(n_uniq):
                            pc[idx[j]] = 0.0
                    cdf = cdf_of(pc)
                    seen = Set[int]()
                    for i in range(x.size):
                        v = search_right(cdf, x.data[i])
                        if v not in seen:
                            seen.add(v)
                            idx.append(v)
                    n_uniq = len(idx)
            else:
                perm = self.permutation(pop_size)
                for i in range(n):
                    idx.append(perm.data[i])

        if isinstance(a, int):
            if size is None:
                return idx[0]
            else:
                out = empty(shape, int)
                for i in range(n):
                    out.data[i] = idx[i]
                return out
        else:
            arr = asarray(a)
            if size is None:
                return arr[idx[0]]
            else:
                out = empty(shape, arr.dtype)
                for i in range(n):
                    out.data[i] = arr[idx[i]]
                return out

    # multivariate

    def multinomial(self, n: int, pvals, size = None):
        parr = ascontiguousarray(asarray(pvals), dtype=float)
        if static.len(parr.shape) != 1:
            compile_error("pvals must be a 1-d sequence")
        d = parr.size
        pix = parr.data
        check_array_constraint(parr, 'pvals', CONS_BOUNDED_0_1)
        if d and kahan_sum(pix, d - 1) > (1.0 + 1e-12):
            raise ValueError("sum(pvals[:-1]) > 1.0")
        shape = _size_shape(size) + (d,)
        multin = zeros(shape, dtype=int)
        mnix = multin.data
        sz = multin.size
        check_array_constraint(n, 'n', CONS_NON_NEGATIVE)
        offset = 0
        niter = sz // d if d else 0
        with self.lock:
            for _ in range(niter):
                self._l_multinomial(n, mnix + offset, pix, d)
                offset += d
        return multin

    def dirichlet(self, alpha, size = None):
        alpha_arr = ascontiguousarray(asarray(alpha), dtype=float)
        if static.len(alpha_arr.shape) != 1:
            compile_error("alpha must be a one-dimensional sequence of floats")
        k = alpha_arr.size
        for v in alpha_arr:
            if v <= 0:
                raise ValueError('alpha <= 0')
        alpha_data = alpha_arr.data
        shape = _size_shape(size) + (k,)
        diric = zeros(shape, dtype=float)
        val_data = diric.data
        totsize = diric.size
        i = 0
        with self.lock:
            while i < totsize:
                acc = 0.0
                for j in range(k):
                    val_data[i + j] = self._std_gamma(alpha_data[j])
                    acc = acc + val_data[i + j]
                invacc = 1 / acc
                for j in range(k):
                    val_data[i + j] = val_data[i + j] * invacc
                i = i + k
        return diric

    def multivariate_normal(self, mean, cov, size = None, check_valid: Literal[str] = 'warn',
                            tol: float = 1e-8):
        from ..linalg import svd, _linalg_dot
        mean = asarray(mean).astype(float)
        cov = asarray(cov).astype(float)
        if static.len(mean.shape) != 1:
            compile_error("mean must be 1 dimensional")
        if static.len(cov.shape) != 2:
            compile_error("cov must be 2 dimensional and square")
        if cov.shape[0] != cov.shape[1]:
            raise ValueError("cov must be 2 dimensional and square")
        if mean.shape[0] != cov.shape[0]:
            raise ValueError("mean and cov must have same length")
        nd = mean.shape[0]
        final_shape = _size_shape(size) + (nd,)
        x = self.standard_normal(final_shape).reshape(-1, nd)
        (_u, s, v) = svd(cov)
        if check_valid != 'ignore':
            if check_valid != 'warn' and check_valid != 'raise':
                compile_error("check_valid must equal 'warn', 'raise', or 'ignore'")
            psd = allclose(_linalg_dot(v.T * s, v), cov, rtol=tol, atol=tol)
            if not psd and check_valid == 'raise':
                raise ValueError("covariance is not symmetric positive-semidefinite.")
        fac = v.copy()
        for i in range(nd):
            sqi = sqrt(s[i])
            for j in range(nd):
                fac[i, j] = sqi * v[i, j]
        x = _linalg_dot(x, fac)
        x += mean
        return x.reshape(final_shape)


def check_random_state(seed = None):
    """sklearn.utils.check_random_state: None -> global legacy stream,
    int -> new RandomState(seed), RandomState -> itself."""
    if isinstance(seed, NoneType):
        return _rand
    elif isinstance(seed, RandomState):
        return seed
    elif isinstance(seed, int):
        return RandomState(seed)
    elif isinstance(seed, Optional):
        if seed is None:
            return _rand
        return check_random_state(seed.__val__())
    else:
        compile_error("random_state must be None, an int or a RandomState")

_rand = RandomState()
