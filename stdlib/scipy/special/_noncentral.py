# Ported from scipy 1.18.1 (BSD-3-Clause)
# Algorithms from Boost.Math (BSL-1.0) non_central_{chi_squared,beta,t,f}
# as wrapped by scipy (boost_special_functions.h).
#
# Scalar kernels: chndtr, chndtrix, nctdtr, nctdtrit, ncfdtr, ncfdtri,
# plus helper noncentral-beta CDF.  Root inversion uses xsf-style bracketing
# and Chandrupatla (to full double precision) instead of Boost's TOMS748.

from . import _kernels as K
from ._kernels import exp, log, sqrt, fabs, pow
from . import _tools as T

NAN = float('nan')
INF = float('inf')
EPS = 2.220446049250313e-16
DBL_MIN = 2.2250738585072014e-308
MAX_ITER = 1000000
SQRT2 = 1.41421356237309504880

def _isnan(x: float) -> bool:
    return x != x

def _isinf(x: float) -> bool:
    return x == INF or x == -INF

def _gamma_p_derivative(a: float, x: float) -> float:
    # x^(a-1) e^-x / Gamma(a)
    if x == 0:
        return 0.0 if a > 1 else (1.0 if a == 1 else INF)
    return K._igam_fac(a, x) / x

def _ibeta_deriv_term(a: float, b: float, x: float, y: float) -> float:
    # x^(a-1) y^(b-1) / B(a, b)  (Boost ibeta_imp p_derivative)
    if x == 0 or y == 0:
        return 0.0
    return exp((a - 1) * log(x) + (b - 1) * log(y) - K.betaln(a, b))

def _ibeta(a: float, b: float, x: float, y: float, inv: bool) -> float:
    # regularized I_x(a, b) (inv: complement)
    if inv:
        return K.betaincc(a, b, x)
    return K.betainc(a, b, x)

def _llround(x: float) -> int:
    return int(K.floor(x + 0.5))

# ------------------------------------------------------------------ noncentral chi2
def _ncx2_q(x: float, f: float, theta: float, init_sum: float) -> float:
    if x == 0:
        return 1.0
    lam = theta / 2
    de = f / 2
    y = x / 2
    errtol = EPS
    s = init_sum
    k = _llround(lam)
    poisf = _gamma_p_derivative(float(1 + k), lam)
    poisb = poisf * k / lam
    gamf = K.gammaincc(de + k, y)
    xtermf = _gamma_p_derivative(de + 1 + k, y)
    xtermb = xtermf * (de + k) / y
    gamb = gamf - xtermb
    i = k
    while i - k < MAX_ITER:
        term = poisf * gamf
        s += term
        poisf *= lam / (i + 1)
        gamf += xtermf
        xtermf *= y / (de + i + 1)
        if (s == 0 or fabs(term / s) < errtol) and term >= poisf * gamf:
            break
        i += 1
    i = k - 1
    while i >= 0:
        term = poisb * gamb
        s += term
        poisb *= i / lam
        xtermb *= (de + i) / y
        gamb -= xtermb
        if s == 0 or fabs(term / s) < errtol:
            break
        i -= 1
    return s

def _ncx2_p_ding(x: float, f: float, theta: float, init_sum: float) -> float:
    if x == 0:
        return 0.0
    tk = _gamma_p_derivative(f / 2 + 1, x / 2)
    lam = theta / 2
    vk = exp(-lam)
    uk = vk
    s = init_sum + tk * vk
    if s == 0:
        return s
    lterm = 0.0
    term = 0.0
    for i in range(1, MAX_ITER):
        tk = tk * x / (f + 2 * i)
        uk = uk * lam / i
        vk = vk + uk
        lterm = term
        term = vk * tk
        s += term
        if fabs(term / s) < EPS and term <= lterm:
            break
    return s

def _ncx2_p(y: float, n: float, lam: float, init_sum: float) -> float:
    if y == 0:
        return 0.0
    errorf = 0.0
    errorb = 0.0
    x = y / 2
    de = lam / 2
    k = _llround(de)
    a = n / 2 + k
    gamkf = K.gammainc(a, x)
    if lam == 0:
        return gamkf
    gamkb = gamkf
    poiskf = _gamma_p_derivative(float(k + 1), de)
    poiskb = poiskf
    xtermf = _gamma_p_derivative(a, x)
    xtermb = xtermf * x / a
    s = init_sum + poiskf * gamkf
    if s == 0:
        return s
    i = 1
    while i <= k:
        xtermb *= (a - i + 1) / x
        gamkb += xtermb
        poiskb = poiskb * (k - i + 1) / de
        errorf = errorb
        errorb = gamkb * poiskb
        s += errorb
        if fabs(errorb / s) < EPS and errorb <= errorf:
            break
        i += 1
    i = 1
    while True:
        xtermf = xtermf * x / (a + i - 1)
        gamkf = gamkf - xtermf
        poiskf = poiskf * de / (k + i)
        errorf = poiskf * gamkf
        s += errorf
        i += 1
        if not (fabs(errorf / s) > EPS and i < MAX_ITER):
            break
    return s

def _ncx2_cdf(x: float, k: float, l: float, invert: bool) -> float:
    result = 0.0
    if l == 0:
        return K.chdtrc(k, x) if invert else K.chdtr(k, x)
    elif x > k + l:
        result = _ncx2_q(x, k, l, 0.0 if invert else -1.0)
        invert = not invert
    elif l < 200:
        result = _ncx2_p_ding(x, k, l, -1.0 if invert else 0.0)
    else:
        result = _ncx2_p(x, k, l, -1.0 if invert else 0.0)
    if invert:
        result = -result
    return result

def chndtr(x: float, k: float, l: float) -> float:
    if _isnan(x) or _isnan(k) or _isnan(l):
        return NAN
    if k < 0 or x < 0 or l < 0:
        return NAN
    if l == 0 and (k == 0 or _isinf(x)):
        return 1.0
    if l == 0 and _isinf(k):
        return 0.0
    if k == 0 or (not _isinf(l) and _isinf(k)):
        return NAN
    if _isinf(l):
        return NAN if _isinf(x) else 0.0
    if _isinf(x) and not _isinf(k):
        return 1.0
    return _ncx2_cdf(x, k, l, False)

def chndtrc(x: float, k: float, l: float) -> float:
    """Survival function of the noncentral chi2 (Boost complement cdf)."""
    if _isnan(x) or _isnan(k) or _isnan(l):
        return NAN
    if k <= 0 or x < 0 or l < 0:
        return NAN
    if _isinf(x):
        return 0.0
    return _ncx2_cdf(x, k, l, True)

class _CdfRoot:
    # f(x) = cdf(x) - p  (or q - sf(x) when comp)
    kind: int
    a: float
    b: float
    c: float
    target: float
    comp: bool
    def __init__(self, kind: int, a: float, b: float, c: float, target: float, comp: bool):
        self.kind = kind
        self.a = a
        self.b = b
        self.c = c
        self.target = target
        self.comp = comp
    def cdf(self, x: float, comp: bool) -> float:
        if self.kind == 0:
            return _ncx2_cdf(x, self.a, self.b, comp)
        elif self.kind == 1:
            return _nct_cdf(self.a, self.b, x, comp)
        else:
            return _ncf_cdf(x, self.a, self.b, self.c, comp)
    def __call__(self, x: float) -> float:
        if self.comp:
            return self.target - self.cdf(x, True)
        return self.cdf(x, False) - self.target

def _solve_increasing(f: _CdfRoot, guess: float, lo: float, hi: float) -> float:
    # bracket by doubling steps around guess, then Chandrupatla
    x0 = guess
    f0 = f(x0)
    if f0 == 0:
        return x0
    step = max(fabs(x0) * 0.5, 0.5)
    xl = x0
    xr = x0
    fl = f0
    fr = f0
    if f0 < 0:
        for _ in range(2000):
            xr = x0 + step
            if xr >= hi:
                xr = hi
            fr = f(xr)
            if fr >= 0 or xr == hi:
                break
            x0 = xr
            f0 = fr
            step *= 2
        xl = x0
        fl = f0
        if fr < 0:
            return INF if hi == INF or xr == hi else NAN
    else:
        for _ in range(2000):
            xl = x0 - step
            if xl <= lo:
                xl = lo
            fl = f(xl)
            if fl <= 0 or xl == lo:
                break
            x0 = xl
            f0 = fl
            step *= 2
        xr = x0
        fr = f0
        if fl > 0:
            return lo
    r = T.chandrupatla(f, xl, xr, fl, fr, EPS, 1e-300, 200)
    return r[0]

def chndtrix(p: float, k: float, l: float) -> float:
    if _isnan(p) or _isnan(k) or _isnan(l):
        return NAN
    if k < 0 or p < 0 or l < 0 or p > 1:
        return NAN
    if k <= 0:
        return NAN
    if p == 0:
        return 0.0
    if p == 1:
        return INF
    b = -(l * l) / (k + 3 * l)
    c = (k + 3 * l) / (k + 2 * l)
    ff = (k + 2 * l) / (c * c)
    guess = b + c * K.chdtri(ff, 1.0 - p) if ff > 0 else 1.0
    if not (guess >= 0.005):
        pp = p
        guess = pow(pow(2.0, (k / 2 - 1)) * exp(l / 2) * pp * k * K.gamma(k / 2), (2 / k))
        if guess == 0 or _isnan(guess):
            guess = DBL_MIN
    comp = p > 0.5
    f = _CdfRoot(0, k, l, 0.0, 1.0 - p if comp else p, comp)
    return _solve_increasing(f, guess, 0.0, INF)

# ------------------------------------------------------------------ noncentral beta
def _ncbeta_p(a: float, b: float, lam: float, x: float, y: float, init_val: float) -> float:
    l2 = lam / 2
    k = int(l2)
    if k == 0:
        k = 1
    pois = _gamma_p_derivative(float(k + 1), l2)
    if pois == 0:
        return init_val
    beta = _ibeta(a + k, b, x, y, False)
    xterm = _ibeta_deriv_term(a + k, b, x, y)
    while fabs(beta * pois) < DBL_MIN:
        if k == 0 or pois == 0:
            return init_val
        k //= 2
        pois = _gamma_p_derivative(float(k + 1), l2)
        beta = _ibeta(a + k, b, x, y, False)
        xterm = _ibeta_deriv_term(a + k, b, x, y)
    xterm *= y / (a + b + k - 1)
    poisf = pois
    betaf = beta
    xtermf = xterm
    s = init_val
    if beta == 0 and xterm == 0:
        return init_val
    last_term = 0.0
    count = k
    i = k
    while i >= 0:
        term = beta * pois
        s += term
        if (fabs(term / s) < EPS and fabs(last_term) >= fabs(term)) or term == 0:
            count = k - i
            break
        pois *= i / l2
        beta += xterm
        if a + b + i != 2:
            xterm *= (a + i - 1) / (x * (a + b + i - 2))
        last_term = term
        i -= 1
    last_term = 0.0
    betaf_lim = betaf * EPS * 4
    i = k + 1
    while True:
        poisf *= l2 / i
        xtermf *= (x * (a + b + i - 2)) / (a + i - 1)
        betaf -= xtermf
        if betaf < betaf_lim:
            break
        term = poisf * betaf
        s += term
        if (fabs(term / s) < EPS and fabs(last_term) >= fabs(term)) or term == 0:
            break
        last_term = term
        if count + i - k > MAX_ITER:
            return NAN
        i += 1
    return s

def _ncbeta_q(a: float, b: float, lam: float, x: float, y: float, init_val: float) -> float:
    l2 = lam / 2
    k = int(l2)
    if k <= 30:
        if a + b > 1:
            k = 0
        elif k == 0:
            k = 1
    pois = exp(-l2) if k == 0 else _gamma_p_derivative(float(k + 1), l2)
    if pois == 0:
        return init_val
    beta = _ibeta(a + k, b, x, y, True)
    xterm = _ibeta_deriv_term(a + k, b, x, y)
    xterm *= y / (a + b + k - 1)
    poisf = pois
    betaf = beta
    xtermf = xterm
    s = init_val
    if beta == 0 and xterm == 0:
        return init_val
    last_term = 0.0
    count = 0
    i = k + 1
    while True:
        poisf *= l2 / i
        xtermf *= (x * (a + b + (i - 2))) / (a + (i - 1))
        betaf += xtermf
        term = poisf * betaf
        s += term
        if fabs(term / s) < EPS and last_term >= term:
            count = i - k
            break
        if i - k > MAX_ITER:
            return NAN
        last_term = term
        i += 1
    i = k
    while i >= 0:
        term = beta * pois
        s += term
        if fabs(term / s) < EPS:
            break
        pois *= i / l2
        beta -= xterm
        if a + b + i - 2 != 0:
            xterm *= (a + i - 1) / (x * (a + b + i - 2))
        i -= 1
    return s

def _ncbeta_cdf(x: float, y: float, a: float, b: float, l: float, invert: bool) -> float:
    if x == 0:
        return 1.0 if invert else 0.0
    if y == 0:
        return 0.0 if invert else 1.0
    result = 0.0
    c = a + b + l / 2
    cross = 1 - (b / c) * (1 + l / (2 * c * c))
    if l == 0:
        result = _ibeta(a, b, x, y, False)
    elif x > cross:
        result = _ncbeta_q(a, b, l, x, y, 0.0 if invert else -1.0)
        invert = not invert
    else:
        result = _ncbeta_p(a, b, l, x, y, -1.0 if invert else 0.0)
    if invert:
        result = -result
    return result

# ------------------------------------------------------------------ noncentral F
def _ncf_cdf(x: float, v1: float, v2: float, l: float, comp: bool) -> float:
    alpha = v1 / 2
    beta = v2 / 2
    y = x * alpha / beta
    c = y / (1 + y)
    cp = 1 / (1 + y)
    return _ncbeta_cdf(c, cp, alpha, beta, l, comp)

def ncfdtr(v1: float, v2: float, l: float, x: float) -> float:
    if _isnan(x) or _isnan(v1) or _isnan(v2) or _isnan(l):
        return NAN
    if v1 <= 0 or v2 <= 0 or l < 0 or x < 0:
        return NAN
    if _isinf(x):
        return 1.0
    y = _ncf_cdf(x, v1, v2, l, False)
    if y < 0 or y > 1:
        return NAN
    return y

def ncfdtri(v1: float, v2: float, l: float, p: float) -> float:
    if _isnan(p) or _isnan(v1) or _isnan(v2) or _isnan(l):
        return NAN
    if v1 <= 0 or v2 <= 0 or l < 0 or p < 0 or p > 1:
        return NAN
    if p == 0:
        return 0.0
    if p == 1:
        return INF
    # Boost: quantile of the noncentral beta, then map back
    comp = p > 0.5
    f = _CdfRoot(2, v1, v2, l, 1.0 - p if comp else p, comp)
    guess = 1.0 + l / v1
    return _solve_increasing(f, guess, 0.0, INF)

# ------------------------------------------------------------------ noncentral t
def _nct2_p(v: float, delta: float, x: float, y: float, init_val: float) -> float:
    d2 = delta * delta / 2
    k = int(d2)
    if k == 0:
        k = 1
    pois = _gamma_p_derivative(float(k + 1), d2) * K._tgamma_delta_ratio_half(float(k + 1)) * delta / SQRT2
    if pois == 0:
        return init_val
    beta = _ibeta(float(k + 1), v / 2, x, y, False)
    xterm = _ibeta_deriv_term(float(k + 1), v / 2, x, y)
    while fabs(beta * pois) < DBL_MIN:
        if k == 0 or pois == 0:
            return init_val
        k //= 2
        pois = _gamma_p_derivative(float(k + 1), d2) * K._tgamma_delta_ratio_half(float(k + 1)) * delta / SQRT2
        beta = _ibeta(float(k + 1), v / 2, x, y, False)
        xterm = _ibeta_deriv_term(float(k + 1), v / 2, x, y)
    xterm *= y / (v / 2 + k)
    poisf = pois
    betaf = beta
    xtermf = xterm
    s = init_val
    if xterm == 0 and beta == 0:
        return init_val
    count = 0
    last_term = 0.0
    i = k
    while i >= 0:
        term = beta * pois
        s += term
        if (fabs(last_term) > fabs(term) and fabs(term / s) < EPS) or (v == 2 and i == 0):
            break
        last_term = term
        pois *= (i + 0.5) / d2
        beta += xterm
        xterm *= i / (x * (v / 2 + i - 1))
        count += 1
        i -= 1
    last_term = 0.0
    betaf_lim = betaf * EPS * 4
    i = k + 1
    while True:
        poisf *= d2 / (i + 0.5)
        xtermf *= (x * (v / 2 + i - 1)) / i
        betaf -= xtermf
        if betaf < betaf_lim:
            break
        term = poisf * betaf
        s += term
        if fabs(last_term) >= fabs(term) and fabs(term / s) < EPS:
            break
        last_term = term
        count += 1
        if count > MAX_ITER:
            return NAN
        i += 1
    return s

def _nct2_q(v: float, delta: float, x: float, y: float, init_val: float) -> float:
    d2 = delta * delta / 2
    k = int(d2)
    if k == 0:
        k = 1
    pois = 0.0
    if k < 170 and d2 < 709.78 and log(d2) * k < 709.78:
        pois = exp(-d2)
        pois *= pow(d2, float(k))
        pois /= K.gamma(k + 1 + 0.5)
        pois *= delta / SQRT2
    else:
        pois = _gamma_p_derivative(float(k + 1), d2) * K._tgamma_delta_ratio_half(float(k + 1)) * delta / SQRT2
    if pois == 0:
        return init_val
    xterm = 0.0
    beta = 0.0
    if k != 0:
        beta = _ibeta(float(k + 1), v / 2, x, y, True)
        xterm = _ibeta_deriv_term(float(k + 1), v / 2, x, y)
        xterm *= y / (v / 2 + k)
    else:
        beta = pow(y, v / 2)
        xterm = beta
    poisf = pois
    betaf = beta
    xtermf = xterm
    s = init_val
    if xterm == 0 and beta == 0:
        return init_val
    count = 0
    last_term = 0.0
    i = k + 1
    j = k
    while True:
        poisf *= d2 / (i + 0.5)
        xtermf *= (x * (v / 2 + i - 1)) / i
        betaf += xtermf
        term = poisf * betaf
        if j >= 0:
            term += beta * pois
            pois *= (j + 0.5) / d2
            beta -= xterm
            if not (v == 2 and j == 0):
                xterm *= j / (x * (v / 2 + j - 1))
        s += term
        if fabs(last_term) > fabs(term) and fabs(term / s) < EPS:
            break
        last_term = term
        if count > MAX_ITER:
            return NAN
        count += 1
        i += 1
        j -= 1
    return s

def _nct_cdf(v: float, delta: float, t: float, invert: bool) -> float:
    if _isinf(v):
        r = K.ndtr(t - delta)
        return 1 - r if invert else r
    if t < 0:
        t = -t
        delta = -delta
        invert = not invert
    if fabs(delta / (4 * v)) < EPS:
        result = K.stdtr(v, t - delta)
        return 1 - result if invert else result
    x = t * t / (v + t * t)
    y = v / (v + t * t)
    d2 = delta * delta
    a = 0.5
    b = v / 2
    c = a + b + d2 / 2
    cross = 1 - (b / c) * (1 + d2 / (2 * c * c))
    result = 0.0
    if x < cross:
        if x != 0:
            result = _ncbeta_p(a, b, d2, x, y, 0.0)
            result = _nct2_p(v, delta, x, y, result)
            result /= 2
        else:
            result = 0.0
        if invert:
            result = K.ndtr(delta) - result
            invert = False
        else:
            result += K.ndtr(-delta)
    else:
        invert = not invert
        if x != 0:
            result = _ncbeta_q(a, b, d2, x, y, 0.0)
            result = _nct2_q(v, delta, x, y, result)
            result /= 2
        else:
            result = K.ndtr(delta)
    if invert:
        result = 1 - result
    return result

def nctdtr(v: float, l: float, x: float) -> float:
    if _isnan(x) or _isnan(v) or _isnan(l):
        return NAN
    if v <= 0:
        return NAN
    if _isinf(x):
        return 1.0 if x > 0 else 0.0
    y = _nct_cdf(v, l, x, False)
    if y < 0 or y > 1:
        return NAN
    return y

def nctdtrit(v: float, l: float, p: float) -> float:
    if _isnan(p) or _isnan(v) or _isnan(l):
        return NAN
    if v <= 0 or p < 0 or p > 1:
        return NAN
    if p == 0:
        return -INF
    if p == 1:
        return INF
    q = 1.0 - p
    if _isinf(v) or v > 1 / EPS:
        return l + K.ndtri(p)
    guess = 0.0
    if v > 3:
        mean = l * sqrt(v / 2) * exp(K.gammaln((v - 1) * 0.5) - K.gammaln(v * 0.5))
        var = ((l * l + 1) * v) / (v - 2) - mean * mean
        guess = mean + sqrt(var) * K.ndtri(p) if var > 0 else mean
    comp = p >= q
    f = _CdfRoot(1, v, l, 0.0, q if comp else p, comp)
    return _solve_real(f, guess)

def _solve_real(f: _CdfRoot, guess: float) -> float:
    x0 = guess
    f0 = f(x0)
    if f0 == 0:
        return x0
    step = max(fabs(x0) * 0.5, 1.0)
    xl = x0
    xr = x0
    fl = f0
    fr = f0
    if f0 < 0:
        for _ in range(3000):
            xr = x0 + step
            fr = f(xr)
            if fr >= 0 or _isinf(xr):
                break
            x0 = xr
            f0 = fr
            step *= 2
        xl = x0
        fl = f0
    else:
        for _ in range(3000):
            xl = x0 - step
            fl = f(xl)
            if fl <= 0 or _isinf(xl):
                break
            x0 = xl
            f0 = fl
            step *= 2
        xr = x0
        fr = f0
    r = T.chandrupatla(f, xl, xr, fl, fr, EPS, 1e-300, 200)
    return r[0]
