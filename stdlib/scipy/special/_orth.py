# Ported from scipy 1.18.1 (BSD-3-Clause)
# Algorithms from scipy orthogonal_eval.pxd (BSD-3); recurrences after
# Abramowitz & Stegun 22.5 and Levrie & Piessens (1985).
#
# eval_* kernels.  ``*_l`` take an integer degree (scipy's 'p' loops, used
# for integer-dtype n); the plain versions take float n (hypergeometric).

from . import _kernels as K
from ._kernels import exp, log, sqrt, fabs, sin, pow
from . import _misc as M
from . import _hyp as H

NAN = float('nan')

def _isnan(x: float) -> bool:
    return x != x

# ------------------------------------------------------------------ Jacobi
def eval_jacobi(n: float, alpha: float, beta: float, x: float) -> float:
    if alpha == -1 and fabs(beta) == 1:
        if n == 0:
            return 1.0
        elif n == 1:
            return 0.5 * (1.0 + beta) * (x - 1.0)
        elif n > 1:
            return ((n + beta) / (2.0 * n)) * (x - 1) * eval_jacobi(n - 1, 1.0, beta, x)
    d = M.binom(n + alpha, n)
    return d * H.hyp2f1(-n, n + alpha + beta + 1, alpha + 1, 0.5 * (1 - x))

def eval_jacobi_l(n: int, alpha: float, beta: float, x: float) -> float:
    if n < 0:
        return eval_jacobi(float(n), alpha, beta, x)
    elif n == 0:
        return 1.0
    elif n == 1:
        return 0.5 * (2 * (alpha + 1) + (alpha + beta + 2) * (x - 1))
    elif alpha == -1 and fabs(beta) == 1:
        return ((n + beta) / (2.0 * n)) * (x - 1) * eval_jacobi(float(n - 1), 1.0, beta, x)
    d = (alpha + beta + 2) * (x - 1) / (2 * (alpha + 1))
    p = d + 1
    for kk in range(n - 1):
        k = kk + 1.0
        t = 2 * k + alpha + beta
        d = ((t * (t + 1) * (t + 2)) * (x - 1) * p + 2 * k * (k + beta) * (t + 2) * d) / (2 * (k + alpha + 1) * (k + alpha + beta + 1) * t)
        p = d + p
    return M.binom(n + alpha, float(n)) * p

def eval_sh_jacobi(n: float, p: float, q: float, x: float) -> float:
    return eval_jacobi(n, p - q, q - 1, 2 * x - 1) / M.binom(2 * n + p - 1, n)

def eval_sh_jacobi_l(n: int, p: float, q: float, x: float) -> float:
    return eval_jacobi_l(n, p - q, q - 1, 2 * x - 1) / M.binom(2.0 * n + p - 1, float(n))

# ------------------------------------------------------------------ Gegenbauer
def eval_gegenbauer_l(n: int, alpha: float, x: float) -> float:
    if _isnan(alpha) or _isnan(x):
        return NAN
    if alpha == 0.0:
        return 0.0
    if n < 0:
        return 0.0
    if n == 0:
        return 1.0
    if n == 1:
        return 2 * alpha * x
    sign = -1.0 if (x < 0 and n % 2 == 1) else 1.0
    x = fabs(x)
    if x < 1e-5:
        a = n // 2
        d = 1.0 if a % 2 == 0 else -1.0
        d /= K.beta(alpha, 1.0 + a)
        if n == 2 * a:
            d /= (a + alpha)
        else:
            d *= 2 * x
        p = 0.0
        for kk in range(a + 1):
            p += d
            d *= -4 * x * x * (a - kk) * (-a + alpha + kk + n) / ((n + 1 - 2 * a + 2 * kk) * (n + 2 - 2 * a + 2 * kk))
            if fabs(d) < 1e-20 * fabs(p):
                break
        return sign * p
    d = x - 1
    p = x
    for kk in range(n - 1):
        k = kk + 1.0
        d = (2 * (k + alpha) / (k + 2 * alpha)) * (x - 1) * p + (k / (k + 2 * alpha)) * d
        p = d + p
    if fabs(alpha / n) < 1e-8:
        return sign * 2 * alpha / n * p
    return sign * M.binom(n + 2 * alpha - 1, float(n)) * p

def eval_gegenbauer(n: float, alpha: float, x: float) -> float:
    if n >= 0 and n < 1e10 and n == float(int(n)):
        return eval_gegenbauer_l(int(n), alpha, x)
    d = K.gamma(n + 2 * alpha) / K.gamma(1 + n) / K.gamma(2 * alpha)
    return d * H.hyp2f1(-n, n + 2 * alpha, alpha + 0.5, (1 - x) / 2.0)

# ------------------------------------------------------------------ Chebyshev
def eval_chebyt(n: float, x: float) -> float:
    return H.hyp2f1(-n, n, 0.5, 0.5 * (1 - x))

def eval_chebyt_l(k: int, x: float) -> float:
    if k < 0:
        k = -k
    b2 = 0.0
    b1 = -1.0
    b0 = 0.0
    x = 2 * x
    for m in range(k + 1):
        b2 = b1
        b1 = b0
        b0 = x * b1 - b2
    return (b0 - b2) / 2.0

def eval_chebyu(n: float, x: float) -> float:
    return (n + 1) * H.hyp2f1(-n, n + 2, 1.5, 0.5 * (1 - x))

def eval_chebyu_l(k: int, x: float) -> float:
    sign = 1.0
    if k == -1:
        return 0.0
    elif k < -1:
        k = -k - 2
        sign = -1.0
    b2 = 0.0
    b1 = -1.0
    b0 = 0.0
    x = 2 * x
    for m in range(k + 1):
        b2 = b1
        b1 = b0
        b0 = x * b1 - b2
    return b0 * sign

def eval_chebys(n: float, x: float) -> float:
    return eval_chebyu(n, 0.5 * x)

def eval_chebys_l(n: int, x: float) -> float:
    return eval_chebyu_l(n, 0.5 * x)

def eval_chebyc(n: float, x: float) -> float:
    return 2 * eval_chebyt(n, 0.5 * x)

def eval_chebyc_l(n: int, x: float) -> float:
    return 2 * eval_chebyt_l(n, 0.5 * x)

def eval_sh_chebyt(n: float, x: float) -> float:
    return eval_chebyt(n, 2 * x - 1)

def eval_sh_chebyt_l(n: int, x: float) -> float:
    return eval_chebyt_l(n, 2 * x - 1)

def eval_sh_chebyu(n: float, x: float) -> float:
    return eval_chebyu(n, 2 * x - 1)

def eval_sh_chebyu_l(n: int, x: float) -> float:
    return eval_chebyu_l(n, 2 * x - 1)

# ------------------------------------------------------------------ Legendre
def eval_legendre(n: float, x: float) -> float:
    return H.hyp2f1(-n, n + 1, 1.0, 0.5 * (1 - x))

def eval_legendre_l(n: int, x: float) -> float:
    if n < 0:
        n = -n - 1
    if n == 0:
        return 1.0
    elif n == 1:
        return x
    elif fabs(x) < 1e-5:
        a = n // 2
        d = 1.0 if a % 2 == 0 else -1.0
        if n == 2 * a:
            d *= -2 / K.beta(a + 1.0, -0.5)
        else:
            d *= 2 * x / K.beta(a + 1.0, 0.5)
        p = 0.0
        for kk in range(a + 1):
            p += d
            d *= -2 * x * x * (a - kk) * (2 * n + 1 - 2 * a + 2 * kk) / ((n + 1 - 2 * a + 2 * kk) * (n + 2 - 2 * a + 2 * kk))
            if fabs(d) < 1e-20 * fabs(p):
                break
        return p
    d = x - 1
    p = x
    for kk in range(n - 1):
        k = kk + 1.0
        d = ((2 * k + 1) / (k + 1)) * (x - 1) * p + (k / (k + 1)) * d
        p = d + p
    return p

def eval_sh_legendre(n: float, x: float) -> float:
    return eval_legendre(n, 2 * x - 1)

def eval_sh_legendre_l(n: int, x: float) -> float:
    return eval_legendre_l(n, 2 * x - 1)

# ------------------------------------------------------------------ Laguerre
def eval_genlaguerre(n: float, alpha: float, x: float) -> float:
    if alpha <= -1:
        return NAN
    d = M.binom(n + alpha, n)
    return d * H.hyp1f1(-n, alpha + 1, x)

def eval_genlaguerre_l(n: int, alpha: float, x: float) -> float:
    if alpha <= -1:
        return NAN
    if _isnan(alpha) or _isnan(x):
        return NAN
    if n < 0:
        return 0.0
    elif n == 0:
        return 1.0
    elif n == 1:
        return -x + alpha + 1
    d = -x / (alpha + 1)
    p = d + 1
    for kk in range(n - 1):
        k = kk + 1.0
        d = -x / (k + alpha + 1) * p + (k / (k + alpha + 1)) * d
        p = d + p
    return M.binom(n + alpha, float(n)) * p

def eval_laguerre(n: float, x: float) -> float:
    return eval_genlaguerre(n, 0., x)

def eval_laguerre_l(n: int, x: float) -> float:
    return eval_genlaguerre_l(n, 0., x)

# ------------------------------------------------------------------ Hermite
def eval_hermitenorm(n: int, x: float) -> float:
    if _isnan(x):
        return x
    if n < 0:
        return NAN
    elif n == 0:
        return 1.0
    elif n == 1:
        return x
    y3 = 0.0
    y2 = 1.0
    k = n
    while k > 1:
        y1 = x * y2 - k * y3
        y3 = y2
        y2 = y1
        k -= 1
    return x * y2 - y3

def eval_hermite(n: int, x: float) -> float:
    if n < 0:
        return NAN
    return eval_hermitenorm(n, sqrt(2.0) * x) * pow(2.0, n / 2.0)
