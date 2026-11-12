# Ported from scipy 1.18.1 (BSD-3-Clause)
# scipy.special._orthogonal roots_* (Golub-Welsch + one Newton step +
# log-normalized weights), with a self-contained symmetric tridiagonal
# eigenvalue solver (implicit QL with Wilkinson shifts) in place of LAPACK
# eigvals_banded.

import numpy as np
from . import _kernels as K
from ._kernels import sqrt, fabs, log, exp
from . import _orth as O
from . import _bessel as B

PI = 3.14159265358979323846

def _tridiag_eigvals(d: List[float], e: List[float]) -> List[float]:
    """Eigenvalues (ascending) of the symmetric tridiagonal matrix with
    diagonal d and off-diagonal e (len(e) == len(d), e[-1] unused)."""
    n = len(d)
    dd = d.copy()
    ee = e.copy()
    for l in range(n):
        it = 0
        while True:
            m = l
            while m < n - 1:
                s = fabs(dd[m]) + fabs(dd[m + 1])
                if fabs(ee[m]) <= 2.220446049250313e-16 * s:
                    break
                m += 1
            if m == l:
                break
            it += 1
            if it > 60:
                break
            g = (dd[l + 1] - dd[l]) / (2.0 * ee[l])
            r = sqrt(g * g + 1.0)
            g = dd[m] - dd[l] + ee[l] / (g + (r if g >= 0 else -r))
            s = 1.0
            c = 1.0
            p = 0.0
            i = m - 1
            underflow = False
            while i >= l:
                f = s * ee[i]
                b = c * ee[i]
                r = sqrt(f * f + g * g)
                ee[i + 1] = r
                if r == 0.0:
                    dd[i + 1] -= p
                    ee[m] = 0.0
                    underflow = True
                    break
                s = f / r
                c = g / r
                g = dd[i + 1] - p
                r = (dd[i] - g) * s + 2.0 * c * b
                p = s * r
                dd[i + 1] = g + p
                g = c * r - b
                i -= 1
            if underflow and i >= l:
                continue
            dd[l] -= p
            ee[l] = g
            ee[m] = 0.0
    dd.sort()
    return dd

# family codes for f/df evaluation
_JAC = 0
_LAG = 1
_HER = 2
_HERN = 3
_GEG = 4
_LEG = 5

def _f(fam: int, n: int, a: float, b: float, x: float) -> float:
    if fam == _JAC:
        return O.eval_jacobi_l(n, a, b, x)
    elif fam == _LAG:
        return O.eval_genlaguerre_l(n, a, x)
    elif fam == _HER:
        return O.eval_hermite(n, x)
    elif fam == _HERN:
        return O.eval_hermitenorm(n, x)
    elif fam == _GEG:
        return O.eval_gegenbauer_l(n, a, x)
    return O.eval_legendre_l(n, x)

def _df(fam: int, n: int, a: float, b: float, x: float) -> float:
    if fam == _JAC:
        return 0.5 * (n + a + b + 1) * O.eval_jacobi_l(n - 1, a + 1, b + 1, x)
    elif fam == _LAG:
        return (n * O.eval_genlaguerre_l(n, a, x) - (n + a) * O.eval_genlaguerre_l(n - 1, a, x)) / x
    elif fam == _HER:
        return 2.0 * n * O.eval_hermite(n - 1, x)
    elif fam == _HERN:
        return n * O.eval_hermitenorm(n - 1, x)
    elif fam == _GEG:
        return (-n * x * O.eval_gegenbauer_l(n, a, x) + (n + 2 * a - 1) * O.eval_gegenbauer_l(n - 1, a, x)) / (1 - x * x)
    return (-n * x * O.eval_legendre_l(n, x) + n * O.eval_legendre_l(n - 1, x)) / (1 - x * x)

def _an(fam: int, k: float, a: float, b: float) -> float:
    if fam == _JAC:
        if a + b == 0.0:
            return (b - a) / (2 + a + b) if k == 0 else 0.0
        if k == 0:
            return (b - a) / (2 + a + b)
        return (b * b - a * a) / ((2.0 * k + a + b) * (2.0 * k + a + b + 2))
    elif fam == _LAG:
        return 2 * k + a + 1
    return 0.0 * k

def _bn(fam: int, k: float, a: float, b: float) -> float:
    if fam == _JAC:
        t = sqrt(k * (k + a + b) / (2.0 * k + a + b - 1)) if k != 1 else 1.0
        return 2.0 / (2.0 * k + a + b) * sqrt((k + a) * (k + b) / (2 * k + a + b + 1)) * t
    elif fam == _LAG:
        return -sqrt(k * (k + a))
    elif fam == _HER:
        return sqrt(k / 2.0)
    elif fam == _HERN:
        return sqrt(k)
    elif fam == _GEG:
        return sqrt(k * (k + 2 * a - 1) / (4 * (k + a) * (k + a - 1)))
    return k * sqrt(1.0 / (4 * k * k - 1))

def _gen_roots_and_weights(n: int, mu0: float, fam: int, a: float, b: float, symmetrize: bool):
    d = [0.0] * n
    e = [0.0] * n
    for k in range(n):
        d[k] = _an(fam, float(k), a, b)
    for k in range(1, n):
        e[k - 1] = _bn(fam, float(k), a, b)
    x = _tridiag_eigvals(d, e)
    dy = [0.0] * n
    for i in range(n):
        y = _f(fam, n, a, b, x[i])
        dyi = _df(fam, n, a, b, x[i])
        x[i] -= y / dyi
        dy[i] = dyi
    fm = [0.0] * n
    for i in range(n):
        fm[i] = _f(fam, n - 1, a, b, x[i])
    lfmax = -1e308
    lfmin = 1e308
    ldmax = -1e308
    ldmin = 1e308
    for i in range(n):
        lf = log(fabs(fm[i]))
        ld = log(fabs(dy[i]))
        lfmax = max(lfmax, lf)
        lfmin = min(lfmin, lf)
        ldmax = max(ldmax, ld)
        ldmin = min(ldmin, ld)
    sf = exp((lfmax + lfmin) / 2.)
    sd = exp((ldmax + ldmin) / 2.)
    w = [0.0] * n
    for i in range(n):
        w[i] = 1.0 / ((fm[i] / sf) * (dy[i] / sd))
    if symmetrize:
        w2 = [0.0] * n
        x2 = [0.0] * n
        for i in range(n):
            w2[i] = (w[i] + w[n - 1 - i]) / 2
            x2[i] = (x[i] - x[n - 1 - i]) / 2
        w = w2
        x = x2
    # numpy pairwise sum of w
    tot = float(np.sum(np.array(w)))
    for i in range(n):
        w[i] *= mu0 / tot
    return (np.array(x), np.array(w), mu0)

def _check_n(n) -> int:
    m = int(n)
    if n < 1 or n != m:
        raise ValueError("n must be a positive integer.")
    return m

def roots_legendre_mu(n):
    m = _check_n(n)
    return _gen_roots_and_weights(m, 2.0, _LEG, 0.0, 0.0, True)

def roots_chebyt_mu(n):
    m = _check_n(n)
    x = np.empty(m, dtype=float)
    k = -m + 1
    for i in range(m):
        x[i] = B.sinpi(float(k) / (2 * m))
        k += 2
    w = np.full(m, PI / m)
    return (x, w, PI)

def roots_chebyu_mu(n):
    m = _check_n(n)
    x = np.empty(m, dtype=float)
    w = np.empty(m, dtype=float)
    for i in range(m):
        t = float(m - i) * PI / (m + 1)
        x[i] = K.cos(t)
        s = K.sin(t)
        w[i] = PI * (s * s) / (m + 1)
    return (x, w, PI / 2)

def roots_gegenbauer_mu(n, alpha: float):
    m = _check_n(n)
    if alpha < -0.5:
        raise ValueError("alpha must be greater than -0.5.")
    elif alpha == 0.0:
        return roots_chebyt_mu(n)
    mu0 = 0.0
    if alpha <= 170:
        mu0 = (sqrt(PI) * K.gamma(alpha + 0.5)) / K.gamma(alpha + 1)
    else:
        inv_alpha = 1. / alpha
        coeffs = (0.000207186, -0.00152206, -0.000640869, 0.00488281, 0.0078125, -0.125, 1.)
        mu0 = coeffs[0]
        for term in range(1, len(coeffs)):
            mu0 = mu0 * inv_alpha + coeffs[term]
        mu0 = mu0 * sqrt(PI / alpha)
    return _gen_roots_and_weights(m, mu0, _GEG, alpha, 0.0, True)

def roots_jacobi_mu(n, alpha: float, beta: float):
    m = _check_n(n)
    if alpha <= -1 or beta <= -1:
        raise ValueError("alpha and beta must be greater than -1.")
    if alpha == 0.0 and beta == 0.0:
        return roots_legendre_mu(m)
    if alpha == beta:
        return roots_gegenbauer_mu(m, alpha + 0.5)
    mu0 = 0.0
    if (alpha + beta) <= 1000:
        mu0 = K.pow(2.0, alpha + beta + 1) * K.beta(alpha + 1, beta + 1)
    else:
        mu0 = exp((alpha + beta + 1) * log(2.0) + K.betaln(alpha + 1, beta + 1))
    return _gen_roots_and_weights(m, mu0, _JAC, alpha, beta, False)

def roots_genlaguerre_mu(n, alpha: float):
    m = _check_n(n)
    if alpha < -1:
        raise ValueError("alpha must be greater than -1.")
    mu0 = K.gamma(alpha + 1)
    if m == 1:
        return (np.array([alpha + 1.0]), np.array([mu0]), mu0)
    return _gen_roots_and_weights(m, mu0, _LAG, alpha, 0.0, False)

def roots_hermite_mu(n):
    m = _check_n(n)
    return _gen_roots_and_weights(m, sqrt(PI), _HER, 0.0, 0.0, True)

def roots_hermitenorm_mu(n):
    m = _check_n(n)
    return _gen_roots_and_weights(m, sqrt(2.0 * PI), _HERN, 0.0, 0.0, True)
