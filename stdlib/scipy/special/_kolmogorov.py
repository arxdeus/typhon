# Ported from scipy 1.18.1 (BSD-3-Clause)
# Algorithms from Cephes (BSD-3) / xsf (BSD-3) cephes/kolmogorov.h
#
# kolmogorov, kolmogi, kolmogc, kolmogp, smirnov, smirnovi, smirnovc, smirnovp
# (scalar float kernels).  Return tuples (sf, cdf, pdf) internally.

from . import _kernels as K
from ._kernels import exp, log, sqrt, fabs, pow
from . import _dd as D

NAN = float('nan')
INF = float('inf')
PI = 3.14159265358979323846
LN2 = 0.69314718055994530942
SQRT2 = 1.41421356237309504880
LOGSQRT2PI = 9.189385332046727417803297364056176398614E-1
MINLOG = -7.451332191019412076235245305675398106812E2
SCIPY_El = 2.718281828459045235360287471352662498
EPS = 2.220446049250313e-16
KOLMOG_MAXITER = 500
KOLMOG_CUTOVER = 0.82
SMIRNOV_MAX_COMPUTE_N = 1000000
SM_UPPER_MAX_TERMS = 3
SM_UPPERSUM_MIN_N = 10
MIN_EXPABLE = -708 - 38
SM_MAX_EXPONENT = 960
_xtol = EPS
_rtol = 2 * EPS

def _isnan(x: float) -> bool:
    return x != x

def _clamp(x: float, a: float, b: float) -> float:
    return a if x < a else (b if x > b else x)

def _within_tol(x: float, y: float, atol: float, rtol: float) -> bool:
    return fabs(x - y) <= atol + rtol * fabs(y)

def _kolmogorov(x: float):
    P = 1.0
    Dd = 0.0
    sf = 0.0
    cdf = 0.0
    pdf = 0.0
    if _isnan(x):
        return (NAN, NAN, NAN)
    if x <= 0:
        return (1.0, 0.0, 0.0)
    if x <= PI / sqrt(-MIN_EXPABLE * 8.0):
        return (1.0, 0.0, 0.0)
    if x <= KOLMOG_CUTOVER:
        w = sqrt(2 * PI) / x
        logu8 = -PI * PI / (x * x)
        u = exp(logu8 / 8)
        if u == 0:
            logP = logu8 / 8 + log(w)
            P = exp(logP)
        else:
            u8 = exp(logu8)
            u8cub = pow(u8, 3.0)
            P = 1 + u8cub * P
            Dd = 5 * 5 + u8cub * Dd
            P = 1 + u8 * u8 * P
            Dd = 3 * 3 + u8 * u8 * Dd
            P = 1 + u8 * P
            Dd = 1 * 1 + u8 * Dd
            Dd = PI * PI / 4 / (x * x) * Dd - P
            Dd *= w * u / x
            P = w * u * P
        cdf = P
        sf = 1 - P
        pdf = Dd
    else:
        logv = -2 * x * x
        v = exp(logv)
        vsq = v * v
        v3 = pow(v, 3.0)
        vpwr = v3 * v3 * v
        P = 1 - vpwr * P
        Dd = 3 * 3 - vpwr * Dd
        vpwr = v3 * vsq
        P = 1 - vpwr * P
        Dd = 2 * 2 - vpwr * Dd
        vpwr = v3
        P = 1 - vpwr * P
        Dd = 1 * 1 - vpwr * Dd
        P = 2 * v * P
        Dd = 8 * v * x * Dd
        sf = P
        cdf = 1 - sf
        pdf = Dd
    pdf = max(0.0, pdf)
    return (_clamp(sf, 0.0, 1.0), _clamp(cdf, 0.0, 1.0), pdf)

def _kolmogi(psf: float, pcdf: float) -> float:
    a = 0.0
    b = INF
    x = 0.0
    if not (psf >= 0.0 and pcdf >= 0.0 and pcdf <= 1.0 and psf <= 1.0):
        return NAN
    if fabs(1.0 - pcdf - psf) > 4 * EPS:
        return NAN
    if pcdf == 0.0:
        return 0.0
    if psf == 0.0:
        return INF
    if pcdf <= 0.5:
        logpcdf = log(pcdf)
        a = PI / (2 * SQRT2 * sqrt(-(logpcdf + logpcdf / 2 - LOGSQRT2PI)))
        b = PI / (2 * SQRT2 * sqrt(-(logpcdf + 0 - LOGSQRT2PI)))
        a = PI / (2 * SQRT2 * sqrt(-(logpcdf + log(a) - LOGSQRT2PI)))
        b = PI / (2 * SQRT2 * sqrt(-(logpcdf + log(b) - LOGSQRT2PI)))
        x = (a + b) / 2.0
    else:
        jiggerb = 256 * EPS
        pba = psf / (1.0 - exp(-4.0)) / 2
        pbb = psf * (1 - jiggerb) / 2
        a = sqrt(-0.5 * log(pba))
        b = sqrt(-0.5 * log(pbb))
        p = psf / 2.0
        p2 = p * p
        p3 = p * p * p
        q0 = 1 + p3 * (1 + p3 * (4 + p2 * (-1 + p * (22 + p2 * (-13 + 140 * p)))))
        q0 *= p
        x = sqrt(-log(q0) / 2)
        if x < a or x > b:
            x = (a + b) / 2
    iterations = 0
    while True:
        x0 = x
        probs = _kolmogorov(x0)
        df = (pcdf - probs[1]) if pcdf < 0.5 else (probs[0] - psf)
        if fabs(df) == 0:
            break
        if df > 0 and x > a:
            a = x
        elif df < 0 and x < b:
            b = x
        dfdx = -probs[2]
        if fabs(dfdx) <= 0.0:
            x = (a + b) / 2
        else:
            t = df / dfdx
            x = x0 - t
        if x >= a and x <= b:
            if _within_tol(x, x0, _xtol, _rtol):
                break
            if x == a or x == b:
                x = (a + b) / 2.0
                if x == a or x == b:
                    break
        else:
            x = (a + b) / 2.0
            if _within_tol(x, x0, _xtol, _rtol):
                break
        iterations += 1
        if iterations > KOLMOG_MAXITER:
            break
    return x

def kolmogorov(x: float) -> float:
    if _isnan(x):
        return NAN
    return _kolmogorov(x)[0]

def kolmogc(x: float) -> float:
    if _isnan(x):
        return NAN
    return _kolmogorov(x)[1]

def kolmogp(x: float) -> float:
    if _isnan(x):
        return NAN
    if x <= 0:
        return -0.0
    return -_kolmogorov(x)[2]

def kolmogi(p: float) -> float:
    if _isnan(p):
        return NAN
    return _kolmogi(p, 1 - p)

def kolmogci(p: float) -> float:
    if _isnan(p):
        return NAN
    return _kolmogi(1 - p, p)

# ------------------------------------------------------------------ smirnov
def _next_pow2(x: float) -> float:
    q = K.ldexp(x, i32(1 - 53))
    L = fabs(q + x)
    if L == 0:
        L = fabs(x)
    else:
        Lint = int(L)
        if float(Lint) == L:
            L = float(Lint)
    return L

def _modNX(n: int, x: float):
    """Returns (alpha, nxfloor, nx)."""
    nxD = D.mul_d((x, 0.0), float(n))
    nxfloorD = D.floor(nxD)
    alphaD = D.sub(nxD, nxfloorD)
    alpha = alphaD[0]
    nxfloor = int(nxfloorD[0])
    if alpha == 1:
        nxfloor += 1
        alpha = 0.0
    return (alpha, nxfloor, nxD[0])

def _pow_D(a, m: int):
    if m <= 0:
        if m == 0:
            return (1.0, 0.0)
        return D.d_div(1.0, _pow_D(a, -m))
    if a[0] == 0.0 and a[1] == 0.0:
        return (0.0, 0.0)
    ans = pow(a[0], float(m))
    r = a[1] / a[0]
    adj = m * r
    if fabs(adj) > 1e-8:
        if fabs(adj) < 1e-4:
            adj += (m * r) * ((m - 1) / 2.0 * r)
        else:
            adj = K.expm1(m * K.log1p(r))
    return D.add_d((ans, 0.0), ans * adj)

def _pow2(a: float, b: float, m: int) -> float:
    return _pow_D(D.add_d((a, 0.0), b), m)[0]

def _pow2Scaled_D(a, m: int):
    """Returns (mantissa_dd, exponent)."""
    maxExpt = SM_MAX_EXPONENT
    if m <= 0:
        if m == 0:
            return ((1.0, 0.0), 0)
        r1 = _pow2Scaled_D(a, -m)
        r2 = D.frexp_dd(D.d_div(1.0, r1[0]))
        return (r2[0], -r1[1] + r2[1])
    fy = D.frexp_dd(a)
    y = fy[0]
    yE = fy[1]
    if m == 1:
        return (y, yE)
    if m * (y[0] - 1) / y[0] < -SM_MAX_EXPONENT * LN2:
        lg2y = log(y[0]) / LN2
        lgAns = m * lg2y
        if lgAns <= -SM_MAX_EXPONENT:
            maxExpt = int(_next_pow2(-SM_MAX_EXPONENT / lg2y + 1) / 2)
    if m <= maxExpt:
        ans1 = _pow_D(y, m)
        fa = D.frexp_dd(ans1)
        return (fa[0], fa[1] + m * yE)
    q = m // maxExpt
    r = m % maxExpt
    a2r = _pow2Scaled_D(y, r)
    a2m = _pow2Scaled_D(y, maxExpt)
    a2mq = _pow2Scaled_D(a2m[0], q)
    fa = D.frexp_dd(D.mul(a2r[0], a2mq[0]))
    y2mqE = a2mq[1] + a2m[1] * q
    return (fa[0], fa[1] + y2mqE + a2r[1] + m * yE)

def _pow4_D(a: float, b: float, c: float, d: float, m: int):
    if m <= 0:
        if m == 0:
            return (1.0, 0.0)
        return _pow4_D(c, d, a, b, -m)
    A = D.add_d((a, 0.0), b)
    C = D.add_d((c, 0.0), d)
    if A[0] == 0.0 and A[1] == 0.0:
        return (NAN, NAN) if (C[0] == 0.0 and C[1] == 0.0) else (0.0, 0.0)
    if C[0] == 0.0 and C[1] == 0.0:
        return (-INF, -INF) if A[0] < 0 else (INF, INF)
    return _pow_D(D.div(A, C), m)

def _pow4(a: float, b: float, c: float, d: float, m: int) -> float:
    return _pow4_D(a, b, c, d, m)[0]

def _logpow4(a: float, b: float, c: float, d: float, m: int) -> float:
    if m == 0:
        return 0.0
    A = D.add_d((a, 0.0), b)
    C = D.add_d((c, 0.0), d)
    if A[0] == 0.0 and A[1] == 0.0:
        return 0.0 if (C[0] == 0.0 and C[1] == 0.0) else -INF
    if C[0] == 0.0 and C[1] == 0.0:
        return INF
    X = D.div(A, C)
    ans = (0.0, 0.0)
    if 0.5 <= X[0] and X[0] <= 1.5:
        A1 = D.sub(A, C)
        X1 = D.div(A1, C)
        ans = D.log1p(X1)
    else:
        ans = D.log(X)
    return D.mul_d(ans, float(m))[0]

def _computeAv(n: int, x: float, v: int, Cman, Cexpt: int):
    t2x = D.sub_d(D.div_d((float(n - v), 0.0), float(n)), x)
    r2 = _pow2Scaled_D(t2x, n - v)
    t1x = D.add_d(D.div_d((float(v), 0.0), float(n)), x)
    r1 = _pow2Scaled_D(t1x, v - 1)
    ans = D.mul(r1[0], r2[0])
    ans = D.mul(ans, Cman)
    ansE = Cexpt + r1[1] + r2[1]
    return D.ldexp(ans, ansE)

def _updateBinomial(Cman, Cexpt: int, n: int, j: int):
    rat = D.div_d((float(n - j), 0.0), j + 1.0)
    man2 = D.mul(Cman, rat)
    f = D.frexp_dd(man2)
    return (f[0], Cexpt + f[1])

def _smirnov(n: int, x: float):
    """Returns (sf, cdf, pdf)."""
    if not (n > 0 and x >= 0.0 and x <= 1.0):
        return (NAN, NAN, NAN)
    if n == 1:
        return (1 - x, x, 1.0)
    if x == 0.0:
        return (1.0, 0.0, 1.0)
    if x == 1.0:
        return (0.0, 1.0, 0.0)
    m = _modNX(n, x)
    alpha = m[0]
    nxfl = m[1]
    nx = m[2]
    n1mxfl = n - nxfl - (0 if alpha == 0 else 1)
    n1mxceil = n - nxfl
    if alpha == 0:
        n1mxfl -= 1
        n1mxceil += 1
    sf = 0.0
    cdf = 0.0
    pdf = 0.0
    if nxfl == 0 or (nxfl == 1 and alpha == 0):
        t = _pow2(1.0, x, n - 1)
        pdf = (nx + 1) * t / (1 + x)
        cdf = x * t
        sf = 1 - cdf
        if nxfl == 1:
            pdf -= 0.5
        return (sf, cdf, pdf)
    if -2 * n * x * x < MINLOG:
        return (0.0, 1.0, 0.0)
    if nxfl >= n - 1:
        sf = _pow2(1.0, -x, n)
        cdf = 1 - sf
        pdf = n * sf / (1 - x)
        return (sf, cdf, pdf)
    if n > SMIRNOV_MAX_COMPUTE_N:
        logp = -pow(6.0 * n * x + 1, 2.0) / 18.0 / n
        if logp < -LN2:
            sf = exp(logp)
            cdf = 1 - sf
        else:
            cdf = -K.expm1(logp)
            sf = 1 - cdf
        pdf = (6.0 * n * x + 1) * 2 * sf / 3
        return (sf, cdf, pdf)
    nUpperTerms = n - n1mxceil + 1
    bUseUpperSum = nUpperTerms <= 1 and x < 0.5
    bUseUpperSum = bUseUpperSum or (n >= SM_UPPERSUM_MIN_N and nUpperTerms <= SM_UPPER_MAX_TERMS and x <= 0.5 / sqrt(float(n)))
    start = 0
    step = 1
    nTerms = n1mxfl + 1
    vmid = n // 2
    Cman = (1.0, 0.0)
    Cexpt = 0
    AjSum = (0.0, 0.0)
    dAjSum = (0.0, 0.0)
    oneOverX = D.d_div(1.0, (x, 0.0))
    Aj = (0.0, 0.0)
    dAjCoeff = (0.0, 0.0)
    if bUseUpperSum:
        start = n
        step = -1
        nTerms = n - n1mxceil + 1
        t1 = _pow4_D(1.0, x, 1.0, 0.0, n - 1)
        Aj = t1
        dAjCoeff = D.d_div(float(n - 1), D.add_d((1.0, 0.0), x))
        dAjCoeff = D.add(dAjCoeff, oneOverX)
    else:
        t2 = _pow4_D(1.0, -x, 1.0, 0.0, n)
        Aj = D.div_d(t2, x)
        dAjCoeff = D.div(D.d_sub(-1.0, D.mul_d((float(n - 1), 0.0), x)), D.sub_d((1.0, 0.0), x))
        dAjCoeff = D.div_d(dAjCoeff, x)
        dAjCoeff = D.add(dAjCoeff, oneOverX)
    dAj = D.mul(Aj, dAjCoeff)
    AjSum = D.add(AjSum, Aj)
    dAjSum = D.add(dAjSum, dAj)
    cb = _updateBinomial(Cman, Cexpt, n, 0)
    Cman = cb[0]
    Cexpt = cb[1]
    j = 1
    while j < nTerms:
        v = start + j * step
        Aj = _computeAv(n, x, v, Cman, Cexpt)
        if D.isfinite(Aj) and Aj[0] != 0.0:
            c1 = D.d_div(float(n * (v - 1)), D.add_d((float(nxfl + v), 0.0), alpha))
            c2 = D.d_div(float((n - v) * n), D.sub_d((float(n - nxfl - v), 0.0), alpha))
            dAjCoeff = D.add(D.sub(c1, c2), oneOverX)
            dAj = D.mul(Aj, dAjCoeff)
            AjSum = D.add(AjSum, Aj)
            dAjSum = D.add(dAjSum, dAj)
        if Aj[0] != 0.0:
            if (4 * (nTerms - j) * fabs(Aj[0])) < (EPS * AjSum[0]) and j != nTerms - 1:
                break
        elif j > vmid:
            break
        cb = _updateBinomial(Cman, Cexpt, n, j)
        Cman = cb[0]
        Cexpt = cb[1]
        j += 1
    deriv = D.mul_d(dAjSum, x)[0]
    prob = D.mul_d(AjSum, x)[0]
    if step < 0:
        cdf = prob
        sf = 1 - prob
        pdf = deriv
    else:
        cdf = 1 - prob
        sf = prob
        pdf = -deriv
    pdf = max(0.0, pdf)
    return (_clamp(sf, 0.0, 1.0), _clamp(cdf, 0.0, 1.0), pdf)

def _smirnovi(n: int, psf: float, pcdf: float) -> float:
    a = 0.0
    b = 1.0
    x = 0.0
    if not (n > 0 and psf >= 0.0 and pcdf >= 0.0 and pcdf <= 1.0 and psf <= 1.0):
        return NAN
    if fabs(1.0 - pcdf - psf) > 4 * EPS:
        return NAN
    if pcdf == 0.0:
        return 0.0
    if psf == 0.0:
        return 1.0
    if n == 1:
        return pcdf
    psfrootn = pow(psf, 1.0 / n)
    if n < 150 and n * psfrootn <= 1:
        return 1 - psfrootn
    logpcdf = log(pcdf) if pcdf < 0.5 else K.log1p(-psf)
    maxlogpcdf = _logpow4(1.0, 0.0, float(n), 0.0, 1) + _logpow4(float(n), 1.0, float(n), 0.0, n - 1)
    if logpcdf <= maxlogpcdf:
        xmin = pcdf / SCIPY_El
        xmax = pcdf
        P1 = _pow4(float(n), 1.0, float(n), 0.0, n - 1) / n
        R = pcdf / P1
        z0 = R
        if R >= 1:
            return 1.0 / n
        z0 = (z0 * z0 + R * exp(1 - z0)) / (1 + z0)
        x = z0 / n
        a = xmin * (1 - 4 * EPS)
        a = max(a, 0.0)
        b = xmax * (1 + 4 * EPS)
        b = min(b, 1.0 / n)
        x = _clamp(x, a, b)
    else:
        xmin = 1 - psfrootn
        logpsf = log(psf) if psf < 0.5 else K.log1p(-pcdf)
        xmax = sqrt(-logpsf / (2.0 * n))
        xmax6 = xmax - 1.0 / (6 * n)
        a = max(xmin, 1.0 / n)
        b = min(xmax, 1 - 1.0 / n)
        x = xmax6
    if x < a or x > b:
        x = (a + b) / 2
    dxold = b - a
    dx = dxold
    iterations = 0
    while True:
        x0 = x
        probs = _smirnov(n, x0)
        df = (pcdf - probs[1]) if pcdf < 0.5 else (probs[0] - psf)
        dfdx = -probs[2]
        if df == 0:
            return x
        if df > 0 and x > a:
            a = x
        elif df < 0 and x < b:
            b = x
        deltax = 0.0
        if dfdx == 0:
            x = (a + b) / 2
            deltax = x0 - x
        else:
            deltax = df / dfdx
            x = x0 - deltax
        if a <= x and x <= b and (fabs(2 * deltax) <= fabs(dxold) or fabs(dxold) < 256 * EPS):
            dxold = dx
            dx = deltax
        else:
            dxold = dx
            dx = dx / 2
            x = (a + b) / 2
            deltax = x0 - x
        if _within_tol(x, x0, 0.0 if psf < 0.5 else _xtol, _rtol):
            break
        iterations += 1
        if iterations > KOLMOG_MAXITER:
            return x
    return x

def _nint(n: float) -> int:
    return int(n)

def smirnov(n: float, d: float) -> float:
    if _isnan(n) or _isnan(d):
        return NAN
    return _smirnov(_nint(n), d)[0]

def smirnovc(n: float, d: float) -> float:
    if _isnan(n) or _isnan(d):
        return NAN
    return _smirnov(_nint(n), d)[1]

def smirnovp(n: float, d: float) -> float:
    nn = _nint(n)
    if not (nn > 0 and d >= 0.0 and d <= 1.0):
        return NAN
    if nn == 1:
        return -1.0
    if d == 1.0:
        return -0.0
    if d == 0.0:
        return -1.0
    return -_smirnov(nn, d)[2]

def smirnovi(n: float, p: float) -> float:
    if _isnan(n) or _isnan(p):
        return NAN
    return _smirnovi(_nint(n), p, 1 - p)

def smirnovci(n: float, p: float) -> float:
    if _isnan(n) or _isnan(p):
        return NAN
    return _smirnovi(_nint(n), 1 - p, p)
