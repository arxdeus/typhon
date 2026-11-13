# Copyright (C) 2026 the Typhon authors
# Ported from scipy 1.18.1 (BSD-3-Clause)
# Ellipsoidal harmonics (Lame functions): port of scipy/special/_ellip_harm.pxd
# and _ellip_harm_2.pyx (Copyright (c) 2012 Jaydeep P. Bardhan, Matthew G.
# Knepley; (c) 2014 Janani Padmanabhan; BSD-3-Clause).

import numpy as np
from ._kernels import sqrt, fabs, pow

NAN = float('nan')

def lame_coefficients(h2: float, k2: float, n: int, p: int, signm: float, signn: float) -> List[float]:
    """Eigenvector (Lame polynomial coefficients); empty list on invalid args."""
    if n < 0 or p < 1 or p > 2 * n + 1 or fabs(signm) != 1.0 or fabs(signn) != 1.0:
        return List[float]()
    r = n // 2
    alpha = h2
    beta = k2 - h2
    gamma = alpha - beta
    t = 'K'
    tp = p
    size = r + 1
    if p - 1 < r + 1:
        t = 'K'
        tp = p
        size = r + 1
    elif p - 1 < (n - r) + (r + 1):
        t = 'L'
        tp = p - (r + 1)
        size = n - r
    elif p - 1 < (n - r) + (n - r) + (r + 1):
        t = 'M'
        tp = p - (n - r) - (r + 1)
        size = n - r
    else:
        t = 'N'
        tp = p - (n - r) - (n - r) - (r + 1)
        size = r
    g = [0.0] * size
    d = [0.0] * size
    f = [0.0] * size
    ss = [0.0] * size
    odd = n % 2 == 1
    if t == 'K':
        for j in range(r + 1):
            g[j] = -(2 * j + 2) * (2 * j + 1) * beta
            if odd:
                f[j] = -alpha * (2 * (r - (j + 1)) + 2) * (2 * ((j + 1) + r) + 1)
                d[j] = ((2 * r + 1) * (2 * r + 2) - 4 * j * j) * alpha + (2 * j + 1) * (2 * j + 1) * beta
            else:
                f[j] = -alpha * (2 * (r - (j + 1)) + 2) * (2 * (r + (j + 1)) - 1)
                d[j] = 2 * r * (2 * r + 1) * alpha - 4 * j * j * gamma
    elif t == 'L':
        for j in range(n - r):
            g[j] = -(2 * j + 2) * (2 * j + 3) * beta
            if odd:
                f[j] = -alpha * (2 * (r - (j + 1)) + 2) * (2 * ((j + 1) + r) + 1)
                d[j] = (2 * r + 1) * (2 * r + 2) * alpha - (2 * j + 1) * (2 * j + 1) * gamma
            else:
                f[j] = -alpha * (2 * (r - (j + 1))) * (2 * (r + (j + 1)) + 1)
                d[j] = (2 * r * (2 * r + 1) - (2 * j + 1) * (2 * j + 1)) * alpha + (2 * j + 2) * (2 * j + 2) * beta
    elif t == 'M':
        for j in range(n - r):
            g[j] = -(2 * j + 2) * (2 * j + 1) * beta
            if odd:
                f[j] = -alpha * (2 * (r - (j + 1)) + 2) * (2 * ((j + 1) + r) + 1)
                d[j] = ((2 * r + 1) * (2 * r + 2) - (2 * j + 1) * (2 * j + 1)) * alpha + 4 * j * j * beta
            else:
                f[j] = -alpha * (2 * (r - (j + 1))) * (2 * (r + (j + 1)) + 1)
                d[j] = 2 * r * (2 * r + 1) * alpha - (2 * j + 1) * (2 * j + 1) * gamma
    else:
        for j in range(r):
            g[j] = -(2 * j + 2) * (2 * j + 3) * beta
            if odd:
                f[j] = -alpha * (2 * (r - (j + 1))) * (2 * ((j + 1) + r) + 3)
                d[j] = (2 * r + 1) * (2 * r + 2) * alpha - (2 * j + 2) * (2 * j + 2) * gamma
            else:
                f[j] = -alpha * (2 * (r - (j + 1))) * (2 * (r + (j + 1)) + 1)
                d[j] = 2 * r * (2 * r + 1) * alpha - (2 * j + 2) * (2 * j + 2) * alpha + (2 * j + 1) * (2 * j + 1) * beta
    for i in range(size):
        if i == 0:
            ss[i] = 1.0
        else:
            ss[i] = sqrt(g[i - 1] / f[i - 1]) * ss[i - 1]
    dd = [0.0] * (size if size > 1 else 1)
    for i in range(size - 1):
        dd[i] = g[i] * ss[i] / ss[i + 1]
    eigv = [0.0] * size
    if size == 1:
        eigv[0] = 1.0
    else:
        from scipy.linalg import eigh_tridiagonal
        dv = np.array(d)
        ev = np.array(dd[:size - 1])
        res = eigh_tridiagonal(dv, ev, False, 'i', (tp - 1, tp - 1))
        vec = res[1]
        for i in range(size):
            eigv[i] = float(vec[i, 0])
    for i in range(size):
        eigv[i] /= ss[i]
    den = eigv[size - 1] / pow(-h2, float(size - 1))
    for i in range(size):
        eigv[i] = eigv[i] / den
    return eigv

def ellip_harm_eval(h2: float, k2: float, n: int, p: int, s: float, eigv: List[float], signm: float, signn: float) -> float:
    s2 = s * s
    r = n // 2
    size = 0
    psi = 0.0
    if p - 1 < r + 1:
        size = r + 1
        psi = pow(s, float(n - 2 * r))
    elif p - 1 < (n - r) + (r + 1):
        size = n - r
        psi = pow(s, float(1 - n + 2 * r)) * signm * sqrt(fabs(s2 - h2))
    elif p - 1 < (n - r) + (n - r) + (r + 1):
        size = n - r
        psi = pow(s, float(1 - n + 2 * r)) * signn * sqrt(fabs(s2 - k2))
    elif p - 1 < 2 * n + 1:
        size = r
        psi = pow(s, float(n - 2 * r)) * signm * signn * sqrt(fabs((s2 - h2) * (s2 - k2)))
    else:
        return NAN
    lambda_romain = 1.0 - s2 / h2
    pp = eigv[size - 1]
    j = size - 2
    while j >= 0:
        pp = pp * lambda_romain + eigv[j]
        j -= 1
    return pp * psi

def ellip_harm(h2: float, k2: float, n: int, p: int, s: float, signm: float, signn: float) -> float:
    eigv = lame_coefficients(h2, k2, n, p, signm, signn)
    if len(eigv) == 0:
        return NAN
    return ellip_harm_eval(h2, k2, n, p, s, eigv, signm, signn)

class _EData:
    h2: float
    k2: float
    n: int
    p: int
    eigv: List[float]
    kind: int
    def __init__(self, h2: float, k2: float, n: int, p: int, eigv: List[float], kind: int):
        self.h2 = h2
        self.k2 = k2
        self.n = n
        self.p = p
        self.eigv = eigv
        self.kind = kind
    def __call__(self, t: float) -> float:
        h = sqrt(self.h2)
        k = sqrt(self.k2)
        t2 = t * t
        if self.kind == 0:
            i = ellip_harm_eval(self.h2, self.k2, self.n, self.p, 1.0 / t, self.eigv, 1.0, 1.0)
            return 1.0 / (i * i * sqrt(1.0 - t2 * self.k2) * sqrt(1.0 - t2 * self.h2))
        i = ellip_harm_eval(self.h2, self.k2, self.n, self.p, t, self.eigv, 1.0, 1.0)
        if self.kind == 1:
            return i * i / sqrt((t + h) * (t + k))
        if self.kind == 2:
            return t2 * i * i / sqrt((t + h) * (t + k))
        if self.kind == 3:
            return i * i / sqrt((t + h) * (self.k2 - t2))
        return i * i * t2 / sqrt((t + h) * (self.k2 - t2))

def ellipsoid(h2: float, k2: float, n: int, p: int, s: float) -> float:
    from scipy.integrate import quad
    eigv = lame_coefficients(h2, k2, n, p, 1.0, 1.0)
    if len(eigv) == 0:
        return NAN
    data = _EData(h2, k2, n, p, eigv, 0)
    q = quad(data, 0.0, 1.0 / s, (), 0, 1e-300, 1e-15, 50)
    res = q[0]
    err = q[1]
    if err > 1e-10 * fabs(res) + 1e-290:
        return NAN
    return res * (2 * n + 1) * ellip_harm_eval(h2, k2, n, p, s, eigv, 1.0, 1.0)

def ellipsoid_norm(h2: float, k2: float, n: int, p: int) -> float:
    from scipy.integrate import quad
    eigv = lame_coefficients(h2, k2, n, p, 1.0, 1.0)
    if len(eigv) == 0:
        return NAN
    h = sqrt(h2)
    k = sqrt(k2)
    q0 = quad(_EData(h2, k2, n, p, eigv, 1), h, k, (), 0, 1e-300, 1e-15, 50, None, 'alg', (-0.5, -0.5))
    q1 = quad(_EData(h2, k2, n, p, eigv, 2), h, k, (), 0, 1e-300, 1e-15, 50, None, 'alg', (-0.5, -0.5))
    q2 = quad(_EData(h2, k2, n, p, eigv, 3), 0.0, h, (), 0, 1e-300, 1e-15, 50, None, 'alg', (0.0, -0.5))
    q3 = quad(_EData(h2, k2, n, p, eigv, 4), 0.0, h, (), 0, 1e-300, 1e-15, 50, None, 'alg', (0.0, -0.5))
    res = q0[0]
    err = q0[1]
    res1 = q1[0]
    err1 = q1[1]
    res2 = q2[0]
    err2 = q2[1]
    res3 = q3[0]
    err3 = q3[1]
    error = 8.0 * (res2 * err1 + err2 * res1 + res * err3 + res3 * err)
    result = 8.0 * (res1 * res2 - res * res3)
    if error > 10e-8 * fabs(result):
        return NAN
    return result
