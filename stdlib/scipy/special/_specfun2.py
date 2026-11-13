# Copyright (C) 2026 the Typhon authors
# Ported from scipy 1.18.1 (BSD-3-Clause)
# Algorithms from xsf (BSD-3) specfun/specfun.h, bessel.h, struve.h, airy.h,
# legendre.h, fresnel.h: translations of the Fortran routines of Shanjie Zhang
# & Jianming Jin, "Computation of Special Functions" (Wiley 1996).
#
# Sequence functions (lamn/lamv, riccati, lqn/lqmn, bernoulli/euler), integrals
# of Bessel/Struve functions, Airy and Fresnel zeros, modified Fresnel.

from ._kernels import exp, log, sqrt, fabs, sin, cos, pow
from ._specfun import msta1, msta2

NAN = float('nan')
PI = 3.141592653589793
EL = 0.5772156649015329

def gam0(x: float) -> float:
    g = [1.0, 0.5772156649015329, -0.6558780715202538, -0.420026350340952e-1,
         0.1665386113822915, -0.421977345555443e-1, -0.96219715278770e-2,
         0.72189432466630e-2, -0.11651675918591e-2, -0.2152416741149e-3,
         0.1280502823882e-3, -0.201348547807e-4, -0.12504934821e-5,
         0.11330272320e-5, -0.2056338417e-6, 0.61160950e-8, 0.50020075e-8,
         -0.11812746e-8, 0.1043427e-9, 0.77823e-11, -0.36968e-11, 0.51e-12,
         -0.206e-13, -0.54e-14, 0.14e-14]
    gr = g[24]
    for k in range(23, -1, -1):
        gr = gr * x + g[k]
    return 1.0 / (gr * x)

def bernob(n: int) -> List[float]:
    bn = [0.0] * (n + 1)
    bn[0] = 1.0
    bn[1] = -0.5
    bn[2] = 1.0 / 6.0
    tpi = 6.283185307179586
    r1 = (2.0 / tpi) ** 2
    m = 4
    while m < n + 1:
        r1 = -r1 * (m - 1) * m / (tpi * tpi)
        r2 = 1.0
        for k in range(2, 10001):
            s = pow(1.0 / k, float(m))
            r2 += s
            if s < 1e-15:
                break
        bn[m] = r1 * r2
        m += 2
    return bn

def eulerb(n: int) -> List[float]:
    en = [0.0] * (n + 1)
    hpi = 2.0 / 3.141592653589793
    en[0] = 1.0
    en[2] = -1.0
    r1 = -4.0 * hpi ** 3
    m = 4
    while m <= n:
        r1 = -r1 * (m - 1) * m * hpi * hpi
        r2 = 1.0
        isgn = 1
        k = 3
        while k <= 1000:
            isgn = -isgn
            s = pow(1.0 / k, float(m + 1))
            r2 += isgn * s
            if s < 1e-15:
                break
            k += 2
        en[m] = r1 * r2
        m += 2
    return en

def lamn(n: int, x: float):
    """Lambda functions of integer order: (bl[0..n], dl[0..n])."""
    bl = [0.0] * (n + 2)
    dl = [0.0] * (n + 2)
    nm = n
    if fabs(x) < 1e-100:
        bl[0] = 1.0
        dl[1] = 0.5
        return (bl, dl)
    if x <= 12.0:
        x2 = x * x
        for k in range(n + 1):
            bk = 1.0
            r = 1.0
            for i in range(1, 51):
                r = -0.25 * r * x2 / (i * (i + k))
                bk += r
                if fabs(r) < fabs(bk) * 1.0e-15:
                    break
            bl[k] = bk
            if k >= 1:
                dl[k - 1] = -0.5 * x / k * bk
        uk = 1.0
        r = 1.0
        for i in range(1, 51):
            r = -0.25 * r * x2 / (i * (i + n + 1.0))
            uk += r
            if fabs(r) < fabs(uk) * 1.0e-15:
                break
        dl[n] = -0.5 * x / (n + 1.0) * uk
        return (bl, dl)
    if n == 0:
        nm = 1
    m = msta1(x, 200)
    if m < nm:
        nm = m
    else:
        m = msta2(x, nm, 15)
    bs = 0.0
    f = 0.0
    f0 = 0.0
    f1 = 1e-100
    for k in range(m, -1, -1):
        f = 2.0 * (k + 1.0) * f1 / x - f0
        if k <= nm:
            bl[k] = f
        if k % 2 == 0:
            bs += 2.0 * f
        f0 = f1
        f1 = f
    bg = bs - f
    for k in range(nm + 1):
        bl[k] /= bg
    r0 = 1.0
    for k in range(1, nm + 1):
        r0 = 2.0 * r0 * k / x
        bl[k] *= r0
    dl[0] = -0.5 * x * bl[1]
    for k in range(1, nm + 1):
        dl[k] = 2.0 * k / x * (bl[k - 1] - bl[k])
    return (bl, dl)

def lamv(v: float, x: float):
    """Lambda function of order v: (vl[0..n], dl[0..n]) for n = int(v)."""
    x = fabs(x)
    x2 = x * x
    n = int(v)
    v0 = v - n
    sz = max(n, 1) + 2
    vl = [0.0] * sz
    dl = [0.0] * sz
    if x <= 12.0:
        for k in range(n + 1):
            vk = v0 + k
            bk = 1.0
            r = 1.0
            for i in range(1, 51):
                r = -0.25 * r * x2 / (i * (i + vk))
                bk = bk + r
                if fabs(r) < fabs(bk) * 1.0e-15:
                    break
            vl[k] = bk
            uk = 1.0
            r = 1.0
            for i in range(1, 51):
                r = -0.25 * r * x2 / (i * (i + vk + 1.0))
                uk = uk + r
                if fabs(r) < fabs(uk) * 1.0e-15:
                    break
            dl[k] = -0.5 * x / (vk + 1.0) * uk
        return (vl, dl)
    k0 = 8 if x >= 50.0 else (10 if x >= 35.0 else 11)
    bjv0 = 0.0
    bjv1 = 0.0
    rp2 = 0.63661977236758
    for j in range(2):
        vv = 4.0 * (j + v0) * (j + v0)
        px = 1.0
        rp = 1.0
        for k in range(1, k0 + 1):
            rp = -0.78125e-2 * rp * (vv - (4.0 * k - 3.0) ** 2) * (vv - (4.0 * k - 1.0) ** 2) / (k * (2.0 * k - 1.0) * x2)
            px += rp
        qx = 1.0
        rp = 1.0
        for k in range(1, k0 + 1):
            rp = -0.78125e-2 * rp * (vv - (4.0 * k - 1.0) ** 2) * (vv - (4.0 * k + 1.0) ** 2) / (k * (2.0 * k + 1.0) * x2)
            qx += rp
        qx = 0.125 * (vv - 1.0) * qx / x
        xk = x - (0.5 * (j + v0) + 0.25) * PI
        a0 = sqrt(rp2 / x)
        ck = cos(xk)
        sk = sin(xk)
        if j == 0:
            bjv0 = a0 * (px * ck - qx * sk)
        if j == 1:
            bjv1 = a0 * (px * ck - qx * sk)
    if v0 == 0.0:
        ga = 1.0
    else:
        ga = gam0(v0)
        ga *= v0
    fac = pow(2.0 / x, v0) * ga
    vl[0] = bjv0
    dl[0] = -bjv1 + v0 / x * bjv0
    vl[1] = bjv1
    dl[1] = bjv0 - (1.0 + v0) / x * bjv1
    r0 = 2.0 * (1.0 + v0) / x
    if n <= 1:
        vl[0] *= fac
        dl[0] = fac * dl[0] - v0 / x * vl[0]
        vl[1] *= fac * r0
        dl[1] = fac * r0 * dl[1] - (1.0 + v0) / x * vl[1]
        return (vl, dl)
    if n >= 2 and n <= int(0.9 * x):
        f0 = bjv0
        f1 = bjv1
        for k in range(2, n + 1):
            f = 2.0 * (k + v0 - 1.0) / x * f1 - f0
            f0 = f1
            f1 = f
            vl[k] = f
    elif n >= 2:
        m = msta1(x, 200)
        if m < n:
            n = m
        else:
            m = msta2(x, n, 15)
        f = 0.0
        f2 = 0.0
        f1 = 1.0e-100
        for k in range(m, -1, -1):
            f = 2.0 * (v0 + k + 1.0) / x * f1 - f2
            if k <= n:
                vl[k] = f
            f2 = f1
            f1 = f
        cs = 0.0
        if fabs(bjv0) > fabs(bjv1):
            cs = bjv0 / f
        else:
            cs = bjv1 / f2
        for k in range(n + 1):
            vl[k] *= cs
    vl[0] *= fac
    for j in range(1, n + 1):
        vl[j] *= fac * r0
        dl[j - 1] = -0.5 * x / (j + v0) * vl[j]
        r0 = 2.0 * (j + v0 + 1) / x * r0
    dl[n] = 2.0 * (v0 + n) * (vl[n - 1] - vl[n]) / x
    return (vl, dl)

def rctj(x: float, n: int):
    """Riccati-Bessel j: (rj[0..n], dj[0..n])."""
    rj = [0.0] * (n + 2)
    dj = [0.0] * (n + 2)
    nm = n
    if fabs(x) < 1.0e-100:
        dj[0] = 1.0
        return (rj, dj)
    rj[0] = sin(x)
    rj[1] = rj[0] / x - cos(x)
    rj0 = rj[0]
    rj1 = rj[1]
    if n >= 2:
        m = msta1(x, 200)
        if m < n:
            nm = m
        else:
            m = msta2(x, n, 15)
        f0 = 0.0
        f1 = 1.0e-100
        f = 0.0
        for k in range(m, -1, -1):
            f = (2.0 * k + 3.0) * f1 / x - f0
            if k <= nm:
                rj[k] = f
            f0 = f1
            f1 = f
        cs = rj0 / f if fabs(rj0) > fabs(rj1) else rj1 / f0
        for k in range(nm + 1):
            rj[k] = cs * rj[k]
    dj[0] = cos(x)
    for k in range(1, nm + 1):
        dj[k] = -k * rj[k] / x + rj[k - 1]
    return (rj, dj)

def rcty(x: float, n: int):
    """Riccati-Bessel y: (ry[0..n], dy[0..n])."""
    ry = [0.0] * (n + 2)
    dy = [0.0] * (n + 2)
    nm = n
    if x < 1.0e-60:
        for k in range(n + 1):
            ry[k] = -1.0e+300
            dy[k] = 1.0e+300
        ry[0] = -1.0
        dy[0] = 0.0
        return (ry, dy)
    ry[0] = -cos(x)
    ry[1] = ry[0] / x - sin(x)
    rf0 = ry[0]
    rf1 = ry[1]
    k = 2
    while k <= n:
        rf2 = (2.0 * k - 1.0) * rf1 / x - rf0
        if fabs(rf2) > 1.0e+300:
            break
        ry[k] = rf2
        rf0 = rf1
        rf1 = rf2
        k += 1
    nm = k - 1
    dy[0] = sin(x)
    for k in range(1, nm + 1):
        dy[k] = -k * ry[k] / x + ry[k - 1]
    return (ry, dy)

# ---------------------------------------------------------------- Legendre Q
def lqn(x: float, n: int):
    """Legendre functions of the second kind Qn(x), Qn'(x), n = 0..n (n >= 1)."""
    qn = [0.0] * (n + 1)
    qd = [0.0] * (n + 1)
    eps = 1.0e-14
    if fabs(x) == 1.0:
        for k in range(n + 1):
            qn[k] = 1.0e300
            qd[k] = 1.0e300
        return (qn, qd)
    if x <= 1.021:
        x2 = fabs((1.0 + x) / (1.0 - x))
        q0 = 0.5 * log(x2)
        q1 = x * q0 - 1.0
        qn[0] = q0
        qn[1] = q1
        qd[0] = 1.0 / (1.0 - x * x)
        qd[1] = qn[0] + x * qd[0]
        for k in range(2, n + 1):
            qf = ((2.0 * k - 1.0) * x * q1 - (k - 1.0) * q0) / k
            qn[k] = qf
            qd[k] = (qn[k - 1] - x * qf) * k / (1.0 - x * x)
            q0 = q1
            q1 = qf
    else:
        qc1 = 0.0 if n == 0 else 1.0 / x
        qc2 = 1.0 / x
        for j in range(1, n + 1):
            qc2 *= j / ((2.0 * j + 1.0) * x)
            if j == n - 1:
                qc1 = qc2
        for l in range(2):
            nl = n + l
            qf = 1.0
            qr = 1.0
            for k in range(1, 501):
                qr = qr * (0.5 * nl + k - 1.0) * (0.5 * (nl - 1) + k) / ((nl + k - 0.5) * k * x * x)
                qf += qr
                if fabs(qr / qf) < eps:
                    break
            if l == 0:
                qn[n - 1] = qf * qc1
            else:
                qn[n] = qf * qc2
        qf2 = qn[n]
        qf1 = qn[n - 1]
        for k in range(n, 1, -1):
            qf0 = ((2 * k - 1.0) * x * qf1 - k * qf2) / (k - 1.0)
            qn[k - 2] = qf0
            qf2 = qf1
            qf1 = qf0
        qd[0] = 1.0 / (1.0 - x * x)
        for k in range(1, n + 1):
            qd[k] = k * (qn[k - 1] - x * qn[k]) / (1.0 - x * x)
    return (qn, qd)

def lqmn(x: float, m: int, n: int):
    """Associated Legendre Q_mn(x) and derivative; row-major (m+1) x (n+1) lists."""
    w = n + 1
    qm = [0.0] * ((m + 1) * w)
    qd = [0.0] * ((m + 1) * w)
    if fabs(x) == 1.0:
        for i in range((m + 1) * w):
            qm[i] = 1e300
            qd[i] = 1e300
        return (qm, qd)
    ls = 1
    if fabs(x) > 1.0:
        ls = -1
    xs = ls * (1.0 - x * x)
    xq = sqrt(xs)
    q0 = 0.5 * log(fabs((x + 1.0) / (x - 1.0)))
    if fabs(x) < 1.0001:
        qm[0] = q0
        qm[1] = x * q0 - 1.0
        qm[w] = -1.0 / xq
        qm[w + 1] = -ls * xq * (q0 + x / (1. - x * x))
        for i in range(2):
            for j in range(2, n + 1):
                qm[i * w + j] = ((2.0 * j - 1.) * x * qm[i * w + j - 1] - (j + i - 1) * qm[i * w + j - 2]) / (j - i)
        for i in range(2, m + 1):
            for j in range(n + 1):
                qm[i * w + j] = -2.0 * (i - 1.0) * x / xq * qm[(i - 1) * w + j] - ls * (j + i - 1.0) * (j - i + 2.0) * qm[(i - 2) * w + j]
    else:
        km = 0
        if fabs(x) > 1.1:
            km = 40 + m + n
        else:
            km = (40 + m + n) * int(-1. - 1.8 * log(x - 1.))
        qf2 = 0.0
        qf1 = 1.0
        qf0 = 0.0
        for k in range(km, -1, -1):
            qf0 = ((2.0 * k + 3.0) * x * qf1 - (k + 2.0) * qf2) / (k + 1.0)
            if k <= n:
                qm[k] = qf0
            qf2 = qf1
            qf1 = qf0
        for k in range(n + 1):
            qm[k] *= q0 / qf0
        qf2 = 0.0
        qf1 = 1.0
        for k in range(km, -1, -1):
            qf0 = ((2.0 * k + 3.0) * x * qf1 - (k + 1.0) * qf2) / (k + 2.0)
            if k <= n:
                qm[w + k] = qf0
            qf2 = qf1
            qf1 = qf0
        q10 = -1.0 / xq
        for k in range(n + 1):
            qm[w + k] *= q10 / qf0
        for j in range(n + 1):
            a0 = qm[j]
            a1 = qm[w + j]
            for i in range(m - 1):
                qf = -2. * (i + 1.) * x / xq * a1 + (j - i) * (j + i + 1.) * a0
                qm[(i + 2) * w + j] = qf
                a0 = a1
                a1 = qf
    qd[0] = ls / xs
    for j in range(1, n + 1):
        qd[j] = ls * j * (qm[j - 1] - x * qm[j]) / xs
    for i in range(1, m + 1):
        for j in range(n + 1):
            qd[i * w + j] = ls * i * x / xs * qm[i * w + j] + (i + j) * (j - i + 1.) / xq * qm[(i - 1) * w + j]
    return (qm, qd)

# ------------------------------------------------- integrals of Bessel funcs
def itjya(x: float):
    if x == 0.0:
        return (0.0, 0.0)
    eps = 1.0e-12
    if x <= 20.0:
        x2 = x * x
        tj = x
        r = x
        for k in range(1, 61):
            r = -0.25 * r * (2.0 * k - 1.0) / (2.0 * k + 1.0) / (k * k) * x2
            tj += r
            if fabs(r) < fabs(tj) * eps:
                break
        ty1 = (EL + log(x / 2.0)) * tj
        rs = 0.0
        ty2 = 1.0
        r = 1.0
        for k in range(1, 61):
            r = -0.25 * r * (2.0 * k - 1.0) / (2.0 * k + 1.0) / (k * k) * x2
            rs += 1.0 / k
            r2 = r * (rs + 1.0 / (2.0 * k + 1.0))
            ty2 += r2
            if fabs(r2) < fabs(ty2) * eps:
                break
        return (tj, (ty1 - x * ty2) * 2.0 / PI)
    a = [0.0] * 18
    a0 = 1.0
    a1 = 5.0 / 8.0
    a[0] = a1
    for k in range(1, 17):
        af = ((1.5 * (k + 0.5) * (k + 5.0 / 6.0) * a1 - 0.5 * (k + 0.5) * (k + 0.5) * (k - 0.5) * a0)) / (k + 1.0)
        a[k] = af
        a0 = a1
        a1 = af
    bf = 1.0
    r = 1.0
    for k in range(1, 9):
        r = -r / (x * x)
        bf += a[2 * k - 1] * r
    bg = a[0] / x
    r = 1.0 / x
    for k in range(1, 9):
        r = -r / (x * x)
        bg += a[2 * k] * r
    xp = x + 0.25 * PI
    rc = sqrt(2.0 / (PI * x))
    return (1.0 - rc * (bf * cos(xp) + bg * sin(xp)), rc * (bg * cos(xp) - bf * sin(xp)))

def ittjya(x: float):
    if x == 0.0:
        return (0.0, -1.0e+300)
    if x <= 20.0:
        ttj = 1.0
        r = 1.0
        for k in range(2, 101):
            r = -0.25 * r * (k - 1.0) / (k * k * k) * x * x
            ttj += r
            if fabs(r) < fabs(ttj) * 1.0e-12:
                break
        ttj *= 0.125 * x * x
        e0 = 0.5 * (PI * PI / 6.0 - EL * EL) - (0.5 * log(x / 2.0) + EL) * log(x / 2.0)
        b1 = EL + log(x / 2.0) - 1.5
        rs = 1.0
        r = -1.0
        for k in range(2, 101):
            r = -0.25 * r * (k - 1.0) / (k * k * k) * x * x
            rs = rs + 1.0 / k
            r2 = r * (rs + 1.0 / (2.0 * k) - (EL + log(x / 2.0)))
            b1 = b1 + r2
            if fabs(r2) < fabs(b1) * 1.0e-12:
                break
        return (ttj, 2.0 / PI * (e0 + 0.125 * x * x * b1))
    a0 = sqrt(2.0 / (PI * x))
    bj0 = 0.0
    by0 = 0.0
    bj1 = 0.0
    by1 = 0.0
    for l in range(2):
        vt = 4.0 * l * l
        px = 1.0
        r = 1.0
        for k in range(1, 15):
            r = -0.0078125 * r * (vt - (4.0 * k - 3.0) ** 2) / (x * k) * (vt - (4.0 * k - 1.0) ** 2) / ((2.0 * k - 1.0) * x)
            px += r
            if fabs(r) < fabs(px) * 1.0e-12:
                break
        qx = 1.0
        r = 1.0
        for k in range(1, 15):
            r = -0.0078125 * r * (vt - (4.0 * k - 1.0) ** 2) / (x * k) * (vt - (4.0 * k + 1.0) ** 2) / ((2.0 * k + 1.0) * x)
            qx += r
            if fabs(r) < fabs(qx) * 1.0e-12:
                break
        qx = 0.125 * (vt - 1.0) / x * qx
        xk = x - (0.25 + 0.5 * l) * PI
        bj1 = a0 * (px * cos(xk) - qx * sin(xk))
        by1 = a0 * (px * sin(xk) + qx * cos(xk))
        if l == 0:
            bj0 = bj1
            by0 = by1
    t = 2.0 / x
    g0 = 1.0
    r0 = 1.0
    for k in range(1, 11):
        r0 = -r0 * k * k * t * t
        g0 += r0
    g1 = 1.0
    r1 = 1.0
    for k in range(1, 11):
        r1 = -r1 * k * (k + 1.0) * t * t
        g1 += r1
    return (2.0 * g1 * bj0 / (x * x) - g0 * bj1 / x + EL + log(x / 2.0), 2.0 * g1 * by0 / (x * x) - g0 * by1 / x)

_AIK = [0.625, 1.0078125, 2.5927734375, 9.1868591308594, 4.1567974090576e+1,
        2.2919635891914e+2, 1.491504060477e+3, 1.1192354495579e+4,
        9.515939374212e+4, 9.0412425769041e+5]

def itika(x: float):
    if x == 0.0:
        return (0.0, 0.0)
    x2 = 0.0
    ti = 0.0
    if x < 20.0:
        x2 = x * x
        ti = 1.0
        r = 1.0
        for k in range(1, 51):
            r = 0.25 * r * (2 * k - 1.0) / (2 * k + 1.0) / (k * k) * x2
            ti += r
            if fabs(r / ti) < 1.0e-12:
                break
        ti *= x
    else:
        ti = 1.0
        r = 1.0
        for k in range(1, 11):
            r = r / x
            ti += _AIK[k - 1] * r
        ti = (1.0 / sqrt(2.0 * PI * x)) * exp(x) * ti
    tk = 0.0
    if x < 12.0:
        e0 = EL + log(x / 2.0)
        b1 = 1.0 - e0
        b2 = 0.0
        rs = 0.0
        r = 1.0
        tw = 0.0
        for k in range(1, 51):
            r = 0.25 * r * (2 * k - 1.0) / (2 * k + 1.0) / (k * k) * x2
            b1 += r * (1.0 / (2 * k + 1) - e0)
            rs += 1.0 / k
            b2 += r * rs
            tk = b1 + b2
            if fabs((tk - tw) / tk) < 1.0e-12:
                break
            tw = tk
        tk *= x
    else:
        tk = 1.0
        r = 1.0
        for k in range(1, 11):
            r = -r / x
            tk = tk + _AIK[k - 1] * r
        tk = PI / 2.0 - sqrt(PI / (2.0 * x)) * tk * exp(-x)
    return (ti, tk)

_CIK = [1.625, 4.1328125, 1.45380859375, 6.553353881835, 3.6066157150269, 2.3448727161884, 1.7588273098916, 1.4950639538279]

def ittika(x: float):
    if x == 0.0:
        return (0.0, 1.0e+300)
    tti = 0.0
    if x < 40.0:
        tti = 1.0
        r = 1.0
        for k in range(2, 51):
            r = 0.25 * r * (k - 1.0) / (k * k * k) * x * x
            tti += r
            if fabs(r / tti) < 1.0e-12:
                break
        tti *= 0.125 * x * x
    else:
        tti = 1.0
        r = 1.0
        for k in range(1, 9):
            r = r / x
            tti += _CIK[k - 1] * r
        tti = tti * exp(x) / (x * sqrt(2.0 * PI * x))
    ttk = 0.0
    if x <= 12.0:
        e0 = (0.5 * log(x / 2.0) + EL) * log(x / 2.0) + PI * PI / 24.0 + 0.5 * EL * EL
        b1 = 1.5 - (EL + log(x / 2.0))
        rs = 1.0
        r = 1.0
        for k in range(2, 51):
            r = 0.25 * r * (k - 1.0) / (k * k * k) * x * x
            rs += 1.0 / k
            r2 = r * (rs + 1.0 / (2.0 * k) - (EL + log(x / 2.0)))
            b1 += r2
            if fabs(r2 / b1) < 1.0e-12:
                break
        ttk = e0 - 0.125 * x * x * b1
    else:
        ttk = 1.0
        r = 1.0
        for k in range(1, 9):
            r = -r / x
            ttk += _CIK[k - 1] * r
        ttk = ttk * exp(-x) / (x * sqrt(2.0 / (PI * x)))
    return (tti, ttk)

def it1j0y0(x: float):
    flag = x < 0
    if flag:
        x = -x
    t = itjya(x)
    if flag:
        return (-t[0], NAN)
    return t

def it2j0y0(x: float):
    flag = x < 0
    if flag:
        x = -x
    t = ittjya(x)
    if flag:
        return (t[0], NAN)
    return t

def it1i0k0(x: float):
    flag = x < 0
    if flag:
        x = -x
    t = itika(x)
    if flag:
        return (-t[0], NAN)
    return t

def it2i0k0(x: float):
    flag = x < 0
    if flag:
        x = -x
    t = ittika(x)
    if flag:
        return (t[0], NAN)
    return t

# ------------------------------------------------- integrals of Struve H0, L0
def itth0(x: float) -> float:
    s = 1.0
    r = 1.0
    if x < 24.5:
        for k in range(1, 61):
            r = -r * x * x * (2.0 * k - 1.0) / pow(2.0 * k + 1.0, 3.0)
            s += r
            if fabs(r) < fabs(s) * 1.0e-12:
                break
        return PI / 2.0 - 2.0 / PI * x * s
    for k in range(1, 11):
        r = -r * pow(2.0 * k - 1.0, 3.0) / ((2.0 * k + 1.0) * x * x)
        s += r
        if fabs(r) < fabs(s) * 1.0e-12:
            break
    tth = 2.0 / (PI * x) * s
    t = 8.0 / x
    xt = x + 0.25 * PI
    f0 = (((((0.18118e-2 * t - 0.91909e-2) * t + 0.017033) * t - 0.9394e-3) * t - 0.051445) * t - 0.11e-5) * t + 0.7978846
    g0 = (((((-0.23731e-2 * t + 0.59842e-2) * t + 0.24437e-2) * t - 0.0233178) * t + 0.595e-4) * t + 0.1620695) * t
    tty = (f0 * sin(xt) - g0 * cos(xt)) / (sqrt(x) * x)
    return tth + tty

def itsh0(x: float) -> float:
    r = 1.0
    if x <= 30.0:
        s = 0.5
        for k in range(1, 101):
            rd = 0.5 if k == 1 else 1.0
            r = -r * rd * k / (k + 1.0) * pow(x / (2.0 * k + 1.0), 2.0)
            s += r
            if fabs(r) < fabs(s) * 1.0e-12:
                break
        return 2.0 / PI * x * x * s
    s = 1.0
    for k in range(1, 13):
        r = -r * k / (k + 1.0) * pow((2.0 * k + 1.0) / x, 2.0)
        s += r
        if fabs(r) < fabs(s) * 1.0e-12:
            break
    s0 = s / (PI * x * x) + 2.0 / PI * (log(2.0 * x) + 0.57721566490153)
    a = [0.0] * 25
    a0 = 1.0
    a1 = 5.0 / 8.0
    a[0] = a1
    for k in range(1, 21):
        af = ((1.5 * (k + 0.5) * (k + 5.0 / 6.0) * a1 - 0.5 * (k + 0.5) * (k + 0.5) * (k - 0.5) * a0)) / (k + 1.0)
        a[k] = af
        a0 = a1
        a1 = af
    bf = 1.0
    r = 1.0
    for k in range(1, 11):
        r = -r / (x * x)
        bf += a[2 * k - 1] * r
    bg = a[0] * x
    r = 1.0 / x
    for k in range(1, 10):
        r = -r / (x * x)
        bg += a[2 * k] * r
    xp = x + 0.25 * PI
    ty = sqrt(2.0 / (PI * x)) * (bg * cos(xp) - bf * sin(xp))
    return ty + s0

def itsl0(x: float) -> float:
    r = 1.0
    if x <= 20.0:
        s = 0.5
        for k in range(1, 101):
            rd = 0.5 if k == 1 else 1.0
            r = r * rd * k / (k + 1.0) * pow(x / (2.0 * k + 1.0), 2.0)
            s += r
            if fabs(r / s) < 1.0e-12:
                break
        return 2.0 / PI * x * x * s
    s = 1.0
    for k in range(1, 11):
        r = r * k / (k + 1.0) * pow((2.0 * k + 1.0) / x, 2.0)
        s += r
        if fabs(r / s) < 1.0e-12:
            break
    s0 = -s / (PI * x * x) + 2.0 / PI * (log(2.0 * x) + 0.57721566490153)
    a = [0.0] * 18
    a0 = 1.0
    a1 = 5.0 / 8.0
    a[0] = a1
    for k in range(1, 11):
        af = ((1.5 * (k + .50) * (k + 5.0 / 6.0) * a1 - 0.5 * pow(k + 0.5, 2.0) * (k - 0.5) * a0)) / (k + 1.0)
        a[k] = af
        a0 = a1
        a1 = af
    ti = 1.0
    r = 1.0
    for k in range(1, 11):
        r = r / x
        ti += a[k - 1] * r
    return ti / sqrt(2 * PI * x) * exp(x) + s0

def itstruve0(x: float) -> float:
    if x < 0:
        x = -x
    return itsh0(x)

def it2struve0(x: float) -> float:
    flag = x < 0
    if flag:
        x = -x
    out = itth0(x)
    if flag:
        out = PI - out
    return out

def itmodstruve0(x: float) -> float:
    if x < 0:
        x = -x
    return itsl0(x)

# ------------------------------------------------------------ Airy integrals
def itairy(x: float):
    """(int_0^x Ai, int_0^x Bi, int_0^-x Ai, int_0^-x Bi) after scipy's wrapper."""
    a = [0.569444444444444, 0.891300154320988, 0.226624344493027e+01,
         0.798950124766861e+01, 0.360688546785343e+02, 0.198670292131169e+03,
         0.129223456582211e+04, 0.969483869669600e+04, 0.824184704952483e+05,
         0.783031092490225e+06, 0.822210493622814e+07, 0.945557399360556e+08,
         0.118195595640730e+10, 0.159564653040121e+11, 0.231369166433050e+12,
         0.358622522796969e+13]
    c1 = 0.355028053887817
    c2 = 0.258819403792807
    sr3 = 1.732050807568877
    q0 = 1.0 / 3.0
    q1 = 2.0 / 3.0
    q2 = 1.4142135623730951
    eps = 1e-5
    apt = 0.0
    bpt = 0.0
    ant = 0.0
    bnt = 0.0
    if x == 0.0:
        return (0.0, 0.0, 0.0, 0.0)
    if fabs(x) <= 9.25:
        for l in range(2):
            x = x * (-1.0 if l == 1 else 1.0)
            fx = x
            r = x
            for k in range(1, 41):
                r = r * (3.0 * k - 2.0) / (3.0 * k + 1.0) * x / (3.0 * k) * x / (3.0 * k - 1.0) * x
                fx += r
                if fabs(r) < fabs(fx) * eps:
                    break
            gx = 0.5 * x * x
            r = gx
            for k in range(1, 41):
                r = r * (3.0 * k - 1.0) / (3.0 * k + 2.0) * x / (3.0 * k) * x / (3.0 * k + 1.0) * x
                gx += r
                if fabs(r) < fabs(gx) * eps:
                    break
            ant = c1 * fx - c2 * gx
            bnt = sr3 * (c1 * fx + c2 * gx)
            if l == 0:
                apt = ant
                bpt = bnt
            else:
                ant = -ant
                bnt = -bnt
                x = -x
    else:
        xe = x * sqrt(x) / 1.5
        xp6 = 1.0 / sqrt(6.0 * PI * xe)
        su1 = 1.0
        r = 1.0
        xr1 = 1.0 / xe
        for k in range(1, 17):
            r = -r * xr1
            su1 += a[k - 1] * r
        su2 = 1.0
        r = 1.0
        for k in range(1, 17):
            r = r * xr1
            su2 += a[k - 1] * r
        apt = q0 - exp(-xe) * xp6 * su1
        bpt = 2.0 * exp(xe) * xp6 * su2
        su3 = 1.0
        r = 1.0
        xr2 = 1.0 / (xe * xe)
        for k in range(1, 9):
            r = -r * xr2
            su3 += a[2 * k - 1] * r
        su4 = a[0] * xr1
        r = xr1
        for k in range(1, 8):
            r = -r * xr2
            su4 += a[2 * k] * r
        su5 = su3 + su4
        su6 = su3 - su4
        ant = q1 - q2 * xp6 * (su5 * cos(xe) - su6 * sin(xe))
        bnt = q2 * xp6 * (su5 * sin(xe) + su6 * cos(xe))
    return (apt, bpt, ant, bnt)

def itairy_signed(x: float):
    neg = x < 0 or (x == 0.0 and 1.0 / x < 0)
    if neg:
        x = -x
    t = itairy(x)
    if neg:
        return (-t[2], -t[3], -t[0], -t[1])
    return t

# -------------------------------------------------- Airy / Fresnel / erf zeros
import cmath
from . import _bessel as _B

def airyzo(nt: int, kf: int):
    xa = [0.0] * nt
    xb = [0.0] * nt
    xc = [0.0] * nt
    xd = [0.0] * nt
    for i in range(1, nt + 1):
        rt0 = 0.0
        u = 0.0
        u1 = 0.0
        if kf == 1:
            u = 3.0 * PI * (4.0 * i - 1) / 8.0
            u1 = 1 / (u * u)
        elif kf == 2:
            if i == 1:
                rt0 = -1.17371
            else:
                u = 3.0 * PI * (4.0 * i - 3.0) / 8.0
                u1 = 1 / (u * u)
        if rt0 == 0:
            rt0 = -pow(u * u, 1.0 / 3.0) * (1.0 + u1 * (5.0 / 48.0 + u1 * (-5.0 / 36.0 + u1 * (77125.0 / 82944.0 + u1 * (-108056875.0 / 6967296.0)))))
        rt = 0.0
        err = 1.0
        ad = 0.0
        bd = 0.0
        while True:
            t = _B.airy(rt0)
            if kf == 1:
                rt = rt0 - t[0] / t[1]
            else:
                rt = rt0 - t[2] / t[3]
            err = fabs((rt - rt0) / rt)
            if err <= 1.0e-12:
                break
            rt0 = rt
        xa[i - 1] = rt
        t2 = _B.airy(rt)
        xd[i - 1] = t2[1] if kf == 1 else t2[3]
    for i in range(1, nt + 1):
        rt0 = 0.0
        u = 0.0
        u1 = 0.0
        if kf == 1:
            if i == 1:
                rt0 = -1.01879
            else:
                u = 3.0 * PI * (4.0 * i - 3.0) / 8.0
                u1 = 1 / (u * u)
        elif kf == 2:
            if i == 1:
                rt0 = -2.29444
            else:
                u = 3.0 * PI * (4.0 * i - 1.0) / 8.0
                u1 = 1 / (u * u)
        if rt0 == 0:
            rt0 = -pow(u * u, 1.0 / 3.0) * (1.0 + u1 * (-7.0 / 48.0 + u1 * (35.0 / 288.0 + u1 * (-181223.0 / 207360.0 + u1 * (18683371.0 / 1244160.0)))))
        rt = 0.0
        while True:
            t = _B.airy(rt0)
            if kf == 1:
                rt = rt0 - t[1] / (t[0] * rt0)
            else:
                rt = rt0 - t[3] / (t[2] * rt0)
            err = fabs((rt - rt0) / rt)
            if err <= 1.0e-12:
                break
            rt0 = rt
        xb[i - 1] = rt
        t2 = _B.airy(rt)
        xc[i - 1] = t2[0] if kf == 1 else t2[2]
    return (xa, xb, xc, xd)

def cfc(z: complex):
    eps = 1.0e-14
    w0 = abs(z)
    zp = 0.5 * PI * z * z
    zp2 = zp * zp
    c = complex(0.0, 0.0)
    zd = cmath.cos(0.5 * PI * z * z)
    if z == complex(0.0, 0.0):
        return (c, zd)
    elif w0 <= 2.5:
        cr = z
        c = cr
        wa0 = 0.0
        for k in range(1, 81):
            cr = -0.5 * cr * (4.0 * k - 3.0) / float(k) / (2.0 * k - 1.0) / (4.0 * k + 1.0) * zp2
            c += cr
            wa = abs(c)
            if (fabs((wa - wa0) / wa) < eps) and (k > 10):
                return (c, zd)
            wa0 = wa
        return (c, zd)
    elif w0 > 2.5 and w0 < 4.5:
        c = complex(0.0, 0.0)
        cf1 = complex(0.0, 0.0)
        cf0 = complex(1.0e-100, 0.0)
        cf = complex(0.0, 0.0)
        for k in range(85, -1, -1):
            cf = (2.0 * k + 3.0) * cf0 / zp - cf1
            if k % 2 == 0:
                c += cf
            cf1 = cf0
            cf0 = cf
        c = c * (2.0 / (PI * z) * cmath.sin(zp) / cf)
        return (c, zd)
    else:
        d = complex(0.0, 0.0)
        if (z.imag > -z.real) and (z.imag <= z.real):
            d = complex(0.5, 0.0)
        elif (z.imag > z.real) and (z.imag >= -z.real):
            d = complex(0.0, 0.5)
        elif (z.imag < -z.real) and (z.imag >= z.real):
            d = complex(-0.5, 0.0)
        else:
            d = complex(0.0, -0.5)
        cr = complex(1.0, 0.0)
        cf = complex(1.0, 0.0)
        for k in range(1, 21):
            cr = -0.25 * cr * (4.0 * k - 1.0) * (4.0 * k - 3.0) / zp2
            cf += cr
        cr = 1.0 / (PI * z * z)
        cg = cr
        for k in range(1, 13):
            cr = -0.25 * cr * (4.0 * k + 1.0) * (4.0 * k - 1.0) / zp2
            cg += cr
        c = d + (cf * cmath.sin(zp) - cg * cmath.cos(zp)) / (PI * z)
        return (c, zd)

def cfs(z: complex):
    eps = 1.0e-14
    w0 = abs(z)
    zp = 0.5 * PI * z * z
    zp2 = zp * zp
    zd = cmath.sin(0.5 * PI * z * z)
    s = complex(0.0, 0.0)
    if z == complex(0.0, 0.0):
        return (s, zd)
    elif w0 <= 2.5:
        s = z * zp / 3.0
        cr = s
        wb0 = 0.0
        for k in range(1, 81):
            cr = -0.5 * cr * (4.0 * k - 1.0) / float(k) / (2.0 * k + 1.0) / (4.0 * k + 3.0) * zp2
            s += cr
            wb = abs(s)
            if (fabs(wb - wb0) < eps) and (k > 10):
                return (s, zd)
            wb0 = wb
        return (s, zd)
    elif w0 > 2.5 and w0 < 4.5:
        s = complex(0.0, 0.0)
        cf1 = complex(0.0, 0.0)
        cf0 = complex(1.0e-100, 0.0)
        cf = complex(0.0, 0.0)
        for k in range(85, -1, -1):
            cf = (2.0 * k + 3.0) * cf0 / zp - cf1
            if k % 2 == 1:
                s += cf
            cf1 = cf0
            cf0 = cf
        s = 2.0 / (PI * z) * cmath.sin(zp) / cf * s
        return (s, zd)
    else:
        d = complex(0.0, 0.0)
        if (z.imag > -z.real) and (z.imag <= z.real):
            d = complex(0.5, 0.0)
        elif (z.imag > z.real) and (z.imag >= -z.real):
            d = complex(0.0, -0.5)
        elif (z.imag < -z.real) and (z.imag >= z.real):
            d = complex(-0.5, 0.0)
        else:
            d = complex(0.0, 0.5)
        cr = complex(1.0, 0.0)
        cf = complex(1.0, 0.0)
        for k in range(1, 21):
            cr = -0.25 * cr * (4.0 * k - 1.0) * (4.0 * k - 3.0) / zp2
            cf += cr
        cr = complex(1.0, 0.0)
        cg = complex(1.0, 0.0)
        for k in range(1, 13):
            cr = -0.25 * cr * (4.0 * k + 1.0) * (4.0 * k - 1.0) / zp2
            cg += cr
        cg = cg / (PI * z * z)
        s = d - (cf * cmath.cos(zp) + cg * cmath.sin(zp)) / (PI * z)
        return (s, zd)

def fcszo(kf: int, nt: int) -> List[complex]:
    zo = [complex(0.0, 0.0)] * nt
    w = 0.0
    psq = 0.0
    for nr in range(1, nt + 1):
        if kf == 1:
            psq = sqrt(4.0 * nr - 1.0)
        if kf == 2:
            psq = 2.0 * sqrt(float(nr))
        px = psq - log(PI * psq) / (PI * PI * psq * psq * psq)
        py = log(PI * psq) / (PI * psq)
        z = complex(px, py)
        if kf == 2:
            if nr == 2:
                z = complex(2.8334, 0.2443)
            if nr == 3:
                z = complex(3.4674, 0.2185)
            if nr == 4:
                z = complex(4.0025, 0.2008)
        it = 0
        while True:
            it += 1
            t = cfc(z) if kf == 1 else cfs(z)
            zf = t[0]
            zd = t[1]
            zp = complex(1.0, 0.0)
            for i in range(1, nr):
                zp *= (z - zo[i - 1])
            zfd = zf / zp
            zq = complex(0.0, 0.0)
            for i in range(1, nr):
                zw = complex(1.0, 0.0)
                for j in range(1, nr):
                    if j == i:
                        continue
                    zw *= (z - zo[j - 1])
                zq += zw
            zgd = (zd - zq * zfd) / zp
            z -= zfd / zgd
            w0 = w
            w = abs(z)
            if not ((it <= 50) and (fabs((w - w0) / w) > 1.0e-12)):
                break
        zo[nr - 1] = z
    return zo

def cerf(z: complex):
    eps = 1.0e-12
    x = z.real
    y = z.imag
    x2 = x * x
    er0 = 0.0
    if x <= 3.5:
        er = 1.0
        r = 1.0
        w = 0.0
        for k in range(1, 101):
            r = r * x2 / (k + 0.5)
            er += r
            if fabs(er - w) <= eps * fabs(er):
                break
            w = er
        c0 = 2.0 / sqrt(PI) * x * exp(-x2)
        er0 = c0 * er
    else:
        er = 1.0
        r = 1.0
        for k in range(1, 13):
            r = -r * (k - 0.5) / x2
            er += r
        c0 = exp(-x2) / (x * sqrt(PI))
        er0 = 1.0 - c0 * er
    cder = 2.0 / sqrt(PI) * cmath.exp(-z * z)
    if y == 0.0:
        return (complex(er0, 0.0), cder)
    cs = cos(2.0 * x * y)
    ss = sin(2.0 * x * y)
    er1 = exp(-x2) * (1.0 - cs) / (2.0 * PI * x)
    ei1 = exp(-x2) * ss / (2.0 * PI * x)
    er2 = 0.0
    w1 = 0.0
    for n in range(1, 101):
        er2 += exp(-0.25 * n * n) / (n * n + 4.0 * x2) * (2.0 * x - 2.0 * x * cmath.cosh(complex(n * y, 0.0)).real * cs + n * cmath.sinh(complex(n * y, 0.0)).real * ss)
        if fabs((er2 - w1) / er2) < eps:
            break
        w1 = er2
    c0 = 2.0 * exp(-x2) / PI
    err = er0 + er1 + c0 * er2
    ei2 = 0.0
    w2 = 0.0
    for n in range(1, 101):
        ei2 += exp(-0.25 * n * n) / (n * n + 4.0 * x2) * (2.0 * x * cmath.cosh(complex(n * y, 0.0)).real * ss + n * cmath.sinh(complex(n * y, 0.0)).real * cs)
        if fabs((ei2 - w2) / ei2) < eps:
            break
        w2 = ei2
    return (complex(err, ei1 + c0 * ei2), cder)

def cerzo(nt: int) -> List[complex]:
    zo = [complex(0.0, 0.0)] * nt
    w = 0.0
    for nr in range(1, nt + 1):
        pu = sqrt(PI * (4.0 * nr - 0.5))
        pv = PI * sqrt(2.0 * nr - 0.25)
        px = 0.5 * pu - 0.5 * log(pv) / pu
        py = 0.5 * pu + 0.5 * log(pv) / pu
        z = complex(px, py)
        it = 0
        while True:
            it += 1
            t = cerf(z)
            zf = t[0]
            zd = t[1]
            zp = complex(1.0, 0.0)
            for i in range(1, nr):
                zp *= (z - zo[i - 1])
            zfd = zf / zp
            zq = complex(0.0, 0.0)
            for i in range(1, nr):
                zw = complex(1.0, 0.0)
                for j in range(1, nr):
                    if j == i:
                        continue
                    zw *= (z - zo[j - 1])
                zq += zw
            zgd = (zd - zq * zfd) / zp
            z -= zfd / zgd
            w0 = w
            w = abs(z)
            if not ((it <= 50) and (fabs((w - w0) / w) > 1.0e-11)):
                break
        zo[nr - 1] = z
    return zo

_CA = [-0.703125e-01, 0.112152099609375, -0.5725014209747314, 0.6074042001273483,
       -0.1100171402692467, 0.3038090510922384, -0.1188384262567832, 0.6252951493434797,
       -0.4259392165047669, 0.3646840080706556, -0.3833534661393944, 0.4854014686852901]
_CB = [0.732421875e-01, -0.2271080017089844, 0.1727727502584457, -0.2438052969955606,
       0.5513358961220206, -0.1825775547429318, 0.8328593040162893, -0.5006958953198893,
       0.3836255180230433, -0.3649010818849833, 0.4218971570284096, -0.5827244631566907]
_CA1 = [0.1171875, -0.144195556640625, 0.6765925884246826, -0.6883914268109947,
        0.1215978918765359, -0.3302272294480852, 0.1276412726461746, -0.6656367718817688,
        0.4502786003050393, -0.3833857520742790, 0.4011838599133198, -0.5060568503314727]
_CB1 = [-0.1025390625, 0.2775764465332031, -0.1993531733751297, 0.2724882731126854,
        -0.6038440767050702, 0.1971837591223663, -0.8902978767070678, 0.5310411010968522,
        -0.4043620325107754, 0.3827011346598605, -0.4406481417852278, 0.6065091351222699]

def cy01(kf: int, z: complex):
    rp2 = 2.0 / PI
    ci = complex(0.0, 1.0)
    one = complex(1.0, 0.0)
    a0 = abs(z)
    z1 = z
    z2 = z * z
    if a0 == 0.0:
        cby0 = complex(-1e300, 0.0)
        cby1 = complex(-1e300, 0.0)
        cdy0 = complex(1e300, 0.0)
        cdy1 = complex(1e300, 0.0)
        if kf == 0:
            return (cby0, cdy0)
        elif kf == 1:
            return (cby1, cdy1)
        return (cdy1, -cdy1 / z - (1.0 - 1.0 / (z * z)) * cby1)
    if z.real < 0.0:
        z1 = -z
    cbj0 = one
    cbj1 = one
    cby0 = one
    cby1 = one
    if a0 <= 12.0:
        cbj0 = one
        cr = one
        for k in range(1, 41):
            cr = -0.25 * cr * z2 / float(k * k)
            cbj0 += cr
            if abs(cr) < abs(cbj0) * 1.0e-15:
                break
        cbj1 = one
        cr = one
        for k in range(1, 41):
            cr = -0.25 * cr * z2 / (k * (k + 1.0))
            cbj1 += cr
            if abs(cr) < abs(cbj1) * 1.0e-15:
                break
        cbj1 = cbj1 * (0.5 * z1)
        w0 = 0.0
        cr = one
        cs = complex(0.0, 0.0)
        for k in range(1, 41):
            w0 += 1.0 / k
            cr = -0.25 * cr / float(k * k) * z2
            cp = cr * w0
            cs += cp
            if abs(cp) < abs(cs) * 1.0e-15:
                break
        cby0 = rp2 * (cmath.log(z1 / 2.0) + EL) * cbj0 - rp2 * cs
        w1 = 0.0
        cr = one
        cs = one
        for k in range(1, 41):
            w1 += 1.0 / k
            cr = -0.25 * cr / float(k * (k + 1)) * z2
            cp = cr * (2.0 * w1 + 1.0 / (k + 1.0))
            cs += cp
            if abs(cp) < abs(cs) * 1.0e-15:
                break
        cby1 = rp2 * ((cmath.log(z1 / 2.0) + EL) * cbj1 - 1.0 / z1 - 0.25 * z1 * cs)
    else:
        k0 = 12
        if a0 >= 35.0:
            k0 = 10
        if a0 >= 50.0:
            k0 = 8
        ct1 = z1 - 0.25 * PI
        iz = 1.0 / z1
        iz2 = iz * iz
        cp0 = one
        pw = one
        for k in range(1, k0 + 1):
            pw = pw * iz2
            cp0 += _CA[k - 1] * pw
        cq0 = -0.125 / z1
        pw = iz
        for k in range(1, k0 + 1):
            pw = pw * iz2
            cq0 += _CB[k - 1] * pw
        cu = cmath.sqrt(rp2 / z1)
        cbj0 = cu * (cp0 * cmath.cos(ct1) - cq0 * cmath.sin(ct1))
        cby0 = cu * (cp0 * cmath.sin(ct1) + cq0 * cmath.cos(ct1))
        ct2 = z1 - 0.75 * PI
        cp1 = one
        pw = one
        for k in range(1, k0 + 1):
            pw = pw * iz2
            cp1 += _CA1[k - 1] * pw
        cq1 = 0.375 / z1
        pw = iz
        for k in range(1, k0 + 1):
            pw = pw * iz2
            cq1 = cq1 + _CB1[k - 1] * pw
        cbj1 = cu * (cp1 * cmath.cos(ct2) - cq1 * cmath.sin(ct2))
        cby1 = cu * (cp1 * cmath.sin(ct2) + cq1 * cmath.cos(ct2))
    if z.real < 0.0:
        if z.imag < 0.0:
            cby0 = cby0 - 2.0 * ci * cbj0
        if z.imag > 0.0:
            cby0 = cby0 + 2.0 * ci * cbj0
        if z.imag < 0.0:
            cby1 = -(cby1 - 2.0 * ci * cbj1)
        if z.imag > 0.0:
            cby1 = -(cby1 + 2.0 * ci * cbj1)
        cbj1 = -cbj1
    cdy0 = -cby1
    cdy1 = cby0 - 1.0 / z * cby1
    if kf == 0:
        return (cby0, cdy0)
    elif kf == 1:
        return (cby1, cdy1)
    return (cdy1, -cdy1 / z - (1.0 - 1.0 / (z * z)) * cby1)

def cyzo(nt: int, kf: int, kc: int):
    zo = [complex(0.0, 0.0)] * nt
    zv = [complex(0.0, 0.0)] * nt
    x = 0.0
    y = 0.0
    h = 0.0
    if kc == 0:
        x = -2.4
        y = 0.54
        h = 3.14
    elif kc == 1:
        x = 0.89
        y = 0.0
        h = -3.14
    if kf == 1:
        x = -0.503
    if kf == 2:
        x = 0.577
    z = complex(x, y)
    w = 0.0
    for nr in range(1, nt + 1):
        if nr > 1:
            z = zo[nr - 2] - h
        it = 0
        while True:
            it += 1
            t = cy01(kf, z)
            zf = t[0]
            zd = t[1]
            zp = complex(1.0, 0.0)
            for i in range(1, nr):
                zp *= (z - zo[i - 1])
            zfd = zf / zp
            zq = complex(0.0, 0.0)
            for i in range(1, nr):
                zw = complex(1.0, 0.0)
                for j in range(1, nr):
                    if j == i:
                        continue
                    zw *= (z - zo[j - 1])
                zq += zw
            zgd = (zd - zq * zfd) / zp
            z -= zfd / zgd
            w0 = w
            w = abs(z)
            if not ((it <= 50) and (fabs((w - w0) / w) > 1.0e-12)):
                break
        zo[nr - 1] = z
    for i in range(nt):
        z = zo[i]
        if kf == 0 or kf == 2:
            zv[i] = cy01(1, z)[0]
        elif kf == 1:
            zv[i] = cy01(0, z)[0]
    return (zo, zv)

def besselpoly(a: float, lmbda: float, nu: float) -> float:
    from . import _kernels as K
    factor = 0
    if a == 0.0:
        if nu == 0.0:
            return 1.0 / (lmbda + 1)
        return 0.0
    if nu < 0 and K.floor(nu) == nu:
        nu = -nu
        factor = int(nu) % 2
    sm = exp(nu * log(a)) / (K.gamma(nu + 1) * (lmbda + nu + 1))
    m = 0
    sm_sum = 0.0
    while True:
        sm_sum += sm
        sol = sm
        sm *= -a * a * (lmbda + nu + 1 + 2 * m) / ((nu + m + 1) * (m + 1) * (lmbda + nu + 1 + 2 * m + 2))
        m += 1
        if sm == 0.0:
            break
        relerr = fabs((sm - sol) / sm)
        if not (relerr > 1.0e-17 and m < 1000):
            break
    return -sm_sum if factor else sm_sum

def modified_fresnel(ks: int, x: float):
    """ffk: (f, g) complex pairs. ks=0 plus, ks=1 minus."""
    eps = 1.0e-15
    pp2 = 1.2533141373155
    p2p = 0.7978845608028654
    sg = -1.0 if ks == 1 else 1.0
    xa = fabs(x)
    x2 = x * x
    x4 = x2 * x2
    if x == 0.0:
        fr = 0.5 * sqrt(0.5 * PI)
        return (complex(fr, sg * fr), complex(0.5, 0.0))
    c1 = 0.0
    s1 = 0.0
    if xa <= 2.5:
        xr = p2p * xa
        c1 = xr
        for k in range(1, 51):
            xr = -0.5 * xr * (4.0 * k - 3.0) / k / (2.0 * k - 1.0) / (4.0 * k + 1.0) * x4
            c1 += xr
            if fabs(xr / c1) < eps:
                break
        s1 = p2p * xa * xa * xa / 3.0
        xr = s1
        for k in range(1, 51):
            xr = -0.5 * xr * (4.0 * k - 1.0) / k / (2.0 * k + 1.0) / (4.0 * k + 3.0) * x4
            s1 += xr
            if fabs(xr / s1) < eps:
                break
    elif xa < 5.5:
        m = int(42 + 1.75 * x2)
        xsu = 0.0
        xc = 0.0
        xs = 0.0
        xf1 = 0.0
        xf0 = 1.0e-100
        for k in range(m, -1, -1):
            xf = (2.0 * k + 3.0) * xf0 / x2 - xf1
            if k % 2 == 0:
                xc += xf
            else:
                xs += xf
            xsu += (2.0 * k + 1.0) * xf * xf
            xf1 = xf0
            xf0 = xf
        xq = sqrt(xsu)
        xw = p2p * xa / xq
        c1 = xc * xw
        s1 = xs * xw
    else:
        xr = 1.0
        xf = 1.0
        for k in range(1, 13):
            xr = -0.25 * xr * (4.0 * k - 1.0) * (4.0 * k - 3.0) / x4
            xf += xr
        xr = 1.0 / (2.0 * xa * xa)
        xg = xr
        for k in range(1, 13):
            xr = -0.25 * xr * (4.0 * k + 1.0) * (4.0 * k - 1.0) / x4
            xg += xr
        c1 = 0.5 + (xf * sin(x2) - xg * cos(x2)) / sqrt(2.0 * PI) / xa
        s1 = 0.5 - (xf * cos(x2) + xg * sin(x2)) / sqrt(2.0 * PI) / xa
    fre = pp2 * (0.5 - c1)
    fi0 = pp2 * (0.5 - s1)
    fim = sg * fi0
    xp = x2 + PI / 4.0
    cs = cos(xp)
    ss = sin(xp)
    xq2 = 1.0 / sqrt(PI)
    gre = xq2 * (fre * cs + fi0 * ss)
    gim = sg * xq2 * (fi0 * cs - fre * ss)
    if x < 0.0:
        fre = pp2 - fre
        fim = sg * pp2 - fre
        gre = cos(x2) - gre
        gim = -sg * sin(x2) - gim
    return (complex(fre, fim), complex(gre, gim))
