# Ported from scipy 1.18.1 (BSD-3-Clause)
# Algorithms from xsf (BSD-3) specfun/specfun.h: translation of the Fortran
# routines of Shanjie Zhang & Jianming Jin, "Computation of Special
# Functions" (Wiley 1996), as distributed with scipy.
#
# gamma2, psi_spec, chgu (confluent hypergeometric U) and its helpers.

from ._kernels import exp, log, sqrt, fabs, sin, cos, pow

from C import log10(float) -> float

NAN = float('nan')
INF = float('inf')
PI = 3.141592653589793
EL = 0.5772156649015329

def _ci(x: float) -> int:
    # C (int) cast with saturation (avoid UB/overflow in Python int())
    if x != x:
        return 0
    if x > 2147483647.0:
        return 2147483647
    if x < -2147483648.0:
        return -2147483648
    return int(x)

def _isint(x: float) -> bool:
    return x == float(_ci(x))

_G2 = (1.0000000000000000e+00, 0.5772156649015329e+00, -0.6558780715202538e+00,
       -0.4200263503409520e-01, 0.1665386113822915e+00, -0.4219773455554430e-01,
       -0.9621971527877000e-02, 0.7218943246663000e-02, -0.1165167591859100e-02,
       -0.2152416741149000e-03, 0.1280502823882000e-03, -0.2013485478070000e-04,
       -0.1250493482100000e-05, 0.1133027232000000e-05, -0.2056338417000000e-06,
       0.6116095000000000e-08, 0.5002007500000000e-08, -0.1181274600000000e-08,
       0.1043427000000000e-09, 0.7782300000000000e-11, -0.3696800000000000e-11,
       0.5100000000000000e-12, -0.2060000000000000e-13, -0.5400000000000000e-14,
       0.1400000000000000e-14, 0.1000000000000000e-15)

def gamma2(x: float) -> float:
    ga = 0.0
    if _isint(x):
        if x > 0.0:
            ga = 1.0
            m = _ci(x - 1)
            for k in range(2, m + 1):
                ga *= k
        else:
            ga = 1e300
    else:
        r = 1.0
        z = 0.0
        if fabs(x) > 1.0:
            z = fabs(x)
            m = _ci(z)
            for k in range(1, m + 1):
                r *= (z - k)
            z -= m
        else:
            z = x
        gr = _G2[25]
        k = 25
        while k > 0:
            gr *= z
            gr += _G2[k - 1]
            k -= 1
        ga = 1.0 / (gr * z)
        if fabs(x) > 1.0:
            ga *= r
            if x < 0.0:
                ga = -PI / (x * ga * sin(PI * x))
    return ga

_PSA = (-0.8333333333333e-01, 0.83333333333333333e-02, -0.39682539682539683e-02,
        0.41666666666666667e-02, -0.75757575757575758e-02, 0.21092796092796093e-01,
        -0.83333333333333333e-01, 0.4432598039215686)

def psi_spec(x: float) -> float:
    s = 0.0
    xa = fabs(x)
    ps = 0.0
    if _isint(x) and x <= 0.0:
        return 1e300
    elif _isint(xa):
        n = _ci(xa)
        for k in range(1, n):
            s += 1.0 / k
        ps = -EL + s
    elif _isint(xa + 0.5):
        n = _ci(xa - 0.5)
        for k in range(1, n + 1):
            s += 1.0 / (2.0 * k - 1.0)
        ps = -EL + 2.0 * s - 1.386294361119891
    else:
        if xa < 10.0:
            n = _ci(10.0 - float(_ci(xa)))
            for k in range(n):
                s += 1.0 / (xa + k)
            xa += n
        x2 = 1.0 / (xa * xa)
        ps = log(xa) - 0.5 / xa
        a = _PSA
        ps += x2 * (((((((a[7] * x2 + a[6]) * x2 + a[5]) * x2 + a[4]) * x2 + a[3]) * x2 + a[2]) * x2 + a[1]) * x2 + a[0])
        ps -= s
    if x < 0.0:
        ps -= PI * cos(PI * x) / sin(PI * x) + 1.0 / x
    return ps

def _chgus(x: float, a: float, b: float):
    ident = 100
    ga = gamma2(a)
    gb = gamma2(b)
    gab = gamma2(1.0 + a - b)
    gb2 = gamma2(2.0 - b)
    hu0 = PI / sin(PI * b)
    r1 = hu0 / (gab * gb)
    r2 = hu0 * pow(x, 1.0 - b) / (ga * gb2)
    hu = r1 - r2
    hmax = 0.0
    hmin = 1e300
    h0 = 0.0
    for j in range(1, 151):
        r1 = r1 * (a + j - 1.0) / (j * (b + j - 1.0)) * x
        r2 = r2 * (a - b + j) / (j * (1.0 - b + j)) * x
        hu += r1 - r2
        hua = fabs(hu)
        if hua > hmax:
            hmax = hua
        if hua < hmin:
            hmin = hua
        if fabs(hu - h0) < fabs(hu) * 1e-15:
            break
        h0 = hu
    d1 = log10(hmax)
    d2 = 0.0
    if hmin != 0.0:
        d2 = log10(hmin)
    ident = _ci(15 - (d2 - d1 if d1 - d2 < 0 else d1 - d2))
    return (hu, ident)

def _chgul(x: float, a: float, b: float):
    ident = -100
    aa = a - b + 1.0
    il1 = _isint(a) and a <= 0.0
    il2 = _isint(aa) and aa <= 0.0
    nm = 0
    if il1:
        nm = _ci(fabs(a))
    if il2:
        nm = _ci(fabs(aa))
    hu = 1.0
    r = 1.0
    if il1 or il2:
        for k in range(1, nm + 1):
            r = -r * (a + k - 1.0) * (a - b + k) / (k * x)
            hu += r
        hu *= pow(x, -a)
        ident = 10
    else:
        r0 = 0.0
        ra = 0.0
        for k in range(1, 26):
            r = -r * (a + k - 1.0) * (a - b + k) / (k * x)
            ra = fabs(r)
            if (k > 5 and ra >= r0) or ra < 1e-15:
                break
            r0 = ra
            hu += r
        ident = _ci(fabs(log10(ra)))
        hu *= pow(x, -a)
    return (hu, ident)

def _chgubi(x: float, a: float, b: float):
    ident = -100
    n = _ci(fabs(b - 1))
    rn1 = 1.0
    rn = 1.0
    for j in range(1, n + 1):
        rn *= j
        if j == n - 1:
            rn1 = rn
    ps = psi_spec(a)
    ga = gamma2(a)
    a0 = 0.0
    a1 = 0.0
    a2 = 0.0
    ua = 0.0
    ub = 0.0
    sgn = 1.0 if (n - 1) % 2 == 0 else -1.0
    if b > 0.0:
        a0 = a
        a1 = a - n
        a2 = a1
        ga1 = gamma2(a1)
        ua = sgn / (rn * ga1)
        ub = rn1 / ga * pow(x, float(-n))
    else:
        a0 = a + n
        a1 = a0
        a2 = a
        ga1 = gamma2(a1)
        ua = sgn / (rn * ga) * pow(x, float(n))
        ub = rn1 / ga1
    hm1 = 1.0
    r = 1.0
    hmax = 0.0
    hmin = 1e300
    h0 = 0.0
    for k in range(1, 151):
        r = r * (a0 + k - 1) * x / ((n + k) * k)
        hm1 += r
        hu1 = fabs(hm1)
        if hu1 > hmax:
            hmax = hu1
        if hu1 < hmin:
            hmin = hu1
        if fabs(hm1 - h0) < fabs(hm1) * 1.0e-15:
            break
        h0 = hm1
    da1 = log10(hmax)
    da2 = 0.0
    if hmin != 0:
        da2 = log10(hmin)
    ident = 15 - _ci(fabs(da1 - da2))
    hm1 *= log(x)
    s0 = 0.0
    for m in range(1, n + 1):
        if b >= 0:
            s0 -= 1.0 / m
        if b < 0:
            s0 += (1.0 - a) / (m * (a + m - 1))
    hm2 = ps + 2 * EL + s0
    r = 1.0
    hmax = 0.0
    hmin = 1.0e+300
    for k in range(1, 151):
        s1 = 0.0
        s2 = 0.0
        if b > 0:
            for m in range(1, k + 1):
                s1 -= (m + 2 * a - 2) / (m * (m + a - 1))
            for m in range(1, n + 1):
                s2 += 1.0 / (k + m)
        else:
            for m in range(1, k + n + 1):
                s1 += (1.0 - a) / (m * (m + a - 1))
            for m in range(1, k + 1):
                s2 += 1.0 / m
        hw = 2 * EL + ps + s1 - s2
        r = r * (a0 + k - 1) * x / ((n + k) * k)
        hm2 += r * hw
        hu2 = fabs(hm2)
        if hu2 > hmax:
            hmax = hu2
        if hu2 < hmin:
            hmin = hu2
        if fabs((hm2 - h0) / hm2) < 1.0e-15:
            break
        h0 = hm2
    db1 = log10(hmax)
    db2 = 0.0
    if hmin != 0.0:
        db2 = log10(hmin)
    id1 = 15 - _ci(fabs(db1 - db2))
    if id1 < ident:
        ident = id1
    hm3 = 1.0
    if n == 0:
        hm3 = 0.0
    r = 1.0
    for k in range(1, n):
        r = r * (a2 + k - 1.0) / ((k - n) * k) * x
        hm3 += r
    sa = ua * (hm1 + hm2)
    sb = ub * hm3
    hu = sa + sb
    id2 = 0
    if sa != 0.0:
        id1 = _ci(log10(fabs(sa)))
    if hu != 0.0:
        id2 = _ci(log10(fabs(hu)))
    if sa * sb < 0.0:
        ident -= abs(id1 - id2)
    return (hu, ident)

_TQ = (0.259597723012478e-01, 0.778093339495366e-01, 0.129449135396945e+00,
       0.180739964873425e+00, 0.231543551376029e+00, 0.281722937423262e+00,
       0.331142848268448e+00, 0.379670056576798e+00, 0.427173741583078e+00,
       0.473525841761707e+00, 0.518601400058570e+00, 0.562278900753945e+00,
       0.604440597048510e+00, 0.644972828489477e+00, 0.683766327381356e+00,
       0.720716513355730e+00, 0.755723775306586e+00, 0.788693739932264e+00,
       0.819537526162146e+00, 0.848171984785930e+00, 0.874519922646898e+00,
       0.898510310810046e+00, 0.920078476177628e+00, 0.939166276116423e+00,
       0.955722255839996e+00, 0.969701788765053e+00, 0.981067201752598e+00,
       0.989787895222222e+00, 0.995840525118838e+00, 0.999210123227436e+00)
_WQ = (0.519078776312206e-01, 0.517679431749102e-01, 0.514884515009810e-01,
       0.510701560698557e-01, 0.505141845325094e-01, 0.498220356905502e-01,
       0.489955754557568e-01, 0.480370318199712e-01, 0.469489888489122e-01,
       0.457343797161145e-01, 0.443964787957872e-01, 0.429388928359356e-01,
       0.413655512355848e-01, 0.396806954523808e-01, 0.378888675692434e-01,
       0.359948980510845e-01, 0.340038927249464e-01, 0.319212190192963e-01,
       0.297524915007890e-01, 0.275035567499248e-01, 0.251804776215213e-01,
       0.227895169439978e-01, 0.203371207294572e-01, 0.178299010142074e-01,
       0.152746185967848e-01, 0.126781664768159e-01, 0.100475571822880e-01,
       0.738993116334531e-02, 0.471272992695363e-02, 0.202681196887362e-02)

def _chguit(x: float, a: float, b: float):
    ident = 9
    a1 = a - 1.0
    b1 = b - a - 1.0
    c = 12.0 / x
    hu0 = 0.0
    hu1 = 0.0
    m = 10
    while m <= 100:
        hu1 = 0.0
        g = 0.5 * c / m
        d = g
        for j in range(1, m + 1):
            s = 0.0
            for k in range(1, 31):
                t1 = d + g * _TQ[k - 1]
                t2 = d - g * _TQ[k - 1]
                f1 = exp(-x * t1) * pow(t1, a1) * pow(1.0 + t1, b1)
                f2 = exp(-x * t2) * pow(t2, a1) * pow(1.0 + t2, b1)
                s += _WQ[k - 1] * (f1 + f2)
            hu1 += s * g
            d += 2.0 * g
        if fabs(1.0 - hu0 / hu1) < 1.0e-9:
            break
        hu0 = hu1
        m += 5
    ga = gamma2(a)
    hu1 /= ga
    hu2 = 0.0
    m = 2
    while m <= 10:
        hu2 = 0.0
        g = 0.5 / m
        d = g
        for j in range(1, m + 1):
            s = 0.0
            for k in range(1, 31):
                t1 = d + g * _TQ[k - 1]
                t2 = d - g * _TQ[k - 1]
                t3 = c / (1.0 - t1)
                t4 = c / (1.0 - t2)
                f1 = t3 * t3 / c * exp(-x * t3) * pow(t3, a1) * pow(1.0 + t3, b1)
                f2 = t4 * t4 / c * exp(-x * t4) * pow(t4, a1) * pow(1.0 + t4, b1)
                s += _WQ[k - 1] * (f1 + f2)
            hu2 += s * g
            d += 2.0 * g
        if fabs(1.0 - hu0 / hu2) < 1.0e-9:
            break
        hu0 = hu2
        m += 2
    ga = gamma2(a)
    hu2 /= ga
    return (hu1 + hu2, ident)

def chgu(x: float, a: float, b: float):
    """Returns (hu, isfer)."""
    aa = a - b + 1.0
    isfer = 0
    il1 = _isint(a) and a <= 0.0
    il2 = _isint(aa) and aa <= 0.0
    il3 = fabs(a * (a - b + 1.0)) / x <= 2.0
    bl1 = (x <= 5.0) or (x <= 10.0 and a <= 2.0)
    bl2 = (x > 5.0) and (x <= 12.5) and ((a >= 1.0) and (b >= a + 4.0))
    bl3 = (x > 12.5) and (a >= 5.0) and (b >= a + 5.0)
    bn = _isint(b) and b != 0.0
    id1 = 0
    ident = -100
    hu1 = 0.0
    hu = 0.0
    if not _isint(b):
        r = _chgus(x, a, b)
        hu = r[0]
        id1 = r[1]
        if id1 >= 9:
            return (hu, 0)
        hu1 = hu
    if il1 or il2 or il3:
        r = _chgul(x, a, b)
        hu = r[0]
        ident = r[1]
        if ident >= 9:
            return (hu, 0)
        if id1 > ident:
            ident = id1
            hu = hu1
    if a >= 1.0:
        if bn and (bl1 or bl2 or bl3):
            r = _chgubi(x, a, b)
        else:
            r = _chguit(x, a, b)
        hu = r[0]
        ident = r[1]
    else:
        if b <= a:
            b00 = b
            a -= b - 1.0
            b = 2.0 - b
            r = _chguit(x, a, b)
            hu = r[0] * pow(x, 1.0 - b00)
            ident = r[1]
        elif bn:
            # C: `bn && (~il1)`; ~ of an int flag is always nonzero
            r = _chgubi(x, a, b)
            hu = r[0]
            ident = r[1]
    if ident < 6:
        isfer = 6
    return (hu, isfer)

def hypu(a: float, b: float, x: float) -> float:
    r = chgu(x, a, b)
    out = r[0]
    if out == 1e300:
        out = INF
    if r[1] != 0:
        out = NAN
    return out

# ------------------------------------------------------------------ Bessel zeros (jyzo / jyndd / jynbh)
def msta1(x: float, mp: int) -> int:
    a0 = fabs(x)
    n0 = _ci(1.1 * a0) + 1
    f0 = 0.5 * log10(6.28 * n0) - n0 * log10(1.36 * a0 / n0) - mp
    n1 = n0 + 5
    f1 = 0.5 * log10(6.28 * n1) - n1 * log10(1.36 * a0 / n1) - mp
    nn = 0
    for it in range(1, 21):
        nn = _ci(n1 - (n1 - n0) / (1.0 - f0 / f1))
        f = 0.5 * log10(6.28 * nn) - nn * log10(1.36 * a0 / nn) - mp
        if abs(nn - n1) < 1:
            break
        n0 = n1
        f0 = f1
        n1 = nn
        f1 = f
    return nn

def msta2(x: float, n: int, mp: int) -> int:
    a0 = fabs(x)
    hmp = 0.5 * mp
    ejn = 0.5 * log10(6.28 * n) - n * log10(1.36 * a0 / n)
    obj = 0.0
    n0 = 0
    if ejn <= hmp:
        obj = float(mp)
        n0 = _ci(1.1 * a0) + 1
    else:
        obj = hmp + ejn
        n0 = n
    f0 = 0.5 * log10(6.28 * n0) - n0 * log10(1.36 * a0 / n0) - obj
    n1 = n0 + 5
    f1 = 0.5 * log10(6.28 * n1) - n1 * log10(1.36 * a0 / n1) - obj
    nn = 0
    for it in range(1, 21):
        nn = _ci(n1 - (n1 - n0) / (1.0 - f0 / f1))
        f = 0.5 * log10(6.28 * nn) - nn * log10(1.36 * a0 / nn) - obj
        if abs(nn - n1) < 1:
            break
        n0 = n1
        f0 = f1
        n1 = nn
        f1 = f
    return nn + 10

_JA = (-0.0703125, 0.112152099609375, -0.5725014209747314, 0.6074042001273483e+01)
_JB = (0.0732421875, -0.2271080017089844, 0.1727727502584457e+01, -0.2438052969955606e+02)
_JA1 = (0.1171875, -0.144195556640625, 0.6765925884246826, -0.6883914268109947e+01)
_JB1 = (-0.1025390625, 0.2775764465332031, -0.1993531733751297e+01, 0.2724882731126854e+02)

def jynbh(n: int, nmin: int, x: float, bj: List[float], by: List[float]) -> int:
    r2p = 0.63661977236758
    nm = n
    if x < 1.0e-100:
        for k in range(nmin, n + 1):
            bj[k - nmin] = 0.0
            by[k - nmin] = -1.0e+300
        if nmin == 0:
            bj[0] = 1.0
        return nm
    by0 = 0.0
    by1 = 0.0
    if x <= 300.0 or n > _ci(0.9 * x):
        if n == 0:
            nm = 1
        m = msta1(x, 200)
        if m < nm:
            nm = m
        else:
            m = msta2(x, nm, 15)
        bs = 0.0
        su = 0.0
        sv = 0.0
        f2 = 0.0
        f1 = 1.0e-100
        f = 0.0
        k = m
        while k >= 0:
            f = 2.0 * (k + 1.0) / x * f1 - f2
            if k <= nm and k >= nmin:
                bj[k - nmin] = f
            if k == 2 * (k // 2) and k != 0:
                bs += 2.0 * f
                su += (1.0 if (k // 2) % 2 == 0 else -1.0) * f / k
            elif k > 1:
                sv += (1.0 if (k // 2) % 2 == 0 else -1.0) * k / (k * k - 1.0) * f
            f2 = f1
            f1 = f
            k -= 1
        s0 = bs + f
        for k2 in range(nmin, nm + 1):
            bj[k2 - nmin] /= s0
        bj0 = f1 / s0
        bj1 = f2 / s0
        ec = log(x / 2.0) + 0.5772156649015329
        by0 = r2p * (ec * bj0 - 4.0 * su / s0)
        by1 = r2p * ((ec - 1.0) * bj1 - bj0 / x - 4.0 * sv / s0)
        if 0 >= nmin:
            by[0 - nmin] = by0
        if 1 >= nmin:
            by[1 - nmin] = by1
    else:
        t1 = x - 0.25 * PI
        p0 = 1.0
        q0 = -0.125 / x
        for k in range(1, 5):
            p0 += _JA[k - 1] * pow(x, -2.0 * k)
            q0 += _JB[k - 1] * pow(x, -2.0 * k - 1)
        cu = sqrt(r2p / x)
        bj0 = cu * (p0 * cos(t1) - q0 * sin(t1))
        by0 = cu * (p0 * sin(t1) + q0 * cos(t1))
        if 0 >= nmin:
            bj[0 - nmin] = bj0
            by[0 - nmin] = by0
        t2 = x - 0.75 * PI
        p1 = 1.0
        q1 = 0.375 / x
        for k in range(1, 5):
            p1 += _JA1[k - 1] * pow(x, -2.0 * k)
            q1 += _JB1[k - 1] * pow(x, -2.0 * k - 1)
        bj1 = cu * (p1 * cos(t2) - q1 * sin(t2))
        by1 = cu * (p1 * sin(t2) + q1 * cos(t2))
        if 1 >= nmin:
            bj[1 - nmin] = bj1
            by[1 - nmin] = by1
        for k in range(2, nm + 1):
            bjk = 2.0 * (k - 1.0) / x * bj1 - bj0
            if k >= nmin:
                bj[k - nmin] = bjk
            bj0 = bj1
            bj1 = bjk
    for k in range(2, nm + 1):
        byk = 2.0 * (k - 1.0) * by1 / x - by0
        if k >= nmin:
            by[k - nmin] = byk
        by0 = by1
        by1 = byk
    return nm

def jyndd(n: int, x: float):
    """Returns (bjn, djn, fjn, byn, dyn, fyn)."""
    bj = [0.0, 0.0]
    by = [0.0, 0.0]
    jynbh(n + 1, n, x, bj, by)
    bjn = bj[0]
    byn = by[0]
    djn = -bj[1] + n * bj[0] / x
    dyn = -by[1] + n * by[0] / x
    fjn = (n * n / (x * x) - 1.0) * bjn - djn / x
    fyn = (n * n / (x * x) - 1.0) * byn - dyn / x
    return (bjn, djn, fjn, byn, dyn, fyn)

def jyzo(n: int, nt: int):
    """Returns (rj0, rj1, ry0, ry1): first nt zeros of Jn, Jn', Yn, Yn'."""
    from C import atan2(float, float) -> float
    from C import atan(float) -> float
    from C import tan(float) -> float
    rj0 = [0.0] * (nt + 1)
    rj1 = [0.0] * nt
    ry0 = [0.0] * (nt + 1)
    ry1 = [0.0] * nt
    n3 = pow(float(n), 0.33333) if n > 0 else 1.0
    x = 2.4 if n == 0 else n + 1.85576 * n3 + 1.03315 / n3
    L = 0
    while True:
        while True:
            x0 = x
            r = jyndd(n, x)
            x -= r[0] / r[1]
            if not (fabs(x - x0) > 1e-11):
                break
        L += 1
        rj0[L - 1] = x
        if L == 1:
            x = 5.52 if n == 0 else n + 3.24460 * n3 + 3.15824 / n3
        else:
            x = rj0[L - 1] + (rj0[L - 1] - rj0[L - 2])
        if L <= n + 10:
            r = jyndd(n, x)
            h = atan2(fabs(r[1]), sqrt(fabs(r[2] * r[0])))
            b = -r[1] / (r[0] * atan(h))
            x -= (h - PI / 2) / b
        if not (L < nt):
            break
    x = 3.8317 if n == 0 else n + 0.80861 * n3 + 0.07249 / n3
    L = 0
    while True:
        while True:
            x0 = x
            r = jyndd(n, x)
            x -= r[1] / r[2]
            if not (fabs(x - x0) > 1e-11):
                break
        L += 1
        rj1[L - 1] = x
        if L < nt:
            x = rj1[L - 1] + (rj0[L] - rj0[L - 1])
        else:
            break
    x = 0.89357697 if n == 0 else n + 0.93158 * n3 + 0.26035 / n3
    L = 0
    while True:
        while True:
            x0 = x
            r = jyndd(n, x)
            x -= r[3] / r[4]
            if not (fabs(x - x0) > 1.0e-11):
                break
        L += 1
        ry0[L - 1] = x
        if L == 1:
            x = 3.957678419314858 if n == 0 else n + 2.59626 * n3 + 2.022183 / n3
        else:
            x = ry0[L - 1] + (ry0[L - 1] - ry0[L - 2])
        if L <= n + 10:
            r = jyndd(n, x)
            h = atan2(fabs(r[4]), sqrt(fabs(r[5] * r[3])))
            b = -r[4] / (r[3] * tan(h))
            x -= (h - PI / 2) / b
        if not (L < nt):
            break
    x = 2.67257 if n == 0 else n + 1.8211 * n3 + 0.94001 / n3
    L = 0
    while True:
        while True:
            x0 = x
            r = jyndd(n, x)
            x -= r[4] / r[5]
            if not (fabs(x - x0) > 1.0e-11):
                break
        L += 1
        ry1[L - 1] = x
        if L < nt:
            x = ry1[L - 1] + (ry0[L] - ry0[L - 1])
        else:
            break
    return (rj0[:nt], rj1, ry0[:nt], ry1)
