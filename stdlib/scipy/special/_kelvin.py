# Ported from scipy 1.18.1 (BSD-3-Clause)
# Algorithm from xsf (BSD-3) kelvin.h (specfun KLVNA, Zhang & Jin).

from ._kernels import exp, log, sqrt, fabs
from C import cos(float) -> float
from C import sin(float) -> float

NAN = float('nan')
INF = float('inf')

def _conv(v: float) -> float:
    if v == 1e300:
        return INF
    if v == -1e300:
        return -INF
    return v

def klvna(x: float) -> Tuple[float, float, float, float, float, float, float, float]:
    pi = 3.141592653589793
    el = 0.5772156649015329
    eps = 1.0e-15
    if x == 0.0:
        return (1.0, 0.0, 1.0e300, -0.25 * pi, 0.0, 0.0, -1.0e300, 0.0)
    x2 = 0.25 * x * x
    x4 = x2 * x2
    ber = 0.0
    bei = 0.0
    ger = 0.0
    gei = 0.0
    der = 0.0
    dei = 0.0
    her = 0.0
    hei = 0.0
    if fabs(x) < 10.0:
        ber = 1.0
        r = 1.0
        for m in range(1, 61):
            r = -0.25 * r / float(m * m) / ((2.0 * m - 1.0) * (2.0 * m - 1.0)) * x4
            ber += r
            if fabs(r) < fabs(ber) * eps:
                break
        bei = x2
        r = x2
        for m in range(1, 61):
            r = -0.25 * r / float(m * m) / ((2.0 * m + 1.0) * (2.0 * m + 1.0)) * x4
            bei += r
            if fabs(r) < fabs(bei) * eps:
                break
        ger = -(log(x / 2.0) + el) * ber + 0.25 * pi * bei
        r = 1.0
        gs = 0.0
        for m in range(1, 61):
            r = -0.25 * r / float(m * m) / ((2.0 * m - 1.0) * (2.0 * m - 1.0)) * x4
            gs = gs + 1.0 / (2.0 * m - 1.0) + 1.0 / (2.0 * m)
            ger += r * gs
            if fabs(r * gs) < fabs(ger) * eps:
                break
        gei = x2 - (log(x / 2.0) + el) * bei - 0.25 * pi * ber
        r = x2
        gs = 1.0
        for m in range(1, 61):
            r = -0.25 * r / float(m * m) / ((2.0 * m + 1.0) * (2.0 * m + 1.0)) * x4
            gs = gs + 1.0 / (2.0 * m) + 1.0 / (2.0 * m + 1.0)
            gei += r * gs
            if fabs(r * gs) < fabs(gei) * eps:
                break
        der = -0.25 * x * x2
        r = der
        for m in range(1, 61):
            r = -0.25 * r / m / (m + 1.0) / ((2.0 * m + 1.0) * (2.0 * m + 1.0)) * x4
            der += r
            if fabs(r) < fabs(der) * eps:
                break
        dei = 0.5 * x
        r = dei
        for m in range(1, 61):
            r = -0.25 * r / float(m * m) / (2.0 * m - 1.0) / (2.0 * m + 1.0) * x4
            dei += r
            if fabs(r) < fabs(dei) * eps:
                break
        r = -0.25 * x * x2
        gs = 1.5
        her = 1.5 * r - ber / x - (log(x / 2.0) + el) * der + 0.25 * pi * dei
        for m in range(1, 61):
            r = -0.25 * r / m / (m + 1.0) / ((2.0 * m + 1.0) * (2.0 * m + 1.0)) * x4
            gs = gs + 1.0 / (2.0 * m + 1.0) + 1.0 / (2 * m + 2.0)
            her += r * gs
            if fabs(r * gs) < fabs(her) * eps:
                break
        r = 0.5 * x
        gs = 1.0
        hei = 0.5 * x - bei / x - (log(x / 2.0) + el) * dei - 0.25 * pi * der
        for m in range(1, 61):
            r = -0.25 * r / float(m * m) / (2 * m - 1.0) / (2 * m + 1.0) * x4
            gs = gs + 1.0 / (2.0 * m) + 1.0 / (2 * m + 1.0)
            hei += r * gs
            if fabs(r * gs) < fabs(hei) * eps:
                break
    else:
        pp0 = 1.0
        pn0 = 1.0
        qp0 = 0.0
        qn0 = 0.0
        r0 = 1.0
        km = 18
        if fabs(x) >= 40.0:
            km = 10
        fac = 1.0
        for k in range(1, km + 1):
            fac = -fac
            xt = 0.25 * k * pi - float(int(0.125 * k)) * 2.0 * pi
            cs = cos(xt)
            ss = sin(xt)
            r0 = 0.125 * r0 * ((2.0 * k - 1.0) * (2.0 * k - 1.0)) / k / x
            rc = r0 * cs
            rs = r0 * ss
            pp0 += rc
            pn0 += fac * rc
            qp0 += rs
            qn0 += fac * rs
        xd = x / sqrt(2.0)
        xe1 = exp(xd)
        xe2 = exp(-xd)
        xc1 = 1.0 / sqrt(2.0 * pi * x)
        xc2 = sqrt(0.5 * pi / x)
        cp0 = cos(xd + 0.125 * pi)
        cn0 = cos(xd - 0.125 * pi)
        sp0 = sin(xd + 0.125 * pi)
        sn0 = sin(xd - 0.125 * pi)
        ger = xc2 * xe2 * (pn0 * cp0 - qn0 * sp0)
        gei = xc2 * xe2 * (-pn0 * sp0 - qn0 * cp0)
        ber = xc1 * xe1 * (pp0 * cn0 + qp0 * sn0) - gei / pi
        bei = xc1 * xe1 * (pp0 * sn0 - qp0 * cn0) + ger / pi
        pp1 = 1.0
        pn1 = 1.0
        qp1 = 0.0
        qn1 = 0.0
        r1 = 1.0
        fac = 1.0
        for k in range(1, km + 1):
            fac = -fac
            xt = 0.25 * k * pi - float(int(0.125 * k)) * 2.0 * pi
            cs = cos(xt)
            ss = sin(xt)
            r1 = 0.125 * r1 * (4.0 - (2.0 * k - 1.0) * (2.0 * k - 1.0)) / (k * x)
            rc = r1 * cs
            rs = r1 * ss
            pp1 += fac * rc
            pn1 += rc
            qp1 += fac * rs
            qn1 += rs
        her = xc2 * xe2 * (-pn1 * cn0 + qn1 * sn0)
        hei = xc2 * xe2 * (pn1 * sn0 + qn1 * cn0)
        der = xc1 * xe1 * (pp1 * cp0 + qp1 * sp0) - hei / pi
        dei = xc1 * xe1 * (pp1 * sp0 - qp1 * cp0) + her / pi
    return (ber, bei, ger, gei, der, dei, her, hei)

def ber(x: float) -> float:
    return _conv(klvna(fabs(x))[0])

def bei(x: float) -> float:
    return _conv(klvna(fabs(x))[1])

def ker(x: float) -> float:
    if x < 0:
        return NAN
    return _conv(klvna(x)[2])

def kei(x: float) -> float:
    if x < 0:
        return NAN
    return _conv(klvna(x)[3])

def berp(x: float) -> float:
    r = _conv(klvna(fabs(x))[4])
    return -r if x < 0 else r

def beip(x: float) -> float:
    r = _conv(klvna(fabs(x))[5])
    return -r if x < 0 else r

def kerp(x: float) -> float:
    if x < 0:
        return NAN
    return _conv(klvna(x)[6])

def keip(x: float) -> float:
    if x < 0:
        return NAN
    return _conv(klvna(x)[7])

def kelvin(x: float) -> Tuple[complex, complex, complex, complex]:
    flag = x < 0
    t = klvna(fabs(x))
    be = complex(_conv(t[0]), _conv(t[1]))
    ke = complex(_conv(t[2]), _conv(t[3]))
    bep = complex(_conv(t[4]), _conv(t[5]))
    kep = complex(_conv(t[6]), _conv(t[7]))
    if flag:
        bep = -bep
        ke = complex(NAN, NAN)
        kep = complex(NAN, NAN)
    return (be, ke, bep, kep)

def klvnzo(nt: int, kd: int) -> List[float]:
    rt0 = [0.0, 2.84891, 5.02622, 1.71854, 3.91467, 6.03871, 3.77268, 2.66584, 4.93181]
    rt = rt0[kd]
    zo = List[float]()
    for m in range(1, nt + 1):
        while True:
            t = klvna(rt)
            if kd == 1:
                rt -= t[0] / t[4]
            elif kd == 2:
                rt -= t[1] / t[5]
            elif kd == 3:
                rt -= t[2] / t[6]
            elif kd == 4:
                rt -= t[3] / t[7]
            elif kd == 5:
                rt -= t[4] / (-t[1] - t[4] / rt)
            elif kd == 6:
                rt -= t[5] / (t[0] - t[5] / rt)
            elif kd == 7:
                rt -= t[6] / (-t[3] - t[6] / rt)
            else:
                rt -= t[7] / (t[2] - t[7] / rt)
            if fabs(rt - rt0[kd]) <= 5e-10:
                break
            else:
                rt0[kd] = rt
        zo.append(rt)
        rt += 4.44
    return zo
