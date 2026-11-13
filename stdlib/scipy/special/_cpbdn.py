# Ported from scipy 1.18.1 (BSD-3-Clause)
# Algorithms from xsf (BSD-3) specfun/specfun.h: CPBDN, CPDSA, CPDLA, GAIH
# (parabolic cylinder functions Dn(z) of complex argument, Zhang & Jin).

import cmath
from ._kernels import sqrt, fabs, pow

PI = 3.141592653589793

def gaih(x: float) -> float:
    if x == float(int(x)) and x > 0.0:
        ga = 1.0
        m = int(x - 1.0)
        for k in range(2, m + 1):
            ga *= k
        return ga
    elif (x + 0.5) == float(int(x + 0.5)) and x > 0.0:
        m = int(x)
        ga = sqrt(PI)
        for k in range(1, m + 1):
            ga *= 0.5 * (2.0 * k - 1.0)
        return ga
    return float('nan')

def cpdla(n: int, z: complex) -> complex:
    cb0 = cmath.exp(complex(n) * cmath.log(z)) * cmath.exp(-0.25 * z * z) if z != 0j else complex(0.0 if n > 0 else 1.0)
    cr = complex(1.0)
    cdn = complex(1.0)
    for k in range(1, 17):
        cr = -0.5 * cr * (2.0 * k - n - 1.0) * (2.0 * k - n - 2.0) / (float(k) * z * z)
        cdn += cr
        if abs(cr) < abs(cdn) * 1e-12:
            break
    return cdn * cb0

def cpdsa(n: int, z: complex) -> complex:
    eps = 1.0e-15
    sq2 = sqrt(2.0)
    ca0 = cmath.exp(-0.25 * z * z)
    va0 = 0.5 * (1.0 - n)
    cdn = complex(0.0)
    if n == 0:
        cdn = ca0
    else:
        if abs(z) == 0.0:
            if va0 <= 0.0 and va0 == float(int(va0)):
                cdn = complex(0.0)
            else:
                ga0 = gaih(va0)
                cdn = complex(sqrt(PI) / (pow(2.0, -0.5 * n) * ga0))
        else:
            xn = float(-n)
            g1 = gaih(xn)
            cb0 = pow(2.0, -0.5 * n - 1.0) * ca0 / g1
            vt = -0.5 * n
            g0 = gaih(vt)
            cdn = complex(g0)
            cr = complex(1.0)
            for m in range(1, 251):
                vm = 0.5 * (m - n)
                gm = gaih(vm)
                cr = -cr * sq2 * z / float(m)
                cdw = gm * cr
                cdn += cdw
                if abs(cdw) < abs(cdn) * eps:
                    break
            cdn *= cb0
    return cdn

def cpbdn(n: int, z: complex):
    """Returns (cpb[:|n|+2], cpd[:|n|+2]) following the xsf array layout."""
    sz = abs(n) + 3
    cpb = [complex(0.0)] * sz
    cpd = [complex(0.0)] * sz
    x = z.real
    a0 = abs(z)
    ca0 = cmath.exp(-0.25 * z * z.conjugate())
    n0 = 0
    if n >= 0:
        cf0 = ca0
        cf1 = z * ca0
        cpb[0] = cf0
        cpb[1] = cf1
        for k in range(2, n + 1):
            cf = z * cf1 - (k - 1.0) * cf0
            cpb[k] = cf
            cf0 = cf1
            cf1 = cf
    else:
        n0 = -n
        if x <= 0.0 or a0 == 0.0:
            cf0 = ca0
            cpb[0] = cf0
            z1 = -z
            if a0 <= 7.0:
                cpb[1] = cpdsa(-1, z1)
            else:
                cpb[1] = cpdla(-1, z1)
            cf1 = sqrt(2.0 * PI) / ca0 - cpb[1]
            cpb[1] = cf1
            for k in range(2, n0):
                cf = (-z * cf1 + cf0) / (k - 1.0)
                cpb[k] = cf
                cf0 = cf1
                cf1 = cf
        elif a0 <= 3.0:
            cpb[n0] = cpdsa(-n0, z)
            n1 = n0 + 1
            cpb[n1] = cpdsa(-n1, z)
            nm1 = n0 - 1
            k = nm1
            while k >= 0:
                cf = z * cpb[n0] + (k + 1.0) * cpb[n1]
                cpb[k] = cf
                cpb[n1] = cpb[n0]
                cpb[n0] = cf
                k -= 1
        else:
            m = 100 + abs(n)
            cfa = complex(0.0)
            cfb = complex(1.0e-30)
            big = [complex(0.0)] * (m + 2)
            k = m
            while k >= 0:
                cf = z * cfb + (k + 1.0) * cfa
                if k <= n0:
                    cpb[k] = cf
                cfa = cfb
                cfb = cf
                k -= 1
            cs0 = ca0 / cfb
            for k in range(n0 + 1):
                cpb[k] = cs0 * cpb[k]
    cpd[0] = -0.5 * z * cpb[0]
    if n >= 0:
        for k in range(1, n + 1):
            cpd[k] = -0.5 * z * cpb[k] + float(k) * cpb[k - 1]
    else:
        for k in range(1, n0):
            cpd[k] = 0.5 * z * cpb[k] - cpb[k - 1]
    return (cpb, cpd)
