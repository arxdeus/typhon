# Ported from scipy 1.18.1 (BSD-3-Clause)
# Algorithms from xsf (BSD-3) specfun/specfun.h: BJNDD and JDZO (zeros of
# Bessel functions Jn(x) and Jn'(x), Zhang & Jin).

from ._kernels import sqrt, fabs, pow

from C import log10(float) -> float

def bjndd(x: float, n: int, bj: List[float], dj: List[float], fj: List[float]):
    bs = 0.0
    f = 0.0
    f0 = 0.0
    f1 = 1e-35
    m = 1
    while m < 901:
        mt = int(0.5 * log10(6.28 * m) - m * log10(1.36 * fabs(x) / m))
        if mt > 20:
            break
        m += 1
    if m == 901:
        m -= 1
    k = m
    while k > -1:
        f = 2.0 * (k + 1.0) * f1 / x - f0
        if k <= n:
            bj[k] = f
        if k % 2 == 0:
            bs += 2.0 * f
        f0 = f1
        f1 = f
        k -= 1
    for k in range(n + 1):
        bj[k] /= (bs - f)
    dj[0] = -bj[1]
    fj[0] = -bj[0] - dj[0] / x
    for k in range(1, n + 1):
        dj[k] = bj[k - 1] - k * bj[k] / x
        fj[k] = (k * k / (x * x) - 1.0) * bj[k] - dj[k] / x

def jdzo(nt: int):
    """Returns (zo[1..nt], n[:nt], m[:nt], p[:nt])."""
    p1 = [0] * 1500
    mm_ = [0.0] * 1500   # M[L1] (1-based)
    nn_ = [0.0] * 1500   # N[L1]
    zc_ = [0.0] * 1500   # ZOC[L1]
    zo = [0.0] * 1500
    n = [0] * 1500
    m = [0] * 1500
    p = [0] * 1500
    bj = [0.0] * 120
    dj = [0.0] * 120
    fj = [0.0] * 120
    x = 0.0
    xm = 0.0
    nm = 0
    mmx = 0
    if nt < 600:
        xm = -1.0 + 2.248485 * sqrt(float(nt)) - 0.0159382 * nt + 3.208775e-4 * pow(float(nt), 1.5)
        nm = int(14.5 + 0.05875 * nt)
        mmx = int(0.02 * nt) + 6
    else:
        xm = 5.0 + 1.445389 * sqrt(float(nt)) + 0.01889876 * nt - 2.147763e-4 * pow(float(nt), 1.5)
        nm = int(27.8 + 0.0327 * nt)
        mmx = int(0.01088 * nt) + 10
    L0 = 0
    for i in range(1, nm + 1):
        x1 = 0.407658 + 0.4795504 * sqrt(i - 1.0) + 0.983618 * (i - 1)
        x2 = 1.99535 + 0.8333883 * sqrt(i - 1.0) + 0.984584 * (i - 1)
        L1 = 0
        for j in range(1, mmx + 1):
            skip1 = False
            if i != 1 or j != 1:
                x = x1
                while True:
                    bjndd(x, i, bj, dj, fj)
                    x0 = x
                    x -= dj[i - 1] / fj[i - 1]
                    if x1 > xm:
                        skip1 = True
                        break
                    if not (fabs(x - x0) > 1e-10):
                        break
            if not skip1:
                L1 += 1
                nn_[L1] = i - 1
                mm_[L1] = j
                if i == 1:
                    mm_[L1] = j - 1
                p1[L1] = 1
                zc_[L1] = x
                if i <= 15:
                    x1 = x + 3.057 + 0.0122 * (i - 1) + (1.555 + 0.41575 * (i - 1)) / pow(j + 1.0, 2.0)
                else:
                    x1 = x + 2.918 + 0.01924 * (i - 1) + (6.26 + 0.13205 * (i - 1)) / pow(j + 1.0, 2.0)
            x = x2
            skip2 = False
            while True:
                bjndd(x, i, bj, dj, fj)
                x0 = x
                x -= bj[i - 1] / dj[i - 1]
                if x > xm:
                    skip2 = True
                    break
                if not (fabs(x - x0) > 1e-10):
                    break
            if skip2:
                continue
            L1 += 1
            nn_[L1] = i - 1
            mm_[L1] = j
            p1[L1] = 0
            zc_[L1] = x
            if i <= 15:
                x2 = x + 3.11 + 0.0138 * (i - 1) + (0.04832 + 0.2804 * (i - 1)) / pow(j + 1.0, 2.0)
            else:
                x2 = x + 3.001 + 0.0105 * (i - 1) + (11.52 + 0.48525 * (i - 1)) / pow(j + 3.0, 2.0)
        L = L0 + L1
        L2 = L
        while True:
            if L0 == 0:
                for k in range(1, L + 1):
                    p[k - 1] = p1[k]
                    m[k - 1] = int(mm_[k])
                    n[k - 1] = int(nn_[k])
                    zo[k] = zc_[k]
                L1 = 0
            else:
                if zo[L0] >= zc_[L1]:
                    p[L0 + L1 - 1] = p[L0 - 1]
                    m[L0 + L1 - 1] = m[L0 - 1]
                    n[L0 + L1 - 1] = n[L0 - 1]
                    zo[L0 + L1] = zo[L0]
                    L0 -= 1
                else:
                    p[L0 + L1 - 1] = p1[L1]
                    m[L0 + L1 - 1] = int(mm_[L1])
                    n[L0 + L1 - 1] = int(nn_[L1])
                    zo[L0 + L1] = zc_[L1]
                    L1 -= 1
            if L1 == 0:
                break
        L0 = L2
    return (zo[1:nt + 1], n[:nt], m[:nt], p[:nt])
