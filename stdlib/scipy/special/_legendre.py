# Ported from scipy 1.18.1 (BSD-3-Clause)
# Algorithms from xsf (BSD-3) specfun.h (LPMV0/LPMV, Zhang & Jin) and
# cephes tukey.h (Tukey-lambda CDF).

from ._kernels import exp, log, sqrt, fabs, pow
from ._specfun import gamma2, psi_spec
from C import sin(float) -> float
from C import tan(float) -> float

NAN = float('nan')
INF = float('inf')
PI = 3.141592653589793

def _m1pow(n: int) -> float:
    return 1.0 if n % 2 == 0 else -1.0

def lpmv0(v: float, m: int, x: float) -> float:
    el = 0.5772156649015329
    eps = 1e-14
    nv = int(v)
    v0 = v - nv
    if x == -1.0 and v != float(nv):
        if m == 0:
            return -1.0e300
        return 1.0e300
    c0 = 1.0
    if m != 0:
        rg = v * (v + m)
        for j in range(1, m):
            rg *= (v * v - float(j * j))
        xq = sqrt(1.0 - x * x)
        r0 = 1.0
        for j in range(1, m + 1):
            r0 = 0.5 * r0 * xq / j
        c0 = r0 * rg
    if v0 == 0.0:
        pmv = 1.0
        r = 1.0
        for k in range(1, nv - m + 1):
            r = 0.5 * r * (-nv + m + k - 1.0) * (nv + m + k) / float(k * (k + m)) * (1.0 + x)
            pmv += r
        return _m1pow(nv) * c0 * pmv
    if x >= -0.35:
        pmv = 1.0
        r = 1.0
        for k in range(1, 101):
            r = 0.5 * r * (-v + m + k - 1.0) * (v + m + k) / float(k * (m + k)) * (1.0 - x)
            pmv += r
            if k > 12 and fabs(r / pmv) < eps:
                break
        return _m1pow(m) * c0 * pmv
    vs = sin(v * PI) / PI
    pv0 = 0.0
    if m != 0:
        qr = sqrt((1.0 - x) / (1.0 + x))
        r2 = 1.0
        for j in range(1, m + 1):
            r2 *= qr * j
        s0 = 1.0
        r1 = 1.0
        for k in range(1, m):
            r1 = 0.5 * r1 * (-v + k - 1) * (v + k) / float(k * (k - m)) * (1.0 + x)
            s0 += r1
        pv0 = -vs * r2 / m * s0
    pa = 2.0 * (psi_spec(v) + el) + PI / tan(PI * v) + 1.0 / v
    s1 = 0.0
    for j in range(1, m + 1):
        s1 += (j * j + v * v) / (j * (j * j - v * v))
    pmv = pa + s1 - 1.0 / (m - v) + log(0.5 * (1.0 + x))
    r = 1.0
    for k in range(1, 101):
        r = 0.5 * r * (-v + m + k - 1.0) * (v + m + k) / float(k * (k + m)) * (1.0 + x)
        s = 0.0
        for j in range(1, m + 1):
            s += ((k + j) * (k + j) + v * v) / ((k + j) * ((k + j) * (k + j) - v * v))
        s2 = 0.0
        for j in range(1, k + 1):
            s2 = s2 + 1.0 / (j * (j * j - v * v))
        pss = pa + s + 2.0 * v * v * s2 - 1.0 / (m + k - v) + log(0.5 * (1.0 + x))
        r2 = pss * r
        pmv += r2
        if fabs(r2 / pmv) < eps:
            break
    return pv0 + pmv * vs * c0

def _lpmv(x: float, m: int, v: float) -> float:
    if x == -1.0 and v != float(int(v)):
        if m == 0:
            vs = sin(v * PI) / (v * PI)
            return -1e300 if vs > 0 else 1e300
        return 1e300
    vx = v
    mx = m
    if v < 0:
        vx = -vx - 1.0
    neg_m = False
    if m < 0:
        if (vx + m + 1) > 0 or vx != float(int(vx)):
            neg_m = True
            mx = -m
        else:
            return NAN
    nv = int(vx)
    v0 = vx - nv
    pmv = 0.0
    if nv > 2 and nv > mx:
        p0 = lpmv0(v0 + mx, mx, x)
        p1 = lpmv0(v0 + mx + 1, mx, x)
        pmv = p1
        for j in range(mx + 2, nv + 1):
            pmv = ((2 * (v0 + j) - 1) * x * p1 - (v0 + j - 1 + mx) * p0) / (v0 + j - mx)
            p0 = p1
            p1 = pmv
    else:
        pmv = lpmv0(vx, mx, x)
    if neg_m and fabs(pmv) < 1.e300:
        g1 = gamma2(vx - mx + 1)
        g2 = gamma2(vx + mx + 1)
        pmv = pmv * g1 / g2 * _m1pow(mx)
    return pmv

def lpmv(m: float, v: float, x: float) -> float:
    if m != m or v != v or x != x:
        return NAN
    if m != float(int(m)):
        return NAN
    out = _lpmv(x, int(m), v)
    if out == 1e300:
        return INF
    if out == -1e300:
        return -INF
    return out

def tklmbda(x: float, lmbda: float) -> float:
    SMALLVAL = 1e-4
    EPS = 1.0e-18
    if x != x or lmbda != lmbda:
        return NAN
    xeval = 1.0 / lmbda
    if lmbda > 0.0:
        if x <= -xeval:
            return 0.0
        if x >= xeval:
            return 1.0
    if -SMALLVAL < lmbda and lmbda < SMALLVAL:
        if x >= 0:
            return 1.0 / (1.0 + exp(-x))
        return exp(x) / (1.0 + exp(x))
    pmid = 0.5
    plow = 0.0
    phigh = 1.0
    count = 0
    while count < 60 and fabs(pmid - plow) > EPS:
        xeval = (pow(pmid, lmbda) - pow(1.0 - pmid, lmbda)) / lmbda
        if xeval == x:
            return pmid
        if xeval > x:
            phigh = pmid
            pmid = (pmid + plow) / 2.0
        else:
            plow = pmid
            pmid = (pmid + phigh) / 2.0
        count += 1
    return pmid
