# Ported from scipy 1.18.1 (BSD-3-Clause)
# Algorithms from xsf (BSD-3) par_cyl.h (specfun PBDV/PBVV/PBWA, Zhang & Jin,
# plus the DLMF 12.14 asymptotic + Taylor ODE stepping for W(a, x)).

import cmath
from ._kernels import exp, log, sqrt, fabs, pow
from ._specfun import gamma2
from C import cos(float) -> float
from C import sin(float) -> float
from C import cosh(float) -> float
from C import sinh(float) -> float
from C import atan2(float, float) -> float
from C import ceil(float) -> float

NAN = float('nan')
PI = 3.141592653589793

def cgama(z: complex, kf: int) -> complex:
    a = (8.333333333333333e-02, -2.777777777777778e-03, 7.936507936507937e-04,
         -5.952380952380952e-04, 8.417508417508418e-04, -1.917526917526918e-03,
         6.410256410256410e-03, -2.955065359477124e-02, 1.796443723688307e-01,
         -1.392432216905900e+00)
    xx = z.real
    yy = z.imag
    z1 = z
    if yy == 0.0 and xx <= 0.0 and xx == float(int(xx)):
        return complex(1e300, 0.0)
    elif xx < 0.0:
        z1 = z
        z = -z
        xx = -xx
        yy = -yy
    else:
        z1 = complex(xx, 0.0)
    x0 = xx
    na = 0
    if xx <= 7.0:
        na = int(7 - xx)
        x0 = xx + na
    az0 = abs(complex(x0, yy))
    th = atan2(yy, x0)
    gr = (x0 - 0.5) * log(az0) - th * yy - x0 + 0.5 * log(2.0 * PI)
    gi = th * (x0 - 0.5) + yy * log(az0) - yy
    for k in range(1, 11):
        t = pow(az0, float(1 - 2 * k))
        gr += a[k - 1] * t * cos((2.0 * k - 1.0) * th)
        gi += -a[k - 1] * t * sin((2.0 * k - 1.0) * th)
    if xx <= 7.0:
        gr1 = 0.0
        gi1 = 0.0
        for j in range(na):
            gr1 += 0.5 * log((xx + j) * (xx + j) + yy * yy)
            gi1 += atan2(yy, xx + j)
        gr -= gr1
        gi -= gi1
    if z1.real < 0.0:
        az0 = abs(z)
        th1 = atan2(yy, xx)
        sr = -sin(PI * xx) * cosh(PI * yy)
        si = -cos(PI * xx) * sinh(PI * yy)
        az1 = abs(complex(sr, si))
        th2 = atan2(si, sr)
        gr = log(PI / (az0 * az1)) - gr
        gi = -th1 - th2 - gi
    if kf == 1:
        return exp(gr) * complex(cos(gi), sin(gi))
    return complex(gr, gi)

def dvsa(x: float, va: float) -> float:
    eps = 1.0e-15
    sq2 = sqrt(2.0)
    ep = exp(-0.25 * x * x)
    va0 = 0.5 * (1.0 - va)
    pd = 0.0
    if va == 0.0:
        pd = ep
    elif x == 0.0:
        if va0 <= 0.0 and va0 == float(int(va0)):
            pd = 0.0
        else:
            ga0 = gamma2(va0)
            pd = sqrt(PI) / (pow(2.0, -0.5 * va) * ga0)
    else:
        g1 = gamma2(-va)
        a0 = pow(2.0, -0.5 * va - 1.0) * ep / g1
        vt = -0.5 * va
        g0 = gamma2(vt)
        pd = g0
        r = 1.0
        for m in range(1, 251):
            vm = 0.5 * (m - va)
            gm = gamma2(vm)
            r = -r * sq2 * x / m
            r1 = gm * r
            pd += r1
            if fabs(r1) < fabs(pd) * eps:
                break
        pd *= a0
    return pd

def dvla(x: float, va: float) -> float:
    eps = 1.0e-12
    ep = exp(-.25 * x * x)
    a0 = pow(fabs(x), va) * ep
    r = 1.0
    pd = 1.0
    for k in range(1, 17):
        r = -0.5 * r * (2.0 * k - va - 1.0) * (2.0 * k - va - 2.0) / (k * x * x)
        pd += r
        if fabs(r / pd) < eps:
            break
    pd *= a0
    if x < 0.0:
        vl = vvla(-x, va)
        gl = gamma2(-va)
        pd = PI * vl / gl + cos(PI * va) * pd
    return pd

def vvla(x: float, va: float) -> float:
    eps = 1e-12
    qe = exp(0.25 * x * x)
    a0 = pow(fabs(x), -va - 1) * sqrt(2.0 / PI) * qe
    r = 1.0
    pv = 1.0
    for k in range(1, 19):
        r = 0.5 * r * (2.0 * k + va - 1.0) * (2.0 * k + va) / (k * x * x)
        pv += r
        if fabs(r / pv) < eps:
            break
    pv *= a0
    if x < 0.0:
        pdl = dvla(-x, va)
        gl = gamma2(-va)
        dsl = sin(PI * va) * sin(PI * va)
        pv = dsl * gl / PI * pdl - cos(PI * va) * pv
    return pv

def vvsa(x: float, va: float) -> float:
    eps = 1.0e-15
    ep = exp(-0.25 * x * x)
    va0 = 1.0 + 0.5 * va
    pv = 0.0
    if x == 0.0:
        if (va0 <= 0.0 and va0 == float(int(va0))) or va == 0.0:
            pv = 0.0
        else:
            vb0 = -0.5 * va
            sv0 = sin(va0 * PI)
            ga0 = gamma2(va0)
            pv = pow(2.0, vb0) * sv0 / ga0
    else:
        sq2 = sqrt(2.0)
        a0 = pow(2.0, -0.5 * va) * ep / (2.0 * PI)
        sv = sin(-(va + 0.5) * PI)
        v1 = -0.5 * va
        g1 = gamma2(v1)
        pv = (sv + 1.0) * g1
        r = 1.0
        fac = 1.0
        for m in range(1, 251):
            vm = 0.5 * (m - va)
            gm = gamma2(vm)
            r = r * sq2 * x / m
            fac = -fac
            gw = fac * sv + 1.0
            r1 = gw * r * gm
            pv += r1
            if fabs(r1 / pv) < eps and gw != 0.0:
                break
        pv *= a0
    return pv

def _pbdv_arr(x: float, v: float):
    xa = fabs(x)
    v += -1.0 if v < 0 else 1.0
    nv = int(v)
    v0 = v - nv
    na = abs(nv)
    ep = exp(-0.25 * x * x)
    ja = 1 if na >= 1 else 0
    sz = na + 4
    dv = [0.0] * sz
    dp = [0.0] * sz
    pd0 = 0.0
    pd1 = 0.0
    if v >= 0.0:
        if v0 == 0.0:
            pd0 = ep
            pd1 = x * ep
        else:
            for l in range(ja + 1):
                v1 = v0 + l
                if xa <= 5.8:
                    pd1 = dvsa(x, v1)
                else:
                    pd1 = dvla(x, v1)
                if l == 0:
                    pd0 = pd1
        dv[0] = pd0
        dv[1] = pd1
        for k in range(2, na + 1):
            pdf = x * pd1 - (k + v0 - 1.0) * pd0
            dv[k] = pdf
            pd0 = pd1
            pd1 = pdf
    else:
        if x <= 0.0:
            if xa <= 5.8:
                pd0 = dvsa(x, v0)
                pd1 = dvsa(x, v0 - 1.0)
            else:
                pd0 = dvla(x, v0)
                pd1 = dvla(x, v0 - 1.0)
            dv[0] = pd0
            dv[1] = pd1
            for k in range(2, na + 1):
                pd = (-x * pd1 + pd0) / (k - 1.0 - v0)
                dv[k] = pd
                pd0 = pd1
                pd1 = pd
        elif x <= 2.0:
            v2 = nv + v0
            if nv == 0:
                v2 -= 1.0
            nk = int(-v2)
            if nk + 2 > sz:
                dv = dv + [0.0] * (nk + 2 - sz)
            f1 = dvsa(x, v2)
            f0 = dvsa(x, v2 + 1.0)
            dv[nk] = f1
            dv[nk - 1] = f0
            k = nk - 2
            while k >= 0:
                f = x * f0 + (k - v0 + 1.0) * f1
                dv[k] = f
                f1 = f0
                f0 = f
                k -= 1
        else:
            if xa <= 5.8:
                pd0 = dvsa(x, v0)
            else:
                pd0 = dvla(x, v0)
            dv[0] = pd0
            m = 100 + na
            f1 = 0.0
            f0 = 1e-30
            f = 0.0
            k = m
            while k >= 0:
                f = x * f0 + (k - v0 + 1.0) * f1
                if k <= na:
                    dv[k] = f
                f1 = f0
                f0 = f
                k -= 1
            s0 = pd0 / f
            for k2 in range(na + 1):
                dv[k2] *= s0
    for k in range(na):
        v1 = fabs(v0) + k
        if v >= 0.0:
            dp[k] = 0.5 * x * dv[k] - dv[k + 1]
        else:
            dp[k] = -0.5 * x * dv[k] - v1 * dv[k + 1]
    return (dv, dp, na)

def _pbdv(x: float, v: float) -> Tuple[float, float]:
    t = _pbdv_arr(x, v)
    return (t[0][t[2] - 1], t[1][t[2] - 1])

def _pbvv_arr(x: float, v: float):
    xa = fabs(x)
    v += -1.0 if v < 0 else 1.0
    nv = int(v)
    v0 = v - nv
    na = abs(nv)
    qe = exp(0.25 * x * x)
    q2p = sqrt(2.0 / PI)
    ja = 1 if na >= 1 else 0
    sz = na + 4
    vv = [0.0] * sz
    vp = [0.0] * sz
    f = 0.0
    f0 = 0.0
    f1 = 0.0
    if v <= 0.0:
        if v0 == 0.0:
            pv0 = vvsa(x, v0) if xa <= 7.5 else vvla(x, v0)
            f0 = q2p * qe
            f1 = x * f0
            vv[0] = pv0
            vv[1] = f0
            vv[2] = f1
        else:
            for l in range(ja + 1):
                v1 = v0 - l
                if xa <= 7.5:
                    f1 = vvsa(x, v1)
                else:
                    f1 = vvla(x, v1)
                if l == 0:
                    f0 = f1
            vv[0] = f0
            vv[1] = f1
        kv = 3 if v0 == 0.0 else 2
        for k in range(kv, na + 1):
            f = x * f1 + (k - v0 - 2.0) * f0
            vv[k] = f
            f0 = f1
            f1 = f
    else:
        if x >= 0.0 and x <= 7.5:
            v2 = v
            if v2 < 1.0:
                v2 = v2 + 1.0
            f1 = vvsa(x, v2)
            v1 = v2 - 1.0
            kv = int(v2)
            if kv + 2 > sz:
                vv = vv + [0.0] * (kv + 2 - sz)
            f0 = vvsa(x, v1)
            vv[kv] = f1
            vv[kv - 1] = f0
            k = kv - 2
            while k >= 0:
                f = x * f0 - (k + v0 + 2.0) * f1
                if k <= na:
                    vv[k] = f
                f1 = f0
                f0 = f
                k -= 1
        elif x > 7.5:
            pv0 = vvla(x, v0)
            m = 100 + abs(na)
            vv[1] = pv0
            f1 = 0.0
            f0 = 1.0e-40
            k = m
            while k >= 0:
                f = x * f0 - (k + v0 + 2.0) * f1
                if k <= na:
                    vv[k] = f
                f1 = f0
                f0 = f
                k -= 1
            s0 = pv0 / f
            for k2 in range(na + 1):
                vv[k2] *= s0
        else:
            if xa <= 7.5:
                f0 = vvsa(x, v0)
                f1 = vvsa(x, v0 + 1.0)
            else:
                f0 = vvla(x, v0)
                f1 = vvla(x, v0 + 1.0)
            vv[0] = f0
            vv[1] = f1
            for k in range(2, na + 1):
                f = (x * f1 - f0) / (k + v0)
                vv[k] = f
                f0 = f1
                f1 = f
    for k in range(na):
        v1 = v0 + k
        if v >= 0.0:
            vp[k] = 0.5 * x * vv[k] - (v1 + 1.0) * vv[k + 1]
        else:
            vp[k] = -0.5 * x * vv[k] + vv[k + 1]
    return (vv, vp, na)

def _pbvv(x: float, v: float) -> Tuple[float, float]:
    t = _pbvv_arr(x, v)
    return (t[0][t[2] - 1], t[1][t[2] - 1])

def pbdv(v: float, x: float) -> Tuple[float, float]:
    if v != v or x != x:
        return (NAN, NAN)
    return _pbdv(x, v)

def pbvv(v: float, x: float) -> Tuple[float, float]:
    if v != v or x != x:
        return (NAN, NAN)
    return _pbvv(x, v)

def _pbwa_series(a: float, x: float) -> Tuple[float, float, float, float]:
    eps = 1e-15
    p0 = 0.59460355750136
    g1 = 0.0
    g2 = 0.0
    if a == 0.0:
        g1 = 3.6256099082219083
        g2 = 1.2254167024651776
    else:
        g1 = abs(cgama(complex(0.25, 0.5 * a), 1))
        g2 = abs(cgama(complex(0.75, 0.5 * a), 1))
    f1 = sqrt(g1 / g2)
    f2 = sqrt(2.0 * g2 / g1)
    h = [0.0] * 100
    h0 = 1.0
    h1 = a
    h[0] = a
    for L1 in range(2, 101):
        hl = a * h1 - 0.25 * (2 * L1 - 2.0) * (2 * L1 - 3.0) * h0
        h[L1 - 1] = hl
        h0 = h1
        h1 = hl
    y1f = 1.0
    r = 1.0
    for k in range(1, 101):
        r = 0.5 * r * x * x / (k * (2.0 * k - 1.0))
        r1 = h[k - 1] * r
        y1f += r1
        if fabs(r1) <= eps * fabs(y1f) and k > 30:
            break
    y1d = a
    r = 1.0
    for k in range(1, 100):
        r = 0.5 * r * x * x / (k * (2.0 * k + 1.0))
        r1 = h[k] * r
        y1d += r1
        if fabs(r1) <= eps * fabs(y1d) and k > 30:
            break
    y1d *= x
    d = [0.0] * 80
    d1 = 1.0
    d2 = a
    d[0] = 1.0
    d[1] = a
    for L2 in range(3, 81):
        dl = a * d2 - 0.25 * ((2 * L2 - 1) - 2.0) * ((2 * L2 - 1) - 3.0) * d1
        d[L2 - 1] = dl
        d1 = d2
        d2 = dl
    y2f = 1.0
    r = 1.0
    for k in range(1, 80):
        r = 0.5 * r * x * x / (k * (2.0 * k + 1.0))
        r1 = d[k] * r
        y2f += r1
        if fabs(r1) <= eps * fabs(y2f) and k > 30:
            break
    y2f *= x
    y2d = 1.0
    r = 1.0
    for k in range(1, 80):
        r = 0.5 * r * x * x / (k * (2.0 * k - 1.0))
        r1 = d[k] * r
        y2d += r1
        if fabs(r1) <= eps * fabs(y2d) and k > 30:
            break
    return (p0 * (f1 * y1f - f2 * y2f), p0 * (f1 * y1d - f2 * y2d),
            p0 * (f1 * y1f + f2 * y2f), p0 * (f1 * y1d + f2 * y2d))

def _pbwa_ode(a: float, x: float) -> Tuple[float, float]:
    x0 = 10.0
    z = complex(0.5, a)
    t = complex(1.0, 0.0)
    s = complex(1.0, 0.0)
    ds = complex(0.0, 0.0)
    r = 1
    while r <= 50 and abs(t) >= 2.220446049250313e-16:
        t *= complex(0.0, -1.0) * (z + float(2 * r - 2)) * (z + float(2 * r - 1)) / float(2 * r * x0 * x0)
        s += t
        ds -= t * (2 * r / x0)
        r += 1
    e = exp(PI * a)
    kk = 1.0 / (sqrt(1.0 + e * e) + e)
    om = 0.25 * x0 * x0 - a * log(x0) + 0.25 * PI + 0.5 * cgama(z, 0).imag
    rho = sqrt(2 * kk / x0)
    f = complex(rho * cos(om), rho * sin(om))
    u = s * f
    du = u * complex(-0.5 / x0, 0.5 * x0 - a / x0) + ds * f
    wf = u.real
    wd = du.real
    n = int(ceil(2.0 * (x0 - x)))
    hh = (x - x0) / n
    b = [0.0] * 34
    for i in range(n):
        xi = x0 + i * hh
        q = a - 0.25 * xi * xi
        for j in range(34):
            b[j] = 0.0
        b[2] = wf
        b[3] = hh * wd
        wf += b[3]
        wd = b[3]
        for m in range(30):
            c = q * b[m + 2] - 0.5 * hh * xi * b[m + 1] - 0.25 * hh * hh * b[m]
            b[m + 4] = hh * hh * c / ((m + 2.0) * (m + 1.0))
            wf += b[m + 4]
            wd += (m + 2) * b[m + 4]
        wd /= hh
    return (wf, wd)

def pbwa(a: float, x: float) -> Tuple[float, float]:
    if x < -5 or x > 5 or a < -5 or a > 5:
        return (NAN, NAN)
    elif fabs(x) > 0.5:
        return _pbwa_ode(a, x)
    else:
        if x < 0:
            w = _pbwa_series(a, -x)
            return (w[2], -w[3])
        w = _pbwa_series(a, x)
        return (w[0], w[1])

def pbdv_seq(v: float, x: float):
    n = int(v)
    v0 = v - n
    n1 = 1 if n < 1 else n
    v1 = n1 + v0
    t = _pbdv_arr(x, v1)
    return (t[0][:n1 + 1], t[1][:n1 + 1])

def pbvv_seq(v: float, x: float):
    n = int(v)
    v0 = v - n
    n1 = 1 if n <= 1 else n
    v1 = n1 + v0
    t = _pbvv_arr(x, v1)
    return (t[0][:n1 + 1], t[1][:n1 + 1])
