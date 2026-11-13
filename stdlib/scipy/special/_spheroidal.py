# Ported from scipy 1.18.1 (BSD-3-Clause)
# Algorithms from xsf (BSD-3) specfun/specfun.h + sphd_wave.h (Zhang & Jin,
# Computation of Special Functions): SEGV, SDMN, SCKB, ASWFA, RMN1, RMN2L,
# RMN2SO, RMN2SP, KMN, QSTAR, CBK, GMN, LPMNS, LQMNS, SPHJ, SPHY, RSWFP, RSWFO.

from ._kernels import exp, log, sqrt, fabs, sin, cos, pow
from ._specfun import msta1, msta2

from C import log10(float) -> float
from C import atan(float) -> float

NAN = float('nan')
PI = 3.141592653589793

def _sg(k: int) -> float:
    # pow(-1, k)
    return -1.0 if (k % 2 != 0) else 1.0

def _zeros(n: int) -> List[float]:
    return [0.0] * n

# ---------------------------------------------------------------- sphj/sphy
def sphj(x: float, n: int, sj: List[float], dj: List[float]) -> int:
    nm = n
    if fabs(x) < 1e-100:
        for k in range(n + 1):
            sj[k] = 0.0
            dj[k] = 0.0
        sj[0] = 1.0
        if n > 0:
            dj[1] = 1.0 / 3.0
        return nm
    sj[0] = sin(x) / x
    dj[0] = (cos(x) - sin(x) / x) / x
    if n < 1:
        return nm
    sj[1] = (sj[0] - cos(x)) / x
    if n >= 2:
        sa = sj[0]
        sb = sj[1]
        m = msta1(x, 200)
        if m < n:
            nm = m
        else:
            m = msta2(x, n, 15)
        f = 0.0
        f0 = 0.0
        f1 = 1e-100
        k = m
        while k >= 0:
            f = (2.0 * k + 3.0) * f1 / x - f0
            if k <= nm:
                sj[k] = f
            f0 = f1
            f1 = f
            k -= 1
        cs = sa / f if fabs(sa) > fabs(sb) else sb / f0
        for k in range(nm + 1):
            sj[k] *= cs
    for k in range(1, nm + 1):
        dj[k] = sj[k - 1] - (k + 1.0) * sj[k] / x
    return nm

def sphy(x: float, n: int, sy: List[float], dy: List[float]) -> int:
    if x < 1.0e-60:
        for k in range(n + 1):
            sy[k] = -1.0e300
            dy[k] = 1.0e300
        return n
    sy[0] = -cos(x) / x
    f0 = sy[0]
    dy[0] = (sin(x) + cos(x) / x) / x
    if n < 1:
        return n
    sy[1] = (sy[0] - sin(x)) / x
    f1 = sy[1]
    for k in range(2, n + 1):
        f = ((2.0 * k - 1.0) * f1 / x) - f0
        sy[k] = f
        if fabs(f) >= 1.0e300:
            return k - 1
        f0 = f1
        f1 = f
    nm = n - 1
    for k in range(1, nm + 1):
        dy[k] = sy[k - 1] - (k + 1.0) * sy[k] / x
    return nm

# ---------------------------------------------------------------- lpmns/lqmns
def lpmns(m: int, n: int, x: float, pm: List[float], pd: List[float]):
    for k in range(n + 1):
        pm[k] = 0.0
        pd[k] = 0.0
    if fabs(x) == 1.0:
        for k in range(n + 1):
            if m == 0:
                pm[k] = 1.0
                pd[k] = 0.5 * k * (k + 1.0)
                if x < 0.0:
                    pm[k] *= 1.0 if (k % 2) == 0 else -1.0
                    pd[k] *= 1.0 if ((k + 1) % 2) == 0 else -1.0
            elif m == 1:
                pd[k] = 1e300
            elif m == 2:
                pd[k] = -0.25 * (k + 2.0) * (k + 1.0) * k * (k - 1.0)
                if x < 0.0:
                    pd[k] *= 1.0 if ((k + 1) % 2) == 0 else -1.0
        return
    x0 = fabs(1.0 - x * x)
    pm0 = 1.0
    pmk = pm0
    for k in range(1, m + 1):
        pmk = (2.0 * k - 1.0) * sqrt(x0) * pm0
        pm0 = pmk
    pm1 = (2.0 * m + 1.0) * x * pm0
    pm[m] = pmk
    pm[m + 1] = pm1
    for k in range(m + 2, n + 1):
        pm2 = ((2.0 * k - 1.0) * x * pm1 - (k + m - 1.0) * pmk) / (k - m)
        pm[k] = pm2
        pmk = pm1
        pm1 = pm2
    pd[0] = ((1.0 - m) * pm[1] - x * pm[0]) / (x * x - 1.0)
    for k in range(1, n + 1):
        pd[k] = (k * x * pm[k] - (k + m) * pm[k - 1]) / (x * x - 1.0)
    coef = 1.0 if (m % 2) == 0 else -1.0
    for k in range(1, n + 1):
        pm[k] *= coef
        pd[k] *= coef

def lqmns(m: int, n: int, x: float, qm: List[float], qd: List[float]):
    val = 0.0
    if fabs(x) == 1.0:
        val = 1e300
    for k in range(n + 1):
        qm[k] = val
        qd[k] = val
    if fabs(x) == 1.0:
        return
    ls = -1 if fabs(x) > 1.0 else 1
    xq = sqrt(ls * (1.0 - x * x))
    q0 = 0.5 * log(fabs((x + 1.0) / (x - 1.0)))
    q00 = q0
    q10 = -1.0 / xq
    q01 = x * q0 - 1.0
    q11 = -ls * xq * (q0 + x / (1.0 - x * x))
    qf0 = q00
    qf1 = q10
    qm0 = 0.0
    qm1 = 0.0
    for k in range(2, m + 1):
        qm0 = -2.0 * (k - 1.0) / xq * x * qf1 - ls * (k - 1.0) * (2.0 - k) * qf0
        qf0 = qf1
        qf1 = qm0
    if m == 0:
        qm0 = q00
    if m == 1:
        qm0 = q10
    qm[0] = qm0
    if fabs(x) < 1.0001:
        if m == 0 and n > 0:
            qf0 = q00
            qf1 = q01
            for k in range(2, n + 1):
                qf2 = ((2.0 * k - 1.0) * x * qf1 - (k - 1.0) * qf0) / k
                qm[k] = qf2
                qf0 = qf1
                qf1 = qf2
        qg0 = q01
        qg1 = q11
        for k in range(2, m + 1):
            qm1 = -2.0 * (k - 1.0) / xq * x * qg1 - ls * k * (3.0 - k) * qg0
            qg0 = qg1
            qg1 = qm1
        if m == 0:
            qm1 = q01
        if m == 1:
            qm1 = q11
        qm[1] = qm1
        if m == 1 and n > 1:
            qh0 = q10
            qh1 = q11
            for k in range(2, n + 1):
                qh2 = ((2.0 * k - 1.0) * x * qh1 - k * qh0) / (k - 1.0)
                qm[k] = qh2
                qh0 = qh1
                qh1 = qh2
        elif m >= 2:
            qg0 = q00
            qg1 = q01
            qh0 = q10
            qh1 = q11
            qmk = 0.0
            for l in range(2, n + 1):
                q0l = ((2.0 * l - 1.0) * x * qg1 - (l - 1.0) * qg0) / l
                q1l = ((2.0 * l - 1.0) * x * qh1 - l * qh0) / (l - 1.0)
                qf0 = q0l
                qf1 = q1l
                for k in range(2, m + 1):
                    qmk = -2.0 * (k - 1.0) / xq * x * qf1 - ls * (k + l - 1.0) * (l + 2.0 - k) * qf0
                    qf0 = qf1
                    qf1 = qmk
                qm[l] = qmk
                qg0 = qg1
                qg1 = q0l
                qh0 = qh1
                qh1 = q1l
    else:
        km = 0
        if fabs(x) > 1.1:
            km = 40 + m + n
        else:
            km = (40 + m + n) * int(-1.0 - 1.8 * log(x - 1.0))
        qf2 = 0.0
        qf1 = 1.0
        qf0 = 0.0
        k = km
        while k >= 0:
            qf0 = ((2.0 * k + 3.0) * x * qf1 - (k + 2.0 - m) * qf2) / (k + m + 1.0)
            if k <= n:
                qm[k] = qf0
            qf2 = qf1
            qf1 = qf0
            k -= 1
        for k in range(n + 1):
            qm[k] = qm[k] * qm0 / qf0
    if fabs(x) < 1.0:
        for k in range(n + 1):
            qm[k] = _sg(m) * qm[k]
    qd[0] = ((1.0 - m) * qm[1] - x * qm[0]) / (x * x - 1.0)
    for k in range(1, n + 1):
        qd[k] = (k * x * qm[k] - (k + m) * qm[k - 1]) / (x * x - 1.0)

# ---------------------------------------------------------------- sckb/sdmn/segv
def sckb(m: int, n: int, c0: float, df: List[float], ck: List[float]):
    c = c0
    if c <= 1.0e-10:
        c = 1.0e-10
    nm = 25 + int(0.5 * (n - m) + c)
    ip = (n - m) % 2
    reg = 1.0e-200 if (m + nm) > 80 else 1.0
    fac = -pow(0.5, float(m))
    sw = 0.0
    for k in range(nm):
        fac = -fac
        i1 = 2 * k + ip + 1
        r = reg
        for i in range(i1, i1 + 2 * m):
            r *= i
        i2 = k + m + ip
        for i in range(i2, i2 + k):
            r *= (i + 0.5)
        sm = r * df[k]
        for i in range(k + 1, nm + 1):
            d1 = 2.0 * i + ip
            d2 = 2.0 * m + d1
            d3 = i + m + ip - 0.5
            r = r * d2 * (d2 - 1.0) * i * (d3 + k) / (d1 * (d1 - 1.0) * (i - k) * d3)
            sm += r * df[i]
            if fabs(sw - sm) < fabs(sm) * 1.0e-14:
                break
            sw = sm
        r1 = reg
        for i in range(2, m + k + 1):
            r1 *= i
        ck[k] = fac * sm / r1

def sdmn(m: int, n: int, c: float, cv: float, kd: int, df: List[float]):
    nm = 25 + int(0.5 * (n - m) + c)
    if c < 1e-10:
        for i in range(1, nm + 1):
            df[i - 1] = 0.0
        df[(n - m) // 2] = 1.0
        return
    a = _zeros(nm + 3)
    d = _zeros(nm + 3)
    g = _zeros(nm + 3)
    cs = c * c * kd
    ip = (n - m) % 2
    for i in range(1, nm + 3):
        k = 2 * (i - 1) if ip == 0 else 2 * i - 1
        dk0 = float(m + k)
        dk1 = float(m + k + 1)
        dk2 = float(2 * (m + k))
        d2k = float(2 * m + k)
        a[i - 1] = (d2k + 2.0) * (d2k + 1.0) / ((dk2 + 3.0) * (dk2 + 5.0)) * cs
        d[i - 1] = dk0 * dk1 + (2.0 * dk0 * dk1 - 2.0 * m * m - 1.0) / ((dk2 - 1.0) * (dk2 + 3.0)) * cs
        g[i - 1] = k * (k - 1.0) / ((dk2 - 3.0) * (dk2 - 1.0)) * cs
    fs = 1.0
    f1 = 0.0
    f0 = 1.0e-100
    kb = 0
    df[nm] = 0.0
    fl = 0.0
    k = nm
    while k >= 1:
        f = -((d[k] - cv) * f0 + a[k] * f1) / g[k]
        if fabs(f) > fabs(df[k]):
            df[k - 1] = f
            f1 = f0
            f0 = f
            if fabs(f) > 1.0e+100:
                for k1 in range(k, nm + 1):
                    df[k1 - 1] *= 1.0e-100
                f1 *= 1.0e-100
                f0 *= 1.0e-100
        else:
            kb = k
            fl = df[k]
            f1 = 1.0e-100
            f2 = -((d[0] - cv) / a[0]) * f1
            df[0] = f1
            if kb == 1:
                fs = f2
            elif kb == 2:
                df[1] = f2
                fs = -((d[1] - cv) * f2 + g[1] * f1) / a[1]
            else:
                df[1] = f2
                for j in range(3, kb + 2):
                    f = -((d[j - 2] - cv) * f2 + g[j - 2] * f1) / a[j - 2]
                    if j <= kb:
                        df[j - 1] = f
                    if fabs(f) > 1.0e+100:
                        for k1 in range(1, j + 1):
                            df[k1 - 1] *= 1.0e-100
                        f *= 1.0e-100
                        f2 *= 1.0e-100
                    f1 = f2
                    f2 = f
                fs = f
            break
        k -= 1
    r1 = 1.0
    for j in range(m + ip + 1, 2 * (m + ip) + 1):
        r1 *= j
    su1 = df[0] * r1
    for k in range(2, kb + 1):
        r1 = -r1 * (k + m + ip - 1.5) / (k - 1.0)
        su1 += r1 * df[k - 1]
    su2 = 0.0
    sw = 0.0
    for k in range(kb + 1, nm + 1):
        if k != 1:
            r1 = -r1 * (k + m + ip - 1.5) / (k - 1.0)
        su2 += r1 * df[k - 1]
        if fabs(sw - su2) < fabs(su2) * 1.0e-14:
            break
        sw = su2
    r3 = 1.0
    for j in range(1, (m + n + ip) // 2 + 1):
        r3 *= (j + 0.5 * (n + m + ip))
    r4 = 1.0
    for j in range(1, (n - m - ip) // 2 + 1):
        r4 *= -4.0 * j
    s0 = r3 / (fl * (su1 / fs) + su2) / r4
    for k in range(1, kb + 1):
        df[k - 1] *= fl / fs * s0
    for k in range(kb + 1, nm + 1):
        df[k - 1] *= s0

def segv(m: int, n: int, c: float, kd: int, eg: List[float]) -> float:
    if c < 1e-10:
        for i in range(1, n - m + 2):
            eg[i - 1] = float((i + m) * (i + m - 1))
        return eg[n - m]
    a = _zeros(300)
    b = _zeros(100)
    cv0 = _zeros(100)
    d = _zeros(300)
    e = _zeros(300)
    f = _zeros(300)
    g = _zeros(300)
    h = _zeros(100)
    icm = (n - m + 2) // 2
    nm = 10 + int(0.5 * (n - m) + c)
    cs = c * c * kd
    for l in range(2):
        for i in range(1, nm + 1):
            k = 2 * (i - 1) if l == 0 else 2 * i - 1
            dk0 = float(m + k)
            dk1 = float(m + k + 1)
            dk2 = float(2 * (m + k))
            d2k = float(2 * m + k)
            a[i - 1] = (d2k + 2.0) * (d2k + 1.0) / ((dk2 + 3.0) * (dk2 + 5.0)) * cs
            d[i - 1] = dk0 * dk1 + (2.0 * dk0 * dk1 - 2.0 * m * m - 1.0) / ((dk2 - 1.0) * (dk2 + 3.0)) * cs
            g[i - 1] = k * (k - 1.0) / ((dk2 - 3.0) * (dk2 - 1.0)) * cs
        for k in range(2, nm + 1):
            e[k - 1] = sqrt(a[k - 2] * g[k - 1])
            f[k - 1] = e[k - 1] * e[k - 1]
        f[0] = 0.0
        e[0] = 0.0
        xa = d[nm - 1] + fabs(e[nm - 1])
        xb = d[nm - 1] - fabs(e[nm - 1])
        nm1 = nm - 1
        for i in range(1, nm1 + 1):
            t = fabs(e[i - 1]) + fabs(e[i])
            t1 = d[i - 1] + t
            if xa < t1:
                xa = t1
            t1 = d[i - 1] - t
            if t1 < xb:
                xb = t1
        for i in range(1, icm + 1):
            b[i - 1] = xa
            h[i - 1] = xb
        for k in range(1, icm + 1):
            for k1 in range(k, icm + 1):
                if b[k1 - 1] < b[k - 1]:
                    b[k - 1] = b[k1 - 1]
                    break
            if k != 1:
                if h[k - 1] < h[k - 2]:
                    h[k - 1] = h[k - 2]
            x1 = 0.0
            while True:
                x1 = (b[k - 1] + h[k - 1]) / 2.0
                cv0[k - 1] = x1
                if fabs((b[k - 1] - h[k - 1]) / x1) < 1e-14:
                    break
                j = 0
                s = 1.0
                for i in range(1, nm + 1):
                    if s == 0.0:
                        s += 1e-30
                    t = f[i - 1] / s
                    s = d[i - 1] - t - x1
                    if s < 0.0:
                        j += 1
                if j < k:
                    h[k - 1] = x1
                else:
                    b[k - 1] = x1
                    if j >= icm:
                        b[icm - 1] = x1
                    else:
                        if h[j] < x1:
                            h[j] = x1
                        if x1 < b[j - 1]:
                            b[j - 1] = x1
            cv0[k - 1] = x1
            index = 0
            if l == 0:
                index = 2 * k - 2
            if l == 1:
                index = 2 * k - 1
            if index >= n - m + 1:
                break
            eg[index] = cv0[k - 1]
    return eg[n - m]

# ---------------------------------------------------------------- aswfa
def aswfa(x0: float, m: int, n: int, c: float, kd: int, cv: float):
    ck = _zeros(260)
    df = _zeros(260)
    eps = 1e-14
    x = fabs(x0)
    ip = 0 if (n - m) % 2 == 0 else 1
    nm = 40 + int((n - m) // 2 + c)
    nm2 = nm // 2 - 2
    sdmn(m, n, c, cv, kd, df)
    sckb(m, n, c, df, ck)
    x1 = 1.0 - x * x
    a0 = 0.0
    if m == 0 and x1 == 0.0:
        a0 = 1.0
    else:
        a0 = pow(x1, 0.5 * m)
    su1 = ck[0]
    for k in range(1, nm2 + 1):
        r = ck[k] * pow(x1, float(k))
        su1 += r
        if k >= 10 and fabs(r / su1) < eps:
            break
    s1f = a0 * pow(x, float(ip)) * su1
    s1d = 0.0
    if x == 1.0:
        if m == 0:
            s1d = ip * ck[0] - 2.0 * ck[1]
        elif m == 1:
            s1d = -1e100
        elif m == 2:
            s1d = -2.0 * ck[0]
        else:
            s1d = 0.0
    else:
        d0 = ip - m / x1 * pow(x, ip + 1.0)
        d1 = -2.0 * a0 * pow(x, ip + 1.0)
        su2 = ck[1]
        for k in range(2, nm2 + 1):
            r = float(k) * ck[k] * pow(x1, k - 1.0)
            su2 += r
            if k >= 10 and fabs(r / su2) < eps:
                break
        s1d = d0 * a0 * su1 + d1 * su2
    if x0 < 0.0 and ip == 0:
        s1d = -s1d
    if x0 < 0.0 and ip == 1:
        s1f = -s1f
    return (s1f, s1d)

# ---------------------------------------------------------------- radial
def rmn1(m: int, n: int, c: float, x: float, kd: int, df: List[float]):
    eps = 1.0e-14
    ck = _zeros(260)
    nm1 = (n - m) // 2
    ip = 0 if (n - m == 2 * nm1) else 1
    nm = 25 + nm1 + int(c)
    reg = 1.0e-200 if (m + nm > 80) else 1.0
    r0 = reg
    for j in range(1, 2 * m + ip + 1):
        r0 *= j
    r = r0
    suc = r * df[0]
    sw = 0.0
    for k in range(2, nm + 1):
        r = r * (m + k - 1.0) * (m + k + ip - 1.5) / (k - 1.0) / (k + ip - 1.5)
        suc += r * df[k - 1]
        if k > nm1 and fabs(suc - sw) < fabs(suc) * eps:
            break
        sw = suc
    if x == 0.0:
        sckb(m, n, c, df, ck)
        sm = 0.0
        sw1 = 0.0
        for j in range(1, nm + 1):
            sm += ck[j - 1]
            if fabs(sm - sw1) < fabs(sm) * eps:
                break
            sw1 = sm
        r1 = 1.0
        for j in range(1, (n + m + ip) // 2 + 1):
            r1 = r1 * (j + 0.5 * (n + m + ip))
        r2 = 1.0
        for j in range(1, m + 1):
            r2 *= 2.0 * c * j
        r3 = 1.0
        for j in range(1, (n - m - ip) // 2 + 1):
            r3 *= j
        sa0 = (2.0 * (m + ip) + 1.0) * r1 / (pow(2.0, float(n)) * pow(c, float(ip)) * r2 * r3)
        if ip == 0:
            return (sm / (sa0 * suc) * df[0] * reg, 0.0)
        return (0.0, sm / (sa0 * suc) * df[0] * reg)
    cx = c * x
    nm2 = 2 * nm + m
    sj = _zeros(nm2 + 6)
    dj = _zeros(nm2 + 6)
    nm2 = sphj(cx, nm2, sj, dj)
    a0 = pow(1.0 - kd / (x * x), 0.5 * m) / suc
    r1f = 0.0
    sw = 0.0
    for k in range(1, nm + 1):
        l = 2 * k + m - n - 2 + ip
        lg = 1.0 if (l % 4 == 0) else -1.0
        if k == 1:
            r = r0
        else:
            r = r * (m + k - 1.0) * (m + k + ip - 1.5) / (k - 1.0) / (k + ip - 1.5)
        np = m + 2 * k - 2 + ip
        r1f += lg * r * df[k - 1] * sj[np]
        if k > nm1 and fabs(r1f - sw) < fabs(r1f) * eps:
            break
        sw = r1f
    r1f *= a0
    b0 = kd * m / pow(x, 3.0) / (1.0 - kd / (x * x)) * r1f
    sud = 0.0
    sw = 0.0
    for k in range(1, nm + 1):
        l = 2 * k + m - n - 2 + ip
        lg = 1.0 if (l % 4 == 0) else -1.0
        if k == 1:
            r = r0
        else:
            r = r * (m + k - 1.0) * (m + k + ip - 1.5) / (k - 1.0) / (k + ip - 1.5)
        np = m + 2 * k - 2 + ip
        sud = sud + lg * r * df[k - 1] * dj[np]
        if k > nm1 and fabs(sud - sw) < fabs(sud) * eps:
            break
        sw = sud
    return (r1f, b0 + a0 * c * sud)

def rmn2l(m: int, n: int, c: float, x: float, kd: int, df: List[float]):
    """Returns (r2f, r2d, id)."""
    eps = 1.0e-14
    nm1 = (n - m) // 2
    ip = 0 if (n - m == 2 * nm1) else 1
    nm = 25 + nm1 + int(c)
    reg = 1.0e-200 if (m + nm > 80) else 1.0
    nm2 = 2 * nm + m
    cx = c * x
    sy = _zeros(nm2 + 6)
    dy = _zeros(nm2 + 6)
    nm2 = sphy(cx, nm2, sy, dy)
    r0 = reg
    for j in range(1, 2 * m + ip + 1):
        r0 *= j
    r = r0
    suc = r * df[0]
    sw = 0.0
    for k in range(2, nm + 1):
        r = r * (m + k - 1.0) * (m + k + ip - 1.5) / (k - 1.0) / (k + ip - 1.5)
        suc += r * df[k - 1]
        if k > nm1 and fabs(suc - sw) < fabs(suc) * eps:
            break
        sw = suc
    a0 = pow(1.0 - kd / (x * x), 0.5 * m) / suc
    r2f = 0.0
    eps1 = 0.0
    np = 0
    for k in range(1, nm + 1):
        l = 2 * k + m - n - 2 + ip
        lg = 1.0 if (l % 4 == 0) else -1.0
        if k == 1:
            r = r0
        else:
            r = r * (m + k - 1.0) * (m + k + ip - 1.5) / (k - 1.0) / (k + ip - 1.5)
        np = m + 2 * k - 2 + ip
        r2f += lg * r * (df[k - 1] * sy[np])
        eps1 = fabs(r2f - sw)
        if k > nm1 and eps1 < fabs(r2f) * eps:
            break
        sw = r2f
    id1 = int(log10(eps1 / fabs(r2f) + eps))
    r2f *= a0
    if np >= nm2:
        return (r2f, 0.0, 10)
    b0 = kd * m / pow(x, 3.0) / (1.0 - kd / (x * x)) * r2f
    sud = 0.0
    eps2 = 0.0
    for k in range(1, nm + 1):
        l = 2 * k + m - n - 2 + ip
        lg = 1.0 if (l % 4 == 0) else -1.0
        if k == 1:
            r = r0
        else:
            r = r * (m + k - 1.0) * (m + k + ip - 1.5) / (k - 1.0) / (k + ip - 1.5)
        np = m + 2 * k - 2 + ip
        sud += lg * r * (df[k - 1] * dy[np])
        eps2 = fabs(sud - sw)
        if k > nm1 and eps2 < fabs(sud) * eps:
            break
        sw = sud
    r2d = b0 + a0 * c * sud
    id2 = int(log10(eps2 / fabs(sud) + eps))
    return (r2f, r2d, id1 if id1 > id2 else id2)

def kmn(m: int, n: int, c: float, cv: float, kd: int, df: List[float], dn: List[float]):
    """Fills dn, returns (ck1, ck2)."""
    eps = 1.0e-14
    nm = 25 + int(0.5 * (n - m) + c)
    nn = nm + m
    u = _zeros(nn + 5)
    v = _zeros(nn + 5)
    w = _zeros(nn + 5)
    tp = _zeros(nn + 5)
    rk = _zeros(nn + 5)
    cs = c * c * kd
    ck1 = 0.0
    ck2 = 0.0
    ip = 0 if (n - m) % 2 == 0 else 1
    for i in range(1, nn + 4):
        k = -2 * (i - 1) if ip == 0 else -(2 * i - 3)
        gk0 = 2.0 * m + k
        gk1 = (m + k) * (m + k + 1.0)
        gk2 = 2.0 * (m + k) - 1.0
        gk3 = 2.0 * (m + k) + 3.0
        u[i - 1] = gk0 * (gk0 - 1.0) * cs / (gk2 * (gk2 + 2.0))
        v[i - 1] = gk1 - cv + (2.0 * (gk1 - m * m) - 1.0) * cs / (gk2 * gk3)
        w[i - 1] = (k + 1.0) * (k + 2.0) * cs / ((gk2 + 2.0) * gk3)
    for k in range(1, m + 1):
        t = v[m]
        for l in range(0, m - k):
            t = v[m - l - 1] - w[m - l] * u[m - l - 1] / t
        rk[k - 1] = -u[k - 1] / t
    r = 1.0
    for k in range(1, m + 1):
        r = r * rk[k - 1]
        dn[k - 1] = df[0] * r
    tp[nn - 1] = v[nn]
    k = nn - 1
    while k >= m + 1:
        tp[k - 1] = v[k] - w[k + 1] * u[k] / tp[k]
        if k > m + 1:
            rk[k - 1] = -u[k - 1] / tp[k - 1]
        k -= 1
    dnp = df[0] if m == 0 else dn[m - 1]
    dn[m] = _sg(ip) * dnp * cs / ((2.0 * m - 1.0) * (2.0 * m + 1.0 - 4.0 * ip) * tp[m])
    for k in range(m + 2, nn + 1):
        dn[k - 1] = rk[k - 1] * dn[k - 2]
    r1 = 1.0
    for j in range(1, (n + m + ip) // 2 + 1):
        r1 = r1 * (j + 0.5 * (n + m + ip))
    r = 1.0
    for j in range(1, 2 * m + ip + 1):
        r *= j
    su0 = r * df[0]
    sw = 0.0
    for k in range(2, nm + 1):
        r = r * (m + k - 1.0) * (m + k + ip - 1.5) / (k - 1.0) / (k + ip - 1.5)
        su0 = su0 + r * df[k - 1]
        if k > (n - m) // 2 and fabs((su0 - sw) / su0) < eps:
            break
        sw = su0
    if kd != 1:
        r2 = 1.0
        for j in range(1, m + 1):
            r2 = 2.0 * c * r2 * j
        r3 = 1.0
        for j in range(1, (n - m - ip) // 2 + 1):
            r3 = r3 * j
        sa0 = (2.0 * (m + ip) + 1.0) * r1 / (pow(2.0, float(n)) * pow(c, float(ip)) * r2 * r3 * df[0])
        ck1 = sa0 * su0
        if kd == -1:
            return (ck1, ck2)
    r4 = 1.0
    for j in range(1, (n - m - ip) // 2 + 1):
        r4 *= 4.0 * j
    r5 = 1.0
    for j in range(1, m + 1):
        r5 = r5 * (j + m) / c
    g0 = df[0] if m == 0 else dn[m - 1]
    sb0 = (ip + 1.0) * pow(c, ip + 1.0) / (2.0 * ip * (m - 2.0) + 1.0) / (2.0 * m - 1.0)
    ck2 = _sg(ip) * sb0 * r4 * r5 * g0 / r1 * su0
    return (ck1, ck2)

def qstar(m: int, n: int, c: float, ck1: float, ck: List[float]):
    ap = _zeros(210)
    ip = 0 if (n - m) == 2 * ((n - m) // 2) else 1
    r = 1.0 / (ck[0] * ck[0])
    ap[0] = r
    for i in range(1, m + 1):
        s = 0.0
        for l in range(1, i + 1):
            sk = 0.0
            for k in range(l + 1):
                sk += ck[k] * ck[l - k]
            s += sk * ap[i - l]
        ap[i] = -r * s
    qs0 = ap[m - 1]
    for l in range(1, m):
        r = 1.0
        for k in range(1, l + 1):
            r = r * (2.0 * k + ip) * (2.0 * k - 1.0 + ip) / pow(2.0 * k, 2.0)
        qs0 += ap[m - l] * r
    qs = _sg(ip) * ck1 * (ck1 * qs0) / c
    qt = -2.0 / ck1 * qs
    return (qs, qt)

def cbk(m: int, n: int, c: float, cv: float, qt: float, ck: List[float], bk: List[float]):
    eps = 1.0e-14
    ip = 0 if (n - m) % 2 == 0 else 1
    nm = 25 + int(0.5 * (n - m) + c)
    u = _zeros(210)
    v = _zeros(210)
    w = _zeros(210)
    u[0] = 0.0
    n2 = nm - 2
    for j in range(1, n2 + 1):
        u[j] = c * c
    for j in range(1, n2 + 1):
        v[j - 1] = (2.0 * j - 1.0 - ip) * (2.0 * (j - m) - ip) + m * (m - 1.0) - cv
    for j in range(1, nm):
        w[j - 1] = (2.0 * j - ip) * (2.0 * j + 1.0 - ip)
    if ip == 0:
        sw = 0.0
        for k in range(n2):
            s1 = 0.0
            i1 = k - m + 1
            for i in range(i1, nm + 1):
                if i < 0:
                    continue
                r1 = 1.0
                for j in range(1, k + 1):
                    r1 = r1 * (i + m - j) / (1.0 * j)
                s1 += ck[i] * (2.0 * i + m) * r1
                if fabs(s1 - sw) < fabs(s1) * eps:
                    break
                sw = s1
            bk[k] = qt * s1
    else:
        sw = 0.0
        for k in range(n2):
            s1 = 0.0
            i1 = k - m + 1
            for i in range(i1, nm + 1):
                if i < 0:
                    continue
                r1 = 1.0
                for j in range(1, k + 1):
                    r1 = r1 * (i + m - j) / (1.0 * j)
                if i > 0:
                    s1 += ck[i - 1] * (2.0 * i + m - 1) * r1
                s1 -= ck[i] * (2.0 * i + m) * r1
                if fabs(s1 - sw) < fabs(s1) * eps:
                    break
                sw = s1
            bk[k] = qt * s1
    w[0] /= v[0]
    bk[0] /= v[0]
    for k in range(2, n2 + 1):
        t = v[k - 1] - w[k - 2] * u[k - 1]
        w[k - 1] /= t
        bk[k - 1] = (bk[k - 1] - bk[k - 2] * u[k - 1]) / t
    k = n2 - 1
    while k >= 1:
        bk[k - 1] -= w[k - 1] * bk[k]
        k -= 1

def gmn(m: int, n: int, c: float, x: float, bk: List[float]):
    eps = 1.0e-14
    ip = 0 if (n - m) % 2 == 0 else 1
    nm = 25 + int(0.5 * (n - m) + c)
    xm = pow(1.0 + x * x, -0.5 * m)
    gf0 = 0.0
    gw = 0.0
    for k in range(1, nm + 1):
        gf0 += bk[k - 1] * pow(x, 2.0 * k - 2.0)
        if fabs((gf0 - gw) / gf0) < eps and k >= 10:
            break
        gw = gf0
    gf = xm * gf0 * pow(x, 1.0 - ip)
    gd1 = -m * x / (1.0 + x * x) * gf
    gd0 = 0.0
    for k in range(1, nm):
        if ip == 0:
            gd0 += (2.0 * k - 1.0) * bk[k - 1] * pow(x, 2.0 * k - 2.0)
        else:
            gd0 += 2.0 * k * bk[k - 1] * pow(x, 2.0 * k - 1.0)
        if fabs((gd0 - gw) / gd0) < eps and k >= 10:
            break
        gw = gd0
    return (gf, gd1 + xm * gd0)

def rmn2so(m: int, n: int, c: float, x: float, cv: float, kd: int, df: List[float]):
    eps = 1.0e-14
    if fabs(df[0]) <= 1.0e-280:
        return (1.0e300, 1.0e300)
    bk = _zeros(210)
    ck = _zeros(260)
    dn = _zeros(260)
    nm = 25 + int((n - m) // 2 + c)
    ip = (n - m) % 2
    sckb(m, n, c, df, ck)
    k12 = kmn(m, n, c, cv, kd, df, dn)
    ck1 = k12[0]
    qq = qstar(m, n, c, ck1, ck)
    qs = qq[0]
    cbk(m, n, c, cv, qq[1], ck, bk)
    r2f = 0.0
    r2d = 0.0
    if x == 0.0:
        sm = 0.0
        sw = 0.0
        for j in range(nm):
            sm += ck[j]
            if fabs(sm - sw) < fabs(sm) * eps:
                break
            sw = sm
        if ip == 0:
            r1f = sm / ck1
            r2f = -0.5 * PI * qs * r1f
            r2d = qs * r1f + bk[0]
        else:
            r1d = sm / ck1
            r2f = bk[0]
            r2d = -0.5 * PI * qs * r1d
    else:
        g = gmn(m, n, c, x, bk)
        r1 = rmn1(m, n, c, x, kd, df)
        h0 = atan(x) - 0.5 * PI
        r2f = qs * r1[0] * h0 + g[0]
        r2d = qs * (r1[1] * h0 + r1[0] / (1.0 + x * x)) + g[1]
    return (r2f, r2d)

def rmn2sp(m: int, n: int, c: float, x: float, cv: float, kd: int, df: List[float]):
    eps = 1.0e-14
    pm = _zeros(560)
    pd = _zeros(560)
    qm = _zeros(560)
    qd = _zeros(560)
    dn = _zeros(260)
    nm1 = (n - m) // 2
    nm = int(25.0 + nm1 + c)
    nm2 = 2 * nm + m
    ip = (n - m) % 2
    k12 = kmn(m, n, c, cv, kd, df, dn)
    ck2 = k12[1]
    lpmns(m, nm2, x, pm, pd)
    lqmns(m, nm2, x, qm, qd)
    su0 = 0.0
    sw = 0.0
    for k in range(1, nm + 1):
        j = 2 * k - 2 + m + ip
        su0 += df[k - 1] * qm[j]
        if k > nm1 and fabs(su0 - sw) < fabs(su0) * eps:
            break
        sw = su0
    sd0 = 0.0
    for k in range(1, nm + 1):
        j = 2 * k - 2 + m + ip
        sd0 += df[k - 1] * qd[j]
        if k > nm1 and fabs(sd0 - sw) < fabs(sd0) * eps:
            break
        sw = sd0
    su1 = 0.0
    sd1 = 0.0
    for k in range(1, m + 1):
        j = m - 2 * k + ip
        if j < 0:
            j = -j - 1
        su1 += dn[k - 1] * qm[j]
        sd1 += dn[k - 1] * qd[j]
    ga = pow((x - 1.0) / (x + 1.0), 0.5 * m)
    for k in range(1, m + 1):
        j = m - 2 * k + ip
        if j >= 0:
            continue
        j = -j - 1
        r1 = 1.0
        for j1 in range(j):
            r1 *= (m + j1)
        r2 = 1.0
        for j2 in range(1, m - j - 1):
            r2 *= j2
        r3 = 1.0
        sf = 1.0
        for l1 in range(1, j + 1):
            r3 = 0.5 * r3 * (-j + l1 - 1.0) * (j + l1) / ((m + l1) * l1) * (1.0 - x)
            sf += r3
        gb = 1.0
        if m - j >= 2:
            gb = (m - j - 1.0) * r2
        spl = r1 * ga * gb * sf
        su1 += _sg(j + m) * dn[k - 1] * spl
        spd1 = m / (x * x - 1.0) * spl
        gc = 0.5 * j * (j + 1.0) / (m + 1.0)
        sd = 1.0
        r4 = 1.0
        for l1 in range(1, j):
            r4 = 0.5 * r4 * (-j + l1) * (j + l1 + 1.0) / ((m + l1 + 1.0) * l1) * (1.0 - x)
            sd += r4
        spd2 = r1 * ga * gb * gc * sd
        sd1 += _sg(j + m) * dn[k - 1] * (spd1 + spd2)
    su2 = 0.0
    ki = (2 * m + 1 + ip) // 2
    if ki < 1:
        ki = 1
    nm3 = nm + ki
    for k in range(ki, nm3 + 1):
        j = 2 * k - 1 - m - ip
        su2 += dn[k - 1] * pm[j]
        if j > m and fabs(su2 - sw) < fabs(su2) * eps:
            break
        sw = su2
    sd2 = 0.0
    for k in range(ki, nm3):
        j = 2 * k - 1 - m - ip
        sd2 += dn[k - 1] * pd[j]
        if j > m and fabs(sd2 - sw) < fabs(sd2) * eps:
            break
        sw = sd2
    sm = su0 + su1 + su2
    sdm = sd0 + sd1 + sd2
    return (sm / ck2, sdm / ck2)

def rswfp(m: int, n: int, c: float, x: float, cv: float, kf: int):
    """Returns (r1f, r1d, r2f, r2d)."""
    df = _zeros(260)
    sdmn(m, n, c, cv, 1, df)
    r1f = 0.0
    r1d = 0.0
    r2f = 0.0
    r2d = 0.0
    if kf != 2:
        t = rmn1(m, n, c, x, 1, df)
        r1f = t[0]
        r1d = t[1]
    if kf > 1:
        t2 = rmn2l(m, n, c, x, 1, df)
        r2f = t2[0]
        r2d = t2[1]
        if t2[2] > -8:
            t3 = rmn2sp(m, n, c, x, cv, 1, df)
            r2f = t3[0]
            r2d = t3[1]
    return (r1f, r1d, r2f, r2d)

def rswfo(m: int, n: int, c: float, x: float, cv: float, kf: int):
    df = _zeros(260)
    sdmn(m, n, c, cv, -1, df)
    r1f = 0.0
    r1d = 0.0
    r2f = 0.0
    r2d = 0.0
    if kf != 2:
        t = rmn1(m, n, c, x, -1, df)
        r1f = t[0]
        r1d = t[1]
    if kf > 1:
        idv = 10
        if x > 1e-8:
            t2 = rmn2l(m, n, c, x, -1, df)
            r2f = t2[0]
            r2d = t2[1]
            idv = t2[2]
        if idv > -1:
            t3 = rmn2so(m, n, c, x, cv, -1, df)
            r2f = t3[0]
            r2d = t3[1]
    return (r1f, r1d, r2f, r2d)

# ---------------------------------------------------------------- wrappers
def _bad_modes(m: float, n: float, nocv: bool) -> bool:
    if m < 0 or n < m or m != float(int(m)) or n != float(int(n)):
        return True
    if nocv and (n - m) > 198:
        return True
    return False

def segv_k(m: float, n: float, c: float, kd: int) -> float:
    if _bad_modes(m, n, True):
        return NAN
    eg = _zeros(int(n - m) + 3)
    return segv(int(m), int(n), c, kd, eg)

def ang1(m: float, n: float, c: float, cv: float, x: float, kd: int, nocv: bool):
    if x >= 1 or x <= -1 or _bad_modes(m, n, nocv):
        return (NAN, NAN)
    cvv = cv
    if nocv:
        cvv = segv_k(m, n, c, kd)
    return aswfa(x, int(m), int(n), c, kd, cvv)

def rad(m: float, n: float, c: float, cv: float, x: float, kd: int, kf: int, nocv: bool):
    """Returns (f, d) of the first (kf=1) or second (kf=2) kind."""
    bad = False
    if kd == 1:
        bad = x <= 1.0
    else:
        bad = x < 0.0
    if bad or _bad_modes(m, n, nocv):
        return (NAN, NAN)
    cvv = cv
    if nocv:
        cvv = segv_k(m, n, c, kd)
    r = (0.0, 0.0, 0.0, 0.0)
    if kd == 1:
        r = rswfp(int(m), int(n), c, x, cvv, kf)
    else:
        r = rswfo(int(m), int(n), c, x, cvv, kf)
    if kf == 1:
        return (r[0], r[1])
    return (r[2], r[3])

def cv_seq(m: int, n: int, c: float, kd: int) -> List[float]:
    eg = _zeros(n - m + 3)
    segv(m, n, c, kd, eg)
    return eg[:n - m + 1]
