# Copyright (C) 2026 the Typhon authors
# Ported from scipy 1.18.1 (BSD-3-Clause)
# Algorithms from xsf (BSD-3) specfun/specfun.h (Zhang & Jin, Computation of
# Special Functions): CVA2/CV0/CVF/CVQL/CVQM/REFINE, FCOEF, MTU0, MTU12 and
# the xsf mathieu_legacy.h wrappers.

from ._kernels import exp, sqrt, fabs, sin, cos, pow
from ._specfun import jynbh

NAN = float('nan')

def cvqm(m: int, q: float) -> float:
    mm = float(m) * m
    hm1 = 0.5 * q / (mm - 1.0)
    hm3 = 0.25 * hm1 * hm1 * hm1 / (mm - 4.0)
    hm5 = hm1 * hm3 * q / ((mm - 1.0) * (mm - 9.0))
    return mm + q * (hm1 + (5.0 * mm + 7.0) * hm3 + (9.0 * mm * mm + 58.0 * mm + 29.0) * hm5)

def cvql(kd: int, m: int, q: float) -> float:
    w = 0.0
    if kd == 1 or kd == 2:
        w = 2.0 * m + 1.0
    if kd == 3 or kd == 4:
        w = 2.0 * m - 1.0
    w2 = w * w
    w3 = w * w2
    w4 = w2 * w2
    w6 = w2 * w4
    d1 = 5.0 + 34.0 / w2 + 9.0 / w4
    d2 = (33.0 + 410.0 / w2 + 405.0 / w4) / w
    d3 = (63.0 + 1260.0 / w2 + 2943.0 / w4 + 486.0 / w6) / w2
    d4 = (527.0 + 15617.0 / w2 + 69001.0 / w4 + 41607.0 / w6) / w3
    c1 = 128.0
    p2 = q / w4
    p1 = sqrt(p2)
    cv1 = -2.0 * q + 2.0 * w * sqrt(q) - (w2 + 1.0) / 8.0
    cv2 = (w + 3.0 / w) + d1 / (32.0 * p1) + d2 / (8.0 * c1 * p2)
    cv2 = cv2 + d3 / (64.0 * c1 * p1 * p2) + d4 / (16.0 * c1 * c1 * p2 * p2)
    return cv1 - cv2 / (c1 * p1)

def cvf(kd: int, m: int, q: float, a: float, mj: int) -> float:
    ic = m // 2
    l = 0
    l0 = 0
    j0 = 2
    jf = ic
    t0 = 0.0
    t1 = 0.0
    t2 = 0.0
    b = a
    if kd == 1:
        j0 = 3
        l0 = 2
    if kd == 2 or kd == 3:
        l = 1
    if kd == 4:
        jf = ic - 1
    j = mj
    while j >= ic + 1:
        t1 = -q * q / (pow(2.0 * j + l, 2.0) - b + t1)
        j -= 1
    if m <= 2:
        if kd == 1 and m == 0:
            t1 += t1
        if kd == 1 and m == 2:
            t1 = -2.0 * q * q / (4.0 - b + t1) - 4.0
        if kd == 2 and m == 1:
            t1 += q
        if kd == 3 and m == 1:
            t1 -= q
    else:
        if kd == 1:
            t0 = 4.0 - b + 2.0 * q * q / b
        if kd == 2:
            t0 = 1.0 - b + q
        if kd == 3:
            t0 = 1.0 - b - q
        if kd == 4:
            t0 = 4.0 - b
        t2 = -q * q / t0
        j = j0
        while j <= jf:
            t2 = -q * q / (pow(2.0 * j - l - l0, 2.0) - b + t2)
            j += 1
    return pow(2.0 * ic + l, 2.0) + t1 + t2 - b

def refine(kd: int, m: int, q: float, a: float) -> float:
    eps = 1e-14
    mj = 10 + m
    x0 = a
    f0 = cvf(kd, m, q, x0, mj)
    x1 = 1.002 * a
    f1 = cvf(kd, m, q, x1, mj)
    x = 0.0
    for it in range(1, 101):
        mj += 1
        x = x1 - (x1 - x0) / (1.0 - f0 / f1)
        f = cvf(kd, m, q, x, mj)
        if fabs(1.0 - x1 / x) < eps or f == 0.0:
            break
        x0 = x1
        f0 = f1
        x1 = x
        f1 = f
    return x

def cv0(kd: int, m: int, q: float) -> float:
    a0 = 0.0
    q2 = q * q
    if m == 0:
        if q <= 1.0:
            a0 = (((0.0036392 * q2 - 0.0125868) * q2 + 0.0546875) * q2 - 0.5) * q2
        elif q <= 10.0:
            a0 = ((3.999267e-3 * q - 9.638957e-2) * q - 0.88297) * q + 0.5542818
        else:
            a0 = cvql(kd, m, q)
    elif m == 1:
        if q <= 1.0 and kd == 2:
            a0 = (((-6.51e-4 * q - 0.015625) * q - 0.125) * q + 1.0) * q + 1.0
        elif q <= 1.0 and kd == 3:
            a0 = (((-6.51e-4 * q + 0.015625) * q - 0.125) * q - 1.0) * q + 1.0
        elif q <= 10.0 and kd == 2:
            a0 = (((-4.94603e-4 * q + 1.92917e-2) * q - 0.3089229) * q + 1.33372) * q + 0.811752
        elif q <= 10.0 and kd == 3:
            a0 = ((1.971096e-3 * q - 5.482465e-2) * q - 1.152218) * q + 1.10427
        else:
            a0 = cvql(kd, m, q)
    elif m == 2:
        if q <= 1.0 and kd == 1:
            a0 = (((-0.0036391 * q2 + 0.0125888) * q2 - 0.0551939) * q2 + 0.416667) * q2 + 4.0
        elif q <= 1.0 and kd == 4:
            a0 = (0.0003617 * q2 - 0.0833333) * q2 + 4.0
        elif q <= 15.0 and kd == 1:
            a0 = (((3.200972e-4 * q - 8.667445e-3) * q - 1.829032e-4) * q + 0.9919999) * q + 3.3290504
        elif q <= 10.0 and kd == 4:
            a0 = ((2.38446e-3 * q - 0.08725329) * q - 4.732542e-3) * q + 4.00909
        else:
            a0 = cvql(kd, m, q)
    elif m == 3:
        if q <= 1.0 and kd == 2:
            a0 = (((6.348e-4 * q + 0.015625) * q + 0.0625) * q2 + 9.0)
        elif q <= 1.0 and kd == 3:
            a0 = (((6.348e-4 * q - 0.015625) * q + 0.0625) * q2 + 9.0)
        elif q <= 20.0 and kd == 2:
            a0 = (((3.035731e-4 * q - 1.453021e-2) * q + 0.19069602) * q - 0.1039356) * q + 8.9449274
        elif q <= 15.0 and kd == 3:
            a0 = ((9.369364e-5 * q - 0.03569325) * q + 0.2689874) * q + 8.771735
        else:
            a0 = cvql(kd, m, q)
    elif m == 4:
        if q <= 1.0 and kd == 1:
            a0 = ((-2.1e-6 * q2 + 5.012e-4) * q2 + 0.0333333) * q2 + 16.0
        elif q <= 1.0 and kd == 4:
            a0 = ((3.7e-6 * q2 - 3.669e-4) * q2 + 0.0333333) * q2 + 16.0
        elif q <= 25.0 and kd == 1:
            a0 = (((1.076676e-4 * q - 7.9684875e-3) * q + 0.17344854) * q - 0.5924058) * q + 16.620847
        elif q <= 20.0 and kd == 4:
            a0 = ((-7.08719e-4 * q + 3.8216144e-3) * q + 0.1907493) * q + 15.744
        else:
            a0 = cvql(kd, m, q)
    elif m == 5:
        if q <= 1.0 and kd == 2:
            a0 = ((6.8e-6 * q + 1.42e-5) * q2 + 0.0208333) * q2 + 25.0
        elif q <= 1.0 and kd == 3:
            a0 = ((-6.8e-6 * q + 1.42e-5) * q2 + 0.0208333) * q2 + 25.0
        elif q <= 35.0 and kd == 2:
            a0 = (((2.238231e-5 * q - 2.983416e-3) * q + 0.10706975) * q - 0.600205) * q + 25.93515
        elif q <= 25.0 and kd == 3:
            a0 = ((-7.425364e-4 * q + 2.18225e-2) * q + 4.16399e-2) * q + 24.897
        else:
            a0 = cvql(kd, m, q)
    elif m == 6:
        if q <= 1.0:
            a0 = (4e-6 * q2 + 0.0142857) * q2 + 36.0
        elif q <= 40.0 and kd == 1:
            a0 = (((-1.66846e-5 * q + 4.80263e-4) * q + 2.53998e-2) * q - 0.181233) * q + 36.423
        elif q <= 35.0 and kd == 4:
            a0 = ((-4.57146e-4 * q + 2.16609e-2) * q - 2.349616e-2) * q + 35.99251
        else:
            a0 = cvql(kd, m, q)
    elif m == 7:
        if q <= 10.0:
            a0 = cvqm(m, q)
        elif q <= 50.0 and kd == 2:
            a0 = (((-1.411114e-5 * q + 9.730514e-4) * q - 3.097887e-3) * q + 3.533597e-2) * q + 49.0547
        elif q <= 40.0 and kd == 3:
            a0 = ((-3.043872e-4 * q + 2.05511e-2) * q - 9.16292e-2) * q + 49.19035
        else:
            a0 = cvql(kd, m, q)
    elif m >= 8:
        if q <= 3 * m:
            a0 = cvqm(m, q)
        elif q > m * m:
            a0 = cvql(kd, m, q)
        else:
            if m == 8 and kd == 1:
                a0 = (((8.634308e-6 * q - 2.100289e-3) * q + 0.169072) * q - 4.64336) * q + 109.4211
            elif m == 8 and kd == 4:
                a0 = ((-6.7842e-5 * q + 2.2057e-3) * q + 0.48296) * q + 56.59
            elif m == 9 and kd == 2:
                a0 = (((2.906435e-6 * q - 1.019893e-3) * q + 0.1101965) * q - 3.821851) * q + 127.6098
            elif m == 9 and kd == 3:
                a0 = ((-9.577289e-5 * q + 0.01043839) * q + 0.06588934) * q + 78.0198
            elif m == 10 and kd == 1:
                a0 = (((5.44927e-7 * q - 3.926119e-4) * q + 0.0612099) * q - 2.600805) * q + 138.1923
            elif m == 10 and kd == 4:
                a0 = ((-7.660143e-5 * q + 0.01132506) * q - 0.09746023) * q + 99.29494
            elif m == 11 and kd == 2:
                a0 = (((-5.67615e-7 * q + 7.152722e-6) * q + 0.01920291) * q - 1.081583) * q + 140.88
            elif m == 11 and kd == 3:
                a0 = ((-6.310551e-5 * q + 0.0119247) * q - 0.2681195) * q + 123.667
            elif m == 12 and kd == 1:
                a0 = (((-2.38351e-7 * q - 2.90139e-5) * q + 0.02023088) * q - 1.289) * q + 171.2723
            elif m == 12 and kd == 4:
                a0 = (((3.08902e-7 * q - 1.577869e-4) * q + 0.0247911) * q - 1.05454) * q + 161.471
    return a0

def cva2(kd: int, m: int, q: float) -> float:
    a = 0.0
    if m <= 12 or q <= 3.0 * m or q > m * m:
        a = cv0(kd, m, q)
        if q != 0.0 and m != 2:
            a = refine(kd, m, q, a)
        if q > 2.0e-3 and m == 2:
            a = refine(kd, m, q, a)
    else:
        ndiv = 10
        delta = (m - 3.0) * m / ndiv
        if (q - 3.0 * m) <= (m * m - q):
            nn = int((q - 3.0 * m) / delta) + 1
            delta = (q - 3.0 * m) / nn
            q1 = 2.0 * m
            a1 = cvqm(m, q1)
            q2 = 3.0 * m
            a2 = cvqm(m, q2)
            qq = 3.0 * m
            for i in range(1, nn + 1):
                qq = qq + delta
                a = (a1 * q2 - a2 * q1 + (a2 - a1) * qq) / (q2 - q1)
                a = refine(kd, m, qq, a)
                q1 = q2
                q2 = qq
                a1 = a2
                a2 = a
        else:
            nn = int((m * m - q) / delta) + 1
            delta = (m * m - q) / nn
            q1 = m * (m - 1.0)
            a1 = cvql(kd, m, q1)
            q2 = float(m) * m
            a2 = cvql(kd, m, q2)
            qq = float(m) * m
            for i in range(1, nn + 1):
                qq = qq - delta
                a = (a1 * q2 - a2 * q1 + (a2 - a1) * qq) / (q2 - q1)
                a = refine(kd, m, qq, a)
                q1 = q2
                q2 = qq
                a1 = a2
                a2 = a
    return a

def _neg(fc: List[float], km: int):
    if fc[0] < 0.0:
        for j in range(km):
            fc[j] = -fc[j]

def fcoef(kd: int, m: int, q: float, a: float) -> List[float]:
    fc = [0.0] * 256
    if fabs(q) <= 1.0e-7:
        jm = 0
        if kd == 1:
            jm = m // 2 + 1
        elif kd == 2 or kd == 3:
            jm = (m - 1) // 2 + 1
        elif kd == 4:
            jm = m // 2
        if jm + 1 > 251:
            for i in range(251):
                fc[i] = NAN
            return fc
        if kd == 1 or kd == 2:
            if m == 0:
                fc[0] = 1.0 / sqrt(2.0)
                fc[1] = -q / (2.0 * sqrt(2.0))
            elif m == 1:
                fc[0] = 1.0
                fc[1] = -q / 8.0
            elif m == 2:
                fc[0] = q / 4.0
                fc[1] = 1.0
                fc[2] = -q / 12.0
            else:
                fc[jm - 1] = 1.0
                fc[jm] = -q / (4.0 * (m + 1))
                fc[jm - 2] = q / (4.0 * (m - 1))
        elif kd == 3 or kd == 4:
            if m == 1:
                fc[0] = 1.0
                fc[1] = -q / 8.0
            elif m == 2:
                fc[0] = 1.0
                fc[1] = -q / 12.0
            else:
                fc[jm - 1] = 1.0
                fc[jm] = -q / (4.0 * (m + 1))
                fc[jm - 2] = q / (4.0 * (m - 1))
        return fc
    qm = 0.0
    if q <= 1.0:
        qm = 7.5 + 56.1 * sqrt(q) - 134.7 * q + 90.7 * sqrt(q) * q
    else:
        qm = 17.0 + 3.1 * sqrt(q) - 0.126 * q + 0.0037 * sqrt(q) * q
    km = int(qm + 0.5 * m)
    if km > 251:
        for i in range(251):
            fc[i] = NAN
        return fc
    kb = 0
    s = 0.0
    f = 1.0e-100
    u = 0.0
    fc[km - 1] = 0.0
    f2 = 0.0
    f3 = 0.0
    v = 0.0
    if kd == 1:
        k = km
        while k >= 3:
            v = u
            u = f
            f = (a - 4.0 * k * k) * u / q - v
            if fabs(f) < fabs(fc[k]):
                kb = k
                fc[0] = 1.0e-100
                sp = 0.0
                f3 = fc[k]
                fc[1] = a / q * fc[0]
                fc[2] = (a - 4.0) * fc[1] / q - 2.0 * fc[0]
                u = fc[1]
                f1 = fc[2]
                for i in range(3, kb + 1):
                    v = u
                    u = f1
                    f1 = (a - 4.0 * (i - 1.0) * (i - 1.0)) * u / q - v
                    fc[i] = f1
                    if i == kb:
                        f2 = f1
                    if i != kb:
                        sp += f1 * f1
                sp += 2.0 * fc[0] * fc[0] + fc[1] * fc[1] + fc[2] * fc[2]
                ss = s + sp * (f3 / f2) * (f3 / f2)
                s0 = sqrt(1.0 / ss)
                for j in range(1, km + 1):
                    if j <= kb + 1:
                        fc[j - 1] = s0 * fc[j - 1] * f3 / f2
                    else:
                        fc[j - 1] *= s0
                _neg(fc, km)
                return fc
            else:
                fc[k - 1] = f
                s += f * f
            k -= 1
        fc[1] = q * fc[2] / (a - 4.0 - 2.0 * q * q / a)
        fc[0] = q / a * fc[1]
        s += 2.0 * fc[0] * fc[0] + fc[1] * fc[1]
        s0 = sqrt(1.0 / s)
        for k in range(1, km + 1):
            fc[k - 1] *= s0
    elif kd == 2 or kd == 3:
        found = False
        k = km
        while k >= 3:
            v = u
            u = f
            f = (a - (2.0 * k - 1) * (2.0 * k - 1)) * u / q - v
            if fabs(f) >= fabs(fc[k - 1]):
                fc[k - 2] = f
                s += f * f
            else:
                kb = k
                f3 = fc[k - 1]
                found = True
                break
            k -= 1
        sg = 1.0 if kd == 2 else -1.0   # pow(-1, kd)... kd=2 -> 1, kd=3 -> -1
        if not found:
            fc[0] = q / (a - 1.0 - sg * q) * fc[1]
            s += fc[0] * fc[0]
            s0 = sqrt(1.0 / s)
            for k in range(1, km + 1):
                fc[k - 1] *= s0
            _neg(fc, km)
            return fc
        fc[0] = 1.0e-100
        fc[1] = (a - 1.0 - sg * q) / q * fc[0]
        sp = 0.0
        u = fc[0]
        f1 = fc[1]
        for i in range(2, kb):
            v = u
            u = f1
            f1 = (a - (2.0 * i - 1) * (2.0 * i - 1)) * u / q - v
            if i != kb - 1:
                fc[i] = f1
                sp += f1 * f1
            else:
                f2 = f1
        sp += fc[0] * fc[0] + fc[1] * fc[1]
        ss = s + sp * (f3 / f2) * (f3 / f2)
        s0 = sqrt(1.0 / ss)
        for j in range(1, km + 1):
            if j < kb:
                fc[j - 1] *= s0 * f3 / f2
            else:
                fc[j - 1] *= s0
    elif kd == 4:
        found = False
        k = km
        while k >= 3:
            v = u
            u = f
            f = (a - 4.0 * k * k) * u / q - v
            if fabs(f) >= fabs(fc[k]):
                fc[k - 2] = f
                s += f * f
            else:
                kb = k
                f3 = fc[k - 1]
                found = True
                break
            k -= 1
        if not found:
            fc[0] = q / (a - 4.0) * fc[1]
            s += fc[0] * fc[0]
            s0 = sqrt(1.0 / s)
            for k in range(1, km + 1):
                fc[k - 1] *= s0
            _neg(fc, km)
            return fc
        fc[0] = 1.0e-100
        fc[1] = (a - 4.0) / q * fc[0]
        sp = 0.0
        u = fc[0]
        f1 = fc[1]
        for i in range(2, kb):
            v = u
            u = f1
            f1 = (a - 4.0 * i * i) * u / q - v
            if i != kb - 1:
                fc[i] = f1
                sp = sp + f1 * f1
            else:
                f2 = f1
        sp += fc[0] * fc[0] + fc[1] * fc[1]
        ss = s + sp * (f3 / f2) * (f3 / f2)
        s0 = sqrt(1.0 / ss)
        for j in range(1, km + 1):
            if j < kb:
                fc[j - 1] *= s0 * f3 / f2
            else:
                fc[j - 1] *= s0
    _neg(fc, km)
    return fc

def _kd(kf: int, m: int) -> int:
    if kf == 1:
        return 1 if m % 2 == 0 else 2
    return 3 if m % 2 != 0 else 4

def _qm(q: float) -> float:
    if q <= 1.0:
        return 7.5 + 56.1 * sqrt(q) - 134.7 * q + 90.7 * sqrt(q) * q
    return 17.0 + 3.1 * sqrt(q) - 0.126 * q + 0.0037 * sqrt(q) * q

def mtu0(kf: int, m: int, q: float, x: float):
    eps = 1.0e-14
    rd = 1.74532925199433e-2
    kd = _kd(kf, m)
    a = cva2(kd, m, q)
    km = int(_qm(q) + 0.5 * m)
    if km > 251:
        return (NAN, NAN)
    fg = fcoef(kd, m, q, a)
    ic = m // 2 + 1
    xr = x * rd
    csf = 0.0
    for k in range(1, km + 1):
        if kd == 1:
            csf += fg[k - 1] * cos((2 * k - 2) * xr)
        elif kd == 2:
            csf += fg[k - 1] * cos((2 * k - 1) * xr)
        elif kd == 3:
            csf += fg[k - 1] * sin((2 * k - 1) * xr)
        elif kd == 4:
            csf += fg[k - 1] * sin(2 * k * xr)
        if k >= ic and fabs(fg[k]) < fabs(csf) * eps:
            break
    csd = 0.0
    for k in range(1, km + 1):
        if kd == 1:
            csd -= (2 * k - 2) * fg[k - 1] * sin((2 * k - 2) * xr)
        elif kd == 2:
            csd -= (2 * k - 1) * fg[k - 1] * sin((2 * k - 1) * xr)
        elif kd == 3:
            csd += (2 * k - 1) * fg[k - 1] * cos((2 * k - 1) * xr)
        elif kd == 4:
            csd += 2.0 * k * fg[k - 1] * cos(2 * k * xr)
        if k >= ic and fabs(fg[k - 1]) < fabs(csd) * eps:
            break
    return (csf, csd)

def _jynb(n: int, x: float):
    bj = [0.0] * (n + 3)
    by = [0.0] * (n + 3)
    dj = [0.0] * (n + 3)
    dy = [0.0] * (n + 3)
    nm = jynbh(n, 0, x, bj, by)
    if x < 1.0e-100:
        for k in range(n + 1):
            dj[k] = 0.0
            dy[k] = 1.0e+300
        dj[1] = 0.5
    else:
        dj[0] = -bj[1]
        for k in range(1, nm + 1):
            dj[k] = bj[k - 1] - k / x * bj[k]
        dy[0] = -by[1]
        for k in range(1, nm + 1):
            dy[k] = by[k - 1] - k * by[k] / x
    return (bj, dj, by, dy)

def mtu12(kf: int, kc: int, m: int, q: float, x: float):
    """Returns (f1r, d1r, f2r, d2r)."""
    eps = 1.0e-14
    kd = _kd(kf, m)
    a = cva2(kd, m, q)
    km = int(_qm(q) + 0.5 * m)
    if km >= 251:
        return (NAN, NAN, NAN, NAN)
    fg = fcoef(kd, m, q, a)
    ic = m // 2 + 1
    if kd == 4:
        ic = m // 2
    c1 = exp(-x)
    c2 = exp(x)
    u1 = sqrt(q) * c1
    u2 = sqrt(q) * c2
    t1 = _jynb(km + 1, u1)
    t2 = _jynb(km + 1, u2)
    bj1 = t1[0]
    dj1 = t1[1]
    bj2 = t2[0]
    dj2 = t2[1]
    by2 = t2[2]
    dy2 = t2[3]
    w1 = 0.0
    w2 = 0.0
    sk = 1.0 if kd % 2 == 0 else -1.0
    f1r = 0.0
    d1r = 0.0
    if kc != 2:
        for k in range(1, km + 1):
            sg = -1.0 if (ic + k) % 2 != 0 else 1.0
            if kd == 1:
                f1r += sg * fg[k - 1] * bj1[k - 1] * bj2[k - 1]
            elif kd == 2 or kd == 3:
                f1r += sg * fg[k - 1] * (bj1[k - 1] * bj2[k] + sk * bj1[k] * bj2[k - 1])
            else:
                f1r += sg * fg[k - 1] * (bj1[k - 1] * bj2[k + 1] - bj1[k + 1] * bj2[k - 1])
            if k >= 5 and fabs(f1r - w1) < fabs(f1r) * eps:
                break
            w1 = f1r
        f1r /= fg[0]
        for k in range(1, km + 1):
            sg = -1.0 if (ic + k) % 2 != 0 else 1.0
            if kd == 1:
                d1r += sg * fg[k - 1] * (c2 * bj1[k - 1] * dj2[k - 1] - c1 * dj1[k - 1] * bj2[k - 1])
            elif kd == 2 or kd == 3:
                d1r += sg * fg[k - 1] * (c2 * (bj1[k - 1] * dj2[k] + sk * bj1[k] * dj2[k - 1]) - c1 * (dj1[k - 1] * bj2[k] + sk * dj1[k] * bj2[k - 1]))
            else:
                d1r += sg * fg[k - 1] * (c2 * (bj1[k - 1] * dj2[k + 1] - bj1[k + 1] * dj2[k - 1]) - c1 * (dj1[k - 1] * bj2[k + 1] - dj1[k + 1] * bj2[k - 1]))
            if k >= 5 and fabs(d1r - w2) < fabs(d1r) * eps:
                break
            w2 = d1r
        d1r *= sqrt(q) / fg[0]
        if kc == 1:
            return (f1r, d1r, 0.0, 0.0)
    f2r = 0.0
    d2r = 0.0
    for k in range(1, km + 1):
        sg = -1.0 if (ic + k) % 2 != 0 else 1.0
        if kd == 1:
            f2r += sg * fg[k - 1] * bj1[k - 1] * by2[k - 1]
        elif kd == 2 or kd == 3:
            f2r += sg * fg[k - 1] * (bj1[k - 1] * by2[k] + sk * bj1[k] * by2[k - 1])
        else:
            f2r += sg * fg[k - 1] * (bj1[k - 1] * by2[k + 1] - bj1[k + 1] * by2[k - 1])
        if k >= 5 and fabs(f2r - w1) < fabs(f2r) * eps:
            break
        w1 = f2r
    f2r /= fg[0]
    for k in range(1, km + 1):
        sg = -1.0 if (ic + k) % 2 != 0 else 1.0
        by1k = t1[2]
        if kd == 1:
            d2r += sg * fg[k - 1] * (c2 * bj1[k - 1] * dy2[k - 1] - c1 * dj1[k - 1] * by2[k - 1])
        elif kd == 2 or kd == 3:
            d2r += sg * fg[k - 1] * (c2 * (bj1[k - 1] * dy2[k] + sk * bj1[k] * dy2[k - 1]) - c1 * (dj1[k - 1] * by2[k] + sk * dj1[k] * by2[k - 1]))
        else:
            d2r += sg * fg[k - 1] * (c2 * (bj1[k - 1] * dy2[k + 1] - bj1[k + 1] * dy2[k - 1]) - c1 * (dj1[k - 1] * by2[k + 1] - dj1[k + 1] * by2[k - 1]))
        if k >= 5 and fabs(d2r - w2) < fabs(d2r) * eps:
            break
        w2 = d2r
    d2r = d2r * sqrt(q) / fg[0]
    return (f1r, d1r, f2r, d2r)

# ---- xsf mathieu_legacy.h wrappers (degrees for angular functions)
def _isint(m: float) -> bool:
    return m == float(int(m))

def cem_cva(m: float, q: float) -> float:
    if m < 0 or not _isint(m):
        return NAN
    im = int(m)
    if q < 0:
        if im % 2 == 0:
            return cem_cva(m, -q)
        return sem_cva(m, -q)
    return cva2(2 if im % 2 != 0 else 1, im, q)

def sem_cva(m: float, q: float) -> float:
    if m <= 0 or not _isint(m):
        return NAN
    im = int(m)
    if q < 0:
        if im % 2 == 0:
            return sem_cva(m, -q)
        return cem_cva(m, -q)
    return cva2(3 if im % 2 != 0 else 4, im, q)

def _sgn(im: int, odd_flip: bool) -> float:
    s = 1.0 if (im // 2) % 2 == 0 else -1.0
    return s

def cem(m: float, q: float, x: float):
    if m < 0 or not _isint(m):
        return (NAN, NAN)
    im = int(m)
    if q < 0:
        sg = 1.0 if (im // 2) % 2 == 0 else -1.0
        if im % 2 == 0:
            r = cem(m, -q, 90.0 - x)
        else:
            r = sem(m, -q, 90.0 - x)
        return (sg * r[0], -sg * r[1])
    return mtu0(1, im, q, x)

def sem(m: float, q: float, x: float):
    if m < 0 or not _isint(m):
        return (NAN, NAN)
    im = int(m)
    if im == 0:
        return (0.0, 0.0)
    if q < 0:
        if im % 2 == 0:
            sg = -1.0 if (im // 2) % 2 == 0 else 1.0
            r = sem(m, -q, 90.0 - x)
        else:
            sg = 1.0 if (im // 2) % 2 == 0 else -1.0
            r = cem(m, -q, 90.0 - x)
        return (sg * r[0], -sg * r[1])
    return mtu0(2, im, q, x)

def mcm1(m: float, q: float, x: float):
    if m < 0 or not _isint(m) or q < 0:
        return (NAN, NAN)
    r = mtu12(1, 1, int(m), q, x)
    return (r[0], r[1])

def msm1(m: float, q: float, x: float):
    if m < 1 or not _isint(m) or q < 0:
        return (NAN, NAN)
    r = mtu12(2, 1, int(m), q, x)
    return (r[0], r[1])

def mcm2(m: float, q: float, x: float):
    if m < 0 or not _isint(m) or q < 0:
        return (NAN, NAN)
    r = mtu12(1, 2, int(m), q, x)
    return (r[2], r[3])

def msm2(m: float, q: float, x: float):
    if m < 1 or not _isint(m) or q < 0:
        return (NAN, NAN)
    r = mtu12(2, 2, int(m), q, x)
    return (r[2], r[3])

def even_coef(m: int, q: float) -> List[float]:
    qm = _qm(q)
    km = int(qm + 0.5 * m)
    kd = 2 if m % 2 != 0 else 1
    a = cem_cva(float(m), q)
    fc = fcoef(kd, m, q, a)
    return fc[:km]

def odd_coef(m: int, q: float) -> List[float]:
    qm = _qm(q)
    km = int(qm + 0.5 * m)
    kd = 3 if m % 2 != 0 else 4
    b = sem_cva(float(m), q)
    fc = fcoef(kd, m, q, b)
    return fc[:km]
