# Ported from scipy 1.18.1 (BSD-3-Clause), FITPACK by P. Dierckx
# (scipy/interpolate/src/dfitpack.c). Faithful translation of curfit/fpcurf,
# parcur/fppara, fpchec, fpknot, fpdisc, fpback, fpbspl, fpgivs, fprota,
# fprati, fpader/spalde, insert/fpinst, sproot/fpcuro.  Arrays are flat
# float64 buffers; 2-D work arrays use column-major (i-1)+(j-1)*nest indexing
# exactly as the C source.

import numpy as np
from math import sqrt, fabs, pow, atan2, cos, copysign

_PI = 3.1415926535897932384626433832795

def fpbspl(t, k: int, x: float, l: int, h):
    hh = [0.0] * 20
    h[0] = 1.0
    for j in range(1, k + 1):
        for i in range(1, j + 1):
            hh[i - 1] = h[i - 1]
        h[0] = 0.0
        for i in range(1, j + 1):
            li = l + i
            lj = li - j
            if t[li - 1] != t[lj - 1]:
                f = hh[i - 1] / (t[li - 1] - t[lj - 1])
                h[i - 1] = h[i - 1] + f * (t[li - 1] - x)
                h[i] = f * (x - t[lj - 1])
            else:
                h[i] = 0.0

def fpgivs(piv: float, ww: float):
    store = fabs(piv)
    dd = 0.0
    if store >= ww:
        dd = store * sqrt(1.0 + (ww / piv) * (ww / piv))
    else:
        dd = ww * sqrt(1.0 + (piv / ww) * (piv / ww))
    return (ww / dd, piv / dd, dd)

def fpback(a, z, n: int, k: int, c, nest: int):
    c[n - 1] = z[n - 1] / a[n - 1]
    if n == 1:
        return
    k1 = k - 1
    i = n - 1
    for j in range(2, n + 1):
        store = z[i - 1]
        i1 = k1
        if j <= k1:
            i1 = j - 1
        m = i
        for l in range(1, i1 + 1):
            m += 1
            store = store - c[m - 1] * a[(i - 1) + l * nest]
        c[i - 1] = store / a[i - 1]
        i -= 1

def fpchec(x, m: int, t, n: int, k: int) -> int:
    k1 = k + 1
    k2 = k1 + 1
    nk1 = n - k1
    nk2 = nk1 + 1
    if nk1 < k1 or nk1 > m:
        return 10
    j = n
    for i in range(1, k + 1):
        if t[i - 1] > t[i]:
            return 20
        if t[j - 1] < t[j - 2]:
            return 20
        j -= 1
    for i in range(k2, nk2 + 1):
        if t[i - 1] <= t[i - 2]:
            return 30
    if x[0] < t[k] or x[m - 1] > t[nk2 - 1]:
        return 40
    if x[0] >= t[k2 - 1] or x[m - 1] <= t[nk1 - 1]:
        return 50
    i = 1
    l = k2
    nk3 = nk1 - 1
    if nk3 < 2:
        return 0
    for j in range(2, nk3 + 1):
        tj = t[j - 1]
        l += 1
        tl = t[l - 1]
        while True:
            i += 1
            if i >= m:
                return 50
            if not (x[i - 1] <= tj):
                break
        if x[i - 1] >= tl:
            return 50
    return 0

def fpdisc(t, n: int, k2: int, b, nest: int):
    k1 = k2 - 1
    k = k1 - 1
    nk1 = n - k1
    nrint = nk1 - k
    an = float(nrint)
    fac = an / (t[nk1] - t[k1 - 1])
    h = [0.0] * 12
    for l in range(k2, nk1 + 1):
        lmk = l - k1
        for j in range(1, k1 + 1):
            ik = j + k1
            lj = l + j
            lk = lj - k2
            h[j - 1] = t[l - 1] - t[lk - 1]
            h[ik - 1] = t[l - 1] - t[lj - 1]
        lp = lmk
        for j in range(1, k2 + 1):
            jk = j
            prod = h[j - 1]
            for i in range(1, k + 1):
                jk += 1
                prod = prod * h[jk - 1] * fac
            lk = lp + k1
            b[lmk - 1 + (j - 1) * nest] = (t[lk - 1] - t[lp - 1]) / prod
            lp += 1

def fpknot(x, t, n: int, fpint, nrdata, nrint: int, istart: int):
    # returns (n, nrint)
    k = (n - nrint - 1) // 2
    fpmax = 0.0
    jbegin = istart
    iserr = True
    maxbeg = 0
    maxpt = 0
    number = 0
    for j in range(1, nrint + 1):
        jpoint = nrdata[j - 1]
        if not (fpmax >= fpint[j - 1] or jpoint == 0):
            iserr = False
            fpmax = fpint[j - 1]
            number = j
            maxpt = jpoint
            maxbeg = jbegin
        jbegin += jpoint + 1
    if iserr:
        return (n + 1, nrint + 1)
    ihalf = maxpt // 2 + 1
    nrx = maxbeg + ihalf
    nxt = number + 1
    if nxt <= nrint:
        for j in range(nxt, nrint + 1):
            jj = nxt + nrint - j
            fpint[jj] = fpint[jj - 1]
            nrdata[jj] = nrdata[jj - 1]
            jk = jj + k
            t[jk] = t[jk - 1]
    nrdata[number - 1] = ihalf - 1
    nrdata[nxt - 1] = maxpt - ihalf
    am = float(maxpt)
    an = float(nrdata[number - 1])
    fpint[number - 1] = fpmax * an / am
    an = float(nrdata[nxt - 1])
    fpint[nxt - 1] = fpmax * an / am
    jk = nxt + k
    t[jk - 1] = x[nrx - 1]
    return (n + 1, nrint + 1)

def fprati(p1: float, f1: float, p2: float, f2: float, p3: float, f3: float):
    # returns (p, p1, f1, p3, f3)
    p = 0.0
    if p3 > 0.0:
        h1 = f1 * (f2 - f3)
        h2 = f2 * (f3 - f1)
        h3 = f3 * (f1 - f2)
        p = -(p1 * p2 * h3 + p2 * p3 * h1 + p3 * p1 * h2) / (p1 * h1 + p2 * h2 + p3 * h3)
    else:
        p = (p1 * (f1 - f3) * f2 - p2 * (f2 - f3) * f1) / ((f1 - f2) * f3)
    if f2 < 0:
        p3 = p2
        f3 = f2
    else:
        p1 = p2
        f1 = f2
    return (p, p1, f1, p3, f3)

def _interp_knots(x, t, m: int, k: int):
    k1 = k + 1
    k2 = k1 + 1
    mk1 = m - k1
    if mk1 != 0:
        k3 = k // 2
        i = k2
        j = k3 + 2
        if k3 * 2 == k:
            for l in range(mk1):
                t[i - 1] = (x[j - 1] + x[j - 2]) * 0.5
                i += 1
                j += 1
        else:
            for l in range(mk1):
                t[i - 1] = x[j - 1]
                i += 1
                j += 1


class CurfitWork:
    # persistent FITPACK work arrays (scipy keeps them between iopt=1 calls)
    nest: int
    idim: int
    t: np.ndarray[float, 1]
    c: np.ndarray[float, 1]
    fpint: np.ndarray[float, 1]
    nrdata: List[int]
    n: int
    fp: float
    ier: int

    def __init__(self, nest: int, idim: int = 1):
        self.nest = nest
        self.idim = idim
        self.t = np.zeros(nest)
        self.c = np.zeros(nest * idim)
        self.fpint = np.zeros(nest)
        self.nrdata = [0] * nest
        self.n = 0
        self.fp = 0.0
        self.ier = 0

def _fpcurve(iopt: int, u, x, w, m: int, ub: float, ue: float, k: int, s: float,
             wk: CurfitWork, tol: float, maxit: int, para: bool):
    # fpcurf (para=False, idim=1, x = y values) and fppara (para=True):
    # x holds m*idim values point-major, c/z use stride n per dimension.
    idim = wk.idim
    nest = wk.nest
    t = wk.t
    c = wk.c
    fpint = wk.fpint
    nrdata = wk.nrdata
    k1 = k + 1
    k2 = k1 + 1
    nc = nest * idim
    z = np.zeros(nc)
    a = np.zeros(nest * k1)
    b = np.zeros(nest * k2)
    g = np.zeros(nest * k2)
    q = np.zeros(m * k1)
    h = [0.0] * 7
    xi = [0.0] * 10
    n = wk.n
    ier = wk.ier
    fp = 0.0
    acc = 0.0
    fpold = 0.0
    fp0 = 0.0
    nplus = 0
    nmax = 0
    fpms = 0.0
    con1 = 0.1
    con9 = 0.9
    con4 = 0.04
    nmin = 2 * k1
    if iopt >= 0:
        acc = tol * s
        nmax = m + k1
        if s > 0.0:
            skip = False
            if iopt != 0 and n != nmin:
                fp0 = fpint[n - 1]
                fpold = fpint[n - 2]
                nplus = nrdata[n - 1]
                if fp0 > s:
                    skip = True
            if not skip:
                n = nmin
                fpold = 0.0
                nplus = 0
                nrdata[0] = m - 2
        else:
            n = nmax
            if nmax > nest:
                wk.n = n
                wk.ier = 1
                return
            _interp_knots(u, t, m, k)
    restart = True
    while restart:
        restart = False
        for iter in range(1, m + 1):
            if n == nmin:
                ier = -2
            nrint = n - nmin + 1
            nk1 = n - k1
            i = n
            for j in range(1, k1 + 1):
                t[j - 1] = ub
                t[i - 1] = ue
                i -= 1
            fp = 0.0
            for i in range(nc):
                z[i] = 0.0
            for i in range(1, nk1 + 1):
                for j in range(1, k1 + 1):
                    a[(i - 1) + (j - 1) * nest] = 0.0
            l = k1
            jj = 0
            for it in range(1, m + 1):
                ui = u[it - 1]
                wi = w[it - 1]
                for j in range(idim):
                    xi[j] = x[jj] * wi
                    jj += 1
                while not (ui < t[l] or l == nk1):
                    l += 1
                fpbspl(t, k, ui, l, h)
                for i in range(1, k1 + 1):
                    q[(it - 1) + (i - 1) * m] = h[i - 1]
                    h[i - 1] = h[i - 1] * wi
                j = l - k1
                for i in range(1, k1 + 1):
                    j += 1
                    piv = h[i - 1]
                    if piv == 0.0:
                        continue
                    cs, sn, ww = fpgivs(piv, a[j - 1])
                    a[j - 1] = ww
                    j1 = j
                    for j2 in range(idim):
                        tmp = cs * xi[j2] - sn * z[j1 - 1]
                        z[j1 - 1] = cs * z[j1 - 1] + sn * xi[j2]
                        xi[j2] = tmp
                        j1 += n
                    if i == k1:
                        break
                    i2 = 1
                    for i1 in range(i + 1, k1 + 1):
                        i2 += 1
                        idx = (j - 1) + (i2 - 1) * nest
                        tmp = cs * h[i1 - 1] - sn * a[idx]
                        a[idx] = cs * a[idx] + sn * h[i1 - 1]
                        h[i1 - 1] = tmp
                if para:
                    for j2 in range(idim):
                        fp += pow(xi[j2], 2.0)
                else:
                    fp = fp + xi[0] * xi[0]
            if ier == -2:
                fp0 = fp
            fpint[n - 1] = fp0
            fpint[n - 2] = fpold
            nrdata[n - 1] = nplus
            for j2 in range(idim):
                fpback(a, z[j2 * n:], nk1, k1, c[j2 * n:], nest)
            if iopt < 0:
                wk.n = n
                wk.fp = fp
                wk.ier = ier
                return
            fpms = fp - s
            if fabs(fpms) < acc:
                wk.n = n
                wk.fp = fp
                wk.ier = ier
                return
            if fpms < 0.0:
                break
            if n == nmax:
                wk.n = n
                wk.fp = fp
                wk.ier = -1
                return
            if n == nest:
                wk.n = n
                wk.fp = fp
                wk.ier = 1
                return
            if ier == 0:
                npl1 = nplus * 2
                rn = float(nplus)
                if fpold - fp > acc:
                    val = rn * fpms / (fpold - fp)
                    npl1 = 2147483647 if val > 2147483647.0 else int(val)
                temp1 = npl1
                temp2 = nplus // 2
                if temp2 > temp1:
                    temp1 = temp2
                if 1 > temp1:
                    temp1 = 1
                temp3 = nplus * 2
                if temp1 < temp3:
                    temp3 = temp1
                nplus = temp3
            else:
                nplus = 1
                ier = 0
            fpold = fp
            fpart = 0.0
            i = 1
            l = k2
            new = False
            jj = 0
            for it in range(1, m + 1):
                if not (u[it - 1] < t[l - 1] or l > nk1):
                    new = True
                    l += 1
                term = 0.0
                l0 = l - k2
                for j2 in range(idim):
                    fac = 0.0
                    j1 = l0
                    for j in range(1, k1 + 1):
                        j1 += 1
                        fac = fac + c[j1 - 1] * q[(it - 1) + (j - 1) * m]
                    term += pow(w[it - 1] * (fac - x[jj]), 2.0)
                    jj += 1
                    l0 += n
                fpart = fpart + term
                if not new:
                    continue
                store = term * 0.5
                fpint[i - 1] = fpart - store
                i += 1
                fpart = store
                new = False
            fpint[nrint - 1] = fpart
            for ll in range(1, nplus + 1):
                n, nrint = fpknot(u, t, n, fpint, nrdata, nrint, 1)
                if n == nmax:
                    _interp_knots(u, t, m, k)
                    restart = True
                    break
                if n == nest:
                    break
            if restart:
                break
    if ier == -2:
        wk.n = n
        wk.fp = fp
        wk.ier = ier
        return
    nk1 = n - k1
    fpdisc(t, n, k2, b, nest)
    p1 = 0.0
    f1 = fp0 - s
    p3 = -1.0
    f3 = fpms
    p = 0.0
    for i in range(1, nk1 + 1):
        p = p + a[i - 1]
    rn = float(nk1)
    p = rn / p
    ich1 = 0
    ich3 = 0
    n8 = n - nmin
    for iter in range(1, maxit + 1):
        pinv = 1.0 / p
        for i in range(nc):
            c[i] = z[i]
        for i in range(1, nk1 + 1):
            g[(i - 1) + (k2 - 1) * nest] = 0.0
            for j in range(1, k1 + 1):
                g[(i - 1) + (j - 1) * nest] = a[(i - 1) + (j - 1) * nest]
        for it in range(1, n8 + 1):
            for i in range(1, k2 + 1):
                h[i - 1] = b[(it - 1) + (i - 1) * nest] * pinv
            for j2 in range(idim):
                xi[j2] = 0.0
            for j in range(it, nk1 + 1):
                piv = h[0]
                cs, sn, ww = fpgivs(piv, g[j - 1])
                g[j - 1] = ww
                j1 = j
                for j2 in range(idim):
                    tmp = cs * xi[j2] - sn * c[j1 - 1]
                    c[j1 - 1] = cs * c[j1 - 1] + sn * xi[j2]
                    xi[j2] = tmp
                    j1 += n
                if j == nk1:
                    break
                i2 = k1
                if j > n8:
                    i2 = nk1 - j
                for i in range(1, i2 + 1):
                    idx = (j - 1) + i * nest
                    tmp = cs * h[i] - sn * g[idx]
                    g[idx] = cs * g[idx] + sn * h[i]
                    h[i] = tmp
                    h[i - 1] = h[i]
                h[i2] = 0.0
        for j2 in range(idim):
            cv = c[j2 * n:]
            fpback(g, cv, nk1, k2, cv, nest)
        fp = 0.0
        l = k2
        jj = 0
        for it in range(1, m + 1):
            if not (u[it - 1] < t[l - 1] or l > nk1):
                l += 1
            l0 = l - k2
            term = 0.0
            if para:
                for j2 in range(idim):
                    fac = 0.0
                    j1 = l0
                    for j in range(1, k1 + 1):
                        j1 += 1
                        fac += c[j1 - 1] * q[(it - 1) + (j - 1) * m]
                    term += (fac - x[jj]) * (fac - x[jj])
                    jj += 1
                    l0 += n
                fp += term * w[it - 1] * w[it - 1]
            else:
                for j in range(1, k1 + 1):
                    l0 += 1
                    term = term + c[l0 - 1] * q[(it - 1) + (j - 1) * m]
                fp = fp + pow(w[it - 1] * (term - x[it - 1]), 2.0)
        fpms = fp - s
        if fabs(fpms) < acc:
            break
        if iter == maxit:
            ier = 3
            break
        p2 = p
        f2 = fpms
        if ich3 == 0:
            if (f2 - f3) > acc:
                if f2 < 0.0:
                    ich3 = 1
            else:
                p3 = p2
                f3 = f2
                p = p * con4
                if p <= p1:
                    p = p1 * con9 + p2 * con1
                continue
        if ich1 == 0:
            if (f1 - f2) > acc:
                if f2 > 0.0:
                    ich1 = 1
            else:
                p1 = p2
                f1 = f2
                p = p / con4
                if p3 < 0.0:
                    continue
                if p >= p3:
                    p = p2 * con1 + p3 * con9
                continue
        if f2 >= f1 or f2 <= f3:
            ier = 2
            break
        p, p1, f1, p3, f3 = fprati(p1, f1, p2, f2, p3, f3)
    wk.n = n
    wk.fp = fp
    wk.ier = ier

def curfit(iopt: int, x, y, w, xb: float, xe: float, k: int, s: float, wk: CurfitWork):
    m = x.shape[0]
    nest = wk.nest
    wk.ier = 10
    if k <= 0 or k > 5:
        return
    k1 = k + 1
    if iopt < -1 or iopt > 1:
        return
    nmin = 2 * k1
    if m < k1 or nest < nmin:
        return
    if xb > x[0] or xe < x[m - 1]:
        return
    for i in range(1, m):
        if x[i - 1] > x[i]:
            return
    if iopt >= 0:
        if s < 0.0:
            return
        if s == 0.0 and nest < m + k1:
            return
        wk.ier = 0
    else:
        n = wk.n
        if n < nmin or n > nest:
            return
        j = n
        for i in range(k1):
            wk.t[i] = xb
            wk.t[j - 1] = xe
            j -= 1
        wk.ier = fpchec(x, m, wk.t, n, k)
        if wk.ier != 0:
            return
    _fpcurve(iopt, x, y, w, m, xb, xe, k, s, wk, 1e-3, 20, False)

def parcur(iopt: int, ipar: int, u, x, w, ub: float, ue: float, k: int, s: float,
           wk: CurfitWork):
    # returns (ub, ue); u is updated in place when ipar == 0
    idim = wk.idim
    m = u.shape[0]
    nest = wk.nest
    wk.ier = 10
    if iopt < -1 or iopt > 1 or ipar < 0 or ipar > 1 or idim <= 0 or idim > 10:
        return (ub, ue)
    if k <= 0 or k > 5:
        return (ub, ue)
    k1 = k + 1
    nmin = 2 * k1
    if m < k1 or nest < nmin:
        return (ub, ue)
    if ipar == 0 and iopt <= 0:
        i1 = 0
        i2 = idim
        u[0] = 0.0
        for i in range(2, m + 1):
            dist = 0.0
            for j in range(idim):
                i1 += 1
                i2 += 1
                dist += pow(x[i2 - 1] - x[i1 - 1], 2.0)
            u[i - 1] = u[i - 2] + sqrt(dist)
        if u[m - 1] <= 0.0:
            return (ub, ue)
        for i in range(2, m + 1):
            u[i - 1] /= u[m - 1]
        ub = 0.0
        ue = 1.0
        u[m - 1] = ue
    if ub > u[0] or ue < u[m - 1] or w[0] <= 0.0:
        return (ub, ue)
    for i in range(1, m):
        if u[i - 1] >= u[i] or w[i] <= 0.0:
            return (ub, ue)
    if iopt >= 0:
        if s < 0.0:
            return (ub, ue)
        if s == 0.0 and nest < m + k1:
            return (ub, ue)
        wk.ier = 0
    else:
        n = wk.n
        if n < nmin or n > nest:
            return (ub, ue)
        j = n
        for i in range(k1):
            wk.t[i] = ub
            wk.t[j - 1] = ue
            j -= 1
        wk.ier = fpchec(u, m, wk.t, n, k)
        if wk.ier != 0:
            return (ub, ue)
    _fpcurve(iopt, u, x, w, m, ub, ue, k, s, wk, 1e-3, 20, True)
    return (ub, ue)

def fpader(t, c, k1: int, x: float, l: int, d):
    h = [0.0] * 20
    lk = l - k1
    for i in range(1, k1 + 1):
        ik = i + lk
        h[i - 1] = c[ik - 1]
    kj = k1
    fac = 1.0
    for j in range(1, k1 + 1):
        ki = kj
        j1 = j + 1
        if j != 1:
            i = k1
            for jj in range(j, k1 + 1):
                li = i + lk
                lj = li + kj
                h[i - 1] = (h[i - 1] - h[i - 2]) / (t[lj - 1] - t[li - 1])
                i -= 1
        for i in range(j, k1 + 1):
            d[i - 1] = h[i - 1]
        if j != k1:
            for jj in range(j1, k1 + 1):
                ki -= 1
                i = k1
                for j2 in range(jj, k1 + 1):
                    li = i + lk
                    lj = li + ki
                    d[i - 1] = ((x - t[li - 1]) * d[i - 1] + (t[lj - 1] - x) * d[i - 2]) / (t[lj - 1] - t[li - 1])
                    i -= 1
        d[j - 1] = d[k1 - 1] * fac
        fac *= float(k1 - j)
        kj -= 1

def spalde(t, n: int, c, k1: int, x: float, d) -> int:
    nk1 = n - k1
    if x < t[k1 - 1] or x > t[nk1]:
        return 10
    l = k1
    while not (x < t[l] or l == nk1):
        l += 1
    if t[l - 1] >= t[l]:
        return 10
    fpader(t, c, k1, x, l, d)
    return 0

def fpinst(iopt: int, t, n: int, c, k: int, x: float, l: int, tt, cc) -> int:
    k1 = k + 1
    nk1 = n - k1
    ll = l + 1
    i = n
    for j in range(ll, n + 1):
        tt[i] = t[i - 1]
        i -= 1
    tt[ll - 1] = x
    for j in range(1, l + 1):
        tt[j - 1] = t[j - 1]
    i = nk1
    for j in range(l, nk1 + 1):
        cc[i] = c[i - 1]
        i -= 1
    i = l
    for j in range(1, k + 1):
        m = i + k1
        fac = (x - tt[i - 1]) / (tt[m - 1] - tt[i - 1])
        i1 = i - 1
        cc[i - 1] = fac * c[i - 1] + (1.0 - fac) * c[i1 - 1]
        i -= 1
    for j in range(1, i + 1):
        cc[j - 1] = c[j - 1]
    nn = n + 1
    if iopt == 0:
        return nn
    nk = nn - k
    nl = nk - k1
    per = tt[nk - 1] - tt[k1 - 1]
    i = k1
    j = nk
    if ll > nl:
        for m in range(1, k + 1):
            mk = m + nl
            cc[m - 1] = cc[mk - 1]
            i -= 1
            j -= 1
            tt[i - 1] = tt[j - 1] - per
        return nn
    if ll > k1 + k:
        return nn
    for m in range(1, k + 1):
        mk = m + nl
        cc[mk - 1] = cc[m - 1]
        i += 1
        j += 1
        tt[j - 1] = tt[i - 1] + per
    return nn

def insert(iopt: int, t, n: int, c, k: int, x: float, tt, cc, nest: int):
    # returns (nn, ier)
    if nest <= n:
        return (0, 10)
    k1 = k + 1
    nk = n - k
    if x < t[k1 - 1] or x > t[nk - 1]:
        return (0, 10)
    l = k1
    while x >= t[l]:
        l += 1
        if l == nk:
            l = nk - 1
            while x <= t[l - 1]:
                l -= 1
                if l == k:
                    return (0, 10)
            break
    if t[l - 1] >= t[l]:
        return (0, 10)
    if iopt != 0:
        kk = 2 * k
        if l <= kk and l >= n - kk:
            return (0, 10)
    nn = fpinst(iopt, t, n, c, k, x, l, tt, cc)
    return (nn, 0)

def fpcuro(a: float, b: float, c: float, d: float, x) -> int:
    e3 = 0.1 / 0.3
    pi3 = _PI / 3.0
    a1 = fabs(a)
    b1 = fabs(b)
    c1 = fabs(c)
    d1 = fabs(d)
    n = 0
    if max(max(b1, c1), d1) < a1 * 1.0e4:
        b1 = b / a * e3
        c1 = c / a
        d1 = d / a
        q = c1 * e3 - b1 * b1
        r = b1 * b1 * b1 + (d1 - b1 * c1) * 0.5
        disc = q * q * q + r * r
        if disc > 0.0:
            u = sqrt(disc)
            u1 = -r + u
            u2 = -r - u
            n = 1
            x[0] = copysign(pow(fabs(u1), e3), u1) + copysign(pow(fabs(u2), e3), u2) - b1
        else:
            u = sqrt(fabs(q))
            if r < 0.0:
                u = -u
            p3 = atan2(sqrt(-disc), fabs(r)) * e3
            u2 = u + u
            n = 3
            x[0] = -u2 * cos(p3) - b1
            x[1] = u2 * cos(pi3 - p3) - b1
            x[2] = u2 * cos(pi3 + p3) - b1
    elif max(c1, d1) < b1 * 1.0e4:
        disc = c * c - 4.0 * b * d
        if disc < 0.0:
            return 0
        n = 2
        u = sqrt(disc)
        b1 = b + b
        x[0] = (-c + u) / b1
        x[1] = (-c - u) / b1
    elif d1 < c1 * 1.0e4:
        n = 1
        x[0] = -d / c
    else:
        return 0
    for i in range(n):
        y = x[i]
        f = ((a * y + b) * y + c) * y + d
        df = (3.0 * a * y + 2.0 * b) * y + c
        step = 0.0
        if fabs(f) < fabs(df) * 0.1:
            step = f / df
        x[i] = y - step
    return n

def sproot(t, n: int, c, mest: int):
    # returns (zeros list, ier)
    zero = List[float]()
    if n < 8:
        return (zero, 10)
    j = n
    for i in range(3):
        if t[i] > t[i + 1]:
            return (zero, 10)
        if t[j - 1] < t[j - 2]:
            return (zero, 10)
        j -= 1
    for i in range(3, n - 4):
        if t[i] >= t[i + 1]:
            return (zero, 10)
    ier = 0
    y = [0.0, 0.0, 0.0]
    h1 = t[3] - t[2]
    h2 = t[4] - t[3]
    t1 = t[3] - t[1]
    t2 = t[4] - t[2]
    t3 = t[5] - t[3]
    t4 = t[4] - t[1]
    t5 = t[5] - t[2]
    c1 = c[0]
    c2 = c[1]
    c3 = c[2]
    c4 = (c2 - c1) / t4
    c5 = (c3 - c2) / t5
    d4 = (h2 * c1 + t1 * c2) / t4
    d5 = (t3 * c2 + h1 * c3) / t5
    a0 = (h2 * d4 + h1 * d5) / t2
    ah = 3.0 * (h2 * c4 + h1 * c5) / t2
    z1 = ah >= 0.0
    nz1 = not z1
    for l in range(3, n - 4):
        h1 = h2
        h2 = t[l + 2] - t[l + 1]
        t1 = t2
        t2 = t3
        t3 = t[l + 3] - t[l + 1]
        t4 = t5
        t5 = t[l + 3] - t[l]
        c1 = c2
        c2 = c3
        c3 = c[l]
        c4 = c5
        c5 = (c3 - c2) / t5
        d4 = (h2 * c1 + t1 * c2) / t4
        d5 = (h1 * c3 + t3 * c2) / t5
        b0 = (h2 * d4 + h1 * d5) / t2
        bh = 3.0 * (h2 * c4 + h1 * c5) / t2
        a1 = ah * h1
        b1 = bh * h1
        a2 = 3.0 * (b0 - a0) - b1 - 2.0 * a1
        a3 = 2.0 * (a0 - b0) + b1 + a1
        z3 = b1 >= 0.0
        nz3 = not z3
        look = True
        if not (a0 * b0 <= 0.0):
            z0 = a0 >= 0.0
            nz0 = not z0
            z2 = a2 >= 0.0
            nz2 = not z2
            z4 = (3.0 * a3 + a2) >= 0.0
            nz4 = not z4
            look = ((z0 and ((nz1 and (z3 or (z2 and nz4))) or (nz2 and z3 and z4))) or
                    (nz0 and ((z1 and (nz3 or (nz2 and z4))) or (z2 and nz3 and nz4))))
        if look:
            jn = fpcuro(a3, a2, a1, a0, y)
            for i in range(jn):
                if y[i] < 0.0 or y[i] > 1.0:
                    continue
                if len(zero) >= mest:
                    return (zero, 1)
                zero.append(t[l] + h1 * y[i])
        a0 = b0
        ah = bh
        z1 = z3
        nz1 = nz3
    m = len(zero)
    if m < 2:
        return (zero, ier)
    for i in range(1, m):
        j = i
        while j > 0:
            if zero[j] >= zero[j - 1]:
                break
            zz = zero[j]
            zero[j] = zero[j - 1]
            zero[j - 1] = zz
            j -= 1
    out = [zero[0]]
    for i in range(1, m):
        if zero[i] == out[len(out) - 1]:
            continue
        out.append(zero[i])
    return (out, ier)
