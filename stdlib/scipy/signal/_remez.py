# Copyright (C) 2026 the Typhon authors
# Ported from scipy 1.18.1 (BSD-3-Clause)

# Parks-McClellan (remez exchange) FIR design, ported from scipy's
# _sigtoolsmodule.cc (McClellan, Parks & Rabiner; C by E. Kvaleberg).
# The goto-based exchange loop is expressed as a label state machine.

import math

_PI = 3.14159265358979323846
_TWOPI = _PI + _PI

def _lagrange_interp(k: int, n: int, m: int, x: List[float]) -> float:
    retval = 1.0
    q = x[k]
    for l in range(1, m + 1):
        j = l
        while j <= n:
            if j != k:
                retval *= 2.0 * (q - x[j])
            j += m
    return 1.0 / retval

def _freq_eval(k: int, n: int, grid: List[float], x: List[float], y: List[float], ad: List[float]) -> float:
    d = 0.0
    p = 0.0
    xf = math.cos(_TWOPI * grid[k])
    for j in range(1, n + 1):
        c = ad[j] / (xf - x[j])
        d += c
        p += c * y[j]
    return p / d

def _remez_core(des: List[float], grid: List[float], edge: List[float], wt: List[float],
                ngrid: int, nbands: int, iext: List[int], alpha: List[float],
                nfcns: int, itrmax: int, dimsize: int):
    # returns (status, niter); status -1 = failure to converge
    a = [0.0] * (dimsize + 2)
    p = [0.0] * (dimsize + 2)
    q = [0.0] * (dimsize + 2)
    ad = [0.0] * (dimsize + 2)
    x = [0.0] * (dimsize + 2)
    y = [0.0] * (dimsize + 2)
    devl = -1.0
    nz = nfcns + 1
    nzz = nfcns + 2
    niter = 0
    dev = 0.0
    ynz = 0.0; comp = 0.0; y1 = 0.0
    luck = 0; nut1 = 0
    k1 = 0; knz = 0; klow = 0; kup = 0; nu = 0; nut = 0; jchnge = 0
    j = 0; l = 0; err = 0.0
    lab = 100
    while True:
        if lab == 100:
            iext[nzz] = ngrid + 1
            niter += 1
            if niter > itrmax:
                break
            for jj in range(1, nz + 1):
                x[jj] = math.cos(grid[iext[jj]] * _TWOPI)
            jet = (nfcns - 1) // 15 + 1
            for jj in range(1, nz + 1):
                ad[jj] = _lagrange_interp(jj, nz, jet, x)
            dnum = 0.0
            dden = 0.0
            k = 1
            for jj in range(1, nz + 1):
                ll = iext[jj]
                dnum += ad[jj] * des[ll]
                dden += float(k) * ad[jj] / wt[ll]
                k = -k
            dev = dnum / dden
            nu = 1
            if dev > 0.0:
                nu = -1
            dev = -float(nu) * dev
            k = nu
            for jj in range(1, nz + 1):
                ll = iext[jj]
                y[jj] = des[ll] + float(k) * dev / wt[ll]
                k = -k
            if dev <= devl:
                return -1, niter
            devl = dev
            jchnge = 0
            k1 = iext[1]
            knz = iext[nz]
            klow = 0
            nut = -nu
            j = 1
            lab = 200
        elif lab == 200:
            if j == nzz:
                ynz = comp
            if j >= nzz:
                lab = 300
                continue
            kup = iext[j + 1]
            l = iext[j] + 1
            nut = -nut
            if j == 2:
                y1 = comp
            comp = dev
            if l >= kup:
                lab = 220
                continue
            err = (_freq_eval(l, nz, grid, x, y, ad) - des[l]) * wt[l]
            if (float(nut) * err - comp) <= 0.0:
                lab = 220
                continue
            comp = float(nut) * err
            lab = 210
        elif lab == 210:
            l += 1
            if l >= kup:
                lab = 215
                continue
            err = (_freq_eval(l, nz, grid, x, y, ad) - des[l]) * wt[l]
            if (float(nut) * err - comp) <= 0.0:
                lab = 215
                continue
            comp = float(nut) * err
            lab = 210
        elif lab == 215:
            iext[j] = l - 1
            j += 1
            klow = l - 1
            jchnge += 1
            lab = 200
        elif lab == 220:
            l -= 1
            lab = 225
        elif lab == 225:
            l -= 1
            if l <= klow:
                lab = 250
                continue
            err = (_freq_eval(l, nz, grid, x, y, ad) - des[l]) * wt[l]
            if (float(nut) * err - comp) > 0.0:
                lab = 230
                continue
            if jchnge <= 0:
                lab = 225
                continue
            lab = 260
        elif lab == 230:
            comp = float(nut) * err
            lab = 235
        elif lab == 235:
            l -= 1
            if l <= klow:
                lab = 240
                continue
            err = (_freq_eval(l, nz, grid, x, y, ad) - des[l]) * wt[l]
            if (float(nut) * err - comp) <= 0.0:
                lab = 240
                continue
            comp = float(nut) * err
            lab = 235
        elif lab == 240:
            klow = iext[j]
            iext[j] = l + 1
            j += 1
            jchnge += 1
            lab = 200
        elif lab == 250:
            l = iext[j] + 1
            if jchnge > 0:
                lab = 215
                continue
            lab = 255
        elif lab == 255:
            l += 1
            if l >= kup:
                lab = 260
                continue
            err = (_freq_eval(l, nz, grid, x, y, ad) - des[l]) * wt[l]
            if (float(nut) * err - comp) <= 0.0:
                lab = 255
                continue
            comp = float(nut) * err
            lab = 210
        elif lab == 260:
            klow = iext[j]
            j += 1
            lab = 200
        elif lab == 300:
            if j > nzz:
                lab = 320
                continue
            if k1 > iext[1]:
                k1 = iext[1]
            if knz < iext[nz]:
                knz = iext[nz]
            nut1 = nut
            nut = -nu
            l = 0
            kup = k1
            comp = ynz * 1.00001
            luck = 1
            lab = 310
        elif lab == 310:
            l += 1
            if l >= kup:
                lab = 315
                continue
            err = (_freq_eval(l, nz, grid, x, y, ad) - des[l]) * wt[l]
            if (float(nut) * err - comp) <= 0.0:
                lab = 310
                continue
            comp = float(nut) * err
            j = nzz
            lab = 210
        elif lab == 315:
            luck = 6
            lab = 325
        elif lab == 320:
            if luck > 9:
                lab = 350
                continue
            if comp > y1:
                y1 = comp
            k1 = iext[nzz]
            lab = 325
        elif lab == 325:
            l = ngrid + 1
            klow = knz
            nut = -nut1
            comp = y1 * 1.00001
            lab = 330
        elif lab == 330:
            l -= 1
            if l <= klow:
                lab = 340
                continue
            err = (_freq_eval(l, nz, grid, x, y, ad) - des[l]) * wt[l]
            if (float(nut) * err - comp) <= 0.0:
                lab = 330
                continue
            j = nzz
            comp = float(nut) * err
            luck = luck + 10
            lab = 235
        elif lab == 340:
            if luck == 6:
                lab = 370
                continue
            for jj in range(1, nfcns + 1):
                iext[nzz - jj] = iext[nz - jj]
            iext[1] = k1
            lab = 100
        elif lab == 350:
            kn = iext[nzz]
            for jj in range(1, nfcns + 1):
                iext[jj] = iext[jj + 1]
            iext[nz] = kn
            lab = 100
        else:  # 370: end of do-while body
            if jchnge > 0:
                lab = 100
                continue
            break

    # coefficients of the best approximation (inverse DFT)
    nm1 = nfcns - 1
    fsh = 1.0e-06
    gtemp = grid[1]
    x[nzz] = -2.0
    cn = 2 * nfcns - 1
    delf = 1.0 / cn
    l = 1
    kkk = 0
    aa = 0.0
    bb = 0.0
    if edge[1] == 0.0 and edge[2 * nbands] == 0.5:
        kkk = 1
    if nfcns <= 3:
        kkk = 1
    if kkk != 1:
        dtemp = math.cos(_TWOPI * grid[1])
        dnum = math.cos(_TWOPI * grid[ngrid])
        aa = 2.0 / (dtemp - dnum)
        bb = -(dtemp + dnum) / (dtemp - dnum)
    for jj in range(1, nfcns + 1):
        ft = (jj - 1) * delf
        xt = math.cos(_TWOPI * ft)
        if kkk != 1:
            xt = (xt - bb) / aa
            ft = math.acos(xt) / _TWOPI
        while True:
            xe = x[l]
            if xt > xe:
                if (xt - xe) < fsh:
                    a[jj] = y[l]
                else:
                    grid[1] = ft
                    a[jj] = _freq_eval(1, nz, grid, x, y, ad)
                break
            if (xe - xt) < fsh:
                a[jj] = y[l]
                break
            l += 1
        if l > 1:
            l = l - 1
    grid[1] = gtemp
    dden = _TWOPI / cn
    for jj in range(1, nfcns + 1):
        dtemp = 0.0
        dnum = (jj - 1) * dden
        if nm1 >= 1:
            for k in range(1, nm1 + 1):
                dtemp += a[k + 1] * math.cos(dnum * k)
        alpha[jj] = 2.0 * dtemp + a[1]
    for jj in range(2, nfcns + 1):
        alpha[jj] *= 2.0 / cn
    alpha[1] /= cn
    if kkk != 1:
        p[1] = 2.0 * alpha[nfcns] * bb + alpha[nm1]
        p[2] = 2.0 * aa * alpha[nfcns]
        q[1] = alpha[nfcns - 2] - alpha[nfcns]
        for jj in range(2, nm1 + 1):
            if jj >= nm1:
                aa *= 0.5
                bb *= 0.5
            p[jj + 1] = 0.0
            for k in range(1, jj + 1):
                a[k] = p[k]
                p[k] = 2.0 * bb * a[k]
            p[2] += a[1] * 2.0 * aa
            for k in range(1, jj):
                p[k] += q[k] + aa * a[k + 1]
            for k in range(3, jj + 2):
                p[k] += aa * a[k - 1]
            if jj != nm1:
                for k in range(1, jj + 1):
                    q[k] = -a[k]
                q[1] += alpha[nfcns - 1 - jj]
        for jj in range(1, nfcns + 1):
            alpha[jj] = p[jj]
    if nfcns <= 3:
        alpha[nfcns + 1] = 0.0
        alpha[nfcns + 2] = 0.0
    return 0, niter

def _eff(freq: float, fx: List[float], lband: int, jtype: int) -> float:
    if jtype != 2:
        return fx[lband]
    return fx[lband] * freq

def _wate(freq: float, fx: List[float], wtx: List[float], lband: int, jtype: int) -> float:
    if jtype != 2:
        return wtx[lband]
    if fx[lband] >= 0.0001:
        return wtx[lband] / freq
    return wtx[lband]

def _pre_remez(numtaps: int, numbands: int, bands: List[float], response: List[float],
               weight: List[float], jtype: int, maxiter: int, grid_density: int) -> List[float]:
    lgrid = grid_density
    dimsize = int(math.ceil(numtaps / 2.0 + 2))
    wrksize = grid_density * dimsize
    nfilt = numtaps
    edge = [0.0] + bands
    fx = [0.0] + response
    wtx = [0.0] + weight
    des = [0.0] * (wrksize + 2)
    grid = [0.0] * (wrksize + 2)
    wt = [0.0] * (wrksize + 2)
    alpha = [0.0] * (dimsize + 3)
    iext = [0] * (dimsize + 3)
    neg = 0 if jtype == 1 else 1
    nodd = nfilt % 2
    nfcns = nfilt // 2
    if nodd == 1 and neg == 0:
        nfcns = nfcns + 1
    grid[1] = edge[1]
    delf = 0.5 / (lgrid * nfcns)
    if neg != 0:
        if edge[1] < delf:
            grid[1] = delf
    j = 1
    l = 1
    lband = 1
    while True:
        fup = edge[l + 1]
        while True:
            temp = grid[j]
            des[j] = _eff(temp, fx, lband, jtype)
            wt[j] = _wate(temp, fx, wtx, lband, jtype)
            j += 1
            if j > wrksize:
                raise ValueError('Failure to converge at iteration 0, try reducing transition band width.\n')
            grid[j] = temp + delf
            if not (grid[j] <= fup):
                break
        grid[j - 1] = fup
        des[j - 1] = _eff(fup, fx, lband, jtype)
        wt[j - 1] = _wate(fup, fx, wtx, lband, jtype)
        lband += 1
        l += 2
        if lband > numbands:
            break
        grid[j] = edge[l]
    ngrid = j - 1
    if neg == nodd:
        if grid[ngrid] > (0.5 - delf):
            ngrid -= 1
    if ngrid < nfcns + 1:
        raise ValueError(f'Band edges are too close together to build the dense frequency grid: it has only {ngrid} point(s) but at least {nfcns + 1} (one per extremal frequency) are required. Widen the bands, reduce numtaps, or increase grid_density.')
    if neg <= 0:
        if nodd != 1:
            for jj in range(1, ngrid + 1):
                change = math.cos(_PI * grid[jj])
                des[jj] = des[jj] / change
                wt[jj] = wt[jj] * change
    else:
        if nodd != 1:
            for jj in range(1, ngrid + 1):
                change = math.sin(_PI * grid[jj])
                des[jj] = des[jj] / change
                wt[jj] = wt[jj] * change
        else:
            for jj in range(1, ngrid + 1):
                change = math.sin(_TWOPI * grid[jj])
                des[jj] = des[jj] / change
                wt[jj] = wt[jj] * change
    temp = float(ngrid - 1) / float(nfcns)
    for jj in range(1, nfcns + 1):
        iext[jj] = int((jj - 1) * temp) + 1
    iext[nfcns + 1] = ngrid
    nm1 = nfcns - 1
    nz = nfcns + 1
    st, niter = _remez_core(des, grid, edge, wt, ngrid, numbands, iext, alpha, nfcns,
                            maxiter, dimsize)
    if st < 0:
        raise ValueError(f'Failure to converge at iteration {niter}, try reducing transition band width.\n')
    h = [0.0] * (nfilt + 2)
    if neg <= 0:
        if nodd != 0:
            for jj in range(1, nm1 + 1):
                h[jj] = 0.5 * alpha[nz - jj]
            h[nfcns] = alpha[1]
        else:
            h[1] = 0.25 * alpha[nfcns]
            for jj in range(2, nm1 + 1):
                h[jj] = 0.25 * (alpha[nz - jj] + alpha[nfcns + 2 - jj])
            h[nfcns] = 0.5 * alpha[1] + 0.25 * alpha[2]
    else:
        if nodd != 0:
            h[1] = 0.25 * alpha[nfcns]
            h[2] = 0.25 * alpha[nm1]
            for jj in range(3, nm1 + 1):
                h[jj] = 0.25 * (alpha[nz - jj] - alpha[nfcns + 3 - jj])
            h[nfcns] = 0.5 * alpha[1] - 0.25 * alpha[3]
            h[nz] = 0.0
        else:
            h[1] = 0.25 * alpha[nfcns]
            for jj in range(2, nm1 + 1):
                h[jj] = 0.25 * (alpha[nz - jj] - alpha[nfcns + 2 - jj])
            h[nfcns] = 0.5 * alpha[1] - 0.25 * alpha[2]
    for jj in range(1, nfcns + 1):
        k = nfilt + 1 - jj
        if neg == 0:
            h[k] = h[jj]
        else:
            h[k] = -h[jj]
    if neg == 1 and nodd == 1:
        h[nz] = 0.0
    return h[1:nfilt + 1]
