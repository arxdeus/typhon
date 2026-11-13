# Ported from scipy 1.18.1 (BSD-3-Clause), FITPACK by P. Dierckx
# (scipy/interpolate/src/dfitpack.c): surfit/fpsurf (scattered-data
# smoothing / least-squares bivariate splines), fporde, fprank.
# Work arrays are sized so that the rank-deficient branch never runs out of
# space (scipy's wrappers allocate lwrk2 = u*v*(b2+1)+b2 which suffices).

import numpy as np
from math import sqrt, fabs, pow
from _dierckx import fpbspl, fpgivs, fpback, fpdisc, fprati

def _rot(cs: float, sn: float, a: float, b: float):
    return (cs * a - sn * b, cs * b + sn * a)

def _fporde(x, y, m: int, kx: int, ky: int, tx, nx: int, ty, ny: int, nummer, index, nreg: int):
    kx1 = kx + 1
    ky1 = ky + 1
    nk1x = nx - kx1
    nk1y = ny - ky1
    nyy = nk1y - ky
    for i in range(nreg):
        index[i] = 0
    for im in range(1, m + 1):
        xi = x[im - 1]
        yi = y[im - 1]
        l = kx1
        l1 = l + 1
        while not (xi < tx[l1 - 1] or l == nk1x):
            l = l1
            l1 = l + 1
        k = ky1
        k1 = k + 1
        while not (yi < ty[k1 - 1] or k == nk1y):
            k = k1
            k1 = k + 1
        num = (l - kx1) * nyy + k - ky
        nummer[im - 1] = index[num - 1]
        index[num - 1] = im

def _fprank(a, f, n: int, m: int, na: int, tol: float, c):
    # returns (sq, rank); a, f are modified like the C routine
    m1 = m - 1
    nl = 0
    sq = 0.0
    h = [0.0] * (m + 1)
    for i in range(1, n + 1):
        if a[i - 1] > tol:
            continue
        nl += 1
        if i == n:
            continue
        yi = f[i - 1]
        for j in range(1, m1 + 1):
            h[j - 1] = a[(i - 1) + j * na]
        h[m - 1] = 0.0
        for ii in range(i + 1, n + 1):
            i2 = min(n - ii, m1)
            piv = h[0]
            if piv == 0.0:
                if i2 == 0:
                    break
                for j in range(1, i2 + 1):
                    h[j - 1] = h[j]
                h[i2] = 0.0
                continue
            cs, sn, ww = fpgivs(piv, a[ii - 1])
            a[ii - 1] = ww
            yi, f[ii - 1] = _rot(cs, sn, yi, f[ii - 1])
            if i2 == 0:
                break
            for j in range(1, i2 + 1):
                idx = (ii - 1) + j * na
                h[j], a[idx] = _rot(cs, sn, h[j], a[idx])
                h[j - 1] = h[j]
            h[i2] = 0.0
        sq = sq + yi * yi
    rank = n - nl
    aa = np.zeros(n * m)
    ff = np.zeros(n)
    ii = 0
    for i in range(1, n + 1):
        if a[i - 1] <= tol:
            continue
        ii += 1
        ff[ii - 1] = f[i - 1]
        aa[ii - 1] = a[i - 1]
        jj = ii
        kk = 1
        j = i
        j1 = min(j - 1, m1)
        if j1 == 0:
            continue
        for k in range(1, j1 + 1):
            j -= 1
            if a[j - 1] <= tol:
                continue
            kk += 1
            jj -= 1
            aa[(jj - 1) + (kk - 1) * n] = a[(j - 1) + k * na]
    ii = 0
    for i in range(1, n + 1):
        if a[i - 1] > tol:
            ii += 1
            continue
        if ii == 0:
            continue
        jj = 1
        j = i
        j1 = min(j - 1, m1)
        for k in range(1, j1 + 1):
            j -= 1
            if a[j - 1] <= tol:
                continue
            h[jj - 1] = a[(j - 1) + k * na]
            jj += 1
        for kk in range(jj, m + 1):
            h[kk - 1] = 0.0
        jj = ii
        for i1 in range(1, ii + 1):
            j1 = min(jj - 1, m1)
            piv = h[0]
            if piv != 0.0:
                cs, sn, ww = fpgivs(piv, aa[jj - 1])
                aa[jj - 1] = ww
                if j1 == 0:
                    break
                kk = jj
                for j2 in range(1, j1 + 1):
                    kk -= 1
                    idx = (kk - 1) + j2 * n
                    h[j2], aa[idx] = _rot(cs, sn, h[j2], aa[idx])
                    h[j2 - 1] = h[j2]
            else:
                if j1 == 0:
                    break
                for j2 in range(1, j1 + 1):
                    h[j2 - 1] = h[j2]
            jj -= 1
            h[j1] = 0.0
    ff[rank - 1] = ff[rank - 1] / aa[rank - 1]
    i = rank - 1
    if i != 0:
        for j in range(2, rank + 1):
            store = ff[i - 1]
            i1 = min(j - 1, m1)
            k = i
            for q in range(1, i1 + 1):
                k += 1
                store -= ff[k - 1] * aa[(i - 1) + q * n]
            ff[i - 1] = store / aa[i - 1]
            i -= 1
    ff[0] = ff[0] / aa[0]
    if rank > 1:
        for j in range(2, rank + 1):
            store = ff[j - 1]
            i1 = min(j - 1, m1)
            k = j
            for q in range(1, i1 + 1):
                k -= 1
                store -= ff[k - 1] * aa[(k - 1) + q * n]
            ff[j - 1] = store / aa[j - 1]
    k = 0
    for i in range(1, n + 1):
        store = 0.0
        if a[i - 1] > tol:
            k += 1
        j1 = min(i, m)
        kk = k
        ij = i + 1
        for j in range(1, j1 + 1):
            ij -= 1
            if a[ij - 1] <= tol:
                continue
            store += a[(ij - 1) + (j - 1) * na] * ff[kk - 1]
            kk -= 1
        c[i - 1] = store
    stor3 = 0.0
    for i in range(1, n + 1):
        if a[i - 1] > tol:
            continue
        store = f[i - 1]
        i1 = min(n - i, m1)
        if i1 != 0:
            for j in range(1, i1 + 1):
                ij = i + j
                store -= c[ij - 1] * a[(i - 1) + j * na]
        stor3 += (a[i - 1] * c[i - 1]) * ((a[i - 1] * c[i - 1]) - store - store)
    sq += stor3
    return sq, rank

def surfit(iopt: int, xin, yin, z, w, xb: float, xe: float, yb: float, ye: float,
           kxx: int, kyy: int, s: float, nxest: int, nyest: int, eps_in: float,
           tx, ty, nx0: int, ny0: int):
    # returns (nx, ny, c, fp, ier); tx/ty (length nmax) updated in place.
    m = xin.shape[0]
    maxit = 20
    tol = 0.001
    x = xin.copy()
    y = yin.copy()
    kx1 = kxx + 1
    ky1 = kyy + 1
    if eps_in <= 0.0 or eps_in >= 1.0 or kxx <= 0 or kxx > 5 or kyy <= 0 or kyy > 5:
        return nx0, ny0, np.zeros(0), 0.0, 10
    if iopt < -1 or iopt > 1 or m < kx1 * ky1:
        return nx0, ny0, np.zeros(0), 0.0, 10
    nminx = 2 * kx1
    nminy = 2 * ky1
    nmax = max(nxest, nyest)
    if nxest < nminx or nyest < nminy:
        return nx0, ny0, np.zeros(0), 0.0, 10
    nxk = nxest - kx1
    nyk = nyest - ky1
    nc = nxk * nyk
    nmx = nxest - nminx + 1
    nmy = nyest - nminy + 1
    nrintest = nmx + nmy
    nregest = nmx * nmy
    ib1 = kxx * nyk + ky1
    jb1 = kyy * nxk + kx1
    ib3 = kx1 * nyk + 1
    if ib1 > jb1:
        ib1 = jb1
        ib3 = ky1 * nxk + 1
    if xb >= xe or yb >= ye:
        return nx0, ny0, np.zeros(0), 0.0, 10
    for i in range(m):
        if w[i] <= 0.0 or x[i] < xb or x[i] > xe or y[i] < yb or y[i] > ye:
            return nx0, ny0, np.zeros(0), 0.0, 10
    if iopt < 0:
        if nx0 < nminx or nx0 > nxest or ny0 < nminy or ny0 > nyest:
            return nx0, ny0, np.zeros(0), 0.0, 10
        nxk2 = nx0 - kx1
        tx[kx1 - 1] = xb
        tx[nxk2] = xe
        for i in range(kx1, nxk2 + 1):
            if tx[i] <= tx[i - 1]:
                return nx0, ny0, np.zeros(0), 0.0, 10
        nyk2 = ny0 - ky1
        ty[ky1 - 1] = yb
        ty[nyk2] = ye
        for i in range(ky1, nyk2 + 1):
            if ty[i] <= ty[i - 1]:
                return nx0, ny0, np.zeros(0), 0.0, 10
    elif s < 0.0:
        return nx0, ny0, np.zeros(0), 0.0, 10
    ier = 0
    # ---- fpsurf
    ncol = max(ib1, ib3) + 2
    a = np.zeros(nc * ncol)
    q = np.zeros(nc * ncol)
    f = np.zeros(nc)
    ff = np.zeros(nc)
    c = np.zeros(nc)
    h = [0.0] * (ncol + 2)
    km2 = max(kxx, kyy) + 2
    bx = np.zeros(nmax * km2)
    by = np.zeros(nmax * km2)
    km1 = max(kxx, kyy) + 1
    spx = np.zeros(m * km1)
    spy = np.zeros(m * km1)
    fpint = np.zeros(nrintest + 2)
    coord = np.zeros(nrintest + 2)
    index = [0] * (nregest + 2)
    nummer = [0] * m
    hx = [0.0] * 7
    hy = [0.0] * 7
    ichang = -1
    x0 = xb
    x1 = xe
    y0 = yb
    y1 = ye
    kx = kxx
    ky = kyy
    nxe = nxest
    nye = nyest
    eps = sqrt(eps_in)
    acc = 0.0
    fp = 0.0
    fp0 = 0.0
    fpms = 0.0
    nx = 0
    ny = 0
    if iopt < 0:
        nx = nx0
        ny = ny0
    else:
        acc = tol * s
        nx = nminx
        ny = nminy
        ier = -2
    ncof = 0
    nk1x = 0
    nk1y = 0
    nxx = 0
    nyy = 0
    nrint = 0
    nreg = 0
    iband = 0
    iband1 = 0
    rank = 0
    finished = False
    for iter in range(1, m + 1):
        l = nx
        for i in range(1, kx1 + 1):
            tx[i - 1] = x0
            tx[l - 1] = x1
            l -= 1
        l = ny
        for i in range(1, ky1 + 1):
            ty[i - 1] = y0
            ty[l - 1] = y1
            l -= 1
        nxx = nx - 2 * kx1 + 1
        nyy = ny - 2 * ky1 + 1
        nrint = nxx + nyy
        nreg = nxx * nyy
        iband1 = kx * (ny - ky1) + ky
        l = ky * (nx - kx1) + kx
        if iband1 > l:
            iband1 = l
            ichang = -ichang
            for i in range(m):
                t = x[i]
                x[i] = y[i]
                y[i] = t
            t = x0
            x0 = y0
            y0 = t
            t = x1
            x1 = y1
            y1 = t
            n = min(nx, ny)
            for i in range(n):
                t = tx[i]
                tx[i] = ty[i]
                ty[i] = t
            if nx < ny:
                for i in range(n, ny):
                    tx[i] = ty[i]
            elif nx > ny:
                for i in range(n, nx):
                    ty[i] = tx[i]
            l = nx
            nx = ny
            ny = l
            l = nxe
            nxe = nye
            nye = l
            l = nxx
            nxx = nyy
            nyy = l
            l = kx
            kx = ky
            ky = l
            kx1 = kx + 1
            ky1 = ky + 1
        iband = iband1 + 1
        _fporde(x, y, m, kx, ky, tx, nx, ty, ny, nummer, index, nreg)
        nk1x = nx - kx1
        nk1y = ny - ky1
        ncof = nk1x * nk1y
        for i in range(1, ncof + 1):
            f[i - 1] = 0.0
            for j in range(1, iband + 1):
                a[(i - 1) + (j - 1) * nc] = 0.0
        fp = 0.0
        for num in range(1, nreg + 1):
            num1 = num - 1
            lx = num1 // nyy
            l1 = lx + kx1
            ly = num1 - lx * nyy
            l2 = ly + ky1
            jrot = lx * nk1y + ly
            inn = index[num - 1]
            while inn != 0:
                wi = w[inn - 1]
                zi = z[inn - 1] * wi
                fpbspl(tx, kx, x[inn - 1], l1, hx)
                fpbspl(ty, ky, y[inn - 1], l2, hy)
                for i in range(1, kx1 + 1):
                    spx[(inn - 1) + (i - 1) * m] = hx[i - 1]
                for i in range(1, ky1 + 1):
                    spy[(inn - 1) + (i - 1) * m] = hy[i - 1]
                for i in range(1, iband + 1):
                    h[i - 1] = 0.0
                i1 = 0
                for i in range(1, kx1 + 1):
                    hxi = hx[i - 1]
                    j1 = i1
                    for j in range(1, ky1 + 1):
                        j1 += 1
                        h[j1 - 1] = hxi * hy[j - 1] * wi
                    i1 = i1 + nk1y
                irot = jrot
                for i in range(1, iband + 1):
                    irot += 1
                    piv = h[i - 1]
                    if piv == 0.0:
                        continue
                    cs, sn, ww = fpgivs(piv, a[irot - 1])
                    a[irot - 1] = ww
                    zi, f[irot - 1] = _rot(cs, sn, zi, f[irot - 1])
                    if i == iband:
                        break
                    i2 = 1
                    for j in range(i + 1, iband + 1):
                        i2 += 1
                        idx = (irot - 1) + (i2 - 1) * nc
                        h[j - 1], a[idx] = _rot(cs, sn, h[j - 1], a[idx])
                fp = fp + zi * zi
                inn = nummer[inn - 1]
        dmax = 0.0
        for i in range(1, ncof + 1):
            if a[i - 1] > dmax:
                dmax = a[i - 1]
        sigma = eps * dmax
        deficient = False
        for i in range(1, ncof + 1):
            if a[i - 1] <= sigma:
                deficient = True
                break
        if deficient:
            for i in range(1, ncof + 1):
                ff[i - 1] = f[i - 1]
                for j in range(1, iband + 1):
                    q[(i - 1) + (j - 1) * nc] = a[(i - 1) + (j - 1) * nc]
            sq, rank = _fprank(q, ff, ncof, iband, nc, sigma, c)
            for i in range(1, ncof + 1):
                q[i - 1] = q[i - 1] / dmax
            fp = fp + sq
        else:
            fpback(a, f, ncof, iband, c, nc)
            rank = ncof
            for i in range(1, ncof + 1):
                q[i - 1] = a[i - 1] / dmax
        if ier == -2:
            fp0 = fp
        if iopt < 0:
            if ncof != rank:
                ier = -rank
            finished = True
            break
        fpms = fp - s
        if fabs(fpms) <= acc:
            if fp <= 0.0:
                ier = -1
                fp = 0.0
            if ncof != rank:
                ier = -rank
            finished = True
            break
        if fpms < 0.0:
            break
        if ncof > m:
            ier = 4
            finished = True
            break
        ier = 0
        for i in range(1, nrint + 1):
            fpint[i - 1] = 0.0
            coord[i - 1] = 0.0
        for num in range(1, nreg + 1):
            num1 = num - 1
            lx = num1 // nyy
            l1 = lx + 1
            ly = num1 - lx * nyy
            l2 = ly + 1 + nxx
            jrot = lx * nk1y + ly
            inn = index[num - 1]
            while inn != 0:
                store = 0.0
                i1 = jrot
                for i in range(1, kx1 + 1):
                    hxi = spx[(inn - 1) + (i - 1) * m]
                    j1 = i1
                    for j in range(1, ky1 + 1):
                        j1 += 1
                        store = store + hxi * spy[(inn - 1) + (j - 1) * m] * c[j1 - 1]
                    i1 = i1 + nk1y
                store = pow(w[inn - 1] * (z[inn - 1] - store), 2.0)
                fpint[l1 - 1] = fpint[l1 - 1] + store
                coord[l1 - 1] = coord[l1 - 1] + store * x[inn - 1]
                fpint[l2 - 1] = fpint[l2 - 1] + store
                coord[l2 - 1] = coord[l2 - 1] + store * y[inn - 1]
                inn = nummer[inn - 1]
        stop = False
        while True:
            l = 0
            fpmax = 0.0
            l1 = 1
            l2 = nrint
            if nx == nxe:
                l1 = nxx + 1
            if ny == nye:
                l2 = nxx
            if l1 > l2:
                ier = 1
                stop = True
                break
            for i in range(l1, l2 + 1):
                if fpmax < fpint[i - 1]:
                    l = i
                    fpmax = fpint[i - 1]
            if l == 0:
                ier = 5
                stop = True
                break
            arg = coord[l - 1] / fpint[l - 1]
            if l <= nxx:
                jxy = l + kx1
                fpint[l - 1] = 0.0
                fac1 = tx[jxy - 1] - arg
                fac2 = arg - tx[jxy - 2]
                if fac1 > 10.0 * fac2 or fac2 > 10.0 * fac1:
                    continue
                j = nx
                for i in range(jxy, nx + 1):
                    tx[j] = tx[j - 1]
                    j -= 1
                tx[jxy - 1] = arg
                nx += 1
                break
            else:
                jxy = l + ky1 - nxx
                fpint[l - 1] = 0.0
                fac1 = ty[jxy - 1] - arg
                fac2 = arg - ty[jxy - 2]
                if fac1 > 10.0 * fac2 or fac2 > 10.0 * fac1:
                    continue
                j = ny
                for i in range(jxy, ny + 1):
                    ty[j] = ty[j - 1]
                    j -= 1
                ty[jxy - 1] = arg
                ny += 1
                break
        if stop:
            finished = True
            break
    if not finished and ier != -2:
        kx2 = kx1 + 1
        if nk1x != kx1:
            fpdisc(tx, nx, kx2, bx, nmax)
        ky2 = ky1 + 1
        if nk1y != ky1:
            fpdisc(ty, ny, ky2, by, nmax)
        p1 = 0.0
        f1 = fp0 - s
        p3 = -1.0
        f3 = fpms
        p = 0.0
        for i in range(1, ncof + 1):
            p = p + a[i - 1]
        p = float(ncof) / p
        iband3 = kx1 * nk1y
        iband4 = iband3 + 1
        ich1 = 0
        ich3 = 0
        for iter in range(1, maxit + 1):
            pinv = 1.0 / p
            for i in range(1, ncof + 1):
                ff[i - 1] = f[i - 1]
                for j in range(1, iband + 1):
                    q[(i - 1) + (j - 1) * nc] = a[(i - 1) + (j - 1) * nc]
                for j in range(iband + 1, iband4 + 1):
                    q[(i - 1) + (j - 1) * nc] = 0.0
            if nk1y != ky1:
                for i in range(ky2, nk1y + 1):
                    ii = i - ky1
                    for j in range(1, nk1x + 1):
                        for l in range(iband):
                            h[l] = 0.0
                        for l in range(ky2):
                            h[l] = by[(ii - 1) + l * nmax] * pinv
                        zi = 0.0
                        jrot = (j - 1) * nk1y + ii
                        for irot in range(jrot, ncof + 1):
                            piv = h[0]
                            i2 = min(iband1, ncof - irot)
                            if piv != 0.0:
                                cs, sn, ww = fpgivs(piv, q[irot - 1])
                                q[irot - 1] = ww
                                zi, ff[irot - 1] = _rot(cs, sn, zi, ff[irot - 1])
                                if i2 == 0:
                                    break
                                for l in range(1, i2 + 1):
                                    idx = (irot - 1) + l * nc
                                    h[l], q[idx] = _rot(cs, sn, h[l], q[idx])
                            else:
                                if i2 <= 0:
                                    break
                            for l in range(i2):
                                h[l] = h[l + 1]
                            h[i2] = 0.0
            if nk1x != kx1:
                for i in range(kx2, nk1x + 1):
                    ii = i - kx1
                    for j in range(1, nk1y + 1):
                        for l in range(iband4):
                            h[l] = 0.0
                        j1 = 1
                        for l in range(1, kx2 + 1):
                            h[j1 - 1] = bx[(ii - 1) + (l - 1) * nmax] * pinv
                            j1 = j1 + nk1y
                        zi = 0.0
                        jrot = (i - kx2) * nk1y + j
                        for irot in range(jrot, ncof + 1):
                            piv = h[0]
                            i2 = min(iband3, ncof - irot)
                            if piv != 0.0:
                                cs, sn, ww = fpgivs(piv, q[irot - 1])
                                q[irot - 1] = ww
                                zi, ff[irot - 1] = _rot(cs, sn, zi, ff[irot - 1])
                                if i2 == 0:
                                    break
                                for l in range(1, i2 + 1):
                                    idx = (irot - 1) + l * nc
                                    h[l], q[idx] = _rot(cs, sn, h[l], q[idx])
                            else:
                                if i2 <= 0:
                                    break
                            for l in range(i2):
                                h[l] = h[l + 1]
                            h[i2] = 0.0
            dmax = 0.0
            for i in range(1, ncof + 1):
                if q[i - 1] > dmax:
                    dmax = q[i - 1]
            sigma = eps * dmax
            deficient = False
            for i in range(1, ncof + 1):
                if q[i - 1] <= sigma:
                    deficient = True
                    break
            if deficient:
                sq, rank = _fprank(q, ff, ncof, iband4, nc, sigma, c)
            else:
                fpback(q, ff, ncof, iband4, c, nc)
                rank = ncof
            for i in range(ncof):
                q[i] = q[i] / dmax
            fp = 0.0
            for num in range(1, nreg + 1):
                num1 = num - 1
                lx = num1 // nyy
                ly = num1 - lx * nyy
                jrot = lx * nk1y + ly
                inn = index[num - 1]
                while inn != 0:
                    store = 0.0
                    i1 = jrot
                    for i in range(1, kx1 + 1):
                        hxi = spx[(inn - 1) + (i - 1) * m]
                        j1 = i1
                        for j in range(1, ky1 + 1):
                            j1 += 1
                            store = store + hxi * spy[(inn - 1) + (j - 1) * m] * c[j1 - 1]
                        i1 = i1 + nk1y
                    fp = fp + pow(w[inn - 1] * (z[inn - 1] - store), 2.0)
                    inn = nummer[inn - 1]
            fpms = fp - s
            if fabs(fpms) <= acc:
                if ncof != rank:
                    ier = -rank
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
                    p = p * 0.04
                    if p <= p1:
                        p = p1 * 0.9 + p2 * 0.1
                    continue
            if ich1 == 0:
                if (f1 - f2) > acc:
                    if f2 > 0.0:
                        ich1 = 1
                else:
                    p1 = p2
                    f1 = f2
                    p = p / 0.04
                    if p3 >= 0.0 and p >= p3:
                        p = p2 * 0.1 + p3 * 0.9
                    continue
            if f2 >= f1 or f2 <= f3:
                ier = 2
                break
            p, p1, f1, p3, f3 = fprati(p1, f1, p2, f2, p3, f3)
    # interpolating_solution: undo the x/y swap
    if ichang >= 0:
        l1 = 1
        for i in range(1, nk1x + 1):
            l2 = i
            for j in range(1, nk1y + 1):
                f[l2 - 1] = c[l1 - 1]
                l1 += 1
                l2 = l2 + nk1x
        for i in range(1, ncof + 1):
            c[i - 1] = f[i - 1]
        n = min(nx, ny)
        for i in range(n):
            t = tx[i]
            tx[i] = ty[i]
            ty[i] = t
        if nx < ny:
            for i in range(n, ny):
                tx[i] = ty[i]
        elif nx > ny:
            for i in range(n, nx):
                ty[i] = tx[i]
        l = nx
        nx = ny
        ny = l
    return nx, ny, c, fp, ier
