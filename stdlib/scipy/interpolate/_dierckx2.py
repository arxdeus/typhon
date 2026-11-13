# Ported from scipy 1.18.1 (BSD-3-Clause), FITPACK by P. Dierckx
# (scipy/interpolate/src/dfitpack.c): regrid/fpregr/fpgrre (rectangular-grid
# smoothing), fpbisp/bispev, parder, pardtc, dblint/fpintb.

import numpy as np
from math import sqrt, fabs, pow
from _dierckx import fpbspl, fpgivs, fpback, fpchec, fpdisc, fpknot, fprati

def _rot(cs: float, sn: float, a: float, b: float):
    return (cs * a - sn * b, cs * b + sn * a)

class _GrreWork:
    # persistent FITPACK state for fpgrre (ifsx/ifsy/ifbx/ifby reuse)
    spx: np.ndarray[float, 1]
    spy: np.ndarray[float, 1]
    nrx: List[int]
    nry: List[int]
    bx: np.ndarray[float, 1]
    by: np.ndarray[float, 1]
    ifsx: int
    ifsy: int

    def __init__(self, mx: int, my: int, kx: int, ky: int, nxest: int, nyest: int):
        self.spx = np.zeros(mx * (kx + 1))
        self.spy = np.zeros(my * (ky + 1))
        self.nrx = [0] * mx
        self.nry = [0] * my
        self.bx = np.zeros(nxest * (kx + 2))
        self.by = np.zeros(nyest * (ky + 2))
        self.ifsx = 0
        self.ifsy = 0

def _fpgrre(W: _GrreWork, x, mx: int, y, my: int, z, kx: int, ky: int, tx, nx: int,
            ty, ny: int, p: float, c, fpx, fpy) -> float:
    kx1 = kx + 1
    kx2 = kx1 + 1
    ky1 = ky + 1
    ky2 = ky1 + 1
    nk1x = nx - kx1
    nk1y = ny - ky1
    h = [0.0] * 7
    pinv = 0.0
    if p > 0.0:
        pinv = 1.0 / p
    spx = W.spx
    spy = W.spy
    nrx = W.nrx
    nry = W.nry
    if W.ifsx == 0:
        l = kx1
        l1 = kx2
        number = 0
        for it in range(1, mx + 1):
            arg = x[it - 1]
            while not (arg < tx[l1 - 1] or l == nk1x):
                l = l1
                l1 = l + 1
                number += 1
            fpbspl(tx, kx, arg, l, h)
            for i in range(1, kx1 + 1):
                spx[(it - 1) + (i - 1) * mx] = h[i - 1]
            nrx[it - 1] = number
        W.ifsx = 1
    if W.ifsy == 0:
        l = ky1
        l1 = ky2
        number = 0
        for it in range(1, my + 1):
            arg = y[it - 1]
            while not (arg < ty[l1 - 1] or l == nk1y):
                l = l1
                l1 = l + 1
                number += 1
            fpbspl(ty, ky, arg, l, h)
            for i in range(1, ky1 + 1):
                spy[(it - 1) + (i - 1) * my] = h[i - 1]
            nry[it - 1] = number
        W.ifsy = 1
    bx = W.bx
    by = W.by
    if p > 0.0:
        if nx != 2 * kx1:
            fpdisc(tx, nx, kx2, bx, nx)
        if ny != 2 * ky1:
            fpdisc(ty, ny, ky2, by, ny)
    q = np.zeros(my * nk1x)
    ax = np.zeros(nx * kx2)
    right = np.zeros(max(my, nk1x) + 1)
    l = 0
    nrold = 0
    ibandx = kx1
    for it in range(1, mx + 1):
        number = nrx[it - 1]
        while True:
            irot = 0
            if nrold == number:
                h[ibandx - 1] = 0.0
                for j in range(1, kx1 + 1):
                    h[j - 1] = spx[(it - 1) + (j - 1) * mx]
                for j in range(1, my + 1):
                    l += 1
                    right[j - 1] = z[l - 1]
                irot = number
            else:
                if p <= 0.0:
                    nrold += 1
                    continue
                ibandx = kx2
                n1 = nrold + 1
                for j in range(1, kx2 + 1):
                    h[j - 1] = bx[(n1 - 1) + (j - 1) * nx] * pinv
                for j in range(1, my + 1):
                    right[j - 1] = 0.0
                irot = nrold
            for i in range(1, ibandx + 1):
                irot += 1
                piv = h[i - 1]
                if piv == 0.0:
                    continue
                cs, sn, ww = fpgivs(piv, ax[irot - 1])
                ax[irot - 1] = ww
                iq = (irot - 1) * my
                for j in range(1, my + 1):
                    iq += 1
                    right[j - 1], q[iq - 1] = _rot(cs, sn, right[j - 1], q[iq - 1])
                if i == ibandx:
                    break
                i2 = 1
                for j in range(i + 1, ibandx + 1):
                    i2 += 1
                    idx = (irot - 1) + (i2 - 1) * nx
                    h[j - 1], ax[idx] = _rot(cs, sn, h[j - 1], ax[idx])
            if nrold == number:
                break
            nrold += 1
    ncof = nk1x * nk1y
    for i in range(ncof):
        c[i] = 0.0
    ay = np.zeros(ny * ky2)
    nrold = 0
    ibandy = ky1
    for it in range(1, my + 1):
        number = nry[it - 1]
        while True:
            irot = 0
            if nrold == number:
                h[ibandy - 1] = 0.0
                for j in range(1, ky1 + 1):
                    h[j - 1] = spy[(it - 1) + (j - 1) * my]
                l = it
                for j in range(1, nk1x + 1):
                    right[j - 1] = q[l - 1]
                    l += my
                irot = number
            else:
                if p <= 0.0:
                    nrold += 1
                    continue
                ibandy = ky2
                n1 = nrold + 1
                for j in range(1, ky2 + 1):
                    h[j - 1] = by[(n1 - 1) + (j - 1) * ny] * pinv
                for j in range(1, nk1x + 1):
                    right[j - 1] = 0.0
                irot = nrold
            for i in range(1, ibandy + 1):
                irot += 1
                piv = h[i - 1]
                if piv == 0.0:
                    continue
                cs, sn, ww = fpgivs(piv, ay[irot - 1])
                ay[irot - 1] = ww
                ic = irot
                for j in range(1, nk1x + 1):
                    right[j - 1], c[ic - 1] = _rot(cs, sn, right[j - 1], c[ic - 1])
                    ic += nk1y
                if i == ibandy:
                    break
                i2 = 1
                for j in range(i + 1, ibandy + 1):
                    i2 += 1
                    idx = (irot - 1) + (i2 - 1) * ny
                    h[j - 1], ay[idx] = _rot(cs, sn, h[j - 1], ay[idx])
            if nrold == number:
                break
            nrold += 1
    k = 1
    for i in range(1, nk1x + 1):
        cv = c[k - 1:]
        fpback(ay, cv, nk1y, ibandy, cv, ny)
        k += nk1y
    k = 0
    for j in range(1, nk1y + 1):
        k += 1
        l = k
        for i in range(1, nk1x + 1):
            right[i - 1] = c[l - 1]
            l += nk1y
        fpback(ax, right, nk1x, ibandx, right, nx)
        l = k
        for i in range(1, nk1x + 1):
            c[l - 1] = right[i - 1]
            l += nk1y
    fp = 0.0
    for i in range(nx):
        fpx[i] = 0.0
    for i in range(ny):
        fpy[i] = 0.0
    iz = 0
    nroldx = 0
    for i1 in range(1, mx + 1):
        numx = nrx[i1 - 1]
        numx1 = numx + 1
        nroldy = 0
        for i2 in range(1, my + 1):
            numy = nry[i2 - 1]
            numy1 = numy + 1
            iz += 1
            term = 0.0
            k1 = numx * nk1y + numy
            for l1 in range(1, kx1 + 1):
                k2 = k1
                fac = spx[(i1 - 1) + (l1 - 1) * mx]
                for l2 in range(1, ky1 + 1):
                    k2 += 1
                    term = term + fac * spy[(i2 - 1) + (l2 - 1) * my] * c[k2 - 1]
                k1 += nk1y
            term = pow(z[iz - 1] - term, 2.0)
            fp = fp + term
            fpx[numx1 - 1] = fpx[numx1 - 1] + term
            fpy[numy1 - 1] = fpy[numy1 - 1] + term
            fac = term * 0.5
            if numy != nroldy:
                fpy[numy1 - 1] = fpy[numy1 - 1] - fac
                fpy[numy - 1] = fpy[numy - 1] + fac
            nroldy = numy
            if numx != nroldx:
                fpx[numx1 - 1] = fpx[numx1 - 1] - fac
                fpx[numx - 1] = fpx[numx - 1] + fac
        nroldx = numx
    return fp

def _interp_knots1(x, t, m: int, k: int):
    k1 = k + 1
    mk1 = m - k1
    if mk1 == 0:
        return
    k3 = k // 2
    i = k1 + 1
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

def regrid_smth(x, y, z, xb: float, xe: float, yb: float, ye: float, kx: int, ky: int,
                s: float, maxit: int):
    # returns (tx, ty, c, fp, ier); iopt = 0, nxest = mx+kx+1, nyest = my+ky+1
    mx = x.shape[0]
    my = y.shape[0]
    nxest = mx + kx + 1
    nyest = my + ky + 1
    tx = np.zeros(nxest)
    ty = np.zeros(nyest)
    nc = (nxest - kx - 1) * (nyest - ky - 1)
    c = np.zeros(nc)
    tol = 0.001
    kx1 = kx + 1
    ky1 = ky + 1
    kx2 = kx1 + 1
    ky2 = ky1 + 1
    nminx = 2 * kx1
    nminy = 2 * ky1
    if kx <= 0 or kx > 5 or ky <= 0 or ky > 5 or mx < kx1 or my < ky1:
        return tx[:0], ty[:0], c[:0], 0.0, 10
    if xb > x[0] or xe < x[mx - 1] or yb > y[0] or ye < y[my - 1]:
        return tx[:0], ty[:0], c[:0], 0.0, 10
    for i in range(1, mx):
        if x[i - 1] >= x[i]:
            return tx[:0], ty[:0], c[:0], 0.0, 10
    for i in range(1, my):
        if y[i - 1] >= y[i]:
            return tx[:0], ty[:0], c[:0], 0.0, 10
    if s < 0.0:
        return tx[:0], ty[:0], c[:0], 0.0, 10
    ier = 0
    fpintx = np.zeros(nxest)
    fpinty = np.zeros(nyest)
    nrdatx = [0] * nxest
    nrdaty = [0] * nyest
    W = _GrreWork(mx, my, kx, ky, nxest, nyest)
    acc = tol * s
    nmaxx = mx + kx1
    nmaxy = my + ky1
    nxe = min(nxest, nmaxx)
    nye = min(nyest, nmaxy)
    nx = 0
    ny = 0
    fp = 0.0
    fp0 = 0.0
    fpold = 0.0
    reducx = 0.0
    reducy = 0.0
    lastdi = 0
    nplusx = 0
    nplusy = 0
    if s > 0.0:
        nx = nminx
        ny = nminy
        nrdatx[0] = mx - 2
        nrdaty[0] = my - 2
    else:
        nx = nmaxx
        ny = nmaxy
        _interp_knots1(x, tx, mx, kx)
        _interp_knots1(y, ty, my, ky)
    p = -1.0
    done_part1 = False
    while not done_part1:
        done_part1 = True
        mpm = mx + my
        restart = False
        for iter in range(1, mpm + 1):
            if nx == nminx and ny == nminy:
                ier = -2
            nrintx = nx - nminx + 1
            nrinty = ny - nminy + 1
            i = nx
            for j in range(1, kx1 + 1):
                tx[j - 1] = xb
                tx[i - 1] = xe
                i -= 1
            i = ny
            for j in range(1, ky1 + 1):
                ty[j - 1] = yb
                ty[i - 1] = ye
                i -= 1
            W.ifsx = 0
            W.ifsy = 0
            fp = _fpgrre(W, x, mx, y, my, z, kx, ky, tx, nx, ty, ny, p, c, fpintx, fpinty)
            if ier == -2:
                fp0 = fp
            fpms = fp - s
            if fabs(fpms) < acc:
                return tx[:nx].copy(), ty[:ny].copy(), c[:(nx - kx1) * (ny - ky1)].copy(), fp, ier
            if fpms < 0.0:
                break
            if nx == nmaxx and ny == nmaxy:
                return tx[:nx].copy(), ty[:ny].copy(), c[:(nx - kx1) * (ny - ky1)].copy(), 0.0, -1
            if nx == nxe and ny == nye:
                return tx[:nx].copy(), ty[:ny].copy(), c[:(nx - kx1) * (ny - ky1)].copy(), fp, 1
            ier = 0
            if lastdi < 0:
                reducx = fpold - fp
            elif lastdi > 0:
                reducy = fpold - fp
            fpold = fp
            nplx = 1
            if nx != nminx:
                npl1 = nplusx * 2
                rn = float(nplusx)
                if reducx > acc:
                    val = rn * fpms / reducx
                    npl1 = 2147483647 if val > 2147483647.0 else int(val)
                temp1 = nplusx * 2
                temp2 = max(nplusx // 2, 1)
                nplx = npl1 if npl1 > temp2 else temp2
                if nplx > temp1:
                    nplx = temp1
            nply = 1
            if ny != nminy:
                npl1 = nplusy * 2
                rn = float(nplusy)
                if reducy > acc:
                    val = rn * fpms / reducy
                    npl1 = 2147483647 if val > 2147483647.0 else int(val)
                temp1 = nplusy * 2
                temp2 = max(nplusy // 2, 1)
                nply = npl1 if npl1 > temp2 else temp2
                if nply > temp1:
                    nply = temp1
            # choose direction (L210: x, L230: y)
            dox = False
            if nplx < nply:
                dox = True
            elif nplx == nply:
                dox = not (lastdi < 0)
            else:
                dox = False
            if dox and nx == nxe:
                dox = False
            if (not dox) and ny == nye:
                dox = True
            if dox:
                lastdi = -1
                nplusx = nplx
                for l in range(1, nplx + 1):
                    nx, nrintx = fpknot(x, tx, nx, fpintx, nrdatx, nrintx, 1)
                    if nx == nmaxx:
                        _interp_knots1(x, tx, mx, kx)
                        # dfitpack.c L10: for odd kx control falls through to
                        # L60 and also resets the y interpolation knots
                        if (kx // 2) * 2 != kx or mx - kx1 == 0:
                            _interp_knots1(y, ty, my, ky)
                        restart = True
                        break
                    if nx == nxe:
                        break
            else:
                lastdi = 1
                nplusy = nply
                for l in range(1, nply + 1):
                    ny, nrinty = fpknot(y, ty, ny, fpinty, nrdaty, nrinty, 1)
                    if ny == nmaxy:
                        _interp_knots1(y, ty, my, ky)
                        restart = True
                        break
                    if ny == nye:
                        break
            if restart:
                break
        if restart:
            done_part1 = False
            continue
    if ier == -2:
        return tx[:nx].copy(), ty[:ny].copy(), c[:(nx - kx1) * (ny - ky1)].copy(), fp, ier
    p1 = 0.0
    f1 = fp0 - s
    p3 = -1.0
    fpms = fp - s
    f3 = fpms
    p = 1.0
    ich1 = 0
    ich3 = 0
    for iter in range(1, maxit + 1):
        W.ifsx = 0
        W.ifsy = 0
        fp = _fpgrre(W, x, mx, y, my, z, kx, ky, tx, nx, ty, ny, p, c, fpintx, fpinty)
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
                if p3 >= 0.0:
                    if p >= p3:
                        p = p2 * 0.1 + p3 * 0.9
                continue
        if f2 >= f1 or f2 <= f3:
            ier = 2
            break
        p, p1, f1, p3, f3 = fprati(p1, f1, p2, f2, p3, f3)
    return tx[:nx].copy(), ty[:ny].copy(), c[:(nx - kx1) * (ny - ky1)].copy(), fp, ier

def fpbisp(tx, nx: int, ty, ny: int, c, kx: int, ky: int, x, mx: int, y, my: int, z):
    kx1 = kx + 1
    nkx1 = nx - kx1
    tb = tx[kx1 - 1]
    te = tx[nkx1]
    l = kx1
    l1 = l + 1
    h = [0.0] * 7
    wx = np.empty(mx * kx1)
    lx = [0] * mx
    for i in range(1, mx + 1):
        arg = x[i - 1]
        if arg < tb:
            arg = tb
        if arg > te:
            arg = te
        while not (arg < tx[l1 - 1] or l == nkx1):
            l = l1
            l1 = l + 1
        fpbspl(tx, kx, arg, l, h)
        lx[i - 1] = l - kx1
        for j in range(1, kx1 + 1):
            wx[(i - 1) + (j - 1) * mx] = h[j - 1]
    ky1 = ky + 1
    nky1 = ny - ky1
    tb = ty[ky1 - 1]
    te = ty[nky1]
    l = ky1
    l1 = l + 1
    wy = np.empty(my * ky1)
    ly = [0] * my
    for i in range(1, my + 1):
        arg = y[i - 1]
        if arg < tb:
            arg = tb
        if arg > te:
            arg = te
        while not (arg < ty[l1 - 1] or l == nky1):
            l = l1
            l1 = l + 1
        fpbspl(ty, ky, arg, l, h)
        ly[i - 1] = l - ky1
        for j in range(1, ky1 + 1):
            wy[(i - 1) + (j - 1) * my] = h[j - 1]
    m = 0
    for i in range(1, mx + 1):
        l = lx[i - 1] * nky1
        for i1 in range(1, kx1 + 1):
            h[i1 - 1] = wx[(i - 1) + (i1 - 1) * mx]
        for j in range(1, my + 1):
            l1 = l + ly[j - 1]
            sp = 0.0
            for i1 in range(1, kx1 + 1):
                l2 = l1
                for j1 in range(1, ky1 + 1):
                    l2 += 1
                    sp = sp + c[l2 - 1] * h[i1 - 1] * wy[(j - 1) + (j1 - 1) * my]
                l1 += nky1
            m += 1
            z[m - 1] = sp

def pardtc(tx, nx: int, ty, ny: int, c, kx: int, ky: int, nux: int, nuy: int):
    kx1 = kx + 1
    ky1 = ky + 1
    nkx1 = nx - kx1
    nky1 = ny - ky1
    nc = nkx1 * nky1
    newc = np.zeros(max(nc, 1))
    for i in range(nc):
        newc[i] = c[i]
    nxx = nkx1
    nyy = nky1
    nkx = kx
    nky = ky
    if nux != 0:
        lx = 1
        for j in range(1, nux + 1):
            ak = float(nkx)
            nxx -= 1
            l1 = lx
            m0 = 1
            for i in range(1, nxx + 1):
                l1 += 1
                l2 = l1 + nkx
                fac = tx[l2 - 1] - tx[l1 - 1]
                if fac <= 0.0:
                    continue
                for m in range(1, nyy + 1):
                    m1 = m0 + nyy
                    newc[m0 - 1] = (newc[m1 - 1] - newc[m0 - 1]) * ak / fac
                    m0 += 1
            lx += 1
            nkx -= 1
    if nuy != 0:
        ly = 1
        for j in range(1, nuy + 1):
            ak = float(nky)
            nyy -= 1
            l1 = ly
            for i in range(1, nyy + 1):
                l1 += 1
                l2 = l1 + nky
                fac = ty[l2 - 1] - ty[l1 - 1]
                if fac <= 0.0:
                    continue
                m0 = i
                for m in range(1, nxx + 1):
                    m1 = m0 + 1
                    newc[m0 - 1] = (newc[m1 - 1] - newc[m0 - 1]) * ak / fac
                    m0 += nky1
            ly += 1
            nky -= 1
        m0 = nyy
        m1 = nky1
        for m in range(2, nxx + 1):
            for i in range(1, nyy + 1):
                m0 += 1
                m1 += 1
                newc[m0 - 1] = newc[m1 - 1]
            m1 += nuy
    return newc, nxx, nyy

def fpintb(t, n: int, nk1: int, x: float, y: float):
    bint = np.zeros(nk1)
    k1 = n - nk1
    ak = float(k1)
    k = k1 - 1
    a = x
    b = y
    mmin = 0
    if a >= b:
        if a == b:
            return bint
        a = y
        b = x
        mmin = 1
    if a < t[k1 - 1]:
        a = t[k1 - 1]
    if b > t[nk1]:
        b = t[nk1]
    if a > b:
        return bint
    aint = [0.0] * 7
    h = [0.0] * 7
    h1 = [0.0] * 7
    l = k1
    l0 = l + 1
    arg = a
    ia = 0
    for it in range(1, 3):
        while not (arg < t[l0 - 1] or l == nk1):
            l = l0
            l0 = l + 1
        for j in range(1, k1 + 1):
            aint[j - 1] = 0.0
        aint[0] = (arg - t[l - 1]) / (t[l] - t[l - 1])
        h1[0] = 1.0
        for j in range(1, k + 1):
            h[0] = 0.0
            for i in range(1, j + 1):
                li = l + i
                lj = li - j
                f = h1[i - 1] / (t[li - 1] - t[lj - 1])
                h[i - 1] = h[i - 1] + f * (t[li - 1] - arg)
                h[i] = f * (arg - t[lj - 1])
            j1 = j + 1
            for i in range(1, j1 + 1):
                li = l + i
                lj = li - j1
                aint[i - 1] = aint[i - 1] + h[i - 1] * (arg - t[lj - 1]) / (t[li - 1] - t[lj - 1])
                h1[i - 1] = h[i - 1]
        if it == 2:
            break
        lk = l - k
        ia = lk
        for i in range(1, k1 + 1):
            bint[lk - 1] = -aint[i - 1]
            lk += 1
        arg = b
    lk = l - k
    ib = lk - 1
    for i in range(1, k1 + 1):
        bint[lk - 1] += aint[i - 1]
        lk += 1
    if ib >= ia:
        for i in range(ia, ib + 1):
            bint[i - 1] += 1.0
    f = 1.0 / ak
    for i in range(1, nk1 + 1):
        j = i + k1
        bint[i - 1] = bint[i - 1] * (t[j - 1] - t[i - 1]) * f
    if mmin != 0:
        for i in range(nk1):
            bint[i] = -bint[i]
    return bint

def dblint(tx, nx: int, ty, ny: int, c, kx: int, ky: int, xb: float, xe: float, yb: float, ye: float) -> float:
    nkx1 = nx - kx - 1
    nky1 = ny - ky - 1
    wx = fpintb(tx, nx, nkx1, xb, xe)
    wy = fpintb(ty, ny, nky1, yb, ye)
    result = 0.0
    for i in range(1, nkx1 + 1):
        res = wx[i - 1]
        if res == 0.0:
            continue
        m = (i - 1) * nky1
        for j in range(1, nky1 + 1):
            m += 1
            result = result + res * wy[j - 1] * c[m - 1]
    return result
