# Ported from scipy 1.18.1 (BSD-3-Clause)
#
# Weighted QUADPACK routines used by scipy.integrate.quad(weight=...):
# dqawce (cauchy), dqawse (alg*), dqawoe (cos/sin, finite) and dqawfe
# (cos/sin, semi-infinite Fourier) with dqc25c, dqc25s, dqc25f, dqk15w,
# dqcheb, dqmomo and a 0-based dqpsrt. Translated from scipy's C version
# `scipy/integrate/__quadpack.c` (0-based indexing as there).

import math
from _quadpack import _qelg, _Eps, qagie

_EPMACH = 2.220446049250313e-16
_UFLOW = 2.2250738585072014e-308
_OFLOW = 1.7976931348623157e+308

_X11 = [0.991444861373810411144557526928563, 0.965925826289068286749743199728897,
        0.923879532511286756128183189396788, 0.866025403784438646763723170752936,
        0.793353340291235164579776961501299, 0.707106781186547524400844362104849,
        0.608761429008720639416097542898164, 0.500000000000000000000000000000000,
        0.382683432365089771728459984030399, 0.258819045102520762348898837624048,
        0.130526192220051591548406227895489]

_WG15W = [0.1294849661688697, 0.2797053914892767, 0.3818300505051189, 0.4179591836734694]
_XGK15W = [0.9914553711208126, 0.9491079123427585, 0.8648644233597691, 0.7415311855993944,
           0.5860872354676911, 0.4058451513773972, 0.2077849550078985, 0.0]
_WGK15W = [0.2293532201052922e-01, 0.6309209262997855e-01, 0.1047900103222502,
           0.1406532597155259, 0.1690047266392679, 0.1903505780647854,
           0.2044329400752989, 0.2094821410847278]

class _WList:
    alist: List[float]
    blist: List[float]
    rlist: List[float]
    elist: List[float]
    iord: List[int]
    level: List[int]
    nnlog: List[int]

    def __init__(self, limit: int):
        m = max(limit, 2) + 2
        self.alist = [0.0] * m
        self.blist = [0.0] * m
        self.rlist = [0.0] * m
        self.elist = [0.0] * m
        self.iord = [0] * m
        self.level = [0] * m
        self.nnlog = [0] * m

def _qpsrt0(limit: int, last: int, maxerr: int, elist: List[float], iord: List[int], nrmax: int):
    # returns (maxerr, ermax, nrmax)
    if last <= 2:
        iord[0] = 0
        iord[1] = 1
        maxerr = iord[nrmax]
        return maxerr, elist[maxerr], nrmax
    errmax = elist[maxerr]
    while nrmax > 0 and errmax > elist[iord[nrmax - 1]]:
        iord[nrmax] = iord[nrmax - 1]
        nrmax -= 1
    jupbn = (limit - last + 2) if last > (limit // 2 + 2) else (last - 1)
    errmin = elist[last - 1]
    jbnd = jupbn - 1
    ibeg = nrmax + 1
    found = -1
    if ibeg <= jbnd:
        for i in range(ibeg, jbnd + 1):
            if errmax >= elist[iord[i]]:
                found = i
                break
            iord[i - 1] = iord[i]
    if found < 0:
        iord[jbnd] = maxerr
        iord[jupbn] = last - 1
        maxerr = iord[nrmax]
        return maxerr, elist[maxerr], nrmax
    i = found
    iord[i - 1] = maxerr
    k = jbnd
    for j in range(i, jbnd + 1):
        if errmin < elist[iord[k]]:
            break
        iord[k + 1] = iord[k]
        k -= 1
    if errmin < elist[iord[k]]:
        iord[k + 1] = last - 1
    else:
        iord[i] = last - 1
    maxerr = iord[nrmax]
    return maxerr, elist[maxerr], nrmax

def _wgt(kind: int, x: float, p1: float, p2: float, p3: float, p4: float, integr: int) -> float:
    if kind == 0:     # dqwgtc
        return 1.0 / (x - p1)
    elif kind == 1:   # dqwgtf
        omx = p1 * x
        return math.cos(omx) if integr == 1 else math.sin(omx)
    xma = x - p1
    bmx = p2 - x
    ret = math.pow(xma, p3) * math.pow(bmx, p4)
    if integr == 2:
        ret = ret * math.log(xma)
    elif integr == 3:
        ret = ret * math.log(bmx)
    elif integr == 4:
        ret = ret * math.log(xma) * math.log(bmx)
    return ret

def _qk15w(f, kind: int, p1: float, p2: float, p3: float, p4: float, kp: int, a: float, b: float):
    # returns (result, abserr, resabs, resasc)
    fv1 = [0.0] * 7
    fv2 = [0.0] * 7
    centr = 0.5 * (b + a)
    hlgth = 0.5 * (b - a)
    dhlgth = abs(hlgth)
    fc = f(centr) * _wgt(kind, centr, p1, p2, p3, p4, kp)
    resg = _WG15W[3] * fc
    resk = _WGK15W[7] * fc
    resabs = abs(resk)
    for j in range(3):
        jtw = 2 * j + 1
        absc = hlgth * _XGK15W[jtw]
        absc1 = centr - absc
        absc2 = centr + absc
        fval1 = f(absc1) * _wgt(kind, absc1, p1, p2, p3, p4, kp)
        fval2 = f(absc2) * _wgt(kind, absc2, p1, p2, p3, p4, kp)
        fv1[jtw] = fval1
        fv2[jtw] = fval2
        fsum = fval1 + fval2
        resg = resg + _WG15W[j] * fsum
        resk = resk + _WGK15W[jtw] * fsum
        resabs = resabs + _WGK15W[jtw] * (abs(fval1) + abs(fval2))
    for j in range(4):
        jtwm1 = 2 * j
        absc = hlgth * _XGK15W[jtwm1]
        absc1 = centr - absc
        absc2 = centr + absc
        fval1 = f(absc1) * _wgt(kind, absc1, p1, p2, p3, p4, kp)
        fval2 = f(absc2) * _wgt(kind, absc2, p1, p2, p3, p4, kp)
        fv1[jtwm1] = fval1
        fv2[jtwm1] = fval2
        fsum = fval1 + fval2
        resk = resk + _WGK15W[jtwm1] * fsum
        resabs = resabs + _WGK15W[jtwm1] * (abs(fval1) + abs(fval2))
    reskh = resk * 0.5
    resasc = _WGK15W[7] * abs(fc - reskh)
    for j in range(7):
        resasc = resasc + _WGK15W[j] * (abs(fv1[j] - reskh) + abs(fv2[j] - reskh))
    result = resk * hlgth
    resabs = resabs * dhlgth
    resasc = resasc * dhlgth
    abserr = abs((resk - resg) * hlgth)
    if resasc != 0.0 and abserr != 0.0:
        abserr = resasc * min(1.0, math.pow(200.0 * abserr / resasc, 1.5))
    if resabs > _UFLOW / (50.0 * _EPMACH):
        abserr = max(_EPMACH * 50.0 * resabs, abserr)
    return result, abserr, resabs, resasc

def _qcheb(x: List[float], fval: List[float], cheb12: List[float], cheb24: List[float]):
    v = [0.0] * 12
    for i in range(12):
        j = 24 - i
        v[i] = fval[i] - fval[j]
        fval[i] = fval[i] + fval[j]
    alam1 = v[0] - v[8]
    alam2 = x[5] * (v[2] - v[6] - v[10])
    cheb12[3] = alam1 + alam2
    cheb12[9] = alam1 - alam2
    alam1 = v[1] - v[7] - v[9]
    alam2 = v[3] - v[5] - v[11]
    alam = x[2] * alam1 + x[8] * alam2
    cheb24[3] = cheb12[3] + alam
    cheb24[21] = cheb12[3] - alam
    alam = x[8] * alam1 - x[2] * alam2
    cheb24[9] = cheb12[9] + alam
    cheb24[15] = cheb12[9] - alam
    part1 = x[3] * v[4]
    part2 = x[7] * v[8]
    part3 = x[5] * v[6]
    alam1 = v[0] + part1 + part2
    alam2 = x[1] * v[2] + part3 + x[9] * v[10]
    cheb12[1] = alam1 + alam2
    cheb12[11] = alam1 - alam2
    alam = x[0] * v[1] + x[2] * v[3] + x[4] * v[5] + x[6] * v[7] + x[8] * v[9] + x[10] * v[11]
    cheb24[1] = cheb12[1] + alam
    cheb24[23] = cheb12[1] - alam
    alam = x[10] * v[1] - x[8] * v[3] + x[6] * v[5] - x[4] * v[7] + x[2] * v[9] - x[0] * v[11]
    cheb24[11] = cheb12[11] + alam
    cheb24[13] = cheb12[11] - alam
    alam1 = v[0] - part1 + part2
    alam2 = x[9] * v[2] - part3 + x[1] * v[10]
    cheb12[5] = alam1 + alam2
    cheb12[7] = alam1 - alam2
    alam = x[4] * v[1] - x[8] * v[3] - x[0] * v[5] - x[10] * v[7] + x[2] * v[9] + x[6] * v[11]
    cheb24[5] = cheb12[5] + alam
    cheb24[19] = cheb12[5] - alam
    alam = x[6] * v[1] - x[2] * v[3] - x[10] * v[5] + x[0] * v[7] - x[8] * v[9] - x[4] * v[11]
    cheb24[7] = cheb12[7] + alam
    cheb24[17] = cheb12[7] - alam
    for i in range(6):
        j = 12 - i
        v[i] = fval[i] - fval[j]
        fval[i] = fval[i] + fval[j]
    alam1 = v[0] + x[7] * v[4]
    alam2 = x[3] * v[2]
    cheb12[2] = alam1 + alam2
    cheb12[10] = alam1 - alam2
    cheb12[6] = v[0] - v[4]
    alam = x[1] * v[1] + x[5] * v[3] + x[9] * v[5]
    cheb24[2] = cheb12[2] + alam
    cheb24[22] = cheb12[2] - alam
    alam = x[5] * (v[1] - v[3] - v[5])
    cheb24[6] = cheb12[6] + alam
    cheb24[18] = cheb12[6] - alam
    alam = x[9] * v[1] - x[5] * v[3] + x[1] * v[5]
    cheb24[10] = cheb12[10] + alam
    cheb24[14] = cheb12[10] - alam
    for i in range(3):
        j = 6 - i
        v[i] = fval[i] - fval[j]
        fval[i] = fval[i] + fval[j]
    cheb12[4] = v[0] + x[7] * v[2]
    cheb12[8] = fval[0] - x[7] * fval[2]
    alam = x[3] * v[1]
    cheb24[4] = cheb12[4] + alam
    cheb24[20] = cheb12[4] - alam
    alam = x[7] * fval[1] - fval[3]
    cheb24[8] = cheb12[8] + alam
    cheb24[16] = cheb12[8] - alam
    cheb12[0] = fval[0] + fval[2]
    alam = fval[1] + fval[3]
    cheb24[0] = cheb12[0] + alam
    cheb24[24] = cheb12[0] - alam
    cheb12[12] = v[0] - v[2]
    cheb24[12] = cheb12[12]
    alam = 1.0 / 6.0
    for i in range(1, 12):
        cheb12[i] = cheb12[i] * alam
    alam = 0.5 * alam
    cheb12[0] = cheb12[0] * alam
    cheb12[12] = cheb12[12] * alam
    for i in range(1, 24):
        cheb24[i] = cheb24[i] * alam
    cheb24[0] = 0.5 * alam * cheb24[0]
    cheb24[24] = 0.5 * alam * cheb24[24]

def _fval25(f, centr: float, hlgth: float) -> List[float]:
    fval = [0.0] * 25
    fval[0] = 0.5 * f(centr + hlgth)
    fval[12] = f(centr)
    fval[24] = 0.5 * f(centr - hlgth)
    for i in range(1, 12):
        u = hlgth * _X11[i - 1]
        fval[i] = f(centr + u)
        fval[24 - i] = f(centr - u)
    return fval

def _qc25c(f, a: float, b: float, c: float, krul: int):
    # returns (result, abserr, krul, neval)
    cc = (2.0 * c - b - a) / (b - a)
    if abs(cc) >= 1.1:
        krul -= 1
        result, abserr, resabs, resasc = _qk15w(f, 0, c, 0.0, 0.0, 0.0, 0, a, b)
        if resasc == abserr:
            krul += 1
        return result, abserr, krul, 15
    hlgth = 0.5 * (b - a)
    centr = 0.5 * (b + a)
    fval = _fval25(f, centr, hlgth)
    cheb12 = [0.0] * 13
    cheb24 = [0.0] * 25
    _qcheb(_X11, fval, cheb12, cheb24)
    amom0 = math.log(abs((1.0 - cc) / (1.0 + cc)))
    amom1 = 2.0 + (cc * amom0)
    res12 = cheb12[0] * amom0 + cheb12[1] * amom1
    res24 = cheb24[0] * amom0 + cheb24[1] * amom1
    for k in range(2, 13):
        amom2 = 2.0 * cc * amom1 - amom0
        ak22 = float((k - 1) * (k - 1))
        if k % 2 == 1:
            amom2 = amom2 - (4.0 / (ak22 - 1.0))
        res12 = res12 + cheb12[k] * amom2
        res24 = res24 + cheb24[k] * amom2
        amom0 = amom1
        amom1 = amom2
    for k in range(13, 25):
        amom2 = 2.0 * cc * amom1 - amom0
        ak22 = float((k - 1) * (k - 1))
        if k % 2 == 1:
            amom2 = amom2 - (4.0 / (ak22 - 1.0))
        res24 = res24 + cheb24[k] * amom2
        amom0 = amom1
        amom1 = amom2
    return res24, abs(res24 - res12), krul, 25

def qawce(f, a: float, b: float, c: float, epsabs: float, epsrel: float, limit: int):
    # returns (result, abserr, neval, ier, last, lists)
    W = _WList(limit)
    alist = W.alist
    blist = W.blist
    rlist = W.rlist
    elist = W.elist
    iord = W.iord
    ier = 6
    neval = 0
    last = 0
    alist[0] = a
    blist[0] = b
    result = 0.0
    abserr = 0.0
    if c == a or c == b or (epsabs <= 0.0 and epsrel < max(50.0 * _EPMACH, 0.5e-28)):
        return result, abserr, neval, ier, last, W
    ier = 0
    aa = b if a > b else a
    bb = a if a > b else b
    krule = 1
    result, abserr, krule, neval = _qc25c(f, aa, bb, c, krule)
    last = 1
    rlist[0] = result
    elist[0] = abserr
    iord[0] = 0
    alist[0] = a
    blist[0] = b
    errbnd = max(epsabs, epsrel * abs(result))
    if limit == 1:
        ier = 1
    if abserr < min(0.01 * abs(result), errbnd) or ier == 1:
        if aa == b:
            result = -result
        return result, abserr, neval, ier, last, W
    alist[0] = aa
    blist[0] = bb
    rlist[0] = result
    errmax = abserr
    maxerr = 0
    area = result
    errsum = abserr
    nrmax = 0
    iroff1 = 0
    iroff2 = 0
    L = 1
    while L < limit:
        last = L + 1
        a1 = alist[maxerr]
        b1 = 0.5 * (alist[maxerr] + blist[maxerr])
        b2 = blist[maxerr]
        if c <= b1 and c > a1:
            b1 = 0.5 * (c + b2)
        if c > b1 and c < b2:
            b1 = 0.5 * (a1 + c)
        a2 = b1
        krule = 2
        area1, error1, krule, nev = _qc25c(f, a1, b1, c, krule)
        neval += nev
        area2, error2, krule, nev = _qc25c(f, a2, b2, c, krule)
        neval += nev
        area12 = area1 + area2
        error12 = error1 + error2
        errsum = errsum + error12 - errmax
        area = area + area12 - rlist[maxerr]
        if abs(rlist[maxerr] - area12) < 1.0e-5 * abs(area12) and error12 >= 0.99 * errmax and krule == 0:
            iroff1 += 1
        if L > 9 and error12 > errmax and krule == 0:
            iroff2 += 1
        rlist[maxerr] = area1
        rlist[L] = area2
        errbnd = max(epsabs, epsrel * abs(area))
        if not (errsum <= errbnd):
            if iroff1 >= 6 and iroff2 > 20:
                ier = 2
            if last == limit:
                ier = 1
            if max(abs(a1), abs(b2)) <= (1.0 + 100.0 * _EPMACH) * (abs(a2) + 1000.0 * _UFLOW):
                ier = 3
        if not (error2 > error1):
            alist[L] = a2
            blist[maxerr] = b1
            blist[L] = b2
            elist[maxerr] = error1
            elist[L] = error2
        else:
            alist[maxerr] = a2
            alist[L] = a1
            blist[L] = b1
            rlist[maxerr] = area2
            rlist[L] = area1
            elist[maxerr] = error2
            elist[L] = error1
        maxerr, errmax, nrmax = _qpsrt0(limit, last, maxerr, elist, iord, nrmax)
        if ier != 0 or errsum <= errbnd:
            break
        L += 1
    if L == limit:
        L -= 1
    result = 0.0
    for k in range(L + 1):
        result = result + rlist[k]
    abserr = errsum
    if aa == b:
        result = -result
    return result, abserr, neval, ier, last, W

def _qmomo(alfa: float, beta: float, ri: List[float], rj: List[float], rg: List[float],
           rh: List[float], integr: int):
    alfp1 = alfa + 1.0
    betp1 = beta + 1.0
    alfp2 = alfa + 2.0
    betp2 = beta + 2.0
    ralf = math.pow(2.0, alfp1)
    rbet = math.pow(2.0, betp1)
    ri[0] = ralf / alfp1
    rj[0] = rbet / betp1
    ri[1] = ri[0] * alfa / alfp2
    rj[1] = rj[0] * beta / betp2
    an = 2.0
    for i in range(2, 25):
        ri[i] = -(ralf + an * (an - alfp2) * ri[i - 1]) / ((an - 1.0) * (an + alfp1))
        rj[i] = -(rbet + an * (an - betp2) * rj[i - 1]) / ((an - 1.0) * (an + betp1))
        an = an + 1.0
    if integr == 1:
        for i in range(1, 25, 2):
            rj[i] = -rj[i]
        return
    if integr != 3:
        rg[0] = -ri[0] / alfp1
        rg[1] = -(ralf + ralf) / (alfp2 * alfp2) - rg[0]
        an = 2.0
        for i in range(2, 25):
            rg[i] = -(an * (an - alfp2) * rg[i - 1] - an * ri[i - 1] + (an - 1.0) * ri[i]) / ((an - 1.0) * (an + alfp1))
            an = an + 1.0
        if integr == 2:
            for i in range(1, 25, 2):
                rj[i] = -rj[i]
            return
    rh[0] = -rj[0] / betp1
    rh[1] = -(rbet + rbet) / (betp2 * betp2) - rh[0]
    an = 2.0
    for i in range(2, 25):
        rh[i] = -(an * (an - betp2) * rh[i - 1] - an * rj[i - 1] + (an - 1.0) * rj[i]) / ((an - 1.0) * (an + betp1))
        an = an + 1.0
    for i in range(1, 25, 2):
        rh[i] = -rh[i]
    for i in range(1, 25, 2):
        rj[i] = -rj[i]

def _dot(c: List[float], m: List[float], k: int) -> float:
    s = 0.0
    for i in range(k):
        s = s + c[i] * m[i]
    return s

def _qc25s(f, a: float, b: float, bl: float, br: float, alfa: float, beta: float,
           ri: List[float], rj: List[float], rg: List[float], rh: List[float], integr: int,
           resasc_in: float):
    # returns (result, abserr, resasc, nev); Chebyshev branches keep resasc_in (as in C)
    cheb12 = [0.0] * 13
    cheb24 = [0.0] * 25
    if bl == a and (alfa != 0.0 or integr == 2 or integr == 4):
        hlgth = 0.5 * (br - bl)
        centr = 0.5 * (br + bl)
        fix = b - centr
        fval = [0.0] * 25
        fval[0] = 0.5 * f(centr + hlgth) * math.pow(fix - hlgth, beta)
        fval[12] = f(centr) * math.pow(fix, beta)
        fval[24] = 0.5 * f(centr - hlgth) * math.pow(fix + hlgth, beta)
        for i in range(1, 12):
            u = hlgth * _X11[i - 1]
            fval[i] = f(centr + u) * math.pow(fix - u, beta)
            fval[24 - i] = f(centr - u) * math.pow(fix + u, beta)
        factor = math.pow(hlgth, alfa + 1.0)
        result = 0.0
        abserr = 0.0
        if integr > 2:
            fval[0] = fval[0] * math.log(fix - hlgth)
            fval[12] = fval[12] * math.log(fix)
            fval[24] = fval[24] * math.log(fix + hlgth)
            for i in range(1, 12):
                u = hlgth * _X11[i - 1]
                fval[i] = fval[i] * math.log(fix - u)
                fval[24 - i] = fval[24 - i] * math.log(fix + u)
        _qcheb(_X11, fval, cheb12, cheb24)
        res12 = _dot(cheb12, ri, 13)
        res24 = _dot(cheb24, ri, 25)
        if integr == 1 or integr == 3:
            return (result + res24) * factor, (abserr + abs(res24 - res12)) * factor, resasc_in, 25
        dc = math.log(br - bl)
        result = res24 * dc
        abserr = abs((res24 - res12) * dc)
        res12 = _dot(cheb12, rg, 13)
        res24 = _dot(cheb24, rg, 25)
        return (result + res24) * factor, (abserr + abs(res24 - res12)) * factor, resasc_in, 25
    if br == b and (beta != 0.0 or integr == 3 or integr == 4):
        hlgth = 0.5 * (b - bl)
        centr = 0.5 * (br + bl)
        fix = centr - a
        fval = [0.0] * 25
        fval[0] = 0.5 * f(centr + hlgth) * math.pow(fix + hlgth, alfa)
        fval[12] = f(centr) * math.pow(fix, alfa)
        fval[24] = 0.5 * f(centr - hlgth) * math.pow(fix - hlgth, alfa)
        for i in range(1, 12):
            u = hlgth * _X11[i - 1]
            fval[i] = f(centr + u) * math.pow(fix + u, alfa)
            fval[24 - i] = f(centr - u) * math.pow(fix - u, alfa)
        factor = math.pow(hlgth, beta + 1.0)
        result = 0.0
        abserr = 0.0
        if integr == 2 or integr == 4:
            fval[0] = fval[0] * math.log(hlgth + fix)
            fval[12] = fval[12] * math.log(fix)
            fval[24] = fval[24] * math.log(fix - hlgth)
            for i in range(1, 12):
                u = hlgth * _X11[i - 1]
                fval[i] = fval[i] * math.log(fix + u)
                fval[24 - i] = fval[24 - i] * math.log(fix - u)
        _qcheb(_X11, fval, cheb12, cheb24)
        res12 = _dot(cheb12, rj, 13)
        res24 = _dot(cheb24, rj, 25)
        if integr == 1 or integr == 2:
            return (result + res24) * factor, (abserr + abs(res24 - res12)) * factor, resasc_in, 25
        dc = math.log(br - bl)
        result = res24 * dc
        abserr = abs((res24 - res12) * dc)
        res12 = _dot(cheb12, rh, 13)
        res24 = _dot(cheb24, rh, 25)
        return (result + res24) * factor, (abserr + abs(res24 - res12)) * factor, resasc_in, 25
    result, abserr, resabs, resasc = _qk15w(f, 2, a, b, alfa, beta, integr, bl, br)
    return result, abserr, resasc, 15

def qawse(f, a: float, b: float, alfa: float, beta: float, integr: int, epsabs: float,
          epsrel: float, limit: int):
    W = _WList(limit)
    alist = W.alist
    blist = W.blist
    rlist = W.rlist
    elist = W.elist
    iord = W.iord
    ier = 6
    neval = 0
    last = 0
    result = 0.0
    abserr = 0.0
    if (b <= a or (epsabs == 0.0 and epsrel < max(50 * _EPMACH, 0.5e-28)) or alfa <= -1.0
            or beta <= -1.0 or integr < 1 or integr > 4 or limit < 2):
        return result, abserr, neval, ier, last, W
    ier = 0
    ri = [0.0] * 25
    rj = [0.0] * 25
    rg = [0.0] * 25
    rh = [0.0] * 25
    _qmomo(alfa, beta, ri, rj, rg, rh, integr)
    centre = 0.5 * (b + a)
    resas1 = 0.0
    resas2 = 0.0
    area1, error1, resas1, nev = _qc25s(f, a, b, a, centre, alfa, beta, ri, rj, rg, rh, integr, resas1)
    neval = nev
    area2, error2, resas2, nev = _qc25s(f, a, b, centre, b, alfa, beta, ri, rj, rg, rh, integr, resas2)
    last = 2
    neval += nev
    result = area1 + area2
    abserr = error1 + error2
    errbnd = max(epsabs, epsrel * abs(result))
    if not (error2 > error1):
        alist[0] = a
        alist[1] = centre
        blist[0] = centre
        blist[1] = b
        rlist[0] = area1
        rlist[1] = area2
        elist[0] = error1
        elist[1] = error2
    else:
        alist[0] = centre
        alist[1] = a
        blist[0] = b
        blist[1] = centre
        rlist[0] = area2
        rlist[1] = area1
        elist[0] = error2
        elist[1] = error1
    iord[0] = 0
    iord[1] = 1
    if limit == 2:
        ier = 1
    if abserr <= errbnd or ier == 1:
        return result, abserr, neval, ier, last, W
    errmax = elist[0]
    maxerr = 0
    nrmax = 0
    area = result
    errsum = abserr
    iroff1 = 0
    iroff2 = 0
    L = 2
    while L < limit:
        last = L + 1
        a1 = alist[maxerr]
        b1 = 0.5 * (alist[maxerr] + blist[maxerr])
        a2 = b1
        b2 = blist[maxerr]
        area1, error1, resas1, nev = _qc25s(f, a, b, a1, b1, alfa, beta, ri, rj, rg, rh, integr, resas1)
        neval += nev
        area2, error2, resas2, nev = _qc25s(f, a, b, a2, b2, alfa, beta, ri, rj, rg, rh, integr, resas2)
        neval += nev
        area12 = area1 + area2
        error12 = error1 + error2
        errsum = errsum + error12 - errmax
        area = area + area12 - rlist[maxerr]
        if not (a == a1 or b == b2):
            if resas1 != error1 and resas2 != error2:
                if abs(rlist[maxerr] - area12) < 1.0e-5 * abs(area12) and error12 >= 0.99 * errmax:
                    iroff1 += 1
                if L > 9 and error12 > errmax:
                    iroff2 += 1
        rlist[maxerr] = area1
        rlist[L] = area2
        errbnd = max(epsabs, epsrel * abs(area))
        if not (errsum <= errbnd):
            if last == limit:
                ier = 1
            if iroff1 >= 6 or iroff2 >= 20:
                ier = 2
            if max(abs(a1), abs(b2)) <= (1.0 + 100.0 * _EPMACH) * (abs(a2) + 1.0e3 * _UFLOW):
                ier = 3
        if not (error2 > error1):
            alist[L] = a2
            blist[maxerr] = b1
            blist[L] = b2
            elist[maxerr] = error1
            elist[L] = error2
        else:
            alist[maxerr] = a2
            alist[L] = a1
            blist[L] = b1
            rlist[maxerr] = area2
            rlist[L] = area1
            elist[maxerr] = error2
            elist[L] = error1
        maxerr, errmax, nrmax = _qpsrt0(limit, last, maxerr, elist, iord, nrmax)
        if ier != 0 or errsum <= errbnd:
            break
        L += 1
    if L == limit:
        L -= 1
    result = 0.0
    for k in range(L + 1):
        result = result + rlist[k]
    abserr = errsum
    return result, abserr, neval, ier, last, W

def _tridiag(d: List[float], d1: List[float], d2: List[float], v: List[float], off: int, noequ: int):
    # Gaussian elimination with partial pivoting on the tridiagonal system (dqc25f)
    for i in range(noequ - 1):
        if abs(d[i]) >= abs(d1[i + 1]):
            fact = d1[i + 1] / d[i]
            d[i + 1] -= fact * d2[i]
            v[off + i + 1] -= fact * v[off + i]
            d1[i + 1] = 0.0
        else:
            fact = d[i] / d1[i + 1]
            d[i] = d1[i + 1]
            temp = d[i + 1]
            d[i + 1] = d2[i] - fact * temp
            if i != noequ - 1:
                d1[i + 1] = d2[i + 1]
                d2[i + 1] = -fact * d1[i + 1]
            d2[i] = temp
            temp = v[off + i]
            v[off + i] = v[off + i + 1]
            v[off + i + 1] = temp - fact * v[off + i + 1]
    v[off + noequ] /= d[noequ]
    v[off + noequ - 1] = (v[off + noequ - 1] - d2[noequ - 1] * v[off + noequ]) / d[noequ - 1]
    for i in range(noequ - 2, -1, -1):
        v[off + i] = (v[off + i] - d2[i] * v[off + i + 1] - d1[i + 1] * v[off + i + 2]) / d[i]

class _Moments:
    momcom: int
    maxp1: int
    chebmo: List[float]

    def __init__(self, maxp1: int):
        self.momcom = 0
        self.maxp1 = maxp1
        self.chebmo = [0.0] * (maxp1 * 25)

def _qc25f(f, a: float, b: float, omega: float, integr: int, nrmom: int, ksave: int, M: _Moments):
    # returns (result, abserr, neval, resabs, resasc)
    maxp1 = M.maxp1
    chebmo = M.chebmo
    centr = 0.5 * (b + a)
    hlgth = 0.5 * (b - a)
    parint = omega * hlgth
    if abs(parint) <= 2.0:
        result, abserr, resabs, resasc = _qk15w(f, 1, omega, 0.0, 0.0, 0.0, integr, a, b)
        return result, abserr, 15, resabs, resasc
    conc = hlgth * math.cos(centr * omega)
    cons = hlgth * math.sin(centr * omega)
    resasc = _OFLOW
    m = 0
    if nrmom >= M.momcom and ksave != 1:
        m = M.momcom
        d = [0.0] * 25
        d1 = [0.0] * 25
        d2 = [0.0] * 25
        v = [0.0] * 28
        par2 = parint * parint
        par22 = par2 + 2.0
        sinpar = math.sin(parint)
        cospar = math.cos(parint)
        v[0] = 2.0 * sinpar / parint
        v[1] = (8.0 * cospar + (par2 + par2 - 8.0) * sinpar / parint) / par2
        v[2] = (32.0 * (par2 - 12.0) * cospar + (2.0 * ((par2 - 80.0) * par2 + 192.0) * sinpar) / parint) / (par2 * par2)
        ac = 8.0 * cospar
        as_ = 24.0 * parint * sinpar
        noequ = 24
        if abs(parint) <= 24.0:
            an = 6.0
            for k in range(noequ):
                an2 = an * an
                d[k] = -2.0 * (an2 - 4.0) * (par22 - an2 - an2)
                d2[k] = (an - 1.0) * (an - 2.0) * par2
                d1[k + 1] = (an + 3.0) * (an + 4.0) * par2
                v[k + 3] = as_ - (an2 - 4.0) * ac
                an = an + 2.0
            an2 = an * an
            d[noequ] = -2.0 * (an2 - 4.0) * (par22 - an2 - an2)
            v[noequ + 3] = as_ - (an2 - 4.0) * ac
            v[3] = v[3] - (56.0 * par2 * v[2])
            ass = parint * sinpar
            asap = (((((210.0 * par2 - 1.0) * cospar - (105.0 * par2 - 63.0) * ass) / an2
                      - (1.0 - 15.0 * par2) * cospar + 15.0 * ass) / an2
                     - cospar + 3.0 * ass) / an2 - cospar) / an2
            v[noequ + 3] = v[noequ + 3] - (2.0 * asap * par2 * (an - 1.0) * (an - 2.0))
            _tridiag(d, d1, d2, v, 3, noequ)
        else:
            an = 4.0
            for i in range(3, 13):
                an2 = an * an
                v[i] = ((an2 - 4.0) * (2.0 * (par22 - an2 - an2) * v[i - 1] - ac)
                        + as_ - par2 * (an + 1.0) * (an + 2.0) * v[i - 2]) / (par2 * (an - 1.0) * (an - 2.0))
                an = an + 2.0
        for j in range(13):
            chebmo[M.momcom + maxp1 * (2 * j)] = v[j]
        v[0] = 2.0 * (sinpar - parint * cospar) / par2
        v[1] = (18.0 - 48.0 / par2) * sinpar / par2 + (-2.0 + 48.0 / par2) * cospar / parint
        ac = -24.0 * parint * cospar
        as_ = -8.0 * sinpar
        if abs(parint) <= 24.0:
            for k in range(25):
                d[k] = 0.0
                d1[k] = 0.0
                d2[k] = 0.0
            an = 5.0
            for k in range(noequ):
                an2 = an * an
                d[k] = -2.0 * (an2 - 4.0) * (par22 - an2 - an2)
                d2[k] = (an - 1.0) * (an - 2.0) * par2
                d1[k + 1] = (an + 3.0) * (an + 4.0) * par2
                v[k + 2] = ac + (an2 - 4.0) * as_
                an = an + 2.0
            an2 = an * an
            d[noequ] = -2.0 * (an2 - 4.0) * (par22 - an2 - an2)
            v[noequ + 2] = ac + (an2 - 4.0) * as_
            v[2] = v[2] - 42.0 * par2 * v[1]
            ass = parint * cospar
            asap = (((((105.0 * par2 - 63.0) * ass + (210.0 * par2 - 1.0) * sinpar) / an2
                      + (15.0 * par2 - 1.0) * sinpar - 15.0 * ass) / an2
                     - 3.0 * ass - sinpar) / an2 - sinpar) / an2
            v[noequ + 2] = v[noequ + 2] - 2.0 * asap * par2 * (an - 1.0) * (an - 2.0)
            _tridiag(d, d1, d2, v, 2, noequ)
        else:
            an = 3.0
            for i in range(2, 12):
                an2 = an * an
                v[i] = ((an2 - 4.0) * (2.0 * (par22 - an2 - an2) * v[i - 1] + as_)
                        + ac - par2 * (an + 1.0) * (an + 2.0) * v[i - 2]) / (par2 * (an - 1.0) * (an - 2.0))
                an = an + 2.0
        for j in range(12):
            chebmo[m + maxp1 * (2 * j + 1)] = v[j]
    if nrmom < M.momcom:
        m = nrmom
    if M.momcom < maxp1 - 1 and nrmom >= M.momcom:
        M.momcom += 1
    fval = _fval25(f, centr, hlgth)
    cheb12 = [0.0] * 13
    cheb24 = [0.0] * 25
    _qcheb(_X11, fval, cheb12, cheb24)
    resc12 = cheb12[12] * chebmo[m + maxp1 * 12]
    ress12 = 0.0
    k = 10
    while k >= 0:
        resc12 = resc12 + cheb12[k] * chebmo[m + maxp1 * k]
        ress12 = ress12 + cheb12[k + 1] * chebmo[m + maxp1 * (k + 1)]
        k -= 2
    resc24 = cheb24[24] * chebmo[m + maxp1 * 24]
    ress24 = 0.0
    resabs = abs(cheb24[24])
    k = 22
    while k >= 0:
        resc24 = resc24 + cheb24[k] * chebmo[m + maxp1 * k]
        ress24 = ress24 + cheb24[k + 1] * chebmo[m + maxp1 * (k + 1)]
        resabs = resabs + abs(cheb24[k]) + abs(cheb24[k + 1])
        k -= 2
    estc = abs(resc24 - resc12)
    ests = abs(ress24 - ress12)
    resabs = resabs * abs(hlgth)
    if integr != 2:
        return conc * resc24 - cons * ress24, abs(conc * estc) + abs(cons * ests), 25, resabs, resasc
    return conc * ress24 + cons * resc24, abs(conc * ests) + abs(cons * estc), 25, resabs, resasc

def qawoe(f, a: float, b: float, omega: float, integr: int, epsabs: float, epsrel: float,
          limit: int, icall: int, maxp1: int, M: _Moments):
    # returns (result, abserr, neval, ier, last, W); rlist2 uses _Eps (1-based)
    W = _WList(limit)
    alist = W.alist
    blist = W.blist
    rlist = W.rlist
    elist = W.elist
    iord = W.iord
    nnlog = W.nnlog
    ier = 6
    neval = 0
    last = 0
    result = 0.0
    abserr = 0.0
    alist[0] = a
    blist[0] = b
    if (integr != 1 and integr != 2) or (epsabs <= 0.0 and epsrel < max(50.0 * _EPMACH, 0.5e-28)) or icall < 1 or maxp1 < 1:
        return result, abserr, neval, ier, last, W
    ier = 0
    domega = abs(omega)
    nrmom = 0
    if icall < 2:
        M.momcom = 0
    result, abserr, neval, defabs, resabs = _qc25f(f, a, b, domega, integr, nrmom, 0, M)
    dres = abs(result)
    errbnd = max(epsabs, epsrel * dres)
    rlist[0] = result
    elist[0] = abserr
    iord[0] = 0
    if abserr <= 100.0 * _EPMACH * defabs and abserr > errbnd:
        ier = 2
    if limit == 1:
        ier = 1
    if ier != 0 or abserr <= errbnd:
        if integr == 2 and omega < 0.0:
            result = -result
        return result, abserr, neval, ier, last, W
    last = 1
    errmax = abserr
    maxerr = 0
    area = result
    errsum = abserr
    abserr = _OFLOW
    nrmax = 0
    extrap = False
    noext = False
    ierror = 0
    iroff1 = 0
    iroff2 = 0
    iroff3 = 0
    ktmin = 0
    small = abs(b - a) * 0.75
    E = _Eps()
    numrl2 = -1
    extall = False
    correc = 0.0
    erlarg = 0.0
    ertest = 0.0
    if 0.5 * abs(b - a) * domega <= 2.0:
        numrl2 = 0
        extall = True
        E.epstab[1] = result
    elif 0.25 * abs(b - a) * domega <= 2.0:
        extall = True
    ksgn = 1 if dres >= (1.0 - 50.0 * _EPMACH) * defabs else -1
    L = 1
    goto170 = False
    while L < limit:
        last = L + 1
        nrmom = nnlog[maxerr] + 1
        a1 = alist[maxerr]
        b1 = 0.5 * (alist[maxerr] + blist[maxerr])
        a2 = b1
        b2 = blist[maxerr]
        erlast = errmax
        area1, error1, nev, resabs, defab1 = _qc25f(f, a1, b1, domega, integr, nrmom, 0, M)
        neval += nev
        area2, error2, nev, resabs, defab2 = _qc25f(f, a2, b2, domega, integr, nrmom, 1, M)
        neval += nev
        area12 = area1 + area2
        error12 = error1 + error2
        errsum = errsum + error12 - errmax
        area = area + area12 - rlist[maxerr]
        if defab1 != error1 and defab2 != error2:
            if not (abs(rlist[maxerr] - area12) > 1.0e-5 * abs(area12) or error12 < 0.99 * errmax):
                if extrap:
                    iroff2 += 1
                else:
                    iroff1 += 1
            if L > 9 and error12 > errmax:
                iroff3 += 1
        rlist[maxerr] = area1
        rlist[L] = area2
        nnlog[maxerr] = nrmom
        nnlog[L] = nrmom
        errbnd = max(epsabs, epsrel * abs(area))
        if iroff1 + iroff2 >= 10 or iroff3 >= 20:
            ier = 2
        if iroff2 >= 5:
            ierror = 3
        if last == limit:
            ier = 1
        if max(abs(a1), abs(b2)) <= (1.0 + 100.0 * _EPMACH) * (abs(a2) + 1.0e3 * _EPMACH):
            ier = 4
        if not (error2 > error1):
            alist[L] = a2
            blist[maxerr] = b1
            blist[L] = b2
            elist[maxerr] = error1
            elist[L] = error2
        else:
            alist[maxerr] = a2
            alist[L] = a1
            blist[L] = b1
            rlist[maxerr] = area2
            rlist[L] = area1
            elist[maxerr] = error2
            elist[L] = error1
        maxerr, errmax, nrmax = _qpsrt0(limit, last, maxerr, elist, iord, nrmax)
        if errsum <= errbnd:
            goto170 = True
            break
        if ier != 0:
            break
        # label dispatch: 0 -> continue loop, 1 -> LINE50, 2 -> LINE70, 3 -> LINE90,
        # 4 -> LINE120, 5 -> LINE130, 6 -> LINE140 (nothing)
        lab = 1
        if L == 1 and extall:
            lab = 4
        elif noext:
            L += 1
            continue
        elif not extall:
            lab = 1
        else:
            erlarg = erlarg - erlast
            if abs(b1 - a1) > small:
                erlarg = erlarg + error12
            lab = 2 if extrap else 1
        if lab == 1:
            width = abs(blist[maxerr] - alist[maxerr])
            if width > small:
                L += 1
                continue
            if not extall:
                small = small * 0.5
                if 0.25 * width * domega > 2.0:
                    L += 1
                    continue
                extall = True
                lab = 5
            else:
                extrap = True
                nrmax = 1
                lab = 2
        if lab == 2:
            lab = 3
            if not (ierror == 3 or erlarg <= ertest):
                jupbnd = (limit + 3 - last) if last > 2 + (limit // 2) else last
                k = nrmax
                while k < jupbnd:
                    maxerr = iord[nrmax]
                    errmax = elist[maxerr]
                    if abs(blist[maxerr] - alist[maxerr]) > small:
                        lab = 6
                        break
                    nrmax += 1
                    k += 1
        if lab == 3:
            numrl2 += 1
            E.epstab[numrl2 + 1] = area
            brk = False
            if numrl2 >= 2:
                E.n = numrl2 + 1
                reseps, abseps = _qelg(E)
                numrl2 = E.n - 1
                ktmin += 1
                if ktmin > 5 and abserr < 1.0e-3 * errsum:
                    ier = 5
                if abseps < abserr:
                    ktmin = 0
                    abserr = abseps
                    result = reseps
                    correc = erlarg
                    ertest = max(epsabs, epsrel * abs(reseps))
                    if abserr <= ertest:
                        brk = True
                if not brk:
                    if numrl2 == 0:
                        noext = True
                    if ier == 5:
                        brk = True
            if brk:
                break
            maxerr = iord[0]
            errmax = elist[maxerr]
            nrmax = 0
            extrap = False
            small = small * 0.5
            erlarg = errsum
            L += 1
            continue
        if lab == 4:
            small = small * 0.5
            numrl2 += 1
            E.epstab[numrl2 + 1] = area
            lab = 5
        if lab == 5:
            ertest = errbnd
            erlarg = errsum
        L += 1
    if L == limit:
        L -= 1
    done = False
    if not goto170:
        if abserr == _OFLOW or E.nres == 0:
            goto170 = True
        elif ier + ierror == 0:
            pass
        else:
            if ierror == 3:
                abserr = abserr + correc
            if ier == 0:
                ier = 3
            if result != 0.0 and area != 0.0:
                if abserr / abs(result) > errsum / abs(area):
                    goto170 = True
            elif abserr > errsum:
                goto170 = True
            elif area == 0.0:
                done = True
        if not goto170 and not done:
            if not (ksgn == -1 and max(abs(result), abs(area)) <= defabs * 0.01):
                if 0.01 > result / area or result / area > 100.0 or errsum >= abs(area):
                    ier = 6
    if goto170:
        result = 0.0
        for k in range(L + 1):
            result = result + rlist[k]
        abserr = errsum
    if ier > 2:
        ier -= 1
    if integr == 2 and omega < 0.0:
        result = -result
    return result, abserr, neval, ier, last, W

class _QawfOut:
    rslst: List[float]
    erlst: List[float]
    ierlst: List[int]
    lst: int

    def __init__(self, limlst: int):
        self.rslst = [0.0] * max(limlst, 1)
        self.erlst = [0.0] * max(limlst, 1)
        self.ierlst = [0] * max(limlst, 1)
        self.lst = 0

def qawfe(f, a: float, omega: float, integr: int, epsabs: float, limlst: int, limit: int,
          maxp1: int):
    # returns (result, abserr, neval, ier, out)
    O = _QawfOut(limlst)
    result = 0.0
    abserr = 0.0
    neval = 0
    ier = 6
    if (integr != 1 and integr != 2) or epsabs <= 0.0 or limlst < 3:
        return result, abserr, neval, ier, O
    ier = 0
    if omega == 0.0:
        if integr == 1:
            result, abserr, neval, ier = qagie(f, 0.0, 1, epsabs, 0.0, limit)
        O.rslst[0] = result
        O.erlst[0] = abserr
        O.ierlst[0] = ier
        O.lst = 1
        return result, abserr, neval, ier, O
    p = 0.9
    l = int(abs(omega))
    dl = 2 * l + 1.0
    cycle = dl * math.pi / abs(omega)
    ktmin = 0
    E = _Eps()
    numrl2 = -1
    c1 = a
    c2 = cycle + a
    p1 = 1.0 - p
    eps = epsabs * p1 if epsabs > (_UFLOW / p1) else epsabs
    ep = eps
    fact = 1.0
    correc = 0.0
    errsum = 0.0
    drl = 0.0
    ll = 0
    M = _Moments(maxp1)
    goto80 = False
    it = 0
    while it < limlst:
        O.lst = it + 1
        epsa = eps * fact
        r, e, nev, ie, last, W = qawoe(f, c1, c2, omega, integr, epsa, 0.0, limit, O.lst, maxp1, M)
        O.rslst[it] = r
        O.erlst[it] = e
        O.ierlst[it] = ie
        neval += nev
        fact = fact * p
        errsum = errsum + O.erlst[it]
        drl = 50.0 * abs(O.rslst[it])
        if errsum + drl <= epsabs and it >= 5:
            goto80 = True
            break
        correc = max(correc, O.erlst[it])
        if O.ierlst[it] != 0:
            eps = max(ep, correc * p1)
            ier = 7
        if ier == 7 and errsum + drl <= correc * 10.0 and it > 4:
            goto80 = True
            break
        numrl2 += 1
        if it == 0:
            E.epstab[1] = O.rslst[0]
        else:
            E.epstab[numrl2 + 1] = E.epstab[ll + 1] + O.rslst[it]
            if it != 1:
                if O.lst == limlst:
                    ier = 1
                E.n = numrl2 + 1
                reseps, abseps = _qelg(E)
                numrl2 = E.n - 1
                ktmin += 1
                if ktmin >= 15 and abserr <= 0.001 * (errsum + drl):
                    ier = 4
                if not (abseps > abserr and it != 2):
                    abserr = abseps
                    result = reseps
                    ktmin = 0
                    if abserr + 10.0 * correc <= epsabs or (abserr <= epsabs and 10.0 * correc >= epsabs):
                        break
                if ier != 0 and ier != 7:
                    break
        ll = numrl2
        c1 = c2
        c2 = c2 + cycle
        it += 1
    psum_last = E.epstab[numrl2 + 1]
    if not goto80:
        abserr = abserr + 10.0 * correc
        if ier == 0:
            return result, abserr, neval, ier, O
        if result != 0.0 and psum_last != 0.0:
            if abserr / abs(result) > (errsum + drl) / abs(psum_last):
                goto80 = True
            else:
                if ier >= 1 and ier != 7:
                    abserr = abserr + drl
                return result, abserr, neval, ier, O
        elif abserr > errsum:
            goto80 = True
        elif psum_last == 0.0:
            return result, abserr, neval, ier, O
        else:
            if abserr / abs(result) > (errsum + drl) / abs(psum_last):
                goto80 = True
            else:
                if ier >= 1 and ier != 7:
                    abserr = abserr + drl
                return result, abserr, neval, ier, O
    result = psum_last
    abserr = errsum + drl
    return result, abserr, neval, ier, O
