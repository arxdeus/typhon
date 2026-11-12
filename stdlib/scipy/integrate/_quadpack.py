# Ported from scipy 1.18.1 (BSD-3-Clause)
#
# Pure Typhon port of the QUADPACK routines used by scipy.integrate.quad
# (dqagse, dqagie, dqagpe with dqk21, dqk15i, dqelg, dqpsrt). Arrays are
# 1-based (index 0 unused) to follow the reference code closely.
# Deviations: `full_output`, `weight`, `wvar`, `wopts`, `complex_func` are
# not supported. Warnings (ier 1..5) are silently ignored; invalid input
# (ier 6) raises ValueError.

import math

_EPMACH = 2.220446049250313e-16
_UFLOW = 2.2250738585072014e-308
_OFLOW = 1.7976931348623157e+308

_XGK21 = [0.0, 0.995657163025808080735527280689003, 0.973906528517171720077964012084452,
          0.930157491355708226001207180059508, 0.865063366688984510732096688423493,
          0.780817726586416897063717578345042, 0.679409568299024406234327365114874,
          0.562757134668604683339000099272694, 0.433395394129247190799265943165784,
          0.294392862701460198131126603103866, 0.148874338981631210884826001129720, 0.0]
_WGK21 = [0.0, 0.011694638867371874278064396062192, 0.032558162307964727478818972459390,
          0.054755896574351996031381300244580, 0.075039674810919952767043140916190,
          0.093125454583697605535065465083366, 0.109387158802297641899210590325805,
          0.123491976262065851077958109831074, 0.134709217311473325928054001771707,
          0.142775938577060080797094273138717, 0.147739104901338491374841515972068,
          0.149445554002916905664936468389821]
_WG10 = [0.0, 0.066671344308688137593568809893332, 0.149451349150580593145776339657697,
         0.219086362515982043995534934228163, 0.269266719309996355091226921569469,
         0.295524224714752870173892994651338]

_XGK15 = [0.0, 0.991455371120812639206854697526329, 0.949107912342758524526189684047851,
          0.864864423359769072789712788640926, 0.741531185599394439863864773280788,
          0.586087235467691130294144838258730, 0.405845151377397166906606412076961,
          0.207784955007898467600689403773245, 0.0]
_WGK15 = [0.0, 0.022935322010529224963732008058970, 0.063092092629978553290700663189204,
          0.104790010322250183839876322541518, 0.140653259715525918745189590510238,
          0.169004726639267902826583426598550, 0.190350578064785409913256402421014,
          0.204432940075298892414161999234649, 0.209482141084727828012999174891714]
_WG7 = [0.0, 0.0, 0.129484966168869693270611432679082, 0.0,
        0.279705391489276667901467771423780, 0.0, 0.381830050505118944950369775488975,
        0.0, 0.417959183673469387755102040816327]

def _finish_err(abserr: float, resabs: float, resasc: float) -> float:
    if resasc != 0.0 and abserr != 0.0:
        abserr = resasc * min(1.0, math.pow(200.0 * abserr / resasc, 1.5))
    if resabs > _UFLOW / (50.0 * _EPMACH):
        abserr = max(_EPMACH * 50.0 * resabs, abserr)
    return abserr

def _qk21(f, a: float, b: float):
    centr = 0.5 * (a + b)
    hlgth = 0.5 * (b - a)
    dhlgth = abs(hlgth)
    fv1 = [0.0] * 11
    fv2 = [0.0] * 11
    resg = 0.0
    fc = f(centr)
    resk = _WGK21[11] * fc
    resabs = abs(resk)
    for j in range(1, 6):
        jtw = 2 * j
        absc = hlgth * _XGK21[jtw]
        fval1 = f(centr - absc)
        fval2 = f(centr + absc)
        fv1[jtw] = fval1
        fv2[jtw] = fval2
        fsum = fval1 + fval2
        resg += _WG10[j] * fsum
        resk += _WGK21[jtw] * fsum
        resabs += _WGK21[jtw] * (abs(fval1) + abs(fval2))
    for j in range(1, 6):
        jtwm1 = 2 * j - 1
        absc = hlgth * _XGK21[jtwm1]
        fval1 = f(centr - absc)
        fval2 = f(centr + absc)
        fv1[jtwm1] = fval1
        fv2[jtwm1] = fval2
        fsum = fval1 + fval2
        resk += _WGK21[jtwm1] * fsum
        resabs += _WGK21[jtwm1] * (abs(fval1) + abs(fval2))
    reskh = resk * 0.5
    resasc = _WGK21[11] * abs(fc - reskh)
    for j in range(1, 11):
        resasc += _WGK21[j] * (abs(fv1[j] - reskh) + abs(fv2[j] - reskh))
    result = resk * hlgth
    resabs *= dhlgth
    resasc *= dhlgth
    abserr = _finish_err(abs((resk - resg) * hlgth), resabs, resasc)
    return result, abserr, resabs, resasc

def _qk15i(f, boun: float, inf: int, a: float, b: float):
    dinf = float(min(1, inf))
    centr = 0.5 * (a + b)
    hlgth = 0.5 * (b - a)
    tabsc1 = boun + dinf * (1.0 - centr) / centr
    fval1 = f(tabsc1)
    if inf == 2:
        fval1 += f(-tabsc1)
    fc = (fval1 / centr) / centr
    resg = _WG7[8] * fc
    resk = _WGK15[8] * fc
    resabs = abs(resk)
    fv1 = [0.0] * 8
    fv2 = [0.0] * 8
    for j in range(1, 8):
        absc = hlgth * _XGK15[j]
        absc1 = centr - absc
        absc2 = centr + absc
        t1 = boun + dinf * (1.0 - absc1) / absc1
        t2 = boun + dinf * (1.0 - absc2) / absc2
        f1 = f(t1)
        f2 = f(t2)
        if inf == 2:
            f1 += f(-t1)
            f2 += f(-t2)
        f1 = (f1 / absc1) / absc1
        f2 = (f2 / absc2) / absc2
        fv1[j] = f1
        fv2[j] = f2
        fsum = f1 + f2
        resg += _WG7[j] * fsum
        resk += _WGK15[j] * fsum
        resabs += _WGK15[j] * (abs(f1) + abs(f2))
    reskh = resk * 0.5
    resasc = _WGK15[8] * abs(fc - reskh)
    for j in range(1, 8):
        resasc += _WGK15[j] * (abs(fv1[j] - reskh) + abs(fv2[j] - reskh))
    result = resk * hlgth
    resasc *= hlgth
    resabs *= hlgth
    abserr = _finish_err(abs((resk - resg) * hlgth), resabs, resasc)
    return result, abserr, resabs, resasc

class _QState:
    limit: int
    alist: List[float]
    blist: List[float]
    rlist: List[float]
    elist: List[float]
    iord: List[int]
    level: List[int]
    maxerr: int
    errmax: float
    nrmax: int

    def __init__(self, limit: int):
        self.limit = limit
        m = max(limit, 2) + 4
        self.alist = [0.0] * m
        self.blist = [0.0] * m
        self.rlist = [0.0] * m
        self.elist = [0.0] * m
        self.iord = [0] * m
        self.level = [0] * m
        self.maxerr = 1
        self.errmax = 0.0
        self.nrmax = 1

def _qpsrt(s: _QState, last: int):
    limit = s.limit
    elist = s.elist
    iord = s.iord
    if last <= 2:
        iord[1] = 1
        iord[2] = 2
    else:
        errmax = elist[s.maxerr]
        if s.nrmax != 1:
            ido = s.nrmax - 1
            for _ in range(ido):
                isucc = iord[s.nrmax - 1]
                if errmax <= elist[isucc]:
                    break
                iord[s.nrmax] = isucc
                s.nrmax -= 1
        jupbn = last
        if last > (limit // 2 + 2):
            jupbn = limit + 3 - last
        errmin = elist[last]
        jbnd = jupbn - 1
        ibeg = s.nrmax + 1
        found = False
        i = ibeg
        while i <= jbnd:
            isucc = iord[i]
            if errmax >= elist[isucc]:
                found = True
                break
            iord[i - 1] = isucc
            i += 1
        if not found:
            iord[jbnd] = s.maxerr
            iord[jupbn] = last
        else:
            iord[i - 1] = s.maxerr
            k = jbnd
            placed = False
            for _ in range(i, jbnd + 1):
                isucc = iord[k]
                if errmin < elist[isucc]:
                    iord[k + 1] = last
                    placed = True
                    break
                iord[k + 1] = isucc
                k -= 1
            if not placed:
                iord[i] = last
    s.maxerr = iord[s.nrmax]
    s.errmax = elist[s.maxerr]

class _Eps:
    epstab: List[float]
    res3la: List[float]
    nres: int
    n: int

    def __init__(self):
        self.epstab = [0.0] * 56
        self.res3la = [0.0] * 4
        self.nres = 0
        self.n = 0

def _qelg(e: _Eps):
    # returns (result, abserr); updates e.n, e.nres, tables
    epstab = e.epstab
    e.nres += 1
    n = e.n
    abserr = _OFLOW
    result = epstab[n]
    if n < 3:
        abserr = max(abserr, 5.0 * _EPMACH * abs(result))
        return result, abserr
    limexp = 50
    epstab[n + 2] = epstab[n]
    newelm = (n - 1) // 2
    epstab[n] = _OFLOW
    num = n
    k1 = n
    for i in range(1, newelm + 1):
        k2 = k1 - 1
        k3 = k1 - 2
        res = epstab[k1 + 2]
        e0 = epstab[k3]
        e1 = epstab[k2]
        e2 = res
        e1abs = abs(e1)
        delta2 = e2 - e1
        err2 = abs(delta2)
        tol2 = max(abs(e2), e1abs) * _EPMACH
        delta3 = e1 - e0
        err3 = abs(delta3)
        tol3 = max(e1abs, abs(e0)) * _EPMACH
        if err2 <= tol2 and err3 <= tol3:
            result = res
            abserr = err2 + err3
            abserr = max(abserr, 5.0 * _EPMACH * abs(result))
            e.n = n
            return result, abserr
        e3 = epstab[k1]
        epstab[k1] = e1
        delta1 = e1 - e3
        err1 = abs(delta1)
        tol1 = max(e1abs, abs(e3)) * _EPMACH
        if err1 <= tol1 or err2 <= tol2 or err3 <= tol3:
            n = i + i - 1
            break
        ss = 1.0 / delta1 + 1.0 / delta2 - 1.0 / delta3
        epsinf = abs(ss * e1)
        if epsinf <= 1e-4:
            n = i + i - 1
            break
        res = e1 + 1.0 / ss
        epstab[k1] = res
        k1 -= 2
        error = err2 + abs(res - e2) + err3
        if error <= abserr:
            abserr = error
            result = res
    if n == limexp:
        n = 2 * (limexp // 2) - 1
    ib = 1 if num % 2 == 1 else 2
    ie = newelm + 1
    for _ in range(ie):
        ib2 = ib + 2
        epstab[ib] = epstab[ib2]
        ib = ib2
    if num != n:
        indx = num - n + 1
        for i in range(1, n + 1):
            epstab[i] = epstab[indx]
            indx += 1
    if e.nres < 4:
        e.res3la[e.nres] = result
        abserr = _OFLOW
    else:
        r3 = e.res3la
        abserr = abs(result - r3[3]) + abs(result - r3[2]) + abs(result - r3[1])
        r3[1] = r3[2]
        r3[2] = r3[3]
        r3[3] = result
    abserr = max(abserr, 5.0 * _EPMACH * abs(result))
    e.n = n
    return result, abserr

def _check_tol(epsabs: float, epsrel: float):
    if epsabs <= 0.0 and epsrel < max(50.0 * _EPMACH, 5e-29):
        raise ValueError("If 'epsabs'<=0, 'epsrel' must be greater than both"
                         " 5e-29 and 50*(machine epsilon).")

def _final(ier: int, ierro: int, result: float, abserr: float, area: float,
           errsum: float, correc: float, ksgn: int, defabs: float, s: _QState, last: int):
    # labels 100..130 of dqagse/dqagie (170..210 of dqagpe)
    use_sum = False
    done = False
    if abserr == _OFLOW:
        use_sum = True
    elif ier + ierro != 0:
        if ierro == 3:
            abserr += correc
        if ier == 0:
            ier = 3
        if result != 0.0 and area != 0.0:
            if abserr / abs(result) > errsum / abs(area):
                use_sum = True
        elif abserr > errsum:
            use_sum = True
        elif area == 0.0:
            done = True
    if not use_sum and not done:
        if not (ksgn == -1 and max(abs(result), abs(area)) <= defabs * 1e-2):
            if 1e-2 > (result / area) or (result / area) > 100.0 or errsum > abs(area):
                ier = 6
    if use_sum:
        result = 0.0
        for k in range(1, last + 1):
            result += s.rlist[k]
        abserr = errsum
    if ier > 2:
        ier -= 1
    return result, abserr, ier

def _qags_core(rule, a0: float, b0: float, epsabs: float, epsrel: float, limit: int):
    # dqagse / dqagie main loop; rule(a, b) -> (result, abserr, resabs, resasc)
    _check_tol(epsabs, epsrel)
    if limit < 1:
        raise ValueError("Invalid 'limit' argument. There must be at least one subinterval")
    s = _QState(limit)
    ier = 0
    ierro = 0
    s.alist[1] = a0
    s.blist[1] = b0
    result, abserr, defabs, resabs = rule(a0, b0)
    dres = abs(result)
    errbnd = max(epsabs, epsrel * dres)
    last = 1
    s.rlist[1] = result
    s.elist[1] = abserr
    s.iord[1] = 1
    if abserr <= 1e2 * _EPMACH * defabs and abserr > errbnd:
        ier = 2
    if limit == 1:
        ier = 1
    if ier != 0 or (abserr <= errbnd and abserr != resabs) or abserr == 0.0:
        return result, abserr, last, ier
    ep = _Eps()
    ep.epstab[1] = result
    s.errmax = abserr
    s.maxerr = 1
    area = result
    errsum = abserr
    abserr = _OFLOW
    s.nrmax = 1
    ep.n = 2
    ktmin = 0
    extrap = False
    noext = False
    iroff1 = 0
    iroff2 = 0
    iroff3 = 0
    ksgn = -1
    if dres >= (1.0 - 50.0 * _EPMACH) * defabs:
        ksgn = 1
    small = 0.0
    erlarg = 0.0
    ertest = 0.0
    correc = 0.0
    goto_sum = False
    for lst in range(2, limit + 1):
        last = lst
        maxerr = s.maxerr
        a1 = s.alist[maxerr]
        b1 = 0.5 * (s.alist[maxerr] + s.blist[maxerr])
        a2 = b1
        b2 = s.blist[maxerr]
        erlast = s.errmax
        area1, error1, _r1, defab1 = rule(a1, b1)
        area2, error2, _r2, defab2 = rule(a2, b2)
        area12 = area1 + area2
        erro12 = error1 + error2
        errsum = errsum + erro12 - s.errmax
        area = area + area12 - s.rlist[maxerr]
        if not (defab1 == error1 or defab2 == error2):
            if not (abs(s.rlist[maxerr] - area12) > 1e-5 * abs(area12) or erro12 < 0.99 * s.errmax):
                if extrap:
                    iroff2 += 1
                else:
                    iroff1 += 1
            if last > 10 and erro12 > s.errmax:
                iroff3 += 1
        s.rlist[maxerr] = area1
        s.rlist[last] = area2
        errbnd = max(epsabs, epsrel * abs(area))
        if iroff1 + iroff2 >= 10 or iroff3 >= 20:
            ier = 2
        if iroff2 >= 5:
            ierro = 3
        if last == limit:
            ier = 1
        if max(abs(a1), abs(b2)) <= (1.0 + 100.0 * _EPMACH) * (abs(a2) + 1000.0 * _UFLOW):
            ier = 4
        if error2 > error1:
            s.alist[maxerr] = a2
            s.alist[last] = a1
            s.blist[last] = b1
            s.rlist[maxerr] = area2
            s.rlist[last] = area1
            s.elist[maxerr] = error2
            s.elist[last] = error1
        else:
            s.alist[last] = a2
            s.blist[maxerr] = b1
            s.blist[last] = b2
            s.elist[maxerr] = error1
            s.elist[last] = error2
        _qpsrt(s, last)
        if errsum <= errbnd:
            goto_sum = True
            break
        if ier != 0:
            break
        if last == 2:
            small = abs(b0 - a0) * 0.375
            erlarg = errsum
            ertest = errbnd
            ep.epstab[2] = area
            continue
        if noext:
            continue
        erlarg -= erlast
        if abs(b1 - a1) > small:
            erlarg += erro12
        if not extrap:
            if abs(s.blist[s.maxerr] - s.alist[s.maxerr]) > small:
                continue
            extrap = True
            s.nrmax = 2
        skip = False
        if ierro != 3 and erlarg > ertest:
            jupbnd = last
            if last > (2 + limit // 2):
                jupbnd = limit + 3 - last
            for _k in range(s.nrmax, jupbnd + 1):
                s.maxerr = s.iord[s.nrmax]
                s.errmax = s.elist[s.maxerr]
                if abs(s.blist[s.maxerr] - s.alist[s.maxerr]) > small:
                    skip = True
                    break
                s.nrmax += 1
        if skip:
            continue
        ep.n += 1
        ep.epstab[ep.n] = area
        reseps, abseps = _qelg(ep)
        ktmin += 1
        if ktmin > 5 and abserr < 1e-3 * errsum:
            ier = 5
        if abseps < abserr:
            ktmin = 0
            abserr = abseps
            result = reseps
            correc = erlarg
            ertest = max(epsabs, epsrel * abs(reseps))
            if abserr <= ertest:
                break
        if ep.n == 1:
            noext = True
        if ier == 5:
            break
        s.maxerr = s.iord[1]
        s.errmax = s.elist[s.maxerr]
        s.nrmax = 1
        extrap = False
        small *= 0.5
        erlarg = errsum
    if goto_sum:
        result = 0.0
        for k in range(1, last + 1):
            result += s.rlist[k]
        abserr = errsum
        if ier > 2:
            ier -= 1
        return result, abserr, last, ier
    r, e, ie = _final(ier, ierro, result, abserr, area, errsum, correc, ksgn, defabs, s, last)
    return r, e, last, ie

def qagse(f, a: float, b: float, epsabs: float, epsrel: float, limit: int):
    rule = lambda lo, hi: _qk21(f, lo, hi)
    result, abserr, last, ier = _qags_core(rule, a, b, epsabs, epsrel, limit)
    return result, abserr, 42 * last - 21, ier

def qagie(f, bound: float, inf: int, epsabs: float, epsrel: float, limit: int):
    boun = 0.0 if inf == 2 else bound
    rule = lambda lo, hi: _qk15i(f, boun, inf, lo, hi)
    result, abserr, last, ier = _qags_core(rule, 0.0, 1.0, epsabs, epsrel, limit)
    neval = 30 * last - 15
    if inf == 2:
        neval *= 2
    return result, abserr, neval, ier

def qagpe(f, a: float, b: float, pts_in: List[float], epsabs: float, epsrel: float, limit: int):
    # dqagpe; pts_in sorted, strictly inside (a, b), a < b
    _check_tol(epsabs, epsrel)
    npts = len(pts_in)
    npts2 = npts + 2
    if limit < 1:
        raise ValueError("Invalid 'limit' argument. There must be at least one subinterval")
    if npts2 > limit:
        raise ValueError(f"Number of break points ({npts}) must be less than subinterval limit ({limit})")
    s = _QState(max(limit, npts2))
    s.limit = limit
    nint = npts + 1
    pts = [a] + pts_in + [b]
    ndin = [0] * (nint + 2)
    ier = 0
    result = 0.0
    abserr = 0.0
    resabs = 0.0
    a1 = pts[0]
    for i in range(1, nint + 1):
        b1 = pts[i]
        area1, error1, defabs, resa = _qk21(f, a1, b1)
        abserr += error1
        result += area1
        if error1 == resa and error1 != 0.0:
            ndin[i] = 1
        resabs += defabs
        s.level[i] = 0
        s.elist[i] = error1
        s.alist[i] = a1
        s.blist[i] = b1
        s.rlist[i] = area1
        s.iord[i] = i
        a1 = b1
    errsum = 0.0
    for i in range(1, nint + 1):
        if ndin[i] == 1:
            s.elist[i] = abserr
        errsum += s.elist[i]
    last = nint
    dres = abs(result)
    errbnd = max(epsabs, epsrel * dres)
    if abserr <= 100.0 * _EPMACH * resabs and abserr > errbnd:
        ier = 2
    if nint != 1:
        for i in range(1, npts + 1):
            ind1 = s.iord[i]
            k = i
            for j in range(i + 1, nint + 1):
                ind2 = s.iord[j]
                if s.elist[ind1] > s.elist[ind2]:
                    continue
                ind1 = ind2
                k = j
            if ind1 != s.iord[i]:
                s.iord[k] = s.iord[i]
                s.iord[i] = ind1
    if ier != 0 or abserr <= errbnd:
        if ier > 2:
            ier -= 1
        return result, abserr, 21 * last, ier
    ep = _Eps()
    s.maxerr = s.iord[1]
    s.errmax = s.elist[s.maxerr]
    area = result
    s.nrmax = 1
    ep.n = 1
    ep.epstab[1] = result
    ktmin = 0
    extrap = False
    noext = False
    erlarg = errsum
    ertest = errbnd
    levmax = 1
    iroff1 = 0
    iroff2 = 0
    iroff3 = 0
    ierro = 0
    abserr = _OFLOW
    ksgn = -1
    if dres >= (1.0 - 50.0 * _EPMACH) * resabs:
        ksgn = 1
    correc = 0.0
    goto_sum = False
    for lst in range(npts2, limit + 1):
        last = lst
        maxerr = s.maxerr
        levcur = s.level[maxerr] + 1
        a1 = s.alist[maxerr]
        b1 = 0.5 * (s.alist[maxerr] + s.blist[maxerr])
        a2 = b1
        b2 = s.blist[maxerr]
        erlast = s.errmax
        area1, error1, _r1, defab1 = _qk21(f, a1, b1)
        area2, error2, _r2, defab2 = _qk21(f, a2, b2)
        area12 = area1 + area2
        erro12 = error1 + error2
        errsum = errsum + erro12 - s.errmax
        area = area + area12 - s.rlist[maxerr]
        if not (defab1 == error1 or defab2 == error2):
            if not (abs(s.rlist[maxerr] - area12) > 1e-5 * abs(area12) or erro12 < 0.99 * s.errmax):
                if extrap:
                    iroff2 += 1
                else:
                    iroff1 += 1
            if last > 10 and erro12 > s.errmax:
                iroff3 += 1
        s.level[maxerr] = levcur
        s.level[last] = levcur
        s.rlist[maxerr] = area1
        s.rlist[last] = area2
        errbnd = max(epsabs, epsrel * abs(area))
        if iroff1 + iroff2 >= 10 or iroff3 >= 20:
            ier = 2
        if iroff2 >= 5:
            ierro = 3
        if last == limit:
            ier = 1
        if max(abs(a1), abs(b2)) <= (1.0 + 100.0 * _EPMACH) * (abs(a2) + 1000.0 * _UFLOW):
            ier = 4
        if error2 > error1:
            s.alist[maxerr] = a2
            s.alist[last] = a1
            s.blist[last] = b1
            s.rlist[maxerr] = area2
            s.rlist[last] = area1
            s.elist[maxerr] = error2
            s.elist[last] = error1
        else:
            s.alist[last] = a2
            s.blist[maxerr] = b1
            s.blist[last] = b2
            s.elist[maxerr] = error1
            s.elist[last] = error2
        _qpsrt(s, last)
        if errsum <= errbnd:
            goto_sum = True
            break
        if ier != 0:
            break
        if noext:
            continue
        erlarg -= erlast
        if levcur + 1 <= levmax:
            erlarg += erro12
        if not extrap:
            if s.level[s.maxerr] + 1 <= levmax:
                continue
            extrap = True
            s.nrmax = 2
        skip = False
        if ierro != 3 and erlarg > ertest:
            jupbnd = last
            if last > (2 + limit // 2):
                jupbnd = limit + 3 - last
            for _k in range(s.nrmax, jupbnd + 1):
                s.maxerr = s.iord[s.nrmax]
                s.errmax = s.elist[s.maxerr]
                if s.level[s.maxerr] + 1 <= levmax:
                    skip = True
                    break
                s.nrmax += 1
        if skip:
            continue
        ep.n += 1
        ep.epstab[ep.n] = area
        if ep.n > 2:
            reseps, abseps = _qelg(ep)
            ktmin += 1
            if ktmin > 5 and abserr < 1e-3 * errsum:
                ier = 5
            if abseps < abserr:
                ktmin = 0
                abserr = abseps
                result = reseps
                correc = erlarg
                ertest = max(epsabs, epsrel * abs(reseps))
                if abserr < ertest:
                    break
            if ep.n == 1:
                noext = True
            if ier >= 5:
                break
        s.maxerr = s.iord[1]
        s.errmax = s.elist[s.maxerr]
        s.nrmax = 1
        extrap = False
        levmax += 1
        erlarg = errsum
    neval = 21 * nint + 42 * (last - nint)
    if goto_sum:
        result = 0.0
        for k in range(1, last + 1):
            result += s.rlist[k]
        abserr = errsum
        if ier > 2:
            ier -= 1
        return result, abserr, neval, ier
    r, e, ie = _final(ier, ierro, result, abserr, area, errsum, correc, ksgn, resabs, s, last)
    return r, e, neval, ie
