# Ported from scipy 1.18.1 (BSD-3-Clause)

# SLSQP (scipy/optimize/src/slsqp.c, the C translation of Dieter Kraft's
# Fortran code) and the Lawson-Hanson NNLS solver (src/nnls.c), translated
# line for line with the same buffer layout and LAPACK/BLAS kernels.

import math
from numpy.linalg.blas import fint
from ._blaslapack import gemv, tpmv, tpsv, trsv, trsm, dot, nrm2, gerq2, \
    ormr2, geqr2, orm2r, gelsy, larfgp, larf_l, lartgp

_EPSMACH = 2.220446049250313e-16
_NAN = math.nan

def _nextafter_up(u: float) -> float:
    # nextafter(u, 2u) for u > 0
    from C import nextafter(float, float) -> float
    return nextafter(u, 2 * u)

def _nnls(m: int, n: int, a: Ptr[float], b: Ptr[float], x: Ptr[float],
          w: Ptr[float], zz: Ptr[float], indices: Ptr[int], maxiter: int):
    # returns (rnorm, info)
    ii = 0; ip = 0; indz = 0; iteration = 0; iz = 0; izmax = 0
    j = 0; jj = 0; k = 0
    tau = 0.0
    pivot = 1.0
    info = 1
    if m <= 0 or n <= 0:
        return 0.0, 2
    tmp_work = Ptr[float](max(n, 1) + 1)
    for i in range(n):
        indices[i] = i
    for i in range(n):
        x[i] = 0.0
    done = False
    while indz < (m if m < n else n):
        for i in range(indz, n):
            j = indices[i]
            w[j] = dot(m - indz, a + (indz + j * m), b + indz)
        while True:
            wmax = 0.0
            for k in range(indz, n):
                j = indices[k]
                if w[j] > wmax:
                    wmax = w[j]
                    izmax = k
            if wmax <= 0.0:
                done = True
                break
            iz = izmax
            j = indices[iz]
            pivot = a[indz + j * m]
            pv = Ptr[float](1)
            pv[0] = pivot
            tau = larfgp(m - indz, pv, a + (indz + 1 + j * m))
            pivot = pv[0]
            unorm = nrm2(indz, a + j * m) if indz > 0 else 0.0
            spacing = (_nextafter_up(unorm) - unorm) if unorm > 0.0 else 0.0
            if abs(pivot) > 100.0 * spacing:
                for i in range(m):
                    zz[i] = b[i]
                pivot2 = a[indz + j * m]
                a[indz + j * m] = 1.0
                larf_l(m - indz, a + (indz + j * m), tau, zz + indz, m - indz, tmp_work)
                ztest = zz[indz] / pivot
                if ztest > 0.0:
                    break
                else:
                    a[indz + j * m] = pivot2
            w[j] = 0.0
        if done:
            break
        for i in range(m):
            b[i] = zz[i]
        indices[iz] = indices[indz]
        indices[indz] = j
        indz += 1
        if indz < n:
            tmpint = m - indz + 1
            for k in range(indz, n):
                jj = indices[k]
                larf_l(tmpint, a + (indz - 1 + j * m), tau, a + (indz - 1 + jj * m), tmpint, tmp_work)
        a[indz - 1 + j * m] = pivot
        if indz < m:
            for i in range(indz, m):
                a[j * m + i] = 0.0
        w[j] = 0.0
        for k in range(indz):
            ip = indz - 1 - k
            if k != 0:
                for i in range(ip + 1):
                    zz[i] = zz[i] - a[i + jj * m] * zz[ip + 1]
            jj = indices[ip]
            zz[ip] = zz[ip] / a[ip + jj * m]
        while True:
            iteration += 1
            if iteration >= maxiter:
                info = 3
                done = True
                break
            alpha = 2.0
            for ip in range(indz):
                k = indices[ip]
                if zz[ip] <= 0.0:
                    T = -x[k] / (zz[ip] - x[k])
                    if alpha > T:
                        alpha = T
                        jj = ip
            if alpha == 2.0:
                break
            for ip in range(indz):
                k = indices[ip]
                x[k] = x[k] + alpha * (zz[ip] - x[k])
            i = indices[jj]
            while True:
                x[i] = 0.0
                if jj != indz - 1:
                    jj += 1
                    for j2 in range(jj, indz):
                        ii = indices[j2]
                        indices[j2 - 1] = ii
                        cc, ss = lartgp(a + (j2 - 1 + ii * m), a + (j2 + ii * m),
                                        a + (j2 - 1 + ii * m))
                        a[j2 + ii * m] = 0.0
                        for k in range(n):
                            if k != ii:
                                tmp = a[j2 - 1 + k * m]
                                a[j2 - 1 + k * m] = cc * tmp + ss * a[j2 + k * m]
                                a[j2 + k * m] = -ss * tmp + cc * a[j2 + k * m]
                        tmp = b[j2 - 1]
                        b[j2 - 1] = cc * tmp + ss * b[j2]
                        b[j2] = -ss * tmp + cc * b[j2]
                indz -= 1
                indices[indz] = i
                nobreak = False
                jj = 0
                while jj < indz:
                    i = indices[jj]
                    if x[i] <= 0.0:
                        break
                    if jj == indz - 1:
                        nobreak = True
                    jj += 1
                if nobreak:
                    break
            for i2 in range(m):
                zz[i2] = b[i2]
            for k in range(indz):
                ip = indz - 1 - k
                if k != 0:
                    for i2 in range(ip + 1):
                        zz[i2] = zz[i2] - a[i2 + jj * m] * zz[ip + 1]
                jj = indices[ip]
                zz[ip] = zz[ip] / a[ip + jj * m]
        if done:
            break
        for k in range(indz):
            i = indices[k]
            x[i] = zz[k]
    rnorm = 0.0
    if indz < m:
        rnorm = nrm2(m - indz, b + indz)
    else:
        for i in range(n):
            w[i] = 0.0
    return rnorm, info

# ------------------------------------------------------------------- LDP

def _ldp(m: int, n: int, g: Ptr[float], h: Ptr[float], x: Ptr[float],
         buffer: Ptr[float], indices: Ptr[int]):
    # returns (xnorm, mode)
    if n <= 0:
        return 0.0, 2
    for i in range(n):
        x[i] = 0.0
    if m == 0:
        return 0.0, 1
    a = buffer
    b = buffer + m * (n + 1)
    zz = buffer + (m + 1) * (n + 1)
    y = buffer + (m + 2) * (n + 1)
    w = buffer + ((m + 2) * (n + 1) + m)
    for j in range(m):
        for i in range(n):
            a[i + j * (n + 1)] = g[j + i * m]
        a[n + j * (n + 1)] = h[j]
    for i in range(n):
        b[i] = 0.0
    b[n] = 1.0
    rnorm, mode = _nnls(n + 1, m, a, b, y, w, zz, indices, 3 * m)
    if mode != 1:
        return 0.0, mode
    if rnorm <= 0.0:
        return 0.0, 4
    fac = 1.0 - dot(m, h, y)
    if not ((1.0 + fac) - 1.0 > 0.0):
        return 0.0, 4
    fac = 1.0 / fac
    gemv('T', m, n, fac, g, m, y, 0.0, x)
    xnorm = nrm2(n, x)
    for i in range(m):
        buffer[i] = fac * y[i]
    return xnorm, 1

# ------------------------------------------------------------------- LSI

def _lsi(ma: int, mg: int, n: int, a: Ptr[float], b: Ptr[float], g: Ptr[float],
         h: Ptr[float], x: Ptr[float], buffer: Ptr[float], jw: Ptr[int]):
    tmp_int = ma if ma < n else n
    geqr2(ma, n, a, ma, buffer, buffer + tmp_int)
    orm2r('L', 'T', ma, 1, tmp_int, a, ma, buffer, b, ma, buffer + tmp_int)
    for i in range(tmp_int):
        if not (abs(a[i + i * ma]) >= _EPSMACH):
            return 0.0, 5
    trsm('R', 'U', 'N', 'N', mg, n, 1.0, a, ma, g, mg)
    gemv('N', mg, n, -1.0, g, mg, b, 1.0, h)
    xnorm, mode = _ldp(mg, n, g, h, x, buffer, jw)
    if mode != 1:
        return xnorm, mode
    for i in range(n):
        x[i] = x[i] + b[i]
    trsv('U', 'N', 'N', n, a, ma, x)
    tmp_int = ma - n
    off = (ma if n + 1 > ma else n + 1) - 1
    tmp_dbl = nrm2(tmp_int, b + off)
    return math.hypot(xnorm, tmp_dbl), mode

# ------------------------------------------------------------------ LSEI

def _lsei(ma: int, me: int, mg: int, n: int, a: Ptr[float], b: Ptr[float],
          e: Ptr[float], f: Ptr[float], g: Ptr[float], h: Ptr[float],
          x: Ptr[float], buffer: Ptr[float], jw: Ptr[int]):
    # returns (xnorm, mode)
    xnorm = 0.0
    for i in range(n):
        x[i] = 0.0
    if me > n:
        return xnorm, 2
    nvars = n - me
    gmults = buffer
    emults = buffer + mg
    wb = buffer + (me + mg)
    tau = buffer + (me + mg + ma)
    a2 = buffer + (mg + 2 * me + ma)
    g2 = buffer + (mg + 2 * me + ma + ma * nvars)
    lsi_scratch = buffer + (mg + 2 * me + ma + (ma + mg) * nvars)
    lde = me if me > 0 else 1
    ldg = mg if mg > 0 else 1
    gerq2(me, n, e, lde, tau, lsi_scratch)
    ormr2('R', 'T', ma, n, me, e, lde, tau, a, ma, lsi_scratch)
    ormr2('R', 'T', mg, n, me, e, lde, tau, g, ldg, lsi_scratch)
    for i in range(me):
        if not (abs(e[i + (nvars + i) * me]) >= _EPSMACH):
            return xnorm, 6
    for i in range(me):
        x[nvars + i] = f[i]
    trsv('U', 'N', 'N', me, e + nvars * me, lde, x + nvars)
    mode = 1
    for i in range(mg):
        gmults[i] = 0.0
    if me != n:
        for i in range(ma):
            wb[i] = b[i]
        gemv('N', ma, me, -1.0, a + ma * nvars, ma, x + nvars, 1.0, wb)
        for j in range(nvars):
            for i in range(ma):
                a2[i + j * ma] = a[i + j * ma]
            for i in range(mg):
                g2[i + j * mg] = g[i + j * mg]
        if mg == 0:
            lwork = ma * nvars + 3 * nvars + 1
            wb_orig = lsi_scratch + lwork
            for i in range(ma):
                wb_orig[i] = wb[i]
            jp = Ptr[fint](max(nvars, 1))
            for i in range(nvars):
                jp[i] = fint(0)
            krank = gelsy(ma, nvars, a2, ma, wb, ma, jp, math.sqrt(_EPSMACH),
                          lsi_scratch, lwork)
            for i in range(nvars):
                x[i] = wb[i]
            gemv('N', ma, nvars, 1.0, a, ma, x, -1.0, wb_orig)
            xnorm = nrm2(ma, wb_orig)
            if krank < nvars:
                return xnorm, 7
            mode = 1
        else:
            gemv('N', mg, me, -1.0, g + mg * nvars, ldg, x + nvars, 1.0, h)
            xnorm, mode = _lsi(ma, mg, nvars, a2, wb, g2, h, x, lsi_scratch, jw)
            for i in range(mg):
                gmults[i] = lsi_scratch[i]
            if me == 0:
                return xnorm, mode
            t = nrm2(me, x + nvars)
            xnorm = math.hypot(xnorm, t)
            if mode != 1:
                return xnorm, mode
    # ORIGINAL_BASIS
    gemv('N', ma, n, 1.0, a, ma, x, -1.0, b)
    gemv('T', ma, me, 1.0, a + nvars * ma, ma, b, 0.0, f)
    gemv('T', mg, me, -1.0, g + nvars * mg, ldg, gmults, 1.0, f)
    ormr2('L', 'T', n, 1, me, e, lde, tau, x, n, lsi_scratch)
    for i in range(me):
        emults[i] = f[i]
    trsv('U', 'T', 'N', me, e + (n - me) * me, lde, emults)
    return xnorm, mode

# ------------------------------------------------------------------- LSQ

def _isnan(v: float) -> bool:
    return v != v

def _lsq(m: int, meq: int, n: int, augment: bool, aug_weight: float,
         Lf: Ptr[float], gradx: Ptr[float], C: Ptr[float], d: Ptr[float],
         xl: Ptr[float], xu: Ptr[float], x: Ptr[float], y: Ptr[float],
         buffer: Ptr[float], jw: Ptr[int]) -> int:
    orign = n
    mineq = m - meq
    cursor = 0
    ld = n
    if augment:
        ld = n + 1
        x[n] = 1.0
        xl[n] = 0.0
        xu[n] = 1.0
    for i in range((ld + 2) * ld):
        buffer[i] = 0.0
    wA = buffer
    wb = buffer + ld * (ld + 1)
    for j in range(n):
        diag = math.sqrt(Lf[cursor])
        cursor += 1
        wA[j + j * ld] = diag
        for i in range(j + 1, n):
            wA[j + i * ld] = Lf[cursor] * diag
            cursor += 1
    for i in range(n):
        wb[i] = gradx[i]
    tpsv('L', 'N', 'U', n, Lf, wb)
    cursor = 0
    for i in range(n):
        wb[i] /= -math.sqrt(Lf[cursor])
        cursor += n - i
    if augment:
        wA[ld * ld - 1] = aug_weight
        n += 1
    wE = buffer + (n * (n + 1) + n)
    wf = buffer + (n * (n + 1) + n + n * meq)
    if meq > 0:
        for j in range(n - 1):
            for i in range(meq):
                wE[i + j * meq] = C[i + j * m]
        if augment:
            for i in range(meq):
                wE[i + (n - 1) * meq] = -d[i]
        else:
            for i in range(meq):
                wE[i + (n - 1) * meq] = C[i + (n - 1) * m]
        for i in range(meq):
            wf[i] = -d[i]
    wG = buffer + (n * (n + 1) + n + n * meq + meq)
    wh = buffer + (n * (n + 1) + n + n * meq + meq + (mineq + 2 * n) * ld)
    for i in range((mineq + 2 * n) * (ld + 1)):
        wG[i] = 0.0
    nancount = 0
    nrow = mineq
    if m > meq:
        for i in range(mineq):
            wh[i] = -d[meq + i]
    for i in range(n):
        if _isnan(xl[i]):
            nancount += 1
        else:
            wh[nrow] = xl[i]
            nrow += 1
    for i in range(n):
        if _isnan(xu[i]):
            nancount += 1
        else:
            wh[nrow] = -xu[i]
            nrow += 1
    n_wG_rows = mineq + 2 * n - nancount
    if m > meq:
        for j in range(orign):
            for i in range(mineq):
                wG[i + j * n_wG_rows] = C[meq + i + j * m]
    if augment:
        for i in range(mineq):
            wG[i + orign * n_wG_rows] = max(-d[meq + i], 0.0)
    nrow = mineq
    for i in range(n):
        if not _isnan(xl[i]):
            wG[nrow + i * n_wG_rows] = 1.0
            nrow += 1
    for i in range(n):
        if not _isnan(xu[i]):
            wG[nrow + i * n_wG_rows] = -1.0
            nrow += 1
    lsei_scratch = wh + (mineq + 2 * n)
    xnorm, mode = _lsei(ld, meq, n_wG_rows, n, wA, wb, wE, wf, wG, wh, x,
                        lsei_scratch, jw)
    if mode == 1:
        for i in range(meq):
            y[i] = lsei_scratch[i + n_wG_rows]
        for i in range(mineq):
            y[meq + i] = lsei_scratch[i]
        for i in range(2 * n):
            y[m + i] = _NAN
    for i in range(n):
        if (not _isnan(xl[i])) and x[i] < xl[i]:
            x[i] = xl[i]
        elif (not _isnan(xu[i])) and x[i] > xu[i]:
            x[i] = xu[i]
    return mode

# ------------------------------------------------------------ ldl_update

def _ldl_update(n: int, a: Ptr[float], z: Ptr[float], sigma: float, w: Ptr[float]):
    ij = 0
    if sigma == 0.0:
        return
    t = 1.0 / sigma
    if sigma <= 0.0:
        for i in range(n):
            w[i] = z[i]
        for i in range(n):
            v = w[i]
            t = t + v * v / a[ij]
            for j in range(i + 1, n):
                ij += 1
                w[j] = w[j] - v * a[ij]
            ij += 1
        if t >= 0.0:
            t = _EPSMACH / sigma
        for i in range(n):
            j = n - i - 1
            ij -= i + 1
            u = w[j]
            w[j] = t
            t = t - u * u / a[ij]
    for i in range(n):
        v = z[i]
        delta = v / a[ij]
        tp = w[i] if sigma < 0.0 else t + delta * v
        alpha = tp / t
        a[ij] = alpha * a[ij]
        if i == n - 1:
            return
        beta = delta / tp
        if alpha <= 4.0:
            for j in range(i + 1, n):
                ij += 1
                z[j] = z[j] - v * a[ij]
                a[ij] = a[ij] + beta * z[j]
        else:
            gamma = t / tp
            for j in range(i + 1, n):
                ij += 1
                u = a[ij]
                a[ij] = gamma * u + beta * z[j]
                z[j] = z[j] - v * u
        ij += 1
        t = tp

# ------------------------------------------------------------ SLSQP body

class _SLSQPVars:
    acc: float
    alpha: float
    f0: float
    gs: float
    h1: float
    h2: float
    h3: float
    h4: float
    t: float
    t0: float
    tol: float
    exact: int
    inconsistent: int
    reset: int
    iter: int
    itermax: int
    line: int
    m: int
    meq: int
    mode: int
    n: int

    def __init__(self, acc: float, itermax: int, m: int, meq: int, n: int):
        self.acc = acc
        self.alpha = 0.0
        self.f0 = 0.0
        self.gs = 0.0
        self.h1 = 0.0
        self.h2 = 0.0
        self.h3 = 0.0
        self.h4 = 0.0
        self.t = 0.0
        self.t0 = 0.0
        self.tol = 10.0 * acc
        self.exact = 0
        self.inconsistent = 0
        self.reset = 0
        self.iter = 0
        self.itermax = itermax
        self.line = 0
        self.m = m
        self.meq = meq
        self.mode = 0
        self.n = n

_L_MODE0 = 0
_L_RESET = 1
_L_ITER = 2
_L_LINE = 3
_L_MODE1 = 4
_L_255 = 5
_L_MODEM1 = 6

def _violation(S: _SLSQPVars, d: Ptr[float]) -> float:
    h = 0.0
    for j in range(S.m):
        hj = d[j] if j < S.meq else 0.0
        h = h + max(-d[j], hj)
    return h

def _slsqp_body(S: _SLSQPVars, funx: float, gradx: Ptr[float], C: Ptr[float],
                d: Ptr[float], sol: Ptr[float], mult: Ptr[float],
                xl: Ptr[float], xu: Ptr[float], buffer: Ptr[float],
                indices: Ptr[int]):
    lda = S.m if S.m > 0 else 1
    alfmin = 0.1
    n = S.n
    m = S.m
    n1 = n + 1
    n2 = n1 * n // 2
    bfgs = buffer
    x0 = buffer + n2
    mu = buffer + (n2 + n)
    s = buffer + (n2 + n + m)
    u = buffer + (n2 + n + m + n1)
    v = buffer + (n2 + n + m + n1 + n1)
    lsq_buffer = buffer + (n2 + n + m + n1 + n1 + n1)
    badlin = False
    pc = _L_MODE0
    if S.mode == -1:
        pc = _L_MODEM1
    elif S.mode == 1:
        pc = _L_MODE1
    while True:
        if pc == _L_MODE0:
            S.exact = 0
            S.acc = abs(S.acc)
            S.tol = 10 * S.acc
            S.iter = 0
            S.reset = 0
            for i in range(n):
                s[i] = 0.0
            for i in range(m):
                mu[i] = 0.0
            pc = _L_RESET
        elif pc == _L_RESET:
            S.reset += 1
            if S.reset > 5:
                pc = _L_255
                continue
            for i in range(n2):
                bfgs[i] = 0.0
            j = 0
            for i in range(n):
                bfgs[j] = 1.0
                j += n - i
            pc = _L_ITER
        elif pc == _L_ITER:
            S.mode = 9
            if S.iter >= S.itermax:
                return
            S.iter += 1
            for i in range(n):
                u[i] = -sol[i] + xl[i]
                v[i] = -sol[i] + xu[i]
            S.h4 = 1.0
            S.mode = _lsq(m, S.meq, n, False, 0.0, bfgs, gradx, C, d, u, v, s,
                          mult, lsq_buffer, indices)
            badlin = False
            if S.mode == 6 and n == S.meq:
                S.mode = 4
            if S.mode == 4:
                badlin = True
                for i in range(n):
                    s[i] = 0.0
                S.h3 = 0.0
                rho = 100.0
                S.inconsistent = 0
                while True:
                    S.mode = _lsq(m, S.meq, n, True, rho, bfgs, gradx, C, d, u,
                                  v, s, mult, lsq_buffer, indices)
                    S.h4 = 1.0 - s[n]
                    if S.mode == 4:
                        rho *= 10.0
                        S.inconsistent += 1
                        if S.inconsistent > 5:
                            return
                        continue
                    elif S.mode != 1:
                        return
                    break
            elif S.mode != 1:
                return
            for i in range(n):
                v[i] = gradx[i]
            gemv('T', m, n, -1.0, C, lda, mult, 1.0, v)
            S.f0 = funx
            for i in range(n):
                x0[i] = sol[i]
            S.gs = dot(n, gradx, s)
            S.h1 = abs(S.gs)
            S.h2 = 0.0
            for j in range(m):
                S.h3 = d[j] if j < S.meq else 0.0
                S.h2 = S.h2 + max(-d[j], S.h3)
                S.h3 = abs(mult[j])
                mu[j] = max(S.h3, (mu[j] + S.h3) / 2.0)
                S.h1 = S.h1 + S.h3 * abs(d[j])
            S.mode = 0
            if S.h1 < S.acc and S.h2 < S.acc and (not badlin) and funx == funx:
                return
            S.h1 = 0.0
            for j in range(m):
                S.h3 = d[j] if j < S.meq else 0.0
                S.h1 += mu[j] * max(-d[j], S.h3)
            S.t0 = funx + S.h1
            S.h3 = S.gs - S.h1 * S.h4
            S.mode = 8
            if S.h3 >= 0.0:
                pc = _L_RESET
                continue
            S.line = 0
            S.alpha = 1.0
            pc = _L_LINE
        elif pc == _L_LINE:
            S.line += 1
            S.h3 = S.alpha * S.h3
            for i in range(n):
                s[i] = S.alpha * s[i]
            for i in range(n):
                sol[i] = x0[i]
            for i in range(n):
                sol[i] = sol[i] + s[i]
            S.mode = 1
            return
        elif pc == _L_MODE1:
            S.t = funx
            for j in range(m):
                S.h1 = d[j] if j < S.meq else 0.0
                S.t = S.t + mu[j] * max(-d[j], S.h1)
            S.h1 = S.t - S.t0
            if not (S.h1 <= S.h3 / 10.0 or S.line > 10):
                S.alpha = max(S.h3 / (2.0 * (S.h3 - S.h1)), alfmin)
                pc = _L_LINE
                continue
            S.h3 = _violation(S, d)
            if (abs(funx - S.f0) < S.acc or nrm2(n, s) < S.acc) and \
                    S.h3 < S.acc and (not badlin) and funx == funx:
                S.mode = 0
            else:
                S.mode = -1
            return
        elif pc == _L_255:
            S.h3 = _violation(S, d)
            if (abs(funx - S.f0) < S.tol or nrm2(n, s) < S.tol) and \
                    S.h3 < S.tol and (not badlin) and funx == funx:
                S.mode = 0
            else:
                S.mode = 8
            return
        else:
            for i in range(n):
                u[i] = gradx[i]
            gemv('T', m, n, -1.0, C, lda, mult, 1.0, u)
            for i in range(n):
                u[i] = u[i] - v[i]
            for i in range(n):
                v[i] = s[i]
            tpmv('L', 'T', 'U', n, bfgs, v)
            j = 0
            for i in range(n):
                v[i] = bfgs[j] * v[i]
                j += n - i
            tpmv('L', 'N', 'U', n, bfgs, v)
            S.h1 = dot(n, s, u)
            S.h2 = dot(n, s, v)
            S.h3 = 0.2 * S.h2
            if S.h1 < S.h3:
                S.h4 = (S.h2 - S.h3) / (S.h2 - S.h1)
                S.h1 = S.h3
                tmp_dbl = 1.0 - S.h4
                for i in range(n):
                    u[i] = S.h4 * u[i]
                for i in range(n):
                    u[i] = u[i] + tmp_dbl * v[i]
            if S.h1 == 0.0 or S.h2 == 0.0:
                pc = _L_RESET
                continue
            _ldl_update(n, bfgs, u, 1.0 / S.h1, v)
            _ldl_update(n, bfgs, v, -1.0 / S.h2, u)
            pc = _L_ITER

# ------------------------------------------------------- Python driver

import numpy as np
from ._common import OptimizeResult, _opt_float, _opt_int, _has_opt, \
    _epsilon, INF
from ._common import _ScalarFunction
from ._cons import _eval_constraints
from ._callback import _cb_halt, _STOP_MSG

_SLSQP_MODES = ('Gradient evaluation required (g & a)',
                'Optimization terminated successfully',
                'Function evaluation required (f & c)',
                'More equality constraints than independent variables',
                'More than 3*n iterations in LSQ subproblem',
                'Inequality constraints incompatible',
                'Singular matrix E in LSQ subproblem',
                'Singular matrix C in LSQ subproblem',
                'Rank-deficient equality constraint subproblem HFTI',
                'Positive directional derivative for linesearch',
                'Iteration limit reached')

def _slsqp_msg(mode: int) -> str:
    return _SLSQP_MODES[mode + 1]

class _Clip[F]:
    # _clip_x_for_func
    f: F
    lb: np.ndarray[float, 1]
    ub: np.ndarray[float, 1]

    def __init__(self, f: F, lb: np.ndarray[float, 1], ub: np.ndarray[float, 1]):
        self.f = f
        self.lb = lb
        self.ub = ub

    def x(self, x: np.ndarray[float, 1]) -> np.ndarray[float, 1]:
        out = False
        for i in range(x.size):
            if x[i] < self.lb[i] or x[i] > self.ub[i]:
                out = True
        if out:
            return np.clip(x, self.lb, self.ub)
        return x

def _fill_cons(constraints, x: np.ndarray[float, 1], C: np.ndarray[float, 2],
               d: np.ndarray[float, 1], want_jac: bool, eps: Optional[float],
               fd: str, rel: Optional[float], lb: np.ndarray[float, 1],
               ub: np.ndarray[float, 1]):
    # cjac: approx_derivative(method=jac, rel_step) when jac is a string,
    # else 2-point with abs_step=epsilon; x clipped to the bounds
    xc = np.clip(x, lb, ub) if want_jac else x
    e, q, E, Q = _eval_constraints(constraints, xc, want_jac, eps, fd, rel, lb, ub)
    meq = e.size
    if want_jac:
        for i in range(meq):
            for j in range(x.size):
                C[i, j] = E[i, j]
        for i in range(q.size):
            for j in range(x.size):
                C[meq + i, j] = Q[i, j]
    else:
        for i in range(meq):
            d[i] = e[i]
        for i in range(q.size):
            d[meq + i] = q[i]

def _minimize_slsqp(func, x0: np.ndarray[float, 1], args, jac, bnds, constraints,
                    callback, options):
    maxiter = _opt_int(options, 'maxiter', 100)
    acc = _opt_float(options, 'ftol', 1e-06)
    eps = _opt_float(options, 'eps', _epsilon)
    frs: Optional[float] = None
    if _has_opt(options, 'finite_diff_rel_step'):
        frs = _opt_float(options, 'finite_diff_rel_step', 0.0)
    n = x0.size
    blo = np.full(n, -INF)
    bup = np.full(n, INF)
    has_bounds = bnds is not None
    if bnds is not None:
        blo, bup = bnds
    x = np.clip(x0, blo, bup)
    fd = '2-point'
    ceps: Optional[float] = eps
    crel: Optional[float] = None
    if isinstance(jac, str):
        fd = jac
        ceps = None
        crel = frs
    e0, q0, _, _ = _eval_constraints(constraints, x, False)
    meq = e0.size
    mieq = q0.size
    m = meq + mieq
    xl = np.empty(n)
    xu = np.empty(n)
    for i in range(n):
        xl[i] = blo[i] if (has_bounds and abs(blo[i]) != INF) else _NAN
        xu[i] = bup[i] if (has_bounds and abs(bup[i]) != INF) else _NAN
        if has_bounds and blo[i] > bup[i]:
            raise ValueError("SLSQP Error: lb > ub in bounds.")
    if isinstance(jac, str):
        sf = _ScalarFunction(func, x, args, jac, None, frs, blo, bup)
    else:
        sf = _ScalarFunction(func, x, args, jac, eps, frs, blo, bup)
    clip = _Clip(0, blo, bup)
    S = _SLSQPVars(acc, maxiter, m, meq, n)
    indices = Ptr[int](max(m + 2 * n + 2, 1))
    for i in range(max(m + 2 * n + 2, 1)):
        indices[i] = 0
    buffer_size = n * (n + 1) // 2 + 3 * m * n - (m + 5 * n + 7) * meq + \
        9 * m + 8 * n * n + 35 * n + meq * meq + 28
    if mieq == 0:
        buffer_size += 2 * n * (n + 1)
    bs = max(buffer_size, 1)
    buffer = Ptr[float](bs)
    for i in range(bs):
        buffer[i] = 0.0
    xs = clip.x(x)
    fx = sf.fun(xs)
    g = sf.grad(xs).copy()
    nm = max(1, m + 2 * n + 2)
    mult = Ptr[float](nm)
    for i in range(nm):
        mult[i] = 0.0
    mm = max(1, m)
    # column-major C (Fortran order) for the C kernels
    Cm = np.zeros((mm, n))
    dv = np.zeros(mm)
    _fill_cons(constraints, x, Cm, dv, True, ceps, fd, crel, blo, bup)
    _fill_cons(constraints, x, Cm, dv, False, ceps, fd, crel, blo, bup)
    Cp = Ptr[float](mm * n)
    dp = Ptr[float](mm)
    xp = Ptr[float](n + 1)
    xlp = Ptr[float](n + 1)
    xup = Ptr[float](n + 1)
    gp = Ptr[float](n + 1)
    for i in range(n):
        xlp[i] = xl[i]
        xup[i] = xu[i]
    iter_prev = 0
    stopped = False
    while True:
        for j in range(n):
            for i in range(mm):
                Cp[i + j * mm] = Cm[i, j]
        for i in range(mm):
            dp[i] = dv[i]
        for i in range(n):
            xp[i] = x[i]
            gp[i] = g[i]
        _slsqp_body(S, fx, gp, Cp, dp, xp, mult, xlp, xup, buffer, indices)
        x = np.empty(n)
        for i in range(n):
            x[i] = xp[i]
        if S.mode == 1:
            fx = sf.fun(clip.x(x))
            _fill_cons(constraints, x, Cm, dv, False, ceps, fd, crel, blo, bup)
        if S.mode == -1:
            g = sf.grad(clip.x(x)).copy()
            _fill_cons(constraints, x, Cm, dv, True, ceps, fd, crel, blo, bup)
        if S.iter > iter_prev:
            if _cb_halt(callback, x, fx):
                stopped = True
                break
        if abs(S.mode) != 1:
            break
        iter_prev = S.iter
    r = OptimizeResult(x, fx, g, nit=S.iter, nfev=sf.nfev, njev=sf.ngev,
                       status=S.mode, message=_slsqp_msg(S.mode),
                       success=(S.mode == 0))
    mu = np.empty(m)
    for i in range(m):
        mu[i] = mult[i]
    r.multipliers = mu
    if stopped:
        r.success = False
        r.status = 99
        r.message = _STOP_MSG
    return r
