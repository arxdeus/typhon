# Ported from scipy 1.18.1 (BSD-3-Clause)

# MINPACK: hybrd/hybrj (Powell hybrid method) and lmdif/lmder
# (Levenberg-Marquardt), translated from scipy's C version (minpack.c).
# Matrices are flat column-major arrays with leading dimension ld.

import math
import numpy as np
from ._common import *
from ._common import (_ScalarFunction, _approx_grad, _approx_jac, _bound_val, _call_scalar, _call_vec, _e1, _e2, _eps_for_method, _epsilon, _fd_steps, _has_opt, _norm2, _norm_inf, _old_bound_to_new, _opt_bool, _opt_float, _opt_int, _prepare_bounds_pair, _scalar_eps, _sign, _sqrt, _standardize_bounds, _status_message_maxfev, _status_message_maxiter, _status_message_nan, _status_message_oob, _status_message_pr_loss, _status_message_success, _vec)

_DP0 = 2.220446049250313e-16
_DP1 = 2.2250738585072014e-308
_DP2 = 1.7976931348623157e+308

def _enorm(n: int, x: np.ndarray[float, 1], off: int = 0) -> float:
    rgiant = 2.0 ** 63.5
    rdwarf = 0.5 / rgiant
    s1 = 0.0
    s2 = 0.0
    s3 = 0.0
    x1max = 0.0
    x3max = 0.0
    agiant = rgiant / n
    for i in range(n):
        xabs = abs(x[off + i])
        if xabs <= rdwarf or xabs >= agiant:
            if xabs > rdwarf:
                if xabs > x1max:
                    s1 = 1.0 + s1 * (x1max / xabs) ** 2
                    x1max = xabs
                else:
                    s1 += (xabs / x1max) ** 2
            else:
                if xabs > x3max:
                    s3 = 1.0 + s3 * (x3max / xabs) ** 2
                    x3max = xabs
                else:
                    if xabs != 0.0:
                        s3 += (xabs / x3max) ** 2
        else:
            s2 += xabs * xabs
    if s1 != 0.0:
        return x1max * math.sqrt(s1 + (s2 / x1max) / x1max)
    if s2 != 0.0:
        if s2 >= x3max:
            return math.sqrt(s2 * (1.0 + (x3max / s2) * (x3max * s3)))
        else:
            return math.sqrt(x3max * ((s2 / x3max) + (x3max * s3)))
    return x3max * math.sqrt(s3)

def _qrfac(m: int, n: int, a: np.ndarray[float, 1], lda: int, pivot: bool,
           ipvt: np.ndarray[int, 1], rdiag: np.ndarray[float, 1],
           acnorm: np.ndarray[float, 1], wa: np.ndarray[float, 1]):
    for j in range(n):
        acnorm[j] = _enorm(m, a, j * lda)
        rdiag[j] = acnorm[j]
        wa[j] = rdiag[j]
        if pivot:
            ipvt[j] = j
    minmn = min(m, n)
    for j in range(minmn):
        if pivot:
            kmax = j
            for k in range(j, n):
                if rdiag[k] > rdiag[kmax]:
                    kmax = k
            if kmax != j:
                for i in range(m):
                    temp = a[i + lda * j]
                    a[i + lda * j] = a[i + lda * kmax]
                    a[i + lda * kmax] = temp
                rdiag[kmax] = rdiag[j]
                wa[kmax] = wa[j]
                k = ipvt[j]
                ipvt[j] = ipvt[kmax]
                ipvt[kmax] = k
        ajnorm = _enorm(m - j, a, lda * j + j)
        if ajnorm == 0.0:
            rdiag[j] = -ajnorm
            continue
        if a[lda * j + j] < 0.0:
            ajnorm = -ajnorm
        for i in range(j, m):
            a[i + lda * j] /= ajnorm
        a[j + lda * j] += 1.0
        if j == n - 1:
            rdiag[j] = -ajnorm
            continue
        for k in range(j + 1, n):
            ssum = 0.0
            for i in range(j, m):
                ssum += a[i + lda * j] * a[i + lda * k]
            temp = ssum / a[j + lda * j]
            for i in range(j, m):
                a[i + lda * k] -= temp * a[i + lda * j]
            if pivot and rdiag[k] != 0.0:
                temp = a[j + lda * k] / rdiag[k]
                rdiag[k] *= math.sqrt(max(0.0, 1.0 - temp * temp))
                if 0.05 * (rdiag[k] / wa[k]) ** 2 <= _DP0:
                    rdiag[k] = _enorm(m - j - 1, a, (j + 1) + lda * k)
                    wa[k] = rdiag[k]
        rdiag[j] = -ajnorm

def _qform(m: int, n: int, q: np.ndarray[float, 1], ldq: int,
           wa: np.ndarray[float, 1]):
    minmn = min(m, n)
    if minmn > 1:
        for j in range(1, minmn):
            for i in range(j):
                q[i + j * ldq] = 0.0
    if m >= n + 1:
        for j in range(n, m):
            for i in range(m):
                q[i + j * ldq] = 0.0
            q[j + j * ldq] = 1.0
    for l in range(minmn):
        k = minmn - l - 1
        for i in range(k, m):
            wa[i] = q[i + ldq * k]
            q[i + ldq * k] = 0.0
        q[k + ldq * k] = 1.0
        if wa[k] != 0.0:
            for j in range(k, m):
                ssum = 0.0
                for i in range(k, m):
                    ssum += q[i + ldq * j] * wa[i]
                temp = ssum / wa[k]
                for i in range(k, m):
                    q[i + ldq * j] -= temp * wa[i]

def _qrsolv(n: int, r: np.ndarray[float, 1], ldr: int, ipvt: np.ndarray[int, 1],
            diag: np.ndarray[float, 1], qtb: np.ndarray[float, 1],
            x: np.ndarray[float, 1], sdiag: np.ndarray[float, 1],
            wa: np.ndarray[float, 1]):
    for j in range(n):
        for i in range(j, n):
            r[i + j * ldr] = r[j + i * ldr]
        x[j] = r[j + j * ldr]
        wa[j] = qtb[j]
    for j in range(n):
        l = ipvt[j]
        if diag[l] != 0.0:
            for k in range(j, n):
                sdiag[k] = 0.0
            sdiag[j] = diag[l]
            qtbpj = 0.0
            for k in range(j, n):
                if sdiag[k] != 0.0:
                    ccos = 0.0
                    ssin = 0.0
                    if abs(r[ldr * k + k]) < abs(sdiag[k]):
                        ccotan = r[ldr * k + k] / sdiag[k]
                        ssin = 0.5 / math.sqrt(0.25 + 0.25 * ccotan ** 2)
                        ccos = ssin * ccotan
                    else:
                        ttan = sdiag[k] / r[ldr * k + k]
                        ccos = 0.5 / math.sqrt(0.25 + 0.25 * ttan ** 2)
                        ssin = ccos * ttan
                    r[ldr * k + k] *= ccos
                    r[ldr * k + k] += ssin * sdiag[k]
                    temp = ccos * wa[k] + ssin * qtbpj
                    qtbpj = -ssin * wa[k] + ccos * qtbpj
                    wa[k] = temp
                    if k != n - 1:
                        for i in range(k + 1, n):
                            temp = ccos * r[i + ldr * k] + ssin * sdiag[i]
                            sdiag[i] = -ssin * r[i + ldr * k] + ccos * sdiag[i]
                            r[i + ldr * k] = temp
        sdiag[j] = r[ldr * j + j]
        r[ldr * j + j] = x[j]
    nsing = n
    for j in range(n):
        if sdiag[j] == 0.0 and nsing == n:
            nsing = j
        if nsing < n:
            wa[j] = 0.0
    if nsing > 0:
        for k in range(1, nsing + 1):
            j = nsing - k
            ssum = 0.0
            for i in range(j + 1, nsing):
                ssum += r[i + j * ldr] * wa[i]
            wa[j] = (wa[j] - ssum) / sdiag[j]
    for j in range(n):
        l = ipvt[j]
        x[l] = wa[j]

def _lmpar(n: int, r: np.ndarray[float, 1], ldr: int, ipvt: np.ndarray[int, 1],
           diag: np.ndarray[float, 1], qtb: np.ndarray[float, 1], delta: float,
           par: float, x: np.ndarray[float, 1], sdiag: np.ndarray[float, 1],
           wa1: np.ndarray[float, 1], wa2: np.ndarray[float, 1]) -> float:
    dwarf = _DP1
    nsing = n
    for j in range(n):
        wa1[j] = qtb[j]
        if r[j + j * ldr] == 0.0 and nsing == n:
            nsing = j
        if nsing < n:
            wa1[j] = 0.0
    if nsing > 0:
        for k in range(1, nsing + 1):
            j = nsing - k
            wa1[j] /= r[j + j * ldr]
            temp = wa1[j]
            for i in range(j):
                wa1[i] -= r[i + j * ldr] * temp
    for j in range(n):
        l = ipvt[j]
        x[l] = wa1[j]
    iter = 0
    for j in range(n):
        wa2[j] = diag[j] * x[j]
    dxnorm = _enorm(n, wa2)
    fp = dxnorm - delta
    if fp <= 0.1 * delta:
        return 0.0
    parl = 0.0
    if nsing == n:
        for j in range(n):
            l = ipvt[j]
            wa1[j] = diag[l] * (wa2[l] / dxnorm)
        for j in range(n):
            ssum = 0.0
            for i in range(j):
                ssum += r[i + j * ldr] * wa1[i]
            wa1[j] = (wa1[j] - ssum) / r[j + j * ldr]
        temp = _enorm(n, wa1)
        parl = ((fp / delta) / temp) / temp
    for j in range(n):
        ssum = 0.0
        for i in range(j + 1):
            ssum += r[i + j * ldr] * qtb[i]
        l = ipvt[j]
        wa1[j] = ssum / diag[l]
    gnorm = _enorm(n, wa1)
    paru = gnorm / delta
    if paru == 0.0:
        paru = dwarf / min(delta, 0.1)
    par = min(max(par, parl), paru)
    if par == 0.0:
        par = gnorm / dxnorm
    while True:
        iter += 1
        if par == 0.0:
            par = max(dwarf, paru * 0.001)
        temp = math.sqrt(par)
        for j in range(n):
            wa1[j] = temp * diag[j]
        _qrsolv(n, r, ldr, ipvt, wa1, qtb, x, sdiag, wa2)
        for j in range(n):
            wa2[j] = diag[j] * x[j]
        dxnorm = _enorm(n, wa2)
        temp = fp
        fp = dxnorm - delta
        if (abs(fp) <= 0.1 * delta) or (parl == 0.0 and fp <= temp and temp < 0.0) or iter == 10:
            return par
        for j in range(n):
            l = ipvt[j]
            wa1[j] = diag[l] * (wa2[l] / dxnorm)
        for j in range(n):
            wa1[j] /= sdiag[j]
            temp = wa1[j]
            if j != n - 1:
                for i in range(j + 1, n):
                    wa1[i] -= r[i + j * ldr] * temp
        temp = _enorm(n, wa1)
        parc = ((fp / delta) / temp) / temp
        if fp > 0.0:
            parl = max(parl, par)
        if fp < 0.0:
            paru = min(paru, par)
        par = max(parl, par + parc)
    return par

def _dogleg(n: int, r: np.ndarray[float, 1], diag: np.ndarray[float, 1],
            qtb: np.ndarray[float, 1], delta: float, x: np.ndarray[float, 1],
            wa1: np.ndarray[float, 1], wa2: np.ndarray[float, 1]):
    epsmch = _DP0
    jj = (n * (n + 1)) // 2
    for k in range(n):
        j = n - k - 1
        jj -= k + 1
        l = jj + 1
        ssum = 0.0
        if n - 1 > j:
            for i in range(j + 1, n):
                ssum += r[l] * x[i]
                l += 1
        temp = r[jj]
        if temp == 0.0:
            l = j
            for i in range(j + 1):
                temp = max(temp, abs(r[l]))
                l += n - i - 1
            temp = epsmch * temp
            if temp == 0.0:
                temp = epsmch
        x[j] = (qtb[j] - ssum) / temp
    for j in range(n):
        wa1[j] = 0.0
        wa2[j] = diag[j] * x[j]
    qnorm = _enorm(n, wa2)
    if qnorm <= delta:
        return
    l = 0
    for j in range(n):
        temp = qtb[j]
        for i in range(j, n):
            wa1[i] += r[l] * temp
            l += 1
        wa1[j] /= diag[j]
    gnorm = _enorm(n, wa1)
    sgnorm = 0.0
    alpha = delta / qnorm
    if gnorm != 0.0:
        for j in range(n):
            wa1[j] /= gnorm
            wa1[j] /= diag[j]
        l = 0
        for j in range(n):
            ssum = 0.0
            for i in range(j, n):
                ssum += r[l] * wa1[i]
                l += 1
            wa2[j] = ssum
        temp = _enorm(n, wa2)
        sgnorm = (gnorm / temp) / temp
        alpha = 0.0
        if sgnorm < delta:
            bnorm = _enorm(n, qtb)
            temp = (bnorm / gnorm) * (bnorm / qnorm) * (sgnorm / delta)
            temp = (temp - (delta / qnorm) * (sgnorm / delta) ** 2
                    + math.sqrt((temp - (delta / qnorm)) ** 2
                                + ((1.0 - (delta / qnorm) ** 2)
                                   * (1.0 - (sgnorm / delta) ** 2))))
            alpha = ((delta / qnorm) * (1.0 - (sgnorm / delta) ** 2)) / temp
    temp = (1.0 - alpha) * min(sgnorm, delta)
    for j in range(n):
        x[j] = temp * wa1[j] + alpha * x[j]

def _r1mpyq(m: int, n: int, a: np.ndarray[float, 1], lda: int,
            v: np.ndarray[float, 1], w: np.ndarray[float, 1]):
    if n < 2:
        return
    for k in range(n - 1):
        j = n - k - 2
        ccos = 0.0
        ssin = 0.0
        if abs(v[j]) > 1.0:
            ccos = 1 / v[j]
            ssin = math.sqrt(1.0 - ccos * ccos)
        else:
            ssin = v[j]
            ccos = math.sqrt(1.0 - ssin * ssin)
        for i in range(m):
            temp = ccos * a[i + j * lda] - ssin * a[i + (n - 1) * lda]
            a[i + (n - 1) * lda] = ssin * a[i + j * lda] + ccos * a[i + (n - 1) * lda]
            a[i + j * lda] = temp
    for j in range(n - 1):
        ccos = 0.0
        ssin = 0.0
        if abs(w[j]) > 1.0:
            ccos = 1 / w[j]
            ssin = math.sqrt(1 - ccos * ccos)
        else:
            ssin = w[j]
            ccos = math.sqrt(1 - ssin * ssin)
        for i in range(m):
            temp = ccos * a[i + lda * j] + ssin * a[i + (n - 1) * lda]
            a[i + (n - 1) * lda] = -ssin * a[i + j * lda] + ccos * a[i + (n - 1) * lda]
            a[i + lda * j] = temp

def _r1updt(m: int, n: int, s: np.ndarray[float, 1], u: np.ndarray[float, 1],
            v: np.ndarray[float, 1], w: np.ndarray[float, 1]) -> bool:
    giant = _DP2
    jj = (n * (2 * m - n + 1)) // 2 - (m - n)
    l = jj
    for i in range(n, m + 1):
        w[i - 1] = s[l - 1]
        l += 1
    nm1 = n - 1
    if nm1 >= 1:
        for nmj in range(1, nm1 + 1):
            j = n - nmj
            jj -= m - j + 1
            w[j - 1] = 0.0
            if v[j - 1] != 0.0:
                ccos = 0.0
                ssin = 0.0
                tau = 0.0
                if abs(v[n - 1]) < abs(v[j - 1]):
                    ccotan = v[n - 1] / v[j - 1]
                    ssin = 0.5 / math.sqrt(0.25 + 0.25 * (ccotan * ccotan))
                    ccos = ssin * ccotan
                    tau = 1.0
                    if abs(ccos) * giant > 1.0:
                        tau = 1.0 / ccos
                else:
                    ttan = v[j - 1] / v[n - 1]
                    ccos = 0.5 / math.sqrt(0.25 + 0.25 * (ttan * ttan))
                    ssin = ccos * ttan
                    tau = ssin
                v[n - 1] = ssin * v[j - 1] + ccos * v[n - 1]
                v[j - 1] = tau
                l = jj
                for i in range(j, m + 1):
                    temp = ccos * s[l - 1] - ssin * w[i - 1]
                    w[i - 1] = ssin * s[l - 1] + ccos * w[i - 1]
                    s[l - 1] = temp
                    l += 1
    for i in range(m):
        w[i] += v[n - 1] * u[i]
    sing = False
    if nm1 >= 1:
        for j in range(1, nm1 + 1):
            if w[j - 1] != 0.0:
                ccos = 0.0
                ssin = 0.0
                tau = 0.0
                if abs(s[jj - 1]) < abs(w[j - 1]):
                    ccotan = s[jj - 1] / w[j - 1]
                    ssin = 0.5 / math.sqrt(0.25 + 0.25 * ccotan ** 2)
                    ccos = ssin * ccotan
                    tau = 1.0
                    if abs(ccos) * giant > 1.0:
                        tau = 1 / ccos
                else:
                    ttan = w[j - 1] / s[jj - 1]
                    ccos = 0.5 / math.sqrt(0.25 + 0.25 * ttan ** 2)
                    ssin = ccos * ttan
                    tau = ssin
                l = jj
                for i in range(j, m + 1):
                    temp = ccos * s[l - 1] + ssin * w[i - 1]
                    w[i - 1] = -ssin * s[l - 1] + ccos * w[i - 1]
                    s[l - 1] = temp
                    l += 1
                w[j - 1] = tau
            if s[jj - 1] == 0.0:
                sing = True
            jj += (m - j + 1)
    l = jj
    for i in range(n, m + 1):
        s[l - 1] = w[i - 1]
        l += 1
    if s[jj - 1] == 0.0:
        sing = True
    return sing

def _hybrd(fcn, jacf, x: np.ndarray[float, 1], xtol: float, maxfev: int,
           ml: int, mu: int, epsfcn: float, diag: np.ndarray[float, 1],
           mode: int, factor: float, use_jac: bool):
    # Powell hybrid method (HYBRD; HYBRJ when use_jac). fcn(x) -> fvec,
    # jacf(x) -> (n, n) Jacobian. Returns (fvec, fjac, r, qtf, info, nfev, njev)
    n = x.size
    ldfjac = n
    lr = n * (n + 1) // 2
    fjac = np.zeros(n * n)
    r = np.zeros(lr)
    qtf = np.zeros(n)
    wa1 = np.zeros(n)
    wa2 = np.zeros(n)
    wa3 = np.zeros(n)
    wa4 = np.zeros(n)
    fvec = np.zeros(n)
    info = 0
    nfev = 0
    njev = 0
    epsmch = _DP0
    if n <= 0 or xtol < 0.0 or maxfev <= 0 or ml < 0 or mu < 0 or factor <= 0.0:
        return fvec, fjac, r, qtf, 0, nfev, njev
    if mode == 2:
        for j in range(n):
            if diag[j] <= 0.0:
                return fvec, fjac, r, qtf, 0, nfev, njev
    fvec = fcn(x)
    nfev = 1
    fnorm = _enorm(n, fvec)
    msum = min(ml + mu + 1, n)
    iter = 1
    ncsuc = 0
    ncfail = 0
    nslow1 = 0
    nslow2 = 0
    xnorm = 0.0
    delta = 0.0
    iwa = np.zeros(1, int)
    while True:
        jeval = True
        if use_jac:
            J = jacf(x)
            for j in range(n):
                for i in range(n):
                    fjac[i + ldfjac * j] = J[i, j]
            njev += 1
        else:
            eps = math.sqrt(max(epsfcn, epsmch))
            if ml + mu + 1 >= n:
                for j in range(n):
                    temp = x[j]
                    h = eps * abs(temp)
                    if h == 0.0:
                        h = eps
                    x[j] = temp + h
                    wv = fcn(x)
                    x[j] = temp
                    for i in range(n):
                        fjac[i + ldfjac * j] = (wv[i] - fvec[i]) / h
            else:
                mm = ml + mu + 1
                for k in range(1, mm + 1):
                    for j in range(k, n + 1, mm):
                        wa2[j - 1] = x[j - 1]
                        h = eps * abs(wa2[j - 1])
                        if h == 0.0:
                            h = eps
                        x[j - 1] = wa2[j - 1] + h
                    wv = fcn(x)
                    for j in range(k, n + 1, mm):
                        x[j - 1] = wa2[j - 1]
                        h = eps * abs(wa2[j - 1])
                        if h == 0.0:
                            h = eps
                        for i in range(1, n + 1):
                            fjac[(i - 1) + ldfjac * (j - 1)] = 0.0
                            if i >= j - mu and i <= j + ml:
                                fjac[(i - 1) + ldfjac * (j - 1)] = (wv[i - 1] - fvec[i - 1]) / h
            nfev += msum
        _qrfac(n, n, fjac, ldfjac, False, iwa, wa1, wa2, wa3)
        if iter == 1:
            if mode != 2:
                for j in range(n):
                    diag[j] = wa2[j]
                    if wa2[j] == 0.0:
                        diag[j] = 1.0
            for j in range(n):
                wa3[j] = diag[j] * x[j]
            xnorm = _enorm(n, wa3)
            delta = factor * xnorm
            if delta == 0.0:
                delta = factor
        for i in range(n):
            qtf[i] = fvec[i]
        for j in range(n):
            if fjac[j + ldfjac * j] != 0.0:
                ssum = 0.0
                for i in range(j, n):
                    ssum += fjac[i + ldfjac * j] * qtf[i]
                temp = -ssum / fjac[j + ldfjac * j]
                for i in range(j, n):
                    qtf[i] += fjac[i + ldfjac * j] * temp
        sing = False
        for j in range(n):
            l = j
            if j > 0:
                for i in range(j):
                    r[l] = fjac[i + ldfjac * j]
                    l += n - i - 1
            r[l] = wa1[j]
            if wa1[j] == 0.0:
                sing = True
        _qform(n, n, fjac, ldfjac, wa1)
        if mode != 2:
            for j in range(n):
                diag[j] = max(diag[j], wa2[j])
        while True:
            _dogleg(n, r, diag, qtf, delta, wa1, wa2, wa3)
            for j in range(n):
                wa1[j] = -wa1[j]
                wa2[j] = x[j] + wa1[j]
                wa3[j] = diag[j] * wa1[j]
            pnorm = _enorm(n, wa3)
            if iter == 1:
                delta = min(delta, pnorm)
            wa4 = fcn(wa2)
            nfev += 1
            fnorm1 = _enorm(n, wa4)
            actred = 1.0 - (fnorm1 / fnorm) ** 2 if fnorm1 < fnorm else -1.0
            l = 0
            for i in range(n):
                ssum = 0.0
                for j in range(i, n):
                    ssum += r[l] * wa1[j]
                    l += 1
                wa3[i] = qtf[i] + ssum
            temp = _enorm(n, wa3)
            prered = 1.0 - (temp / fnorm) ** 2 if temp < fnorm else 0.0
            ratio = actred / prered if prered > 0.0 else 0.0
            if ratio < 0.1:
                ncsuc = 0
                ncfail += 1
                delta = 0.5 * delta
            else:
                ncfail = 0
                ncsuc += 1
                if ratio >= 0.5 or ncsuc > 1:
                    delta = max(delta, pnorm / 0.5)
                if abs(ratio - 1.0) <= 0.1:
                    delta = pnorm / 0.5
            if ratio >= 0.0001:
                for j in range(n):
                    x[j] = wa2[j]
                    wa2[j] = diag[j] * x[j]
                    fvec[j] = wa4[j]
                xnorm = _enorm(n, wa2)
                fnorm = fnorm1
                iter += 1
            nslow1 += 1
            if actred >= 0.001:
                nslow1 = 0
            if jeval:
                nslow2 += 1
            if actred >= 0.1:
                nslow2 = 0
            if delta <= xtol * xnorm or fnorm == 0.0:
                info = 1
            if info != 0:
                return fvec, fjac, r, qtf, info, nfev, njev
            if nfev >= maxfev:
                info = 2
            if 0.1 * max(0.1 * delta, pnorm) <= epsmch * xnorm:
                info = 3
            if nslow2 == 5:
                info = 4
            if nslow1 == 10:
                info = 5
            if info != 0:
                return fvec, fjac, r, qtf, info, nfev, njev
            if ncfail == 2:
                break
            for j in range(n):
                ssum = 0.0
                for i in range(n):
                    ssum += fjac[i + ldfjac * j] * wa4[i]
                wa2[j] = (ssum - wa3[j]) / pnorm
                wa1[j] = diag[j] * ((diag[j] * wa1[j]) / pnorm)
                if ratio >= 0.0001:
                    qtf[j] = ssum
            sing = _r1updt(n, n, r, wa1, wa2, wa3)
            _r1mpyq(n, n, fjac, ldfjac, wa2, wa3)
            _r1mpyq(1, n, qtf, 1, wa2, wa3)
            jeval = False
    return fvec, fjac, r, qtf, info, nfev, njev

def _lmdif(fcn, jacf, x: np.ndarray[float, 1], m: int, ftol: float, xtol: float,
           gtol: float, maxfev: int, epsfcn: float, diag: np.ndarray[float, 1],
           mode: int, factor: float, use_jac: bool):
    # Levenberg-Marquardt (LMDIF; LMDER when use_jac). fcn(x) -> fvec (m,),
    # jacf(x) -> (m, n). Returns (fvec, fjac, ipvt, qtf, info, nfev, njev)
    n = x.size
    ldfjac = m
    fjac = np.zeros(m * n)
    ipvt = np.zeros(n, int)
    qtf = np.zeros(n)
    wa1 = np.zeros(n)
    wa2 = np.zeros(n)
    wa3 = np.zeros(n)
    wa4 = np.zeros(m)
    fvec = np.zeros(m)
    epsmch = _DP0
    delta = 0.0
    xnorm = 0.0
    info = 0
    nfev = 0
    njev = 0
    if n <= 0 or m < n or ftol < 0.0 or xtol < 0.0 or gtol < 0.0 or maxfev <= 0 or factor <= 0.0:
        return fvec, fjac, ipvt, qtf, 0, nfev, njev
    if mode == 2:
        for j in range(n):
            if diag[j] <= 0.0:
                return fvec, fjac, ipvt, qtf, 0, nfev, njev
    fvec = fcn(x)
    nfev = 1
    fnorm = _enorm(m, fvec)
    par = 0.0
    iter = 1
    while True:
        if use_jac:
            J = jacf(x)
            for j in range(n):
                for i in range(m):
                    fjac[i + ldfjac * j] = J[i, j]
            njev += 1
        else:
            eps = math.sqrt(max(epsfcn, epsmch))
            for j in range(n):
                temp = x[j]
                h = eps * abs(temp)
                if h == 0.0:
                    h = eps
                x[j] = temp + h
                wv = fcn(x)
                x[j] = temp
                for i in range(m):
                    fjac[i + ldfjac * j] = (wv[i] - fvec[i]) / h
            nfev += n
        _qrfac(m, n, fjac, ldfjac, True, ipvt, wa1, wa2, wa3)
        if iter == 1:
            if mode != 2:
                for j in range(n):
                    diag[j] = wa2[j]
                    if wa2[j] == 0.0:
                        diag[j] = 1.0
            for j in range(n):
                wa3[j] = diag[j] * x[j]
            xnorm = _enorm(n, wa3)
            delta = factor * xnorm
            if delta == 0.0:
                delta = factor
        for i in range(m):
            wa4[i] = fvec[i]
        for j in range(n):
            if fjac[j + ldfjac * j] != 0.0:
                ssum = 0.0
                for i in range(j, m):
                    ssum += fjac[i + ldfjac * j] * wa4[i]
                temp = -ssum / fjac[j + ldfjac * j]
                for i in range(j, m):
                    wa4[i] += fjac[i + ldfjac * j] * temp
            fjac[j + ldfjac * j] = wa1[j]
            qtf[j] = wa4[j]
        gnorm = 0.0
        if fnorm != 0.0:
            for j in range(n):
                l = ipvt[j]
                if wa2[l] != 0.0:
                    ssum = 0.0
                    for i in range(j + 1):
                        ssum += fjac[i + ldfjac * j] * (qtf[i] / fnorm)
                    gnorm = max(gnorm, abs(ssum / wa2[l]))
        if gnorm <= gtol:
            info = 4
        if info != 0:
            return fvec, fjac, ipvt, qtf, info, nfev, njev
        if mode != 2:
            for j in range(n):
                diag[j] = max(diag[j], wa2[j])
        while True:
            par = _lmpar(n, fjac, ldfjac, ipvt, diag, qtf, delta, par, wa1, wa2,
                         wa3, wa4)
            for j in range(n):
                wa1[j] = -wa1[j]
                wa2[j] = x[j] + wa1[j]
                wa3[j] = diag[j] * wa1[j]
            pnorm = _enorm(n, wa3)
            if iter == 1:
                delta = min(delta, pnorm)
            wa4 = fcn(wa2)
            nfev += 1
            fnorm1 = _enorm(m, wa4)
            actred = 1.0 - (fnorm1 / fnorm) ** 2 if 0.1 * fnorm1 < fnorm else -1.0
            for j in range(n):
                wa3[j] = 0.0
                l = ipvt[j]
                temp = wa1[l]
                for i in range(j + 1):
                    wa3[i] += fjac[i + ldfjac * j] * temp
            temp1 = _enorm(n, wa3) / fnorm
            temp2 = (math.sqrt(par) * pnorm) / fnorm
            prered = temp1 * temp1 + temp2 * temp2 / 0.5
            dirder = -(temp1 * temp1 + temp2 * temp2)
            ratio = actred / prered if prered != 0.0 else 0.0
            if ratio <= 0.25:
                temp = (0.5 * dirder) / (dirder + 0.5 * actred) if actred < 0.0 else 0.5
                if 0.1 * fnorm1 >= fnorm or temp < 0.1:
                    temp = 0.1
                delta = temp * min(delta, pnorm / 0.1)
                par /= temp
            else:
                if par == 0.0 or ratio >= 0.75:
                    delta = pnorm / 0.5
                    par *= 0.5
            if ratio >= 0.0001:
                for j in range(n):
                    x[j] = wa2[j]
                    wa2[j] = diag[j] * x[j]
                for i in range(m):
                    fvec[i] = wa4[i]
                xnorm = _enorm(n, wa2)
                fnorm = fnorm1
                iter += 1
            if abs(actred) <= ftol and prered <= ftol and 0.5 * ratio <= 1.0:
                info = 1
            if delta <= xtol * xnorm:
                info = 2
            if abs(actred) <= ftol and prered <= ftol and 0.5 * ratio <= 1.0 and info == 2:
                info = 3
            if info != 0:
                return fvec, fjac, ipvt, qtf, info, nfev, njev
            if nfev >= maxfev:
                info = 5
            if abs(actred) <= epsmch and prered <= epsmch and 0.5 * ratio <= 1.0:
                info = 6
            if delta <= epsmch * xnorm:
                info = 7
            if gnorm <= epsmch:
                info = 8
            if info != 0:
                return fvec, fjac, ipvt, qtf, info, nfev, njev
            if not (ratio < 0.0001):
                break
    return fvec, fjac, ipvt, qtf, info, nfev, njev

def _leastsq_cov(fjac: np.ndarray[float, 1], ipvt: np.ndarray[int, 1], m: int,
                 n: int):
    # cov_x from the R factor of the pivoted QR (as scipy.optimize.leastsq)
    # returns (ok, cov)
    R = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            R[i, j] = fjac[i + m * j]
    # invert upper-triangular R (trtri)
    inv = np.zeros((n, n))
    for i in range(n):
        if R[i, i] == 0.0:
            return False, inv
    for j in range(n):
        inv[j, j] = 1.0 / R[j, j]
        for i in range(j - 1, -1, -1):
            s = 0.0
            for k in range(i + 1, j + 1):
                s += R[i, k] * inv[k, j]
            inv[i, j] = -s / R[i, i]
    invp = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            invp[ipvt[i], j] = inv[i, j]
    return True, invp @ invp.T
