# Copyright (C) 2026 the Typhon authors
# Ported from scipy 1.18.1 (BSD-3-Clause)

# Krylov solvers of scipy.sparse.linalg._isolve: cg, cgs, bicg, bicgstab,
# gmres, minres, lsqr, lsmr. The iterations are line-by-line ports of the
# scipy 1.18 Python sources, so iterates (and iteration counts) agree.

import math
import numpy as np
from .._base import _up, _is_cplx
from ._interface import aslinearoperator, _is_linop

_EPS = 2.220446049250313e-16

def _xtype(A, b):
    # make_system: x dtype = coerce(A.dtype, b.dtype) in {float64, complex128}
    R = type(_up(type(_up(A.dtype, np.asarray(b).dtype)), float))
    return R()

def _dot(x, y):
    if _is_cplx(type(x[0])):
        return np.vdot(x, y)
    else:
        s = type(x[0])()
        for i in range(len(x)):
            s += x[i] * y[i]
        return s

def _nrm(x) -> float:
    s = 0.0
    for v in x:
        s += abs(v) * abs(v)
    return math.sqrt(s)

def _nrm2(x) -> float:
    return float(np.linalg.norm(x))

def _get_atol(bnrm2: float, atol: float, rtol: float) -> float:
    if atol < 0:
        raise ValueError("`atol` must be a real, non-negative number.")
    return max(float(atol), float(rtol) * float(bnrm2))

def _make_system(A, M, x0, b):
    Aop = aslinearoperator(A)
    if Aop.shape[0] != Aop.shape[1]:
        raise ValueError(f"expected square matrix, but got shape={Aop.shape}")
    N = Aop.shape[0]
    R = type(_xtype(Aop, b))
    bb = np.asarray(b).astype(R).ravel()
    if len(bb) != N:
        raise ValueError(f"shapes of A {Aop.shape} and b {np.asarray(b).shape} are incompatible")
    if x0 is None:
        x = np.zeros(N, dtype=R)
    else:
        x = np.asarray(x0).astype(R).ravel().copy()
        if len(x) != N:
            raise ValueError("shapes of A and x0 are incompatible")
    return Aop, x, bb

def _psolve(M, r):
    if M is None:
        return r.copy()
    else:
        return np.asarray(M.matvec(r)).astype(type(r[0]))

def _asM(M):
    if M is None:
        return None
    else:
        return aslinearoperator(M)

def _any(x) -> bool:
    for v in x:
        if v != type(v)():
            return True
    return False

def cg(A, b, x0 = None, rtol: float = 1e-05, atol: float = 0.0, maxiter = None,
       M = None, callback = None):
    Aop, x, bv = _make_system(A, M, x0, b)
    Mop = _asM(M)
    bnrm2 = _nrm2(bv)
    atol = _get_atol(bnrm2, atol, rtol)
    if bnrm2 == 0:
        return bv, 0
    n = len(bv)
    mi = n * 10 if maxiter is None else maxiter
    r = bv - Aop.matvec(x) if _any(x) else bv.copy()
    T = type(bv[0])
    rho_prev = T()
    p = np.empty_like(r)
    for iteration in range(mi):
        if _nrm2(r) < atol:
            return x, 0
        z = _psolve(Mop, r)
        rho_cur = _dot(r, z)
        if iteration > 0:
            beta = rho_cur / rho_prev
            p *= beta
            p += z
        else:
            p = z.copy()
        q = Aop.matvec(p)
        alpha = rho_cur / _dot(p, q)
        x += alpha * p
        r -= alpha * q
        rho_prev = rho_cur
        if callback is not None:
            callback(x)
    return x, mi

def cgs(A, b, x0 = None, rtol: float = 1e-05, atol: float = 0.0, maxiter = None,
        M = None, callback = None):
    Aop, x, bv = _make_system(A, M, x0, b)
    Mop = _asM(M)
    bnrm2 = _nrm2(bv)
    atol = _get_atol(bnrm2, atol, rtol)
    if bnrm2 == 0:
        return bv, 0
    n = len(bv)
    mi = n * 10 if maxiter is None else maxiter
    T = type(bv[0])
    rhotol = _EPS ** 2
    r = bv - Aop.matvec(x) if _any(x) else bv.copy()
    rtilde = r.copy()
    rho_prev = T()
    u = np.empty_like(r)
    q = np.empty_like(r)
    p = np.empty_like(r)
    for iteration in range(mi):
        rnorm = _nrm2(r)
        if rnorm < atol:
            return x, 0
        rho_cur = _dot(rtilde, r)
        if abs(rho_cur) < rhotol:
            return x, -10
        if iteration > 0:
            beta = rho_cur / rho_prev
            u[:] = r[:]
            u += beta * q
            p *= beta
            p += q
            p *= beta
            p += u
        else:
            p = r.copy()
            u = r.copy()
            q = np.empty_like(r)
        phat = _psolve(Mop, p)
        vhat = Aop.matvec(phat)
        rv = _dot(rtilde, vhat)
        if rv == T():
            return x, -11
        alpha = rho_cur / rv
        q[:] = u[:]
        q -= alpha * vhat
        uhat = _psolve(Mop, u + q)
        x += alpha * uhat
        r = bv - Aop.matvec(x)
        rho_prev = rho_cur
        if callback is not None:
            callback(x)
    return x, mi

def bicg(A, b, x0 = None, rtol: float = 1e-05, atol: float = 0.0, maxiter = None,
         M = None, callback = None):
    Aop, x, bv = _make_system(A, M, x0, b)
    Mop = _asM(M)
    bnrm2 = _nrm2(bv)
    atol = _get_atol(bnrm2, atol, rtol)
    if bnrm2 == 0:
        return bv, 0
    n = len(bv)
    mi = n * 10 if maxiter is None else maxiter
    T = type(bv[0])
    rhotol = _EPS ** 2
    r = bv - Aop.matvec(x) if _any(x) else bv.copy()
    rtilde = r.copy()
    rho_prev = T()
    p = np.empty_like(r)
    ptilde = np.empty_like(r)
    for iteration in range(mi):
        if _nrm2(r) < atol:
            return x, 0
        z = _psolve(Mop, r)
        ztilde = rtilde.copy() if Mop is None else np.asarray(Mop.rmatvec(rtilde)).astype(T)
        rho_cur = _dot(rtilde, z)
        if abs(rho_cur) < rhotol:
            return x, -10
        if iteration > 0:
            beta = rho_cur / rho_prev
            p *= beta
            p += z
            ptilde *= np.conj(beta) if _is_cplx(T) else beta
            ptilde += ztilde
        else:
            p = z.copy()
            ptilde = ztilde.copy()
        q = Aop.matvec(p)
        qtilde = Aop.rmatvec(ptilde)
        rv = _dot(ptilde, q)
        if rv == T():
            return x, -11
        alpha = rho_cur / rv
        x += alpha * p
        r -= alpha * q
        rtilde -= (np.conj(alpha) if _is_cplx(T) else alpha) * qtilde
        rho_prev = rho_cur
        if callback is not None:
            callback(x)
    return x, mi

def bicgstab(A, b, x0 = None, rtol: float = 1e-05, atol: float = 0.0,
             maxiter = None, M = None, callback = None):
    Aop, x, bv = _make_system(A, M, x0, b)
    Mop = _asM(M)
    bnrm2 = _nrm2(bv)
    atol = _get_atol(bnrm2, atol, rtol)
    if bnrm2 == 0:
        return bv, 0
    n = len(bv)
    mi = n * 10 if maxiter is None else maxiter
    T = type(bv[0])
    rhotol = _EPS ** 2
    omegatol = rhotol
    rho_prev = T()
    omega = T()
    alpha = T()
    p = np.empty_like(bv)
    v = np.empty_like(bv)
    s = np.empty_like(bv)
    r = bv - Aop.matvec(x) if _any(x) else bv.copy()
    rtilde = r.copy()
    for iteration in range(mi):
        if _nrm2(r) < atol:
            return x, 0
        rho = _dot(rtilde, r)
        if abs(rho) < rhotol:
            return x, -10
        if iteration > 0:
            if abs(omega) < omegatol:
                return x, -11
            beta = (rho / rho_prev) * (alpha / omega)
            p -= omega * v
            p *= beta
            p += r
        else:
            s = np.empty_like(r)
            p = r.copy()
        phat = _psolve(Mop, p)
        v = Aop.matvec(phat)
        rv = _dot(rtilde, v)
        if rv == T():
            return x, -11
        alpha = rho / rv
        r -= alpha * v
        s[:] = r[:]
        if _nrm2(s) < atol:
            x += alpha * phat
            return x, 0
        shat = _psolve(Mop, s)
        t = Aop.matvec(shat)
        omega = _dot(t, s) / _dot(t, t)
        x += alpha * phat
        x += omega * shat
        r -= omega * t
        rho_prev = rho
        if callback is not None:
            callback(x)
    return x, mi

def _lartg(f: T, g: T, T: type):
    # LAPACK 3.10+ dlartg / zlartg (no scaling branch needed for moderate data)
    if _is_cplx(T):
        if g == T():
            return 1.0, T(0), f
        if f == T():
            ag = abs(g)
            return 0.0, g.conjugate() / ag, T(ag)
        f2 = f.real * f.real + f.imag * f.imag
        g2 = g.real * g.real + g.imag * g.imag
        h2 = f2 + g2
        d = math.sqrt(f2 * h2)
        c = f2 / d
        r = f * (h2 / d)
        s = g.conjugate() * (f / d)
        return c, s, r
    else:
        if g == 0.0:
            return 1.0, 0.0, f
        if f == 0.0:
            return 0.0, (1.0 if g > 0 else -1.0), abs(g)
        d = math.sqrt(f * f + g * g)
        c = abs(f) / d
        r = d if f > 0 else -d
        s = g / r
        return c, s, r

def gmres(A, b, x0 = None, rtol: float = 1e-05, atol: float = 0.0, restart = None,
          maxiter = None, M = None, callback = None, callback_type = None):
    Aop, x, bv = _make_system(A, M, x0, b)
    Mop = _asM(M)
    T = type(bv[0])
    n = len(bv)
    bnrm2 = _nrm2(bv)
    atol = _get_atol(bnrm2, atol, rtol)
    if bnrm2 == 0:
        return bv, 0
    eps = _EPS
    mi = n * 10 if maxiter is None else maxiter
    rs = 20 if restart is None else restart
    rs = min(rs, n)
    Mb_nrm2 = _nrm2(_psolve(Mop, bv))
    ptol_max_factor = 1.0
    ptol = Mb_nrm2 * min(ptol_max_factor, atol / bnrm2)
    presid = 0.0
    v = np.empty((rs + 1, n), dtype=T)
    h = np.zeros((rs, rs + 1), dtype=T)
    givens = np.zeros((rs, 2), dtype=T)
    inner_iter = 0
    r = bv.copy()
    rnorm = 0.0
    for iteration in range(mi):
        if iteration == 0:
            r = bv - Aop.matvec(x) if _any(x) else bv.copy()
            if _nrm2(r) < atol:
                return x, 0
        pr = _psolve(Mop, r)
        tmp0 = _nrm2(pr)
        for i in range(n):
            v[0, i] = pr[i] * T(1.0 / tmp0)
        S = np.zeros(rs + 1, dtype=T)
        S[0] = T(tmp0)
        breakdown = False
        col = 0
        for col in range(rs):
            av = Aop.matvec(v[col, :])
            w = _psolve(Mop, av)
            h0 = _nrm2(w)
            for k in range(col + 1):
                vk = v[k, :]
                tmp = _dot(vk, w)
                h[col, k] = tmp
                w -= tmp * vk
            h1 = _nrm2(w)
            h[col, col + 1] = T(h1)
            for i in range(n):
                v[col + 1, i] = w[i]
            if h1 <= eps * h0:
                h[col, col + 1] = T()
                breakdown = True
            else:
                inv = T(1.0 / h1)
                for i in range(n):
                    v[col + 1, i] *= inv
            for k in range(col):
                c = givens[k, 0]
                s = givens[k, 1]
                n0 = h[col, k]
                n1 = h[col, k + 1]
                sc = s.conjugate() if _is_cplx(T) else s
                h[col, k] = c * n0 + s * n1
                h[col, k + 1] = -sc * n0 + c * n1
            c, s, mag = _lartg(h[col, col], h[col, col + 1])
            givens[col, 0] = T(c)
            givens[col, 1] = T(s)
            h[col, col] = T(mag)
            h[col, col + 1] = T()
            sc2 = T(s).conjugate() if _is_cplx(T) else T(s)
            tmp = -sc2 * S[col]
            S[col] = T(c) * S[col]
            S[col + 1] = tmp
            presid = abs(tmp)
            inner_iter += 1
            if presid <= ptol or breakdown:
                break
        if h[col, col] == T():
            S[col] = T()
        y = np.zeros(col + 1, dtype=T)
        for i in range(col + 1):
            y[i] = S[i]
        for k in range(col, 0, -1):
            if y[k] != T():
                y[k] /= h[k, k]
                tmp = y[k]
                for i in range(k):
                    y[i] -= tmp * h[k, i]
        if y[0] != T():
            y[0] /= h[0, 0]
        x += y @ v[:col + 1, :]
        r = bv - Aop.matvec(x)
        rnorm = _nrm2(r)
        if callback is not None:
            callback(x)
        if rnorm <= atol:
            break
        elif breakdown:
            break
        elif presid <= ptol:
            ptol_max_factor = max(eps, 0.25 * ptol_max_factor)
        else:
            ptol_max_factor = min(1.0, 1.5 * ptol_max_factor)
        ptol = presid * min(ptol_max_factor, atol / rnorm)
    info = 0 if rnorm <= atol else mi
    return x, info

def minres(A, b, x0 = None, shift: float = 0.0, rtol: float = 1e-05,
           maxiter = None, M = None, callback = None, show: bool = False,
           check: bool = False):
    Aop, x, bv = _make_system(A, M, x0, b)
    Mop = _asM(M)
    T = type(bv[0])
    n = Aop.shape[0]
    mi = 5 * n if maxiter is None else maxiter
    istop = 0
    itn = 0
    Anorm = 0.0
    Acond = 0.0
    rnorm = 0.0
    ynorm = 0.0
    eps = _EPS
    if x0 is None:
        r1 = bv.copy()
    else:
        r1 = bv - Aop.matvec(x)
    y = _psolve(Mop, r1)
    beta1c = _dot(r1, y)
    beta1r = beta1c.real if _is_cplx(T) else beta1c
    if beta1r < 0:
        raise ValueError("indefinite preconditioner")
    elif beta1r == 0:
        return x, 0
    bnorm = _nrm2(bv)
    if bnorm == 0:
        return bv, 0
    beta1 = math.sqrt(beta1r)
    if check:
        w = Aop.matvec(y)
        r2c = Aop.matvec(w)
        s = _dot(w, w)
        t = _dot(y, r2c)
        z = abs(s - t)
        epsa = (abs(s) + eps) * eps ** (1.0 / 3.0)
        if z > epsa:
            raise ValueError("non-symmetric matrix")
    oldb = 0.0
    beta = beta1
    dbar = T()
    epsln = T()
    qrnorm = beta1
    phibar = T(beta1)
    rhs1 = beta1
    rhs2 = 0.0
    tnorm2 = 0.0
    gmax = 0.0
    gmin = 1.7976931348623157e308
    cs = T(-1.0)
    sn = T(0.0)
    w = np.zeros(n, dtype=T)
    w2 = np.zeros(n, dtype=T)
    r2 = r1
    while itn < mi:
        itn += 1
        s = 1.0 / beta
        v = T(s) * y
        y = Aop.matvec(v).astype(T)
        y = y - T(shift) * v
        if itn >= 2:
            y = y - T(beta / oldb) * r1
        alfa = _dot(v, y)
        y = y - (alfa / T(beta)) * r2
        r1 = r2
        r2 = y
        y = _psolve(Mop, r2)
        oldb = beta
        betac = _dot(r2, y)
        betar = betac.real if _is_cplx(T) else betac
        if betar < 0:
            raise ValueError("non-symmetric matrix")
        beta = math.sqrt(betar)
        tnorm2 += abs(alfa) ** 2 + oldb ** 2 + beta ** 2
        if itn == 1:
            if beta / beta1 <= 10 * eps:
                istop = -1
        oldeps = epsln
        delta = cs * dbar + sn * alfa
        gbar = sn * dbar - cs * alfa
        epsln = sn * T(beta)
        dbar = -cs * T(beta)
        root = math.sqrt(abs(gbar) ** 2 + abs(dbar) ** 2)
        Arnorm = abs(phibar) * root
        gamma = math.sqrt(abs(gbar) ** 2 + beta ** 2)
        gamma = max(gamma, eps)
        cs = gbar / T(gamma)
        sn = T(beta / gamma)
        phi = cs * phibar
        phibar = sn * phibar
        denom = 1.0 / gamma
        w1 = w2
        w2 = w
        w = (v - oldeps * w1 - delta * w2) * T(denom)
        x = x + phi * w
        gmax = max(gmax, gamma)
        gmin = min(gmin, gamma)
        z = rhs1 / gamma
        rhs1 = rhs2 - (delta.real if _is_cplx(T) else delta) * z
        rhs2 = -(epsln.real if _is_cplx(T) else epsln) * z
        Anorm = math.sqrt(tnorm2)
        ynorm = _nrm2(x)
        epsa = Anorm * eps
        epsx = Anorm * ynorm * eps
        diag = gbar
        if diag == T():
            diag = T(epsa)
        qrnorm = abs(phibar)
        rnorm = qrnorm
        if ynorm == 0 or Anorm == 0:
            test1 = math.inf
        else:
            test1 = rnorm / (Anorm * ynorm)
        if Anorm == 0:
            test2 = math.inf
        else:
            test2 = root / Anorm
        Acond = gmax / gmin
        if istop == 0:
            t1 = 1 + test1
            t2 = 1 + test2
            if t2 <= 1:
                istop = 2
            if t1 <= 1:
                istop = 1
            if itn >= mi:
                istop = 6
            if Acond >= 0.1 / eps:
                istop = 4
            if epsx >= beta1:
                istop = 3
            if test2 <= rtol:
                istop = 2
            if test1 <= rtol:
                istop = 1
        if callback is not None:
            callback(x)
        if istop != 0:
            break
    info = mi if istop == 6 else 0
    return x, info

def _sign(a: float) -> float:
    if a > 0:
        return 1.0
    elif a < 0:
        return -1.0
    else:
        return 0.0

def _sym_ortho(a: float, b: float):
    if b == 0:
        return _sign(a), 0.0, abs(a)
    elif a == 0:
        return 0.0, _sign(b), abs(b)
    elif abs(b) > abs(a):
        tau = a / b
        s = _sign(b) / math.sqrt(1 + tau * tau)
        c = s * tau
        r = b / s
        return c, s, r
    else:
        tau = b / a
        c = _sign(a) / math.sqrt(1 + tau * tau)
        s = c * tau
        r = a / c
        return c, s, r

def _rect_op(A):
    return aslinearoperator(A)

def lsqr(A, b, damp: float = 0.0, atol: float = 1e-06, btol: float = 1e-06,
         conlim: float = 1e8, iter_lim = None, show: bool = False,
         calc_var: bool = False, x0 = None):
    Aop = _rect_op(A)
    bv = np.asarray(b).astype(float).ravel()
    m, n = Aop.shape
    il = 2 * n if iter_lim is None else iter_lim
    var = np.zeros(n)
    itn = 0
    istop = 0
    ctol = 0.0
    if conlim > 0:
        ctol = 1 / conlim
    anorm = 0.0
    acond = 0.0
    dampsq = damp ** 2
    ddnorm = 0.0
    res2 = 0.0
    xnorm = 0.0
    xxnorm = 0.0
    z = 0.0
    cs2 = -1.0
    sn2 = 0.0
    u = bv
    bnorm = _nrm2(bv)
    if x0 is None:
        x = np.zeros(n)
        beta = bnorm
    else:
        x = np.asarray(x0).astype(float).copy()
        u = u - Aop.matvec(x)
        beta = _nrm2(u)
    if beta > 0:
        u = (1 / beta) * u
        v = np.asarray(Aop.rmatvec(u)).astype(float)
        alfa = _nrm2(v)
    else:
        v = x.copy()
        alfa = 0.0
    if alfa > 0:
        v = (1 / alfa) * v
    w = v.copy()
    rhobar = alfa
    phibar = beta
    rnorm = beta
    r1norm = rnorm
    r2norm = rnorm
    arnorm = alfa * beta
    if arnorm == 0:
        return x, istop, itn, r1norm, r2norm, anorm, acond, arnorm, xnorm, var
    eps = _EPS
    while itn < il:
        itn = itn + 1
        u = np.asarray(Aop.matvec(v)).astype(float) - alfa * u
        beta = _nrm2(u)
        if beta > 0:
            u = (1 / beta) * u
            anorm = math.sqrt(anorm ** 2 + alfa ** 2 + beta ** 2 + dampsq)
            v = np.asarray(Aop.rmatvec(u)).astype(float) - beta * v
            alfa = _nrm2(v)
            if alfa > 0:
                v = (1 / alfa) * v
        if damp > 0:
            rhobar1 = math.sqrt(rhobar ** 2 + dampsq)
            cs1 = rhobar / rhobar1
            sn1 = damp / rhobar1
            psi = sn1 * phibar
            phibar = cs1 * phibar
        else:
            rhobar1 = rhobar
            psi = 0.0
        cs, sn, rho = _sym_ortho(rhobar1, beta)
        theta = sn * alfa
        rhobar = -cs * alfa
        phi = cs * phibar
        phibar = sn * phibar
        tau = sn * phi
        t1 = phi / rho
        t2 = -theta / rho
        dk = (1 / rho) * w
        x = x + t1 * w
        w = v + t2 * w
        ddnorm = ddnorm + _nrm2(dk) ** 2
        if calc_var:
            var = var + dk ** 2
        delta = sn2 * rho
        gambar = -cs2 * rho
        rhs = phi - delta * z
        zbar = rhs / gambar
        xnorm = math.sqrt(xxnorm + zbar ** 2)
        gamma = math.sqrt(gambar ** 2 + theta ** 2)
        cs2 = gambar / gamma
        sn2 = theta / gamma
        z = rhs / gamma
        xxnorm = xxnorm + z ** 2
        acond = anorm * math.sqrt(ddnorm)
        res1 = phibar ** 2
        res2 = res2 + psi ** 2
        rnorm = math.sqrt(res1 + res2)
        arnorm = alfa * abs(tau)
        if damp > 0:
            r1sq = rnorm ** 2 - dampsq * xxnorm
            r1norm = math.sqrt(abs(r1sq))
            if r1sq < 0:
                r1norm = -r1norm
        else:
            r1norm = rnorm
        r2norm = rnorm
        test1 = rnorm / bnorm
        test2 = arnorm / (anorm * rnorm + eps)
        test3 = 1 / (acond + eps)
        t1 = test1 / (1 + anorm * xnorm / bnorm)
        rtol = btol + atol * anorm * xnorm / bnorm
        if itn >= il:
            istop = 7
        if 1 + test3 <= 1:
            istop = 6
        if 1 + test2 <= 1:
            istop = 5
        if 1 + t1 <= 1:
            istop = 4
        if test3 <= ctol:
            istop = 3
        if test2 <= atol:
            istop = 2
        if test1 <= rtol:
            istop = 1
        if istop != 0:
            break
    return x, istop, itn, r1norm, r2norm, anorm, acond, arnorm, xnorm, var

def lsmr(A, b, damp: float = 0.0, atol: float = 1e-06, btol: float = 1e-06,
         conlim: float = 1e8, maxiter = None, show: bool = False, x0 = None):
    Aop = _rect_op(A)
    bv = np.asarray(b).astype(float).ravel()
    m, n = Aop.shape
    minDim = min(m, n)
    mi = minDim if maxiter is None else maxiter
    u = bv
    normb = _nrm2(bv)
    if x0 is None:
        x = np.zeros(n)
        beta = normb
    else:
        x = np.asarray(x0).astype(float).copy()
        u = u - Aop.matvec(x)
        beta = _nrm2(u)
    if beta > 0:
        u = (1 / beta) * u
        v = np.asarray(Aop.rmatvec(u)).astype(float)
        alpha = _nrm2(v)
    else:
        v = np.zeros(n)
        alpha = 0.0
    if alpha > 0:
        v = (1 / alpha) * v
    itn = 0
    zetabar = alpha * beta
    alphabar = alpha
    rho = 1.0
    rhobar = 1.0
    cbar = 1.0
    sbar = 0.0
    h = v.copy()
    hbar = np.zeros(n)
    betadd = beta
    betad = 0.0
    rhodold = 1.0
    tautildeold = 0.0
    thetatilde = 0.0
    zeta = 0.0
    d = 0.0
    normA2 = alpha * alpha
    maxrbar = 0.0
    minrbar = 1e100
    normA = math.sqrt(normA2)
    condA = 1.0
    normx = 0.0
    istop = 0
    ctol = 0.0
    if conlim > 0:
        ctol = 1 / conlim
    normr = beta
    normar = alpha * beta
    if normar == 0:
        return x, istop, itn, normr, normar, normA, condA, normx
    if normb == 0:
        x[:] = 0.0
        return x, istop, itn, normr, normar, normA, condA, normx
    while itn < mi:
        itn = itn + 1
        u *= -alpha
        u += np.asarray(Aop.matvec(v)).astype(float)
        beta = _nrm2(u)
        if beta > 0:
            u *= 1 / beta
            v *= -beta
            v += np.asarray(Aop.rmatvec(u)).astype(float)
            alpha = _nrm2(v)
            if alpha > 0:
                v *= 1 / alpha
        chat, shat, alphahat = _sym_ortho(alphabar, damp)
        rhoold = rho
        c, s, rho = _sym_ortho(alphahat, beta)
        thetanew = s * alpha
        alphabar = c * alpha
        rhobarold = rhobar
        zetaold = zeta
        thetabar = sbar * rho
        rhotemp = cbar * rho
        cbar, sbar, rhobar = _sym_ortho(cbar * rho, thetanew)
        zeta = cbar * zetabar
        zetabar = -sbar * zetabar
        hbar *= -(thetabar * rho / (rhoold * rhobarold))
        hbar += h
        x += (zeta / (rho * rhobar)) * hbar
        h *= -(thetanew / rho)
        h += v
        betaacute = chat * betadd
        betacheck = -shat * betadd
        betahat = c * betaacute
        betadd = -s * betaacute
        thetatildeold = thetatilde
        ctildeold, stildeold, rhotildeold = _sym_ortho(rhodold, thetabar)
        thetatilde = stildeold * rhobar
        rhodold = ctildeold * rhobar
        betad = -stildeold * betad + ctildeold * betahat
        tautildeold = (zetaold - thetatildeold * tautildeold) / rhotildeold
        taud = (zeta - thetatilde * tautildeold) / rhodold
        d = d + betacheck * betacheck
        normr = math.sqrt(d + (betad - taud) ** 2 + betadd * betadd)
        normA2 = normA2 + beta * beta
        normA = math.sqrt(normA2)
        normA2 = normA2 + alpha * alpha
        maxrbar = max(maxrbar, rhobarold)
        if itn > 1:
            minrbar = min(minrbar, rhobarold)
        condA = max(maxrbar, rhotemp) / min(minrbar, rhotemp)
        normar = abs(zetabar)
        normx = _nrm2(x)
        test1 = normr / normb
        if normA * normr != 0:
            test2 = normar / (normA * normr)
        else:
            test2 = math.inf
        test3 = 1 / condA
        t1 = test1 / (1 + normA * normx / normb)
        rtol = btol + atol * normA * normx / normb
        if itn >= mi:
            istop = 7
        if 1 + test3 <= 1:
            istop = 6
        if 1 + test2 <= 1:
            istop = 5
        if 1 + t1 <= 1:
            istop = 4
        if test3 <= ctol:
            istop = 3
        if test2 <= atol:
            istop = 2
        if test1 <= rtol:
            istop = 1
        if istop > 0:
            break
    return x, istop, itn, normr, normar, normA, condA, normx
