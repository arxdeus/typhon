# Copyright (C) 2026 the Typhon authors
# Ported from scipy 1.18.1 (BSD-3-Clause)

# Further Krylov solvers of scipy.sparse.linalg._isolve: qmr (iterative.py),
# tfqmr (tfqmr.py), lgmres (lgmres.py) and gcrotmk (_gcrotmk.py, including
# the shared flexible GMRES kernel _fgmres). Line-by-line ports; BLAS axpy /
# dot / scal / nrm2 are plain loops. Deviations: _fgmres updates its QR
# factorization with one Givens rotation (what scipy.linalg.qr_insert does
# for a column appended last) and solves the small triangular least-squares
# problem by back substitution (scipy: lstsq), so iterates agree with scipy
# to rounding; gcrotmk's truncate='smallest' is not supported.

import math
import numpy as np
from .._base import _up, _is_cplx
from ._interface import aslinearoperator
from ._isolve import _make_system, _psolve, _asM, _get_atol, _nrm2, _any, _EPS

def _dotc(x, y, T: type):
    # BLAS dot (real) / dotc (complex)
    s = T()
    if _is_cplx(T):
        for i in range(len(x)):
            s += x[i].conjugate() * y[i]
    else:
        for i in range(len(x)):
            s += x[i] * y[i]
    return s

def _rpsolve(M, r, T: type):
    if M is None:
        return r.copy()
    else:
        return np.asarray(M.rmatvec(r)).astype(T)

def _conj(v, T: type):
    if _is_cplx(T):
        return v.conjugate()
    else:
        return v

# ----------------------------------------------------------------- qmr

def qmr(A, b, x0 = None, rtol: float = 1e-05, atol: float = 0.0, maxiter = None,
        M1 = None, M2 = None, callback = None):
    Aop, x, bv = _make_system(A, None, x0, b)
    return _qmr(Aop, x, bv, rtol, atol, maxiter, M1, M2, callback)

def _qmr(Aop, x, bv, rtol: float, atol: float, maxiter, M1, M2, callback):
    T = type(bv[0])
    bnrm2 = _nrm2(bv)
    atol = _get_atol(bnrm2, atol, rtol)
    if bnrm2 == 0:
        return bv, 0
    M1op = _asM(M1)
    M2op = _asM(M2)
    n = len(bv)
    mi = n * 10 if maxiter is None else maxiter
    tolb = _EPS
    r = bv - Aop.matvec(x) if _any(x) else bv.copy()
    vtilde = r.copy()
    y = _psolve(M1op, vtilde)
    rho = _nrm2(y)
    wtilde = r.copy()
    z = _rpsolve(M2op, wtilde, T)
    xi = _nrm2(z)
    gamma = 1.0
    eta = T(-1.0)
    theta = 0.0
    v = np.empty(n, dtype=T)
    w = np.empty(n, dtype=T)
    epsilon = T()
    q = np.zeros(n, dtype=T)
    d = np.zeros(n, dtype=T)
    p = np.zeros(n, dtype=T)
    s = np.zeros(n, dtype=T)
    for iteration in range(mi):
        if _nrm2(r) < atol:
            return x, 0
        if abs(rho) < tolb:
            return x, -10
        if abs(xi) < tolb:
            return x, -15
        v = vtilde * T(1.0 / rho)
        y = y * T(1.0 / rho)
        w = wtilde * T(1.0 / xi)
        z = z * T(1.0 / xi)
        delta = _dotc(z, y, T)
        if abs(delta) < tolb:
            return x, -13
        ytilde = _psolve(M2op, y)
        ztilde = _rpsolve(M1op, z, T)
        if iteration > 0:
            ytilde -= (T(xi) * delta / epsilon) * p
            p = ytilde.copy()
            ztilde -= (T(rho) * _conj(delta / epsilon, T)) * q
            q = ztilde.copy()
        else:
            p = ytilde.copy()
            q = ztilde.copy()
        ptilde = Aop.matvec(p)
        epsilon = _dotc(q, ptilde, T)
        if abs(epsilon) < tolb:
            return x, -14
        beta = epsilon / delta
        if abs(beta) < tolb:
            return x, -11
        vtilde = ptilde - beta * v
        y = _psolve(M1op, vtilde)
        rho_prev = rho
        rho = _nrm2(y)
        wtilde = w * (-_conj(beta, T))
        wtilde += Aop.rmatvec(q)
        z = _rpsolve(M2op, wtilde, T)
        xi = _nrm2(z)
        gamma_prev = gamma
        theta_prev = theta
        theta = rho / (gamma_prev * abs(beta))
        gamma = 1.0 / math.sqrt(1.0 + theta * theta)
        if abs(gamma) < tolb:
            return x, -12
        eta *= -(T(rho_prev) / beta) * T((gamma / gamma_prev) ** 2)
        if iteration > 0:
            d *= T((theta_prev * gamma) ** 2)
            d += eta * p
            s *= T((theta_prev * gamma) ** 2)
            s += eta * ptilde
        else:
            d = p * eta
            s = ptilde * eta
        x += d
        r -= s
        if callback is not None:
            callback(x)
    return x, mi

# ----------------------------------------------------------------- tfqmr

def tfqmr(A, b, x0 = None, rtol: float = 1e-05, atol: float = 0.0, maxiter = None,
          M = None, callback = None, show: bool = False):
    Aop, x, bv = _make_system(A, M, x0, b)
    return _tfqmr(Aop, x, bv, x0 is None, rtol, atol, maxiter, M, callback)

def _tfqmr(Aop, x, bv, nox0: bool, rtol: float, atol: float, maxiter, M, callback):
    T = type(bv[0])
    Mop = _asM(M)
    if _nrm2(bv) == 0.0:
        return bv.copy(), 0
    ndofs = len(bv)
    mi = min(10000, ndofs * 10) if maxiter is None else maxiter
    if nox0:
        r = bv.copy()
    else:
        r = bv - Aop.matvec(x)
    u = r
    w = r.copy()
    rstar = r
    v = _psolve(Mop, Aop.matvec(r))
    uhat = v
    d = np.zeros(ndofs, dtype=T)
    theta = 0.0
    eta = T()
    rho = T(_dotc(rstar, r, T).real)
    rhoLast = rho
    r0norm = math.sqrt(abs(rho.real) if _is_cplx(T) else rho)
    tau = r0norm
    if r0norm == 0:
        return x, 0
    atol = _get_atol(r0norm, atol, rtol)
    alpha = T()
    uNext = u
    for it in range(mi):
        even = it % 2 == 0
        if even:
            vtrstar = _dotc(rstar, v, T)
            if vtrstar == T():
                return x, -1
            alpha = rho / vtrstar
            uNext = u - alpha * v
        w -= alpha * uhat
        d = u + (T(theta * theta) / alpha) * eta * d
        theta = _nrm2(w) / tau
        c = math.sqrt(1.0 / (1.0 + theta * theta))
        tau *= theta * c
        eta = T(c * c) * alpha
        z = _psolve(Mop, d)
        x += eta * z
        if callback is not None:
            callback(x)
        if tau * math.sqrt(it + 1) < atol:
            return x, 0
        if not even:
            rho = _dotc(rstar, w, T)
            beta = rho / rhoLast
            u = w + beta * u
            v = beta * uhat + (beta * beta) * v
            uhat = _psolve(Mop, Aop.matvec(u))
            v += uhat
        else:
            uhat = _psolve(Mop, Aop.matvec(uNext))
            u = uNext
            rhoLast = rho
    return x, mi

# ----------------------------------------------------------------- _fgmres

def _givens(f: T, g: T, T: type):
    from ._isolve import _lartg
    return _lartg(f, g)

class _FG[T]:
    Q: np.ndarray[T, 2]
    R: np.ndarray[T, 2]
    B: np.ndarray[T, 2]
    vs: List[np.ndarray[T, 1]]
    zs: List[np.ndarray[T, 1]]
    y: np.ndarray[T, 1]
    res: float

    def __init__(self, Q: np.ndarray[T, 2], R: np.ndarray[T, 2], B: np.ndarray[T, 2],
                 vs: List[np.ndarray[T, 1]], zs: List[np.ndarray[T, 1]],
                 y: np.ndarray[T, 1], res: float):
        self.Q = Q
        self.R = R
        self.B = B
        self.vs = vs
        self.zs = zs
        self.y = y
        self.res = res

def _fgmres(matvec, v0: np.ndarray[T, 1], m: int, atol: float, lpsolve, rpsolve,
            cs: List[np.ndarray[T, 1]], outer_v, prepend_outer_v: bool, T: type):
    # outer_v: List[Tuple[ndarray, ndarray, bool]] (z, Az, has_Az)
    n = len(v0)
    vs = [v0]
    zs = List[np.ndarray[T, 1]]()
    nov = len(outer_v)
    m = m + nov
    Bm = np.zeros((len(cs), m), dtype=T)
    Q = np.ones((1, 1), dtype=T)
    R = np.zeros((1, 0), dtype=T)
    eps = _EPS
    breakdown = False
    res = float("nan")
    j = 0
    for jj in range(m):
        j = jj
        have_w = False
        if prepend_outer_v and j < nov:
            z = outer_v[j][0]
            w = outer_v[j][1]
            have_w = outer_v[j][2]
        elif prepend_outer_v and j == nov:
            z = rpsolve(v0)
            w = z
        elif (not prepend_outer_v) and j >= m - nov:
            z = outer_v[j - (m - nov)][0]
            w = outer_v[j - (m - nov)][1]
            have_w = outer_v[j - (m - nov)][2]
        else:
            z = rpsolve(vs[len(vs) - 1])
            w = z
        if not have_w:
            w = lpsolve(matvec(z))
        else:
            w = w.copy()
        w_norm = _nrm2(w)
        for i in range(len(cs)):
            c = cs[i]
            alpha = _dotc(c, w, T)
            Bm[i, j] = alpha
            w = w - alpha * c
        hcur = np.zeros(j + 2, dtype=T)
        for i in range(len(vs)):
            v = vs[i]
            alpha = _dotc(v, w, T)
            hcur[i] = alpha
            w = w - alpha * v
        hcur[j + 1] = T(_nrm2(w))
        hl = abs(hcur[j + 1])
        if hl != 0.0:
            w = w * T(1.0 / hl)
        if not (hl > eps * w_norm):
            breakdown = True
        vs.append(w)
        zs.append(z)
        # qr_insert of hcur as the last column of [[Q, 0], [0, 1]], [[R], [0]]
        Q2 = np.zeros((j + 2, j + 2), dtype=T)
        Q2[:j + 1, :j + 1] = Q
        Q2[j + 1, j + 1] = T(1)
        t = np.zeros(j + 2, dtype=T)
        for a in range(j + 2):
            s0 = T()
            for b2 in range(j + 2):
                s0 += _conj(Q2[b2, a], T) * hcur[b2]
            t[a] = s0
        cg, sg, rg = _givens(t[j], t[j + 1])
        R2 = np.zeros((j + 2, j + 1), dtype=T)
        for a in range(j + 1):
            for b2 in range(j):
                R2[a, b2] = R[a, b2]
        for a in range(j):
            R2[a, j] = t[a]
        R2[j, j] = T(rg)
        cgT = T(cg)
        sgT = T(sg)
        for a in range(j + 2):
            qa = Q2[a, j]
            qb = Q2[a, j + 1]
            Q2[a, j] = cgT * qa + _conj(sgT, T) * qb
            Q2[a, j + 1] = -sgT * qa + cgT * qb
        Q = Q2
        R = R2
        res = abs(Q[0, j + 1])
        if res < atol or breakdown:
            break
    if not math.isfinite(abs(R[j, j])):
        raise ValueError("LinAlgError")
    # y = lstsq(R[:j+1, :j+1], Q[0, :j+1].conj())  (upper triangular)
    y = np.zeros(j + 1, dtype=T)
    for i in range(j + 1):
        y[i] = _conj(Q[0, i], T)
    for i in range(j, -1, -1):
        s1 = y[i]
        for c2 in range(i + 1, j + 1):
            s1 -= R[i, c2] * y[c2]
        y[i] = s1 / R[i, i]
    return _FG[T](Q, R, np.ascontiguousarray(Bm[:, :j + 1]), vs, zs, y, res)

def _ident(x):
    return x

# ----------------------------------------------------------------- lgmres

def lgmres(A, b, x0 = None, rtol: float = 1e-05, atol: float = 0.0,
           maxiter: int = 1000, M = None, callback = None, inner_m: int = 30,
           outer_k: int = 3, outer_v = None, store_outer_Av: bool = True,
           prepend_outer_v: bool = False):
    Aop, x, bv = _make_system(A, M, x0, b)
    return _lgmres(Aop, x, bv, rtol, atol, maxiter, M, callback, inner_m, outer_k,
                   outer_v, store_outer_Av, prepend_outer_v)

def _lgmres(Aop, x, bv, rtol: float, atol: float, maxiter: int, M, callback,
            inner_m: int, outer_k: int, outer_v, store_outer_Av: bool,
            prepend_outer_v: bool):
    T = type(bv[0])
    Mop = _asM(M)
    if outer_v is None:
        ov = List[Tuple[np.ndarray[T, 1], np.ndarray[T, 1], bool]]()
    else:
        ov = outer_v
    b_norm = _nrm2(bv)
    atol = _get_atol(b_norm, atol, rtol)
    if b_norm == 0:
        return bv, 0
    ptol_max_factor = 1.0
    for k_outer in range(maxiter):
        r_outer = Aop.matvec(x) - bv
        if callback is not None:
            callback(x)
        r_norm = _nrm2(r_outer)
        if r_norm <= max(atol, rtol * b_norm):
            return x, 0
        v0 = -_psolve(Mop, r_outer)
        inner_res_0 = _nrm2(v0)
        if inner_res_0 == 0:
            raise RuntimeError(f"Preconditioner returned a zero vector; |v| ~ {r_norm}, |M v| = 0")
        v0 = v0 * T(1.0 / inner_res_0)
        ptol = min(ptol_max_factor, max(atol, rtol * b_norm) / r_norm)
        try:
            F = _fgmres(lambda u: Aop.matvec(u), v0, inner_m, ptol,
                        lambda u: _psolve(Mop, u), _ident,
                        List[np.ndarray[T, 1]](), ov, prepend_outer_v)
        except:
            return x, k_outer + 1
        y = F.y * T(inner_res_0)
        for i in range(len(y)):
            if not math.isfinite(abs(y[i])):
                return x, k_outer + 1
        if F.res > ptol:
            ptol_max_factor = min(1.0, 1.5 * ptol_max_factor)
        else:
            ptol_max_factor = max(1e-16, 0.25 * ptol_max_factor)
        dx = F.zs[0] * y[0]
        for q in range(1, len(F.zs)):
            dx = dx + F.zs[q] * y[q]
        nx = _nrm2(dx)
        if nx > 0:
            if store_outer_Av:
                qv = F.Q @ (F.R @ y)
                ax = F.vs[0] * qv[0]
                for q in range(1, len(F.vs)):
                    ax = ax + F.vs[q] * qv[q]
                ov.append((dx * T(1.0 / nx), ax * T(1.0 / nx), True))
            else:
                ov.append((dx * T(1.0 / nx), dx, False))
        while len(ov) > outer_k:
            ov.pop(0)
        x += dx
    return x, maxiter

# ----------------------------------------------------------------- gcrotmk

def gcrotmk(A, b, x0 = None, rtol: float = 1e-05, atol: float = 0.0,
            maxiter: int = 1000, M = None, callback = None, m: int = 20,
            k = None, CU = None, discard_C: bool = False, truncate: str = "oldest"):
    Aop, x, bv = _make_system(A, M, x0, b)
    return _gcrotmk(Aop, x, bv, x0 is None, rtol, atol, maxiter, M, callback, m, k,
                    CU, discard_C, truncate)

def _gcrotmk(Aop, x, bv, nox0: bool, rtol: float, atol: float, maxiter: int, M,
             callback, m: int, k, CU, discard_C: bool, truncate: str):
    T = type(bv[0])
    Mop = _asM(M)
    if truncate != "oldest" and truncate != "smallest":
        raise ValueError(f"Invalid value for 'truncate': {truncate!r}")
    if truncate == "smallest":
        raise NotImplementedError("gcrotmk: truncate='smallest' is not supported")
    kk = m if k is None else int(k)
    # CU: list of (c, u) pairs; c may be absent (has_c False)
    if CU is None:
        cu = List[Tuple[np.ndarray[T, 1], np.ndarray[T, 1], bool]]()
    else:
        cu = CU
    if nox0:
        r = bv.copy()
    else:
        r = bv - Aop.matvec(x)
    b_norm = _nrm2(bv)
    atol = _get_atol(b_norm, atol, rtol)
    if b_norm == 0:
        return bv, 0
    if discard_C:
        for i in range(len(cu)):
            cu[i] = (cu[i][1], cu[i][1], False)
    if len(cu) > 0:
        # re-orthonormalize the recycled space: C = A U, QR with pivoting
        raise NotImplementedError("gcrotmk: a non-empty CU on entry is not supported")
    j_outer = 0
    for jo in range(maxiter):
        j_outer = jo
        if callback is not None:
            callback(x)
        beta = _nrm2(r)
        beta_tol = max(atol, rtol * b_norm)
        if beta <= beta_tol and (j_outer > 0 or len(cu) > 0):
            r = bv - Aop.matvec(x)
            beta = _nrm2(r)
        if beta <= beta_tol:
            j_outer = -1
            break
        ml = m + max(kk - len(cu), 0)
        cs = List[np.ndarray[T, 1]]()
        for q in range(len(cu)):
            cs.append(cu[q][0])
        try:
            F = _fgmres(lambda u: Aop.matvec(u), r * T(1.0 / beta), ml,
                        max(atol, rtol * b_norm) / beta, _ident,
                        lambda u: _psolve(Mop, u), cs,
                        List[Tuple[np.ndarray[T, 1], np.ndarray[T, 1], bool]](), False)
        except:
            break
        y = F.y * T(beta)
        ux = F.zs[0] * y[0]
        for q in range(1, len(F.zs)):
            ux = ux + F.zs[q] * y[q]
        by = F.B @ y
        for q in range(len(cu)):
            ux = ux - cu[q][1] * by[q]
        hy = F.Q @ (F.R @ y)
        cx = F.vs[0] * hy[0]
        for q in range(1, len(F.vs)):
            cx = cx + F.vs[q] * hy[q]
        nc = _nrm2(cx)
        if nc == 0.0 or not math.isfinite(1.0 / nc):
            continue
        alpha = T(1.0 / nc)
        cx = cx * alpha
        ux = ux * alpha
        gamma = _dotc(cx, r, T)
        r = r - gamma * cx
        x = x + gamma * ux
        while len(cu) >= kk and len(cu) > 0:
            cu.pop(0)
        cu.append((cx, ux, True))
    cu.append((x.copy(), x.copy(), False))
    return x, j_outer + 1
