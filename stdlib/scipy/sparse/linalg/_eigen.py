# Copyright (C) 2026 the Typhon authors
# Ported from scipy 1.18.1 (BSD-3-Clause)

# eigsh / eigs / svds. scipy calls ARPACK (implicitly restarted Lanczos /
# Arnoldi). This port uses the mathematically equivalent Krylov-Schur
# restart (thick restart): a Krylov decomposition A V_m = V_{m+1} H is
# expanded by Arnoldi with full re-orthogonalization (twice), the wanted Ritz
# vectors of H span the restarted subspace, and pairs are accepted when
# |h_{m+1} y_i| <= tol * |theta_i| (ARPACK's criterion, tol = eps default).
# Converged eigenvalues agree with ARPACK to rounding; the start vector is
# deterministic (ARPACK draws a random one), so eigenvectors may differ by a
# phase (sign). Output ordering follows scipy: eigsh returns ascending
# eigenvalues; eigs returns them in ARPACK's "wanted" order.

import math
import numpy as np
from .._base import _compressed, _coo, _up, _is_cplx, _issparse
from ._interface import aslinearoperator, _is_linop
from ._dsolve import splu, _as_csc

_EPS = 2.220446049250313e-16

class ArpackNoConvergence(Exception):
    def __init__(self, msg: str = ""):
        super().__init__(msg)

class ArpackError(Exception):
    def __init__(self, msg: str = ""):
        super().__init__(msg)

def _start_vector(n: int, T: type):
    v = np.empty(n, dtype=T)
    for i in range(n):
        v[i] = T(1.0 + 0.5 * math.sin(1.7 * i + 0.3) + 0.25 * math.cos(0.37 * i * i + 1.1))
    return v

def _key(theta: complex, which: str) -> float:
    # larger key = more wanted
    if which == "LM":
        return abs(theta)
    elif which == "SM":
        return -abs(theta)
    elif which == "LR" or which == "LA":
        return theta.real
    elif which == "SR" or which == "SA":
        return -theta.real
    elif which == "LI":
        return abs(theta.imag)
    elif which == "SI":
        return -abs(theta.imag)
    else:
        raise ValueError(f"which must be one of LM, SM, LR, SR, LI, SI, LA, SA, BE (got {which})")
        return 0.0

def _wanted(thetas: List[complex], k: int, which: str) -> List[int]:
    m = len(thetas)
    idx = list(range(m))
    if which == "BE":
        # alternate from both ends of the algebraic spectrum (ARPACK BE)
        srt = sorted(idx, key=lambda i: thetas[i].real)
        out = List[int]()
        lo = 0
        hi = m - 1
        take_hi = True
        while len(out) < k and lo <= hi:
            if take_hi:
                out.append(srt[hi])
                hi -= 1
            else:
                out.append(srt[lo])
                lo += 1
            take_hi = not take_hi
        return out
    srt = sorted(idx, key=lambda i: -_key(thetas[i], which))
    return srt[:k]

def _orth_against(V, j: int, w, T: type):
    # classical Gram-Schmidt twice (DGKS-like); returns (h, w)
    h = np.zeros(j, dtype=T)
    for _ in range(2):
        for i in range(j):
            vi = V[:, i]
            if _is_cplx(T):
                c = np.vdot(vi, w)
            else:
                c = T()
                for q in range(len(w)):
                    c += vi[q] * w[q]
            h[i] += c
            w = w - c * vi
    return h, w

def _krylov_schur(op, n: int, k: int, which: str, ncv: int, maxiter: int,
                  tol: float, v0, herm: bool, T: type):
    # returns (thetas: List[complex], Y: n x k complex/real basis vectors)
    m = ncv
    V = np.zeros((n, m + 1), dtype=T)
    H = np.zeros((m + 1, m), dtype=T)
    if v0 is None:
        v = _start_vector(n, T)
    else:
        v = np.asarray(v0).astype(T).copy()
    nv = float(np.linalg.norm(v))
    for q in range(n):
        V[q, 0] = v[q] / T(nv)
    kk = 0                       # current size of the kept decomposition
    tol_eff = _EPS if tol <= 0 else tol
    it = 0
    C = complex
    best_th = List[complex]()
    best_vec = np.zeros((m, k), dtype=C)
    nconv = 0
    while True:
        it += 1
        # expand from kk to m
        for j in range(kk, m):
            w = np.asarray(op(V[:, j])).astype(T)
            h, w = _orth_against(V, j + 1, w, T)
            for i in range(j + 1):
                H[i, j] = h[i]
            beta = float(np.linalg.norm(w))
            H[j + 1, j] = T(beta)
            if beta < 1e-300:
                # invariant subspace found: restart direction orthogonal to V
                r = _start_vector(n, T)
                for q in range(n):
                    r[q] = r[q] * T(1.0 + q % 7)
                h2, r = _orth_against(V, j + 1, r, T)
                nr = float(np.linalg.norm(r))
                for q in range(n):
                    V[q, j + 1] = r[q] / T(nr)
                H[j + 1, j] = T()
            else:
                for q in range(n):
                    V[q, j + 1] = w[q] / T(beta)
        Hm = np.ascontiguousarray(H[:m, :m])
        brow = H[m, :m]
        # Ritz pairs of Hm
        if herm:
            Hs = (Hm + np.conj(Hm.T)) * T(0.5) if _is_cplx(T) else (Hm + Hm.T) * 0.5
            wv, Z = np.linalg.eigh(Hs)
            th = [complex(float(wv[i]), 0.0) for i in range(m)]
            Zc = Z.astype(C)
        else:
            wv2, Z2 = np.linalg.eig(Hm.astype(C))
            th = [complex(wv2[i]) for i in range(m)]
            Zc = Z2.astype(C)
        sel = _wanted(th, k, which)
        # convergence of the wanted pairs
        conv = 0
        for i in sel:
            y = Zc[:, i]
            ny = float(np.linalg.norm(y))
            res = 0.0
            r0 = complex()
            for q in range(m):
                r0 += complex(brow[q]) * y[q]
            res = abs(r0) / ny
            if res <= tol_eff * max(abs(th[i]), _EPS ** (2.0 / 3.0)):
                conv += 1
        if conv >= k or it > maxiter:
            best_th = [th[i] for i in sel]
            for c in range(k):
                for q in range(m):
                    best_vec[q, c] = Zc[q, sel[c]]
            nconv = conv
            break
        # thick restart: keep p = k + (m - k) // 2 wanted Ritz vectors
        p = min(m - 1, k + (m - k) // 2)
        keep = _wanted(th, p, which)
        if herm:
            # real symmetric (or Hermitian) case: eigenvectors are orthonormal
            Q = np.zeros((m, p), dtype=T)
            for c in range(p):
                for q in range(m):
                    if _is_cplx(T):
                        Q[q, c] = T(Zc[q, keep[c]])
                    else:
                        Q[q, c] = Zc[q, keep[c]].real
        else:
            # keep conjugate pairs together for real data
            if not _is_cplx(T):
                # real basis of the invariant subspace: (Re z, Im z) for each
                # conjugate pair, which must never be split (otherwise
                # span(Q) is not Hm-invariant and the Krylov relation breaks)
                cols = List[np.ndarray[float, 1]]()
                used = np.zeros(m, dtype=bool)
                last_pair = False
                for i in keep:
                    if used[i]:
                        continue
                    used[i] = True
                    zi = Zc[:, i]
                    if abs(th[i].imag) > 1e-10 * abs(th[i]):
                        best = -1
                        bd = 1e300
                        for j2 in range(m):
                            if not used[j2]:
                                dd = abs(th[j2] - th[i].conjugate())
                                if dd < bd:
                                    bd = dd
                                    best = j2
                        if best >= 0:
                            used[best] = True
                        cols.append(np.ascontiguousarray(zi.real))
                        cols.append(np.ascontiguousarray(zi.imag))
                        last_pair = True
                    else:
                        cols.append(np.ascontiguousarray(zi.real))
                        last_pair = False
                p = len(cols)
                if p > m - 1:
                    p -= 2 if last_pair else 1
                Qr = np.zeros((m, p), dtype=float)
                for c in range(p):
                    for q in range(m):
                        Qr[q, c] = cols[c][q]
                Qq, _ = np.linalg.qr(Qr)
                Q = Qq.astype(T)
            else:
                Qc = np.zeros((m, p), dtype=C)
                for c in range(p):
                    for q in range(m):
                        Qc[q, c] = Zc[q, keep[c]]
                Qq2, _ = np.linalg.qr(Qc)
                Q = Qq2.astype(T)
        Vk = V[:, :m] @ Q
        QH = np.conj(Q.T) if _is_cplx(T) else Q.T
        Hk = QH @ Hm @ Q
        bk = brow @ Q
        vlast = V[:, m].copy()
        V = np.zeros((n, m + 1), dtype=T)
        H = np.zeros((m + 1, m), dtype=T)
        for c in range(p):
            for q in range(n):
                V[q, c] = Vk[q, c]
            for r in range(p):
                H[r, c] = Hk[r, c]
            H[p, c] = bk[c]
        for q in range(n):
            V[q, p] = vlast[q]
        kk = p
    if nconv < k:
        raise ArpackNoConvergence(f"ARPACK error -1: No convergence ({it} iterations, {nconv}/{k} eigenvectors converged)")
    X = V[:, :m].astype(C) @ best_vec
    return best_th, X

def _apply_A(A):
    if _is_linop(A):
        return A
    else:
        return aslinearoperator(A)

def _op_dtype(A):
    if _issparse(A):
        return A.data.dtype()
    elif isinstance(A, np.ndarray):
        return A.dtype()
    else:
        return A.dtype()

def eigsh(A, k: int = 6, M = None, sigma = None, which: str = "LM", v0 = None,
          ncv = None, maxiter = None, tol: float = 0.0,
          return_eigenvectors: Literal[bool] = True, Minv = None, OPinv = None,
          mode: str = "normal"):
    Aop = _apply_A(A)
    n = Aop.shape[0]
    if Aop.shape[0] != Aop.shape[1]:
        raise ValueError(f"expected square matrix (shape={Aop.shape})")
    if k <= 0 or k >= n:
        raise ValueError(f"k must be greater than 0 and smaller than n={n}, got k={k}")
    T = type(_up(type(_op_dtype(A)), float))
    nc = min(n, max(2 * k + 1, 20)) if ncv is None else ncv
    if nc > n or nc <= k:
        raise ValueError(f"ncv must be k<ncv<=n, ncv={nc}")
    mi = n * 10 if maxiter is None else maxiter
    if sigma is None:
        th, X = _krylov_schur(lambda x: Aop.matvec(x), n, k, which, nc, mi, tol,
                              v0, True, T)
        lam = [t.real for t in th]
    else:
        s = float(sigma)
        C = _as_csc(A)
        Sh = _shifted(C.astype(T), T(s))
        lu = splu(Sh)
        wh = "LM" if which == "LM" else which
        th, X = _krylov_schur(lambda x: lu.solve(x), n, k, wh, nc, mi, tol, v0,
                              True, T)
        lam = [s + 1.0 / t.real for t in th]
    if not return_eigenvectors:
        # dseupd with rvec=False: ordered on the operator's Ritz values in
        # regular mode; after the shift-invert back-transformation dseupd
        # sorts the eigenvalues ascending (dsesrt 'LA')
        if sigma is None:
            o2 = _arpack_noev_order(th, which)
        else:
            o2 = sorted(range(k), key=lambda i: lam[i])
        return np.array([lam[i] for i in o2], dtype=float)
    # rvec=True: ascending algebraic order (dseupd)
    order = sorted(range(k), key=lambda i: lam[i])
    w = np.array([lam[i] for i in order], dtype=float)
    Vout = np.empty((n, k), dtype=T)
    for c in range(k):
        col = X[:, order[c]]
        # normalize and fix phase so the largest component is positive real
        nrm = float(np.linalg.norm(col))
        big = 0
        for q in range(n):
            if abs(col[q]) > abs(col[big]):
                big = q
        ph = col[big] / abs(col[big]) if abs(col[big]) > 0 else complex(1.0)
        for q in range(n):
            z = col[q] / ph / nrm
            if _is_cplx(T):
                Vout[q, c] = T(z)
            else:
                Vout[q, c] = z.real
    return w, Vout

def _arpack_noev_order(th: List[complex], which: str) -> List[int]:
    # order of the Ritz values returned by ARPACK's d/zseupd, dneupd with
    # rvec=False: the wanted values sorted so that the "most wanted" comes
    # last (dsortr / dsortc conventions); th are the operator's Ritz values
    # (nu = 1/(lam - sigma) in shift-invert mode)
    idx = list(range(len(th)))
    if which == "LM":
        return sorted(idx, key=lambda i: abs(th[i]))
    elif which == "SM":
        return sorted(idx, key=lambda i: -abs(th[i]))
    elif which == "LA" or which == "LR" or which == "BE":
        return sorted(idx, key=lambda i: th[i].real)
    elif which == "SA" or which == "SR":
        return sorted(idx, key=lambda i: -th[i].real)
    elif which == "LI":
        return sorted(idx, key=lambda i: abs(th[i].imag))
    else:
        return sorted(idx, key=lambda i: -abs(th[i].imag))

def _rkey(t: complex, which: str) -> float:
    if which == "LR" or which == "SR":
        return t.real
    elif which == "LI" or which == "SI":
        return abs(t.imag)
    else:
        return abs(t)

def _eigs_rvec_order(th: List[complex], which: str) -> List[int]:
    # scipy _UnsymmetricArpackParams.extract (real case): argsort of the
    # operator values by the `which` key, largest-first for LM/LR/LI and
    # smallest-first for SM/SR/SI. A conjugate pair is kept together with
    # the positive imaginary part first, as dneupd stores it.
    k = len(th)
    base = sorted(list(range(k)), key=lambda i: -th[i].imag)
    if which == "LR" or which == "LM" or which == "LI":
        ordd = sorted(base, key=lambda i: -_rkey(th[i], which))
    else:
        ordd = sorted(base, key=lambda i: _rkey(th[i], which))
    # restore (+imag, -imag) adjacency for pairs with equal keys
    out = List[int]()
    used = [False] * k
    for q in range(k):
        i = ordd[q]
        if used[i]:
            continue
        used[i] = True
        out.append(i)
        if th[i].imag > 0:
            for r in range(q + 1, k):
                j = ordd[r]
                if not used[j] and th[j] == th[i].conjugate():
                    used[j] = True
                    out.append(j)
                    break
    return out

def _shifted(C, s):
    # C - s I in CSC
    n = C.shape[0]
    T = type(s)
    rows = List[int]()
    cols = List[int]()
    vals = List[T]()
    for j in range(n):
        for p in range(C.indptr[j], C.indptr[j + 1]):
            rows.append(C.indices[p])
            cols.append(j)
            vals.append(C.data[p])
        rows.append(j)
        cols.append(j)
        vals.append(-s)
    from .._construct import coo_array
    return coo_array((np.array(vals, dtype=T), (np.array(rows), np.array(cols))),
                     shape=(n, n)).tocsc()

def eigs(A, k: int = 6, M = None, sigma = None, which: str = "LM", v0 = None,
         ncv = None, maxiter = None, tol: float = 0.0,
         return_eigenvectors: Literal[bool] = True, Minv = None, OPinv = None,
         OPpart = None):
    Aop = _apply_A(A)
    n = Aop.shape[0]
    if Aop.shape[0] != Aop.shape[1]:
        raise ValueError(f"expected square matrix (shape={Aop.shape})")
    if k <= 0 or k >= n - 1:
        raise ValueError(f"k={k} must be greater than 0 and less than n-1={n - 1}")
    T = type(_up(type(_op_dtype(A)), float))
    nc = min(n, max(2 * k + 1, 20)) if ncv is None else ncv
    mi = n * 10 if maxiter is None else maxiter
    if sigma is None:
        th, X = _krylov_schur(lambda x: Aop.matvec(x), n, k, which, nc, mi, tol,
                              v0, False, T)
        lam = th
    else:
        C = _as_csc(A)
        if _is_cplx(type(sigma)) or _is_cplx(T):
            Sh = _shifted(C.astype(complex), complex(sigma))
            lu = splu(Sh)
            th, X = _krylov_schur(lambda x: lu.solve(x.astype(complex)), n, k, which,
                                  nc, mi, tol, v0, False, complex)
            lam = [complex(sigma) + 1.0 / t for t in th]
        else:
            Sh2 = _shifted(C.astype(T), T(sigma))
            lu2 = splu(Sh2)
            th, X = _krylov_schur(lambda x: lu2.solve(x), n, k, which, nc, mi, tol,
                                  v0, False, T)
            lam = [complex(float(sigma)) + 1.0 / t for t in th]
    if not return_eigenvectors:
        o2 = _arpack_noev_order(th, which)
        return np.array([lam[i] for i in o2], dtype=complex)
    order = _eigs_rvec_order(th, which)
    w = np.array([lam[i] for i in order], dtype=complex)
    Vout = np.empty((n, k), dtype=complex)
    for c in range(k):
        col = X[:, order[c]]
        nrm = float(np.linalg.norm(col))
        for q in range(n):
            Vout[q, c] = col[q] / nrm
    return w, Vout

def svds(A, k: int = 6, ncv = None, tol: float = 0.0, which: str = "LM", v0 = None,
         maxiter = None, return_singular_vectors: Literal[bool] = True,
         solver: str = "arpack", rng = None, options = None):
    Aop = _apply_A(A)
    n, m = Aop.shape
    if k <= 0 or k >= min(n, m):
        raise ValueError("`k` must be an integer satisfying `0 < k < min(A.shape)`.")
    T = type(_up(type(_op_dtype(A)), float))
    transpose = n < m
    mn = min(n, m)
    def xhx(x):
        if transpose:
            return Aop.matvec(Aop.rmatvec(x))
        else:
            return Aop.rmatvec(Aop.matvec(x))
    nc = min(mn, max(2 * k + 1, 20)) if ncv is None else ncv
    mi = mn * 10 if maxiter is None else maxiter
    th, X = _krylov_schur(xhx, mn, k, which, nc, mi, tol * tol, v0, True, T)
    E = np.empty((mn, k), dtype=T)
    for c in range(k):
        for q in range(mn):
            E[q, c] = T(X[q, c]) if _is_cplx(T) else X[q, c].real
    eigvec, _ = np.linalg.qr(E)
    if transpose:
        Av = Aop.rmatmat(eigvec)
    else:
        Av = Aop.matmat(eigvec)
    u, s, vh = np.linalg.svd(Av, full_matrices=False)
    kk = len(s)
    if not return_singular_vectors:
        return np.array([s[kk - 1 - i] for i in range(kk)])
    s2 = np.array([s[kk - 1 - i] for i in range(kk)])
    u2 = np.ascontiguousarray(u[:, ::-1])
    vh2 = np.ascontiguousarray(vh[::-1, :])
    if transpose:
        U = eigvec @ (np.conj(vh2.T) if _is_cplx(T) else vh2.T)
        VH = np.conj(u2.T) if _is_cplx(T) else np.ascontiguousarray(u2.T)
    else:
        U = u2
        VH = vh2 @ (np.conj(eigvec.T) if _is_cplx(T) else eigvec.T)
    return U, s2, VH
