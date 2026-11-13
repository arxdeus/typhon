# Copyright (C) 2026 the Typhon authors
# Ported from scipy 1.18.1 (BSD-3-Clause)
# place_poles: pole placement by Kautsky/Nichols/Van Dooren (KNV0) and
# Tits/Yang (YT) (scipy/signal/_ltisys.py). Deviations: poles are always held
# as a complex array (`requested_poles`/`computed_poles` are complex even when
# all poles are real); `X` is complex; `rtol`/`nb_iter` are NaN when B is full
# rank; the convergence warning is not emitted (check `.rtol`).

import math
import numpy as np
from numpy.linalg import det, svd, eig, norm, solve, matrix_rank, lstsq
from scipy.linalg import qr as s_qr

class PlacePolesResult:
    gain_matrix: np.ndarray[float, 2]
    computed_poles: np.ndarray[complex, 1]
    requested_poles: np.ndarray[complex, 1]
    X: np.ndarray[complex, 2]
    rtol: float
    nb_iter: float

    def __init__(self, gain_matrix: np.ndarray[float, 2], computed_poles: np.ndarray[complex, 1],
                 requested_poles: np.ndarray[complex, 1], X: np.ndarray[complex, 2], rtol: float, nb_iter: float):
        self.gain_matrix = gain_matrix
        self.computed_poles = computed_poles
        self.requested_poles = requested_poles
        self.X = X
        self.rtol = rtol
        self.nb_iter = nb_iter

def _order_complex_poles(poles: np.ndarray[complex, 1]) -> np.ndarray[complex, 1]:
    n = poles.shape[0]
    reals = List[float]()
    ims = List[complex]()
    for k in range(n):
        if poles[k].imag == 0.0:
            reals.append(poles[k].real)
    reals.sort()
    neg = List[complex]()
    for k in range(n):
        if poles[k].imag < 0.0:
            neg.append(poles[k])
    # np.sort of complex: lexicographic (real, imag)
    for a in range(1, len(neg)):
        v = neg[a]
        b = a - 1
        while b >= 0 and (neg[b].real > v.real or (neg[b].real == v.real and neg[b].imag > v.imag)):
            neg[b + 1] = neg[b]
            b -= 1
        neg[b + 1] = v
    for p in neg:
        c = p.conjugate()
        found = False
        for k in range(n):
            if poles[k] == c:
                found = True
        if found:
            ims.append(p)
            ims.append(c)
    if n != len(reals) + len(ims):
        raise ValueError("Complex poles must come with their conjugates")
    out = np.empty(n, dtype=complex)
    for k in range(len(reals)):
        out[k] = complex(reals[k], 0.0)
    for k in range(len(ims)):
        out[len(reals) + k] = ims[k]
    return out

def _col(M: np.ndarray[float, 2], k: int) -> np.ndarray[float, 1]:
    n = M.shape[0]
    out = np.empty(n)
    for r in range(n):
        out[r] = M[r, k]
    return out

def _colmat(v: np.ndarray[float, 1]) -> np.ndarray[float, 2]:
    n = v.shape[0]
    out = np.empty((n, 1))
    for r in range(n):
        out[r, 0] = v[r]
    return out

def _delete_cols(M: np.ndarray[float, 2], i: int, j: int) -> np.ndarray[float, 2]:
    n = M.shape[0]
    m = M.shape[1]
    keep = List[int]()
    for c in range(m):
        if c != i and c != j:
            keep.append(c)
    out = np.empty((n, len(keep)))
    for k in range(len(keep)):
        for r in range(n):
            out[r, k] = M[r, keep[k]]
    return out

def _allclose_zero(M: np.ndarray[float, 2]) -> bool:
    for r in range(M.shape[0]):
        for c in range(M.shape[1]):
            if abs(M[r, c]) > 1e-8:
                return False
    return True

def _allclose_scalar(a: float, b: float) -> bool:
    return abs(a - b) <= 1e-8 + 1e-5 * abs(b)

def _fnorm(M: np.ndarray[float, 2]) -> float:
    s = 0.0
    for r in range(M.shape[0]):
        for c in range(M.shape[1]):
            s += M[r, c] * M[r, c]
    return math.sqrt(s)

def _KNV0(ker_pole: List[np.ndarray[float, 2]], tm: np.ndarray[float, 2], j: int):
    tnj = _delete_cols(tm, j, -1)
    qr_ = s_qr(tnj, mode='full')
    Q = qr_[0]
    n = tm.shape[0]
    qlast = _col(Q, Q.shape[1] - 1)
    kj = ker_pole[j]
    mat_ker_pj = np.dot(kj, kj.T)
    yj = np.dot(mat_ker_pj, qlast)
    allz = True
    for r in range(n):
        if abs(yj[r]) > 1e-8:
            allz = False
    if not allz:
        s = 0.0
        for r in range(n):
            s += yj[r] * yj[r]
        s = math.sqrt(s)
        for r in range(n):
            tm[r, j] = yj[r] / s

def _YT_real(ker_pole: List[np.ndarray[float, 2]], Q: np.ndarray[float, 2], tm: np.ndarray[float, 2], i: int, j: int):
    n = tm.shape[0]
    nq = Q.shape[1]
    u = _colmat(_col(Q, nq - 2))
    v = _colmat(_col(Q, nq - 1))
    ki = ker_pole[i]
    kj = ker_pole[j]
    m = np.dot(np.dot(ki.T, np.dot(u, v.T) - np.dot(v, u.T)), kj)
    sv = svd(m)
    um = sv[0]
    sm = sv[1]
    vm = sv[2]
    mu1 = _colmat(_col(um, 0))
    mu2 = _colmat(_col(um, 1))
    nu1 = np.empty((vm.shape[1], 1))
    nu2 = np.empty((vm.shape[1], 1))
    for r in range(vm.shape[1]):
        nu1[r, 0] = vm[0, r]
        nu2[r, 0] = vm[1, r]
    tji = np.empty((2 * n, 1))
    for r in range(n):
        tji[r, 0] = tm[r, i]
        tji[n + r, 0] = tm[r, j]
    kpmn = np.zeros((1, 1))
    if not _allclose_scalar(sm[0], sm[1]):
        a = np.dot(ki, mu1)
        b = np.dot(kj, nu1)
        kpmn = np.vstack((a, b))
    else:
        kij = np.vstack((np.hstack((ki, np.zeros(ki.shape))), np.hstack((np.zeros(kj.shape), kj))))
        mn = np.vstack((np.hstack((mu1, mu2)), np.hstack((nu1, nu2))))
        kpmn = np.dot(kij, mn)
    tij = np.dot(np.dot(kpmn, kpmn.T), tji)
    if not _allclose_zero(tij):
        sc = math.sqrt(2.0) / _fnorm(tij)
        for r in range(n):
            tm[r, i] = tij[r, 0] * sc
            tm[r, j] = tij[n + r, 0] * sc
    else:
        for r in range(n):
            tm[r, i] = kpmn[r, 0]
            tm[r, j] = kpmn[n + r, 0]

def _YT_complex(ker_c: List[np.ndarray[complex, 2]], Q: np.ndarray[float, 2], tm: np.ndarray[float, 2], i: int, j: int):
    n = tm.shape[0]
    nq = Q.shape[1]
    s2 = math.sqrt(2.0)
    u = np.empty((n, 1), dtype=complex)
    for r in range(n):
        u[r, 0] = complex(s2 * Q[r, nq - 2], s2 * Q[r, nq - 1])
    kij = ker_c[i]
    uc = u.conj()
    m = np.dot(np.dot(kij.conj().T, np.dot(u, uc.T) - np.dot(uc, u.T)), kij)
    ev = eig(m)
    e_val = ev[0]
    e_vec = ev[1]
    ne = e_val.shape[0]
    mag = np.empty(ne)
    for k in range(ne):
        mag[k] = abs(e_val[k])
    idx = np.argsort(mag)
    mu1 = np.empty((e_vec.shape[0], 1), dtype=complex)
    mu2 = np.empty((e_vec.shape[0], 1), dtype=complex)
    for r in range(e_vec.shape[0]):
        mu1[r, 0] = e_vec[r, idx[ne - 1]]
        mu2[r, 0] = e_vec[r, idx[ne - 2]]
    tji = np.empty((n, 1), dtype=complex)
    for r in range(n):
        tji[r, 0] = complex(tm[r, i], tm[r, j])
    a1 = abs(e_val[idx[ne - 1]])
    a2 = abs(e_val[idx[ne - 2]])
    kmu = np.dot(kij, mu1)
    if _allclose_scalar(a1, a2):
        kmu = np.dot(kij, np.hstack((mu1, mu2)))
    tij = np.dot(np.dot(kmu, kmu.conj().T), tji)
    allz = True
    for r in range(n):
        if abs(tij[r, 0]) > 1e-8:
            allz = False
    if not allz:
        s = 0.0
        for r in range(n):
            s += tij[r, 0].real ** 2 + tij[r, 0].imag ** 2
        s = math.sqrt(s)
        for r in range(n):
            tm[r, i] = tij[r, 0].real / s
            tm[r, j] = tij[r, 0].imag / s
    else:
        for r in range(n):
            tm[r, i] = kmu[r, 0].real
            tm[r, j] = kmu[r, 0].imag

def _is_real(p: complex) -> bool:
    return p.imag == 0.0

def _YT_order(poles: np.ndarray[complex, 1]) -> List[Tuple[int, int]]:
    npol = poles.shape[0]
    nb_real = 0
    for k in range(npol):
        if _is_real(poles[k]):
            nb_real += 1
    hnb = nb_real // 2
    a = List[int]()
    b = List[int]()
    if nb_real > 0:
        a.append(nb_real)
        b.append(1)
    r_comp = List[int]()
    c = nb_real + 1
    while c < npol + 1:
        r_comp.append(c)
        c += 2
    first_real = _is_real(poles[0])
    # step 1.a
    for r in range(1, hnb + nb_real % 2):
        a.append(2 * r)
        b.append(2 * r + 1)
    # 1.b
    for c in r_comp:
        a.append(c)
        b.append(c + 1)
    # 1.c
    for r in range(1, hnb + 1):
        a.append(2 * r - 1)
        b.append(2 * r)
    # 1.d
    if hnb == 0 and first_real:
        a.append(1)
        b.append(1)
    for c in r_comp:
        a.append(c)
        b.append(c + 1)
    # 2.a
    for jj in range(2, hnb + nb_real % 2):
        for ii in range(1, hnb + 1):
            a.append(ii)
            b.append(ii + jj)
    # 2.b
    if hnb == 0 and first_real:
        a.append(1)
        b.append(1)
    for c in r_comp:
        a.append(c)
        b.append(c + 1)
    # 2.c
    for jj in range(2, hnb + nb_real % 2):
        for ii in range(hnb + 1, nb_real + 1):
            idx1 = ii + jj
            if idx1 > nb_real:
                idx1 = ii + jj - nb_real
            a.append(ii)
            b.append(idx1)
    # 2.d
    if hnb == 0 and first_real:
        a.append(1)
        b.append(1)
    for c in r_comp:
        a.append(c)
        b.append(c + 1)
    # 3.a
    for ii in range(1, hnb + 1):
        a.append(ii)
        b.append(ii + hnb)
    # 3.b
    if hnb == 0 and first_real:
        a.append(1)
        b.append(1)
    for c in r_comp:
        a.append(c)
        b.append(c + 1)
    out = List[Tuple[int, int]]()
    for k in range(len(a)):
        out.append((a[k] - 1, b[k] - 1))
    return out

def _YT_loop(ker_pole: List[np.ndarray[float, 2]], ker_c: List[np.ndarray[complex, 2]], tm: np.ndarray[float, 2], poles: np.ndarray[complex, 1],
             maxiter: int, rtol: float):
    order = _YT_order(poles)
    stop = False
    nb_try = 0
    cur_rtol = 0.0
    sp = math.sqrt(2.220446049250313e-16)
    while nb_try < maxiter and not stop:
        detb = abs(det(tm))
        for pr in order:
            i = pr[0]
            j = pr[1]
            if i == j:
                _KNV0(ker_pole, tm, i)
            else:
                tnij = _delete_cols(tm, i, j)
                Q = s_qr(tnij, mode='full')[0]
                if _is_real(poles[i]):
                    _YT_real(ker_pole, Q, tm, i, j)
                else:
                    _YT_complex(ker_c, Q, tm, i, j)
        dt = max(sp, abs(det(tm)))
        cur_rtol = abs((dt - detb) / dt)
        if cur_rtol < rtol and dt > sp:
            stop = True
        nb_try += 1
    return stop, cur_rtol, nb_try

def _KNV0_loop(ker_pole: List[np.ndarray[float, 2]], tm: np.ndarray[float, 2], nrows: int, maxiter: int, rtol: float):
    stop = False
    nb_try = 0
    cur_rtol = 0.0
    sp = math.sqrt(2.220446049250313e-16)
    while nb_try < maxiter and not stop:
        detb = abs(det(tm))
        for j in range(nrows):
            _KNV0(ker_pole, tm, j)
        dt = max(sp, abs(det(tm)))
        cur_rtol = abs((dt - detb) / dt)
        if cur_rtol < rtol and dt > sp:
            stop = True
        nb_try += 1
    return stop, cur_rtol, nb_try

def place_poles(A, B, poles, method: str = "YT", rtol: float = 1e-3, maxiter: int = 30):
    Aa = np.asarray(A).astype(float)
    Ba = np.asarray(B).astype(float)
    pa = np.asarray(poles)
    if pa.ndim > 1:
        raise ValueError("Poles must be a 1D array like.")
    pc = pa.astype(complex).reshape(pa.size)
    pord = _order_complex_poles(pc)
    if Aa.ndim != 2:
        raise ValueError("A must be a 2D array/matrix.")
    if Ba.ndim != 2:
        raise ValueError("B must be a 2D array/matrix")
    if Aa.shape[0] != Aa.shape[1]:
        raise ValueError("A must be square")
    n = Aa.shape[0]
    if pord.shape[0] > n:
        raise ValueError(f"maximum number of poles is {n} but you asked for {pord.shape[0]}")
    if pord.shape[0] < n:
        raise ValueError(f"number of poles is {pord.shape[0]} but you should provide {n}")
    rankB = int(matrix_rank(Ba))
    for k in range(n):
        cnt = 0
        for l in range(n):
            if pord[l] == pord[k]:
                cnt += 1
        if cnt > rankB:
            raise ValueError("at least one of the requested pole is repeated more than rank(B) times")
    if method != 'KNV0' and method != 'YT':
        raise ValueError("The method keyword must be one of 'YT' or 'KNV0'")
    if method == 'KNV0':
        for k in range(n):
            if not _is_real(pord[k]):
                raise ValueError("Complex poles are not supported by KNV0")
    if maxiter < 1:
        raise ValueError("maxiter must be at least equal to 1")
    if rtol > 1:
        raise ValueError("rtol can not be greater than 1")
    cur_rtol = 0.0
    nb_iter = 0.0
    qrb = s_qr(Ba, mode='full')
    u = qrb[0]
    zf = qrb[1]
    nb = Ba.shape[0]
    u0 = np.empty((nb, rankB))
    for r in range(nb):
        for c in range(rankB):
            u0[r, c] = u[r, c]
    u1 = np.empty((nb, nb - rankB))
    for r in range(nb):
        for c in range(nb - rankB):
            u1[r, c] = u[r, rankB + c]
    z = np.empty((rankB, zf.shape[1]))
    for r in range(rankB):
        for c in range(zf.shape[1]):
            z[r, c] = zf[r, c]
    gain = np.zeros((1, 1))
    X = np.zeros((n, n), dtype=complex)
    if nb == rankB:
        diag_poles = np.zeros(Aa.shape)
        idx = 0
        while idx < n:
            p = pord[idx]
            diag_poles[idx, idx] = p.real
            if not _is_real(p):
                diag_poles[idx, idx + 1] = -p.imag
                diag_poles[idx + 1, idx + 1] = p.real
                diag_poles[idx + 1, idx] = p.imag
                idx += 1
            idx += 1
        gain = lstsq(Ba, diag_poles - Aa, -1.0)[0]
        for r in range(n):
            X[r, r] = complex(1.0, 0.0)
        cur_rtol = float('nan')
        nb_iter = float('nan')
    else:
        ker_pole = List[np.ndarray[float, 2]]()
        ker_c = List[np.ndarray[complex, 2]]()
        skip = False
        tm = np.zeros((nb, 0))
        for j in range(nb):
            if skip:
                skip = False
                continue
            pole_space_j = np.empty((n, nb - rankB))
            # (u1.T @ (A - p I)).T   (real poles only; complex handled below)
            pj = pord[j]
            M = np.empty((nb - rankB, n), dtype=complex)
            for r in range(nb - rankB):
                for c in range(n):
                    acc = complex(0.0, 0.0)
                    for k in range(n):
                        a_kc = Aa[k, c] - (pj if k == c else complex(0.0, 0.0))
                        acc += u1[k, r] * a_kc
                    M[r, c] = acc
            if _is_real(pj):
                for r in range(n):
                    for c in range(nb - rankB):
                        pole_space_j[r, c] = M[c, r].real
                Q = s_qr(pole_space_j, mode='full')[0]
                kc = Q.shape[1] - (nb - rankB)
                kj = np.empty((n, kc))
                for r in range(n):
                    for c in range(kc):
                        kj[r, c] = Q[r, (nb - rankB) + c]
                tj = np.empty((n, 1))
                nrm = 0.0
                for r in range(n):
                    sacc = 0.0
                    for c in range(kc):
                        sacc += kj[r, c]
                    tj[r, 0] = sacc
                    nrm += sacc * sacc
                nrm = math.sqrt(nrm)
                for r in range(n):
                    tj[r, 0] = tj[r, 0] / nrm
                ker_pole.append(kj)
                ker_c.append(np.zeros((0, 0), dtype=complex))
                if tm.shape[1] == 0:
                    tm = tj
                else:
                    tm = np.hstack((tm, tj))
            else:
                psc = M.T.copy()
                Qc = s_qr(psc, mode='full')[0]
                nk = nb - rankB
                kcj = np.empty((n, Qc.shape[1] - nk), dtype=complex)
                for r in range(n):
                    for c in range(Qc.shape[1] - nk):
                        kcj[r, c] = Qc[r, nk + c]
                tsum = np.empty(n, dtype=complex)
                nrm = 0.0
                for r in range(n):
                    acc = complex(0.0, 0.0)
                    for c in range(kcj.shape[1]):
                        acc += kcj[r, c]
                    tsum[r] = acc
                    nrm += acc.real ** 2 + acc.imag ** 2
                nrm = math.sqrt(nrm)
                tj2 = np.empty((n, 2))
                for r in range(n):
                    v = tsum[r] / nrm
                    tj2[r, 0] = v.real
                    tj2[r, 1] = v.imag
                ker_pole.append(np.zeros((0, 0)))
                ker_pole.append(np.zeros((0, 0)))
                ker_c.append(kcj)
                ker_c.append(kcj)
                if tm.shape[1] == 0:
                    tm = tj2
                else:
                    tm = np.hstack((tm, tj2))
                skip = True
                continue
        if rankB > 1:
            res = (False, 0.0, 0)
            if method == 'KNV0':
                res = _KNV0_loop(ker_pole, tm, nb, maxiter, rtol)
            else:
                res = _YT_loop(ker_pole, ker_c, tm, pord, maxiter, rtol)
            cur_rtol = res[1]
            nb_iter = float(res[2])
        tmc = tm.astype(complex)
        idx = 0
        while idx < n - 1:
            if not _is_real(pord[idx]):
                for r in range(n):
                    rel = tmc[r, idx]
                    img = tmc[r, idx + 1]
                    tmc[r, idx] = rel - 1j * img
                    tmc[r, idx + 1] = rel + 1j * img
                idx += 1
            idx += 1
        X = tmc
        dp = np.zeros((n, n), dtype=complex)
        for k in range(n):
            dp[k, k] = pord[k]
        mm = solve(X.T, np.dot(dp, X.T)).T
        rhs = np.dot(u0.T, mm - Aa)
        gc = solve(z.astype(complex), rhs)
        gain = np.empty(gc.shape)
        for r in range(gc.shape[0]):
            for c in range(gc.shape[1]):
                gain[r, c] = gc[r, c].real
    g = -gain
    gm = np.empty(g.shape)
    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            gm[r, c] = g[r, c]
    cl = Aa - np.dot(Ba, gm)
    ev = eig(cl)[0].astype(complex)
    comp = _order_complex_poles(ev)
    return PlacePolesResult(gm, comp, pord, X, cur_rtol, nb_iter)
