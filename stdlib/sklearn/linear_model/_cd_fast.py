# Ported from scikit-learn 1.9.1 (BSD-3-Clause)
#
# Coordinate descent kernels of sklearn/linear_model/_cd_fast.pyx (dense
# float64): enet_coordinate_descent, enet_coordinate_descent_gram,
# enet_coordinate_descent_multi_task, with gap-safe screening and the
# our_rand_r random coordinate selection. BLAS dot/axpy are plain loops
# (summation order may differ from the vendor BLAS in the last ulp).

import math
import numpy as np
from ._sag import _our_rand_r

def _rng_seed(rng) -> u32:
    return u32(rng.randint(0, 2147483647))

def _dual_gap_A(alpha: float, beta: float, w_l1: float, w_l2sq: float, R_norm2: float,
                Ry: float, dual_norm_XtA: float) -> float:
    primal = 0.5 * (R_norm2 + beta * w_l2sq) + alpha * w_l1
    scale = alpha / dual_norm_XtA if dual_norm_XtA > alpha else 1.0
    dual = -0.5 * (scale ** 2) * (R_norm2 + beta * w_l2sq) + scale * Ry
    return primal - dual

def _fsign(f: float) -> float:
    if f == 0.0:
        return 0.0
    elif f > 0.0:
        return 1.0
    else:
        return -1.0

def _dot(n: int, a: Ptr[float], b: Ptr[float]) -> float:
    s = 0.0
    for i in range(n):
        s += a[i] * b[i]
    return s

def _axpy(n: int, alpha: float, x: Ptr[float], y: Ptr[float]):
    for i in range(n):
        y[i] += alpha * x[i]

def _gap_enet(n: int, p: int, w: Ptr[float], alpha: float, beta: float, X: Ptr[float],
              y: Ptr[float], R: Ptr[float], XtA: Ptr[float], positive: bool):
    # X column major (n x p)
    w_l2sq = 0.0
    if beta > 0.0:
        w_l2sq = _dot(p, w, w)
    R_norm2 = _dot(n, R, R)
    Ry = 0.0
    if not (alpha == 0.0 and beta == 0.0):
        Ry = _dot(n, R, y)
    if alpha == 0.0:
        for j in range(p):
            XtA[j] = _dot(n, X + j * n, R)
        dn = _dot(p, XtA, XtA)
        if beta == 0.0:
            return dn, dn
        gap = R_norm2 + 0.5 * beta * w_l2sq - Ry
        gap += 1.0 / (2.0 * beta) * dn
        return gap, dn
    for j in range(p):
        XtA[j] = _dot(n, X + j * n, R) - beta * w[j]
    dn = 0.0
    if positive:
        dn = XtA[0]
        for j in range(1, p):
            if XtA[j] > dn:
                dn = XtA[j]
    else:
        dn = abs(XtA[0])
        for j in range(1, p):
            if abs(XtA[j]) > dn:
                dn = abs(XtA[j])
    l1 = 0.0
    for j in range(p):
        l1 += abs(w[j])
    return _dual_gap_A(alpha, beta, l1, w_l2sq, R_norm2, Ry, dn), dn

def enet_coordinate_descent(w: np.ndarray[float, 1], alpha: float, beta: float, X, y,
                            max_iter: int, tol: float, rng, random: bool = False,
                            positive: bool = False, do_screening: bool = True):
    """Returns (w, gap, tol, n_iter); w is updated in place."""
    Xf = np.asfortranarray(np.asarray(X, dtype=float))
    ya = np.ascontiguousarray(np.asarray(y, dtype=float))
    n, p = Xf.shape
    Xp = Xf.data
    yp = ya.data
    wp = w.data
    norm2 = Ptr[float](p)
    for j in range(p):
        norm2[j] = _dot(n, Xp + j * n, Xp + j * n)
    R = Ptr[float](n)
    XtA = Ptr[float](p)
    seed = _rng_seed(rng)
    if alpha == 0.0:
        do_screening = False
    active = Ptr[int](p)
    excluded = Ptr[bool](p)
    n_active = p
    for i in range(n):
        R[i] = yp[i]
    for j in range(p):
        if wp[j] != 0.0:
            _axpy(n, -wp[j], Xp + j * n, R)
    d_w_tol = tol
    tol *= _dot(n, yp, yp)
    gap, dn = _gap_enet(n, p, wp, alpha, beta, Xp, yp, R, XtA, positive)
    if gap <= tol:
        return w, gap, tol, 0
    if do_screening:
        n_active = 0
        for j in range(p):
            if norm2[j] == 0.0:
                wp[j] = 0.0
                excluded[j] = True
                continue
            Xj_theta = XtA[j] / max(alpha, dn)
            d_j = (1.0 - abs(Xj_theta)) / math.sqrt(norm2[j] + beta)
            if d_j <= math.sqrt(2.0 * gap) / alpha:
                active[n_active] = j
                excluded[j] = False
                n_active += 1
            else:
                if wp[j] != 0.0:
                    _axpy(n, wp[j], Xp + j * n, R)
                    wp[j] = 0.0
                excluded[j] = True
    return _enet_loop(w, alpha, beta, Xp, yp, n, p, norm2, R, XtA, seed, active, excluded,
                      n_active, max_iter, tol, d_w_tol, random, positive, do_screening, gap, dn)

def _enet_loop(w: np.ndarray[float, 1], alpha: float, beta: float, Xp: Ptr[float],
               yp: Ptr[float], n: int, p: int, norm2: Ptr[float], R: Ptr[float],
               XtA: Ptr[float], seed: u32, active: Ptr[int], excluded: Ptr[bool],
               n_active: int, max_iter: int, tol: float, d_w_tol: float, random: bool, positive: bool,
               do_screening: bool, gap: float, dn: float):
    wp = w.data
    n_iter = 0
    for it in range(max_iter):
        n_iter = it
        w_max = 0.0
        d_w_max = 0.0
        for f_iter in range(n_active):
            j = f_iter
            if random:
                j = int(u32(_our_rand_r(__ptr__(seed))) % u32(n_active))
            if do_screening:
                j = active[j]
            if norm2[j] == 0.0:
                continue
            w_j = wp[j]
            tmp = _dot(n, Xp + j * n, R) + w_j * norm2[j]
            if positive and tmp < 0.0:
                wp[j] = 0.0
            else:
                wp[j] = _fsign(tmp) * max(abs(tmp) - alpha, 0.0) / (norm2[j] + beta)
            if wp[j] != w_j:
                _axpy(n, w_j - wp[j], Xp + j * n, R)
            d_w_j = abs(wp[j] - w_j)
            d_w_max = max(d_w_max, d_w_j)
            w_max = max(w_max, abs(wp[j]))
        if w_max == 0.0 or d_w_max / w_max <= d_w_tol or it == max_iter - 1:
            gap, dn = _gap_enet(n, p, wp, alpha, beta, Xp, yp, R, XtA, positive)
            if gap <= tol:
                break
            if do_screening:
                n_active = 0
                for j in range(p):
                    if excluded[j]:
                        continue
                    Xj_theta = XtA[j] / max(alpha, dn)
                    d_j = (1.0 - abs(Xj_theta)) / math.sqrt(norm2[j] + beta)
                    if d_j <= math.sqrt(2.0 * gap) / alpha:
                        active[n_active] = j
                        excluded[j] = False
                        n_active += 1
                    else:
                        if wp[j] != 0.0:
                            _axpy(n, wp[j], Xp + j * n, R)
                            wp[j] = 0.0
                        excluded[j] = True
    return w, gap, tol, n_iter + 1

def _gap_enet_gram(p: int, w: Ptr[float], alpha: float, beta: float, Qw: Ptr[float],
                   q: Ptr[float], y_norm2: float, XtA: Ptr[float], positive: bool):
    w_l2sq = 0.0
    if beta > 0.0:
        w_l2sq = _dot(p, w, w)
    q_dot_w = _dot(p, w, q)
    wQw = _dot(p, w, Qw)
    R_norm2 = y_norm2 + wQw - 2.0 * q_dot_w
    Ry = 0.0
    if not (alpha == 0.0 and beta == 0.0):
        Ry = y_norm2 - q_dot_w
    if alpha == 0.0:
        for j in range(p):
            XtA[j] = q[j] - Qw[j]
        dn = _dot(p, XtA, XtA)
        if beta == 0.0:
            return dn, dn
        gap = R_norm2 + 0.5 * beta * w_l2sq - Ry
        gap += 1.0 / (2.0 * beta) * dn
        return gap, dn
    for j in range(p):
        XtA[j] = q[j] - Qw[j] - beta * w[j]
    dn = 0.0
    if positive:
        dn = XtA[0]
        for j in range(1, p):
            if XtA[j] > dn:
                dn = XtA[j]
    else:
        dn = abs(XtA[0])
        for j in range(1, p):
            if abs(XtA[j]) > dn:
                dn = abs(XtA[j])
    l1 = 0.0
    for j in range(p):
        l1 += abs(w[j])
    return _dual_gap_A(alpha, beta, l1, w_l2sq, R_norm2, Ry, dn), dn

def enet_coordinate_descent_gram(w: np.ndarray[float, 1], alpha: float, beta: float, Q, q, y,
                                 max_iter: int, tol: float, rng, random: bool = False,
                                 positive: bool = False, do_screening: bool = True):
    Qa = np.ascontiguousarray(np.asarray(Q, dtype=float))
    qa = np.ascontiguousarray(np.asarray(q, dtype=float))
    ya = np.asarray(y, dtype=float)
    p = Qa.shape[0]
    Qp = Qa.data
    qp = qa.data
    wp = w.data
    Qwa = np.ascontiguousarray(Qa @ w)
    Qw = Qwa.data
    XtA = Ptr[float](p)
    for j in range(p):
        XtA[j] = 0.0
    y_norm2 = float(np.dot(ya, ya))
    d_w_tol = tol
    seed = _rng_seed(rng)
    if alpha == 0.0:
        do_screening = False
    active = Ptr[int](p)
    excluded = Ptr[bool](p)
    n_active = p
    tol *= y_norm2
    gap, dn = _gap_enet_gram(p, wp, alpha, beta, Qw, qp, y_norm2, XtA, positive)
    if 0.0 <= gap <= tol:
        return w, gap, tol, 0
    if do_screening:
        radius = math.sqrt(2.0 * abs(gap)) / alpha
        n_active = 0
        for j in range(p):
            if Qp[j * p + j] == 0.0:
                wp[j] = 0.0
                excluded[j] = True
                continue
            Xj_theta = XtA[j] / max(alpha, dn)
            d_j = (1.0 - abs(Xj_theta)) / math.sqrt(Qp[j * p + j] + beta)
            if d_j <= radius:
                active[n_active] = j
                excluded[j] = False
                n_active += 1
            else:
                if wp[j] != 0.0:
                    _axpy(p, -wp[j], Qp + j * p, Qw)
                    wp[j] = 0.0
                excluded[j] = True
    n_iter = 0
    for it in range(max_iter):
        n_iter = it
        w_max = 0.0
        d_w_max = 0.0
        for f_iter in range(n_active):
            j = f_iter
            if random:
                j = int(u32(_our_rand_r(__ptr__(seed))) % u32(n_active))
            if do_screening:
                j = active[j]
            Qjj = Qp[j * p + j]
            if Qjj == 0.0:
                continue
            w_j = wp[j]
            tmp = qp[j] - Qw[j] + w_j * Qjj
            if positive and tmp < 0.0:
                wp[j] = 0.0
            else:
                wp[j] = _fsign(tmp) * max(abs(tmp) - alpha, 0.0) / (Qjj + beta)
            if wp[j] != w_j:
                _axpy(p, wp[j] - w_j, Qp + j * p, Qw)
            d_w_j = abs(wp[j] - w_j)
            if d_w_j > d_w_max:
                d_w_max = d_w_j
            if abs(wp[j]) > w_max:
                w_max = abs(wp[j])
        if w_max == 0.0 or d_w_max / w_max <= d_w_tol or it == max_iter - 1:
            gap, dn = _gap_enet_gram(p, wp, alpha, beta, Qw, qp, y_norm2, XtA, positive)
            if gap <= tol:
                break
            if do_screening:
                radius = math.sqrt(2.0 * abs(gap)) / alpha
                n_active = 0
                for j in range(p):
                    if excluded[j]:
                        continue
                    Xj_theta = XtA[j] / max(alpha, dn)
                    d_j = (1.0 - abs(Xj_theta)) / math.sqrt(Qp[j * p + j] + beta)
                    if d_j <= radius:
                        active[n_active] = j
                        excluded[j] = False
                        n_active += 1
                    else:
                        if wp[j] != 0.0:
                            _axpy(p, -wp[j], Qp + j * p, Qw)
                            wp[j] = 0.0
                        excluded[j] = True
    return w, gap, tol, n_iter + 1

def _nrm2(n: int, a: Ptr[float], stride: int) -> float:
    s = 0.0
    for i in range(n):
        s += a[i * stride] * a[i * stride]
    return math.sqrt(s)

def _gap_mt(n: int, p: int, k: int, W: Ptr[float], alpha: float, beta: float, X: Ptr[float],
            Y: Ptr[float], R: Ptr[float], XtA: Ptr[float], XtA_rn: Ptr[float]):
    # W (k, p) row-major: W[t*p + j]; X col-major n x p; Y, R col-major n x k
    # XtA (p, k) row-major
    w_l2sq = 0.0
    if beta > 0.0:
        w_l2sq = _dot(p * k, W, W)
    R_norm2 = _dot(n * k, R, R)
    Ry = 0.0
    if not (alpha == 0.0 and beta == 0.0):
        Ry = _dot(n * k, R, Y)
    if alpha == 0.0:
        for j in range(p):
            for t in range(k):
                XtA[j * k + t] = _dot(n, X + j * n, R + t * n)
        dn = _dot(p * k, XtA, XtA)
        if beta == 0.0:
            return dn, dn
        gap = R_norm2 + 0.5 * beta * w_l2sq - Ry
        gap += 1.0 / (2.0 * beta) * dn
        return gap, dn
    for j in range(p):
        for t in range(k):
            XtA[j * k + t] = _dot(n, X + j * n, R + t * n) - beta * W[t * p + j]
    dn = 0.0
    for j in range(p):
        XtA_rn[j] = _nrm2(k, XtA + j * k, 1)
        if XtA_rn[j] > dn:
            dn = XtA_rn[j]
    l21 = 0.0
    for j in range(p):
        l21 += _nrm2(k, W + j, p)
    return _dual_gap_A(alpha, beta, l21, w_l2sq, R_norm2, Ry, dn), dn

def enet_coordinate_descent_multi_task(W: np.ndarray[float, 2], alpha: float, beta: float, X, Y,
                                       max_iter: int, tol: float, rng, random: bool = False,
                                       do_screening: bool = True):
    """W (n_tasks, n_features) updated in place. Returns (W, gap, tol, n_iter)."""
    Xf = np.asfortranarray(np.asarray(X, dtype=float))
    Yf = np.asfortranarray(np.asarray(Y, dtype=float))
    n, p = Xf.shape
    k = Yf.shape[1]
    Xp = Xf.data
    Yp = Yf.data
    Wc = np.ascontiguousarray(W)
    Wp = Wc.data
    norm2 = Ptr[float](p)
    for j in range(p):
        norm2[j] = _dot(n, Xp + j * n, Xp + j * n)
    R = Ptr[float](n * k)
    for i in range(n * k):
        R[i] = Yp[i]
    XtA = Ptr[float](p * k)
    XtA_rn = Ptr[float](p)
    tmp = Ptr[float](k)
    w_j = Ptr[float](k)
    d_w_tol = tol
    seed = _rng_seed(rng)
    if alpha == 0.0:
        do_screening = False
    active = Ptr[int](p)
    excluded = Ptr[bool](p)
    n_active = p
    for j in range(p):
        for t in range(k):
            if Wp[t * p + j] != 0.0:
                _axpy(n, -Wp[t * p + j], Xp + j * n, R + t * n)
    tol *= _dot(n * k, Yp, Yp)
    gap, dn = _gap_mt(n, p, k, Wp, alpha, beta, Xp, Yp, R, XtA, XtA_rn)
    if gap <= tol:
        W[:, :] = Wc
        return W, gap, tol, 0
    if do_screening:
        n_active = 0
        for j in range(p):
            if norm2[j] == 0.0:
                for t in range(k):
                    Wp[t * p + j] = 0.0
                excluded[j] = True
                continue
            Xj_theta = XtA_rn[j] / max(alpha, dn)
            d_j = (1.0 - Xj_theta) / math.sqrt(norm2[j] + beta)
            if d_j <= math.sqrt(2.0 * gap) / alpha:
                active[n_active] = j
                excluded[j] = False
                n_active += 1
            else:
                for t in range(k):
                    if Wp[t * p + j] != 0.0:
                        _axpy(n, Wp[t * p + j], Xp + j * n, R + t * n)
                        Wp[t * p + j] = 0.0
                excluded[j] = True
    n_iter = 0
    for it in range(max_iter):
        n_iter = it
        w_max = 0.0
        d_w_max = 0.0
        for f_iter in range(n_active):
            j = f_iter
            if random:
                j = int(u32(_our_rand_r(__ptr__(seed))) % u32(n_active))
            if do_screening:
                j = active[j]
            if norm2[j] == 0.0:
                continue
            for t in range(k):
                w_j[t] = Wp[t * p + j]
            for t in range(k):
                tmp[t] = _dot(n, Xp + j * n, R + t * n) + w_j[t] * norm2[j]
            nn = _nrm2(k, tmp, 1)
            sc = max(1.0 - alpha / nn, 0.0) / (norm2[j] + beta)
            for t in range(k):
                Wp[t * p + j] = tmp[t] * sc
            for t in range(k):
                if Wp[t * p + j] != w_j[t]:
                    _axpy(n, w_j[t] - Wp[t * p + j], Xp + j * n, R + t * n)
            d_w_j = abs(Wp[j] - w_j[0])
            wam = abs(Wp[j])
            for t in range(1, k):
                d_w_j = max(d_w_j, abs(Wp[t * p + j] - w_j[t]))
                wam = max(wam, abs(Wp[t * p + j]))
            if d_w_j > d_w_max:
                d_w_max = d_w_j
            if wam > w_max:
                w_max = wam
        if w_max == 0.0 or d_w_max / w_max <= d_w_tol or it == max_iter - 1:
            gap, dn = _gap_mt(n, p, k, Wp, alpha, beta, Xp, Yp, R, XtA, XtA_rn)
            if gap <= tol:
                break
            if do_screening:
                n_active = 0
                for j in range(p):
                    if excluded[j]:
                        continue
                    Xj_theta = XtA_rn[j] / max(alpha, dn)
                    d_j = (1.0 - Xj_theta) / math.sqrt(norm2[j] + beta)
                    if d_j <= math.sqrt(2.0 * gap) / alpha:
                        active[n_active] = j
                        excluded[j] = False
                        n_active += 1
                    else:
                        for t in range(k):
                            if Wp[t * p + j] != 0.0:
                                _axpy(n, Wp[t * p + j], Xp + j * n, R + t * n)
                                Wp[t * p + j] = 0.0
                        excluded[j] = True
    W[:, :] = Wc
    return W, gap, tol, n_iter + 1
