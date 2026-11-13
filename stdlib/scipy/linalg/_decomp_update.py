# Ported from scipy 1.18.1 (BSD-3-Clause)

# QR updating (qr_update, qr_insert, qr_delete), qr_multiply, cdf2rdf,
# clarkson_woodruff_transform and the find_best_blas_type helper.
# Deviations: real float64 data only. Rank-p updates / block insertions /
# block deletions are applied as p sequential rank-1 operations (scipy uses
# blocked Householder variants), so Q may differ from scipy's by column signs
# while Q @ R is identical. Economic column insertion needs n + p <= m.

import math
import numpy as np
from numpy.linalg import LinAlgError

EPS = 2.220446049250313e-16
INV_ROOT2 = 0.7071067811865476

def _lartg(f: float, g: float):
    # reference LAPACK dlartg (pre-3.10 convention): returns (c, s, r)
    if g == 0.0:
        return 1.0, 0.0, f
    if f == 0.0:
        return 0.0, 1.0, g
    r = math.sqrt(f * f + g * g)
    c = f / r
    s = g / r
    if abs(f) > abs(g) and c < 0.0:
        return -c, -s, -r
    return c, s, r

def _rot_rows(A: np.ndarray[float, 2], i1: int, i2: int, j0: int, j1: int,
              c: float, s: float):
    for j in range(j0, j1):
        x = A[i1, j]
        y = A[i2, j]
        A[i1, j] = c * x + s * y
        A[i2, j] = c * y - s * x

def _rot_cols(A: np.ndarray[float, 2], c1: int, c2: int, i0: int, i1: int,
              c: float, s: float):
    for i in range(i0, i1):
        x = A[i, c1]
        y = A[i, c2]
        A[i, c1] = c * x + s * y
        A[i, c2] = c * y - s * x

def _swap_rows(A: np.ndarray[float, 2], i1: int, i2: int, ncol: int):
    for j in range(ncol):
        t = A[i1, j]
        A[i1, j] = A[i2, j]
        A[i2, j] = t

def _as2(x, what: str) -> np.ndarray[float, 2]:
    a = np.asarray(x, float)
    if a.ndim != 2:
        raise ValueError(f'expected matrix {what}')
    return np.array(a, order='C')

def _check_qr(Q, R):
    q = _as2(Q, 'Q')
    r = _as2(R, 'R')
    economic = q.shape[1] != q.shape[0]
    if economic:
        if q.shape[1] != r.shape[0] or r.shape[0] != r.shape[1]:
            raise ValueError('Q and R must be either (M,M),(M,N) or (M,N),(N,N)')
    elif q.shape[0] != r.shape[0]:
        raise ValueError('Q and R must be either (M,M),(M,N) or (M,N),(N,N)')
    return q, r, economic

# ----------------------------------------------------------------- reorth

def _reorth(m: int, n: int, q: np.ndarray[float, 2], u: np.ndarray[float, 1],
            s: np.ndarray[float, 1], rcond: float):
    # returns (info, rcond); u (len m) and s (len >= n+1) are updated
    unorm = math.sqrt(float(np.dot(u[:m], u[:m])))
    for i in range(m):
        u[i] = u[i] / unorm
    for j in range(n):
        t = 0.0
        for i in range(m):
            t += q[i, j] * u[i]
        s[j] = t
    snorm = 0.0
    for j in range(n):
        snorm += s[j] * s[j]
    snorm = math.sqrt(snorm)
    sigma_max = math.sqrt(1 + snorm)
    for i in range(m):
        t = 0.0
        for j in range(n):
            t += q[i, j] * s[j]
        u[i] -= t
    wnorm = math.sqrt(float(np.dot(u[:m], u[:m])))
    sigma_min = wnorm / sigma_max
    rc = sigma_min / sigma_max
    if rc < rcond:
        return 2, rc
    if INV_ROOT2 < wnorm:
        for i in range(m):
            u[i] = u[i] / wnorm
        for j in range(n):
            s[j] = s[j] * unorm
        s[n] = unorm * wnorm
        return 0, rc
    s2 = np.zeros(n)
    for j in range(n):
        t = 0.0
        for i in range(m):
            t += q[i, j] * u[i]
        s2[j] = t
    for i in range(m):
        t = 0.0
        for j in range(n):
            t += q[i, j] * s2[j]
        u[i] -= t
    wpnorm = math.sqrt(float(np.dot(u[:m], u[:m])))
    if wpnorm <= wnorm * INV_ROOT2:
        for i in range(m):
            u[i] = 0.0
        for j in range(n):
            s[j] = s[j] * unorm
        s[n] = 0.0
        return 1, rc
    for i in range(m):
        u[i] = u[i] / wpnorm
    for j in range(n):
        s[j] = s[j] * unorm
    s[n] = wpnorm * unorm
    return 0, rc

def _reorthx(m: int, n: int, q: np.ndarray[float, 2], j: int,
             u: np.ndarray[float, 1], s: np.ndarray[float, 1]) -> bool:
    u[j] = 1.0
    for k in range(n):
        s[k] = q[j, k]
    for i in range(m):
        t = 0.0
        for k in range(n):
            t += q[i, k] * s[k]
        u[i] -= t
    wnorm = math.sqrt(float(np.dot(u[:m], u[:m])))
    if INV_ROOT2 < wnorm:
        for i in range(m):
            u[i] = u[i] / wnorm
        s[n] = wnorm
        return True
    s2 = np.zeros(n)
    for k in range(n):
        t = 0.0
        for i in range(m):
            t += q[i, k] * u[i]
        s2[k] = t
    for i in range(m):
        t = 0.0
        for k in range(n):
            t += q[i, k] * s2[k]
        u[i] -= t
    wpnorm = math.sqrt(float(np.dot(u[:m], u[:m])))
    if wpnorm < wnorm * INV_ROOT2:
        for i in range(m):
            u[i] = 0.0
        s[n] = 0.0
        return False
    for i in range(m):
        u[i] = u[i] / wpnorm
    s[n] = wpnorm
    return True

# ---------------------------------------------------------------- helpers

def _hessenberg_qr(m: int, n: int, q: np.ndarray[float, 2], r: np.ndarray[float, 2], k: int):
    limit = min(m - 1, n)
    for j in range(k, limit):
        c, s, rr = _lartg(r[j, j], r[j + 1, j])
        r[j, j] = rr
        r[j + 1, j] = 0.0
        if j + 1 < m:
            _rot_rows(r, j, j + 1, j + 1, n, c, s)
        _rot_cols(q, j, j + 1, 0, m, c, s)

def _qr_rank1_full(q: np.ndarray[float, 2], r: np.ndarray[float, 2],
                   u: np.ndarray[float, 1], v: np.ndarray[float, 1]):
    m = q.shape[0]
    n = r.shape[1]
    qtu = np.zeros(m)
    for j in range(m):
        t = 0.0
        for i in range(m):
            t += q[i, j] * u[i]
        qtu[j] = t
    for j in range(m - 2, -1, -1):
        c, s, rr = _lartg(qtu[j], qtu[j + 1])
        qtu[j] = rr
        qtu[j + 1] = 0.0
        if n - j > 0:
            _rot_rows(r, j, j + 1, j, n, c, s)
        _rot_cols(q, j, j + 1, 0, m, c, s)
    for j in range(n):
        r[0, j] += qtu[0] * v[j]
    _hessenberg_qr(m, n, q, r, 0)

def _qr_rank1_thin(q: np.ndarray[float, 2], r: np.ndarray[float, 2],
                   u: np.ndarray[float, 1], v: np.ndarray[float, 1]):
    m = q.shape[0]
    n = q.shape[1]
    s = np.zeros(2 * n + 1)
    uu = u.copy()
    info, rc = _reorth(m, n, q, uu, s, 0.0)
    c, sn, rr = _lartg(s[n - 1], s[n])
    s[n - 1] = rr
    s[n] = 0.0
    t = r[n - 1, n - 1]
    rlast = -t * sn
    r[n - 1, n - 1] = t * c
    for i in range(m):
        x = q[i, n - 1]
        y = uu[i]
        q[i, n - 1] = c * x + sn * y
        uu[i] = c * y - sn * x
    for j in range(n - 2, -1, -1):
        c, sn, rr = _lartg(s[j], s[j + 1])
        s[j] = rr
        s[j + 1] = 0.0
        _rot_rows(r, j, j + 1, j, n, c, sn)
        _rot_cols(q, j, j + 1, 0, m, c, sn)
    for j in range(n):
        r[0, j] += s[0] * v[j]
    for j in range(n - 1):
        c, sn, rr = _lartg(r[j, j], r[j + 1, j])
        r[j, j] = rr
        r[j + 1, j] = 0.0
        _rot_rows(r, j, j + 1, j + 1, n, c, sn)
        _rot_cols(q, j, j + 1, 0, m, c, sn)
    c, sn, rr = _lartg(r[n - 1, n - 1], rlast)
    r[n - 1, n - 1] = rr
    for i in range(m):
        x = q[i, n - 1]
        y = uu[i]
        q[i, n - 1] = c * x + sn * y
        uu[i] = c * y - sn * x

def qr_update(Q, R, u, v, overwrite_qruv: bool = False, check_finite: bool = True):
    q, r, eco = _check_qr(Q, R)
    m = q.shape[0]
    n = r.shape[1]
    ua = np.asarray(u, float)
    va = np.asarray(v, float)
    if ua.ndim < 1 or ua.ndim > 2:
        raise ValueError('u must be either 1- or 2-D.')
    if va.ndim < 1 or va.ndim > 2:
        raise ValueError('v must be either 1- or 2-D.')
    if ua.shape[0] != m:
        raise ValueError('u.shape[0] must equal Q.shape[0]')
    if va.shape[0] != n:
        raise ValueError('v.shape[0] must equal R.shape[1]')
    if ua.ndim != va.ndim:
        raise ValueError('u and v must have the same number of dimensions')
    p = 1
    if ua.ndim == 2:
        if ua.shape[1] != va.shape[1]:
            raise ValueError('Second dimension of u and v must be the same')
        p = ua.shape[1]
    if p > n or p > m:
        raise ValueError('Update rank larger than np.dot(Q, R).')
    if check_finite:
        for x in ua.flat:
            if not math.isfinite(x):
                raise ValueError('array must not contain infs or NaNs')
        for x in va.flat:
            if not math.isfinite(x):
                raise ValueError('array must not contain infs or NaNs')
    if m == 1:
        rn = r.copy()
        for j in range(n):
            s0 = 0.0
            for t in range(p):
                if ua.ndim == 2:
                    s0 += ua[0, t] * va[j, t]
                else:
                    s0 += ua[0] * va[j]
            rn[0, j] += q[0, 0] * s0
        return q.copy(), rn
    for t in range(p):
        uc = np.empty(m)
        vc = np.empty(n)
        for i in range(m):
            uc[i] = ua[i, t] if ua.ndim == 2 else ua[i]
        for j in range(n):
            vc[j] = va[j, t] if va.ndim == 2 else va[j]
        if eco:
            _qr_rank1_thin(q, r, uc, vc)
        else:
            _qr_rank1_full(q, r, uc, vc)
    return q, r

# ----------------------------------------------------------------- delete

def _block_row_delete(q: np.ndarray[float, 2], r: np.ndarray[float, 2], m: int,
                      n: int, k: int, p: int):
    if k != 0:
        for j in range(k, 0, -1):
            _swap_rows(q, j + p - 1, j - 1, m)
    for i in range(p):
        for j in range(m - 2, i - 1, -1):
            c, s, rr = _lartg(q[i, j], q[i, j + 1])
            q[i, j] = rr
            q[i, j + 1] = 0.0
            if i + 1 < p:
                for t in range(i + 1, p):
                    x = q[t, j]
                    y = q[t, j + 1]
                    q[t, j] = c * x + s * y
                    q[t, j + 1] = c * y - s * x
            if j - i < n:
                _rot_rows(r, j, j + 1, j - i, n, c, s)
            _rot_cols(q, j, j + 1, p, m, c, s)

def _thin_row_delete(q: np.ndarray[float, 2], r: np.ndarray[float, 2], m0: int,
                     n: int, k: int, p_eco: int, p_full: int) -> bool:
    m = m0
    for it in range(p_eco):
        u = np.zeros(m)
        s = np.zeros(3 * n + 2)
        if k != m - 1:
            for j in range(k, m - 1):
                _swap_rows(q, j, j + 1, n)
        if not _reorthx(m, n, q, m - 1, u, s):
            mn = 0.0
            for kk in range(n):
                mn += q[0, kk] * q[0, kk]
            mn = math.sqrt(mn)
            am = 0
            for j in range(1, m):
                rn = 0.0
                for kk in range(n):
                    rn += q[j, kk] * q[j, kk]
                rn = math.sqrt(rn)
                if rn < mn:
                    mn = rn
                    am = j
            for i in range(m):
                u[i] = 0.0
            if not _reorthx(m, n, q, am, u, s):
                return False
            s[n] = 0.0
        s2 = np.zeros(n)
        for j in range(n - 1, -1, -1):
            c, sn, rr = _lartg(s[n], s[j])
            s[n] = rr
            s[j] = 0.0
            for t in range(n - j):
                x = s2[j + t]
                y = r[j, j + t]
                s2[j + t] = c * x + sn * y
                r[j, j + t] = c * y - sn * x
            for i in range(m - 1):
                x = u[i]
                y = q[i, j]
                u[i] = c * x + sn * y
                q[i, j] = c * y - sn * x
        m -= 1
    if p_full > 0:
        _block_row_delete(q, r, m, n, k, p_full)
    return True

def _col_delete_one(q: np.ndarray[float, 2], r: np.ndarray[float, 2], eco: bool,
                    k: int):
    m = q.shape[0]
    n = r.shape[1]
    o = r.shape[0]
    limit = min(o, n)
    for j in range(k, n - 1):
        for i in range(limit):
            r[i, j] = r[i, j + 1]
    _hessenberg_qr(m, n - 1, q, r, k)
    for i in range(r.shape[0]):
        r[i, n - 1] = 0.0

def qr_delete(Q, R, k: int, p: int = 1, which: str = 'row',
              overwrite_qr: bool = False, check_finite: bool = True):
    q, r, eco = _check_qr(Q, R)
    m = q.shape[0]
    n = r.shape[1]
    if which == 'row':
        k1 = k
        if not (-m <= k1 and k1 < m):
            raise ValueError("'k' is out of bounds")
        if k1 < 0:
            k1 += m
        if k1 + p > m or p <= 0:
            raise ValueError("'p' is out of range")
        if eco:
            if n == 1:
                qn = np.empty((m - p, 1))
                t = 0
                nrm = 0.0
                for i in range(m):
                    if i < k1 or i >= k1 + p:
                        qn[t, 0] = q[i, 0]
                        nrm += q[i, 0] * q[i, 0]
                        t += 1
                nrm = math.sqrt(nrm)
                return qn / nrm, r * nrm
            p_eco = p
            p_full = 0
            if m - p < n:
                p_eco = m - n
                p_full = p - p_eco
            ok = _thin_row_delete(q, r, m, n, k1, p_eco, p_full)
            if not ok:
                raise ValueError('Reorthogonalization Failed, unable to perform row deletion.')
            mm = m - p_eco
            qo = np.empty((mm - p_full, q.shape[1] - p_full))
            for i in range(mm - p_full):
                for j in range(q.shape[1] - p_full):
                    qo[i, j] = q[p_full + i, p_full + j]
            ro = np.empty((r.shape[0] - p_full, n))
            for i in range(r.shape[0] - p_full):
                for j in range(n):
                    ro[i, j] = r[p_full + i, j]
            return qo, ro
        else:
            _block_row_delete(q, r, m, n, k1, p)
            qo = np.empty((m - p, m - p))
            for i in range(m - p):
                for j in range(m - p):
                    qo[i, j] = q[p + i, p + j]
            ro = np.empty((m - p, n))
            for i in range(m - p):
                for j in range(n):
                    ro[i, j] = r[p + i, j]
            return qo, ro
    elif which == 'col':
        k1 = k
        if not (-n <= k1 and k1 < n):
            raise ValueError("'k' is out of bounds")
        if k1 < 0:
            k1 += n
        if k1 + p > n or p <= 0:
            raise ValueError("'p' is out of range")
        for t in range(p):
            _col_delete_one(q, r, eco, k1)
            # after one deletion the matrix has n-1-t effective columns
            n2 = n - 1 - t
            if t + 1 < p:
                # shrink logical view by trimming the last column
                r = r[:, :n2].copy()
        nn = n - p
        if eco:
            qo = q[:, :q.shape[1] - p].copy()
            ro = np.empty((r.shape[0] - p, nn))
            for i in range(r.shape[0] - p):
                for j in range(nn):
                    ro[i, j] = r[i, j]
            return qo, ro
        ro = np.empty((r.shape[0], nn))
        for i in range(r.shape[0]):
            for j in range(nn):
                ro[i, j] = r[i, j]
        return q, ro
    else:
        raise ValueError("'which' must be either 'row' or 'col'")

# ----------------------------------------------------------------- insert

def _row_insert_full(q: np.ndarray[float, 2], r: np.ndarray[float, 2], k: int,
                     u: np.ndarray[float, 1]):
    m0 = q.shape[0]
    n = r.shape[1]
    m = m0 + 1
    qn = np.zeros((m, m))
    for i in range(m0):
        for j in range(m0):
            qn[i, j] = q[i, j]
    qn[m0, m0] = 1.0
    rn = np.zeros((m, n))
    for i in range(m0):
        for j in range(n):
            rn[i, j] = r[i, j]
    for j in range(n):
        rn[m0, j] = u[j]
    limit = min(m - 1, n)
    for j in range(limit):
        c, s, rr = _lartg(rn[j, j], rn[m - 1, j])
        rn[j, j] = rr
        rn[m - 1, j] = 0.0
        _rot_rows(rn, j, m - 1, j + 1, n, c, s)
        _rot_cols(qn, j, m - 1, 0, m, c, s)
    for j in range(m - 1, k, -1):
        _swap_rows(qn, j, j - 1, m)
    return qn, rn

def _row_insert_thin(q: np.ndarray[float, 2], r: np.ndarray[float, 2], k: int,
                     u: np.ndarray[float, 1]):
    m0 = q.shape[0]
    n = q.shape[1]
    m = m0 + 1
    qn = np.zeros((m, n + 1))
    for i in range(m0):
        for j in range(n):
            qn[i, j] = q[i, j]
    qn[m0, n] = 1.0
    rn = r.copy()
    uu = u.copy()
    for j in range(n):
        c, s, rr = _lartg(rn[j, j], uu[j])
        rn[j, j] = rr
        uu[j] = 0.0
        for t in range(j + 1, n):
            x = rn[j, t]
            y = uu[t]
            rn[j, t] = c * x + s * y
            uu[t] = c * y - s * x
        _rot_cols(qn, j, n, 0, m, c, s)
    for j in range(m - 1, k, -1):
        _swap_rows(qn, j, j - 1, n)
    qo = np.empty((m, n))
    for i in range(m):
        for j in range(n):
            qo[i, j] = qn[i, j]
    return qo, rn

def _col_insert_full(q: np.ndarray[float, 2], r: np.ndarray[float, 2], k: int,
                     u: np.ndarray[float, 1]):
    m = q.shape[0]
    n = r.shape[1] + 1
    rn = np.zeros((m, n))
    for i in range(m):
        for j in range(k):
            rn[i, j] = r[i, j]
        for j in range(k, n - 1):
            rn[i, j + 1] = r[i, j]
    for j in range(m):
        t = 0.0
        for i in range(m):
            t += q[i, j] * u[i]
        rn[j, k] = t
    qn = q.copy()
    for j in range(m - 2, k - 1, -1):
        c, s, rr = _lartg(rn[j, k], rn[j + 1, k])
        rn[j, k] = rr
        rn[j + 1, k] = 0.0
        if j + 1 < n:
            _rot_rows(rn, j, j + 1, j + 1, n, c, s)
        _rot_cols(qn, j, j + 1, 0, m, c, s)
    return qn, rn

def _col_insert_thin(q: np.ndarray[float, 2], r: np.ndarray[float, 2], k: int,
                     u: np.ndarray[float, 1], rcond: float):
    m = q.shape[0]
    n = q.shape[1]
    qn = np.zeros((m, n + 1))
    for i in range(m):
        for j in range(n):
            qn[i, j] = q[i, j]
    rn = np.zeros((n + 1, n + 1))
    for i in range(n):
        for j in range(k):
            rn[i, j] = r[i, j]
        for j in range(k, n):
            rn[i, j + 1] = r[i, j]
    uu = u.copy()
    s = np.zeros(2 * (n + 1) + 1)
    info, rc = _reorth(m, n, qn, uu, s, rcond)
    if info == 2:
        raise LinAlgError(f'One of the columns of u lies in the span of Q. Found reciprocal condition number of {rc} for Q augmented with u/||u||.')
    for i in range(m):
        qn[i, n] = uu[i]
    for i in range(n + 1):
        rn[i, k] = s[i]
    for i in range(n - 1, k - 1, -1):
        c, sn, rr = _lartg(rn[i, k], rn[i + 1, k])
        rn[i, k] = rr
        rn[i + 1, k] = 0.0
        _rot_rows(rn, i, i + 1, i + 1, n + 1, c, sn)
        _rot_cols(qn, i, i + 1, 0, m, c, sn)
    return qn, rn

def qr_insert(Q, R, u, k: int, which: str = 'row', rcond = None,
              overwrite_qru: bool = False, check_finite: bool = True):
    q, r, eco = _check_qr(Q, R)
    m = q.shape[0]
    n = r.shape[1]
    ua = np.asarray(u, float)
    if check_finite:
        for x in ua.flat:
            if not math.isfinite(x):
                raise ValueError('array must not contain infs or NaNs')
    if which == 'row':
        if not (-m <= k and k <= m):
            raise ValueError("'k' is out of bounds")
        k1 = k + m if k < 0 else k
        p = 1
        if ua.ndim == 2:
            p = ua.shape[0]
            if ua.shape[1] != n:
                raise ValueError(f"'u' should be either (N,) or (p,N) when inserting rows. Found {ua.shape}.")
        elif ua.ndim == 1:
            if ua.shape[0] != n:
                raise ValueError(f"'u' should be either (N,) or (p,N) when inserting rows. Found {ua.shape}.")
        else:
            raise ValueError("'u' must be either 1- or 2-D")
        qc = q
        rc2 = r
        for t in range(p):
            ur = np.empty(n)
            for j in range(n):
                ur[j] = ua[t, j] if ua.ndim == 2 else ua[j]
            if eco:
                qc, rc2 = _row_insert_thin(qc, rc2, k1 + t, ur)
            else:
                qc, rc2 = _row_insert_full(qc, rc2, k1 + t, ur)
        return qc, rc2
    elif which == 'col':
        if not (-n <= k and k <= n):
            raise ValueError("'k' is out of bounds")
        k1 = k + n if k < 0 else k
        p = 1
        if ua.ndim == 2:
            p = ua.shape[1]
            if ua.shape[0] != m:
                raise ValueError(f"'u' should be either (M,) or (M,p) when inserting columns. Found {ua.shape}.")
        elif ua.ndim == 1:
            if ua.shape[0] != m:
                raise ValueError(f"'u' should be either (M,) or (M,p) when inserting columns. Found {ua.shape}.")
        else:
            raise ValueError("'u' must be either 1- or 2-D")
        rcv = EPS
        if rcond is not None:
            if not eco:
                raise ValueError("'rcond' is not used when updating full, (M,M) (M,N) decompositions and must be None.")
            rcv = float(rcond)
        if m == 1:
            rn = np.empty((1, n + p))
            for j in range(k1):
                rn[0, j] = r[0, j]
            for t in range(p):
                rn[0, k1 + t] = q[0, 0] * (ua[0, t] if ua.ndim == 2 else ua[0])
            for j in range(k1, n):
                rn[0, j + p] = r[0, j]
            return q.copy(), rn
        if eco and n + p > m:
            raise NotImplementedError('qr_insert: economic column insertion with N + p > M is not supported')
        qc = q
        rc2 = r
        for t in range(p):
            uc = np.empty(m)
            for i in range(m):
                uc[i] = ua[i, t] if ua.ndim == 2 else ua[i]
            if eco:
                qc, rc2 = _col_insert_thin(qc, rc2, k1 + t, uc, rcv)
            else:
                qc, rc2 = _col_insert_full(qc, rc2, k1 + t, uc)
        return qc, rc2
    else:
        raise ValueError("'which' must be either 'row' or 'col'")

# ----------------------------------------------------------- qr_multiply

def qr_multiply(a, c, mode: str = 'right', pivoting: Literal[bool] = False,
                conjugate: bool = False, overwrite_a: bool = False,
                overwrite_c: bool = False):
    from . import qr
    if mode != 'left' and mode != 'right':
        raise ValueError(f"Mode argument can only be 'left' or 'right' but not '{mode}'")
    ca = np.asarray(c, float)
    aa = np.atleast_2d(np.asarray(a, float))
    M = aa.shape[0]
    N = aa.shape[1]
    kk = min(M, N)
    if pivoting:
        Qe, Re, P = qr(aa, mode='economic', pivoting=True)
        if ca.ndim == 1:
            if mode == 'left':
                if ca.shape[0] != kk:
                    raise ValueError(f'Array shapes are not compatible for Q @ c operation: {aa.shape} vs {ca.shape}')
                return np.dot(Qe, ca), Re, P
            if ca.shape[0] != M:
                raise ValueError(f'Array shapes are not compatible for c @ Q operation: {ca.shape} vs {aa.shape}')
            return np.dot(ca, Qe), Re, P
        if mode == 'left':
            if ca.shape[0] != kk:
                raise ValueError(f'Array shapes are not compatible for Q @ c operation: {aa.shape} vs {ca.shape}')
            return np.dot(Qe, ca), Re, P
        if ca.shape[1] != M:
            raise ValueError(f'Array shapes are not compatible for c @ Q operation: {ca.shape} vs {aa.shape}')
        return np.dot(ca, Qe), Re, P
    Qe2, Re2 = qr(aa, mode='economic')
    if ca.ndim == 1:
        if mode == 'left':
            if ca.shape[0] != kk:
                raise ValueError(f'Array shapes are not compatible for Q @ c operation: {aa.shape} vs {ca.shape}')
            return np.dot(Qe2, ca), Re2
        if ca.shape[0] != M:
            raise ValueError(f'Array shapes are not compatible for c @ Q operation: {ca.shape} vs {aa.shape}')
        return np.dot(ca, Qe2), Re2
    if mode == 'left':
        if ca.shape[0] != kk:
            raise ValueError(f'Array shapes are not compatible for Q @ c operation: {aa.shape} vs {ca.shape}')
        return np.dot(Qe2, ca), Re2
    if ca.shape[1] != M:
        raise ValueError(f'Array shapes are not compatible for c @ Q operation: {ca.shape} vs {aa.shape}')
    return np.dot(ca, Qe2), Re2

# ----------------------------------------------------- cdf2rdf / misc

def cdf2rdf(w, v):
    wa = np.asarray(w)
    va = np.asarray(v)
    if wa.ndim != 1:
        raise ValueError('cdf2rdf: only 1-D eigenvalues and 2-D eigenvectors are supported')
    if va.ndim != 2:
        raise ValueError('expected v to be at least 2D')
    n = wa.shape[0]
    if va.shape[0] != va.shape[1]:
        raise ValueError('expected v to be a square matrix or stacked square matrices: v.shape[-2] = v.shape[-1]')
    if va.shape[1] != n:
        raise ValueError('expected the same number of eigenvalues as eigenvectors')
    idx: List[int] = []
    for i in range(n):
        if complex(wa[i]).imag != 0.0:
            idx.append(i)
    if len(idx) % 2 != 0:
        raise ValueError('expected complex-conjugate pairs of eigenvalues')
    wr = np.zeros((n, n))
    u = np.zeros((n, n), dtype=complex)
    for i in range(n):
        wr[i, i] = complex(wa[i]).real
        u[i, i] = complex(1.0, 0.0)
    t = 0
    while t < len(idx):
        j = idx[t]
        k = idx[t + 1]
        wr[j, k] = complex(wa[j]).imag
        wr[k, j] = complex(wa[k]).imag
        u[j, j] = complex(0.0, 0.5)
        u[j, k] = complex(0.5, 0.0)
        u[k, j] = complex(0.0, -0.5)
        u[k, k] = complex(0.5, 0.0)
        t += 2
    vc = va.astype(complex)
    vr = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            s = complex(0.0, 0.0)
            for l in range(n):
                s += vc[i, l] * u[l, j]
            vr[i, j] = s.real
    return wr, vr

def find_best_blas_type(arrays = (), dtype = None):
    return 'd', float, False

def clarkson_woodruff_transform(input_matrix, sketch_size: int, rng = None, seed = None):
    A = np.atleast_2d(np.asarray(input_matrix, float))
    n = A.shape[0]
    d = A.shape[1]
    g = np.random.default_rng(rng if rng is not None else seed)
    rows = g.integers(0, sketch_size, n)
    signs = g.choice([1, -1], n)
    out = np.zeros((sketch_size, d))
    for i in range(n):
        sg = float(signs[i])
        r0 = int(rows[i])
        for j in range(d):
            out[r0, j] += sg * A[i, j]
    return out
