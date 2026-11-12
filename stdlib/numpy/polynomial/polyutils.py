# Ported from numpy 2.5.4 (BSD-3-Clause)
#
# Shared machinery for numpy.polynomial. Every family is identified by a
# static integer K and described by its three-term recurrence
#     P_{n+1}(x) = (A_n x + B_n) P_n(x) - C_n P_{n-1}(x),  P_0 = 1
# K: 0 power series, 1 Chebyshev, 2 Legendre, 3 Hermite, 4 HermiteE,
#    5 Laguerre. Coefficients are 1-D float64 arrays (low degree first).

import numpy as np

def _rec(K: Literal[int], n: int):
    nf = float(n)
    if K == 0:
        return 1.0, 0.0, 0.0
    elif K == 1:
        return (1.0 if n == 0 else 2.0), 0.0, 1.0
    elif K == 2:
        return (2.0 * nf + 1.0) / (nf + 1.0), 0.0, nf / (nf + 1.0)
    elif K == 3:
        return 2.0, 0.0, 2.0 * nf
    elif K == 4:
        return 1.0, 0.0, nf
    else:
        return -1.0 / (nf + 1.0), (2.0 * nf + 1.0) / (nf + 1.0), nf / (nf + 1.0)

def as_series(c):
    """Return `c` as a fresh 1-D float64 coefficient array."""
    a = np.atleast_1d(np.array(c, dtype=float))
    if a.ndim != 1:
        raise ValueError("Coefficient array is not 1-d")
    if a.size == 0:
        raise ValueError("Coefficient array is empty")
    return a.copy()

def trimseq(seq):
    n = len(seq)
    if n == 0 or seq[n - 1] != 0.0:
        return seq
    i = n - 1
    while i > 0 and seq[i] == 0.0:
        i -= 1
    return seq[:i + 1].copy()

def trimcoef(c, tol: float = 0.0):
    if tol < 0:
        raise ValueError("tol must be non-negative")
    c = as_series(c)
    last = -1
    for i in range(len(c)):
        if abs(c[i]) > tol:
            last = i
    if last < 0:
        return c[:1] * 0.0
    return c[:last + 1].copy()

def getdomain(x):
    x = np.asarray(x, dtype=float)
    return np.array([x.min(), x.max()])

def mapparms(old, new):
    o = np.asarray(old, dtype=float)
    n = np.asarray(new, dtype=float)
    oldlen = o[1] - o[0]
    newlen = n[1] - n[0]
    off = (o[1] * n[0] - o[0] * n[1]) / oldlen
    scl = newlen / oldlen
    return off, scl

def mapdomain(x, old, new):
    off, scl = mapparms(old, new)
    if isinstance(x, list) or isinstance(x, tuple):
        return off + scl * np.asarray(x, dtype=float)
    return off + scl * x

def _pad(c, n: int):
    r = np.zeros(n)
    m = min(n, len(c))
    r[:m] = c[:m]
    return r

def _add(c1, c2):
    a = as_series(c1)
    b = as_series(c2)
    n = max(len(a), len(b))
    return trimseq(_pad(a, n) + _pad(b, n))

def _sub(c1, c2):
    a = as_series(c1)
    b = as_series(c2)
    n = max(len(a), len(b))
    return trimseq(_pad(a, n) - _pad(b, n))

def _mulx_into(K: Literal[int], c, out):
    # out += x * c (out must have room for len(c)+1 entries)
    for n in range(len(c)):
        v = c[n]
        if v == 0.0:
            continue
        A, B, C = _rec(K, n)
        out[n + 1] += v / A
        if B != 0.0:
            out[n] -= v * B / A
        if n > 0 and C != 0.0:
            out[n - 1] += v * C / A

def _mulx(K: Literal[int], c):
    c = as_series(c)
    if len(c) == 1 and c[0] == 0.0:
        return c
    out = np.zeros(len(c) + 1)
    _mulx_into(K, c, out)
    return out

def _clenshaw_series(K: Literal[int], M: Literal[int], c, s, L: int):
    # sum_k c[k] P^K_k(x) * s(x), with s and the result in family M basis
    b1 = np.zeros(L)
    b2 = np.zeros(L)
    sp = _pad(s, L)
    for k in range(len(c) - 1, -1, -1):
        A, B, _ = _rec(K, k)
        _, _, C1 = _rec(K, k + 1)
        b = sp * c[k] + B * b1 - C1 * b2
        if A != 0.0:
            t = np.zeros(L + 1)
            _mulx_into(M, b1, t)
            b += A * t[:L]
        b2 = b1
        b1 = b
    return b1

def _mul(K: Literal[int], c1, c2):
    a = as_series(c1)
    b = as_series(c2)
    if K == 0:
        return trimseq(np.convolve(a, b))
    if len(a) > len(b):
        a, b = b, a
    return trimseq(_clenshaw_series(K, K, a, b, len(a) + len(b) - 1))

def _div(K: Literal[int], c1, c2):
    a = as_series(c1)
    b = trimseq(as_series(c2))
    if b[-1] == 0.0:
        raise ZeroDivisionError("division by zero polynomial")
    la = len(a)
    lb = len(b)
    if la < lb:
        return a[:1] * 0.0, a
    elif lb == 1:
        return a / b[-1], a[:1] * 0.0
    quo = np.empty(la - lb + 1)
    rem = a
    for i in range(la - lb, -1, -1):
        e = np.zeros(i + 1)
        e[i] = 1.0
        p = _mul(K, e, b)
        q = rem[-1] / p[-1]
        rem = rem[:-1] - q * p[:-1]
        quo[i] = q
    return quo, trimseq(rem)

def _pow(K: Literal[int], c, pw: int, maxpower: int = 16):
    c = as_series(c)
    if pw < 0:
        raise ValueError("Power must be a non-negative integer.")
    if pw > maxpower:
        raise ValueError("Power is too large")
    if pw == 0:
        return np.array([1.0])
    prd = c
    for i in range(2, pw + 1):
        prd = _mul(K, prd, c)
    return prd

def _val(K: Literal[int], x, c):
    c = as_series(c)
    if isinstance(x, list) or isinstance(x, tuple):
        return _val(K, np.asarray(x, dtype=float), c)
    else:
        z = x * 0.0
        b1 = z
        b2 = z
        for k in range(len(c) - 1, -1, -1):
            A, B, _ = _rec(K, k)
            _, _, C1 = _rec(K, k + 1)
            b = c[k] + (A * x + B) * b1 - C1 * b2
            b2 = b1
            b1 = b
        return b1

def _vander(K: Literal[int], x, deg: int):
    if deg < 0:
        raise ValueError("deg must be non-negative")
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = x.size
    xf = x.reshape(-1)
    v = np.empty((n, deg + 1))
    for i in range(n):
        p0 = 1.0
        p1 = 0.0
        v[i, 0] = 1.0
        for k in range(deg):
            A, B, C = _rec(K, k)
            p = (A * xf[i] + B) * p0 - C * p1
            p1 = p0
            p0 = p
            v[i, k + 1] = p
    return v

def _line(K: Literal[int], off: float, scl: float):
    if scl == 0.0:
        return np.array([off])
    r = _mulx(K, np.array([1.0])) * scl
    r[0] += off
    return r

def _fromroots(K: Literal[int], roots):
    r = np.sort(np.atleast_1d(np.asarray(roots, dtype=float)))
    if r.size == 0:
        return np.ones(1)
    p = [_line(K, -r[i], 1.0) for i in range(r.size)]
    n = len(p)
    while n > 1:
        m = n // 2
        tmp = [_mul(K, p[i], p[i + m]) for i in range(m)]
        if n % 2 == 1:
            tmp[0] = _mul(K, tmp[0], p[-1])
        p = tmp
        n = m
    return p[0]

def _der(K: Literal[int], c, m: int = 1, scl: float = 1.0):
    c = as_series(c)
    if m < 0:
        raise ValueError("The order of derivation must be non-negative")
    if m == 0:
        return c
    n = len(c)
    if m >= n:
        return c[:1] * 0.0
    for _ in range(m):
        n -= 1
        c *= scl
        der = np.zeros(n)
        if K == 0 or K == 3 or K == 4:
            f = 2.0 if K == 3 else 1.0
            for j in range(n, 0, -1):
                der[j - 1] = f * j * c[j]
        elif K == 1:
            for j in range(n, 2, -1):
                der[j - 1] = (2 * j) * c[j]
                c[j - 2] += (j * c[j]) / (j - 2)
            if n > 1:
                der[1] = 4 * c[2]
            der[0] = c[1]
        elif K == 2:
            for j in range(n, 2, -1):
                der[j - 1] = (2 * j - 1) * c[j]
                c[j - 2] += c[j]
            if n > 1:
                der[1] = 3 * c[2]
            der[0] = c[1]
        else:
            for j in range(n, 1, -1):
                der[j - 1] = -c[j]
                c[j - 1] += c[j]
            der[0] = -c[1]
        c = der
    return c

def _int(K: Literal[int], c, m: int = 1, k = None, lbnd: float = 0.0, scl: float = 1.0):
    c = as_series(c)
    if m < 0:
        raise ValueError("The order of integration must be non-negative")
    ks = [0.0 for _ in range(m)]
    if k is not None:
        if isinstance(k, list) or isinstance(k, tuple) or isinstance(k, np.ndarray):
            if len(k) > m:
                raise ValueError("Too many integration constants")
            for i in range(len(k)):
                ks[i] = float(k[i])
        else:
            if m < 1:
                raise ValueError("Too many integration constants")
            ks[0] = float(k)
    for i in range(m):
        n = len(c)
        c *= scl
        if n == 1 and c[0] == 0.0:
            c[0] += ks[i]
            continue
        tmp = np.zeros(n + 1)
        if K == 0 or K == 4:
            for j in range(n):
                tmp[j + 1] = c[j] / (j + 1)
        elif K == 3:
            for j in range(n):
                tmp[j + 1] = c[j] / (2 * (j + 1))
        elif K == 1:
            tmp[1] = c[0]
            if n > 1:
                tmp[2] = c[1] / 4
            for j in range(2, n):
                tmp[j + 1] = c[j] / (2 * (j + 1))
                tmp[j - 1] -= c[j] / (2 * (j - 1))
        elif K == 2:
            tmp[1] = c[0]
            if n > 1:
                tmp[2] = c[1] / 3
            for j in range(2, n):
                t = c[j] / (2 * j + 1)
                tmp[j + 1] = t
                tmp[j - 1] -= t
        else:
            tmp[0] = c[0]
            tmp[1] = -c[0]
            for j in range(1, n):
                tmp[j] += c[j]
                tmp[j + 1] = -c[j]
        tmp[0] += ks[i] - _val(K, lbnd, tmp)
        c = tmp
    return c

def _fit(K: Literal[int], x, y, deg, rcond = None, full: Literal[bool] = False, w = None):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 1:
        raise TypeError("expected 1D vector for x")
    if x.size == 0:
        raise TypeError("expected non-empty vector for x")
    if y.ndim != 1:
        raise TypeError("expected 1D array for y")
    if len(x) != len(y):
        raise TypeError("expected x and y to have same length")
    if isinstance(deg, int):
        if deg < 0:
            raise ValueError("expected deg >= 0")
        return _fit_cols(K, x, y, deg, [i for i in range(deg + 1)], rcond, full, w)
    else:
        ds = sorted([int(d) for d in deg])
        if len(ds) == 0:
            raise TypeError("deg must be an int or non-empty 1-D array of int")
        if ds[0] < 0:
            raise ValueError("expected deg >= 0")
        return _fit_cols(K, x, y, ds[-1], ds, rcond, full, w)

def _fit_cols(K: Literal[int], x, y, lmax: int, cols: List[int], rcond, full: Literal[bool], w):
    vfull = _vander(K, x, lmax)
    van = np.empty((len(x), len(cols)))
    for j in range(len(cols)):
        van[:, j] = vfull[:, cols[j]]
    rhs = y.copy()
    if w is not None:
        ww = np.asarray(w, dtype=float)
        if ww.ndim != 1:
            raise TypeError("expected 1D vector for w")
        if len(x) != len(ww):
            raise TypeError("expected x and w to have same length")
        for i in range(len(x)):
            van[i, :] *= ww[i]
        rhs = rhs * ww
    rc = len(x) * np.finfo(float).eps
    if rcond is not None:
        rc = float(rcond)
    scl = np.sqrt((van * van).sum(0))
    for j in range(len(cols)):
        if scl[j] == 0.0:
            scl[j] = 1.0
    sol, resids, rank, sv = np.linalg.lstsq(van / scl, rhs, rc)
    sol = sol / scl
    out = np.zeros(lmax + 1)
    for j in range(len(cols)):
        out[cols[j]] = sol[j]
    if full:
        return out, (resids, rank, sv, rc)
    else:
        return out

def _companion(K: Literal[int], c):
    c = trimseq(as_series(c))
    if len(c) < 2:
        raise ValueError("Series must have maximum degree of at least 1.")
    n = len(c) - 1
    if n == 1:
        if K == 3:
            return np.array([[-0.5 * c[0] / c[1]]])
        if K == 5:
            return np.array([[1.0 + c[0] / c[1]]])
        return np.array([[-c[0] / c[1]]])
    mat = np.zeros((n, n))
    if K == 0:
        for i in range(n - 1):
            mat[i + 1, i] = 1.0
        for i in range(n):
            mat[i, n - 1] -= c[i] / c[n]
    elif K == 5:
        for i in range(n):
            mat[i, i] = 2.0 * i + 1.0
            if i < n - 1:
                mat[i, i + 1] = -(i + 1.0)
                mat[i + 1, i] = -(i + 1.0)
        for i in range(n):
            mat[i, n - 1] += (c[i] / c[n]) * n
    else:
        # scaled symmetric (orthonormal) form
        scl = np.ones(n)
        top = np.zeros(n - 1)
        if K == 1:
            for i in range(1, n):
                scl[i] = np.sqrt(0.5)
            top[:] = 0.5
            top[0] = np.sqrt(0.5)
        elif K == 2:
            for i in range(n):
                scl[i] = 1.0 / np.sqrt(2.0 * i + 1.0)
            for i in range(n - 1):
                top[i] = (i + 1.0) * scl[i] * scl[i + 1]
        else:
            f = 2.0 if K == 3 else 1.0
            # scl[i] = prod_{j=1..i} 1/sqrt(f*j)
            for i in range(1, n):
                scl[i] = scl[i - 1] / np.sqrt(f * i)
            for i in range(n - 1):
                top[i] = np.sqrt((i + 1.0) / f)
        for i in range(n - 1):
            mat[i, i + 1] = top[i]
            mat[i + 1, i] = top[i]
        fac = 0.5 if K == 1 else ((n / (2.0 * n - 1.0)) if K == 2 else (0.5 if K == 3 else 1.0))
        for i in range(n):
            if K == 3 or K == 4:
                mat[i, n - 1] -= scl[n - 1] / scl[i] * c[i] / c[n] * fac
            else:
                mat[i, n - 1] -= (c[i] / c[n]) * (scl[i] / scl[n - 1]) * fac
    return mat

def _roots(K: Literal[int], c):
    c = trimseq(as_series(c))
    if len(c) < 2:
        return np.empty(0, dtype=complex)
    if len(c) == 2:
        if K == 3:
            return np.array([complex(-0.5 * c[0] / c[1])])
        if K == 5:
            return np.array([complex(1.0 + c[0] / c[1])])
        return np.array([complex(-c[0] / c[1])])
    m = _companion(K, c)
    n = m.shape[0]
    rot = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            rot[i, j] = m[n - 1 - i, n - 1 - j]
    r = np.linalg.eigvals(rot).astype(complex)
    return np.sort(r)

def _gauss(K: Literal[int], deg: int):
    if deg <= 0:
        raise ValueError("deg must be a positive integer")
    if K == 1:
        x = np.cos(np.pi * np.arange(1, 2 * deg, 2) / (2.0 * deg))
        w = np.ones(deg) * (np.pi / deg)
        return x, w
    c = np.zeros(deg + 1)
    c[deg] = 1.0
    x = np.linalg.eigvalsh(_companion(K, c))
    if K == 3 or K == 4:
        f = 2.0 if K == 3 else 1.0
        dy = _normed_herm(K, x, deg)
        df = _normed_herm(K, x, deg - 1) * np.sqrt(f * deg)
        x = x - dy / df
        fm = _normed_herm(K, x, deg - 1)
        fm = fm / np.abs(fm).max()
        w = 1.0 / (fm * fm)
    else:
        dc = _der(K, c)
        dy = _val(K, x, c)
        df = _val(K, x, dc)
        x = x - dy / df
        fm = _val(K, x, c[1:])
        fm = fm / np.abs(fm).max()
        df = df / np.abs(df).max()
        w = 1.0 / (fm * df)
    if K != 5:
        w = (w + w[::-1]) / 2.0
        x = (x - x[::-1]) / 2.0
    tot = 1.0
    if K == 2:
        tot = 2.0
    elif K == 3:
        tot = np.sqrt(np.pi)
    elif K == 4:
        tot = np.sqrt(2.0 * np.pi)
    w = w * (tot / w.sum())
    return x, w

def _normed_herm(K: Literal[int], x, n: int):
    f = 2.0 if K == 3 else 1.0
    base = np.sqrt(np.pi) if K == 3 else np.sqrt(2.0 * np.pi)
    if n == 0:
        return x * 0.0 + 1.0 / np.sqrt(base)
    c0 = x * 0.0
    c1 = x * 0.0 + 1.0 / np.sqrt(base)
    nd = float(n)
    for i in range(n - 1):
        tmp = c0
        c0 = -c1 * np.sqrt((nd - 1.0) / nd)
        c1 = tmp + c1 * x * np.sqrt(f / nd)
        nd = nd - 1.0
    return c0 + c1 * x * np.sqrt(f)

def _to_poly(K: Literal[int], c):
    c = as_series(c)
    if K == 0:
        return c
    return trimseq(_clenshaw_series(K, 0, c, np.array([1.0]), len(c)))

def _from_poly(K: Literal[int], pol):
    pol = as_series(pol)
    if K == 0:
        return pol
    n = len(pol)
    res = np.zeros(n + 1)
    for i in range(n - 1, -1, -1):
        t = np.zeros(n + 1)
        _mulx_into(K, res[:n], t)
        t[0] += pol[i]
        res = t
    return trimseq(res[:n])

def _weight(K: Literal[int], x):
    if K == 1:
        return 1.0 / (np.sqrt(1.0 + x) * np.sqrt(1.0 - x))
    elif K == 2:
        return x * 0.0 + 1.0
    elif K == 3:
        return np.exp(-x * x)
    elif K == 4:
        return np.exp(-0.5 * x * x)
    else:
        return np.exp(-x)
