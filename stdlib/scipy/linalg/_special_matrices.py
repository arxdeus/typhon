# Copyright (C) 2026 the Typhon authors
# Ported from scipy 1.18.1 (BSD-3-Clause)

# Special matrices not defined in _basic.py: hadamard, leslie, companion,
# helmert, hilbert, invhilbert, pascal, invpascal, dft, fiedler,
# fiedler_companion, convolution_matrix, khatri_rao, diagsvd.

import numpy as np
from ._basic import toeplitz

def _comb_i(n: int, k: int) -> int:
    if k < 0 or k > n or n < 0:
        return 0
    k = min(k, n - k)
    r = 1
    for i in range(1, k + 1):
        r = r * (n - k + i) // i
    return r

def _comb_f(n: int, k: int) -> float:
    if k < 0 or k > n or n < 0:
        return 0.0
    import scipy.special as sp
    return float(sp.comb(float(n), float(k)))

def hadamard(n: int, dtype: type = int):
    if n < 1:
        lg2 = 0
    else:
        lg2 = 0
        while (1 << (lg2 + 1)) <= n:
            lg2 += 1
    if (1 << lg2) != n:
        raise ValueError("n must be a positive integer, and n must be a power of 2")
    H = np.empty((n, n), dtype=dtype)
    for i in range(n):
        for j in range(n):
            # Sylvester: H[i,j] = (-1)^{popcount(i & j)}
            x = i & j
            c = 0
            while x:
                c += x & 1
                x >>= 1
            H[i, j] = dtype(1) if c % 2 == 0 else dtype(-1)
    return H

def leslie(f, s):
    f1 = np.atleast_1d(np.asarray(f))
    s1 = np.atleast_1d(np.asarray(s))
    if f1.ndim != 1 or s1.ndim != 1:
        compile_error("leslie: only 1-D input is supported")
    if f1.shape[0] != s1.shape[0] + 1:
        raise ValueError("Incorrect lengths for f and s. The length of s along the last axis must be one less than the length of f.")
    if s1.shape[0] == 0:
        raise ValueError("The length of s must be at least 1.")
    n = f1.shape[0]
    T = type(f1.data[0] + s1.data[0])
    a = np.zeros((n, n), dtype=T)
    for j in range(n):
        a[0, j] = T(f1[j])
    for i in range(1, n):
        a[i, i - 1] = T(s1[i - 1])
    return a

def companion(a):
    a1 = np.atleast_1d(np.asarray(a))
    if a1.ndim != 1:
        compile_error("companion: only 1-D input is supported")
    n = a1.shape[0]
    if n < 2:
        raise ValueError("The length of `a` along the last axis must be at least 2.")
    if a1[0] == type(a1[0])():
        raise ValueError("The first coefficient(s) of `a` (i.e. elements of `a[..., 0]`) must not be zero.")
    T = type(1.0 * a1.data[0])
    c = np.zeros((n - 1, n - 1), dtype=T)
    d = 1.0 * a1[0]
    for j in range(n - 1):
        c[0, j] = -a1[j + 1] / d
    for i in range(1, n - 1):
        c[i, i - 1] = T(1.0)
    return c

def helmert(n: int, full: bool = False):
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(i):
            H[i, j] = 1.0
        H[i, i] = -float(i)
    for j in range(n):
        H[0, j] = 1.0
    for i in range(n):
        d = float(n) if i == 0 else float(i * (i + 1))
        sd = np.sqrt(d)
        for j in range(n):
            H[i, j] = H[i, j] / sd
    if full:
        return H
    return H[1:].copy()

def hilbert(n: int):
    h = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            h[i, j] = 1.0 / (1.0 + float(i + j))
    return h

def invhilbert(n: int, exact: Literal[bool] = False):
    if exact:
        if n > 14:
            raise OverflowError("invhilbert(exact=True) with n > 14 needs object arrays (unsupported)")
        invh = np.empty((n, n), dtype=int)
        for i in range(n):
            for j in range(i + 1):
                s = i + j
                c = _comb_i(s, i)
                v = (s + 1) * _comb_i(n + i, n - j - 1) * _comb_i(n + j, n - i - 1) * c * c
                invh[i, j] = -v if s % 2 else v
                if i != j:
                    invh[j, i] = invh[i, j]
        return invh
    else:
        invh = np.empty((n, n))
        for i in range(n):
            for j in range(i + 1):
                s = i + j
                c = _comb_f(s, i)
                v = float(s + 1) * _comb_f(n + i, n - j - 1) * _comb_f(n + j, n - i - 1) * (c ** 2)
                invh[i, j] = (-1.0 if s % 2 else 1.0) * v
                if i != j:
                    invh[j, i] = invh[i, j]
        return invh

def _check_kind(kind: str):
    if kind != 'symmetric' and kind != 'lower' and kind != 'upper':
        raise ValueError("kind must be 'symmetric', 'lower', or 'upper'")

def pascal(n: int, kind: str = 'symmetric', exact: Literal[bool] = True):
    _check_kind(kind)
    if exact:
        if n >= 35:
            raise OverflowError("pascal(exact=True) with n >= 35 needs object arrays (unsupported)")
        L = np.zeros((n, n), dtype=u64)
        for i in range(n):
            for j in range(i + 1):
                L[i, j] = u64(_comb_i(i, j))
    else:
        L = np.empty((n, n))
        for i in range(n):
            for j in range(n):
                L[i, j] = _comb_f(i, j)
    if kind == 'lower':
        return L
    elif kind == 'upper':
        return L.T.copy()
    else:
        return np.dot(L, L.T)

def invpascal(n: int, kind: str = 'symmetric', exact: Literal[bool] = True):
    _check_kind(kind)
    if exact:
        if n > 34:
            raise OverflowError("invpascal(exact=True) with n > 34 needs object arrays (unsupported)")
        invp = np.empty((n, n), dtype=int)
        if kind == 'symmetric':
            for i in range(n):
                for j in range(i + 1):
                    v = 0
                    for k in range(n - i):
                        v += _comb_i(i + k, k) * _comb_i(i + k, i + k - j)
                    invp[i, j] = -v if (i - j) % 2 else v
                    if i != j:
                        invp[j, i] = invp[i, j]
        else:
            for i in range(n):
                for j in range(n):
                    a, b = (i, j) if kind == 'lower' else (j, i)
                    v = _comb_i(a, b)
                    invp[i, j] = -v if (i - j) % 2 else v
        return invp
    else:
        invq = np.empty((n, n))
        if kind == 'symmetric':
            for i in range(n):
                for j in range(i + 1):
                    v = 0.0
                    for k in range(n - i):
                        v += _comb_f(i + k, k) * _comb_f(i + k, i + k - j)
                    invq[i, j] = (-1.0 if (i - j) % 2 else 1.0) * v
                    if i != j:
                        invq[j, i] = invq[i, j]
        else:
            for i in range(n):
                for j in range(n):
                    a, b = (i, j) if kind == 'lower' else (j, i)
                    invq[i, j] = _comb_f(a, b) * (-1.0 if (i - j) % 2 else 1.0)
        return invq

def dft(n: int, scale: Optional[str] = None):
    if scale is not None:
        if scale != 'sqrtn' and scale != 'n':
            raise ValueError(f"scale must be None, 'sqrtn', or 'n'; '{scale}' is not valid.")
    m = np.empty((n, n), dtype=complex)
    for i in range(n):
        # omega_i = exp(-2j*pi*i/n); m[i, j] = omega_i ** j (numpy complex power)
        om = np.exp(complex(0.0, -2.0) * np.pi * float(i) / float(n))
        for j in range(n):
            m[i, j] = om ** complex(float(j), 0.0) if j > 0 else complex(1.0, 0.0)
    if scale is not None:
        d = np.sqrt(float(n)) if scale == 'sqrtn' else float(n)
        for i in range(n):
            for j in range(n):
                m[i, j] = m[i, j] / d
    return m

def fiedler(a):
    a1 = np.atleast_1d(np.asarray(a))
    if a1.ndim != 1:
        compile_error("fiedler: only 1-D input is supported")
    n = a1.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    F = np.empty((n, n), dtype=a1.dtype)
    for i in range(n):
        for j in range(n):
            F[i, j] = abs(a1[i] - a1[j])
    return F

def fiedler_companion(a):
    a0 = np.atleast_1d(np.asarray(a))
    if a0.ndim != 1:
        compile_error("fiedler_companion: only 1-D input is supported")
    T = type(a0.data[0] / a0.data[0])
    sz = a0.shape[0]
    if sz <= 2:
        if sz == 2:
            c = np.empty((1, 1), dtype=T)
            c[0, 0] = -(a0[1] / a0[0])
            return c
        return np.empty((0, 0), dtype=T)
    if a0[0] == type(a0[0])():
        raise ValueError('Leading coefficient is zero.')
    a1 = np.empty(sz, dtype=T)
    for i in range(sz):
        a1[i] = a0[i] / a0[0]
    n = sz - 1
    c = np.zeros((n, n), dtype=T)
    for i, j in zip(range(3, n, 2), range(1, n - 2, 2)):
        c[i, j] = T(1.0)
    k = 3
    for i, j in zip(range(2, n, 2), range(1, n - 1, 2)):
        c[i, j] = -a1[k]
        k += 2
    for i, j in zip(range(0, n - 2, 2), range(2, n, 2)):
        c[i, j] = T(1.0)
    k = 2
    for i, j in zip(range(0, n - 1, 2), range(1, n, 2)):
        c[i, j] = -a1[k]
        k += 2
    c[0, 0] = -a1[1]
    c[1, 0] = T(1.0)
    return c

def convolution_matrix(a, n: int, mode: str = 'full'):
    if n <= 0:
        raise ValueError('n must be a positive integer.')
    a1 = np.asarray(a)
    if a1.ndim != 1:
        compile_error("convolution_matrix: only 1-D input is supported")
    la = a1.shape[0]
    if la == 0:
        raise ValueError('len(a) must be at least 1.')
    if mode != 'full' and mode != 'valid' and mode != 'same':
        raise ValueError("'mode' argument must be one of ('full', 'valid', 'same')")
    L = la + n - 1
    az = np.zeros(L, dtype=a1.dtype)
    raz = np.zeros(L, dtype=a1.dtype)
    for i in range(la):
        az[i] = a1[i]
        raz[i] = a1[la - 1 - i]
    if mode == 'full':
        tb = 0
        te = 0
    elif mode == 'same':
        trim = min(n, la) - 1
        tb = trim // 2
        te = trim - tb
    else:
        tb = min(n, la) - 1
        te = tb
    col0 = az[tb:L - te].copy()
    row0 = raz[L - n - tb:L - tb].copy()
    return toeplitz(col0, row0)

def khatri_rao(a, b):
    a1 = np.asarray(a)
    b1 = np.asarray(b)
    if a1.ndim != 2 or b1.ndim != 2:
        compile_error("The both arrays should be 2-dimensional.")
    if a1.shape[1] != b1.shape[1]:
        raise ValueError("The number of columns for both arrays should be equal.")
    m, k = a1.shape
    p = b1.shape[0]
    T = type(a1.data[0] * b1.data[0])
    c = np.empty((m * p, k), dtype=T)
    for i in range(m):
        for r in range(p):
            for j in range(k):
                c[i * p + r, j] = a1[i, j] * b1[r, j]
    return c

def diagsvd(s, M: int, N: int):
    s1 = np.asarray(s)
    MorN = s1.shape[0]
    if MorN != M and MorN != N:
        raise ValueError("Length of s must be M or N.")
    out = np.zeros((M, N), dtype=s1.dtype)
    for i in range(MorN):
        out[i, i] = s1[i]
    return out
