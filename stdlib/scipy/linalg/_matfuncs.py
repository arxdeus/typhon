# Ported from scipy 1.18.1 (BSD-3-Clause)

# Matrix functions: expm (Al-Mohy & Higham 2009 scaling and squaring, as in
# scipy.sparse.linalg._expm with exact one-norms), sqrtm (complex Schur +
# Bjorck-Hammarling recurrence), logm (complex Schur + inverse scaling and
# squaring with Gauss-Legendre quadrature of log(I+X)), and
# fractional_matrix_power (integer powers exactly, otherwise expm(t*logm(A))).
#
# Deviation: for real input sqrtm/logm/fractional_matrix_power return a real
# array (the real part) because the return type must be static. scipy returns
# a complex array when the result has a non-negligible imaginary part.

import numpy as np
from numpy.linalg import LinAlgError
from numpy.linalg.blas import fint
from ._lapack import *
from ._util import _arr, _arr_lapack, _to_f_as, _from_f, _is_complex, _work_size
from ._basic import _solve_2d as solve

def _onenorm(A) -> float:
    n, m = A.shape
    best = 0.0
    for j in range(m):
        s = 0.0
        for i in range(n):
            s += abs(A[i, j])
        best = max(best, s)
    return best

def _abs_mat(A):
    n, m = A.shape
    r = np.empty((n, m), dtype=float)
    for i in range(n):
        for j in range(m):
            r[i, j] = abs(A[i, j])
    return r

def _onenorm_matrix_power_nnm(A, p: int) -> float:
    # ||A^p||_1 for nonnegative A, computed with vectors
    n = A.shape[0]
    v = np.ones(n, dtype=float)
    At = A.T.copy()
    for _ in range(p):
        v = At @ v
    mx = 0.0
    for x in v:
        mx = max(mx, x)
    return mx

def _ell(A, m: int) -> int:
    p = 2 * m + 1
    # c_i = abs(choose(2m+1 choose ...)): coefficient of the backward error
    c_i = 1.0
    # c = 1/(binom(2p, p) * factorial(2p+1)) = (p!)^2 / ((2p)! (2p+1)!)
    lg = 2.0 * _lgam(p + 1.0) - _lgam(2.0 * p + 1.0) - _lgam(2.0 * p + 2.0)
    abs_c_recip = np.exp(-lg)
    u = 2.0 ** -53
    A_abs_onenorm = _onenorm_matrix_power_nnm(_abs_mat(A), p)
    if A_abs_onenorm == 0.0:
        return 0
    alpha = A_abs_onenorm / (_onenorm(A) * abs_c_recip)
    log2_alpha_div_u = np.log2(alpha / u)
    value = int(np.ceil(log2_alpha_div_u / (2 * m)))
    return max(value, 0)

def _lgam(x: float) -> float:
    from math import lgamma
    return lgamma(x)

def _eye(n: int, T: type):
    return np.eye(n, dtype=T)

def _expm_2d(A):
    """Matrix exponential (2-D input)."""
    a = _arr_lapack(A, False)
    if a.ndim != 2:
        compile_error("expm: only 2-D input is supported")
    if a.shape[0] != a.shape[1]:
        raise LinAlgError('Last 2 dimensions of the array must be square')
    T = a.dtype
    n = a.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=T)
    if n == 1:
        r = np.empty((1, 1), dtype=T)
        r[0, 0] = np.exp(a[0, 0])
        return r
    upper = True
    for i in range(n):
        for j in range(i):
            if a[i, j] != T():
                upper = False
    ident = _eye(n, T)
    A2 = a @ a
    A4 = A2 @ A2
    A6 = A4 @ A2
    d4 = _onenorm(A4) ** 0.25
    d6 = _onenorm(A6) ** (1.0 / 6)
    b3 = (120., 60., 12., 1.)
    b5 = (30240., 15120., 3360., 420., 30., 1.)
    b7 = (17297280., 8648640., 1995840., 277200., 25200., 1512., 56., 1.)
    b9 = (17643225600., 8821612800., 2075673600., 302702400., 30270240.,
          2162160., 110880., 3960., 90., 1.)
    eta_1 = max(d4, d6)
    if eta_1 < 1.495585217958292e-002 and _ell(a, 3) == 0:
        U = a @ (b3[3] * A2 + b3[1] * ident)
        V = b3[2] * A2 + b3[0] * ident
        return _solve_PQ(U, V)
    eta_2 = max(d4, d6)
    if eta_2 < 2.539398330063230e-001 and _ell(a, 5) == 0:
        U = a @ (b5[5] * A4 + b5[3] * A2 + b5[1] * ident)
        V = b5[4] * A4 + b5[2] * A2 + b5[0] * ident
        return _solve_PQ(U, V)
    A8 = A6 @ A2
    d8 = _onenorm(A8) ** 0.125
    eta_3 = max(d6, d8)
    if eta_3 < 9.504178996162932e-001 and _ell(a, 7) == 0:
        U = a @ (b7[7] * A6 + b7[5] * A4 + b7[3] * A2 + b7[1] * ident)
        V = b7[6] * A6 + b7[4] * A4 + b7[2] * A2 + b7[0] * ident
        return _solve_PQ(U, V)
    if eta_3 < 2.097847961257068e+000 and _ell(a, 9) == 0:
        U = a @ (b9[9] * A8 + b9[7] * A6 + b9[5] * A4 + b9[3] * A2 + b9[1] * ident)
        V = b9[8] * A8 + b9[6] * A6 + b9[4] * A4 + b9[2] * A2 + b9[0] * ident
        return _solve_PQ(U, V)
    A10 = A4 @ A6
    d10 = _onenorm(A10) ** 0.1
    eta_4 = max(d8, d10)
    eta_5 = min(eta_3, eta_4)
    theta_13 = 4.25
    s = 0
    if eta_5 != 0:
        s = max(int(np.ceil(np.log2(eta_5 / theta_13))), 0)
    s = s + _ell((2.0 ** -s) * a, 13)
    b = (64764752532480000., 32382376266240000., 7771770303897600.,
         1187353796428800., 129060195264000., 10559470521600.,
         670442572800., 33522128640., 1323241920., 40840800., 960960.,
         16380., 182., 1.)
    B = a * (2.0 ** -s)
    B2 = A2 * (2.0 ** (-2 * s))
    B4 = A4 * (2.0 ** (-4 * s))
    B6 = A6 * (2.0 ** (-6 * s))
    U2 = B6 @ (b[13] * B6 + b[11] * B4 + b[9] * B2)
    U = B @ (U2 + b[7] * B6 + b[5] * B4 + b[3] * B2 + b[1] * ident)
    V2 = B6 @ (b[12] * B6 + b[10] * B4 + b[8] * B2)
    V = V2 + b[6] * B6 + b[4] * B4 + b[2] * B2 + b[0] * ident
    X = _solve_PQ(U, V)
    if upper:
        X = _fragment_2_1(X, a, s)
    else:
        for _ in range(s):
            X = X @ X
    return X

def _solve_PQ(U, V):
    P = U + V
    Q = -U + V
    return solve(Q, P)

def _exp_sinch(a: float, x: float) -> float:
    if abs(x) < 1e-8:
        return np.exp(a)
    return (np.exp(a + x) - np.exp(a - x)) / (2 * x)

def _eq_1_14(l1: float, l2: float, t12: float) -> float:
    a = (l1 + l2) / 2
    b = (l2 - l1) / 2
    return t12 * _exp_sinch(a, b)

def _fragment_2_1(X, T, s: int):
    n = X.shape[0]
    diag_T = [T[i, i] for i in range(n)]
    if not _is_complex(T.dtype):
        scale = 2.0 ** -s
        for i in range(n):
            X[i, i] = np.exp(scale * diag_T[i])
        for i in range(s - 1, -1, -1):
            X = X @ X
            scale = 2.0 ** -i
            for k in range(n):
                X[k, k] = np.exp(scale * diag_T[k])
            for k in range(n - 1):
                X[k, k + 1] = _eq_1_14(scale * diag_T[k], scale * diag_T[k + 1], scale * T[k, k + 1])
    else:
        for _ in range(s):
            X = X @ X
    return X

# Complex Schur decomposition: A = Z T Z^H
def _zschur(a):
    n = a.shape[0]
    A = _to_f_as(a, complex)
    VS = Ptr[complex](max(n * n, 1))
    W = Ptr[complex](max(n, 1))
    rwork = Ptr[float](max(n, 1))
    bwork = Ptr[fint](max(n, 1))
    jobvs = u8(86)
    srt = u8(78)
    N = fint(n); LD = fint(max(n, 1)); sdim = fint(0); info = fint(0)
    q = complex()
    LW = fint(-1)
    sel = cobj()
    for it in range(2):
        if it == 1:
            LW = fint(max(_work_size(q), 1))
            work = Ptr[complex](int(LW))
        else:
            work = __ptr__(q)
        zgees_(__ptr__(jobvs).as_byte(), __ptr__(srt).as_byte(), sel, __ptr__(N).as_byte(),
               A.as_byte(), __ptr__(LD).as_byte(), __ptr__(sdim).as_byte(), W.as_byte(),
               VS.as_byte(), __ptr__(LD).as_byte(), work.as_byte(), __ptr__(LW).as_byte(),
               rwork.as_byte(), bwork.as_byte(), __ptr__(info).as_byte())
    if int(info) != 0:
        raise LinAlgError("Schur form not found. Possibly ill-conditioned.")
    return _from_f(A, n, n, n), _from_f(VS, n, n, n)

def _ctrans(Z):
    n, m = Z.shape
    r = np.empty((m, n), dtype=complex)
    for i in range(n):
        for j in range(m):
            r[j, i] = Z[i, j].conjugate()
    return r

def _sqrtm_triu(T):
    n = T.shape[0]
    R = np.zeros((n, n), dtype=complex)
    for j in range(n):
        R[j, j] = _csqrt(T[j, j])
        for i in range(j - 1, -1, -1):
            s = complex()
            for k in range(i + 1, j):
                s += R[i, k] * R[k, j]
            denom = R[i, i] + R[j, j]
            num = T[i, j] - s
            if denom == complex() and num == complex():
                R[i, j] = complex()
            else:
                R[i, j] = num / denom
    return R

def _csqrt(z: complex) -> complex:
    import cmath
    return cmath.sqrt(z)

def _real_out(X, T: type):
    if _is_complex(T):
        return X
    else:
        n, m = X.shape
        r = np.empty((n, m), dtype=float)
        for i in range(n):
            for j in range(m):
                r[i, j] = X[i, j].real
        return r

def _sqrtm_2d(A):
    """Principal matrix square root (2-D input)."""
    a = _arr_lapack(A, True)
    if a.ndim != 2:
        compile_error("sqrtm: only 2-D input is supported")
    if a.shape[0] != a.shape[1]:
        raise LinAlgError('Last 2 dimensions of the array must be square')
    n = a.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=a.dtype)
    T, Z = _zschur(a)
    R = _sqrtm_triu(T)
    X = Z @ R @ _ctrans(Z)
    return _real_out(X, a.dtype)

# Gauss-Legendre nodes/weights on [0, 1] for log(I + X) = int_0^1 X (I + tX)^-1 dt
def _gauss_legendre(m: int):
    # Golub-Welsch would need eigh of the Jacobi matrix; use Newton iteration.
    xs = []
    ws = []
    for i in range(1, m + 1):
        x = np.cos(np.pi * (i - 0.25) / (m + 0.5))
        for _ in range(100):
            p0 = 1.0
            p1 = x
            for k in range(2, m + 1):
                p0, p1 = p1, ((2 * k - 1) * x * p1 - (k - 1) * p0) / k
            dp = m * (x * p1 - p0) / (x * x - 1)
            dx = p1 / dp
            x -= dx
            if abs(dx) < 1e-16:
                break
        p0 = 1.0
        p1 = x
        for k in range(2, m + 1):
            p0, p1 = p1, ((2 * k - 1) * x * p1 - (k - 1) * p0) / k
        dp = m * (x * p1 - p0) / (x * x - 1)
        xs.append((x + 1) / 2)
        ws.append(1.0 / ((1 - x * x) * dp * dp))
    return xs, ws

def _logm_triu(T0):
    n = T0.shape[0]
    T = T0.copy()
    k = 0
    ident = np.eye(n, dtype=complex)
    while True:
        mx = 0.0
        for i in range(n):
            for j in range(n):
                mx = max(mx, abs(T[i, j] - ident[i, j]))
        if mx < 0.05 or k > 60:
            break
        T = _sqrtm_triu(T)
        k += 1
    X = T - ident
    xs, ws = _gauss_legendre(12)
    L = np.zeros((n, n), dtype=complex)
    for q in range(len(xs)):
        M = ident + xs[q] * X
        L += ws[q] * _tri_solve_upper(M, X)
    return L * float(2 ** k)

def _tri_solve_upper(M, B):
    # solve M Y = B for upper triangular M
    n = M.shape[0]
    Y = np.zeros((n, n), dtype=complex)
    for c in range(n):
        for i in range(n - 1, -1, -1):
            s = B[i, c]
            for k in range(i + 1, n):
                s -= M[i, k] * Y[k, c]
            Y[i, c] = s / M[i, i]
    return Y

def _logm_2d(A, disp: bool = True):
    """Principal matrix logarithm (2-D input)."""
    a = _arr_lapack(A, True)
    if a.ndim != 2:
        compile_error("logm: only 2-D input is supported")
    if a.shape[0] != a.shape[1]:
        raise ValueError('expected square array_like input')
    n = a.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=a.dtype)
    T, Z = _zschur(a)
    for i in range(n):
        if T[i, i] == complex():
            raise LinAlgError("Matrix is singular; logm is undefined")
    L = _logm_triu(T)
    X = Z @ L @ _ctrans(Z)
    return _real_out(X, a.dtype)

def _fractional_matrix_power_2d(A, t):
    a = _arr_lapack(A, True)
    if a.ndim != 2:
        compile_error("fractional_matrix_power: only 2-D input is supported")
    if a.shape[0] != a.shape[1]:
        raise ValueError('expected a square matrix')
    tf = float(t)
    if tf == float(int(tf)):
        return np.linalg.matrix_power(a, int(tf))
    n = a.shape[0]
    T, Z = _zschur(a)
    L = _logm_triu(T)
    E = _expm_2d(tf * L)
    X = Z @ E @ _ctrans(Z)
    return _real_out(X, a.dtype)
