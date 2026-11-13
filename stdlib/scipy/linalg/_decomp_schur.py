# Copyright (C) 2026 the Typhon authors
# Ported from scipy 1.18.1 (BSD-3-Clause)

# schur (?gees, sorting by ?trsen as ?gees does internally), rsf2csf,
# hessenberg (?gebal + ?gehrd + ?orghr), matrix_balance (?gebal),
# qz / ordqz (?gges + ?tgsen).
#
# Deviation: ``output`` and ``sort``-is-None must be compile-time constants
# because they change the result type. Callable ``sort`` receives (re, im)
# for real output and a complex number for complex output, as in scipy.

import numpy as np
from numpy.linalg import LinAlgError
from numpy.linalg.blas import fint
from ._lapack import *
from ._util import _arr, _arr_lapack, _to_f_as, _from_f, _vec, _is_complex, \
                   _work_size, _Pair, _Triple

def _ch(c: str):
    return u8(ord(c))

def _real_t(T: type):
    return float()

def _gees(A: Ptr[T], n: int, T: type):
    # unsorted Schur form in place; returns (VS, wr, wi) (wi unused for complex)
    VS = Ptr[T](max(n * n, 1))
    WR = Ptr[T](max(n, 1))
    WI = Ptr[float](max(n, 1))
    rwork = Ptr[float](max(n, 1))
    bwork = Ptr[fint](max(n, 1))
    jobvs = _ch('V'); srt = _ch('N')
    N = fint(n); LD = fint(max(n, 1)); sdim = fint(0); info = fint(0)
    q = T()
    LW = fint(-1)
    sel = cobj()
    for it in range(2):
        if it == 1:
            LW = fint(max(_work_size(q), 1))
            work = Ptr[T](int(LW))
        else:
            work = __ptr__(q)
        if _is_complex(T):
            zgees_(__ptr__(jobvs).as_byte(), __ptr__(srt).as_byte(), sel, __ptr__(N).as_byte(),
                   A.as_byte(), __ptr__(LD).as_byte(), __ptr__(sdim).as_byte(), WR.as_byte(),
                   VS.as_byte(), __ptr__(LD).as_byte(), work.as_byte(), __ptr__(LW).as_byte(),
                   rwork.as_byte(), bwork.as_byte(), __ptr__(info).as_byte())
        else:
            dgees_(__ptr__(jobvs).as_byte(), __ptr__(srt).as_byte(), sel, __ptr__(N).as_byte(),
                   A.as_byte(), __ptr__(LD).as_byte(), __ptr__(sdim).as_byte(), WR.as_byte(),
                   WI.as_byte(), VS.as_byte(), __ptr__(LD).as_byte(), work.as_byte(),
                   __ptr__(LW).as_byte(), bwork.as_byte(), __ptr__(info).as_byte())
    i = int(info)
    if i < 0:
        raise ValueError(f'illegal value in {-i}-th argument of internal gees')
    elif i > 0:
        raise LinAlgError("Schur form not found. Possibly ill-conditioned.")
    return VS, WR, WI

def _sort_pred(sort, x: float, y: float) -> bool:
    if isinstance(sort, str):
        z = complex(x, y)
        if sort == 'lhp':
            return x < 0.0
        elif sort == 'rhp':
            return x >= 0.0
        elif sort == 'iuc':
            return abs(z) <= 1.0
        elif sort == 'ouc':
            return abs(z) > 1.0
        else:
            raise ValueError("'sort' parameter must either be 'None', or a callable, or one of ('lhp','rhp','iuc','ouc')")
    else:
        return bool(sort(x, y))

def _sort_pred_c(sort, z: complex) -> bool:
    if isinstance(sort, str):
        return _sort_pred(sort, z.real, z.imag)
    else:
        return bool(sort(z))

def _trsen(A: Ptr[T], VS: Ptr[T], n: int, sel: Ptr[fint], T: type) -> int:
    job = _ch('N'); compq = _ch('V')
    N = fint(n); LD = fint(max(n, 1)); M = fint(0); info = fint(0)
    s = 0.0; sep = 0.0
    WR = Ptr[T](max(n, 1))
    WI = Ptr[float](max(n, 1))
    LW = fint(max(n, 1)); LIW = fint(1)
    work = Ptr[T](max(n, 1))
    iwork = Ptr[fint](1)
    if _is_complex(T):
        ztrsen_(__ptr__(job).as_byte(), __ptr__(compq).as_byte(), sel.as_byte(), __ptr__(N).as_byte(),
                A.as_byte(), __ptr__(LD).as_byte(), VS.as_byte(), __ptr__(LD).as_byte(), WR.as_byte(),
                __ptr__(M).as_byte(), __ptr__(s).as_byte(), __ptr__(sep).as_byte(), work.as_byte(),
                __ptr__(LW).as_byte(), __ptr__(info).as_byte())
    else:
        dtrsen_(__ptr__(job).as_byte(), __ptr__(compq).as_byte(), sel.as_byte(), __ptr__(N).as_byte(),
                A.as_byte(), __ptr__(LD).as_byte(), VS.as_byte(), __ptr__(LD).as_byte(), WR.as_byte(),
                WI.as_byte(), __ptr__(M).as_byte(), __ptr__(s).as_byte(), __ptr__(sep).as_byte(),
                work.as_byte(), __ptr__(LW).as_byte(), iwork.as_byte(), __ptr__(LIW).as_byte(),
                __ptr__(info).as_byte())
    if int(info) != 0:
        raise LinAlgError('Eigenvalues could not be separated for reordering.')
    return int(M)

def _schur_impl(a, output: Literal[str], sort, check_finite: bool):
    if output != 'real' and output != 'complex' and output != 'r' and output != 'c':
        compile_error("argument must be 'real', or 'complex'")
    a0 = _arr_lapack(a, check_finite)
    if a0.ndim != 2:
        compile_error("expected square matrix")
    if a0.shape[0] != a0.shape[1]:
        raise ValueError('expected square matrix')
    n = a0.shape[0]
    if output == 'complex' or output == 'c':
        a1 = a0.astype(complex)
    else:
        a1 = a0
    T = a1.dtype
    A = _to_f_as(a1, T)
    if n == 0:
        VS = Ptr[T](1)
        WR = Ptr[T](1)
        WI = Ptr[float](1)
    else:
        VS, WR, WI = _gees(A, n, T)
    sdim = 0
    if sort is not None:
        sel = Ptr[fint](max(n, 1))
        for i in range(n):
            if _is_complex(T):
                sel[i] = fint(1) if _sort_pred_c(sort, WR[i]) else fint(0)
            else:
                sel[i] = fint(1) if _sort_pred(sort, WR[i], WI[i]) else fint(0)
        if n > 0:
            sdim = _trsen(A, VS, n, sel, T)
            # ?gees: for real pairs, the count includes both members when selected
    return _Triple(_from_f(A, n, n, max(n, 1)), _from_f(VS, n, n, max(n, 1)), sdim)

def schur(a, output: Literal[str] = 'real', lwork = None, overwrite_a: bool = False,
          sort = None, check_finite: bool = True):
    r = _schur_impl(a, output, sort, check_finite)
    if sort is None:
        return _Pair(r[0], r[1])
    else:
        return r

_eps = 2.220446049250313e-16

def rsf2csf(T, Z, check_finite: bool = True):
    Z1 = _arr_lapack(Z, check_finite).astype(complex)
    T1 = _arr_lapack(T, check_finite).astype(complex)
    if Z1.ndim != 2 or Z1.shape[0] != Z1.shape[1]:
        raise ValueError("Input 'Z' must be square.")
    if T1.shape[0] != T1.shape[1]:
        raise ValueError("Input 'T' must be square.")
    if T1.shape[0] != Z1.shape[0]:
        raise ValueError(f'Input array shapes must match: Z: {Z1.shape} vs. T: {T1.shape}')
    N = T1.shape[0]
    for m in range(N - 1, 0, -1):
        if abs(T1[m, m - 1]) > _eps * (abs(T1[m - 1, m - 1]) + abs(T1[m, m])):
            # eigenvalues of the 2x2 block (as scipy's eigvals via ?geev)
            from ._basic import _eigvals_2d
            blk = np.empty((2, 2), dtype=complex)
            blk[0, 0] = T1[m - 1, m - 1]; blk[0, 1] = T1[m - 1, m]
            blk[1, 0] = T1[m, m - 1]; blk[1, 1] = T1[m, m]
            ev = _eigvals_2d(blk)
            mu0 = ev[0] - T1[m, m]
            t = T1[m, m - 1]
            r = np.sqrt(abs(mu0) ** 2 + abs(t) ** 2)
            c = mu0 / r
            s = t / r
            G00 = c.conjugate(); G01 = s; G10 = -s; G11 = c
            for j in range(m - 1, N):
                x = T1[m - 1, j]; y = T1[m, j]
                T1[m - 1, j] = G00 * x + G01 * y
                T1[m, j] = G10 * x + G11 * y
            # right-multiply by G^H: [[conj(G00), conj(G10)], [conj(G01), conj(G11)]]
            for i in range(m + 1):
                x = T1[i, m - 1]; y = T1[i, m]
                T1[i, m - 1] = x * G00.conjugate() + y * G01.conjugate()
                T1[i, m] = x * G10.conjugate() + y * G11.conjugate()
            for i in range(N):
                x = Z1[i, m - 1]; y = Z1[i, m]
                Z1[i, m - 1] = x * G00.conjugate() + y * G01.conjugate()
                Z1[i, m] = x * G10.conjugate() + y * G11.conjugate()
        T1[m, m - 1] = complex()
    return _Pair(T1, Z1)

def _gebal(A: Ptr[T], n: int, job: u8, T: type):
    N = fint(n); LD = fint(max(n, 1)); lo = fint(0); hi = fint(0); info = fint(0)
    scl = Ptr[float](max(n, 1))
    if _is_complex(T):
        zgebal_(__ptr__(job).as_byte(), __ptr__(N).as_byte(), A.as_byte(), __ptr__(LD).as_byte(),
                __ptr__(lo).as_byte(), __ptr__(hi).as_byte(), scl.as_byte(), __ptr__(info).as_byte())
    else:
        dgebal_(__ptr__(job).as_byte(), __ptr__(N).as_byte(), A.as_byte(), __ptr__(LD).as_byte(),
                __ptr__(lo).as_byte(), __ptr__(hi).as_byte(), scl.as_byte(), __ptr__(info).as_byte())
    if int(info) < 0:
        raise ValueError(f'xGEBAL exited with the internal error "illegal value in argument number {-int(info)}.". See LAPACK documentation for the xGEBAL error codes.')
    return int(lo) - 1, int(hi) - 1, scl

def matrix_balance(A, permute: bool = True, scale: bool = True,
                   separate: Literal[bool] = False, overwrite_a: bool = False):
    a1 = _arr_lapack(np.atleast_2d(_arr(A)), True)
    if a1.shape[0] != a1.shape[1]:
        raise ValueError('The data matrix for balancing should be square.')
    n = a1.shape[0]
    T = a1.dtype
    Ap = _to_f_as(a1, T)
    job = _ch('B') if (permute and scale) else (_ch('P') if permute else (_ch('S') if scale else _ch('N')))
    lo = 0; hi = n - 1
    ps = Ptr[float](max(n, 1))
    if n > 0:
        lo, hi, ps = _gebal(Ap, n, job)
    B = _from_f(Ap, n, n, max(n, 1))
    scaling = np.ones(n, dtype=float)
    for i in range(lo, hi + 1):
        scaling[i] = ps[i]
    psi = [int(ps[i]) - 1 for i in range(n)]
    perm = np.arange(n)
    if hi < n:
        ind = 1
        for k in range(n - 1, hi, -1):
            x = psi[k]
            if n - ind != x:
                t = perm[x]; perm[x] = perm[n - ind]; perm[n - ind] = t
            ind += 1
    if lo > 0:
        for ind in range(lo):
            x = psi[ind]
            if ind != x:
                t = perm[x]; perm[x] = perm[ind]; perm[ind] = t
    if separate:
        return _Pair(B, _Pair(scaling, perm))
    else:
        iperm = np.empty(n, dtype=int)
        for i in range(n):
            iperm[perm[i]] = i
        Tm = np.zeros((n, n), dtype=float)
        for i in range(n):
            Tm[i, iperm[i]] = scaling[iperm[i]]
        return _Pair(B, Tm)

def hessenberg(a, calc_q: Literal[bool] = False, overwrite_a: bool = False,
               check_finite: bool = True):
    a1 = _arr_lapack(a, check_finite)
    if a1.ndim != 2:
        compile_error('expected square matrix')
    if a1.shape[0] != a1.shape[1]:
        raise ValueError('expected square matrix')
    T = a1.dtype
    n = a1.shape[0]
    if n <= 2:
        if calc_q:
            return _Pair(a1.copy(), np.eye(n, dtype=T))
        else:
            return a1.copy()
    A = _to_f_as(a1, T)
    # scipy: gebal(a1, permute=0) with f2py default scale=0 -> job 'N'
    lo, hi, _ = _gebal(A, n, _ch('N'))
    N = fint(n); LO = fint(lo + 1); HI = fint(hi + 1); LD = fint(n); info = fint(0)
    tau = Ptr[T](max(n - 1, 1))
    q = T()
    LW = fint(-1)
    for it in range(2):
        if it == 1:
            LW = fint(max(_work_size(q), 1))
            work = Ptr[T](int(LW))
        else:
            work = __ptr__(q)
        if _is_complex(T):
            zgehrd_(__ptr__(N).as_byte(), __ptr__(LO).as_byte(), __ptr__(HI).as_byte(), A.as_byte(), __ptr__(LD).as_byte(), tau.as_byte(), work.as_byte(), __ptr__(LW).as_byte(), __ptr__(info).as_byte())
        else:
            dgehrd_(__ptr__(N).as_byte(), __ptr__(LO).as_byte(), __ptr__(HI).as_byte(), A.as_byte(), __ptr__(LD).as_byte(), tau.as_byte(), work.as_byte(), __ptr__(LW).as_byte(), __ptr__(info).as_byte())
    h = np.zeros((n, n), dtype=T)
    for i in range(n):
        for j in range(max(i - 1, 0), n):
            h[i, j] = A[i + j * n]
    if not calc_q:
        return h
    else:
        q2 = T()
        LW = fint(-1)
        for it in range(2):
            if it == 1:
                LW = fint(max(_work_size(q2), 1))
                work = Ptr[T](int(LW))
            else:
                work = __ptr__(q2)
            if _is_complex(T):
                zunghr_(__ptr__(N).as_byte(), __ptr__(LO).as_byte(), __ptr__(HI).as_byte(), A.as_byte(), __ptr__(LD).as_byte(), tau.as_byte(), work.as_byte(), __ptr__(LW).as_byte(), __ptr__(info).as_byte())
            else:
                dorghr_(__ptr__(N).as_byte(), __ptr__(LO).as_byte(), __ptr__(HI).as_byte(), A.as_byte(), __ptr__(LD).as_byte(), tau.as_byte(), work.as_byte(), __ptr__(LW).as_byte(), __ptr__(info).as_byte())
        return _Pair(h, _from_f(A, n, n, n))

# ---------------------------------------------------------------- QZ

def _gges(A: Ptr[T], B: Ptr[T], n: int, T: type):
    VSL = Ptr[T](max(n * n, 1))
    VSR = Ptr[T](max(n * n, 1))
    AR = Ptr[T](max(n, 1))
    AI = Ptr[float](max(n, 1))
    BE = Ptr[T](max(n, 1))
    rwork = Ptr[float](max(8 * n, 1))
    bwork = Ptr[fint](max(n, 1))
    jv = _ch('V'); srt = _ch('N')
    N = fint(n); LD = fint(max(n, 1)); sdim = fint(0); info = fint(0)
    q = T()
    LW = fint(-1)
    sel = cobj()
    for it in range(2):
        if it == 1:
            LW = fint(max(_work_size(q), 1))
            work = Ptr[T](int(LW))
        else:
            work = __ptr__(q)
        if _is_complex(T):
            zgges_(__ptr__(jv).as_byte(), __ptr__(jv).as_byte(), __ptr__(srt).as_byte(), sel,
                   __ptr__(N).as_byte(), A.as_byte(), __ptr__(LD).as_byte(), B.as_byte(), __ptr__(LD).as_byte(),
                   __ptr__(sdim).as_byte(), AR.as_byte(), BE.as_byte(), VSL.as_byte(), __ptr__(LD).as_byte(),
                   VSR.as_byte(), __ptr__(LD).as_byte(), work.as_byte(), __ptr__(LW).as_byte(),
                   rwork.as_byte(), bwork.as_byte(), __ptr__(info).as_byte())
        else:
            dgges_(__ptr__(jv).as_byte(), __ptr__(jv).as_byte(), __ptr__(srt).as_byte(), sel,
                   __ptr__(N).as_byte(), A.as_byte(), __ptr__(LD).as_byte(), B.as_byte(), __ptr__(LD).as_byte(),
                   __ptr__(sdim).as_byte(), AR.as_byte(), AI.as_byte(), BE.as_byte(), VSL.as_byte(),
                   __ptr__(LD).as_byte(), VSR.as_byte(), __ptr__(LD).as_byte(), work.as_byte(),
                   __ptr__(LW).as_byte(), bwork.as_byte(), __ptr__(info).as_byte())
    i = int(info)
    if i < 0:
        raise ValueError(f'Illegal value in argument {-i} of gges')
    elif i == n + 1:
        raise LinAlgError('Something other than QZ iteration failed')
    elif i > n:
        raise LinAlgError('Reordering failed in <s,d,c,z>tgsen')
    return VSL, VSR, AR, AI, BE

def _qz_prep(A, B, output: Literal[str], check_finite: bool):
    if output != 'real' and output != 'complex' and output != 'r' and output != 'c':
        compile_error("argument must be 'real', or 'complex'")
    a0 = _arr_lapack(A, check_finite)
    b0 = _arr_lapack(B, check_finite)
    if a0.ndim != 2 or b0.ndim != 2:
        compile_error("expected 2-D arrays")
    n = a0.shape[0]
    if not (a0.shape[1] == n and b0.shape[0] == n and b0.shape[1] == n):
        raise ValueError('Array dimensions must be square and agree')
    if output == 'complex' or output == 'c' or _is_complex(a0.dtype) or _is_complex(b0.dtype):
        return a0.astype(complex), b0.astype(complex), n
    else:
        return a0, b0, n

def qz(A, B, output: Literal[str] = 'real', lwork = None, sort = None,
       overwrite_a: bool = False, overwrite_b: bool = False, check_finite: bool = True):
    if sort is not None:
        raise ValueError("The 'sort' input of qz() has to be None and will be removed in a future release. Use ordqz instead.")
    a1, b1, n = _qz_prep(A, B, output, check_finite)
    T = a1.dtype
    Ap = _to_f_as(a1, T)
    Bp = _to_f_as(b1, T)
    VSL, VSR, AR, AI, BE = _gges(Ap, Bp, n, T)
    ld = max(n, 1)
    return (_from_f(Ap, n, n, ld), _from_f(Bp, n, n, ld), _from_f(VSL, n, n, ld), _from_f(VSR, n, n, ld))

def _qz_select(sort, x: complex, y) -> bool:
    if isinstance(sort, str):
        if sort == 'lhp':
            return False if y == type(y)() else (x / y).real < 0.0
        elif sort == 'rhp':
            return False if y == type(y)() else (x / y).real > 0.0
        elif sort == 'iuc':
            return False if y == type(y)() else abs(x / y) < 1.0
        elif sort == 'ouc':
            if y == type(y)():
                return x != complex()
            return abs(x / y) > 1.0
        else:
            raise ValueError("sort parameter must be None, a callable, or one of ('lhp','rhp','iuc','ouc')")
    else:
        return bool(sort(x, y))

def ordqz(A, B, sort = 'lhp', output: Literal[str] = 'real', overwrite_a: bool = False,
          overwrite_b: bool = False, check_finite: bool = True):
    a1, b1, n = _qz_prep(A, B, output, check_finite)
    T = a1.dtype
    Ap = _to_f_as(a1, T)
    Bp = _to_f_as(b1, T)
    VSL, VSR, AR, AI, BE = _gges(Ap, Bp, n, T)
    sel = Ptr[fint](max(n, 1))
    for i in range(n):
        if _is_complex(T):
            al = AR[i]
        else:
            al = complex(AR[i], AI[i])
        sel[i] = fint(1) if _qz_select(sort, al, BE[i]) else fint(0)
    IJOB = fint(0); WQ = fint(1); WZ = fint(1)
    N = fint(n); LD = fint(max(n, 1)); M = fint(0); info = fint(0)
    pl = 0.0; pr = 0.0
    dif = Ptr[float](2)
    LW = fint(4 * n + 16 if not _is_complex(T) else 1)
    LIW = fint(1)
    work = Ptr[T](max(int(LW), 1))
    iwork = Ptr[fint](1)
    if n > 0:
        if _is_complex(T):
            ztgsen_(__ptr__(IJOB).as_byte(), __ptr__(WQ).as_byte(), __ptr__(WZ).as_byte(), sel.as_byte(),
                    __ptr__(N).as_byte(), Ap.as_byte(), __ptr__(LD).as_byte(), Bp.as_byte(), __ptr__(LD).as_byte(),
                    AR.as_byte(), BE.as_byte(), VSL.as_byte(), __ptr__(LD).as_byte(), VSR.as_byte(),
                    __ptr__(LD).as_byte(), __ptr__(M).as_byte(), __ptr__(pl).as_byte(), __ptr__(pr).as_byte(),
                    dif.as_byte(), work.as_byte(), __ptr__(LW).as_byte(), iwork.as_byte(), __ptr__(LIW).as_byte(),
                    __ptr__(info).as_byte())
        else:
            dtgsen_(__ptr__(IJOB).as_byte(), __ptr__(WQ).as_byte(), __ptr__(WZ).as_byte(), sel.as_byte(),
                    __ptr__(N).as_byte(), Ap.as_byte(), __ptr__(LD).as_byte(), Bp.as_byte(), __ptr__(LD).as_byte(),
                    AR.as_byte(), AI.as_byte(), BE.as_byte(), VSL.as_byte(), __ptr__(LD).as_byte(), VSR.as_byte(),
                    __ptr__(LD).as_byte(), __ptr__(M).as_byte(), __ptr__(pl).as_byte(), __ptr__(pr).as_byte(),
                    dif.as_byte(), work.as_byte(), __ptr__(LW).as_byte(), iwork.as_byte(), __ptr__(LIW).as_byte(),
                    __ptr__(info).as_byte())
    i = int(info)
    if i < 0:
        raise ValueError(f'Illegal value in argument {-i} of tgsen')
    elif i == 1:
        raise ValueError('Reordering of (A, B) failed because the transformed matrix pair (A, B) would be too far from generalized Schur form; the problem is very ill-conditioned. (A, B) may have been partially reordered.')
    alpha = np.empty(n, dtype=complex)
    beta = np.empty(n, dtype=T)
    for k in range(n):
        if _is_complex(T):
            alpha[k] = AR[k]
        else:
            alpha[k] = complex(AR[k], AI[k])
        beta[k] = BE[k]
    ld = max(n, 1)
    return (_from_f(Ap, n, n, ld), _from_f(Bp, n, n, ld), alpha, beta,
            _from_f(VSL, n, n, ld), _from_f(VSR, n, n, ld))
