# Copyright (C) 2026 the Typhon authors
# Ported from scipy 1.18.1 (BSD-3-Clause)
# scipy.linalg.cossin (scipy/linalg/_decomp_cossin.py) on top of LAPACK
# ?orcsd / ?uncsd. Deviations: 2-D inputs only (no batching); with
# compute_u / compute_vh False the corresponding factors are returned as
# empty (0, 0) arrays.

import math
import numpy as np
from numpy.linalg import LinAlgError
from numpy.linalg.blas import fint
from ._lapack import dorcsd_, zuncsd_
from ._util import _to_f, _from_f, _vec, _work_size
from ._basic import block_diag

def _any_cplx(*xs) -> Literal[bool]:
    if static.len(xs) == 0:
        return False
    elif np.asarray(xs[0]).dtype is complex or np.asarray(xs[0]).dtype is complex64:
        return True
    else:
        return _any_cplx(*xs[1:])

def _csd_core(x11, x12, x21, x22, T: type, swap_sign: bool, compute_u: bool, compute_vh: bool):
    a11 = np.asarray(x11).astype(T)
    a12 = np.asarray(x12).astype(T)
    a21 = np.asarray(x21).astype(T)
    a22 = np.asarray(x22).astype(T)
    for blk in (a11, a12, a21, a22):
        if blk.ndim != 2:
            raise ValueError("blocks must be 2-D")
    if a11.shape[1] == 0:
        raise ValueError("x11 can't be empty")
    if a12.shape[1] == 0:
        raise ValueError("x12 can't be empty")
    if a21.shape[1] == 0:
        raise ValueError("x21 can't be empty")
    if a22.shape[1] == 0:
        raise ValueError("x22 can't be empty")
    p = a11.shape[0]
    q = a11.shape[1]
    mmp = a22.shape[0]
    mmq = a22.shape[1]
    if a12.shape[0] != p or a12.shape[1] != mmq:
        raise ValueError(f"Invalid x12 dimensions: desired {(p, mmq)}, got {a12.shape}")
    if a21.shape[0] != mmp or a21.shape[1] != q:
        raise ValueError(f"Invalid x21 dimensions: desired {(mmp, q)}, got {a21.shape}")
    if p + mmp != q + mmq:
        raise ValueError(f"The subblocks have compatible sizes but don't form a square array (instead they form a {p + mmp}x{q + mmq} array). This might be due to missing p, q arguments.")
    m = p + mmp
    r = min(min(p, q), min(m - p, m - q))
    cplx: Literal[bool] = T is complex
    jobu1 = u8(89) if compute_u else u8(78)
    jobv1t = u8(89) if compute_vh else u8(78)
    tr = u8(78)
    sg = u8(79) if swap_sign else u8(68)
    M = fint(m)
    P = fint(p)
    Q = fint(q)
    X11 = _to_f(a11, T)
    X12 = _to_f(a12, T)
    X21 = _to_f(a21, T)
    X22 = _to_f(a22, T)
    LD11 = fint(max(p, 1))
    LD12 = fint(max(p, 1))
    LD21 = fint(max(mmp, 1))
    LD22 = fint(max(mmp, 1))
    theta = Ptr[float](max(r, 1))
    ncu = p if compute_u else 0
    ncu2 = mmp if compute_u else 0
    ncv = q if compute_vh else 0
    ncv2 = mmq if compute_vh else 0
    U1 = Ptr[T](max(ncu * ncu, 1))
    U2 = Ptr[T](max(ncu2 * ncu2, 1))
    V1 = Ptr[T](max(ncv * ncv, 1))
    V2 = Ptr[T](max(ncv2 * ncv2, 1))
    LU1 = fint(max(ncu, 1))
    LU2 = fint(max(ncu2, 1))
    LV1 = fint(max(ncv, 1))
    LV2 = fint(max(ncv2, 1))
    iwork = Ptr[fint](max(m - r, 1))
    info = fint(0)
    qw = T()
    qrw = 0.0
    LW = fint(-1)
    LRW = fint(-1)
    if cplx:
        zuncsd_(__ptr__(jobu1).as_byte(), __ptr__(jobu1).as_byte(), __ptr__(jobv1t).as_byte(), __ptr__(jobv1t).as_byte(), __ptr__(tr).as_byte(), __ptr__(sg).as_byte(),
                __ptr__(M).as_byte(), __ptr__(P).as_byte(), __ptr__(Q).as_byte(), X11.as_byte(), __ptr__(LD11).as_byte(), X12.as_byte(), __ptr__(LD12).as_byte(),
                X21.as_byte(), __ptr__(LD21).as_byte(), X22.as_byte(), __ptr__(LD22).as_byte(), theta.as_byte(), U1.as_byte(), __ptr__(LU1).as_byte(), U2.as_byte(), __ptr__(LU2).as_byte(),
                V1.as_byte(), __ptr__(LV1).as_byte(), V2.as_byte(), __ptr__(LV2).as_byte(), __ptr__(qw).as_byte(), __ptr__(LW).as_byte(), __ptr__(qrw).as_byte(), __ptr__(LRW).as_byte(),
                iwork.as_byte(), __ptr__(info).as_byte())
    else:
        dorcsd_(__ptr__(jobu1).as_byte(), __ptr__(jobu1).as_byte(), __ptr__(jobv1t).as_byte(), __ptr__(jobv1t).as_byte(), __ptr__(tr).as_byte(), __ptr__(sg).as_byte(),
                __ptr__(M).as_byte(), __ptr__(P).as_byte(), __ptr__(Q).as_byte(), X11.as_byte(), __ptr__(LD11).as_byte(), X12.as_byte(), __ptr__(LD12).as_byte(),
                X21.as_byte(), __ptr__(LD21).as_byte(), X22.as_byte(), __ptr__(LD22).as_byte(), theta.as_byte(), U1.as_byte(), __ptr__(LU1).as_byte(), U2.as_byte(), __ptr__(LU2).as_byte(),
                V1.as_byte(), __ptr__(LV1).as_byte(), V2.as_byte(), __ptr__(LV2).as_byte(), __ptr__(qw).as_byte(), __ptr__(LW).as_byte(),
                iwork.as_byte(), __ptr__(info).as_byte())
    LW = fint(max(_work_size(qw), 1))
    LRW = fint(max(int(qrw), 1))
    work = Ptr[T](int(LW))
    rwork = Ptr[float](int(LRW))
    info = fint(0)
    if cplx:
        zuncsd_(__ptr__(jobu1).as_byte(), __ptr__(jobu1).as_byte(), __ptr__(jobv1t).as_byte(), __ptr__(jobv1t).as_byte(), __ptr__(tr).as_byte(), __ptr__(sg).as_byte(),
                __ptr__(M).as_byte(), __ptr__(P).as_byte(), __ptr__(Q).as_byte(), X11.as_byte(), __ptr__(LD11).as_byte(), X12.as_byte(), __ptr__(LD12).as_byte(),
                X21.as_byte(), __ptr__(LD21).as_byte(), X22.as_byte(), __ptr__(LD22).as_byte(), theta.as_byte(), U1.as_byte(), __ptr__(LU1).as_byte(), U2.as_byte(), __ptr__(LU2).as_byte(),
                V1.as_byte(), __ptr__(LV1).as_byte(), V2.as_byte(), __ptr__(LV2).as_byte(), work.as_byte(), __ptr__(LW).as_byte(), rwork.as_byte(), __ptr__(LRW).as_byte(),
                iwork.as_byte(), __ptr__(info).as_byte())
    else:
        dorcsd_(__ptr__(jobu1).as_byte(), __ptr__(jobu1).as_byte(), __ptr__(jobv1t).as_byte(), __ptr__(jobv1t).as_byte(), __ptr__(tr).as_byte(), __ptr__(sg).as_byte(),
                __ptr__(M).as_byte(), __ptr__(P).as_byte(), __ptr__(Q).as_byte(), X11.as_byte(), __ptr__(LD11).as_byte(), X12.as_byte(), __ptr__(LD12).as_byte(),
                X21.as_byte(), __ptr__(LD21).as_byte(), X22.as_byte(), __ptr__(LD22).as_byte(), theta.as_byte(), U1.as_byte(), __ptr__(LU1).as_byte(), U2.as_byte(), __ptr__(LU2).as_byte(),
                V1.as_byte(), __ptr__(LV1).as_byte(), V2.as_byte(), __ptr__(LV2).as_byte(), work.as_byte(), __ptr__(LW).as_byte(),
                iwork.as_byte(), __ptr__(info).as_byte())
    meth = 'zuncsd' if cplx else 'dorcsd'
    if int(info) < 0:
        raise ValueError(f'illegal value in argument {-int(info)} of internal {meth}')
    if int(info) > 0:
        raise LinAlgError(f"{meth} did not converge: {int(info)}")
    th = _vec(theta, r)
    u1 = _from_f(U1, ncu, ncu, ncu, T)
    u2 = _from_f(U2, ncu2, ncu2, ncu2, T)
    v1h = _from_f(V1, ncv, ncv, ncv, T)
    v2h = _from_f(V2, ncv2, ncv2, ncv2, T)
    return th, u1, u2, v1h, v2h, p, q, m

def _build_cs(th: np.ndarray[float, 1], p: int, q: int, m: int, swap_sign: bool) -> np.ndarray[float, 2]:
    r = min(min(p, q), min(m - p, m - q))
    n11 = min(p, q) - r
    n12 = min(p, m - q) - r
    n21 = min(m - p, q) - r
    n22 = min(m - p, m - q) - r
    CS = np.zeros((m, m))
    for i in range(n11):
        CS[i, i] = 1.0
    sg12 = 1.0 if swap_sign else -1.0
    xs = n11 + r
    ys = n11 + n21 + n22 + 2 * r
    for i in range(n12):
        CS[xs + i, ys + i] = sg12
    sg21 = -1.0 if swap_sign else 1.0
    xs = p + n22 + r
    ys = n11 + r
    for i in range(n21):
        CS[xs + i, ys + i] = sg21
    for i in range(n22):
        CS[p + i, q + i] = 1.0
    for i in range(r):
        c = math.cos(th[i])
        s = math.sin(th[i])
        CS[n11 + i, n11 + i] = c
        CS[p + n22 + i, n11 + r + n21 + n22 + i] = c
        CS[n11 + i, n11 + n21 + n22 + r + i] = s if swap_sign else -s
        CS[p + n22 + i, n11 + i] = -s if swap_sign else s
    return CS

def _cossin_blocks(x11, x12, x21, x22, separate: Literal[bool], swap_sign: bool, compute_u: bool, compute_vh: bool):
    C: Literal[bool] = _any_cplx(x11, x12, x21, x22)
    T = complex if C else float
    res = _csd_core(x11, x12, x21, x22, T, swap_sign, compute_u, compute_vh)
    th = res[0]
    if separate:
        return (res[1], res[2]), th, (res[3], res[4])
    U = block_diag(res[1], res[2])
    VDH = block_diag(res[3], res[4])
    CS = _build_cs(th, res[5], res[6], res[7], swap_sign)
    return U, CS, VDH

def cossin(X, p=None, q=None, separate: Literal[bool] = False, swap_sign: bool = False,
           compute_u: bool = True, compute_vh: bool = True):
    if isinstance(p, NoneType) and isinstance(q, NoneType):
        if static.len(X) != 4:
            compile_error("When p and q are None, exactly four arrays should be in X")
        return _cossin_blocks(np.atleast_2d(X[0]), np.atleast_2d(X[1]), np.atleast_2d(X[2]), np.atleast_2d(X[3]),
                              separate, swap_sign, compute_u, compute_vh)
    else:
        pp = 1 if isinstance(p, NoneType) else int(p)
        qq = 1 if isinstance(q, NoneType) else int(q)
        Xa = np.asarray(X)
        if Xa.ndim != 2 or Xa.shape[0] != Xa.shape[1]:
            raise ValueError(f"Cosine Sine decomposition only supports square matrices, got {Xa.shape}")
        m = Xa.shape[0]
        if pp >= m or pp <= 0:
            raise ValueError(f"invalid p={pp}, 0<p<{m} must hold")
        if qq >= m or qq <= 0:
            raise ValueError(f"invalid q={qq}, 0<q<{m} must hold")
        x11 = Xa[:pp, :qq].copy()
        x12 = Xa[:pp, qq:].copy()
        x21 = Xa[pp:, :qq].copy()
        x22 = Xa[pp:, qq:].copy()
        return _cossin_blocks(x11, x12, x21, x22, separate, swap_sign, compute_u, compute_vh)
