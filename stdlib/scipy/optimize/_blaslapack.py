# Ported from scipy 1.18.1 (BSD-3-Clause)

# Fortran BLAS/LAPACK entry points used by the SLSQP and NNLS ports
# (scipy/optimize/src/blaslapack_declarations.h). All arguments are
# passed by reference as in Fortran.

from numpy.linalg.blas import fint, _suffix_lapack

@C
@_C.linkname("dgerq2" + _suffix_lapack)
def dgerq2_(m: cobj, n: cobj, a: cobj, lda: cobj, tau: cobj, work: cobj, info: cobj) -> None:
    pass

@C
@_C.linkname("dormr2" + _suffix_lapack)
def dormr2_(side: cobj, trans: cobj, m: cobj, n: cobj, k: cobj, a: cobj, lda: cobj, tau: cobj, c: cobj, ldc: cobj, work: cobj, info: cobj) -> None:
    pass

@C
@_C.linkname("dgeqr2" + _suffix_lapack)
def dgeqr2_(m: cobj, n: cobj, a: cobj, lda: cobj, tau: cobj, work: cobj, info: cobj) -> None:
    pass

@C
@_C.linkname("dorm2r" + _suffix_lapack)
def dorm2r_(side: cobj, trans: cobj, m: cobj, n: cobj, k: cobj, a: cobj, lda: cobj, tau: cobj, c: cobj, ldc: cobj, work: cobj, info: cobj) -> None:
    pass

@C
@_C.linkname("dgelsy" + _suffix_lapack)
def dgelsy_(m: cobj, n: cobj, nrhs: cobj, a: cobj, lda: cobj, b: cobj, ldb: cobj, jpvt: cobj, rcond: cobj, rank: cobj, work: cobj, lwork: cobj, info: cobj) -> None:
    pass

@C
@_C.linkname("dtrsm" + _suffix_lapack)
def dtrsm_(side: cobj, uplo: cobj, transa: cobj, diag: cobj, m: cobj, n: cobj, alpha: cobj, a: cobj, lda: cobj, b: cobj, ldb: cobj) -> None:
    pass

@C
@_C.linkname("dtrsv" + _suffix_lapack)
def dtrsv_(uplo: cobj, trans: cobj, diag: cobj, n: cobj, a: cobj, lda: cobj, x: cobj, incx: cobj) -> None:
    pass

@C
@_C.linkname("dgemv" + _suffix_lapack)
def dgemv_(trans: cobj, m: cobj, n: cobj, alpha: cobj, a: cobj, lda: cobj, x: cobj, incx: cobj, beta: cobj, y: cobj, incy: cobj) -> None:
    pass

@C
@_C.linkname("dtpmv" + _suffix_lapack)
def dtpmv_(uplo: cobj, trans: cobj, diag: cobj, n: cobj, ap: cobj, x: cobj, incx: cobj) -> None:
    pass

@C
@_C.linkname("dtpsv" + _suffix_lapack)
def dtpsv_(uplo: cobj, trans: cobj, diag: cobj, n: cobj, ap: cobj, x: cobj, incx: cobj) -> None:
    pass

@C
@_C.linkname("ddot" + _suffix_lapack)
def ddot_(n: cobj, x: cobj, incx: cobj, y: cobj, incy: cobj) -> float:
    pass

@C
@_C.linkname("dnrm2" + _suffix_lapack)
def dnrm2_(n: cobj, x: cobj, incx: cobj) -> float:
    pass

@C
@_C.linkname("dlarfgp" + _suffix_lapack)
def dlarfgp_(n: cobj, alpha: cobj, x: cobj, incx: cobj, tau: cobj) -> None:
    pass

@C
@_C.linkname("dlarf" + _suffix_lapack)
def dlarf_(side: cobj, m: cobj, n: cobj, v: cobj, incv: cobj, tau: cobj, c: cobj, ldc: cobj, work: cobj) -> None:
    pass

@C
@_C.linkname("dlartgp" + _suffix_lapack)
def dlartgp_(f: cobj, g: cobj, cs: cobj, sn: cobj, r: cobj) -> None:
    pass

def _c(s: str) -> u8:
    return u8(ord(s))

def _I(v: int) -> fint:
    return fint(v)

# thin typed wrappers (pointers into float buffers)

def gemv(tr: str, m: int, n: int, alpha: float, a: Ptr[float], lda: int,
         x: Ptr[float], beta: float, y: Ptr[float]):
    t = _c(tr); M = _I(m); N = _I(n); L = _I(max(lda, 1)); one = _I(1)
    al = alpha; be = beta
    dgemv_(__ptr__(t).as_byte(), __ptr__(M).as_byte(), __ptr__(N).as_byte(),
           __ptr__(al).as_byte(), a.as_byte(), __ptr__(L).as_byte(),
           x.as_byte(), __ptr__(one).as_byte(), __ptr__(be).as_byte(),
           y.as_byte(), __ptr__(one).as_byte())

def tpmv(uplo: str, tr: str, diag: str, n: int, ap: Ptr[float], x: Ptr[float]):
    u = _c(uplo); t = _c(tr); d = _c(diag); N = _I(n); one = _I(1)
    dtpmv_(__ptr__(u).as_byte(), __ptr__(t).as_byte(), __ptr__(d).as_byte(),
           __ptr__(N).as_byte(), ap.as_byte(), x.as_byte(), __ptr__(one).as_byte())

def tpsv(uplo: str, tr: str, diag: str, n: int, ap: Ptr[float], x: Ptr[float]):
    u = _c(uplo); t = _c(tr); d = _c(diag); N = _I(n); one = _I(1)
    dtpsv_(__ptr__(u).as_byte(), __ptr__(t).as_byte(), __ptr__(d).as_byte(),
           __ptr__(N).as_byte(), ap.as_byte(), x.as_byte(), __ptr__(one).as_byte())

def trsv(uplo: str, tr: str, diag: str, n: int, a: Ptr[float], lda: int, x: Ptr[float]):
    u = _c(uplo); t = _c(tr); d = _c(diag); N = _I(n); L = _I(max(lda, 1)); one = _I(1)
    dtrsv_(__ptr__(u).as_byte(), __ptr__(t).as_byte(), __ptr__(d).as_byte(),
           __ptr__(N).as_byte(), a.as_byte(), __ptr__(L).as_byte(), x.as_byte(),
           __ptr__(one).as_byte())

def trsm(side: str, uplo: str, tr: str, diag: str, m: int, n: int, alpha: float,
         a: Ptr[float], lda: int, b: Ptr[float], ldb: int):
    s = _c(side); u = _c(uplo); t = _c(tr); d = _c(diag)
    M = _I(m); N = _I(n); LA = _I(max(lda, 1)); LB = _I(max(ldb, 1)); al = alpha
    dtrsm_(__ptr__(s).as_byte(), __ptr__(u).as_byte(), __ptr__(t).as_byte(),
           __ptr__(d).as_byte(), __ptr__(M).as_byte(), __ptr__(N).as_byte(),
           __ptr__(al).as_byte(), a.as_byte(), __ptr__(LA).as_byte(),
           b.as_byte(), __ptr__(LB).as_byte())

def dot(n: int, x: Ptr[float], y: Ptr[float]) -> float:
    N = _I(n); one = _I(1)
    return ddot_(__ptr__(N).as_byte(), x.as_byte(), __ptr__(one).as_byte(),
                 y.as_byte(), __ptr__(one).as_byte())

def nrm2(n: int, x: Ptr[float]) -> float:
    if n <= 0:
        return 0.0
    N = _I(n); one = _I(1)
    return dnrm2_(__ptr__(N).as_byte(), x.as_byte(), __ptr__(one).as_byte())

def gerq2(m: int, n: int, a: Ptr[float], lda: int, tau: Ptr[float], work: Ptr[float]) -> int:
    M = _I(m); N = _I(n); L = _I(max(lda, 1)); info = _I(0)
    dgerq2_(__ptr__(M).as_byte(), __ptr__(N).as_byte(), a.as_byte(),
            __ptr__(L).as_byte(), tau.as_byte(), work.as_byte(), __ptr__(info).as_byte())
    return int(info)

def ormr2(side: str, tr: str, m: int, n: int, k: int, a: Ptr[float], lda: int,
          tau: Ptr[float], c: Ptr[float], ldc: int, work: Ptr[float]) -> int:
    s = _c(side); t = _c(tr); M = _I(m); N = _I(n); K = _I(k)
    LA = _I(max(lda, 1)); LC = _I(max(ldc, 1)); info = _I(0)
    dormr2_(__ptr__(s).as_byte(), __ptr__(t).as_byte(), __ptr__(M).as_byte(),
            __ptr__(N).as_byte(), __ptr__(K).as_byte(), a.as_byte(),
            __ptr__(LA).as_byte(), tau.as_byte(), c.as_byte(),
            __ptr__(LC).as_byte(), work.as_byte(), __ptr__(info).as_byte())
    return int(info)

def geqr2(m: int, n: int, a: Ptr[float], lda: int, tau: Ptr[float], work: Ptr[float]) -> int:
    M = _I(m); N = _I(n); L = _I(max(lda, 1)); info = _I(0)
    dgeqr2_(__ptr__(M).as_byte(), __ptr__(N).as_byte(), a.as_byte(),
            __ptr__(L).as_byte(), tau.as_byte(), work.as_byte(), __ptr__(info).as_byte())
    return int(info)

def orm2r(side: str, tr: str, m: int, n: int, k: int, a: Ptr[float], lda: int,
          tau: Ptr[float], c: Ptr[float], ldc: int, work: Ptr[float]) -> int:
    s = _c(side); t = _c(tr); M = _I(m); N = _I(n); K = _I(k)
    LA = _I(max(lda, 1)); LC = _I(max(ldc, 1)); info = _I(0)
    dorm2r_(__ptr__(s).as_byte(), __ptr__(t).as_byte(), __ptr__(M).as_byte(),
            __ptr__(N).as_byte(), __ptr__(K).as_byte(), a.as_byte(),
            __ptr__(LA).as_byte(), tau.as_byte(), c.as_byte(),
            __ptr__(LC).as_byte(), work.as_byte(), __ptr__(info).as_byte())
    return int(info)

def gelsy(m: int, n: int, a: Ptr[float], lda: int, b: Ptr[float], ldb: int,
          jpvt: Ptr[fint], rcond: float, work: Ptr[float], lwork: int) -> int:
    # returns rank
    M = _I(m); N = _I(n); one = _I(1); LA = _I(max(lda, 1)); LB = _I(max(ldb, 1))
    rc = rcond; rank = _I(0); LW = _I(lwork); info = _I(0)
    dgelsy_(__ptr__(M).as_byte(), __ptr__(N).as_byte(), __ptr__(one).as_byte(),
            a.as_byte(), __ptr__(LA).as_byte(), b.as_byte(), __ptr__(LB).as_byte(),
            jpvt.as_byte(), __ptr__(rc).as_byte(), __ptr__(rank).as_byte(),
            work.as_byte(), __ptr__(LW).as_byte(), __ptr__(info).as_byte())
    return int(rank)

def larfgp(n: int, alpha: Ptr[float], x: Ptr[float]) -> float:
    N = _I(n); one = _I(1); tau = 0.0
    dlarfgp_(__ptr__(N).as_byte(), alpha.as_byte(), x.as_byte(),
             __ptr__(one).as_byte(), __ptr__(tau).as_byte())
    return tau

def larf_l(m: int, v: Ptr[float], tau: float, c: Ptr[float], ldc: int, work: Ptr[float]):
    s = _c('L'); M = _I(m); one = _I(1); t = tau; L = _I(max(ldc, 1))
    dlarf_(__ptr__(s).as_byte(), __ptr__(M).as_byte(), __ptr__(one).as_byte(),
           v.as_byte(), __ptr__(one).as_byte(), __ptr__(t).as_byte(), c.as_byte(),
           __ptr__(L).as_byte(), work.as_byte())

def lartgp(f: Ptr[float], g: Ptr[float], r: Ptr[float]):
    cs = 0.0; sn = 0.0
    ff = f[0]; gg = g[0]
    dlartgp_(__ptr__(ff).as_byte(), __ptr__(gg).as_byte(), __ptr__(cs).as_byte(),
             __ptr__(sn).as_byte(), r.as_byte())
    return cs, sn
