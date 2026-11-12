# Ported from scipy 1.18.1 (BSD-3-Clause)

# Raw LAPACK bindings used by scipy.linalg. Every argument is passed as an
# opaque pointer (Fortran calling convention), so the declarations stay
# compact. Symbol suffixes and integer width follow numpy.linalg.blas.

from numpy.linalg.blas import fint, _suffix_lapack

# Symmetric / Hermitian eigenproblems
@C
@_C.linkname("dsyevr" + _suffix_lapack)
def dsyevr_(jobz: cobj, rng: cobj, uplo: cobj, n: cobj, a: cobj, lda: cobj, vl: cobj, vu: cobj, il: cobj, iu: cobj, abstol: cobj, m: cobj, w: cobj, z: cobj, ldz: cobj, isuppz: cobj, work: cobj, lwork: cobj, iwork: cobj, liwork: cobj, info: cobj) -> None:
    pass

@C
@_C.linkname("zheevr" + _suffix_lapack)
def zheevr_(jobz: cobj, rng: cobj, uplo: cobj, n: cobj, a: cobj, lda: cobj, vl: cobj, vu: cobj, il: cobj, iu: cobj, abstol: cobj, m: cobj, w: cobj, z: cobj, ldz: cobj, isuppz: cobj, work: cobj, lwork: cobj, rwork: cobj, lrwork: cobj, iwork: cobj, liwork: cobj, info: cobj) -> None:
    pass

@C
@_C.linkname("dsyevx" + _suffix_lapack)
def dsyevx_(jobz: cobj, rng: cobj, uplo: cobj, n: cobj, a: cobj, lda: cobj, vl: cobj, vu: cobj, il: cobj, iu: cobj, abstol: cobj, m: cobj, w: cobj, z: cobj, ldz: cobj, work: cobj, lwork: cobj, iwork: cobj, ifail: cobj, info: cobj) -> None:
    pass

@C
@_C.linkname("zheevx" + _suffix_lapack)
def zheevx_(jobz: cobj, rng: cobj, uplo: cobj, n: cobj, a: cobj, lda: cobj, vl: cobj, vu: cobj, il: cobj, iu: cobj, abstol: cobj, m: cobj, w: cobj, z: cobj, ldz: cobj, work: cobj, lwork: cobj, rwork: cobj, iwork: cobj, ifail: cobj, info: cobj) -> None:
    pass

@C
@_C.linkname("dsyevd" + _suffix_lapack)
def dsyevd_(jobz: cobj, uplo: cobj, n: cobj, a: cobj, lda: cobj, w: cobj, work: cobj, lwork: cobj, iwork: cobj, liwork: cobj, info: cobj) -> None:
    pass

@C
@_C.linkname("zheevd" + _suffix_lapack)
def zheevd_(jobz: cobj, uplo: cobj, n: cobj, a: cobj, lda: cobj, w: cobj, work: cobj, lwork: cobj, rwork: cobj, lrwork: cobj, iwork: cobj, liwork: cobj, info: cobj) -> None:
    pass

@C
@_C.linkname("dsyev" + _suffix_lapack)
def dsyev_(jobz: cobj, uplo: cobj, n: cobj, a: cobj, lda: cobj, w: cobj, work: cobj, lwork: cobj, info: cobj) -> None:
    pass

@C
@_C.linkname("zheev" + _suffix_lapack)
def zheev_(jobz: cobj, uplo: cobj, n: cobj, a: cobj, lda: cobj, w: cobj, work: cobj, lwork: cobj, rwork: cobj, info: cobj) -> None:
    pass

@C
@_C.linkname("dsygvd" + _suffix_lapack)
def dsygvd_(itype: cobj, jobz: cobj, uplo: cobj, n: cobj, a: cobj, lda: cobj, b: cobj, ldb: cobj, w: cobj, work: cobj, lwork: cobj, iwork: cobj, liwork: cobj, info: cobj) -> None:
    pass

@C
@_C.linkname("zhegvd" + _suffix_lapack)
def zhegvd_(itype: cobj, jobz: cobj, uplo: cobj, n: cobj, a: cobj, lda: cobj, b: cobj, ldb: cobj, w: cobj, work: cobj, lwork: cobj, rwork: cobj, lrwork: cobj, iwork: cobj, liwork: cobj, info: cobj) -> None:
    pass

@C
@_C.linkname("dsygv" + _suffix_lapack)
def dsygv_(itype: cobj, jobz: cobj, uplo: cobj, n: cobj, a: cobj, lda: cobj, b: cobj, ldb: cobj, w: cobj, work: cobj, lwork: cobj, info: cobj) -> None:
    pass

@C
@_C.linkname("zhegv" + _suffix_lapack)
def zhegv_(itype: cobj, jobz: cobj, uplo: cobj, n: cobj, a: cobj, lda: cobj, b: cobj, ldb: cobj, w: cobj, work: cobj, lwork: cobj, rwork: cobj, info: cobj) -> None:
    pass

@C
@_C.linkname("dsygvx" + _suffix_lapack)
def dsygvx_(itype: cobj, jobz: cobj, rng: cobj, uplo: cobj, n: cobj, a: cobj, lda: cobj, b: cobj, ldb: cobj, vl: cobj, vu: cobj, il: cobj, iu: cobj, abstol: cobj, m: cobj, w: cobj, z: cobj, ldz: cobj, work: cobj, lwork: cobj, iwork: cobj, ifail: cobj, info: cobj) -> None:
    pass

@C
@_C.linkname("zhegvx" + _suffix_lapack)
def zhegvx_(itype: cobj, jobz: cobj, rng: cobj, uplo: cobj, n: cobj, a: cobj, lda: cobj, b: cobj, ldb: cobj, vl: cobj, vu: cobj, il: cobj, iu: cobj, abstol: cobj, m: cobj, w: cobj, z: cobj, ldz: cobj, work: cobj, lwork: cobj, rwork: cobj, iwork: cobj, ifail: cobj, info: cobj) -> None:
    pass

# LU / Cholesky / triangular
@C
@_C.linkname("dgetrf" + _suffix_lapack)
def dgetrf_(m: cobj, n: cobj, a: cobj, lda: cobj, ipiv: cobj, info: cobj) -> None:
    pass

@C
@_C.linkname("zgetrf" + _suffix_lapack)
def zgetrf_(m: cobj, n: cobj, a: cobj, lda: cobj, ipiv: cobj, info: cobj) -> None:
    pass

@C
@_C.linkname("dgetrs" + _suffix_lapack)
def dgetrs_(trans: cobj, n: cobj, nrhs: cobj, a: cobj, lda: cobj, ipiv: cobj, b: cobj, ldb: cobj, info: cobj) -> None:
    pass

@C
@_C.linkname("zgetrs" + _suffix_lapack)
def zgetrs_(trans: cobj, n: cobj, nrhs: cobj, a: cobj, lda: cobj, ipiv: cobj, b: cobj, ldb: cobj, info: cobj) -> None:
    pass

@C
@_C.linkname("dgetri" + _suffix_lapack)
def dgetri_(n: cobj, a: cobj, lda: cobj, ipiv: cobj, work: cobj, lwork: cobj, info: cobj) -> None:
    pass

@C
@_C.linkname("zgetri" + _suffix_lapack)
def zgetri_(n: cobj, a: cobj, lda: cobj, ipiv: cobj, work: cobj, lwork: cobj, info: cobj) -> None:
    pass

@C
@_C.linkname("dpotrf" + _suffix_lapack)
def dpotrf_(uplo: cobj, n: cobj, a: cobj, lda: cobj, info: cobj) -> None:
    pass

@C
@_C.linkname("zpotrf" + _suffix_lapack)
def zpotrf_(uplo: cobj, n: cobj, a: cobj, lda: cobj, info: cobj) -> None:
    pass

@C
@_C.linkname("dpotrs" + _suffix_lapack)
def dpotrs_(uplo: cobj, n: cobj, nrhs: cobj, a: cobj, lda: cobj, b: cobj, ldb: cobj, info: cobj) -> None:
    pass

@C
@_C.linkname("zpotrs" + _suffix_lapack)
def zpotrs_(uplo: cobj, n: cobj, nrhs: cobj, a: cobj, lda: cobj, b: cobj, ldb: cobj, info: cobj) -> None:
    pass

@C
@_C.linkname("dpotri" + _suffix_lapack)
def dpotri_(uplo: cobj, n: cobj, a: cobj, lda: cobj, info: cobj) -> None:
    pass

@C
@_C.linkname("zpotri" + _suffix_lapack)
def zpotri_(uplo: cobj, n: cobj, a: cobj, lda: cobj, info: cobj) -> None:
    pass

@C
@_C.linkname("dtrtrs" + _suffix_lapack)
def dtrtrs_(uplo: cobj, trans: cobj, diag: cobj, n: cobj, nrhs: cobj, a: cobj, lda: cobj, b: cobj, ldb: cobj, info: cobj) -> None:
    pass

@C
@_C.linkname("ztrtrs" + _suffix_lapack)
def ztrtrs_(uplo: cobj, trans: cobj, diag: cobj, n: cobj, nrhs: cobj, a: cobj, lda: cobj, b: cobj, ldb: cobj, info: cobj) -> None:
    pass

@C
@_C.linkname("dtrtri" + _suffix_lapack)
def dtrtri_(uplo: cobj, diag: cobj, n: cobj, a: cobj, lda: cobj, info: cobj) -> None:
    pass

@C
@_C.linkname("ztrtri" + _suffix_lapack)
def ztrtri_(uplo: cobj, diag: cobj, n: cobj, a: cobj, lda: cobj, info: cobj) -> None:
    pass

@C
@_C.linkname("dsysv" + _suffix_lapack)
def dsysv_(uplo: cobj, n: cobj, nrhs: cobj, a: cobj, lda: cobj, ipiv: cobj, b: cobj, ldb: cobj, work: cobj, lwork: cobj, info: cobj) -> None:
    pass

@C
@_C.linkname("zsysv" + _suffix_lapack)
def zsysv_(uplo: cobj, n: cobj, nrhs: cobj, a: cobj, lda: cobj, ipiv: cobj, b: cobj, ldb: cobj, work: cobj, lwork: cobj, info: cobj) -> None:
    pass

@C
@_C.linkname("zhesv" + _suffix_lapack)
def zhesv_(uplo: cobj, n: cobj, nrhs: cobj, a: cobj, lda: cobj, ipiv: cobj, b: cobj, ldb: cobj, work: cobj, lwork: cobj, info: cobj) -> None:
    pass

# Schur
@C
@_C.linkname("dgees" + _suffix_lapack)
def dgees_(jobvs: cobj, sort: cobj, select: cobj, n: cobj, a: cobj, lda: cobj, sdim: cobj, wr: cobj, wi: cobj, vs: cobj, ldvs: cobj, work: cobj, lwork: cobj, bwork: cobj, info: cobj) -> None:
    pass

@C
@_C.linkname("zgees" + _suffix_lapack)
def zgees_(jobvs: cobj, sort: cobj, select: cobj, n: cobj, a: cobj, lda: cobj, sdim: cobj, w: cobj, vs: cobj, ldvs: cobj, work: cobj, lwork: cobj, rwork: cobj, bwork: cobj, info: cobj) -> None:
    pass

# SVD / QR
@C
@_C.linkname("dgesdd" + _suffix_lapack)
def dgesdd_(jobz: cobj, m: cobj, n: cobj, a: cobj, lda: cobj, s: cobj, u: cobj, ldu: cobj, vt: cobj, ldvt: cobj, work: cobj, lwork: cobj, iwork: cobj, info: cobj) -> None:
    pass

@C
@_C.linkname("zgesdd" + _suffix_lapack)
def zgesdd_(jobz: cobj, m: cobj, n: cobj, a: cobj, lda: cobj, s: cobj, u: cobj, ldu: cobj, vt: cobj, ldvt: cobj, work: cobj, lwork: cobj, rwork: cobj, iwork: cobj, info: cobj) -> None:
    pass

@C
@_C.linkname("dgeqrf" + _suffix_lapack)
def dgeqrf_(m: cobj, n: cobj, a: cobj, lda: cobj, tau: cobj, work: cobj, lwork: cobj, info: cobj) -> None:
    pass

@C
@_C.linkname("zgeqrf" + _suffix_lapack)
def zgeqrf_(m: cobj, n: cobj, a: cobj, lda: cobj, tau: cobj, work: cobj, lwork: cobj, info: cobj) -> None:
    pass

@C
@_C.linkname("dorgqr" + _suffix_lapack)
def dorgqr_(m: cobj, n: cobj, k: cobj, a: cobj, lda: cobj, tau: cobj, work: cobj, lwork: cobj, info: cobj) -> None:
    pass

@C
@_C.linkname("zungqr" + _suffix_lapack)
def zungqr_(m: cobj, n: cobj, k: cobj, a: cobj, lda: cobj, tau: cobj, work: cobj, lwork: cobj, info: cobj) -> None:
    pass

@C
@_C.linkname("dgelsd" + _suffix_lapack)
def dgelsd_(m: cobj, n: cobj, nrhs: cobj, a: cobj, lda: cobj, b: cobj, ldb: cobj, s: cobj, rcond: cobj, rank: cobj, work: cobj, lwork: cobj, iwork: cobj, info: cobj) -> None:
    pass

# General eigenproblems / pivoted QR
# ?geev is declared in numpy.linalg.blas (re-declaring the same symbol
# with a different signature would clash at link time)
from numpy.linalg.blas import dgeev_, zgeev_

@C
@_C.linkname("dggev" + _suffix_lapack)
def dggev_(jobvl: cobj, jobvr: cobj, n: cobj, a: cobj, lda: cobj, b: cobj, ldb: cobj, alphar: cobj, alphai: cobj, beta: cobj, vl: cobj, ldvl: cobj, vr: cobj, ldvr: cobj, work: cobj, lwork: cobj, info: cobj) -> None:
    pass

@C
@_C.linkname("zggev" + _suffix_lapack)
def zggev_(jobvl: cobj, jobvr: cobj, n: cobj, a: cobj, lda: cobj, b: cobj, ldb: cobj, alpha: cobj, beta: cobj, vl: cobj, ldvl: cobj, vr: cobj, ldvr: cobj, work: cobj, lwork: cobj, rwork: cobj, info: cobj) -> None:
    pass

@C
@_C.linkname("dgeqp3" + _suffix_lapack)
def dgeqp3_(m: cobj, n: cobj, a: cobj, lda: cobj, jpvt: cobj, tau: cobj, work: cobj, lwork: cobj, info: cobj) -> None:
    pass

@C
@_C.linkname("zgeqp3" + _suffix_lapack)
def zgeqp3_(m: cobj, n: cobj, a: cobj, lda: cobj, jpvt: cobj, tau: cobj, work: cobj, lwork: cobj, rwork: cobj, info: cobj) -> None:
    pass
