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

# --- added by scipy-linalg-signal lane ---
@C
@_C.linkname("dgebal" + _suffix_lapack)
def dgebal_(job_: cobj, n_: cobj, a_: cobj, lda_: cobj, ilo_: cobj, ihi_: cobj, scl_: cobj, info_: cobj) -> None:
    pass

@C
@_C.linkname("zgebal" + _suffix_lapack)
def zgebal_(job_: cobj, n_: cobj, a_: cobj, lda_: cobj, ilo_: cobj, ihi_: cobj, scl_: cobj, info_: cobj) -> None:
    pass

@C
@_C.linkname("dgehrd" + _suffix_lapack)
def dgehrd_(n_: cobj, ilo_: cobj, ihi_: cobj, a_: cobj, lda_: cobj, tau_: cobj, work_: cobj, lwork_: cobj, info_: cobj) -> None:
    pass

@C
@_C.linkname("zgehrd" + _suffix_lapack)
def zgehrd_(n_: cobj, ilo_: cobj, ihi_: cobj, a_: cobj, lda_: cobj, tau_: cobj, work_: cobj, lwork_: cobj, info_: cobj) -> None:
    pass

@C
@_C.linkname("dorghr" + _suffix_lapack)
def dorghr_(n_: cobj, ilo_: cobj, ihi_: cobj, a_: cobj, lda_: cobj, tau_: cobj, work_: cobj, lwork_: cobj, info_: cobj) -> None:
    pass

@C
@_C.linkname("zunghr" + _suffix_lapack)
def zunghr_(n_: cobj, ilo_: cobj, ihi_: cobj, a_: cobj, lda_: cobj, tau_: cobj, work_: cobj, lwork_: cobj, info_: cobj) -> None:
    pass

@C
@_C.linkname("dtrsyl" + _suffix_lapack)
def dtrsyl_(trana_: cobj, tranb_: cobj, isgn_: cobj, m_: cobj, n_: cobj, a_: cobj, lda_: cobj, b_: cobj, ldb_: cobj, c_: cobj, ldc_: cobj, scale_: cobj, info_: cobj) -> None:
    pass

@C
@_C.linkname("ztrsyl" + _suffix_lapack)
def ztrsyl_(trana_: cobj, tranb_: cobj, isgn_: cobj, m_: cobj, n_: cobj, a_: cobj, lda_: cobj, b_: cobj, ldb_: cobj, c_: cobj, ldc_: cobj, scale_: cobj, info_: cobj) -> None:
    pass

@C
@_C.linkname("dtrsen" + _suffix_lapack)
def dtrsen_(job_: cobj, compq_: cobj, select_: cobj, n_: cobj, t_: cobj, ldt_: cobj, q_: cobj, ldq_: cobj, wr_: cobj, wi_: cobj, m_: cobj, s_: cobj, sep_: cobj, work_: cobj, lwork_: cobj, iwork_: cobj, liwork_: cobj, info_: cobj) -> None:
    pass

@C
@_C.linkname("ztrsen" + _suffix_lapack)
def ztrsen_(job_: cobj, compq_: cobj, select_: cobj, n_: cobj, t_: cobj, ldt_: cobj, q_: cobj, ldq_: cobj, w_: cobj, m_: cobj, s_: cobj, sep_: cobj, work_: cobj, lwork_: cobj, info_: cobj) -> None:
    pass

@C
@_C.linkname("dgges" + _suffix_lapack)
def dgges_(jobvsl_: cobj, jobvsr_: cobj, sort_: cobj, selctg_: cobj, n_: cobj, a_: cobj, lda_: cobj, b_: cobj, ldb_: cobj, sdim_: cobj, alphar_: cobj, alphai_: cobj, beta_: cobj, vsl_: cobj, ldvsl_: cobj, vsr_: cobj, ldvsr_: cobj, work_: cobj, lwork_: cobj, bwork_: cobj, info_: cobj) -> None:
    pass

@C
@_C.linkname("zgges" + _suffix_lapack)
def zgges_(jobvsl_: cobj, jobvsr_: cobj, sort_: cobj, selctg_: cobj, n_: cobj, a_: cobj, lda_: cobj, b_: cobj, ldb_: cobj, sdim_: cobj, alpha_: cobj, beta_: cobj, vsl_: cobj, ldvsl_: cobj, vsr_: cobj, ldvsr_: cobj, work_: cobj, lwork_: cobj, rwork_: cobj, bwork_: cobj, info_: cobj) -> None:
    pass

@C
@_C.linkname("dtgsen" + _suffix_lapack)
def dtgsen_(ijob_: cobj, wantq_: cobj, wantz_: cobj, select_: cobj, n_: cobj, a_: cobj, lda_: cobj, b_: cobj, ldb_: cobj, alphar_: cobj, alphai_: cobj, beta_: cobj, q_: cobj, ldq_: cobj, z_: cobj, ldz_: cobj, m_: cobj, pl_: cobj, pr_: cobj, dif_: cobj, work_: cobj, lwork_: cobj, iwork_: cobj, liwork_: cobj, info_: cobj) -> None:
    pass

@C
@_C.linkname("ztgsen" + _suffix_lapack)
def ztgsen_(ijob_: cobj, wantq_: cobj, wantz_: cobj, select_: cobj, n_: cobj, a_: cobj, lda_: cobj, b_: cobj, ldb_: cobj, alpha_: cobj, beta_: cobj, q_: cobj, ldq_: cobj, z_: cobj, ldz_: cobj, m_: cobj, pl_: cobj, pr_: cobj, dif_: cobj, work_: cobj, lwork_: cobj, iwork_: cobj, liwork_: cobj, info_: cobj) -> None:
    pass

@C
@_C.linkname("dsytrf" + _suffix_lapack)
def dsytrf_(uplo_: cobj, n_: cobj, a_: cobj, lda_: cobj, ipiv_: cobj, work_: cobj, lwork_: cobj, info_: cobj) -> None:
    pass

@C
@_C.linkname("zsytrf" + _suffix_lapack)
def zsytrf_(uplo_: cobj, n_: cobj, a_: cobj, lda_: cobj, ipiv_: cobj, work_: cobj, lwork_: cobj, info_: cobj) -> None:
    pass

@C
@_C.linkname("zhetrf" + _suffix_lapack)
def zhetrf_(uplo_: cobj, n_: cobj, a_: cobj, lda_: cobj, ipiv_: cobj, work_: cobj, lwork_: cobj, info_: cobj) -> None:
    pass

@C
@_C.linkname("dgtsv" + _suffix_lapack)
def dgtsv_(n_: cobj, nrhs_: cobj, dl_: cobj, d_: cobj, du_: cobj, b_: cobj, ldb_: cobj, info_: cobj) -> None:
    pass

@C
@_C.linkname("zgtsv" + _suffix_lapack)
def zgtsv_(n_: cobj, nrhs_: cobj, dl_: cobj, d_: cobj, du_: cobj, b_: cobj, ldb_: cobj, info_: cobj) -> None:
    pass

@C
@_C.linkname("dgbsv" + _suffix_lapack)
def dgbsv_(n_: cobj, kl_: cobj, ku_: cobj, nrhs_: cobj, ab_: cobj, ldab_: cobj, ipiv_: cobj, b_: cobj, ldb_: cobj, info_: cobj) -> None:
    pass

@C
@_C.linkname("zgbsv" + _suffix_lapack)
def zgbsv_(n_: cobj, kl_: cobj, ku_: cobj, nrhs_: cobj, ab_: cobj, ldab_: cobj, ipiv_: cobj, b_: cobj, ldb_: cobj, info_: cobj) -> None:
    pass

@C
@_C.linkname("dptsv" + _suffix_lapack)
def dptsv_(n_: cobj, nrhs_: cobj, d_: cobj, e_: cobj, b_: cobj, ldb_: cobj, info_: cobj) -> None:
    pass

@C
@_C.linkname("zptsv" + _suffix_lapack)
def zptsv_(n_: cobj, nrhs_: cobj, d_: cobj, e_: cobj, b_: cobj, ldb_: cobj, info_: cobj) -> None:
    pass

@C
@_C.linkname("dpbsv" + _suffix_lapack)
def dpbsv_(uplo_: cobj, n_: cobj, kd_: cobj, nrhs_: cobj, ab_: cobj, ldab_: cobj, b_: cobj, ldb_: cobj, info_: cobj) -> None:
    pass

@C
@_C.linkname("zpbsv" + _suffix_lapack)
def zpbsv_(uplo_: cobj, n_: cobj, kd_: cobj, nrhs_: cobj, ab_: cobj, ldab_: cobj, b_: cobj, ldb_: cobj, info_: cobj) -> None:
    pass

@C
@_C.linkname("dpbtrf" + _suffix_lapack)
def dpbtrf_(uplo_: cobj, n_: cobj, kd_: cobj, ab_: cobj, ldab_: cobj, info_: cobj) -> None:
    pass

@C
@_C.linkname("zpbtrf" + _suffix_lapack)
def zpbtrf_(uplo_: cobj, n_: cobj, kd_: cobj, ab_: cobj, ldab_: cobj, info_: cobj) -> None:
    pass

@C
@_C.linkname("dpbtrs" + _suffix_lapack)
def dpbtrs_(uplo_: cobj, n_: cobj, kd_: cobj, nrhs_: cobj, ab_: cobj, ldab_: cobj, b_: cobj, ldb_: cobj, info_: cobj) -> None:
    pass

@C
@_C.linkname("zpbtrs" + _suffix_lapack)
def zpbtrs_(uplo_: cobj, n_: cobj, kd_: cobj, nrhs_: cobj, ab_: cobj, ldab_: cobj, b_: cobj, ldb_: cobj, info_: cobj) -> None:
    pass

@C
@_C.linkname("dsbevd" + _suffix_lapack)
def dsbevd_(jobz_: cobj, uplo_: cobj, n_: cobj, kd_: cobj, ab_: cobj, ldab_: cobj, w_: cobj, z_: cobj, ldz_: cobj, work_: cobj, lwork_: cobj, iwork_: cobj, liwork_: cobj, info_: cobj) -> None:
    pass

@C
@_C.linkname("zhbevd" + _suffix_lapack)
def zhbevd_(jobz_: cobj, uplo_: cobj, n_: cobj, kd_: cobj, ab_: cobj, ldab_: cobj, w_: cobj, z_: cobj, ldz_: cobj, work_: cobj, lwork_: cobj, rwork_: cobj, lrwork_: cobj, iwork_: cobj, liwork_: cobj, info_: cobj) -> None:
    pass

@C
@_C.linkname("dsbevx" + _suffix_lapack)
def dsbevx_(jobz_: cobj, range_: cobj, uplo_: cobj, n_: cobj, kd_: cobj, ab_: cobj, ldab_: cobj, q_: cobj, ldq_: cobj, vl_: cobj, vu_: cobj, il_: cobj, iu_: cobj, abstol_: cobj, m_: cobj, w_: cobj, z_: cobj, ldz_: cobj, work_: cobj, iwork_: cobj, ifail_: cobj, info_: cobj) -> None:
    pass

@C
@_C.linkname("zhbevx" + _suffix_lapack)
def zhbevx_(jobz_: cobj, range_: cobj, uplo_: cobj, n_: cobj, kd_: cobj, ab_: cobj, ldab_: cobj, q_: cobj, ldq_: cobj, vl_: cobj, vu_: cobj, il_: cobj, iu_: cobj, abstol_: cobj, m_: cobj, w_: cobj, z_: cobj, ldz_: cobj, work_: cobj, rwork_: cobj, iwork_: cobj, ifail_: cobj, info_: cobj) -> None:
    pass

@C
@_C.linkname("dstevd" + _suffix_lapack)
def dstevd_(jobz_: cobj, n_: cobj, d_: cobj, e_: cobj, z_: cobj, ldz_: cobj, work_: cobj, lwork_: cobj, iwork_: cobj, liwork_: cobj, info_: cobj) -> None:
    pass

@C
@_C.linkname("dstev" + _suffix_lapack)
def dstev_(jobz_: cobj, n_: cobj, d_: cobj, e_: cobj, z_: cobj, ldz_: cobj, work_: cobj, info_: cobj) -> None:
    pass

@C
@_C.linkname("dsterf" + _suffix_lapack)
def dsterf_(n_: cobj, d_: cobj, e_: cobj, info_: cobj) -> None:
    pass

@C
@_C.linkname("dstebz" + _suffix_lapack)
def dstebz_(range_: cobj, order_: cobj, n_: cobj, vl_: cobj, vu_: cobj, il_: cobj, iu_: cobj, abstol_: cobj, d_: cobj, e_: cobj, m_: cobj, nsplit_: cobj, w_: cobj, iblock_: cobj, isplit_: cobj, work_: cobj, iwork_: cobj, info_: cobj) -> None:
    pass

@C
@_C.linkname("dstein" + _suffix_lapack)
def dstein_(n_: cobj, d_: cobj, e_: cobj, m_: cobj, w_: cobj, iblock_: cobj, isplit_: cobj, z_: cobj, ldz_: cobj, work_: cobj, iwork_: cobj, ifail_: cobj, info_: cobj) -> None:
    pass

@C
@_C.linkname("dstemr" + _suffix_lapack)
def dstemr_(jobz_: cobj, range_: cobj, n_: cobj, d_: cobj, e_: cobj, vl_: cobj, vu_: cobj, il_: cobj, iu_: cobj, m_: cobj, w_: cobj, z_: cobj, ldz_: cobj, nzc_: cobj, isuppz_: cobj, tryrac_: cobj, work_: cobj, lwork_: cobj, iwork_: cobj, liwork_: cobj, info_: cobj) -> None:
    pass

@C
@_C.linkname("dgerqf" + _suffix_lapack)
def dgerqf_(m_: cobj, n_: cobj, a_: cobj, lda_: cobj, tau_: cobj, work_: cobj, lwork_: cobj, info_: cobj) -> None:
    pass

@C
@_C.linkname("zgerqf" + _suffix_lapack)
def zgerqf_(m_: cobj, n_: cobj, a_: cobj, lda_: cobj, tau_: cobj, work_: cobj, lwork_: cobj, info_: cobj) -> None:
    pass

@C
@_C.linkname("dorgrq" + _suffix_lapack)
def dorgrq_(m_: cobj, n_: cobj, k_: cobj, a_: cobj, lda_: cobj, tau_: cobj, work_: cobj, lwork_: cobj, info_: cobj) -> None:
    pass

@C
@_C.linkname("zungrq" + _suffix_lapack)
def zungrq_(m_: cobj, n_: cobj, k_: cobj, a_: cobj, lda_: cobj, tau_: cobj, work_: cobj, lwork_: cobj, info_: cobj) -> None:
    pass

@C
@_C.linkname("dormqr" + _suffix_lapack)
def dormqr_(side_: cobj, trans_: cobj, m_: cobj, n_: cobj, k_: cobj, a_: cobj, lda_: cobj, tau_: cobj, c_: cobj, ldc_: cobj, work_: cobj, lwork_: cobj, info_: cobj) -> None:
    pass

@C
@_C.linkname("zunmqr" + _suffix_lapack)
def zunmqr_(side_: cobj, trans_: cobj, m_: cobj, n_: cobj, k_: cobj, a_: cobj, lda_: cobj, tau_: cobj, c_: cobj, ldc_: cobj, work_: cobj, lwork_: cobj, info_: cobj) -> None:
    pass

@C
@_C.linkname("dgelss" + _suffix_lapack)
def dgelss_(m_: cobj, n_: cobj, nrhs_: cobj, a_: cobj, lda_: cobj, b_: cobj, ldb_: cobj, s_: cobj, rcond_: cobj, rank_: cobj, work_: cobj, lwork_: cobj, info_: cobj) -> None:
    pass

@C
@_C.linkname("dgelsy" + _suffix_lapack)
def dgelsy_(m_: cobj, n_: cobj, nrhs_: cobj, a_: cobj, lda_: cobj, b_: cobj, ldb_: cobj, jpvt_: cobj, rcond_: cobj, rank_: cobj, work_: cobj, lwork_: cobj, info_: cobj) -> None:
    pass

@C
@_C.linkname("zgelsd" + _suffix_lapack)
def zgelsd_(m_: cobj, n_: cobj, nrhs_: cobj, a_: cobj, lda_: cobj, b_: cobj, ldb_: cobj, s_: cobj, rcond_: cobj, rank_: cobj, work_: cobj, lwork_: cobj, rwork_: cobj, iwork_: cobj, info_: cobj) -> None:
    pass

@C
@_C.linkname("dgecon" + _suffix_lapack)
def dgecon_(norm_: cobj, n_: cobj, a_: cobj, lda_: cobj, anorm_: cobj, rcond_: cobj, work_: cobj, iwork_: cobj, info_: cobj) -> None:
    pass


@C
@_C.linkname("dorcsd" + _suffix_lapack)
def dorcsd_(jobu1: cobj, jobu2: cobj, jobv1t: cobj, jobv2t: cobj, trans: cobj, signs: cobj, m: cobj, p: cobj, q: cobj, x11: cobj, ldx11: cobj, x12: cobj, ldx12: cobj, x21: cobj, ldx21: cobj, x22: cobj, ldx22: cobj, theta: cobj, u1: cobj, ldu1: cobj, u2: cobj, ldu2: cobj, v1t: cobj, ldv1t: cobj, v2t: cobj, ldv2t: cobj, work: cobj, lwork: cobj, iwork: cobj, info: cobj) -> None:
    pass

@C
@_C.linkname("zuncsd" + _suffix_lapack)
def zuncsd_(jobu1: cobj, jobu2: cobj, jobv1t: cobj, jobv2t: cobj, trans: cobj, signs: cobj, m: cobj, p: cobj, q: cobj, x11: cobj, ldx11: cobj, x12: cobj, ldx12: cobj, x21: cobj, ldx21: cobj, x22: cobj, ldx22: cobj, theta: cobj, u1: cobj, ldu1: cobj, u2: cobj, ldu2: cobj, v1t: cobj, ldv1t: cobj, v2t: cobj, ldv2t: cobj, work: cobj, lwork: cobj, rwork: cobj, lrwork: cobj, iwork: cobj, info: cobj) -> None:
    pass
