//! `#[no_mangle]` CBLAS entry points (ILP64), matching `stdlib/numpy/linalg/blas.py`.
#![allow(clippy::missing_safety_doc, clippy::too_many_arguments)]
use super::{copy, dot, gemm, gemv, syrk};
use crate::ffi::{Int, c32, c64};

macro_rules! copy_fn {
    ($name:ident, $t:ty) => {
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $name(n: Int, x: *const $t, incx: Int, y: *mut $t, incy: Int) {
            unsafe { copy(n, x, incx, y, incy) }
        }
    };
}
copy_fn!(cblas_scopy, f32);
copy_fn!(cblas_dcopy, f64);
copy_fn!(cblas_ccopy, c32);
copy_fn!(cblas_zcopy, c64);

macro_rules! real_fns {
    ($gemm:ident, $gemv:ident, $syrk:ident, $dot:ident, $t:ty) => {
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $gemm(
            order: Int, ta: Int, tb: Int, m: Int, n: Int, k: Int, alpha: $t, a: *const $t, lda: Int,
            b: *const $t, ldb: Int, beta: $t, c: *mut $t, ldc: Int,
        ) {
            unsafe { gemm(order, ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc) }
        }
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $gemv(
            order: Int, trans: Int, m: Int, n: Int, alpha: $t, a: *const $t, lda: Int,
            x: *const $t, incx: Int, beta: $t, y: *mut $t, incy: Int,
        ) {
            unsafe { gemv(order, trans, m, n, alpha, a, lda, x, incx, beta, y, incy) }
        }
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $syrk(
            order: Int, uplo: Int, trans: Int, n: Int, k: Int, alpha: $t, a: *const $t, lda: Int,
            beta: $t, c: *mut $t, ldc: Int,
        ) {
            unsafe { syrk(order, uplo, trans, n, k, alpha, a, lda, beta, c, ldc) }
        }
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $dot(n: Int, x: *const $t, incx: Int, y: *const $t, incy: Int) -> $t {
            unsafe { dot(n, x, incx, y, incy, false) }
        }
    };
}
real_fns!(cblas_sgemm, cblas_sgemv, cblas_ssyrk, cblas_sdot, f32);
real_fns!(cblas_dgemm, cblas_dgemv, cblas_dsyrk, cblas_ddot, f64);

macro_rules! cplx_fns {
    ($gemm:ident, $gemv:ident, $syrk:ident, $dotu:ident, $dotc:ident, $t:ty) => {
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $gemm(
            order: Int, ta: Int, tb: Int, m: Int, n: Int, k: Int, alpha: *const $t, a: *const $t,
            lda: Int, b: *const $t, ldb: Int, beta: *const $t, c: *mut $t, ldc: Int,
        ) {
            unsafe { gemm(order, ta, tb, m, n, k, *alpha, a, lda, b, ldb, *beta, c, ldc) }
        }
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $gemv(
            order: Int, trans: Int, m: Int, n: Int, alpha: *const $t, a: *const $t, lda: Int,
            x: *const $t, incx: Int, beta: *const $t, y: *mut $t, incy: Int,
        ) {
            unsafe { gemv(order, trans, m, n, *alpha, a, lda, x, incx, *beta, y, incy) }
        }
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $syrk(
            order: Int, uplo: Int, trans: Int, n: Int, k: Int, alpha: *const $t, a: *const $t,
            lda: Int, beta: *const $t, c: *mut $t, ldc: Int,
        ) {
            unsafe { syrk(order, uplo, trans, n, k, *alpha, a, lda, *beta, c, ldc) }
        }
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $dotu(n: Int, x: *const $t, incx: Int, y: *const $t, incy: Int, out: *mut $t) {
            unsafe { *out = dot(n, x, incx, y, incy, false) }
        }
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $dotc(n: Int, x: *const $t, incx: Int, y: *const $t, incy: Int, out: *mut $t) {
            unsafe { *out = dot(n, x, incx, y, incy, true) }
        }
    };
}
cplx_fns!(cblas_cgemm, cblas_cgemv, cblas_csyrk, cblas_cdotu_sub, cblas_cdotc_sub, c32);
cplx_fns!(cblas_zgemm, cblas_zgemv, cblas_zsyrk, cblas_zdotu_sub, cblas_zdotc_sub, c64);
