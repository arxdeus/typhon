//! CBLAS subset (`cblas_*`) with ILP64 integers, on top of faer's matmul.
//!
//! Exported: `{s,d,c,z}copy`, `{s,d,c,z}gemm`, `{s,d,c,z}gemv`, `{s,d,c,z}syrk`,
//! `sdot`, `ddot`, `{c,z}dot{u,c}_sub`.
use crate::ffi::{Int, c32, c64, par};
use faer::traits::ComplexField;
use faer::{Accum, MatMut, MatRef};

pub const ROW_MAJOR: Int = 101;
pub const NO_TRANS: Int = 111;
pub const CONJ_TRANS: Int = 113;
pub const UPPER: Int = 121;

/// Scalar types handled by the BLAS layer.
pub trait Scalar: ComplexField + Copy + PartialEq + core::fmt::Debug {
    const IS_COMPLEX: bool;
    fn zero() -> Self;
    fn one() -> Self;
    fn cj(self) -> Self;
    fn smul(self, o: Self) -> Self;
    fn sadd(self, o: Self) -> Self;
}

macro_rules! real {
    ($t:ty) => {
        impl Scalar for $t {
            const IS_COMPLEX: bool = false;
            fn zero() -> Self { 0.0 }
            fn one() -> Self { 1.0 }
            fn cj(self) -> Self { self }
            fn smul(self, o: Self) -> Self { self * o }
            fn sadd(self, o: Self) -> Self { self + o }
        }
    };
}
macro_rules! cplx {
    ($t:ty) => {
        impl Scalar for $t {
            const IS_COMPLEX: bool = true;
            fn zero() -> Self { <$t>::new(0.0, 0.0) }
            fn one() -> Self { <$t>::new(1.0, 0.0) }
            fn cj(self) -> Self { self.conj() }
            fn smul(self, o: Self) -> Self { self * o }
            fn sadd(self, o: Self) -> Self { self + o }
        }
    };
}
real!(f32);
real!(f64);
cplx!(c32);
cplx!(c64);

/// View of a strided vector of length `n` (BLAS semantics for negative `inc`).
unsafe fn vec_ref<'a, T>(p: *const T, n: Int, inc: Int) -> MatRef<'a, T> {
    let n = n.max(0);
    if n == 0 || p.is_null() {
        return unsafe { MatRef::from_raw_parts(core::ptr::NonNull::dangling().as_ptr(), 0, 1, 1, 1) };
    }
    let start = if inc < 0 { unsafe { p.offset(((n - 1) * -inc) as isize) } } else { p };
    unsafe { MatRef::from_raw_parts(start, n as usize, 1, inc as isize, 1) }
}
unsafe fn vec_mut<'a, T>(p: *mut T, n: Int, inc: Int) -> MatMut<'a, T> {
    let n = n.max(0);
    if n == 0 || p.is_null() {
        return unsafe { MatMut::from_raw_parts_mut(core::ptr::NonNull::dangling().as_ptr(), 0, 1, 1, 1) };
    }
    let start = if inc < 0 { unsafe { p.offset(((n - 1) * -inc) as isize) } } else { p };
    unsafe { MatMut::from_raw_parts_mut(start, n as usize, 1, inc as isize, 1) }
}

/// Logical operand `op(S)` where `S` is a stored column-major matrix.
#[derive(Clone, Copy)]
struct Op<'a, T> {
    m: MatRef<'a, T>, // already transposed if needed
    conj: bool,
}

/// dst = beta*dst (beta==0 => exact zero, NaNs cleared).
fn scale<T: Scalar>(mut dst: MatMut<'_, T>, beta: T) {
    if beta == T::one() {
        return;
    }
    for j in 0..dst.ncols() {
        for i in 0..dst.nrows() {
            let v = if beta == T::zero() { T::zero() } else { beta.smul(dst[(i, j)]) };
            dst[(i, j)] = v;
        }
    }
}

/// dst = alpha * a * b + beta * dst.
fn gemm_core<T: Scalar>(dst: MatMut<'_, T>, a: Op<'_, T>, b: Op<'_, T>, alpha: T, beta: T) {
    let mut dst = dst;
    scale(dst.as_mut(), beta);
    if dst.nrows() == 0 || dst.ncols() == 0 || a.m.ncols() == 0 || alpha == T::zero() {
        return;
    }
    let p = par();
    use faer::linalg::matmul::matmul;
    match (a.conj, b.conj) {
        (false, false) => matmul(dst, Accum::Add, a.m, b.m, alpha, p),
        (true, false) => matmul(dst, Accum::Add, a.m.conjugate(), b.m, alpha, p),
        (false, true) => matmul(dst, Accum::Add, a.m, b.m.conjugate(), alpha, p),
        (true, true) => matmul(dst, Accum::Add, a.m.conjugate(), b.m.conjugate(), alpha, p),
    }
}

/// Column-major operand from (trans flag, ptr, stored rows, stored cols, ld).
unsafe fn op<'a, T: Scalar>(trans: Int, p: *const T, r: Int, c: Int, ld: Int) -> Op<'a, T> {
    let m = unsafe { crate::ffi::mat_ref(p, r, c, ld) };
    if trans == NO_TRANS {
        Op { m, conj: false }
    } else {
        Op { m: m.transpose(), conj: trans == CONJ_TRANS && T::IS_COMPLEX }
    }
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn gemm<T: Scalar>(
    order: Int, ta: Int, tb: Int, m: Int, n: Int, k: Int, alpha: T,
    a: *const T, lda: Int, b: *const T, ldb: Int, beta: T, c: *mut T, ldc: Int,
) {
    if order == ROW_MAJOR {
        return unsafe { gemm(102, tb, ta, n, m, k, alpha, b, ldb, a, lda, beta, c, ldc) };
    }
    if m <= 0 || n <= 0 {
        return;
    }
    let (ar, ac) = if ta == NO_TRANS { (m, k) } else { (k, m) };
    let (br, bc) = if tb == NO_TRANS { (k, n) } else { (n, k) };
    unsafe {
        let dst = crate::ffi::mat_mut(c, m, n, ldc);
        gemm_core(dst, op(ta, a, ar, ac, lda), op(tb, b, br, bc, ldb), alpha, beta);
    }
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn gemv<T: Scalar>(
    order: Int, trans: Int, m: Int, n: Int, alpha: T, a: *const T, lda: Int,
    x: *const T, incx: Int, beta: T, y: *mut T, incy: Int,
) {
    if m <= 0 || n <= 0 {
        // y still gets scaled by beta when its length is > 0.
    }
    // Stored column-major matrix S and the logical op on it.
    let (sr, sc, t, cj) = if order == ROW_MAJOR {
        // S = A^T (n x m)
        match trans {
            NO_TRANS => (n, m, true, false),
            CONJ_TRANS => (n, m, false, T::IS_COMPLEX),
            _ => (n, m, false, false),
        }
    } else {
        match trans {
            NO_TRANS => (m, n, false, false),
            CONJ_TRANS => (m, n, true, T::IS_COMPLEX),
            _ => (m, n, true, false),
        }
    };
    let s = unsafe { crate::ffi::mat_ref(a, sr, sc, lda) };
    let am = if t { s.transpose() } else { s };
    let (ylen, xlen) = (am.nrows() as Int, am.ncols() as Int);
    let (ylen, xlen) = if m <= 0 || n <= 0 {
        let lens = if trans == NO_TRANS { (m.max(0), n.max(0)) } else { (n.max(0), m.max(0)) };
        (lens.0, lens.1)
    } else {
        (ylen, xlen)
    };
    let yv = unsafe { vec_mut(y, ylen, incy) };
    if m <= 0 || n <= 0 {
        scale(yv, beta);
        return;
    }
    let xv = unsafe { vec_ref(x, xlen, incx) };
    gemm_core(yv, Op { m: am, conj: cj }, Op { m: xv, conj: false }, alpha, beta);
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn syrk<T: Scalar>(
    order: Int, uplo: Int, trans: Int, n: Int, k: Int, alpha: T, a: *const T, lda: Int,
    beta: T, c: *mut T, ldc: Int,
) {
    // Row-major: C stored transposed (flip triangle), A stored transposed (flip trans).
    let (upper, tr) = if order == ROW_MAJOR {
        (uplo != UPPER, trans == NO_TRANS)
    } else {
        (uplo == UPPER, trans != NO_TRANS)
    };
    if n <= 0 {
        return;
    }
    let s = unsafe { if tr { crate::ffi::mat_ref(a, k, n, lda) } else { crate::ffi::mat_ref(a, n, k, lda) } };
    let lhs = if tr { s.transpose() } else { s }; // n x k
    let mut tmp = faer::Mat::<T>::zeros(n as usize, n as usize);
    gemm_core(tmp.as_mut(), Op { m: lhs, conj: false }, Op { m: lhs.transpose(), conj: false }, alpha, T::zero());
    let mut cm = unsafe { crate::ffi::mat_mut(c, n, n, ldc) };
    for j in 0..n as usize {
        let (lo, hi) = if upper { (0, j + 1) } else { (j, n as usize) };
        for i in lo..hi {
            let old = if beta == T::zero() { T::zero() } else { beta.smul(cm[(i, j)]) };
            cm[(i, j)] = old.sadd(tmp[(i, j)]);
        }
    }
}

pub unsafe fn copy<T: Copy>(n: Int, x: *const T, incx: Int, y: *mut T, incy: Int) {
    if n <= 0 {
        return;
    }
    let xs = if incx < 0 { (n - 1) * -incx } else { 0 };
    let ys = if incy < 0 { (n - 1) * -incy } else { 0 };
    for i in 0..n {
        unsafe { *y.offset((ys + i * incy) as isize) = *x.offset((xs + i * incx) as isize) };
    }
}

pub unsafe fn dot<T: Scalar>(n: Int, x: *const T, incx: Int, y: *const T, incy: Int, conj: bool) -> T {
    let mut acc = T::zero();
    if n <= 0 {
        return acc;
    }
    let xs = if incx < 0 { (n - 1) * -incx } else { 0 };
    let ys = if incy < 0 { (n - 1) * -incy } else { 0 };
    for i in 0..n {
        let (a, b) = unsafe { (*x.offset((xs + i * incx) as isize), *y.offset((ys + i * incy) as isize)) };
        acc = acc.sadd(if conj { a.cj() } else { a }.smul(b));
    }
    acc
}

pub mod exports;
#[cfg(test)]
mod tests;
