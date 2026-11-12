//! Fortran-ABI LAPACK subset (`{s,d,c,z}...._`), ILP64. Owned by lane linalg-lapack.
//!
//! Conventions: every argument is passed by pointer, integers are `i64`, matrices are
//! column-major with a leading dimension, char options are single bytes (no hidden
//! string-length arguments: `stdlib/numpy/linalg/blas.py` declares none).
//! Only the option combinations used by `stdlib/numpy/linalg/linalg.py` are supported,
//! others return `info = -k` for the offending argument.
#![allow(clippy::missing_safety_doc, clippy::too_many_arguments)]

use core::ops::{Add, Div, Mul, Neg, Sub};
use faer::traits::{ComplexField, RealField};
use faer::{c32, c64};

pub mod lu;
pub mod chol;
pub mod qr;
pub mod eigh;
pub mod geev;
pub mod svd;
pub mod lstsq;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests2;

/// Real scalar helper (f32 / f64).
pub trait Re: RealField + Copy + PartialOrd + Default {
    fn f(x: f64) -> Self;
    fn g(self) -> f64;
}
impl Re for f32 {
    fn f(x: f64) -> Self { x as f32 }
    fn g(self) -> f64 { self as f64 }
}
impl Re for f64 {
    fn f(x: f64) -> Self { x }
    fn g(self) -> f64 { self }
}

/// Scalar helper over f32 / f64 / c32 / c64.
pub trait Sc:
    ComplexField<Real: Re>
    + Copy
    + Default
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
{
    const CPLX: bool;
    fn re(self) -> f64;
    fn im(self) -> f64;
    fn mk(re: f64, im: f64) -> Self;
    #[inline]
    fn cj(self) -> Self { Self::mk(self.re(), -self.im()) }
    #[inline]
    fn ab(self) -> f64 { self.re().hypot(self.im()) }
    #[inline]
    fn ab2(self) -> f64 { self.re() * self.re() + self.im() * self.im() }
    #[inline]
    fn r(x: f64) -> Self { Self::mk(x, 0.0) }
    #[inline]
    fn zero() -> Self { Self::mk(0.0, 0.0) }
}
impl Sc for f32 {
    const CPLX: bool = false;
    fn re(self) -> f64 { self as f64 }
    fn im(self) -> f64 { 0.0 }
    fn mk(re: f64, _: f64) -> Self { re as f32 }
}
impl Sc for f64 {
    const CPLX: bool = false;
    fn re(self) -> f64 { self }
    fn im(self) -> f64 { 0.0 }
    fn mk(re: f64, _: f64) -> Self { re }
}
impl Sc for c32 {
    const CPLX: bool = true;
    fn re(self) -> f64 { self.re as f64 }
    fn im(self) -> f64 { self.im as f64 }
    fn mk(re: f64, im: f64) -> Self { c32::new(re as f32, im as f32) }
}
impl Sc for c64 {
    const CPLX: bool = true;
    fn re(self) -> f64 { self.re }
    fn im(self) -> f64 { self.im }
    fn mk(re: f64, im: f64) -> Self { c64::new(re, im) }
}

/// Machine epsilon of the real type (LAPACK `xLAMCH('E')`-like, relative spacing).
pub fn eps_of<T: Sc>() -> f64 {
    if core::mem::size_of::<T::Real>() == 4 { f32::EPSILON as f64 } else { f64::EPSILON }
}

/// Write a workspace size into `work[0]` (real part) if non-null.
#[inline]
pub(crate) unsafe fn set_work<T: Sc>(work: *mut T, n: i64) {
    if !work.is_null() {
        unsafe { *work = T::r(n.max(1) as f64) };
    }
}

/// Write a real-workspace size if non-null.
#[inline]
pub(crate) unsafe fn set_rwork<R: Re>(rwork: *mut R, n: i64) {
    if !rwork.is_null() {
        unsafe { *rwork = R::f(n.max(1) as f64) };
    }
}

/// Write an integer-workspace size if non-null.
#[inline]
pub(crate) unsafe fn set_iwork(iwork: *mut i64, n: i64) {
    if !iwork.is_null() {
        unsafe { *iwork = n.max(1) };
    }
}

/// Copy a column-major `rows x cols` region into an owned faer matrix.
pub(crate) unsafe fn load<T: Sc>(p: *const T, rows: usize, cols: usize, ld: usize) -> faer::Mat<T> {
    faer::Mat::from_fn(rows, cols, |i, j| unsafe { *p.add(i + j * ld) })
}
