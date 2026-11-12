//! Shared FFI helpers for the BLAS and LAPACK exports.
//!
//! * Integers are ILP64: `Int = i64`.
//! * Complex numbers are `faer::c32` / `faer::c64` (= `num_complex::Complex<f32/f64>`,
//!   `#[repr(C)]` `{re, im}`, layout-identical to `[f32;2]` / `[f64;2]` and to the
//!   Typhon `complex64` / `complex` types).
//! * Matrices are column-major `(ptr, rows, cols, ld)`; use [`mat_ref`] / [`mat_mut`].
use faer::{MatMut, MatRef, Par};

pub use faer::{c32, c64};

/// BLAS/LAPACK integer (ILP64).
pub type Int = i64;

/// Column-major read-only view of `rows x cols` at `ptr` with leading dimension `ld`.
///
/// # Safety
/// `ptr` must be valid for that region (may be dangling/null if rows or cols is 0).
#[inline]
pub unsafe fn mat_ref<'a, T>(ptr: *const T, rows: Int, cols: Int, ld: Int) -> MatRef<'a, T> {
    let (r, c) = (rows.max(0) as usize, cols.max(0) as usize);
    let p = if ptr.is_null() || r == 0 || c == 0 { core::ptr::NonNull::<T>::dangling().as_ptr() as *const T } else { ptr };
    unsafe { MatRef::from_raw_parts(p, r, c, 1, ld.max(1) as isize) }
}

/// Column-major mutable view, see [`mat_ref`].
///
/// # Safety
/// As for [`mat_ref`], plus exclusive access.
#[inline]
pub unsafe fn mat_mut<'a, T>(ptr: *mut T, rows: Int, cols: Int, ld: Int) -> MatMut<'a, T> {
    let (r, c) = (rows.max(0) as usize, cols.max(0) as usize);
    let p = if ptr.is_null() || r == 0 || c == 0 { core::ptr::NonNull::<T>::dangling().as_ptr() } else { ptr };
    unsafe { MatMut::from_raw_parts_mut(p, r, c, 1, ld.max(1) as isize) }
}

/// Read a Fortran `CHARACTER*1` argument (pointer to one byte), upper-cased.
///
/// # Safety
/// `p` must point to at least one readable byte.
#[inline]
pub unsafe fn char_arg(p: *const u8) -> u8 {
    if p.is_null() { b' ' } else { unsafe { *p }.to_ascii_uppercase() }
}

/// Read an `Int` passed by reference (Fortran style).
///
/// # Safety
/// `p` must be valid.
#[inline]
pub unsafe fn rd(p: *const Int) -> Int {
    unsafe { *p }
}

/// Parallelism for faer kernels. Sequential unless `TYPHON_LINALG_THREADS=n>1`
/// and the `rayon` feature is enabled (avoids oversubscription with OpenMP `@par`).
pub fn par() -> Par {
    #[cfg(feature = "rayon")]
    {
        use std::sync::OnceLock;
        static N: OnceLock<usize> = OnceLock::new();
        let n = *N.get_or_init(|| {
            std::env::var("TYPHON_LINALG_THREADS").ok().and_then(|s| s.parse().ok()).unwrap_or(1)
        });
        if n > 1 {
            return Par::rayon(n);
        }
    }
    Par::Seq
}
