//! `xPOTRF` (Cholesky).
use super::Sc;
use crate::ffi::{char_arg, mat_mut, par, rd, Int};
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::cholesky::llt::factor::{cholesky_in_place, cholesky_in_place_scratch, LltError};
use faer::{c32, c64};

pub(crate) unsafe fn potrf<T: Sc>(uplo: u8, n: Int, a: *mut T, lda: Int) -> Int {
    if uplo != b'U' && uplo != b'L' {
        return -1;
    }
    if n < 0 {
        return -2;
    }
    if lda < n.max(1) {
        return -4;
    }
    let nu = n as usize;
    if nu == 0 {
        return 0;
    }
    let ld = lda as usize;
    let at = |i: usize, j: usize| unsafe { a.add(i + j * ld) };
    // Work on a lower-triangular copy (for 'U', L = U^H).
    let mut w = faer::Mat::<T>::from_fn(nu, nu, |i, j| {
        if i < j {
            T::zero()
        } else if uplo == b'L' {
            unsafe { *at(i, j) }
        } else {
            unsafe { (*at(j, i)).cj() }
        }
    });
    let p = par();
    let mut buf = MemBuffer::new(cholesky_in_place_scratch::<T>(nu, p, Default::default()));
    let r = cholesky_in_place(w.as_mut(), Default::default(), p, MemStack::new(&mut buf), Default::default());
    let info = match r {
        Ok(_) => 0,
        Err(LltError::NonPositivePivot { index }) => (index + 1) as Int,
    };
    // Write back the computed columns (on failure LAPACK leaves a partial factor; numpy
    // discards the result anyway).
    let lim = if info == 0 { nu } else { (info - 1) as usize };
    for j in 0..lim {
        for i in j..nu {
            let v = w[(i, j)];
            unsafe {
                if uplo == b'L' {
                    *at(i, j) = v;
                } else {
                    *at(j, i) = v.cj();
                }
            }
        }
    }
    let _ = mat_mut::<T>;
    info
}

macro_rules! export_potrf {
    ($t:ty, $name:ident) => {
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $name(uplo: *const u8, n: *const Int, a: *mut $t, lda: *const Int, info: *mut Int) {
            unsafe { *info = potrf(char_arg(uplo), rd(n), a, rd(lda)) };
        }
    };
}
export_potrf!(f32, spotrf_);
export_potrf!(f64, dpotrf_);
export_potrf!(c32, cpotrf_);
export_potrf!(c64, zpotrf_);
