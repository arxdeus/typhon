//! `xSYEVD` / `xHEEVD`: symmetric/Hermitian eigen-decomposition (ascending eigenvalues,
//! orthonormal eigenvectors overwrite A for jobz='V').
use super::{load, set_iwork, set_rwork, set_work, Re, Sc};
use crate::ffi::{char_arg, par, rd, Int};
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::evd::{self_adjoint_evd, self_adjoint_evd_scratch, ComputeEigenvectors};
use faer::diag::Diag;
use faer::{c32, c64, Mat};

/// Core; returns info. `w` receives n real eigenvalues.
pub(crate) unsafe fn heevd<T: Sc>(jobz: u8, uplo: u8, n: Int, a: *mut T, lda: Int, w: *mut T::Real) -> Int {
    let vec = match jobz {
        b'V' => true,
        b'N' => false,
        _ => return -1,
    };
    if uplo != b'U' && uplo != b'L' {
        return -2;
    }
    if n < 0 {
        return -3;
    }
    if lda < n.max(1) {
        return -5;
    }
    let nu = n as usize;
    if nu == 0 {
        return 0;
    }
    let ld = lda as usize;
    let src = unsafe { load(a as *const T, nu, nu, ld) };
    // Full Hermitian matrix from the referenced triangle (diagonal imaginary part ignored).
    let full = Mat::<T>::from_fn(nu, nu, |i, j| {
        let lower = uplo == b'L';
        if i == j {
            T::r(src[(i, i)].re())
        } else if (i > j) == lower {
            src[(i, j)]
        } else {
            src[(j, i)].cj()
        }
    });
    let p = par();
    let cu = if vec { ComputeEigenvectors::Yes } else { ComputeEigenvectors::No };
    let mut s = Diag::<T>::zeros(nu);
    let mut u = if vec { Mat::<T>::zeros(nu, nu) } else { Mat::<T>::zeros(0, 0) };
    let mut buf = MemBuffer::new(self_adjoint_evd_scratch::<T>(nu, cu, p, Default::default()));
    let r = self_adjoint_evd(
        full.as_ref(),
        s.as_mut(),
        if vec { Some(u.as_mut()) } else { None },
        p,
        MemStack::new(&mut buf),
        Default::default(),
    );
    if r.is_err() {
        return n.max(1);
    }
    // faer sorts ascending; enforce just in case.
    let mut idx: Vec<usize> = (0..nu).collect();
    let sv: Vec<f64> = (0..nu).map(|i| s[i].re()).collect();
    idx.sort_by(|&x, &y| sv[x].partial_cmp(&sv[y]).unwrap_or(core::cmp::Ordering::Equal));
    for (k, &i) in idx.iter().enumerate() {
        unsafe { *w.add(k) = <T::Real as Re>::f(sv[i]) };
        if vec {
            for r in 0..nu {
                unsafe { *a.add(r + k * ld) = u[(r, i)] };
            }
        }
    }
    0
}

unsafe fn syevd_c<T: Sc>(
    jobz: u8, uplo: u8, n: Int, a: *mut T, lda: Int, w: *mut T::Real,
    work: *mut T, lwork: Int, rwork: *mut T::Real, lrwork: Int, iwork: *mut Int, liwork: Int,
) -> Int {
    let nn = n.max(1);
    unsafe {
        set_work(work, nn);
        set_rwork(rwork, nn);
        set_iwork(iwork, 1);
    }
    if lwork == -1 || lrwork == -1 || liwork == -1 {
        if jobz != b'V' && jobz != b'N' {
            return -1;
        }
        return 0;
    }
    unsafe { heevd(jobz, uplo, n, a, lda, w) }
}

macro_rules! export_syevd {
    ($t:ty, $r:ty, $name:ident) => {
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $name(
            jobz: *const u8, uplo: *const u8, n: *const Int, a: *mut $t, lda: *const Int, w: *mut $r,
            work: *mut $t, lwork: *const Int, iwork: *mut Int, liwork: *const Int, info: *mut Int,
        ) {
            unsafe {
                *info = syevd_c(char_arg(jobz), char_arg(uplo), rd(n), a, rd(lda), w, work, rd(lwork),
                    core::ptr::null_mut(), 0, iwork, rd(liwork));
            }
        }
    };
}
macro_rules! export_heevd {
    ($t:ty, $r:ty, $name:ident) => {
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $name(
            jobz: *const u8, uplo: *const u8, n: *const Int, a: *mut $t, lda: *const Int, w: *mut $r,
            work: *mut $t, lwork: *const Int, rwork: *mut $r, lrwork: *const Int,
            iwork: *mut Int, liwork: *const Int, info: *mut Int,
        ) {
            unsafe {
                *info = syevd_c(char_arg(jobz), char_arg(uplo), rd(n), a, rd(lda), w, work, rd(lwork),
                    rwork, rd(lrwork), iwork, rd(liwork));
            }
        }
    };
}
export_syevd!(f32, f32, ssyevd_);
export_syevd!(f64, f64, dsyevd_);
export_heevd!(c32, f32, cheevd_);
export_heevd!(c64, f64, zheevd_);
