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
    if nu <= 2 {
        unsafe { small(vec, uplo, nu, a, ld, w) };
        return 0;
    }
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

/// Exact emulation of LAPACK for n <= 2 (xHETRD reflector + xSTEQR/xLAEV2 rotation +
/// ascending selection sort), so eigenvector signs/phases match reference LAPACK.
unsafe fn small<T: Sc>(vec: bool, uplo: u8, n: usize, a: *mut T, ld: usize, w: *mut T::Real) {
    let at = |i: usize, j: usize| unsafe { a.add(i + j * ld) };
    if n == 1 {
        unsafe {
            *w = <T::Real as Re>::f((*at(0, 0)).re());
            if vec {
                *at(0, 0) = T::r(1.0);
            }
        }
        return;
    }
    let (d1, d2) = unsafe { ((*at(0, 0)).re(), (*at(1, 1)).re()) };
    let alpha = unsafe { if uplo == b'L' { *at(1, 0) } else { *at(0, 1) } };
    // xLARFG with n=1: tau=0 unless alpha has an imaginary part.
    let (e, q) = if alpha.im() == 0.0 {
        (alpha.re(), T::r(1.0))
    } else {
        let beta = if alpha.re() >= 0.0 { -alpha.ab() } else { alpha.ab() };
        (beta, T::mk(alpha.re() / beta, alpha.im() / beta))
    };
    // q multiplies row 1 for 'L', row 0 for 'U'.
    let (mut ev, mut z) = if e == 0.0 {
        ([d1, d2], [[1.0, 0.0], [0.0, 1.0]])
    } else {
        let (rt1, rt2, c, s) = laev2(d1, e, d2);
        // columns: (c, s) for rt1, (-s, c) for rt2; z[col][row]
        ([rt1, rt2], [[c, s], [-s, c]])
    };
    if ev[1] < ev[0] {
        ev.swap(0, 1);
        z.swap(0, 1);
    }
    unsafe {
        *w = <T::Real as Re>::f(ev[0]);
        *w.add(1) = <T::Real as Re>::f(ev[1]);
    }
    if vec {
        let qr = if uplo == b'L' { 1 } else { 0 };
        for j in 0..2 {
            for i in 0..2 {
                let v = T::r(z[j][i]);
                unsafe { *at(i, j) = if i == qr { q * v } else { v } };
            }
        }
    }
}

/// LAPACK xLAEV2: eigen-decomposition of [[a, b], [b, c]]; (cs1, sn1) is the unit
/// eigenvector for rt1 (the eigenvalue of larger absolute value).
fn laev2(a: f64, b: f64, c: f64) -> (f64, f64, f64, f64) {
    let sm = a + c;
    let df = a - c;
    let adf = df.abs();
    let tb = b + b;
    let ab = tb.abs();
    let (acmx, acmn) = if a.abs() > c.abs() { (a, c) } else { (c, a) };
    let rt = if adf > ab {
        adf * (1.0 + (ab / adf).powi(2)).sqrt()
    } else if adf < ab {
        ab * (1.0 + (adf / ab).powi(2)).sqrt()
    } else {
        ab * 2f64.sqrt()
    };
    let (rt1, rt2, sgn1) = if sm < 0.0 {
        let rt1 = 0.5 * (sm - rt);
        (rt1, (acmx / rt1) * acmn - (b / rt1) * b, -1.0)
    } else if sm > 0.0 {
        let rt1 = 0.5 * (sm + rt);
        (rt1, (acmx / rt1) * acmn - (b / rt1) * b, 1.0)
    } else {
        (0.5 * rt, -0.5 * rt, 1.0)
    };
    let (cs, sgn2) = if df >= 0.0 { (df + rt, 1.0) } else { (df - rt, -1.0) };
    let (mut cs1, mut sn1);
    if cs.abs() > ab {
        let ct = -tb / cs;
        sn1 = 1.0 / (1.0 + ct * ct).sqrt();
        cs1 = ct * sn1;
    } else if ab == 0.0 {
        cs1 = 1.0;
        sn1 = 0.0;
    } else {
        let tn = -cs / tb;
        cs1 = 1.0 / (1.0 + tn * tn).sqrt();
        sn1 = tn * cs1;
    }
    if sgn1 == sgn2 {
        let tn = cs1;
        cs1 = -sn1;
        sn1 = tn;
    }
    (rt1, rt2, cs1, sn1)
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
