//! `xGEQRF` and `xORGQR`/`xUNGQR`: unblocked Householder QR in exact LAPACK compact
//! format (xLARFG reflectors `H = I - tau v v^H`, `v(0) = 1`, `A = Q R`, `Q = H1..Hk`).
use super::{set_work, Sc};
use crate::ffi::{rd, Int};
use faer::{c32, c64};

/// xLARFG on (alpha, x[0..len] with stride 1). Returns tau; overwrites alpha with beta
/// and x with v(1:).
pub(crate) unsafe fn larfg<T: Sc>(alpha: &mut T, x: *mut T, len: usize) -> T {
    let mut xn2 = 0.0f64;
    for i in 0..len {
        xn2 += unsafe { *x.add(i) }.ab2();
    }
    let (ar, ai) = (alpha.re(), alpha.im());
    if xn2 == 0.0 && ai == 0.0 {
        return T::zero();
    }
    let nrm = (ar * ar + ai * ai + xn2).sqrt();
    let beta = if ar >= 0.0 { -nrm } else { nrm };
    let tau = T::mk((beta - ar) / beta, -ai / beta);
    let scal = T::r(1.0) / (*alpha - T::r(beta));
    for i in 0..len {
        unsafe { *x.add(i) = *x.add(i) * scal };
    }
    *alpha = T::r(beta);
    tau
}

/// Apply `H^H` (if `herm`) or `H` from the left to columns `c0..c1` of A rows `r0..m`,
/// where v = [1; A(r0+1..m, vc)].
unsafe fn apply_left<T: Sc>(a: *mut T, lda: usize, m: usize, r0: usize, vc: usize, tau: T, c0: usize, c1: usize) {
    if tau == T::zero() {
        return;
    }
    let v = |i: usize| if i == r0 { T::r(1.0) } else { unsafe { *a.add(i + vc * lda) } };
    for j in c0..c1 {
        let col = unsafe { a.add(j * lda) };
        let mut s = T::zero();
        for i in r0..m {
            s = s + v(i).cj() * unsafe { *col.add(i) };
        }
        let s = s * tau;
        for i in r0..m {
            unsafe { *col.add(i) = *col.add(i) - v(i) * s };
        }
    }
}

pub(crate) unsafe fn geqrf<T: Sc>(m: usize, n: usize, a: *mut T, lda: usize, tau: *mut T) {
    let k = m.min(n);
    for i in 0..k {
        let p = unsafe { a.add(i + i * lda) };
        let mut alpha = unsafe { *p };
        let t = unsafe { larfg(&mut alpha, p.add(1), m - i - 1) };
        unsafe { *tau.add(i) = t };
        unsafe { *p = T::r(1.0) };
        unsafe { apply_left(a, lda, m, i, i, t.cj(), i + 1, n) };
        unsafe { *p = alpha };
    }
}

/// Form the `m x n` Q from the first `k` reflectors stored in A (xORG2R/xUNG2R).
pub(crate) unsafe fn orgqr<T: Sc>(m: usize, n: usize, k: usize, a: *mut T, lda: usize, tau: *const T) {
    let at = |i: usize, j: usize| unsafe { a.add(i + j * lda) };
    for j in k..n {
        for l in 0..m {
            unsafe { *at(l, j) = T::zero() };
        }
        unsafe { *at(j, j) = T::r(1.0) };
    }
    for i in (0..k).rev() {
        let t = unsafe { *tau.add(i) };
        if i + 1 < n {
            unsafe { *at(i, i) = T::r(1.0) };
            unsafe { apply_left(a, lda, m, i, i, t, i + 1, n) };
        }
        for l in i + 1..m {
            unsafe { *at(l, i) = -t * *at(l, i) };
        }
        unsafe { *at(i, i) = T::r(1.0) - t };
        for l in 0..i {
            unsafe { *at(l, i) = T::zero() };
        }
    }
}

unsafe fn geqrf_c<T: Sc>(m: Int, n: Int, a: *mut T, lda: Int, tau: *mut T, work: *mut T, lwork: Int) -> Int {
    if m < 0 {
        return -1;
    }
    if n < 0 {
        return -2;
    }
    if lda < m.max(1) {
        return -4;
    }
    let opt = n.max(1);
    unsafe { set_work(work, opt) };
    if lwork == -1 {
        return 0;
    }
    if lwork < opt {
        return -7;
    }
    unsafe { geqrf(m as usize, n as usize, a, lda as usize, tau) };
    0
}

unsafe fn orgqr_c<T: Sc>(m: Int, n: Int, k: Int, a: *mut T, lda: Int, tau: *const T, work: *mut T, lwork: Int) -> Int {
    if m < 0 {
        return -1;
    }
    if n < 0 || n > m {
        return -2;
    }
    if k < 0 || k > n {
        return -3;
    }
    if lda < m.max(1) {
        return -5;
    }
    let opt = n.max(1);
    unsafe { set_work(work, opt) };
    if lwork == -1 {
        return 0;
    }
    if lwork < opt {
        return -8;
    }
    unsafe { orgqr(m as usize, n as usize, k as usize, a, lda as usize, tau) };
    0
}

macro_rules! export_qr {
    ($t:ty, $geqrf:ident, $orgqr:ident) => {
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $geqrf(
            m: *const Int, n: *const Int, a: *mut $t, lda: *const Int, tau: *mut $t,
            work: *mut $t, lwork: *const Int, info: *mut Int,
        ) {
            unsafe { *info = geqrf_c(rd(m), rd(n), a, rd(lda), tau, work, rd(lwork)) };
        }
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $orgqr(
            m: *const Int, n: *const Int, k: *const Int, a: *mut $t, lda: *const Int, tau: *const $t,
            work: *mut $t, lwork: *const Int, info: *mut Int,
        ) {
            unsafe { *info = orgqr_c(rd(m), rd(n), rd(k), a, rd(lda), tau, work, rd(lwork)) };
        }
    };
}
export_qr!(f32, sgeqrf_, sorgqr_);
export_qr!(f64, dgeqrf_, dorgqr_);
export_qr!(c32, cgeqrf_, cungqr_);
export_qr!(c64, zgeqrf_, zungqr_);
