//! `xGESDD`: SVD, jobz 'A' (full U/VT), 'S' (thin), 'N' (values only).
//! U is m x ucol (ldu), VT is vrow x n (ldvt), VT = V^H. 'O' is not used by numpy.
use super::{load, set_iwork, set_work, Re, Sc};
use crate::ffi::{char_arg, par, rd, Int};
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::svd::{svd, svd_scratch, ComputeSvdVectors};
use faer::diag::Diag;
use faer::{c32, c64, Mat};

/// SVD of the m x n matrix at `a` (not modified). Returns (s, U, V) with faer layout
/// (U: m x ucols, V: n x vcols) or None on no convergence.
pub(crate) unsafe fn svd_core<T: Sc>(
    m: usize, n: usize, a: *const T, lda: usize, mode: ComputeSvdVectors,
) -> Option<(Vec<f64>, Mat<T>, Mat<T>)> {
    let am = unsafe { load(a, m, n, lda) };
    let k = m.min(n);
    let (uc, vc) = match mode {
        ComputeSvdVectors::No => (0, 0),
        ComputeSvdVectors::Thin => (k, k),
        ComputeSvdVectors::Full => (m, n),
    };
    let want = !matches!(mode, ComputeSvdVectors::No);
    let mut u = Mat::<T>::zeros(if want { m } else { 0 }, uc);
    let mut v = Mat::<T>::zeros(if want { n } else { 0 }, vc);
    let mut s = Diag::<T>::zeros(k);
    let p = par();
    let mut buf = MemBuffer::new(svd_scratch::<T>(m, n, mode, mode, p, Default::default()));
    let r = svd(
        am.as_ref(),
        s.as_mut(),
        if want { Some(u.as_mut()) } else { None },
        if want { Some(v.as_mut()) } else { None },
        p,
        MemStack::new(&mut buf),
        Default::default(),
    );
    if r.is_err() {
        return None;
    }
    Some(((0..k).map(|i| s[i].re().abs()).collect(), u, v))
}

unsafe fn gesdd_c<T: Sc>(
    jobz: u8, m: Int, n: Int, a: *mut T, lda: Int, s: *mut T::Real, u: *mut T, ldu: Int,
    vt: *mut T, ldvt: Int, work: *mut T, lwork: Int, iwork: *mut Int,
) -> Int {
    let mode = match jobz {
        b'A' => ComputeSvdVectors::Full,
        b'S' => ComputeSvdVectors::Thin,
        b'N' => ComputeSvdVectors::No,
        _ => return -1,
    };
    if m < 0 {
        return -2;
    }
    if n < 0 {
        return -3;
    }
    if lda < m.max(1) {
        return -5;
    }
    let k = m.min(n);
    let (ucols, vrows) = match jobz {
        b'A' => (m, n),
        b'S' => (k, k),
        _ => (0, 0),
    };
    if ldu < 1 || (ucols > 0 && ldu < m) {
        return -8;
    }
    if ldvt < 1 || (vrows > 0 && ldvt < vrows) {
        return -10;
    }
    unsafe { set_work(work, (m.max(n) * 4 + 1).max(1)) };
    let _ = iwork;
    unsafe { set_iwork(core::ptr::null_mut(), 0) };
    if lwork == -1 {
        return 0;
    }
    let (mu, nu) = (m as usize, n as usize);
    if mu == 0 || nu == 0 {
        return 0;
    }
    let Some((sv, uu, vv)) = (unsafe { svd_core(mu, nu, a, lda as usize, mode) }) else {
        return 1;
    };
    for (i, &x) in sv.iter().enumerate() {
        unsafe { *s.add(i) = <T::Real as Re>::f(x) };
    }
    if ucols > 0 {
        let ld = ldu as usize;
        for j in 0..ucols as usize {
            for i in 0..mu {
                unsafe { *u.add(i + j * ld) = uu[(i, j)] };
            }
        }
        let ld = ldvt as usize;
        for i in 0..vrows as usize {
            for j in 0..nu {
                unsafe { *vt.add(i + j * ld) = vv[(j, i)].cj() };
            }
        }
    }
    0
}

macro_rules! export_gesdd {
    ($t:ty, $r:ty, $name:ident) => {
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $name(
            jobz: *const u8, m: *const Int, n: *const Int, a: *mut $t, lda: *const Int, s: *mut $r,
            u: *mut $t, ldu: *const Int, vt: *mut $t, ldvt: *const Int, work: *mut $t, lwork: *const Int,
            iwork: *mut Int, info: *mut Int,
        ) {
            unsafe {
                *info = gesdd_c(char_arg(jobz), rd(m), rd(n), a, rd(lda), s, u, rd(ldu), vt, rd(ldvt),
                    work, rd(lwork), iwork);
            }
        }
    };
    ($t:ty, $r:ty, $name:ident, cplx) => {
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $name(
            jobz: *const u8, m: *const Int, n: *const Int, a: *mut $t, lda: *const Int, s: *mut $r,
            u: *mut $t, ldu: *const Int, vt: *mut $t, ldvt: *const Int, work: *mut $t, lwork: *const Int,
            _rwork: *mut $r, iwork: *mut Int, info: *mut Int,
        ) {
            unsafe {
                *info = gesdd_c(char_arg(jobz), rd(m), rd(n), a, rd(lda), s, u, rd(ldu), vt, rd(ldvt),
                    work, rd(lwork), iwork);
            }
        }
    };
}
export_gesdd!(f32, f32, sgesdd_);
export_gesdd!(f64, f64, dgesdd_);
export_gesdd!(c32, f32, cgesdd_, cplx);
export_gesdd!(c64, f64, zgesdd_, cplx);
