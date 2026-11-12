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
    if mu == 2 && nu == 2 && !T::CPLX {
        // Match reference LAPACK signs exactly (faer's are arbitrary and differ).
        let at = |i: usize| unsafe { (*a.add(i % 2 + (i / 2) * lda as usize)).re() };
        let (sv, uu, vv) = gesdd_2x2([at(0), at(1), at(2), at(3)], super::eps_of::<T>() * 0.5);
        for i in 0..2 {
            unsafe { *s.add(i) = <T::Real as Re>::f(sv[i]) };
        }
        if ucols > 0 {
            for j in 0..2 {
                for i in 0..2 {
                    unsafe {
                        *u.add(i + j * ldu as usize) = T::r(uu[i + 2 * j]);
                        *vt.add(i + j * ldvt as usize) = T::r(vv[i + 2 * j]);
                    }
                }
            }
        }
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

fn sgn(a: f64, b: f64) -> f64 {
    if b >= 0.0 { a.abs() } else { -a.abs() }
}

/// Reference `DLASV2(F, G, H)` -> (ssmin, ssmax, snr, csr, snl, csl).
fn dlasv2(f: f64, g: f64, h: f64, eps: f64) -> (f64, f64, f64, f64, f64, f64) {
    let (mut ft, mut fa, mut ht, mut ha) = (f, f.abs(), h, h.abs());
    let mut pmax = 1;
    let swap = ha > fa;
    if swap {
        pmax = 3;
        core::mem::swap(&mut ft, &mut ht);
        core::mem::swap(&mut fa, &mut ha);
    }
    let (gt, ga) = (g, g.abs());
    let (mut ssmin, mut ssmax, mut clt, mut crt, mut slt, mut srt);
    if ga == 0.0 {
        (ssmin, ssmax, clt, crt, slt, srt) = (ha, fa, 1.0, 1.0, 0.0, 0.0);
    } else {
        let mut gasmal = true;
        (ssmin, ssmax, clt, crt, slt, srt) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        if ga > fa {
            pmax = 2;
            if fa / ga < eps {
                gasmal = false;
                ssmax = ga;
                ssmin = if ha > 1.0 { fa / (ga / ha) } else { (fa / ga) * ha };
                clt = 1.0;
                slt = ht / gt;
                srt = 1.0;
                crt = ft / gt;
            }
        }
        if gasmal {
            let d = fa - ha;
            let l = if d == fa { 1.0 } else { d / fa };
            let m = gt / ft;
            let mut t = 2.0 - l;
            let (mm, tt) = (m * m, t * t);
            let s = (tt + mm).sqrt();
            let r = if l == 0.0 { m.abs() } else { (l * l + mm).sqrt() };
            let a = 0.5 * (s + r);
            ssmin = ha / a;
            ssmax = fa * a;
            if mm == 0.0 {
                t = if l == 0.0 { sgn(2.0, ft) * sgn(1.0, gt) } else { gt / sgn(d, ft) + m / t };
            } else {
                t = (m / (s + t) + m / (r + l)) * (1.0 + a);
            }
            let l = (t * t + 4.0).sqrt();
            crt = 2.0 / l;
            srt = t / l;
            clt = (crt + srt * m) / a;
            slt = (ht / ft) * srt / a;
        }
    }
    let (csl, snl, csr, snr) = if swap { (srt, crt, slt, clt) } else { (clt, slt, crt, srt) };
    let tsign = match pmax {
        1 => sgn(1.0, csr) * sgn(1.0, csl) * sgn(1.0, f),
        2 => sgn(1.0, snr) * sgn(1.0, csl) * sgn(1.0, g),
        _ => sgn(1.0, snr) * sgn(1.0, snl) * sgn(1.0, h),
    };
    ssmax = sgn(ssmax, tsign);
    ssmin = sgn(ssmin, tsign * sgn(1.0, f) * sgn(1.0, h));
    (ssmin, ssmax, snr, csr, snl, csl)
}

/// Exact emulation of reference `xGESDD` for a real 2x2 matrix (DGEBD2 Householder,
/// DBDSQR/DLASV2 rotation, sign fix, descending sort) so singular-vector signs match
/// LAPACK. Returns (s, U, VT) column-major 2x2.
pub(crate) fn gesdd_2x2(a: [f64; 4], eps: f64) -> ([f64; 2], [f64; 4], [f64; 4]) {
    let (a00, a10, a01, a11) = (a[0], a[1], a[2], a[3]);
    // DLARFG(2, a00, a10)
    let (d1, v1, tau) = if a10 == 0.0 {
        (a00, 0.0, 0.0)
    } else {
        let beta = -sgn(a00.hypot(a10), a00);
        (beta, a10 / (a00 - beta), (beta - a00) / beta)
    };
    let w = a01 + v1 * a11;
    let (e, d2) = (a01 - tau * w, a11 - tau * v1 * w);
    // Q = I - tau v v^T, v = [1, v1]
    let q = [1.0 - tau, -tau * v1, -tau * v1, 1.0 - tau * v1 * v1];
    let mut u = [1.0, 0.0, 0.0, 1.0];
    let mut vt = [1.0, 0.0, 0.0, 1.0];
    let mut d = [d1, d2];
    if e != 0.0 {
        let (smin, smax, snr, csr, snl, csl) = dlasv2(d1, e, d2, eps);
        d = [smax, smin];
        // DROT on VT rows 0,1 (column-major: vt[i + 2j])
        for j in 0..2 {
            let (x, y) = (vt[2 * j], vt[1 + 2 * j]);
            vt[2 * j] = csr * x + snr * y;
            vt[1 + 2 * j] = csr * y - snr * x;
        }
        // DROT on U columns 0,1
        for i in 0..2 {
            let (x, y) = (u[i], u[i + 2]);
            u[i] = csl * x + snl * y;
            u[i + 2] = csl * y - snl * x;
        }
    }
    for i in 0..2 {
        if d[i] < 0.0 {
            d[i] = -d[i];
            vt[i] = -vt[i];
            vt[i + 2] = -vt[i + 2];
        }
    }
    if d[0] < d[1] {
        d.swap(0, 1);
        vt.swap(0, 1);
        vt.swap(2, 3);
        u.swap(0, 2);
        u.swap(1, 3);
    }
    // U = Q * U_bd (P = I for 2x2)
    let uf = [
        q[0] * u[0] + q[2] * u[1],
        q[1] * u[0] + q[3] * u[1],
        q[0] * u[2] + q[2] * u[3],
        q[1] * u[2] + q[3] * u[3],
    ];
    (d, uf, vt)
}

#[cfg(test)]
mod tests_2x2 {
    #[test]
    fn matches_lapack_signs() {
        // Reference (Accelerate dgesdd) for [[1,.5],[.5,2]] and [[1,2],[3,4]].
        let (s, u, vt) = super::gesdd_2x2([1.0, 0.5, 0.5, 2.0], f64::EPSILON * 0.5);
        let c = 0.38268343236508984;
        let sn = 0.9238795325112867;
        let want = [c, sn, sn, -c];
        for i in 0..4 {
            assert!((u[i] - want[i]).abs() < 1e-12, "u {u:?}");
            assert!((vt[i] - want[i]).abs() < 1e-12, "vt {vt:?}");
        }
        assert!((s[0] - 2.2071067811865475).abs() < 1e-12 && (s[1] - 0.7928932188134524).abs() < 1e-12);
        let (s, u, vt) = super::gesdd_2x2([1.0, 3.0, 2.0, 4.0], f64::EPSILON * 0.5);
        // reconstruction
        for i in 0..2 {
            for j in 0..2 {
                let r: f64 = (0..2).map(|k| u[i + 2 * k] * s[k] * vt[k + 2 * j]).sum();
                assert!((r - [1.0, 3.0, 2.0, 4.0][i + 2 * j]).abs() < 1e-12);
            }
        }
    }
}
