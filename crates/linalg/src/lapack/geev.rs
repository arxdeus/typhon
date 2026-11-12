//! `xGEEV`: general eigen-decomposition (right eigenvectors only; numpy uses jobvl='N').
//! Eigenvectors are normalized like LAPACK: unit 2-norm, largest component real.
//! Real version packs complex pairs as (re, im) columns with wi[j] > 0 first.
use super::{load, set_work, Re, Sc};
use crate::ffi::{char_arg, par, rd, Int};
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::evd::{evd_cplx, evd_real, evd_scratch, ComputeEigenvectors};
use faer::diag::Diag;
use faer::{c32, c64, Mat};

fn check(jobvl: u8, jobvr: u8, n: Int, lda: Int, ldvl: Int, ldvr: Int, vr_needed: bool) -> Int {
    if jobvl != b'N' {
        return -1; // left eigenvectors not supported (unused by numpy)
    }
    if jobvr != b'N' && jobvr != b'V' {
        return -2;
    }
    if n < 0 {
        return -3;
    }
    if lda < n.max(1) {
        return -5;
    }
    if ldvl < 1 {
        return -9;
    }
    if ldvr < 1 || (vr_needed && ldvr < n) {
        return -11;
    }
    0
}

/// Real geev core. Returns info.
pub(crate) unsafe fn geev_real<R: Re + Sc>(
    jobvr: u8, n: usize, a: *mut R, lda: usize, wr: *mut R, wi: *mut R, vr: *mut R, ldvr: usize,
) -> Int {
    let vec = jobvr == b'V';
    let am = unsafe { load(a as *const R, n, n, lda) };
    let p = par();
    let cu = if vec { ComputeEigenvectors::Yes } else { ComputeEigenvectors::No };
    let mut sr = Diag::<R>::zeros(n);
    let mut si = Diag::<R>::zeros(n);
    let mut u = if vec { Mat::<R>::zeros(n, n) } else { Mat::<R>::zeros(0, 0) };
    let mut buf = MemBuffer::new(evd_scratch::<R>(n, ComputeEigenvectors::No, cu, p, Default::default()));
    let r = evd_real(am.as_ref(), sr.as_mut(), si.as_mut(), None, if vec { Some(u.as_mut()) } else { None },
        p, MemStack::new(&mut buf), Default::default());
    if r.is_err() {
        return n.max(1) as Int;
    }
    let mut j = 0;
    while j < n {
        let im = Sc::re(si[j]);
        if im == 0.0 || j + 1 == n {
            unsafe { *wr.add(j) = sr[j]; *wi.add(j) = <R as Sc>::zero() };
            if vec {
                let nrm = (0..n).map(|i| u[(i, j)].ab2()).sum::<f64>().sqrt();
                let s = if nrm > 0.0 { 1.0 / nrm } else { 1.0 };
                for i in 0..n {
                    unsafe { *vr.add(i + j * ldvr) = R::r(u[(i, j)].re() * s) };
                }
            }
            j += 1;
            continue;
        }
        let flip = im < 0.0;
        let aim = im.abs();
        unsafe {
            *wr.add(j) = sr[j];
            *wr.add(j + 1) = sr[j];
            *wi.add(j) = R::r(aim);
            *wi.add(j + 1) = R::r(-aim);
        }
        if vec {
            // complex vector x = u_j + i*s*u_{j+1}, normalized, rotated so max |x_i| is real
            let sg = if flip { -1.0 } else { 1.0 };
            let x: Vec<(f64, f64)> = (0..n).map(|i| (u[(i, j)].re(), sg * u[(i, j + 1)].re())).collect();
            let (re, imv) = normalize(&x);
            for i in 0..n {
                unsafe {
                    *vr.add(i + j * ldvr) = R::r(re[i]);
                    *vr.add(i + (j + 1) * ldvr) = R::r(imv[i]);
                }
            }
        }
        j += 2;
    }
    0
}

/// Unit 2-norm, then multiply by conj(x_k)/|x_k| for the component of largest modulus
/// (LAPACK sets that component's imaginary part to exactly 0).
fn normalize(x: &[(f64, f64)]) -> (Vec<f64>, Vec<f64>) {
    let nrm = x.iter().map(|&(a, b)| a * a + b * b).sum::<f64>().sqrt();
    let s = if nrm > 0.0 { 1.0 / nrm } else { 1.0 };
    let mut k = 0;
    let mut best = -1.0;
    for (i, &(a, b)) in x.iter().enumerate() {
        let m = a * a + b * b;
        if m > best {
            best = m;
            k = i;
        }
    }
    let (ka, kb) = x[k];
    let km = (ka * ka + kb * kb).sqrt();
    let (ca, cb) = if km > 0.0 { (ka / km, -kb / km) } else { (1.0, 0.0) };
    let mut re = Vec::with_capacity(x.len());
    let mut im = Vec::with_capacity(x.len());
    for (i, &(a, b)) in x.iter().enumerate() {
        re.push((a * ca - b * cb) * s);
        im.push(if i == k { 0.0 } else { (a * cb + b * ca) * s });
    }
    (re, im)
}

/// Complex geev core.
pub(crate) unsafe fn geev_cplx<T: Sc>(jobvr: u8, n: usize, a: *mut T, lda: usize, w: *mut T, vr: *mut T, ldvr: usize) -> Int
where
    T: faer::traits::ComplexField,
    (): CplxEvd<T>,
{
    unsafe { <() as CplxEvd<T>>::run(jobvr, n, a, lda, w, vr, ldvr) }
}

pub(crate) trait CplxEvd<T> {
    unsafe fn run(jobvr: u8, n: usize, a: *mut T, lda: usize, w: *mut T, vr: *mut T, ldvr: usize) -> Int;
}
macro_rules! cplx_evd {
    ($t:ty, $r:ty) => {
        impl CplxEvd<$t> for () {
            unsafe fn run(jobvr: u8, n: usize, a: *mut $t, lda: usize, w: *mut $t, vr: *mut $t, ldvr: usize) -> Int {
                let vec = jobvr == b'V';
                let am = unsafe { load(a as *const $t, n, n, lda) };
                let p = par();
                let cu = if vec { ComputeEigenvectors::Yes } else { ComputeEigenvectors::No };
                let mut s = Diag::<$t>::zeros(n);
                let mut u = if vec { Mat::<$t>::zeros(n, n) } else { Mat::<$t>::zeros(0, 0) };
                let mut buf = MemBuffer::new(evd_scratch::<$t>(n, ComputeEigenvectors::No, cu, p, Default::default()));
                let r = evd_cplx::<$r>(am.as_ref(), s.as_mut(), None, if vec { Some(u.as_mut()) } else { None },
                    p, MemStack::new(&mut buf), Default::default());
                if r.is_err() {
                    return n.max(1) as Int;
                }
                for j in 0..n {
                    unsafe { *w.add(j) = s[j] };
                    if vec {
                        let x: Vec<(f64, f64)> = (0..n).map(|i| (u[(i, j)].re(), u[(i, j)].im())).collect();
                        let (re, im) = normalize(&x);
                        for i in 0..n {
                            unsafe { *vr.add(i + j * ldvr) = <$t as Sc>::mk(re[i], im[i]) };
                        }
                    }
                }
                0
            }
        }
    };
}
cplx_evd!(c32, f32);
cplx_evd!(c64, f64);

macro_rules! export_geev {
    ($t:ty, $name:ident, $zname:ident, $c:ty) => {
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $name(
            jobvl: *const u8, jobvr: *const u8, n: *const Int, a: *mut $t, lda: *const Int,
            wr: *mut $t, wi: *mut $t, _vl: *mut $t, ldvl: *const Int, vr: *mut $t, ldvr: *const Int,
            work: *mut $t, lwork: *const Int, info: *mut Int,
        ) {
            let (jl, jr, nn) = unsafe { (char_arg(jobvl), char_arg(jobvr), rd(n)) };
            let e = check(jl, jr, nn, unsafe { rd(lda) }, unsafe { rd(ldvl) }, unsafe { rd(ldvr) }, jr == b'V');
            unsafe { *info = e; set_work(work, 4 * nn.max(1)) };
            if e != 0 || unsafe { rd(lwork) } == -1 || nn == 0 {
                return;
            }
            unsafe { *info = geev_real(jr, nn as usize, a, rd(lda) as usize, wr, wi, vr, rd(ldvr).max(1) as usize) };
        }
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $zname(
            jobvl: *const u8, jobvr: *const u8, n: *const Int, a: *mut $c, lda: *const Int,
            w: *mut $c, _vl: *mut $c, ldvl: *const Int, vr: *mut $c, ldvr: *const Int,
            work: *mut $c, lwork: *const Int, _rwork: *mut $t, info: *mut Int,
        ) {
            let (jl, jr, nn) = unsafe { (char_arg(jobvl), char_arg(jobvr), rd(n)) };
            let e = check(jl, jr, nn, unsafe { rd(lda) }, unsafe { rd(ldvl) }, unsafe { rd(ldvr) }, jr == b'V');
            unsafe { *info = e; set_work(work, 2 * nn.max(1)) };
            if e != 0 || unsafe { rd(lwork) } == -1 || nn == 0 {
                return;
            }
            unsafe { *info = geev_cplx(jr, nn as usize, a, rd(lda) as usize, w, vr, rd(ldvr).max(1) as usize) };
        }
    };
}
export_geev!(f32, sgeev_, cgeev_, c32);
export_geev!(f64, dgeev_, zgeev_, c64);
