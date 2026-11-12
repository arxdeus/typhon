//! Unit tests: syevd/heevd, geev, gesdd, gelsd for all four dtypes.
use super::tests::{at, close, mm, rand, tol, Rng};
use super::{Re, Sc};
use crate::ffi::Int;
use faer::{c32, c64};

trait Lap2: Sc {
    type R: Re + Sc;
    unsafe fn evd(j: &u8, u: &u8, n: &Int, a: *mut Self, lda: &Int, w: *mut Self::R, work: *mut Self, lw: &Int, rw: *mut Self::R, lrw: &Int, iw: *mut Int, liw: &Int, info: &mut Int);
    unsafe fn sdd(j: &u8, m: &Int, n: &Int, a: *mut Self, lda: &Int, s: *mut Self::R, u: *mut Self, ldu: &Int, vt: *mut Self, ldvt: &Int, w: *mut Self, lw: &Int, rw: *mut Self::R, iw: *mut Int, info: &mut Int);
    unsafe fn lsd(m: &Int, n: &Int, r: &Int, a: *mut Self, lda: &Int, b: *mut Self, ldb: &Int, s: *mut Self::R, rc: &Self::R, rank: &mut Int, w: *mut Self, lw: &Int, rw: *mut Self::R, iw: *mut Int, info: &mut Int);
    /// Returns eigenvalues and eigenvectors as complex f64 pairs.
    unsafe fn eig(n: usize, a: &mut [Self], ld: usize) -> (Int, Vec<(f64, f64)>, Vec<(f64, f64)>);
}
macro_rules! real2 {
    ($t:ty, $evd:ident, $sdd:ident, $lsd:ident, $geev:ident) => {
        impl Lap2 for $t {
            type R = $t;
            unsafe fn evd(j: &u8, u: &u8, n: &Int, a: *mut Self, lda: &Int, w: *mut $t, work: *mut Self, lw: &Int, _rw: *mut $t, _lrw: &Int, iw: *mut Int, liw: &Int, info: &mut Int) { unsafe { super::eigh::$evd(j, u, n, a, lda, w, work, lw, iw, liw, info) } }
            unsafe fn sdd(j: &u8, m: &Int, n: &Int, a: *mut Self, lda: &Int, s: *mut $t, u: *mut Self, ldu: &Int, vt: *mut Self, ldvt: &Int, w: *mut Self, lw: &Int, _rw: *mut $t, iw: *mut Int, info: &mut Int) { unsafe { super::svd::$sdd(j, m, n, a, lda, s, u, ldu, vt, ldvt, w, lw, iw, info) } }
            unsafe fn lsd(m: &Int, n: &Int, r: &Int, a: *mut Self, lda: &Int, b: *mut Self, ldb: &Int, s: *mut $t, rc: &$t, rank: &mut Int, w: *mut Self, lw: &Int, _rw: *mut $t, iw: *mut Int, info: &mut Int) { unsafe { super::lstsq::$lsd(m, n, r, a, lda, b, ldb, s, rc, rank, w, lw, iw, info) } }
            unsafe fn eig(n: usize, a: &mut [Self], ld: usize) -> (Int, Vec<(f64, f64)>, Vec<(f64, f64)>) {
                let (mut wr, mut wi, mut vr) = (vec![0.0 as $t; n], vec![0.0 as $t; n], vec![0.0 as $t; n * n]);
                let (mut q, mut info) = (0.0 as $t, 0);
                let mut d = 0.0 as $t;
                unsafe { super::geev::$geev(&b'N', &b'V', &(n as Int), a.as_mut_ptr(), &(ld as Int), wr.as_mut_ptr(), wi.as_mut_ptr(), &mut d, &1, vr.as_mut_ptr(), &(n as Int), &mut q, &-1, &mut info) };
                let mut w = vec![0.0 as $t; q as usize];
                unsafe { super::geev::$geev(&b'N', &b'V', &(n as Int), a.as_mut_ptr(), &(ld as Int), wr.as_mut_ptr(), wi.as_mut_ptr(), &mut d, &1, vr.as_mut_ptr(), &(n as Int), w.as_mut_ptr(), &(q as Int), &mut info) };
                let ev: Vec<(f64, f64)> = (0..n).map(|i| (wr[i] as f64, wi[i] as f64)).collect();
                let mut v = vec![(0.0, 0.0); n * n];
                let mut j = 0;
                while j < n {
                    if wi[j] == 0.0 { for i in 0..n { v[i + j * n] = (vr[i + j * n] as f64, 0.0); } j += 1; }
                    else {
                        assert!(wi[j] > 0.0 && wi[j + 1] == -wi[j]);
                        for i in 0..n { let (re, im) = (vr[i + j * n] as f64, vr[i + (j + 1) * n] as f64); v[i + j * n] = (re, im); v[i + (j + 1) * n] = (re, -im); }
                        j += 2;
                    }
                }
                (info, ev, v)
            }
        }
    };
}
macro_rules! cplx2 {
    ($t:ty, $r:ty, $evd:ident, $sdd:ident, $lsd:ident, $geev:ident) => {
        impl Lap2 for $t {
            type R = $r;
            unsafe fn evd(j: &u8, u: &u8, n: &Int, a: *mut Self, lda: &Int, w: *mut $r, work: *mut Self, lw: &Int, rw: *mut $r, lrw: &Int, iw: *mut Int, liw: &Int, info: &mut Int) { unsafe { super::eigh::$evd(j, u, n, a, lda, w, work, lw, rw, lrw, iw, liw, info) } }
            unsafe fn sdd(j: &u8, m: &Int, n: &Int, a: *mut Self, lda: &Int, s: *mut $r, u: *mut Self, ldu: &Int, vt: *mut Self, ldvt: &Int, w: *mut Self, lw: &Int, rw: *mut $r, iw: *mut Int, info: &mut Int) { unsafe { super::svd::$sdd(j, m, n, a, lda, s, u, ldu, vt, ldvt, w, lw, rw, iw, info) } }
            unsafe fn lsd(m: &Int, n: &Int, r: &Int, a: *mut Self, lda: &Int, b: *mut Self, ldb: &Int, s: *mut $r, rc: &$r, rank: &mut Int, w: *mut Self, lw: &Int, rw: *mut $r, iw: *mut Int, info: &mut Int) { unsafe { super::lstsq::$lsd(m, n, r, a, lda, b, ldb, s, rc, rank, w, lw, rw, iw, info) } }
            unsafe fn eig(n: usize, a: &mut [Self], ld: usize) -> (Int, Vec<(f64, f64)>, Vec<(f64, f64)>) {
                let (mut w, mut vr, mut rw) = (vec![<$t>::default(); n], vec![<$t>::default(); n * n], vec![0.0 as $r; 2 * n]);
                let (mut q, mut info, mut d) = (<$t>::default(), 0, <$t>::default());
                unsafe { super::geev::$geev(&b'N', &b'V', &(n as Int), a.as_mut_ptr(), &(ld as Int), w.as_mut_ptr(), &mut d, &1, vr.as_mut_ptr(), &(n as Int), &mut q, &-1, rw.as_mut_ptr(), &mut info) };
                let mut wk = vec![<$t>::default(); q.re as usize];
                unsafe { super::geev::$geev(&b'N', &b'V', &(n as Int), a.as_mut_ptr(), &(ld as Int), w.as_mut_ptr(), &mut d, &1, vr.as_mut_ptr(), &(n as Int), wk.as_mut_ptr(), &(q.re as Int), rw.as_mut_ptr(), &mut info) };
                (info, w.iter().map(|x| (x.re as f64, x.im as f64)).collect(), vr.iter().map(|x| (x.re as f64, x.im as f64)).collect())
            }
        }
    };
}
real2!(f32, ssyevd_, sgesdd_, sgelsd_, sgeev_);
real2!(f64, dsyevd_, dgesdd_, dgelsd_, dgeev_);
cplx2!(c32, f32, cheevd_, cgesdd_, cgelsd_, cgeev_);
cplx2!(c64, f64, zheevd_, zgesdd_, zgelsd_, zgeev_);

fn c(x: (f64, f64)) -> c64 { c64::new(x.0, x.1) }

fn evd_check<T: Lap2>(n: usize, up: bool) {
    let mut r = Rng::new(n as u64 + 11);
    let ld = n + 2;
    let g = rand::<T>(&mut r, n, n, ld);
    let a0: Vec<T> = (0..ld * n).map(|x| { let (i, j) = (x % ld, x / ld); if i < n { at(&g, ld, i, j) + at(&g, ld, j, i).cj() } else { T::zero() } }).collect();
    let mut a = a0.clone();
    let (mut w, mut wq, mut rq, mut iq, mut info) = (vec![<T::R as Sc>::zero(); n], T::zero(), <T::R as Sc>::zero(), 0 as Int, 0);
    let uplo = if up { b'U' } else { b'L' };
    unsafe { T::evd(&b'V', &uplo, &(n as Int), a.as_mut_ptr(), &(ld as Int), w.as_mut_ptr(), &mut wq, &-1, &mut rq, &-1, &mut iq, &-1, &mut info) };
    assert_eq!(info, 0);
    let (lw, lrw, liw) = (wq.re() as usize, Sc::re(rq) as usize, iq as usize);
    let (mut wk, mut rw, mut iw) = (vec![T::zero(); lw], vec![<T::R as Sc>::zero(); lrw.max(1)], vec![0; liw]);
    unsafe { T::evd(&b'V', &uplo, &(n as Int), a.as_mut_ptr(), &(ld as Int), w.as_mut_ptr(), wk.as_mut_ptr(), &(lw as Int), rw.as_mut_ptr(), &(lrw as Int), iw.as_mut_ptr(), &(liw as Int), &mut info) };
    assert_eq!(info, 0);
    for i in 1..n { assert!(Sc::re(w[i - 1]) <= Sc::re(w[i])); }
    let av = mm(&a0, ld, &a, ld, n, n, n);
    for j in 0..n { for i in 0..n { assert!(close(av[i + j * n], at(&a, ld, i, j) * T::r(Sc::re(w[j])), 10.0 * n as f64), "evd"); } }
}

fn geev_check<T: Lap2>(n: usize) {
    let mut r = Rng::new(n as u64 + 5);
    let ld = n + 1;
    let a0 = rand::<T>(&mut r, n, n, ld);
    let mut a = a0.clone();
    let (info, ev, v) = unsafe { T::eig(n, &mut a, ld) };
    assert_eq!(info, 0);
    for j in 0..n {
        let nrm: f64 = (0..n).map(|i| c(v[i + j * n]).norm_sqr()).sum();
        assert!((nrm - 1.0).abs() < tol::<T>() * 10.0, "unit norm");
        for i in 0..n {
            let mut s = c64::new(0.0, 0.0);
            for p in 0..n { s += c((at(&a0, ld, i, p).re(), at(&a0, ld, i, p).im())) * c(v[p + j * n]); }
            assert!((s - c(ev[j]) * c(v[i + j * n])).norm() < tol::<T>() * 20.0 * n as f64, "geev {n}");
        }
    }
}

fn sdd_check<T: Lap2>(m: usize, n: usize, job: u8) {
    let mut r = Rng::new((m * 7 + n) as u64);
    let ld = m + 1;
    let k = m.min(n);
    let a0 = rand::<T>(&mut r, m, n, ld);
    let mut a = a0.clone();
    let (uc, vr) = if job == b'A' { (m, n) } else { (k, k) };
    let (mut s, mut u, mut vt) = (vec![<T::R as Sc>::zero(); k], vec![T::zero(); m * uc.max(1)], vec![T::zero(); vr.max(1) * n]);
    let (mut q, mut info) = (T::zero(), 0);
    let mut rw = vec![<T::R as Sc>::zero(); 5 * k * k + 7 * k + 1];
    let mut iw = vec![0; 8 * k];
    unsafe { T::sdd(&job, &(m as Int), &(n as Int), a.as_mut_ptr(), &(ld as Int), s.as_mut_ptr(), u.as_mut_ptr(), &(m as Int), vt.as_mut_ptr(), &(vr.max(1) as Int), &mut q, &-1, rw.as_mut_ptr(), iw.as_mut_ptr(), &mut info) };
    let mut w = vec![T::zero(); q.re() as usize];
    unsafe { T::sdd(&job, &(m as Int), &(n as Int), a.as_mut_ptr(), &(ld as Int), s.as_mut_ptr(), u.as_mut_ptr(), &(m as Int), vt.as_mut_ptr(), &(vr.max(1) as Int), w.as_mut_ptr(), &(w.len() as Int), rw.as_mut_ptr(), iw.as_mut_ptr(), &mut info) };
    assert_eq!(info, 0);
    for i in 1..k { assert!(Sc::re(s[i - 1]) >= Sc::re(s[i]) && Sc::re(s[i]) >= 0.0); }
    let us: Vec<T> = (0..m * k).map(|x| u[x] * T::r(Sc::re(s[x / m]))).collect();
    let rec = mm(&us, m, &vt, vr.max(1), m, k, n);
    for j in 0..n { for i in 0..m { assert!(close(rec[i + j * m], at(&a0, ld, i, j), 10.0), "svd {m}x{n} {}", job as char); } }
}

fn lsd_check<T: Lap2>(m: usize, n: usize) {
    let mut r = Rng::new((m * 3 + n) as u64);
    let (ld, ldb, nr) = (m, m.max(n), 2);
    let a0 = rand::<T>(&mut r, m, n, ld);
    let mut b0 = vec![T::zero(); ldb * nr];
    for j in 0..nr { for i in 0..m { b0[i + j * ldb] = r.s(); } }
    let (mut a, mut b) = (a0.clone(), b0.clone());
    let mut s = vec![<T::R as Sc>::zero(); m.min(n)];
    let (mut q, mut rq, mut iq, mut rank, mut info) = (T::zero(), <T::R as Sc>::zero(), 0, 0, 0);
    let rc = <T::R as Re>::f(-1.0);
    unsafe { T::lsd(&(m as Int), &(n as Int), &(nr as Int), a.as_mut_ptr(), &(ld as Int), b.as_mut_ptr(), &(ldb as Int), s.as_mut_ptr(), &rc, &mut rank, &mut q, &-1, &mut rq, &mut iq, &mut info) };
    let (mut w, mut rw, mut iw) = (vec![T::zero(); q.re() as usize], vec![<T::R as Sc>::zero(); Sc::re(rq) as usize], vec![0; iq as usize]);
    unsafe { T::lsd(&(m as Int), &(n as Int), &(nr as Int), a.as_mut_ptr(), &(ld as Int), b.as_mut_ptr(), &(ldb as Int), s.as_mut_ptr(), &rc, &mut rank, w.as_mut_ptr(), &(w.len() as Int), rw.as_mut_ptr(), iw.as_mut_ptr(), &mut info) };
    assert_eq!(info, 0);
    assert_eq!(rank as usize, m.min(n));
    // normal equations A^H (A x - b) = 0
    let ax = mm(&a0, ld, &b, ldb, m, n, nr);
    for j in 0..nr { for p in 0..n {
        let mut g = T::zero();
        for i in 0..m { g = g + at(&a0, ld, i, p).cj() * (ax[i + j * m] - b0[i + j * ldb]); }
        assert!(close(g, T::zero(), 20.0), "lstsq normal eq");
    } }
    if m > n { // residual from trailing rows
        for j in 0..nr {
            let r1: f64 = (0..m).map(|i| (ax[i + j * m] - b0[i + j * ldb]).ab2()).sum();
            let r2: f64 = (n..m).map(|i| b[i + j * ldb].ab2()).sum();
            assert!((r1 - r2).abs() < tol::<T>() * 20.0, "residual");
        }
    }
}

fn all<T: Lap2>() {
    for &n in &[1, 3, 8, 25] { evd_check::<T>(n, false); evd_check::<T>(n, true); geev_check::<T>(n); }
    for &(m, n) in &[(1, 1), (6, 6), (9, 4), (4, 9), (30, 21)] {
        for &j in &[b'A', b'S', b'N'] { if j != b'N' { sdd_check::<T>(m, n, j); } }
        lsd_check::<T>(m, n);
    }
}
#[test] fn eig_svd_lstsq_f32() { all::<f32>(); }
#[test] fn eig_svd_lstsq_f64() { all::<f64>(); }
#[test] fn eig_svd_lstsq_c32() { all::<c32>(); }
#[test] fn eig_svd_lstsq_c64() { all::<c64>(); }
