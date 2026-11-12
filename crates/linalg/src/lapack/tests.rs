//! Unit tests: LU / gesv / potrf / QR for all four dtypes.
use super::Sc;
use crate::ffi::Int;
use faer::{c32, c64};

pub(super) struct Rng(u64);
impl Rng {
    pub fn new(s: u64) -> Self { Rng(s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407)) }
    pub fn f(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }
    pub fn s<T: Sc>(&mut self) -> T { let a = self.f(); let b = self.f(); T::mk(a, if T::CPLX { b } else { 0.0 }) }
}

/// Column-major matrix with leading dimension (rows + pad).
pub(super) fn rand<T: Sc>(r: &mut Rng, _m: usize, n: usize, ld: usize) -> Vec<T> {
    (0..ld * n.max(1)).map(|_| r.s::<T>()).collect()
}
pub(super) fn tol<T: Sc>() -> f64 { if super::eps_of::<T>() > 1e-10 { 2e-3 } else { 1e-9 } }
pub(super) fn at<T: Sc>(a: &[T], ld: usize, i: usize, j: usize) -> T { a[i + j * ld] }
/// C = A (m x k) * B (k x n), dense col-major with given lds.
pub(super) fn mm<T: Sc>(a: &[T], lda: usize, b: &[T], ldb: usize, m: usize, k: usize, n: usize) -> Vec<T> {
    let mut c = vec![T::zero(); m * n];
    for j in 0..n { for p in 0..k { let bv = b[p + j * ldb]; for i in 0..m { c[i + j * m] = c[i + j * m] + a[i + p * lda] * bv; } } }
    c
}
pub(super) fn close<T: Sc>(x: T, y: T, scale: f64) -> bool { (x - y).ab() <= tol::<T>() * scale.max(1.0) }

pub(super) trait Lap: Sc {
    unsafe fn getrf(m: &Int, n: &Int, a: *mut Self, lda: &Int, ipiv: *mut Int, info: &mut Int);
    unsafe fn gesv(n: &Int, r: &Int, a: *mut Self, lda: &Int, ipiv: *mut Int, b: *mut Self, ldb: &Int, info: &mut Int);
    unsafe fn potrf(u: &u8, n: &Int, a: *mut Self, lda: &Int, info: &mut Int);
    unsafe fn geqrf(m: &Int, n: &Int, a: *mut Self, lda: &Int, tau: *mut Self, w: *mut Self, lw: &Int, info: &mut Int);
    unsafe fn orgqr(m: &Int, n: &Int, k: &Int, a: *mut Self, lda: &Int, tau: *const Self, w: *mut Self, lw: &Int, info: &mut Int);
}
macro_rules! lap {
    ($t:ty, $getrf:ident, $gesv:ident, $potrf:ident, $geqrf:ident, $orgqr:ident) => {
        impl Lap for $t {
            unsafe fn getrf(m: &Int, n: &Int, a: *mut Self, lda: &Int, ipiv: *mut Int, info: &mut Int) { unsafe { super::lu::$getrf(m, n, a, lda, ipiv, info) } }
            unsafe fn gesv(n: &Int, r: &Int, a: *mut Self, lda: &Int, ipiv: *mut Int, b: *mut Self, ldb: &Int, info: &mut Int) { unsafe { super::lu::$gesv(n, r, a, lda, ipiv, b, ldb, info) } }
            unsafe fn potrf(u: &u8, n: &Int, a: *mut Self, lda: &Int, info: &mut Int) { unsafe { super::chol::$potrf(u, n, a, lda, info) } }
            unsafe fn geqrf(m: &Int, n: &Int, a: *mut Self, lda: &Int, tau: *mut Self, w: *mut Self, lw: &Int, info: &mut Int) { unsafe { super::qr::$geqrf(m, n, a, lda, tau, w, lw, info) } }
            unsafe fn orgqr(m: &Int, n: &Int, k: &Int, a: *mut Self, lda: &Int, tau: *const Self, w: *mut Self, lw: &Int, info: &mut Int) { unsafe { super::qr::$orgqr(m, n, k, a, lda, tau, w, lw, info) } }
        }
    };
}
lap!(f32, sgetrf_, sgesv_, spotrf_, sgeqrf_, sorgqr_);
lap!(f64, dgetrf_, dgesv_, dpotrf_, dgeqrf_, dorgqr_);
lap!(c32, cgetrf_, cgesv_, cpotrf_, cgeqrf_, cungqr_);
lap!(c64, zgetrf_, zgesv_, zpotrf_, zgeqrf_, zungqr_);

fn lu_check<T: Lap>(m: usize, n: usize) {
    let mut r = Rng::new((m * 31 + n) as u64);
    let ld = m + 2;
    let a0 = rand::<T>(&mut r, m, n, ld);
    let mut a = a0.clone();
    let k = m.min(n);
    let mut ipiv = vec![0 as Int; k];
    let mut info = -99;
    unsafe { T::getrf(&(m as Int), &(n as Int), a.as_mut_ptr(), &(ld as Int), ipiv.as_mut_ptr(), &mut info) };
    assert_eq!(info, 0);
    // L (m x k) * U (k x n), then undo swaps in reverse order.
    let l: Vec<T> = (0..m * k).map(|x| { let (i, j) = (x % m, x / m); if i == j { T::r(1.0) } else if i > j { at(&a, ld, i, j) } else { T::zero() } }).collect();
    let u: Vec<T> = (0..k * n).map(|x| { let (i, j) = (x % k, x / k); if i <= j { at(&a, ld, i, j) } else { T::zero() } }).collect();
    let mut p = mm(&l, m, &u, k, m, k, n);
    for i in (0..k).rev() {
        let q = (ipiv[i] - 1) as usize;
        assert!(q >= i && q < m);
        for j in 0..n { p.swap(i + j * m, q + j * m); }
    }
    for j in 0..n { for i in 0..m { assert!(close(p[i + j * m], at(&a0, ld, i, j), 4.0), "lu {m}x{n}"); } }
}

fn gesv_check<T: Lap>(n: usize) {
    let mut r = Rng::new(n as u64 + 7);
    let (ld, ldb, nr) = (n + 1, n + 3, 3);
    let a0 = rand::<T>(&mut r, n, n, ld);
    let b0 = rand::<T>(&mut r, n, nr, ldb);
    let (mut a, mut b) = (a0.clone(), b0.clone());
    let mut ipiv = vec![0; n];
    let mut info = -99;
    unsafe { T::gesv(&(n as Int), &(nr as Int), a.as_mut_ptr(), &(ld as Int), ipiv.as_mut_ptr(), b.as_mut_ptr(), &(ldb as Int), &mut info) };
    assert_eq!(info, 0);
    let ax = mm(&a0, ld, &b, ldb, n, n, nr);
    for j in 0..nr { for i in 0..n { assert!(close(ax[i + j * n], b0[i + j * ldb], 50.0), "gesv"); } }
    // singular: two equal rows -> info > 0
    let mut s = a0.clone();
    for j in 0..n { s[1 + j * ld] = s[j * ld]; }
    unsafe { T::gesv(&(n as Int), &(nr as Int), s.as_mut_ptr(), &(ld as Int), ipiv.as_mut_ptr(), b.as_mut_ptr(), &(ldb as Int), &mut info) };
    let _ = info; // exact-zero pivot is not guaranteed in floating point; zero matrix is:
    let mut z = vec![T::zero(); ld * n];
    unsafe { T::gesv(&(n as Int), &(nr as Int), z.as_mut_ptr(), &(ld as Int), ipiv.as_mut_ptr(), b.as_mut_ptr(), &(ldb as Int), &mut info) };
    assert_eq!(info, 1);
}

fn potrf_check<T: Lap>(n: usize, up: bool) {
    let mut r = Rng::new(n as u64 + 3);
    let ld = n + 2;
    let g = rand::<T>(&mut r, n, n, ld);
    // A = G G^H + n I
    let mut a0 = vec![T::zero(); ld * n];
    for i in 0..n { for j in 0..n { let mut s = T::zero(); for p in 0..n { s = s + at(&g, ld, i, p) * at(&g, ld, j, p).cj(); } a0[i + j * ld] = s + if i == j { T::r(n as f64) } else { T::zero() }; } }
    let mut a = a0.clone();
    let mut info = -9;
    let u = if up { b'U' } else { b'L' };
    unsafe { T::potrf(&u, &(n as Int), a.as_mut_ptr(), &(ld as Int), &mut info) };
    assert_eq!(info, 0);
    for i in 0..n { for j in 0..n {
        let mut s = T::zero();
        for p in 0..n {
            let (x, y) = if up {
                (if p <= i { at(&a, ld, p, i).cj() } else { T::zero() }, if p <= j { at(&a, ld, p, j) } else { T::zero() })
            } else {
                (if p <= i { at(&a, ld, i, p) } else { T::zero() }, if p <= j { at(&a, ld, j, p).cj() } else { T::zero() })
            };
            s = s + x * y;
        }
        assert!(close(s, a0[i + j * ld], 4.0 * n as f64), "potrf");
    } }
    let mut neg = a0.clone();
    neg[2 + 2 * ld] = T::r(-1000.0);
    unsafe { T::potrf(&u, &(n as Int), neg.as_mut_ptr(), &(ld as Int), &mut info) };
    assert_eq!(info, 3);
}

fn qr_check<T: Lap>(m: usize, n: usize) {
    let mut r = Rng::new((m * 13 + n) as u64);
    let ld = m + 1;
    let k = m.min(n);
    let a0 = rand::<T>(&mut r, m, n, ld);
    let mut a = a0.clone();
    let mut tau = vec![T::zero(); k.max(1)];
    let mut wq = T::zero();
    let mut info = -9;
    unsafe { T::geqrf(&(m as Int), &(n as Int), a.as_mut_ptr(), &(ld as Int), tau.as_mut_ptr(), &mut wq, &-1, &mut info) };
    assert_eq!(info, 0);
    let lw = wq.re() as usize;
    assert!(lw >= n.max(1));
    let mut w = vec![T::zero(); lw];
    unsafe { T::geqrf(&(m as Int), &(n as Int), a.as_mut_ptr(), &(ld as Int), tau.as_mut_ptr(), w.as_mut_ptr(), &(lw as Int), &mut info) };
    assert_eq!(info, 0);
    let rr: Vec<T> = (0..k * n).map(|x| { let (i, j) = (x % k, x / k); if i <= j { at(&a, ld, i, j) } else { T::zero() } }).collect();
    // complete Q (m x m) via orgqr on a copy padded to m columns
    let mc = m;
    let mut q = vec![T::zero(); ld * mc];
    for j in 0..k { for i in 0..m { q[i + j * ld] = a[i + j * ld]; } }
    unsafe { T::orgqr(&(m as Int), &(mc as Int), &(k as Int), q.as_mut_ptr(), &(ld as Int), tau.as_ptr(), w.as_mut_ptr(), &(lw.max(mc) as Int), &mut info) };
    assert_eq!(info, 0);
    let qr = mm(&q, ld, &rr, k, m, k, n);
    for j in 0..n { for i in 0..m { assert!(close(qr[i + j * m], at(&a0, ld, i, j), 4.0), "qr {m}x{n}"); } }
    for i in 0..mc { for j in 0..mc {
        let mut s = T::zero();
        for p in 0..m { s = s + at(&q, ld, p, i).cj() * at(&q, ld, p, j); }
        assert!(close(s, T::r(if i == j { 1.0 } else { 0.0 }), 2.0), "Q orth");
    } }
}

fn all<T: Lap>() {
    for &(m, n) in &[(1, 1), (5, 5), (7, 4), (4, 7), (20, 20), (33, 17)] { lu_check::<T>(m, n); qr_check::<T>(m, n); }
    for &n in &[1, 4, 16] { gesv_check::<T>(n); }
    for &n in &[4, 9] { potrf_check::<T>(n, false); potrf_check::<T>(n, true); }
}
#[test] fn lu_qr_chol_f32() { all::<f32>(); }
#[test] fn lu_qr_chol_f64() { all::<f64>(); }
#[test] fn lu_qr_chol_c32() { all::<c32>(); }
#[test] fn lu_qr_chol_c64() { all::<c64>(); }
