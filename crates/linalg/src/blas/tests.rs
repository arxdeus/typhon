//! Tests of the cblas exports against naive references.
use super::exports::*;
use super::Scalar;
use crate::ffi::{Int, c32, c64};

struct Rng(u64);
impl Rng {
    fn f(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }
    fn n(&mut self, k: Int) -> Int {
        ((self.f() + 1.0) * 0.5 * k as f64) as Int % k
    }
}

trait T: Scalar {
    fn rand(r: &mut Rng) -> Self;
    fn dist(a: Self, b: Self) -> f64;
    const TOL: f64;
}
impl T for f32 { fn rand(r: &mut Rng) -> Self { r.f() as f32 } fn dist(a: Self, b: Self) -> f64 { (a - b).abs() as f64 } const TOL: f64 = 1e-4; }
impl T for f64 { fn rand(r: &mut Rng) -> Self { r.f() } fn dist(a: Self, b: Self) -> f64 { (a - b).abs() } const TOL: f64 = 1e-11; }
impl T for c32 { fn rand(r: &mut Rng) -> Self { c32::new(r.f() as f32, r.f() as f32) } fn dist(a: Self, b: Self) -> f64 { (a - b).norm() as f64 } const TOL: f64 = 1e-4; }
impl T for c64 { fn rand(r: &mut Rng) -> Self { c64::new(r.f(), r.f()) } fn dist(a: Self, b: Self) -> f64 { (a - b).norm() } const TOL: f64 = 1e-11; }

fn rv<X: T>(r: &mut Rng, n: usize) -> Vec<X> { (0..n).map(|_| X::rand(r)).collect() }
fn at(row: bool, ld: Int, i: Int, j: Int) -> usize { (if row { i * ld + j } else { i + j * ld }) as usize }
fn opel<X: T>(a: &[X], row: bool, ld: Int, tr: Int, i: Int, j: Int) -> X {
    match tr { 111 => a[at(row, ld, i, j)], 112 => a[at(row, ld, j, i)], _ => a[at(row, ld, j, i)].cj() }
}
fn close<X: T>(a: &[X], b: &[X], scale: f64) {
    for (k, (x, y)) in a.iter().zip(b).enumerate() {
        assert!(X::dist(*x, *y) <= X::TOL * scale, "idx {k}: {x:?} vs {y:?}");
    }
}

type GemmF<X, S> = unsafe extern "C" fn(Int, Int, Int, Int, Int, Int, S, *const X, Int, *const X, Int, S, *mut X, Int);

fn check_gemm<X: T, S: Copy>(f: GemmF<X, S>, wrap: fn(&X) -> S) {
    let mut r = Rng(42);
    let trs: &[Int] = if X::IS_COMPLEX { &[111, 112, 113] } else { &[111, 112] };
    for it in 0..60 {
        let (m, n, k) = (1 + r.n(17), 1 + r.n(17), if it == 0 { 0 } else { 1 + r.n(17) });
        for &row in &[false, true] { for &ta in trs { for &tb in trs {
            let (ar, ac) = if ta == 111 { (m, k) } else { (k, m) };
            let (br, bc) = if tb == 111 { (k, n) } else { (n, k) };
            let lda = (if row { ac } else { ar }).max(1) + r.n(3);
            let ldb = (if row { bc } else { br }).max(1) + r.n(3);
            let ldc = (if row { n } else { m }) + r.n(3);
            let a: Vec<X> = rv(&mut r, (lda * if row { ar } else { ac }).max(1) as usize);
            let b: Vec<X> = rv(&mut r, (ldb * if row { br } else { bc }).max(1) as usize);
            let c0: Vec<X> = rv(&mut r, (ldc * if row { m } else { n }) as usize);
            let (al, be) = (X::rand(&mut r), if it % 3 == 0 { X::zero() } else { X::rand(&mut r) });
            let mut want = c0.clone();
            for i in 0..m { for j in 0..n {
                let mut s = X::zero();
                for p in 0..k { s = s.sadd(opel(&a, row, lda, ta, i, p).smul(opel(&b, row, ldb, tb, p, j))); }
                let idx = at(row, ldc, i, j);
                let old = if be == X::zero() { X::zero() } else { be.smul(c0[idx]) };
                want[idx] = al.smul(s).sadd(old);
            }}
            let mut got = c0.clone();
            unsafe { f(if row { 101 } else { 102 }, ta, tb, m, n, k, wrap(&al), a.as_ptr(), lda, b.as_ptr(), ldb, wrap(&be), got.as_mut_ptr(), ldc) };
            close(&got, &want, (k + 1) as f64);
        }}}
    }
}

#[test] fn sgemm() { check_gemm::<f32, f32>(cblas_sgemm, |x| *x) }
#[test] fn dgemm() { check_gemm::<f64, f64>(cblas_dgemm, |x| *x) }
#[test] fn cgemm() { check_gemm::<c32, *const c32>(cblas_cgemm, |x| x as *const c32) }
#[test] fn zgemm() { check_gemm::<c64, *const c64>(cblas_zgemm, |x| x as *const c64) }

type GemvF<X, S> = unsafe extern "C" fn(Int, Int, Int, Int, S, *const X, Int, *const X, Int, S, *mut X, Int);

fn vidx(n: Int, inc: Int, i: Int) -> usize { (if inc < 0 { (n - 1 - i) * -inc } else { i * inc }) as usize }

fn check_gemv<X: T, S: Copy>(f: GemvF<X, S>, wrap: fn(&X) -> S) {
    let mut r = Rng(7);
    let trs: &[Int] = if X::IS_COMPLEX { &[111, 112, 113] } else { &[111, 112] };
    for _ in 0..60 {
        let (m, n) = (1 + r.n(19), 1 + r.n(19));
        for &row in &[false, true] { for &tr in trs {
            let lda = (if row { n } else { m }) + r.n(3);
            let a: Vec<X> = rv(&mut r, (lda * if row { m } else { n }) as usize);
            let (xl, yl) = if tr == 111 { (n, m) } else { (m, n) };
            let incx: Int = [1, 2, -1, -3][r.n(4) as usize];
            let incy: Int = [1, 3, -1, -2][r.n(4) as usize];
            let x: Vec<X> = rv(&mut r, (xl * incx.abs()) as usize);
            let y0: Vec<X> = rv(&mut r, (yl * incy.abs()) as usize);
            let (al, be) = (X::rand(&mut r), X::rand(&mut r));
            let mut want = y0.clone();
            for i in 0..yl {
                let mut s = X::zero();
                for p in 0..xl {
                    let aij = if tr == 111 { a[at(row, lda, i, p)] } else { opel(&a, row, lda, tr, i, p) };
                    s = s.sadd(aij.smul(x[vidx(xl, incx, p)]));
                }
                let id = vidx(yl, incy, i);
                want[id] = al.smul(s).sadd(be.smul(y0[id]));
            }
            let mut got = y0.clone();
            unsafe { f(if row { 101 } else { 102 }, tr, m, n, wrap(&al), a.as_ptr(), lda, x.as_ptr(), incx, wrap(&be), got.as_mut_ptr(), incy) };
            close(&got, &want, (xl + 1) as f64);
        }}
    }
}

#[test] fn sgemv() { check_gemv::<f32, f32>(cblas_sgemv, |x| *x) }
#[test] fn dgemv() { check_gemv::<f64, f64>(cblas_dgemv, |x| *x) }
#[test] fn cgemv() { check_gemv::<c32, *const c32>(cblas_cgemv, |x| x as *const c32) }
#[test] fn zgemv() { check_gemv::<c64, *const c64>(cblas_zgemv, |x| x as *const c64) }

type SyrkF<X, S> = unsafe extern "C" fn(Int, Int, Int, Int, Int, S, *const X, Int, S, *mut X, Int);

fn check_syrk<X: T, S: Copy>(f: SyrkF<X, S>, wrap: fn(&X) -> S) {
    let mut r = Rng(3);
    for _ in 0..60 {
        let (n, k) = (1 + r.n(15), 1 + r.n(15));
        for &row in &[false, true] { for &up in &[121, 122] { for &tr in &[111, 112] {
            let (ar, ac) = if tr == 111 { (n, k) } else { (k, n) };
            let lda = (if row { ac } else { ar }) + r.n(3);
            let ldc = n + r.n(3);
            let a: Vec<X> = rv(&mut r, (lda * if row { ar } else { ac }) as usize);
            let c0: Vec<X> = rv(&mut r, (ldc * n) as usize);
            let (al, be) = (X::rand(&mut r), X::rand(&mut r));
            let mut want = c0.clone();
            for i in 0..n { for j in 0..n {
                if (up == 121 && i > j) || (up == 122 && i < j) { continue; }
                let mut s = X::zero();
                for p in 0..k { s = s.sadd(opel(&a, row, lda, tr, i, p).smul(opel(&a, row, lda, tr, j, p))); }
                let id = at(row, ldc, i, j);
                want[id] = al.smul(s).sadd(be.smul(c0[id]));
            }}
            let mut got = c0.clone();
            unsafe { f(if row { 101 } else { 102 }, up, tr, n, k, wrap(&al), a.as_ptr(), lda, wrap(&be), got.as_mut_ptr(), ldc) };
            close(&got, &want, (k + 1) as f64);
        }}}
    }
}

#[test] fn ssyrk() { check_syrk::<f32, f32>(cblas_ssyrk, |x| *x) }
#[test] fn dsyrk() { check_syrk::<f64, f64>(cblas_dsyrk, |x| *x) }
#[test] fn csyrk() { check_syrk::<c32, *const c32>(cblas_csyrk, |x| x as *const c32) }
#[test] fn zsyrk() { check_syrk::<c64, *const c64>(cblas_zsyrk, |x| x as *const c64) }

fn check_copy_dot<X: T>(cp: unsafe extern "C" fn(Int, *const X, Int, *mut X, Int), dot: &dyn Fn(Int, &[X], Int, &[X], Int, bool) -> X) {
    let mut r = Rng(11);
    for _ in 0..50 {
        let n = r.n(20);
        let incx: Int = [1, 2, -1, -2][r.n(4) as usize];
        let incy: Int = [1, 3, -1, -3][r.n(4) as usize];
        let x: Vec<X> = rv(&mut r, (n * incx.abs()).max(1) as usize);
        let y: Vec<X> = rv(&mut r, (n * incy.abs()).max(1) as usize);
        let mut got = y.clone();
        unsafe { cp(n, x.as_ptr(), incx, got.as_mut_ptr(), incy) };
        let mut want = y.clone();
        for i in 0..n { want[vidx(n, incy, i)] = x[vidx(n, incx, i)]; }
        close(&got, &want, 1.0);
        for conj in [false, true] {
            if conj && !X::IS_COMPLEX { continue; }
            let mut s = X::zero();
            for i in 0..n {
                let a = x[vidx(n, incx, i)];
                s = s.sadd(if conj { a.cj() } else { a }.smul(y[vidx(n, incy, i)]));
            }
            close(&[dot(n, &x, incx, &y, incy, conj)], &[s], (n + 1) as f64);
        }
    }
}

#[test] fn s_copy_dot() { check_copy_dot::<f32>(cblas_scopy, &|n, x, ix, y, iy, _| unsafe { cblas_sdot(n, x.as_ptr(), ix, y.as_ptr(), iy) }) }
#[test] fn d_copy_dot() { check_copy_dot::<f64>(cblas_dcopy, &|n, x, ix, y, iy, _| unsafe { cblas_ddot(n, x.as_ptr(), ix, y.as_ptr(), iy) }) }
#[test] fn c_copy_dot() {
    check_copy_dot::<c32>(cblas_ccopy, &|n, x, ix, y, iy, c| {
        let mut o = c32::new(0.0, 0.0);
        unsafe { if c { cblas_cdotc_sub(n, x.as_ptr(), ix, y.as_ptr(), iy, &mut o) } else { cblas_cdotu_sub(n, x.as_ptr(), ix, y.as_ptr(), iy, &mut o) } };
        o
    })
}
#[test] fn z_copy_dot() {
    check_copy_dot::<c64>(cblas_zcopy, &|n, x, ix, y, iy, c| {
        let mut o = c64::new(0.0, 0.0);
        unsafe { if c { cblas_zdotc_sub(n, x.as_ptr(), ix, y.as_ptr(), iy, &mut o) } else { cblas_zdotu_sub(n, x.as_ptr(), ix, y.as_ptr(), iy, &mut o) } };
        o
    })
}
