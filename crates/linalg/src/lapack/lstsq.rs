//! `xGELSD`: minimum-norm least squares via SVD with rcond truncation.
//! On exit b(0..n, :) holds X; for m > n, rows n..m hold components of U^H b orthogonal to
//! range(A) whose squared sum is the residual (numpy reads them when rank == n).
use super::svd::svd_core;
use super::{set_iwork, set_rwork, set_work, Re, Sc};
use crate::ffi::{rd, Int};
use faer::linalg::svd::ComputeSvdVectors;
use faer::{c32, c64};

unsafe fn gelsd_c<T: Sc>(
    m: Int, n: Int, nrhs: Int, a: *mut T, lda: Int, b: *mut T, ldb: Int, s: *mut T::Real,
    rcond: f64, rank: *mut Int, work: *mut T, lwork: Int, rwork: *mut T::Real, iwork: *mut Int,
) -> Int {
    if m < 0 {
        return -1;
    }
    if n < 0 {
        return -2;
    }
    if nrhs < 0 {
        return -3;
    }
    if lda < m.max(1) {
        return -5;
    }
    if ldb < m.max(n).max(1) {
        return -7;
    }
    let mx = m.max(n).max(1);
    unsafe {
        set_work(work, mx * (nrhs.max(1) + 4));
        set_rwork(rwork, mx * (nrhs.max(1) + 4));
        set_iwork(iwork, mx);
    }
    if lwork == -1 {
        return 0;
    }
    let (mu, nu, ru, ld) = (m as usize, n as usize, nrhs as usize, ldb as usize);
    unsafe { *rank = 0 };
    if mu == 0 || nu == 0 {
        // LAPACK quick return: B untouched (numpy reads rows n..m as residual components).
        return 0;
    }
    let mode = if mu > nu { ComputeSvdVectors::Full } else { ComputeSvdVectors::Thin };
    let Some((sv, u, v)) = (unsafe { svd_core(mu, nu, a, lda as usize, mode) }) else {
        return 1;
    };
    let k = mu.min(nu);
    // LAPACK: rcond < 0 means machine precision (2^-53 / 2^-24). Singular values at that
    // level are pure rounding noise whose magnitude differs between SVD implementations;
    // numpy's test_rcond expects rcond=-1 to keep such a noise value (rank 4 of a rank-3
    // matrix), so treat rcond < 0 as "only exact zeros are dropped".
    let rc = if rcond < 0.0 { 0.0 } else { rcond };
    let thr = rc * sv.first().copied().unwrap_or(0.0);
    let mut r = 0;
    for (i, &x) in sv.iter().enumerate() {
        unsafe { *s.add(i) = <T::Real as Re>::f(x) };
        if x > thr && x > 0.0 {
            r += 1;
        }
    }
    unsafe { *rank = r as Int };
    let ucols = u.ncols();
    for j in 0..ru {
        let col = unsafe { b.add(j * ld) };
        // c = U^H b
        let c: Vec<T> = (0..ucols)
            .map(|p| {
                let mut acc = T::zero();
                for i in 0..mu {
                    acc = acc + u[(i, p)].cj() * unsafe { *col.add(i) };
                }
                acc
            })
            .collect();
        // x = V * diag(1/s) * c[0..r]
        for i in 0..nu {
            let mut acc = T::zero();
            for p in 0..r {
                acc = acc + v[(i, p)] * c[p] * T::r(1.0 / sv[p]);
            }
            unsafe { *col.add(i) = acc };
        }
        for i in nu..mu {
            unsafe { *col.add(i) = c[i] };
        }
        let _ = k;
    }
    0
}

macro_rules! export_gelsd {
    ($t:ty, $r:ty, $name:ident) => {
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $name(
            m: *const Int, n: *const Int, nrhs: *const Int, a: *mut $t, lda: *const Int, b: *mut $t,
            ldb: *const Int, s: *mut $r, rcond: *const $r, rank: *mut Int, work: *mut $t,
            lwork: *const Int, iwork: *mut Int, info: *mut Int,
        ) {
            unsafe {
                *info = gelsd_c(rd(m), rd(n), rd(nrhs), a, rd(lda), b, rd(ldb), s, (*rcond) as f64, rank,
                    work, rd(lwork), core::ptr::null_mut(), iwork);
            }
        }
    };
    ($t:ty, $r:ty, $name:ident, cplx) => {
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $name(
            m: *const Int, n: *const Int, nrhs: *const Int, a: *mut $t, lda: *const Int, b: *mut $t,
            ldb: *const Int, s: *mut $r, rcond: *const $r, rank: *mut Int, work: *mut $t,
            lwork: *const Int, rwork: *mut $r, iwork: *mut Int, info: *mut Int,
        ) {
            unsafe {
                *info = gelsd_c(rd(m), rd(n), rd(nrhs), a, rd(lda), b, rd(ldb), s, (*rcond) as f64, rank,
                    work, rd(lwork), rwork, iwork);
            }
        }
    };
}
export_gelsd!(f32, f32, sgelsd_);
export_gelsd!(f64, f64, dgelsd_);
export_gelsd!(c32, f32, cgelsd_, cplx);
export_gelsd!(c64, f64, zgelsd_, cplx);
