//! `xGETRF` (LU with partial pivoting) and `xGESV` (solve via LU).
use super::Sc;
use crate::ffi::{mat_mut, par, rd, Int};
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::lu::partial_pivoting::factor::{lu_in_place, lu_in_place_scratch};
use faer::linalg::triangular_solve::{solve_unit_lower_triangular_in_place, solve_upper_triangular_in_place};
use faer::{c32, c64};

/// LU of the `m x n` matrix at `a` in place; writes 1-based LAPACK pivots to `ipiv`.
/// Returns LAPACK info (0, or i>0 when U(i,i) is exactly zero).
pub(crate) unsafe fn getrf<T: Sc>(m: usize, n: usize, a: *mut T, lda: usize, ipiv: *mut Int) -> Int {
    let k = m.min(n);
    if k == 0 {
        return 0;
    }
    let mut mat = unsafe { mat_mut(a, m as Int, n as Int, lda as Int) };
    let mut fwd = vec![0usize; m];
    let mut bwd = vec![0usize; m];
    let p = par();
    let mut buf = MemBuffer::new(lu_in_place_scratch::<usize, T>(m, n, p, Default::default()));
    lu_in_place(mat.as_mut(), &mut fwd, &mut bwd, p, MemStack::new(&mut buf), Default::default());
    // Convert the permutation (row i of P*A = row fwd[i] of A) into a sequence of swaps.
    // at[r] = current position of original row r, who[q] = original row at position q.
    let mut at: Vec<usize> = (0..m).collect();
    let mut who: Vec<usize> = (0..m).collect();
    for i in 0..k {
        let q = at[fwd[i]];
        unsafe { *ipiv.add(i) = (q + 1) as Int };
        let (ri, rq) = (who[i], who[q]);
        who.swap(i, q);
        at[ri] = q;
        at[rq] = i;
    }
    let mut info = 0;
    for i in 0..k {
        if mat[(i, i)] == T::zero() {
            info = (i + 1) as Int;
            break;
        }
    }
    info
}

/// Apply LAPACK row interchanges `ipiv[0..k]` to the `n x nrhs` matrix `b` (xLASWP forward).
pub(crate) unsafe fn laswp<T: Sc>(b: *mut T, ldb: usize, nrhs: usize, k: usize, ipiv: *const Int) {
    for i in 0..k {
        let q = (unsafe { *ipiv.add(i) } - 1) as usize;
        if q != i {
            for j in 0..nrhs {
                unsafe { core::ptr::swap(b.add(i + j * ldb), b.add(q + j * ldb)) };
            }
        }
    }
}

pub(crate) unsafe fn gesv<T: Sc>(n: Int, nrhs: Int, a: *mut T, lda: Int, ipiv: *mut Int, b: *mut T, ldb: Int) -> Int {
    if n < 0 {
        return -1;
    }
    if nrhs < 0 {
        return -2;
    }
    if lda < n.max(1) {
        return -4;
    }
    if ldb < n.max(1) {
        return -7;
    }
    let (nu, ru) = (n as usize, nrhs as usize);
    let info = unsafe { getrf(nu, nu, a, lda as usize, ipiv) };
    if info != 0 || nu == 0 || ru == 0 {
        return info;
    }
    unsafe { laswp(b, ldb as usize, ru, nu, ipiv) };
    let lu = unsafe { mat_mut(a, n, n, lda) };
    let mut bm = unsafe { mat_mut(b, n, nrhs, ldb) };
    solve_unit_lower_triangular_in_place(lu.rb(), bm.as_mut(), par());
    solve_upper_triangular_in_place(lu.rb(), bm.as_mut(), par());
    0
}

unsafe fn getrf_c<T: Sc>(m: Int, n: Int, a: *mut T, lda: Int, ipiv: *mut Int) -> Int {
    if m < 0 {
        -1
    } else if n < 0 {
        -2
    } else if lda < m.max(1) {
        -4
    } else {
        unsafe { getrf(m as usize, n as usize, a, lda as usize, ipiv) }
    }
}

macro_rules! export_lu {
    ($t:ty, $getrf:ident, $gesv:ident) => {
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $getrf(m: *const Int, n: *const Int, a: *mut $t, lda: *const Int, ipiv: *mut Int, info: *mut Int) {
            unsafe { *info = getrf_c(rd(m), rd(n), a, rd(lda), ipiv) };
        }
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $gesv(
            n: *const Int, nrhs: *const Int, a: *mut $t, lda: *const Int, ipiv: *mut Int,
            b: *mut $t, ldb: *const Int, info: *mut Int,
        ) {
            unsafe { *info = gesv(rd(n), rd(nrhs), a, rd(lda), ipiv, b, rd(ldb)) };
        }
    };
}
export_lu!(f32, sgetrf_, sgesv_);
export_lu!(f64, dgetrf_, dgesv_);
export_lu!(c32, cgetrf_, cgesv_);
export_lu!(c64, zgetrf_, zgesv_);

#[allow(unused_imports)]
use faer::reborrow::Reborrow as _;
