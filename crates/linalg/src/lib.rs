//! typhon-linalg: pure-Rust BLAS/LAPACK backend for `numpy.linalg`.
//!
//! Exports the CBLAS (`cblas_*`) and Fortran LAPACK (`xxxxx_`) symbols used by
//! `stdlib/numpy/linalg/blas.py`, with ILP64 (`i64`) integers, implemented on faer.
pub mod ffi;
pub mod blas;
pub mod lapack;
