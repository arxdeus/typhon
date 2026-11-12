//! OpenMP lowering. The schedule type lives in `typhon_ir::schedule`.

pub mod openmp;

pub use openmp::OpenMPPass;
