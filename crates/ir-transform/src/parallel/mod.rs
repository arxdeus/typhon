//! Port of codon/cir/transform/parallel/openmp.{h,cpp} (openmp lane)
//! (`schedule.{h,cpp}` lives in `typhon_ir::schedule`.)

pub mod openmp;

pub use openmp::OpenMPPass;
