//! Typhon IR transformations.
//!
//! Modules: `manager` (PassManager, standard pipeline), `cleanup`, `folding`, `lowering`,
//! `pythonic`, `numpy` and `parallel` (OpenMP).
//!
//! Shared pass traits (`Pass`, `PassGroup`, `Analysis`, `PassContext`), `OMPSched`
//! and `Rewriter` live in `typhon_ir::{pass, schedule, rewrite}`.

pub mod cleanup;
pub mod folding;
pub mod lowering;
pub mod manager;
pub mod numpy;
pub mod parallel;
pub mod pythonic;
