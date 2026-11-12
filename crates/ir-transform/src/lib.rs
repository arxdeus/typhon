//! Typhon IR transformations.
//!
//! Module ownership (one lane each, see ../../DESIGN.md):
//! - `manager`                      : transform/manager.{h,cpp} (PassManager, standard pipeline)
//! - `cleanup`, `folding`           : transform/cleanup/*, transform/folding/*
//! - `lowering`, `pythonic`         : transform/lowering/*, transform/pythonic/*
//! - `numpy`                        : transform/numpy/*
//! - `parallel`                     : transform/parallel/openmp.*
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
