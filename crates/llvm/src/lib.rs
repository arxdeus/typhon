//! typhon-llvm: LLVM backend infrastructure.
//!
//! Modules: [`target`] and [`optimize`] (pass pipeline), [`gpu`], [`native`].
//!
//! See `API.md` for the public surface used by the IR->LLVM visitor. Output is AOT only (object/exe/lib).

pub(crate) mod analysis;
pub(crate) mod ffi;
pub mod gpu;
pub mod llvisitor;
pub mod native;
pub mod optimize;
pub mod target;

pub use inkwell;
pub use llvm_sys;

pub use llvisitor::LlvmVisitor;
pub use optimize::{GLOBAL_CTOR_NAME, PassStage, optimize, verify};
pub use target::{
    data_layout_string, features_str, host_target_machine, host_triple, initialize_targets, target_machine_for,
    target_machine_for_module,
};

#[doc(hidden)]
pub mod test_support;
