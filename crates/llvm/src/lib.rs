//! typhon-llvm: LLVM backend infrastructure.
//!
//! Ported from:
//! - `codon/cir/llvm/optimize.{h,cpp}` -> [`target`], [`optimize`]
//! - `codon/cir/llvm/gpu.{h,cpp}` -> [`gpu`]
//! - `codon/cir/llvm/native/**` -> [`native`]
//! - `codon/compiler/engine.{h,cpp}` -> [`engine`]
//! - `codon/compiler/memory_manager.{h,cpp}` -> [`memory_manager`]
//! - `codon/compiler/debug_listener.{h,cpp}` -> [`debug_listener`]
//!
//! See `API.md` for the public surface used by the IR->LLVM visitor and the JIT.

pub(crate) mod analysis;
pub mod debug_listener;
mod eh_frames;
pub mod engine;
pub(crate) mod ffi;
pub mod gpu;
pub mod llvisitor;
pub mod memory_manager;
pub mod native;
pub mod optimize;
pub mod target;

pub use inkwell;
pub use llvm_sys;

pub use debug_listener::{DebugListener, LineInfo};
pub use engine::{Engine, EngineConfig, GcRootCallbacks, ResourceTracker};
pub use llvisitor::{CodegenBuilder, LlvmVisitor, TypeBuilder, ValueBuilder};
pub use optimize::{GLOBAL_CTOR_NAME, PassStage, optimize, verify};
pub use target::{
    data_layout_string, features_str, host_target_machine, host_triple, initialize_targets, target_machine_for,
    target_machine_for_module,
};

#[doc(hidden)]
pub mod test_support;
