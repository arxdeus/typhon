//! The Typhon LLVM pass pipeline.
//!
//! # Pass ordering relative to the standard pipeline
//!
//! Custom passes would normally hook into `PassBuilder` extension points. The LLVM C API
//! can only run textual pipelines (`LLVMRunPasses`), so each `runLLVMOptimizationPasses`
//! invocation is emulated as:
//!
//! 1. `applyDebugTransformations`, `applyFastMathTransformations` (as in C++);
//! 2. **PipelineStart / EarlySimplification** stage: `ArchNativePass` (native),
//!    `CoroResumeInliner` (marking + `always-inline`);
//! 3. **LateLoopOptimizations** stage: `CoroUnrollControl`, `CoroBranchSimplifier`;
//! 4. **Peephole** stage: `OpenMPThreadIdOptimizer`, `AllocationRemover`,
//!    `loop-simplify,lcssa`, `AllocationHoister`, `AllocationAutoFree` (if
//!    `autofree`);
//! 5. `default<O3>` (or `default<O0>` in debug mode) via `LLVMRunPasses`;
//! 6. **Peephole (late)**: the peephole group is run again after the standard
//!    pipeline followed by `instcombine,simplifycfg` — this approximates the
//!    C++ placement inside the pipeline (where the peephole EP runs after every
//!    InstCombine), so allocations exposed by inlining are still demoted;
//! 7. `CoroUnrollControl::cleanup`, `applyDebugTransformations`.
//!
//! Every custom pass therefore runs at most twice per pipeline
//! invocation rather than at each extension point instance inside the O3
//! pipeline; `optimize()` still runs the whole pipeline twice (opt1/opt2) .

pub mod alloc;
pub mod coro;

use inkwell::module::Module;
use llvm_sys::prelude::*;
use llvm_sys::target_machine::LLVMTargetMachineRef;
use typhon_common::{CompilerError, Options, Result, Timer};

use crate::{ffi, gpu, native, target};

pub use coro::GLOBAL_CTOR_NAME;

/// Public entry points to the individual passes.
pub mod passes {
    pub use super::alloc::{
        AllocInfo, allocation_auto_free, allocation_hoister, allocation_hoister_with, allocation_remover,
        allocation_remover_with,
    };
    pub use super::coro::{
        apply_debug_transformations, apply_fast_math_transformations, coro_branch_simplifier, coro_resume_inliner_mark,
        coro_unroll_cleanup, coro_unroll_control, openmp_thread_id_optimizer,
    };
}

/// The `PassBuilder` extension points at which Typhon
/// passes are registered.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PassStage {
    PipelineStart,
    PipelineEarlySimplification,
    LateLoopOptimizations,
    LoopOptimizerEnd,
    ScalarOptimizerLate,
    Peephole,
    VectorizerStart,
    OptimizerEarly,
    OptimizerLast,
}

/// Run a textual new-PM pipeline over a raw module.
pub(crate) fn run_pipeline(m: LLVMModuleRef, passes: &str, tm: LLVMTargetMachineRef) -> Result<()> {
    unsafe {
        let opts = llvm_sys::transforms::pass_builder::LLVMCreatePassBuilderOptions();
        let p = ffi::cstr(passes);
        let e = llvm_sys::transforms::pass_builder::LLVMRunPasses(m, p.as_ptr(), tm, opts);
        llvm_sys::transforms::pass_builder::LLVMDisposePassBuilderOptions(opts);
        match ffi::take_error(e) {
            None => Ok(()),
            Some(msg) => Err(CompilerError::Llvm(format!("pipeline '{passes}': {msg}"))),
        }
    }
}

/// Run a textual pipeline over a module.
pub fn run_passes(module: &Module, passes: &str, tm: Option<&inkwell::targets::TargetMachine>) -> Result<()> {
    run_pipeline(module.as_mut_ptr(), passes, tm.map_or(std::ptr::null_mut(), |t| t.as_mut_ptr()))
}

fn for_each_definition(m: LLVMModuleRef, mut f: impl FnMut(ffi::V) -> bool) -> bool {
    let mut changed = false;
    for func in ffi::functions(m) {
        if !ffi::is_declaration(func) {
            changed |= f(func);
        }
    }
    changed
}

/// The peephole group (`registerPeepholeEPCallback`).
fn peephole(m: LLVMModuleRef, options: &Options, tm: LLVMTargetMachineRef) -> Result<()> {
    for_each_definition(m, coro::openmp_thread_id_optimizer);
    for_each_definition(m, alloc::allocation_remover);
    run_pipeline(m, "function(loop-simplify,lcssa)", tm)?;
    for_each_definition(m, alloc::allocation_hoister);
    if options.autofree {
        for_each_definition(m, alloc::allocation_auto_free);
    }
    Ok(())
}

/// Port of `runLLVMOptimizationPasses`.
fn run_llvm_optimization_passes(module: &Module, options: &Options) -> Result<()> {
    let m = module.as_mut_ptr();
    coro::apply_debug_transformations(m, options.debug);
    if options.fastmath {
        coro::apply_fast_math_transformations(m);
    }
    let machine = if options.native { target::target_machine_for_module(module, options, true, false)? } else { None };
    let tm = machine.as_ref().map_or(std::ptr::null_mut(), |t| t.as_mut_ptr());
    let debug = options.debug;
    // `opt.isOptimizingForSpeed()` is false for O0 (debug).
    let speed = !debug;

    // PipelineStart / EarlySimplification
    if options.native {
        native::add_native_llvm_passes(module);
    }
    if speed {
        let marked = coro::coro_resume_inliner_mark(m);
        if !marked.is_empty() {
            run_pipeline(m, "always-inline", tm)?;
            // Unmark any surviving call sites (inlining failed).
            let k = ffi::attr_kind("alwaysinline");
            for f in ffi::functions(m) {
                for i in ffi::fn_instructions(f) {
                    if ffi::is_call(i) {
                        unsafe { llvm_sys::core::LLVMRemoveCallSiteEnumAttribute(i, ffi::FN_INDEX, k) };
                    }
                }
            }
        }
    }

    // LateLoopOptimizations
    if speed {
        for_each_definition(m, coro::coro_unroll_control);
        for_each_definition(m, coro::coro_branch_simplifier);
    }

    // Peephole (early)
    if speed {
        peephole(m, options, tm)?;
    }

    // Standard pipeline
    run_pipeline(m, if debug { "default<O0>" } else { "default<O3>" }, tm)?;

    // Peephole (late) + cleanup
    if speed {
        for_each_definition(m, coro::coro_branch_simplifier);
        peephole(m, options, tm)?;
        // `instcombine` named directly verifies a 1-iteration fixpoint and aborts
        // (O3's internal instances don't); this extra run is a port approximation.
        run_pipeline(m, "function(instcombine<no-verify-fixpoint>,simplifycfg)", tm)?;
    }

    coro::coro_unroll_cleanup(m);
    coro::apply_debug_transformations(m, options.debug);
    Ok(())
}

/// Port of `verify()`: on failure dumps the module to `_dump.ll` and errors.
pub fn verify(module: &Module) -> Result<()> {
    if let Err(e) = module.verify() {
        let _ = module.print_to_file("_dump.ll");
        return Err(CompilerError::Llvm(format!(
            "Generated LLVM IR is invalid and has been dumped to '_dump.ll'. Please submit a bug report at \
             https://github.com/typhon-lang/typhon including the code and generated LLVM IR.\n{e}"
        )));
    }
    Ok(())
}

/// Port of `ir::optimize(module, options)`.
pub fn optimize(module: &Module, options: &Options) -> Result<()> {
    verify(module)?;
    let gpu_module = {
        let _t = Timer::new("preparing/gpu");
        gpu::prepare_gpu_module(module, options)?
    };
    {
        let _t = Timer::new("llvm/opt1");
        run_llvm_optimization_passes(module, options)?;
    }
    if !options.debug {
        let _t = Timer::new("llvm/opt2");
        run_llvm_optimization_passes(module, options)?;
    }
    if let Some(g) = gpu_module {
        let mut gpu_options = options.clone();
        gpu_options.native = false;
        {
            let _t = Timer::new("llvm/gpuopt1");
            run_llvm_optimization_passes(&g, &gpu_options)?;
        }
        {
            let _t = Timer::new("llvm/gpuopt2");
            run_llvm_optimization_passes(&g, &gpu_options)?;
        }
        let _t = Timer::new("llvm/gpu");
        gpu::apply_gpu_transformations(module, g, options, "")?;
    }
    verify(module)
}
