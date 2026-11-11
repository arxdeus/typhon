//! Port of `codon/compiler/engine.{h,cpp}`: the ORC LLJIT engine.
//!
//! Layering (all through the LLVM C API):
//!
//! ```text
//! IRTransformLayer  -- optional `optimize()` (C++ always optimizes here)
//!   IRCompileLayer
//!     ObjectTransformLayer -- captures object bytes for the DebugListener
//!       RTDyldObjectLinkingLayer -- MCJIT-like memory manager callbacks
//!                                   (GC root registration, see memory_manager.rs)
//! main JITDylib + DynamicLibrarySearchGenerator(current process)
//! ```
//!
//! PORT-NOTE: C++ uses a JITLink `ObjectLinkingLayer` with
//! `BoehmGCJITLinkMemoryManager`, `DebugObjectManagerPlugin`(GDB) and
//! `DebugPlugin`. The C API can only build an RTDyld linking layer with custom
//! memory-manager callbacks; GDB registration uses
//! `LLVMCreateGDBRegistrationListener`, and debug symbolization uses the
//! object-transform hook + `DebugListener`.

use std::ffi::CStr;
use std::sync::Arc;

use inkwell::module::Module;
use llvm_sys::error::LLVMErrorRef;
use llvm_sys::orc2::ee::*;
use llvm_sys::orc2::lljit::*;
use llvm_sys::orc2::*;
use llvm_sys::prelude::*;
use typhon_common::{CompilerError, Options, Result};

use crate::debug_listener::DebugListener;
use crate::ffi;
pub use crate::memory_manager::GcRootCallbacks;
use crate::memory_manager::{self as mm, Shared};

/// Engine configuration beyond `Options`.
#[derive(Clone, Copy, Default)]
pub struct EngineConfig {
    /// `seq_gc_add_roots` / `seq_gc_remove_roots` from the runtime. When `None`,
    /// writable JIT sections are not registered with any GC.
    pub gc_roots: Option<GcRootCallbacks>,
    /// Run `optimize(module, options)` in the IR transform layer (C++ behaviour).
    pub optimize_on_add: bool,
}

/// A removable group of JIT definitions (`llvm::orc::ResourceTracker`).
pub struct ResourceTracker {
    rt: LLVMOrcResourceTrackerRef,
}

impl ResourceTracker {
    /// Remove all code/data added through this tracker.
    pub fn remove(self) -> Result<()> {
        let e = unsafe { LLVMOrcResourceTrackerRemove(self.rt) };
        let r = check(e);
        unsafe { LLVMOrcReleaseResourceTracker(self.rt) };
        std::mem::forget(self);
        r
    }
}

impl Drop for ResourceTracker {
    fn drop(&mut self) {
        unsafe { LLVMOrcReleaseResourceTracker(self.rt) };
    }
}

fn check(e: LLVMErrorRef) -> Result<()> {
    match ffi::take_error(e) {
        None => Ok(()),
        Some(m) => Err(CompilerError::Llvm(m)),
    }
}

struct TransformCtx {
    options: Options,
}

/// Port of `codon::jit::Engine`.
pub struct Engine {
    jit: LLVMOrcLLJITRef,
    shared: Arc<Shared>,
    debug: Arc<DebugListener>,
    transform: Box<TransformCtx>,
    options: Options,
    // ORC is internally synchronized but our callbacks hold raw pointers; keep !Send.
    _not_send: std::marker::PhantomData<*mut ()>,
}

extern "C" fn create_object_layer(
    ctx: *mut libc::c_void,
    es: LLVMOrcExecutionSessionRef,
    _triple: *const libc::c_char,
) -> LLVMOrcObjectLayerRef {
    unsafe {
        let layer = LLVMOrcCreateRTDyldObjectLinkingLayerWithMCJITMemoryManagerLikeCallbacks(
            es,
            ctx,
            mm::create_context,
            mm::notify_terminating,
            mm::allocate_code,
            mm::allocate_data,
            mm::finalize,
            Some(mm::destroy),
        );
        // GDB JIT interface (C++: createJITLoaderGDBRegistrar + DebugObjectManagerPlugin).
        let gdb = llvm_sys::execution_engine::LLVMCreateGDBRegistrationListener();
        if !gdb.is_null() {
            LLVMOrcRTDyldObjectLinkingLayerRegisterJITEventListener(layer, gdb);
        }
        layer
    }
}

/// Object transform: copy the object bytes for the debug listener and pass the
/// buffer through unchanged.
extern "C" fn capture_object(ctx: *mut libc::c_void, obj: *mut LLVMMemoryBufferRef) -> LLVMErrorRef {
    unsafe {
        let shared = &*(ctx as *const Shared);
        let buf = *obj;
        let start = llvm_sys::core::LLVMGetBufferStart(buf) as *const u8;
        let size = llvm_sys::core::LLVMGetBufferSize(buf);
        let bytes = std::slice::from_raw_parts(start, size).to_vec();
        shared.pending_objects.lock().push_back(bytes);
    }
    std::ptr::null_mut()
}

extern "C" fn optimize_module_op(ctx: *mut libc::c_void, m: LLVMModuleRef) -> LLVMErrorRef {
    let t = unsafe { &*(ctx as *const TransformCtx) };
    // SAFETY: borrow the module for the duration of the call without taking ownership.
    let module = std::mem::ManuallyDrop::new(unsafe { Module::new(m) });
    match crate::optimize::optimize(&module, &t.options, &[]) {
        Ok(()) => std::ptr::null_mut(),
        Err(e) => {
            let msg = ffi::cstr(&e.to_string());
            unsafe { llvm_sys::error::LLVMCreateStringError(msg.as_ptr()) }
        }
    }
}

extern "C" fn ir_transform(
    ctx: *mut libc::c_void,
    module: *mut LLVMOrcThreadSafeModuleRef,
    _mr: LLVMOrcMaterializationResponsibilityRef,
) -> LLVMErrorRef {
    unsafe { LLVMOrcThreadSafeModuleWithModuleDo(*module, optimize_module_op, ctx) }
}

impl Engine {
    /// Port of `Engine::Engine(Options*)`.
    pub fn new(options: &Options, config: EngineConfig) -> Result<Engine> {
        crate::target::initialize_targets();
        unsafe { llvm_sys::execution_engine::LLVMLinkInMCJIT() };
        let debug = Arc::new(DebugListener::new());
        let shared = Arc::new(Shared::new(config.gc_roots, debug.clone()));
        let transform = Box::new(TransformCtx { options: options.clone() });
        unsafe {
            // EngineBuilder().setMArch/MCPU/MAttrs().selectTarget()
            let tm = crate::target::host_target_machine(options, false)?;
            let raw_tm = tm.as_mut_ptr();
            std::mem::forget(tm); // ownership moves into the JTMB below
            let jtmb = LLVMOrcJITTargetMachineBuilderCreateFromTargetMachine(raw_tm);
            let builder = LLVMOrcCreateLLJITBuilder();
            LLVMOrcLLJITBuilderSetJITTargetMachineBuilder(builder, jtmb);
            LLVMOrcLLJITBuilderSetObjectLinkingLayerCreator(
                builder,
                create_object_layer,
                Arc::as_ptr(&shared) as *mut libc::c_void,
            );
            let mut jit: LLVMOrcLLJITRef = std::ptr::null_mut();
            check(LLVMOrcCreateLLJIT(&mut jit, builder))?;

            let mut generator: LLVMOrcDefinitionGeneratorRef = std::ptr::null_mut();
            check(LLVMOrcCreateDynamicLibrarySearchGeneratorForProcess(
                &mut generator,
                LLVMOrcLLJITGetGlobalPrefix(jit),
                None,
                std::ptr::null_mut(),
            ))?;
            LLVMOrcJITDylibAddGenerator(LLVMOrcLLJITGetMainJITDylib(jit), generator);

            LLVMOrcObjectTransformLayerSetTransform(
                LLVMOrcLLJITGetObjTransformLayer(jit),
                capture_object,
                Arc::as_ptr(&shared) as *mut libc::c_void,
            );
            if config.optimize_on_add {
                LLVMOrcIRTransformLayerSetTransform(
                    LLVMOrcLLJITGetIRTransformLayer(jit),
                    ir_transform,
                    &*transform as *const TransformCtx as *mut libc::c_void,
                );
            }
            Ok(Engine { jit, shared, debug, transform, options: options.clone(), _not_send: Default::default() })
        }
    }

    pub fn options(&self) -> &Options {
        &self.options
    }

    /// `getDataLayout()` as a string.
    pub fn data_layout(&self) -> String {
        unsafe { CStr::from_ptr(LLVMOrcLLJITGetDataLayoutStr(self.jit)).to_string_lossy().into_owned() }
    }

    pub fn triple(&self) -> String {
        unsafe { CStr::from_ptr(LLVMOrcLLJITGetTripleString(self.jit)).to_string_lossy().into_owned() }
    }

    /// `getDebugListener()`.
    pub fn debug_listener(&self) -> &DebugListener {
        &self.debug
    }

    /// Copy `module` into a fresh thread-safe context (via bitcode).
    ///
    /// PORT-NOTE: C++ hands over a `ThreadSafeModule` that owns its context. An
    /// inkwell `Module` borrows its `Context`, so the module is round-tripped
    /// through bitcode into a new ORC-owned context; the caller keeps its module.
    fn thread_safe_module(&self, module: &Module) -> Result<LLVMOrcThreadSafeModuleRef> {
        unsafe {
            let buf = llvm_sys::bit_writer::LLVMWriteBitcodeToMemoryBuffer(module.as_mut_ptr());
            let tsc = LLVMOrcCreateNewThreadSafeContext();
            let ctx = LLVMOrcThreadSafeContextGetContext(tsc);
            let mut m: LLVMModuleRef = std::ptr::null_mut();
            let failed = llvm_sys::bit_reader::LLVMParseBitcodeInContext2(ctx, buf, &mut m);
            llvm_sys::core::LLVMDisposeMemoryBuffer(buf);
            if failed != 0 || m.is_null() {
                LLVMOrcDisposeThreadSafeContext(tsc);
                return Err(CompilerError::Llvm("could not transfer module to JIT context".into()));
            }
            let dl = ffi::cstr(&self.data_layout());
            llvm_sys::core::LLVMSetDataLayout(m, dl.as_ptr());
            let tsm = LLVMOrcCreateNewThreadSafeModule(m, tsc);
            LLVMOrcDisposeThreadSafeContext(tsc); // TSM keeps its own reference
            Ok(tsm)
        }
    }

    /// Port of `Engine::addModule(module)` into the default resource tracker.
    pub fn add_module(&self, module: &Module) -> Result<()> {
        let tsm = self.thread_safe_module(module)?;
        unsafe {
            let jd = LLVMOrcLLJITGetMainJITDylib(self.jit);
            let e = LLVMOrcLLJITAddLLVMIRModule(self.jit, jd, tsm);
            if !e.is_null() {
                LLVMOrcDisposeThreadSafeModule(tsm);
            }
            check(e)
        }
    }

    /// Port of `Engine::addModule(module, rt)` with a fresh resource tracker.
    pub fn add_module_tracked(&self, module: &Module) -> Result<ResourceTracker> {
        let tsm = self.thread_safe_module(module)?;
        unsafe {
            let jd = LLVMOrcLLJITGetMainJITDylib(self.jit);
            let rt = LLVMOrcJITDylibCreateResourceTracker(jd);
            let e = LLVMOrcLLJITAddLLVMIRModuleWithRT(self.jit, rt, tsm);
            if !e.is_null() {
                LLVMOrcDisposeThreadSafeModule(tsm);
                LLVMOrcReleaseResourceTracker(rt);
                return Err(CompilerError::Llvm(ffi::take_error(e).unwrap_or_default()));
            }
            Ok(ResourceTracker { rt })
        }
    }

    /// Port of `Engine::lookup(name)`: address of a JIT symbol.
    pub fn lookup(&self, name: &str) -> Result<u64> {
        let c = ffi::cstr(name);
        let mut addr: LLVMOrcExecutorAddress = 0;
        check(unsafe { LLVMOrcLLJITLookup(self.jit, &mut addr, c.as_ptr()) })?;
        Ok(addr)
    }

    /// Define absolute symbols in the main JITDylib (used e.g. for runtime
    /// hooks and C++ `defineImageBase` on Windows).
    pub fn define_absolute(&self, symbols: &[(&str, u64)]) -> Result<()> {
        unsafe {
            let mut pairs: Vec<LLVMOrcCSymbolMapPair> = symbols
                .iter()
                .map(|&(n, a)| {
                    let c = ffi::cstr(n);
                    LLVMOrcCSymbolMapPair {
                        Name: LLVMOrcLLJITMangleAndIntern(self.jit, c.as_ptr()),
                        Sym: LLVMJITEvaluatedSymbol {
                            Address: a,
                            Flags: LLVMJITSymbolFlags {
                                GenericFlags: LLVMJITSymbolGenericFlags::LLVMJITSymbolGenericFlagsExported as u8
                                    | LLVMJITSymbolGenericFlags::LLVMJITSymbolGenericFlagsCallable as u8,
                                TargetFlags: 0,
                            },
                        },
                    }
                })
                .collect();
            let mu = LLVMOrcAbsoluteSymbols(pairs.as_mut_ptr(), pairs.len());
            let jd = LLVMOrcLLJITGetMainJITDylib(self.jit);
            let e = LLVMOrcJITDylibDefine(jd, mu);
            if !e.is_null() {
                LLVMOrcDisposeMaterializationUnit(mu);
            }
            check(e)
        }
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        unsafe {
            if let Some(m) = ffi::take_error(LLVMOrcDisposeLLJIT(self.jit)) {
                eprintln!("error disposing JIT: {m}");
            }
        }
        // `shared` / `transform` outlive the JIT (fields drop after this body).
        let _ = (&self.shared, &self.transform);
    }
}
