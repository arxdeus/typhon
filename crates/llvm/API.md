# typhon-llvm public API (backend infrastructure)

Optimization, GPU and native-target support. The IR->LLVM visitor
(`src/llvisitor/`) is a separate lane and builds on the items below.

All fallible functions return `typhon_common::Result<T>`; LLVM failures map to
`CompilerError::Llvm(String)`.

## Targets (`typhon_llvm::target`, re-exported at crate root)

```rust
/// Initialize all LLVM targets compiled in (native, AArch64, X86, ARM, NVPTX). Idempotent.
pub fn initialize_targets();

/// C++ getFeaturesStr: SubtargetFeatures from `-mattr` list ("avx2" -> "+avx2", joined by ',').
pub fn features_str(mattrs: &[String]) -> String;

/// C++ getTargetMachine(triple, cpu, features, targetOptions, march, pic).
/// `march` (if non-empty) selects the target by name. Opt level Aggressive,
/// reloc PIC if `pic` else Default, code model Default. None if target lookup fails.
pub fn target_machine_for(triple: &str, cpu: &str, features: &str, march: &str, pic: bool)
    -> Result<Option<inkwell::targets::TargetMachine>>;

/// C++ getTargetMachine(module, options, setFunctionAttributes, pic): uses the module
/// triple, `options.mcpu`, `options.mattrs`, `options.march`. Returns Ok(None) when the
/// module has no triple. With `set_function_attributes` adds "target-cpu"/"target-features"
/// to every function (llvm::codegen::setFunctionAttributes).
pub fn target_machine_for_module(module: &Module, options: &Options,
    set_function_attributes: bool, pic: bool) -> Result<Option<TargetMachine>>;

/// Host triple / CPU / features helpers.
pub fn host_triple() -> String;
pub fn host_cpu_name() -> String;
pub fn host_cpu_features() -> String;
pub fn host_target_machine(options: &Options, pic: bool) -> Result<TargetMachine>;
pub fn host_target_machine_with_model(options: &Options, pic: bool, cm: CodeModel) -> Result<TargetMachine>;
pub fn target_machine_with_model(triple, cpu, features, march, pic, cm) -> Result<Option<TargetMachine>>;
pub fn data_layout_string(tm: &TargetMachine) -> String;
pub fn set_function_attributes_for(module: &Module, cpu: &str, features: &str);
```

Typical llvisitor use: `module.set_triple(&TargetTriple::create(&host_triple()))`,
then `let tm = host_target_machine(opts, pic)?; module.set_data_layout(&tm.get_target_data().get_data_layout())`.

## Optimization (`typhon_llvm::optimize`)

```rust
/// Stage at which plugin passes are requested (mirrors C++ PassBuilder EP callbacks).
pub enum PassStage { PipelineStart, PipelineEarlySimplification, LateLoopOptimizations,
                     LoopOptimizerEnd, ScalarOptimizerLate, Peephole, VectorizerStart,
                     OptimizerEarly, OptimizerLast }

/// Replaces `DSL::addLLVMPasses(PassBuilder*, bool debug)`.
pub trait LlvmPassHooks {
    /// Run custom transformations over `module` at `stage`. Default: nothing.
    fn add_llvm_passes(&self, stage: PassStage, module: &Module, debug: bool) -> Result<()> { Ok(()) }
    /// Optional extra textual new-PM pipeline appended at `stage` (run via LLVMRunPasses).
    fn pipeline_text(&self, stage: PassStage, debug: bool) -> Option<String> { None }
}

/// C++ ir::optimize(module, options, plugins): verify, GPU split, opt1 (+opt2 if !debug),
/// GPU pipeline + PTX embedding, verify. Panics-free: invalid IR -> Err (and `_dump.ll`).
pub fn optimize(module: &Module, options: &Options, hooks: &[&dyn LlvmPassHooks]) -> Result<()>;

/// C++ verify(): Err with message + dump to `_dump.ll` when broken.
pub fn verify(module: &Module) -> Result<()>;

/// Run a textual new-PM pipeline (LLVMRunPasses).
pub fn run_passes(module: &Module, passes: &str, tm: Option<&TargetMachine>) -> Result<()>;

/// Global ctor name used by the visitor (LLVMVisitor::getGlobalCtorName()).
pub const GLOBAL_CTOR_NAME: &str = ".main.ctor";
```

Individual passes are public in `typhon_llvm::optimize::passes`
(`allocation_remover`, `allocation_hoister`, `allocation_auto_free`,
`coro_branch_simplifier`, `coro_unroll_control`, `coro_unroll_cleanup`,
`coro_resume_inliner`, `openmp_thread_id_optimizer`, `apply_debug_transformations`,
`apply_fast_math_transformations`). Each takes `&Module` (or `FunctionValue`) and
returns `bool` (changed). Function passes take a raw `LLVMValueRef`
(`FunctionValue::as_value_ref()`), module passes a raw `LLVMModuleRef` (`Module::as_mut_ptr()`).

NOTE for JIT users: the engine compiles with the **Large code model** (RTDyld on
MachO/AArch64 has no far-branch stubs). The visitor need not care: the engine
re-targets modules on `add_module`.

## GPU (`typhon_llvm::gpu`)

```rust
pub fn prepare_gpu_module<'ctx>(module: &Module<'ctx>, options: &Options) -> Result<Option<Module<'ctx>>>;
pub fn apply_gpu_transformations(module: &Module, clone: Module, options: &Options,
                                 ptx_filename: &str) -> Result<()>;
```

## Native (`typhon_llvm::native`)

```rust
pub fn native_cpu_and_features(triple: &str) -> Option<(String, String)>;
pub fn apply_native_attributes(module: &Module, cpu: &str, features: &str); // ArchNativePass
```

## JIT engine (`typhon_llvm::engine`)

```rust
pub struct GcRootCallbacks { pub add_roots: unsafe extern "C" fn(*mut c_void, *mut c_void),
                             pub remove_roots: unsafe extern "C" fn(*mut c_void, *mut c_void) }

pub struct EngineConfig { pub gc_roots: Option<GcRootCallbacks>, pub optimize_on_add: bool }

pub struct Engine { .. }   // owns LLVMOrcLLJITRef; Drop disposes it. !Send.
// Engine::options(&self) -> &Options
impl Engine {
    pub fn new(options: &Options, config: EngineConfig) -> Result<Engine>;
    pub fn data_layout(&self) -> String;
    pub fn triple(&self) -> String;
    /// Add a module (copied into an ORC ThreadSafeContext via bitcode; `module` is not consumed).
    /// If `optimize_on_add`, the IR transform layer runs `optimize(module, options, &[])`.
    pub fn add_module(&self, module: &Module) -> Result<()>;
    /// Same, but into a fresh ResourceTracker which can later be removed.
    pub fn add_module_tracked(&self, module: &Module) -> Result<ResourceTracker>;
    /// Address of a symbol (unmangled name; global prefix handled by LLJIT).
    pub fn lookup(&self, name: &str) -> Result<u64>;
    /// Define absolute symbols (e.g. runtime functions) in the main JITDylib.
    pub fn define_absolute(&self, symbols: &[(&str, u64)]) -> Result<()>;
    /// Debug info listener (symbolization for runtime backtraces).
    pub fn debug_listener(&self) -> &DebugListener;
}

impl DebugListener {
    pub fn symbolize(&self, pc: usize) -> Option<LineInfo>;       // function, file, line, col
    pub fn pretty_backtrace_frame(&self, pc: usize) -> String;      // "" if unknown
    pub fn pretty_backtrace(&self, pcs: &[usize]) -> String;
}
```
