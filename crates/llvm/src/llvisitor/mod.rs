//! CIR -> LLVM IR code generator.
//!
//! Layout of the port:
//! - `mod.rs`    : visitor state, naming, globals, module/function visits, runtime helpers
//! - `types.rs`  : `getLLVMType`, `getLLVMFuncType`, `getDIType`
//! - `flows.rs`  : series/if/while/for/imperative-for/try-catch/pipeline flows
//! - `instrs.rs` : constants and instructions (calls, coroutines, throw, ...)
//! - `rtti.rs`   : runtime type info, object layouts, field addresses, intrinsics
//! - `win.rs`    : Win64 ABI wrappers (only active for `*-windows-msvc` triples)
//! - `output.rs` : object/bitcode/ll/executable/library writers, runtime link args
//! - `ll.rs`     : thin `llvm-sys` helpers (IRBuilder / DIBuilder subset)
//!
//! PORT-NOTE: the C++ visitor builds IR with `llvm::IRBuilder<>` on raw `llvm::Value*`.
//! The Rust port keeps raw `LLVMValueRef` handles (the visitor's maps store pointers
//! across modules exactly like C++), building through the LLVM C API; the module
//! itself is owned as an `inkwell::module::Module` so callers get a safe handle.

pub mod cross;
mod flows;
mod instrs;
pub mod ll;
mod output;
pub use output::{runtime_link_args, runtime_lib_dirs, static_runtime_archive};
mod rtti;
mod types;
mod win;

use std::collections::BTreeMap;
use std::ptr::null_mut;

use inkwell::context::Context;
use inkwell::module::Module as LlModule;
use llvm_sys::LLVMLinkage::*;
use llvm_sys::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};
use typhon_common::{CompilerError, Options, Result, SrcInfo};
use typhon_ir::ids::{FuncId, Id, NodeRef, TypeId, ValueId, VarId};
use typhon_ir::module::Module;
use typhon_ir::types::TypeKind;
use typhon_ir::util::irtools;
use typhon_ir::var::{FuncKind, Intrinsic};

use self::ll::{BB, Builder, DiBuilder, Md, Ty, V};


const MAIN_UNCLASH: &str = ".main.unclash";
const MAIN_CTOR: &str = ".main.ctor";
/// Marks `@export` function definitions (consumed by library linking).
pub(crate) const EXPORT_ATTR: &str = "typhon-export";

/// `SEQ_FLAG_*` (runtime flags).
const SEQ_FLAG_DEBUG: i64 = 1 << 0;
const SEQ_FLAG_STANDALONE: i64 = 1 << 2;
/// `SEQ_EXCEPTION_CLASS`.
const SEQ_EXCEPTION_CLASS: u64 = 0x6f62_6a00_7365_7100;

fn llvm_memory_none_attr() -> String {
    typhon_common::mangle::func("std.internal.attributes", "llvm_memory_none")
}
fn llvm_no_capture_attr() -> String {
    typhon_common::mangle::func("std.internal.attributes", "llvm_nocapture")
}
fn llvm_no_free_attr() -> String {
    typhon_common::mangle::func("std.internal.attributes", "llvm_nofree")
}
fn llvm_c_call_attr() -> String {
    typhon_common::mangle::func("std.internal.attributes", "llvm_c_call")
}

/// `seq_exc_offset()`: offset from the `_Unwind_Exception` back to the start of the
/// Typhon exception object `{ void *obj; Backtrace bt; _Unwind_Exception ue; }`.
///
/// PORT-NOTE: C++ calls the runtime function at compile time; the compiler does not
/// link the runtime here, so the layout (16-byte aligned `_Unwind_Exception` after
/// three pointer-sized words) is computed for the module's pointer size.
fn seq_exc_offset(ptr_size: u64) -> i64 {
    let header = 3 * ptr_size; // obj + Backtrace{frames, count}
    let aligned = header.div_ceil(16) * 16;
    -(aligned as i64)
}

// ------------------------------------------------------------------ state

#[derive(Clone, Copy)]
struct CoroData {
    promise: V,
    handle: V,
    cleanup: BB,
    suspend: BB,
    exit: BB,
    is_async: bool,
}

impl Default for CoroData {
    fn default() -> Self {
        CoroData { promise: null_mut(), handle: null_mut(), cleanup: null_mut(), suspend: null_mut(), exit: null_mut(), is_async: false }
    }
}

#[derive(Clone)]
struct LoopData {
    sequence_number: i64,
    break_block: BB,
    continue_block: BB,
    loop_id: Id,
}

/// `TryCatchData::State`.
#[derive(Clone, Copy)]
#[repr(i64)]
enum TcState {
    NotThrown = 0,
    Thrown,
    Caught,
    Return,
    Break,
    Continue,
    Rethrow,
}

#[derive(Clone)]
struct TryCatchData {
    sequence_number: i64,
    exception_block: BB,
    exception_route_block: BB,
    finally_block: BB,
    finally_exception_block: BB,
    catch_types: Vec<Option<TypeId>>,
    handlers: Vec<BB>,
    exc_flag: V,
    catch_store: V,
    delegate_depth: V,
    ret_store: V,
    loop_sequence: V,
}

impl Default for TryCatchData {
    fn default() -> Self {
        TryCatchData {
            sequence_number: -1,
            exception_block: null_mut(),
            exception_route_block: null_mut(),
            finally_block: null_mut(),
            finally_exception_block: null_mut(),
            catch_types: Vec::new(),
            handlers: Vec::new(),
            exc_flag: null_mut(),
            catch_store: null_mut(),
            delegate_depth: null_mut(),
            ret_store: null_mut(),
            loop_sequence: null_mut(),
        }
    }
}

#[derive(Clone, Copy)]
#[allow(dead_code)]
struct CatchData {
    sequence_number: i64,
    exception: V,
    type_id: V,
}

#[allow(dead_code)]
struct DebugInfo {
    builder: Option<DiBuilder>,
    unit: Md,
    flags: String,
}

impl DebugInfo {
    fn file(&self, path: &str) -> Md {
        self.b().file(path)
    }
    fn b(&self) -> &DiBuilder {
        self.builder.as_ref().expect("no debug info builder")
    }
}

#[derive(Clone, Copy)]
struct FieldOffsetResolver {
    constant: bool,
    offset: u64,
    function: V,
}

/// IR to LLVM visitor.
pub struct LlvmVisitor<'ctx> {
    // NOTE: `db` must drop before `module` (the DIBuilder references the module).
    db: DebugInfo,
    module: Option<LlModule<'ctx>>,
    ctx: &'ctx Context,
    c: LLVMContextRef,
    options: Options,
    b: Builder,
    func: V,
    block: BB,
    value: V,
    vars: FxHashMap<Id, V>,
    funcs: FxHashMap<Id, V>,
    win_callbacks: FxHashMap<usize, V>,
    function_names: FxHashMap<Id, String>,
    function_name_counts: FxHashMap<String, usize>,
    field_offset_resolvers: BTreeMap<(i64, i64), FieldOffsetResolver>,
    virtual_lookup_types: FxHashMap<usize, TypeId>,
    virtual_targets: BTreeMap<(i64, i64), Vec<(i64, V)>>,
    coro: CoroData,
    loops: Vec<LoopData>,
    trycatch: Vec<TryCatchData>,
    finally: Vec<TryCatchData>,
    catches: Vec<CatchData>,
    next_sequence_number: i64,
    error: Option<CompilerError>,
}

impl<'ctx> LlvmVisitor<'ctx> {
    /// Constructs an LLVM visitor (`LLVMVisitor(Options*)`).
    pub fn new(ctx: &'ctx Context, options: Options) -> Self {
        crate::target::initialize_targets();
        let c = ctx.raw();
        LlvmVisitor {
            db: DebugInfo { builder: None, unit: null_mut(), flags: String::new() },
            module: None,
            ctx,
            c,
            options,
            b: Builder::new(c),
            func: null_mut(),
            block: null_mut(),
            value: null_mut(),
            vars: FxHashMap::default(),
            funcs: FxHashMap::default(),
            win_callbacks: FxHashMap::default(),
            function_names: FxHashMap::default(),
            function_name_counts: FxHashMap::default(),
            field_offset_resolvers: BTreeMap::new(),
            virtual_lookup_types: FxHashMap::default(),
            virtual_targets: BTreeMap::new(),
            coro: CoroData::default(),
            loops: Vec::new(),
            trycatch: Vec::new(),
            finally: Vec::new(),
            catches: Vec::new(),
            next_sequence_number: 0,
            error: None,
        }
    }

    // ---------------------------------------------------------------- accessors

    pub fn options(&self) -> &Options {
        &self.options
    }
    pub fn options_mut(&mut self) -> &mut Options {
        &mut self.options
    }
    pub fn set_debug(&mut self, v: bool) {
        self.options.debug = v;
    }
    pub fn set_standalone(&mut self, v: bool) {
        self.options.standalone = v;
    }
    pub fn context(&self) -> &'ctx Context {
        self.ctx
    }
    /// `getModule()`.
    pub fn module(&self) -> Option<&LlModule<'ctx>> {
        self.module.as_ref()
    }
    /// Raw LLVM builder handle (`getBuilder()`), for custom codegen.
    pub fn builder(&self) -> LLVMBuilderRef {
        self.b.b
    }
    /// `getFunc()`.
    pub fn func(&self) -> LLVMValueRef {
        self.func
    }
    /// `getBlock()`.
    pub fn block(&self) -> LLVMBasicBlockRef {
        self.block
    }
    /// `getValue()`.
    pub fn value(&self) -> LLVMValueRef {
        self.value
    }
    pub fn set_func(&mut self, f: LLVMValueRef) {
        self.func = f;
    }
    pub fn set_block(&mut self, b: LLVMBasicBlockRef) {
        self.block = b;
    }
    pub fn set_value(&mut self, v: LLVMValueRef) {
        self.value = v;
    }
    /// `getVars()`.
    pub fn vars_mut(&mut self) -> &mut FxHashMap<Id, LLVMValueRef> {
        &mut self.vars
    }
    /// `getFuncs()`.
    pub fn funcs_mut(&mut self) -> &mut FxHashMap<Id, LLVMValueRef> {
        &mut self.funcs
    }
    /// Debug-info builder and compile unit (`getDebugInfo()`).
    pub fn debug_info(&self) -> (LLVMDIBuilderRef, LLVMMetadataRef) {
        (self.db.builder.as_ref().map(|b| b.0).unwrap_or(null_mut()), self.db.unit)
    }
    /// `getGlobalCtorName()`.
    pub fn global_ctor_name() -> &'static str {
        MAIN_CTOR
    }

    pub(crate) fn m(&self) -> LLVMModuleRef {
        self.module.as_ref().expect("no LLVM module").as_mut_ptr()
    }
    fn is_win_msvc(&self) -> bool {
        self.module.is_some() && ll::is_win_msvc_triple(&ll::triple(self.m()))
    }
    fn fail(&mut self, msg: impl Into<String>) {
        if self.error.is_none() {
            self.error = Some(CompilerError::Llvm(msg.into()));
        }
    }

    // ---------------------------------------------------------------- naming

    /// `getNameForFunction`.
    pub fn get_name_for_function(m: &Module, f: FuncId) -> String {
        let fd = m.func(f).expect("not a function");
        if fd.is_external() || irtools::is_export(m, f) {
            fd.unmangled_name.clone()
        } else if irtools::is_gpu_kernel(m, f) {
            m.var_name(f).to_string()
        } else {
            m.reference_string(f.into())
        }
    }

    /// `getNameForVar`.
    pub fn get_name_for_var(m: &Module, v: VarId) -> String {
        if m.is_func(v) {
            return Self::get_name_for_function(m, v);
        }
        let name = m.var_name(v);
        if m.is_external(v) {
            name.to_string()
        } else if name.starts_with('L') {
            // ".Lxxx" is a linker-local name, so add an underscore if needed
            format!("._{name}")
        } else {
            format!(".{name}")
        }
    }

    /// `getDebugNameForVariable`.
    pub fn get_debug_name_for_variable(m: &Module, v: VarId) -> String {
        let name = m.var_name(v);
        match name.find('.') {
            Some(p) if p != 0 => name[..p].to_string(),
            _ => name.to_string(),
        }
    }

    /// `getSrcInfo(x)` (default `<internal>:0:0`).
    fn src_info(m: &Module, n: NodeRef) -> SrcInfo {
        m.src_info(n)
    }
    fn src_file(s: &SrcInfo) -> String {
        let f = s.file.as_str();
        if f.is_empty() { "<internal>".to_string() } else { f.to_string() }
    }

    /// `getLLVMNameForFunction`.
    fn get_llvm_name_for_function(&mut self, m: &Module, f: FuncId) -> String {
        let fd = m.func(f).expect("not a function");
        if fd.is_external() || irtools::is_export(m, f) || irtools::is_gpu_kernel(m, f) {
            return Self::get_name_for_function(m, f);
        }
        let uid = m.var_uid(f);
        if let Some(n) = self.function_names.get(&uid) {
            return n.clone();
        }
        let base = m.var_name(f).to_string();
        let cnt = self.function_name_counts.entry(base.clone()).or_insert(0);
        let name = format!("{base}.{cnt}");
        *cnt += 1;
        self.function_names.insert(uid, name.clone());
        name
    }

    // ---------------------------------------------------------------- globals

    /// `registerGlobal`.
    pub fn register_global(&mut self, m: &Module, var: VarId) {
        let var = m.actual_var(var);
        if !m.is_global(var) {
            return;
        }
        if let Some(fd) = m.func(var) {
            if let FuncKind::Internal { intrinsic, .. } = fd.kind
                && intrinsic.is_type_query()
            {
                return;
            }
            let function = self.make_llvm_function(m, var, false);
            self.funcs.insert(m.var_uid(var), function);
            if let FuncKind::Internal { intrinsic: Intrinsic::VirtualLookup, intrinsic_type } = fd.kind
                && let Some(t) = intrinsic_type
            {
                self.virtual_lookup_types.insert(function as usize, t);
            }
        } else {
            let llvm_type = self.get_llvm_type(m, m.var_type(var).expect("var without type"));
            if !self.is_storable_type(llvm_type) {
                let d = self.get_dummy_value(llvm_type);
                self.insert_var(m.var_uid(var), d);
                return;
            }
            let external = m.is_external(var);
            let tls = m.is_thread_local(var);
            let linkage = if external { LLVMExternalLinkage } else { LLVMPrivateLinkage };
            let init = (!external).then(|| ll::const_null(llvm_type));
            let storage = ll::new_global(self.m(), llvm_type, false, linkage, init, &Self::get_name_for_var(m, var));
            self.insert_var(m.var_uid(var), storage);
            let triple = ll::triple(self.m());
            if external {
                ll::set_unnamed_addr(storage, llvm_sys::LLVMUnnamedAddr::LLVMLocalUnnamedAddr);
            } else if !(tls && ll::is_windows_triple(&triple)) {
                // debug info
                let src = Self::src_info(m, var.into());
                let file = self.db.file(&Self::src_file(&src));
                let ty = self.get_di_type(m, m.var_type(var).unwrap());
                let gve = self.db.b().global_var_expr(
                    self.db.unit,
                    &Self::get_debug_name_for_variable(m, var),
                    m.var_name(var),
                    file,
                    src.line.max(0) as u32,
                    ty,
                    true,
                );
                ll::add_global_debug_info(storage, gve);
            }
            if tls {
                ll::set_thread_local(storage, true);
                // Bionic (arm64) rejects executables whose PT_TLS alignment is < 64.
                if self.options.target_info().os == typhon_common::target::TargetOs::Android {
                    unsafe {
                        let cur = llvm_sys::core::LLVMGetAlignment(storage);
                        llvm_sys::core::LLVMSetAlignment(storage, cur.max(64));
                    }
                }
            }
        }
    }

    /// `getVar`.
    fn get_var(&mut self, m: &Module, var: VarId) -> V {
        let var = m.actual_var(var);
        let uid = m.var_uid(var);
        self.vars.get(&uid).copied().unwrap_or(null_mut())
    }

    /// `getFunc`.
    fn get_func(&mut self, m: &Module, f: FuncId) -> V {
        let f = m.actual_var(f);
        let uid = m.var_uid(f);
        let existing = self.funcs.get(&uid).copied();
        let fd = m.func(f).expect("not a function");
        if let FuncKind::Internal { intrinsic, .. } = fd.kind
            && intrinsic.is_type_query()
            && existing.is_none_or(|v| v.is_null())
        {
            // Direct queries never need an LLVM function. Materialize a wrapper only
            // when its address is used, without changing the caller's insertion point.
            let wrapper = self.make_llvm_function(m, f, false);
            ll::set_linkage(wrapper, LLVMPrivateLinkage);
            ll::add_fn_attr(wrapper, "alwaysinline");
            ll::set_memory_none(wrapper);
            ll::add_fn_attr(wrapper, "nounwind");
            ll::add_fn_attr(wrapper, "willreturn");
            let tmp = Builder::new(self.c);
            tmp.at_end(ll::append_block(self.c, wrapper, "entry"));
            let q = self.codegen_type_query(m, f);
            tmp.ret(q);
            self.funcs.insert(uid, wrapper);
            return wrapper;
        }
        existing.unwrap_or(null_mut())
    }

    // ---------------------------------------------------------------- module setup

    /// `makeModule`: a new module initialized for the host (or `march`/`mcpu`) target.
    fn make_module(&mut self, src: Option<SrcInfo>) -> Result<LlModule<'ctx>> {
        let tm = crate::target::compile_target_machine(&self.options, false)?;
        let module = self.ctx.create_module("typhon");
        module.set_triple(&tm.get_triple());
        module.set_data_layout(&tm.get_target_data().get_data_layout());
        let src_file = src.as_ref().map(Self::src_file).unwrap_or_else(|| "<internal>".to_string());
        unsafe {
            llvm_sys::core::LLVMSetSourceFileName(module.as_mut_ptr(), src_file.as_ptr() as *const _, src_file.len());
        }
        // debug info setup
        let db = DiBuilder::new(module.as_mut_ptr());
        let file = db.file(&src_file);
        let producer = format!("typhon version {}", typhon_common::config::VERSION);
        self.db.unit = db.compile_unit(file, &producer, !self.options.debug);
        self.db.builder = Some(db);
        ll::add_module_flag_int(module.as_mut_ptr(), "Debug Info Version", ll::debug_metadata_version());
        let triple = ll::triple(module.as_mut_ptr());
        // darwin only supports dwarf2
        if ll::is_darwin_triple(&triple) {
            ll::add_module_flag_int(module.as_mut_ptr(), "Dwarf Version", 2);
        }
        if ll::is_windows_triple(&triple) {
            ll::add_module_flag_int(module.as_mut_ptr(), "CodeView", 1);
        }
        Ok(module)
    }

    /// `clearLLVMData`.
    fn clear_llvm_data(&mut self) {
        self.func = null_mut();
        self.block = null_mut();
        self.value = null_mut();
        self.field_offset_resolvers.clear();
        self.virtual_lookup_types.clear();
        self.virtual_targets.clear();
        self.function_names.clear();
        self.function_name_counts.clear();
        self.win_callbacks.clear();
        self.funcs.retain(|_, f| !(!f.is_null() && ll::linkage(*f) == LLVMPrivateLinkage));
        for f in self.funcs.values_mut() {
            *f = null_mut();
        }
        self.vars.retain(|_, v| v.is_null() || unsafe { !llvm_sys::core::LLVMIsAGlobalValue(*v).is_null() });
        for v in self.vars.values_mut() {
            *v = null_mut();
        }
        self.coro = CoroData::default();
        self.loops.clear();
        self.trycatch.clear();
        self.finally.clear();
        self.catches.clear();
        self.db.builder = None;
        self.db.unit = null_mut();
        self.module = None;
    }

    /// `takeModule`: returns the current module and replaces it with a fresh one.
    /// References to variables/functions of the old module become "external".
    ///
    /// PORT-NOTE: the LLVM context is borrowed by the visitor (inkwell lifetime), so
    /// the new module lives in the same context instead of a fresh one.
    pub fn take_module(&mut self, m: Option<&Module>, src: Option<SrcInfo>) -> Result<LlModule<'ctx>> {
        if let Some(m) = m {
            let mut to_process = FxHashSet::default();
            for var in m.vars() {
                let uid = m.var_uid(var);
                if m.is_func(var) {
                    if self.funcs.contains_key(&uid) {
                        continue;
                    }
                    to_process.insert(uid);
                } else if self.vars.contains_key(&uid) {
                    continue;
                }
                self.register_global(m, var);
            }
            for var in m.vars() {
                if m.is_func(var) && to_process.contains(&m.var_uid(var)) {
                    self.process_func(m, var);
                }
            }
        }
        self.prepare_win_exports();
        if let Some(db) = &self.db.builder {
            db.finalize();
        }
        self.db.builder = None;
        let current = self.module.take().expect("no module");
        self.clear_llvm_data();
        let fresh = self.make_module(src)?;
        self.module = Some(fresh);
        if let Some(e) = self.error.take() {
            return Err(e);
        }
        Ok(current)
    }

    /// `setDebugInfoForNode`.
    pub fn set_debug_info_for_node(&mut self, m: &Module, n: Option<NodeRef>) {
        if let Some(n) = n
            && !self.func.is_null()
        {
            let sp = ll::subprogram(self.func);
            if !sp.is_null() {
                let s = Self::src_info(m, n);
                self.b.set_debug_loc(ll::di_location(self.c, s.line, s.col, sp));
                return;
            }
        }
        self.b.set_debug_loc(null_mut());
    }

    /// `process(Value)`.
    pub fn process_value(&mut self, m: &Module, v: ValueId) {
        let v = m.actual_value(v);
        self.set_debug_info_for_node(m, Some(v.into()));
        self.visit_value(m, v);
    }

    /// `process(Func)`.
    pub fn process_func(&mut self, m: &Module, f: FuncId) {
        let f = m.actual_var(f);
        self.set_debug_info_for_node(m, Some(f.into()));
        match &m.func(f).expect("cannot visit var").kind {
            FuncKind::Bodied { .. } => self.visit_bodied_func(m, f),
            FuncKind::External => self.visit_external_func(m, f),
            FuncKind::Internal { .. } => self.visit_internal_func(m, f),
            FuncKind::Llvm { .. } => self.visit_llvm_func(m, f),
        }
    }

    // ---------------------------------------------------------------- runtime helpers

    fn decl(&self, name: &str, ret: Ty, args: &[Ty]) -> (Ty, V) {
        let ty = ll::fn_ty(ret, args, false);
        let f = ll::get_or_insert_function(self.m(), name, ty);
        (ll::function_type(f), f)
    }

    /// `makeAllocFunc`.
    fn make_alloc_func(&self, atomic: bool, uncollectable: bool) -> (Ty, V) {
        let name = match (atomic, uncollectable) {
            (true, true) => "seq_alloc_atomic_uncollectable",
            (true, false) => "seq_alloc_atomic",
            (false, true) => "seq_alloc_uncollectable",
            (false, false) => "seq_alloc",
        };
        let (t, g) = self.decl(name, self.b.ptr(), &[self.b.i64()]);
        ll::add_fn_attr(g, "nounwind");
        ll::add_enum_attr(g, ll::RET_INDEX, "noalias", 0);
        ll::set_only_inaccessible_mem(g);
        ll::add_enum_attr(g, ll::RET_INDEX, "noundef", 0);
        ll::add_enum_attr(g, ll::RET_INDEX, "nonnull", 0);
        // AllocFnKind::Alloc | Uninitialized
        ll::add_enum_attr(g, ll::FN_INDEX, "allockind", 1 | 8);
        // allocsize(0): ElemSizeArg=0, NumElemsArg = none (0xFFFFFFFF)
        ll::add_enum_attr(g, ll::FN_INDEX, "allocsize", 0xFFFF_FFFF);
        ll::add_string_attr(g, ll::FN_INDEX, "alloc-family", "seq_alloc");
        (t, g)
    }

    /// `makeReallocFunc` (`seq_realloc(ptr, new_size, old_size)`).
    fn make_realloc_func(&self) -> (Ty, V) {
        let (t, g) = self.decl("seq_realloc", self.b.ptr(), &[self.b.ptr(), self.b.i64(), self.b.i64()]);
        ll::add_fn_attr(g, "nounwind");
        ll::add_enum_attr(g, ll::RET_INDEX, "noundef", 0);
        ll::add_enum_attr(g, ll::RET_INDEX, "nonnull", 0);
        ll::add_enum_attr(g, 1, "allocptr", 0);
        // AllocFnKind::Realloc | Uninitialized
        ll::add_enum_attr(g, ll::FN_INDEX, "allockind", 2 | 8);
        ll::add_enum_attr(g, ll::FN_INDEX, "allocsize", (1u64 << 32) | 0xFFFF_FFFF);
        ll::add_string_attr(g, ll::FN_INDEX, "alloc-family", "seq_alloc");
        (t, g)
    }

    /// `makeFreeFunc`.
    fn make_free_func(&self) -> (Ty, V) {
        let (t, g) = self.decl("seq_free", self.b.void(), &[self.b.ptr()]);
        ll::add_fn_attr(g, "nounwind");
        ll::add_enum_attr(g, 1, "allocptr", 0);
        ll::add_enum_attr(g, ll::FN_INDEX, "allockind", 4);
        ll::add_string_attr(g, ll::FN_INDEX, "alloc-family", "seq_alloc");
        (t, g)
    }

    /// `makePersonalityFunc`.
    fn make_personality_func(&self) -> (Ty, V) {
        if self.is_win_msvc() {
            let ty = ll::fn_ty(self.b.i32(), &[], true);
            let f = ll::get_or_insert_function(self.m(), "__C_specific_handler", ty);
            return (ty, f);
        }
        let i32t = self.b.i32();
        self.decl("seq_personality", i32t, &[i32t, i32t, self.b.i64(), self.b.ptr(), self.b.ptr()])
    }

    /// `makeWinEHFilter`.
    fn make_win_eh_filter(&self) -> (Ty, V) {
        let filter_ty = ll::fn_ty(self.b.i32(), &[self.b.ptr(), self.b.ptr()], false);
        let real = ll::get_or_insert_function(self.m(), "seq_exc_filter", filter_ty);
        if !self.is_win_msvc() {
            return (filter_ty, real);
        }
        let thunk_name = "seq_exc_filter.thunk";
        if let Some(t) = ll::get_function(self.m(), thunk_name) {
            return (filter_ty, t);
        }
        // Reference an in-module thunk and reach the real filter through a
        // 64-bit pointer (keeps the thunk within short-branch range).
        let ptr_global = ll::new_global(self.m(), self.b.ptr(), true, LLVMPrivateLinkage, Some(real), "seq_exc_filter.addr");
        let thunk = ll::get_or_insert_function(self.m(), thunk_name, filter_ty);
        ll::set_linkage(thunk, LLVMPrivateLinkage);
        let tb = Builder::new(self.c);
        tb.at_end(ll::append_block(self.c, thunk, "entry"));
        let fp = tb.load(self.b.ptr(), ptr_global);
        let call = tb.call(filter_ty, fp, &[ll::param(thunk, 0), ll::param(thunk, 1)]);
        tb.ret(call);
        (filter_ty, thunk)
    }

    /// `makeExcAllocFunc`.
    fn make_exc_alloc_func(&self) -> (Ty, V) {
        let (t, g) = self.decl("seq_alloc_exc", self.b.ptr(), &[self.b.ptr()]);
        ll::add_fn_attr(g, "nounwind");
        (t, g)
    }

    /// `makeThrowFunc`.
    fn make_throw_func(&self) -> (Ty, V) {
        let (t, g) = self.decl("seq_throw", self.b.void(), &[self.b.ptr()]);
        ll::add_fn_attr(g, "noreturn");
        (t, g)
    }

    /// `makeTerminateFunc`.
    fn make_terminate_func(&self) -> (Ty, V) {
        let (t, g) = self.decl("seq_terminate", self.b.void(), &[self.b.ptr()]);
        ll::add_fn_attr(g, "noreturn");
        (t, g)
    }

    fn exc_current_func(&self) -> (Ty, V) {
        self.decl("seq_exc_current", self.b.ptr(), &[])
    }

    /// `getTypeInfoType`.
    fn get_type_info_type(&self) -> Ty {
        ll::struct_ty(self.c, &[self.b.i32()])
    }

    /// `getPadType`.
    fn get_pad_type(&self) -> Ty {
        ll::struct_ty(self.c, &[self.b.ptr(), self.b.i32()])
    }

    /// `typeIdxLookup`.
    ///
    /// PORT-NOTE: C++ asks the parser cache for the realization id of the type's AST
    /// class. The realization id is exactly `RuntimeInfo::id` (set by the type checker),
    /// so the Rust port reads it from the IR type; types without runtime info fall back
    /// to a stable hash of the type name.
    fn type_idx_lookup(m: &Module, t: Option<TypeId>) -> i32 {
        let Some(t) = t else { return 0 }; // CATCH_ALL_SELECTOR
        let t = m.actual_type(t);
        if let Some(info) = m.runtime_info(t) {
            if info.exception_base.map(|b| m.actual_type(b)) == Some(t) {
                return -1; // BASE_EXCEPTION_SELECTOR
            }
            return info.id as i32;
        }
        let mut h: u32 = 2166136261;
        for b in m.type_name(t).bytes() {
            h = (h ^ b as u32).wrapping_mul(16777619);
        }
        (h & 0x3fff_ffff) as i32 + 1
    }

    /// `getTypeIdxVar`.
    fn get_type_idx_var(&self, m: &Module, t: Option<TypeId>) -> V {
        let tit = self.get_type_info_type();
        let name = match t {
            Some(t) => format!("typhon.typeidx.{}", m.type_name(t)),
            None => "typhon.typeidx.<all>".to_string(),
        };
        if let Some(g) = ll::get_global(self.m(), &name) {
            return g;
        }
        let idx = Self::type_idx_lookup(m, t);
        let init = ll::const_struct(self.c, &[self.b.c_i32(idx as i64)]);
        let g = ll::new_global(self.m(), tit, true, LLVMPrivateLinkage, Some(init), &name);
        ll::set_unnamed_addr(g, llvm_sys::LLVMUnnamedAddr::LLVMGlobalUnnamedAddr);
        g
    }

    /// `getTypeIdx`.
    fn get_type_idx(m: &Module, t: Option<TypeId>) -> i32 {
        Self::type_idx_lookup(m, t)
    }

    /// `call(callee, args)`: emits a call, or an invoke when inside a try/finally.
    fn call(&mut self, fnty: Ty, callee: V, args: &[V]) -> V {
        self.b.at_end(self.block);
        if (self.trycatch.is_empty() && self.finally.is_empty()) || self.options.noexc {
            let is_throw = unsafe { !llvm_sys::core::LLVMIsAFunction(callee).is_null() } && ll::name(callee) == "seq_throw";
            if !self.options.noexc && self.is_win_msvc() && ll::has_string_attr(self.func, "typhon.seh.scopes") && is_throw {
                // Give a rethrow outside all local handlers an explicit unwind-to-caller
                // edge, so SEH scope ranges cannot absorb it into an earlier try region.
                let normal = ll::append_block(self.c, self.func, "throw.unreachable");
                let unwind = ll::append_block(self.c, self.func, "throw.unwind");
                let result = self.b.invoke(fnty, callee, args, normal, unwind);
                self.b.at_end(unwind);
                let cleanup = self.b.cleanup_pad(ll::token_none(self.c));
                let (st, sf) = ll::intrinsic(self.m(), "llvm.sideeffect", &[]);
                self.b.call_funclet(st, sf, &[], cleanup);
                self.b.cleanup_ret(cleanup);
                self.block = normal;
                self.b.at_end(self.block);
                return result;
            }
            return self.b.call(fnty, callee, args);
        }
        let normal = ll::append_block(self.c, self.func, "invoke.normal");
        // use non-empty of finally-stack and try-stack, or whichever is most recent
        let unwind = if self.finally.is_empty() {
            self.trycatch.last().unwrap().exception_block
        } else if self.trycatch.is_empty() {
            self.finally.last().unwrap().finally_exception_block
        } else if self.trycatch.last().unwrap().sequence_number > self.finally.last().unwrap().sequence_number {
            self.trycatch.last().unwrap().exception_block
        } else {
            self.finally.last().unwrap().finally_exception_block
        };
        let result = self.b.invoke(fnty, callee, args, normal, unwind);
        self.block = normal;
        self.b.at_end(self.block);
        result
    }

    fn next_seq(&mut self) -> i64 {
        let n = self.next_sequence_number;
        self.next_sequence_number += 1;
        n
    }
    fn enter_loop(&mut self, mut d: LoopData) {
        d.sequence_number = self.next_seq();
        self.loops.push(d);
    }
    fn exit_loop(&mut self) {
        assert!(self.loops.pop().is_some(), "no loops present");
    }
    fn enter_try(&mut self, mut d: TryCatchData) {
        // Only functions with local SEH scopes need an explicit rethrow boundary.
        if self.is_win_msvc() {
            ll::add_string_attr(self.func, ll::FN_INDEX, "typhon.seh.scopes", "");
        }
        d.sequence_number = self.next_seq();
        self.trycatch.push(d);
    }
    fn exit_try(&mut self) {
        assert!(self.trycatch.pop().is_some(), "no try catches present");
    }
    fn enter_finally(&mut self, mut d: TryCatchData) {
        d.sequence_number = self.next_seq();
        self.finally.push(d);
    }
    fn exit_finally(&mut self) {
        assert!(self.finally.pop().is_some(), "no finally present");
    }
    fn enter_catch(&mut self, mut d: CatchData) {
        d.sequence_number = self.next_seq();
        self.catches.push(d);
    }
    fn exit_catch(&mut self) {
        assert!(self.catches.pop().is_some(), "no catches present");
    }
    /// `getInnermostTryCatch`.
    fn innermost_try_catch(&self) -> Option<TryCatchData> {
        match (self.trycatch.last(), self.finally.last()) {
            (None, f) => f.cloned(),
            (Some(t), None) => Some(t.clone()),
            (Some(t), Some(f)) => Some(if t.sequence_number > f.sequence_number { t.clone() } else { f.clone() }),
        }
    }
    /// `getInnermostTryCatchBeforeLoop`.
    #[allow(dead_code)]
    fn innermost_try_catch_before_loop(&self) -> Option<TryCatchData> {
        let t = self.trycatch.last()?;
        if self.loops.is_empty() || t.sequence_number > self.loops.last().unwrap().sequence_number {
            Some(t.clone())
        } else {
            None
        }
    }
    /// `getLoopData`.
    fn get_loop_data(&self, loop_id: Id) -> Option<LoopData> {
        self.loops.iter().find(|d| d.loop_id == loop_id).cloned()
    }

    fn get_dummy_void_value(&self) -> V {
        ll::token_none(self.c)
    }
    /// `isStorableType`.
    fn is_storable_type(&self, t: Ty) -> bool {
        !ll::is_void(t) && ll::alloc_size(self.m(), t) != 0
    }
    /// `getDummyValue`.
    /// `insertVar`: C++ `vars.emplace(id, x)`; never overwrites an existing entry (a
    /// global registered at module level must win over a shadowing local alloca, e.g.
    /// a top-level loop variable listed in main's symbols).
    fn insert_var(&mut self, uid: Id, x: V) {
        self.vars.entry(uid).or_insert(x);
    }

    fn get_dummy_value(&self, t: Ty) -> V {
        if ll::is_void(t) { self.get_dummy_void_value() } else { ll::undef(t) }
    }

    // ---------------------------------------------------------------- module

    /// `visit(const Module*)`.
    pub fn visit_module(&mut self, m: &Module) -> Result<()> {
        let module = self.make_module(Some(Self::src_info(m, NodeRef::Module)))?;
        self.module = Some(module);

        // args variable
        assert!(m.is_global(m.argv_var()), "argv var is not global");
        assert!(m.is_global(m.argc_var()), "argc var is not global");
        self.register_global(m, m.argv_var());
        self.register_global(m, m.argc_var());

        // set up global variables and initialize functions
        let vars = m.vars();
        for &var in &vars {
            self.register_global(m, var);
        }
        // process functions
        for &var in &vars {
            if m.is_func(var) {
                self.process_func(m, var);
            }
        }

        let main = m.main_func();
        let real_main = self.make_llvm_function(m, main, false);
        let real_main_ty = ll::function_type(real_main);
        self.process_func(m, main);
        self.set_debug_info_for_node(m, None);

        // build canonical main function
        let b = &self.b;
        let str_type = ll::struct_ty(self.c, &[b.ptr(), b.i64()]);
        let (init_ty, init_func) = self.decl("seq_init", b.void(), &[b.i32()]);
        let (strlen_ty, strlen_func) = self.decl("strlen", b.i64(), &[b.ptr()]);

        // check if main exists already as an exported function
        let main_name = if ll::get_function(self.m(), "main").is_some() { MAIN_UNCLASH } else { "main" };
        let canon_ty = ll::fn_ty(b.i32(), &[b.i32(), b.ptr()], false);
        let canonical = ll::get_or_insert_function(self.m(), main_name, canon_ty);
        let (_, pers) = self.make_personality_func();
        ll::set_personality(canonical, pers);
        let argc = ll::param(canonical, 0);
        let argv = ll::param(canonical, 1);
        ll::set_name(argc, "argc");
        ll::set_name(argv, "argv");

        // for (int i = 0; i < argc; i++) array[i] = {argv[i], strlen(argv[i])}
        let c = self.c;
        let entry = ll::append_block(c, canonical, "entry");
        let loop_ = ll::append_block(c, canonical, "loop");
        let body = ll::append_block(c, canonical, "body");
        let exit = ll::append_block(c, canonical, "exit");

        let b = &self.b;
        b.at_end(entry);
        let (alloc_ty, alloc_func) = self.make_alloc_func(false, false);
        let len = b.zext(argc, b.i64());
        let elem_size = b.c_i64(ll::alloc_size(self.m(), str_type) as i64);
        let alloc_size = b.mul(len, elem_size);
        let ptr = b.call(alloc_ty, alloc_func, &[alloc_size]);
        b.br(loop_);

        b.at_end(loop_);
        let control = b.phi(b.i32());
        ll::set_name(control, "i");
        let next = b.add(control, b.c_i32(1));
        ll::set_name(next, "next");
        let cond = b.icmp(llvm_sys::LLVMIntPredicate::LLVMIntSLT, control, argc);
        Builder::add_incoming(control, b.c_i32(0), entry);
        Builder::add_incoming(control, next, body);
        b.cond_br(cond, body, exit);

        b.at_end(body);
        let arg = b.load(b.ptr(), b.gep(b.ptr(), argv, &[control]));
        let arg_len = b.zext_or_trunc(b.call(strlen_ty, strlen_func, &[arg]), b.i64());
        let mut s = ll::undef(str_type);
        s = b.insert_value(s, arg, 0);
        s = b.insert_value(s, arg_len, 1);
        b.store(s, b.gep(str_type, ptr, &[control]));
        b.br(loop_);

        b.at_end(exit);
        let argv_storage = self.get_var(m, m.argv_var());
        assert!(!argv_storage.is_null(), "argv storage missing");
        self.b.store(ptr, argv_storage);
        let argc_storage = self.get_var(m, m.argc_var());
        assert!(!argc_storage.is_null(), "argc storage missing");
        self.b.store(len, argc_storage);

        let flags = (if self.options.debug { SEQ_FLAG_DEBUG } else { 0 })
            | (if self.options.standalone { SEQ_FLAG_STANDALONE } else { 0 });
        self.b.call(init_ty, init_func, &[self.b.c_i32(flags)]);

        // Put the entire program in a new function
        {
            let proxy_ty = ll::fn_ty(self.b.void(), &[], false);
            let proxy = ll::get_or_insert_function(self.m(), "typhon.proxy_main", proxy_ty);
            ll::set_linkage(proxy, LLVMPrivateLinkage);
            let (_, pers) = self.make_personality_func();
            ll::set_personality(proxy, pers);
            let p_entry = ll::append_block(c, proxy, "entry");
            let p_main = ll::append_block(c, proxy, "main");
            let p_exit = ll::append_block(c, proxy, "exit");
            self.b.at_end(p_entry);
            self.b.cond_br(self.b.c_i1(false), p_exit, p_main);
            self.b.at_end(p_exit);
            self.b.ret_void();

            // invoke real main
            let normal = ll::append_block(c, proxy, "normal");
            let unwind = ll::append_block(c, proxy, "unwind");
            self.b.at_end(p_main);
            self.b.invoke(real_main_ty, real_main, &[], normal, unwind);

            self.b.at_end(unwind);
            if self.is_win_msvc() {
                let cs = self.b.catch_switch(ll::token_none(c), None, 1);
                let catch_block = ll::append_block(c, proxy, "catch");
                Builder::add_handler(cs, catch_block);
                self.b.at_end(catch_block);
                let (_, filter) = self.make_win_eh_filter();
                let cp = self.b.catch_pad(cs, &[filter]);
                let (ect, ecf) = self.exc_current_func();
                let unwind_exc = self.b.call_funclet(ect, ecf, &[], cp);
                let (tt, tf) = self.make_terminate_func();
                self.b.call_funclet(tt, tf, &[unwind_exc], cp);
                self.b.unreachable();
            } else {
                let lp = self.b.landing_pad(self.get_pad_type(), 1);
                Builder::set_cleanup(lp, true);
                Builder::add_clause(lp, self.get_type_idx_var(m, None));
                let unwind_exc = self.b.extract_value(lp, 0);
                let (tt, tf) = self.make_terminate_func();
                self.b.call(tt, tf, &[unwind_exc]);
                self.b.unreachable();
            }
            self.b.at_end(normal);
            self.b.ret_void();

            // actually make the call
            self.b.at_end(exit);
            self.b.call(proxy_ty, proxy, &[]);
        }
        self.b.at_end(exit);
        self.b.ret(self.b.c_i32(0));

        // make sure allocation functions have the correct attributes
        let mm = self.m();
        if ll::get_function(mm, "seq_alloc").is_some() {
            self.make_alloc_func(false, false);
        }
        if ll::get_function(mm, "seq_alloc_atomic").is_some() {
            self.make_alloc_func(true, false);
        }
        if ll::get_function(mm, "seq_alloc_uncollectable").is_some() {
            self.make_alloc_func(false, true);
        }
        if ll::get_function(mm, "seq_alloc_atomic_uncollectable").is_some() {
            self.make_alloc_func(true, true);
        }
        if ll::get_function(mm, "seq_realloc").is_some() {
            self.make_realloc_func();
        }
        if ll::get_function(mm, "seq_free").is_some() {
            self.make_free_func();
        }
        match self.error.take() {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }

    // ---------------------------------------------------------------- functions

    /// `getDISubprogramForFunc`.
    fn get_di_subprogram_for_func(&mut self, m: &Module, f: FuncId) -> Md {
        let src = Self::src_info(m, f.into());
        let file = self.db.file(&Self::src_file(&src));
        let subroutine = self.get_di_subroutine_type(m, m.var_type(f).expect("function without type"));
        let fd = m.func(f).unwrap();
        let mut base_name = fd.unmangled_name.clone();
        if let Some(parent) = fd.parent_type {
            base_name = format!("{}.{}", m.type_name(parent), base_name);
        }
        let linkage = self.get_llvm_name_for_function(m, f);
        self.db.b().function(file, &base_name, &linkage, file, src.line.max(0) as u32, subroutine, true, true, !self.options.debug)
    }

    /// `makeLLVMFunction`.
    fn make_llvm_function(&mut self, m: &Module, f: FuncId, declaration_only: bool) -> V {
        let f = m.actual_var(f);
        let fd = m.func(f).expect("not a function");
        // process LLVM functions in full immediately
        if fd.is_llvm() && !declaration_only {
            let old = self.func;
            self.process_func(m, f);
            self.set_debug_info_for_node(m, None);
            let new = self.func;
            if self.needs_win_export_wrapper(m, f) && !new.is_null() {
                ll::add_string_attr(new, ll::FN_INDEX, "typhon-c-export", &Self::get_name_for_function(m, f));
            }
            self.func = old;
            return new;
        }
        let ft = m.var_type(f).expect("function without type");
        let ret = m.func_type_ret(ft).expect("not a function type");
        let return_type = self.get_llvm_type(m, ret);
        let arg_types: Vec<Ty> = m.func_type_args(ft).into_iter().map(|a| self.get_llvm_type(m, a)).collect();
        let variadic = matches!(m.type_kind(m.actual_type(ft)), TypeKind::Func { variadic: true, .. });
        let name = self.get_llvm_name_for_function(m, f);
        let is_external = fd.is_external();
        let win_abi = is_external && self.is_win_msvc();
        let signature = ll::fn_ty(return_type, &arg_types, variadic);
        let llvm_ty = if win_abi { win::signature(self.m(), signature) } else { signature };
        let func = ll::get_or_insert_function(self.m(), &name, llvm_ty);
        if win_abi && win::kind(self.m(), return_type, true) == win::Kind::Indirect {
            ll::add_type_attr(func, 1, "sret", return_type);
        }
        if self.needs_win_export_wrapper(m, f) {
            ll::add_string_attr(func, ll::FN_INDEX, "typhon-c-export", &Self::get_name_for_function(m, f));
        }
        if !is_external && !declaration_only {
            let sp = self.get_di_subprogram_for_func(m, f);
            ll::set_subprogram(func, sp);
        }
        func
    }

    /// `makeYield`.
    fn make_yield(&mut self, value: Option<V>, final_yield: bool) {
        self.b.at_end(self.block);
        if let Some(v) = value {
            assert!(!self.coro.promise.is_null(), "promise is null");
            self.b.store(v, self.coro.promise);
        }
        let (st, sf) = ll::intrinsic(self.m(), "llvm.coro.suspend", &[]);
        let res = self.b.call(st, sf, &[ll::token_none(self.c), self.b.c_i1(final_yield)]);
        self.block = ll::append_block(self.c, self.func, "yield.new");
        assert!(!self.coro.suspend.is_null() && !self.coro.cleanup.is_null(), "suspend and/or cleanup is null");
        let sw = self.b.switch(res, self.coro.suspend, 2);
        Builder::add_case(sw, self.b.c_i8(0), self.block);
        Builder::add_case(sw, self.b.c_i8(1), self.coro.cleanup);
    }

    /// `visit(const ExternalFunc*)`.
    fn visit_external_func(&mut self, m: &Module, f: FuncId) {
        let name = self.get_llvm_name_for_function(m, f);
        self.func = match ll::get_function(self.m(), &name) {
            Some(x) => x,
            None => {
                let x = self.make_llvm_function(m, f, false);
                self.funcs.insert(m.var_uid(f), x);
                x
            }
        };
        self.coro = CoroData::default();
        let func = self.func;
        ll::add_fn_attr(func, "nounwind");
        ll::add_fn_attr(func, "willreturn");
        if irtools::has_attribute(m, f, &llvm_memory_none_attr()) {
            ll::set_memory_none(func);
        }
        if irtools::has_attribute(m, f, &llvm_no_free_attr()) {
            ll::add_fn_attr(func, "nofree");
        }
        if irtools::has_attribute(m, f, &llvm_no_capture_attr()) {
            for (i, p) in ll::params(func).into_iter().enumerate() {
                if ll::is_pointer(ll::type_of(p)) {
                    ll::add_nocapture(func, i as u32 + 1);
                }
            }
        }
    }

    /// `visit(const InternalFunc*)`.
    fn visit_internal_func(&mut self, m: &Module, f: FuncId) {
        // PORT-NOTE: `hasValidIntrinsicSignature` needs `&mut Module` in the Rust IR;
        // the type checker validates intrinsic signatures before codegen.
        let fd = m.func(f).unwrap();
        if fd.intrinsic().is_type_query() {
            return;
        }
        let name = self.get_llvm_name_for_function(m, f);
        self.func = ll::get_function(self.m(), &name).unwrap_or_else(|| panic!("{name} not inserted"));
        self.coro = CoroData::default();
        self.set_debug_info_for_node(m, Some(f.into()));
        ll::set_linkage(self.func, LLVMPrivateLinkage);
        ll::add_fn_attr(self.func, "alwaysinline");
        let args = ll::params(self.func);
        self.block = ll::append_block(self.c, self.func, "entry");
        self.b.at_end(self.block);
        let result = self.codegen_intrinsic(m, f, &args);
        if ll::terminator(self.b.insert_block()).is_some() {
            return;
        }
        let result = result.unwrap_or_else(|| panic!("internal function {name} not found"));
        self.b.ret(result);
    }

    /// `buildLLVMCodeString`.
    fn build_llvm_code_string(&mut self, m: &Module, f: FuncId) -> String {
        let ft = m.var_type(f).expect("function without type");
        let ret = m.func_type_ret(ft).expect("not a function type");
        let mut sig = String::from("define ");
        let rt = self.get_llvm_type(m, ret);
        sig += &ll::type_to_string(rt);
        sig += &format!(" @\"{}\"(", self.get_llvm_name_for_function(m, f));
        let args = m.func_args(f);
        let n = args.len();
        for (i, a) in args.iter().enumerate() {
            let t = self.get_llvm_type(m, m.var_type(*a).unwrap());
            sig += &ll::type_to_string(t);
            sig += &format!(" %{}", m.var_name(*a));
            if i + 1 < n {
                sig += ", ";
            }
        }
        sig += ")";
        // replace literal '{' and '}'
        let sig = sig.replace('{', "{{").replace('}', "}}");
        let fd = m.func(f).unwrap();
        format!("{}\n{} {{{{\n{}\n}}}}", fd.llvm_declarations(), sig, fd.llvm_body())
    }

    /// `visit(const LLVMFunc*)`.
    fn visit_llvm_func(&mut self, m: &Module, f: FuncId) {
        let name = self.get_llvm_name_for_function(m, f);
        let existing = ll::get_function(self.m(), &name);
        self.coro = CoroData::default();
        if let Some(e) = existing {
            self.func = e;
            return;
        }
        self.func = null_mut();

        if self.is_win_msvc() && irtools::has_attribute(m, f, &llvm_c_call_attr()) {
            let signature = self.get_llvm_func_type(m, m.var_type(f).unwrap());
            let func = ll::add_function(self.m(), &name, signature);
            ll::set_linkage(func, LLVMPrivateLinkage);
            ll::add_fn_attr(func, "alwaysinline");
            let sp = self.get_di_subprogram_for_func(m, f);
            ll::set_subprogram(func, sp);
            self.func = func;
            let tb = Builder::new(self.c);
            tb.at_end(ll::append_block(self.c, func, "entry"));
            let src = Self::src_info(m, f.into());
            tb.set_debug_loc(ll::di_location(self.c, src.line, src.col, sp));
            let tuple = ll::param(func, 1);
            let tuple_ty = ll::type_of(tuple);
            let elems = ll::struct_elements(tuple_ty);
            let arguments: Vec<V> = (0..elems.len()).map(|i| tb.extract_value(tuple, i as u32)).collect();
            let native = ll::fn_ty(ll::fn_ret_ty(signature), &elems, false);
            let result = win::call(self.m(), &tb, ll::param(func, 0), native, &arguments);
            if ll::is_void(ll::fn_ret_ty(signature)) {
                tb.ret_void();
            } else {
                tb.ret(result.expect("missing result"));
            }
            return;
        }

        // build code
        let code = self.build_llvm_code_string(m, f);
        // format code
        let mut literals = Vec::new();
        for lit in m.func(f).unwrap().llvm_literals().to_vec() {
            use typhon_ir::types::Generic;
            match lit {
                Generic::Static(v) => literals.push(v.to_string()),
                Generic::StaticStr(s) => literals.push(s),
                Generic::Type(t) => {
                    let lt = self.get_llvm_type(m, t);
                    literals.push(ll::type_to_string(lt));
                }
            }
        }
        let code = match fmt_format(&code, &literals) {
            Ok(c) => c,
            Err(e) => {
                self.fail(format!("{e} ({})", m.var_name(f)));
                return;
            }
        };
        if let Err(e) = ll::parse_and_link(self.m(), &code) {
            self.fail(format!("LLVM: {e} ({})", m.var_name(f)));
            return;
        }
        let func = ll::get_function(self.m(), &name).expect("function not linked in");
        self.func = func;
        ll::set_linkage(func, LLVMPrivateLinkage);
        ll::add_fn_attr(func, "alwaysinline");
        let sp = self.get_di_subprogram_for_func(m, f);
        ll::set_subprogram(func, sp);
        // set up debug info: for now we just set all to func's source location
        let src = Self::src_info(m, f.into());
        for bb in ll::blocks(func) {
            for inst in ll::instructions(bb) {
                if ll::inst_debug_loc(inst).is_null() {
                    ll::inst_set_debug_loc(inst, ll::di_location(self.c, src.line, src.col, sp));
                }
            }
        }
    }

    /// `visit(const BodiedFunc*)`.
    fn visit_bodied_func(&mut self, m: &Module, f: FuncId) {
        let name = self.get_llvm_name_for_function(m, f);
        self.func = ll::get_function(self.m(), &name).unwrap_or_else(|| panic!("{name} not inserted"));
        self.coro = CoroData::default();
        self.set_debug_info_for_node(m, Some(f.into()));
        let func = self.func;
        let fd = m.func(f).unwrap();
        if irtools::is_export(m, f) {
            ll::set_linkage(func, LLVMExternalLinkage);
            // Used by library builds: thread registration + export list.
            ll::add_string_attr(func, ll::FN_INDEX, EXPORT_ATTR, "");
        } else {
            ll::set_linkage(func, LLVMPrivateLinkage);
        }
        if irtools::is_inline(m, f) {
            ll::add_fn_attr(func, "alwaysinline");
        }
        if irtools::is_no_inline(m, f) {
            ll::add_fn_attr(func, "noinline");
        }
        if irtools::is_gpu_kernel(m, f) {
            ll::add_fn_attr(func, "noinline");
            ll::add_string_attr(func, ll::FN_INDEX, "kernel", "");
            ll::set_linkage(func, LLVMExternalLinkage);
        }
        if !self.options.noexc {
            let (_, p) = self.make_personality_func();
            ll::set_personality(func, p);
        }

        let ft = m.var_type(f).expect("function without type");
        let return_type = m.func_type_ret(ft).expect("not a function type");
        let entry = ll::append_block(self.c, func, "entry");
        self.b.at_end(entry);

        // set up arguments and other symbols
        let args = fd.args.clone();
        assert_eq!(ll::params(func).len(), args.len(), "argument length does not match");
        let sp = ll::subprogram(func);
        for (i, &var) in args.iter().enumerate() {
            let vt = m.var_type(var).unwrap();
            let llvm_type = self.get_llvm_type(m, vt);
            self.b.at_end(entry);
            let storage = self.b.alloca(llvm_type);
            self.insert_var(m.var_uid(var), storage);
            if self.is_storable_type(llvm_type) {
                self.b.store(ll::param(func, i as u32), storage);
            }
            // debug info
            let src = Self::src_info(m, var.into());
            let file = self.db.file(&Self::src_file(&src));
            let dt = self.get_di_type(m, vt);
            let dv = self.db.b().parameter_variable(sp, &Self::get_debug_name_for_variable(m, var), i as u32 + 1, file, src.line.max(0) as u32, dt, self.options.debug);
            let loc = ll::di_location(self.c, src.line, src.col, sp);
            self.db.b().insert_declare(storage, dv, loc, entry);
        }

        for var in fd.symbols().to_vec() {
            let vt = m.var_type(var).unwrap();
            let llvm_type = self.get_llvm_type(m, vt);
            if ll::is_void(llvm_type) {
                let d = self.get_dummy_value(llvm_type);
                self.insert_var(m.var_uid(var), d);
            } else {
                self.b.at_end(entry);
                let storage = self.b.alloca(llvm_type);
                self.insert_var(m.var_uid(var), storage);
                let src = Self::src_info(m, var.into());
                let file = self.db.file(&Self::src_file(&src));
                let dt = self.get_di_type(m, vt);
                let dv = self.db.b().auto_variable(sp, &Self::get_debug_name_for_variable(m, var), file, src.line.max(0) as u32, dt, self.options.debug);
                let loc = ll::di_location(self.c, src.line, src.col, sp);
                self.db.b().insert_declare(storage, dv, loc, entry);
            }
        }
        self.b.at_end(entry);

        let start = ll::append_block(self.c, func, "start");
        let generator = fd.generator || fd.is_async;
        if generator {
            if !fd.is_async {
                ll::add_string_attr(func, ll::FN_INDEX, "typhon.generator", "");
            }
            ll::add_fn_attr(func, "presplitcoroutine");
            let base = match m.type_kind(m.actual_type(return_type)) {
                TypeKind::Generator { base } => *base,
                _ => panic!("{} is not a generator type", m.type_name(return_type)),
            };
            let mm = self.m();
            let (id_t, id_f) = ll::intrinsic(mm, "llvm.coro.id", &[]);
            let (begin_t, begin_f) = ll::intrinsic(mm, "llvm.coro.begin", &[]);
            let (size_t, size_f) = ll::intrinsic(mm, "llvm.coro.size", &[self.b.i64()]);
            let (end_t, end_f) = ll::intrinsic(mm, "llvm.coro.end", &[]);
            let (alloc_t, alloc_f) = ll::intrinsic(mm, "llvm.coro.alloc", &[]);
            let (free_t, free_f) = ll::intrinsic(mm, "llvm.coro.free", &[]);

            self.coro.cleanup = ll::append_block(self.c, func, "coro.cleanup");
            self.coro.suspend = ll::append_block(self.c, func, "coro.suspend");
            self.coro.exit = ll::append_block(self.c, func, "coro.exit");
            self.coro.is_async = fd.is_async;
            let alloc_block = ll::append_block(self.c, func, "coro.alloc");
            let free_block = ll::append_block(self.c, func, "coro.free");

            // coro ID and promise
            let null = self.b.null_ptr();
            let pt = self.get_llvm_type(m, base);
            self.b.at_end(entry);
            self.coro.promise = self.b.alloca(pt);
            ll::set_name(self.coro.promise, "coro.promise");
            let id = self.b.call(id_t, id_f, &[self.b.c_i32(0), self.coro.promise, null, null]);
            ll::set_name(id, "coro.id");
            let need_alloc = self.b.call(alloc_t, alloc_f, &[id]);
            self.b.cond_br(need_alloc, alloc_block, start);

            // coro alloc
            self.b.at_end(alloc_block);
            let size = self.b.call(size_t, size_f, &[]);
            let (af_t, af) = self.make_alloc_func(false, false);
            let alloc = self.b.call(af_t, af, &[size]);
            self.b.br(start);

            // coro start
            self.b.at_end(start);
            let phi = self.b.phi(self.b.ptr());
            Builder::add_incoming(phi, null, entry);
            Builder::add_incoming(phi, alloc, alloc_block);
            self.coro.handle = self.b.call(begin_t, begin_f, &[id, phi]);
            ll::set_name(self.coro.handle, "coro.handle");

            // coro cleanup
            self.b.at_end(self.coro.cleanup);
            let mem = self.b.call(free_t, free_f, &[id, self.coro.handle]);
            let need_free = self.b.is_not_null(mem);
            self.b.cond_br(need_free, free_block, self.coro.suspend);

            // coro free: no-op, GC will free automatically
            self.b.at_end(free_block);
            self.b.br(self.coro.suspend);

            // coro suspend
            self.b.at_end(self.coro.suspend);
            self.b.call(end_t, end_f, &[self.coro.handle, self.b.c_i1(false), ll::token_none(self.c)]);
            self.b.ret(self.coro.handle);

            // coro exit
            self.block = self.coro.exit;
            self.make_yield(None, true);
            self.b.at_end(self.block);
            self.b.unreachable();

            // initial yield: coroutine will be initially suspended
            self.block = start;
            self.make_yield(None, false);
        } else {
            self.b.br(start);
            self.block = start;
        }

        let body = fd.body().unwrap_or_else(|| panic!("{} has no body", m.var_name(f)));
        self.process_value(m, body);
        self.b.at_end(self.block);
        if generator {
            self.b.br(self.coro.exit);
        } else {
            let rt = self.get_llvm_type(m, return_type);
            if ll::is_void(rt) {
                self.b.ret_void();
            } else {
                self.b.ret(ll::const_null(rt));
            }
        }
    }

    // ---------------------------------------------------------------- dispatch

    fn visit_value(&mut self, m: &Module, v: ValueId) {
        use typhon_ir::value::ValueKind as K;
        match m.kind(v) {
            K::IntConst(x) => {
                self.b.at_end(self.block);
                self.value = self.b.c_i64(x.val);
            }
            K::FloatConst(x) => {
                self.b.at_end(self.block);
                self.value = ll::const_real(ll::double_ty(self.c), x.val);
            }
            K::BoolConst(x) => {
                self.b.at_end(self.block);
                self.value = self.b.c_i8(x.val as i64);
            }
            K::StringConst(x) => {
                let s = x.val.clone();
                self.b.at_end(self.block);
                self.value = self.string_const(&s);
            }
            K::BytesConst(x) => {
                let s = x.val.clone();
                self.b.at_end(self.block);
                // PORT-NOTE: bytes literals are stored byte-per-char (Latin-1); raw `rb`
                // literals may contain arbitrary text, which is emitted as UTF-8.
                let raw: Vec<u8> = if s.chars().all(|c| (c as u32) < 0x100) {
                    s.chars().map(|c| c as u8).collect()
                } else {
                    s.into_bytes()
                };
                self.value = self.bytes_const(&raw);
            }
            K::VarValue(_) => self.visit_var_value(m, v),
            K::PointerValue(_) => self.visit_pointer_value(m, v),
            K::SeriesFlow(x) => {
                for s in x.series.clone() {
                    self.process_value(m, s);
                }
            }
            K::IfFlow(_) => self.visit_if_flow(m, v),
            K::WhileFlow(_) => self.visit_while_flow(m, v),
            K::ForFlow(_) => self.visit_for_flow(m, v),
            K::ImperativeForFlow(_) => self.visit_imperative_for_flow(m, v),
            K::TryCatchFlow(_) => self.visit_try_catch_flow(m, v),
            K::PipelineFlow(_) => self.visit_pipeline_flow(m, v),
            K::AssignInstr(_) => self.visit_assign_instr(m, v),
            K::ExtractInstr(_) => self.visit_extract_instr(m, v),
            K::InsertInstr(_) => self.visit_insert_instr(m, v),
            K::CallInstr(_) => self.visit_call_instr(m, v),
            K::StackAllocInstr(_) => self.visit_stack_alloc_instr(m, v),
            K::YieldInInstr(_) => self.visit_yield_in_instr(m, v),
            K::TernaryInstr(_) => self.visit_ternary_instr(m, v),
            K::BreakInstr(_) => self.visit_break_instr(m, v),
            K::ContinueInstr(_) => self.visit_continue_instr(m, v),
            K::ReturnInstr(_) => self.visit_return_instr(m, v),
            K::YieldInstr(_) => self.visit_yield_instr(m, v),
            K::AwaitInstr(_) => panic!("await instruction not lowered"),
            K::ThrowInstr(_) => self.visit_throw_instr(m, v),
            K::FlowInstr(x) => {
                let (flow, val) = (x.flow, x.val);
                self.process_value(m, flow);
                self.process_value(m, val);
            }
            K::Any { .. } => panic!("cannot codegen wildcard value"),
        }
    }
}

/// Minimal `fmt::vformat` with positional `{}` / `{N}` arguments and `{{`/`}}` escapes.
fn fmt_format(code: &str, args: &[String]) -> std::result::Result<String, String> {
    let mut out = String::with_capacity(code.len());
    let bytes = code.as_bytes();
    let mut i = 0;
    let mut next = 0usize;
    while i < bytes.len() {
        let ch = bytes[i];
        if ch == b'{' {
            if i + 1 < bytes.len() && bytes[i + 1] == b'{' {
                out.push('{');
                i += 2;
                continue;
            }
            let close = code[i..].find('}').ok_or("formatting failed: unmatched '{'")? + i;
            let spec = &code[i + 1..close];
            let idx = if spec.is_empty() {
                let k = next;
                next += 1;
                k
            } else {
                let num = spec.split(':').next().unwrap_or("");
                num.parse::<usize>().map_err(|_| format!("formatting failed: bad argument '{spec}'"))?
            };
            out.push_str(args.get(idx).ok_or("formatting failed: argument index out of range")?);
            i = close + 1;
        } else if ch == b'}' {
            if i + 1 < bytes.len() && bytes[i + 1] == b'}' {
                out.push('}');
                i += 2;
                continue;
            }
            return Err("formatting failed: unmatched '}'".into());
        } else {
            let s = &code[i..];
            let c = s.chars().next().unwrap();
            out.push(c);
            i += c.len_utf8();
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn format_literals() {
        assert_eq!(fmt_format("a {{x}} {} {}", &["1".into(), "i64".into()]).unwrap(), "a {x} 1 i64");
        assert_eq!(fmt_format("{1} {0}", &["a".into(), "b".into()]).unwrap(), "b a");
    }
    #[test]
    fn exc_offset() {
        assert_eq!(seq_exc_offset(8), -32);
    }
}
