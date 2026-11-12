//! Port of `codon/cir/llvm/gpu.{h,cpp}`: NVPTX module preparation, PTX
//! generation and host-module patching.

use inkwell::module::Module;
use llvm_sys::LLVMLinkage as L;
use llvm_sys::LLVMOpcode as Op;
use llvm_sys::core::*;
use llvm_sys::prelude::*;
use rustc_hash::FxHashSet;
use typhon_common::{CompilerError, Options, Result};

use crate::ffi::{self, BB, V};

pub const GPU_TRIPLE: &str = "nvptx64-nvidia-cuda";
pub const GPU_DL: &str = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-\
f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64";

fn err(msg: impl Into<String>) -> CompilerError {
    CompilerError::Llvm(msg.into())
}

// ------------------------------------------------------------------ GVExtractor

/// `GVExtractor::makeVisible`.
fn make_visible(gv: V, del: bool) {
    let local = ffi::has_local_linkage(gv);
    if local || del {
        ffi::set_linkage(gv, L::LLVMExternalLinkage);
        if local {
            unsafe { LLVMSetVisibility(gv, llvm_sys::LLVMVisibility::LLVMHiddenVisibility) };
        }
        return;
    }
    match ffi::linkage(gv) {
        L::LLVMLinkOnceAnyLinkage => ffi::set_linkage(gv, L::LLVMWeakAnyLinkage),
        L::LLVMLinkOnceODRLinkage => ffi::set_linkage(gv, L::LLVMWeakODRLinkage),
        _ => {}
    }
}

/// `Function::deleteBody()` via the C API: drop all instructions and blocks,
/// then make the function an external declaration.
pub(crate) fn delete_body(f: V) {
    let blocks = ffi::blocks(f);
    for &b in &blocks {
        for i in ffi::instructions(b) {
            if !ffi::is_void_ty(ffi::type_of(i)) && !ffi::use_empty(i) {
                ffi::replace_all_uses(i, unsafe { LLVMGetPoison(ffi::type_of(i)) });
            }
        }
    }
    for &b in &blocks {
        for i in ffi::instructions(b).into_iter().rev() {
            ffi::erase(i);
        }
    }
    for b in blocks {
        unsafe { LLVMDeleteBasicBlock(b) };
    }
    unsafe {
        if LLVMHasPersonalityFn(f) != 0 {
            LLVMSetPersonalityFn(f, std::ptr::null_mut());
        }
        let dbg = ffi::md_kind(ffi::value_context(f), "dbg");
        LLVMGlobalEraseMetadata(f, dbg);
    }
    ffi::set_linkage(f, L::LLVMExternalLinkage);
}

/// Port of `GVExtractor(named, deleteStuff=false)::run`: keep only `named`.
///
/// PORT-NOTE: global aliases are only made visible, never replaced by
/// declarations (the C API cannot erase an alias); Codon never emits aliases.
fn gv_extract_keep(m: LLVMModuleRef, named: &FxHashSet<V>) {
    unsafe { LLVMSetModuleInlineAsm2(m, c"".as_ptr(), 0) };
    for gv in ffi::globals(m) {
        let del = !named.contains(&gv) && !ffi::is_declaration(gv);
        if !del {
            if ffi::linkage(gv) == L::LLVMAvailableExternallyLinkage {
                continue;
            }
            if ffi::name(gv) == "llvm.global_ctors" {
                continue;
            }
        }
        make_visible(gv, del);
        if del {
            unsafe {
                LLVMSetInitializer(gv, std::ptr::null_mut());
                llvm_sys::comdat::LLVMSetComdat(gv, std::ptr::null_mut());
            }
        }
    }
    for f in ffi::functions(m) {
        let del = !named.contains(&f) && !ffi::is_declaration(f);
        if !del && ffi::linkage(f) == L::LLVMAvailableExternallyLinkage {
            continue;
        }
        make_visible(f, del);
        if del {
            delete_body(f);
            unsafe { llvm_sys::comdat::LLVMSetComdat(f, std::ptr::null_mut()) };
        }
    }
    for a in ffi::aliases(m) {
        let del = !named.contains(&a);
        make_visible(a, del);
    }
}

/// `getRequiredGVs` / `exploreGV`.
fn required_gvs(kernels: &[V]) -> FxHashSet<V> {
    let mut keep = FxHashSet::default();
    fn explore(g: V, keep: &mut FxHashSet<V>) {
        if !keep.insert(g) {
            return;
        }
        if ffi::is_function(g) {
            for i in ffi::fn_instructions(g) {
                for op in ffi::operands(i) {
                    collect_globals(op, keep, 0);
                }
            }
        }
    }
    fn collect_globals(v: V, keep: &mut FxHashSet<V>, depth: u32) {
        if v.is_null() {
            return;
        }
        if ffi::is_global_value(v) {
            explore(v, keep);
        }
        // PORT-NOTE: C++ only follows direct GlobalValue operands (constant
        // expressions wrapping globals are not explored). Same here.
        let _ = depth;
    }
    for &k in kernels {
        explore(k, &mut keep);
    }
    keep
}

fn prune(m: LLVMModuleRef, keep: &FxHashSet<V>) -> Result<()> {
    gv_extract_keep(m, keep);
    crate::optimize::run_pipeline(m, "globaldce,strip-dead-debug-info,strip-dead-prototypes", std::ptr::null_mut())
}

fn clean_up_name(name: &str) -> String {
    name.chars()
        .enumerate()
        .map(|(i, c)| {
            let ok = c.is_ascii_alphabetic() || c == '_' || (i > 0 && c.is_ascii_digit());
            if ok { c } else { '_' }
        })
        .collect()
}

fn link_libdevice(m: LLVMModuleRef, path: &str) -> Result<()> {
    unsafe {
        let ctx = ffi::module_context(m);
        let cpath = ffi::cstr(path);
        let mut buf: LLVMMemoryBufferRef = std::ptr::null_mut();
        let mut msg: *mut libc::c_char = std::ptr::null_mut();
        if LLVMCreateMemoryBufferWithContentsOfFile(cpath.as_ptr(), &mut buf, &mut msg) != 0 {
            let s = std::ffi::CStr::from_ptr(msg).to_string_lossy().into_owned();
            LLVMDisposeMessage(msg);
            return Err(err(format!("{path}: {s}")));
        }
        let mut lib: LLVMModuleRef = std::ptr::null_mut();
        if llvm_sys::ir_reader::LLVMParseIRInContext(ctx, buf, &mut lib, &mut msg) != 0 {
            let s = std::ffi::CStr::from_ptr(msg).to_string_lossy().into_owned();
            LLVMDisposeMessage(msg);
            return Err(err(format!("{path}: {s}")));
        }
        LLVMSetDataLayout(lib, LLVMGetDataLayoutStr(m));
        LLVMSetTarget(lib, LLVMGetTarget(m));
        if llvm_sys::linker::LLVMLinkModules2(m, lib) != 0 {
            return Err(err("linking libdevice failed"));
        }
    }
    Ok(())
}

fn copy_prototype(f: V, name: &str, external: bool) -> V {
    let m = ffi::global_module(f);
    let n = ffi::cstr(if name.is_empty() { "" } else { name });
    let g = unsafe { LLVMAddFunction(m, n.as_ptr(), ffi::function_type(f)) };
    if name.is_empty() {
        ffi::set_name(g, &ffi::name(f));
    }
    ffi::set_linkage(g, if external { L::LLVMExternalLinkage } else { L::LLVMPrivateLinkage });
    g
}

fn make_no_op(f: V) -> V {
    let m = ffi::global_module(f);
    let dummy_name = format!(".codon.gpu.dummy.{}", ffi::name(f));
    if let Some(d) = ffi::get_function(m, &dummy_name) {
        return d;
    }
    let dummy = copy_prototype(f, &dummy_name, false);
    let ctx = ffi::module_context(m);
    unsafe {
        let entry = LLVMAppendBasicBlockInContext(ctx, dummy, c"entry".as_ptr());
        let b = ffi::Builder::new(ctx);
        b.at_end(entry);
        let rt = ffi::return_type(f);
        if ffi::is_void_ty(rt) {
            LLVMBuildRetVoid(b.0);
        } else {
            LLVMBuildRet(b.0, LLVMGetUndef(rt));
        }
    }
    dummy
}

/// IR building context for fill-ins (`llvm::IRBuilder<>` equivalent).
struct Ir {
    b: ffi::Builder,
    ctx: LLVMContextRef,
    m: LLVMModuleRef,
    f: V,
}

impl Ir {
    fn ptr(&self) -> LLVMTypeRef {
        ffi::ptr_ty(self.ctx)
    }
    fn i64(&self) -> LLVMTypeRef {
        unsafe { LLVMInt64TypeInContext(self.ctx) }
    }
    fn i8(&self) -> LLVMTypeRef {
        unsafe { LLVMInt8TypeInContext(self.ctx) }
    }
    fn c64(&self, v: u64) -> V {
        unsafe { LLVMConstInt(self.i64(), v, 0) }
    }
    fn block(&self, name: &std::ffi::CStr) -> BB {
        unsafe { LLVMAppendBasicBlockInContext(self.ctx, self.f, name.as_ptr()) }
    }
    fn call(&self, callee: V, args: &mut [V]) -> V {
        unsafe { LLVMBuildCall2(self.b.0, ffi::function_type(callee), callee, args.as_mut_ptr(), args.len() as u32, c"".as_ptr()) }
    }
    fn alloca(&self, t: LLVMTypeRef) -> V {
        unsafe { LLVMBuildAlloca(self.b.0, t, c"".as_ptr()) }
    }
    fn store(&self, v: V, p: V) {
        unsafe { LLVMBuildStore(self.b.0, v, p) };
    }
    fn load(&self, t: LLVMTypeRef, p: V) -> V {
        unsafe { LLVMBuildLoad2(self.b.0, t, p, c"".as_ptr()) }
    }
    fn gep(&self, t: LLVMTypeRef, p: V, idx: V) -> V {
        let mut i = [idx];
        unsafe { LLVMBuildGEP2(self.b.0, t, p, i.as_mut_ptr(), 1, c"".as_ptr()) }
    }
    fn ret_undef(&self) {
        unsafe { LLVMBuildRet(self.b.0, LLVMGetUndef(ffi::return_type(self.f))) };
    }

    /// `getOrInsertFunction(name, ret, params...)` + `setWillReturn()`.
    fn get_or_insert(&self, name: &str, ret: LLVMTypeRef, params: &mut [LLVMTypeRef]) -> V {
        let f = match ffi::get_function(self.m, name) {
            Some(f) => f,
            None => unsafe {
                let fty = LLVMFunctionType(ret, params.as_mut_ptr(), params.len() as u32, 0);
                let n = ffi::cstr(name);
                LLVMAddFunction(self.m, n.as_ptr(), fty)
            },
        };
        ffi::add_fn_enum_attr(f, "willreturn", 0);
        f
    }

    fn malloc(&self) -> V {
        let g = match ffi::get_function(self.m, "malloc") {
            Some(g) => g,
            None => unsafe {
                let mut p = [self.i64()];
                let fty = LLVMFunctionType(self.ptr(), p.as_mut_ptr(), 1, 0);
                LLVMAddFunction(self.m, c"malloc".as_ptr(), fty)
            },
        };
        ffi::set_linkage(g, L::LLVMExternalLinkage);
        ffi::add_fn_enum_attr(g, "nounwind", 0);
        unsafe {
            let a = LLVMCreateEnumAttribute(self.ctx, ffi::attr_kind("noalias"), 0);
            LLVMAddAttributeAtIndex(g, ffi::RET_INDEX, a);
            // setOnlyAccessesInaccessibleMemory: memory(inaccessiblemem: readwrite)
            // PORT-NOTE: the `memory` attribute's integer encoding is (ModRef::ModRef
            // << (2 * InaccessibleMem=1)) = 3 << 2 = 12.
            let a = LLVMCreateEnumAttribute(self.ctx, ffi::attr_kind("memory"), 12);
            LLVMAddAttributeAtIndex(g, ffi::FN_INDEX, a);
        }
        ffi::add_fn_enum_attr(g, "willreturn", 0);
        g
    }

    fn intrinsic(&self, name: &str, tys: &mut [LLVMTypeRef]) -> V {
        let id = ffi::lookup_intrinsic(name);
        unsafe { LLVMGetIntrinsicDeclaration(self.m, id, tys.as_mut_ptr(), tys.len()) }
    }
}

#[derive(Clone, Copy)]
enum VecLoop {
    Unary,
    Binary,
    Complex,
}

/// `codegenVectorized{Unary,Binary,Complex}Loop`.
fn codegen_vec_loop(ir: &Ir, args: &[V], func: V, kind: VecLoop) {
    let ty = ffi::return_type(func);
    let (ins, strides, out, os, n): (Vec<V>, Vec<V>, V, V, V) = match kind {
        VecLoop::Unary | VecLoop::Complex => (vec![args[0]], vec![args[1]], args[2], args[3], args[4]),
        VecLoop::Binary => (vec![args[0], args[2]], vec![args[1], args[3]], args[4], args[5], args[6]),
    };
    let lp = ir.block(c"loop");
    let exit = ir.block(c"exit");
    let pin_stores: Vec<V> = ins.iter().map(|_| ir.alloca(ir.ptr())).collect();
    let pout_store = ir.alloca(ir.ptr());
    let idx_store = ir.alloca(ir.i64());
    for (s, &i) in pin_stores.iter().zip(&ins) {
        ir.store(i, *s);
    }
    ir.store(out, pout_store);
    ir.store(ir.c64(0), idx_store);
    unsafe {
        let c = LLVMBuildICmp(ir.b.0, llvm_sys::LLVMIntPredicate::LLVMIntSGT, n, ir.c64(0), c"".as_ptr());
        LLVMBuildCondBr(ir.b.0, c, lp, exit);
    }
    ir.b.at_end(lp);
    let pins: Vec<V> = pin_stores.iter().map(|&s| ir.load(ir.ptr(), s)).collect();
    let pout = ir.load(ir.ptr(), pout_store);
    let y = match kind {
        VecLoop::Unary => {
            let x = ir.load(ty, pins[0]);
            ir.call(func, &mut [x])
        }
        VecLoop::Binary => {
            let x1 = ir.load(ty, pins[0]);
            let x2 = ir.load(ty, pins[1]);
            ir.call(func, &mut [x1, x2])
        }
        VecLoop::Complex => {
            let pim = ir.gep(ty, pins[0], ir.c64(1));
            let re = ir.load(ty, pins[0]);
            let im = ir.load(ty, pim);
            ir.call(func, &mut [re, im])
        }
    };
    ir.store(y, pout);
    let idx = ir.load(ir.i64(), idx_store);
    unsafe {
        let next = LLVMBuildAdd(ir.b.0, idx, ir.c64(1), c"".as_ptr());
        ir.store(next, idx_store);
    }
    for ((&p, &s), &st) in pins.iter().zip(&strides).zip(&pin_stores) {
        let g = ir.gep(ir.i8(), p, s);
        ir.store(g, st);
    }
    let g = ir.gep(ir.i8(), pout, os);
    ir.store(g, pout_store);
    let idx = ir.load(ir.i64(), idx_store);
    unsafe {
        let c = LLVMBuildICmp(ir.b.0, llvm_sys::LLVMIntPredicate::LLVMIntSLT, idx, n, c"".as_ptr());
        LLVMBuildCondBr(ir.b.0, c, lp, exit);
    }
    ir.b.at_end(exit);
    ir.ret_undef();
}

#[derive(Clone, Copy)]
enum FillIn {
    Memcmp,
    Alloc,
    Realloc,
    Calloc,
    Unreachable,
    Vec(VecLoop, &'static str, bool /* f32 */),
}

fn fill_in_table() -> Vec<(&'static str, FillIn)> {
    use FillIn::*;
    use VecLoop::*;
    let mut t = vec![
        ("memcmp", Memcmp),
        ("seq_alloc", Alloc),
        ("seq_alloc_uncollectable", Alloc),
        ("seq_alloc_atomic", Alloc),
        ("seq_alloc_atomic_uncollectable", Alloc),
        ("seq_realloc", Realloc),
        ("seq_calloc", Calloc),
        ("seq_calloc_atomic", Calloc),
        ("seq_alloc_exc", Unreachable),
        ("seq_throw", Unreachable),
    ];
    let funcs: [(&str, VecLoop); 21] = [
        ("acos", Unary),
        ("acosh", Unary),
        ("asin", Unary),
        ("asinh", Unary),
        ("atan", Unary),
        ("atanh", Unary),
        ("atan2", Binary),
        ("exp", Unary),
        ("exp2", Unary),
        ("expm1", Unary),
        ("log", Unary),
        ("log10", Unary),
        ("log1p", Unary),
        ("log2", Unary),
        ("sin", Unary),
        ("sinh", Unary),
        ("cos", Unary),
        ("cosh", Unary),
        ("tan", Unary),
        ("tanh", Unary),
        ("hypot", Binary),
    ];
    // Names are leaked once: the table is tiny and static in C++.
    static NAMES: once_cell::sync::Lazy<std::vec::Vec<(String, String, String, String)>> = once_cell::sync::Lazy::new(|| {
        [
            "acos", "acosh", "asin", "asinh", "atan", "atanh", "atan2", "exp", "exp2", "expm1", "log", "log10", "log1p",
            "log2", "sin", "sinh", "cos", "cosh", "tan", "tanh", "hypot",
        ]
        .iter()
        .map(|f| (format!("cnp_{f}_float64"), format!("__nv_{f}"), format!("cnp_{f}_float32"), format!("__nv_{f}f")))
        .collect()
    });
    for (i, (_, k)) in funcs.iter().enumerate() {
        let (l64, f64n, _, _) = &NAMES[i];
        t.push((l64.as_str(), Vec(*k, f64n.as_str(), false)));
    }
    for (i, (_, k)) in funcs.iter().enumerate() {
        let (_, _, l32, f32n) = &NAMES[i];
        t.push((l32.as_str(), Vec(*k, f32n.as_str(), true)));
    }
    t.push(("cnp_abs_complex128", Vec(Complex, "__nv_hypot", false)));
    t.push(("cnp_abs_complex64", Vec(Complex, "__nv_hypotf", true)));
    t
}

fn codegen_fill_in(ir: &Ir, args: &[V], kind: FillIn) {
    match kind {
        FillIn::Memcmp => unsafe {
            let entry = LLVMGetInsertBlock(ir.b.0);
            let lp = ir.block(c"loop");
            let next = ir.block(c"next");
            let different = ir.block(c"different");
            let equal = ir.block(c"equal");
            let size_ty = ffi::type_of(args[2]);
            let zero = LLVMConstInt(size_ty, 0, 0);
            let one = LLVMConstInt(size_ty, 1, 0);
            let c = LLVMBuildICmp(ir.b.0, llvm_sys::LLVMIntPredicate::LLVMIntEQ, args[2], zero, c"".as_ptr());
            LLVMBuildCondBr(ir.b.0, c, equal, lp);
            ir.b.at_end(lp);
            let index = LLVMBuildPhi(ir.b.0, size_ty, c"".as_ptr());
            let mut v = [zero];
            let mut bb = [entry];
            LLVMAddIncoming(index, v.as_mut_ptr(), bb.as_mut_ptr(), 1);
            let mut i0 = [index];
            let pl = LLVMBuildInBoundsGEP2(ir.b.0, ir.i8(), args[0], i0.as_mut_ptr(), 1, c"".as_ptr());
            let lhs = ir.load(ir.i8(), pl);
            let mut i1 = [index];
            let pr = LLVMBuildInBoundsGEP2(ir.b.0, ir.i8(), args[1], i1.as_mut_ptr(), 1, c"".as_ptr());
            let rhs = ir.load(ir.i8(), pr);
            let c = LLVMBuildICmp(ir.b.0, llvm_sys::LLVMIntPredicate::LLVMIntEQ, lhs, rhs, c"".as_ptr());
            LLVMBuildCondBr(ir.b.0, c, next, different);
            ir.b.at_end(next);
            let ni = LLVMBuildAdd(ir.b.0, index, one, c"".as_ptr());
            let mut v = [ni];
            let mut bb = [next];
            LLVMAddIncoming(index, v.as_mut_ptr(), bb.as_mut_ptr(), 1);
            let c = LLVMBuildICmp(ir.b.0, llvm_sys::LLVMIntPredicate::LLVMIntEQ, ni, args[2], c"".as_ptr());
            LLVMBuildCondBr(ir.b.0, c, equal, lp);
            ir.b.at_end(different);
            let rt = ffi::return_type(ir.f);
            let l = LLVMBuildZExt(ir.b.0, lhs, rt, c"".as_ptr());
            let r = LLVMBuildZExt(ir.b.0, rhs, rt, c"".as_ptr());
            LLVMBuildRet(ir.b.0, LLVMBuildSub(ir.b.0, l, r, c"".as_ptr()));
            ir.b.at_end(equal);
            LLVMBuildRet(ir.b.0, LLVMConstInt(rt, 0, 0));
        },
        FillIn::Alloc => unsafe {
            let mem = ir.call(ir.malloc(), &mut [args[0]]);
            LLVMBuildRet(ir.b.0, mem);
        },
        FillIn::Realloc => unsafe {
            let mem = ir.call(ir.malloc(), &mut [args[1]]);
            let f = ir.intrinsic("llvm.memcpy", &mut [ir.ptr(), ir.ptr(), ir.i64()]);
            let fls = LLVMConstInt(LLVMInt1TypeInContext(ir.ctx), 0, 0);
            ir.call(f, &mut [mem, args[0], args[2], fls]);
            LLVMBuildRet(ir.b.0, mem);
        },
        FillIn::Calloc => unsafe {
            let size = LLVMBuildMul(ir.b.0, args[0], args[1], c"".as_ptr());
            let mem = ir.call(ir.malloc(), &mut [size]);
            let f = ir.intrinsic("llvm.memset", &mut [ir.ptr(), ir.i64()]);
            let fls = LLVMConstInt(LLVMInt1TypeInContext(ir.ctx), 0, 0);
            let zero8 = LLVMConstInt(ir.i8(), 0, 0);
            ir.call(f, &mut [mem, zero8, size, fls]);
            LLVMBuildRet(ir.b.0, mem);
        },
        FillIn::Unreachable => unsafe {
            LLVMBuildUnreachable(ir.b.0);
        },
        FillIn::Vec(kind, func, f32) => {
            let ft = unsafe { if f32 { LLVMFloatTypeInContext(ir.ctx) } else { LLVMDoubleTypeInContext(ir.ctx) } };
            let f = match kind {
                VecLoop::Unary => ir.get_or_insert(func, ft, &mut [ft]),
                VecLoop::Binary | VecLoop::Complex => ir.get_or_insert(func, ft, &mut [ft, ft]),
            };
            codegen_vec_loop(ir, args, f, kind);
        }
    }
}

fn make_fill_in(f: V, kind: FillIn) -> V {
    let m = ffi::global_module(f);
    let name = format!(".codon.gpu.fillin.{}", ffi::name(f));
    if let Some(g) = ffi::get_function(m, &name) {
        return g;
    }
    let fill = copy_prototype(f, &name, false);
    let ctx = ffi::module_context(m);
    let args: Vec<V> = (0..ffi::count_params(fill)).map(|i| ffi::param(fill, i)).collect();
    let ir = Ir { b: ffi::Builder::new(ctx), ctx, m, f: fill };
    let entry = ir.block(c"entry");
    ir.b.at_end(entry);
    codegen_fill_in(&ir, &args, kind);
    fill
}

const REMAPPING: &[(&str, &str)] = &[
    ("llvm.ceil.f64", "__nv_ceil"),
    ("llvm.floor.f64", "__nv_floor"),
    ("llvm.fabs.f64", "__nv_fabs"),
    ("llvm.exp.f64", "__nv_exp"),
    ("llvm.log.f64", "__nv_log"),
    ("llvm.log2.f64", "__nv_log2"),
    ("llvm.log10.f64", "__nv_log10"),
    ("llvm.sqrt.f64", "__nv_sqrt"),
    ("llvm.pow.f64", "__nv_pow"),
    ("llvm.sin.f64", "__nv_sin"),
    ("llvm.cos.f64", "__nv_cos"),
    ("llvm.copysign.f64", "__nv_copysign"),
    ("llvm.trunc.f64", "__nv_trunc"),
    ("llvm.rint.f64", "__nv_rint"),
    ("llvm.nearbyint.f64", "__nv_nearbyint"),
    ("llvm.round.f64", "__nv_round"),
    ("llvm.minnum.f64", "__nv_fmin"),
    ("llvm.maxnum.f64", "__nv_fmax"),
    ("llvm.copysign.f64", "__nv_copysign"),
    ("llvm.fma.f64", "__nv_fma"),
    ("expm1", "__nv_expm1"),
    ("ldexp", "__nv_ldexp"),
    ("acos", "__nv_acos"),
    ("asin", "__nv_asin"),
    ("atan", "__nv_atan"),
    ("atan2", "__nv_atan2"),
    ("hypot", "__nv_hypot"),
    ("tan", "__nv_tan"),
    ("cosh", "__nv_cosh"),
    ("sinh", "__nv_sinh"),
    ("tanh", "__nv_tanh"),
    ("acosh", "__nv_acosh"),
    ("asinh", "__nv_asinh"),
    ("atanh", "__nv_atanh"),
    ("erf", "__nv_erf"),
    ("erfc", "__nv_erfc"),
    ("log1p", "__nv_log1p"),
    ("tgamma", "__nv_tgamma"),
    ("lgamma", "__nv_lgamma"),
    ("remainder", "__nv_remainder"),
    ("frexp", "__nv_frexp"),
    ("modf", "__nv_modf"),
    ("llvm.ceil.f32", "__nv_ceilf"),
    ("llvm.floor.f32", "__nv_floorf"),
    ("llvm.fabs.f32", "__nv_fabsf"),
    ("llvm.exp.f32", "__nv_expf"),
    ("llvm.log.f32", "__nv_logf"),
    ("llvm.log2.f32", "__nv_log2f"),
    ("llvm.log10.f32", "__nv_log10f"),
    ("llvm.sqrt.f32", "__nv_sqrtf"),
    ("llvm.pow.f32", "__nv_powf"),
    ("llvm.sin.f32", "__nv_sinf"),
    ("llvm.cos.f32", "__nv_cosf"),
    ("llvm.copysign.f32", "__nv_copysignf"),
    ("llvm.trunc.f32", "__nv_truncf"),
    ("llvm.rint.f32", "__nv_rintf"),
    ("llvm.nearbyint.f32", "__nv_nearbyintf"),
    ("llvm.round.f32", "__nv_roundf"),
    ("llvm.minnum.f32", "__nv_fminf"),
    ("llvm.maxnum.f32", "__nv_fmaxf"),
    ("llvm.copysign.f32", "__nv_copysignf"),
    ("llvm.fma.f32", "__nv_fmaf"),
    ("expm1f", "__nv_expm1f"),
    ("ldexpf", "__nv_ldexpf"),
    ("acosf", "__nv_acosf"),
    ("asinf", "__nv_asinf"),
    ("atanf", "__nv_atanf"),
    ("atan2f", "__nv_atan2f"),
    ("hypotf", "__nv_hypotf"),
    ("tanf", "__nv_tanf"),
    ("coshf", "__nv_coshf"),
    ("sinhf", "__nv_sinhf"),
    ("tanhf", "__nv_tanhf"),
    ("acoshf", "__nv_acoshf"),
    ("asinhf", "__nv_asinhf"),
    ("atanhf", "__nv_atanhf"),
    ("erff", "__nv_erff"),
    ("erfcf", "__nv_erfcf"),
    ("log1pf", "__nv_log1pf"),
    ("tgammaf", "__nv_tgammaf"),
    ("lgammaf", "__nv_lgammaf"),
    ("remainderf", "__nv_remainderf"),
    ("frexpf", "__nv_frexpf"),
    ("modff", "__nv_modff"),
    ("seq_free", "free"),
    ("seq_register_finalizer", ""),
    ("seq_gc_add_roots", ""),
    ("seq_gc_remove_roots", ""),
    ("seq_gc_clear_roots", ""),
    ("seq_gc_exclude_static_roots", ""),
];

/// Port of `remapFunctions`.
pub(crate) fn remap_functions(m: LLVMModuleRef) {
    for &(from, to) in REMAPPING {
        let Some(f) = ffi::get_function(m, from) else { continue };
        let g = if to.is_empty() {
            make_no_op(f)
        } else {
            ffi::get_function(m, to).unwrap_or_else(|| copy_prototype(f, to, true))
        };
        ffi::add_fn_enum_attr(g, "willreturn", 0);
        ffi::replace_all_uses(f, g);
        unsafe { LLVMDeleteFunction(f) };
    }
    for (name, kind) in fill_in_table() {
        let Some(f) = ffi::get_function(m, name) else { continue };
        let g = make_fill_in(f, kind);
        ffi::replace_all_uses(f, g);
        unsafe { LLVMDeleteFunction(f) };
    }
}

/// Port of `normalizeKernelReturnToVoid`.
fn normalize_kernel_return_to_void(f: V) -> V {
    if ffi::is_declaration(f) || !ffi::fn_has_string_attr(f, "kernel") {
        return f;
    }
    let rt = ffi::return_type(f);
    if ffi::is_void_ty(rt) || !ffi::is_empty_literal_struct(rt) {
        return f;
    }
    let m = ffi::global_module(f);
    let ctx = ffi::module_context(m);
    unsafe {
        let old_ty = ffi::function_type(f);
        let n = LLVMCountParamTypes(old_ty);
        let mut params = vec![std::ptr::null_mut(); n as usize];
        if n > 0 {
            LLVMGetParamTypes(old_ty, params.as_mut_ptr());
        }
        let new_ty = LLVMFunctionType(LLVMVoidTypeInContext(ctx), params.as_mut_ptr(), n, LLVMIsFunctionVarArg(old_ty));
        let name = ffi::name(f);
        ffi::set_name(f, "");
        let cname = ffi::cstr(&name);
        let g = LLVMAddFunction(m, cname.as_ptr(), new_ty);
        ffi::set_linkage(g, ffi::linkage(f));
        // copyAttributesFrom: function + parameter attributes
        for a in ffi::fn_attributes(f, ffi::FN_INDEX) {
            LLVMAddAttributeAtIndex(g, ffi::FN_INDEX, a);
        }
        for i in 0..n {
            for a in ffi::fn_attributes(f, i + 1) {
                LLVMAddAttributeAtIndex(g, i + 1, a);
            }
        }
        LLVMSetFunctionCallConv(g, LLVMGetFunctionCallConv(f));
        LLVMSetVisibility(g, LLVMGetVisibility(f));
        LLVMSetDLLStorageClass(g, LLVMGetDLLStorageClass(f));
        LLVMSetUnnamedAddress(g, LLVMGetUnnamedAddress(f));
        for b in ffi::blocks(f) {
            LLVMRemoveBasicBlockFromParent(b);
            LLVMAppendExistingBasicBlock(g, b);
        }
        for i in 0..n {
            let old = ffi::param(f, i);
            let new = ffi::param(g, i);
            ffi::set_name(new, &ffi::name(old));
            ffi::replace_all_uses(old, new);
        }
        let b = ffi::Builder::new(ctx);
        for bb in ffi::blocks(g) {
            if let Some(t) = ffi::terminator(bb) {
                if ffi::opcode(t) == Op::LLVMRet {
                    b.before(t);
                    LLVMBuildRetVoid(b.0);
                    ffi::erase(t);
                }
            }
        }
        typhon_common::seqassert!(ffi::use_empty(f), "kernel still has uses after ABI normalization");
        LLVMDeleteFunction(f);
        g
    }
}

/// Port of `moduleToPTX`.
fn module_to_ptx(m: LLVMModuleRef, kernels: &[V], options: &Options) -> Result<String> {
    let machine = crate::target::target_machine_for(GPU_TRIPLE, &options.gpu_name, &options.gpu_feat, "nvptx64", false)?
        .ok_or_else(|| err("couldn't lookup target: nvptx64"))?;
    for f in ffi::functions(m) {
        ffi::add_fn_enum_attr(f, "nounwind", 0);
        unsafe {
            if LLVMHasPersonalityFn(f) != 0 {
                LLVMSetPersonalityFn(f, std::ptr::null_mut());
            }
        }
    }
    let dl = crate::target::data_layout_string(&machine);
    let cdl = ffi::cstr(&dl);
    unsafe { LLVMSetDataLayout(m, cdl.as_ptr()) };
    let keep = required_gvs(kernels);
    prune(m, &keep)?;

    link_libdevice(m, &options.libdevice)?;
    remap_functions(m);

    for f in ffi::functions(m) {
        if ffi::intrinsic_id(f) != 0 {
            continue;
        }
        ffi::remove_fn_enum_attr(f, "noinline");
        ffi::add_fn_enum_attr(f, "willreturn", 0);
    }
    unsafe { llvm_sys::debuginfo::LLVMStripModuleDebugInfo(m) };

    // registerPipelineStartEPCallback(InternalizePass(keep)) + default<O3>.
    // PORT-NOTE: InternalizePass with a predicate is not reachable through the
    // pipeline text; it is applied by hand right before the pipeline, which is
    // where the PipelineStart extension point runs.
    internalize(m, &keep);
    crate::optimize::run_pipeline(m, "default<O3>", machine.as_mut_ptr())?;

    let keep = required_gvs(kernels);
    prune(m, &keep)?;

    for g in ffi::globals(m) {
        ffi::set_name(g, &clean_up_name(&ffi::name(g)));
    }
    for f in ffi::functions(m) {
        if ffi::instruction_count(f) > 0 {
            ffi::set_name(f, &clean_up_name(&ffi::name(f)));
        }
    }
    // PORT-NOTE: identified struct types are not renamed (no LLVMStructSetName in the
    // C API); PTX output does not contain LLVM struct type names.

    let buf = machine
        .write_to_memory_buffer(&unsafe { Module::new(m) }.leak_for_codegen(), inkwell::targets::FileType::Assembly)
        .map_err(|e| err(e.to_string()))?;
    Ok(String::from_utf8_lossy(buf.as_slice()).into_owned())
}

/// Helper to borrow a raw module as an inkwell `Module` without taking ownership.
trait LeakForCodegen<'a> {
    fn leak_for_codegen(self) -> std::mem::ManuallyDrop<Module<'a>>;
}

impl<'a> LeakForCodegen<'a> for Module<'a> {
    fn leak_for_codegen(self) -> std::mem::ManuallyDrop<Module<'a>> {
        std::mem::ManuallyDrop::new(self)
    }
}

/// `InternalizePass(isMustPreserve)`: give every definition not in `keep`
/// internal linkage (LLVM-reserved `llvm.*` globals are always preserved).
fn internalize(m: LLVMModuleRef, keep: &FxHashSet<V>) {
    let candidates = ffi::functions(m).into_iter().chain(ffi::globals(m));
    for g in candidates {
        if ffi::is_declaration(g) || keep.contains(&g) || ffi::has_local_linkage(g) || ffi::name(g).starts_with("llvm.") {
            continue;
        }
        ffi::set_linkage(g, L::LLVMInternalLinkage);
        unsafe { llvm_sys::comdat::LLVMSetComdat(g, std::ptr::null_mut()) };
    }
}

/// Port of `cleanUpIntrinsics`.
fn clean_up_intrinsics(m: LLVMModuleRef) {
    let remove: Vec<V> =
        ffi::functions(m).into_iter().filter(|&f| ffi::intrinsic_id(f) != 0 && ffi::name(f).starts_with("llvm.nvvm")).collect();
    for f in remove {
        let noop = make_no_op(f);
        ffi::replace_all_uses(f, noop);
        unsafe { LLVMDeleteFunction(f) };
    }
}

/// Port of `patchPTXVar`.
fn patch_ptx_var(m: LLVMModuleRef, ptx_var: Option<V>, target: &str) -> Result<()> {
    let mut calls = Vec::new();
    for f in ffi::functions(m) {
        for i in ffi::fn_instructions(f) {
            if !ffi::is_call(i) {
                continue;
            }
            if let Some(callee) = ffi::called_function(i) {
                if ffi::name(callee) == target && ffi::num_args(i) == 0 {
                    calls.push(i);
                }
            }
        }
    }
    let ctx = ffi::module_context(m);
    for call in calls {
        let rep = ptx_var.unwrap_or_else(|| unsafe { LLVMConstPointerNull(ffi::ptr_ty(ctx)) });
        ffi::replace_all_uses(call, rep);
        ffi::erase(call);
    }
    if let Some(f) = ffi::get_function(m, target) {
        if !ffi::use_empty(f) {
            return Err(err("some __typhon_ptx__() calls not replaced in module"));
        }
        unsafe { LLVMDeleteFunction(f) };
    }
    Ok(())
}

fn has_kernels(m: LLVMModuleRef) -> bool {
    ffi::functions(m).into_iter().any(|f| ffi::fn_has_string_attr(f, "kernel"))
}

/// Port of `prepareGPUmodule`.
pub fn prepare_gpu_module<'ctx>(module: &Module<'ctx>, options: &Options) -> Result<Option<Module<'ctx>>> {
    let m = module.as_mut_ptr();
    if !has_kernels(m) {
        patch_ptx_var(m, None, "__typhon_ptx__")?;
        return Ok(None);
    }
    unsafe {
        let clone = LLVMCloneModule(m);
        let t = ffi::cstr(GPU_TRIPLE);
        LLVMSetTarget(clone, t.as_ptr());
        let dl = ffi::cstr(GPU_DL);
        LLVMSetDataLayout(clone, dl.as_ptr());
        if options.fastmath {
            let ctx = ffi::module_context(clone);
            let one = LLVMConstInt(LLVMInt32TypeInContext(ctx), 1, 0);
            let key = "nvvm-reflect-ftz";
            LLVMAddModuleFlag(
                clone,
                llvm_sys::LLVMModuleFlagBehavior::LLVMModuleFlagBehaviorOverride,
                key.as_ptr() as *const _,
                key.len(),
                LLVMValueAsMetadata(one),
            );
        }
        Ok(Some(Module::new(clone)))
    }
}

/// Port of `applyGPUTransformations`.
pub fn apply_gpu_transformations(module: &Module, clone: Module, options: &Options, ptx_filename: &str) -> Result<()> {
    let m = module.as_mut_ptr();
    let cm = clone.as_mut_ptr();
    let ctx = ffi::module_context(m);
    let candidates: Vec<V> = ffi::functions(cm).into_iter().filter(|&f| ffi::fn_has_string_attr(f, "kernel")).collect();
    let mut kernels = Vec::new();
    for f in candidates {
        let kernel = normalize_kernel_return_to_void(f);
        unsafe {
            let s = "kernel";
            let mut md = [
                LLVMValueAsMetadata(kernel),
                LLVMMDStringInContext2(ctx, s.as_ptr() as *const _, s.len()),
                LLVMValueAsMetadata(LLVMConstInt(LLVMInt32TypeInContext(ctx), 1, 0)),
            ];
            let node = LLVMMDNodeInContext2(ctx, md.as_mut_ptr(), 3);
            LLVMAddNamedMetadataOperand(cm, c"nvvm.annotations".as_ptr(), LLVMMetadataAsValue(ctx, node));
        }
        kernels.push(kernel);
    }
    if kernels.is_empty() {
        return patch_ptx_var(m, None, "__typhon_ptx__");
    }
    let ptx = module_to_ptx(cm, &kernels, options)?;
    clean_up_intrinsics(m);

    let out = if !options.gpu_output.is_empty() { options.gpu_output.as_str() } else { ptx_filename };
    if !out.is_empty() {
        std::fs::write(out, &ptx).map_err(|e| err(format!("Could not open file: {e}")))?;
    }
    unsafe {
        let data = LLVMConstStringInContext2(ctx, ptx.as_ptr() as *const _, ptx.len(), 0);
        let g = LLVMAddGlobal(m, ffi::type_of(data), c".ptx".as_ptr());
        LLVMSetInitializer(g, data);
        LLVMSetGlobalConstant(g, 1);
        ffi::set_linkage(g, L::LLVMPrivateLinkage);
        LLVMSetUnnamedAddress(g, llvm_sys::LLVMUnnamedAddr::LLVMGlobalUnnamedAddr);
        patch_ptx_var(m, Some(g), "__typhon_ptx__")?;
    }
    drop(clone);
    Ok(())
}
