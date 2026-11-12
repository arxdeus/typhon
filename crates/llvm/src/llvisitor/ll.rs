//! Raw LLVM C-API helpers used by the visitor (the C++ code uses `llvm::IRBuilder<>`,
//! `llvm::DIBuilder` and friends directly). Every function here wraps an `llvm-sys`
//! call; all handles passed in come from LLVM itself.

#![allow(dead_code)]

use std::ffi::{CStr, CString};
use std::ptr::null_mut;

use llvm_sys::LLVMIntPredicate;
use llvm_sys::core::*;
use llvm_sys::debuginfo::*;
use llvm_sys::prelude::*;
use llvm_sys::target::*;

pub type V = LLVMValueRef;
pub type Ty = LLVMTypeRef;
pub type BB = LLVMBasicBlockRef;
pub type Md = LLVMMetadataRef;

pub const FN_INDEX: u32 = llvm_sys::LLVMAttributeFunctionIndex;
pub const RET_INDEX: u32 = llvm_sys::LLVMAttributeReturnIndex;

pub fn cstr(s: &str) -> CString {
    CString::new(s.replace('\0', "")).expect("no interior NUL")
}

const EMPTY: &[u8] = b"\0";
fn noname() -> *const libc::c_char {
    EMPTY.as_ptr() as *const _
}

// ------------------------------------------------------------------ types

pub fn void_ty(c: LLVMContextRef) -> Ty {
    unsafe { LLVMVoidTypeInContext(c) }
}
pub fn int_ty(c: LLVMContextRef, bits: u32) -> Ty {
    unsafe { LLVMIntTypeInContext(c, bits) }
}
pub fn ptr_ty(c: LLVMContextRef) -> Ty {
    unsafe { LLVMPointerTypeInContext(c, 0) }
}
pub fn double_ty(c: LLVMContextRef) -> Ty {
    unsafe { LLVMDoubleTypeInContext(c) }
}
pub fn float_ty(c: LLVMContextRef) -> Ty {
    unsafe { LLVMFloatTypeInContext(c) }
}
pub fn half_ty(c: LLVMContextRef) -> Ty {
    unsafe { LLVMHalfTypeInContext(c) }
}
pub fn bfloat_ty(c: LLVMContextRef) -> Ty {
    unsafe { LLVMBFloatTypeInContext(c) }
}
pub fn fp128_ty(c: LLVMContextRef) -> Ty {
    unsafe { LLVMFP128TypeInContext(c) }
}
pub fn token_ty(c: LLVMContextRef) -> Ty {
    unsafe { LLVMTokenTypeInContext(c) }
}
pub fn struct_ty(c: LLVMContextRef, fields: &[Ty]) -> Ty {
    let mut f = fields.to_vec();
    unsafe { LLVMStructTypeInContext(c, f.as_mut_ptr(), f.len() as u32, 0) }
}
pub fn named_struct_ty(c: LLVMContextRef, name: &str, fields: &[Ty]) -> Ty {
    let n = cstr(name);
    let mut f = fields.to_vec();
    unsafe {
        let t = LLVMStructCreateNamed(c, n.as_ptr());
        LLVMStructSetBody(t, f.as_mut_ptr(), f.len() as u32, 0);
        t
    }
}
pub fn array_ty(elem: Ty, n: u64) -> Ty {
    unsafe { LLVMArrayType2(elem, n) }
}
pub fn vector_ty(elem: Ty, n: u32) -> Ty {
    unsafe { LLVMVectorType(elem, n) }
}
pub fn fn_ty(ret: Ty, args: &[Ty], variadic: bool) -> Ty {
    let mut a = args.to_vec();
    unsafe { LLVMFunctionType(ret, a.as_mut_ptr(), a.len() as u32, variadic as i32) }
}
pub fn type_of(v: V) -> Ty {
    unsafe { LLVMTypeOf(v) }
}
pub fn global_value_type(g: V) -> Ty {
    unsafe { LLVMGlobalGetValueType(g) }
}
pub fn is_void(t: Ty) -> bool {
    unsafe { LLVMGetTypeKind(t) == llvm_sys::LLVMTypeKind::LLVMVoidTypeKind }
}
pub fn is_struct(t: Ty) -> bool {
    unsafe { LLVMGetTypeKind(t) == llvm_sys::LLVMTypeKind::LLVMStructTypeKind }
}
pub fn is_pointer(t: Ty) -> bool {
    unsafe { LLVMGetTypeKind(t) == llvm_sys::LLVMTypeKind::LLVMPointerTypeKind }
}
pub fn is_integer(t: Ty) -> bool {
    unsafe { LLVMGetTypeKind(t) == llvm_sys::LLVMTypeKind::LLVMIntegerTypeKind }
}
pub fn int_width(t: Ty) -> u32 {
    unsafe { LLVMGetIntTypeWidth(t) }
}
pub fn is_aggregate(t: Ty) -> bool {
    use llvm_sys::LLVMTypeKind::*;
    matches!(unsafe { LLVMGetTypeKind(t) }, LLVMStructTypeKind | LLVMArrayTypeKind)
}
pub fn struct_elements(t: Ty) -> Vec<Ty> {
    unsafe {
        let n = LLVMCountStructElementTypes(t);
        (0..n).map(|i| LLVMStructGetTypeAtIndex(t, i)).collect()
    }
}
pub fn fn_ret_ty(fnty: Ty) -> Ty {
    unsafe { LLVMGetReturnType(fnty) }
}
pub fn fn_param_tys(fnty: Ty) -> Vec<Ty> {
    unsafe {
        let n = LLVMCountParamTypes(fnty);
        let mut v = vec![null_mut(); n as usize];
        LLVMGetParamTypes(fnty, v.as_mut_ptr());
        v
    }
}
pub fn fn_is_vararg(fnty: Ty) -> bool {
    unsafe { LLVMIsFunctionVarArg(fnty) != 0 }
}
pub fn type_to_string(t: Ty) -> String {
    unsafe {
        let p = LLVMPrintTypeToString(t);
        let s = CStr::from_ptr(p).to_string_lossy().into_owned();
        LLVMDisposeMessage(p);
        s
    }
}
/// Structural type identity (LLVM uniques literal types per context).
pub fn same_type(a: Ty, b: Ty) -> bool {
    a == b
}

// ------------------------------------------------------------------ constants

pub fn const_int(t: Ty, v: u64, signed: bool) -> V {
    unsafe { LLVMConstInt(t, v, signed as i32) }
}
pub fn const_real(t: Ty, v: f64) -> V {
    unsafe { LLVMConstReal(t, v) }
}
pub fn const_null(t: Ty) -> V {
    unsafe { LLVMConstNull(t) }
}
pub fn undef(t: Ty) -> V {
    unsafe { LLVMGetUndef(t) }
}
pub fn const_struct(c: LLVMContextRef, vals: &[V]) -> V {
    let mut v = vals.to_vec();
    unsafe { LLVMConstStructInContext(c, v.as_mut_ptr(), v.len() as u32, 0) }
}
pub fn const_named_struct(t: Ty, vals: &[V]) -> V {
    let mut v = vals.to_vec();
    unsafe { LLVMConstNamedStruct(t, v.as_mut_ptr(), v.len() as u32) }
}
pub fn const_array(elem: Ty, vals: &[V]) -> V {
    let mut v = vals.to_vec();
    unsafe { LLVMConstArray2(elem, v.as_mut_ptr(), v.len() as u64) }
}
pub fn const_zext(v: V) -> u64 {
    unsafe { LLVMConstIntGetZExtValue(v) }
}
pub fn const_sext(v: V) -> i64 {
    unsafe { LLVMConstIntGetSExtValue(v) }
}
pub fn is_const_int(v: V) -> bool {
    unsafe { !LLVMIsAConstantInt(v).is_null() }
}
pub fn aggregate_element(c: V, i: u32) -> V {
    unsafe { LLVMGetAggregateElement(c, i) }
}
/// `ConstantTokenNone::get(ctx)` (`Constant::getNullValue(token)`).
pub fn token_none(c: LLVMContextRef) -> V {
    const_null(token_ty(c))
}

// ------------------------------------------------------------------ module-level

pub fn get_function(m: LLVMModuleRef, name: &str) -> Option<V> {
    let n = cstr(name);
    let f = unsafe { LLVMGetNamedFunction(m, n.as_ptr()) };
    (!f.is_null()).then_some(f)
}
pub fn get_global(m: LLVMModuleRef, name: &str) -> Option<V> {
    let n = cstr(name);
    let g = unsafe { LLVMGetNamedGlobal(m, n.as_ptr()) };
    (!g.is_null()).then_some(g)
}
/// `Module::getNamedValue`: function, global variable or alias.
pub fn get_named_value(m: LLVMModuleRef, name: &str) -> Option<V> {
    get_function(m, name).or_else(|| get_global(m, name)).or_else(|| {
        let a = unsafe { LLVMGetNamedGlobalAlias(m, name.as_ptr() as *const _, name.len()) };
        (!a.is_null()).then_some(a)
    })
}
/// `Module::getOrInsertFunction(name, type)`: the existing function if any.
pub fn get_or_insert_function(m: LLVMModuleRef, name: &str, fnty: Ty) -> V {
    if let Some(f) = get_function(m, name) {
        return f;
    }
    let n = cstr(name);
    unsafe { LLVMAddFunction(m, n.as_ptr(), fnty) }
}
pub fn add_function(m: LLVMModuleRef, name: &str, fnty: Ty) -> V {
    let n = cstr(name);
    unsafe { LLVMAddFunction(m, n.as_ptr(), fnty) }
}
pub fn add_global(m: LLVMModuleRef, ty: Ty, name: &str) -> V {
    let n = cstr(name);
    unsafe { LLVMAddGlobal(m, ty, n.as_ptr()) }
}
/// `new GlobalVariable(M, type, isConstant, linkage, init, name)`.
pub fn new_global(m: LLVMModuleRef, ty: Ty, constant: bool, linkage: llvm_sys::LLVMLinkage, init: Option<V>, name: &str) -> V {
    let g = add_global(m, ty, name);
    unsafe {
        LLVMSetGlobalConstant(g, constant as i32);
        LLVMSetLinkage(g, linkage);
        if let Some(i) = init {
            LLVMSetInitializer(g, i);
        }
    }
    g
}
pub fn set_initializer(g: V, init: V) {
    unsafe { LLVMSetInitializer(g, init) }
}
pub fn set_linkage(g: V, l: llvm_sys::LLVMLinkage) {
    unsafe { LLVMSetLinkage(g, l) }
}
pub fn linkage(g: V) -> llvm_sys::LLVMLinkage {
    unsafe { LLVMGetLinkage(g) }
}
pub fn set_unnamed_addr(g: V, u: llvm_sys::LLVMUnnamedAddr) {
    unsafe { LLVMSetUnnamedAddress(g, u) }
}
pub fn set_thread_local(g: V, t: bool) {
    unsafe { LLVMSetThreadLocal(g, t as i32) }
}
pub fn set_externally_initialized(g: V, t: bool) {
    unsafe { LLVMSetExternallyInitialized(g, t as i32) }
}
pub fn set_dll_storage(g: V, c: llvm_sys::LLVMDLLStorageClass) {
    unsafe { LLVMSetDLLStorageClass(g, c) }
}
pub fn set_dso_local(g: V) {
    // The C API has no `setDSOLocal`; globals keep default visibility.
    let _ = g;
}
pub fn name(v: V) -> String {
    unsafe {
        let mut len = 0usize;
        let p = LLVMGetValueName2(v, &mut len);
        if p.is_null() {
            return String::new();
        }
        String::from_utf8_lossy(std::slice::from_raw_parts(p as *const u8, len)).into_owned()
    }
}
pub fn set_name(v: V, n: &str) {
    unsafe { LLVMSetValueName2(v, n.as_ptr() as *const _, n.len()) }
}
pub fn is_declaration(g: V) -> bool {
    unsafe { LLVMIsDeclaration(g) != 0 }
}
pub fn functions(m: LLVMModuleRef) -> Vec<V> {
    let mut out = Vec::new();
    unsafe {
        let mut f = LLVMGetFirstFunction(m);
        while !f.is_null() {
            out.push(f);
            f = LLVMGetNextFunction(f);
        }
    }
    out
}
pub fn param(f: V, i: u32) -> V {
    unsafe { LLVMGetParam(f, i) }
}
pub fn params(f: V) -> Vec<V> {
    unsafe { (0..LLVMCountParams(f)).map(|i| LLVMGetParam(f, i)).collect() }
}
pub fn function_type(f: V) -> Ty {
    global_value_type(f)
}
pub fn set_personality(f: V, p: V) {
    unsafe { LLVMSetPersonalityFn(f, p) }
}
pub fn entry_block(f: V) -> BB {
    unsafe { LLVMGetEntryBasicBlock(f) }
}
pub fn has_blocks(f: V) -> bool {
    unsafe { LLVMCountBasicBlocks(f) > 0 }
}
pub fn blocks(f: V) -> Vec<BB> {
    let mut out = Vec::new();
    unsafe {
        let mut b = LLVMGetFirstBasicBlock(f);
        while !b.is_null() {
            out.push(b);
            b = LLVMGetNextBasicBlock(b);
        }
    }
    out
}
pub fn instructions(b: BB) -> Vec<V> {
    let mut out = Vec::new();
    unsafe {
        let mut i = LLVMGetFirstInstruction(b);
        while !i.is_null() {
            out.push(i);
            i = LLVMGetNextInstruction(i);
        }
    }
    out
}
pub fn terminator(b: BB) -> Option<V> {
    let t = unsafe { LLVMGetBasicBlockTerminator(b) };
    (!t.is_null()).then_some(t)
}
pub fn append_block(c: LLVMContextRef, f: V, name: &str) -> BB {
    let n = cstr(name);
    unsafe { LLVMAppendBasicBlockInContext(c, f, n.as_ptr()) }
}
pub fn data_layout(m: LLVMModuleRef) -> LLVMTargetDataRef {
    unsafe { LLVMGetModuleDataLayout(m) }
}
/// `DataLayout::getTypeAllocSize`.
pub fn alloc_size(m: LLVMModuleRef, t: Ty) -> u64 {
    unsafe { LLVMABISizeOfType(data_layout(m), t) }
}
pub fn abi_align(m: LLVMModuleRef, t: Ty) -> u64 {
    unsafe { LLVMABIAlignmentOfType(data_layout(m), t) as u64 }
}
pub fn pref_align(m: LLVMModuleRef, t: Ty) -> u64 {
    unsafe { LLVMPreferredAlignmentOfType(data_layout(m), t) as u64 }
}
/// `StructLayout::getElementOffset`.
pub fn element_offset(m: LLVMModuleRef, t: Ty, i: u32) -> u64 {
    unsafe { LLVMOffsetOfElement(data_layout(m), t, i) }
}
pub fn pointer_size(m: LLVMModuleRef) -> u64 {
    unsafe { LLVMPointerSize(data_layout(m)) as u64 }
}
pub fn triple(m: LLVMModuleRef) -> String {
    unsafe { CStr::from_ptr(LLVMGetTarget(m)).to_string_lossy().into_owned() }
}
pub fn is_win_msvc_triple(t: &str) -> bool {
    t.contains("windows") && !t.contains("gnu") && !t.contains("cygnus") && !t.contains("itanium")
}
pub fn is_windows_triple(t: &str) -> bool {
    t.contains("windows") || t.contains("win32")
}
pub fn is_darwin_triple(t: &str) -> bool {
    t.contains("apple") || t.contains("darwin") || t.contains("macos")
}

// ------------------------------------------------------------------ intrinsics

/// `Intrinsic::getDeclaration(M, id, tys)` returning `(function type, function)`.
pub fn intrinsic(m: LLVMModuleRef, name: &str, overloads: &[Ty]) -> (Ty, V) {
    let mut tys = overloads.to_vec();
    unsafe {
        let id = LLVMLookupIntrinsicID(name.as_ptr() as *const _, name.len());
        assert!(id != 0, "unknown intrinsic {name}");
        let f = LLVMGetIntrinsicDeclaration(m, id, tys.as_mut_ptr(), tys.len());
        let ty = LLVMIntrinsicGetType(LLVMGetModuleContext(m), id, tys.as_mut_ptr(), tys.len());
        (ty, f)
    }
}

// ------------------------------------------------------------------ attributes

pub fn attr_kind(n: &str) -> u32 {
    unsafe { LLVMGetEnumAttributeKindForName(n.as_ptr() as *const _, n.len()) }
}
pub fn add_enum_attr(f: V, idx: u32, kind: &str, val: u64) {
    let k = attr_kind(kind);
    if k == 0 {
        return;
    }
    unsafe {
        let c = LLVMGetTypeContext(type_of(f));
        LLVMAddAttributeAtIndex(f, idx, LLVMCreateEnumAttribute(c, k, val));
    }
}
pub fn add_fn_attr(f: V, kind: &str) {
    add_enum_attr(f, FN_INDEX, kind, 0)
}
pub fn add_type_attr(f: V, idx: u32, kind: &str, t: Ty) {
    unsafe {
        let c = LLVMGetTypeContext(type_of(f));
        LLVMAddAttributeAtIndex(f, idx, LLVMCreateTypeAttribute(c, attr_kind(kind), t));
    }
}
pub fn add_call_type_attr(call: V, idx: u32, kind: &str, t: Ty) {
    unsafe {
        let c = LLVMGetTypeContext(type_of(call));
        LLVMAddCallSiteAttribute(call, idx, LLVMCreateTypeAttribute(c, attr_kind(kind), t));
    }
}
pub fn add_string_attr(f: V, idx: u32, key: &str, val: &str) {
    unsafe {
        let c = LLVMGetTypeContext(type_of(f));
        let a = LLVMCreateStringAttribute(c, key.as_ptr() as *const _, key.len() as u32, val.as_ptr() as *const _, val.len() as u32);
        LLVMAddAttributeAtIndex(f, idx, a);
    }
}
pub fn has_string_attr(f: V, key: &str) -> bool {
    unsafe { !LLVMGetStringAttributeAtIndex(f, FN_INDEX, key.as_ptr() as *const _, key.len() as u32).is_null() }
}
pub fn string_attr_value(f: V, key: &str) -> Option<String> {
    unsafe {
        let a = LLVMGetStringAttributeAtIndex(f, FN_INDEX, key.as_ptr() as *const _, key.len() as u32);
        if a.is_null() {
            return None;
        }
        let mut len = 0u32;
        let p = LLVMGetStringAttributeValue(a, &mut len);
        Some(String::from_utf8_lossy(std::slice::from_raw_parts(p as *const u8, len as usize)).into_owned())
    }
}
pub fn remove_string_attr(f: V, key: &str) {
    unsafe { LLVMRemoveStringAttributeAtIndex(f, FN_INDEX, key.as_ptr() as *const _, key.len() as u32) }
}
pub fn has_enum_fn_attr(f: V, kind: &str) -> bool {
    unsafe { !LLVMGetEnumAttributeAtIndex(f, FN_INDEX, attr_kind(kind)).is_null() }
}
/// `memory(none)` (`setDoesNotAccessMemory` / `MemoryEffects::none()`).
pub fn set_memory_none(f: V) {
    add_enum_attr(f, FN_INDEX, "memory", 0)
}
/// `setOnlyAccessesInaccessibleMemory`: `memory(inaccessiblemem: readwrite)`.
pub fn set_only_inaccessible_mem(f: V) {
    // MemoryEffects packs two ModRef bits per location; InaccessibleMem is location 1.
    add_enum_attr(f, FN_INDEX, "memory", 3 << 2)
}
/// `Argument::addAttr(NoCapture)` (LLVM 20: `captures(none)` where available).
pub fn add_nocapture(f: V, idx: u32) {
    if attr_kind("nocapture") != 0 {
        add_enum_attr(f, idx, "nocapture", 0)
    } else {
        add_enum_attr(f, idx, "captures", 0)
    }
}

// ------------------------------------------------------------------ instructions

pub fn erase(i: V) {
    unsafe { LLVMInstructionEraseFromParent(i) }
}
pub fn set_operand(i: V, idx: u32, v: V) {
    unsafe { LLVMSetOperand(i, idx, v) }
}
pub fn num_operands(i: V) -> u32 {
    unsafe { LLVMGetNumOperands(i) as u32 }
}
pub fn operand(i: V, idx: u32) -> V {
    unsafe { LLVMGetOperand(i, idx) }
}
pub fn is_return(i: V) -> bool {
    unsafe { !LLVMIsAReturnInst(i).is_null() }
}
pub fn is_call(i: V) -> bool {
    unsafe { !LLVMIsACallInst(i).is_null() }
}
pub fn called_value(call: V) -> V {
    unsafe { LLVMGetCalledValue(call) }
}
pub fn num_arg_operands(call: V) -> u32 {
    unsafe { LLVMGetNumArgOperands(call) }
}
pub fn use_empty(v: V) -> bool {
    unsafe { LLVMGetFirstUse(v).is_null() }
}
pub fn set_metadata_empty(i: V, kind: &str) {
    unsafe {
        let c = LLVMGetTypeContext(type_of(i));
        let k = LLVMGetMDKindIDInContext(c, kind.as_ptr() as *const _, kind.len() as u32);
        let node = LLVMMDNodeInContext2(c, null_mut(), 0);
        LLVMSetMetadata(i, k, LLVMMetadataAsValue(c, node));
    }
}
pub fn inst_debug_loc(i: V) -> Md {
    unsafe { LLVMInstructionGetDebugLoc(i) }
}
pub fn inst_set_debug_loc(i: V, loc: Md) {
    unsafe { LLVMInstructionSetDebugLoc(i, loc) }
}

// ------------------------------------------------------------------ IR builder

/// Port of the subset of `llvm::IRBuilder<>` used by the visitor.
pub struct Builder {
    pub b: LLVMBuilderRef,
    pub c: LLVMContextRef,
}

impl Drop for Builder {
    fn drop(&mut self) {
        unsafe { LLVMDisposeBuilder(self.b) }
    }
}

macro_rules! bin {
    ($name:ident, $f:ident) => {
        pub fn $name(&self, a: V, b: V) -> V {
            unsafe { $f(self.b, a, b, noname()) }
        }
    };
}

impl Builder {
    pub fn new(c: LLVMContextRef) -> Self {
        Builder { b: unsafe { LLVMCreateBuilderInContext(c) }, c }
    }
    pub fn at_end(&self, bb: BB) {
        unsafe { LLVMPositionBuilderAtEnd(self.b, bb) }
    }
    pub fn before(&self, i: V) {
        unsafe { LLVMPositionBuilderBefore(self.b, i) }
    }
    pub fn insert_block(&self) -> BB {
        unsafe { LLVMGetInsertBlock(self.b) }
    }
    pub fn set_debug_loc(&self, loc: Md) {
        unsafe { LLVMSetCurrentDebugLocation2(self.b, loc) }
    }
    pub fn debug_loc(&self) -> Md {
        unsafe { LLVMGetCurrentDebugLocation2(self.b) }
    }

    // types / constants
    pub fn void(&self) -> Ty {
        void_ty(self.c)
    }
    pub fn i1(&self) -> Ty {
        int_ty(self.c, 1)
    }
    pub fn i8(&self) -> Ty {
        int_ty(self.c, 8)
    }
    pub fn i16(&self) -> Ty {
        int_ty(self.c, 16)
    }
    pub fn i32(&self) -> Ty {
        int_ty(self.c, 32)
    }
    pub fn i64(&self) -> Ty {
        int_ty(self.c, 64)
    }
    pub fn ptr(&self) -> Ty {
        ptr_ty(self.c)
    }
    pub fn c_i1(&self, v: bool) -> V {
        const_int(self.i1(), v as u64, false)
    }
    pub fn c_i8(&self, v: i64) -> V {
        const_int(self.i8(), v as u64, true)
    }
    pub fn c_i32(&self, v: i64) -> V {
        const_int(self.i32(), v as u64, true)
    }
    pub fn c_i64(&self, v: i64) -> V {
        const_int(self.i64(), v as u64, true)
    }
    pub fn null_ptr(&self) -> V {
        const_null(self.ptr())
    }

    // memory
    pub fn alloca(&self, t: Ty) -> V {
        unsafe { LLVMBuildAlloca(self.b, t, noname()) }
    }
    pub fn array_alloca(&self, t: Ty, n: V) -> V {
        unsafe { LLVMBuildArrayAlloca(self.b, t, n, noname()) }
    }
    pub fn load(&self, t: Ty, p: V) -> V {
        unsafe { LLVMBuildLoad2(self.b, t, p, noname()) }
    }
    pub fn store(&self, v: V, p: V) -> V {
        unsafe { LLVMBuildStore(self.b, v, p) }
    }
    pub fn gep(&self, t: Ty, p: V, idx: &[V]) -> V {
        let mut i = idx.to_vec();
        unsafe { LLVMBuildGEP2(self.b, t, p, i.as_mut_ptr(), i.len() as u32, noname()) }
    }
    pub fn inbounds_gep(&self, t: Ty, p: V, idx: &[V]) -> V {
        let mut i = idx.to_vec();
        unsafe { LLVMBuildInBoundsGEP2(self.b, t, p, i.as_mut_ptr(), i.len() as u32, noname()) }
    }
    pub fn struct_gep(&self, t: Ty, p: V, idx: u32) -> V {
        unsafe { LLVMBuildStructGEP2(self.b, t, p, idx, noname()) }
    }
    /// `CreateConstGEP1_64(i8, p, off)`.
    pub fn const_gep_i8(&self, p: V, off: i64) -> V {
        self.gep(self.i8(), p, &[self.c_i64(off)])
    }
    /// `CreateThreadLocalAddress`.
    pub fn thread_local_address(&self, m: LLVMModuleRef, g: V) -> V {
        let (ty, f) = intrinsic(m, "llvm.threadlocal.address", &[type_of(g)]);
        self.call(ty, f, &[g])
    }

    // values
    pub fn extract_value(&self, agg: V, i: u32) -> V {
        unsafe { LLVMBuildExtractValue(self.b, agg, i, noname()) }
    }
    pub fn insert_value(&self, agg: V, v: V, i: u32) -> V {
        unsafe { LLVMBuildInsertValue(self.b, agg, v, i, noname()) }
    }
    pub fn trunc(&self, v: V, t: Ty) -> V {
        unsafe { LLVMBuildTrunc(self.b, v, t, noname()) }
    }
    pub fn zext(&self, v: V, t: Ty) -> V {
        unsafe { LLVMBuildZExt(self.b, v, t, noname()) }
    }
    /// `CreateZExtOrTrunc`.
    pub fn zext_or_trunc(&self, v: V, t: Ty) -> V {
        let w = int_width(type_of(v));
        let tw = int_width(t);
        if w < tw {
            self.zext(v, t)
        } else if w > tw {
            self.trunc(v, t)
        } else {
            v
        }
    }
    pub fn bitcast(&self, v: V, t: Ty) -> V {
        if type_of(v) == t {
            return v;
        }
        unsafe { LLVMBuildBitCast(self.b, v, t, noname()) }
    }
    bin!(add, LLVMBuildAdd);
    bin!(sub, LLVMBuildSub);
    bin!(mul, LLVMBuildMul);
    bin!(and, LLVMBuildAnd);
    bin!(or, LLVMBuildOr);
    bin!(xor, LLVMBuildXor);
    pub fn lshr(&self, a: V, n: u64) -> V {
        unsafe { LLVMBuildLShr(self.b, a, const_int(type_of(a), n, false), noname()) }
    }
    pub fn icmp(&self, p: LLVMIntPredicate, a: V, b: V) -> V {
        unsafe { LLVMBuildICmp(self.b, p, a, b, noname()) }
    }
    pub fn icmp_eq(&self, a: V, b: V) -> V {
        self.icmp(LLVMIntPredicate::LLVMIntEQ, a, b)
    }
    pub fn icmp_ne(&self, a: V, b: V) -> V {
        self.icmp(LLVMIntPredicate::LLVMIntNE, a, b)
    }
    pub fn is_null(&self, v: V) -> V {
        unsafe { LLVMBuildIsNull(self.b, v, noname()) }
    }
    pub fn is_not_null(&self, v: V) -> V {
        unsafe { LLVMBuildIsNotNull(self.b, v, noname()) }
    }
    pub fn phi(&self, t: Ty) -> V {
        unsafe { LLVMBuildPhi(self.b, t, noname()) }
    }
    pub fn add_incoming(phi: V, v: V, bb: BB) {
        let mut vs = [v];
        let mut bs = [bb];
        unsafe { LLVMAddIncoming(phi, vs.as_mut_ptr(), bs.as_mut_ptr(), 1) }
    }

    // control flow
    pub fn br(&self, bb: BB) -> V {
        unsafe { LLVMBuildBr(self.b, bb) }
    }
    pub fn cond_br(&self, c: V, t: BB, f: BB) -> V {
        unsafe { LLVMBuildCondBr(self.b, c, t, f) }
    }
    pub fn switch(&self, v: V, default: BB, n: u32) -> V {
        unsafe { LLVMBuildSwitch(self.b, v, default, n) }
    }
    pub fn add_case(sw: V, v: V, bb: BB) {
        unsafe { LLVMAddCase(sw, v, bb) }
    }
    pub fn ret(&self, v: V) -> V {
        unsafe { LLVMBuildRet(self.b, v) }
    }
    pub fn ret_void(&self) -> V {
        unsafe { LLVMBuildRetVoid(self.b) }
    }
    pub fn unreachable(&self) -> V {
        unsafe { LLVMBuildUnreachable(self.b) }
    }
    pub fn call(&self, fnty: Ty, f: V, args: &[V]) -> V {
        let mut a = args.to_vec();
        unsafe { LLVMBuildCall2(self.b, fnty, f, a.as_mut_ptr(), a.len() as u32, noname()) }
    }
    /// Call with a single `funclet` operand bundle.
    pub fn call_funclet(&self, fnty: Ty, f: V, args: &[V], pad: V) -> V {
        let mut a = args.to_vec();
        let tag = "funclet";
        unsafe {
            let mut padv = [pad];
            let bundle = LLVMCreateOperandBundle(tag.as_ptr() as *const _, tag.len(), padv.as_mut_ptr(), 1);
            let mut bundles = [bundle];
            let r = LLVMBuildCallWithOperandBundles(self.b, fnty, f, a.as_mut_ptr(), a.len() as u32, bundles.as_mut_ptr(), 1, noname());
            LLVMDisposeOperandBundle(bundle);
            r
        }
    }
    pub fn invoke(&self, fnty: Ty, f: V, args: &[V], normal: BB, unwind: BB) -> V {
        let mut a = args.to_vec();
        unsafe { LLVMBuildInvoke2(self.b, fnty, f, a.as_mut_ptr(), a.len() as u32, normal, unwind, noname()) }
    }
    pub fn landing_pad(&self, t: Ty, n: u32) -> V {
        unsafe { LLVMBuildLandingPad(self.b, t, null_mut(), n, noname()) }
    }
    pub fn set_cleanup(lp: V, v: bool) {
        unsafe { LLVMSetCleanup(lp, v as i32) }
    }
    pub fn add_clause(lp: V, c: V) {
        unsafe { LLVMAddClause(lp, c) }
    }
    pub fn resume(&self, v: V) -> V {
        unsafe { LLVMBuildResume(self.b, v) }
    }
    pub fn catch_switch(&self, parent: V, unwind: Option<BB>, n: u32) -> V {
        unsafe { LLVMBuildCatchSwitch(self.b, parent, unwind.unwrap_or(null_mut()), n, noname()) }
    }
    pub fn add_handler(cs: V, bb: BB) {
        unsafe { LLVMAddHandler(cs, bb) }
    }
    pub fn catch_pad(&self, parent: V, args: &[V]) -> V {
        let mut a = args.to_vec();
        unsafe { LLVMBuildCatchPad(self.b, parent, a.as_mut_ptr(), a.len() as u32, noname()) }
    }
    pub fn catch_ret(&self, pad: V, bb: BB) -> V {
        unsafe { LLVMBuildCatchRet(self.b, pad, bb) }
    }
    pub fn cleanup_pad(&self, parent: V) -> V {
        unsafe { LLVMBuildCleanupPad(self.b, parent, null_mut(), 0, noname()) }
    }
    pub fn cleanup_ret(&self, pad: V) -> V {
        unsafe { LLVMBuildCleanupRet(self.b, pad, null_mut()) }
    }
}

// ------------------------------------------------------------------ debug info

/// RAII `llvm::DIBuilder`.
pub struct DiBuilder(pub LLVMDIBuilderRef);

impl Drop for DiBuilder {
    fn drop(&mut self) {
        unsafe { LLVMDisposeDIBuilder(self.0) }
    }
}

pub const DW_ATE_ADDRESS: u32 = 0x01;
pub const DW_ATE_BOOLEAN: u32 = 0x02;
pub const DW_ATE_FLOAT: u32 = 0x04;
pub const DW_ATE_SIGNED: u32 = 0x05;
pub const DW_ATE_UNSIGNED: u32 = 0x08;
pub const DW_ATE_HP_FLOAT128: u32 = 0x82;
pub const DW_TAG_REFERENCE_TYPE: u32 = 0x10;
pub const DW_TAG_STRUCTURE_TYPE: u32 = 0x13;
pub const DI_FLAG_ZERO: LLVMDIFlags = 0;
pub const DI_FLAG_ARTIFICIAL: LLVMDIFlags = 1 << 6;

impl DiBuilder {
    pub fn new(m: LLVMModuleRef) -> Self {
        DiBuilder(unsafe { LLVMCreateDIBuilder(m) })
    }
    pub fn finalize(&self) {
        unsafe { LLVMDIBuilderFinalize(self.0) }
    }
    pub fn file(&self, path: &str) -> Md {
        // DebugInfo::getFile
        let (filename, directory) = match path.rfind('/') {
            Some(p) => (&path[p + 1..], &path[..p]),
            None => (path, "."),
        };
        unsafe { LLVMDIBuilderCreateFile(self.0, filename.as_ptr() as *const _, filename.len(), directory.as_ptr() as *const _, directory.len()) }
    }
    pub fn compile_unit(&self, file: Md, producer: &str, optimized: bool) -> Md {
        unsafe {
            LLVMDIBuilderCreateCompileUnit(
                self.0,
                LLVMDWARFSourceLanguage::LLVMDWARFSourceLanguageC,
                file,
                producer.as_ptr() as *const _,
                producer.len(),
                optimized as i32,
                noname(),
                0,
                0,
                noname(),
                0,
                LLVMDWARFEmissionKind::LLVMDWARFEmissionKindFull,
                0,
                1,
                0,
                noname(),
                0,
                noname(),
                0,
            )
        }
    }
    pub fn basic_type(&self, name: &str, bits: u64, enc: u32) -> Md {
        unsafe { LLVMDIBuilderCreateBasicType(self.0, name.as_ptr() as *const _, name.len(), bits, enc, DI_FLAG_ZERO) }
    }
    pub fn pointer_type(&self, pointee: Md, bits: u64) -> Md {
        unsafe { LLVMDIBuilderCreatePointerType(self.0, pointee, bits, 0, 0, noname(), 0) }
    }
    pub fn reference_type(&self, tag: u32, t: Md) -> Md {
        unsafe { LLVMDIBuilderCreateReferenceType(self.0, tag, t) }
    }
    pub fn subroutine_type(&self, file: Md, types: &[Md]) -> Md {
        let mut t = types.to_vec();
        unsafe { LLVMDIBuilderCreateSubroutineType(self.0, file, t.as_mut_ptr(), t.len() as u32, DI_FLAG_ZERO) }
    }
    /// Temporary (replaceable) struct used to break recursion; finalized via
    /// [`DiBuilder::finish_struct`] (C++ `createStructType` + `replaceArrays`).
    pub fn temp_struct(&self, scope: Md, name: &str, file: Md, line: u32, bits: u64) -> Md {
        unsafe {
            LLVMDIBuilderCreateReplaceableCompositeType(
                self.0,
                DW_TAG_STRUCTURE_TYPE,
                name.as_ptr() as *const _,
                name.len(),
                scope,
                file,
                line,
                0,
                bits,
                0,
                DI_FLAG_ZERO,
                noname(),
                0,
            )
        }
    }
    pub fn finish_struct(&self, temp: Md, scope: Md, name: &str, file: Md, line: u32, bits: u64, members: &[Md]) -> Md {
        let mut mem = members.to_vec();
        unsafe {
            let real = LLVMDIBuilderCreateStructType(
                self.0,
                scope,
                name.as_ptr() as *const _,
                name.len(),
                file,
                line,
                bits,
                0,
                DI_FLAG_ZERO,
                null_mut(),
                mem.as_mut_ptr(),
                mem.len() as u32,
                0,
                null_mut(),
                noname(),
                0,
            );
            LLVMMetadataReplaceAllUsesWith(temp, real);
            real
        }
    }
    #[allow(clippy::too_many_arguments)]
    pub fn member_type(&self, scope: Md, name: &str, file: Md, line: u32, bits: u64, offset_bits: u64, flags: LLVMDIFlags, ty: Md) -> Md {
        unsafe { LLVMDIBuilderCreateMemberType(self.0, scope, name.as_ptr() as *const _, name.len(), file, line, bits, 0, offset_bits, flags, ty) }
    }
    #[allow(clippy::too_many_arguments)]
    pub fn function(&self, scope: Md, name: &str, linkage: &str, file: Md, line: u32, ty: Md, local: bool, def: bool, optimized: bool) -> Md {
        unsafe {
            LLVMDIBuilderCreateFunction(
                self.0,
                scope,
                name.as_ptr() as *const _,
                name.len(),
                linkage.as_ptr() as *const _,
                linkage.len(),
                file,
                line,
                ty,
                local as i32,
                def as i32,
                0,
                DI_FLAG_ZERO,
                optimized as i32,
            )
        }
    }
    #[allow(clippy::too_many_arguments)]
    pub fn global_var_expr(&self, scope: Md, name: &str, linkage: &str, file: Md, line: u32, ty: Md, local: bool) -> Md {
        unsafe {
            let expr = LLVMDIBuilderCreateExpression(self.0, null_mut(), 0);
            LLVMDIBuilderCreateGlobalVariableExpression(
                self.0,
                scope,
                name.as_ptr() as *const _,
                name.len(),
                linkage.as_ptr() as *const _,
                linkage.len(),
                file,
                line,
                ty,
                local as i32,
                expr,
                null_mut(),
                0,
            )
        }
    }
    #[allow(clippy::too_many_arguments)]
    pub fn parameter_variable(&self, scope: Md, name: &str, arg: u32, file: Md, line: u32, ty: Md, preserve: bool) -> Md {
        unsafe { LLVMDIBuilderCreateParameterVariable(self.0, scope, name.as_ptr() as *const _, name.len(), arg, file, line, ty, preserve as i32, DI_FLAG_ZERO) }
    }
    pub fn auto_variable(&self, scope: Md, name: &str, file: Md, line: u32, ty: Md, preserve: bool) -> Md {
        unsafe { LLVMDIBuilderCreateAutoVariable(self.0, scope, name.as_ptr() as *const _, name.len(), file, line, ty, preserve as i32, DI_FLAG_ZERO, 0) }
    }
    pub fn insert_declare(&self, storage: V, var: Md, loc: Md, bb: BB) {
        unsafe {
            let expr = LLVMDIBuilderCreateExpression(self.0, null_mut(), 0);
            LLVMDIBuilderInsertDeclareRecordAtEnd(self.0, storage, var, expr, loc, bb);
        }
    }
}

pub fn di_location(c: LLVMContextRef, line: i32, col: i32, scope: Md) -> Md {
    unsafe { LLVMDIBuilderCreateDebugLocation(c, line.max(0) as u32, col.max(0) as u32, scope, null_mut()) }
}
pub fn subprogram(f: V) -> Md {
    unsafe { LLVMGetSubprogram(f) }
}
pub fn set_subprogram(f: V, sp: Md) {
    unsafe { LLVMSetSubprogram(f, sp) }
}
/// `GlobalVariable::addDebugInfo(gve)` (`!dbg` attachment).
pub fn add_global_debug_info(g: V, gve: Md) {
    unsafe {
        let c = LLVMGetTypeContext(type_of(g));
        let k = LLVMGetMDKindIDInContext(c, "dbg".as_ptr() as *const _, 3);
        LLVMGlobalSetMetadata(g, k, gve);
    }
}
pub fn add_module_flag_int(m: LLVMModuleRef, key: &str, val: u32) {
    unsafe {
        let c = LLVMGetModuleContext(m);
        let v = LLVMValueAsMetadata(const_int(int_ty(c, 32), val as u64, false));
        LLVMAddModuleFlag(m, llvm_sys::LLVMModuleFlagBehavior::LLVMModuleFlagBehaviorWarning, key.as_ptr() as *const _, key.len(), v);
    }
}
pub fn debug_metadata_version() -> u32 {
    unsafe { LLVMDebugMetadataVersion() }
}

// ------------------------------------------------------------------ IR parsing / linking

/// `parseIR` + `Linker::linkInModule`. Returns the parse error on failure.
pub fn parse_and_link(dest: LLVMModuleRef, code: &str) -> Result<(), String> {
    unsafe {
        let c = LLVMGetModuleContext(dest);
        let name = cstr("llvm-func");
        let buf = LLVMCreateMemoryBufferWithMemoryRangeCopy(code.as_ptr() as *const _, code.len(), name.as_ptr());
        let mut sub: LLVMModuleRef = null_mut();
        let mut msg: *mut libc::c_char = null_mut();
        // LLVMParseIRInContext takes ownership of `buf`.
        if llvm_sys::ir_reader::LLVMParseIRInContext(c, buf, &mut sub, &mut msg) != 0 {
            let s = if msg.is_null() { String::new() } else { CStr::from_ptr(msg).to_string_lossy().into_owned() };
            if !msg.is_null() {
                LLVMDisposeMessage(msg);
            }
            return Err(s);
        }
        LLVMSetDataLayout(sub, LLVMGetDataLayoutStr(dest));
        if llvm_sys::linker::LLVMLinkModules2(dest, sub) != 0 {
            return Err("linking failed".into());
        }
        Ok(())
    }
}
