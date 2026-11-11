//! Thin helpers over the raw LLVM C API (`llvm-sys`).
//!
//! The C++ passes use `llvm::Instruction`, `llvm::CallBase`, ... directly. The
//! Rust port operates on raw `LLVMValueRef`s through these helpers. Every helper
//! expects a valid, non-null handle of the documented kind; they are crate
//! private and only ever called with handles obtained from LLVM itself.

#![allow(dead_code)]

use std::ffi::{CStr, CString};

use llvm_sys::LLVMOpcode;
use llvm_sys::LLVMTypeKind;
use llvm_sys::core::*;
use llvm_sys::prelude::*;

pub type V = LLVMValueRef;
pub type BB = LLVMBasicBlockRef;
pub type Ty = LLVMTypeRef;

pub const RET_INDEX: u32 = llvm_sys::LLVMAttributeReturnIndex;
pub const FN_INDEX: u32 = llvm_sys::LLVMAttributeFunctionIndex;

/// Convert a (possibly null) LLVM error into a Rust string, consuming it.
pub fn take_error(err: llvm_sys::error::LLVMErrorRef) -> Option<String> {
    if err.is_null() {
        return None;
    }
    unsafe {
        let msg = llvm_sys::error::LLVMGetErrorMessage(err);
        let s = CStr::from_ptr(msg).to_string_lossy().into_owned();
        llvm_sys::error::LLVMDisposeErrorMessage(msg);
        Some(s)
    }
}

pub fn cstr(s: &str) -> CString {
    CString::new(s.replace('\0', "")).expect("no interior NUL")
}

// ---------------------------------------------------------------- names

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

pub fn block_name(b: BB) -> String {
    unsafe {
        let p = LLVMGetBasicBlockName(b);
        if p.is_null() { String::new() } else { CStr::from_ptr(p).to_string_lossy().into_owned() }
    }
}

// ---------------------------------------------------------------- module iteration

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

pub fn globals(m: LLVMModuleRef) -> Vec<V> {
    let mut out = Vec::new();
    unsafe {
        let mut g = LLVMGetFirstGlobal(m);
        while !g.is_null() {
            out.push(g);
            g = LLVMGetNextGlobal(g);
        }
    }
    out
}

pub fn aliases(m: LLVMModuleRef) -> Vec<V> {
    let mut out = Vec::new();
    unsafe {
        let mut g = LLVMGetFirstGlobalAlias(m);
        while !g.is_null() {
            out.push(g);
            g = LLVMGetNextGlobalAlias(g);
        }
    }
    out
}

pub fn get_function(m: LLVMModuleRef, n: &str) -> Option<V> {
    let c = cstr(n);
    let f = unsafe { LLVMGetNamedFunction(m, c.as_ptr()) };
    (!f.is_null()).then_some(f)
}

pub fn get_global(m: LLVMModuleRef, n: &str) -> Option<V> {
    let c = cstr(n);
    let f = unsafe { LLVMGetNamedGlobal(m, c.as_ptr()) };
    (!f.is_null()).then_some(f)
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

pub fn fn_instructions(f: V) -> Vec<V> {
    blocks(f).into_iter().flat_map(instructions).collect()
}

pub fn instruction_count(f: V) -> usize {
    blocks(f).into_iter().map(|b| instructions(b).len()).sum()
}

pub fn first_instruction(b: BB) -> Option<V> {
    let i = unsafe { LLVMGetFirstInstruction(b) };
    (!i.is_null()).then_some(i)
}

pub fn first_non_phi(b: BB) -> Option<V> {
    instructions(b).into_iter().find(|&i| opcode(i) != LLVMOpcode::LLVMPHI)
}

/// `BasicBlock::getFirstInsertionPt` (skips PHIs and EH pads).
pub fn first_insertion_pt(b: BB) -> Option<V> {
    instructions(b).into_iter().find(|&i| {
        !matches!(
            opcode(i),
            LLVMOpcode::LLVMPHI
                | LLVMOpcode::LLVMLandingPad
                | LLVMOpcode::LLVMCatchPad
                | LLVMOpcode::LLVMCleanupPad
                | LLVMOpcode::LLVMCatchSwitch
        )
    })
}

pub fn terminator(b: BB) -> Option<V> {
    let t = unsafe { LLVMGetBasicBlockTerminator(b) };
    (!t.is_null()).then_some(t)
}

pub fn successors(b: BB) -> Vec<BB> {
    match terminator(b) {
        Some(t) => unsafe { (0..LLVMGetNumSuccessors(t)).map(|i| LLVMGetSuccessor(t, i)).collect() },
        None => Vec::new(),
    }
}

pub fn parent_block(i: V) -> BB {
    unsafe { LLVMGetInstructionParent(i) }
}

pub fn block_function(b: BB) -> V {
    unsafe { LLVMGetBasicBlockParent(b) }
}

pub fn inst_function(i: V) -> V {
    block_function(parent_block(i))
}

pub fn global_module(g: V) -> LLVMModuleRef {
    unsafe { LLVMGetGlobalParent(g) }
}

pub fn entry_block(f: V) -> Option<BB> {
    if is_declaration(f) {
        return None;
    }
    Some(unsafe { LLVMGetEntryBasicBlock(f) })
}

/// True if this block's first non-PHI instruction is an EH pad.
pub fn is_eh_pad(b: BB) -> bool {
    first_non_phi(b).is_some_and(|i| {
        matches!(
            opcode(i),
            LLVMOpcode::LLVMLandingPad
                | LLVMOpcode::LLVMCatchPad
                | LLVMOpcode::LLVMCleanupPad
                | LLVMOpcode::LLVMCatchSwitch
        )
    })
}

// ---------------------------------------------------------------- value kinds

pub fn opcode(i: V) -> LLVMOpcode {
    unsafe { LLVMGetInstructionOpcode(i) }
}

pub fn is_instruction(v: V) -> bool {
    unsafe { !LLVMIsAInstruction(v).is_null() }
}

pub fn is_argument(v: V) -> bool {
    unsafe { !LLVMIsAArgument(v).is_null() }
}

pub fn is_function(v: V) -> bool {
    unsafe { !LLVMIsAFunction(v).is_null() }
}

pub fn is_global_value(v: V) -> bool {
    unsafe { !LLVMIsAGlobalValue(v).is_null() }
}

pub fn is_global_variable(v: V) -> bool {
    unsafe { !LLVMIsAGlobalVariable(v).is_null() }
}

pub fn is_constant(v: V) -> bool {
    unsafe { LLVMIsConstant(v) != 0 }
}

pub fn is_constant_int(v: V) -> bool {
    unsafe { !LLVMIsAConstantInt(v).is_null() }
}

pub fn is_null_ptr_constant(v: V) -> bool {
    unsafe { !LLVMIsAConstantPointerNull(v).is_null() }
}

/// `Constant::isNullValue()`.
pub fn is_null_value(v: V) -> bool {
    is_constant(v) && unsafe { LLVMIsNull(v) != 0 }
}

pub fn const_zext(v: V) -> Option<u64> {
    is_constant_int(v).then(|| unsafe { LLVMConstIntGetZExtValue(v) })
}

pub fn const_sext(v: V) -> Option<i64> {
    is_constant_int(v).then(|| unsafe { LLVMConstIntGetSExtValue(v) })
}

pub fn is_call(i: V) -> bool {
    is_instruction(i) && matches!(opcode(i), LLVMOpcode::LLVMCall | LLVMOpcode::LLVMInvoke | LLVMOpcode::LLVMCallBr)
}

pub fn is_phi(v: V) -> bool {
    unsafe { !LLVMIsAPHINode(v).is_null() }
}

// ---------------------------------------------------------------- types

pub fn type_of(v: V) -> Ty {
    unsafe { LLVMTypeOf(v) }
}

pub fn type_kind(t: Ty) -> LLVMTypeKind {
    unsafe { LLVMGetTypeKind(t) }
}

pub fn is_pointer_ty(t: Ty) -> bool {
    type_kind(t) == LLVMTypeKind::LLVMPointerTypeKind
}

pub fn is_void_ty(t: Ty) -> bool {
    type_kind(t) == LLVMTypeKind::LLVMVoidTypeKind
}

pub fn is_int_ty(t: Ty, bits: u32) -> bool {
    type_kind(t) == LLVMTypeKind::LLVMIntegerTypeKind && unsafe { LLVMGetIntTypeWidth(t) } == bits
}

pub fn is_fp_ty(t: Ty) -> bool {
    use LLVMTypeKind::*;
    matches!(
        type_kind(t),
        LLVMHalfTypeKind
            | LLVMBFloatTypeKind
            | LLVMFloatTypeKind
            | LLVMDoubleTypeKind
            | LLVMX86_FP80TypeKind
            | LLVMFP128TypeKind
            | LLVMPPC_FP128TypeKind
    )
}

pub fn pointer_address_space(t: Ty) -> u32 {
    unsafe { LLVMGetPointerAddressSpace(t) }
}

/// `StructType` with no name and no elements (`{}`).
pub fn is_empty_literal_struct(t: Ty) -> bool {
    type_kind(t) == LLVMTypeKind::LLVMStructTypeKind
        && unsafe { LLVMCountStructElementTypes(t) } == 0
        && unsafe { LLVMGetStructName(t) }.is_null()
}

pub fn function_type(f: V) -> Ty {
    unsafe { LLVMGlobalGetValueType(f) }
}

pub fn return_type(f: V) -> Ty {
    unsafe { LLVMGetReturnType(function_type(f)) }
}

// ---------------------------------------------------------------- operands / uses

pub fn num_operands(v: V) -> u32 {
    unsafe { LLVMGetNumOperands(v) as u32 }
}

pub fn operand(v: V, i: u32) -> V {
    unsafe { LLVMGetOperand(v, i) }
}

pub fn operands(v: V) -> Vec<V> {
    (0..num_operands(v)).map(|i| operand(v, i)).collect()
}

/// Distinct users of `v` (in use-list order; a user appears once per use).
pub fn users(v: V) -> Vec<V> {
    let mut out = Vec::new();
    unsafe {
        let mut u = LLVMGetFirstUse(v);
        while !u.is_null() {
            out.push(LLVMGetUser(u));
            u = LLVMGetNextUse(u);
        }
    }
    out
}

pub fn unique_users(v: V) -> Vec<V> {
    let mut out: Vec<V> = Vec::new();
    for u in users(v) {
        if !out.contains(&u) {
            out.push(u);
        }
    }
    out
}

pub fn uses(v: V) -> Vec<LLVMUseRef> {
    let mut out = Vec::new();
    unsafe {
        let mut u = LLVMGetFirstUse(v);
        while !u.is_null() {
            out.push(u);
            u = LLVMGetNextUse(u);
        }
    }
    out
}

pub fn use_user(u: LLVMUseRef) -> V {
    unsafe { LLVMGetUser(u) }
}

/// Operand index of `u` inside its user, if found.
pub fn use_operand_no(u: LLVMUseRef) -> Option<u32> {
    let user = use_user(u);
    (0..num_operands(user)).find(|&i| unsafe { LLVMGetOperandUse(user, i) } == u)
}

pub fn use_empty(v: V) -> bool {
    unsafe { LLVMGetFirstUse(v).is_null() }
}

pub fn replace_all_uses(old: V, new: V) {
    unsafe { LLVMReplaceAllUsesWith(old, new) }
}

pub fn erase(i: V) {
    unsafe { LLVMInstructionEraseFromParent(i) }
}

/// `Instruction::dropAllReferences`: set every operand to poison/undef of its type.
pub fn drop_all_references(i: V) {
    for idx in 0..num_operands(i) {
        let op = operand(i, idx);
        if op.is_null() {
            continue;
        }
        let ty = type_of(op);
        if type_kind(ty) == LLVMTypeKind::LLVMLabelTypeKind || type_kind(ty) == LLVMTypeKind::LLVMMetadataTypeKind {
            continue;
        }
        unsafe { LLVMSetOperand(i, idx, LLVMGetPoison(ty)) };
    }
}

// ---------------------------------------------------------------- calls

pub fn num_args(call: V) -> u32 {
    unsafe { LLVMGetNumArgOperands(call) }
}

pub fn arg(call: V, i: u32) -> V {
    operand(call, i)
}

pub fn args(call: V) -> Vec<V> {
    (0..num_args(call)).map(|i| arg(call, i)).collect()
}

pub fn called_value(call: V) -> V {
    unsafe { LLVMGetCalledValue(call) }
}

/// `CallBase::getCalledFunction()`: direct callee with matching function type.
pub fn called_function(call: V) -> Option<V> {
    if !is_call(call) {
        return None;
    }
    let c = called_value(call);
    if c.is_null() || !is_function(c) {
        return None;
    }
    if function_type(c) != unsafe { LLVMGetCalledFunctionType(call) } {
        return None;
    }
    Some(c)
}

pub fn intrinsic_id(f: V) -> u32 {
    unsafe { LLVMGetIntrinsicID(f) }
}

/// If `v` is a call to an intrinsic, its intrinsic ID.
pub fn call_intrinsic_id(v: V) -> Option<u32> {
    let f = called_function(v)?;
    let id = intrinsic_id(f);
    (id != 0).then_some(id)
}

pub fn lookup_intrinsic(n: &str) -> u32 {
    unsafe { LLVMLookupIntrinsicID(n.as_ptr() as *const _, n.len()) }
}

pub fn num_operand_bundles(call: V) -> u32 {
    if opcode(call) == LLVMOpcode::LLVMCall || opcode(call) == LLVMOpcode::LLVMInvoke {
        unsafe { LLVMGetNumOperandBundles(call) }
    } else {
        0
    }
}

pub fn is_tail_call(call: V) -> bool {
    opcode(call) == LLVMOpcode::LLVMCall && unsafe { LLVMIsTailCall(call) } != 0
}

pub fn is_must_tail_call(call: V) -> bool {
    opcode(call) == LLVMOpcode::LLVMCall
        && unsafe { LLVMGetTailCallKind(call) } == llvm_sys::LLVMTailCallKind::LLVMTailCallKindMustTail
}

pub fn set_tail_call(call: V, tail: bool) {
    unsafe { LLVMSetTailCall(call, tail as i32) }
}

pub fn attr_kind(n: &str) -> u32 {
    unsafe { LLVMGetEnumAttributeKindForName(n.as_ptr() as *const _, n.len()) }
}

fn call_enum_attr(call: V, idx: u32, kind: &str) -> Option<u64> {
    let a = unsafe { LLVMGetCallSiteEnumAttribute(call, idx, attr_kind(kind)) };
    (!a.is_null()).then(|| unsafe { LLVMGetEnumAttributeValue(a) })
}

fn fn_enum_attr(f: V, idx: u32, kind: &str) -> Option<u64> {
    let a = unsafe { LLVMGetEnumAttributeAtIndex(f, idx, attr_kind(kind)) };
    (!a.is_null()).then(|| unsafe { LLVMGetEnumAttributeValue(a) })
}

/// `CallBase::hasFnAttr(kind)`: call site or callee.
pub fn call_has_fn_attr(call: V, kind: &str) -> bool {
    call_enum_attr(call, FN_INDEX, kind).is_some()
        || called_function(call).is_some_and(|f| fn_enum_attr(f, FN_INDEX, kind).is_some())
}

/// Call-site-only function attribute.
pub fn call_site_has_fn_attr(call: V, kind: &str) -> bool {
    call_enum_attr(call, FN_INDEX, kind).is_some()
}

/// `CallBase::paramHasAttr(i, kind)`: call site or callee.
pub fn call_param_has_attr(call: V, i: u32, kind: &str) -> bool {
    call_enum_attr(call, i + 1, kind).is_some()
        || called_function(call)
            .is_some_and(|f| i < count_params(f) && fn_enum_attr(f, i + 1, kind).is_some())
}

/// `CallBase::getParamAlign(i)` (call-site only).
pub fn call_param_align(call: V, i: u32) -> Option<u64> {
    call_enum_attr(call, i + 1, "align")
}

/// `Function::getAttributes().getParamAlignment(i)`.
pub fn fn_param_align(f: V, i: u32) -> Option<u64> {
    if i >= count_params(f) {
        return None;
    }
    fn_enum_attr(f, i + 1, "align")
}

/// `CallBase::getRetAlign()` (call-site only).
pub fn call_ret_align(call: V) -> Option<u64> {
    call_enum_attr(call, RET_INDEX, "align")
}

pub fn fn_ret_align(f: V) -> Option<u64> {
    fn_enum_attr(f, RET_INDEX, "align")
}

pub fn call_ret_has_attr(call: V, kind: &str) -> bool {
    call_enum_attr(call, RET_INDEX, kind).is_some()
        || called_function(call).is_some_and(|f| fn_enum_attr(f, RET_INDEX, kind).is_some())
}

pub fn fn_has_attr(f: V, kind: &str) -> bool {
    fn_enum_attr(f, FN_INDEX, kind).is_some()
}

pub fn fn_param_has_attr(f: V, i: u32, kind: &str) -> bool {
    fn_enum_attr(f, i + 1, kind).is_some()
}

pub fn fn_has_string_attr(f: V, key: &str) -> bool {
    unsafe { !LLVMGetStringAttributeAtIndex(f, FN_INDEX, key.as_ptr() as *const _, key.len() as u32).is_null() }
}

pub fn call_site_has_string_attr(call: V, key: &str) -> bool {
    unsafe { !LLVMGetCallSiteStringAttribute(call, FN_INDEX, key.as_ptr() as *const _, key.len() as u32).is_null() }
}

pub fn add_fn_enum_attr(f: V, kind: &str, val: u64) {
    unsafe {
        let ctx = LLVMGetTypeContext(type_of(f));
        let a = LLVMCreateEnumAttribute(ctx, attr_kind(kind), val);
        LLVMAddAttributeAtIndex(f, FN_INDEX, a);
    }
}

pub fn add_call_enum_attr(call: V, idx: u32, kind: &str, val: u64) {
    unsafe {
        let ctx = LLVMGetTypeContext(type_of(call));
        let a = LLVMCreateEnumAttribute(ctx, attr_kind(kind), val);
        LLVMAddCallSiteAttribute(call, idx, a);
    }
}

pub fn remove_fn_enum_attr(f: V, kind: &str) {
    unsafe { LLVMRemoveEnumAttributeAtIndex(f, FN_INDEX, attr_kind(kind)) }
}

pub fn add_fn_string_attr(f: V, key: &str, val: &str) {
    unsafe {
        let ctx = LLVMGetTypeContext(type_of(f));
        let a = LLVMCreateStringAttribute(
            ctx,
            key.as_ptr() as *const _,
            key.len() as u32,
            val.as_ptr() as *const _,
            val.len() as u32,
        );
        LLVMAddAttributeAtIndex(f, FN_INDEX, a);
    }
}

/// All attributes at `idx` (to copy them).
pub fn fn_attributes(f: V, idx: u32) -> Vec<llvm_sys::prelude::LLVMAttributeRef> {
    unsafe {
        let n = LLVMGetAttributeCountAtIndex(f, idx);
        let mut out = vec![std::ptr::null_mut(); n as usize];
        if n > 0 {
            LLVMGetAttributesAtIndex(f, idx, out.as_mut_ptr());
        }
        out
    }
}

// ---------------------------------------------------------------- functions / globals

pub fn is_declaration(g: V) -> bool {
    unsafe { LLVMIsDeclaration(g) != 0 }
}

pub fn count_params(f: V) -> u32 {
    unsafe { LLVMCountParams(f) }
}

pub fn param(f: V, i: u32) -> V {
    unsafe { LLVMGetParam(f, i) }
}

pub fn linkage(g: V) -> llvm_sys::LLVMLinkage {
    unsafe { LLVMGetLinkage(g) }
}

pub fn set_linkage(g: V, l: llvm_sys::LLVMLinkage) {
    unsafe { LLVMSetLinkage(g, l) }
}

pub fn has_local_linkage(g: V) -> bool {
    matches!(
        linkage(g),
        llvm_sys::LLVMLinkage::LLVMInternalLinkage | llvm_sys::LLVMLinkage::LLVMPrivateLinkage
    )
}

/// `GlobalValue::hasExactDefinition()`.
pub fn has_exact_definition(f: V) -> bool {
    use llvm_sys::LLVMLinkage::*;
    !is_declaration(f)
        && !matches!(
            linkage(f),
            LLVMAvailableExternallyLinkage
                | LLVMLinkOnceAnyLinkage
                | LLVMLinkOnceODRLinkage
                | LLVMWeakAnyLinkage
                | LLVMWeakODRLinkage
                | LLVMExternalWeakLinkage
                | LLVMCommonLinkage
        )
}

pub fn module_context(m: LLVMModuleRef) -> LLVMContextRef {
    unsafe { LLVMGetModuleContext(m) }
}

pub fn value_context(v: V) -> LLVMContextRef {
    unsafe { LLVMGetTypeContext(type_of(v)) }
}

// ---------------------------------------------------------------- misc instruction queries

pub fn icmp_predicate(i: V) -> llvm_sys::LLVMIntPredicate {
    unsafe { LLVMGetICmpPredicate(i) }
}

pub fn is_equality_icmp(i: V) -> bool {
    opcode(i) == LLVMOpcode::LLVMICmp
        && matches!(
            icmp_predicate(i),
            llvm_sys::LLVMIntPredicate::LLVMIntEQ | llvm_sys::LLVMIntPredicate::LLVMIntNE
        )
}

pub fn is_volatile(i: V) -> bool {
    unsafe { LLVMGetVolatile(i) != 0 }
}

pub fn alignment(i: V) -> u64 {
    unsafe { LLVMGetAlignment(i) as u64 }
}

/// Indices of an extractvalue/insertvalue instruction.
pub fn agg_indices(i: V) -> Vec<u32> {
    unsafe {
        let n = LLVMGetNumIndices(i);
        let p = LLVMGetIndices(i);
        if p.is_null() { Vec::new() } else { std::slice::from_raw_parts(p, n as usize).to_vec() }
    }
}

pub fn phi_incoming(phi: V) -> Vec<(V, BB)> {
    unsafe {
        (0..LLVMCountIncoming(phi)).map(|i| (LLVMGetIncomingValue(phi, i), LLVMGetIncomingBlock(phi, i))).collect()
    }
}

pub fn phis(b: BB) -> Vec<V> {
    instructions(b).into_iter().take_while(|&i| opcode(i) == LLVMOpcode::LLVMPHI).collect()
}

pub fn is_conditional_br(t: V) -> bool {
    opcode(t) == LLVMOpcode::LLVMBr && unsafe { LLVMIsConditional(t) } != 0
}

pub fn br_condition(t: V) -> V {
    unsafe { LLVMGetCondition(t) }
}

pub fn successor(t: V, i: u32) -> BB {
    unsafe { LLVMGetSuccessor(t, i) }
}

pub fn set_successor(t: V, i: u32, b: BB) {
    unsafe { LLVMSetSuccessor(t, i, b) }
}

pub fn num_successors(t: V) -> u32 {
    unsafe { LLVMGetNumSuccessors(t) }
}

// ---------------------------------------------------------------- metadata

pub fn md_kind(ctx: LLVMContextRef, n: &str) -> u32 {
    unsafe { LLVMGetMDKindIDInContext(ctx, n.as_ptr() as *const _, n.len() as u32) }
}

/// Metadata operands of an MDNode (given as MetadataAsValue).
pub fn md_operands(node: V) -> Vec<V> {
    unsafe {
        let n = LLVMGetMDNodeNumOperands(node);
        let mut out = vec![std::ptr::null_mut(); n as usize];
        if n > 0 {
            LLVMGetMDNodeOperands(node, out.as_mut_ptr());
        }
        out
    }
}

pub fn md_string(v: V) -> Option<String> {
    if v.is_null() {
        return None;
    }
    unsafe {
        if LLVMIsAMDString(v).is_null() {
            return None;
        }
        let mut len = 0u32;
        let p = LLVMGetMDString(v, &mut len);
        if p.is_null() {
            return None;
        }
        Some(String::from_utf8_lossy(std::slice::from_raw_parts(p as *const u8, len as usize)).into_owned())
    }
}

pub fn is_md_node(v: V) -> bool {
    !v.is_null() && unsafe { !LLVMIsAMDNode(v).is_null() }
}

// ---------------------------------------------------------------- builders

/// RAII wrapper over `LLVMBuilderRef`.
pub struct Builder(pub LLVMBuilderRef);

impl Builder {
    pub fn new(ctx: LLVMContextRef) -> Self {
        Builder(unsafe { LLVMCreateBuilderInContext(ctx) })
    }
    pub fn before(&self, i: V) {
        unsafe { LLVMPositionBuilderBefore(self.0, i) }
    }
    pub fn at_end(&self, b: BB) {
        unsafe { LLVMPositionBuilderAtEnd(self.0, b) }
    }
    pub fn insert(&self, i: V) {
        unsafe { LLVMInsertIntoBuilder(self.0, i) }
    }
}

impl Drop for Builder {
    fn drop(&mut self) {
        unsafe { LLVMDisposeBuilder(self.0) }
    }
}

/// `Instruction::moveBefore(pos)`.
pub fn move_before(i: V, pos: V) {
    let b = Builder::new(value_context(i));
    b.before(pos);
    unsafe {
        LLVMInstructionRemoveFromParent(i);
    }
    b.insert(i);
}

/// Move `i` to the end of block `bb` (before nothing, i.e. append).
pub fn move_to_end(i: V, bb: BB) {
    let b = Builder::new(value_context(i));
    b.at_end(bb);
    unsafe {
        LLVMInstructionRemoveFromParent(i);
    }
    b.insert(i);
}

/// `BasicBlock::splitBasicBlock(i)`: move `i` and everything after it into a new
/// block inserted right after `bb`, and branch `bb` to it. PHIs in the successors
/// of the moved terminator are updated to reference the new block.
pub fn split_block(i: V, name: &str) -> BB {
    let bb = parent_block(i);
    let ctx = value_context(i);
    let new_bb = new_block_after(bb, name);
    let mut moving = Vec::new();
    let mut cur = i;
    while !cur.is_null() {
        moving.push(cur);
        cur = unsafe { LLVMGetNextInstruction(cur) };
    }
    for &m in &moving {
        move_to_end(m, new_bb);
    }
    for succ in successors(new_bb) {
        for phi in phis(succ) {
            replace_phi_block(phi, bb, new_bb);
        }
    }
    let b = Builder::new(ctx);
    b.at_end(bb);
    unsafe { LLVMBuildBr(b.0, new_bb) };
    new_bb
}

/// Replace incoming block `old` with `new` in a PHI.
pub fn replace_phi_block(phi: V, old: BB, new: BB) {
    // The C API has no setIncomingBlock; rebuild the PHI if needed.
    let inc = phi_incoming(phi);
    if !inc.iter().any(|&(_, b)| b == old) {
        return;
    }
    let b = Builder::new(value_context(phi));
    b.before(phi);
    let n = cstr(&name(phi));
    let np = unsafe { LLVMBuildPhi(b.0, type_of(phi), n.as_ptr()) };
    let mut vals: Vec<V> = inc.iter().map(|&(v, _)| v).collect();
    let mut blks: Vec<BB> = inc.iter().map(|&(_, bb)| if bb == old { new } else { bb }).collect();
    unsafe { LLVMAddIncoming(np, vals.as_mut_ptr(), blks.as_mut_ptr(), vals.len() as u32) };
    replace_all_uses(phi, np);
    // self-references were rewritten by RAUW
    erase(phi);
}

/// Safe block insertion helper: create a new block after `after` in its function.
pub fn new_block_after(after: BB, name: &str) -> BB {
    let ctx = value_context(block_function(after));
    let n = cstr(name);
    let next = unsafe { LLVMGetNextBasicBlock(after) };
    if next.is_null() {
        unsafe { LLVMAppendBasicBlockInContext(ctx, block_function(after), n.as_ptr()) }
    } else {
        unsafe { LLVMInsertBasicBlockInContext(ctx, next, n.as_ptr()) }
    }
}

pub fn i1_const(ctx: LLVMContextRef, v: bool) -> V {
    unsafe { LLVMConstInt(LLVMInt1TypeInContext(ctx), v as u64, 0) }
}

pub fn ptr_ty(ctx: LLVMContextRef) -> Ty {
    unsafe { LLVMPointerTypeInContext(ctx, 0) }
}

/// Module data layout handle (borrowed).
pub fn data_layout(m: LLVMModuleRef) -> llvm_sys::target::LLVMTargetDataRef {
    unsafe { llvm_sys::target::LLVMGetModuleDataLayout(m) }
}

pub fn pointer_size(m: LLVMModuleRef) -> u64 {
    unsafe { llvm_sys::target::LLVMPointerSize(data_layout(m)) as u64 }
}

pub fn pointer_abi_align(m: LLVMModuleRef) -> u64 {
    unsafe { llvm_sys::target::LLVMABIAlignmentOfType(data_layout(m), ptr_ty(module_context(m))) as u64 }
}

pub fn module_of_inst(i: V) -> LLVMModuleRef {
    global_module(inst_function(i))
}
