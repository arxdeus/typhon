//! Win64 ABI helpers (port of the `winABI*` functions and Windows export wrappers in
//! llvisitor.cpp). Only exercised for `*-windows-msvc` targets.

use std::ptr::null_mut;

use llvm_sys::prelude::LLVMModuleRef;

use super::LlvmVisitor;
use super::ll::{self, Builder, Ty, V};
use typhon_ir::ids::FuncId;
use typhon_ir::module::Module;
use typhon_ir::util::irtools;

/// `WinABIKind`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    Direct,
    Empty,
    Integer,
    Bitcast,
    Indirect,
}

/// `winABIKind`.
pub fn kind(m: LLVMModuleRef, t: Ty, result: bool) -> Kind {
    if ll::is_void(t) {
        return Kind::Empty;
    }
    if ll::is_integer(t) && ll::int_width(t) > 64 {
        if result && ll::int_width(t) == 128 {
            return Kind::Bitcast; // Clang/GCC return __int128 in XMM0.
        }
        return Kind::Indirect;
    }
    if !ll::is_aggregate(t) {
        return Kind::Direct;
    }
    match ll::alloc_size(m, t) {
        0 => Kind::Empty,
        1 | 2 | 4 | 8 => Kind::Integer,
        _ => Kind::Indirect,
    }
}

/// `winABIType`.
pub fn abi_type(m: LLVMModuleRef, t: Ty, result: bool) -> Ty {
    let c = unsafe { llvm_sys::core::LLVMGetModuleContext(m) };
    match kind(m, t, result) {
        Kind::Empty => ll::void_ty(c),
        Kind::Integer => ll::int_ty(c, (ll::alloc_size(m, t) * 8) as u32),
        Kind::Bitcast => ll::vector_ty(ll::int_ty(c, 64), 2),
        Kind::Indirect => ll::ptr_ty(c),
        Kind::Direct => t,
    }
}

/// `winABISlot`: a fixed, 16-byte aligned frame slot in the entry block.
fn slot(b: &Builder, m: LLVMModuleRef, t: Ty) -> V {
    let block = b.insert_block();
    let func = unsafe { llvm_sys::core::LLVMGetBasicBlockParent(block) };
    let entry = ll::entry_block(func);
    let alloc = Builder::new(b.c);
    match ll::instructions(entry).first() {
        Some(&first) => alloc.before(first),
        None => alloc.at_end(entry),
    }
    let s = alloc.alloca(t);
    let align = ll::abi_align(m, t).max(16);
    unsafe { llvm_sys::core::LLVMSetAlignment(s, align as u32) };
    s
}

/// `winABIEncode`.
fn encode(m: LLVMModuleRef, b: &Builder, v: V, result: bool) -> Option<V> {
    let t = ll::type_of(v);
    match kind(m, t, result) {
        Kind::Empty => None,
        Kind::Bitcast => Some(b.bitcast(v, abi_type(m, t, result))),
        Kind::Integer => {
            let s = slot(b, m, t);
            b.store(v, s);
            Some(b.load(abi_type(m, t, false), s))
        }
        Kind::Indirect => {
            let s = slot(b, m, t);
            b.store(v, s);
            Some(s)
        }
        Kind::Direct => Some(v),
    }
}

/// `winABIDecode`.
fn decode(m: LLVMModuleRef, b: &Builder, t: Ty, v: V, result: bool) -> Option<V> {
    match kind(m, t, result) {
        Kind::Empty => (!ll::is_void(t)).then(|| ll::const_null(t)),
        Kind::Bitcast => Some(b.bitcast(v, t)),
        Kind::Integer => {
            let s = slot(b, m, t);
            b.store(v, s);
            Some(b.load(t, s))
        }
        Kind::Indirect => Some(b.load(t, v)),
        Kind::Direct => Some(v),
    }
}

/// `winABISignature`.
pub fn signature(m: LLVMModuleRef, sig: Ty) -> Ty {
    let c = unsafe { llvm_sys::core::LLVMGetModuleContext(m) };
    let result = ll::fn_ret_ty(sig);
    let indirect = kind(m, result, true) == Kind::Indirect;
    let mut args = Vec::new();
    if indirect {
        args.push(ll::ptr_ty(c));
    }
    for t in ll::fn_param_tys(sig) {
        if kind(m, t, false) != Kind::Empty {
            args.push(abi_type(m, t, false));
        }
    }
    let ret = if indirect { ll::void_ty(c) } else { abi_type(m, result, true) };
    ll::fn_ty(ret, &args, ll::fn_is_vararg(sig))
}

/// First half of `winABICall`: native signature and encoded arguments.
pub struct Prepared {
    pub sig: Ty,
    pub args: Vec<V>,
    result_type: Ty,
    return_slot: V,
}

pub fn prepare(m: LLVMModuleRef, b: &Builder, sig: Ty, arguments: &[V]) -> Prepared {
    let result_type = ll::fn_ret_ty(sig);
    let indirect = kind(m, result_type, true) == Kind::Indirect;
    let mut values = Vec::new();
    let mut return_slot = null_mut();
    if indirect {
        return_slot = slot(b, m, result_type);
        values.push(return_slot);
    }
    for &a in arguments {
        if let Some(v) = encode(m, b, a, false) {
            values.push(v);
        }
    }
    Prepared { sig: signature(m, sig), args: values, result_type, return_slot }
}

/// Second half of `winABICall`: decode the emitted call's result.
pub fn finish(m: LLVMModuleRef, b: &Builder, p: &Prepared, call: V) -> Option<V> {
    if !p.return_slot.is_null() {
        ll::add_call_type_attr(call, 1, "sret", p.result_type);
        return decode(m, b, p.result_type, p.return_slot, true);
    }
    decode(m, b, p.result_type, call, true)
}

/// `winABICall` with a plain builder call.
pub fn call(m: LLVMModuleRef, b: &Builder, callee: V, sig: Ty, arguments: &[V]) -> Option<V> {
    let p = prepare(m, b, sig, arguments);
    let c = b.call(p.sig, callee, &p.args);
    finish(m, b, &p, c)
}

impl<'ctx> LlvmVisitor<'ctx> {
    /// `needsWinExportWrapper`.
    pub(super) fn needs_win_export_wrapper(&self, m: &Module, f: FuncId) -> bool {
        self.module.is_some()
            && self.is_win_msvc()
            && irtools::is_export(m, f)
            && !m.func(f).unwrap().is_external()
            && !irtools::is_gpu_kernel(m, f)
    }

    /// `makeWinExportWrapper`.
    pub(super) fn make_win_export_wrapper(&mut self, name: &str, body: V, exported: bool) -> V {
        let mm = self.m();
        let sig = ll::function_type(body);
        let wrapper = ll::get_or_insert_function(mm, name, signature(mm, sig));
        if exported && !self.options.jit {
            ll::set_dll_storage(wrapper, llvm_sys::LLVMDLLStorageClass::LLVMDLLExportStorageClass);
        }
        if ll::has_blocks(wrapper) {
            return wrapper;
        }
        ll::add_string_attr(wrapper, ll::FN_INDEX, "typhon-abi-body", &ll::name(body));
        let b = Builder::new(self.c);
        b.at_end(ll::append_block(self.c, wrapper, "entry"));
        let result_type = ll::fn_ret_ty(sig);
        let indirect = kind(mm, result_type, true) == Kind::Indirect;
        let mut params = ll::params(wrapper).into_iter();
        let return_slot = if indirect { params.next() } else { None };
        if indirect {
            ll::add_type_attr(wrapper, 1, "sret", result_type);
        }
        let mut values = Vec::new();
        for t in ll::fn_param_tys(sig) {
            let native = if kind(mm, t, false) == Kind::Empty { null_mut() } else { params.next().unwrap() };
            values.push(decode(mm, &b, t, native, false).unwrap_or_else(|| ll::const_null(t)));
        }
        let result = b.call(sig, body, &values);
        if let Some(rs) = return_slot {
            b.store(result, rs);
            b.ret_void();
        } else if kind(mm, result_type, true) == Kind::Empty {
            b.ret_void();
        } else {
            let r = encode(mm, &b, result, true).unwrap();
            b.ret(r);
        }
        wrapper
    }

    /// `getFunctionValue`.
    pub(super) fn get_function_value(&mut self, m: &Module, f: FuncId) -> V {
        let body = self.get_func(m, f);
        if body.is_null() || !self.is_win_msvc() || m.func(f).unwrap().is_external() || irtools::is_gpu_kernel(m, f) {
            return body;
        }
        let exported = self.needs_win_export_wrapper(m, f);
        let mm = self.m();
        if !exported && signature(mm, ll::function_type(body)) == ll::function_type(body) {
            return body;
        }
        // An earlier JIT cell already emitted an exported function's native wrapper.
        if self.options.jit && exported && ll::is_declaration(body) {
            return ll::get_or_insert_function(mm, &Self::get_name_for_function(m, f), signature(mm, ll::function_type(body)));
        }
        if let Some(&w) = self.win_callbacks.get(&(body as usize)) {
            return w;
        }
        let wrapper = self.make_win_export_wrapper(&format!("{}.c_callback", ll::name(body)), body, false);
        // JIT cells can take the same function's address repeatedly.
        ll::set_linkage(
            wrapper,
            if self.options.jit { llvm_sys::LLVMLinkage::LLVMWeakODRLinkage } else { llvm_sys::LLVMLinkage::LLVMPrivateLinkage },
        );
        self.win_callbacks.insert(body as usize, wrapper);
        wrapper
    }

    /// `prepareWinExports`.
    pub(super) fn prepare_win_exports(&mut self) {
        if self.module.is_none() || !self.is_win_msvc() {
            return;
        }
        let mut exports = Vec::new();
        for f in ll::functions(self.m()) {
            if ll::has_string_attr(f, "typhon-c-export") && !ll::is_declaration(f) {
                exports.push((f, ll::string_attr_value(f, "typhon-c-export").unwrap_or_default()));
            }
        }
        for (body, name) in exports {
            ll::remove_string_attr(body, "typhon-c-export");
            ll::set_name(body, &format!("{name}.codon"));
            ll::set_linkage(
                body,
                if self.options.jit { llvm_sys::LLVMLinkage::LLVMExternalLinkage } else { llvm_sys::LLVMLinkage::LLVMPrivateLinkage },
            );
            ll::set_dll_storage(body, llvm_sys::LLVMDLLStorageClass::LLVMDefaultStorageClass);
            if let Some(&wrapper) = self.win_callbacks.get(&(body as usize)) {
                ll::set_name(wrapper, &name);
                ll::set_linkage(wrapper, llvm_sys::LLVMLinkage::LLVMExternalLinkage);
                ll::add_string_attr(wrapper, ll::FN_INDEX, "typhon-abi-body", &ll::name(body));
                if !self.options.jit {
                    ll::set_dll_storage(wrapper, llvm_sys::LLVMDLLStorageClass::LLVMDLLExportStorageClass);
                }
            } else {
                self.make_win_export_wrapper(&name, body, true);
            }
        }
    }
}
