//! `writeToPythonExtension` / `createPyTryCatchWrapper` (for `codon build -pyext`).

use llvm_sys::LLVMLinkage::*;
use typhon_ir::ids::FuncId;
use typhon_ir::module::Module;
use typhon_ir::pyext::{PyFunction, PyFunctionType, PyGetSet, PyMember};

use super::ll::{self, Builder, Ty, V};
use super::{LlvmVisitor, seq_exc_offset};

// CPython constants (methodobject.h, modsupport.h, descrobject.h).
const METH_VARARGS: i64 = 0x0001;
const METH_KEYWORDS: i64 = 0x0002;
const METH_NOARGS: i64 = 0x0004;
const METH_O: i64 = 0x0008;
const METH_CLASS: i64 = 0x0010;
const METH_STATIC: i64 = 0x0020;
const METH_COEXIST: i64 = 0x0040;
const METH_FASTCALL: i64 = 0x0080;
pub(super) const PYTHON_ABI_VERSION: i64 = 1013;
const PY_READONLY: i64 = 1;

pub(super) struct PyTypes {
    pub method_def: Ty,
    pub object: Ty,
    pub var_object: Ty,
    pub module_def_base: Ty,
    pub module_def: Ty,
    pub member_def: Ty,
    pub getset_def: Ty,
    pub number: Ty,
    pub sequence: Ty,
    pub mapping: Ty,
    pub type_object: Ty,
}

impl<'ctx> LlvmVisitor<'ctx> {
    /// `createPyTryCatchWrapper`: invoke `func`, convert Codon exceptions into Python ones.
    pub(super) fn create_py_try_catch_wrapper(&mut self, m: &Module, func: V) -> V {
        let fty = ll::function_type(func);
        let wrap = ll::get_or_insert_function(self.m(), &format!("{}.tc_wrap", ll::name(func)), fty);
        let (_, pers) = self.make_personality_func();
        ll::set_personality(wrap, pers);
        let c = self.c;
        let entry = ll::append_block(c, wrap, "entry");
        let normal = ll::append_block(c, wrap, "normal");
        let unwind = ll::append_block(c, wrap, "unwind");
        let b = Builder::new(c);
        b.at_end(entry);
        let win = self.is_win_msvc();
        let storage = if win { b.alloca(b.ptr()) } else { std::ptr::null_mut() };
        let result = b.invoke(fty, func, &ll::params(wrap), normal, unwind);
        b.at_end(normal);
        b.ret(result);
        b.at_end(unwind);
        let unwind_exc = if win {
            let cs = b.catch_switch(ll::token_none(c), None, 1);
            let catch_block = ll::append_block(c, wrap, "catch");
            let caught = ll::append_block(c, wrap, "caught");
            Builder::add_handler(cs, catch_block);
            b.at_end(catch_block);
            let (_, filter) = self.make_win_eh_filter();
            let cp = b.catch_pad(cs, &[filter]);
            let (et, ef) = self.exc_current_func();
            b.store(b.call_funclet(et, ef, &[], cp), storage);
            b.catch_ret(cp, caught);
            b.at_end(caught);
            b.load(b.ptr(), storage)
        } else {
            let lp = b.landing_pad(self.get_pad_type(), 1);
            Builder::set_cleanup(lp, true);
            Builder::add_clause(lp, self.get_type_idx_var(m, None));
            b.extract_value(lp, 0)
        };
        let exc_val = b.const_gep_i8(unwind_exc, seq_exc_offset(ll::pointer_size(self.m())));
        let loaded = b.load(b.ptr(), exc_val);
        let (st, sf) = self.decl("seq_set_python_exception", b.void(), &[b.ptr()]);
        b.call(st, sf, &[loaded]);
        let ret = ll::fn_ret_ty(fty);
        if ret == b.i32() {
            b.ret(b.c_i32(-1));
        } else {
            b.ret(ll::const_null(ret));
        }
        wrap
    }

    pub(super) fn py_types(&self) -> PyTypes {
        let c = self.c;
        let (i64t, i32t, i8t, p) = (self.b.i64(), self.b.i32(), self.b.i8(), self.b.ptr());
        let method_def = ll::named_struct_ty(c, "PyMethodDef", &[p, p, i32t, p]);
        let object = ll::named_struct_ty(c, "PyObject", &[i64t, p]);
        let var_object = ll::named_struct_ty(c, "PyVarObject", &[object, i64t]);
        let module_def_base = ll::named_struct_ty(c, "PyMethodDefBase", &[object, p, i64t, p]);
        let module_def = ll::named_struct_ty(c, "PyModuleDef", &[module_def_base, p, p, i64t, p, p, p, p, p]);
        let member_def = ll::named_struct_ty(c, "PyMemberDef", &[p, i32t, i64t, i32t, p]);
        let getset_def = ll::named_struct_ty(c, "PyGetSetDef", &[p, p, p, p, p]);
        let number = ll::named_struct_ty(c, "PyNumberMethods", &[p; 36]);
        let sequence = ll::named_struct_ty(c, "PySequenceMethods", &[p; 10]);
        let mapping = ll::named_struct_ty(c, "PyMappingMethods", &[p; 3]);
        let _async = ll::named_struct_ty(c, "PyAsyncMethods", &[p; 4]);
        let _buffer = ll::named_struct_ty(c, "PyBufferProcs", &[p, p]);
        let to = [
            var_object, p, i64t, i64t, p, i64t, p, p, p, p, p, p, p, p, p, p, p, p, p, i64t, p, p, p, p, i64t, p, p, p, p, p, p, p, p,
            p, i64t, p, p, p, p, p, p, p, p, p, p, p, i32t, p, p, i8t,
        ];
        let type_object = ll::named_struct_ty(c, "PyTypeObject", &to);
        PyTypes { method_def, object, var_object, module_def_base, module_def, member_def, getset_def, number, sequence, mapping, type_object }
    }

    pub(super) fn py_string(&self, s: &str) -> V {
        if s.is_empty() {
            return self.b.null_ptr();
        }
        let mut bytes: Vec<V> = s.bytes().map(|x| self.b.c_i8(x as i64)).collect();
        bytes.push(self.b.c_i8(0));
        let at = ll::array_ty(self.b.i8(), bytes.len() as u64);
        let g = ll::new_global(self.m(), at, true, LLVMPrivateLinkage, Some(ll::const_array(self.b.i8(), &bytes)), ".pyext_str");
        ll::set_unnamed_addr(g, llvm_sys::LLVMUnnamedAddr::LLVMGlobalUnnamedAddr);
        g
    }

    pub(super) fn py_func_wrap(&mut self, m: &Module, f: Option<FuncId>, wrap: bool) -> V {
        let Some(f) = f else { return self.b.null_ptr() };
        let name = self.get_llvm_name_for_function(m, f);
        let func = ll::get_function(self.m(), &name).unwrap_or_else(|| panic!("function {name} not found in LLVM module"));
        if wrap { self.create_py_try_catch_wrapper(m, func) } else { func }
    }

    pub(super) fn py_functions(&mut self, m: &Module, t: &PyTypes, functions: &[PyFunction]) -> V {
        if functions.is_empty() {
            return self.b.null_ptr();
        }
        let mut rows = Vec::new();
        for pf in functions {
            let mut flag = if pf.keywords {
                METH_FASTCALL | METH_KEYWORDS
            } else {
                match pf.nargs {
                    0 => METH_NOARGS,
                    1 => METH_O,
                    _ => METH_FASTCALL,
                }
            };
            let _ = METH_VARARGS;
            match pf.ty {
                PyFunctionType::Class => flag |= METH_CLASS,
                PyFunctionType::Static => flag |= METH_STATIC,
                _ => {}
            }
            if pf.coexist {
                flag |= METH_COEXIST;
            }
            let name = self.py_string(&pf.name);
            let func = self.py_func_wrap(m, pf.func, true);
            let doc = self.py_string(&pf.doc);
            rows.push(ll::const_named_struct(t.method_def, &[name, func, self.b.c_i32(flag), doc]));
        }
        let null = self.b.null_ptr();
        rows.push(ll::const_named_struct(t.method_def, &[null, null, self.b.c_i32(0), null]));
        let at = ll::array_ty(t.method_def, rows.len() as u64);
        ll::new_global(self.m(), at, false, LLVMPrivateLinkage, Some(ll::const_array(t.method_def, &rows)), ".pyext_methods")
    }

    pub(super) fn py_members(&mut self, t: &PyTypes, members: &[PyMember], object_type: Ty) -> V {
        if members.is_empty() {
            return self.b.null_ptr();
        }
        let null = self.b.null_ptr();
        let mut rows = Vec::new();
        for mb in members {
            // offset of {0, 1, indexes...} in `object_type`, as a constant GEP on null
            let mut idx = vec![self.b.c_i64(0), self.b.c_i32(1)];
            idx.extend(mb.indexes.iter().map(|&i| self.b.c_i32(i as i64)));
            let offset = unsafe {
                let gep = llvm_sys::core::LLVMConstGEP2(object_type, null, idx.as_mut_ptr(), idx.len() as u32);
                llvm_sys::core::LLVMConstPtrToInt(gep, self.b.i64())
            };
            let name = self.py_string(&mb.name);
            let doc = self.py_string(&mb.doc);
            let ro = self.b.c_i32(if mb.readonly { PY_READONLY } else { 0 });
            rows.push(ll::const_named_struct(t.member_def, &[name, self.b.c_i32(mb.ty as i64), offset, ro, doc]));
        }
        rows.push(ll::const_named_struct(t.member_def, &[null, self.b.c_i32(0), self.b.c_i64(0), self.b.c_i32(0), null]));
        let at = ll::array_ty(t.member_def, rows.len() as u64);
        ll::new_global(self.m(), at, false, LLVMPrivateLinkage, Some(ll::const_array(t.member_def, &rows)), ".pyext_members")
    }

    pub(super) fn py_getset(&mut self, m: &Module, t: &PyTypes, gs: &[PyGetSet]) -> V {
        if gs.is_empty() {
            return self.b.null_ptr();
        }
        let null = self.b.null_ptr();
        let mut rows = Vec::new();
        for g in gs {
            let name = self.py_string(&g.name);
            let get = self.py_func_wrap(m, g.get, true);
            let set = self.py_func_wrap(m, g.set, true);
            let doc = self.py_string(&g.doc);
            rows.push(ll::const_named_struct(t.getset_def, &[name, get, set, doc, null]));
        }
        rows.push(ll::const_named_struct(t.getset_def, &[null, null, null, null, null]));
        let at = ll::array_ty(t.getset_def, rows.len() as u64);
        ll::new_global(self.m(), at, false, LLVMPrivateLinkage, Some(ll::const_array(t.getset_def, &rows)), ".pyext_getset")
    }

    pub(super) fn slots_global(&self, ty: Ty, slots: &[V], name: String) -> V {
        let null = self.b.null_ptr();
        if slots.iter().all(|s| *s == null) {
            return null;
        }
        ll::new_global(self.m(), ty, false, LLVMPrivateLinkage, Some(ll::const_named_struct(ty, slots)), &name)
    }
}

