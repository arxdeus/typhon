//! `writeToPythonExtension`: CPython type objects and the `PyInit_<name>` hook.

use llvm_sys::LLVMIntPredicate::LLVMIntSLT;
use llvm_sys::LLVMLinkage::*;
use rustc_hash::FxHashMap;
use typhon_common::Result;
use typhon_ir::module::Module;
use typhon_ir::pyext::PyModule;
use typhon_ir::types::TypeKind;

use super::pyext::PYTHON_ABI_VERSION;
use super::ll::{self, Builder, V};
use super::{LlvmVisitor, MAIN_UNCLASH};

impl<'ctx> LlvmVisitor<'ctx> {
    /// `writeToPythonExtension`.
    pub fn write_to_python_extension(&mut self, m: &Module, pymod: &PyModule, filename: &str) -> Result<()> {
        let t = self.py_types();
        let c = self.c;
        let (i64t, i32t, p) = (self.b.i64(), self.b.i32(), self.b.ptr());
        let null = self.b.null_ptr();
        let zero64 = self.b.c_i64(0);
        let zero32 = self.b.c_i32(0);
        let mm = self.m();
        let py_type_type = ll::new_global(mm, p, false, LLVMExternalLinkage, None, "PyType_Type");
        let import_data = self.is_win_msvc();
        if import_data {
            ll::set_dll_storage(py_type_type, llvm_sys::LLVMDLLStorageClass::LLVMDLLImportStorageClass);
        }
        let (alloc_ty, alloc_unc) = self.make_alloc_func(false, true);
        let (free_ty, free) = self.make_free_func();
        let _ = free_ty;

        // module definition
        let obj_const = ll::const_named_struct(t.object, &[self.b.c_i64(1), null]);
        let base_const = ll::const_named_struct(t.module_def_base, &[obj_const, null, zero64, null]);
        let name = self.py_string(&pymod.name);
        let doc = self.py_string(&pymod.doc);
        let funcs = self.py_functions(m, &t, &pymod.functions);
        let module_def = ll::const_named_struct(t.module_def, &[base_const, name, doc, self.b.c_i64(-1), funcs, null, null, null, null]);
        let module_var = ll::new_global(mm, t.module_def, false, LLVMPrivateLinkage, Some(module_def), ".pyext_module");

        let mut type_vars: FxHashMap<usize, V> = FxHashMap::default();
        for (ti, pt) in pymod.types.iter().enumerate() {
            let f = |s: &mut Self, x| s.py_func_wrap(m, x, true);
            let number: Vec<V> = vec![
                f(self, pt.add), f(self, pt.sub), f(self, pt.mul), f(self, pt.mod_), f(self, pt.divmod), f(self, pt.pow),
                f(self, pt.neg), f(self, pt.pos), f(self, pt.abs), f(self, pt.bool_), f(self, pt.invert), f(self, pt.lshift),
                f(self, pt.rshift), f(self, pt.and_), f(self, pt.xor_), f(self, pt.or_), f(self, pt.int_), null,
                f(self, pt.float_), f(self, pt.iadd), f(self, pt.isub), f(self, pt.imul), f(self, pt.imod), f(self, pt.ipow),
                f(self, pt.ilshift), f(self, pt.irshift), f(self, pt.iand), f(self, pt.ixor), f(self, pt.ior),
                f(self, pt.floordiv), f(self, pt.truediv), f(self, pt.ifloordiv), f(self, pt.itruediv), f(self, pt.index),
                f(self, pt.matmul), f(self, pt.imatmul),
            ];
            let len = f(self, pt.len);
            let contains = f(self, pt.contains);
            let sequence = vec![len, null, null, null, null, null, null, contains, null, null];
            let getitem = f(self, pt.getitem);
            let setitem = f(self, pt.setitem);
            let mapping = vec![null, getitem, setitem];
            let num_c = self.slots_global(t.number, &number, format!(".pyext_number_slots.{}", pt.name));
            let seq_c = self.slots_global(t.sequence, &sequence, format!(".pyext_sequence_slots.{}", pt.name));
            let map_c = self.slots_global(t.mapping, &mapping, format!(".pyext_mapping_slots.{}", pt.name));

            let ty = pt.ty.expect("python extension type without IR type");
            let aty = m.actual_type(ty);
            let ref_contents = match m.type_kind(aty) {
                TypeKind::Ref { contents, polymorphic } => {
                    assert!(!polymorphic, "Python extension types cannot be polymorphic");
                    Some(*contents)
                }
                _ => None,
            };
            let llvm_type = self.get_llvm_type(m, ty);
            let object_type = ll::struct_ty(c, &[t.object, llvm_type]);
            let codon_size = match ref_contents {
                Some(ct) => {
                    let l = self.get_llvm_type(m, ct);
                    ll::alloc_size(mm, l)
                }
                None => 0,
            };
            let py_size = ll::alloc_size(mm, object_type);

            // allocator
            let alloc = ll::get_or_insert_function(mm, &format!("{}.py_alloc", pt.name), ll::fn_ty(p, &[p, i64t], false));
            let b = Builder::new(c);
            b.at_end(ll::append_block(c, alloc, "entry"));
            let py_obj = b.call(alloc_ty, alloc_unc, &[b.c_i64(py_size as i64)]);
            let header = b.insert_value(ll::const_named_struct(t.object, &[b.c_i64(1), null]), ll::param(alloc, 0), 1);
            b.store(header, py_obj);
            if let Some(ct) = ref_contents {
                let (at, af) = self.make_alloc_func(m.type_is_atomic(ct), false);
                let codon_obj = b.call(at, af, &[b.c_i64(codon_size as i64)]);
                b.store(codon_obj, b.gep(object_type, py_obj, &[zero64, b.c_i32(1)]));
            }
            b.ret(py_obj);

            // deallocator
            let del_fn = self.py_func_wrap(m, pt.del, false);
            let dealloc_ty = ll::fn_ty(b.void(), &[p], false);
            let dealloc = ll::get_or_insert_function(mm, &format!("{}.py_dealloc", pt.name), dealloc_ty);
            b.at_end(ll::append_block(c, dealloc, "entry"));
            if del_fn != null {
                b.call(dealloc_ty, del_fn, &[ll::param(dealloc, 0)]);
            }
            b.call(ll::function_type(free), free, &[ll::param(dealloc, 0)]);
            b.ret_void();

            let py_new = ll::get_or_insert_function(mm, "PyType_GenericNew", ll::fn_ty(p, &[p, p, p], false));
            let head = ll::const_named_struct(
                t.var_object,
                &[ll::const_named_struct(t.object, &[self.b.c_i64(1), if import_data { null } else { py_type_type }]), zero64],
            );
            let tp_name = self.py_string(&format!("{}.{}", pymod.name, pt.name));
            let repr = f(self, pt.repr);
            let hash = f(self, pt.hash);
            let call = f(self, pt.call);
            let str_ = f(self, pt.str);
            let tp_doc = self.py_string(&pt.doc);
            let cmp = f(self, pt.cmp);
            let iter = f(self, pt.iter);
            let iternext = f(self, pt.iternext);
            let methods = self.py_functions(m, &t, &pt.methods);
            let members = self.py_members(&t, &pt.members, object_type);
            let getset = self.py_getset(m, &t, &pt.getset);
            let init = f(self, pt.init);
            let slots = [
                head, tp_name, self.b.c_i64(py_size as i64), zero64, dealloc, zero64, null, null, null, repr, num_c, seq_c,
                map_c, hash, call, str_, null, null, null, zero64, tp_doc, null, null, cmp, zero64, iter, iternext, methods,
                members, getset, null, null, null, null, zero64, init, alloc, py_new, free, null, null, null, null, null,
                null, null, zero32, free, null, self.b.c_i8(0),
            ];
            let type_var = ll::new_global(
                mm,
                t.type_object,
                false,
                LLVMPrivateLinkage,
                Some(ll::const_named_struct(t.type_object, &slots)),
                &format!(".pyext_type.{}", pt.name),
            );
            if pt.type_ptr_hook.is_some() {
                let hook = self.py_func_wrap(m, pt.type_ptr_hook, false);
                for bb in ll::blocks(hook) {
                    for inst in ll::instructions(bb) {
                        if ll::is_return(inst) && ll::num_operands(inst) > 0 {
                            ll::set_operand(inst, 0, type_var);
                        }
                    }
                }
            }
            type_vars.insert(ti, type_var);
        }

        // initialization hook
        let void = self.b.void();
        let (inc_ty, inc) = self.decl("Py_IncRef", void, &[p]);
        let (dec_ty, dec) = self.decl("Py_DecRef", void, &[p]);
        let (create_ty, create) = self.decl("PyModule_Create2", p, &[p, i32t]);
        let (ready_ty, ready) = self.decl("PyType_Ready", i32t, &[p]);
        let (add_ty, add) = self.decl("PyModule_AddObject", i32t, &[p, p, p]);
        for f in [inc, dec, create, ready, add] {
            ll::add_fn_attr(f, "nounwind");
        }
        let init = ll::get_or_insert_function(mm, &format!("PyInit_{}", pymod.name), ll::fn_ty(p, &[], false));
        let b = Builder::new(c);
        b.at_end(ll::append_block(c, init, "entry"));
        if let Some(main) = ll::get_function(mm, "main") {
            ll::set_name(main, MAIN_UNCLASH);
            b.call(ll::function_type(main), main, &[zero32, null]);
        }
        // Imported DLL data addresses cannot be resolved in static initializers.
        if import_data {
            for tv in type_vars.values() {
                b.store(py_type_type, b.struct_gep(t.object, *tv, 1));
            }
        }
        // base types (30 = tp_base)
        for (ti, pt) in pymod.types.iter().enumerate() {
            if let Some(base) = pt.base {
                let (sub, bv) = (type_vars[&ti], type_vars[&base]);
                b.store(bv, b.inbounds_gep(t.type_object, sub, &[b.c_i64(0), b.c_i32(30)]));
            }
        }
        let fail_ret = |_b: &Builder| {
            let fail = ll::append_block(c, init, "failure");
            let ok = ll::append_block(c, init, "success");
            (fail, ok)
        };
        for ti in 0..pymod.types.len() {
            let tv = type_vars[&ti];
            let status = b.call(ready_ty, ready, &[tv]);
            let (fail, ok) = fail_ret(&b);
            b.cond_br(b.icmp(LLVMIntSLT, status, zero32), fail, ok);
            b.at_end(fail);
            b.ret(null);
            b.at_end(ok);
        }
        let module = b.call(create_ty, create, &[module_var, b.c_i32(PYTHON_ABI_VERSION)]);
        let (fail, ok) = fail_ret(&b);
        b.cond_br(b.icmp_eq(module, null), fail, ok);
        b.at_end(fail);
        b.ret(null);
        b.at_end(ok);
        for (ti, pt) in pymod.types.iter().enumerate() {
            let tv = type_vars[&ti];
            b.call(inc_ty, inc, &[tv]);
            let nm = self.py_string(&pt.name);
            let status = b.call(add_ty, add, &[module, nm, tv]);
            let (fail, ok) = fail_ret(&b);
            b.cond_br(b.icmp(LLVMIntSLT, status, zero32), fail, ok);
            b.at_end(fail);
            b.call(dec_ty, dec, &[tv]);
            b.call(dec_ty, dec, &[module]);
            b.ret(null);
            b.at_end(ok);
        }
        b.ret(module);
        self.write_to_object_file(filename, false, false)
    }
}
