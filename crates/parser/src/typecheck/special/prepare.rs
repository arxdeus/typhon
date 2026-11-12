//! `prepareVTables` (fixpoint driver and value-method installation).

use std::collections::BTreeSet;

use typhon_common::mangle::{self, stdlib};
use typhon_common::Result;

use super::super::{CtxRef, TypecheckVisitor};
use super::mm;
use crate::ast::CallArg;
use crate::cache::AstType;
use crate::ctx::BindingCheckpoint;
use crate::ids::TypeId;
use crate::typecheck::ctx::SavedInferenceState;

impl<'a> TypecheckVisitor<'a> {
    /// `getFunction(name)` realizations exist.
    fn sp_has_realizations(&self, name: &str) -> bool {
        self.cache.functions.get(name).is_some_and(|f| !f.realizations.is_empty())
    }

    /// Sorted realization types of a function (`sorted_view(getFunction(name)->realizations)`).
    fn sp_sorted_realization_types(&self, name: &str) -> Vec<TypeId> {
        let Some(f) = self.cache.functions.get(name) else { return vec![] };
        let mut v: Vec<(String, TypeId)> =
            f.realizations.iter().filter_map(|(k, r)| r.borrow().ty.map(|t| (k.clone(), t))).collect();
        v.sort_by(|a, b| a.0.cmp(&b.0));
        v.into_iter().map(|x| x.1).collect()
    }

    /// `install(member, count, fallback)` of `realize_value_methods`.
    fn sp_install_value_method(
        &mut self,
        ir_type: typhon_ir::TypeId,
        receiver: TypeId,
        member: &str,
        count: usize,
        fallback: &str,
    ) -> Result<bool> {
        let types = self.types();
        let module = self.cache.module.clone().expect("no IR module");
        let id = self.value_method_id(member);
        if module.borrow().runtime_info(ir_type).is_some_and(|i| i.methods.contains_key(&id)) {
            return Ok(false);
        }
        let any = self.get_stdlib_type(&mangle::class("", "Any"))?;
        let pointer = self.get_class_field_types(types.get_class(any).unwrap())?[0];
        let mut arguments = vec![];
        for _ in 0..count {
            let v = self.sp_typed(pointer);
            arguments.push(CallArg::new("", Some(v)));
        }
        let rn = types.realized_name(receiver);
        let r = self.sp_id(&rn);
        arguments.push(CallArg::new("", Some(r)));

        let old_context = self.ctx.clone();
        let old_level = typhon_common::logger().level;
        let old_errors = self.cache.errors.clone();
        // TODO: generalize this to be able to typecheck any fn/code blocks with error
        // recovery
        let mut contexts: Vec<CtxRef> = vec![];
        for imported in self.cache.imports.values() {
            if let Some(c) = &imported.ctx
                && !contexts.iter().any(|x| std::rc::Rc::ptr_eq(x, c))
            {
                contexts.push(c.clone());
            }
        }
        for c in &contexts {
            c.borrow_mut().base.push_checkpoint(true);
        }
        self.ctx.borrow_mut().prepare_checkpoint();
        let mut incomplete: BTreeSet<(String, String)> = BTreeSet::new();
        for (name, function) in &self.cache.functions {
            for (key, realization) in &function.realizations {
                if realization.borrow().ir.is_none() {
                    incomplete.insert((name.clone(), key.clone()));
                }
            }
        }
        let f = self.sp_id(&mm("TypeInfo", member));
        let c = self.sp_call_args(f, arguments.clone());
        let res = self.tr(c);
        let close = |contexts: &[CtxRef], rollback: bool| {
            for active in contexts {
                let cp: Option<BindingCheckpoint<_>> = active.borrow_mut().base.pop_checkpoint();
                let Some(mut cp) = cp else { continue };
                if rollback && let Some(state) = cp.state.take() {
                    let mut a = active.borrow_mut();
                    a.base.rollback(&cp, |item| item.borrow().is_global());
                    if let Ok(saved) = state.downcast::<SavedInferenceState>() {
                        a.restore_inference_state(*saved);
                    }
                }
            }
        };
        let call = match res {
            Ok(c) => {
                close(&contexts, false);
                c
            }
            Err(e) if e.parser_errors().is_some() => {
                self.ctx = old_context;
                typhon_common::logger().level = old_level;
                self.cache.errors = old_errors;
                close(&contexts, true);
                for (name, function) in self.cache.functions.iter_mut() {
                    function.realizations.retain(|key, r| r.borrow().ir.is_some() || incomplete.contains(&(name.clone(), key.clone())));
                }
                if fallback.is_empty() {
                    return Err(e);
                }
                let f = self.sp_id(&mm("TypeInfo", fallback));
                let c = self.sp_call_args(f, arguments);
                self.tr(c)?
            }
            Err(e) => {
                close(&contexts, false);
                return Err(e);
            }
        };
        let callee = self.cache.ast[call].kind.as_call().and_then(|c| c.expr).expect("value method call was replaced");
        let resolved = types.get_func(self.ety(callee)).expect("value method is not a function");
        let rrn = types.realized_name(resolved);
        let fir = self
            .get_function_type(resolved)
            .and_then(|f| f.realizations.get(&rrn).cloned())
            .and_then(|r| r.borrow().ir)
            .expect("value method was not realized");
        let mut m = module.borrow_mut();
        m.ty_mut(ir_type).runtime_info.get_or_insert_with(Default::default).methods.insert(id, fir);
        Ok(true)
    }

    /// `realize_value_methods()` of `prepareVTables`.
    fn sp_realize_value_methods(&mut self) -> Result<bool> {
        let types = self.types();
        let hash_name = mm("Any", "__hash__");
        let equal_name = mm("Any", "__eq__");
        let dict_name = mm("TypeInfo", "value_dict");
        let need_hash = self.sp_has_realizations(&hash_name);
        let need_equal = self.sp_has_realizations(&equal_name);
        let need_dict = self.sp_has_realizations(&dict_name);
        if !need_hash && !need_equal && !need_dict {
            return Ok(false);
        }
        let dict_type = if need_dict { Some(self.get_stdlib_type(&stdlib().dict)?) } else { None };
        let dict_name_cls = dict_type.and_then(|d| types.class_name(d)).map(|n| n.to_string());
        let mut pending: Vec<typhon_ir::TypeId> = vec![];
        for t in self.sp_sorted_realization_types(&mm("TypeInfo", "_get")) {
            let g = types.get_class(self.extract_func_generic(t, 0)).unwrap();
            pending.push(self.make_ir_type(g)?);
        }
        for t in self.sp_sorted_realization_types(&mangle::func("std.internal.types.rtti", "get_typeinfo")) {
            let a = types.get_class(self.extract_func_arg_type(t, 0)).unwrap();
            pending.push(self.make_ir_type(a)?);
        }
        let module = self.cache.module.clone().expect("no IR module");
        {
            let m = module.borrow();
            for t in m.types() {
                if m.runtime_info(t).is_some_and(|i| i.value_accessed) {
                    pending.push(t);
                }
            }
        }
        let any_name = mangle::class("", "Any");
        let mut seen: BTreeSet<typhon_ir::TypeId> = BTreeSet::new();
        let mut added = false;
        while let Some(ty) = pending.pop() {
            if !seen.insert(ty) {
                continue;
            }
            let (ast, used, info) = {
                let m = module.borrow();
                let ast = m.ast_type(ty).and_then(|r| AstType::from_ref(&r));
                (ast, m.used_types(typhon_ir::NodeRef::Type(ty)), m.runtime_info(ty).cloned())
            };
            if ast.is_some_and(|a| types.is(a, &any_name)) {
                continue;
            }
            pending.extend(used);
            let Some(info) = info else { continue };
            let receiver = types.get_class(ast.expect("IR type without AST type")).unwrap();
            let rrn = types.realized_name(receiver);
            let mut descendants: Vec<String> = self.get_class_type(receiver).descendants.iter().cloned().collect();
            descendants.sort();
            for d in descendants {
                let Some(dc) = self.cache.classes.get(&d) else { continue };
                let mut reals: Vec<(String, _)> = dc.realizations.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
                reals.sort_by(|a, b| a.0.cmp(&b.0));
                for (_, r) in reals {
                    let r = r.borrow();
                    if r.bases.iter().any(|b| types.realized_name(*b) == rrn)
                        && let Some(ir) = r.ir
                    {
                        pending.push(ir);
                    }
                }
            }
            if need_hash {
                added |= self.sp_install_value_method(ty, receiver, "_hash_value", 1, "_unsupported_hash")?;
            }
            if need_equal {
                added |= self.sp_install_value_method(ty, receiver, "_equal_values", 2, "_unsupported_equal")?;
            }
            if need_dict {
                let is_dict = {
                    let m = module.borrow();
                    info.mro.iter().any(|b| {
                        m.ast_type(*b)
                            .and_then(|r| AstType::from_ref(&r))
                            .and_then(|a| types.get_class(a))
                            .and_then(|c| types.class_name(c))
                            .is_some_and(|n| Some(n.to_string()) == dict_name_cls)
                    })
                };
                if is_dict {
                    added |= self.sp_install_value_method(ty, receiver, "_dict_value", 1, "")?;
                }
            }
        }
        Ok(added)
    }

    /// `prepareVTables()`: build virtual tables for all realized polymorphic classes.
    pub fn prepare_vtables(&mut self) -> Result<()> {
        let mut added = true;
        while added {
            added = false;
            let mut class_names: Vec<String> = self.cache.classes.keys().cloned().collect();
            class_names.sort();
            for cls_name in class_names {
                // Iterate all methods
                let mut virtuals: Vec<String> = self.cache.classes[&cls_name].virtuals.iter().cloned().collect();
                virtuals.sort();
                for method in virtuals {
                    let Some(method_root) = self.cache.classes[&cls_name].methods.get(&method).cloned() else { continue };
                    for method_name in self.get_overloads(&method_root) {
                        for t in self.sp_sorted_realization_types(&method_name) {
                            if self.is_polymorphic(t) {
                                added |= self.sp_realize_descendant_virtuals(t)?;
                            }
                        }
                    }
                }
            }
            // Realizing a target can register more super calls; process a snapshot.
            let super_calls: Vec<(usize, TypeId)> = self.cache.super_calls.iter().map(|(k, v)| (*k, *v)).collect();
            for (id, signature) in super_calls {
                added |= self.sp_realize_descendant_supers(id, signature)?;
            }
            added |= self.sp_realize_value_methods()?;
        }

        let mut names: Vec<String> = self.cache.functions.keys().cloned().collect();
        names.sort();
        for name in names {
            let mut reals: Vec<(String, _)> =
                self.cache.functions[&name].realizations.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
            reals.sort_by(|a, b| a.0.cmp(&b.0));
            for (_, r) in reals {
                let (check, ast, ir, ty) = {
                    let r = r.borrow();
                    (r.check_initialization, r.ast, r.ir, r.ty)
                };
                if check && let (Some(ast), Some(_), Some(ty)) = (ast, ir, ty) {
                    self.check_field_initialization(ty, ast)?;
                }
            }
        }
        Ok(())
    }
}
