//! `vTableSignature`, `valueMethodId` and the virtual-dispatch half of `prepareVTables`.

use typhon_common::{ErrorKind, Result};

use super::super::TypecheckVisitor;
use super::mm;
use crate::ast::{Attr, CallArg, KeywordStarExpr, ParamKind, StarExpr};
use crate::cache::ClassRealizationRef;
use crate::ids::TypeId;
use crate::types::Unification;

impl<'a> TypecheckVisitor<'a> {
    /// `vTableSignature(ft)`.
    pub fn vtable_signature(&self, f: TypeId) -> String {
        let types = &self.cache.types;
        typhon_common::seqassert!(types.can_realize(f), "cannot realize {}", types.debug_string(f, 2));
        let base = types.get_class(self.extract_func_arg_type(f, 0)).expect("bad vtable base");
        let gs: Vec<String> = types.func_args(f).iter().map(|a| types.realized_name(a.ty())).collect();
        let mut f1 = gs.join(",");
        let gs: Vec<String> =
            types.func_generics(f).iter().filter(|a| !a.name.is_empty()).map(|a| types.realized_name(a.ty())).collect();
        let f2 = gs.join(",");
        if !f1.is_empty() && !f2.is_empty() {
            f1 = format!("{f1};{f2}");
        } else if f1.is_empty() {
            f1 = f2;
        }
        format!("{}.{}({})", types.class_name(base).unwrap(), self.get_unmangled_name(&types.func_name(f)), f1)
    }

    /// `valueMethodId(method)`.
    pub fn value_method_id(&mut self, method: &str) -> usize {
        self.sp_thunk_id(format!("$any.{method}"))
    }

    /// `thunkIds.emplace(key, thunkIds.size() + 1)`.
    pub(super) fn sp_thunk_id(&mut self, key: String) -> usize {
        let n = self.cache.thunk_ids.len() + 1;
        *self.cache.thunk_ids.entry(key).or_insert(n)
    }

    /// `source->unify(destination, nullptr) >= 0`, undoing a failed attempt.
    pub(super) fn sp_unifies(&mut self, a: TypeId, b: TypeId) -> Result<bool> {
        let types = self.types();
        let mut u = Unification::new();
        if types.unify(a, b, Some(&mut u), self)? >= 0 {
            return Ok(true);
        }
        u.undo(&types);
        Ok(false)
    }

    /// `is_subtype(source, destination)`: virtual results are covariant; arguments flow in
    /// the opposite direction, from caller to implementation.
    pub(super) fn sp_is_subtype(&mut self, source: TypeId, destination: TypeId) -> Result<bool> {
        let types = self.types();
        if self.sp_unifies(source, destination)? {
            return Ok(true);
        }
        let (Some(cls), Some(base)) = (types.get_class(source), types.get_class(destination)) else { return Ok(false) };
        if types.is_record(cls) || types.is_record(base) {
            return Ok(false);
        }
        for parent in self.get_mro(cls)? {
            if self.sp_unifies(parent, base)? {
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Number of value parameters of a function's AST (`valueArity`).
    pub(super) fn sp_value_arity(&self, f: TypeId) -> usize {
        self.sp_fn_params(f).iter().filter(|p| p.is_value()).count()
    }

    /// `adapt_target(signature, receiver, target)`.
    fn sp_adapt_target(&mut self, signature: TypeId, receiver: TypeId, target: TypeId) -> Result<TypeId> {
        let types = self.types();
        // return types are covariant (derived class can return an extended/derived type)
        let (tr, sr) = (types.func_ret(target), types.func_ret(signature));
        if !self.sp_is_subtype(tr, sr)? {
            let src = self.cache.ast[types.func_ast_id(signature)].src;
            typhon_common::bail_e!(
                ErrorKind::CUSTOM,
                src,
                "function {}.{} returns {}, expected {}",
                self.get_unmangled_name(&types.class_name(receiver).unwrap()),
                self.get_unmangled_name(&types.func_name(target)),
                types.pretty_string(tr),
                types.pretty_string(sr)
            );
        }

        // dispatch_variance: a covariant result also needs a wrapper so the CIR call
        // retains its declared return type; the wrapper performs the result upcast.
        let mut compatible =
            types.func_size(signature) == types.func_size(target) && types.realized_name(sr) == types.realized_name(tr);
        let mut index = 1;
        while compatible && index < types.func_size(signature) {
            compatible = types.realized_name(self.extract_func_arg_type(signature, index))
                == types.realized_name(self.extract_func_arg_type(target, index));
            index += 1;
        }
        if compatible {
            return Ok(target);
        }

        let mut parameters = vec![];
        let mut arguments = vec![];
        let mut placeholders = vec![];
        let mut argument_stars = vec![];
        let mut argument_names = vec![];
        for parameter in self.sp_fn_params(signature) {
            let (stars, name) = parameter.name_with_stars();
            if parameter.is_value() && !self.get_unmangled_name(&name).starts_with('$') {
                argument_stars.push(stars);
                argument_names.push(if parameter.kind == ParamKind::KeywordOnly { self.get_unmangled_name(&name) } else { String::new() });
            }
        }
        typhon_common::seqassert!(argument_stars.len() == types.func_size(signature), "dispatch signature argument count mismatch");
        let src = self.src;
        for index in 0..types.func_size(signature) {
            let mut argument_type = self.extract_func_arg_type(signature, index);
            if types.get_static(argument_type).is_some() {
                argument_type = types.non_static_type(argument_type).unwrap_or(argument_type);
            }
            let name = format!("argument{index}");
            let rn = types.realized_name(argument_type);
            let ty = self.sp_id(&rn);
            parameters.push(self.cache.ast.new_param(src, name.clone(), Some(ty), None, 0, ParamKind::Normal));
            let mut argument = self.sp_id(&name);
            if index == 0 {
                let fc = self.sp_id(&mm("type", "_force_cast"));
                let trn = types.realized_name(self.extract_func_arg_type(target, 0));
                let tt = self.sp_id(&trn);
                argument = self.sp_call(fc, vec![argument, tt]);
            } else if argument_stars.get(index) == Some(&1) {
                argument = self.n_expr(StarExpr { expr: Some(argument) });
            } else if argument_stars.get(index) == Some(&2) {
                argument = self.n_expr(KeywordStarExpr { expr: Some(argument) });
            }
            arguments.push(CallArg::new(argument_names.get(index).cloned().unwrap_or_default(), Some(argument)));
            placeholders.push(self.sp_typed(argument_type));
        }
        let name = self.get_temporary_var("dispatch.adapter");
        let ret_rn = types.realized_name(sr);
        let ret = self.sp_id(&ret_rn);
        let tf = self.sp_id(&types.func_name(target));
        let call = self.sp_call_args(tf, arguments);
        let r = self.sp_return(Some(call));
        let body = self.sp_suite(vec![r]);
        let adapter = self.cache.ast.new_function(src, name.clone(), Some(ret), parameters, Some(body), vec![], false)?;
        let wrapped = self.sp_suite(vec![adapter]);
        if let Some(sc) = self.cache.hooks.scoping.clone() {
            sc(self.cache, wrapped, None)?;
        }
        let t = self.get_time();
        self.cache.ast[adapter].attrs.set_int(Attr::ExprTime, t);
        self.tr_stmt(adapter)?;
        let id = self.sp_id(&name);
        let c = self.sp_call(id, placeholders);
        let c = self.tr(c)?;
        let callee = self.cache.ast[c].kind.as_call().and_then(|c| c.expr).expect("adapter call was replaced");
        Ok(self.ety(callee))
    }

    /// `virtual_arguments(signature, receiver)`.
    fn sp_virtual_arguments(&mut self, signature: TypeId, receiver: TypeId) -> Vec<CallArg> {
        let types = self.types();
        let mut arguments = vec![];
        let mut index = 0;
        for parameter in self.sp_fn_params(signature) {
            let (stars, name) = parameter.name_with_stars();
            if !parameter.is_value() || self.get_unmangled_name(&name).starts_with('$') {
                continue;
            }
            let mut ty = self.extract_func_arg_type(signature, index);
            index += 1;
            if types.get_static(ty).is_some() {
                ty = types.non_static_type(ty).unwrap_or(ty);
            }
            // The signature stores variadic arguments in their packed runtime form.
            // Reconstruct the source call for overload resolution: passing a pack as
            // one argument would repack it on every virtual-realization iteration.
            // Keep the signature itself packed for dispatch and adapter generation.
            if stars == 1 {
                for g in types.generics(types.get_class(ty).unwrap()) {
                    let v = self.sp_typed(g.ty());
                    arguments.push(CallArg::new("", Some(v)));
                }
            } else if stars == 2 {
                let names = self.sp_tuple_names(self.get_int_literal(ty, 0));
                let values = types.get_class(self.extract_class_generic(ty, 1)).unwrap();
                for (keyword, g) in types.generics(values).into_iter().enumerate().take(names.len()) {
                    let v = self.sp_typed(g.ty());
                    arguments.push(CallArg::new(names[keyword].clone(), Some(v)));
                }
            } else {
                let v = self.sp_typed(ty);
                let n = if parameter.kind == ParamKind::KeywordOnly { self.get_unmangled_name(&name) } else { String::new() };
                arguments.push(CallArg::new(n, Some(v)));
            }
        }
        if let Some(front) = arguments.first().and_then(|a| a.value) {
            self.set_type(front, receiver);
        }
        arguments
    }

    /// `super_arguments(signature, receiver)`.
    fn sp_super_arguments(&mut self, signature: TypeId, receiver: TypeId) -> Vec<CallArg> {
        let types = self.types();
        let mut arguments = vec![];
        let r = self.sp_typed(receiver);
        arguments.push(CallArg::new("", Some(r)));
        let a1 = types.get_class(self.extract_func_arg_type(signature, 1)).unwrap();
        for g in types.generics(a1) {
            let v = self.sp_typed(g.ty());
            arguments.push(CallArg::new("", Some(v)));
        }
        let keyword_type = self.extract_func_arg_type(signature, 2);
        let names = self.sp_tuple_names(self.get_int_literal(keyword_type, 0));
        let values = types.get_class(self.extract_class_generic(keyword_type, 1)).unwrap();
        for (index, g) in types.generics(values).into_iter().enumerate().take(names.len()) {
            let v = self.sp_typed(g.ty());
            arguments.push(CallArg::new(names[index].clone(), Some(v)));
        }
        arguments
    }

    /// `find_target(receiver, member, arguments, requireOwn)`.
    fn sp_find_target(&mut self, receiver: TypeId, member: &str, arguments: &[CallArg], require_own: bool) -> Result<Option<TypeId>> {
        let types = self.types();
        let rc = types.get_class(receiver);
        let ms = self.find_method(rc, member)?;
        let methods = self.find_matching_methods(rc, &ms, arguments, None)?;
        let rname = types.class_name(receiver).map(|n| n.to_string()).unwrap_or_default();
        for method in methods {
            if !require_own {
                return Ok(Some(method));
            }
            let aid = types.func_ast_id(method);
            if self.cache.ast[aid].attrs.get_str(Attr::ParentClass) == Some(rname.as_str()) {
                return Ok(Some(method));
            }
        }
        Ok(None)
    }

    /// `realize_target(target, arguments)`.
    fn sp_realize_target(&mut self, target: TypeId, arguments: Vec<CallArg>) -> Result<TypeId> {
        let types = self.types();
        let f = self.sp_id(&types.func_name(target));
        let request = self.sp_call_args(f, arguments);
        self.cache.ast[request].attrs.set(Attr::Internal);
        let call = self.tr(request)?;
        let ct = self.ety(call);
        let return_type = self.realize(ct)?;
        typhon_common::seqassert!(return_type.is_some(), "cannot realize dispatch return type");
        let callee = self.cache.ast[call].kind.as_call().and_then(|c| c.expr).expect("dispatch call was replaced");
        let resolved = types.get_func(self.ety(callee));
        typhon_common::seqassert!(resolved.is_some_and(|r| types.can_realize(r)), "bad dispatch realization");
        Ok(resolved.expect("bad dispatch realization"))
    }

    /// `install_target(realization, id, signature, receiver, target)`.
    fn sp_install_target(
        &mut self,
        realization: &ClassRealizationRef,
        id: usize,
        signature: TypeId,
        receiver: TypeId,
        target: TypeId,
    ) -> Result<TypeId> {
        let types = self.types();
        let adapted = self.sp_adapt_target(signature, receiver, target)?;
        realization.borrow_mut().vtable.insert(id, adapted);
        let rn = types.realized_name(adapted);
        let fir = self
            .get_function_type(adapted)
            .and_then(|f| f.realizations.get(&rn).cloned())
            .and_then(|r| r.borrow().ir)
            .expect("dispatch target was not realized");
        let ir = realization.borrow().ir.expect("class realization without IR type");
        let module = self.cache.module.clone().expect("no IR module");
        let mut m = module.borrow_mut();
        let ty = m.ty_mut(ir);
        ty.runtime_info.get_or_insert_with(Default::default).methods.insert(id, fir);
        Ok(adapted)
    }

    /// Realizations of `base`'s descendants that do not have vtable slot `id` yet
    /// (`for_each_descendant`).
    ///
    /// Note: the reference iterates the live (sorted) maps; we take a snapshot. Realizations added
    /// meanwhile are handled by the next iteration of the `prepareVTables` fixpoint loop.
    fn sp_descendant_realizations(&self, base: TypeId, id: usize) -> Vec<ClassRealizationRef> {
        let mut out = vec![];
        let Some(cls) = self.cache.get_class_by_type(base) else { return out };
        let mut descendants: Vec<String> = cls.descendants.iter().cloned().collect();
        descendants.sort();
        for d in descendants {
            let Some(dc) = self.cache.classes.get(&d) else { continue };
            let mut reals: Vec<(String, ClassRealizationRef)> = dc.realizations.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
            reals.sort_by(|a, b| a.0.cmp(&b.0));
            for (_, r) in reals {
                if !r.borrow().vtable.contains_key(&id) {
                    out.push(r);
                }
            }
        }
        out
    }

    /// `realize_descendant_virtuals(method)`.
    pub(super) fn sp_realize_descendant_virtuals(&mut self, method: TypeId) -> Result<bool> {
        let types = self.types();
        let base = types.get_class(self.extract_func_arg_type(method, 0));
        typhon_common::seqassert!(base.is_some(), "not a valid base");
        let base = base.expect("not a valid base");
        let key = self.vtable_signature(method);
        let id = self.sp_thunk_id(key.clone());
        let base_rn = types.realized_name(base);
        let mname = types.func_name(method);
        let member = self.get_unmangled_name(&mname);
        let mut added = false;
        for realization in self.sp_descendant_realizations(base, id) {
            if realization.borrow().vtable.contains_key(&id) {
                continue;
            }
            let (receiver, bases) = {
                let r = realization.borrow();
                (r.ty.expect("realization without type"), r.bases.clone())
            };
            let receiver = types.get_class(receiver).unwrap();
            if types.realized_name(receiver) != base_rn && !bases.iter().any(|p| types.realized_name(*p) == base_rn) {
                continue;
            }
            let arguments = self.sp_virtual_arguments(method, receiver);
            let target = self.sp_find_target(receiver, &member, &arguments, false)?;
            // Every reachable receiver needs this thunk. Silently omitting it
            // accepts a program that can only fail in the runtime lookup trap.
            let Some(target) = target else {
                let src = self.cache.ast[types.func_ast_id(method)].src;
                typhon_common::bail_e!(
                    ErrorKind::CUSTOM,
                    src,
                    "no compatible virtual implementation of {} for {}",
                    key,
                    types.pretty_string(receiver)
                );
            };
            // inherited overrides must not acquire the descendant's Self
            let owner = self.cache.ast[types.func_ast_id(target)].attrs.get_str(Attr::ParentClass).map(|s| s.to_string());
            if let Some(owner) = owner {
                for b in self.get_mro(receiver)? {
                    if types.get_class(b).and_then(|c| types.class_name(c)).is_some_and(|n| n.as_str() == owner) {
                        if let Some(front) = arguments.first().and_then(|a| a.value) {
                            self.set_type(front, b);
                        }
                        break;
                    }
                }
            }
            let mut resolved =
                if types.func_name(target) == mname { method } else { self.sp_realize_target(target, arguments)? };
            // A subclass method with a different parameter list (e.g. extra defaulted
            // parameters) and an incompatible result is a new method that shadows the
            // base one for static calls, not an override: virtual dispatch through
            // the base keeps the nearest compatible implementation in the MRO.
            if resolved != method
                && self.sp_value_arity(resolved) != self.sp_value_arity(method)
                && !self.sp_is_subtype(types.func_ret(resolved), types.func_ret(method))?
            {
                let mro = self.get_mro(receiver)?;
                let mut fallback = method;
                for &mb in mro.iter().skip(1) {
                    let ancestor = types.get_class(mb).unwrap();
                    let inherited = self.sp_virtual_arguments(method, receiver);
                    if let Some(front) = inherited.first().and_then(|a| a.value) {
                        self.set_type(front, mb);
                    }
                    let Some(candidate) = self.sp_find_target(ancestor, &member, &inherited, true)? else { continue };
                    if types.func_name(candidate) == mname {
                        fallback = method;
                        break;
                    }
                    let r = self.sp_realize_target(candidate, inherited)?;
                    if self.sp_value_arity(r) == self.sp_value_arity(method) || self.sp_is_subtype(types.func_ret(r), types.func_ret(method))? {
                        fallback = r;
                        break;
                    }
                }
                resolved = fallback;
            }
            let adapted = self.sp_install_target(&realization, id, method, receiver, resolved)?;
            typhon_common::log_flag!(
                REALIZE,
                "[dispatch] {} (desc={}, id={}) = {}",
                key,
                types.debug_string(receiver, 2),
                id,
                types.debug_string(adapted, 2)
            );
            added = true;
        }
        Ok(added)
    }

    /// `realize_descendant_supers(id, signature)`.
    pub(super) fn sp_realize_descendant_supers(&mut self, id: usize, signature: TypeId) -> Result<bool> {
        let types = self.types();
        let lexical = types.get_class(self.extract_class_generic(self.extract_func_arg_type(signature, 0), 0)).unwrap();
        let member = self.get_str_literal(self.extract_func_generic(signature, 0), 0);
        let lexical_rn = types.realized_name(lexical);
        let mut added = false;
        for realization in self.sp_descendant_realizations(lexical, id) {
            if realization.borrow().vtable.contains_key(&id) {
                continue;
            }
            let rty = realization.borrow().ty.expect("realization without type");
            let mro = self.get_mro(rty)?;
            let Some(anchor) = mro.iter().position(|b| types.realized_name(*b) == lexical_rn) else { continue };
            // super continues after the lexical class in the runtime receiver's MRO.
            for &candidate in &mro[anchor + 1..] {
                let receiver = types.get_class(candidate).unwrap();
                let arguments = self.sp_super_arguments(signature, receiver);
                let Some(target) = self.sp_find_target(receiver, &member, &arguments, true)? else { continue };
                let resolved = self.sp_realize_target(target, arguments)?;
                let sig = *self.cache.super_call_signatures.get(&id).expect("missing super call signature");
                self.sp_install_target(&realization, id, sig, receiver, resolved)?;
                added = true;
                break;
            }
        }
        Ok(added)
    }
}
