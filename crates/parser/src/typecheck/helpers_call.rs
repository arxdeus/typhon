//! Part 3: method lookup, call
//! matching (`canCall`, `matchCallable`, `reorderNamedArgs`), class generics, fields.

use std::cell::RefCell;

use smallvec::SmallVec;
use typhon_common::mangle::stdlib;
use typhon_common::{ErrorKind, Result, emsg};

use super::{CallArgSlots, ReorderDoneFn, ReorderErrorFn, TypecheckVisitor, TypeContext};
use crate::ast::{CallArg, NoneExpr, ParamKind};
use crate::cache::FN_DISPATCH_SUFFIX;
use crate::ids::{ExprId, TypeId};
use crate::types::{LinkKind, LiteralKind, Unification, class::partial_flag};

thread_local! {
    /// Journals of probing `matchCallable` calls (`static thread_local probes`).
    static PROBES: RefCell<Vec<usize>> = const { RefCell::new(Vec::new()) };
}

impl<'a> TypecheckVisitor<'a> {
    fn dummy_args(&mut self, args: &[(String, TypeId)]) -> Vec<CallArg> {
        args.iter()
            .map(|(n, t)| {
                let v = self.n_expr(NoneExpr); // dummy expression
                self.set_type(v, *t);
                CallArg::new(n.clone(), Some(v))
            })
            .collect()
    }
    /// `findBestMethod(typ, member, vector<Type*>)`.
    pub fn find_best_method_types(&mut self, typ: TypeId, member: &str, args: &[TypeId]) -> Result<Option<TypeId>> {
        let a: Vec<_> = args.iter().map(|t| (String::new(), *t)).collect();
        let ca = self.dummy_args(&a);
        let methods = self.find_method(Some(typ), member)?;
        Ok(self.find_matching_methods(Some(typ), &methods, &ca, None)?.first().copied())
    }
    /// `findBestMethod(typ, member, vector<Expr*>)`.
    pub fn find_best_method_exprs(&mut self, typ: TypeId, member: &str, args: &[ExprId]) -> Result<Option<TypeId>> {
        let ca: Vec<CallArg> = args.iter().map(|e| CallArg::new("", Some(*e))).collect();
        let methods = self.find_method(Some(typ), member)?;
        Ok(self.find_matching_methods(Some(typ), &methods, &ca, None)?.first().copied())
    }
    /// `findBestMethod(typ, member, vector<pair<string, Type*>>)`.
    pub fn find_best_method_named(&mut self, typ: TypeId, member: &str, args: &[(String, TypeId)]) -> Result<Option<TypeId>> {
        let ca = self.dummy_args(args);
        let methods = self.find_method(Some(typ), member)?;
        Ok(self.find_matching_methods(Some(typ), &methods, &ca, None)?.first().copied())
    }

    /// `findMethod(type, method)`: all overloads (latest first) along the MRO.
    pub fn find_method(&mut self, ty: Option<TypeId>, method: &str) -> Result<Vec<TypeId>> {
        let types = self.types();
        let st = stdlib();
        let populate = |this: &Self, cls: &str, vv: &mut Vec<TypeId>| {
            let Some(t) = this.cache.classes.get(cls).and_then(|c| c.methods.get(method)) else { return };
            let mt = this.cache.overloads.get(t).cloned().unwrap_or_default();
            for m in mt.iter().rev() {
                let f = this.cache.functions.get(m);
                if m.ends_with(FN_DISPATCH_SUFFIX) || f.and_then(|f| f.ty).is_none() {
                    continue;
                }
                vv.push(f.unwrap().ty.unwrap());
            }
        };
        let mut vv = vec![];
        let Some(ty) = ty else { return Ok(vv) };
        if types.is(ty, &st.tuple) && method == "__new__" && types.generics_len(ty) > 0 {
            let n = types.generics_len(ty);
            self.generate_tuple(n, true)?;
            populate(self, &st.tuple, &mut vv);
            return Ok(vv.into_iter().filter(|f| types.func_size(*f) == n).take(1).collect());
        }
        if let Some(cls) = self.cache.get_class_by_type(ty) {
            let mro = cls.mro.clone();
            for pc in mro {
                let n = types.class_name(pc).unwrap();
                let n = if n == "__NTuple__" { st.tuple.clone() } else { n.to_string() };
                populate(self, &n, &mut vv);
            }
        }
        Ok(vv)
    }

    /// `getBaseClasses(type)`.
    pub fn get_base_classes(&mut self, ty: TypeId) -> Result<Vec<TypeId>> {
        let mro = self.get_class_type(ty).mro.clone();
        mro.into_iter().map(|m| self.instantiate_type(m, Some(ty))).collect()
    }

    /// `getClassFieldTypes(cls)`: instantiated field types (realizing lazy annotations).
    pub fn get_class_field_types(&mut self, cls: TypeId) -> Result<Vec<TypeId>> {
        self.with_class_generics(cls, false, false, false, |tv| {
            let mut result = vec![];
            for f in tv.get_class_fields(cls) {
                let ft = tv.instantiate_type(f.ty.unwrap(), Some(cls))?;
                if !tv.cache.types.can_realize(ft) {
                    if let Some(te) = f.type_expr {
                        let c = tv.cache.ast.clone_expr(te, true);
                        let e = tv.tr(c)?;
                        let t = tv.extract_type_expr(e);
                        tv.unify(ft, t)?;
                    }
                }
                result.push(ft);
            }
            Ok(result)
        })
    }

    /// `isHeterogenous(type)`.
    pub fn is_heterogenous(&mut self, ty: TypeId) -> Result<bool> {
        let types = self.types();
        if types.get_class(ty).is_none() || !types.is_record(ty) {
            return Ok(false);
        }
        let fs: Vec<TypeId> = if types.is(ty, &stdlib().tuple) {
            types.generics(ty).iter().map(|g| g.ty()).collect()
        } else {
            let c = types.get_class(ty).unwrap();
            self.get_class_field_types(c)?
        };
        if fs.len() > 1 {
            let first = types.realized_name(fs[0]);
            return Ok(fs[1..].iter().any(|f| types.realized_name(*f) != first));
        }
        Ok(false)
    }

    /// `addClassGenerics(typ, func, onlyMangled, instantiate)`: add generics to the
    /// context; returns the names added.
    pub fn add_class_generics(&mut self, typ: TypeId, func: bool, only_mangled: bool, instantiate: bool) -> Result<Vec<String>> {
        let types = self.types();
        let st = stdlib();
        let mut added = vec![];
        let mut gens = vec![];
        if func && types.get_func(typ).is_some() {
            let mut parent = types.func_parent(typ);
            while let Some(p) = parent {
                if let Some(f) = types.get_func(p) {
                    gens.extend(types.func_generics(f));
                    parent = types.func_parent(f);
                } else {
                    gens.extend(types.hidden_generics(p));
                    gens.extend(types.generics(p));
                    break;
                }
            }
            gens.extend(types.func_generics(typ));
        } else {
            gens.extend(types.hidden_generics(typ));
            gens.extend(types.generics(typ));
        }
        for g in gens {
            let mut t = g.ty();
            if instantiate {
                if let Some(l) = types.link(t) {
                    if l.kind == LinkKind::Generic {
                        let mut lx = l.clone();
                        lx.kind = LinkKind::Unbound;
                        t = types.new_link_type(lx, types.src(t));
                    }
                }
            }
            if g.static_kind == LiteralKind::Runtime && (!types.is(t, &st.type_) || self.is_metaclass_type(t)) {
                t = self.instantiate_type_var(t)?;
            }
            let n = if only_mangled { g.name.to_string() } else { self.get_unmangled_name(&g.name) };
            let v = TypeContext::add_type(self.cache, &self.ctx, &n, &g.name, t, Default::default());
            if n != g.name.as_str() {
                added.push(g.name.to_string());
            }
            added.push(n);
            v.borrow_mut().generic = true;
        }
        Ok(added)
    }

    /// `withClassGenerics(typ, fn, func, onlyMangled, instantiate)`.
    pub fn with_class_generics<R>(
        &mut self,
        typ: TypeId,
        func: bool,
        only_mangled: bool,
        instantiate: bool,
        f: impl FnOnce(&mut Self) -> Result<R>,
    ) -> Result<R> {
        self.ctx.borrow_mut().add_block();
        let added = match self.add_class_generics(typ, func, only_mangled, instantiate) {
            Ok(a) => a,
            Err(e) => {
                self.ctx.borrow_mut().pop_block();
                return Err(e);
            }
        };
        let t = f(self);
        // do not remove stuff that was added in the meantime, potentially by AssignExpr
        let block = self.ctx.borrow().block().clone();
        let mut add_later = vec![];
        for i in block {
            if !added.contains(&i) {
                add_later.push((i.clone(), self.force_find(&i)));
            }
        }
        self.ctx.borrow_mut().pop_block();
        for (n, v) in add_later {
            self.ctx.borrow_mut().add(&n, v);
        }
        t
    }

    /// `getPartialArgTypes(fn, part)`: captured types indexed by AST parameter.
    pub fn get_partial_arg_types(&self, f: TypeId, part: Option<TypeId>) -> Vec<Option<TypeId>> {
        let types = &self.cache.types;
        let params = types.func_ast(types.func_ast_id(f)).params.clone();
        let mut result = vec![None; params.len()];
        if let Some(part) = part {
            let known = types.partial_mask(part);
            let captured = self.extract_class_generic(part, 1);
            let cg = types.generics(captured);
            let mut pi = 0;
            for (i, k) in known.bytes().enumerate() {
                if params.get(i).is_some_and(|p| p.1 == crate::ast::ParamStatus::Value)
                    && (k == partial_flag::INCLUDED || k == partial_flag::DEFAULT)
                {
                    result[i] = cg.get(pi).map(|g| g.ty());
                    pi += 1;
                }
            }
        }
        result
    }

    /// `acceptsCallShape(func, args)`: cheap arity/keyword check.
    pub fn accepts_call_shape(&self, func: TypeId, args: &[CallArg]) -> bool {
        let a = &self.cache.ast;
        for arg in args {
            if arg.value.is_some_and(|v| a[v].kind.as_ellipsis().is_some()) {
                return true;
            }
        }
        let ast = self.cache.types.func_ast_id(func);
        let Some(fs) = a[ast].kind.as_function() else { return true };
        let Some(layout) = fs.call_layout.as_deref() else { return true };
        let positional = args.iter().filter(|a| a.name.is_empty()).count();
        if positional > layout.positional_limit as usize && layout.star == -1 {
            return false;
        }
        let mut named_slots = vec![];
        for arg in args.iter().filter(|a| !a.name.is_empty()) {
            let slot = layout.keywords.iter().rev().find(|k| k.0 == arg.name).map(|k| k.1).unwrap_or(-1);
            if slot == -1 {
                if layout.kwstar == -1 {
                    return false;
                }
            } else {
                if (slot as usize) < positional.min(layout.positional_limit as usize) {
                    return false;
                }
                named_slots.push(slot);
            }
        }
        layout.required_values.iter().all(|s| (*s as usize) < positional || named_slots.contains(s))
    }

    /// `reorderNamedArgs(func, args, onDone, onError, known)`.
    pub fn reorder_named_args(
        &mut self,
        func: TypeId,
        args: &[CallArg],
        on_done: &mut ReorderDoneFn<'_, 'a>,
        mut on_error: Option<&mut ReorderErrorFn<'_, 'a>>,
        known: &str,
    ) -> Result<i32> {
        let types = self.types();
        let a = &self.cache.ast;
        let partial = args.last().is_some_and(|l| {
            l.name.is_empty() && l.value.and_then(|v| a[v].kind.as_ellipsis()).is_some_and(|e| e.mode == crate::ast::EllipsisType::Partial)
        });
        let ast_id = types.func_ast_id(func);
        let fs = a[ast_id].kind.as_function().unwrap().clone();
        let layout = fs.call_layout.as_deref().cloned().unwrap_or_default();
        let (star, kwstar, plimit) = (layout.star, layout.kwstar, layout.positional_limit);
        let mut score = layout.variadic_penalty + 2 * (fs.items.len() as i32 - types.func_generics(func).len() as i32);
        let mut slots: CallArgSlots = vec![SmallVec::new(); fs.items.len()];
        let kb = known.as_bytes();
        typhon_common::seqassert!(known.is_empty() || slots.len() == known.len(), "bad 'known' string");
        let mut extra: Vec<i32> = vec![];
        let mut named: Vec<(String, i32)> = vec![];
        let mut extra_named: Vec<(String, i32)> = vec![];
        let mut si = 0usize;
        for ai in 0..args.len() - partial as usize {
            if args[ai].name.is_empty() {
                while !kb.is_empty() && si < slots.len() && kb[si] == partial_flag::INCLUDED {
                    si += 1;
                }
                if (si as i32) < plimit {
                    slots[si].push(ai as i32);
                    si += 1;
                } else {
                    extra.push(ai as i32);
                }
            } else if let Some(p) = named.iter_mut().find(|v| v.0 == args[ai].name) {
                p.1 = ai as i32;
            } else {
                named.push((args[ai].name.clone(), ai as i32));
            }
        }
        for index in [star.max(kwstar), star.min(kwstar)] {
            if index != -1 && !slots[index as usize].is_empty() {
                extra.insert(0, index);
                slots[index as usize].clear();
            }
        }
        for (name, arg) in &named {
            let slot = layout
                .keywords
                .iter()
                .rev()
                .find(|k| &k.0 == name && (kb.is_empty() || kb[k.1 as usize] != partial_flag::INCLUDED))
                .map(|k| k.1)
                .unwrap_or(-1);
            if slot == -1 {
                extra_named.push((name.clone(), *arg));
            } else if slots[slot as usize].is_empty() {
                slots[slot as usize].push(*arg);
            } else {
                let Some(oe) = on_error.as_deref_mut() else { return Ok(-1) };
                let src = args[*arg as usize].value.map(|v| self.cache.ast[v].src).unwrap_or(self.src);
                return oe(self, ErrorKind::CALL_REPEATED_NAME, src, emsg(ErrorKind::CALL_REPEATED_NAME, &[name]));
            }
        }
        if !extra.is_empty() && star == -1 {
            let Some(oe) = on_error.as_deref_mut() else { return Ok(-1) };
            let n = self.get_user_facing_name(&fs.name);
            let cnt = args.len() - partial as usize;
            let m = emsg(ErrorKind::CALL_ARGS_MANY, &[&n, &plimit, &cnt]);
            return oe(self, ErrorKind::CALL_ARGS_MANY, self.src, m);
        }
        if star != -1 {
            slots[star as usize] = extra.iter().copied().collect();
        }
        if !extra_named.is_empty() && kwstar == -1 {
            let Some(oe) = on_error.as_deref_mut() else { return Ok(-1) };
            let (n, a0) = &extra_named[0];
            let src = args[*a0 as usize].value.map(|v| self.cache.ast[v].src).unwrap_or(self.src);
            let m = emsg(ErrorKind::CALL_ARGS_INVALID, &[n, &self.get_unmangled_name(&fs.name)]);
            return oe(self, ErrorKind::CALL_ARGS_INVALID, src, m);
        }
        if kwstar != -1 {
            for (_, a) in &extra_named {
                slots[kwstar as usize].push(*a);
            }
        }
        for i in 0..slots.len() {
            if !slots[i].is_empty() || i as i32 == star || i as i32 == kwstar {
                continue;
            }
            let p = &fs.items[i];
            if layout.optional.get(i).copied().unwrap_or(false) || (p.is_value() && !kb.is_empty() && kb[i] == partial_flag::INCLUDED) {
                score -= 2;
            } else if !partial && p.is_value() {
                let Some(oe) = on_error.as_deref_mut() else { return Ok(-1) };
                let (_, name) = p.name_with_stars();
                let fname = self.get_unmangled_name(&fs.name);
                let m = if p.kind == ParamKind::KeywordOnly {
                    format!("{}() missing 1 required keyword-only argument: '{}'", fname, self.get_unmangled_name(&name))
                } else {
                    emsg(ErrorKind::CALL_ARGS_MISSING, &[&fname, &self.get_unmangled_name(&name)])
                };
                return oe(self, ErrorKind::CALL_ARGS_MISSING, self.src, m);
            }
        }
        let r = on_done(self, star, kwstar, &slots, partial)?;
        Ok(if r != -1 { score + r } else { -1 })
    }

    /// `findMatchingMethods(typ, methods, args, part)`: methods accepting `args`.
    pub fn find_matching_methods(&mut self, typ: Option<TypeId>, methods: &[TypeId], args: &[CallArg], part: Option<TypeId>) -> Result<Vec<TypeId>> {
        let types = self.types();
        let mut results = vec![];
        for &mi in methods {
            if part.is_none() && !typ.is_some_and(|t| types.get_partial(t).is_some()) && !self.accepts_call_shape(mi, args) {
                continue;
            }
            let m = self.instantiate_type(mi, typ)?;
            if self.match_callable_impl(m, args, None, None, part, None, true)? != -1 {
                results.push(mi);
            }
        }
        Ok(results)
    }

    /// `matchCallable(type, args, returnType, us, captured, preserveUnions)`.
    pub fn match_callable(
        &mut self,
        ty: TypeId,
        args: &[CallArg],
        ret: Option<TypeId>,
        us: Option<&mut Unification>,
        captured: Option<TypeId>,
        preserve: Option<&mut Vec<bool>>,
    ) -> Result<i32> {
        self.match_callable_impl(ty, args, ret, us, captured, preserve, false)
    }

    /// `matchCallableImpl(...)`.
    #[allow(clippy::too_many_arguments)]
    pub fn match_callable_impl(
        &mut self,
        ty: TypeId,
        args: &[CallArg],
        ret: Option<TypeId>,
        us: Option<&mut Unification>,
        captured: Option<TypeId>,
        preserve: Option<&mut Vec<bool>>,
        instantiated: bool,
    ) -> Result<i32> {
        let mut probe = Unification::new();
        let owns = us.is_none();
        let changes: &mut Unification = match us {
            Some(u) => u,
            None => &mut probe,
        };
        let key = changes as *const Unification as usize;
        if owns {
            PROBES.with(|p| p.borrow_mut().push(key));
        }
        let probing = owns || PROBES.with(|p| p.borrow().contains(&key));
        let r = self.match_callable_body(ty, args, ret, changes, captured, preserve, instantiated, probing);
        if owns {
            PROBES.with(|p| p.borrow_mut().pop());
            let types = self.types();
            changes.undo(&types);
        }
        r
    }

    #[allow(clippy::too_many_arguments)]
    fn match_callable_body(
        &mut self,
        ty: TypeId,
        args: &[CallArg],
        ret: Option<TypeId>,
        changes: &mut Unification,
        captured: Option<TypeId>,
        preserve: Option<&mut Vec<bool>>,
        instantiated: bool,
        probing: bool,
    ) -> Result<i32> {
        let types = self.types();
        let st = stdlib();
        let Some(cls) = types.get_class(ty) else { return Ok(-1) };
        if !types.is(cls, &st.function) && types.get_partial(cls).is_none() && !types.is(cls, &st.type_wrap) {
            return Ok(-1);
        }
        let part = captured.or_else(|| types.get_partial(cls));
        let mut function = None;
        if let Some(p) = types.get_partial(cls) {
            let g = types.generalize(types.partial_func(p), 0);
            function = Some(self.instantiate_type(g, None)?);
        } else if types.is(cls, &st.type_wrap) {
            let m = self.find_method(Some(cls), "__call_no_self__")?;
            let Some(m0) = m.first() else { return Ok(-1) };
            function = Some(self.instantiate_type(*m0, Some(cls))?);
        } else if types.get_func(cls).is_some() {
            function = Some(if instantiated { cls } else { self.instantiate_type(cls, None)? });
        }
        if let Some(f) = function {
            let f = types.get_func(f).unwrap();
            let score = self.can_call(f, args, part, preserve, Some(changes), probing)?;
            if score < 0 {
                return Ok(-1);
            }
            if !probing && ret.is_some() && types.can_realize(f) {
                let Some(r) = self.realize(f)? else { return Ok(-1) };
                let rf = types.get_func(r).unwrap();
                if types.unify(types.func_ret(f), types.func_ret(rf), Some(changes), self)? < 0 {
                    return Ok(-1);
                }
            }
            return Ok(match ret {
                None => score,
                Some(r) => {
                    if types.unify(r, types.func_ret(f), Some(changes), self)? >= 0 { score } else { -1 }
                }
            });
        }
        // Function pointers have a fixed ABI and no AST parameters/defaults to reorder.
        let params = self.extract_class_generic(cls, 0);
        let pg = types.generics(params);
        if pg.len() != args.len() {
            return Ok(-1);
        }
        for (i, a) in args.iter().enumerate() {
            if !a.name.is_empty() || types.unify(self.ety(a.value.unwrap()), pg[i].ty(), Some(changes), self)? < 0 {
                return Ok(-1);
            }
        }
        Ok(match ret {
            None => 1,
            Some(r) => {
                if types.unify(r, self.extract_class_generic(cls, 1), Some(changes), self)? >= 0 { 1 } else { -1 }
            }
        })
    }
}


impl<'a> TypecheckVisitor<'a> {
    /// `getCallDefault(param)`: transformed default of a parameter.
    pub fn get_call_default(&mut self, p: &crate::ast::Param) -> Result<ExprId> {
        use crate::ast::{CallExpr, IdExpr, InstantiateExpr};
        let st = stdlib();
        let d = p.default_value.unwrap();
        // An unannotated None default participates in optional inference at calls.
        if self.cache.ast[d].kind.as_none().is_some() && p.ty.is_none() {
            let o = self.n_expr(IdExpr::new(st.optional.clone()));
            let n = self.n_expr(IdExpr::new(st.none_type.clone()));
            let i = self.n_expr(InstantiateExpr { expr: Some(o), items: vec![n] });
            let c = self.n_expr(CallExpr { expr: Some(i), items: vec![], ordered: false, partial: false });
            return self.tr(c);
        }
        // `__callsite_file__` / `__callsite_line__` defaults expand to the caller's
        // source location (used by e.g. `warnings.warn`).
        if let Some(name) = self.cache.ast[d].kind.as_id().map(|i| i.value.clone()) {
            if name == "__callsite_file__" || name == "__callsite_line__" {
                let src = self.src;
                let c = if name == "__callsite_file__" {
                    self.n_expr(crate::ast::StringExpr::new(src.file.as_str(), ""))
                } else {
                    self.n_expr(crate::ast::IntExpr::new(src.line as i64))
                };
                return self.tr(c);
            }
        }
        let c = self.cache.ast.clone_expr(d, true);
        self.tr(c)
    }

    /// `getCallArgType(fn, parameterIndex, valueIndex, actual)`: union specialization.
    pub fn get_call_arg_type(&mut self, f: TypeId, pi: usize, vi: usize, actual: TypeId) -> Result<TypeId> {
        let types = self.types();
        let expected = self.extract_func_arg_type(f, vi);
        if let Some(u) = types.get_union(expected) {
            if types.can_realize(u) {
                let fname = types.func_name(f);
                let Some(sig) = self.cache.functions.get(&fname).and_then(|x| x.ty) else { return Ok(expected) };
                let ast = types.func_ast_id(f);
                let pname = self.cache.ast[ast].kind.as_function().unwrap().items[pi].name_with_stars().1;
                let binding = self.cache.ast[ast].attrs.get_bindings().and_then(|b| b.bindings.get(&self.get_unmangled_name(&pname)).cloned());
                let at = &self.cache.ast[ast].attrs;
                if types.can_realize(actual)
                    && types.get_union(self.extract_func_arg_type(sig, vi)).is_some()
                    && binding.is_none_or(|b| b.count == 1 && !b.is_nonlocal)
                    && !at.has(crate::ast::Attr::LLVM)
                    && !at.has(crate::ast::Attr::C)
                {
                    for m in types.union_realization_types(u) {
                        if types.unify(m, actual, None, self)? >= 0 {
                            return Ok(actual);
                        }
                    }
                }
            }
        }
        Ok(expected)
    }

    /// `canCall(fn, args, part, preserveUnions, us, probing)`: score (-1 = no match).
    pub fn can_call(
        &mut self,
        f: TypeId,
        args: &[CallArg],
        part: Option<TypeId>,
        mut preserve: Option<&mut Vec<bool>>,
        us: Option<&mut Unification>,
        probing: bool,
    ) -> Result<i32> {
        let Some(changes) = us else {
            return self.match_callable(f, args, None, None, part, preserve);
        };
        if let Some(p) = preserve.as_deref_mut() {
            *p = vec![false; args.len()];
        }
        let types = self.types();
        let captured = self.get_partial_arg_types(f, part);
        let known = part.map(|p| types.partial_mask(p)).unwrap_or_default();
        let ast_id = types.func_ast_id(f);
        let ni = self.cache.ast.non_inferrable_generics(ast_id);
        let params = self.cache.ast[ast_id].kind.as_function().unwrap().items.clone();
        let mut done = |tv: &mut TypecheckVisitor<'a>, star: i32, kwstar: i32, slots: &CallArgSlots, partial: bool| -> Result<i32> {
            let changes: &mut Unification = &mut *changes;
            let mut preserve = preserve.as_deref_mut();
            tv.with_class_generics(f, true, false, false, |tv| {
                let types = tv.types();
                let mut provided = 0usize;
                let mut replacements: Vec<Option<TypeId>> = vec![];
                let (mut vi, mut gi) = (0usize, 0usize);
                for si in 0..slots.len() {
                    let param = &params[si];
                    if param.is_generic() {
                        let expected = tv.extract_func_generic(f, gi);
                        gi += 1;
                        if let Some(&ai) = slots[si].first() {
                            let value = args[ai as usize].value.unwrap();
                            let vt = tv.ety(value);
                            let ek = types.static_kind(expected) != LiteralKind::Runtime;
                            if (!ek && !Self::is_type_expr(tv.cache, value)) || (ek && types.static_kind(vt) == LiteralKind::Runtime) {
                                return Ok(-1);
                            }
                            if types.unify(tv.extract_type_expr(value), expected, Some(changes), tv)? < 0 {
                                return Ok(-1);
                            }
                            provided += 1;
                        } else if !partial && types.get_unbound(expected).is_some() && ni.contains(&param.name) && param.default_value.is_none() {
                            return Ok(-1);
                        }
                        continue;
                    }
                    let expected = tv.extract_func_arg_type(f, vi);
                    vi += 1;
                    let mut actual: Option<TypeId> = None;
                    if si as i32 == star || si as i32 == kwstar {
                        let mut values: Vec<TypeId> = vec![];
                        let mut names: Vec<String> = vec![];
                        let supplied: Vec<String> = slots[si].iter().map(|i| args[*i as usize].name.clone()).collect();
                        if let Some(p) = part {
                            if si as i32 == star {
                                let tuple = tv.extract_class_generic(p, 1);
                                let last = types.generics(tuple).last().unwrap().ty();
                                values.extend(types.generics(last).iter().map(|g| g.ty()));
                            } else {
                                let kw = tv.extract_class_generic(p, 2);
                                let id = tv.get_int_literal(kw, 0) as usize;
                                let keys = types.generated_tuple_names.borrow()[id].clone();
                                let tuple = tv.extract_class_generic(kw, 1);
                                let tg = types.generics(tuple);
                                for (i, k) in keys.iter().enumerate() {
                                    if !supplied.contains(k) {
                                        names.push(k.clone());
                                        values.push(tg[i].ty());
                                    }
                                }
                            }
                        }
                        for &index in &slots[si] {
                            values.push(tv.ety(args[index as usize].value.unwrap()));
                            if si as i32 == kwstar {
                                names.push(args[index as usize].name.clone());
                            }
                        }
                        // Typed variadics coerce each element before building their tuple.
                        if let Some(pt) = param.ty {
                            let c = tv.cache.ast.clone_expr(pt, false);
                            let e = tv.tr(c)?;
                            let ann = tv.extract_type_expr(e);
                            for v in values.iter_mut() {
                                let Some(w) = tv.wrap_for_call(*v, ann, f, changes)? else { return Ok(-1) };
                                *v = w;
                            }
                        }
                        let tup = tv.generate_tuple(values.len(), true)?;
                        let mut a = tv.instantiate_type_with(tup, &values)?;
                        if si as i32 == kwstar {
                            let id = tv.generate_kw_id(&names);
                            let idt = types.new_int_static(id as i64);
                            let nt = tv.get_stdlib_type(&stdlib().named_tuple)?;
                            a = tv.instantiate_type_with(nt, &[idt, a])?;
                        }
                        actual = Some(a);
                    } else if let Some(&index) = slots[si].first() {
                        let a = tv.ety(args[index as usize].value.unwrap());
                        if let Some(p) = preserve.as_deref_mut() {
                            if types.get_union(a).is_some() && types.get_union(expected).is_some() && types.unify(a, expected, None, tv)? >= 0 {
                                p[index as usize] = true;
                            }
                        }
                        actual = Some(a);
                    } else if let Some(c) = captured[si] {
                        actual = Some(c);
                    } else if !probing && param.default_value.is_some() {
                        let d = tv.get_call_default(param)?;
                        actual = Some(tv.ety(d));
                    } else if !probing && param.name.starts_with('$') {
                        let n = tv.get_unmangled_name(&param.name)[1..].to_string();
                        let id = tv.n_expr(crate::ast::IdExpr::new(n));
                        let e = tv.tr(id)?;
                        actual = Some(tv.ety(e));
                    }
                    let Some(actual) = actual else { continue };
                    let ua = tv.get_underlying_static_type(actual)?;
                    let selected = tv.get_call_arg_type(f, si, vi - 1, ua)?;
                    if tv.wrap_for_call(actual, selected, f, changes)?.is_none() {
                        return Ok(-1);
                    }
                    if !probing && selected != expected {
                        // The signature is private; preserve the call's union specialization.
                        if replacements.is_empty() {
                            replacements = vec![None; types.func_size(f)];
                        }
                        replacements[vi - 1] = Some(selected);
                        let ast_ret = tv.cache.ast[ast_id].kind.as_function().unwrap().ret;
                        if ast_ret.is_none() && types.can_realize(types.func_ret(f)) {
                            let fname = types.func_name(f);
                            let sig = tv.cache.functions.get(&fname).and_then(|x| x.ty).unwrap();
                            let nr = tv.instantiate_type(types.func_ret(sig), None)?;
                            types.with_class_mut(f, |c| c.generics[1].ty = Some(nr));
                        }
                    }
                }
                if !replacements.is_empty() {
                    types.replace_arg_types(f, &replacements);
                }
                // Defaults are applied after argument inference, just as at real calls.
                if !probing && !partial {
                    let mut gi = 0;
                    for p in &params {
                        if p.is_generic() {
                            let expected = tv.extract_func_generic(f, gi);
                            gi += 1;
                            if types.get_unbound(expected).is_some() {
                                if let Some(d) = p.default_value {
                                    let c = tv.cache.ast.clone_expr(d, true);
                                    let e = tv.tr(c)?;
                                    if types.unify(tv.extract_type_expr(e), expected, Some(changes), tv)? < 0 {
                                        return Ok(-1);
                                    }
                                }
                            }
                        }
                    }
                }
                Ok((provided == types.func_generics(f).len()) as i32)
            })
        };
        self.reorder_named_args(f, args, &mut done, None, &known)
    }

    /// The `wrap` lambda of `canCall`: coerce `actual` to `expected` and unify.
    fn wrap_for_call(&mut self, actual: TypeId, expected: TypeId, f: TypeId, changes: &mut Unification) -> Result<Option<TypeId>> {
        let types = self.types();
        let (_, mut wrapped, conversion) = self.can_wrap_expr(actual, expected, Some(f), true, false)?;
        if conversion.is_none() && types.static_kind(expected) == LiteralKind::Runtime {
            // special case: @no_argument_wrap prevents needed static wrapping
            wrapped = Some(self.get_underlying_static_type(actual)?);
        }
        let w = wrapped.unwrap_or(actual);
        if types.unify(w, expected, Some(changes), self)? < 0 {
            return Ok(None);
        }
        Ok(Some(if types.get_class(expected).is_some() { expected } else { w }))
    }
}
