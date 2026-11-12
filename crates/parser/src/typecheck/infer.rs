//! Part 1: `unify`,
//! `joinNullableTypes`, `inferTypes`, `realize`, `realizeType`. Function realization is
//! in [`super::infer_func`], IR stubs in [`super::infer_ir`].

use std::collections::BTreeMap;

use typhon_common::mangle::stdlib;
use typhon_common::{CompilerError, ErrorKind, ErrorMessage, ParserErrors, Result};

use super::{TypecheckItem, TypeContext, TypecheckVisitor};
use crate::cache::ClassRealization;
use crate::ids::{StmtId, TypeId};
use crate::types::{Generic, Unification};

pub const MAX_TYPECHECK_ITER: i32 = 1000;

impl<'a> TypecheckVisitor<'a> {
    /// `unify(a, b)`: destructive unification; error if the types cannot be unified.
    pub fn unify(&mut self, a: TypeId, b: TypeId) -> Result<TypeId> {
        let types = self.types();
        if !types.unify_or_undo(a, b, self)? {
            let mut undo = Unification::new();
            types.unify(a, b, Some(&mut undo), self)?;
            let (pa, pb) = (types.pretty_string(a), types.pretty_string(b));
            typhon_common::bail_e!(ErrorKind::TYPE_UNIFY, self.src, pa, pb);
        }
        Ok(a)
    }

    /// `joinNullableTypes(left, right)`: nullable target without committing payloads.
    pub fn join_nullable_types(&mut self, left: TypeId, right: TypeId) -> Result<Option<TypeId>> {
        let types = self.types();
        let st = stdlib();
        let left = if types.get_static(left).is_some() { types.non_static_type(left).unwrap_or(left) } else { left };
        let right = if types.get_static(right).is_some() { types.non_static_type(right).unwrap_or(right) } else { right };
        // Numeric widening (CPython-like): int and float returns join to float.
        let payload = |t: TypeId| if types.is(t, &st.optional) && types.generics_len(t) > 0 { types.generics(t)[0].ty() } else { t };
        let (lp, rp) = (payload(left), payload(right));
        let widen = (types.is(lp, "Int") && types.is(rp, "float")) || (types.is(lp, "float") && types.is(rp, "Int"));
        if widen {
            let fl = if types.is(lp, "float") { lp } else { rp };
            if types.is(left, &st.optional) || types.is(right, &st.optional) {
                let o = self.get_stdlib_type(&st.optional)?;
                return Ok(Some(self.instantiate_type_with(o, &[fl])?));
            }
            return Ok(Some(fl));
        }
        // A `None` literal is `Optional[T]` with T unbound: join it like NoneType.
        let bare_opt = |t: TypeId| types.is(t, &st.optional) && !types.can_realize(t);
        if bare_opt(left) && !bare_opt(right) && types.can_realize(right) {
            if types.is(right, &st.optional) {
                return Ok(Some(right));
            }
            let o = self.get_stdlib_type(&st.optional)?;
            return Ok(Some(self.instantiate_type_with(o, &[right])?));
        }
        if bare_opt(right) && !bare_opt(left) && types.can_realize(left) {
            if types.is(left, &st.optional) {
                return Ok(Some(left));
            }
            let o = self.get_stdlib_type(&st.optional)?;
            return Ok(Some(self.instantiate_type_with(o, &[left])?));
        }
        if types.is(right, &st.optional) {
            return Ok(Some(right));
        }
        if types.is(left, &st.optional) {
            return Ok(Some(left));
        }
        if !types.is(left, &st.none_type) && !types.is(right, &st.none_type) {
            return Ok(None);
        }
        let member = if types.is(left, &st.none_type) { right } else { left };
        if types.is(member, &st.none_type) {
            return Ok(Some(member));
        }
        let o = self.get_stdlib_type(&st.optional)?;
        Ok(Some(self.instantiate_type_with(o, &[member])?))
    }

    /// `inferTypes(stmt, sweepForceRealize)`: LTS-DI fixpoint iteration. `Ok(None)` if
    /// inference must be delayed / failed (no progress).
    pub fn infer_types(&mut self, result: Option<StmtId>, sweep_force_realize: bool) -> Result<Option<StmtId>> {
        let Some(mut result) = result else { return Ok(None) };
        self.ctx.borrow_mut().get_base().iteration = 1;
        loop {
            let (iter, bname) = {
                let c = self.ctx.borrow();
                (c.base_ref().iteration, c.base_name().to_string())
            };
            typhon_common::log_flag!(TYPECHECK, "[iter] {} :: {}", bname, iter);
            if iter >= MAX_TYPECHECK_ITER {
                let mut errors = ParserErrors::default();
                let name = if bname.is_empty() { "toplevel".to_string() } else { self.get_unmangled_name(&bname) };
                let src = self.cache.ast[result].src;
                errors.add_error(typhon_common::error::Backtrace {
                    trace: vec![ErrorMessage::new(format!("cannot typecheck '{name}' in reasonable time"), src)],
                });
                errors.append(&self.find_typecheck_errors(result));
                return Err(CompilerError::Parser(errors));
            }
            // Keep iterating until: (1) success (done) or (2) failure (no changed nodes).
            let (mut changed, mut return_early) = {
                let mut c = self.ctx.borrow_mut();
                c.typecheck_level += 1;
                let ch = std::mem::replace(&mut c.changed_nodes, 0);
                let re = std::mem::replace(&mut c.return_early, false);
                (ch, re)
            };
            let r = {
                let pre = self.preamble;
                let mut tv = TypecheckVisitor::new(self.cache, self.ctx.clone(), Some(pre), None);
                tv.transform_stmt_opt(Some(result))
            };
            let r = match r {
                Ok(r) => r,
                Err(e) => {
                    self.ctx.borrow_mut().typecheck_level -= 1;
                    return Err(e);
                }
            };
            result = r.unwrap();
            // Join reached returns before wrapping them.
            let (infer, rets, ret_type) = {
                let c = self.ctx.borrow();
                let b = c.base_ref();
                (b.infer_return_type, b.return_statements.clone(), b.return_type)
            };
            if infer && !rets.is_empty() {
                let types = self.types();
                let (mut first, mut inferred) = (None, None);
                let (mut has_empty, mut has_unknown) = (false, false);
                for (st, bare) in rets {
                    let e = self.cache.ast[st].kind.as_return().and_then(|r| r.expr).unwrap();
                    let returned = self.ety(e);
                    has_empty |= bare;
                    if types.get_unbound(returned).is_some() {
                        has_unknown = true;
                    } else {
                        first.get_or_insert(returned);
                        match inferred {
                            None => inferred = Some(returned),
                            Some(i) => {
                                if let Some(n) = self.join_nullable_types(i, returned)? {
                                    inferred = Some(n);
                                }
                            }
                        }
                    }
                }
                if let Some(inf) = inferred {
                    if !types.is(inf, &stdlib().none_type) || !has_unknown {
                        let target = if has_empty { first.unwrap() } else { inf };
                        self.unify(ret_type.unwrap(), target)?;
                        let mut c = self.ctx.borrow_mut();
                        c.get_base().infer_return_type = false;
                        c.get_base().return_statements.clear();
                        c.changed_nodes += 1;
                    }
                }
            }
            {
                let mut c = self.ctx.borrow_mut();
                std::mem::swap(&mut c.changed_nodes, &mut changed);
                std::mem::swap(&mut c.return_early, &mut return_early);
                c.typecheck_level -= 1;
            }

            if sweep_force_realize && self.ctx.borrow().base_ref().iteration == 1 {
                // Realize all @force_realize functions (copy keys: map may change)
                let mut names: Vec<String> = self.cache.functions.keys().cloned().collect();
                names.sort();
                for fname in names {
                    let f = self.cache.functions.get(&fname).unwrap();
                    let (Some(fty), Some(ast)) = (f.ty, f.ast) else { continue };
                    if !f.realizations.is_empty() {
                        continue;
                    }
                    let at = &self.cache.ast[ast].attrs;
                    use crate::ast::Attr;
                    if at.has(Attr::ForceRealize) || at.has(Attr::Export) || (at.has(Attr::C) && !at.has(Attr::CVarArg)) {
                        typhon_common::seqassert!(self.cache.types.can_realize(fty), "cannot realize {}", fname);
                        typhon_common::log_flag!(REALIZE, "[force_realize] {}", self.cache.types.debug_string(fty, 2));
                        let t = self.instantiate_type(fty, None)?;
                        self.realize(t)?;
                    }
                }
            }

            if self.cache.ast[result].done {
                break;
            } else if changed != 0 {
                self.ctx.borrow_mut().get_base().iteration += 1;
                continue;
            }
            // Nothing changed: try unbounds with default values (explicit first).
            let mut another = false;
            let rt = self.ctx.borrow().base_ref().return_type;
            if let Some(t) = rt {
                self.ctx.borrow_mut().get_base().pending_defaults.entry(0).or_default().insert(t);
            }
            let keys: Vec<i32> = self.ctx.borrow().base_ref().pending_defaults.keys().copied().collect();
            let types = self.types();
            for k in keys {
                let unbounds: Vec<TypeId> = self.ctx.borrow().base_ref().pending_defaults.get(&k).map(|s| s.iter().copied().collect()).unwrap_or_default();
                if unbounds.is_empty() {
                    continue;
                }
                for u in unbounds {
                    let Some(l) = types.link(u) else { continue };
                    let Some(d) = l.default_type else { continue };
                    let mut undo = Unification::new();
                    let target = if types.get_class(d).is_some() {
                        match self.extract_class_type(d) {
                            Some(c) => c,
                            None => continue,
                        }
                    } else {
                        d
                    };
                    if types.unify(u, target, Some(&mut undo), self)? >= 0 {
                        another = true;
                    }
                }
                self.ctx.borrow_mut().get_base().pending_defaults.get_mut(&k).unwrap().clear();
                if another {
                    break;
                }
            }
            if another {
                self.ctx.borrow_mut().get_base().iteration += 1;
                continue;
            }
            return Ok(None);
        }
        Ok(Some(result))
    }

    /// `realize(type)`: realize a type (and the function for function types).
    pub fn realize(&mut self, typ: TypeId) -> Result<Option<TypeId>> {
        let types = self.types();
        if !types.can_realize(typ) {
            return Ok(None);
        }
        let r = self.realize_inner(typ);
        match r {
            Ok(r) => Ok(r),
            Err(CompilerError::Parser(mut errs)) => {
                let Some(bt) = errs.errors.last_mut() else { return Err(CompilerError::Parser(errs)) };
                if bt.trace.first().is_some_and(|m| m.error_code == ErrorKind::MAX_REALIZATION as i32) {
                    return Err(CompilerError::Parser(errs));
                }
                if let Some(f) = types.get_func(typ) {
                    let ast_id = types.func_ast_id(f);
                    if self.cache.ast[ast_id].attrs.has(crate::ast::Attr::HiddenFromUser) {
                        if let Some(m) = bt.trace.last_mut() {
                            m.loc = self.src;
                        }
                    } else {
                        let fs = self.cache.ast[ast_id].kind.as_function().unwrap().clone();
                        let (mut ai, mut gi) = (0, 0);
                        let mut args = vec![];
                        for p in &fs.items {
                            let (ns, n) = p.name_with_stars();
                            let t = if p.is_generic() {
                                gi += 1;
                                types.pretty_string(self.extract_func_generic(f, gi - 1))
                            } else {
                                ai += 1;
                                types.pretty_string(self.extract_func_arg_type(f, ai - 1))
                            };
                            args.push(format!("{}{}: {}", "*".repeat(ns), self.get_user_facing_name(&n), t));
                        }
                        let mut name = fs.name.clone();
                        let mut name_args = String::new();
                        if name.starts_with("%_import_") {
                            for i in self.cache.imports.values() {
                                if typhon_common::mangle::mangled_func("", &format!("{}_call", i.import_var), 0, 0, true) == name {
                                    name = i.name.clone();
                                    break;
                                }
                            }
                            name = format!("<import {name}>");
                        } else {
                            name = self.get_user_facing_name(&fs.name);
                            name_args = format!("({})", args.join(", "));
                        }
                        bt.add_message(format!("during the realization of {name}{name_args}"), self.src);
                    }
                } else {
                    bt.add_message(format!("during the realization of {}", types.pretty_string(typ)), self.src);
                }
                Err(CompilerError::Parser(errs))
            }
            Err(e) => Err(e),
        }
    }

    fn realize_inner(&mut self, typ: TypeId) -> Result<Option<TypeId>> {
        let types = self.types();
        if let Some(f) = types.get_func(typ) {
            if let Some(ret) = self.realize_func(f, false)? {
                // Realize Function[..] type as well
                let t = types.copy_class(ret);
                self.realize_type(t)?;
                // Needed for return type unification
                let (a, b) = (types.func_ret(f), self.extract_class_generic(ret, 1));
                self.unify(a, b)?;
                return Ok(Some(ret));
            }
        } else if let Some(c) = types.get_class(typ) {
            return self.realize_type(c);
        }
        Ok(None)
    }

    /// `realizeType(type)`: realize a class and create its IR type stub.
    pub fn realize_type(&mut self, ty: TypeId) -> Result<Option<TypeId>> {
        let types = self.types();
        let st = stdlib();
        if !types.can_realize(ty) {
            return Ok(None);
        }
        // generalize generics to ensure that they do not get unified later!
        if types.is(ty, &st.unrealized_type) {
            let g = types.generalize(self.extract_class_generic(ty, 0), 0);
            types.with_class_mut(ty, |c| c.generics[0].ty = Some(g));
        }
        if types.is(ty, "__NTuple__") {
            let n = self.get_int_literal(ty, 0).max(0) as usize;
            let tt = self.extract_class_generic(ty, 1);
            let ttg = types.generics(tt);
            let tup = self.generate_tuple(n * ttg.len(), true)?;
            let t = self.instantiate_type(tup, None)?;
            let tgs = types.generics(t);
            let mut gens: Vec<Generic> = vec![];
            let mut j = 0;
            for _ in 0..n {
                for g in &ttg {
                    self.unify(tgs[j].ty(), g.ty())?;
                    gens.push(tgs[j].clone());
                    j += 1;
                }
            }
            types.with_class_mut(ty, |c| {
                c.name = st.tuple.as_str().into();
                c.generics = gens;
                c.rn.clear();
            });
        }
        // Check if the type was already realized
        let rn = types.class_realized_name(ty);
        let cname = types.class_name(ty).unwrap().to_string();
        if let Some(r) = self.cache.classes.get(&cname).and_then(|c| c.realizations.get(&rn)) {
            return Ok(r.borrow().ty.and_then(|t| types.get_class(t)));
        }
        let mut realized = types.get_class(ty).unwrap();
        let fields = self.get_class_fields(realized);
        if self.cache.classes.get(&cname).and_then(|c| c.ast).is_none() {
            return Ok(None); // not yet done!
        }
        let ftypes = self.get_class_field_types(realized)?;
        if types.get_static(ty).is_some() {
            realized = types.non_static_type(ty).and_then(|t| types.get_class(t)).unwrap_or(realized);
        }
        // Realize generics
        if !types.is(ty, &st.unrealized_type) {
            for g in types.generics(realized) {
                if self.realize(g.ty())?.is_none() {
                    return Ok(None);
                }
                if let Some(f) = types.get_func(g.ty()) {
                    if !types.can_realize(types.func_ret(f)) {
                        return Ok(None);
                    }
                }
            }
        }
        // Realizations should always be visible, so add them to the toplevel
        let rn = types.class_realized_name(ty);
        let rt = types.generalize(realized, 0);
        types.freeze(rt);
        let module = self.ctx.borrow().module();
        let mut val = TypecheckItem::new(rn.clone(), "", module, Some(rt), 0);
        if !types.is(rt, &st.type_) || self.is_metaclass_type(rt) {
            val.ty = Some(self.instantiate_type_var(realized)?);
        }
        if !self.is_metaclass_type(ty) {
            TypeContext::add_always_visible(self.cache, &self.ctx, val.into_ref(), true);
        }
        self.cache.class_realization_cnt += 1;
        let real = ClassRealization { ty: Some(rt), id: self.cache.class_realization_cnt, ..Default::default() };
        let real = std::rc::Rc::new(std::cell::RefCell::new(real));
        let rname = types.class_name(realized).unwrap().to_string();
        self.cache.classes.get_mut(&rname).unwrap().realizations.insert(rn.clone(), real.clone());

        let mros = self.cache.classes.get(&rname).unwrap().mro.clone();
        for m in mros.iter().skip(1) {
            let g = types.generalize(*m, 0); // generics are not yet generalized when parsing methods
            let gc = types.get_class(g).unwrap();
            let mt = self.instantiate_type(gc, Some(realized))?;
            typhon_common::seqassert!(types.can_realize(mt), "cannot realize {}", types.debug_string(mt, 2));
            real.borrow_mut().bases.push(mt);
        }
        // Create IR stub
        let lt = self.make_ir_type(realized)?;

        // Realize fields
        let mut type_args = vec![];
        let mut names = vec![];
        let mut member_info = BTreeMap::new();
        for (i, ft) in ftypes.iter().enumerate() {
            if self.realize(*ft)?.is_none() {
                let rp = types.pretty_string(realized);
                typhon_common::bail_e!(ErrorKind::TYPE_CANNOT_REALIZE_ATTR, self.src, fields[i].name, rp);
            }
            real.borrow_mut().fields.push((fields[i].name.clone(), *ft));
            names.push(fields[i].name.clone());
            let fc = types.get_class(*ft).unwrap();
            type_args.push(self.make_ir_type(fc)?);
            member_info.insert(fields[i].name.clone(), types.src(*ft));
        }
        self.finish_ir_type_fields(lt, type_args, names, member_info, &fields)?;
        Ok(Some(rt))
    }
}
