//! Part 2: `realizeFunc`,
//! `realizeFuncOnce`.

use std::cell::RefCell;
use std::rc::Rc;

use typhon_common::mangle::stdlib;
use typhon_common::{CompilerError, ErrorKind, ParserErrors, Result};

use super::{Base, TypecheckItem, TypeContext, TypecheckVisitor};
use crate::ast::{AssignStmt, Attr, BoolExpr, CaptureType, IdExpr, Param, UpdateMode};
use crate::cache::{FunctionRealization, MAX_REALIZATION_DEPTH, VAR_USED_SUFFIX};
use crate::ids::TypeId;
use crate::types::{FuncAstInfo, LiteralKind};

impl<'a> TypecheckVisitor<'a> {
    /// Register the AST snapshot that `FuncType` printing needs (see `TypeArena::func_ast`).
    pub fn register_func_ast(&self, ast: crate::ids::StmtId) {
        let a = &self.cache.ast;
        let Some(f) = a[ast].kind.as_function() else { return };
        let info = FuncAstInfo {
            name: f.name.clone(),
            params: f.items.iter().map(|p| (p.name.clone(), p.status)).collect(),
            allow_pass_through: a[ast].attrs.has(Attr::AllowPassThrough),
        };
        self.cache.types.register_func_ast(ast, info);
    }

    /// `realizeFunc(type, force)`: realize with restarts for widened empty literals.
    pub fn realize_func(&mut self, ty: TypeId, force: bool) -> Result<Option<TypeId>> {
        let types = self.types();
        let name = types.realized_name(ty);
        let old_ctx = self.ctx.clone();
        let ast = types.func_ast_id(ty);
        let module = self.cache.ast[ast].attrs.get_str(Attr::Module).unwrap_or("").to_string();
        let fctx = self.get_import(&module).ctx.clone().expect("bad import");
        let mut attempt = 0;
        loop {
            let (bases, stack, level, block_level) = {
                let f = fctx.borrow();
                (f.bases.len(), f.stack_size(), f.typecheck_level, f.block_level)
            };
            let log_level = typhon_common::logger().level;
            let drop_slots = |c: &mut crate::cache::Cache| {
                let prefix = format!("{name}\x01");
                c.empty_literal_slots.retain(|_, v| !v.0.first().is_some_and(|f| f.1.starts_with(&prefix)));
            };
            match self.realize_func_once(ty, force) {
                Ok(r) => {
                    drop_slots(self.cache);
                    return Ok(r);
                }
                Err(e) => {
                    drop_slots(self.cache);
                    let retry = attempt < 16 && self.cache.empty_literal_restarts.contains(&name);
                    if !retry && self.cache.empty_literal_restarts.is_empty() {
                        return Err(e); // fast path: compilation aborts anyway
                    }
                    {
                        let mut f = fctx.borrow_mut();
                        while f.stack_size() > stack {
                            f.pop_block();
                        }
                        f.bases.truncate(bases);
                        f.typecheck_level = level;
                        f.block_level = block_level;
                    }
                    typhon_common::logger().level = log_level;
                    self.ctx = old_ctx.clone();
                    let fname = types.func_name(ty);
                    if let Some(f) = self.cache.functions.get_mut(&fname) {
                        if f.realizations.get(&name).is_some_and(|r| r.borrow().ast.is_none()) {
                            f.realizations.remove(&name);
                        }
                    }
                    if !retry {
                        return Err(e);
                    }
                    self.cache.empty_literal_restarts.remove(&name);
                }
            }
            attempt += 1;
        }
    }

    fn pop_realization_base(&mut self) {
        let mut c = self.ctx.borrow_mut();
        c.bases.pop();
        c.pop_block();
        c.typecheck_level -= 1;
        drop(c);
        adj_log(-1);
    }

    /// `realizeFuncOnce(type, force)`.
    pub fn realize_func_once(&mut self, ty: TypeId, force: bool) -> Result<Option<TypeId>> {
        let types = self.types();
        let fname = types.func_name(ty);
        let ast_id = types.func_ast_id(ty);
        let module = self.cache.ast[ast_id].attrs.get_str(Attr::Module).unwrap_or("").to_string();
        let key0 = types.realized_name(ty);
        if let Some(r) = self.cache.functions.get(&fname).and_then(|f| f.realizations.get(&key0)).cloned() {
            if !force {
                let r = r.borrow();
                let rt = r.ty.unwrap();
                if r.ast.is_none() && !types.can_realize(types.func_ret(rt)) {
                    self.ctx.borrow_mut().get_base().recursive_dependencies.insert(rt);
                }
                return Ok(Some(rt));
            }
        }
        let old_ctx = self.ctx.clone();
        self.ctx = self.get_import(&module).ctx.clone().expect("bad import");
        self.ctx.borrow_mut().prepare_checkpoint();
        if self.ctx.borrow().realization_depth() > MAX_REALIZATION_DEPTH {
            let n = self.get_user_facing_name(&fname);
            self.ctx = old_ctx;
            typhon_common::bail_e!(ErrorKind::MAX_REALIZATION, self.src, n);
        }
        let is_import = Self::is_import_fn(&fname);
        if !is_import {
            adj_log(1);
            let mut c = self.ctx.borrow_mut();
            c.add_block();
            c.typecheck_level += 1;
            c.bases.push(Base::new(&fname, Some(ty), Some(types.func_ret(ty))));
            let bn = c.base_name().to_string();
            for t in (0..c.bases.len() - 1).rev() {
                if bn.starts_with(c.bases[t].name.as_str()) {
                    c.get_base().parent = t as i32;
                    break;
                }
            }
        }
        let res = self.realize_func_body(ty, force, is_import, old_ctx.clone());
        if res.is_err() {
            self.ctx = old_ctx;
        }
        res
    }

    fn realize_func_body(&mut self, ty: TypeId, force: bool, is_import: bool, old_ctx: super::CtxRef) -> Result<Option<TypeId>> {
        let types = self.types();
        let st = stdlib();
        let fname = types.func_name(ty);
        let ast_id = types.func_ast_id(ty);
        // Types might change after realization, fix it
        for a in types.func_args(ty) {
            if let Some(c) = types.get_class(a.ty()) {
                self.realize_type(c)?;
            }
        }
        // Clone the generic AST that is to be realized
        let ast = self.cache.ast.clone_stmt(ast_id, true);
        if let Some(s) = self.generate_special_ast(ty)? {
            self.cache.ast[ast].kind.as_function_mut().unwrap().suite = Some(s);
        }
        self.add_class_generics(ty, true, false, false)?;
        let fstmt = self.cache.ast[ast].kind.as_function().unwrap().clone();
        let fattrs = self.cache.ast[ast].attrs.clone();
        {
            let infer = fstmt.ret.is_none() && self.is_unbound(types.func_ret(ty)) && !fstmt.async_ && !fattrs.has(Attr::IsGenerator);
            let mut c = self.ctx.borrow_mut();
            c.get_base().func = Some(ast);
            c.get_base().infer_return_type = infer;
        }
        // Internal functions have no AST that can be realized
        let has_ast = fstmt.suite.is_some() && !fattrs.has(Attr::Internal);
        // Add function arguments
        if let Some(b) = fattrs.get_bindings() {
            for (c, t) in &b.captures {
                if *t == CaptureType::Global {
                    match self.find(c, 0) {
                        None => typhon_common::bail_e!(ErrorKind::ID_NOT_FOUND, self.src, c),
                        Some(cp) if !cp.borrow().is_global() => {
                            typhon_common::bail_e!(ErrorKind::FN_GLOBAL_NOT_FOUND, self.src, "global", c)
                        }
                        _ => {}
                    }
                }
            }
            for (name, canonical) in &b.local_renames {
                let v = self.force_find(canonical);
                self.ctx.borrow_mut().add(name, v);
            }
        }
        let fgens = types.func_generics(ty);
        let (mut j, mut gi) = (0, 0);
        for p in fstmt.items.iter().filter(|_| has_ast) {
            let (_, var_name) = p.name_with_stars();
            let mut un = self.get_unmangled_name(&var_name);
            if p.is_value() {
                let mut at = self.extract_func_arg_type(ty, j);
                j += 1;
                let is_static = self.cache.ast.get_static_generic(p.ty) != LiteralKind::Runtime;
                if !is_static && types.get_static(at).is_some() {
                    at = types.non_static_type(at).unwrap_or(at);
                }
                if let Some(s) = un.strip_prefix('$') {
                    un = s.to_string();
                }
                if types.is(at, &st.type_wrap) {
                    let tv = self.instantiate_type_var(self.extract_class_generic(at, 0))?;
                    TypeContext::add_type(self.cache, &self.ctx, &un, &var_name, tv, Default::default());
                } else {
                    let l = types.new_link(at);
                    TypeContext::add_var(self.cache, &self.ctx, &un, &var_name, l, 0, Default::default());
                }
            } else {
                if let Some(s) = un.strip_prefix('$') {
                    let g = &fgens[gi];
                    let mut t = g.ty();
                    if g.static_kind == LiteralKind::Runtime && (!types.is(t, &st.type_) || self.is_metaclass_type(t)) {
                        t = self.instantiate_type_var(t)?;
                    }
                    let v = TypeContext::add_type(self.cache, &self.ctx, s, &var_name, t, Default::default());
                    v.borrow_mut().generic = true;
                }
                gi += 1;
            }
        }
        // Populate realization table in advance to support recursive realizations
        let key = types.realized_name(ty);
        let realizations_owner = fname.clone();
        let old_ir = self.cache.functions.get(&fname).and_then(|f| f.realizations.get(&key)).and_then(|r| r.borrow().ir);
        let r = Rc::new(RefCell::new(FunctionRealization { ty: Some(ty), ir: old_ir, ..Default::default() }));
        self.cache.functions.get_mut(&realizations_owner).unwrap().realizations.insert(key.clone(), r.clone());
        if let Some(b) = fattrs.get_bindings() {
            for c in b.captures.keys() {
                let h = self.find(c, 0);
                r.borrow_mut().captures.push(h.map(|h| h.borrow().canonical_name.clone()).unwrap_or_default());
            }
        }
        // Realizations should always be visible, so add them to the toplevel
        let module = self.ctx.borrow().module();
        let val = TypecheckItem::new(key.clone(), "", module, Some(ty), 0).into_ref();
        TypeContext::add_always_visible(self.cache, &self.ctx, val, true);
        self.ctx.borrow_mut().get_base().suite = fstmt.suite;

        if has_ast {
            let suite = fstmt.suite.unwrap();
            // Prepare captured parameters once per realization.
            let bindings = fattrs.get_bindings().cloned();
            let mut declared = self.cache.ast[suite].attrs.get_bindings().cloned();
            let mut prepend = vec![];
            for p in fstmt.items.iter().filter(|p| p.is_value()) {
                let name = self.get_unmangled_name(&p.name_with_stars().1);
                let binding = bindings.as_ref().and_then(|b| b.bindings.get(&name)).cloned();
                let decl = declared.as_ref().and_then(|b| b.bindings.get(&name)).cloned();
                if decl.is_some() || binding.as_ref().is_some_and(|b| b.is_nonlocal) {
                    let l = self.n_expr(IdExpr::new(name.clone()));
                    let rr = self.n_expr(IdExpr::new(name.clone()));
                    prepend.push(self.n_stmt(AssignStmt { lhs: Some(l), rhs: Some(rr), type_expr: None, update: UpdateMode::Assign }));
                    if let Some(d) = decl {
                        if d.count > 0 {
                            let l = self.n_expr(IdExpr::new(format!("{name}{VAR_USED_SUFFIX}")));
                            let rr = self.n_expr(BoolExpr { value: false });
                            prepend.push(self.n_stmt(AssignStmt { lhs: Some(l), rhs: Some(rr), type_expr: None, update: UpdateMode::Assign }));
                        }
                        declared.as_mut().unwrap().bindings.remove(&name);
                    }
                }
            }
            if let Some(d) = declared {
                if let Some(b) = self.cache.ast[suite].attrs.get_bindings_mut() {
                    *b = d;
                }
            }
            let time = self.cache.ast[suite].attrs.get_int(Attr::ExprTime).unwrap_or(0);
            let ssrc = self.cache.ast[suite].src;
            for s in &prepend {
                self.cache.ast[*s].attrs.set_int(Attr::ExprTime, time);
                self.cache.ast[*s].src = ssrc;
            }
            if let Some(k) = self.cache.ast[suite].kind.as_suite_mut() {
                prepend.extend(k.items.drain(..));
                k.items = prepend;
            }
            let old_bl = std::mem::replace(&mut self.ctx.borrow_mut().block_level, 0);
            let old_sb = std::mem::replace(&mut self.ctx.borrow_mut().static_branch_depth, 0);
            let ret = self.infer_types(Some(suite), false);
            self.ctx.borrow_mut().block_level = old_bl;
            self.ctx.borrow_mut().static_branch_depth = old_sb;
            match ret? {
                None => {
                    self.cache.functions.get_mut(&realizations_owner).unwrap().realizations.remove(&key);
                    let deps = std::mem::take(&mut self.ctx.borrow_mut().get_base().recursive_dependencies);
                    let mut errors = ParserErrors::default();
                    if !fstmt.name.starts_with("%_lambda") {
                        // Lambda typecheck failures are "ignored" (treated as statements).
                        errors = self.find_typecheck_errors(suite);
                    }
                    if !is_import {
                        self.pop_realization_base();
                    }
                    let mut deferred = false;
                    for d in deps {
                        let Some(df) = types.get_func(d) else { continue };
                        if types.can_realize(types.func_ret(df)) {
                            continue;
                        }
                        for imp in self.cache.imports.values() {
                            let Some(ic) = &imp.ctx else { continue };
                            if ic.borrow().bases.iter().any(|b| b.ty == Some(d)) {
                                old_ctx.borrow_mut().get_base().recursive_dependencies.insert(d);
                                deferred = true;
                            }
                        }
                    }
                    self.ctx = old_ctx;
                    // A helper realized with a partially deduced `self` (pass-through field
                    // unbounds) is retried once the owner `__init__` assigns the field.
                    let pending_self = fattrs.has(Attr::AllowPassThrough)
                        && types.get_unbounds(ty, true).iter().any(|u| types.link(*u).is_some_and(|l| l.pass_through));
                    if !errors.is_empty() && !deferred && !pending_self {
                        return Err(CompilerError::Parser(errors));
                    }
                    return Ok(None); // inference must be delayed
                }
                Some(ret) => self.ctx.borrow_mut().get_base().suite = Some(ret),
            }
            // NoneType return when unspecified and no return statement
            if fstmt.ret.is_none() && self.is_unbound(types.func_ret(ty)) {
                let mut rt = self.get_stdlib_type(&st.none_type)?;
                if fstmt.async_ {
                    let co = self.get_stdlib_type(&st.coroutine)?;
                    rt = self.instantiate_type_with(co, &[rt])?;
                }
                self.unify(types.func_ret(ty), rt)?;
            }
        }
        // Realize the return type
        let ret = self.realize(types.func_ret(ty))?;
        if types.has_unbounds(ty, false) {
            self.cache.functions.get_mut(&realizations_owner).unwrap().realizations.remove(&key);
            self.pop_realization_base();
            self.ctx = old_ctx;
            return Ok(None);
        }
        typhon_common::seqassert!(ret.is_some(), "cannot realize return type");
        let args: Vec<Param> = fstmt
            .items
            .iter()
            .map(|i| {
                let mut p = Param::with_status(i.name_with_stars().1, None, None, i.status);
                p.kind = i.kind;
                p
            })
            .collect();
        let suite = self.ctx.borrow().base_ref().suite;
        let rname = types.realized_name(ty);
        let src = self.cache.ast[ast_id].src;
        let rast = self.cache.ast.new_function(src, rname, None, args, suite, vec![], fstmt.async_)?;
        self.cache.ast[rast].attrs = fattrs.clone();
        self.register_func_ast(rast);
        r.borrow_mut().ast = Some(rast);

        let new_type = types.generalize(ty, 0);
        let new_key = types.realized_name(new_type);
        let pending = self.cache.pending_realizations.contains(&(types.func_name(new_type), new_key.clone()));
        {
            let f = self.cache.functions.get_mut(&realizations_owner).unwrap();
            if !pending {
                f.realizations.insert(new_key.clone(), r.clone());
            } else if let Some(x) = f.realizations.get(&new_key).cloned() {
                f.realizations.insert(key.clone(), x);
            }
            if force {
                if let Some(x) = f.realizations.get(&new_key) {
                    x.borrow_mut().ast = Some(rast);
                }
            }
        }
        r.borrow_mut().ty = Some(new_type);
        if r.borrow().ir.is_none() {
            let ir = self.make_ir_function(&r)?;
            r.borrow_mut().ir = Some(ir);
        }
        let module = self.ctx.borrow().module();
        let val = TypecheckItem::new(new_key, "", module, Some(new_type), 0).into_ref();
        TypeContext::add_always_visible(self.cache, &self.ctx, val, true);
        if !is_import {
            self.pop_realization_base();
        }
        self.ctx = old_ctx;
        typhon_common::log_flag!(REALIZE, "[func] {}", types.debug_string(new_type, 2));
        Ok(Some(new_type))
    }
}

/// `getLogger().level += d`.
fn adj_log(d: i32) {
    let mut l = typhon_common::logger();
    l.level += d;
}
