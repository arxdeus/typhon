
use std::cell::RefCell;
use std::collections::BTreeSet;
use std::rc::Rc;

use typhon_common::mangle::{self, stdlib};
use typhon_common::{ErrorKind, Result, SrcInfo};

use super::{Base, CtxRef, TypeContext, TypecheckVisitor};
use crate::ast::{
    AssignStmt, Attr, AttrValue, CallLayout, CaptureType, EllipsisExpr, EllipsisType, ExprKind,
    ExprStmt, ForStmt, IdExpr, IndexExpr, KeywordStarExpr, LiteralKind, Param, ParamKind, ParamStatus,
    ReturnStmt, StarExpr, StringExpr, SuiteStmt, UpdateMode, YieldStmt, is_id,
};
use crate::cache::{Function, VAR_CLASS_TOPLEVEL};
use crate::ids::{ExprId, StmtId, TypeId};
use crate::types::{Generic, LinkKind, string_from_literal};

/// the reference `Param(name, type, default, status, kind)` status argument.
fn status_hint(s: ParamStatus) -> i32 {
    match s {
        ParamStatus::Value => 0,
        ParamStatus::Generic => 1,
        ParamStatus::HiddenGeneric => 2,
    }
}

impl<'a> TypecheckVisitor<'a> {
    /// `ctx->generateCanonicalName(name, includeBase, noSuffix)`.
    pub(crate) fn gen_canonical(&mut self, name: &str, include_base: bool, no_suffix: bool) -> String {
        let ctx = self.ctx.clone();
        let c = ctx.borrow();
        c.generate_canonical_name(self.cache, name, include_base, no_suffix)
    }

    /// `auto nctx = make_shared<TypeContext>(cache); *nctx = *ctx; nctx->bases.pop_back();`
    /// and, if `global`, drop everything but the toplevel base.
    pub(crate) fn copy_ctx_without_base(&self, global: bool) -> CtxRef {
        let mut n = self.ctx.borrow().clone();
        n.bases.pop();
        if global && n.bases.len() > 1 {
            n.bases.truncate(1);
        }
        Rc::new(RefCell::new(n))
    }

    /// `func->isAsync()` of the current base function.
    fn base_func_is_async(&self) -> bool {
        let f = self.ctx.borrow().base_ref().func;
        f.and_then(|f| self.cache.ast[f].kind.as_function().map(|x| x.async_)).unwrap_or(false)
    }
    pub(crate) fn base_func_has(&self, a: Attr) -> bool {
        let f = self.ctx.borrow().base_ref().func;
        f.is_some_and(|f| self.cache.ast[f].attrs.has(a))
    }
    fn base_return_type(&self) -> TypeId {
        self.ctx.borrow().base_ref().return_type.expect("no return type")
    }

    /// `visit(LambdaExpr*)`: lift the lambda into a function definition.
    pub fn visit_lambda(&mut self, e: ExprId) -> Result<()> {
        let name = self.get_temporary_var("lambda");
        let (params, body) = {
            let l = self.cache.ast[e].kind.as_lambda().expect("not a lambda");
            (l.items.clone(), l.expr)
        };
        let ret = self.n_stmt(ReturnStmt { expr: body });
        let suite = self.n_stmt(SuiteStmt { items: vec![ret] });
        let src = self.src;
        let mut f = self.cache.ast.new_function(src, name.clone(), None, params, Some(suite), vec![], false)?;
        let t = self.get_time();
        if t != 0 {
            self.cache.ast[f].attrs.set_int(Attr::ExprTime, t);
        }
        // TODO: just copy BindingsAttribute from expr instead?
        let wrapper = self.n_stmt(SuiteStmt { items: vec![f] });
        if let Some(sc) = self.cache.hooks.scoping.clone() {
            sc(self.cache, wrapper, None)?;
        }
        // Note: scoping may rewrite the suite in place; pick the function back up.
        if let Some(first) = self.cache.ast.first_in_block(wrapper) {
            if self.cache.ast[first].kind.as_function().is_some() {
                f = first;
            }
        }
        self.cache.ast[f].attrs.set_int(Attr::ExprTime, t); // to handle captures properly
        f = self.tr_stmt(f)?;
        if let Some(a) = self.cache.ast[e].attrs.get(Attr::Bindings).cloned() {
            self.cache.ast[f].attrs.set_value(Attr::Bindings, a);
        }
        self.prepend_stmts.borrow_mut().push(f);
        let id = self.n_expr(IdExpr::new(name));
        self.result_expr = Some(self.tr(id)?);
        Ok(())
    }

    /// `visit(YieldExpr*)`: unify the function return type with `Generator[?]`.
    pub fn visit_yield_expr(&mut self, e: ExprId) -> Result<()> {
        if !self.ctx.borrow().in_function() {
            let src = self.cache.ast[e].src;
            typhon_common::bail_e!(ErrorKind::FN_OUTSIDE_ERROR, src, "yield");
        }
        let rt = self.base_return_type();
        let g = self.get_stdlib_type("Generator")?;
        let et = self.ety(e);
        let it = self.instantiate_type_with(g, &[et])?;
        self.unify(rt, it)?;
        if self.realize(et)?.is_some() {
            self.set_done_expr(e);
        }
        Ok(())
    }

    /// `visit(AwaitExpr*)`.
    pub fn visit_await(&mut self, e: ExprId) -> Result<()> {
        let src = self.cache.ast[e].src;
        if !self.ctx.borrow().in_function() {
            typhon_common::bail_e!(ErrorKind::FN_OUTSIDE_ERROR, src, "await");
        }
        if !self.base_func_is_async() {
            typhon_common::bail_e!(ErrorKind::FN_OUTSIDE_ERROR, src, "await");
        }
        let types = self.types();
        let st = stdlib();
        let inner = self.cache.ast[e].kind.as_await().unwrap().expr;
        let inner = self.transform(inner)?;
        self.cache.ast[e].kind.as_await_mut().unwrap().expr = inner;
        let inner = inner.unwrap();
        let future = mangle::class("std.asyncio", "Future");
        let task = mangle::class("std.asyncio", "Task");
        let transformed = self.cache.ast[e].kind.as_await().unwrap().transformed;
        if !transformed {
            if let Some(c) = types.get_class(self.ety(inner)) {
                let mut is_coroutine = types.is(c, &st.coroutine) || types.is(c, &future) || types.is(c, &task);
                if !is_coroutine {
                    if !self.find_method(Some(c), "__await__")?.is_empty() {
                        let d = self.dot_n(inner, "__await__");
                        let call = self.call_n(d, vec![]);
                        let ne = self.tr(call)?;
                        let nt = self.ety(ne);
                        is_coroutine = types.is(nt, &st.coroutine)
                            || types.is(nt, &future)
                            || types.is(nt, &task)
                            || types.is(nt, &st.generator);
                        if !is_coroutine {
                            typhon_common::bail_e!(ErrorKind::EXPECTED_TYPE, src, "awaitable");
                        } else {
                            let a = self.cache.ast[e].kind.as_await_mut().unwrap();
                            a.expr = Some(ne);
                            a.transformed = true;
                        }
                    } else {
                        typhon_common::bail_e!(ErrorKind::EXPECTED_TYPE, src, "awaitable");
                    }
                }
            }
        }
        let inner = self.cache.ast[e].kind.as_await().unwrap().expr.unwrap();
        let it = self.ety(inner);
        if types.get_class(it).is_some() {
            let g = self.extract_class_generic(types.get_class(it).unwrap(), 0);
            let et = self.ety(e);
            self.unify(et, g)?;
        }
        if self.cache.ast[inner].done {
            self.set_done_expr(e);
        }
        Ok(())
    }

    /// `visit(ReturnStmt*)`. Empty return is transformed to `return NoneType()`.
    /// Also partialize functions if they are being returned.
    pub fn visit_return(&mut self, s: StmtId) -> Result<()> {
        let types = self.types();
        let st = stdlib();
        if self.cache.ast[s].attrs.has(Attr::Internal) {
            let id = self.id_n(&mangle::mangled_method("", "NoneType", "__new__", 0, 0));
            let call = self.call_n(id, vec![]);
            let e = self.tr(call)?;
            self.cache.ast[s].kind.as_return_mut().unwrap().expr = Some(e);
            self.set_done_stmt(s);
            return Ok(());
        }
        let src = self.cache.ast[s].src;
        if !self.ctx.borrow().in_function() {
            typhon_common::bail_e!(ErrorKind::FN_OUTSIDE_ERROR, src, "return");
        }
        let is_async = self.base_func_is_async();
        let mut expr = self.cache.ast[s].kind.as_return().unwrap().expr;
        let has_empty_return = expr.is_none();
        let is_gen = self.base_func_has(Attr::IsGenerator);

        if expr.is_none() && is_gen {
            self.set_done_stmt(s);
        } else {
            if is_gen {
                typhon_common::bail_e!(ErrorKind::CUSTOM, src, "returning values from generators not yet supported");
            }
            if expr.is_none() {
                let id = self.id_n(&st.none_type);
                expr = Some(self.call_n(id, vec![]));
            }
            let mut ex = self.tr(expr.unwrap())?;
            self.cache.ast[s].kind.as_return_mut().unwrap().expr = Some(ex);
            let can_realize = types.can_realize(self.ety(ex));

            // Wrap expression to match the return type
            let (infer, rt) = {
                let c = self.ctx.borrow();
                (c.base_ref().infer_return_type, c.base_ref().return_type.unwrap())
            };
            if !infer && types.get_unbound(rt).is_none() {
                let ok = self.wrap_expr(&mut ex, rt, None, true)?;
                self.cache.ast[s].kind.as_return_mut().unwrap().expr = Some(ex);
                if !ok {
                    return Ok(());
                }
            }

            // Special case: partialize functions if we are returning them
            if let Some(f) = types.get_func(self.ety(ex)) {
                if !(types.get_class(rt).is_some() && types.is(rt, &st.function)) {
                    let id = self.id_n(&types.func_name(f));
                    let ell = self.n_expr(EllipsisExpr { mode: EllipsisType::Partial });
                    let call = self.call_n(id, vec![ell]);
                    ex = self.tr(call)?;
                    self.cache.ast[s].kind.as_return_mut().unwrap().expr = Some(ex);
                }
            }

            if types.static_kind(rt) == LiteralKind::Runtime && types.get_static(self.ety(ex)).is_some() {
                if let Some(ns) = types.non_static_type(self.ety(ex)) {
                    self.set_type(ex, ns);
                }
            }

            let et = self.ety(ex);
            if is_async {
                let co = self.get_stdlib_type(&st.coroutine)?;
                let it = self.instantiate_type_with(co, &[et])?;
                self.unify(rt, it)?;
            } else if infer {
                let mut c = self.ctx.borrow_mut();
                let returns = &mut c.get_base().return_statements;
                if !returns.iter().any(|(r, _)| *r == s) {
                    returns.push((s, has_empty_return));
                }
            } else {
                self.unify(rt, et)?;
            }
            if !can_realize && types.can_realize(self.ety(ex)) {
                self.ctx.borrow_mut().changed_nodes += 1;
            }
        }

        // If we are not within conditional block, ignore later statements in this function.
        // Useful with static if statements.
        if self.ctx.borrow().block_level == 0 {
            self.ctx.borrow_mut().return_early = true;
        }

        let infer = self.ctx.borrow().base_ref().infer_return_type;
        let e = self.cache.ast[s].kind.as_return().unwrap().expr;
        if !infer && e.is_none_or(|e| self.cache.ast[e].done) {
            self.set_done_stmt(s);
        }
        Ok(())
    }

    /// `visit(YieldStmt*)`: empty yields assume `NoneType`.
    pub fn visit_yield(&mut self, s: StmtId) -> Result<()> {
        let src = self.cache.ast[s].src;
        if !self.ctx.borrow().in_function() {
            typhon_common::bail_e!(ErrorKind::FN_OUTSIDE_ERROR, src, "yield");
        }
        let is_async = self.base_func_is_async();
        let e = match self.cache.ast[s].kind.as_yield().unwrap().expr {
            Some(e) => e,
            None => {
                let id = self.id_n(&stdlib().none_type);
                self.call_n(id, vec![])
            }
        };
        let e = self.tr(e)?;
        self.cache.ast[s].kind.as_yield_mut().unwrap().expr = Some(e);
        let g = self.get_stdlib_type(if !is_async { "Generator" } else { "AsyncGenerator" })?;
        let et = self.ety(e);
        let it = self.instantiate_type_with(g, &[et])?;
        let rt = self.base_return_type();
        self.unify(rt, it)?;
        if self.cache.ast[e].done {
            self.set_done_stmt(s);
        }
        Ok(())
    }

    /// `visit(YieldFromStmt*)`: `yield from a` -> `for var in a: yield var`.
    pub fn visit_yield_from(&mut self, s: StmtId) -> Result<()> {
        let var = self.get_temporary_var("yield");
        let e = self.cache.ast[s].kind.as_yield_from().unwrap().expr;
        let v1 = self.id_n(&var);
        let v2 = self.id_n(&var);
        let y = self.n_stmt(YieldStmt { expr: Some(v2) });
        let f = self.n_stmt(ForStmt { var: Some(v1), iter: e, suite: Some(y), ..Default::default() });
        self.result_stmt = Some(self.tr_stmt(f)?);
        Ok(())
    }

    /// `visit(GlobalStmt*)`: remove upon completion.
    pub fn visit_global(&mut self, _s: StmtId) -> Result<()> {
        self.result_stmt = Some(self.n_stmt(SuiteStmt::default()));
        Ok(())
    }

    /// `prepareCallLayout(ast)`: finalize the argument structure after captures, names,
    /// and defaults are normalized.
    pub fn prepare_call_layout(&mut self, ast: StmtId) {
        let f = self.cache.ast[ast].kind.as_function().expect("not a function");
        typhon_common::seqassert!(f.call_layout.is_none(), "function call layout was already initialized");
        let n = f.items.len() as i32;
        let mut layout = CallLayout { positional_limit: n, star: -1, kwstar: -1, ..Default::default() };
        layout.optional.reserve(f.items.len());
        for (i, p) in f.items.iter().enumerate() {
            let i = i as i32;
            if p.kind == ParamKind::KeywordOnly || p.name.starts_with('*') {
                layout.positional_limit = layout.positional_limit.min(i);
            }
            if p.name.starts_with("**") {
                layout.kwstar = i;
                layout.variadic_penalty -= 2;
            } else if p.name.starts_with('*') {
                layout.star = i;
                layout.variadic_penalty -= 2;
            }
            let optional = (p.is_value() && p.default_value.is_some()) || p.name.starts_with('$');
            layout.optional.push(optional);
            if p.is_value() && !optional && !p.name.starts_with('*') {
                layout.required_values.push(i);
            }
            if p.kind != ParamKind::PositionalOnly {
                let (_, name) = p.name_with_stars();
                layout.keywords.push((self.get_unmangled_name(&name), i));
            }
        }
        self.cache.ast[ast].kind.as_function_mut().unwrap().call_layout = Some(Box::new(layout));
    }
}

/// Output of the base-guarded part of `visit(FunctionStmt*)`.
struct FnSignature {
    args: Vec<Param>,
    suite: Option<StmtId>,
    ret: Option<ExprId>,
    explicits: Vec<Generic>,
    base_type: TypeId,
}

impl<'a> TypecheckVisitor<'a> {
    /// `visit(FunctionStmt*)`: parse a function stub and create a corresponding generic
    /// function type. Also realize built-ins and extern C functions.
    pub fn visit_function(&mut self, s: StmtId) -> Result<()> {
        let types = self.types();
        let orig_stmt = self.cache.ast.clone_stmt(s, true);

        // Parse attributes
        let mut decorators: Vec<Option<ExprId>> =
            self.cache.ast[s].kind.as_function().unwrap().decorators.iter().map(|d| Some(*d)).collect();
        let mut has_decorators = false;
        for i in (0..decorators.len()).rev() {
            let Some(d) = decorators[i] else { continue };
            let (is_attr, attr_name, attr_realized_name) = self.get_decorator(d)?;
            if !attr_name.is_empty() {
                let attrs = &mut self.cache.ast[s].attrs;
                if attr_name == mangle::func("std.internal.attributes", "test") {
                    attrs.set(Attr::Test);
                } else if attr_name == mangle::func("std.internal.attributes", "export") {
                    attrs.set(Attr::Export);
                } else if attr_name == mangle::func("std.internal.attributes", "inline") {
                    attrs.set(Attr::Inline);
                } else if attr_name == mangle::func("std.internal.attributes", "no_arg_reorder") {
                    attrs.set(Attr::NoArgReorder);
                } else if attr_name == mangle::func("", "overload") {
                    attrs.set(Attr::Overload);
                }
                let mut kv = attrs.get_key_value(Attr::FunctionAttributes).cloned().unwrap_or_default();
                kv.insert(attr_name.clone(), attr_realized_name);

                let fn_ast = self.get_function_name(&attr_name).and_then(|f| f.ast);
                if let Some(fa) = fn_ast {
                    let fattrs = self.cache.ast[fa].attrs.clone();
                    let attrs = &mut self.cache.ast[s].attrs;
                    for a in [Attr::Export, Attr::Inline, Attr::NoArgReorder, Attr::ForceRealize] {
                        if fattrs.has(a) {
                            attrs.set(a);
                        }
                    }
                    if let Some(m) = fattrs.get_key_value(Attr::FunctionAttributes) {
                        for (k, v) in m {
                            kv.insert(k.clone(), v.clone());
                        }
                    }
                }
                self.cache.ast[s].attrs.set_value(Attr::FunctionAttributes, AttrValue::KeyValue(kv));
                if is_attr {
                    decorators[i] = None; // remove it from further consideration
                }
            }
            if !is_attr {
                has_decorators = true;
            }
        }

        let is_class_member = self.ctx.borrow().in_class();
        let stmt_name = self.cache.ast[s].kind.as_function().unwrap().name.clone();
        let has = |v: &Self, a: Attr| v.cache.ast[s].attrs.has(a);
        if has(self, Attr::ForceRealize) && (!self.ctx.borrow().is_global() || is_class_member) {
            typhon_common::bail_e!(ErrorKind::EXPECTED_TOPLEVEL, self.src, "builtin function");
        }

        // All overloads share the same canonical name except for the number at the
        // end (e.g., `foo.1:0`, `foo.1:1` etc.)
        let mut root_name = String::new();
        if is_class_member {
            // Case 1: method overload
            let bn = self.ctx.borrow().base_name().to_string();
            if let Some(n) = self.get_class_name(&bn).and_then(|c| c.methods.get(&stmt_name)) {
                root_name = n.clone();
            }
            if root_name.is_empty() && has(self, Attr::Overload) {
                let src = self.src;
                typhon_common::diag::compilation_warning(
                    &format!("function '{}' marked with unnecessary @overload", stmt_name),
                    src.file.as_str(),
                    src.line,
                    0,
                    0,
                    -1,
                    false,
                    Default::default(),
                );
            }
        } else if has(self, Attr::Overload) {
            // Case 2: function overload
            let t = self.get_time();
            if let Some(c) = self.find(&stmt_name, t) {
                let c = c.borrow();
                let ctx = self.ctx.borrow();
                if c.is_func(&types) && c.module_name == ctx.module() && c.base_name == ctx.base_name() {
                    root_name = c.canonical_name.clone();
                }
            }
        }
        if root_name.is_empty() {
            root_name = self.gen_canonical(&stmt_name, true, is_class_member);
        }
        // Append overload number to the name
        self.cache.overloads.entry(root_name.clone()).or_default();
        let canonical_name = format!("{}:{}", root_name, self.get_overloads(&root_name).len());
        self.cache.set_reverse(canonical_name.clone(), stmt_name.clone());

        if is_class_member {
            let bn = self.ctx.borrow().base_name().to_string();
            // Set the enclosing class name
            self.cache.ast[s].attrs.set_str(Attr::ParentClass, bn.clone());
            // Add the method to the class' method list
            self.cache.classes.entry(bn).or_default().methods.insert(stmt_name.clone(), root_name.clone());
        }

        self.handle_function_captures(s, &stmt_name, has_decorators)?;

        let is_global = self.ctx.borrow().is_global();
        let sig = {
            // Set up the base
            {
                let mut c = self.ctx.borrow_mut();
                let mut b = Base::new(&canonical_name, None, None);
                b.func = Some(s);
                c.bases.push(b);
                c.add_block();
            }
            let r = self.function_signature(s, &canonical_name, is_class_member, is_global);
            let mut c = self.ctx.borrow_mut();
            c.bases.pop();
            c.pop_block();
            r?
        };
        self.finish_function(s, orig_stmt, decorators, sig, root_name, canonical_name, is_class_member)
    }

    /// Handle captures. Add additional argument to the function for every capture.
    /// Make sure to account for **kwargs if present.
    fn handle_function_captures(&mut self, s: StmtId, stmt_name: &str, has_decorators: bool) -> Result<()> {
        let types = self.types();
        let st = stdlib();
        let Some(b) = self.cache.ast[s].attrs.get_bindings().cloned() else { return Ok(()) };
        let mut insert_size = {
            let f = self.cache.ast[s].kind.as_function().unwrap();
            let mut n = f.items.len();
            if f.items.last().is_some_and(|p| p.name.starts_with("**")) {
                n -= 1;
            }
            n
        };
        let mut local_renames = vec![];
        let src = self.src;
        let insert = |v: &mut Self, p: Param, at: &mut usize| {
            let f = v.cache.ast[s].kind.as_function_mut().unwrap();
            f.items.insert(*at, p);
            f.non_inferrable_generics = None;
            *at += 1;
        };
        // sorted_view: BTreeMap iteration is already sorted
        for (c, t) in &b.captures {
            let cc = format!("${c}");
            let time = self.get_time();
            if let Some(v) = self.find(c, time) {
                let (is_global, v_base, is_generic, sk, is_func, is_type, vname) = {
                    let v = v.borrow();
                    (
                        v.is_global(),
                        v.base_name.to_string(),
                        v.is_generic(),
                        v.static_kind(&types),
                        v.is_func(&types),
                        v.is_type(&types),
                        v.canonical_name.clone(),
                    )
                };
                if *t != CaptureType::Global && !is_global {
                    let parent_class_generic = {
                        let ctx = self.ctx.borrow();
                        let last = ctx.bases.last().unwrap();
                        last.is_type() && last.name.as_str() == v_base
                    };
                    if is_generic && parent_class_generic {
                        self.cache.ast[s].attrs.set(Attr::Method);
                    }
                    if !is_generic || (sk != LiteralKind::Runtime && !parent_class_generic) {
                        if !is_func {
                            let p = if is_type {
                                let ty = self.id_n(&st.type_);
                                self.cache.ast.new_param(src, cc, Some(ty), None, 0, ParamKind::Normal)
                            } else if sk != LiteralKind::Runtime {
                                let lit = self.id_n("Literal");
                                let k = self.id_n(string_from_literal(sk));
                                let ix = self.n_expr(IndexExpr { expr: Some(lit), index: Some(k) });
                                self.cache.ast.new_param(src, cc, Some(ix), None, 0, ParamKind::Normal)
                            } else if *t == CaptureType::Nonlocal {
                                let ty = self.id_n(&st.capsule);
                                self.cache.ast.new_param(src, cc, Some(ty), None, 0, ParamKind::Normal)
                            } else {
                                self.cache.ast.new_param(src, cc, None, None, 0, ParamKind::Normal)
                            };
                            // Note: the reference `Param(name, type)` has no source location.
                            let mut p = p;
                            p.src = SrcInfo::default();
                            insert(self, p, &mut insert_size);
                        } else {
                            // Local function is captured. Just note its canonical name and add it to
                            // the context during realization.
                            local_renames.push((c.clone(), vname));
                        }
                    }
                    continue;
                }
            }
            let shadow = self.ctx.borrow().global_shadows.contains_key(c);
            if (c == stmt_name && has_decorators) /* decorated recursive fns */ || shadow {
                let p = Param::with_status(cc, None, None, ParamStatus::Value);
                insert(self, p, &mut insert_size);
            }
        }
        if !local_renames.is_empty() {
            if let Some(bm) = self.cache.ast[s].attrs.get_bindings_mut() {
                for (k, v) in local_renames {
                    bm.local_renames.insert(k, v);
                }
            }
        }
        Ok(())
    }

    /// Base-guarded part of `visit(FunctionStmt*)`: arguments, generics, return type and body.
    fn function_signature(&mut self, s: StmtId, canonical_name: &str, is_class_member: bool, is_global: bool) -> Result<FnSignature> {
        let types = self.types();
        let st = stdlib();
        let items = self.cache.ast[s].kind.as_function().unwrap().items.clone();
        let mut args: Vec<Param> = vec![];
        let mut explicits: Vec<Generic> = vec![];

        // Parse arguments and add them to the context
        for a in &items {
            let (stars, mut var_name) = a.name_with_stars();
            let name = self.gen_canonical(&var_name, false, false);

            // Mark as method if the first argument is self
            if is_class_member && self.cache.ast[s].attrs.has(Attr::HasSelf) && a.name == "self" {
                self.cache.ast[s].attrs.set(Attr::Method);
            }

            // Handle default values
            let mut default_value = a.default_value;
            let simple = default_value.is_some_and(|d| match &self.cache.ast[d].kind {
                ExprKind::None(_) | ExprKind::Int(_) | ExprKind::Bool(_) | ExprKind::Float(_) | ExprKind::Id(_) | ExprKind::String(_) => true,
                ExprKind::Unary(u) => u.expr.is_some_and(|x| matches!(self.cache.ast[x].kind, ExprKind::Int(_))),
                _ => false,
            });
            if simple {
                // Special case: all simple types and Nones are handled at call site
                // (as they are not mutable).
                let d = default_value.unwrap();
                if matches!(self.cache.ast[d].kind, ExprKind::None(_)) {
                    if is_id(&self.cache.ast, a.ty, &st.type_) || is_id(&self.cache.ast, a.ty, &st.type_trait) {
                        // Special case: `arg: type = None` -> `arg: type = NoneType
                        default_value = Some(self.id_n(&st.none_type));
                    } else {
                        // Do nothing. NoneExpr will be handled later (we don't want it
                        // to be converted to Optional call yet.)
                    }
                } else if matches!(&self.cache.ast[d].kind, ExprKind::Id(i) if i.value == "__callsite_file__" || i.value == "__callsite_line__") {
                    // Resolved per call site in `get_call_default`.
                } else {
                    default_value = self.transform(default_value)?;
                }
            } else if let Some(d) = default_value {
                if !a.is_value() {
                    // Special case: generic defaults are evaluated as-is!
                    default_value = self.transform(Some(d))?;
                } else if self.is_per_call_default(d) {
                    // `__per_call_default__(expr)`: evaluated at each call site.
                } else {
                    default_value = Some(self.function_default_value(s, canonical_name, a, d, is_class_member, is_global)?);
                }
            }
            let mut np = Param::with_status(format!("{}{}", "*".repeat(stars), name), a.ty, default_value, a.status);
            np.kind = a.kind;
            np.status = a.status;
            // Note: the reference `Param(name, type, default, status, kind)` re-infers the status
            // only for status 0; mirror it through new_param.
            let np = {
                let hint = status_hint(np.status);
                self.cache.ast.new_param(SrcInfo::default(), np.name, np.ty, np.default_value, hint, np.kind)
            };
            args.push(np);

            // Add generics to the context
            if !a.is_value() {
                // Generic and static types
                let generic = self.instantiate_unbound();
                let typ_id = types.link(generic).unwrap().id;
                types.with_link_mut(generic, |l| l.generic_name = var_name.clone());
                let dd = self.cache.ast.clone_expr_opt(a.default_value, false);
                let def_type = self.transform(dd)?;
                let sk = self.cache.ast.get_static_generic(a.ty);
                if sk != LiteralKind::Runtime {
                    let val = TypeContext::add_var(self.cache, &self.ctx, &var_name, &name, generic, 0, Default::default());
                    val.borrow_mut().generic = true;
                    types.with_link_mut(generic, |l| l.static_kind = sk);
                    if let Some(dt) = def_type {
                        let t = self.extract_type_expr(dt);
                        types.with_link_mut(generic, |l| l.default_type = Some(t));
                    }
                } else {
                    let is_trait_var = a.ty.and_then(|t| self.cache.ast[t].kind.as_instantiate().cloned()).filter(|i| {
                        is_id(&self.cache.ast, i.expr, &st.type_trait)
                    });
                    if let Some(ti) = is_trait_var {
                        // Parse TraitVar
                        let front = ti.items.first().copied();
                        let le = self.transform_type(front, true)?.unwrap();
                        let l = self.ety(le);
                        let tr = types.link(l).and_then(|x| x.trait_);
                        let tr = match tr {
                            Some(tr) => tr,
                            None => types.new_type_trait(l),
                        };
                        types.with_link_mut(generic, |x| x.trait_ = Some(tr));
                    }
                    let val = TypeContext::add_type(self.cache, &self.ctx, &var_name, &name, generic, Default::default());
                    val.borrow_mut().generic = true;
                    if let Some(dt) = def_type {
                        let t = self.extract_type_expr(dt);
                        types.with_link_mut(generic, |l| l.default_type = Some(t));
                    }
                }
                let level = self.ctx.borrow().typecheck_level;
                let g = types.generalize(generic, level);
                if let Some(v) = var_name.strip_prefix('$') {
                    var_name = v.to_string();
                }
                let _ = &var_name;
                explicits.push(Generic::new(&name, Some(g), typ_id, types.static_kind(g)));
            }
        }

        // Prepare list of all generic types
        let mut parent_class = None;
        if is_class_member && self.cache.ast[s].attrs.has(Attr::Method) {
            // Get class generics (e.g., T for `class Cls[T]: def foo:`)
            let aa = self.cache.ast[s].attrs.get_str(Attr::ParentClass).unwrap().to_string();
            parent_class = self.extract_class_type_name(&aa);
        }
        // Add function generics
        let mut generics: Vec<TypeId> = explicits.iter().map(|i| self.extract_type_name(&i.name)).collect();

        // Handle function arguments
        // Base type: `Function[[args,...], ret]`
        let nargs = self.cache.ast[s].kind.as_function().unwrap().items.len() - explicits.len();
        let base_type = self.get_func_type_base(nargs)?;
        self.ctx.borrow_mut().typecheck_level += 1;
        let r = self.function_arg_types(s, &mut args, &mut generics, base_type, parent_class);
        self.ctx.borrow_mut().typecheck_level -= 1;
        let ret = r?;

        // Callable factory constraints can link a method generic to a class generic
        // (defaultdict.__new__). Generalizing the method must not mutate that outer slot.
        let mut parent_generics = BTreeSet::new();
        if let Some(pc) = parent_class {
            for u in types.get_unbounds(pc, true) {
                parent_generics.insert(types.link(u).unwrap().id);
            }
        }
        // Generalize generics and remove them from the context
        for g in &generics {
            for u in types.get_unbounds(*g, false) {
                if let Some(ub) = types.get_unbound(u) {
                    let id = types.link(ub).unwrap().id;
                    if !parent_generics.contains(&id) {
                        types.with_link_mut(ub, |l| l.kind = LinkKind::Generic);
                    }
                }
            }
        }

        // Parse function body
        let mut suite = None;
        let attrs = self.cache.ast[s].attrs.clone();
        if !attrs.has(Attr::Internal) && !attrs.has(Attr::C) {
            let fs = self.cache.ast[s].kind.as_function().unwrap().suite;
            if attrs.has(Attr::LLVM) {
                let code = fs.and_then(|x| self.cache.ast.first_in_block(x)).expect("invalid LLVM definition");
                suite = self.transform_llvm_definition(code)?;
            } else if attrs.has(Attr::C) {
                // Do nothing
            } else {
                suite = self.cache.ast.clone_stmt_opt(fs, false);
            }
        }
        Ok(FnSignature { args, suite, ret, explicits, base_type })
    }

    /// Whether a default is wrapped in `__per_call_default__(...)` (dataclass
    /// `default_factory`): such defaults are not hoisted to a global.
    pub(crate) fn is_per_call_default(&self, d: ExprId) -> bool {
        let Some(c) = self.cache.ast[d].kind.as_call() else { return false };
        c.expr.and_then(|f| self.cache.ast[f].kind.as_id()).is_some_and(|i| i.value.contains("__per_call_default__"))
    }

    /// Default value of a value parameter: stored in a global `.default.<fn>.<arg>`.
    fn function_default_value(
        &mut self,
        s: StmtId,
        canonical_name: &str,
        a: &Param,
        d: ExprId,
        is_class_member: bool,
        is_global: bool,
    ) -> Result<ExprId> {
        let types = self.types();
        let def_name = format!(".default.{}.{}", canonical_name, a.name);
        // class variable; go to the global context!
        let nctx = self.copy_ctx_without_base(is_class_member);
        let lhs = self.id_n(&def_name);
        let te = if a.is_value() { None } else { a.ty };
        let as_ = self.n_stmt(AssignStmt { lhs: Some(lhs), rhs: Some(d), type_expr: te, update: UpdateMode::Assign });
        if is_class_member {
            let l2 = self.id_n(&def_name);
            let decl = self.n_stmt(AssignStmt { lhs: Some(l2), rhs: None, type_expr: None, update: UpdateMode::Assign });
            let src = self.src;
            let h = {
                let mut tv = self.with_ctx(nctx.clone());
                tv.src = src;
                tv.tr_stmt(decl)?
            };
            let pre = self.preamble;
            self.cache.ast.suite_add_stmt(pre, Some(h));
            self.register_global(&def_name);
            self.cache.ast[as_].kind.as_assign_mut().unwrap().update = UpdateMode::Update;
        } else if is_global {
            self.register_global(&def_name);
        }
        let src = self.src;
        let das = {
            let mut tv = self.with_ctx(nctx.clone());
            tv.src = src;
            tv.tr_stmt(as_)?
        };
        self.prepend_stmts.borrow_mut().push(das);
        // Default unbounds must be allowed to pass through
        // to support cases such as `a = []`
        let f = self.force_find(&def_name);
        let ft = f.borrow().ty();
        for u in types.get_unbounds(ft, false) {
            if let Some(ub) = types.get_unbound(u) {
                types.with_link_mut(ub, |l| l.pass_through = true);
            }
            self.cache.ast[s].attrs.set(Attr::AllowPassThrough);
        }
        let id = self.id_n(&def_name);
        let mut tv = self.with_ctx(nctx);
        tv.src = src;
        tv.tr(id)
    }

    /// Argument type parsing and unification with the `Function[...]` base
    /// (runs at `typecheckLevel + 1`). Returns the transformed return type expression.
    fn function_arg_types(
        &mut self,
        s: StmtId,
        args: &mut [Param],
        generics: &mut Vec<TypeId>,
        base_type: TypeId,
        parent_class: Option<TypeId>,
    ) -> Result<Option<ExprId>> {
        let types = self.types();
        // Parse arguments to the context. Needs to be done after adding generics
        // to support cases like `foo(a: T, T: type)`
        for a in args.iter_mut() {
            a.ty = self.transform_type(a.ty, true)?;
        }

        // Unify base type generics with argument types. Add non-generic arguments to the
        // context. Delayed to prevent cases like `def foo(a, b=a)`
        let arg_type = types.get_class(self.extract_class_generic(base_type, 0)).unwrap();
        let items = self.cache.ast[s].kind.as_function().unwrap().items.clone();
        let mut aj = 0;
        for (ai, p) in items.iter().enumerate() {
            if !p.is_value() {
                continue;
            }
            // Mark __init__ generics as AllowPassThrough
            if let Some(pc) = parent_class {
                let attrs = &self.cache.ast[s].attrs;
                let fname = &self.cache.ast[s].kind.as_function().unwrap().name;
                if ai == 0 && attrs.has(Attr::HasSelf) && attrs.has(Attr::ClassDeduce) && fname == "__init__" {
                    let pname = types.class_name(pc).unwrap().to_string();
                    let deduce = self
                        .get_class_name(&pname)
                        .and_then(|c| c.ast)
                        .is_some_and(|a| self.cache.ast[a].attrs.has(Attr::ClassDeduce));
                    if deduce {
                        for u in types.get_unbounds(pc, true) {
                            self.cache.ast[s].attrs.set(Attr::AllowPassThrough);
                            types.with_link_mut(u, |l| l.pass_through = true);
                        }
                    }
                }
            }
            if p.ty.is_none() {
                generics.push(self.extract_class_generic(arg_type, aj));
            } else if p.name.starts_with('*') {
                // Special case: `*args: type` and `**kwargs: type`. Do not add this type to the
                // signature (as the real type is `Tuple[type, ...]`); it will be used during
                // call typechecking
                generics.push(self.extract_class_generic(arg_type, aj));
            } else {
                let g = self.extract_class_generic(arg_type, aj);
                let te = self.transform_type(p.ty, true)?.unwrap();
                let t = self.extract_type_expr(te);
                self.unify(g, t)?;
            }
            aj += 1;
        }

        // Parse the return type
        let r = self.cache.ast[s].kind.as_function().unwrap().ret;
        // Forward reference to a class defined later in the module (`-> B` before
        // `class B`): CPython does not evaluate it eagerly, so infer the return type.
        let ret = match self.transform_type(r, true) {
            Err(e) if r.is_some_and(|x| self.cache.ast[x].kind.as_id().is_some()) && e.to_string().contains("is not defined") => None,
            x => x?,
        };
        let ret_type = self.extract_class_generic(base_type, 1);
        if let Some(re) = ret {
            // Fix for functions returning Literal types
            let sk = self.cache.ast.get_static_generic(Some(re));
            if sk != LiteralKind::Runtime {
                types.with_class_mut(base_type, |c| c.generics[1].static_kind = sk);
            }
            let t = self.extract_type_expr(re);
            self.unify(ret_type, t)?;
        } else {
            let u = self.instantiate_unbound();
            generics.push(self.unify(ret_type, u)?);
        }
        Ok(ret)
    }

    /// Tail of `visit(FunctionStmt*)`: cache the canonical AST, build the FuncType,
    /// register overloads and apply remaining decorators.
    #[allow(clippy::too_many_arguments)]
    fn finish_function(
        &mut self,
        s: StmtId,
        orig_stmt: StmtId,
        decorators: Vec<Option<ExprId>>,
        sig: FnSignature,
        root_name: String,
        canonical_name: String,
        is_class_member: bool,
    ) -> Result<()> {
        let types = self.types();
        let module_path = self.ctx.borrow().module_info().path.clone();
        self.cache.ast[s].attrs.set_str(Attr::Module, module_path);

        // Make function AST and cache it for later realization
        let (is_async, stmt_name) = {
            let f = self.cache.ast[s].kind.as_function().unwrap();
            (f.async_, f.name.clone())
        };
        let src = self.src;
        let f = self.cache.ast.new_function(src, canonical_name.clone(), sig.ret, sig.args, sig.suite, vec![], is_async)?;
        let t = self.get_time();
        if t != 0 {
            self.cache.ast[f].attrs.set_int(Attr::ExprTime, t);
        }
        // cloneAttributesFrom
        self.cache.ast[f].attrs = self.cache.ast[s].attrs.clone();
        self.prepare_call_layout(f);
        self.register_func_ast(f);
        let (module, toplevel) = {
            let c = self.ctx.borrow();
            (c.module_path(), c.module().is_empty() && c.is_global())
        };
        self.cache.functions.insert(
            canonical_name.clone(),
            Function {
                module,
                root_name: root_name.clone(),
                ast: Some(f),
                ty: None,
                orig_ast: Some(orig_stmt),
                is_toplevel: toplevel,
                ..Default::default()
            },
        );
        self.set_done_stmt(f);
        let parent_class = self.cache.ast[s]
            .attrs
            .get_str(Attr::ParentClass)
            .map(|x| x.to_string())
            .and_then(|aa| self.extract_class_type_name(&aa));

        // Construct the type
        let parent = if is_class_member && self.cache.ast[s].attrs.has(Attr::Method) { parent_class } else { None };
        let func_typ = types.new_func(sig.base_type, f, sig.explicits, parent);
        types.set_src(func_typ, self.src);
        let level = self.ctx.borrow().typecheck_level;
        let func_typ = types.generalize(func_typ, level);
        self.cache.functions.get_mut(&canonical_name).unwrap().ty = Some(func_typ);

        if root_name == "Tuple.__new__" {
            let ngen = |c: &crate::cache::Cache, n: &str| {
                c.functions.get(n).and_then(|f| f.ty).map(|t| c.types.func_generics(t).len()).unwrap_or(0)
            };
            let mine = ngen(self.cache, &canonical_name);
            let ov = self.cache.overloads.get(&root_name).cloned().unwrap_or_default();
            // upper_bound: first element whose generics count is greater than ours
            let pos = ov.iter().position(|o| mine < ngen(self.cache, o)).unwrap_or(ov.len());
            self.cache.overloads.get_mut(&root_name).unwrap().insert(pos, canonical_name.clone());
        } else {
            self.cache.overloads.entry(root_name.clone()).or_default().push(canonical_name.clone());
        }

        TypeContext::add_func(self.cache, &self.ctx, &stmt_name, &root_name, func_typ, Default::default());
        TypeContext::add_func(self.cache, &self.ctx, &canonical_name, &canonical_name, func_typ, Default::default());
        if self.cache.ast[s].attrs.has(Attr::Overload) || is_class_member {
            self.ctx.borrow_mut().remove(&stmt_name); // first overload will handle it!
        }

        // Special method handling
        if is_class_member {
            let pc = parent_class.expect("no parent class");
            let um = self.get_unmangled_name(&canonical_name);
            let m = self.get_class_method(pc, &um);
            let mut found = false;
            for i in self.get_overloads(&m) {
                if i == canonical_name {
                    self.cache.functions.get_mut(&i).unwrap().ty = Some(func_typ);
                    found = true;
                    break;
                }
            }
            typhon_common::seqassert!(found, "cannot find matching class method for {}", canonical_name);
        } else {
            // Hack so that we can later use same helpers for class overloads
            self.cache.classes.entry(VAR_CLASS_TOPLEVEL.to_string()).or_default().methods.insert(stmt_name.clone(), root_name.clone());
        }

        // Ensure that functions with @C, @force_realize, and @export attributes can be
        // realized
        let attrs = &self.cache.ast[s].attrs;
        if (attrs.has(Attr::ForceRealize) || attrs.has(Attr::Export) || (attrs.has(Attr::C) && !attrs.has(Attr::CVarArg)))
            && !types.can_realize(func_typ)
        {
            let src = self.cache.ast[s].src;
            typhon_common::bail_e!(ErrorKind::FN_REALIZE_BUILTIN, src);
        }

        // Expression to be used if function binding is modified by captures or decorators
        let mut final_expr: Option<ExprId> = None;
        // Parse remaining decorators
        for d in decorators.iter().rev().flatten() {
            // Replace each decorator with `decorator(finalExpr)` in the reverse order
            final_expr = Some(match final_expr {
                Some(fe) => self.call_n(*d, vec![fe]),
                None => {
                    let id = self.id_n(&canonical_name);
                    self.cache.ast[id].attrs.set(Attr::ExprDoNotRealize);
                    self.call_n(*d, vec![id])
                }
            });
        }
        let Some(final_expr) = final_expr else {
            self.result_stmt = Some(f);
            return Ok(());
        };
        let lhs = self.id_n(&stmt_name);
        let a = self.n_stmt(AssignStmt { lhs: Some(lhs), rhs: Some(final_expr), type_expr: None, update: UpdateMode::Assign });
        if is_class_member {
            // class method decorator
            let nctx = self.copy_ctx_without_base(true); // global context
            let def_name = self.gen_canonical(&stmt_name, false, false);
            let l2 = self.id_n(&def_name);
            let decl = self.n_stmt(AssignStmt { lhs: Some(l2), rhs: None, type_expr: None, update: UpdateMode::Assign });
            let src = self.src;
            let h = {
                let mut tv = self.with_ctx(nctx);
                tv.src = src;
                tv.tr_stmt(decl)?
            };
            let pre = self.preamble;
            self.cache.ast.suite_add_stmt(pre, Some(h));
            self.register_global(&def_name);
            {
                let asg = self.cache.ast[a].kind.as_assign_mut().unwrap();
                asg.update = UpdateMode::Update;
            }
            self.cache.ast[lhs].kind = ExprKind::Id(IdExpr::new(def_name.clone()));

            let (items, ret) = {
                let fs = self.cache.ast[s].kind.as_function().unwrap();
                (fs.items.clone(), fs.ret)
            };
            let mut cargs = vec![];
            for arg in &items {
                if self.get_unmangled_name(&arg.name) == "Self" {
                    // Generic parameters (e.g. the implicit `Self`) are inferred.
                    continue;
                }
                let id = self.id_n(&arg.name);
                let e = if arg.name.starts_with("**") {
                    self.n_expr(KeywordStarExpr { expr: Some(id) })
                } else if arg.name.starts_with('*') {
                    self.n_expr(StarExpr { expr: Some(id) })
                } else {
                    id
                };
                cargs.push(e);
            }
            let callee = self.id_n(&def_name);
            let call = self.call_n(callee, cargs);
            let rs = self.n_stmt(ReturnStmt { expr: Some(call) });
            let body = self.n_stmt(SuiteStmt { items: vec![rs] });
            let ret = self.cache.ast.clone_expr_opt(ret, false);
            // The implicit `Self` generic belongs to the original method only; the
            // forwarding wrapper takes an unannotated `self` instead.
            let mut params: Vec<Param> = vec![];
            for p in &items {
                if self.get_unmangled_name(&p.name) == "Self" {
                    continue;
                }
                let mut q = self.cache.ast.clone_param(p, false);
                let self_ty = q.ty.and_then(|t| self.cache.ast[t].kind.as_id().map(|i| i.value.clone()));
                if q.name == "self" && self_ty.is_some_and(|n| self.get_unmangled_name(&n) == "Self") {
                    q.ty = None;
                }
                params.push(q);
            }
            let new_func = self.cache.ast.new_function(src, stmt_name, ret, params, Some(body), vec![], is_async)?;
            let t = self.get_time();
            if t != 0 {
                self.cache.ast[new_func].attrs.set_int(Attr::ExprTime, t);
            }
            let new_func = self.tr_stmt(new_func)?;
            let ta = self.tr_stmt(a)?;
            let inner = self.n_stmt(SuiteStmt { items: vec![ta, new_func] });
            self.result_stmt = Some(self.n_stmt(SuiteStmt { items: vec![f, inner] }));
        } else {
            let ta = self.tr_stmt(a)?;
            self.result_stmt = Some(self.n_stmt(SuiteStmt { items: vec![f, ta] }));
        }
        Ok(())
    }

    /// `transformLLVMDefinition(code)`: `{=expr}` blocks are parsed into the
    /// `referenced_types` suite and replaced with `{}`; other braces are escaped.
    pub fn transform_llvm_definition(&mut self, code_stmt: StmtId) -> Result<Option<StmtId>> {
        let code_expr = self.cache.ast[code_stmt].kind.as_expr().and_then(|e| e.expr);
        let code = code_expr.and_then(|e| self.cache.ast[e].kind.as_string().map(|s| s.value()));
        typhon_common::seqassert!(code.is_some(), "invalid LLVM definition");
        let mut code: Vec<u8> = code.unwrap_or_default().into_bytes();
        // Remove docstring (if any)
        let mut start = 0;
        while start < code.len() && code[start].is_ascii_whitespace() {
            start += 1;
        }
        if code[start..].starts_with(b"\"\"\"") {
            start += 3;
            let mut found = false;
            while start + 2 < code.len() {
                if code[start] == b'"' && code[start + 1] == b'"' && code[start + 2] == b'"' {
                    found = true;
                    start += 3;
                    break;
                }
                start += 1;
            }
            if found {
                code = code[start..].to_vec();
            }
        }

        let mut items: Vec<StmtId> = vec![];
        let mut final_code = String::new();
        let esc = |c: &[u8], st: usize, len: usize| {
            let s = String::from_utf8_lossy(&c[st..st + len]).into_owned();
            typhon_common::strings::escape_fstring_braces(&s, 0, s.len())
        };

        // Parse LLVM code and look for expression blocks that start with `{=`
        let (mut brace_count, mut brace_start) = (0usize, 0usize);
        let mut i = 0;
        while i < code.len() {
            if i + 1 < code.len() && code[i] == b'\\' && code[i + 1] == b'\n' {
                code[i] = b' ';
                code[i + 1] = b' ';
            }
            if i + 1 < code.len() && code[i] == b'{' && code[i + 1] == b'=' {
                if brace_start <= i {
                    final_code += &esc(&code, brace_start, i - brace_start);
                    final_code.push('{');
                }
                if brace_count == 0 {
                    brace_start = i + 2;
                    brace_count += 1;
                } else {
                    typhon_common::bail_e!(ErrorKind::FN_BAD_LLVM, self.src);
                }
            } else if brace_count > 0 && code[i] == b'}' {
                brace_count -= 1;
                let expr_code = String::from_utf8_lossy(&code[brace_start..i]).into_owned();
                let mut offset = self.src;
                offset.col += i as i32;
                let (expr, _) = crate::peg::parse_expr(&mut self.cache.ast, &expr_code, &offset)?;
                items.push(self.n_stmt(ExprStmt { expr: Some(expr) }));
                brace_start = i + 1;
                final_code.push('}');
            }
            i += 1;
        }
        if brace_count > 0 {
            typhon_common::bail_e!(ErrorKind::FN_BAD_LLVM, self.src);
        }
        if brace_start != code.len() {
            final_code += &esc(&code, brace_start, code.len() - brace_start);
        }
        let sx = self.n_expr(StringExpr::new(final_code, ""));
        let first = self.n_stmt(ExprStmt { expr: Some(sx) });
        items.insert(0, first);
        Ok(Some(self.n_stmt(SuiteStmt { items })))
    }

    /// `getDecorator(expr)`: `(isAttribute, attrName, decoratorName)`. The first member
    /// indicates if a decorator is actually an attribute (a function with `@__attribute__`).
    pub fn get_decorator(&mut self, e: ExprId) -> Result<(bool, String, String)> {
        let types = self.types();
        let c = self.cache.ast.clone_expr(e, false);
        let mut dt = self.tr(c)?;
        dt = self.get_head_expr(dt);
        let id_e = match self.cache.ast[dt].kind.as_call() {
            Some(call) => call.expr,
            None => Some(dt),
        };
        let Some(id_e) = id_e else { return Ok((false, String::new(), String::new())) };
        let Some(id_val) = self.cache.ast[id_e].kind.as_id().map(|i| i.value.clone()) else {
            return Ok((false, String::new(), String::new()));
        };
        let t = self.get_time();
        let ci = self.find(&id_val, t);
        if let Some(ci) = ci {
            let (is_func, ty) = {
                let c = ci.borrow();
                (c.is_func(&types), c.ty)
            };
            if is_func {
                let fty = types.get_func(ty.unwrap()).unwrap();
                let fname = types.func_name(fty);
                let mut f_ast = self.get_function_name(&fname).and_then(|f| f.ast);
                let mut found = self.get_function_name(&fname).is_some();
                if !found {
                    if let Some(o) = self.cache.overloads.get(&fname) {
                        if o.len() == 1 {
                            let o0 = o[0].clone();
                            found = self.get_function_name(&o0).is_some();
                            f_ast = self.get_function_name(&o0).and_then(|f| f.ast);
                        }
                    }
                }
                let Some(f_ast) = f_ast.filter(|_| found) else {
                    // Note: the reference dereferences a null `f` here (UB); treat as not found.
                    return Ok((false, String::new(), String::new()));
                };
                let is_attr = self.cache.ast[f_ast].attrs.has(Attr::Attribute);
                // Special case: Id to Call
                if is_attr && self.cache.ast[dt].kind.as_id().is_some() {
                    let call = self.call_n(dt, vec![]);
                    dt = self.tr(call)?;
                }
                let done = self.cache.ast[dt].done;
                return Ok((is_attr, fname, if done { id_val } else { String::new() }));
            }
        }
        Ok((false, String::new(), String::new()))
    }

    /// `getFuncTypeBase(nargs)`: `Function[Tuple[args...], ret]`.
    pub fn get_func_type_base(&mut self, nargs: usize) -> Result<TypeId> {
        let types = self.types();
        let f = self.get_stdlib_type(&stdlib().function)?;
        let base_type = self.instantiate_type(f, None)?;
        let bc = types.get_class(base_type).unwrap();
        let g = self.extract_class_generic(bc, 0);
        let tup = self.generate_tuple(nargs, false)?;
        let it = self.instantiate_type(tup, None)?;
        self.unify(g, it)?;
        Ok(bc)
    }
}
