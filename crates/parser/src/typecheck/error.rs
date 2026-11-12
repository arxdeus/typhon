
use typhon_common::mangle::{self, stdlib};
use typhon_common::{ErrorKind, Result};

use super::{TypeContext, TypecheckVisitor};
use crate::ast::{
    AssignStmt, Attr, AwaitExpr, BoolExpr, BreakStmt, CallArg, CallExpr, DotExpr, ExceptStmt, ExprStmt, IdExpr, IfStmt,
    IntExpr, StmtExpr, StringExpr, SuiteStmt, ThrowStmt, TryStmt, UnaryExpr, UpdateMode, WhileStmt,
};
use crate::cache::VAR_USED_SUFFIX;
use crate::ids::{ExprId, StmtId};
use crate::matching::{M, match_expr};

/// Small AST builders shared by the error/loops/import lane (`N<T>(...)` shorthands).
impl<'a> TypecheckVisitor<'a> {
    pub(super) fn xm_id(&mut self, s: &str) -> ExprId {
        self.n_expr(IdExpr::new(s))
    }
    pub(super) fn xm_bool(&mut self, b: bool) -> ExprId {
        self.n_expr(BoolExpr { value: b })
    }
    pub(super) fn xm_str(&mut self, s: &str) -> ExprId {
        self.n_expr(StringExpr::new(s, ""))
    }
    pub(super) fn xm_int(&mut self, i: i64) -> ExprId {
        self.n_expr(IntExpr::new(i))
    }
    pub(super) fn xm_dot(&mut self, e: ExprId, m: &str) -> ExprId {
        self.n_expr(DotExpr { expr: Some(e), member: m.into() })
    }
    pub(super) fn xm_call(&mut self, f: ExprId, args: &[ExprId]) -> ExprId {
        let items = args.iter().map(|&a| self.cache.ast.call_arg("", Some(a))).collect();
        self.xm_call_args(f, items)
    }
    pub(super) fn xm_call_args(&mut self, f: ExprId, items: Vec<CallArg>) -> ExprId {
        self.n_expr(CallExpr { expr: Some(f), items, ordered: false, partial: false })
    }
    pub(super) fn xm_assign(&mut self, lhs: ExprId, rhs: Option<ExprId>, ty: Option<ExprId>, update: UpdateMode) -> StmtId {
        self.n_stmt(AssignStmt { lhs: Some(lhs), rhs, type_expr: ty, update })
    }
    /// `N<SuiteStmt>(a, b, ...)` (null items dropped).
    pub(super) fn xm_suite(&mut self, items: &[Option<StmtId>]) -> StmtId {
        self.n_stmt(SuiteStmt { items: items.iter().flatten().copied().collect() })
    }
    pub(super) fn xm_if(&mut self, cond: ExprId, if_suite: Option<StmtId>, else_suite: Option<StmtId>) -> StmtId {
        self.n_stmt(IfStmt { cond: Some(cond), if_suite, else_suite })
    }
    pub(super) fn xm_expr_stmt(&mut self, e: ExprId) -> StmtId {
        self.n_stmt(ExprStmt { expr: Some(e) })
    }
    pub(super) fn xm_not(&mut self, e: ExprId) -> ExprId {
        self.n_expr(UnaryExpr { op: "!".into(), expr: Some(e) })
    }
    /// `SuiteStmt::wrap(transform(s))` with `ctx->blockLevel++/--` around it.
    pub(super) fn xm_block(&mut self, s: Option<StmtId>) -> Result<Option<StmtId>> {
        self.ctx.borrow_mut().block_level += 1;
        let r = self.transform_stmt_opt(s);
        self.ctx.borrow_mut().block_level -= 1;
        let r = r?;
        Ok(self.cache.ast.wrap_suite(r))
    }
    pub(super) fn xm_done(&self, s: Option<StmtId>) -> bool {
        s.is_some_and(|s| self.cache.ast[s].done)
    }
}

impl<'a> TypecheckVisitor<'a> {
    /// `visit(AssertStmt*)`.
    pub fn visit_assert(&mut self, s: StmtId) -> Result<()> {
        let a = self.cache.ast[s].kind.as_assert().unwrap().clone();
        let mut msg = self.xm_str("");
        if let Some(m) = a.message {
            let f = self.xm_id("str");
            msg = self.xm_call(f, &[m]);
        }
        let test = {
            let c = self.ctx.borrow();
            c.in_function() && c.base_ref().func.is_some_and(|f| self.cache.ast[f].attrs.has(Attr::Test))
        };
        let src = self.cache.ast[s].src;
        let ii = self.xm_id("__internal__");
        let d = self.xm_dot(ii, if test { "seq_assert_test" } else { "seq_assert" });
        let file = self.xm_str(src.file.as_str());
        let line = self.xm_int(src.line as i64);
        let ex = self.xm_call(d, &[file, line, msg]);
        let cond = self.n_expr(UnaryExpr { op: "!".into(), expr: a.expr });
        let body = if test { self.xm_expr_stmt(ex) } else { self.n_stmt(ThrowStmt { expr: Some(ex), from: None, transformed: false }) };
        let i = self.xm_if(cond, Some(body), None);
        self.result_stmt = Some(self.tr_stmt(i)?);
        Ok(())
    }
}

impl<'a> TypecheckVisitor<'a> {
    /// `visit(TryStmt*)`. Python exceptions (`pyobj` / `PyError`) are collected into a
    /// single `except PyError` handler with a `while True` dispatch loop.
    pub fn visit_try(&mut self, s: StmtId) -> Result<()> {
        let st = stdlib();
        let suite = self.cache.ast[s].kind.as_try().unwrap().suite;
        let suite = self.xm_block(suite)?;
        self.cache.ast[s].kind.as_try_mut().unwrap().suite = suite;

        let mut catches: Vec<StmtId> = vec![];
        let t = self.xm_bool(true);
        let py_suite = self.xm_suite(&[]);
        let py_catch = self.n_stmt(WhileStmt { cond: Some(t), suite: Some(py_suite), else_suite: None, goto_var: String::new() });
        // `new_stmt` may re-wrap the suite slot: read it back.
        let py_suite = self.cache.ast[py_catch].kind.as_while().unwrap().suite.unwrap();

        let mut done = self.xm_done(suite);
        let items = self.cache.ast[s].kind.as_try().unwrap().items.clone();
        for c in items {
            let mut val = None;
            let cvar = self.cache.ast[c].kind.as_except().unwrap().var.clone();
            if !cvar.is_empty() {
                let dom = self.cache.ast[c].attrs.has(Attr::ExprDominated);
                let dom_used = self.cache.ast[c].attrs.has(Attr::ExprDominatedUsed);
                let v = if !dom && !dom_used {
                    let un = self.get_unmangled_name(&cvar);
                    let canon = self.ctx.borrow().generate_canonical_name(self.cache, &cvar, false, false);
                    let u = self.instantiate_unbound();
                    let time = self.get_time();
                    TypeContext::add_var(self.cache, &self.ctx, &un, &canon, u, time, Default::default())
                } else if dom_used {
                    let v = self.force_find(&cvar);
                    self.cache.ast[c].attrs.erase(Attr::ExprDominatedUsed);
                    self.cache.ast[c].attrs.set(Attr::ExprDominated);
                    let un = self.get_unmangled_name(&cvar);
                    let l = self.xm_id(&format!("{un}{VAR_USED_SUFFIX}"));
                    let r = self.xm_bool(true);
                    let a = self.xm_assign(l, Some(r), None, UpdateMode::Update);
                    let cs = self.cache.ast[c].kind.as_except().unwrap().suite;
                    let ns = self.xm_suite(&[Some(a), cs]);
                    self.cache.ast[c].kind.as_except_mut().unwrap().suite = Some(ns);
                    v
                } else {
                    self.force_find(&cvar)
                };
                let cn = v.borrow().canonical_name.clone();
                self.cache.ast[c].kind.as_except_mut().unwrap().var = cn;
                val = Some(v);
            }
            let exc = self.cache.ast[c].kind.as_except().unwrap().exc;
            let exc = self.transform(exc)?;
            self.cache.ast[c].kind.as_except_mut().unwrap().exc = exc;
            let ecls = exc.and_then(|e| self.extract_class_type_expr(e));
            let is_pyobj = exc.is_some() && ecls.is_some_and(|t| self.cache.types.is(t, "pyobj"));
            let is_pyerr = !is_pyobj && exc.is_some() && ecls.is_some_and(|t| self.cache.types.is(t, &st.py_error));
            if is_pyobj || is_pyerr {
                if !self.cache.ast[s].attrs.has(Attr::TryPyVar) {
                    let v = self.get_temporary_var("pyexc");
                    self.cache.ast[s].attrs.set_str(Attr::TryPyVar, v);
                }
                let py_var = self.cache.ast[s].attrs.get_str(Attr::TryPyVar).unwrap().to_string();
                let cvar = self.cache.ast[c].kind.as_except().unwrap().var.clone();
                if is_pyobj {
                    // Transform python.Error exceptions
                    if !cvar.is_empty() {
                        let l = self.xm_id(&cvar);
                        let pv = self.xm_id(&py_var);
                        let r = self.xm_dot(pv, "pytype");
                        let a = self.xm_assign(l, Some(r), None, UpdateMode::Assign);
                        let cs = self.cache.ast[c].kind.as_except().unwrap().suite;
                        let ns = self.xm_suite(&[Some(a), cs]);
                        self.cache.ast[c].kind.as_except_mut().unwrap().suite = Some(ns);
                    }
                    let isi = self.xm_id("isinstance");
                    let pv = self.xm_id(&py_var);
                    let pt = self.xm_dot(pv, "pytype");
                    let cond = self.xm_call(isi, &[pt, exc.unwrap()]);
                    let cs = self.cache.ast[c].kind.as_except().unwrap().suite;
                    let brk = self.n_stmt(BreakStmt);
                    let body = self.xm_suite(&[cs, Some(brk)]);
                    let i = self.xm_if(cond, Some(body), None);
                    let w = self.cache.ast.wrap_suite(Some(i));
                    self.cache.ast[c].kind.as_except_mut().unwrap().suite = w;
                } else {
                    // Transform PyExc exceptions
                    if !cvar.is_empty() {
                        let l = self.xm_id(&cvar);
                        let r = self.xm_id(&py_var);
                        let a = self.xm_assign(l, Some(r), None, UpdateMode::Assign);
                        let cs = self.cache.ast[c].kind.as_except().unwrap().suite;
                        let ns = self.xm_suite(&[Some(a), cs]);
                        self.cache.ast[c].kind.as_except_mut().unwrap().suite = Some(ns);
                    }
                    let cs = self.cache.ast[c].kind.as_except().unwrap().suite;
                    let brk = self.n_stmt(BreakStmt);
                    let ns = self.xm_suite(&[cs, Some(brk)]);
                    self.cache.ast[c].kind.as_except_mut().unwrap().suite = Some(ns);
                }
                let cs = self.cache.ast[c].kind.as_except().unwrap().suite;
                self.cache.ast.suite_add_stmt(py_suite, cs);
            } else {
                // Handle all other exceptions
                let exc = self.transform_type(exc, false)?;
                self.cache.ast[c].kind.as_except_mut().unwrap().exc = exc;
                if let Some(e) = exc {
                    let t = self.extract_class_type_expr(e).expect("bad exception class");
                    let mut ok = false;
                    for p in self.get_mro(t)? {
                        if self.cache.types.is(p, &st.base_exception) {
                            ok = true;
                            break;
                        }
                    }
                    if !ok {
                        let src = self.cache.ast[e].src;
                        typhon_common::bail_e!(ErrorKind::CATCH_EXCEPTION_TYPE, src, self.cache.types.pretty_string(t));
                    }
                    if let Some(v) = &val {
                        let vt = v.borrow().ty();
                        let et = self.extract_type_expr(e);
                        self.unify(vt, et)?;
                    }
                }
                let cs = self.cache.ast[c].kind.as_except().unwrap().suite;
                let cs = self.xm_block(cs)?;
                self.cache.ast[c].kind.as_except_mut().unwrap().suite = cs;
                done &= exc.is_none_or(|e| self.cache.ast[e].done) && self.xm_done(cs);
                catches.push(c);
            }
        }
        if !self.cache.ast[py_suite].kind.as_suite().unwrap().items.is_empty() {
            // Process PyError catches
            let py_var = self.cache.ast[s].attrs.get_str(Attr::TryPyVar).unwrap().to_string();
            let exc = self.xm_id(&st.py_error);
            let th = self.n_stmt(ThrowStmt { expr: None, from: None, transformed: false });
            self.cache.ast.suite_add_stmt(py_suite, Some(th));
            let exc = self.transform_type(Some(exc), false)?;
            let c = self.n_stmt(ExceptStmt { var: py_var.clone(), exc, suite: Some(py_catch) });
            let et = self.extract_type_expr(exc.unwrap());
            let time = self.get_time();
            TypeContext::add_var(self.cache, &self.ctx, &py_var, &py_var, et, time, Default::default());
            let cs = self.cache.ast[c].kind.as_except().unwrap().suite;
            let cs = self.xm_block(cs)?;
            self.cache.ast[c].kind.as_except_mut().unwrap().suite = cs;
            done &= exc.is_none_or(|e| self.cache.ast[e].done) && self.xm_done(cs);
            catches.push(c);
        }
        self.cache.ast[s].kind.as_try_mut().unwrap().items = catches;

        let els = self.cache.ast[s].kind.as_try().unwrap().else_suite;
        if els.is_some() {
            let e = self.xm_block(els)?;
            self.cache.ast[s].kind.as_try_mut().unwrap().else_suite = e;
            done &= self.xm_done(e);
        }
        let fin = self.cache.ast[s].kind.as_try().unwrap().finally;
        if fin.is_some() {
            let f = self.xm_block(fin)?;
            self.cache.ast[s].kind.as_try_mut().unwrap().finally = f;
            done &= self.xm_done(f);
        }
        if done {
            self.set_done_stmt(s);
        }
        Ok(())
    }
}

impl<'a> TypecheckVisitor<'a> {
    /// `visit(ThrowStmt*)`: `raise exc` -> `raise BaseException._set_header(exc, ...)`.
    pub fn visit_throw(&mut self, s: StmtId) -> Result<()> {
        let th = self.cache.ast[s].kind.as_throw().unwrap().clone();
        let Some(expr) = th.expr else {
            self.set_done_stmt(s);
            return Ok(());
        };
        let expr = self.tr(expr)?;
        self.cache.ast[s].kind.as_throw_mut().unwrap().expr = Some(expr);
        let set_header = mangle::mangled_method("std.internal.types.error", "BaseException", "_set_header", 0, 0);
        let head = self.get_head_expr(expr);
        let pat = M::expr(crate::ast::ExprTag::Call, vec![M::expr(crate::ast::ExprTag::Id, vec![M::str(&set_header)]), M::ignore()]);
        if !self.cache.ast[s].attrs.has(Attr::Internal) && !match_expr(&self.cache.ast, Some(head), &pat) {
            let mut exception = expr;
            let mut cause = match th.from {
                Some(f) => self.tr(f)?,
                None => {
                    let n = self.xm_id(&stdlib().none_type);
                    let c = self.xm_call(n, &[]);
                    self.tr(c)?
                }
            };
            let mut setup: Vec<StmtId> = vec![];
            // `raise Error from cause()` evaluates cause() before constructing Error().
            if Self::is_type_expr(self.cache, exception) && th.from.is_some() {
                let en = self.get_temporary_var("exception");
                let cn = self.get_temporary_var("cause");
                let l = self.xm_id(&en);
                let a = self.xm_assign(l, Some(exception), None, UpdateMode::Assign);
                setup.push(self.tr_stmt(a)?);
                let l = self.xm_id(&cn);
                let a = self.xm_assign(l, Some(cause), None, UpdateMode::Assign);
                setup.push(self.tr_stmt(a)?);
                let i = self.xm_id(&en);
                exception = self.tr(i)?;
                let i = self.xm_id(&cn);
                cause = self.tr(i)?;
            }
            if Self::is_type_expr(self.cache, exception) {
                let c = self.xm_call(exception, &[]);
                exception = self.tr(c)?;
            }
            if Self::is_type_expr(self.cache, cause) {
                let c = self.xm_call(cause, &[]);
                cause = self.tr(c)?;
            }
            let src = self.cache.ast[s].src;
            let base_name = self.ctx.borrow().base_ref().name.to_string();
            let f = self.xm_id(&set_header);
            let bn = self.xm_str(&base_name);
            let file = self.xm_str(src.file.as_str());
            let line = self.xm_int(src.line as i64);
            let col = self.xm_int(src.col as i64);
            let header = self.xm_call(f, &[exception, bn, file, line, col, cause]);
            let e = if setup.is_empty() { header } else { self.n_expr(StmtExpr { items: setup, expr: Some(header) }) };
            let e = self.tr(e)?;
            self.cache.ast[s].kind.as_throw_mut().unwrap().expr = Some(e);
        }
        let e = self.cache.ast[s].kind.as_throw().unwrap().expr.unwrap();
        if self.cache.ast[e].done {
            self.set_done_stmt(s);
        }
        Ok(())
    }

    /// `visit(WithStmt*)`: see the implementation for the full details(manager saved, exit
    /// callable saved, legacy-none / full cleanup branches, nested in reverse order).
    pub fn visit_with(&mut self, s: StmtId) -> Result<()> {
        let w = self.cache.ast[s].kind.as_with().unwrap().clone();
        typhon_common::seqassert!(!w.items.is_empty(), "stmt->items is empty");
        let is_async = w.async_;
        let st = stdlib();
        let em = |m: &str| mangle::mangled_method("std.internal.types.error", "BaseException", m, 0, 0);

        let mut content: Vec<StmtId> = vec![];
        for i in (0..w.items.len()).rev() {
            let var = self.get_temporary_var("with");
            let l = self.xm_id(&var);
            let as_ = self.xm_assign(l, Some(w.items[i]), None, UpdateMode::Assign);
            let exit_var = self.get_temporary_var("exit");
            let exceptional = self.get_temporary_var("with_exception");
            let exception = self.get_temporary_var("exception");
            let exit_result = self.get_temporary_var("exit_result");
            let l = self.xm_id(&exit_var);
            let v = self.xm_id(&var);
            let r = self.xm_dot(v, if is_async { "__aexit__" } else { "__exit__" });
            let save_exit = self.xm_assign(l, Some(r), None, UpdateMode::Assign);

            let v = self.xm_id(&var);
            let d = self.xm_dot(v, if is_async { "__aenter__" } else { "__enter__" });
            let mut enter = self.xm_call(d, &[]);
            if is_async {
                enter = self.n_expr(AwaitExpr { expr: Some(enter), transformed: false });
            }
            let call_exit = |tv: &mut Self, et: ExprId, ev: ExprId| -> ExprId {
                let f = tv.xm_id(&em(if is_async { "_context_aexit" } else { "_context_exit" }));
                let ev_ = tv.xm_id(&exit_var);
                let c = tv.xm_call(f, &[ev_, et, ev]);
                if is_async { tv.n_expr(AwaitExpr { expr: Some(c), transformed: false }) } else { c }
            };
            let mut body = if !content.is_empty() {
                self.n_stmt(SuiteStmt { items: std::mem::take(&mut content) })
            } else {
                self.cache.ast.clone_stmt(w.suite.expect("with without suite"), false)
            };
            let mut entry = self.xm_expr_stmt(enter);
            if !w.vars[i].is_empty() {
                let entered = self.get_temporary_var("entered");
                let l = self.xm_id(&entered);
                entry = self.xm_assign(l, Some(enter), None, UpdateMode::Assign);
                let mode = if self.cache.ast[w.items[i]].attrs.has(Attr::ExprDominated) { UpdateMode::Update } else { UpdateMode::Assign };
                let l = self.xm_id(&w.vars[i]);
                let r = self.xm_id(&entered);
                let a = self.xm_assign(l, Some(r), None, mode);
                body = self.xm_suite(&[Some(a), Some(body)]);
            }
            // An awaited exit can suspend outside the original native catch frame.
            let ex = self.xm_id(&exception);
            let rethrow = self.n_stmt(ThrowStmt { expr: Some(ex), from: None, transformed: false });
            self.cache.ast[rethrow].attrs.set(Attr::Internal);
            let l = self.xm_id(&exceptional);
            let t = self.xm_bool(true);
            let a1 = self.xm_assign(l, Some(t), None, UpdateMode::Update);
            let gt = self.xm_id(&mangle::func("std.internal.types.rtti", "get_typeinfo"));
            let ex = self.xm_id(&exception);
            let gtc = self.xm_call(gt, &[ex]);
            let ex = self.xm_id(&exception);
            let ce = call_exit(self, gtc, ex);
            let l = self.xm_id(&exit_result);
            let a2 = self.xm_assign(l, Some(ce), None, UpdateMode::Assign);
            let er = self.xm_id(&exit_result);
            let ner = self.xm_not(er);
            let i3 = self.xm_if(ner, Some(rethrow), None);
            let hsuite = self.xm_suite(&[Some(a1), Some(a2), Some(i3)]);
            let be = self.xm_id(&st.base_exception);
            let handler = self.n_stmt(ExceptStmt { var: exception.clone(), exc: Some(be), suite: Some(hsuite) });

            let l = self.xm_id(&exceptional);
            let f = self.xm_bool(false);
            let init = self.xm_assign(l, Some(f), None, UpdateMode::Assign);
            let n1 = self.xm_id(&st.none_type);
            let n1 = self.xm_call(n1, &[]);
            let n2 = self.xm_id(&st.none_type);
            let n2 = self.xm_call(n2, &[]);
            let ce = call_exit(self, n1, n2);
            let ces = self.xm_expr_stmt(ce);
            let exn = self.xm_id(&exceptional);
            let nexn = self.xm_not(exn);
            let fi = self.xm_if(nexn, Some(ces), None);
            let fin = self.xm_suite(&[Some(fi)]);
            let tr = self.n_stmt(TryStmt { items: vec![handler], suite: Some(body), else_suite: None, finally: Some(fin) });
            let full_cleanup = self.xm_suite(&[Some(init), Some(tr)]);

            let ev = self.xm_id(&exit_var);
            let mut legacy_exit = self.xm_call(ev, &[]);
            if is_async {
                legacy_exit = self.n_expr(AwaitExpr { expr: Some(legacy_exit), transformed: false });
            }
            let les = self.xm_expr_stmt(legacy_exit);
            let lfin = self.xm_suite(&[Some(les)]);
            // Note: the reference shares the `body` node between both branches (AST is a DAG
            // there); Rust shares the same StmtId as well.
            let legacy_cleanup = self.n_stmt(TryStmt { items: vec![], suite: Some(body), else_suite: None, finally: Some(lfin) });
            let f = self.xm_id(&em("_context_exit_is_legacy_none"));
            let ev = self.xm_id(&exit_var);
            let ia = self.xm_bool(is_async);
            let legacy_none = self.xm_call(f, &[ev, ia]);
            let ifs = self.xm_if(legacy_none, Some(legacy_cleanup), Some(full_cleanup));
            content = vec![as_, save_exit, entry, ifs];
        }
        let su = self.n_stmt(SuiteStmt { items: content });
        self.result_stmt = Some(self.tr_stmt(su)?);
        Ok(())
    }
}
