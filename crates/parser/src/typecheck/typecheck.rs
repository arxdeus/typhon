//! Part 1: `apply`,
//! `loadStdLibrary`, `transform*`, node dispatch and the statements owned by
//! typecheck.cpp (`SuiteStmt`, `ExprStmt`, `StmtExpr`, `CommentStmt`,
//! `DirectiveStmt`). Helpers are in [`super::helpers`].

use std::collections::HashMap;

use typhon_common::fs::{ImportFile, ImportStatus};
use typhon_common::mangle::stdlib;
use typhon_common::{CompilerError, ErrorKind, Result, SrcInfo};

use super::{CtxRef, TypeContext, TypecheckVisitor};
use crate::ast::{AssignStmt, Attr, BoolExpr, ExprTag, IdExpr, IndexExpr, IntExpr, StmtTag, StringExpr, SuiteStmt, UpdateMode};
use crate::cache::{
    Cache, ImportModule, MAIN_IMPORT, MODULE_MAIN, STDLIB_IMPORT, STDLIB_INTERNAL_MODULE, VAR_CLASS_TOPLEVEL, VAR_USED_SUFFIX,
};
use crate::ids::{ExprId, StmtId};

impl<'a> TypecheckVisitor<'a> {
    fn define_stmt(&mut self, name: &str, value: &str) -> StmtId {
        let lhs = self.n_expr(IdExpr::new(name));
        let lit = self.n_expr(IdExpr::new("Literal"));
        let (rhs, kind) = if let Some(s) = value.strip_prefix("str:") {
            (self.n_expr(StringExpr::new(s, "")), "str")
        } else if value.starts_with("bool:") {
            (self.n_expr(BoolExpr { value: value == "bool:True" }), "bool")
        } else {
            let v = value.strip_prefix("int:").unwrap_or(value);
            (self.n_expr(IntExpr::parse(v, "")), "int")
        };
        let k = self.n_expr(IdExpr::new(kind));
        let ty = self.n_expr(IndexExpr { expr: Some(lit), index: Some(k) });
        self.n_stmt(AssignStmt { lhs: Some(lhs), rhs: Some(rhs), type_expr: Some(ty), update: UpdateMode::Assign })
    }

    /// `apply(cache, node, file, defines, earlyDefines, barebones)`: typecheck a main
    /// module, loading the standard library if needed.
    pub fn apply(
        cache: &'a mut Cache,
        node: StmtId,
        file: &str,
        defines: &HashMap<String, String>,
        early_defines: &HashMap<String, String>,
        barebones: bool,
    ) -> Result<StmtId> {
        let preamble = cache.ast.new_stmt(SuiteStmt::default(), SrcInfo::default());
        typhon_common::seqassert!(cache.module.is_some(), "cache's module is not set");
        // Load standard library if it has not been loaded
        if !cache.imports.contains_key(STDLIB_IMPORT) {
            Self::load_std_library(cache, preamble, early_defines, barebones)?;
        }
        // Set up the context and the cache
        let ctx = TypeContext::new(cache, file).into_ref();
        let mut m = ImportModule::new();
        m.update(MAIN_IMPORT, file, ctx.clone());
        cache.imports.insert(file.to_string(), m.clone());
        cache.imports.insert(MAIN_IMPORT.to_string(), m);
        ctx.borrow_mut().set_filename(file);
        ctx.borrow_mut().set_module(ImportFile { status: ImportStatus::Package, path: file.into(), module: MODULE_MAIN.into() });

        // Prepare the code
        let mut tv = TypecheckVisitor::new(cache, ctx.clone(), Some(preamble), None);
        let mut stmts = vec![];
        let mut ds: Vec<_> = defines.iter().collect();
        ds.sort();
        for (k, v) in ds {
            stmts.push(tv.define_stmt(k, v));
        }
        // Set up __name__ / __file__
        for (n, v) in [("__name__", MODULE_MAIN), ("__file__", file)] {
            let l = tv.n_expr(IdExpr::new(n));
            let r = tv.n_expr(StringExpr::new(v, ""));
            stmts.push(tv.n_stmt(AssignStmt { lhs: Some(l), rhs: Some(r), type_expr: None, update: UpdateMode::Assign }));
        }
        stmts.push(node);
        let suite = tv.n_stmt(SuiteStmt { items: stmts });
        if let Some(sc) = tv.cache.hooks.scoping.clone() {
            sc(tv.cache, suite, Some(ctx.clone()))?;
        }
        let n = tv.infer_types(Some(suite), true)?;
        let Some(n) = n else {
            return Err(CompilerError::Parser(tv.find_typecheck_errors(suite)));
        };
        let out = tv.n_stmt(SuiteStmt { items: vec![preamble, n] });
        if !tv.cache.options.as_ref().is_some_and(|o| o.pyext) {
            tv.prepare_vtables()?;
        }
        if !tv.cache.errors.is_empty() {
            return Err(CompilerError::Parser(tv.cache.errors.clone()));
        }
        Ok(out)
    }

    /// `loadStdLibrary(cache, preamble, earlyDefines, barebones)`.
    pub fn load_std_library(
        cache: &mut Cache,
        preamble: StmtId,
        early_defines: &HashMap<String, String>,
        barebones: bool,
    ) -> Result<()> {
        let stdlib_ctx = TypeContext::new(cache, STDLIB_IMPORT).into_ref();
        let path = crate::typecheck::helpers::get_import_file(cache, STDLIB_INTERNAL_MODULE, "", true);
        const INIT: &str = "__init__.py";
        let Some(mut path) = path.filter(|p| p.path.ends_with(INIT)) else {
            return Err(CompilerError::parser(ErrorKind::COMPILER_NO_STDLIB, SrcInfo::default(), &[]));
        };
        // Use __init_test__ for faster testing
        if barebones {
            path.path = format!("{}__init_test__.py", &path.path[..path.path.len() - INIT.len()]);
        }
        stdlib_ctx.borrow_mut().set_filename(&path.path);
        let mut m = ImportModule::new();
        m.update(STDLIB_IMPORT, &path.path, stdlib_ctx.clone());
        cache.imports.insert(path.path.clone(), m.clone());
        cache.imports.insert(STDLIB_IMPORT.to_string(), m);
        {
            let mut s = stdlib_ctx.borrow_mut();
            s.is_stdlib_loading = true;
            s.set_module(ImportFile { status: ImportStatus::Stdlib, path: path.path.clone(), module: "__init__".into() });
        }

        // 1. Core definitions
        cache.classes.insert(VAR_CLASS_TOPLEVEL.to_string(), Default::default());
        let parse_code = cache.hooks.parse_code.clone().ok_or_else(|| CompilerError::custom("no parser hook installed", SrcInfo::default()))?;
        let core = parse_code(cache, &path.path, "from internal.core import *", 0)?;
        if let Some(sc) = cache.hooks.scoping.clone() {
            sc(cache, core, None)?;
        }
        {
            let mut tv = TypecheckVisitor::new(cache, stdlib_ctx.clone(), Some(preamble), None);
            let Some(core) = tv.infer_types(Some(core), true)? else {
                return Err(CompilerError::Parser(tv.find_typecheck_errors(core)));
            };
            tv.cache.ast.suite_add_stmt(preamble, Some(core));

            // 2. Load early compile-time defines (for standard library)
            let mut ds: Vec<_> = early_defines.iter().collect();
            ds.sort();
            for (k, v) in ds {
                let s = tv.define_stmt(k, v);
                let def = tv.transform_stmt_opt(Some(s))?;
                tv.cache.ast.suite_add_stmt(preamble, def);
            }
        }

        // 3. Load stdlib
        let parse_file = cache.hooks.parse_file.clone().ok_or_else(|| CompilerError::custom("no parser hook installed", SrcInfo::default()))?;
        let std = parse_file(cache, &path.path)?;
        if let Some(sc) = cache.hooks.scoping.clone() {
            sc(cache, std, Some(stdlib_ctx.clone()))?;
        }
        let mut tv = TypecheckVisitor::new(cache, stdlib_ctx.clone(), Some(preamble), None);
        let Some(std) = tv.infer_types(Some(std), true)? else {
            return Err(CompilerError::Parser(tv.find_typecheck_errors(std)));
        };
        tv.cache.ast.suite_add_stmt(preamble, Some(std));
        stdlib_ctx.borrow_mut().is_stdlib_loading = false;
        Ok(())
    }

    /// `apply(ctx, node, file, sweepForceRealize)`: typecheck a node in an existing context
    /// (standard library must be loaded).
    pub fn apply_ctx(cache: &mut Cache, ctx: CtxRef, node: StmtId, file: &str, sweep_force_realize: bool) -> Result<StmtId> {
        ctx.borrow_mut().prepare_checkpoint();
        let old = ctx.borrow().filename();
        ctx.borrow_mut().set_filename(file);
        let preamble = cache.ast.new_stmt(SuiteStmt::default(), SrcInfo::default());
        let mut tv = TypecheckVisitor::new(cache, ctx.clone(), Some(preamble), None);
        let n = tv.infer_types(Some(node), sweep_force_realize);
        ctx.borrow_mut().set_filename(&old);
        let Some(n) = n? else {
            return Err(CompilerError::Parser(tv.find_typecheck_errors(node)));
        };
        if !tv.cache.errors.is_empty() {
            return Err(CompilerError::Parser(tv.cache.errors.clone()));
        }
        let suite = tv.cache.ast.new_stmt(SuiteStmt { items: vec![preamble] }, SrcInfo::default());
        tv.cache.ast.suite_add_stmt(suite, Some(n));
        Ok(suite)
    }
}

impl<'a> TypecheckVisitor<'a> {
    /// `transform(Expr*)` = `transform(expr, true)`.
    pub fn transform(&mut self, e: Option<ExprId>) -> Result<Option<ExprId>> {
        self.transform_allow(e, true)
    }
    /// Non-optional convenience for [`Self::transform`].
    pub fn tr(&mut self, e: ExprId) -> Result<ExprId> {
        Ok(self.transform(Some(e))?.expect("transform returned null"))
    }

    /// `transform(expr, allowTypes)`: typecheck an expression (with write-back of the
    /// replacement) and try to realize its type.
    pub fn transform_allow(&mut self, expr: Option<ExprId>, allow_types: bool) -> Result<Option<ExprId>> {
        let Some(mut expr) = expr else { return Ok(None) };
        if self.cache.ast[expr].ty.is_none() {
            let u = self.instantiate_unbound();
            self.set_type(expr, u);
        }
        if !self.cache.ast[expr].done {
            let src = self.cache.ast[expr].src;
            self.ctx.borrow_mut().base.push_node(src);
            let res = {
                let mut v = self.nested(true);
                v.src = src;
                v.visit_expr_node(expr).map(|_| v.result_expr)
            };
            self.ctx.borrow_mut().base.pop_node();
            if let Some(r) = res? {
                let attrs: Vec<_> = self.cache.ast[expr].attrs.iter().map(|(k, v)| (k, v.clone())).collect();
                for (k, v) in attrs {
                    if !self.cache.ast[r].attrs.has(k) {
                        self.cache.ast[r].attrs.set_value(k, v);
                    }
                }
                let orig = self.cache.ast[expr].orig_expr.unwrap_or(expr);
                self.cache.ast[r].orig_expr = Some(orig);
                expr = r;
                if self.cache.ast[expr].ty.is_none() {
                    let u = self.instantiate_unbound();
                    self.set_type(expr, u);
                }
            }
            if !allow_types && Self::is_type_expr(self.cache, expr) {
                let src = self.cache.ast[expr].src;
                typhon_common::bail_e!(ErrorKind::UNEXPECTED_TYPE, src, "type");
            }
            if self.cache.ast[expr].done {
                self.ctx.borrow_mut().changed_nodes += 1;
            }
        }
        if !self.cache.ast[expr].attrs.has(Attr::ExprDoNotRealize) {
            let t = self.ety(expr);
            if let Some(p) = self.realize(t)? {
                self.unify(t, p)?;
            }
        }
        typhon_common::log_flag!(TYPECHECK, "[expr] {:?}: {}{}", self.src, self.cache.ast.expr_to_string(expr, -1), if self.cache.ast[expr].done { "[done]" } else { "" });
        Ok(Some(expr))
    }

    /// `transformType(expr, simple)`: typecheck a type expression (`None` -> `NoneType`).
    pub fn transform_type(&mut self, expr: Option<ExprId>, simple: bool) -> Result<Option<ExprId>> {
        let st = stdlib();
        let Some(mut expr) = expr else { return Ok(None) };
        // In annotations, the builtin type describes arbitrary class values
        if crate::ast::is_id(&self.cache.ast, Some(expr), &st.type_) {
            let t = self.get_time();
            if let Some(b) = self.find(&st.type_, t) {
                if b.borrow().canonical_name == st.type_ {
                    let a = self.n_expr(IdExpr::new(st.type_.clone()));
                    let tt = self.get_stdlib_type(&st.type_)?;
                    let it = self.instantiate_type(tt, None)?;
                    self.set_type(a, it);
                    self.set_done_expr(a);
                    return Ok(Some(a));
                }
            }
        }
        if self.cache.ast[expr].kind.tag() == ExprTag::None {
            let src = self.cache.ast[expr].src;
            let ne = self.n_expr(IdExpr::new(st.none_type.clone()));
            self.cache.ast[ne].src = src;
            expr = ne;
        }
        let old = std::mem::replace(&mut self.ctx.borrow_mut().simple_types, simple);
        let r = self.transform(Some(expr));
        self.ctx.borrow_mut().simple_types = old;
        let Some(expr) = r? else { return Ok(None) };
        let types = self.types();
        let t = self.ety(expr);
        if types.static_kind(t) != crate::types::LiteralKind::Runtime {
        } else if Self::is_type_expr(self.cache, expr)
            || types.get_unbound(t).is_some_and(|u| {
                let l = types.link(u).unwrap();
                !l.generic_name.is_empty() || l.trait_.is_some()
            })
        {
            let it = self.instantiate_type(t, None)?;
            self.set_type(expr, it);
        } else {
            let src = self.cache.ast[expr].src;
            typhon_common::bail_e!(ErrorKind::EXPECTED_TYPE, src, "type");
        }
        Ok(Some(expr))
    }

    /// `transform(Stmt*)`: typecheck a statement; prepended statements are wrapped into a
    /// suite with the statement.
    pub fn transform_stmt_opt(&mut self, stmt: Option<StmtId>) -> Result<Option<StmtId>> {
        let Some(mut stmt) = stmt else { return Ok(None) };
        if self.cache.ast[stmt].done {
            return Ok(Some(stmt));
        }
        let src = self.cache.ast[stmt].src;
        self.ctx.borrow_mut().base.push_node(src);
        let time = self.cache.ast[stmt].attrs.get_int(Attr::ExprTime).unwrap_or(0);
        let old_time = std::mem::replace(&mut self.ctx.borrow_mut().time, time);
        let res = {
            let mut v = self.nested(false);
            v.src = src;
            v.visit_stmt_node(stmt).map(|_| (v.result_stmt, v.prepend_stmts.clone()))
        };
        self.ctx.borrow_mut().time = old_time;
        self.ctx.borrow_mut().base.pop_node();
        let (rs, prepend) = res?;
        if let Some(r) = rs {
            stmt = r;
        }
        let mut pre = prepend.borrow().clone();
        if !pre.is_empty() {
            pre.push(stmt);
            let done = pre.iter().all(|s| self.cache.ast[*s].done);
            stmt = self.n_stmt(SuiteStmt { items: pre });
            if done {
                self.set_done_stmt(stmt);
            }
        }
        if self.cache.ast[stmt].done {
            self.ctx.borrow_mut().changed_nodes += 1;
        }
        Ok(Some(stmt))
    }
    /// Non-optional convenience for [`Self::transform_stmt_opt`].
    pub fn tr_stmt(&mut self, s: StmtId) -> Result<StmtId> {
        Ok(self.transform_stmt_opt(Some(s))?.expect("transform returned null"))
    }

    /// `expr->accept(v)`: dispatch to the per-kind rule.
    pub fn visit_expr_node(&mut self, e: ExprId) -> Result<()> {
        match self.cache.ast[e].kind.tag() {
            ExprTag::None => self.visit_none(e),
            ExprTag::Bool => self.visit_bool(e),
            ExprTag::Int => self.visit_int(e),
            ExprTag::Float => self.visit_float(e),
            ExprTag::String => self.visit_string(e),
            ExprTag::Id => self.visit_id(e),
            ExprTag::Star => self.visit_star(e),
            ExprTag::KeywordStar => self.visit_keyword_star(e),
            ExprTag::Tuple => self.visit_tuple(e),
            ExprTag::List => self.visit_list(e),
            ExprTag::Set => self.visit_set(e),
            ExprTag::Dict => self.visit_dict(e),
            ExprTag::Generator => self.visit_generator(e),
            ExprTag::If => self.visit_if_expr(e),
            ExprTag::Unary => self.visit_unary(e),
            ExprTag::Binary => self.visit_binary(e),
            ExprTag::ChainBinary => self.visit_chain_binary(e),
            ExprTag::Pipe => self.visit_pipe(e),
            ExprTag::Index => self.visit_index(e),
            ExprTag::Call => self.visit_call(e),
            ExprTag::Dot => self.visit_dot(e),
            ExprTag::Slice => self.visit_slice(e),
            ExprTag::Ellipsis => self.visit_ellipsis(e),
            ExprTag::Lambda => self.visit_lambda(e),
            ExprTag::Yield => self.visit_yield_expr(e),
            ExprTag::Await => self.visit_await(e),
            ExprTag::Assign => self.visit_assign_expr(e),
            ExprTag::Range => self.visit_range(e),
            ExprTag::Stmt => self.visit_stmt_expr(e),
            ExprTag::Instantiate => self.visit_instantiate(e),
        }
    }

    /// `stmt->accept(v)`: dispatch to the per-kind rule (`ExceptStmt` is handled by
    /// `TryStmt`; visiting it directly is `defaultVisit` = internal error).
    pub fn visit_stmt_node(&mut self, s: StmtId) -> Result<()> {
        match self.cache.ast[s].kind.tag() {
            StmtTag::Suite => self.visit_suite(s),
            StmtTag::Break => self.visit_break(s),
            StmtTag::Continue => self.visit_continue(s),
            StmtTag::Expr => self.visit_expr_stmt(s),
            StmtTag::Assign => self.visit_assign(s),
            StmtTag::AssignMember => self.visit_assign_member(s),
            StmtTag::Del => self.visit_del(s),
            StmtTag::Print => self.visit_print(s),
            StmtTag::Return => self.visit_return(s),
            StmtTag::Yield => self.visit_yield(s),
            StmtTag::Assert => self.visit_assert(s),
            StmtTag::While => self.visit_while(s),
            StmtTag::For => self.visit_for(s),
            StmtTag::If => self.visit_if(s),
            StmtTag::Match => self.visit_match(s),
            StmtTag::Import => self.visit_import(s),
            StmtTag::Try => self.visit_try(s),
            StmtTag::Global => self.visit_global(s),
            StmtTag::Throw => self.visit_throw(s),
            StmtTag::Function => self.visit_function(s),
            StmtTag::Class => self.visit_class(s),
            StmtTag::YieldFrom => self.visit_yield_from(s),
            StmtTag::With => self.visit_with(s),
            StmtTag::Directive => self.visit_directive(s),
            StmtTag::Comment => self.visit_comment(s),
            StmtTag::Except => Err(CompilerError::custom(
                format!("unexpected AST node {}", self.cache.ast.stmt_to_string(s, -1)),
                self.src,
            )),
        }
    }

    /// `visit(StmtExpr*)`.
    pub fn visit_stmt_expr(&mut self, e: ExprId) -> Result<()> {
        let se = self.cache.ast[e].kind.as_stmt().unwrap().clone();
        let mut done = true;
        let mut items = vec![];
        for s in se.items {
            let s = self.tr_stmt(s)?;
            done &= self.cache.ast[s].done;
            items.push(s);
        }
        let x = self.transform(se.expr)?;
        if let Some(k) = self.cache.ast[e].kind.as_stmt_mut() {
            k.items = items;
            k.expr = x;
        }
        let x = x.unwrap();
        let (a, b) = (self.ety(e), self.ety(x));
        self.unify(a, b)?;
        if done && self.cache.ast[x].done {
            self.set_done_expr(e);
        }
        Ok(())
    }

    /// `visit(SuiteStmt*)`.
    pub fn visit_suite(&mut self, s: StmtId) -> Result<()> {
        let mut prepend = vec![];
        if let Some(b) = self.cache.ast[s].attrs.get_bindings().cloned() {
            for (n, bd) in &b.bindings {
                let l = self.n_expr(IdExpr::new(n.clone()));
                prepend.push(self.n_stmt(AssignStmt { lhs: Some(l), rhs: None, type_expr: None, update: UpdateMode::Assign }));
                if bd.count > 0 {
                    let l = self.n_expr(IdExpr::new(format!("{n}{VAR_USED_SUFFIX}")));
                    let r = self.n_expr(BoolExpr { value: false });
                    prepend.push(self.n_stmt(AssignStmt { lhs: Some(l), rhs: Some(r), type_expr: None, update: UpdateMode::Assign }));
                }
            }
            self.cache.ast[s].attrs.erase(Attr::Bindings);
        }
        if !prepend.is_empty() {
            if let Some(k) = self.cache.ast[s].kind.as_suite_mut() {
                prepend.extend(k.items.drain(..));
                k.items = prepend;
            }
        }
        let lr = self.cache.ast[s].attrs.get_key_value(Attr::LocalRenames).cloned();
        if let Some(lr) = &lr {
            for (k, v) in lr {
                let item = self.force_find(v);
                self.ctx.borrow_mut().add(k, item);
            }
        }
        let items = self.cache.ast[s].kind.as_suite().unwrap().items.clone();
        let mut stmts = vec![];
        let mut done = true;
        for it in items {
            if self.ctx.borrow().return_early {
                // If returnEarly is set (e.g., in the function) ignore the rest
                break;
            }
            if let Some(x) = self.transform_stmt_opt(Some(it))? {
                match self.cache.ast[x].kind.as_suite() {
                    None => {
                        done &= self.cache.ast[x].done;
                        stmts.push(x);
                    }
                    Some(sx) => {
                        for ss in sx.items.clone() {
                            done &= self.cache.ast[ss].done;
                            stmts.push(ss);
                        }
                    }
                }
            }
        }
        self.cache.ast[s].kind.as_suite_mut().unwrap().items = stmts;
        if let Some(lr) = &lr {
            for k in lr.keys() {
                self.ctx.borrow_mut().remove(k);
            }
        }
        if done {
            self.set_done_stmt(s);
        }
        Ok(())
    }

    /// `visit(ExprStmt*)`.
    pub fn visit_expr_stmt(&mut self, s: StmtId) -> Result<()> {
        let e = self.cache.ast[s].kind.as_expr().unwrap().expr;
        let e = self.transform(e)?;
        self.cache.ast[s].kind.as_expr_mut().unwrap().expr = e;
        if e.is_some_and(|e| self.cache.ast[e].done) {
            self.set_done_stmt(s);
        }
        Ok(())
    }

    /// `visit(CommentStmt*)`.
    pub fn visit_comment(&mut self, s: StmtId) -> Result<()> {
        self.set_done_stmt(s);
        Ok(())
    }

    /// `visit(DirectiveStmt*)`.
    pub fn visit_directive(&mut self, s: StmtId) -> Result<()> {
        let d = self.cache.ast[s].kind.as_directive().unwrap().clone();
        let src = self.cache.ast[s].src;
        let msg = if d.key == "auto_python" {
            self.ctx.borrow_mut().auto_python = d.value == "1";
            format!("directive '{}' = {}", d.key, d.value == "1")
        } else {
            format!("unknown directive '{}'", d.key)
        };
        typhon_common::diag::compilation_warning(&msg, src.file.as_str(), src.line, src.col, 0, -1, false, Default::default());
        self.set_done_stmt(s);
        Ok(())
    }
}
