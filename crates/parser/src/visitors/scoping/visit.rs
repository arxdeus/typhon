
use rustc_hash::FxHashMap;
use typhon_common::ErrorKind;
use typhon_common::srcinfo::SrcInfo;

use super::{Context, Node, ScopeBlock, ScopingVisitor};
use crate::ast::{AstArena, Attr, ExprKind, FunctionStmt, GeneratorKind, StmtKind, SuiteStmt, is_id};
use crate::ids::{ExprId, StmtId};
use crate::visitors::{AstCtx, Visitor};

macro_rules! check {
    ($e:expr) => {
        if !$e {
            return;
        }
    };
}

impl AstCtx for ScopingVisitor<'_> {
    fn ast(&self) -> &AstArena {
        &self.cache.ast
    }
    fn ast_mut(&mut self) -> &mut AstArena {
        &mut self.cache.ast
    }
}

impl ScopingVisitor<'_> {
    /// Run `f` with a different context (nested `ScopingVisitor v; v.ctx = c;`).
    /// The nested visitor's errors are kept separate (returned with the context).
    pub(crate) fn in_ctx<R>(
        &mut self,
        ctx: &mut Context,
        errs: &mut typhon_common::ParserErrors,
        f: impl FnOnce(&mut Self) -> R,
    ) -> R {
        std::mem::swap(&mut self.ctx, ctx);
        std::mem::swap(&mut self.errors, errs);
        let saved = std::mem::take(&mut self.src);
        let r = f(self);
        self.src = saved;
        std::mem::swap(&mut self.ctx, ctx);
        std::mem::swap(&mut self.errors, errs);
        r
    }

    pub(crate) fn set_root(&mut self, r: Option<Node>) -> Option<Node> {
        std::mem::replace(&mut self.ctx.root, r)
    }

    /// `transformPattern`.
    pub(crate) fn transform_pattern(&mut self, pattern: Option<ExprId>) -> bool {
        let Some(p) = pattern else { return self.transform_e(None) };
        match self.cache.ast[p].kind.clone() {
            ExprKind::Id(id) => {
                if id.value == "_" {
                    return true;
                }
                let o = self.set_root(Some(Node::Expr(p)));
                let r = self.transform_e(Some(p));
                self.ctx.root = o;
                r
            }
            ExprKind::Tuple(t) => t.items.iter().all(|i| self.transform_pattern(Some(*i))),
            ExprKind::List(t) => t.items.iter().all(|i| self.transform_pattern(Some(*i))),
            ExprKind::Binary(b) if b.op == "|" || b.op == "||" => {
                self.transform_pattern(b.lexpr) && self.transform_pattern(b.rexpr)
            }
            ExprKind::Assign(a) => self.transform_pattern(a.expr) && self.transform_pattern(a.var),
            ExprKind::Call(c) => {
                if !self.transform_e(c.expr) {
                    return false;
                }
                for arg in &c.items {
                    let ok = if arg.name.is_empty() { self.transform_e(arg.value) } else { self.transform_pattern(arg.value) };
                    if !ok {
                        return false;
                    }
                }
                true
            }
            ExprKind::Star(s) => self.transform_pattern(s.expr),
            _ => self.transform_e(Some(p)),
        }
    }

    /// Builds the `BindingsAttr` of a nested function/lambda context.
    fn child_bindings(&self, c: &Context) -> crate::ast::BindingsAttr {
        let mut b = crate::ast::BindingsAttr::default();
        b.captures = c.captures.iter().map(|(k, v)| (k.clone(), *v)).collect();
        b
    }
}

impl Visitor for ScopingVisitor<'_> {
    fn transform_expr(&mut self, id: ExprId) -> ExprId {
        self.transform_e(Some(id));
        id
    }
    fn transform_stmt(&mut self, id: StmtId) -> StmtId {
        self.transform_s(Some(id));
        id
    }

    fn visit_id(&mut self, id: ExprId) {
        let mut value = self.cache.ast[id].kind.as_id().unwrap().value.clone();
        if self.ctx.root.is_some() && self.ctx.temp_scope {
            let t = self.cache.get_temporary_var(&value);
            self.ctx.renames.last_mut().unwrap().insert(value.clone(), t);
        }
        for r in self.ctx.renames.iter().rev() {
            if let Some(v) = r.get(&value) {
                value = v.clone();
                self.cache.ast[id].kind.as_id_mut().unwrap().value = value.clone();
                break;
            }
        }
        if self.ctx.root.is_some() {
            self.cache.ast[id].attrs.set(Attr::ExprNoUndefCheck);
        }
        let src = self.cache.ast[id].src.clone();
        let adding = self.ctx.root.is_some();
        // C++: `visitName(value, ctx->root != nullptr, expr, ...)`: the binding node is the
        // IdExpr itself, so domination attributes land on the assignment lhs.
        self.visit_name(&value, adding, Some(Node::Expr(id)), &src);
    }

    fn visit_dot(&mut self, id: ExprId) {
        let o = self.set_root(None); // to handle a.x, y = b
        crate::visitors::walk_expr(self, id);
        self.ctx.root = o;
        let d = self.cache.ast[id].kind.as_dot().unwrap().clone();
        if !self.ctx.class_deduce.0.is_empty() && is_id(&self.cache.ast, d.expr, &self.ctx.class_deduce.0) {
            self.ctx.class_deduce.1.insert(d.member);
        }
    }

    fn visit_index(&mut self, id: ExprId) {
        let o = self.set_root(None); // to handle a[x], y = b
        crate::visitors::walk_expr(self, id);
        self.ctx.root = o;
    }

    fn visit_string(&mut self, id: ExprId) {
        self.scope_string(id);
    }

    fn visit_generator(&mut self, id: ExprId) {
        let o = std::mem::replace(&mut self.ctx.temp_scope, true);
        self.ctx.renames.push(FxHashMap::default());
        let s = self.cache.ast.generator_final_suite(id);
        let ok = self.transform_s(s);
        // Note: the reference implementation `CHECK` skips the restore on failure (the SetInScope destructor still
        // restores `tempScope`); renames are popped unconditionally here.
        self.ctx.renames.pop();
        self.ctx.temp_scope = o;
        let _ = (ok, GeneratorKind::Generator);
    }

    fn visit_if_expr(&mut self, id: ExprId) {
        let e = self.cache.ast[id].kind.as_if().unwrap().clone();
        check!(self.transform_e(e.cond));
        check!(self.transform_scope_e(e.ifexpr));
        check!(self.transform_scope_e(e.elsexpr));
    }

    fn visit_binary(&mut self, id: ExprId) {
        let e = self.cache.ast[id].kind.as_binary().unwrap().clone();
        check!(self.transform_e(e.lexpr));
        if e.op == "&&" || e.op == "||" {
            check!(self.transform_scope_e(e.rexpr));
        } else {
            check!(self.transform_e(e.rexpr));
        }
    }

    fn visit_assign_expr(&mut self, id: ExprId) {
        let e = self.cache.ast[id].kind.as_assign().unwrap().clone();
        typhon_common::seqassert!(
            e.var.is_some_and(|v| self.cache.ast[v].kind.as_id().is_some()),
            "only simple assignment expression are supported"
        );
        let ot = std::mem::replace(&mut self.ctx.temp_scope, false);
        if self.transform_e(e.expr) {
            let o = self.set_root(Some(Node::Expr(id)));
            self.transform_e(e.var);
            self.ctx.root = o;
        }
        self.ctx.temp_scope = ot;
    }

    fn visit_lambda(&mut self, id: ExprId) {
        let l = self.cache.ast[id].kind.as_lambda().unwrap().clone();
        // Note: the reference implementation uses stack-allocated dummy `FunctionStmt("lambda")` and `SuiteStmt`;
        // here they are allocated (and then abandoned) in the arena.
        let src = self.src.clone();
        let f = self.cache.ast.alloc_stmt(crate::ast::Stmt::new(
            StmtKind::Function(FunctionStmt { name: "lambda".into(), ..Default::default() }),
            src.clone(),
        ));
        let suite = self.cache.ast.alloc_stmt(crate::ast::Stmt::new(StmtKind::Suite(SuiteStmt::default()), src));
        let mut c = Context { function_scope: Some(f), renames: self.ctx.renames.clone(), ..Default::default() };
        let mut errs = Default::default();
        c.scope.push(ScopeBlock { id: 0, suite: None, seen_vars: None });
        for a in &l.items {
            let (_, n) = a.name_with_stars();
            self.in_ctx(&mut c, &mut errs, |v| v.visit_name(&n, true, Some(Node::Expr(id)), &a.src));
            if a.default_value.is_some() {
                check!(self.transform_e(a.default_value));
            }
        }
        c.scope.pop();
        c.scope.push(ScopeBlock { id: 0, suite: Some(suite), seen_vars: None });
        self.in_ctx(&mut c, &mut errs, |v| {
            v.transform_e(l.expr);
            v.process_child_captures();
        });
        c.scope.pop();
        self.errors.append(&errs);
        if !self.can_continue() {
            return;
        }
        let mut b = self.child_bindings(&c);
        for (k, v) in &c.captures {
            self.ctx.child_captures.entry(k.clone()).or_insert(*v);
        }
        for (u, v) in &c.map {
            b.bindings.insert(u.clone(), crate::ast::Binding { name: u.clone(), count: v.len(), is_nonlocal: false });
        }
        self.cache.ast[id].attrs.set_value(Attr::Bindings, crate::ast::AttrValue::Bindings(b));
    }

    fn visit_yield_expr(&mut self, _id: ExprId) {
        if let Some(f) = self.ctx.function_scope {
            self.cache.ast[f].attrs.set(Attr::IsGenerator);
        }
    }

    fn visit_assign(&mut self, id: StmtId) {
        let s = self.cache.ast[id].kind.as_assign().unwrap().clone();
        check!(self.transform_e(s.rhs));
        check!(self.transform_e(s.type_expr));
        let o = self.set_root(Some(Node::Stmt(id)));
        if s.rhs.is_none() && s.type_expr.is_some() && !self.ctx.in_class {
            // Reuse conditional-binding used flags so annotation-only locals and globals
            // raise NameError when read before assignment. Class fields are checked later.
            self.push_block(None, None);
            self.transform_e(s.lhs);
            self.pop_block();
        } else {
            self.transform_e(s.lhs);
        }
        self.ctx.root = o;
    }

    fn visit_if(&mut self, id: StmtId) {
        let s = self.cache.ast[id].kind.as_if().unwrap().clone();
        check!(self.transform_e(s.cond));
        check!(self.transform_scope_s(s.if_suite));
        check!(self.transform_scope_s(s.else_suite));
    }

    fn visit_match(&mut self, id: StmtId) {
        let s = self.cache.ast[id].kind.as_match().unwrap().clone();
        check!(self.transform_e(s.expr));
        for m in &s.items {
            // match_capture_scope: captures define names for both the guard and body.
            self.push_block(m.suite, None);
            let ok = self.transform_pattern(m.pattern) && self.transform_e(m.guard) && self.transform_s(m.suite);
            self.pop_block();
            check!(ok);
        }
    }

    fn visit_while(&mut self, id: StmtId) {
        let s = self.cache.ast[id].kind.as_while().unwrap().clone();
        let mut seen: Vec<String> = vec![];
        self.push_block(s.suite, None);
        self.ctx.scope.last_mut().unwrap().seen_vars = Some(Default::default());
        let ok = self.transform_e(s.cond);
        if ok {
            seen.extend(self.ctx.scope.last().unwrap().seen_vars.as_ref().unwrap().iter().cloned());
        }
        self.pop_block();
        check!(ok);
        for var in seen.clone() {
            self.find_dominating_binding(&var, true);
        }
        self.push_block(s.suite, None);
        self.ctx.scope.last_mut().unwrap().seen_vars = Some(Default::default());
        let ok = self.transform_s(s.suite);
        if ok {
            for v in self.ctx.scope.last().unwrap().seen_vars.as_ref().unwrap() {
                if !seen.contains(v) {
                    seen.push(v.clone());
                }
            }
        }
        self.pop_block();
        check!(ok);
        for var in seen {
            self.find_dominating_binding(&var, true);
        }
        check!(self.transform_scope_s(s.else_suite));
    }

    fn visit_for(&mut self, id: StmtId) {
        let s = self.cache.ast[id].kind.as_for().unwrap().clone();
        check!(self.transform_e(s.iter));
        check!(self.transform_e(s.decorator));
        for a in &s.omp_args {
            check!(self.transform_e(a.value));
        }
        self.push_block(s.suite, None);
        self.ctx.scope.last_mut().unwrap().seen_vars = Some(Default::default());
        let o = self.set_root(Some(Node::Stmt(id)));
        let ok = self.transform_e(s.var);
        self.ctx.root = o;
        if !ok {
            self.pop_block();
            return;
        }
        let seen_def = self.ctx.scope.last_mut().unwrap().seen_vars.replace(Default::default()).unwrap();
        let ok = self.transform_s(s.suite);
        let seen = self.ctx.scope.last_mut().unwrap().seen_vars.take().unwrap();
        self.pop_block();
        check!(ok);
        for var in seen {
            if !seen_def.contains(&var) {
                self.find_dominating_binding(&var, true);
            }
        }
        check!(self.transform_scope_s(s.else_suite));
    }

    fn visit_import(&mut self, id: StmtId) {
        self.scope_import(id);
    }

    fn visit_try(&mut self, id: StmtId) {
        let s = self.cache.ast[id].kind.as_try().unwrap().clone();
        check!(self.transform_scope_s(s.suite));
        for &a in &s.items {
            let e = self.cache.ast[a].kind.as_except().unwrap().clone();
            check!(self.transform_e(e.exc));
            self.push_block(e.suite, None);
            if !e.var.is_empty() {
                let new_name = self.cache.get_temporary_var(&e.var);
                let mut m = FxHashMap::default();
                m.insert(e.var.clone(), new_name.clone());
                self.ctx.renames.push(m);
                self.cache.ast[a].kind.as_except_mut().unwrap().var = new_name.clone();
                // Note: the reference implementation dereferences `getException()` unconditionally.
                let src = e.exc.map(|x| self.cache.ast[x].src.clone()).unwrap_or_else(|| self.cache.ast[a].src.clone());
                self.visit_name(&new_name, true, Some(Node::Stmt(a)), &src);
            }
            let ok = self.transform_s(e.suite);
            if !e.var.is_empty() && ok {
                self.ctx.renames.pop();
            }
            self.pop_block();
            check!(ok);
        }
        check!(self.transform_scope_s(s.else_suite));
        check!(self.transform_s(s.finally));
    }

    fn visit_del(&mut self, id: StmtId) {
        let e = self.cache.ast[id].kind.as_del().unwrap().expr;
        check!(self.transform_e(e));
    }

    /// Process `global` statements.
    fn visit_global(&mut self, id: StmtId) {
        let g = self.cache.ast[id].kind.as_global().unwrap().clone();
        let src: SrcInfo = self.cache.ast[id].src.clone();
        if self.ctx.function_scope.is_none() {
            let kw = if g.non_local { "nonlocal" } else { "global" };
            return self.add_error(ErrorKind::FN_OUTSIDE_ERROR, &src, &[&kw]);
        }
        if self.ctx.map.contains_key(&g.var) || self.ctx.captures.contains_key(&g.var) {
            return self.add_error(ErrorKind::FN_GLOBAL_ASSIGNED, &src, &[&g.var]);
        }
        self.visit_name(&g.var, true, Some(Node::Stmt(id)), &src);
        // No shadowing od global/nonlocal allowed
        self.find_dominating_binding(&g.var, false);
        self.ctx.captures.insert(
            g.var,
            if g.non_local { crate::ast::CaptureType::Nonlocal } else { crate::ast::CaptureType::Global },
        );
    }

    fn visit_yield(&mut self, id: StmtId) {
        if let Some(f) = self.ctx.function_scope {
            self.cache.ast[f].attrs.set(Attr::IsGenerator);
        }
        let e = self.cache.ast[id].kind.as_yield().unwrap().expr;
        check!(self.transform_e(e));
    }

    fn visit_function(&mut self, id: StmtId) {
        self.scope_function(id);
    }

    fn visit_class(&mut self, id: StmtId) {
        self.scope_class(id);
    }

    fn visit_with(&mut self, id: StmtId) {
        let s = self.cache.ast[id].kind.as_with().unwrap().clone();
        let src = self.cache.ast[id].src.clone();
        self.push_block(s.suite, None);
        let mut ok = true;
        for (i, item) in s.items.iter().enumerate() {
            ok = self.transform_e(Some(*item));
            if !ok {
                break;
            }
            if !s.vars[i].is_empty() {
                self.visit_name(&s.vars[i], true, Some(Node::Stmt(id)), &src);
            }
        }
        if ok {
            self.transform_s(s.suite);
        }
        self.pop_block();
    }
}
