
use std::collections::BTreeMap;

use typhon_common::mangle::{self, stdlib};
use typhon_common::strings::replace_first;
use typhon_common::{ErrorKind, Result};

use super::TypecheckVisitor;
use crate::ast::{
    AssignStmt, Attr, AttrValue, BinaryExpr, BoolExpr, BreakStmt, CallExpr, ExceptStmt, ExprKind, ExprStmt, ExprTag,
    IfStmt, IndexExpr, InstantiateExpr, IntExpr, NoneExpr, StmtExpr, StringExpr, SuiteStmt, TryStmt, UpdateMode, WhileStmt,
};
use crate::ids::{ExprId, StmtId};
use crate::types::LiteralKind;

impl<'a> TypecheckVisitor<'a> {
    /// `evaluateStaticCondition`: `Some(isTrue)` if the static condition can be evaluated.
    fn evaluate_static_condition(&self, cond: ExprId) -> Option<bool> {
        let types = self.types();
        let t = self.ety(cond);
        typhon_common::seqassert!(types.static_kind(t) != LiteralKind::Runtime, "not a static condition");
        if !types.can_realize(t) {
            return None;
        }
        Some(if let Some(s) = types.get_str_static(t) {
            !s.is_empty()
        } else if let Some(i) = types.get_int_static(t) {
            i != 0
        } else {
            types.get_bool_static(t).unwrap_or(false)
        })
    }
    fn c_if(&mut self, cond: ExprId, i: Option<StmtId>, e: Option<StmtId>) -> StmtId {
        self.n_stmt(IfStmt { cond: Some(cond), if_suite: i, else_suite: e })
    }
    fn c_bin(&mut self, l: ExprId, op: &str, r: ExprId) -> ExprId {
        self.n_expr(BinaryExpr { op: op.into(), lexpr: Some(l), rexpr: Some(r), in_place: false })
    }
    fn c_assign(&mut self, l: ExprId, r: ExprId) -> StmtId {
        self.n_stmt(AssignStmt { lhs: Some(l), rhs: Some(r), type_expr: None, update: UpdateMode::Assign })
    }
    fn c_suite(&mut self, items: Vec<StmtId>) -> StmtId {
        self.n_stmt(SuiteStmt { items })
    }
    fn c_clone(&mut self, e: ExprId) -> ExprId {
        self.cache.ast.clone_expr(e, false)
    }

    /// `visit(RangeExpr*)`: only allowed in `MatchStmt`.
    pub fn visit_range(&mut self, e: ExprId) -> Result<()> {
        let src = self.cache.ast[e].src;
        typhon_common::bail_e!(ErrorKind::UNEXPECTED_TYPE, src, "range");
    }

    /// `visit(IfExpr*)`: evaluate static if blocks if possible. Also wrap conditional
    /// expressions to match each other.
    pub fn visit_if_expr(&mut self, e: ExprId) -> Result<()> {
        let st = stdlib();
        let types = self.types();
        let ie = self.cache.ast[e].kind.as_if().unwrap().clone();
        let b = self.get_stdlib_type(&st.bool_)?;
        self.cache.ast[ie.cond.unwrap()].expected_type = Some(b);
        let narrow = self.none_check_narrowing(ie.cond.unwrap());
        let mut cond = self.transform(ie.cond)?.unwrap();
        self.cache.ast[e].kind.as_if_mut().unwrap().cond = Some(cond);

        // Static if evaluation
        if types.static_kind(self.ety(cond)) != LiteralKind::Runtime {
            if let Some(is_true) = self.evaluate_static_condition(cond) {
                typhon_common::log_flag!(TYPECHECK, "[static::cond] {:?}: {}", self.src, is_true);
                let mut s = if is_true { ie.ifexpr.unwrap() } else { ie.elsexpr.unwrap() };
                if self.has_side_effect(cond) {
                    let es = self.n_stmt(ExprStmt { expr: Some(cond) });
                    s = self.n_expr(StmtExpr { items: vec![es], expr: Some(s) });
                }
                self.result_expr = self.transform(Some(s))?;
            }
            if let Some(r) = self.result_expr {
                let (a, b) = (self.ety(e), self.ety(r));
                self.unify(a, b)?;
            } else if let Some(u) = types.get_unbound(self.ety(e)) {
                types.with_link_mut(u, |l| l.static_kind = LiteralKind::Int); // determine later!
            }
            return Ok(());
        }

        let raw_id = |s: &Self, x: Option<ExprId>| x.and_then(|x| s.cache.ast[x].kind.as_id().map(|i| i.value.clone()));
        let (if_raw, el_raw) = (raw_id(self, ie.ifexpr), raw_id(self, ie.elsexpr));
        let mut ifx = self.transform(ie.ifexpr)?.unwrap();
        let mut elx = self.transform(ie.elsexpr)?.unwrap();
        // `x if opt is None else opt` / `opt if opt is not None else x`: the branch
        // where `opt` is known to be non-None uses the unwrapped value (mypy narrowing).
        if let Some((name, non_none_in_if)) = narrow {
            let (br, raw) = if non_none_in_if { (&mut ifx, if_raw) } else { (&mut elx, el_raw) };
            let is_var = raw.is_some_and(|r| r == name);
            if is_var && types.is(self.ety(*br), &st.optional) {
                let d = self.dot_n(*br, "__val__");
                let c = self.call_n(d, vec![]);
                *br = self.tr(c)?;
            }
        }

        self.wrap_expr(&mut cond, b, None, true)?;
        // Add wrappers and unify both sides
        if types.get_static(self.ety(ifx)).is_some() {
            let t = types.non_static_type(self.ety(ifx)).unwrap();
            self.set_type(ifx, t);
        }
        if types.get_static(self.ety(elx)).is_some() {
            let t = types.non_static_type(self.ety(elx)).unwrap();
            self.set_type(elx, t);
        }
        // Select the same nullable target as explicit returns, then coerce each branch once.
        let nullable = self.join_nullable_types(self.ety(ifx), self.ety(elx))?;
        let t = nullable.unwrap_or(self.ety(ifx));
        self.wrap_expr(&mut elx, t, None, false)?;
        let t = nullable.unwrap_or(self.ety(elx));
        self.wrap_expr(&mut ifx, t, None, false)?;
        self.op_if_set(e, cond, ifx, elx);

        // Types not compatible! Check if an union can be made
        if types.unify(self.ety(ifx), self.ety(elx), None, self)? < 0
            && self.cache.ast[e].expected_type.is_some_and(|t| types.is(t, &st.union))
        {
            if !types.can_realize(self.ety(ifx)) || !types.can_realize(self.ety(elx)) {
                return Ok(());
            }
            let u = self.id_n(&st.union);
            let a = self.id_n(&types.realized_name(self.ety(ifx)));
            let b2 = self.id_n(&types.realized_name(self.ety(elx)));
            let t = self.n_expr(InstantiateExpr { expr: Some(u), items: vec![a, b2] });
            let c = self.call_n(t, vec![ifx]);
            ifx = self.tr(c)?;
            let t2 = self.c_clone(t);
            let c = self.call_n(t2, vec![elx]);
            elx = self.tr(c)?;
            self.op_if_set(e, cond, ifx, elx);
        }

        let et = self.ety(e);
        let (it, lt) = (self.ety(ifx), self.ety(elx));
        self.unify(et, it)?;
        self.unify(et, lt)?;
        let a = &self.cache.ast;
        if a[cond].done && a[ifx].done && a[elx].done {
            self.set_done_expr(e);
        }
        Ok(())
    }
    /// Recognise `v is None` / `v is not None` (either operand order) on a plain
    /// variable. Returns the variable's name and whether the *if* branch is the one
    /// where `v` is not None.
    fn none_check_narrowing(&self, cond: ExprId) -> Option<(String, bool)> {
        let a = &self.cache.ast;
        let bx = a[cond].kind.as_binary()?;
        let non_none_in_if = match bx.op.as_str() {
            "is" => false,
            "is not" => true,
            _ => return None,
        };
        let (l, r) = (bx.lexpr?, bx.rexpr?);
        let var = if a[r].kind.as_none().is_some() {
            l
        } else if a[l].kind.as_none().is_some() {
            r
        } else {
            return None;
        };
        let name = a[var].kind.as_id()?.value.clone();
        Some((name, non_none_in_if))
    }
    fn op_if_set(&mut self, e: ExprId, c: ExprId, i: ExprId, el: ExprId) {
        let k = self.cache.ast[e].kind.as_if_mut().unwrap();
        k.cond = Some(c);
        k.ifexpr = Some(i);
        k.elsexpr = Some(el);
    }

    /// `visit(IfStmt*)`: evaluate static if blocks if possible.
    pub fn visit_if(&mut self, s: StmtId) -> Result<()> {
        let st = stdlib();
        let types = self.types();
        let is = self.cache.ast[s].kind.as_if().unwrap().clone();
        let b = self.get_stdlib_type(&st.bool_)?;
        self.cache.ast[is.cond.unwrap()].expected_type = Some(b);
        // Note: the reference reads `(*ci)[1]` after call reordering dropped the generic `T`
        // slot from `items` (stale read past `size()` that yields the original type arg);
        // keep that original argument explicitly.
        let orig_type_arg = self.cache.ast[is.cond.unwrap()].kind.as_call().and_then(|c| c.items.get(1)).and_then(|a| a.value);
        let mut cond = self.transform(is.cond)?.unwrap();
        self.cache.ast[s].kind.as_if_mut().unwrap().cond = Some(cond);

        if let Some(ci) = self.cache.ast[cond].kind.as_call().cloned() {
            let fname = ci
                .expr
                .filter(|x| self.cache.ast[*x].kind.tag() == ExprTag::Id)
                .and_then(|x| self.expr_type(x))
                .and_then(|t| types.get_func(t))
                .map(|f| types.func_name(f));
            if let Some(fname) = fname.filter(|n| {
                n.starts_with(&mangle::mangled_method("", "RTTIType", "_isinstance", 0, 0))
                    || n.starts_with(&mangle::mangled_method("", "Any", "_isinstance", 0, 0))
            }) {
                let a0 = ci.items[0].value.unwrap();
                if let Some(arg) = self.cache.ast[a0].kind.as_id().map(|i| i.value.clone()) {
                    // isinstance(a, T) ->
                    // { if (c := isinstance(a, T)): i = getinstance(a, T)) } ; c
                    let name = self.get_unmangled_name(&arg);
                    let tmp_name = {
                        let ctx = self.ctx.clone();
                        let c = ctx.borrow();
                        c.generate_canonical_name(self.cache, &name, false, false)
                    };
                    let cond_name = self.get_temporary_var("cond");
                    let if_suite = is.if_suite.unwrap();
                    let mut renames: BTreeMap<String, String> =
                        self.cache.ast[if_suite].attrs.get_key_value(Attr::LocalRenames).cloned().unwrap_or_default();
                    // Previous aliases of an Optional-unwrapped match subject must be
                    // narrowed along with that subject before accessing subclass fields.
                    for target in renames.values_mut() {
                        if *target == name || *target == arg {
                            *target = tmp_name.clone();
                        }
                    }
                    renames.insert(name, tmp_name.clone());
                    self.cache.ast[if_suite].attrs.set_value(Attr::LocalRenames, AttrValue::KeyValue(renames));
                    let cn = self.id_n(&cond_name);
                    let a1 = self.c_assign(cn, cond);
                    let tn = self.id_n(&tmp_name);
                    let gf = self.id_n(&replace_first(&fname, "_isinstance", "_getinstance"));
                    let gc = self.call_n(gf, vec![a0, ci.items.get(1).and_then(|a| a.value).or(orig_type_arg).unwrap()]);
                    let a2 = self.c_assign(tn, gc);
                    let inner = self.c_suite(vec![a2, if_suite]);
                    let cn = self.id_n(&cond_name);
                    let ifs = self.c_if(cn, Some(inner), is.else_suite);
                    let suite = self.c_suite(vec![a1, ifs]);
                    self.result_stmt = self.transform_stmt_opt(Some(suite))?;
                    return Ok(());
                }
            }
        }

        // Static if evaluation
        if types.static_kind(self.ety(cond)) != LiteralKind::Runtime {
            if let Some(is_true) = self.evaluate_static_condition(cond) {
                typhon_common::log_flag!(TYPECHECK, "[static::cond] {:?}: {}", self.src, is_true);
                let mut x = if is_true { is.if_suite } else { is.else_suite };
                if self.has_side_effect(cond) {
                    let es = self.n_stmt(ExprStmt { expr: Some(cond) });
                    let mut items = vec![es];
                    items.extend(x);
                    x = Some(self.c_suite(items));
                }
                self.ctx.borrow_mut().static_branch_depth += 1;
                let r = self.transform_stmt_opt(x);
                self.ctx.borrow_mut().static_branch_depth -= 1;
                let r = r?;
                self.result_stmt = match r {
                    Some(r) => Some(r),
                    None => {
                        let e = self.c_suite(vec![]);
                        self.transform_stmt_opt(Some(e))?
                    }
                };
            }
            return Ok(());
        }

        self.wrap_expr(&mut cond, b, None, true)?;
        self.ctx.borrow_mut().block_level += 1;
        let r = (|| -> Result<(Option<StmtId>, Option<StmtId>)> {
            let i = self.transform_stmt_opt(is.if_suite)?;
            let i = self.cache.ast.wrap_suite(i);
            let e = self.transform_stmt_opt(is.else_suite)?;
            let e = self.cache.ast.wrap_suite(e);
            Ok((i, e))
        })();
        self.ctx.borrow_mut().block_level -= 1;
        let (i, el) = r?;
        {
            let k = self.cache.ast[s].kind.as_if_mut().unwrap();
            k.cond = Some(cond);
            k.if_suite = i;
            k.else_suite = el;
        }
        let a = &self.cache.ast;
        if a[cond].done && i.is_none_or(|x| a[x].done) && el.is_none_or(|x| a[x].done) {
            self.set_done_stmt(s);
        }
        Ok(())
    }

    /// `visit(MatchStmt*)`: transform into a series of conditional statements inside a
    /// `while True` loop (used to simulate goto with break). See `transformPattern`.
    pub fn visit_match(&mut self, s: StmtId) -> Result<()> {
        let ms = self.cache.ast[s].kind.as_match().unwrap().clone();
        let var = self.get_temporary_var("match");
        let result = self.c_suite(vec![]);
        let v = self.id_n(&var);
        let x = self.c_clone(ms.expr.unwrap());
        let a = self.c_assign(v, x);
        let a = self.transform_stmt_opt(Some(a))?;
        self.cache.ast.suite_add_stmt(result, a);
        for c in &ms.items {
            let br = self.n_stmt(BreakStmt);
            let mut items = vec![];
            items.extend(c.suite);
            items.push(br);
            let mut suite = self.c_suite(items);
            if let Some(g) = c.guard {
                suite = self.c_if(g, Some(suite), None);
            }
            let v = self.id_n(&var);
            let p = self.transform_pattern(v, c.pattern.unwrap(), suite)?;
            self.cache.ast.suite_add_stmt(result, p);
        }
        // Make sure to break even if there is no case _ to prevent infinite loop
        let br = self.n_stmt(BreakStmt);
        self.cache.ast.suite_add_stmt(result, Some(br));
        let t = self.n_expr(BoolExpr { value: true });
        let w = self.n_stmt(WhileStmt { cond: Some(t), suite: Some(result), else_suite: None, goto_var: String::new() });
        self.result_stmt = self.transform_stmt_opt(Some(w))?;
        Ok(())
    }
}

impl<'a> TypecheckVisitor<'a> {
    fn c_isinstance(&mut self, e: ExprId, typ: &str) -> ExprId {
        let f = self.id_n("isinstance");
        let c = self.c_clone(e);
        let t = self.id_n(typ);
        self.call_n(f, vec![c, t])
    }
    /// `findWildcard`: index of the `...` / `*_` gap (or `items.len()`).
    fn c_find_wildcard(&self, items: &[ExprId]) -> Result<usize> {
        let a = &self.cache.ast;
        let mut i = items.len();
        for (it, x) in items.iter().enumerate() {
            let star = a[*x].kind.as_star();
            if let Some(s) = star {
                if !crate::ast::is_id(a, s.expr, "_") {
                    typhon_common::bail_e!(ErrorKind::CUSTOM, a[*x].src, "only *_ is supported in sequence patterns");
                }
            }
            if a[*x].kind.tag() == ExprTag::Ellipsis || star.is_some() {
                if i != items.len() {
                    typhon_common::bail_e!(ErrorKind::MATCH_MULTI_ELLIPSIS, a[*x].src);
                }
                i = it;
            }
        }
        Ok(i)
    }

    /// `transformPattern(var, pattern, suite)`: transform a match pattern into a series of
    /// if statements.
    pub fn transform_pattern(&mut self, var: ExprId, pattern: ExprId, suite: StmtId) -> Result<Option<StmtId>> {
        let st = stdlib();
        let mut suite = suite;
        // Use stable identifiers for nested isinstance narrowing and evaluate each
        // field/index only once.
        if self.cache.ast[var].kind.tag() != ExprTag::Id {
            let temporary = self.get_temporary_var("match");
            let t = self.id_n(&temporary);
            let a = self.c_assign(t, var);
            let t = self.id_n(&temporary);
            let p = self.transform_pattern(t, pattern, suite)?;
            let mut items = vec![a];
            items.extend(p);
            return Ok(Some(self.c_suite(items)));
        }

        // See the above examples for transformation details
        let kind = self.cache.ast[pattern].kind.clone();
        match &kind {
            ExprKind::None(_) => {
                let b = self.c_bin(var, "is", pattern);
                return Ok(Some(self.c_if(b, Some(suite), None)));
            }
            ExprKind::Int(_) | ExprKind::Bool(_) => {
                // Bool and int patterns
                let is_bool = matches!(kind, ExprKind::Bool(_));
                let i = self.c_isinstance(var, if is_bool { &st.bool_ } else { "int" });
                let b = self.c_bin(var, "==", pattern);
                let inner = self.c_if(b, Some(suite), None);
                return Ok(Some(self.c_if(i, Some(inner), None)));
            }
            ExprKind::Range(er) => {
                // Range pattern
                let i = self.c_isinstance(var, "int");
                let vc = self.c_clone(var);
                let b2 = self.c_bin(vc, "<=", er.stop.unwrap());
                let in2 = self.c_if(b2, Some(suite), None);
                let b1 = self.c_bin(var, ">=", er.start.unwrap());
                let in1 = self.c_if(b1, Some(in2), None);
                return Ok(Some(self.c_if(i, Some(in1), None)));
            }
            ExprKind::Tuple(_) | ExprKind::List(_) => {
                let tuple = matches!(kind, ExprKind::Tuple(_));
                let items = match &kind {
                    ExprKind::Tuple(t) => t.items.clone(),
                    ExprKind::List(l) => l.items.clone(),
                    _ => unreachable!(),
                };
                let wildcard = self.c_find_wildcard(&items)?;
                let size = items.len();
                let mut it = size;
                while it > wildcard + 1 {
                    it -= 1;
                    let vc = self.c_clone(var);
                    let n = self.n_expr(IntExpr::new(it as i64 - size as i64));
                    let ix = self.n_expr(IndexExpr { expr: Some(vc), index: Some(n) });
                    suite = self.transform_pattern(ix, items[it], suite)?.unwrap();
                }
                let mut it = wildcard;
                while it > 0 {
                    it -= 1;
                    let vc = self.c_clone(var);
                    let n = self.n_expr(IntExpr::new(it as i64));
                    let ix = self.n_expr(IndexExpr { expr: Some(vc), index: Some(n) });
                    suite = self.transform_pattern(ix, items[it], suite)?.unwrap();
                }
                let length = if tuple {
                    let f = self.id_n(&mangle::mangled_func("std.internal.static", "len", 0, 0, false));
                    let vc = self.c_clone(var);
                    self.call_n(f, vec![vc])
                } else {
                    let f = self.id_n("len");
                    let vc = self.c_clone(var);
                    self.call_n(f, vec![vc])
                };
                let gap = wildcard != size;
                let i = self.c_isinstance(var, if tuple { &st.tuple } else { &st.list });
                let n = self.n_expr(IntExpr::new((size - gap as usize) as i64));
                let b = self.c_bin(length, if gap { ">=" } else { "==" }, n);
                let inner = self.c_if(b, Some(suite), None);
                return Ok(Some(self.c_if(i, Some(inner), None)));
            }
            ExprKind::Binary(eb) => {
                // Or pattern
                if eb.op == "|" || eb.op == "||" {
                    let vc = self.c_clone(var);
                    let sc = self.cache.ast.clone_stmt(suite, false);
                    let p1 = self.transform_pattern(vc, eb.lexpr.unwrap(), sc)?;
                    let p2 = self.transform_pattern(var, eb.rexpr.unwrap(), suite)?;
                    let items: Vec<StmtId> = p1.into_iter().chain(p2).collect();
                    return Ok(Some(self.c_suite(items)));
                }
            }
            ExprKind::Id(ei) => {
                // Wildcard pattern
                if ei.value != "_" {
                    let a = self.c_assign(pattern, var);
                    return Ok(Some(self.c_suite(vec![a, suite])));
                } else {
                    return Ok(Some(suite));
                }
            }
            ExprKind::Assign(ea) => {
                // Bound pattern
                typhon_common::seqassert!(
                    ea.var.is_some_and(|v| self.cache.ast[v].kind.tag() == ExprTag::Id),
                    "only simple assignment expressions are supported"
                );
                // Bind aliases after the subpattern succeeds, within its narrowed scope.
                let vc = self.c_clone(var);
                let a = self.c_assign(ea.var.unwrap(), vc);
                let s2 = self.c_suite(vec![a, suite]);
                return self.transform_pattern(var, ea.expr.unwrap(), s2);
            }
            ExprKind::Call(call) => {
                if let Some(r) = self.c_class_pattern(var, pattern, call.clone(), suite)? {
                    return Ok(Some(r));
                }
            }
            _ => {}
        }

        // transform to check for pattern errors
        let mut pattern = self.tr(pattern)?;
        if self.cache.ast[pattern].kind.tag() == ExprTag::Ellipsis {
            let f = self.id_n("ellipsis");
            pattern = self.call_n(f, vec![]);
        }
        // Fallback (`__match__`) pattern
        let h = self.id_n("hasattr");
        let (vc, pc) = (self.c_clone(var), self.c_clone(pattern));
        let m = self.n_expr(StringExpr::new("__match__", ""));
        let hc = self.call_n(h, vec![vc, m, pc]);
        let vc = self.c_clone(var);
        let d = self.dot_n(vc, "__match__");
        let pc = self.c_clone(pattern);
        let mc = self.call_n(d, vec![pc]);
        let sc = self.cache.ast.clone_stmt(suite, false);
        let inner1 = self.c_if(mc, Some(sc), None);
        let isf = self.id_n("isinstance");
        let (t1, t2) = (self.id_n(&st.type_), self.id_n(&st.type_));
        let vc = self.c_clone(var);
        let c1 = self.call_n(t1, vec![vc]);
        let pc = self.c_clone(pattern);
        let c2 = self.call_n(t2, vec![pc]);
        let ic = self.call_n(isf, vec![c1, c2]);
        let eq = self.c_bin(var, "==", pattern);
        let in_eq = self.c_if(eq, Some(suite), None);
        let else_ = self.c_if(ic, Some(in_eq), None);
        Ok(Some(self.c_if(hc, Some(inner1), Some(else_))))
    }

    /// Class (`CallExpr`) pattern part of `transformPattern`. `None` falls through to the
    /// generic `__match__` path.
    fn c_class_pattern(&mut self, var: ExprId, pattern: ExprId, call: CallExpr, suite: StmtId) -> Result<Option<StmtId>> {
        let st = stdlib();
        let mut suite = suite;
        let hc = self.c_clone(call.expr.unwrap());
        let head = self.tr(hc)?;
        // Never evaluate a function-call pattern.
        if !Self::is_type_expr(self.cache, head) {
            let src = self.cache.ast[pattern].src;
            typhon_common::bail_e!(ErrorKind::CUSTOM, src, "called match pattern must be a class");
        }
        // Keyword/empty class patterns inspect the existing object. Preserve the
        // existing value-pattern behavior for positional constructor expressions.
        let keyword = call.items.iter().any(|a| !a.name.is_empty());
        if !(call.items.is_empty() || keyword) {
            return Ok(None);
        }
        for arg in call.items.iter().rev() {
            if arg.name.is_empty() {
                let src = arg.value.map(|v| self.cache.ast[v].src).unwrap_or(arg.src);
                typhon_common::bail_e!(ErrorKind::CUSTOM, src, "expected a keyword class pattern");
            }
            let attribute = self.get_temporary_var("match_attr");
            let h = self.id_n("hasattr");
            let vc = self.c_clone(var);
            let n = self.n_expr(StringExpr::new(arg.name.clone(), ""));
            let hcall = self.call_n(h, vec![vc, n]);
            let at = self.id_n(&attribute);
            let vc = self.c_clone(var);
            let d = self.dot_n(vc, &arg.name);
            let a = self.c_assign(at, d);
            let exc = self.id_n(&mangle::mangled_class("std.internal.types.error", "AttributeError", 0));
            let es = self.c_suite(vec![]);
            let ex = self.n_stmt(ExceptStmt { var: String::new(), exc: Some(exc), suite: Some(es) });
            let at = self.id_n(&attribute);
            let p = self.transform_pattern(at, arg.value.unwrap(), suite)?;
            // `if hasattr(var, name): try: attr = var.name except AttributeError: pass else: <pattern>`
            let tr = self.n_stmt(TryStmt { items: vec![ex], suite: Some(a), else_suite: p, finally: None });
            suite = self.c_if(hcall, Some(tr), None);
        }
        // Param.type and other AST fields are optional.
        // Unwrap only a non-None optional and keep Any subjects on their RTTI path.
        let temporary = self.get_temporary_var("match");
        let sc = self.cache.ast.clone_stmt(suite, false);
        let optional_suite = self.cache.ast.wrap_suite(Some(sc)).unwrap();
        let vname = self.cache.ast[var].kind.as_id().unwrap().value.clone();
        let mut renames = std::collections::BTreeMap::new();
        renames.insert(self.get_unmangled_name(&vname), temporary.clone());
        self.cache.ast[optional_suite].attrs.set_value(Attr::LocalRenames, AttrValue::KeyValue(renames));
        let i_opt = self.c_isinstance(var, &st.optional);
        let vc = self.c_clone(var);
        let none = self.n_expr(NoneExpr);
        let not_none = self.c_bin(vc, "is not", none);
        let t = self.id_n(&temporary);
        let u = self.id_n("unwrap");
        let vc = self.c_clone(var);
        let uc = self.call_n(u, vec![vc]);
        let a = self.c_assign(t, uc);
        let isf = self.id_n("isinstance");
        let t = self.id_n(&temporary);
        let hc = self.c_clone(head);
        let ic = self.call_n(isf, vec![t, hc]);
        let inner_if = self.c_if(ic, Some(optional_suite), None);
        let s2 = self.c_suite(vec![a, inner_if]);
        let nn_if = self.c_if(not_none, Some(s2), None);
        let isf = self.id_n("isinstance");
        let ic2 = self.call_n(isf, vec![var, head]);
        let else_if = self.c_if(ic2, Some(suite), None);
        Ok(Some(self.c_if(i_opt, Some(nn_if), Some(else_if))))
    }
}
