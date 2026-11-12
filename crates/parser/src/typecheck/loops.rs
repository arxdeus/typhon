
use typhon_common::mangle::{self, stdlib};
use typhon_common::{CompilerError, ErrorKind, Result, SrcInfo};

use super::{AstNode, StaticLoopWrapFn, TypeContext, TypecheckVisitor};
use crate::ast::{Attr, BoolExpr, BreakStmt, CallArg, ExprKind, ImportStmt, IntExpr, StringExpr, SuiteStmt, UpdateMode, WhileStmt};
use crate::cache::VAR_USED_SUFFIX;
use crate::ids::{ExprId, StmtId};

impl<'a> TypecheckVisitor<'a> {
    fn loop_flat_false(&mut self, s: StmtId, what: &str) -> Result<()> {
        let mut c = self.ctx.borrow_mut();
        match c.get_base().get_loop() {
            None => {
                drop(c);
                let src = self.cache.ast[s].src;
                typhon_common::bail_e!(ErrorKind::EXPECTED_LOOP, src, what);
            }
            Some(l) => l.flat = false,
        }
        Ok(())
    }

    /// `visit(BreakStmt*)`: `break` -> `no_break = False; break` inside loop-else loops.
    pub fn visit_break(&mut self, s: StmtId) -> Result<()> {
        self.loop_flat_false(s, "break")?;
        let bv = self.ctx.borrow_mut().get_base().get_loop().unwrap().break_var.clone();
        if !bv.is_empty() {
            let l = self.xm_id(&bv);
            let f = self.xm_bool(false);
            let a = self.xm_assign(l, Some(f), None, UpdateMode::Update);
            let a = self.tr_stmt(a)?;
            let b = self.n_stmt(BreakStmt);
            self.result_stmt = Some(self.xm_suite(&[Some(a), Some(b)]));
        } else {
            self.set_done_stmt(s);
            let sl = self.ctx.borrow().static_loops.last().cloned().unwrap_or_default();
            if !sl.is_empty() {
                let l = self.xm_id(&sl);
                let f = self.xm_bool(false);
                let a = self.xm_assign(l, Some(f), None, UpdateMode::Update);
                let su = self.xm_suite(&[Some(a), Some(s)]);
                self.result_stmt = Some(self.tr_stmt(su)?);
            }
        }
        Ok(())
    }

    /// `visit(ContinueStmt*)`.
    pub fn visit_continue(&mut self, s: StmtId) -> Result<()> {
        self.loop_flat_false(s, "continue")?;
        self.set_done_stmt(s);
        let sl = self.ctx.borrow().static_loops.last().cloned().unwrap_or_default();
        if !sl.is_empty() {
            let b = self.n_stmt(BreakStmt);
            self.set_done_stmt(b);
            self.result_stmt = Some(b);
        }
        Ok(())
    }

    /// `visit(WhileStmt*)`: handles `while ... else` via a `no_break` flag.
    pub fn visit_while(&mut self, s: StmtId) -> Result<()> {
        let w = self.cache.ast[s].kind.as_while().unwrap().clone();
        let has_else = w.else_suite.is_some_and(|e| self.cache.ast.first_in_block(e).is_some());
        let mut break_var = String::new();
        if has_else {
            break_var = self.get_temporary_var("no_break");
            let l = self.xm_id(&break_var);
            let t = self.xm_bool(true);
            let a = self.xm_assign(l, Some(t), None, UpdateMode::Assign);
            let a = self.tr_stmt(a)?;
            self.prepend_stmts.borrow_mut().push(a);
        }
        self.ctx.borrow_mut().static_loops.push(w.goto_var.clone());
        self.ctx.borrow_mut().get_base().loops.push(super::Loop::new(break_var.clone()));

        let cond = w.cond.expect("while without condition");
        let bt = self.get_stdlib_type(&stdlib().bool_)?;
        self.cache.ast[cond].expected_type = Some(bt);
        let mut cond = self.tr(cond)?;
        self.wrap_expr(&mut cond, bt, None, true)?;
        self.cache.ast[s].kind.as_while_mut().unwrap().cond = Some(cond);

        let suite = self.xm_block(w.suite);
        self.ctx.borrow_mut().static_loops.pop();
        let suite = suite?;
        self.cache.ast[s].kind.as_while_mut().unwrap().suite = suite;

        if has_else {
            let es = w.else_suite;
            self.cache.ast[s].kind.as_while_mut().unwrap().else_suite = None;
            let bv = self.xm_id(&break_var);
            let i = self.xm_if(bv, es, None);
            let su = self.xm_suite(&[Some(s), Some(i)]);
            let r = self.tr_stmt(su);
            if r.is_err() {
                self.ctx.borrow_mut().get_base().loops.pop();
            }
            self.result_stmt = Some(r?);
        }
        self.ctx.borrow_mut().get_base().loops.pop();

        let cond = self.cache.ast[s].kind.as_while().unwrap().cond.unwrap();
        let suite = self.cache.ast[s].kind.as_while().unwrap().suite;
        if self.cache.ast[cond].done && self.xm_done(suite) {
            self.set_done_stmt(s);
        }
        Ok(())
    }
}

impl<'a> TypecheckVisitor<'a> {
    /// `visit(ForStmt*)`: wrap the iterator with `__iter__` if needed; static loops are
    /// handled by [`Self::transform_static_for_loop`].
    pub fn visit_for(&mut self, s: StmtId) -> Result<()> {
        let dec = self.cache.ast[s].kind.as_for().unwrap().decorator;
        let dec = self.transform_for_decorator(dec)?;
        self.cache.ast[s].kind.as_for_mut().unwrap().decorator = dec;

        if let Some(fc) = dec.and_then(|d| self.cache.ast[d].kind.as_call().cloned()) {
            if let Some(fi) = fc.expr.filter(|e| self.cache.ast[*e].kind.as_id().is_some()) {
                let ft = self.ety(fi);
                let types = self.types();
                if let Some(f) = types.get_func(ft)
                    && types.func_name(f) == mangle::func("std.openmp", "for_par")
                {
                    let g = self.extract_func_generic(ft, 3);
                    if types.get_bool_static(g) == Some(true) {
                        let what = self.xm_id("gpu");
                        let as_ = self.get_temporary_var("_");
                        let imp = self.n_stmt(ImportStmt { from: Some(what), what: None, as_, ..Default::default() });
                        let imp = self.tr_stmt(imp)?;
                        self.prepend_stmts.borrow_mut().push(imp);
                    }
                }
            }
        }

        let mut break_var = String::new();
        // Needs in-advance transformation to prevent name clashes with the iterator variable
        let iter = self.cache.ast[s].kind.as_for().unwrap().iter.expect("for without iter");
        // do not expand special calls here, might be needed for static loops!
        self.cache.ast[iter].attrs.set(Attr::ExprNoSpecial);
        let iter = self.tr(iter)?;
        self.cache.ast[s].kind.as_for_mut().unwrap().iter = Some(iter);

        // Check for for-else clause
        let mut assign = None;
        let els = self.cache.ast[s].kind.as_for().unwrap().else_suite;
        let has_else = els.is_some_and(|e| self.cache.ast.first_in_block(e).is_some());
        if has_else {
            break_var = self.get_temporary_var("no_break");
            let l = self.xm_id(&break_var);
            let t = self.xm_bool(true);
            let a = self.xm_assign(l, Some(t), None, UpdateMode::Assign);
            assign = Some(self.tr_stmt(a)?);
        }

        // Extract the iterator type of the for
        let Some(mut iter_type) = self.extract_class_type_expr(iter) else {
            return Ok(()); // wait until the iterator is known
        };

        let (delay, static_loop) = self.transform_static_for_loop(s)?;
        if delay {
            return Ok(());
        }
        if let Some(sl) = static_loop {
            self.result_stmt = Some(sl);
            return Ok(());
        }

        // Case: iterating a non-generator. Wrap with `__iter__`
        let is_async = self.cache.ast[s].kind.as_for().unwrap().async_;
        let gen_name = if is_async { "AsyncGenerator" } else { "Generator" };
        let types = self.types();
        let mut is_generator = types.class_name(iter_type).is_some_and(|n| n.as_str() == gen_name);
        if !is_generator && !self.cache.ast[s].kind.as_for().unwrap().wrapped {
            let it = self.cache.ast[s].kind.as_for().unwrap().iter.unwrap();
            let d = self.xm_dot(it, "__iter__");
            let c = self.xm_call(d, &[]);
            let it = self.tr(c)?;
            self.cache.ast[s].kind.as_for_mut().unwrap().iter = Some(it);
            self.cache.ast[s].kind.as_for_mut().unwrap().wrapped = true;
            match self.extract_class_type_expr(it) {
                None => return Ok(()),
                Some(t) => iter_type = t,
            }
            is_generator = types.class_name(iter_type).is_some_and(|n| n.as_str() == gen_name);
        }

        // Delay target unpacking until the yielded type is known.
        let var0 = self.cache.ast[s].kind.as_for().unwrap().var.expect("for without var");
        if self.cache.ast[var0].kind.as_id().is_none() {
            let tv = self.cache.get_temporary_var("for");
            let var = self.xm_id(&tv);
            if is_generator {
                let g = self.extract_class_generic(iter_type, 0);
                self.set_type(var, g);
            }
            let ns = self.unpack_assignment(var0, var, true)?;
            let su = self.cache.ast[s].kind.as_for().unwrap().suite;
            let su = self.xm_suite(&[ns, su]);
            let f = self.cache.ast[s].kind.as_for_mut().unwrap();
            f.suite = Some(su);
            f.var = Some(var);
        }

        self.ctx.borrow_mut().get_base().loops.push(super::Loop::new(break_var.clone()));
        let var = self.cache.ast[s].kind.as_for().unwrap().var.unwrap();
        let var_name = match self.cache.ast[var].kind.as_id() {
            Some(i) => i.value.clone(),
            None => panic!("corrupt for variable: {}", self.cache.ast.expr_to_string(var, -1)),
        };
        let dominated = self.cache.ast[var].attrs.has(Attr::ExprDominated);
        let dominated_used = self.cache.ast[var].attrs.has(Attr::ExprDominatedUsed);
        if !dominated && !dominated_used {
            let un = self.get_unmangled_name(&var_name);
            let canon = self.ctx.borrow().generate_canonical_name(self.cache, &var_name, false, false);
            let t = match self.expr_type(var) {
                Some(t) => t,
                None => self.instantiate_unbound(),
            };
            let time = self.get_time();
            TypeContext::add_var(self.cache, &self.ctx, &un, &canon, t, time, SrcInfo::default());
        } else if dominated_used {
            self.cache.ast[var].attrs.erase(Attr::ExprDominatedUsed);
            self.cache.ast[var].attrs.set(Attr::ExprDominated);
            let l = self.xm_id(&format!("{var_name}{VAR_USED_SUFFIX}"));
            let t = self.xm_bool(true);
            let a = self.xm_assign(l, Some(t), None, UpdateMode::Update);
            let su = self.cache.ast[s].kind.as_for().unwrap().suite;
            let su = self.xm_suite(&[Some(a), su]);
            self.cache.ast[s].kind.as_for_mut().unwrap().suite = Some(su);
        }
        let var = self.tr(var)?;
        self.cache.ast[s].kind.as_for_mut().unwrap().var = Some(var);

        // Unify iterator variable and the iterator type
        if !is_generator {
            let it = self.cache.ast[s].kind.as_for().unwrap().iter.unwrap();
            let src = self.cache.ast[it].src;
            typhon_common::bail_e!(ErrorKind::EXPECTED_GENERATOR, src);
        }
        let vt = self.ety(var);
        let g = self.extract_class_generic(iter_type, 0);
        self.unify(vt, g)?;

        self.ctx.borrow_mut().static_loops.push(String::new());
        let su = self.cache.ast[s].kind.as_for().unwrap().suite;
        let su = self.xm_block(su);
        self.ctx.borrow_mut().static_loops.pop();
        let su = su?;
        self.cache.ast[s].kind.as_for_mut().unwrap().suite = su;

        if self.ctx.borrow_mut().get_base().get_loop().unwrap().flat {
            self.cache.ast[s].kind.as_for_mut().unwrap().flat = true;
        }

        // Complete for-else clause
        if has_else {
            self.cache.ast[s].kind.as_for_mut().unwrap().else_suite = None;
            let bv = self.xm_id(&break_var);
            let i = self.xm_if(bv, els, None);
            let su = self.xm_suite(&[assign, Some(s), Some(i)]);
            self.result_stmt = Some(self.tr_stmt(su)?);
            self.cache.ast[s].kind.as_for_mut().unwrap().else_suite = None;
        }

        self.ctx.borrow_mut().get_base().loops.pop();

        let f = self.cache.ast[s].kind.as_for().unwrap();
        let (it, su) = (f.iter.unwrap(), f.suite);
        if self.cache.ast[it].done && self.xm_done(su) {
            self.set_done_stmt(s);
        }
        Ok(())
    }

    /// `transformForDecorator(decorator)`:
    /// `@par(num_threads=2, openmp="schedule(static)")` -> `for_par(num_threads=2, schedule="static")`.
    pub fn transform_for_decorator(&mut self, d: Option<ExprId>) -> Result<Option<ExprId>> {
        let Some(decorator) = d else { return Ok(None) };
        // A re-visited ForStmt (e.g. after a delayed realization) already holds the
        // transformed `for_par(...)` call: its Literal args (schedule/ordered/...) now live
        // only in the callee's generics, so rebuilding the call would reset them to defaults.
        let call = self.cache.ast[decorator].kind.as_call().cloned();
        let callee = call.as_ref().and_then(|c| c.expr).unwrap_or(decorator);
        let fp = mangle::func("std.openmp", "for_par");
        let already = call.is_some() && self.cache.ast[callee].kind.as_id().is_some_and(|i| i.value.starts_with(&fp));
        if already {
            return Ok(Some(self.tr(decorator)?));
        }
        let ci = self.tr(callee)?;
        let ok = self.cache.ast[ci].kind.as_id().is_some_and(|i| i.value.starts_with(&mangle::func("std.openmp", "for_par")));
        if !ok {
            let src = self.cache.ast[decorator].src;
            typhon_common::bail_e!(ErrorKind::LOOP_DECORATOR, src);
        }
        let mut args: Vec<CallArg> = vec![];
        let openmp = String::new(); // Note: never assigned in the reference either
        let mut omp: Vec<CallArg> = vec![];
        if let Some(c) = &call {
            for a in &c.items {
                let is_str = a.value.is_some_and(|v| self.cache.ast[v].kind.as_string().is_some());
                if a.name == "openmp" || (a.name.is_empty() && openmp.is_empty() && is_str) {
                    let v = a.value.unwrap();
                    let code = match self.cache.ast[v].kind.as_string() {
                        Some(s) => s.value(),
                        None => {
                            let src = self.cache.ast[v].src;
                            typhon_common::bail_e!(ErrorKind::LOOP_DECORATOR, src);
                        }
                    };
                    let src = self.cache.ast[v].src;
                    omp = self.parse_openmp_pragma(&code, src)?;
                } else {
                    let v = self.transform(a.value)?;
                    args.push(CallArg { name: a.name.clone(), value: v, src: a.src });
                }
            }
        }
        for a in omp {
            let v = self.transform(a.value)?;
            args.push(CallArg { name: a.name, value: v, src: a.src });
        }
        let fp = self.xm_id("for_par");
        let fp = self.tr(fp)?;
        let c = self.xm_call_args(fp, args);
        Ok(Some(self.tr(c)?))
    }

    /// `parseOpenMP(cache, code, loc)` (openmp.peg).
    /// Note: hand-written recursive-descent equivalent of `openmp.peg` (the grammar
    /// lane's `crate::peg` has no OpenMP entry point yet). `collapse` takes `(n)` (
    /// grammar rule reads a missing value).
    fn parse_openmp_pragma(&mut self, code: &str, loc: SrcInfo) -> Result<Vec<CallArg>> {
        let b = code.as_bytes();
        let mut p = 0usize;
        let ws = |p: &mut usize| {
            while *p < b.len() && (b[*p] == b' ' || b[*p] == b'\t') {
                *p += 1;
            }
        };
        let lit = |p: &mut usize, s: &str| -> bool {
            if code[*p..].starts_with(s) {
                *p += s.len();
                true
            } else {
                false
            }
        };
        let int = |p: &mut usize| -> Option<i64> {
            if *p < b.len() && (b'1'..=b'9').contains(&b[*p]) {
                let st = *p;
                while *p < b.len() && b[*p].is_ascii_digit() {
                    *p += 1;
                }
                code[st..*p].parse().ok()
            } else {
                None
            }
        };
        let err = |p: usize| CompilerError::parser(ErrorKind::CUSTOM, loc, &[&format!("openmp: syntax error, unexpected '{}'", &code[p..])]);
        let mut out: Vec<CallArg> = vec![];
        lit(&mut p, "omp");
        ws(&mut p);
        lit(&mut p, "parallel");
        ws(&mut p);
        while p < b.len() {
            let start = p;
            if lit(&mut p, "schedule") {
                ws(&mut p);
                if !lit(&mut p, "(") {
                    return Err(err(start));
                }
                ws(&mut p);
                let kinds = ["static", "dynamic", "guided", "auto", "runtime"];
                let Some(k) = kinds.iter().find(|k| code[p..].starts_with(**k)) else { return Err(err(start)) };
                p += k.len();
                let e = self.n_expr(StringExpr::new(*k, ""));
                self.cache.ast[e].src = loc;
                out.push(CallArg { name: "schedule".into(), value: Some(e), src: loc });
                let save = p;
                ws(&mut p);
                if lit(&mut p, ",") {
                    ws(&mut p);
                    let Some(n) = int(&mut p) else { return Err(err(start)) };
                    let e = self.n_expr(IntExpr::new(n));
                    self.cache.ast[e].src = loc;
                    out.push(CallArg { name: "chunk_size".into(), value: Some(e), src: loc });
                } else {
                    p = save;
                }
                ws(&mut p);
                if !lit(&mut p, ")") {
                    return Err(err(start));
                }
            } else if lit(&mut p, "num_threads") || lit(&mut p, "collapse") {
                let name = if code[start..].starts_with("num_threads") { "num_threads" } else { "collapse" };
                ws(&mut p);
                if !lit(&mut p, "(") {
                    return Err(err(start));
                }
                ws(&mut p);
                let Some(n) = int(&mut p) else { return Err(err(start)) };
                ws(&mut p);
                if !lit(&mut p, ")") {
                    return Err(err(start));
                }
                let e = self.n_expr(IntExpr::new(n));
                self.cache.ast[e].src = loc;
                out.push(CallArg { name: name.into(), value: Some(e), src: loc });
            } else if lit(&mut p, "ordered") || lit(&mut p, "gpu") {
                let name = if code[start..].starts_with("ordered") { "ordered" } else { "gpu" };
                let e = self.n_expr(BoolExpr { value: true });
                self.cache.ast[e].src = loc;
                out.push(CallArg { name: name.into(), value: Some(e), src: loc });
            } else {
                return Err(err(start));
            }
            ws(&mut p);
        }
        Ok(out)
    }

    /// `transformStaticForLoop(stmt)`: `(delay, replacement)`. each static
    /// iteration becomes `while loop: (assigns; suite; break)` unless the loop is flat.
    pub fn transform_static_for_loop(&mut self, s: StmtId) -> Result<(bool, Option<StmtId>)> {
        let loop_var = self.get_temporary_var("loop");
        let f = self.cache.ast[s].kind.as_for().unwrap().clone();
        let mut suite = f.suite.map(|x| self.cache.ast.clone_stmt(x, true));
        let flat = f.flat;
        let orig_suite = f.suite;
        let suite_for_wrap = suite;
        let lv = loop_var.clone();
        let mut wrap = move |tv: &mut TypecheckVisitor<'a>, assigns: StmtId| -> Result<AstNode> {
            let ret = if !flat {
                let brk = tv.n_stmt(BreakStmt);
                tv.set_done_stmt(brk); // Avoid transforming this one to skip extra checks
                let cs = suite_for_wrap.map(|x| tv.cache.ast.clone_stmt(x, false));
                let body = tv.xm_suite(&[Some(assigns), cs, Some(brk)]);
                let cond = tv.xm_id(&lv);
                tv.n_stmt(WhileStmt { cond: Some(cond), suite: Some(body), else_suite: None, goto_var: lv.clone() })
            } else {
                let cs = orig_suite.map(|x| tv.cache.ast.clone_stmt(x, false));
                tv.xm_suite(&[Some(assigns), cs])
            };
            Ok(AstNode::Stmt(ret))
        };
        let (ok, delay, preamble, items) =
            self.transform_static_loop_call(f.var.unwrap(), &mut suite, f.iter.unwrap(), &mut wrap, false)?;
        if !ok {
            return Ok((false, None));
        }
        if delay {
            return Ok((true, None));
        }

        // Close the loop
        let block = self.n_stmt(SuiteStmt::default());
        self.cache.ast.suite_add_stmt(block, preamble);
        for i in items {
            if let AstNode::Stmt(x) = i {
                self.cache.ast.suite_add_stmt(block, Some(x));
            }
        }
        let lp = if !flat {
            self.ctx.borrow_mut().block_level += 1;
            let l = self.xm_id(&loop_var);
            let fl = self.xm_bool(false);
            let a = self.xm_assign(l, Some(fl), None, UpdateMode::Update);
            self.cache.ast.suite_add_stmt(block, Some(a));
            let l = self.xm_id(&loop_var);
            let t = self.xm_bool(true);
            let init = self.xm_assign(l, Some(t), None, UpdateMode::Assign);
            let c = self.xm_id(&loop_var);
            let w = self.n_stmt(WhileStmt { cond: Some(c), suite: Some(block), else_suite: None, goto_var: String::new() });
            let su = self.xm_suite(&[Some(init), Some(w)]);
            let r = self.tr_stmt(su);
            self.ctx.borrow_mut().block_level -= 1;
            r?
        } else {
            self.tr_stmt(block)?
        };
        Ok((false, Some(lp)))
    }

    /// `transformStaticLoopCall(varExpr, varSuite, iter, wrap, allowNonHeterogenous)`:
    /// `(ok, delay, preamble, items)`. `var_suite` is the slot holding the loop suite.
    pub fn transform_static_loop_call(
        &mut self,
        var: ExprId,
        _var_suite: &mut Option<StmtId>,
        iter: ExprId,
        wrap: &mut StaticLoopWrapFn<'_, 'a>,
        allow_non_heterogenous: bool,
    ) -> Result<(bool, bool, Option<StmtId>, Vec<AstNode>)> {
        if self.expr_class(iter).is_none() {
            return Ok((true, true, None, vec![]));
        }
        let mut vars: Vec<String> = vec![];
        match &self.cache.ast[var].kind {
            ExprKind::Id(i) => vars.push(i.value.clone()),
            ExprKind::List(crate::ast::ListExpr { items }) | ExprKind::Tuple(crate::ast::TupleExpr { items }) => {
                for it in items {
                    match self.cache.ast[*it].kind.as_id() {
                        Some(i) => vars.push(i.value.clone()),
                        None => return Ok((false, false, None, vec![])),
                    }
                }
            }
            _ => return Ok((false, false, None, vec![])),
        }

        let mut preamble = None;
        let iter = self.get_head_expr(iter);
        let fn_name = self.cache.ast[iter]
            .kind
            .as_call()
            .and_then(|c| c.expr)
            .and_then(|e| self.cache.ast[e].kind.as_id().map(|i| i.value.clone()));
        let sw = |p: &str| fn_name.as_ref().is_some_and(|f| f.starts_with(p));
        let m = "std.internal.static";
        let mut block = if sw(&mangle::func(m, "tuple")) {
            self.populate_static_tuple_loop(iter, &vars)?
        } else if sw(&mangle::mangled_func(m, "range", 1, 0, false)) {
            self.populate_simple_static_range_loop(iter, &vars)?
        } else if sw(&mangle::func(m, "range")) {
            self.populate_static_range_loop(iter, &vars)?
        } else if sw(&mangle::mangled_method(m, "function", "overloads", 0, 0)) {
            self.populate_static_fn_overloads_loop(iter, &vars)?
        } else if sw(&mangle::func(m, "enumerate")) {
            self.populate_static_enumerate_loop(iter, &vars)?
        } else if sw(&mangle::func(m, "vars")) {
            self.populate_static_vars_loop(iter, &vars)?
        } else if sw(&mangle::func(m, "methods")) {
            self.populate_static_methods_loop(iter, &vars)?
        } else if sw(&mangle::func(m, "vars_types")) {
            self.populate_static_var_types_loop(iter, &vars)?
        } else {
            let t = self.ety(iter);
            if self.cache.types.is(t, &stdlib().tuple) {
                // Maybe heterogenous?
                if !self.cache.types.can_realize(t) {
                    return Ok((true, true, None, vec![])); // wait until the tuple is fully realizable
                }
                let c = self.expr_class(iter).unwrap();
                if !self.is_heterogenous(c)? && !allow_non_heterogenous {
                    return Ok((false, false, None, vec![]));
                }
                let mut b = self.populate_static_heterogenous_tuple_loop(iter, &vars)?;
                preamble = b.pop();
                b
            } else {
                return Ok((false, false, None, vec![]));
            }
        };
        let mut out = Vec::with_capacity(block.len());
        for b in block.drain(..) {
            out.push(wrap(self, b)?);
        }
        Ok((true, false, preamble, out))
    }
}
