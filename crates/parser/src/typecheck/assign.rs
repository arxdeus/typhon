//! Port of codon/parser/visitors/typecheck/assign.cpp

use typhon_common::mangle::{self, stdlib};
use typhon_common::{ErrorKind, Result};

use super::{TypeContext, TypecheckItem, TypecheckVisitor};
use crate::ast::{
    AssignMemberStmt, AssignStmt, Attr, BinaryExpr, BoolExpr, ExprStmt, ExprTag, IndexExpr, IntExpr, SliceExpr, StmtExpr,
    StringExpr, SuiteStmt, UpdateMode,
};
use crate::cache::{FN_SETTER_SUFFIX, STDLIB_IMPORT, VAR_USED_SUFFIX, fn_optional_unwrap, var_argc, var_argv};
use crate::ids::{ExprId, StmtId};
use crate::types::LiteralKind;

impl<'a> TypecheckVisitor<'a> {
    fn as_assign(&self, s: StmtId) -> AssignStmt {
        self.cache.ast[s].kind.as_assign().expect("not an assignment").clone()
    }
    fn new_assign(&mut self, lhs: ExprId, rhs: Option<ExprId>, ty: Option<ExprId>, update: UpdateMode) -> StmtId {
        self.n_stmt(AssignStmt { lhs: Some(lhs), rhs, type_expr: ty, update })
    }
    fn expr_stmt_n(&mut self, e: ExprId) -> StmtId {
        self.n_stmt(ExprStmt { expr: Some(e) })
    }

    /// `visit(AssignExpr*)`: walrus. `(expr := var)` -> `var = expr; var`
    pub fn visit_assign_expr(&mut self, e: ExprId) -> Result<()> {
        let ae = self.cache.ast[e].kind.as_assign().unwrap().clone();
        let var = ae.var.unwrap();
        let v = self.cache.ast.clone_expr(var, false);
        let a = self.new_assign(v, ae.expr, None, UpdateMode::Assign);
        let attrs = self.cache.ast[e].attrs.clone();
        self.cache.ast[a].attrs = attrs;
        let se = self.n_expr(StmtExpr { items: vec![a], expr: Some(var) });
        self.result_expr = self.transform(Some(se))?;
        Ok(())
    }

    /// `visit(AssignStmt*)`: handle dominated assignments, forward declarations, static
    /// assignments and type/function aliases.
    pub fn visit_assign(&mut self, s: StmtId) -> Result<()> {
        let st = self.as_assign(s);
        let lhs = st.lhs.unwrap();
        if matches!(self.cache.ast[lhs].kind.tag(), ExprTag::Tuple | ExprTag::List) {
            let u = self.unpack_assignment(lhs, st.rhs.unwrap(), false)?;
            self.result_stmt = self.transform_stmt_opt(u)?;
            return Ok(());
        }

        let mut must_update = st.is_update() || st.is_atomic_update();
        must_update |= self.cache.ast[lhs].attrs.has(Attr::ExprDominated);
        must_update |= self.cache.ast[lhs].attrs.has(Attr::ExprDominatedUsed);
        if st.rhs.and_then(|r| self.cache.ast[r].kind.as_binary()).is_some_and(|b| b.in_place) {
            // Update case: a += b
            typhon_common::seqassert!(st.type_expr.is_none(), "invalid AssignStmt");
            must_update = true;
        }

        self.result_stmt = self.transform_assignment(s, must_update)?;
        // PORT-NOTE: C++ re-reads `stmt->getLhs()`; transformAssignment may have replaced it.
        let lhs = self.as_assign(s).lhs.unwrap();
        if self.cache.ast[lhs].attrs.has(Attr::ExprDominatedUsed) {
            // If this is dominated, set __used__ if needed
            self.cache.ast[lhs].attrs.erase(Attr::ExprDominatedUsed);
            let name = self.cache.ast[lhs].kind.as_id().expect("dominated bad assignment").value.clone();
            // A type declaration alone must not mark a binding initialized!
            if self.as_assign(s).rhs.is_some() {
                let n = self.id_n(&format!("{}{}", self.get_unmangled_name(&name), VAR_USED_SUFFIX));
                let t = self.n_expr(BoolExpr { value: true });
                let a = self.new_assign(n, Some(t), None, UpdateMode::Update);
                let mut items = vec![];
                items.extend(self.result_stmt);
                items.push(a);
                let suite = self.n_stmt(SuiteStmt { items });
                self.result_stmt = self.transform_stmt_opt(Some(suite))?;
            }
        }
        Ok(())
    }

    /// `visit(DelStmt*)`: `del a` -> `a = type(a)()` and remove `a` from the context;
    /// `del a[x]` -> `a.__delitem__(x)`
    pub fn visit_del(&mut self, s: StmtId) -> Result<()> {
        let x = self.cache.ast[s].kind.as_del().unwrap().expr.unwrap();
        if let Some(idx) = self.cache.ast[x].kind.as_index().cloned() {
            let d = self.dot_n(idx.expr.unwrap(), "__delitem__");
            let c = self.call_n(d, vec![idx.index.unwrap()]);
            let c = self.transform(Some(c))?;
            self.result_stmt = Some(self.n_stmt(ExprStmt { expr: c }));
        } else if let Some(ei) = self.cache.ast[x].kind.as_id().cloned() {
            // Assign `a` to `type(a)()` to mark it for deletion
            let t = self.id_n(&stdlib().type_);
            let xc = self.cache.ast.clone_expr(x, false);
            let c = self.call_n(t, vec![xc]);
            let c = self.call_n(c, vec![]);
            let a = self.new_assign(x, Some(c), None, UpdateMode::Update);
            self.result_stmt = self.transform_stmt_opt(Some(a))?;

            // Allow deletion *only* if the binding is dominated
            let t = self.get_time();
            if self.find(&ei.value, t).is_none() {
                let src = self.cache.ast[x].src;
                typhon_common::bail_e!(ErrorKind::ID_NOT_FOUND, src, ei.value);
            }
            // TODO: check if variable can be deleted (e.g., can you delete a variable in
            // outside scope?!)
            let un = self.get_unmangled_name(&ei.value);
            self.ctx.borrow_mut().remove(&ei.value);
            self.ctx.borrow_mut().remove(&un);
        } else {
            let src = self.cache.ast[s].src;
            typhon_common::bail_e!(ErrorKind::DEL_INVALID, src);
        }
        Ok(())
    }

    /// `unpackAssignment(lhs, rhs, isLoopTarget)`: unpack `lhs = rhs` into simple
    /// assignments following Python unpacking rules.
    /// `(a, b) = c` -> `a = c[0]; b = c[1]`; `[a, *x, b] = c` -> `a = c[0]; x = c[1:-1]; b = c[-1]`.
    pub fn unpack_assignment(&mut self, lhs: ExprId, rhs: ExprId, is_loop_target: bool) -> Result<Option<StmtId>> {
        let left_side: Vec<ExprId> = match &self.cache.ast[lhs].kind {
            // Case: (a, b) = ...
            crate::ast::ExprKind::Tuple(t) => t.items.clone(),
            // Case: [a, b] = ...
            crate::ast::ExprKind::List(l) => l.items.clone(),
            _ => return Ok(Some(self.new_assign(lhs, Some(rhs), None, UpdateMode::Assign))),
        };

        // Prepare the right-side expression
        let old_src = self.src;
        self.src = self.cache.ast[rhs].src;
        let r = self.unpack_assignment_inner(&left_side, rhs, is_loop_target);
        self.src = old_src;
        r.map(Some)
    }

    fn unpack_assignment_inner(&mut self, left_side: &[ExprId], mut rhs: ExprId, is_loop_target: bool) -> Result<StmtId> {
        let types = self.types();
        let is_star = |s: &Self, x: ExprId| s.cache.ast[x].kind.tag() == ExprTag::Star;
        let block = self.n_stmt(SuiteStmt::default());
        // Consume each nested unpacking source once, including generators.
        let has_star = left_side.iter().any(|x| is_star(self, *x));
        let tmp = self.get_temporary_var("assign");
        let new_rhs = self.id_n(&tmp);
        let known_shape = match self.cache.ast[rhs].kind.as_tuple() {
            Some(t) => {
                !t.items.iter().any(|x| is_star(self, *x))
                    && if has_star { t.items.len() + 1 >= left_side.len() } else { t.items.len() == left_side.len() }
            }
            None => false,
        };
        let source_type = self.expr_type(rhs);
        let mut direct_loop_tuple = false;
        if is_loop_target {
            if let Some(stt) = source_type.filter(|t| types.is(*t, &stdlib().tuple)) {
                let source_size = self.get_class_fields(types.get_class(stt).unwrap()).len();
                direct_loop_tuple = if has_star { source_size + 1 >= left_side.len() } else { source_size == left_side.len() };
            }
        }
        if !direct_loop_tuple {
            let mut source = self.cache.ast.clone_expr(rhs, false);
            if !known_shape {
                let f = self.id_n(&mangle::mangled_method("std.internal.types.collections.list", "List", "_unpack", 0, 0));
                let n = self.n_expr(IntExpr::new((left_side.len() - has_star as usize) as i64));
                let b = self.n_expr(BoolExpr { value: has_star });
                source = self.call_n(f, vec![source, n, b]);
            }
            let a = self.new_assign(new_rhs, Some(source), None, UpdateMode::Assign);
            self.cache.ast.suite_add_stmt(block, Some(a));
            rhs = new_rhs;
        }

        let assignments = self.n_stmt(SuiteStmt::default());
        let mut starred_snapshot = None;
        let mut add_assignment = |s: &mut Self, target: ExprId, value: ExprId, starred: bool| -> Result<()> {
            if has_star {
                let sn = s.get_temporary_var("unpack");
                let snapshot = s.id_n(&sn);
                let extract = s.new_assign(snapshot, Some(value), None, UpdateMode::Assign);
                if starred {
                    starred_snapshot = Some(extract);
                } else {
                    s.cache.ast.suite_add_stmt(block, Some(extract));
                }
                let u = s.unpack_assignment(target, snapshot, false)?;
                s.cache.ast.suite_add_stmt(assignments, u);
            } else {
                let u = s.unpack_assignment(target, value, false)?;
                s.cache.ast.suite_add_stmt(block, u);
            }
            Ok(())
        };

        // Process assignments until the fist StarExpr (if any)
        let mut st = 0usize;
        while st < left_side.len() {
            if is_star(self, left_side[st]) {
                break;
            }
            // Transformation: `leftSide_st = rhs[st]` where `st` is static integer
            let rc = self.cache.ast.clone_expr(rhs, false);
            let i = self.n_expr(IntExpr::new(st as i64));
            let right_side = self.n_expr(IndexExpr { expr: Some(rc), index: Some(i) });
            // Recursively process the assignment because of cases like `(a, (b, c)) = d)`
            add_assignment(self, left_side[st], right_side, false)?;
            st += 1;
        }
        // Process StarExpr (if any) and the assignments that follow it
        if st < left_side.len() && is_star(self, left_side[st]) {
            // StarExpr becomes SliceExpr (e.g., `b` in `(a, *b, c) = d` becomes
            // `list(d[1:-2])`)
            let f = self.id_n(&mangle::mangled_method("std.internal.types.collections.list", "List", "_as_list", 0, 0));
            let rc = self.cache.ast.clone_expr(rhs, false);
            let start = self.n_expr(IntExpr::new(st as i64));
            // this slice is either [st:] or [st:-lhs_len + st + 1]
            let stop = if left_side.len() == st + 1 {
                None
            } else {
                Some(self.n_expr(IntExpr::new(-(left_side.len() as i64) + st as i64 + 1)))
            };
            let sl = self.n_expr(SliceExpr { start: Some(start), stop, step: None });
            let ix = self.n_expr(IndexExpr { expr: Some(rc), index: Some(sl) });
            let right_side = self.call_n(f, vec![ix]);
            let target = self.cache.ast[left_side[st]].kind.as_star().unwrap().expr.unwrap();
            add_assignment(self, target, right_side, true)?;
            st += 1;
            // Process remaining assignments. They will use negative indices (-1, -2 etc.)
            // because we do not know how big is StarExpr
            while st < left_side.len() {
                if is_star(self, left_side[st]) {
                    let src = self.cache.ast[left_side[st]].src;
                    typhon_common::bail_e!(ErrorKind::ASSIGN_MULTI_STAR, src);
                }
                let rc = self.cache.ast.clone_expr(rhs, false);
                let i = self.n_expr(IntExpr::new(-((left_side.len() - st) as i64)));
                let right_side = self.n_expr(IndexExpr { expr: Some(rc), index: Some(i) });
                add_assignment(self, left_side[st], right_side, false)?;
                st += 1;
            }
        }
        if has_star {
            self.cache.ast.suite_add_stmt(block, starred_snapshot);
            self.cache.ast.suite_add_stmt(block, Some(assignments));
        }
        Ok(block)
    }
}

impl<'a> TypecheckVisitor<'a> {
    fn assign_set(&mut self, s: StmtId, f: impl FnOnce(&mut AssignStmt)) {
        f(self.cache.ast[s].kind.as_assign_mut().unwrap());
    }
    fn tr_type_opt(&mut self, e: Option<ExprId>) -> Result<Option<ExprId>> {
        self.transform_type(e, false)
    }

    /// `transformAssignment(stmt, mustExist)`.
    /// `a[x] = b` -> `a.__setitem__(x, b)`; `a.x = b` -> `AssignMemberStmt`;
    /// `a: type = b` / `a = b` -> `AssignStmt` or an update.
    pub fn transform_assignment(&mut self, s: StmtId, must_exist: bool) -> Result<Option<StmtId>> {
        let st = stdlib();
        let types = self.types();
        let stmt = self.as_assign(s);
        let lhs = stmt.lhs.unwrap();

        if let Some(idx) = self.cache.ast[lhs].kind.as_index().cloned() {
            // Case: a[x] = b
            typhon_common::seqassert!(stmt.type_expr.is_none(), "unexpected type annotation");
            if let Some(b) = stmt.rhs.and_then(|r| self.cache.ast[r].kind.as_binary().cloned()) {
                // Case: a[x] += b (inplace operator)
                if must_exist && b.in_place && !b.rexpr.is_some_and(|r| self.cache.ast[r].kind.tag() == ExprTag::Id) {
                    let var = self.get_temporary_var("assign");
                    let v = self.id_n(&var);
                    let a1 = self.new_assign(v, idx.index, None, UpdateMode::Assign);
                    let d = self.dot_n(idx.expr.unwrap(), "__setitem__");
                    let v1 = self.id_n(&var);
                    let ec = self.cache.ast.clone_expr(idx.expr.unwrap(), false);
                    let v2 = self.id_n(&var);
                    let ix = self.n_expr(IndexExpr { expr: Some(ec), index: Some(v2) });
                    let bin = self.n_expr(BinaryExpr { op: b.op.clone(), lexpr: Some(ix), rexpr: b.rexpr, in_place: true });
                    let c = self.call_n(d, vec![v1, bin]);
                    let es = self.expr_stmt_n(c);
                    let suite = self.n_stmt(SuiteStmt { items: vec![a1, es] });
                    return self.transform_stmt_opt(Some(suite));
                }
            }
            let d = self.dot_n(idx.expr.unwrap(), "__setitem__");
            let mut args = vec![idx.index.unwrap()];
            args.extend(stmt.rhs);
            let c = self.call_n(d, args);
            let es = self.expr_stmt_n(c);
            return self.transform_stmt_opt(Some(es));
        }

        if let Some(dot) = self.cache.ast[lhs].kind.as_dot().cloned() {
            // Case: a.x = b
            let de = self.transform_allow(dot.expr, true)?;
            self.cache.ast[lhs].kind.as_dot_mut().unwrap().expr = de;
            let rhs = self.transform(stmt.rhs)?;
            let am = self.n_stmt(AssignMemberStmt { lhs: de, member: dot.member, rhs, type_expr: stmt.type_expr });
            return self.transform_stmt_opt(Some(am));
        }

        // Case: a (: t) = b
        let Some(name) = self.cache.ast[lhs].kind.as_id().map(|i| i.value.clone()) else {
            let src = self.cache.ast[lhs].src;
            typhon_common::bail_e!(ErrorKind::ASSIGN_INVALID, src);
        };
        // Never do undef checks on assignments!
        self.cache.ast[lhs].attrs.set(Attr::ExprNoUndefCheck);

        // Ensure that captured values are in a Capsule
        if self.ctx.borrow().in_function() && stmt.rhs.is_some() && !must_exist {
            let func = self.ctx.borrow().base_ref().func;
            let nonlocal = func
                .and_then(|f| self.cache.ast[f].attrs.get_bindings().map(|b| b.bindings.get(&name).is_some_and(|bd| bd.is_nonlocal)))
                .unwrap_or(false);
            if nonlocal {
                let cap = self.id_n(&st.capsule);
                let t = match self.as_assign(s).type_expr {
                    Some(te) => self.n_expr(IndexExpr { expr: Some(cap), index: Some(te) }),
                    None => cap,
                };
                self.assign_set(s, |a| a.type_expr = Some(t));
            }
        }

        let mut is_thread_local = false;
        let orig_type = self.as_assign(s).type_expr;
        let mut type_expr = self.tr_type_opt(orig_type)?;
        if let Some(te) = type_expr {
            if types.is(self.extract_type_expr(te), &st.thread_local) {
                is_thread_local = true;
                // PORT-NOTE: C++ inspects the original (pre-transform) annotation node, which
                // is transformed in place; the original id is used here.
                match orig_type.and_then(|o| self.cache.ast[o].kind.as_index().cloned()) {
                    Some(ti) => type_expr = self.tr_type_opt(ti.index)?,
                    None => type_expr = None,
                }
            }
        }

        // Make sure that existing values that cannot be shadowed are only updated
        // mustExist |= val && !ctx->isOuter(val);
        if must_exist {
            let t = self.get_time();
            if self.find(&name, t).is_none() {
                let src = self.cache.ast[lhs].src;
                typhon_common::bail_e!(ErrorKind::ASSIGN_LOCAL_REFERENCE, src, name, src);
            }
            let rhs = self.as_assign(s).rhs;
            let atomic = {
                let c = self.ctx.borrow();
                let b = c.base_ref();
                !b.is_type() && b.func.is_some_and(|f| self.cache.ast[f].attrs.has(Attr::Atomic))
            };
            let mode = if atomic { UpdateMode::UpdateAtomic } else { UpdateMode::Update };
            let ns = self.new_assign(lhs, rhs, type_expr, mode);
            return match self.transform_update(ns)? {
                Some(u) => Ok(Some(u)),
                None => Ok(Some(ns)), // delay
            };
        }

        // Construct collection literals with their declared union storage (union_*_layout).
        let rhs0 = self.as_assign(s).rhs;
        if let (Some(te), Some(r)) = (type_expr, rhs0) {
            if matches!(self.cache.ast[r].kind.tag(), ExprTag::List | ExprTag::Set | ExprTag::Dict) {
                let t = self.extract_type_expr(te);
                self.cache.ast[r].expected_type = Some(t);
            }
        }
        let rhs = self.transform_allow(rhs0, true)?;
        self.assign_set(s, |a| {
            a.rhs = rhs;
            a.type_expr = type_expr;
        });

        // Generate new canonical variable name for this assignment and add it to the context
        let canonical = if let Some(base) = name.strip_suffix(VAR_USED_SUFFIX) {
            let f = self.force_find(base);
            let c = f.borrow().canonical_name.clone();
            c + VAR_USED_SUFFIX
        } else {
            let ctx = self.ctx.clone();
            let c = ctx.borrow();
            c.generate_canonical_name(self.cache, &name, false, false)
        };
        let new_lhs = self.id_n(&canonical);
        let a = self.new_assign(new_lhs, rhs, type_expr, UpdateMode::Assign);
        let lattrs = self.cache.ast[lhs].attrs.clone();
        self.cache.ast[new_lhs].attrs = lattrs;
        let lt = match self.expr_type(lhs) {
            Some(t) => t,
            None => {
                let l = self.ctx.borrow().typecheck_level;
                self.instantiate_unbound_at(self.cache.ast[new_lhs].src, l)
            }
        };
        self.set_type(new_lhs, lt);
        if is_thread_local {
            self.assign_set(a, |x| x.update = UpdateMode::ThreadLocalAssign);
        }
        if rhs.is_none() && type_expr.is_none() && self.find(&st.none_type, 0).is_some() {
            // All declarations that are not handled are to be marked with NoneType later on
            // (useful for dangling declarations that are not initialized afterwards due to
            //  static check)
            let nt = self.get_stdlib_type(&st.none_type)?;
            if let Some(l) = types.get_link(lt) {
                types.with_link_mut(l, |x| x.default_type = Some(nt));
            }
            self.ctx.borrow_mut().get_base().pending_defaults.entry(1).or_default().insert(lt);
        }
        if let Some(te) = type_expr {
            let t = self.extract_type_expr(te);
            let old = std::mem::replace(&mut self.src, self.cache.ast[te].src);
            let it = self.instantiate_type(t, None);
            self.src = old;
            self.unify(lt, it?)?;
        }
        let val = {
            let c = self.ctx.borrow();
            let mut v = TypecheckItem::new(canonical.clone(), c.base_name(), c.module(), Some(lt), c.block_level);
            v.time = self.get_time();
            v.src = self.src;
            v.into_ref()
        };
        self.ctx.borrow_mut().add(&name, val.clone());
        TypeContext::add_always_visible(self.cache, &self.ctx, val.clone(), false);

        if let Some(ar) = rhs {
            // not a declaration!
            // Inferred chain temporaries preserve compile-time comparison operands.
            if self.cache.ast[s].attrs.has(Attr::MaybeLiteral) {
                if let Some(u) = types.get_unbound(lt) {
                    let k = types.static_kind(self.ety(ar));
                    types.with_link_mut(u, |l| l.static_kind = k);
                }
                self.cache.ast[a].attrs.set(Attr::MaybeLiteral);
            }
            // Check if we can wrap the expression (e.g., `a: float = 3` -> `a = float(3)`)
            let mut r = ar;
            let ok = self.wrap_expr(&mut r, lt, None, true)?;
            self.assign_set(a, |x| x.rhs = Some(r));
            if ok {
                let rt = self.ety(r);
                self.unify(lt, rt)?;
            }

            // Generalize non-variable types. That way we can support cases like:
            // `a = foo(x, ...); a(1); a('s')`
            if !val.borrow().is_var(&types) {
                let lvl = self.ctx.borrow().typecheck_level - 1;
                let g = types.generalize(val.borrow().ty(), lvl);
                val.borrow_mut().ty = Some(g);
                // See capture_function_partial_proper_realize test
                self.set_type(new_lhs, g);
                self.set_type(r, g);
            }
        }

        // Mark declarations or generalized type/functions as done
        let arhs = self.as_assign(a).rhs;
        let lt = self.ety(new_lhs);
        if (arhs.is_none() || arhs.is_some_and(|r| self.cache.ast[r].done)) && types.can_realize(lt) {
            if let Some(r) = self.realize(lt)? {
                // overwrite types to remove dangling unbounds with some partials...
                self.set_type(new_lhs, r);
                if let Some(ar) = arhs {
                    self.set_type(ar, r);
                }
                self.set_done_stmt(a);
            }
        } else if arhs.is_some() && !val.borrow().is_var(&types) && !types.has_unbounds(val.borrow().ty(), false) {
            self.set_done_stmt(a);
        }

        // Register all toplevel variables as global in JIT mode
        //   OR if they are in imported module (not toplevel)
        let jit = self.cache.options.as_ref().is_some_and(|o| o.jit);
        let (is_g, is_gen, module) = {
            let v = val.borrow();
            (v.is_global(), v.is_generic(), v.module().to_string())
        };
        let is_global = (jit && is_g && !is_gen) || canonical == var_argv() || canonical == var_argc() || (is_g && !module.is_empty());
        if is_global && val.borrow().is_var(&types) {
            self.register_global(&canonical);
            if jit {
                let un = self.get_unmangled_name(&val.borrow().canonical_name);
                if let Some(sc) = self.cache.imports.get(STDLIB_IMPORT).and_then(|i| i.ctx.clone()) {
                    sc.borrow_mut().add_toplevel(&un, val.clone());
                }
            }
        }
        Ok(Some(a))
    }

    /// `transformUpdate(stmt)`: binding updates. Special handling is done for atomic or
    /// in-place statements (e.g., `a += b`).
    pub fn transform_update(&mut self, s: StmtId) -> Result<Option<StmtId>> {
        let lhs = self.transform(self.as_assign(s).lhs)?;
        self.assign_set(s, |a| a.lhs = lhs);
        let lhs = lhs.unwrap();

        // Check inplace updates
        let (in_place, in_place_stmt) = self.transform_inplace_update(s)?;
        if in_place {
            return Ok(in_place_stmt);
        }

        let rhs = self.transform(self.as_assign(s).rhs)?;
        let te = self.tr_type_opt(self.as_assign(s).type_expr)?;
        self.assign_set(s, |a| {
            a.rhs = rhs;
            a.type_expr = te;
        });
        if let Some(te) = te {
            let t = self.extract_type_expr(te);
            let old = std::mem::replace(&mut self.src, self.cache.ast[te].src);
            let it = self.instantiate_type(t, None);
            self.src = old;
            let lt = self.ety(lhs);
            self.unify(lt, it?)?;
        }

        // Annotation-only updates refine the hoisted variable's type, not its value.
        let Some(mut r) = rhs else {
            if self.realize(self.ety(lhs))?.is_some() {
                self.set_done_stmt(s);
            }
            return Ok(None);
        };

        // Case: wrap expressions if needed (e.g. floats or optionals)
        let lt = self.ety(lhs);
        let ok = self.wrap_expr(&mut r, lt, None, true)?;
        self.assign_set(s, |a| a.rhs = Some(r));
        if ok {
            let rt = self.ety(r);
            self.unify(rt, lt)?;
        }
        if self.cache.ast[r].done && self.realize(lt)?.is_some() {
            self.set_done_stmt(s);
        }
        Ok(None)
    }
}

impl<'a> TypecheckVisitor<'a> {
    /// `visit(AssignMemberStmt*)`: typecheck instance member assignments (e.g., `a.b = c`)
    /// and handle optional instances. Disallow tuple updates.
    /// `opt.foo = bar` -> `unwrap(opt).foo = wrap(bar)`
    pub fn visit_assign_member(&mut self, s: StmtId) -> Result<()> {
        let st = stdlib();
        let types = self.types();
        let am = self.cache.ast[s].kind.as_assign_member().unwrap().clone();
        let lhs = self.transform(am.lhs)?.unwrap();
        self.cache.ast[s].kind.as_assign_member_mut().unwrap().lhs = Some(lhs);
        let member_name = am.member.clone();
        let rhs0 = am.rhs.unwrap();

        let Some(lhs_class) = self.extract_class_type_expr(lhs) else { return Ok(()) };
        // enum_reassign_err: use the existing member globals, without a second
        // registry of enum names in the class cache.
        if self.is_enum(lhs_class) && !member_name.starts_with("__") {
            let var = self.cache.get_class_by_type(lhs_class).and_then(|c| c.class_vars.get(&member_name).cloned());
            if let Some(var) = var {
                let vt = self.force_find(&var).borrow().ty();
                if types.is(vt, &types.class_name(lhs_class).unwrap()) {
                    let src = self.cache.ast[s].src;
                    typhon_common::bail_e!(ErrorKind::CUSTOM, src, "cannot reassign enumeration member '{}'", member_name);
                }
            }
        }
        let member = self.find_member(lhs_class, &member_name);
        if member.is_none() {
            // Case: property setters
            let setters = self.find_method(Some(lhs_class), &format!("{FN_SETTER_SUFFIX}{member_name}"))?;
            if let Some(f) = setters.first() {
                let fi = self.id_n(&types.func_name(*f));
                let c = self.call_n(fi, vec![lhs, rhs0]);
                let es = self.expr_stmt_n(c);
                self.result_stmt = self.transform_stmt_opt(Some(es))?;
                return Ok(());
            }
            // Case: class variables
            let var = self.cache.get_class_by_type(lhs_class).and_then(|c| c.class_vars.get(&member_name).cloned());
            if let Some(var) = var {
                let v = self.id_n(&var);
                let r = self.transform(Some(rhs0))?;
                let a = self.new_assign(v, r, None, UpdateMode::Update);
                self.result_stmt = self.transform_stmt_opt(Some(a))?;
                return Ok(());
            }
        }
        if member.is_none() && types.is(lhs_class, &st.optional) {
            // Unwrap optional and look up there
            let f = self.id_n(&fn_optional_unwrap());
            let c = self.call_n(f, vec![lhs]);
            let n = self.n_stmt(AssignMemberStmt { lhs: Some(c), member: member_name, rhs: Some(rhs0), type_expr: None });
            self.result_stmt = self.transform_stmt_opt(Some(n))?;
            return Ok(());
        }
        // Case: transform `pyobj.member = value` to `pyobj._setattr("member", value)`.
        // Mirrors the read path in `getClassMember`, deferring the attribute write to
        // CPython at runtime.
        if member.is_none() && types.is(lhs_class, "pyobj") {
            let d = self.dot_n(lhs, "_setattr");
            let m = self.n_expr(StringExpr::new(member_name, ""));
            let c = self.call_n(d, vec![m, rhs0]);
            let es = self.expr_stmt_n(c);
            self.result_stmt = self.transform_stmt_opt(Some(es))?;
            return Ok(());
        }
        // Case: __setattr__ support. Ensure that only Literal[str] arguments are accepted.
        if member.is_none() {
            let (u, v) = (self.instantiate_unbound(), self.instantiate_unbound());
            types.with_link_mut(u, |l| l.static_kind = LiteralKind::String);
            if let Some(m) = self.find_best_method_types(lhs_class, "__setattr__", &[lhs_class, u, v])? {
                if !types.func_generics(m).is_empty() && types.static_kind(self.extract_func_generic(m, 0)) == LiteralKind::String {
                    let d = self.dot_n(lhs, "__setattr__");
                    let mn = self.n_expr(StringExpr::new(member_name, ""));
                    let c = self.call_n(d, vec![mn, rhs0]);
                    let es = self.expr_stmt_n(c);
                    self.result_stmt = self.transform_stmt_opt(Some(es))?;
                    return Ok(());
                }
            }
        }

        let Some(member) = member else {
            let src = self.cache.ast[lhs].src;
            typhon_common::bail_e!(ErrorKind::DOT_NO_ATTR, src, types.pretty_string(lhs_class), member_name);
        };
        if types.is_record(lhs_class) {
            // prevent tuple member assignment
            let src = self.cache.ast[lhs].src;
            typhon_common::bail_e!(ErrorKind::ASSIGN_UNEXPECTED_FROZEN, src);
        }

        let rhs = self.transform(Some(rhs0))?.unwrap();
        let te = self.tr_type_opt(am.type_expr)?;
        {
            let k = self.cache.ast[s].kind.as_assign_member_mut().unwrap();
            k.rhs = Some(rhs);
            k.type_expr = te;
        }
        if let Some(te) = te {
            let t = self.extract_type_expr(te);
            let old = std::mem::replace(&mut self.src, self.cache.ast[te].src);
            let it = self.instantiate_type(t, None);
            self.src = old;
            let rt = self.ety(rhs);
            self.unify(rt, it?)?;
        }

        let old = std::mem::replace(&mut self.src, self.cache.ast[lhs].src);
        let ftyp = self.instantiate_type(member.ty.expect("member without type"), Some(lhs_class));
        self.src = old;
        let ftyp = ftyp?;
        if !types.can_realize(ftyp) {
            if let Some(mte) = member.type_expr {
                let x = self.with_class_generics(lhs_class, false, false, false, |v| {
                    let c = v.cache.ast.clone_expr(mte, true);
                    v.tr(c)
                })?;
                let t = self.extract_type_expr(x);
                self.unify(ftyp, t)?;
            }
        }

        let mut r = rhs;
        let ok = self.wrap_expr(&mut r, ftyp, None, true)?;
        self.cache.ast[s].kind.as_assign_member_mut().unwrap().rhs = Some(r);
        if !ok {
            return Ok(());
        }
        let rt = self.ety(r);
        self.unify(rt, ftyp)?;
        if self.cache.ast[r].done {
            self.set_done_stmt(s);
        }
        Ok(())
    }

    /// `transformInplaceUpdate(stmt)`: in-place and atomic updates.
    /// `a += b` -> `a.__iadd__(a, b)`; capsules: `a = b` -> `a.val[0] = b`;
    /// atomics: `a = b` -> `type(a).__atomic_xchg__(__ptr__(a), b)`,
    /// `a = min(a, b)` -> `type(a).__atomic_min__(__ptr__(a), b)`.
    /// Returns whether the update can be replaced, and the replacement.
    pub fn transform_inplace_update(&mut self, s: StmtId) -> Result<(bool, Option<StmtId>)> {
        let st = stdlib();
        let types = self.types();
        let stmt = self.as_assign(s);
        let lhs = stmt.lhs.unwrap();
        let is_atomic = stmt.is_atomic_update();

        // Case: capsule operations
        if let Some(c) = self.cache.ast[lhs].kind.as_call().cloned() {
            if self.is_function_expr(c.expr, &mangle::mangled_method("", "Capsule", "_get", 0, 0)) {
                let f = self.id_n(&mangle::mangled_method("", "Capsule", "_ptr", 0, 0));
                let cc = self.call_n(f, vec![c.items[0].value.unwrap()]);
                let z = self.n_expr(IntExpr::new(0));
                let ix = self.n_expr(IndexExpr { expr: Some(cc), index: Some(z) });
                let a = self.new_assign(ix, stmt.rhs, None, UpdateMode::Assign);
                return Ok((true, self.transform_stmt_opt(Some(a))?));
            }
        }

        // Case: in-place updates (e.g., `a += b`).
        // They are stored as `Update(a, Binary(a + b, inPlace=true))`
        let rhs = stmt.rhs.unwrap();
        if let Some(bin) = self.cache.ast[rhs].kind.as_binary().cloned().filter(|b| b.in_place) {
            let bl = self.transform(bin.lexpr)?;
            let br = self.transform(bin.rexpr)?;
            {
                let b = self.cache.ast[rhs].kind.as_binary_mut().unwrap();
                b.lexpr = bl;
                b.rexpr = br;
            }
            if self.cache.ast[rhs].ty.is_none() {
                let u = self.instantiate_unbound();
                self.set_type(rhs, u);
            }
            if self.expr_class(bl.unwrap()).is_some() && self.expr_class(br.unwrap()).is_some() {
                if let Some(t) = self.transform_binary_inplace_magic(rhs, is_atomic)? {
                    let (a, b) = (self.ety(rhs), self.ety(t));
                    self.unify(a, b)?;
                    let es = self.expr_stmt_n(t);
                    return Ok((true, self.transform_stmt_opt(Some(es))?));
                } else if !is_atomic {
                    // If atomic, call normal magic and then use __atomic_xchg__ below
                    return Ok((false, None));
                }
            } else {
                // Delay
                let u = self.instantiate_unbound();
                let rt = self.ety(rhs);
                let x = self.unify(rt, u)?;
                let lt = self.ety(lhs);
                self.unify(lt, x)?;
                return Ok((true, None));
            }
        }

        // Case: atomic min/max operations.
        // Note: check only `a = min(a, b)`; does NOT check `a = min(b, a)`
        let lhs_class = self.extract_class_type_expr(lhs);
        let rhs = self.as_assign(s).rhs.unwrap();
        let call = self.cache.ast[rhs].kind.as_call().cloned();
        let lei = self.cache.ast[lhs].kind.as_id().map(|i| i.value.clone());
        let cei = call.as_ref().and_then(|c| c.expr).and_then(|x| self.cache.ast[x].kind.as_id().map(|i| i.value.clone()));
        if let (true, Some(call), Some(lei), Some(cei)) = (is_atomic, call, lei, cei) {
            if (cei == "min" || cei == "max") && call.items.len() == 2 {
                let f0 = self.transform(call.items[0].value)?;
                self.cache.ast[rhs].kind.as_call_mut().unwrap().items[0].value = f0;
                if f0.and_then(|f| self.cache.ast[f].kind.as_id()).is_some_and(|i| i.value == lei) {
                    // `type(a).__atomic_min__(__ptr__(a), b)`
                    let lc = lhs_class.expect("lhs class");
                    let p = self.get_stdlib_type(&st.ptr)?;
                    let old = std::mem::replace(&mut self.src, self.cache.ast[lhs].src);
                    let ptr_typ = self.instantiate_type_with(p, &[lc]);
                    self.src = old;
                    let ptr_typ = ptr_typ?;
                    let a1 = self.transform(call.items[1].value)?;
                    self.cache.ast[rhs].kind.as_call_mut().unwrap().items[1].value = a1;
                    let rhs_typ = self.extract_class_type_expr(a1.unwrap());
                    if let Some(rt) = rhs_typ {
                        if let Some(m) = self.find_best_method_types(lc, &format!("__atomic_{cei}__"), &[ptr_typ, rt])? {
                            let fi = self.id_n(&types.func_name(m));
                            let pi = self.id_n("__ptr__");
                            let pc = self.call_n(pi, vec![lhs]);
                            let c = self.call_n(fi, vec![pc, a1.unwrap()]);
                            let es = self.expr_stmt_n(c);
                            return Ok((true, self.transform_stmt_opt(Some(es))?));
                        }
                    }
                }
            }
        }

        // Case: atomic assignments
        if let (true, Some(lc)) = (is_atomic, lhs_class) {
            // `type(a).__atomic_xchg__(__ptr__(a), b)`
            let r = self.transform(Some(rhs))?;
            self.assign_set(s, |a| a.rhs = r);
            let r = r.unwrap();
            if let Some(rc) = self.expr_class(r) {
                let p = self.get_stdlib_type(&st.ptr)?;
                let old = std::mem::replace(&mut self.src, self.cache.ast[lhs].src);
                let ptr_type = self.instantiate_type_with(p, &[lc]);
                self.src = old;
                let ptr_type = ptr_type?;
                if let Some(m) = self.find_best_method_types(lc, "__atomic_xchg__", &[ptr_type, rc])? {
                    let fi = self.id_n(&types.func_name(m));
                    let pi = self.id_n("__ptr__");
                    let pc = self.call_n(pi, vec![lhs]);
                    let c = self.call_n(fi, vec![pc, r]);
                    let es = self.expr_stmt_n(c);
                    return Ok((true, self.transform_stmt_opt(Some(es))?));
                }
            }
        }
        Ok((false, None))
    }
}
