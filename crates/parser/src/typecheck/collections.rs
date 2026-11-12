
use std::collections::BTreeMap;

use typhon_common::mangle::stdlib;
use typhon_common::{ErrorKind, Result};

use super::{AstNode, TypecheckVisitor};
use crate::ast::{
    AssignExpr, AssignStmt, Attr, CallExpr, ExprStmt, ForStmt, GeneratorKind, IdExpr, IfExpr, IndexExpr, InstantiateExpr, IntExpr,
    StarExpr, StmtExpr, StringExpr, SuiteStmt, TupleExpr, UpdateMode,
};
use crate::ids::{ExprId, StmtId, TypeId};
use crate::types::Unification;

impl<'a> TypecheckVisitor<'a> {
    fn assign_n(&mut self, l: ExprId, r: ExprId) -> StmtId {
        self.n_stmt(AssignStmt { lhs: Some(l), rhs: Some(r), type_expr: None, update: UpdateMode::Assign })
    }

    /// `visit(TupleExpr*)`: `(a1, ..., aN)` -> `Tuple.__new__(a1, ..., aN)`.
    pub fn visit_tuple(&mut self, e: ExprId) -> Result<()> {
        let items = self.ast()[e].kind.as_tuple().unwrap().items.clone();
        let t = self.id_n(&stdlib().tuple);
        let d = self.dot_n(t, "__new__");
        let c = self.call_n(d, items);
        self.result_expr = self.transform(Some(c))?;
        Ok(())
    }

    fn visit_collection(&mut self, e: ExprId, ty: &str, f: &str, attr: Attr) -> Result<()> {
        let u = self.instantiate_unbound();
        self.set_type(e, u);
        let mut items = match &self.ast()[e].kind {
            crate::ast::ExprKind::List(l) => l.items.clone(),
            crate::ast::ExprKind::Set(l) => l.items.clone(),
            crate::ast::ExprKind::Dict(l) => l.items.clone(),
            _ => unreachable!(),
        };
        let expected = self.ast()[e].expected_type;
        let r = self.transform_comprehension(ty, f, &mut items, expected);
        match &mut self.ast_mut()[e].kind {
            crate::ast::ExprKind::List(l) => l.items = items,
            crate::ast::ExprKind::Set(l) => l.items = items,
            crate::ast::ExprKind::Dict(l) => l.items = items,
            _ => {}
        }
        self.result_expr = r?;
        if let Some(r) = self.result_expr {
            self.ast_mut()[r].attrs.set(attr);
        }
        Ok(())
    }

    /// `visit(ListExpr*)`: see [`Self::transform_comprehension`].
    pub fn visit_list(&mut self, e: ExprId) -> Result<()> {
        self.visit_collection(e, &stdlib().list, "append", Attr::ExprList)
    }
    /// `visit(SetExpr*)`.
    pub fn visit_set(&mut self, e: ExprId) -> Result<()> {
        self.visit_collection(e, &stdlib().set, "add", Attr::ExprSet)
    }
    /// `visit(DictExpr*)`.
    pub fn visit_dict(&mut self, e: ExprId) -> Result<()> {
        self.visit_collection(e, &stdlib().dict, "__setitem__", Attr::ExprDict)
    }

    /// Lowest common type of a collection slot (`lctyp` lambda).
    fn collection_lctyp(&mut self, coll: Option<TypeId>, ti: TypeId) -> Result<Option<TypeId>> {
        let types = self.types();
        let st = stdlib();
        let Some(coll) = coll else { return Ok(Some(ti)) };
        let opt = |v: &mut Self, m: TypeId| -> Result<TypeId> {
            let o = v.get_stdlib_type(&st.optional)?;
            v.instantiate_type_with(o, &[m])
        };
        // NoneType introduces nullability without becoming the element type.
        if types.is(coll, &st.none_type) != types.is(ti, &st.none_type) {
            let member = if types.is(coll, &st.none_type) { ti } else { coll };
            return Ok(Some(if types.is(member, &st.optional) { member } else { opt(self, member)? }));
        }
        let (cn, tn) = (types.class_name(coll).unwrap(), types.class_name(ti).unwrap());
        if types.is(coll, "Int") && types.is(ti, "float") {
            // Rule: int derives from float
            return Ok(Some(ti));
        } else if cn.as_str() != st.optional && tn.as_str() == st.optional {
            // Rule: T derives from Optional[T]
            return Ok(Some(opt(self, coll)?));
        } else if cn.as_str() == st.optional && tn.as_str() != st.optional {
            return Ok(Some(opt(self, ti)?));
        } else if !types.is(coll, "pyobj") && types.is(ti, "pyobj") {
            // Rule: anything derives from pyobj
            return Ok(Some(ti));
        } else if cn != tn {
            // Rule: subclass derives from its base class
            let mros = self.get_class_type(coll).mro.clone();
            let tmros = self.get_class_type(ti).mro.clone();
            for m in mros {
                let t = self.instantiate_type(m, Some(coll))?;
                for tm in &tmros {
                    let tj = self.instantiate_type(*tm, Some(ti))?;
                    if types.unify(t, tj, None, self)? >= 0 {
                        return Ok(Some(t));
                    }
                }
            }
        }
        Ok(None)
    }

    /// `Union[members...]` type (via `transformType`).
    fn union_of(&mut self, members: &[TypeId]) -> Result<TypeId> {
        let types = self.types();
        let mut ms = vec![];
        for t in members {
            let n = types.class_name(types.get_class(*t).unwrap()).unwrap().to_string();
            let m = self.id_n(&n);
            let tv = self.instantiate_type_var(*t)?;
            self.set_type(m, tv);
            ms.push(m);
        }
        let u = self.id_n(&stdlib().union);
        let ie = self.n_expr(InstantiateExpr { expr: Some(u), items: ms });
        let r = self.transform_type(Some(ie), false)?.unwrap();
        Ok(self.extract_type_expr(r))
    }

    /// `transformComprehension(type, fn, items, expectedType)`: transform a collection of
    /// type `type` to a statement expression:
    ///   `[a1, ..., aN]` -> `cont = [type](); (cont.[fn](a1); ...); cont`.
    /// Star-expressions are expanded (`[a, *b]` -> `cont.[fn](a); for i in b: cont.[fn](i)`).
    pub fn transform_comprehension(&mut self, ty: &str, f: &str, items: &mut Vec<ExprId>, expected: Option<TypeId>) -> Result<Option<ExprId>> {
        let types = self.types();
        let st = stdlib();
        let is_dict = ty == st.dict;
        let mut root = self.get_stdlib_type(ty)?;
        if is_dict {
            // Instantiate Dict's class template, not the alias.
            let n = types.class_name(root).unwrap().to_string();
            root = self.get_stdlib_type(&n)?;
        }
        let root_name = types.class_name(root).unwrap().to_string();
        let nslots = if is_dict { 2 } else { 1 };
        let mut element_types: Vec<TypeId> = vec![];
        let mut declared_unions = vec![];
        let mut declared_slot = vec![];
        for di in 0..nslots {
            // Dict is an alias: compare the resolved class, e.g. OrderedDict.
            let member = expected.filter(|t| types.is(*t, &root_name)).map(|t| self.extract_class_generic(t, di));
            let du = member.is_some_and(|m| types.get_union(m).is_some());
            declared_unions.push(du);
            declared_slot.push(member.is_some());
            element_types.push(if du { member.unwrap() } else { self.instantiate_unbound() });
        }
        // Empty unannotated literal inside a function: its slots are inferred from later
        // inserts. Track them so that conflicting inserts (no common type) can widen the
        // slot to a Union; the realization is then restarted (see realizeFunc).
        let mut empty_slot_keys: Vec<(usize, String)> = vec![];
        let owner = self.empty_literal_owner();
        if items.is_empty() && self.ctx.borrow().in_function() && !owner.is_empty() {
            let src = self.src;
            let site = format!("{}\x01{}:{}:{}", owner, src.file, src.line, src.col);
            for di in 0..element_types.len() {
                if declared_slot[di] {
                    continue;
                }
                let key = format!("{site}\x01{di}");
                if let Some(u) = self.cache.empty_literal_unions.get(&key).cloned() {
                    element_types[di] = self.union_of(&u)?;
                }
                empty_slot_keys.push((di, key));
            }
        }
        // Heterogeneous slots (no lowest common type) are inferred as Union[T1, T2, ...]
        // of the distinct item types, as if the user had annotated them.
        let mut need_union = vec![false; element_types.len()];
        let mut seen: Vec<BTreeMap<String, Option<TypeId>>> = vec![BTreeMap::new(); element_types.len()];
        let mut done = true;
        for i in items.iter_mut() {
            let mut typ: Option<TypeId> = None;
            let is_star = !is_dict && self.ast()[*i].kind.as_star().is_some();
            let is_kw = is_dict && self.ast()[*i].kind.as_keyword_star().is_some();
            if is_star || is_kw {
                let inner = {
                    let k = &self.ast()[*i].kind;
                    k.as_star().map(|s| s.expr).or_else(|| k.as_keyword_star().map(|s| s.expr)).flatten().unwrap()
                };
                let d = self.dot_n(inner, if is_star { "__iter__" } else { "items" });
                let c = self.call_n(d, vec![]);
                let x = self.tr(c)?;
                match &mut self.ast_mut()[*i].kind {
                    crate::ast::ExprKind::Star(s) => s.expr = Some(x),
                    crate::ast::ExprKind::KeywordStar(s) => s.expr = Some(x),
                    _ => {}
                }
                let xt = self.ety(x);
                if types.is(xt, "Generator") {
                    typ = types.get_class(self.extract_class_generic(xt, 0));
                }
            } else {
                *i = self.tr(*i)?;
                typ = self.expr_class(*i);
            }
            let Some(mut typ) = typ else {
                done = false;
                continue;
            };
            if is_dict {
                let t2 = self.generate_tuple(2, true)?;
                let t2 = self.instantiate_type(t2, None)?;
                let u = self.unify(typ, t2)?;
                typ = types.get_class(u).unwrap();
            }
            for di in 0..element_types.len() {
                // A declared union fixes storage; only inferred slots participate in lctyp.
                // Keep dict keys/values independent (union_collection_common_type).
                if declared_unions[di] {
                    continue;
                }
                let member = if is_dict { self.extract_class_generic(typ, di) } else { typ };
                if types.can_realize(member) {
                    seen[di].insert(types.realized_name(member), Some(member));
                } else {
                    seen[di].insert(String::new(), None); // not realizable (yet): never build a union from it
                }
                let ec = types.get_class(element_types[di]);
                if ec.is_none() {
                    self.unify(element_types[di], member)?;
                } else if let Some(t) = self.collection_lctyp(ec, types.get_class(member).unwrap())? {
                    element_types[di] = t;
                } else if !declared_slot[di] {
                    need_union[di] = true; // candidate: re-checked against the final slot type
                }
            }
        }
        if !done {
            return Ok(None);
        }
        for di in 0..element_types.len() {
            if !need_union[di] {
                continue;
            }
            // Some item type is not realizable: keep the pre-existing behaviour
            // (no union inference) to avoid delaying the expression forever.
            if seen[di].contains_key("") {
                continue;
            }
            // Only switch to a union if some item does not fit the final slot type
            // through unification, lowest-common-type rules or implicit wrapping.
            let fin = element_types[di];
            let mut fits = types.get_class(fin).is_some() && types.can_realize(fin);
            for t in seen[di].values().map(|t| t.unwrap()).collect::<Vec<_>>() {
                if !fits {
                    break;
                }
                if types.unify(fin, t, None, self)? >= 0 {
                    continue;
                }
                let lt = self.collection_lctyp(types.get_class(fin), types.get_class(t).unwrap())?;
                if let Some(lt) = lt {
                    if types.unify(lt, fin, None, self)? >= 0 {
                        continue;
                    }
                }
                let (ok, _, wf) = self.can_wrap_expr(t, fin, None, true, false)?;
                fits = ok && wf.is_some();
            }
            if fits || types.get_class(fin).is_none() {
                continue;
            }
            let members: Vec<TypeId> = seen[di].values().map(|t| t.unwrap()).collect();
            element_types[di] = self.union_of(&members)?;
        }
        let mut stmts = vec![];
        let var_name = self.get_temporary_var("cont");
        let var = self.id_n(&var_name);

        let mut constructor_args = vec![];
        if ty == st.list && !items.is_empty() {
            // Optimization: pre-allocate the list with the exact number of elements
            constructor_args.push(self.n_expr(IntExpr::new(items.len() as i64)));
        }
        let t = self.id_n(if is_dict { &root_name } else { ty });
        let ta = self.instantiate_type_with(root, &element_types)?;
        let tv = self.instantiate_type_var(ta)?;
        self.set_type(t, tv);
        let v0 = self.cache.ast.clone_expr(var, false);
        let ctor = self.call_n(t, constructor_args);
        stmts.push(self.assign_n(v0, ctor));
        for it in items.iter() {
            let it = *it;
            let is_star = !is_dict && self.ast()[it].kind.as_star().is_some();
            let is_kw = is_dict && self.ast()[it].kind.as_keyword_star().is_some();
            if is_star || is_kw {
                // Unpack star-expression by iterating over it
                // `*star` -> `for i in star: cont.[fn](i)`
                let inner = {
                    let k = &self.ast()[it].kind;
                    k.as_star().map(|s| s.expr).or_else(|| k.as_keyword_star().map(|s| s.expr)).flatten().unwrap()
                };
                let fv_name = self.get_temporary_var(if is_star { "i" } else { "it" });
                let for_var = self.id_n(&fv_name);
                self.ast_mut()[inner].attrs.set(Attr::ExprStarSequenceItem);
                let vc = self.cache.ast.clone_expr(var, false);
                let d = self.dot_n(vc, f);
                let args = if is_star {
                    vec![self.cache.ast.clone_expr(for_var, false)]
                } else {
                    let a = self.cache.ast.clone_expr(for_var, false);
                    let z = self.n_expr(IntExpr::new(0));
                    let i0 = self.n_expr(IndexExpr { expr: Some(a), index: Some(z) });
                    let b = self.cache.ast.clone_expr(for_var, false);
                    let o = self.n_expr(IntExpr::new(1));
                    let i1 = self.n_expr(IndexExpr { expr: Some(b), index: Some(o) });
                    vec![i0, i1]
                };
                let c = self.call_n(d, args);
                let body = self.n_stmt(ExprStmt { expr: Some(c) });
                let fv = self.cache.ast.clone_expr(for_var, false);
                stmts.push(self.n_stmt(ForStmt { var: Some(fv), iter: Some(inner), suite: Some(body), ..Default::default() }));
            } else {
                self.ast_mut()[it].attrs.set(Attr::ExprSequenceItem);
                let vc = self.cache.ast.clone_expr(var, false);
                let d = self.dot_n(vc, f);
                if is_dict {
                    let mut head = it;
                    let lead;
                    if self.has_side_effect(head) {
                        let tv = self.get_temporary_var("star");
                        let l = self.id_n(&tv);
                        lead = self.n_expr(AssignExpr { var: Some(l), expr: Some(head) });
                        head = self.id_n(&tv);
                    } else {
                        lead = self.cache.ast.clone_expr(head, false);
                    }
                    self.ast_mut()[lead].attrs.set(Attr::ExprSequenceItem);
                    self.ast_mut()[head].attrs.set(Attr::ExprSequenceItem);
                    let z = self.n_expr(IntExpr::new(0));
                    let i0 = self.n_expr(IndexExpr { expr: Some(lead), index: Some(z) });
                    let o = self.n_expr(IntExpr::new(1));
                    let i1 = self.n_expr(IndexExpr { expr: Some(head), index: Some(o) });
                    let c = self.call_n(d, vec![i0, i1]);
                    stmts.push(self.n_stmt(ExprStmt { expr: Some(c) }));
                } else {
                    let c = self.call_n(d, vec![it]);
                    stmts.push(self.n_stmt(ExprStmt { expr: Some(c) }));
                }
            }
        }
        let se = self.n_expr(StmtExpr { items: stmts, expr: Some(var) });
        let result = self.transform(Some(se))?;
        // The constructor call instantiates fresh element generics. Register the
        // container type so that later inserts into it can be traced back to the literal.
        if let (false, Some(r)) = (empty_slot_keys.is_empty(), result) {
            let rf = types.follow(self.ety(r));
            if let Some(rt) = types.get_class(rf) {
                let gens = types.generics(rt);
                let mut entry_keys = vec![];
                for (di, key) in &empty_slot_keys {
                    if let Some(gt) = gens.get(*di).and_then(|g| g.ty) {
                        if types.get_unbound(gt).is_some() {
                            // A widened slot: pin the constructed container to the union.
                            if types.get_union(element_types[*di]).is_some() {
                                self.unify(gt, element_types[*di])?;
                            }
                            entry_keys.push((*di, key.clone()));
                        }
                    }
                }
                let entry = self.cache.empty_literal_slots.entry(rt).or_insert_with(|| (vec![], rt));
                entry.1 = rt;
                entry.0.extend(entry_keys);
            }
        }
        Ok(result)
    }

    /// `emptyLiteralOwner()`: realization that owns empty-literal union inference (empty
    /// outside functions).
    pub fn empty_literal_owner(&self) -> String {
        let types = &self.cache.types;
        let t = self.ctx.borrow().base_ref().ty;
        match t.and_then(|t| types.get_func(t)) {
            Some(f) => types.realized_name(f),
            None => String::new(),
        }
    }

    /// `widenEmptyLiteralSlot(receiver, slot, actual)`: called before an insert into a
    /// collection unifies the inserted value with the element slot. If the slot came from
    /// an unannotated empty literal and the value has no common type with it, record a
    /// Union for the slot and request a restart of the owning realization.
    pub fn widen_empty_literal_slot(&mut self, receiver: ExprId, slot_index: usize, actual: TypeId) -> Result<()> {
        let types = self.types();
        let st = stdlib();
        // Find the container literal behind the receiver: either the receiver's own type
        // or (for variables, whose types may be copies) the type bound to its name.
        let mut candidates = vec![types.follow(self.ety(receiver))];
        if let Some(id) = self.ast()[receiver].kind.as_id().map(|i| i.value.clone()) {
            if let Some(t) = self.find(&id, 0).and_then(|v| v.borrow().ty) {
                candidates.push(types.follow(t));
            }
        }
        let mut keys = vec![];
        let mut expected = None;
        for c in candidates {
            if let Some((slots, tt)) = self.cache.empty_literal_slots.get(&c) {
                for (di, key) in slots {
                    if *di == slot_index {
                        keys.push(key.clone());
                        expected = types.generic_type(*tt, *di);
                    }
                }
                break;
            }
        }
        if keys.is_empty() {
            return Ok(());
        }
        let slot = expected.and_then(|e| types.get_class(types.follow(e)));
        let value = types.get_class(types.follow(actual));
        let (Some(slot), Some(value)) = (slot, value) else { return Ok(()) };
        if !types.can_realize(slot) || !types.can_realize(value) {
            return Ok(());
        }
        let mut undo = Unification::new();
        let score = types.unify(slot, value, Some(&mut undo), self)?;
        undo.undo(&types);
        if score >= 0 {
            return Ok(());
        }
        // Existing implicit conversions (int->float, Optional, subclass->base) win.
        let (ok, _, wf) = self.can_wrap_expr(value, slot, None, true, false)?;
        if ok && wf.is_some() {
            return Ok(());
        }
        // Common-type rules of collection literals: int -> float, None -> Optional.
        let mut common = None;
        if types.is(slot, "Int") && types.is(value, "float") {
            common = Some(value);
        } else if types.is(slot, &st.none_type) != types.is(value, &st.none_type) {
            let member = if types.is(slot, &st.none_type) { value } else { slot };
            common = Some(if types.is(member, &st.optional) {
                member
            } else {
                let o = self.get_stdlib_type(&st.optional)?;
                self.instantiate_type_with(o, &[member])?
            });
        }
        for key in keys {
            let owner = key.split('\x01').next().unwrap_or_default().to_string();
            if let Some(c) = common {
                self.cache.empty_literal_unions.insert(key, vec![c]);
                self.cache.empty_literal_restarts.insert(owner);
                continue;
            }
            let mut members = self.cache.empty_literal_unions.get(&key).cloned().unwrap_or_default();
            let mut pending = vec![value, slot];
            while let Some(t) = pending.pop() {
                if let Some(u) = types.get_union(t) {
                    for m in types.union_realization_types(u) {
                        if let Some(c) = types.get_class(m) {
                            pending.push(c);
                        }
                    }
                    continue;
                }
                let rn = types.realized_name(t);
                if !members.iter().any(|m| types.realized_name(*m) == rn) {
                    members.push(t);
                }
            }
            self.cache.empty_literal_unions.insert(key, members);
            self.cache.empty_literal_restarts.insert(owner);
        }
        Ok(())
    }

    fn gen_root_for(&self, e: ExprId) -> StmtId {
        self.ast().generator_final_suite(e).expect("generator without loops")
    }
    fn gen_root_iter(&self, e: ExprId) -> ExprId {
        let r = self.gen_root_for(e);
        self.ast()[r].kind.as_for().expect("generator root is not a for").iter.unwrap()
    }

    /// `visit(GeneratorExpr*)`: comprehensions and generators.
    ///   `tuple(expr for i in tuple_generator)` -> `Tuple.N.__new__(expr...)`
    pub fn visit_generator(&mut self, e: ExprId) -> Result<()> {
        let types = self.types();
        let st = stdlib();
        let kind = self.ast()[e].kind.as_generator().unwrap().kind;
        // List comprehension optimization:
        // Use `iter.__len__()` when creating list if there is a single for loop
        // without any if conditions in the comprehension
        let mut can_optimize = kind == GeneratorKind::ListGenerator && self.ast().generator_loop_count(e) == 1;
        if can_optimize {
            let it = self.gen_root_iter(e);
            let c = self.cache.ast.clone_expr(it, false);
            let iter = self.tr(c)?;
            if let Some(ce) = self.ast()[iter].kind.as_call() {
                if let Some(id) = ce.expr.and_then(|x| self.ast()[x].kind.as_id()) {
                    // Turn off this optimization for static items
                    can_optimize &= !id.value.starts_with("std.internal.static");
                }
            }
        }

        let var_name = self.get_temporary_var("gen");
        let var = self.id_n(&var_name);
        let cl = |v: &mut Self, x: ExprId| v.cache.ast.clone_expr(x, false);
        match kind {
            GeneratorKind::ListGenerator => {
                // List comprehensions
                let fe = self.ast().generator_final_expr(e).expect("no final expr");
                let v1 = cl(self, var);
                let d = self.dot_n(v1, "append");
                let c = self.call_n(d, vec![fe]);
                self.ast_mut().generator_set_final_expr(e, c);
                let suite = self.gen_root_for(e);
                let v2 = cl(self, var);
                let l = self.id_n(&st.list);
                let lc = self.call_n(l, vec![]);
                let a = self.assign_n(v2, lc);
                let no_opt = self.n_stmt(SuiteStmt { items: vec![a, suite] });

                let r = if can_optimize {
                    let opt_var = self.get_temporary_var("i");
                    let orig_iter = self.gen_root_iter(e);
                    let opt = self.cache.ast.clone_stmt(no_opt, false);
                    let for_s = self.ast()[opt].kind.as_suite().unwrap().items[1];
                    let ov = self.id_n(&opt_var);
                    self.ast_mut()[for_s].kind.as_for_mut().unwrap().iter = Some(ov);
                    let ov2 = self.id_n(&opt_var);
                    let oi = cl(self, orig_iter);
                    let a1 = self.assign_n(ov2, oi);
                    let v3 = cl(self, var);
                    let ov3 = self.id_n(&opt_var);
                    let ld = self.dot_n(ov3, "__len__");
                    let lenc = self.call_n(ld, vec![]);
                    let l2 = self.id_n(&st.list);
                    let lc2 = self.call_n(l2, vec![lenc]);
                    let a2 = self.assign_n(v3, lc2);
                    let opt = self.n_stmt(SuiteStmt { items: vec![a1, a2, for_s] });
                    let h = self.id_n("hasattr");
                    let oi2 = cl(self, orig_iter);
                    let s = self.n_expr(StringExpr::new("__len__", ""));
                    let cond = self.call_n(h, vec![oi2, s]);
                    let v4 = cl(self, var);
                    let t = self.n_expr(StmtExpr { items: vec![opt], expr: Some(v4) });
                    let f = self.n_expr(StmtExpr { items: vec![no_opt], expr: Some(var) });
                    self.n_expr(IfExpr { cond: Some(cond), ifexpr: Some(t), elsexpr: Some(f) })
                } else {
                    self.n_expr(StmtExpr { items: vec![no_opt], expr: Some(var) })
                };
                self.result_expr = self.transform(Some(r))?;
            }
            GeneratorKind::SetGenerator | GeneratorKind::DictGenerator => {
                // Set / dictionary comprehensions
                let is_set = kind == GeneratorKind::SetGenerator;
                let v1 = cl(self, var);
                let ctor = self.id_n(if is_set { &st.set } else { &st.dict });
                let cc = self.call_n(ctor, vec![]);
                let head = self.assign_n(v1, cc);
                let fe = self.ast().generator_final_expr(e).expect("no final expr");
                let v2 = cl(self, var);
                let d = self.dot_n(v2, if is_set { "add" } else { "__setitem__" });
                let arg = if is_set { fe } else { self.n_expr(StarExpr { expr: Some(fe) }) };
                let c = self.call_n(d, vec![arg]);
                self.ast_mut().generator_set_final_expr(e, c);
                let suite = self.gen_root_for(e);
                let s = self.n_stmt(SuiteStmt { items: vec![head, suite] });
                let se = self.n_expr(StmtExpr { items: vec![s], expr: Some(var) });
                self.result_expr = self.transform(Some(se))?;
            }
            GeneratorKind::TupleGenerator => {
                typhon_common::seqassert!(self.ast().generator_loop_count(e) == 1, "invalid tuple generator");
                let root = self.gen_root_for(e);
                let it = self.gen_root_iter(e);
                let gexp = self.tr(it)?;
                self.ast_mut()[root].kind.as_for_mut().unwrap().iter = Some(gexp);
                if !types.can_realize(self.ety(gexp)) {
                    return Ok(()); // Wait until the iterator can be realized
                }

                let mut block_items = vec![];
                // `tuple = tuple_generator`
                let tuple_var = self.get_temporary_var("tuple");
                let tv = self.id_n(&tuple_var);
                block_items.push(self.assign_n(tv, gexp));

                let for_stmt = self.cache.ast.clone_stmt(root, false);
                let final_expr = self.ast().generator_final_expr(e);
                let loop_var = self.ast()[root].kind.as_for().unwrap().var.unwrap();
                let mut suite = self.ast()[for_stmt].kind.as_for().unwrap().suite;
                let mut wrap = |v: &mut TypecheckVisitor<'a>, w: StmtId| -> Result<AstNode> {
                    let ws = v.cache.ast.clone_stmt(w, false);
                    let fe = final_expr.map(|f| v.cache.ast.clone_expr(f, false));
                    Ok(AstNode::Expr(v.n_expr(StmtExpr { items: vec![ws], expr: fe })))
                };
                let (ok, delay, preamble, static_items) = self.transform_static_loop_call(loop_var, &mut suite, gexp, &mut wrap, true)?;
                self.ast_mut()[for_stmt].kind.as_for_mut().unwrap().suite = suite;
                if !ok {
                    let src = self.ast()[gexp].src;
                    typhon_common::bail_e!(ErrorKind::CALL_BAD_ITER, src, types.pretty_string(self.ety(gexp)));
                }
                if delay {
                    return Ok(());
                }
                let tuple_items: Vec<ExprId> = static_items
                    .into_iter()
                    .filter_map(|i| match i {
                        AstNode::Expr(x) => Some(x),
                        AstNode::Stmt(_) => None,
                    })
                    .collect();
                if let Some(p) = preamble {
                    block_items.push(p);
                }
                let block = self.n_stmt(SuiteStmt { items: block_items });
                let tu = self.n_expr(TupleExpr { items: tuple_items });
                let se = self.n_expr(StmtExpr { items: vec![block], expr: Some(tu) });
                self.result_expr = self.transform(Some(se))?;
            }
            GeneratorKind::Generator => {
                let root = self.gen_root_for(e);
                let l = self.tr_stmt(root)?; // assume: internal data will be changed
                self.ast_mut()[e].kind.as_generator_mut().unwrap().loops = Some(l);
                let Some(fe) = self.ast().generator_final_expr(e) else {
                    // Case such as (0 for _ in static.range(2))
                    let src = self.ast()[e].src;
                    typhon_common::bail_e!(
                        ErrorKind::CUSTOM,
                        src,
                        "generator cannot be compiled. If using static tuple generator, use tuple(...) instead."
                    );
                };
                let g = self.get_stdlib_type("Generator")?;
                let ft = self.ety(fe);
                let gt = self.instantiate_type_with(g, &[ft])?;
                let et = self.ety(e);
                self.unify(et, gt)?;
                if self.realize(et)?.is_some() {
                    self.set_done_expr(e);
                }
            }
        }
        Ok(())
    }
}

#[allow(dead_code)]
fn _unused(_: CallExpr, _: IdExpr) {}
