//! `populate*Loop` helpers  (bodies of static `for` loops).

use typhon_common::mangle::stdlib;
use typhon_common::{ErrorKind, Result};

use super::super::TypecheckVisitor;
use crate::ast::{Attr, CallArg, EllipsisExpr, EllipsisType};
use crate::cache::MAX_STATIC_ITER;
use crate::ids::{ExprId, StmtId, TypeId};

impl<'a> TypecheckVisitor<'a> {
    /// `cast<IdExpr>(cast<CallExpr>(iter)->getExpr())` type.
    fn sp_loop_fn_type(&self, iter: ExprId) -> TypeId {
        let c = self.cache.ast[iter].kind.as_call().expect("static loop over a non-call");
        let f = c.expr.expect("call without callee");
        typhon_common::seqassert!(self.cache.ast[f].kind.as_id().is_some(), "static loop callee is not an identifier");
        self.ety(f)
    }
    /// The callee expression (`fn`) used as the error location.
    fn sp_loop_fn_src(&self, iter: ExprId) -> typhon_common::SrcInfo {
        let c = self.cache.ast[iter].kind.as_call().expect("static loop over a non-call");
        self.cache.ast[c.expr.unwrap()].src
    }
    /// `N<AssignStmt>(N<IdExpr>(var), rhs, type)`.
    fn sp_assign_var(&mut self, var: &str, rhs: ExprId, ty: Option<ExprId>) -> StmtId {
        let l = self.sp_id(var);
        self.sp_assign(l, Some(rhs), ty)
    }

    /// `populateStaticTupleLoop(iter, vars)`.
    pub fn populate_static_tuple_loop(&mut self, iter: ExprId, vars: &[String]) -> Result<Vec<StmtId>> {
        let types = self.types();
        let mut block = vec![];
        let inner = self.cache.ast[iter].kind.as_call().and_then(|c| c.items.first()).and_then(|a| a.value).expect("bad static tuple loop");
        let call = self.cache.ast[inner].kind.as_call().cloned().expect("bad static tuple loop");
        if vars.len() != 1 {
            typhon_common::bail_e!(ErrorKind::CUSTOM, self.src, "expected one item");
        }
        for a in call.items {
            let c = self.cache.ast.clone_expr(a.value.unwrap(), true);
            let rhs = self.tr(c)?;
            let ty = match types.get_static(self.ety(rhs)) {
                Some(st) => {
                    let name = types.class_name(st).unwrap().to_string();
                    Some(self.sp_literal(&name))
                }
                None => None,
            };
            // the reference clones a reused AssignStmt; a fresh node per iteration is equivalent.
            let s = self.sp_assign_var(&vars[0], rhs, ty);
            block.push(self.cache.ast.clone_stmt(s, false));
        }
        Ok(block)
    }

    /// `populateSimpleStaticRangeLoop(iter, vars)`.
    pub fn populate_simple_static_range_loop(&mut self, iter: ExprId, vars: &[String]) -> Result<Vec<StmtId>> {
        if vars.len() != 1 {
            typhon_common::bail_e!(ErrorKind::CUSTOM, self.src, "expected one item");
        }
        let ft = self.sp_loop_fn_type(iter);
        let mut block = vec![];
        let ed = self.get_int_literal(self.extract_func_generic(ft, 0), 0);
        if ed > MAX_STATIC_ITER as i64 {
            typhon_common::bail_e!(ErrorKind::STATIC_RANGE_BOUNDS, self.sp_loop_fn_src(iter), MAX_STATIC_ITER, ed);
        }
        for i in 0..ed {
            let rhs = self.sp_int(i);
            let ty = self.sp_literal("int");
            block.push(self.sp_assign_var(&vars[0], rhs, Some(ty)));
        }
        Ok(block)
    }

    /// `populateStaticRangeLoop(iter, vars)`.
    pub fn populate_static_range_loop(&mut self, iter: ExprId, vars: &[String]) -> Result<Vec<StmtId>> {
        if vars.len() != 1 {
            typhon_common::bail_e!(ErrorKind::CUSTOM, self.src, "expected one item");
        }
        let ft = self.sp_loop_fn_type(iter);
        let mut block = vec![];
        let st = self.get_int_literal(self.extract_func_generic(ft, 0), 0);
        let ed = self.get_int_literal(self.extract_func_generic(ft, 1), 0);
        let step = self.get_int_literal(self.extract_func_generic(ft, 2), 0);
        // Note: the reference divides by `abs(step)` unguarded (UB for 0); treat step 0 as an error.
        if step == 0 || (st - ed).abs() / step.abs() > MAX_STATIC_ITER as i64 {
            let n = if step == 0 { i64::MAX } else { (st - ed).abs() / step.abs() };
            typhon_common::bail_e!(ErrorKind::STATIC_RANGE_BOUNDS, self.sp_loop_fn_src(iter), MAX_STATIC_ITER, n);
        }
        let mut i = st;
        while if step > 0 { i < ed } else { i > ed } {
            let rhs = self.sp_int(i);
            let ty = self.sp_literal("int");
            block.push(self.sp_assign_var(&vars[0], rhs, Some(ty)));
            i += step;
        }
        Ok(block)
    }

    /// `populateStaticFnOverloadsLoop(iter, vars)`.
    pub fn populate_static_fn_overloads_loop(&mut self, iter: ExprId, vars: &[String]) -> Result<Vec<StmtId>> {
        let types = self.types();
        if vars.len() != 1 {
            typhon_common::bail_e!(ErrorKind::CUSTOM, self.src, "expected one item");
        }
        let ft = self.sp_loop_fn_type(iter);
        let mut block = vec![];
        let typ = types.get_class(self.extract_func_generic(ft, 0)).expect("not a class");
        typhon_common::seqassert!(types.get_str_static(self.extract_func_generic(ft, 1)).is_some(), "bad static string");
        let name = self.get_str_literal(self.extract_func_generic(ft, 1), 0);

        let mut overloads = vec![];
        if types.is(typ, &stdlib().none_type) {
            let tctx = self.cache.type_ctx.clone();
            let func = super::super::TypeContext::find_in(self.cache, &tctx, &name, 0, None);
            if let Some(func) = func {
                let ft = func.borrow().ty();
                let f = types.get_func(ft).expect("not a function");
                let root = self.get_root_name(f);
                overloads = self.get_overloads(&root);
            }
        } else if let Some(n) = self.get_class_type(typ).methods.get(&name).cloned() {
            overloads = self.get_overloads(&n);
        }
        if !overloads.is_empty() {
            for method in overloads.iter().rev() {
                let (cfn_ty, cfn_ast) = match self.get_function_name(method) {
                    Some(f) => (f.ty, f.ast),
                    None => (None, None),
                };
                if Self::is_dispatch_name(method) || cfn_ty.is_none() {
                    continue;
                }
                if self.is_heterogenous(typ)? {
                    let ast = cfn_ast.expect("function without AST");
                    let aname = self.cache.ast[ast].kind.as_function().map(|f| f.name.clone()).unwrap_or_default();
                    if self.cache.ast[ast].attrs.has(Attr::AutoGenerated)
                        && (aname.ends_with(".__iter__:0") || aname.ends_with(".__getitem__:0"))
                    {
                        // ignore __getitem__ and other heterogenuous methods
                        continue;
                    }
                }
                let mut rhs = self.sp_id(method);
                if !types.is(typ, &stdlib().none_type) {
                    for generic in types.func_generics(cfn_ty.unwrap()) {
                        if self.get_unmangled_name(&generic.name) == "Self" {
                            let st = self.get_self_type(typ)?.expect("no Self type");
                            let rn = types.realized_name(st);
                            let s = self.sp_id(&rn);
                            let el = self.n_expr(EllipsisExpr { mode: EllipsisType::Partial });
                            rhs = self.sp_call_args(rhs, vec![CallArg::new("Self", Some(s)), CallArg::new("", Some(el))]);
                            break;
                        }
                    }
                }
                block.push(self.sp_assign_var(&vars[0], rhs, None));
            }
        }
        Ok(block)
    }

    /// `populateStaticEnumerateLoop(iter, vars)`.
    pub fn populate_static_enumerate_loop(&mut self, iter: ExprId, vars: &[String]) -> Result<Vec<StmtId>> {
        let types = self.types();
        if vars.len() != 2 {
            typhon_common::bail_e!(ErrorKind::CUSTOM, self.src, "expected two items");
        }
        let ft = self.sp_loop_fn_type(iter);
        let mut block = vec![];
        let typ = types.get_class(self.extract_func_arg_type(ft, 0));
        match typ {
            Some(typ) if types.is_record(typ) => {
                let arg0 = self.sp_arg(iter, 0);
                for i in 0..self.get_class_fields(typ).len() {
                    let ri = self.sp_int(i as i64);
                    let ty = self.sp_literal("int");
                    let a = self.sp_assign_var(&vars[0], ri, Some(ty));
                    let c = self.cache.ast.clone_expr(arg0, false);
                    let ix = self.sp_int(i as i64);
                    let x = self.sp_index(c, ix);
                    let b = self.sp_assign_var(&vars[1], x, None);
                    block.push(self.sp_suite(vec![a, b]));
                }
            }
            _ => {
                typhon_common::bail_e!(ErrorKind::CUSTOM, self.src, "static.enumerate needs a tuple");
            }
        }
        Ok(block)
    }

    /// `populateStaticVarsLoop(iter, vars)`.
    pub fn populate_static_vars_loop(&mut self, iter: ExprId, vars: &[String]) -> Result<Vec<StmtId>> {
        let types = self.types();
        let ft = self.sp_loop_fn_type(iter);
        let with_idx = self.get_bool_literal(self.extract_func_generic(ft, 0), 0);
        if !with_idx && vars.len() != 2 {
            typhon_common::bail_e!(ErrorKind::CUSTOM, self.src, "expected two items");
        } else if with_idx && vars.len() != 3 {
            typhon_common::bail_e!(ErrorKind::CUSTOM, self.src, "expected three items");
        }
        let wi = with_idx as usize;
        let mut block = vec![];
        let typ = types.get_class(self.extract_func_arg_type(ft, 0)).expect("not a class");
        if types.is(typ, &stdlib().type_wrap) {
            // type passed!
            let inner = self.extract_class_generic(typ, 0);
            let mut cv: Vec<(String, String)> =
                self.get_class_type(inner).class_vars.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
            cv.sort();
            for (idx, (field_name, field_var)) in cv.into_iter().enumerate() {
                let mut stmts = vec![];
                if with_idx {
                    let i = self.sp_int(idx as i64);
                    let ty = self.sp_literal("int");
                    stmts.push(self.sp_assign_var(&vars[0], i, Some(ty)));
                }
                let s = self.sp_str(&field_name);
                let ty = self.sp_literal("str");
                stmts.push(self.sp_assign_var(&vars[wi], s, Some(ty)));
                let v = self.sp_id(&field_var);
                stmts.push(self.sp_assign_var(&vars[wi + 1], v, None));
                block.push(self.sp_suite(stmts));
            }
        } else {
            let arg0 = self.sp_arg(iter, 0);
            for (idx, f) in self.get_class_fields(typ).into_iter().enumerate() {
                let mut stmts = vec![];
                if with_idx {
                    let i = self.sp_int(idx as i64);
                    let ty = self.sp_literal("int");
                    stmts.push(self.sp_assign_var(&vars[0], i, Some(ty)));
                }
                let s = self.sp_str(&f.name);
                let ty = self.sp_literal("str");
                stmts.push(self.sp_assign_var(&vars[wi], s, Some(ty)));
                let c = self.cache.ast.clone_expr(arg0, false);
                let d = self.sp_dot(c, &f.name);
                stmts.push(self.sp_assign_var(&vars[wi + 1], d, None));
                block.push(self.sp_suite(stmts));
            }
        }
        Ok(block)
    }

    /// `populateStaticVarTypesLoop(iter, vars)`.
    pub fn populate_static_var_types_loop(&mut self, iter: ExprId, vars: &[String]) -> Result<Vec<StmtId>> {
        let types = self.types();
        let ft = self.sp_loop_fn_type(iter);
        let g0 = self.extract_func_generic(ft, 0);
        let typ = match types.get_class(g0) {
            Some(c) => self.realize(c)?,
            None => None,
        };
        let with_idx = self.get_bool_literal(self.extract_func_generic(ft, 1), 0);
        if !with_idx && vars.len() != 1 {
            typhon_common::bail_e!(ErrorKind::CUSTOM, self.src, "expected one item");
        } else if with_idx && vars.len() != 2 {
            typhon_common::bail_e!(ErrorKind::CUSTOM, self.src, "expected two items");
        }
        typhon_common::seqassert!(typ.is_some(), "vars_types expects a realizable type, got '{}' instead", types.debug_string(g0, 2));
        let typ = typ.expect("vars_types expects a realizable type");
        let wi = with_idx as usize;
        let mut block = vec![];
        if let Some(utyp) = types.get_union(typ) {
            for (i, ut) in types.union_realization_types(utyp).into_iter().enumerate() {
                let mut stmts = vec![];
                if with_idx {
                    let ie = self.sp_int(i as i64);
                    let ty = self.sp_literal("int");
                    stmts.push(self.sp_assign_var(&vars[0], ie, Some(ty)));
                }
                // Note: the reference uses `vars[1]` regardless of `withIdx` (out of bounds without
                // an index); fall back to the only variable in that case.
                let rn = types.realized_name(ut);
                let te = self.sp_id(&rn);
                let vi = if vars.len() > 1 { 1 } else { 0 };
                stmts.push(self.sp_assign_var(&vars[vi], te, None));
                block.push(self.sp_suite(stmts));
            }
        } else {
            let tc = types.get_class(typ).unwrap();
            for (idx, f) in self.get_class_fields(tc).into_iter().enumerate() {
                let fty = f.ty.expect("field without type");
                let it = self.instantiate_type(fty, Some(tc))?;
                let ta = self.realize(it)?;
                typhon_common::seqassert!(ta.is_some(), "cannot realize '{}'", types.debug_string(fty, 2));
                let ta = ta.expect("cannot realize field");
                let mut stmts = vec![];
                if with_idx {
                    let ie = self.sp_int(idx as i64);
                    let ty = self.sp_literal("int");
                    stmts.push(self.sp_assign_var(&vars[0], ie, Some(ty)));
                }
                let rn = types.realized_name(ta);
                let te = self.sp_id(&rn);
                stmts.push(self.sp_assign_var(&vars[wi], te, None));
                block.push(self.sp_suite(stmts));
            }
        }
        Ok(block)
    }

    /// `populateStaticMethodsLoop(iter, vars)`.
    pub fn populate_static_methods_loop(&mut self, iter: ExprId, vars: &[String]) -> Result<Vec<StmtId>> {
        let types = self.types();
        let ft = self.sp_loop_fn_type(iter);
        let g0 = self.extract_func_generic(ft, 0);
        let typ = match types.get_class(g0) {
            Some(c) => self.realize(c)?,
            None => None,
        };
        typhon_common::seqassert!(typ.is_some(), "methods expects a realizable type, got '{}' instead", types.debug_string(g0, 2));
        let mut typ = typ.expect("methods expects a realizable type");
        if types.is(typ, &stdlib().type_wrap) {
            typ = types.get_class(self.extract_class_generic(typ, 0)).unwrap();
        }
        let mut block = vec![];
        let tc = types.get_class(typ).unwrap();
        let mut methods: Vec<String> = self.get_class_type(tc).methods.keys().cloned().collect();
        methods.sort();
        for m in methods {
            let s = self.sp_str(&m);
            let ty = self.sp_literal("str");
            let a = self.sp_assign_var(&vars[0], s, Some(ty));
            block.push(self.sp_suite(vec![a]));
        }
        Ok(block)
    }

    /// `populateStaticHeterogenousTupleLoop(iter, vars)`: the last element of the result is
    /// the preamble (`None` in the reference when `iter` is an identifier).
    ///
    /// Note: the reference pushes a possibly-null preamble as the last element; Rust cannot store a
    /// null `StmtId`, so an empty `SuiteStmt` stands in for the null preamble.
    pub fn populate_static_heterogenous_tuple_loop(&mut self, iter: ExprId, vars: &[String]) -> Result<Vec<StmtId>> {
        let types = self.types();
        let mut block = vec![];
        let (tuple_var, preamble) = match self.cache.ast[iter].kind.as_id() {
            Some(id) => (id.value.clone(), None),
            None => {
                let tv = self.get_temporary_var("tuple");
                let p = self.sp_assign_var(&tv, iter, None);
                (tv, Some(p))
            }
        };
        let n = self.expr_class(iter).map(|c| types.generics_len(c)).unwrap_or(0);
        for i in 0..n {
            let s = self.sp_suite(vec![]);
            if vars.len() > 1 {
                for (j, var) in vars.iter().enumerate() {
                    let t = self.sp_id(&tuple_var);
                    let ii = self.sp_int(i as i64);
                    let x = self.sp_index(t, ii);
                    let jj = self.sp_int(j as i64);
                    let y = self.sp_index(x, jj);
                    let a = self.sp_assign_var(var, y, None);
                    self.cache.ast.suite_add_stmt(s, Some(a));
                }
            } else {
                let t = self.sp_id(&tuple_var);
                let ii = self.sp_int(i as i64);
                let x = self.sp_index(t, ii);
                let a = self.sp_assign_var(&vars[0], x, None);
                self.cache.ast.suite_add_stmt(s, Some(a));
            }
            block.push(s);
        }
        let preamble = match preamble {
            Some(p) => p,
            None => self.sp_suite(vec![]),
        };
        block.push(preamble);
        Ok(block)
    }
}
