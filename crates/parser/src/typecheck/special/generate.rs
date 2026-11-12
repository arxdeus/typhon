//! `generateSpecialAst` and the simple `generate*AST` helpers.

use typhon_common::mangle::{self, stdlib};
use typhon_common::{ErrorKind, Result};

use super::super::TypecheckVisitor;
use super::mm;
use crate::ast::{Attr, CallArg, StringExpr};
use crate::ids::{StmtId, TypeId};

impl<'a> TypecheckVisitor<'a> {
    /// `generateFunctionCallInternalAST(type)`: `Function.__call_internal__`.
    pub fn generate_function_call_internal_ast(&mut self, f: TypeId) -> Result<Option<StmtId>> {
        // Special case: Function.__call_internal__
        // TODO: move to IR one day
        let types = self.types();
        let mut items: Vec<Option<StmtId>> = vec![None];
        let mut ll: Vec<String> = vec![];
        let mut lla: Vec<String> = vec![];
        let a1 = self.extract_func_arg_type(f, 1);
        typhon_common::seqassert!(types.is(a1, &stdlib().tuple), "bad function base: {}", types.debug_string(a1, 2));
        let as_ = types.generics_len(types.get_class(a1).unwrap());
        let (_, ag) = self.sp_fn_params(f)[1].name_with_stars();
        for i in 0..as_ {
            ll.push(format!("%{i} = extractvalue {{}} %args, {i}"));
            let e = self.sp_id(&ag);
            items.push(Some(self.sp_expr_stmt(e)));
        }
        let tr = self.sp_id("TR");
        items.push(Some(self.sp_expr_stmt(tr)));
        for i in 0..as_ {
            let e = self.sp_id(&ag);
            let ix = self.sp_int(i as i64);
            let x = self.sp_index(e, ix);
            items.push(Some(self.sp_expr_stmt(x)));
            lla.push(format!("{{}} %{i}"));
        }
        let tr = self.sp_id("TR");
        items.push(Some(self.sp_expr_stmt(tr)));
        ll.push(format!("%{} = call {{}} %self({})", as_, lla.join(",")));
        ll.push(format!("ret {{}} %{as_}"));
        let s = self.n_expr(StringExpr::new(ll.join("\n"), ""));
        items[0] = Some(self.sp_expr_stmt(s));
        Ok(Some(self.sp_suite(items.into_iter().flatten().collect())))
    }

    /// `generateUnionNewAST(type)`.
    pub fn generate_union_new_ast(&mut self, f: TypeId) -> Result<Option<StmtId>> {
        let types = self.types();
        let parent = types.func_parent(f);
        let union_type = parent.and_then(|p| types.get_union(p));
        typhon_common::seqassert!(union_type.is_some(), "expected union, got {:?}", parent.map(|p| types.debug_string(p, 2)));
        let union_type = union_type.unwrap();
        let first = self.sp_fn_params(f)[0].name.clone();
        let u = self.sp_id(&stdlib().union);
        let d = self.sp_dot(u, "_new");
        let a = self.sp_id(&first);
        let rn = self.sp_rn(union_type);
        let b = self.sp_id(&rn);
        let c = self.sp_call(d, vec![a, b]);
        let suite = self.sp_return(Some(c));
        Ok(self.cache.ast.wrap_suite(Some(suite)))
    }

    /// `generateUnionTagAST(type)`: `return Union._get_data(union, T0)`.
    pub fn generate_union_tag_ast(&mut self, f: TypeId) -> Result<Option<StmtId>> {
        let types = self.types();
        let tag = self.get_int_literal(self.extract_func_generic(f, 0), 0);
        let union_type = types.get_union(self.extract_func_arg_type(f, 0)).expect("not a union");
        let union_types = types.union_realization_types(union_type);
        if tag < 0 || tag as usize >= union_types.len() {
            typhon_common::bail_e!(ErrorKind::CUSTOM, self.src, "bad union tag");
        }
        let self_var = self.sp_fn_params(f)[0].name.clone();
        let g = self.sp_id(&mm("Union", "_get_data"));
        let s = self.sp_id(&self_var);
        let rn = self.sp_rn(union_types[tag as usize]);
        let t = self.sp_id(&rn);
        let c = self.sp_call(g, vec![s, t]);
        let r = self.sp_return(Some(c));
        Ok(Some(self.sp_suite(vec![r])))
    }

    /// `generateNamedKeysAST(type)`.
    pub fn generate_named_keys_ast(&mut self, f: TypeId) -> Result<Option<StmtId>> {
        let n = self.get_int_literal(self.extract_func_generic(f, 0), 0);
        let len = self.cache.types.generated_tuple_names.borrow().len();
        if n < 0 || n as usize >= len {
            typhon_common::bail_e!(ErrorKind::CUSTOM, self.src, "bad namedkeys index");
        }
        let mut s = vec![];
        for k in self.sp_tuple_names(n) {
            s.push(self.sp_str(&k));
        }
        let t = self.sp_tuple(s);
        let r = self.sp_return(Some(t));
        Ok(Some(self.sp_suite(vec![r])))
    }

    /// `generateTupleMulAST(type)`.
    pub fn generate_tuple_mul_ast(&mut self, f: TypeId) -> Result<Option<StmtId>> {
        let types = self.types();
        let n = self.get_int_literal(self.extract_func_generic(f, 0), 0).max(0);
        let t = types.get_class(self.extract_func_arg_type(f, 0));
        let Some(t) = t.filter(|t| types.is(*t, &stdlib().tuple)) else { return Ok(None) };
        let first = self.sp_fn_params(f)[0].name.clone();
        let mut exprs = vec![];
        for _ in 0..n {
            for j in 0..types.generics_len(t) {
                let a = self.sp_id(&first);
                let b = self.sp_int(j as i64);
                exprs.push(self.sp_index(a, b));
            }
        }
        let tup = self.sp_tuple(exprs);
        let r = self.sp_return(Some(tup));
        Ok(Some(self.sp_suite(vec![r])))
    }

    /// `generateSpecialAst(type)`: generate ASTs for dynamically generated functions.
    pub fn generate_special_ast(&mut self, f: TypeId) -> Result<Option<StmtId>> {
        // Clone the generic AST that is to be realized
        let aid = self.cache.types.func_ast_id(f);
        let auto_generated = self.cache.ast[aid].attrs.has(Attr::AutoGenerated);
        let name = self.sp_fn_ast_name(f);
        if auto_generated && name.ends_with(".__iter__:0") && {
            let a0 = self.extract_func_arg_type(f, 0);
            self.is_heterogenous(a0)?
        } {
            // Special case: do not realize auto-generated heterogenous __iter__
            typhon_common::bail_e!(ErrorKind::EXPECTED_TYPE, self.src, "iterable");
        } else if auto_generated && name.ends_with(".__getitem__:0") && {
            let a0 = self.extract_func_arg_type(f, 0);
            self.is_heterogenous(a0)?
        } {
            // Special case: do not realize auto-generated heterogenous __getitem__
            typhon_common::bail_e!(ErrorKind::EXPECTED_TYPE, self.src, "iterable");
        } else if name.starts_with("Function.__call_internal__")
            || name.starts_with(&mangle::mangled_func("std.internal.dlopen", "_call_c", 0, 0, false))
        {
            return self.generate_function_call_internal_ast(f);
        } else if name.starts_with("Union.__new__") {
            return self.generate_union_new_ast(f);
        } else if name.starts_with(&mm("Union", "_tag")) {
            return self.generate_union_tag_ast(f);
        } else if name.starts_with(&mm("Union", "_dispatch")) {
            return self.generate_union_dispatch_ast(f);
        } else if name.starts_with(&mm("Union", "__iter__")) {
            return self.generate_union_iterator_ast(f);
        } else if name.starts_with(&mm("Union", "_binary")) {
            return self.generate_union_binary_ast(f);
        } else if name.starts_with(&mm("NamedTuple", "_namedkeys")) {
            return self.generate_named_keys_ast(f);
        } else if name.starts_with(&mm("__magic__", "mul")) {
            return self.generate_tuple_mul_ast(f);
        } else if name.starts_with(&mm("TypeInfo", "_init_params")) {
            return self.generate_type_info_init_ast(f);
        } else if name.starts_with(&mm("TypeInfo", "_value_method_id")) {
            let s = self.get_str_literal(self.extract_func_generic(f, 0), 0);
            let id = self.value_method_id(&s);
            let i = self.sp_int(id as i64);
            let r = self.sp_return(Some(i));
            return Ok(Some(self.sp_suite(vec![r])));
        } else if name.starts_with(&mm("Super", "_dispatch")) {
            return self.generate_super_dispatch_ast(f);
        }
        Ok(None)
    }

    /// `generateTypeInfoInitAST(type)`.
    pub fn generate_type_info_init_ast(&mut self, f: TypeId) -> Result<Option<StmtId>> {
        let types = self.types();
        let st = stdlib();
        let Some(t) = types.get_class(self.extract_func_generic(f, 0)) else { return Ok(None) };
        if !types.can_realize(t) {
            return Ok(None);
        }
        let suite = self.sp_suite(vec![]);
        // Add extra initialization here!
        let slf = self.sp_id("self");
        let lhs = self.sp_dot(slf, "_base_name");
        let cname = types.class_name(t).unwrap().to_string();
        let rhs = self.sp_str(&cname);
        let a = self.sp_assign(lhs, Some(rhs), None);
        self.cache.ast.suite_add_stmt(suite, Some(a));
        if !types.is(t, &st.unrealized_type) {
            for g in types.generics(t) {
                let mut tp = g.ty();
                if types.get_static(tp).is_some() {
                    tp = types.non_static_type(tp).unwrap_or(tp);
                }
                if types.get_func(tp).is_some() {
                    let r = self.realize(tp)?.expect("cannot realize");
                    tp = types.copy_class(types.get_class(r).unwrap());
                }
                let s = self.sp_type_info_cache(tp);
                self.cache.ast.suite_add_stmt(suite, Some(s));
                let slf = self.sp_id("self");
                let p = self.sp_dot(slf, "_params");
                let ap = self.sp_dot(p, "append");
                let id = self.get_class_realization(tp).expect("not realized").borrow().id;
                let i = self.sp_int(id as i64);
                let c = self.sp_call(ap, vec![i]);
                let s = self.sp_expr_stmt(c);
                self.cache.ast.suite_add_stmt(suite, Some(s));
            }
        }
        let fields = self.get_class_realization(t).expect("not realized").borrow().fields.clone();
        for (fi, (fname, ft)) in fields.into_iter().enumerate() {
            let mut tp = ft;
            if types.get_static(tp).is_some() {
                tp = types.non_static_type(tp).unwrap_or(tp);
            }
            let s = self.sp_type_info_cache(tp);
            self.cache.ast.suite_add_stmt(suite, Some(s));
            let slf = self.sp_id("self");
            let p = self.sp_dot(slf, "_fields");
            let ap = self.sp_dot(p, "append");
            let fs = self.sp_str(&fname);
            let id = self.get_class_realization(tp).expect("not realized").borrow().id;
            let fid = self.sp_int(id as i64);
            let off = self.sp_id(&mm("type", "_get_class_offset"));
            let trn = self.sp_rn(t);
            let tn = self.sp_id(&trn);
            let fin = self.sp_int(fi as i64);
            let oc = self.sp_call(off, vec![tn, fin]);
            let tup = self.sp_tuple(vec![fs, fid, oc]);
            let c = self.sp_call(ap, vec![tup]);
            let s = self.sp_expr_stmt(c);
            self.cache.ast.suite_add_stmt(suite, Some(s));
        }
        Ok(Some(suite))
    }

    /// `TypeInfo.cache(vtable=vt, T=<tp>)` statement.
    fn sp_type_info_cache(&mut self, tp: TypeId) -> StmtId {
        let f = self.sp_id(&mm("TypeInfo", "cache"));
        let vt = self.sp_id("vt");
        let rn = self.sp_rn(tp);
        let te = self.sp_id(&rn);
        let args = vec![CallArg::new("vtable", Some(vt)), CallArg::new("T", Some(te))];
        let c = self.sp_call_args(f, args);
        self.sp_expr_stmt(c)
    }

    /// Remove `suite[0]` and append it at the end (`suite->addStmt(suite->front()); (*suite)[0] = nullptr`).
    pub(super) fn sp_move_front_to_back(&mut self, suite: StmtId) {
        if let Some(s) = self.cache.ast[suite].kind.as_suite_mut()
            && !s.items.is_empty()
        {
            let f = s.items.remove(0);
            s.items.push(f);
        }
    }
}
