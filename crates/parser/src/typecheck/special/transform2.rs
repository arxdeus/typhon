//! Special-call handlers , part 2 (`transformIsInstance`, `transformGetAttr`,
//! `transformSetAttr`, `transformCompileError`, `transformTupleFn`, `transformTypeFn`,
//! `transformRealizedFn`, `transformStaticPrintFn`, `transformHasRttiFn`).

use typhon_common::mangle::{self, stdlib};
use typhon_common::{ErrorKind, Result};

use super::super::TypecheckVisitor;
use super::mm;
use crate::ast::{AssignMemberStmt, ExprTag, InstantiateExpr, StmtExpr};
use crate::ids::ExprId;
use crate::types::{LiteralKind, Unification};

impl<'a> TypecheckVisitor<'a> {
    /// `transformIsInstance(expr)`: static boolean `isinstance`.
    /// Special cases:
    ///   `isinstance(obj, ByVal)` is True if `type(obj)` is a tuple type
    ///   `isinstance(obj, ByRef)` is True if `type(obj)` is a reference type
    pub fn transform_is_instance(&mut self, e: ExprId) -> Result<Option<ExprId>> {
        let types = self.types();
        let st = stdlib();
        self.sp_set_static_kind(e, LiteralKind::Bool);

        let a0 = self.sp_arg(e, 0);
        let a0 = self.tr(a0)?;
        self.sp_set_arg(e, 0, a0);
        let typ = self.expr_class(a0);
        let Some(typ) = typ.filter(|t| types.can_realize(*t)) else { return Ok(None) };

        let a0 = self.tr(a0)?; // again to realize it
        self.sp_set_arg(e, 0, a0);

        let typ = self.extract_class_type(typ).expect("not a class");
        let typ_expr = self.sp_arg(e, 1);
        if self.cache.ast[typ_expr].kind.tag() == ExprTag::Call {
            // Handle `isinstance(obj, (type1, type2, ...))`
            if let Some(orig) = self.cache.ast[typ_expr].orig_expr
                && let Some(tup) = self.cache.ast[orig].kind.as_tuple().cloned()
            {
                let b = self.sp_bool(false);
                let mut result = self.tr(b)?;
                for i in tup.items {
                    let isi = self.sp_id("isinstance");
                    let x = self.sp_arg(e, 0);
                    let c = self.sp_call(isi, vec![x, i]);
                    let bin = self.sp_bin(result, "||", c);
                    result = self.tr(bin)?;
                }
                return Ok(Some(result));
            }
        }

        let tei = self.cache.ast[typ_expr].kind.as_id().map(|i| i.value.clone());
        let tei = tei.as_deref();
        let te_type = self.extract_type_expr(typ_expr);
        if Self::is_type_expr(self.cache, typ_expr) && types.is(te_type, &st.type_) {
            let a0 = self.sp_arg(e, 0);
            let v = Self::is_type_expr(self.cache, a0) || types.is(typ, &st.type_wrap);
            let b = self.sp_bool(v);
            return self.transform(Some(b));
        } else if tei == Some("type[Tuple]") {
            let b = self.sp_bool(types.is(typ, &st.tuple));
            return self.transform(Some(b));
        } else if tei == Some("type[ByVal]") {
            let b = self.sp_bool(types.is_record(typ));
            return self.transform(Some(b));
        } else if tei == Some("type[ByRef]") {
            let b = self.sp_bool(!types.is_record(typ));
            return self.transform(Some(b));
        } else if types.get_union(te_type).is_none() && types.get_union(typ).is_some() {
            let union_types = types.union_realization_types(types.get_union(typ).unwrap());
            let a0 = self.sp_arg(e, 0);
            if !Self::is_type_expr(self.cache, a0) && union_types.iter().any(|m| types.is(*m, &st.optional)) {
                let u = self.sp_id(&st.union);
                let d = self.sp_dot(u, "_isinstance");
                let c = self.sp_call(d, vec![a0, typ_expr]);
                return self.transform(Some(c));
            }
            let mut tag: i64 = -1;
            for (ui, ut) in union_types.iter().enumerate() {
                let te = self.extract_type_expr(typ_expr);
                let mut probe = Unification::new();
                let r = types.unify(te, *ut, Some(&mut probe), self)?;
                // Note: the reference `unify(.., nullptr)` is destructive on success; we keep the
                // successful unification (same effect) and only undo failed attempts.
                if r >= 0 {
                    tag = ui as i64;
                    break;
                }
                probe.undo(&types);
            }
            if tag == -1 {
                let b = self.sp_bool(false);
                return self.transform(Some(b));
            }
            let u = self.sp_id(&st.union);
            let d = self.sp_dot(u, "_get_tag");
            let c = self.sp_call(d, vec![a0]);
            let t = self.sp_int(tag);
            let bin = self.sp_bin(c, "==", t);
            return self.transform(Some(bin));
        } else if types.is(self.ety(typ_expr), "pyobj") {
            if types.is(typ, "pyobj") {
                let f = self.sp_id(&mangle::func("std.internal.python", "_isinstance"));
                let a0 = self.sp_arg(e, 0);
                let a1 = self.sp_arg(e, 1);
                let c = self.sp_call(f, vec![a0, a1]);
                return self.transform(Some(c));
            } else {
                let b = self.sp_bool(false);
                return self.transform(Some(b));
            }
        }

        let typ_expr = self.transform_type(Some(typ_expr), false)?.unwrap();
        self.sp_set_arg(e, 1, typ_expr);
        let target_type = self.extract_type_expr(typ_expr);

        // Check type match
        let tcls = types.get_class(target_type).expect("not a class");
        if types.class_name(tcls) == types.class_name(typ) {
            let mut us = Unification::new();
            let s = types.unify(typ, target_type, Some(&mut us), self)?;
            us.undo(&types);
            if s >= 0 {
                let b = self.sp_bool(true);
                return self.transform(Some(b));
            }
        }

        let inst_call = "_isinstance";
        let a0 = self.sp_arg(e, 0);
        if types.is(typ, &st.any) && !Self::is_type_expr(self.cache, a0) {
            let f = self.sp_id(&mm("Any", inst_call));
            let a1 = self.sp_arg(e, 1);
            let c = self.sp_call(f, vec![a0, a1]);
            return self.transform(Some(c));
        }

        if self.sp_has_rtti(tcls) && self.sp_has_rtti(typ) {
            // Check RTTI super types
            for tx in self.get_mro(typ)? {
                let mut us = Unification::new();
                let s = types.unify(tx, target_type, Some(&mut us), self)?;
                us.undo(&types);
                if s >= 0 {
                    let b = self.sp_bool(true);
                    return self.transform(Some(b));
                }
            }

            // Check whether the target can be a subtype of the value's static type. Besides
            // rejecting unrelated classes, this specializes target generics through its base
            // type (e.g. Future[int] against Task[R] binds R to int).
            for tx in self.get_mro(tcls)? {
                let mut us = Unification::new();
                if types.unify(tx, typ, Some(&mut us), self)? >= 0 {
                    let f = self.sp_id(&mm("RTTIType", inst_call));
                    let a1 = self.sp_arg(e, 1);
                    let c = self.sp_call(f, vec![a0, a1]);
                    return self.transform(Some(c));
                }
                us.undo(&types);
            }
        }

        let b = self.sp_bool(false);
        self.transform(Some(b))
    }

    /// `transformGetAttr(expr)`: `getattr` -> `DotExpr`.
    pub fn transform_get_attr(&mut self, e: ExprId) -> Result<Option<ExprId>> {
        let types = self.types();
        let ct = self.sp_callee_type(e);
        let attr = self.get_str_literal(self.extract_func_generic(ct, 0), 0);
        let attr_type = self.extract_type(self.extract_func_generic(ct, 1));
        let a0 = self.sp_arg(e, 0);
        let (new_expr, found) = self.get_attr(a0, &attr, None, None)?;

        let typ = self.extract_class_type_expr(a0);
        let typ = typ.and_then(|t| if types.is(t, &stdlib().type_wrap) { types.get_class(self.extract_class_generic(t, 0)) } else { Some(t) });
        if !found {
            let Some(typ) = typ else { return Ok(None) };
            if self.sp_has_rtti(typ) {
                let f = self.sp_id(&mm("RTTIType", "_getattr"));
                let s = self.sp_str(&attr);
                let rn = self.sp_rn(attr_type);
                let t = self.sp_id(&rn);
                let c = self.sp_call(f, vec![a0, s, t]);
                return self.transform(Some(c));
            } else {
                let p = types.pretty_string(self.extract_type_expr(a0));
                typhon_common::bail_e!(ErrorKind::DOT_NO_ATTR, self.cache.ast[e].src, p, attr);
            }
        } else {
            let mut new_expr = match new_expr {
                Some(x) => x,
                None => {
                    let d = self.sp_dot(a0, &attr);
                    self.tr(d)?
                }
            };
            if !types.is(attr_type, &stdlib().none_type) && self.wrap_expr(&mut new_expr, attr_type, None, true)? {
                let nt = self.ety(new_expr);
                self.unify(nt, attr_type)?;
            }
            Ok(Some(new_expr))
        }
    }

    /// `transformSetAttr(expr)`: `setattr` -> `AssignMemberStmt`.
    pub fn transform_set_attr(&mut self, e: ExprId) -> Result<Option<ExprId>> {
        let types = self.types();
        let ct = self.sp_callee_type(e);
        let attr = self.get_str_literal(self.extract_func_generic(ct, 0), 0);
        let a0 = self.sp_arg(e, 0);
        let a1 = self.sp_arg(e, 1);
        let typ = self.extract_class_type_expr(a0);
        let typ = typ.and_then(|t| if types.is(t, &stdlib().type_wrap) { types.get_class(self.extract_class_generic(t, 0)) } else { Some(t) });
        let Some(typ) = typ else { return Ok(None) };
        if self.sp_has_rtti(typ) {
            let (_, found) = self.get_attr(a0, &attr, None, None)?;
            if !found {
                let f = self.sp_id(&mm("RTTIType", "_setattr"));
                let s = self.sp_str(&attr);
                let c = self.sp_call(f, vec![a0, s, a1]);
                return self.transform(Some(c));
            }
        }
        let am = self.n_stmt(AssignMemberStmt { lhs: Some(a0), member: attr, rhs: Some(a1), type_expr: None });
        let nt = self.sp_id(&stdlib().none_type);
        let nc = self.sp_call(nt, vec![]);
        let se = self.n_expr(StmtExpr { items: vec![am], expr: Some(nc) });
        self.transform(Some(se))
    }

    /// `transformCompileError(expr)`: raise a compiler error.
    pub fn transform_compile_error(&mut self, e: ExprId) -> Result<Option<ExprId>> {
        let ct = self.sp_callee_type(e);
        let msg = self.get_str_literal(self.extract_func_generic(ct, 0), 0);
        typhon_common::bail_e!(ErrorKind::CUSTOM, self.cache.ast[e].src, msg);
    }

    /// `transformTupleFn(expr)`: convert a class to a tuple.
    pub fn transform_tuple_fn(&mut self, e: ExprId) -> Result<Option<ExprId>> {
        let types = self.types();
        let n = self.sp_callx(e).items.len();
        for i in 0..n {
            let a = self.sp_arg(e, i);
            let a = self.tr(a)?;
            self.sp_set_arg(e, i, a);
        }
        let a0 = self.sp_arg(e, 0);
        let Some(cls) = self.extract_class_type(self.ety(a0)) else { return Ok(None) };

        // tuple(ClassType) is a tuple type that corresponds to a class
        if Self::is_type_expr(self.cache, a0) {
            if self.realize(cls)?.is_none() {
                return Ok(Some(e));
            }

            let mut items = vec![];
            let ft = self.get_class_field_types(cls)?;
            for (i, f) in ft.into_iter().enumerate() {
                let rt = self.realize(f)?;
                typhon_common::seqassert!(rt.is_some(), "cannot realize '{}' in {}", self.get_class_type(cls).fields[i].name, types.debug_string(cls, 2));
                let rn = types.realized_name(rt.expect("cannot realize field"));
                items.push(self.sp_id(&rn));
            }
            let t = self.sp_id(&stdlib().tuple);
            let ie = self.n_expr(InstantiateExpr { expr: Some(t), items });
            return self.transform(Some(ie));
        }

        let mut args = vec![];
        let var = self.get_temporary_var("tup");
        for field in self.get_class_fields(cls) {
            let v = self.sp_id(&var);
            args.push(self.sp_dot(v, &field.name));
        }
        let v = self.sp_id(&var);
        let a = self.sp_assign(v, Some(a0), None);
        let t = self.sp_tuple(args);
        let se = self.n_expr(StmtExpr { items: vec![a], expr: Some(t) });
        self.transform(Some(se))
    }

    /// `transformTypeFn(expr)`: `type(x)` -> type `IdExpr`.
    pub fn transform_type_fn(&mut self, e: ExprId) -> Result<Option<ExprId>> {
        let types = self.types();
        let a0 = self.sp_arg(e, 0);
        let a0 = self.tr(a0)?;
        self.sp_set_arg(e, 0, a0);
        let mut value_type = self.ety(a0);
        if types.is(value_type, &stdlib().optional) && types.has_unbounds(value_type, false) {
            return Ok(None);
        }
        // Metaclass is type[] w/o parameters: type(type) is type
        if Self::is_type_expr(self.cache, a0) || types.is(value_type, &stdlib().type_wrap) {
            value_type = self.get_metaclass_type()?;
        }
        let et = self.ety(e);
        let tv = self.instantiate_type_var(value_type)?;
        self.unify(et, tv)?;
        if self.realize(et)?.is_none() {
            return Ok(None);
        }

        let rn = types.realized_name(et);
        let id = self.sp_id(&rn);
        self.set_type(id, et);
        self.set_done_expr(id);
        Ok(Some(id))
    }

    /// `transformRealizedFn(expr)`: `static.realized` -> fully realized function identifier.
    pub fn transform_realized_fn(&mut self, e: ExprId) -> Result<Option<ExprId>> {
        let types = self.types();
        let a0 = self.sp_arg(e, 0);
        let mut f = self.extract_type(self.ety(a0));
        if let Some(fns) = types.get_str_static(f) {
            // First argument can just be a literal string of function canonical name
            let val = TypecheckVisitor::find(self, &fns, 0);
            if let Some(val) = val
                && val.borrow().is_func(&types)
            {
                let vt = val.borrow().ty();
                f = self.instantiate_type(vt, None)?;
            }
        } else {
            let pt = types.get_partial(self.ety(a0));
            if types.get_func(f).is_none()
                && let Some(pt) = pt.filter(|p| types.is_partial_empty(*p))
            {
                let pft = types.generalize(types.partial_func(pt), 0);
                f = self.instantiate_type(pft, None)?;
            }
        }
        let Some(fnc) = types.get_func(f) else {
            typhon_common::bail_e!(ErrorKind::CALL_REALIZED_FN, self.cache.ast[a0].src)
        };
        let a1 = self.sp_arg(e, 1);
        let Some(argt) = self.expr_class(a1) else { return Ok(None) };
        typhon_common::seqassert!(types.is(argt, &stdlib().tuple), "not a tuple");
        let mut call_args = vec![];
        for argument in types.generics(argt) {
            let mut t = argument.ty();
            if types.is(t, &stdlib().type_wrap) {
                t = self.extract_class_generic(t, 0);
            }
            let v = self.sp_none_of(t);
            call_args.push(self.cache.ast.call_arg("", Some(v)));
        }
        let mut changes = Unification::new();
        if self.can_call(fnc, &call_args, None, None, Some(&mut changes), false)? < 0 {
            typhon_common::bail_e!(ErrorKind::CALL_REALIZED_FN, self.cache.ast[a0].src);
        }
        if let Some(r) = self.realize(f)? {
            let fr = types.get_func(r).unwrap();
            let rn = types.realized_name(fr);
            let id = self.sp_id(&rn);
            self.set_type(id, r);
            self.set_done_expr(id);
            return Ok(Some(id));
        }
        Ok(None)
    }

    /// `transformStaticPrintFn(expr)`: print static types of the arguments.
    pub fn transform_static_print_fn(&mut self, e: ExprId) -> Result<Option<ExprId>> {
        let types = self.types();
        let a0 = self.sp_arg(e, 0);
        let items = self.cache.ast[a0].kind.as_call().map(|c| c.items.clone()).unwrap_or_default();
        for a in items {
            let t = a.value.and_then(|v| self.expr_type(v));
            eprintln!(
                "[print] {}: {} ({}){}",
                self.src,
                t.map(|t| types.debug_string(t, 2)).unwrap_or_else(|| "-".into()),
                t.map(|t| types.realized_name(t)).unwrap_or_else(|| "-".into()),
                if t.is_some_and(|t| types.get_static(t).is_some()) { " [static]" } else { "" }
            );
        }
        Ok(None)
    }

    /// `transformHasRttiFn(expr)`: static boolean RTTI status of a type.
    pub fn transform_has_rtti_fn(&mut self, e: ExprId) -> Result<Option<ExprId>> {
        let types = self.types();
        self.sp_set_static_kind(e, LiteralKind::Bool);
        let ct = self.sp_callee_type(e);
        let Some(t) = types.get_class(self.extract_func_generic(ct, 0)) else { return Ok(None) };
        let b = self.sp_bool(self.sp_has_rtti(t));
        self.transform(Some(b))
    }
}
