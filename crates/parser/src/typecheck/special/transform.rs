//! Special-call handlers  (`transformNamedTuple`, `transformSuperF`,
//! `transformSuper`, `transformPtr`, `transformArray`, `transformIsInstance`,
//! `transformStaticLen`, `transformHasAttr`, `transformGetAttr`, `transformSetAttr`,
//! `transformCompileError`, `transformTupleFn`, `transformTypeFn`, `transformRealizedFn`,
//! `transformStaticPrintFn`, `transformHasRttiFn`).

use typhon_common::mangle::stdlib;
use typhon_common::{ErrorKind, Result};

use super::super::TypecheckVisitor;
use super::mm;
use crate::ast::{Attr, CallArg, EllipsisExpr, EllipsisType, ExprTag, Param, ParamStatus};
use crate::ids::{ExprId, TypeId};
use crate::types::LiteralKind;

impl<'a> TypecheckVisitor<'a> {
    /// `transformNamedTuple(expr)`.
    /// @example
    ///   `namedtuple("NT", ["a", ("b", int)])` -> ```@tuple
    ///                                               class NT[T1]:
    ///                                                 a: T1
    ///                                                 b: int```
    pub fn transform_named_tuple(&mut self, e: ExprId) -> Result<Option<ExprId>> {
        // Ensure that namedtuple call is valid
        let ct = self.sp_callee_type(e);
        let name = self.get_str_literal(self.extract_func_generic(ct, 0), 0);
        let call = self.sp_callx(e);
        if call.items.len() != 1 {
            typhon_common::bail_e!(ErrorKind::CALL_NAMEDTUPLE, self.cache.ast[e].src);
        }

        // Construct the class statement
        let mut generics = vec![];
        let mut params = vec![];
        let a0 = call.items[0].value.unwrap();
        let orig = self.cache.ast[a0].orig_expr.expect("namedtuple without original expression");
        let items = self.cache.ast[orig].kind.as_tuple().map(|t| t.items.clone()).unwrap_or_default();
        let mut ti = 1;
        for i in items {
            if let Some(s) = self.cache.ast[i].kind.as_string() {
                let v = s.value();
                let ty = self.sp_id(&stdlib().type_);
                generics.push(Param::with_status(format!("T{ti}"), Some(ty), None, ParamStatus::Generic));
                let ty = self.sp_id(&format!("T{ti}"));
                ti += 1;
                params.push(Param::with_status(v, Some(ty), None, ParamStatus::Value));
                continue;
            }
            if let Some(t) = self.cache.ast[i].kind.as_tuple().cloned()
                && t.items.len() == 2
                && let Some(s) = self.cache.ast[t.items[0]].kind.as_string()
            {
                let v = s.value();
                let ty = self.transform_type(Some(t.items[1]), false)?;
                params.push(Param::with_status(v, ty, None, ParamStatus::Value));
                continue;
            }
            typhon_common::bail_e!(ErrorKind::CALL_NAMEDTUPLE, self.cache.ast[i].src);
        }
        params.extend(generics);
        let tup = self.sp_id("tuple");
        let src = self.src;
        let cls = self.cache.ast.new_class(src, name.clone(), params, None, vec![tup], vec![]);
        let cls = self.sp_suite(vec![cls]);
        if let Some(sc) = self.cache.hooks.scoping.clone() {
            sc(self.cache, cls, None)?;
        }
        let c = self.tr_stmt(cls)?;
        self.prepend_stmts.borrow_mut().push(c);
        let id = self.sp_id(&name);
        self.transform_type(Some(id), false)
    }

    /// `transformFunctoolsPartial(expr)`.
    /// @example
    ///   `partial(foo, 1, a=2)` -> `foo(1, a=2, ...)`
    pub fn transform_functools_partial(&mut self, e: ExprId) -> Result<Option<ExprId>> {
        let call = self.sp_callx(e);
        if call.items.is_empty() {
            typhon_common::bail_e!(ErrorKind::CALL_PARTIAL, self.src);
        }
        let mut args: Vec<CallArg> = call.items[1..].to_vec();
        let el = self.n_expr(EllipsisExpr { mode: EllipsisType::Partial });
        args.push(CallArg::new("", Some(el)));
        let c = self.sp_call_args(call.items[0].value.unwrap(), args);
        self.transform(Some(c))
    }

    /// `transformSuperF(expr)`: access to the previous matching overload.
    pub fn transform_super_f(&mut self, e: ExprId) -> Result<Option<ExprId>> {
        let types = self.types();
        let func = {
            let mut c = self.ctx.borrow_mut();
            c.get_base().ty
        };
        let func = func.and_then(|f| types.get_func(f)).expect("superf outside of a function");

        // Find list of matching superf methods
        let mut supers = vec![];
        if !self.is_dispatch(func) {
            let aid = types.func_ast_id(func);
            if let Some(a) = self.cache.ast[aid].attrs.get_str(Attr::ParentClass).map(|s| s.to_string()) {
                let um = self.get_unmangled_name(&types.func_name(func));
                let m = self.get_class_name(&a).and_then(|c| c.methods.get(&um).cloned());
                if let Some(m) = m {
                    let fname = types.func_name(func);
                    for overload in self.get_overloads(&m) {
                        if Self::is_dispatch_name(&overload) {
                            continue;
                        }
                        if overload == fname {
                            break;
                        }
                        supers.push(self.get_function_name(&overload).and_then(|f| f.ty).expect("bad overload"));
                    }
                }
                supers.reverse();
            }
        }
        if supers.is_empty() {
            typhon_common::bail_e!(ErrorKind::CALL_SUPERF, self.cache.ast[e].src);
        }

        let call = self.sp_callx(e);
        let inner = call.items.first().and_then(|a| a.value).and_then(|v| self.cache.ast[v].kind.as_call().cloned());
        typhon_common::seqassert!(call.items.len() == 1 && inner.is_some(), "bad superf call");
        let inner = inner.expect("bad superf call");
        let new_args: Vec<CallArg> = inner.items.iter().map(|a| self.cache.ast.call_arg("", a.value)).collect();
        let parent = types.func_parent(func).and_then(|p| types.get_class(p));
        let m = self.find_matching_methods(parent, &supers, &new_args, None)?;
        if m.is_empty() {
            typhon_common::bail_e!(ErrorKind::CALL_SUPERF, self.cache.ast[e].src);
        }
        let fname = types.func_name(m[0]);
        let f = self.sp_id(&fname);
        let c = self.sp_call_args(f, new_args);
        self.transform(Some(c))
    }

    /// `transformSuper()`: replace `super()` with the current self object cast to the first
    /// inherited type.
    /// TODO: only an empty super() is currently supported.
    pub fn transform_super(&mut self) -> Result<Option<ExprId>> {
        let types = self.types();
        let base_ty = {
            let mut c = self.ctx.borrow_mut();
            c.get_base().ty
        };
        let Some(base_ty) = base_ty else { typhon_common::bail_e!(ErrorKind::CALL_SUPER_PARENT, self.src) };
        let func_typ = types.get_func(base_ty);
        let Some(func_typ) = func_typ.filter(|f| self.cache.ast[types.func_ast_id(*f)].attrs.has(Attr::Method)) else {
            typhon_common::bail_e!(ErrorKind::CALL_SUPER_PARENT, self.src)
        };
        if types.func_size(func_typ) == 0 {
            typhon_common::bail_e!(ErrorKind::CALL_SUPER_PARENT, self.src);
        }

        let typ = types.get_class(self.extract_func_arg_type(func_typ, 0)).unwrap();
        let first = self.sp_fn_params(func_typ)[0].name.clone();
        let slf = self.sp_id(&first);
        self.set_type(slf, typ);
        // Self may specialize an inherited method for a derived receiver. super()
        // must still start after the method's defining class, not after Self.
        let lexical = types.func_parent(func_typ).and_then(|p| types.get_class(p)).unwrap_or(typ);
        let lname = types.class_name(lexical).unwrap().to_string();
        let typ_expr = self.sp_id(&lname);
        let tv = self.instantiate_type_var(lexical)?;
        self.set_type(typ_expr, tv);
        let f = self.sp_id(&mm("Super", "__new__"));
        let c = self.sp_call(f, vec![typ_expr, slf]);
        self.transform(Some(c))
    }

    /// `transformPtr(expr)`: `__ptr__` creates a pointer to an object. Ensure that the
    /// argument is a variable binding.
    pub fn transform_ptr(&mut self, e: ExprId) -> Result<Option<ExprId>> {
        let a0 = self.sp_arg(e, 0);
        let a0 = self.tr(a0)?;
        self.sp_set_arg(e, 0, a0);

        let mut head = self.get_head_expr(a0);
        loop {
            if self.extract_class_type_expr(head).is_none() {
                return Ok(None);
            }
            match self.cache.ast[head].kind.tag() {
                ExprTag::Id => {
                    let v = self.cache.ast[head].kind.as_id().unwrap().value.clone();
                    let val = self.find(&v, self.get_time());
                    if !val.is_some_and(|v| v.borrow().is_var(&self.cache.types)) {
                        typhon_common::bail_e!(ErrorKind::CALL_PTR_VAR, self.cache.ast[a0].src);
                    }
                    break;
                }
                ExprTag::Dot => {
                    head = self.cache.ast[head].kind.as_dot().unwrap().expr.unwrap();
                }
                _ => {
                    typhon_common::bail_e!(ErrorKind::CALL_PTR_VAR, self.cache.ast[a0].src);
                }
            }
        }

        let ptr = self.get_stdlib_type(&stdlib().ptr)?;
        let at = self.ety(a0);
        let pt = self.instantiate_type_with(ptr, &[at])?;
        let et = self.ety(e);
        self.unify(et, pt)?;
        if self.cache.ast[a0].done {
            self.set_done_expr(e);
        }
        Ok(None)
    }

    /// `transformArray(expr)`: `__array__` creates a stack-allocated array via alloca.
    pub fn transform_array(&mut self, e: ExprId) -> Result<Option<ExprId>> {
        let types = self.types();
        let arr_typ = types.get_func(self.sp_callee_type(e)).expect("not a function");
        let parent = types.func_parent(arr_typ).expect("__array__ without parent");
        let g = self.extract_class_generic(parent, 0);
        let ptr = self.get_stdlib_type(&stdlib().ptr)?;
        let pt = self.instantiate_type_with(ptr, &[g])?;
        let et = self.ety(e);
        self.unify(et, pt)?;
        if self.realize(et)?.is_some() {
            self.set_done_expr(e);
        }
        Ok(None)
    }

    /// `transformStaticLen(expr)`: static integer length of static strings and tuples.
    pub fn transform_static_len(&mut self, e: ExprId) -> Result<Option<ExprId>> {
        let types = self.types();
        self.sp_set_static_kind(e, LiteralKind::Int);

        let a0 = self.sp_arg(e, 0);
        let a0 = self.tr(a0)?;
        self.sp_set_arg(e, 0, a0);
        let typ = self.extract_type_expr(a0);

        if let Some(ss) = types.get_str_static(typ) {
            // Case: staticlen on static strings
            let n = self.sp_int(typhon_common::strings::utf8_strlen(&ss) as i64);
            return self.transform(Some(n));
        }
        let Some(cls) = types.get_class(typ) else { return Ok(None) };
        if let Some(u) = types.get_union(typ) {
            if self.realize(typ)?.is_some() {
                let n = self.sp_int(types.union_realization_types(u).len() as i64);
                return self.transform(Some(n));
            }
            return Ok(None);
        }
        if !types.is_record(cls) {
            typhon_common::bail_e!(ErrorKind::EXPECTED_TUPLE, self.cache.ast[a0].src);
        }
        let n = self.sp_int(self.get_class_fields(cls).len() as i64);
        self.transform(Some(n))
    }

    /// Strip `TypeWrap[T]` -> `T` (class).
    fn sp_unwrap_type_wrap(&self, t: Option<TypeId>) -> Option<TypeId> {
        let types = &self.cache.types;
        let t = t?;
        if types.is(t, &stdlib().type_wrap) { types.get_class(self.extract_class_generic(t, 0)) } else { Some(t) }
    }

    /// `transformHasAttr(expr, allowDynamic)`: static boolean `hasattr`. Also supports
    /// additional argument types that are used to check for a matching overload.
    pub fn transform_has_attr(&mut self, e: ExprId, allow_dynamic: bool) -> Result<Option<ExprId>> {
        let types = self.types();
        self.sp_set_static_kind(e, LiteralKind::Bool);

        let a0 = self.sp_arg(e, 0);
        let typ = self.sp_unwrap_type_wrap(self.extract_class_type_expr(a0));
        let Some(typ) = typ else { return Ok(None) };

        let ct = self.sp_callee_type(e);
        let attr = self.get_str_literal(self.extract_func_generic(ct, 0), 0);
        let mut args: Vec<(String, TypeId)> = vec![(String::new(), typ)];

        let a1 = self.sp_arg(e, 1);
        if let Some(tup) = self.cache.ast[a1].kind.as_call().cloned() {
            let mut items = tup.items.clone();
            for a in items.iter_mut() {
                let v = self.tr(a.value.unwrap())?;
                a.value = Some(v);
                if self.expr_class(v).is_none() {
                    self.cache.ast[a1].kind.as_call_mut().unwrap().items = items;
                    return Ok(None);
                }
                let t = self.extract_type_expr(v);
                let t = if types.is(t, &stdlib().type_wrap) { self.extract_class_generic(t, 0) } else { t };
                args.push((String::new(), t));
            }
            self.cache.ast[a1].kind.as_call_mut().unwrap().items = items;
        }
        let a2 = self.sp_arg(e, 2);
        for (n, ne) in self.extract_named_tuple(a2) {
            let ne = self.tr(ne)?;
            let t = self.extract_type_expr(ne);
            let t = if types.is(t, &stdlib().type_wrap) { self.extract_class_generic(t, 0) } else { t };
            args.push((n, t));
        }

        if types.get_union(typ).is_some() && allow_dynamic {
            let mut cond: Option<ExprId> = None;
            let union_types = types.union_realization_types(types.get_union(typ).unwrap());
            for union_type in union_types {
                let Some(tu) = self.realize(union_type)? else { return Ok(None) };
                let rn = types.realized_name(types.get_class(tu).unwrap());
                let te = self.sp_id(&rn);
                let isi = self.sp_id("isinstance");
                let x = self.sp_arg(e, 0);
                let c1 = self.sp_call(isi, vec![x, te]);
                let ha = self.sp_id("hasattr");
                let s = self.sp_str(&attr);
                let c2 = self.sp_call(ha, vec![te, s]);
                let ex = self.sp_bin(c1, "&&", c2);
                cond = Some(match cond {
                    None => ex,
                    Some(c) => self.sp_bin(c, "||", ex),
                });
            }
            let Some(cond) = cond else {
                let b = self.sp_bool(false);
                return self.transform(Some(b));
            };
            return self.transform(Some(cond));
        } else if types.is(typ, &stdlib().named_tuple) {
            if !types.can_realize(typ) {
                return Ok(None);
            }
            let id = self.get_int_literal(typ, 0);
            let len = self.cache.types.generated_tuple_names.borrow().len();
            typhon_common::seqassert!(id >= 0 && (id as usize) < len, "bad id: {}", id);
            let names = self.sp_tuple_names(id);
            let b = self.sp_bool(names.contains(&attr));
            return self.transform(Some(b));
        }

        // enum_test: __members__ is an ordinary ClassVar, so hasattr must include
        // class variables just as attribute lookup does (also through instances).
        let cls = types.get_class(typ);
        let mut exists = !self.find_method(cls, &attr)?.is_empty()
            || self.find_member(typ, &attr).is_some()
            || self.get_class_type(typ).class_vars.contains_key(&attr);
        if exists && args.len() > 1 {
            exists &= self.find_best_method_named(typ, &attr, &args)?.is_some();
        }

        if !exists && allow_dynamic && self.sp_has_rtti(typ) {
            let f = self.sp_id(&mm("RTTIType", "_hasattr"));
            let x = self.sp_arg(e, 0);
            let s = self.sp_str(&attr);
            let c = self.sp_call(f, vec![x, s]);
            return self.transform(Some(c));
        }
        let b = self.sp_bool(exists);
        self.transform(Some(b))
    }
}
