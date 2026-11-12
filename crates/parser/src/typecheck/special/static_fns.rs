//! `std.internal.static` handlers.

use typhon_common::mangle::stdlib;
use typhon_common::{ErrorKind, Result};

use super::super::TypecheckVisitor;
use crate::ast::{CallArg, KeywordStarExpr, StarExpr, StmtExpr};
use crate::ids::{ExprId, TypeId};
use crate::types::LiteralKind;

impl<'a> TypecheckVisitor<'a> {
    /// `extractFunction(arg0 type)` or the reference "expected a function" error.
    fn sp_expect_function(&self, t: TypeId, shown: TypeId) -> Result<TypeId> {
        match self.extract_function(t) {
            Some(f) => Ok(f),
            None => {
                let p = self.cache.types.pretty_string(shown);
                typhon_common::bail_e!(ErrorKind::CUSTOM, self.src, "expected a function, got '{}'", p)
            }
        }
    }
    /// `extractFuncGeneric(expr->getExpr()->getType())->getIntStatic()->value`.
    fn sp_static_index(&self, e: ExprId) -> i64 {
        let ct = self.sp_callee_type(e);
        let idx = self.cache.types.get_int_static(self.extract_func_generic(ct, 0));
        typhon_common::seqassert!(idx.is_some(), "expected a static integer");
        idx.expect("expected a static integer")
    }

    /// `transformStaticFnCanCall(expr)`.
    pub fn transform_static_fn_can_call(&mut self, e: ExprId) -> Result<Option<ExprId>> {
        let types = self.types();
        self.sp_set_static_kind(e, LiteralKind::Bool);

        let a0 = self.sp_arg(e, 0);
        let Some(typ) = self.extract_class_type_expr(a0) else { return Ok(None) };

        let a1 = self.sp_arg(e, 1);
        let inargs = self.unpack_tuple_types(a1)?;
        let a2 = self.sp_arg(e, 2);
        let kwargs = self.unpack_tuple_types(a2)?;
        typhon_common::seqassert!(inargs.is_some() && kwargs.is_some(), "bad call to fn_can_call");
        let (Some(inargs), Some(kwargs)) = (inargs, kwargs) else { return Ok(None) };

        let mut call_args = vec![];
        for (v, t) in inargs.into_iter().chain(kwargs) {
            let x = self.sp_none_of(t); // dummy expression
            call_args.push(CallArg::new(v, Some(x)));
        }
        let st = stdlib();
        if !types.is(typ, &st.function) && types.get_partial(typ).is_none() && !types.is(typ, &st.type_wrap) {
            self.warn(ErrorKind::CUSTOM, self.src, &[&"cannot use fn_can_call on non-functions"]);
            let b = self.sp_bool(false);
            return self.transform(Some(b));
        }
        let r = self.match_callable(typ, &call_args, None, None, None, None)?;
        let b = self.sp_bool(r >= 0);
        self.transform(Some(b))
    }

    /// `transformStaticFnArgHasType(expr)`.
    pub fn transform_static_fn_arg_has_type(&mut self, e: ExprId) -> Result<Option<ExprId>> {
        let types = self.types();
        self.sp_set_static_kind(e, LiteralKind::Bool);
        let a0t = self.ety(self.sp_arg(e, 0));
        let f = self.sp_expect_function(a0t, a0t)?;
        let idx = self.sp_static_index(e);
        let v = idx >= 0 && (idx as usize) < types.func_size(f) && types.can_realize(types.func_arg(f, idx as usize));
        let b = self.sp_bool(v);
        self.transform(Some(b))
    }

    /// `transformStaticFnArgGetType(expr)`.
    pub fn transform_static_fn_arg_get_type(&mut self, e: ExprId) -> Result<Option<ExprId>> {
        let types = self.types();
        let a0t = self.ety(self.sp_arg(e, 0));
        let f = self.sp_expect_function(a0t, a0t)?;
        let idx = self.sp_static_index(e);
        if idx < 0 || idx as usize >= types.func_size(f) || !types.can_realize(types.func_arg(f, idx as usize)) {
            typhon_common::bail_e!(ErrorKind::CUSTOM, self.src, "argument does not have type");
        }
        let rn = types.realized_name(types.func_arg(f, idx as usize));
        let id = self.sp_id(&rn);
        self.transform(Some(id))
    }

    /// `transformStaticFnArgs(expr)`.
    pub fn transform_static_fn_args(&mut self, e: ExprId) -> Result<Option<ExprId>> {
        let a0t = self.ety(self.sp_arg(e, 0));
        let f = self.sp_expect_function(a0t, a0t)?;
        let mut v = vec![];
        for a in self.sp_fn_params(f) {
            if a.is_generic() {
                continue;
            }
            let (_, n) = a.name_with_stars();
            let n = self.get_unmangled_name(&n);
            v.push(self.sp_str(&n));
        }
        let t = self.sp_tuple(v);
        self.transform(Some(t))
    }

    /// `transformStaticFnHasDefault(expr)`.
    pub fn transform_static_fn_has_default(&mut self, e: ExprId) -> Result<Option<ExprId>> {
        self.sp_set_static_kind(e, LiteralKind::Bool);
        let a0t = self.ety(self.sp_arg(e, 0));
        let f = self.sp_expect_function(a0t, a0t)?;
        let idx = self.sp_static_index(e);
        let mut argument_index = 0i64;
        for argument in self.sp_fn_params(f) {
            if argument.is_generic() {
                continue;
            }
            let cur = argument_index;
            argument_index += 1;
            if cur == idx {
                let b = self.sp_bool(argument.default_value.is_some());
                return self.transform(Some(b));
            }
        }
        typhon_common::bail_e!(ErrorKind::CUSTOM, self.src, "argument out of bounds");
    }

    /// `transformStaticFnGetDefault(expr)`.
    pub fn transform_static_fn_get_default(&mut self, e: ExprId) -> Result<Option<ExprId>> {
        let a0t = self.ety(self.sp_arg(e, 0));
        let f = self.sp_expect_function(a0t, a0t)?;
        let idx = self.sp_static_index(e);
        let mut argument_index = 0i64;
        for argument in self.sp_fn_params(f) {
            if argument.is_generic() {
                continue;
            }
            let cur = argument_index;
            argument_index += 1;
            if cur == idx {
                return self.transform(argument.default_value);
            }
        }
        typhon_common::bail_e!(ErrorKind::CUSTOM, self.src, "argument out of bounds");
    }

    /// `transformStaticFnWrapCallArgs(expr)`.
    pub fn transform_static_fn_wrap_call_args(&mut self, e: ExprId) -> Result<Option<ExprId>> {
        let a0 = self.sp_arg(e, 0);
        if self.expr_class(a0).is_none() {
            return Ok(None);
        }
        let a0t = self.ety(a0);
        let et = self.extract_type(a0t);
        self.sp_expect_function(et, a0t)?;

        let a1 = self.sp_arg(e, 1);
        let a2 = self.sp_arg(e, 2);
        let s = self.n_expr(StarExpr { expr: Some(a1) });
        let k = self.n_expr(KeywordStarExpr { expr: Some(a2) });
        let c = self.sp_call(a0, vec![s, k]);
        let mut temp_call = self.tr(c)?;
        if !self.cache.ast[temp_call].done {
            return Ok(None);
        }

        let mut stmts = vec![];
        while let Some(suite) = self.cache.ast[temp_call].kind.as_stmt().cloned() {
            stmts.extend(suite.items);
            temp_call = suite.expr.expect("StmtExpr without expression");
        }
        let call = self.cache.ast[temp_call].kind.as_call().cloned();
        typhon_common::seqassert!(call.is_some(), "expected wrapped function call: {}", self.cache.ast.expr_to_string(temp_call, -1));
        let call = call.expect("expected wrapped function call");
        stmts.push(self.sp_expr_stmt(call.expr.unwrap()));
        let tup_args: Vec<ExprId> = call.items.iter().filter_map(|a| a.value).collect();
        let t = self.sp_tuple(tup_args);
        let se = self.n_expr(StmtExpr { items: stmts, expr: Some(t) });
        self.transform(Some(se))
    }

    /// `transformStaticVars(expr)`.
    pub fn transform_static_vars(&mut self, e: ExprId) -> Result<Option<ExprId>> {
        let types = self.types();
        let ct = self.sp_callee_type(e);
        let t = self.extract_func_generic(ct, 0);
        if types.get_class(t).is_none() {
            return Ok(None);
        }
        let with_idx = self.get_bool_literal(t, 0);

        let a0 = self.sp_arg(e, 0);
        let ex = self.tr(a0)?;
        let Some(typ) = self.expr_class(ex) else { return Ok(None) };

        let mut tuple_items = vec![];
        // Note: the reference reuses `expr->begin()->value` (the original, untransformed slot) as
        // the receiver of each `DotExpr`; we do the same.
        let recv = self.sp_arg(e, 0);
        for (idx, f) in self.get_class_fields(typ).into_iter().enumerate() {
            let k = self.sp_str(&f.name);
            let v = self.sp_dot(recv, &f.name);
            if with_idx {
                let i = self.sp_int(idx as i64);
                tuple_items.push(self.sp_tuple(vec![i, k, v]));
            } else {
                tuple_items.push(self.sp_tuple(vec![k, v]));
            }
        }
        let t = self.sp_tuple(tuple_items);
        self.transform(Some(t))
    }

    /// `transformStaticChildren(expr)`.
    pub fn transform_static_children(&mut self, e: ExprId) -> Result<Option<ExprId>> {
        let types = self.types();
        let ct = self.sp_callee_type(e);
        let t = self.extract_func_generic(ct, 0);
        let Some(typ) = types.get_class(t) else { return Ok(None) };

        let mut tuple_items = vec![];
        let tname = types.class_name(typ).unwrap().to_string();
        let trn = types.realized_name(typ);
        let mut descendants: Vec<String> = self.get_class_type(typ).descendants.iter().cloned().collect();
        descendants.sort();
        for n in descendants {
            if n == tname {
                continue;
            }
            let mut reals: Vec<_> = self.get_class_name(&n).map(|c| c.realizations.iter().map(|(k, v)| (k.clone(), v.clone())).collect()).unwrap_or_default();
            reals.sort_by(|a: &(String, _), b| a.0.cmp(&b.0));
            for (_, cr) in reals {
                let (bases, rty) = {
                    let r = cr.borrow();
                    (r.bases.clone(), r.ty)
                };
                for b in bases {
                    if types.realized_name(b) == trn {
                        let rn = types.realized_name(rty.expect("realization without type"));
                        tuple_items.push(self.sp_id(&rn));
                        break;
                    }
                }
            }
        }
        let t = self.sp_tuple(tuple_items);
        self.transform(Some(t))
    }

    /// `transformStaticTupleType(expr)`.
    pub fn transform_static_tuple_type(&mut self, e: ExprId) -> Result<Option<ExprId>> {
        let types = self.types();
        let func_typ = types.get_func(self.sp_callee_type(e)).expect("not a function");
        let t = types.get_class(self.extract_func_generic(func_typ, 0));
        let Some(t) = t else { return Ok(None) };
        if self.realize(t)?.is_none() {
            return Ok(None);
        }
        let n = self.get_int_literal(self.extract_func_generic(func_typ, 1), 0);
        let f = self.get_class_fields(t);
        if n < 0 || n as usize >= f.len() {
            typhon_common::bail_e!(ErrorKind::CUSTOM, self.src, "invalid index");
        }
        let it = self.instantiate_type(f[n as usize].ty.expect("field without type"), Some(t))?;
        let rt = self.realize(it)?.expect("cannot realize tuple field");
        let rn = types.realized_name(rt);
        let id = self.sp_id(&rn);
        self.transform(Some(id))
    }

    /// `transformStaticFormat(expr)`.
    pub fn transform_static_format(&mut self, e: ExprId) -> Result<Option<ExprId>> {
        let types = self.types();
        self.sp_set_static_kind(e, LiteralKind::String);
        let func_typ = types.get_func(self.sp_callee_type(e)).expect("not a function");
        let fmt = self.get_str_literal(self.extract_func_generic(func_typ, 0), 0);
        let arg = self.get_str_literal(self.extract_func_generic(func_typ, 1), 0);
        // Note: the reference replaces each `%%` with `{}` and formats with fmt::vformat; other
        // braces in the format string are fmt syntax too. We substitute `%%` directly and
        // handle `{{`/`}}` escapes the same way fmt would.
        let r = fmt.replace("{{", "\u{0}").replace("}}", "\u{1}").replace("%%", &arg.replace('{', "\u{2}").replace('}', "\u{3}"));
        let r = r.replace('\u{0}', "{").replace('\u{1}', "}").replace('\u{2}', "{").replace('\u{3}', "}");
        let s = self.sp_str(&r);
        self.transform(Some(s))
    }

    /// `transformStaticIntToStr(expr)`.
    pub fn transform_static_int_to_str(&mut self, e: ExprId) -> Result<Option<ExprId>> {
        let types = self.types();
        self.sp_set_static_kind(e, LiteralKind::String);
        let func_typ = types.get_func(self.sp_callee_type(e)).expect("not a function");
        let val = self.get_int_literal(self.extract_func_generic(func_typ, 0), 0);
        let s = self.sp_str(&val.to_string());
        self.transform(Some(s))
    }

    /// `transformStaticPlatform(expr)`.
    pub fn transform_static_platform(&mut self, e: ExprId) -> Result<Option<ExprId>> {
        self.sp_set_static_kind(e, LiteralKind::String);
        let platform = if cfg!(windows) {
            "win32"
        } else if cfg!(target_os = "linux") {
            "linux"
        } else if cfg!(target_os = "macos") {
            "darwin"
        } else if cfg!(unix) {
            "unix"
        } else {
            ""
        };
        if platform.is_empty() {
            typhon_common::bail_e!(ErrorKind::CUSTOM, self.cache.ast[e].src, "unsupported platform");
        }
        let s = self.sp_str(platform);
        self.transform(Some(s))
    }

    /// `transformStaticContains(expr)`: `static.contains` for literal strings.
    pub fn transform_static_contains(&mut self, e: ExprId) -> Result<Option<ExprId>> {
        let types = self.types();
        self.sp_set_static_kind(e, LiteralKind::Bool);
        let func_typ = types.get_func(self.sp_callee_type(e)).expect("not a function");
        let t = self.get_str_literal(self.extract_func_generic(func_typ, 0), 0);
        let p = self.get_str_literal(self.extract_func_generic(func_typ, 1), 0);
        let b = self.sp_bool(t.contains(&p));
        self.transform(Some(b))
    }
}
