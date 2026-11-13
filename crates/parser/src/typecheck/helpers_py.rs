//! Part 4: `wrapExpr`,
//! `canWrapExpr`, `unpackTupleTypes`, `extractNamedTuple`.

use std::rc::Rc;

use typhon_common::mangle::{self, stdlib};
use typhon_common::{ErrorKind, Result};

use super::{TypecheckVisitor, WrapFn};
use crate::ast::{CallArg, CallExpr, DotExpr, EllipsisExpr, EllipsisType, FunctionStmt, IdExpr, IndexExpr, InstantiateExpr, IntExpr, Param, ParamStatus, ReturnStmt, StarExpr, StmtExpr, SuiteStmt};
use crate::cache::fn_optional_unwrap;
use crate::ids::{ExprId, TypeId};
use crate::types::LiteralKind;

/// Result of `canWrapExpr`: `(canWrap, newType, wrapFn)`.
pub type CanWrap = (bool, Option<TypeId>, Option<WrapFn>);

impl<'a> TypecheckVisitor<'a> {
    pub(crate) fn call_n(&mut self, f: ExprId, args: Vec<ExprId>) -> ExprId {
        let items = args.into_iter().map(|a| CallArg::new("", Some(a))).collect();
        self.n_expr(CallExpr { expr: Some(f), items, ordered: false, partial: false })
    }
    pub(crate) fn id_n(&mut self, s: &str) -> ExprId {
        self.n_expr(IdExpr::new(s))
    }
    pub(crate) fn dot_n(&mut self, e: ExprId, m: &str) -> ExprId {
        self.n_expr(DotExpr { expr: Some(e), member: m.into() })
    }

    /// `wrapExpr(&expr, expectedType, callee, allowUnwrap)`: coerce `*expr` if needed.
    pub fn wrap_expr(&mut self, expr: &mut ExprId, expected: TypeId, callee: Option<TypeId>, allow_unwrap: bool) -> Result<bool> {
        let types = self.types();
        let is_ellipsis = self.cache.ast[*expr].kind.as_ellipsis().is_some();
        let et = self.ety(*expr);
        let (can, _, f) = self.can_wrap_expr(et, expected, callee, allow_unwrap, is_ellipsis)?;
        // TODO: get rid of this line one day!
        if types.static_kind(et) != LiteralKind::Runtime && types.static_kind(expected) == LiteralKind::Runtime {
            let u = self.get_underlying_static_type(et)?;
            self.set_type(*expr, u);
        }
        if can {
            if let Some(f) = f {
                let w = f(self, *expr)?;
                *expr = self.tr(w)?;
            }
        }
        Ok(can)
    }

    /// Element-wise tuple coercion: `Tuple[int, str]` -> `Tuple[float, str]`,
    /// `Tuple[NoneType, float]` -> `Tuple[Optional[float], float]` (Python/mypy accept these).
    /// Returns `None` if not applicable (no element needs a wrapper, or one cannot be wrapped).
    fn can_wrap_tuple(&mut self, xc: Option<TypeId>, ec: Option<TypeId>, callee: Option<TypeId>, _allow_unwrap: bool) -> Result<Option<CanWrap>> {
        let types = self.types();
        let st = stdlib();
        let (Some(x), Some(e)) = (xc, ec) else { return Ok(None) };
        if !types.is(x, &st.tuple) || !types.is(e, &st.tuple) {
            return Ok(None);
        }
        let n = types.generics_len(x);
        if n == 0 || n != types.generics_len(e) || !types.can_realize(e) {
            return Ok(None);
        }
        let xs: Vec<TypeId> = types.generics(x).iter().map(|g| g.ty()).collect();
        let es: Vec<TypeId> = types.generics(e).iter().map(|g| g.ty()).collect();
        // Every source element must be concrete, except `None` literals (`Optional[T]`
        // with unbound T) which are wrapped into the expected Optional below.
        if xs.iter().any(|t| !types.can_realize(*t) && !types.is(*t, &st.optional)) {
            return Ok(None);
        }
        let mut wraps: Vec<Option<WrapFn>> = vec![];
        let mut out = vec![];
        let mut any = false;
        for i in 0..n {
            if types.can_realize(xs[i]) && types.realized_name(xs[i]) == types.realized_name(es[i]) {
                wraps.push(None);
                out.push(xs[i]);
                continue;
            }
            if !types.can_realize(xs[i]) {
                // `None` literal (Optional[T], T unbound) against an expected Optional[U].
                if types.is(es[i], &st.optional) && types.unify_or_undo(xs[i], es[i], self)? {
                    wraps.push(None);
                    out.push(es[i]);
                    continue;
                }
                return Ok(None);
            }
            // Only widening conversions (int -> float, T/None -> Optional[T]); never unwrap.
            let widening = (types.is(es[i], "float") && types.is(xs[i], "Int")) || types.is(es[i], &st.optional);
            if !widening {
                return Ok(None);
            }
            let (ok, t, f) = self.can_wrap_expr(xs[i], es[i], callee, false, false)?;
            if !ok || f.is_none() {
                return Ok(None);
            }
            any = true;
            out.push(t.unwrap_or(es[i]));
            wraps.push(f);
        }
        if !any {
            return Ok(None);
        }
        let tt = self.generate_tuple(n, true)?;
        let ty = self.instantiate_type_with(tt, &out)?;
        let w: WrapFn = Box::new(move |tv, ex| {
            let tmp = tv.get_temporary_var("tw");
            let l = tv.id_n(&tmp);
            let asg = tv.n_stmt(crate::ast::AssignStmt { lhs: Some(l), rhs: Some(ex), type_expr: None, update: Default::default() });
            let mut items = vec![];
            for (i, wf) in wraps.into_iter().enumerate() {
                let v = tv.id_n(&tmp);
                let idx = tv.n_expr(IntExpr::new(i as i64));
                let it = tv.n_expr(IndexExpr { expr: Some(v), index: Some(idx) });
                items.push(match wf {
                    Some(g) => g(tv, it)?,
                    None => it,
                });
            }
            let t = tv.n_expr(crate::ast::TupleExpr { items });
            Ok(tv.n_expr(StmtExpr { items: vec![asg], expr: Some(t) }))
        });
        Ok(Some((true, Some(ty), Some(w))))
    }

    /// `canWrapExpr(exprType, expectedType, callee, allowUnwrap, isEllipsis)`.
    pub fn can_wrap_expr(
        &mut self,
        expr_type: TypeId,
        expected: TypeId,
        callee: Option<TypeId>,
        allow_unwrap: bool,
        is_ellipsis: bool,
    ) -> Result<CanWrap> {
        let types = self.types();
        let st = stdlib();
        let ec = types.get_class(expected);
        let mut expr_type = expr_type;
        let mut xc = types.get_class(expr_type);
        let no_wrap = mangle::func("std.internal.attributes", "no_argument_wrap");
        if let Some(c) = callee {
            let ast = types.func_ast_id(c);
            if self.cache.ast.has_function_attribute(ast, &no_wrap) {
                return Ok((true, Some(expected), None));
            }
        }
        let is = |t: Option<TypeId>, n: &str| t.is_some_and(|t| types.is(t, n));
        let mut ty: Option<TypeId> = None;
        let mut f: Option<WrapFn> = None;

        // Case: types are wrapped in TypeWrap when type is not expected
        if callee.is_some() && types.is(expr_type, &st.type_) {
            let Some(c) = self.extract_class_type(expr_type) else { return Ok((false, None, None)) };
            if is(ec, &st.generator) && self.find_best_method_types(c, "__iter__", &[])?.is_some() {
                let rn = types.realized_name(c);
                let i = self.id_n(&rn);
                let d = self.dot_n(i, "__iter__");
                let call = self.call_n(d, vec![]);
                let it = self.tr(call)?;
                let w: WrapFn = Box::new(|tv, e| {
                    let d = tv.dot_n(e, "__iter__");
                    Ok(tv.call_n(d, vec![]))
                });
                return Ok((true, Some(self.ety(it)), Some(w)));
            }
            if !types.is(expected, &st.type_) {
                let tw = self.get_stdlib_type(&st.type_wrap)?;
                ty = Some(self.instantiate_type_with(tw, &[c])?);
                f = Some(Box::new(|tv, e| {
                    let i = tv.id_n(&stdlib().type_wrap);
                    Ok(tv.call_n(i, vec![e]))
                }));
            }
            return Ok((true, ty, f));
        }
        if types.static_kind(expected) == LiteralKind::Runtime && types.static_kind(expr_type) != LiteralKind::Runtime {
            expr_type = self.get_underlying_static_type(expr_type)?;
            xc = types.get_class(expr_type);
            ty = Some(expr_type);
        }
        if xc.is_none() && ec.is_some_and(|e| {
            let n = types.class_name(e).unwrap();
            n == st.generator.as_str() || n == "float" || n == st.optional.as_str()
        }) {
            return Ok((false, None, None)); // argument type not yet known.
        }
        let ecn = ec.map(|e| types.class_name(e).unwrap().to_string()).unwrap_or_default();

        if allow_unwrap && is(ec, &st.capsule) && xc.is_some() && !is(xc, &st.capsule) {
            let cap = self.get_stdlib_type(&st.capsule)?;
            ty = Some(self.instantiate_type_with(cap, &[xc.unwrap()])?);
            f = Some(Box::new(|tv, e| {
                let get = mangle::mangled_method("", "Capsule", "_get", 0, 0);
                if let Some(c) = tv.cache.ast[e].kind.as_call().cloned() {
                    if tv.is_function_expr(c.expr, &get) {
                        return Ok(c.items[0].value.unwrap()); // Do not wrap already wrapped vars
                    }
                }
                let m = tv.id_n(&mangle::mangled_method("", "Capsule", "_make", 0, 0));
                Ok(tv.call_n(m, vec![e]))
            }));
        } else if ec.is_some() && !is(ec, "Any") && is(xc, "Any") {
            ty = ec;
            let t = ec.unwrap();
            f = Some(Box::new(move |tv, e| {
                let r = tv.realize(t)?.expect("not realizable");
                let rn = tv.cache.types.realized_name(r);
                let u = tv.id_n(&fn_optional_unwrap());
                let ti = tv.id_n(&rn);
                Ok(tv.call_n(u, vec![e, ti]))
            }));
        } else if is(ec, "Any") && xc.is_some() && !is(xc, "Any") {
            ty = ec;
            f = Some(Box::new(|tv, e| {
                let a = tv.id_n("Any");
                Ok(tv.call_n(a, vec![e]))
            }));
        } else if is(ec, "Generator") && !is(xc, &ecn) && !is_ellipsis && !(allow_unwrap && is(xc, &st.optional)) {
            // (Optional[Generator[T]] is unwrapped below, not iterated via Optional.__iter__.)
            let x = xc.unwrap();
            let Some(m) = self.find_best_method_types(x, "__iter__", &[x])? else { return Ok((false, None, None)) };
            let it = self.instantiate_type(m, Some(x))?;
            ty = Some(types.func_ret(it));
            f = Some(Box::new(|tv, e| {
                let d = tv.dot_n(e, "__iter__");
                Ok(tv.call_n(d, vec![]))
            }));
        } else if let Some(w) = self.can_wrap_tuple(xc, ec, callee, allow_unwrap)? {
            return Ok(w);
        } else if is(ec, "float") && is(xc, "Int") {
            ty = Some(self.instantiate_type(ec.unwrap(), None)?);
            f = Some(Box::new(|tv, e| {
                let fl = tv.id_n("float");
                Ok(tv.call_n(fl, vec![e]))
            }));
        } else if is(ec, &st.complex) && (is(xc, "Int") || is(xc, "float")) && callee.is_some_and(|c| {
            let n = if types.get_func(c).is_some() { types.func_name(c) } else { String::new() };
            (n.contains(".List.") && n.contains(".append:"))
                || (n.contains(".Set.") && n.contains(".add:"))
                || (n.contains(".Dict.") && n.contains(".__setitem__:"))
        }) {
            // Collection literals (`[1., 2j]`): numeric tower int/float -> complex.
            ty = Some(self.instantiate_type(ec.unwrap(), None)?);
            f = Some(Box::new(|tv, e| {
                let c = tv.id_n("complex");
                Ok(tv.call_n(c, vec![e]))
            }));
        } else if is(ec, &st.list) && is(xc, &st.list) && ec.is_some_and(|e| types.can_realize(e))
            && xc.is_some_and(|x| types.can_realize(x))
            && types.realized_name(ec.unwrap()) != types.realized_name(xc.unwrap())
            && self.list_promotable(xc.unwrap(), ec.unwrap())
            && callee.is_some_and(|c| {
                let n = if types.get_func(c).is_some() { types.func_name(c) } else { String::new() };
                n.contains(".List.") && n.contains(".append:")
            })
        {
            // Nested list literal promotion: `[[1, 2], [3.5, 4]]`.
            let t = ec.unwrap();
            ty = Some(t);
            let inner = types.realized_name(self.extract_class_generic(t, 0));
            f = Some(Box::new(move |tv, e| {
                let l = tv.id_n(&stdlib().list);
                let d = tv.dot_n(l, "_promote");
                let ti = tv.id_n(&inner);
                Ok(tv.call_n(d, vec![e, ti]))
            }));
        } else if callee.is_none() && is(ec, &st.bool_) && xc.is_some() && !is(xc, &st.bool_) {
            // Do not do this in function calls---only use for if-else wrapping
            ty = Some(self.instantiate_type(ec.unwrap(), None)?);
            f = Some(Box::new(|tv, e| {
                let d = tv.dot_n(e, "__bool__");
                Ok(tv.call_n(d, vec![]))
            }));
        } else if is(ec, &st.optional) && xc.is_some() && !is(xc, &ecn) {
            // Converting NoneType to Optional[T] constructs an absent value.
            if is(xc, &st.none_type) {
                let t = ec.unwrap();
                let payload = self.extract_class_generic(t, 0);
                let w: WrapFn = Box::new(move |tv, e| {
                    let c = tv.id_n(&stdlib().optional);
                    let ct = tv.instantiate_type_var(t)?;
                    tv.set_type(c, ct);
                    let m = tv.id_n("T");
                    let mt = tv.instantiate_type_var(payload)?;
                    tv.set_type(m, mt);
                    tv.set_done_expr(m);
                    let d = tv.dot_n(c, "_from_none");
                    Ok(tv.call_n(d, vec![e, m]))
                });
                return Ok((true, Some(t), Some(w)));
            }
            let eu = self.extract_class_generic(ec.unwrap(), 0);
            let (_, tp, fnp) = self.can_wrap_expr(xc.unwrap(), eu, callee, allow_unwrap, is_ellipsis)?;
            let o = self.get_stdlib_type(&st.optional)?;
            ty = Some(self.instantiate_type_with(o, &[tp.unwrap_or(xc.unwrap())])?);
            f = Some(Box::new(move |tv, e| {
                let x = match fnp {
                    Some(g) => g(tv, e)?,
                    None => e,
                };
                let o = tv.id_n(&stdlib().optional);
                Ok(tv.call_n(o, vec![x]))
            }));
        } else if allow_unwrap
            && ec.is_some()
            && is(xc, &st.optional)
            && !is(xc, &ecn)
            && !(types.get_union(ec.unwrap()).is_some()
                && types.can_realize(ec.unwrap())
                && types.union_realization_types(ec.unwrap()).iter().any(|m| types.is(*m, &st.optional)))
        {
            // unwrap optional
            let xu = self.extract_class_generic(xc.unwrap(), 0);
            let (_, tp, fnp) = self.can_wrap_expr(xu, ec.unwrap(), callee, allow_unwrap, is_ellipsis)?;
            ty = Some(self.instantiate_type(tp.unwrap_or(xu), None)?);
            f = Some(Box::new(move |tv, e| {
                let x = match fnp {
                    Some(g) => g(tv, e)?,
                    None => e,
                };
                let u = tv.id_n(&fn_optional_unwrap());
                Ok(tv.call_n(u, vec![x]))
            }));
        } else if is(ec, &st.callable)
            && xc.is_some_and(|x| types.get_partial(x).is_some() || types.get_func(x).is_some() || types.is(x, &st.function))
        {
            return self.can_wrap_callable(xc.unwrap(), ec.unwrap(), expected);
        } else if callee.is_some() && xc.is_some() && types.get_func(expr_type).is_some() && !is(ec, &st.function) {
            // Wrap raw Seq functions into Partial(...) call for easy realization.
            if let Some(e) = ec {
                ty = Some(self.instantiate_type(e, None)?);
            }
            let fname = types.func_name(expr_type);
            f = Some(Box::new(move |tv, e| {
                let i = tv.id_n(&fname);
                let el = tv.n_expr(EllipsisExpr { mode: EllipsisType::Partial });
                let p = tv.call_n(i, vec![el]);
                if let Some(se) = tv.cache.ast[e].kind.as_stmt().cloned() {
                    return Ok(tv.n_expr(StmtExpr { items: se.items, expr: Some(p) }));
                }
                Ok(p)
            }));
        } else if is(ec, &st.function) && xc.is_some_and(|x| types.get_partial(x).is_some_and(|p| types.is_partial_empty(p))) {
            ty = Some(self.instantiate_type(ec.unwrap(), None)?);
            let pf = types.partial_func(types.get_partial(xc.unwrap()).unwrap());
            let fname = types.func_name(pf);
            f = Some(Box::new(move |tv, _e| Ok(tv.id_n(&fname))));
        } else if allow_unwrap && xc.is_some() && types.get_union(expr_type).is_some() && ec.is_some_and(|e| types.get_union(e).is_none()) {
            // Extract union types via Union._get
            let Some(t) = self.realize(ec.unwrap())? else { return Ok((false, None, None)) };
            let Some(e) = self.realize(expr_type)? else { return Ok((false, None, None)) };
            let mut ok = false;
            for ut in types.union_realization_types(e) {
                if types.unify(ut, t, None, self)? >= 0
                    || (types.is(ut, &st.optional)
                        && (types.is(t, &st.none_type) || types.unify(self.extract_class_generic(ut, 0), t, None, self)? >= 0))
                {
                    ok = true;
                    break;
                }
            }
            if ok {
                ty = Some(t);
                let rn = types.realized_name(t);
                f = Some(Box::new(move |tv, e| {
                    let g = tv.id_n(&mangle::mangled_method("", "Union", "_get", 0, 0));
                    let ti = tv.id_n(&rn);
                    Ok(tv.call_n(g, vec![e, ti]))
                }));
            }
        } else if xc.is_some() && ec.is_some_and(|e| types.get_union(e).is_some()) {
            // Make union types via Union._new
            let Some(t) = self.realize(ec.unwrap())? else { return Ok((false, None, None)) };
            if types.unify(ec.unwrap(), xc.unwrap(), None, self)? == -1 {
                ty = Some(t);
                let rn = types.realized_name(t);
                f = Some(Box::new(move |tv, e| {
                    let u = tv.id_n(&stdlib().union);
                    let d = tv.dot_n(u, "_new");
                    let ti = tv.id_n(&rn);
                    Ok(tv.call_n(d, vec![e, ti]))
                }));
            }
        } else if is(xc, &st.type_) && is(ec, &st.type_wrap) {
            let tw = self.get_stdlib_type(&st.type_wrap)?;
            ty = Some(self.instantiate_type_with(tw, &[xc.unwrap()])?);
            f = Some(Box::new(|tv, e| {
                let i = tv.id_n(&stdlib().type_wrap);
                Ok(tv.call_n(i, vec![e]))
            }));
        } else if let (Some(x), Some(e)) = (xc, ec) {
            let (mut src, mut dest) = (Some(x), Some(e));
            let optional = types.is(x, &st.optional) && types.is(e, &st.optional);
            if optional {
                src = types.get_class(self.extract_class_generic(x, 0));
                dest = types.get_class(self.extract_class_generic(e, 0));
            }
            if let (Some(src), Some(dest)) = (src, dest) {
                let dn = types.class_name(dest).unwrap();
                if !types.is(src, &dn) {
                    // Cast derived classes to base classes
                    let mros = self.cache.get_class_by_type(src).map(|c| c.mro.clone()).unwrap_or_default();
                    for m in mros.iter().skip(1) {
                        let base = self.instantiate_type(*m, Some(src))?;
                        if types.unify(base, dest, None, self)? >= 0 {
                            self.unify(base, dest)?;
                            ty = Some(e);
                            f = Some(Box::new(move |tv, ex| {
                                let types = tv.types();
                                let bc = types.get_class(base).unwrap();
                                let mut te = tv.id_n(&types.class_name(bc).unwrap());
                                let tvv = tv.instantiate_type_var(bc)?;
                                tv.set_type(te, tvv);
                                if optional {
                                    let o = tv.id_n(&stdlib().optional);
                                    te = tv.n_expr(InstantiateExpr { expr: Some(o), items: vec![te] });
                                }
                                let fc = tv.id_n(&mangle::func("std.internal.types.rtti", "force_cast"));
                                let c = tv.call_n(fc, vec![ex, te]);
                                tv.tr(c)
                            }));
                            break;
                        }
                    }
                }
            }
        }
        Ok((true, ty, f))
    }

    /// `Callable[...]` branch of `canWrapExpr` (builds a `.proxy.*` wrapper function).
    fn can_wrap_callable(&mut self, xc: TypeId, ec: TypeId, expected: TypeId) -> Result<CanWrap> {
        let types = self.types();
        let ea = self.extract_class_generic(ec, 0);
        let arg_types: Vec<TypeId> = types.generics(ea).iter().map(|g| g.ty()).collect();
        let call_args: Vec<CallArg> = arg_types
            .iter()
            .map(|t| {
                let v = self.n_expr(crate::ast::NoneExpr);
                self.set_type(v, *t);
                CallArg::new("", Some(v))
            })
            .collect();
        let er = self.extract_class_generic(ec, 1);
        if self.match_callable(xc, &call_args, Some(er), None, None, None)? < 0 {
            return Ok((false, None, None));
        }
        let call_args = Rc::new(call_args);
        let w: WrapFn = Box::new(move |tv, expr| {
            let types = tv.types();
            let st = stdlib();
            let xc = tv.expr_class(expr).unwrap();
            let ec = types.get_class(expected).unwrap();
            let ea = tv.extract_class_generic(ec, 0);
            let er = tv.extract_class_generic(ec, 1);
            let mut changes = crate::types::Unification::new();
            if tv.match_callable(xc, &call_args, Some(er), Some(&mut changes), None, None)? < 0 {
                let src = tv.cache.ast[expr].src;
                typhon_common::bail_e!(ErrorKind::TYPE_UNIFY, src, types.pretty_string(xc), types.pretty_string(ec));
            }
            let cobj = st.cobj.clone();
            let (mut fname, ret_fn, data_arg) = if types.get_partial(xc).is_some() {
                let rf = tv.realize(xc)?.expect("not realizable");
                let rn = types.realized_name(rf);
                let p = tv.id_n(&st.ptr);
                let r = tv.id_n(&rn);
                let ix = tv.n_expr(IndexExpr { expr: Some(p), index: Some(r) });
                let d = tv.id_n("data");
                let c = tv.call_n(ix, vec![d]);
                let z = tv.n_expr(IntExpr::new(0));
                (rn, tv.n_expr(IndexExpr { expr: Some(c), index: Some(z) }), None)
            } else if types.get_func(xc).is_some() {
                let fname = types.func_name(xc);
                let r = tv.id_n(&fname);
                let co = tv.id_n(&cobj);
                let da = tv.call_n(co, vec![]);
                (fname, r, Some(da))
            } else {
                let rf = tv.realize(xc)?.expect("not realizable");
                let rn = types.realized_name(rf);
                let r = tv.id_n(&rn);
                let d = tv.id_n("data");
                (rn.clone(), tv.call_n(r, vec![d]), None)
            };
            // One stored template can adapt to several input/return signatures.
            fname = format!(".proxy.{}.{}", fname, types.realized_name(ec));
            if tv.find(&fname, 0).is_none() {
                let dt = tv.id_n(&cobj);
                let at = tv.id_n(&types.realized_name(ea));
                let params = vec![
                    Param::with_status("data", Some(dt), None, ParamStatus::Value),
                    Param::with_status("args", Some(at), None, ParamStatus::Value),
                ];
                let a = tv.id_n("args");
                let sa = tv.n_expr(StarExpr { expr: Some(a) });
                let call = tv.call_n(ret_fn, vec![sa]);
                let rs = tv.n_stmt(ReturnStmt { expr: Some(call) });
                let suite = tv.n_stmt(SuiteStmt { items: vec![rs] });
                let src = tv.src;
                let fs = tv.cache.ast.new_function(src, fname.clone(), None, params, Some(suite), vec![], false)?;
                tv.tr_stmt(fs)?;
            }
            let c = tv.id_n(&st.callable);
            let fi = tv.id_n(&fname);
            Ok(tv.call_n(c, vec![fi, data_arg.unwrap_or(expr)]))
        });
        let _ = FunctionStmt::default;
        Ok((true, Some(expected), Some(w)))
    }

    /// `unpackTupleTypes(expr)`: `(name, type)` pairs of a Tuple/NamedTuple expression.
    pub fn unpack_tuple_types(&mut self, expr: ExprId) -> Result<Option<Vec<(String, TypeId)>>> {
        let types = self.types();
        let orig = self.cache.ast[expr].orig_expr.unwrap_or(expr);
        let mut ret = vec![];
        if let Some(tup) = self.cache.ast[orig].kind.as_tuple().cloned() {
            let mut items = vec![];
            for a in tup.items {
                let a = self.tr(a)?;
                items.push(a);
                if self.expr_class(a).is_none() {
                    return Ok(None);
                }
                ret.push((String::new(), self.ety(a)));
            }
            self.cache.ast[orig].kind.as_tuple_mut().unwrap().items = items;
        } else if self.cache.ast[orig].kind.as_call().is_some() {
            let Some(val) = self.extract_class_type(self.ety(expr)) else { return Ok(None) };
            if !types.is(val, &stdlib().named_tuple)
                || types.get_class(self.extract_class_generic(val, 1)).is_none()
                || !types.can_realize(self.extract_class_generic(val, 0))
            {
                return Ok(None);
            }
            let id = self.get_int_literal(val, 0) as usize;
            let names = types.generated_tuple_names.borrow()[id].clone();
            let tys = types.generics(self.extract_class_generic(val, 1));
            for (i, g) in tys.iter().enumerate() {
                ret.push((names[i].clone(), g.ty()));
            }
        } else {
            return Ok(None);
        }
        Ok(Some(ret))
    }

    /// `extractNamedTuple(expr)`: `(name, expr.args[i])` pairs.
    pub fn extract_named_tuple(&mut self, expr: ExprId) -> Vec<(String, ExprId)> {
        let c = self.expr_class(expr).unwrap();
        let id = self.get_int_literal(c, 0) as usize;
        let names = self.cache.types.generated_tuple_names.borrow()[id].clone();
        let mut ret = vec![];
        for (i, n) in names.into_iter().enumerate() {
            let d = self.dot_n(expr, "args");
            let ix = self.n_expr(IntExpr::new(i as i64));
            ret.push((n, self.n_expr(IndexExpr { expr: Some(d), index: Some(ix) })));
        }
        ret
    }
}
