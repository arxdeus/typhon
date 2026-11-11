//! Union/super dispatch generators of special.cpp (`generateUnionIteratorAST`,
//! `generateUnionBinaryAST`, `generateUnionDispatchAST`, `generateSuperDispatchAST`).

use std::collections::BTreeMap;

use typhon_common::mangle::stdlib;
use typhon_common::{ErrorKind, Result};

use super::super::{TypeContext, TypecheckVisitor};
use super::mm;
use crate::ast::{Attr, CallArg, InstantiateExpr, KeywordStarExpr, StarExpr};
use crate::ids::{ExprId, StmtId, TypeId};

/// State of the recursive `dispatch` lambda of `generateUnionDispatchAST`.
struct DispatchState {
    /// The dispatch function type.
    ty: TypeId,
    /// Active union member (`t`).
    member: TypeId,
    /// `receiver` expression (`Union._get_data(union, T)`).
    receiver: ExprId,
    arguments: Vec<(String, TypeId)>,
    preserve_unions: Vec<bool>,
    selected: Vec<Option<TypeId>>,
    candidates: Vec<(Option<ExprId>, ExprId)>,
    union_types: BTreeMap<String, TypeId>,
}

impl<'a> TypecheckVisitor<'a> {
    /// `ctx->addBlock()`.
    pub(super) fn sp_add_block(&mut self) {
        self.ctx.borrow_mut().add_block();
    }
    /// `ctx->popBlock()`.
    pub(super) fn sp_pop_block(&mut self) {
        self.ctx.borrow_mut().pop_block();
    }
    /// `ctx->addVar(name, name, type)`.
    pub(super) fn sp_add_var(&mut self, name: &str, t: TypeId) {
        let ctx = self.ctx.clone();
        TypeContext::add_var(self.cache, &ctx, name, name, t, 0, Default::default());
    }
    /// `N<InstantiateExpr>(N<IdExpr>(head), members)` over a sorted name map.
    fn sp_union_of(&mut self, head: &str, names: &BTreeMap<String, TypeId>) -> ExprId {
        let members: Vec<ExprId> = names.keys().cloned().collect::<Vec<_>>().iter().map(|n| self.sp_id(n)).collect();
        let h = self.sp_id(head);
        self.n_expr(InstantiateExpr { expr: Some(h), items: members })
    }
    /// `Union._get_tag(<name>) == tag`.
    fn sp_tag_eq(&mut self, value: ExprId, tag: usize) -> ExprId {
        let f = self.sp_id(&mm("Union", "_get_tag"));
        let c = self.sp_call(f, vec![value]);
        let i = self.sp_int(tag as i64);
        self.sp_bin(c, "==", i)
    }
    /// `Union._get_data(value, T)`.
    fn sp_get_data(&mut self, value: ExprId, t: TypeId) -> ExprId {
        let f = self.sp_id(&mm("Union", "_get_data"));
        let rn = self.sp_rn(t);
        let te = self.sp_id(&rn);
        self.sp_call(f, vec![value, te])
    }
    /// `clone(type->ast->getSuite())`.
    fn sp_clone_fn_suite(&mut self, f: TypeId) -> Option<StmtId> {
        let aid = self.cache.types.func_ast_id(f);
        let s = self.cache.ast[aid].kind.as_function().and_then(|f| f.suite);
        self.cache.ast.clone_stmt_opt(s, false)
    }
    /// `clone(getFunction(type->getFuncName())->ast->getSuite())`.
    fn sp_clone_generic_suite(&mut self, f: TypeId) -> StmtId {
        let n = self.cache.types.func_name(f);
        let aid = self.cache.functions.get(&n).and_then(|f| f.ast).expect("bad function");
        let s = self.cache.ast[aid].kind.as_function().and_then(|f| f.suite);
        let s = self.cache.ast.clone_stmt_opt(s, false);
        self.cache.ast.wrap_suite(s).expect("function without a suite")
    }

    /// `generateUnionIteratorAST(type)`: eager iterator acquisition for the active union
    /// alternative. Join yielded element types rather than returning a union of generator
    /// objects, so ordinary loops can consume heterogeneous iterable alternatives.
    /// This dispatcher must not itself yield: iter(value) must run the active __iter__
    /// immediately, including its side effects and exceptions, rather than at next().
    pub fn generate_union_iterator_ast(&mut self, f: TypeId) -> Result<Option<StmtId>> {
        let types = self.types();
        let u = types.get_union(self.extract_func_arg_type(f, 0)).expect("not a union");
        let alternatives = types.union_realization_types(u);
        let mut elements: BTreeMap<String, TypeId> = BTreeMap::new();
        let mut iterable_alternatives = vec![false; alternatives.len()];
        for (tag, &alternative) in alternatives.iter().enumerate() {
            self.sp_add_block();
            let name = self.get_temporary_var("iterable");
            self.sp_add_var(&name, alternative);
            let value = self.sp_id(&name);
            self.set_type(value, alternative);
            // Union[list[int], int] remains iterable when its list alternative is active.
            // A missing method leaves this tag to the runtime TypeError fallback below.
            if !self.get_attr(value, "__iter__", None, None)?.1 {
                self.sp_pop_block();
                continue;
            }
            let n = self.sp_id(&name);
            let d = self.sp_dot(n, "__iter__");
            let c = self.sp_call(d, vec![]);
            let iterator = self.tr(c)?;
            let iterator_type = self.ety(iterator);
            if !types.is(iterator_type, &stdlib().generator) {
                typhon_common::bail_e!(ErrorKind::EXPECTED_TYPE, self.src, "iterable");
            }
            let element = self.extract_class_generic(iterator_type, 0);
            elements.insert(types.realized_name(element), element);
            iterable_alternatives[tag] = true;
            self.sp_pop_block();
        }
        let mut element_type = None;
        // Distinct containers with the same element type need no per-element union box.
        if elements.len() > 1 {
            element_type = Some(self.sp_union_of(&stdlib().union, &elements));
        }
        let suite = self.sp_suite(vec![]);
        for (tag, &alt) in alternatives.iter().enumerate() {
            if !iterable_alternatives[tag] {
                continue;
            }
            let un = self.sp_id("union");
            let iterable = self.sp_get_data(un, alt);
            let d = self.sp_dot(iterable, "__iter__");
            let mut iterator = self.sp_call(d, vec![]);
            if let Some(et) = element_type {
                let f = self.sp_id(&mm("Union", "_iter_union"));
                let c = self.cache.ast.clone_expr(et, false);
                iterator = self.sp_call(f, vec![iterator, c]);
            }
            let un = self.sp_id("union");
            let cond = self.sp_tag_eq(un, tag);
            let r = self.sp_return(Some(iterator));
            let s = self.sp_if(Some(cond), r);
            self.cache.ast.suite_add_stmt(suite, Some(s));
        }
        let orig = self.sp_clone_fn_suite(f);
        self.cache.ast.suite_add_stmt(suite, orig);
        Ok(Some(suite))
    }

    /// `generateUnionBinaryAST(type)`: binary operator dispatch for the active pair of
    /// operand types. Either operand may be concrete or a union; retain native results such
    /// as boolean ndarrays, joining result types only when supported pairs return different
    /// types.
    pub fn generate_union_binary_ast(&mut self, f: TypeId) -> Result<Option<StmtId>> {
        let types = self.types();
        let Some(operation) = types.get_str_static(self.extract_func_generic(f, 0)) else { return Ok(None) };
        let left = self.extract_func_arg_type(f, 0);
        let right = self.extract_func_arg_type(f, 1);
        let left_union = types.get_union(left);
        let right_union = types.get_union(right);
        let left_types = left_union.map(|u| types.union_realization_types(u)).unwrap_or_else(|| vec![left]);
        let right_types = right_union.map(|u| types.union_realization_types(u)).unwrap_or_else(|| vec![right]);
        let mut candidates: Vec<(Option<ExprId>, ExprId)> = vec![];
        let mut results: BTreeMap<String, TypeId> = BTreeMap::new();
        for (left_tag, &lt) in left_types.iter().enumerate() {
            for (right_tag, &rt) in right_types.iter().enumerate() {
                self.sp_add_block();
                let left_name = self.get_temporary_var("left");
                let right_name = self.get_temporary_var("right");
                self.sp_add_var(&left_name, lt);
                self.sp_add_var(&right_name, rt);
                // Probe magic-method availability without rejecting the whole union when a
                // pair is unsupported. Such pairs use the runtime TypeError fallback below.
                // Ordinary operator resolution also preserves reflected methods such as __rsub__.
                let l = self.sp_id(&left_name);
                let l = self.tr(l)?;
                let r = self.sp_id(&right_name);
                let r = self.tr(r)?;
                let b = self.sp_bin(l, &operation, r);
                let Some(compared) = self.transform_binary_magic(b)? else {
                    self.sp_pop_block();
                    continue;
                };
                let result = self.ety(compared);
                typhon_common::seqassert!(types.can_realize(result), "cannot realize {}", types.debug_string(result, 2));
                results.insert(types.realized_name(result), result);
                self.sp_pop_block();

                let mut left_value = self.sp_id("left");
                let mut right_value = self.sp_id("right");
                let mut condition = None;
                if left_union.is_some() {
                    left_value = self.sp_get_data(left_value, lt);
                    let l = self.sp_id("left");
                    condition = Some(self.sp_tag_eq(l, left_tag));
                }
                if right_union.is_some() {
                    right_value = self.sp_get_data(right_value, rt);
                    let r = self.sp_id("right");
                    let selected = self.sp_tag_eq(r, right_tag);
                    condition = Some(match condition {
                        Some(c) => self.sp_bin(c, "&&", selected),
                        None => selected,
                    });
                }
                let v = self.sp_bin(left_value, &operation, right_value);
                candidates.push((condition, v));
            }
        }
        // Keep a shared native result type, such as ndarray[bool] for array comparisons.
        // Arithmetic can return different dtypes across branches, requiring a result union.
        let mut result_type = None;
        if results.len() > 1 {
            result_type = Some(self.sp_union_of(&stdlib().union, &results));
        }
        let suite = self.sp_suite(vec![]);
        for (condition, mut value) in candidates {
            if let Some(rt) = result_type {
                let f = self.sp_id(&mm("Union", "_new"));
                let c = self.cache.ast.clone_expr(rt, false);
                value = self.sp_call(f, vec![value, c]);
            }
            let r = self.sp_return(Some(value));
            let s = self.sp_if(condition, r);
            self.cache.ast.suite_add_stmt(suite, Some(s));
        }
        // The original generated-method stub reports unsupported active operand pairs.
        let orig = self.sp_clone_fn_suite(f);
        self.cache.ast.suite_add_stmt(suite, orig);
        Ok(Some(suite))
    }
}

impl<'a> TypecheckVisitor<'a> {
    /// `argumentValue(index, selected, args, kwargs)`: read saved argument packs so
    /// dispatch never repeats source argument effects. `None` keeps the union intact for
    /// tag tests; the original type keeps it intact for union-typed parameters. Only a
    /// concrete member denotes a payload extraction.
    fn sp_argument_value(&mut self, st: &DispatchState, index: usize, selected: Option<TypeId>, args: &str, kwargs: &str) -> ExprId {
        let (name, original) = st.arguments[index].clone();
        let mut value = if name.is_empty() {
            let a = self.sp_id(args);
            let i = self.sp_int(index as i64);
            self.sp_index(a, i)
        } else {
            let k = self.sp_id(kwargs);
            let s = self.sp_str(&name);
            self.sp_index(k, s)
        };
        if let Some(sel) = selected
            && sel != original
            && self.cache.types.get_union(original).is_some()
        {
            value = self.sp_get_data(value, sel);
        }
        value
    }

    /// Recursive `dispatch` lambda of `generateUnionDispatchAST`.
    fn sp_union_dispatch(&mut self, st: &mut DispatchState, index: usize, selected_condition: Option<ExprId>) -> Result<()> {
        let types = self.types();
        if index < st.arguments.len() {
            let original = st.arguments[index].1;
            match types.get_union(original) {
                Some(alternatives) if !st.preserve_unions[index] => {
                    for (argument_tag, member) in types.union_realization_types(alternatives).into_iter().enumerate() {
                        st.selected[index] = Some(member);
                        let v = self.sp_argument_value(st, index, None, "args", "kwargs");
                        let matches = self.sp_tag_eq(v, argument_tag);
                        let c = selected_condition.map(|c| self.cache.ast.clone_expr(c, false));
                        // PORT-NOTE: C++ builds `clone(nullptr) && matches`; a null condition
                        // never happens here because the union tag test is always present.
                        let cond = match c {
                            Some(c) => self.sp_bin(c, "&&", matches),
                            None => matches,
                        };
                        self.sp_union_dispatch(st, index + 1, Some(cond))?;
                    }
                }
                _ => {
                    st.selected[index] = Some(original);
                    self.sp_union_dispatch(st, index + 1, selected_condition)?;
                }
            }
            return Ok(());
        }
        // Preserve keyword names and omit unsupplied arguments so ordinary call
        // matching still handles overloads, defaults and bound receiver arguments.
        let mut call_args = vec![];
        for argument in 0..st.arguments.len() {
            let v = self.sp_none_of(st.selected[argument].unwrap());
            call_args.push(CallArg::new(st.arguments[argument].0.clone(), Some(v)));
        }
        // Skip incompatible signatures before realizing their bodies. An unsupported
        // active combination reaches the original runtime TypeError fallback instead.
        let t = st.member;
        if let Some(partial) = types.get_partial(t) {
            let generic = types.generalize(types.partial_func(partial), 0);
            let function = self.instantiate_type(generic, None)?;
            let fnc = types.get_func(function).unwrap();
            if self.can_call(fnc, &call_args, Some(partial), None, None, false)? < 0 {
                return Ok(());
            }
        } else {
            let cls = types.get_class(t);
            let methods = self.find_method(cls, "__call__")?;
            if self.find_matching_methods(cls, &methods, &call_args, None)?.is_empty() {
                return Ok(());
            }
        }
        self.sp_add_block();
        let var = self.get_temporary_var("union");
        self.sp_add_var(&var, t);
        let var_a = self.get_temporary_var("args");
        let a1 = self.extract_func_arg_type(st.ty, 1);
        self.sp_add_var(&var_a, a1);
        let var_k = self.get_temporary_var("kwargs");
        let a2 = self.extract_func_arg_type(st.ty, 2);
        self.sp_add_var(&var_k, a2);
        let mut call_args = vec![];
        for argument in 0..st.arguments.len() {
            let v = self.sp_argument_value(st, argument, st.selected[argument], &var_a, &var_k);
            call_args.push(CallArg::new(st.arguments[argument].0.clone(), Some(v)));
        }
        // Realize each viable call before joining branch results: the same generic
        // method may return int for one argument tag and str for another.
        let ve = self.sp_id(&var);
        let c = self.sp_call_args(ve, call_args);
        let c = self.tr(c)?;
        let result = self.ety(c);
        typhon_common::seqassert!(types.can_realize(result), "cannot realize {}", types.debug_string(result, 2));
        st.union_types.insert(types.realized_name(result), result);
        self.sp_pop_block();
        let mut call_args = vec![];
        for argument in 0..st.arguments.len() {
            let v = self.sp_argument_value(st, argument, st.selected[argument], "args", "kwargs");
            call_args.push(CallArg::new(st.arguments[argument].0.clone(), Some(v)));
        }
        let recv = self.cache.ast.clone_expr(st.receiver, false);
        let call = self.sp_call_args(recv, call_args);
        st.candidates.push((selected_condition, call));
        Ok(())
    }

    /// `generateUnionDispatchAST(type)`: attribute access or calls for a union
    /// receiver/callable. Method lookup can return a union of bound callables. Selecting that
    /// callable alone is insufficient for generic methods: their bodies also need the active
    /// types of union-valued positional and keyword arguments, rather than the unions
    /// themselves.
    pub fn generate_union_dispatch_ast(&mut self, f: TypeId) -> Result<Option<StmtId>> {
        let types = self.types();
        let st = stdlib();
        let Some(attr) = types.get_str_static(self.extract_func_generic(f, 0)) else { return Ok(None) };
        let a0 = self.extract_func_arg_type(f, 0);
        let union_typ = types.get_union(a0);
        typhon_common::seqassert!(union_typ.is_some(), "not an union: {}", types.debug_string(a0, 2));
        let union_typ = union_typ.unwrap();
        let suite = self.sp_clone_generic_suite(f);

        let a1 = self.extract_func_arg_type(f, 1);
        let a2 = self.extract_func_arg_type(f, 2);
        let is_call = types.is(a1, &st.tuple) && types.is(a2, &st.named_tuple);
        let mut arguments: Vec<(String, TypeId)> = vec![];
        if is_call {
            for g in types.generics(types.get_class(a1).unwrap()) {
                arguments.push((String::new(), g.ty()));
            }
            let id = self.get_int_literal(a2, 0);
            let names = self.sp_tuple_names(id);
            let keywords = types.get_class(self.extract_class_generic(a2, 1)).unwrap();
            for (index, g) in types.generics(keywords).into_iter().enumerate() {
                arguments.push((names[index].clone(), g.ty()));
            }
        }
        let mut state = DispatchState {
            ty: f,
            member: a0,
            receiver: a0_placeholder(self),
            arguments,
            preserve_unions: vec![],
            selected: vec![],
            candidates: vec![],
            union_types: BTreeMap::new(),
        };
        for (tag, t) in types.union_realization_types(union_typ).into_iter().enumerate() {
            let un = self.sp_id("union");
            let condition = self.sp_tag_eq(un, tag);
            let un = self.sp_id("union");
            let receiver = self.sp_get_data(un, t);
            if is_call {
                // Splitting N int|str arguments creates 2^N branches even if their declared
                // parameters already accept int|str. Probe the original argument types first
                // and preserve a union only when every viable signature accepts it intact;
                // otherwise its active member can affect overload selection or specialization.
                let mut signature_args = vec![];
                for (name, argument) in state.arguments.clone() {
                    let v = self.sp_none_of(argument);
                    signature_args.push(CallArg::new(name, Some(v)));
                }
                let mut preserve_unions = vec![true; state.arguments.len()];
                let mut matched = false;
                let mut inspect = |this: &mut Self, function: TypeId, partial: Option<TypeId>| -> Result<()> {
                    let mut accepted = vec![];
                    if this.can_call(function, &signature_args, partial, Some(&mut accepted), None, false)? < 0 {
                        return Ok(());
                    }
                    matched = true;
                    for (index, acc) in accepted.iter().enumerate() {
                        if index < preserve_unions.len() {
                            preserve_unions[index] = preserve_unions[index] && *acc;
                        }
                    }
                    Ok(())
                };
                if let Some(partial) = types.get_partial(t) {
                    let generic = types.generalize(types.partial_func(partial), 0);
                    let function = self.instantiate_type(generic, None)?;
                    inspect(self, types.get_func(function).unwrap(), Some(partial))?;
                } else {
                    let cls = types.get_class(t);
                    for method in self.find_method(cls, "__call__")? {
                        let m = self.instantiate_type(method, cls)?;
                        inspect(self, types.get_func(m).unwrap(), None)?;
                    }
                }
                if !matched {
                    preserve_unions.iter_mut().for_each(|p| *p = false);
                }
                state.member = t;
                state.receiver = receiver;
                state.preserve_unions = preserve_unions;
                state.selected = vec![None; state.arguments.len()];
                self.sp_union_dispatch(&mut state, 0, Some(condition))?;
                continue;
            }
            // Do full expression typechecking.
            self.sp_add_block();
            let var = self.get_temporary_var("union");
            self.sp_add_var(&var, t);
            let ei = self.sp_id(&var);
            self.set_type(ei, t);
            let (_, found) = self.get_attr(ei, &attr, None, None)?;
            let mut typ = None;
            if found {
                let d = self.sp_dot(ei, &attr);
                let expr = self.tr(d)?;
                typ = Some(self.ety(expr));
            }
            self.sp_pop_block();
            if let Some(typ) = typ {
                typhon_common::seqassert!(types.can_realize(typ), "cannot realize {}", types.debug_string(typ, 2));
                state.union_types.insert(types.realized_name(typ), typ);
                let d = self.sp_dot(receiver, &attr);
                state.candidates.push((Some(condition), d));
            }
        }
        let mut candidates = std::mem::take(&mut state.candidates);
        if state.union_types.len() > 1 {
            let w = self.sp_union_of(&typhon_common::mangle::class("", "Union"), &state.union_types);
            for (_, e) in candidates.iter_mut() {
                let f = self.sp_id(&mm("Union", "_new"));
                // PORT-NOTE: C++ shares one `w` node between candidates; arena nodes are
                // rewritten in place by the type checker, so each candidate gets a clone.
                let wc = self.cache.ast.clone_expr(w, false);
                *e = self.sp_call(f, vec![*e, wc]);
            }
        }
        for (condition, expr) in candidates {
            let r = self.sp_return(Some(expr));
            let s = self.sp_if(condition, r);
            self.cache.ast.suite_add_stmt(suite, Some(s));
        }
        // Move the final check to the end
        self.sp_move_front_to_back(suite);
        Ok(Some(suite))
    }

    /// `generateSuperDispatchAST(type)`.
    pub fn generate_super_dispatch_ast(&mut self, f: TypeId) -> Result<Option<StmtId>> {
        let types = self.types();
        let st = stdlib();
        let Some(attr) = types.get_str_static(self.extract_func_generic(f, 0)) else { return Ok(None) };
        let super_typ = types.get_class(self.extract_func_arg_type(f, 0)).unwrap();
        let typ = types.get_class(self.extract_class_generic(super_typ, 0)).unwrap();
        let _suite = self.sp_clone_generic_suite(f);

        let a1 = self.extract_func_arg_type(f, 1);
        let a2 = self.extract_func_arg_type(f, 2);
        typhon_common::seqassert!(types.is(a1, &st.tuple) && types.is(a2, &st.named_tuple), "invalid arguments");
        let mut call_args = vec![];
        let first = self.n_expr(crate::ast::NoneExpr);
        call_args.push(CallArg::new("", Some(first)));
        for g in types.generics(types.get_class(a1).unwrap()) {
            let v = self.sp_none_of(g.ty());
            call_args.push(CallArg::new("", Some(v)));
        }
        let id = self.get_int_literal(a2, 0);
        let names = self.sp_tuple_names(id);
        let kwt = types.get_class(self.extract_class_generic(a2, 1)).unwrap();
        for (gi, g) in types.generics(kwt).into_iter().enumerate() {
            let v = self.sp_none_of(g.ty());
            call_args.push(CallArg::new(names[gi].clone(), Some(v)));
        }

        let key = format!("super.{}.{}({},{})", types.realized_name(typ), attr, types.realized_name(a1), types.realized_name(a2));
        let n = self.cache.thunk_ids.len() + 1;
        let thunk_id = *self.cache.thunk_ids.entry(key).or_insert(n);
        self.cache.super_calls.insert(thunk_id, f);
        let mro = self.get_mro(typ)?;
        for &base in mro.iter().skip(1) {
            self.realize(base)?;
            self.set_type(first, base);
            let bc = types.get_class(base);
            let ms = self.find_method(bc, &attr)?;
            let methods = self.find_matching_methods(bc, &ms, &call_args, None)?;
            if methods.is_empty() {
                continue;
            }
            let brn = types.realized_name(base);
            let b = self.sp_id(&brn);
            let d = self.sp_dot(b, &attr);
            let fc = self.sp_id(&mm("type", "_force_cast"));
            let s = self.sp_id("self");
            let o = self.sp_dot(s, "_obj");
            let b2 = self.sp_id(&brn);
            let cast = self.sp_call(fc, vec![o, b2]);
            let a = self.sp_id("args");
            let sa = self.n_expr(StarExpr { expr: Some(a) });
            let k = self.sp_id("kwargs");
            let ka = self.n_expr(KeywordStarExpr { expr: Some(k) });
            let call = self.sp_call(d, vec![cast, sa, ka]);
            self.cache.ast[call].attrs.set_int(Attr::ExprVirtual, thunk_id as i64);
            let r = self.sp_return(Some(call));
            return Ok(Some(self.sp_suite(vec![r])));
        }
        typhon_common::bail_e!(ErrorKind::CUSTOM, self.src, "super has no matching method '{}'", attr);
    }
}

/// Placeholder receiver used before the first union member is processed.
fn a0_placeholder(tv: &mut TypecheckVisitor<'_>) -> ExprId {
    tv.n_expr(crate::ast::NoneExpr)
}
