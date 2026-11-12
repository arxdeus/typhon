
use std::cell::RefCell;
use std::collections::{BTreeSet, HashMap};

use typhon_common::mangle::{self, stdlib};
use typhon_common::{ErrorKind, Result};

use super::{CallArgSlots, PartialCallData, TypecheckVisitor};
use crate::ast::{
    AssignExpr, AssignStmt, Attr, CallArg, CallExpr, DotExpr, EllipsisExpr, EllipsisType, ExprStmt, GeneratorKind, IdExpr,
    IfStmt, IntExpr, KeywordStarExpr, StarExpr, StmtExpr, StringExpr, TupleExpr, UpdateMode,
};
use crate::cache::{FN_DISPATCH_SUFFIX, fn_optional_unwrap};
use crate::ids::{ExprId, StmtId, TypeId};
use crate::types::class::partial_flag;

thread_local! {
    /// `(callee, call)` pairs of calls whose callee is being transformed.
    /// Note: the reference implementation reads `ctx->getParentNode()` in `visit(DotExpr*)`; the Rust node stack
    /// only stores source locations, so `visit_call` registers the parent call here.
    static PARENT_CALLS: RefCell<Vec<(ExprId, ExprId)>> = const { RefCell::new(Vec::new()) };
}

/// The call whose callee `e` is currently being transformed (`ctx->getParentNode()`).
pub(crate) fn parent_call_of(e: ExprId) -> Option<ExprId> {
    PARENT_CALLS.with(|p| p.borrow().iter().rev().find(|(c, _)| *c == e).map(|(_, call)| *call))
}

impl<'a> TypecheckVisitor<'a> {
    fn call_items(&self, e: ExprId) -> Vec<CallArg> {
        self.ast()[e].kind.as_call().unwrap().items.clone()
    }
    fn set_call_items(&mut self, e: ExprId, items: Vec<CallArg>) {
        self.ast_mut()[e].kind.as_call_mut().unwrap().items = items;
    }
    fn callee(&self, e: ExprId) -> ExprId {
        self.ast()[e].kind.as_call().unwrap().expr.unwrap()
    }
    fn set_callee(&mut self, e: ExprId, c: ExprId) {
        self.ast_mut()[e].kind.as_call_mut().unwrap().expr = Some(c);
    }
    fn call_args_n(&mut self, f: ExprId, items: Vec<CallArg>) -> ExprId {
        self.n_expr(CallExpr { expr: Some(f), items, ordered: false, partial: false })
    }
    fn ellipsis_n(&mut self) -> ExprId {
        self.n_expr(EllipsisExpr { mode: EllipsisType::Partial })
    }

    /// `visit(PrintStmt*)`: `print a, b` -> `print(a, b)`; `print a, b,` -> `print(a, b, end=' ')`.
    pub fn visit_print(&mut self, s: StmtId) -> Result<()> {
        let p = self.ast()[s].kind.as_print().unwrap().clone();
        let mut args: Vec<CallArg> = p.items.iter().map(|i| self.ast().call_arg("", Some(*i))).collect();
        if p.no_newline {
            let e = self.n_expr(StringExpr::new(" ", ""));
            args.push(self.ast().call_arg("end", Some(e)));
        }
        let f = self.id_n("print");
        let c = self.call_args_n(f, args);
        let es = self.n_stmt(ExprStmt { expr: Some(c) });
        self.result_stmt = self.transform_stmt_opt(Some(es))?;
        Ok(())
    }

    /// `visit(StarExpr*)`: must be handled by CallExpr.
    pub fn visit_star(&mut self, e: ExprId) -> Result<()> {
        let src = self.ast()[e].src;
        typhon_common::bail_e!(ErrorKind::UNEXPECTED_TYPE, src, "star");
    }
    /// `visit(KeywordStarExpr*)`: must be handled by CallExpr.
    pub fn visit_keyword_star(&mut self, e: ExprId) -> Result<()> {
        let src = self.ast()[e].src;
        typhon_common::bail_e!(ErrorKind::UNEXPECTED_TYPE, src, "kwstar");
    }

    /// `visit(EllipsisExpr*)`: the only remaining ellipses are those of PipeExprs.
    pub fn visit_ellipsis(&mut self, e: ExprId) -> Result<()> {
        let el = self.ast()[e].kind.as_ellipsis().unwrap().clone();
        if el.is_pipe() && self.realize(self.ety(e))?.is_some() {
            self.set_done_expr(e);
        } else if el.is_standalone() {
            let f = self.id_n("ellipsis");
            let c = self.call_n(f, vec![]);
            let r = self.tr(c)?;
            self.result_expr = Some(r);
            let (a, b) = (self.ety(e), self.ety(r));
            self.unify(a, b)?;
        }
        Ok(())
    }

    /// `visit(CallExpr*)`: typecheck a call expression.
    ///   `fn(1, 2, x=3, y=4)` -> `func(a=1, x=3, args=(2,), kwargs=KwArgs(y=4), T=int)`
    ///   `fn(arg1, ...)`      -> `(_v = Partial.N10(arg1); _v)`
    pub fn visit_call(&mut self, e: ExprId) -> Result<()> {
        let types = self.types();
        if self.ctx.borrow().simple_types {
            let src = self.ast()[e].src;
            typhon_common::bail_e!(ErrorKind::CALL_NO_TYPE, src);
        }
        if crate::ast::is_id(self.ast(), Some(self.callee(e)), "tuple") && self.call_items(e).len() == 1 {
            self.ast_mut()[e].attrs.set(Attr::TupleCall);
        }

        self.validate_call(e)?;

        // Check if this call is partial call
        let mut part = PartialCallData::default();
        if let Some(last) = self.call_items(e).last() {
            if last.value.and_then(|v| self.ast()[v].kind.as_ellipsis()).is_some_and(|el| el.is_partial()) {
                part.is_partial = true;
            }
        }

        // Do not allow realization here (function will be realized later);
        // used to prevent early realization of compile_error
        self.ast_mut()[e].attrs.set(Attr::ParentCallExpr);
        let callee = self.callee(e);
        self.ast_mut()[callee].attrs.set(Attr::ExprDoNotRealize);
        let h = self.get_head_expr(callee);
        self.ast_mut()[h].attrs.set(Attr::ExprDoNotRealize);
        PARENT_CALLS.with(|p| p.borrow_mut().push((callee, e)));
        let r = self.transform(Some(callee));
        PARENT_CALLS.with(|p| {
            p.borrow_mut().pop();
        });
        let callee = r?.unwrap();
        self.set_callee(e, callee);
        self.ast_mut()[e].attrs.erase(Attr::ParentCallExpr);
        if self.is_unbound_expr(callee) {
            return Ok(()); // delay
        }

        let (callee_fn, new_expr) = self.get_callee_fn(e, &mut part)?;
        // Transform `tuple(i for i in tup)` into a GeneratorExpr
        // that will be handled during the type checking.
        if callee_fn.is_none() && self.ast()[e].attrs.has(Attr::TupleCall) {
            let first = self.call_items(e)[0].value.unwrap();
            if self.ast()[first].kind.as_generator().is_some() {
                let g = self.ast()[first].kind.as_generator().unwrap();
                if g.kind != GeneratorKind::Generator || self.ast().generator_loop_count(first) != 1 {
                    let src = self.ast()[first].src;
                    typhon_common::bail_e!(ErrorKind::CALL_TUPLE_COMPREHENSION, src);
                }
                self.ast_mut()[first].kind.as_generator_mut().unwrap().kind = GeneratorKind::TupleGenerator;
                self.result_expr = self.transform(Some(first))?;
            } else {
                self.result_expr = self.transform_tuple_fn(e)?;
            }
            return Ok(());
        } else if new_expr.is_some() {
            self.result_expr = new_expr;
            return Ok(());
        }
        let Some(mut callee_fn) = callee_fn else { return Ok(()) };

        if !self.with_class_generics(callee_fn, true, true, false, |v| v.transform_call_args(e))? {
            return Ok(());
        }
        // Early dispatch modifier
        if self.is_dispatch(callee_fn) {
            if types.func_name(callee_fn).starts_with("Tuple.__new__") {
                let n = self.call_items(e).len();
                self.generate_tuple(n, true)?;
            }
            let mut m: Option<Vec<TypeId>> = None;
            let head = self.get_head_expr(self.callee(e));
            let id = self.ast()[head].kind.as_id().map(|i| i.value.clone());
            let parent_cls = types.func_parent(callee_fn).and_then(|p| types.get_class(p));
            if let (Some(mut key), true) = (id, part.var.is_empty()) {
                // Case: function overloads (IdExpr)
                // Make sure to ignore partial constructs (they are also StmtExpr(IdExpr, ...))
                if Self::is_dispatch_name(&key) {
                    key.truncate(key.len() - FN_DISPATCH_SUFFIX.len());
                }
                let mut methods = vec![];
                for ovs in self.get_overloads(&key) {
                    if !Self::is_dispatch_name(&ovs) {
                        methods.push(self.get_function_name(&ovs).and_then(|f| f.ty).expect("bad overload"));
                    }
                }
                methods.reverse();
                let items = self.call_items(e);
                let pt = types.get_partial(self.ety(self.callee(e)));
                m = Some(self.find_matching_methods(parent_cls, &methods, &items, pt)?);
            }
            // partials have dangling ellipsis that messes up with the unbound check below
            let do_dispatch = m.as_ref().is_none_or(|m| m.is_empty()) || part.is_partial;
            if !do_dispatch && m.as_ref().is_some_and(|m| m.len() > 1) {
                for a in self.call_items(e) {
                    if self.is_unbound_expr(a.value.unwrap()) {
                        return Ok(()); // typecheck this later once we know the argument
                    }
                }
            }
            if !do_dispatch {
                callee_fn = self.instantiate_type(m.as_ref().unwrap()[0], parent_cls)?;
                let ne = self.id_n(&types.func_name(callee_fn));
                self.set_type(ne, callee_fn);
                let ce = self.callee(e);
                if self.ast()[ce].kind.as_id().is_some() {
                    self.set_callee(e, ne);
                } else if self.ast()[ce].kind.as_stmt().is_some() {
                    // Side effect...
                    let mut se = ce;
                    loop {
                        let inner = self.ast()[se].kind.as_stmt().unwrap().expr.unwrap();
                        if self.ast()[inner].kind.as_stmt().is_some() {
                            se = inner;
                        } else {
                            self.ast_mut()[se].kind.as_stmt_mut().unwrap().expr = Some(ne);
                            break;
                        }
                    }
                } else {
                    let s = self.n_stmt(ExprStmt { expr: Some(ce) });
                    let x = self.n_expr(StmtExpr { items: vec![s], expr: Some(ne) });
                    self.set_callee(e, x);
                }
                let ce = self.callee(e);
                self.set_type(ce, callee_fn);
            } else if m.as_ref().is_some_and(|m| m.is_empty()) {
                let mut a = vec![];
                for t in self.call_items(e) {
                    let tt = self.ety(t.value.unwrap());
                    let s = if types.get_static(tt).is_some() {
                        types.pretty_string(types.non_static_type(tt).unwrap_or(tt))
                    } else {
                        types.pretty_string(tt)
                    };
                    a.push(s);
                }
                let args_nice = format!("({})", a.join(", "));
                let mut name = self.get_unmangled_name(&types.func_name(callee_fn));
                let fa = types.func_ast_id(callee_fn);
                if let Some(p) = self.ast()[fa].attrs.get_str(Attr::ParentClass).map(|s| s.to_string()) {
                    name = format!("{}.{}", self.get_user_facing_name(&p), name);
                }
                let src = self.ast()[e].src;
                typhon_common::bail_e!(ErrorKind::FN_NO_ATTR_ARGS, src, name, args_nice);
            }
        }

        // Handle named and default arguments
        if let Some(r) = self.call_reorder_arguments(callee_fn, e, &mut part)? {
            self.result_expr = Some(r);
            return Ok(());
        }

        // Handle special calls
        if !part.is_partial {
            let (is_special, special) = self.transform_special_call(e)?;
            if is_special {
                self.result_expr = special;
                return Ok(());
            }
        }

        // Typecheck arguments with the function signature
        let mut items = self.call_items(e);
        let r = self.typecheck_call_args(callee_fn, &mut items, &part);
        self.set_call_items(e, items);
        let mut done = r?;
        if !part.is_partial && types.can_realize(callee_fn) {
            // Clean up DoNotRealize
            let mut head = self.callee(e);
            loop {
                self.ast_mut()[head].attrs.erase(Attr::ExprDoNotRealize);
                self.set_type(head, callee_fn);
                match self.ast()[head].kind.as_stmt() {
                    Some(s) => head = s.expr.unwrap(),
                    None => break,
                }
            }
            // Fix original fn (without replacements) with the updated one!
            if self.ast()[head].kind.as_id().is_some() {
                if let Some(r) = self.realize(callee_fn)? {
                    let rn = types.realized_name(r);
                    self.ast_mut()[head].kind.as_id_mut().unwrap().value = rn;
                    self.set_done_expr(head);
                }
            }
            let c = self.callee(e);
            let c = self.tr(c)?;
            self.set_callee(e, c);
        }
        done &= self.ast()[self.callee(e)].done;

        // Emit the final call
        if part.is_partial {
            // Case: partial call. `calleeFn(args...)` -> `Partial(args..., fn, mask)`
            let mut new_args = vec![];
            for r in self.call_items(e) {
                let v = r.value.unwrap();
                if self.ast()[v].kind.as_ellipsis().is_none() {
                    self.ast_mut()[v].attrs.set(Attr::ExprSequenceItem);
                    new_args.push(v);
                }
            }
            new_args.push(part.args.expect("partial args"));
            let mask = if self.ast()[e].attrs.has(Attr::ExprVirtual) { format!("v{}", part.known) } else { part.known.clone() };
            let tu = self.n_expr(TupleExpr { items: new_args });
            let f = types.get_func(callee_fn).unwrap();
            let partial_call = self.generate_partial_call(&mask, f, Some(tu), part.kw_args)?;
            let var = self.get_temporary_var("part");
            let call = if !part.var.is_empty() {
                // Callee is already a partial call
                let ce = self.callee(e);
                let mut stmts = self.ast()[ce].kind.as_stmt().unwrap().items.clone();
                let l = self.id_n(&var);
                stmts.push(self.n_stmt(AssignStmt { lhs: Some(l), rhs: Some(partial_call), type_expr: None, update: UpdateMode::Assign }));
                let v = self.id_n(&var);
                self.n_expr(StmtExpr { items: stmts, expr: Some(v) })
            } else {
                // New partial call: `(part = Partial(stored_args...); part)`
                let l = self.id_n(&var);
                let a = self.n_stmt(AssignStmt { lhs: Some(l), rhs: Some(partial_call), type_expr: None, update: UpdateMode::Assign });
                let v = self.id_n(&var);
                self.n_expr(StmtExpr { items: vec![a], expr: Some(v) })
            };
            self.ast_mut()[call].attrs.set(Attr::ExprPartial);
            self.result_expr = self.transform(Some(call))?;
        } else {
            // Case: normal function call
            let (et, rt) = (self.ety(e), types.func_ret(callee_fn));
            self.unify(et, rt)?;
            if done {
                let thunk = self.ast()[e].attrs.get_int(Attr::ExprVirtual);
                if let Some(t) = thunk.filter(|t| *t != -1) {
                    self.cache.super_call_signatures.insert(t as usize, callee_fn);
                } else if !self.ast()[e].attrs.has(Attr::Internal) && self.get_unmangled_name(&types.func_name(callee_fn)) == "__init__" {
                    let rn = types.realized_name(callee_fn);
                    if let Some(r) = self.get_function_type(callee_fn).and_then(|f| f.realizations.get(&rn).cloned()) {
                        r.borrow_mut().check_initialization = true;
                    }
                }
                self.set_done_expr(e);
            }
        }
        Ok(())
    }

    /// `validateCall(expr)` (static in C++).
    pub fn validate_call(&mut self, e: ExprId) -> Result<()> {
        if self.ast()[e].attrs.has(Attr::Validated) {
            return Ok(());
        }
        let (mut names_started, mut found_ellipsis) = (false, false);
        for a in self.call_items(e) {
            let v = a.value.unwrap();
            let k = &self.ast()[v].kind;
            let (is_star, is_kw, is_el) = (k.as_star().is_some(), k.as_keyword_star().is_some(), k.as_ellipsis().is_some());
            let src = self.ast()[v].src;
            if a.name.is_empty() && names_started && !(is_kw || is_el) {
                typhon_common::bail_e!(ErrorKind::CALL_NAME_ORDER, src);
            }
            if !a.name.is_empty() && (is_star || is_kw) {
                typhon_common::bail_e!(ErrorKind::CALL_NAME_STAR, src);
            }
            if is_el && found_ellipsis {
                typhon_common::bail_e!(ErrorKind::CALL_ELLIPSIS, src);
            }
            found_ellipsis |= is_el;
            names_started |= !a.name.is_empty();
        }
        self.ast_mut()[e].attrs.set(Attr::Validated);
        Ok(())
    }

    /// Unwrap `Optional` star arguments (`while typ->is(Optional)`); returns the class type.
    fn unwrap_star_optional(&mut self, star: ExprId) -> Result<Option<TypeId>> {
        let types = self.types();
        let get = |v: &Self| {
            let k = &v.ast()[star].kind;
            k.as_star().map(|s| s.expr).or_else(|| k.as_keyword_star().map(|s| s.expr)).flatten().unwrap()
        };
        let set = |v: &mut Self, x: ExprId| {
            let k = &mut v.ast_mut()[star].kind;
            if let Some(s) = k.as_star_mut() {
                s.expr = Some(x);
            } else if let Some(s) = k.as_keyword_star_mut() {
                s.expr = Some(x);
            }
        };
        let x = get(self);
        let x = self.tr(x)?;
        set(self, x);
        let mut typ = self.expr_class(x);
        while typ.is_some_and(|t| types.is(t, &stdlib().optional)) {
            let f = self.id_n(&fn_optional_unwrap());
            let cur = get(self);
            let c = self.call_n(f, vec![cur]);
            let c = self.tr(c)?;
            set(self, c);
            typ = self.expr_class(c);
        }
        Ok(typ)
    }

    /// `transformCallArgs(expr)`: expand `*args` and `**kwargs`; false if expansion could
    /// not be completed.
    pub fn transform_call_args(&mut self, e: ExprId) -> Result<bool> {
        let types = self.types();
        let mut items = self.call_items(e);
        let mut ai = 0;
        while ai < items.len() {
            let v = items[ai].value.unwrap();
            let is_star = self.ast()[v].kind.as_star().is_some();
            let is_kw = self.ast()[v].kind.as_keyword_star().is_some();
            if is_star || is_kw {
                let typ = self.unwrap_star_optional(v);
                let typ = match typ {
                    Ok(t) => t,
                    Err(err) => {
                        self.set_call_items(e, items);
                        return Err(err);
                    }
                };
                let Some(typ) = typ else {
                    // Process later
                    self.set_call_items(e, items);
                    return Ok(false);
                };
                let inner = {
                    let k = &self.ast()[v].kind;
                    k.as_star().map(|s| s.expr).or_else(|| k.as_keyword_star().map(|s| s.expr)).flatten().unwrap()
                };
                let arg_src = items[ai].src;
                if is_star && !types.is_record(typ) {
                    typhon_common::bail_e!(ErrorKind::CALL_BAD_UNPACK, arg_src, types.pretty_string(typ));
                }
                let mut head = inner;
                let mut lead = None;
                if self.has_side_effect(head) {
                    let var = self.get_temporary_var("star");
                    let l = self.id_n(&var);
                    lead = Some(self.n_expr(AssignExpr { var: Some(l), expr: Some(head) }));
                    head = self.id_n(&var);
                }
                let pick = |v: &mut Self, i: usize| {
                    let src = if lead.is_some() && i == 0 { lead.unwrap() } else { head };
                    v.cache.ast.clone_expr(src, false)
                };
                let mut new_items: Vec<CallArg> = vec![];
                if is_star {
                    let fields = self.get_class_fields(typ);
                    for (i, f) in fields.iter().enumerate() {
                        let b = pick(self, i);
                        let d = self.dot_n(b, &f.name);
                        let d = self.tr(d)?;
                        new_items.push(CallArg::new("", Some(d)));
                    }
                } else if types.is(typ, &stdlib().named_tuple) {
                    let id = self.get_int_literal(typ, 0);
                    let names = {
                        let n = types.generated_tuple_names.borrow();
                        typhon_common::seqassert!(id >= 0 && (id as usize) < n.len(), "bad id: {}", id);
                        n[id as usize].clone()
                    };
                    for (i, name) in names.iter().enumerate() {
                        let b = pick(self, i);
                        let a = self.dot_n(b, "args");
                        let d = self.dot_n(a, &format!("item{}", i + 1));
                        let d = self.tr(d)?;
                        new_items.push(CallArg::new(name.clone(), Some(d)));
                    }
                } else if types.is_record(typ) {
                    let fields = self.get_class_fields(typ);
                    for (i, f) in fields.iter().enumerate() {
                        let b = pick(self, i);
                        let d = self.dot_n(b, &f.name);
                        let d = self.tr(d)?;
                        new_items.push(CallArg::new(f.name.clone(), Some(d)));
                    }
                } else {
                    typhon_common::bail_e!(ErrorKind::CALL_BAD_KWUNPACK, arg_src, types.pretty_string(typ));
                }
                let n = new_items.len();
                items.splice(ai..ai + 1, new_items);
                ai += n;
            } else {
                // Case: normal argument (no expansion)
                let r = self.transform(Some(v));
                match r {
                    Ok(x) => items[ai].value = x,
                    Err(err) => {
                        self.set_call_items(e, items);
                        return Err(err);
                    }
                }
                ai += 1;
            }
        }
        self.set_call_items(e, items.clone());

        // Check if some argument names are reused after the expansion
        let mut seen = BTreeSet::new();
        for a in &items {
            if !a.name.is_empty() {
                if seen.contains(&a.name) {
                    typhon_common::bail_e!(ErrorKind::CALL_REPEATED_NAME, a.src, a.name);
                }
                seen.insert(a.name.clone());
            }
        }
        Ok(true)
    }

    /// `transformOptionalCall(expr, optionalType)`: normalize Optional construction without
    /// introducing redundant value layers.
    pub fn transform_optional_call(&mut self, e: ExprId, opt: TypeId) -> Result<Option<ExprId>> {
        let types = self.types();
        let items = self.call_items(e);
        if items.is_empty() {
            let v = self.n_expr(crate::ast::NoneExpr);
            self.set_type(v, opt);
            return self.transform(Some(v));
        }
        if items.len() != 1 || (!items[0].name.is_empty() && items[0].name != "what") {
            return Ok(None);
        }
        let argument = self.tr(items[0].value.unwrap())?;
        self.ast_mut()[e].kind.as_call_mut().unwrap().items[0].value = Some(argument);
        let at = self.ety(argument);
        if types.is(at, &stdlib().optional) && types.unify(opt, at, None, self)? >= 0 {
            return Ok(Some(argument));
        }
        if !types.is(at, &stdlib().none_type) {
            return Ok(None);
        }
        let g = self.extract_class_generic(opt, 0);
        if types.get_unbound(g).is_some() {
            return Ok(Some(argument));
        }
        let member = self.id_n("T");
        let tv = self.instantiate_type_var(g)?;
        self.set_type(member, tv);
        self.set_done_expr(member);
        let ce = self.callee(e);
        let d = self.dot_n(ce, "_from_none");
        let c = self.call_n(d, vec![argument, member]);
        self.transform(Some(c))
    }

    /// `getCalleeFn(expr, part)`: extract the `FuncType` to be called; handle constructors
    /// and partial functions. Returns `(callee function type, replacement expr)`.
    pub fn get_callee_fn(&mut self, e: ExprId, part: &mut PartialCallData) -> Result<(Option<TypeId>, Option<ExprId>)> {
        let types = self.types();
        let st = stdlib();
        let ce = self.callee(e);
        let Some(callee) = self.expr_class(ce) else {
            // Case: unknown callee, wait until it becomes known
            return Ok((None, None));
        };

        if self.ast()[e].attrs.has(Attr::TupleCall)
            && (types.is(self.extract_type_expr(ce), &st.tuple)
                || types.get_func(callee).is_some_and(|f| types.func_name(f).starts_with("std.internal.static.tuple.")))
        {
            return Ok((None, None));
        }

        if Self::is_type_expr(self.cache, ce) {
            let mut typ = Some(callee);
            if !crate::ast::is_id(self.ast(), Some(ce), &st.type_) {
                typ = types.get_class(self.extract_class_generic(callee, 0));
            }
            let Some(typ) = typ else { return Ok((None, None)) };
            if types.is(typ, &st.optional) {
                if let Some(v) = self.transform_optional_call(e, typ)? {
                    return Ok((None, Some(v)));
                }
            }
            let ce = self.callee(e);
            let items = self.call_items(e);
            if types.is(typ, &mangle::class("std.internal.types.error", "OSError")) {
                let d = self.dot_n(ce, "_construct");
                let c = self.call_args_n(d, items);
                return Ok((None, Some(self.tr(c)?)));
            }
            if types.is_record(typ) {
                self.ast_mut()[e].attrs.erase(Attr::TupleCall);
                // Case: tuple constructor. Transform to: `T.__new__(args)`
                let d = self.dot_n(ce, "__new__");
                let c = self.call_args_n(d, items);
                return Ok((None, Some(self.tr(c)?)));
            }

            // Reference classes with a custom __new__ receive constructor arguments.
            let allocators = self.find_method(Some(typ), "__new__")?;
            let cls_internal = self
                .get_class_type(typ)
                .ast
                .is_some_and(|a| self.ast()[a].attrs.has(Attr::Internal));
            let custom_new = !cls_internal
                && !allocators.is_empty()
                && !self.ast()[types.func_ast_id(allocators[0])].attrs.has(Attr::AutoGenerated);
            let mut stmts = vec![];
            let mut args = items;
            if custom_new {
                for arg in args.iter_mut() {
                    let tmp_name = self.get_temporary_var("ctor_arg");
                    let tmp = self.id_n(&tmp_name);
                    let v = arg.value.unwrap();
                    let k = self.ast()[v].kind.clone();
                    let value = k.as_star().map(|s| s.expr).or_else(|| k.as_keyword_star().map(|s| s.expr)).flatten().unwrap_or(v);
                    let t2 = self.cache.ast.clone_expr(tmp, false);
                    stmts.push(self.n_stmt(AssignStmt { lhs: Some(t2), rhs: Some(value), type_expr: None, update: UpdateMode::Assign }));
                    arg.value = Some(if k.as_star().is_some() {
                        self.n_expr(StarExpr { expr: Some(tmp) })
                    } else if k.as_keyword_star().is_some() {
                        self.n_expr(KeywordStarExpr { expr: Some(tmp) })
                    } else {
                        tmp
                    });
                }
            }

            // The generated allocator still takes no arguments; __init__ receives them.
            let var_name = self.get_temporary_var("ctr");
            let var = self.id_n(&var_name);
            let v1 = self.cache.ast.clone_expr(var, false);
            let d = self.dot_n(ce, "__new__");
            let nc = self.call_args_n(d, if custom_new { args.clone() } else { vec![] });
            stmts.push(self.n_stmt(AssignStmt { lhs: Some(v1), rhs: Some(nc), type_expr: None, update: UpdateMode::Assign }));
            let initializers = self.find_method(Some(typ), "__init__")?;
            // Check if object.__init__ (which is a stub) is being used (no default __init__).
            let ignore_init = custom_new && initializers.is_empty();
            if !ignore_init {
                let v2 = self.cache.ast.clone_expr(var, false);
                let d = self.dot_n(v2, "__init__");
                let ic = self.call_args_n(d, args);
                let init = self.n_stmt(ExprStmt { expr: Some(ic) });
                // class_custom_new: a factory may return another type, in which case its
                // result must not be initialized as T.
                let s = if custom_new {
                    let i = self.id_n("isinstance");
                    let v3 = self.cache.ast.clone_expr(var, false);
                    let c2 = self.cache.ast.clone_expr(ce, false);
                    let cond = self.call_n(i, vec![v3, c2]);
                    self.n_stmt(IfStmt { cond: Some(cond), if_suite: Some(init), else_suite: None })
                } else {
                    init
                };
                stmts.push(s);
            }
            let v4 = self.cache.ast.clone_expr(var, false);
            let se = self.n_expr(StmtExpr { items: stmts, expr: Some(v4) });
            return Ok((None, Some(self.tr(se)?)));
        }

        if let Some(part_type) = types.get_partial(callee) {
            let mask = types.partial_mask(part_type);
            if types.is_partial_virtual(part_type) {
                self.ast_mut()[e].attrs.set_int(Attr::ExprVirtual, -1);
            }
            let gen_fn = types.generalize(types.partial_func(part_type), 0);
            let callee_fn = self.instantiate_type(gen_fn, None)?;
            let ce = self.callee(e);
            if !types.is_partial_empty(part_type) || mask.bytes().any(|c| c != partial_flag::MISSING) {
                // Case: calling partial object `p`. Transform roughly to
                // `part = callee; partial_fn(*part.args, args...)`
                part.var = self.get_temporary_var("partcall");
                let var = self.id_n(&part.var);
                let id = self.id_n(&types.func_name(callee_fn));
                self.set_type(id, callee_fn);
                self.ast_mut()[id].attrs.set(Attr::ExprDoNotRealize);
                let a = self.n_stmt(AssignStmt { lhs: Some(var), rhs: Some(ce), type_expr: None, update: UpdateMode::Assign });
                let c = self.n_expr(StmtExpr { items: vec![a], expr: Some(id) });
                self.ast_mut()[c].attrs.set(Attr::ExprDoNotRealize);
                let c = self.tr(c)?;
                self.set_callee(e, c);
                part.known = mask.clone();
            } else {
                let id = self.id_n(&types.func_name(callee_fn));
                self.ast_mut()[id].attrs.set(Attr::ExprDoNotRealize);
                let mut c = id;
                if self.has_side_effect(ce) {
                    // In case this partial is returned from a fn that might have side effects
                    let s = self.n_stmt(ExprStmt { expr: Some(ce) });
                    c = self.n_expr(StmtExpr { items: vec![s], expr: Some(id) });
                    self.ast_mut()[c].attrs.set(Attr::ExprDoNotRealize);
                }
                let c = self.tr(c)?;
                self.set_callee(e, c);
            }
            let ct = self.ety(self.callee(e));
            typhon_common::seqassert!(types.get_func(ct).is_some(), "not a function: {}", types.debug_string(ct, 0));
            self.unify(ct, callee_fn)?;

            // Bind captured values using the same AST/value index mapping as call probes.
            let captured = self.get_partial_arg_types(callee_fn, Some(part_type));
            let params = types.func_ast(types.func_ast_id(callee_fn)).params.clone();
            let mut vi = 0;
            for (si, m) in mask.bytes().enumerate() {
                if params.get(si).is_some_and(|p| p.1 == crate::ast::ParamStatus::Value) {
                    if m == partial_flag::INCLUDED {
                        let a = self.extract_func_arg_type(callee_fn, vi);
                        if let Some(c) = captured[si] {
                            self.unify(a, c)?;
                        }
                    }
                    vi += 1;
                }
            }
            Ok((Some(callee_fn), None))
        } else if types.get_func(callee).is_none() {
            // Case: callee is not a function. Try __call__ method instead
            let ce = self.callee(e);
            let items = self.call_items(e);
            let d = self.dot_n(ce, "__call__");
            let c = self.call_args_n(d, items);
            Ok((None, Some(self.tr(c)?)))
        } else {
            Ok((types.get_func(callee), None))
        }
    }

    /// Extract the pi-th partial argument from a partial object (`getPartialArg`).
    fn get_partial_arg(&mut self, var: &str, pi: i64) -> Result<ExprId> {
        let i = self.id_n(var);
        let d = self.dot_n(i, "args");
        let id = self.tr(d)?;
        // Manually call @c transformStaticTupleIndex to avoid spurious InstantiateExpr
        let c = self.expr_class(id).expect("partial args without class");
        let n = self.n_expr(IntExpr::new(pi));
        let ex = self.transform_static_tuple_index(c, id, n)?;
        typhon_common::seqassert!(ex.0 && ex.1.is_some(), "partial indexing failed: {}", self.types().debug_string(self.ety(id), 0));
        Ok(ex.1.unwrap())
    }

    /// `callReorderArguments(callee, expr, part)`: reorder the call arguments to match the
    /// signature order; form `*args`/`**kwargs` tuples; use partial and default values.
    pub fn call_reorder_arguments(&mut self, callee_fn: TypeId, e: ExprId, part: &mut PartialCallData) -> Result<Option<ExprId>> {
        let types = self.types();
        let fast = types.func_ast_id(callee_fn);
        if self.ast()[fast].attrs.has(Attr::NoArgReorder) {
            return Ok(None);
        }
        let fparams = self.ast()[fast].kind.as_function().unwrap().items.clone();
        let fname = self.ast()[fast].kind.as_function().unwrap().name.clone();
        let has_parent_class = self.ast()[fast].attrs.has(Attr::ParentClass);
        let func_generics = types.func_generics(callee_fn);

        struct St {
            in_order: bool,
            ordered: Vec<usize>,
            args: Vec<CallArg>,
            type_args: Vec<Option<ExprId>>,
            star_idx: i64,
            star_args: Vec<ExprId>,
            kw_star_idx: i64,
            kw_star_names: Vec<String>,
            kw_star_args: Vec<ExprId>,
            new_mask: Vec<u8>,
            partial: bool,
        }
        let mut st = St {
            in_order: true,
            ordered: vec![],
            args: vec![],
            type_args: vec![],
            star_idx: -1,
            star_args: vec![],
            kw_star_idx: -1,
            kw_star_names: vec![],
            kw_star_args: vec![],
            new_mask: vec![partial_flag::INCLUDED; fparams.len()],
            partial: false,
        };
        let items = self.call_items(e);
        let known = part.known.clone();
        let kb = known.as_bytes().to_vec();
        let pvar = part.var.clone();

        fn add_reordered(v: &TypecheckVisitor<'_>, st: &mut St, items: &[CallArg], i: usize) -> bool {
            if v.has_side_effect(items[i].value.unwrap()) {
                if st.ordered.last().is_some_and(|l| i < *l) {
                    st.in_order = false;
                }
                st.ordered.push(i);
                return true;
            }
            false
        }

        // Reorder arguments if needed
        part.args = None;
        part.kw_args = None; // Stores partial *args/**kwargs expression
        if self.ast()[e].attrs.has(Attr::ExprOrderedCall) {
            st.args = items.clone();
        } else {
            let mut on_done = |v: &mut TypecheckVisitor<'a>, star_arg: i32, kwstar_arg: i32, slots: &CallArgSlots, partial: bool| -> Result<i32> {
                st.partial = partial;
                v.with_class_generics(callee_fn, true, false, false, |v| {
                    let (mut pi, mut gi) = (0i64, 0usize);
                    for si in 0..slots.len() {
                        let p = &fparams[si];
                        // Get the argument name to be used later
                        let rn = p.name_with_stars().1;
                        let real_name = v.get_unmangled_name(&rn);
                        let slot = &slots[si];
                        let known_is = |f: u8| !kb.is_empty() && kb[si] == f;
                        if p.is_generic() {
                            // Case: generic arguments. Populate typeArgs
                            if let Some(rest) = real_name.strip_prefix('$') {
                                if slot.is_empty() {
                                    if known_is(partial_flag::INCLUDED) {
                                        let t = v.id_n(&real_name);
                                        v.set_type(t, func_generics[gi].ty());
                                        st.type_args.push(Some(t));
                                    } else {
                                        let t = v.id_n(rest);
                                        st.type_args.push(Some(v.tr(t)?));
                                    }
                                } else {
                                    st.type_args.push(items[slot[0] as usize].value);
                                    if add_reordered(v, &mut st, &items, slot[0] as usize) {
                                        st.in_order = false; // type arguments always need preprocessing
                                    }
                                }
                                st.new_mask[si] = partial_flag::INCLUDED;
                            } else if slot.is_empty() {
                                st.type_args.push(None);
                                st.new_mask[si] = partial_flag::MISSING;
                            } else {
                                st.type_args.push(items[slot[0] as usize].value);
                                st.new_mask[si] = partial_flag::INCLUDED;
                                if add_reordered(v, &mut st, &items, slot[0] as usize) {
                                    st.in_order = false; // type arguments always need preprocessing
                                }
                            }
                            gi += 1;
                        } else if si as i32 == star_arg
                            && !(slot.len() == 1 && v.ast()[items[slot[0] as usize].value.unwrap()].attrs.has(Attr::ExprStarArgument))
                        {
                            // Case: *args. Build the tuple that holds them all
                            if !kb.is_empty() {
                                let pa = v.get_partial_arg(&pvar, -1)?;
                                st.star_args.push(v.n_expr(StarExpr { expr: Some(pa) }));
                            }
                            for &x in slot.iter() {
                                st.star_args.push(items[x as usize].value.unwrap());
                                add_reordered(v, &mut st, &items, x as usize);
                            }
                            st.star_idx = st.args.len() as i64;
                            st.args.push(CallArg::new(real_name, None));
                            if partial {
                                st.new_mask[si] = partial_flag::MISSING;
                            }
                        } else if si as i32 == kwstar_arg
                            && !(slot.len() == 1 && v.ast()[items[slot[0] as usize].value.unwrap()].attrs.has(Attr::ExprKwStarArgument))
                        {
                            // Case: **kwargs. Build the named tuple that holds them all
                            let mut new_keywords: HashMap<String, usize> = HashMap::new();
                            for &x in slot.iter() {
                                new_keywords.insert(items[x as usize].name.clone(), x as usize);
                            }
                            if !kb.is_empty() {
                                let i = v.id_n(&pvar);
                                let d = v.dot_n(i, "kwargs");
                                let captured = v.tr(d)?;
                                for (name, value) in v.extract_named_tuple(captured) {
                                    st.kw_star_names.push(name.clone());
                                    if let Some(r) = new_keywords.remove(&name) {
                                        st.kw_star_args.push(items[r].value.unwrap());
                                        add_reordered(v, &mut st, &items, r);
                                    } else {
                                        st.kw_star_args.push(v.tr(value)?);
                                    }
                                }
                            }
                            for &x in slot.iter() {
                                let n = &items[x as usize].name;
                                if !new_keywords.contains_key(n) {
                                    continue;
                                }
                                st.kw_star_names.push(n.clone());
                                st.kw_star_args.push(items[x as usize].value.unwrap());
                                add_reordered(v, &mut st, &items, x as usize);
                            }
                            st.kw_star_idx = st.args.len() as i64;
                            st.args.push(CallArg::new(real_name, None));
                            if partial {
                                st.new_mask[si] = partial_flag::MISSING;
                            }
                        } else if slot.is_empty() {
                            // Case: no arguments provided.
                            if known_is(partial_flag::INCLUDED) {
                                // Case 1: Argument captured by partial
                                let a = v.get_partial_arg(&pvar, pi)?;
                                st.args.push(CallArg::new(real_name, Some(a)));
                                pi += 1;
                            } else if let Some(rest) = real_name.strip_prefix('$') {
                                // Case 3: Local name capture
                                let mut added = false;
                                if partial {
                                    if let Some(val) = v.find(rest, 0) {
                                        let vt = val.borrow().ty;
                                        let same = vt.and_then(|t| types.get_func(t)).is_some_and(|f| {
                                            let a = types.func_ast_id(f);
                                            v.ast()[a].kind.as_function().is_some_and(|x| x.name == fname)
                                        });
                                        if same {
                                            // Special case: fn(fn=fn)
                                            // Delay this one.
                                            let el = v.ellipsis_n();
                                            let el = v.tr(el)?;
                                            st.args.push(CallArg::new(real_name.clone(), Some(el)));
                                            st.new_mask[si] = partial_flag::MISSING;
                                            added = true;
                                        }
                                    }
                                }
                                if !added {
                                    let i = v.id_n(rest);
                                    let i = v.tr(i)?;
                                    st.args.push(CallArg::new(real_name, Some(i)));
                                }
                            } else if let Some(def) = p.default_value {
                                // Case 4: Default is present
                                if let Some(ai) = v.ast()[def].kind.as_id().map(|i| i.value.clone()) {
                                    // Non-values (Ids / .default names)
                                    let slotv = v.find(&ai, 0);
                                    // Compiler-owned module slots keep definition-time values
                                    // without storing another copy.
                                    let global_default = slotv.is_some_and(|s| s.borrow().is_global())
                                        && ai.starts_with(".default.")
                                        && !has_parent_class;
                                    if partial {
                                        st.new_mask[si] = partial_flag::DEFAULT;
                                    }
                                    if known_is(partial_flag::DEFAULT) {
                                        // Case 4a/1: Default already captured by partial.
                                        let a = v.get_partial_arg(&pvar, pi)?;
                                        st.args.push(CallArg::new(real_name, Some(a)));
                                        pi += 1;
                                    } else if partial && global_default {
                                        let el = v.ellipsis_n();
                                        let el = v.tr(el)?;
                                        st.args.push(CallArg::new(real_name, Some(el)));
                                        st.new_mask[si] = partial_flag::MISSING;
                                    } else {
                                        let i = v.id_n(&ai);
                                        let x = v.tr(i)?;
                                        typhon_common::seqassert!(types.get_link(v.ety(x)).is_some(), "not a link type");
                                        st.args.push(CallArg::new(real_name, Some(x)));
                                    }
                                } else if !partial {
                                    // Case 4b: Values / non-Id defaults (None, etc.)
                                    let d = v.get_call_default(p)?;
                                    st.args.push(CallArg::new(real_name, Some(d)));
                                } else {
                                    let el = v.ellipsis_n();
                                    let el = v.tr(el)?;
                                    st.args.push(CallArg::new(real_name, Some(el)));
                                    st.new_mask[si] = partial_flag::MISSING;
                                }
                            } else if partial {
                                // Case 5: This is partial call. Just add ... for missing arguments
                                let el = v.ellipsis_n();
                                let el = v.tr(el)?;
                                st.args.push(CallArg::new(real_name, Some(el)));
                                st.new_mask[si] = partial_flag::MISSING;
                            } else {
                                typhon_common::seqassert!(false, "cannot happen");
                            }
                        } else {
                            // Case: argument provided
                            typhon_common::seqassert!(slot.len() == 1, "call transformation failed");
                            st.args.push(CallArg::new(real_name, items[slot[0] as usize].value));
                            add_reordered(v, &mut st, &items, slot[0] as usize);
                        }
                    }
                    Ok(0)
                })
            };
            let mut on_error = |_v: &mut TypecheckVisitor<'a>, _k: ErrorKind, o: typhon_common::SrcInfo, msg: String| -> Result<i32> {
                Err(typhon_common::CompilerError::parser(ErrorKind::CUSTOM, o, &[&msg]))
            };
            self.reorder_named_args(callee_fn, &items, &mut on_done, Some(&mut on_error), &known)?;
        }
        self.finish_reorder(callee_fn, e, part, ReorderState {
            in_order: st.in_order,
            ordered: st.ordered,
            args: st.args,
            type_args: st.type_args,
            star_idx: st.star_idx,
            star_args: st.star_args,
            kw_star_idx: st.kw_star_idx,
            kw_star_names: st.kw_star_names,
            kw_star_args: st.kw_star_args,
            new_mask: st.new_mask,
            partial: st.partial,
        })
    }
}

struct ReorderState {
    in_order: bool,
    ordered: Vec<usize>,
    args: Vec<CallArg>,
    type_args: Vec<Option<ExprId>>,
    star_idx: i64,
    star_args: Vec<ExprId>,
    kw_star_idx: i64,
    kw_star_names: Vec<String>,
    kw_star_args: Vec<ExprId>,
    new_mask: Vec<u8>,
    partial: bool,
}

impl<'a> TypecheckVisitor<'a> {
    /// Second half of `callReorderArguments` (after `reorderNamedArgs`).
    fn finish_reorder(&mut self, callee_fn: TypeId, e: ExprId, part: &mut PartialCallData, mut st: ReorderState) -> Result<Option<ExprId>> {
        let types = self.types();
        let fast = types.func_ast_id(callee_fn);
        let fparams = self.ast()[fast].kind.as_function().unwrap().items.clone();

        // Do reordering
        if !st.in_order {
            let mut prepends = vec![];
            st.ordered.sort();
            let mut items = self.call_items(e);
            for &i in &st.ordered {
                let name = self.get_temporary_var("call");
                let ev = items[i].value.unwrap();
                let l = self.id_n(&name);
                let pt = self.expr_type(ev);
                let te = self.get_param_type(pt);
                let a = self.n_stmt(AssignStmt { lhs: Some(l), rhs: Some(ev), type_expr: te, update: UpdateMode::Assign });
                let front = self.tr_stmt(a)?;
                let s = self.id_n(&name);
                let swap = self.tr(s)?;
                items[i].value = Some(swap);
                prepends.push(front);
            }
            self.set_call_items(e, items);
            // getCalleeFn has already replaced the partial with its underlying function.
            // Restore the saved partial args and run its setup before new arguments.
            if !part.var.is_empty() {
                let ce = self.callee(e);
                let cs = self.ast()[ce].kind.as_stmt().unwrap().items.clone();
                prepends.splice(0..0, cs);
                let v = self.id_n(&part.var);
                self.set_callee(e, v);
            }
            let se = self.n_expr(StmtExpr { items: prepends, expr: Some(e) });
            return Ok(Some(self.tr(se)?));
        }

        // Handle *args
        if st.star_idx != -1 {
            let mut se = self.n_expr(TupleExpr { items: st.star_args.clone() });
            self.ast_mut()[se].attrs.set(Attr::ExprStarArgument);
            if !crate::ast::is_id(self.ast(), Some(self.callee(e)), "hasattr") {
                se = self.tr(se)?;
            }
            if st.partial {
                part.args = Some(se);
                let el = self.ellipsis_n();
                st.args[st.star_idx as usize].value = Some(self.tr(el)?);
            } else {
                st.args[st.star_idx as usize].value = Some(se);
            }
        }

        // Handle **kwargs
        if st.kw_star_idx != -1 {
            let kwid = self.generate_kw_id(&st.kw_star_names);
            let nt = self.id_n(&stdlib().named_tuple);
            let tu = self.n_expr(TupleExpr { items: st.kw_star_args.clone() });
            let ki = self.n_expr(IntExpr::new(kwid as i64));
            let c = self.call_n(nt, vec![tu, ki]);
            let kwe = self.tr(c)?;
            self.ast_mut()[kwe].attrs.set(Attr::ExprKwStarArgument);
            // Note: the reference implementation assigns an `Expr*` to the whole `CallArg` (implicit
            // conversion drops the name); the name is cleared to match.
            if st.partial {
                part.kw_args = Some(kwe);
                let el = self.ellipsis_n();
                let el = self.tr(el)?;
                st.args[st.kw_star_idx as usize] = self.ast().call_arg("", Some(el));
            } else {
                st.args[st.kw_star_idx as usize] = self.ast().call_arg("", Some(kwe));
            }
        }

        // Populate partial data
        if let Some(a) = part.args {
            self.ast_mut()[a].attrs.set(Attr::ExprSequenceItem);
        }
        if let Some(a) = part.kw_args {
            self.ast_mut()[a].attrs.set(Attr::ExprSequenceItem);
        }
        if part.is_partial {
            self.ast_mut()[e].kind.as_call_mut().unwrap().items.pop();
            if part.args.is_none() {
                let t = self.n_expr(TupleExpr { items: vec![] });
                part.args = Some(self.tr(t)?); // use ()
            }
            if part.kw_args.is_none() {
                let nt = self.id_n(&stdlib().named_tuple);
                let c = self.call_n(nt, vec![]);
                part.kw_args = Some(self.tr(c)?); // use NamedTuple()
            }
        }

        // Unify function type generics with the provided generics
        let ordered_call = self.ast()[e].attrs.has(Attr::ExprOrderedCall);
        let fg = types.func_generics(callee_fn);
        typhon_common::seqassert!(
            (ordered_call && st.type_args.is_empty()) || (!ordered_call && st.type_args.len() == fg.len()),
            "bad vector sizes"
        );
        if !fg.is_empty() {
            let ni = self.ast_mut().non_inferrable_generics(fast);
            let mut si = 0;
            while !ordered_call && si < fg.len() {
                let g = &fg[si];
                if let Some(ta) = st.type_args[si] {
                    let typ = self.extract_type_expr(ta);
                    if g.static_kind != crate::types::LiteralKind::Runtime && types.static_kind(typ) == crate::types::LiteralKind::Runtime {
                        let src = self.ast()[ta].src;
                        typhon_common::bail_e!(ErrorKind::EXPECTED_STATIC, src);
                    }
                    self.unify(typ, g.ty())?;
                } else if self.is_unbound(g.ty())
                    && fparams[si].default_value.is_none()
                    && !st.partial
                    && ni.contains(g.name.as_str())
                {
                    let n = self.get_unmangled_name(&g.name);
                    typhon_common::bail_e!(ErrorKind::CUSTOM, self.src, "generic '{}' not provided", n);
                }
                si += 1;
            }
        }

        self.set_call_items(e, st.args);
        self.ast_mut()[e].attrs.set(Attr::ExprOrderedCall);
        part.known = String::from_utf8(st.new_mask).unwrap();
        Ok(None)
    }
}

impl<'a> TypecheckVisitor<'a> {
    /// `typecheckCallArgs(callee, args, part)`: unify the call arguments' types with the
    /// function signature; wrap arguments; handle default generics.
    pub fn typecheck_call_args(&mut self, callee_fn: TypeId, args: &mut Vec<CallArg>, partial: &PartialCallData) -> Result<bool> {
        let types = self.types();
        let st = stdlib();
        let fast = types.func_ast_id(callee_fn);
        let fparams = self.ast()[fast].kind.as_function().unwrap().items.clone();
        let fname = self.ast()[fast].kind.as_function().unwrap().name.clone();
        let has_ret = self.ast()[fast].kind.as_function().unwrap().ret.is_some();
        let is_static = self.ast()[fast].attrs.has(Attr::StaticMethod);
        let mut wrapping_done = true; // tracks whether all arguments are wrapped
        let mut replacements: Vec<Option<TypeId>> = vec![]; // list of replacement arguments

        // Unify Self with the argument so that generics (e.g. N in Int[N].__lshift__)
        // propagate
        let parent = types.func_parent(callee_fn).and_then(|p| types.get_class(p));
        if let Some(parent) = parent {
            if !types.can_realize(parent)
                && !args.is_empty()
                && !fparams.is_empty()
                && !is_static
                && self.get_unmangled_name(&fparams[0].name) == "self"
            {
                let at = self.ety(args[0].value.unwrap());
                if let Some(receiver) = types.get_class(self.get_underlying_static_type(at)?) {
                    let mro = self.cache.get_class_by_type(receiver).map(|c| c.mro.clone()).unwrap_or_default();
                    let pname = types.class_name(parent);
                    for base in mro {
                        if types.class_name(base) == pname {
                            let it = self.instantiate_type(base, Some(receiver))?;
                            self.unify(parent, it)?;
                            break;
                        }
                    }
                }
            }
        }

        self.with_class_generics(callee_fn, true, false, false, |v| {
            let mut si = 0;
            for (i, p) in fparams.iter().enumerate() {
                if p.is_generic() {
                    continue;
                }
                if p.name.starts_with('*') && p.ty.is_some() {
                    // Special case: `*args: type` and `**kwargs: type`
                    let av = args[si].value.unwrap();
                    if v.ast()[av].kind.as_call().is_some() {
                        let c = v.cache.ast.clone_expr(p.ty.unwrap(), false);
                        let c = v.tr(c)?;
                        let typ = v.extract_type_expr(c);
                        let mut call_expr = av;
                        if p.name.starts_with("**") {
                            call_expr = v.ast()[av].kind.as_call().unwrap().items[0].value.unwrap();
                        }
                        let mut citems = v.call_items(call_expr);
                        for ca in citems.iter_mut() {
                            let mut x = ca.value.unwrap();
                            if v.wrap_expr(&mut x, typ, Some(callee_fn), true)? {
                                let xt = v.ety(x);
                                v.unify(xt, typ)?;
                            } else {
                                wrapping_done = false;
                            }
                            ca.value = Some(x);
                        }
                        v.set_call_items(call_expr, citems.clone());
                        let name = types.class_name(v.expr_class(call_expr).unwrap()).unwrap().to_string();
                        let n = v.id_n(&name);
                        let tc = v.call_args_n(n, citems);
                        let tup = v.tr(tc)?;
                        if p.name.starts_with("**") {
                            let f = v.id_n(&mangle::mangled_method("", "NamedTuple", "__new__", 0, 0));
                            let g = v.extract_class_generic(v.ety(av), 0);
                            let iv = types.get_int_static(g).expect("not an int static");
                            let ie = v.n_expr(IntExpr::new(iv));
                            let c = v.call_n(f, vec![tup, ie]);
                            args[si].value = Some(v.tr(c)?);
                        } else {
                            args[si].value = Some(tup);
                        }
                    }
                    replacements.push(Some(v.ety(args[si].value.unwrap())));
                    // else this is empty and is a partial call; leave it for later
                } else if partial.is_partial && !partial.known.is_empty() && partial.known.as_bytes()[si] == partial_flag::DEFAULT {
                    // Defaults should not be unified (yet)!
                    replacements.push(Some(v.extract_func_arg_type(callee_fn, si)));
                } else {
                    let original = v.extract_func_arg_type(callee_fn, si);
                    let at = v.ety(args[si].value.unwrap());
                    let ust = v.get_underlying_static_type(at)?;
                    let expected = v.get_call_arg_type(callee_fn, i, si, ust)?;
                    if expected != original && !has_ret && types.can_realize(types.func_ret(callee_fn)) {
                        let sig = v.get_function_name(&types.func_name(callee_fn)).and_then(|f| f.ty).unwrap();
                        let r = v.instantiate_type(types.func_ret(sig), None)?;
                        types.with_class_mut(callee_fn, |c| c.generics[1].ty = Some(r));
                    }
                    if !v.cache.empty_literal_slots.is_empty() && si > 0 && !args.is_empty() {
                        let a0 = v.ety(args[0].value.unwrap());
                        let dict_name = types.class_name(v.get_stdlib_type(&st.dict)?).unwrap();
                        let recv = types.get_class(v.get_underlying_static_type(a0)?);
                        if let Some(recv) = recv.filter(|r| types.is(*r, &st.list) || types.is(*r, &st.set) || types.is(*r, &dict_name)) {
                            let mut method = v.get_unmangled_name(&fname);
                            if let Some(pos) = method.rfind('.') {
                                method = method[pos + 1..].to_string();
                            }
                            let ng = types.generics_len(recv);
                            let is_dict_recv = ng == 2;
                            let mut slot: i32 = -1;
                            if method == "append" || method == "add" || (method == "insert" && si == 2) {
                                slot = 0;
                            } else if method == "__setitem__" || method == "setdefault" {
                                slot = if is_dict_recv { si as i32 - 1 } else if si == 2 { 0 } else { -1 };
                            }
                            if slot >= 0 && (slot as usize) < ng {
                                v.widen_empty_literal_slot(args[0].value.unwrap(), slot as usize, at)?;
                            }
                        }
                    }
                    let mut x = args[si].value.unwrap();
                    if v.wrap_expr(&mut x, expected, Some(callee_fn), true)? {
                        let xt = v.ety(x);
                        v.unify(xt, expected)?;
                    } else {
                        wrapping_done = false;
                    }
                    args[si].value = Some(x);
                    replacements.push(Some(if types.get_class(expected).is_none() { v.ety(x) } else { expected }));
                }
                si += 1;
            }
            Ok(true)
        })?;

        // Realize arguments
        let mut done = true;
        for a in args.iter_mut() {
            let at = self.ety(a.value.unwrap());
            if types.is(at, &st.optional) {
                let payload = self.extract_class_generic(at, 0);
                if types.get_unbound(payload).is_some() {
                    let mut c = self.ctx.borrow_mut();
                    let b = c.get_base();
                    b.pending_defaults.entry(super::Base::NONE_ARGUMENT_DEFAULT_PRIORITY).or_default().insert(payload);
                    b.pending_defaults.entry(0).or_default().remove(&payload);
                }
            }
            // Previous unifications can qualify existing identifiers.
            // Transform again to get the full identifier
            if self.realize(at)?.is_some() {
                a.value = self.transform(a.value)?;
            }
            done &= self.ast()[a.value.unwrap()].done;
        }

        // Handle default generics
        if !partial.is_partial {
            let mut j = 0;
            for p in fparams.iter() {
                if !wrapping_done {
                    break;
                }
                if p.is_generic() {
                    if let Some(d) = p.default_value {
                        if self.is_unbound(self.extract_func_generic(callee_fn, j)) {
                            let r = self.with_class_generics(callee_fn, true, false, false, |v| {
                                let c = v.cache.ast.clone_expr(d, true);
                                v.tr(c)
                            })?;
                            let def = self.extract_type_expr(r);
                            let g = self.extract_func_generic(callee_fn, j);
                            self.unify(g, def)?;
                        }
                    }
                    j += 1;
                }
            }
        }

        types.replace_arg_types(callee_fn, &replacements);
        Ok(done)
    }

    /// `transformSpecialCall(expr)`: `(handled, replacement)` for `superf`, `super`,
    /// `__ptr__`, `__array__`, `isinstance`, `static.len`, `hasattr`, `getattr`, `type`,
    /// `compile_error`, ...
    pub fn transform_special_call(&mut self, e: ExprId) -> Result<(bool, Option<ExprId>)> {
        if self.ast()[e].attrs.has(Attr::ExprNoSpecial) {
            return Ok((false, None));
        }
        let ce = self.callee(e);
        let Some(v) = self.ast()[ce].kind.as_id().map(|i| i.value.clone()) else { return Ok((false, None)) };
        let is_f = |module: &str, cls: &str, name: &str| {
            if name.is_empty() { v == mangle::func(module, cls) } else { v == mangle::mangled_method(module, cls, name, 0, 0) }
        };
        let sm = "std.internal.static";
        if is_f("", "superf", "") {
            Ok((true, self.transform_super_f(e)?))
        } else if is_f("", "super", "") && self.call_items(e).is_empty() {
            Ok((true, self.transform_super()?))
        } else if is_f("", "__ptr__", "") {
            Ok((true, self.transform_ptr(e)?))
        } else if is_f("", "__array__", "__new__") {
            Ok((true, self.transform_array(e)?))
        } else if is_f("", "isinstance", "") {
            Ok((true, self.transform_is_instance(e)?))
        } else if is_f(sm, "len", "") {
            Ok((true, self.transform_static_len(e)?))
        } else if is_f("", "hasattr", "") {
            Ok((true, self.transform_has_attr(e, false)?))
        } else if is_f("", "hasattr_dynamic", "") {
            Ok((true, self.transform_has_attr(e, true)?))
        } else if is_f("", "_getattr", "") {
            Ok((true, self.transform_get_attr(e)?))
        } else if is_f("", "setattr", "") {
            Ok((true, self.transform_set_attr(e)?))
        } else if is_f("", "type", "__new__") {
            Ok((true, self.transform_type_fn(e)?))
        } else if is_f("", "compile_error", "") {
            Ok((true, self.transform_compile_error(e)?))
        } else if is_f(sm, "print", "") {
            Ok((false, self.transform_static_print_fn(e)?))
        } else if is_f("std.collections", "namedtuple", "") {
            Ok((true, self.transform_named_tuple(e)?))
        } else if is_f("std.functools", "partial", "") {
            Ok((true, self.transform_functools_partial(e)?))
        } else if is_f(sm, "has_rtti", "") {
            Ok((true, self.transform_has_rtti_fn(e)?))
        } else if is_f(sm, "function", "realized") {
            Ok((true, self.transform_realized_fn(e)?))
        } else if is_f(sm, "function", "can_call") {
            Ok((true, self.transform_static_fn_can_call(e)?))
        } else if is_f(sm, "function", "has_type") {
            Ok((true, self.transform_static_fn_arg_has_type(e)?))
        } else if is_f(sm, "function", "get_type") {
            Ok((true, self.transform_static_fn_arg_get_type(e)?))
        } else if is_f(sm, "function", "args") {
            Ok((true, self.transform_static_fn_args(e)?))
        } else if is_f(sm, "function", "has_default") {
            Ok((true, self.transform_static_fn_has_default(e)?))
        } else if is_f(sm, "function", "get_default") {
            Ok((true, self.transform_static_fn_get_default(e)?))
        } else if is_f(sm, "function", "wrap_args") {
            Ok((true, self.transform_static_fn_wrap_call_args(e)?))
        } else if is_f(sm, "vars", "") {
            Ok((true, self.transform_static_vars(e)?))
        } else if is_f(sm, "children", "") {
            Ok((true, self.transform_static_children(e)?))
        } else if is_f(sm, "tuple_type", "") {
            Ok((true, self.transform_static_tuple_type(e)?))
        } else if is_f(sm, "format", "") {
            Ok((true, self.transform_static_format(e)?))
        } else if is_f(sm, "int_to_string", "") {
            Ok((true, self.transform_static_int_to_str(e)?))
        } else if is_f(sm, "platform", "") {
            Ok((true, self.transform_static_platform(e)?))
        } else if is_f(sm, "contains", "") {
            Ok((true, self.transform_static_contains(e)?))
        } else {
            Ok((false, None))
        }
    }

    /// `getMRO(type)`: the inheritance hierarchy of a type (most recent first).
    pub fn get_mro(&mut self, t: TypeId) -> Result<Vec<TypeId>> {
        let mut result = vec![];
        let Some(c) = self.cache.get_class_by_type(t) else { return Ok(result) };
        for umt in c.mro.clone() {
            let mt = self.instantiate_type(umt, Some(t))?;
            self.realize(mt)?; // ensure that parent types are realized
            result.push(mt);
        }
        Ok(result)
    }

    /// `generatePartialCall(mask, fn, args, kwargs)`: `Partial(args, kwargs, fn, mask)`.
    pub fn generate_partial_call(&mut self, mask: &str, f: TypeId, args: Option<ExprId>, kwargs: Option<ExprId>) -> Result<ExprId> {
        let types = self.types();
        let args = match args {
            Some(a) => a,
            None => {
                let inner = self.n_expr(TupleExpr { items: vec![] });
                self.n_expr(TupleExpr { items: vec![inner] })
            }
        };
        let kwargs = match kwargs {
            Some(k) => k,
            None => {
                let nt = self.id_n(&stdlib().named_tuple);
                self.call_n(nt, vec![])
            }
        };
        let efn = self.id_n(&types.func_name(f));
        let ut = self.get_stdlib_type(&stdlib().unrealized_type)?;
        let it = self.instantiate_type_with(ut, &[f])?;
        self.set_type(efn, it);
        self.set_done_expr(efn);
        let m = self.n_expr(StringExpr::new(mask, ""));
        let p = self.id_n("Partial");
        let items = vec![
            CallArg::new("args", Some(args)),
            CallArg::new("kwargs", Some(kwargs)),
            CallArg::new("M", Some(m)),
            CallArg::new("F", Some(efn)),
        ];
        Ok(self.call_args_n(p, items))
    }
}

#[allow(dead_code)]
fn _unused(_: DotExpr, _: IdExpr) {}
