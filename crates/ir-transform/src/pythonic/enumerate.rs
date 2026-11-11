//! Port of codon/cir/transform/pythonic/enumerate.{h,cpp}

use rustc_hash::FxHashMap;
use typhon_common::mangle;
use typhon_ir::util::cloning::CloneVisitor;
use typhon_ir::util::irtools;
use typhon_ir::util::operator::{Operator, OperatorState};
use typhon_ir::{FuncId, Id, ImperativeForFlow, Module, NodeRef, TypeKind, ValueId, ValueKind, VarId, magic};

use crate::lowering::{operator_pass, operator_state};

/// The yielded tuple can disappear only if its sole uses are the loop target
/// definition and the two matched unpacking reads.
struct TupleUseChecker {
    state: OperatorState,
    loop_: Id,
    index: Id,
    element: Id,
    loop_var: Id,
    valid: bool,
}

impl Operator for TupleUseChecker {
    operator_state!();
    fn pre_hook(&mut self, m: &mut Module, n: NodeRef) {
        let Some(v) = n.as_value() else { return };
        let uid = m.value_uid(v);
        if uid == self.loop_ || uid == self.index || uid == self.element {
            return;
        }
        for var in m.used_variables(v.into()) {
            if m.var_uid(var) == self.loop_var {
                self.valid = false;
            }
        }
    }
}

/// Returns `(assign, extract)` if `value` is `x = tuple.<field>`.
fn get_unpack(m: &Module, value: ValueId, tuple: VarId, field: &str) -> Option<(ValueId, ValueId)> {
    let a = m.as_assign_instr(value)?;
    let e = m.as_extract_instr(a.rhs)?;
    let var = irtools::get_var(m, e.val)?;
    (m.var_uid(var) == m.var_uid(tuple) && e.field == field).then_some((value, a.rhs))
}

/// Replaces `for i, x in enumerate(it, start)` with a private counter, avoiding
/// generator and tuple construction.
#[derive(Default)]
pub struct EnumerateOptimization {
    state: OperatorState,
}

impl EnumerateOptimization {
    pub const KEY: &'static str = "core-pythonic-enumerate-opt";
    pub fn new() -> Self {
        Self::default()
    }
}

impl Operator for EnumerateOptimization {
    operator_state!();
    fn handle_for_flow(&mut self, m: &mut Module, lp: ValueId) {
        let ff = m.as_for_flow(lp).unwrap().clone();
        // A shared incrementing counter assumes serial, synchronous iteration.
        if ff.is_parallel() || ff.is_async {
            return;
        }
        // The local use check cannot see other functions observing a global target.
        if m.var(ff.var).global {
            return;
        }
        // Require a direct builtin call.
        let Some(call) = m.as_call_instr(ff.iter).cloned() else { return };
        let Some(func) = irtools::get_func(m, call.callee) else { return };
        let prefix = mangle::func("std.internal.builtin", "enumerate");
        if call.args.len() != 2 || !m.name(func.into()).starts_with(&prefix) {
            return;
        }
        // "index, value" = tuple var + index = tuple.item1; value = tuple.item2.
        let Some(parent) = self.parent_func(m).filter(|p| m.func(*p).is_some_and(|d| d.is_bodied())) else { return };
        let body = ff.body;
        let Some(series) = m.as_series_flow(body).map(|s| s.series.clone()) else { return };
        if series.is_empty() {
            return;
        }
        let Some((index_assign, index_extract)) = get_unpack(m, series[0], ff.var, "item1") else { return };
        if series.len() < 2 {
            return;
        }
        let Some((element_assign, element_extract)) = get_unpack(m, series[1], ff.var, "item2") else { return };
        let position = 2usize;

        // Check the whole function, not just the loop body.
        let index_val = m.as_extract_instr(index_extract).unwrap().val;
        let element_val = m.as_extract_instr(element_extract).unwrap().val;
        let mut uses = TupleUseChecker {
            state: OperatorState::new(false),
            loop_: m.value_uid(lp),
            index: m.value_uid(index_val),
            element: m.value_uid(element_val),
            loop_var: m.var_uid(ff.var),
            valid: true,
        };
        if let Some(pb) = m.func_body(parent) {
            uses.process(m, pb.into());
        }
        if !uses.valid {
            return;
        }

        let (iterable, start) = (call.args[0], call.args[1]);
        let st = m.value_type_mut(start);
        let it = m.get_int_type();
        if !m.type_is(st, it) {
            return;
        }
        let iterable_t = m.value_type_mut(iterable);
        let array = m.type_name(iterable_t).starts_with(&format!("{}[", mangle::class("std.numpy.ndarray", "ndarray")));
        let mut len_func: Option<FuncId> = None;
        let mut iter_info: Option<(FuncId, ValueId)> = None;
        if array {
            len_func = m.get_or_realize_method(iterable_t, magic::LEN, &[iterable_t], &[]);
            if len_func.is_none() {
                return;
            }
        } else if !matches!(m.type_kind(iterable_t), TypeKind::Generator { .. }) {
            // Realize iter(x) through the frontend; decline unless its body is
            // exactly `return <generator expr>`.
            let Some(f) = m.get_or_realize_func("iter", &[iterable_t], &[], "std.internal.builtin") else { return };
            if !m.func(f).is_some_and(|d| d.is_bodied()) {
                return;
            }
            let Some(ib) = m.func_body(f).and_then(|b| m.as_series_flow(b).map(|s| s.series.clone())) else { return };
            if ib.len() != 1 {
                return;
            }
            let Some(rv) = m.as_return_instr(ib[0]).and_then(|r| r.value) else { return };
            let rt = m.value_type_mut(rv);
            if !matches!(m.type_kind(rt), TypeKind::Generator { .. }) {
                return;
            }
            iter_info = Some((f, rv));
        }

        // Evaluate source and start once, in that order.
        let setup = m.nr_series_flow();
        let source = irtools::make_var(m, iterable, setup, Some(parent), false);
        let counter = irtools::make_var(m, start, setup, Some(parent), false);
        let elem_lhs = m.as_assign_instr(element_assign).unwrap().lhs;
        let et = m.var_type(elem_lhs).expect("element type");
        let element = m.nr_var(et, false, "");
        m.func_push_symbol(parent, element);
        let cv0 = m.nr_var_value(counter);
        m.as_assign_instr_mut(index_assign).unwrap().rhs = cv0;
        let ev0 = m.nr_var_value(element);
        m.as_assign_instr_mut(element_assign).unwrap().rhs = ev0;

        // Advance the counter before the user's body so continue cannot skip it.
        let cv = m.nr_var_value(counter);
        let one = m.get_int(1);
        let inc = m.binary_op(cv, magic::ADD, one).expect("int __add__ not found");
        let inc_assign = m.nr_assign(counter, inc);
        m.series_insert(body, position, inc_assign);

        if array {
            let offset = m.nr_var(it, false, "");
            m.func_push_symbol(parent, offset);
            let sv = m.nr_var_value(source);
            let length = irtools::call(m, len_func.unwrap(), vec![sv]);
            let end = m.nr_flow_instr(setup, length);
            let sv = m.nr_var_value(source);
            let ov = m.nr_var_value(offset);
            let load = m.getitem(sv, ov).expect("ndarray __getitem__ not found");
            let a = m.nr_assign(element, load);
            m.series_insert(body, 0, a);
            let src = m.src_info(lp.into());
            let zero = m.get_int(0);
            let r = m.n(
                src,
                ValueKind::ImperativeForFlow(ImperativeForFlow { start: zero, step: 1, end, body, var: offset, schedule: None }),
            );
            m.replace_all_value(lp, r);
        } else {
            let mut iter = m.nr_var_value(source);
            if let Some((f, expr)) = iter_info {
                let arg0 = m.func_args(f)[0];
                let mut remaps = FxHashMap::default();
                remaps.insert(m.var_uid(arg0), source);
                let mut clone = CloneVisitor::new(false);
                iter = clone.clone_value_into(m, expr, Some(parent), &remaps);
            }
            // Preserve direct __iter__(source) calls so list lowering can recognize them.
            let direct = irtools::is_call_of(m, iter, magic::ITER, 1, None, false)
                && m.as_call_instr(iter).and_then(|c| irtools::get_var(m, c.args[0])) == Some(m.actual_var(source));
            if direct {
                let front = m.as_call_instr(iter).unwrap().args[0];
                let sv = m.nr_var_value(source);
                let fi = m.nr_flow_instr(setup, sv);
                m.replace_all_value(front, fi);
            } else {
                iter = m.nr_flow_instr(setup, iter);
            }
            let f = m.as_for_flow_mut(lp).unwrap();
            f.iter = iter;
            f.var = element;
        }
    }
}

operator_pass!(EnumerateOptimization);
