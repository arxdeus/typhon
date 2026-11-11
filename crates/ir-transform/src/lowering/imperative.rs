//! Port of codon/cir/transform/lowering/imperative.{h,cpp}

use typhon_common::mangle::stdlib;
use typhon_ir::util::cloning::CloneVisitor;
use typhon_ir::util::irtools;
use typhon_ir::util::operator::{Operator, OperatorState};
use typhon_ir::{ImperativeForFlow, Module, ValueId, ValueKind, magic};

use super::{operator_pass, operator_state};

fn get_range_iter(m: &mut Module, iter: ValueId) -> Option<ValueId> {
    let call = m.as_call_instr(iter)?;
    if call.args.len() != 1 {
        return None;
    }
    let (callee, arg) = (call.callee, call.args[0]);
    let f = irtools::get_func(m, callee)?;
    if m.func(f)?.unmangled_name != magic::ITER {
        return None;
    }
    let range_call = m.as_call_instr(arg)?;
    let nf = irtools::get_func(m, range_call.callee)?;
    let fd = m.func(nf)?;
    if fd.unmangled_name != magic::NEW {
        return None;
    }
    let parent = fd.parent_type?;
    let range_type = m.get_or_realize_type(&stdlib().range, &[])?;
    if m.type_name(parent) != m.type_name(range_type) {
        return None;
    }
    Some(arg)
}

fn get_list_iter(m: &mut Module, iter: ValueId) -> Option<ValueId> {
    let call = m.as_call_instr(iter)?;
    if call.args.len() != 1 {
        return None;
    }
    let (callee, list) = (call.callee, call.args[0]);
    let f = irtools::get_func(m, callee)?;
    if m.func(f)?.unmangled_name != magic::ITER {
        return None;
    }
    let lt = m.value_type_mut(list);
    if !m.type_name(lt).starts_with(&format!("{}[", stdlib().list)) {
        return None;
    }
    Some(list)
}

/// Lowers `for` loops over `range(...)` and lists to imperative for loops.
#[derive(Default)]
pub struct ImperativeForFlowLowering {
    state: OperatorState,
}

impl ImperativeForFlowLowering {
    pub const KEY: &'static str = "core-imperative-for-lowering";
    pub fn new() -> Self {
        Self::default()
    }
}

impl Operator for ImperativeForFlowLowering {
    operator_state!();
    fn handle_for_flow(&mut self, m: &mut Module, v: ValueId) {
        let ff = m.as_for_flow(v).unwrap().clone();
        let src = m.src_info(v.into());
        let sched = ff.schedule.clone();

        if let Some(range_call) = get_range_iter(m, ff.iter) {
            let args = m.as_call_instr(range_call).unwrap().args.clone();
            let mut cv = CloneVisitor::new(false);
            let (start, end, step) = match args.len() {
                1 => (m.get_int(0), cv.clone_value(m, args[0]), 1),
                2 => (cv.clone_value(m, args[0]), cv.clone_value(m, args[1]), 1),
                3 => {
                    let Some(step) = m.as_int_const(args[2]).map(|c| c.val) else { return };
                    (cv.clone_value(m, args[0]), cv.clone_value(m, args[1]), step)
                }
                _ => panic!("unknown range constructor"),
            };
            if step == 0 {
                return;
            }
            let r = m.n(
                src,
                ValueKind::ImperativeForFlow(ImperativeForFlow {
                    start,
                    step,
                    end,
                    body: ff.body,
                    var: ff.var,
                    schedule: sched,
                }),
            );
            m.replace_all_value(v, r);
        } else if let Some(list) = get_list_iter(m, ff.iter) {
            // for a in list: body  ==>  v = list; n = v._len; p = v._ptr;
            // imp_for i in range(0, n, 1): a = p[i]; body
            let parent = self.parent_func(m);
            let series = m.n(src.clone(), ValueKind::SeriesFlow(Default::default()));
            let list_var = irtools::make_var(m, list, series, parent, false);
            let lv = m.nr_var_value(list_var);
            let len_val = m.nr_extract(lv, "_len");
            let len_var = irtools::make_var(m, len_val, series, parent, false);
            let lv = m.nr_var_value(list_var);
            let ptr_val = m.nr_extract(lv, "_ptr");
            let ptr_var = irtools::make_var(m, ptr_val, series, parent, false);

            let body = ff.body;
            assert!(m.as_series_flow(body).is_some(), "loop body is not a series flow [{}]", src);
            let int_t = m.get_int_type();
            let new_loop_var = m.nr_var(int_t, false, "");
            if let Some(p) = parent {
                m.func_push_symbol(p, new_loop_var);
            }
            let zero = m.get_int(0);
            let end = m.nr_var_value(len_var);
            let rep = m.n(
                src,
                ValueKind::ImperativeForFlow(ImperativeForFlow {
                    start: zero,
                    step: 1,
                    end,
                    body,
                    var: new_loop_var,
                    schedule: sched,
                }),
            );
            m.series_push(series, rep);
            let pv = m.nr_var_value(ptr_var);
            let iv = m.nr_var_value(new_loop_var);
            let item = m.getitem(pv, iv).expect("pointer __getitem__ not found");
            let assign = m.nr_assign(ff.var, item);
            m.series_insert(body, 0, assign);
            m.replace_all_value(v, series);
        }
    }
}

operator_pass!(ImperativeForFlowLowering);
