//! Port of codon/cir/transform/lowering/await.{h,cpp}

use typhon_ir::util::cloning::CloneVisitor;
use typhon_ir::util::irtools;
use typhon_ir::util::operator::{Operator, OperatorState};
use typhon_ir::{Module, TypeId, TypeKind, ValueId};

use super::{operator_pass, operator_state};

fn is_future(m: &Module, t: TypeId) -> bool {
    m.type_name(t).starts_with("std.asyncio.Future.")
}
fn is_task(m: &Module, t: TypeId) -> bool {
    m.type_name(t).starts_with("std.asyncio.Task.")
}

/// Lowers `await` instructions on futures, tasks and coroutines.
#[derive(Default)]
pub struct AwaitLowering {
    state: OperatorState,
}

impl AwaitLowering {
    pub const KEY: &'static str = "core-await-lowering";
    pub fn new() -> Self {
        Self::default()
    }
}

impl Operator for AwaitLowering {
    operator_state!();
    fn handle_await_instr(&mut self, m: &mut Module, v: ValueId) {
        let aw = m.as_await_instr(v).unwrap().clone();
        let value = aw.value;
        let value_type = m.value_type_mut(value);
        let mut cv = CloneVisitor::new(false);
        let parent = self.parent_func(m);
        let am = "std.asyncio";

        if is_future(m, value_type) || is_task(m, value_type) {
            let get_result =
                m.get_or_realize_method(value_type, "result", &[value_type], &[]).expect("get-result method not found");
            let wait_on = m.get_or_realize_func("_wait_on", &[value_type], &[], am).expect("wait-on function not found");
            let cancel = m
                .get_or_realize_func("_cancel_checkpoint", &[], &[], am)
                .expect("cancel-checkpoint function not found");
            // cancel_checkpoint(); if _wait_on(f): yield; cancel_checkpoint()
            // => future.result()
            let series = m.nr_series_flow();
            let cl = cv.clone_value(m, value);
            let fut = irtools::make_var(m, cl, series, parent, false);
            let c0 = irtools::call(m, cancel, vec![]);
            m.series_push(series, c0);
            let fv = m.nr_var_value(fut);
            let w = irtools::call(m, wait_on, vec![fv]);
            let y = m.nr_yield(None, false);
            let c1 = irtools::call(m, cancel, vec![]);
            let s = irtools::series(m, &[y, c1]).unwrap();
            let iff = m.nr_if_flow(w, s, None);
            m.series_push(series, iff);
            let fv = m.nr_var_value(fut);
            let r = irtools::call(m, get_result, vec![fv]);
            let rep = m.nr_flow_instr(series, r);
            m.replace_all_value(v, rep);
        } else if let TypeKind::Generator { base } = *m.type_kind(value_type) {
            let promise =
                m.get_or_realize_func("_promise", &[value_type], &[], am).expect("promise function not found");
            // for _ in coro: yield  => coro.__promise__()
            let series = m.nr_series_flow();
            let cl = cv.clone_value(m, value);
            let coro = irtools::make_var(m, cl, series, parent, false);
            let var = m.nr_var(base, false, "");
            if let Some(p) = parent {
                m.func_push_symbol(p, var);
            }
            let body = if aw.generator {
                let requeue =
                    m.get_or_realize_func("_requeue", &[], &[], am).expect("requeue function not found");
                let c = irtools::call(m, requeue, vec![]);
                let y = m.nr_yield(None, false);
                irtools::series(m, &[c, y]).unwrap()
            } else {
                let y = m.nr_yield(None, false);
                irtools::series(m, &[y]).unwrap()
            };
            let cvv = m.nr_var_value(coro);
            let lp = m.nr_for_flow(cvv, body, var);
            m.series_push(series, lp);
            let cvv = m.nr_var_value(coro);
            let r = irtools::call(m, promise, vec![cvv]);
            let rep = m.nr_flow_instr(series, r);
            m.replace_all_value(v, rep);
        } else {
            panic!("unexpected value type '{}' in await instruction", m.type_name(value_type));
        }
    }
}

operator_pass!(AwaitLowering);
