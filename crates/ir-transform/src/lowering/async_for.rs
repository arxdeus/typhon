
use typhon_ir::util::cloning::CloneVisitor;
use typhon_ir::util::irtools;
use typhon_ir::util::operator::{Operator, OperatorState};
use typhon_ir::{Module, ValueId};

use super::{operator_pass, operator_state};

/// Lowers `async for` loops into explicit coroutine driving loops.
#[derive(Default)]
pub struct AsyncForLowering {
    state: OperatorState,
}

impl AsyncForLowering {
    pub const KEY: &'static str = "core-async-for-lowering";
    pub fn new() -> Self {
        Self::default()
    }
}

impl Operator for AsyncForLowering {
    operator_state!();
    fn handle_for_flow(&mut self, m: &mut Module, v: ValueId) {
        let ff = m.as_for_flow(v).unwrap().clone();
        if !ff.is_async {
            return;
        }
        let coro = ff.iter;
        let coro_type = m.value_type_mut(coro);
        let mut cv = CloneVisitor::new(false);
        let am = "std.asyncio";
        let promise = m.get_or_realize_func("_promise", &[coro_type], &[], am).expect("promise func not found");
        let done = m.get_or_realize_func("_done", &[coro_type], &[], am).expect("done func not found");
        let resume = m.get_or_realize_func("_resume", &[coro_type], &[], am).expect("resume func not found");
        let curr_task = m.get_or_realize_func("_curr_task", &[], &[], am).expect("curr-task func not found");
        let ct_ret = irtools::return_type(m, curr_task).expect("curr-task return type");
        let waiting = m.get_or_realize_func("_is_waiting", &[ct_ret], &[], am).expect("is-waiting func not found");

        // task = curr_task()
        // if not coro.__done__():
        //     while True:
        //         coro.__resume__()
        //         if coro.__done__(): break
        //         if is_waiting(task): yield
        //         else: i = coro.__promise__(); <BODY>
        let parent = self.parent_func(m);
        let series = m.nr_series_flow();
        let c = irtools::call(m, curr_task, vec![]);
        let task_var = irtools::make_var(m, c, series, parent, false);
        let cc = cv.clone_value(m, coro);
        let coro_var = irtools::make_var(m, cc, series, parent, false);

        let cvv = m.nr_var_value(coro_var);
        let done0 = irtools::call(m, done, vec![cvv]);
        let not_done = m.unary_op(done0, typhon_ir::magic::INVERT).expect("bool __invert__ not found");

        let cvv = m.nr_var_value(coro_var);
        let res = irtools::call(m, resume, vec![cvv]);
        let cvv = m.nr_var_value(coro_var);
        let done1 = irtools::call(m, done, vec![cvv]);
        let brk = m.nr_break(None);
        let brk_s = irtools::series(m, &[brk]).unwrap();
        let if_done = m.nr_if_flow(done1, brk_s, None);

        let tvv = m.nr_var_value(task_var);
        let w = irtools::call(m, waiting, vec![tvv]);
        let y = m.nr_yield(None, false);
        let y_s = irtools::series(m, &[y]).unwrap();
        let cvv = m.nr_var_value(coro_var);
        let p = irtools::call(m, promise, vec![cvv]);
        let assign = m.nr_assign(ff.var, p);
        let body = cv.clone_value(m, ff.body);
        let else_s = irtools::series(m, &[assign, body]).unwrap();
        let if_wait = m.nr_if_flow(w, y_s, Some(else_s));

        let loop_body = irtools::series(m, &[res, if_done, if_wait]).unwrap();
        let t = m.get_bool(true);
        let wl = m.nr_while_flow(t, loop_body);
        let wl_s = irtools::series(m, &[wl]).unwrap();
        let outer = m.nr_if_flow(not_done, wl_s, None);
        m.series_push(series, outer);
        m.replace_all_value(v, series);
    }
}

operator_pass!(AsyncForLowering);
