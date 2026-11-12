//! Read-only traversal with the exact visiting order of `util::Operator`
//! (pre-order, `childrenFirst == false`), for analyses that only need `&Module`.
//!
//! Note: the reference implementation analyses `const_cast` the module to run an `Operator`; in Rust
//! `typhon_ir::util::operator::Operator` needs `&mut Module`, so analyses use this walker.

use rustc_hash::FxHashSet;
use typhon_ir::{FuncId, Id, Module, ValueId};

/// Walker state (`seen` set of `Operator`).
#[derive(Default, Debug)]
pub struct ConstWalker {
    seen: FxHashSet<Id>,
}

impl ConstWalker {
    pub fn new() -> Self {
        Self::default()
    }

    /// `reset()`.
    pub fn reset(&mut self) {
        self.seen.clear();
    }

    /// `visit(Module*)`: main function, then every module variable.
    pub fn walk_module(&mut self, m: &Module, cb: &mut dyn FnMut(ValueId)) {
        self.walk_func(m, m.main_func(), cb);
        for v in m.vars() {
            if m.is_var_alive(v) {
                self.walk_func(m, v, cb);
            }
        }
    }

    /// `visit(BodiedFunc*)` (no-op for other variables).
    pub fn walk_func(&mut self, m: &Module, f: FuncId, cb: &mut dyn FnMut(ValueId)) {
        let f = m.actual_var(f);
        if let Some(body) = m.func(f).and_then(|fd| fd.body()) {
            self.seen.insert(m.value_uid(body));
            self.walk_value(m, body, cb);
        }
    }

    /// `process(value)`.
    pub fn walk_value(&mut self, m: &Module, v: ValueId, cb: &mut dyn FnMut(ValueId)) {
        let v = m.actual_value(v);
        cb(v);
        if let Some(s) = m.as_series_flow(v) {
            // processSeriesFlowChildren: no `seen` check.
            let items = s.series.clone();
            for c in items {
                self.walk_value(m, c, cb);
            }
            return;
        }
        let children = match m.as_call_instr(v) {
            Some(call) => {
                let mut c = call.args.clone();
                c.push(call.callee);
                c
            }
            None => m.used_values(v.into()),
        };
        for c in children {
            if self.seen.insert(m.value_uid(c)) {
                self.walk_value(m, c, cb);
            }
        }
    }
}
