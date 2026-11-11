//! Port of codon/cir/transform/cleanup/dead_code.{h,cpp}

use typhon_common::log_flag;
use typhon_ir::pass::{Pass, PassContext, run_operator_pass};
use typhon_ir::util::cloning::CloneVisitor;
use typhon_ir::util::format::format_value;
use typhon_ir::util::operator::{Operator, OperatorState};
use typhon_ir::{Module, ValueId};
use typhon_ir_analyze::module::side_effect::SideEffectResult;

/// Cleanup pass that removes dead code.
pub struct DeadCodeCleanupPass {
    state: OperatorState,
    side_effects_key: String,
    num_replacements: i32,
}

impl DeadCodeCleanupPass {
    pub const KEY: &'static str = "core-cleanup-dce";

    pub fn new(side_effects_key: impl Into<String>) -> Self {
        Self { state: OperatorState::new(false), side_effects_key: side_effects_key.into(), num_replacements: 0 }
    }

    /// `getNumReplacements()`.
    pub fn num_replacements(&self) -> i32 {
        self.num_replacements
    }

    fn do_replacement(&mut self, m: &mut Module, og: ValueId, v: ValueId) {
        self.num_replacements += 1;
        m.replace_all_value(og, v);
    }
}

struct Runner<'a, 'c> {
    pass: &'a mut DeadCodeCleanupPass,
    ctx: &'a (dyn PassContext + 'c),
}

impl Operator for Runner<'_, '_> {
    fn state(&self) -> &OperatorState {
        &self.pass.state
    }
    fn state_mut(&mut self) -> &mut OperatorState {
        &mut self.pass.state
    }

    fn handle_series_flow(&mut self, m: &mut Module, v: ValueId) {
        let r = self
            .ctx
            .get_analysis::<SideEffectResult>(&self.pass.side_effects_key)
            .expect("side effect analysis result missing");
        let series = m.as_series_flow(v).unwrap().series.clone();
        let mut kept = Vec::with_capacity(series.len());
        for x in series {
            if !r.has_side_effect(m, x) {
                log_flag!(IR, "[{}] no side effect, deleting: {}", DeadCodeCleanupPass::KEY, format_value(m, x));
                self.pass.num_replacements += 1;
            } else {
                kept.push(x);
            }
        }
        m.as_series_flow_mut(v).unwrap().series = kept;
    }

    fn handle_flow_instr(&mut self, m: &mut Module, v: ValueId) {
        let fi = m.as_flow_instr(v).unwrap();
        let (flow, val) = (fi.flow, fi.val);
        if m.as_series_flow(flow).is_some_and(|s| s.series.is_empty()) {
            self.pass.do_replacement(m, v, val);
        }
    }

    fn handle_if_flow(&mut self, m: &mut Module, v: ValueId) {
        let f = m.as_if_flow(v).unwrap();
        let (cond, t, fb) = (f.cond, f.true_branch, f.false_branch);
        let Some(c) = m.as_bool_const(cond).map(|c| c.val) else { return };
        let mut cv = CloneVisitor::new(true);
        let rep = if c {
            cv.clone_value(m, t)
        } else if let Some(fb) = fb {
            cv.clone_value(m, fb)
        } else {
            m.nr_series_flow()
        };
        self.pass.do_replacement(m, v, rep);
    }

    fn handle_while_flow(&mut self, m: &mut Module, v: ValueId) {
        let cond = m.as_while_flow(v).unwrap().cond;
        if m.as_bool_const(cond).is_some_and(|c| !c.val) {
            let s = m.nr_series_flow();
            self.pass.do_replacement(m, v, s);
        }
    }

    fn handle_imperative_for_flow(&mut self, m: &mut Module, v: ValueId) {
        let f = m.as_imperative_for_flow(v).unwrap();
        let (start, end, step) = (f.start, f.end, f.step);
        let (Some(s), Some(e)) = (m.as_int_const(start).map(|c| c.val), m.as_int_const(end).map(|c| c.val)) else { return };
        if (step < 0 && s <= e) || (step > 0 && s >= e) {
            let sf = m.nr_series_flow();
            self.pass.do_replacement(m, v, sf);
        }
    }

    fn handle_ternary_instr(&mut self, m: &mut Module, v: ValueId) {
        let t = m.as_ternary_instr(v).unwrap();
        let (cond, tv, fv) = (t.cond, t.true_value, t.false_value);
        let Some(c) = m.as_bool_const(cond).map(|c| c.val) else { return };
        let rep = CloneVisitor::new(true).clone_value(m, if c { tv } else { fv });
        self.pass.do_replacement(m, v, rep);
    }
}

impl Operator for DeadCodeCleanupPass {
    fn state(&self) -> &OperatorState {
        &self.state
    }
    fn state_mut(&mut self) -> &mut OperatorState {
        &mut self.state
    }
}

impl Pass for DeadCodeCleanupPass {
    fn key(&self) -> String {
        Self::KEY.to_string()
    }
    fn run(&mut self, m: &mut Module, ctx: &dyn PassContext) {
        self.num_replacements = 0;
        let mut r = Runner { pass: self, ctx };
        run_operator_pass(&mut r, m);
    }
}
