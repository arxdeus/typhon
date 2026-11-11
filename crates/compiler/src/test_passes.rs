//! Test-only IR passes from `test/main.cpp` (`TestOutliner`, `TestInliner`,
//! `EscapeValidator`). Registered by the driver when `TYPHON_TEST_MODE=1`.

use typhon_ir::pass::{Pass, PassContext, run_operator_pass};
use typhon_ir::util::irtools;
use typhon_ir::util::operator::{Operator, OperatorState};
use typhon_ir::{FuncId, Module, NodeRef, ValueId};
use typhon_ir_analyze::dataflow::capture::{CaptureInfo, CaptureResult};
use typhon_ir_analyze::util::inlining::inline_call;
use typhon_ir_analyze::util::outlining::outline_region;
use typhon_ir_transform::manager::PassManager;

fn unmangled(m: &Module, f: FuncId) -> String {
    m.func(f).map(|d| d.unmangled_name.clone()).unwrap_or_default()
}

/// Port of `TestOutliner`.
#[derive(Default)]
pub struct TestOutliner {
    state: OperatorState,
    successes: i64,
    failures: i64,
    successes_return: Option<ValueId>,
    failures_return: Option<ValueId>,
}

impl TestOutliner {
    pub const KEY: &'static str = "test-outliner-pass";

    fn update_returns(&self, m: &mut Module) {
        for (r, n) in [(self.successes_return, self.successes), (self.failures_return, self.failures)] {
            if let Some(r) = r {
                let c = m.get_int(n);
                if let Some(ri) = m.as_return_instr_mut(r) {
                    ri.value = Some(c);
                }
            }
        }
    }
}

impl Operator for TestOutliner {
    fn state(&self) -> &OperatorState {
        &self.state
    }
    fn state_mut(&mut self) -> &mut OperatorState {
        &mut self.state
    }
    fn handle_series_flow(&mut self, m: &mut Module, v: ValueId) {
        let items = m.as_series_flow(v).map(|s| s.series.clone()).unwrap_or_default();
        let (mut begin, mut end) = (None, None);
        for (i, it) in items.iter().enumerate() {
            if begin.is_none() && irtools::is_call_of(m, *it, "__outline_begin__", -1, None, false) {
                begin = Some(i);
            } else if end.is_none() && irtools::is_call_of(m, *it, "__outline_end__", -1, None, false) {
                end = Some(i);
            }
        }
        if let (Some(b), Some(e)) = (begin, end) {
            let Some(parent) = self.parent_func(m) else { return };
            let r = outline_region(m, parent, v, b, e, false, false, false);
            if r.func.is_some() {
                self.successes += 1;
            } else {
                self.failures += 1;
            }
            self.update_returns(m);
        }
    }
    fn handle_return_instr(&mut self, m: &mut Module, v: ValueId) {
        let Some(pf) = self.parent_func(m) else { return };
        let name = unmangled(m, pf);
        if name == "__outline_successes__" {
            let c = m.get_int(self.successes);
            if let Some(ri) = m.as_return_instr_mut(v) {
                ri.value = Some(c);
            }
            self.successes_return = Some(v);
        }
        if name == "__outline_failures__" {
            let c = m.get_int(self.failures);
            if let Some(ri) = m.as_return_instr_mut(v) {
                ri.value = Some(c);
            }
            self.failures_return = Some(v);
        }
    }
}

impl Pass for TestOutliner {
    fn key(&self) -> String {
        Self::KEY.into()
    }
    fn run(&mut self, m: &mut Module, _ctx: &dyn PassContext) {
        run_operator_pass(self, m);
    }
}

/// Port of `TestInliner`.
#[derive(Default)]
pub struct TestInliner {
    state: OperatorState,
}

impl TestInliner {
    pub const KEY: &'static str = "test-inliner-pass";
}

impl Operator for TestInliner {
    fn state(&self) -> &OperatorState {
        &self.state
    }
    fn state_mut(&mut self) -> &mut OperatorState {
        &mut self.state
    }
    fn handle_call_instr(&mut self, m: &mut Module, v: ValueId) {
        let Some(callee) = m.as_call_instr(v).map(|c| c.callee) else { return };
        let Some(f) = irtools::get_func(m, callee) else { return };
        if m.func_body(f).is_none() {
            return;
        }
        let it = m.get_int_type();
        let neg = m.get_or_realize_method(it, typhon_ir::magic::NEG, &[it], &[]);
        let name = unmangled(m, f);
        if !name.contains("inline_me") {
            return;
        }
        let aggressive = name.contains("aggressive");
        let res = inline_call(m, v, aggressive);
        let Some(mut result) = res.result else { return };
        if let Some(pf) = self.parent_func(m) {
            for var in res.new_vars {
                m.func_push_symbol(pf, var);
            }
        }
        let rt = m.value_type_mut(result);
        if m.type_is(rt, it)
            && let Some(neg) = neg
        {
            result = irtools::call(m, neg, vec![result]);
        }
        m.replace_all_value(v, result);
    }
}

impl Pass for TestInliner {
    fn key(&self) -> String {
        Self::KEY.into()
    }
    fn run(&mut self, m: &mut Module, _ctx: &dyn PassContext) {
        run_operator_pass(self, m);
    }
}

/// Port of `PartitionArgsByEscape`.
#[derive(Default)]
struct PartitionArgsByEscape {
    state: OperatorState,
    expected: Vec<CaptureInfo>,
    calls: Vec<ValueId>,
}

impl Operator for PartitionArgsByEscape {
    fn state(&self) -> &OperatorState {
        &self.state
    }
    fn state_mut(&mut self) -> &mut OperatorState {
        &mut self.state
    }
    fn handle_call_instr(&mut self, m: &mut Module, v: ValueId) {
        let Some(call) = m.as_call_instr(v).cloned() else { return };
        let Some(f) = irtools::get_func(m, call.callee) else { return };
        if unmangled(m, f) != "expect_capture" {
            return;
        }
        assert!(call.args.len() == 3, "bad escape-test call (size)");
        let rc = m.as_bool_const(call.args[0]).map(|c| c.val);
        let ec = m.as_bool_const(call.args[1]).map(|c| c.val);
        let (Some(rc), Some(ec)) = (rc, ec) else { panic!("bad escape-test call (arg types)") };
        let tuple = m.as_call_instr(call.args[2]).cloned().expect("last escape-test call argument should be a const tuple literal");
        let mut info = CaptureInfo { return_captures: rc, extern_captures: ec, ..Default::default() };
        for a in tuple.args {
            let c = m.as_int_const(a).expect("final args should be int");
            info.arg_captures.push(c.val as u32);
        }
        self.expected.push(info);
        self.calls.push(v);
    }
}

/// Port of `EscapeValidator`.
pub struct EscapeValidator {
    cap_analysis_key: String,
}

impl EscapeValidator {
    pub const KEY: &'static str = "test-escape-validator-pass";
    pub fn new(cap_analysis_key: impl Into<String>) -> Self {
        Self { cap_analysis_key: cap_analysis_key.into() }
    }
}

impl Pass for EscapeValidator {
    fn key(&self) -> String {
        Self::KEY.into()
    }
    fn run(&mut self, m: &mut Module, ctx: &dyn PassContext) {
        let Some(cap) = ctx.get_analysis::<CaptureResult>(&self.cap_analysis_key) else { return };
        for var in m.vars() {
            if !m.is_func(var) {
                continue;
            }
            let mut pabe = PartitionArgsByEscape::default();
            pabe.reset();
            pabe.process(m, NodeRef::Var(var));
            if pabe.expected.is_empty() {
                continue;
            }
            let uid = m.var_uid(var);
            let received = cap.results.get(&uid).expect("function not found in capture results");
            assert!(pabe.expected.len() == received.len(), "size mismatch in capture results");
            for (i, exp) in pabe.expected.iter().enumerate() {
                let mut exp = exp.clone();
                let mut got = received[i].clone();
                exp.arg_captures.sort();
                got.arg_captures.sort();
                let good = exp.return_captures == got.return_captures
                    && exp.extern_captures == got.extern_captures
                    && exp.arg_captures == got.arg_captures;
                let b = m.get_bool(good);
                m.replace_all_value(pabe.calls[i], b);
            }
        }
    }
}

/// Register the three test passes like `runCompileAndRun` in `test/main.cpp`.
pub fn register_test_passes(pm: &mut PassManager) {
    use typhon_ir_analyze::dataflow::capture::CaptureAnalysis;
    use typhon_ir_analyze::dataflow::dominator::DominatorAnalysis;
    use typhon_ir_analyze::dataflow::reaching::RDAnalysis;
    pm.register_pass_simple(Box::new(TestOutliner::default()));
    pm.register_pass_simple(Box::new(TestInliner::default()));
    // PORT-NOTE: in debug mode the standard pipeline has no RD/dominator analyses
    // registered; C++ silently relies on the keys existing. Register them if missing.
    let cfg = typhon_ir_analyze::dataflow::cfg::CFAnalysis::KEY;
    if !pm.has_analysis(cfg) {
        pm.register_analysis(Box::new(typhon_ir_analyze::dataflow::cfg::CFAnalysis::default()), vec![]);
    }
    if !pm.has_analysis(RDAnalysis::KEY) {
        pm.register_analysis(Box::new(RDAnalysis::new(cfg.to_string())), vec![cfg.to_string()]);
    }
    if !pm.has_analysis(DominatorAnalysis::KEY) {
        pm.register_analysis(Box::new(DominatorAnalysis::new(cfg.to_string())), vec![cfg.to_string()]);
    }
    let cap = pm.register_analysis(
        Box::new(CaptureAnalysis::new(RDAnalysis::KEY.to_string(), DominatorAnalysis::KEY.to_string())),
        vec![RDAnalysis::KEY.to_string(), DominatorAnalysis::KEY.to_string()],
    );
    pm.register_pass(Box::new(EscapeValidator::new(cap.clone())), "", vec![cap], vec![]);
}
