//! `PassManagerTest.HonorsDisabledPassesAtRegistration`.

use std::cell::Cell;
use std::rc::Rc;

use typhon_common::Options;
use typhon_ir::pass::{Analysis, AnalysisResult, Pass, PassContext};
use typhon_ir::Module;
use typhon_ir_transform::manager::PassManager;

const ANALYSIS_KEY: &str = "**test_analysis**";
const PASS_KEY: &str = "**test_pass**";

struct DummyResult;
impl AnalysisResult for DummyResult {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[derive(Default)]
struct Counters {
    counter: Cell<i32>,
    analysis_run: Cell<i32>,
    pass_run: Cell<i32>,
}

struct DummyAnalysis(Rc<Counters>);
impl Analysis for DummyAnalysis {
    fn key(&self) -> String {
        ANALYSIS_KEY.into()
    }
    fn run(&mut self, _m: &Module, _ctx: &dyn PassContext) -> Box<dyn AnalysisResult> {
        let c = &self.0;
        c.analysis_run.set(c.counter.get());
        c.counter.set(c.counter.get() + 1);
        Box::new(DummyResult)
    }
}

struct DummyPass {
    c: Rc<Counters>,
    required: String,
    skip: bool,
}
impl Pass for DummyPass {
    fn key(&self) -> String {
        PASS_KEY.into()
    }
    fn should_run(&self, _m: &Module) -> bool {
        !self.skip
    }
    fn run(&mut self, _m: &mut Module, ctx: &dyn PassContext) {
        self.c.pass_run.set(self.c.counter.get());
        self.c.counter.set(self.c.counter.get() + 1);
        assert!(ctx.get_analysis::<DummyResult>(&self.required).is_some());
    }
}

fn empty_manager() -> PassManager {
    let mut o = Options::default();
    o.pmempty = true;
    PassManager::new(o)
}
fn pass(c: &Rc<Counters>, skip: bool) -> Box<dyn Pass> {
    Box::new(DummyPass { c: c.clone(), required: ANALYSIS_KEY.into(), skip })
}
fn s(k: &str) -> Vec<String> {
    vec![k.to_string()]
}

#[test]
fn pass_manager_no_invalidations() {
    let c = Rc::new(Counters::default());
    let mut m = Module::new("test");
    let mut pm = empty_manager();
    pm.register_analysis(Box::new(DummyAnalysis(c.clone())), vec![]);
    pm.register_pass(pass(&c, false), "", s(ANALYSIS_KEY), vec![]);
    pm.run(&mut m);
    assert_eq!(c.analysis_run.get(), 0);
    assert_eq!(c.pass_run.get(), 1);
}

#[test]
fn pass_manager_skips_inapplicable_passes() {
    let c = Rc::new(Counters::default());
    let mut m = Module::new("test");
    let mut pm = empty_manager();
    pm.register_analysis(Box::new(DummyAnalysis(c.clone())), vec![]);
    pm.register_pass(pass(&c, true), "", s(ANALYSIS_KEY), s(ANALYSIS_KEY));
    pm.run(&mut m);
    assert_eq!(c.counter.get(), 0);

    pm.register_pass(pass(&c, false), "", s(ANALYSIS_KEY), vec![]);
    pm.register_pass(pass(&c, true), "", s(ANALYSIS_KEY), s(ANALYSIS_KEY));
    pm.register_pass(pass(&c, false), "", s(ANALYSIS_KEY), vec![]);
    pm.run(&mut m);
    assert_eq!(c.counter.get(), 3);
    assert_eq!(c.analysis_run.get(), 0);
}

#[test]
fn pass_manager_invalidations() {
    let c = Rc::new(Counters::default());
    let mut m = Module::new("test");
    let mut pm = empty_manager();
    pm.register_analysis(Box::new(DummyAnalysis(c.clone())), vec![]);
    pm.register_pass(pass(&c, false), "", s(ANALYSIS_KEY), s(ANALYSIS_KEY));
    pm.register_pass(pass(&c, false), "", s(ANALYSIS_KEY), vec![]);
    pm.run(&mut m);
    assert_eq!(c.analysis_run.get(), 2);
    assert_eq!(c.pass_run.get(), 3);
}

#[test]
fn unique_keys_and_insert_before() {
    let c = Rc::new(Counters::default());
    let mut pm = empty_manager();
    pm.register_analysis(Box::new(DummyAnalysis(c.clone())), vec![]);
    let k1 = pm.register_pass(pass(&c, false), "", vec![], vec![]);
    let k2 = pm.register_pass(pass(&c, false), "", vec![], vec![]);
    let k3 = pm.register_pass(pass(&c, false), &k1, vec![], vec![]);
    assert_eq!(k1, PASS_KEY);
    assert_eq!(k2, format!("{PASS_KEY}:2"));
    assert_eq!(k3, format!("{PASS_KEY}:3"));
    assert_eq!(pm.execution_order(), &[k3, k1, k2]);
    assert!(pm.has_pass(PASS_KEY) && pm.has_analysis(ANALYSIS_KEY));
}

#[test]
fn disabled_at_registration() {
    let c = Rc::new(Counters::default());
    let mut o = Options::default();
    o.pmempty = true;
    o.disabled = vec![PASS_KEY.into()];
    let mut pm = PassManager::new(o);
    assert_eq!(pm.register_pass(pass(&c, false), "", vec![], vec![]), "");
    assert!(!pm.has_pass(PASS_KEY));
    pm.options_mut().disabled.clear();
    assert!(!pm.is_disabled(PASS_KEY));
}

/// `PassManagerTest.HonorsDisabledPassesAtRegistration`.
#[test]
fn honors_disabled_passes_at_registration() {
    let mut o = Options::default();
    o.debug = false;
    let enabled = PassManager::new(o.clone());
    assert!(enabled.has_pass("core-folding-pass-group"));
    assert!(enabled.has_pass("core-devirtualize"));
    assert!(enabled.has_pass("core-numpy-inline"));
    assert!(enabled.has_pass("core-numpy-fusion"));
    assert!(enabled.has_pass("core-numpy-lifetime"));
    assert!(!enabled.is_disabled("core-numpy-fusion"));

    o.disabled = vec!["core-numpy-fusion".into(), "core-numpy-lifetime".into(), "not-a-pass".into(), "core-devirtualize".into()];
    let mut disabled = PassManager::new(o);
    assert!(!disabled.has_pass("core-numpy-fusion"));
    assert!(!disabled.has_pass("core-numpy-lifetime"));
    assert!(!disabled.has_pass("core-devirtualize"));
    assert!(disabled.has_pass("core-numpy-inline"));
    assert!(disabled.is_disabled("not-a-pass"));
    disabled.options_mut().disabled.clear();
    assert!(!disabled.is_disabled("core-numpy-fusion"));
    assert!(!disabled.has_pass("core-numpy-fusion"));
}

#[test]
fn standard_pipelines_order() {
    let mut o = Options::default();
    o.debug = true;
    let dbg = PassManager::new(o.clone());
    assert_eq!(dbg.execution_order()[0], "core-pipeline-lowering");
    assert!(!dbg.has_pass("core-folding-pass-group"));

    o.debug = false;
    let rel = PassManager::new(o);
    let order = rel.execution_order();
    assert_eq!(order[0], "core-pythonic-dict-arithmetic-opt");
    assert!(order.iter().any(|k| k == "core-folding-pass-group:2"));
    assert_eq!(order.last().unwrap(), "core-devirtualize");
}
