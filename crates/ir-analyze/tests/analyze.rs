//! side effects, global vars, inlining and outlining.

use std::rc::Rc;

use pretty_assertions::assert_eq;
use rustc_hash::FxHashSet;
use typhon_ir::pass::{Analysis, NullPassContext};
use typhon_ir::{Id, Module, ValueId, VarId, reset_id};
use typhon_ir_analyze::dataflow::capture::{CaptureAnalysis, CaptureResult};
use typhon_ir_analyze::dataflow::cfg::{CFAnalysis, CFResult, SyntheticAssignKind, build_cf_graph};
use typhon_ir_analyze::dataflow::dominator::{DominatorAnalysis, DominatorInspector, DominatorResult};
use typhon_ir_analyze::dataflow::reaching::{RDAnalysis, RDInspector, RDResult};
use typhon_ir_analyze::module::global_vars::{GlobalVarsAnalyses, GlobalVarsResult};
use typhon_ir_analyze::module::side_effect::{SideEffectAnalysis, SideEffectResult};
use typhon_ir_analyze::util::inlining::inline_call;
use typhon_ir_analyze::util::outlining::outline_series;

fn setup() -> Module {
    reset_id();
    Module::new("test")
}

fn rdset(m: &Module, rd: &RDInspector, var: VarId, loc: ValueId) -> FxHashSet<Id> {
    rd.get_reaching_definitions(m, var, loc).iter().filter(|d| d.known()).map(|d| d.assignee_uid().unwrap()).collect()
}

fn func_with_body(m: &mut Module) -> (VarId, ValueId, VarId) {
    let f = m.nr_bodied_func("test_f");
    let b = m.nr_series_flow();
    m.set_func_body(f, Some(b));
    let it = m.get_int_type();
    let v = m.nr_var(it, false, "v");
    m.func_push_symbol(f, v);
    (f, b, v)
}

fn rd_of(m: &Module, f: VarId) -> RDInspector {
    let mut rd = RDInspector::new(Rc::new(build_cf_graph(m, f)));
    rd.analyze(m);
    rd
}

#[test]
fn dominator_analysis_simple() {
    let mut m = setup();
    let (f, b, _) = func_with_body(&mut m);
    let start = m.get_bool(false);
    let end = m.get_bool(false);
    m.series_push(b, start);
    m.series_push(b, end);
    let mut dom = DominatorInspector::new(Rc::new(build_cf_graph(&m, f)));
    dom.analyze();
    assert!(dom.is_dominated(&m, end, start));
    assert!(!dom.is_dominated(&m, start, end));
}

#[test]
fn dominator_analysis_ternary() {
    let mut m = setup();
    let (f, b, _) = func_with_body(&mut m);
    let start = m.get_bool(false);
    let middle = m.get_bool(false);
    let c = m.get_bool(true);
    let fv = m.get_bool(true);
    let end = m.nr_ternary(c, middle, fv);
    m.series_push(b, start);
    m.series_push(b, end);
    let mut dom = DominatorInspector::new(Rc::new(build_cf_graph(&m, f)));
    dom.analyze();
    assert!(dom.is_dominated(&m, end, start));
    assert!(dom.is_dominated(&m, middle, start));
    assert!(!dom.is_dominated(&m, start, end));
    assert!(!dom.is_dominated(&m, end, middle));
}

#[test]
fn rd_analysis_simple() {
    let mut m = setup();
    let (f, b, v) = func_with_body(&mut m);
    let first = m.get_int(1);
    let second = m.get_int(2);
    let start = m.get_bool(false);
    let a1 = m.nr_assign(v, first);
    let a2 = m.nr_assign(v, second);
    let end = m.get_bool(false);
    for x in [start, a1, a2, end] {
        m.series_push(b, x);
    }
    let rd = rd_of(&m, f);
    assert_eq!(rdset(&m, &rd, v, start).len(), 0);
    assert_eq!(rdset(&m, &rd, v, a1), [m.value_uid(first)].into_iter().collect());
    assert_eq!(rdset(&m, &rd, v, a2), [m.value_uid(second)].into_iter().collect());
    assert_eq!(rdset(&m, &rd, v, end), [m.value_uid(second)].into_iter().collect());
    assert_eq!(rdset(&m, &rd, v, first).len(), 0);
    assert_eq!(rdset(&m, &rd, v, second).len(), 1);
}

#[test]
fn rd_analysis_indexed_locations() {
    let mut m = setup();
    let (f, b, v) = func_with_body(&mut m);
    let first = m.get_int(1);
    let second = m.get_int(2);
    let shared = m.get_bool(false);
    let a1 = m.nr_assign(v, first);
    let a2 = m.nr_assign(v, second);
    for x in [shared, a1, shared, a2] {
        m.series_push(b, x);
    }
    let rd = rd_of(&m, f);
    assert!(rdset(&m, &rd, v, shared).is_empty());
    assert_eq!(rdset(&m, &rd, v, a1), [m.value_uid(first)].into_iter().collect());
    assert_eq!(rdset(&m, &rd, v, a2), [m.value_uid(second)].into_iter().collect());
    let missing = m.get_bool(true);
    assert!(rdset(&m, &rd, v, missing).is_empty());
    rd.cfg().remap_value_of(&m, missing, a1.into());
    assert_eq!(rdset(&m, &rd, v, missing), [m.value_uid(second)].into_iter().collect());
    let replacement = m.get_int(3);
    m.as_assign_instr_mut(a2).unwrap().rhs = replacement;
    assert_eq!(rdset(&m, &rd, v, a2), [m.value_uid(replacement)].into_iter().collect());
}

#[test]
fn rd_analysis_if_conditional() {
    let mut m = setup();
    let (f, b, v) = func_with_body(&mut m);
    let tb = m.nr_series_flow();
    let fb = m.nr_series_flow();
    let c = m.get_bool(false);
    let iff = m.nr_if_flow(c, tb, Some(fb));
    let first = m.get_int(1);
    let second = m.get_int(2);
    let start = m.get_bool(false);
    let a1 = m.nr_assign(v, first);
    let a2 = m.nr_assign(v, second);
    let end = m.get_bool(false);
    m.series_push(b, start);
    m.series_push(b, iff);
    m.series_push(tb, a1);
    m.series_push(fb, a2);
    m.series_push(b, end);
    let rd = rd_of(&m, f);
    assert_eq!(rdset(&m, &rd, v, start).len(), 0);
    let e = rdset(&m, &rd, v, end);
    assert_eq!(e.len(), 2);
    assert!(e.contains(&m.value_uid(first)) && e.contains(&m.value_uid(second)));
}

#[test]
fn rd_analysis_try_catch() {
    let mut m = setup();
    let (f, b, v) = func_with_body(&mut m);
    let body = m.nr_series_flow();
    let except = m.nr_series_flow();
    let finally = m.nr_series_flow();
    let tc = m.nr_try_catch_flow(body, Some(finally), None);
    m.try_catch_push(tc, except, None, None);
    let first = m.get_int(1);
    let second = m.get_int(2);
    let third = m.get_int(3);
    let initial = m.get_int(0);
    let a0 = m.nr_assign(v, initial);
    let start = m.get_bool(false);
    let a1 = m.nr_assign(v, first);
    let a2 = m.nr_assign(v, second);
    let middle = m.get_bool(false);
    let a3 = m.nr_assign(v, third);
    let end = m.get_bool(false);
    m.series_push(b, start);
    m.series_push(b, a0);
    m.series_push(b, tc);
    m.series_push(body, a1);
    m.series_push(except, a3);
    m.series_push(finally, middle);
    m.series_push(finally, a2);
    m.series_push(b, end);
    let rd = rd_of(&m, f);
    assert_eq!(rdset(&m, &rd, v, start).len(), 0);
    assert_eq!(rdset(&m, &rd, v, end), [m.value_uid(second)].into_iter().collect());
    let mid = rdset(&m, &rd, v, middle);
    for x in [first, third, initial] {
        assert!(mid.contains(&m.value_uid(x)));
    }
}

#[test]
fn rd_analysis_while_loop() {
    let mut m = setup();
    let (f, b, v) = func_with_body(&mut m);
    let lb = m.nr_series_flow();
    let c = m.get_bool(false);
    let w = m.nr_while_flow(c, lb);
    let first = m.get_int(1);
    let second = m.get_int(2);
    let start = m.get_bool(false);
    let a1 = m.nr_assign(v, first);
    let a2 = m.nr_assign(v, second);
    let end = m.get_bool(false);
    m.series_push(b, start);
    m.series_push(b, a1);
    m.series_push(b, w);
    m.series_push(lb, a2);
    m.series_push(b, end);
    let rd = rd_of(&m, f);
    assert_eq!(rdset(&m, &rd, v, start).len(), 0);
    let e = rdset(&m, &rd, v, end);
    assert_eq!(e.len(), 2);
    assert!(e.contains(&m.value_uid(first)) && e.contains(&m.value_uid(second)));
}

#[test]
fn cfg_while_and_imperative_for() {
    let mut m = setup();
    let (f, b, v) = func_with_body(&mut m);
    let lb = m.nr_series_flow();
    let s = m.get_int(0);
    let e = m.get_int(10);
    let fl = m.nr_imperative_for_flow(s, 1, e, lb, v);
    let use_ = m.nr_var_value(v);
    m.series_push(lb, use_);
    m.series_push(b, fl);
    let cfg = build_cf_graph(&m, f);
    let names: Vec<&str> = cfg.blocks().iter().map(|b| b.name()).collect();
    assert_eq!(names, vec!["entry", "endFor", "forBegin", "forCheck", "forUpdate", "forBody"]);
    assert!(cfg.format(&m).starts_with("digraph \"test_f\""));
    // loop var reaches the body via the two synthetic assignments
    let mut rd = RDInspector::new(Rc::new(cfg));
    rd.analyze(&m);
    let defs = rd.get_reaching_definitions(&m, v, use_);
    assert_eq!(defs.len(), 2);
    let kinds: FxHashSet<_> =
        defs.iter().map(|d| format!("{:?}", rd.cfg().as_synthetic_assign(d.assignment).unwrap().kind)).collect();
    assert!(kinds.contains(&format!("{:?}", SyntheticAssignKind::Known)));
    assert!(kinds.contains(&format!("{:?}", SyntheticAssignKind::Add)));
}

#[test]
fn analyses_through_pass_interface() {
    let mut m = setup();
    let (f, b, v) = func_with_body(&mut m);
    let one = m.get_int(1);
    let a = m.nr_assign(v, one);
    m.series_push(b, a);
    let it = m.get_int_type();
    let g = m.nr_var(it, true, "g");
    let two = m.get_int(2);
    let ga = m.nr_assign(g, two);
    m.series_push(b, ga);
    let ctx = NullPassContext;
    let cfg = CFAnalysis.run(&m, &ctx);
    let cfg = cfg.as_any().downcast_ref::<CFResult>().unwrap();
    assert!(cfg.graphs.contains_key(&m.var_uid(f)));
    let rd = RDAnalysis::new(CFAnalysis::KEY).run(&m, &ctx);
    assert!(rd.as_any().downcast_ref::<RDResult>().unwrap().results.contains_key(&m.var_uid(f)));
    let dom = DominatorAnalysis::new(CFAnalysis::KEY).run(&m, &ctx);
    assert!(dom.as_any().downcast_ref::<DominatorResult>().is_some());
    let gv = GlobalVarsAnalyses.run(&m, &ctx);
    let gv = gv.as_any().downcast_ref::<GlobalVarsResult>().unwrap();
    assert_eq!(gv.assignments.get(&m.var_uid(g)), Some(&m.value_uid(two)));
    assert!(!gv.assignments.contains_key(&m.var_uid(v)));
    assert_eq!(GlobalVarsAnalyses::KEY, "core-analyses-global-vars");
    assert_eq!(SideEffectAnalysis::KEY, "core-analyses-side-effect");
    assert_eq!(CaptureAnalysis::KEY, "core-analyses-capture");
}

/// Builds `f(a: T) -> T { return a }` with a reference type T.
fn ref_identity(m: &mut Module, name: &str) -> (VarId, typhon_ir::TypeId) {
    let it = m.get_int_type();
    let rec = m.nr_record_type(format!("{name}_rec"), vec![it], vec!["x".into()]);
    let rt = m.nr_ref_type(format!("{name}_ref"), rec, false);
    let ft = m.get_func_type(rt, vec![rt], false);
    let f = m.nr_bodied_func(name);
    m.realize_func(f, ft, &["a".to_string()]);
    let body = m.nr_series_flow();
    m.set_func_body(f, Some(body));
    (f, rt)
}

#[test]
fn capture_and_side_effects() {
    let mut m = setup();
    // f(a) { return a }
    let (f, rt) = ref_identity(&mut m, "f");
    let a = m.func_args(f)[0];
    let av = m.nr_var_value(a);
    let r = m.nr_return(Some(av));
    let fb = m.func_body(f).unwrap();
    m.series_push(fb, r);
    // h(a) { g = a; return a }
    let (h, _) = ref_identity(&mut m, "h");
    let ha = m.func_args(h)[0];
    let g = m.nr_var(rt, true, "g");
    let hav = m.nr_var_value(ha);
    let asg = m.nr_assign(g, hav);
    let hav2 = m.nr_var_value(ha);
    let hr = m.nr_return(Some(hav2));
    let hb = m.func_body(h).unwrap();
    m.series_push(hb, asg);
    m.series_push(hb, hr);

    let ctx = NullPassContext;
    let cr = CaptureAnalysis::new(RDAnalysis::KEY, DominatorAnalysis::KEY).run(&m, &ctx);
    let cr = cr.as_any().downcast_ref::<CaptureResult>().unwrap();
    let fi = &cr.results[&m.var_uid(f)][0];
    assert!(fi.return_captures && !fi.extern_captures && fi.arg_captures.is_empty());
    let hi = &cr.results[&m.var_uid(h)][0];
    assert!(hi.extern_captures);

    let se = SideEffectAnalysis::compute(&m, cr, true);
    let _: &SideEffectResult = &se;
    assert_eq!(se.status(m.var_uid(f)), Some(typhon_ir::util::side_effect::SideEffectStatus::Pure));
    assert!(se.has_side_effect(&m, asg));
    assert!(!se.has_side_effect(&m, av));
}

#[test]
fn inline_simple_call() {
    let mut m = setup();
    let it = m.get_int_type();
    let ft = m.get_func_type(it, vec![it], false);
    let f = m.nr_bodied_func("id");
    m.realize_func(f, ft, &["x".to_string()]);
    let body = m.nr_series_flow();
    m.set_func_body(f, Some(body));
    let x = m.func_args(f)[0];
    let xv = m.nr_var_value(x);
    let r = m.nr_return(Some(xv));
    m.series_push(body, r);
    let callee = m.nr_var_value(f);
    let arg = m.get_int(5);
    let c = m.nr_call(callee, vec![arg]);
    let res = inline_call(&mut m, c, false);
    assert!(res.is_some());
    let fi = m.as_flow_instr(res.result.unwrap()).expect("flow instr");
    assert!(m.as_series_flow(fi.flow).is_some());
    assert_eq!(res.new_vars.len(), 2); // cloned arg + return var
}

#[test]
fn outline_private_region() {
    let mut m = setup();
    let (f, b, _) = func_with_body(&mut m);
    let it = m.get_int_type();
    let y = m.nr_var(it, false, "y");
    m.func_push_symbol(f, y);
    let two = m.get_int(2);
    let a = m.nr_assign(y, two);
    let region = m.nr_series_flow();
    m.series_push(region, a);
    m.series_push(b, region);
    let res = outline_series(&mut m, f, region, true, false, false);
    assert!(res.is_some());
    assert!(res.arg_kinds.is_empty());
    assert_eq!(res.num_out_flows, 0);
    let s = &m.as_series_flow(region).unwrap().series;
    assert_eq!(s.len(), 1);
    assert_eq!(s[0], res.call.unwrap());
    let of = res.func.unwrap();
    let ob = m.func_body(of).unwrap();
    assert_eq!(m.as_series_flow(ob).unwrap().series.len(), 1);
}
