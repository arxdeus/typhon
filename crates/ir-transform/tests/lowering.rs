//! Structural tests for `typhon_ir_transform::lowering` (no TypeRealizer).
//! Note: paths needing stdlib realization (range/list lowering, async, await,
//! throw fallback, missing-return helpers) are covered by the end-to-end testsuite.

use typhon_ir::pass::{NullPassContext, Pass};
use typhon_ir::util::format::format;
use typhon_ir::{FuncId, Module, NodeRef, Stage, ValueId};
use typhon_ir_transform::lowering::*;

fn func(m: &mut Module, name: &str, ret: typhon_ir::TypeId, args: Vec<typhon_ir::TypeId>) -> FuncId {
    let f = m.nr_bodied_func(name);
    let ft = m.unsafe_get_func_type(&format!("{name}_t"), ret, args.clone(), false);
    let names: Vec<String> = (0..args.len()).map(|i| format!("a{i}")).collect();
    m.realize_func(f, ft, &names);
    m.func_mut(f).unwrap().unmangled_name = name.to_string();
    f
}

fn main_with(m: &mut Module, body: ValueId) {
    let main = m.main_func();
    let i = m.get_int_type();
    let ft = m.unsafe_get_func_type("main_t", i, vec![], false);
    m.realize_func(main, ft, &[]);
    m.set_func_body(main, Some(body));
}

#[test]
fn keys() {
    assert_eq!(PipelineLowering::KEY, "core-pipeline-lowering");
    assert_eq!(ImperativeForFlowLowering::KEY, "core-imperative-for-lowering");
    assert_eq!(MissingReturnLowering::KEY, "core-missing-return-lowering");
    assert_eq!(ThrowLowering::KEY, "core-throw-lowering");
    assert_eq!(AsyncForLowering::KEY, "core-async-for-lowering");
    assert_eq!(AwaitLowering::KEY, "core-await-lowering");
    assert_eq!(PipelineLowering::new().key(), PipelineLowering::KEY);
}

#[test]
fn pipeline_non_generator_becomes_calls() {
    let mut m = Module::new("");
    let i = m.get_int_type();
    let f = func(&mut m, "f", i, vec![i]);
    let g = func(&mut m, "g", i, vec![i, i]);
    let one = m.get_int(1);
    let fv = m.nr_var_value(f);
    let gv = m.nr_var_value(g);
    let two = m.get_int(2);
    let p = m.nr_pipeline_flow(vec![
        Stage { callee: one, args: vec![], generator: false, parallel: false },
        Stage { callee: fv, args: vec![None], generator: false, parallel: false },
        Stage { callee: gv, args: vec![Some(two), None], generator: false, parallel: false },
    ]);
    let body = m.nr_series_flow();
    m.series_push(body, p);
    main_with(&mut m, body);
    PipelineLowering::new().run(&mut m, &NullPassContext);
    let s = format(&m, NodeRef::Value(body));
    assert!(!s.contains("pipeline"), "{s}");
    let outer = m.as_call_instr(p).expect("pipeline replaced by call");
    assert_eq!(outer.args.len(), 2);
    let inner = m.as_call_instr(outer.args[1]).expect("nested call");
    assert_eq!(m.as_int_const(inner.args[0]).unwrap().val, 1);
}

#[test]
fn pipeline_generator_becomes_for_loop() {
    let mut m = Module::new("");
    let i = m.get_int_type();
    let gt = m.get_generator_type(i);
    let gen_var = m.nr_var(gt, true, "gen");
    let f = func(&mut m, "f", i, vec![i]);
    let gv = m.nr_var_value(gen_var);
    let fv = m.nr_var_value(f);
    let p = m.nr_pipeline_flow(vec![
        Stage { callee: gv, args: vec![], generator: true, parallel: false },
        Stage { callee: fv, args: vec![None], generator: false, parallel: true },
    ]);
    let body = m.nr_series_flow();
    m.series_push(body, p);
    main_with(&mut m, body);
    PipelineLowering::new().run(&mut m, &NullPassContext);
    let lp = m.as_for_flow(p).expect("pipeline replaced by for");
    assert!(lp.is_parallel());
    assert_eq!(m.var_type(lp.var), Some(i));
    let lbody = m.as_series_flow(lp.body).unwrap();
    assert!(m.as_call_instr(lbody.series[0]).is_some());
}

#[test]
fn missing_return_leaves_returning_functions_alone() {
    let mut m = Module::new("");
    let i = m.get_int_type();
    let body = m.nr_series_flow();
    let c = m.get_int(1);
    let r = m.nr_return(Some(c));
    m.series_push(body, r);
    main_with(&mut m, body);
    // if/else both returning, and an infinite loop, also need nothing.
    let f = func(&mut m, "f", i, vec![]);
    let fb = m.nr_series_flow();
    let t = m.get_bool(true);
    let wb = m.nr_series_flow();
    let w = m.nr_while_flow(t, wb);
    m.series_push(fb, w);
    m.set_func_body(f, Some(fb));
    MissingReturnLowering::new().run(&mut m, &NullPassContext);
    assert_eq!(m.as_series_flow(body).unwrap().series.len(), 1);
    assert_eq!(m.as_series_flow(fb).unwrap().series.len(), 1);
}

#[test]
#[should_panic(expected = "missing-return helper not found")]
fn missing_return_requires_helper_for_fallthrough() {
    let mut m = Module::new("");
    let body = m.nr_series_flow();
    let c = m.get_int(1);
    let t = m.get_bool(false);
    let r = m.nr_return(Some(c));
    let rs = m.nr_series_flow();
    m.series_push(rs, r);
    let iff = m.nr_if_flow(t, rs, None);
    m.series_push(body, iff);
    main_with(&mut m, body);
    MissingReturnLowering::new().run(&mut m, &NullPassContext);
}

#[test]
fn throw_with_value_is_untouched() {
    let mut m = Module::new("");
    let body = m.nr_series_flow();
    let c = m.get_int(1);
    let t = m.nr_throw(Some(c));
    m.series_push(body, t);
    main_with(&mut m, body);
    ThrowLowering::new().run(&mut m, &NullPassContext);
    assert!(m.as_throw_instr(t).unwrap().fallback.is_none());
}

#[test]
fn imperative_ignores_non_range_loops() {
    let mut m = Module::new("");
    let i = m.get_int_type();
    let gt = m.get_generator_type(i);
    let gen_var = m.nr_var(gt, true, "gen");
    let x = m.nr_var(i, false, "x");
    let it = m.nr_var_value(gen_var);
    let lb = m.nr_series_flow();
    let lp = m.nr_for_flow(it, lb, x);
    let body = m.nr_series_flow();
    m.series_push(body, lp);
    main_with(&mut m, body);
    ImperativeForFlowLowering::new().run(&mut m, &NullPassContext);
    AsyncForLowering::new().run(&mut m, &NullPassContext);
    assert!(m.as_for_flow(lp).is_some());
}
