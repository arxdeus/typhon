use super::*;
use typhon_ir::pass::NullPassContext;

fn shared_of(m: &mut Module, base: TypeId) -> VarId {
    let p = m.get_pointer_type(Some(base));
    m.nr_var(p, false, "s")
}

#[test]
fn key_matches_cpp() {
    assert_eq!(OpenMPPass::KEY, "core-parallel-openmp");
    assert_eq!(OpenMPPass::new().key(), "core-parallel-openmp");
}

#[test]
fn template_selection() {
    let m = Module::new("t");
    let mut s = OMPSched::default();
    assert_eq!(imperative_template_name(&s), "_static_loop_outline_template");
    s.chunk = { let mut m = m; Some(m.get_int(4)) };
    assert_eq!(imperative_template_name(&s), "_static_chunked_loop_outline_template");
    s.dynamic = true;
    assert_eq!(imperative_template_name(&s), "_dynamic_loop_outline_template");
    s.gpu = true;
    assert_eq!(imperative_template_name(&s), "_gpu_loop_outline_template");
}

#[test]
fn int_reduction_initials() {
    let mut m = Module::new("t");
    let it = m.get_int_type();
    let s = shared_of(&mut m, it);
    let cases = [
        (ReductionKind::Add, 0),
        (ReductionKind::Sub, 0),
        (ReductionKind::Mul, 1),
        (ReductionKind::And, !0),
        (ReductionKind::Or, 0),
        (ReductionKind::Xor, 0),
        (ReductionKind::Min, i64::MAX),
        (ReductionKind::Max, i64::MIN),
    ];
    for (k, v) in cases {
        let init = Reduction::new(k, s).initial(&mut m).unwrap();
        assert_eq!(m.as_int_const(init).unwrap().val, v, "{k:?}");
    }
    assert!(Reduction::default().initial(&mut m).is_none());
    assert!(!Reduction::default().is_some());
}

#[test]
fn float_and_bool_reduction_initials() {
    let mut m = Module::new("t");
    let ft = m.get_float_type();
    let s = shared_of(&mut m, ft);
    let i = Reduction::new(ReductionKind::Mul, s).initial(&mut m).unwrap();
    assert_eq!(m.as_float_const(i).unwrap().val, 1.0);
    assert!(Reduction::new(ReductionKind::Xor, s).initial(&mut m).is_none());

    let bt = m.get_bool_type();
    let b = shared_of(&mut m, bt);
    let i = Reduction::new(ReductionKind::And, b).initial(&mut m).unwrap();
    assert!(m.as_bool_const(i).unwrap().val);
    let i = Reduction::new(ReductionKind::Or, b).initial(&mut m).unwrap();
    assert!(!m.as_bool_const(i).unwrap().val);
    assert!(Reduction::new(ReductionKind::Add, b).initial(&mut m).is_none());
}

#[test]
fn non_setitem_call_is_not_reduction() {
    let mut m = Module::new("t");
    let it = m.get_int_type();
    let s = shared_of(&mut m, it);
    let f = m.nr_bodied_func("foo");
    let sv = m.nr_var_value(s);
    let c = util::call(&mut m, f, vec![sv]);
    let ri = ReductionIdentifier::new(vec![s], None);
    assert!(!ri.reduction_from_call(&mut m, c).is_some());
    // loop var is never shared
    let ri = ReductionIdentifier::new(vec![s], Some(s));
    assert!(!ri.is_shared(&m, s));
    assert!(ReductionIdentifier::new(vec![s], None).is_shared(&m, s));
}

#[test]
fn collapse_preconditions() {
    let mut m = Module::new("t");
    let it = m.get_int_type();
    let parent = m.nr_bodied_func("p");
    let i = m.nr_var(it, false, "i");
    let body = m.nr_series_flow();
    let x = m.get_int(1);
    m.series_push(body, x);
    let (a, b) = (m.get_int(0), m.get_int(10));
    let lp = m.nr_imperative_for_flow(a, 1, b, body, i);
    let r = collapse_loop(&mut m, parent, lp, 0);
    assert!(r.collapsed.is_none());
    assert_eq!(r.error, "'collapse' must be at least 1");
    let r = collapse_loop(&mut m, parent, lp, 2);
    assert_eq!(r.error, "loop nest not collapsible");
}

#[test]
fn pass_without_parallel_loops_is_noop() {
    let mut m = Module::new("t");
    let main = m.main_func();
    let body = m.nr_series_flow();
    let it = m.get_int_type();
    let i = m.nr_var(it, false, "i");
    let lb = m.nr_series_flow();
    let (a, b) = (m.get_int(0), m.get_int(3));
    let lp = m.nr_imperative_for_flow(a, 1, b, lb, i);
    m.series_push(body, lp);
    m.set_func_body(main, Some(body));
    let mut p = OpenMPPass::new();
    p.run(&mut m, &NullPassContext);
    assert!(m.as_imperative_for_flow(lp).is_some());
    assert_eq!(m.as_series_flow(body).unwrap().series, vec![lp]);
}
