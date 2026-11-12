
mod common;

use common::*;
use std::panic::{AssertUnwindSafe, catch_unwind};
use typhon_ir::util::matching::match_values;
use typhon_ir::*;

fn int_const(m: &mut Module, v: i64) -> ValueId {
    let i = m.get_int_type();
    m.nr_int_const(v, i)
}


#[test]
fn series_flow_insertion_erase_and_iterators() {
    let (mut m, _) = setup();
    let f = m.nr_series_flow();
    let second = int_const(&mut m, 1);
    m.series_push(f, second);
    let first = int_const(&mut m, 2);
    m.series_insert(f, 0, first);
    let contents = m.as_series_flow(f).unwrap().series.clone();
    assert_eq!(2, contents.len());
    assert_eq!(2, m.as_int_const(contents[0]).unwrap().val);
    assert_eq!(1, m.as_int_const(contents[1]).unwrap().val);
    m.as_series_flow_mut(f).unwrap().series.remove(0);
    let c0 = m.as_series_flow(f).unwrap().series[0];
    assert_eq!(1, m.as_int_const(c0).unwrap().val);
}

#[test]
fn series_flow_query_and_replace() {
    let (mut m, _) = setup();
    let f = m.nr_series_flow();
    let a = int_const(&mut m, 1);
    m.series_push(f, a);
    let b = int_const(&mut m, 2);
    m.series_insert(f, 0, b);
    let contents = m.as_series_flow(f).unwrap().series.clone();
    assert_eq!(contents, m.used_values(f.into()));
    assert_eq!(0, m.used_types(f.into()).len());
    assert_eq!(0, m.used_variables(f.into()).len());
    assert_eq!(1, m.replace_used_value(f.into(), contents[0], contents[1]));
}

#[test]
fn series_flow_cloning() {
    let (mut m, mut cv) = setup();
    let f = m.nr_series_flow();
    let a = int_const(&mut m, 1);
    m.series_push(f, a);
    let b = int_const(&mut m, 2);
    m.series_insert(f, 0, b);
    let c = cv.clone_value(&mut m, f);
    assert!(match_values(&m, f, c));
}

#[test]
fn while_flow_query_and_replace() {
    let (mut m, _) = setup();
    let cond = m.get_bool(true);
    let body = m.nr_series_flow();
    let f = m.nr_while_flow(cond, body);
    assert_eq!(cond, m.as_while_flow(f).unwrap().cond);
    assert_eq!(body, m.as_while_flow(f).unwrap().body);
    assert_eq!(vec![cond, body], m.used_values(f.into()));
    assert_eq!(0, m.used_types(f.into()).len());
    assert_eq!(0, m.used_variables(f.into()).len());
    let c2 = m.get_bool(true);
    assert_eq!(1, m.replace_used_value(f.into(), cond, c2));
    let b2 = m.nr_series_flow();
    assert_eq!(1, m.replace_used_value(f.into(), body, b2));
    let bad = m.get_bool(true);
    let r = catch_unwind(AssertUnwindSafe(|| m.replace_used_value(f.into(), b2, bad)));
    assert!(r.is_err());
    assert_ne!(vec![cond, body], m.used_values(f.into()));
}

#[test]
fn while_flow_cloning() {
    let (mut m, mut cv) = setup();
    let cond = m.get_bool(true);
    let body = m.nr_series_flow();
    let f = m.nr_while_flow(cond, body);
    let c = cv.clone_value(&mut m, f);
    assert!(match_values(&m, f, c));
}

#[test]
fn for_flow_query_and_replace() {
    let (mut m, _) = setup();
    let iter = m.get_string("hi");
    let body = m.nr_series_flow();
    let s = m.get_string_type();
    let var = m.nr_var(s, false, "x");
    let f = m.nr_for_flow(iter, body, var);
    assert_eq!(vec![iter, body], m.used_values(f.into()));
    assert_eq!(0, m.used_types(f.into()).len());
    assert_eq!(vec![var], m.used_variables(f.into()));
    let i2 = m.get_string("hi");
    assert_eq!(1, m.replace_used_value(f.into(), iter, i2));
    let b2 = m.nr_series_flow();
    assert_eq!(1, m.replace_used_value(f.into(), body, b2));
    let v2 = m.nr_var(s, false, "x");
    assert_eq!(1, m.replace_used_variable(f.into(), var, v2));
    assert_ne!(var, m.as_for_flow(f).unwrap().var);
}

#[test]
fn for_flow_cloning() {
    let (mut m, mut cv) = setup();
    let iter = m.get_string("hi");
    let body = m.nr_series_flow();
    let s = m.get_string_type();
    let var = m.nr_var(s, false, "x");
    let f = m.nr_for_flow(iter, body, var);
    let c = cv.clone_value(&mut m, f);
    assert!(match_values(&m, f, c));
}

#[test]
fn if_flow_query_and_replace() {
    let (mut m, _) = setup();
    let cond = m.get_bool(true);
    let t = m.nr_series_flow();
    let e = m.nr_series_flow();
    let f = m.nr_if_flow(cond, t, Some(e));
    assert_eq!(vec![cond, t, e], m.used_values(f.into()));
    m.as_if_flow_mut(f).unwrap().false_branch = None;
    assert_eq!(vec![cond, t], m.used_values(f.into()));
    m.as_if_flow_mut(f).unwrap().false_branch = Some(e);
    let c2 = m.get_bool(true);
    assert_eq!(1, m.replace_used_value(f.into(), cond, c2));
    let t2 = m.nr_series_flow();
    assert_eq!(1, m.replace_used_value(f.into(), t, t2));
    let e2 = m.nr_series_flow();
    assert_eq!(1, m.replace_used_value(f.into(), e, e2));
    let bad = m.get_bool(true);
    assert!(catch_unwind(AssertUnwindSafe(|| m.replace_used_value(f.into(), t2, bad))).is_err());
    assert!(catch_unwind(AssertUnwindSafe(|| m.replace_used_value(f.into(), e2, bad))).is_err());
}

#[test]
fn if_flow_cloning() {
    let (mut m, mut cv) = setup();
    let cond = m.get_bool(true);
    let t = m.nr_series_flow();
    let e = m.nr_series_flow();
    let f = m.nr_if_flow(cond, t, Some(e));
    let c = cv.clone_value(&mut m, f);
    assert!(match_values(&m, f, c));
}

#[test]
fn try_catch_flow_single_catch_query_and_replace() {
    let (mut m, _) = setup();
    let body = m.nr_series_flow();
    let finally = m.nr_series_flow();
    let f = m.nr_try_catch_flow(body, Some(finally), None);
    let handler = m.nr_series_flow();
    let i = m.get_int_type();
    let fl = m.get_float_type();
    let var = m.nr_var(i, false, "");
    m.try_catch_push(f, handler, Some(i), Some(var));
    let v2 = m.nr_var(fl, false, "");
    assert_eq!(1, m.replace_used_variable(f.into(), var, v2));
    assert_eq!(1, m.replace_used_type(f.into(), i, fl));
    let h2 = m.nr_series_flow();
    assert_eq!(1, m.replace_used_value(f.into(), handler, h2));
}


#[test]
fn module_node_building_removal_and_iterators() {
    let (mut m, _) = setup();
    {
        let i = m.get_int_type();
        let n1 = m.add_type("Optional[Int64]", TypeKind::Optional { base: i });
        let num = m.types().len();
        assert!(m.types().contains(&n1));
        m.remove_type(n1);
        assert_eq!(num - 1, m.types().len());
    }
    {
        let n1 = int_const(&mut m, 1);
        let num = m.values().len();
        assert!(m.values().contains(&n1));
        m.remove_value(n1);
        assert_eq!(num - 1, m.values().len());
    }
    {
        let i = m.get_int_type();
        let n1 = m.nr_var(i, false, "");
        let num = m.vars().len();
        assert!(m.vars().contains(&n1));
        m.remove_var(n1);
        assert_eq!(num - 1, m.vars().len());
    }
}

#[test]
fn intrinsic_factory_reuse_and_rollback() {
    let (mut m, _) = setup();
    let integer = m.get_int_type();
    let query = m.get_intrinsic(Intrinsic::Sizeof, Some(integer), None);
    assert!(m.has_valid_intrinsic_signature(query));
    assert!(m.func(query).unwrap().intrinsic().is_type_query());
    assert!(m.is_global(query));
    assert_eq!(query, m.get_intrinsic(Intrinsic::Sizeof, Some(integer), None));
    let fv = m.nr_var_value(query);
    let call = m.nr_call(fv, vec![]);
    assert_eq!(integer, m.value_type(call));
    assert_eq!(m.intrinsic_side_effect_status(query), util::side_effect::SideEffectStatus::Pure);
    assert!(!m.intrinsic_derives_from_arguments(query));

    let atomic = m.get_intrinsic(Intrinsic::IsAtomic, Some(integer), None);
    assert_ne!(query, atomic);
    let at = m.var_type(atomic).unwrap();
    let b = m.get_bool_type();
    assert_eq!(Some(b), m.func_type_ret(at));
    assert!(m.has_valid_intrinsic_signature(atomic));

    m.set_intrinsic(query, Intrinsic::Sizeof, Some(b));
    let mut replacement = m.get_intrinsic(Intrinsic::Sizeof, Some(integer), None);
    assert_ne!(query, replacement);
    let removed = m.var_uid(replacement);
    m.remove_var(replacement);
    replacement = m.get_intrinsic(Intrinsic::Sizeof, Some(integer), None);
    assert_ne!(removed, m.var_uid(replacement));

    m.push_arena();
    let temporary = m.unsafe_get_membered_type("temporary.intrinsic.type", false);
    let tf = m.get_intrinsic(Intrinsic::Sizeof, Some(temporary), None);
    let temporary_id = m.var_uid(tf);
    m.pop_arena();
    assert_eq!(None, m.get_var(temporary_id));
    assert_eq!(None, m.get_type("temporary.intrinsic.type"));
    assert_eq!(replacement, m.get_intrinsic(Intrinsic::Sizeof, Some(integer), None));
    let temporary = m.unsafe_get_membered_type("temporary.intrinsic.type", false);
    let tf2 = m.get_intrinsic(Intrinsic::Sizeof, Some(temporary), None);
    assert_ne!(temporary_id, m.var_uid(tf2));
}

#[test]
fn intrinsic_signatures_and_effects() {
    use util::side_effect::SideEffectStatus as S;
    let (mut m, _) = setup();
    let reference = m.unsafe_get_membered_type("intrinsic.reference", true);
    let sig = m.unsafe_get_func_type("intrinsic.alloc.signature", reference, vec![], false);
    let allocate = m.get_intrinsic(Intrinsic::Allocate, Some(reference), Some(sig));
    assert!(m.has_valid_intrinsic_signature(allocate));
    assert_eq!(m.intrinsic_side_effect_status(allocate), S::NoSideEffect);
    assert!(!m.intrinsic_derives_from_arguments(allocate));

    let record = m.unsafe_get_membered_type("intrinsic.record", false);
    let sig = m.unsafe_get_func_type("intrinsic.record.signature", record, vec![], false);
    let construct = m.get_intrinsic(Intrinsic::RecordNew, Some(record), Some(sig));
    assert!(m.intrinsic_derives_from_arguments(construct));
    assert!(m.has_valid_intrinsic_signature(construct));

    let i = m.get_int_type();
    let generator = m.unsafe_get_generator_type(i);
    let pi = m.unsafe_get_pointer_type(Some(i));
    let sig = m.unsafe_get_func_type("intrinsic.promise.signature", pi, vec![generator], false);
    let promise = m.get_intrinsic(Intrinsic::GeneratorPromise, Some(generator), Some(sig));
    assert!(m.intrinsic_derives_from_arguments(promise));
    assert!(m.has_valid_intrinsic_signature(promise));

    let sig = m.unsafe_get_func_type("intrinsic.lookup.signature", reference, vec![i], false);
    let lookup = m.get_intrinsic(Intrinsic::TypeinfoLookup, None, Some(sig));
    assert_eq!(m.intrinsic_side_effect_status(lookup), S::Unknown);
    assert!(m.has_valid_intrinsic_signature(lookup));
    m.set_intrinsic(lookup, Intrinsic::None, None);
    assert!(!m.has_valid_intrinsic_signature(lookup));
    m.set_intrinsic(lookup, Intrinsic::Sizeof, Some(reference));
    assert!(!m.has_valid_intrinsic_signature(lookup));
}

#[test]
fn module_main_function_and_arg_var() {
    let (mut m, _) = setup();
    let main = m.main_func();
    let i = m.get_int_type();
    let ft = m.unsafe_get_func_type("my_func_type", i, vec![], false);
    m.realize_func(main, ft, &[]);
    let mt = m.var_type(main).unwrap();
    assert!(m.type_kind(mt).is_func());
    assert_eq!(0, m.func_type_args(mt).len());
    assert!(!m.is_replaceable(main.into()));
    let argv = m.argv_var();
    let s = m.get_string_type();
    let ps = m.unsafe_get_pointer_type(Some(s));
    assert!(m.type_is(m.var_type(argv).unwrap(), ps));
    assert!(!m.is_replaceable(argv.into()));
    let argc = m.argc_var();
    assert!(m.type_is(m.var_type(argc).unwrap(), i));
    assert!(!m.is_replaceable(argc.into()));
}

#[test]
fn module_type_get_and_lookup() {
    let (mut m, _) = setup();
    let name = "**test_type**";
    let t = m.unsafe_get_membered_type(name, false);
    assert!(m.type_kind(t).is_record());
    assert_eq!(Some(t), m.get_type(name));
    m.remove_type(t);
    let t = m.unsafe_get_membered_type(name, true);
    assert!(m.type_kind(t).is_ref());
    assert_eq!(Some(t), m.get_type(name));
}

#[test]
fn module_format_smoke() {
    let (mut m, _) = setup();
    let main = m.main_func();
    let i = m.get_int_type();
    let ft = m.unsafe_get_func_type("main_t", i, vec![], false);
    m.realize_func(main, ft, &[]);
    let body = m.nr_series_flow();
    let c = m.get_int(42);
    let r = m.nr_return(Some(c));
    m.series_push(body, r);
    m.set_func_body(main, Some(body));
    let s = util::format::format(&m, NodeRef::Module);
    assert!(s.starts_with("(module\n(argv (var '\".argv."));
    assert!(s.contains("(bodied_func '\"main."));
    assert!(s.contains("(return 42)"));
}
