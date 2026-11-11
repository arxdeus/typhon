//! Port of `test/cir/{base,constant,value,var,type}.cpp`.

mod common;

use common::*;
use typhon_ir::util::matching::match_values;
use typhon_ir::util::visitor::{ConstVisitor, Visitor, accept_value, const_accept_value};
use typhon_ir::*;

struct TestVisitor;
impl Visitor for TestVisitor {
    fn visit_int_const(&mut self, _m: &mut Module, _v: ValueId) {
        panic!("visited IntConst");
    }
    fn visit_bool_const(&mut self, _m: &mut Module, _v: ValueId) {}
}
struct ConstTestVisitor;
impl ConstVisitor for ConstTestVisitor {
    fn visit_int_const(&mut self, _m: &Module, _v: ValueId) {
        panic!("visited IntConst");
    }
    fn visit_bool_const(&mut self, _m: &Module, _v: ValueId) {}
}

// ---- base.cpp ----

#[test]
fn node_no_replacement_rtti() {
    let (mut m, _) = setup();
    let i = m.get_int_type();
    let derived = m.nr_int_const(1, i);
    assert!(!m.has_replacement(derived.into()));
    assert!(m.as_int_const(derived).is_some());
    assert!(m.kind(derived).is_const());
    assert!(!m.kind(derived).is_flow());
}

#[test]
fn node_no_replacement_attributes() {
    let (mut m, _) = setup();
    let i = m.get_int_type();
    let n = m.nr_int_const(1, i);
    assert!(!m.has_replacement(n.into()));
    assert!(!m.has_attribute(n.into(), attr_id::KEY_VALUE));
    assert!(m.has_attribute(n.into(), attr_id::SRC_INFO));
    assert!(m.get_attribute(n.into(), attr_id::SRC_INFO).is_some());
    assert_eq!(1, m.attributes(n.into()).len());
}

#[test]
fn node_replacement_rtti() {
    let (mut m, _) = setup();
    let i = m.get_int_type();
    let node = m.nr_int_const(1, i);
    assert!(!m.has_replacement(node.into()));
    assert!(m.as_int_const(node).is_some());
    let b = m.get_bool_type();
    let r = m.nr_bool_const(false, b);
    m.replace_all_value(node, r);
    assert!(m.has_replacement(node.into()));
    assert!(m.as_int_const(node).is_none());
    assert!(m.as_bool_const(node).is_some());
}

#[test]
fn node_replacement_delegates() {
    let (mut m, _) = setup();
    let i = m.get_int_type();
    let original = m.nr_int_const(1, i);
    let b = m.get_bool_type();
    let new = m.nr_named(ValueKind::BoolConst(TemplatedConst { val: false, ty: b }), "foo");
    m.set_attribute(new.into(), Attribute::KeyValue(KeyValueAttribute::default()));
    assert_eq!(0, m.value_name(original).len());
    assert_eq!(1, m.attributes(original.into()).len());
    m.replace_all_value(original, new);
    assert_eq!("foo", m.value_name(original));
    assert_eq!(2, m.attributes(original.into()).len());
    accept_value(&mut TestVisitor, &mut m, original);
    accept_value(&mut TestVisitor, &mut m, new);
    const_accept_value(&mut ConstTestVisitor, &m, original);
    const_accept_value(&mut ConstTestVisitor, &m, new);
}

#[test]
#[should_panic]
fn node_non_replaceable_fails() {
    let (mut m, _) = setup();
    let i = m.get_int_type();
    let n = m.nr_int_const(1, i);
    m.set_replaceable(n.into(), false);
    m.replace_all_value(n, n);
}

// ---- constant.cpp ----

#[test]
fn const_type_query_and_replace() {
    let (mut m, _) = setup();
    let i = m.get_int_type();
    let n = m.nr_int_const(1, i);
    assert_eq!(i, m.value_type(n));
    let used = m.used_types(n.into());
    assert_eq!(vec![i], used);
    assert_eq!(1, m.replace_used_type(n.into(), i, i));
}

#[test]
fn const_cloning() {
    let (mut m, mut cv) = setup();
    let i = m.get_int_type();
    let n = m.nr_int_const(1, i);
    let c = cv.clone_value(&mut m, n);
    assert_eq!(1, m.as_int_const(c).unwrap().val);
    assert_eq!(i, m.value_type(c));
}

// ---- value.cpp ----

#[test]
fn value_query_methods_delegate() {
    let (mut m, _) = setup();
    let i = m.get_int_type();
    let original = m.nr_named(ValueKind::IntConst(TemplatedConst { val: 1, ty: i }), "foo");
    let original_ref = m.reference_string(original.into());
    let f = m.nr_bodied_func("");
    let ft = m.unsafe_get_func_type("<internal_func_type>", i, vec![], false);
    m.realize_func(f, ft, &[]);
    let replacement = m.nr_named(ValueKind::VarValue(VarValue { var: f }), "baz");
    m.replace_all_value(original, replacement);
    assert_ne!(original_ref, m.reference_string(original.into()));
    assert_eq!(m.reference_string(original.into()), m.reference_string(replacement.into()));
    assert_eq!(1, m.used_variables(original.into()).len());

    let replacement2 = m.nr_call(replacement, vec![]);
    m.replace_all_value(original, replacement2);
    assert_eq!(0, m.used_variables(original.into()).len());
    assert_eq!(1, m.used_values(original.into()).len());
    assert_eq!(i, m.value_type(original));

    let replacement3 = m.nr_yield_in(i, true);
    m.replace_all_value(original, replacement3);
    assert_eq!(0, m.used_variables(original.into()).len());
    assert_eq!(0, m.used_values(original.into()).len());
    assert_eq!(1, m.used_types(original.into()).len());
}

#[test]
fn value_replace_methods_delegate() {
    let (mut m, _) = setup();
    let i = m.get_int_type();
    let original = m.nr_named(ValueKind::IntConst(TemplatedConst { val: 1, ty: i }), "foo");
    let original_id = m.value_uid(original);
    let var = m.nr_bodied_func("");
    let replacement = m.nr_named(ValueKind::VarValue(VarValue { var }), "baz");
    m.replace_all_value(original, replacement);
    assert_eq!(1, m.replace_used_variable(original.into(), var, var));

    let val = replacement;
    let replacement2 = m.nr_call(replacement, vec![]);
    m.replace_all_value(original, replacement2);
    assert_eq!(1, m.replace_used_value(original.into(), val, val));

    let replacement3 = m.nr_yield_in(i, true);
    m.replace_all_value(original, replacement3);
    let f = m.get_float_type();
    assert_eq!(1, m.replace_used_type(original.into(), i, f));
    assert_ne!(original_id, m.value_uid(original));
    assert_eq!(m.value_uid(original), m.value_uid(replacement3));
}

// ---- var.cpp ----

#[test]
fn var_query_methods_delegate() {
    let (mut m, _) = setup();
    let i = m.get_int_type();
    let f = m.get_float_type();
    let original = m.nr_var(i, false, "");
    let replacement = m.nr_var(f, false, "");
    m.replace_all_var(original, replacement);
    assert_eq!(Some(f), m.var_type(original));
    assert_eq!(Some(&f), m.used_types(original.into()).last());
}

#[test]
fn var_replace_methods_delegate() {
    let (mut m, _) = setup();
    let i = m.get_int_type();
    let f = m.get_float_type();
    let original = m.nr_var(i, false, "");
    let replacement = m.nr_var(f, false, "");
    let original_id = m.var_uid(original);
    m.replace_all_var(original, replacement);
    assert_eq!(1, m.replace_used_type(original.into(), f, i));
    assert_ne!(original_id, m.var_uid(original));
    assert_eq!(m.var_uid(original), m.var_uid(replacement));
}

// ---- type.cpp ----

#[test]
fn record_type_query() {
    let (mut m, _) = setup();
    let i = m.get_int_type();
    let t = m.nr_record_type_unnamed("foo", vec![i]);
    assert_eq!(Some(i), m.member_type(t, "1"));
    assert_eq!(0, m.member_index(t, "1"));
    assert_eq!(1, m.type_fields(t).len());
    assert_eq!(i, m.type_fields(t)[0].ty);
    m.realize_membered(t, vec![i], names(&["baz"]));
    assert!(m.type_is_atomic(t));
    assert_eq!(vec![i], m.used_types(t.into()));
}

#[test]
fn ref_type_query() {
    let (mut m, _) = setup();
    let i = m.get_int_type();
    let contents = m.nr_record_type_unnamed("foo", vec![i]);
    let t = m.nr_ref_type("baz", contents, false);
    assert_eq!(Some(i), m.member_type(t, "1"));
    assert_eq!(0, m.member_index(t, "1"));
    assert_eq!(1, m.type_fields(t).len());
    m.realize_membered(t, vec![i], names(&["baz"]));
    assert_eq!(Some(i), m.member_type(t, "baz"));
    assert!(!m.type_is_atomic(t));
    assert_eq!(vec![contents], m.used_types(t.into()));
}

#[test]
fn func_type_query() {
    let (mut m, _) = setup();
    let i = m.get_int_type();
    let f = m.get_float_type();
    let t = m.nr_func_type("foo", i, vec![f], false);
    assert_eq!(vec![f], m.func_type_args(t));
    assert_eq!(2, m.used_types(t.into()).len());
}

#[test]
fn union_type_identity() {
    let (mut m, _) = setup();
    let i = m.get_int_type();
    let f = m.get_float_type();
    let numeric = m.unsafe_get_union_type(&[i, f]);
    let integer = m.unsafe_get_union_type(&[i]);
    assert_eq!(m.type_name(numeric), types::union_type_name(&["Int64", "float"]));
    assert!(!m.type_name(integer).is_empty());
    assert_ne!(m.type_name(numeric), m.type_name(integer));
    assert_eq!(numeric, m.unsafe_get_union_type(&[i, f]));
    assert_eq!(integer, m.unsafe_get_union_type(&[i]));
    let p1 = m.unsafe_get_pointer_type(Some(numeric));
    let p2 = m.unsafe_get_pointer_type(Some(integer));
    assert_ne!(p1, p2);
}

#[test]
fn value_and_clone_match() {
    let (mut m, mut cv) = setup();
    let s = m.get_string("hi\n");
    let c = cv.clone_value(&mut m, s);
    assert!(match_values(&m, s, c));
}
