//! Unit tests for the static operator helpers of `typecheck/op.rs`.

use typhon_parser::typecheck::op::{div_mod, eval_static_int_binary, get_magic_raw, slice_adjust_indices_raw, translate_index_raw};

#[test]
fn magic_names() {
    assert_eq!(get_magic_raw("+"), ("add".into(), "radd".into()));
    assert_eq!(get_magic_raw("<"), ("lt".into(), "gt".into()));
    assert_eq!(get_magic_raw("=="), ("eq".into(), "eq".into()));
    assert_eq!(get_magic_raw("//"), ("floordiv".into(), "rfloordiv".into()));
    assert_eq!(get_magic_raw("@"), ("matmul".into(), "rmatmul".into()));
}

#[test]
fn python_and_c_division() {
    assert_eq!(div_mod(true, -7, 2), Some((-4, 1)));
    assert_eq!(div_mod(false, -7, 2), Some((-3, -1)));
    assert_eq!(div_mod(true, 7, -2), Some((-4, -1)));
    assert_eq!(div_mod(true, 1, 0), None);
}

#[test]
fn static_int_binary() {
    assert_eq!(eval_static_int_binary("+", 2, 3, true), Some(5));
    assert_eq!(eval_static_int_binary("<", 2, 3, true), Some(1));
    assert_eq!(eval_static_int_binary("&&", 0, 3, true), Some(0));
    assert_eq!(eval_static_int_binary("&&", 2, 3, true), Some(3));
    assert_eq!(eval_static_int_binary("||", 0, 3, true), Some(3));
    assert_eq!(eval_static_int_binary("<<", 1, 4, true), Some(16));
    assert_eq!(eval_static_int_binary("%", -7, 3, true), Some(2));
    assert_eq!(eval_static_int_binary("//", 1, 0, true), None);
}

#[test]
fn tuple_indices() {
    assert_eq!(translate_index_raw(-1, 3, false), Ok(2));
    assert_eq!(translate_index_raw(3, 3, false), Err(3));
    assert_eq!(translate_index_raw(-5, 3, true), Ok(0));
    assert_eq!(translate_index_raw(9, 3, true), Ok(3));
}

#[test]
fn slices() {
    let (mut s, mut e) = (0, 5);
    assert_eq!(slice_adjust_indices_raw(5, &mut s, &mut e, 2), 3);
    // [::-1] on length 4: start = 3, stop = -(4+1)
    let (mut s, mut e) = (3, -5);
    assert_eq!(slice_adjust_indices_raw(4, &mut s, &mut e, -1), 4);
    assert_eq!((s, e), (3, -1));
    let (mut s, mut e) = (1, -1);
    assert_eq!(slice_adjust_indices_raw(4, &mut s, &mut e, 1), 2);
}
