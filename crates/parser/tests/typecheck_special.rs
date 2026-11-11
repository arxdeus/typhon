//! Unit tests for typecheck special.cpp port (stdlib-free parts).

use typhon_parser::ast::IdExpr;
use typhon_parser::cache::Cache;
use typhon_parser::typecheck::{TypeContext, TypecheckVisitor};

#[test]
fn value_method_ids_are_stable() {
    let mut cache = Cache::new(None);
    let ctx = TypeContext::new(&mut cache, "x").into_ref();
    let mut tv = TypecheckVisitor::new(&mut cache, ctx, None, None);
    let a = tv.value_method_id("_hash_value");
    let b = tv.value_method_id("_equal_values");
    assert_ne!(a, b);
    assert_eq!(tv.value_method_id("_hash_value"), a);
    assert_eq!(tv.cache.thunk_ids.get("$any._hash_value"), Some(&a));
}

#[test]
fn prepare_vtables_empty_cache_is_noop() {
    let mut cache = Cache::new(None);
    let ctx = TypeContext::new(&mut cache, "x").into_ref();
    let mut tv = TypecheckVisitor::new(&mut cache, ctx, None, None);
    tv.prepare_vtables().unwrap();
    assert!(tv.cache.thunk_ids.is_empty());
}

#[test]
fn heterogenous_loop_on_identifier_has_empty_preamble() {
    let mut cache = Cache::new(None);
    let ctx = TypeContext::new(&mut cache, "x").into_ref();
    let mut tv = TypecheckVisitor::new(&mut cache, ctx, None, None);
    let it = tv.n_expr(IdExpr::new("t"));
    let block = tv.populate_static_heterogenous_tuple_loop(it, &["x".into()]).unwrap();
    // untyped identifier: no items, only the (empty) preamble placeholder
    assert_eq!(block.len(), 1);
    assert!(tv.cache.ast[block[0]].kind.as_suite().is_some_and(|s| s.items.is_empty()));
}
