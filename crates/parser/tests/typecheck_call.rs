//! Unit tests for typecheck call/access/collections (stdlib-free parts).

use typhon_parser::ast::{CallArg, CallExpr, EllipsisExpr, EllipsisType, IdExpr, IntExpr, KeywordStarExpr, StarExpr};
use typhon_parser::cache::Cache;
use typhon_parser::typecheck::{TypeContext, TypecheckVisitor};

fn call(cache: &mut Cache, items: Vec<CallArg>) -> typhon_parser::ids::ExprId {
    let f = cache.ast.new_expr(IdExpr::new("f"), Default::default());
    cache.ast.new_expr(CallExpr { expr: Some(f), items, ordered: false, partial: false }, Default::default())
}

#[test]
fn validate_call_errors() {
    let mut cache = Cache::new(None);
    let i1 = cache.ast.new_expr(IntExpr::new(1), Default::default());
    let i2 = cache.ast.new_expr(IntExpr::new(2), Default::default());
    let bad_order = call(&mut cache, vec![CallArg::new("x", Some(i1)), CallArg::new("", Some(i2))]);
    let s = cache.ast.new_expr(StarExpr { expr: Some(i1) }, Default::default());
    let named_star = call(&mut cache, vec![CallArg::new("x", Some(s))]);
    let e1 = cache.ast.new_expr(EllipsisExpr { mode: EllipsisType::Partial }, Default::default());
    let e2 = cache.ast.new_expr(EllipsisExpr { mode: EllipsisType::Partial }, Default::default());
    let two_ell = call(&mut cache, vec![CallArg::new("", Some(e1)), CallArg::new("", Some(e2))]);
    let k = cache.ast.new_expr(KeywordStarExpr { expr: Some(i2) }, Default::default());
    let ok = call(&mut cache, vec![CallArg::new("x", Some(i1)), CallArg::new("", Some(k))]);

    let ctx = TypeContext::new(&mut cache, "x").into_ref();
    let mut tv = TypecheckVisitor::new(&mut cache, ctx, None, None);
    let e = tv.validate_call(bad_order).unwrap_err().to_string();
    assert!(e.contains("positional argument follows keyword argument"), "{e}");
    let e = tv.validate_call(named_star).unwrap_err().to_string();
    assert!(e.contains("cannot use starred expression here"), "{e}");
    assert!(tv.validate_call(two_ell).is_err());
    tv.validate_call(ok).unwrap();
    assert!(tv.cache.ast[ok].attrs.has(typhon_parser::ast::Attr::Validated));
}

#[test]
fn star_outside_call() {
    let mut cache = Cache::new(None);
    let i = cache.ast.new_expr(IntExpr::new(1), Default::default());
    let s = cache.ast.new_expr(StarExpr { expr: Some(i) }, Default::default());
    let k = cache.ast.new_expr(KeywordStarExpr { expr: Some(i) }, Default::default());
    let ctx = TypeContext::new(&mut cache, "x").into_ref();
    let mut tv = TypecheckVisitor::new(&mut cache, ctx, None, None);
    assert!(tv.visit_star(s).unwrap_err().to_string().contains("star"));
    assert!(tv.visit_keyword_star(k).unwrap_err().to_string().contains("kwstar"));
}

#[test]
fn id_not_found() {
    let mut cache = Cache::new(None);
    let x = cache.ast.new_expr(IdExpr::new("nope"), Default::default());
    let ctx = TypeContext::new(&mut cache, "x").into_ref();
    let mut tv = TypecheckVisitor::new(&mut cache, ctx, None, None);
    let e = tv.visit_id(x).unwrap_err().to_string();
    assert!(e.contains("name 'nope' is not defined"), "{e}");
}
