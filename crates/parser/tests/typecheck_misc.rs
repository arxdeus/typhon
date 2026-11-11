//! Unit tests for typecheck import/loops/error (stdlib-free parts).

use typhon_parser::ast::{BreakStmt, ContinueStmt, DotExpr, IdExpr};
use typhon_parser::cache::Cache;
use typhon_parser::typecheck::{TypeContext, TypecheckVisitor};

fn setup() -> Cache {
    Cache::new(None)
}

#[test]
fn import_path_components() {
    let mut cache = setup();
    let a = cache.ast.new_expr(IdExpr::new("a"), Default::default());
    let ab = cache.ast.new_expr(DotExpr { expr: Some(a), member: "b".into() }, Default::default());
    let abc = cache.ast.new_expr(DotExpr { expr: Some(ab), member: "c".into() }, Default::default());
    let ctx = TypeContext::new(&mut cache, "x").into_ref();
    let tv = TypecheckVisitor::new(&mut cache, ctx, None, None);
    assert_eq!(tv.get_import_path(Some(abc), 0), vec!["a", "b", "c"]);
    assert_eq!(tv.get_import_path(Some(abc), 1), vec!["a", "b", "c"]);
    assert_eq!(tv.get_import_path(Some(a), 3), vec!["..", "..", "a"]);
    assert!(tv.get_import_path(None, 1).is_empty());
}

#[test]
fn break_continue_outside_loop() {
    let mut cache = setup();
    let b = cache.ast.new_stmt(BreakStmt, Default::default());
    let c = cache.ast.new_stmt(ContinueStmt, Default::default());
    let ctx = TypeContext::new(&mut cache, "x").into_ref();
    let mut tv = TypecheckVisitor::new(&mut cache, ctx, None, None);
    let e = tv.visit_break(b).unwrap_err();
    assert!(e.to_string().contains("'break' outside loop"), "{e}");
    let e = tv.visit_continue(c).unwrap_err();
    assert!(e.to_string().contains("'continue' outside loop"), "{e}");
}
