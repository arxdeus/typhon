//! Unit tests for typecheck class.rs / function.rs (stdlib-free parts).
//!
//! `FieldInitializationTest.*` and `EnumConstructionTest.*` (test/types.cpp) need the full
//! compiler pipeline with the stdlib (`Compiler::parse_code`); they belong to the e2e lane.

use typhon_parser::ast::{Attr, BreakStmt, ExprStmt, IdExpr, IntExpr, Param, ParamKind, StringExpr, SuiteStmt};
use typhon_parser::cache::Cache;
use typhon_parser::typecheck::{TypeContext, TypecheckVisitor};

fn p(cache: &mut Cache, name: &str, ty: Option<&str>, def: bool, kind: ParamKind) -> Param {
    let t = ty.map(|t| cache.ast.new_expr(IdExpr::new(t), Default::default()));
    let d = def.then(|| cache.ast.new_expr(IntExpr::new(1), Default::default()));
    cache.ast.new_param(Default::default(), name, t, d, 0, kind)
}

#[test]
fn call_layout() {
    let mut cache = Cache::new(None);
    let params = vec![
        p(&mut cache, "a", None, false, ParamKind::Normal),
        p(&mut cache, "b", None, true, ParamKind::Normal),
        p(&mut cache, "*args", None, false, ParamKind::Normal),
        p(&mut cache, "c", None, false, ParamKind::KeywordOnly),
        p(&mut cache, "$cap", None, false, ParamKind::Normal),
        p(&mut cache, "T", Some("type"), false, ParamKind::Normal),
        p(&mut cache, "**kw", None, false, ParamKind::Normal),
    ];
    let f = cache.ast.new_function(Default::default(), "f", None, params, None, vec![], false).unwrap();
    let ctx = TypeContext::new(&mut cache, "x").into_ref();
    let mut tv = TypecheckVisitor::new(&mut cache, ctx, None, None);
    tv.prepare_call_layout(f);
    let l = tv.cache.ast[f].kind.as_function().unwrap().call_layout().clone();
    assert_eq!(l.positional_limit, 2);
    assert_eq!(l.star, 2);
    assert_eq!(l.kwstar, 6);
    assert_eq!(l.variadic_penalty, -4);
    assert_eq!(l.optional, vec![false, true, false, false, true, false, false]);
    assert_eq!(l.required_values, vec![0, 3]);
    let kws: Vec<_> = l.keywords.iter().map(|(n, i)| (n.as_str(), *i)).collect();
    assert_eq!(kws, vec![("a", 0), ("b", 1), ("args", 2), ("c", 3), ("$cap", 4), ("T", 5), ("kw", 6)]);
}

#[test]
fn class_methods_and_bad_attr() {
    let mut cache = Cache::new(None);
    let doc = cache.ast.new_expr(StringExpr::new("doc", ""), Default::default());
    let doc = cache.ast.new_stmt(ExprStmt { expr: Some(doc) }, Default::default());
    let f = cache.ast.new_function(Default::default(), "m", None, vec![], None, vec![], false).unwrap();
    let inner = cache.ast.new_stmt(SuiteStmt { items: vec![f] }, Default::default());
    let suite = cache.ast.new_stmt(SuiteStmt { items: vec![doc, inner] }, Default::default());
    let bad = cache.ast.new_stmt(BreakStmt, Default::default());
    let bad_suite = cache.ast.new_stmt(SuiteStmt { items: vec![bad] }, Default::default());
    let ctx = TypeContext::new(&mut cache, "x").into_ref();
    let tv = TypecheckVisitor::new(&mut cache, ctx, None, None);
    assert_eq!(tv.get_class_methods(suite).unwrap(), vec![f]);
    let e = tv.get_class_methods(bad_suite).unwrap_err();
    assert!(e.to_string().contains("unexpected expression in class definition"), "{e}");
}

#[test]
fn llvm_definition_braces() {
    let mut cache = Cache::new(None);
    let code = "\"\"\"doc\"\"\"\n%0 = add i64 %a, 1 ; {x}\nret i64 %0";
    let s = cache.ast.new_expr(StringExpr::new(code, ""), Default::default());
    let s = cache.ast.new_stmt(ExprStmt { expr: Some(s) }, Default::default());
    let ctx = TypeContext::new(&mut cache, "x").into_ref();
    let mut tv = TypecheckVisitor::new(&mut cache, ctx, None, None);
    let r = tv.transform_llvm_definition(s).unwrap().unwrap();
    let items = tv.cache.ast[r].kind.as_suite().unwrap().items.clone();
    assert_eq!(items.len(), 1);
    let e = tv.cache.ast[items[0]].kind.as_expr().unwrap().expr.unwrap();
    let v = tv.cache.ast[e].kind.as_string().unwrap().value();
    assert_eq!(v, "\n%0 = add i64 %a, 1 ; {{x}}\nret i64 %0");
}

#[test]
fn llvm_definition_unbalanced() {
    let mut cache = Cache::new(None);
    let s = cache.ast.new_expr(StringExpr::new("{=a {=b}", ""), Default::default());
    let s = cache.ast.new_stmt(ExprStmt { expr: Some(s) }, Default::default());
    let ctx = TypeContext::new(&mut cache, "x").into_ref();
    let mut tv = TypecheckVisitor::new(&mut cache, ctx, None, None);
    let e = tv.transform_llvm_definition(s).unwrap_err();
    assert!(e.to_string().contains("invalid LLVM code"), "{e}");
}

#[test]
fn return_outside_function() {
    let mut cache = Cache::new(None);
    let r = cache.ast.new_stmt(typhon_parser::ast::ReturnStmt { expr: None }, Default::default());
    let i = cache.ast.new_stmt(typhon_parser::ast::ReturnStmt { expr: None }, Default::default());
    cache.ast[i].attrs.set(Attr::Internal);
    let ctx = TypeContext::new(&mut cache, "x").into_ref();
    let mut tv = TypecheckVisitor::new(&mut cache, ctx, None, None);
    let e = tv.visit_return(r).unwrap_err();
    assert!(e.to_string().contains("'return' outside function"), "{e}");
}
