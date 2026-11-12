//! Tests for `visitors::doc` JSON helpers and the visitor on hand-built ASTs.

use serde_json::json;
use typhon_common::srcinfo::SrcInfo;
use typhon_parser::ast::*;
use typhon_parser::cache::Cache;
use typhon_parser::visitors::doc::{DocShared, DocVisitor, json_escape, json_to_string};

#[test]
fn json_format() {
    assert_eq!(json_escape("a\"b\n"), "a\\\"b\\n");
    assert_eq!(json_to_string(&json!({})), "{}");
    assert_eq!(json_to_string(&json!(["1", "2"])), "[ \"1\", \"2\" ]");
    assert_eq!(json_to_string(&json!({"a": "x", "b": ["y"]})), "{ \"a\": \"x\", \"b\": [ \"y\" ] }");
}

#[test]
fn module_with_function_and_variable() {
    let mut cache = Cache::new(None);
    let s = SrcInfo::new("m.py", 1, 1, 3);
    let doc = cache.ast.new_expr(StringExpr::new("Docs", ""), s.clone());
    let ds = cache.ast.new_stmt(ExprStmt { expr: Some(doc) }, s.clone());
    let body = cache.ast.new_stmt(SuiteStmt { items: vec![ds] }, s.clone());
    let f = cache.ast.new_stmt(FunctionStmt { name: "foo".into(), suite: Some(body), ..Default::default() }, s.clone());
    let lhs = cache.ast.new_expr(IdExpr::new("x"), s.clone());
    let rhs = cache.ast.new_expr(IntExpr::new(5), s.clone());
    let a = cache.ast.new_stmt(AssignStmt { lhs: Some(lhs), rhs: Some(rhs), ..Default::default() }, s.clone());
    let top = cache.ast.new_stmt(SuiteStmt { items: vec![f, a] }, s);

    let mut shared = DocShared::new("typhon", cache);
    let ctx = shared.new_context("m.py");
    DocVisitor::new(&mut shared, ctx).transform_module(top).unwrap();
    let j = serde_json::Value::Object(shared.j.clone());
    assert_eq!(j["1"]["kind"], "function");
    assert_eq!(j["1"]["name"], "foo");
    assert_eq!(j["1"]["doc"], "Docs");
    assert_eq!(j["2"]["kind"], "variable");
    assert_eq!(j["2"]["value"], "5");
    assert_eq!(j["3"]["kind"], "module");
    assert_eq!(j["3"]["children"], json!(["1", "2"]));
}
