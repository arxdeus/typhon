//! Unit tests for `visitors::scoping` on hand-built ASTs.

use typhon_common::srcinfo::SrcInfo;
use typhon_parser::ast::*;
use typhon_parser::cache::Cache;
use typhon_parser::visitors::scoping::{ScopingVisitor, install_hooks};

fn src(line: i32) -> SrcInfo {
    SrcInfo::new("t.py", line, 1, 1)
}
fn id(c: &mut Cache, n: &str, l: i32) -> ExprId {
    c.ast.new_expr(IdExpr::new(n), src(l))
}
fn assign(c: &mut Cache, n: &str, v: i64, l: i32) -> StmtId {
    let lhs = id(c, n, l);
    let rhs = c.ast.new_expr(IntExpr::new(v), src(l));
    c.ast.new_stmt(AssignStmt { lhs: Some(lhs), rhs: Some(rhs), ..Default::default() }, src(l))
}
fn use_(c: &mut Cache, n: &str, l: i32) -> StmtId {
    let e = id(c, n, l);
    c.ast.new_stmt(ExprStmt { expr: Some(e) }, src(l))
}
fn suite(c: &mut Cache, items: Vec<StmtId>) -> StmtId {
    c.ast.new_stmt(SuiteStmt { items }, src(0))
}
fn func(c: &mut Cache, name: &str, body: Vec<StmtId>, l: i32) -> StmtId {
    let s = suite(c, body);
    c.ast.new_stmt(FunctionStmt { name: name.into(), suite: Some(s), ..Default::default() }, src(l))
}
fn global(c: &mut Cache, n: &str, nl: bool, l: i32) -> StmtId {
    c.ast.new_stmt(GlobalStmt { var: n.into(), non_local: nl }, src(l))
}
fn err(c: &mut Cache, s: StmtId) -> String {
    ScopingVisitor::apply(c, s, None, false).unwrap_err().parser_errors().unwrap().message()
}

#[test]
fn function_bindings_and_captures() {
    let mut c = Cache::new(None);
    let a = assign(&mut c, "y", 1, 2);
    let u = use_(&mut c, "x", 3);
    let f = func(&mut c, "f", vec![a, u], 1);
    let gx = assign(&mut c, "x", 0, 0);
    let top = suite(&mut c, vec![gx, f]);
    ScopingVisitor::apply(&mut c, top, None, false).unwrap();
    let b = c.ast[f].attrs.get_bindings().unwrap();
    assert_eq!(b.captures.get("x"), Some(&CaptureType::Read));
    assert_eq!(b.bindings["y"].count, 1);
    assert!(b.bindings.contains_key("f"));
    assert!(c.ast[a].attrs.get_int(Attr::ExprTime).is_some());
}

#[test]
fn global_and_nonlocal() {
    let mut c = Cache::new(None);
    let g = global(&mut c, "x", false, 2);
    let a = assign(&mut c, "x", 1, 3);
    let f = func(&mut c, "f", vec![g, a], 1);
    let top = suite(&mut c, vec![f]);
    ScopingVisitor::apply(&mut c, top, None, false).unwrap();
    let b = c.ast[f].attrs.get_bindings().unwrap();
    assert_eq!(b.captures.get("x"), Some(&CaptureType::Global));
    let lhs = c.ast[a].kind.as_assign().unwrap().lhs.unwrap();
    assert!(c.ast[lhs].attrs.has(Attr::ExprDominated));
}

#[test]
fn errors_match_cpp() {
    let mut c = Cache::new(None);
    let g = global(&mut c, "x", false, 1);
    let top = suite(&mut c, vec![g]);
    assert_eq!(err(&mut c, top), "'global' outside function");

    let mut c = Cache::new(None);
    let g = global(&mut c, "x", true, 1);
    let top = suite(&mut c, vec![g]);
    assert_eq!(err(&mut c, top), "'nonlocal' outside function");

    let mut c = Cache::new(None);
    let a = assign(&mut c, "x", 1, 2);
    let g = global(&mut c, "x", false, 3);
    let f = func(&mut c, "f", vec![a, g], 1);
    let top = suite(&mut c, vec![f]);
    assert_eq!(err(&mut c, top), "name 'x' is assigned to before global declaration");

    let mut c = Cache::new(None);
    let u = use_(&mut c, "x", 2);
    let a = assign(&mut c, "x", 1, 3);
    let f = func(&mut c, "f", vec![u, a], 1);
    let top = suite(&mut c, vec![f]);
    assert_eq!(err(&mut c, top), "local variable 'x' referenced before assignment at t.py:3:1");
}

#[test]
fn conditional_binding_dominated() {
    // if c: x = 1 ; print(x)  -> suite gets binding x with count 1 (x__used__)
    let mut c = Cache::new(None);
    let a = assign(&mut c, "x", 1, 2);
    let isuite = suite(&mut c, vec![a]);
    let cond = id(&mut c, "c", 1);
    let ifs = c.ast.new_stmt(IfStmt { cond: Some(cond), if_suite: Some(isuite), else_suite: None }, src(1));
    let u = use_(&mut c, "x", 3);
    let top = suite(&mut c, vec![ifs, u]);
    ScopingVisitor::apply(&mut c, top, None, false).unwrap();
    let b = c.ast[top].attrs.get_bindings().unwrap();
    assert_eq!(b.bindings["x"].count, 1);
    let lhs = c.ast[a].kind.as_assign().unwrap().lhs.unwrap();
    assert!(c.ast[lhs].attrs.has(Attr::ExprDominatedUsed));
}

#[test]
fn comprehension_renames_and_shadows() {
    // [i for i in y]
    let mut c = Cache::new(None);
    let var = id(&mut c, "i", 1);
    let iter = id(&mut c, "y", 1);
    let fin = id(&mut c, "i", 1);
    let fs = c.ast.new_stmt(ExprStmt { expr: Some(fin) }, src(1));
    let fsuite = suite(&mut c, vec![fs]);
    let f = c.ast.new_stmt(ForStmt { var: Some(var), iter: Some(iter), suite: Some(fsuite), ..Default::default() }, src(1));
    let g = c.ast.new_expr(GeneratorExpr { kind: GeneratorKind::ListGenerator, loops: Some(f) }, src(1));
    let ya = assign(&mut c, "y", 1, 0);
    let es = c.ast.new_stmt(ExprStmt { expr: Some(g) }, src(1));
    let top = suite(&mut c, vec![ya, es]);
    let mut shadows = Default::default();
    ScopingVisitor::apply(&mut c, top, Some(&mut shadows), false).unwrap();
    let v = c.ast[var].kind.as_id().unwrap().value.clone();
    assert!(v.starts_with("%_i_"), "{v}");
    assert_eq!(c.ast[fin].kind.as_id().unwrap().value, v);
    assert_eq!(c.ast[iter].kind.as_id().unwrap().value, "y");
}

#[test]
fn install_hook_works() {
    let mut c = Cache::new(None);
    install_hooks(&mut c);
    let a = assign(&mut c, "x", 1, 1);
    let top = suite(&mut c, vec![a]);
    let h = c.hooks.scoping.clone().unwrap();
    h(&mut c, top, None).unwrap();
}
