//! AST tests: construction, cloning, S-expression / Typhon / Python printing,
//! serialization. Includes `ASTFormattingTest`.

use typhon_common::SrcInfo;
use typhon_parser::ast::*;

fn s() -> SrcInfo {
    SrcInfo::default()
}

fn id(a: &mut AstArena, n: &str) -> ExprId {
    a.new_expr(IdExpr::new(n), s())
}
fn int(a: &mut AstArena, v: i64) -> ExprId {
    a.new_expr(IntExpr::new(v), s())
}

/// TEST(ASTFormattingTest, FloatValues)
#[test]
fn ast_formatting_float_values() {
    let cases: &[(f64, &str)] = &[
        (0.0, "0.0"),
        (-0.0, "-0.0"),
        (1.0, "1.0"),
        (-1.5, "-1.5"),
        (0.1, "0.1"),
        (1.2345678901234567, "1.2345678901234567"),
        (1e-5, "1e-05"),
        (1e-4, "0.0001"),
        (-1e-4, "-0.0001"),
        (1e15, "1000000000000000.0"),
        (-1e15, "-1000000000000000.0"),
        (1e16, "1e+16"),
        (1e20, "1e+20"),
        (f64::MIN_POSITIVE, "2.2250738585072014e-308"),
        (f64::MAX, "1.7976931348623157e+308"),
        (5e-324, "5e-324"),
        (f64::INFINITY, "inf"),
        (f64::NEG_INFINITY, "-inf"),
        (f64::NAN, "nan"),
    ];
    let mut a = AstArena::new();
    for &(v, expected) in cases {
        let e = FloatExpr::new(v);
        let raw = e.raw_data().0;
        let x = a.new_expr(e, s());
        assert_eq!(a.expr_to_debug(x, false, -1, 0), format!("FloatExpr(value='{raw}', float_value={expected})"));
    }
}

#[test]
fn sexpr_basic() {
    let mut a = AstArena::new();
    let f = id(&mut a, "foo");
    let one = int(&mut a, 1);
    let k = int(&mut a, 2);
    let mut call = CallExpr { expr: Some(f), items: vec![a.call_arg("", Some(one)), a.call_arg("k", Some(k))], ..Default::default() };
    call.partial = false;
    let c = a.new_expr(call, s());
    assert_eq!(a.expr_to_string(c, -1), "(call 'foo (int 1) #:name 'k (int 2))");
    assert_eq!(a.expr_to_string(c, 0), "(call 'foo\n    (int 1)\n    #:name 'k\n    (int 2))");

    let st = a.new_expr(StringExpr::new("a\"b\n", "f"), s());
    assert_eq!(a.expr_to_string(st, -1), "(string (\"a\"b\\n\" #:prefix \"f\"))");

    let l = a.new_expr(ListExpr::default(), s());
    assert_eq!(a.expr_to_string(l, -1), "(list)");
    let d = a.new_expr(DictExpr::default(), s());
    assert_eq!(a.expr_to_string(d, -1), "(set)");

    let sl = a.new_expr(SliceExpr { start: Some(one), stop: None, step: Some(k) }, s());
    assert_eq!(a.expr_to_string(sl, -1), "(slice #:start (int 1) #:step (int 2))");

    // done marker
    a[one].done = true;
    assert_eq!(a.expr_to_string(one, -1), "(*int 1)");
    // typed (fallback type printer)
    a[k].ty = Some(typhon_parser::TypeId(7));
    assert_eq!(a.expr_to_string(k, -1), "(int 2 #:type \"T7\")");
    let y = a.new_expr(YieldExpr, s());
    assert_eq!(a.expr_to_string(y, -1), "yield-expr");
}

#[test]
fn sexpr_stmts() {
    let mut a = AstArena::new();
    let x = id(&mut a, "x");
    let one = int(&mut a, 1);
    let asg = a.new_stmt(AssignStmt { lhs: Some(x), rhs: Some(one), ..Default::default() }, s());
    assert_eq!(a.stmt_to_string(asg, -1), "(assign 'x (int 1))");
    let c = a.new_expr(BoolExpr { value: true }, s());
    let br = a.new_stmt(BreakStmt, s());
    let ifs = a.new_stmt(IfStmt { cond: Some(c), if_suite: Some(br), else_suite: None }, s());
    // suite slot got wrapped
    let suite = a[ifs].kind.as_if().unwrap().if_suite.unwrap();
    assert!(a[suite].kind.as_suite().is_some());
    assert_eq!(a.stmt_to_string(ifs, -1), "(if (bool 1))");
    assert_eq!(a.stmt_to_string(ifs, 0), "(if (bool 1) (suite \n    (break)))");
    let top = a.new_stmt(SuiteStmt { items: vec![asg, ifs] }, s());
    assert_eq!(a.stmt_to_string(top, 0), "(suite \n  (assign 'x (int 1))\n  (if (bool 1)\n    (suite \n      (break))))");
    let ret = a.new_stmt(ReturnStmt { expr: None }, s());
    assert_eq!(a.stmt_to_string(ret, -1), "(return)");
    let g = a.new_stmt(GlobalStmt { var: "v".into(), non_local: true }, s());
    assert_eq!(a.stmt_to_string(g, -1), "(nonlocal 'v)");
}

#[test]
fn function_params_and_clone() {
    let mut a = AstArena::new();
    let t = id(&mut a, "type");
    let tt = id(&mut a, "int");
    let st = id(&mut a, "Static");
    let stix = a.new_expr(IndexExpr { expr: Some(st), index: Some(tt) }, s());
    let p_t = a.new_param(s(), "T", Some(t), None, 0, ParamKind::Normal);
    assert!(p_t.is_generic());
    let p_n = a.new_param(s(), "N", Some(stix), None, 0, ParamKind::Normal);
    assert!(p_n.is_generic());
    let tref = id(&mut a, "T");
    let p_x = a.new_param(s(), "x", Some(tref), None, 0, ParamKind::Normal);
    let slash = a.new_param(s(), "/", None, None, 0, ParamKind::Normal);
    let star = a.new_param(s(), "*", None, None, 0, ParamKind::Normal);
    let p_y = a.new_param(s(), "y", None, None, 0, ParamKind::Normal);
    let body = a.new_stmt(PassLike::pass(), s());
    let f = a
        .new_function(s(), "f", None, vec![p_t, p_n, p_x, slash, star, p_y], Some(body), vec![], false)
        .unwrap();
    let func = a[f].kind.as_function().unwrap();
    assert_eq!(func.items.len(), 4);
    assert_eq!(func.items[2].kind, ParamKind::PositionalOnly);
    assert_eq!(func.items[3].kind, ParamKind::KeywordOnly);
    let ng = a.non_inferrable_generics(f);
    assert!(ng.contains("N") && !ng.contains("T"));

    // invalid params
    let p = a.new_param(s(), "*", None, None, 0, ParamKind::Normal);
    assert!(a.new_function(s(), "g", None, vec![p], None, vec![], false).is_err());

    // clone
    a[f].done = true;
    let c1 = a.clone_stmt(f, false);
    let c2 = a.clone_stmt(f, true);
    assert_ne!(c1, f);
    assert!(a[c1].done);
    assert!(!a[c2].done);
    assert_eq!(a.stmt_to_string(c1, 0), a.stmt_to_string(f, 0));
    let s1 = a[c1].kind.as_function().unwrap().suite.unwrap();
    let s0 = a[f].kind.as_function().unwrap().suite.unwrap();
    assert_ne!(s1, s0);
}

/// Helper: a `pass` is an empty suite in Typhon.
struct PassLike;
impl PassLike {
    fn pass() -> SuiteStmt {
        SuiteStmt::default()
    }
}

#[test]
fn expr_clone_clean() {
    let mut a = AstArena::new();
    let x = id(&mut a, "x");
    let y = id(&mut a, "y");
    let b = a.new_expr(BinaryExpr { op: "+".into(), lexpr: Some(x), rexpr: Some(y), in_place: false }, s());
    a[b].ty = Some(typhon_parser::TypeId(1));
    a[b].done = true;
    a[b].orig_expr = Some(x);
    a[b].attrs.set(Attr::ExprSequenceItem);
    let c = a.clone_expr(b, true);
    assert!(a[c].ty.is_none() && !a[c].done);
    assert_eq!(a[c].orig_expr, Some(x));
    assert!(a[c].attrs.has(Attr::ExprSequenceItem));
    let cb = a[c].kind.as_binary().unwrap();
    assert_ne!(cb.lexpr, Some(x));
    let d = a.clone_expr(b, false);
    assert_eq!(a[d].ty, Some(typhon_parser::TypeId(1)));
}

#[test]
fn generator() {
    let mut a = AstArena::new();
    let i = id(&mut a, "i");
    let r = id(&mut a, "r");
    let f = a.new_stmt(ForStmt { var: Some(i), iter: Some(r), ..Default::default() }, s());
    let cond = id(&mut a, "c");
    let ifs = a.new_stmt(IfStmt { cond: Some(cond), ..Default::default() }, s());
    let e = id(&mut a, "i");
    let g = a.new_generator(s(), GeneratorKind::ListGenerator, e, vec![f, ifs]);
    assert_eq!(a.generator_loop_count(g), 2);
    assert_eq!(a.generator_final_expr(g), Some(e));
    assert_eq!(a.expr_to_string(g, -1), "(list-gen (for 'i 'r))");
    assert_eq!(
        a.expr_to_python(g, false, -1, 0),
        "ListComp(elt=Name(id='i'), generators=[Comprehension(target=Name(id='i'), iter=Name(id='r'), ifs=[Name(id='c')])])"
    );
    let z = id(&mut a, "z");
    a.generator_set_final_expr(g, z);
    assert_eq!(a.generator_final_expr(g), Some(z));
    let cg = a.clone_expr(g, false);
    assert_eq!(a.generator_loop_count(cg), 2);
}

#[test]
fn debug_string_forms() {
    let mut a = AstArena::new();
    let x = id(&mut a, "x");
    let one = int(&mut a, 1);
    let b = a.new_expr(BinaryExpr { op: "+".into(), lexpr: Some(x), rexpr: Some(one), in_place: false }, s());
    assert_eq!(
        a.expr_to_debug(b, false, -1, 0),
        "BinaryExpr(lexpr=IdExpr(value='x'), op='+', rexpr=IntExpr(value='1', int_value=1), in_place=False)"
    );
    assert_eq!(
        a.expr_to_debug(b, false, 2, 0),
        "BinaryExpr(\n  lexpr=IdExpr(value='x'),\n  op='+',\n  rexpr=IntExpr(\n    value='1',\n    int_value=1),\n  in_place=False)"
    );
    a[x].attrs.set_str(Attr::Module, "m");
    a[x].attrs.set(Attr::ExprVirtual);
    assert_eq!(a.expr_to_debug(x, true, -1, 0), "IdExpr(value='x', attrs=(Module='m'))");
    let es = a.new_stmt(ExprStmt { expr: Some(one) }, s());
    let br = a.new_stmt(BreakStmt, s());
    let suite = a.new_stmt(SuiteStmt { items: vec![es, br] }, s());
    assert_eq!(a.stmt_to_debug(suite, false, -1, 0), "[ExprStmt(expr=IntExpr(value='1', int_value=1)), BreakStmt()]");
    let empty = a.new_stmt(SuiteStmt::default(), s());
    assert_eq!(a.stmt_to_debug(empty, false, -1, 0), "");
    // orig_expr cycles are cut
    a[x].orig_expr = Some(x);
    assert_eq!(a.expr_to_debug(x, false, -1, 0), "IdExpr(value='x')");
}

#[test]
fn python_string_forms() {
    let mut a = AstArena::new();
    let x = id(&mut a, "x");
    let one = int(&mut a, 1);
    let asg = a.new_stmt(AssignStmt { lhs: Some(x), rhs: Some(one), ..Default::default() }, s());
    assert_eq!(a.stmt_to_python(asg, false, -1, 0), "Assign(targets=[Name(id='x')], value=Num(value=1))");
    let w = a.new_expr(IdExpr::new("w"), s());
    a[w].attrs.set(Attr::Internal);
    assert_eq!(a.expr_to_python(w, true, -1, 0), "Name(id='w', attributes=[Internal])");
    let what = id(&mut a, "c");
    let from = id(&mut a, "a");
    let fromd = a.new_expr(DotExpr { expr: Some(from), member: "b".into() }, s());
    let imp = a.new_stmt(ImportStmt { from: Some(fromd), what: Some(what), dots: 1, ..Default::default() }, s());
    assert_eq!(
        a.stmt_to_python(imp, false, -1, 0),
        "ImportFrom(module=a.b, names=[Alias(name=c, params=Arguments())], level=1)"
    );
    assert_eq!(a.stmt_to_string(imp, -1), "(import (dot 'a 'b) #:what 'c #:dots 1)");
}

#[test]
fn serialization_roundtrip() {
    let mut a = AstArena::new();
    let x = id(&mut a, "x");
    let one = int(&mut a, 1);
    let mut part = StringPart::new("v", "f", Some(x));
    part.format.spec = ">3".into();
    let st = a.new_expr(StringExpr::from_parts(vec![part]), s());
    let es = a.new_stmt(ExprStmt { expr: Some(st) }, s());
    let asg = a.new_stmt(AssignStmt { lhs: Some(x), rhs: Some(one), ..Default::default() }, SrcInfo::new("f.py", 3, 4, 5));
    a[asg].attrs.set_str(Attr::Module, "mod");
    let p = a.new_param(s(), "a", None, Some(one), 0, ParamKind::Normal);
    let f = a.new_function(s(), "fn", None, vec![p], Some(asg), vec![], false).unwrap();
    let top = a.new_stmt(SuiteStmt { items: vec![es, f] }, s());
    let bytes = a.serialize_stmt(top).unwrap();
    let (b, root) = AstArena::deserialize_stmt(&bytes).unwrap();
    assert_eq!(b.stmt_to_debug(root, true, 2, 0), a.stmt_to_debug(top, true, 2, 0));
    assert_eq!(b.stmt_to_string(root, 0), a.stmt_to_string(top, 0));
    let mut c = AstArena::new();
    let r2 = c.import_stmt(&b, root);
    assert_eq!(c.stmt_to_string(r2, 0), a.stmt_to_string(top, 0));
    let fs = b[root].kind.as_suite().unwrap().items[1];
    let suite = b[fs].kind.as_function().unwrap().suite.unwrap();
    let asg2 = b[suite].kind.as_suite().unwrap().items[0];
    assert_eq!(b[asg2].src.line, 3);
    assert_eq!(b[asg2].attrs.get_str(Attr::Module), Some("mod"));

    let eb = a.serialize_expr(st).unwrap();
    let (eb2, er) = AstArena::deserialize_expr(&eb).unwrap();
    assert_eq!(eb2.expr_to_debug(er, false, -1, 0), a.expr_to_debug(st, false, -1, 0));
}

#[test]
fn pipe_marks_ellipsis() {
    let mut a = AstArena::new();
    let x = id(&mut a, "x");
    let f = id(&mut a, "f");
    let el = a.new_expr(EllipsisExpr::default(), s());
    let call = a.new_call(s(), f, &[el]);
    let p = a.new_expr(
        PipeExpr { items: vec![Pipe { op: String::new(), expr: Some(x) }, Pipe { op: "|>".into(), expr: Some(call) }], ..Default::default() },
        s(),
    );
    assert!(a[el].kind.as_ellipsis().unwrap().is_pipe());
    assert_eq!(a.expr_to_string(p, -1), "(pipe ('x \"\") ((call 'f (ellipsis #:pipe)) \"|>\"))");
}

#[test]
fn suite_helpers() {
    let mut a = AstArena::new();
    let b1 = a.new_stmt(BreakStmt, s());
    let b2 = a.new_stmt(ContinueStmt, s());
    let inner = a.new_stmt(SuiteStmt { items: vec![b2] }, s());
    let outer = a.new_stmt(SuiteStmt { items: vec![inner, b1] }, s());
    assert_eq!(a.first_in_block(outer), Some(b2));
    a.suite_flatten(outer);
    assert_eq!(a[outer].kind.as_suite().unwrap().items, vec![b2, b1]);
    a[outer].done = true;
    let c = a.new_stmt(CommentStmt { comment: "hi".into() }, s());
    a.suite_add_stmt(outer, Some(c));
    assert!(!a[outer].done);
    assert_eq!(a.wrap_suite(Some(outer)), Some(outer));
}
