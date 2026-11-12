//! Unit tests for the PEG parser .

use typhon_common::SrcInfo;
use typhon_parser::ast::*;
use typhon_parser::peg::{ParseOptions, parse_code, parse_expr, parse_openmp};

fn sx(code: &str) -> String {
    sx_opts(code, &ParseOptions::default())
}
fn sx_opts(code: &str, o: &ParseOptions) -> String {
    let mut a = AstArena::new();
    let s = parse_code(&mut a, o, "t.py", code, 0).unwrap_or_else(|e| panic!("{code}: {e}"));
    a.stmt_to_string(s, 0).split_whitespace().collect::<Vec<_>>().join(" ")
}
fn err(code: &str) -> String {
    let mut a = AstArena::new();
    match parse_code(&mut a, &ParseOptions::default(), "t.py", code, 0) {
        Ok(s) => panic!("expected error, got {}", a.stmt_to_string(s, 0)),
        Err(e) => e.to_string().trim().to_string(),
    }
}
fn has(code: &str, needle: &str) {
    let s = sx(code);
    assert!(s.contains(needle), "{code:?}\n  => {s}\n  missing {needle}");
}

#[test]
fn peg_simple_statements() {
    has("x = 1", "(assign 'x (int 1))");
    has("a = b = 2", "(assign 'b (int 2))");
    has("x: int = 3", "(assign 'x (int 3) #:type 'int)");
    has("x += 1", "(binary \"+\" 'x (int 1) #:in-place)");
    has("pass; break; continue", "(break)");
    has("del a, b", "(del 'b)");
    has("assert x, 'msg'", "(assert 'x");
    has("return a, b", "(return (tuple 'a 'b))");
    has("raise E from y", "(throw 'E :from 'y)");
    has("global a, b", "(global 'b)");
    has("nonlocal a", "(nonlocal 'a)");
    has("yield from x", "yield-from");
    has("print a, b,", "(print #:inline");
}

#[test]
fn peg_imports() {
    has("import a.b as c", "(import (dot 'a 'b) #:as 'c)");
    has("from .. import *", "#:dots 2");
    has("from a import b as c, d", "(import 'a #:what 'd)");
    has("from C import foo(int) -> float", "(import 'C #:what 'foo #:args (( #:type 'int)) #:ret 'float)");
    has("from C import x: int", "'int");
}

#[test]
fn peg_compound() {
    has("if a:\n  x\nelif b:\n  y\nelse:\n  z\n", "(if 'a");
    has("while x:\n  pass\nelse:\n  y\n", "(while-else 'x");
    has("for i in range(3):\n  print i\n", "(for 'i (call 'range (int 3))");
    has("try:\n  x\nexcept E as e:\n  y\nfinally:\n  z\n", "(try");
    has("with open(f) as g, h:\n  pass\n", "(with");
    has("class A[T](B):\n  x: int\n  def f(self): pass\n", "(class 'A");
    has("@dec\ndef f[T](a: int = 1, *args, **kw) -> int:\n  return a\n", "(fn 'f ((a #:type 'int #:default (int 1)) (*args) (**kw) (T #:type 'type #:generic))");
    has("async def f():\n  await g()\n", "(await (call 'g))");
    has("match x:\n  case 1 | 2 if y:\n    pass\n  case [a, b] as z:\n    pass\n", "(match 'x");
    has("@llvm\ndef f(a: int) -> int:\n  %0 = add i64 %a, 1\n  ret i64 %0\n", "add i64");
}

#[test]
fn peg_expressions() {
    has("a if b else c", "(if-expr 'b 'a 'c)");
    has("not a and b or c", "(binary \"||\" (binary \"&&\" (unary \"!\" 'a) 'b) 'c)");
    has("1 < x <= 3", "(chain");
    has("a is not b", "\"is not\"");
    has("a not in b", "\"not in\"");
    has("-x ** 2", "(unary \"-\" (binary \"**\" 'x (int 2)))");
    has("x |> f(...) ||> g", "(pipe");
    has("a[1:2:3, ::]", "(slice");
    has("f(a, *b, c=1, **d)", "(call 'f");
    has("f(a, ...)", "#:partial");
    has("(x := 5)", "(assign-expr");
    has("{1: 2, **d}", "(dict");
    has("{1, 2}", "(set");
    has("[]", "(list)");
    has("()", "(tuple )");
    has("0xff + 1_000 + 0b1 + 0o7 + 1.5e3 + 3u", "(int 3 #:suffix \"u\")");
    has("0x_ff", "(int 0 #:suffix \"x_ff\")");
    has("1 ... 3", "(range");
    has("x.y.z", "(dot (dot 'x 'y) 'z)");
}

#[test]
fn peg_lambdas_and_comprehensions() {
    has("lambda x, y=1: x + y", "(lambda");
    has("lambda: 0", "(lambda ()");
    has("[x for x in y if x]", "list-gen");
    has("{k: v for k, v in d}", "dict-gen");
    has("{x for x in y}", "set-gen");
    has("f(x for x in y)", "(call 'f (gen");
}

#[test]
fn peg_strings() {
    has("'a\\nb'", "\"a\\nb\"");
    has("r'a\\n'", "\"a\\\\n\"");
    has("b'xy'", "#:prefix \"b\"");
    has("f'{x}' 'y'", "#:prefix \"f\"");
    has("'''multi\nline'''", "multi");
}

#[test]
fn peg_line_continuation_and_comments() {
    has("x = (1 +\n 2)", "(binary \"+\" (int 1) (int 2))");
    has("x = 1 + \\\n 2", "(binary \"+\" (int 1) (int 2))");
    has("x = 1  # comment\n", "(assign 'x (int 1))");
    has("## typhon: foo = 1\n", "(directive");
}

#[test]
fn peg_srcinfo() {
    let mut a = AstArena::new();
    let s = parse_code(&mut a, &ParseOptions::default(), "f", "\nx = foo", 10).unwrap();
    let st = a.first_in_block(s).unwrap();
    let src = a[st].src;
    assert_eq!((src.line, src.col, src.len), (12, 1, 7));
    let rhs = a[st].kind.as_assign().unwrap().rhs.unwrap();
    assert_eq!((a[rhs].src.line, a[rhs].src.col, a[rhs].src.len), (12, 5, 3));
}

#[test]
fn peg_errors() {
    assert_eq!(err("x = = 1"), "t.py:1:5: syntax error, unexpected '='");
    assert_eq!(err("if x:\npass\n"), "t.py:2:1: unexpected indentation");
    assert!(err("x = 1\n  y = 2\n").contains("syntax error"));
    assert!(err("def f(*, a, *, b): pass").contains("'*' may appear only once"));
    assert_eq!(err("x = '\\xZZ'"), "t.py:1:5: invalid code in a string: invalid \\x code");
    assert_eq!(err("def f():\n    x\n  y\n"), "t.py:3:3: syntax error, unexpected 'y'");
}

#[test]
fn peg_fstring_expr() {
    let mut a = AstArena::new();
    let o = ParseOptions::default();
    let src = SrcInfo::new("f", 3, 4, 0);
    let (e, fs) = parse_expr(&mut a, &o, " x + 1 = !r:>10", &src).unwrap();
    assert_eq!(a.expr_to_string(e, -1), "(binary \"+\" 'x (int 1))");
    assert_eq!((fs.text.as_str(), fs.conversion.as_str(), fs.spec.as_str()), ("x + 1 = ", "r", ">10"));
    let (_, fs) = parse_expr(&mut a, &o, "y:.2f", &src).unwrap();
    assert_eq!(fs.spec, ".2f");
}

#[test]
fn peg_openmp() {
    let mut a = AstArena::new();
    let o = ParseOptions::default();
    let v = parse_openmp(&mut a, &o, "omp parallel schedule(dynamic, 4) num_threads(8) ordered", &SrcInfo::default()).unwrap();
    let names: Vec<_> = v.iter().map(|x| x.name.as_str()).collect();
    assert_eq!(names, ["schedule", "chunk_size", "num_threads", "ordered"]);
    assert!(parse_openmp(&mut a, &o, "bogus", &SrcInfo::default()).unwrap_err().to_string().contains("openmp: syntax error"));
}
