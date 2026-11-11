//! Oracle test: compare our AST dump (`stmt_to_codon`) with the C++
//! `codon_ast_dump_code` (parse + scoping, no typecheck) from build/libcodonc.dylib.
//! Uses raw `dlopen` (no extra dev-dependency). Skipped if the library is missing.

use std::ffi::{CStr, CString, c_char, c_int, c_void};
use typhon_parser::ast::AstArena;
use typhon_parser::peg::{ParseOptions, parse_code};

#[repr(C)]
struct DumpResult {
    output: *mut c_char,
    error: *mut c_char,
}
type DumpFn = unsafe extern "C" fn(*const c_char, *const c_char, c_int, u8, c_int, c_int) -> DumpResult;
unsafe extern "C" {
    fn dlopen(path: *const c_char, mode: c_int) -> *mut c_void;
    fn dlsym(h: *mut c_void, s: *const c_char) -> *mut c_void;
}

pub fn oracle() -> Option<impl Fn(&str) -> Result<String, String>> {
    let p = CString::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../../../build/libcodonc.dylib")).unwrap();
    // SAFETY: FFI to the C API declared in codon/parser/capi.h.
    let h = unsafe { dlopen(p.as_ptr(), 2) };
    if h.is_null() {
        return None;
    }
    let s = unsafe { dlsym(h, c"codon_ast_dump_code".as_ptr()) };
    if s.is_null() {
        return None;
    }
    let f: DumpFn = unsafe { std::mem::transmute(s) };
    Some(move |code: &str| {
        let c = CString::new(code).unwrap();
        let r = unsafe { f(c.as_ptr(), c"".as_ptr(), 0, 0, -1, 0) };
        let take = |p: *mut c_char| (!p.is_null()).then(|| unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned());
        match (take(r.output), take(r.error)) {
            (Some(o), _) => Ok(o),
            (_, e) => Err(e.unwrap_or_default()),
        }
    })
}

pub fn ours(code: &str) -> Result<String, String> {
    let mut a = AstArena::new();
    parse_code(&mut a, &ParseOptions::default(), "", code, 0)
        .map(|s| a.stmt_to_codon(s, false, -1, 0))
        .map_err(|e| e.to_string())
}

#[test]
fn peg_oracle_simple() {
    let Some(cpp) = oracle() else {
        eprintln!("libcodonc.dylib not available; skipping");
        return;
    };
    let cases = [
        "x = 1", "a, b = c", "print a, b", "assert x, 'm'", "del a",
        "f(a, *b, c=1, **d)", "x[1:2, ::3]", "a if b else c", "not a and b or c",
        "1 < x <= 3", "x |> f(...) ||> g", "lambda x, y=1: x + y",
        "{1: 2}", "{1, 2}", "(1,)", "-x ** 2", "x.y.z",
        "0x10 + 1_000 + 1.5e3", "b'x'", "r'\\n'", "(x := 5)",
        "def f(a: int, *args, b=2, **kw) -> int:\n  return a\n",
        "class A[T](B):\n  x: int = 1\n  y = 2\n",
        "for i in range(3):\n  print(i)\nelse:\n  pass\n",
        "try:\n  pass\nexcept E:\n  raise\nfinally:\n  pass\n",
        "with a as b, c:\n  pass\n", "import a.b as c", "from .x import y as z, w",
        "match x:\n  case 1:\n    pass\n  case [a, b] if a:\n    pass\n",
        "@deco(1)\ndef h():\n  yield 1\n", "while x:\n  break\n",
    ];
    // Excluded (differences come from C++ ScopingVisitor, not the parser): augmented
    // assignment to an unbound name, comprehension variable renaming, string merging.
    let (mut same, mut diff) = (0, 0);
    for c in cases {
        match (cpp(c), ours(c)) {
            (Ok(a), Ok(b)) if a.trim() == b.trim() => same += 1,
            (a, b) => {
                diff += 1;
                eprintln!("DIFF {c:?}\n  cpp:  {a:?}\n  rust: {b:?}");
            }
        }
    }
    eprintln!("oracle: same={same} diff={diff}");
    assert_eq!(diff, 0);
}
