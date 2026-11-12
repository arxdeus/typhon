//! Port of `test/cir/parse_code.cpp` (`ParseCodeTest`).
//!
//! `ImportsInitializeInCurrentJITCell` needs the incremental JIT (out of scope).
//! PORT-NOTE: `ImportsFromIRPassRunBeforeUserCode` ran the module in-process with
//! `options->capture`. Here the module is linked into an executable (same pipeline as
//! `typhon build -exe`) and its stdout is compared, avoiding a second copy of
//! libtyphonrt inside the test process.

use std::collections::HashMap;

use typhon_common::Options;
use typhon_compiler::Compiler;
use typhon_compiler::typhon_llvm::inkwell::context::Context;
use typhon_ir::pass::{Pass, PassContext};
use typhon_ir::Module;

fn main_body_len(m: &Module) -> usize {
    let body = m.func_body(m.main_func()).expect("main body");
    m.as_series_flow(body).map(|s| s.series.len()).unwrap_or(0)
}

#[test]
fn definitions_and_new_import() {
    let mut c = Compiler::new(Options::with_argv0("build/typhon_test"), None);
    c.parse_code("parse_code_test.codon", "print(42)\n", 0, 0, &HashMap::new()).expect("parse");
    let m = c.module().clone();
    let before = main_body_len(&m.borrow());
    let ctx = typhon_parser::visitors::translate::ctx::existing_codegen_ctx(&c.cache().borrow()).expect("codegen ctx");
    let (series, bases) = {
        let x = ctx.borrow();
        (x.series.clone(), x.bases.clone())
    };

    // Definitions alone must not insert executable code into main.
    m.borrow_mut().parse_code("def __pc_plain(x: int) -> int:\n    return x + 1\n").expect("parse plain");
    {
        let mut mm = m.borrow_mut();
        let i = mm.get_int_type();
        assert!(mm.get_or_realize_func("__pc_plain", &[i], &[], "").is_some());
    }
    assert_eq!(before, main_body_len(&m.borrow()));

    // A first import emits executable initialization, even for definitions-only modules.
    m.borrow_mut()
        .parse_code("import bisect\ndef __pc_bisect(x: int) -> int:\n    return bisect.bisect_left([1, 5, 9], x)\n")
        .expect("parse bisect");
    {
        let mut mm = m.borrow_mut();
        let i = mm.get_int_type();
        assert!(mm.get_or_realize_func("__pc_bisect", &[i], &[], "").is_some());
    }
    assert!(main_body_len(&m.borrow()) > before);
    let x = ctx.borrow();
    assert_eq!(x.series, series);
    assert_eq!(x.bases, bases);
}

/// C++ `ParseImportPass`.
struct ParseImportPass;

impl Pass for ParseImportPass {
    fn key(&self) -> String {
        "test-parse-import".into()
    }
    fn run(&mut self, m: &mut Module, _ctx: &dyn PassContext) {
        m.parse_code(
            r#"
import string
class ParsedBase:
    def value(self) -> int: return 0
class ParsedChild(ParsedBase):
    def value(self) -> int: return len(string.ascii_letters)
def __pc_dispatch(value: ParsedBase) -> int:
    return value.value()
def __pc_import(x: int) -> int:
    return len(string.ascii_letters) + x
print(__pc_dispatch(ParsedChild()))
print(string.ascii_lowercase)
"#,
        )
        .expect("first parse");
        let i = m.get_int_type();
        assert!(m.get_or_realize_func("__pc_import", &[i], &[], "").is_some());
        m.parse_code("import string\nprint(__pc_import(7))\n").expect("second parse");
    }
}

#[test]
fn imports_from_ir_pass_run_before_user_code() {
    let dir = tempfile::tempdir().expect("tempdir");
    for debug in [true, false] {
        let mut o = Options::with_argv0("build/typhon_test");
        o.debug = debug;
        o.standalone = true;
        let mut c = Compiler::new(o, None);
        c.parse_code("parse_code_test.codon", "print(42)\n", 0, 0, &HashMap::new()).expect("parse");
        c.pass_manager().register_pass(Box::new(ParseImportPass), "core-pipeline-lowering", vec![], vec![]);
        c.run_passes();
        let ctx = Context::create();
        let mut v = c.codegen(&ctx).expect("codegen");
        let exe = dir.path().join(format!("parse_code_{debug}"));
        let exe_s = exe.to_string_lossy().into_owned();
        // argv0 = this test binary: libtyphonrt is found next to it (target/<profile>/deps/..).
        let argv0 = std::env::current_exe().unwrap().parent().unwrap().parent().unwrap().join("typhon");
        v.write_to_executable(&exe_s, &argv0.to_string_lossy(), false, &[], "").expect("link");
        let out = std::process::Command::new(&exe).output().expect("run");
        assert!(out.status.success(), "debug={debug}: {}", String::from_utf8_lossy(&out.stderr));
        assert_eq!(String::from_utf8_lossy(&out.stdout), "52\nabcdefghijklmnopqrstuvwxyz\n59\n42\n", "debug={debug}");
    }
}
