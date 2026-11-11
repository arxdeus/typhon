//! Compiler-level ports of `test/types.cpp` (`ContextManagerLoweringTest`) and
//! `test/cir/module.cpp` (`CIRModuleTest.CanonicalizesCheckedOptionalTypes`).
//! `JITOptionsTest`/`JITRTTITest` exercise the incremental JIT, which is out of scope.

use std::collections::HashMap;

use typhon_common::Options;
use typhon_compiler::Compiler;
use typhon_ir::{NodeRef, TypeKind};

fn compiler() -> Compiler {
    Compiler::new(Options::with_argv0("build/codon_test"), None)
}

#[test]
fn cir_module_canonicalizes_checked_optional_types() {
    let mut c = compiler();
    c.parse_code("optional_types.codon", "pass\n", 0, 0, &HashMap::new()).expect("parse");
    let m = c.module().clone();
    let mut m = m.borrow_mut();
    let none = m.get_none_type();
    assert_eq!(m.get_optional_type(none), none);
    assert!(m.ty(none).kind.is_record());
    let integer = m.get_int_type();
    let optional = m.get_optional_type(integer);
    match m.ty(optional).kind {
        TypeKind::Optional { base } => assert_eq!(base, integer),
        ref k => panic!("expected optional type, got {k:?}"),
    }
    assert_eq!(m.get_optional_type(optional), optional);
}

#[test]
fn context_manager_lowering_omits_handlers_only_for_legacy_none_exits() {
    let cases: [(&str, &str, bool, bool); 7] = [
        ("context_legacy", "LegacyNone", false, true),
        ("context_bool", "LegacyBool", false, false),
        ("context_aware", "AwareNone", false, false),
        ("context_defaults", "DefaultNone", false, false),
        ("context_overloaded", "OverloadedNone", false, false),
        ("context_async_legacy", "AsyncNone", true, true),
        ("context_async_bool", "AsyncBool", true, false),
    ];
    let mut source = String::from(
        r#"
class LegacyNone:
  def __enter__(self):
    return 7
  def __exit__(self):
    pass

class LegacyBool:
  def __enter__(self):
    return 7
  def __exit__(self):
    return True

class AwareNone:
  def __enter__(self):
    return 7
  def __exit__(self, exc_type, exc_value, traceback):
    pass

class DefaultNone:
  def __enter__(self):
    return 7
  def __exit__(self, exc_type=None, exc_value=None, traceback=None):
    pass

class OverloadedNone:
  def __enter__(self):
    return 7
  def __exit__(self):
    pass
  def __exit__(self, exc_type, exc_value, traceback):
    pass

class AsyncNone:
  async def __aenter__(self):
    return 7
  async def __aexit__(self):
    pass

class AsyncBool:
  async def __aenter__(self):
    return 7
  async def __aexit__(self):
    return True
"#,
    );
    for (name, manager, is_async, _) in &cases {
        source += &format!(
            "{}{name}():\n    {}{manager}() as value:\n        return value\n{name}()\n",
            if *is_async { "async def " } else { "def " },
            if *is_async { "async with " } else { "with " }
        );
    }
    let mut c = compiler();
    c.parse_code("context_manager_lowering.codon", &source, 0, 0, &HashMap::new()).expect("parse");
    let m = c.module().borrow();
    let mut checked = 0;
    for var in m.vars() {
        let Some(fd) = m.func(var) else { continue };
        if !fd.is_bodied() {
            continue;
        }
        let Some(expected) = cases.iter().find(|c| c.0 == fd.unmangled_name) else { continue };
        checked += 1;
        let (mut tries, mut catches, mut flags) = (0, 0, 0);
        let mut pending: Vec<_> = m.func_body(var).into_iter().collect();
        while let Some(v) = pending.pop() {
            if let Some(tc) = m.as_try_catch_flow(v) {
                tries += 1;
                catches += tc.catches.len();
                assert!(tc.finally.is_some(), "{}: try without finally", expected.0);
            }
            pending.extend(m.used_values(NodeRef::Value(v)));
        }
        for local in fd.symbols() {
            if m.var_name(*local).contains("with_exception") {
                flags += 1;
            }
        }
        let legacy = expected.3;
        assert_eq!(tries, 1, "{}", expected.0);
        assert_eq!(catches, if legacy { 0 } else { 1 }, "{}", expected.0);
        assert_eq!(flags, if legacy { 0 } else { 1 }, "{}", expected.0);
    }
    assert_eq!(checked, cases.len());
}
