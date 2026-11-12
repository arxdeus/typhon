//! Ports of `FieldInitializationTest` / `EnumConstructionTest` from `test/types.cpp`.
//!
//! PORT-NOTE: the C++ tests capture stderr (gtest `CaptureStderr`) to assert that no
//! "may be uninitialized" warning was printed. Warnings go straight to the process
//! stderr via `typhon_common::diag`, so those negative checks are reduced to
//! "compiles without error".

use std::collections::HashMap;

use typhon_common::Options;
use typhon_compiler::Compiler;

fn compile(src: &str, file: &str, f: impl FnOnce(&mut Options)) -> (Compiler, Result<(), String>) {
    let mut o = Options::with_argv0("build/typhon_test");
    f(&mut o);
    let mut c = Compiler::new(o, None);
    let r = c.parse_code(file, src, 0, 0, &HashMap::new()).map_err(|e| e.to_string());
    (c, r)
}

fn expect_err(r: &Result<(), String>, needle: &str, ctx: &str) {
    match r {
        Ok(()) => panic!("{ctx}: expected error containing {needle:?}"),
        Err(e) => assert!(e.contains(needle), "{ctx}: error {e:?} does not contain {needle:?}"),
    }
}

fn expect_ok(r: &Result<(), String>, ctx: &str) {
    if let Err(e) = r {
        panic!("{ctx}: unexpected error: {e}");
    }
}

#[test]
fn rejects_uninitialized_fields() {
    for (name, body) in [
        ("Missing", "    pass\n"),
        ("Conditional", "    if flag:\n      self.values = [1]\n"),
        ("EarlyReturn", "    if flag:\n      return\n    self.values = [1]\n"),
    ] {
        for debug in [false, true] {
            let src = format!("class {name}:\n  values: list[int]\n  def __init__(self, flag: bool):\n{body}{name}(False)\n");
            let (_, r) = compile(&src, "field_initialization.py", |o| o.debug = debug);
            expect_err(&r, &format!("field '{name}.values' may be uninitialized when __init__ returns"), name);
        }
    }
}

#[test]
fn tracks_helpers_and_supports_opt_out() {
    let src = r#"
def initialize_values(Element: type, instance):
  target = instance
  target.values = [Element(1)]

class ThroughHelpers:
  values: list[int]
  def __init__(self):
    self.initialize()
  def initialize(self):
    initialize_values(int, self)

class Parent:
  value: int
  def __init__(self):
    self.value = 1

class Inherits(Parent):
  def __init__(self):
    super().__init__()

@dataclass(init=False)
class Unchecked:
  values: list[int]
  def __new__():
    return type._ref_new(Unchecked)

@tuple
class Value:
  number: int
  def __init__(self):
    pass

ThroughHelpers()
Inherits()
Unchecked()
Value(1).__init__()
"#;
    let (_, r) = compile(src, "field_initialization.py", |_| {});
    expect_ok(&r, "helpers");
}

#[test]
fn checks_cooperative_super_initialization() {
    for cooperative in [false, true] {
        for debug in [false, true] {
            let mut src = String::from(
                "\nclass Base:\n  value: int\n  def __init__(self, value: int):\n    self.value = value\n\
                 class Left(Base):\n  left: int\n  def __init__(self, value: int):\n",
            );
            src += if cooperative { "    super().__init__(value)\n" } else { "    Base.__init__(self, value)\n" };
            src += r#"
    self.left = value + 1
class Right(Base):
  right: int
  def __init__(self, value: int):
    super().__init__(value)
    self.right = value + 2
class Both(Left, Right):
  own: int
  def __init__(self, value: int):
    super().__init__(value)
    self.own = value + 3
instance = Both(10)
assert instance.value == 10
assert instance.left == 11
assert instance.right == 12
assert instance.own == 13
"#;
            let (_, r) = compile(&src, "field_initialization_super.py", |o| o.debug = debug);
            if cooperative {
                expect_ok(&r, "cooperative");
            } else {
                expect_err(&r, "field 'Both.right' may be uninitialized", "skipped right base");
            }
        }
    }
}

#[test]
fn tracks_virtual_initialization() {
    for base_writes in [false, true] {
        for child_writes in [false, true] {
            for debug in [false, true] {
                let mut src = String::from("\nclass Base:\n  value: int\n  def __init__(self):\n    initialize_base(self)\n  def initialize(self):\n");
                src += if base_writes { "    self.value = 1\n" } else { "    pass\n" };
                src += "class Child(Base):\n  def initialize(self):\n";
                src += if child_writes { "    self.value = 42\n" } else { "    pass\n" };
                src += "\ndef initialize_base(instance: Base):\n  instance.initialize()\nChild()\n";
                let ctx = format!("base_writes={base_writes} child_writes={child_writes} debug={debug}");
                let (c, r) = compile(&src, "field_initialization_virtual.py", |o| o.debug = debug);
                if child_writes {
                    expect_ok(&r, &ctx);
                    if !base_writes {
                        let m = c.module().clone();
                        let mut m = m.borrow_mut();
                        let base = m.get_or_realize_type("Base.0", &[]).expect("Base.0");
                        assert!(m.get_or_realize_method(base, "__init__", &[base], &[]).is_none(), "{ctx}");
                    }
                } else {
                    expect_err(&r, ".value' may be uninitialized", &ctx);
                }
            }
        }
    }
}

#[test]
fn rejects_deferred_writes() {
    for asynchronous in [false, true] {
        for constructor in [false, true] {
            for debug in [false, true] {
                let mut src = String::from("class Record:\n  value: int\n");
                if !constructor {
                    src += "  def __init__(self):\n    self.initialize()\n";
                }
                src += if asynchronous { "  async def " } else { "  def " };
                src += if constructor { "__init__" } else { "initialize" };
                src += "(self):\n    self.value = 42\n";
                if !asynchronous {
                    src += "    yield 1\n";
                }
                src += "Record()\n";
                let (_, r) = compile(&src, "field_initialization_deferred.py", |o| o.debug = debug);
                expect_err(&r, "field 'Record.value' may be uninitialized", &format!("async={asynchronous} ctor={constructor}"));
            }
        }
    }
}

#[test]
fn rejects_rebound_aliases() {
    let bodies = [
        "      for unused in range(1):\n        target = other\n      target.value = 42\n",
        "      active = True\n      while active:\n        target = other\n        active = False\n      target.value = 42\n",
        "      for index in range(2):\n        if index == 1:\n          target.value = 42\n          return\n        target = other\n      self.value = 99\n",
        "      for target in [other]:\n        target.value = 42\n        return\n      self.value = 99\n",
        "      try:\n        target = other\n        raise ValueError()\n      except ValueError:\n        target.value = 42\n",
        "      try:\n        fail(target := other)\n      except ValueError:\n        target.value = 42\n",
        "      try:\n        for target in instances(other):\n          pass\n      except ValueError:\n        target.value = 42\n",
    ];
    let head = "\ndef fail(instance):\n  raise ValueError()\ndef instances(instance):\n  yield instance\n  raise ValueError()\nclass Record:\n  value: int\n  def __init__(self, other=None):\n    if other is None:\n      self.value = 1\n    else:\n      target = self\n";
    for body in bodies {
        for debug in [false, true] {
            let src = format!("{head}{body}Record(Record())\n");
            let (_, r) = compile(&src, "field_initialization_alias.py", |o| o.debug = debug);
            expect_err(&r, "field 'Record.value' may be uninitialized", body);
        }
    }
}

#[test]
fn checks_finally_exit_overrides() {
    let bodies = [
        ("    try:\n      raise ValueError()\n    finally:\n      return\n", false),
        ("    try:\n      fail()\n    finally:\n      return\n", false),
        ("    for unused in range(1):\n      try:\n        break\n      finally:\n        return\n    self.value = 42\n", false),
        ("    for unused in range(1):\n      try:\n        continue\n      finally:\n        return\n    self.value = 42\n", false),
        ("    try:\n      raise ValueError()\n    finally:\n      self.value = 42\n      return\n", true),
        ("    try:\n      return\n    finally:\n      self.value = 42\n", true),
        ("    try:\n      raise ValueError()\n    except:\n      self.value = 42\n    finally:\n      return\n", true),
        ("    try:\n      raise ValueError()\n    except BaseException:\n      self.value = 42\n    finally:\n      return\n", true),
    ];
    for (body, valid) in bodies {
        for debug in [false, true] {
            let src = format!("def fail():\n  raise ValueError()\nclass Record:\n  value: int\n  def __init__(self):\n{body}Record()\n");
            let (_, r) = compile(&src, "field_initialization_finally.py", |o| o.debug = debug);
            if valid {
                expect_ok(&r, body);
            } else {
                expect_err(&r, "field 'Record.value' may be uninitialized", body);
            }
        }
    }
}

#[test]
fn checks_late_realizations() {
    for initialized in [false, true] {
        let mut src = String::from("\nclass Late:\n  value: int\n  def __init__(self):\n");
        src += if initialized { "    self.value = 42\n" } else { "    pass\n" };
        src += "\ndef make() -> int:\n  return Late().value\n";
        let ctx = format!("initialized={initialized}");
        let (c, r) = compile(&src, "field_initialization_late.py", |_| {});
        expect_ok(&r, &ctx);
        let m = c.module().clone();
        let got = m.borrow_mut().get_or_realize_func("make", &[], &[], "").is_some();
        assert_eq!(got, initialized, "{ctx}");
    }
}

#[test]
fn reuses_shared_helper_summaries() {
    let mut src = String::from("def initialize_0(instance, seed: int):\n    instance.value = seed\n");
    for level in 1..=24 {
        let prev = format!("initialize_{}", level - 1);
        src += &format!(
            "def initialize_{level}(instance, seed: int):\n    if seed % 2 == 0:\n        {prev}(instance, seed + 1)\n    else:\n        {prev}(instance, seed + 2)\n"
        );
    }
    src += r#"
class Record:
    value: int
    def __init__(self, seed: int):
        initialize_24(self, seed)

def initialize_pair(first, second):
    first.first = 1
    second.second = 2

class Pair:
    first: int
    second: int
    def __init__(self):
        initialize_pair(self, self)

assert Record(1).value == 49
assert Pair().first == 1 and Pair().second == 2
"#;
    let (_, r) = compile(&src, "field_initialization_shared.py", |_| {});
    expect_ok(&r, "shared");
}

#[test]
fn rejects_tuple_init_option() {
    let (_, r) = compile("\n@tuple(check_init=False)\nclass Value:\n  number: int\n", "tuple_init_option.py", |_| {});
    expect_err(&r, "class decorator got unexpected argument", "tuple option");
}

#[test]
fn enum_bounds_helper_realizations() {
    let mut src = String::from("from enum import Enum, auto\nclass Explicit(Enum):\n");
    for i in 0..32 {
        src += &format!("    MEMBER_{i} = {i}\n");
    }
    src += "class Automatic(Enum):\n";
    for i in 0..32 {
        src += &format!("    MEMBER_{i} = auto()\n");
    }
    src += "assert Explicit(31) is Explicit.MEMBER_31\nassert Automatic(32) is Automatic.MEMBER_31\n";
    for unordered in [false, true] {
        let (c, r) = compile(&src, "enum_construction.py", |o| {
            o.debug = false;
            o.unordereddict = unordered;
        });
        expect_ok(&r, "enum");
        let cache = c.cache().borrow();
        let count = |helper: &str| -> usize {
            let prefix = typhon_common::mangle::mangled_func("std.enum", helper, 0, 0, false);
            cache.functions.iter().filter(|(n, _)| n.starts_with(&prefix)).map(|(_, f)| f.realizations.len()).sum()
        };
        assert_eq!(count("_make_member"), 2, "unordered={unordered}");
        assert_eq!(count("_next_value"), 1, "unordered={unordered}");
        assert_eq!(count("_auto_maximum"), 1, "unordered={unordered}");
    }
}

#[test]
fn enum_skips_lookup_initialization() {
    let src = r#"
from enum import Enum
class Color(Enum):
    RED = 1
    ALIAS = 1
assert Color(1) is Color.RED
assert Color(Color.RED) is Color.RED
assert Color.ALIAS is Color.RED

class ExplicitEmpty:
    @staticmethod
    def __new__(value: int):
        return type._ref_new(ExplicitEmpty)
    def __init__(self, value: int):
        pass
ExplicitEmpty(1)
"#;
    let (c, r) = compile(src, "enum_lookup_initialization.py", |_| {});
    expect_ok(&r, "enum lookup");
    let cache = c.cache().borrow();
    let enum_init = format!("{}.__init__:", typhon_common::mangle::mangled_class("std.enum", "Enum", 0));
    let mut realized_explicit_empty = false;
    for (name, f) in &cache.functions {
        assert!(!name.starts_with(&enum_init), "{name}");
        let auto_gen = f.ast.is_some_and(|a| cache.ast[a].attrs.has(typhon_parser::ast::attr::Attr::AutoGenerated));
        if name.contains("ExplicitEmpty") && name.contains(".__init__:") && !auto_gen {
            realized_explicit_empty |= !f.realizations.is_empty();
        }
    }
    assert!(realized_explicit_empty);
}
