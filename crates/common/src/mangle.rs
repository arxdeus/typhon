//! Name mangling (`mangled_class/func/method/var`) and the `StdlibTypes` table.
//!
//! Placed in `typhon-common` because IR passes (side effects, numpy,
//! lowering) use these names and must not
//! depend on the parser crate.

use once_cell::sync::Lazy;

fn is_core(module: &str) -> bool {
    module.is_empty() || module == "std.internal.core"
}

pub fn mangled_class(module: &str, cls: &str, id: usize) -> String {
    if is_core(module) {
        return cls.to_string();
    }
    let num = if cls.contains('.') { String::new() } else { format!(".{id}") };
    format!("{module}.{cls}{num}")
}

pub fn mangled_func(module: &str, func: &str, overload: usize, id: usize, no_core: bool) -> String {
    if !no_core && is_core(module) {
        return format!("{func}:{overload}");
    }
    let num = if func.contains('.') { String::new() } else { format!(".{id}") };
    let prefix = if module.is_empty() { String::new() } else { format!("{module}.") };
    format!("{prefix}{func}{num}:{overload}")
}

pub fn mangled_method(module: &str, cls: &str, method: &str, overload: usize, id: usize) -> String {
    if is_core(module) {
        return format!("{cls}.{method}:{overload}");
    }
    let num = if cls.contains('.') { String::new() } else { format!(".{id}") };
    format!("{module}.{cls}{num}.{method}:{overload}")
}

pub fn mangled_var(module: &str, var: &str, id: usize) -> String {
    let num = if var.contains('.') { String::new() } else { format!(".{id}") };
    let prefix = if module.is_empty() { String::new() } else { format!("{module}.") };
    format!("{prefix}{var}{num}")
}

/// Shorthands with the reference default arguments (`overload = 0`, `id = 0`, `noCore = false`).
pub fn class(module: &str, cls: &str) -> String {
    mangled_class(module, cls, 0)
}
pub fn func(module: &str, f: &str) -> String {
    mangled_func(module, f, 0, 0, false)
}
pub fn var(module: &str, v: &str) -> String {
    mangled_var(module, v, 0)
}

pub struct StdlibTypes {
    pub any: String,
    pub array: String,
    pub base_exception: String,
    pub bool_: String,
    pub callable: String,
    pub capsule: String,
    pub cobj: String,
    pub complex: String,
    pub complex64: String,
    pub coroutine: String,
    pub dict: String,
    pub float: String,
    pub float16: String,
    pub function: String,
    pub generator: String,
    pub int: String,
    pub list: String,
    pub named_tuple: String,
    pub ndarray: String,
    pub none_type: String,
    pub object: String,
    pub optional: String,
    pub ptr: String,
    pub range: String,
    pub set: String,
    pub slice: String,
    pub string: String,
    pub thread_local: String,
    pub tuple: String,
    pub type_: String,
    pub type_wrap: String,
    pub uint: String,
    pub union: String,
    pub unrealized_type: String,
    pub vec: String,
    pub callable_trait: String,
    pub type_trait: String,
}

pub static STDLIB_TYPES: Lazy<StdlibTypes> = Lazy::new(|| StdlibTypes {
    any: class("", "Any"),
    array: class("", "Array"),
    base_exception: class("std.internal.types.error", "BaseException"),
    bool_: class("", "bool"),
    callable: class("", "Callable"),
    capsule: class("", "Capsule"),
    cobj: class("", "cobj"),
    complex: class("std.internal.types.complex", "complex"),
    complex64: class("std.internal.types.complex", "complex64"),
    coroutine: class("", "Coroutine"),
    dict: var("", "Dict"),
    float: class("", "float"),
    float16: class("", "float16"),
    function: class("", "Function"),
    generator: class("", "Generator"),
    int: class("", "Int"),
    list: class("std.internal.types.collections.list", "List"),
    named_tuple: class("", "NamedTuple"),
    ndarray: class("std.numpy.ndarray", "ndarray"),
    none_type: class("", "NoneType"),
    object: class("", "object"),
    optional: class("", "Optional"),
    ptr: class("", "Ptr"),
    range: class("std.internal.types.range", "range"),
    set: class("std.internal.types.collections.set", "Set"),
    slice: class("std.internal.types.slice", "Slice"),
    string: class("", "str"),
    thread_local: class("std.threading", "ThreadLocal"),
    tuple: class("", "Tuple"),
    type_: class("", "type"),
    type_wrap: class("", "TypeWrap"),
    uint: class("", "UInt"),
    union: class("", "Union"),
    unrealized_type: class("", "unrealized_type"),
    vec: class("std.simd", "Vec"),
    callable_trait: "CallableTrait".into(),
    type_trait: "TypeTrait".into(),
});

pub fn stdlib() -> &'static StdlibTypes {
    &STDLIB_TYPES
}

pub static VAR_ARGV: Lazy<String> = Lazy::new(|| var("", "__argv__"));
pub static VAR_ARGC: Lazy<String> = Lazy::new(|| var("", "__argc__"));

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mangling_scheme() {
        assert_eq!(class("", "int"), "int");
        assert_eq!(class("std.internal.types.range", "range"), "std.internal.types.range.range.0");
        assert_eq!(func("std.internal.builtin", "len"), "std.internal.builtin.len.0:0");
        assert_eq!(func("", "pure"), "pure:0");
        assert_eq!(var("", "Dict"), "Dict.0");
        assert_eq!(mangled_method("m", "C", "f", 1, 2), "m.C.2.f:1");
    }
}
