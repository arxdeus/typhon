//! Port of `codon/cir/util/irtools.{h,cpp}`.

use once_cell::sync::Lazy;
use typhon_common::mangle::func as mangled;

use crate::attribute::{Attribute, KeyValueAttribute, attr_id};
use crate::ids::{FuncId, NodeRef, TypeId, ValueId, VarId};
use crate::module::{Module, magic};
use crate::types::TypeKind;
use crate::value::ValueKind;
use crate::var::{FuncKind, Intrinsic};

static EXPORT_ATTR: Lazy<String> = Lazy::new(|| mangled("std.internal.attributes", "export"));
static GPU_KERNEL_ATTR: Lazy<String> = Lazy::new(|| mangled("std.internal.gpu", "kernel"));
static INLINE_ATTR: Lazy<String> = Lazy::new(|| mangled("std.internal.attributes", "inline"));
static NOINLINE_ATTR: Lazy<String> = Lazy::new(|| mangled("std.internal.attributes", "noinline"));

/// `hasAttribute(func, attribute)`.
pub fn has_attribute(m: &Module, f: FuncId, attribute: &str) -> bool {
    m.get_attribute(f.into(), attr_id::KEY_VALUE).and_then(|a| a.as_key_value()).map(|kv| kv.has(attribute)).unwrap_or(false)
}
pub fn is_export(m: &Module, f: FuncId) -> bool {
    has_attribute(m, f, &EXPORT_ATTR)
}
/// `setExport(func, value)`.
pub fn set_export(m: &mut Module, f: FuncId, value: bool) {
    let n: NodeRef = f.into();
    if !m.has_attribute(n, attr_id::KEY_VALUE) {
        if !value {
            return;
        }
        m.set_attribute(n, Attribute::KeyValue(KeyValueAttribute::default()));
    }
    let kv = m.get_attribute_mut(n, attr_id::KEY_VALUE).and_then(|a| a.as_key_value_mut()).expect("key-value attribute");
    if value {
        kv.attributes.insert(EXPORT_ATTR.clone(), String::new());
    } else {
        kv.attributes.remove(EXPORT_ATTR.as_str());
    }
}
pub fn is_gpu_kernel(m: &Module, f: FuncId) -> bool {
    has_attribute(m, f, &GPU_KERNEL_ATTR)
}
pub fn is_inline(m: &Module, f: FuncId) -> bool {
    has_attribute(m, f, &INLINE_ATTR)
}
pub fn is_no_inline(m: &Module, f: FuncId) -> bool {
    has_attribute(m, f, &NOINLINE_ATTR)
}

/// `getDerivedFuncName(name, parent, site)`.
pub fn derived_func_name(m: &Module, name: &str, parent: FuncId, site: Option<NodeRef>) -> String {
    let owner = m.var_name(parent);
    let mut result = format!("{}.context.{}:{}", name, owner.len(), owner);
    if let Some(site) = site {
        let s = m.src_info(site);
        result += &format!(".{}:{}", s.line, s.col);
    }
    result
}

/// `isStdlibFunc(func, submodule)`.
pub fn is_stdlib_func(m: &Module, f: FuncId, submodule: &str) -> bool {
    match m.get_attribute(f.into(), attr_id::KEY_VALUE).and_then(|a| a.as_key_value()) {
        Some(kv) => kv.get(".module").starts_with(&format!("std::{submodule}")),
        None => false,
    }
}

/// `isInt(type, len, sign)`.
pub fn is_int(m: &Module, t: TypeId, len: u32, sign: bool) -> bool {
    matches!(m.type_kind(t), TypeKind::Int { len: l, sign: s } if *l == len && *s == sign)
}

/// `call(func, args)`.
pub fn call(m: &mut Module, f: FuncId, args: Vec<ValueId>) -> ValueId {
    let fv = m.nr_var_value(f);
    m.nr_call(fv, args)
}

/// `virtualCall(callee, args, thunkId, parent)`.
pub fn virtual_call(m: &mut Module, callee: ValueId, mut args: Vec<ValueId>, thunk_id: usize, parent: FuncId) -> ValueId {
    assert!(thunk_id != 0 && !args.is_empty(), "virtual call requires a parent, method ID and receiver");
    let receiver = args[0];
    let rt = m.value_type_mut(receiver);
    let ct = m.value_type_mut(callee);
    let temporary = m.n_var(Default::default(), Some(rt), false, false, false, "rtti.self");
    m.func_push_symbol(parent, temporary);
    let int_t = m.get_int_type();
    let sig_name = format!("rtti.lookup[{}].{}", m.type_name(rt), m.type_name(ct));
    let signature = m.unsafe_get_func_type(&sig_name, ct, vec![rt, int_t], false);
    let lookup = m.get_intrinsic(Intrinsic::VirtualLookup, Some(rt), Some(signature));
    let prefix = m.nr_series_flow();
    m.series_push(prefix, callee);
    let assign = m.nr_assign(temporary, receiver);
    m.series_push(prefix, assign);
    let tv = m.nr_var_value(temporary);
    let tid = m.get_int(thunk_id as i64);
    let lookup_call = call(m, lookup, vec![tv, tid]);
    let target = m.nr_flow_instr(prefix, lookup_call);
    args[0] = m.nr_var_value(temporary);
    let src = m.src_info(callee.into());
    m.n(src, ValueKind::CallInstr(crate::value::CallInstr { callee: target, args }))
}

fn func_of_call(m: &Module, v: ValueId) -> Option<(FuncId, Vec<ValueId>)> {
    let call = m.as_call_instr(v)?;
    let f = get_func(m, call.callee)?;
    Some((f, call.args.clone()))
}

/// `isCallOf(value, name, inputs, output, method)`.
pub fn is_call_of_types(
    m: &mut Module,
    value: ValueId,
    name: &str,
    inputs: &[Option<TypeId>],
    output: Option<TypeId>,
    method: bool,
) -> bool {
    let Some((f, args)) = func_of_call(m, value) else { return false };
    let fd = m.func(f).unwrap();
    if fd.unmangled_name != name || args.len() != inputs.len() {
        return false;
    }
    let parent = fd.parent_type;
    for (arg, input) in args.iter().zip(inputs) {
        if let Some(i) = input {
            let at = m.value_type_mut(*arg);
            if !m.type_is(at, *i) {
                return false;
            }
        }
    }
    if let Some(o) = output {
        let vt = m.value_type_mut(value);
        if !m.type_is(vt, o) {
            return false;
        }
    }
    if method {
        let Some(p) = parent else { return false };
        if inputs.is_empty() {
            return false;
        }
        if let Some(i) = inputs[0] {
            if !m.type_is(p, i) {
                return false;
            }
        }
    }
    true
}

/// `isCallOf(value, name, numArgs, output, method)` (`num_args < 0`: no check).
pub fn is_call_of(m: &mut Module, value: ValueId, name: &str, num_args: i32, output: Option<TypeId>, method: bool) -> bool {
    let Some((f, args)) = func_of_call(m, value) else { return false };
    let fd = m.func(f).unwrap();
    if fd.unmangled_name != name || (num_args >= 0 && args.len() != num_args as usize) {
        return false;
    }
    let parent = fd.parent_type;
    if let Some(o) = output {
        let vt = m.value_type_mut(value);
        if !m.type_is(vt, o) {
            return false;
        }
    }
    if method {
        let Some(p) = parent else { return false };
        if args.is_empty() {
            return false;
        }
        let at = m.value_type_mut(args[0]);
        if !m.type_is(at, p) {
            return false;
        }
    }
    true
}

/// `isMagicMethodCall(value)`.
pub fn is_magic_method_call(m: &mut Module, value: ValueId) -> bool {
    let Some((f, args)) = func_of_call(m, value) else { return false };
    let fd = m.func(f).unwrap();
    let Some(p) = fd.parent_type else { return false };
    let name = fd.unmangled_name.clone();
    if args.is_empty() {
        return false;
    }
    let at = m.value_type_mut(args[0]);
    if !m.type_is(at, p) {
        return false;
    }
    name.len() >= 5 && name.starts_with("__") && name.ends_with("__")
}

/// `makeTuple(args, M)`.
pub fn make_tuple(m: &mut Module, args: &[ValueId]) -> Option<ValueId> {
    let types: Vec<TypeId> = args.iter().map(|a| m.value_type_mut(*a)).collect();
    let tuple_type = m.get_tuple_type(types.clone());
    let new_func = m.get_or_realize_method(tuple_type, magic::NEW, &types, &[]);
    let Some(new_func) = new_func else {
        typhon_common::log_flag!(IR, "could not realize {} new function", m.type_name(tuple_type));
        return None;
    };
    Some(call(m, new_func, args.to_vec()))
}

thread_local! {
    static ANON_COUNTER: std::cell::Cell<u64> = const { std::cell::Cell::new(1) };
}

/// `makeVar(x, flow, parent, prepend)`.
pub fn make_var(m: &mut Module, x: ValueId, flow: ValueId, parent: Option<FuncId>, prepend: bool) -> VarId {
    let global = parent.is_none();
    let t = m.value_type_mut(x);
    let v = m.nr_var(t, global, "");
    if global {
        let c = ANON_COUNTER.with(|c| {
            let v = c.get();
            c.set(v + 1);
            v
        });
        m.set_name(v.into(), format!(".anon_global.{c}"));
    }
    let a = m.nr_assign(v, x);
    if prepend {
        m.series_insert(flow, 0, a);
    } else {
        m.series_push(flow, a);
    }
    if let Some(p) = parent {
        m.func_push_symbol(p, v);
    }
    v
}

/// `alloc(type, count)`.
pub fn alloc(m: &mut Module, ty: TypeId, count: ValueId) -> Option<ValueId> {
    let ptr = m.get_pointer_type(Some(ty));
    m.construct(ptr, vec![count])
}
/// `alloc(type, int64_t count)`.
pub fn alloc_n(m: &mut Module, ty: TypeId, count: i64) -> Option<ValueId> {
    let c = m.get_int(count);
    alloc(m, ty, c)
}

/// `series(args...)`.
pub fn series(m: &mut Module, vals: &[ValueId]) -> Option<ValueId> {
    if vals.is_empty() {
        return None;
    }
    let s = m.nr_series_flow();
    for v in vals {
        m.series_push(s, *v);
    }
    Some(s)
}

/// Typed constant extraction (`isConst<T>`, `getConst<T>`).
pub trait ConstValue: Sized + PartialEq + Clone {
    fn get(m: &Module, v: ValueId) -> Option<Self>;
}
impl ConstValue for i64 {
    fn get(m: &Module, v: ValueId) -> Option<Self> {
        m.as_int_const(v).map(|c| c.val)
    }
}
impl ConstValue for f64 {
    fn get(m: &Module, v: ValueId) -> Option<Self> {
        m.as_float_const(v).map(|c| c.val)
    }
}
impl ConstValue for bool {
    fn get(m: &Module, v: ValueId) -> Option<Self> {
        m.as_bool_const(v).map(|c| c.val)
    }
}
/// `TemplatedConst<std::string>` covers both string and bytes constants.
impl ConstValue for String {
    fn get(m: &Module, v: ValueId) -> Option<Self> {
        m.as_string_const(v).or_else(|| m.as_bytes_const(v)).map(|c| c.val.clone())
    }
}

/// `isConst<T>(x)`.
pub fn is_const<T: ConstValue>(m: &Module, v: ValueId) -> bool {
    T::get(m, v).is_some()
}
/// `isConst<T>(x, value)`.
pub fn is_const_val<T: ConstValue>(m: &Module, v: ValueId, value: &T) -> bool {
    T::get(m, v).as_ref() == Some(value)
}
/// `getConst<T>(x)`. Panics if not a constant of that type.
pub fn get_const<T: ConstValue>(m: &Module, v: ValueId) -> T {
    T::get(m, v).unwrap_or_else(|| panic!("{} is not a constant [{}]", m.reference_string(v.into()), m.src_info(v.into())))
}

/// `getVar(x)`: variable (non-function) referenced by a `VarValue`.
pub fn get_var(m: &Module, v: ValueId) -> Option<VarId> {
    let x = m.as_var_value(v)?;
    let var = m.actual_var(x.var);
    if m.is_func(var) { None } else { Some(var) }
}
/// `getFunc(x)`: function referenced by a `VarValue`.
pub fn get_func(m: &Module, v: ValueId) -> Option<FuncId> {
    let x = m.as_var_value(v)?;
    let var = m.actual_var(x.var);
    if m.is_func(var) { Some(var) } else { None }
}

/// `ptrLoad(ptr)`.
pub fn ptr_load(m: &mut Module, ptr: ValueId) -> ValueId {
    let zero = m.get_int(0);
    m.getitem(ptr, zero).unwrap_or_else(|| panic!("pointer getitem not found [{}]", m.src_info(ptr.into())))
}
/// `ptrStore(ptr, val)`.
pub fn ptr_store(m: &mut Module, ptr: ValueId, val: ValueId) -> ValueId {
    let pt = m.value_type_mut(ptr);
    let it = m.get_int_type();
    let vt = m.value_type_mut(val);
    let setitem = m
        .get_or_realize_method(pt, magic::SETITEM, &[pt, it, vt], &[])
        .unwrap_or_else(|| panic!("pointer setitem not found [{}]", m.src_info(ptr.into())));
    let zero = m.get_int(0);
    call(m, setitem, vec![ptr, zero, val])
}
/// `tupleGet(tuple, index)`.
pub fn tuple_get(m: &mut Module, tuple: ValueId, index: usize) -> ValueId {
    m.nr_extract(tuple, format!("item{}", index + 1))
}
/// `tupleStore(tuple, index, val)`.
pub fn tuple_store(m: &mut Module, tuple: ValueId, index: usize, val: ValueId) -> Option<ValueId> {
    let t = m.value_type_mut(tuple);
    assert!(m.type_kind(t).is_record(), "argument is not a tuple [{}]", m.src_info(tuple.into()));
    let n = m.type_fields(t).len();
    let mut elems = Vec::with_capacity(n);
    for i in 0..n {
        elems.push(if i == index { val } else { tuple_get(m, tuple, i) });
    }
    make_tuple(m, &elems)
}

/// `getStdlibFunc(x, name, submodule)`.
pub fn get_stdlib_func(m: &Module, x: ValueId, name: &str, submodule: &str) -> Option<FuncId> {
    let f = get_func(m, x)?;
    let fd = m.func(f)?;
    if matches!(fd.kind, FuncKind::Bodied { .. }) && is_stdlib_func(m, f, submodule) && fd.unmangled_name == name {
        Some(f)
    } else {
        None
    }
}

/// `getReturnType(func)`.
pub fn return_type(m: &Module, f: FuncId) -> Option<TypeId> {
    m.var_type(f).and_then(|t| m.func_type_ret(t))
}
/// `setReturnType(func, rType)`.
pub fn set_return_type(m: &mut Module, f: FuncId, ret: TypeId) {
    let t = m.var_type(f).expect("function has no type");
    assert!(m.type_kind(t).is_func(), "{} is not a function type [{}]", m.type_name(t), m.src_info(f.into()));
    let args = m.func_type_args(t);
    let nt = m.get_func_type(ret, args, false);
    m.set_var_type(f, nt);
}
