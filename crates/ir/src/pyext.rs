
use crate::ids::{FuncId, TypeId};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PyFunctionType {
    #[default]
    Toplevel,
    Method,
    Class,
    Static,
}

#[derive(Clone, Debug, Default)]
pub struct PyFunction {
    pub name: String,
    pub doc: String,
    pub func: Option<FuncId>,
    pub ty: PyFunctionType,
    pub nargs: i32,
    pub keywords: bool,
    pub coexist: bool,
}

/// `PyMember::Type` (values match CPython's `T_*` constants).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(i32)]
pub enum PyMemberType {
    #[default]
    Short = 0,
    Int = 1,
    Long = 2,
    Float = 3,
    Double = 4,
    String = 5,
    Object = 6,
    Char = 7,
    Byte = 8,
    UByte = 9,
    UShort = 10,
    UInt = 11,
    ULong = 12,
    StringInplace = 13,
    Bool = 14,
    ObjectEx = 16,
    LongLong = 17,
    ULongLong = 18,
    PySsizeT = 19,
}

#[derive(Clone, Debug, Default)]
pub struct PyMember {
    pub name: String,
    pub doc: String,
    pub ty: PyMemberType,
    pub readonly: bool,
    pub indexes: Vec<i32>,
}

#[derive(Clone, Debug, Default)]
pub struct PyGetSet {
    pub name: String,
    pub doc: String,
    pub get: Option<FuncId>,
    pub set: Option<FuncId>,
}

/// `base` is an index into `PyModule::types`
/// (Note: the reference stores a `PyType*` into the same vector).
#[derive(Clone, Debug, Default)]
pub struct PyType {
    pub name: String,
    pub doc: String,
    pub ty: Option<TypeId>,
    pub base: Option<usize>,
    pub repr: Option<FuncId>,
    pub add: Option<FuncId>,
    pub iadd: Option<FuncId>,
    pub sub: Option<FuncId>,
    pub isub: Option<FuncId>,
    pub mul: Option<FuncId>,
    pub imul: Option<FuncId>,
    pub mod_: Option<FuncId>,
    pub imod: Option<FuncId>,
    pub divmod: Option<FuncId>,
    pub pow: Option<FuncId>,
    pub ipow: Option<FuncId>,
    pub neg: Option<FuncId>,
    pub pos: Option<FuncId>,
    pub abs: Option<FuncId>,
    pub bool_: Option<FuncId>,
    pub invert: Option<FuncId>,
    pub lshift: Option<FuncId>,
    pub ilshift: Option<FuncId>,
    pub rshift: Option<FuncId>,
    pub irshift: Option<FuncId>,
    pub and_: Option<FuncId>,
    pub iand: Option<FuncId>,
    pub xor_: Option<FuncId>,
    pub ixor: Option<FuncId>,
    pub or_: Option<FuncId>,
    pub ior: Option<FuncId>,
    pub int_: Option<FuncId>,
    pub float_: Option<FuncId>,
    pub floordiv: Option<FuncId>,
    pub ifloordiv: Option<FuncId>,
    pub truediv: Option<FuncId>,
    pub itruediv: Option<FuncId>,
    pub index: Option<FuncId>,
    pub matmul: Option<FuncId>,
    pub imatmul: Option<FuncId>,
    pub len: Option<FuncId>,
    pub getitem: Option<FuncId>,
    pub setitem: Option<FuncId>,
    pub contains: Option<FuncId>,
    pub hash: Option<FuncId>,
    pub call: Option<FuncId>,
    pub str: Option<FuncId>,
    pub cmp: Option<FuncId>,
    pub iter: Option<FuncId>,
    pub iternext: Option<FuncId>,
    pub del: Option<FuncId>,
    pub init: Option<FuncId>,
    pub methods: Vec<PyFunction>,
    pub members: Vec<PyMember>,
    pub getset: Vec<PyGetSet>,
    pub type_ptr_hook: Option<FuncId>,
}

#[derive(Clone, Debug, Default)]
pub struct PyModule {
    pub name: String,
    pub doc: String,
    pub functions: Vec<PyFunction>,
    pub types: Vec<PyType>,
}
