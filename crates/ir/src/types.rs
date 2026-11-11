//! Port of `codon/cir/type.{h,cpp}` (data part; behaviour lives on `Module`).

use std::collections::BTreeMap;

use crate::base::NodeBase;
use crate::dsl::CustomTypeRc;
use crate::ids::{FuncId, TypeId};
use crate::realize::AstTypeRef;

/// Port of `Generic`: a static int, a static string or a type.
#[derive(Clone, Debug, PartialEq)]
pub enum Generic {
    Static(i64),
    StaticStr(String),
    Type(TypeId),
}

impl Generic {
    pub fn is_type(&self) -> bool {
        matches!(self, Generic::Type(_))
    }
    pub fn is_static(&self) -> bool {
        matches!(self, Generic::Static(_))
    }
    pub fn is_static_str(&self) -> bool {
        matches!(self, Generic::StaticStr(_))
    }
    pub fn static_value(&self) -> i64 {
        match self {
            Generic::Static(v) => *v,
            _ => 0,
        }
    }
    pub fn static_string_value(&self) -> String {
        match self {
            Generic::StaticStr(v) => v.clone(),
            _ => String::new(),
        }
    }
    pub fn type_value(&self) -> Option<TypeId> {
        match self {
            Generic::Type(t) => Some(*t),
            _ => None,
        }
    }
}

impl From<i64> for Generic {
    fn from(v: i64) -> Self {
        Generic::Static(v)
    }
}
impl From<&str> for Generic {
    fn from(v: &str) -> Self {
        Generic::StaticStr(v.to_string())
    }
}
impl From<String> for Generic {
    fn from(v: String) -> Self {
        Generic::StaticStr(v)
    }
}
impl From<TypeId> for Generic {
    fn from(v: TypeId) -> Self {
        Generic::Type(v)
    }
}

/// Port of `Type::RuntimeInfo`.
#[derive(Clone, Debug, Default)]
pub struct RuntimeInfo {
    pub id: i64,
    pub raw_name: String,
    pub nice_name: String,
    pub base_name: String,
    pub mro: Vec<TypeId>,
    pub field_owners: Vec<TypeId>,
    pub parameters: Vec<TypeId>,
    pub methods: BTreeMap<usize, FuncId>,
    pub value_accessed: bool,
    pub exception_base: Option<TypeId>,
    pub system_exit_base: Option<TypeId>,
}

/// Port of `MemberedType::Field`.
#[derive(Clone, Debug, PartialEq)]
pub struct Field {
    pub name: String,
    pub ty: TypeId,
}

impl Field {
    pub fn new(name: impl Into<String>, ty: TypeId) -> Self {
        Self { name: name.into(), ty }
    }
}

/// Port of the `Type` class hierarchy.
#[derive(Clone, Debug)]
pub enum TypeKind {
    Int { len: u32, sign: bool },
    Float,
    Float32,
    Float16,
    BFloat16,
    Float128,
    Bool,
    Record { fields: Vec<Field> },
    Ref { contents: TypeId, polymorphic: bool },
    Func { ret: TypeId, args: Vec<TypeId>, variadic: bool },
    Pointer { base: TypeId },
    Optional { base: TypeId },
    Generator { base: TypeId },
    Vector { count: u32, base: TypeId },
    Union { types: Vec<TypeId> },
    Custom(CustomTypeRc),
}

impl TypeKind {
    /// `isA<PrimitiveType>`.
    pub fn is_primitive(&self) -> bool {
        matches!(
            self,
            TypeKind::Int { .. }
                | TypeKind::Float
                | TypeKind::Float32
                | TypeKind::Float16
                | TypeKind::BFloat16
                | TypeKind::Float128
                | TypeKind::Bool
                | TypeKind::Vector { .. }
        )
    }
    /// `isA<MemberedType>`.
    pub fn is_membered(&self) -> bool {
        matches!(self, TypeKind::Record { .. } | TypeKind::Ref { .. })
    }
    /// `isA<DerivedType>`.
    pub fn is_derived(&self) -> bool {
        matches!(self, TypeKind::Pointer { .. } | TypeKind::Optional { .. } | TypeKind::Generator { .. })
    }
    pub fn is_record(&self) -> bool {
        matches!(self, TypeKind::Record { .. })
    }
    pub fn is_ref(&self) -> bool {
        matches!(self, TypeKind::Ref { .. })
    }
    pub fn is_func(&self) -> bool {
        matches!(self, TypeKind::Func { .. })
    }
    pub fn is_pointer(&self) -> bool {
        matches!(self, TypeKind::Pointer { .. })
    }
    pub fn is_optional(&self) -> bool {
        matches!(self, TypeKind::Optional { .. })
    }
    pub fn is_generator(&self) -> bool {
        matches!(self, TypeKind::Generator { .. })
    }
    pub fn is_union(&self) -> bool {
        matches!(self, TypeKind::Union { .. })
    }
    pub fn is_int(&self) -> bool {
        matches!(self, TypeKind::Int { .. })
    }
    pub fn is_vector(&self) -> bool {
        matches!(self, TypeKind::Vector { .. })
    }
    /// Base of a derived type (`DerivedType::getBase`) or vector element type.
    pub fn base(&self) -> Option<TypeId> {
        match self {
            TypeKind::Pointer { base }
            | TypeKind::Optional { base }
            | TypeKind::Generator { base }
            | TypeKind::Vector { base, .. } => Some(*base),
            _ => None,
        }
    }
    /// Short kind name used in diagnostics.
    pub fn kind_name(&self) -> &'static str {
        match self {
            TypeKind::Int { .. } => "IntType",
            TypeKind::Float => "FloatType",
            TypeKind::Float32 => "Float32Type",
            TypeKind::Float16 => "Float16Type",
            TypeKind::BFloat16 => "BFloat16Type",
            TypeKind::Float128 => "Float128Type",
            TypeKind::Bool => "BoolType",
            TypeKind::Record { .. } => "RecordType",
            TypeKind::Ref { .. } => "RefType",
            TypeKind::Func { .. } => "FuncType",
            TypeKind::Pointer { .. } => "PointerType",
            TypeKind::Optional { .. } => "OptionalType",
            TypeKind::Generator { .. } => "GeneratorType",
            TypeKind::Vector { .. } => "VectorType",
            TypeKind::Union { .. } => "UnionType",
            TypeKind::Custom(_) => "CustomType",
        }
    }
}

/// Storage of one IR type.
#[derive(Clone, Debug)]
pub struct TypeData {
    pub base: NodeBase,
    pub kind: TypeKind,
    pub ast_type: Option<AstTypeRef>,
    pub runtime_info: Option<Box<RuntimeInfo>>,
    pub(crate) replacement: Option<TypeId>,
    pub(crate) alive: bool,
}

impl TypeData {
    pub fn new(name: impl Into<String>, kind: TypeKind) -> Self {
        Self {
            base: NodeBase::new(name),
            kind,
            ast_type: None,
            runtime_info: None,
            replacement: None,
            alive: true,
        }
    }
    pub fn name(&self) -> &str {
        &self.base.name
    }
}

/// `IntType::MAX_LEN`.
pub const INT_MAX_LEN: u32 = 2048;

/// `IntType::getInstanceName`.
pub fn int_type_name(len: u32, sign: bool) -> String {
    format!("{}Int{}", if sign { "" } else { "U" }, len)
}
/// `PointerType::getInstanceName` (takes the base reference string, i.e. its name).
pub fn pointer_type_name(base: &str) -> String {
    format!("Pointer[{base}]")
}
/// `OptionalType::getInstanceName`.
pub fn optional_type_name(base: &str) -> String {
    format!("Optional[{base}]")
}
/// `GeneratorType::getInstanceName`.
pub fn generator_type_name(base: &str) -> String {
    format!("Generator[{base}]")
}
/// `VectorType::getInstanceName`.
pub fn vector_type_name(count: u32, base: &str) -> String {
    format!("Vector[{count}, {base}]")
}
/// `UnionType::getInstanceName`.
pub fn union_type_name<S: AsRef<str>>(types: &[S]) -> String {
    let names: Vec<&str> = types.iter().map(|s| s.as_ref()).collect();
    format!("Union[{}]", names.join(", "))
}
