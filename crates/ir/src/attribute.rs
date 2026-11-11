//! Port of `codon/cir/attribute.{h,cpp}`.

use std::any::Any;
use std::collections::BTreeMap;
use std::fmt;

use rustc_hash::FxHashMap;
use smallvec::SmallVec;
use typhon_common::SrcInfo;

use crate::ids::{FuncId, TypeId, ValueId};

/// Attribute keys (`XAttribute::AttributeID`).
pub mod attr_id {
    pub const STRING_VALUE: i32 = 101;
    pub const INT_VALUE: i32 = 102;
    pub const STRING_LIST: i32 = 103;
    pub const KEY_VALUE: i32 = 104;
    pub const MEMBER: i32 = 105;
    pub const PYTHON_WRAPPER: i32 = 106;
    pub const SRC_INFO: i32 = 107;
    pub const DOCSTRING: i32 = 108;
    pub const TUPLE_LITERAL: i32 = 109;
    // NB: C++ assigns 111 to both list and set literal attributes; preserved.
    pub const LIST_LITERAL: i32 = 111;
    pub const SET_LITERAL: i32 = 111;
    pub const DICT_LITERAL: i32 = 112;
    pub const PARTIAL_FUNCTION: i32 = 113;
    pub const EXACT_TYPE: i32 = 114;
}

/// Extension point for attributes defined outside this crate (C++ subclasses of `Attribute`).
pub trait CustomAttribute: fmt::Debug {
    /// Attribute key used when none is given explicitly.
    fn id(&self) -> i32;
    fn needs_clone(&self) -> bool {
        true
    }
    fn clone_box(&self) -> Box<dyn CustomAttribute>;
    /// Remaps values held by the attribute during cloning (`clone(CloneVisitor&)`).
    fn remap_values(&mut self, _f: &mut dyn FnMut(ValueId) -> ValueId) {}
    fn format(&self) -> String {
        String::new()
    }
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

/// Port of `KeyValueAttribute`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct KeyValueAttribute {
    pub attributes: FxHashMap<String, String>,
}

impl KeyValueAttribute {
    pub fn new(attributes: FxHashMap<String, String>) -> Self {
        Self { attributes }
    }
    pub fn has(&self, key: &str) -> bool {
        self.attributes.contains_key(key)
    }
    pub fn get(&self, key: &str) -> String {
        self.attributes.get(key).cloned().unwrap_or_default()
    }
}

/// Port of `LiteralElement`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LiteralElement {
    pub value: ValueId,
    pub star: bool,
}

/// Port of `DictLiteralAttribute::KeyValuePair`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KeyValuePair {
    pub key: ValueId,
    /// `None` means `**key` (dict unpacking).
    pub value: Option<ValueId>,
}

/// Closed set of IR attributes plus an escape hatch.
#[derive(Debug)]
pub enum Attribute {
    SrcInfo(SrcInfo),
    StringValue(String),
    IntValue(i64),
    StringList(Vec<String>),
    KeyValue(KeyValueAttribute),
    Member(BTreeMap<String, SrcInfo>),
    PythonWrapper(FuncId),
    Docstring(String),
    TupleLiteral(Vec<ValueId>),
    ListLiteral(Vec<LiteralElement>),
    SetLiteral(Vec<LiteralElement>),
    DictLiteral(Vec<KeyValuePair>),
    PartialFunction { name: String, args: Vec<Option<ValueId>> },
    ExactType(TypeId),
    Custom(Box<dyn CustomAttribute>),
}

impl Clone for Attribute {
    fn clone(&self) -> Self {
        match self {
            Attribute::SrcInfo(s) => Attribute::SrcInfo(s.clone()),
            Attribute::StringValue(s) => Attribute::StringValue(s.clone()),
            Attribute::IntValue(s) => Attribute::IntValue(*s),
            Attribute::StringList(s) => Attribute::StringList(s.clone()),
            Attribute::KeyValue(s) => Attribute::KeyValue(s.clone()),
            Attribute::Member(s) => Attribute::Member(s.clone()),
            Attribute::PythonWrapper(s) => Attribute::PythonWrapper(*s),
            Attribute::Docstring(s) => Attribute::Docstring(s.clone()),
            Attribute::TupleLiteral(s) => Attribute::TupleLiteral(s.clone()),
            Attribute::ListLiteral(s) => Attribute::ListLiteral(s.clone()),
            Attribute::SetLiteral(s) => Attribute::SetLiteral(s.clone()),
            Attribute::DictLiteral(s) => Attribute::DictLiteral(s.clone()),
            Attribute::PartialFunction { name, args } => {
                Attribute::PartialFunction { name: name.clone(), args: args.clone() }
            }
            Attribute::ExactType(t) => Attribute::ExactType(*t),
            Attribute::Custom(c) => Attribute::Custom(c.clone_box()),
        }
    }
}

impl Attribute {
    /// Default key of this attribute (`AttributeType::AttributeID`).
    pub fn id(&self) -> i32 {
        use attr_id::*;
        match self {
            Attribute::SrcInfo(_) => SRC_INFO,
            Attribute::StringValue(_) => STRING_VALUE,
            Attribute::IntValue(_) => INT_VALUE,
            Attribute::StringList(_) => STRING_LIST,
            Attribute::KeyValue(_) => KEY_VALUE,
            Attribute::Member(_) => MEMBER,
            Attribute::PythonWrapper(_) => PYTHON_WRAPPER,
            Attribute::Docstring(_) => DOCSTRING,
            Attribute::TupleLiteral(_) => TUPLE_LITERAL,
            Attribute::ListLiteral(_) => LIST_LITERAL,
            Attribute::SetLiteral(_) => SET_LITERAL,
            Attribute::DictLiteral(_) => DICT_LITERAL,
            Attribute::PartialFunction { .. } => PARTIAL_FUNCTION,
            Attribute::ExactType(_) => EXACT_TYPE,
            Attribute::Custom(c) => c.id(),
        }
    }

    /// Port of `Attribute::needsClone()`.
    pub fn needs_clone(&self) -> bool {
        match self {
            Attribute::PythonWrapper(_) | Attribute::ExactType(_) => false,
            Attribute::Custom(c) => c.needs_clone(),
            _ => true,
        }
    }

    /// Whether this attribute references values (and therefore must be cloned through a
    /// `CloneVisitor`; C++ asserts on plain `clone()` for these).
    pub fn holds_values(&self) -> bool {
        matches!(
            self,
            Attribute::TupleLiteral(_)
                | Attribute::ListLiteral(_)
                | Attribute::SetLiteral(_)
                | Attribute::DictLiteral(_)
                | Attribute::PartialFunction { .. }
                | Attribute::PythonWrapper(_)
        )
    }

    /// Clone with every referenced value mapped through `f` (`clone(CloneVisitor&)`).
    pub fn map_values(&self, f: &mut dyn FnMut(ValueId) -> ValueId) -> Attribute {
        match self {
            Attribute::TupleLiteral(v) => Attribute::TupleLiteral(v.iter().map(|x| f(*x)).collect()),
            Attribute::ListLiteral(v) => Attribute::ListLiteral(
                v.iter().map(|e| LiteralElement { value: f(e.value), star: e.star }).collect(),
            ),
            Attribute::SetLiteral(v) => Attribute::SetLiteral(
                v.iter().map(|e| LiteralElement { value: f(e.value), star: e.star }).collect(),
            ),
            Attribute::DictLiteral(v) => Attribute::DictLiteral(
                v.iter().map(|e| KeyValuePair { key: f(e.key), value: e.value.map(&mut *f) }).collect(),
            ),
            Attribute::PartialFunction { name, args } => Attribute::PartialFunction {
                name: name.clone(),
                args: args.iter().map(|a| a.map(&mut *f)).collect(),
            },
            Attribute::Custom(c) => {
                let mut c = c.clone_box();
                c.remap_values(f);
                Attribute::Custom(c)
            }
            other => other.clone(),
        }
    }

    pub fn as_src_info(&self) -> Option<&SrcInfo> {
        match self {
            Attribute::SrcInfo(s) => Some(s),
            _ => None,
        }
    }
    pub fn as_key_value(&self) -> Option<&KeyValueAttribute> {
        match self {
            Attribute::KeyValue(s) => Some(s),
            _ => None,
        }
    }
    pub fn as_key_value_mut(&mut self) -> Option<&mut KeyValueAttribute> {
        match self {
            Attribute::KeyValue(s) => Some(s),
            _ => None,
        }
    }
    pub fn as_string_value(&self) -> Option<&str> {
        match self {
            Attribute::StringValue(s) => Some(s),
            _ => None,
        }
    }
    pub fn as_int_value(&self) -> Option<i64> {
        match self {
            Attribute::IntValue(s) => Some(*s),
            _ => None,
        }
    }
    pub fn as_string_list(&self) -> Option<&[String]> {
        match self {
            Attribute::StringList(s) => Some(s),
            _ => None,
        }
    }
    pub fn as_docstring(&self) -> Option<&str> {
        match self {
            Attribute::Docstring(s) => Some(s),
            _ => None,
        }
    }
    pub fn as_member(&self) -> Option<&BTreeMap<String, SrcInfo>> {
        match self {
            Attribute::Member(s) => Some(s),
            _ => None,
        }
    }
    pub fn as_python_wrapper(&self) -> Option<FuncId> {
        match self {
            Attribute::PythonWrapper(s) => Some(*s),
            _ => None,
        }
    }
    pub fn as_exact_type(&self) -> Option<TypeId> {
        match self {
            Attribute::ExactType(s) => Some(*s),
            _ => None,
        }
    }
    pub fn as_custom<T: 'static>(&self) -> Option<&T> {
        match self {
            Attribute::Custom(c) => c.as_any().downcast_ref(),
            _ => None,
        }
    }
}

/// Port of `AttributeMap`: small ordered association list keyed by attribute id.
#[derive(Clone, Debug, Default)]
pub struct AttributeMap {
    values: SmallVec<[(i32, Attribute); 2]>,
}

impl AttributeMap {
    pub fn iter(&self) -> impl Iterator<Item = &(i32, Attribute)> {
        self.values.iter()
    }
    pub fn keys(&self) -> impl Iterator<Item = i32> + '_ {
        self.values.iter().map(|(k, _)| *k)
    }
    pub fn get(&self, key: i32) -> Option<&Attribute> {
        self.values.iter().find(|(k, _)| *k == key).map(|(_, v)| v)
    }
    pub fn get_mut(&mut self, key: i32) -> Option<&mut Attribute> {
        self.values.iter_mut().find(|(k, _)| *k == key).map(|(_, v)| v)
    }
    pub fn contains(&self, key: i32) -> bool {
        self.values.iter().any(|(k, _)| *k == key)
    }
    /// `attributes[key] = value`.
    pub fn insert(&mut self, key: i32, value: Attribute) {
        if let Some(slot) = self.get_mut(key) {
            *slot = value;
        } else {
            self.values.push((key, value));
        }
    }
    pub fn erase(&mut self, key: i32) {
        if let Some(pos) = self.values.iter().position(|(k, _)| *k == key) {
            self.values.remove(pos);
        }
    }
    pub fn len(&self) -> usize {
        self.values.len()
    }
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}
