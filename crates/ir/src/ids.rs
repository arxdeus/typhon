//! Typed node handles and the `IdMixin` id counter.

use std::cell::Cell;

use la_arena::Idx;

use crate::types::TypeData;
use crate::value::ValueData;
use crate::var::VarData;

/// the `IdMixin` unique id of values and variables.
pub type Id = i64;

/// Handle of a value (`Value*`).
pub type ValueId = Idx<ValueData>;
/// Handle of a variable (`Var*`); functions are variables too.
pub type VarId = Idx<VarData>;
/// Handle of a function (`Func*`). Same arena as variables.
pub type FuncId = VarId;
/// Handle of a type (`Type*`).
pub type TypeId = Idx<TypeData>;

/// Any IR node (`Node*`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum NodeRef {
    Value(ValueId),
    Var(VarId),
    Type(TypeId),
    Module,
}

impl From<ValueId> for NodeRef {
    fn from(v: ValueId) -> Self {
        NodeRef::Value(v)
    }
}
impl From<VarId> for NodeRef {
    fn from(v: VarId) -> Self {
        NodeRef::Var(v)
    }
}
impl From<TypeId> for NodeRef {
    fn from(v: TypeId) -> Self {
        NodeRef::Type(v)
    }
}

impl NodeRef {
    pub fn as_value(self) -> Option<ValueId> {
        match self {
            NodeRef::Value(v) => Some(v),
            _ => None,
        }
    }
    pub fn as_var(self) -> Option<VarId> {
        match self {
            NodeRef::Var(v) => Some(v),
            _ => None,
        }
    }
    pub fn as_type(self) -> Option<TypeId> {
        match self {
            NodeRef::Type(v) => Some(v),
            _ => None,
        }
    }
}

thread_local! {
    // Note: the reference implementation `IdMixin::currentId` is a process-wide static. It is thread-local here
    // so that independent modules compiled on different threads (and parallel unit
    // tests) do not interfere; ids are only compared within one module.
    static CURRENT_ID: Cell<Id> = const { Cell::new(0) };
}

pub fn reset_id() {
    CURRENT_ID.with(|c| c.set(0));
}

/// Allocates the next `IdMixin` id.
pub fn next_id() -> Id {
    CURRENT_ID.with(|c| {
        let v = c.get();
        c.set(v + 1);
        v
    })
}
