//! Port of `codon/cir/dsl/nodes.{h,cpp}`: custom node extension points.
//!
//! C++ `dsl::CustomConst/CustomFlow/CustomInstr` are abstract subclasses; here they are a
//! single trait object stored in `ValueKind::Custom`. The LLVM backend obtains builders by
//! downcasting through `as_any()` to a trait it defines (`typhon_llvm::LlvmHooks`).

use std::any::Any;
use std::cell::RefCell;
use std::fmt;
use std::rc::Rc;

use crate::ids::{Id, TypeId, ValueId, VarId};
use crate::module::Module;
use crate::types::TypeData;
use crate::util::cloning::CloneVisitor;
use crate::util::side_effect::SideEffectStatus;

/// Which C++ base class a custom value derives from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CustomCategory {
    Const,
    Flow,
    Instr,
}

/// Port of `CustomConst` / `CustomFlow` / `CustomInstr`.
pub trait CustomValue: fmt::Debug {
    /// Result type (`doGetType`). `None` means the module's none type (default for
    /// flows/instrs, like C++ `Flow::doGetType`/`Instr::doGetType`).
    fn ty(&self, _m: &Module) -> Option<TypeId> {
        None
    }
    fn used_values(&self) -> Vec<ValueId> {
        Vec::new()
    }
    fn replace_used_value(&mut self, _m: &Module, _id: Id, _new: Option<ValueId>) -> i32 {
        0
    }
    fn used_types(&self) -> Vec<TypeId> {
        Vec::new()
    }
    fn replace_used_type(&mut self, _m: &Module, _name: &str, _new: TypeId) -> i32 {
        0
    }
    fn used_variables(&self) -> Vec<VarId> {
        Vec::new()
    }
    fn replace_used_variable(&mut self, _m: &Module, _id: Id, _new: Option<VarId>) -> i32 {
        0
    }
    /// `match(const Value*)`: structural equality with another custom node.
    fn matches(&self, other: &dyn CustomValue) -> bool;
    /// `doClone(CloneVisitor&)`: returns the new value's kind payload.
    fn do_clone(&self, cv: &mut CloneVisitor, m: &mut Module) -> Box<dyn CustomValue>;
    /// `getSideEffectStatus(local)` (flows and instrs only).
    fn side_effect_status(&self, _local: bool) -> SideEffectStatus {
        SideEffectStatus::Unknown
    }
    /// `doFormat`.
    fn format(&self, m: &Module) -> String;
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

/// Payload of `ValueKind::Custom`.
#[derive(Clone, Debug)]
pub struct CustomNode {
    pub category: CustomCategory,
    pub node: Rc<RefCell<Box<dyn CustomValue>>>,
}

impl CustomNode {
    pub fn new(category: CustomCategory, node: Box<dyn CustomValue>) -> Self {
        Self { category, node: Rc::new(RefCell::new(node)) }
    }
}

/// Port of `dsl::CustomType`.
pub trait CustomType: fmt::Debug {
    /// `match(const Type*)`.
    fn matches(&self, other: &TypeData) -> bool;
    fn format(&self, m: &Module) -> String;
    fn is_atomic(&self) -> bool {
        false
    }
    fn is_content_atomic(&self) -> bool {
        true
    }
    fn used_types(&self) -> Vec<TypeId> {
        Vec::new()
    }
    fn as_any(&self) -> &dyn Any;
}

pub type CustomTypeRc = Rc<dyn CustomType>;
