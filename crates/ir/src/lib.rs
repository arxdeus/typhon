//! typhon-ir: the Typhon IR core (nodes, types, module, util, pass/analysis traits,
//! OpenMP schedule, rewriter).
//!
//! See `API.md` in the crate root for the public API 

pub mod attribute;
pub mod base;
pub mod ids;
pub mod module;
pub mod pass;
pub mod realize;
pub mod rewrite;
pub mod schedule;
pub mod types;
pub mod util;
pub mod value;
pub mod var;

pub use attribute::{Attribute, AttributeMap, KeyValueAttribute, attr_id};
pub use base::NodeBase;
pub use ids::{FuncId, Id, NodeRef, TypeId, ValueId, VarId, reset_id};
pub use module::{Module, magic};
pub use realize::{AstTypeRef, TypeRealizer};
pub use types::{Field, Generic, RuntimeInfo, TypeData, TypeKind};
pub use value::*;
pub use var::{FuncData, FuncKind, Intrinsic, VarData};
