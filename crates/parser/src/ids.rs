//! Shared typed handles for the front-end, defined up-front so that independently
//! ported modules (AST, type system, type checker) agree on them.

use la_arena::{Idx, RawIdx};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// Handle to a type-checker type stored in the `TypeArena` owned by `Cache`
/// (port of `ast::types::TypePtr` / `types::Type*`). Defined here, owned by `types/`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize, serde::Deserialize)]
pub struct TypeId(pub u32);

macro_rules! arena_id {
    ($(#[$m:meta])* $name:ident => $node:ty) => {
        $(#[$m])*
        #[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
        pub struct $name(pub Idx<$node>);

        impl $name {
            /// Raw arena index.
            pub fn index(self) -> u32 {
                self.0.into_raw().into_u32()
            }
            /// Build from a raw arena index (used by (de)serialization).
            pub fn from_index(i: u32) -> Self {
                Self(Idx::from_raw(RawIdx::from_u32(i)))
            }
        }
        impl std::fmt::Debug for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "{}({})", stringify!($name), self.index())
            }
        }
        impl Serialize for $name {
            fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
                self.index().serialize(s)
            }
        }
        impl<'de> Deserialize<'de> for $name {
            fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
                Ok(Self::from_index(u32::deserialize(d)?))
            }
        }
    };
}

arena_id!(
    /// Handle to an AST expression in [`crate::ast::AstArena`] (C++ `ast::Expr *`).
    ExprId => crate::ast::Expr
);
arena_id!(
    /// Handle to an AST statement in [`crate::ast::AstArena`] (C++ `ast::Stmt *`).
    StmtId => crate::ast::Stmt
);
