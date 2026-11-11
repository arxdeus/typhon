//! Shared typed handles for the front-end, defined up-front so that independently
//! ported modules (AST, type system, type checker) agree on them.

/// Handle to a type-checker type stored in the `TypeArena` owned by `Cache`
/// (port of `ast::types::TypePtr` / `types::Type*`). Defined here, owned by `types/`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize, serde::Deserialize)]
pub struct TypeId(pub u32);
