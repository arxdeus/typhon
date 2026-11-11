//! Typhon front-end. Port of `codon/parser/**`.
//!
//! Module ownership (see ../../DESIGN.md):
//! - `ast`, `visitors::visitor`, `visitors::format`, `matching`  : phase 1 (AST lane)
//! - `peg`                                                       : phase 2 (grammar lane)
//! - `types`, `cache`, `ctx`, `typecheck` skeleton               : phase 2 (sema lane)
//! - `typecheck/*` bodies, `scoping`, `translate`, `doc`, `capi` : phase 3

pub mod ids;
pub mod ast;
pub mod matching;
pub mod visitors;

pub use ids::TypeId;
