//! Typhon front-end. Port of `codon/parser/**`.
//!
//! Module ownership (see ../../DESIGN.md):
//! - `ast`, `visitors::visitor`, `visitors::format`, `matching`  : AST lane
//! - `peg`                                                       : grammar lane
//! - `types`, `cache`, `ctx`, `typecheck` (core)                 : sema-core lane
//! - `typecheck/*` remaining bodies, `scoping`, `translate`, `doc`, `capi` : later lanes

pub mod ids;
pub mod ast;
pub mod matching;
pub mod visitors;
pub mod peg;
pub mod types;
pub mod cache;
pub mod ctx;
pub mod typecheck;

pub use ids::TypeId;
