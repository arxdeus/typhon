//! Port of codon/parser/visitors/*
//!
//! - `visitor` : visitors/visitor.{h,cpp} (AST lane)
//! - `format`  : visitors/format/format.{h,cpp} (AST lane)
//! - `scoping`, `doc` : visitors/scoping/*, visitors/doc/* (scoping-doc lane)
//! - `translate` : visitors/translate/* (translate lane)

pub mod doc;
pub mod format;
pub mod scoping;
pub mod translate;
pub mod visitor;

pub use format::{FormatCacheHooks, FormatClassRealization, FormatFuncRealization, FormatVisitor};
pub use visitor::{AstCtx, Visitor, expr_children, stmt_children, walk_expr, walk_stmt};
