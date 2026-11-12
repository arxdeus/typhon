//! AST visitors: `visitor` (traversal traits), `format` (source-like printer), `scoping`,
//! `doc` (documentation generator) and `translate` (AST -> IR).

pub mod doc;
pub mod format;
pub mod scoping;
pub mod translate;
pub mod visitor;

pub use format::{FormatCacheHooks, FormatClassRealization, FormatFuncRealization, FormatVisitor};
pub use visitor::{AstCtx, Visitor, expr_children, stmt_children, walk_expr, walk_stmt};
