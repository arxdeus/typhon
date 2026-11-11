//! Port of the AST parts of codon/util/serialize.h + codon/util/tser.h:
//! binary (de)serialization of AST subtrees, using `serde` + `postcard`.

use serde::{Deserialize, Serialize};
use typhon_common::{CompilerError, Result};

use super::arena::AstArena;
use super::expr::Expr;
use super::stmt::Stmt;
use crate::ids::{ExprId, StmtId};

/// Serialized form: a compact arena holding only the subtree, plus its root.
#[derive(Serialize, Deserialize)]
struct Archive {
    exprs: Vec<Expr>,
    stmts: Vec<Stmt>,
    root_stmt: Option<StmtId>,
    root_expr: Option<ExprId>,
}

fn io_err(e: postcard::Error) -> CompilerError {
    CompilerError::Io(format!("AST serialization: {e}"))
}

impl AstArena {
    fn to_archive(&self, root_stmt: Option<StmtId>, root_expr: Option<ExprId>) -> Archive {
        let mut tmp = AstArena::new();
        let rs = root_stmt.map(|s| tmp.import_stmt(self, s));
        let re = root_expr.map(|e| tmp.import_expr(self, e));
        let exprs = tmp.exprs().map(|(_, e)| e.clone()).collect();
        let stmts = tmp.stmts().map(|(_, s)| s.clone()).collect();
        Archive { exprs, stmts, root_stmt: rs, root_expr: re }
    }

    /// Serialize a statement subtree (`tser::operator<<(Stmt*)`).
    pub fn serialize_stmt(&self, root: StmtId) -> Result<Vec<u8>> {
        postcard::to_allocvec(&self.to_archive(Some(root), None)).map_err(io_err)
    }

    /// Serialize an expression subtree (`tser::operator<<(Expr*)`).
    pub fn serialize_expr(&self, root: ExprId) -> Result<Vec<u8>> {
        postcard::to_allocvec(&self.to_archive(None, Some(root))).map_err(io_err)
    }

    fn from_archive(bytes: &[u8]) -> Result<(AstArena, Archive)> {
        let mut ar: Archive = postcard::from_bytes(bytes).map_err(io_err)?;
        let mut a = AstArena::new();
        for e in std::mem::take(&mut ar.exprs) {
            a.alloc_expr(e);
        }
        for s in std::mem::take(&mut ar.stmts) {
            a.alloc_stmt(s);
        }
        Ok((a, ar))
    }

    /// Deserialize a statement subtree into a fresh arena; use
    /// [`AstArena::import_stmt`] to move it into another arena.
    pub fn deserialize_stmt(bytes: &[u8]) -> Result<(AstArena, StmtId)> {
        let (a, ar) = Self::from_archive(bytes)?;
        let root = ar.root_stmt.ok_or_else(|| io_err(postcard::Error::DeserializeUnexpectedEnd))?;
        Ok((a, root))
    }

    /// Deserialize an expression subtree into a fresh arena.
    pub fn deserialize_expr(bytes: &[u8]) -> Result<(AstArena, ExprId)> {
        let (a, ar) = Self::from_archive(bytes)?;
        let root = ar.root_expr.ok_or_else(|| io_err(postcard::Error::DeserializeUnexpectedEnd))?;
        Ok((a, root))
    }
}
