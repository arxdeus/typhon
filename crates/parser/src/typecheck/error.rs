//! Port of codon/parser/visitors/typecheck/error.cpp (phase-3 lane: replace STUB bodies).

use typhon_common::Result;

use super::{TypecheckVisitor, not_ported};
use crate::ids::StmtId;

impl<'a> TypecheckVisitor<'a> {
    /// `visit(AssertStmt*)`.
    pub fn visit_assert(&mut self, _s: StmtId) -> Result<()> {
        Err(not_ported("error::visit_assert", self.src)) // STUB(error.cpp)
    }
    /// `visit(TryStmt*)`.
    pub fn visit_try(&mut self, _s: StmtId) -> Result<()> {
        Err(not_ported("error::visit_try", self.src)) // STUB(error.cpp)
    }
    /// `visit(ThrowStmt*)`.
    pub fn visit_throw(&mut self, _s: StmtId) -> Result<()> {
        Err(not_ported("error::visit_throw", self.src)) // STUB(error.cpp)
    }
    /// `visit(WithStmt*)`.
    pub fn visit_with(&mut self, _s: StmtId) -> Result<()> {
        Err(not_ported("error::visit_with", self.src)) // STUB(error.cpp)
    }
}
