//! Port of codon/parser/visitors/typecheck/assign.cpp (phase-3 lane: replace STUB bodies).

use typhon_common::Result;

use super::{TypecheckVisitor, not_ported};
use crate::ids::{ExprId, StmtId};

impl<'a> TypecheckVisitor<'a> {
    /// `visit(AssignExpr*)` (walrus).
    pub fn visit_assign_expr(&mut self, _e: ExprId) -> Result<()> {
        Err(not_ported("assign::visit_assign_expr", self.src)) // STUB(assign.cpp)
    }
    /// `visit(AssignStmt*)`.
    pub fn visit_assign(&mut self, _s: StmtId) -> Result<()> {
        Err(not_ported("assign::visit_assign", self.src)) // STUB(assign.cpp)
    }
    /// `unpackAssignment(lhs, rhs, isLoopTarget)`.
    pub fn unpack_assignment(&mut self, _lhs: ExprId, _rhs: ExprId, _is_loop_target: bool) -> Result<Option<StmtId>> {
        Err(not_ported("assign::unpack_assignment", self.src)) // STUB(assign.cpp)
    }
    /// `transformUpdate(stmt)`.
    pub fn transform_update(&mut self, _s: StmtId) -> Result<Option<StmtId>> {
        Err(not_ported("assign::transform_update", self.src)) // STUB(assign.cpp)
    }
    /// `transformAssignment(stmt, mustExist)`.
    pub fn transform_assignment(&mut self, _s: StmtId, _must_exist: bool) -> Result<Option<StmtId>> {
        Err(not_ported("assign::transform_assignment", self.src)) // STUB(assign.cpp)
    }
    /// `visit(DelStmt*)`.
    pub fn visit_del(&mut self, _s: StmtId) -> Result<()> {
        Err(not_ported("assign::visit_del", self.src)) // STUB(assign.cpp)
    }
    /// `visit(AssignMemberStmt*)`.
    pub fn visit_assign_member(&mut self, _s: StmtId) -> Result<()> {
        Err(not_ported("assign::visit_assign_member", self.src)) // STUB(assign.cpp)
    }
    /// `transformInplaceUpdate(stmt)`: `(handled, replacement)`.
    pub fn transform_inplace_update(&mut self, _s: StmtId) -> Result<(bool, Option<StmtId>)> {
        Err(not_ported("assign::transform_inplace_update", self.src)) // STUB(assign.cpp)
    }
}
