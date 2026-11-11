//! Port of codon/parser/visitors/typecheck/cond.cpp (phase-3 lane: replace STUB bodies).

use typhon_common::Result;

use super::{TypecheckVisitor, not_ported};
use crate::ids::{ExprId, StmtId};

impl<'a> TypecheckVisitor<'a> {
    /// `visit(RangeExpr*)`.
    pub fn visit_range(&mut self, _e: ExprId) -> Result<()> {
        Err(not_ported("cond::visit_range", self.src)) // STUB(cond.cpp)
    }
    /// `visit(IfExpr*)`.
    pub fn visit_if_expr(&mut self, _e: ExprId) -> Result<()> {
        Err(not_ported("cond::visit_if_expr", self.src)) // STUB(cond.cpp)
    }
    /// `visit(IfStmt*)`.
    pub fn visit_if(&mut self, _s: StmtId) -> Result<()> {
        Err(not_ported("cond::visit_if", self.src)) // STUB(cond.cpp)
    }
    /// `visit(MatchStmt*)`.
    pub fn visit_match(&mut self, _s: StmtId) -> Result<()> {
        Err(not_ported("cond::visit_match", self.src)) // STUB(cond.cpp)
    }
    /// `transformPattern(var, pattern, suite)`.
    pub fn transform_pattern(&mut self, _var: ExprId, _pattern: ExprId, _suite: StmtId) -> Result<Option<StmtId>> {
        Err(not_ported("cond::transform_pattern", self.src)) // STUB(cond.cpp)
    }
}
