//! Port of codon/parser/visitors/typecheck/loops.cpp (phase-3 lane: replace STUB bodies).

use typhon_common::Result;

use super::{AstNode, StaticLoopWrapFn, TypecheckVisitor, not_ported};
use crate::ids::{ExprId, StmtId};

impl<'a> TypecheckVisitor<'a> {
    /// `visit(BreakStmt*)`.
    pub fn visit_break(&mut self, _s: StmtId) -> Result<()> {
        Err(not_ported("loops::visit_break", self.src)) // STUB(loops.cpp)
    }
    /// `visit(ContinueStmt*)`.
    pub fn visit_continue(&mut self, _s: StmtId) -> Result<()> {
        Err(not_ported("loops::visit_continue", self.src)) // STUB(loops.cpp)
    }
    /// `visit(WhileStmt*)`.
    pub fn visit_while(&mut self, _s: StmtId) -> Result<()> {
        Err(not_ported("loops::visit_while", self.src)) // STUB(loops.cpp)
    }
    /// `visit(ForStmt*)`.
    pub fn visit_for(&mut self, _s: StmtId) -> Result<()> {
        Err(not_ported("loops::visit_for", self.src)) // STUB(loops.cpp)
    }
    /// `transformForDecorator(decorator)`.
    pub fn transform_for_decorator(&mut self, _d: Option<ExprId>) -> Result<Option<ExprId>> {
        Err(not_ported("loops::transform_for_decorator", self.src)) // STUB(loops.cpp)
    }
    /// `transformStaticForLoop(stmt)`: `(handled, replacement)`.
    pub fn transform_static_for_loop(&mut self, _s: StmtId) -> Result<(bool, Option<StmtId>)> {
        Err(not_ported("loops::transform_static_for_loop", self.src)) // STUB(loops.cpp)
    }
    /// `transformStaticLoopCall(varExpr, varSuite, iter, wrap, allowNonHeterogenous)`:
    /// `(ok, delay, preamble, items)`. `var_suite` is the slot holding the loop suite.
    pub fn transform_static_loop_call(
        &mut self,
        _var: ExprId,
        _var_suite: &mut Option<StmtId>,
        _iter: ExprId,
        _wrap: &mut StaticLoopWrapFn<'_, 'a>,
        _allow_non_heterogenous: bool,
    ) -> Result<(bool, bool, Option<StmtId>, Vec<AstNode>)> {
        Err(not_ported("loops::transform_static_loop_call", self.src)) // STUB(loops.cpp)
    }
}
