//! Port of codon/parser/visitors/typecheck/call.cpp (phase-3 lane: replace STUB bodies).

use typhon_common::Result;

use super::{PartialCallData, TypecheckVisitor, not_ported};
use crate::ast::CallArg;
use crate::ids::{ExprId, StmtId, TypeId};

impl<'a> TypecheckVisitor<'a> {
    /// `visit(PrintStmt*)`.
    pub fn visit_print(&mut self, _s: StmtId) -> Result<()> {
        Err(not_ported("call::visit_print", self.src)) // STUB(call.cpp)
    }
    /// `visit(StarExpr*)`.
    pub fn visit_star(&mut self, _e: ExprId) -> Result<()> {
        Err(not_ported("call::visit_star", self.src)) // STUB(call.cpp)
    }
    /// `visit(KeywordStarExpr*)`.
    pub fn visit_keyword_star(&mut self, _e: ExprId) -> Result<()> {
        Err(not_ported("call::visit_keyword_star", self.src)) // STUB(call.cpp)
    }
    /// `visit(EllipsisExpr*)`.
    pub fn visit_ellipsis(&mut self, _e: ExprId) -> Result<()> {
        Err(not_ported("call::visit_ellipsis", self.src)) // STUB(call.cpp)
    }
    /// `visit(CallExpr*)`.
    pub fn visit_call(&mut self, _e: ExprId) -> Result<()> {
        Err(not_ported("call::visit_call", self.src)) // STUB(call.cpp)
    }
    /// `validateCall(expr)` (static in C++).
    pub fn validate_call(&mut self, _e: ExprId) -> Result<()> {
        Err(not_ported("call::validate_call", self.src)) // STUB(call.cpp)
    }
    /// `transformCallArgs(expr)`: false if arguments are not yet ready.
    pub fn transform_call_args(&mut self, _e: ExprId) -> Result<bool> {
        Err(not_ported("call::transform_call_args", self.src)) // STUB(call.cpp)
    }
    /// `transformOptionalCall(expr, optionalType)`.
    pub fn transform_optional_call(&mut self, _e: ExprId, _opt: TypeId) -> Result<Option<ExprId>> {
        Err(not_ported("call::transform_optional_call", self.src)) // STUB(call.cpp)
    }
    /// `getCalleeFn(expr, part)`: `(callee function type, replacement expr)`.
    pub fn get_callee_fn(&mut self, _e: ExprId, _part: &mut PartialCallData) -> Result<(Option<TypeId>, Option<ExprId>)> {
        Err(not_ported("call::get_callee_fn", self.src)) // STUB(call.cpp)
    }
    /// `callReorderArguments(callee, expr, part)`.
    pub fn call_reorder_arguments(&mut self, _callee: TypeId, _e: ExprId, _part: &mut PartialCallData) -> Result<Option<ExprId>> {
        Err(not_ported("call::call_reorder_arguments", self.src)) // STUB(call.cpp)
    }
    /// `typecheckCallArgs(callee, args, part)`.
    pub fn typecheck_call_args(&mut self, _callee: TypeId, _args: &mut Vec<CallArg>, _part: &PartialCallData) -> Result<bool> {
        Err(not_ported("call::typecheck_call_args", self.src)) // STUB(call.cpp)
    }
    /// `transformSpecialCall(expr)`: `(handled, replacement)`.
    pub fn transform_special_call(&mut self, _e: ExprId) -> Result<(bool, Option<ExprId>)> {
        Err(not_ported("call::transform_special_call", self.src)) // STUB(call.cpp)
    }
    /// `getMRO(type)`.
    pub fn get_mro(&mut self, _t: TypeId) -> Result<Vec<TypeId>> {
        Err(not_ported("call::get_mro", self.src)) // STUB(call.cpp)
    }
    /// `generatePartialCall(mask, fn, args, kwargs)`.
    pub fn generate_partial_call(&mut self, _mask: &str, _f: TypeId, _args: Option<ExprId>, _kwargs: Option<ExprId>) -> Result<ExprId> {
        Err(not_ported("call::generate_partial_call", self.src)) // STUB(call.cpp)
    }
}
