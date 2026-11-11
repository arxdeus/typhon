//! Port of codon/parser/visitors/typecheck/function.cpp (phase-3 lane: replace STUB bodies).

use typhon_common::Result;

use super::{TypecheckVisitor, not_ported};
use crate::ast::Param;
use crate::ids::{ExprId, StmtId, TypeId};

impl<'a> TypecheckVisitor<'a> {
    /// `visit(YieldExpr*)`.
    pub fn visit_yield_expr(&mut self, _e: ExprId) -> Result<()> {
        Err(not_ported("function::visit_yield_expr", self.src)) // STUB(function.cpp)
    }
    /// `visit(AwaitExpr*)`.
    pub fn visit_await(&mut self, _e: ExprId) -> Result<()> {
        Err(not_ported("function::visit_await", self.src)) // STUB(function.cpp)
    }
    /// `visit(ReturnStmt*)`.
    pub fn visit_return(&mut self, _s: StmtId) -> Result<()> {
        Err(not_ported("function::visit_return", self.src)) // STUB(function.cpp)
    }
    /// `visit(YieldStmt*)`.
    pub fn visit_yield(&mut self, _s: StmtId) -> Result<()> {
        Err(not_ported("function::visit_yield", self.src)) // STUB(function.cpp)
    }
    /// `visit(YieldFromStmt*)`.
    pub fn visit_yield_from(&mut self, _s: StmtId) -> Result<()> {
        Err(not_ported("function::visit_yield_from", self.src)) // STUB(function.cpp)
    }
    /// `visit(LambdaExpr*)`.
    pub fn visit_lambda(&mut self, _e: ExprId) -> Result<()> {
        Err(not_ported("function::visit_lambda", self.src)) // STUB(function.cpp)
    }
    /// `visit(GlobalStmt*)`.
    pub fn visit_global(&mut self, _s: StmtId) -> Result<()> {
        Err(not_ported("function::visit_global", self.src)) // STUB(function.cpp)
    }
    /// `visit(FunctionStmt*)`. Must call `TypeArena::register_func_ast` for every new
    /// canonical/realized `FunctionStmt` that a `FuncType` refers to.
    pub fn visit_function(&mut self, _s: StmtId) -> Result<()> {
        Err(not_ported("function::visit_function", self.src)) // STUB(function.cpp)
    }
    /// `transformPythonDefinition(name, args, ret, code)`.
    pub fn transform_python_definition(&mut self, _name: &str, _args: &[Param], _ret: Option<ExprId>, _code: StmtId) -> Result<Option<StmtId>> {
        Err(not_ported("function::transform_python_definition", self.src)) // STUB(function.cpp)
    }
    /// `transformLLVMDefinition(code)`.
    pub fn transform_llvm_definition(&mut self, _code: StmtId) -> Result<Option<StmtId>> {
        Err(not_ported("function::transform_llvm_definition", self.src)) // STUB(function.cpp)
    }
    /// `getDecorator(expr)`: `(isAttribute, attrName, decoratorName)`.
    pub fn get_decorator(&mut self, _e: ExprId) -> Result<(bool, String, String)> {
        Err(not_ported("function::get_decorator", self.src)) // STUB(function.cpp)
    }
    /// `getFuncTypeBase(nargs)`: `Function[Tuple[...], ret]` base for a new FuncType.
    pub fn get_func_type_base(&mut self, _nargs: usize) -> Result<TypeId> {
        Err(not_ported("function::get_func_type_base", self.src)) // STUB(function.cpp)
    }
    /// `prepareCallLayout(ast)` (fills `FunctionStmt::call_layout`).
    pub fn prepare_call_layout(&mut self, _ast: StmtId) {
        // STUB(function.cpp)
    }
}
