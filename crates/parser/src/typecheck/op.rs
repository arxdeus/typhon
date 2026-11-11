//! Port of codon/parser/visitors/typecheck/op.cpp (phase-3 lane: replace STUB bodies).

use typhon_common::Result;

use super::{TypecheckVisitor, not_ported};
use crate::ids::{ExprId, TypeId};

impl<'a> TypecheckVisitor<'a> {
    /// `visit(UnaryExpr*)`.
    pub fn visit_unary(&mut self, _e: ExprId) -> Result<()> {
        Err(not_ported("op::visit_unary", self.src)) // STUB(op.cpp)
    }
    /// `evaluateStaticUnary(expr)`.
    pub fn evaluate_static_unary(&mut self, _e: ExprId) -> Result<Option<ExprId>> {
        Err(not_ported("op::evaluate_static_unary", self.src)) // STUB(op.cpp)
    }
    /// `visit(BinaryExpr*)`.
    pub fn visit_binary(&mut self, _e: ExprId) -> Result<()> {
        Err(not_ported("op::visit_binary", self.src)) // STUB(op.cpp)
    }
    /// `evaluateStaticBinary(expr)`.
    pub fn evaluate_static_binary(&mut self, _e: ExprId) -> Result<Option<ExprId>> {
        Err(not_ported("op::evaluate_static_binary", self.src)) // STUB(op.cpp)
    }
    /// `transformBinarySimple(expr)`.
    pub fn transform_binary_simple(&mut self, _e: ExprId) -> Result<Option<ExprId>> {
        Err(not_ported("op::transform_binary_simple", self.src)) // STUB(op.cpp)
    }
    /// `transformBinaryIs(expr)`.
    pub fn transform_binary_is(&mut self, _e: ExprId) -> Result<Option<ExprId>> {
        Err(not_ported("op::transform_binary_is", self.src)) // STUB(op.cpp)
    }
    /// `getMagic(op)`: `(magic, rightMagic)`.
    pub fn get_magic(&self, _op: &str) -> (String, String) {
        (String::new(), String::new()) // STUB(op.cpp)
    }
    /// `transformBinaryInplaceMagic(expr, isAtomic)`.
    pub fn transform_binary_inplace_magic(&mut self, _e: ExprId, _is_atomic: bool) -> Result<Option<ExprId>> {
        Err(not_ported("op::transform_binary_inplace_magic", self.src)) // STUB(op.cpp)
    }
    /// `transformBinaryMagic(expr)`.
    pub fn transform_binary_magic(&mut self, _e: ExprId) -> Result<Option<ExprId>> {
        Err(not_ported("op::transform_binary_magic", self.src)) // STUB(op.cpp)
    }
    /// `visit(ChainBinaryExpr*)`.
    pub fn visit_chain_binary(&mut self, _e: ExprId) -> Result<()> {
        Err(not_ported("op::visit_chain_binary", self.src)) // STUB(op.cpp)
    }
    /// `visit(PipeExpr*)`.
    pub fn visit_pipe(&mut self, _e: ExprId) -> Result<()> {
        Err(not_ported("op::visit_pipe", self.src)) // STUB(op.cpp)
    }
    /// `visit(IndexExpr*)`.
    pub fn visit_index(&mut self, _e: ExprId) -> Result<()> {
        Err(not_ported("op::visit_index", self.src)) // STUB(op.cpp)
    }
    /// `transformStaticTupleIndex(tuple, expr, index)`: `(handled, replacement)`.
    pub fn transform_static_tuple_index(&mut self, _tuple: TypeId, _e: ExprId, _index: ExprId) -> Result<(bool, Option<ExprId>)> {
        Err(not_ported("op::transform_static_tuple_index", self.src)) // STUB(op.cpp)
    }
    /// `translateIndex(idx, len, clamp)`.
    pub fn translate_index(&self, idx: i64, _len: i64, _clamp: bool) -> i64 {
        idx // STUB(op.cpp)
    }
    /// `sliceAdjustIndices(length, start, stop, step)`: returns the slice length.
    pub fn slice_adjust_indices(&self, _length: i64, _start: &mut i64, _stop: &mut i64, _step: i64) -> i64 {
        0 // STUB(op.cpp)
    }
    /// `visit(InstantiateExpr*)`.
    pub fn visit_instantiate(&mut self, _e: ExprId) -> Result<()> {
        Err(not_ported("op::visit_instantiate", self.src)) // STUB(op.cpp)
    }
    /// `visit(SliceExpr*)`.
    pub fn visit_slice(&mut self, _e: ExprId) -> Result<()> {
        Err(not_ported("op::visit_slice", self.src)) // STUB(op.cpp)
    }
    /// `findEllipsis(expr)` (static in C++): `(position, ellipsis expr)` pairs.
    pub fn find_ellipsis(&self, _e: ExprId) -> Vec<(usize, ExprId)> {
        vec![] // STUB(op.cpp)
    }
}
