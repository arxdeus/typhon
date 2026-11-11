//! Port of codon/parser/visitors/typecheck/collections.cpp (phase-3 lane: replace STUB bodies).

use typhon_common::Result;

use super::{TypecheckVisitor, not_ported};
use crate::ids::{ExprId, TypeId};

impl<'a> TypecheckVisitor<'a> {
    /// `visit(TupleExpr*)`.
    pub fn visit_tuple(&mut self, _e: ExprId) -> Result<()> {
        Err(not_ported("collections::visit_tuple", self.src)) // STUB(collections.cpp)
    }
    /// `visit(ListExpr*)`.
    pub fn visit_list(&mut self, _e: ExprId) -> Result<()> {
        Err(not_ported("collections::visit_list", self.src)) // STUB(collections.cpp)
    }
    /// `visit(SetExpr*)`.
    pub fn visit_set(&mut self, _e: ExprId) -> Result<()> {
        Err(not_ported("collections::visit_set", self.src)) // STUB(collections.cpp)
    }
    /// `visit(DictExpr*)`.
    pub fn visit_dict(&mut self, _e: ExprId) -> Result<()> {
        Err(not_ported("collections::visit_dict", self.src)) // STUB(collections.cpp)
    }
    /// `transformComprehension(type, fn, items, expectedType)`.
    pub fn transform_comprehension(&mut self, _ty: &str, _f: &str, _items: &mut Vec<ExprId>, _expected: Option<TypeId>) -> Result<Option<ExprId>> {
        Err(not_ported("collections::transform_comprehension", self.src)) // STUB(collections.cpp)
    }
    /// `emptyLiteralOwner()`.
    pub fn empty_literal_owner(&self) -> String {
        String::new() // STUB(collections.cpp)
    }
    /// `widenEmptyLiteralSlot(receiver, slot, actual)`.
    pub fn widen_empty_literal_slot(&mut self, _receiver: ExprId, _slot: usize, _actual: TypeId) -> Result<()> {
        Err(not_ported("collections::widen_empty_literal_slot", self.src)) // STUB(collections.cpp)
    }
    /// `visit(GeneratorExpr*)`.
    pub fn visit_generator(&mut self, _e: ExprId) -> Result<()> {
        Err(not_ported("collections::visit_generator", self.src)) // STUB(collections.cpp)
    }
}
