//! Port of codon/parser/visitors/typecheck/access.cpp (phase-3 lane: replace STUB bodies).

use typhon_common::Result;

use super::{ItemRef, TypecheckVisitor, not_ported};
use crate::ids::{ExprId, TypeId};

impl<'a> TypecheckVisitor<'a> {
    /// `visit(IdExpr*)`: resolve an identifier, check captures, set type.
    pub fn visit_id(&mut self, _e: ExprId) -> Result<()> {
        Err(not_ported("access::visit_id", self.src)) // STUB(access.cpp)
    }
    /// `checkCapture(item)`.
    pub fn check_capture(&mut self, _item: &ItemRef) -> Result<()> {
        Err(not_ported("access::check_capture", self.src)) // STUB(access.cpp)
    }
    /// `visit(DotExpr*)`.
    pub fn visit_dot(&mut self, _e: ExprId) -> Result<()> {
        Err(not_ported("access::visit_dot", self.src)) // STUB(access.cpp)
    }
    /// `getImport(chain)` (access.cpp overload): longest import prefix of a dotted chain.
    /// PORT-NOTE: renamed from `getImport` to avoid clashing with `getImport(string)`.
    pub fn get_import_chain(&mut self, _chain: &[String]) -> Result<(usize, Option<ItemRef>)> {
        Err(not_ported("access::get_import_chain", self.src)) // STUB(access.cpp)
    }
    /// `getAttr(expr, member, type, parentCall)`: `(replacement, handled)`.
    pub fn get_attr(&mut self, _e: ExprId, _member: &str, _ty: Option<TypeId>, _call: Option<ExprId>) -> Result<(Option<ExprId>, bool)> {
        Err(not_ported("access::get_attr", self.src)) // STUB(access.cpp)
    }
    /// `getClassMember(expr, member, type)`.
    pub fn get_class_member(&mut self, _e: ExprId, _member: &str, _ty: Option<TypeId>) -> Result<(Option<ExprId>, bool)> {
        Err(not_ported("access::get_class_member", self.src)) // STUB(access.cpp)
    }
    /// `getDispatch(fn)`: dispatch function type for an overloaded root.
    pub fn get_dispatch(&mut self, _f: &str) -> Result<Option<TypeId>> {
        Err(not_ported("access::get_dispatch", self.src)) // STUB(access.cpp)
    }
}
