//!
//! Split into submodules (all `impl TypecheckVisitor` blocks):
//! * [`vtables`]: `vTableSignature`, `valueMethodId`, `prepareVTables`;
//! * [`generate`]: `generateSpecialAst` and the `generate*AST` helpers;
//! * [`transform`]: `transform*` handlers of special calls (`isinstance`, `hasattr`, ...);
//! * [`static_fns`]: `std.internal.static` handlers (`fn_can_call`, `vars`, ...);
//! * [`static_loops`]: `populate*Loop` helpers of static `for` loops.

mod generate;
mod prepare;
mod static_fns;
mod static_loops;
mod transform;
mod transform2;
mod union;
mod vtables;

use typhon_common::mangle;

use super::TypecheckVisitor;
use crate::ast::{
    AssignStmt, BinaryExpr, BoolExpr, CallArg, CallExpr, DotExpr, ExprStmt, IdExpr, IfStmt, IndexExpr, IntExpr, NoneExpr,
    ReturnStmt, StringExpr, SuiteStmt, TupleExpr,
};
use crate::ids::{ExprId, StmtId, TypeId};

/// `getMangledMethod("", cls, method)`.
pub(super) fn mm(cls: &str, method: &str) -> String {
    mangle::mangled_method("", cls, method, 0, 0)
}

impl<'a> TypecheckVisitor<'a> {
    // ------------------------------------------------------------------
    // Node construction shorthands (private to the special lane; `sp_` prefix avoids
    // clashes with helpers of other lanes in the same inherent impl namespace).
    // ------------------------------------------------------------------

    pub(super) fn sp_id(&mut self, s: &str) -> ExprId {
        self.n_expr(IdExpr::new(s))
    }
    pub(super) fn sp_int(&mut self, v: i64) -> ExprId {
        self.n_expr(IntExpr::new(v))
    }
    pub(super) fn sp_str(&mut self, s: &str) -> ExprId {
        self.n_expr(StringExpr::new(s, ""))
    }
    pub(super) fn sp_bool(&mut self, v: bool) -> ExprId {
        self.n_expr(BoolExpr { value: v })
    }
    pub(super) fn sp_dot(&mut self, e: ExprId, m: &str) -> ExprId {
        self.n_expr(DotExpr { expr: Some(e), member: m.into() })
    }
    pub(super) fn sp_index(&mut self, e: ExprId, i: ExprId) -> ExprId {
        self.n_expr(IndexExpr { expr: Some(e), index: Some(i) })
    }
    pub(super) fn sp_tuple(&mut self, items: Vec<ExprId>) -> ExprId {
        self.n_expr(TupleExpr { items })
    }
    pub(super) fn sp_bin(&mut self, l: ExprId, op: &str, r: ExprId) -> ExprId {
        self.n_expr(BinaryExpr { op: op.into(), lexpr: Some(l), rexpr: Some(r), in_place: false })
    }
    /// `N<CallExpr>(f, args...)` with positional arguments.
    pub(super) fn sp_call(&mut self, f: ExprId, args: Vec<ExprId>) -> ExprId {
        let items = args.into_iter().map(|a| self.cache.ast.call_arg("", Some(a))).collect();
        self.sp_call_args(f, items)
    }
    /// `N<CallExpr>(f, std::vector<CallArg>)`.
    pub(super) fn sp_call_args(&mut self, f: ExprId, items: Vec<CallArg>) -> ExprId {
        self.n_expr(CallExpr { expr: Some(f), items, ordered: false, partial: false })
    }
    /// `typed_argument(type)`: a done `NoneExpr` placeholder of a given type.
    pub(super) fn sp_typed(&mut self, t: TypeId) -> ExprId {
        let v = self.n_expr(NoneExpr);
        self.set_type(v, t);
        self.set_done_expr(v);
        v
    }
    /// A `NoneExpr` placeholder with a type (not marked done).
    pub(super) fn sp_none_of(&mut self, t: TypeId) -> ExprId {
        let v = self.n_expr(NoneExpr);
        self.set_type(v, t);
        v
    }
    /// `Literal[kind]` type annotation.
    pub(super) fn sp_literal(&mut self, kind: &str) -> ExprId {
        let l = self.sp_id("Literal");
        let k = self.sp_id(kind);
        self.sp_index(l, k)
    }
    pub(super) fn sp_return(&mut self, e: Option<ExprId>) -> StmtId {
        self.n_stmt(ReturnStmt { expr: e })
    }
    pub(super) fn sp_expr_stmt(&mut self, e: ExprId) -> StmtId {
        self.n_stmt(ExprStmt { expr: Some(e) })
    }
    pub(super) fn sp_suite(&mut self, items: Vec<StmtId>) -> StmtId {
        self.n_stmt(SuiteStmt { items })
    }
    pub(super) fn sp_if(&mut self, cond: Option<ExprId>, s: StmtId) -> StmtId {
        let s = self.cache.ast.wrap_suite(Some(s));
        self.n_stmt(IfStmt { cond, if_suite: s, else_suite: None })
    }
    pub(super) fn sp_assign(&mut self, lhs: ExprId, rhs: Option<ExprId>, ty: Option<ExprId>) -> StmtId {
        self.n_stmt(AssignStmt { lhs: Some(lhs), rhs, type_expr: ty, update: Default::default() })
    }

    // ------------------------------------------------------------------
    // Accessors
    // ------------------------------------------------------------------

    /// Copy of a `CallExpr` node.
    pub(super) fn sp_callx(&self, e: ExprId) -> CallExpr {
        self.cache.ast[e].kind.as_call().expect("expected a call").clone()
    }
    /// `expr->getExpr()->getType()` of a call.
    pub(super) fn sp_callee_type(&self, e: ExprId) -> TypeId {
        let c = self.cache.ast[e].kind.as_call().expect("expected a call");
        self.ety(c.expr.expect("call without callee"))
    }
    /// `(*expr)[i].getExpr()`.
    pub(super) fn sp_arg(&self, e: ExprId, i: usize) -> ExprId {
        self.cache.ast[e].kind.as_call().expect("expected a call").items[i].value.expect("missing call argument")
    }
    /// `(*expr)[i].value = v`.
    pub(super) fn sp_set_arg(&mut self, e: ExprId, i: usize, v: ExprId) {
        self.cache.ast[e].kind.as_call_mut().expect("expected a call").items[i].value = Some(v);
    }
    /// `if (auto u = expr->getType()->getUnbound()) u->staticKind = kind;`
    pub(super) fn sp_set_static_kind(&mut self, e: ExprId, kind: crate::types::LiteralKind) {
        let types = self.types();
        if let Some(t) = self.expr_type(e)
            && let Some(u) = types.get_unbound(t)
        {
            types.with_link_mut(u, |l| l.static_kind = kind);
        }
    }
    /// Parameters of the `FunctionStmt` behind a function type.
    pub(super) fn sp_fn_params(&self, f: TypeId) -> Vec<crate::ast::Param> {
        let aid = self.cache.types.func_ast_id(f);
        self.cache.ast[aid].kind.as_function().map(|f| f.items.clone()).unwrap_or_default()
    }
    /// Name of the `FunctionStmt` behind a function type (`type->ast->name`).
    pub(super) fn sp_fn_ast_name(&self, f: TypeId) -> String {
        let aid = self.cache.types.func_ast_id(f);
        self.cache.ast[aid].kind.as_function().map(|f| f.name.clone()).unwrap_or_default()
    }
    /// `getClass(t)->hasRTTI()`.
    pub(super) fn sp_has_rtti(&self, t: TypeId) -> bool {
        let Some(cls) = self.cache.get_class_by_type(t) else { return false };
        let cname = cls.ast.and_then(|a| self.cache.ast[a].kind.as_class().map(|c| c.name.clone())).unwrap_or_default();
        cls.has_rtti(&cname)
    }
    /// `realizedName()` shorthand.
    pub(super) fn sp_rn(&self, t: TypeId) -> String {
        self.cache.types.realized_name(t)
    }
    /// `cache->generatedTupleNames[id]`.
    pub(super) fn sp_tuple_names(&self, id: i64) -> Vec<String> {
        self.cache.types.generated_tuple_names.borrow().get(id as usize).cloned().unwrap_or_default()
    }
}
