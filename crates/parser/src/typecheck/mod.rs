//!
//! File -> module map (one `impl TypecheckVisitor` block per the reference implementation file):
//! `typecheck.cpp` -> [`typecheck`] (transform/apply/dispatch) + [`helpers`] (the rest),
//! `infer.cpp` -> [`infer`], `basic.cpp` -> [`basic`], `ctx.{h,cpp}` -> [`ctx`],
//! `access/assign/call/class/collections/cond/error/function/import/loops/op/special.cpp`
//! -> same-named modules (phase-3 lanes). See `../../TYPECHECK_API.md`.

pub mod access;
pub mod assign;
pub mod basic;
pub mod call;
pub mod class;
pub mod collections;
pub mod cond;
pub mod ctx;
pub mod error;
pub mod function;
pub mod helpers;
pub mod helpers_call;
pub mod helpers_py;
pub mod import;
pub mod infer_func;
pub mod infer_ir;
pub mod infer;
pub mod loops;
pub mod op;
pub mod special;
pub mod typecheck;

use std::cell::RefCell;
use std::rc::Rc;

use typhon_common::{Result, SrcInfo};

use crate::ast::{AstArena, Attr, ExprKind, StmtKind};
use crate::cache::Cache;
use crate::ids::{ExprId, StmtId, TypeId};
use crate::types::{TypeArena, TypeHost, Unification};
use crate::visitors::visitor::AstCtx;
pub use ctx::{Base, CtxRef, ItemRef, Loop, TypeContext, TypecheckItem};

/// Statements to prepend before the current statement (shared between nested visitors).
pub type PrependStmts = Rc<RefCell<Vec<StmtId>>>;

/// `TypecheckVisitor::PartialCallData`.
#[derive(Clone, Debug, Default)]
pub struct PartialCallData {
    /// True if the call is partial.
    pub is_partial: bool,
    /// Set if calling a partial type itself.
    pub var: String,
    /// Mask of known arguments.
    pub known: String,
    /// Partial `*args` / `**kwargs` expressions.
    pub args: Option<ExprId>,
    pub kw_args: Option<ExprId>,
}

/// `CallArgSlots`: for each parameter, the indices of the supplied arguments.
pub type CallArgSlots = Vec<smallvec::SmallVec<[i32; 1]>>;
/// `ReorderDoneFn(starIdx, kwstarIdx, slots, partial) -> score`.
pub type ReorderDoneFn<'f, 'a> = dyn FnMut(&mut TypecheckVisitor<'a>, i32, i32, &CallArgSlots, bool) -> Result<i32> + 'f;
/// `ReorderErrorFn(error, src, message) -> score` (usually returns an `Err`).
pub type ReorderErrorFn<'f, 'a> =
    dyn FnMut(&mut TypecheckVisitor<'a>, typhon_common::ErrorKind, SrcInfo, String) -> Result<i32> + 'f;
/// Wrapping function produced by `canWrapExpr` (`std::function<Expr *(Expr *)>`).
pub type WrapFn = Box<dyn for<'b> FnOnce(&mut TypecheckVisitor<'b>, ExprId) -> Result<ExprId>>;
/// Static-loop wrapper (`std::function<ASTNode *(Stmt *)>`); returns the produced node.
pub type StaticLoopWrapFn<'f, 'a> = dyn FnMut(&mut TypecheckVisitor<'a>, StmtId) -> Result<AstNode> + 'f;

/// `ASTNode *` (expression or statement).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AstNode {
    Expr(ExprId),
    Stmt(StmtId),
}

/// Visitor that infers expression types and performs type-guided transformations.
/// Modifies the AST in place (semantics).
pub struct TypecheckVisitor<'a> {
    /// Shared cache (owns the AST and type arenas).
    pub cache: &'a mut Cache,
    /// Current typechecking context.
    pub ctx: CtxRef,
    /// Statements to prepend before the current statement.
    pub prepend_stmts: PrependStmts,
    /// Preamble suite (realized definitions emitted before the module body).
    pub preamble: StmtId,
    /// Replacement for the node being visited (`resultExpr`).
    pub result_expr: Option<ExprId>,
    /// Replacement for the statement being visited (`resultStmt`).
    pub result_stmt: Option<StmtId>,
    /// Source location of the current node (`SrcObject` base of the visitor).
    pub src: SrcInfo,
}

impl<'a> TypecheckVisitor<'a> {
    /// `TypecheckVisitor(ctx, preamble, stmts)`.
    pub fn new(cache: &'a mut Cache, ctx: CtxRef, preamble: Option<StmtId>, stmts: Option<PrependStmts>) -> Self {
        let preamble = match preamble {
            Some(p) => p,
            None => {
                let src = SrcInfo::default();
                cache.ast.new_stmt(crate::ast::SuiteStmt::default(), src)
            }
        };
        Self {
            cache,
            ctx,
            prepend_stmts: stmts.unwrap_or_default(),
            preamble,
            result_expr: None,
            result_stmt: None,
            src: SrcInfo::default(),
        }
    }

    /// A nested visitor sharing context/preamble (`TypecheckVisitor v(ctx, preamble, ...)`).
    pub fn nested(&mut self, share_prepend: bool) -> TypecheckVisitor<'_> {
        let ctx = self.ctx.clone();
        let pre = self.preamble;
        let ps = if share_prepend { Some(self.prepend_stmts.clone()) } else { None };
        TypecheckVisitor::new(self.cache, ctx, Some(pre), ps)
    }
    /// A visitor over another context (`TypecheckVisitor(otherCtx)`).
    pub fn with_ctx(&mut self, ctx: CtxRef) -> TypecheckVisitor<'_> {
        TypecheckVisitor::new(self.cache, ctx, None, None)
    }

    /// Shared type arena.
    pub fn types(&self) -> Rc<TypeArena> {
        self.cache.types.clone()
    }
    /// `getSrcInfo()`.
    pub fn src_info(&self) -> SrcInfo {
        self.src
    }
    pub fn ast(&self) -> &AstArena {
        &self.cache.ast
    }
    pub fn ast_mut(&mut self) -> &mut AstArena {
        &mut self.cache.ast
    }

    /// `N<T>(...)` for expressions: allocate with the visitor's source location.
    pub fn n_expr(&mut self, kind: impl Into<ExprKind>) -> ExprId {
        let src = self.src;
        self.cache.ast.new_expr(kind, src)
    }
    /// `N<T>(...)` for statements: visitor's location + `Attr::ExprTime` if time is set.
    pub fn n_stmt(&mut self, kind: impl Into<StmtKind>) -> StmtId {
        let src = self.src;
        let s = self.cache.ast.new_stmt(kind, src);
        let t = self.get_time();
        if t != 0 {
            self.cache.ast[s].attrs.set_int(Attr::ExprTime, t);
        }
        s
    }
    /// `NC<T>(...)` (no source location).
    pub fn nc_expr(&mut self, kind: impl Into<ExprKind>) -> ExprId {
        self.cache.ast.new_expr(kind, SrcInfo::default())
    }
    pub fn nc_stmt(&mut self, kind: impl Into<StmtKind>) -> StmtId {
        self.cache.ast.new_stmt(kind, SrcInfo::default())
    }
    /// `expr->getType()`, allocating nothing (None if unset).
    pub fn expr_type(&self, e: ExprId) -> Option<TypeId> {
        self.cache.ast[e].ty
    }
    /// `expr->getType()` (asserts set).
    pub fn ety(&self, e: ExprId) -> TypeId {
        self.cache.ast[e].ty.expect("expression has no type")
    }
    /// `expr->setType(t)`.
    pub fn set_type(&mut self, e: ExprId, t: TypeId) {
        self.cache.ast[e].ty = Some(t);
    }
    /// `expr->setDone()` / `stmt->setDone()`.
    pub fn set_done_expr(&mut self, e: ExprId) {
        self.cache.ast[e].done = true;
    }
    pub fn set_done_stmt(&mut self, s: StmtId) {
        self.cache.ast[s].done = true;
    }
    /// `expr->getClassType()`.
    pub fn expr_class(&self, e: ExprId) -> Option<TypeId> {
        self.cache.ast[e].ty.and_then(|t| self.cache.types.get_class(t))
    }

    /// `W(error, src, args...)`: emit a compilation warning.
    pub fn warn(&self, kind: typhon_common::ErrorKind, src: SrcInfo, args: &[&dyn std::fmt::Display]) {
        let msg = typhon_common::emsg(kind, args);
        typhon_common::diag::compilation_warning(
            &msg,
            src.file.as_str(),
            src.line,
            src.col,
            src.len,
            kind as i32,
            false,
            Default::default(),
        );
    }
}

impl AstCtx for TypecheckVisitor<'_> {
    fn ast(&self) -> &AstArena {
        &self.cache.ast
    }
    fn ast_mut(&mut self) -> &mut AstArena {
        &mut self.cache.ast
    }
}

/// Type operations that call back into the type checker.
impl TypeHost for TypecheckVisitor<'_> {
    fn match_callable(&mut self, typ: TypeId, args: &[TypeId], ret: Option<TypeId>, undo: Option<&mut Unification>) -> Result<i32> {
        let mut call_args = vec![];
        for a in args {
            let v = self.n_expr(crate::ast::NoneExpr);
            self.set_type(v, *a);
            call_args.push(crate::ast::CallArg::new("", Some(v)));
        }
        self.match_callable(typ, &call_args, ret, undo, None, None)
    }
    fn instantiate_tuple(&mut self, n: usize, args: Option<&[TypeId]>) -> Result<TypeId> {
        let t = self.generate_tuple(n, true)?;
        match args {
            Some(a) => self.instantiate_type_with(t, a),
            None => self.instantiate_type(t, None),
        }
    }
}

/// Error returned by not-yet-ported methods (`// STUB(<file>.cpp)` bodies).
pub(crate) fn not_ported(what: &str, src: SrcInfo) -> typhon_common::CompilerError {
    typhon_common::CompilerError::custom(format!("typecheck::{what} not ported yet"), src)
}
