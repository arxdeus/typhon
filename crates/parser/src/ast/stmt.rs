
use std::collections::HashSet;
use std::rc::Rc;

use serde::{Deserialize, Serialize};
use typhon_common::SrcInfo;

use super::attr::Attrs;
use super::expr::{CallArg, Param};
use super::src_serde;
use crate::ids::{ExprId, StmtId};

/// Base statement node (`ast::Stmt`).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Stmt {
    pub kind: StmtKind,
    #[serde(with = "src_serde")]
    pub src: SrcInfo,
    pub attrs: Attrs,
    /// True if all types in a statement are inferred.
    pub done: bool,
}

impl Stmt {
    pub fn new(kind: StmtKind, src: SrcInfo) -> Self {
        Self { kind, src, attrs: Attrs::new(), done: false }
    }
    pub fn is_done(&self) -> bool {
        self.done
    }
    pub fn set_done(&mut self) {
        self.done = true;
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct SuiteStmt {
    pub items: Vec<StmtId>,
}
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct BreakStmt;
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ContinueStmt;
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ExprStmt {
    pub expr: Option<ExprId>,
}

/// `AssignStmt::UpdateMode`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum UpdateMode {
    #[default]
    Assign,
    Update,
    UpdateAtomic,
    ThreadLocalAssign,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct AssignStmt {
    pub lhs: Option<ExprId>,
    pub rhs: Option<ExprId>,
    pub type_expr: Option<ExprId>,
    pub update: UpdateMode,
}
impl AssignStmt {
    pub fn is_assignment(&self) -> bool {
        self.update == UpdateMode::Assign
    }
    pub fn is_update(&self) -> bool {
        self.update == UpdateMode::Update
    }
    pub fn is_atomic_update(&self) -> bool {
        self.update == UpdateMode::UpdateAtomic
    }
    pub fn is_thread_local(&self) -> bool {
        self.update == UpdateMode::ThreadLocalAssign
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct DelStmt {
    pub expr: Option<ExprId>,
}
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct PrintStmt {
    pub items: Vec<ExprId>,
    /// True if there is a dangling comma after print: `print a,`.
    pub no_newline: bool,
}
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ReturnStmt {
    pub expr: Option<ExprId>,
}
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct YieldStmt {
    pub expr: Option<ExprId>,
}
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct AssertStmt {
    pub expr: Option<ExprId>,
    pub message: Option<ExprId>,
}
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct WhileStmt {
    pub cond: Option<ExprId>,
    /// Always a `SuiteStmt` (or `None`).
    pub suite: Option<StmtId>,
    pub else_suite: Option<StmtId>,
    /// Set if the loop emulates goto (`while gotoVar: ...`).
    pub goto_var: String,
}
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ForStmt {
    pub var: Option<ExprId>,
    pub iter: Option<ExprId>,
    pub suite: Option<StmtId>,
    pub else_suite: Option<StmtId>,
    pub decorator: Option<ExprId>,
    pub omp_args: Vec<CallArg>,
    pub async_: bool,
    /// Indicates if iter was wrapped with `__iter__()` call.
    pub wrapped: bool,
    /// True if there are no break/continue within the loop.
    pub flat: bool,
    /// Comprehension/generator clause (`[e for x in it]`), not a `for` statement.
    #[serde(default)]
    pub comprehension: bool,
}
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct IfStmt {
    pub cond: Option<ExprId>,
    pub if_suite: Option<StmtId>,
    pub else_suite: Option<StmtId>,
}

/// `ast::MatchCase`.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct MatchCase {
    pub pattern: Option<ExprId>,
    pub guard: Option<ExprId>,
    pub suite: Option<StmtId>,
}
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct MatchStmt {
    pub expr: Option<ExprId>,
    pub items: Vec<MatchCase>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ImportStmt {
    pub from: Option<ExprId>,
    pub what: Option<ExprId>,
    pub as_: String,
    /// Number of dots in a relative import (e.g. 3 for `from ...foo`).
    pub dots: usize,
    /// Function argument types for C imports.
    pub args: Vec<Param>,
    /// Function return type for C imports.
    pub ret: Option<ExprId>,
    /// Set if this is a function C import (not variable import).
    pub is_function: bool,
}
impl Default for ImportStmt {
    fn default() -> Self {
        Self { from: None, what: None, as_: String::new(), dots: 0, args: vec![], ret: None, is_function: true }
    }
}
impl ImportStmt {
    pub fn is_cvar(&self) -> bool {
        !self.is_function
    }
}
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ExceptStmt {
    /// Empty if an except is unnamed.
    pub var: String,
    pub exc: Option<ExprId>,
    pub suite: Option<StmtId>,
}
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct TryStmt {
    /// `ExceptStmt` nodes.
    pub items: Vec<StmtId>,
    pub suite: Option<StmtId>,
    pub else_suite: Option<StmtId>,
    pub finally: Option<StmtId>,
}
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ThrowStmt {
    pub expr: Option<ExprId>,
    pub from: Option<ExprId>,
    pub transformed: bool,
}
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct GlobalStmt {
    pub var: String,
    pub non_local: bool,
}

/// `FunctionStmt::CallLayout`: argument structure computed once for the normalized
/// declaration (filled by the cache/typecheck lane).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CallLayout {
    pub positional_limit: i32,
    pub star: i32,
    pub kwstar: i32,
    pub variadic_penalty: i32,
    pub keywords: Vec<(String, i32)>,
    pub required_values: Vec<i32>,
    pub optional: Vec<bool>,
}

/// Build with [`super::AstArena::new_function`] (runs `parseParams`).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct FunctionStmt {
    pub name: String,
    pub items: Vec<Param>,
    /// `None` if return type is not specified.
    pub ret: Option<ExprId>,
    pub suite: Option<StmtId>,
    pub decorators: Vec<ExprId>,
    pub async_: bool,
    /// Cache of non-inferrable generics (see `AstArena::non_inferrable_generics`).
    #[serde(skip)]
    pub non_inferrable_generics: Option<Rc<HashSet<String>>>,
    /// Only canonical declarations have a layout; clones start without one.
    #[serde(skip)]
    pub call_layout: Option<Box<CallLayout>>,
}
impl FunctionStmt {
    /// `addParam` (asserts no layout yet; invalidates the generics cache).
    pub fn add_param(&mut self, p: Param) {
        typhon_common::seqassert!(self.call_layout.is_none(), "cannot add a parameter to a finalized function");
        self.items.push(p);
        self.non_inferrable_generics = None;
    }
    /// `getCallLayout`.
    pub fn call_layout(&self) -> &CallLayout {
        self.call_layout.as_deref().expect("function call layout was not initialized")
    }
    /// `getStarArgs`: index of `*args` (or `items.len()`).
    pub fn star_args(&self) -> usize {
        self.items.iter().position(|p| p.name.starts_with('*') && !p.name.starts_with("**")).unwrap_or(self.items.len())
    }
    /// `getKwStarArgs`: index of `**kwargs` (or `items.len()`).
    pub fn kw_star_args(&self) -> usize {
        self.items.iter().position(|p| p.name.starts_with("**")).unwrap_or(self.items.len())
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ClassStmt {
    pub name: String,
    pub items: Vec<Param>,
    pub suite: Option<StmtId>,
    pub decorators: Vec<ExprId>,
    pub base_classes: Vec<ExprId>,
}
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct YieldFromStmt {
    pub expr: Option<ExprId>,
}
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct WithStmt {
    pub items: Vec<ExprId>,
    /// Empty string if a corresponding item is unnamed.
    pub vars: Vec<String>,
    pub suite: Option<StmtId>,
    pub async_: bool,
}
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct DirectiveStmt {
    pub key: String,
    pub value: String,
}
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct AssignMemberStmt {
    pub lhs: Option<ExprId>,
    pub member: String,
    pub rhs: Option<ExprId>,
    pub type_expr: Option<ExprId>,
}
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct CommentStmt {
    pub comment: String,
}

macro_rules! stmt_kinds {
    ($($variant:ident($ty:ident) => $as:ident, $as_mut:ident;)*) => {
        /// Statement node kinds (one per the reference `*Stmt` class).
        #[derive(Clone, Debug, Serialize, Deserialize)]
        pub enum StmtKind { $($variant($ty)),* }

        /// Discriminant of [`StmtKind`].
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
        pub enum StmtTag { $($variant),* }

        impl StmtKind {
            pub fn tag(&self) -> StmtTag {
                match self { $(StmtKind::$variant(_) => StmtTag::$variant),* }
            }
            $(
                pub fn $as(&self) -> Option<&$ty> {
                    match self { StmtKind::$variant(x) => Some(x), _ => None }
                }
                pub fn $as_mut(&mut self) -> Option<&mut $ty> {
                    match self { StmtKind::$variant(x) => Some(x), _ => None }
                }
            )*
        }
        $(
            impl From<$ty> for StmtKind {
                fn from(x: $ty) -> Self { StmtKind::$variant(x) }
            }
        )*
    };
}

stmt_kinds! {
    Suite(SuiteStmt) => as_suite, as_suite_mut;
    Break(BreakStmt) => as_break, as_break_mut;
    Continue(ContinueStmt) => as_continue, as_continue_mut;
    Expr(ExprStmt) => as_expr, as_expr_mut;
    Assign(AssignStmt) => as_assign, as_assign_mut;
    Del(DelStmt) => as_del, as_del_mut;
    Print(PrintStmt) => as_print, as_print_mut;
    Return(ReturnStmt) => as_return, as_return_mut;
    Yield(YieldStmt) => as_yield, as_yield_mut;
    Assert(AssertStmt) => as_assert, as_assert_mut;
    While(WhileStmt) => as_while, as_while_mut;
    For(ForStmt) => as_for, as_for_mut;
    If(IfStmt) => as_if, as_if_mut;
    Match(MatchStmt) => as_match, as_match_mut;
    Import(ImportStmt) => as_import, as_import_mut;
    Except(ExceptStmt) => as_except, as_except_mut;
    Try(TryStmt) => as_try, as_try_mut;
    Throw(ThrowStmt) => as_throw, as_throw_mut;
    Global(GlobalStmt) => as_global, as_global_mut;
    Function(FunctionStmt) => as_function, as_function_mut;
    Class(ClassStmt) => as_class, as_class_mut;
    YieldFrom(YieldFromStmt) => as_yield_from, as_yield_from_mut;
    With(WithStmt) => as_with, as_with_mut;
    Directive(DirectiveStmt) => as_directive, as_directive_mut;
    AssignMember(AssignMemberStmt) => as_assign_member, as_assign_member_mut;
    Comment(CommentStmt) => as_comment, as_comment_mut;
}

impl StmtTag {
    /// original name (e.g. `"ForStmt"`).
    pub fn class_name(self) -> &'static str {
        use StmtTag::*;
        match self {
            Suite => "SuiteStmt",
            Break => "BreakStmt",
            Continue => "ContinueStmt",
            Expr => "ExprStmt",
            Assign => "AssignStmt",
            Del => "DelStmt",
            Print => "PrintStmt",
            Return => "ReturnStmt",
            Yield => "YieldStmt",
            Assert => "AssertStmt",
            While => "WhileStmt",
            For => "ForStmt",
            If => "IfStmt",
            Match => "MatchStmt",
            Import => "ImportStmt",
            Except => "ExceptStmt",
            Try => "TryStmt",
            Throw => "ThrowStmt",
            Global => "GlobalStmt",
            Function => "FunctionStmt",
            Class => "ClassStmt",
            YieldFrom => "YieldFromStmt",
            With => "WithStmt",
            Directive => "DirectiveStmt",
            AssignMember => "AssignMemberStmt",
            Comment => "CommentStmt",
        }
    }
}
