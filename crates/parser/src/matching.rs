//! Port of codon/parser/match.{h,cpp}: AST pattern matching helpers
//! (`M<T>(...)`, `MOr`, `MVar`, `MCall`, `M_`, `MAny`, `MStarts`, `MEnds`, `MContains`).
//!
//! PORT-NOTE: C++ builds patterns with variadic templates checked against each node's
//! `match_members()` tuple. Rust uses a runtime pattern tree; the member list of each node
//! kind is produced by [`expr_members`] / [`stmt_members`] in the same order as C++.

use std::cell::Cell;
use std::rc::Rc;

use crate::ast::{AstArena, ExprKind, ExprTag, StmtKind, StmtTag};
use crate::ids::{ExprId, StmtId};

/// A matchable value (one entry of `match_members()`).
#[derive(Clone, Debug)]
pub enum Val {
    Expr(Option<ExprId>),
    Stmt(Option<StmtId>),
    Str(String),
    Int(i64),
    Float(f64),
    Bool(bool),
    /// Members that patterns can only ignore (vectors, enums, ...).
    Opaque,
}

type ExprCb = Rc<dyn Fn(ExprId)>;
type StmtCb = Rc<dyn Fn(StmtId)>;

/// A pattern.
#[derive(Clone)]
pub enum M {
    /// `M_`: matches anything.
    Ignore,
    /// `MAny()`: matches the rest of the members.
    Any,
    Str(String),
    Int(i64),
    Float(f64),
    Bool(bool),
    StartsWith(String),
    EndsWith(String),
    Contains(String),
    Or(Vec<M>),
    /// `M<XExpr>(args...)` / `MCall` / `MVar`.
    Expr { tag: ExprTag, args: Vec<M>, cb: Option<ExprCb> },
    /// `M<XStmt>(args...)`.
    Stmt { tag: StmtTag, args: Vec<M>, cb: Option<StmtCb> },
}

impl M {
    pub fn ignore() -> M {
        M::Ignore
    }
    pub fn any() -> M {
        M::Any
    }
    pub fn str(s: impl Into<String>) -> M {
        M::Str(s.into())
    }
    pub fn starts(s: impl Into<String>) -> M {
        M::StartsWith(s.into())
    }
    pub fn ends(s: impl Into<String>) -> M {
        M::EndsWith(s.into())
    }
    pub fn contains(s: impl Into<String>) -> M {
        M::Contains(s.into())
    }
    pub fn or(v: Vec<M>) -> M {
        M::Or(v)
    }
    pub fn expr(tag: ExprTag, args: Vec<M>) -> M {
        M::Expr { tag, args, cb: None }
    }
    pub fn stmt(tag: StmtTag, args: Vec<M>) -> M {
        M::Stmt { tag, args, cb: None }
    }
    /// `MVar<T>(args..., var)`: store the matched node in `var`.
    pub fn bind_expr(self, var: &Rc<Cell<Option<ExprId>>>) -> M {
        let v = var.clone();
        self.call_expr(move |e| v.set(Some(e)))
    }
    pub fn bind_stmt(self, var: &Rc<Cell<Option<StmtId>>>) -> M {
        let v = var.clone();
        self.call_stmt(move |s| v.set(Some(s)))
    }
    /// `MCall<T>(args..., fn)`.
    pub fn call_expr(self, f: impl Fn(ExprId) + 'static) -> M {
        match self {
            M::Expr { tag, args, .. } => M::Expr { tag, args, cb: Some(Rc::new(f)) },
            m => m,
        }
    }
    pub fn call_stmt(self, f: impl Fn(StmtId) + 'static) -> M {
        match self {
            M::Stmt { tag, args, .. } => M::Stmt { tag, args, cb: Some(Rc::new(f)) },
            m => m,
        }
    }
}

/// `match_members()` of an expression node.
pub fn expr_members(a: &AstArena, id: ExprId) -> Vec<Val> {
    use Val::*;
    let s = |x: &String| Str(x.clone());
    match &a[id].kind {
        ExprKind::None(_) | ExprKind::Yield(_) => vec![],
        ExprKind::Bool(b) => vec![Bool(b.value)],
        ExprKind::Int(i) => vec![s(&i.value), s(&i.suffix), i.int_value.map(Int).unwrap_or(Opaque)],
        ExprKind::Float(i) => vec![s(&i.value), s(&i.suffix), i.float_value.map(Float).unwrap_or(Opaque)],
        // C++ matches StringExpr against its `strings` vector; we expose the simple value so
        // `M<StringExpr>("x")` is usable.
        ExprKind::String(x) => vec![if x.is_simple() { Str(x.strings[0].value.clone()) } else { Opaque }],
        ExprKind::Id(i) => vec![s(&i.value)],
        ExprKind::Star(e) => vec![Expr(e.expr)],
        ExprKind::KeywordStar(e) => vec![Expr(e.expr)],
        ExprKind::Tuple(_) | ExprKind::List(_) | ExprKind::Set(_) | ExprKind::Dict(_) => vec![Opaque],
        ExprKind::Generator(g) => vec![Opaque, Stmt(g.loops)],
        ExprKind::If(e) => vec![Expr(e.cond), Expr(e.ifexpr), Expr(e.elsexpr)],
        ExprKind::Unary(e) => vec![s(&e.op), Expr(e.expr)],
        ExprKind::Binary(e) => vec![s(&e.op), Expr(e.lexpr), Expr(e.rexpr)],
        ExprKind::ChainBinary(_) | ExprKind::Pipe(_) => vec![Opaque],
        ExprKind::Index(e) => vec![Expr(e.expr), Expr(e.index)],
        ExprKind::Call(e) => vec![Expr(e.expr), Opaque, Bool(e.ordered), Bool(e.partial)],
        ExprKind::Dot(e) => vec![Expr(e.expr), s(&e.member)],
        ExprKind::Slice(e) => vec![Expr(e.start), Expr(e.stop), Expr(e.step)],
        ExprKind::Ellipsis(_) => vec![Opaque],
        ExprKind::Lambda(e) => vec![Expr(e.expr), Opaque],
        ExprKind::Await(e) => vec![Expr(e.expr)],
        ExprKind::Assign(e) => vec![Expr(e.var), Expr(e.expr)],
        ExprKind::Range(e) => vec![Expr(e.start), Expr(e.stop)],
        ExprKind::Stmt(e) => vec![Expr(e.expr), Opaque],
        ExprKind::Instantiate(e) => vec![Expr(e.expr), Opaque],
    }
}

/// `match_members()` of a statement node.
pub fn stmt_members(a: &AstArena, id: StmtId) -> Vec<Val> {
    use Val::*;
    let s = |x: &String| Str(x.clone());
    match &a[id].kind {
        StmtKind::Suite(_) => vec![Opaque],
        StmtKind::Break(_) | StmtKind::Continue(_) => vec![],
        StmtKind::Expr(x) => vec![Expr(x.expr)],
        StmtKind::Assign(x) => vec![Expr(x.lhs), Expr(x.rhs), Expr(x.type_expr), Opaque],
        StmtKind::Del(x) => vec![Expr(x.expr)],
        StmtKind::Print(x) => vec![Opaque, Bool(x.no_newline)],
        StmtKind::Return(x) => vec![Expr(x.expr)],
        StmtKind::Yield(x) => vec![Expr(x.expr)],
        StmtKind::Assert(x) => vec![Expr(x.expr), Expr(x.message)],
        StmtKind::While(x) => vec![Expr(x.cond), Stmt(x.suite), Stmt(x.else_suite), s(&x.goto_var)],
        StmtKind::For(x) => vec![
            Expr(x.var),
            Expr(x.iter),
            Stmt(x.suite),
            Stmt(x.else_suite),
            Expr(x.decorator),
            Opaque,
            Bool(x.async_),
            Bool(x.wrapped),
            Bool(x.flat),
        ],
        StmtKind::If(x) => vec![Expr(x.cond), Stmt(x.if_suite), Stmt(x.else_suite)],
        StmtKind::Match(x) => vec![Opaque, Expr(x.expr)],
        StmtKind::Import(x) => {
            vec![Expr(x.from), Expr(x.what), s(&x.as_), Int(x.dots as i64), Opaque, Expr(x.ret), Bool(x.is_function)]
        }
        StmtKind::Except(x) => vec![s(&x.var), Expr(x.exc), Stmt(x.suite)],
        StmtKind::Try(x) => vec![Opaque, Stmt(x.suite), Stmt(x.else_suite), Stmt(x.finally)],
        StmtKind::Throw(x) => vec![Expr(x.expr), Expr(x.from), Bool(x.transformed)],
        StmtKind::Global(x) => vec![s(&x.var), Bool(x.non_local)],
        StmtKind::Function(x) => vec![s(&x.name), Opaque, Expr(x.ret), Stmt(x.suite), Opaque, Bool(x.async_)],
        StmtKind::Class(x) => vec![s(&x.name), Stmt(x.suite), Opaque, Opaque, Opaque],
        StmtKind::YieldFrom(x) => vec![Expr(x.expr)],
        StmtKind::With(x) => vec![Opaque, Opaque, Stmt(x.suite)],
        StmtKind::Custom(x) => vec![s(&x.keyword), Expr(x.expr), Stmt(x.suite)],
        StmtKind::Directive(x) => vec![s(&x.key), s(&x.value)],
        StmtKind::AssignMember(x) => vec![Expr(x.lhs), s(&x.member), Expr(x.rhs), Expr(x.type_expr)],
        StmtKind::Comment(x) => vec![s(&x.comment)],
    }
}

fn match_members(a: &AstArena, members: &[Val], args: &[M]) -> bool {
    for (i, m) in args.iter().enumerate() {
        if matches!(m, M::Any) {
            return true;
        }
        let Some(v) = members.get(i) else { return false };
        if !match_val(a, v, m) {
            return false;
        }
    }
    // C++: all members must be covered unless no pattern args are given.
    args.is_empty() || args.len() == members.len()
}

/// Match a single member value against a pattern.
pub fn match_val(a: &AstArena, v: &Val, m: &M) -> bool {
    match (m, v) {
        (M::Ignore, _) | (M::Any, _) => true,
        (M::Or(alts), _) => alts.iter().any(|alt| match_val(a, v, alt)),
        (M::Str(p), Val::Str(s)) => p == s,
        (M::StartsWith(p), Val::Str(s)) => s.starts_with(p.as_str()),
        (M::EndsWith(p), Val::Str(s)) => s.ends_with(p.as_str()),
        (M::Contains(p), Val::Str(s)) => s.contains(p.as_str()),
        (M::Int(p), Val::Int(s)) => p == s,
        (M::Bool(p), Val::Bool(s)) => p == s,
        (M::Float(p), Val::Float(s)) => (p - s).abs() < f32::EPSILON as f64,
        (M::Expr { .. }, Val::Expr(e)) => match_expr(a, *e, m),
        (M::Stmt { .. }, Val::Stmt(s)) => match_stmt(a, *s, m),
        _ => false,
    }
}

/// `match(Expr*, pattern)`.
pub fn match_expr(a: &AstArena, e: Option<ExprId>, m: &M) -> bool {
    match m {
        M::Ignore | M::Any => true,
        M::Or(alts) => alts.iter().any(|alt| match_expr(a, e, alt)),
        M::Expr { tag, args, cb } => {
            let Some(e) = e else { return false };
            if a[e].kind.tag() != *tag {
                return false;
            }
            let r = args.is_empty() || match_members(a, &expr_members(a, e), args);
            if r && let Some(cb) = cb {
                cb(e);
            }
            r
        }
        _ => false,
    }
}

/// `match(Stmt*, pattern)`.
pub fn match_stmt(a: &AstArena, s: Option<StmtId>, m: &M) -> bool {
    match m {
        M::Ignore | M::Any => true,
        M::Or(alts) => alts.iter().any(|alt| match_stmt(a, s, alt)),
        M::Stmt { tag, args, cb } => {
            let Some(s) = s else { return false };
            if a[s].kind.tag() != *tag {
                return false;
            }
            let r = args.is_empty() || match_members(a, &stmt_members(a, s), args);
            if r && let Some(cb) = cb {
                cb(s);
            }
            r
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::*;
    use typhon_common::SrcInfo;

    #[test]
    fn matching() {
        let mut a = AstArena::new();
        let s = SrcInfo::default();
        let st = a.new_expr(IdExpr::new("Static"), s);
        let int = a.new_expr(IdExpr::new("int"), s);
        let ix = a.new_expr(IndexExpr { expr: Some(st), index: Some(int) }, s);
        let var = Rc::new(Cell::new(None));
        let p = M::expr(
            ExprTag::Index,
            vec![
                M::expr(ExprTag::Id, vec![M::or(vec![M::str("Static"), M::str("Literal")])]),
                M::expr(ExprTag::Id, vec![]).bind_expr(&var),
            ],
        );
        assert!(match_expr(&a, Some(ix), &p));
        assert_eq!(var.get(), Some(int));
        assert!(!match_expr(&a, Some(int), &p));
        assert!(match_expr(&a, Some(int), &M::expr(ExprTag::Id, vec![M::starts("in")])));
        let es = a.new_stmt(ExprStmt { expr: Some(int) }, s);
        assert!(match_stmt(&a, Some(es), &M::stmt(StmtTag::Expr, vec![M::expr(ExprTag::Id, vec![])])));
        assert!(!match_stmt(&a, Some(es), &M::stmt(StmtTag::Expr, vec![M::expr(ExprTag::String, vec![])])));
        assert_eq!(a.get_static_generic(Some(ix)), LiteralKind::Int);
    }
}
