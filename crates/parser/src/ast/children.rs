//! Generic child traversal of AST nodes (the field order of C++
//! `CallbackASTVisitor::visit` / `ReplacingCallbackASTVisitor::visit`).

use super::expr::ExprKind;
use super::stmt::StmtKind;
use crate::ids::{ExprId, StmtId};

/// Maps every child id of a node (in C++ visitor order) to a new id.
pub trait IdMapper {
    fn expr(&mut self, id: ExprId) -> ExprId;
    fn stmt(&mut self, id: StmtId) -> StmtId;
    /// Slots that C++ keeps as `SuiteStmt *` (re-wrapped by the replacing visitor).
    fn suite(&mut self, id: StmtId) -> StmtId {
        self.stmt(id)
    }
    /// Whether `StringExpr` f-string part expressions are children. C++ visitors skip them
    /// (`visit(StringExpr*) {}`); deep clone includes them.
    fn string_parts(&self) -> bool {
        false
    }
}

/// One child slot, used by visitors to replay traversal.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Child {
    Expr(ExprId),
    Stmt(StmtId),
    Suite(StmtId),
}

fn oe(m: &mut dyn IdMapper, e: &mut Option<ExprId>) {
    if let Some(x) = e {
        *x = m.expr(*x);
    }
}
fn os(m: &mut dyn IdMapper, s: &mut Option<StmtId>) {
    if let Some(x) = s {
        *x = m.stmt(*x);
    }
}
fn osu(m: &mut dyn IdMapper, s: &mut Option<StmtId>) {
    if let Some(x) = s {
        *x = m.suite(*x);
    }
}
fn ve(m: &mut dyn IdMapper, v: &mut [ExprId]) {
    for x in v {
        *x = m.expr(*x);
    }
}
fn params(m: &mut dyn IdMapper, ps: &mut [super::expr::Param]) {
    for p in ps {
        oe(m, &mut p.ty);
        oe(m, &mut p.default_value);
    }
}

impl ExprKind {
    /// Map child ids in place (C++ visitor order).
    pub fn map_ids(&mut self, m: &mut dyn IdMapper) {
        use ExprKind::*;
        match self {
            None(_) | Bool(_) | Int(_) | Float(_) | Id(_) | Ellipsis(_) | Yield(_) => {}
            String(s) => {
                if m.string_parts() {
                    for p in &mut s.strings {
                        oe(m, &mut p.expr);
                    }
                }
            }
            Star(e) => oe(m, &mut e.expr),
            KeywordStar(e) => oe(m, &mut e.expr),
            Tuple(e) => ve(m, &mut e.items),
            List(e) => ve(m, &mut e.items),
            Set(e) => ve(m, &mut e.items),
            Dict(e) => ve(m, &mut e.items),
            Generator(e) => os(m, &mut e.loops),
            If(e) => {
                oe(m, &mut e.cond);
                oe(m, &mut e.ifexpr);
                oe(m, &mut e.elsexpr);
            }
            Unary(e) => oe(m, &mut e.expr),
            Binary(e) => {
                oe(m, &mut e.lexpr);
                oe(m, &mut e.rexpr);
            }
            ChainBinary(e) => {
                for (_, x) in &mut e.exprs {
                    *x = m.expr(*x);
                }
            }
            Pipe(e) => {
                for p in &mut e.items {
                    oe(m, &mut p.expr);
                }
            }
            Index(e) => {
                oe(m, &mut e.expr);
                oe(m, &mut e.index);
            }
            Call(e) => {
                oe(m, &mut e.expr);
                for a in &mut e.items {
                    oe(m, &mut a.value);
                }
            }
            Dot(e) => oe(m, &mut e.expr),
            Slice(e) => {
                oe(m, &mut e.start);
                oe(m, &mut e.stop);
                oe(m, &mut e.step);
            }
            Lambda(e) => {
                params(m, &mut e.items);
                oe(m, &mut e.expr);
            }
            Await(e) => oe(m, &mut e.expr),
            Assign(e) => {
                oe(m, &mut e.var);
                oe(m, &mut e.expr);
            }
            Range(e) => {
                oe(m, &mut e.start);
                oe(m, &mut e.stop);
            }
            Instantiate(e) => {
                oe(m, &mut e.expr);
                ve(m, &mut e.items);
            }
            Stmt(e) => {
                for s in &mut e.items {
                    *s = m.stmt(*s);
                }
                oe(m, &mut e.expr);
            }
        }
    }
}

impl StmtKind {
    /// Map child ids in place (C++ visitor order).
    pub fn map_ids(&mut self, m: &mut dyn IdMapper) {
        use StmtKind::*;
        match self {
            Break(_) | Continue(_) | Global(_) | Directive(_) | Comment(_) => {}
            Suite(s) => {
                for x in &mut s.items {
                    *x = m.stmt(*x);
                }
            }
            Expr(s) => oe(m, &mut s.expr),
            Assign(s) => {
                oe(m, &mut s.lhs);
                oe(m, &mut s.rhs);
                oe(m, &mut s.type_expr);
            }
            AssignMember(s) => {
                oe(m, &mut s.lhs);
                oe(m, &mut s.rhs);
                oe(m, &mut s.type_expr);
            }
            Del(s) => oe(m, &mut s.expr),
            Print(s) => ve(m, &mut s.items),
            Return(s) => oe(m, &mut s.expr),
            Yield(s) => oe(m, &mut s.expr),
            YieldFrom(s) => oe(m, &mut s.expr),
            Assert(s) => {
                oe(m, &mut s.expr);
                oe(m, &mut s.message);
            }
            While(s) => {
                oe(m, &mut s.cond);
                osu(m, &mut s.suite);
                osu(m, &mut s.else_suite);
            }
            For(s) => {
                oe(m, &mut s.var);
                oe(m, &mut s.iter);
                osu(m, &mut s.suite);
                osu(m, &mut s.else_suite);
                oe(m, &mut s.decorator);
                for a in &mut s.omp_args {
                    oe(m, &mut a.value);
                }
            }
            If(s) => {
                oe(m, &mut s.cond);
                osu(m, &mut s.if_suite);
                osu(m, &mut s.else_suite);
            }
            Match(s) => {
                oe(m, &mut s.expr);
                for c in &mut s.items {
                    oe(m, &mut c.pattern);
                    oe(m, &mut c.guard);
                    osu(m, &mut c.suite);
                }
            }
            Import(s) => {
                oe(m, &mut s.from);
                oe(m, &mut s.what);
                params(m, &mut s.args);
                oe(m, &mut s.ret);
            }
            Try(s) => {
                osu(m, &mut s.suite);
                for x in &mut s.items {
                    *x = m.stmt(*x);
                }
                osu(m, &mut s.else_suite);
                osu(m, &mut s.finally);
            }
            Except(s) => {
                oe(m, &mut s.exc);
                osu(m, &mut s.suite);
            }
            Throw(s) => {
                oe(m, &mut s.expr);
                oe(m, &mut s.from);
            }
            Function(s) => {
                oe(m, &mut s.ret);
                params(m, &mut s.items);
                osu(m, &mut s.suite);
                ve(m, &mut s.decorators);
            }
            Class(s) => {
                params(m, &mut s.items);
                osu(m, &mut s.suite);
                ve(m, &mut s.decorators);
                ve(m, &mut s.base_classes);
            }
            With(s) => {
                ve(m, &mut s.items);
                osu(m, &mut s.suite);
            }
            Custom(s) => {
                oe(m, &mut s.expr);
                osu(m, &mut s.suite);
            }
        }
    }
}

/// Collects children without changing them.
pub struct ChildCollector {
    pub children: Vec<Child>,
    pub include_string_parts: bool,
}
impl IdMapper for ChildCollector {
    fn expr(&mut self, id: ExprId) -> ExprId {
        self.children.push(Child::Expr(id));
        id
    }
    fn stmt(&mut self, id: StmtId) -> StmtId {
        self.children.push(Child::Stmt(id));
        id
    }
    fn suite(&mut self, id: StmtId) -> StmtId {
        self.children.push(Child::Suite(id));
        id
    }
    fn string_parts(&self) -> bool {
        self.include_string_parts
    }
}

/// Replays a list of replacement ids in traversal order.
pub struct ChildReplacer<'a> {
    pub children: &'a [Child],
    pub pos: usize,
    pub include_string_parts: bool,
}
impl IdMapper for ChildReplacer<'_> {
    fn expr(&mut self, id: ExprId) -> ExprId {
        let r = match self.children.get(self.pos) {
            Some(Child::Expr(e)) => *e,
            _ => id,
        };
        self.pos += 1;
        r
    }
    fn stmt(&mut self, id: StmtId) -> StmtId {
        let r = match self.children.get(self.pos) {
            Some(Child::Stmt(s)) | Some(Child::Suite(s)) => *s,
            _ => id,
        };
        self.pos += 1;
        r
    }
    fn string_parts(&self) -> bool {
        self.include_string_parts
    }
}
