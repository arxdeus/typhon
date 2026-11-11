//! Port of codon/parser/visitors/visitor.{h,cpp}: `ASTVisitor`, `CallbackASTVisitor` and
//! `ReplacingCallbackASTVisitor`.
//!
//! PORT-NOTE: the three C++ classes collapse into one trait. Default `visit_*` methods walk
//! children (in C++ order) through `transform_expr`/`transform_stmt` and write the returned
//! ids back, so:
//! * a plain callback visitor overrides `visit_*` and leaves `transform_*` as identity-walk;
//! * a replacing visitor overrides `transform_*` to return new ids;
//! * an `ASTVisitor` with no-op defaults overrides `default_visit_expr/stmt`... or the
//!   specific `visit_*` methods.

use crate::ast::children::{Child, ChildCollector, ChildReplacer};
use crate::ast::{AstArena, ExprTag, StmtTag};
use crate::ids::{ExprId, StmtId};

/// Access to the arena owned by a visitor's context (e.g. `TypeContext` / `Cache`).
pub trait AstCtx {
    fn ast(&self) -> &AstArena;
    fn ast_mut(&mut self) -> &mut AstArena;
}

macro_rules! visitor_trait {
    (
        exprs { $($ev:ident => $et:ident),* $(,)? }
        stmts { $($sv:ident => $st:ident),* $(,)? }
    ) => {
        /// AST visitor (one method per node kind).
        pub trait Visitor: AstCtx {
            /// `CallbackASTVisitor::transform(Expr*)`: visit and return the (possibly new) node.
            fn transform_expr(&mut self, id: ExprId) -> ExprId {
                self.visit_expr(id);
                id
            }
            /// `CallbackASTVisitor::transform(Stmt*)`.
            fn transform_stmt(&mut self, id: StmtId) -> StmtId {
                self.visit_stmt(id);
                id
            }
            /// `accept`: dispatch on the node kind.
            fn visit_expr(&mut self, id: ExprId) {
                match self.ast()[id].kind.tag() {
                    $(ExprTag::$et => self.$ev(id)),*
                }
            }
            /// `accept`: dispatch on the node kind.
            fn visit_stmt(&mut self, id: StmtId) {
                match self.ast()[id].kind.tag() {
                    $(StmtTag::$st => self.$sv(id)),*
                }
            }
            /// `ASTVisitor::defaultVisit(Expr*)` (default: walk children).
            fn default_visit_expr(&mut self, id: ExprId) {
                walk_expr(self, id)
            }
            /// `ASTVisitor::defaultVisit(Stmt*)` (default: walk children).
            fn default_visit_stmt(&mut self, id: StmtId) {
                walk_stmt(self, id)
            }
            $(
                fn $ev(&mut self, id: ExprId) { self.default_visit_expr(id) }
            )*
            $(
                fn $sv(&mut self, id: StmtId) { self.default_visit_stmt(id) }
            )*
        }
    };
}

visitor_trait! {
    exprs {
        visit_none => None, visit_bool => Bool, visit_int => Int, visit_float => Float,
        visit_string => String, visit_id => Id, visit_star => Star,
        visit_keyword_star => KeywordStar, visit_tuple => Tuple, visit_list => List,
        visit_set => Set, visit_dict => Dict, visit_generator => Generator,
        visit_if_expr => If, visit_unary => Unary, visit_binary => Binary,
        visit_chain_binary => ChainBinary, visit_pipe => Pipe, visit_index => Index,
        visit_call => Call, visit_dot => Dot, visit_slice => Slice,
        visit_ellipsis => Ellipsis, visit_lambda => Lambda, visit_yield_expr => Yield,
        visit_await => Await, visit_assign_expr => Assign, visit_range => Range,
        visit_stmt_expr => Stmt, visit_instantiate => Instantiate,
    }
    stmts {
        visit_suite => Suite, visit_break => Break, visit_continue => Continue,
        visit_expr_stmt => Expr, visit_assign => Assign, visit_del => Del,
        visit_print => Print, visit_return => Return, visit_yield => Yield,
        visit_assert => Assert, visit_while => While, visit_for => For, visit_if => If,
        visit_match => Match, visit_import => Import, visit_except => Except,
        visit_try => Try, visit_throw => Throw, visit_global => Global,
        visit_function => Function, visit_class => Class, visit_yield_from => YieldFrom,
        visit_with => With, visit_custom => Custom, visit_directive => Directive,
        visit_assign_member => AssignMember, visit_comment => Comment,
    }
}

/// Children of an expression in C++ visitor order (f-string parts excluded).
pub fn expr_children(a: &AstArena, id: ExprId) -> Vec<Child> {
    let mut c = ChildCollector { children: vec![], include_string_parts: false };
    a[id].kind.clone().map_ids(&mut c);
    c.children
}

/// Children of a statement in C++ visitor order.
pub fn stmt_children(a: &AstArena, id: StmtId) -> Vec<Child> {
    let mut c = ChildCollector { children: vec![], include_string_parts: false };
    a[id].kind.clone().map_ids(&mut c);
    c.children
}

fn transform_children<V: Visitor + ?Sized>(v: &mut V, children: Vec<Child>) -> (Vec<Child>, bool) {
    let mut changed = false;
    let out = children
        .into_iter()
        .map(|c| {
            let n = match c {
                Child::Expr(e) => Child::Expr(v.transform_expr(e)),
                Child::Stmt(s) => Child::Stmt(v.transform_stmt(s)),
                Child::Suite(s) => {
                    let t = v.transform_stmt(s);
                    // ReplacingCallbackASTVisitor: `SuiteStmt::wrap(transform(suite))`.
                    Child::Suite(v.ast_mut().wrap_suite(Some(t)).unwrap_or(t))
                }
            };
            changed |= n != c;
            n
        })
        .collect();
    (out, changed)
}

/// Default traversal of an expression's children; replacements are written back.
pub fn walk_expr<V: Visitor + ?Sized>(v: &mut V, id: ExprId) {
    let ch = expr_children(v.ast(), id);
    if ch.is_empty() {
        return;
    }
    let (new, changed) = transform_children(v, ch);
    if changed {
        let a = v.ast_mut();
        let mut r = ChildReplacer { children: &new, pos: 0, include_string_parts: false };
        a[id].kind.map_ids(&mut r);
    }
}

/// Default traversal of a statement's children; replacements are written back.
pub fn walk_stmt<V: Visitor + ?Sized>(v: &mut V, id: StmtId) {
    let ch = stmt_children(v.ast(), id);
    if ch.is_empty() {
        return;
    }
    let (new, changed) = transform_children(v, ch);
    if changed {
        let a = v.ast_mut();
        let mut r = ChildReplacer { children: &new, pos: 0, include_string_parts: false };
        a[id].kind.map_ids(&mut r);
    }
}

/// Simple context owning an arena (handy for tests and standalone visitors).
impl AstCtx for AstArena {
    fn ast(&self) -> &AstArena {
        self
    }
    fn ast_mut(&mut self) -> &mut AstArena {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::*;
    use typhon_common::SrcInfo;

    struct Renamer<'a> {
        a: &'a mut AstArena,
        seen: Vec<String>,
    }
    impl AstCtx for Renamer<'_> {
        fn ast(&self) -> &AstArena {
            self.a
        }
        fn ast_mut(&mut self) -> &mut AstArena {
            self.a
        }
    }
    impl Visitor for Renamer<'_> {
        fn transform_expr(&mut self, id: ExprId) -> ExprId {
            if let Some(i) = self.a[id].kind.as_id() {
                self.seen.push(i.value.clone());
                if i.value == "x" {
                    return self.a.new_expr(IdExpr::new("y"), SrcInfo::default());
                }
            }
            self.visit_expr(id);
            id
        }
    }

    #[test]
    fn replacing_visitor() {
        let mut a = AstArena::new();
        let s = SrcInfo::default();
        let x = a.new_expr(IdExpr::new("x"), s);
        let one = a.new_expr(IntExpr::new(1), s);
        let b = a.new_expr(BinaryExpr { op: "+".into(), lexpr: Some(x), rexpr: Some(one), in_place: false }, s);
        let es = a.new_stmt(ExprStmt { expr: Some(b) }, s);
        let ret = a.new_stmt(ReturnStmt { expr: None }, s);
        let w = a.new_stmt(WhileStmt { cond: Some(x), suite: Some(ret), else_suite: None, goto_var: String::new() }, s);
        let suite = a.new_stmt(SuiteStmt { items: vec![es, w] }, s);
        let mut v = Renamer { a: &mut a, seen: vec![] };
        v.transform_stmt(suite);
        assert_eq!(v.seen, vec!["x", "x"]);
        assert_eq!(a.expr_to_string(b, -1), "(binary \"+\" 'y (int 1))");
        assert_eq!(a.stmt_to_string(w, -1), "(while 'y)");
    }
}
