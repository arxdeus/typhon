//! Port of codon/parser/visitors/scoping/scoping.{h,cpp} (scoping-doc lane)
//!
//! `ScopingVisitor::apply` resolves name bindings (local/global/nonlocal, captures,
//! comprehension renames, conditional-binding domination) before type checking and
//! records them as `Attr::Bindings` (`BindingsAttr`) on the AST.
//!
//! PORT-NOTE: C++ copies the visitor for each `transform` call (sharing the context
//! through a `shared_ptr` and collecting errors per copy). Here one visitor per
//! context is used; the per-copy `SrcInfo` is saved/restored around every
//! `transform`, and errors go to one list (deduplicated like `ParserErrors::append`).

mod bind;
mod decl;
mod fstring;
mod visit;

use indexmap::{IndexMap, IndexSet};
use rustc_hash::{FxBuildHasher, FxHashMap};
use typhon_common::srcinfo::SrcInfo;
use typhon_common::{CompilerError, ErrorKind, ErrorMessage, ParserErrors, Result, emsg};

use crate::ast::{Attr, AttrValue, BindingsAttr, CaptureType, StmtKind};
use crate::cache::{Cache, MAX_ERRORS};
use crate::ids::{ExprId, StmtId};
use crate::typecheck::ctx::CtxRef;

pub use fstring::{ParseExprFn, set_parse_expr_hook};

type FxIndexMap<K, V> = IndexMap<K, V, FxBuildHasher>;
type FxIndexSet<K> = IndexSet<K, FxBuildHasher>;

/// C++ `ASTNode *` used as a binding site (either an expression or a statement).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Node {
    Expr(ExprId),
    Stmt(StmtId),
}

/// `Context::ScopeBlock`.
#[derive(Debug)]
pub(crate) struct ScopeBlock {
    pub id: i32,
    /// Associated SuiteStmt.
    pub suite: Option<StmtId>,
    /// Variables "seen" before their assignment within a loop.
    pub seen_vars: Option<FxIndexSet<String>>,
}

/// `Context::Item`.
#[derive(Clone, Debug)]
pub(crate) struct Item {
    pub src: SrcInfo,
    pub scope: Vec<i32>,
    pub binding: Option<Node>,
    pub ignore: bool,
    /// List of scopes where the identifier is accessible without `__used__` check.
    pub access_checked: Vec<Vec<i32>>,
}

impl Item {
    fn new(src: SrcInfo, scope: Vec<i32>, binding: Option<Node>, access_checked: Vec<Vec<i32>>) -> Self {
        Self { src, scope, binding, ignore: false, access_checked }
    }
}

/// `ScopingVisitor::Context`.
#[derive(Debug)]
pub(crate) struct Context {
    pub scope: Vec<ScopeBlock>,
    /// Name -> bindings (front = most recent; C++ `std::list`).
    pub map: FxIndexMap<String, Vec<Item>>,
    pub captures: FxIndexMap<String, CaptureType>,
    /// For functions.
    pub child_captures: FxIndexMap<String, CaptureType>,
    pub first_seen: std::collections::BTreeMap<String, SrcInfo>,
    pub class_deduce: (String, FxIndexSet<String>),
    pub root: Option<Node>,
    pub function_scope: Option<StmtId>,
    pub in_class: bool,
    pub renames: Vec<FxHashMap<String, String>>,
    pub temp_scope: bool,
    /// Time to track positions of assignments and references to them.
    pub time: i64,
}

impl Default for Context {
    fn default() -> Self {
        Self {
            scope: vec![],
            map: Default::default(),
            captures: Default::default(),
            child_captures: Default::default(),
            first_seen: Default::default(),
            class_deduce: Default::default(),
            root: None,
            function_scope: None,
            in_class: false,
            renames: vec![FxHashMap::default()],
            temp_scope: false,
            time: 0,
        }
    }
}

impl Context {
    fn get_scope(&self) -> Vec<i32> {
        self.scope.iter().map(|b| b.id).collect()
    }
}

/// Port of `ScopingVisitor`.
pub struct ScopingVisitor<'a> {
    pub cache: &'a mut Cache,
    pub(crate) ctx: Context,
    pub errors: ParserErrors,
    pub(crate) src: SrcInfo,
}

/// Install `ScopingVisitor::apply` as `Cache::hooks.scoping`.
pub fn install_hooks(cache: &mut Cache) {
    cache.hooks.scoping = Some(std::rc::Rc::new(|cache: &mut Cache, stmt: StmtId, ctx: Option<CtxRef>| {
        let mut shadows = FxHashMap::default();
        ScopingVisitor::apply(cache, stmt, ctx.is_some().then_some(&mut shadows), false)?;
        if let Some(ctx) = ctx {
            ctx.borrow_mut().global_shadows.extend(shadows);
        }
        Ok(())
    }));
}

impl<'a> ScopingVisitor<'a> {
    pub(crate) fn with_ctx(cache: &'a mut Cache, ctx: Context) -> Self {
        Self { cache, ctx, errors: ParserErrors::default(), src: SrcInfo::default() }
    }

    /// `ScopingVisitor::apply`.
    pub fn apply(
        cache: &mut Cache,
        s: StmtId,
        global_shadows: Option<&mut FxHashMap<String, usize>>,
        dominate_all: bool,
    ) -> Result<()> {
        let mut v = ScopingVisitor::with_ctx(cache, Context::default());
        v.ctx.scope.push(ScopeBlock { id: 0, suite: Some(s), seen_vars: None });
        if !v.transform_s(Some(s)) || v.has_errors() {
            return Err(CompilerError::Parser(v.errors));
        }
        v.process_child_captures();
        if dominate_all {
            let names: Vec<String> = v.ctx.map.keys().cloned().collect();
            for u in names {
                v.find_dominating_binding(&u, true);
            }
        }
        // Count number of shadowed names to know which names change or not later on
        if let Some(gs) = global_shadows {
            for (u, items) in &v.ctx.map {
                let i = items.iter().filter(|ii| !ii.ignore).count();
                if i > 1 {
                    gs.insert(u.clone(), i);
                }
            }
        }
        Ok(())
    }

    pub fn has_errors(&self) -> bool {
        !self.errors.is_empty()
    }
    pub fn can_continue(&self) -> bool {
        self.errors.errors.len() <= MAX_ERRORS
    }

    /// `addError(error::Error, SrcInfo, args...)`.
    pub(crate) fn add_error(&mut self, e: ErrorKind, src: &SrcInfo, args: &[&dyn std::fmt::Display]) {
        let msg = ErrorMessage::with_code(emsg(e, args), src.clone(), e as i32);
        self.errors.add_error(typhon_common::error::Backtrace { trace: vec![msg] });
    }
    /// `addError(llvm::Error)`.
    pub(crate) fn add_compiler_error(&mut self, e: CompilerError) {
        match e {
            CompilerError::Parser(p) => self.errors.append(&p),
            other => {
                let msg = ErrorMessage::new(other.to_string(), self.src.clone());
                self.errors.add_error(typhon_common::error::Backtrace { trace: vec![msg] });
            }
        }
    }

    // ------------------------------------------------------------ node helpers

    pub(crate) fn node_has(&self, n: Node, a: Attr) -> bool {
        match n {
            Node::Expr(e) => self.cache.ast[e].attrs.has(a),
            Node::Stmt(s) => self.cache.ast[s].attrs.has(a),
        }
    }
    pub(crate) fn node_attrs_mut(&mut self, n: Node) -> &mut crate::ast::Attrs {
        match n {
            Node::Expr(e) => &mut self.cache.ast[e].attrs,
            Node::Stmt(s) => &mut self.cache.ast[s].attrs,
        }
    }
    pub(crate) fn node_src(&self, n: Node) -> SrcInfo {
        match n {
            Node::Expr(e) => self.cache.ast[e].src.clone(),
            Node::Stmt(s) => self.cache.ast[s].src.clone(),
        }
    }
    fn node_bindings_mut(&mut self, n: Node) -> &mut BindingsAttr {
        let attrs = self.node_attrs_mut(n);
        if !attrs.has(Attr::Bindings) {
            attrs.set_value(Attr::Bindings, AttrValue::Bindings(BindingsAttr::default()));
        }
        attrs.get_bindings_mut().expect("bindings")
    }

    // -------------------------------------------------------------- transforms

    /// `transform(Expr *)`.
    pub(crate) fn transform_e(&mut self, e: Option<ExprId>) -> bool {
        if !self.can_continue() {
            return false;
        }
        if let Some(e) = e {
            let saved = std::mem::replace(&mut self.src, self.cache.ast[e].src.clone());
            crate::visitors::Visitor::visit_expr(self, e);
            self.src = saved;
            if !self.can_continue() {
                return false;
            }
        }
        true
    }

    /// `transform(Stmt *)`.
    pub(crate) fn transform_s(&mut self, s: Option<StmtId>) -> bool {
        if !self.can_continue() {
            return false;
        }
        if let Some(s) = s {
            let saved = std::mem::replace(&mut self.src, self.cache.ast[s].src.clone());
            self.ctx.time += 1;
            let t = self.ctx.time;
            self.cache.ast[s].attrs.set_int(Attr::ExprTime, t);
            crate::visitors::Visitor::visit_stmt(self, s);
            self.src = saved;
            if !self.can_continue() {
                return false;
            }
        }
        true
    }

    /// `ConditionalBlock` constructor.
    pub(crate) fn push_block(&mut self, s: Option<StmtId>, id: Option<i32>) {
        let id = id.unwrap_or_else(|| {
            let b = self.cache.block_count;
            self.cache.block_count += 1;
            b
        });
        self.ctx.scope.push(ScopeBlock { id, suite: s, seen_vars: None });
    }
    /// `ConditionalBlock` destructor.
    pub(crate) fn pop_block(&mut self) {
        typhon_common::seqassert!(
            !self.ctx.scope.is_empty() && (self.ctx.scope.last().unwrap().id == 0 || self.ctx.scope.len() > 1),
            "empty scope"
        );
        self.ctx.scope.pop();
    }

    pub(crate) fn transform_scope_e(&mut self, e: Option<ExprId>) -> bool {
        if e.is_none() {
            return true;
        }
        self.push_block(None, None);
        let r = self.transform_e(e);
        self.pop_block();
        r
    }
    pub(crate) fn transform_scope_s(&mut self, s: Option<StmtId>) -> bool {
        if s.is_none() {
            return true;
        }
        self.push_block(s, None);
        let r = self.transform_s(s);
        self.pop_block();
        r
    }

    // ------------------------------------------------------------ binding logic

    /// `processChildCaptures`.
    pub(crate) fn process_child_captures(&mut self) {
        let cc: Vec<(String, CaptureType)> = self.ctx.child_captures.iter().map(|(k, v)| (k.clone(), *v)).collect();
        for (n, t) in cc {
            if let Some(i) = self.ctx.map.get(&n)
                && let Some(Node::Stmt(b)) = i.last().and_then(|x| x.binding)
                && matches!(self.cache.ast[b].kind, StmtKind::Class(_))
            {
                continue;
            }
            if self.find_dominating_binding(&n, true).is_none() {
                self.ctx.captures.entry(n).or_insert(t); // propagate!
            }
        }
    }

    /// `switchToUpdate`.
    pub(crate) fn switch_to_update(&mut self, binding: Option<Node>, name: &str, got_used_var: bool) {
        let Some(b) = binding else { return };
        if let Some(bs) = self.node_attrs_mut(b).get_bindings_mut() {
            bs.bindings.remove(name);
        }
        let attrs = self.node_attrs_mut(b);
        if !got_used_var && attrs.has(Attr::ExprDominatedUsed) {
            attrs.erase(Attr::ExprDominatedUsed);
        }
        attrs.set(if got_used_var { Attr::ExprDominatedUsed } else { Attr::ExprDominated });
        if let Node::Stmt(s) = b
            && matches!(self.cache.ast[s].kind, StmtKind::Function(_) | StmtKind::Class(_))
        {
            let src = self.node_src(b);
            self.add_error(ErrorKind::ID_INVALID_BIND, &src, &[&name]);
        }
    }
}
