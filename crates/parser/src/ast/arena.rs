//! The AST arena (C++ `Cache::N<T>` node storage) plus node logic from
//! codon/parser/ast/{expr,stmt}.cpp: constructors with normalization, deep clone,
//! suite helpers, generator helpers and FunctionStmt/ClassStmt helpers.

use std::collections::HashSet;
use std::ops::{Index, IndexMut};
use std::rc::Rc;

use la_arena::Arena;
use typhon_common::{ErrorKind, Result, SrcInfo};

use super::attr::Attr;
use super::children::IdMapper;
use super::expr::*;
use super::stmt::*;
use crate::ids::{ExprId, StmtId, TypeId};

/// `types::LiteralKind`.
// PORT-NOTE: defined here because the types lane is not ported yet; the types module
// should re-export (or replace) it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LiteralKind {
    Runtime,
    Int,
    String,
    Bool,
}

impl LiteralKind {
    /// `Type::literalFromString`.
    pub fn from_str(s: &str) -> Self {
        match s {
            "int" | "Int" => LiteralKind::Int,
            "str" => LiteralKind::String,
            "bool" => LiteralKind::Bool,
            _ => LiteralKind::Runtime,
        }
    }
}

/// Hook through which the AST prints type-checker types (implemented by the types lane).
pub trait TypeFormatter {
    /// `Type::debugString(mode)`.
    fn debug_string(&self, ty: TypeId, mode: i32) -> String;
    /// `Type::realizedName()`.
    fn realized_name(&self, ty: TypeId) -> String;
    /// `Type::getFunc() != nullptr`.
    fn is_func(&self, ty: TypeId) -> bool;
}

/// Owner of all AST nodes. C++ never frees individual nodes; neither does this arena.
#[derive(Default)]
pub struct AstArena {
    exprs: Arena<Expr>,
    stmts: Arena<Stmt>,
    /// Type printer used by `toString`/`toCodonString`/format (falls back to `T<n>`).
    pub type_fmt: Option<Rc<dyn TypeFormatter>>,
}

impl std::fmt::Debug for AstArena {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "AstArena {{ exprs: {}, stmts: {} }}", self.exprs.len(), self.stmts.len())
    }
}

impl Index<ExprId> for AstArena {
    type Output = Expr;
    fn index(&self, id: ExprId) -> &Expr {
        &self.exprs[id.0]
    }
}
impl IndexMut<ExprId> for AstArena {
    fn index_mut(&mut self, id: ExprId) -> &mut Expr {
        &mut self.exprs[id.0]
    }
}
impl Index<StmtId> for AstArena {
    type Output = Stmt;
    fn index(&self, id: StmtId) -> &Stmt {
        &self.stmts[id.0]
    }
}
impl IndexMut<StmtId> for AstArena {
    fn index_mut(&mut self, id: StmtId) -> &mut Stmt {
        &mut self.stmts[id.0]
    }
}

/// Location of the final statement of a generator (C++ `Stmt **getFinalStmt()`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GenSlot {
    /// `GeneratorExpr::loops` itself.
    Loops(ExprId),
    /// `ForStmt::suite` of the given for.
    ForSuite(StmtId),
    /// `IfStmt::if_suite` of the given if.
    IfSuite(StmtId),
    /// `items[idx]` of the given suite.
    SuiteItem(StmtId, usize),
}

/// `isId(e, s)`.
pub fn is_id(arena: &AstArena, e: Option<ExprId>, s: &str) -> bool {
    e.and_then(|e| arena[e].kind.as_id()).is_some_and(|i| i.value == s)
}

impl AstArena {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn expr(&self, id: ExprId) -> &Expr {
        &self[id]
    }
    pub fn expr_mut(&mut self, id: ExprId) -> &mut Expr {
        &mut self[id]
    }
    pub fn stmt(&self, id: StmtId) -> &Stmt {
        &self[id]
    }
    pub fn stmt_mut(&mut self, id: StmtId) -> &mut Stmt {
        &mut self[id]
    }
    pub fn expr_count(&self) -> usize {
        self.exprs.len()
    }
    pub fn stmt_count(&self) -> usize {
        self.stmts.len()
    }
    /// Iterate all expressions.
    pub fn exprs(&self) -> impl Iterator<Item = (ExprId, &Expr)> {
        self.exprs.iter().map(|(i, e)| (ExprId(i), e))
    }
    /// Iterate all statements.
    pub fn stmts(&self) -> impl Iterator<Item = (StmtId, &Stmt)> {
        self.stmts.iter().map(|(i, s)| (StmtId(i), s))
    }

    /// Allocate a fully built expression node without normalization.
    pub fn alloc_expr(&mut self, e: Expr) -> ExprId {
        ExprId(self.exprs.alloc(e))
    }
    /// Allocate a fully built statement node without normalization.
    pub fn alloc_stmt(&mut self, s: Stmt) -> StmtId {
        StmtId(self.stmts.alloc(s))
    }

    /// `cache->NS<T>(src, args...)` for expressions. Applies C++ constructor side effects
    /// (PipeExpr marks call ellipses as `Pipe`; DictExpr item validation).
    pub fn new_expr(&mut self, kind: impl Into<ExprKind>, src: SrcInfo) -> ExprId {
        let mut kind = kind.into();
        match &mut kind {
            ExprKind::Pipe(p) => {
                let calls: Vec<ExprId> = p.items.iter().filter_map(|i| i.expr).collect();
                for c in calls {
                    let args: Vec<ExprId> = match self[c].kind.as_call() {
                        Some(call) => call.items.iter().filter_map(|a| a.value).collect(),
                        None => continue,
                    };
                    for a in args {
                        if let Some(el) = self[a].kind.as_ellipsis_mut() {
                            el.mode = EllipsisType::Pipe;
                        }
                    }
                }
            }
            ExprKind::Dict(d) => {
                for &i in &d.items {
                    let ok = self[i].kind.as_tuple().is_some_and(|t| t.items.len() == 2)
                        || self[i].kind.as_keyword_star().is_some();
                    typhon_common::seqassert!(ok, "dictionary items are invalid");
                    let _ = ok;
                }
            }
            _ => {}
        }
        self.alloc_expr(Expr::new(kind, src))
    }

    /// `cache->NS<T>(src, args...)` for statements. Suite slots are wrapped with
    /// [`AstArena::wrap_suite`] as the C++ constructors do.
    pub fn new_stmt(&mut self, kind: impl Into<StmtKind>, src: SrcInfo) -> StmtId {
        let mut kind = kind.into();
        struct Wrap<'a>(&'a mut AstArena);
        impl IdMapper for Wrap<'_> {
            fn expr(&mut self, id: ExprId) -> ExprId {
                id
            }
            fn stmt(&mut self, id: StmtId) -> StmtId {
                id
            }
            fn suite(&mut self, id: StmtId) -> StmtId {
                self.0.wrap_suite(Some(id)).unwrap_or(id)
            }
        }
        kind.map_ids(&mut Wrap(self));
        self.alloc_stmt(Stmt::new(kind, src))
    }

    /// `SuiteStmt::wrap`: wraps a non-suite statement into a new suite (with its src).
    pub fn wrap_suite(&mut self, s: Option<StmtId>) -> Option<StmtId> {
        let s = s?;
        if self[s].kind.as_suite().is_some() {
            return Some(s);
        }
        let src = self[s].src;
        Some(self.alloc_stmt(Stmt::new(StmtKind::Suite(SuiteStmt { items: vec![s] }), src)))
    }

    /// `CallArg(name, value)`: copies the value's source location.
    pub fn call_arg(&self, name: impl Into<String>, value: Option<ExprId>) -> CallArg {
        let mut a = CallArg::new(name, value);
        if let Some(v) = value {
            a.src = self[v].src;
        }
        a
    }

    /// `CallExpr(expr, std::vector<Expr*>)`: positional args (null args skipped).
    pub fn new_call(&mut self, src: SrcInfo, expr: ExprId, args: &[ExprId]) -> ExprId {
        let items = args.iter().map(|&a| self.call_arg("", Some(a))).collect();
        self.new_expr(CallExpr { expr: Some(expr), items, ordered: false, partial: false }, src)
    }

    /// `Param(name, type, defaultValue, status, kind)` including generic-status inference:
    /// with `status_hint == 0`, a parameter typed `type`, `TypeTrait`, `TypeTrait[...]`,
    /// `Static[...]` or `Literal[...]` becomes `Generic`.
    pub fn new_param(
        &self,
        src: SrcInfo,
        name: impl Into<String>,
        ty: Option<ExprId>,
        default_value: Option<ExprId>,
        status_hint: i32,
        kind: ParamKind,
    ) -> Param {
        let stdlib = typhon_common::mangle::stdlib();
        let is_type_param = ty.is_some_and(|t| match &self[t].kind {
            ExprKind::Id(i) => i.value == stdlib.type_ || i.value == stdlib.type_trait,
            ExprKind::Index(ix) => is_id(self, ix.expr, &stdlib.type_trait),
            _ => false,
        });
        let status = if status_hint == 0 && (is_type_param || self.get_static_generic(ty) != LiteralKind::Runtime) {
            ParamStatus::Generic
        } else {
            match status_hint {
                0 => ParamStatus::Value,
                1 => ParamStatus::Generic,
                _ => ParamStatus::HiddenGeneric,
            }
        };
        Param { name: name.into(), ty, default_value, kind, status, src }
    }

    /// `getStaticGeneric`: `Static[T]` / `Literal[T]` -> literal kind of `T`.
    pub fn get_static_generic(&self, e: Option<ExprId>) -> LiteralKind {
        let Some(e) = e else { return LiteralKind::Runtime };
        if let ExprKind::Index(ix) = &self[e].kind {
            if (is_id(self, ix.expr, "Static") || is_id(self, ix.expr, "Literal"))
                && let Some(i) = ix.index.and_then(|i| self[i].kind.as_id())
            {
                return LiteralKind::from_str(&i.value);
            }
        }
        LiteralKind::Runtime
    }

    /// `FunctionStmt(name, ret, args, suite, decorators, async)` (runs `parseParams`).
    #[allow(clippy::too_many_arguments)]
    pub fn new_function(
        &mut self,
        src: SrcInfo,
        name: impl Into<String>,
        ret: Option<ExprId>,
        params: Vec<Param>,
        suite: Option<StmtId>,
        decorators: Vec<ExprId>,
        async_: bool,
    ) -> Result<StmtId> {
        let items = parse_params(params)?;
        let f = FunctionStmt {
            name: name.into(),
            items,
            ret,
            suite,
            decorators,
            async_,
            non_inferrable_generics: None,
            call_layout: None,
        };
        Ok(self.new_stmt(f, src))
    }

    /// `LambdaExpr(vars, expr)` (runs `parseParams`).
    pub fn new_lambda(&mut self, src: SrcInfo, params: Vec<Param>, expr: Option<ExprId>) -> Result<ExprId> {
        let items = parse_params(params)?;
        Ok(self.new_expr(LambdaExpr { items, expr }, src))
    }

    /// `ClassStmt(name, args, suite, decorators, baseClasses)`.
    pub fn new_class(
        &mut self,
        src: SrcInfo,
        name: impl Into<String>,
        items: Vec<Param>,
        suite: Option<StmtId>,
        decorators: Vec<ExprId>,
        base_classes: Vec<ExprId>,
    ) -> StmtId {
        self.new_stmt(ClassStmt { name: name.into(), items, suite, decorators, base_classes }, src)
    }

    /// `WithStmt(std::vector<std::pair<Expr*, Expr*>>, suite, async)`: variables that are
    /// `IdExpr`s are kept by name, others become unnamed.
    pub fn new_with_pairs(&mut self, src: SrcInfo, pairs: Vec<(ExprId, Option<ExprId>)>, suite: Option<StmtId>, async_: bool) -> StmtId {
        let mut items = vec![];
        let mut vars = vec![];
        for (i, j) in pairs {
            items.push(i);
            vars.push(j.and_then(|j| self[j].kind.as_id()).map(|x| x.value.clone()).unwrap_or_default());
        }
        self.new_stmt(WithStmt { items, vars, suite, async_ }, src)
    }

    // ---------------------------------------------------------------- generators

    /// `GeneratorExpr(cache, kind, expr, loops)`: `loops[0]` must be a `ForStmt`; the
    /// remaining For/If statements are nested and `expr` becomes the final `ExprStmt`.
    pub fn new_generator(&mut self, src: SrcInfo, kind: GeneratorKind, expr: ExprId, mut loops: Vec<StmtId>) -> ExprId {
        typhon_common::seqassert!(
            !loops.is_empty() && self[loops[0]].kind.as_for().is_some(),
            "bad generator constructor"
        );
        let es = self.alloc_stmt(Stmt::new(StmtKind::Expr(ExprStmt { expr: Some(expr) }), src));
        let suite = self.alloc_stmt(Stmt::new(StmtKind::Suite(SuiteStmt { items: vec![es] }), src));
        loops.push(suite);
        let root = self.form_complete_stmt(&loops);
        self.new_expr(GeneratorExpr { kind, loops: root }, src)
    }

    /// `GeneratorExpr(cache, key, expr, loops)`: dict comprehension.
    pub fn new_dict_generator(&mut self, src: SrcInfo, key: ExprId, expr: ExprId, loops: Vec<StmtId>) -> ExprId {
        let t = self.new_expr(TupleExpr { items: vec![key, expr] }, src);
        self.new_generator(src, GeneratorKind::DictGenerator, t, loops)
    }

    fn form_complete_stmt(&mut self, loops: &[StmtId]) -> Option<StmtId> {
        let mut fin: Option<StmtId> = None;
        for &l in loops.iter().rev() {
            let w = self.wrap_suite(fin);
            match &mut self[l].kind {
                StmtKind::If(s) => s.if_suite = w,
                StmtKind::For(s) => s.suite = w,
                _ => {}
            }
            fin = Some(l);
        }
        loops.first().copied()
    }

    /// `GeneratorExpr::loopCount`.
    pub fn generator_loop_count(&self, g: ExprId) -> usize {
        let mut cnt = 0;
        let mut i = self[g].kind.as_generator().and_then(|g| g.loops);
        while let Some(s) = i {
            match &self[s].kind {
                StmtKind::For(f) => {
                    i = f.suite;
                    cnt += 1;
                }
                StmtKind::If(f) => {
                    i = f.if_suite;
                    cnt += 1;
                }
                StmtKind::Suite(ss) => {
                    if ss.items.is_empty() {
                        break;
                    }
                    i = ss.items.last().copied();
                }
                _ => break,
            }
        }
        cnt
    }

    /// `GeneratorExpr::getFinalSuite` (the loops root).
    pub fn generator_final_suite(&self, g: ExprId) -> Option<StmtId> {
        self[g].kind.as_generator().and_then(|g| g.loops)
    }

    /// `GeneratorExpr::getFinalStmt`: slot holding the innermost statement.
    pub fn generator_final_slot(&self, g: ExprId) -> GenSlot {
        let mut slot = GenSlot::Loops(g);
        loop {
            let Some(s) = self.gen_slot_get(slot) else { return slot };
            slot = match &self[s].kind {
                StmtKind::For(_) => GenSlot::ForSuite(s),
                StmtKind::If(_) => GenSlot::IfSuite(s),
                StmtKind::Suite(ss) => {
                    if ss.items.is_empty() {
                        return slot;
                    }
                    GenSlot::SuiteItem(s, ss.items.len() - 1)
                }
                _ => return slot,
            };
        }
    }

    /// Read a generator slot.
    pub fn gen_slot_get(&self, slot: GenSlot) -> Option<StmtId> {
        match slot {
            GenSlot::Loops(g) => self[g].kind.as_generator().and_then(|g| g.loops),
            GenSlot::ForSuite(s) => self[s].kind.as_for().and_then(|f| f.suite),
            GenSlot::IfSuite(s) => self[s].kind.as_if().and_then(|f| f.if_suite),
            GenSlot::SuiteItem(s, i) => self[s].kind.as_suite().and_then(|f| f.items.get(i).copied()),
        }
    }

    /// Write a generator slot.
    pub fn gen_slot_set(&mut self, slot: GenSlot, v: StmtId) {
        match slot {
            GenSlot::Loops(g) => {
                if let Some(x) = self[g].kind.as_generator_mut() {
                    x.loops = Some(v)
                }
            }
            GenSlot::ForSuite(s) => {
                if let Some(x) = self[s].kind.as_for_mut() {
                    x.suite = Some(v)
                }
            }
            GenSlot::IfSuite(s) => {
                if let Some(x) = self[s].kind.as_if_mut() {
                    x.if_suite = Some(v)
                }
            }
            GenSlot::SuiteItem(s, i) => {
                if let Some(x) = self[s].kind.as_suite_mut()
                    && i < x.items.len()
                {
                    x.items[i] = v
                }
            }
        }
    }

    /// `GeneratorExpr::getFinalExpr`.
    pub fn generator_final_expr(&self, g: ExprId) -> Option<ExprId> {
        let s = self.gen_slot_get(self.generator_final_slot(g))?;
        self[s].kind.as_expr().and_then(|e| e.expr)
    }

    /// `GeneratorExpr::setFinalExpr`.
    pub fn generator_set_final_expr(&mut self, g: ExprId, e: ExprId) {
        let src = self[g].src;
        let s = self.alloc_stmt(Stmt::new(StmtKind::Expr(ExprStmt { expr: Some(e) }), src));
        self.generator_set_final_stmt(g, s);
    }

    /// `GeneratorExpr::setFinalStmt`.
    pub fn generator_set_final_stmt(&mut self, g: ExprId, s: StmtId) {
        let slot = self.generator_final_slot(g);
        self.gen_slot_set(slot, s);
    }

    // ---------------------------------------------------------------- suites

    /// `Stmt::firstInBlock`.
    pub fn first_in_block(&self, s: StmtId) -> Option<StmtId> {
        match &self[s].kind {
            StmtKind::Suite(ss) => ss.items.first().and_then(|&f| self.first_in_block(f)),
            _ => Some(s),
        }
    }

    /// `SuiteStmt::flatten`: drops nulls and inlines nested suites (one level).
    pub fn suite_flatten(&mut self, s: StmtId) {
        let Some(items) = self[s].kind.as_suite().map(|x| x.items.clone()) else { return };
        let mut ns = vec![];
        for i in items {
            match self[i].kind.as_suite() {
                Some(inner) => ns.extend(inner.items.iter().copied()),
                None => ns.push(i),
            }
        }
        if let Some(x) = self[s].kind.as_suite_mut() {
            x.items = ns;
        }
    }

    /// `SuiteStmt::addStmt`.
    pub fn suite_add_stmt(&mut self, s: StmtId, x: Option<StmtId>) {
        if let Some(x) = x {
            let st = &mut self[s];
            if let Some(ss) = st.kind.as_suite_mut() {
                ss.items.push(x);
                st.done = false;
            }
        }
    }

    // ---------------------------------------------------------------- functions/classes

    fn docstr(&self, suite: Option<StmtId>) -> String {
        if let Some(f) = suite.and_then(|s| self.first_in_block(s))
            && let Some(e) = self[f].kind.as_expr().and_then(|e| e.expr)
            && let Some(s) = self[e].kind.as_string()
        {
            return s.value();
        }
        String::new()
    }
    /// `FunctionStmt::getDocstr`.
    pub fn function_docstr(&self, f: StmtId) -> String {
        self.docstr(self[f].kind.as_function().and_then(|f| f.suite))
    }
    /// `ClassStmt::getDocstr`.
    pub fn class_docstr(&self, c: StmtId) -> String {
        self.docstr(self[c].kind.as_class().and_then(|f| f.suite))
    }
    /// `ClassStmt::isRecord`.
    pub fn class_is_record(&self, c: StmtId) -> bool {
        self[c].attrs.has(Attr::Tuple)
    }
    /// `ClassStmt::isClassVar`.
    pub fn is_class_var(&self, p: &Param) -> bool {
        match p.ty {
            None => true,
            Some(t) => match &self[t].kind {
                ExprKind::Index(i) => is_id(self, i.expr, "ClassVar"),
                _ => false,
            },
        }
    }
    /// `FunctionStmt::hasFunctionAttribute`.
    pub fn has_function_attribute(&self, f: StmtId, attr: &str) -> bool {
        self[f].attrs.get_key_value(Attr::FunctionAttributes).is_some_and(|m| m.contains_key(attr))
    }

    /// True if an `IdExpr` named `what` occurs in the subtree (C++ `IdSearchVisitor`,
    /// which does not look inside f-string parts).
    pub fn contains_id(&self, e: ExprId, what: &str) -> bool {
        use super::children::{Child, ChildCollector};
        let mut stack = vec![Child::Expr(e)];
        while let Some(c) = stack.pop() {
            let mut col = ChildCollector { children: vec![], include_string_parts: false };
            match c {
                Child::Expr(e) => {
                    if self[e].kind.as_id().is_some_and(|i| i.value == what) {
                        return true;
                    }
                    self[e].kind.clone().map_ids(&mut col);
                }
                Child::Stmt(s) | Child::Suite(s) => self[s].kind.clone().map_ids(&mut col),
            }
            stack.extend(col.children);
        }
        false
    }

    /// `FunctionStmt::getNonInferrableGenerics` (cached on the node).
    pub fn non_inferrable_generics(&mut self, f: StmtId) -> Rc<HashSet<String>> {
        let Some(func) = self[f].kind.as_function() else { return Rc::default() };
        if let Some(c) = &func.non_inferrable_generics {
            return c.clone();
        }
        let mut set = HashSet::new();
        for a in &func.items {
            if a.status == ParamStatus::Generic && a.default_value.is_none() {
                let mut inferrable = func.items.iter().any(|b| b.ty.is_some_and(|t| self.contains_id(t, &a.name)));
                if func.ret.is_some_and(|r| self.contains_id(r, &a.name)) {
                    inferrable = true;
                }
                if !inferrable {
                    set.insert(a.name.clone());
                }
            }
        }
        let rc = Rc::new(set);
        if let Some(func) = self[f].kind.as_function_mut() {
            func.non_inferrable_generics = Some(rc.clone());
        }
        rc
    }

    // ---------------------------------------------------------------- cloning

    /// Deep copy (C++ `ast::clone(e, clean)`): `clean` strips `ty`, `expected_type` and
    /// `done`. `orig_expr` is copied as a reference (not cloned), attributes are cloned.
    pub fn clone_expr(&mut self, id: ExprId, clean: bool) -> ExprId {
        let mut e = self[id].clone();
        struct M<'a> {
            a: &'a mut AstArena,
            clean: bool,
        }
        impl IdMapper for M<'_> {
            fn expr(&mut self, id: ExprId) -> ExprId {
                self.a.clone_expr(id, self.clean)
            }
            fn stmt(&mut self, id: StmtId) -> StmtId {
                self.a.clone_stmt(id, self.clean)
            }
        }
        e.kind.map_ids(&mut M { a: self, clean });
        if let ExprKind::String(s) = &mut e.kind {
            // C++ StringExpr copy ctor clones part expressions with clean=false.
            for p in &mut s.strings {
                if let Some(x) = p.expr {
                    p.expr = Some(self.clone_expr(x, false));
                }
            }
        }
        if clean {
            e.ty = None;
            e.done = false;
            e.expected_type = None;
        }
        self.alloc_expr(e)
    }

    pub fn clone_expr_opt(&mut self, id: Option<ExprId>, clean: bool) -> Option<ExprId> {
        id.map(|i| self.clone_expr(i, clean))
    }

    /// Deep copy of a statement (C++ `ast::clone(s, clean)`).
    pub fn clone_stmt(&mut self, id: StmtId, clean: bool) -> StmtId {
        let mut s = self[id].clone();
        struct M<'a> {
            a: &'a mut AstArena,
            clean: bool,
        }
        impl IdMapper for M<'_> {
            fn expr(&mut self, id: ExprId) -> ExprId {
                self.a.clone_expr(id, self.clean)
            }
            fn stmt(&mut self, id: StmtId) -> StmtId {
                self.a.clone_stmt(id, self.clean)
            }
        }
        s.kind.map_ids(&mut M { a: self, clean });
        if let StmtKind::Function(f) = &mut s.kind {
            f.non_inferrable_generics = None;
            f.call_layout = None;
        }
        // PORT-NOTE: C++ ExceptStmt's copy constructor does not reset `done` on clean.
        if clean && s.kind.as_except().is_none() {
            s.done = false;
        }
        self.alloc_stmt(s)
    }

    pub fn clone_stmt_opt(&mut self, id: Option<StmtId>, clean: bool) -> Option<StmtId> {
        id.map(|i| self.clone_stmt(i, clean))
    }

    /// Clone a parameter (`Param::clone`): re-runs status inference like C++.
    pub fn clone_param(&mut self, p: &Param, clean: bool) -> Param {
        let ty = self.clone_expr_opt(p.ty, clean);
        let d = self.clone_expr_opt(p.default_value, clean);
        let hint = match p.status {
            ParamStatus::Value => 0,
            ParamStatus::Generic => 1,
            ParamStatus::HiddenGeneric => 2,
        };
        // PORT-NOTE: C++ Param::clone does not copy the source location.
        self.new_param(SrcInfo::default(), p.name.clone(), ty, d, hint, p.kind)
    }

    /// Deep copy an expression from another arena (keeps `ty`/`done`; `orig_expr` is
    /// dropped since it refers to the other arena).
    pub fn import_expr(&mut self, other: &AstArena, id: ExprId) -> ExprId {
        let mut e = other[id].clone();
        e.orig_expr = None;
        struct M<'a, 'b> {
            a: &'a mut AstArena,
            o: &'b AstArena,
        }
        impl IdMapper for M<'_, '_> {
            fn expr(&mut self, id: ExprId) -> ExprId {
                self.a.import_expr(self.o, id)
            }
            fn stmt(&mut self, id: StmtId) -> StmtId {
                self.a.import_stmt(self.o, id)
            }
            fn string_parts(&self) -> bool {
                true
            }
        }
        e.kind.map_ids(&mut M { a: self, o: other });
        self.alloc_expr(e)
    }

    /// Deep copy a statement from another arena.
    pub fn import_stmt(&mut self, other: &AstArena, id: StmtId) -> StmtId {
        let mut s = other[id].clone();
        struct M<'a, 'b> {
            a: &'a mut AstArena,
            o: &'b AstArena,
        }
        impl IdMapper for M<'_, '_> {
            fn expr(&mut self, id: ExprId) -> ExprId {
                self.a.import_expr(self.o, id)
            }
            fn stmt(&mut self, id: StmtId) -> StmtId {
                self.a.import_stmt(self.o, id)
            }
            fn string_parts(&self) -> bool {
                true
            }
        }
        s.kind.map_ids(&mut M { a: self, o: other });
        if let StmtKind::Function(f) = &mut s.kind {
            f.non_inferrable_generics = None;
            f.call_layout = None;
        }
        self.alloc_stmt(s)
    }

    /// Debug name of a type for printing (`debugString(mode)`).
    pub fn type_debug_string(&self, t: TypeId, mode: i32) -> String {
        match &self.type_fmt {
            Some(f) => f.debug_string(t, mode),
            None => format!("T{}", t.0),
        }
    }
    /// `realizedName()` of a type.
    pub fn type_realized_name(&self, t: TypeId) -> String {
        match &self.type_fmt {
            Some(f) => f.realized_name(t),
            None => format!("T{}", t.0),
        }
    }
}

/// `FunctionStmt::parseParams`: consumes `/` and bare `*` separators, marks
/// positional-only / keyword-only parameters and validates their placement.
pub fn parse_params(args: Vec<Param>) -> Result<Vec<Param>> {
    let mut params: Vec<Param> = vec![];
    let (mut slash, mut star, mut bare_star, mut keyword) = (false, false, false, false);
    let mut star_info = SrcInfo::default();
    for mut p in args {
        if p.name == "/" {
            if slash || star || params.is_empty() {
                typhon_common::bail_e!(ErrorKind::CUSTOM, p.src, "'/' must follow positional parameters and precede '*'");
            }
            for q in &mut params {
                q.kind = ParamKind::PositionalOnly;
            }
            slash = true;
        } else if p.name == "*" {
            if star {
                typhon_common::bail_e!(ErrorKind::CUSTOM, p.src, "'*' may appear only once");
            }
            star = true;
            bare_star = true;
            star_info = p.src;
        } else {
            let stars = p.name_with_stars().0;
            if stars > 0 {
                if bare_star && stars == 1 {
                    typhon_common::bail_e!(ErrorKind::CUSTOM, p.src, "'*' may appear only once");
                }
                star = true;
            } else if star {
                p.kind = ParamKind::KeywordOnly;
                keyword = true;
            }
            params.push(p);
        }
    }
    if bare_star && !keyword {
        typhon_common::bail_e!(ErrorKind::CUSTOM, star_info, "named arguments must follow bare '*'");
    }
    Ok(params)
}
