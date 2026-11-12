//! `@dataclass` semantics on top of the generic class magic codegen:
//! per-instance `field(default_factory=...)`, `__post_init__`, `kw_only=True`,
//! `frozen=True` and the CPython-style `Name(a=1)` repr.

use typhon_common::Result;

use super::TypecheckVisitor;
use crate::ast::{Attr, CallExpr, ExprKind, IfStmt, ParamKind, StringExpr, SuiteStmt, ThrowStmt};
use crate::ids::{ExprId, StmtId, TypeId};

impl<'a> TypecheckVisitor<'a> {
    fn class_magic_has(&self, s: StmtId, m: &str) -> bool {
        self.cache.ast[s].attrs.get_str_list(Attr::ClassMagic).is_some_and(|l| l.iter().any(|x| x == m))
    }

    /// Whether a class statement is `@dataclass`-decorated (not `@tuple`).
    pub(crate) fn is_dataclass_stmt(&self, s: StmtId) -> bool {
        self.class_magic_has(s, "dataclass")
    }

    /// `x: T = field(default_factory=F)` -> `x: T = __per_call_default__(F())`, so the
    /// factory runs for every instance instead of once at class definition.
    pub(crate) fn dataclass_field_default(&mut self, d: Option<ExprId>) -> Option<ExprId> {
        let d = d?;
        let ExprKind::Call(c) = &self.cache.ast[d].kind else { return Some(d) };
        let c = c.clone();
        let is_field = c.expr.is_some_and(|f| match &self.cache.ast[f].kind {
            ExprKind::Id(i) => i.value == "field",
            ExprKind::Dot(dd) => dd.member == "field",
            _ => false,
        });
        if !is_field {
            return Some(d);
        }
        let Some(factory) = c.items.iter().find(|a| a.name == "default_factory").and_then(|a| a.value) else {
            return Some(d);
        };
        let call = self.n_expr(CallExpr { expr: Some(factory), items: vec![], ordered: false, partial: false });
        let w = self.id_n("__per_call_default__");
        Some(self.call_n(w, vec![call]))
    }

    /// Adjust a generated magic method of a dataclass: `__init__` gets keyword-only
    /// parameters (`kw_only=True`) and a trailing `__post_init__()` call.
    pub(crate) fn dataclass_postprocess_magic(&mut self, s: StmtId, m: &str, f: Option<StmtId>) -> Option<StmtId> {
        let f = f?;
        if m != "init" || !self.is_dataclass_stmt(s) {
            return Some(f);
        }
        let kw_only = self.class_magic_has(s, "kw_only");
        let suite = {
            let fd = self.cache.ast[f].kind.as_function_mut().unwrap();
            if kw_only {
                for p in fd.items.iter_mut().skip(1) {
                    if p.is_value() && !p.name.starts_with('*') {
                        p.kind = ParamKind::KeywordOnly;
                    }
                }
            }
            fd.suite
        };
        // if hasattr(self, "__post_init__"): self.__post_init__()
        let ha = self.id_n("hasattr");
        let sf = self.id_n("self");
        let nm = self.n_expr(StringExpr::new("__post_init__", ""));
        let cond = self.call_n(ha, vec![sf, nm]);
        let sf = self.id_n("self");
        let d = self.dot_n(sf, "__post_init__");
        let c = self.call_n(d, vec![]);
        let es = self.n_stmt(crate::ast::ExprStmt { expr: Some(c) });
        let st = self.n_stmt(IfStmt { cond: Some(cond), if_suite: Some(es), else_suite: None });
        if let Some(suite) = suite {
            self.cache.ast.suite_add_stmt(suite, Some(st));
        } else {
            let ns = self.n_stmt(SuiteStmt { items: vec![st] });
            self.cache.ast[f].kind.as_function_mut().unwrap().suite = Some(ns);
        }
        Some(f)
    }

    /// `obj.field = v` on a `@dataclass(frozen=True)` instance outside its generated
    /// `__init__` becomes `raise FrozenInstanceError("cannot assign to field 'field'")`.
    pub(crate) fn dataclass_frozen_assign(&mut self, lhs_class: TypeId, member: &str) -> Result<Option<StmtId>> {
        let Some(cast) = self.cache.get_class_by_type(lhs_class).and_then(|c| c.ast) else { return Ok(None) };
        if !self.class_magic_has(cast, "frozen") {
            return Ok(None);
        }
        let in_init = {
            let c = self.ctx.borrow();
            !c.is_global() && {
                let n = c.base_ref().name.to_string();
                let un = self.get_unmangled_name(&n);
                un == "__init__" || un == "__post_init__" || un == "__new__"
            }
        };
        if in_init {
            return Ok(None);
        }
        let e = self.id_n("FrozenInstanceError");
        let msg = self.n_expr(StringExpr::new(format!("cannot assign to field '{member}'"), ""));
        let c = self.call_n(e, vec![msg]);
        let t = self.n_stmt(ThrowStmt { expr: Some(c), from: None, transformed: false });
        Ok(self.transform_stmt_opt(Some(t))?)
    }
}
