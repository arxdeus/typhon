//! Statement translation (`TranslateVisitor::visit(XStmt*)`).

use typhon_common::mangle::{self, stdlib};
use typhon_common::{CompilerError, Result, SrcInfo};
use typhon_ir::schedule::OMPSched;
use typhon_ir::{Catch, ValueId, ValueKind};

use crate::ast::{Attr, ExprKind, StmtId, StmtKind};
use crate::cache::{var_argc, var_argv};

use super::{TranslateItem, TranslateVisitor};

impl TranslateVisitor<'_> {
    /// `transform(Stmt*)`: translate and append the result to the current series.
    pub fn transform_stmt(&mut self, s: StmtId) -> Result<Option<ValueId>> {
        let old = std::mem::replace(&mut self.src, self.cache.ast[s].src);
        let r = self.visit_stmt(s);
        self.src = old;
        let r = r?;
        if let Some(v) = r {
            self.push_series(v);
        }
        Ok(r)
    }

    fn tr_suite(&mut self, s: Option<StmtId>) -> Result<()> {
        if let Some(s) = s {
            self.transform_stmt(s)?;
        }
        Ok(())
    }

    /// Translate `s` into a fresh series named `name`.
    fn in_series(&mut self, series: ValueId, s: Option<StmtId>) -> Result<()> {
        self.ctx.borrow_mut().series.push(series);
        let r = self.tr_suite(s);
        self.ctx.borrow_mut().series.pop();
        r
    }

    fn set_generator(&self) {
        let b = self.ctx.borrow().get_base();
        if let Some(f) = self.module.borrow_mut().func_mut(b) {
            f.generator = true;
        }
    }

    /// `make<ir::Var>(stmt, type, false, false, false, name)` + `getBase()->push_back`.
    fn new_local(&self, src: SrcInfo, ty: Option<typhon_ir::TypeId>, name: &str) -> typhon_ir::VarId {
        let mut m = self.module.borrow_mut();
        m.n_var(src, ty, false, false, false, name.to_string())
    }
    fn push_symbol(&self, v: typhon_ir::VarId) {
        let b = self.ctx.borrow().get_base();
        self.module.borrow_mut().func_push_symbol(b, v);
    }

    fn visit_stmt(&mut self, s: StmtId) -> Result<Option<ValueId>> {
        let src = self.src;
        let kind = self.cache.ast[s].kind.clone();
        let types = self.cache.types.clone();
        match kind {
            StmtKind::Suite(st) => {
                for i in st.items {
                    self.transform_stmt(i)?;
                }
                Ok(None)
            }
            StmtKind::Break(_) => Ok(Some(self.make(src, ValueKind::BreakInstr(Default::default())))),
            StmtKind::Continue(_) => Ok(Some(self.make(src, ValueKind::ContinueInstr(Default::default())))),
            StmtKind::Expr(es) => {
                let e = es.expr.ok_or_else(|| CompilerError::custom("empty expression", src))?;
                if let ExprKind::Call(c) = &self.cache.ast[e].kind {
                    let is_final = c.expr.and_then(|x| self.cache.ast[x].kind.as_id().map(|i| i.value.clone()))
                        == Some(mangle::mangled_method("", "Generator", "_yield_final", 0, 0));
                    if is_final {
                        let a = c.items[0].value;
                        let v = self.tr_opt(a)?;
                        let r = self.make(src, ValueKind::YieldInstr(typhon_ir::YieldInstr { value: Some(v), final_: true }));
                        self.set_generator();
                        return Ok(Some(r));
                    }
                }
                Ok(Some(self.transform_expr(e)?))
            }
            StmtKind::Assign(a) => {
                let lhs = a.lhs.ok_or_else(|| CompilerError::custom("bad assignment", src))?;
                let var = match &self.cache.ast[lhs].kind {
                    ExprKind::Id(i) => i.value.clone(),
                    _ => return Err(CompilerError::custom("expected IdExpr", src)),
                };
                if var == var_argv() || var == var_argc() {
                    return Ok(None);
                }
                let is_global = self.cache.globals.contains_key(&var);
                let ty_src = a.rhs.unwrap_or(lhs);
                if a.is_update() {
                    // This might not exist if static if-else was combined with types.
                    if let Some(TranslateItem::Var(v)) = self.find(&var) {
                        if self.module.borrow().var(v).ty.is_none() {
                            let t = self.get_type(self.ety(ty_src)?)?;
                            let mut m = self.module.borrow_mut();
                            m.set_src_info(v.into(), src);
                            m.set_var_type(v, t);
                        }
                        if let Some(r) = a.rhs {
                            let rhs = self.transform_expr(r)?;
                            return Ok(Some(self.make(src, ValueKind::AssignInstr(typhon_ir::AssignInstr { lhs: v, rhs }))));
                        }
                    }
                    return Ok(None);
                }
                let lt = self.ety(lhs)?;
                if !types.is_instantiated(lt) || types.is(lt, &stdlib().type_) || types.get_func(lt).is_some() {
                    if let Some(r) = a.rhs {
                        if !matches!(self.cache.ast[r].kind, ExprKind::Id(_)) {
                            return Ok(Some(self.transform_expr(r)?)); // side effect
                        }
                    }
                    return Ok(None); // type aliases/fn aliases etc
                }
                let t = self.get_type(self.ety(ty_src)?)?;
                let v = if is_global {
                    let v = self.find(&var).and_then(|i| i.var()).ok_or_else(|| CompilerError::custom(format!("cannot find global '{var}'"), src))?;
                    let mut m = self.module.borrow_mut();
                    m.set_src_info(v.into(), src);
                    m.set_var_type(v, t);
                    v
                } else {
                    let v = self.new_local(src, Some(t), &var);
                    self.push_symbol(v);
                    self.ctx.borrow_mut().add(&var, TranslateItem::Var(v));
                    v
                };
                if a.is_thread_local() {
                    let mut m = self.module.borrow_mut();
                    m.var_mut(v).tls = true;
                    m.set_global(v, true);
                }
                if self.cache.ast[lhs].attrs.has(Attr::ExprExternVar) {
                    let n = self.cache.rev(&var);
                    let mut m = self.module.borrow_mut();
                    m.var_mut(v).external = true;
                    m.set_name(v.into(), n);
                    m.set_global(v, true);
                    return Ok(None);
                }
                match a.rhs {
                    Some(r) => {
                        let rhs = self.transform_expr(r)?;
                        Ok(Some(self.make(src, ValueKind::AssignInstr(typhon_ir::AssignInstr { lhs: v, rhs }))))
                    }
                    None => Ok(None),
                }
            }
            StmtKind::AssignMember(a) => {
                let lhs = a.lhs.unwrap();
                let lt = self.ety(lhs)?;
                let name = types.get_class(lt).and_then(|c| types.class_name(c)).map(|s| s.to_string()).unwrap_or_default();
                let rtti = self.cache.classes.get(&name).is_some_and(|c| {
                    let an = c.ast.and_then(|x| self.cache.ast[x].kind.as_class().map(|k| k.name.clone())).unwrap_or_default();
                    c.has_rtti(&an)
                });
                let l = self.transform_expr(lhs)?;
                let r = self.tr_opt(a.rhs)?;
                Ok(Some(self.make(
                    src,
                    ValueKind::InsertInstr(typhon_ir::InsertInstr { lhs: l, field: a.member.clone(), rhs: r, polymorphic: rtti }),
                )))
            }
            StmtKind::Return(r) => {
                let v = match r.expr {
                    Some(e) => Some(self.transform_expr(e)?),
                    None => None,
                };
                Ok(Some(self.make(src, ValueKind::ReturnInstr(typhon_ir::ReturnInstr { value: v }))))
            }
            StmtKind::Yield(y) => {
                let v = match y.expr {
                    Some(e) => Some(self.transform_expr(e)?),
                    None => None,
                };
                let r = self.make(src, ValueKind::YieldInstr(typhon_ir::YieldInstr { value: v, final_: false }));
                self.set_generator();
                Ok(Some(r))
            }
            StmtKind::While(w) => {
                let cond = self.tr_opt(w.cond)?;
                let body = self.new_series(src, "body");
                let lp = self.make(src, ValueKind::WhileFlow(typhon_ir::WhileFlow { cond, body }));
                self.in_series(body, w.suite)?;
                Ok(Some(lp))
            }
            StmtKind::For(_) => self.visit_for(s),
            StmtKind::If(i) => {
                let cond = self.tr_opt(i.cond)?;
                let t = self.new_series(src, "ifstmt_true");
                self.in_series(t, i.if_suite)?;
                let f = match i.else_suite {
                    Some(e) => {
                        let f = self.new_series(src, "ifstmt_false");
                        self.in_series(f, Some(e))?;
                        Some(f)
                    }
                    None => None,
                };
                Ok(Some(self.make(src, ValueKind::IfFlow(typhon_ir::IfFlow { cond, true_branch: t, false_branch: f }))))
            }
            StmtKind::Try(_) => self.visit_try(s),
            StmtKind::Throw(t) => {
                let v = match t.expr {
                    Some(e) => Some(self.transform_expr(e)?),
                    None => None,
                };
                Ok(Some(self.make(src, ValueKind::ThrowInstr(typhon_ir::ThrowInstr { value: v, fallback: None }))))
            }
            StmtKind::Function(f) => {
                let llvm = self.cache.ast[s].attrs.has(Attr::LLVM);
                self.transform_function_realizations(&f.name, llvm)?;
                Ok(None)
            }
            // All type handles are already generated; methods are handled by FunctionStmt.
            StmtKind::Class(_) | StmtKind::Comment(_) | StmtKind::Directive(_) => Ok(None),
            _ => Err(CompilerError::custom(format!("invalid node {}", self.cache.ast.stmt_to_string(s, -1)), src)),
        }
    }

    fn visit_for(&mut self, s: StmtId) -> Result<Option<ValueId>> {
        let src = self.src;
        let types = self.cache.types.clone();
        let f = self.cache.ast[s].kind.as_for().cloned().unwrap();
        let mut os = None;
        if let Some(d) = f.decorator {
            let c = self.cache.ast[d].kind.as_call().cloned().ok_or_else(|| CompilerError::custom("for par is not a call", src))?;
            let fc = types.get_func(self.ety(c.expr.unwrap())?).ok_or_else(|| CompilerError::custom("for par is not a function", src))?;
            typhon_common::seqassert!(types.func_name(fc) == mangle::func("std.openmp", "for_par"), "for par is not a function");
            let g: Vec<_> = types.func_generics(fc).iter().map(|g| g.ty.unwrap()).collect();
            let schedule = types.get_str_static(g[0]).unwrap_or_default();
            let ordered = types.get_bool_static(g[1]).unwrap_or(false);
            let threads = self.tr_opt(c.items[0].value)?;
            let chunk = self.tr_opt(c.items[1].value)?;
            let collapse = types.get_int_static(g[2]).unwrap_or(0);
            let gpu = types.get_bool_static(g[3]).unwrap_or(false);
            let m = self.module.borrow();
            os = Some(Box::new(OMPSched::from_schedule(&m, &schedule, Some(threads), Some(chunk), ordered, collapse, gpu)));
        }
        let ve = f.var.ok_or_else(|| CompilerError::custom("expected IdExpr", src))?;
        let var_name = match &self.cache.ast[ve].kind {
            ExprKind::Id(i) => i.value.clone(),
            _ => return Err(CompilerError::custom("expected IdExpr", src)),
        };
        let existing = self.find(&var_name);
        let var = match existing.and_then(|i| i.var()) {
            Some(v) if self.cache.ast[ve].attrs.has(Attr::ExprDominated) => v,
            _ => {
                let t = self.get_type(self.ety(ve)?)?;
                self.new_local(src, Some(t), &var_name)
            }
        };
        self.push_symbol(var);
        let body = self.new_series(src, "body");
        let iter = self.tr_opt(f.iter)?;
        let lp = self.make(
            src,
            ValueKind::ForFlow(typhon_ir::ForFlow { iter, body, var, schedule: os, is_async: f.async_ }),
        );
        self.ctx.borrow_mut().add(&var_name, TranslateItem::Var(var));
        self.in_series(body, f.suite)?;
        Ok(Some(lp))
    }

    fn visit_try(&mut self, s: StmtId) -> Result<Option<ValueId>> {
        let src = self.src;
        let t = self.cache.ast[s].kind.as_try().cloned().unwrap();
        let body = self.new_series(src, "body");
        self.in_series(body, t.suite)?;
        let finally = self.new_series(src, "finally");
        if t.finally.is_some() {
            self.in_series(finally, t.finally)?;
        }
        let else_ = match t.else_suite {
            Some(e) => {
                let es = self.new_series(src, "else");
                self.in_series(es, Some(e))?;
                Some(es)
            }
            None => None,
        };
        let mut catches = vec![];
        for c in t.items {
            let ex = self.cache.ast[c].kind.as_except().cloned().unwrap();
            let catch_body = self.new_series(src, "catch");
            let exc_type = match ex.exc {
                Some(e) => {
                    let et = self.ety(e)?;
                    let et = self.extract_type(et);
                    Some(self.get_type(et)?)
                }
                None => None,
            };
            let mut catch_var = None;
            if !ex.var.is_empty() {
                let dominated = self.cache.ast[c].attrs.has(Attr::ExprDominated);
                let v = match self.find(&ex.var).and_then(|i| i.var()) {
                    Some(v) if dominated => v,
                    _ => self.new_local(src, exc_type, &ex.var),
                };
                self.ctx.borrow_mut().add(&ex.var, TranslateItem::Var(v));
                self.push_symbol(v);
                catch_var = Some(v);
            }
            self.in_series(catch_body, ex.suite)?;
            catches.push(Catch { handler: catch_body, ty: exc_type, var: catch_var });
        }
        let tc = self.make(
            src,
            ValueKind::TryCatchFlow(typhon_ir::TryCatchFlow { catches, body, else_, finally: Some(finally) }),
        );
        Ok(Some(tc))
    }
}
