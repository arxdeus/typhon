//! Function realization translation (`transformFunctionRealizations`,
//! `transformFunction`, `transformLLVMFunction`) and the old-IR rebody helper used by
//! `cythonizeClass`.

use rustc_hash::FxHashMap;
use typhon_common::{CompilerError, Result};
use typhon_ir::{Attribute, FuncId, FuncKind, Generic, KeyValueAttribute, Module, ValueKind};

use crate::ast::{Attr, ExprKind, ParamStatus, StmtId, StmtKind};
use crate::ids::TypeId;

use super::{TranslateItem, TranslateVisitor};

/// `cythonizeClass` tail: make the old IR function `old` forward to `new`
/// (`oldIR->setBody(series(call(newIR, args)))`).
pub fn rebody_forward(m: &mut Module, old: FuncId, new: FuncId) {
    let args: Vec<_> = m.func_args(old).into_iter().map(|a| m.nr_var_value(a)).collect();
    let call = typhon_ir::util::irtools::call(m, new, args);
    let body = typhon_ir::util::irtools::series(m, &[call]);
    m.set_func_body(old, body);
}

impl TranslateVisitor<'_> {
    pub(crate) fn transform_function_realizations(&mut self, name: &str, is_llvm: bool) -> Result<()> {
        let reals: Vec<(String, crate::cache::FunctionRealizationRef)> = match self.cache.functions.get(name) {
            Some(f) => f.realizations.iter().map(|(k, v)| (k.clone(), v.clone())).collect(),
            None => return Ok(()),
        };
        for (rn, r) in reals {
            let key = (name.to_string(), rn.clone());
            if !self.cache.pending_realizations.remove(&key) {
                continue;
            }
            typhon_common::log_flag!(TYPECHECK, "[translate] generating fn {}", rn);
            let (ty, ast, ir) = {
                let b = r.borrow();
                (b.ty, b.ast, b.ir)
            };
            let (Some(ty), Some(ast), Some(ir)) = (ty, ast, ir) else {
                return Err(CompilerError::custom(format!("AST not set for {rn}"), self.src));
            };
            self.module.borrow_mut().set_src_info(ir.into(), self.src);
            if is_llvm {
                self.transform_llvm_function(ast, ir)?;
            } else {
                self.transform_function(ty, ast, ir)?;
            }
        }
        Ok(())
    }

    /// Function attributes (`KeyValueAttribute` + `.module`).
    fn function_attributes(&self, ast: StmtId) -> KeyValueAttribute {
        let attrs = &self.cache.ast[ast].attrs;
        let mut kv: FxHashMap<String, String> = FxHashMap::default();
        if let Some(a) = attrs.get_key_value(Attr::FunctionAttributes) {
            kv.extend(a.iter().map(|(k, v)| (k.clone(), v.clone())));
        }
        kv.insert(".module".into(), attrs.get_str(Attr::Module).unwrap_or("").to_string());
        KeyValueAttribute::new(kv)
    }

    fn transform_function(&mut self, ty: TypeId, ast: StmtId, func: FuncId) -> Result<()> {
        let types = self.cache.types.clone();
        let fs = self.cache.ast[ast].kind.as_function().cloned().unwrap();
        let mut names = vec![];
        let mut indices = vec![];
        let mut j = 0;
        for (i, p) in fs.items.iter().enumerate() {
            if p.status == ParamStatus::Value {
                if types.get_func(types.func_arg(ty, j)).is_none() {
                    names.push(self.cache.rev(&p.name));
                    indices.push(i);
                }
                j += 1;
            }
        }
        let attrs = self.cache.ast[ast].attrs.clone();
        if attrs.has(Attr::CVarArg) {
            names.pop();
            indices.pop();
        }
        let kv = self.function_attributes(ast);
        {
            let mut m = self.module.borrow_mut();
            m.set_attribute(func.into(), Attribute::KeyValue(kv));
            for (n, &i) in names.iter().zip(&indices) {
                if let Some(av) = m.func_arg_var(func, n) {
                    m.set_src_info(av.into(), fs.items[i].src);
                }
            }
        }
        if !attrs.has(Attr::C) && !attrs.has(Attr::Internal) {
            self.ctx.borrow_mut().base.add_block();
            for (n, &i) in names.iter().zip(&indices) {
                let av = self.module.borrow().func_arg_var(func, n);
                if let Some(av) = av {
                    self.ctx.borrow_mut().add(&fs.items[i].name, TranslateItem::Var(av));
                }
            }
            let body = self.new_series(self.cache.ast[ast].src, "body");
            {
                let mut c = self.ctx.borrow_mut();
                c.bases.push(func);
                c.series.push(body);
            }
            let r = match fs.suite {
                Some(s) => self.transform_stmt(s).map(|_| ()),
                None => Ok(()),
            };
            {
                let mut c = self.ctx.borrow_mut();
                c.series.pop();
                c.bases.pop();
                c.base.pop_block();
            }
            r?;
            self.module.borrow_mut().set_func_body(func, Some(body));
        }
        if fs.async_ {
            if let Some(f) = self.module.borrow_mut().func_mut(func) {
                f.is_async = true;
            }
        }
        Ok(())
    }

    fn transform_llvm_function(&mut self, ast: StmtId, func: FuncId) -> Result<()> {
        let src = self.src;
        let fs = self.cache.ast[ast].kind.as_function().cloned().unwrap();
        let mut names = vec![];
        let mut indices = vec![];
        for (i, p) in fs.items.iter().enumerate() {
            if p.status == ParamStatus::Value {
                names.push(self.cache.rev(&p.name));
                indices.push(i);
            }
        }
        let kv = self.function_attributes(ast);
        {
            let mut m = self.module.borrow_mut();
            m.set_attribute(func.into(), Attribute::KeyValue(kv));
            for (n, &i) in names.iter().zip(&indices) {
                if let Some(av) = m.func_arg_var(func, n) {
                    m.set_src_info(av.into(), fs.items[i].src);
                }
            }
        }
        let bad = || CompilerError::custom("LLVM function does not begin with a string", src);
        let suite = fs.suite.ok_or_else(bad)?;
        let items: Vec<StmtId> = match &self.cache.ast[suite].kind {
            StmtKind::Suite(s) => s.items.clone(),
            _ => vec![suite],
        };
        let first = self.cache.ast.first_in_block(suite).ok_or_else(bad)?;
        let code = match self.cache.ast[first].kind.as_expr().and_then(|e| e.expr).map(|e| &self.cache.ast[e].kind) {
            Some(ExprKind::String(s)) => s.value(),
            _ => return Err(bad()),
        };
        let types = self.cache.types.clone();
        let mut literals = vec![];
        for &st in items.iter().skip(1) {
            let e = self.cache.ast[st].kind.as_expr().and_then(|e| e.expr).ok_or_else(|| CompilerError::custom("invalid LLVM type argument", src))?;
            let t = self.ety(e)?;
            if let Some(i) = types.get_int_static(t) {
                literals.push(Generic::Static(i));
            } else if let Some(s) = types.get_str_static(t) {
                literals.push(Generic::StaticStr(s));
            } else {
                let et = self.extract_type(t);
                literals.push(Generic::Type(self.get_type(et)?));
            }
        }
        let mut is_declare = true;
        let mut declare = String::new();
        let mut lines: Vec<String> = vec![];
        for l in code.lines() {
            let lp = l.trim();
            if is_declare && !lp.starts_with("declare ") && !lp.starts_with('@') {
                let is_const = lp.contains("private constant");
                if !is_const {
                    is_declare = false;
                    if !lp.is_empty() && !lp.ends_with(':') {
                        lines.push("entry:".into());
                    }
                }
            }
            if is_declare {
                declare.push_str(lp);
                declare.push('\n');
            } else {
                lines.push(l.to_string());
            }
        }
        if let Some(fd) = self.module.borrow_mut().func_mut(func) {
            fd.kind = FuncKind::Llvm { literals, declarations: declare, body: lines.join("\n") };
        }
        let _ = ValueKind::Any { flow: false };
        Ok(())
    }
}
