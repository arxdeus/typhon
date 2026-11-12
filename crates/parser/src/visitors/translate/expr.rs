//! Expression translation (`TranslateVisitor::visit(XExpr*)`).

use std::collections::BTreeSet;

use typhon_common::mangle::{self, stdlib};
use typhon_common::{CompilerError, Result};
use typhon_ir::attribute::{KeyValuePair, LiteralElement};
use typhon_ir::{Attribute, CallInstr, ExtractInstr, Intrinsic, Stage, TemplatedConst, ValueId, ValueKind};

use crate::ast::{Attr, Child, ExprId, ExprKind, ParamStatus, YieldStmt};
use crate::types::class::partial_flag;
use crate::visitors::visitor::{expr_children, stmt_children};

use super::{TranslateItem, TranslateVisitor};

impl TranslateVisitor<'_> {
    fn err<T>(&self, msg: impl Into<String>) -> Result<T> {
        Err(CompilerError::custom(msg, self.src))
    }

    /// `transform(Expr*)` with literal/partial attribute bookkeeping.
    pub fn transform_expr(&mut self, e: ExprId) -> Result<ValueId> {
        let old_src = std::mem::replace(&mut self.src, self.cache.ast[e].src);
        let is_list = self.has_attr_e(e, Attr::ExprList);
        let is_set = self.has_attr_e(e, Attr::ExprSet);
        let is_dict = self.has_attr_e(e, Attr::ExprDict);
        let is_partial = self.has_attr_e(e, Attr::ExprPartial);
        if is_list || is_set || is_dict || is_partial {
            self.ctx.borrow_mut().seq_items.push(vec![]);
        }
        let p = if is_partial { self.cache.types.get_partial(self.ety(e)?) } else { None };
        let r = self.visit_expr(e);
        self.src = old_src;
        let ir = r?;
        let mut m_attrs: Vec<Attribute> = vec![];
        if is_list || is_set {
            let items = self.ctx.borrow_mut().seq_items.pop().unwrap_or_default();
            let le: Vec<LiteralElement> =
                items.iter().map(|(a, v)| LiteralElement { value: *v, star: *a == Attr::ExprStarSequenceItem }).collect();
            m_attrs.push(if is_list { Attribute::ListLiteral(le) } else { Attribute::SetLiteral(le) });
        }
        if is_dict {
            let items = self.ctx.borrow_mut().seq_items.pop().unwrap_or_default();
            let mut dla = vec![];
            let mut pi = 0;
            while pi < items.len() {
                if items[pi].0 == Attr::ExprStarSequenceItem {
                    dla.push(KeyValuePair { key: items[pi].1, value: None });
                } else {
                    let v = items.get(pi + 1).map(|x| x.1);
                    typhon_common::seqassert!(v.is_some(), "invalid dict element");
                    dla.push(KeyValuePair { key: items[pi].1, value: v });
                    pi += 1;
                }
                pi += 1;
            }
            m_attrs.push(Attribute::DictLiteral(dla));
        }
        if is_partial {
            let items = self.ctx.borrow_mut().seq_items.pop().unwrap_or_default();
            let p = match p {
                Some(p) => p,
                None => return self.err("invalid partial element"),
            };
            let types = self.cache.types.clone();
            let known = types.partial_mask(p);
            let func = types.partial_func(p);
            let info = types.func_ast(types.func_ast_id(func));
            let mut vals = vec![];
            let mut j = 0;
            for (i, k) in known.bytes().enumerate() {
                let is_value = info.params.get(i).is_some_and(|x| x.1 == ParamStatus::Value);
                if k == partial_flag::INCLUDED && is_value {
                    vals.push(items.get(j).map(|x| x.1));
                    j += 1;
                } else if is_value {
                    vals.push(None);
                }
            }
            m_attrs.push(Attribute::PartialFunction { name: info.name.clone(), args: vals });
        }
        {
            let mut m = self.module.borrow_mut();
            for a in m_attrs {
                m.set_attribute(ir.into(), a);
            }
        }
        if self.has_attr_e(e, Attr::ExprSequenceItem) {
            if let Some(b) = self.ctx.borrow_mut().seq_items.last_mut() {
                b.push((Attr::ExprSequenceItem, ir));
            }
        }
        if self.has_attr_e(e, Attr::ExprStarSequenceItem) {
            if let Some(b) = self.ctx.borrow_mut().seq_items.last_mut() {
                b.push((Attr::ExprStarSequenceItem, ir));
            }
        }
        Ok(ir)
    }

    pub(crate) fn tr_opt(&mut self, e: Option<ExprId>) -> Result<ValueId> {
        match e {
            Some(e) => self.transform_expr(e),
            None => self.err("missing expression"),
        }
    }

    fn call_func(&mut self, f: typhon_ir::FuncId, args: Vec<ValueId>) -> ValueId {
        let src = self.src;
        let fv = self.make(src, ValueKind::VarValue(typhon_ir::VarValue { var: f }));
        self.make(src, ValueKind::CallInstr(CallInstr { callee: fv, args }))
    }

    fn visit_expr(&mut self, e: ExprId) -> Result<ValueId> {
        let src = self.src;
        let kind = self.cache.ast[e].kind.clone();
        let types = self.cache.types.clone();
        match kind {
            ExprKind::None(_) => {
                let t = self.ety(e)?;
                let f = if types.is(t, &stdlib().optional) {
                    format!("{}:{}", types.realized_name(t), mangle::mangled_method("", "Optional", "__new__", 0, 0))
                } else {
                    Self::none_new_name()
                };
                let f = self.force_find(&f)?.func().ok_or_else(|| CompilerError::custom("not a function", src))?;
                Ok(self.call_func(f, vec![]))
            }
            ExprKind::Bool(b) => {
                let t = self.get_type(self.ety(e)?)?;
                Ok(self.make(src, ValueKind::BoolConst(TemplatedConst { val: b.value, ty: t })))
            }
            ExprKind::Int(i) => {
                let t = self.get_type(self.ety(e)?)?;
                Ok(self.make(src, ValueKind::IntConst(TemplatedConst { val: i.value(), ty: t })))
            }
            ExprKind::Float(f) => {
                let t = self.get_type(self.ety(e)?)?;
                Ok(self.make(src, ValueKind::FloatConst(TemplatedConst { val: f.value(), ty: t })))
            }
            ExprKind::String(s) => {
                let et = self.ety(e)?;
                let t = self.get_type(et)?;
                let val = s.value();
                Ok(if types.is(et, "bytes") {
                    self.make(src, ValueKind::BytesConst(TemplatedConst { val, ty: t }))
                } else {
                    self.make(src, ValueKind::StringConst(TemplatedConst { val, ty: t }))
                })
            }
            ExprKind::Id(id) => match self.force_find(&id.value)? {
                TranslateItem::Var(v) | TranslateItem::Func(v) => {
                    Ok(self.make(src, ValueKind::VarValue(typhon_ir::VarValue { var: v })))
                }
                TranslateItem::Type(_) => {
                    // Just use NoneType which is {} (same as type)
                    let f = self.force_find(&Self::none_new_name())?.func().ok_or_else(|| CompilerError::custom("not a function", src))?;
                    Ok(self.call_func(f, vec![]))
                }
            },
            ExprKind::If(i) => {
                let cond = self.tr_opt(i.cond)?;
                let t = self.tr_opt(i.ifexpr)?;
                let f = self.tr_opt(i.elsexpr)?;
                Ok(self.make(src, ValueKind::TernaryInstr(typhon_ir::TernaryInstr { cond, true_value: t, false_value: f })))
            }
            ExprKind::Generator(_) => self.visit_generator(e),
            ExprKind::Call(_) => self.visit_call(e),
            ExprKind::Dot(d) => {
                let inner = d.expr.ok_or_else(|| CompilerError::custom("bad dot", src))?;
                if matches!(d.member.as_str(), "__atomic__" | "__elemsize__" | "__contents_atomic__") {
                    let t = self.ety(inner)?;
                    let t = self.extract_type(t);
                    let rn = types.realized_name(t);
                    let ty = self.force_find(&rn)?.ty().ok_or_else(|| CompilerError::custom(format!("{rn} is not a type"), src))?;
                    let kind = match d.member.as_str() {
                        "__atomic__" => Intrinsic::IsAtomic,
                        "__contents_atomic__" => Intrinsic::IsContentAtomic,
                        _ => Intrinsic::Sizeof,
                    };
                    let f = self.module.borrow_mut().get_intrinsic(kind, Some(ty), None);
                    Ok(self.call_func(f, vec![]))
                } else {
                    let it = self.ety(inner)?;
                    let name = types.get_class(it).and_then(|c| types.class_name(c)).map(|s| s.to_string()).unwrap_or_default();
                    let rtti = self.cache.classes.get(&name).is_some_and(|c| {
                        let an = c.ast.and_then(|a| self.cache.ast[a].kind.as_class().map(|x| x.name.clone())).unwrap_or_default();
                        c.has_rtti(&an)
                    });
                    let val = self.transform_expr(inner)?;
                    Ok(self.make(src, ValueKind::ExtractInstr(ExtractInstr { val, field: d.member.clone(), polymorphic: rtti })))
                }
            }
            ExprKind::Yield(_) => {
                let ty = self.get_type(self.ety(e)?)?;
                Ok(self.make(src, ValueKind::YieldInInstr(typhon_ir::YieldInInstr { ty, suspend: true })))
            }
            ExprKind::Pipe(_) => self.visit_pipe(e),
            ExprKind::Await(a) => {
                let inner = a.expr.ok_or_else(|| CompilerError::custom("bad await", src))?;
                let is_gen = types.is(self.ety(inner)?, &stdlib().generator);
                let value = self.transform_expr(inner)?;
                let ty = self.get_type(self.ety(e)?)?;
                Ok(self.make(src, ValueKind::AwaitInstr(typhon_ir::AwaitInstr { value, ty, generator: is_gen })))
            }
            ExprKind::Stmt(se) => {
                let body = self.new_series(src, "body");
                self.ctx.borrow_mut().series.push(body);
                let mut r = Ok(());
                for s in &se.items {
                    if let Err(x) = self.transform_stmt(*s) {
                        r = Err(x);
                        break;
                    }
                }
                self.ctx.borrow_mut().series.pop();
                r?;
                let val = self.tr_opt(se.expr)?;
                Ok(self.make(src, ValueKind::FlowInstr(typhon_ir::FlowInstr { flow: body, val })))
            }
            _ => self.err(format!("invalid node {}", self.cache.ast.expr_to_string(e, -1))),
        }
    }

    /// `IdVisitor`: all identifiers in an expression tree (including statement children).
    fn collect_ids(&self, e: ExprId, out: &mut BTreeSet<String>) {
        let mut stack = vec![Child::Expr(e)];
        while let Some(c) = stack.pop() {
            match c {
                Child::Expr(e) => {
                    if let ExprKind::Id(i) = &self.cache.ast[e].kind {
                        out.insert(i.value.clone());
                    }
                    stack.extend(expr_children(&self.cache.ast, e));
                }
                Child::Stmt(s) | Child::Suite(s) => stack.extend(stmt_children(&self.cache.ast, s)),
            }
        }
    }

    fn visit_generator(&mut self, e: ExprId) -> Result<ValueId> {
        let src = self.src;
        let main_ctx = self.cache.imports.get(crate::cache::MAIN_IMPORT).and_then(|i| i.ctx.clone());
        let name = match main_ctx {
            Some(c) => {
                let c = c.borrow();
                c.generate_canonical_name(self.cache, "_generator", false, false)
            }
            None => {
                let n = self.cache.identifier_count.entry("_generator".into()).or_insert(0);
                *n += 1;
                format!("_generator.{}", *n - 1)
            }
        };
        let f = {
            let mut m = self.module.borrow_mut();
            let f = m.nr_bodied_func(name.clone());
            m.set_global(f, true);
            if let Some(fd) = m.func_mut(f) {
                fd.generator = true;
            }
            f
        };
        // Note: the reference implementation iterates an unordered_set; a sorted set is used for determinism.
        let mut ids = BTreeSet::new();
        self.collect_ids(e, &mut ids);
        let mut names = vec![];
        let mut tys = vec![];
        let mut items = vec![];
        for i in ids {
            if let Some(TranslateItem::Var(v)) = self.find(&i) {
                let (global, ty) = {
                    let m = self.module.borrow();
                    (m.var(v).global, m.var(v).ty)
                };
                if !global {
                    if let Some(t) = ty {
                        tys.push(t);
                    }
                    names.push(i);
                    items.push(self.make(src, ValueKind::VarValue(typhon_ir::VarValue { var: v })));
                }
            }
        }
        let rn = self.cache.types.realized_name(self.ety(e)?);
        let ret = self.force_find(&rn)?.ty().ok_or_else(|| CompilerError::custom("not a type", src))?;
        {
            let mut m = self.module.borrow_mut();
            let t = m.unsafe_get_func_type(&name, ret, tys, false);
            m.realize_func(f, t, &names);
        }
        self.ctx.borrow_mut().base.add_block();
        for n in &names {
            let av = self.module.borrow().func_arg_var(f, n);
            if let Some(av) = av {
                self.ctx.borrow_mut().add(n, TranslateItem::Var(av));
            }
        }
        let body = self.new_series(src, "body");
        {
            let mut c = self.ctx.borrow_mut();
            c.bases.push(f);
            c.series.push(body);
        }
        let fe = self.cache.ast.generator_final_expr(e);
        let ys = self.cache.ast.new_stmt(YieldStmt { expr: fe }, src);
        self.cache.ast.generator_set_final_stmt(e, ys);
        let suite = self.cache.ast.generator_final_suite(e);
        let r = match suite {
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
        self.module.borrow_mut().set_func_body(f, Some(body));
        let fv = self.make(src, ValueKind::VarValue(typhon_ir::VarValue { var: f }));
        Ok(self.make(src, ValueKind::CallInstr(CallInstr { callee: fv, args: items })))
    }

    fn visit_call(&mut self, e: ExprId) -> Result<ValueId> {
        let src = self.src;
        let types = self.cache.types.clone();
        let call = self.cache.ast[e].kind.as_call().cloned().unwrap();
        let callee_e = call.expr.ok_or_else(|| CompilerError::custom("bad call", src))?;
        let ei = self.cache.ast[callee_e].kind.as_id().map(|i| i.value.clone());
        if ei.as_deref() == Some(mangle::func("", "__ptr__").as_str()) {
            let mut head = call.items[0].value.ok_or_else(|| CompilerError::custom("bad __ptr__", src))?;
            let pre = self.transform_expr(head)?;
            let pre = self.module.borrow().as_flow_instr(pre).is_some().then_some(pre);
            while let ExprKind::Stmt(s) = &self.cache.ast[head].kind {
                head = s.expr.unwrap();
            }
            let mut members = vec![];
            while let ExprKind::Dot(d) = &self.cache.ast[head].kind {
                members.push(d.member.clone());
                head = d.expr.unwrap();
            }
            members.reverse();
            let key = match &self.cache.ast[head].kind {
                ExprKind::Id(i) => i.value.clone(),
                _ => return self.err("expected IdExpr"),
            };
            let var = self.find(&key).and_then(|i| i.var()).ok_or_else(|| CompilerError::custom(format!("{key} is not a variable"), src))?;
            let pv = self.make(src, ValueKind::PointerValue(typhon_ir::PointerValue { var, fields: members }));
            if let Some(pre) = pre {
                if let Some(fi) = self.module.borrow_mut().as_flow_instr_mut(pre) {
                    fi.val = pv;
                }
                return Ok(pre);
            }
            return Ok(pv);
        } else if ei.as_deref() == Some(mangle::mangled_method("", "__array__", "__new__", 0, 0).as_str()) {
            let fnt = types.get_func(self.ety(callee_e)?).unwrap();
            let sz = types.get_int_static(types.func_generics(fnt)[0].ty.unwrap()).unwrap_or(0);
            let parent = types.func_parent(fnt).and_then(|p| types.get_class(p)).unwrap();
            let typ = types.generic_type(parent, 0).unwrap();
            let base = self.get_type(typ)?;
            let ast_t = self.ety(e)?;
            let ptr = {
                let mut m = self.module.borrow_mut();
                let p = m.unsafe_get_pointer_type(Some(base));
                m.set_ast_type(p, Some(crate::cache::AstType::to_ref(ast_t)));
                p
            };
            return Ok(self.make(src, ValueKind::StackAllocInstr(typhon_ir::StackAllocInstr { ptr_type: ptr, count: sz })));
        } else if ei.as_deref().is_some_and(|s| s.starts_with(&mangle::mangled_method("", "Generator", "_yield_in_no_suspend", 0, 0))) {
            let ty = self.get_type(self.ety(e)?)?;
            return Ok(self.make(src, ValueKind::YieldInInstr(typhon_ir::YieldInInstr { ty, suspend: false })));
        }

        let ft = types.get_func(self.ety(callee_e)?).ok_or_else(|| CompilerError::custom("not calling function", src))?;
        let callee = self.transform_expr(callee_e)?;
        let fast = types.func_ast_id(ft);
        let is_variadic = self.cache.ast[fast].attrs.has(Attr::CVarArg);
        let mut items = vec![];
        let n = call.items.len();
        for (i, a) in call.items.iter().enumerate() {
            let v = a.value.ok_or_else(|| CompilerError::custom("missing argument", src))?;
            if i + 1 == n && is_variadic {
                let inner = self.cache.ast[v].kind.as_call().cloned().ok_or_else(|| CompilerError::custom("expected *args tuple", src))?;
                for arg in inner.items {
                    items.push(self.tr_opt(arg.value)?);
                }
            } else {
                items.push(self.transform_expr(v)?);
            }
        }
        let mut thunk_id = 0usize;
        let id = self.cache.ast[e].attrs.get_int(Attr::ExprVirtual).or_else(|| self.cache.ast[e].attrs.has(Attr::ExprVirtual).then_some(-1));
        if let Some(id) = id {
            if id != -1 {
                thunk_id = id as usize;
            } else {
                let tc = self.cache.type_ctx.clone();
                let tv = crate::typecheck::TypecheckVisitor::new(self.cache, tc, None, None);
                if tv.is_polymorphic(ft) {
                    let sig = tv.vtable_signature(ft);
                    match tv.cache.thunk_ids.get(&sig) {
                        Some(k) => thunk_id = *k,
                        None => return Err(CompilerError::custom(format!("not a valid thunk: {sig}"), src)),
                    }
                }
            }
        }
        if thunk_id != 0 {
            let base = self.ctx.borrow().get_base();
            let mut m = self.module.borrow_mut();
            let r = typhon_ir::util::irtools::virtual_call(&mut m, callee, items, thunk_id, base);
            m.set_src_info(r.into(), src);
            Ok(r)
        } else {
            Ok(self.make(src, ValueKind::CallInstr(CallInstr { callee, args: items })))
        }
    }

    fn is_gen(&self, v: ValueId) -> bool {
        let mut m = self.module.borrow_mut();
        let t = m.value_type_mut(v);
        match &m.ty(t).kind {
            typhon_ir::TypeKind::Generator { .. } => true,
            typhon_ir::TypeKind::Func { ret, .. } => matches!(m.ty(*ret).kind, typhon_ir::TypeKind::Generator { .. }),
            _ => false,
        }
    }

    /// Note: the reference implementation pushes a non-trivial `PipelineFlow` into the current series and
    /// yields a null result; here the flow is returned and the enclosing statement pushes it.
    fn visit_pipe(&mut self, e: ExprId) -> Result<ValueId> {
        let src = self.src;
        let pipe = self.cache.ast[e].kind.as_pipe().cloned().unwrap();
        let first = self.tr_opt(pipe.items[0].expr)?;
        let first_gen = self.is_gen(first);
        let mut stages = vec![Stage { callee: first, args: vec![], generator: first_gen, parallel: false }];
        let mut simple = !first_gen;
        for i in 1..pipe.items.len() {
            let ce = pipe.items[i].expr.unwrap();
            let call = self.cache.ast[ce].kind.as_call().cloned().ok_or_else(|| CompilerError::custom("pipe stage is not a call", src))?;
            let f = self.tr_opt(call.expr)?;
            let g = self.is_gen(f);
            if i + 1 != pipe.items.len() {
                simple &= !g;
            }
            let mut args = vec![];
            for a in &call.items {
                let v = a.value.unwrap();
                if matches!(self.cache.ast[v].kind, ExprKind::Ellipsis(_)) {
                    args.push(None);
                } else {
                    args.push(Some(self.transform_expr(v)?));
                }
            }
            stages.push(Stage { callee: f, args, generator: g, parallel: false });
        }
        if simple {
            let mut cv = typhon_ir::util::cloning::CloneVisitor::new(true);
            let mut m = self.module.borrow_mut();
            let mut result = cv.clone_value(&mut m, stages[0].callee);
            for st in &stages[1..] {
                let mut new_args = vec![];
                for a in &st.args {
                    new_args.push(match a {
                        Some(a) => cv.clone_value(&mut m, *a),
                        None => result,
                    });
                }
                let callee = cv.clone_value(&mut m, st.callee);
                result = m.n(src, ValueKind::CallInstr(CallInstr { callee, args: new_args }));
            }
            Ok(result)
        } else {
            for (i, p) in pipe.items.iter().enumerate() {
                if p.op == "||>" {
                    stages[i].parallel = true;
                }
            }
            Ok(self.make(src, ValueKind::PipelineFlow(typhon_ir::PipelineFlow { stages })))
        }
    }
}
