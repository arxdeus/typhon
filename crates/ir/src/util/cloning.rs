//! Deep cloning of IR nodes.

use rustc_hash::{FxHashMap, FxHashSet};

use crate::attribute::Attribute;
use crate::ids::{FuncId, Id, NodeRef, ValueId, VarId};
use crate::module::Module;
use crate::util::operator::{Operator, OperatorState};
use crate::value::*;
use crate::var::FuncKind;

/// Port of anonymous `GatherLocals` (collects non-global variables used in a value tree).
struct GatherLocals {
    state: OperatorState,
    locals: Vec<VarId>,
    seen: FxHashSet<Id>,
}

impl Operator for GatherLocals {
    fn state(&self) -> &OperatorState {
        &self.state
    }
    fn state_mut(&mut self) -> &mut OperatorState {
        &mut self.state
    }
    fn pre_hook(&mut self, m: &mut Module, n: NodeRef) {
        for v in m.used_variables(n) {
            // Note: the reference implementation collects duplicates and creates one clone per occurrence
            // (only the last remap survives); duplicates are skipped here so that no
            // unused clones are added to the target function.
            if !m.is_global(v) && self.seen.insert(m.var_uid(v)) {
                self.locals.push(v);
            }
        }
    }
}

pub struct CloneVisitor {
    ctx: FxHashMap<Id, NodeRef>,
    clone_loop: bool,
}

impl Default for CloneVisitor {
    fn default() -> Self {
        Self::new(true)
    }
}

impl CloneVisitor {
    /// `CloneVisitor(module, cloneLoop)`. The module is passed to each call instead.
    pub fn new(clone_loop: bool) -> Self {
        Self { ctx: FxHashMap::default(), clone_loop }
    }

    /// `forceRemap(original, newVal)`.
    pub fn force_remap(&mut self, m: &Module, original: NodeRef, new: NodeRef) {
        let uid = m.node_uid(original);
        self.ctx.insert(uid, new);
    }

    /// `clone(const Value*)`.
    pub fn clone_value(&mut self, m: &mut Module, other: ValueId) -> ValueId {
        self.clone_value_into(m, other, None, &FxHashMap::default())
    }
    /// `clone(const Value*)` with a nullable argument.
    pub fn clone_value_opt(&mut self, m: &mut Module, other: Option<ValueId>) -> Option<ValueId> {
        other.map(|o| self.clone_value(m, o))
    }

    /// `clone(const Value *other, BodiedFunc *cloneTo, remaps)`.
    pub fn clone_value_into(
        &mut self,
        m: &mut Module,
        other: ValueId,
        clone_to: Option<FuncId>,
        remaps: &FxHashMap<Id, VarId>,
    ) -> ValueId {
        if let Some(to) = clone_to {
            let mut gl = GatherLocals { state: OperatorState::new(false), locals: Vec::new(), seen: FxHashSet::default() };
            gl.process(m, other.into());
            for v in gl.locals {
                let uid = m.var_uid(v);
                if let Some(r) = remaps.get(&uid) {
                    self.ctx.insert(uid, NodeRef::Var(*r));
                } else {
                    let d = m.var(v);
                    let (ty, g, e, t, name) = (d.ty, d.global, d.external, d.tls, d.base.name.clone());
                    let src = m.src_info(v.into());
                    let cloned = m.n_var(src, ty, g, e, t, name);
                    m.func_push_symbol(to, cloned);
                    self.ctx.insert(uid, NodeRef::Var(cloned));
                }
            }
        } else {
            for (uid, var) in remaps {
                self.ctx.insert(*uid, NodeRef::Var(*var));
            }
        }
        let id = m.value_uid(other);
        if !self.ctx.contains_key(&id) {
            let result = self.visit_value(m, other);
            self.ctx.insert(id, NodeRef::Value(result));
            self.clone_attributes(m, other.into(), result.into(), false);
        }
        self.ctx[&id].as_value().expect("clone context maps a value to a non-value")
    }

    /// `clone(const Var*)`: remapped variable or the original.
    pub fn clone_var(&mut self, m: &Module, other: VarId) -> VarId {
        let id = m.var_uid(other);
        match self.ctx.get(&id) {
            Some(NodeRef::Var(v)) => *v,
            _ => other,
        }
    }
    pub fn clone_var_opt(&mut self, m: &Module, other: Option<VarId>) -> Option<VarId> {
        other.map(|o| self.clone_var(m, o))
    }

    /// `forceClone(const Value*)`.
    pub fn force_clone_value(&mut self, m: &mut Module, other: ValueId) -> ValueId {
        let id = m.value_uid(other);
        if !self.ctx.contains_key(&id) {
            let result = self.visit_value(m, other);
            self.ctx.insert(id, NodeRef::Value(result));
            self.clone_attributes(m, other.into(), result.into(), true);
        }
        self.ctx[&id].as_value().expect("clone context maps a value to a non-value")
    }

    /// `forceClone(const Var*)` / `forceClone(const Func*)`.
    pub fn force_clone_var(&mut self, m: &mut Module, other: VarId) -> VarId {
        let id = m.var_uid(other);
        if !self.ctx.contains_key(&id) {
            let result = self.visit_var(m, other);
            self.ctx.insert(id, NodeRef::Var(result));
            self.clone_attributes(m, other.into(), result.into(), true);
        }
        self.ctx[&id].as_var().expect("clone context maps a variable to a non-variable")
    }

    /// `clone(const PipelineFlow::Stage&)`.
    pub fn clone_stage(&mut self, m: &mut Module, s: &Stage) -> Stage {
        let args = s.args.iter().map(|a| self.clone_value_opt(m, *a)).collect();
        Stage { callee: self.clone_value(m, s.callee), args, generator: s.generator, parallel: s.parallel }
    }

    fn clone_attributes(&mut self, m: &mut Module, from: NodeRef, to: NodeRef, force: bool) {
        let attrs: Vec<(i32, Attribute)> = m.attributes(from).iter().cloned().collect();
        for (key, attr) in attrs {
            if !attr.needs_clone() {
                continue;
            }
            let new = if attr.holds_values() {
                attr.map_values(&mut |x| if force { self.force_clone_value(m, x) } else { self.clone_value(m, x) })
            } else {
                attr
            };
            m.set_attribute_key(to, new, key);
        }
    }

    /// `Nt(source, ...)`: new value with the source's src info and name.
    fn nt(&mut self, m: &mut Module, source: ValueId, kind: ValueKind) -> ValueId {
        let src = m.src_info(source.into());
        let name = m.value_name(source).to_string();
        m.n_named(src, kind, name)
    }

    fn clone_schedule(&mut self, m: &mut Module, s: &Option<Box<crate::schedule::OMPSched>>) -> Option<Box<crate::schedule::OMPSched>> {
        let s = s.as_ref()?;
        let mut cloned = s.clone();
        for val in s.used_values() {
            let uid = m.value_uid(val);
            let c = self.clone_value(m, val);
            cloned.replace_used_value(m, uid, Some(c));
        }
        Some(cloned)
    }

    fn visit_value(&mut self, m: &mut Module, v: ValueId) -> ValueId {
        let v = m.actual_value(v);
        let kind = m.kind(v).clone();
        match kind {
            ValueKind::VarValue(x) => {
                let var = self.clone_var(m, x.var);
                self.nt(m, v, ValueKind::VarValue(VarValue { var }))
            }
            ValueKind::PointerValue(x) => {
                let var = self.clone_var(m, x.var);
                self.nt(m, v, ValueKind::PointerValue(PointerValue { var, fields: x.fields }))
            }
            ValueKind::SeriesFlow(f) => {
                let res = self.nt(m, v, ValueKind::SeriesFlow(SeriesFlow::default()));
                for c in f.series {
                    let cc = self.clone_value(m, c);
                    m.series_push(res, cc);
                }
                res
            }
            ValueKind::IfFlow(f) => {
                let cond = self.clone_value(m, f.cond);
                let t = self.clone_value(m, f.true_branch);
                let e = self.clone_value_opt(m, f.false_branch);
                self.nt(m, v, ValueKind::IfFlow(IfFlow { cond, true_branch: t, false_branch: e }))
            }
            ValueKind::WhileFlow(f) => {
                let lp = self.nt(m, v, ValueKind::WhileFlow(WhileFlow { cond: f.cond, body: f.body }));
                self.force_remap(m, v.into(), lp.into());
                let cond = self.clone_value(m, f.cond);
                let body = self.clone_value(m, f.body);
                let w = m.as_while_flow_mut(lp).unwrap();
                w.cond = cond;
                w.body = body;
                lp
            }
            ValueKind::ForFlow(f) => {
                let lp = self.nt(
                    m,
                    v,
                    ValueKind::ForFlow(ForFlow { iter: f.iter, body: f.body, var: f.var, schedule: None, is_async: false }),
                );
                self.force_remap(m, v.into(), lp.into());
                let iter = self.clone_value(m, f.iter);
                let body = self.clone_value(m, f.body);
                let var = self.clone_var(m, f.var);
                let schedule = self.clone_schedule(m, &f.schedule);
                let x = m.as_for_flow_mut(lp).unwrap();
                x.iter = iter;
                x.body = body;
                x.var = var;
                x.schedule = schedule;
                x.is_async = f.is_async;
                lp
            }
            ValueKind::ImperativeForFlow(f) => {
                let lp = self.nt(
                    m,
                    v,
                    ValueKind::ImperativeForFlow(ImperativeForFlow {
                        start: f.start,
                        step: f.step,
                        end: f.end,
                        body: f.body,
                        var: f.var,
                        schedule: None,
                    }),
                );
                self.force_remap(m, v.into(), lp.into());
                let start = self.clone_value(m, f.start);
                let body = self.clone_value(m, f.body);
                let var = self.clone_var(m, f.var);
                let end = self.clone_value(m, f.end);
                let schedule = self.clone_schedule(m, &f.schedule);
                let x = m.as_imperative_for_flow_mut(lp).unwrap();
                x.start = start;
                x.body = body;
                x.var = var;
                x.end = end;
                x.schedule = schedule;
                lp
            }
            ValueKind::TryCatchFlow(f) => {
                let body = self.clone_value(m, f.body);
                let finally = self.clone_value_opt(m, f.finally);
                let else_ = self.clone_value_opt(m, f.else_);
                let res = self.nt(m, v, ValueKind::TryCatchFlow(TryCatchFlow { catches: Vec::new(), body, else_, finally }));
                for c in f.catches {
                    let h = self.clone_value(m, c.handler);
                    let var = self.clone_var_opt(m, c.var);
                    m.try_catch_push(res, h, c.ty, var);
                }
                res
            }
            ValueKind::PipelineFlow(f) => {
                let stages = f.stages.iter().map(|s| self.clone_stage(m, s)).collect();
                self.nt(m, v, ValueKind::PipelineFlow(PipelineFlow { stages }))
            }
            k @ (ValueKind::IntConst(_)
            | ValueKind::FloatConst(_)
            | ValueKind::BoolConst(_)
            | ValueKind::StringConst(_)
            | ValueKind::BytesConst(_)
            | ValueKind::StackAllocInstr(_)
            | ValueKind::YieldInInstr(_)
            | ValueKind::Any { .. }) => self.nt(m, v, k),
            ValueKind::AssignInstr(i) => {
                let lhs = self.clone_var(m, i.lhs);
                let rhs = self.clone_value(m, i.rhs);
                self.nt(m, v, ValueKind::AssignInstr(AssignInstr { lhs, rhs }))
            }
            ValueKind::ExtractInstr(i) => {
                let val = self.clone_value(m, i.val);
                self.nt(m, v, ValueKind::ExtractInstr(ExtractInstr { val, field: i.field, polymorphic: i.polymorphic }))
            }
            ValueKind::InsertInstr(i) => {
                let lhs = self.clone_value(m, i.lhs);
                let rhs = self.clone_value(m, i.rhs);
                self.nt(
                    m,
                    v,
                    ValueKind::InsertInstr(InsertInstr { lhs, field: i.field, rhs, polymorphic: i.polymorphic }),
                )
            }
            ValueKind::CallInstr(i) => {
                let args: Vec<ValueId> = i.args.iter().map(|a| self.clone_value(m, *a)).collect();
                let callee = self.clone_value(m, i.callee);
                self.nt(m, v, ValueKind::CallInstr(CallInstr { callee, args }))
            }
            ValueKind::TernaryInstr(i) => {
                let cond = self.clone_value(m, i.cond);
                let t = self.clone_value(m, i.true_value);
                let f = self.clone_value(m, i.false_value);
                self.nt(m, v, ValueKind::TernaryInstr(TernaryInstr { cond, true_value: t, false_value: f }))
            }
            ValueKind::BreakInstr(i) => {
                let lp = if self.clone_loop { self.clone_value_opt(m, i.loop_) } else { i.loop_ };
                self.nt(m, v, ValueKind::BreakInstr(BreakInstr { loop_: lp }))
            }
            ValueKind::ContinueInstr(i) => {
                let lp = if self.clone_loop { self.clone_value_opt(m, i.loop_) } else { i.loop_ };
                self.nt(m, v, ValueKind::ContinueInstr(ContinueInstr { loop_: lp }))
            }
            ValueKind::ReturnInstr(i) => {
                let value = self.clone_value_opt(m, i.value);
                self.nt(m, v, ValueKind::ReturnInstr(ReturnInstr { value }))
            }
            ValueKind::YieldInstr(i) => {
                let value = self.clone_value_opt(m, i.value);
                self.nt(m, v, ValueKind::YieldInstr(YieldInstr { value, final_: i.final_ }))
            }
            ValueKind::AwaitInstr(i) => {
                let value = self.clone_value(m, i.value);
                self.nt(m, v, ValueKind::AwaitInstr(AwaitInstr { value, ty: i.ty, generator: i.generator }))
            }
            ValueKind::ThrowInstr(i) => {
                let value = self.clone_value_opt(m, i.value);
                let res = self.nt(m, v, ValueKind::ThrowInstr(ThrowInstr { value, fallback: None }));
                let fb = self.clone_value_opt(m, i.fallback);
                m.set_throw_fallback(res, fb);
                res
            }
            ValueKind::FlowInstr(i) => {
                let flow = self.clone_value(m, i.flow);
                let val = self.clone_value(m, i.val);
                self.nt(m, v, ValueKind::FlowInstr(FlowInstr { flow, val }))
            }
        }
    }

    fn visit_var(&mut self, m: &mut Module, v: VarId) -> VarId {
        let v = m.actual_var(v);
        let d = m.var(v).clone();
        let src = m.src_info(v.into());
        let Some(fd) = d.func.as_deref() else {
            return m.n_var(src, d.ty, d.global, d.external, d.tls, d.base.name.clone());
        };
        let res = m.n_func(src, d.base.name.clone(), match &fd.kind {
            FuncKind::Bodied { .. } => FuncKind::bodied(),
            FuncKind::External => FuncKind::External,
            FuncKind::Internal { .. } => FuncKind::internal(),
            FuncKind::Llvm { .. } => FuncKind::llvm(),
        });
        let arg_names: Vec<String> = fd.args.iter().map(|a| m.var_name(*a).to_string()).collect();
        if let FuncKind::Internal { intrinsic, intrinsic_type } = &fd.kind {
            m.set_intrinsic(res, *intrinsic, *intrinsic_type);
        }
        if let FuncKind::Bodied { symbols, .. } = &fd.kind {
            for s in symbols {
                let nv = self.force_clone_var(m, *s);
                m.func_push_symbol(res, nv);
            }
        }
        {
            let rf = m.func_mut(res).unwrap();
            rf.unmangled_name = fd.unmangled_name.clone();
            rf.generator = fd.generator;
            rf.is_async = fd.is_async;
        }
        let ty = d.ty.expect("function has no type");
        m.realize_func(res, ty, &arg_names);
        let new_args = m.func_args(res);
        for (a, b) in fd.args.iter().zip(new_args) {
            self.force_remap(m, (*a).into(), b.into());
        }
        match &fd.kind {
            FuncKind::Bodied { body, jit, .. } => {
                self.force_remap(m, v.into(), res.into());
                if let Some(b) = body {
                    let nb = self.clone_value(m, *b);
                    m.set_func_body(res, Some(nb));
                }
                m.func_mut(res).unwrap().set_jit(*jit);
            }
            FuncKind::Internal { .. } => {
                m.func_mut(res).unwrap().parent_type = fd.parent_type;
            }
            FuncKind::Llvm { literals, declarations, body } => {
                let rf = m.func_mut(res).unwrap();
                rf.set_llvm_body(body.clone());
                rf.set_llvm_declarations(declarations.clone());
                rf.set_llvm_literals(literals.clone());
            }
            FuncKind::External => {}
        }
        res
    }
}
