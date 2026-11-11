//! Port of codon/cir/util/inlining.{h,cpp}.

use typhon_common::SrcInfo;
use typhon_ir::util::cloning::CloneVisitor;
use typhon_ir::util::irtools::get_func;
use typhon_ir::util::operator::{Operator, OperatorState};
use typhon_ir::{FuncId, Module, NodeRef, TypeKind, ValueId, ValueKind, VarId};
use typhon_ir::{AssignInstr, BreakInstr, FlowInstr, SeriesFlow, VarValue, WhileFlow};

/// Port of `InlineResult`.
#[derive(Clone, Debug, Default)]
pub struct InlineResult {
    /// the inlined value (flow or `FlowInstr`), `None` on failure
    pub result: Option<ValueId>,
    /// variables that must be added to the enclosing function
    pub new_vars: Vec<VarId>,
}

impl InlineResult {
    /// `operator bool()`.
    pub fn is_some(&self) -> bool {
        self.result.is_some()
    }
}

struct ReturnVerifier {
    st: OperatorState,
    need_loop: bool,
}

impl Operator for ReturnVerifier {
    fn state(&self) -> &OperatorState {
        &self.st
    }
    fn state_mut(&mut self) -> &mut OperatorState {
        &mut self.st
    }
    fn handle_return_instr(&mut self, m: &mut Module, v: ValueId) {
        if self.need_loop {
            return;
        }
        let parents = self.parents().to_vec();
        if parents.is_empty() {
            self.need_loop = true;
            return;
        }
        let mut prev: Option<ValueId> = None;
        for p in parents {
            let cur = match p {
                NodeRef::Value(x) if m.as_series_flow(x).is_some() => m.actual_value(x),
                _ => {
                    self.need_loop = true;
                    return;
                }
            };
            if let Some(pr) = prev {
                let back = m.as_series_flow(pr).and_then(|s| s.series.last().copied());
                if back.map(|b| m.value_uid(b)) != Some(m.value_uid(cur)) {
                    self.need_loop = true;
                    return;
                }
            }
            prev = Some(cur);
        }
        let back = prev.and_then(|p| m.as_series_flow(p).and_then(|s| s.series.last().copied()));
        self.need_loop = back.map(|b| m.value_uid(b)) != Some(m.value_uid(v));
    }
}

struct ReturnReplacer<'a> {
    st: OperatorState,
    implicit_loop: Option<ValueId>,
    var: Option<VarId>,
    aggressive: bool,
    cv: &'a mut CloneVisitor,
}

impl Operator for ReturnReplacer<'_> {
    fn state(&self) -> &OperatorState {
        &self.st
    }
    fn state_mut(&mut self) -> &mut OperatorState {
        &mut self.st
    }
    fn handle_return_instr(&mut self, m: &mut Module, v: ValueId) {
        let src = m.src_info(v.into());
        let value = m.as_return_instr(v).and_then(|r| r.value);
        let rep = m.n(src.clone(), ValueKind::SeriesFlow(SeriesFlow::default()));
        if let Some(var) = self.var {
            let val = value.map(|x| self.cv.clone_value(m, x)).expect("return without value in non-void function");
            let a = m.n(src.clone(), ValueKind::AssignInstr(AssignInstr { lhs: var, rhs: val }));
            m.series_push(rep, a);
        } else if let Some(x) = value {
            let c = self.cv.clone_value(m, x);
            m.series_push(rep, c);
        }
        if self.aggressive {
            let b = m.n(src, ValueKind::BreakInstr(BreakInstr { loop_: self.implicit_loop }));
            m.series_push(rep, b);
        }
        m.replace_all_value(v, rep);
    }
}

/// `inlineFunction(func, args, aggressive, callInfo)`.
pub fn inline_function(m: &mut Module, func: FuncId, args: Vec<ValueId>, aggressive: bool, info: SrcInfo) -> InlineResult {
    let func = m.actual_var(func);
    let Some(fd) = m.func(func) else { return InlineResult::default() };
    if !fd.is_bodied() {
        return InlineResult::default();
    }
    let fargs = fd.args.clone();
    let symbols = fd.symbols().to_vec();
    let body = fd.body();
    let Some(fty) = m.var_type(func) else { return InlineResult::default() };
    let ret_ty = match m.type_kind(fty) {
        TypeKind::Func { ret, .. } => *ret,
        _ => return InlineResult::default(),
    };
    if args.len() != fargs.len() {
        return InlineResult::default();
    }
    let Some(body) = body else { return InlineResult::default() };

    let mut cv = CloneVisitor::new(true);
    let name = format!("{}_inlined", m.var_name(func));
    let new_flow = m.n(info.clone(), ValueKind::SeriesFlow(SeriesFlow::default()));
    m.set_name(new_flow.into(), name);
    let mut new_vars = Vec::new();
    for (a, arg) in fargs.iter().zip(args) {
        let nv = cv.force_clone_var(m, *a);
        new_vars.push(nv);
        let c = cv.clone_value(m, arg);
        let asg = m.n(info.clone(), ValueKind::AssignInstr(AssignInstr { lhs: nv, rhs: c }));
        m.series_push(new_flow, asg);
    }
    for s in symbols {
        new_vars.push(cv.force_clone_var(m, s));
    }
    let none = m.get_none_type();
    let ret_val = if !m.type_is(ret_ty, none) {
        let v = m.n_var(info.clone(), Some(ret_ty), false, false, false, "");
        new_vars.push(v);
        Some(v)
    } else {
        None
    };

    let cloned_body = cv.clone_value(m, body);
    let mut rv = ReturnVerifier { st: OperatorState::new(false), need_loop: false };
    rv.reset();
    rv.process(m, cloned_body.into());
    if !aggressive && rv.need_loop {
        return InlineResult::default();
    }
    let mut implicit = None;
    if rv.need_loop {
        let loop_body = m.n(info.clone(), ValueKind::SeriesFlow(SeriesFlow::default()));
        let t = m.get_bool(true);
        let w = m.n(info.clone(), ValueKind::WhileFlow(WhileFlow { cond: t, body: loop_body }));
        m.series_push(loop_body, cloned_body);
        if ret_val.is_none() {
            let b = m.n(info.clone(), ValueKind::BreakInstr(BreakInstr { loop_: Some(w) }));
            m.series_push(loop_body, b);
        }
        implicit = Some(w);
    }
    let mut rr = ReturnReplacer { st: OperatorState::new(false), implicit_loop: implicit, var: ret_val, aggressive: rv.need_loop, cv: &mut cv };
    rr.reset();
    rr.process(m, cloned_body.into());
    m.series_push(new_flow, implicit.unwrap_or(cloned_body));
    if let Some(rv) = ret_val {
        let vv = m.n(info.clone(), ValueKind::VarValue(VarValue { var: rv }));
        let fi = m.n(info, ValueKind::FlowInstr(FlowInstr { flow: new_flow, val: vv }));
        return InlineResult { result: Some(fi), new_vars };
    }
    InlineResult { result: Some(new_flow), new_vars }
}

/// `inlineCall(v, aggressive)`.
pub fn inline_call(m: &mut Module, v: ValueId, aggressive: bool) -> InlineResult {
    let Some(call) = m.as_call_instr(v) else { return InlineResult::default() };
    let (callee, args) = (call.callee, call.args.clone());
    let Some(f) = get_func(m, callee) else { return InlineResult::default() };
    let src = m.src_info(v.into());
    inline_function(m, f, args, aggressive, src)
}
