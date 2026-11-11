//! Port of codon/cir/analyze/module/side_effect.{h,cpp}.

use rustc_hash::FxHashMap;
use typhon_ir::pass::{Analysis, AnalysisResult, PassContext};
use typhon_ir::util::irtools::{get_func, has_attribute};
use typhon_ir::util::side_effect::{NON_PURE_ATTR, NO_CAPTURE_ATTR, NO_SIDE_EFFECT_ATTR, PURE_ATTR, SideEffectStatus};
use typhon_ir::{FuncId, Id, Module, ValueId, ValueKind, VarId};

use crate::dataflow::capture::{CaptureAnalysis, CaptureResult, escapes};
use crate::dataflow::cfg::CFAnalysis;
use crate::dataflow::dominator::DominatorAnalysis;
use crate::dataflow::reaching::RDAnalysis;
use crate::util::walk::ConstWalker;

type Status = SideEffectStatus;

/// Port of `SideEffectResult`.
#[derive(Clone, Debug, Default)]
pub struct SideEffectResult {
    /// node uid (values and functions) -> status
    pub result: FxHashMap<Id, SideEffectStatus>,
}

impl SideEffectResult {
    /// `hasSideEffect(v)`: `true` unless `v` is known to be pure.
    pub fn has_side_effect(&self, m: &Module, v: ValueId) -> bool {
        self.result.get(&m.value_uid(v)).map(|s| *s != Status::Pure).unwrap_or(true)
    }
    pub fn status(&self, uid: Id) -> Option<SideEffectStatus> {
        self.result.get(&uid).copied()
    }
}

impl AnalysisResult for SideEffectResult {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Port of `SideEffectAnalysis`.
#[derive(Clone, Debug)]
pub struct SideEffectAnalysis {
    cap_analysis_key: String,
    global_assignment_has_side_effects: bool,
}

impl SideEffectAnalysis {
    pub const KEY: &'static str = "core-analyses-side-effect";
    pub fn new(cap_analysis_key: impl Into<String>, global_assignment_has_side_effects: bool) -> Self {
        Self { cap_analysis_key: cap_analysis_key.into(), global_assignment_has_side_effects }
    }

    pub fn compute(m: &Module, cr: &CaptureResult, global_assignment_has_side_effects: bool) -> SideEffectResult {
        // VarUseAnalyzer
        let mut var_counts: FxHashMap<Id, i64> = FxHashMap::default();
        let mut var_assign_counts: FxHashMap<Id, i64> = FxHashMap::default();
        ConstWalker::new().walk_module(m, &mut |v| {
            for var in m.used_variables(v.into()) {
                *var_counts.entry(m.var_uid(var)).or_default() += 1;
            }
            if let Some(a) = m.as_assign_instr(v) {
                *var_assign_counts.entry(m.var_uid(a.lhs)).or_default() += 1;
            }
        });
        let mut sea = Analyzer {
            m,
            var_counts,
            var_assign_counts,
            cr,
            global_assignment_has_side_effects,
            result: FxHashMap::default(),
            escape_purity: FxHashMap::default(),
            func_stack: Vec::new(),
            func_status: Status::Pure,
        };
        sea.process_func(m.main_func());
        for v in m.vars() {
            sea.process_var(v);
        }
        SideEffectResult { result: sea.result }
    }
}

impl Analysis for SideEffectAnalysis {
    fn key(&self) -> String {
        Self::KEY.to_string()
    }
    fn run(&mut self, m: &Module, ctx: &dyn PassContext) -> Box<dyn AnalysisResult> {
        // PORT-NOTE: the capture analysis is recomputed when missing from the context.
        let owned;
        let cr = match ctx.get_analysis::<CaptureResult>(&self.cap_analysis_key) {
            Some(c) => c,
            None => {
                let rd = RDAnalysis::compute(m, &CFAnalysis::compute(m));
                let dom = DominatorAnalysis::compute(&rd.cfg_result);
                owned = CaptureAnalysis::compute(m, &rd, &dom);
                &owned
            }
        };
        Box::new(Self::compute(m, cr, self.global_assignment_has_side_effects))
    }
}

fn max(a: Status, b: Status) -> Status {
    a.max(b)
}

struct Analyzer<'a> {
    m: &'a Module,
    var_counts: FxHashMap<Id, i64>,
    var_assign_counts: FxHashMap<Id, i64>,
    cr: &'a CaptureResult,
    global_assignment_has_side_effects: bool,
    result: FxHashMap<Id, Status>,
    escape_purity: FxHashMap<(Id, Id), bool>,
    func_stack: Vec<FuncId>,
    func_status: Status,
}

impl Analyzer<'_> {
    fn status_from_attributes(m: &Module, f: FuncId) -> (Status, bool) {
        let attr = |s: &str| has_attribute(m, f, s);
        if attr(&PURE_ATTR) {
            (Status::Pure, true)
        } else if attr(&NO_SIDE_EFFECT_ATTR) {
            (Status::NoSideEffect, true)
        } else if attr(&NO_CAPTURE_ATTR) {
            (Status::NoCapture, true)
        } else if attr(&NON_PURE_ATTR) {
            (Status::Unknown, true)
        } else {
            (Status::Unknown, false)
        }
    }

    fn set(&mut self, uid: Id, expr: Status, func: Status) {
        self.result.insert(uid, expr);
        self.func_status = max(self.func_status, func);
    }

    fn var_assign_status(&self, var: Option<VarId>) -> (Status, Status) {
        let Some(var) = var else { return (Status::Pure, Status::Pure) };
        let m = self.m;
        let id = m.var_uid(var);
        let c1 = self.var_counts.get(&id).copied().unwrap_or(0);
        let c2 = self.var_assign_counts.get(&id).copied().unwrap_or(0);
        let global = m.is_global(var);
        let used = c1 != c2;
        let default = if global { Status::Unknown } else { Status::NoCapture };
        let s = |b: bool| if b { default } else { Status::Pure };
        if self.global_assignment_has_side_effects || m.is_external(var) {
            (s(used || global), s(global))
        } else {
            (s(used), s(used && global))
        }
    }

    fn handle_var_assign(&mut self, uid: Id, var: VarId, base: Status) {
        let (a, b) = self.var_assign_status(Some(var));
        self.set(uid, max(a, base), b);
    }

    fn process_var(&mut self, v: VarId) -> Status {
        let m = self.m;
        let v = m.actual_var(v);
        if m.is_func(v) {
            return self.process_func(v);
        }
        let uid = m.var_uid(v);
        if let Some(s) = self.result.get(&uid) {
            return *s;
        }
        self.set(uid, Status::Pure, Status::Pure);
        Status::Pure
    }

    fn process_func(&mut self, f: FuncId) -> Status {
        let m = self.m;
        let f = m.actual_var(f);
        let uid = m.var_uid(f);
        if let Some(s) = self.result.get(&uid) {
            return *s;
        }
        let Some(fd) = m.func(f) else { return Status::Unknown };
        if fd.is_bodied() {
            let (s, force) = Self::status_from_attributes(m, f);
            self.set(uid, s, s); // avoid infinite recursion
            let old = self.func_status;
            self.func_status = Status::Pure;
            self.func_stack.push(f);
            self.process(fd.body());
            self.func_stack.pop();
            if force {
                self.func_status = s;
            }
            let fs = self.func_status;
            self.set(uid, fs, Status::Pure);
            self.func_status = old;
        } else if fd.is_internal() {
            let s = m.intrinsic_side_effect_status(f);
            self.set(uid, s, Status::Pure);
        } else {
            let s = Self::status_from_attributes(m, f).0;
            self.set(uid, s, Status::Pure);
        }
        self.result[&uid]
    }

    fn process(&mut self, v: Option<ValueId>) -> Status {
        let Some(v) = v else { return Status::Pure };
        let m = self.m;
        let v = m.actual_value(v);
        let uid = m.value_uid(v);
        if let Some(s) = self.result.get(&uid) {
            return *s;
        }
        self.visit(v, uid);
        *self.result.get(&uid).expect("node not added to results")
    }

    fn visit(&mut self, v: ValueId, uid: Id) {
        let m = self.m;
        let p = Status::Pure;
        match m.kind(v).clone() {
            ValueKind::SeriesFlow(f) => {
                let mut s = p;
                for x in f.series {
                    s = max(s, self.process(Some(x)));
                }
                self.set(uid, s, p);
            }
            ValueKind::IfFlow(f) => {
                let s = max(max(self.process(Some(f.cond)), self.process(Some(f.true_branch))), self.process(f.false_branch));
                self.set(uid, s, p);
            }
            ValueKind::WhileFlow(f) => {
                let s = max(self.process(Some(f.cond)), self.process(Some(f.body)));
                self.set(uid, s, p);
            }
            ValueKind::ForFlow(f) => {
                let mut s = max(self.process(Some(f.iter)), self.process(Some(f.body)));
                for x in f.schedule.as_ref().map(|s| s.used_values()).unwrap_or_default() {
                    s = max(s, self.process(Some(x)));
                }
                self.handle_var_assign(uid, f.var, s);
            }
            ValueKind::ImperativeForFlow(f) => {
                let mut s =
                    max(max(self.process(Some(f.start)), self.process(Some(f.end))), self.process(Some(f.body)));
                for x in f.schedule.as_ref().map(|s| s.used_values()).unwrap_or_default() {
                    s = max(s, self.process(Some(x)));
                }
                self.handle_var_assign(uid, f.var, s);
            }
            ValueKind::TryCatchFlow(f) => {
                let mut s = max(max(self.process(Some(f.body)), self.process(f.else_)), self.process(f.finally));
                let mut call_status = p;
                for c in &f.catches {
                    let (a, b) = self.var_assign_status(c.var);
                    s = max(max(s, a), self.process(Some(c.handler)));
                    call_status = max(call_status, b);
                }
                self.set(uid, s, call_status);
            }
            ValueKind::PipelineFlow(f) => {
                let mut s = p;
                let mut call_status = p;
                for stage in &f.stages {
                    if let Some(func) = get_func(m, stage.callee) {
                        let cs = self.process_func(func);
                        call_status = max(call_status, cs);
                        s = max(s, cs);
                    } else {
                        self.process(Some(stage.callee));
                        call_status = Status::Unknown;
                        s = Status::Unknown;
                    }
                    for a in stage.args.iter().flatten() {
                        s = max(s, self.process(Some(*a)));
                    }
                }
                self.set(uid, s, call_status);
            }
            ValueKind::Custom(c) => {
                if c.category == typhon_ir::dsl::CustomCategory::Const {
                    self.set(uid, p, p);
                } else {
                    let n = c.node.borrow();
                    let (a, b) = (n.side_effect_status(true), n.side_effect_status(false));
                    drop(n);
                    self.set(uid, a, b);
                }
            }
            ValueKind::AssignInstr(a) => {
                let s = self.process(Some(a.rhs));
                self.handle_var_assign(uid, a.lhs, s);
            }
            ValueKind::ExtractInstr(e) => {
                let s = self.process(Some(e.val));
                self.set(uid, s, p);
            }
            ValueKind::InsertInstr(ins) => {
                self.process(Some(ins.lhs));
                self.process(Some(ins.rhs));
                let pure = match self.func_stack.last().copied() {
                    Some(func) => {
                        let fuid = m.var_uid(func);
                        let mut pure = match self.cr.results.get(&fuid) {
                            Some(info) => {
                                !info.iter().any(|i| i.extern_captures || i.modified || !i.arg_captures.is_empty())
                            }
                            // PORT-NOTE: C++ asserts the function is in the capture results.
                            None => false,
                        };
                        if pure {
                            let key = (fuid, m.value_uid(ins.lhs));
                            pure = match self.escape_purity.get(&key) {
                                Some(b) => *b,
                                None => {
                                    let e = escapes(m, func, ins.lhs, self.cr);
                                    let b = !e.any()
                                        || (e.return_captures && !e.extern_captures && e.arg_captures.is_empty());
                                    self.escape_purity.insert(key, b);
                                    b
                                }
                            };
                        }
                        pure
                    }
                    None => false,
                };
                self.set(uid, Status::Unknown, if pure { p } else { Status::Unknown });
            }
            ValueKind::CallInstr(c) => {
                let mut s = self.process(Some(c.callee));
                let mut call_status = Status::Unknown;
                for x in &c.args {
                    s = max(s, self.process(Some(*x)));
                }
                if let Some(f) = get_func(m, c.callee) {
                    call_status = self.process_func(f);
                    s = max(s, call_status);
                } else {
                    s = Status::Unknown;
                }
                self.set(uid, s, call_status);
            }
            ValueKind::YieldInInstr(_) | ValueKind::BreakInstr(_) | ValueKind::ContinueInstr(_) => {
                self.set(uid, Status::NoCapture, p)
            }
            ValueKind::TernaryInstr(t) => {
                let s = max(
                    max(self.process(Some(t.cond)), self.process(Some(t.true_value))),
                    self.process(Some(t.false_value)),
                );
                self.set(uid, s, p);
            }
            ValueKind::ReturnInstr(r) => {
                let s = max(Status::NoCapture, self.process(r.value));
                self.set(uid, s, p);
            }
            ValueKind::YieldInstr(r) => {
                let s = max(Status::NoCapture, self.process(r.value));
                self.set(uid, s, p);
            }
            ValueKind::AwaitInstr(r) => {
                let s = max(Status::NoCapture, self.process(Some(r.value)));
                self.set(uid, s, p);
            }
            ValueKind::ThrowInstr(r) => {
                self.process(r.value);
                self.set(uid, Status::Unknown, Status::NoCapture);
            }
            ValueKind::FlowInstr(f) => {
                let s = max(self.process(Some(f.flow)), self.process(Some(f.val)));
                self.set(uid, s, p);
            }
            // consts, VarValue, PointerValue, StackAlloc, Any
            _ => self.set(uid, p, p),
        }
    }
}
