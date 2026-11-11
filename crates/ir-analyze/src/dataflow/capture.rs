//! Port of codon/cir/analyze/dataflow/capture.{h,cpp}: capture / escape analysis.

use std::cell::RefCell;
use std::rc::Rc;

use indexmap::IndexSet;
use rustc_hash::{FxHashMap, FxHashSet};
use typhon_ir::pass::{Analysis, AnalysisResult, PassContext};
use typhon_ir::util::irtools::{get_func, has_attribute, return_type};
use typhon_ir::util::side_effect::{DERIVES_ATTR, NO_CAPTURE_ATTR, NO_SIDE_EFFECT_ATTR, PURE_ATTR, SELF_CAPTURES_ATTR};
use typhon_ir::{FuncId, Id, Module, TypeId, ValueId, ValueKind, VarId};

use crate::dataflow::cfg::{BlockId, CFAnalysis, CFGraph, CfValue, CfVar, SyntheticAssignKind};
use crate::dataflow::dominator::{DominatorAnalysis, DominatorResult};
use crate::dataflow::reaching::{RDAnalysis, RDInspector, RDResult};
use crate::util::walk::ConstWalker;

/// Port of `CaptureInfo`: capture information for one function argument.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CaptureInfo {
    /// indices of arguments that capture this argument
    pub arg_captures: Vec<u32>,
    /// whether the return value captures this argument
    pub return_captures: bool,
    /// whether this argument is captured externally (globals, unknown code)
    pub extern_captures: bool,
    /// whether this argument is modified
    pub modified: bool,
}

impl CaptureInfo {
    /// `operator bool()`.
    pub fn any(&self) -> bool {
        !self.arg_captures.is_empty() || self.return_captures || self.extern_captures
    }
    /// `CaptureInfo::nothing()`.
    pub fn nothing() -> Self {
        Self::default()
    }
    /// `CaptureInfo::unknown(func, type)`.
    pub fn unknown(m: &Module, func: FuncId, ty: Option<TypeId>) -> Self {
        if !should_track_type(m, ty) {
            return Self::nothing();
        }
        let mut c = Self::default();
        for (i, a) in m.func_args(func).into_iter().enumerate() {
            if should_track_var(m, a) {
                c.arg_captures.push(i as u32);
            }
        }
        c.return_captures = should_track_type(m, return_type(m, func));
        c.extern_captures = true;
        c.modified = true;
        c
    }
}

fn should_track_type(m: &Module, t: Option<TypeId>) -> bool {
    t.map(|t| !m.type_is_atomic(t)).unwrap_or(false)
}
fn should_track_var(m: &Module, v: VarId) -> bool {
    should_track_type(m, m.var_type(v))
}
fn should_track_value(m: &Module, v: ValueId) -> bool {
    match m.kind(v) {
        // pointers are never atomic (and the pointer type may not have been created yet)
        ValueKind::PointerValue(_) => true,
        _ => should_track_type(m, Some(m.value_type(v))),
    }
}

/// Port of `CaptureResult`.
#[derive(Clone, Debug, Default)]
pub struct CaptureResult {
    pub rd_result: RDResult,
    pub dom_result: DominatorResult,
    /// function uid -> per-argument capture info
    pub results: FxHashMap<Id, Vec<CaptureInfo>>,
}

impl AnalysisResult for CaptureResult {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Port of `CaptureAnalysis`.
#[derive(Clone, Debug)]
pub struct CaptureAnalysis {
    rd_analysis_key: String,
    dom_analysis_key: String,
}

impl CaptureAnalysis {
    pub const KEY: &'static str = "core-analyses-capture";
    pub fn new(rd_analysis_key: impl Into<String>, dom_analysis_key: impl Into<String>) -> Self {
        Self { rd_analysis_key: rd_analysis_key.into(), dom_analysis_key: dom_analysis_key.into() }
    }

    pub fn compute(m: &Module, rd: &RDResult, dom: &DominatorResult) -> CaptureResult {
        let cc = CaptureContext::new(rd, None);
        let mut results: FxHashMap<Id, Vec<CaptureInfo>> = FxHashMap::default();
        let main = m.main_func();
        if m.func(main).map(|f| f.is_bodied()).unwrap_or(false) {
            results.insert(m.var_uid(main), cc.get(m, main));
        }
        for v in m.vars() {
            if m.is_func(v) {
                let ans = cc.get(m, v);
                results.entry(m.var_uid(v)).or_insert(ans);
            }
        }
        CaptureResult { rd_result: rd.clone(), dom_result: dom.clone(), results }
    }
}

impl Analysis for CaptureAnalysis {
    fn key(&self) -> String {
        Self::KEY.to_string()
    }
    fn run(&mut self, m: &Module, ctx: &dyn PassContext) -> Box<dyn AnalysisResult> {
        // PORT-NOTE: dependencies are recomputed when missing from the context.
        let rd = match ctx.get_analysis::<RDResult>(&self.rd_analysis_key) {
            Some(r) => r.clone(),
            None => RDAnalysis::compute(m, &CFAnalysis::compute(m)),
        };
        let dom = match ctx.get_analysis::<DominatorResult>(&self.dom_analysis_key) {
            Some(d) => d.clone(),
            None => DominatorAnalysis::compute(&rd.cfg_result),
        };
        Box::new(Self::compute(m, &rd, &dom))
    }
}

/// `escapes(func, value, cr)`: capture info of an arbitrary value within `func`.
pub fn escapes(m: &Module, func: FuncId, value: ValueId, cr: &CaptureResult) -> CaptureInfo {
    if !should_track_value(m, value) {
        return CaptureInfo::nothing();
    }
    let cc = CaptureContext::new(&cr.rd_result, Some(&cr.results));
    let Some(mut ct) = CaptureTracker::for_value(&cc, m, func, value) else {
        return CaptureInfo::unknown(m, func, Some(m.value_type(value)));
    };
    ct.run_to_completion(m, func);
    debug_assert_eq!(ct.dsets.len(), 1, "unexpected dsets size");
    ct.dsets.swap_remove(0).result
}

// ---------------------------------------------------------------------------

/// `OrderCache`: conditional ordering of values within one immutable CFG.
struct OrderCache {
    cfg: Rc<CFGraph>,
    positions: FxHashMap<BlockId, FxHashMap<Id, usize>>,
    block_order: FxHashMap<BlockId, FxHashMap<BlockId, bool>>,
}

impl OrderCache {
    fn new(cfg: Rc<CFGraph>) -> Self {
        Self { cfg, positions: FxHashMap::default(), block_order: FxHashMap::default() }
    }

    fn happens_before(&mut self, m: &Module, before: CfValue, after: CfValue) -> bool {
        let cfg = self.cfg.clone();
        let (Some(bb), Some(ab)) = (cfg.get_block(m, before), cfg.get_block(m, after)) else {
            return false;
        };
        if bb == ab {
            let indices = self.positions.entry(bb).or_insert_with(|| {
                let mut map = FxHashMap::default();
                for (i, v) in cfg.block(bb).values().iter().enumerate() {
                    map.entry(cfg.uid(m, *v)).or_insert(i);
                }
                map
            });
            let bi = indices.get(&cfg.uid(m, before));
            let ai = indices.get(&cfg.uid(m, after));
            match (bi, ai) {
                (None, _) => return false,
                (Some(b), Some(a)) if b > a => return false,
                _ => {}
            }
        }
        let answers = self.block_order.entry(bb).or_default();
        if let Some(a) = answers.get(&ab) {
            return *a;
        }
        let mut answer = true;
        let mut seen: FxHashSet<BlockId> = FxHashSet::default();
        let mut pending: Vec<BlockId> = cfg.block(ab).successors().collect();
        while let Some(b) = pending.pop() {
            if b == bb {
                answer = false;
                break;
            }
            if !seen.insert(b) {
                continue;
            }
            pending.extend(cfg.block(b).successors());
        }
        answers.insert(ab, answer);
        answer
    }
}

/// `RDManager`: caches reaching-definition queries as sets of assignment ids.
struct RDManager {
    rd: Rc<RDInspector>,
    cache: RefCell<FxHashMap<(Id, Id), Rc<FxHashSet<Id>>>>,
}

impl RDManager {
    fn new(rd: Rc<RDInspector>) -> Self {
        Self { rd, cache: RefCell::default() }
    }
    fn get(&self, m: &Module, var: VarId, loc: CfValue) -> Rc<FxHashSet<Id>> {
        let key = (m.var_uid(var), self.rd.cfg().uid(m, loc));
        if let Some(s) = self.cache.borrow().get(&key) {
            return s.clone();
        }
        let set: FxHashSet<Id> =
            self.rd.get_reaching_definitions(m, var, loc).iter().map(|d| d.assignment_uid()).collect();
        let set = Rc::new(set);
        self.cache.borrow_mut().insert(key, set.clone());
        set
    }
    fn is_invalid(&self, m: &Module, v: VarId) -> bool {
        self.rd.is_invalid(m, v)
    }
}

/// `DerivedSet`.
struct DerivedSet {
    func: FuncId,
    root: Option<VarId>,
    args: Vec<Id>,
    derived_vals: FxHashSet<Id>,
    /// var uid -> causes (values whose evaluation made the var derived)
    derived_vars: FxHashMap<Id, Vec<(Id, CfValue)>>,
    result: CaptureInfo,
}

impl DerivedSet {
    fn new(func: FuncId, root: Option<VarId>) -> Self {
        Self {
            func,
            root,
            args: Vec::new(),
            derived_vals: FxHashSet::default(),
            derived_vars: FxHashMap::default(),
            result: CaptureInfo::default(),
        }
    }

    fn set_return_captured(&mut self, m: &Module) {
        if should_track_type(m, return_type(m, self.func)) {
            self.result.return_captures = true;
        }
    }
    fn set_extern_captured(&mut self, m: &Module) {
        self.set_return_captured(m);
        self.result.extern_captures = true;
    }

    fn is_derived_var(&self, m: &Module, v: VarId, loc: CfValue, rd: &RDManager) -> bool {
        let Some(causes) = self.derived_vars.get(&m.var_uid(v)) else {
            return false;
        };
        if m.is_global(v) || rd.is_invalid(m, v) {
            return true;
        }
        let my_set = rd.get(m, v, loc);
        for (_, cause) in causes {
            let other = rd.get(m, v, *cause);
            if my_set.iter().any(|e| other.contains(e)) {
                return true;
            }
        }
        false
    }

    fn is_derived_val(&self, m: &Module, v: ValueId) -> bool {
        self.derived_vals.contains(&m.value_uid(v))
    }

    fn set_derived_var(&mut self, m: &Module, cfg: &CFGraph, v: VarId, cause: CfValue, should_arg_capture: bool) {
        if !should_track_var(m, v) {
            return;
        }
        if m.is_global(v) {
            self.set_extern_captured(m);
        }
        let id = m.var_uid(v);
        if should_arg_capture && let Some(root) = self.root
            && id != m.var_uid(root)
        {
            for (i, a) in self.args.iter().enumerate() {
                if *a == id && !self.result.arg_captures.contains(&(i as u32)) {
                    self.result.arg_captures.push(i as u32);
                }
            }
        }
        let cuid = cfg.uid(m, cause);
        let causes = self.derived_vars.entry(id).or_default();
        if !causes.iter().any(|(u, _)| *u == cuid) {
            causes.push((cuid, cause));
        }
    }

    fn set_derived_val(&mut self, m: &Module, v: ValueId) {
        if !should_track_value(m, v) {
            return;
        }
        self.derived_vals.insert(m.value_uid(v));
    }

    fn size(&self) -> usize {
        self.derived_vals.len() + self.derived_vars.values().map(|c| c.len()).sum::<usize>()
    }
}

fn no_capture_by_annotation(m: &Module, f: FuncId) -> bool {
    has_attribute(m, f, &PURE_ATTR) || has_attribute(m, f, &NO_SIDE_EFFECT_ATTR) || has_attribute(m, f, &NO_CAPTURE_ATTR)
}

fn make_all_capture_info(m: &Module, f: FuncId) -> Vec<CaptureInfo> {
    m.func_args(f).into_iter().map(|a| CaptureInfo::unknown(m, f, m.var_type(a))).collect()
}

fn make_no_capture_info(m: &Module, f: FuncId, derives: bool) -> Vec<CaptureInfo> {
    m.func_args(f)
        .into_iter()
        .map(|a| {
            let mut info = CaptureInfo::nothing();
            if derives && should_track_var(m, a) {
                info.return_captures = true;
            }
            info
        })
        .collect()
}

/// `CaptureContext`.
struct CaptureContext<'a> {
    reaching: &'a RDResult,
    results: RefCell<FxHashMap<Id, Vec<CaptureInfo>>>,
    known_results: Option<&'a FxHashMap<Id, Vec<CaptureInfo>>>,
}

impl<'a> CaptureContext<'a> {
    fn new(reaching: &'a RDResult, known_results: Option<&'a FxHashMap<Id, Vec<CaptureInfo>>>) -> Self {
        Self { reaching, results: RefCell::default(), known_results }
    }

    fn set(&self, m: &Module, f: FuncId, r: Vec<CaptureInfo>) {
        self.results.borrow_mut().insert(m.var_uid(f), r);
    }

    fn cfg(&self, m: &Module, f: FuncId) -> Option<Rc<CFGraph>> {
        self.reaching.cfg_result.graphs.get(&m.var_uid(f)).cloned()
    }
    fn rd(&self, m: &Module, f: FuncId) -> Option<Rc<RDInspector>> {
        self.reaching.results.get(&m.var_uid(f)).cloned()
    }

    fn get(&self, m: &Module, func: FuncId) -> Vec<CaptureInfo> {
        let func = m.actual_var(func);
        let uid = m.var_uid(func);
        if let Some(k) = self.known_results.and_then(|k| k.get(&uid)) {
            return k.clone();
        }
        let Some(fd) = m.func(func) else {
            return Vec::new();
        };
        if fd.is_external() || fd.is_llvm() {
            let derives = has_attribute(m, func, &DERIVES_ATTR);
            if has_attribute(m, func, &SELF_CAPTURES_ATTR) {
                let mut ans = make_no_capture_info(m, func, derives);
                if let Some(first) = ans.first_mut() {
                    first.modified = true;
                }
                let arg_vars = m.func_args(func);
                for i in 1..ans.len() {
                    if should_track_var(m, arg_vars[i]) {
                        ans[i].arg_captures.push(0);
                    }
                }
                return ans;
            }
            return if no_capture_by_annotation(m, func) {
                make_no_capture_info(m, func, derives)
            } else {
                make_all_capture_info(m, func)
            };
        }
        if fd.is_internal() {
            return make_no_capture_info(m, func, m.intrinsic_derives_from_arguments(func));
        }
        // bodied
        if let Some(r) = self.results.borrow().get(&uid) {
            return r.clone();
        }
        self.set(m, func, make_all_capture_info(m, func));
        let Some(mut ct) = CaptureTracker::for_func(self, m, func) else {
            // PORT-NOTE: C++ asserts that every bodied function has a CFG; functions that
            // are not listed in the module get conservative (unknown) results instead.
            return make_all_capture_info(m, func);
        };
        ct.run_to_completion(m, func);
        let mut answer = Vec::new();
        let mut i = 0;
        for a in m.func_args(func) {
            if should_track_var(m, a) {
                answer.push(ct.dsets[i].result.clone());
                i += 1;
            } else {
                answer.push(CaptureInfo::nothing());
            }
        }
        self.set(m, func, answer.clone());
        answer
    }
}

/// `ExtractVars`: which vars are relevant to track in a capturing expression.
fn extract_vars(cc: &CaptureContext<'_>, m: &Module, v: ValueId) -> (Vec<VarId>, bool) {
    struct Ev<'c, 'a> {
        cc: &'c CaptureContext<'a>,
        vars: IndexSet<VarId>,
        escapes: bool,
    }
    impl Ev<'_, '_> {
        fn add(&mut self, m: &Module, v: VarId) {
            let v = m.actual_var(v);
            if should_track_var(m, v) {
                self.vars.insert(v);
            }
        }
        fn process(&mut self, m: &Module, v: ValueId) {
            let v = m.actual_value(v);
            match m.kind(v) {
                ValueKind::VarValue(x) => self.add(m, x.var),
                ValueKind::PointerValue(x) => self.add(m, x.var),
                ValueKind::CallInstr(c) => {
                    let args = c.args.clone();
                    if let Some(func) = get_func(m, c.callee) {
                        let cap = self.cc.get(m, func);
                        for (i, arg) in args.into_iter().enumerate() {
                            // PORT-NOTE: C++ reads `capInfo[i]` here even past the end (vararg
                            // C functions); the bounds-checked `info` is used instead.
                            let info = cap.get(i).cloned().unwrap_or_else(|| {
                                CaptureInfo::unknown(m, func, Some(m.value_type(arg)))
                            });
                            if should_track_value(m, arg) && info.return_captures {
                                self.process(m, arg);
                            }
                        }
                    } else {
                        for arg in args {
                            if should_track_value(m, arg) {
                                self.process(m, arg);
                            }
                        }
                    }
                }
                ValueKind::YieldInInstr(_) => self.escapes = true,
                ValueKind::TernaryInstr(t) => {
                    let (a, b) = (t.true_value, t.false_value);
                    self.process(m, a);
                    self.process(m, b);
                }
                ValueKind::ExtractInstr(e) => {
                    let x = e.val;
                    self.process(m, x);
                }
                ValueKind::FlowInstr(f) => {
                    let x = f.val;
                    self.process(m, x);
                }
                _ => {}
            }
        }
    }
    let mut ev = Ev { cc, vars: IndexSet::new(), escapes: false };
    ev.process(m, v);
    (ev.vars.into_iter().collect(), ev.escapes)
}

/// `CaptureTracker`.
struct CaptureTracker<'c, 'a> {
    cc: &'c CaptureContext<'a>,
    cfg: Rc<CFGraph>,
    rd: RDManager,
    order: OrderCache,
    dsets: Vec<DerivedSet>,
}

impl<'c, 'a> CaptureTracker<'c, 'a> {
    fn base(cc: &'c CaptureContext<'a>, m: &Module, func: FuncId) -> Option<Self> {
        let cfg = cc.cfg(m, func)?;
        let rd = cc.rd(m, func)?;
        Some(Self { cc, order: OrderCache::new(cfg.clone()), cfg, rd: RDManager::new(rd), dsets: Vec::new() })
    }

    fn for_func(cc: &'c CaptureContext<'a>, m: &Module, func: FuncId) -> Option<Self> {
        let mut ct = Self::base(cc, m, func)?;
        let cfg = ct.cfg.clone();
        let mut synth_assigns: FxHashMap<Id, CfValue> = FxHashMap::default();
        for &v in cfg.block(cfg.entry_block()).values() {
            if let Some(s) = cfg.as_synthetic_assign(v)
                && let CfVar::Real(lhs) = s.lhs
                && should_track_var(m, lhs)
            {
                synth_assigns.insert(m.var_uid(lhs), v);
            }
        }
        let args = m.func_args(func);
        let arg_ids: Vec<Id> = args.iter().map(|a| m.var_uid(*a)).collect();
        for a in args {
            if !should_track_var(m, a) {
                continue;
            }
            let cause = *synth_assigns.get(&m.var_uid(a)).expect("could not find synthetic assignment for arg var");
            let mut ds = DerivedSet::new(func, Some(a));
            ds.args = arg_ids.clone();
            ds.set_derived_var(m, &cfg, a, cause, true);
            ct.dsets.push(ds);
        }
        Some(ct)
    }

    fn for_value(cc: &'c CaptureContext<'a>, m: &Module, func: FuncId, value: ValueId) -> Option<Self> {
        let mut ct = Self::base(cc, m, func)?;
        let mut ds = DerivedSet::new(func, None);
        let (vars, esc) = extract_vars(cc, m, value);
        if esc {
            ds.set_extern_captured(m);
        }
        ds.set_derived_val(m, value);
        for v in vars {
            ds.set_derived_var(m, &ct.cfg, v, value.into(), true);
        }
        ct.dsets.push(ds);
        Some(ct)
    }

    fn size(&self) -> usize {
        self.dsets.iter().map(|d| d.size()).sum()
    }

    fn dsets_of_val(&self, m: &Module, v: Option<ValueId>) -> Vec<usize> {
        let Some(v) = v else { return Vec::new() };
        (0..self.dsets.len()).filter(|i| self.dsets[*i].is_derived_val(m, v)).collect()
    }
    fn dsets_of_var(&self, m: &Module, v: VarId, loc: CfValue) -> Vec<usize> {
        (0..self.dsets.len()).filter(|i| self.dsets[*i].is_derived_var(m, v, loc, &self.rd)).collect()
    }

    fn forward_link(&mut self, m: &Module, from: ValueId, cause: CfValue, to_vars: &[VarId], to_escapes: bool, arg_cap: bool) {
        for i in self.dsets_of_val(m, Some(from)) {
            let ds = &mut self.dsets[i];
            if to_escapes {
                ds.set_extern_captured(m);
            }
            for tv in to_vars {
                ds.set_derived_var(m, &self.cfg, *tv, cause, arg_cap);
            }
        }
    }

    fn backward_link_func(
        &mut self,
        m: &Module,
        i: usize,
        cause: CfValue,
        to_vars: &[VarId],
        from_vars: &[VarId],
        from_escapes: bool,
    ) {
        if from_escapes {
            self.dsets[i].set_extern_captured(m);
        }
        for &to_var in to_vars {
            let Some(to_causes) = self.dsets[i].derived_vars.get(&m.var_uid(to_var)).cloned() else {
                continue;
            };
            for (_, to_cause) in to_causes {
                if self.cfg.is_assign(m, to_cause)
                    || self.cfg.as_synthetic_assign(to_cause).is_some()
                    || self.order.happens_before(m, to_cause, cause)
                {
                    continue;
                }
                let derived = if m.is_global(to_var) || self.rd.is_invalid(m, to_var) {
                    true
                } else {
                    let my_set = self.rd.get(m, to_var, cause);
                    let other = self.rd.get(m, to_var, to_cause);
                    my_set.iter().any(|e| other.contains(e))
                };
                if derived {
                    for fv in from_vars {
                        self.dsets[i].set_derived_var(m, &self.cfg, *fv, to_cause, true);
                    }
                }
            }
        }
    }

    /// `link(Value *from, Value *to, Value *cause)`.
    fn link_vals(&mut self, m: &Module, from: ValueId, to: ValueId, cause: ValueId) {
        let (from_vars, from_esc) = extract_vars(self.cc, m, from);
        let (to_vars, to_esc) = extract_vars(self.cc, m, to);
        self.forward_link(m, from, cause.into(), &to_vars, to_esc, true);
        for i in self.dsets_of_val(m, Some(to)) {
            self.backward_link_func(m, i, cause.into(), &to_vars, &from_vars, from_esc);
        }
    }

    /// `link(Value *from, Var *to, Value *cause)`.
    fn link_var(&mut self, m: &Module, from: ValueId, to: VarId, cause: ValueId) {
        let (from_vars, from_esc) = extract_vars(self.cc, m, from);
        let to_vars = [to];
        self.forward_link(m, from, cause.into(), &to_vars, false, false);
        for i in self.dsets_of_var(m, to, cause.into()) {
            self.backward_link_func(m, i, cause.into(), &to_vars, &from_vars, from_esc);
        }
    }

    fn mark_val(&mut self, m: &Module, of: Option<ValueId>, f: impl Fn(&mut DerivedSet, &Module)) {
        for i in self.dsets_of_val(m, of) {
            f(&mut self.dsets[i], m);
        }
    }

    fn handle(&mut self, m: &Module, v: ValueId) {
        match m.kind(v).clone() {
            ValueKind::VarValue(typhon_ir::VarValue { var }) | ValueKind::PointerValue(typhon_ir::PointerValue { var, .. }) => {
                let var = m.actual_var(var);
                for i in self.dsets_of_var(m, var, v.into()) {
                    self.dsets[i].set_derived_val(m, v);
                }
            }
            ValueKind::AssignInstr(a) => self.link_var(m, a.rhs, m.actual_var(a.lhs), v),
            ValueKind::ExtractInstr(e) => {
                if should_track_value(m, v) {
                    self.mark_val(m, Some(e.val), |d, m| d.set_derived_val(m, v));
                }
            }
            ValueKind::InsertInstr(ins) => {
                self.link_vals(m, ins.rhs, ins.lhs, v);
                self.mark_val(m, Some(ins.lhs), |d, _| d.result.modified = true);
            }
            ValueKind::CallInstr(c) => {
                let args = c.args.clone();
                let func = get_func(m, c.callee);
                let cap_info: Vec<CaptureInfo> = match func {
                    Some(f) => self.cc.get(m, f),
                    None => {
                        let arg_captures: Vec<u32> = args
                            .iter()
                            .enumerate()
                            .filter(|(_, a)| should_track_value(m, **a))
                            .map(|(i, _)| i as u32)
                            .collect();
                        let return_captures = should_track_value(m, v);
                        args.iter()
                            .map(|a| {
                                if should_track_value(m, *a) {
                                    CaptureInfo {
                                        arg_captures: arg_captures.clone(),
                                        return_captures,
                                        extern_captures: true,
                                        modified: true,
                                    }
                                } else {
                                    CaptureInfo::nothing()
                                }
                            })
                            .collect()
                    }
                };
                for (i, &arg) in args.iter().enumerate() {
                    let info = cap_info.get(i).cloned().unwrap_or_else(|| match func {
                        Some(f) => CaptureInfo::unknown(m, f, Some(m.value_type(arg))),
                        None => CaptureInfo::nothing(),
                    });
                    for &argno in &info.arg_captures {
                        if let Some(&other) = args.get(argno as usize) {
                            self.link_vals(m, arg, other, v);
                        }
                    }
                    self.mark_val(m, Some(arg), |d, m| {
                        if info.return_captures {
                            d.set_derived_val(m, v);
                        }
                        if info.extern_captures {
                            d.set_extern_captured(m);
                        }
                        if info.modified {
                            d.result.modified = true;
                        }
                    });
                }
            }
            ValueKind::ForFlow(f) => {
                let var = m.actual_var(f.var);
                if !should_track_var(m, var) {
                    return;
                }
                let var_uid = m.var_uid(var);
                let cfg = self.cfg.clone();
                let synth: Vec<CfValue> = cfg
                    .synthetics()
                    .filter_map(|(cv, _)| {
                        let s = cfg.as_synthetic_assign(cv)?;
                        (s.kind == SyntheticAssignKind::NextValue && cfg.var_uid(m, s.lhs) == var_uid).then_some(cv)
                    })
                    .collect();
                for i in self.dsets_of_val(m, Some(f.iter)) {
                    for s in &synth {
                        self.dsets[i].set_derived_var(m, &cfg, var, *s, true);
                    }
                }
            }
            ValueKind::TernaryInstr(t) => {
                self.mark_val(m, Some(t.true_value), |d, m| d.set_derived_val(m, v));
                self.mark_val(m, Some(t.false_value), |d, m| d.set_derived_val(m, v));
            }
            ValueKind::FlowInstr(f) => self.mark_val(m, Some(f.val), |d, m| d.set_derived_val(m, v)),
            ValueKind::ReturnInstr(r) => self.mark_val(m, r.value, |d, _| d.result.return_captures = true),
            ValueKind::YieldInstr(r) => self.mark_val(m, r.value, |d, _| d.result.return_captures = true),
            ValueKind::AwaitInstr(r) => self.mark_val(m, Some(r.value), |d, _| d.result.return_captures = true),
            ValueKind::ThrowInstr(r) => self.mark_val(m, r.value, |d, m| d.set_extern_captured(m)),
            _ => {}
        }
    }

    /// `runToCompletion(func)`.
    fn run_to_completion(&mut self, m: &Module, func: FuncId) {
        loop {
            let old = self.size();
            let mut order = Vec::new();
            ConstWalker::new().walk_func(m, func, &mut |v| order.push(v));
            for v in order {
                self.handle(m, v);
            }
            if self.size() == old {
                break;
            }
        }
    }
}
