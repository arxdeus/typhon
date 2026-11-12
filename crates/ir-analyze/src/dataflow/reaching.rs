//! Reaching definitions.

use std::cell::RefCell;
use std::collections::VecDeque;
use std::hash::{Hash, Hasher};
use std::rc::Rc;

use indexmap::IndexSet;
use rustc_hash::{FxHashMap, FxHashSet};
use typhon_ir::pass::{Analysis, AnalysisResult, PassContext};
use typhon_ir::{Id, Module, ValueId};

use crate::dataflow::cfg::{BlockId, CFAnalysis, CFGraph, CFResult, CfValue, CfVar, SyntheticAssignKind};
use crate::util::bitset::BitSet;

#[derive(Clone, Copy, Debug)]
pub struct ReachingDef {
    /// the assignment instruction (`AssignInstr` or `SyntheticAssignInstr`)
    pub assignment: CfValue,
    /// the assigned value, if known
    pub assignee: Option<CfValue>,
    assignment_uid: Id,
    assignee_uid: Option<Id>,
}

impl ReachingDef {
    pub fn new(cfg: &CFGraph, m: &Module, assignment: CfValue, assignee: Option<CfValue>) -> Self {
        Self {
            assignment,
            assignee,
            assignment_uid: cfg.uid(m, assignment),
            assignee_uid: assignee.map(|a| cfg.uid(m, a)),
        }
    }
    /// `known()`.
    pub fn known(&self) -> bool {
        self.assignee.is_some()
    }
    /// `getId()`.
    pub fn id(&self) -> Id {
        self.assignee_uid.unwrap_or(self.assignment_uid)
    }
    /// `assignment->getId()`.
    pub fn assignment_uid(&self) -> Id {
        self.assignment_uid
    }
    /// `assignee->getId()`.
    pub fn assignee_uid(&self) -> Option<Id> {
        self.assignee_uid
    }
    /// The assignee as a real IR value (`None` if unknown or synthetic).
    pub fn assignee_value(&self) -> Option<ValueId> {
        self.assignee.and_then(|a| a.as_real())
    }
    /// The assignment as a real IR value (`AssignInstr`).
    pub fn assignment_value(&self) -> Option<ValueId> {
        self.assignment.as_real()
    }
}

impl PartialEq for ReachingDef {
    fn eq(&self, other: &Self) -> bool {
        if self.known() != other.known() {
            return false;
        }
        self.id() == other.id()
    }
}
impl Eq for ReachingDef {}
impl Hash for ReachingDef {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id().hash(state)
    }
}

/// `getKilled(val)`: uid of the variable assigned by `val`, or `-1`.
pub fn get_killed(cfg: &CFGraph, m: &Module, val: CfValue) -> Id {
    match val {
        CfValue::Real(v) => m.as_assign_instr(v).map(|a| m.var_uid(a.lhs)).unwrap_or(-1),
        CfValue::Synth(_) => cfg.as_synthetic_assign(val).map(|s| cfg.var_uid(m, s.lhs)).unwrap_or(-1),
    }
}

/// `getGenerated(val)`.
pub fn get_generated(cfg: &CFGraph, m: &Module, val: CfValue) -> Option<(Id, ReachingDef)> {
    match val {
        CfValue::Real(v) => {
            let a = m.as_assign_instr(v)?;
            Some((m.var_uid(a.lhs), ReachingDef::new(cfg, m, val, Some(a.rhs.into()))))
        }
        CfValue::Synth(_) => {
            let s = cfg.as_synthetic_assign(val)?;
            let assignee = if s.kind == SyntheticAssignKind::Known { s.arg } else { None };
            Some((cfg.var_uid(m, s.lhs), ReachingDef::new(cfg, m, val, assignee)))
        }
    }
}


#[derive(Debug, Default)]
struct BlockIndex {
    positions: FxHashMap<Id, usize>,
    assignments: FxHashMap<Id, Vec<(usize, CfValue)>>,
}

#[derive(Debug)]
pub struct RDInspector {
    invalid: FxHashSet<Id>,
    /// Per-block incoming definitions (bit `i` = `ordering[i]`), queried lazily
    /// instead of materializing a per-block var -> def-set map (O(blocks*defs)).
    in_bits: Vec<BitSet>,
    /// Assignment values, in definition-index order.
    ordering: Vec<CfValue>,
    /// var uid -> ascending definition indices into `ordering`.
    var_defs: FxHashMap<Id, Vec<usize>>,
    index: RefCell<FxHashMap<BlockId, Rc<BlockIndex>>>,
    cfg: Rc<CFGraph>,
}

impl RDInspector {
    pub fn new(cfg: Rc<CFGraph>) -> Self {
        Self { invalid: FxHashSet::default(), in_bits: Vec::new(), ordering: Vec::new(), var_defs: FxHashMap::default(), index: RefCell::default(), cfg }
    }
    pub fn cfg(&self) -> &Rc<CFGraph> {
        &self.cfg
    }

    /// `analyze()`.
    pub fn analyze(&mut self, m: &Module) {
        let cfg = self.cfg.clone();
        let mut ordering: Vec<CfValue> = Vec::new();
        let mut lookup: FxHashMap<Id, usize> = FxHashMap::default();
        let mut var_to_assignments: FxHashMap<Id, Vec<CfValue>> = FxHashMap::default();
        for b in cfg.block_ids() {
            for &val in cfg.block(b).values() {
                let k = get_killed(&cfg, m, val);
                if k != -1 {
                    lookup.entry(cfg.uid(m, val)).or_insert(ordering.len());
                    ordering.push(val);
                    var_to_assignments.entry(k).or_default().push(val);
                }
            }
        }
        let n = ordering.len();
        struct Sets {
            gen_: BitSet,
            kill: BitSet,
            in_: BitSet,
            out: BitSet,
        }
        let mut bitsets: Vec<Sets> = Vec::with_capacity(cfg.size());
        for b in cfg.block_ids() {
            let mut gen_ = BitSet::new(n);
            let mut kill = BitSet::new(n);
            let mut generated: IndexSet<Id> = IndexSet::new();
            let mut generated_map: FxHashMap<Id, Id> = FxHashMap::default();
            for &val in cfg.block(b).values() {
                if let CfValue::Real(v) = val
                    && let Some(p) = m.as_pointer_value(v)
                {
                    self.invalid.insert(m.var_uid(p.var));
                    continue;
                }
                if let Some((var, _)) = get_generated(&cfg, m, val) {
                    generated.insert(var);
                    generated_map.insert(var, cfg.uid(m, val));
                }
                let k = get_killed(&cfg, m, val);
                if k != -1 {
                    let vuid = cfg.uid(m, val);
                    for a in var_to_assignments.get(&k).into_iter().flatten() {
                        let auid = cfg.uid(m, *a);
                        if auid != vuid {
                            kill.set(lookup[&auid]);
                        }
                    }
                }
            }
            for var in generated {
                gen_.set(lookup[&generated_map[&var]]);
            }
            let out = gen_.clone();
            bitsets.push(Sets { gen_, kill, in_: BitSet::new(n), out });
        }

        let mut have: FxHashSet<BlockId> = FxHashSet::default();
        let mut queue: VecDeque<BlockId> = VecDeque::new();
        for b in cfg.block_ids() {
            if have.insert(b) {
                queue.push_back(b);
            }
        }
        while let Some(b) = queue.pop_front() {
            have.remove(&b);
            let mut in_ = BitSet::new(n);
            for p in cfg.block(b).predecessors() {
                in_.union(&bitsets[p].out);
            }
            let data = &mut bitsets[b];
            data.in_ = in_;
            let Sets { gen_, kill, in_, out } = data;
            let changed = out.apply_transfer(in_, kill, gen_);
            if changed {
                for s in cfg.block(b).successors() {
                    if have.insert(s) {
                        queue.push_back(s);
                    }
                }
            }
        }

        let mut var_defs: FxHashMap<Id, Vec<usize>> = FxHashMap::default();
        for (i, &val) in ordering.iter().enumerate() {
            if let Some((var, _)) = get_generated(&cfg, m, val) {
                var_defs.entry(var).or_default().push(i);
            }
        }
        self.in_bits = bitsets.into_iter().map(|s| s.in_).collect();
        self.ordering = ordering;
        self.var_defs = var_defs;
    }

    fn block_index(&self, m: &Module, b: BlockId) -> Rc<BlockIndex> {
        if let Some(i) = self.index.borrow().get(&b) {
            return i.clone();
        }
        let cfg = &self.cfg;
        let mut idx = BlockIndex::default();
        for (pos, &val) in cfg.block(b).values().iter().enumerate() {
            idx.positions.entry(cfg.uid(m, val)).or_insert(pos);
            let var = get_killed(cfg, m, val);
            if var != -1 {
                idx.assignments.entry(var).or_default().push((pos, val));
            }
        }
        let idx = Rc::new(idx);
        self.index.borrow_mut().insert(b, idx.clone());
        idx
    }

    /// `getReachingDefinitions(var, loc)`.
    pub fn get_reaching_definitions(&self, m: &Module, var: impl Into<CfVar>, loc: impl Into<CfValue>) -> Vec<ReachingDef> {
        let var = var.into();
        let loc = loc.into();
        let cfg = &self.cfg;
        let var_uid = cfg.var_uid(m, var);
        if self.invalid.contains(&var_uid) {
            return Vec::new();
        }
        if let CfVar::Real(v) = var
            && m.is_global(v)
        {
            return Vec::new();
        }
        let Some(blk) = cfg.get_block(m, loc) else {
            return Vec::new();
        };
        let idx = self.block_index(m, blk);
        let limit = idx.positions.get(&cfg.uid(m, loc)).copied().unwrap_or(usize::MAX);
        if let Some(values) = idx.assignments.get(&var_uid) {
            // upper_bound: first assignment with position > limit
            let found = values.partition_point(|(p, _)| *p <= limit);
            if found > 0 {
                return get_generated(cfg, m, values[found - 1].1).map(|(_, d)| vec![d]).unwrap_or_default();
            }
        }
        if blk == cfg.entry_block() {
            return Vec::new();
        }
        let (Some(bits), Some(defs)) = (self.in_bits.get(blk), self.var_defs.get(&var_uid)) else {
            return Vec::new();
        };
        let mut out: Vec<ReachingDef> = Vec::new();
        for &i in defs {
            if bits.get(i)
                && let Some((_, d)) = get_generated(cfg, m, self.ordering[i])
                && !out.contains(&d)
            {
                out.push(d);
            }
        }
        out
    }

    /// `isInvalid(var)`.
    pub fn is_invalid(&self, m: &Module, var: impl Into<CfVar>) -> bool {
        self.invalid.contains(&self.cfg.var_uid(m, var.into()))
    }
}

#[derive(Clone, Debug, Default)]
pub struct RDResult {
    pub cfg_result: CFResult,
    pub results: FxHashMap<Id, Rc<RDInspector>>,
}

impl AnalysisResult for RDResult {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[derive(Clone, Debug)]
pub struct RDAnalysis {
    cf_analysis_key: String,
}

impl RDAnalysis {
    pub const KEY: &'static str = "core-analyses-rd";
    pub fn new(cf_analysis_key: impl Into<String>) -> Self {
        Self { cf_analysis_key: cf_analysis_key.into() }
    }
    pub fn compute(m: &Module, cfg_result: &CFResult) -> RDResult {
        let mut ret = RDResult { cfg_result: cfg_result.clone(), results: FxHashMap::default() };
        for (id, g) in &cfg_result.graphs {
            let mut insp = RDInspector::new(g.clone());
            insp.analyze(m);
            ret.results.insert(*id, Rc::new(insp));
        }
        ret
    }
}

impl Analysis for RDAnalysis {
    fn key(&self) -> String {
        Self::KEY.to_string()
    }
    fn run(&mut self, m: &Module, ctx: &dyn PassContext) -> Box<dyn AnalysisResult> {
        // Note: recompute the CFG when the dependency is not available.
        let owned;
        let cfg = match ctx.get_analysis::<CFResult>(&self.cf_analysis_key) {
            Some(c) => c,
            None => {
                owned = CFAnalysis::compute(m);
                &owned
            }
        };
        Box::new(Self::compute(m, cfg))
    }
}
