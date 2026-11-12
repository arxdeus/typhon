//! Note: the reference iterative bit-vector algorithm is ported directly (it also gives
//! unreachable blocks a defined dominator set, which `petgraph::algo::dominators` does not).

use std::rc::Rc;

use rustc_hash::FxHashMap;
use typhon_ir::pass::{Analysis, AnalysisResult, PassContext};
use typhon_ir::{Id, Module};

use crate::dataflow::cfg::{BlockId, CFAnalysis, CFGraph, CFResult, CfValue};
use crate::util::bitset::BitSet;

#[derive(Debug)]
pub struct DominatorInspector {
    /// block -> set of blocks dominating it
    sets: Vec<BitSet>,
    cfg: Rc<CFGraph>,
}

impl DominatorInspector {
    pub fn new(cfg: Rc<CFGraph>) -> Self {
        Self { sets: Vec::new(), cfg }
    }
    pub fn cfg(&self) -> &Rc<CFGraph> {
        &self.cfg
    }

    /// `analyze()`.
    pub fn analyze(&mut self) {
        let cfg = &self.cfg;
        let n = cfg.size();
        let mut bitvecs: Vec<BitSet> = (0..n)
            .map(|id| {
                let mut bv = BitSet::new(n);
                if id == cfg.entry_block() {
                    bv.set(id);
                } else {
                    bv.set_all(n);
                }
                bv
            })
            .collect();
        let mut changed = true;
        while changed {
            changed = false;
            for id in 0..n {
                let blk = cfg.block(id);
                let mut new_set: Option<BitSet> = None;
                for p in blk.predecessors() {
                    match &mut new_set {
                        None => new_set = Some(bitvecs[p].clone()),
                        Some(s) => s.intersect(&bitvecs[p]),
                    }
                }
                let mut new_set = new_set.unwrap_or_else(|| BitSet::new(n));
                new_set.set(id);
                if new_set != bitvecs[id] {
                    bitvecs[id] = new_set;
                    changed = true;
                }
            }
        }
        self.sets = bitvecs;
    }

    /// Whether block `b` is dominated by block `d`.
    pub fn block_dominated(&self, b: BlockId, d: BlockId) -> bool {
        self.sets.get(b).map(|s| s.get(d)).unwrap_or(false)
    }

    /// `isDominated(v, dominator)`.
    pub fn is_dominated(&self, m: &Module, v: impl Into<CfValue>, dominator: impl Into<CfValue>) -> bool {
        let (v, dominator) = (v.into(), dominator.into());
        let cfg = &self.cfg;
        let (Some(vb), Some(db)) = (cfg.get_block(m, v), cfg.get_block(m, dominator)) else {
            // Note: the reference dereferences null blocks here.
            return false;
        };
        if vb == db {
            let vals = cfg.block(vb).values();
            let len = vals.len();
            let vuid = cfg.uid(m, v);
            let duid = cfg.uid(m, dominator);
            let pos = |u: Id| vals.iter().position(|x| cfg.uid(m, *x) == u).unwrap_or(len);
            return pos(duid) <= pos(vuid);
        }
        self.block_dominated(vb, db)
    }
}

#[derive(Clone, Debug, Default)]
pub struct DominatorResult {
    pub cfg_result: CFResult,
    pub results: FxHashMap<Id, Rc<DominatorInspector>>,
}

impl AnalysisResult for DominatorResult {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[derive(Clone, Debug)]
pub struct DominatorAnalysis {
    cf_analysis_key: String,
}

impl DominatorAnalysis {
    pub const KEY: &'static str = "core-analyses-dominator";
    pub fn new(cf_analysis_key: impl Into<String>) -> Self {
        Self { cf_analysis_key: cf_analysis_key.into() }
    }
    pub fn compute(cfg_result: &CFResult) -> DominatorResult {
        let mut ret = DominatorResult { cfg_result: cfg_result.clone(), results: FxHashMap::default() };
        for (id, g) in &cfg_result.graphs {
            let mut insp = DominatorInspector::new(g.clone());
            insp.analyze();
            ret.results.insert(*id, Rc::new(insp));
        }
        ret
    }
}

impl Analysis for DominatorAnalysis {
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
        Box::new(Self::compute(cfg))
    }
}
