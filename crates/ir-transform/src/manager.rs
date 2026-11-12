//! `PassManager` owns the passes, the analyses and their cached results. Passes reach
//! the manager through the [`PassContext`] trait object handed to `Pass::run`
//! (`Pass::manager` back-pointer).

use std::collections::{HashMap, HashSet};

use typhon_common::{Options, Timer};
use typhon_ir::module::Module;
use typhon_ir::pass::{Analysis, AnalysisResult, Pass, PassContext};


/// Manager for keys of passes (`PassManager::KeyManager`).
#[derive(Default, Debug, Clone)]
struct KeyManager {
    /// mapping of raw key to number of occurences
    keys: HashMap<String, i32>,
}

impl KeyManager {
    /// Returns a unique'd key for a given raw key, appending ":<number>" if the key
    /// has been seen.
    fn unique_key(&mut self, key: &str) -> String {
        // make sure we can't ever produce duplicate "unique'd" keys
        assert!(!key.contains(':'), "pass key '{key}' contains invalid character ':'");
        match self.keys.get_mut(key) {
            None => {
                self.keys.insert(key.to_string(), 1);
                key.to_string()
            }
            Some(n) => {
                *n += 1;
                format!("{key}:{n}")
            }
        }
    }
}

/// Container for pass metadata.
struct PassMetadata {
    pass: Box<dyn Pass>,
    reqs: Vec<String>,
    invalidates: Vec<String>,
}

/// Container for analysis metadata.
struct AnalysisMetadata {
    analysis: Box<dyn Analysis>,
    reqs: Vec<String>,
}

/// The [`PassContext`] seen by passes and analyses run by a [`PassManager`].
struct ManagerContext<'a> {
    results: &'a HashMap<String, Box<dyn AnalysisResult>>,
    options: &'a Options,
}

impl PassContext for ManagerContext<'_> {
    fn analysis_result(&self, key: &str) -> Option<&dyn AnalysisResult> {
        self.results.get(key).map(|r| r.as_ref())
    }
    fn options(&self) -> Option<&Options> {
        Some(self.options)
    }
}

/// Utility class to run a series of passes (port of `transform::PassManager`).
pub struct PassManager {
    /// compiler options
    // Note: the reference stores an `Options*` shared with the compiler; here the manager
    // owns a copy, mutable through `options_mut()` (`isDisabled` reads it live).
    options: Options,
    /// key manager to handle duplicate keys (i.e. passes being added twice)
    km: KeyManager,
    /// map of keys to passes
    passes: HashMap<String, PassMetadata>,
    /// map of keys to analyses
    analyses: HashMap<String, AnalysisMetadata>,
    /// reverse dependency map
    deps: HashMap<String, Vec<String>>,
    /// execution order of passes
    execution_order: Vec<String>,
    /// map of valid analysis results
    results: HashMap<String, Box<dyn AnalysisResult>>,
}

impl PassManager {
    /// `PassManager(options)`: registers the standard pipeline for `options`.
    pub fn new(options: Options) -> Self {
        let mut pm = Self {
            options,
            km: KeyManager::default(),
            passes: HashMap::new(),
            analyses: HashMap::new(),
            deps: HashMap::new(),
            execution_order: Vec::new(),
            results: HashMap::new(),
        };
        pm.register_standard_passes();
        pm
    }

    /// `getOptions()`.
    pub fn options(&self) -> &Options {
        &self.options
    }
    pub fn options_mut(&mut self) -> &mut Options {
        &mut self.options
    }

    /// `hasPass(key)`.
    pub fn has_pass(&self, key: &str) -> bool {
        self.passes.contains_key(key)
    }

    /// `hasAnalysis(key)`.
    pub fn has_analysis(&self, key: &str) -> bool {
        self.analyses.contains_key(key)
    }

    /// Execution order of the (unique'd) pass keys.
    pub fn execution_order(&self) -> &[String] {
        &self.execution_order
    }

    /// `registerPass(pass, insertBefore, reqs, invalidates)`. Returns the unique'd key,
    /// or an empty string if the pass is disabled.
    pub fn register_pass(
        &mut self,
        pass: Box<dyn Pass>,
        insert_before: &str,
        reqs: Vec<String>,
        invalidates: Vec<String>,
    ) -> String {
        let key = pass.key();
        if self.is_disabled(&key) {
            return String::new();
        }
        let key = self.km.unique_key(&key);

        for req in &reqs {
            let d = self.deps.get_mut(req).unwrap_or_else(|| panic!("required key '{req}' not found"));
            d.push(key.clone());
        }

        // the reference `setManager` is replaced by passing a `PassContext` at run time.
        self.passes.insert(key.clone(), PassMetadata { pass, reqs, invalidates });
        if insert_before.is_empty() {
            self.execution_order.push(key.clone());
        } else {
            let pos = self
                .execution_order
                .iter()
                .position(|k| k == insert_before)
                .unwrap_or_else(|| panic!("pass with key '{insert_before}' not found in manager"));
            self.execution_order.insert(pos, key.clone());
        }
        key
    }

    /// `registerPass(pass)` with default arguments.
    pub fn register_pass_simple(&mut self, pass: Box<dyn Pass>) -> String {
        self.register_pass(pass, "", Vec::new(), Vec::new())
    }

    /// `registerAnalysis(analysis, reqs)`. Returns the unique'd key, or an empty
    /// string if the analysis is disabled.
    pub fn register_analysis(&mut self, analysis: Box<dyn Analysis>, reqs: Vec<String>) -> String {
        let key = analysis.key();
        if self.is_disabled(&key) {
            return String::new();
        }
        let key = self.km.unique_key(&key);

        for req in &reqs {
            let d = self.deps.get_mut(req).unwrap_or_else(|| panic!("required key '{req}' not found"));
            d.push(key.clone());
        }

        self.analyses.insert(key.clone(), AnalysisMetadata { analysis, reqs });
        self.deps.insert(key.clone(), Vec::new());
        key
    }

    /// `run(module)`: runs all passes in execution order.
    pub fn run(&mut self, m: &mut Module) {
        let order = self.execution_order.clone();
        for p in &order {
            self.run_pass(m, p);
        }
    }

    /// `getAnalysisResult(key)`.
    pub fn analysis_result(&self, key: &str) -> Option<&dyn AnalysisResult> {
        self.results.get(key).map(|r| r.as_ref())
    }

    /// `getAnalysisResult<T>(key)`.
    pub fn get_analysis<T: AnalysisResult>(&self, key: &str) -> Option<&T> {
        self.analysis_result(key).and_then(|r| r.as_any().downcast_ref::<T>())
    }

    /// `isDisabled(key)`.
    pub fn is_disabled(&self, key: &str) -> bool {
        self.options.disabled.iter().any(|d| d == key)
    }

    fn run_pass(&mut self, m: &mut Module, name: &str) {
        let mut it = 0;
        loop {
            let (should_run, reqs) = {
                let meta = self.passes.get(name).unwrap_or_else(|| panic!("pass '{name}' not found"));
                (meta.pass.should_run(m), meta.reqs.clone())
            };
            if !should_run {
                break;
            }
            for dep in &reqs {
                self.run_analysis(m, dep);
            }

            let meta = self.passes.get_mut(name).unwrap();
            let mut timer = Timer::new(format!("  ir pass    : {}", meta.pass.key()));
            let ctx = ManagerContext { results: &self.results, options: &self.options };
            meta.pass.run(m, &ctx);
            timer.log();

            let invalidates = meta.invalidates.clone();
            it += 1;
            let again = meta.pass.should_repeat(it);
            for inv in &invalidates {
                self.invalidate(inv);
            }
            if !again {
                break;
            }
        }
    }

    fn run_analysis(&mut self, m: &mut Module, name: &str) {
        if self.results.contains_key(name) {
            return;
        }
        let reqs = self.analyses.get(name).unwrap_or_else(|| panic!("analysis '{name}' not found")).reqs.clone();
        for dep in &reqs {
            self.run_analysis(m, dep);
        }

        let meta = self.analyses.get_mut(name).unwrap();
        let mut timer = Timer::new(format!("  ir analysis: {}", meta.analysis.key()));
        let ctx = ManagerContext { results: &self.results, options: &self.options };
        let result = meta.analysis.run(m, &ctx);
        timer.log();
        self.results.insert(name.to_string(), result);
    }

    fn invalidate(&mut self, key: &str) {
        let mut open: HashSet<String> = HashSet::from([key.to_string()]);
        while !open.is_empty() {
            let mut new_open = HashSet::new();
            for k in &open {
                if self.results.remove(k).is_some() {
                    if let Some(d) = self.deps.get(k) {
                        new_open.extend(d.iter().cloned());
                    }
                }
            }
            open = new_open;
        }
    }

    fn register_standard_passes(&mut self) {
        if self.options.pmempty {
            // do nothing
        } else if self.options.debug && !self.options.test {
            lanes::register_debug(self);
        } else {
            lanes::register_release(self);
        }
    }
}

/// Standard pipeline registration (`registerStandardPasses`), in original order.
mod lanes {
    use typhon_ir::pass::{Analysis, Pass};
    use typhon_ir_analyze::dataflow::capture::CaptureAnalysis;
    use typhon_ir_analyze::dataflow::cfg::CFAnalysis;
    use typhon_ir_analyze::dataflow::dominator::DominatorAnalysis;
    use typhon_ir_analyze::dataflow::reaching::RDAnalysis;
    use typhon_ir_analyze::module::global_vars::GlobalVarsAnalyses;
    use typhon_ir_analyze::module::side_effect::SideEffectAnalysis;

    use super::PassManager;
    use crate::cleanup::DevirtualizationPass;
    use crate::folding::FoldingPassGroup;
    use crate::lowering::*;
    use crate::pythonic::*;

    fn p(pm: &mut PassManager, pass: Box<dyn Pass>) -> String {
        pm.register_pass_simple(pass)
    }
    fn pr(pm: &mut PassManager, pass: Box<dyn Pass>, reqs: &[&String], inv: &[&String]) -> String {
        pm.register_pass(pass, "", reqs.iter().map(|s| s.to_string()).collect(), inv.iter().map(|s| s.to_string()).collect())
    }
    fn a(pm: &mut PassManager, an: Box<dyn Analysis>, reqs: &[&String]) -> String {
        pm.register_analysis(an, reqs.iter().map(|s| s.to_string()).collect())
    }

    pub(super) fn register_debug(pm: &mut PassManager) {
        p(pm, Box::new(PipelineLowering::new()));
        p(pm, Box::new(ImperativeForFlowLowering::new()));
        p(pm, Box::new(MissingReturnLowering::new()));
        p(pm, Box::new(ThrowLowering::new()));
        p(pm, Box::new(AsyncForLowering::new()));
        p(pm, Box::new(AwaitLowering::new()));
        if let Some(omp) = gated::openmp() {
            p(pm, omp);
        }
        p(pm, Box::new(FormattingOptimization::new()));
    }

    pub(super) fn register_release(pm: &mut PassManager) {
        // Pythonic
        p(pm, Box::new(DictArithmeticOptimization::new()));
        p(pm, Box::new(ListAdditionOptimization::new()));
        p(pm, Box::new(StrAdditionOptimization::new()));
        p(pm, Box::new(GeneratorArgumentOptimization::new()));
        p(pm, Box::new(EnumerateOptimization::new()));
        p(pm, Box::new(IOCatOptimization::new()));
        p(pm, Box::new(FormattingOptimization::new()));

        // lowering
        p(pm, Box::new(PipelineLowering::new()));
        p(pm, Box::new(ImperativeForFlowLowering::new()));
        p(pm, Box::new(MissingReturnLowering::new()));
        p(pm, Box::new(ThrowLowering::new()));

        // folding
        let cfg = a(pm, Box::new(CFAnalysis::default()), &[]);
        let rd = a(pm, Box::new(RDAnalysis::new(cfg.clone())), &[&cfg]);
        let dom = a(pm, Box::new(DominatorAnalysis::new(cfg.clone())), &[&cfg]);
        let cap = a(pm, Box::new(CaptureAnalysis::new(rd.clone(), dom.clone())), &[&rd, &dom]);
        let global = a(pm, Box::new(GlobalVarsAnalyses::default()), &[]);
        let se1 = a(pm, Box::new(SideEffectAnalysis::new(cap.clone(), true)), &[&cap]);
        let se2 = a(pm, Box::new(SideEffectAnalysis::new(cap.clone(), false)), &[&cap]);
        let numpy = gated::numpy_expression_analysis(&rd, &se2).map(|an| a(pm, an, &[&rd, &se2]));
        if let Some(p) = numpy.as_ref().and_then(|np| gated::numpy_inline(np).map(|p| (np, p))) {
            pr(pm, p.1, &[p.0], &[&se1, &se2, &rd, &cfg, &global, &cap]);
        }
        pr(
            pm,
            Box::new(FoldingPassGroup::new(se1.clone(), rd.clone(), global.clone(), 5, false)),
            &[&se1, &rd, &global],
            &[&se1, &rd, &cfg, &global, &cap],
        );
        if let Some(p) = numpy.as_ref().and_then(|np| gated::numpy_fusion(np, &se2).map(|p| (np, p))) {
            pr(pm, p.1, &[p.0, &se2], &[&se1, &rd, &cfg, &global, &cap]);
        }
        pr(pm, Box::new(GeneratorLoopFusion::new()), &[], &[&se1, &se2, &rd, &cfg, &global, &cap]);
        if let Some(l) = gated::numpy_lifetime(&rd) {
            pr(pm, l, &[&rd], &[&se1, &se2, &rd, &cfg, &global, &cap]);
        }
        p(pm, Box::new(ImperativeForFlowLowering::new()));
        if let Some(b) = gated::numpy_bounds_check_elision(&rd) {
            pr(pm, b, &[&rd], &[&se1, &rd, &cfg, &global, &cap]);
        }

        // async & parallel
        p(pm, Box::new(AsyncForLowering::new()));
        p(pm, Box::new(AwaitLowering::new()));
        if let Some(omp) = gated::openmp() {
            pr(pm, omp, &[], &[&cfg, &global]);
        }

        if !pm.options().jit {
            // Don't demote globals in JIT mode, since they might be used later
            // by another user input.
            pr(
                pm,
                Box::new(FoldingPassGroup::new(se2.clone(), rd.clone(), global.clone(), 5, true)),
                &[&se2, &rd, &global],
                &[&se2, &rd, &cfg, &global],
            );
        }
        pr(pm, Box::new(DevirtualizationPass::new(rd.clone())), &[&rd], &[&cfg, &rd, &global]);
    }

    /// Passes owned by the numpy and parallel modules (kept as small factories so the
    /// pipeline above reads like the reference `registerStandardPasses`).
    mod gated {
        use typhon_ir::pass::{Analysis, Pass};

        use crate::numpy::{
            NumPyBoundsCheckElisionPass, NumPyExpressionAnalysis, NumPyFusionPass, NumPyInlinePass,
            NumPyLifetimePass,
        };

        pub(super) fn openmp() -> Option<Box<dyn Pass>> {
            Some(Box::new(crate::parallel::OpenMPPass::new()))
        }
        pub(super) fn numpy_expression_analysis(rd: &str, se: &str) -> Option<Box<dyn Analysis>> {
            Some(Box::new(NumPyExpressionAnalysis::new(rd, se)))
        }
        pub(super) fn numpy_inline(np: &str) -> Option<Box<dyn Pass>> {
            Some(Box::new(NumPyInlinePass::new(np)))
        }
        pub(super) fn numpy_fusion(np: &str, se: &str) -> Option<Box<dyn Pass>> {
            Some(Box::new(NumPyFusionPass::new(np, se)))
        }
        pub(super) fn numpy_lifetime(rd: &str) -> Option<Box<dyn Pass>> {
            Some(Box::new(NumPyLifetimePass::new(rd)))
        }
        pub(super) fn numpy_bounds_check_elision(rd: &str) -> Option<Box<dyn Pass>> {
            Some(Box::new(NumPyBoundsCheckElisionPass::new(rd)))
        }
    }
}
