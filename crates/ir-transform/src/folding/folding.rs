
use typhon_ir::Module;
use typhon_ir::pass::{Pass, PassContext};

use super::const_fold::FoldingPass;
use super::const_prop::ConstPropPass;
use crate::cleanup::{CanonicalizationPass, DeadCodeCleanupPass, GlobalDemotionPass};

/// Group of constant folding passes.
// Note: the reference implementation derives from `PassGroup` and keeps raw pointers to the members to
// query their replacement counters; here the members are typed fields run in the
// same order (global demotion, const prop, canonicalization, folding, DCE).
pub struct FoldingPassGroup {
    repeat: i32,
    gd: Option<GlobalDemotionPass>,
    cp: ConstPropPass,
    canon: CanonicalizationPass,
    fp: FoldingPass,
    dce: DeadCodeCleanupPass,
}

impl FoldingPassGroup {
    pub const KEY: &'static str = "core-folding-pass-group";

    /// `FoldingPassGroup(sideEffectsPass, reachingDefPass, globalVarPass, repeat, runGlobalDemotion)`.
    pub fn new(
        side_effects_pass: impl Into<String>,
        reaching_def_pass: impl Into<String>,
        global_var_pass: impl Into<String>,
        repeat: i32,
        run_global_demotion: bool,
    ) -> Self {
        let se = side_effects_pass.into();
        Self {
            repeat,
            gd: run_global_demotion.then(GlobalDemotionPass::new),
            cp: ConstPropPass::new(reaching_def_pass, global_var_pass),
            canon: CanonicalizationPass::new(se.clone()),
            fp: FoldingPass::new(),
            dce: DeadCodeCleanupPass::new(se),
        }
    }

    pub fn repeat(&self) -> i32 {
        self.repeat
    }
    pub fn set_repeat(&mut self, r: i32) {
        self.repeat = r;
    }
}

impl Pass for FoldingPassGroup {
    fn key(&self) -> String {
        Self::KEY.to_string()
    }
    fn run(&mut self, m: &mut Module, ctx: &dyn PassContext) {
        if let Some(gd) = &mut self.gd {
            gd.run(m, ctx);
        }
        self.cp.run(m, ctx);
        self.canon.run(m, ctx);
        self.fp.run(m, ctx);
        self.dce.run(m, ctx);
    }
    fn should_repeat(&self, num: i32) -> bool {
        num < self.repeat
            && (self.gd.as_ref().is_some_and(|g| g.num_demotions() != 0)
                || self.canon.num_replacements() != 0
                || self.fp.num_replacements() != 0
                || self.dce.num_replacements() != 0)
    }
}
