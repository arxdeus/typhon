
use rustc_hash::FxHashMap;
use typhon_ir::pass::{Analysis, AnalysisResult, PassContext};
use typhon_ir::{Id, Module};

use crate::util::walk::ConstWalker;

#[derive(Clone, Debug, Default)]
pub struct GlobalVarsResult {
    /// global var uid -> uid of its single assigned value, `-1` if assigned several times or address taken
    pub assignments: FxHashMap<Id, Id>,
}

impl AnalysisResult for GlobalVarsResult {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[derive(Clone, Debug, Default)]
pub struct GlobalVarsAnalyses;

impl GlobalVarsAnalyses {
    pub const KEY: &'static str = "core-analyses-global-vars";

    pub fn compute(m: &Module) -> GlobalVarsResult {
        let mut assignments: FxHashMap<Id, Id> = FxHashMap::default();
        ConstWalker::new().walk_module(m, &mut |v| {
            if let Some(p) = m.as_pointer_value(v) {
                let var = m.actual_var(p.var);
                if m.is_global(var) {
                    assignments.insert(m.var_uid(var), -1);
                }
            } else if let Some(a) = m.as_assign_instr(v) {
                let lhs = m.actual_var(a.lhs);
                if m.is_global(lhs) {
                    let id = m.var_uid(lhs);
                    if assignments.contains_key(&id) {
                        assignments.insert(id, -1);
                    } else {
                        assignments.insert(id, m.value_uid(a.rhs));
                    }
                }
            }
        });
        GlobalVarsResult { assignments }
    }
}

impl Analysis for GlobalVarsAnalyses {
    fn key(&self) -> String {
        Self::KEY.to_string()
    }
    fn run(&mut self, m: &Module, _ctx: &dyn PassContext) -> Box<dyn AnalysisResult> {
        Box::new(Self::compute(m))
    }
}
