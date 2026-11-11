//! Port of codon/cir/transform/lowering/pipeline.{h,cpp}

use typhon_ir::util::irtools;
use typhon_ir::util::operator::{Operator, OperatorState};
use typhon_ir::{CallInstr, FuncId, Module, Stage, TypeKind, TypeId, ValueId, ValueKind};

use super::{operator_pass, operator_state};

fn call_stage(m: &mut Module, stage: &Stage, last: ValueId) -> ValueId {
    let args = stage.args.iter().map(|a| a.unwrap_or(last)).collect();
    let src = m.src_info(stage.callee.into());
    m.n(src, ValueKind::CallInstr(CallInstr { callee: stage.callee, args }))
}

/// `Stage::getOutputElementType()`.
pub(crate) fn stage_output_element_type(m: &mut Module, stage: &Stage) -> TypeId {
    let ct = m.value_type_mut(stage.callee);
    let out = if stage.args.is_empty() {
        ct
    } else {
        m.func_type_ret(ct).unwrap_or_else(|| panic!("{} is not a function type", m.type_name(ct)))
    };
    if stage.generator {
        match m.type_kind(out) {
            TypeKind::Generator { base } => *base,
            _ => panic!("generator type not found"),
        }
    } else {
        out
    }
}

fn helper(m: &mut Module, parent: FuncId, stages: &[Stage], idx: usize, last: Option<ValueId>) -> Option<ValueId> {
    if idx >= stages.len() {
        return last;
    }
    let stage = &stages[idx];
    if idx == 0 {
        return helper(m, parent, stages, idx + 1, Some(stage.callee));
    }
    let last = last.expect("pipeline stage without input");
    let prev = &stages[idx - 1];
    if prev.generator {
        let t = stage_output_element_type(m, prev);
        let var = m.nr_var(t, false, "");
        m.func_push_symbol(parent, var);
        let vv = m.nr_var_value(var);
        let c = call_stage(m, stage, vv);
        let body = helper(m, parent, stages, idx + 1, Some(c)).expect("body");
        let body = irtools::series(m, &[body]).expect("series");
        let src = m.src_info(last.into());
        let lp = m.nr_for_flow(last, body, var);
        m.set_src_info(lp.into(), src);
        if stage.parallel {
            m.as_for_flow_mut(lp).unwrap().set_parallel(true);
        }
        Some(lp)
    } else {
        let c = call_stage(m, stage, last);
        helper(m, parent, stages, idx + 1, Some(c))
    }
}

/// Converts pipeline flows into (nested) for loops and calls.
#[derive(Default)]
pub struct PipelineLowering {
    state: OperatorState,
}

impl PipelineLowering {
    pub const KEY: &'static str = "core-pipeline-lowering";
    pub fn new() -> Self {
        Self::default()
    }
}

impl Operator for PipelineLowering {
    operator_state!();
    fn handle_pipeline_flow(&mut self, m: &mut Module, v: ValueId) {
        let stages = m.as_pipeline_flow(v).unwrap().stages.clone();
        let parent = self.parent_func(m).expect("pipeline outside function");
        if let Some(r) = helper(m, parent, &stages, 0, None) {
            m.replace_all_value(v, r);
        }
    }
}

operator_pass!(PipelineLowering);
