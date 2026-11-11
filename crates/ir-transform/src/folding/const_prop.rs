//! Port of codon/cir/transform/folding/const_prop.{h,cpp}

use typhon_ir::magic;
use typhon_ir::pass::{Pass, PassContext, run_operator_pass};
use typhon_ir::types::TypeKind;
use typhon_ir::util::cloning::CloneVisitor;
use typhon_ir::util::irtools::get_func;
use typhon_ir::util::operator::{Operator, OperatorState};
use typhon_ir::{Module, ValueId};
use typhon_ir_analyze::dataflow::reaching::RDResult;
use typhon_ir_analyze::module::global_vars::GlobalVarsResult;

fn ok_const(m: &Module, v: Option<ValueId>) -> bool {
    v.is_some_and(|v| m.as_int_const(v).is_some() || m.as_float_const(v).is_some() || m.as_bool_const(v).is_some())
}

fn is_record(m: &Module, t: typhon_ir::TypeId) -> bool {
    matches!(m.type_kind(t), TypeKind::Record { .. })
}

/// Constant propagation pass.
pub struct ConstPropPass {
    state: OperatorState,
    /// Key of the reaching definition analysis
    reaching_def_key: String,
    /// Key of the global variables analysis
    global_vars_key: String,
}

impl ConstPropPass {
    pub const KEY: &'static str = "core-folding-const-prop";

    pub fn new(reaching_def_key: impl Into<String>, global_vars_key: impl Into<String>) -> Self {
        Self { state: OperatorState::new(false), reaching_def_key: reaching_def_key.into(), global_vars_key: global_vars_key.into() }
    }

    fn handle_var_value_ctx(&mut self, m: &mut Module, v: ValueId, ctx: &dyn PassContext) {
        let Some(var) = m.as_var_value(v).map(|x| x.var) else { return };
        let var = m.actual_var(var);
        if m.is_thread_local(var) {
            return;
        }
        let const_def = if m.is_global(var) {
            let Some(r) = ctx.get_analysis::<GlobalVarsResult>(&self.global_vars_key) else { return };
            let Some(&uid) = r.assignments.get(&m.var_uid(var)) else { return };
            m.get_value(uid)
        } else {
            let Some(r) = ctx.get_analysis::<RDResult>(&self.reaching_def_key) else { return };
            let Some(pf) = self.parent_func(m) else { return };
            let Some(rd) = r.results.get(&m.var_uid(pf)) else { return };
            let reaching = rd.get_reaching_definitions(m, var, v);
            if reaching.len() != 1 || !reaching[0].known() {
                return;
            }
            reaching[0].assignee_value()
        };
        if !ok_const(m, const_def) {
            return;
        }
        let replacement = CloneVisitor::new(true).clone_value(m, const_def.unwrap());
        m.replace_all_value(v, replacement);
    }

    fn handle_extract_ctx(&mut self, m: &mut Module, v: ValueId, ctx: &dyn PassContext) {
        // Propagate constant tuples
        let Some(ex) = m.as_extract_instr(v) else { return };
        let (val, field) = (ex.val, ex.field.clone());
        let Some(var) = m.as_var_value(val).map(|x| m.actual_var(x.var)) else { return };
        let vt = m.value_type_mut(val);
        if !is_record(m, vt) || m.is_global(var) {
            return;
        }
        let Some(r) = ctx.get_analysis::<RDResult>(&self.reaching_def_key) else { return };
        let Some(pf) = self.parent_func(m) else { return };
        let Some(rd) = r.results.get(&m.var_uid(pf)) else { return };
        let reaching = rd.get_reaching_definitions(m, var, v);
        if reaching.len() != 1 || !reaching[0].known() {
            return;
        }
        let Some(call) = reaching[0].assignee_value() else { return };
        let Some(ci) = m.as_call_instr(call) else { return };
        let (callee, args) = (ci.callee, ci.args.clone());
        let Some(func) = get_func(m, callee) else { return };
        let fd = m.func(func).unwrap();
        if fd.unmangled_name != magic::NEW {
            return;
        }
        let Some(tuple) = fd.parent_type else { return };
        if !is_record(m, tuple) || m.type_name(tuple) != "Tuple" {
            return;
        }
        let idx = m.member_index(vt, &field);
        if idx < 0 || idx as usize >= args.len() || !ok_const(m, Some(args[idx as usize])) {
            return;
        }
        let replacement = CloneVisitor::new(true).clone_value(m, args[idx as usize]);
        m.replace_all_value(v, replacement);
    }
}

/// Operator adapter carrying the pass context through the traversal.
struct Runner<'a, 'c> {
    pass: &'a mut ConstPropPass,
    ctx: &'a (dyn PassContext + 'c),
}

impl Operator for Runner<'_, '_> {
    fn state(&self) -> &OperatorState {
        &self.pass.state
    }
    fn state_mut(&mut self) -> &mut OperatorState {
        &mut self.pass.state
    }
    fn handle_var_value(&mut self, m: &mut Module, v: ValueId) {
        let ctx = self.ctx;
        // `parent_func` must be read from the runner's (shared) state.
        self.pass.handle_var_value_ctx(m, v, ctx);
    }
    fn handle_extract_instr(&mut self, m: &mut Module, v: ValueId) {
        let ctx = self.ctx;
        self.pass.handle_extract_ctx(m, v, ctx);
    }
}

impl Operator for ConstPropPass {
    fn state(&self) -> &OperatorState {
        &self.state
    }
    fn state_mut(&mut self) -> &mut OperatorState {
        &mut self.state
    }
}

impl Pass for ConstPropPass {
    fn key(&self) -> String {
        Self::KEY.to_string()
    }
    fn run(&mut self, m: &mut Module, ctx: &dyn PassContext) {
        let mut r = Runner { pass: self, ctx };
        run_operator_pass(&mut r, m);
    }
}
