
use indexmap::IndexMap;
use typhon_ir::pass::{Pass, PassContext};
use typhon_ir::util::operator::{Operator, OperatorState};
use typhon_ir::{FuncId, Id, Module, NodeRef, VarId};
use typhon_common::log_flag;

#[derive(Default)]
struct GetUsedGlobals {
    state: OperatorState,
    vars: Vec<VarId>,
}

impl Operator for GetUsedGlobals {
    fn state(&self) -> &OperatorState {
        &self.state
    }
    fn state_mut(&mut self) -> &mut OperatorState {
        &mut self.state
    }
    fn pre_hook(&mut self, m: &mut Module, n: NodeRef) {
        for var in m.used_variables(n) {
            let var = m.actual_var(var);
            if !m.is_func(var) && m.is_global(var) && !m.is_thread_local(var) {
                self.vars.push(var);
            }
        }
    }
}

/// Demotes global variables that are used in only one function to locals of that function.
#[derive(Default)]
pub struct GlobalDemotionPass {
    /// number of variables we've demoted
    num_demotions: i32,
}

impl GlobalDemotionPass {
    pub const KEY: &'static str = "core-cleanup-global-demote";

    pub fn new() -> Self {
        Self::default()
    }

    /// `getNumDemotions()`.
    pub fn num_demotions(&self) -> i32 {
        self.num_demotions
    }
}

impl Pass for GlobalDemotionPass {
    fn key(&self) -> String {
        Self::KEY.to_string()
    }
    fn run(&mut self, m: &mut Module, _ctx: &dyn PassContext) {
        self.num_demotions = 0;
        // Note: insertion-ordered map instead of `unordered_map` for determinism.
        let mut local_globals: IndexMap<Id, (VarId, Option<FuncId>)> = IndexMap::new();

        let mut worklist = vec![m.main_func()];
        worklist.extend(m.vars().into_iter().filter(|v| m.is_func(*v)));

        for func in worklist {
            let mut globals = GetUsedGlobals::default();
            globals.reset();
            globals.process(m, func.into());
            for g in globals.vars {
                log_flag!(IR, "[{}] global {} used in {}", Self::KEY, m.reference_string(g.into()), m.var_name(func));
                let uid = m.var_uid(g);
                match local_globals.get_mut(&uid) {
                    None => {
                        local_globals.insert(uid, (g, Some(func)));
                    }
                    Some(entry) => {
                        if entry.1.is_some_and(|f| f != func) {
                            entry.1 = None;
                        }
                    }
                }
            }
        }

        let argv = m.var_uid(m.argv_var());
        let argc = m.var_uid(m.argc_var());
        for (uid, (var, func)) in local_globals {
            let Some(func) = func else { continue };
            if uid == argv || uid == argc || m.is_external(var) {
                continue;
            }
            assert!(m.is_global(var), "var was not global [{}]", m.src_info(var.into()));
            m.set_global(var, false);
            if m.func(func).is_some_and(|f| f.is_bodied()) {
                m.func_push_symbol(func, var);
                self.num_demotions += 1;
                log_flag!(IR, "[{}] demoted {} to a local of {}", Self::KEY, m.reference_string(var.into()), m.var_name(func));
            }
        }
    }
}
