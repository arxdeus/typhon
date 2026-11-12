//! Outline a region of a series flow into a new function.

use indexmap::IndexSet;
use rustc_hash::FxHashSet;
use typhon_ir::util::cloning::CloneVisitor;
use typhon_ir::util::irtools::{call, derived_func_name, ptr_load, ptr_store};
use typhon_ir::util::operator::{Operator, OperatorState};
use typhon_ir::{FuncId, FuncKind, Id, Module, NodeRef, ValueId, ValueKind, VarId, magic};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ArgKind {
    /// Argument is not modified by outlined function
    Constant,
    /// Argument is modified and passed by pointer
    Modified,
}

#[derive(Clone, Debug, Default)]
pub struct OutlineResult {
    pub func: Option<FuncId>,
    pub call: Option<ValueId>,
    pub arg_kinds: Vec<ArgKind>,
    pub num_out_flows: i32,
}

impl OutlineResult {
    /// `operator bool()`.
    pub fn is_some(&self) -> bool {
        self.func.is_some()
    }
}

struct Outliner {
    region: ValueId,
    begin: usize,
    end: usize,
    outline_globals: bool,
    all_by_value: bool,
    in_region: bool,
    invalid: bool,
    in_vars: IndexSet<Id>,
    out_vars: IndexSet<Id>,
    modified_in_vars: IndexSet<Id>,
    globals_to_outline: IndexSet<Id>,
    in_loops: FxHashSet<Id>,
    out_flows: Vec<ValueId>,
    stack: Vec<ValueId>,
    seen: FxHashSet<Id>,
}

impl Outliner {
    fn is_loop(m: &Module, v: ValueId) -> bool {
        matches!(m.kind(v), ValueKind::WhileFlow(_) | ValueKind::ForFlow(_) | ValueKind::ImperativeForFlow(_))
    }

    fn is_enclosing_loop_in_region(&self, m: &Module, loop_id: Option<Id>) -> bool {
        for v in self.stack.iter().rev() {
            if Self::is_loop(m, *v) && (loop_id.is_none() || loop_id == Some(m.value_uid(*v))) {
                return self.in_loops.contains(&m.value_uid(*v));
            }
        }
        false
    }

    fn mark_var(&mut self, m: &Module, var: VarId) {
        let var = m.actual_var(var);
        self.modified_in_vars.insert(m.var_uid(var));
        if self.outline_globals && m.is_global(var) && !m.is_thread_local(var) {
            self.globals_to_outline.insert(m.var_uid(var));
        }
    }

    fn pre_hook(&mut self, m: &Module, n: NodeRef) {
        for var in m.used_variables(n) {
            let var = m.actual_var(var);
            if !m.is_global(var) {
                let id = m.var_uid(var);
                if self.in_region {
                    self.in_vars.insert(id);
                } else {
                    self.out_vars.insert(id);
                }
            } else if self.in_region && self.all_by_value && !m.is_func(var) {
                self.globals_to_outline.insert(m.var_uid(var));
            }
        }
    }

    fn handle(&mut self, m: &Module, v: ValueId) {
        if !self.in_region {
            return;
        }
        match m.kind(v) {
            ValueKind::WhileFlow(_) | ValueKind::ForFlow(_) | ValueKind::ImperativeForFlow(_) => {
                self.in_loops.insert(m.value_uid(v));
            }
            ValueKind::ReturnInstr(_) => self.out_flows.push(v),
            ValueKind::BreakInstr(b) => {
                let id = b.loop_.map(|l| m.value_uid(l));
                if !self.is_enclosing_loop_in_region(m, id) {
                    self.out_flows.push(v);
                }
            }
            ValueKind::ContinueInstr(b) => {
                let id = b.loop_.map(|l| m.value_uid(l));
                if !self.is_enclosing_loop_in_region(m, id) {
                    self.out_flows.push(v);
                }
            }
            ValueKind::YieldInstr(_) | ValueKind::AwaitInstr(_) | ValueKind::YieldInInstr(_) | ValueKind::StackAllocInstr(_) => {
                self.invalid = true
            }
            ValueKind::AssignInstr(a) => {
                let lhs = a.lhs;
                self.mark_var(m, lhs);
            }
            ValueKind::PointerValue(p) => {
                let var = p.var;
                self.mark_var(m, var);
            }
            _ => {}
        }
    }

    fn visit(&mut self, m: &Module, v: ValueId) {
        let v = m.actual_value(v);
        if let Some(s) = m.as_series_flow(v) {
            let items = s.series.clone();
            if m.value_uid(v) == m.value_uid(self.region) {
                for (i, c) in items.into_iter().enumerate() {
                    self.in_region = i >= self.begin && i < self.end;
                    self.visit(m, c);
                }
                self.in_region = false;
                return;
            }
            self.pre_hook(m, v.into());
            self.handle(m, v);
            self.stack.push(v);
            for c in items {
                self.visit(m, c);
            }
            self.stack.pop();
            return;
        }
        self.pre_hook(m, v.into());
        self.handle(m, v);
        let children = match m.as_call_instr(v) {
            Some(c) => {
                let mut x = c.args.clone();
                x.push(c.callee);
                x
            }
            None => m.used_values(v.into()),
        };
        self.stack.push(v);
        for c in children {
            if self.seen.insert(m.value_uid(c)) {
                self.visit(m, c);
            }
        }
        self.stack.pop();
    }

    fn shared_vars(&self) -> IndexSet<Id> {
        self.in_vars.iter().filter(|i| self.out_vars.contains(*i)).copied().collect()
    }
    fn private_vars(&self) -> IndexSet<Id> {
        self.in_vars.iter().filter(|i| !self.out_vars.contains(*i)).copied().collect()
    }
    fn mod_vars(&self) -> FxHashSet<Id> {
        if self.all_by_value {
            return FxHashSet::default();
        }
        let shared = self.shared_vars();
        self.modified_in_vars
            .iter()
            .filter(|i| self.globals_to_outline.contains(*i) || shared.contains(*i))
            .copied()
            .collect()
    }
}

struct OutlineReplacer<'a> {
    st: OperatorState,
    mod_vars: &'a FxHashSet<Id>,
    remap: &'a [(VarId, VarId)],
    out_flows: &'a mut Vec<ValueId>,
    cv: CloneVisitor,
}

impl OutlineReplacer<'_> {
    fn mapped(&self, m: &Module, v: VarId) -> VarId {
        let id = m.var_uid(v);
        self.remap.iter().find(|(o, _)| m.var_uid(*o) == id).map(|(_, n)| *n).expect("unmapped outlined var")
    }
    fn replace_out_flow(&mut self, m: &mut Module, v: ValueId) {
        let uid = m.value_uid(v);
        for i in 0..self.out_flows.len() {
            if m.value_uid(self.out_flows[i]) == uid {
                let copy = self.cv.clone_value(m, v);
                let code = m.get_int(i as i64 + 1);
                let r = m.nr_return(Some(code));
                m.replace_all_value(v, r);
                self.out_flows[i] = copy;
                break;
            }
        }
    }
}

impl Operator for OutlineReplacer<'_> {
    fn state(&self) -> &OperatorState {
        &self.st
    }
    fn state_mut(&mut self) -> &mut OperatorState {
        &mut self.st
    }
    fn post_hook(&mut self, m: &mut Module, n: NodeRef) {
        for (o, nw) in self.remap {
            m.replace_used_variable(n, *o, *nw);
        }
    }
    fn handle_return_instr(&mut self, m: &mut Module, v: ValueId) {
        self.replace_out_flow(m, v);
    }
    fn handle_break_instr(&mut self, m: &mut Module, v: ValueId) {
        self.replace_out_flow(m, v);
    }
    fn handle_continue_instr(&mut self, m: &mut Module, v: ValueId) {
        self.replace_out_flow(m, v);
    }
    fn handle_var_value(&mut self, m: &mut Module, v: ValueId) {
        let var = m.actual_var(m.as_var_value(v).unwrap().var);
        if self.mod_vars.contains(&m.var_uid(var)) {
            let nv = self.mapped(m, var);
            let vv = m.nr_var_value(nv);
            let deref = ptr_load(m, vv);
            self.see(m, deref);
            m.replace_all_value(v, deref);
        }
    }
    fn handle_pointer_value(&mut self, m: &mut Module, v: ValueId) {
        let var = m.actual_var(m.as_pointer_value(v).unwrap().var);
        if self.mod_vars.contains(&m.var_uid(var)) {
            let nv = self.mapped(m, var);
            let r = m.nr_var_value(nv);
            self.see(m, r);
            m.replace_all_value(v, r);
        }
    }
    fn handle_assign_instr(&mut self, m: &mut Module, v: ValueId) {
        let a = m.as_assign_instr(v).unwrap().clone();
        let lhs = m.actual_var(a.lhs);
        if self.mod_vars.contains(&m.var_uid(lhs)) {
            let nv = self.mapped(m, lhs);
            let vv = m.nr_var_value(nv);
            let st = ptr_store(m, vv, a.rhs);
            self.see(m, st);
            m.replace_all_value(v, st);
        }
    }
}

/// `outlineRegion(parent, series, begin, end, allowOutflows, outlineGlobals, allByValue)`:
/// outlines `series[begin..end]`.
#[allow(clippy::too_many_arguments)]
pub fn outline_region(
    m: &mut Module,
    parent: FuncId,
    series: ValueId,
    begin: usize,
    end: usize,
    allow_outflows: bool,
    outline_globals: bool,
    all_by_value: bool,
) -> OutlineResult {
    if begin >= end {
        return OutlineResult::default();
    }
    let series = m.actual_value(series);
    let mut o = Outliner {
        region: series,
        begin,
        end,
        outline_globals,
        all_by_value,
        in_region: false,
        invalid: false,
        in_vars: IndexSet::new(),
        out_vars: IndexSet::new(),
        modified_in_vars: IndexSet::new(),
        globals_to_outline: IndexSet::new(),
        in_loops: FxHashSet::default(),
        out_flows: Vec::new(),
        stack: Vec::new(),
        seen: FxHashSet::default(),
    };
    // visit(BodiedFunc)
    for a in m.func_args(parent) {
        o.out_vars.insert(m.var_uid(a));
    }
    if let Some(body) = m.func_body(parent) {
        o.seen.insert(m.value_uid(body));
        o.visit(m, body);
    }
    if o.invalid {
        return OutlineResult::default();
    }

    let mut shared = o.shared_vars();
    shared.extend(o.globals_to_outline.iter().copied());
    let modv = o.mod_vars();
    let mut remap: Vec<(VarId, Option<VarId>)> = Vec::new();
    let mut arg_types = Vec::new();
    let mut arg_names = Vec::new();
    let mut arg_kinds = Vec::new();
    for id in &shared {
        let var = m.get_var(*id).unwrap_or_else(|| panic!("unknown var id [{id}]"));
        remap.push((var, None));
        let is_mod = modv.contains(id);
        let ty = m.var_type(var).expect("var without type");
        let ty = if is_mod { m.get_pointer_type(Some(ty)) } else { ty };
        arg_types.push(ty);
        arg_names.push(m.var_name(var).to_string());
        arg_kinds.push(if is_mod { ArgKind::Modified } else { ArgKind::Constant });
    }
    let call_indicates_control = !o.out_flows.is_empty();
    if call_indicates_control && !allow_outflows {
        return OutlineResult::default();
    }
    let ret = if call_indicates_control { m.get_int_type() } else { m.get_none_type() };
    let func_type = m.get_func_type(ret, arg_types, false);
    let items = m.as_series_flow(series).unwrap().series.clone();
    let first = items[begin];
    let name = derived_func_name(m, "__outlined", parent, Some(first.into()));
    let src = m.src_info(first.into());
    let outlined = m.n_func(src.clone(), name, FuncKind::bodied());
    m.realize_func(outlined, func_type, &arg_names);
    for (i, a) in m.func_args(outlined).into_iter().enumerate() {
        remap[i].1 = Some(a);
    }
    let mut remap: Vec<(VarId, VarId)> = remap.into_iter().map(|(a, b)| (a, b.unwrap())).collect();
    for id in o.private_vars() {
        let var = m.get_var(id).unwrap_or_else(|| panic!("unknown var id [{id}]"));
        let vsrc = m.src_info(var.into());
        let ty = m.var_type(var);
        let name = m.var_name(var).to_string();
        let nv = m.n_var(vsrc, ty, false, false, false, name);
        remap.push((var, nv));
        m.func_push_symbol(outlined, nv);
    }

    let body = m.n(src, ValueKind::SeriesFlow(Default::default()));
    for v in &items[begin..end] {
        m.series_push(body, *v);
    }
    m.as_series_flow_mut(series).unwrap().series.drain(begin..end);
    m.set_func_body(outlined, Some(body));

    let mut out_flows = o.out_flows.clone();
    {
        let mut rep = OutlineReplacer { st: OperatorState::new(false), mod_vars: &modv, remap: &remap, out_flows: &mut out_flows, cv: CloneVisitor::new(false) };
        rep.reset();
        rep.process(m, body.into());
    }

    let mut args = Vec::new();
    for (var, _) in remap.iter().take(shared.len()) {
        let a = if modv.contains(&m.var_uid(*var)) { m.nr_pointer_value(*var, Vec::new()) } else { m.nr_var_value(*var) };
        args.push(a);
    }
    let outlined_call = call(m, outlined, args);
    let mut it = begin;
    if call_indicates_control {
        let int_ty = m.get_int_type();
        let code_var = m.nr_var(int_ty, false, "");
        m.func_push_symbol(parent, code_var);
        let asg = m.nr_assign(code_var, outlined_call);
        m.series_insert(series, it, asg);
        for (i, of) in out_flows.iter().enumerate() {
            let code_val = m.get_int(i as i64 + 1);
            let vv = m.nr_var_value(code_var);
            let check = m.binary_op(code_val, magic::EQ, vv).expect("int __eq__ not found");
            let body = m.nr_series_flow();
            m.series_push(body, *of);
            let iff = m.nr_if_flow(check, body, None);
            it += 1;
            m.series_insert(series, it, iff);
        }
    } else {
        m.series_insert(series, it, outlined_call);
    }
    OutlineResult { func: Some(outlined), call: Some(outlined_call), arg_kinds, num_out_flows: out_flows.len() as i32 }
}

/// `outlineRegion(parent, series, allowOutflows, outlineGlobals, allByValue)`: outlines the whole series.
pub fn outline_series(
    m: &mut Module,
    parent: FuncId,
    series: ValueId,
    allow_outflows: bool,
    outline_globals: bool,
    all_by_value: bool,
) -> OutlineResult {
    let n = m.as_series_flow(series).map(|s| s.series.len()).unwrap_or(0);
    outline_region(m, parent, series, 0, n, allow_outflows, outline_globals, all_by_value)
}
