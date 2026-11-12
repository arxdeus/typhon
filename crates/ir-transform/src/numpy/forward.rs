//!
//! Note: units are referenced by their index in the function's unit vector
//! (`NumPyOptimizationUnit*`), expression nodes by [`ExprId`]. DAGs are
//! `IndexMap`s so iteration order is deterministic (uses `unordered_map`).

use indexmap::IndexMap;
use rustc_hash::{FxHashMap, FxHashSet};
use typhon_ir::util::operator::Operator;
use typhon_ir::{Id, Module, ValueId, VarId};
use typhon_ir_analyze::dataflow::cfg::{BlockId, CFGraph, CfValue, SyntheticKind};
use typhon_ir_analyze::dataflow::cfg::CfVar;
use typhon_ir_analyze::dataflow::reaching::RDInspector;
use typhon_ir_analyze::module::side_effect::SideEffectResult;

use super::NumPyOptimizationUnit;
use super::expr::{ExprArena, ExprId};
use super::numpy::{Walk, has_owned_result_m};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Forwarding {
    pub dst: usize,
    pub src: usize,
    pub var: VarId,
    pub dst_leaf: ExprId,
    pub dst_id: i64,
    pub src_id: i64,
}

/// Port of `ForwardingDAG` (unit index -> forwardings into that unit).
pub type ForwardingDAG = IndexMap<usize, Vec<Forwarding>>;

/// `GetVars`: non-function variables used anywhere under `v`.
fn get_vars(m: &mut Module, v: ValueId, vids: &mut FxHashSet<Id>) {
    Walk::new().run(m, v, &mut |_, m, x| {
        for var in m.used_variables(x.into()) {
            if !m.is_func(var) {
                vids.insert(m.var_uid(var));
            }
        }
    });
}

/// `OkToForwardPast`.
fn ok_to_forward_past(
    m: &mut Module,
    v: ValueId,
    vids: &FxHashSet<Id>,
    parsed: &FxHashMap<Id, ExprId>,
    arena: &ExprArena,
    se: &SideEffectResult,
) -> bool {
    let mut ok = true;
    Walk::new().run(m, v, &mut |w, m, x| {
        if !ok {
            return;
        }
        if let Some(a) = m.as_assign_instr(x) {
            let lhs = a.lhs;
            if vids.contains(&m.var_uid(lhs)) {
                ok = false;
            }
        } else if let Some(&e) = parsed.get(&m.value_uid(x)) {
            for y in arena.preorder(e) {
                if arena[y].is_leaf() && arena[y].val.is_some_and(|val| se.has_side_effect(m, val)) {
                    ok = false;
                }
            }
            // Skip children since we are processing them manually above.
            for used in m.used_values(x.into()) {
                w.see(m, used);
            }
        } else if se.has_side_effect(m, x) {
            ok = false;
        }
    });
    ok
}

/// `GetAllUses`: values (anywhere in `func`) whose used variables include `var`.
pub(crate) fn get_all_uses(m: &mut Module, func: typhon_ir::FuncId, var: VarId) -> Vec<ValueId> {
    let mut uses = vec![];
    let var = m.actual_var(var);
    Walk::new().run_func(m, func, &mut |_, m, x| {
        if m.used_variables(x.into()).iter().any(|u| m.actual_var(*u) == var) {
            uses.push(x);
        }
    });
    uses
}

fn block_of(cfg: &CFGraph, m: &Module, v: ValueId) -> Option<BlockId> {
    cfg.get_block(m, v)
}

/// Synthetic CFG values: an assignment to a tracked variable blocks forwarding.
/// Note: the reference implementation runs the operator on synthetic instructions; they carry no
/// side-effect result, so only their (real) assigned variable is checked here.
fn ok_synthetic(cfg: &CFGraph, m: &Module, v: CfValue, vids: &FxHashSet<Id>) -> bool {
    if let CfValue::Synth(s) = v
        && let SyntheticKind::Assign(a) = &cfg.synth(s).kind
        && let CfVar::Real(var) = a.lhs
    {
        return !vids.contains(&m.var_uid(var));
    }
    true
}

#[allow(clippy::too_many_arguments)]
fn can_forward_expression_in_block(
    m: &mut Module,
    source: Option<ValueId>,
    destination: Option<ValueId>,
    vids: &FxHashSet<Id>,
    parsed: &FxHashMap<Id, ExprId>,
    arena: &ExprArena,
    se: &SideEffectResult,
    cfg: &CFGraph,
    block: BlockId,
) -> bool {
    let mut go = source.is_none();
    let values: Vec<CfValue> = cfg.block(block).values().to_vec();
    for value in values {
        if !go {
            go = source.is_some_and(|s| value == CfValue::Real(m.actual_value(s)) || value == CfValue::Real(s));
            continue;
        }
        if let Some(d) = destination
            && (value == CfValue::Real(d) || value == CfValue::Real(m.actual_value(d)))
        {
            return true;
        }
        match value {
            CfValue::Real(v) => {
                if !ok_to_forward_past(m, v, vids, parsed, arena, se) {
                    return false;
                }
            }
            s => {
                if !ok_synthetic(cfg, m, s, vids) {
                    return false;
                }
            }
        }
    }
    go && destination.is_none()
}

#[allow(clippy::too_many_arguments)]
fn can_forward_expression(
    m: &mut Module,
    units: &[NumPyOptimizationUnit],
    arena: &ExprArena,
    expr: usize,
    target: ValueId,
    parsed: &FxHashMap<Id, ExprId>,
    cfg: &CFGraph,
    se: &SideEffectResult,
) -> bool {
    let mut vids = FxHashSet::default();
    let mut pure = true;
    for e in arena.preorder(units[expr].expr) {
        if arena[e].is_leaf() {
            let val = arena[e].v();
            if se.has_side_effect(m, val) {
                pure = false;
            } else {
                get_vars(m, val, &mut vids);
            }
        }
    }
    if !pure {
        return false;
    }
    let source = units[expr].assign.expect("forwarded unit has no assignment");
    let (Some(start), Some(end)) = (block_of(cfg, m, source), block_of(cfg, m, target)) else {
        panic!("start/end CFG block not found");
    };
    // A consumer must not run again without reexecuting its producer. Single-use
    // syntax alone does not prove this when the consumer is inside a loop.
    let mut visited = FxHashSet::default();
    let mut pending: Vec<BlockId> = cfg.block(end).successors().collect();
    while let Some(curr) = pending.pop() {
        if curr == start {
            continue;
        }
        if curr == end {
            return false;
        }
        if visited.insert(curr) {
            pending.extend(cfg.block(curr).successors());
        }
    }

    // Every path must reach the consumer, or forwarding could suppress an eager
    // shape error. Also reject cycles and intervening writes/effects.
    let mut blocks = vec![start];
    let mut incoming: IndexMap<BlockId, usize> = IndexMap::new();
    incoming.insert(start, 0);
    let mut index = 0;
    while index < blocks.len() {
        let block = blocks[index];
        index += 1;
        if !can_forward_expression_in_block(
            m,
            if block == start { Some(source) } else { None },
            if block == end { Some(target) } else { None },
            &vids,
            parsed,
            arena,
            se,
            cfg,
            block,
        ) {
            return false;
        }
        if block == end {
            continue;
        }
        if cfg.block(block).num_successors() == 0 {
            return false;
        }
        for succ in cfg.block(block).successors() {
            let fresh = !incoming.contains_key(&succ);
            *incoming.entry(succ).or_insert(0) += 1;
            if fresh {
                blocks.push(succ);
            }
        }
    }
    if !incoming.contains_key(&end) {
        return false;
    }
    let mut ready: Vec<BlockId> = incoming.iter().filter(|(_, c)| **c == 0).map(|(b, _)| *b).collect();
    let mut i = 0;
    while i < ready.len() {
        let block = ready[i];
        i += 1;
        if block == end {
            continue;
        }
        for succ in cfg.block(block).successors() {
            let c = incoming.get_mut(&succ).unwrap();
            *c -= 1;
            if *c == 0 {
                ready.push(succ);
            }
        }
    }
    ready.len() == blocks.len()
}

/// `canForwardVariable(assign, destination, func, rd)`.
pub(crate) fn can_forward_variable(
    m: &mut Module,
    assign: ValueId,
    destination: ValueId,
    func: typhon_ir::FuncId,
    rd: &RDInspector,
) -> bool {
    let var = m.as_assign_instr(assign).expect("not an assignment").lhs;
    let assign_uid = m.value_uid(assign);
    // Check 1: Only the given assignment should reach the destination.
    let reaching = rd.get_reaching_definitions(m, var, destination);
    if reaching.len() != 1 || reaching[0].assignment_uid() != assign_uid {
        return false;
    }
    // Check 2: There should be no other uses reached by this assignment. These are
    // individual references, so even repeated operands in one consumer disqualify it.
    let dest = m.actual_value(destination);
    for u in get_all_uses(m, func, var) {
        if u != dest && m.value_uid(u) != assign_uid {
            for d in rd.get_reaching_definitions(m, var, u) {
                if d.assignment_uid() == assign_uid {
                    return false;
                }
            }
        }
    }
    true
}

/// Edges point from a consumer to a producer that can replace one of its leaves.
/// Reaching definitions establish single-use; CFG/effect checks require the consumer
/// on every path and reject intervening writes before moving the producer.
fn build_forwarding_dag(
    m: &mut Module,
    func: typhon_ir::FuncId,
    rd: &RDInspector,
    cfg: &CFGraph,
    se: &SideEffectResult,
    arena: &ExprArena,
    units: &[NumPyOptimizationUnit],
) -> ForwardingDAG {
    let mut parsed: FxHashMap<Id, ExprId> = FxHashMap::default();
    for u in units {
        // A summarized helper still occupies its original call site in the CFG.
        // Its proven expression, rather than the opaque call's effects, determines
        // whether another producer may move past it.
        parsed.entry(m.value_uid(u.value)).or_insert(u.expr);
        for e in arena.preorder(u.expr) {
            if let Some(v) = arena[e].val {
                parsed.entry(m.value_uid(v)).or_insert(e);
            }
        }
    }

    let mut dag = ForwardingDAG::new();
    for (dst_id, dst) in units.iter().enumerate() {
        let mut vec = vec![];
        let mut vars: Vec<(VarId, ExprId)> = vec![];
        let mut occurrences: FxHashMap<ValueId, u32> = FxHashMap::default();
        for e in arena.preorder(dst.expr) {
            if arena[e].is_leaf()
                && let Some(v) = arena[e].val.and_then(|v| m.as_var_value(v))
            {
                vars.push((m.actual_var(v.var), e));
                *occurrences.entry(arena[e].v()).or_insert(0) += 1;
            }
        }
        for (var, leaf) in vars {
            // A composed helper may read one actual more than once (e.g. square(t)).
            // Those leaves share the original call-site read so RD remains valid, but
            // forwarding its producer into both would duplicate work and allocations.
            if occurrences[&arena[leaf].v()] != 1 {
                continue;
            }
            for (src_id, src) in units.iter().enumerate() {
                let Some(sa) = src.assign else { continue };
                if src_id == dst_id || m.actual_var(m.as_assign_instr(sa).unwrap().lhs) != var {
                    continue;
                }
                let check = can_forward_variable(m, sa, arena[leaf].v(), func, rd);
                // Keep reductions materialized. Their buffers can be reused later
                // without moving the reduction itself into another expression.
                if check
                    && !arena[src.expr].is_reduction()
                    && can_forward_expression(m, units, arena, src_id, arena[leaf].v(), &parsed, cfg, se)
                {
                    vec.push(Forwarding {
                        dst: dst_id,
                        src: src_id,
                        var,
                        dst_leaf: leaf,
                        dst_id: dst_id as i64,
                        src_id: src_id as i64,
                    });
                }
            }
        }
        dag.insert(dst_id, vec);
    }
    dag
}

#[derive(Clone, Debug)]
pub struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
}

impl UnionFind {
    pub fn new(n: usize) -> Self {
        Self { parent: (0..n).collect(), rank: vec![0; n] }
    }
    pub fn find(&mut self, u: usize) -> usize {
        if self.parent[u] != u {
            let p = self.find(self.parent[u]);
            self.parent[u] = p;
        }
        self.parent[u]
    }
    pub fn union(&mut self, u: usize, v: usize) {
        let (ru, rv) = (self.find(u), self.find(v));
        if ru != rv {
            if self.rank[ru] > self.rank[rv] {
                self.parent[rv] = ru;
            } else if self.rank[ru] < self.rank[rv] {
                self.parent[ru] = rv;
            } else {
                self.parent[rv] = ru;
                self.rank[ru] += 1;
            }
        }
    }
}

/// `getForwardingDAGConnectedComponents`.
pub fn connected_components(dag: &ForwardingDAG, n: usize) -> Vec<ForwardingDAG> {
    let mut uf = UnionFind::new(n);
    for i in 0..n {
        for f in dag.get(&i).map(|v| v.as_slice()).unwrap_or(&[]) {
            uf.union(i, f.src_id as usize);
        }
    }
    let mut components: Vec<Vec<usize>> = vec![vec![]; n];
    for i in 0..n {
        let r = uf.find(i);
        components[r].push(i);
    }
    components
        .into_iter()
        .filter(|c| !c.is_empty())
        .map(|c| c.into_iter().map(|e| (e, dag.get(&e).cloned().unwrap_or_default())).collect())
        .collect()
}

fn has_cycle_helper(v: usize, dag: &ForwardingDAG, visited: &mut [bool], rec: &mut [bool]) -> bool {
    visited[v] = true;
    rec[v] = true;
    for nb in dag.get(&v).map(|x| x.as_slice()).unwrap_or(&[]) {
        let s = nb.src_id as usize;
        if !visited[s] {
            if has_cycle_helper(s, dag, visited, rec) {
                return true;
            }
        } else if rec[s] {
            return true;
        }
    }
    rec[v] = false;
    false
}

/// `hasCycle(dag, exprs)`.
pub fn has_cycle(dag: &ForwardingDAG, n: usize) -> bool {
    let mut visited = vec![false; n];
    let mut rec = vec![false; n];
    (0..n).any(|i| dag.contains_key(&i) && !visited[i] && has_cycle_helper(i, dag, &mut visited, &mut rec))
}

fn do_forwarding_helper(
    dag: &ForwardingDAG,
    arena: &mut ExprArena,
    units: &mut [NumPyOptimizationUnit],
    curr: usize,
    done: &mut FxHashSet<usize>,
    assigns_to_delete: &mut Vec<ValueId>,
    substitutions: &mut Option<&mut Vec<(ValueId, ValueId)>>,
) {
    if done.contains(&curr) {
        return;
    }
    let fwds = dag.get(&curr).cloned().unwrap_or_default();
    for f in fwds {
        do_forwarding_helper(dag, arena, units, f.src, done, assigns_to_delete, substitutions);
        // Note that order of leaves here doesn't matter since they're guaranteed to have no
        // side effects based on forwarding checks.
        let src_leaves = units[f.src].leaves.clone();
        units[f.dst].leaves.extend(src_leaves);
        if let Some(s) = substitutions.as_mut() {
            s.push((arena[f.dst_leaf].v(), units[f.src].value));
        }
        let src_root = units[f.src].expr;
        arena.replace(f.dst_leaf, src_root);
        assigns_to_delete.push(units[f.src].assign.expect("forwarded unit has no assignment"));
    }
    done.insert(curr);
}

/// `getForwardingRoot(dag)`.
pub fn get_forwarding_root(dag: &ForwardingDAG) -> usize {
    let not_root: FxHashSet<usize> = dag.values().flat_map(|v| v.iter().map(|f| f.src)).collect();
    assert!(not_root.len() + 1 == dag.len(), "multiple roots found in forwarding DAG");
    *dag.keys().find(|k| !not_root.contains(k)).expect("could not find root in forwarding DAG")
}

/// Allow a final pointwise consumer to reuse a buffer after earlier reductions
/// have finished reading it. Any other reached use prevents this ownership transfer.
#[allow(clippy::too_many_arguments)]
fn can_reuse_after_reads(
    m: &mut Module,
    source: usize,
    destination: ExprId,
    consumer: usize,
    units: &[NumPyOptimizationUnit],
    arena: &ExprArena,
    cfg: &CFGraph,
    rd: &RDInspector,
    se: &SideEffectResult,
) -> bool {
    let ce = units[consumer].expr;
    if arena[ce].is_reduction()
        || arena.depth(ce) != 2
        || (Some(destination) != arena[ce].lhs && Some(destination) != arena[ce].rhs)
    {
        return false;
    }
    let assign = units[source].assign.unwrap();
    let assign_uid = m.value_uid(assign);
    let var = m.as_assign_instr(assign).unwrap().lhs;
    let dval = arena[destination].v();
    let reaching = rd.get_reaching_definitions(m, var, dval);
    if reaching.len() != 1 || reaching[0].assignment_uid() != assign_uid {
        return false;
    }
    let Some(block) = block_of(cfg, m, assign) else { return false };
    let mut positions: FxHashMap<Id, usize> = FxHashMap::default();
    for v in cfg.block(block).values() {
        let n = positions.len();
        positions.entry(cfg.uid(m, *v)).or_insert(n);
    }
    let Some(&target) = positions.get(&m.value_uid(dval)) else { return false };
    let apos = positions[&assign_uid];

    let mut completed: FxHashSet<Id> = FxHashSet::default();
    for reader in units {
        let e = &arena[reader.expr];
        let Some(lhs) = e.lhs else { continue };
        if !e.is_reduction() || !arena[lhs].is_leaf() {
            continue;
        }
        let Some(&pos) = positions.get(&m.value_uid(reader.value)) else { continue };
        if pos <= apos || pos >= target {
            continue;
        }
        let operand = arena[lhs].v();
        let Some(ov) = m.as_var_value(operand).map(|x| m.actual_var(x.var)) else { continue };
        let Some(args) = e.val.and_then(|v| m.as_call_instr(v)).map(|c| c.args.clone()) else { continue };
        if ov != m.actual_var(var) {
            continue;
        }
        let pure = args.iter().skip(1).all(|a| !se.has_side_effect(m, *a));
        if pure {
            completed.insert(m.value_uid(operand));
        }
    }
    if completed.is_empty() {
        return false;
    }
    // Alias/view creation is also a use. Reject it even if the alias is not returned:
    // it could mutate or retain the allocation without reading this variable again.
    let dact = m.actual_value(dval);
    for u in get_all_uses(m, units[consumer].func, var) {
        if u == dact || m.value_uid(u) == assign_uid {
            continue;
        }
        for d in rd.get_reaching_definitions(m, var, u) {
            if d.assignment_uid() == assign_uid && !completed.contains(&m.value_uid(u)) {
                return false;
            }
        }
    }
    true
}

/// `getForwardingDAGs(func, rd, cfg, se, exprs)`.
pub fn get_forwarding_dags(
    m: &mut Module,
    func: typhon_ir::FuncId,
    rd: &RDInspector,
    cfg: &CFGraph,
    se: &SideEffectResult,
    arena: &mut ExprArena,
    units: &[NumPyOptimizationUnit],
) -> Vec<ForwardingDAG> {
    let n = units.len();
    let dag = build_forwarding_dag(m, func, rd, cfg, se, arena, units);
    let mut dags = connected_components(&dag, n);
    dags.retain(|d| !has_cycle(d, n));
    // Transfer ownership separately from expression forwarding. An ownedLastUse leaf
    // proves ownership and no subsequent alias access, permitting reuse or release
    // without moving the source computation or adding a forwarding edge.
    for component in &dags {
        let root = get_forwarding_root(component);
        let block = block_of(cfg, m, units[root].value);
        let mut leaf_occ: FxHashMap<VarId, u32> = FxHashMap::default();
        for &u in component.keys() {
            for e in arena.preorder(units[u].expr) {
                if arena[e].is_leaf()
                    && let Some(v) = arena[e].val.and_then(|v| m.as_var_value(v))
                {
                    *leaf_occ.entry(m.actual_var(v.var)).or_insert(0) += 1;
                }
            }
        }
        for (si, source) in units.iter().enumerate() {
            let Some(sa) = source.assign else { continue };
            let Some(block) = block else { continue };
            let svar = m.actual_var(m.as_assign_instr(sa).unwrap().lhs);
            if component.contains_key(&si)
                || leaf_occ.get(&svar).copied().unwrap_or(0) != 1
                || !has_owned_result_m(m, arena, source.expr)
                || m.is_global(svar)
                || block_of(cfg, m, sa) != Some(block)
            {
                continue;
            }
            let mut available = false;
            let root_v = CfValue::Real(m.actual_value(units[root].value));
            let sa_v = CfValue::Real(m.actual_value(sa));
            for v in cfg.block(block).values() {
                if *v == root_v {
                    break;
                }
                if *v == sa_v {
                    available = true;
                }
            }
            if !available {
                continue;
            }
            for &u in component.keys() {
                for e in arena.preorder(units[u].expr) {
                    let Some(val) = arena[e].val else { continue };
                    if !arena[e].is_leaf() {
                        continue;
                    }
                    let Some(v) = m.as_var_value(val).map(|x| m.actual_var(x.var)) else { continue };
                    if v != svar {
                        continue;
                    }
                    let ok = (arena[source.expr].is_reduction() && can_forward_variable(m, sa, val, source.func, rd))
                        || (u == root && can_reuse_after_reads(m, si, e, root, units, arena, cfg, rd, se));
                    if ok {
                        arena[e].owned_last_use = true;
                    }
                }
            }
        }
    }
    dags
}

/// `doForwarding(dag, assignsToDelete, substitutions)`: returns the root unit index.
pub fn do_forwarding(
    dag: &ForwardingDAG,
    arena: &mut ExprArena,
    units: &mut [NumPyOptimizationUnit],
    assigns_to_delete: &mut Vec<ValueId>,
    mut substitutions: Option<&mut Vec<(ValueId, ValueId)>>,
) -> usize {
    assert!(!dag.is_empty(), "empty forwarding DAG encountered");
    let mut done = FxHashSet::default();
    for &e in dag.keys() {
        do_forwarding_helper(dag, arena, units, e, &mut done, assigns_to_delete, &mut substitutions);
    }
    get_forwarding_root(dag)
}
