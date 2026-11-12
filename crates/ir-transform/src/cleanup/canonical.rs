
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use once_cell::sync::Lazy;
use typhon_common::mangle::func as mangled;
use typhon_ir::magic;
use typhon_ir::pass::{Pass, PassContext, run_operator_pass};
use typhon_ir::rewrite::{RewriteRule, Rewriter};
use typhon_ir::util::irtools::{call, get_const, get_func, has_attribute, is_call_of, is_call_of_types, is_const};
use typhon_ir::util::matching::match_nodes;
use typhon_ir::util::operator::{Operator, OperatorState};
use typhon_ir::{FuncId, Module, NodeRef, TypeId, ValueId};
use typhon_ir_analyze::module::side_effect::SideEffectResult;

static COMMUTATIVE_ATTR: Lazy<String> = Lazy::new(|| mangled("std.internal.attributes", "commutative"));
static ASSOCIATIVE_ATTR: Lazy<String> = Lazy::new(|| mangled("std.internal.attributes", "associative"));
static DISTRIBUTIVE_ATTR: Lazy<String> = Lazy::new(|| mangled("std.internal.attributes", "distributive"));

/// Rank: (constant last, deeper first, hash).
type Rank = (i32, i64, u64);

/// Nodes are ranked lexicographically by constness, max depth and a hash.
struct NodeRanker {
    state: OperatorState,
    root: Option<NodeRef>,
    max_depth: usize,
    hash: u64,
}

impl NodeRanker {
    fn hash_combine(&mut self, s: &str) {
        // Note: `std::hash<std::string>` replaced by `DefaultHasher` (fixed keys,
        // deterministic); only the ordering it induces matters.
        let mut h = DefaultHasher::new();
        s.hash(&mut h);
        let hv = h.finish();
        self.hash ^= hv.wrapping_add(0x9e3779b9).wrapping_add(self.hash << 6).wrapping_add(self.hash >> 2);
    }
}

impl Operator for NodeRanker {
    fn state(&self) -> &OperatorState {
        &self.state
    }
    fn state_mut(&mut self) -> &mut OperatorState {
        &mut self.state
    }
    fn pre_hook(&mut self, m: &mut Module, n: NodeRef) {
        if self.root.is_none() {
            self.root = Some(n);
        }
        self.max_depth = self.max_depth.max(self.depth());
        for v in m.used_variables(n) {
            let name = m.var_name(v).to_string();
            self.hash_combine(&name);
        }
        for t in m.used_types(n) {
            let name = m.type_name(t).to_string();
            self.hash_combine(&name);
        }
    }
}

fn get_rank(m: &mut Module, v: ValueId) -> Rank {
    let mut r = NodeRanker { state: OperatorState::new(false), root: None, max_depth: 0, hash: 0 };
    r.process(m, v.into());
    let is_c = matches!(r.root, Some(NodeRef::Value(x)) if m.is_const(x));
    (if is_c { 1 } else { -1 }, -(r.max_depth as i64), r.hash)
}

fn fn_has(m: &Module, f: Option<FuncId>, attr: &str) -> bool {
    f.is_some_and(|f| has_attribute(m, f, attr))
}
fn is_commutative_op(m: &Module, f: Option<FuncId>) -> bool {
    fn_has(m, f, &COMMUTATIVE_ATTR)
}
fn is_associative_op(m: &Module, f: Option<FuncId>) -> bool {
    fn_has(m, f, &ASSOCIATIVE_ATTR)
}
fn is_distributive_op(m: &Module, f: Option<FuncId>) -> bool {
    fn_has(m, f, &DISTRIBUTIVE_ATTR)
}
fn is_inequality_op(m: &Module, f: Option<FuncId>) -> bool {
    f.is_some_and(|f| {
        let n = m.func(f).unwrap().unmangled_name.as_str();
        [magic::EQ, magic::NE, magic::LT, magic::LE, magic::GT, magic::GE].contains(&n)
    })
}

fn callee_func(m: &Module, v: ValueId) -> Option<FuncId> {
    m.as_call_instr(v).and_then(|c| get_func(m, c.callee))
}
fn args(m: &Module, v: ValueId) -> Vec<ValueId> {
    m.as_call_instr(v).map(|c| c.args.clone()).unwrap_or_default()
}
fn same(m: &Module, a: ValueId, b: ValueId) -> bool {
    m.actual_value(a) == m.actual_value(b)
}
fn matches(m: &Module, a: ValueId, b: ValueId) -> bool {
    match_nodes(m, Some(a.into()), Some(b.into()), false, true)
}

/// c + b + a --> a + b + c
struct CanonOpChain;

impl CanonOpChain {
    fn extract_associative_op_chain(m: &mut Module, v: ValueId, op: &str, ty: TypeId, result: &mut Vec<ValueId>) {
        let t = Some(ty);
        if is_call_of_types(m, v, op, &[t, t], t, true) {
            let a = args(m, v);
            Self::extract_associative_op_chain(m, a[0], op, ty, result);
            Self::extract_associative_op_chain(m, a[1], op, ty, result);
        } else {
            result.push(v);
        }
    }

    fn order_operands(m: &mut Module, operands: &mut Vec<ValueId>) {
        let mut ranked: Vec<(Rank, usize, ValueId)> =
            operands.iter().enumerate().map(|(i, v)| (get_rank(m, *v), i, *v)).collect();
        // Note: the reference sorts (rank, pointer) pairs; ties are broken by original position.
        ranked.sort();
        *operands = ranked.into_iter().map(|x| x.2).collect();
    }
}

impl RewriteRule for CanonOpChain {
    fn apply(&mut self, m: &mut Module, v: ValueId) -> Option<ValueId> {
        let f = callee_func(m, v)?;
        let op = m.func(f).unwrap().unmangled_name.clone();
        let ty = m.value_type_mut(v);
        let is_assoc = is_associative_op(m, Some(f));
        let is_comm = is_commutative_op(m, Some(f));
        let t = Some(ty);
        if !is_call_of_types(m, v, &op, &[t, t], t, true) {
            return None;
        }
        let mut operands = Vec::new();
        if is_assoc {
            Self::extract_associative_op_chain(m, v, &op, ty, &mut operands);
        } else {
            let a = args(m, v);
            operands.push(a[0]);
            operands.push(a[1]);
        }
        assert!(operands.len() >= 2, "bad call canonicalization");
        if is_comm {
            Self::order_operands(m, &mut operands);
        }

        let mut left = v;
        let mut remaining = operands.len();
        while remaining > 1 {
            let Some(c) = m.as_call_instr(left) else { break };
            let (callee, cargs) = (c.callee, c.args.clone());
            if cargs.len() != 2
                || get_func(m, callee) != Some(f)
                || !m.value_name(left).is_empty()
                || !m.value_name(callee).is_empty()
                || !same(m, cargs[1], operands[remaining - 1])
            {
                break;
            }
            left = cargs[0];
            remaining -= 1;
        }
        if remaining == 1 && same(m, left, operands[0]) {
            return None;
        }

        let mut new_call = call(m, f, vec![operands[0], operands[1]]);
        for x in &operands[2..] {
            new_call = call(m, f, vec![new_call, *x]);
        }
        if !matches(m, v, new_call) { Some(new_call) } else { None }
    }
}

/// b > a --> a < b (etc.)
struct CanonInequality;

impl RewriteRule for CanonInequality {
    fn apply(&mut self, m: &mut Module, v: ValueId) -> Option<ValueId> {
        let f = callee_func(m, v)?;
        let op = m.func(f).unwrap().unmangled_name.clone();
        let ty = m.value_type_mut(v);
        let a = args(m, v);
        if a.len() != 2 || !is_inequality_op(m, Some(f)) {
            return None;
        }
        let (lhs, rhs) = (a[0], a[1]);
        if get_rank(m, lhs) <= get_rank(m, rhs) {
            return None;
        }
        let flipped = match op.as_str() {
            magic::EQ => magic::EQ,
            magic::NE => magic::NE,
            magic::LT => magic::GT,
            magic::LE => magic::GE,
            magic::GT => magic::LT,
            magic::GE => magic::LE,
            _ => panic!("unknown comparison op: {op}"),
        };
        let new_call = m.binary_op(rhs, flipped, lhs)?;
        let nt = m.value_type_mut(new_call);
        if m.type_is(nt, ty) && !matches(m, v, new_call) { Some(new_call) } else { None }
    }
}

/// a*x + b*x --> (a + b) * x
struct CanonAddMul;

impl CanonAddMul {
    fn var_match(m: &Module, a: ValueId, b: ValueId) -> bool {
        match (m.as_var_value(a), m.as_var_value(b)) {
            (Some(x), Some(y)) => m.var_uid(x.var) == m.var_uid(y.var),
            _ => false,
        }
    }

    fn get_op(m: &Module, v: ValueId) -> Option<FuncId> {
        callee_func(m, v)
    }

    /// (a + b) * x, or None if invalid
    fn add_mul(m: &mut Module, a: ValueId, b: ValueId, x: ValueId) -> Option<ValueId> {
        let y = match m.binary_op(a, magic::ADD, b) {
            Some(y) => y,
            None => {
                let y = m.binary_op(b, magic::ADD, a)?;
                if !is_commutative_op(m, Self::get_op(m, y)) {
                    return None;
                }
                y
            }
        };
        match m.binary_op(y, magic::MUL, x) {
            Some(z) => Some(z),
            None => {
                let z = m.binary_op(x, magic::MUL, y)?;
                if !is_commutative_op(m, Self::get_op(m, z)) {
                    return None;
                }
                Some(z)
            }
        }
    }
}

impl RewriteRule for CanonAddMul {
    fn apply(&mut self, m: &mut Module, v: ValueId) -> Option<ValueId> {
        let f = callee_func(m, v);
        if !is_commutative_op(m, f) || !is_call_of(m, v, magic::ADD, 2, None, true) {
            return None;
        }
        let a = args(m, v);
        let (lhs, rhs) = (a[0], a[1]);
        let decompose = |m: &mut Module, x: ValueId| -> (ValueId, ValueId) {
            if is_call_of(m, x, magic::MUL, 2, None, true) {
                let xa = args(m, x);
                (xa[0], xa[1])
            } else {
                (x, m.get_int(1))
            }
        };
        let (lhs1, lhs2) = decompose(m, lhs);
        let (rhs1, rhs2) = decompose(m, rhs);

        let new_call = if Self::var_match(m, lhs1, rhs1) {
            Self::add_mul(m, lhs2, rhs2, lhs1)
        } else if Self::var_match(m, lhs1, rhs2) {
            Self::add_mul(m, lhs2, rhs1, lhs1)
        } else if Self::var_match(m, lhs2, rhs1) {
            Self::add_mul(m, lhs1, rhs2, lhs2)
        } else if Self::var_match(m, lhs2, rhs2) {
            Self::add_mul(m, lhs1, rhs1, lhs2)
        } else {
            None
        }?;
        let vt = m.value_type_mut(v);
        let nt = m.value_type_mut(new_call);
        if is_distributive_op(m, Self::get_op(m, new_call)) && m.type_is(nt, vt) && !matches(m, v, new_call) {
            Some(new_call)
        } else {
            None
        }
    }
}

/// x - c --> x + (-c)
struct CanonConstSub;

impl RewriteRule for CanonConstSub {
    fn apply(&mut self, m: &mut Module, v: ValueId) -> Option<ValueId> {
        let ty = m.value_type_mut(v);
        if !is_call_of(m, v, magic::SUB, 2, None, true) {
            return None;
        }
        let a = args(m, v);
        let (lhs, rhs) = (a[0], a[1]);
        let lt = m.value_type_mut(lhs);
        let rt = m.value_type_mut(rhs);
        if !m.type_is(lt, rt) {
            return None;
        }
        let new_call = if is_const::<i64>(m, rhs) {
            let c = get_const::<i64>(m, rhs);
            if c == i64::MIN {
                // ensure no overflow
                return None;
            }
            let nc = m.get_int(-c);
            m.binary_op(lhs, magic::ADD, nc)
        } else if is_const::<f64>(m, rhs) {
            let c = get_const::<f64>(m, rhs);
            let nc = m.get_float(-c);
            m.binary_op(lhs, magic::ADD, nc)
        } else {
            None
        }?;
        let nt = m.value_type_mut(new_call);
        if m.type_is(nt, ty) && !matches(m, v, new_call) { Some(new_call) } else { None }
    }
}

/// Canonicalization pass that flattens nested series flows, puts operands in a
/// predefined order, etc.
pub struct CanonicalizationPass {
    state: OperatorState,
    rewriter: Rewriter,
    side_effects_key: String,
}

impl CanonicalizationPass {
    pub const KEY: &'static str = "core-cleanup-canon";

    pub fn new(side_effects_key: impl Into<String>) -> Self {
        Self { state: OperatorState::new(true), rewriter: Rewriter::new(), side_effects_key: side_effects_key.into() }
    }

    /// `getNumReplacements()`.
    pub fn num_replacements(&self) -> i32 {
        self.rewriter.num_replacements()
    }

    fn register_standard_rules(&mut self) {
        self.rewriter.register_rule("op-chain", Box::new(CanonOpChain));
        self.rewriter.register_rule("inequality", Box::new(CanonInequality));
        self.rewriter.register_rule("add-mul", Box::new(CanonAddMul));
        self.rewriter.register_rule("const-sub", Box::new(CanonConstSub));
    }
}

struct Runner<'a, 'c> {
    pass: &'a mut CanonicalizationPass,
    ctx: &'a (dyn PassContext + 'c),
}

impl Operator for Runner<'_, '_> {
    fn state(&self) -> &OperatorState {
        &self.pass.state
    }
    fn state_mut(&mut self) -> &mut OperatorState {
        &mut self.pass.state
    }

    fn handle_call_instr(&mut self, m: &mut Module, v: ValueId) {
        let r = self
            .ctx
            .get_analysis::<SideEffectResult>(&self.pass.side_effects_key)
            .expect("side effect analysis result missing");
        if !r.has_side_effect(m, v) {
            self.pass.rewriter.rewrite(m, v);
        }
    }

    fn handle_series_flow(&mut self, m: &mut Module, v: ValueId) {
        let mut items = m.as_series_flow(v).unwrap().series.clone();
        let mut i = 0;
        while i < items.len() {
            let x = items[i];
            if let Some(s) = m.as_series_flow(x) {
                let inner = s.series.clone();
                let n = inner.len();
                items.splice(i..=i, inner);
                i += n;
            } else if let Some(fi) = m.as_flow_instr(x) {
                let (flow, val) = (fi.flow, fi.val);
                // [flow, value]; don't increment, re-traverse in case a new series is added
                items.splice(i..=i, [flow, val]);
            } else {
                i += 1;
            }
        }
        m.as_series_flow_mut(v).unwrap().series = items;
    }
}

impl Operator for CanonicalizationPass {
    fn state(&self) -> &OperatorState {
        &self.state
    }
    fn state_mut(&mut self) -> &mut OperatorState {
        &mut self.state
    }
}

impl Pass for CanonicalizationPass {
    fn key(&self) -> String {
        Self::KEY.to_string()
    }
    fn run(&mut self, m: &mut Module, ctx: &dyn PassContext) {
        self.register_standard_rules();
        self.rewriter.reset();
        let mut r = Runner { pass: self, ctx };
        run_operator_pass(&mut r, m);
    }
}
