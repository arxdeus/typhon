//! Two complementary rewrites avoid the generator protocol: turn direct
//! sum/any/all calls into ordinary functions, and splice a producer's body into
//! its sole for-loop consumer.

use rustc_hash::{FxHashMap, FxHashSet};
use typhon_common::mangle;
use typhon_ir::util::cloning::CloneVisitor;
use typhon_ir::util::irtools;
use typhon_ir::util::operator::{Operator, OperatorState, run_operator};
use typhon_ir::{FuncId, Id, Module, NodeRef, TypeId, ValueId, ValueKind, VarId, magic};
use typhon_ir_analyze::util::inlining::inline_call;

use crate::lowering::{operator_pass, operator_state};

fn name_starts(m: &Module, f: Option<FuncId>, n: &str) -> bool {
    f.is_some_and(|f| m.var_name(f).starts_with(&mangle::func("std.internal.builtin", n)))
}

fn bodied(m: &Module, f: Option<FuncId>) -> Option<FuncId> {
    f.filter(|f| m.func(*f).is_some_and(|d| d.is_bodied()))
}

/// Replaces yields with updates to the accumulator variable.
struct GeneratorSumTransformer {
    state: OperatorState,
    accumulator: VarId,
    valid: bool,
}

impl Operator for GeneratorSumTransformer {
    operator_state!();
    fn handle_yield_instr(&mut self, m: &mut Module, v: ValueId) {
        let Some(val) = m.as_yield_instr(v).unwrap().value else {
            self.valid = false;
            return;
        };
        let mut rhs = val;
        let vt = m.value_type_mut(val);
        let bt = m.get_bool_type();
        if m.type_is(vt, bt) {
            let one = m.get_int(1);
            let zero = m.get_int(0);
            rhs = m.nr_ternary(rhs, one, zero);
        }
        let av = m.nr_var_value(self.accumulator);
        let Some(add) = m.binary_op(av, magic::ADD, rhs) else {
            self.valid = false;
            return;
        };
        let at = m.value_type_mut(add);
        if !m.var_type(self.accumulator).is_some_and(|t| m.type_is(at, t)) {
            self.valid = false;
            return;
        }
        let assign = m.nr_assign(self.accumulator, add);
        m.replace_all_value(v, assign);
    }
    fn handle_return_instr(&mut self, m: &mut Module, v: ValueId) {
        let av = m.nr_var_value(self.accumulator);
        let nr = m.nr_return(Some(av));
        self.see(m, nr);
        let rep = match m.as_return_instr(v).unwrap().value {
            Some(val) => irtools::series(m, &[val, nr]).unwrap(),
            None => nr,
        };
        m.replace_all_value(v, rep);
    }
    fn handle_yield_in_instr(&mut self, _m: &mut Module, _v: ValueId) {
        self.valid = false;
    }
}

/// Replaces yields with conditional returns of the any/all answer.
struct GeneratorAnyAllTransformer {
    state: OperatorState,
    any: bool,
    valid: bool,
}

impl Operator for GeneratorAnyAllTransformer {
    operator_state!();
    fn handle_yield_instr(&mut self, m: &mut Module, v: ValueId) {
        let val = m.as_yield_instr(v).unwrap().value;
        let bt = m.get_bool_type();
        let Some(mut vb) = val.and_then(|x| m.construct(bt, vec![x])) else {
            self.valid = false;
            return;
        };
        if !self.any {
            let f = m.get_bool(false);
            let t = m.get_bool(true);
            vb = m.nr_ternary(vb, f, t);
        }
        let b = m.get_bool(self.any);
        let nr = m.nr_return(Some(b));
        self.see(m, nr);
        let s = irtools::series(m, &[nr]).unwrap();
        let rep = m.nr_if_flow(vb, s, None);
        m.replace_all_value(v, rep);
    }
    fn handle_return_instr(&mut self, m: &mut Module, v: ValueId) {
        // Leave the short-circuit returns inserted at yield sites unchanged.
        if self.saw(m, v) {
            return;
        }
        let b = m.get_bool(!self.any);
        let nr = m.nr_return(Some(b));
        self.see(m, nr);
        let rep = match m.as_return_instr(v).unwrap().value {
            Some(val) => irtools::series(m, &[val, nr]).unwrap(),
            None => nr,
        };
        m.replace_all_value(v, rep);
    }
    fn handle_yield_in_instr(&mut self, _m: &mut Module, _v: ValueId) {
        self.valid = false;
    }
}

/// Creates the wrapper function `name: fn_type` (generator args + `extra_names`) and
/// clones the generator body into it, remapping the arguments.
fn clone_gen_into(
    m: &mut Module,
    gen_: FuncId,
    name: String,
    fn_type: TypeId,
    extra_names: &[&str],
    site: ValueId,
) -> Option<(FuncId, ValueId)> {
    let mut names: Vec<String> = m.func_args(gen_).iter().map(|a| m.var_name(*a).to_string()).collect();
    names.extend(extra_names.iter().map(|s| s.to_string()));
    let f = m.nr_bodied_func(name);
    let src = m.src_info(site.into());
    m.set_src_info(f.into(), src);
    m.realize_func(f, fn_type, &names);
    let mut remap = FxHashMap::default();
    for (a, b) in m.func_args(gen_).into_iter().zip(m.func_args(f)) {
        remap.insert(m.var_uid(a), b);
    }
    let Some(gb) = m.func_body(gen_) else {
        m.remove_var(f);
        return None;
    };
    let mut cv = CloneVisitor::new(true);
    let body = cv.clone_value_into(m, gb, Some(f), &remap);
    m.set_func_body(f, Some(body));
    if m.as_series_flow(body).is_none() {
        m.remove_var(f);
        return None;
    }
    Some((f, body))
}

fn gen_func_arg_types(m: &Module, gen_: FuncId) -> Option<Vec<TypeId>> {
    let gt = m.var_type(gen_)?;
    m.type_kind(gt).is_func().then(|| m.func_type_args(gt))
}

// def gen(items): for item in items: if keep(item): yield compute(item)
// sum(gen(items), start)  ==>  sum_gen(items, start) with an accumulator.
fn gen_to_sum(m: &mut Module, gen_: Option<FuncId>, start_type: TypeId, out_type: TypeId, parent: FuncId, site: ValueId) -> Option<FuncId> {
    let gen_ = gen_.filter(|g| m.func(*g).is_some_and(|d| d.generator))?;
    let mut arg_types = gen_func_arg_types(m, gen_)?;
    arg_types.push(start_type);
    let fn_type = m.get_func_type(out_type, arg_types, false);
    let full = format!("__sum_wrapper.{}.{}", m.var_name(gen_), m.type_name(fn_type));
    let name = irtools::derived_func_name(m, &full, parent, Some(site.into()));
    let (f, body) = clone_gen_into(m, gen_, name, fn_type, &["start"], site)?;

    let last = *m.func_args(f).last()?;
    let mut init = Some(m.nr_var_value(last));
    let it = m.get_int_type();
    let ft = m.get_float_type();
    if m.type_is(start_type, it) && m.type_is(out_type, ft) {
        init = m.construct(ft, vec![init.unwrap()]);
    }
    let ok = init.is_some_and(|i| {
        let t = m.value_type_mut(i);
        m.type_is(t, out_type)
    });
    if !ok {
        m.remove_var(f);
        return None;
    }
    let acc = irtools::make_var(m, init.unwrap(), body, Some(f), true);
    let mut x = GeneratorSumTransformer { state: OperatorState::new(false), accumulator: acc, valid: true };
    x.process(m, f.into());
    let av = m.nr_var_value(acc);
    let r = m.nr_return(Some(av));
    m.series_push(body, r);
    if !x.valid {
        m.remove_var(f);
        return None;
    }
    Some(f)
}

// any(gen(items)) / all(gen(items)) ==> short-circuiting bool functions.
fn gen_to_any_all(m: &mut Module, gen_: Option<FuncId>, any: bool, parent: FuncId, site: ValueId) -> Option<FuncId> {
    let gen_ = gen_.filter(|g| m.func(*g).is_some_and(|d| d.generator))?;
    let base = format!("{}{}", if any { "__any_wrapper." } else { "__all_wrapper." }, m.var_name(gen_));
    let name = irtools::derived_func_name(m, &base, parent, Some(site.into()));
    let arg_types = gen_func_arg_types(m, gen_)?;
    let bt = m.get_bool_type();
    let fn_type = m.get_func_type(bt, arg_types, false);
    let (f, body) = clone_gen_into(m, gen_, name, fn_type, &[], site)?;
    let mut x = GeneratorAnyAllTransformer { state: OperatorState::new(false), any, valid: true };
    x.process(m, f.into());
    let b = m.get_bool(!any);
    let r = m.nr_return(Some(b));
    m.series_push(body, r);
    if !x.valid {
        m.remove_var(f);
        return None;
    }
    Some(f)
}

/// Pass to optimize passing a generator to some built-in functions
/// like sum(), any() or all(), which will be converted to regular
/// for-loops.
#[derive(Default)]
pub struct GeneratorArgumentOptimization {
    state: OperatorState,
}

impl GeneratorArgumentOptimization {
    pub const KEY: &'static str = "core-pythonic-generator-argument-opt";
    pub fn new() -> Self {
        Self::default()
    }
}

impl Operator for GeneratorArgumentOptimization {
    operator_state!();
    // Specialize only reducers applied directly to a generator call.
    fn handle_call_instr(&mut self, m: &mut Module, v: ValueId) {
        let c = m.as_call_instr(v).unwrap().clone();
        let func = irtools::get_func(m, c.callee);
        let Some(parent) = self.parent_func(m) else { return };
        if name_starts(m, func, "sum") && c.args.len() == 2 {
            let Some(call) = m.as_call_instr(c.args[0]).cloned() else { return };
            let gen_ = bodied(m, irtools::get_func(m, call.callee));
            let start = c.args[1];
            let st = m.value_type_mut(start);
            let vt = m.value_type_mut(v);
            if let Some(f) = gen_to_sum(m, gen_, st, vt, parent, v) {
                let mut args = call.args.clone();
                args.push(start);
                let r = irtools::call(m, f, args);
                m.replace_all_value(v, r);
            }
        } else {
            let any = name_starts(m, func, "any");
            let all = name_starts(m, func, "all");
            let vt = m.value_type_mut(v);
            let bt = m.get_bool_type();
            if !(any || all) || c.args.len() != 1 || !m.type_is(vt, bt) {
                return;
            }
            let Some(call) = m.as_call_instr(c.args[0]).cloned() else { return };
            let gen_ = bodied(m, irtools::get_func(m, call.callee));
            if let Some(f) = gen_to_any_all(m, gen_, any, parent, v) {
                let r = irtools::call(m, f, call.args.clone());
                m.replace_all_value(v, r);
            }
        }
    }
}

operator_pass!(GeneratorArgumentOptimization);

// ---------------------------------------------------------------------------
// Loop fusion
// ---------------------------------------------------------------------------

/// Restrict fusion to small synchronous bodies.
struct FusionVerifier {
    state: OperatorState,
    nodes: i32,
    yields: i32,
    valid: bool,
}

impl FusionVerifier {
    fn new() -> Self {
        Self { state: OperatorState::new(false), nodes: 0, yields: 0, valid: true }
    }
}

impl Operator for FusionVerifier {
    operator_state!();
    fn pre_hook(&mut self, _m: &mut Module, _n: NodeRef) {
        self.nodes += 1;
        self.valid &= self.nodes <= 256;
    }
    fn handle_yield_instr(&mut self, m: &mut Module, v: ValueId) {
        let y = m.as_yield_instr(v).unwrap();
        self.valid &= y.value.is_some() && !y.final_;
        self.yields += 1;
    }
    fn handle_yield_in_instr(&mut self, _m: &mut Module, _v: ValueId) {
        self.valid = false;
    }
    fn handle_await_instr(&mut self, _m: &mut Module, _v: ValueId) {
        self.valid = false;
    }
    fn handle_try_catch_flow(&mut self, _m: &mut Module, _v: ValueId) {
        self.valid = false;
    }
    fn handle_pointer_value(&mut self, m: &mut Module, v: ValueId) {
        let var = m.as_pointer_value(v).unwrap().var;
        self.valid &= m.var(var).global;
    }
    fn handle_stack_alloc_instr(&mut self, _m: &mut Module, _v: ValueId) {
        self.valid = false;
    }
    fn handle_for_flow(&mut self, m: &mut Module, v: ValueId) {
        let f = m.as_for_flow(v).unwrap();
        self.valid &= !f.is_parallel() && !f.is_async;
    }
}

/// Reject user break/continue in a consumer body, but allow breaks targeting
/// compiler-created exit wrappers.
struct ConsumerVerifier<'a> {
    state: OperatorState,
    wrappers: &'a FxHashSet<Id>,
    valid: bool,
}

impl Operator for ConsumerVerifier<'_> {
    operator_state!();
    fn handle_break_instr(&mut self, m: &mut Module, v: ValueId) {
        let l = m.as_break_instr(v).unwrap().loop_;
        self.valid &= l.is_some_and(|l| self.wrappers.contains(&m.value_uid(l)));
    }
    fn handle_continue_instr(&mut self, _m: &mut Module, _v: ValueId) {
        self.valid = false;
    }
}

/// A named iterator is eligible only when one assignment feeds one later read,
/// with no exposed handle and matching enclosing control-flow contexts.
struct IteratorUseVerifier<'a> {
    state: OperatorState,
    iterator: Id,
    consumer: Id,
    wrappers: &'a FxHashSet<Id>,
    assignment: Option<ValueId>,
    reads: i32,
    writes: i32,
    position: i32,
    creation_position: i32,
    consumption_position: i32,
    address_taken: bool,
    creation_loops: Vec<Id>,
    consumption_loops: Vec<Id>,
}

impl<'a> IteratorUseVerifier<'a> {
    fn new(m: &Module, iterator: VarId, consumer: ValueId, wrappers: &'a FxHashSet<Id>) -> Self {
        Self {
            state: OperatorState::new(false),
            iterator: m.var_uid(iterator),
            consumer: m.value_uid(consumer),
            wrappers,
            assignment: None,
            reads: 0,
            writes: 0,
            position: 0,
            creation_position: 0,
            consumption_position: 0,
            address_taken: false,
            creation_loops: vec![],
            consumption_loops: vec![],
        }
    }

    fn enclosing_loops(&self, m: &Module) -> Vec<Id> {
        let parents = self.parents();
        let mut result = Vec::new();
        for (i, n) in parents.iter().enumerate() {
            let Some(v) = n.as_value() else { continue };
            let k = m.kind(v);
            let uid = m.value_uid(v);
            if !k.is_flow() || uid == self.consumer || self.wrappers.contains(&uid) {
                continue;
            }
            let branchy = matches!(k, ValueKind::IfFlow(_) | ValueKind::TryCatchFlow(_));
            if matches!(k, ValueKind::ForFlow(_) | ValueKind::WhileFlow(_) | ValueKind::ImperativeForFlow(_)) || branchy {
                result.push(uid);
                if branchy {
                    if let Some(b) = parents.get(i + 1).and_then(|n| n.as_value()) {
                        result.push(m.value_uid(b));
                    }
                }
            }
        }
        result
    }

    fn valid(&self) -> bool {
        self.reads == 1
            && self.writes == 1
            && !self.address_taken
            && self.creation_position < self.consumption_position
            && self.creation_loops == self.consumption_loops
    }
}

impl Operator for IteratorUseVerifier<'_> {
    operator_state!();
    fn pre_hook(&mut self, m: &mut Module, n: NodeRef) {
        self.position += 1;
        let Some(v) = n.as_value() else { return };
        if matches!(m.kind(v), ValueKind::VarValue(_) | ValueKind::PointerValue(_) | ValueKind::AssignInstr(_)) {
            return;
        }
        for var in m.used_variables(v.into()) {
            self.address_taken |= m.var_uid(var) == self.iterator;
        }
    }
    fn handle_var_value(&mut self, m: &mut Module, v: ValueId) {
        let var = m.as_var_value(v).unwrap().var;
        if m.var_uid(var) == self.iterator {
            self.reads += 1;
            self.consumption_position = self.position;
            self.consumption_loops = self.enclosing_loops(m);
        }
    }
    fn handle_pointer_value(&mut self, m: &mut Module, v: ValueId) {
        let var = m.as_pointer_value(v).unwrap().var;
        self.address_taken |= m.var_uid(var) == self.iterator;
    }
    fn handle_assign_instr(&mut self, m: &mut Module, v: ValueId) {
        let lhs = m.as_assign_instr(v).unwrap().lhs;
        if m.var_uid(lhs) == self.iterator {
            self.assignment = Some(v);
            self.writes += 1;
            self.creation_position = self.position;
            self.creation_loops = self.enclosing_loops(m);
        }
    }
}

/// Traverse the producer bottom-up: the consumer body inserted at a yield must
/// not itself be rewritten.
struct FusionTransformer {
    state: OperatorState,
    consumer_var: VarId,
    consumer_body: ValueId,
    exit: ValueId,
}

impl Operator for FusionTransformer {
    operator_state!();
    fn handle_yield_instr(&mut self, m: &mut Module, v: ValueId) {
        let val = m.as_yield_instr(v).unwrap().value.expect("verified yield value");
        let a = m.nr_assign(self.consumer_var, val);
        let s = irtools::series(m, &[a, self.consumer_body]).unwrap();
        m.replace_all_value(v, s);
    }
    fn handle_return_instr(&mut self, m: &mut Module, v: ValueId) {
        let rep = m.nr_series_flow();
        // Exhaustion discards the producer's return value, but not its side effects.
        if let Some(val) = m.as_return_instr(v).unwrap().value {
            m.series_push(rep, val);
        }
        let b = m.nr_break(Some(self.exit));
        m.series_push(rep, b);
        m.replace_all_value(v, rep);
    }
}

/// Fuses generator producers into their sole for-loop consumer and inlines
/// compiler-generated `__sum_wrapper.*` reducers to expose further fusion.
#[derive(Default)]
pub struct GeneratorLoopFusion {
    state: OperatorState,
    wrappers: FxHashSet<Id>,
    growth: FxHashMap<Id, i32>,
}

impl GeneratorLoopFusion {
    pub const KEY: &'static str = "core-pythonic-generator-loop-fusion";
    pub fn new() -> Self {
        Self::default()
    }

    fn parent_bodied(&self, m: &Module) -> Option<FuncId> {
        self.parent_func(m).filter(|p| m.func(*p).is_some_and(|d| d.is_bodied()))
    }
}

impl Operator for GeneratorLoopFusion {
    operator_state!();

    fn handle_call_instr(&mut self, m: &mut Module, call: ValueId) {
        let Some(parent) = self.parent_bodied(m) else { return };
        let callee = m.as_call_instr(call).unwrap().callee;
        let Some(function) = bodied(m, irtools::get_func(m, callee)) else { return };
        if !m.var_name(function).starts_with("__sum_wrapper.") {
            return;
        }
        let mut verifier = FusionVerifier::new();
        if let Some(b) = m.func_body(function) {
            verifier.process(m, b.into());
        }
        let pid = m.var_uid(parent);
        if !verifier.valid || self.growth.get(&pid).copied().unwrap_or(0) + verifier.nodes > 1024 {
            return;
        }
        let inlined = inline_call(m, call, true);
        let Some(result) = inlined.result else { return };
        for var in inlined.new_vars {
            m.func_push_symbol(parent, var);
        }
        // The inliner implements early returns with a one-shot loop; remember it.
        if let Some(flow) = m.as_flow_instr(result).map(|f| f.flow) {
            if let Some(&last) = m.as_series_flow(flow).and_then(|s| s.series.last()) {
                if m.as_while_flow(last).is_some() {
                    self.wrappers.insert(m.value_uid(last));
                }
            }
        }
        *self.growth.entry(pid).or_default() += verifier.nodes;
        m.replace_all_value(call, result);
    }

    fn handle_for_flow(&mut self, m: &mut Module, lp: ValueId) {
        let ff = m.as_for_flow(lp).unwrap().clone();
        if ff.is_parallel() || ff.is_async {
            return;
        }
        let Some(parent) = self.parent_bodied(m) else { return };
        let Some(parent_body) = m.func_body(parent) else { return };

        // Unpack a single-use tuple that forwards an iterator, then retry.
        if let Some(ext) = m.as_extract_instr(ff.iter).cloned() {
            let Some(variable) = irtools::get_var(m, ext.val) else { return };
            if m.var(variable).global {
                return;
            }
            let wrappers = std::mem::take(&mut self.wrappers);
            let mut uses = IteratorUseVerifier::new(m, variable, lp, &wrappers);
            uses.process(m, parent_body.into());
            let (valid, assignment) = (uses.valid(), uses.assignment);
            drop(uses);
            self.wrappers = wrappers;
            let tuple = if valid {
                assignment.and_then(|a| {
                    let rhs = m.as_assign_instr(a).unwrap().rhs;
                    m.as_call_instr(rhs).map(|c| (rhs, c.clone()))
                })
            } else {
                None
            };
            let Some((_, tuple)) = tuple else { return };
            let Some(ctor) = irtools::get_func(m, tuple.callee) else { return };
            let fd = m.func(ctor).unwrap();
            let Some(ty) = fd.parent_type else { return };
            if !m.type_kind(ty).is_record() || m.type_name(ty) != "Tuple" || fd.unmangled_name != magic::NEW {
                return;
            }
            let vt = m.value_type_mut(ext.val);
            let index = m.member_index(vt, &ext.field);
            if index < 0 || index as usize >= tuple.args.len() {
                return;
            }
            let setup = m.nr_series_flow();
            let mut selected = None;
            for (pos, value) in tuple.args.iter().enumerate() {
                let var = irtools::make_var(m, *value, setup, Some(parent), false);
                if pos == index as usize {
                    selected = Some(var);
                }
            }
            m.replace_all_value(assignment.unwrap(), setup);
            let sv = m.nr_var_value(selected.unwrap());
            m.as_for_flow_mut(lp).unwrap().iter = sv;
            self.handle_for_flow(m, lp);
            return;
        }

        // Accept either an immediate generator call or a uniquely owned local handle.
        let mut call = m.as_call_instr(ff.iter).map(|_| ff.iter);
        let mut creation = None;
        if let Some(iterator) = irtools::get_var(m, ff.iter) {
            if m.var(iterator).global {
                return;
            }
            let wrappers = std::mem::take(&mut self.wrappers);
            let mut uses = IteratorUseVerifier::new(m, iterator, lp, &wrappers);
            uses.process(m, parent_body.into());
            let (valid, assignment) = (uses.valid(), uses.assignment);
            drop(uses);
            self.wrappers = wrappers;
            if valid {
                creation = assignment;
                let rhs = m.as_assign_instr(assignment.unwrap()).unwrap().rhs;
                call = m.as_call_instr(rhs).map(|_| rhs);
            }
        }
        let Some(call) = call else { return };
        let c = m.as_call_instr(call).unwrap().clone();
        let Some(generator) = bodied(m, irtools::get_func(m, c.callee)) else { return };
        let gd = m.func(generator).unwrap();
        if !gd.generator
            || gd.body().is_none()
            || gd.is_async
            || m.var_uid(parent) == m.var_uid(generator)
            || c.args.len() != gd.args.len()
            || irtools::is_no_inline(m, generator)
        {
            return;
        }
        let gen_body = m.func_body(generator).unwrap();
        let mut verifier = FusionVerifier::new();
        verifier.process(m, gen_body.into());
        let mut cverifier = ConsumerVerifier { state: OperatorState::new(false), wrappers: &self.wrappers, valid: true };
        cverifier.process(m, ff.body.into());
        let consumer_valid = cverifier.valid;
        let pid = m.var_uid(parent);
        if !verifier.valid
            || verifier.yields != 1
            || !consumer_valid
            || self.growth.get(&pid).copied().unwrap_or(0) + verifier.nodes > 1024
        {
            return;
        }
        *self.growth.entry(pid).or_default() += verifier.nodes;

        let mut setup = m.nr_series_flow();
        let mut arguments = FxHashMap::default();
        // Snapshot arguments once, left-to-right.
        for (value, arg) in c.args.iter().zip(m.func_args(generator)) {
            let v = irtools::make_var(m, *value, setup, Some(parent), false);
            arguments.insert(m.var_uid(arg), v);
        }
        if let Some(cr) = creation {
            m.replace_all_value(cr, setup);
            setup = m.nr_series_flow();
        }
        let mut clone = CloneVisitor::new(true);
        let body = clone.clone_value_into(m, gen_body, Some(parent), &arguments);
        // A one-shot loop gives producer returns a common exit.
        let wrapper = m.nr_series_flow();
        let t = m.get_bool(true);
        let exit = m.nr_while_flow(t, wrapper);
        self.wrappers.insert(m.value_uid(exit));
        let mut transformer = FusionTransformer {
            state: OperatorState::new(true),
            consumer_var: ff.var,
            consumer_body: ff.body,
            exit,
        };
        transformer.process(m, body.into());
        m.series_push(wrapper, body);
        let b = m.nr_break(Some(exit));
        m.series_push(wrapper, b);
        m.series_push(setup, exit);
        m.replace_all_value(lp, setup);
    }
}

impl typhon_ir::pass::Pass for GeneratorLoopFusion {
    fn key(&self) -> String {
        Self::KEY.to_string()
    }
    fn run(&mut self, m: &mut Module, _ctx: &dyn typhon_ir::pass::PassContext) {
        self.wrappers.clear();
        self.growth.clear();
        run_operator(self, m);
    }
}

