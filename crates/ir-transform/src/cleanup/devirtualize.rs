//!
//! Proves exact runtime types with reaching definitions and bounded interprocedural
//! specialization, then annotates field accesses and resolves AOT virtual calls.
//! Preserves dynamic JIT dispatch and leaves inlining to LLVM.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use rustc_hash::FxHashSet;
use typhon_ir::attribute::{Attribute, KeyValueAttribute, attr_id};
use typhon_ir::pass::{Pass, PassContext};
use typhon_ir::types::TypeKind;
use typhon_ir::util::cloning::CloneVisitor;
use typhon_ir::util::irtools::{get_func, has_attribute, is_gpu_kernel, set_export};
use typhon_ir::util::operator::{Operator, OperatorState};
use typhon_ir::var::FuncKind;
use typhon_ir::{FuncId, Id, Intrinsic, Module, NodeRef, TypeId, ValueId};
use typhon_ir_analyze::dataflow::reaching::RDResult;

// Bound proof search separately from code growth. Exhausting either budget may
// lose an optimization, but must never turn an unknown type into an exact one.
const MAX_TYPE_DEPTH: usize = 256;
const MAX_SUMMARY_DEPTH: usize = 32;
const MAX_ARGUMENT_FIELDS: usize = 64;
const MAX_SPECIALIZATION_SIZE: usize = 256;
const MAX_SPECIALIZATION_GROWTH: usize = 32768;
const MAX_VARIANTS: usize = 8;
const SPECIALIZED_ATTR: &str = ".devirtualized";

/// A query path projects through zero-based record fields. Argument facts prefix
/// that path with an argument index: {1, 0} is the first field of the second arg.
type Path = Vec<i32>;
type Facts = BTreeMap<Path, TypeId>;
type ContextKey = (Id, Vec<(Path, String)>);
type Query = (Id, Path);

/// Inspect the final callee without discarding its FlowInstr setup.
fn direct_callee(m: &Module, mut value: ValueId) -> Option<FuncId> {
    while let Some(f) = m.as_flow_instr(value) {
        value = f.val;
    }
    get_func(m, value)
}

fn is_record(m: &Module, t: TypeId) -> bool {
    matches!(m.type_kind(t), TypeKind::Record { .. })
}
fn as_ref_type(m: &Module, t: TypeId) -> Option<(TypeId, bool)> {
    let t = m.actual_type(t);
    match m.type_kind(t) {
        TypeKind::Ref { polymorphic, .. } => Some((t, *polymorphic)),
        _ => None,
    }
}
fn is_bodied(m: &Module, f: FuncId) -> bool {
    m.func(f).is_some_and(|f| f.is_bodied())
}

/// Context-independent source-body inventory.
#[derive(Default)]
struct BodyInfo {
    state: OperatorState,
    size: usize,
    has_shared_values: bool,
    operands: FxHashSet<Id>,
    returns: Vec<Option<ValueId>>,
    calls: Vec<ValueId>,
    fields: Vec<(ValueId, ValueId)>,
}

impl Operator for BodyInfo {
    fn state(&self) -> &OperatorState {
        &self.state
    }
    fn state_mut(&mut self) -> &mut OperatorState {
        &mut self.state
    }
    fn pre_hook(&mut self, m: &mut Module, n: NodeRef) {
        self.size += 1;
        if self.has_shared_values {
            return;
        }
        for value in m.used_values(n) {
            if !m.is_const(value) && get_func(m, value).is_none() && !self.operands.insert(m.value_uid(value)) {
                self.has_shared_values = true;
                break;
            }
        }
    }
    fn handle_return_instr(&mut self, m: &mut Module, v: ValueId) {
        self.returns.push(m.as_return_instr(v).unwrap().value);
    }
    fn handle_call_instr(&mut self, _m: &mut Module, v: ValueId) {
        self.calls.push(v);
    }
    fn handle_extract_instr(&mut self, m: &mut Module, v: ValueId) {
        let x = m.as_extract_instr(v).unwrap().val;
        self.fields.push((v, x));
    }
    fn handle_insert_instr(&mut self, m: &mut Module, v: ValueId) {
        let x = m.as_insert_instr(v).unwrap().lhs;
        self.fields.push((v, x));
    }
}

/// Analysis of the original body under one set of exact argument facts.
struct Context {
    source: FuncId,
    target: FuncId,
    arguments: Facts,
    clone: Option<CloneVisitor>,
    cache: BTreeMap<Query, Option<TypeId>>,
    active: BTreeSet<Query>,
    returns: BTreeMap<Path, Option<TypeId>>,
    active_returns: BTreeSet<Path>,
    fields: Vec<(ValueId, Option<TypeId>)>,
    calls: Vec<(ValueId, FuncId)>,
    lookups: Vec<(ValueId, FuncId)>,
}

struct Specializer<'a> {
    m: &'a mut Module,
    reaching: &'a RDResult,
    jit: bool,
    growth: usize,
    summary_depth: usize,
    bodies: HashMap<Id, BodyInfo>,
    variants: HashMap<Id, usize>,
    contexts: BTreeMap<ContextKey, usize>,
    store: Vec<Context>,
    pending: Vec<usize>,
}

impl<'a> Specializer<'a> {
    fn body(&mut self, f: FuncId) -> &BodyInfo {
        let uid = self.m.var_uid(f);
        if !self.bodies.contains_key(&uid) {
            let mut info = BodyInfo::default();
            if let Some(b) = self.m.func_body(f) {
                info.reset();
                info.process(self.m, b.into());
            }
            self.bodies.insert(uid, info);
        }
        &self.bodies[&uid]
    }

    /// Reuse a context or create a bounded specialization.
    fn context(&mut self, function: FuncId, arguments: Facts) -> Option<usize> {
        let fd = self.m.func(function)?;
        if !fd.is_bodied()
            || fd.generator
            || fd.is_async
            || !self.reaching.results.contains_key(&self.m.var_uid(function))
            || has_attribute(self.m, function, SPECIALIZED_ATTR)
        {
            return None;
        }
        let fuid = self.m.var_uid(function);
        let key: ContextKey =
            (fuid, arguments.iter().map(|(p, t)| (p.clone(), self.m.type_name(*t).to_string())).collect());
        if let Some(&i) = self.contexts.get(&key) {
            return Some(i);
        }
        if !arguments.is_empty() {
            let (shared, size) = {
                let b = self.body(function);
                (b.has_shared_values, b.size)
            };
            if shared
                || self.variants.get(&fuid).copied().unwrap_or(0) >= MAX_VARIANTS
                || size > MAX_SPECIALIZATION_SIZE
                || self.growth + size > MAX_SPECIALIZATION_GROWTH
                || is_gpu_kernel(self.m, function)
            {
                return self.context(function, Facts::new());
            }
        }
        let idx = self.store.len();
        self.store.push(Context {
            source: function,
            target: function,
            arguments,
            clone: None,
            cache: BTreeMap::new(),
            active: BTreeSet::new(),
            returns: BTreeMap::new(),
            active_returns: BTreeSet::new(),
            fields: Vec::new(),
            calls: Vec::new(),
            lookups: Vec::new(),
        });
        // Publish before processing calls so recursion can find this same context.
        self.contexts.insert(key, idx);
        self.pending.push(idx);
        if !self.store[idx].arguments.is_empty() {
            *self.variants.entry(fuid).or_default() += 1;
            self.growth += self.body(function).size;
            let mut clone = CloneVisitor::new(true);
            let m = &mut *self.m;
            let mut name = format!("{}.devirtualized", m.var_name(function));
            for (path, ty) in &self.store[idx].arguments {
                name.push('.');
                for i in path {
                    name.push_str(&format!("{i}_"));
                }
                let tn = m.type_name(*ty);
                name.push_str(&format!("{}:{}", tn.len(), tn));
            }
            let src = m.src_info(function.into());
            let target = m.n_func(src, name, FuncKind::bodied());
            let (unmangled, parent) = {
                let fd = m.func(function).unwrap();
                (fd.unmangled_name.clone(), fd.parent_type)
            };
            {
                let td = m.func_mut(target).unwrap();
                td.unmangled_name = unmangled;
                td.parent_type = parent;
            }
            let src_args = m.func_args(function);
            let names: Vec<String> = src_args.iter().map(|a| m.var_name(*a).to_string()).collect();
            let fty = m.var_type(function).expect("function has no type");
            m.realize_func(target, fty, &names);
            let tgt_args = m.func_args(target);
            for (a, b) in src_args.iter().zip(&tgt_args) {
                clone.force_remap(m, (*a).into(), (*b).into());
            }
            let symbols: Vec<_> = m.func(function).unwrap().symbols().to_vec();
            for s in symbols {
                let ns = clone.force_clone_var(m, s);
                m.func_push_symbol(target, ns);
            }
            // Remap locals and parameters, not the source function itself.
            if let Some(body) = m.func_body(function) {
                let nb = clone.clone_value(m, body);
                m.set_func_body(target, Some(nb));
            }
            let attrs: Vec<(i32, Attribute)> = m.attributes(function.into()).iter().cloned().collect();
            for (k, a) in attrs {
                if a.needs_clone() {
                    let na = if a.holds_values() { a.map_values(&mut |x| clone.clone_value(m, x)) } else { a };
                    m.set_attribute_key(target.into(), na, k);
                }
            }
            if !m.has_attribute(target.into(), attr_id::KEY_VALUE) {
                m.set_attribute(target.into(), Attribute::KeyValue(KeyValueAttribute::default()));
            }
            // Preserve optimization contracts such as noinline, but do not expose a
            // specialized entry point to callers that cannot guarantee its type facts.
            set_export(m, target, false);
            m.get_attribute_mut(target.into(), attr_id::KEY_VALUE)
                .and_then(|a| a.as_key_value_mut())
                .unwrap()
                .attributes
                .insert(SPECIALIZED_ATTR.to_string(), String::new());
            self.store[idx].target = target;
            self.store[idx].clone = Some(clone);
        }
        Some(idx)
    }

    /// Retain exact polymorphic references, including those nested in value records.
    fn collect(&mut self, caller: usize, value: ValueId, ty: TypeId, path: &mut Path, facts: &mut Facts, remaining: &mut usize) {
        if *remaining == 0 || path.len() >= MAX_SUMMARY_DEPTH {
            return;
        }
        *remaining -= 1;
        let ty = self.m.actual_type(ty);
        if is_record(self.m, ty) {
            let fields = self.m.type_fields(ty);
            for (index, field) in fields.iter().enumerate() {
                path.push(index as i32);
                self.collect(caller, value, field.ty, path, facts, remaining);
                path.pop();
                if *remaining == 0 {
                    break;
                }
            }
        } else if let Some(exact) = self.get(caller, value, &path[1..].to_vec()) {
            if as_ref_type(self.m, exact).is_some_and(|(_, p)| p) {
                facts.insert(path.clone(), exact);
            }
        }
    }

    fn callee_context(&mut self, caller: usize, call: ValueId, callee: FuncId) -> Option<usize> {
        let mut facts = Facts::new();
        let mut remaining = MAX_ARGUMENT_FIELDS;
        let args = self.m.as_call_instr(call).unwrap().args.clone();
        for (index, arg) in args.into_iter().enumerate() {
            let t = self.m.value_type_mut(arg);
            let mut path = vec![index as i32];
            self.collect(caller, arg, t, &mut path, &mut facts, &mut remaining);
        }
        self.context(callee, facts)
    }

    /// A return projection is exact only when every collected return agrees.
    fn returned(&mut self, ctx: usize, fields: &Path) -> Option<TypeId> {
        if let Some(r) = self.store[ctx].returns.get(fields) {
            return *r;
        }
        if self.summary_depth >= MAX_SUMMARY_DEPTH || !self.store[ctx].active_returns.insert(fields.clone()) {
            return None;
        }
        self.summary_depth += 1;
        let source = self.store[ctx].source;
        let returns = self.body(source).returns.clone();
        let mut result = None;
        for value in returns {
            let ty = value.and_then(|v| self.get(ctx, v, fields));
            match ty {
                Some(t) if result.is_none() || result == Some(t) => result = Some(t),
                _ => {
                    result = None;
                    break;
                }
            }
        }
        self.summary_depth -= 1;
        self.store[ctx].active_returns.remove(fields);
        self.store[ctx].returns.insert(fields.clone(), result);
        result
    }

    fn infer(&mut self, ctx: usize, value: ValueId, mut fields: Path) -> Option<TypeId> {
        let m = &*self.m;
        if let Some(vv) = m.as_var_value(value) {
            let var = m.actual_var(vv.var);
            let source = self.store[ctx].source;
            let analysis = self.reaching.results.get(&m.var_uid(source))?.clone();
            if m.is_global(var) || analysis.is_invalid(m, var) {
                return None;
            }
            // Intersect all reaching definitions, including loop-carried assignments.
            let mut result: Option<TypeId> = None;
            let defs = analysis.get_reaching_definitions(m, var, value);
            for def in defs {
                let cfg = analysis.cfg();
                let entry = cfg.as_synthetic_assign(def.assignment).is_some()
                    && cfg.synth_source(def.assignment) == Some(NodeRef::Var(source));
                let ty = if entry {
                    // Only CFG entry assignments represent incoming argument facts.
                    let args = self.m.func_args(source);
                    let pos = args.iter().position(|a| self.m.actual_var(*a) == var).map(|p| p as i32).unwrap_or(args.len() as i32);
                    let mut path = fields.clone();
                    path.insert(0, pos);
                    self.store[ctx].arguments.get(&path).copied()
                } else if def.known() {
                    def.assignee_value().and_then(|a| self.get(ctx, a, &fields))
                } else {
                    None
                };
                match ty {
                    Some(t) if result.is_none() || result == Some(t) => result = Some(t),
                    _ => return None,
                }
            }
            return result;
        }
        if let Some(f) = m.as_flow_instr(value) {
            let v = f.val;
            return self.get(ctx, v, &fields);
        }
        if let Some(ex) = m.as_extract_instr(value) {
            let (val, field) = (ex.val, ex.field.clone());
            let t = self.m.value_type_mut(val);
            if is_record(self.m, t) {
                // Move the projection back onto the containing record, preserving order.
                fields.insert(0, self.m.member_index(t, &field));
                return self.get(ctx, val, &fields);
            }
        }
        let m = &*self.m;
        if let Some(t) = m.as_ternary_instr(value) {
            let (tv, fv) = (t.true_value, t.false_value);
            let first = self.get(ctx, tv, &fields)?;
            return if Some(first) == self.get(ctx, fv, &fields) { Some(first) } else { None };
        }
        if let Some(c) = m.as_call_instr(value) {
            let (callee_v, args) = (c.callee, c.args.clone());
            let callee = direct_callee(m, callee_v)?;
            let fd = m.func(callee)?;
            if fd.is_internal() {
                let (intr, ity) = (fd.intrinsic(), fd.intrinsic_type());
                if !self.m.has_valid_intrinsic_signature(callee) {
                    return None;
                }
                // Allocation establishes exactness; casts preserve it; record construction
                // routes a projection to its corresponding operand.
                match intr {
                    Intrinsic::Allocate => {
                        return if fields.is_empty() { ity.and_then(|t| as_ref_type(self.m, t)).map(|x| x.0) } else { None };
                    }
                    Intrinsic::RepresentationCast => return self.get(ctx, *args.first()?, &fields),
                    Intrinsic::RecordNew if !fields.is_empty() => {
                        let index = fields[0];
                        if index < 0 || index as usize >= args.len() {
                            return None;
                        }
                        fields.remove(0);
                        return self.get(ctx, args[index as usize], &fields);
                    }
                    _ => {}
                }
            } else if fd.is_bodied() {
                if let Some(target) = self.callee_context(ctx, value, callee) {
                    return self.returned(target, &fields);
                }
            }
        }
        None
    }

    /// Memoize proofs within a context. `None` means unknown.
    fn get(&mut self, ctx: usize, value: ValueId, fields: &Path) -> Option<TypeId> {
        let source = self.store[ctx].source;
        if self.body(source).has_shared_values {
            return None;
        }
        let query: Query = (self.m.value_uid(value), fields.clone());
        if let Some(r) = self.store[ctx].cache.get(&query) {
            return *r;
        }
        if self.store[ctx].active.len() >= MAX_TYPE_DEPTH
            || fields.len() >= MAX_TYPE_DEPTH
            || !self.store[ctx].active.insert(query.clone())
        {
            return None;
        }
        let result = self.infer(ctx, value, fields.clone());
        self.store[ctx].active.remove(&query);
        self.store[ctx].cache.insert(query, result);
        result
    }

    fn virtual_target(&mut self, ctx: usize, mut value: ValueId) -> Option<FuncId> {
        while let Some(f) = self.m.as_flow_instr(value) {
            value = f.val;
        }
        let c = self.m.as_call_instr(value)?;
        let (callee, args) = (c.callee, c.args.clone());
        let function = direct_callee(self.m, callee)?;
        // Exact object layout is stable in the JIT, but method tables can be refreshed.
        // Only AOT compilation may bind a virtual slot to its target.
        let fd = self.m.func(function)?;
        if self.jit || !fd.is_internal() || fd.intrinsic() != Intrinsic::VirtualLookup || args.len() != 2 {
            return None;
        }
        let ty = self.get(ctx, args[0], &Vec::new())?;
        let method = self.m.as_int_const(args[1])?.val;
        let info = self.m.runtime_info(ty)?;
        let target = *info.methods.get(&(method as usize))?;
        let ct = self.m.value_type_mut(value);
        let tt = self.m.var_type(target)?;
        if self.m.actual_type(tt) == self.m.actual_type(ct) { Some(target) } else { None }
    }

    fn mapped(&mut self, ctx: usize, v: ValueId) -> ValueId {
        // Resolve clone identities now: replaceAll changes a source node's id.
        match self.store[ctx].clone.as_mut() {
            Some(c) => c.clone_value(self.m, v),
            None => v,
        }
    }

    fn plan(&mut self, ctx: usize) {
        let source = self.store[ctx].source;
        let (fields, calls) = {
            let b = self.body(source);
            (b.fields.clone(), b.calls.clone())
        };
        for (node, receiver) in fields {
            let mut exact = None;
            let rt = self.m.value_type_mut(receiver);
            if let Some((declared, true)) = as_ref_type(self.m, rt) {
                if let Some(ty) = self.get(ctx, receiver, &Vec::new()) {
                    // The proof must be compatible with the declared receiver.
                    if let Some(info) = self.m.runtime_info(ty) {
                        if info.mro.iter().any(|t| self.m.actual_type(*t) == declared) {
                            exact = Some(ty);
                        }
                    }
                }
            }
            let dest = self.mapped(ctx, node);
            self.store[ctx].fields.push((dest, exact));
        }
        for call in calls {
            let destination = self.mapped(ctx, call);
            if let Some(target) = self.virtual_target(ctx, call) {
                self.store[ctx].lookups.push((destination, target));
            }
            let callee_v = self.m.as_call_instr(call).unwrap().callee;
            // A proven virtual lookup can also expose a body worth specializing.
            let callee = match direct_callee(self.m, callee_v) {
                Some(c) => Some(c),
                None => self.virtual_target(ctx, callee_v),
            };
            if let Some(function) = callee.filter(|f| is_bodied(self.m, *f)) {
                if let Some(target) = self.callee_context(ctx, call, function) {
                    let t = self.store[target].target;
                    if t != function {
                        self.store[ctx].calls.push((destination, t));
                    }
                }
            }
        }
    }

    fn apply(&mut self, ctx: usize) {
        let m = &mut *self.m;
        let c = &self.store[ctx];
        for (value, ty) in &c.fields {
            m.erase_attribute((*value).into(), attr_id::EXACT_TYPE);
            if let Some(t) = ty {
                m.set_attribute((*value).into(), Attribute::ExactType(*t));
            }
        }
        let src_of = |m: &mut Module, v: ValueId, from: ValueId| {
            let s = m.src_info(from.into());
            m.set_src_info(v.into(), s);
            v
        };
        // Preserve callee setup before the existing arguments.
        for (call, target) in &c.calls {
            let callee = m.as_call_instr(*call).unwrap().callee;
            let prefix = m.nr_series_flow();
            src_of(m, prefix, *call);
            m.series_push(prefix, callee);
            let vv = m.nr_var_value(*target);
            src_of(m, vv, *call);
            let fi = m.nr_flow_instr(prefix, vv);
            src_of(m, fi, *call);
            m.as_call_instr_mut(*call).unwrap().callee = fi;
        }
        for (call, target) in &c.lookups {
            let ci = m.as_call_instr(*call).unwrap();
            let (callee, args) = (ci.callee, ci.args.clone());
            let prefix = m.nr_series_flow();
            src_of(m, prefix, *call);
            m.series_push(prefix, callee);
            for a in args {
                m.series_push(prefix, a);
            }
            let vv = m.nr_var_value(*target);
            src_of(m, vv, *call);
            let fi = m.nr_flow_instr(prefix, vv);
            src_of(m, fi, *call);
            m.replace_all_value(*call, fi);
        }
    }

    fn run(&mut self) {
        let mut functions = Vec::new();
        let main = self.m.main_func();
        if is_bodied(self.m, main) {
            functions.push(main);
        }
        for v in self.m.vars() {
            if is_bodied(self.m, v) {
                functions.push(v);
            }
        }
        for f in functions {
            self.context(f, Facts::new());
        }
        // Planning can append newly discovered contexts, so consume a growing worklist.
        let mut i = 0;
        while i < self.pending.len() {
            let c = self.pending[i];
            self.plan(c);
            i += 1;
        }
        for c in self.pending.clone() {
            self.apply(c);
        }
    }
}

/// Devirtualization pass (`DevirtualizationPass`).
pub struct DevirtualizationPass {
    reaching_key: String,
}

impl DevirtualizationPass {
    pub const KEY: &'static str = "core-devirtualize";

    /// @param reaching_key the module's reaching-definition analysis key
    pub fn new(reaching_key: impl Into<String>) -> Self {
        Self { reaching_key: reaching_key.into() }
    }
}

impl Pass for DevirtualizationPass {
    fn key(&self) -> String {
        Self::KEY.to_string()
    }
    fn run(&mut self, m: &mut Module, ctx: &dyn PassContext) {
        // Note: the reference implementation dereferences a missing result; here the pass is a no-op.
        let Some(result) = ctx.get_analysis::<RDResult>(&self.reaching_key) else { return };
        let jit = ctx.options().map(|o| o.jit).unwrap_or(false);
        let mut s = Specializer {
            m,
            reaching: result,
            jit,
            growth: 0,
            summary_depth: 0,
            bodies: HashMap::new(),
            variants: HashMap::new(),
            contexts: BTreeMap::new(),
            store: Vec::new(),
            pending: Vec::new(),
        };
        s.run();
    }
}
