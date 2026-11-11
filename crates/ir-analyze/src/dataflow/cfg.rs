//! Port of codon/cir/analyze/dataflow/cfg.{h,cpp}: control-flow graph construction.
//!
//! C++ blocks hold `const Value*` pointers that are either real IR values or synthetic
//! instructions owned by the graph (`SyntheticAssignInstr`, `SyntheticPhiInstr`, plus the
//! temporary `CallInstr`/`VarValue`/`ForFlow`/`SeriesFlow` nodes created when converting
//! pipelines to loops). Here they are [`CfValue::Real`] / [`CfValue::Synth`]; synthetic
//! values live in the graph's own arena and get fresh `IdMixin` ids (`next_id`).

use std::any::TypeId as StdTypeId;
use std::cell::RefCell;
use std::fmt::Write as _;
use std::rc::Rc;

use indexmap::IndexSet;
use rustc_hash::{FxHashMap, FxHashSet};
use typhon_ir::dsl::CustomValue;
use typhon_ir::ids::next_id;
use typhon_ir::pass::{Analysis, AnalysisResult, PassContext};
use typhon_ir::{FuncId, Id, Module, NodeRef, TypeId, TypeKind, ValueId, ValueKind, VarId};

/// Index of a block in its [`CFGraph`].
pub type BlockId = usize;
/// Index of a synthetic value in its [`CFGraph`].
pub type SynthId = usize;

/// A value inside a CFG: a real IR value or a synthetic one owned by the graph.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CfValue {
    Real(ValueId),
    Synth(SynthId),
}

impl From<ValueId> for CfValue {
    fn from(v: ValueId) -> Self {
        CfValue::Real(v)
    }
}

impl CfValue {
    pub fn as_real(self) -> Option<ValueId> {
        match self {
            CfValue::Real(v) => Some(v),
            CfValue::Synth(_) => None,
        }
    }
}

/// A variable referenced by synthetic instructions: a real IR var or a synthetic one
/// (pipeline-to-loop conversion creates new loop variables).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CfVar {
    Real(VarId),
    Synth(usize),
}

impl From<VarId> for CfVar {
    fn from(v: VarId) -> Self {
        CfVar::Real(v)
    }
}

impl CfVar {
    pub fn as_real(self) -> Option<VarId> {
        match self {
            CfVar::Real(v) => Some(v),
            CfVar::Synth(_) => None,
        }
    }
}

/// Synthetic variable (`CFGraph::N<Var>`).
#[derive(Clone, Debug)]
pub struct SyntheticVar {
    pub uid: Id,
    pub ty: Option<TypeId>,
    pub name: String,
}

/// Port of `SyntheticAssignInstr::Kind`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SyntheticAssignKind {
    Unknown,
    Known,
    NextValue,
    Add,
}

/// Port of `SyntheticAssignInstr`.
#[derive(Clone, Debug)]
pub struct SyntheticAssignInstr {
    pub lhs: CfVar,
    pub kind: SyntheticAssignKind,
    pub arg: Option<CfValue>,
    pub diff: i64,
}

/// Port of `SyntheticPhiInstr::Predecessor`.
#[derive(Clone, Debug)]
pub struct PhiPredecessor {
    pub pred: BlockId,
    pub result: CfValue,
}

/// Port of `SyntheticPhiInstr`.
#[derive(Clone, Debug, Default)]
pub struct SyntheticPhiInstr {
    pub preds: Vec<PhiPredecessor>,
}

/// Kinds of synthetic values.
#[derive(Clone, Debug)]
pub enum SyntheticKind {
    Assign(SyntheticAssignInstr),
    Phi(SyntheticPhiInstr),
    /// `N<VarValue>(var)`.
    VarValue { var: CfVar },
    /// `N<CallInstr>(callee, args)` (pipeline stage call).
    Call { callee: CfValue, args: Vec<CfValue> },
    /// `N<ForFlow>(iter, body, var)` (pipeline generator stage).
    For { iter: CfValue, body: CfValue, var: CfVar },
    /// `N<SeriesFlow>()`.
    Series { items: Vec<CfValue> },
}

/// A synthetic value owned by the graph.
#[derive(Clone, Debug)]
pub struct SyntheticValue {
    pub uid: Id,
    /// `SyntheticInstr::getSource()`.
    pub source: Option<NodeRef>,
    pub kind: SyntheticKind,
}

impl SyntheticValue {
    /// `getUsedValues()`.
    pub fn used_values(&self) -> Vec<CfValue> {
        match &self.kind {
            SyntheticKind::Assign(a) => a.arg.into_iter().collect(),
            SyntheticKind::Phi(p) => p.preds.iter().map(|p| p.result).collect(),
            SyntheticKind::VarValue { .. } => Vec::new(),
            SyntheticKind::Call { callee, args } => {
                let mut r = args.clone();
                r.push(*callee);
                r
            }
            SyntheticKind::For { iter, body, .. } => vec![*iter, *body],
            SyntheticKind::Series { items } => items.clone(),
        }
    }
}

/// Port of `CFBlock`.
#[derive(Clone, Debug)]
pub struct CFBlock {
    pub uid: Id,
    name: String,
    values: Vec<CfValue>,
    successors: IndexSet<BlockId>,
    predecessors: IndexSet<BlockId>,
}

impl CFBlock {
    pub fn name(&self) -> &str {
        &self.name
    }
    pub fn values(&self) -> &[CfValue] {
        &self.values
    }
    pub fn front(&self) -> Option<CfValue> {
        self.values.first().copied()
    }
    pub fn back(&self) -> Option<CfValue> {
        self.values.last().copied()
    }
    pub fn len(&self) -> usize {
        self.values.len()
    }
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
    pub fn successors(&self) -> impl Iterator<Item = BlockId> + '_ {
        self.successors.iter().copied()
    }
    pub fn predecessors(&self) -> impl Iterator<Item = BlockId> + '_ {
        self.predecessors.iter().copied()
    }
    pub fn num_successors(&self) -> usize {
        self.successors.len()
    }
    pub fn num_predecessors(&self) -> usize {
        self.predecessors.len()
    }
}

/// Port of `CFGraph`.
#[derive(Debug)]
pub struct CFGraph {
    blocks: Vec<CFBlock>,
    cur: BlockId,
    func: FuncId,
    synthetics: Vec<SyntheticValue>,
    synth_vars: Vec<SyntheticVar>,
    synth_by_uid: FxHashMap<Id, SynthId>,
    /// `valueMapping` (interior mutability so `remapValue` works on shared graphs).
    value_mapping: RefCell<FxHashMap<Id, CfValue>>,
    value_locations: FxHashMap<Id, BlockId>,
}

impl CFGraph {
    /// `CFGraph(f)`: creates the entry block.
    pub fn new(func: FuncId) -> Self {
        let mut g = Self {
            blocks: Vec::new(),
            cur: 0,
            func,
            synthetics: Vec::new(),
            synth_vars: Vec::new(),
            synth_by_uid: FxHashMap::default(),
            value_mapping: RefCell::new(FxHashMap::default()),
            value_locations: FxHashMap::default(),
        };
        g.new_block("entry", true);
        g
    }

    pub fn size(&self) -> usize {
        self.blocks.len()
    }
    pub fn blocks(&self) -> &[CFBlock] {
        &self.blocks
    }
    pub fn block(&self, b: BlockId) -> &CFBlock {
        &self.blocks[b]
    }
    pub fn block_ids(&self) -> std::ops::Range<BlockId> {
        0..self.blocks.len()
    }
    /// `getEntryBlock()`.
    pub fn entry_block(&self) -> BlockId {
        0
    }
    pub fn current_block(&self) -> BlockId {
        self.cur
    }
    pub fn set_current_block(&mut self, b: BlockId) {
        self.cur = b;
    }
    pub fn func(&self) -> FuncId {
        self.func
    }
    pub fn set_func(&mut self, f: FuncId) {
        self.func = f;
    }

    /// `newBlock(name, setCur)`.
    pub fn new_block(&mut self, name: impl Into<String>, set_cur: bool) -> BlockId {
        self.blocks.push(CFBlock {
            uid: next_id(),
            name: name.into(),
            values: Vec::new(),
            successors: IndexSet::new(),
            predecessors: IndexSet::new(),
        });
        let id = self.blocks.len() - 1;
        if set_cur {
            self.cur = id;
        }
        id
    }

    fn reg(&mut self, m: &Module, b: BlockId, v: CfValue) {
        let uid = self.uid(m, v);
        self.value_locations.insert(uid, b);
    }
    /// `CFBlock::push_back`.
    pub fn push_back(&mut self, m: &Module, b: BlockId, v: CfValue) {
        self.blocks[b].values.push(v);
        self.reg(m, b, v);
    }
    /// `CFBlock::insert`.
    pub fn insert(&mut self, m: &Module, b: BlockId, idx: usize, v: CfValue) {
        self.blocks[b].values.insert(idx, v);
        self.reg(m, b, v);
    }
    /// `CFBlock::erase` (the value location is kept, like C++).
    pub fn erase(&mut self, b: BlockId, idx: usize) {
        self.blocks[b].values.remove(idx);
    }
    /// `successors_insert`.
    pub fn successors_insert(&mut self, from: BlockId, to: BlockId) {
        self.blocks[from].successors.insert(to);
        self.blocks[to].predecessors.insert(from);
    }
    /// `successors_erase`.
    pub fn successors_erase(&mut self, from: BlockId, to: BlockId) {
        self.blocks[from].successors.shift_remove(&to);
        self.blocks[to].predecessors.shift_remove(&from);
    }

    /// `getId()` of a CFG value.
    pub fn uid(&self, m: &Module, v: CfValue) -> Id {
        match v {
            CfValue::Real(v) => m.value_uid(v),
            CfValue::Synth(s) => self.synthetics[s].uid,
        }
    }
    /// `getId()` of a CFG variable.
    pub fn var_uid(&self, m: &Module, v: CfVar) -> Id {
        match v {
            CfVar::Real(v) => m.var_uid(v),
            CfVar::Synth(s) => self.synth_vars[s].uid,
        }
    }

    /// `getBlock(v)`.
    pub fn get_block(&self, m: &Module, v: impl Into<CfValue>) -> Option<BlockId> {
        let mut v = v.into();
        let uid = self.uid(m, v);
        if let Some(mapped) = self.value_mapping.borrow().get(&uid) {
            v = *mapped;
        }
        self.value_locations.get(&self.uid(m, v)).copied()
    }

    /// `remapValue(id, newValue)`.
    pub fn remap_value(&self, uid: Id, new: CfValue) {
        self.value_mapping.borrow_mut().insert(uid, new);
    }
    /// `remapValue(original, newValue)`.
    pub fn remap_value_of(&self, m: &Module, original: impl Into<CfValue>, new: CfValue) {
        let uid = self.uid(m, original.into());
        self.remap_value(uid, new);
    }

    /// `getValue(id)`.
    pub fn get_value(&self, m: &Module, uid: Id) -> Option<CfValue> {
        if let Some(v) = self.value_mapping.borrow().get(&uid) {
            return Some(*v);
        }
        if let Some(s) = self.synth_by_uid.get(&uid) {
            return Some(CfValue::Synth(*s));
        }
        m.get_value(uid).map(CfValue::Real)
    }

    /// Adds a synthetic value (`N<...>`).
    pub fn add_synthetic(&mut self, source: Option<NodeRef>, kind: SyntheticKind) -> CfValue {
        let uid = next_id();
        self.synthetics.push(SyntheticValue { uid, source, kind });
        let id = self.synthetics.len() - 1;
        self.synth_by_uid.insert(uid, id);
        CfValue::Synth(id)
    }
    /// Adds a synthetic variable (`N<Var>(type)`).
    pub fn add_synthetic_var(&mut self, ty: Option<TypeId>, name: impl Into<String>) -> CfVar {
        self.synth_vars.push(SyntheticVar { uid: next_id(), ty, name: name.into() });
        CfVar::Synth(self.synth_vars.len() - 1)
    }
    pub fn synth(&self, s: SynthId) -> &SyntheticValue {
        &self.synthetics[s]
    }
    pub fn synth_mut(&mut self, s: SynthId) -> &mut SyntheticValue {
        &mut self.synthetics[s]
    }
    pub fn synth_var(&self, s: usize) -> &SyntheticVar {
        &self.synth_vars[s]
    }
    /// `synth_begin()/synth_end()`.
    pub fn synthetics(&self) -> impl Iterator<Item = (CfValue, &SyntheticValue)> {
        self.synthetics.iter().enumerate().map(|(i, s)| (CfValue::Synth(i), s))
    }

    /// `cast<SyntheticAssignInstr>(v)`.
    pub fn as_synthetic_assign(&self, v: CfValue) -> Option<&SyntheticAssignInstr> {
        match v {
            CfValue::Synth(s) => match &self.synthetics[s].kind {
                SyntheticKind::Assign(a) => Some(a),
                _ => None,
            },
            _ => None,
        }
    }
    /// `cast<SyntheticPhiInstr>(v)`.
    pub fn as_synthetic_phi(&self, v: CfValue) -> Option<&SyntheticPhiInstr> {
        match v {
            CfValue::Synth(s) => match &self.synthetics[s].kind {
                SyntheticKind::Phi(p) => Some(p),
                _ => None,
            },
            _ => None,
        }
    }
    /// `SyntheticInstr::getSource()`.
    pub fn synth_source(&self, v: CfValue) -> Option<NodeRef> {
        match v {
            CfValue::Synth(s) => self.synthetics[s].source,
            _ => None,
        }
    }
    /// `cast<VarValue>(v)->getVar()` for real and synthetic var values.
    pub fn value_var(&self, m: &Module, v: CfValue) -> Option<CfVar> {
        match v {
            CfValue::Real(v) => m.as_var_value(v).map(|x| CfVar::Real(m.actual_var(x.var))),
            CfValue::Synth(s) => match &self.synthetics[s].kind {
                SyntheticKind::VarValue { var } => Some(*var),
                _ => None,
            },
        }
    }
    /// `isA<AssignInstr>(v)`.
    pub fn is_assign(&self, m: &Module, v: CfValue) -> bool {
        matches!(v, CfValue::Real(x) if m.as_assign_instr(x).is_some())
    }
    /// `isA<PointerValue>(v)`.
    pub fn is_pointer_value(&self, m: &Module, v: CfValue) -> bool {
        matches!(v, CfValue::Real(x) if m.as_pointer_value(x).is_some())
    }

    /// `operator<<`: graphviz output.
    pub fn format(&self, m: &Module) -> String {
        let mut os = String::new();
        let _ = writeln!(os, "digraph \"{}\" {{", m.var_name(self.func));
        for (i, b) in self.blocks.iter().enumerate() {
            let _ = write!(os, "  {}_{} [ label=\"{}\"", b.name, i, b.name);
            if i == self.entry_block() {
                os.push_str(" shape=square");
            }
            os.push_str(" ];\n");
        }
        for (i, b) in self.blocks.iter().enumerate() {
            for s in b.successors() {
                let _ = writeln!(os, "  {}_{} -> {}_{};", b.name, i, self.blocks[s].name, s);
            }
        }
        os.push('}');
        os
    }
}

/// Port of `CFResult`.
#[derive(Clone, Debug, Default)]
pub struct CFResult {
    /// map from function id to control-flow graph
    pub graphs: FxHashMap<Id, Rc<CFGraph>>,
}

impl AnalysisResult for CFResult {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Port of `CFAnalysis`.
#[derive(Clone, Debug, Default)]
pub struct CFAnalysis;

impl CFAnalysis {
    pub const KEY: &'static str = "core-analyses-cfg";

    /// Runs the analysis directly.
    pub fn compute(m: &Module) -> CFResult {
        let mut res = CFResult::default();
        let main = m.main_func();
        if is_bodied(m, main) {
            res.graphs.insert(m.var_uid(main), Rc::new(build_cf_graph(m, main)));
        }
        for var in m.vars() {
            if is_bodied(m, var) {
                res.graphs.entry(m.var_uid(var)).or_insert_with(|| Rc::new(build_cf_graph(m, var)));
            }
        }
        res
    }
}

impl Analysis for CFAnalysis {
    fn key(&self) -> String {
        Self::KEY.to_string()
    }
    fn run(&mut self, m: &Module, _ctx: &dyn PassContext) -> Box<dyn AnalysisResult> {
        Box::new(Self::compute(m))
    }
}

pub(crate) fn is_bodied(m: &Module, v: VarId) -> bool {
    m.func(v).map(|f| f.is_bodied()).unwrap_or(false)
}

/// `buildCFGraph(f)`.
pub fn build_cf_graph(m: &Module, f: FuncId) -> CFGraph {
    let mut g = CFGraph::new(f);
    let mut v = CFVisitor::new(m, &mut g);
    v.visit_bodied_func(f);
    g
}

/// Custom CF builder callback (`dsl::codegen::CFBuilder::buildCFNodes`).
pub type CfBuilderFn = Rc<dyn Fn(&dyn CustomValue, ValueId, &mut CFVisitor<'_>)>;

thread_local! {
    static CF_BUILDERS: RefCell<FxHashMap<StdTypeId, CfBuilderFn>> = RefCell::new(FxHashMap::default());
}

/// Registers the CFG builder of a DSL custom node type (C++ `CustomFlow/CustomInstr::getCFBuilder`).
///
/// PORT-NOTE: `typhon_ir::dsl::CustomValue` has no CF-builder hook, so DSL plugins register
/// one per concrete type here. Unregistered custom nodes process their used values and are
/// then inserted as ordinary values.
pub fn register_cf_builder<T: CustomValue + 'static>(f: impl Fn(&T, ValueId, &mut CFVisitor<'_>) + 'static) {
    let cb: CfBuilderFn = Rc::new(move |cv: &dyn CustomValue, v, vis| {
        if let Some(t) = cv.as_any().downcast_ref::<T>() {
            f(t, v, vis)
        }
    });
    CF_BUILDERS.with(|b| b.borrow_mut().insert(StdTypeId::of::<T>(), cb));
}

#[derive(Clone, Copy, Debug)]
struct Loop {
    next_it: BlockId,
    end: BlockId,
    loop_id: Id,
    tc_index: i32,
}

/// Port of `CFVisitor`.
pub struct CFVisitor<'a> {
    m: &'a Module,
    graph: &'a mut CFGraph,
    try_catch_stack: Vec<(BlockId, Option<BlockId>)>,
    seen_ids: FxHashSet<Id>,
    loop_stack: Vec<Loop>,
}

impl<'a> CFVisitor<'a> {
    pub fn new(m: &'a Module, graph: &'a mut CFGraph) -> Self {
        Self { m, graph, try_catch_stack: Vec::new(), seen_ids: FxHashSet::default(), loop_stack: Vec::new() }
    }
    pub fn module(&self) -> &'a Module {
        self.m
    }
    pub fn graph(&self) -> &CFGraph {
        self.graph
    }
    pub fn graph_mut(&mut self) -> &mut CFGraph {
        self.graph
    }

    fn tc_level(&self) -> i32 {
        self.try_catch_stack.len() as i32 - 1
    }

    /// `visit(const BodiedFunc*)`.
    pub fn visit_bodied_func(&mut self, f: FuncId) {
        let m = self.m;
        let uid = m.var_uid(f);
        if !self.seen_ids.insert(uid) {
            return;
        }
        let blk = self.graph.current_block();
        for arg in m.func_args(f) {
            let vv = self.graph.add_synthetic(None, SyntheticKind::VarValue { var: CfVar::Real(arg) });
            let sa = self.graph.add_synthetic(
                Some(NodeRef::Var(f)),
                SyntheticKind::Assign(SyntheticAssignInstr {
                    lhs: CfVar::Real(arg),
                    kind: SyntheticAssignKind::Known,
                    arg: Some(vv),
                    diff: 0,
                }),
            );
            self.graph.push_back(m, blk, sa);
        }
        if let Some(body) = m.func_body(f) {
            self.process(body);
        }
    }

    /// `process(v)`.
    pub fn process(&mut self, v: impl Into<CfValue>) {
        let v = match v.into() {
            CfValue::Real(x) => CfValue::Real(self.m.actual_value(x)),
            s => s,
        };
        let uid = self.graph.uid(self.m, v);
        if !self.seen_ids.insert(uid) {
            return;
        }
        match v {
            CfValue::Real(x) => self.visit_real(x),
            CfValue::Synth(s) => self.visit_synth(s),
        }
    }
    pub fn process_opt(&mut self, v: Option<ValueId>) {
        if let Some(v) = v {
            self.process(v);
        }
    }

    fn visit_synth(&mut self, s: SynthId) {
        let kind = self.graph.synth(s).kind.clone();
        match kind {
            SyntheticKind::Series { items } => {
                for i in items {
                    self.process(i);
                }
            }
            SyntheticKind::Call { callee, args } => {
                self.process(callee);
                for a in args {
                    self.process(a);
                }
                self.default_insert(CfValue::Synth(s));
            }
            SyntheticKind::For { iter, body, var } => {
                let uid = self.graph.synth(s).uid;
                let source = self.graph.synth(s).source;
                self.visit_for(source, uid, Vec::new(), iter, body, var);
            }
            _ => self.default_insert(CfValue::Synth(s)),
        }
    }

    fn visit_real(&mut self, v: ValueId) {
        let m = self.m;
        let kind = m.kind(v).clone();
        match kind {
            ValueKind::SeriesFlow(f) => {
                for c in f.series {
                    self.process(c);
                }
            }
            ValueKind::IfFlow(f) => {
                self.process(f.cond);
                let original = self.graph.current_block();
                let end = self.graph.new_block("endIf", false);
                let t_branch = self.graph.new_block("trueBranch", true);
                self.process(f.true_branch);
                let cur = self.graph.current_block();
                self.graph.successors_insert(cur, end);
                let mut f_branch = None;
                if let Some(fb) = f.false_branch {
                    let b = self.graph.new_block("falseBranch", true);
                    f_branch = Some(b);
                    self.process(fb);
                    let cur = self.graph.current_block();
                    self.graph.successors_insert(cur, end);
                }
                self.graph.successors_insert(original, t_branch);
                self.graph.successors_insert(original, f_branch.unwrap_or(end));
                self.graph.set_current_block(end);
            }
            ValueKind::WhileFlow(f) => {
                let original = self.graph.current_block();
                let end = self.graph.new_block("endWhile", false);
                let loop_begin = self.graph.new_block("whileBegin", true);
                self.graph.successors_insert(original, loop_begin);
                self.process(f.cond);
                let cond_end = self.graph.current_block();
                self.graph.successors_insert(cond_end, end);
                self.loop_stack.push(Loop { next_it: loop_begin, end, loop_id: m.value_uid(v), tc_index: self.tc_level() });
                let body = self.graph.new_block("whileBody", true);
                self.graph.successors_insert(cond_end, body);
                self.process(f.body);
                self.loop_stack.pop();
                let cur = self.graph.current_block();
                self.graph.successors_insert(cur, loop_begin);
                self.graph.set_current_block(end);
            }
            ValueKind::ForFlow(f) => {
                let sched = f.schedule.as_ref().map(|s| s.used_values()).unwrap_or_default();
                self.visit_for(Some(NodeRef::Value(v)), m.value_uid(v), sched, f.iter.into(), f.body.into(), CfVar::Real(f.var));
            }
            ValueKind::ImperativeForFlow(f) => {
                if let Some(s) = &f.schedule {
                    for x in s.used_values() {
                        self.process(x);
                    }
                }
                let src = Some(NodeRef::Value(v));
                let original = self.graph.current_block();
                let end = self.graph.new_block("endFor", false);
                let loop_begin = self.graph.new_block("forBegin", true);
                self.graph.successors_insert(original, loop_begin);
                let sa = self.graph.add_synthetic(
                    src,
                    SyntheticKind::Assign(SyntheticAssignInstr {
                        lhs: CfVar::Real(f.var),
                        kind: SyntheticAssignKind::Known,
                        arg: Some(f.start.into()),
                        diff: 0,
                    }),
                );
                self.graph.push_back(m, loop_begin, sa);
                self.process(f.start);
                self.process(f.end);
                let loop_check = self.graph.new_block("forCheck", false);
                let cur = self.graph.current_block();
                self.graph.successors_insert(cur, loop_check);
                self.graph.successors_insert(loop_check, end);
                let loop_next = self.graph.new_block("forUpdate", false);
                let upd = self.graph.add_synthetic(
                    src,
                    SyntheticKind::Assign(SyntheticAssignInstr {
                        lhs: CfVar::Real(f.var),
                        kind: SyntheticAssignKind::Add,
                        arg: None,
                        diff: f.step,
                    }),
                );
                self.graph.push_back(m, loop_next, upd);
                self.graph.successors_insert(loop_next, loop_check);
                self.loop_stack.push(Loop { next_it: loop_check, end, loop_id: m.value_uid(v), tc_index: self.tc_level() });
                let body = self.graph.new_block("forBody", true);
                self.graph.successors_insert(loop_check, body);
                self.process(f.body);
                let cur = self.graph.current_block();
                self.graph.successors_insert(cur, loop_check);
                self.loop_stack.pop();
                self.graph.set_current_block(end);
            }
            ValueKind::TryCatchFlow(f) => {
                let route = self.graph.new_block("tcRoute", false);
                let end = self.graph.new_block("tcEnd", false);
                let else_ = f.else_.map(|_| self.graph.new_block("tcElse", false));
                let finally = f.finally.map(|_| self.graph.new_block("tcFinally", false));
                let dst = finally.unwrap_or(end);
                self.try_catch_stack.push((route, finally));
                self.process(f.body);
                let cur = self.graph.current_block();
                self.graph.successors_insert(cur, else_.unwrap_or(dst));
                for c in &f.catches {
                    let c_block = self.graph.new_block("catch", true);
                    if let Some(var) = c.var {
                        let sa = self.graph.add_synthetic(
                            Some(NodeRef::Value(v)),
                            SyntheticKind::Assign(SyntheticAssignInstr {
                                lhs: CfVar::Real(var),
                                kind: SyntheticAssignKind::Unknown,
                                arg: None,
                                diff: 0,
                            }),
                        );
                        self.graph.push_back(m, c_block, sa);
                    }
                    self.process(c.handler);
                    self.graph.successors_insert(route, c_block);
                    let cur = self.graph.current_block();
                    self.graph.successors_insert(cur, dst);
                }
                if let (Some(e), Some(eb)) = (f.else_, else_) {
                    self.graph.set_current_block(eb);
                    self.process(e);
                    let cur = self.graph.current_block();
                    self.graph.successors_insert(cur, dst);
                }
                self.try_catch_stack.pop();
                if let (Some(fin), Some(fb)) = (f.finally, finally) {
                    self.graph.set_current_block(fb);
                    self.process(fin);
                    let cur = self.graph.current_block();
                    self.graph.successors_insert(cur, end);
                    self.graph.successors_insert(route, fb);
                }
                if let Some(&(outer_route, _)) = self.try_catch_stack.last() {
                    self.graph.successors_insert(finally.unwrap_or(route), outer_route);
                }
                self.graph.set_current_block(end);
            }
            ValueKind::PipelineFlow(_) => {
                if let Some(loops) = self.convert_pipeline_to_for_loops(v) {
                    self.process(loops);
                }
            }
            ValueKind::AssignInstr(i) => {
                self.process(i.rhs);
                self.default_insert(v.into());
            }
            ValueKind::ExtractInstr(i) => {
                self.process(i.val);
                self.default_insert(v.into());
            }
            ValueKind::InsertInstr(i) => {
                self.process(i.lhs);
                self.process(i.rhs);
                self.default_insert(v.into());
            }
            ValueKind::CallInstr(i) => {
                self.process(i.callee);
                for a in i.args {
                    self.process(a);
                }
                self.default_insert(v.into());
            }
            ValueKind::TernaryInstr(i) => {
                let end = self.graph.new_block("ternaryDone", false);
                let t_branch = self.graph.new_block("ternaryTrue", false);
                let f_branch = self.graph.new_block("ternaryFalse", false);
                self.process(i.cond);
                let cur = self.graph.current_block();
                self.graph.successors_insert(cur, t_branch);
                self.graph.successors_insert(cur, f_branch);
                self.graph.set_current_block(t_branch);
                self.process(i.true_value);
                let cur = self.graph.current_block();
                self.graph.successors_insert(cur, end);
                self.graph.set_current_block(f_branch);
                self.process(i.false_value);
                let cur = self.graph.current_block();
                self.graph.successors_insert(cur, end);
                // PORT-NOTE: like C++, predecessors are the branch entry blocks.
                let phi = self.graph.add_synthetic(
                    Some(NodeRef::Value(v)),
                    SyntheticKind::Phi(SyntheticPhiInstr {
                        preds: vec![
                            PhiPredecessor { pred: t_branch, result: i.true_value.into() },
                            PhiPredecessor { pred: f_branch, result: i.false_value.into() },
                        ],
                    }),
                );
                self.graph.push_back(m, end, phi);
                self.graph.remap_value_of(m, v, phi);
                self.graph.set_current_block(end);
            }
            ValueKind::BreakInstr(i) => {
                if let Some(lp) = self.find_loop(i.loop_) {
                    self.default_jump(Some(lp.end), lp.tc_index);
                }
                self.default_insert(v.into());
            }
            ValueKind::ContinueInstr(i) => {
                if let Some(lp) = self.find_loop(i.loop_) {
                    self.default_jump(Some(lp.next_it), lp.tc_index);
                }
                self.default_insert(v.into());
            }
            ValueKind::ReturnInstr(i) => {
                self.process_opt(i.value);
                self.default_jump(None, -1);
                self.default_insert(v.into());
            }
            ValueKind::YieldInstr(i) => {
                self.process_opt(i.value);
                self.default_insert(v.into());
            }
            ValueKind::AwaitInstr(i) => {
                self.process(i.value);
                self.default_insert(v.into());
            }
            ValueKind::ThrowInstr(i) => {
                self.process_opt(i.value);
                self.default_insert(v.into());
            }
            ValueKind::FlowInstr(i) => {
                self.process(i.flow);
                self.process(i.val);
                self.default_insert(v.into());
            }
            ValueKind::Custom(c) if c.category != typhon_ir::dsl::CustomCategory::Const => {
                let tid = c.node.borrow().as_any().type_id();
                let cb = CF_BUILDERS.with(|b| b.borrow().get(&tid).cloned());
                match cb {
                    Some(cb) => {
                        let node = c.node.clone();
                        let borrowed = node.borrow();
                        cb(&**borrowed, v, self);
                    }
                    None => {
                        let used = c.node.borrow().used_values();
                        for u in used {
                            self.process(u);
                        }
                        if c.category == typhon_ir::dsl::CustomCategory::Instr {
                            self.default_insert(v.into());
                        }
                    }
                }
            }
            // VarValue, PointerValue, constants, StackAlloc, YieldIn, custom consts, Any
            _ => self.default_insert(v.into()),
        }
    }

    fn visit_for(
        &mut self,
        source: Option<NodeRef>,
        loop_uid: Id,
        sched: Vec<ValueId>,
        iter: CfValue,
        body: CfValue,
        var: CfVar,
    ) {
        let m = self.m;
        for x in sched {
            self.process(x);
        }
        let original = self.graph.current_block();
        let end = self.graph.new_block("endFor", false);
        let loop_begin = self.graph.new_block("forBegin", true);
        self.graph.successors_insert(original, loop_begin);
        self.process(iter);
        let loop_check = self.graph.new_block("forCheck", false);
        let cur = self.graph.current_block();
        self.graph.successors_insert(cur, loop_check);
        self.graph.successors_insert(loop_check, end);
        let loop_next = self.graph.new_block("forNext", false);
        self.graph.successors_insert(loop_check, loop_next);
        let sa = self.graph.add_synthetic(
            source,
            SyntheticKind::Assign(SyntheticAssignInstr {
                lhs: var,
                kind: SyntheticAssignKind::NextValue,
                arg: Some(iter),
                diff: 0,
            }),
        );
        self.graph.push_back(m, loop_next, sa);
        self.loop_stack.push(Loop { next_it: loop_check, end, loop_id: loop_uid, tc_index: self.tc_level() });
        let loop_body = self.graph.new_block("forBody", true);
        self.graph.successors_insert(loop_next, loop_body);
        self.process(body);
        let cur = self.graph.current_block();
        self.graph.successors_insert(cur, loop_check);
        self.loop_stack.pop();
        self.graph.set_current_block(end);
    }

    fn find_loop(&self, loop_: Option<ValueId>) -> Option<Loop> {
        match loop_ {
            Some(l) => {
                let id = self.m.value_uid(l);
                // PORT-NOTE: C++ dereferences `end()` if the loop is not found; fall back
                // to the innermost loop instead.
                self.loop_stack.iter().find(|x| x.loop_id == id).or(self.loop_stack.last()).copied()
            }
            None => self.loop_stack.last().copied(),
        }
    }

    /// `defaultInsert(v)`.
    pub fn default_insert(&mut self, v: CfValue) {
        let m = self.m;
        if self.try_catch_stack.is_empty() {
            let cur = self.graph.current_block();
            self.graph.push_back(m, cur, v);
        } else {
            let original = self.graph.current_block();
            let nb = self.graph.new_block("default", true);
            self.graph.successors_insert(original, nb);
            let route = self.try_catch_stack.last().unwrap().0;
            self.graph.successors_insert(nb, route);
            self.graph.push_back(m, nb, v);
        }
        let uid = self.graph.uid(m, v);
        self.seen_ids.insert(uid);
    }

    /// `defaultJump(cf, newTcLevel)`.
    pub fn default_jump(&mut self, cf: Option<BlockId>, new_tc_level: i32) {
        let cur_tc = self.tc_level();
        let cur = self.graph.current_block();
        if cur_tc == -1 || cur_tc <= new_tc_level {
            if let Some(cf) = cf {
                self.graph.successors_insert(cur, cf);
            }
        } else {
            let mut nearest_finally = None;
            for i in (new_tc_level + 1)..=cur_tc {
                if let Some(n) = self.try_catch_stack[i as usize].1 {
                    nearest_finally = Some(n);
                    break;
                }
            }
            if let Some(nf) = nearest_finally {
                let route = self.try_catch_stack.last().unwrap().0;
                self.graph.successors_insert(cur, route);
                if let Some(cf) = cf {
                    self.graph.successors_insert(nf, cf);
                }
            } else if let Some(cf) = cf {
                self.graph.successors_insert(cur, cf);
            }
        }
    }

    // ---- pipeline conversion (convertPipelineToForLoops) ----

    fn stage_output_element_type(&self, callee: ValueId, has_args: bool) -> Option<TypeId> {
        let m = self.m;
        let t = m.value_type(callee);
        let gen_ty = if has_args { m.func_type_ret(t)? } else { t };
        match m.type_kind(gen_ty) {
            TypeKind::Generator { base } => Some(*base),
            _ => None,
        }
    }

    fn call_stage(&mut self, src: NodeRef, callee: ValueId, args: &[Option<ValueId>], last: CfValue) -> CfValue {
        let args = args.iter().map(|a| a.map(CfValue::Real).unwrap_or(last)).collect();
        self.graph.add_synthetic(Some(src), SyntheticKind::Call { callee: callee.into(), args })
    }

    fn convert_pipeline_to_for_loops(&mut self, p: ValueId) -> Option<CfValue> {
        let stages = self.m.as_pipeline_flow(p)?.stages.clone();
        self.pipeline_helper(p, &stages, 0, None)
    }

    fn pipeline_helper(
        &mut self,
        p: ValueId,
        stages: &[typhon_ir::Stage],
        idx: usize,
        last: Option<CfValue>,
    ) -> Option<CfValue> {
        if idx >= stages.len() {
            return last;
        }
        let src = NodeRef::Value(p);
        let stage = &stages[idx];
        if idx == 0 {
            return self.pipeline_helper(p, stages, idx + 1, Some(stage.callee.into()));
        }
        let prev = &stages[idx - 1];
        let last = last.expect("pipeline stage without input");
        if prev.generator {
            let ty = self.stage_output_element_type(prev.callee, !prev.args.is_empty());
            let var = self.graph.add_synthetic_var(ty, "");
            let vv = self.graph.add_synthetic(None, SyntheticKind::VarValue { var });
            let call = self.call_stage(src, stage.callee, &stage.args, vv);
            let body = self.pipeline_helper(p, stages, idx + 1, Some(call));
            let series = self.graph.add_synthetic(None, SyntheticKind::Series { items: body.into_iter().collect() });
            Some(self.graph.add_synthetic(Some(src), SyntheticKind::For { iter: last, body: series, var }))
        } else {
            let call = self.call_stage(src, stage.callee, &stage.args, last);
            self.pipeline_helper(p, stages, idx + 1, Some(call))
        }
    }
}
