# typhon-ir-analyze public API

Port of `codon/cir/analyze/{dataflow,module}/*` and `codon/cir/util/{inlining,outlining}`.
All analyses implement `typhon_ir::pass::Analysis`; results implement `AnalysisResult` and are
looked up by the C++ `KEY` strings: `ctx.get_analysis::<RDResult>(rd_key)`.

| Module | C++ | Analysis (key) | Result |
|---|---|---|---|
| `dataflow::cfg` | `cfg.{h,cpp}` | `CFAnalysis` (`"core-analyses-cfg"`) | `CFResult { graphs: FxHashMap<Id, Rc<CFGraph>> }` |
| `dataflow::dominator` | `dominator.*` | `DominatorAnalysis::new(cf_key)` (`"core-analyses-dominator"`) | `DominatorResult { cfg_result: CFResult, results: FxHashMap<Id, Rc<DominatorInspector>> }` |
| `dataflow::reaching` | `reaching.*` | `RDAnalysis::new(cf_key)` (`"core-analyses-rd"`) | `RDResult { cfg_result: CFResult, results: FxHashMap<Id, Rc<RDInspector>> }` |
| `dataflow::capture` | `capture.*` | `CaptureAnalysis::new(rd_key, dom_key)` (`"core-analyses-capture"`) | `CaptureResult { rd_result, dom_result, results: FxHashMap<Id, Vec<CaptureInfo>> }` |
| `module::global_vars` | `global_vars.*` | `GlobalVarsAnalyses` (`"core-analyses-global-vars"`) | `GlobalVarsResult { assignments: FxHashMap<Id, Id> }` (var uid -> rhs value uid, `-1` = not constant) |
| `module::side_effect` | `module/side_effect.*` | `SideEffectAnalysis::new(cap_key, global_assignment_has_side_effects)` (`"core-analyses-side-effect"`) | `SideEffectResult { result: FxHashMap<Id, SideEffectStatus> }`, `has_side_effect(m, v)` |
| `util::inlining` | `util/inlining.*` | - | `inline_function(m, f, args, aggressive, src) -> InlineResult`, `inline_call(m, call, aggressive)` |
| `util::outlining` | `util/outlining.*` | - | `outline_region(m, parent, series, begin, end, allow_outflows, outline_globals, all_by_value) -> OutlineResult`, `outline_series(..)` |
| `util::side_effect` | `util/side_effect.*` | re-export of `typhon_ir::util::side_effect` | |

Function result maps are keyed by the function uid (`m.var_uid(f)`), exactly like C++ `getId()`.
Results are cheap to clone (`Rc` inside) so dependent results can keep their inputs (C++ raw `cfgResult` pointers).
If a dependency key is missing from the `PassContext`, the analysis computes the dependency itself (PORT-NOTE).

## CFG (`dataflow::cfg`)

CFG values are `CfValue::{Real(ValueId), Synth(SynthId)}` (C++ `const Value*` incl. synthetic
instructions owned by the graph). Synthetic variables (pipeline loops) are `CfVar::Synth(idx)`;
real ones `CfVar::Real(VarId)`. `impl From<ValueId> for CfValue`.

```rust
pub type BlockId = usize;
pub struct CFBlock { /* name, values: Vec<CfValue>, successors, predecessors (insertion-ordered sets) */ }
impl CFBlock { name(), values() -> &[CfValue], front(), back(), len(), is_empty(),
               successors() -> impl Iterator<BlockId>, predecessors(), num_successors(), num_predecessors() }
pub struct CFGraph { .. }
impl CFGraph {
  new(func) ; size(); blocks() -> &[CFBlock]; block(b) -> &CFBlock; entry_block() -> BlockId (=0);
  current_block(); set_current_block(b); func() -> FuncId;
  new_block(name, set_cur) -> BlockId; push_back(b, v); insert(b, idx, v); erase(b, idx);
  successors_insert(from, to); successors_erase(from, to);
  get_block(m, v: impl Into<CfValue>) -> Option<BlockId>;           // getBlock (applies remaps)
  remap_value(uid: Id, v: CfValue); remap_value_of(m, original, v);   // &self (interior mutability)
  get_value(m, uid) -> Option<CfValue>;
  uid(m, v: CfValue) -> Id;  synth(s) -> &SyntheticValue;  synthetics() -> iterator of (SynthId, &SyntheticValue);
  synth_var(i) -> &SyntheticVar; var_uid(m, CfVar) -> Id;
  add_synthetic(source, kind) -> CfValue; add_synthetic_var(ty, name) -> CfVar;
  format(m) -> String  // graphviz, `operator<<`
}
pub struct SyntheticValue { pub uid: Id, pub source: Option<NodeRef>, pub kind: SyntheticKind }
pub enum SyntheticKind { Assign(SyntheticAssignInstr), Phi(SyntheticPhiInstr),
                         VarValue { var: CfVar }, Call { callee: CfValue, args: Vec<CfValue> } }
pub struct SyntheticAssignInstr { pub lhs: CfVar, pub kind: SyntheticAssignKind, pub arg: Option<CfValue>, pub diff: i64 }
pub enum SyntheticAssignKind { Unknown, Known, NextValue, Add }
pub struct SyntheticPhiInstr { pub preds: Vec<PhiPredecessor { pred: BlockId, result: CfValue }> }
CFGraph::as_synthetic_assign(v: CfValue) -> Option<&SyntheticAssignInstr>   // cast<SyntheticAssignInstr>
CFGraph::as_synthetic_phi(v) -> Option<&SyntheticPhiInstr>
pub fn build_cf_graph(m: &Module, f: FuncId) -> CFGraph;      // buildCFGraph
pub struct CFVisitor<'a> { .. }  // process(v), default_insert(v), default_jump(Option<BlockId>, i32), graph_mut(), module()
pub fn register_cf_builder::<T: CustomValue + 'static>(f: fn(&T, ValueId, &mut CFVisitor))  // dsl CFBuilder
```
`SyntheticValue.source` is the node that gave rise to it (`getSource()`): the `BodiedFunc` (args),
`ForFlow`/`ImperativeForFlow`, `TryCatchFlow`, `TernaryInstr`, or the `PipelineFlow` (PORT-NOTE:
C++ uses the synthetic `ForFlow` built from the pipeline).

## Dominators

`DominatorInspector::new(Rc<CFGraph>)`, `.analyze()`, `.is_dominated(m, v, dominator) -> bool`.

## Reaching definitions

```rust
pub struct ReachingDef { pub assignment: CfValue, pub assignee: Option<CfValue>, /* uids */ }
impl ReachingDef { known(), id() -> Id /* assignee uid if known else assignment uid */,
                   assignment_uid() -> Id, assignee_value() -> Option<ValueId> /* real assignee */ }
RDInspector::new(Rc<CFGraph>); .analyze(m); .get_reaching_definitions(m, var: VarId, loc: impl Into<CfValue>) -> Vec<ReachingDef>;
.is_invalid(m, var) -> bool; .cfg() -> &Rc<CFGraph>
```
Typical transform use (`const_prop`):
```rust
let r = ctx.get_analysis::<RDResult>(&self.rd_key)?;
let rd = r.results.get(&m.var_uid(parent_func))?;
let defs = rd.get_reaching_definitions(m, var, v);
if defs.len() == 1 && defs[0].known() { let val = defs[0].assignee_value()?; ... }
// SyntheticAssignInstr check (devirtualize/indexing):
let cfg = rd.cfg(); if let Some(s) = cfg.as_synthetic_assign(defs[0].assignment) { s.kind, cfg.synth_source(defs[0].assignment) }
```

## Capture

`CaptureInfo { arg_captures: Vec<u32>, return_captures, extern_captures, modified }`, `.any()` (= `operator bool`),
`CaptureInfo::nothing()`, `CaptureInfo::unknown(m, func, ty)`. `escapes(m, func, value, &CaptureResult) -> CaptureInfo`.

## Side effects

`SideEffectResult::has_side_effect(&self, m, v: ValueId) -> bool` (`true` if unknown), `status(uid)`.

## Inlining / outlining

```rust
pub struct InlineResult { pub result: Option<ValueId>, pub new_vars: Vec<VarId> }  // is_some() == operator bool
pub struct OutlineResult { pub func: Option<FuncId>, pub call: Option<ValueId>, pub arg_kinds: Vec<ArgKind>, pub num_out_flows: i32 }
pub enum ArgKind { Constant, Modified }
```
Outlining with modified vars / outflows needs a `TypeRealizer` (`ptr_load/ptr_store`, `==`).
