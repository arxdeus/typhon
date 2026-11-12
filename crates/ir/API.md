# typhon-ir public API


## Memory model in one paragraph

`Module` owns three `la_arena` arenas: values (`ValueData`), vars (`VarData`, functions are
vars with `func: Some(FuncData)`), types (`TypeData`). Handles are `Copy` newtypes
`ValueId`, `VarId` (`FuncId` is an alias of `VarId`), `TypeId`. `NodeRef` is the
"any node" handle (`Value|Var|Type|Module`) used where a generic node is taken.
`IdMixin` ids (`getId()`) are `Id = i64`, exposed by `m.value_uid(v)` / `m.var_uid(v)`
(both resolve replacement). Every accessor resolves `replaceAll` forwarding
(`actual_*`), so storing unresolved ids in nodes is fine (same).
Nodes are never freed: `remove_*` / `pop_arena` mark them dead and unlink them from
the module lists/maps (`get_value(uid)`, `vars()`, `types()`...).

## Ids (`ids.rs`)

| Rust | Responsibility |
|---|---|
| `ValueId`, `VarId`, `FuncId = VarId`, `TypeId` | `Value*`, `Var*`, `Func*`, `Type*` |
| `NodeRef::{Value, Var, Type, Module}` (`From` impls for each id) | `Node*` |
| `Id = i64`, `ids::reset_id()` | `id_t`, `IdMixin::resetId()` (thread-local counter, Note) |

## Node data

```rust
pub struct NodeBase { pub name: String, pub attributes: AttributeMap, pub replaceable: bool }
pub struct ValueData { pub base: NodeBase, pub kind: ValueKind, /* uid, replacement */ }
pub struct VarData   { pub base: NodeBase, pub ty: Option<TypeId>, pub global, pub external,
                       pub tls: bool, pub func: Option<Box<FuncData>>, pub any: bool }
pub struct TypeData  { pub base: NodeBase, pub kind: TypeKind, pub ast_type: Option<AstTypeRef>,
                       pub runtime_info: Option<Box<RuntimeInfo>> }
```

`ValueKind` (flat; each variant wraps a struct with `pub` fields named like the getters):

| Variant / struct | Description | category |
|---|---|---|
| `IntConst(TemplatedConst<i64>)`, `FloatConst(..<f64>)`, `BoolConst(..<bool>)`, `StringConst(..<String>)`, `BytesConst(..<String>)` (`val`, `ty`) | `IntConst`... | const |
| `VarValue{var}`, `PointerValue{var, fields}` | same | value |
| `SeriesFlow{series: Vec<ValueId>}`, `IfFlow{cond, true_branch, false_branch: Option}`, `WhileFlow{cond, body}`, `ForFlow{iter, body, var, schedule: Option<Box<OMPSched>>, is_async}`, `ImperativeForFlow{start, step, end, body, var, schedule}`, `TryCatchFlow{body, else_, finally, catches: Vec<Catch{handler, ty: Option, var: Option}>}`, `PipelineFlow{stages: Vec<Stage{callee, args: Vec<Option<ValueId>>, generator, parallel}>}` | flows | flow |
| `AssignInstr{lhs: VarId, rhs}`, `ExtractInstr{val, field, polymorphic}`, `InsertInstr{lhs, field, rhs, polymorphic}`, `CallInstr{callee, args}`, `StackAllocInstr{ptr_type, count}`, `YieldInInstr{ty, suspend}`, `TernaryInstr{cond, true_value, false_value}`, `BreakInstr{loop_: Option}`, `ContinueInstr{loop_: Option}`, `ReturnInstr{value: Option}`, `YieldInstr{value: Option, final_}`, `AwaitInstr{value, ty, generator}`, `ThrowInstr{value: Option, fallback: Option}`, `FlowInstr{flow, val}` | instrs | instr |
| `Any{flow: bool}` | `util::AnyValue/AnyFlow` | matching only |

Predicates on `ValueKind`: `is_const()`, `is_flow()`, `is_instr()`, `is_control_flow_instr()`;
`Module::as_<snake>(id) -> Option<&T>` / `as_<snake>_mut` for every struct
(`as_call_instr`, `as_series_flow`, `as_int_const`, ...) = `cast<T>`; `m.is::<..>` = `isA`
is just `m.as_x(v).is_some()` or `matches!(m.kind(v), ValueKind::X(_))`.

`FuncData { unmangled_name, generator, is_async, parent_type: Option<TypeId>, args: Vec<VarId>, kind: FuncKind }`,
`FuncKind::{Bodied{symbols, body: Option<ValueId>, jit}, External, Internal{intrinsic: Intrinsic, intrinsic_type: Option<TypeId>}, Llvm{literals: Vec<Generic>, declarations, body}}`.
`Intrinsic` enum = `InternalFunc::Intrinsic` with `name()`, `is_type_query()`.

`TypeKind::{Int{len,sign}, Float, Float32, Float16, BFloat16, Float128, Bool, Record{fields: Vec<Field{name,ty}>}, Ref{contents: TypeId, polymorphic}, Func{ret, args, variadic}, Pointer{base}, Optional{base}, Generator{base}, Vector{count, base}, Union{types}, Custom(Rc<dyn CustomType>)}`
with `is_primitive()`, `is_membered()`, `is_derived()`; name helpers `int_type_name`, `pointer_type_name`, ... = `getInstanceName`.
`Generic::{Static(i64), StaticStr(String), Type(TypeId)}`. `RuntimeInfo` = `Type::RuntimeInfo`.

## Module (`module.rs`)

* `Module::new(name)`, `set_realizer(Rc<RefCell<dyn TypeRealizer>>)`, `realizer()`.
* `main_func()`, `argv_var()`, `argc_var()`; lists: `vars()`, `values()`, `types()` (live, insertion order);
  lookup: `get_var(uid)`, `get_value(uid)`, `get_type(name)`; `remove_var/remove_value/remove_type`.
* builders (`Nr<T>` / `N<T>(src,...)`): `nr(kind) -> ValueId`, `n(src, kind)`, `nr_named(kind, name)`;
  helpers `nr_int_const(v, ty)`, `nr_float_const`, `nr_bool_const`, `nr_string_const`, `nr_bytes_const`,
  `nr_var_value(var)`, `nr_pointer_value(var, fields)`, `nr_series_flow()`, `nr_if_flow(c,t,f)`,
  `nr_while_flow(c,b)`, `nr_for_flow(iter, body, var)`, `nr_imperative_for_flow(start, step, end, body, var)`,
  `nr_try_catch_flow(body, finally, else_)`, `nr_pipeline_flow(stages)`, `nr_assign(lhs, rhs)`,
  `nr_extract(val, field)`, `nr_insert(lhs, field, rhs)`, `nr_call(callee, args)`, `nr_stack_alloc(ty, n)`,
  `nr_yield_in(ty, suspend)`, `nr_ternary(c,t,f)`, `nr_break(loop)`, `nr_continue(loop)`, `nr_return(v)`,
  `nr_yield(v, final)`, `nr_await(v, ty, gen)`, `nr_throw(v)`, `nr_flow_instr(flow, v)`;
  vars: `nr_var(ty, global, name)`, `nr_var_full(src, ty, global, external, tls, name)`,
  `nr_func(name, FuncKind)`, `nr_bodied_func(name)`, `nr_external_func`, `nr_internal_func`, `nr_llvm_func`;
  types: `add_type(name, kind) -> TypeId` (`Nr<SomeType>`), `nr_record_type(name, types, names)`, `nr_ref_type(name, contents)`, `nr_func_type(name, ret, args, variadic)`.
* generic node API (`NodeRef`): `name`, `set_name`, `attributes`, `attributes_mut`, `has_attribute(n,key)`,
  `get_attribute(n,key)`, `set_attribute(n, Attribute)`, `set_attribute_key`, `erase_attribute`,
  `src_info`, `set_src_info`, `reference_string`, `has_replacement`, `is_replaceable`, `set_replaceable`,
  `used_values`, `replace_used_value(n, old: ValueId, new: impl Into<Option<ValueId>>)`,
  `replace_used_value_uid(n, uid, new)`, `used_types`, `replace_used_type(n, old, new)`,
  `replace_used_type_name(n, name, new)`, `used_variables`, `replace_used_variable(n, old, new)`,
  `replace_used_variable_uid`.
* values: `value(id)`, `value_mut`, `kind`, `kind_mut`, `actual_value`, `value_uid`, `value_type(id)` (`&self`; panics if the
  pointer type of a `PointerValue` was never created) / `value_type_mut(id)` (creates it),
  `replace_all_value(a, b)` (`replaceAll`).
* vars/funcs: `var`, `var_mut`, `actual_var`, `var_uid`, `var_type`, `set_var_type`, `func(id) -> Option<&FuncData>`,
  `func_mut`, `is_func`, `replace_all_var`, `realize_func(f, ty, names)` (`Func::realize`),
  `func_arg_var(f, name)` (`getArgVar`), `has_valid_intrinsic_signature(f)`,
  `intrinsic_side_effect_status(f)`, `intrinsic_derives_from_arguments(f)`, `is_variadic(f)`.
* types: `ty`, `ty_mut`, `actual_type`, `type_name`, `replace_all_type`, `type_is(a,b)` (`Type::is`),
  `type_is_atomic`, `type_is_content_atomic`, `type_generics`, `member_type(t, name)`, `member_index`,
  `type_fields(t)`, `realize_membered(t, types, names)`, `construct(t, args)`, `ref_contents(t)`,
  `func_type_ret/args`, `set_runtime_info`.
* type factories: `get_bool_type`, `get_int_type()`, `get_int_type_n(len, sign)`, `get_float_type`,
  `get_float32_type`, `get_float16_type`, `get_bfloat16_type`, `get_float128_type`, `get_string_type`,
  `get_bytes_type`, `get_pointer_type(base: Option)`, `get_generator_type`, `get_optional_type`,
  `get_func_type(ret, args, variadic)`, `get_vector_type`, `get_tuple_type`, `get_union_type`,
  `get_none_type`, `get_intrinsic(kind, ty, sig)`, `unsafe_get_*` (pointer, generator, optional,
  func, membered, int, vector, union).
* constants: `get_int`, `get_float`, `get_bool`, `get_string`, `get_bytes`.
* realization: `get_or_realize_method(parent, name, args, generics)`, `get_or_realize_func(name, args, generics, module)`,
  `get_or_realize_type(name, generics)`, `parse_code(code)`.
* value operators (`Value::operator+` ...): `unary_op(v, magic)`, `binary_op(v, magic, other)`,
  `call_value(v, args)`, `op_or`, `op_and`, `to_int`, `to_float`, `to_bool`, `to_str`, `len`, `iter`.
  Magic names: `module::magic::{EQ, NE, ..., NEW, INIT}` (`Module::EQ_MAGIC_NAME` ...).
* `push_arena()`, `pop_arena()`.

Note: without a `TypeRealizer` (tests), stdlib-backed factories (`get_pointer_type`,
`get_none_type`, `get_tuple_type`, ...) fall back to the `unsafe_get_*` structural types.

## TypeRealizer (`realize.rs`)

```rust
pub struct AstTypeRef(pub Rc<dyn Any>);          // opaque parser handle (types and functions)
pub trait TypeRealizer {
  fn find_class(&mut self, name: &str) -> Option<AstTypeRef>;
  fn find_function(&mut self, name: &str) -> Option<AstTypeRef>;
  fn find_method(&mut self, m: &Module, cls: &AstTypeRef, method: &str, args: &[TypeId]) -> Option<AstTypeRef>;
  fn realize_type(&mut self, m: &mut Module, cls: &AstTypeRef, generics: &[Generic]) -> Result<Option<TypeId>>;
  fn realize_function(&mut self, m: &mut Module, func: &AstTypeRef, args: &[TypeId],
                      generics: &[Generic], parent_cls: Option<&AstTypeRef>) -> Result<Option<FuncId>>;
  fn make_function(&mut self, m: &mut Module, ret: TypeId, args: &[TypeId]) -> Result<TypeId>;
  fn make_tuple(&mut self, m: &mut Module, args: &[TypeId]) -> Result<TypeId>;
  fn make_union(&mut self, m: &mut Module, types: &[TypeId]) -> Result<TypeId>;
  fn parse_code(&mut self, m: &mut Module, code: &str) -> Result<()>;
  fn generics(&mut self, m: &mut Module, ast: &AstTypeRef, func_generics: bool) -> Vec<Generic>;
}
```
`args` for `realize_function` are the IR arg types (parser prepends the unbound return
type like `translateArgs`). Module calls the realizer via `try_borrow_mut`; if the realizer
is already borrowed (re-entrant call from the parser) the call fails soft (logged, `None`).

## Attributes (`attribute.rs`)

`AttributeMap` (small vec of `(i32, Attribute)`), `Attribute::{SrcInfo, StringValue, IntValue,
StringList, KeyValue(KeyValueAttribute{attributes}), Member(BTreeMap<String,SrcInfo>),
PythonWrapper(FuncId), Docstring, TupleLiteral(Vec<ValueId>), ListLiteral(Vec<LiteralElement>),
SetLiteral(..), DictLiteral(Vec<KeyValuePair{key, value: Option}>), PartialFunction{name, args: Vec<Option<ValueId>>},
ExactType(TypeId), Custom(Box<dyn CustomAttribute>)}`. Keys: `attr_id::{STRING_VALUE=101, INT_VALUE=102,
STRING_LIST=103, KEY_VALUE=104, MEMBER=105, SRC_INFO=107, DOCSTRING=108,
TUPLE_LITERAL=109, LIST_LITERAL=111, SET_LITERAL=111, DICT_LITERAL=112, PARTIAL_FUNCTION=113, EXACT_TYPE=114}`.
`Attribute::id()`, `needs_clone()`, `format(m)`, typed accessors `as_key_value()` etc.

## Visitors (`util::visitor`)

`trait Visitor` (`&mut Module`) and `trait ConstVisitor` (`&Module`): one method per node kind
(`visit_module`, `visit_var`, `visit_bodied_func`, `visit_external_func`, `visit_internal_func`,
`visit_llvm_func`, `visit_var_value`, `visit_pointer_value`, `visit_series_flow`, `visit_if_flow`,
`visit_while_flow`, `visit_for_flow`, `visit_imperative_for_flow`, `visit_try_catch_flow`,
`visit_pipeline_flow`, `visit_int_const`, `visit_float_const`,
`visit_bool_const`, `visit_string_const`, `visit_bytes_const`,
`visit_assign_instr`, `visit_extract_instr`, `visit_insert_instr`, `visit_call_instr`,
`visit_stack_alloc_instr`, `visit_yield_in_instr`, `visit_ternary_instr`, `visit_break_instr`,
`visit_continue_instr`, `visit_return_instr`, `visit_yield_instr`, `visit_await_instr`,
`visit_throw_instr`, `visit_flow_instr`, `visit_any_value`, and per type kind
`visit_int_type` ... `visit_union_type`). Default = `default_visit(node)` (no-op). Dispatch: `util::visitor::{accept, accept_value, accept_var, accept_type}` (and
`const_accept*`), resolving replacements like `Node::accept`.

## Operator (`util::operator`)

```rust
pub trait Operator {
  fn state(&self) -> &OperatorState; fn state_mut(&mut self) -> &mut OperatorState;
  fn pre_hook(&mut self, m: &mut Module, n: NodeRef) {}  fn post_hook(..) {}
  fn handle_<kind>(&mut self, m: &mut Module, v: ValueId) {}   // every value kind
  fn visit_module_op(..), fn visit_bodied_func_op(..)          // overridable traversal
  // provided:
  fn process(&mut self, m, n: NodeRef); fn process_value(..); fn reset(&mut self);
  fn parent(&self, level) -> Option<NodeRef>; fn depth(); fn find_last(pred); fn parent_func(m);
  fn saw(m, v); fn see(m, v); fn insert_before(m, v); fn insert_after(m, v);
}
OperatorState::new(children_first)
```
`OperatorPass` = implement `Pass` + `Operator` and call `pass::run_operator_pass(self, m)` (reset + process module).
Other provided methods: `parent_series(m)`, `parents()`, `process_value`, `visit_value_op` (LAMBDA_VISIT body).
`insert_before/insert_after` operate on the innermost series flow being iterated (index-based iterator stack).

## Formatting, cloning, matching, tools

* `util::format::{format(m, n) -> String, format_value, format_type, format_var}`; `escape_string`.
* `util::cloning::CloneVisitor::new(clone_loop)`, `.clone_value(m, v) -> ValueId`,
  `.clone_value_opt`, `.clone_value_into(m, v, clone_to: Option<FuncId>, remaps: &FxHashMap<Id, VarId>)`,
  `.clone_var(m, v)`, `.force_clone_value`, `.force_clone_var`, `.force_remap(NodeRef, NodeRef)`, `.clone_stage`.
* `util::matching::{match_nodes(m, Option<NodeRef>, Option<NodeRef>, check_names, var_id_match), match_values, match_vars,
  match_types, match_nodes_strict_ids}`. Wildcards: `m.nr_any_value(flow)`, `m.nr_any_var(func)`.
* `util::irtools::*`: `has_attribute`, `is_export`, `set_export`, `is_gpu_kernel`, `is_inline`, `is_no_inline`,
  `derived_func_name`, `is_stdlib_func`, `is_int`, `call`, `virtual_call`, `is_call_of`, `is_call_of_types`,
  `is_magic_method_call`, `make_tuple`, `make_var`, `alloc`, `alloc_n`, `series`, `is_const::<T>`, `is_const_val`,
  `get_const::<T>` (T = i64/f64/bool/String), `get_var`, `get_func`, `ptr_load`, `ptr_store`, `tuple_get`,
  `tuple_store`, `get_stdlib_func`, `return_type`, `set_return_type`.
* `util::side_effect::{SideEffectStatus, NON_PURE_ATTR, PURE_ATTR, ...}`.
* `util::context::CirContext<Frame>`.

## Passes (`pass.rs`), schedule, rewrite

```rust
pub trait PassContext { fn analysis_result(&self, key: &str) -> Option<&dyn AnalysisResult>;
                        fn options(&self) -> Option<&Options>; }
impl dyn PassContext { fn get_analysis<T: AnalysisResult>(&self, key) -> Option<&T> }
pub trait Pass { fn key(&self) -> String; fn should_run(&self, m: &Module) -> bool { true }
                 fn run(&mut self, m: &mut Module, ctx: &dyn PassContext);
                 fn should_repeat(&self, num: i32) -> bool { false } }
pub struct PassGroup { pub key, pub repeat, pub passes: Vec<Box<dyn Pass>> }  // impl Pass
pub trait AnalysisResult: Any { fn as_any(&self) -> &dyn Any; }
pub trait Analysis { fn key(&self) -> String;
                     fn run(&mut self, m: &Module, ctx: &dyn PassContext) -> Box<dyn AnalysisResult>; }
pub struct NullPassContext;   // no analyses, no options
```
`schedule::OMPSched { code, dynamic, threads, chunk, ordered, collapse, gpu }` with
`new(m, code, dynamic, threads, chunk, ordered, collapse, gpu)`, `from_schedule(m, "dynamic", ...)`,
`used_values()`, `replace_used_value(m, uid, new)`.
`rewrite::{RewriteRule (fn apply(&mut self, m, v) -> Option<ValueId>), Rewriter{register_rule, rewrite, num_replacements, reset}}`.
`dsl::{CustomValue, CustomType, CustomNode, CustomCategory}` (builders: `as_any()` for downcast in typhon-llvm).
`CustomValue::ty(&self, m) -> Option<TypeId>` (`None` = none type), `matches`, `do_clone(cv, m)`, `format(m)`,
`side_effect_status(local)`, used/replace hooks for values/types/vars.

## Series / helpers
`m.series_push(s, v)`, `m.series_insert(s, idx, v)`, `m.try_catch_push(tc, handler, ty, var)`,
`m.set_throw_fallback`, `m.func_push_symbol(f, v)`, `m.set_func_body(f, b)`, `m.func_body(f)`, `m.func_args(f)`,
`m.set_intrinsic(f, kind, ty)`. Payload structs are directly mutable via `as_*_mut`.

## Semantics notes (Notes)
* `replace_used_value*` panics (`"X is not a flow"`) when a flow slot gets a non-flow value or a required slot gets
  `None`; the node is left consistent.
* `Module::new` eagerly creates the structural `NoneType` record so `value_type(&self)` works for flows/instrs.
* No realizer: `get_pointer_type`/`get_int_type_n`/`get_optional_type`/`get_tuple_type`/... use structural types.
* `IdMixin` counter is thread-local.
* Visitor `default_visit` is a no-op.
* `util::match` ignores `var_id_match` (bug preserved); `match_nodes_strict_ids` honours it.
  `TryCatchFlow`s never match.
* `Rewriter` rules apply in registration order.

