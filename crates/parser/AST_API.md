# typhon-parser AST API (port of `codon/parser/ast/**`, `visitors/visitor.h`, `visitors/format/*`, `match.h`)

Status: implemented. This file is the contract for the grammar (peg), scoping, typecheck,
translate and doc lanes. Keep it in sync with `src/ast/**`.

## Modules

| Rust module | C++ source |
|---|---|
| `ast::ids` (`ExprId`, `StmtId`) | raw `Expr*` / `Stmt*` |
| `ast::attr` (`Attr`, `AttrValue`, `Attrs`, `BindingsAttr`) | `ast/attr.h`, `ir::Attribute` subclasses used by the AST, `scoping.h::BindingsAttribute` |
| `ast::expr` (`Expr`, `ExprKind`, `*Expr`, `Param`, `CallArg`, `Pipe`, `StringPart`, `FormatSpec`) | `ast/expr.h` |
| `ast::stmt` (`Stmt`, `StmtKind`, `*Stmt`, `MatchCase`, `CallLayout`) | `ast/stmt.h` |
| `ast::arena` (`AstArena`, constructors, clone, generator helpers, misc methods) | `Cache::N<T>`, copy ctors `T(const T&, bool clean)`, `ast/{expr,stmt}.cpp` logic |
| `ast::sexpr` (`AstArena::expr_to_string/stmt_to_string`) | `toString(int)` |
| `ast::python_string` (`AstArena::expr_to_python/stmt_to_python`, `py_quote`, ...) | `toPythonString`, `pyNode`, `pyList`, ... |
| `ast::codon_string` (`AstArena::expr_to_codon/stmt_to_codon`, `format_float`) | `ast/codon_string.cpp` |
| `ast::serial` (`AstArena::serialize_*`/`deserialize`) | `util/serialize.h`, `tser.h` |
| `visitors::visitor` (`AstCtx`, `Visitor`, `walk_*`, `IdMapper`) | `ASTVisitor`, `CallbackASTVisitor`, `ReplacingCallbackASTVisitor` |
| `visitors::format` (`FormatVisitor`, `FormatCacheHooks`) | `visitors/format/format.{h,cpp}` |
| `matching` (`M`, `match_expr`, `match_stmt`) | `match.{h,cpp}` |

## Memory model

```rust
pub struct ExprId(Idx<Expr>);   // Copy, Hash, Ord, serde as u32
pub struct StmtId(Idx<Stmt>);
pub struct AstArena { exprs: Arena<Expr>, stmts: Arena<Stmt>, pub type_fmt: Option<Rc<dyn TypeFormatter>> }
arena[expr_id] / arena[stmt_id]           // Index/IndexMut -> &Expr / &Stmt
arena.new_expr(kind: impl Into<ExprKind>, src) -> ExprId
arena.new_stmt(kind: impl Into<StmtKind>, src) -> StmtId   // normalizes like C++ ctors:
     // suite slots wrapped via wrap_suite, PipeExpr ellipses -> Pipe mode
arena.wrap_suite(Option<StmtId>) -> Option<StmtId>          // SuiteStmt::wrap
arena.new_param(src, name, ty, default, status_hint /*0,1,2*/, ParamKind) -> Param // status inference
FunctionStmt::parse_params(Vec<Param>) -> Result<Vec<Param>>   // '/', '*' handling (errors)
arena.new_function(src, name, ret, params, suite, decorators, async) -> Result<StmtId>
arena.new_lambda(src, params, body) -> Result<ExprId>
arena.new_generator(src, GeneratorKind, expr, loops: Vec<StmtId>) -> ExprId
arena.new_dict_generator(src, key, expr, loops) -> ExprId
arena.clone_expr(id, clean) / clone_stmt(id, clean)        // deep copy (C++ clone(bool))
arena.import_expr(&other, id) / import_stmt(&other, id)    // deep copy across arenas
```
Every `Expr` / `Stmt` carries `src: SrcInfo` and `attrs: Attrs`.

```rust
pub struct Expr { pub kind: ExprKind, pub src: SrcInfo, pub attrs: Attrs,
                  pub ty: Option<TypeId>, pub done: bool,
                  pub orig_expr: Option<ExprId>, pub expected_type: Option<TypeId> }
pub struct Stmt { pub kind: StmtKind, pub src: SrcInfo, pub attrs: Attrs, pub done: bool }
```
Types are owned by the types lane (`crate::ids::TypeId`). Printing of types goes through
`trait TypeFormatter { fn debug_string(&self, TypeId, i32) -> String; fn realized_name(&self, TypeId) -> String; fn is_func(&self, TypeId) -> bool; }`
installed in `AstArena::type_fmt` (fallback prints `T<n>`).

Node accessors: `arena.expr(id).kind.as_call() -> Option<&CallExpr>`, `as_call_mut`, ..., one per
kind (`as_<snake>`), plus `ExprKind::tag() -> ExprTag`, `StmtKind::tag() -> StmtTag`, `is_id(arena, e, "x")`.

## Expression kinds (`ExprKind::<Variant>(<Struct>)`) — fields

| C++ | Rust variant / struct | fields |
|---|---|---|
| NoneExpr | `None(NoneExpr)` | – |
| BoolExpr | `Bool(BoolExpr)` | `value: bool` |
| IntExpr | `Int(IntExpr)` | `value: String, suffix: String, int_value: Option<i64>`; `IntExpr::new(i64)`, `IntExpr::parse(&str, suffix)` |
| FloatExpr | `Float(FloatExpr)` | `value, suffix, float_value: Option<f64>`; `FloatExpr::new(f64)` (`{:g}`), `FloatExpr::parse` |
| StringExpr | `String(StringExpr)` | `strings: Vec<StringPart{value, prefix, expr: Option<ExprId>, format: FormatSpec{text, conversion, spec}, src}>`; `value()`, `is_simple()` |
| IdExpr | `Id(IdExpr)` | `value: String` |
| StarExpr / KeywordStarExpr | `Star` / `KeywordStar` | `expr: Option<ExprId>` |
| Tuple/List/Set/DictExpr | `Tuple`/`List`/`Set`/`Dict` | `items: Vec<ExprId>` (dict items are 2-tuples or `KeywordStar`) |
| GeneratorExpr | `Generator(GeneratorExpr)` | `kind: GeneratorKind{Generator,ListGenerator,SetGenerator,TupleGenerator,DictGenerator}, loops: Option<StmtId>` |
| IfExpr | `If(IfExpr)` | `cond, ifexpr, elsexpr: Option<ExprId>` |
| UnaryExpr | `Unary` | `op: String, expr` |
| BinaryExpr | `Binary` | `op, lexpr, rexpr, in_place: bool` |
| ChainBinaryExpr | `ChainBinary` | `exprs: Vec<(String, ExprId)>` |
| PipeExpr | `Pipe(PipeExpr)` | `items: Vec<Pipe{op, expr}>, in_types: Vec<Option<TypeId>>` |
| IndexExpr | `Index` | `expr, index` |
| CallExpr | `Call` | `expr, items: Vec<CallArg{name, value: Option<ExprId>, src}>, ordered, partial` |
| DotExpr | `Dot` | `expr, member: String` |
| SliceExpr | `Slice` | `start, stop, step` |
| EllipsisExpr | `Ellipsis` | `mode: EllipsisType{Pipe,Partial,Standalone}` |
| LambdaExpr | `Lambda` | `items: Vec<Param>, expr` |
| YieldExpr | `Yield` | – |
| AwaitExpr | `Await` | `expr, transformed: bool` |
| AssignExpr | `Assign` | `var, expr` |
| RangeExpr | `Range` | `start, stop` |
| StmtExpr | `Stmt(StmtExpr)` | `items: Vec<StmtId>, expr` |
| InstantiateExpr | `Instantiate` | `expr, items: Vec<ExprId>` |

`Param { name, ty: Option<ExprId>, default_value: Option<ExprId>, kind: ParamKind{Normal,PositionalOnly,KeywordOnly}, status: ParamStatus{Value,Generic,HiddenGeneric}, src }`
(`is_value/is_generic/is_hidden_generic/name_with_stars`). Single-expr child slots are `Option<ExprId>`
(C++ nullable pointers).

## Statement kinds (`StmtKind::<Variant>(<Struct>)`)

| C++ | Rust | fields |
|---|---|---|
| SuiteStmt | `Suite` | `items: Vec<StmtId>` (`flatten`, `add_stmt`, `first_in_block`) |
| BreakStmt / ContinueStmt | `Break` / `Continue` | – |
| ExprStmt | `Expr` | `expr` |
| AssignStmt | `Assign` | `lhs, rhs, type_expr, update: UpdateMode{Assign,Update,UpdateAtomic,ThreadLocalAssign}` |
| DelStmt | `Del` | `expr` |
| PrintStmt | `Print` | `items, no_newline` |
| ReturnStmt / YieldStmt / YieldFromStmt | `Return` / `Yield` / `YieldFrom` | `expr` |
| AssertStmt | `Assert` | `expr, message` |
| WhileStmt | `While` | `cond, suite, else_suite: Option<StmtId>, goto_var: String` |
| ForStmt | `For` | `var, iter, suite, else_suite, decorator, omp_args: Vec<CallArg>, async_, wrapped, flat` |
| IfStmt | `If` | `cond, if_suite, else_suite` |
| MatchStmt | `Match` | `expr, items: Vec<MatchCase{pattern, guard, suite}>` |
| ImportStmt | `Import` | `from, what, as_, dots: usize, args: Vec<Param>, ret, is_function` |
| ExceptStmt | `Except` | `var: String, exc, suite` |
| TryStmt | `Try` | `items: Vec<StmtId /*Except*/>, suite, else_suite, finally` |
| ThrowStmt | `Throw` | `expr, from, transformed` |
| GlobalStmt | `Global` | `var, non_local` |
| FunctionStmt | `Function` | `name, items: Vec<Param>, ret, suite, decorators: Vec<ExprId>, async_`, non-serialized caches `non_inferrable_generics`, `call_layout: Option<Box<CallLayout>>` |
| ClassStmt | `Class` | `name, items: Vec<Param>, suite, decorators, base_classes` |
| WithStmt | `With` | `items: Vec<ExprId>, vars: Vec<String>, suite, async_` |
| CustomStmt | `Custom` | `keyword, expr, suite` |
| DirectiveStmt | `Directive` | `key, value` |
| AssignMemberStmt | `AssignMember` | `lhs, member, rhs, type_expr` |
| CommentStmt | `Comment` | `comment` |

`FunctionStmt::{star_args, kw_star_args, add_param, call_layout}`. Arena helpers: `function_docstr`, `class_docstr`,
`non_inferrable_generics(stmt)`, `has_function_attribute(stmt, name)`, `class_is_record`,
`is_class_var(&Param)`, `contains_id(expr, name)`, `parse_params(Vec<Param>)`,
`new_call(src, f, &[args])`, `call_arg(name, value)`, `new_class`, `new_with_pairs`, `clone_param`,
`first_in_block`, `suite_flatten`, `suite_add_stmt`, `generator_final_slot/gen_slot_get/gen_slot_set`, `generator_loop_count`, `generator_final_expr`,
`generator_set_final_expr`, `generator_set_final_stmt`, `get_static_generic -> LiteralKind`.

## Attributes (`ast::attr`)

`Attr` = enum of every `Attr::X` (discriminants 201.. as C++). `Attrs` is an insertion-ordered
small map `Attr -> AttrValue`:
`AttrValue::{Flag, Int(i64), Str(String), StrList(Vec<String>), KeyValue(BTreeMap<String,String>), Bindings(BindingsAttr)}`.
Methods: `set(a)`, `set_int`, `set_str`, `set_value(a, AttrValue)`, `has`, `get`, `get_int`, `get_str`,
`get_str_list`, `get_key_value`, `get_bindings(_mut)`, `erase`, `iter`.
`Expr`/`Stmt` expose them as `.attrs`. Arena shortcuts: `arena.expr(id).attrs.has(Attr::X)`.

## Printing

* `arena.expr_to_string(id, indent)`, `arena.stmt_to_string(id, indent)` — S-expressions, exact C++ format (`indent=-1` = one line).
* `arena.expr_to_codon(id, attributes, indent, level)`, `stmt_to_codon` — `toCodonString`.
* `arena.expr_to_python(id, attrs, indent, level)`, `stmt_to_python` — `toPythonString`.
* `codon_string::format_float(f64)` — Python `repr(float)`.
* `FormatVisitor::apply_stmt(&arena, id, Option<&dyn FormatCacheHooks>, html)` / `apply_expr` — source-like
  text / HTML. `FormatCacheHooks` (cache lane implements) supplies function/class realizations.

## Visitors (`visitors::visitor`)

```rust
pub trait AstCtx { fn ast(&self) -> &AstArena; fn ast_mut(&mut self) -> &mut AstArena; }
pub trait Visitor: AstCtx {
    fn transform_expr(&mut self, id: ExprId) -> ExprId { self.visit_expr(id); id }   // override for replacing
    fn transform_stmt(&mut self, id: StmtId) -> StmtId { self.visit_stmt(id); id }
    fn visit_expr(&mut self, id: ExprId) { dispatch_expr(self, id) }               // match on tag
    fn visit_stmt(&mut self, id: StmtId) { dispatch_stmt(self, id) }
    fn visit_none(&mut self, id: ExprId) { walk_expr(self, id) } ... one per kind:
    // visit_bool, visit_int, visit_float, visit_string, visit_id, visit_star, visit_keyword_star,
    // visit_tuple, visit_list, visit_set, visit_dict, visit_generator, visit_if_expr, visit_unary,
    // visit_binary, visit_chain_binary, visit_pipe, visit_index, visit_call, visit_dot, visit_slice,
    // visit_ellipsis, visit_lambda, visit_yield_expr, visit_await, visit_assign_expr, visit_range,
    // visit_stmt_expr, visit_instantiate,
    // visit_suite, visit_break, visit_continue, visit_expr_stmt, visit_assign, visit_assign_member,
    // visit_del, visit_print, visit_return, visit_yield, visit_assert, visit_while, visit_for,
    // visit_if, visit_match, visit_import, visit_try, visit_except, visit_global, visit_throw,
    // visit_function, visit_class, visit_yield_from, visit_with, visit_custom, visit_directive,
    // visit_comment
}
pub fn walk_expr<V: Visitor + ?Sized>(v: &mut V, id: ExprId)   // CallbackASTVisitor / Replacing default
pub fn walk_stmt<V: Visitor + ?Sized>(v: &mut V, id: StmtId)
```
Default `visit_*` calls `default_visit_expr/stmt`, which defaults to `walk_*`: walk children in C++
order, calling `transform_*` and **writing back** the returned ids (ReplacingCallbackASTVisitor
semantics; suite slots are re-wrapped with `wrap_suite`). A non-replacing visitor just returns the
same id. For C++ `ASTVisitor` no-op defaults, override `default_visit_expr/stmt` with `{}`.
Like C++, f-string part expressions of `StringExpr` are not visited. `expr_children/stmt_children`
list children; `IdMapper` + `ExprKind::map_ids/StmtKind::map_ids` give generic traversal.
`impl AstCtx for AstArena` exists for standalone visitors.

## Matching (`matching`)

`M` pattern tree: `M::ignore()` (`M_`), `M::any()` (`MAny`), `M::str("x")`, `M::starts/ends/contains`,
`M::Int(i)`, `M::Bool(b)`, `M::Float(f)`, `M::or(vec)`, `M::expr(ExprTag::Id, vec![...])`
(`M<IdExpr>(...)`, empty vec = any node of that kind), `M::stmt(StmtTag::Expr, ...)`,
`.bind_expr(&Rc<Cell<Option<ExprId>>>)` / `.bind_stmt` (`MVar`), `.call_expr(|id| ..)` / `.call_stmt` (`MCall`).
`match_expr(arena, Option<ExprId>, &M) -> bool`, `match_stmt`. Members per node follow C++
`match_members()` order (`expr_members` / `stmt_members`); vector/enum members are opaque (only `M_` matches).
`StringExpr` exposes its simple value as the single member.

## Serialization

All node structs derive `serde::{Serialize, Deserialize}`. `arena.serialize_stmt(root) -> Result<Vec<u8>>` /
`serialize_expr` (postcard, subtree copied into a compact arena), `AstArena::deserialize_stmt(&[u8]) ->
Result<(AstArena, StmtId)>` / `deserialize_expr`, then `arena.import_stmt(&loaded, root)` to merge.
`ty`, `expected_type`, `PipeExpr::in_types`, `AwaitExpr::transformed`, FunctionStmt caches are not
serialized (as in C++). `orig_expr` is dropped by `import_*`.

## C++ -> Rust name table

| C++ | Rust |
|---|---|
| `cache->N<T>(args)` / `NS<T>(src, args)` | `arena.new_expr(T{..}, src)` / `new_stmt` |
| `ast::clone(e, clean)` | `arena.clone_expr(e, clean)` |
| `cast<CallExpr>(e)` | `arena[e].kind.as_call()` |
| `e->getType()/setType` | `arena[e].ty` |
| `e->isDone()/setDone()` | `arena[e].done` |
| `getOrigExpr/setOrigExpr` | `arena[e].orig_expr` |
| `getExpectedType/setExpectedType` | `arena[e].expected_type` |
| `e->toString(indent)` | `arena.expr_to_string(e, indent)` |
| `toCodonString(a, i, l)` | `arena.expr_to_codon(e, a, i, l)` |
| `toPythonString(a, i, l)` | `arena.expr_to_python(e, a, i, l)` |
| `setAttribute(Attr::X)` | `arena[e].attrs.set(Attr::X)` |
| `getAttribute<ir::StringValueAttribute>(k)->value` | `attrs.get_str(k)` |
| `SuiteStmt::wrap(s)` | `arena.wrap_suite(s)` |
| `stmt->firstInBlock()` | `arena.first_in_block(s)` |
| `SuiteStmt::flatten/addStmt` | `arena.suite_flatten(s)/suite_add_stmt(s, x)` |
| `ASTVisitor::visit(XExpr*)` | `Visitor::visit_x(&mut self, ExprId)` |
| `CallbackASTVisitor::transform(e)` | `Visitor::transform_expr` |
| `isId(e, s)` | `is_id(arena, e, s)` |
| `getStaticGeneric(e)` | `arena.get_static_generic(e)` |
| `types::LiteralKind` | `ast::LiteralKind` (re-export from types lane later) |
