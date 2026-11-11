# typhon-parser type checker API (sema-core lane)

Port of `codon/parser/ast/types/**`, `cache.{h,cpp}`, `ctx.h`, `visitors/typecheck/**`.
Contract for the phase-3 typecheck lanes, translate, scoping and the compiler driver.

## Modules

| Rust | C++ |
|---|---|
| `types` (`TypeArena`, `TypeKind`, `LinkType`, `ClassType`, `Generic`, `FuncData`, `TraitKind`, `Unification`, `TypeHost`) | `ast/types/*` |
| `ctx` (`Context<I>`, `BindingCheckpoint`) | `parser/ctx.h` |
| `cache` (`Cache`, `Class`, `ClassField`, `ClassRealization`, `Function`, `FunctionRealization`, `ImportModule`, `CacheHooks`, `CacheRealizer`, `AstType`) | `parser/cache.{h,cpp}` |
| `typecheck::ctx` (`TypeContext`, `TypecheckItem`, `Base`, `Loop`, `CtxRef`, `ItemRef`) | `typecheck/ctx.{h,cpp}` |
| `typecheck` (`TypecheckVisitor`, `PartialCallData`, `CallArgSlots`, `WrapFn`, `AstNode`) | `typecheck/typecheck.h` |
| `typecheck::typecheck` | `typecheck.cpp`: `apply`, `apply_ctx`, `load_std_library`, `transform*`, dispatch, Suite/ExprStmt/StmtExpr/Custom/Comment/Directive |
| `typecheck::helpers` | the remaining `typecheck.cpp` helpers (extract*, find*, canCall, matchCallable, wrapExpr, reorderNamedArgs, instantiate*, getters, cythonize*) |
| `typecheck::infer` | `infer.cpp` |
| `typecheck::basic` | `basic.cpp` (ported) |
| `typecheck::{access,assign,call,class,collections,cond,error,function,import,loops,op,special}` | same-named `.cpp` (STUBS, phase-3 lanes) |

## Types (`TypeArena`)

* `TypeId` = slot in `Cache::types: Rc<TypeArena>`. Never freed. C++ `shared_from_this()` == same id;
  `make_shared<X>` == `types.new_*` (`new_unbound`, `new_link`, `new_class`, `new_class_from`, `copy_class`,
  `new_func(base, ast, func_generics, parent)`, `new_int_static/new_str_static/new_bool_static`, `new_union`,
  `new_callable_trait`, `new_type_trait`, `new_link_type(LinkType{..}, src)`).
* All arena methods take `&self` (interior `RefCell`, borrows held only per slot access). Raw access:
  `with/with_mut/link/with_link_mut/class/with_class/with_class_mut`.
* Virtual getters (follow links): `follow`, `get_link`, `get_unbound`, `get_class` (applies canonical Optional),
  `get_func`, `get_partial`, `get_static`, `get_int_static -> Option<i64>`, `get_str_static`, `get_bool_static`,
  `get_union`, `is(t, name)`, `class_name`, `static_kind`, `generics(t)`, `generic_type(t, i)`, `is_record`.
* Virtual ops: `unify(a, b, Option<&mut Unification>, &mut dyn TypeHost) -> Result<i32>`, `unify_or_undo`
  (`operator<<`), `generalize(t, level)`, `instantiate(t, level, Option<&mut InstantiationCache>)`,
  `has_unbounds`, `get_unbounds`, `can_realize`, `is_instantiated`, `debug_string(t, mode)`,
  `pretty_string`, `to_string` (mode 2), `realized_name`, `non_static_type`, `freeze`.
* FuncType: `func_args/func_arg(i)/func_size/func_ret/func_generics/func_parent/func_ast_id/func_name`,
  `replace_arg_types`. Partial: `partial_func/partial_mask/is_partial_virtual/is_partial_empty`.
  Union: `union_realization_types`. Static: `static_expr(t, &mut AstArena)`.
* `FuncType` needs name/param info from its `FunctionStmt`: whoever creates/realizes a function AST that a
  FuncType points at MUST call `types.register_func_ast(stmt, FuncAstInfo{name, params:(name,status), allow_pass_through})`
  (function.cpp / infer.rs do this; re-register after setting `Attr::AllowPassThrough`).
* `TypeHost` = callbacks C++ types make into a fresh `TypecheckVisitor` (CallableTrait -> `matchCallable`,
  `__NTuple__` unify -> tuple instantiation). `TypecheckVisitor` implements it; pass `self`. `NoHost` for tests.
* `Unification::undo(&mut self, &TypeArena)` reverts linked/leveled/traits/statics.

## Speculative typechecking / errors

* C++ `throw ParserException` -> `Err(CompilerError::Parser(..))`; use `typhon_common::bail_e!(ErrorKind::X, src, args..)`.
* `try { .. } catch (exc::ParserException &e) { .. }` -> `match r { Err(e) if e.parser_errors().is_some() => .., }`.
* Speculative unification: create `Unification::new()`, pass `Some(&mut u)`, call `u.undo(&types)` on failure.
* `TypeContext` checkpoints: `ctx.base.push_checkpoint(wants_state)`, `ctx.prepare_checkpoint()`, on rollback
  `let cp = ctx.base.pop_checkpoint(); ctx.base.rollback(&cp, preserve)` and, if `cp.state` holds a
  `SavedInferenceState`, `ctx.restore_inference_state(state)`.

## Cache, AST allocation

* `TypecheckVisitor<'a> { cache: &'a mut Cache, ctx: CtxRef, prepend_stmts, preamble, result_expr, result_stmt, src }`.
  Nested visitor: `self.nested(share_prepend)` / `self.with_ctx(other)`; `TypecheckVisitor::new(cache, ctx, preamble, stmts)`.
* `N<T>(..)` -> `self.n_expr(XExpr{..})` / `self.n_stmt(XStmt{..})` (visitor src; stmt gets `Attr::ExprTime`).
  `NC<T>` -> `nc_expr/nc_stmt`. AST lives in `self.cache.ast` (`AstArena`, see AST_API.md).
* `expr->getType()` -> `self.ety(e)` (`expr_type` for Option), `setType` -> `set_type`, `setDone` -> `set_done_expr/stmt`,
  `getClassType` -> `expr_class`.
* Visit rules: `visit_<kind>(&mut self, id) -> Result<()>`; set `self.result_expr`/`self.result_stmt` for replacements
  (same as C++ `resultExpr`). Call `self.transform(Some(e))`/`self.tr(e)` and `self.transform_stmt_opt`/`tr_stmt`
  for children and WRITE THE RESULT BACK into the parent node.
* Never hold a `ctx.borrow()`/`borrow_mut()` across a call into the visitor. Copy data out first.
* Context lookup: `self.find(name, time) -> Option<ItemRef>`, `self.force_find(name)`,
  `TypeContext::add_var/add_type/add_func/add_always_visible(cache, &ctx, ..)` (static fns: they need the cache).
* `Cache` tables: `classes: IndexMap<String, Class>`, `functions: IndexMap<String, Function>`, `overloads`,
  realizations are `Rc<RefCell<..Realization>>`. `cache.rev(name)`, `cache.set_reverse(k, v)`.

## IR

* `Cache::module: Option<Rc<RefCell<typhon_ir::Module>>>`. IR types/funcs carry `AstType::to_ref(type_id)` as `ast_type`.
* `CacheRealizer(Rc<RefCell<Cache>>)` implements `typhon_ir::TypeRealizer`; install with `module.set_realizer(..)`.
* `make_ir_type(class) -> typhon_ir::TypeId`, `make_ir_function(&FunctionRealizationRef) -> FuncId` (infer.rs).
* Translation (later lane) plugs in via `Cache::hooks.translate` / `translate_parsed`; parsing via
  `hooks.parse_code/parse_file`; scoping via `hooks.scoping` (all `Option<Rc<dyn Fn(&mut Cache, ..)>>`).
  `Cache::codegen_ctx: Option<Rc<dyn Any>>` is owned by translate.

## Method -> file ownership

Every method of `typecheck.h` is a method of `TypecheckVisitor` in the module of the `.cpp` that defines it.
Stubs are marked `// STUB(<file>.cpp)`; replace bodies, keep signatures. Grep `STUB(` for the list.
C++ overloads get distinct names: `getImport(vector)` -> `get_import_chain`; `findBestMethod` ->
`find_best_method_types/_exprs/_named`; `instantiateType(src,t,cls)` -> `instantiate_type(t, Option<cls>)`,
`instantiateType(t, vector)` -> `instantiate_type_with(t, &[..])`; `instantiateStatic` -> `types.new_*_static`;
`extractType(Expr|Type|string)` -> `extract_type_expr/extract_type/extract_type_name` (same for `extract_class_type*`);
`getClass(string|Type)` -> `get_class_name/get_class_type`; `getFunction` -> `get_function_name/get_function_type`.
