# typhon-dsl API

Port of `codon/dsl/{dsl.h,plugins.h,plugins.cpp}`.

```rust
pub struct Info { name, description, version, url, supported, stdlib_path, dylib_path,
                  link_args: Vec<String>, compiler_dylib_path: Option<String>,
                  runtime_dylib_path: Option<String> }
impl Info { fn compiler_dylib_path(&self) -> &str; fn runtime_dylib_path(&self) -> &str; }

pub type KeywordCallback = typhon_parser::cache::CustomStmtFn;   // same type as Cache stores
pub struct ExprKeyword  { keyword: String, callback: KeywordCallback }
pub struct BlockKeyword { keyword: String, callback: KeywordCallback, has_expr: bool }

pub trait Dsl {                     // all methods default to no-ops
    fn add_ir_passes(&self, pm: &mut typhon_ir_transform::manager::PassManager, debug: bool);
    fn add_llvm_passes(&self, stage: PassStage, module: &Module, debug: bool) -> Result<()>;
    fn pipeline_text(&self, stage: PassStage, debug: bool) -> Option<String>;
    fn expr_keywords(&self) -> Vec<ExprKeyword>;
    fn block_keywords(&self) -> Vec<BlockKeyword>;
}
pub struct DefaultDsl;              // used for stdlib-only plugins

pub struct Plugin { pub dsl: Box<dyn Dsl>, pub info: Info, /* lib handle */ }
impl Plugin { fn has_library(&self) -> bool }
impl typhon_llvm::LlvmPassHooks for Plugin   // forwards to dsl

pub struct PluginManager;
impl PluginManager {
    fn new(argv0) -> Self;
    fn load(&mut self, dir: &str) -> Result<&mut Plugin>;   // CompilerError::Plugin on failure
    fn load_runtime_libraries(&mut self) -> Result<()>;
    fn iter(&self) / iter_mut(&mut self) / len / is_empty;
    fn llvm_hooks(&self) -> Vec<&dyn LlvmPassHooks>;        // pass to typhon_llvm::optimize
}
pub fn parse_info(&toml::Table, root: &Path) -> Result<Info>;
pub struct VersionRange;            // npm-style ranges with `||`
```

## Driver integration (C++ `Compiler` equivalents)
- After `load`: if `info.stdlib_path` non-empty, add it to the parser's plugin import paths.
- Register keywords: for each `ExprKeyword` insert into `cache.custom_expr_stmts[keyword] = callback`;
  for `BlockKeyword` insert `cache.custom_block_stmts[keyword] = (has_expr, callback)`.
- IR: `for p in pm.iter() { p.dsl.add_ir_passes(&mut pass_manager, debug) }`.
- LLVM: `typhon_llvm::optimize(module, opts, &plugins.llvm_hooks())`.
- Linking: use `info.link_args`, else `info.runtime_dylib_path()` if non-empty.
- JIT: call `load_runtime_libraries()` before execution.
