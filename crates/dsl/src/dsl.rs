//! Port of `codon/dsl/dsl.h`: the `DSL` base class.

use inkwell::module::Module;
use typhon_common::Result;
use typhon_ir_transform::manager::PassManager;
use typhon_llvm::PassStage;
use typhon_llvm::inkwell;
pub use typhon_parser::cache::CustomStmtFn;

/// C++ `DSL::Info`: general information about a plugin.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Info {
    /// Extension name.
    pub name: String,
    /// Extension description.
    pub description: String,
    /// Extension version.
    pub version: String,
    /// Extension URL.
    pub url: String,
    /// Supported Typhon/Codon versions (semver range).
    pub supported: String,
    /// Plugin stdlib path (empty if none).
    pub stdlib_path: String,
    /// Legacy library path (`library.cpp`), used for both compiler and runtime by default.
    pub dylib_path: String,
    /// Linker arguments (replace the default runtime library argument if present).
    pub link_args: Vec<String>,
    /// Compiler library: `None` inherits `dylib_path`; empty disables loading.
    pub compiler_dylib_path: Option<String>,
    /// Runtime library: `None` inherits `dylib_path`; empty disables default linking.
    pub runtime_dylib_path: Option<String>,
}

impl Info {
    /// C++ `getCompilerDylibPath()`.
    pub fn compiler_dylib_path(&self) -> &str {
        self.compiler_dylib_path.as_deref().unwrap_or(&self.dylib_path)
    }
    /// C++ `getRuntimeDylibPath()`.
    pub fn runtime_dylib_path(&self) -> &str {
        self.runtime_dylib_path.as_deref().unwrap_or(&self.dylib_path)
    }
}

/// C++ `DSL::KeywordCallback` (`Stmt *(TypecheckVisitor *, CustomStmt *)`).
/// Same type the parser stores in `Cache::custom_{block,expr}_stmts`.
pub type KeywordCallback = CustomStmtFn;

/// C++ `DSL::ExprKeyword`: `keyword <expr>`.
#[derive(Clone)]
pub struct ExprKeyword {
    pub keyword: String,
    pub callback: KeywordCallback,
}

/// C++ `DSL::BlockKeyword`: `keyword <expr>: <block>`.
#[derive(Clone)]
pub struct BlockKeyword {
    pub keyword: String,
    pub callback: KeywordCallback,
    pub has_expr: bool,
}

/// C++ `class DSL`. All hooks default to no-ops.
pub trait Dsl {
    /// Registers this DSL's IR passes with the given pass manager.
    fn add_ir_passes(&self, _pm: &mut PassManager, _debug: bool) {}

    /// C++ `addLLVMPasses(PassBuilder*, debug)`.
    // PORT-NOTE: there is no PassBuilder callback registration from Rust; LLVM hooks are
    // expressed as `typhon_llvm::LlvmPassHooks` (called per pipeline stage on the module,
    // plus optional textual new-PM pipelines). `Plugin` forwards these.
    fn add_llvm_passes(&self, _stage: PassStage, _module: &Module, _debug: bool) -> Result<()> {
        Ok(())
    }
    /// Optional textual new-PM pipeline appended at `stage`.
    fn pipeline_text(&self, _stage: PassStage, _debug: bool) -> Option<String> {
        None
    }

    /// Expression keywords (`keyword <expr>`).
    fn expr_keywords(&self) -> Vec<ExprKeyword> {
        Vec::new()
    }
    /// Block keywords (`keyword <expr>: <block>`).
    fn block_keywords(&self) -> Vec<BlockKeyword> {
        Vec::new()
    }
}

/// C++ `std::make_unique<DSL>()` used for stdlib-only plugins.
#[derive(Debug, Default, Clone, Copy)]
pub struct DefaultDsl;
impl Dsl for DefaultDsl {}
