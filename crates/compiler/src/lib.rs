//! typhon-compiler: port of `codon/compiler/compiler.{h,cpp}`.
//!
//! [`Compiler`] owns the parser [`Cache`], the IR [`Module`], the IR [`PassManager`]
//! and the LLVM codegen entry point. LLVM codegen needs an `inkwell::context::Context`
//! that outlives the visitor, so instead of owning an `LLVMVisitor` (C++), the
//! compiler hands out one bound to a caller-provided context via [`Compiler::codegen`].

pub mod hooks;
pub mod stdlib;
pub mod test_passes;

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use typhon_common::config::{VERSION_MAJOR, VERSION_MINOR, VERSION_PATCH};
use typhon_common::fs::FileSystem;
use typhon_common::{CompilerError, LoggerFlags, Options, Result, SrcInfo, Timer, logger};
use typhon_ir::{Module, NodeRef};
use typhon_ir_transform::manager::PassManager;
use typhon_llvm::inkwell::context::Context;
use typhon_llvm::LlvmVisitor;
use typhon_parser::cache::{Cache, CacheRealizer};
use typhon_parser::typecheck::TypecheckVisitor;

pub use typhon_llvm;

/// Port of `codon::Compiler`.
pub struct Compiler {
    input: String,
    options: Options,
    cache: Rc<RefCell<Cache>>,
    module: Rc<RefCell<Module>>,
    pm: PassManager,
}

impl Compiler {
    /// `Compiler(options, fs)`; `fs = None` uses the default disk/embedded filesystem.
    pub fn new(options: Options, fs: Option<Rc<RefCell<dyn FileSystem>>>) -> Self {
        let fs = fs.unwrap_or_else(|| stdlib::default_filesystem(&options.argv0));
        let mut cache = Cache::new(Some(fs));
        cache.options = Some(Rc::new(options.clone()));
        hooks::install(&mut cache);
        let module = Rc::new(RefCell::new(Module::new("")));
        cache.module = Some(module.clone());
        let cache = Rc::new(RefCell::new(cache));
        module.borrow_mut().set_realizer(Some(Rc::new(RefCell::new(CacheRealizer(cache.clone())))));
        let pm = PassManager::new(options.clone());
        Compiler {
            input: String::new(),
            options,
            cache,
            module,
            pm,
        }
    }

    pub fn input(&self) -> &str {
        &self.input
    }
    pub fn options(&self) -> &Options {
        &self.options
    }
    pub fn cache(&self) -> &Rc<RefCell<Cache>> {
        &self.cache
    }
    pub fn module(&self) -> &Rc<RefCell<Module>> {
        &self.module
    }
    pub fn pass_manager(&mut self) -> &mut PassManager {
        &mut self.pm
    }
    /// `parse(isCode, file, code, startLine, testFlags, defines)`.
    pub fn parse(
        &mut self,
        is_code: bool,
        file: &str,
        code: &str,
        start_line: i32,
        test_flags: i32,
        defines: &HashMap<String, String>,
    ) -> Result<()> {
        self.input = file.to_string();
        let mut cache = self.cache.borrow_mut();
        let fs = cache.fs.clone().ok_or_else(|| CompilerError::Io("no filesystem".into()))?;
        let abspath = if file != "-" {
            fs.borrow().canonical(std::path::Path::new(file)).to_string_lossy().replace('\\', "/")
        } else {
            file.to_string()
        };
        let node = if is_code {
            let f = cache.hooks.parse_code.clone().ok_or_else(|| CompilerError::Io("parser not available".into()))?;
            f(&mut cache, &abspath, code, start_line)?
        } else {
            let f = cache.hooks.parse_file.clone().ok_or_else(|| CompilerError::Io("parser not available".into()))?;
            f(&mut cache, &abspath)?
        };
        fs.borrow_mut().set_module0(file);

        let mut t2 = Timer::new("typecheck");
        let early = self.early_defines();
        let typechecked = TypecheckVisitor::apply(&mut cache, node, &abspath, defines, &early, test_flags > 1)?;
        t2.log();

        if logger().flags.contains(LoggerFlags::USER) {
            let s = cache.ast.stmt_to_string(typechecked, 0);
            let _ = std::fs::write("_dump_typecheck.sexp", format!("{s}\n"));
        }

        let mut t4 = Timer::new("translate");
        let tr = cache.hooks.translate_apply.clone().ok_or_else(|| CompilerError::Io("translator not available".into()))?;
        tr(&mut cache, typechecked)?;
        t4.log();
        drop(cache);
        self.module.borrow_mut().set_src_info(NodeRef::Module, SrcInfo::new(&abspath, 0, 0, 0));
        if logger().flags.contains(LoggerFlags::USER) {
            let s = typhon_ir::util::format::format(&self.module.borrow(), NodeRef::Module);
            let _ = std::fs::write("_dump_ir.sexp", format!("{s}\n"));
        }
        Ok(())
    }

    /// `parseFile(file, testFlags, defines)`.
    pub fn parse_file(&mut self, file: &str, test_flags: i32, defines: &HashMap<String, String>) -> Result<()> {
        self.parse(false, file, "", 0, test_flags, defines)
    }

    /// `parseCode(file, code, startLine, testFlags, defines)`.
    pub fn parse_code(&mut self, file: &str, code: &str, start_line: i32, test_flags: i32, defines: &HashMap<String, String>) -> Result<()> {
        self.parse(true, file, code, start_line, test_flags, defines)
    }

    /// First half of `compile()`: run the IR pass pipeline.
    pub fn run_passes(&mut self) {
        let mut m = self.module.borrow_mut();
        self.pm.run(&mut m);
        if logger().flags.contains(LoggerFlags::USER) {
            let s = typhon_ir::util::format::format(&m, NodeRef::Module);
            let _ = std::fs::write("_dump_ir_opt.sexp", format!("{s}\n"));
        }
    }

    /// Second half of `compile()`: lower the IR module to LLVM in `ctx`.
    pub fn codegen<'ctx>(&self, ctx: &'ctx Context) -> Result<LlvmVisitor<'ctx>> {
        let mut v = LlvmVisitor::new(ctx, self.options.clone());
        v.visit_module(&self.module.borrow())?;
        if logger().flags.contains(LoggerFlags::USER)
            && let Some(m) = v.module()
        {
            let _ = std::fs::write("_dump_llvm.ll", m.print_to_string().to_string());
        }
        Ok(v)
    }

    /// `compile()`: IR passes + LLVM codegen.
    pub fn compile<'ctx>(&mut self, ctx: &'ctx Context) -> Result<LlvmVisitor<'ctx>> {
        self.run_passes();
        self.codegen(ctx)
    }

    /// `docgen(files)`.
    pub fn docgen(&self, files: &[String]) -> Result<String> {
        hooks::docgen(&self.options.argv0, files)
    }

    /// `getEarlyDefines()`.
    pub fn early_defines(&self) -> HashMap<String, String> {
        let o = &self.options;
        let b = |v: bool| if v { "1" } else { "0" }.to_string();
        HashMap::from([
            ("__debug__".into(), b(o.debug)),
            ("__py_numerics__".into(), b(o.pynum)),
            ("__py_extension__".into(), b(o.pyext)),
            ("__dict_unordered__".into(), b(o.unordereddict)),
            ("__typhon_version_major__".into(), VERSION_MAJOR.to_string()),
            ("__typhon_version_minor__".into(), VERSION_MINOR.to_string()),
            ("__typhon_version_micro__".into(), VERSION_PATCH.to_string()),
            ("__apple__".into(), b(cfg!(target_os = "macos"))),
            ("__windows__".into(), b(cfg!(windows))),
        ])
    }
}
