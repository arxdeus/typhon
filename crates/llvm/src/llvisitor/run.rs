//! `LLVMVisitor::run`: JIT-execute the module through the ORC engine.

use std::ffi::CString;

use typhon_common::{CompilerError, Result};

use super::LlvmVisitor;
use crate::engine::{Engine, EngineConfig};

impl<'ctx> LlvmVisitor<'ctx> {
    /// `run(args, libs)`: optimize, load `libs`, JIT the module and call `main`.
    /// Returns `main`'s exit status.
    ///
    /// PORT-NOTE: C++ builds a dedicated LLJIT with a GC-aware JITLink memory manager;
    /// the Rust port reuses `typhon_llvm::Engine` (same memory manager / debug
    /// listener). Uncaught exceptions are reported by the runtime (`seq_terminate`).
    pub fn run(&mut self, args: &[String], libs: &[String], config: EngineConfig) -> Result<i32> {
        self.run_llvm_pipeline_pub()?;
        for lib in libs {
            let c = CString::new(lib.as_str()).map_err(|e| CompilerError::Io(e.to_string()))?;
            // SAFETY: FFI; LLVM keeps the library loaded for the process lifetime.
            let failed = unsafe { llvm_sys::support::LLVMLoadLibraryPermanently(c.as_ptr()) };
            if failed != 0 {
                return Err(CompilerError::Io(format!("could not load library {lib}")));
            }
        }
        let mut opts = self.options.clone();
        opts.jit = true;
        let engine = Engine::new(&opts, EngineConfig { optimize_on_add: false, ..config })?;
        if !self.jit_symbols.is_empty() {
            let syms: Vec<(&str, u64)> = self.jit_symbols.iter().map(|(n, a)| (n.as_str(), *a)).collect();
            engine.define_absolute(&syms)?;
        }
        let module = self.module.take().expect("no module");
        engine.add_module(&module)?;
        self.clear_llvm_data();
        drop(module);
        let main = engine.lookup("main")?;
        // argv: NUL-terminated C strings kept alive for the call.
        let cargs: Vec<CString> = args.iter().map(|a| CString::new(a.as_str()).unwrap_or_default()).collect();
        let mut ptrs: Vec<*const libc::c_char> = cargs.iter().map(|c| c.as_ptr()).collect();
        ptrs.push(std::ptr::null());
        // SAFETY: `main` is the JIT'd canonical `i32 main(i32, ptr)`.
        let f: extern "C" fn(i32, *const *const libc::c_char) -> i32 = unsafe { std::mem::transmute(main as usize) };
        Ok(f(args.len() as i32, ptrs.as_ptr()))
    }

    pub(super) fn run_llvm_pipeline_pub(&mut self) -> Result<()> {
        self.prepare_win_exports();
        if let Some(db) = &self.db.builder {
            db.finalize();
        }
        crate::optimize::optimize(self.module.as_ref().expect("no module"), &self.options)
    }
}
