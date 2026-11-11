//! Output paths: object/bitcode/LL/executable writers, `compile`, `run`
//! (port of llvisitor.cpp `writeTo*`, `setupGlobalCtor`, `compile`, `run`).

use std::path::{Path, PathBuf};
use std::process::Command;

use inkwell::targets::FileType;
use llvm_sys::LLVMLinkage::*;
use typhon_common::options::GlobalCtorMode;
use typhon_common::{CompilerError, Result};

use super::ll::{self, Builder};
use super::{LlvmVisitor, MAIN_CTOR, MAIN_UNCLASH};

/// Link information of a loaded DSL plugin (`plugin->info`).
#[derive(Clone, Debug, Default)]
pub struct PluginLinkInfo {
    /// Runtime dylib path (`getRuntimeDylibPath()`), may be empty.
    pub runtime_dylib: String,
    /// Explicit link arguments (`linkArgs`).
    pub link_args: Vec<String>,
}

fn io_err(e: impl std::fmt::Display) -> CompilerError {
    CompilerError::Io(e.to_string())
}

/// `executeCommand`.
fn execute_command(args: &[String]) -> Result<()> {
    if std::env::var_os("TYPHON_LOG_COMMANDS").is_some() {
        eprintln!("Executing '{}'", args.join(" "));
    }
    let status = Command::new(&args[0])
        .args(&args[1..])
        .status()
        .map_err(|e| CompilerError::Io(format!("process for '{}' could not be started: {e}", args[0])))?;
    if !status.success() {
        return Err(CompilerError::Io(format!(
            "process for '{}' exited with status {}",
            args[0],
            status.code().unwrap_or(-1)
        )));
    }
    Ok(())
}

/// Directory holding `libcodonrt`: `TYPHON_LIB_DIR`, else next to/around the executable.
pub fn runtime_lib_dirs(argv0: &str) -> Vec<String> {
    if let Ok(d) = std::env::var("TYPHON_LIB_DIR")
        && !d.is_empty()
    {
        return vec![d];
    }
    let exe = if argv0.is_empty() {
        std::env::current_exe().unwrap_or_default()
    } else {
        let p = PathBuf::from(argv0);
        std::fs::canonicalize(&p).or_else(|_| std::env::current_exe()).unwrap_or(p)
    };
    let base = exe.parent().map(Path::to_path_buf).unwrap_or_default();
    let mut rpaths = Vec::new();
    for rel in ["../lib", "../lib/codon"] {
        let p = base.join(rel);
        if p.exists()
            && let Ok(c) = std::fs::canonicalize(&p)
        {
            rpaths.push(c.to_string_lossy().into_owned());
        }
    }
    if rpaths.is_empty() {
        rpaths.push(base.to_string_lossy().into_owned());
    }
    rpaths
}

impl<'ctx> LlvmVisitor<'ctx> {
    /// `runLLVMPipeline`.
    fn run_llvm_pipeline(&mut self) -> Result<()> {
        self.prepare_win_exports();
        if let Some(db) = &self.db.builder {
            db.finalize();
        }
        let hooks: Vec<&dyn crate::optimize::LlvmPassHooks> = self.plugins.iter().map(|p| p.as_ref()).collect();
        crate::optimize::optimize(self.module.as_ref().expect("no module"), &self.options, &hooks)
    }

    /// `dump`.
    pub fn dump(&mut self, filename: &str) -> Result<()> {
        self.write_to_ll_file(filename, false)
    }

    /// `setupGlobalCtor`.
    fn setup_global_ctor(&mut self) {
        let mm = self.m();
        if ll::get_named_value(mm, "llvm.global_ctors").is_some() {
            return;
        }
        let main = match ll::get_function(mm, MAIN_UNCLASH) {
            Some(f) => f,
            None => {
                let Some(f) = ll::get_function(mm, "main") else { return };
                ll::set_name(f, MAIN_UNCLASH); // avoid clash with other main
                ll::set_linkage(f, LLVMPrivateLinkage);
                f
            }
        };
        let b = Builder::new(self.c);
        let ctor_ty = ll::fn_ty(b.void(), &[], false);
        let entry_ty = ll::struct_ty(self.c, &[b.i32(), b.ptr(), b.ptr()]);
        let ctor = ll::get_or_insert_function(mm, MAIN_CTOR, ctor_ty);
        ll::set_linkage(ctor, LLVMPrivateLinkage);
        b.at_end(ll::append_block(self.c, ctor, "entry"));
        b.call(ll::function_type(main), main, &[b.c_i32(0), b.null_ptr()]);
        b.ret_void();
        let entry = ll::const_struct(self.c, &[b.c_i32(65535), ctor, b.null_ptr()]);
        let arr_ty = ll::array_ty(entry_ty, 1);
        ll::new_global(mm, arr_ty, true, LLVMAppendingLinkage, Some(ll::const_array(entry_ty, &[entry])), "llvm.global_ctors");
    }

    /// `writeToObjectFile` (`assembly` writes a `.s` file instead).
    pub fn write_to_object_file(&mut self, filename: &str, pic: bool, assembly: bool) -> Result<()> {
        if self.options.ctor == GlobalCtorMode::Yes {
            self.setup_global_ctor();
        }
        self.run_llvm_pipeline()?;
        let module = self.module.as_ref().expect("no module");
        let tm = crate::target::target_machine_for_module(module, &self.options, false, pic)?
            .ok_or_else(|| CompilerError::Llvm("could not create target machine".into()))?;
        let ft = if assembly { FileType::Assembly } else { FileType::Object };
        tm.write_to_file(module, ft, Path::new(filename)).map_err(|e| CompilerError::Llvm(e.to_string()))
    }

    /// `writeToBitcodeFile`.
    pub fn write_to_bitcode_file(&mut self, filename: &str) -> Result<()> {
        self.run_llvm_pipeline()?;
        if !self.module.as_ref().unwrap().write_bitcode_to_path(filename) {
            return Err(io_err(format!("could not write {filename}")));
        }
        Ok(())
    }

    /// `writeToLLFile`.
    pub fn write_to_ll_file(&mut self, filename: &str, optimize: bool) -> Result<()> {
        if self.options.ctor == GlobalCtorMode::Yes {
            self.setup_global_ctor();
        }
        if optimize {
            self.run_llvm_pipeline()?;
        }
        self.module.as_ref().unwrap().print_to_file(filename).map_err(|e| io_err(e.to_string()))
    }

    /// `writeToExecutable`: object file + system linker invocation.
    pub fn write_to_executable(&mut self, filename: &str, argv0: &str, library: bool, libs: &[String], lflags: &str) -> Result<()> {
        let mut user_flags: Vec<String> = lflags.split(' ').filter(|s| !s.is_empty()).map(String::from).collect();
        let mut pie = false;
        for f in user_flags.iter_mut() {
            if f == "--pie" {
                *f = "-pie".into();
            }
            if f == "-pie" {
                pie = true;
            } else if f == "-no-pie" {
                pie = false;
            }
        }
        if library && pie {
            return Err(CompilerError::Llvm("PIE is only supported for executables".into()));
        }
        if library && self.options.ctor != GlobalCtorMode::No {
            self.setup_global_ctor();
        }
        let obj_file = format!("{filename}.o");
        self.write_to_object_file(&obj_file, library || pie, false)?;
        let rpaths = runtime_lib_dirs(argv0);

        // PORT-NOTE: Windows (clang + lld + MSVC CRT) command line is cfg-gated.
        #[cfg(windows)]
        let mut command: Vec<String> =
            ["clang", "-fuse-ld=lld", "-fms-runtime-lib=dll", "-Wl,/nodefaultlib:libcmt", "-lmsvcrt", "-lvcruntime", "-lucrt"]
                .iter()
                .map(|s| s.to_string())
                .collect();
        #[cfg(not(windows))]
        let mut command: Vec<String> = vec![std::env::var("TYPHON_LINKER").unwrap_or_else(|_| "c++".into())];
        command.push("-Wno-unused-command-line-argument".into());
        command.push(obj_file.clone()); // MUST go before -llib on Linux
        if library {
            command.push("-shared".into());
        }
        let add_rpath = |cmd: &mut Vec<String>, r: &str| {
            if !r.is_empty() {
                cmd.push(format!("-L{r}"));
                #[cfg(not(windows))]
                cmd.push(format!("-Wl,-rpath,{r}"));
            }
        };
        for r in &rpaths {
            add_rpath(&mut command, r);
        }
        for p in &self.plugin_libs {
            if p.runtime_dylib.is_empty() {
                continue;
            }
            let dir = Path::new(&p.runtime_dylib).parent().map(|d| std::fs::canonicalize(d).unwrap_or(d.to_path_buf()));
            if let Some(d) = dir {
                add_rpath(&mut command, &d.to_string_lossy());
            }
        }
        for l in libs {
            command.push(format!("-l{l}"));
        }
        for p in &self.plugin_libs {
            if p.link_args.is_empty() {
                if p.runtime_dylib.is_empty() {
                    continue;
                }
                let stem = Path::new(&p.runtime_dylib).file_stem().unwrap_or_default().to_string_lossy().into_owned();
                command.push(format!("-l{}", stem.strip_prefix("lib").unwrap_or(&stem)));
            } else {
                command.extend(p.link_args.iter().cloned());
            }
        }
        #[cfg(windows)]
        let extra = ["-lcodonrt", "-llibomp", "-llibopenblas", "-o", filename];
        #[cfg(not(windows))]
        let extra = ["-lcodonrt", "-lomp", "-lpthread", "-ldl", "-lz", "-lm", "-lc", "-o", filename];
        command.extend(extra.iter().map(|s| s.to_string()));
        command.extend(user_flags);
        #[cfg(not(windows))]
        if !library && !cfg!(target_os = "macos") {
            command.push(if pie { "-pie" } else { "-no-pie" }.into());
        }
        // PORT-NOTE: Apple ld rejects `-no-pie`; on macOS it is omitted.
        let r = execute_command(&command);
        #[cfg(target_os = "macos")]
        if r.is_ok() && self.options.debug {
            let _ = execute_command(&["dsymutil".to_string(), filename.to_string()]);
        }
        let _ = std::fs::remove_file(&obj_file);
        r
    }

    /// `compile`: output kind chosen by file extension.
    pub fn compile(&mut self, filename: &str, argv0: &str, libs: &[String], lflags: &str) -> Result<()> {
        let f = filename;
        let lower = f.to_lowercase();
        if f.ends_with(".ll") {
            self.write_to_ll_file(f, true)
        } else if f.ends_with(".bc") {
            self.write_to_bitcode_file(f)
        } else if f.ends_with(".o") || f.ends_with(".obj") {
            self.write_to_object_file(f, false, false)
        } else if f.ends_with(".s") || f.ends_with(".S") || f.ends_with(".asm") {
            self.write_to_object_file(f, false, true)
        } else if f.ends_with(".so") || f.ends_with(".dylib") || lower.ends_with(".dll") {
            self.write_to_executable(f, argv0, true, libs, lflags)
        } else {
            self.write_to_executable(f, argv0, false, libs, lflags)
        }
    }
}
