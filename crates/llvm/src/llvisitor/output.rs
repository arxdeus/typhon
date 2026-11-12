//! Output paths: object/bitcode/LL/executable writers and `compile`
//! (port of llvisitor.cpp `writeTo*`, `setupGlobalCtor`, `compile`).

use std::path::{Path, PathBuf};
use std::process::Command;

use inkwell::targets::FileType;
use llvm_sys::LLVMLinkage::*;
use typhon_common::options::GlobalCtorMode;
use typhon_common::{CompilerError, Result};

use super::cross;
use super::ll::{self, Builder};
use super::{LlvmVisitor, MAIN_CTOR, MAIN_UNCLASH};

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

/// Directories searched for `libtyphonrt.a`: `TYPHON_LIB_DIR`, else next to the
/// `typhon` executable, `../lib` and `../lib/typhon`.
pub fn runtime_lib_dirs(argv0: &str) -> Vec<PathBuf> {
    if let Ok(d) = std::env::var("TYPHON_LIB_DIR")
        && !d.is_empty()
    {
        return vec![PathBuf::from(d)];
    }
    let exe = if argv0.is_empty() {
        std::env::current_exe().unwrap_or_default()
    } else {
        let p = PathBuf::from(argv0);
        // A non-existent argv0 (e.g. `target/debug/typhon` from tests) still names
        // the directory to search.
        std::fs::canonicalize(&p).unwrap_or_else(|_| if p.parent().is_some_and(|d| d.is_dir()) { p } else { std::env::current_exe().unwrap_or_default() })
    };
    let base = exe.parent().map(Path::to_path_buf).unwrap_or_default();
    let mut dirs = vec![base.clone()];
    for rel in ["../lib", "../lib/typhon"] {
        if let Ok(c) = std::fs::canonicalize(base.join(rel)) {
            dirs.push(c);
        }
    }
    dirs
}

#[cfg(windows)]
const RUNTIME_ARCHIVE: &str = "typhonrt.lib";
#[cfg(not(windows))]
const RUNTIME_ARCHIVE: &str = "libtyphonrt.a";

/// Full path of the static runtime archive linked into every executable / library.
pub fn static_runtime_archive(argv0: &str) -> Result<String> {
    let dirs = runtime_lib_dirs(argv0);
    dirs.iter().map(|d| d.join(RUNTIME_ARCHIVE)).find(|p| p.is_file()).map(|p| p.to_string_lossy().into_owned()).ok_or_else(|| {
        let searched: Vec<String> = dirs.iter().map(|d| d.display().to_string()).collect();
        CompilerError::Llvm(format!("{RUNTIME_ARCHIVE} not found in {} (set TYPHON_LIB_DIR)", searched.join(", ")))
    })
}

/// Linker flags that discard unreferenced sections and unneeded shared libraries.
#[cfg(target_os = "macos")]
const LINK_GC_FLAGS: &[&str] = &["-Wl,-dead_strip", "-Wl,-dead_strip_dylibs"];
#[cfg(all(unix, not(target_os = "macos")))]
const LINK_GC_FLAGS: &[&str] = &["-Wl,--gc-sections", "-Wl,--as-needed"];
#[cfg(windows)]
const LINK_GC_FLAGS: &[&str] = &["-Wl,/OPT:REF"];

/// System libraries `libtyphonrt.a` needs, from `cargo rustc -p typhon-runtime --release
/// --crate-type staticlib -- --print native-static-libs`. zlib, bzip2, liblzma, bdwgc
/// and OpenMP (typhon-omp) are inside the archive; no C++ runtime is needed.
/// Linker arguments that embed the runtime: section/dylib GC flags, the full path of
/// `libtyphonrt.a`, and the system libraries it needs. Append after the program's
/// objects and user libraries.
pub fn runtime_link_args(argv0: &str) -> Result<Vec<String>> {
    let mut v: Vec<String> = LINK_GC_FLAGS.iter().map(|s| s.to_string()).collect();
    v.push(static_runtime_archive(argv0)?);
    v.extend(RUNTIME_NATIVE_LIBS.iter().map(|s| s.to_string()));
    Ok(v)
}

#[cfg(target_os = "macos")]
const RUNTIME_NATIVE_LIBS: &[&str] = &["-liconv", "-lm", "-framework", "Accelerate"];
#[cfg(all(unix, not(target_os = "macos")))]
const RUNTIME_NATIVE_LIBS: &[&str] = &["-lgcc_s", "-lutil", "-lrt", "-lpthread", "-lm", "-ldl", "-lc"];
#[cfg(windows)]
const RUNTIME_NATIVE_LIBS: &[&str] = &["-lkernel32", "-lntdll", "-luserenv", "-lws2_32", "-ldbghelp", "-llibopenblas"];

/// Runtime archive for a cross target: `$TYPHON_LIB_DIR/<T>/libtyphonrt.a`,
/// `<exe dir>/../<T>/release/libtyphonrt.a` (cargo cross output) or
/// `<exe dir>/../lib/typhon/<T>/libtyphonrt.a`.
pub fn cross_runtime_archive(argv0: &str, target: &typhon_common::target::TargetInfo) -> Result<String> {
    let t = target.rust_triple().ok_or_else(|| CompilerError::Llvm(format!("unsupported cross target '{}'", target.triple)))?;
    let mut candidates = Vec::new();
    if let Ok(d) = std::env::var("TYPHON_LIB_DIR")
        && !d.is_empty()
    {
        candidates.push(PathBuf::from(&d).join(&t).join("libtyphonrt.a"));
    }
    let exe = if argv0.is_empty() { std::env::current_exe().unwrap_or_default() } else { std::fs::canonicalize(argv0).unwrap_or_else(|_| PathBuf::from(argv0)) };
    if let Some(dir) = exe.parent() {
        candidates.push(dir.join("..").join(&t).join("release").join("libtyphonrt.a"));
        candidates.push(dir.join("../lib/typhon").join(&t).join("libtyphonrt.a"));
    }
    candidates.iter().find(|p| p.is_file()).map(|p| p.to_string_lossy().into_owned()).ok_or_else(|| {
        let searched: Vec<String> = candidates.iter().map(|d| d.display().to_string()).collect();
        CompilerError::Llvm(format!(
            "runtime for target {t} not found (searched {}); build it with `tools/build-runtime.sh {t}` or set TYPHON_LIB_DIR to a directory containing {t}/libtyphonrt.a",
            searched.join(", ")
        ))
    })
}

impl<'ctx> LlvmVisitor<'ctx> {
    /// `runLLVMPipeline`.
    fn run_llvm_pipeline(&mut self) -> Result<()> {
        self.prepare_win_exports();
        if let Some(db) = &self.db.builder {
            db.finalize();
        }
        crate::optimize::optimize(self.module.as_ref().expect("no module"), &self.options)
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

    /// Library entry points for `@export` functions. Each exported body is
    /// renamed to an internal `<name>.typhon.body` (all internal callers,
    /// including recursion, keep calling it directly) and a thin public wrapper
    /// with the original symbol name and ABI calls the idempotent
    /// `seq_register_current_thread()` (so foreign threads such as Dart isolates
    /// are known to the GC) and then tail-calls the body. Returns the exported
    /// symbol names.
    fn setup_library_exports(&mut self) -> Vec<String> {
        let mm = self.m();
        let exports: Vec<_> =
            ll::functions(mm).into_iter().filter(|&f| !ll::is_declaration(f) && ll::has_string_attr(f, super::EXPORT_ATTR)).collect();
        if exports.is_empty() {
            return Vec::new();
        }
        let b = Builder::new(self.c);
        let reg = ll::get_or_insert_function(mm, "seq_register_current_thread", ll::fn_ty(b.void(), &[], false));
        let mut names = Vec::new();
        for body in exports {
            let name = ll::name(body);
            names.push(name.clone());
            let fnty = ll::function_type(body);
            ll::set_name(body, &format!("{name}.typhon.body"));
            ll::set_linkage(body, LLVMInternalLinkage);
            let wrapper = ll::add_function(mm, &name, fnty);
            ll::copy_all_attrs(body, wrapper);
            ll::set_fn_call_conv(wrapper, ll::fn_call_conv(body));
            ll::set_linkage(wrapper, LLVMExternalLinkage);
            ll::remove_enum_fn_attr(wrapper, "alwaysinline");
            // Export markers move to the wrapper (Windows export handling keys off them).
            ll::remove_string_attr(body, super::EXPORT_ATTR);
            ll::remove_string_attr(body, "typhon-c-export");
            b.at_end(ll::append_block(self.c, wrapper, "entry"));
            b.call(ll::function_type(reg), reg, &[]);
            let call = b.call(fnty, body, &ll::params(wrapper));
            ll::set_tail_call(call);
            ll::set_call_conv(call, ll::fn_call_conv(body));
            if ll::is_void(ll::fn_ret_ty(fnty)) {
                b.ret_void();
            } else {
                b.ret(call);
            }
        }
        names
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
        let exports = if library { self.setup_library_exports() } else { Vec::new() };
        if self.options.is_cross() {
            return self.write_cross_executable(filename, argv0, library, libs, user_flags, &exports);
        }
        let obj_file = format!("{filename}.o");
        self.write_to_object_file(&obj_file, library || pie, false)?;
        let runtime = runtime_link_args(argv0)?;

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
        for d in runtime_lib_dirs(argv0) {
            command.push(format!("-L{}", d.display()));
        }
        for l in libs {
            command.push(format!("-l{l}"));
        }
        // Runtime after user libs so their runtime references resolve. The GC flags
        // in it drop unreferenced sections and unused dylibs (the `c++` driver adds
        // the C++ runtime, which libtyphonrt does not need).
        command.extend(runtime);
        command.extend(["-o".to_string(), filename.to_string()]);
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

    /// Cross link for `-target` (Android NDK clang / Xcode SDK clang).
    fn write_cross_executable(
        &mut self,
        filename: &str,
        argv0: &str,
        library: bool,
        libs: &[String],
        user_flags: Vec<String>,
        exports: &[String],
    ) -> Result<()> {
        let target = self.options.target_info();
        let obj_file = format!("{filename}.o");
        self.write_to_object_file(&obj_file, true, false)?;
        let archive = cross_runtime_archive(argv0, &target)?;
        let list_file = format!("{filename}.exports");
        let mut command = cross::linker_command(&target, &self.options.target, &self.options.sysroot)?;
        command.push("-Wno-unused-command-line-argument".into());
        command.push(obj_file.clone());
        let base = Path::new(filename).file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
        if library {
            command.push("-shared".into());
            if target.is_apple() {
                command.push(format!("-Wl,-install_name,@rpath/{base}"));
            } else {
                command.push(format!("-Wl,-soname,{base}"));
            }
            // Only `@export` functions are visible; runtime symbols stay hidden.
            if target.is_apple() {
                let body: String = exports.iter().map(|e| format!("_{e}\n")).collect();
                std::fs::write(&list_file, body).map_err(io_err)?;
                command.push(format!("-Wl,-exported_symbols_list,{list_file}"));
            } else {
                let body = format!("{{\n  global:\n{}  local: *;\n}};\n", exports.iter().map(|e| format!("    {e};\n")).collect::<String>());
                std::fs::write(&list_file, body).map_err(io_err)?;
                command.push(format!("-Wl,--version-script={list_file}"));
                command.push("-Wl,--exclude-libs,ALL".into());
            }
        } else if !target.is_apple() {
            command.push("-pie".into());
        }
        for l in libs {
            command.push(format!("-l{l}"));
        }
        command.extend(cross::gc_flags(&target).iter().map(|s| s.to_string()));
        command.push(archive);
        command.extend(cross::native_libs(&target).iter().map(|s| s.to_string()));
        command.extend(["-o".to_string(), filename.to_string()]);
        command.extend(user_flags);
        let r = execute_command(&command);
        let _ = std::fs::remove_file(&obj_file);
        let _ = std::fs::remove_file(&list_file);
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
