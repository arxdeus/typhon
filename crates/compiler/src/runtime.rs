//! Locating and loading the runtime library (`libcodonrt`) for JIT execution.

use std::path::{Path, PathBuf};

use typhon_common::{CompilerError, Result};
use typhon_llvm::GcRootCallbacks;

/// Platform file name of the runtime dylib.
pub const RUNTIME_DYLIB: &str = if cfg!(target_os = "macos") {
    "libcodonrt.dylib"
} else if cfg!(windows) {
    "codonrt.dll"
} else {
    "libcodonrt.so"
};

/// Candidate directories: `TYPHON_LIB_DIR`, next to the executable, `<exe>/../lib`,
/// `<exe>/../lib/codon`, and (dev builds) the parent dir (`target/debug/deps` -> `target/debug`).
pub fn runtime_dirs() -> Vec<PathBuf> {
    let mut v = Vec::new();
    if let Ok(d) = std::env::var("TYPHON_LIB_DIR")
        && !d.is_empty()
    {
        v.push(PathBuf::from(d));
    }
    if let Ok(exe) = std::env::current_exe() {
        let exe = std::fs::canonicalize(&exe).unwrap_or(exe);
        if let Some(dir) = exe.parent() {
            v.push(dir.to_path_buf());
            v.push(dir.join("../lib"));
            v.push(dir.join("../lib/codon"));
            v.push(dir.join(".."));
            // Cargo puts dependency cdylibs in `target/<profile>/deps`.
            v.push(dir.join("deps"));
        }
    }
    v
}

/// Path of `libcodonrt` if found.
pub fn find_runtime() -> Option<PathBuf> {
    runtime_dirs().into_iter().map(|d| d.join(RUNTIME_DYLIB)).find(|p| p.exists())
}

/// Load a shared library permanently and globally (RTLD_GLOBAL) so the JIT's
/// process symbol generator resolves its exports.
pub fn load_global(path: &Path) -> Result<*mut libc::c_void> {
    #[cfg(unix)]
    {
        use libloading::os::unix::{Library, RTLD_GLOBAL, RTLD_NOW};
        // SAFETY: loading the runtime runs its initializers, which is what we want.
        let lib = unsafe { Library::open(Some(path), RTLD_NOW | RTLD_GLOBAL) }
            .map_err(|e| CompilerError::Io(format!("could not load {}: {e}", path.display())))?;
        let raw = lib.into_raw();
        Ok(raw)
    }
    #[cfg(not(unix))]
    {
        let lib = unsafe { libloading::Library::new(path) }
            .map_err(|e| CompilerError::Io(format!("could not load {}: {e}", path.display())))?;
        std::mem::forget(lib);
        Ok(std::ptr::null_mut())
    }
}

/// Look up `name` in the process (global namespace).
fn dlsym_global(name: &str) -> Option<usize> {
    #[cfg(unix)]
    {
        let c = std::ffi::CString::new(name).ok()?;
        // SAFETY: plain dlsym on RTLD_DEFAULT.
        let p = unsafe { libc::dlsym(libc::RTLD_DEFAULT, c.as_ptr()) };
        if p.is_null() { None } else { Some(p as usize) }
    }
    #[cfg(not(unix))]
    {
        let _ = name;
        None
    }
}

unsafe extern "C" {
    fn __addtf3();
    fn __subtf3();
    fn __multf3();
    fn __divtf3();
    fn __eqtf2();
    fn __netf2();
    fn __lttf2();
    fn __letf2();
    fn __gttf2();
    fn __getf2();
    fn __unordtf2();
    fn __extenddftf2();
    fn __extendsftf2();
    fn __trunctfdf2();
    fn __trunctfsf2();
    fn __fixtfdi();
    fn __fixtfsi();
    fn __fixunstfdi();
    fn __floatditf();
    fn __floatsitf();
    fn __floatunditf();
}

/// Soft-float `f128` libcalls (compiler-rt builtins) for JIT'd code.
///
/// PORT-NOTE: C++ codon links LLVM's compiler-rt into the binary; Apple's libSystem
/// does not export the `*tf*` routines, so the JIT could not resolve them. Rust's
/// `compiler_builtins` (always linked) provides them; we hand their addresses to the
/// engine as absolute symbols.
pub fn builtin_symbols() -> Vec<(String, u64)> {
    macro_rules! syms {
        ($($f:ident),*) => { vec![$((stringify!($f).to_string(), $f as *const () as usize as u64)),*] };
    }
    syms!(
        __addtf3, __subtf3, __multf3, __divtf3, __eqtf2, __netf2, __lttf2, __letf2, __gttf2, __getf2, __unordtf2,
        __extenddftf2, __extendsftf2, __trunctfdf2, __trunctfsf2, __fixtfdi, __fixtfsi, __fixunstfdi, __floatditf,
        __floatsitf, __floatunditf
    )
}

/// The loaded runtime: GC root callbacks for the JIT engine.
pub struct Runtime {
    pub path: PathBuf,
    pub gc_roots: Option<GcRootCallbacks>,
}

/// Load `libcodonrt` into the process (C++: the `codon` binary links it directly).
pub fn load_runtime() -> Result<Runtime> {
    let path = find_runtime().ok_or_else(|| {
        CompilerError::Io(format!(
            "could not find {RUNTIME_DYLIB} (searched: {}); set TYPHON_LIB_DIR",
            runtime_dirs().iter().map(|p| p.display().to_string()).collect::<Vec<_>>().join(", ")
        ))
    })?;
    load_global(&path)?;
    let add = dlsym_global("seq_gc_add_roots");
    let remove = dlsym_global("seq_gc_remove_roots");
    let gc_roots = match (add, remove) {
        (Some(a), Some(r)) => Some(GcRootCallbacks {
            // SAFETY: both are `extern "C" fn(*mut c_void, *mut c_void)` in libcodonrt.
            add_roots: unsafe { std::mem::transmute::<usize, unsafe extern "C" fn(*mut libc::c_void, *mut libc::c_void)>(a) },
            remove_roots: unsafe { std::mem::transmute::<usize, unsafe extern "C" fn(*mut libc::c_void, *mut libc::c_void)>(r) },
        }),
        _ => None,
    };
    Ok(Runtime { path, gc_roots })
}
