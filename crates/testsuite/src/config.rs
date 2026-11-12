//! Environment-driven configuration.
use std::path::PathBuf;
use std::time::Duration;

#[derive(Clone, Debug)]
pub struct Config {
    /// Compiler CLI (`TYPHON_BIN`).
    pub bin: PathBuf,
    /// Stdlib directory (`TYPHON_STDLIB_DIR`), exported to the child as `TYPHON_PATH`.
    pub stdlib: PathBuf,
    /// Test data dir (`TYPHON_TEST_DIR`), default `<workspace>/test`.
    pub test_dir: PathBuf,
    /// Per-case timeout (`TYPHON_TEST_TIMEOUT_SECS`, default 1200 like the C++ Windows watchdog).
    pub timeout: Duration,
    /// libpython for the python-interop cases (`TYPHON_PYTHON`, else auto-discovered).
    pub python: Option<PathBuf>,
}

/// The `typhon/` workspace root (this crate lives in `typhon/crates/testsuite`).
pub fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().unwrap()
}

impl Config {
    pub fn from_env() -> Config {
        let ws = workspace_root();
        let var = |k: &str| std::env::var_os(k).filter(|v| !v.is_empty()).map(PathBuf::from);
        let bin = var("TYPHON_BIN").unwrap_or_else(|| ws.join("target/debug/typhon"));
        let stdlib = var("TYPHON_STDLIB_DIR").map(|p| if p.is_relative() { ws.join(p) } else { p })
            .unwrap_or_else(|| ws.join("stdlib"));
        let test_dir = var("TYPHON_TEST_DIR").unwrap_or_else(|| ws.join("test"));
        let secs = std::env::var("TYPHON_TEST_TIMEOUT_SECS").ok().and_then(|s| s.parse().ok()).unwrap_or(1200);
        let canon = |p: PathBuf| p.canonicalize().unwrap_or(p);
        let python = var("TYPHON_PYTHON").or_else(find_libpython);
        Config {
            bin: canon(bin),
            stdlib: canon(stdlib),
            test_dir: canon(test_dir),
            timeout: Duration::from_secs(secs),
            python,
        }
    }
}

/// Locate a shared libpython like `test/python/find-python-library.py` does, without
/// needing `distutils`: ask `python3` for LIBDIR + LDLIBRARY / framework prefix.
pub fn find_libpython() -> Option<PathBuf> {
    let script = "import sysconfig,os\n\
        v=sysconfig.get_config_var\n\
        c=[os.path.join(v('LIBDIR') or '', v('LDLIBRARY') or ''),\n\
           os.path.join(v('LIBDIR') or '', 'libpython'+(v('LDVERSION') or '')+('.dylib' if os.uname().sysname=='Darwin' else '.so'))]\n\
        pf=v('PYTHONFRAMEWORKPREFIX')\n\
        if pf: c.append(os.path.join(pf, v('PYTHONFRAMEWORK') or 'Python')+'.framework/Versions/'+(v('VERSION') or '')+'/lib/libpython'+(v('VERSION') or '')+'.dylib')\n\
        print(next((p for p in c if os.path.isfile(p) and not p.endswith('.a')), ''))";
    let out = std::process::Command::new("python3").args(["-c", script]).output().ok()?;
    let path = String::from_utf8_lossy(&out.stdout).trim().to_string();
    (!path.is_empty()).then(|| PathBuf::from(path))
}
