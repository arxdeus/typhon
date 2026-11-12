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
        Config {
            bin: canon(bin),
            stdlib: canon(stdlib),
            test_dir: canon(test_dir),
            timeout: Duration::from_secs(secs),
        }
    }
}
