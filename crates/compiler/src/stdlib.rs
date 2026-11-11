//! Standard-library location and the default import filesystem.

use std::cell::RefCell;
use std::path::PathBuf;
use std::rc::Rc;

use typhon_common::fs::{DiskFileSystem, FileSystem};

/// Workspace default: `<repo>/stdlib` (the Codon stdlib, reused read-only).
pub const WORKSPACE_STDLIB: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../../stdlib");

#[cfg(feature = "embed-stdlib")]
static EMBEDDED: include_dir::Dir<'static> = include_dir::include_dir!("$CARGO_MANIFEST_DIR/../../../stdlib");

/// Candidate stdlib directories in priority order: `TYPHON_STDLIB_DIR`, `CODON_PATH`,
/// `<exe>/../lib/codon/stdlib`, `<exe>/../stdlib`, then the workspace stdlib.
pub fn stdlib_dirs() -> Vec<PathBuf> {
    let mut v = Vec::new();
    for var in ["TYPHON_STDLIB_DIR", "CODON_PATH"] {
        if let Ok(p) = std::env::var(var) {
            for part in std::env::split_paths(&p) {
                if !part.as_os_str().is_empty() {
                    v.push(part);
                }
            }
        }
    }
    if let Ok(exe) = std::env::current_exe()
        && let Some(root) = exe.parent()
    {
        v.push(root.join("../lib/codon/stdlib"));
        v.push(root.join("../stdlib"));
    }
    v.push(PathBuf::from(WORKSPACE_STDLIB));
    v
}

/// True when `dir` looks like a stdlib root (has `internal/__init__.codon`).
fn is_stdlib(dir: &std::path::Path) -> bool {
    dir.join("internal").join("__init__.codon").exists()
}

/// The default filesystem for `Compiler::new`.
pub fn default_filesystem(argv0: &str) -> Rc<RefCell<dyn FileSystem>> {
    let found = stdlib_dirs().into_iter().find(|d| is_stdlib(d));
    #[cfg(feature = "embed-stdlib")]
    if found.is_none() {
        let fs = typhon_common::fs::EmbeddedStdlib::new(&EMBEDDED, argv0, "", true);
        return Rc::new(RefCell::new(fs));
    }
    let mut fs = DiskFileSystem::new(argv0, "");
    // PORT-NOTE: C++ `Filesystem` only knows CODON_PATH and exe-relative paths; the
    // first valid stdlib directory found here is put in front so it wins import lookup.
    if let Some(dir) = found {
        let mut paths: Vec<String> = fs.search_paths().iter().map(|p| p.to_string_lossy().into_owned()).collect();
        let canon = typhon_common::fs::weakly_canonical(&dir).to_string_lossy().into_owned();
        paths.retain(|p| *p != canon);
        let mut fresh = DiskFileSystem::default();
        fresh.argv0 = argv0.to_string();
        fresh.add_search_path(&canon);
        for p in paths {
            fresh.add_search_path(&p);
        }
        fs = fresh;
    }
    Rc::new(RefCell::new(fs))
}
