//! Import-path resolution filesystem. Port of `IFilesystem`, `Filesystem`,
//! `ResourceFilesystem`, `ImportFile` and `library_path`.
//!
//! The the reference implementation `ResourceFilesystem` embeds the stdlib with `cmrc`; the Rust port uses
//! `include_dir` via [`EmbeddedStdlib`] (the embedding itself is done by the crate
//! that owns the `include_dir!` call, usually `typhon-compiler`).

use crate::error::{CompilerError, ErrorKind};
use crate::srcinfo::SrcInfo;
use std::path::{Path, PathBuf};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImportStatus {
    Stdlib,
    Package,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImportFile {
    pub status: ImportStatus,
    pub path: String,
    pub module: String,
}

pub trait FileSystem: Send + Sync {
    fn read_lines(&self, path: &Path) -> Result<Vec<String>, CompilerError>;
    fn exists(&self, path: &Path) -> bool;
    fn search_paths(&self) -> &[PathBuf];
    fn add_search_path(&mut self, p: &str);
    fn module0(&self) -> PathBuf;
    fn set_module0(&mut self, s: &str);

    fn canonical(&self, p: &Path) -> PathBuf {
        weakly_canonical(p)
    }

    /// the reference implementation `IFilesystem::get_root`.
    fn root_of(&self, sp: &Path) -> ImportFile {
        let s = sp.to_string_lossy().replace('\\', "/");
        let mut root = String::new();
        let mut is_stdlib = false;
        for p in self.search_paths() {
            let ps = p.to_string_lossy().replace('\\', "/");
            if s.starts_with(&ps) {
                root = ps;
                is_stdlib = true;
                break;
            }
        }
        let module0 = self.module0().parent().map(|p| p.to_path_buf()).unwrap_or_default();
        let m0 = module0.to_string_lossy().replace('\\', "/");
        if !is_stdlib && !m0.is_empty() && s.starts_with(&m0) {
            root = m0.clone();
        }
        let ext = ".py";
        let start = if root.is_empty() { 0 } else { root.len() + 1 };
        let module = s[start..s.len() - ext.len()].replace('/', ".");
        ImportFile {
            status: if !is_stdlib && root == m0 { ImportStatus::Package } else { ImportStatus::Stdlib },
            path: s,
            module,
        }
    }
}

/// `std::filesystem::weakly_canonical` equivalent.
pub fn weakly_canonical(p: &Path) -> PathBuf {
    if let Ok(c) = std::fs::canonicalize(p) {
        return c;
    }
    let mut existing = p.to_path_buf();
    let mut rest = Vec::new();
    while !existing.exists() {
        match (existing.file_name(), existing.parent()) {
            (Some(n), Some(parent)) => {
                rest.push(n.to_os_string());
                existing = parent.to_path_buf();
            }
            _ => return p.to_path_buf(),
        }
    }
    let mut out = std::fs::canonicalize(&existing).unwrap_or(existing);
    for n in rest.into_iter().rev() {
        out.push(n);
    }
    out
}

/// Port of `Filesystem` (real disk + `TYPHON_PATH` + paths relative to the executable).
#[derive(Clone, Debug, Default)]
pub struct DiskFileSystem {
    pub argv0: String,
    module0: PathBuf,
    search: Vec<PathBuf>,
}

impl DiskFileSystem {
    pub fn new(argv0: &str, module0: &str) -> Self {
        let mut fs = Self { argv0: argv0.to_string(), module0: PathBuf::from(module0), search: vec![] };
        if let Ok(p) = std::env::var("TYPHON_PATH") {
            fs.add_search_path(&p);
        }
        if let Ok(exe) = std::env::current_exe() {
            if let Some(root) = exe.parent() {
                for loci in ["../lib/typhon/stdlib", "../stdlib", "stdlib", "../lib/typhon/plugins", "../plugins"] {
                    fs.add_search_path(&root.join(loci).to_string_lossy());
                }
            }
        }
        fs
    }
}

impl FileSystem for DiskFileSystem {
    fn read_lines(&self, path: &Path) -> Result<Vec<String>, CompilerError> {
        let text = if path == Path::new("-") {
            std::io::read_to_string(std::io::stdin())?
        } else {
            std::fs::read_to_string(path).map_err(|_| {
                CompilerError::parser(ErrorKind::COMPILER_NO_FILE, SrcInfo::default(), &[&path.display()])
            })?
        };
        Ok(text.lines().map(str::to_string).collect())
    }
    fn exists(&self, path: &Path) -> bool {
        path.exists()
    }
    fn search_paths(&self) -> &[PathBuf] {
        &self.search
    }
    fn add_search_path(&mut self, p: &str) {
        let path = Path::new(p);
        if path.exists() {
            self.search.push(weakly_canonical(path));
        }
    }
    fn module0(&self) -> PathBuf {
        if self.module0.as_os_str().is_empty() { PathBuf::new() } else { weakly_canonical(&self.module0) }
    }
    fn set_module0(&mut self, s: &str) {
        self.module0 = weakly_canonical(Path::new(s));
    }
}

/// stdlib served from an embedded directory tree
/// (`include_dir::Dir`) mounted at `/stdlib`, falling back to disk when allowed.
pub struct EmbeddedStdlib {
    pub dir: &'static include_dir::Dir<'static>,
    pub allow_external: bool,
    disk: DiskFileSystem,
    search: Vec<PathBuf>,
}

impl EmbeddedStdlib {
    pub fn new(dir: &'static include_dir::Dir<'static>, argv0: &str, module0: &str, allow_external: bool) -> Self {
        Self {
            dir,
            allow_external,
            disk: DiskFileSystem::new(argv0, module0),
            search: vec![PathBuf::from("/stdlib")],
        }
    }
    fn lookup(&self, path: &Path) -> Option<&'static include_dir::File<'static>> {
        let rel = path.strip_prefix("/stdlib").ok()?;
        self.dir.get_file(rel)
    }
}

impl FileSystem for EmbeddedStdlib {
    fn read_lines(&self, path: &Path) -> Result<Vec<String>, CompilerError> {
        if let Some(f) = self.lookup(path) {
            let text = f.contents_utf8().unwrap_or_default();
            return Ok(text.lines().map(str::to_string).collect());
        }
        if self.allow_external {
            return self.disk.read_lines(path);
        }
        Err(CompilerError::parser(ErrorKind::COMPILER_NO_FILE, SrcInfo::default(), &[&path.display()]))
    }
    fn exists(&self, path: &Path) -> bool {
        self.lookup(path).is_some() || (self.allow_external && path.exists())
    }
    fn search_paths(&self) -> &[PathBuf] {
        &self.search
    }
    fn add_search_path(&mut self, p: &str) {
        self.search.push(PathBuf::from(p));
    }
    fn module0(&self) -> PathBuf {
        self.disk.module0()
    }
    fn set_module0(&mut self, s: &str) {
        self.disk.set_module0(s)
    }
    fn canonical(&self, p: &Path) -> PathBuf {
        if p.starts_with("/stdlib") { p.to_path_buf() } else { weakly_canonical(p) }
    }
}

/// Directory containing the running binary or shared library (`library_path`).
pub fn library_path() -> PathBuf {
    std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(Path::to_path_buf))
        .unwrap_or_default()
}
