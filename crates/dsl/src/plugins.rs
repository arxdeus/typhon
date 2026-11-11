//! Port of `codon/dsl/plugins.{h,cpp}`: `Plugin` and `PluginManager`.

use std::collections::HashSet;
use std::ffi::{CStr, c_char, c_void};
use std::mem::ManuallyDrop;
use std::path::{Path, PathBuf};

use typhon_common::config::{CODON_VERSION, CODON_VERSION_MAJOR, CODON_VERSION_MINOR, CODON_VERSION_PATCH};
use typhon_common::{CompilerError, Result};
use typhon_llvm::inkwell::module::Module;
use typhon_llvm::{LlvmPassHooks, PassStage};

use crate::abi::{ABI_TAG_SYMBOL, ABI_VERSION, LOAD_SYMBOL, abi_tag};
use crate::dsl::{DefaultDsl, Dsl, Info};
use crate::semver_range::VersionRange;

fn plugin_error<T>(msg: impl Into<String>) -> Result<T> {
    Err(CompilerError::Plugin(msg.into()))
}

/// Platform shared-library extension (C++ `libExt`).
pub const LIB_EXT: &str = if cfg!(target_os = "macos") {
    "dylib"
} else if cfg!(windows) {
    "dll"
} else {
    "so"
};

/// C++ `struct Plugin`.
pub struct Plugin {
    /// The associated DSL. Declared first so it drops before the library handle.
    pub dsl: Box<dyn Dsl>,
    /// Plugin information.
    pub info: Info,
    /// Library handle (`None` for stdlib-only plugins). Never unloaded, like C++
    /// `DynamicLibrary::getPermanentLibrary` (code/vtables may outlive the plugin).
    handle: ManuallyDrop<Option<libloading::Library>>,
}

impl Plugin {
    pub fn new(dsl: Box<dyn Dsl>, info: Info) -> Self {
        Plugin { dsl, info, handle: ManuallyDrop::new(None) }
    }
    /// True if a compiler dylib was loaded for this plugin.
    pub fn has_library(&self) -> bool {
        self.handle.is_some()
    }
}

/// Forwards LLVM hooks to the DSL so `&Plugin` can be passed to `typhon_llvm::optimize`.
impl LlvmPassHooks for Plugin {
    fn add_llvm_passes(&self, stage: PassStage, module: &Module, debug: bool) -> Result<()> {
        self.dsl.add_llvm_passes(stage, module, debug)
    }
    fn pipeline_text(&self, stage: PassStage, debug: bool) -> Option<String> {
        self.dsl.pipeline_text(stage, debug)
    }
}

/// C++ `class PluginManager`.
pub struct PluginManager {
    argv0: String,
    plugins: Vec<Box<Plugin>>,
    loaded_runtime_libraries: HashSet<String>,
}

impl PluginManager {
    pub fn new(argv0: impl Into<String>) -> Self {
        PluginManager { argv0: argv0.into(), plugins: Vec::new(), loaded_runtime_libraries: HashSet::new() }
    }

    pub fn iter(&self) -> impl Iterator<Item = &Plugin> {
        self.plugins.iter().map(|p| &**p)
    }
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut Plugin> {
        self.plugins.iter_mut().map(|p| &mut **p)
    }
    pub fn len(&self) -> usize {
        self.plugins.len()
    }
    pub fn is_empty(&self) -> bool {
        self.plugins.is_empty()
    }
    /// All plugins as LLVM hooks (for `typhon_llvm::optimize`).
    pub fn llvm_hooks(&self) -> Vec<&dyn LlvmPassHooks> {
        self.plugins.iter().map(|p| &**p as &dyn LlvmPassHooks).collect()
    }

    fn default_install_toml(&self, path: &str) -> PathBuf {
        // C++: parent(executable_path(argv0)) / "../lib/codon/plugins" / path / plugin.toml
        let exe = if self.argv0.is_empty() {
            std::env::current_exe().unwrap_or_default()
        } else {
            let p = PathBuf::from(&self.argv0);
            std::fs::canonicalize(&p).unwrap_or(p)
        };
        let parent = exe.parent().map(Path::to_path_buf).unwrap_or_default();
        parent.join("../lib/codon/plugins").join(path).join("plugin.toml")
    }

    /// C++ `PluginManager::load`: loads the plugin in directory `path` (containing `plugin.toml`).
    pub fn load(&mut self, path: &str) -> Result<&mut Plugin> {
        let mut toml_path = Path::new(path).join("plugin.toml");
        if !toml_path.exists() {
            toml_path = self.default_install_toml(path);
        }
        let shown = toml_path.display().to_string();
        let text = match std::fs::read_to_string(&toml_path) {
            Ok(t) => t,
            Err(e) => return plugin_error(format!("[toml::parse_file(\"{shown}\")] {e}")),
        };
        let tml: toml::Table = match text.parse() {
            Ok(t) => t,
            Err(e) => return plugin_error(format!("[toml::parse_file(\"{shown}\")] {e}")),
        };
        let root = toml_path.parent().map(Path::to_path_buf).unwrap_or_default();
        let info = parse_info(&tml, &root)?;

        // empty or missing range means any version is supported
        if !info.supported.is_empty() {
            let Some(range) = VersionRange::parse(&info.supported) else {
                return plugin_error(format!(
                    "[semver::try_parse_range(\"{}\")] invalid range",
                    info.supported
                ));
            };
            let v = semver::Version::new(
                CODON_VERSION_MAJOR as u64,
                CODON_VERSION_MINOR as u64,
                CODON_VERSION_PATCH as u64,
            );
            if !range.contains(&v) {
                return plugin_error(format!(
                    "unsupported version {CODON_VERSION} (supported: {})",
                    info.supported
                ));
            }
        }

        let dylib = info.compiler_dylib_path().to_string();
        let plugin = if !dylib.is_empty() {
            let (dsl, lib) = load_compiler_library(&dylib)?;
            Plugin { dsl, info, handle: ManuallyDrop::new(Some(lib)) }
        } else {
            Plugin::new(Box::new(DefaultDsl), info)
        };
        self.plugins.push(Box::new(plugin));
        Ok(self.plugins.last_mut().map(|p| &mut **p).expect("just pushed"))
    }

    /// C++ `loadRuntimeLibraries`: loads (permanently) every plugin runtime library not yet
    /// loaded and distinct from its compiler library. Failures are retried on later calls.
    pub fn load_runtime_libraries(&mut self) -> Result<()> {
        for plugin in &self.plugins {
            let rt = plugin.info.runtime_dylib_path();
            if rt.is_empty()
                || rt == plugin.info.compiler_dylib_path()
                || self.loaded_runtime_libraries.contains(rt)
            {
                continue;
            }
            // SAFETY: loading an arbitrary user-provided library runs its initializers;
            // this mirrors C++ `LoadLibraryPermanently`.
            match unsafe { libloading::Library::new(rt) } {
                Ok(lib) => std::mem::forget(lib),
                Err(e) => {
                    return plugin_error(format!("could not load runtime library '{rt}': {e}"));
                }
            }
            self.loaded_runtime_libraries.insert(rt.to_string());
        }
        Ok(())
    }
}

fn load_compiler_library(path: &str) -> Result<(Box<dyn Dsl>, libloading::Library)> {
    // SAFETY: see load_runtime_libraries; symbols are validated through the ABI tag below.
    let lib = match unsafe { libloading::Library::new(path) } {
        Ok(l) => l,
        Err(e) => {
            return plugin_error(format!("[libloading::Library::new(\"{path}\")] {e}"));
        }
    };
    // PORT-NOTE: C++ plugins export `std::unique_ptr<DSL> load()`; those cannot be loaded.
    // Typhon plugins must export the C-ABI symbols described in PLUGINS.md.
    let tag = unsafe {
        match lib.get::<unsafe extern "C" fn() -> *const c_char>(ABI_TAG_SYMBOL) {
            Ok(f) => {
                let p = f();
                if p.is_null() { String::new() } else { CStr::from_ptr(p).to_string_lossy().into_owned() }
            }
            Err(_) => {
                return plugin_error(format!(
                    "could not find 'typhon_plugin_abi_tag' in plugin shared library: {path} \
                     (C++ Codon plugins are not supported)"
                ));
            }
        }
    };
    if tag != abi_tag() {
        return plugin_error(format!(
            "plugin ABI mismatch in {path}: plugin built for '{tag}', compiler is '{}'",
            abi_tag()
        ));
    }
    let raw = unsafe {
        match lib.get::<unsafe extern "C" fn(u32) -> *mut c_void>(LOAD_SYMBOL) {
            Ok(f) => f(ABI_VERSION),
            Err(_) => {
                return plugin_error(format!(
                    "could not find 'typhon_plugin_load' in plugin shared library: {path}"
                ));
            }
        }
    };
    if raw.is_null() {
        return plugin_error(format!("plugin load() returned null: {path}"));
    }
    // SAFETY: ABI tag matched, so `raw` is a `Box<Box<dyn Dsl>>` from the same toolchain/typhon.
    let dsl: Box<Box<dyn Dsl>> = unsafe { Box::from_raw(raw as *mut Box<dyn Dsl>) };
    Ok((*dsl, lib))
}

fn get_str(t: Option<&toml::Value>) -> String {
    t.and_then(toml::Value::as_str).unwrap_or("").to_string()
}

/// Builds `Info` from a parsed manifest (C++ body of `load` before version checks).
pub fn parse_info(tml: &toml::Table, root: &Path) -> Result<Info> {
    let empty = toml::Table::new();
    let about = tml.get("about").and_then(toml::Value::as_table).unwrap_or(&empty);
    let library = tml.get("library").and_then(toml::Value::as_table).unwrap_or(&empty);
    let library_path = |name: &str| -> String {
        if name.is_empty() {
            return String::new();
        }
        root.join(format!("{name}.{LIB_EXT}")).display().to_string()
    };
    for key in ["compiler", "runtime"] {
        if library.get(key).is_some_and(|v| !v.is_str()) {
            return plugin_error(format!("library.{key} must be a string"));
        }
    }
    let compiler_lib = library.get("compiler").and_then(|v| v.as_str()).map(str::to_string);
    let mut runtime_lib = library.get("runtime").and_then(|v| v.as_str()).map(str::to_string);
    let link = library.get("link");
    if runtime_lib.is_none() && link.and_then(toml::Value::as_bool) == Some(false) {
        runtime_lib = Some(String::new());
    }
    let mut link_args: Vec<String> = match link {
        Some(toml::Value::Array(a)) => {
            a.iter().filter_map(|v| v.as_str()).filter(|s| !s.is_empty()).map(str::to_string).collect()
        }
        other => {
            let l = get_str(other);
            if l.is_empty() { vec![] } else { vec![l] }
        }
    };
    let root_s = root.display().to_string();
    for l in &mut link_args {
        // PORT-NOTE: C++ uses fmt::format with a named `root` arg; only `{root}` is supported.
        *l = l.replace("{root}", &root_s);
    }
    let codon_lib = get_str(library.get("codon"));
    let stdlib_path =
        if codon_lib.is_empty() { String::new() } else { root.join(codon_lib).display().to_string() };
    Ok(Info {
        name: get_str(about.get("name")),
        description: get_str(about.get("description")),
        version: get_str(about.get("version")),
        url: get_str(about.get("url")),
        supported: get_str(about.get("supported")),
        stdlib_path,
        dylib_path: library_path(&get_str(library.get("cpp"))),
        link_args,
        compiler_dylib_path: compiler_lib.map(|s| library_path(&s)),
        runtime_dylib_path: runtime_lib.map(|s| library_path(&s)),
    })
}
