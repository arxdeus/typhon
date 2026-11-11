//! Port of the applicable cases of `test/dsl/plugins.cpp`.
//! PORT-NOTE: the `Executes*` cases (end-to-end `codon build/run -plugin`) and the JIT case
//! belong to the compiler/CLI lanes and are not ported here.

use std::path::{Path, PathBuf};
use typhon_dsl::{Info, LIB_EXT, PluginManager, VersionRange};
use typhon_llvm::{LlvmPassHooks, PassStage};

fn fixture_lib() -> PathBuf {
    let exe = std::env::current_exe().unwrap();
    let debug = exe.parent().unwrap().parent().unwrap();
    let prefix = if cfg!(windows) { "" } else { "lib" };
    let p = debug.join("examples").join(format!("{prefix}test_plugin.{LIB_EXT}"));
    if !p.exists() {
        let st = std::process::Command::new(env!("CARGO"))
            .args(["build", "-p", "typhon-dsl", "--example", "test_plugin"])
            .status()
            .unwrap();
        assert!(st.success());
    }
    p
}

struct Fx {
    dir: tempfile::TempDir,
    manager: PluginManager,
}

impl Fx {
    fn new() -> Self {
        let dir = tempfile::Builder::new().prefix("codon-plugin-libraries").tempdir().unwrap();
        let fx = Fx { dir, manager: PluginManager::new("") };
        for stem in ["compiler/plugin", "runtime/plugin"] {
            let dst = fx.library_path(stem);
            std::fs::create_dir_all(Path::new(&dst).parent().unwrap()).unwrap();
            std::fs::copy(fixture_lib(), &dst).unwrap();
        }
        fx
    }
    fn root(&self) -> String {
        self.dir.path().display().to_string()
    }
    fn manifest_with(&self, libraries: &str, supported: &str) {
        let text = format!("[about]\nname = \"test\"\n{supported}[library]\n{libraries}\n");
        std::fs::write(self.dir.path().join("plugin.toml"), text).unwrap();
    }
    fn manifest(&self, libraries: &str) {
        self.manifest_with(libraries, "supported = \">=0.0.0\"\n");
    }
    fn library_path(&self, name: &str) -> String {
        self.dir.path().join(format!("{name}.{LIB_EXT}")).display().to_string()
    }
    fn load(&mut self) -> Result<Info, String> {
        let r = self.root();
        self.manager.load(&r).map(|p| p.info.clone()).map_err(|e| e.to_string())
    }
}

#[test]
fn legacy_library_defaults_to_compiler_and_runtime() {
    let mut info = Info {
        name: "name".into(),
        supported: "*".into(),
        stdlib_path: "stdlib".into(),
        dylib_path: "legacy".into(),
        link_args: vec!["-lextra".into()],
        ..Default::default()
    };
    assert_eq!(info.compiler_dylib_path(), "legacy");
    assert_eq!(info.runtime_dylib_path(), "legacy");
    info.compiler_dylib_path = Some("compiler".into());
    assert_eq!(info.compiler_dylib_path(), "compiler");
    assert_eq!(info.runtime_dylib_path(), "legacy");
    info.runtime_dylib_path = Some("runtime".into());
    assert_eq!(info.runtime_dylib_path(), "runtime");
    info.runtime_dylib_path = Some(String::new());
    assert!(info.runtime_dylib_path().is_empty());
    info.compiler_dylib_path = Some(String::new());
    assert!(info.compiler_dylib_path().is_empty());
    info.compiler_dylib_path = None;
    info.runtime_dylib_path = None;
    assert_eq!(info.compiler_dylib_path(), "legacy");
}

#[test]
fn runtime_only_library_is_not_loaded_into_compiler() {
    let mut fx = Fx::new();
    fx.manifest("runtime = \"missing-runtime\"");
    let info = fx.load().unwrap();
    assert!(info.compiler_dylib_path().is_empty());
    assert_eq!(info.runtime_dylib_path(), fx.library_path("missing-runtime"));
    let e = fx.manager.load_runtime_libraries().unwrap_err().to_string();
    assert!(e.contains("missing-runtime"), "{e}");
}

#[test]
fn empty_compiler_overrides_legacy_and_runtime_inherits_it() {
    let mut fx = Fx::new();
    fx.manifest("cpp = \"missing-legacy\"\ncompiler = \"\"");
    let info = fx.load().unwrap();
    assert_eq!(info.dylib_path, fx.library_path("missing-legacy"));
    assert!(info.compiler_dylib_path().is_empty());
    assert_eq!(info.runtime_dylib_path(), info.dylib_path);
}

#[test]
fn runtime_library_load_failures_are_retried() {
    let mut fx = Fx::new();
    fx.manifest("runtime = \"retry-runtime\"");
    fx.load().unwrap();
    for _ in 0..2 {
        let e = fx.manager.load_runtime_libraries().unwrap_err().to_string();
        assert!(e.contains("retry-runtime"));
    }
    std::fs::copy(fixture_lib(), fx.library_path("retry-runtime")).unwrap();
    for _ in 0..2 {
        fx.manager.load_runtime_libraries().unwrap();
    }
}

#[test]
fn loads_new_runtime_libraries_after_previous_success() {
    let mut fx = Fx::new();
    fx.manifest("runtime = \"runtime/plugin\"");
    fx.load().unwrap();
    fx.manager.load_runtime_libraries().unwrap();
    fx.manifest("runtime = \"late-runtime\"");
    fx.load().unwrap();
    let e = fx.manager.load_runtime_libraries().unwrap_err().to_string();
    assert!(e.contains("late-runtime"));
    std::fs::copy(fixture_lib(), fx.library_path("late-runtime")).unwrap();
    fx.manager.load_runtime_libraries().unwrap();
}

#[test]
fn empty_runtime_overrides_legacy() {
    let mut fx = Fx::new();
    fx.manifest("cpp = \"missing-legacy\"\ncompiler = \"\"\nruntime = \"\"");
    let info = fx.load().unwrap();
    assert!(info.runtime_dylib_path().is_empty());
    fx.manager.load_runtime_libraries().unwrap();
}

#[test]
fn compiler_override_selects_the_loaded_library() {
    let mut fx = Fx::new();
    fx.manifest("cpp = \"missing-legacy\"\ncompiler = \"missing-compiler\"\nruntime = \"\"");
    let e = fx.load().unwrap_err();
    assert!(e.contains("missing-compiler") && !e.contains("missing-legacy"), "{e}");
}

#[test]
fn empty_runtime_does_not_disable_legacy_compiler_loading() {
    let mut fx = Fx::new();
    fx.manifest("cpp = \"missing-legacy\"\nruntime = \"\"");
    assert!(fx.load().unwrap_err().contains("missing-legacy"));
}

#[test]
fn legacy_link_false_means_empty_runtime() {
    let mut fx = Fx::new();
    fx.manifest("cpp = \"missing-legacy\"\ncompiler = \"\"\nlink = false");
    assert!(fx.load().unwrap().runtime_dylib_path().is_empty());
}

#[test]
fn explicit_link_arguments_remain_independent() {
    let mut fx = Fx::new();
    fx.manifest("runtime = \"\"\nlink = [\"-lstandalone\", \"{root}/custom.a\"]");
    let info = fx.load().unwrap();
    assert_eq!(info.link_args, vec!["-lstandalone".to_string(), format!("{}/custom.a", fx.root())]);
}

#[test]
fn rejects_non_string_library_paths() {
    let mut fx = Fx::new();
    fx.manifest("runtime = false");
    assert!(fx.load().unwrap_err().contains("library.runtime must be a string"));
}

#[test]
fn accepts_supported_version_ranges() {
    let mut fx = Fx::new();
    for s in [
        "",
        "supported = \"\"\n",
        "supported = \"*\"\n",
        "supported = \">=999.0.0 || >=0.0.0\"\n",
        "supported = \"^999.0.0 || ~999.0.0 || >=0.0.0\"\n",
    ] {
        fx.manifest_with("runtime = \"missing-runtime\"", s);
        fx.load().unwrap_or_else(|e| panic!("{s}: {e}"));
    }
}

#[test]
fn rejects_unsupported_version_ranges() {
    let mut fx = Fx::new();
    fx.manifest_with("runtime = \"missing-runtime\"", "supported = \">=999.0.0\"\n");
    assert!(fx.load().unwrap_err().contains("unsupported version"));
}

#[test]
fn rejects_invalid_version_ranges() {
    let mut fx = Fx::new();
    for s in [">=0.0.0 @", ">=0.0.0 ||", " "] {
        fx.manifest_with("runtime = \"missing-runtime\"", &format!("supported = \"{s}\"\n"));
        let e = fx.load().unwrap_err();
        assert!(e.contains("invalid range"), "{s}: {e}");
    }
}

#[test]
fn version_range_spacing() {
    let v = semver::Version::new(0, 20, 6);
    assert!(VersionRange::parse(">= 0.20.0 < 0.21.0").unwrap().contains(&v));
    assert!(!VersionRange::parse(">=0.21.0").unwrap().contains(&v));
}

// Typhon-specific: actual Rust plugin loading through the C-ABI entry points.
#[test]
fn loads_rust_compiler_plugin() {
    let mut fx = Fx::new();
    fx.manifest("compiler = \"compiler/plugin\"\nruntime = \"\"\ncodon = \"stdlib\"");
    let root = fx.root();
    let p = fx.manager.load(&root).unwrap();
    assert!(p.has_library());
    assert_eq!(p.info.stdlib_path, format!("{root}/stdlib"));
    let kw = p.dsl.expr_keywords();
    assert_eq!(kw[0].keyword, "test_expr");
    let bk = p.dsl.block_keywords();
    assert!(bk[0].has_expr && bk[0].keyword == "test_block");
    assert_eq!(p.pipeline_text(PassStage::OptimizerLast, false).as_deref(), Some("instcombine"));
    assert_eq!(fx.manager.llvm_hooks().len(), 1);
}

#[test]
fn stdlib_only_plugin_uses_default_dsl() {
    let mut fx = Fx::new();
    fx.manifest("codon = \"stdlib\"");
    let root = fx.root();
    let p = fx.manager.load(&root).unwrap();
    assert!(!p.has_library());
    assert!(p.dsl.expr_keywords().is_empty());
}

#[test]
fn rejects_non_typhon_library() {
    let mut fx = Fx::new();
    // A dylib lacking the ABI symbols: libSystem/libc stand-in is platform specific, so use a
    // plain non-library file; loading must fail with a plugin error, not crash.
    std::fs::write(fx.library_path("bogus"), b"not a library").unwrap();
    fx.manifest("compiler = \"bogus\"");
    assert!(fx.load().unwrap_err().contains("bogus"));
}
