//! Port of the `PluginLibrariesTest.Executes*` cases of `test/dsl/plugins.cpp`:
//! `typhon build -release -plugin <dir>` + run the executable, and
//! `typhon run -release -plugin <dir>`, against Rust plugin fixtures
//! (`tests/fixtures/plugin_{compiler,runtime}.rs`, ports of `test/dsl/library.cpp`).
//!
//! Needs the `typhon` binary: `TYPHON_BIN` or `<target>/debug/typhon`
//! (`cargo build -p typhon-cli -p typhon-runtime`). Skipped (with a message) if absent.
//! `InteractiveJITLoadsRuntimeForLatePlugin` needs the incremental JIT (out of scope).

use std::path::{Path, PathBuf};
use std::process::Command;

const EXT: &str = if cfg!(target_os = "macos") { "dylib" } else { "so" };

fn target_dir() -> PathBuf {
    // <target>/debug/deps/<test> -> <target>/debug
    std::env::current_exe().unwrap().parent().unwrap().parent().unwrap().to_path_buf()
}

fn typhon_bin() -> Option<PathBuf> {
    let p = std::env::var("TYPHON_BIN").map(PathBuf::from).unwrap_or_else(|_| target_dir().join("typhon"));
    p.exists().then_some(p)
}

fn fixture(name: &str) -> PathBuf {
    let p = target_dir().join("examples").join(format!("lib{name}.{EXT}"));
    if !p.exists() {
        let st = Command::new(env!("CARGO"))
            .args(["build", "-p", "typhon-compiler", "--example", name])
            .status()
            .unwrap();
        assert!(st.success());
    }
    p
}

struct Fx {
    dir: tempfile::TempDir,
    compiler_stem: String,
    runtime_stem: String,
    bin: PathBuf,
}

impl Fx {
    fn new() -> Option<Self> {
        let Some(bin) = typhon_bin() else {
            eprintln!("skipping: typhon binary not built (cargo build -p typhon-cli)");
            return None;
        };
        let dir = tempfile::tempdir().unwrap();
        let fx = Fx {
            dir,
            compiler_stem: "compiler/libtyphon_test_plugin_compiler".into(),
            runtime_stem: "runtime/libtyphon_test_plugin_runtime".into(),
            bin,
        };
        for (src, stem) in [
            (fixture("typhon_test_plugin_compiler"), &fx.compiler_stem),
            (fixture("typhon_test_plugin_runtime"), &fx.runtime_stem),
        ] {
            let dst = fx.dir.path().join(format!("{stem}.{EXT}"));
            std::fs::create_dir_all(dst.parent().unwrap()).unwrap();
            std::fs::copy(&src, &dst).unwrap();
        }
        Some(fx)
    }
    fn root(&self) -> String {
        self.dir.path().to_string_lossy().into_owned()
    }
    fn manifest(&self, libraries: &str) {
        std::fs::write(
            self.dir.path().join("plugin.toml"),
            format!("[about]\nname = \"test\"\nsupported = \">=0.0.0\"\n[library]\n{libraries}\n"),
        )
        .unwrap();
    }
    fn execute(&self, args: &[&str]) -> String {
        let out = Command::new(args[0]).args(&args[1..]).output().unwrap();
        assert!(out.status.success(), "{args:?} failed: {}", String::from_utf8_lossy(&out.stderr));
        String::from_utf8_lossy(&out.stdout).into_owned()
    }
    fn check(&self, libraries: &str, code: &str, expected: &str, links_compiler: bool, links_runtime: bool) {
        self.manifest(libraries);
        let root = self.root();
        let source = format!("{root}/program.codon");
        let exe = format!("{root}/program");
        std::fs::write(&source, code).unwrap();
        let bin = self.bin.to_string_lossy().into_owned();
        // `TYPHON_PLUGIN_TEST_MODE=-debug` exercises the plugin plumbing without release passes.
        let mode = std::env::var("TYPHON_PLUGIN_TEST_MODE").unwrap_or_else(|_| "-release".into());
        self.execute(&[&bin, "build", &mode, "-plugin", &root, "-o", &exe, &source]);
        assert_eq!(self.execute(&[&exe]), expected);
        assert_eq!(self.execute(&[&bin, "run", &mode, "-plugin", &root, &source]), expected);
        if cfg!(target_os = "macos") {
            let deps = self.execute(&["/usr/bin/otool", "-L", &exe]);
            assert_eq!(deps.contains("libtyphon_test_plugin_compiler"), links_compiler, "{deps}");
            assert_eq!(deps.contains("libtyphon_test_plugin_runtime"), links_runtime, "{deps}");
            if !links_compiler {
                let cmds = self.execute(&["/usr/bin/otool", "-l", &exe]);
                assert!(!cmds.contains(&format!("{root}/compiler")), "rpath to compiler dir");
            }
        }
        let _ = Path::new(&exe);
    }
}

#[test]
fn executes_with_separate_compiler_and_runtime_libraries() {
    let Some(fx) = Fx::new() else { return };
    let libs = format!("compiler = \"{}\"\nruntime = \"{}\"", fx.compiler_stem, fx.runtime_stem);
    fx.check(&libs, "from C import codon_test_runtime_value() -> int\nprint(codon_test_runtime_value())\n", "42\n", false, true);
}

#[test]
fn executes_with_legacy_combined_library() {
    let Some(fx) = Fx::new() else { return };
    let libs = format!("cpp = \"{}\"", fx.compiler_stem);
    fx.check(&libs, "from C import codon_test_compiler_value() -> int\nprint(codon_test_compiler_value())\n", "17\n", true, false);
}

#[test]
fn executes_with_only_runtime_library() {
    let Some(fx) = Fx::new() else { return };
    let libs = format!("runtime = \"{}\"", fx.runtime_stem);
    fx.check(&libs, "from C import codon_test_runtime_value() -> int\nprint(codon_test_runtime_value())\n", "42\n", false, true);
}

#[test]
fn executes_with_compiler_only_and_empty_runtime() {
    let Some(fx) = Fx::new() else { return };
    let libs = format!("cpp = \"{}\"\nruntime = \"\"", fx.compiler_stem);
    fx.check(&libs, "print(7)\n", "7\n", false, false);
}

#[test]
fn executes_with_explicit_runtime_link_arguments() {
    let Some(fx) = Fx::new() else { return };
    let libs = format!(
        "compiler = \"{}\"\nruntime = \"{}\"\nlink = [\"{{root}}/{}.{EXT}\"]",
        fx.compiler_stem, fx.runtime_stem, fx.runtime_stem
    );
    fx.check(&libs, "from C import codon_test_runtime_value() -> int\nprint(codon_test_runtime_value())\n", "42\n", false, true);
}
