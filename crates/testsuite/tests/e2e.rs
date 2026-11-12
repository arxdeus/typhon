//! End-to-end `.codon` suites (port of test/main.cpp). Run: `cargo test -p typhon-testsuite --test e2e -- core/`
use libtest_mimic::{Arguments, Failed, Trial};
use typhon_testsuite::{config::Config, runner, suites};

fn main() {
    let args = Arguments::from_args();
    let cfg = Config::from_env();
    let cases = suites::all_cases(|f| std::fs::read_to_string(cfg.test_dir.join(f)).ok());
    let trials = cases
        .into_iter()
        .map(|case| {
            let cfg = cfg.clone();
            let (mut name, kind) = (case.display_name(), case.suite);
            let reason = env_skip_reason(&cfg, &case);
            if let Some(r) = reason {
                name.push_str(&format!(" [ignored: {r}]"));
            }
            Trial::test(name, move || runner::check(&cfg, &case).map_err(Failed::from))
                .with_kind(kind)
                .with_ignored_flag(reason.is_some())
        })
        .collect();
    libtest_mimic::run(&args, trials).exit();
}

/// Environment-only skips. Python-interop cases (`from python import`, pybridge) are in
/// scope and run whenever a libpython is available (`CODON_PYTHON` or auto-discovered via
/// `python3`); they are skipped only if none is found. GPU cases need CUDA libdevice.
fn env_skip_reason(cfg: &Config, case: &typhon_testsuite::suites::Case) -> Option<&'static str> {
    let needs_python = case.file.ends_with("pybridge.codon")
        || case.file.ends_with("stdlib/random_test.codon")
        || case.file.ends_with("test_python.codon");
    if needs_python && cfg.python.is_none() {
        return Some("needs libpython (none found; set CODON_PYTHON)");
    }
    if case.suite == "gpu" && !std::path::Path::new("/usr/local/cuda/nvvm/libdevice").exists() {
        return Some("needs CUDA libdevice");
    }
    None
}
