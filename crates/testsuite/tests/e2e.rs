//! End-to-end `.py` suites. Run: `cargo test -p typhon-testsuite --test e2e -- core/`
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
    let conclusion = libtest_mimic::run(&args, trials);
    typhon_testsuite::android::teardown();
    conclusion.exit();
}

/// Environment-only skips: GPU cases need CUDA libdevice.
fn env_skip_reason(_cfg: &Config, case: &typhon_testsuite::suites::Case) -> Option<&'static str> {
    if case.suite == "gpu" && !std::path::Path::new("/usr/local/cuda/nvvm/libdevice").exists() {
        return Some("needs CUDA libdevice");
    }
    None
}
