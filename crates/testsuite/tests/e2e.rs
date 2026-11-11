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
            let (name, kind) = (case.display_name(), case.suite);
            Trial::test(name, move || runner::check(&cfg, &case).map_err(Failed::from)).with_kind(kind)
        })
        .collect();
    libtest_mimic::run(&args, trials).exit();
}
