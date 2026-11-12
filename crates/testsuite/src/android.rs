//! Cross mode: build on the host with `-target <TYPHON_TEST_TARGET>`, push to an Android
//! device via adb, run there in `/data/local/tmp/typhon-e2e/<unique>/` with the same cwd layout
//! as the host runner (`build/` scratch dir, `test` symlink to the once-pushed test data).
use crate::config::Config;
use crate::runner::{run_process, Outcome};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex, OnceLock};

pub const ROOT: &str = "/data/local/tmp/typhon-e2e";

pub struct Target {
    pub triple: String,
    pub serial: Option<String>,
    pub adb: PathBuf,
    pub jobs: usize,
}

pub fn target() -> Option<&'static Target> {
    static T: OnceLock<Option<Target>> = OnceLock::new();
    T.get_or_init(|| {
        let triple = std::env::var("TYPHON_TEST_TARGET").ok().filter(|s| !s.is_empty())?;
        let serial = std::env::var("TYPHON_ANDROID_SERIAL").ok().filter(|s| !s.is_empty());
        let adb = std::env::var_os("ADB").map(PathBuf::from).unwrap_or_else(|| {
            std::env::var_os("ANDROID_HOME")
                .map(|h| PathBuf::from(h).join("platform-tools/adb"))
                .unwrap_or_else(|| PathBuf::from("adb"))
        });
        let jobs = std::env::var("TYPHON_ANDROID_JOBS").ok().and_then(|s| s.parse().ok()).unwrap_or(4);
        Some(Target { triple, serial, adb, jobs })
    })
    .as_ref()
}

fn adb(t: &Target) -> Command {
    let mut c = Command::new(&t.adb);
    if let Some(s) = &t.serial {
        c.args(["-s", s]);
    }
    c
}

fn adb_ok(t: &Target, args: &[&str]) -> Result<(), String> {
    let o = adb(t).args(args).output().map_err(|e| format!("adb: {e}"))?;
    if o.status.success() { Ok(()) } else {
        Err(format!("adb {:?} failed: {}{}", args, String::from_utf8_lossy(&o.stdout), String::from_utf8_lossy(&o.stderr)))
    }
}

/// Push the test data dir once per process (all cases share it read-only).
fn setup(cfg: &Config, t: &Target) -> Result<(), String> {
    static DONE: OnceLock<Result<(), String>> = OnceLock::new();
    DONE.get_or_init(|| {
        adb_ok(t, &["shell", &format!("rm -rf {ROOT} && mkdir -p {ROOT}/cases")])?;
        adb_ok(t, &["push", "-q", &cfg.test_dir.to_string_lossy(), &format!("{ROOT}/test")])
    })
    .clone()
}

/// Remove everything we pushed (called once after the harness finishes).
pub fn teardown() {
    if let Some(t) = target() {
        let _ = adb(t).args(["shell", &format!("rm -rf {ROOT}")]).output();
    }
}

struct Sem { n: Mutex<usize>, cv: Condvar }
fn sem() -> &'static Sem {
    static S: OnceLock<Sem> = OnceLock::new();
    S.get_or_init(|| Sem { n: Mutex::new(0), cv: Condvar::new() })
}
struct Permit;
impl Permit {
    fn get(max: usize) -> Permit {
        let s = sem();
        let mut n = s.n.lock().unwrap();
        while *n >= max { n = s.cv.wait(n).unwrap(); }
        *n += 1;
        Permit
    }
}
impl Drop for Permit {
    fn drop(&mut self) {
        let s = sem();
        *s.n.lock().unwrap() -= 1;
        s.cv.notify_one();
    }
}

/// Build `file` for the target; on compile failure return the compiler's outcome.
pub fn execute(cfg: &Config, t: &Target, file: &Path, debug: bool, py_numerics: bool, flags: i32) -> Result<Outcome, String> {
    setup(cfg, t)?;
    let tmp = tempfile::tempdir().map_err(|e| e.to_string())?;
    let exe = tmp.path().join("prog");
    let mut cmd = Command::new(&cfg.bin);
    cmd.arg("build").arg(if debug { "-debug" } else { "-release" });
    cmd.arg(if py_numerics { "-numerics=py" } else { "-numerics=c" });
    cmd.args(["-exe", "-target", &t.triple, "-o"]).arg(&exe).arg(file);
    cmd.env("TYPHON_TEST_FLAGS", flags.to_string());
    cmd.env("TYPHON_TEST_MODE", "1");
    cmd.env("TYPHON_PATH", &cfg.stdlib);
    cmd.env("TYPHON_STDLIB_DIR", &cfg.stdlib);
    cmd.current_dir(tmp.path());
    let built = run_process(cmd, cfg.timeout).map_err(|e| format!("spawn failed: {e}"))?;
    if built.timed_out || built.code != Some(0) || !exe.exists() {
        // compile-time output (errors) is what the host `run` would print too
        return Ok(built);
    }

    static ID: AtomicUsize = AtomicUsize::new(0);
    let dir = format!("{ROOT}/cases/{}-{}", std::process::id(), ID.fetch_add(1, Ordering::Relaxed));
    let _p = Permit::get(t.jobs);
    adb_ok(t, &["shell", &format!("mkdir -p {dir}/build && ln -s ../../test {dir}/test")])?;
    adb_ok(t, &["push", "-q", &exe.to_string_lossy(), &format!("{dir}/prog")])?;
    let mut run = adb(t);
    // TMPDIR: bionic has no /tmp; give programs a writable scratch dir.
    run.args(["shell", "-T", &format!("cd {dir} && chmod 755 prog && TMPDIR={dir}/build ./prog")]);
    let out = run_process(run, cfg.timeout).map_err(|e| format!("adb spawn failed: {e}"));
    let _ = adb(t).args(["shell", &format!("rm -rf {dir}")]).output();
    out
}
