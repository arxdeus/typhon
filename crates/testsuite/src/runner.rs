//! Subprocess execution of a single case (replaces the fork()ing `SeqTest::runInChildProcess`).
//!
//! Differences from the in-process C++ harness (all deliberate, see README):
//! * `#%%` snippets are written to a temp dir as `<file>` with `start_line` blank lines
//!   prepended, so reported line numbers equal the real file's. The original file's sibling
//!   entries are symlinked into the temp dir so relative imports (`import a`) still resolve.
//! * `options->test` / `standalone` cannot be set through the CLI; the CLI's `-debug` is used.
//! * Compile-error text: the C++ child printed bare messages to stdout; the CLI prints
//!   decorated diagnostics on stderr. We strip ANSI and extract the `error:` messages.
//! * The C++ harness truncates captured stdout to 64 KiB; we do not.
use crate::config::Config;
use crate::expect;
use crate::suites::Case;
use regex::Regex;
use std::io::Read;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

pub struct Outcome {
    pub code: Option<i32>,
    pub signal: Option<i32>,
    pub stdout: String,
    pub stderr: String,
    pub timed_out: bool,
}

fn strip_ansi(s: &str) -> String {
    Regex::new("\x1b\\[[0-9;]*[A-Za-z]").unwrap().replace_all(s, "").into_owned()
}

/// Extract bare diagnostic messages (`file:line (col): error: msg`) from CLI stderr.
pub fn extract_errors(stderr: &str) -> Vec<String> {
    // `file:line (col): error: msg`, or group continuations `├─ error: msg` / `╰─ error: msg`.
    let re = Regex::new(r"^(?:[├│╰]─ )?(?:.*?:\d+ \([0-9-]+\): )?error: (.*)$").unwrap();
    strip_ansi(stderr)
        .lines()
        .filter_map(|l| re.captures(l).map(|c| c[1].to_string()))
        .collect()
}

fn run_process(mut cmd: Command, timeout: Duration) -> std::io::Result<Outcome> {
    cmd.stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped());
    let mut child = cmd.spawn()?;
    let mut so = child.stdout.take().unwrap();
    let mut se = child.stderr.take().unwrap();
    let t1 = std::thread::spawn(move || { let mut b = Vec::new(); let _ = so.read_to_end(&mut b); b });
    let t2 = std::thread::spawn(move || { let mut b = Vec::new(); let _ = se.read_to_end(&mut b); b });
    let start = Instant::now();
    let mut timed_out = false;
    let status = loop {
        if let Some(s) = child.try_wait()? { break s; }
        if start.elapsed() > timeout {
            timed_out = true;
            let _ = child.kill();
            break child.wait()?;
        }
        std::thread::sleep(Duration::from_millis(10));
    };
    #[cfg(unix)]
    let signal = std::os::unix::process::ExitStatusExt::signal(&status);
    #[cfg(not(unix))]
    let signal = None;
    Ok(Outcome {
        code: status.code(),
        signal,
        stdout: String::from_utf8_lossy(&t1.join().unwrap_or_default()).into_owned(),
        stderr: String::from_utf8_lossy(&t2.join().unwrap_or_default()).into_owned(),
        timed_out,
    })
}

/// Materialise a snippet case; returns (temp dir guard, path of the snippet file).
fn write_snippet(cfg: &Config, case: &Case) -> std::io::Result<(tempfile::TempDir, PathBuf)> {
    let src = cfg.test_dir.join(&case.file);
    let dir = tempfile::tempdir()?;
    // Sibling modules (`import a`) must resolve relative to the snippet, and symlinks would
    // canonicalise module names to absolute paths, so copy the (small) directory tree.
    if let Some(parent) = src.parent() {
        copy_dir(parent, dir.path())?;
    }
    // Keep the original file name (tests print/compare it, and `import test_x` self-imports).
    let path = dir.path().join(src.file_name().unwrap());
    let _ = std::fs::remove_file(&path);
    let mut body = "\n".repeat(case.start_line);
    body.push_str(&case.code);
    std::fs::write(&path, body)?;
    Ok((dir, path))
}

pub fn execute(cfg: &Config, case: &Case) -> Result<Outcome, String> {
    if !cfg.bin.exists() {
        return Err(format!("compiler binary not found: {} (set TYPHON_BIN)", cfg.bin.display()));
    }
    let is_snippet = !case.name.is_empty();
    let (_guard, file): (Option<tempfile::TempDir>, PathBuf) = if is_snippet {
        let (g, p) = write_snippet(cfg, case).map_err(|e| e.to_string())?;
        (Some(g), p)
    } else {
        (None, cfg.test_dir.join(&case.file))
    };

    let mut cmd = Command::new(&cfg.bin);
    let mut compile_out = None;
    if case.run {
        cmd.arg("run");
    } else {
        // compile-only (GPU suite): emit LLVM IR to a throwaway file
        let out = tempfile::Builder::new().suffix(".ll").tempfile().map_err(|e| e.to_string())?;
        cmd.args(["build", "-llvm", "-o"]).arg(out.path());
        compile_out = Some(out);
    }
    cmd.arg(if case.debug { "-debug" } else { "-release" });
    cmd.arg(if case.py_numerics { "-numerics=py" } else { "-numerics=c" });
    cmd.arg(&file);

    // CODON_TEST_FLAGS = 1 + barebones (C++: testFlags = 1 + get<5>)
    cmd.env("CODON_TEST_FLAGS", (1 + case.barebones as i32).to_string());
    // C++ runCompileAndRun: options->test = true (release pipeline in debug too,
    // plus TestOutliner/TestInliner/EscapeValidator).
    cmd.env("TYPHON_TEST_MODE", "1");
    cmd.env("CODON_PATH", &cfg.stdlib);
    cmd.env("TYPHON_STDLIB_DIR", &cfg.stdlib);
    let mut py = cfg.test_dir.join("python").into_os_string();
    if let Some(existing) = std::env::var_os("PYTHONPATH").filter(|v| !v.is_empty()) {
        py.push(if cfg!(windows) { ";" } else { ":" });
        py.push(existing);
    }
    cmd.env("PYTHONPATH", py);
    if let Some(lib) = &cfg.python {
        cmd.env("CODON_PYTHON", lib);
    }
    // C++ ran from the repo root: tests open `build/...` for scratch files and
    // `test/stdlib/data/...` for fixtures. Emulate with a per-case scratch cwd.
    let cwd = tempfile::tempdir().map_err(|e| e.to_string())?;
    std::fs::create_dir(cwd.path().join("build")).map_err(|e| e.to_string())?;
    #[cfg(unix)]
    std::os::unix::fs::symlink(&cfg.test_dir, cwd.path().join("test")).map_err(|e| e.to_string())?;
    // C++ tests ran from the build dir, where `./libcodonrt.<ext>` resolves
    // (types/import_c_dylib*). Expose the runtime next to the scratch cwd.
    #[cfg(unix)]
    if let Some(dir) = cfg.bin.parent() {
        for ext in ["dylib", "so"] {
            let rt = dir.join(format!("libcodonrt.{ext}"));
            if rt.exists() {
                let _ = std::os::unix::fs::symlink(&rt, cwd.path().join(format!("libcodonrt.{ext}")));
            }
        }
    }
    cmd.current_dir(cwd.path());
    let out = run_process(cmd, cfg.timeout).map_err(|e| format!("spawn failed: {e}"));
    drop(compile_out);
    drop(cwd);
    out
}

/// Full check of a case (port of `TEST_P(SeqTest, Run)`).
pub fn check(cfg: &Config, case: &Case) -> Result<(), String> {
    let o = execute(cfg, case)?;
    if o.timed_out {
        return Err(format!("timed out after {:?}\nstdout:\n{}\nstderr:\n{}", cfg.timeout, o.stdout, o.stderr));
    }
    let Some(code) = o.code else {
        return Err(format!("child killed by signal {:?}\nstdout:\n{}\nstderr:\n{}", o.signal, o.stdout, o.stderr));
    };
    let source = if case.name.is_empty() {
        std::fs::read_to_string(cfg.test_dir.join(&case.file)).map_err(|e| e.to_string())?
    } else {
        case.code.clone()
    };
    let (expects, is_error) = expect::find_expects(&source);
    let mut output = o.stdout.clone();
    if code != 0 {
        // compile failure: the C++ child printed bare messages on stdout
        for m in extract_errors(&o.stderr) {
            output.push_str(&m);
            output.push('\n');
        }
    }
    if code != is_error as i32 {
        return Err(format!("exit code {} (expected {})\nstdout:\n{}\nstderr:\n{}", code, is_error as i32, o.stdout, o.stderr));
    }
    let mut errs = String::new();
    if output.contains("TEST FAILED") {
        errs += &format!("output contains TEST FAILED\n{output}\n");
    }
    if let Err(e) = expect::compare(&output, &expects, is_error) {
        errs += &format!("{e}\noutput:\n{output}\nstderr:\n{}", o.stderr);
    }
    if errs.is_empty() { Ok(()) } else { Err(errs) }
}


fn copy_dir(from: &std::path::Path, to: &std::path::Path) -> std::io::Result<()> {
    for e in std::fs::read_dir(from)?.flatten() {
        let t = e.file_type()?;
        let dst = to.join(e.file_name());
        if t.is_dir() {
            std::fs::create_dir_all(&dst)?;
            copy_dir(&e.path(), &dst)?;
        } else if t.is_file() {
            std::fs::copy(e.path(), dst)?;
        }
    }
    Ok(())
}
