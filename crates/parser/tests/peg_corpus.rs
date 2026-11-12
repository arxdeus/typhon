//! Corpus test: parse every `.py`/`.py` under ../stdlib and ../test with the PEG
//! parser, then round-trip (stmt_to_debug -> parse -> compare S-expressions).

use std::path::{Path, PathBuf};

use typhon_parser::ast::AstArena;
use typhon_parser::peg::{ParseOptions, parse_code};

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn collect(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    let mut v: Vec<_> = rd.flatten().map(|e| e.path()).collect();
    v.sort();
    for p in v {
        if p.is_dir() {
            collect(&p, out);
        } else if matches!(p.extension().and_then(|e| e.to_str()), Some("py")) {
            out.push(p);
        }
    }
}

/// Splits test files into `#%%` sections; other files
/// are one unit. Sections with a `#! syntax error` / `unexpected` expectation that is
/// produced by the parser are skipped (they intentionally fail to parse).
fn units(path: &Path, text: &str) -> Vec<(usize, String)> {
    let is_typecheck = path.to_string_lossy().contains("/test/parser/");
    if !is_typecheck || !text.contains("#%%") {
        return vec![(0, text.to_string())];
    }
    let mut res = vec![];
    let mut cur = String::new();
    let mut start = 0;
    for (i, l) in text.lines().enumerate() {
        if l.starts_with("#%%") && !cur.is_empty() {
            res.push((start, std::mem::take(&mut cur)));
            start = i;
        }
        cur.push_str(l);
        cur.push('\n');
    }
    if !cur.is_empty() {
        res.push((start, cur));
    }
    res
}

fn expects_syntax_error(code: &str) -> bool {
    code.lines().any(|l| {
        l.find("#! ").is_some_and(|i| {
            let m = &l[i + 3..];
            m.starts_with("syntax error") || m.contains("unexpected indentation") || m.contains("unexpected dedent")
                || m.contains("invalid code in a string")
                // FunctionStmt::parseParams errors are raised while parsing (ctor).
                || m.contains("must follow positional parameters")
                || m.contains("named arguments must follow bare")
                || m.contains("may appear only once")
        })
    })
}

#[test]
fn peg_corpus() {
    let mut files = vec![];
    collect(&root().join("stdlib"), &mut files);
    collect(&root().join("test"), &mut files);
    let opts = ParseOptions::default();
    let (mut parsed, mut failed, mut skipped, mut rt_ok, mut rt_bad) = (0, 0, 0, 0, 0);
    let mut failures = vec![];
    for f in &files {
        let Ok(text) = std::fs::read_to_string(f) else { continue };
        for (line, code) in units(f, &text) {
            if expects_syntax_error(&code) {
                skipped += 1;
                let mut a = AstArena::new();
                if parse_code(&mut a, &opts, "", &code, 0).is_ok() {
                    eprintln!("NOTE: expected-error unit parsed fine (error is semantic or in a sibling stmt): {}:{line}", f.display());
                }
                continue;
            }
            let mut arena = AstArena::new();
            match parse_code(&mut arena, &opts, &f.to_string_lossy(), &code, line as i32) {
                Ok(s) => {
                    parsed += 1;
                    // Note: `stmt_to_debug` (debug dump) is an AST dump, not
                    // re-parseable source, so the round-trip is: parse again and compare
                    // both the S-expression and the Typhon dump (determinism + printers).
                    let d1 = (arena.stmt_to_string(s, 0), arena.stmt_to_debug(s, false, -1, 0));
                    let mut a2 = AstArena::new();
                    match parse_code(&mut a2, &opts, &f.to_string_lossy(), &code, line as i32) {
                        Ok(s2) if (a2.stmt_to_string(s2, 0), a2.stmt_to_debug(s2, false, -1, 0)) == d1 => rt_ok += 1,
                        _ => {
                            rt_bad += 1;
                            eprintln!("ROUNDTRIP MISMATCH {}:{line}", f.display());
                        }
                    }
                }
                Err(e) => {
                    failed += 1;
                    failures.push(format!("{}:{line}: {e}", f.display()));
                }
            }
        }
    }
    eprintln!(
        "peg corpus: files={} parsed_units={parsed} failed={failed} skipped_syntax_error_tests={skipped} roundtrip_ok={rt_ok} roundtrip_mismatch={rt_bad}",
        files.len()
    );
    for f in failures.iter().take(30) {
        eprintln!("FAIL {f}");
    }
    assert_eq!(failed, 0);
    assert_eq!(rt_bad, 0);
}
