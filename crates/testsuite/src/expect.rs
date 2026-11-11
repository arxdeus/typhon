//! Expectation parsing and output comparison (`findExpectOnLine`, `findExpects`, `splitLines`).

/// Returns `(is_error, text)`; text empty if the line has no marker.
/// Marker priority matches C++: `# EXPECT: `, then `#: `, then `#! ` (error).
pub fn find_expect_on_line(raw: &str) -> (bool, String) {
    let line = raw.strip_suffix('\r').unwrap_or(raw);
    for (is_err, m) in [(false, "# EXPECT: "), (false, "#: "), (true, "#! ")] {
        if let Some(pos) = line.find(m) {
            return (is_err, line[pos + m.len()..].to_string());
        }
    }
    (false, String::new())
}

/// Collect expected lines and whether the case is an expected compile error.
pub fn find_expects(source: &str) -> (Vec<String>, bool) {
    let mut out = Vec::new();
    let mut is_error = false;
    for l in source.lines() {
        let (e, t) = find_expect_on_line(l);
        if !t.is_empty() {
            out.push(t);
            is_error |= e;
        }
    }
    (out, is_error)
}

/// `getline`-style split (no trailing empty element), dropping `\r`.
pub fn split_lines(output: &str) -> Vec<String> {
    output.lines().map(|l| l.strip_suffix('\r').unwrap_or(l).to_string()).collect()
}

/// Compare `output` with `expects`. Error cases compare by prefix (as in C++).
pub fn compare(output: &str, expects: &[String], is_error: bool) -> Result<(), String> {
    if expects.is_empty() {
        return Ok(());
    }
    let results = split_lines(output);
    let mut errs = String::new();
    for i in 0..results.len().min(expects.len()) {
        let ok = if is_error { results[i].starts_with(expects[i].as_str()) } else { results[i] == expects[i] };
        if !ok {
            errs += &format!("line {}: expected {:?}, got {:?}\n", i + 1, expects[i], results[i]);
        }
    }
    if results.len() != expects.len() {
        errs += &format!("expected {} output lines, got {}\n", expects.len(), results.len());
    }
    if errs.is_empty() { Ok(()) } else { Err(errs) }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn markers() {
        assert_eq!(find_expect_on_line("print 1 #: 1"), (false, "1".into()));
        assert_eq!(find_expect_on_line("x #! boom\r"), (true, "boom".into()));
        assert_eq!(find_expect_on_line("nothing").1, "");
    }
}
