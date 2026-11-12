//! Diagnostics printing. Port of `compilationError` / `compilationWarning` /
//! `MessageGroupPos`.
//!
//! Output format is byte-compatible with the reference implementation (ANSI bold header,
//! `file:line (col-end): error: msg`), so the
//! test-suite expectations on stderr carry over. `codespan-reporting` is available
//! for richer rendering via [`render_codespan`].

use crate::error::{CompilerError, ParserErrors};
use std::io::Write;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum MessageGroupPos {
    #[default]
    None,
    Head,
    Mid,
    Last,
}

fn compilation_message(
    header: &str,
    msg: &str,
    file: &str,
    line: i32,
    col: i32,
    len: i32,
    error_code: i32,
    pos: MessageGroupPos,
) {
    let mut out = String::new();
    match pos {
        MessageGroupPos::Mid => out.push_str("├─ "),
        MessageGroupPos::Last => out.push_str("╰─ "),
        _ => {}
    }
    out.push_str("\x1b[1m");
    if !file.is_empty() {
        out.push_str(if file == "-" { "<stdin>" } else { file });
    }
    if line > 0 {
        out.push_str(&format!(":{line}"));
    }
    if col > 0 {
        out.push_str(&format!(" ({col}"));
        if len > 0 {
            out.push_str(&format!("-{})", col + len));
        } else {
            out.push(')');
        }
    }
    if !file.is_empty() {
        out.push_str(": ");
    }
    let _ = error_code;
    out.push_str(&format!("{header}\x1b[1m {msg}\x1b[0m\n"));
    let _ = std::io::stderr().write_all(out.as_bytes());
}

#[allow(clippy::too_many_arguments)]
pub fn compilation_error(
    msg: &str,
    file: &str,
    line: i32,
    col: i32,
    len: i32,
    error_code: i32,
    terminate: bool,
    pos: MessageGroupPos,
) {
    compilation_message("\x1b[1;31merror:\x1b[0m", msg, file, line, col, len, error_code, pos);
    if terminate {
        std::process::exit(1);
    }
}

#[allow(clippy::too_many_arguments)]
pub fn compilation_warning(
    msg: &str,
    file: &str,
    line: i32,
    col: i32,
    len: i32,
    error_code: i32,
    terminate: bool,
    pos: MessageGroupPos,
) {
    compilation_message("\x1b[1;33mwarning:\x1b[0m", msg, file, line, col, len, error_code, pos);
    if terminate {
        std::process::exit(1);
    }
}

/// Print every error in `errs` in the standard compiler format.
pub fn report_parser_errors(errs: &ParserErrors) {
    for bt in &errs.errors {
        let n = bt.trace.len();
        for (i, m) in bt.trace.iter().enumerate() {
            let pos = if n == 1 {
                MessageGroupPos::None
            } else if i == 0 {
                MessageGroupPos::Head
            } else if i + 1 == n {
                MessageGroupPos::Last
            } else {
                MessageGroupPos::Mid
            };
            compilation_error(
                &m.msg,
                m.file(),
                m.loc.line,
                m.loc.col,
                m.loc.len,
                m.error_code,
                false,
                pos,
            );
        }
    }
}

pub fn report(err: &CompilerError) {
    match err {
        CompilerError::Parser(p) => report_parser_errors(p),
        other => compilation_error(&other.to_string(), "", 0, 0, 0, -1, false, MessageGroupPos::None),
    }
}

/// Render an error with source snippets using `codespan-reporting`.
pub fn render_codespan(errs: &ParserErrors, source_of: impl Fn(&str) -> Option<String>) -> String {
    use codespan_reporting::diagnostic::{Diagnostic, Label};
    use codespan_reporting::files::SimpleFiles;
    use codespan_reporting::term::{self, termcolor::Buffer};

    let mut files = SimpleFiles::new();
    let mut buf = Buffer::no_color();
    let config = term::Config::default();
    for bt in &errs.errors {
        let Some(head) = bt.trace.first() else { continue };
        let mut diag = Diagnostic::error().with_message(head.msg.clone());
        if let Some(src) = source_of(head.file()) {
            let id = files.add(head.file().to_string(), src.clone());
            let start = line_col_to_offset(&src, head.loc.line, head.loc.col);
            let end = (start + head.loc.len.max(1) as usize).min(src.len());
            diag = diag.with_labels(vec![Label::primary(id, start..end)]);
        }
        let notes = bt.trace.iter().skip(1).map(|m| m.to_string()).collect();
        diag = diag.with_notes(notes);
        let _ = term::emit_to_write_style(&mut buf, &config, &files, &diag);
    }
    String::from_utf8_lossy(buf.as_slice()).into_owned()
}

fn line_col_to_offset(src: &str, line: i32, col: i32) -> usize {
    let mut off = 0usize;
    for (i, l) in src.split_inclusive('\n').enumerate() {
        if i as i32 + 1 == line {
            return (off + (col.max(1) as usize - 1)).min(src.len());
        }
        off += l.len();
    }
    src.len()
}
