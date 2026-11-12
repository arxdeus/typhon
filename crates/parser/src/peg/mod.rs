//!
//! Driver functions mirroring `peg.h`: [`parse_code`], [`parse_expr`], [`parse_file`],
//! [`parse_openmp`].
//!
//! Indentation handling: like cpp-peglib, INDENT/SAMEDENT/DEDENT are zero-width-ish
//! rules over `[ \t]*` whose predicates consult (and INDENT/DEDENT mutate) the
//! indentation stack held in [`ParseState`] (`RefCell<Vec<usize>>`). They are used under
//! `&` lookahead exactly as in grammar.peg; rust-peg executes custom closures inside
//! lookaheads, so the stack side effects match cpp-peglib (which also runs actions of
//! `__NOPACKRAT` rules inside `&`). These rules are never `#[cache]`d.

pub mod grammar;
pub mod openmp;
pub mod state;

use std::path::Path;

use typhon_common::{CompilerError, ErrorMessage, ParserErrors, Result, SrcInfo};

use crate::ast::*;
pub use state::ParseOptions;
use state::ParseState;

fn finish<T>(
    st: &ParseState<'_>,
    r: std::result::Result<T, peg::error::ParseError<peg::str::LineCol>>,
    file: &str,
) -> Result<T> {
    if let Some(e) = st.fatal.borrow_mut().take() {
        return Err(e);
    }
    r.map_err(|e| st.syntax_error(e.location.offset, file))
}

/// `parseCode(cache, file, code, line_offset)`: parses a code block (rule `program`).
pub fn parse_code(
    arena: &mut AstArena,
    opts: &ParseOptions,
    file: &str,
    code: &str,
    line_offset: i32,
) -> Result<StmtId> {
    let code = format!("{code}\n");
    let st = ParseState::new(arena, opts, file, &code, line_offset, 0);
    let r = grammar::python::program(&code, &st);
    finish(&st, r, file)
}

/// `parseExpr(cache, code, offset)`: parses an f-string expression with an optional
/// `=`, `!conversion` and `:format_spec` (rule `fstring`).
pub fn parse_expr(
    arena: &mut AstArena,
    opts: &ParseOptions,
    code: &str,
    offset: &SrcInfo,
) -> Result<(ExprId, FormatSpec)> {
    let code = code.trim_start_matches([' ', '\t', '\n', '\r', '\x0b', '\x0c']).to_string();
    let file = offset.file.as_str().to_string();
    let st = ParseState::new(arena, opts, &file, &code, offset.line, offset.col);
    let r = grammar::python::fstring(&code, &st);
    finish(&st, r, &file)
}

/// `parseFile(cache, file)`. Returns the AST and the file lines (the reference implementation stores them into
/// `cache->imports[file].content`; the caller does that here).
pub fn parse_file(
    arena: &mut AstArena,
    opts: &ParseOptions,
    fs: &dyn typhon_common::fs::FileSystem,
    file: &str,
) -> Result<(StmtId, Vec<String>)> {
    let lines = fs.read_lines(Path::new(file))?;
    let mut code = lines.join("\n");
    code.push('\n');
    let st = ParseState::new(arena, opts, file, &code, 0, 0);
    let r = grammar::python::program(&code, &st);
    let s = finish(&st, r, file)?;
    Ok((s, lines))
}

/// `parseOpenMP(cache, code, loc)`.
pub fn parse_openmp(
    arena: &mut AstArena,
    opts: &ParseOptions,
    code: &str,
    loc: &SrcInfo,
) -> Result<Vec<CallArg>> {
    let st = ParseState::new(arena, opts, "", code, 0, 0);
    match openmp::omp::pragma(code, &st) {
        Ok(v) => Ok(v),
        Err(e) => {
            let msg = match st.message.borrow().clone() {
                Some((_, m)) => m,
                None => {
                    let tok = state::heuristic_error_token(code, e.location.offset);
                    if tok.is_empty() {
                        "syntax error.".to_string()
                    } else {
                        // Note: cpp-peglib also lists expected tokens here; they are
                        // not reproducible from rust-peg and only the prefix is kept.
                        format!("syntax error, unexpected '{tok}'")
                    }
                }
            };
            Err(CompilerError::Parser(ParserErrors::from_message(ErrorMessage::new(
                format!("openmp: {msg}"),
                SrcInfo::new(loc.file.as_str(), loc.line, loc.col, 0),
            ))))
        }
    }
}
