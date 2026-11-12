//! Wiring of the parser lanes into [`Cache::hooks`]: grammar (`peg`), scoping,
//! translate, plus the doc generator.
//!
//! Each lane exposes its own entry point; this module is the single place that
//! connects them so the rest of the driver only talks to `Cache::hooks`.

use std::rc::Rc;

use typhon_common::{CompilerError, Result, SrcInfo};
use typhon_parser::ast::FormatSpec;
use typhon_parser::cache::{Cache, ImportModule};
use typhon_parser::ids::ExprId;
use typhon_parser::peg;

/// Install all parser-lane hooks into `cache`.
pub fn install(cache: &mut Cache) {
    install_grammar(cache);
    typhon_parser::visitors::scoping::install_hooks(cache);
    typhon_parser::visitors::translate::install_hooks(cache);
}

fn parse_expr_hook(cache: &mut Cache, code: &str, offset: &SrcInfo) -> Result<(ExprId, FormatSpec)> {
    peg::parse_expr(&mut cache.ast, code, offset)
}

fn install_grammar(cache: &mut Cache) {
    cache.hooks.parse_code = Some(Rc::new(|cache: &mut Cache, file: &str, code: &str, line: i32| {
            peg::parse_code(&mut cache.ast, file, code, line)
    }));
    cache.hooks.parse_file = Some(Rc::new(|cache: &mut Cache, file: &str| {
            let fs = cache.fs.clone().ok_or_else(|| CompilerError::Io("no filesystem".into()))?;
        let fsb = fs.borrow();
        let (stmt, lines) = peg::parse_file(&mut cache.ast, &*fsb, file)?;
        cache.imports.entry(file.to_string()).or_insert_with(ImportModule::new).content = lines;
        Ok(stmt)
    }));
    typhon_parser::visitors::scoping::set_parse_expr_hook(parse_expr_hook);
}

/// `DocVisitor::apply(argv0, files)` -> JSON string.
pub fn docgen(argv0: &str, files: &[String]) -> Result<String> {
    let mut cache = Cache::new(Some(crate::stdlib::default_filesystem(argv0)));
    install(&mut cache);
    let j = typhon_parser::visitors::doc::DocVisitor::apply(argv0, files, cache)?;
    Ok(typhon_parser::visitors::doc::json_to_string(&j))
}
