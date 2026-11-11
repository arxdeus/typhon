//! Wiring of the parser lanes into [`Cache::hooks`]: grammar (`peg`), scoping,
//! translate, plus the doc generator.
//!
//! Each lane exposes its own entry point; this module is the single place that
//! connects them so the rest of the driver only talks to `Cache::hooks`.

use typhon_common::{CompilerError, Result};
use typhon_parser::cache::Cache;

/// Install all parser-lane hooks into `cache`.
pub fn install(cache: &mut Cache) {
    install_grammar(cache);
    install_scoping(cache);
    install_translate(cache);
}

fn install_grammar(_cache: &mut Cache) {
    // Filled in when `typhon_parser::peg` lands (parse_code / parse_file).
}

fn install_scoping(_cache: &mut Cache) {
    // Filled in when `typhon_parser::visitors::scoping` lands.
}

fn install_translate(_cache: &mut Cache) {
    // Filled in when `typhon_parser::visitors::translate` lands.
}

/// `DocVisitor::apply(argv0, files)` -> JSON string.
pub fn docgen(_argv0: &str, _files: &[String]) -> Result<String> {
    Err(CompilerError::Io("doc generation is not available in this build".into()))
}
