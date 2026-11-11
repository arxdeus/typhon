//! Wiring of the parser lanes into [`Cache::hooks`]: grammar (`peg`), scoping,
//! translate, plus the doc generator.
//!
//! Each lane exposes its own entry point; this module is the single place that
//! connects them so the rest of the driver only talks to `Cache::hooks`.

use typhon_common::Result;
use typhon_parser::cache::Cache;

/// Install all parser-lane hooks into `cache`.
pub fn install(cache: &mut Cache) {
    install_grammar(cache);
    typhon_parser::visitors::scoping::install_hooks(cache);
    install_translate(cache);
}

fn install_grammar(_cache: &mut Cache) {
    // Filled in when `typhon_parser::peg` lands (parse_code / parse_file / parse_expr).
}

fn install_translate(_cache: &mut Cache) {
    // Filled in when `typhon_parser::visitors::translate` lands.
}

/// `DocVisitor::apply(argv0, files)` -> JSON string.
pub fn docgen(argv0: &str, files: &[String]) -> Result<String> {
    let mut cache = Cache::new(Some(crate::stdlib::default_filesystem(argv0)));
    install(&mut cache);
    let j = typhon_parser::visitors::doc::DocVisitor::apply(argv0, files, cache)?;
    Ok(typhon_parser::visitors::doc::json_to_string(&j))
}
