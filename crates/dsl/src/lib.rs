//! typhon-dsl: port of `codon/dsl` (DSL plugin base trait + plugin manager).
//! See `API.md` and `PLUGINS.md`.

pub mod abi;
pub mod dsl;
pub mod plugins;
pub mod semver_range;

pub use dsl::{BlockKeyword, DefaultDsl, Dsl, ExprKeyword, Info, KeywordCallback};
pub use plugins::{LIB_EXT, Plugin, PluginManager, parse_info};
pub use semver_range::VersionRange;
