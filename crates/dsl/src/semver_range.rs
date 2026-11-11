//! npm-style semver ranges (C++ uses Neargye `semver::try_parse_range`).
//!
//! PORT-NOTE: Rust `semver::VersionReq` lacks `||` and uses commas between comparators, so
//! ranges are split on `||` (OR) and whitespace-separated comparators (AND) are joined with
//! commas. Pre-release matching follows the Rust `semver` crate rules.

use semver::{Version, VersionReq};

#[derive(Debug, Clone)]
pub struct VersionRange {
    alternatives: Vec<VersionReq>,
}

impl VersionRange {
    /// Returns `None` for invalid ranges (including blank/empty alternatives).
    pub fn parse(s: &str) -> Option<Self> {
        let mut alternatives = Vec::new();
        for part in s.split("||") {
            let mut comps: Vec<String> = Vec::new();
            let mut pending_op = String::new();
            for tok in part.split_whitespace() {
                if tok.chars().all(|c| matches!(c, '<' | '>' | '=' | '~' | '^')) {
                    pending_op.push_str(tok);
                    continue;
                }
                comps.push(format!("{}{tok}", std::mem::take(&mut pending_op)));
            }
            if comps.is_empty() || !pending_op.is_empty() {
                return None;
            }
            alternatives.push(VersionReq::parse(&comps.join(", ")).ok()?);
        }
        Some(VersionRange { alternatives })
    }

    pub fn contains(&self, v: &Version) -> bool {
        self.alternatives.iter().any(|r| r.matches(v))
    }
}
