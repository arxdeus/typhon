//! Port of `codon/cir/transform/rewrite.h`.

use indexmap::IndexMap;

use crate::ids::ValueId;
use crate::module::Module;

/// Port of `transform::RewriteRule`: returns the replacement for a value, if any.
pub trait RewriteRule {
    fn apply(&mut self, m: &mut Module, v: ValueId) -> Option<ValueId>;
}

impl<F: FnMut(&mut Module, ValueId) -> Option<ValueId>> RewriteRule for F {
    fn apply(&mut self, m: &mut Module, v: ValueId) -> Option<ValueId> {
        self(m, v)
    }
}

/// Port of `transform::Rewriter`.
#[derive(Default)]
pub struct Rewriter {
    // PORT-NOTE: C++ uses an `unordered_map` (unspecified order); insertion order is
    // used here for determinism.
    rules: IndexMap<String, Box<dyn RewriteRule>>,
    num_replacements: i32,
}

impl Rewriter {
    pub fn new() -> Self {
        Self::default()
    }
    /// `registerRule(key, rule)` (keeps the first rule registered under a key).
    pub fn register_rule(&mut self, key: impl Into<String>, rule: Box<dyn RewriteRule>) {
        self.rules.entry(key.into()).or_insert(rule);
    }
    /// `rewrite(v)`: applies every rule in turn; replaces `v` with the final result.
    pub fn rewrite(&mut self, m: &mut Module, v: ValueId) {
        let mut result = v;
        for rule in self.rules.values_mut() {
            if let Some(rep) = rule.apply(m, result) {
                self.num_replacements += 1;
                result = rep;
            }
        }
        if v != result {
            m.replace_all_value(v, result);
        }
    }
    pub fn num_replacements(&self) -> i32 {
        self.num_replacements
    }
    pub fn reset(&mut self) {
        self.num_replacements = 0;
    }
}
