
use indexmap::IndexMap;

use crate::ids::ValueId;
use crate::module::Module;

/// returns the replacement for a value, if any.
pub trait RewriteRule {
    fn apply(&mut self, m: &mut Module, v: ValueId) -> Option<ValueId>;
    /// Unmangled callee name this rule requires (`apply` returns `None` for any other
    /// value); lets the rewriter skip it without calling `apply`.
    fn magic(&self) -> Option<&str> {
        None
    }
}

impl<F: FnMut(&mut Module, ValueId) -> Option<ValueId>> RewriteRule for F {
    fn apply(&mut self, m: &mut Module, v: ValueId) -> Option<ValueId> {
        self(m, v)
    }
}

#[derive(Default)]
pub struct Rewriter {
    // Note: the reference uses an `unordered_map` (unspecified order); insertion order is
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
        let mut name = callee_name(m, result);
        for rule in self.rules.values_mut() {
            if let Some(want) = rule.magic()
                && name.as_deref() != Some(want)
            {
                continue;
            }
            if let Some(rep) = rule.apply(m, result) {
                self.num_replacements += 1;
                result = rep;
                name = callee_name(m, result);
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

/// Unmangled name of the function called by `v` (`None` if `v` is not a direct call).
fn callee_name(m: &Module, v: ValueId) -> Option<String> {
    let call = m.as_call_instr(v)?;
    let f = crate::util::irtools::get_func(m, call.callee)?;
    Some(m.func(f)?.unmangled_name.clone())
}
