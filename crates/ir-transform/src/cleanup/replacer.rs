//! Port of codon/cir/transform/cleanup/replacer.{h,cpp}

use indexmap::IndexSet;
use typhon_ir::pass::{Pass, PassContext};
use typhon_ir::{Module, NodeRef, TypeId, ValueId, VarId};

/// Cleanup pass that physically replaces nodes.
#[derive(Default)]
pub struct ReplaceCleanupPass;

impl ReplaceCleanupPass {
    pub const KEY: &'static str = "core-cleanup-physical-replace";

    pub fn new() -> Self {
        Self
    }
}

#[derive(Default)]
struct ToDelete {
    values: IndexSet<ValueId>,
    types: IndexSet<TypeId>,
    vars: IndexSet<VarId>,
}

/// Rewrites every used value/type/var of `n` that has a replacement to its actual node.
fn fix(m: &mut Module, n: NodeRef, del: &mut ToDelete) {
    for c in m.used_values(n) {
        if m.has_replacement(c.into()) {
            let actual = m.actual_value(c);
            m.replace_used_value(n, c, actual);
            del.values.insert(c);
        }
    }
    for t in m.used_types(n) {
        if m.has_replacement(t.into()) {
            let actual = m.actual_type(t);
            m.replace_used_type(n, t, actual);
            del.types.insert(t);
        }
    }
    for v in m.used_variables(n) {
        if m.has_replacement(v.into()) {
            let actual = m.actual_var(v);
            m.replace_used_variable(n, v, actual);
            del.vars.insert(v);
        }
    }
}

impl Pass for ReplaceCleanupPass {
    fn key(&self) -> String {
        Self::KEY.to_string()
    }
    fn run(&mut self, m: &mut Module, _ctx: &dyn PassContext) {
        // PORT-NOTE: accessors resolve forwarding, so `used_*` already return
        // resolved ids for most nodes; the pass still rewrites stored ids where the
        // IR exposes the raw slot and removes nodes that carry a replacement.
        let mut del = ToDelete::default();
        let main = m.main_func();
        fix(m, main.into(), &mut del);
        for v in [m.argv_var(), m.argc_var()] {
            fix(m, v.into(), &mut del);
        }
        for v in m.values() {
            fix(m, v.into(), &mut del);
        }
        for v in m.vars() {
            fix(m, v.into(), &mut del);
        }
        for t in m.types() {
            fix(m, t.into(), &mut del);
        }
        for v in del.values {
            m.remove_value(v);
        }
        for v in del.vars {
            m.remove_var(v);
        }
        for t in del.types {
            m.remove_type(t);
        }
    }
}
