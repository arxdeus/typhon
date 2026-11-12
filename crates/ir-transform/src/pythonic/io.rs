
use typhon_ir::util::cloning::CloneVisitor;
use typhon_ir::util::irtools;
use typhon_ir::util::operator::{Operator, OperatorState};
use typhon_ir::{Module, TypeKind, ValueId};

use crate::lowering::{operator_pass, operator_state};

fn unmangled(m: &Module, v: ValueId) -> Option<(typhon_ir::FuncId, String)> {
    let f = irtools::get_func(m, v)?;
    Some((f, m.func(f)?.unmangled_name.clone()))
}

fn optimize_print(m: &mut Module, v: ValueId) {
    let c = m.as_call_instr(v).unwrap().clone();
    let Some(&first) = c.args.first() else { return };
    let Some(inner) = m.as_call_instr(first).cloned() else { return };
    match unmangled(m, inner.callee) {
        Some((_, n)) if n == "__new__" && inner.args.len() == 1 => {}
        _ => return,
    }
    let Some(cat) = m.as_call_instr(inner.args[0]).cloned() else { return };
    let Some((cat_func, n)) = unmangled(m, cat.callee) else { return };
    if n != "cat" || cat.args.is_empty() || c.args.len() < 2 {
        return;
    }
    let s = m.get_string_type();
    let pt = m.value_type_mut(cat.args[0]);
    // Note: the reference dereferences a possibly-null `realCat`; we bail out instead.
    let Some(real_cat) = m.get_or_realize_method(s, "cat", &[pt], &[]) else { return };
    if m.var_uid(real_cat) != m.var_uid(cat_func) {
        return;
    }
    let mut cv = CloneVisitor::new(false);
    let mut args = Vec::new();
    let mut types = Vec::new();
    for a in &c.args {
        args.push(cv.clone_value(m, *a));
        types.push(m.value_type_mut(*a));
    }
    args[0] = cv.clone_value(m, cat.args[0]);
    types[0] = m.value_type_mut(args[0]);
    args[1] = m.get_string("");
    let Some(rep) = m.get_or_realize_func("print", &types, &[], "std.internal.builtin") else { return };
    let r = irtools::call(m, rep, args);
    m.replace_all_value(v, r);
}

fn optimize_write(m: &mut Module, v: ValueId) {
    let c = m.as_call_instr(v).unwrap().clone();
    if c.args.len() != 2 {
        return;
    }
    let file = c.args[0];
    let Some(write_func) = irtools::get_func(m, c.callee) else { return };
    let ft = m.value_type_mut(file);
    match m.func(write_func).unwrap().parent_type {
        Some(p) if m.type_is(p, ft) => {}
        _ => return,
    }
    let Some(cat) = m.as_call_instr(c.args[1]).cloned() else { return };
    if cat.args.len() != 1 {
        return;
    }
    let s = m.get_string_type();
    let Some(cat_func) = irtools::get_func(m, cat.callee) else { return };
    let cfd = m.func(cat_func).unwrap();
    if cfd.unmangled_name != "cat" || !cfd.parent_type.is_some_and(|p| m.type_is(p, s)) {
        return;
    }
    let pieces_type = m.value_type_mut(cat.args[0]);
    let TypeKind::Record { fields } = m.type_kind(pieces_type).clone() else { return };
    let mut piece_types = Vec::new();
    for f in &fields {
        if !m.type_is(f.ty, s) {
            return;
        }
        piece_types.push(f.ty);
    }
    let tt = m.get_tuple_type(piece_types);
    if !m.type_is(pieces_type, tt) {
        return;
    }
    match m.get_or_realize_method(s, "cat", &[pieces_type], &[]) {
        Some(rc) if m.var_uid(rc) == m.var_uid(cat_func) => {}
        _ => return,
    }
    let ct = m.value_type_mut(c.args[1]);
    match m.get_or_realize_method(ft, "write", &[ft, ct], &[]) {
        Some(rw) if m.var_uid(rw) == m.var_uid(write_func) => {}
        _ => return,
    }
    let mut cv = CloneVisitor::new(false);
    let cl = cv.clone_value(m, cat.args[0]);
    let Some(iter) = m.iter(cl) else { return };
    let args = vec![cv.clone_value(m, file), iter];
    let a0 = m.value_type_mut(args[0]);
    let a1 = m.value_type_mut(args[1]);
    let Some(rep) = m.get_or_realize_method(ft, "__file_write_gen__", &[a0, a1], &[]) else { return };
    let vt = m.value_type_mut(v);
    if !irtools::return_type(m, rep).is_some_and(|r| m.type_is(r, vt)) {
        return;
    }
    let r = irtools::call(m, rep, args);
    m.replace_all_value(v, r);
}

/// Pass to optimize print str.cat(...) or file.write(str.cat(...)).
#[derive(Default)]
pub struct IOCatOptimization {
    state: OperatorState,
}

impl IOCatOptimization {
    pub const KEY: &'static str = "core-pythonic-io-cat-opt";
    pub fn new() -> Self {
        Self::default()
    }
}

impl Operator for IOCatOptimization {
    operator_state!();
    fn handle_call_instr(&mut self, m: &mut Module, v: ValueId) {
        let callee = m.as_call_instr(v).unwrap().callee;
        if irtools::get_stdlib_func(m, callee, "print", "").is_some() {
            optimize_print(m, v);
        } else if let Some((_, n)) = unmangled(m, callee) {
            if n == "write" {
                optimize_write(m, v);
            }
        }
    }
}

operator_pass!(IOCatOptimization);
