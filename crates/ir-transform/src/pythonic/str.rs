//! Port of codon/cir/transform/pythonic/str.{h,cpp}

use typhon_ir::util::cloning::CloneVisitor;
use typhon_ir::util::irtools;
use typhon_ir::util::operator::{Operator, OperatorState};
use typhon_ir::{Module, ValueId, magic};

use crate::lowering::{operator_pass, operator_state};

struct InspectionResult {
    valid: bool,
    args: Vec<ValueId>,
}

fn is_string(m: &mut Module, v: ValueId) -> bool {
    let t = m.value_type_mut(v);
    let s = m.get_string_type();
    m.type_is(t, s)
}

fn inspect(m: &mut Module, v: ValueId, r: &mut InspectionResult) {
    if is_string(m, v) {
        if let Some(c) = m.as_call_instr(v).cloned() {
            let is_add = irtools::get_func(m, c.callee).is_some_and(|f| m.func(f).unwrap().unmangled_name == magic::ADD);
            if is_add && c.args.len() == 2 && is_string(m, c.args[0]) && is_string(m, c.args[1]) {
                inspect(m, c.args[0], r);
                inspect(m, c.args[1], r);
                return;
            }
        }
        r.args.push(v);
    } else {
        r.valid = false;
    }
}

/// Pass to optimize str1 + str2 + ...
#[derive(Default)]
pub struct StrAdditionOptimization {
    state: OperatorState,
}

impl StrAdditionOptimization {
    pub const KEY: &'static str = "core-pythonic-str-addition-opt";
    pub fn new() -> Self {
        Self::default()
    }
}

impl Operator for StrAdditionOptimization {
    operator_state!();
    fn handle_call_instr(&mut self, m: &mut Module, v: ValueId) {
        let callee = m.as_call_instr(v).unwrap().callee;
        let Some(f) = irtools::get_func(m, callee) else { return };
        if m.func(f).unwrap().unmangled_name != magic::ADD {
            return;
        }
        let mut r = InspectionResult { valid: true, args: vec![] };
        inspect(m, v, &mut r);
        if r.valid && r.args.len() > 2 {
            let mut cv = CloneVisitor::new(false);
            let args: Vec<ValueId> = r.args.iter().map(|a| cv.clone_value(m, *a)).collect();
            let Some(arg) = irtools::make_tuple(m, &args) else { return };
            let at = m.value_type_mut(arg);
            let s = m.get_string_type();
            let rf = m
                .get_or_realize_method(s, "cat", &[at], &[])
                .unwrap_or_else(|| panic!("could not find cat function [{}]", m.src_info(v.into())));
            let c = irtools::call(m, rf, vec![arg]);
            m.replace_all_value(v, c);
        }
    }
}

operator_pass!(StrAdditionOptimization);
