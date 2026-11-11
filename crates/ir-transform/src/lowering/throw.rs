//! Port of codon/cir/transform/lowering/throw.{h,cpp}

use typhon_ir::util::irtools;
use typhon_ir::util::operator::{Operator, OperatorState};
use typhon_ir::{Module, ValueId};

use super::{operator_pass, operator_state};

/// Gives an empty `raise` (no active value) a fallback that raises
/// "no active exception".
#[derive(Default)]
pub struct ThrowLowering {
    state: OperatorState,
}

impl ThrowLowering {
    pub const KEY: &'static str = "core-throw-lowering";
    pub fn new() -> Self {
        Self::default()
    }
}

impl Operator for ThrowLowering {
    operator_state!();
    fn handle_throw_instr(&mut self, m: &mut Module, v: ValueId) {
        let t = m.as_throw_instr(v).expect("throw");
        if t.value.is_some() || t.fallback.is_some() {
            return;
        }
        let fallback = m
            .get_or_realize_func("_raise_no_active_exception", &[], &[], "std.internal.types.error")
            .expect("empty-raise fallback not found");
        let c = irtools::call(m, fallback, vec![]);
        m.set_throw_fallback(v, Some(c));
    }
}

operator_pass!(ThrowLowering);
