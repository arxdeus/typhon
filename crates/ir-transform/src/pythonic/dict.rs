
use typhon_ir::util::cloning::CloneVisitor;
use typhon_ir::util::irtools;
use typhon_ir::util::matching::match_nodes;
use typhon_ir::util::operator::{Operator, OperatorState};
use typhon_ir::{FuncId, Module, ValueId, ValueKind};

use crate::lowering::{operator_pass, operator_state};

/// get or __getitem__ call metadata
struct GetCall {
    #[allow(dead_code)]
    func: FuncId,
    dict: ValueId,
    key: ValueId,
    dflt: Option<ValueId>,
}

fn is_call(m: &Module, v: ValueId) -> bool {
    m.as_call_instr(v).is_some()
}

/// Identify the call and return its metadata.
fn analyze_get(m: &Module, call: ValueId) -> Option<GetCall> {
    let c = m.as_call_instr(call)?;
    let func = irtools::get_func(m, c.callee)?;
    let unmangled = &m.func(func)?.unmangled_name;
    if c.args.len() < 2 {
        return None;
    }
    let (dict, key) = (c.args[0], c.args[1]);
    if is_call(m, dict) || is_call(m, key) {
        return None;
    }
    if unmangled == "get" && c.args.len() == 3 {
        // Note: the reference implementation `cast<Const>` may yield null (non-const default), which then
        // behaves like the no-default case; preserved.
        let d = c.args[2];
        let dflt = if m.kind(d).is_const() { Some(d) } else { None };
        Some(GetCall { func, dict, key, dflt })
    } else if unmangled == "__getitem__" && c.args.len() == 2 {
        Some(GetCall { func, dict, key, dflt: None })
    } else {
        None
    }
}

/// Pass to optimize calls of form d[x] = func(d[x], any) or
/// d[x] = func(d.get(x, k), any). This will work on any dictionary-like
/// object that implements _do_op and _do_op_throws as well as getters.
#[derive(Default)]
pub struct DictArithmeticOptimization {
    state: OperatorState,
}

impl DictArithmeticOptimization {
    pub const KEY: &'static str = "core-pythonic-dict-arithmetic-opt";
    pub fn new() -> Self {
        Self::default()
    }
}

impl Operator for DictArithmeticOptimization {
    operator_state!();
    fn handle_call_instr(&mut self, m: &mut Module, v: ValueId) {
        let c = m.as_call_instr(v).unwrap().clone();
        let Some(set_func) = irtools::get_func(m, c.callee) else { return };
        if m.func(set_func).unwrap().unmangled_name != "__setitem__" || c.args.len() != 3 {
            return;
        }
        let (dict_value, key_value) = (c.args[0], c.args[1]);
        if is_call(m, dict_value) || is_call(m, key_value) {
            return;
        }
        let Some(op_call) = m.as_call_instr(c.args[2]).cloned() else { return };
        if op_call.args.len() != 2 {
            return;
        }
        let Some(op_func) = irtools::get_func(m, op_call.callee) else { return };
        if !matches!(m.kind(op_call.args[0]), ValueKind::CallInstr(_)) {
            return;
        }
        let get_call = op_call.args[0];
        let int_t = m.get_int_type();
        let float_t = m.get_float_type();
        let Some(parent_type) = m.func(op_func).unwrap().parent_type else { return };
        if !(m.type_is(parent_type, int_t) || m.type_is(parent_type, float_t)) {
            return;
        }
        let Some(ga) = analyze_get(m, get_call) else { return };
        let second = op_call.args[1];
        if !(match_nodes(m, Some(dict_value.into()), Some(ga.dict.into()), false, true)
            && match_nodes(m, Some(key_value.into()), Some(ga.key.into()), false, true))
        {
            return;
        }
        let mut cv = CloneVisitor::new(false);
        let dt = m.value_type_mut(dict_value);
        let kt = m.value_type_mut(key_value);
        let st = m.value_type_mut(second);
        let ft = m.var_type(op_func).expect("function type");
        let replacement = if let Some(d) = ga.dflt {
            let dft = m.value_type_mut(d);
            m.get_or_realize_method(dt, "__dict_do_op__", &[dt, kt, st, dft, ft], &[])
        } else {
            m.get_or_realize_method(dt, "__dict_do_op_throws__", &[dt, kt, st, ft], &[])
        };
        let Some(rf) = replacement else { return };
        let mut args = vec![cv.clone_value(m, dict_value), cv.clone_value(m, key_value), cv.clone_value(m, second)];
        if let Some(d) = ga.dflt {
            args.push(cv.clone_value(m, d));
        }
        // sanity check to make sure function is inlined
        if args.len() != m.func_args(rf).len() {
            let src = m.src_info(v.into());
            let fv = m.nr_var_value(op_func);
            m.set_src_info(fv.into(), src);
            args.push(fv);
        }
        let r = irtools::call(m, rf, args);
        m.replace_all_value(v, r);
    }
}

operator_pass!(DictArithmeticOptimization);
