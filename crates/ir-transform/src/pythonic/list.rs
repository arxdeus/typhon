
use typhon_common::mangle::stdlib;
use typhon_ir::util::cloning::CloneVisitor;
use typhon_ir::util::irtools;
use typhon_ir::util::operator::{Operator, OperatorState};
use typhon_ir::{Attribute, FuncId, Module, TypeId, ValueId, VarId, attr_id, magic};

use crate::lowering::{operator_pass, operator_state};

fn is_list(m: &mut Module, v: ValueId) -> bool {
    let t = m.value_type_mut(v);
    m.type_name(t).starts_with(&format!("{}[", stdlib().list))
}

fn is_slice(m: &mut Module, v: ValueId) -> bool {
    let t = m.value_type_mut(v);
    m.type_name(t) == format!("{}[Int[64],Int[64],Int[64]]", stdlib().slice)
}

fn is_func_named(m: &Module, callee: ValueId, name: &str) -> bool {
    irtools::get_func(m, callee).is_some_and(|f| m.func(f).unwrap().unmangled_name == name)
}

// The handlers account for the possible sub-expressions seen when optimizing
// list1 + list2 + ... listN: slices x[a:b:c], literals [a, b, c], and the default
// (append by iterating). Stages: setup (assign to vars in order), count (total
// length), create (new list with capacity, append everything).
enum HandlerKind {
    Default { element: ValueId },
    Slice { element: ValueId, slice: ValueId },
    Literal { elements: Vec<ValueId> },
}

struct ElementHandler {
    vars: Vec<VarId>,
    kind: HandlerKind,
}

impl ElementHandler {
    fn get(m: &mut Module, v: ValueId, ty: TypeId) -> Option<Self> {
        let vt = m.value_type_mut(v);
        if !m.type_is(vt, ty) {
            return None;
        }
        // SliceHandler
        if let Some(c) = m.as_call_instr(v).cloned() {
            if is_func_named(m, c.callee, magic::GETITEM)
                && c.args.len() == 2
                && is_list(m, c.args[0])
                && is_slice(m, c.args[1])
            {
                return Some(Self { vars: vec![], kind: HandlerKind::Slice { element: c.args[0], slice: c.args[1] } });
            }
        }
        // LiteralHandler
        if let Some(Attribute::ListLiteral(elems)) = m.get_attribute(v.into(), attr_id::LIST_LITERAL) {
            if elems.iter().any(|e| e.star) {
                return None;
            }
            let elements = elems.iter().map(|e| e.value).collect();
            return Some(Self { vars: vec![], kind: HandlerKind::Literal { elements } });
        }
        Some(Self { vars: vec![], kind: HandlerKind::Default { element: v } })
    }

    fn setup(&mut self, m: &mut Module, block: ValueId, parent: Option<FuncId>) {
        let values = match &self.kind {
            HandlerKind::Default { element } => vec![*element],
            HandlerKind::Slice { element, slice } => vec![*element, *slice],
            HandlerKind::Literal { elements } => elements.clone(),
        };
        for v in values {
            self.vars.push(irtools::make_var(m, v, block, parent, false));
        }
    }

    fn length(&self, m: &mut Module) -> ValueId {
        match &self.kind {
            HandlerKind::Default { element } => {
                let e = m.nr_var_value(self.vars[0]);
                let ty = m.value_type_mut(*element);
                let f = m
                    .get_or_realize_method(ty, "_list_add_opt_default_len", &[ty], &[])
                    .expect("could not find default list length helper");
                irtools::call(m, f, vec![e])
            }
            HandlerKind::Slice { element, slice } => {
                let e = m.nr_var_value(self.vars[0]);
                let s = m.nr_var_value(self.vars[1]);
                let ty = m.value_type_mut(*element);
                let st = m.value_type_mut(*slice);
                let f = m
                    .get_or_realize_method(ty, "_list_add_opt_slice_len", &[ty, st], &[])
                    .expect("could not find slice list length helper");
                irtools::call(m, f, vec![e, s])
            }
            HandlerKind::Literal { elements } => m.get_int(elements.len() as i64),
        }
    }

    fn append(&self, m: &mut Module, result: ValueId) -> ValueId {
        let ty = m.value_type_mut(result);
        match &self.kind {
            HandlerKind::Default { .. } => {
                let e = m.nr_var_value(self.vars[0]);
                let f = m
                    .get_or_realize_method(ty, "_list_add_opt_default_append", &[ty, ty], &[])
                    .expect("could not find default list append helper");
                irtools::call(m, f, vec![result, e])
            }
            HandlerKind::Slice { slice, .. } => {
                let e = m.nr_var_value(self.vars[0]);
                let s = m.nr_var_value(self.vars[1]);
                let st = m.value_type_mut(*slice);
                let f = m
                    .get_or_realize_method(ty, "_list_add_opt_slice_append", &[ty, ty, st], &[])
                    .expect("could not find slice list append helper");
                irtools::call(m, f, vec![result, e, s])
            }
            HandlerKind::Literal { elements } => {
                let block = m.nr_series_flow();
                if self.vars.is_empty() {
                    return block;
                }
                let et = m.value_type_mut(elements[0]);
                let f = m
                    .get_or_realize_method(ty, "_list_add_opt_literal_append", &[ty, et], &[])
                    .expect("could not find literal list append helper");
                let mut first = true;
                for var in &self.vars {
                    // Note: the reference reuses the same `result` node for every call; we
                    // create a fresh VarValue per call after the first (same semantics).
                    let r = if first {
                        first = false;
                        result
                    } else {
                        let rv = irtools::get_var(m, result).expect("result var");
                        m.nr_var_value(rv)
                    };
                    let vv = m.nr_var_value(*var);
                    let c = irtools::call(m, f, vec![r, vv]);
                    m.series_push(block, c);
                }
                block
            }
        }
    }
}

struct InspectionResult {
    valid: bool,
    args: Vec<ValueId>,
}

fn inspect(m: &mut Module, v: ValueId, r: &mut InspectionResult) {
    if is_list(m, v) {
        if let Some(c) = m.as_call_instr(v).cloned() {
            if is_func_named(m, c.callee, magic::ADD)
                && c.args.len() == 2
                && is_list(m, c.args[0])
                && is_list(m, c.args[1])
            {
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

fn optimize(m: &mut Module, parent: Option<FuncId>, r: &InspectionResult) -> Option<ValueId> {
    if !r.valid || r.args.len() <= 1 {
        return None;
    }
    let ty = m.value_type_mut(r.args[0]);
    let mut cv = CloneVisitor::new(false);
    let mut handlers = Vec::new();
    for v in &r.args {
        let c = cv.clone_value(m, *v);
        // Note: the reference stores a null handler on type mismatch and would crash
        // later; we bail out instead.
        handlers.push(ElementHandler::get(m, c, ty)?);
    }
    let opt = m.nr_series_flow();
    let zero = m.get_int(0);
    let len = irtools::make_var(m, zero, opt, parent, false);
    for h in handlers.iter_mut() {
        h.setup(m, opt, parent);
    }
    for h in &handlers {
        let lv = m.nr_var_value(len);
        let hl = h.length(m);
        let sum = m.binary_op(lv, magic::ADD, hl).expect("int __add__ not found");
        let a = m.nr_assign(len, sum);
        m.series_push(opt, a);
    }
    let it = m.get_int_type();
    let f = m.get_or_realize_method(ty, "_list_add_opt_opt_new", &[it], &[]).expect("could not find list new helper");
    let lv = m.nr_var_value(len);
    let c = irtools::call(m, f, vec![lv]);
    let result = irtools::make_var(m, c, opt, parent, false);
    for h in &handlers {
        let rv = m.nr_var_value(result);
        let a = h.append(m, rv);
        m.series_push(opt, a);
    }
    let rv = m.nr_var_value(result);
    Some(m.nr_flow_instr(opt, rv))
}

/// Pass to optimize list1 + list2 + ...
#[derive(Default)]
pub struct ListAdditionOptimization {
    state: OperatorState,
}

impl ListAdditionOptimization {
    pub const KEY: &'static str = "core-pythonic-list-addition-opt";
    pub fn new() -> Self {
        Self::default()
    }
}

impl Operator for ListAdditionOptimization {
    operator_state!();
    fn handle_call_instr(&mut self, m: &mut Module, v: ValueId) {
        let callee = m.as_call_instr(v).unwrap().callee;
        if !is_func_named(m, callee, magic::ADD) {
            return;
        }
        let mut r = InspectionResult { valid: true, args: vec![] };
        inspect(m, v, &mut r);
        let parent = self.parent_func(m);
        if let Some(opt) = optimize(m, parent, &r) {
            m.replace_all_value(v, opt);
        }
    }
}

operator_pass!(ListAdditionOptimization);
