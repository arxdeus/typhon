//!
//! `util::Operator` is a pre/post-order walker over the IR with per-kind `handle`
//! callbacks, a parent stack and series-flow insertion support. In Rust the walker
//! state lives in [`OperatorState`] (embedded in the implementor and exposed through
//! [`Operator::state`]/[`Operator::state_mut`]); the traversal is provided by default
//! methods. Per-kind handlers are `handle_<kind>`; overriding [`Operator::handle`] instead
//! receives every value.

use rustc_hash::FxHashSet;

use crate::ids::{FuncId, Id, NodeRef, ValueId};
use crate::module::Module;
use crate::util::visitor::{ValueVisitKind, value_visit_kind};

/// Walker state of an [`Operator`].
#[derive(Clone, Debug, Default)]
pub struct OperatorState {
    seen: FxHashSet<Id>,
    node_stack: Vec<NodeRef>,
    /// Stack of indices into the series flows currently being iterated.
    it_stack: Vec<usize>,
    children_first: bool,
}

impl OperatorState {
    /// `Operator(childrenFirst)`.
    pub fn new(children_first: bool) -> Self {
        Self { children_first, ..Default::default() }
    }
    pub fn children_first(&self) -> bool {
        self.children_first
    }
}

macro_rules! handlers {
    ($($name:ident),*) => {
        $(
            fn $name(&mut self, _m: &mut Module, _v: ValueId) {}
        )*
    };
}

pub trait Operator {
    fn state(&self) -> &OperatorState;
    fn state_mut(&mut self) -> &mut OperatorState;

    /// `preHook(node)`: called before `handle`.
    fn pre_hook(&mut self, _m: &mut Module, _n: NodeRef) {}
    /// `postHook(node)`: called after `handle`.
    fn post_hook(&mut self, _m: &mut Module, _n: NodeRef) {}

    handlers!(
        handle_var_value,
        handle_pointer_value,
        handle_series_flow,
        handle_if_flow,
        handle_while_flow,
        handle_for_flow,
        handle_imperative_for_flow,
        handle_try_catch_flow,
        handle_pipeline_flow,
        handle_int_const,
        handle_float_const,
        handle_bool_const,
        handle_string_const,
        handle_bytes_const,
        handle_assign_instr,
        handle_extract_instr,
        handle_insert_instr,
        handle_call_instr,
        handle_stack_alloc_instr,
        handle_yield_in_instr,
        handle_ternary_instr,
        handle_break_instr,
        handle_continue_instr,
        handle_return_instr,
        handle_yield_instr,
        handle_await_instr,
        handle_throw_instr,
        handle_flow_instr,
        handle_any_value
    );

    /// Dispatches to the `handle_<kind>` method of the (resolved) value.
    fn handle(&mut self, m: &mut Module, v: ValueId) {
        use ValueVisitKind as K;
        match value_visit_kind(m.kind(v)) {
            K::VarValue => self.handle_var_value(m, v),
            K::PointerValue => self.handle_pointer_value(m, v),
            K::SeriesFlow => self.handle_series_flow(m, v),
            K::IfFlow => self.handle_if_flow(m, v),
            K::WhileFlow => self.handle_while_flow(m, v),
            K::ForFlow => self.handle_for_flow(m, v),
            K::ImperativeForFlow => self.handle_imperative_for_flow(m, v),
            K::TryCatchFlow => self.handle_try_catch_flow(m, v),
            K::PipelineFlow => self.handle_pipeline_flow(m, v),
            K::IntConst => self.handle_int_const(m, v),
            K::FloatConst => self.handle_float_const(m, v),
            K::BoolConst => self.handle_bool_const(m, v),
            K::StringConst => self.handle_string_const(m, v),
            K::BytesConst => self.handle_bytes_const(m, v),
            K::AssignInstr => self.handle_assign_instr(m, v),
            K::ExtractInstr => self.handle_extract_instr(m, v),
            K::InsertInstr => self.handle_insert_instr(m, v),
            K::CallInstr => self.handle_call_instr(m, v),
            K::StackAllocInstr => self.handle_stack_alloc_instr(m, v),
            K::YieldInInstr => self.handle_yield_in_instr(m, v),
            K::TernaryInstr => self.handle_ternary_instr(m, v),
            K::BreakInstr => self.handle_break_instr(m, v),
            K::ContinueInstr => self.handle_continue_instr(m, v),
            K::ReturnInstr => self.handle_return_instr(m, v),
            K::YieldInstr => self.handle_yield_instr(m, v),
            K::AwaitInstr => self.handle_await_instr(m, v),
            K::ThrowInstr => self.handle_throw_instr(m, v),
            K::FlowInstr => self.handle_flow_instr(m, v),
            K::Any => self.handle_any_value(m, v),
        }
    }

    /// `visit(Module*)`: main function first, then every module variable.
    fn visit_module_op(&mut self, m: &mut Module) {
        self.state_mut().node_stack.push(NodeRef::Module);
        let main = m.main_func();
        self.state_mut().node_stack.push(main.into());
        self.process(m, main.into());
        self.state_mut().node_stack.pop();
        for s in m.vars() {
            if !m.is_var_alive(s) {
                continue;
            }
            self.state_mut().node_stack.push(s.into());
            self.process(m, s.into());
            self.state_mut().node_stack.pop();
        }
        self.state_mut().node_stack.pop();
    }

    /// `visit(BodiedFunc*)`: processes the body.
    fn visit_bodied_func_op(&mut self, m: &mut Module, f: FuncId) {
        if let Some(body) = m.func_body(f) {
            let uid = m.value_uid(body);
            self.state_mut().seen.insert(uid);
            self.process(m, body.into());
        }
    }

    /// `visit(Value*)` for all value kinds (the `LAMBDA_VISIT` expansion).
    fn visit_value_op(&mut self, m: &mut Module, v: ValueId) {
        let v = m.actual_value(v);
        let is_series = m.as_series_flow(v).is_some();
        let cf = self.state().children_first;
        if cf {
            if is_series {
                self.process_series_flow_children(m, v);
            } else {
                self.process_children(m, v);
            }
        }
        self.pre_hook(m, v.into());
        self.handle(m, v);
        self.post_hook(m, v.into());
        if !cf {
            if is_series {
                self.process_series_flow_children(m, v);
            } else {
                self.process_children(m, v);
            }
        }
    }

    /// `process(node)` (`node->accept(*this)`).
    fn process(&mut self, m: &mut Module, n: NodeRef) {
        match m.actual(n) {
            NodeRef::Module => self.visit_module_op(m),
            NodeRef::Value(v) => self.visit_value_op(m, v),
            NodeRef::Var(v) => {
                if m.func(v).map(|f| f.is_bodied()).unwrap_or(false) {
                    self.visit_bodied_func_op(m, v);
                }
            }
            NodeRef::Type(_) => {}
        }
    }
    fn process_value(&mut self, m: &mut Module, v: ValueId) {
        self.process(m, v.into());
    }

    #[doc(hidden)]
    fn process_children(&mut self, m: &mut Module, v: ValueId) {
        self.state_mut().node_stack.push(v.into());
        // CallInstr: snapshot of args followed by callee (matches the reference implementation explicit branch).
        let children: Vec<ValueId> = match m.as_call_instr(v) {
            Some(call) => {
                let mut c = call.args.clone();
                c.push(call.callee);
                c
            }
            None => m.used_values(v.into()),
        };
        for child in children {
            let uid = m.value_uid(child);
            if !self.state_mut().seen.insert(uid) {
                continue;
            }
            self.process(m, child.into());
        }
        self.state_mut().node_stack.pop();
    }

    #[doc(hidden)]
    fn process_series_flow_children(&mut self, m: &mut Module, v: ValueId) {
        self.state_mut().node_stack.push(v.into());
        let mut i = 0usize;
        loop {
            let Some(child) = m.as_series_flow(v).and_then(|s| s.series.get(i).copied()) else { break };
            self.state_mut().it_stack.push(i);
            self.process(m, child.into());
            i = self.state_mut().it_stack.pop().expect("iterator stack underflow") + 1;
        }
        self.state_mut().node_stack.pop();
    }

    /// `getParent<Node>(level)`.
    fn parent(&self, level: usize) -> Option<NodeRef> {
        let st = &self.state().node_stack;
        st.len().checked_sub(level + 1).map(|i| st[i])
    }
    /// `depth()`.
    fn depth(&self) -> usize {
        self.state().node_stack.len()
    }
    /// `findLast<T>()` with an arbitrary predicate.
    fn find_last(&self, pred: &dyn Fn(NodeRef) -> bool) -> Option<NodeRef> {
        self.state().node_stack.iter().rev().copied().find(|n| pred(*n))
    }
    /// `getParentFunc()`.
    fn parent_func(&self, m: &Module) -> Option<FuncId> {
        self.find_last(&|n| matches!(n, NodeRef::Var(v) if m.is_func(v))).and_then(|n| n.as_var())
    }
    /// Innermost enclosing `SeriesFlow` (`findLast<SeriesFlow>()`).
    fn parent_series(&self, m: &Module) -> Option<ValueId> {
        self.find_last(&|n| matches!(n, NodeRef::Value(v) if m.as_series_flow(v).is_some())).and_then(|n| n.as_value())
    }
    /// `parent_begin()/parent_end()`.
    fn parents(&self) -> &[NodeRef] {
        &self.state().node_stack
    }
    /// `saw(v)`.
    fn saw(&self, m: &Module, v: ValueId) -> bool {
        self.state().seen.contains(&m.value_uid(v))
    }
    /// `see(v)`.
    fn see(&mut self, m: &Module, v: ValueId) {
        let uid = m.value_uid(v);
        self.state_mut().seen.insert(uid);
    }
    /// `insertBefore(v)`: inserts before the series element currently being processed.
    fn insert_before(&mut self, m: &mut Module, v: ValueId) {
        let series = self.parent_series(m).expect("no enclosing series flow");
        let idx = *self.state().it_stack.last().expect("not inside a series flow");
        m.series_insert(series, idx, v);
        *self.state_mut().it_stack.last_mut().unwrap() += 1;
    }
    /// `insertAfter(v)`: inserts after the current series element and marks it seen.
    fn insert_after(&mut self, m: &mut Module, v: ValueId) {
        let series = self.parent_series(m).expect("no enclosing series flow");
        let idx = *self.state().it_stack.last().expect("not inside a series flow");
        self.see(m, v);
        m.series_insert(series, idx + 1, v);
    }
    /// `reset()`.
    fn reset(&mut self) {
        let st = self.state_mut();
        st.seen.clear();
        st.node_stack.clear();
        st.it_stack.clear();
    }
}

/// `OperatorPass::run(module)`: `reset(); process(module);`.
pub fn run_operator<O: Operator + ?Sized>(op: &mut O, m: &mut Module) {
    op.reset();
    op.process(m, NodeRef::Module);
}
