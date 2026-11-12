//!
//! the reference implementation double dispatch (`node->accept(visitor)`) becomes [`accept`] (and the typed helpers
//! [`accept_value`], [`accept_var`], [`accept_type`]) which resolve replacements and call
//! the `visit_*` method matching the node kind. Each method receives the module and the
//! (already resolved) node id. Default implementations call `default_visit`, which is a
//! no-op. Note: the reference implementation `Visitor::defaultVisit` throws `"cannot visit node"`; every
//! in-tree visitor overrides it or the relevant methods, so a no-op default is used.

use crate::ids::{FuncId, NodeRef, TypeId, ValueId, VarId};
use crate::module::Module;
use crate::types::TypeKind;
use crate::value::ValueKind;
use crate::var::FuncKind;

macro_rules! visitor_trait {
    ($tr:ident, $m:ty) => {
        pub trait $tr {
            fn default_visit(&mut self, _m: $m, _n: NodeRef) {}

            fn visit_module(&mut self, m: $m) {
                self.default_visit(m, NodeRef::Module)
            }
            fn visit_var(&mut self, m: $m, v: VarId) {
                self.default_visit(m, v.into())
            }
            fn visit_bodied_func(&mut self, m: $m, v: FuncId) {
                self.default_visit(m, v.into())
            }
            fn visit_external_func(&mut self, m: $m, v: FuncId) {
                self.default_visit(m, v.into())
            }
            fn visit_internal_func(&mut self, m: $m, v: FuncId) {
                self.default_visit(m, v.into())
            }
            fn visit_llvm_func(&mut self, m: $m, v: FuncId) {
                self.default_visit(m, v.into())
            }

            fn visit_var_value(&mut self, m: $m, v: ValueId) {
                self.default_visit(m, v.into())
            }
            fn visit_pointer_value(&mut self, m: $m, v: ValueId) {
                self.default_visit(m, v.into())
            }
            fn visit_series_flow(&mut self, m: $m, v: ValueId) {
                self.default_visit(m, v.into())
            }
            fn visit_if_flow(&mut self, m: $m, v: ValueId) {
                self.default_visit(m, v.into())
            }
            fn visit_while_flow(&mut self, m: $m, v: ValueId) {
                self.default_visit(m, v.into())
            }
            fn visit_for_flow(&mut self, m: $m, v: ValueId) {
                self.default_visit(m, v.into())
            }
            fn visit_imperative_for_flow(&mut self, m: $m, v: ValueId) {
                self.default_visit(m, v.into())
            }
            fn visit_try_catch_flow(&mut self, m: $m, v: ValueId) {
                self.default_visit(m, v.into())
            }
            fn visit_pipeline_flow(&mut self, m: $m, v: ValueId) {
                self.default_visit(m, v.into())
            }
            fn visit_int_const(&mut self, m: $m, v: ValueId) {
                self.default_visit(m, v.into())
            }
            fn visit_float_const(&mut self, m: $m, v: ValueId) {
                self.default_visit(m, v.into())
            }
            fn visit_bool_const(&mut self, m: $m, v: ValueId) {
                self.default_visit(m, v.into())
            }
            fn visit_string_const(&mut self, m: $m, v: ValueId) {
                self.default_visit(m, v.into())
            }
            fn visit_bytes_const(&mut self, m: $m, v: ValueId) {
                self.default_visit(m, v.into())
            }
            fn visit_assign_instr(&mut self, m: $m, v: ValueId) {
                self.default_visit(m, v.into())
            }
            fn visit_extract_instr(&mut self, m: $m, v: ValueId) {
                self.default_visit(m, v.into())
            }
            fn visit_insert_instr(&mut self, m: $m, v: ValueId) {
                self.default_visit(m, v.into())
            }
            fn visit_call_instr(&mut self, m: $m, v: ValueId) {
                self.default_visit(m, v.into())
            }
            fn visit_stack_alloc_instr(&mut self, m: $m, v: ValueId) {
                self.default_visit(m, v.into())
            }
            fn visit_yield_in_instr(&mut self, m: $m, v: ValueId) {
                self.default_visit(m, v.into())
            }
            fn visit_ternary_instr(&mut self, m: $m, v: ValueId) {
                self.default_visit(m, v.into())
            }
            fn visit_break_instr(&mut self, m: $m, v: ValueId) {
                self.default_visit(m, v.into())
            }
            fn visit_continue_instr(&mut self, m: $m, v: ValueId) {
                self.default_visit(m, v.into())
            }
            fn visit_return_instr(&mut self, m: $m, v: ValueId) {
                self.default_visit(m, v.into())
            }
            fn visit_yield_instr(&mut self, m: $m, v: ValueId) {
                self.default_visit(m, v.into())
            }
            fn visit_await_instr(&mut self, m: $m, v: ValueId) {
                self.default_visit(m, v.into())
            }
            fn visit_throw_instr(&mut self, m: $m, v: ValueId) {
                self.default_visit(m, v.into())
            }
            fn visit_flow_instr(&mut self, m: $m, v: ValueId) {
                self.default_visit(m, v.into())
            }
            /// `util::AnyValue` / `util::AnyFlow` (matching wildcards).
            fn visit_any_value(&mut self, m: $m, v: ValueId) {
                self.default_visit(m, v.into())
            }

            fn visit_int_type(&mut self, m: $m, t: TypeId) {
                self.default_visit(m, t.into())
            }
            fn visit_float_type(&mut self, m: $m, t: TypeId) {
                self.default_visit(m, t.into())
            }
            fn visit_float32_type(&mut self, m: $m, t: TypeId) {
                self.default_visit(m, t.into())
            }
            fn visit_float16_type(&mut self, m: $m, t: TypeId) {
                self.default_visit(m, t.into())
            }
            fn visit_bfloat16_type(&mut self, m: $m, t: TypeId) {
                self.default_visit(m, t.into())
            }
            fn visit_float128_type(&mut self, m: $m, t: TypeId) {
                self.default_visit(m, t.into())
            }
            fn visit_bool_type(&mut self, m: $m, t: TypeId) {
                self.default_visit(m, t.into())
            }
            fn visit_record_type(&mut self, m: $m, t: TypeId) {
                self.default_visit(m, t.into())
            }
            fn visit_ref_type(&mut self, m: $m, t: TypeId) {
                self.default_visit(m, t.into())
            }
            fn visit_func_type(&mut self, m: $m, t: TypeId) {
                self.default_visit(m, t.into())
            }
            fn visit_optional_type(&mut self, m: $m, t: TypeId) {
                self.default_visit(m, t.into())
            }
            fn visit_pointer_type(&mut self, m: $m, t: TypeId) {
                self.default_visit(m, t.into())
            }
            fn visit_generator_type(&mut self, m: $m, t: TypeId) {
                self.default_visit(m, t.into())
            }
            fn visit_vector_type(&mut self, m: $m, t: TypeId) {
                self.default_visit(m, t.into())
            }
            fn visit_union_type(&mut self, m: $m, t: TypeId) {
                self.default_visit(m, t.into())
            }
        }
    };
}

visitor_trait!(Visitor, &mut Module);
visitor_trait!(ConstVisitor, &Module);

/// Which `visit_*` method a value dispatches to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ValueVisitKind {
    VarValue,
    PointerValue,
    SeriesFlow,
    IfFlow,
    WhileFlow,
    ForFlow,
    ImperativeForFlow,
    TryCatchFlow,
    PipelineFlow,
    IntConst,
    FloatConst,
    BoolConst,
    StringConst,
    BytesConst,
    AssignInstr,
    ExtractInstr,
    InsertInstr,
    CallInstr,
    StackAllocInstr,
    YieldInInstr,
    TernaryInstr,
    BreakInstr,
    ContinueInstr,
    ReturnInstr,
    YieldInstr,
    AwaitInstr,
    ThrowInstr,
    FlowInstr,
    Any,
}

pub fn value_visit_kind(k: &ValueKind) -> ValueVisitKind {
    use ValueVisitKind as K;
    match k {
        ValueKind::IntConst(_) => K::IntConst,
        ValueKind::FloatConst(_) => K::FloatConst,
        ValueKind::BoolConst(_) => K::BoolConst,
        ValueKind::StringConst(_) => K::StringConst,
        ValueKind::BytesConst(_) => K::BytesConst,
        ValueKind::VarValue(_) => K::VarValue,
        ValueKind::PointerValue(_) => K::PointerValue,
        ValueKind::SeriesFlow(_) => K::SeriesFlow,
        ValueKind::WhileFlow(_) => K::WhileFlow,
        ValueKind::ForFlow(_) => K::ForFlow,
        ValueKind::ImperativeForFlow(_) => K::ImperativeForFlow,
        ValueKind::IfFlow(_) => K::IfFlow,
        ValueKind::TryCatchFlow(_) => K::TryCatchFlow,
        ValueKind::PipelineFlow(_) => K::PipelineFlow,
        ValueKind::AssignInstr(_) => K::AssignInstr,
        ValueKind::ExtractInstr(_) => K::ExtractInstr,
        ValueKind::InsertInstr(_) => K::InsertInstr,
        ValueKind::CallInstr(_) => K::CallInstr,
        ValueKind::StackAllocInstr(_) => K::StackAllocInstr,
        ValueKind::YieldInInstr(_) => K::YieldInInstr,
        ValueKind::TernaryInstr(_) => K::TernaryInstr,
        ValueKind::BreakInstr(_) => K::BreakInstr,
        ValueKind::ContinueInstr(_) => K::ContinueInstr,
        ValueKind::ReturnInstr(_) => K::ReturnInstr,
        ValueKind::YieldInstr(_) => K::YieldInstr,
        ValueKind::AwaitInstr(_) => K::AwaitInstr,
        ValueKind::ThrowInstr(_) => K::ThrowInstr,
        ValueKind::FlowInstr(_) => K::FlowInstr,
        ValueKind::Any { .. } => K::Any,
    }
}

macro_rules! dispatch_value {
    ($vis:expr, $m:expr, $v:expr, $k:expr) => {{
        use ValueVisitKind as K;
        match $k {
            K::VarValue => $vis.visit_var_value($m, $v),
            K::PointerValue => $vis.visit_pointer_value($m, $v),
            K::SeriesFlow => $vis.visit_series_flow($m, $v),
            K::IfFlow => $vis.visit_if_flow($m, $v),
            K::WhileFlow => $vis.visit_while_flow($m, $v),
            K::ForFlow => $vis.visit_for_flow($m, $v),
            K::ImperativeForFlow => $vis.visit_imperative_for_flow($m, $v),
            K::TryCatchFlow => $vis.visit_try_catch_flow($m, $v),
            K::PipelineFlow => $vis.visit_pipeline_flow($m, $v),
            K::IntConst => $vis.visit_int_const($m, $v),
            K::FloatConst => $vis.visit_float_const($m, $v),
            K::BoolConst => $vis.visit_bool_const($m, $v),
            K::StringConst => $vis.visit_string_const($m, $v),
            K::BytesConst => $vis.visit_bytes_const($m, $v),
            K::AssignInstr => $vis.visit_assign_instr($m, $v),
            K::ExtractInstr => $vis.visit_extract_instr($m, $v),
            K::InsertInstr => $vis.visit_insert_instr($m, $v),
            K::CallInstr => $vis.visit_call_instr($m, $v),
            K::StackAllocInstr => $vis.visit_stack_alloc_instr($m, $v),
            K::YieldInInstr => $vis.visit_yield_in_instr($m, $v),
            K::TernaryInstr => $vis.visit_ternary_instr($m, $v),
            K::BreakInstr => $vis.visit_break_instr($m, $v),
            K::ContinueInstr => $vis.visit_continue_instr($m, $v),
            K::ReturnInstr => $vis.visit_return_instr($m, $v),
            K::YieldInstr => $vis.visit_yield_instr($m, $v),
            K::AwaitInstr => $vis.visit_await_instr($m, $v),
            K::ThrowInstr => $vis.visit_throw_instr($m, $v),
            K::FlowInstr => $vis.visit_flow_instr($m, $v),
            K::Any => $vis.visit_any_value($m, $v),
        }
    }};
}

#[derive(Clone, Copy)]
enum VarVisitKind {
    Var,
    Bodied,
    External,
    Internal,
    Llvm,
}

fn var_visit_kind(m: &Module, v: VarId) -> VarVisitKind {
    match m.func(v).map(|f| &f.kind) {
        None => VarVisitKind::Var,
        Some(FuncKind::Bodied { .. }) => VarVisitKind::Bodied,
        Some(FuncKind::External) => VarVisitKind::External,
        Some(FuncKind::Internal { .. }) => VarVisitKind::Internal,
        Some(FuncKind::Llvm { .. }) => VarVisitKind::Llvm,
    }
}

#[derive(Clone, Copy)]
enum TypeVisitKind {
    Int,
    Float,
    Float32,
    Float16,
    BFloat16,
    Float128,
    Bool,
    Record,
    Ref,
    Func,
    Optional,
    Pointer,
    Generator,
    Vector,
    Union,
}

fn type_visit_kind(m: &Module, t: TypeId) -> TypeVisitKind {
    use TypeVisitKind as K;
    match m.type_kind(t) {
        TypeKind::Int { .. } => K::Int,
        TypeKind::Float => K::Float,
        TypeKind::Float32 => K::Float32,
        TypeKind::Float16 => K::Float16,
        TypeKind::BFloat16 => K::BFloat16,
        TypeKind::Float128 => K::Float128,
        TypeKind::Bool => K::Bool,
        TypeKind::Record { .. } => K::Record,
        TypeKind::Ref { .. } => K::Ref,
        TypeKind::Func { .. } => K::Func,
        TypeKind::Optional { .. } => K::Optional,
        TypeKind::Pointer { .. } => K::Pointer,
        TypeKind::Generator { .. } => K::Generator,
        TypeKind::Vector { .. } => K::Vector,
        TypeKind::Union { .. } => K::Union,
    }
}

macro_rules! dispatch_var {
    ($vis:expr, $m:expr, $v:expr, $k:expr) => {
        match $k {
            VarVisitKind::Var => $vis.visit_var($m, $v),
            VarVisitKind::Bodied => $vis.visit_bodied_func($m, $v),
            VarVisitKind::External => $vis.visit_external_func($m, $v),
            VarVisitKind::Internal => $vis.visit_internal_func($m, $v),
            VarVisitKind::Llvm => $vis.visit_llvm_func($m, $v),
        }
    };
}

macro_rules! dispatch_type {
    ($vis:expr, $m:expr, $t:expr, $k:expr) => {{
        use TypeVisitKind as K;
        match $k {
            K::Int => $vis.visit_int_type($m, $t),
            K::Float => $vis.visit_float_type($m, $t),
            K::Float32 => $vis.visit_float32_type($m, $t),
            K::Float16 => $vis.visit_float16_type($m, $t),
            K::BFloat16 => $vis.visit_bfloat16_type($m, $t),
            K::Float128 => $vis.visit_float128_type($m, $t),
            K::Bool => $vis.visit_bool_type($m, $t),
            K::Record => $vis.visit_record_type($m, $t),
            K::Ref => $vis.visit_ref_type($m, $t),
            K::Func => $vis.visit_func_type($m, $t),
            K::Optional => $vis.visit_optional_type($m, $t),
            K::Pointer => $vis.visit_pointer_type($m, $t),
            K::Generator => $vis.visit_generator_type($m, $t),
            K::Vector => $vis.visit_vector_type($m, $t),
            K::Union => $vis.visit_union_type($m, $t),
        }
    }};
}

/// `value->accept(visitor)`.
pub fn accept_value<V: Visitor + ?Sized>(vis: &mut V, m: &mut Module, v: ValueId) {
    let v = m.actual_value(v);
    let k = value_visit_kind(m.kind(v));
    dispatch_value!(vis, m, v, k)
}
/// `var->accept(visitor)`.
pub fn accept_var<V: Visitor + ?Sized>(vis: &mut V, m: &mut Module, v: VarId) {
    let v = m.actual_var(v);
    let k = var_visit_kind(m, v);
    dispatch_var!(vis, m, v, k)
}
/// `type->accept(visitor)`.
pub fn accept_type<V: Visitor + ?Sized>(vis: &mut V, m: &mut Module, t: TypeId) {
    let t = m.actual_type(t);
    let k = type_visit_kind(m, t);
    dispatch_type!(vis, m, t, k)
}
/// `node->accept(visitor)`.
pub fn accept<V: Visitor + ?Sized>(vis: &mut V, m: &mut Module, n: NodeRef) {
    match n {
        NodeRef::Value(v) => accept_value(vis, m, v),
        NodeRef::Var(v) => accept_var(vis, m, v),
        NodeRef::Type(t) => accept_type(vis, m, t),
        NodeRef::Module => vis.visit_module(m),
    }
}

/// `value->accept(constVisitor)`.
pub fn const_accept_value<V: ConstVisitor + ?Sized>(vis: &mut V, m: &Module, v: ValueId) {
    let v = m.actual_value(v);
    let k = value_visit_kind(m.kind(v));
    dispatch_value!(vis, m, v, k)
}
pub fn const_accept_var<V: ConstVisitor + ?Sized>(vis: &mut V, m: &Module, v: VarId) {
    let v = m.actual_var(v);
    let k = var_visit_kind(m, v);
    dispatch_var!(vis, m, v, k)
}
pub fn const_accept_type<V: ConstVisitor + ?Sized>(vis: &mut V, m: &Module, t: TypeId) {
    let t = m.actual_type(t);
    let k = type_visit_kind(m, t);
    dispatch_type!(vis, m, t, k)
}
pub fn const_accept<V: ConstVisitor + ?Sized>(vis: &mut V, m: &Module, n: NodeRef) {
    match n {
        NodeRef::Value(v) => const_accept_value(vis, m, v),
        NodeRef::Var(v) => const_accept_var(vis, m, v),
        NodeRef::Type(t) => const_accept_type(vis, m, t),
        NodeRef::Module => vis.visit_module(m),
    }
}
