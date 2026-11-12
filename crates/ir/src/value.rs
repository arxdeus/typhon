//! And `VarValue`/`PointerValue`
//! from `var.h`: value node payloads.

use crate::base::NodeBase;
use crate::ids::{Id, TypeId, ValueId, VarId};
use crate::schedule::OMPSched;

#[derive(Clone, Debug, PartialEq)]
pub struct TemplatedConst<T> {
    pub val: T,
    pub ty: TypeId,
}

pub type IntConst = TemplatedConst<i64>;
pub type FloatConst = TemplatedConst<f64>;
pub type BoolConst = TemplatedConst<bool>;
pub type StringConst = TemplatedConst<String>;
pub type BytesConst = TemplatedConst<String>;

#[derive(Clone, Debug, PartialEq)]
pub struct VarValue {
    pub var: VarId,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PointerValue {
    pub var: VarId,
    pub fields: Vec<String>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct SeriesFlow {
    pub series: Vec<ValueId>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct WhileFlow {
    pub cond: ValueId,
    pub body: ValueId,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ForFlow {
    pub iter: ValueId,
    pub body: ValueId,
    pub var: VarId,
    pub schedule: Option<Box<OMPSched>>,
    pub is_async: bool,
}

impl ForFlow {
    pub fn is_parallel(&self) -> bool {
        self.schedule.is_some()
    }
    /// `setParallel(a)`.
    pub fn set_parallel(&mut self, a: bool) {
        self.schedule = if a { Some(Box::default()) } else { None };
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ImperativeForFlow {
    pub start: ValueId,
    pub step: i64,
    pub end: ValueId,
    pub body: ValueId,
    pub var: VarId,
    pub schedule: Option<Box<OMPSched>>,
}

impl ImperativeForFlow {
    pub fn is_parallel(&self) -> bool {
        self.schedule.is_some()
    }
    pub fn set_parallel(&mut self, a: bool) {
        self.schedule = if a { Some(Box::default()) } else { None };
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct IfFlow {
    pub cond: ValueId,
    pub true_branch: ValueId,
    pub false_branch: Option<ValueId>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Catch {
    pub handler: ValueId,
    pub ty: Option<TypeId>,
    pub var: Option<VarId>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TryCatchFlow {
    pub catches: Vec<Catch>,
    pub body: ValueId,
    pub else_: Option<ValueId>,
    pub finally: Option<ValueId>,
}

/// `None` args are the pipeline placeholder (`...`).
#[derive(Clone, Debug, PartialEq)]
pub struct Stage {
    pub callee: ValueId,
    pub args: Vec<Option<ValueId>>,
    pub generator: bool,
    pub parallel: bool,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct PipelineFlow {
    pub stages: Vec<Stage>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct AssignInstr {
    pub lhs: VarId,
    pub rhs: ValueId,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExtractInstr {
    pub val: ValueId,
    pub field: String,
    pub polymorphic: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct InsertInstr {
    pub lhs: ValueId,
    pub field: String,
    pub rhs: ValueId,
    pub polymorphic: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CallInstr {
    pub callee: ValueId,
    pub args: Vec<ValueId>,
}

impl CallInstr {
    pub fn num_args(&self) -> usize {
        self.args.len()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct StackAllocInstr {
    pub ptr_type: TypeId,
    pub count: i64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct YieldInInstr {
    pub ty: TypeId,
    pub suspend: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TernaryInstr {
    pub cond: ValueId,
    pub true_value: ValueId,
    pub false_value: ValueId,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct BreakInstr {
    pub loop_: Option<ValueId>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ContinueInstr {
    pub loop_: Option<ValueId>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ReturnInstr {
    pub value: Option<ValueId>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct YieldInstr {
    pub value: Option<ValueId>,
    pub final_: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct AwaitInstr {
    pub value: ValueId,
    pub ty: TypeId,
    pub generator: bool,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ThrowInstr {
    pub value: Option<ValueId>,
    pub fallback: Option<ValueId>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FlowInstr {
    pub flow: ValueId,
    pub val: ValueId,
}

/// Port of the `Value` class hierarchy.
#[derive(Clone, Debug)]
pub enum ValueKind {
    IntConst(IntConst),
    FloatConst(FloatConst),
    BoolConst(BoolConst),
    StringConst(StringConst),
    BytesConst(BytesConst),
    VarValue(VarValue),
    PointerValue(PointerValue),
    SeriesFlow(SeriesFlow),
    WhileFlow(WhileFlow),
    ForFlow(ForFlow),
    ImperativeForFlow(ImperativeForFlow),
    IfFlow(IfFlow),
    TryCatchFlow(TryCatchFlow),
    PipelineFlow(PipelineFlow),
    AssignInstr(AssignInstr),
    ExtractInstr(ExtractInstr),
    InsertInstr(InsertInstr),
    CallInstr(CallInstr),
    StackAllocInstr(StackAllocInstr),
    YieldInInstr(YieldInInstr),
    TernaryInstr(TernaryInstr),
    BreakInstr(BreakInstr),
    ContinueInstr(ContinueInstr),
    ReturnInstr(ReturnInstr),
    YieldInstr(YieldInstr),
    AwaitInstr(AwaitInstr),
    ThrowInstr(ThrowInstr),
    FlowInstr(FlowInstr),
    /// `util::AnyValue` (`flow == false`) / `util::AnyFlow` (`flow == true`): wildcards for matching.
    Any { flow: bool },
}

impl ValueKind {
    /// `isA<Const>`.
    pub fn is_const(&self) -> bool {
        match self {
            ValueKind::IntConst(_)
            | ValueKind::FloatConst(_)
            | ValueKind::BoolConst(_)
            | ValueKind::StringConst(_)
            | ValueKind::BytesConst(_) => true,
            _ => false,
        }
    }
    /// `isA<Flow>`.
    pub fn is_flow(&self) -> bool {
        match self {
            ValueKind::SeriesFlow(_)
            | ValueKind::WhileFlow(_)
            | ValueKind::ForFlow(_)
            | ValueKind::ImperativeForFlow(_)
            | ValueKind::IfFlow(_)
            | ValueKind::TryCatchFlow(_)
            | ValueKind::PipelineFlow(_) => true,
            ValueKind::Any { flow } => *flow,
            _ => false,
        }
    }
    /// `isA<Instr>`.
    pub fn is_instr(&self) -> bool {
        match self {
            ValueKind::AssignInstr(_)
            | ValueKind::ExtractInstr(_)
            | ValueKind::InsertInstr(_)
            | ValueKind::CallInstr(_)
            | ValueKind::StackAllocInstr(_)
            | ValueKind::YieldInInstr(_)
            | ValueKind::TernaryInstr(_)
            | ValueKind::BreakInstr(_)
            | ValueKind::ContinueInstr(_)
            | ValueKind::ReturnInstr(_)
            | ValueKind::YieldInstr(_)
            | ValueKind::AwaitInstr(_)
            | ValueKind::ThrowInstr(_)
            | ValueKind::FlowInstr(_) => true,
            _ => false,
        }
    }
    /// `isA<ControlFlowInstr>`.
    pub fn is_control_flow_instr(&self) -> bool {
        matches!(self, ValueKind::BreakInstr(_) | ValueKind::ContinueInstr(_) | ValueKind::ReturnInstr(_))
    }
    pub fn is_any(&self) -> bool {
        matches!(self, ValueKind::Any { .. })
    }
    /// Name of the original (diagnostics, matching).
    pub fn kind_name(&self) -> &'static str {
        match self {
            ValueKind::IntConst(_) => "IntConst",
            ValueKind::FloatConst(_) => "FloatConst",
            ValueKind::BoolConst(_) => "BoolConst",
            ValueKind::StringConst(_) => "StringConst",
            ValueKind::BytesConst(_) => "BytesConst",
            ValueKind::VarValue(_) => "VarValue",
            ValueKind::PointerValue(_) => "PointerValue",
            ValueKind::SeriesFlow(_) => "SeriesFlow",
            ValueKind::WhileFlow(_) => "WhileFlow",
            ValueKind::ForFlow(_) => "ForFlow",
            ValueKind::ImperativeForFlow(_) => "ImperativeForFlow",
            ValueKind::IfFlow(_) => "IfFlow",
            ValueKind::TryCatchFlow(_) => "TryCatchFlow",
            ValueKind::PipelineFlow(_) => "PipelineFlow",
            ValueKind::AssignInstr(_) => "AssignInstr",
            ValueKind::ExtractInstr(_) => "ExtractInstr",
            ValueKind::InsertInstr(_) => "InsertInstr",
            ValueKind::CallInstr(_) => "CallInstr",
            ValueKind::StackAllocInstr(_) => "StackAllocInstr",
            ValueKind::YieldInInstr(_) => "YieldInInstr",
            ValueKind::TernaryInstr(_) => "TernaryInstr",
            ValueKind::BreakInstr(_) => "BreakInstr",
            ValueKind::ContinueInstr(_) => "ContinueInstr",
            ValueKind::ReturnInstr(_) => "ReturnInstr",
            ValueKind::YieldInstr(_) => "YieldInstr",
            ValueKind::AwaitInstr(_) => "AwaitInstr",
            ValueKind::ThrowInstr(_) => "ThrowInstr",
            ValueKind::FlowInstr(_) => "FlowInstr",
            ValueKind::Any { flow: true } => "AnyFlow",
            ValueKind::Any { flow: false } => "AnyValue",
        }
    }

    /// Port of `doGetUsedValues()` (order matches C++).
    pub fn used_values(&self) -> Vec<ValueId> {
        match self {
            ValueKind::SeriesFlow(f) => f.series.clone(),
            ValueKind::WhileFlow(f) => vec![f.cond, f.body],
            ValueKind::ForFlow(f) => {
                let mut ret = f.schedule.as_ref().map(|s| s.used_values()).unwrap_or_default();
                ret.push(f.iter);
                ret.push(f.body);
                ret
            }
            ValueKind::ImperativeForFlow(f) => {
                let mut ret = f.schedule.as_ref().map(|s| s.used_values()).unwrap_or_default();
                ret.push(f.start);
                ret.push(f.end);
                ret.push(f.body);
                ret
            }
            ValueKind::IfFlow(f) => {
                let mut ret = vec![f.cond, f.true_branch];
                ret.extend(f.false_branch);
                ret
            }
            ValueKind::TryCatchFlow(f) => {
                let mut ret = vec![f.body];
                ret.extend(f.else_);
                ret.extend(f.finally);
                ret.extend(f.catches.iter().map(|c| c.handler));
                ret
            }
            ValueKind::PipelineFlow(f) => {
                let mut ret = Vec::new();
                for s in &f.stages {
                    ret.push(s.callee);
                    ret.extend(s.args.iter().flatten());
                }
                ret
            }
            ValueKind::AssignInstr(i) => vec![i.rhs],
            ValueKind::ExtractInstr(i) => vec![i.val],
            ValueKind::InsertInstr(i) => vec![i.lhs, i.rhs],
            ValueKind::CallInstr(i) => {
                let mut ret = i.args.clone();
                ret.push(i.callee);
                ret
            }
            ValueKind::TernaryInstr(i) => vec![i.cond, i.true_value, i.false_value],
            ValueKind::BreakInstr(i) => i.loop_.into_iter().collect(),
            ValueKind::ContinueInstr(i) => i.loop_.into_iter().collect(),
            ValueKind::ReturnInstr(i) => i.value.into_iter().collect(),
            ValueKind::YieldInstr(i) => i.value.into_iter().collect(),
            ValueKind::AwaitInstr(i) => vec![i.value],
            ValueKind::ThrowInstr(i) => i.value.into_iter().chain(i.fallback).collect(),
            ValueKind::FlowInstr(i) => vec![i.flow, i.val],
            _ => Vec::new(),
        }
    }

    pub fn used_types(&self) -> Vec<TypeId> {
        match self {
            ValueKind::IntConst(c) => vec![c.ty],
            ValueKind::FloatConst(c) => vec![c.ty],
            ValueKind::BoolConst(c) => vec![c.ty],
            ValueKind::StringConst(c) | ValueKind::BytesConst(c) => vec![c.ty],
            ValueKind::TryCatchFlow(f) => f.catches.iter().filter_map(|c| c.ty).collect(),
            ValueKind::StackAllocInstr(i) => vec![i.ptr_type],
            ValueKind::YieldInInstr(i) => vec![i.ty],
            ValueKind::AwaitInstr(i) => vec![i.ty],
            _ => Vec::new(),
        }
    }

    pub fn used_variables(&self) -> Vec<VarId> {
        match self {
            ValueKind::VarValue(v) => vec![v.var],
            ValueKind::PointerValue(v) => vec![v.var],
            ValueKind::ForFlow(f) => vec![f.var],
            ValueKind::ImperativeForFlow(f) => vec![f.var],
            ValueKind::TryCatchFlow(f) => f.catches.iter().filter_map(|c| c.var).collect(),
            ValueKind::AssignInstr(i) => vec![i.lhs],
            _ => Vec::new(),
        }
    }
}

/// Storage of one IR value.
#[derive(Clone, Debug)]
pub struct ValueData {
    pub base: NodeBase,
    pub kind: ValueKind,
    pub(crate) uid: Id,
    pub(crate) replacement: Option<ValueId>,
    pub(crate) alive: bool,
}

impl ValueData {
    pub fn kind(&self) -> &ValueKind {
        &self.kind
    }
    /// Raw `IdMixin` id of this slot (without replacement resolution).
    pub fn raw_uid(&self) -> Id {
        self.uid
    }
    pub fn replacement(&self) -> Option<ValueId> {
        self.replacement
    }
}
