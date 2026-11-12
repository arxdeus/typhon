//! Structural IR equality.
//!
//! `util::AnyValue/AnyFlow/AnyVar/AnyFunc` wildcards are `ValueKind::Any` and
//! `VarData::any` (see `Module::nr_any_value`, `Module::nr_any_var`).

use crate::ids::{NodeRef, TypeId, ValueId, VarId};
use crate::module::Module;
use crate::types::Generic;
use crate::value::ValueKind;
use crate::var::FuncKind;

struct Matcher<'a> {
    m: &'a Module,
    check_name: bool,
    var_id_match: bool,
}

impl<'a> Matcher<'a> {
    fn types(&self, a: Option<TypeId>, b: Option<TypeId>) -> bool {
        match (a, b) {
            (None, None) => true,
            (Some(a), Some(b)) => self.m.type_is(a, b),
            _ => false,
        }
    }

    fn values(&self, a: Option<ValueId>, b: Option<ValueId>) -> bool {
        match (a, b) {
            (None, None) => true,
            (Some(a), Some(b)) => self.value(a, b),
            _ => false,
        }
    }

    #[allow(dead_code)]
    fn vars(&self, a: Option<VarId>, b: Option<VarId>) -> bool {
        match (a, b) {
            (None, None) => true,
            (Some(a), Some(b)) => self.var(a, b),
            _ => false,
        }
    }

    fn value_lists(&self, a: &[ValueId], b: &[ValueId]) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| self.value(*x, *y))
    }

    fn compare_vars(&self, x: VarId, y: VarId) -> bool {
        self.types(self.m.var_type(x), self.m.var_type(y))
            && (!self.var_id_match || self.m.var_uid(x) == self.m.var_uid(y))
    }

    fn compare_funcs(&self, x: VarId, y: VarId) -> bool {
        if !self.compare_vars(x, y) {
            return false;
        }
        let ax = self.m.func_args(x);
        let ay = self.m.func_args(y);
        ax.len() == ay.len() && ax.iter().zip(&ay).all(|(a, b)| self.var(*a, *b))
    }

    fn var(&self, x: VarId, y: VarId) -> bool {
        let m = self.m;
        let (x, y) = (m.actual_var(x), m.actual_var(y));
        let (dx, dy) = (m.var(x), m.var(y));
        if dx.any || dy.any {
            return true;
        }
        if dx.kind_name() != dy.kind_name() || (!self.check_name && dx.base.name != dy.base.name) {
            return false;
        }
        let (Some(fx), Some(fy)) = (dx.func.as_deref(), dy.func.as_deref()) else {
            return self.compare_vars(x, y);
        };
        match (&fx.kind, &fy.kind) {
            (FuncKind::Bodied { symbols: sx, body: bx }, FuncKind::Bodied { symbols: sy, body: by }) => {
                self.compare_funcs(x, y)
                    && sx.len() == sy.len()
                    && sx.iter().zip(sy).all(|(a, b)| self.var(*a, *b))
                    && self.values(*bx, *by)
            }
            (FuncKind::External, FuncKind::External) => {
                fx.unmangled_name == fy.unmangled_name && self.compare_funcs(x, y)
            }
            (
                FuncKind::Internal { intrinsic: ix, intrinsic_type: tx },
                FuncKind::Internal { intrinsic: iy, intrinsic_type: ty },
            ) => {
                fx.parent_type.map(|t| m.actual_type(t)) == fy.parent_type.map(|t| m.actual_type(t))
                    && ix == iy
                    && tx.map(|t| m.actual_type(t)) == ty.map(|t| m.actual_type(t))
                    && self.compare_funcs(x, y)
            }
            (
                FuncKind::Llvm { literals: lx, declarations: dx2, body: bx },
                FuncKind::Llvm { literals: ly, declarations: dy2, body: by },
            ) => {
                lx.len() == ly.len()
                    && lx.iter().zip(ly).all(|(a, b)| match (a, b) {
                        (Generic::Static(a), Generic::Static(b)) => a == b,
                        (Generic::StaticStr(a), Generic::StaticStr(b)) => a == b,
                        (Generic::Type(a), Generic::Type(b)) => m.type_is(*a, *b),
                        _ => false,
                    })
                    && dx2 == dy2
                    && bx == by
                    && self.compare_funcs(x, y)
            }
            _ => false,
        }
    }

    fn value(&self, x: ValueId, y: ValueId) -> bool {
        let m = self.m;
        let (x, y) = (m.actual_value(x), m.actual_value(y));
        let (kx, ky) = (m.kind(x), m.kind(y));
        if kx.is_any() || ky.is_any() {
            return true;
        }
        if kx.kind_name() != ky.kind_name() || (!self.check_name && m.value_name(x) != m.value_name(y)) {
            return false;
        }
        match (kx, ky) {
            (ValueKind::VarValue(a), ValueKind::VarValue(b)) => self.compare_vars(a.var, b.var),
            (ValueKind::PointerValue(a), ValueKind::PointerValue(b)) => {
                self.compare_vars(a.var, b.var) && a.fields == b.fields
            }
            (ValueKind::SeriesFlow(a), ValueKind::SeriesFlow(b)) => self.value_lists(&a.series, &b.series),
            (ValueKind::IfFlow(a), ValueKind::IfFlow(b)) => {
                self.value(a.cond, b.cond)
                    && self.value(a.true_branch, b.true_branch)
                    && self.values(a.false_branch, b.false_branch)
            }
            (ValueKind::WhileFlow(a), ValueKind::WhileFlow(b)) => {
                self.value(a.cond, b.cond) && self.value(a.body, b.body)
            }
            (ValueKind::ForFlow(a), ValueKind::ForFlow(b)) => {
                a.is_async == b.is_async
                    && self.value(a.iter, b.iter)
                    && self.value(a.body, b.body)
                    && self.var(a.var, b.var)
            }
            (ValueKind::ImperativeForFlow(a), ValueKind::ImperativeForFlow(b)) => {
                self.var(a.var, b.var)
                    && self.value(a.body, b.body)
                    && self.value(a.start, b.start)
                    && a.step == b.step
                    && self.value(a.end, b.end)
            }
            // Note: the reference starts from `result && ...` where `result` is still false, so
            // two try-catch flows never match. That bug is preserved for fidelity.
            (ValueKind::TryCatchFlow(_), ValueKind::TryCatchFlow(_)) => false,
            (ValueKind::PipelineFlow(a), ValueKind::PipelineFlow(b)) => {
                a.stages.len() == b.stages.len()
                    && a.stages.iter().zip(&b.stages).all(|(s, t)| {
                        self.value(s.callee, t.callee)
                            && s.args.len() == t.args.len()
                            && s.args.iter().zip(&t.args).all(|(p, q)| self.values(*p, *q))
                            && s.generator == t.generator
                            && s.parallel == t.parallel
                    })
            }
            (ValueKind::IntConst(a), ValueKind::IntConst(b)) => m.type_is(a.ty, b.ty) && a.val == b.val,
            (ValueKind::FloatConst(a), ValueKind::FloatConst(b)) => m.type_is(a.ty, b.ty) && a.val == b.val,
            (ValueKind::BoolConst(a), ValueKind::BoolConst(b)) => m.type_is(a.ty, b.ty) && a.val == b.val,
            (ValueKind::StringConst(a), ValueKind::StringConst(b))
            | (ValueKind::BytesConst(a), ValueKind::BytesConst(b)) => m.type_is(a.ty, b.ty) && a.val == b.val,
            (ValueKind::AssignInstr(a), ValueKind::AssignInstr(b)) => {
                self.var(a.lhs, b.lhs) && self.value(a.rhs, b.rhs)
            }
            (ValueKind::ExtractInstr(a), ValueKind::ExtractInstr(b)) => {
                a.polymorphic == b.polymorphic && self.value(a.val, b.val) && a.field == b.field
            }
            (ValueKind::InsertInstr(a), ValueKind::InsertInstr(b)) => {
                a.polymorphic == b.polymorphic
                    && self.value(a.lhs, b.lhs)
                    && a.field == b.field
                    && self.value(a.rhs, b.rhs)
            }
            (ValueKind::CallInstr(a), ValueKind::CallInstr(b)) => {
                self.value(a.callee, b.callee) && self.value_lists(&a.args, &b.args)
            }
            (ValueKind::StackAllocInstr(a), ValueKind::StackAllocInstr(b)) => {
                a.count == b.count && m.type_is(a.ptr_type, b.ptr_type)
            }
            (ValueKind::YieldInInstr(a), ValueKind::YieldInInstr(b)) => m.type_is(a.ty, b.ty),
            (ValueKind::TernaryInstr(a), ValueKind::TernaryInstr(b)) => {
                self.value(a.cond, b.cond)
                    && self.value(a.true_value, b.true_value)
                    && self.value(a.false_value, b.false_value)
            }
            (ValueKind::BreakInstr(a), ValueKind::BreakInstr(b)) => self.values(a.loop_, b.loop_),
            (ValueKind::ContinueInstr(a), ValueKind::ContinueInstr(b)) => self.values(a.loop_, b.loop_),
            (ValueKind::ReturnInstr(a), ValueKind::ReturnInstr(b)) => self.values(a.value, b.value),
            (ValueKind::YieldInstr(a), ValueKind::YieldInstr(b)) => self.values(a.value, b.value),
            (ValueKind::AwaitInstr(a), ValueKind::AwaitInstr(b)) => {
                m.type_is(a.ty, b.ty) && self.value(a.value, b.value) && a.generator == b.generator
            }
            (ValueKind::ThrowInstr(a), ValueKind::ThrowInstr(b)) => {
                self.values(a.value, b.value) && self.values(a.fallback, b.fallback)
            }
            (ValueKind::FlowInstr(a), ValueKind::FlowInstr(b)) => {
                self.value(a.flow, b.flow) && self.value(a.val, b.val)
            }
            _ => false,
        }
    }

    fn node(&self, a: NodeRef, b: NodeRef) -> bool {
        match (a, b) {
            (NodeRef::Type(x), NodeRef::Type(y)) => self.m.type_is(x, y),
            (NodeRef::Type(_), _) | (_, NodeRef::Type(_)) => false,
            (NodeRef::Value(x), NodeRef::Value(y)) => self.value(x, y),
            (NodeRef::Var(x), NodeRef::Var(y)) => self.var(x, y),
            (NodeRef::Module, NodeRef::Module) => true,
            _ => false,
        }
    }
}

/// `util::match(a, b, checkNames, varIdMatch)`.
///
/// Note:, `check_names == false` means names *are* compared (
/// visitor's `checkName` flag skips the name comparison when true), and nested
/// comparisons are built with `MatchVisitor(checkName)` so `var_id_match` is
/// effectively ignored by the reference (`match` itself calls `MatchVisitor(checkNames)`).
/// Preserved: `var_id_match` is accepted but has no effect.
pub fn match_nodes(m: &Module, a: Option<NodeRef>, b: Option<NodeRef>, check_names: bool, var_id_match: bool) -> bool {
    let _ = var_id_match;
    match (a, b) {
        (None, None) => true,
        (Some(a), Some(b)) => Matcher { m, check_name: check_names, var_id_match: false }.node(a, b),
        _ => false,
    }
}

/// Like [`match_nodes`] but actually enforcing equal variable ids (the evident intent
/// of the reference `varIdMatch` flag).
pub fn match_nodes_strict_ids(m: &Module, a: NodeRef, b: NodeRef, check_names: bool) -> bool {
    Matcher { m, check_name: check_names, var_id_match: true }.node(a, b)
}

/// `util::match(a, b)` for values.
pub fn match_values(m: &Module, a: ValueId, b: ValueId) -> bool {
    match_nodes(m, Some(a.into()), Some(b.into()), false, false)
}
/// `util::match(a, b)` for variables / functions.
pub fn match_vars(m: &Module, a: VarId, b: VarId) -> bool {
    match_nodes(m, Some(a.into()), Some(b.into()), false, false)
}
/// `util::match(a, b)` for types.
pub fn match_types(m: &Module, a: TypeId, b: TypeId) -> bool {
    match_nodes(m, Some(a.into()), Some(b.into()), false, false)
}
