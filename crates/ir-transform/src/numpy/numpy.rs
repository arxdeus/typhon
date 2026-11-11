//! Port of codon/cir/transform/numpy/numpy.cpp
//!
//! Pipeline: expose small array helpers, extract expressions, forward eligible
//! temporaries, then lower to fused loops or specialized runtime helpers.
//! forward.rs handles cross-assignment lifetimes; expr.rs emits expression
//! evaluation, layouts and reductions. This file coordinates those pieces.

use std::cell::RefCell;
use std::rc::Rc;

use rustc_hash::{FxHashMap, FxHashSet};
use typhon_common::mangle;
use typhon_ir::pass::{Analysis, AnalysisResult, Pass, PassContext};
use typhon_ir::types::TypeKind;
use typhon_ir::util::cloning::CloneVisitor;
use typhon_ir::util::irtools::{call, is_no_inline, make_var, return_type, series};
use typhon_ir::util::operator::{Operator, OperatorState};
use typhon_ir::{Attribute, FuncId, Id, KeyValueAttribute, Module, NodeRef, TypeId, ValueId, VarId, attr_id, magic};
use typhon_ir_analyze::dataflow::capture::CaptureAnalysis;
use typhon_ir_analyze::dataflow::cfg::{CFAnalysis, CFGraph, CfValue};
use typhon_ir_analyze::dataflow::dominator::DominatorAnalysis;
use typhon_ir_analyze::dataflow::reaching::{RDAnalysis, RDInspector, RDResult};
use typhon_ir_analyze::module::side_effect::{SideEffectAnalysis, SideEffectResult};

use super::expr::{CodegenContext, ExprArena, ExprId, Op, apply_in, call_args, callee_func, fusion_func, var_ty, vtype};
use super::forward::{do_forwarding, get_forwarding_dags};
use super::{
    BroadcastInfo, NumPyOptimizationUnit, NumPyPrimitiveTypes, NumPyType, TypeCode, is_array_type, is_none_type,
    is_ufunc_type, numpy_options, numpy_should_run, xlog,
};

// ---------------------------------------------------------------------------
// Walk: closure-based util::Operator
// ---------------------------------------------------------------------------

/// A `util::Operator` whose `preHook` (values only) is a closure. The closure gets
/// the walker (for `see`, `parents`, `parent_func`...) and the module.
#[derive(Default)]
pub struct Walk {
    state: OperatorState,
}

impl Operator for Walk {
    fn state(&self) -> &OperatorState {
        &self.state
    }
    fn state_mut(&mut self) -> &mut OperatorState {
        &mut self.state
    }
}

type WalkFn<'f> = dyn FnMut(&mut Walk, &mut Module, ValueId) + 'f;

struct WalkRunner<'f, 'g> {
    w: Walk,
    f: &'g mut WalkFn<'f>,
}

impl Operator for WalkRunner<'_, '_> {
    fn state(&self) -> &OperatorState {
        &self.w.state
    }
    fn state_mut(&mut self) -> &mut OperatorState {
        &mut self.w.state
    }
    fn pre_hook(&mut self, m: &mut Module, n: NodeRef) {
        if let NodeRef::Value(v) = n {
            (self.f)(&mut self.w, m, v);
        }
    }
}

impl Walk {
    pub fn new() -> Self {
        Self { state: OperatorState::new(false) }
    }
    pub fn children_first() -> Self {
        Self { state: OperatorState::new(true) }
    }
    fn go<'f>(self, m: &mut Module, n: NodeRef, f: &mut WalkFn<'f>) {
        let mut r = WalkRunner { w: self, f };
        r.reset();
        r.process(m, n);
    }
    /// `v->accept(op)`.
    pub fn run<'f>(self, m: &mut Module, v: ValueId, f: &mut WalkFn<'f>) {
        self.go(m, v.into(), f);
    }
    /// `func->accept(op)`.
    pub fn run_func<'f>(self, m: &mut Module, func: FuncId, f: &mut WalkFn<'f>) {
        self.go(m, func.into(), f);
    }
    /// `module->accept(op)`.
    pub fn run_module<'f>(self, m: &mut Module, f: &mut WalkFn<'f>) {
        self.go(m, NodeRef::Module, f);
    }
}

/// Bodied functions in `Operator::visit(Module*)` order (main first, deduplicated).
pub(crate) fn bodied_funcs(m: &Module) -> Vec<FuncId> {
    let mut seen = FxHashSet::default();
    let mut out = vec![];
    for f in std::iter::once(m.main_func()).chain(m.vars()) {
        if m.is_var_alive(f) && m.func(f).is_some_and(|d| d.is_bodied()) && seen.insert(m.var_uid(f)) {
            out.push(f);
        }
    }
    out
}

// ---------------------------------------------------------------------------
// small helpers
// ---------------------------------------------------------------------------

fn unmangled(m: &Module, f: FuncId) -> String {
    m.func(f).map(|d| d.unmangled_name.clone()).unwrap_or_default()
}
fn parent_type(m: &Module, f: FuncId) -> Option<TypeId> {
    m.func(f).and_then(|d| d.parent_type)
}
fn fname(m: &Module, f: FuncId) -> String {
    m.var_name(f).to_string()
}
fn npt(m: &mut Module, v: ValueId, t: &NumPyPrimitiveTypes) -> NumPyType {
    let ty = vtype(m, v);
    NumPyType::get(m, ty, t)
}
fn vt_is_none(m: &mut Module, v: ValueId, t: &NumPyPrimitiveTypes) -> bool {
    let ty = vtype(m, v);
    is_none_type(m, ty, t)
}
fn vt_name(m: &mut Module, v: ValueId) -> String {
    let ty = vtype(m, v);
    m.type_name(ty).to_string()
}
fn act(m: &Module, v: ValueId) -> ValueId {
    m.actual_value(v)
}
fn var_of(m: &Module, v: ValueId) -> Option<VarId> {
    m.as_var_value(v).map(|x| m.actual_var(x.var))
}
fn starts(s: &str, prefix: String) -> bool {
    s.starts_with(&prefix)
}

fn is_copy_index_type(m: &mut Module, ty: TypeId, types: &NumPyPrimitiveTypes) -> bool {
    let index = NumPyType::get(m, ty, types);
    if index.is_array() {
        return index.ndim > 0 && matches!(index.dtype, TypeCode::ArrBool | TypeCode::ArrI64);
    }
    if !m.type_name(ty).starts_with("Tuple[") {
        return false;
    }
    let mut has_array = false;
    for g in m.type_generics(ty) {
        let Some(el) = g.type_value() else { return false };
        if m.type_is(el, types.i64) {
            continue;
        }
        if !is_array_type(m, Some(el)) || !is_copy_index_type(m, el, types) {
            return false;
        }
        has_array = true;
    }
    has_array
}

fn is_owned_index(m: &mut Module, value: ValueId, types: &NumPyPrimitiveTypes) -> bool {
    let Some(args) = call_args(m, value) else { return false };
    let Some(callee) = callee_func(m, value) else { return false };
    if !is_array_type(m, parent_type(m, callee)) || unmangled(m, callee) != magic::GETITEM || args.len() != 2 {
        return false;
    }
    if !npt(m, value, types).is_array() {
        return false;
    }
    let f = npt(m, args[0], types);
    if !f.is_array() || f.ndim <= 0 {
        return false;
    }
    let bt = vtype(m, args[1]);
    is_copy_index_type(m, bt, types)
}

pub(crate) fn has_lifetime_tag(m: &Module, v: ValueId, tag: &str) -> bool {
    m.get_attribute(v.into(), attr_id::KEY_VALUE).and_then(|a| a.as_key_value()).is_some_and(|kv| kv.has(tag))
}

pub(crate) fn set_lifetime_tag(m: &mut Module, v: ValueId, tag: &str) {
    let n: NodeRef = v.into();
    if !m.has_attribute(n, attr_id::KEY_VALUE) {
        m.set_attribute(n, Attribute::KeyValue(KeyValueAttribute::default()));
    }
    if let Some(kv) = m.get_attribute_mut(n, attr_id::KEY_VALUE).and_then(|a| a.as_key_value_mut()) {
        kv.attributes.insert(tag.to_string(), String::new());
    }
}

// ---------------------------------------------------------------------------
// parse
// ---------------------------------------------------------------------------

struct Magic {
    name: &'static str,
    op: Op,
    args: usize,
    right: bool,
}

const MAGICS: &[Magic] = &[
    Magic { name: magic::POS, op: Op::Pos, args: 1, right: false },
    Magic { name: magic::NEG, op: Op::Neg, args: 1, right: false },
    Magic { name: magic::INVERT, op: Op::Invert, args: 1, right: false },
    Magic { name: magic::ABS, op: Op::Abs, args: 1, right: false },
    Magic { name: magic::ADD, op: Op::Add, args: 2, right: false },
    Magic { name: magic::SUB, op: Op::Sub, args: 2, right: false },
    Magic { name: magic::MUL, op: Op::Mul, args: 2, right: false },
    Magic { name: magic::MATMUL, op: Op::Matmul, args: 2, right: false },
    Magic { name: magic::TRUE_DIV, op: Op::TrueDiv, args: 2, right: false },
    Magic { name: magic::FLOOR_DIV, op: Op::FloorDiv, args: 2, right: false },
    Magic { name: magic::MOD, op: Op::Mod, args: 2, right: false },
    Magic { name: magic::POW, op: Op::Pow, args: 2, right: false },
    Magic { name: magic::LSHIFT, op: Op::Lshift, args: 2, right: false },
    Magic { name: magic::RSHIFT, op: Op::Rshift, args: 2, right: false },
    Magic { name: magic::AND, op: Op::And, args: 2, right: false },
    Magic { name: magic::OR, op: Op::Or, args: 2, right: false },
    Magic { name: magic::XOR, op: Op::Xor, args: 2, right: false },
    Magic { name: magic::RADD, op: Op::Add, args: 2, right: true },
    Magic { name: magic::RSUB, op: Op::Sub, args: 2, right: true },
    Magic { name: magic::RMUL, op: Op::Mul, args: 2, right: true },
    Magic { name: magic::RMATMUL, op: Op::Matmul, args: 2, right: true },
    Magic { name: magic::RTRUE_DIV, op: Op::TrueDiv, args: 2, right: true },
    Magic { name: magic::RFLOOR_DIV, op: Op::FloorDiv, args: 2, right: true },
    Magic { name: magic::RMOD, op: Op::Mod, args: 2, right: true },
    Magic { name: magic::RPOW, op: Op::Pow, args: 2, right: true },
    Magic { name: magic::RLSHIFT, op: Op::Lshift, args: 2, right: true },
    Magic { name: magic::RRSHIFT, op: Op::Rshift, args: 2, right: true },
    Magic { name: magic::RAND, op: Op::And, args: 2, right: true },
    Magic { name: magic::ROR, op: Op::Or, args: 2, right: true },
    Magic { name: magic::RXOR, op: Op::Xor, args: 2, right: true },
    Magic { name: magic::EQ, op: Op::Eq, args: 2, right: false },
    Magic { name: magic::NE, op: Op::Ne, args: 2, right: false },
    Magic { name: magic::LT, op: Op::Lt, args: 2, right: false },
    Magic { name: magic::LE, op: Op::Le, args: 2, right: false },
    Magic { name: magic::GT, op: Op::Gt, args: 2, right: false },
    Magic { name: magic::GE, op: Op::Ge, args: 2, right: false },
];

/// The ufunc name table (`NumPyUFunc`).
pub const UFUNCS: &[(&str, Op, usize)] = &[
    ("positive", Op::Pos, 1),
    ("negative", Op::Neg, 1),
    ("invert", Op::Invert, 1),
    ("abs", Op::Abs, 1),
    ("absolute", Op::Abs, 1),
    ("add", Op::Add, 2),
    ("subtract", Op::Sub, 2),
    ("multiply", Op::Mul, 2),
    ("divide", Op::TrueDiv, 2),
    ("floor_divide", Op::FloorDiv, 2),
    ("remainder", Op::Mod, 2),
    ("fmod", Op::Fmod, 2),
    ("power", Op::Pow, 2),
    ("left_shift", Op::Lshift, 2),
    ("right_shift", Op::Rshift, 2),
    ("bitwise_and", Op::And, 2),
    ("bitwise_or", Op::Or, 2),
    ("bitwise_xor", Op::Xor, 2),
    ("logical_and", Op::LogicalAnd, 2),
    ("logical_or", Op::LogicalOr, 2),
    ("logical_xor", Op::LogicalXor, 2),
    ("equal", Op::Eq, 2),
    ("not_equal", Op::Ne, 2),
    ("less", Op::Lt, 2),
    ("less_equal", Op::Le, 2),
    ("greater", Op::Gt, 2),
    ("greater_equal", Op::Ge, 2),
    ("minimum", Op::Min, 2),
    ("maximum", Op::Max, 2),
    ("fmin", Op::Fmin, 2),
    ("fmax", Op::Fmax, 2),
    ("sin", Op::Sin, 1),
    ("cos", Op::Cos, 1),
    ("tan", Op::Tan, 1),
    ("arcsin", Op::Arcsin, 1),
    ("arccos", Op::Arccos, 1),
    ("arctan", Op::Arctan, 1),
    ("arctan2", Op::Arctan2, 2),
    ("hypot", Op::Hypot, 2),
    ("sinh", Op::Sinh, 1),
    ("cosh", Op::Cosh, 1),
    ("tanh", Op::Tanh, 1),
    ("arcsinh", Op::Arcsinh, 1),
    ("arccosh", Op::Arccosh, 1),
    ("arctanh", Op::Arctanh, 1),
    ("conjugate", Op::Conj, 1),
    ("exp", Op::Exp, 1),
    ("exp2", Op::Exp2, 1),
    ("log", Op::Log, 1),
    ("log2", Op::Log2, 1),
    ("log10", Op::Log10, 1),
    ("expm1", Op::Expm1, 1),
    ("log1p", Op::Log1p, 1),
    ("sqrt", Op::Sqrt, 1),
    ("square", Op::Square, 1),
    ("cbrt", Op::Cbrt, 1),
    ("logaddexp", Op::Logaddexp, 2),
    ("logaddexp2", Op::Logaddexp2, 2),
    ("reciprocal", Op::Reciprocal, 1),
    ("rint", Op::Rint, 1),
    ("floor", Op::Floor, 1),
    ("ceil", Op::Ceil, 1),
    ("trunc", Op::Trunc, 1),
    ("isnan", Op::Isnan, 1),
    ("isinf", Op::Isinf, 1),
    ("isfinite", Op::Isfinite, 1),
    ("sign", Op::Sign, 1),
    ("signbit", Op::Signbit, 1),
    ("copysign", Op::Copysign, 2),
    ("spacing", Op::Spacing, 1),
    ("nextafter", Op::Nextafter, 2),
    ("deg2rad", Op::Deg2rad, 1),
    ("radians", Op::Deg2rad, 1),
    ("rad2deg", Op::Rad2deg, 1),
    ("degrees", Op::Rad2deg, 1),
    ("heaviside", Op::Heaviside, 2),
];

fn is_order(s: &str) -> bool {
    matches!(s, "C" | "F" | "A" | "K")
}

/// `parse(v, leaves, T, allowReduction, destination)`.
pub fn parse(
    m: &mut Module,
    a: &mut ExprArena,
    v: ValueId,
    leaves: &mut Vec<(ExprId, ValueId)>,
    t: &NumPyPrimitiveTypes,
    allow_reduction: bool,
    mut destination: Option<&mut Option<ValueId>>,
) -> Option<ExprId> {
    let ty = npt(m, v, t);
    if !ty.is_some() {
        return None;
    }

    if allow_reduction
        && let Some(function) = callee_func(m, v)
    {
        let args = call_args(m, v).unwrap();
        let name = unmangled(m, function);
        let op = match name.as_str() {
            "sum" => Some(Op::Sum),
            "prod" => Some(Op::Prod),
            "mean" => Some(Op::Mean),
            "any" => Some(Op::Any),
            "all" => Some(Op::All),
            "min" | "amin" => Some(Op::Amin),
            "max" | "amax" => Some(Op::Amax),
            _ => None,
        };
        let logical = name == "any" || name == "all";
        let mean = name == "mean";
        let extrema = matches!(name.as_str(), "min" | "max" | "amin" | "amax");
        let expected = if logical || mean { 4 } else if extrema { 6 } else { 5 };
        if let Some(op) = op
            && args.len() == expected
            && (is_array_type(m, parent_type(m, function))
                || starts(&fname(m, function), format!("{}[", mangle::func("std.numpy.reductions", &name))))
        {
            if mean {
                let ft = var_ty(m, function);
                let g = m.type_generics(ft);
                let Some(dt) = g.first().and_then(|g| g.type_value()) else { return None };
                if !m.type_is(dt, t.none) && !m.type_is(dt, ty.ir_base_type(t)) {
                    return None;
                }
            }
            let no_value = mangle::class("std.numpy.util", "_NoValue");
            let initial_value = if logical || mean { None } else { Some(args[if extrema { 4 } else { 3 }]) };
            let has_initial = initial_value.is_some_and(|iv| vt_name(m, iv) != no_value);
            let init_ok = !has_initial || {
                let iv = initial_value.unwrap();
                m.kind(iv).is_const() || m.as_var_value(iv).is_some()
            };
            if vt_is_none(m, args[2], t) && vt_name(m, *args.last().unwrap()) == no_value && init_ok {
                let operand = parse(m, a, args[0], leaves, t, false, None);
                let initial = if has_initial { parse(m, a, initial_value.unwrap(), leaves, t, false, None) } else { None };
                if let Some(o) = operand
                    && a[o].ty.is_array()
                    && a[o].ty.ndim > 0
                    && (!has_initial || initial.is_some_and(|i| !a[i].ty.is_array()))
                {
                    return Some(a.node(ty, v, op, Some(o), initial, None));
                }
                leaves.clear();
            }
        }
    }

    // Don't break up expressions that result in scalars or 0-dim arrays since those
    // should only be computed once
    if ty.ndim == 0 {
        let res = a.leaf(ty, v);
        leaves.push((res, v));
        return Some(res);
    }

    if m.as_call_instr(v).is_some() {
        let args = call_args(m, v).unwrap();
        let f = callee_func(m, v);
        let fnm = f.map(|f| fname(m, f)).unwrap_or_default();
        let un = f.map(|f| unmangled(m, f)).unwrap_or_default();
        let ptype = f.and_then(|f| parent_type(m, f));

        if let Some(_f) = f
            && args.len() == 3
        {
            let a0t = vtype(m, args[0]);
            let a1t = vtype(m, args[1]);
            if is_array_type(m, Some(a0t))
                && m.type_is(a1t, t.i64)
                && starts(&fnm, format!("{}[", mangle::func("std.numpy.routines", "roll")))
                && let Some(axis) = m.as_int_const(args[2]).map(|c| c.val)
                && axis >= -ty.ndim
                && axis < ty.ndim
            {
                let operand = parse(m, a, args[0], leaves, t, false, None);
                let shift = parse(m, a, args[1], leaves, t, false, None);
                if let (Some(o), Some(s)) = (operand, shift) {
                    return Some(a.node(ty, v, Op::Roll, Some(o), Some(s), None));
                }
            }
        }

        if f.is_some()
            && args.len() == 4
            && un == "clip"
            && (is_array_type(m, ptype) || starts(&fnm, format!("{}[", mangle::func("std.numpy.routines", "clip"))))
            && vt_is_none(m, args[3], t)
        {
            let no_min = vt_is_none(m, args[1], t);
            let no_max = vt_is_none(m, args[2], t);
            if no_min && no_max {
                return None;
            }
            let operand = parse(m, a, args[0], leaves, t, false, None);
            let bound = parse(m, a, args[if no_min { 2 } else { 1 }], leaves, t, false, None);
            let upper = if !no_min && !no_max { parse(m, a, args[2], leaves, t, false, None) } else { None };
            if operand.is_none() || bound.is_none() || (!no_min && !no_max && upper.is_none()) {
                return None;
            }
            let op = if no_min {
                Op::ClipMax
            } else if no_max {
                Op::ClipMin
            } else {
                Op::Clip
            };
            return Some(a.node(ty, v, op, operand, bound, upper));
        }

        if f.is_some() && args.len() == 3 && starts(&fnm, format!("{}[", mangle::func("std.numpy.routines", "where"))) {
            let c = parse(m, a, args[0], leaves, t, false, None)?;
            let tr = parse(m, a, args[1], leaves, t, false, None)?;
            let fa = parse(m, a, args[2], leaves, t, false, None)?;
            return Some(a.node(ty, v, Op::Where, Some(c), Some(tr), Some(fa)));
        }

        if f.is_some() && args.len() == 2 {
            let a0t = vtype(m, args[0]);
            if is_array_type(m, Some(a0t)) {
                let order = m.as_string_const(args[1]).map(|s| s.val.clone());
                if (un == "zeros_like" || un == "ones_like")
                    && starts(&fnm, format!("{}[", mangle::func("std.numpy.routines", &un)))
                    && order.as_deref().is_some_and(is_order)
                    && let Some(o) = parse(m, a, args[0], leaves, t, false, None)
                {
                    let op = if un == "zeros_like" { Op::ZerosLike } else { Op::OnesLike };
                    return Some(a.node(ty, v, op, Some(o), None, None));
                }
            }
        }

        if f.is_some() && is_array_type(m, ptype) {
            let last_bool = args.last().and_then(|l| m.as_bool_const(*l)).map(|b| b.val);
            if (un == "astype" && args.len() == 3 && last_bool.is_some()) || (un == "copy" && args.len() == 2) {
                let order = m.as_string_const(args[1]).map(|s| s.val.clone());
                // A same-dtype no-copy cast may return the input itself. Keep it opaque
                // rather than letting codegen treat a borrowed alias as a fresh temporary.
                let a0t = vtype(m, args[0]);
                let vty = vtype(m, v);
                let owns = un == "copy" || last_bool == Some(true) || !m.type_is(a0t, vty);
                if owns
                    && order.as_deref().is_some_and(is_order)
                    && let Some(o) = parse(m, a, args[0], leaves, t, false, None)
                {
                    return Some(a.node(ty, v, Op::Cast, Some(o), None, None));
                }
            }
        }

        // Check for matmul
        if f.is_some()
            && args.len() == 3
            && vt_is_none(m, args[2], t)
            && (starts(&fnm, format!("{}[", mangle::func("std.numpy.linalg_sym", "matmul")))
                || (starts(&fnm, format!("{}[]", mangle::func("std.numpy.linalg_sym", "dot"))) && ty.ndim == 2))
        {
            let l = parse(m, a, args[0], leaves, t, false, None)?;
            let r = parse(m, a, args[1], leaves, t, false, None)?;
            return Some(a.node(ty, v, Op::Matmul, Some(l), Some(r), None));
        }

        // Check for builtin abs()
        if f.is_some() && args.len() == 1 && starts(&fnm, format!("{}[", mangle::func("std.internal.builtin", "abs"))) {
            let l = parse(m, a, args[0], leaves, t, false, None)?;
            return Some(a.node(ty, v, Op::Abs, Some(l), None, None));
        }

        // Check for transpose
        if f.is_some() && is_array_type(m, ptype) && args.len() == 1 && un == "T" {
            let l = parse(m, a, args[0], leaves, t, false, None)?;
            return Some(a.node(ty, v, Op::Transpose, Some(l), None, None));
        }

        // Check for ufunc (e.g. "np.exp()") call
        if let Some(func) = f
            && un == magic::CALL
            && is_ufunc_type(m, ptype)
        {
            let ug = m.type_generics(ptype.unwrap());
            let ufunc = match ug.first() {
                Some(g) if g.is_static_str() => g.static_string_value(),
                _ => panic!("unrecognized ufunc class generics"),
            };
            let ft = var_ty(m, func);
            let cg = m.type_generics(ft);
            let dtype = cg.first().and_then(|g| g.type_value()).expect("unrecognized ufunc call generics");
            if m.type_is(dtype, t.none)
                && let Some(&(name, op, nargs)) = UFUNCS.iter().find(|u| u.0 == ufunc)
            {
                // Argument order: ufunc self, operand 1, (if binary) operand 2, 'out', 'where'
                assert!(args.len() == nargs + 3, "unexpected call of {name}");
                let where_ = args[args.len() - 1];
                let out = args[args.len() - 2];
                let where_ok = m.as_bool_const(where_).is_some_and(|b| b.val);
                let out_none = vt_is_none(m, out, t);
                if where_ok && (out_none || destination.is_some()) {
                    let l = parse(m, a, args[1], leaves, t, false, None)?;
                    if nargs == 1 {
                        if let Some(d) = destination.as_deref_mut()
                            && !out_none
                        {
                            *d = Some(out);
                        }
                        return Some(a.node(ty, v, op, Some(l), None, None));
                    }
                    let r = parse(m, a, args[2], leaves, t, false, None)?;
                    if let Some(d) = destination.as_deref_mut()
                        && !out_none
                    {
                        *d = Some(out);
                    }
                    return Some(a.node(ty, v, op, Some(l), Some(r), None));
                }
            }
        }

        // Check for magic method call
        if f.is_some() && is_array_type(m, ptype) {
            for mg in MAGICS {
                if un == mg.name && args.len() == mg.args {
                    let l = parse(m, a, args[0], leaves, t, false, None)?;
                    if mg.args == 1 {
                        return Some(a.node(ty, v, mg.op, Some(l), None, None));
                    }
                    let r = parse(m, a, args[1], leaves, t, false, None)?;
                    return Some(if mg.right {
                        a.node(ty, v, mg.op, Some(r), Some(l), None)
                    } else {
                        a.node(ty, v, mg.op, Some(l), Some(r), None)
                    });
                }
            }
        }
    }

    // Check for right-hand-side magic method call
    // Right-hand-side magics (e.g. __radd__) are compiled into FlowInstr:
    //   <lhs_expr> + <rhs_expr>
    // becomes:
    //   { v1 = <lhs expr> ; v2 = <rhs expr> ; return rhs_class.__radd__(v2, v1) }
    // So we need to check for this to detect r-magics.
    if let Some(fi) = m.as_flow_instr(v) {
        let (flow, value) = (fi.flow, fi.val);
        let assignments = m.as_series_flow(flow).map(|s| s.series.clone());
        let f = callee_func(m, value);
        if let (Some(assignments), Some(f)) = (assignments, f) {
            let vargs = call_args(m, value).unwrap();
            if vargs.len() == 2 {
                let (arg1, arg2) = (vargs[0], vargs[1]);
                let arg1_var = var_of(m, arg1);
                let arg2_var = var_of(m, arg2);
                let un = unmangled(m, f);
                let lhs_of = |m: &Module, x: ValueId| m.as_assign_instr(x).map(|a| (m.actual_var(a.lhs), a.rhs));
                if let Some(mg) = MAGICS.iter().find(|mg| un == mg.name && mg.args == 2 && mg.right) {
                    let op = mg.op;
                    match assignments.len() {
                        // Case 1: Degenerate flow instruction
                        0 => return parse(m, a, value, leaves, t, false, None),
                        // Case 2: One var -- check if it's either of the r-magic operands
                        1 => {
                            if let Some((lv, rhs)) = lhs_of(m, assignments[0]) {
                                if Some(lv) == arg1_var {
                                    let r = parse(m, a, rhs, leaves, t, false, None)?;
                                    let l = parse(m, a, arg2, leaves, t, false, None)?;
                                    return Some(a.node(ty, v, op, Some(l), Some(r), None));
                                } else if Some(lv) == arg2_var {
                                    let l = parse(m, a, rhs, leaves, t, false, None)?;
                                    let r = parse(m, a, arg1, leaves, t, false, None)?;
                                    return Some(a.node(ty, v, op, Some(l), Some(r), None));
                                }
                            }
                        }
                        // Case 2: Two vars -- check both permutations
                        2 => {
                            if let (Some((l1, r1)), Some((l2, r2))) = (lhs_of(m, assignments[0]), lhs_of(m, assignments[1])) {
                                if Some(l1) == arg1_var && Some(l2) == arg2_var {
                                    let r = parse(m, a, r1, leaves, t, false, None)?;
                                    let l = parse(m, a, r2, leaves, t, false, None)?;
                                    return Some(a.node(ty, v, op, Some(l), Some(r), None));
                                } else if Some(l2) == arg1_var && Some(l1) == arg2_var {
                                    let l = parse(m, a, r1, leaves, t, false, None)?;
                                    let r = parse(m, a, r2, leaves, t, false, None)?;
                                    return Some(a.node(ty, v, op, Some(l), Some(r), None));
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    let res = a.leaf(ty, v);
    leaves.push((res, v));
    Some(res)
}

// ---------------------------------------------------------------------------
// matmul-add, optimizeHelper
// ---------------------------------------------------------------------------

/// Match only the static operator/dtype/rank contract. The runtime helper checks
/// shapes and BLAS-compatible strides, falling back to the original operations.
pub fn is_matmul_add(m: &Module, a: &ExprArena, e: ExprId) -> bool {
    let x = &a[e];
    if !numpy_options().fuse_matmul
        || !matches!(x.op, Op::Add | Op::Sub)
        || x.lhs.is_none()
        || x.rhs.is_none()
        || !matches!(x.ty.dtype, TypeCode::ArrF32 | TypeCode::ArrF64)
    {
        return false;
    }
    let (l, r) = (x.lhs.unwrap(), x.rhs.unwrap());
    let product_left = a[l].op == Op::Matmul;
    let (product, bias) = if product_left { (l, r) } else { (r, l) };
    let is_vv = |e: ExprId| a[e].val.is_some_and(|v| m.as_var_value(v).is_some());
    if a[product].op != Op::Matmul || !a[bias].is_leaf() || !is_vv(bias) || !a[bias].ty.is_array() {
        return false;
    }
    let operand = |input: ExprId| {
        let leaf = if a[input].op == Op::Transpose { a[input].lhs } else { Some(input) };
        leaf.is_some_and(|lf| a[lf].is_leaf() && is_vv(lf)) && a[input].ty.dtype == x.ty.dtype
    };
    let (pl, pr) = (a[product].lhs.unwrap(), a[product].rhs.unwrap());
    a[product].ty.dtype == x.ty.dtype
        && a[bias].ty.dtype == x.ty.dtype
        && x.ty.ndim == a[product].ty.ndim
        && a[bias].ty.ndim <= a[product].ty.ndim
        && a[pl].ty.ndim == 2
        && (a[pr].ty.ndim == 1 || a[pr].ty.ndim == 2)
        && operand(pl)
        && operand(pr)
}

fn codegen_matmul_add(m: &mut Module, a: &ExprArena, e: ExprId, c: &CodegenContext, destination: Option<VarId>) -> VarId {
    let operand = |m: &mut Module, input: ExprId| -> ValueId {
        if a[input].is_leaf() {
            return m.nr_var_value(c.vars[&input]);
        }
        let value = m.nr_var_value(c.vars[&a[input].lhs.unwrap()]);
        let vt = vtype(m, value);
        let tr = fusion_func(m, "_transpose", &[vt], &[]);
        call(m, tr, vec![value])
    };
    let x = a[e].clone();
    let product_left = a[x.lhs.unwrap()].op == Op::Matmul;
    let (product, bias) = if product_left { (x.lhs.unwrap(), x.rhs.unwrap()) } else { (x.rhs.unwrap(), x.lhs.unwrap()) };
    let mut args = vec![operand(m, a[product].lhs.unwrap()), operand(m, a[product].rhs.unwrap()), operand(m, bias)];
    args.push(match destination {
        Some(d) => m.nr_var_value(d),
        None => {
            let nt = m.get_none_type();
            m.construct(nt, vec![]).expect("NoneType construction failed")
        }
    });
    let tys: Vec<TypeId> = args.iter().map(|v| vtype(m, *v)).collect();
    // Static modes: product+bias, bias+product, product-bias, bias-product.
    // Specialization keeps the signed BLAS coefficients constant at each call site.
    let mode = (if x.op == Op::Sub { 2 } else { 0 }) + (if product_left { 0 } else { 1 });
    let helper = fusion_func(m, "_matmul_add", &tys, &[typhon_ir::Generic::Static(mode)]);
    let cl = call(m, helper, args);
    let result = make_var(m, cl, c.series, Some(c.func), false);
    for y in a.preorder(e) {
        if a[y].is_leaf() && a[y].owned_last_use {
            let value = m.nr_var_value(c.vars[&y]);
            let vt = vtype(m, value);
            let fr = fusion_func(m, "_free", &[vt], &[]);
            let cl = call(m, fr, vec![value]);
            m.series_push(c.series, cl);
        }
    }
    result
}

fn replace_with_leaf(m: &mut Module, a: &mut ExprArena, e: ExprId, var: VarId, owned: bool) {
    let ty = a[e].ty;
    let vv = m.nr_var_value(var);
    let n = a.leaf(ty, vv);
    a[n].owned_last_use = owned;
    a.replace(e, n);
}

fn free_array(m: &mut Module, arr: VarId) -> ValueId {
    let t = var_ty(m, arr);
    let f = fusion_func(m, "_free", &[t], &[]);
    let vv = m.nr_var_value(arr);
    call(m, f, vec![vv])
}

fn optimize_helper(m: &mut Module, a: &mut ExprArena, ufunc: FuncId, e: ExprId, c: &mut CodegenContext) -> VarId {
    let series_v = c.series;
    // Handle non-elementwise operations before building scalar loops. Match a
    // compound matmul first, before materializing its product loses that pattern.
    apply_in(a, e, &mut |a, x| {
        if !a[x].ty.is_array() {
            return;
        }
        if is_matmul_add(m, a, x) {
            xlog!("-> BLAS matmul-add fuse:\n{}", a.str(m, x));
            let result = codegen_matmul_add(m, a, x, c, None);
            replace_with_leaf(m, a, x, result, true);
            c.vars.insert(x, result);
            return;
        }
        if a[x].op == Op::Transpose {
            let l = a[x].lhs.unwrap();
            let lv = optimize_helper(m, a, ufunc, l, c);
            let lt = var_ty(m, lv);
            let tf = fusion_func(m, "_transpose", &[lt], &[]);
            let vv = m.nr_var_value(lv);
            let cl = call(m, tf, vec![vv]);
            let var = make_var(m, cl, c.series, Some(c.func), false);
            c.vars.insert(x, var);
            let owned = a[l].owned_last_use;
            replace_with_leaf(m, a, x, var, owned);
        }
        if a[x].op == Op::Matmul {
            let (l, r) = (a[x].lhs.unwrap(), a[x].rhs.unwrap());
            let lv = optimize_helper(m, a, ufunc, l, c);
            let rv = optimize_helper(m, a, ufunc, r, c);
            let (lt, rt) = (var_ty(m, lv), var_ty(m, rv));
            let mf = fusion_func(m, "_matmul", &[lt, rt], &[]);
            let (p, q) = (m.nr_var_value(lv), m.nr_var_value(rv));
            let cl = call(m, mf, vec![p, q]);
            let var = make_var(m, cl, c.series, Some(c.func), false);
            let lfree = a[l].ty.is_array() && (a[l].owned_last_use || !a[l].is_leaf());
            let rfree = a[r].ty.is_array() && (a[r].owned_last_use || !a[r].is_leaf());
            if lfree {
                let f = free_array(m, lv);
                m.series_push(series_v, f);
            }
            if rfree {
                let f = free_array(m, rv);
                m.series_push(series_v, f);
            }
            c.vars.insert(x, var);
            replace_with_leaf(m, a, x, var, true);
        }
    });

    // Collapse profitable subtrees to result leaves. When broadcasting is unknown,
    // emit a runtime choice between fusion and sequential array operations.
    let opts = numpy_options();
    loop {
        let mut changed = false;
        apply_in(a, e, &mut |a, x| {
            if a.depth(x) <= 2 {
                return;
            }
            let cost = a.cost(m, x);
            let bc = a.broadcast_info(m, x);
            let mut result = None;
            if cost <= opts.always_fuse_cost_threshold || (cost <= opts.never_fuse_cost_threshold && bc == BroadcastInfo::No) {
                // Don't care about broadcasting; just fuse.
                xlog!("-> static fuse:\n{}", a.str(m, x));
                result = Some(a.codegen_fused_eval(m, c, x, None));
            } else if cost <= opts.never_fuse_cost_threshold && bc != BroadcastInfo::Yes {
                // Check at runtime if we're broadcasting and fuse conditionally.
                xlog!("-> conditional fuse:\n{}", a.str(m, x));
                let broadcasts = a.codegen_broadcasts(m, c, x);
                let seqt = m.nr_series_flow();
                let fuse = m.nr_series_flow();
                let branch = m.nr_if_flow(broadcasts, seqt, Some(fuse));
                // PORT-NOTE: both evaluations work on the same arena; sequential eval
                // does not mutate the tree, so fused eval sees the same expression.
                c.series = seqt;
                let sr = a.codegen_sequential_eval(m, c, x);
                c.series = fuse;
                let fr = a.codegen_fused_eval(m, c, x, None);
                let (st, ft) = (var_ty(m, sr), var_ty(m, fr));
                assert!(m.type_is(st, ft), "types are not the same: {} {}", m.type_name(st), m.type_name(ft));
                let r = m.nr_var(st, false, "");
                m.func_push_symbol(ufunc, r);
                let sv = m.nr_var_value(sr);
                let s_as = m.nr_assign(r, sv);
                m.series_push(seqt, s_as);
                let fv = m.nr_var_value(fr);
                let f_as = m.nr_assign(r, fv);
                m.series_push(fuse, f_as);
                c.series = series_v;
                m.series_push(series_v, branch);
                result = Some(r);
            }
            if let Some(r) = result {
                replace_with_leaf(m, a, x, r, true);
                c.vars.insert(x, r);
                changed = true;
            }
        });
        if !changed {
            break;
        }
    }
    xlog!("-> sequential eval:\n{}", a.str(m, e));
    a.codegen_sequential_eval(m, c, e)
}

fn is_basic_index_type(m: &mut Module, ty: TypeId, types: &NumPyPrimitiveTypes) -> bool {
    if m.type_is(ty, types.i64) || is_none_type(m, ty, types) {
        return true;
    }
    let slice = format!("{}[", mangle::class("std.internal.types.slice", "Slice"));
    let name = m.type_name(ty).to_string();
    let is_slice = name.starts_with(&slice);
    if !is_slice && !name.starts_with("Tuple[") {
        return false;
    }
    let g = m.type_generics(ty);
    !g.is_empty()
        && g.iter().all(|g| match g.type_value() {
            Some(t) => {
                if is_slice {
                    m.type_is(t, types.i64)
                } else {
                    is_basic_index_type(m, t, types)
                }
            }
            None => false,
        })
}

fn is_safe_fusion_leaf(m: &mut Module, value: ValueId, types: &NumPyPrimitiveTypes, se: &SideEffectResult) -> bool {
    if has_lifetime_tag(m, value, "numpy.validation") {
        return true;
    }
    if !se.has_side_effect(m, value) {
        return true;
    }
    let Some(callee) = callee_func(m, value) else { return false };
    let args = call_args(m, value).unwrap();
    let name = unmangled(m, callee);
    let pt = parent_type(m, callee);
    let array_method = is_array_type(m, pt);
    let scalar_type = match pt {
        Some(p) => NumPyType::get(m, p, types),
        None => NumPyType::default(),
    };
    const SCALAR_OPS: [&str; 16] = [
        "__add__", "__radd__", "__sub__", "__rsub__", "__mul__", "__rmul__", "__truediv__", "__rtruediv__", "__floordiv__",
        "__rfloordiv__", "__mod__", "__rmod__", "__pow__", "__rpow__", "__neg__", "__pos__",
    ];
    let complex_ctor = name == "__new__" && matches!(scalar_type.dtype, TypeCode::C64 | TypeCode::C128);
    let mut safe = scalar_type.is_some()
        && !scalar_type.is_array()
        && (SCALAR_OPS.contains(&name.as_str()) || complex_ctor)
        && args.iter().all(|x| {
            let t = npt(m, *x, types);
            t.is_some() && !t.is_array()
        });
    safe |= args.len() == 1 && fname(m, callee).starts_with("Int.__suffix_j__:") && {
        let t = vtype(m, args[0]);
        m.type_is(t, types.i64)
    };
    safe |= array_method && name == magic::GETITEM && args.len() == 2 && npt(m, args[0], types).is_array() && {
        let bt = vtype(m, args[1]);
        is_basic_index_type(m, bt, types) || is_copy_index_type(m, bt, types)
    };
    if name == "clip"
        && args.len() == 4
        && (array_method || starts(&fname(m, callee), format!("{}[", mangle::func("std.numpy.routines", "clip"))))
        && vt_is_none(m, args[3], types)
    {
        safe = args.iter().all(|x| npt(m, *x, types).is_some() || vt_is_none(m, *x, types));
    }
    safe && args.iter().all(|x| is_safe_fusion_leaf(m, *x, types, se))
}

/// Arguments omitted from the arithmetic tree must still preserve observable
/// evaluation, including ufunc receivers and clip's None bounds/out argument.
fn has_ufunc_argument_effects(m: &mut Module, a: &ExprArena, e: ExprId, se: &SideEffectResult) -> bool {
    let mut effects = false;
    for x in a.preorder(e) {
        if a[x].is_leaf() {
            continue;
        }
        let Some(val) = a[x].val else { continue };
        let Some(args) = call_args(m, val) else { continue };
        if let Some(callee) = callee_func(m, val)
            && is_ufunc_type(m, parent_type(m, callee))
        {
            effects |= se.has_side_effect(m, args[0]) || se.has_side_effect(m, args[args.len() - 2]);
        }
        if a[x].is_clip() {
            effects |= se.has_side_effect(m, *args.last().unwrap());
            if a[x].op == Op::ClipMin {
                effects |= se.has_side_effect(m, args[2]);
            }
            if a[x].op == Op::ClipMax {
                effects |= se.has_side_effect(m, args[1]);
            }
        }
    }
    effects
}

/// `hasOwnedResult(expr)`: allocation provenance only, not last-use permission. Fail
/// closed for views, identity operations, and new opcodes until their allocation
/// contract is known.
pub fn has_owned_result(a: &ExprArena, e: ExprId) -> bool {
    has_owned_result_with(None, a, e)
}

fn has_owned_result_with(m: Option<&Module>, a: &ExprArena, e: ExprId) -> bool {
    let x = &a[e];
    if !x.ty.is_array() || x.is_leaf() {
        return false;
    }
    match x.op {
        Op::None | Op::Pos | Op::Transpose | Op::Conj => false,
        Op::Cast => {
            // PORT-NOTE: needs the call's args; without a module only the 2-arg form
            // (`copy`) cannot be distinguished, so the caller-module variant is used
            // wherever available (see `has_owned_result_m`).
            match (m, x.val) {
                (Some(m), Some(v)) => match m.as_call_instr(v) {
                    Some(c) => c.args.len() == 2 || c.args.last().and_then(|l| m.as_bool_const(*l)).is_some_and(|b| b.val),
                    None => false,
                },
                _ => false,
            }
        }
        _ => true,
    }
}

/// `hasOwnedResult(expr)` with access to the module (exact C++ semantics for casts).
pub fn has_owned_result_m(m: &Module, a: &ExprArena, e: ExprId) -> bool {
    has_owned_result_with(Some(m), a, e)
}

// ---------------------------------------------------------------------------
// NumPyOptimizationUnit::codegenValidation / optimize
// ---------------------------------------------------------------------------

fn last_of_series(m: &Module, s: ValueId) -> Option<ValueId> {
    m.as_series_flow(s).and_then(|s| s.series.last().copied())
}

/// `NumPyOptimizationUnit::codegenValidation(context, atDefinitions)`.
pub fn codegen_validation(m: &mut Module, a: &mut ExprArena, unit: &mut NumPyOptimizationUnit, c: &mut CodegenContext, at_definitions: bool) {
    let series_v = c.series;
    let mut clone = CloneVisitor::new(true);
    let owns_result = has_owned_result_m(m, a, unit.expr) && !a[unit.expr].is_reduction();
    let order = unit.validation_order.clone();
    for (step, element) in order {
        c.series = series_v;
        if at_definitions
            && let Some(s) = step
            && Some(s) != unit.assign
        {
            let flow = match unit.validation_flows.get(&s) {
                Some(f) => *f,
                None => {
                    let f = m.nr_series_flow();
                    unit.validation_flows.insert(s, f);
                    f
                }
            };
            c.series = flow;
        }
        if a[element].is_leaf() {
            let cl = clone.clone_value(m, a[element].v());
            let var = make_var(m, cl, c.series, Some(unit.func), false);
            c.vars.entry(element).or_insert(var);
            if owns_result && !a[element].owned_last_use {
                let last = last_of_series(m, c.series).unwrap();
                set_lifetime_tag(m, last, "numpy.lifetime.input");
            }
        } else if a[element].ty.is_array() && !a[element].is_reduction() {
            a.codegen_layout(m, c, element);
        }
    }
    c.series = series_v;
}

/// `NumPyOptimizationUnit::optimize(T, sideEffects)`.
pub fn optimize_unit(
    m: &mut Module,
    a: &mut ExprArena,
    unit: &mut NumPyOptimizationUnit,
    types: &NumPyPrimitiveTypes,
    se: &SideEffectResult,
) -> bool {
    let root = unit.expr;
    let reduction = a[root].is_reduction();
    let mut owned_leaf = false;
    let root_leaf = a[root].is_leaf();
    for x in a.preorder(root) {
        if !root_leaf && a[x].is_leaf() && is_owned_index(m, a[x].v(), types) {
            a[x].owned_last_use = true;
        }
        owned_leaf |= a[x].is_leaf() && a[x].owned_last_use;
    }
    if (!a[root].ty.is_array() && !reduction)
        || (a.depth(root) <= 2 && (reduction || !owned_leaf))
        || has_ufunc_argument_effects(m, a, root, se)
    {
        return false;
    }

    let mut has_roll = false;
    let mut supports_roll = true;
    for x in a.preorder(root) {
        if a[x].op == Op::Roll {
            has_roll = true;
            if !a[a[x].lhs.unwrap()].is_leaf() || !a[a[x].rhs.unwrap()].is_leaf() {
                supports_roll = false;
            }
        }
        if a[x].op == Op::Transpose || a[x].op == Op::Matmul || (a[x].is_reduction() && x != root) {
            supports_roll = false;
        }
    }
    if has_roll && !supports_roll {
        return false;
    }
    for &(_, leaf) in &unit.leaves.clone() {
        if !is_safe_fusion_leaf(m, leaf, types, se) {
            return false;
        }
    }
    if reduction {
        let Some(args) = call_args(m, a[root].v()) else { return false };
        if args.iter().skip(1).any(|x| se.has_side_effect(m, *x)) {
            return false;
        }
        let l = a[root].lhs.unwrap();
        let mut supported = a.cost(m, l) <= numpy_options().never_fuse_cost_threshold;
        for x in a.preorder(l) {
            if (a[x].op == Op::Transpose && !a[a[x].lhs.unwrap()].is_leaf()) || a[x].op == Op::Matmul || a[x].is_reduction() {
                supported = false;
            }
        }
        if !supported {
            return false;
        }
    }

    xlog!("Optimizing expression at {}\n{}", m.src_info(unit.value.into()), a.str(m, root));

    let series_v = m.nr_series_flow();
    let mut c = CodegenContext::new(series_v, unit.func, *types);
    let owns_result = has_owned_result_m(m, a, root) && !reduction;
    codegen_validation(m, a, unit, &mut c, true);

    if reduction && has_roll {
        let element = a[root].lhs.unwrap();
        let input = a.codegen_fused_eval(m, &mut c, element, None);
        replace_with_leaf(m, a, element, input, true);
        c.vars.insert(element, input);
    }
    if reduction {
        let l = a[root].lhs.unwrap();
        let func = unit.func;
        apply_in(a, l, &mut |a, x| {
            if a[x].op == Op::Transpose {
                optimize_helper(m, a, func, x, &mut c);
            }
        });
    }
    let result = if reduction { a.codegen_fused_eval(m, &mut c, root, None) } else { optimize_helper(m, a, unit.func, root, &mut c) };
    let rv = m.nr_var_value(result);
    let replacement = m.nr_flow_instr(c.series, rv);
    set_lifetime_tag(m, replacement, "numpy.lifetime.expression");
    if owns_result {
        set_lifetime_tag(m, replacement, "numpy.lifetime.owned");
    }
    m.replace_all_value(unit.value, replacement);
    true
}

// ---------------------------------------------------------------------------
// DestinationExpression
// ---------------------------------------------------------------------------

/// Destination writes bypass the forwarding DAG: prove the output is safe first,
/// then lower into that buffer without taking ownership of it or freeing it.
#[derive(Clone, Debug)]
pub struct DestinationExpression {
    pub unit: NumPyOptimizationUnit,
    pub destination: ValueId,
    pub assignment: bool,
}

const ALLOCATORS: [&str; 8] = ["empty", "zeros", "ones", "full", "empty_like", "zeros_like", "ones_like", "full_like"];

impl DestinationExpression {
    #[allow(clippy::too_many_arguments)]
    fn eligible(
        &self,
        m: &mut Module,
        a: &ExprArena,
        types: &NumPyPrimitiveTypes,
        rd: &RDInspector,
        cfg: &CFGraph,
        se: &SideEffectResult,
    ) -> bool {
        let root = self.unit.expr;
        let Some(out_var) = var_of(m, self.destination) else { return false };
        if m.is_global(out_var)
            || !a[root].ty.is_array()
            || (!self.assignment && a.depth(root) <= 2)
            || has_ufunc_argument_effects(m, a, root, se)
        {
            return false;
        }
        let out_t = npt(m, self.destination, types);
        if !out_t.is_array() || out_t.dtype != a[root].ty.dtype {
            return false;
        }
        if self.assignment {
            let args = call_args(m, self.unit.value).unwrap();
            if out_t.ndim == 0 || se.has_side_effect(m, args[1]) {
                return false;
            }
        }
        let base = out_t.ir_base_type(types);
        let matmul_add = is_matmul_add(m, a, root);
        if matmul_add && out_t.ndim != a[root].ty.ndim {
            return false;
        }
        let mut supported = true;
        // Exact pointwise self-reads are safe, but BLAS must not overwrite either
        // multiplicative input. This applies whichever side contains the product.
        if matmul_add {
            let product = if a[a[root].lhs.unwrap()].op == Op::Matmul { a[root].lhs.unwrap() } else { a[root].rhs.unwrap() };
            for x in a.preorder(product) {
                if a[x].val.and_then(|v| var_of(m, v)) == Some(out_var) {
                    supported = false;
                }
            }
        }
        let mut allowed: FxHashSet<Id> = FxHashSet::default();
        allowed.insert(m.value_uid(self.destination));
        for x in a.preorder(root) {
            if !m.type_is(a[x].ty.ir_base_type(types), base) || (!matmul_add && a[x].ty.ndim > out_t.ndim) {
                supported = false;
            }
            let val = a[x].v();
            if a[x].is_leaf() {
                if (m.as_var_value(val).is_none() && !m.kind(val).is_const()) || se.has_side_effect(m, val) {
                    supported = false;
                }
                allowed.insert(m.value_uid(val));
                continue;
            }
            match a[x].op {
                Op::Add | Op::Sub | Op::Mul | Op::Neg | Op::Pos => {}
                Op::Matmul | Op::Transpose => {
                    if !matmul_add {
                        supported = false;
                    }
                }
                _ => supported = false,
            }
        }
        if !supported {
            return false;
        }

        // Prove disjointness from a single fresh allocation in this block, not from
        // general alias analysis. Earlier or cross-block uses may expose the buffer.
        let defs = rd.get_reaching_definitions(m, out_var, self.destination);
        if defs.len() != 1 {
            return false;
        }
        let Some(definition) = defs[0].assignment_value().filter(|d| m.as_assign_instr(*d).is_some()) else { return false };
        let rhs = m.as_assign_instr(definition).unwrap().rhs;
        let Some(allocator) = callee_func(m, rhs) else { return false };
        let name = unmangled(m, allocator);
        if !ALLOCATORS.contains(&name.as_str()) || !starts(&fname(m, allocator), format!("{}[", mangle::func("std.numpy.routines", &name))) {
            return false;
        }
        let Some(block) = cfg.get_block(m, self.unit.value) else { return false };
        if cfg.get_block(m, definition) != Some(block) {
            return false;
        }
        let unit_uid = m.value_uid(self.unit.value);
        let mut preceding: FxHashSet<Id> = FxHashSet::default();
        for v in cfg.block(block).values() {
            let uid = cfg.uid(m, *v);
            if uid == unit_uid {
                break;
            }
            preceding.insert(uid);
        }
        let def_uid = m.value_uid(definition);
        if !preceding.contains(&def_uid) {
            return false;
        }

        let mut safe = true;
        Walk::new().run_func(m, self.unit.func, &mut |_, m, value| {
            let uid = m.value_uid(value);
            if !safe || uid == def_uid || allowed.contains(&uid) {
                return;
            }
            if !m.used_variables(value.into()).iter().any(|u| m.actual_var(*u) == out_var) {
                return;
            }
            for r in rd.get_reaching_definitions(m, out_var, value) {
                if r.assignment_uid() == def_uid && (cfg.get_block(m, value) != Some(block) || preceding.contains(&uid)) {
                    safe = false;
                }
            }
        });
        safe
    }

    fn optimize(&mut self, m: &mut Module, a: &mut ExprArena, types: &NumPyPrimitiveTypes) {
        let series_v = m.nr_series_flow();
        let mut c = CodegenContext::new(series_v, self.unit.func, *types);
        let mut clone = CloneVisitor::new(true);
        codegen_validation(m, a, &mut self.unit, &mut c, false);
        let cd = clone.clone_value(m, self.destination);
        let output = make_var(m, cd, series_v, Some(self.unit.func), false);
        xlog!("-> destination fuse at {}:\n{}", m.src_info(self.unit.value.into()), a.str(m, self.unit.expr));
        let root = self.unit.expr;
        let result = if is_matmul_add(m, a, root) {
            codegen_matmul_add(m, a, root, &c, Some(output))
        } else {
            a.codegen_fused_eval(m, &mut c, root, Some(output))
        };
        let returned = if self.assignment {
            let nt = m.get_none_type();
            m.construct(nt, vec![]).expect("NoneType construction failed")
        } else {
            m.nr_var_value(result)
        };
        let fi = m.nr_flow_instr(series_v, returned);
        m.replace_all_value(self.unit.value, fi);
    }
}

// ---------------------------------------------------------------------------
// expression extraction
// ---------------------------------------------------------------------------

/// The inliner supplies a resolver for already-proven helper summaries. Keep the
/// original call/argument identities for CFG and reaching-definition queries; only
/// the arithmetic view and the eventual clone use the expanded expression.
pub type HelperResolver<'r> = dyn FnMut(&mut Module, ValueId, &mut Vec<ValueId>) -> Option<ValueId> + 'r;

type ValidationOrder = Vec<(ValueId, Option<ValueId>)>;

/// Port of `NumPyFunctionExpressions::Expression`.
#[derive(Clone, Debug)]
pub struct Expression {
    /// Index of the root unit in `units`.
    pub unit: usize,
    pub assignments: Vec<ValueId>,
    pub substitutions: Vec<(ValueId, ValueId)>,
}

/// Port of `NumPyFunctionExpressions` (and its base `ExtractArrayExpressions`).
#[derive(Clone, Debug)]
pub struct NumPyFunctionExpressions {
    pub func: FuncId,
    pub types: NumPyPrimitiveTypes,
    pub rd: Rc<RDInspector>,
    pub cfg: Rc<CFGraph>,
    pub side_effects: SideEffectResult,
    pub arena: ExprArena,
    pub exprs: Vec<NumPyOptimizationUnit>,
    pub destinations: Vec<DestinationExpression>,
    pub extracted: FxHashSet<Id>,
    pub helper_substitutions: Vec<(ValueId, ValueId)>,
    pub helper_order: FxHashMap<ValueId, Vec<ValueId>>,
    pub expressions: Vec<Expression>,
}

impl NumPyFunctionExpressions {
    fn extract(&mut self, m: &mut Module, v: ValueId, assign: Option<ValueId>, resolver: &mut Option<&mut HelperResolver<'_>>) {
        if self.extracted.contains(&m.value_uid(v)) {
            return;
        }
        let types = self.types;
        let mut leaves = vec![];
        let mut destination: Option<ValueId> = None;
        let mut assignment = false;
        if let Some(callee) = callee_func(m, v) {
            let args = call_args(m, v).unwrap();
            assignment = is_array_type(m, parent_type(m, callee)) && unmangled(m, callee) == magic::SETITEM && args.len() == 3;
            if assignment {
                let slice = args[1];
                let slice_name = mangle::class("std.internal.types.slice", "Slice");
                let sf = callee_func(m, slice);
                let cargs = call_args(m, slice).unwrap_or_default();
                assignment = vt_name(m, slice).starts_with(&format!("{slice_name}["))
                    && sf.is_some_and(|f| fname(m, f).starts_with(&format!("{slice_name}.__new__:")))
                    && cargs.len() == 3;
                if assignment {
                    let it = m.get_int_type();
                    let opt_int = m.get_optional_type(it);
                    for b in cargs {
                        let ef = callee_func(m, b);
                        let bt = vtype(m, b);
                        let nargs = call_args(m, b).map(|x| x.len()).unwrap_or(1);
                        if ef.is_none()
                            || !m.type_is(bt, opt_int)
                            || !fname(m, ef.unwrap()).starts_with("Optional[Int[64]]:Optional.__new__:")
                            || nargs != 0
                        {
                            assignment = false;
                        }
                    }
                }
                if assignment {
                    destination = Some(args[0]);
                }
            }
        }
        let mut expr = if assignment {
            let last = *call_args(m, v).unwrap().last().unwrap();
            parse(m, &mut self.arena, last, &mut leaves, &types, false, None)
        } else {
            parse(m, &mut self.arena, v, &mut leaves, &types, true, Some(&mut destination))
        };
        if let (Some(d), Some(e)) = (destination, expr)
            && !self.arena[e].is_leaf()
        {
            let cand = DestinationExpression {
                unit: NumPyOptimizationUnit::new(v, self.func, e, leaves.clone(), None),
                destination: d,
                assignment,
            };
            if cand.eligible(m, &self.arena, &types, &self.rd, &self.cfg, &self.side_effects) {
                for x in self.arena.preorder(e) {
                    if let Some(val) = self.arena[x].val {
                        self.extracted.insert(m.value_uid(val));
                    }
                }
                self.extracted.insert(m.value_uid(v));
                self.destinations.push(cand);
                return;
            }
            leaves.clear();
            expr = parse(m, &mut self.arena, v, &mut leaves, &types, true, None);
        }
        let Some(e) = expr else { return };
        if let Some(res) = resolver.as_deref_mut() {
            let mut subs = vec![];
            let mut orders = vec![];
            let mut ext = vec![];
            apply_in(&mut self.arena, e, &mut |a, x| {
                if !a[x].is_leaf() {
                    return;
                }
                let Some(cv) = a[x].val.filter(|c| m.as_call_instr(*c).is_some()) else { return };
                let mut order = vec![];
                let Some(replacement) = res(m, cv, &mut order) else { return };
                let mut nested_leaves = vec![];
                let Some(nested) = parse(m, a, replacement, &mut nested_leaves, &types, true, None) else { return };
                if a[nested].is_leaf() {
                    return;
                }
                a.replace(x, nested);
                subs.push((cv, replacement));
                orders.push((cv, order));
                ext.push(m.value_uid(cv));
            });
            self.helper_substitutions.extend(subs);
            for (k, o) in orders {
                self.helper_order.entry(k).or_insert(o);
            }
            self.extracted.extend(ext);
            leaves.clear();
            for x in self.arena.preorder(e) {
                if self.arena[x].is_leaf() {
                    leaves.push((x, self.arena[x].v()));
                }
            }
        }
        let mut num_array_nodes = 0;
        for x in self.arena.preorder(e) {
            if self.arena[x].ty.is_array() {
                num_array_nodes += 1;
            }
            if let Some(val) = self.arena[x].val {
                self.extracted.insert(m.value_uid(val));
            }
        }
        if num_array_nodes > 0 && self.arena.depth(e) > 1 {
            self.exprs.push(NumPyOptimizationUnit::new(v, self.func, e, leaves, assign));
        }
    }

    fn pre_hook(&mut self, m: &mut Module, v: ValueId, resolver: &mut Option<&mut HelperResolver<'_>>) {
        if let Some(asg) = m.as_assign_instr(v) {
            let (lhs, rhs) = (asg.lhs, asg.rhs);
            let assign = if m.is_global(lhs) { None } else { Some(v) };
            self.extract(m, rhs, assign, resolver);
        } else {
            self.extract(m, v, None, resolver);
        }
    }

    fn validation_order(&self, m: &Module, unit: &NumPyOptimizationUnit, use_helper: bool) -> ValidationOrder {
        let nodes: FxHashSet<ValueId> = self.arena.preorder(unit.expr).iter().filter_map(|x| self.arena[*x].val).map(|v| act(m, v)).collect();
        let mut order = vec![];
        let mut visited = FxHashSet::default();
        fn visit(
            m: &Module,
            v: ValueId,
            nodes: &FxHashSet<ValueId>,
            helper: Option<&FxHashMap<ValueId, Vec<ValueId>>>,
            visited: &mut FxHashSet<ValueId>,
            order: &mut ValidationOrder,
            assign: Option<ValueId>,
        ) {
            let v = act(m, v);
            if !visited.insert(v) {
                return;
            }
            for c in m.used_values(v.into()) {
                visit(m, c, nodes, helper, visited, order, assign);
            }
            if let Some(steps) = helper.and_then(|h| h.get(&v)) {
                // Arguments run before the callee body. The summary's checks retain the
                // body's statement order, which need not match its collapsed expression.
                for s in steps {
                    let s = act(m, *s);
                    if nodes.contains(&s) && visited.insert(s) {
                        order.push((s, assign));
                    }
                }
            }
            if nodes.contains(&v) {
                order.push((v, assign));
            }
        }
        let helper = if use_helper { Some(&self.helper_order) } else { None };
        visit(m, unit.value, &nodes, helper, &mut visited, &mut order, unit.assign);
        order
    }

    fn set_validation_order(m: &Module, a: &ExprArena, unit: &mut NumPyOptimizationUnit, order: &ValidationOrder) {
        // Composing a helper can expose several reads of the same actual IR value.
        // Each expression leaf still needs its own layout/binding entry.
        let mut nodes: FxHashMap<ValueId, Vec<ExprId>> = FxHashMap::default();
        for x in a.preorder(unit.expr) {
            if let Some(v) = a[x].val {
                nodes.entry(act(m, v)).or_default().push(x);
            }
        }
        for (v, asg) in order {
            if let Some(es) = nodes.get(v) {
                for e in es {
                    unit.validation_order.push((*asg, *e));
                }
            }
        }
    }

    /// `NumPyFunctionExpressions(func, rd, cfg, sideEffects, resolveHelper)`.
    pub fn new(
        m: &mut Module,
        func: FuncId,
        rd: Rc<RDInspector>,
        cfg: Rc<CFGraph>,
        side_effects: SideEffectResult,
        mut resolver: Option<&mut HelperResolver<'_>>,
    ) -> Self {
        let types = NumPyPrimitiveTypes::new(m);
        let mut s = Self {
            func,
            types,
            rd,
            cfg,
            side_effects,
            arena: ExprArena::new(),
            exprs: vec![],
            destinations: vec![],
            extracted: FxHashSet::default(),
            helper_substitutions: vec![],
            helper_order: FxHashMap::default(),
            expressions: vec![],
        };
        Walk::new().run_func(m, func, &mut |_, m, v| s.pre_hook(m, v, &mut resolver));
        for i in 0..s.destinations.len() {
            let mut unit = s.destinations[i].unit.clone();
            let order = s.validation_order(m, &unit, false);
            Self::set_validation_order(m, &s.arena, &mut unit, &order);
            s.destinations[i].unit = unit;
        }
        let mut order = vec![];
        for u in &s.exprs {
            order.extend(s.validation_order(m, u, true));
        }
        let rd = s.rd.clone();
        let cfg = s.cfg.clone();
        let dags = get_forwarding_dags(m, func, &rd, &cfg, &s.side_effects, &mut s.arena, &s.exprs);
        for dag in dags {
            let mut assignments = vec![];
            let mut substitutions = s.helper_substitutions.clone();
            let root = do_forwarding(&dag, &mut s.arena, &mut s.exprs, &mut assignments, Some(&mut substitutions));
            let mut unit = s.exprs[root].clone();
            Self::set_validation_order(m, &s.arena, &mut unit, &order);
            s.exprs[root] = unit;
            s.expressions.push(Expression { unit: root, assignments, substitutions });
        }
        s
    }
}

// ---------------------------------------------------------------------------
// NumPyExpressionAnalysis
// ---------------------------------------------------------------------------

/// Port of `NumPyExpressionResult`.
///
/// PORT-NOTE: extraction needs `&mut Module` while analysis results are shared, so
/// the per-function cache uses interior mutability and is filled lazily by the
/// passes (`get`/`take`), like the C++ lazy `get`.
pub struct NumPyExpressionResult {
    pub reaching: RDResult,
    pub side_effects: SideEffectResult,
    functions: RefCell<FxHashMap<Id, Option<NumPyFunctionExpressions>>>,
}

impl AnalysisResult for NumPyExpressionResult {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl NumPyExpressionResult {
    pub fn new(reaching: RDResult, side_effects: SideEffectResult) -> Self {
        Self { reaching, side_effects, functions: RefCell::new(FxHashMap::default()) }
    }

    /// `extract(func, resolver)`.
    pub fn extract(&self, m: &mut Module, func: FuncId, resolver: Option<&mut HelperResolver<'_>>) -> Option<NumPyFunctionExpressions> {
        let uid = m.var_uid(func);
        let rd = self.reaching.results.get(&uid)?.clone();
        let cfg = self.reaching.cfg_result.graphs.get(&uid)?.clone();
        Some(NumPyFunctionExpressions::new(m, func, rd, cfg, self.side_effects.clone(), resolver))
    }

    /// `get(func)` then move out (`take(func)`).
    pub fn take(&self, m: &mut Module, func: FuncId) -> Option<NumPyFunctionExpressions> {
        let uid = m.var_uid(func);
        let cached = self.functions.borrow_mut().remove(&uid);
        match cached {
            Some(c) => c,
            None => self.extract(m, func, None),
        }
    }
}

/// Port of `NumPyExpressionAnalysis`.
pub struct NumPyExpressionAnalysis {
    reaching_def_key: String,
    side_effects_key: String,
}

impl NumPyExpressionAnalysis {
    pub const KEY: &'static str = "core-numpy-expressions";
    pub fn new(reaching_def_key: impl Into<String>, side_effects_key: impl Into<String>) -> Self {
        Self { reaching_def_key: reaching_def_key.into(), side_effects_key: side_effects_key.into() }
    }
}

impl Analysis for NumPyExpressionAnalysis {
    fn key(&self) -> String {
        Self::KEY.to_string()
    }
    fn run(&mut self, m: &Module, ctx: &dyn PassContext) -> Box<dyn AnalysisResult> {
        // PORT-NOTE: missing dependency results are computed here (C++ would crash).
        let rd = match ctx.get_analysis::<RDResult>(&self.reaching_def_key) {
            Some(r) => r.clone(),
            None => RDAnalysis::compute(m, &CFAnalysis::compute(m)),
        };
        let se = match ctx.get_analysis::<SideEffectResult>(&self.side_effects_key) {
            Some(s) => s.clone(),
            None => {
                let dom = DominatorAnalysis::compute(&rd.cfg_result);
                let cap = CaptureAnalysis::compute(m, &rd, &dom);
                SideEffectAnalysis::compute(m, &cap, true)
            }
        };
        Box::new(NumPyExpressionResult::new(rd, se))
    }
}

// ---------------------------------------------------------------------------
// NumPyFusionPass
// ---------------------------------------------------------------------------

/// NumPy operator fusion pass.
pub struct NumPyFusionPass {
    expressions_key: String,
    /// Key of the side effect analysis
    side_effects_key: String,
}

impl NumPyFusionPass {
    pub const KEY: &'static str = "core-numpy-fusion";
    /// Constructs a NumPy fusion pass.
    pub fn new(expressions_key: impl Into<String>, side_effects_key: impl Into<String>) -> Self {
        Self { expressions_key: expressions_key.into(), side_effects_key: side_effects_key.into() }
    }

    fn visit(&self, m: &mut Module, func: FuncId, ctx: &dyn PassContext) {
        let Some(exprs) = ctx.get_analysis::<NumPyExpressionResult>(&self.expressions_key) else { return };
        let Some(mut ex) = exprs.take(m, func) else { return };
        let se = ctx.get_analysis::<SideEffectResult>(&self.side_effects_key).unwrap_or(&exprs.side_effects).clone();
        let types = ex.types;
        for d in ex.destinations.iter_mut() {
            d.optimize(m, &mut ex.arena, &types);
        }
        for e in ex.expressions.clone() {
            let mut unit = ex.exprs[e.unit].clone();
            if optimize_unit(m, &mut ex.arena, &mut unit, &types, &se) {
                // Remove producer assignments only after their replacement was emitted;
                // rejected candidates must retain the original computation and lifetime.
                for asg in &e.assignments {
                    let rep = match unit.validation_flows.get(asg) {
                        Some(f) => *f,
                        None => m.nr_series_flow(),
                    };
                    m.replace_all_value(*asg, rep);
                }
            }
            ex.exprs[e.unit] = unit;
        }
    }
}

impl Pass for NumPyFusionPass {
    fn key(&self) -> String {
        Self::KEY.to_string()
    }
    fn should_run(&self, m: &Module) -> bool {
        numpy_should_run(m)
    }
    fn run(&mut self, m: &mut Module, ctx: &dyn PassContext) {
        for f in bodied_funcs(m) {
            self.visit(m, f, ctx);
        }
    }
}

// ---------------------------------------------------------------------------
// NumPyLifetimePass
// ---------------------------------------------------------------------------

fn owns_array_result(m: &mut Module, v: ValueId, types: &NumPyPrimitiveTypes) -> bool {
    if !npt(m, v, types).is_array() {
        return false;
    }
    if has_lifetime_tag(m, v, "numpy.lifetime.owned") || is_owned_index(m, v, types) {
        return true;
    }
    let mut a = ExprArena::new();
    let mut leaves = vec![];
    parse(m, &mut a, v, &mut leaves, types, false, None).is_some_and(|e| has_owned_result_m(m, &a, e))
}

fn is_array_update(m: &mut Module, call_v: ValueId, types: &NumPyPrimitiveTypes) -> bool {
    let Some(callee) = callee_func(m, call_v) else { return false };
    let args = call_args(m, call_v).unwrap();
    if args.len() < 2 || !is_array_type(m, parent_type(m, callee)) || !npt(m, args[0], types).is_array() {
        return false;
    }
    let name = unmangled(m, callee);
    if name == magic::SETITEM && args.len() == 3 {
        let it = vtype(m, args[1]);
        return is_basic_index_type(m, it, types) || is_copy_index_type(m, it, types);
    }
    const UPDATES: [&str; 12] = [
        "__iadd__", "__isub__", "__imul__", "__itruediv__", "__ifloordiv__", "__imod__", "__ipow__", "__iand__", "__ior__",
        "__ixor__", "__ilshift__", "__irshift__",
    ];
    args.len() == 2 && UPDATES.contains(&name.as_str())
}

#[derive(Clone, Debug)]
struct Definition {
    assignment: ValueId,
    series: ValueId,
}
#[derive(Clone, Debug)]
struct Replacement {
    assignment: ValueId,
    owned: bool,
    in_loop: bool,
    safe: bool,
}
#[derive(Clone, Debug)]
struct Use {
    value: ValueId,
    parents: Vec<NodeRef>,
    safe: bool,
}

fn pv(n: NodeRef) -> Option<ValueId> {
    n.as_value()
}

struct LocalLifetime {
    types: NumPyPrimitiveTypes,
    candidates: Vec<Definition>,
    replacements: indexmap::IndexMap<Id, Vec<Replacement>>,
    uses: FxHashMap<Id, Vec<Use>>,
}

impl LocalLifetime {
    fn handle_assign(&mut self, w: &Walk, m: &mut Module, assignment: ValueId) {
        let types = self.types;
        let series_p = if w.depth() > 0 { w.parent(0).and_then(pv).filter(|p| m.as_series_flow(*p).is_some()) } else { None };
        let (lhs, rhs) = {
            let a = m.as_assign_instr(assignment).unwrap();
            (a.lhs, a.rhs)
        };
        let lt = var_ty(m, lhs);
        if NumPyType::get(m, lt, &types).is_array() {
            let mut safe = series_p.is_some() && !m.is_global(lhs);
            let mut in_loop = false;
            for p in w.parents() {
                let Some(v) = pv(*p) else { continue };
                safe &= m.as_try_catch_flow(v).is_none() && !has_lifetime_tag(m, v, "numpy.lifetime.expression");
                if let Some(l) = m.as_for_flow(v) {
                    safe &= !l.is_parallel();
                }
                if let Some(l) = m.as_imperative_for_flow(v) {
                    safe &= !l.is_parallel();
                }
                in_loop |= m.as_for_flow(v).is_some() || m.as_imperative_for_flow(v).is_some() || m.as_while_flow(v).is_some();
            }
            let owned = owns_array_result(m, rhs, &types);
            self.replacements.entry(m.var_uid(lhs)).or_default().push(Replacement { assignment, owned, in_loop, safe });
        }
        let Some(series_p) = series_p else { return };
        if m.is_global(lhs) || !owns_array_result(m, rhs, &types) {
            return;
        }
        for p in w.parents() {
            if let Some(v) = pv(*p)
                && (m.as_try_catch_flow(v).is_some() || has_lifetime_tag(m, v, "numpy.lifetime.expression"))
            {
                return;
            }
        }
        self.candidates.push(Definition { assignment, series: series_p });
    }

    fn safe_replacement_read(&self, m: &mut Module, u: &Use) -> bool {
        for p in &u.parents {
            if pv(*p).is_some_and(|v| m.as_try_catch_flow(v).is_some()) {
                return false;
            }
        }
        if u.safe {
            return true;
        }
        let read = u.value;
        let Some(parent) = u.parents.last().copied().and_then(pv) else { return false };
        if m.as_var_value(read).is_none() {
            return false;
        }
        // Returning the current buffer is safe only for replacement cleanup: there
        // is no later assignment on this path. Ordinary last-use cleanup rejects it.
        if let Some(r) = m.as_return_instr(parent) {
            return r.value.map(|x| act(m, x)) == Some(act(m, read));
        }
        let callee = callee_func(m, parent);
        let result = if u.parents.len() >= 2 { pv(u.parents[u.parents.len() - 2]).filter(|v| m.as_return_instr(*v).is_some()) } else { None };
        if let (Some(callee), Some(result)) = (callee, result) {
            let rv = m.as_return_instr(result).unwrap().value.map(|x| act(m, x));
            if rv == Some(act(m, parent))
                && unmangled(m, callee) == magic::NEW
                && parent_type(m, callee).is_some_and(|p| m.type_name(p) == "Tuple")
            {
                return true;
            }
        }
        let args = call_args(m, parent).unwrap_or_default();
        let Some(callee) = callee else { return false };
        if args.is_empty() || act(m, args[0]) != act(m, read) {
            return false;
        }
        if args.len() == 1 && starts(&fname(m, callee), format!("{}[", mangle::func("std.internal.builtin", "len"))) {
            return true;
        }
        if !is_array_type(m, parent_type(m, callee)) {
            return false;
        }
        const METADATA: [&str; 6] = ["shape", "strides", "size", "ndim", "itemsize", "nbytes"];
        let un = unmangled(m, callee);
        if args.len() == 1 && (METADATA.contains(&un.as_str()) || un == "__len__") {
            return true;
        }
        let types = self.types;
        is_array_update(m, parent, &types)
            && u.parents.len() >= 2
            && pv(u.parents[u.parents.len() - 2]).is_some_and(|v| m.as_series_flow(v).is_some())
    }

    fn safe_read(&self, w: &Walk, m: &mut Module, read: ValueId) -> bool {
        if w.depth() == 0 {
            return false;
        }
        let Some(parent) = w.parent(0).and_then(pv) else { return false };
        let types = self.types;
        if let Some(asg) = m.as_assign_instr(parent) {
            // Only compiler-generated borrowed input bindings are nonescaping here.
            // User aliases and leaves already released by fusion must not qualify.
            return act(m, asg.rhs) == act(m, read) && has_lifetime_tag(m, parent, "numpy.lifetime.input");
        }
        let Some(args) = call_args(m, parent) else { return false };
        if owns_array_result(m, parent, &types) {
            return true;
        }
        let callee = callee_func(m, parent);
        if let Some(c) = callee
            && is_ufunc_type(m, parent_type(m, c))
            && unmangled(m, c) == "__call__"
            && (args.len() == 4 || args.len() == 5)
            && w.depth() >= 2
            && w.parent(1).and_then(pv).is_some_and(|v| m.as_series_flow(v).is_some())
        {
            let numeric = args.iter().skip(1).all(|x| npt(m, *x, &types).is_some() || vt_is_none(m, *x, &types));
            if numeric {
                return true;
            }
        }
        let r = act(m, read);
        if is_array_update(m, parent, &types) {
            return act(m, *args.last().unwrap()) == r
                || (unmangled(m, callee.unwrap()) == magic::SETITEM && (act(m, args[0]) == r || act(m, args[1]) == r));
        }
        let output = npt(m, parent, &types);
        let Some(c) = callee else { return false };
        if !(is_array_type(m, parent_type(m, c))
            && output.is_some()
            && !output.is_array()
            && unmangled(m, c) == magic::GETITEM
            && args.len() == 2
            && act(m, args[0]) == r)
        {
            return false;
        }
        let bt = vtype(m, args[1]);
        is_basic_index_type(m, bt, &types)
    }

    fn pre_hook(&mut self, w: &Walk, m: &mut Module, v: ValueId) {
        if m.as_assign_instr(v).is_some() {
            self.handle_assign(w, m, v);
            return;
        }
        let safe = m.as_var_value(v).is_some() && self.safe_read(w, m, v);
        for var in m.used_variables(v.into()) {
            let Some(t) = m.var_type(var) else { continue };
            if is_array_type(m, Some(t)) || m.type_name(t).starts_with("Tuple[") {
                self.uses.entry(m.var_uid(var)).or_default().push(Use { value: v, parents: w.parents().to_vec(), safe });
            }
        }
    }
}

/// Release proven-owned, nonescaping arrays after fusion, using fresh use analysis.
pub struct NumPyLifetimePass {
    reaching_def_key: String,
}

impl NumPyLifetimePass {
    pub const KEY: &'static str = "core-numpy-lifetime";
    pub fn new(reaching_def_key: impl Into<String>) -> Self {
        Self { reaching_def_key: reaching_def_key.into() }
    }

    fn visit(&self, m: &mut Module, func: FuncId, ctx: &dyn PassContext) {
        // Analyze the rewritten IR: fusion may have transferred ownership or introduced
        // input bindings. Reusing the pre-fusion use graph could schedule a second free.
        let Some(result) = ctx.get_analysis::<RDResult>(&self.reaching_def_key) else { return };
        let uid = m.var_uid(func);
        let Some(definitions) = result.results.get(&uid).cloned() else { return };
        let Some(graph) = result.cfg_result.graphs.get(&uid).cloned() else { return };

        let types = NumPyPrimitiveTypes::new(m);
        let mut lt = LocalLifetime { types, candidates: vec![], replacements: Default::default(), uses: FxHashMap::default() };
        // PORT-NOTE: C++ handle(AssignInstr) runs after preHook; AssignInstr nodes are
        // skipped by preHook, so dispatching both from one hook is equivalent.
        Walk::new().run_func(m, func, &mut |w, m, v| lt.pre_hook(w, m, v));

        let block_values = |m: &Module, b| -> Vec<Id> { graph.block(b).values().iter().map(|v| graph.uid(m, *v)).collect() };

        // Tuple unpacking can hide fresh gathered arrays. Transfer a field's ownership
        // only from a local constructor with one extraction per field and no tuple escape;
        // otherwise another tuple reader could retain the storage we intend to release.
        let keys: Vec<Id> = lt.replacements.keys().copied().collect();
        for k in &keys {
            let n = lt.replacements[k].len();
            for i in 0..n {
                let rep = lt.replacements[k][i].clone();
                let rhs = m.as_assign_instr(rep.assignment).unwrap().rhs;
                let Some(ex) = m.as_extract_instr(rhs) else { continue };
                let (ev, field) = (ex.val, ex.field.clone());
                let Some(rvar) = var_of(m, ev) else { continue };
                if !rep.safe || definitions.is_invalid(m, rvar) {
                    continue;
                }
                let reaching = definitions.get_reaching_definitions(m, rvar, ev);
                if reaching.len() != 1 || !reaching[0].known() {
                    continue;
                }
                let tuple_assignment = reaching[0].assignment;
                let Some(ctor) = reaching[0].assignee_value() else { continue };
                let Some(callee) = callee_func(m, ctor) else { continue };
                let cargs = call_args(m, ctor).unwrap();
                let Some(pt) = parent_type(m, callee) else { continue };
                if !matches!(m.type_kind(pt), TypeKind::Record { .. }) || m.type_name(pt) != "Tuple" || unmangled(m, callee) != magic::NEW {
                    continue;
                }
                let et = vtype(m, ev);
                let fi = m.member_index(et, &field);
                if fi < 0 || fi as usize >= cargs.len() || !owns_array_result(m, cargs[fi as usize], &types) {
                    continue;
                }
                let Some(block) = graph.get_block(m, tuple_assignment) else { continue };
                let tuple_uid = graph.uid(m, tuple_assignment);
                let mut safe = true;
                let mut fields: FxHashSet<String> = FxHashSet::default();
                let uses = lt.uses.get(&m.var_uid(rvar)).cloned().unwrap_or_default();
                for u in &uses {
                    let at_use = definitions.get_reaching_definitions(m, rvar, u.value);
                    if !at_use.iter().any(|d| d.assignment_uid() == tuple_uid) {
                        continue;
                    }
                    let member = u.parents.last().copied().and_then(pv).filter(|v| m.as_extract_instr(*v).is_some());
                    let assignment = if u.parents.len() < 2 { None } else { pv(u.parents[u.parents.len() - 2]).filter(|v| m.as_assign_instr(*v).is_some()) };
                    let uval = m.value_uid(u.value);
                    let mut after = false;
                    for x in block_values(m, block) {
                        if x == uval {
                            break;
                        }
                        after |= x == tuple_uid;
                    }
                    let ok = at_use.len() == 1
                        && member.is_some()
                        && assignment.is_some_and(|a| act(m, m.as_assign_instr(a).unwrap().rhs) == act(m, member.unwrap()))
                        && u.parents.len() >= 3
                        && pv(u.parents[u.parents.len() - 3]).is_some_and(|v| m.as_series_flow(v).is_some())
                        && graph.get_block(m, u.value) == Some(block)
                        && after
                        && fields.insert(m.as_extract_instr(member.unwrap()).unwrap().field.clone());
                    if !ok {
                        safe = false;
                        break;
                    }
                }
                lt.replacements.get_mut(k).unwrap()[i].owned = safe && fields.contains(&field);
            }
        }

        // Replacement cleanup spans serial iterations; last-use cleanup below stays
        // within one basic block. A variable must participate in only one protocol.
        let mut replaced: FxHashSet<Id> = FxHashSet::default();
        if let Some(body) = m.func_body(func).filter(|b| m.as_series_flow(*b).is_some()) {
            for k in &keys {
                let reps = lt.replacements[k].clone();
                let mut loop_owned = false;
                let mut safe = true;
                for r in &reps {
                    loop_owned |= r.in_loop && r.owned;
                    safe &= r.safe;
                }
                for u in lt.uses.get(k).cloned().unwrap_or_default() {
                    safe &= lt.safe_replacement_read(m, &u);
                }
                let variable = m.as_assign_instr(reps[0].assignment).unwrap().lhs;
                if !loop_owned || !safe || definitions.is_invalid(m, variable) {
                    continue;
                }
                let bt = m.get_bool_type();
                let vt = var_ty(m, variable);
                let owned = m.nr_var(bt, false, "");
                let owner = m.nr_var(vt, false, "");
                m.func_push_symbol(func, owned);
                m.func_push_symbol(func, owner);
                let f0 = m.get_bool(false);
                let init = m.nr_assign(owned, f0);
                m.series_insert(body, 0, init);
                replaced.insert(*k);
                let it = m.get_int_type();
                let release = fusion_func(m, "_free_replacement", &[vt, it], &[]);
                for r in &reps {
                    let s = m.nr_series_flow();
                    // Finish the RHS before freeing the previous buffer: it may read that
                    // buffer or throw. The saved owner is valid only while the flag is true,
                    // so borrowed initial values and unproven replacements are never freed.
                    let rhs = m.as_assign_instr(r.assignment).unwrap().rhs;
                    let res = make_var(m, rhs, s, Some(func), false);
                    let ov = m.nr_var_value(owner);
                    let th = m.get_int(numpy_options().replacement_free_threshold as i64);
                    let rc = call(m, release, vec![ov, th]);
                    let then = series(m, &[rc]).unwrap();
                    let cond = m.nr_var_value(owned);
                    let iff = m.nr_if_flow(cond, then, None);
                    m.series_push(s, iff);
                    if r.owned {
                        let rv = m.nr_var_value(res);
                        let a = m.nr_assign(owner, rv);
                        m.series_push(s, a);
                    }
                    let b = m.get_bool(r.owned);
                    let a = m.nr_assign(owned, b);
                    m.series_push(s, a);
                    let rv = m.nr_var_value(res);
                    let fi = m.nr_flow_instr(s, rv);
                    m.as_assign_instr_mut(r.assignment).unwrap().rhs = fi;
                }
            }
        }

        let mut positions: FxHashMap<Id, usize> = FxHashMap::default();
        for cand in &lt.candidates {
            if let Some(b) = graph.get_block(m, cand.assignment) {
                for (i, uid) in block_values(m, b).into_iter().enumerate() {
                    positions.insert(uid, i);
                }
            }
        }
        let mut insertions: indexmap::IndexMap<ValueId, Vec<ValueId>> = Default::default();
        // Release a named allocation only after its last proven nonescaping statement.
        // Ambiguous definitions, control-flow boundaries, and returned storage retain
        // GC ownership; this deliberately does not follow arbitrary aliases or views.
        for cand in lt.candidates.clone() {
            let assignment = cand.assignment;
            let variable = m.as_assign_instr(assignment).unwrap().lhs;
            let vuid = m.var_uid(variable);
            if replaced.contains(&vuid) {
                continue;
            }
            let Some(block) = graph.get_block(m, assignment) else { continue };
            if definitions.is_invalid(m, variable) {
                continue;
            }
            let auid = m.value_uid(assignment);
            let mut last = assignment;
            let apos = positions[&auid];
            let mut last_pos = apos;
            let mut safe = true;
            for u in lt.uses.get(&vuid).cloned().unwrap_or_default() {
                let reaching = definitions.get_reaching_definitions(m, variable, u.value);
                if !reaching.iter().any(|d| d.assignment_uid() == auid) {
                    continue;
                }
                let sidx = u.parents.iter().position(|p| pv(*p) == Some(cand.series));
                let statement = sidx.and_then(|i| u.parents.get(i + 1).copied()).and_then(pv);
                let pos = positions.get(&m.value_uid(u.value)).copied();
                let bad = !u.safe
                    || reaching.len() != 1
                    || statement.is_none()
                    || graph.get_block(m, u.value) != Some(block)
                    || pos.is_none_or(|p| p <= apos)
                    || statement.is_some_and(|s| m.as_return_instr(s).is_some() || m.kind(s).is_flow());
                if bad {
                    safe = false;
                    break;
                }
                if pos.unwrap() > last_pos {
                    last_pos = pos.unwrap();
                    last = statement.unwrap();
                }
            }
            if !safe {
                continue;
            }
            let vt = var_ty(m, variable);
            let owner = m.nr_var(vt, false, "");
            m.func_push_symbol(func, owner);
            let vv = m.nr_var_value(variable);
            let a = m.nr_assign(owner, vv);
            insertions.entry(act(m, assignment)).or_default().push(a);
            let release = fusion_func(m, "_free", &[vt], &[]);
            let ov = m.nr_var_value(owner);
            let rc = call(m, release, vec![ov]);
            insertions.entry(act(m, last)).or_default().push(rc);
        }

        // InsertReleases
        let mut all_series = vec![];
        Walk::new().run_func(m, func, &mut |_, m, v| {
            if m.as_series_flow(v).is_some() {
                all_series.push(v);
            }
        });
        for s in all_series {
            let mut i = 0;
            while let Some(x) = m.as_series_flow(s).and_then(|f| f.series.get(i).copied()) {
                let key = act(m, x);
                if let Some(vals) = insertions.get(&key) {
                    for (j, v) in vals.clone().into_iter().enumerate() {
                        m.series_insert(s, i + 1 + j, v);
                    }
                }
                i += 1;
            }
        }

        // ReleaseArguments
        let mut calls = vec![];
        Walk::new().run_func(m, func, &mut |_, m, v| {
            if m.as_call_instr(v).is_some() {
                calls.push(v);
            }
        });
        for c in calls {
            let args = call_args(m, c).unwrap();
            if !is_array_update(m, c, &types) || !owns_array_result(m, *args.last().unwrap(), &types) {
                continue;
            }
            let s = m.nr_series_flow();
            let mut clone = CloneVisitor::new(true);
            let mut arguments = vec![];
            // Preserve receiver/index/RHS evaluation order and retain the exact RHS
            // allocation. Re-evaluating an argument for cleanup could repeat side effects.
            for x in &args {
                let cl = clone.clone_value(m, *x);
                let b = make_var(m, cl, s, Some(func), false);
                arguments.push(m.nr_var_value(b));
            }
            let callee = callee_func(m, c).unwrap();
            let temp = *arguments.last().unwrap();
            let cl = call(m, callee, arguments);
            let res = make_var(m, cl, s, Some(func), false);
            let tt = vtype(m, temp);
            let release = fusion_func(m, "_free", &[tt], &[]);
            let rc = call(m, release, vec![temp]);
            m.series_push(s, rc);
            let rv = m.nr_var_value(res);
            let fi = m.nr_flow_instr(s, rv);
            m.replace_all_value(c, fi);
        }
    }
}

impl Pass for NumPyLifetimePass {
    fn key(&self) -> String {
        Self::KEY.to_string()
    }
    fn should_run(&self, m: &Module) -> bool {
        numpy_should_run(m)
    }
    fn run(&mut self, m: &mut Module, ctx: &dyn PassContext) {
        for f in bodied_funcs(m) {
            self.visit(m, f, ctx);
        }
    }
}

// ---------------------------------------------------------------------------
// NumPyInlinePass
// ---------------------------------------------------------------------------

const MAX_HELPER_DEPTH: u32 = 4;
const MAX_EXPRESSION_NODES: u32 = 16;
const MAX_TEMPLATE_NODES: u32 = 128;
const MAX_HELPER_STATEMENTS: usize = 17;
const MAX_CALLER_EXPANSIONS: u32 = 16;
const MAX_CALLER_NODES: u32 = 256;

#[derive(Clone, Debug, Default)]
struct InlineTemplate {
    value: Option<ValueId>,
    size: u32,
    direct_bindings: bool,
    locals: Vec<VarId>,
    // Compose the raw expression and its check order, not the emitted validation
    // FlowInstr: nested wrappers would hide arithmetic from the fusion parser.
    raw: Option<ValueId>,
    order: Vec<ValueId>,
    raw_size: u32,
    nodes: u32,
    depth: u32,
    expansions: u32,
    reduction: bool,
}

fn count_nodes(m: &mut Module, v: ValueId) -> u32 {
    let mut n = 0;
    Walk::new().run(m, v, &mut |_, _, _| n += 1);
    n
}

/// `KeepInvariants`: RD/SE describe the original program, not detached summary
/// clones. Preserve literals and pure, parameter-independent subtrees (notably ufunc
/// receivers and out=None). This shares IR identity, not evaluation.
fn keep_invariants(m: &mut Module, clone: &mut CloneVisitor, root: ValueId, se: &SideEffectResult) {
    let mut keep = vec![];
    Walk::new().run(m, root, &mut |_, m, v| {
        if se.has_side_effect(m, v) {
            return;
        }
        let mut local = false;
        Walk::new().run(m, v, &mut |_, m, x| {
            for var in m.used_variables(x.into()) {
                local |= !m.is_global(var) && !m.is_func(var);
            }
        });
        if !local {
            keep.push(v);
        }
    });
    for v in keep {
        clone.force_remap(m, v.into(), v.into());
    }
}

fn get_statements(m: &Module, callee: FuncId) -> Vec<ValueId> {
    let Some(fd) = m.func(callee) else { return vec![] };
    let Some(body) = fd.body().filter(|b| m.as_series_flow(*b).is_some()) else { return vec![] };
    if fd.generator || fd.is_async || m.var_name(callee).starts_with("std.numpy.") || is_no_inline(m, callee) {
        return vec![];
    }
    fn flatten(m: &Module, s: ValueId, out: &mut Vec<ValueId>) {
        for x in m.as_series_flow(s).unwrap().series.clone() {
            if m.as_series_flow(x).is_some() {
                flatten(m, x, out);
            } else {
                out.push(x);
            }
        }
    }
    let mut st = vec![];
    flatten(m, body, &mut st);
    if st.is_empty() || st.len() > MAX_HELPER_STATEMENTS {
        return vec![];
    }
    let last = *st.last().unwrap();
    if m.as_return_instr(last).is_none_or(|r| r.value.is_none()) {
        return vec![];
    }
    for s in &st {
        if *s == last {
            continue;
        }
        match m.as_assign_instr(*s) {
            Some(a) if !m.is_global(a.lhs) => {}
            _ => return vec![],
        }
    }
    st
}

fn bodied_callee(m: &Module, call_v: ValueId) -> Option<FuncId> {
    callee_func(m, call_v).filter(|f| m.func(*f).is_some_and(|d| d.is_bodied()))
}

fn collect_bodied_callees(m: &mut Module, root: NodeRef) -> Vec<FuncId> {
    let mut out = vec![];
    let mut f = |_: &mut Walk, m: &mut Module, v: ValueId| {
        if m.as_call_instr(v).is_some()
            && let Some(c) = bodied_callee(m, v)
        {
            out.push(c);
        }
    };
    match root {
        NodeRef::Value(v) => Walk::new().run(m, v, &mut f),
        NodeRef::Module => Walk::new().run_module(m, &mut f),
        NodeRef::Var(v) => Walk::new().run_func(m, v, &mut f),
        NodeRef::Type(_) => {}
    }
    out
}

/// Port of `NumPyInlinePass`.
pub struct NumPyInlinePass {
    expressions_key: String,
}

impl NumPyInlinePass {
    pub const KEY: &'static str = "core-numpy-inline";
    pub fn new(expressions_key: impl Into<String>) -> Self {
        Self { expressions_key: expressions_key.into() }
    }

    fn prepare(
        m: &mut Module,
        analysis: &NumPyExpressionResult,
        templates: &FxHashMap<Id, InlineTemplate>,
        callee: FuncId,
    ) -> InlineTemplate {
        let statements = get_statements(m, callee);
        let Some(&returned) = statements.last() else { return InlineTemplate::default() };
        let mut depth = 1u32;
        let mut expansions = 1u32;
        let mut nested_nodes = 0u32;
        let body = m.func_body(callee).unwrap();
        let mut estimated = count_nodes(m, body);
        let mut composable = true;
        let se = analysis.side_effects.clone();

        let mut resolver = |m: &mut Module, call_v: ValueId, order: &mut Vec<ValueId>| -> Option<ValueId> {
            let nested = bodied_callee(m, call_v)?;
            let summary = templates.get(&m.var_uid(nested))?;
            // Reductions are only parsed at expression roots. Do not turn a nested
            // scalar reduction into pointwise work or duplicate its materialization.
            let args = call_args(m, call_v).unwrap();
            let params = m.func_args(nested);
            let mut stable = summary.value.is_some() && summary.direct_bindings && !summary.reduction && args.len() == params.len();
            for x in &args {
                stable &= var_of(m, *x).is_some_and(|v| !m.is_global(v));
            }
            // Count occurrences, not cache entries. Preflight conservative upper bounds
            // before cloning so a branching helper graph cannot grow exponentially.
            if !stable
                || summary.depth >= MAX_HELPER_DEPTH
                || expansions + summary.expansions > MAX_CALLER_EXPANSIONS
                || nested_nodes + summary.nodes > MAX_EXPRESSION_NODES
                || estimated + summary.raw_size > MAX_TEMPLATE_NODES
            {
                composable = false;
                return None;
            }
            depth = depth.max(summary.depth + 1);
            expansions += summary.expansions;
            nested_nodes += summary.nodes;
            estimated += summary.raw_size;
            let raw = summary.raw.unwrap();
            let mut clone = CloneVisitor::new(true);
            keep_invariants(m, &mut clone, raw, &se);
            let arguments: FxHashMap<VarId, ValueId> = params.iter().copied().zip(args.iter().copied()).collect();
            let mut binds = vec![];
            Walk::new().run(m, raw, &mut |_, m, x| {
                if let Some(v) = var_of(m, x)
                    && let Some(actual) = arguments.get(&v)
                {
                    binds.push((x, *actual));
                }
            });
            for (r, actual) in binds {
                clone.force_remap(m, r.into(), actual.into());
            }
            let value = clone.clone_value(m, raw);
            for s in &summary.order {
                order.push(clone.clone_value(m, *s));
            }
            Some(value)
        };
        let extracted = analysis.extract(m, callee, Some(&mut resolver));
        let Some(mut ex) = extracted else { return InlineTemplate::default() };
        if !ex.destinations.is_empty() || ex.expressions.len() != 1 || !composable {
            return InlineTemplate::default();
        }
        let expression = ex.expressions[0].clone();
        let mut unit = ex.exprs[expression.unit].clone();
        let root = unit.expr;
        if ex.arena.count_nodes(root) as u32 > MAX_EXPRESSION_NODES
            || (!expression.substitutions.is_empty() && has_ufunc_argument_effects(m, &ex.arena, root, &ex.side_effects))
        {
            return InlineTemplate::default();
        }
        let ret_val = m.as_return_instr(returned).unwrap().value.unwrap();
        if act(m, unit.value) != act(m, ret_val) {
            let Some(rv) = var_of(m, ret_val) else { return InlineTemplate::default() };
            let Some(asg) = unit.assign else { return InlineTemplate::default() };
            if m.actual_var(m.as_assign_instr(asg).unwrap().lhs) != rv {
                return InlineTemplate::default();
            }
            let defs = ex.rd.get_reaching_definitions(m, rv, ret_val);
            if defs.len() != 1 || defs[0].assignment != CfValue::Real(act(m, asg)) {
                return InlineTemplate::default();
            }
        }
        let mut covered: FxHashSet<ValueId> = expression.assignments.iter().map(|x| act(m, *x)).collect();
        if let Some(a) = unit.assign {
            covered.insert(act(m, a));
        }
        covered.insert(act(m, returned));
        if covered.len() != statements.len() || !statements.iter().all(|s| covered.contains(&act(m, *s))) {
            return InlineTemplate::default();
        }
        let mut clone = CloneVisitor::new(true);
        keep_invariants(m, &mut clone, unit.value, &se);
        for (_, s) in &expression.substitutions {
            keep_invariants(m, &mut clone, *s, &se);
        }
        for (f, s) in &expression.substitutions {
            let cs = clone.clone_value(m, *s);
            clone.force_remap(m, (*f).into(), cs.into());
        }
        let mut value = clone.clone_value(m, unit.value);

        // CheckTemplate
        let params: FxHashSet<VarId> = m.func_args(callee).into_iter().collect();
        let mut size = 0u32;
        let mut valid = true;
        Walk::new().run(m, value, &mut |_, m, x| {
            size += 1;
            if m.kind(x).is_flow()
                || m.as_flow_instr(x).is_some()
                || m.as_assign_instr(x).is_some()
                || m.as_pointer_value(x).is_some()
            {
                valid = false;
            }
            for var in m.used_variables(x.into()) {
                if !m.is_global(var) && !m.is_func(var) && !params.contains(&m.actual_var(var)) {
                    valid = false;
                }
            }
            // Only calls exposed by the NumPy parser participate in composition.
            // A helper hidden inside an opaque leaf must remain in its own function;
            // otherwise later traversal could inline it without charging its depth.
            if m.as_call_instr(x).is_some()
                && let Some(c) = callee_func(m, x)
                && templates.contains_key(&m.var_uid(c))
            {
                valid = false;
            }
        });
        if !valid || size > MAX_TEMPLATE_NODES {
            return InlineTemplate::default();
        }
        let raw = value;
        let raw_size = size;
        let mut order = vec![];
        for (_, e) in unit.validation_order.clone() {
            let v = ex.arena[e].v();
            order.push(clone.clone_value(m, v));
        }
        let mut direct = true;
        for x in ex.arena.preorder(root) {
            if !ex.arena[x].is_leaf() {
                continue;
            }
            let v = ex.arena[x].v();
            direct &= var_of(m, v).is_some_and(|vv| !m.is_global(vv))
                || m.as_int_const(v).is_some()
                || m.as_float_const(v).is_some()
                || m.as_bool_const(v).is_some();
        }
        let original: Vec<ValueId> = unit
            .validation_order
            .iter()
            .filter(|(_, e)| !ex.arena[*e].is_leaf())
            .map(|(_, e)| act(m, ex.arena[*e].v()))
            .collect();
        let mut nodes: FxHashMap<ValueId, ExprId> = FxHashMap::default();
        for x in ex.arena.preorder(root) {
            if let Some(v) = ex.arena[x].val {
                nodes.entry(act(m, v)).or_insert(x);
            }
        }
        let subs: FxHashMap<ValueId, ValueId> = expression.substitutions.iter().map(|(a, b)| (act(m, *a), *b)).collect();
        let mut collapsed = vec![];
        let mut first_leaf: Option<ExprId> = None;
        let mut visited = FxHashSet::default();
        #[allow(clippy::too_many_arguments)]
        fn collect(
            m: &Module,
            cur: ValueId,
            subs: &FxHashMap<ValueId, ValueId>,
            visited: &mut FxHashSet<ValueId>,
            nodes: &FxHashMap<ValueId, ExprId>,
            a: &ExprArena,
            collapsed: &mut Vec<ValueId>,
            first: &mut Option<ExprId>,
        ) {
            let mut cur = act(m, cur);
            while let Some(s) = subs.get(&cur) {
                cur = act(m, *s);
            }
            if !visited.insert(cur) {
                return;
            }
            for c in m.used_values(cur.into()) {
                collect(m, c, subs, visited, nodes, a, collapsed, first);
            }
            let Some(&e) = nodes.get(&cur) else { return };
            if !a[e].is_leaf() {
                collapsed.push(cur);
            } else if first.is_none() {
                *first = Some(e);
            }
        }
        collect(m, unit.value, &subs, &mut visited, &nodes, &ex.arena, &mut collapsed, &mut first_leaf);
        let mut locals = vec![];
        if original != collapsed {
            let Some(first_leaf) = first_leaf.filter(|_| direct) else { return InlineTemplate::default() };
            let validation = m.nr_series_flow();
            let mut c = CodegenContext::new(validation, callee, ex.types);
            codegen_validation(m, &mut ex.arena, &mut unit, &mut c, false);
            let mut checked = CloneVisitor::new(true);
            let mut entries: Vec<(ExprId, VarId)> = c.vars.iter().map(|(a, b)| (*a, *b)).collect();
            entries.sort();
            for (e, var) in entries {
                locals.push(var);
                let vv = m.nr_var_value(var);
                let orig = ex.arena[e].v();
                checked.force_remap(m, orig.into(), vv.into());
            }
            let mut layouts: Vec<(ExprId, VarId)> = c.layouts.iter().map(|(a, b)| (*a, *b)).collect();
            layouts.sort();
            locals.extend(layouts.into_iter().map(|(_, v)| v));
            let fv = m.nr_var_value(c.vars[&first_leaf]);
            let checked_leaf = m.nr_flow_instr(validation, fv);
            set_lifetime_tag(m, checked_leaf, "numpy.validation");
            let fl = ex.arena[first_leaf].v();
            checked.force_remap(m, fl.into(), checked_leaf.into());
            for (f, s) in &expression.substitutions {
                let cs = checked.clone_value(m, *s);
                checked.force_remap(m, (*f).into(), cs.into());
            }
            value = checked.clone_value(m, unit.value);
            let n = count_nodes(m, value);
            if n > MAX_TEMPLATE_NODES {
                return InlineTemplate::default();
            }
            size = n;
        }
        InlineTemplate {
            value: Some(value),
            size,
            direct_bindings: direct,
            locals,
            raw: Some(raw),
            order,
            raw_size,
            nodes: ex.arena.count_nodes(root) as u32,
            depth,
            expansions,
            reduction: ex.arena[root].is_reduction(),
        }
    }
}

impl Pass for NumPyInlinePass {
    fn key(&self) -> String {
        Self::KEY.to_string()
    }
    fn should_run(&self, m: &Module) -> bool {
        numpy_should_run(m)
    }
    fn run(&mut self, m: &mut Module, ctx: &dyn PassContext) {
        let Some(analysis) = ctx.get_analysis::<NumPyExpressionResult>(&self.expressions_key) else { return };
        let types = NumPyPrimitiveTypes::new(m);

        let mut candidates: Vec<FuncId> = vec![];
        let mut seen: FxHashSet<Id> = FxHashSet::default();
        let mut candidate_ids: FxHashSet<Id> = FxHashSet::default();
        for callee in collect_bodied_callees(m, NodeRef::Module) {
            if !seen.insert(m.var_uid(callee)) {
                continue;
            }
            let mut array = return_type(m, callee).is_some_and(|t| NumPyType::get(m, t, &types).is_array());
            for arg in m.func_args(callee) {
                if let Some(t) = m.var_type(arg) {
                    array |= NumPyType::get(m, t, &types).is_array();
                }
            }
            if array && !get_statements(m, callee).is_empty() {
                candidates.push(callee);
                candidate_ids.insert(m.var_uid(callee));
            }
        }

        // Kahn's ordering is iterative even for arbitrarily deep input call graphs.
        // Recursive components (and callers depending on them) never become ready.
        // Finish every immutable summary before rewriting any function body, so a
        // caller cannot bypass depth limits by seeing an already-inlined callee.
        let mut pending: FxHashMap<Id, usize> = FxHashMap::default();
        let mut callers: FxHashMap<Id, Vec<FuncId>> = FxHashMap::default();
        let mut ready: Vec<FuncId> = vec![];
        for &callee in &candidates {
            let body = m.func_body(callee).unwrap();
            let mut deps: Vec<Id> = vec![];
            for d in collect_bodied_callees(m, body.into()) {
                let id = m.var_uid(d);
                if candidate_ids.contains(&id) && !deps.contains(&id) {
                    deps.push(id);
                }
            }
            pending.insert(m.var_uid(callee), deps.len());
            for d in &deps {
                callers.entry(*d).or_default().push(callee);
            }
            if deps.is_empty() {
                ready.push(callee);
            }
        }
        let mut templates: FxHashMap<Id, InlineTemplate> = FxHashMap::default();
        let mut i = 0;
        while i < ready.len() {
            let callee = ready[i];
            i += 1;
            let t = Self::prepare(m, analysis, &templates, callee);
            let id = m.var_uid(callee);
            templates.entry(id).or_insert(t);
            for caller in callers.get(&id).cloned().unwrap_or_default() {
                let p = pending.get_mut(&m.var_uid(caller)).unwrap();
                *p -= 1;
                if *p == 0 {
                    ready.push(caller);
                }
            }
        }

        // Inliner
        for parent in bodied_funcs(m) {
            let mut remaining = MAX_CALLER_EXPANSIONS;
            let mut budget = MAX_CALLER_NODES;
            // PORT-NOTE: calls are collected first, then rewritten in walk order;
            // inlined templates never contain template calls, so this is equivalent.
            let mut calls = vec![];
            Walk::new().run_func(m, parent, &mut |_, m, v| {
                if m.as_call_instr(v).is_some() {
                    calls.push(v);
                }
            });
            for c in calls {
                if remaining == 0 {
                    break;
                }
                let Some(callee) = bodied_callee(m, c) else { continue };
                if callee == parent {
                    continue;
                }
                let Some(prepared) = templates.get(&m.var_uid(callee)) else { continue };
                let args = call_args(m, c).unwrap();
                let params = m.func_args(callee);
                let Some(tvalue) = prepared.value else { continue };
                if prepared.expansions > remaining || args.len() != params.len() {
                    continue;
                }
                // Substitute only stable local reads directly. Otherwise bind every actual
                // argument once, including unused ones, before evaluating the helper body.
                let mut direct = prepared.direct_bindings;
                for x in &args {
                    direct &= var_of(m, *x).is_some_and(|v| !m.is_global(v));
                }
                // The emitted wrapper is part of the caller's budget too. Actual values
                // themselves are retained, not cloned, and still run once in source order.
                let cost = prepared.size + if direct { 0 } else { args.len() as u32 + 2 };
                if cost > budget {
                    continue;
                }
                let mut clone = CloneVisitor::new(true);
                for l in prepared.locals.clone() {
                    let nl = clone.force_clone_var(m, l);
                    m.func_push_symbol(parent, nl);
                }
                let bindings = m.nr_series_flow();
                for (p, x) in params.iter().zip(args.iter()) {
                    if direct {
                        let v = var_of(m, *x).unwrap();
                        clone.force_remap(m, (*p).into(), v.into());
                    } else {
                        let nv = clone.force_clone_var(m, *p);
                        m.func_push_symbol(parent, nv);
                        let a = m.nr_assign(nv, *x);
                        m.series_push(bindings, a);
                    }
                }
                let mut replacement = clone.clone_value(m, tvalue);
                if !direct {
                    replacement = m.nr_flow_instr(bindings, replacement);
                }
                remaining -= prepared.expansions;
                budget -= cost;
                m.replace_all_value(c, replacement);
            }
        }
    }
}
