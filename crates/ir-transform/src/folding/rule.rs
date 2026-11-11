//! Port of codon/cir/transform/folding/rule.h
//!
//! Generic rewrite rules used by the constant folder. C++ templates over the
//! constant type become generics over [`FoldConst`].

use typhon_ir::rewrite::RewriteRule;
use typhon_ir::util::irtools::{ConstValue, is_call_of_types};
use typhon_ir::{Module, TypeId, ValueId};

/// A constant payload that can be both read from and written to the IR
/// (`TemplatedConst<T>`).
pub trait FoldConst: ConstValue + Copy + 'static {
    /// `N<TemplatedConst<T>>(src, val, type)`.
    fn make(m: &mut Module, src: ValueId, val: Self, ty: TypeId) -> ValueId;
}

/// Copies the source info of `src` onto `v` (C++ `N<T>(src->getSrcInfo(), ...)`).
pub(crate) fn with_src(m: &mut Module, v: ValueId, src: ValueId) -> ValueId {
    let s = m.src_info(src.into());
    m.set_src_info(v.into(), s);
    v
}

impl FoldConst for i64 {
    fn make(m: &mut Module, src: ValueId, val: Self, ty: TypeId) -> ValueId {
        let v = m.nr_int_const(val, ty);
        let s = m.src_info(src.into());
        m.set_src_info(v.into(), s);
        v
    }
}
impl FoldConst for f64 {
    fn make(m: &mut Module, src: ValueId, val: Self, ty: TypeId) -> ValueId {
        let v = m.nr_float_const(val, ty);
        let s = m.src_info(src.into());
        m.set_src_info(v.into(), s);
        v
    }
}
impl FoldConst for bool {
    fn make(m: &mut Module, src: ValueId, val: Self, ty: TypeId) -> ValueId {
        let v = m.nr_bool_const(val, ty);
        let s = m.src_info(src.into());
        m.set_src_info(v.into(), s);
        v
    }
}

/// Call arguments of `v` (must be a call).
pub(crate) fn call_args(m: &Module, v: ValueId) -> Vec<ValueId> {
    m.as_call_instr(v).map(|c| c.args.clone()).unwrap_or_default()
}

/// Calculator of [`SingleConstantCommutativeRule`]: maps the other operand to a result.
pub type Calculator = Box<dyn FnMut(&mut Module, ValueId) -> ValueId>;

/// Left, right or commutative.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Left,
    Right,
    Commutative,
}

/// Commutative, binary rule that requires a single constant.
pub struct SingleConstantCommutativeRule<C: FoldConst> {
    val: C,
    ty: TypeId,
    magic: String,
    calc: Calculator,
    kind: Kind,
}

impl<C: FoldConst> SingleConstantCommutativeRule<C> {
    /// Rule whose result is the constant `new_val` (operand still evaluated).
    pub fn new(val: C, new_val: C, magic: impl Into<String>, kind: Kind, ty: TypeId) -> Self {
        let calc: Calculator = Box::new(move |m: &mut Module, value: ValueId| {
            let result = C::make(m, value, new_val, ty);
            let flow = m.nr_series_flow();
            with_src(m, flow, value);
            m.series_push(flow, value);
            let fi = m.nr_flow_instr(flow, result);
            with_src(m, fi, value)
        });
        Self { val, ty, magic: magic.into(), calc, kind }
    }
    /// Rule with a custom calculator.
    pub fn with_calc(val: C, calc: Calculator, magic: impl Into<String>, kind: Kind, ty: TypeId) -> Self {
        Self { val, ty, magic: magic.into(), calc, kind }
    }
}

impl<C: FoldConst> RewriteRule for SingleConstantCommutativeRule<C> {
    fn apply(&mut self, m: &mut Module, v: ValueId) -> Option<ValueId> {
        let t = Some(self.ty);
        if !is_call_of_types(m, v, &self.magic, &[t, t], t, true) {
            return None;
        }
        let args = call_args(m, v);
        let (left, right) = (args[0], args[1]);
        let lc = C::get(m, left);
        let rc = C::get(m, right);
        if matches!(self.kind, Kind::Commutative | Kind::Left) && lc == Some(self.val) {
            return Some((self.calc)(m, right));
        }
        if matches!(self.kind, Kind::Commutative | Kind::Right) && rc == Some(self.val) {
            return Some((self.calc)(m, left));
        }
        None
    }
}

/// Binary rule that requires two constants.
pub struct DoubleConstantBinaryRule<C: FoldConst, O: FoldConst> {
    f: Box<dyn Fn(C, C) -> O>,
    input_type: TypeId,
    result_type: TypeId,
    magic: String,
    /// `DoubleConstantBinaryRuleExcludeRHSZero` (const_fold.cpp).
    exclude_rhs_zero: Option<C>,
}

impl<C: FoldConst, O: FoldConst> DoubleConstantBinaryRule<C, O> {
    pub fn new(f: impl Fn(C, C) -> O + 'static, magic: impl Into<String>, input_type: TypeId, result_type: TypeId) -> Self {
        Self { f: Box::new(f), input_type, result_type, magic: magic.into(), exclude_rhs_zero: None }
    }
    /// `DoubleConstantBinaryRuleExcludeRHSZero`: skip when the RHS equals `zero`
    /// (`ConstantType()`).
    pub fn exclude_rhs_zero(mut self, zero: C) -> Self {
        self.exclude_rhs_zero = Some(zero);
        self
    }
}

impl<C: FoldConst, O: FoldConst> RewriteRule for DoubleConstantBinaryRule<C, O> {
    fn apply(&mut self, m: &mut Module, v: ValueId) -> Option<ValueId> {
        if let Some(zero) = self.exclude_rhs_zero {
            let args = call_args(m, v);
            if args.len() == 2 && C::get(m, args[1]) == Some(zero) {
                return None;
            }
        }
        let it = Some(self.input_type);
        if !is_call_of_types(m, v, &self.magic, &[it, it], Some(self.result_type), true) {
            return None;
        }
        let args = call_args(m, v);
        let l = C::get(m, args[0])?;
        let r = C::get(m, args[1])?;
        let out = (self.f)(l, r);
        Some(O::make(m, v, out, self.result_type))
    }
}

/// Unary rule that requires one constant.
pub struct SingleConstantUnaryRule<C: FoldConst, O: FoldConst> {
    f: Box<dyn Fn(C) -> O>,
    input_type: TypeId,
    result_type: TypeId,
    magic: String,
}

impl<C: FoldConst, O: FoldConst> SingleConstantUnaryRule<C, O> {
    pub fn new(f: impl Fn(C) -> O + 'static, magic: impl Into<String>, input_type: TypeId, result_type: TypeId) -> Self {
        Self { f: Box::new(f), input_type, result_type, magic: magic.into() }
    }
}

impl<C: FoldConst, O: FoldConst> RewriteRule for SingleConstantUnaryRule<C, O> {
    fn apply(&mut self, m: &mut Module, v: ValueId) -> Option<ValueId> {
        if !is_call_of_types(m, v, &self.magic, &[Some(self.input_type)], Some(self.result_type), true) {
            return None;
        }
        let args = call_args(m, v);
        let a = C::get(m, args[0])?;
        let out = (self.f)(a);
        Some(O::make(m, v, out, self.result_type))
    }
}

/// Unary rule that requires no constant.
pub struct UnaryRule {
    f: Calculator,
    input_type: TypeId,
    magic: String,
}

impl UnaryRule {
    pub fn new(f: Calculator, magic: impl Into<String>, input_type: TypeId) -> Self {
        Self { f, input_type, magic: magic.into() }
    }
}

impl RewriteRule for UnaryRule {
    fn apply(&mut self, m: &mut Module, v: ValueId) -> Option<ValueId> {
        let t = Some(self.input_type);
        if !is_call_of_types(m, v, &self.magic, &[t], t, true) {
            return None;
        }
        let arg = call_args(m, v)[0];
        Some((self.f)(m, arg))
    }
}

/// Rule that eliminates an operation, like "+x".
pub struct NoOpRule {
    input_type: TypeId,
    magic: String,
}

impl NoOpRule {
    pub fn new(magic: impl Into<String>, input_type: TypeId) -> Self {
        Self { input_type, magic: magic.into() }
    }
}

impl RewriteRule for NoOpRule {
    fn apply(&mut self, m: &mut Module, v: ValueId) -> Option<ValueId> {
        let t = Some(self.input_type);
        if !is_call_of_types(m, v, &self.magic, &[t], t, true) {
            return None;
        }
        Some(call_args(m, v)[0])
    }
}

/// Rule that eliminates a double-application of an operation, like "-(-x)".
pub struct DoubleApplicationNoOpRule {
    input_type: TypeId,
    magic: String,
}

impl DoubleApplicationNoOpRule {
    pub fn new(magic: impl Into<String>, input_type: TypeId) -> Self {
        Self { input_type, magic: magic.into() }
    }
}

impl RewriteRule for DoubleApplicationNoOpRule {
    fn apply(&mut self, m: &mut Module, v: ValueId) -> Option<ValueId> {
        let t = Some(self.input_type);
        if !is_call_of_types(m, v, &self.magic, &[t], t, true) {
            return None;
        }
        let arg = call_args(m, v)[0];
        if !is_call_of_types(m, arg, &self.magic, &[t], t, true) {
            return None;
        }
        Some(call_args(m, arg)[0])
    }
}
