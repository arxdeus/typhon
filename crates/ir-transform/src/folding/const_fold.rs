
use typhon_ir::magic;
use typhon_ir::pass::{Pass, PassContext, run_operator_pass};
use typhon_ir::rewrite::{RewriteRule, Rewriter};
use typhon_ir::util::cloning::CloneVisitor;
use typhon_ir::util::irtools::is_call_of;
use typhon_ir::util::operator::{Operator, OperatorState};
use typhon_ir::{Module, TypeId, ValueId};

use super::rule::*;

fn py_divmod(a: i64, b: i64) -> (i64, i64) {
    let mut d = a.wrapping_div(b);
    let mut m = a.wrapping_sub(d.wrapping_mul(b));
    if m != 0 && ((b ^ m) < 0) {
        m = m.wrapping_add(b);
        d = d.wrapping_sub(1);
    }
    (d, m)
}

fn int_pow(mut base: i64, mut exp: i64) -> i64 {
    if exp < 0 {
        return 0;
    }
    let mut result: i64 = 1;
    loop {
        if exp & 1 != 0 {
            result = result.wrapping_mul(base);
        }
        exp >>= 1;
        if exp == 0 {
            break;
        }
        base = base.wrapping_mul(base);
    }
    result
}

/// Mixed int/float binary rule (`IntFloatBinaryRule`).
struct IntFloatBinaryRule<O: FoldConst> {
    f: Box<dyn Fn(f64, f64) -> O>,
    magic: String,
    out: TypeId,
    exclude_rhs_zero: bool,
}

impl<O: FoldConst> RewriteRule for IntFloatBinaryRule<O> {
    fn magic(&self) -> Option<&str> {
        Some(&self.magic)
    }
    fn apply(&mut self, m: &mut Module, v: ValueId) -> Option<ValueId> {
        if !is_call_of(m, v, &self.magic, 2, None, true) {
            return None;
        }
        let args = call_args(m, v);
        let (l, r) = (args[0], args[1]);
        if let (Some(a), Some(b)) = (m.as_float_const(l).map(|c| c.val), m.as_int_const(r).map(|c| c.val)) {
            if self.exclude_rhs_zero && b == 0 {
                return None;
            }
            let out = (self.f)(a, b as f64);
            return Some(O::make(m, v, out, self.out));
        }
        if let (Some(a), Some(b)) = (m.as_int_const(l).map(|c| c.val), m.as_float_const(r).map(|c| c.val)) {
            if self.exclude_rhs_zero && b == 0.0 {
                return None;
            }
            let out = (self.f)(a as f64, b);
            return Some(O::make(m, v, out, self.out));
        }
        None
    }
}

fn id_val() -> Calculator {
    Box::new(|m: &mut Module, v: ValueId| CloneVisitor::new(true).clone_value(m, v))
}

const CDIV_METHOD_NAME: &str = "cdiv";
const CMOD_METHOD_NAME: &str = "cmod";

/// Constant folding pass (`FoldingPass`).
pub struct FoldingPass {
    state: OperatorState,
    rewriter: Rewriter,
}

impl Default for FoldingPass {
    fn default() -> Self {
        Self::new()
    }
}

impl FoldingPass {
    pub const KEY: &'static str = "core-folding-const-fold";

    pub fn new() -> Self {
        Self { state: OperatorState::new(true), rewriter: Rewriter::new() }
    }

    /// `getNumReplacements()`.
    pub fn num_replacements(&self) -> i32 {
        self.rewriter.num_replacements()
    }

    fn reg(&mut self, key: &str, r: impl RewriteRule + 'static) {
        self.rewriter.register_rule(key, Box::new(r));
    }

    fn register_standard_rules(&mut self, m: &mut Module, pynum: bool) {
        use Kind::*;
        use magic as M;
        let int = m.get_int_type();
        let float = m.get_float_type();
        let boolean = m.get_bool_type();

        // binary, single constant, int->int
        self.reg("int-multiply-by-zero", SingleConstantCommutativeRule::<i64>::new(0, 0, M::MUL, Commutative, int));
        self.reg("int-multiply-by-one", SingleConstantCommutativeRule::<i64>::with_calc(1, id_val(), M::MUL, Commutative, int));
        self.reg("int-subtract-zero", SingleConstantCommutativeRule::<i64>::with_calc(0, id_val(), M::SUB, Right, int));
        self.reg("int-add-zero", SingleConstantCommutativeRule::<i64>::with_calc(0, id_val(), M::ADD, Commutative, int));
        self.reg("int-floor-div-by-one", SingleConstantCommutativeRule::<i64>::with_calc(1, id_val(), M::FLOOR_DIV, Right, int));
        self.reg("int-floor-cdiv-by-one", SingleConstantCommutativeRule::<i64>::with_calc(1, id_val(), CDIV_METHOD_NAME, Right, int));
        self.reg("int-pos", NoOpRule::new(M::POS, int));
        self.reg("int-double-neg", DoubleApplicationNoOpRule::new(M::NEG, int));
        self.reg("int-double-inv", DoubleApplicationNoOpRule::new(M::INVERT, int));

        // Note: the reference integer overflow (UB) is folded with wrapping semantics;
        // shift counts are masked to 0..63.
        let ii = |f: fn(i64, i64) -> i64, mg: &str| DoubleConstantBinaryRule::<i64, i64>::new(f, mg, int, int);
        self.reg("int-constant-addition", ii(i64::wrapping_add, M::ADD));
        self.reg("int-constant-subtraction", ii(i64::wrapping_sub, M::SUB));
        if pynum {
            self.reg("int-constant-floor-div", ii(|x, y| py_divmod(x, y).0, M::FLOOR_DIV).exclude_rhs_zero(0));
        } else {
            self.reg("int-constant-floor-div", ii(i64::wrapping_div, M::FLOOR_DIV).exclude_rhs_zero(0));
        }
        self.reg("int-constant-floor-cdiv", ii(i64::wrapping_div, CDIV_METHOD_NAME).exclude_rhs_zero(0));
        self.reg("int-constant-mul", ii(i64::wrapping_mul, M::MUL));
        self.reg("int-constant-lshift", ii(|x, y| x.wrapping_shl(y as u32), M::LSHIFT));
        self.reg("int-constant-rshift", ii(|x, y| x.wrapping_shr(y as u32), M::RSHIFT));
        self.reg("int-constant-pow", ii(int_pow, M::POW));
        self.reg("int-constant-xor", ii(|x, y| x ^ y, M::XOR));
        self.reg("int-constant-or", ii(|x, y| x | y, M::OR));
        self.reg("int-constant-and", ii(|x, y| x & y, M::AND));
        if pynum {
            self.reg("int-constant-mod", ii(|x, y| py_divmod(x, y).1, M::MOD).exclude_rhs_zero(0));
        } else {
            self.reg("int-constant-mod", ii(i64::wrapping_rem, M::MOD).exclude_rhs_zero(0));
        }
        self.reg("int-constant-cmod", ii(i64::wrapping_rem, CMOD_METHOD_NAME).exclude_rhs_zero(0));

        // binary, double constant, int->bool
        let ib = |f: fn(&i64, &i64) -> bool, mg: &str| DoubleConstantBinaryRule::<i64, bool>::new(move |x, y| f(&x, &y), mg, int, boolean);
        self.reg("int-constant-eq", ib(i64::eq, M::EQ));
        self.reg("int-constant-ne", ib(i64::ne, M::NE));
        self.reg("int-constant-gt", ib(i64::gt, M::GT));
        self.reg("int-constant-ge", ib(i64::ge, M::GE));
        self.reg("int-constant-lt", ib(i64::lt, M::LT));
        self.reg("int-constant-le", ib(i64::le, M::LE));

        // binary, double constant, bool->bool
        let bb = |f: fn(bool, bool) -> bool, mg: &str| DoubleConstantBinaryRule::<bool, bool>::new(f, mg, boolean, boolean);
        self.reg("bool-constant-xor", bb(|x, y| x ^ y, M::XOR));
        self.reg("bool-constant-or", bb(|x, y| x | y, M::OR));
        self.reg("bool-constant-and", bb(|x, y| x & y, M::AND));

        // unary, single constant, int->int
        self.reg("int-constant-pos", SingleConstantUnaryRule::<i64, i64>::new(|x| x, M::POS, int, int));
        self.reg("int-constant-neg", SingleConstantUnaryRule::<i64, i64>::new(i64::wrapping_neg, M::NEG, int, int));
        self.reg("int-constant-inv", SingleConstantUnaryRule::<i64, i64>::new(|x| !x, M::INVERT, int, int));

        // unary, single constant, float->float
        self.reg("float-constant-pos", SingleConstantUnaryRule::<f64, f64>::new(|x| x, M::POS, float, float));
        self.reg("float-constant-neg", SingleConstantUnaryRule::<f64, f64>::new(|x| -x, M::NEG, float, float));

        // unary, single constant, bool->bool
        self.reg("bool-constant-inv", SingleConstantUnaryRule::<bool, bool>::new(|x| !x, M::INVERT, boolean, boolean));

        // binary, double constant, float->float
        let ff = |f: fn(f64, f64) -> f64, mg: &str| DoubleConstantBinaryRule::<f64, f64>::new(f, mg, float, float);
        self.reg("float-constant-addition", ff(|x, y| x + y, M::ADD));
        self.reg("float-constant-subtraction", ff(|x, y| x - y, M::SUB));
        if pynum {
            self.reg("float-constant-floor-div", ff(|x, y| x / y, M::TRUE_DIV).exclude_rhs_zero(0.0));
        } else {
            self.reg("float-constant-floor-div", ff(|x, y| x / y, M::TRUE_DIV));
        }
        self.reg("float-constant-cdiv", ff(|x, y| x / y, CDIV_METHOD_NAME));
        self.reg("float-constant-mul", ff(|x, y| x * y, M::MUL));
        self.reg("float-constant-pow", ff(f64::powf, M::POW));

        // binary, double constant, float->bool
        let fb = |f: fn(&f64, &f64) -> bool, mg: &str| DoubleConstantBinaryRule::<f64, bool>::new(move |x, y| f(&x, &y), mg, float, boolean);
        self.reg("float-constant-eq", fb(f64::eq, M::EQ));
        self.reg("float-constant-ne", fb(f64::ne, M::NE));
        self.reg("float-constant-gt", fb(f64::gt, M::GT));
        self.reg("float-constant-ge", fb(f64::ge, M::GE));
        self.reg("float-constant-lt", fb(f64::lt, M::LT));
        self.reg("float-constant-le", fb(f64::le, M::LE));

        // binary, double constant, int,float->float
        let iff = |f: fn(f64, f64) -> f64, mg: &str, ex: bool| IntFloatBinaryRule::<f64> { f: Box::new(f), magic: mg.into(), out: float, exclude_rhs_zero: ex };
        self.reg("int-float-constant-addition", iff(|x, y| x + y, M::ADD, false));
        self.reg("int-float-constant-subtraction", iff(|x, y| x - y, M::SUB, false));
        self.reg("int-float-constant-true-div", iff(|x, y| x / y, M::TRUE_DIV, pynum));
        self.reg("int-float-constant-mul", iff(|x, y| x * y, M::MUL, false));

        // binary, double constant, int,float->bool
        let ifb = |f: fn(&f64, &f64) -> bool, mg: &str| IntFloatBinaryRule::<bool> { f: Box::new(move |x, y| f(&x, &y)), magic: mg.into(), out: boolean, exclude_rhs_zero: false };
        self.reg("int-float-constant-eq", ifb(f64::eq, M::EQ));
        self.reg("int-float-constant-ne", ifb(f64::ne, M::NE));
        self.reg("int-float-constant-gt", ifb(f64::gt, M::GT));
        self.reg("int-float-constant-ge", ifb(f64::ge, M::GE));
        self.reg("int-float-constant-lt", ifb(f64::lt, M::LT));
        self.reg("int-float-constant-le", ifb(f64::le, M::LE));

        // type conversions, identity
        self.reg("int-constant-int", UnaryRule::new(id_val(), M::INT, int));
        self.reg("float-constant-float", UnaryRule::new(id_val(), M::FLOAT, float));
        self.reg("bool-constant-bool", UnaryRule::new(id_val(), M::BOOL, boolean));

        // type conversions, distinct
        self.reg("float-constant-int", SingleConstantUnaryRule::<f64, i64>::new(|x| x as i64, M::INT, float, int));
        self.reg("bool-constant-int", SingleConstantUnaryRule::<bool, i64>::new(|x| x as i64, M::INT, boolean, int));
        self.reg("int-constant-float", SingleConstantUnaryRule::<i64, f64>::new(|x| x as f64, M::FLOAT, int, float));
        self.reg("bool-constant-float", SingleConstantUnaryRule::<bool, f64>::new(|x| x as i64 as f64, M::FLOAT, boolean, float));
        self.reg("int-constant-bool", SingleConstantUnaryRule::<i64, bool>::new(|x| x != 0, M::BOOL, int, boolean));
        self.reg("float-constant-bool", SingleConstantUnaryRule::<f64, bool>::new(|x| x != 0.0, M::BOOL, float, boolean));
    }
}

impl Operator for FoldingPass {
    fn state(&self) -> &OperatorState {
        &self.state
    }
    fn state_mut(&mut self) -> &mut OperatorState {
        &mut self.state
    }
    fn handle_call_instr(&mut self, m: &mut Module, v: ValueId) {
        self.rewriter.rewrite(m, v);
    }
}

impl Pass for FoldingPass {
    fn key(&self) -> String {
        Self::KEY.to_string()
    }
    fn run(&mut self, m: &mut Module, ctx: &dyn PassContext) {
        // Note: the reference dereferences `getOptions()`; without a manager the default
        // (`pynum = false`) is used.
        let pynum = ctx.options().map(|o| o.pynum).unwrap_or(false);
        self.register_standard_rules(m, pynum);
        self.rewriter.reset();
        run_operator_pass(self, m);
    }
}
