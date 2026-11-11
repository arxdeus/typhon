//! Port of codon/parser/visitors/typecheck/op.cpp

use typhon_common::mangle::{self, stdlib};
use typhon_common::strings::{utf8_strlen, utf8_substr};
use typhon_common::{ErrorKind, Result};

use super::TypecheckVisitor;
use crate::ast::{
    Attr, AssignExpr, AssignStmt, BinaryExpr, BoolExpr, CallArg, EllipsisExpr, EllipsisType, ExprStmt, ExprTag,
    IfExpr, InstantiateExpr, IntExpr, Pipe, StmtExpr, StringExpr, TupleExpr, UnaryExpr, UpdateMode,
};
use crate::cache::fn_optional_unwrap;
use crate::ids::{ExprId, StmtId, TypeId};
use crate::types::LiteralKind;

/// Static operators allowed on a static kind (unary).
fn static_unary_op(k: LiteralKind, op: &str) -> bool {
    match k {
        LiteralKind::Int => matches!(op, "-" | "+" | "!" | "~"),
        LiteralKind::String | LiteralKind::Bool => op == "!",
        LiteralKind::Runtime => false,
    }
}

/// Static operators allowed on a static kind (binary).
fn static_binary_op(k: LiteralKind, op: &str) -> bool {
    match k {
        LiteralKind::Int => matches!(
            op,
            "<" | "<=" | ">" | ">=" | "==" | "!=" | "&&" | "||" | "+" | "-" | "*" | "//" | "%" | "&" | "|" | "^" | ">>" | "<<"
        ),
        LiteralKind::String => matches!(op, "==" | "!=" | "+"),
        LiteralKind::Bool => matches!(op, "<" | "<=" | ">" | ">=" | "==" | "!=" | "&&" | "||"),
        LiteralKind::Runtime => false,
    }
}

fn is_cmp(op: &str) -> bool {
    matches!(op, "==" | "!=" | "<" | "<=" | ">" | ">=")
}

/// Division and modulus implementations (`divMod`). `None` on division by zero.
pub fn div_mod(pynum: bool, a: i64, b: i64) -> Option<(i64, i64)> {
    if b == 0 {
        None
    } else if pynum {
        // Use Python implementation.
        let mut d = a.wrapping_div(b);
        let mut m = a.wrapping_sub(d.wrapping_mul(b));
        if m != 0 && ((b ^ m) < 0) {
            m += b;
            d -= 1;
        }
        Some((d, m))
    } else {
        // Use C implementation.
        Some((a.wrapping_div(b), a.wrapping_rem(b)))
    }
}

/// Follow Python indexing rules for static tuple indices (`translateIndex`); `Err(())`
/// if out of bounds (and not clamped).
pub fn translate_index_raw(mut idx: i64, len: i64, clamp: bool) -> std::result::Result<i64, i64> {
    if idx < 0 {
        idx += len;
    }
    if clamp {
        if idx < 0 {
            idx = 0;
        }
        if idx > len {
            idx = len;
        }
    } else if idx < 0 || idx >= len {
        return Err(idx);
    }
    Ok(idx)
}

/// Python slice index adjustment (`sliceAdjustIndices`, step must be non-zero).
pub fn slice_adjust_indices_raw(length: i64, start: &mut i64, stop: &mut i64, step: i64) -> i64 {
    if *start < 0 {
        *start += length;
        if *start < 0 {
            *start = if step < 0 { -1 } else { 0 };
        }
    } else if *start >= length {
        *start = if step < 0 { length - 1 } else { length };
    }
    if *stop < 0 {
        *stop += length;
        if *stop < 0 {
            *stop = if step < 0 { -1 } else { 0 };
        }
    } else if *stop >= length {
        *stop = if step < 0 { length - 1 } else { length };
    }
    if step < 0 {
        if *stop < *start {
            return (*start - *stop - 1) / (-step) + 1;
        }
    } else if *start < *stop {
        return (*stop - *start - 1) / step + 1;
    }
    0
}

/// Evaluate a static integer binary operator (`evaluateStaticBinary` core).
/// `None` on static division by zero.
pub fn eval_static_int_binary(op: &str, l: i64, r: i64, pynum: bool) -> Option<i64> {
    Some(match op {
        "<" => (l < r) as i64,
        "<=" => (l <= r) as i64,
        ">" => (l > r) as i64,
        ">=" => (l >= r) as i64,
        "==" => (l == r) as i64,
        "!=" => (l != r) as i64,
        "&&" => {
            if l != 0 { r } else { l }
        }
        "||" => {
            if l != 0 { l } else { r }
        }
        "+" => l.wrapping_add(r),
        "-" => l.wrapping_sub(r),
        "*" => l.wrapping_mul(r),
        "^" => l ^ r,
        "&" => l & r,
        "|" => l | r,
        ">>" => l.wrapping_shr(r as u32),
        "<<" => l.wrapping_shl(r as u32),
        "//" => div_mod(pynum, l, r)?.0,
        "%" => div_mod(pynum, l, r)?.1,
        _ => panic!("unknown static operator {op}"),
    })
}

impl<'a> TypecheckVisitor<'a> {
    fn op_bin(&self, e: ExprId) -> BinaryExpr {
        self.cache.ast[e].kind.as_binary().expect("not a binary").clone()
    }
    fn op_new_bin(&mut self, l: ExprId, op: &str, r: ExprId, in_place: bool) -> ExprId {
        self.n_expr(BinaryExpr { op: op.into(), lexpr: Some(l), rexpr: Some(r), in_place })
    }
    fn op_expr_stmt(&mut self, e: ExprId) -> StmtId {
        self.n_stmt(ExprStmt { expr: Some(e) })
    }
    fn op_stmt_expr(&mut self, items: Vec<StmtId>, e: ExprId) -> ExprId {
        self.n_expr(StmtExpr { items, expr: Some(e) })
    }
    fn op_is_expected(&self, e: ExprId, name: &str) -> bool {
        self.cache.ast[e].expected_type.is_some_and(|t| self.cache.types.is(t, name))
    }
    /// `expr->getType()->getUnbound()->staticKind = k`.
    fn op_set_unbound_static(&self, e: ExprId, k: LiteralKind) {
        let types = self.types();
        if let Some(u) = self.expr_type(e).and_then(|t| types.get_unbound(t)) {
            types.with_link_mut(u, |l| l.static_kind = k);
        }
    }
    fn pynum(&self) -> bool {
        self.cache.options.as_ref().map(|o| o.pynum).unwrap_or_else(|| typhon_common::Options::default().pynum)
    }

    /// `visit(UnaryExpr*)`: replace unary operators with the appropriate magic calls.
    /// Also evaluate static expressions. See [`Self::evaluate_static_unary`].
    pub fn visit_unary(&mut self, e: ExprId) -> Result<()> {
        let ue = self.cache.ast[e].kind.as_unary().unwrap().clone();
        let x = self.transform(ue.expr)?.unwrap();
        self.cache.ast[e].kind.as_unary_mut().unwrap().expr = Some(x);
        let op = ue.op;

        if let Some(ie) = self.cache.ast[x].kind.as_int() {
            if op == "-" {
                // Special case: make - INT(val) same as INT(-val) to simplify IR and everything
                let v = ie.value();
                let n = self.n_expr(IntExpr::new(v.wrapping_neg()));
                self.result_expr = self.transform(Some(n))?;
                return Ok(());
            }
        }

        let types = self.types();
        let mut static_type = None;
        let s = types.static_kind(self.ety(x));
        if s != LiteralKind::Runtime {
            if static_unary_op(s, &op) {
                match self.evaluate_static_unary(e)? {
                    Some(r) => {
                        self.result_expr = Some(r);
                        static_type = types.get_static(self.ety(r));
                    }
                    None => return Ok(()),
                }
            }
        } else if self.is_unbound_expr(x) {
            return Ok(());
        }

        let r = if op == "!" {
            // `not expr` -> `expr.__bool__().__invert__()`
            let d = self.dot_n(x, "__bool__");
            let c = self.call_n(d, vec![]);
            let d = self.dot_n(c, "__invert__");
            self.call_n(d, vec![])
        } else {
            let magic = match op.as_str() {
                "~" => "invert",
                "+" => "pos",
                "-" => "neg",
                _ => panic!("invalid unary operator '{op}'"),
            };
            let d = self.dot_n(x, &format!("__{magic}__"));
            self.call_n(d, vec![])
        };
        self.result_expr = self.transform(Some(r))?;
        if let (Some(st), Some(r)) = (static_type, self.result_expr) {
            self.set_type(r, st);
        }
        Ok(())
    }

    /// `evaluateStaticUnary(expr)`: evaluate a static unary expression; `None` if it cannot
    /// be evaluated yet. Supported operators: (strings) not (ints) not, -, +
    pub fn evaluate_static_unary(&mut self, e: ExprId) -> Result<Option<ExprId>> {
        let ue = self.cache.ast[e].kind.as_unary().unwrap().clone();
        let x = ue.expr.unwrap();
        let types = self.types();
        let xt = self.ety(x);
        let op = ue.op.as_str();

        // Case: static strings
        if types.static_kind(xt) == LiteralKind::String {
            if op == "!" {
                if types.can_realize(xt) {
                    let value = self.get_str_literal(xt, 0).is_empty();
                    typhon_common::log_flag!(TYPECHECK, "[cond::un] {:?}: {}", self.src, value);
                    let n = self.n_expr(IntExpr::new(value as i64));
                    return self.transform(Some(n));
                } else {
                    // Cannot be evaluated yet: just set the type
                    self.op_set_unbound_static(e, LiteralKind::Int);
                }
            }
            return Ok(None);
        }

        // Case: static bools
        if types.static_kind(xt) == LiteralKind::Bool {
            if op == "!" {
                if types.can_realize(xt) {
                    let value = self.get_bool_literal(xt, 0);
                    typhon_common::log_flag!(TYPECHECK, "[cond::un] {:?}: {}", self.src, value);
                    let n = self.n_expr(BoolExpr { value: !value });
                    return self.transform(Some(n));
                } else {
                    // Cannot be evaluated yet: just set the type
                    self.op_set_unbound_static(e, LiteralKind::Bool);
                }
            }
            return Ok(None);
        }

        // Case: static integers
        if matches!(op, "-" | "+" | "!" | "~") {
            if types.can_realize(xt) {
                let mut value = self.get_int_literal(xt, 0);
                match op {
                    "+" => {}
                    "-" => value = value.wrapping_neg(),
                    "~" => value = !value,
                    _ => value = (value == 0) as i64,
                }
                typhon_common::log_flag!(TYPECHECK, "[cond::un] {:?}: {}", self.src, value);
                let n = if op == "!" {
                    self.n_expr(BoolExpr { value: value != 0 })
                } else {
                    self.n_expr(IntExpr::new(value))
                };
                return self.transform(Some(n));
            } else {
                // Cannot be evaluated yet: just set the type
                self.op_set_unbound_static(e, if op == "!" { LiteralKind::Bool } else { LiteralKind::Int });
            }
        }
        Ok(None)
    }

    /// Static short-circuit of `&&` / `||` with a static lhs (part of `visit(BinaryExpr*)`).
    fn op_static_short_circuit(&mut self, e: ExprId, lhs: ExprId, rhs: ExprId, is_and: bool) -> Result<()> {
        let types = self.types();
        let lt = self.ety(lhs);
        let truth = if let Some(b) = types.get_bool_static(lt) {
            Some(b)
        } else if let Some(s) = types.get_str_static(lt) {
            Some(!s.is_empty())
        } else {
            types.get_int_static(lt).map(|i| i != 0)
        };
        let Some(truth) = truth else {
            self.op_set_unbound_static(e, LiteralKind::Bool);
            return Ok(());
        };
        let bool_expected = self.op_is_expected(e, &stdlib().bool_);
        // `&&`: short-circuits when false; `||`: when true
        if truth != is_and {
            if bool_expected {
                let b = self.n_expr(BoolExpr { value: truth });
                self.result_expr = self.transform(Some(b))?;
            } else {
                self.result_expr = Some(lhs);
            }
        } else {
            let s = self.op_expr_stmt(lhs);
            let se = self.op_stmt_expr(vec![s], rhs);
            self.result_expr = self.transform(Some(se))?;
        }
        Ok(())
    }
}

impl<'a> TypecheckVisitor<'a> {
    /// `visit(BinaryExpr*)`: replace binary operators with the appropriate magic calls.
    /// See `transformBinarySimple`, `transformBinaryIs`, `transformBinaryMagic` and
    /// `transformBinaryInplaceMagic`. Also evaluate static expressions.
    pub fn visit_binary(&mut self, e: ExprId) -> Result<()> {
        let st = stdlib();
        let types = self.types();
        let be = self.op_bin(e);
        let op = be.op.clone();
        if self.op_is_expected(e, &st.bool_) && (op == "&&" || op == "||") {
            let b = self.get_stdlib_type(&st.bool_)?;
            self.cache.ast[be.lexpr.unwrap()].expected_type = Some(b);
            self.cache.ast[be.rexpr.unwrap()].expected_type = Some(b);
        }

        let lhs = self.transform_allow(be.lexpr, true)?.unwrap();
        self.cache.ast[e].kind.as_binary_mut().unwrap().lexpr = Some(lhs);
        let rhs0 = be.rexpr.unwrap();

        // Static short-circuit
        let lk = types.static_kind(self.ety(lhs));
        if lk != LiteralKind::Runtime && op == "&&" {
            return self.op_static_short_circuit(e, lhs, rhs0, true);
        } else if lk != LiteralKind::Runtime && op == "||" {
            return self.op_static_short_circuit(e, lhs, rhs0, false);
        }

        let rhs = self.transform_allow(Some(rhs0), true)?.unwrap();
        self.cache.ast[e].kind.as_binary_mut().unwrap().rexpr = Some(rhs);

        let mut static_type = None;
        let (l, r) = (types.static_kind(self.ety(lhs)), types.static_kind(self.ety(rhs)));
        if l != LiteralKind::Runtime && r != LiteralKind::Runtime {
            let mut is_static = l == r && static_binary_op(l, &op);
            if !is_static
                && ((l == LiteralKind::Int && r == LiteralKind::Bool) || (r == LiteralKind::Int && l == LiteralKind::Bool))
                && static_binary_op(LiteralKind::Int, &op)
            {
                is_static = true;
            }
            if is_static {
                match self.evaluate_static_binary(e)? {
                    Some(x) => {
                        self.result_expr = Some(x);
                        static_type = types.get_static(self.ety(x));
                    }
                    None => return Ok(()),
                }
            }
        }

        let is_none = |s: &Self, x: ExprId| s.cache.ast[x].kind.tag() == ExprTag::None;
        if op == "|"
            && (Self::is_type_expr(self.cache, lhs) || is_none(self, lhs))
            && (Self::is_type_expr(self.cache, rhs) || is_none(self, rhs))
        {
            // Case: unions
            let u = self.id_n(&st.union);
            let i = self.n_expr(InstantiateExpr { expr: Some(u), items: vec![lhs, rhs] });
            self.result_expr = self.transform(Some(i))?;
        } else if let Some(x) = self.transform_binary_simple(e)? {
            // Case: simple binary expressions
            self.result_expr = Some(x);
        } else if types.get_unbound(self.ety(lhs)).is_some() || (op != "is" && types.get_unbound(self.ety(rhs)).is_some()) {
            // Case: types are unknown, so continue later
            return Ok(());
        } else if op == "is" {
            // Case: is operator
            self.result_expr = self.transform_binary_is(e)?;
        } else if let Some(ei) = self.transform_binary_inplace_magic(e, false)? {
            // Case: in-place magic methods
            self.result_expr = Some(ei);
        } else if let Some(em) = self.transform_binary_magic(e)? {
            // Case: normal magic methods
            self.result_expr = Some(em);
        } else if types.is(self.ety(lhs), &st.optional) {
            // Special case: handle optionals if everything else fails.
            // Assumes that optionals have no relevant magics (except for __eq__)
            let f = self.id_n(&fn_optional_unwrap());
            let c = self.call_n(f, vec![lhs]);
            let in_place = self.op_bin(e).in_place;
            let b = self.op_new_bin(c, &op, rhs, in_place);
            self.result_expr = self.transform(Some(b))?;
        } else {
            // Nothing found: report an error
            let src = self.cache.ast[e].src;
            let (pl, pr) = (types.pretty_string(self.ety(lhs)), types.pretty_string(self.ety(rhs)));
            typhon_common::bail_e!(ErrorKind::OP_NO_MAGIC, src, op, pl, pr);
        }

        if let (Some(stt), Some(r)) = (static_type, self.result_expr) {
            self.set_type(r, stt);
        }
        Ok(())
    }

    /// `evaluateStaticBinary(expr)`: evaluate a static binary expression; `None` if it
    /// cannot be evaluated yet.
    /// Supported operators: (strings) +, ==, !=
    ///                      (ints) <, <=, >, >=, ==, !=, and, or, +, -, *, //, %, ^, |, &
    pub fn evaluate_static_binary(&mut self, e: ExprId) -> Result<Option<ExprId>> {
        let types = self.types();
        let be = self.op_bin(e);
        let op = be.op.as_str();
        let (lt, rt) = (self.ety(be.lexpr.unwrap()), self.ety(be.rexpr.unwrap()));

        // Case: static strings
        if types.static_kind(rt) == LiteralKind::String {
            if op == "+" {
                // `"a" + "b"` -> `"ab"`
                if let (Some(_), Some(_)) = (types.get_str_static(lt), types.get_str_static(rt)) {
                    let value = self.get_str_literal(lt, 0) + &self.get_str_literal(rt, 0);
                    typhon_common::log_flag!(TYPECHECK, "[cond::bin] {:?}: {}", self.src, value);
                    let n = self.n_expr(StringExpr::new(value, ""));
                    return self.transform(Some(n));
                } else {
                    // Cannot be evaluated yet: just set the type
                    self.op_set_unbound_static(e, LiteralKind::String);
                }
            } else if let (Some(_), Some(_)) = (types.get_str_static(lt), types.get_str_static(rt)) {
                // `"a" == "b"` -> `False` (also handles `!=`)
                let eq = self.get_str_literal(lt, 0) == self.get_str_literal(rt, 0);
                let value = if op == "==" { eq } else { !eq };
                typhon_common::log_flag!(TYPECHECK, "[cond::bin] {:?}: {}", self.src, value);
                let n = self.n_expr(BoolExpr { value });
                return self.transform(Some(n));
            } else {
                // Cannot be evaluated yet: just set the type
                self.op_set_unbound_static(e, LiteralKind::Bool);
            }
            return Ok(None);
        }

        // Case: static integers
        // PORT-NOTE: C++ checks `getBoolStatic()` of the lhs twice (typo); kept as is.
        let lhs_bool = types.get_bool_static(lt).is_some();
        if types.get_static(lt).is_some() && types.get_static(rt).is_some() {
            let lvalue = if types.get_int_static(lt).is_some() { self.get_int_literal(lt, 0) } else { self.get_bool_literal(lt, 0) as i64 };
            let rvalue = if types.get_int_static(rt).is_some() { self.get_int_literal(rt, 0) } else { self.get_bool_literal(rt, 0) as i64 };
            let Some(value) = eval_static_int_binary(op, lvalue, rvalue, self.pynum()) else {
                let src = self.ctx.borrow().src_info();
                typhon_common::bail_e!(ErrorKind::STATIC_DIV_ZERO, src);
            };
            typhon_common::log_flag!(TYPECHECK, "[cond::bin] {:?}: {}", self.src, value);
            let n = if is_cmp(op) || ((op == "&&" || op == "||") && lhs_bool) {
                self.n_expr(BoolExpr { value: value != 0 })
            } else {
                self.n_expr(IntExpr::new(value))
            };
            return self.transform(Some(n));
        } else {
            // Cannot be evaluated yet: just set the type
            if is_cmp(op) || ((op == "&&" || op == "||") && lhs_bool) {
                self.op_set_unbound_static(e, LiteralKind::Bool);
            } else {
                self.op_set_unbound_static(e, LiteralKind::Int);
            }
        }
        Ok(None)
    }

    /// `transformBinarySimple(expr)`.
    /// `a and b` -> `b if a else False`, `a or b` -> `True if a else b`,
    /// `a in b` -> `a.__contains__(b)`, `a not in b` -> `not (a in b)`,
    /// `a is not b` -> `not (a is b)`.
    pub fn transform_binary_simple(&mut self, e: ExprId) -> Result<Option<ExprId>> {
        let st = stdlib();
        let be = self.op_bin(e);
        let (lhs, rhs) = (be.lexpr.unwrap(), be.rexpr.unwrap());
        let bool_expected = self.op_is_expected(e, &st.bool_);
        match be.op.as_str() {
            "&&" | "||" => {
                let is_and = be.op == "&&";
                let x = if bool_expected {
                    let d = self.dot_n(rhs, "__bool__");
                    let c = self.call_n(d, vec![]);
                    let b = self.n_expr(BoolExpr { value: !is_and });
                    let (i, el) = if is_and { (c, b) } else { (b, c) };
                    self.n_expr(IfExpr { cond: Some(lhs), ifexpr: Some(i), elsexpr: Some(el) })
                } else {
                    let vn = self.get_temporary_var("cond");
                    let v = self.id_n(&vn);
                    let ve = self.n_expr(AssignExpr { var: Some(v), expr: Some(lhs) });
                    let v2 = self.id_n(&vn);
                    let (i, el) = if is_and { (rhs, v2) } else { (v2, rhs) };
                    let x = self.n_expr(IfExpr { cond: Some(ve), ifexpr: Some(i), elsexpr: Some(el) });
                    let u = self.get_stdlib_type(&st.union)?;
                    self.cache.ast[x].expected_type = Some(u);
                    x
                };
                self.transform(Some(x))
            }
            "not in" => {
                let d = self.dot_n(rhs, "__contains__");
                let c = self.call_n(d, vec![lhs]);
                let d = self.dot_n(c, "__invert__");
                let c = self.call_n(d, vec![]);
                self.transform(Some(c))
            }
            "in" => {
                let d = self.dot_n(rhs, "__contains__");
                let c = self.call_n(d, vec![lhs]);
                self.transform(Some(c))
            }
            "is" => {
                let ln = self.cache.ast[lhs].kind.tag() == ExprTag::None;
                let rn = self.cache.ast[rhs].kind.tag() == ExprTag::None;
                if ln && rn {
                    let b = self.n_expr(BoolExpr { value: true });
                    self.transform(Some(b))
                } else if ln {
                    let b = self.op_new_bin(rhs, "is", lhs, false);
                    self.transform(Some(b))
                } else {
                    Ok(None)
                }
            }
            "is not" => {
                let b = self.op_new_bin(lhs, "is", rhs, false);
                let u = self.n_expr(UnaryExpr { op: "!".into(), expr: Some(b) });
                self.transform(Some(u))
            }
            _ => Ok(None),
        }
    }
}

impl<'a> TypecheckVisitor<'a> {
    /// `wrapSide` lambda of `transformBinaryIs`.
    fn op_wrap_side(&mut self, lhs: ExprId, rhs: ExprId, side_l: bool, side_r: bool, x: ExprId) -> Result<Option<ExprId>> {
        let r = if side_l && side_r {
            let a = self.op_expr_stmt(lhs);
            let b = self.op_expr_stmt(rhs);
            self.op_stmt_expr(vec![a, b], x)
        } else if side_l || side_r {
            let a = if side_l { self.op_expr_stmt(lhs) } else { self.op_expr_stmt(rhs) };
            self.op_stmt_expr(vec![a], x)
        } else {
            x
        };
        self.transform(Some(r))
    }

    /// `transformBinaryIs(expr)`: transform a binary `is` expression by checking for type
    /// equality. Handle special `is None` cases as well.
    pub fn transform_binary_is(&mut self, e: ExprId) -> Result<Option<ExprId>> {
        let st = stdlib();
        let types = self.types();
        let be = self.op_bin(e);
        typhon_common::seqassert!(be.op == "is", "not an is binary expression");
        let (lhs, rhs) = (be.lexpr.unwrap(), be.rexpr.unwrap());
        let (sl, sr) = (self.has_side_effect(lhs), self.has_side_effect(rhs));

        // Case: `is None` expressions
        if self.cache.ast[rhs].kind.tag() == ExprTag::None {
            let lc = self.extract_class_type_expr(lhs);
            if lc.is_some_and(|c| types.get_union(c).is_some()) && !Self::is_type_expr(self.cache, lhs) {
                let u = self.id_n(&st.union);
                let d = self.dot_n(u, "_isinstance");
                let n = self.id_n(&st.none_type);
                let c = self.call_n(d, vec![lhs, n]);
                return self.transform(Some(c));
            }
            if lc.is_some_and(|c| types.is(c, &st.none_type)) {
                let b = self.n_expr(BoolExpr { value: true });
                let w = self.op_wrap_side(lhs, rhs, sl, sr, b)?;
                return self.transform(w);
            }
            if !lc.is_some_and(|c| types.is(c, &st.optional)) {
                // lhs is not optional: `return False`
                let b = self.n_expr(BoolExpr { value: false });
                let w = self.op_wrap_side(lhs, rhs, sl, sr, b)?;
                return self.transform(w);
            } else {
                // Special case: Optional[Optional[... Optional[NoneType]]...] == NoneType
                let mut g = lc.unwrap();
                while types.is(self.extract_class_generic(g, 0), &st.optional) {
                    g = types.get_class(self.extract_class_generic(g, 0)).unwrap();
                }
                if types.get_class(self.extract_class_generic(g, 0)).is_none() {
                    let typ = self.instantiate_unbound();
                    types.with_link_mut(typ, |l| l.static_kind = LiteralKind::Bool);
                    let et = self.ety(e);
                    self.unify(et, typ)?;
                    return Ok(None);
                }
                if types.is(self.extract_class_generic(g, 0), &st.none_type) {
                    let b = self.n_expr(BoolExpr { value: true });
                    let w = self.op_wrap_side(lhs, rhs, sl, sr, b)?;
                    return self.transform(w);
                }

                // lhs is optional: `return lhs.__has__().__invert__()`
                let et = self.ety(e);
                if types.get_unbound(et).is_some() && types.static_kind(et) != LiteralKind::Runtime {
                    self.op_set_unbound_static(e, LiteralKind::Runtime);
                }
                let d = self.dot_n(lhs, "__has__");
                let c = self.call_n(d, vec![]);
                let d = self.dot_n(c, "__invert__");
                let c = self.call_n(d, vec![]);
                return self.transform(Some(c));
            }
        }

        // Check the type equality (operand types and __raw__ pointers must match).
        if Self::is_type_expr(self.cache, lhs) && Self::is_type_expr(self.cache, rhs) {
            let lt = self.extract_class_type_expr(lhs);
            let rt = self.extract_class_type_expr(rhs);
            let (Some(lt), Some(rt)) = (lt, rt) else { return Ok(None) };
            let same = types.class_name(lt) == types.class_name(rt) && types.unify(lt, rt, None, self)? >= 0;
            let b = self.n_expr(BoolExpr { value: same });
            let w = self.op_wrap_side(lhs, rhs, sl, sr, b)?;
            return self.transform(w);
        }
        let lc = self.realize(self.ety(lhs))?;
        let rc = self.realize(self.ety(rhs))?;
        let (Some(lc), Some(rc)) = (lc, rc) else {
            // Types not known: return early
            let et = self.ety(e);
            let b = self.get_stdlib_type(&st.bool_)?;
            self.unify(et, b)?;
            return Ok(None);
        };
        if !types.is_record(lc) && !types.is_record(rc) {
            // Both reference types: `return type._is(lhs, rhs)`
            let f = self.id_n(&mangle::mangled_method("", "type", "_is", 0, 0));
            let c = self.call_n(f, vec![lhs, rhs]);
            return self.transform(Some(c));
        }
        if types.is(lc, &st.optional) {
            // lhs is optional: `return lhs.__is_optional__(rhs)`
            let d = self.dot_n(lhs, "__is_optional__");
            let c = self.call_n(d, vec![rhs]);
            return self.transform(Some(c));
        }
        if types.is(rc, &st.optional) {
            // rhs is optional: `return rhs.__is_optional__(lhs)`
            let d = self.dot_n(rhs, "__is_optional__");
            let c = self.call_n(d, vec![lhs]);
            return self.transform(Some(c));
        }
        if types.realized_name(lc) != types.realized_name(rc) {
            // tuple names do not match: `return False`
            let b = self.n_expr(BoolExpr { value: false });
            let w = self.op_wrap_side(lhs, rhs, sl, sr, b)?;
            return self.transform(w);
        }
        // Same tuple types: `return lhs == rhs`
        let b = self.op_new_bin(lhs, "==", rhs, false);
        self.transform(Some(b))
    }

    /// `getMagic(op)`: `(magic, rightMagic)` for a binary operator.
    pub fn get_magic(&self, op: &str) -> (String, String) {
        get_magic_raw(op)
    }

    /// `transformBinaryInplaceMagic(expr, isAtomic)`: `a op= b` -> `a.__iopmagic__(b)`.
    /// If `is_atomic`, use atomic magics if available.
    pub fn transform_binary_inplace_magic(&mut self, e: ExprId, is_atomic: bool) -> Result<Option<ExprId>> {
        let st = stdlib();
        let be = self.op_bin(e);
        let (magic, _) = self.get_magic(&be.op);
        let (mut lhs, rhs) = (be.lexpr.unwrap(), be.rexpr.unwrap());
        let lt = self.expr_class(lhs).expect("lhs type not known");

        let mut method = None;
        // Atomic operations: check if `lhs.__atomic_op__(Ptr[lhs], rhs)` exists
        if is_atomic {
            let p = self.get_stdlib_type(&st.ptr)?;
            let ptr = self.instantiate_type_with(p, &[lt])?;
            let rt = self.ety(rhs);
            method = self.find_best_method_types(lt, &format!("__atomic_{magic}__"), &[ptr, rt])?;
            if method.is_some() {
                let f = self.id_n("__ptr__");
                lhs = self.call_n(f, vec![lhs]);
                self.cache.ast[e].kind.as_binary_mut().unwrap().lexpr = Some(lhs);
            }
        }
        // In-place operations: check if `lhs.__iop__(lhs, rhs)` exists
        if method.is_none() && be.in_place {
            method = self.find_best_method_exprs(lt, &format!("__i{magic}__"), &[lhs, rhs])?;
        }
        if let Some(m) = method {
            let f = self.id_n(&self.cache.types.func_name(m));
            let c = self.call_n(f, vec![lhs, rhs]);
            return self.transform(Some(c));
        }
        Ok(None)
    }

    /// `transformBinaryMagic(expr)`: `a op b` -> `a.__opmagic__(b)`.
    pub fn transform_binary_magic(&mut self, e: ExprId) -> Result<Option<ExprId>> {
        let st = stdlib();
        let types = self.types();
        let be = self.op_bin(e);
        let (magic, right_magic) = self.get_magic(&be.op);
        let (lhs, rhs) = (be.lexpr.unwrap(), be.rexpr.unwrap());
        let (lt, rt) = (self.ety(lhs), self.ety(rhs));

        // Resolving only the receiver can leave a union argument inside concrete arithmetic
        // (e.g. an array plus an int-or-complex scalar). Select both active types through
        // a helper that evaluates operands once, in source order, before testing their tags.
        if types.get_union(lt).is_some() || types.get_union(rt).is_some() {
            let f = self.id_n(&mangle::mangled_method("", "Union", "_binary", 0, 0));
            let s = self.n_expr(StringExpr::new(be.op.clone(), ""));
            let c = self.call_n(f, vec![lhs, rhs, s]);
            return self.transform(Some(c));
        }

        if !types.is(lt, "pyobj") && types.is(rt, "pyobj") {
            // Special case: `obj op pyobj` -> `rhs.__rmagic__(lhs)` on lhs
            // Assumes that pyobj implements all left and right magics
            let l = self.get_temporary_var("l");
            let r = self.get_temporary_var("r");
            let (li, ri) = (self.id_n(&l), self.id_n(&r));
            let a1 = self.n_stmt(AssignStmt { lhs: Some(li), rhs: Some(lhs), type_expr: None, update: UpdateMode::Assign });
            let a2 = self.n_stmt(AssignStmt { lhs: Some(ri), rhs: Some(rhs), type_expr: None, update: UpdateMode::Assign });
            let ri = self.id_n(&r);
            let d = self.dot_n(ri, &format!("__{right_magic}__"));
            let li = self.id_n(&l);
            let c = self.call_n(d, vec![li]);
            let se = self.op_stmt_expr(vec![a1, a2], c);
            return self.transform(Some(se));
        }
        // Normal operations: check if `lhs.__magic__(lhs, rhs)` exists
        if let Some(lc) = types.get_class(lt) {
            if let Some(m) = self.find_best_method_exprs(lc, &format!("__{magic}__"), &[lhs, rhs])? {
                // Normal case: `__magic__(lhs, rhs)`
                let f = self.id_n(&types.func_name(m));
                let c = self.call_n(f, vec![lhs, rhs]);
                return self.transform(Some(c));
            }
        }

        // Right-side magics: check if `rhs.__rmagic__(rhs, lhs)` exists
        if let Some(rc) = types.get_class(rt) {
            if let Some(m) = self.find_best_method_exprs(rc, &format!("__{right_magic}__"), &[rhs, lhs])? {
                let l = self.get_temporary_var("l");
                let r = self.get_temporary_var("r");
                let (li, ri) = (self.id_n(&l), self.id_n(&r));
                let a1 = self.n_stmt(AssignStmt { lhs: Some(li), rhs: Some(lhs), type_expr: None, update: UpdateMode::Assign });
                let a2 = self.n_stmt(AssignStmt { lhs: Some(ri), rhs: Some(rhs), type_expr: None, update: UpdateMode::Assign });
                let f = self.id_n(&types.func_name(m));
                let (ri, li) = (self.id_n(&r), self.id_n(&l));
                let c = self.call_n(f, vec![ri, li]);
                let se = self.op_stmt_expr(vec![a1, a2], c);
                return self.transform(Some(se));
            }
        }

        // Union candidate probing calls this helper directly, so Optional unwrapping
        // cannot live only in visit(BinaryExpr): nullable integer alternatives would
        // otherwise lose valid arithmetic. Try declared/reflected methods first to
        // preserve Optional's own operations, including None-aware equality.
        if types.is(lt, &st.optional) {
            let f = self.id_n(&fn_optional_unwrap());
            let c = self.call_n(f, vec![lhs]);
            let c = self.tr(c)?;
            let unwrapped = self.op_new_bin(c, &be.op, rhs, be.in_place);
            if let Some(x) = self.transform_binary_inplace_magic(unwrapped, false)? {
                return Ok(Some(x));
            }
            return self.transform_binary_magic(unwrapped);
        }
        Ok(None)
    }
}

/// `getMagic(op)` table lookup.
pub fn get_magic_raw(op: &str) -> (String, String) {
    let m = match op {
        "+" => "add",
        "-" => "sub",
        "*" => "mul",
        "**" => "pow",
        "/" => "truediv",
        "//" => "floordiv",
        "@" => "matmul",
        "%" => "mod",
        "<" => "lt",
        "<=" => "le",
        ">" => "gt",
        ">=" => "ge",
        "==" => "eq",
        "!=" => "ne",
        "<<" => "lshift",
        ">>" => "rshift",
        "&" => "and",
        "|" => "or",
        "^" => "xor",
        _ => panic!("invalid binary operator '{op}'"),
    };
    let rm = match op {
        "<" => Some("gt"),
        "<=" => Some("ge"),
        ">" => Some("lt"),
        ">=" => Some("le"),
        "==" => Some("eq"),
        "!=" => Some("ne"),
        _ => None,
    };
    (m.to_string(), rm.map(|s| s.to_string()).unwrap_or_else(|| format!("r{m}")))
}

impl<'a> TypecheckVisitor<'a> {
    /// `visit(ChainBinaryExpr*)`: `a <= b <= c` -> `(a <= (chain := b)) and (chain <= c)`.
    /// The assignment above ensures that all expressions are executed only once.
    pub fn visit_chain_binary(&mut self, e: ExprId) -> Result<()> {
        let st = stdlib();
        let exprs = self.cache.ast[e].kind.as_chain_binary().unwrap().exprs.clone();
        typhon_common::seqassert!(exprs.len() >= 2, "not enough expressions in ChainBinaryExpr");
        let is_bool = self.op_is_expected(e, &st.bool_);
        let bool_t = if is_bool { Some(self.get_stdlib_type(&st.bool_)?) } else { None };

        let mut items = vec![];
        let mut prev = String::new();
        for i in 1..exprs.len() {
            let l = if prev.is_empty() { self.cache.ast.clone_expr(exprs[i - 1].1, false) } else { self.id_n(&prev) };
            prev = {
                let ctx = self.ctx.clone();
                let c = ctx.borrow();
                c.generate_canonical_name(self.cache, "chain", false, false)
            };
            let r = if i + 1 == exprs.len() {
                self.cache.ast.clone_expr(exprs[i].1, false)
            } else {
                // Chain temporaries must retain a static operand's literal type.
                let p = self.id_n(&prev);
                let v = self.cache.ast.clone_expr(exprs[i].1, false);
                let a = self.n_stmt(AssignStmt { lhs: Some(p), rhs: Some(v), type_expr: None, update: UpdateMode::Assign });
                self.cache.ast[a].attrs.set(Attr::MaybeLiteral);
                let p = self.id_n(&prev);
                self.op_stmt_expr(vec![a], p)
            };
            let b = self.op_new_bin(l, &exprs[i].0, r, false);
            if is_bool {
                self.cache.ast[b].expected_type = bool_t;
            }
            items.push(b);
        }

        let mut fin = *items.last().unwrap();
        for i in (0..items.len() - 1).rev() {
            fin = self.op_new_bin(items[i], "&&", fin, false);
            if is_bool {
                self.cache.ast[fin].expected_type = bool_t;
            }
        }
        self.result_expr = self.transform(Some(fin))?;
        Ok(())
    }

    /// `findEllipsis(expr)`: locate the pipe ellipsis within (possibly nested) calls.
    /// Returns the calls and the argument positions needed to reach it, innermost first.
    /// `foo(bar(1, baz(...)))` returns `[{0, baz}, {1, bar}, {0, foo}]`.
    pub fn find_ellipsis(&self, e: ExprId) -> Vec<(usize, ExprId)> {
        let a = &self.cache.ast;
        let Some(call) = a[e].kind.as_call() else { return vec![] };
        for (ai, arg) in call.items.iter().enumerate() {
            let Some(v) = arg.value else { continue };
            if let Some(el) = a[v].kind.as_ellipsis() {
                if el.is_pipe() {
                    return vec![(ai, e)];
                }
            } else if a[v].kind.as_call().is_some() {
                let mut r = self.find_ellipsis(v);
                if !r.is_empty() {
                    r.push((ai, e));
                    return r;
                }
            }
        }
        vec![]
    }

    /// Follow `StmtExpr` layers of a pipe stage (`ec` pointer chasing in C++).
    /// Returns the innermost expression and the StmtExpr that holds it (if any).
    fn op_pipe_stage_head(&self, mut x: ExprId) -> (Option<ExprId>, ExprId) {
        let mut holder = None;
        while let Some(se) = self.cache.ast[x].kind.as_stmt() {
            holder = Some(x);
            x = se.expr.unwrap();
        }
        (holder, x)
    }
    fn op_pipe_set_stage(&mut self, e: ExprId, pi: usize, holder: Option<ExprId>, x: ExprId) {
        match holder {
            Some(h) => self.cache.ast[h].kind.as_stmt_mut().unwrap().expr = Some(x),
            None => self.cache.ast[e].kind.as_pipe_mut().unwrap().items[pi].expr = Some(x),
        }
    }

    /// `visit(PipeExpr*)`: each stage call `foo(x)` without an ellipsis becomes
    /// `foo(..., x)`; non-calls become `foo(...)`. Stages that expand to multiple stages are
    /// moved into the pipeline.
    pub fn visit_pipe(&mut self, e: ExprId) -> Result<()> {
        let st = stdlib();
        let types = self.types();
        let mut has_generator = false;
        // Return T if t is of type `Generator[T]`; otherwise just `type(t)`
        let iterable = |s: &Self, t: TypeId, hg: &mut bool| -> TypeId {
            if types.is(t, "Generator") {
                *hg = true;
                return s.extract_class_generic(t, 0);
            }
            t
        };

        self.cache.ast[e].kind.as_pipe_mut().unwrap().in_types.clear();
        // Process the pipeline head
        let front = self.cache.ast[e].kind.as_pipe().unwrap().items[0].expr;
        let front = self.transform(front)?.unwrap();
        self.cache.ast[e].kind.as_pipe_mut().unwrap().items[0].expr = Some(front);
        let mut in_type = self.ety(front); // input type to the next stage
        self.cache.ast[e].kind.as_pipe_mut().unwrap().in_types.push(Some(in_type));
        in_type = iterable(self, in_type, &mut has_generator);
        let mut done = self.cache.ast[front].done;

        let mut pi: usize = 1;
        while pi < self.cache.ast[e].kind.as_pipe().unwrap().items.len() {
            let stage = self.cache.ast[e].kind.as_pipe().unwrap().items[pi].expr.unwrap();
            let (mut holder, mut ec) = self.op_pipe_stage_head(stage);
            let in_type_pos;
            if let Some(call) = self.cache.ast[ec].kind.as_call() {
                // Case: a call. Find the position of the pipe ellipsis within it
                let pos = call.items.iter().position(|a| a.value.is_some_and(|v| self.cache.ast[v].kind.as_ellipsis().is_some()));
                match pos {
                    Some(p) => in_type_pos = p,
                    None => {
                        // No ellipses found? Prepend it as the first argument
                        let el = self.n_expr(EllipsisExpr { mode: EllipsisType::Partial });
                        self.cache.ast[ec].kind.as_call_mut().unwrap().items.insert(0, CallArg::new("", Some(el)));
                        in_type_pos = 0;
                    }
                }
            } else {
                // Case: not a call. Convert it to a call with a single ellipsis
                let el = self.n_expr(EllipsisExpr { mode: EllipsisType::Partial });
                let c = self.call_n(stage, vec![el]);
                self.cache.ast[e].kind.as_pipe_mut().unwrap().items[pi].expr = Some(c);
                holder = None;
                ec = c;
                in_type_pos = 0;
            }

            // Set the ellipsis type
            let el = self.cache.ast[ec].kind.as_call().unwrap().items[in_type_pos].value.unwrap();
            self.cache.ast[el].kind = crate::ast::ExprKind::Ellipsis(EllipsisExpr { mode: EllipsisType::Pipe });
            // Don't unify unbound inType yet (it might become a generator that needs to be
            // extracted)
            if self.cache.ast[el].ty.is_none() {
                let u = self.instantiate_unbound();
                self.set_type(el, u);
            }
            if types.get_unbound(in_type).is_none() {
                let et = self.ety(el);
                self.unify(et, in_type)?;
            }

            // Transform the call. Because a transformation might wrap the ellipsis in layers,
            // make sure to extract these layers and move them to the pipeline.
            // Example: `foo(...)` that is transformed to `foo(unwrap(...))` will become
            // `unwrap(...) |> foo(...)`
            ec = self.tr(ec)?;
            self.op_pipe_set_stage(e, pi, holder, ec);
            let layers = self.find_ellipsis(ec);
            typhon_common::seqassert!(!layers.is_empty(), "can't find the ellipsis");
            if layers.len() > 1 {
                // Prepend layers
                for (pos, prepend) in &layers {
                    let ne = self.n_expr(EllipsisExpr { mode: EllipsisType::Pipe });
                    self.cache.ast[*prepend].kind.as_call_mut().unwrap().items[*pos].value = Some(ne);
                    self.cache.ast[e].kind.as_pipe_mut().unwrap().items.insert(pi, Pipe { op: "|>".into(), expr: Some(*prepend) });
                    pi += 1;
                }
                // Rewind the loop (yes, the current expression will get transformed again)
                self.cache.ast[e].kind.as_pipe_mut().unwrap().items.remove(pi);
                pi = pi - layers.len() - 1;
                pi += 1;
                continue;
            }

            let stage_now = self.cache.ast[e].kind.as_pipe().unwrap().items[pi].expr.unwrap();
            if let Some(ect) = self.expr_type(ec) {
                if self.cache.ast[stage_now].ty.is_none() {
                    let u = self.instantiate_unbound();
                    self.set_type(stage_now, u);
                }
                let st_t = self.ety(stage_now);
                self.unify(st_t, ect)?;
            }
            self.cache.ast[e].kind.as_pipe_mut().unwrap().items[pi].expr = Some(ec);
            in_type = self.ety(ec);
            if self.realize(in_type)?.is_none() {
                done = false;
            }
            self.cache.ast[e].kind.as_pipe_mut().unwrap().in_types.push(Some(in_type));

            // Do not extract the generator in the last stage of a pipeline
            if pi + 1 < self.cache.ast[e].kind.as_pipe().unwrap().items.len() {
                in_type = iterable(self, in_type, &mut has_generator);
            }
            pi += 1;
        }
        let et = self.ety(e);
        let t = if has_generator { self.get_stdlib_type(&st.none_type)? } else { in_type };
        self.unify(et, t)?;
        if done {
            self.set_done_expr(e);
        }
        Ok(())
    }
}

impl<'a> TypecheckVisitor<'a> {
    /// `visit(IndexExpr*)`.
    /// `foo[T]` -> Instantiate(foo, [T]) if `foo` is a type; `tup[1]` -> `tup.item1` if
    /// `tup` is tuple; `foo[idx]` -> `foo.__getitem__(idx)`; expr.itemN or a sub-tuple if
    /// index is static (see `transformStaticTupleIndex`).
    pub fn visit_index(&mut self, e: ExprId) -> Result<()> {
        let st = stdlib();
        let types = self.types();
        let ie = self.cache.ast[e].kind.as_index().unwrap().clone();
        let a = &self.cache.ast;
        let is_lit = |x: Option<ExprId>| crate::ast::is_id(a, x, "Literal") || crate::ast::is_id(a, x, "Static");
        let lit_head = is_lit(ie.expr);
        if lit_head && ["int", "Int", "str", "bool"].iter().any(|s| crate::ast::is_id(a, ie.index, s)) {
            // Special case: static types.
            let typ = self.instantiate_unbound();
            let k = self.cache.ast.get_static_generic(Some(e));
            types.with_link_mut(typ, |l| l.static_kind = k);
            let et = self.ety(e);
            self.unify(et, typ)?;
            self.set_done_expr(e);
            return Ok(());
        } else if lit_head {
            let src = ie.index.map(|i| self.cache.ast[i].src).unwrap_or(self.src);
            typhon_common::bail_e!(ErrorKind::BAD_STATIC_TYPE, src);
        }
        if crate::ast::is_id(&self.cache.ast, ie.expr, "tuple") {
            let x = ie.expr.unwrap();
            self.cache.ast[x].kind.as_id_mut().unwrap().value = st.tuple.clone();
        }
        let head = self.transform_allow(ie.expr, true)?.unwrap();
        self.cache.ast[e].kind.as_index_mut().unwrap().expr = Some(head);
        let index = ie.index.unwrap();

        // IndexExpr[i1, ..., iN] is internally represented as
        // IndexExpr[TupleExpr[i1, ..., iN]] for N > 1
        let (mut items, is_tuple) = match self.cache.ast[index].kind.as_tuple() {
            Some(t) => (t.items.clone(), true),
            None => (vec![index], false),
        };
        let orig_index = self.cache.ast.clone_expr(index, false);
        for i in items.iter_mut() {
            if Self::is_type_expr(self.cache, head) {
                if let Some(li) = self.cache.ast[*i].kind.as_list() {
                    // Special case: `A[[A, B], C]` -> `A[Tuple[A, B], C]` (e.g., in
                    // `Function[...]`)
                    let li = li.items.clone();
                    let t = self.id_n(&st.tuple);
                    *i = self.n_expr(InstantiateExpr { expr: Some(t), items: li });
                }
            }
            *i = self.transform_allow(Some(*i), true)?.unwrap();
        }
        if Self::is_type_expr(self.cache, head) {
            // enum_test / class_static_protocols: static __getitem__ is ordinary class
            // subscription; instance __getitem__ methods still allow generic T[...].
            let ct = self.extract_class_type_expr(head);
            let methods = self.find_method(ct, "__getitem__")?;
            if methods.iter().any(|m| self.cache.ast[types.func_ast_id(*m)].attrs.has(Attr::StaticMethod)) {
                let d = self.dot_n(head, "__getitem__");
                let c = self.call_n(d, vec![index]);
                self.result_expr = self.transform(Some(c))?;
                return Ok(());
            }
            let i = self.n_expr(InstantiateExpr { expr: Some(head), items });
            self.result_expr = self.transform(Some(i))?;
            return Ok(());
        }

        let new_index = if !is_tuple && items.len() == 1 { items[0] } else { self.n_expr(TupleExpr { items }) };
        self.cache.ast[e].kind.as_index_mut().unwrap().index = Some(new_index);
        let Some(cls) = self.expr_class(head) else {
            // Wait until the type becomes known
            return Ok(());
        };

        // Case: static tuple access
        // Note: needs untransformed origIndex to parse statics nicely
        let (is_static_tuple, tuple_expr) = self.transform_static_tuple_index(cls, head, orig_index)?;
        if is_static_tuple {
            if tuple_expr.is_some() {
                self.result_expr = tuple_expr;
            }
        } else {
            // Case: normal __getitem__
            let d = self.dot_n(head, "__getitem__");
            let c = self.call_n(d, vec![new_index]);
            self.result_expr = self.transform(Some(c))?;
        }
        Ok(())
    }

    /// `transformStaticTupleIndex(tuple, expr, index)`: if `index` is static (integer or
    /// slice), statically extract the specified tuple item or a sub-tuple.
    /// Works only on normal tuples and partial functions. `(handled, replacement)`.
    pub fn transform_static_tuple_index(&mut self, tuple: TypeId, e: ExprId, index: ExprId) -> Result<(bool, Option<ExprId>)> {
        let st = stdlib();
        let types = self.types();
        let et = self.ety(e);
        let is_static_string = types.static_kind(et) == LiteralKind::String;
        if is_static_string && !types.can_realize(et) {
            return Ok((true, None));
        } else if !is_static_string {
            if !types.is_record(tuple) {
                return Ok((false, None));
            }
            if !types.is(tuple, &st.tuple) {
                if types.is(tuple, &st.optional) {
                    if let Some(new_tuple) = types.get_class(self.extract_class_generic(tuple, 0)) {
                        let f = self.id_n(&fn_optional_unwrap());
                        let c = self.call_n(f, vec![e]);
                        let c = self.tr(c)?;
                        return self.transform_static_tuple_index(new_tuple, c, index);
                    } else {
                        return Ok((true, None));
                    }
                }
                return Ok((false, None));
            }
        }

        // Extract the static integer value from expression
        let get_int = |s: &mut Self, o: &mut i64, x: Option<ExprId>| -> Result<bool> {
            let Some(x) = x else { return Ok(true) };
            let c = s.cache.ast.clone_expr(x, false);
            let ore = s.tr(c)?;
            if let Some(v) = s.cache.types.get_int_static(s.ety(ore)) {
                *o = v;
                return Ok(true);
            }
            Ok(false)
        };

        let s = if is_static_string { self.get_str_literal(et, 0) } else { String::new() };
        let sz = if is_static_string { utf8_strlen(&s) as i64 } else { self.get_class_fields(tuple).len() as i64 };
        let (mut start, mut stop, mut step, mut multiple) = (0i64, sz, 1i64, false);
        let index_src = self.cache.ast[index].src;
        if get_int(self, &mut start, Some(index))? {
            // Case: `tuple[int]`
            let i = match translate_index_raw(start, stop, false) {
                Ok(i) => i,
                Err(i) => typhon_common::bail_e!(ErrorKind::TUPLE_RANGE_BOUNDS, self.src, stop - 1, i),
            };
            if i < 0 || i >= stop {
                typhon_common::bail_e!(ErrorKind::TUPLE_RANGE_BOUNDS, index_src, stop - 1, i);
            }
            start = i;
        } else if let Some(slice) = self.cache.ast[index].kind.as_slice().cloned() {
            // Case: `tuple[int:int:int]`
            if !get_int(self, &mut start, slice.start)? || !get_int(self, &mut stop, slice.stop)? || !get_int(self, &mut step, slice.step)? {
                return Ok((false, None));
            }
            // Adjust slice indices (Python slicing rules)
            if slice.step.is_some() && slice.start.is_none() {
                start = if step > 0 { 0 } else { sz - 1 };
            }
            if slice.step.is_some() && slice.stop.is_none() {
                stop = if step > 0 { sz } else { -(sz + 1) };
            }
            self.slice_adjust_indices_checked(sz, &mut start, &mut stop, step)?;
            multiple = true;
        } else {
            return Ok((false, None));
        }

        if is_static_string {
            let v = if !multiple {
                utf8_substr(&s, start as usize, (start + 1) as usize, 1)
            } else {
                utf8_substr(&s, start as usize, stop as usize, step as isize)
            };
            let n = self.n_expr(StringExpr::new(v, ""));
            Ok((true, self.transform(Some(n))?))
        } else {
            let class_fields = self.get_class_fields(tuple);
            if !multiple {
                let d = self.dot_n(e, &class_fields[start as usize].name);
                Ok((true, self.transform(Some(d))?))
            } else {
                // Generate a sub-tuple
                let tv = self.get_temporary_var("tup");
                let var = self.id_n(&tv);
                let ass = self.n_stmt(AssignStmt { lhs: Some(var), rhs: Some(e), type_expr: None, update: UpdateMode::Assign });
                let mut te = vec![];
                let mut i = start;
                while if step > 0 { i < stop } else { i > stop } {
                    if i < 0 || i >= sz {
                        typhon_common::bail_e!(ErrorKind::TUPLE_RANGE_BOUNDS, index_src, sz - 1, i);
                    }
                    let v = self.cache.ast.clone_expr(var, false);
                    te.push(self.dot_n(v, &class_fields[i as usize].name));
                    i += step;
                }
                self.generate_tuple(te.len(), true)?;
                let t = self.id_n(&st.tuple);
                let c = self.call_n(t, te);
                let se = self.op_stmt_expr(vec![ass], c);
                Ok((true, self.transform(Some(se))?))
            }
        }
    }

    /// `translateIndex(idx, len, clamp)`: Python indexing rules for static tuple indices.
    /// PORT-NOTE: C++ throws `TUPLE_RANGE_BOUNDS` from this `const` helper; the signature is
    /// infallible here, so out-of-range indices are returned unchanged and callers check
    /// bounds (see [`translate_index_raw`]).
    pub fn translate_index(&self, idx: i64, len: i64, clamp: bool) -> i64 {
        match translate_index_raw(idx, len, clamp) {
            Ok(i) | Err(i) => i,
        }
    }

    /// `sliceAdjustIndices(length, start, stop, step)`: Python slice indexing rules;
    /// returns the slice length.
    /// PORT-NOTE: `step == 0` (C++ `SLICE_STEP_ZERO`) returns 0 here; the fallible check is
    /// done by callers through `slice_adjust_indices_checked`.
    pub fn slice_adjust_indices(&self, length: i64, start: &mut i64, stop: &mut i64, step: i64) -> i64 {
        if step == 0 {
            return 0;
        }
        slice_adjust_indices_raw(length, start, stop, step)
    }
    fn slice_adjust_indices_checked(&self, length: i64, start: &mut i64, stop: &mut i64, step: i64) -> Result<i64> {
        if step == 0 {
            typhon_common::bail_e!(ErrorKind::SLICE_STEP_ZERO, self.src);
        }
        Ok(slice_adjust_indices_raw(length, start, stop, step))
    }

    /// `visit(SliceExpr*)`: `start::step` -> `Slice(start, Optional.__new__(), step)`.
    pub fn visit_slice(&mut self, e: ExprId) -> Result<()> {
        let st = stdlib();
        let se = self.cache.ast[e].kind.as_slice().unwrap().clone();
        let none_name = mangle::mangled_method("", "Optional", "__new__", 0, 0);
        let mk = |s: &mut Self, x: Option<ExprId>| -> ExprId {
            x.unwrap_or_else(|| {
                let f = s.id_n(&none_name);
                s.call_n(f, vec![])
            })
        };
        let (a, b, c) = (mk(self, se.start), mk(self, se.stop), mk(self, se.step));
        let f = self.id_n(&st.slice);
        let call = self.call_n(f, vec![a, b, c]);
        self.result_expr = self.transform(Some(call))?;
        Ok(())
    }
}

impl<'a> TypecheckVisitor<'a> {
    fn op_inst_items(&self, e: ExprId) -> Vec<ExprId> {
        self.cache.ast[e].kind.as_instantiate().unwrap().items.clone()
    }
    fn op_inst_set(&mut self, e: ExprId, i: usize, x: ExprId) {
        self.cache.ast[e].kind.as_instantiate_mut().unwrap().items[i] = x;
    }
    fn op_tr_type(&mut self, x: ExprId) -> Result<ExprId> {
        Ok(self.transform_type(Some(x), false)?.unwrap())
    }

    /// `visit(InstantiateExpr*)`: transform an instantiation to canonical realized name.
    /// `Instantiate(foo, [bar])` -> `Id("foo[bar]")`
    pub fn visit_instantiate(&mut self, e: ExprId) -> Result<()> {
        let st = stdlib();
        let types = self.types();
        let head = self.cache.ast[e].kind.as_instantiate().unwrap().expr;
        let head = self.transform_type(head, false)?.unwrap();
        self.cache.ast[e].kind.as_instantiate_mut().unwrap().expr = Some(head);

        // Collapse Optional[Optional[...]] to Optional[] (handle NoneType as well).
        if types.is(self.extract_type_expr(head), &st.optional) && self.op_inst_items(e).len() == 1 {
            let f = self.op_tr_type(self.op_inst_items(e)[0])?;
            self.op_inst_set(e, 0, f);
            let member = self.extract_type_expr(f);
            if types.is(member, &st.optional) || types.is(member, &st.none_type) {
                self.result_expr = Some(f);
                let (a, b) = (self.ety(e), self.ety(f));
                self.unify(a, b)?;
                return Ok(());
            }
        }

        let type_params_size = self.op_inst_items(e).len();
        let typ;
        if types.is(self.extract_type_expr(head), &st.tuple) {
            if type_params_size > 0 {
                let f = self.tr(self.op_inst_items(e)[0])?;
                self.op_inst_set(e, 0, f);
                if types.static_kind(self.ety(f)) == LiteralKind::Int {
                    let items = self.op_inst_items(e);
                    let t = self.id_n(&st.tuple);
                    let et = self.n_expr(InstantiateExpr { expr: Some(t), items: items[1..].to_vec() });
                    let nt = self.id_n("__NTuple__");
                    let x = self.n_expr(InstantiateExpr { expr: Some(nt), items: vec![items[0], et] });
                    self.result_expr = self.transform(Some(x))?;
                    return Ok(());
                }
            }
            let t = self.generate_tuple(type_params_size, true)?;
            typ = self.instantiate_type(t, None)?;
        } else {
            let old = std::mem::replace(&mut self.src, self.cache.ast[head].src);
            let r = self.instantiate_type(self.extract_type_expr(head), None);
            self.src = old;
            typ = r?;
        }
        typhon_common::seqassert!(types.get_class(typ).is_some(), "unknown type");

        let generics = types.generics(typ);
        let is_union = types.get_union(typ).is_some();
        if !is_union && type_params_size != generics.len() {
            let src = self.cache.ast[e].src;
            let n = self.get_user_facing_name(&types.class_name(typ).unwrap());
            typhon_common::bail_e!(ErrorKind::GENERICS_MISMATCH, src, n, generics.len(), type_params_size);
        }

        if crate::ast::is_id(&self.cache.ast, Some(head), &st.callable_trait) {
            // Case: CallableTrait[...] trait instantiation
            let mut tys = vec![];
            for (i, tp) in self.op_inst_items(e).into_iter().enumerate() {
                let tp = self.op_tr_type(tp)?;
                self.op_inst_set(e, i, tp);
                if types.static_kind(self.ety(tp)) != LiteralKind::Runtime {
                    let src = self.cache.ast[tp].src;
                    typhon_common::bail_e!(ErrorKind::INST_CALLABLE_STATIC, src);
                }
                tys.push(Some(self.extract_type_expr(tp)));
            }
            let ub = self.instantiate_unbound();
            // Set up the CallableTrait
            let tr = types.new_callable_trait(tys);
            types.with_link_mut(ub, |l| l.trait_ = Some(tr));
            let tv = self.instantiate_type_var(ub)?;
            let et = self.ety(e);
            self.unify(et, tv)?;
        } else if crate::ast::is_id(&self.cache.ast, Some(head), &st.type_trait) {
            // Case: TypeTrait[...] trait instantiation
            let f = self.op_tr_type(self.op_inst_items(e)[0])?;
            self.op_inst_set(e, 0, f);
            let ub = self.instantiate_unbound();
            let tr = types.new_type_trait(self.extract_type_expr(f));
            types.with_link_mut(ub, |l| l.trait_ = Some(tr));
            let et = self.ety(e);
            self.unify(et, ub)?;
        } else if is_union {
            let mut is_optional = false;
            let mut tys: Vec<TypeId> = vec![];
            let mut solo_type_expr = None;
            for i in 0..type_params_size {
                let x = self.op_tr_type(self.op_inst_items(e)[i])?;
                self.op_inst_set(e, i, x);
                let t = self.extract_type(self.ety(x));
                is_optional |= types.is(t, &st.optional) || types.is(t, &st.none_type);
                if let Some(u) = types.get_union(t) {
                    let g0 = types.generic_type(u, 0).unwrap();
                    for g in types.generics(g0) {
                        let gt = g.ty();
                        tys.push(gt);
                        is_optional |= types.is(gt, &st.optional);
                    }
                } else if !types.is(t, &st.none_type) {
                    tys.push(t);
                    solo_type_expr = Some(x);
                }
            }
            if is_optional {
                // A | B | ... | None -> Optional[A] | Optional[B] | ...
                let o = self.get_stdlib_type(&st.optional)?;
                for t in tys.iter_mut() {
                    if !types.is(*t, &st.optional) {
                        *t = self.instantiate_type_with(o, &[*t])?;
                    }
                }
                if let Some(s) = solo_type_expr {
                    let oi = self.id_n(&st.optional);
                    let x = self.n_expr(InstantiateExpr { expr: Some(oi), items: vec![s] });
                    solo_type_expr = self.transform(Some(x))?;
                }
            }
            let mut sorted: std::collections::BTreeMap<String, Vec<TypeId>> = Default::default();
            for t in &tys {
                sorted.entry(types.realized_name(*t)).or_default().push(*t);
            }
            if sorted.is_empty() {
                // All nones: None | None ...
                let n = self.id_n(&st.none_type);
                let r = self.tr(n)?;
                self.result_expr = Some(r);
                let (a, b) = (self.ety(e), self.ety(r));
                self.unify(a, b)?;
            } else if sorted.len() == 1 {
                // Union[T] = T. Note that we do not check for the same types here...
                let s = solo_type_expr.expect("type not detected");
                self.result_expr = Some(s);
                let (a, b) = (self.ety(e), self.ety(s));
                self.unify(a, b)?;
            } else {
                let tt: Vec<TypeId> = sorted.values().map(|v| v[0]).collect();
                let gt = self.generate_tuple(tt.len(), true)?;
                let t = self.instantiate_type_with(gt, &tt)?;
                let ug = types.generic_type(types.get_union(typ).unwrap(), 0).unwrap();
                self.unify(ug, t)?;
                let tv = self.instantiate_type_var(typ)?;
                let et = self.ety(e);
                self.unify(et, tv)?;
                if let Some(realized) = self.realize(et)? {
                    if self.result_expr.is_none() {
                        let ut = self.extract_type(realized);
                        let r = self.id_n(&types.realized_name(ut));
                        self.set_type(r, realized);
                        self.set_done_expr(r);
                        self.result_expr = Some(r);
                    }
                }
            }
        } else {
            for i in 0..type_params_size {
                let mut x = self.op_tr_type(self.op_inst_items(e)[i])?;
                self.op_inst_set(e, i, x);
                let old = std::mem::replace(&mut self.src, self.cache.ast[x].src);
                let t = self.instantiate_type(self.extract_type_expr(x), None);
                self.src = old;
                let t = t?;
                let gt = generics[i].ty();
                if types.static_kind(self.ety(x)) != types.static_kind(gt) {
                    if self.cache.ast[x].kind.tag() == ExprTag::None {
                        // `None` -> `NoneType`
                        x = self.op_tr_type(x)?;
                        self.op_inst_set(e, i, x);
                    }
                    if !Self::is_type_expr(self.cache, x) {
                        let src = self.cache.ast[x].src;
                        typhon_common::bail_e!(ErrorKind::EXPECTED_TYPE, src, "type");
                    }
                }
                self.unify(t, gt)?;
            }
            let tv = self.instantiate_type_var(typ)?;
            let et = self.ety(e);
            self.unify(et, tv)?;

            // If the type is realizable, use the realized name instead of instantiation
            // (e.g. use Id("Ptr[u8]") instead of Instantiate(Ptr, {u8}))
            if let Some(rt) = self.realize(et)? {
                if self.result_expr.is_none() {
                    let t = self.extract_type(rt);
                    let r = self.id_n(&types.realized_name(t));
                    self.set_type(r, rt);
                    self.set_done_expr(r);
                    self.result_expr = Some(r);
                }
            }
        }

        // Handle side effects
        if !self.ctx.borrow().simple_types {
            let mut prepends = vec![];
            for (i, t) in self.op_inst_items(e).into_iter().enumerate() {
                if self.has_side_effect(t) {
                    let name = self.get_temporary_var("call");
                    let tt = self.expr_type(t);
                    let pt = self.get_param_type(tt);
                    let n = self.id_n(&name);
                    let a = self.n_stmt(AssignStmt { lhs: Some(n), rhs: Some(t), type_expr: pt, update: UpdateMode::Assign });
                    let front = self.tr_stmt(a)?;
                    let n = self.id_n(&name);
                    let swap = self.op_tr_type(n)?;
                    self.op_inst_set(e, i, swap);
                    prepends.push(front);
                }
            }
            if !prepends.is_empty() {
                let base = self.result_expr.unwrap_or(e);
                let se = self.op_stmt_expr(prepends, base);
                self.result_expr = self.transform(Some(se))?;
            }
        }
        Ok(())
    }
}
