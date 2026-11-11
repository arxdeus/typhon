//! Port of codon/parser/ast/expr.h (node definitions).

use serde::{Deserialize, Serialize};
use typhon_common::SrcInfo;

use super::attr::Attrs;
use super::src_serde;
use crate::ids::{ExprId, StmtId, TypeId};

/// Base expression node (C++ `ast::Expr`). Owned by [`super::AstArena`].
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Expr {
    pub kind: ExprKind,
    #[serde(with = "src_serde")]
    pub src: SrcInfo,
    pub attrs: Attrs,
    /// Type of the expression (`nullptr` by default). Not serialized (as in C++).
    #[serde(skip)]
    pub ty: Option<TypeId>,
    /// True if all types in the expression are inferred.
    pub done: bool,
    /// Original (pre-transformation) expression.
    pub orig_expr: Option<ExprId>,
    /// Expected type of the expression. Not serialized.
    #[serde(skip)]
    pub expected_type: Option<TypeId>,
}

impl Expr {
    pub fn new(kind: ExprKind, src: SrcInfo) -> Self {
        Self { kind, src, attrs: Attrs::new(), ty: None, done: false, orig_expr: None, expected_type: None }
    }
    pub fn is_done(&self) -> bool {
        self.done
    }
    pub fn set_done(&mut self) {
        self.done = true;
    }
}

/// `Param::Kind`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum ParamKind {
    #[default]
    Normal,
    PositionalOnly,
    KeywordOnly,
}

/// `Param::status`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum ParamStatus {
    #[default]
    Value,
    Generic,
    HiddenGeneric,
}

/// Function/lambda/class parameter (C++ `ast::Param`). Build with
/// [`super::AstArena::new_param`] to get C++ generic-status inference.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Param {
    pub name: String,
    pub ty: Option<ExprId>,
    pub default_value: Option<ExprId>,
    pub kind: ParamKind,
    pub status: ParamStatus,
    #[serde(with = "src_serde")]
    pub src: SrcInfo,
}

impl Param {
    /// Raw constructor with an explicit status (no inference).
    pub fn with_status(name: impl Into<String>, ty: Option<ExprId>, default_value: Option<ExprId>, status: ParamStatus) -> Self {
        Self { name: name.into(), ty, default_value, kind: ParamKind::Normal, status, src: SrcInfo::default() }
    }
    pub fn name(&self) -> &str {
        &self.name
    }
    pub fn is_value(&self) -> bool {
        self.status == ParamStatus::Value
    }
    pub fn is_generic(&self) -> bool {
        self.status == ParamStatus::Generic
    }
    pub fn is_hidden_generic(&self) -> bool {
        self.status == ParamStatus::HiddenGeneric
    }
    /// `getNameWithStars`: (number of leading `*`, bare name).
    pub fn name_with_stars(&self) -> (usize, String) {
        let stars = self.name.bytes().take_while(|&c| c == b'*').count();
        (stars, self.name[stars..].to_string())
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct NoneExpr;

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct BoolExpr {
    pub value: bool,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct IntExpr {
    /// Value stored as a string (parsed during type checking).
    pub value: String,
    /// Number suffix (e.g. "u" for "123u").
    pub suffix: String,
    /// Parsed value for "normal" 64-bit integers.
    pub int_value: Option<i64>,
}

impl IntExpr {
    /// `IntExpr(int64_t)`.
    pub fn new(v: i64) -> Self {
        Self { value: v.to_string(), suffix: String::new(), int_value: Some(v) }
    }
    /// `IntExpr(const std::string &value, std::string suffix)`.
    pub fn parse(value: &str, suffix: impl Into<String>) -> Self {
        let value: String = value.chars().filter(|&c| c != '_').collect();
        let lower = value.to_ascii_lowercase();
        // C++ uses std::stoull (unsigned, out_of_range -> no value); the result is then
        // stored into int64 (wrapping).
        let parsed = if let Some(r) = lower.strip_prefix("0b") {
            u64::from_str_radix(r, 2).ok()
        } else if let Some(r) = lower.strip_prefix("0o") {
            u64::from_str_radix(r, 8).ok()
        } else if let Some(r) = lower.strip_prefix("0x") {
            u64::from_str_radix(r, 16).ok()
        } else {
            stoull_base0(&value)
        };
        Self { value, suffix: suffix.into(), int_value: parsed.map(|v| v as i64) }
    }
    pub fn has_stored_value(&self) -> bool {
        self.int_value.is_some()
    }
    pub fn value(&self) -> i64 {
        self.int_value.expect("value not set")
    }
    pub fn raw_data(&self) -> (String, String) {
        (self.value.clone(), self.suffix.clone())
    }
}

/// `std::stoull(s, nullptr, 0)`: auto-detects `0x` / leading-`0` octal; parses the longest
/// valid prefix; `None` on overflow or no digits.
fn stoull_base0(s: &str) -> Option<u64> {
    let t = s.trim_start();
    let (neg, t) = match t.as_bytes().first() {
        Some(b'-') => (true, &t[1..]),
        Some(b'+') => (false, &t[1..]),
        _ => (false, t),
    };
    let (radix, digits) = if t.len() > 2 && (t.starts_with("0x") || t.starts_with("0X")) {
        (16, &t[2..])
    } else if t.len() > 1 && t.starts_with('0') {
        (8, &t[1..])
    } else {
        (10, t)
    };
    let end = digits.find(|c: char| !c.is_digit(radix)).unwrap_or(digits.len());
    let digits = &digits[..end];
    if digits.is_empty() {
        return if radix == 8 { Some(0) } else { None };
    }
    let v = u64::from_str_radix(digits, radix).ok()?;
    Some(if neg { v.wrapping_neg() } else { v })
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct FloatExpr {
    pub value: String,
    pub suffix: String,
    pub float_value: Option<f64>,
}

impl FloatExpr {
    /// `FloatExpr(double)`: value formatted with `{:g}`.
    pub fn new(v: f64) -> Self {
        Self { value: format_g(v), suffix: String::new(), float_value: Some(v) }
    }
    /// `FloatExpr(const std::string &value, std::string suffix)`.
    pub fn parse(value: &str, suffix: impl Into<String>) -> Self {
        let value: String = value.chars().filter(|&c| c != '_').collect();
        let t = value.trim_start();
        let t = t.strip_prefix('+').unwrap_or(t);
        // fast_float::from_chars parses the longest valid prefix.
        let fv = lexical_core::parse_partial::<f64>(t.as_bytes()).ok().and_then(|(v, n)| (n > 0).then_some(v));
        Self { value, suffix: suffix.into(), float_value: fv }
    }
    pub fn has_stored_value(&self) -> bool {
        self.float_value.is_some()
    }
    pub fn value(&self) -> f64 {
        self.float_value.expect("value not set")
    }
    pub fn raw_data(&self) -> (String, String) {
        (self.value.clone(), self.suffix.clone())
    }
}

/// fmt's `{:g}` (precision 6, shortest of %e / %f, trailing zeros removed).
pub fn format_g(v: f64) -> String {
    if v.is_nan() {
        return "nan".into();
    }
    if v.is_infinite() {
        return if v < 0.0 { "-inf".into() } else { "inf".into() };
    }
    if v == 0.0 {
        return if v.is_sign_negative() { "-0".into() } else { "0".into() };
    }
    let p = 6i32;
    let e = format!("{:.*e}", (p - 1) as usize, v);
    let (mant, exp) = e.split_once('e').unwrap_or((&e, "0"));
    let x: i32 = exp.parse().unwrap_or(0);
    let strip = |s: &str| -> String {
        if s.contains('.') { s.trim_end_matches('0').trim_end_matches('.').to_string() } else { s.to_string() }
    };
    if x < -4 || x >= p {
        let m = strip(mant);
        format!("{}e{}{:02}", m, if x < 0 { '-' } else { '+' }, x.abs())
    } else {
        strip(&format!("{:.*}", (p - 1 - x) as usize, v))
    }
}

/// `StringExpr::FormatSpec`.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct FormatSpec {
    pub text: String,
    pub conversion: String,
    pub spec: String,
}

/// `StringExpr::String`: one `{value, prefix}` piece of a (possibly f-) string.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct StringPart {
    pub value: String,
    pub prefix: String,
    pub expr: Option<ExprId>,
    pub format: FormatSpec,
    #[serde(with = "src_serde")]
    pub src: SrcInfo,
}

impl StringPart {
    pub fn new(value: impl Into<String>, prefix: impl Into<String>, expr: Option<ExprId>) -> Self {
        Self { value: value.into(), prefix: prefix.into(), expr, format: FormatSpec::default(), src: SrcInfo::default() }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct StringExpr {
    pub strings: Vec<StringPart>,
}

impl StringExpr {
    pub fn new(value: impl Into<String>, prefix: impl Into<String>) -> Self {
        Self { strings: vec![StringPart::new(value, prefix, None)] }
    }
    pub fn from_parts(strings: Vec<StringPart>) -> Self {
        Self { strings }
    }
    /// `getValue` (asserts `isSimple`).
    pub fn value(&self) -> String {
        typhon_common::seqassert!(self.is_simple(), "invalid StringExpr");
        self.strings.first().map(|s| s.value.clone()).unwrap_or_default()
    }
    pub fn is_simple(&self) -> bool {
        self.strings.len() == 1 && (self.strings[0].prefix.is_empty() || self.strings[0].prefix == "b")
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct IdExpr {
    pub value: String,
}
impl IdExpr {
    pub fn new(v: impl Into<String>) -> Self {
        Self { value: v.into() }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct StarExpr {
    pub expr: Option<ExprId>,
}
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct KeywordStarExpr {
    pub expr: Option<ExprId>,
}
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct TupleExpr {
    pub items: Vec<ExprId>,
}
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ListExpr {
    pub items: Vec<ExprId>,
}
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct SetExpr {
    pub items: Vec<ExprId>,
}
/// Items are `TupleExpr` (key, value) pairs or `KeywordStarExpr` (`**mapping`).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct DictExpr {
    pub items: Vec<ExprId>,
}

/// `GeneratorExpr::GeneratorKind`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum GeneratorKind {
    #[default]
    Generator,
    ListGenerator,
    SetGenerator,
    TupleGenerator,
    DictGenerator,
}

/// Comprehension; `loops` is a nested For/If/Suite chain ending in an `ExprStmt`.
/// Build with [`super::AstArena::new_generator`].
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct GeneratorExpr {
    pub kind: GeneratorKind,
    pub loops: Option<StmtId>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct IfExpr {
    pub cond: Option<ExprId>,
    pub ifexpr: Option<ExprId>,
    pub elsexpr: Option<ExprId>,
}
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct UnaryExpr {
    pub op: String,
    pub expr: Option<ExprId>,
}
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct BinaryExpr {
    pub op: String,
    pub lexpr: Option<ExprId>,
    pub rexpr: Option<ExprId>,
    /// True if an expression modifies lhs in-place (e.g. a += b).
    pub in_place: bool,
}
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ChainBinaryExpr {
    pub exprs: Vec<(String, ExprId)>,
}

/// `ast::Pipe`.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Pipe {
    pub op: String,
    pub expr: Option<ExprId>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct PipeExpr {
    pub items: Vec<Pipe>,
    /// Output type of a "prefix" pipe ending at the index position. Not serialized.
    #[serde(skip)]
    pub in_types: Vec<Option<TypeId>>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct IndexExpr {
    pub expr: Option<ExprId>,
    pub index: Option<ExprId>,
}

/// `ast::CallArg`.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct CallArg {
    pub name: String,
    pub value: Option<ExprId>,
    #[serde(with = "src_serde")]
    pub src: SrcInfo,
}
impl CallArg {
    /// Positional argument (`CallArg(Expr*)`). Use [`super::AstArena::call_arg`] to copy the
    /// value's source location like C++.
    pub fn new(name: impl Into<String>, value: Option<ExprId>) -> Self {
        Self { name: name.into(), value, src: SrcInfo::default() }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct CallExpr {
    pub expr: Option<ExprId>,
    pub items: Vec<CallArg>,
    /// True if type-checker has processed and re-ordered args.
    pub ordered: bool,
    /// True if the call is partial.
    pub partial: bool,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct DotExpr {
    pub expr: Option<ExprId>,
    pub member: String,
}
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct SliceExpr {
    pub start: Option<ExprId>,
    pub stop: Option<ExprId>,
    pub step: Option<ExprId>,
}

/// `EllipsisExpr::EllipsisType`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum EllipsisType {
    Pipe,
    Partial,
    #[default]
    Standalone,
}
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct EllipsisExpr {
    pub mode: EllipsisType,
}
impl EllipsisExpr {
    pub fn is_standalone(&self) -> bool {
        self.mode == EllipsisType::Standalone
    }
    pub fn is_pipe(&self) -> bool {
        self.mode == EllipsisType::Pipe
    }
    pub fn is_partial(&self) -> bool {
        self.mode == EllipsisType::Partial
    }
}

/// Build with [`super::AstArena::new_lambda`] (runs `parseParams`).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct LambdaExpr {
    pub items: Vec<Param>,
    pub expr: Option<ExprId>,
}
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct YieldExpr;
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct AwaitExpr {
    pub expr: Option<ExprId>,
    /// True if transformed during type checking (not serialized, as in C++).
    #[serde(skip)]
    pub transformed: bool,
}
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct AssignExpr {
    pub var: Option<ExprId>,
    pub expr: Option<ExprId>,
}
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct RangeExpr {
    pub start: Option<ExprId>,
    pub stop: Option<ExprId>,
}
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct StmtExpr {
    pub items: Vec<StmtId>,
    pub expr: Option<ExprId>,
}
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct InstantiateExpr {
    pub expr: Option<ExprId>,
    pub items: Vec<ExprId>,
}

macro_rules! expr_kinds {
    ($($variant:ident($ty:ident) => $as:ident, $as_mut:ident;)*) => {
        /// Expression node kinds (one per C++ `*Expr` class).
        #[derive(Clone, Debug, Serialize, Deserialize)]
        pub enum ExprKind { $($variant($ty)),* }

        /// Discriminant of [`ExprKind`] (C++ `NodeId`).
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
        pub enum ExprTag { $($variant),* }

        impl ExprKind {
            pub fn tag(&self) -> ExprTag {
                match self { $(ExprKind::$variant(_) => ExprTag::$variant),* }
            }
            $(
                pub fn $as(&self) -> Option<&$ty> {
                    match self { ExprKind::$variant(x) => Some(x), _ => None }
                }
                pub fn $as_mut(&mut self) -> Option<&mut $ty> {
                    match self { ExprKind::$variant(x) => Some(x), _ => None }
                }
            )*
        }
        $(
            impl From<$ty> for ExprKind {
                fn from(x: $ty) -> Self { ExprKind::$variant(x) }
            }
        )*
    };
}

expr_kinds! {
    None(NoneExpr) => as_none, as_none_mut;
    Bool(BoolExpr) => as_bool, as_bool_mut;
    Int(IntExpr) => as_int, as_int_mut;
    Float(FloatExpr) => as_float, as_float_mut;
    String(StringExpr) => as_string, as_string_mut;
    Id(IdExpr) => as_id, as_id_mut;
    Star(StarExpr) => as_star, as_star_mut;
    KeywordStar(KeywordStarExpr) => as_keyword_star, as_keyword_star_mut;
    Tuple(TupleExpr) => as_tuple, as_tuple_mut;
    List(ListExpr) => as_list, as_list_mut;
    Set(SetExpr) => as_set, as_set_mut;
    Dict(DictExpr) => as_dict, as_dict_mut;
    Generator(GeneratorExpr) => as_generator, as_generator_mut;
    If(IfExpr) => as_if, as_if_mut;
    Unary(UnaryExpr) => as_unary, as_unary_mut;
    Binary(BinaryExpr) => as_binary, as_binary_mut;
    ChainBinary(ChainBinaryExpr) => as_chain_binary, as_chain_binary_mut;
    Pipe(PipeExpr) => as_pipe, as_pipe_mut;
    Index(IndexExpr) => as_index, as_index_mut;
    Call(CallExpr) => as_call, as_call_mut;
    Dot(DotExpr) => as_dot, as_dot_mut;
    Slice(SliceExpr) => as_slice, as_slice_mut;
    Ellipsis(EllipsisExpr) => as_ellipsis, as_ellipsis_mut;
    Lambda(LambdaExpr) => as_lambda, as_lambda_mut;
    Yield(YieldExpr) => as_yield, as_yield_mut;
    Await(AwaitExpr) => as_await, as_await_mut;
    Assign(AssignExpr) => as_assign, as_assign_mut;
    Range(RangeExpr) => as_range, as_range_mut;
    Stmt(StmtExpr) => as_stmt, as_stmt_mut;
    Instantiate(InstantiateExpr) => as_instantiate, as_instantiate_mut;
}

impl ExprTag {
    /// C++ class name (e.g. `"CallExpr"`).
    pub fn class_name(self) -> &'static str {
        use ExprTag::*;
        match self {
            None => "NoneExpr",
            Bool => "BoolExpr",
            Int => "IntExpr",
            Float => "FloatExpr",
            String => "StringExpr",
            Id => "IdExpr",
            Star => "StarExpr",
            KeywordStar => "KeywordStarExpr",
            Tuple => "TupleExpr",
            List => "ListExpr",
            Set => "SetExpr",
            Dict => "DictExpr",
            Generator => "GeneratorExpr",
            If => "IfExpr",
            Unary => "UnaryExpr",
            Binary => "BinaryExpr",
            ChainBinary => "ChainBinaryExpr",
            Pipe => "PipeExpr",
            Index => "IndexExpr",
            Call => "CallExpr",
            Dot => "DotExpr",
            Slice => "SliceExpr",
            Ellipsis => "EllipsisExpr",
            Lambda => "LambdaExpr",
            Yield => "YieldExpr",
            Await => "AwaitExpr",
            Assign => "AssignExpr",
            Range => "RangeExpr",
            Stmt => "StmtExpr",
            Instantiate => "InstantiateExpr",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn int_parse() {
        assert_eq!(IntExpr::parse("1_000", "").int_value, Some(1000));
        assert_eq!(IntExpr::parse("0x1F", "u").int_value, Some(31));
        assert_eq!(IntExpr::parse("0b101", "").int_value, Some(5));
        assert_eq!(IntExpr::parse("0o17", "").int_value, Some(15));
        assert_eq!(IntExpr::parse("18446744073709551615", "").int_value, Some(-1));
        assert_eq!(IntExpr::parse("99999999999999999999999", "").int_value, None);
        assert_eq!(IntExpr::parse("0", "").int_value, Some(0));
    }

    #[test]
    fn float_g() {
        assert_eq!(format_g(1.0), "1");
        assert_eq!(format_g(0.1), "0.1");
        assert_eq!(format_g(1e-5), "1e-05");
        assert_eq!(format_g(1e20), "1e+20");
        assert_eq!(format_g(123456.0), "123456");
        assert_eq!(format_g(1234567.0), "1.23457e+06");
        assert_eq!(FloatExpr::parse("1_0.5", "").float_value, Some(10.5));
        assert_eq!(FloatExpr::parse("1e400", "").float_value, Some(f64::INFINITY));
    }
}
