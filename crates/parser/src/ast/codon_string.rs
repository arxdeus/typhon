//! Port of codon/parser/ast/codon_string.cpp: `ASTNode::toCodonString` (native Codon AST
//! dump with optional attributes and pretty-printing).

use std::cell::RefCell;
use std::collections::{BTreeMap, HashSet};

use super::arena::AstArena;
use super::attr::{AttrValue, Attrs, Binding, BindingsAttr, CaptureType};
use super::expr::*;
use super::stmt::*;
use crate::ids::{ExprId, StmtId, TypeId};

/// `detail::CodonString`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Formatted {
    pub value: String,
    pub simple: bool,
}

fn f(value: impl Into<String>) -> Formatted {
    Formatted { value: value.into(), simple: true }
}
fn empty() -> Formatted {
    f("")
}

fn nested_level(indent: i32, level: i32) -> i32 {
    if indent > 0 { level + 1 } else { level }
}
fn prefix(indent: i32, level: i32) -> String {
    if indent > 0 { format!("\n{}", " ".repeat((indent * level).max(0) as usize)) } else { String::new() }
}
fn separator(indent: i32, level: i32) -> String {
    if indent > 0 { format!(",\n{}", " ".repeat((indent * level).max(0) as usize)) } else { ", ".into() }
}

fn format_items(items: Vec<String>, simple: bool, max_items: usize, indent: i32, level: i32) -> Formatted {
    let items: Vec<String> = items.into_iter().filter(|s| !s.is_empty()).collect();
    if (simple && items.len() <= max_items) || indent <= 0 {
        return f(items.join(", "));
    }
    Formatted { value: prefix(indent, level) + &items.join(&separator(indent, level)), simple: false }
}

/// `quote` (codon_string.cpp): Python-style repr quoting.
pub fn quote(value: &str) -> String {
    let use_double = value.contains('\'') && !value.contains('"');
    let delim = if use_double { b'"' } else { b'\'' };
    let mut r: Vec<u8> = vec![delim];
    const HEX: &[u8] = b"0123456789abcdef";
    for &c in value.as_bytes() {
        if c == delim || c == b'\\' {
            r.push(b'\\');
            r.push(c);
        } else {
            match c {
                0x07 => r.extend_from_slice(b"\\a"),
                0x08 => r.extend_from_slice(b"\\b"),
                0x0c => r.extend_from_slice(b"\\f"),
                b'\n' => r.extend_from_slice(b"\\n"),
                b'\r' => r.extend_from_slice(b"\\r"),
                b'\t' => r.extend_from_slice(b"\\t"),
                0x0b => r.extend_from_slice(b"\\v"),
                c if c < 32 || c == 127 => {
                    r.extend_from_slice(b"\\x");
                    r.push(HEX[(c >> 4) as usize]);
                    r.push(HEX[(c & 15) as usize]);
                }
                c => r.push(c),
            }
        }
    }
    r.push(delim);
    String::from_utf8_lossy(&r).into_owned()
}

fn fstr(v: &str) -> Formatted {
    if v.is_empty() { empty() } else { f(quote(v)) }
}
fn fbool(v: bool) -> Formatted {
    f(if v { "True" } else { "False" })
}
fn fint(v: impl ToString) -> Formatted {
    f(v.to_string())
}

/// `formatValue(double)`: Python `repr(float)`.
pub fn format_float(value: f64) -> String {
    if value.is_nan() {
        return "nan".into();
    }
    if value.is_infinite() {
        return if value < 0.0 { "-inf".into() } else { "inf".into() };
    }
    // Shortest round-trip digits and decimal exponent.
    let e = format!("{value:e}");
    let (mant, exp) = e.split_once('e').unwrap_or((&e, "0"));
    let exponent: i32 = exp.parse().unwrap_or(0);
    let negative = mant.starts_with('-');
    let digits: String = mant.chars().filter(|c| c.is_ascii_digit()).collect();
    let mut result = String::from(if negative { "-" } else { "" });
    if (-4..16).contains(&exponent) {
        let decimal_pos = exponent + 1;
        if decimal_pos <= 0 {
            result += &format!("0.{}{}", "0".repeat((-decimal_pos) as usize), digits);
        } else if decimal_pos as usize >= digits.len() {
            result += &format!("{}{}.0", digits, "0".repeat(decimal_pos as usize - digits.len()));
        } else {
            result += &format!("{}.{}", &digits[..decimal_pos as usize], &digits[decimal_pos as usize..]);
        }
    } else {
        let m = if digits.len() > 1 { format!("{}.{}", &digits[..1], &digits[1..]) } else { digits };
        result += &format!("{}e{}{:02}", m, if exponent < 0 { '-' } else { '+' }, exponent.abs());
    }
    result
}

struct Field {
    value: String,
    simple: bool,
}
fn field(name: &str, v: Formatted) -> Field {
    Field { value: if v.value.is_empty() { String::new() } else { format!("{name}={}", v.value) }, simple: v.simple }
}

fn format_record(name: &str, fields: Vec<Field>, indent: i32, level: i32) -> Formatted {
    let level = nested_level(indent, level);
    let all_simple = fields.iter().all(|x| x.simple);
    let args: Vec<String> = fields.into_iter().map(|x| x.value).filter(|v| !v.is_empty()).collect();
    let fm = format_items(args, all_simple, 1, indent, level);
    Formatted { value: format!("{name}({})", fm.value), simple: fm.simple }
}

fn format_list(items: Vec<String>, open: &str, close: &str, indent: i32, level: i32) -> Formatted {
    let fm = format_items(items, true, 1, indent, level);
    Formatted { value: format!("{open}{}{close}", fm.value), simple: fm.simple }
}

fn format_dict<T>(values: &BTreeMap<String, T>, indent: i32, level: i32, fv: impl Fn(&T) -> Formatted) -> Formatted {
    if values.is_empty() {
        return empty();
    }
    let level = nested_level(indent, level);
    let items = values.iter().map(|(k, v)| format!("{}={}", quote(k), fv(v).value)).collect();
    format_list(items, "{", "}", indent, level)
}

fn format_capture(c: &CaptureType) -> Formatted {
    f(match c {
        CaptureType::Read => "Read",
        CaptureType::Global => "Global",
        CaptureType::Nonlocal => "Nonlocal",
    })
}
fn format_binding(b: &Binding) -> Formatted {
    f(format!(
        "Bindings.Binding(name={}, count={}, is_nonlocal={})",
        quote(&b.name),
        b.count,
        if b.is_nonlocal { "True" } else { "False" }
    ))
}
fn format_bindings(v: &BindingsAttr, indent: i32, level: i32) -> Formatted {
    let child = nested_level(indent, level);
    format_record(
        "Bindings",
        vec![
            field("captures", format_dict(&v.captures, indent, child, format_capture)),
            field("bindings", format_dict(&v.bindings, indent, child, format_binding)),
            field("local_renames", format_dict(&v.local_renames, indent, child, |s| fstr(s))),
        ],
        indent,
        level,
    )
}

fn format_str_vec(v: &[String], indent: i32, level: i32) -> Formatted {
    if v.is_empty() {
        return empty();
    }
    let level = nested_level(indent, level);
    format_list(v.iter().map(|s| fstr(s).value).collect(), "[", "]", indent, level)
}

fn format_attribute(a: &AttrValue, indent: i32, level: i32) -> Formatted {
    match a {
        AttrValue::Flag => empty(),
        AttrValue::Str(s) => fstr(s),
        AttrValue::Int(i) => fint(i),
        AttrValue::StrList(v) => format_str_vec(v, indent, level),
        AttrValue::KeyValue(m) => {
            let items: Vec<String> = m.iter().map(|(k, v)| format!("{}: {}", quote(k), quote(v))).collect();
            f(format!("KeyValueAttribute(attributes={{{}}})", items.join(", ")))
        }
        AttrValue::Bindings(b) => format_bindings(b, indent, level),
    }
}

fn format_attributes(attrs: &Attrs, indent: i32, level: i32) -> Formatted {
    let mut values: Vec<(&str, Formatted)> = attrs
        .iter()
        .filter_map(|(k, v)| k.codon_name().map(|n| (n, format_attribute(v, indent, level + 2))))
        .collect();
    if values.is_empty() {
        return empty();
    }
    values.sort_by(|a, b| a.0.cmp(b.0));
    let all_simple = values.iter().all(|v| v.1.simple);
    let items = values
        .into_iter()
        .map(|(n, v)| if v.value.is_empty() { n.to_string() } else { format!("{n}={}", v.value) })
        .collect();
    let fm = format_items(items, all_simple, 1, indent, level + 1);
    Formatted { value: format!("attrs=({})", fm.value), simple: fm.simple }
}

fn format_node(name: &str, attrs: &Attrs, done: bool, fields: Vec<Field>, attributes: bool, indent: i32, level: i32) -> Formatted {
    let level = nested_level(indent, level);
    let mut all_simple = fields.iter().all(|x| x.simple);
    let mut args: Vec<String> = fields.into_iter().map(|x| x.value).filter(|v| !v.is_empty()).collect();
    if attributes {
        let v = format_attributes(attrs, indent, level);
        all_simple = all_simple && v.simple;
        if !v.value.is_empty() {
            args.push(v.value);
        }
    }
    let fm = format_items(args, all_simple, 1, indent, level);
    Formatted { value: format!("{}{}({})", name, if done { "*" } else { "" }, fm.value), simple: fm.simple }
}

struct Ctx<'a> {
    a: &'a AstArena,
    attributes: bool,
    indent: i32,
    active: RefCell<HashSet<ExprId>>,
}

impl Ctx<'_> {
    fn ty(&self, t: Option<TypeId>) -> Formatted {
        t.map(|t| f(self.a.type_debug_string(t, 2))).unwrap_or_else(empty)
    }
    fn e(&self, e: Option<ExprId>, level: i32) -> Formatted {
        e.map(|e| self.expr(e, level)).unwrap_or_else(empty)
    }
    fn s(&self, s: Option<StmtId>, level: i32) -> Formatted {
        s.map(|s| self.stmt(s, level)).unwrap_or_else(empty)
    }
    fn vec<T>(&self, v: &[T], level: i32, fv: impl Fn(&T, i32) -> Formatted) -> Formatted {
        if v.is_empty() {
            return empty();
        }
        let level = nested_level(self.indent, level);
        let items = v.iter().map(|x| fv(x, level).value).collect();
        format_list(items, "[", "]", self.indent, level)
    }
    fn evec(&self, v: &[ExprId], level: i32) -> Formatted {
        self.vec(v, level, |&x, l| self.expr(x, l))
    }
    fn svec(&self, v: &[StmtId], level: i32) -> Formatted {
        self.vec(v, level, |&x, l| self.stmt(x, l))
    }
    fn strct(&self, name: &str, fields: Vec<Field>, level: i32) -> Formatted {
        format_record(name, fields, self.indent, level)
    }
    fn param(&self, p: &Param, level: i32) -> Formatted {
        let c = nested_level(self.indent, level);
        let status = match p.status {
            ParamStatus::Value => "Value",
            ParamStatus::Generic => "Generic",
            ParamStatus::HiddenGeneric => "HiddenGeneric",
        };
        self.strct(
            "Param",
            vec![
                field("name", fstr(&p.name)),
                field("type", self.e(p.ty, c)),
                field("default", self.e(p.default_value, c)),
                field("status", f(status)),
            ],
            level,
        )
    }
    fn call_arg(&self, a: &CallArg, level: i32) -> Formatted {
        let c = nested_level(self.indent, level);
        self.strct("Arg", vec![field("name", fstr(&a.name)), field("value", self.e(a.value, c))], level)
    }
    fn string_part(&self, p: &StringPart, level: i32) -> Formatted {
        let c = nested_level(self.indent, level);
        let fmt = if p.format.conversion.is_empty() && p.format.spec.is_empty() {
            empty()
        } else {
            self.strct(
                "StringExpr.FormatSpec",
                vec![field("conversion", fstr(&p.format.conversion)), field("spec", fstr(&p.format.spec))],
                c,
            )
        };
        self.strct(
            "String",
            vec![
                field("value", fstr(&p.value)),
                field("prefix", fstr(&p.prefix)),
                field("expr", self.e(p.expr, c)),
                field("format", fmt),
            ],
            level,
        )
    }

    fn expr(&self, id: ExprId, level: i32) -> Formatted {
        if !self.active.borrow_mut().insert(id) {
            return empty();
        }
        let node = &self.a[id];
        let ind = self.indent;
        let c = nested_level(ind, level);
        let mut all = vec![
            field("type", self.ty(node.ty)),
            field("orig", self.e(node.orig_expr, c)),
            field("expected_type", self.ty(node.expected_type)),
        ];
        let fe = |e: Option<ExprId>| self.e(e, c);
        let fields: Vec<Field> = match &node.kind {
            ExprKind::None(_) | ExprKind::Yield(_) => vec![],
            ExprKind::Bool(b) => vec![field("value", fbool(b.value))],
            ExprKind::Int(i) => vec![
                field("value", fstr(&i.value)),
                field("suffix", fstr(&i.suffix)),
                field("int_value", i.int_value.map(fint).unwrap_or_else(empty)),
            ],
            ExprKind::Float(i) => vec![
                field("value", fstr(&i.value)),
                field("suffix", fstr(&i.suffix)),
                field("float_value", i.float_value.map(|v| f(format_float(v))).unwrap_or_else(empty)),
            ],
            ExprKind::String(s) => {
                let (v, p) = if s.strings.len() == 1 {
                    (s.strings[0].value.as_str(), s.strings[0].prefix.as_str())
                } else {
                    ("", "")
                };
                vec![
                    field("strings", self.vec(&s.strings, c, |p, l| self.string_part(p, l))),
                    field("value", fstr(v)),
                    field("prefix", fstr(p)),
                ]
            }
            ExprKind::Id(i) => vec![field("value", fstr(&i.value))],
            ExprKind::Star(e) => vec![field("expr", fe(e.expr))],
            ExprKind::KeywordStar(e) => vec![field("expr", fe(e.expr))],
            ExprKind::Tuple(e) => vec![field("items", self.evec(&e.items, c))],
            ExprKind::List(e) => vec![field("items", self.evec(&e.items, c))],
            ExprKind::Set(e) => vec![field("items", self.evec(&e.items, c))],
            ExprKind::Dict(e) => vec![field("items", self.evec(&e.items, c))],
            ExprKind::Generator(g) => vec![
                field(
                    "kind",
                    f(match g.kind {
                        GeneratorKind::Generator => "Generator",
                        GeneratorKind::ListGenerator => "ListGenerator",
                        GeneratorKind::SetGenerator => "SetGenerator",
                        GeneratorKind::TupleGenerator => "TupleGenerator",
                        GeneratorKind::DictGenerator => "DictGenerator",
                    }),
                ),
                field("loops", self.s(g.loops, c)),
            ],
            ExprKind::If(e) => vec![field("cond", fe(e.cond)), field("ifexpr", fe(e.ifexpr)), field("elsexpr", fe(e.elsexpr))],
            ExprKind::Unary(e) => vec![field("op", fstr(&e.op)), field("expr", fe(e.expr))],
            ExprKind::Binary(e) => vec![
                field("lexpr", fe(e.lexpr)),
                field("op", fstr(&e.op)),
                field("rexpr", fe(e.rexpr)),
                field("in_place", fbool(e.in_place)),
            ],
            ExprKind::ChainBinary(e) => vec![field(
                "exprs",
                self.vec(&e.exprs, c, |(op, x), l| {
                    let l = nested_level(ind, l);
                    let items = vec![fstr(op).value, self.expr(*x, l).value];
                    format_list(items, "(", ")", ind, l)
                }),
            )],
            ExprKind::Pipe(e) => vec![
                field(
                    "items",
                    self.vec(&e.items, c, |p, l| {
                        let cc = nested_level(ind, l);
                        self.strct("Pipe", vec![field("op", fstr(&p.op)), field("expr", self.e(p.expr, cc))], l)
                    }),
                ),
                field("in_types", self.vec(&e.in_types, c, |t, _| self.ty(*t))),
            ],
            ExprKind::Index(e) => vec![field("expr", fe(e.expr)), field("index", fe(e.index))],
            ExprKind::Call(e) => vec![
                field("items", self.vec(&e.items, c, |x, l| self.call_arg(x, l))),
                field("expr", fe(e.expr)),
                field("ordered", fbool(e.ordered)),
                field("partial", fbool(e.partial)),
            ],
            ExprKind::Dot(e) => vec![field("expr", fe(e.expr)), field("member", fstr(&e.member))],
            ExprKind::Slice(e) => vec![field("start", fe(e.start)), field("stop", fe(e.stop)), field("step", fe(e.step))],
            ExprKind::Ellipsis(e) => vec![field(
                "mode",
                f(match e.mode {
                    EllipsisType::Pipe => "Pipe",
                    EllipsisType::Partial => "Partial",
                    EllipsisType::Standalone => "Standalone",
                }),
            )],
            ExprKind::Lambda(e) => {
                vec![field("items", self.vec(&e.items, c, |p, l| self.param(p, l))), field("expr", fe(e.expr))]
            }
            ExprKind::Await(e) => vec![field("expr", fe(e.expr)), field("transformed", fbool(e.transformed))],
            ExprKind::Assign(e) => vec![field("var", fe(e.var)), field("expr", fe(e.expr))],
            ExprKind::Range(e) => vec![field("start", fe(e.start)), field("stop", fe(e.stop))],
            ExprKind::Stmt(e) => vec![field("items", self.svec(&e.items, c)), field("expr", fe(e.expr))],
            ExprKind::Instantiate(e) => vec![field("items", self.evec(&e.items, c)), field("expr", fe(e.expr))],
        };
        all.extend(fields);
        let r = format_node(node.kind.tag().class_name(), &node.attrs, node.done, all, self.attributes, ind, level);
        self.active.borrow_mut().remove(&id);
        r
    }

    fn stmt(&self, id: StmtId, level: i32) -> Formatted {
        let node = &self.a[id];
        let ind = self.indent;
        let c = nested_level(ind, level);
        let fe = |e: Option<ExprId>| self.e(e, c);
        let fs = |s: Option<StmtId>| self.s(s, c);
        let fields: Vec<Field> = match &node.kind {
            StmtKind::Suite(s) => {
                if s.items.is_empty() {
                    return empty();
                }
                let level = nested_level(ind, level);
                let mut all_simple = true;
                let mut args = vec![];
                for &i in &s.items {
                    let v = self.stmt(i, level);
                    all_simple &= v.simple;
                    if !v.value.is_empty() {
                        args.push(v.value);
                    }
                }
                if self.attributes {
                    let v = format_attributes(&node.attrs, ind, level);
                    all_simple &= v.simple;
                    if !v.value.is_empty() {
                        args.push(v.value);
                    }
                }
                let fm = format_items(args, all_simple, 1, ind, level);
                return Formatted { value: format!("{}{}]", if node.done { "*[" } else { "[" }, fm.value), simple: fm.simple };
            }
            StmtKind::Break(_) | StmtKind::Continue(_) => vec![],
            StmtKind::Expr(s) => vec![field("expr", fe(s.expr))],
            StmtKind::Assign(s) => vec![
                field("lhs", fe(s.lhs)),
                field("rhs", fe(s.rhs)),
                field("type_expr", fe(s.type_expr)),
                field(
                    "update",
                    f(match s.update {
                        UpdateMode::Assign => "Assign",
                        UpdateMode::Update => "Update",
                        UpdateMode::UpdateAtomic => "UpdateAtomic",
                        UpdateMode::ThreadLocalAssign => "ThreadLocalAssign",
                    }),
                ),
            ],
            StmtKind::Del(s) => vec![field("expr", fe(s.expr))],
            StmtKind::Print(s) => vec![field("items", self.evec(&s.items, c)), field("no_newline", fbool(s.no_newline))],
            StmtKind::Return(s) => vec![field("expr", fe(s.expr))],
            StmtKind::Yield(s) => vec![field("expr", fe(s.expr))],
            StmtKind::Assert(s) => vec![field("expr", fe(s.expr)), field("message", fe(s.message))],
            StmtKind::While(s) => vec![
                field("cond", fe(s.cond)),
                field("suite", fs(s.suite)),
                field("else_suite", fs(s.else_suite)),
                field("goto_var", fstr(&s.goto_var)),
            ],
            StmtKind::For(s) => vec![
                field("var", fe(s.var)),
                field("iter", fe(s.iter)),
                field("suite", fs(s.suite)),
                field("else_suite", fs(s.else_suite)),
                field("decorator", fe(s.decorator)),
                field("omp_args", self.vec(&s.omp_args, c, |x, l| self.call_arg(x, l))),
                field("async_", fbool(s.async_)),
                field("wrapped", fbool(s.wrapped)),
                field("flat", fbool(s.flat)),
            ],
            StmtKind::If(s) => {
                vec![field("cond", fe(s.cond)), field("if_suite", fs(s.if_suite)), field("else_suite", fs(s.else_suite))]
            }
            StmtKind::Match(s) => vec![
                field(
                    "items",
                    self.vec(&s.items, c, |m, l| {
                        let cc = nested_level(ind, l);
                        self.strct(
                            "Case",
                            vec![
                                field("pattern", self.e(m.pattern, cc)),
                                field("guard", self.e(m.guard, cc)),
                                field("suite", self.s(m.suite, cc)),
                            ],
                            l,
                        )
                    }),
                ),
                field("expr", fe(s.expr)),
            ],
            StmtKind::Import(s) => vec![
                field("from_expr", fe(s.from)),
                field("what", fe(s.what)),
                field("args", self.vec(&s.args, c, |p, l| self.param(p, l))),
                field("ret", fe(s.ret)),
                field("as_", fstr(&s.as_)),
                field("dots", fint(s.dots)),
                field("is_function", fbool(s.is_function)),
            ],
            StmtKind::Except(s) => vec![field("var", fstr(&s.var)), field("exc", fe(s.exc)), field("suite", fs(s.suite))],
            StmtKind::Try(s) => vec![
                field("items", self.svec(&s.items, c)),
                field("suite", fs(s.suite)),
                field("else_suite", fs(s.else_suite)),
                field("finally_suite", fs(s.finally)),
            ],
            StmtKind::Throw(s) => {
                vec![field("expr", fe(s.expr)), field("from_expr", fe(s.from)), field("transformed", fbool(s.transformed))]
            }
            StmtKind::Global(s) => vec![field("var", fstr(&s.var)), field("non_local", fbool(s.non_local))],
            StmtKind::Function(s) => vec![
                field("items", self.vec(&s.items, c, |p, l| self.param(p, l))),
                field("name", fstr(&s.name)),
                field("ret", fe(s.ret)),
                field("suite", fs(s.suite)),
                field("decorators", self.evec(&s.decorators, c)),
                field("async_", fbool(s.async_)),
            ],
            StmtKind::Class(s) => vec![
                field("items", self.vec(&s.items, c, |p, l| self.param(p, l))),
                field("name", fstr(&s.name)),
                field("suite", fs(s.suite)),
                field("decorators", self.evec(&s.decorators, c)),
                field("base_classes", self.evec(&s.base_classes, c)),
            ],
            StmtKind::YieldFrom(s) => vec![field("expr", fe(s.expr))],
            StmtKind::With(s) => vec![
                field("items", self.evec(&s.items, c)),
                field("vars", format_str_vec(&s.vars, ind, c)),
                field("suite", fs(s.suite)),
                field("async_", fbool(s.async_)),
            ],
            StmtKind::Custom(s) => {
                vec![field("keyword", fstr(&s.keyword)), field("expr", fe(s.expr)), field("suite", fs(s.suite))]
            }
            StmtKind::Directive(s) => vec![field("key", fstr(&s.key)), field("value", fstr(&s.value))],
            StmtKind::AssignMember(s) => vec![
                field("lhs", fe(s.lhs)),
                field("member", fstr(&s.member)),
                field("rhs", fe(s.rhs)),
                field("type_expr", fe(s.type_expr)),
            ],
            StmtKind::Comment(s) => vec![field("comment", fstr(&s.comment))],
        };
        let name = match node.kind.tag() {
            StmtTag::Except => "Except",
            t => t.class_name(),
        };
        format_node(name, &node.attrs, node.done, fields, self.attributes, ind, level)
    }
}

impl AstArena {
    /// `Expr::toCodonString(attributes, indent, level)`.
    pub fn expr_to_codon(&self, id: ExprId, attributes: bool, indent: i32, level: i32) -> String {
        Ctx { a: self, attributes, indent, active: RefCell::default() }.expr(id, level).value
    }
    /// `Stmt::toCodonString(attributes, indent, level)`.
    pub fn stmt_to_codon(&self, id: StmtId, attributes: bool, indent: i32, level: i32) -> String {
        Ctx { a: self, attributes, indent, active: RefCell::default() }.stmt(id, level).value
    }
    /// `formatCodonString` for an expression (value + simplicity flag).
    pub fn expr_format_codon(&self, id: ExprId, attributes: bool, indent: i32, level: i32) -> Formatted {
        Ctx { a: self, attributes, indent, active: RefCell::default() }.expr(id, level)
    }
    /// `formatCodonString` for a statement.
    pub fn stmt_format_codon(&self, id: StmtId, attributes: bool, indent: i32, level: i32) -> Formatted {
        Ctx { a: self, attributes, indent, active: RefCell::default() }.stmt(id, level)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn floats() {
        let cases: &[(f64, &str)] = &[
            (0.0, "0.0"),
            (-0.0, "-0.0"),
            (1.0, "1.0"),
            (-1.5, "-1.5"),
            (0.1, "0.1"),
            (1.2345678901234567, "1.2345678901234567"),
            (1e-5, "1e-05"),
            (1e-4, "0.0001"),
            (-1e-4, "-0.0001"),
            (1e15, "1000000000000000.0"),
            (-1e15, "-1000000000000000.0"),
            (1e16, "1e+16"),
            (1e20, "1e+20"),
            (f64::MIN_POSITIVE, "2.2250738585072014e-308"),
            (f64::MAX, "1.7976931348623157e+308"),
            (5e-324, "5e-324"),
            (f64::INFINITY, "inf"),
            (f64::NEG_INFINITY, "-inf"),
            (f64::NAN, "nan"),
        ];
        for (v, e) in cases {
            assert_eq!(format_float(*v), *e, "{v}");
        }
    }

    #[test]
    fn quoting() {
        assert_eq!(quote("a'b"), "\"a'b\"");
        assert_eq!(quote("a'\"b"), "'a\\'\"b'");
        assert_eq!(quote("\x01\n"), "'\\x01\\n'");
    }
}
