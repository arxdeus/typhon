//! Port of `codon/cir/util/format.{h,cpp}`: the IR s-expression pretty printer.

use std::fmt::Write;

use rustc_hash::FxHashSet;

use crate::ids::{Id, NodeRef, TypeId, ValueId, VarId};
use crate::module::Module;
use crate::types::{Generic, TypeKind};
use crate::value::ValueKind;
use crate::var::FuncKind;

/// Port of anonymous `escapeString`.
pub fn escape_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '\x07' => out.push_str("\\a"),
            '\x08' => out.push_str("\\b"),
            '\x0c' => out.push_str("\\f"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\x0b' => out.push_str("\\v"),
            '\\' => out.push_str("\\\\"),
            '\'' => out.push_str("\\'"),
            '"' => out.push_str("\\\""),
            '?' => out.push_str("\\?"),
            c => out.push(c),
        }
    }
    out
}

/// Formats a C++ `double` the way `fmt::format("{}", d)` does (shortest round-trip).
pub fn format_float(f: f64) -> String {
    if f.is_nan() {
        return "nan".into();
    }
    if f.is_infinite() {
        return if f > 0.0 { "inf".into() } else { "-inf".into() };
    }
    let s = format!("{f:?}");
    // Rust prints `1.0` / `1e20`; fmt prints `1` / `1e+20`.
    let s = if let Some(stripped) = s.strip_suffix(".0") { stripped.to_string() } else { s };
    if let Some(pos) = s.find('e') {
        let (mant, exp) = s.split_at(pos);
        let exp = &exp[1..];
        let (sign, digits) = if let Some(d) = exp.strip_prefix('-') { ("-", d) } else { ("+", exp) };
        let digits = if digits.len() < 2 { format!("0{digits}") } else { digits.to_string() };
        return format!("{mant}e{sign}{digits}");
    }
    s
}

struct Formatter<'a> {
    m: &'a Module,
    seen_nodes: FxHashSet<Id>,
    seen_types: FxHashSet<String>,
}

impl<'a> Formatter<'a> {
    fn ref_str(&self, n: NodeRef) -> String {
        self.m.reference_string(n)
    }

    fn fmt_type(&mut self, t: Option<TypeId>, can_show_full: bool) -> String {
        let Some(t) = t else { return "(null)".into() };
        let name = self.m.type_name(t).to_string();
        if self.seen_types.contains(&name) || !can_show_full {
            return format!("(type '\"{}\")", self.ref_str(t.into()));
        }
        self.seen_types.insert(name);
        self.visit_type(t)
    }

    fn fmt_value(&mut self, v: Option<ValueId>) -> String {
        let Some(v) = v else { return "(null)".into() };
        let uid = self.m.value_uid(v);
        if self.seen_nodes.contains(&uid) {
            return format!("(value '\"{}\")", self.ref_str(v.into()));
        }
        self.seen_nodes.insert(uid);
        self.visit_value(v)
    }

    fn fmt_var(&mut self, v: Option<VarId>, can_show_full: bool) -> String {
        let Some(v) = v else { return "(null)".into() };
        let uid = self.m.var_uid(v);
        if self.seen_nodes.contains(&uid) || !can_show_full {
            return format!("(var '\"{}\")", self.ref_str(v.into()));
        }
        self.seen_nodes.insert(uid);
        self.visit_var(v)
    }

    fn visit_module(&mut self) -> String {
        let m = self.m;
        let types: Vec<String> = m.types().into_iter().map(|t| self.fmt_type(Some(t), true)).collect();
        let vars: Vec<String> = m.vars().into_iter().map(|v| self.fmt_var(Some(v), true)).collect();
        let argv = self.fmt_var(Some(m.argv_var()), true);
        let argc = self.fmt_var(Some(m.argc_var()), true);
        let main = self.fmt_var(Some(m.main_func()), true);
        format!(
            "(module\n(argv {})\n(argc {})\n(types {})\n(vars {})\n{})",
            argv,
            argc,
            types.join("\n"),
            vars.join("\n"),
            main
        )
    }

    fn visit_var(&mut self, v: VarId) -> String {
        let m = self.m;
        let v = m.actual_var(v);
        let d = m.var(v);
        let r = self.ref_str(v.into());
        let ty = self.fmt_type(d.ty, false);
        match &d.func {
            None => format!(
                "(var '\"{}\" {} (global {}) (external {}) (thread-local {}))",
                r, ty, d.global, d.external, d.tls
            ),
            Some(f) => match &f.kind {
                FuncKind::Bodied { symbols, body, .. } => {
                    let args: Vec<String> = f.args.iter().map(|a| self.fmt_var(Some(*a), true)).collect();
                    let syms: Vec<String> = symbols.iter().map(|a| self.fmt_var(Some(*a), true)).collect();
                    let body = self.fmt_value(*body);
                    format!(
                        "(bodied_func '\"{}\" {}\n(args {})\n(vars {})\n{})",
                        r,
                        ty,
                        args.join(" "),
                        syms.join(" "),
                        body
                    )
                }
                FuncKind::External => format!("(external_func '\"{}\" {})", r, ty),
                FuncKind::Internal { intrinsic, intrinsic_type } => {
                    let it = self.fmt_type(*intrinsic_type, false);
                    format!("(internal_func '\"{}\" {} (intrinsic {} {}))", r, ty, intrinsic.name(), it)
                }
                FuncKind::Llvm { literals, declarations, body } => {
                    let lits: Vec<String> = literals
                        .iter()
                        .map(|l| match l {
                            Generic::Static(x) => format!("(static {x})"),
                            Generic::StaticStr(s) => format!("(static \"{s}\")"),
                            Generic::Type(t) => format!("(type {})", self.fmt_type(Some(*t), false)),
                        })
                        .collect();
                    format!(
                        "(llvm_func '\"{}\" {}\n(decls \"{}\")\n\"{}\"\n(literals {}))",
                        r,
                        ty,
                        escape_string(declarations),
                        escape_string(body),
                        lits.join("\n")
                    )
                }
            },
        }
    }

    fn values(&mut self, vs: &[ValueId]) -> Vec<String> {
        vs.iter().map(|v| self.fmt_value(Some(*v))).collect()
    }

    fn visit_value(&mut self, v: ValueId) -> String {
        let m = self.m;
        let v = m.actual_value(v);
        match m.kind(v) {
            ValueKind::VarValue(x) => format!("'\"{}\"", self.ref_str(x.var.into())),
            ValueKind::PointerValue(x) => {
                format!("(ptr '\"{}\" \"{}\")", self.ref_str(x.var.into()), x.fields.join("."))
            }
            ValueKind::SeriesFlow(f) => {
                let s = self.values(&f.series);
                format!("(series\n{}\n)", s.join("\n"))
            }
            ValueKind::IfFlow(f) => {
                let c = self.fmt_value(Some(f.cond));
                let t = self.fmt_value(Some(f.true_branch));
                let e = self.fmt_value(f.false_branch);
                format!("(if {c}\n{t}\n{e}\n)")
            }
            ValueKind::WhileFlow(f) => {
                let c = self.fmt_value(Some(f.cond));
                let b = self.fmt_value(Some(f.body));
                format!("(while {c}\n{b}\n)")
            }
            ValueKind::ForFlow(f) => {
                let it = self.fmt_value(Some(f.iter));
                let var = self.fmt_var(Some(f.var), false);
                let b = self.fmt_value(Some(f.body));
                format!(
                    "({}{}for {}\n{}\n{}\n)",
                    if f.is_parallel() { "par_" } else { "" },
                    if f.is_async { "async_" } else { "" },
                    it,
                    var,
                    b
                )
            }
            ValueKind::ImperativeForFlow(f) => {
                let s = self.fmt_value(Some(f.start));
                let e = self.fmt_value(Some(f.end));
                let var = self.fmt_var(Some(f.var), false);
                let b = self.fmt_value(Some(f.body));
                format!(
                    "({}imp_for {}\n{}\n{}\n{}\n{}\n)",
                    if f.is_parallel() { "par_" } else { "" },
                    s,
                    f.step,
                    e,
                    var,
                    b
                )
            }
            ValueKind::TryCatchFlow(f) => {
                let body = self.fmt_value(Some(f.body));
                let mut catches = Vec::new();
                for c in &f.catches {
                    let t = self.fmt_type(c.ty, false);
                    let var = self.fmt_var(c.var, false);
                    let h = self.fmt_value(Some(c.handler));
                    catches.push(format!("(catch {t} {var}\n{h}\n)"));
                }
                let el = self.fmt_value(f.else_);
                let fi = self.fmt_value(f.finally);
                format!("(try {}\n{}\n(else\n{}\n)\n(finally\n{})\n)", body, catches.join("\n"), el, fi)
            }
            ValueKind::PipelineFlow(f) => {
                let mut stages = Vec::new();
                for s in &f.stages {
                    let callee = self.fmt_value(Some(s.callee));
                    let args: Vec<String> = s.args.iter().map(|a| self.fmt_value(*a)).collect();
                    stages.push(format!(
                        "(stage {} {}\n(generator {})\n(parallel {}))",
                        callee,
                        args.join("\n"),
                        s.generator,
                        s.parallel
                    ));
                }
                format!("(pipeline {})", stages.join("\n"))
            }
            ValueKind::IntConst(c) => format!("{}", c.val),
            ValueKind::FloatConst(c) => format_float(c.val),
            ValueKind::BoolConst(c) => format!("{}", c.val),
            ValueKind::StringConst(c) => format!("\"{}\"", escape_string(&c.val)),
            ValueKind::BytesConst(c) => format!("b\"{}\"", escape_string(&c.val)),
            ValueKind::Custom(c) => c.node.borrow().format(m),
            ValueKind::AssignInstr(i) => {
                let l = self.fmt_var(Some(i.lhs), false);
                let r = self.fmt_value(Some(i.rhs));
                format!("(assign {l} {r})")
            }
            ValueKind::ExtractInstr(i) => {
                let val = self.fmt_value(Some(i.val));
                format!("(extract{} {} \"{}\")", if i.polymorphic { "_poly" } else { "" }, val, i.field)
            }
            ValueKind::InsertInstr(i) => {
                let l = self.fmt_value(Some(i.lhs));
                let r = self.fmt_value(Some(i.rhs));
                format!("(insert{} {} \"{}\" {})", if i.polymorphic { "_poly" } else { "" }, l, i.field, r)
            }
            ValueKind::CallInstr(i) => {
                let callee = self.fmt_value(Some(i.callee));
                let args = self.values(&i.args);
                format!("(call {}\n{}\n)", callee, args.join("\n"))
            }
            ValueKind::StackAllocInstr(i) => {
                let t = self.fmt_type(Some(i.ptr_type), false);
                format!("(stack_alloc {} {})", t, i.count)
            }
            ValueKind::YieldInInstr(i) => format!("(yield_in {})", self.fmt_type(Some(i.ty), false)),
            ValueKind::TernaryInstr(i) => {
                let c = self.fmt_value(Some(i.cond));
                let t = self.fmt_value(Some(i.true_value));
                let f = self.fmt_value(Some(i.false_value));
                format!("(select {c}\n{t}\n{f}\n)")
            }
            ValueKind::BreakInstr(i) => format!("(break {})", i.loop_.map(|l| m.value_uid(l)).unwrap_or(-1)),
            ValueKind::ContinueInstr(i) => {
                format!("(continue {})", i.loop_.map(|l| m.value_uid(l)).unwrap_or(-1))
            }
            ValueKind::ReturnInstr(i) => format!("(return {})", self.fmt_value(i.value)),
            ValueKind::YieldInstr(i) => format!("(yield {})", self.fmt_value(i.value)),
            ValueKind::AwaitInstr(i) => {
                let t = self.fmt_type(Some(i.ty), false);
                let val = self.fmt_value(Some(i.value));
                format!("(await {} {} {})", t, val, i.generator)
            }
            ValueKind::ThrowInstr(i) => {
                let val = self.fmt_value(i.value);
                match i.fallback {
                    Some(fb) => {
                        let fb = self.fmt_value(Some(fb));
                        format!("(throw {val} (fallback {fb}))")
                    }
                    None => format!("(throw {val})"),
                }
            }
            ValueKind::FlowInstr(i) => {
                let f = self.fmt_value(Some(i.flow));
                let val = self.fmt_value(Some(i.val));
                format!("(flow {f} {val})")
            }
            ValueKind::Any { .. } => "(unknown_node)".into(),
        }
    }

    fn visit_type(&mut self, t: TypeId) -> String {
        let m = self.m;
        let t = m.actual_type(t);
        let r = self.ref_str(t.into());
        match m.type_kind(t) {
            TypeKind::Int { len, sign } => format!("(int '\"{r}\" {len} (signed {sign}))"),
            TypeKind::Float => format!("(float '\"{r}\")"),
            TypeKind::Float32 => format!("(float32 '\"{r}\")"),
            TypeKind::Float16 => format!("(float16 '\"{r}\")"),
            TypeKind::BFloat16 => format!("(bfloat16 '\"{r}\")"),
            TypeKind::Float128 => format!("(float128 '\"{r}\")"),
            TypeKind::Bool => format!("(bool '\"{r}\")"),
            TypeKind::Record { fields } => {
                let fs: Vec<String> = fields
                    .iter()
                    .map(|f| format!("(\"{}\" {})", f.name, self.fmt_type(Some(f.ty), false)))
                    .collect();
                format!("(record '\"{}\" {})", r, fs.join(" "))
            }
            TypeKind::Ref { contents, .. } => format!("(ref '\"{}\" {})", r, self.fmt_type(Some(*contents), false)),
            TypeKind::Func { ret, args, variadic } => {
                let a: Vec<String> = args.iter().map(|x| self.fmt_type(Some(*x), false)).collect();
                let rt = self.fmt_type(Some(*ret), false);
                format!("(func '\"{}\" {}{} {})", r, a.join(" "), if *variadic { " ..." } else { "" }, rt)
            }
            TypeKind::Optional { base } => format!("(optional '\"{}\" {})", r, self.fmt_type(Some(*base), false)),
            TypeKind::Pointer { base } => format!("(pointer '\"{}\" {})", r, self.fmt_type(Some(*base), false)),
            TypeKind::Generator { base } => {
                format!("(generator '\"{}\" {})", r, self.fmt_type(Some(*base), false))
            }
            TypeKind::Vector { count, base } => {
                format!("(vector '\"{}\" {} (count {}))", r, self.fmt_type(Some(*base), false), count)
            }
            TypeKind::Union { types } => {
                let a: Vec<String> = types.iter().map(|x| self.fmt_type(Some(*x), false)).collect();
                format!("(union '\"{}\" {})", r, a.join(" "))
            }
            TypeKind::Custom(c) => c.format(m),
        }
    }
}

/// `util::format(node)`.
pub fn format(m: &Module, n: NodeRef) -> String {
    let mut f = Formatter { m, seen_nodes: FxHashSet::default(), seen_types: FxHashSet::default() };
    match n {
        NodeRef::Module => f.visit_module(),
        NodeRef::Value(v) => f.fmt_value(Some(v)),
        // `format(const Node*)` calls `accept` directly: full var / type output.
        NodeRef::Var(v) => {
            let uid = m.var_uid(v);
            f.seen_nodes.insert(uid);
            f.visit_var(v)
        }
        NodeRef::Type(t) => f.visit_type(t),
    }
}

/// `util::format(os, node)`.
pub fn format_to(out: &mut impl Write, m: &Module, n: NodeRef) -> std::fmt::Result {
    out.write_str(&format(m, n))
}

pub fn format_value(m: &Module, v: ValueId) -> String {
    format(m, v.into())
}
pub fn format_var(m: &Module, v: VarId) -> String {
    format(m, v.into())
}
pub fn format_type(m: &Module, t: TypeId) -> String {
    format(m, t.into())
}

/// Formats an attribute (`operator<<(ostream&, const Attribute&)`).
pub fn format_attribute(m: &Module, a: &crate::attribute::Attribute) -> String {
    use crate::attribute::Attribute as A;
    match a {
        A::SrcInfo(s) => s.to_string(),
        A::StringValue(s) => s.clone(),
        A::IntValue(i) => i.to_string(),
        A::Docstring(s) => s.clone(),
        A::StringList(v) => v.join(","),
        A::KeyValue(kv) => kv.attributes.values().cloned().collect::<Vec<_>>().join(","),
        A::Member(mm) => {
            let s: Vec<String> = mm.iter().map(|(k, v)| format!("{k}={v}")).collect();
            format!("({})", s.join(","))
        }
        A::PythonWrapper(f) => format!("(pywrap {})", m.reference_string((*f).into())),
        A::TupleLiteral(v) => {
            let s: Vec<String> = v.iter().map(|x| format(m, (*x).into())).collect();
            format!("({})", s.join(","))
        }
        A::ListLiteral(v) => {
            let s: Vec<String> =
                v.iter().map(|e| format!("{}{}", if e.star { "*" } else { "" }, format(m, e.value.into()))).collect();
            format!("[{}]", s.join(","))
        }
        A::SetLiteral(v) => {
            let s: Vec<String> =
                v.iter().map(|e| format!("{}{}", if e.star { "*" } else { "" }, format(m, e.value.into()))).collect();
            format!("set([{}])", s.join(","))
        }
        A::DictLiteral(v) => {
            let s: Vec<String> = v
                .iter()
                .map(|kv| match kv.value {
                    Some(val) => format!("{}:{}", format(m, kv.key.into()), format(m, val.into())),
                    None => format!("**{}", format(m, kv.key.into())),
                })
                .collect();
            format!("dict([{}])", s.join(","))
        }
        A::PartialFunction { name, args } => {
            let s: Vec<String> =
                args.iter().map(|a| a.map(|x| format(m, x.into())).unwrap_or_else(|| "...".into())).collect();
            format!("{}({})", name, s.join(","))
        }
        A::ExactType(t) => m.type_name(*t).to_string(),
        A::Custom(c) => c.format(),
    }
}
