//! The `to_python_string` printers (Python-`ast`-like dump) and their helpers.

use super::arena::AstArena;
use super::attr::Attrs;
use super::expr::*;
use super::stmt::*;
use crate::ids::{ExprId, StmtId};

/// `pyQuote`.
pub fn py_quote(s: &str) -> String {
    let mut r = String::from("'");
    for c in s.chars() {
        if c == '\\' || c == '\'' {
            r.push('\\');
        }
        match c {
            '\n' => r += "\\n",
            '\r' => r += "\\r",
            '\t' => r += "\\t",
            c => r.push(c),
        }
    }
    r + "'"
}

/// `pyList`.
pub fn py_list(v: &[String], indent: i32, level: i32) -> String {
    if indent <= 0 || v.len() < 3 {
        return format!("[{}]", v.join(", "));
    }
    let pad = " ".repeat((level + indent).max(0) as usize);
    let close = " ".repeat(level.max(0) as usize);
    format!("[\n{pad}{}\n{close}]", v.join(&format!(",\n{pad}")))
}

/// `pyOperator`.
pub fn py_operator(op: &str) -> String {
    py_quote(op)
}

/// `pyAttrs`.
pub fn py_attrs(attrs: &Attrs) -> String {
    let mut a: Vec<String> = vec![];
    for (k, v) in attrs.iter() {
        let name = k.py_name();
        if name.is_empty() {
            continue;
        }
        let s = v.display();
        if s.is_empty() { a.push(name.to_string()) } else { a.push(format!("({name}, {s})")) }
    }
    a.sort();
    if a.is_empty() { String::new() } else { format!(", attributes={}", py_list(&a, -1, 0)) }
}

fn keep(f: &str) -> bool {
    !(f.ends_with('=') || f.ends_with("=''") || f.ends_with("=None") || f.ends_with("=[]") || f.ends_with("=()"))
}

/// `pyNode`.
pub fn py_node(name: &str, fields: &[String], attrs: Option<&Attrs>, show_attrs: bool, indent: i32, level: i32) -> String {
    let mut r = format!("{name}(");
    let mut j = 0;
    for f in fields.iter().filter(|f| keep(f)) {
        if j > 0 {
            r += ", ";
        }
        r += f;
        j += 1;
    }
    let pa = match attrs {
        Some(a) if show_attrs => py_attrs(a),
        _ => String::new(),
    };
    if !pa.is_empty() {
        r += if j == 0 { "" } else { "," };
        r += &pa[1..];
    }
    if indent <= 0 || j < 3 {
        return r + ")";
    }
    let pad = " ".repeat((level + indent).max(0) as usize);
    let close = " ".repeat(level.max(0) as usize);
    let mut r = format!("{name}(\n{pad}");
    let mut k = 0;
    for f in fields.iter().filter(|f| keep(f)) {
        if k > 0 {
            r += &format!(",\n{pad}");
        }
        r += f;
        k += 1;
    }
    if !pa.is_empty() {
        if k > 0 {
            r += &format!(",\n{pad}");
        }
        r += &pa[2..];
    }
    format!("{r}\n{close})")
}

impl AstArena {
    /// `pyExpr`.
    pub fn py_expr(&self, e: Option<ExprId>, a: bool, indent: i32, level: i32) -> String {
        e.map(|e| self.expr_to_python(e, a, indent, level)).unwrap_or_else(|| "None".into())
    }
    fn py_stmt(&self, s: Option<StmtId>, a: bool, indent: i32, level: i32) -> String {
        s.map(|s| self.stmt_to_python(s, a, indent, level)).unwrap_or_else(|| "[]".into())
    }

    /// `pyArguments`.
    pub fn py_arguments(&self, params: &[Param], a: bool, indent: i32, level: i32) -> String {
        let mut vararg = String::new();
        let mut kwarg = String::new();
        let (mut pos, mut args, mut kwonly, mut kwdef, mut def, mut typ, mut typdef) =
            (vec![], vec![], vec![], vec![], vec![], vec![], vec![]);
        let arg = |name: &str, p: &Param| {
            py_node(
                "Arg",
                &[format!("arg={}", py_quote(name)), format!("annotation={}", self.py_expr(p.ty, a, indent, indent + level))],
                None,
                a,
                indent,
                level + indent,
            )
        };
        for p in params {
            if let Some(n) = p.name.strip_prefix("**") {
                kwarg = arg(n, p);
            } else if let Some(n) = p.name.strip_prefix('*') {
                vararg = arg(n, p);
            } else if p.status != ParamStatus::Value {
                typ.push(arg(&p.name, p));
                if let Some(d) = p.default_value {
                    typdef.push(self.expr_to_python(d, a, indent, indent + level));
                }
            } else {
                let v = arg(&p.name, p);
                match p.kind {
                    ParamKind::PositionalOnly => pos.push(v),
                    ParamKind::KeywordOnly => kwonly.push(v),
                    ParamKind::Normal => args.push(v),
                }
                if p.kind == ParamKind::KeywordOnly {
                    kwdef.push(self.py_expr(p.default_value, a, indent, indent + level));
                } else if let Some(d) = p.default_value {
                    def.push(self.expr_to_python(d, a, indent, indent + level));
                }
            }
        }
        let l = indent + level;
        py_node(
            "Arguments",
            &[
                format!("posonlyargs={}", py_list(&pos, indent, l)),
                format!("args={}", py_list(&args, indent, l)),
                format!("vararg={vararg}"),
                format!("kwonlyargs={}", py_list(&kwonly, indent, l)),
                format!("kw_defaults={}", py_list(&kwdef, indent, l)),
                format!("kwarg={kwarg}"),
                format!("defaults={}", py_list(&def, indent, l)),
                format!("types={}", py_list(&typ, indent, l)),
                format!("type_defaults={}", py_list(&typdef, indent, l)),
            ],
            None,
            a,
            indent,
            l,
        )
    }

    /// `Expr::toPythonString(attributes, indent, level)`.
    pub fn expr_to_python(&self, id: ExprId, a: bool, indent: i32, level: i32) -> String {
        let node = &self[id];
        let at = Some(&node.attrs);
        let sub = level + indent;
        let pe = |e: Option<ExprId>| self.py_expr(e, a, indent, sub);
        let items = |v: &[ExprId]| -> Vec<String> { v.iter().map(|&x| self.expr_to_python(x, a, indent, sub)).collect() };
        let n = |name: &str, fields: Vec<String>| py_node(name, &fields, at, a, indent, level);
        match &node.kind {
            ExprKind::None(_) => n("NoneValue", vec![]),
            ExprKind::Bool(b) => n("Bool", vec![format!("value={}", if b.value { "True" } else { "False" })]),
            ExprKind::Int(i) => n("Num", vec![format!("value={}", i.value), format!("suffix={}", i.suffix)]),
            ExprKind::Float(i) => n("Num", vec![format!("value={}", i.value), format!("suffix={}", i.suffix)]),
            ExprKind::String(s) => {
                let vals: Vec<String> = s
                    .strings
                    .iter()
                    .map(|p| {
                        let st = n("Str", vec![format!("value={}", py_quote(&p.value)), format!("prefix={}", py_quote(&p.prefix))]);
                        if p.format.conversion.is_empty() && p.format.spec.is_empty() {
                            st
                        } else {
                            n(
                                "FormattedValue",
                                vec![
                                    format!("value={st}"),
                                    format!("conversion={}", py_quote(&p.format.conversion)),
                                    format!("format_spec={}", py_quote(&p.format.spec)),
                                ],
                            )
                        }
                    })
                    .collect();
                if vals.len() == 1 {
                    vals.into_iter().next().unwrap_or_default()
                } else {
                    n("JoinedStr", vec![format!("values={}", py_list(&vals, indent, sub))])
                }
            }
            ExprKind::Id(i) => n("Name", vec![format!("id={}", py_quote(&i.value))]),
            ExprKind::Star(e) => n("Starred", vec![format!("value={}", pe(e.expr))]),
            ExprKind::KeywordStar(e) => n("keyword", vec!["arg=None".into(), format!("value={}", pe(e.expr))]),
            ExprKind::Tuple(e) => n("TupleEx", vec![format!("elts={}", py_list(&items(&e.items), indent, sub))]),
            ExprKind::List(e) => n("ListEx", vec![format!("elts={}", py_list(&items(&e.items), indent, sub))]),
            ExprKind::Set(e) => n("SetEx", vec![format!("elts={}", py_list(&items(&e.items), indent, sub))]),
            ExprKind::Dict(e) => {
                let (mut k, mut v) = (vec![], vec![]);
                for &x in &e.items {
                    match &self[x].kind {
                        ExprKind::Tuple(t) => {
                            k.push(pe(t.items.first().copied()));
                            v.push(pe(t.items.get(1).copied()));
                        }
                        ExprKind::KeywordStar(s) => {
                            k.push("None".into());
                            v.push(pe(s.expr));
                        }
                        _ => {}
                    }
                }
                n("DictEx", vec![format!("keys={}", py_list(&k, indent, sub)), format!("values={}", py_list(&v, indent, sub))])
            }
            ExprKind::Generator(g) => self.generator_to_python(id, g, a, indent, level),
            ExprKind::If(e) => n(
                "IfExp",
                vec![format!("test={}", pe(e.cond)), format!("body={}", pe(e.ifexpr)), format!("orelse={}", pe(e.elsexpr))],
            ),
            ExprKind::Unary(e) => n("UnaryOp", vec![format!("op={}", py_operator(&e.op)), format!("operand={}", pe(e.expr))]),
            ExprKind::Binary(e) => {
                let vals: Vec<String> = [e.lexpr, e.rexpr].iter().flatten().map(|&x| self.expr_to_python(x, a, indent, sub)).collect();
                n("BinOp", vec![format!("op={}", py_operator(&e.op)), format!("values={}", py_list(&vals, indent, sub))])
            }
            ExprKind::ChainBinary(e) => {
                let mut prev = String::new();
                if e.exprs.len() >= 2 {
                    for i in (0..e.exprs.len() - 1).rev() {
                        let first = self.expr_to_python(e.exprs[i].1, a, indent, sub);
                        let second = if prev.is_empty() { self.expr_to_python(e.exprs[i + 1].1, a, indent, sub) } else { prev.clone() };
                        prev = n(
                            "BinOp",
                            vec![format!("op={}", py_operator(&e.exprs[i].0)), format!("values={}", py_list(&[first, second], indent, sub))],
                        );
                    }
                }
                prev
            }
            ExprKind::Pipe(e) => {
                let vals: Vec<String> = e.items.iter().map(|p| format!("({}, {})", py_quote(&p.op), pe(p.expr))).collect();
                n(
                    "PipeOp",
                    vec![
                        format!("head={}", pe(e.items.first().and_then(|p| p.expr))),
                        format!("values={}", py_list(&vals, indent, sub)),
                    ],
                )
            }
            ExprKind::Index(e) => n("Subscript", vec![format!("value={}", pe(e.expr)), format!("slice={}", pe(e.index))]),
            ExprKind::Call(e) => {
                let (mut v, mut kv) = (vec![], vec![]);
                for x in &e.items {
                    if x.name.is_empty() {
                        v.push(pe(x.value));
                    } else {
                        kv.push(n("keyword", vec![format!("arg={}", py_quote(&x.name)), format!("value={}", pe(x.value))]));
                    }
                }
                n(
                    if e.partial { "PartialCall" } else { "Call" },
                    vec![
                        format!("func={}", pe(e.expr)),
                        format!("args={}", py_list(&v, indent, sub)),
                        format!("keywords={}", py_list(&kv, indent, sub)),
                    ],
                )
            }
            ExprKind::Dot(e) => n("Attribute", vec![format!("value={}", pe(e.expr)), format!("attr={}", py_quote(&e.member))]),
            ExprKind::Slice(e) => n(
                "Slice",
                vec![format!("lower={}", pe(e.start)), format!("upper={}", pe(e.stop)), format!("step={}", pe(e.step))],
            ),
            ExprKind::Ellipsis(_) => n("Ellipsis", vec![]),
            ExprKind::Lambda(e) => n(
                "Lambda",
                vec![format!("args={}", self.py_arguments(&e.items, a, indent, sub)), format!("body={}", pe(e.expr))],
            ),
            ExprKind::Yield(_) => n("Yield", vec!["value=None".into()]),
            ExprKind::Await(e) => n("Await", vec![format!("value={}", pe(e.expr))]),
            ExprKind::Assign(e) => n("NamedExpr", vec![format!("target={}", pe(e.var)), format!("value={}", pe(e.expr))]),
            ExprKind::Range(e) => n("Range", vec![format!("start={}", pe(e.start)), format!("stop={}", pe(e.stop))]),
            ExprKind::Stmt(e) => {
                let vals: Vec<String> = e.items.iter().map(|&s| self.stmt_to_python(s, a, indent, sub)).collect();
                n("StmtExpr", vec![format!("elts={}", py_list(&vals, indent, sub)), format!("value={}", pe(e.expr))])
            }
            ExprKind::Instantiate(e) => {
                n("Subscript", vec![format!("value={}", pe(e.expr)), format!("slice={}", py_list(&items(&e.items), indent, sub))])
            }
        }
    }

    fn generator_to_python(&self, id: ExprId, g: &GeneratorExpr, a: bool, indent: i32, level: i32) -> String {
        let name = match g.kind {
            GeneratorKind::ListGenerator => "ListComp",
            GeneratorKind::SetGenerator => "SetComp",
            GeneratorKind::DictGenerator => "DictComp",
            _ => "GeneratorExp",
        };
        let mut gens = vec![];
        let (mut target, mut iter, mut expr): (Option<ExprId>, Option<ExprId>, Option<ExprId>) = (None, None, None);
        let mut ifs: Vec<String> = vec![];
        let l2 = 2 * indent + level;
        let comp = |target: Option<ExprId>, iter: Option<ExprId>, ifs: &[String]| {
            py_node(
                "Comprehension",
                &[
                    format!("target={}", self.py_expr(target, a, indent, l2)),
                    format!("iter={}", self.py_expr(iter, a, indent, l2)),
                    format!("ifs={}", py_list(ifs, indent, l2)),
                ],
                iter.map(|i| &self[i].attrs),
                a,
                indent,
                indent + level,
            )
        };
        let mut i = g.loops;
        while let Some(s) = i {
            match &self[s].kind {
                StmtKind::For(f) => {
                    if iter.is_some() {
                        gens.push(comp(target, iter, &ifs));
                        ifs.clear();
                    }
                    target = f.var;
                    iter = f.iter;
                    i = f.suite;
                }
                StmtKind::If(f) => {
                    ifs.push(self.py_expr(f.cond, a, indent, indent + level));
                    i = f.if_suite;
                }
                StmtKind::Suite(ss) => i = ss.items.last().copied(),
                StmtKind::Expr(se) => {
                    gens.push(comp(target, iter, &ifs));
                    ifs.clear();
                    expr = se.expr;
                    break;
                }
                _ => break,
            }
        }
        typhon_common::seqassert!(expr.is_some(), "bad generator");
        py_node(
            name,
            &[
                format!("elt={}", self.py_expr(expr, a, indent, indent + level)),
                format!("generators={}", py_list(&gens, indent, indent + level)),
            ],
            Some(&self[id].attrs),
            a,
            indent,
            level,
        )
    }

    /// Dotted name of an import path (`a.b.c`).
    fn dotted(&self, mut e: Option<ExprId>) -> String {
        let mut comps = vec![];
        while let Some(x) = e {
            match &self[x].kind {
                ExprKind::Dot(d) => {
                    comps.push(d.member.clone());
                    e = d.expr;
                }
                ExprKind::Id(i) => {
                    comps.push(i.value.clone());
                    break;
                }
                _ => break,
            }
        }
        comps.reverse();
        comps.join(".")
    }

    /// `Stmt::toPythonString(attributes, indent, level)`.
    pub fn stmt_to_python(&self, id: StmtId, a: bool, indent: i32, level: i32) -> String {
        let node = &self[id];
        let at = Some(&node.attrs);
        let sub = level + indent;
        let pe = |e: Option<ExprId>| self.py_expr(e, a, indent, sub);
        let ps = |s: Option<StmtId>| self.py_stmt(s, a, indent, sub);
        let n = |name: &str, fields: Vec<String>| py_node(name, &fields, at, a, indent, level);
        match &node.kind {
            StmtKind::Suite(_) => {
                let mut v = vec![];
                fn f(ar: &AstArena, s: StmtId, v: &mut Vec<String>, a: bool, indent: i32, sub: i32) {
                    if let Some(ss) = ar[s].kind.as_suite() {
                        for &x in &ss.items {
                            if ar[x].kind.as_suite().is_some() {
                                f(ar, x, v, a, indent, sub);
                            } else {
                                v.push(ar.stmt_to_python(x, a, indent, sub));
                            }
                        }
                    }
                }
                f(self, id, &mut v, a, indent, sub);
                py_list(&v, indent, sub)
            }
            StmtKind::Break(_) => n("Break", vec![]),
            StmtKind::Continue(_) => n("Continue", vec![]),
            StmtKind::Expr(s) => n("Expr", vec![format!("value={}", pe(s.expr))]),
            StmtKind::Assign(s) => {
                if let Some(b) = s.rhs.and_then(|r| self[r].kind.as_binary()).filter(|b| b.in_place) {
                    n(
                        "AugAssign",
                        vec![format!("target={}", pe(s.lhs)), format!("op={}", py_quote(&b.op)), format!("value={}", pe(b.rexpr))],
                    )
                } else if s.type_expr.is_none() {
                    n("Assign", vec![format!("targets={}", py_list(&[pe(s.lhs)], indent, sub)), format!("value={}", pe(s.rhs))])
                } else {
                    n(
                        "AnnAssign",
                        vec![format!("target={}", pe(s.lhs)), format!("annotation={}", pe(s.type_expr)), format!("value={}", pe(s.rhs))],
                    )
                }
            }
            StmtKind::Del(s) => n("Delete", vec![format!("targets={}", py_list(&[pe(s.expr)], indent, sub))]),
            StmtKind::Print(s) => {
                let mut v: Vec<String> = s.items.iter().map(|&x| self.expr_to_python(x, a, indent, sub)).collect();
                if s.no_newline {
                    v.push("end=' '".into());
                }
                let call = n("Call", vec!["func=Name(id='print')".into(), format!("args={}", py_list(&v, indent, sub))]);
                n("Expr", vec![format!("value={call}")])
            }
            StmtKind::Return(s) => n("Return", vec![format!("value={}", pe(s.expr))]),
            StmtKind::Yield(s) => {
                let y = n("Yield", vec![format!("value={}", pe(s.expr))]);
                n("Expr", vec![format!("value={y}")])
            }
            StmtKind::YieldFrom(s) => {
                let y = n("YieldFrom", vec![format!("value={}", pe(s.expr))]);
                n("Expr", vec![format!("value={y}")])
            }
            StmtKind::Assert(s) => n("Assert", vec![format!("test={}", pe(s.expr)), format!("msg={}", pe(s.message))]),
            StmtKind::While(s) => n(
                "While",
                vec![format!("test={}", pe(s.cond)), format!("body={}", ps(s.suite)), format!("orelse={}", ps(s.else_suite))],
            ),
            StmtKind::For(s) => n(
                if s.async_ { "AsyncFor" } else { "For" },
                vec![
                    format!("target={}", pe(s.var)),
                    format!("iter={}", pe(s.iter)),
                    format!("body={}", ps(s.suite)),
                    format!("orelse={}", ps(s.else_suite)),
                ],
            ),
            StmtKind::If(s) => n(
                "If",
                vec![format!("test={}", pe(s.cond)), format!("body={}", ps(s.if_suite)), format!("orelse={}", ps(s.else_suite))],
            ),
            StmtKind::Match(s) => {
                let l = 2 * level + indent;
                let v: Vec<String> = s
                    .items
                    .iter()
                    .map(|c| {
                        py_node(
                            "MatchCase",
                            &[
                                format!("pattern={}", self.py_expr(c.pattern, a, indent, l)),
                                format!("guard={}", self.py_expr(c.guard, a, indent, l)),
                                format!("body={}", self.py_stmt(c.suite, a, indent, l)),
                            ],
                            at,
                            a,
                            indent,
                            level + indent,
                        )
                    })
                    .collect();
                n("Match", vec![format!("subject={}", pe(s.expr)), format!("cases={}", py_list(&v, indent, sub))])
            }
            StmtKind::Import(s) => {
                let name = self.dotted(s.what);
                let alias = py_node(
                    "Alias",
                    &[
                        format!("name={name}"),
                        format!("asname={}", py_quote(&s.as_)),
                        format!("params={}", self.py_arguments(&s.args, a, indent, 4 * indent + level)),
                        format!("ret={}", self.py_expr(s.ret, a, indent, 4 * indent + level)),
                    ],
                    None,
                    a,
                    indent,
                    3 * indent + level,
                );
                let names = format!("names={}", py_list(&[alias], indent, 2 * indent + level));
                let wat = s.what.map(|w| &self[w].attrs);
                if s.from.is_none() {
                    py_node("Import", &[names], wat, a, indent, indent + level)
                } else {
                    let m = self.dotted(s.from);
                    py_node("ImportFrom", &[format!("module={m}"), names, format!("level={}", s.dots)], wat, a, indent, indent + level)
                }
            }
            StmtKind::Except(s) => n(
                "ExceptHandler",
                vec![format!("type={}", pe(s.exc)), format!("name={}", py_quote(&s.var)), format!("body={}", ps(s.suite))],
            ),
            StmtKind::Try(s) => {
                let v: Vec<String> = s.items.iter().map(|&x| self.stmt_to_python(x, a, indent, indent + level)).collect();
                n(
                    "Try",
                    vec![
                        format!("body={}", ps(s.suite)),
                        format!("handlers={}", py_list(&v, indent, sub)),
                        format!("orelse={}", ps(s.else_suite)),
                        format!("finalbody={}", ps(s.finally)),
                    ],
                )
            }
            StmtKind::Throw(s) => n("Raise", vec![format!("exc={}", pe(s.expr)), format!("cause={}", pe(s.from))]),
            StmtKind::Global(s) => {
                n(if s.non_local { "Nonlocal" } else { "Global" }, vec![format!("names=[{}]", py_quote(&s.var))])
            }
            StmtKind::Function(s) => {
                let l2 = level + 2 * indent;
                let dv: Vec<String> = s.decorators.iter().map(|&d| self.expr_to_python(d, a, indent, indent + level)).collect();
                py_node(
                    if s.async_ { "AsyncFunctionDef" } else { "FunctionDef" },
                    &[
                        format!("name={}", py_quote(&s.name)),
                        format!("args={}", self.py_arguments(&s.items, a, indent, l2)),
                        format!("body={}", self.py_stmt(s.suite, a, indent, l2)),
                        format!("decorator_list={}", py_list(&dv, indent, l2)),
                        format!("returns={}", self.py_expr(s.ret, a, indent, l2)),
                    ],
                    at,
                    a,
                    indent,
                    indent + level,
                )
            }
            StmtKind::Class(s) => {
                let b: Vec<String> = s.base_classes.iter().map(|&x| self.expr_to_python(x, a, indent, sub)).collect();
                let dv: Vec<String> = s.decorators.iter().map(|&d| self.expr_to_python(d, a, indent, indent + level)).collect();
                n(
                    "ClassDef",
                    vec![
                        format!("name={}", py_quote(&s.name)),
                        format!("bases={}", py_list(&b, indent, sub)),
                        format!("keywords={}", self.py_arguments(&s.items, a, indent, sub)),
                        format!("body={}", ps(s.suite)),
                        format!("decorator_list={}", py_list(&dv, indent, sub)),
                    ],
                )
            }
            StmtKind::With(s) => {
                let v: Vec<String> = s
                    .items
                    .iter()
                    .enumerate()
                    .map(|(i, &it)| {
                        n(
                            "WithItem",
                            vec![
                                format!("context_expr={}", self.expr_to_python(it, a, indent, sub)),
                                format!("optional_vars={}", py_quote(s.vars.get(i).map(String::as_str).unwrap_or(""))),
                            ],
                        )
                    })
                    .collect();
                n(
                    if s.async_ { "AsyncWith" } else { "With" },
                    vec![format!("items={}", py_list(&v, indent, sub)), format!("body={}", ps(s.suite))],
                )
            }
            StmtKind::Directive(s) => {
                n("Directive", vec![format!("key={}", py_quote(&s.key)), format!("value={}", py_quote(&s.value))])
            }
            StmtKind::AssignMember(s) => {
                let target = n("Attribute", vec![format!("value={}", pe(s.lhs)), format!("attr={}", py_quote(&s.member))]);
                n("Assign", vec![format!("targets={}", py_list(&[target], indent, sub)), format!("value={}", pe(s.rhs))])
            }
            StmtKind::Comment(s) => n("Comment", vec![format!("value={}", py_quote(&s.comment))]),
        }
    }
}
