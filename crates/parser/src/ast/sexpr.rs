//! Port of the `toString(int indent)` S-expression printers in
//! codon/parser/ast/{expr,stmt}.cpp.

use typhon_common::strings::{escape, find_star};

use super::arena::AstArena;
use super::attr::INDENT_SIZE;
use super::expr::*;
use super::stmt::*;
use crate::ids::{ExprId, StmtId};

const NULL: &str = "<null>";

impl AstArena {
    fn es(&self, e: Option<ExprId>, indent: i32) -> String {
        // PORT-NOTE: C++ dereferences null children here (UB); we print a marker.
        e.map(|e| self.expr_to_string(e, indent)).unwrap_or_else(|| NULL.into())
    }
    fn ss(&self, s: Option<StmtId>, indent: i32) -> String {
        s.map(|s| self.stmt_to_string(s, indent)).unwrap_or_else(|| NULL.into())
    }
    /// C++ `combine(items)` (delimiter " ", indent -1).
    fn combine(&self, items: &[ExprId]) -> String {
        let mut s = String::new();
        for (i, &it) in items.iter().enumerate() {
            if i > 0 {
                s.push(' ');
            }
            s += &self.expr_to_string(it, -1);
        }
        s
    }

    /// `Param::toString`.
    pub fn param_to_string(&self, p: &Param, indent: i32) -> String {
        format!(
            "({}{}{}{})",
            p.name,
            p.ty.map(|t| format!(" #:type {}", self.expr_to_string(t, indent))).unwrap_or_default(),
            p.default_value.map(|t| format!(" #:default {}", self.expr_to_string(t, indent))).unwrap_or_default(),
            if !p.is_value() { " #:generic" } else { "" }
        )
    }

    /// `fmt::formatter<CallArg>`.
    pub fn call_arg_to_string(&self, a: &CallArg) -> String {
        format!(
            "({}{})",
            if a.name.is_empty() { String::new() } else { format!("{} = ", a.name) },
            a.value.map(|v| self.expr_to_string(v, 0)).unwrap_or_else(|| "-".into())
        )
    }

    /// `Expr::wrapType`.
    fn wrap_type(&self, id: ExprId, sexpr: String) -> String {
        let e = &self[id];
        let mut is = sexpr;
        if e.done {
            let p = find_star(&is);
            is.insert(p, '*');
        }
        let ty = match e.ty {
            Some(t) if !e.done => format!(" #:type \"{}\"", self.type_debug_string(t, 2)),
            _ => String::new(),
        };
        format!("({is}{ty})")
    }

    /// `Expr::toString(indent)`.
    pub fn expr_to_string(&self, id: ExprId, indent: i32) -> String {
        let sub = if indent >= 0 { indent + 2 * INDENT_SIZE } else { -1 };
        let s = match &self[id].kind {
            ExprKind::None(_) => "none".to_string(),
            ExprKind::Bool(b) => format!("bool {}", b.value as i32),
            ExprKind::Int(i) => format!(
                "int {}{}",
                i.value,
                if i.suffix.is_empty() { String::new() } else { format!(" #:suffix \"{}\"", i.suffix) }
            ),
            ExprKind::Float(i) => format!(
                "float {}{}",
                i.value,
                if i.suffix.is_empty() { String::new() } else { format!(" #:suffix \"{}\"", i.suffix) }
            ),
            ExprKind::String(s) => {
                let v: Vec<String> = s
                    .strings
                    .iter()
                    .map(|p| {
                        format!(
                            "\"{}\"{}",
                            escape(&p.value),
                            if p.prefix.is_empty() { String::new() } else { format!(" #:prefix \"{}\"", p.prefix) }
                        )
                    })
                    .collect();
                format!("string ({})", v.join(" "))
            }
            ExprKind::Id(i) => {
                // C++: untyped ids are printed bare (even when `done`).
                if self[id].ty.is_none() {
                    return format!("'{}", i.value);
                }
                format!("'{}", i.value)
            }
            ExprKind::Star(e) => format!("star {}", self.es(e.expr, indent)),
            ExprKind::KeywordStar(e) => format!("kwstar {}", self.es(e.expr, indent)),
            ExprKind::Tuple(e) => format!("tuple {}", self.combine(&e.items)),
            ExprKind::List(e) => {
                if e.items.is_empty() { "list".into() } else { format!("list {}", self.combine(&e.items)) }
            }
            ExprKind::Set(e) => {
                if e.items.is_empty() { "set".into() } else { format!("set {}", self.combine(&e.items)) }
            }
            // C++ prints "set" for an empty dict.
            ExprKind::Dict(e) => {
                if e.items.is_empty() { "set".into() } else { format!("dict {}", self.combine(&e.items)) }
            }
            ExprKind::Generator(g) => {
                let prefix = match g.kind {
                    GeneratorKind::ListGenerator => "list-",
                    GeneratorKind::SetGenerator => "set-",
                    GeneratorKind::DictGenerator => "dict-",
                    _ => "",
                };
                format!("{}gen {}", prefix, self.ss(g.loops, sub))
            }
            ExprKind::If(e) => format!(
                "if-expr {} {} {}",
                self.es(e.cond, indent),
                self.es(e.ifexpr, indent),
                self.es(e.elsexpr, indent)
            ),
            ExprKind::Unary(e) => format!("unary \"{}\" {}", e.op, self.es(e.expr, indent)),
            ExprKind::Binary(e) => format!(
                "binary \"{}\" {} {}{}",
                e.op,
                self.es(e.lexpr, indent),
                self.es(e.rexpr, indent),
                if e.in_place { " #:in-place" } else { "" }
            ),
            ExprKind::ChainBinary(e) => {
                let v: Vec<String> =
                    e.exprs.iter().map(|(op, x)| format!("({} \"{}\")", op, self.expr_to_string(*x, indent))).collect();
                format!("chain {}", v.join(" "))
            }
            ExprKind::Pipe(e) => {
                let v: Vec<String> =
                    e.items.iter().map(|p| format!("({} \"{}\")", self.es(p.expr, indent), p.op)).collect();
                format!("pipe {}", v.join(" "))
            }
            ExprKind::Index(e) => format!("index {} {}", self.es(e.expr, indent), self.es(e.index, indent)),
            ExprKind::Call(e) => {
                let pad = if indent >= 0 { format!("\n{}", " ".repeat(sub as usize)) } else { " ".into() };
                let mut s = String::new();
                for a in &e.items {
                    if !a.name.is_empty() {
                        s += &format!("{pad}#:name '{}", a.name);
                    }
                    s += &pad;
                    s += &self.es(a.value, sub);
                }
                format!("call{} {}{}", if e.partial { "-partial" } else { "" }, self.es(e.expr, indent), s)
            }
            ExprKind::Dot(e) => format!("dot {} '{}", self.es(e.expr, indent), e.member),
            ExprKind::Slice(e) => format!(
                "slice{}{}{}",
                e.start.map(|x| format!(" #:start {}", self.expr_to_string(x, indent))).unwrap_or_default(),
                e.stop.map(|x| format!(" #:end {}", self.expr_to_string(x, indent))).unwrap_or_default(),
                e.step.map(|x| format!(" #:step {}", self.expr_to_string(x, indent))).unwrap_or_default()
            ),
            ExprKind::Ellipsis(e) => format!(
                "ellipsis{}",
                match e.mode {
                    EllipsisType::Pipe => " #:pipe",
                    EllipsisType::Partial => " #:partial",
                    EllipsisType::Standalone => "",
                }
            ),
            ExprKind::Lambda(e) => {
                let v: Vec<String> = e.items.iter().map(|p| self.param_to_string(p, indent)).collect();
                format!("lambda ({}) {}", v.join(" "), self.es(e.expr, indent))
            }
            // C++ does not wrap YieldExpr.
            ExprKind::Yield(_) => return "yield-expr".into(),
            ExprKind::Await(e) => format!("await {}", self.es(e.expr, indent)),
            ExprKind::Assign(e) => format!("assign-expr '{} {}", self.es(e.var, indent), self.es(e.expr, indent)),
            ExprKind::Range(e) => format!("range {} {}", self.es(e.start, indent), self.es(e.stop, indent)),
            ExprKind::Stmt(e) => {
                let pad = if indent >= 0 { format!("\n{}", " ".repeat(sub as usize)) } else { " ".into() };
                let s: String = e.items.iter().map(|&i| format!("{pad}{}", self.stmt_to_string(i, sub))).collect();
                format!("stmt-expr {} ({})", self.es(e.expr, indent), s)
            }
            ExprKind::Instantiate(e) => format!("instantiate {} {}", self.es(e.expr, indent), self.combine(&e.items)),
        };
        self.wrap_type(id, s)
    }

    /// `Stmt::toString(indent)`.
    pub fn stmt_to_string(&self, id: StmtId, indent: i32) -> String {
        let st = &self[id];
        let sub = if indent >= 0 { indent + INDENT_SIZE } else { -1 };
        // `pad` used by most compound statements (`indent > 0`).
        let pad = if indent > 0 { format!("\n{}", " ".repeat((indent + INDENT_SIZE) as usize)) } else { " ".into() };
        match &st.kind {
            StmtKind::Suite(s) => {
                if indent == -1 {
                    return String::new();
                }
                let spad = if indent >= 0 { format!("\n{}", " ".repeat((indent + INDENT_SIZE) as usize)) } else { " ".into() };
                let mut out = String::new();
                for (i, &it) in s.items.iter().enumerate() {
                    let mut is = self.stmt_to_string(it, sub);
                    if self[it].done {
                        let p = find_star(&is);
                        is.insert(p, '*');
                    }
                    if i > 0 {
                        out += &spad;
                    }
                    out += &is;
                }
                format!(
                    "({}suite{})",
                    if st.done { "*" } else { "" },
                    if out.is_empty() { out } else { format!(" {spad}{out}") }
                )
            }
            StmtKind::Break(_) => "(break)".into(),
            StmtKind::Continue(_) => "(continue)".into(),
            StmtKind::Expr(s) => format!("(expr {})", self.es(s.expr, indent)),
            StmtKind::Assign(s) => format!(
                "({} {}{}{})",
                if s.update != UpdateMode::Assign { "update" } else { "assign" },
                self.es(s.lhs, indent),
                s.rhs.map(|r| format!(" {}", self.expr_to_string(r, indent))).unwrap_or_default(),
                s.type_expr.map(|r| format!(" #:type {}", self.expr_to_string(r, indent))).unwrap_or_default()
            ),
            StmtKind::Del(s) => format!("(del {})", self.es(s.expr, indent)),
            StmtKind::Print(s) => {
                format!("(print {}{})", if s.no_newline { "#:inline " } else { "" }, self.combine(&s.items))
            }
            StmtKind::Return(s) => match s.expr {
                Some(e) => format!("(return {})", self.expr_to_string(e, indent)),
                None => "(return)".into(),
            },
            StmtKind::Yield(s) => match s.expr {
                Some(e) => format!("(yield {})", self.expr_to_string(e, indent)),
                None => "(yield)".into(),
            },
            StmtKind::Assert(s) => format!(
                "(assert {}{})",
                self.es(s.expr, indent),
                s.message.map(|m| self.expr_to_string(m, indent)).unwrap_or_default()
            ),
            StmtKind::While(s) => {
                if indent == -1 {
                    return format!("(while {})", self.es(s.cond, indent));
                }
                if s.else_suite.is_some_and(|e| self.first_in_block(e).is_some()) {
                    format!(
                        "(while-else {}{}{}{}{})",
                        self.es(s.cond, indent),
                        pad,
                        self.ss(s.suite, sub),
                        pad,
                        self.ss(s.else_suite, sub)
                    )
                } else {
                    format!("(while {}{}{})", self.es(s.cond, indent), pad, self.ss(s.suite, sub))
                }
            }
            StmtKind::For(s) => {
                let vs = self.es(s.var, indent);
                if indent == -1 {
                    return format!("(for {} {})", vs, self.es(s.iter, indent));
                }
                let mut attr = String::new();
                if let Some(d) = s.decorator {
                    attr += &format!(" {}", self.expr_to_string(d, indent));
                }
                if !attr.is_empty() {
                    attr = format!(" #:attr{attr}");
                }
                if s.else_suite.is_some_and(|e| self.first_in_block(e).is_some()) {
                    format!(
                        "(for-else {} {}{}{}{}{}{})",
                        vs,
                        self.es(s.iter, indent),
                        attr,
                        pad,
                        self.ss(s.suite, sub),
                        pad,
                        self.ss(s.else_suite, sub)
                    )
                } else {
                    format!("(for {} {}{}{}{})", vs, self.es(s.iter, indent), attr, pad, self.ss(s.suite, sub))
                }
            }
            StmtKind::If(s) => {
                if indent == -1 {
                    return format!("(if {})", self.es(s.cond, indent));
                }
                format!(
                    "(if {}{}{}{})",
                    self.es(s.cond, indent),
                    pad,
                    self.ss(s.if_suite, sub),
                    s.else_suite.map(|e| format!("{pad}{}", self.stmt_to_string(e, sub))).unwrap_or_default()
                )
            }
            StmtKind::Match(s) => {
                if indent == -1 {
                    return format!("(match {})", self.es(s.expr, indent));
                }
                let pad_extra = if indent > 0 { " ".repeat(INDENT_SIZE as usize) } else { String::new() };
                let csub = if indent >= 0 { indent + INDENT_SIZE } else { -2 };
                let v: Vec<String> = s
                    .items
                    .iter()
                    .map(|c| {
                        format!(
                            "(case {}{}{}{})",
                            self.es(c.pattern, indent),
                            c.guard.map(|g| format!(" #:guard {}", self.expr_to_string(g, indent))).unwrap_or_default(),
                            format!("{pad}{pad_extra}"),
                            self.ss(c.suite, csub)
                        )
                    })
                    .collect();
                format!("(match {}{}{})", self.es(s.expr, indent), pad, v.join(&pad))
            }
            StmtKind::Import(s) => {
                let va: Vec<String> = s.args.iter().map(|a| self.param_to_string(a, indent)).collect();
                format!(
                    "(import {}{}{}{}{}{})",
                    s.from.map(|f| self.expr_to_string(f, indent)).unwrap_or_default(),
                    if s.as_.is_empty() { String::new() } else { format!(" #:as '{}", s.as_) },
                    s.what.map(|w| format!(" #:what {}", self.expr_to_string(w, indent))).unwrap_or_default(),
                    if s.dots > 0 { format!(" #:dots {}", s.dots) } else { String::new() },
                    if va.is_empty() { String::new() } else { format!(" #:args ({})", va.join(" ")) },
                    s.ret.map(|r| format!(" #:ret {}", self.expr_to_string(r, indent))).unwrap_or_default()
                )
            }
            StmtKind::Except(s) => {
                let pad_extra = if indent > 0 { " ".repeat(INDENT_SIZE as usize) } else { String::new() };
                let csub = if indent >= 0 { indent + INDENT_SIZE } else { -2 };
                format!(
                    "(catch {}{}{}{})",
                    if s.var.is_empty() { String::new() } else { format!("#:var '{}", s.var) },
                    s.exc.map(|e| format!(" #:exc {}", self.expr_to_string(e, indent))).unwrap_or_default(),
                    format!("{pad}{pad_extra}"),
                    self.ss(s.suite, csub)
                )
            }
            StmtKind::Try(s) => {
                if indent == -1 {
                    return "(try)".into();
                }
                let v: Vec<String> = s.items.iter().map(|&i| self.stmt_to_string(i, indent)).collect();
                format!(
                    "(try{}{}{}{}{}{})",
                    pad,
                    self.ss(s.suite, sub),
                    pad,
                    v.join(&pad),
                    s.else_suite.map(|e| format!("{pad}(else {})", self.stmt_to_string(e, sub))).unwrap_or_default(),
                    s.finally.map(|e| format!("{pad}(finally {})", self.stmt_to_string(e, sub))).unwrap_or_default()
                )
            }
            StmtKind::Throw(s) => format!(
                "(throw{}{})",
                s.expr.map(|e| format!(" {}", self.expr_to_string(e, indent))).unwrap_or_default(),
                s.from.map(|e| format!(" :from {}", self.expr_to_string(e, indent))).unwrap_or_default()
            ),
            StmtKind::Global(s) => format!("({} '{})", if s.non_local { "nonlocal" } else { "global" }, s.var),
            StmtKind::Function(s) => {
                let r#as: Vec<String> = s.items.iter().map(|a| self.param_to_string(a, indent)).collect();
                let dec: Vec<String> =
                    s.decorators.iter().map(|&d| format!("(dec {})", self.expr_to_string(d, indent))).collect();
                let ret = s.ret.map(|r| format!(" #:ret {}", self.expr_to_string(r, indent))).unwrap_or_default();
                if indent == -1 {
                    return format!("(fn '{} ({}){})", s.name, r#as.join(" "), ret);
                }
                format!(
                    "(fn '{} ({}){}{}{}{})",
                    s.name,
                    r#as.join(" "),
                    ret,
                    if dec.is_empty() { String::new() } else { format!(" (dec {})", dec.join(" ")) },
                    pad,
                    s.suite.map(|x| self.stmt_to_string(x, sub)).unwrap_or_else(|| "(suite)".into())
                )
            }
            StmtKind::Class(s) => {
                let bases: Vec<String> = s.base_classes.iter().map(|&b| self.expr_to_string(b, indent)).collect();
                let mut r#as = String::new();
                for (i, p) in s.items.iter().enumerate() {
                    if i > 0 {
                        r#as += &pad;
                    }
                    r#as += &self.param_to_string(p, indent);
                }
                let attr: Vec<String> =
                    s.decorators.iter().map(|&d| format!("(dec {})", self.expr_to_string(d, indent))).collect();
                if indent == -1 {
                    return format!("(class '{} ({}))", s.name, r#as);
                }
                format!(
                    "(class '{}{}{}{}{}{})",
                    s.name,
                    if bases.is_empty() { String::new() } else { format!(" (bases {})", bases.join(" ")) },
                    if attr.is_empty() { String::new() } else { format!(" (attr {})", attr.join(" ")) },
                    if r#as.is_empty() { r#as.clone() } else { format!("{pad}{}", r#as) },
                    pad,
                    s.suite.map(|x| self.stmt_to_string(x, sub)).unwrap_or_else(|| "(suite)".into())
                )
            }
            StmtKind::YieldFrom(s) => format!("(yield-from {})", self.es(s.expr, indent)),
            StmtKind::With(s) => {
                let r#as: Vec<String> = s
                    .items
                    .iter()
                    .enumerate()
                    .map(|(i, &it)| {
                        let v = s.vars.get(i).cloned().unwrap_or_default();
                        if v.is_empty() {
                            self.expr_to_string(it, indent)
                        } else {
                            format!("({} #:var '{})", self.expr_to_string(it, indent), v)
                        }
                    })
                    .collect();
                if indent == -1 {
                    return format!("(with ({}))", r#as.join(" "));
                }
                format!("(with ({}){}{})", r#as.join(" "), pad, self.ss(s.suite, sub))
            }
            StmtKind::Directive(s) => format!("(directive {} '{}')", s.key, s.value),
            StmtKind::AssignMember(s) => {
                format!("(assign-member {} {} {})", self.es(s.lhs, indent), s.member, self.es(s.rhs, indent))
            }
            StmtKind::Comment(s) => format!("(comment \"{}\")", s.comment),
        }
    }
}
