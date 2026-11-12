//! Port of codon/parser/visitors/format/format.{h,cpp}: `FormatVisitor` (AST -> source-like
//! text or HTML).
//!
//! PORT-NOTE: C++ reads `cache->functions` / `cache->classes` realizations directly to
//! render `FunctionStmt` / `ClassStmt`. Here this information comes from the
//! [`FormatCacheHooks`] trait, implemented by the cache lane.

use typhon_common::strings::escape;

use crate::ast::{AstArena, Attr, ExprKind, StmtKind};
use crate::ids::{ExprId, StmtId, TypeId};

/// A realized function as seen by the formatter.
#[derive(Clone, Debug, Default)]
pub struct FormatFuncRealization {
    /// The realized function AST (`realization->ast`).
    pub ast: Option<StmtId>,
    /// Realized names of the value-argument types, in order (`(*ft)[j]->realizedName()`).
    pub arg_types: Vec<String>,
    /// `ft->getRetType()->realizedName()`.
    pub ret_type: String,
}

/// A realized class as seen by the formatter.
#[derive(Clone, Debug, Default)]
pub struct FormatClassRealization {
    /// Realized name (`real.first`).
    pub name: String,
    /// Fields to print: `(name, realized type name)` (already cut to the generics count
    /// for tuples, as C++ does).
    pub fields: Vec<(String, String)>,
}

/// Cache information needed to render functions/classes (C++ `Cache*`).
pub trait FormatCacheHooks {
    /// `cache->functions[name].realizations`, or `None` if the function is unknown.
    fn function_realizations(&self, name: &str) -> Option<Vec<FormatFuncRealization>>;
    /// `cache->classes[name].realizations`, or `None` if the class is unknown.
    fn class_realizations(&self, name: &str) -> Option<Vec<FormatClassRealization>>;
}

#[derive(Clone)]
struct Style {
    render_html: bool,
    space: String,
    header: String,
    footer: String,
    nl: String,
    type_start: String,
    type_end: String,
    stmt_start: String,
    stmt_end: String,
    expr_start: String,
    expr_end: String,
    comment_start: String,
    comment_end: String,
    keyword_start: String,
    keyword_end: String,
    literal_start: String,
    literal_end: String,
}

impl Style {
    fn new(html: bool) -> Self {
        let s = |x: &str| x.to_string();
        if html {
            Style {
                render_html: true,
                header: s("<html><head><link rel=stylesheet href=\"../code.css\"/></head>\n<body><div class=code>\n"),
                footer: s("\n</div></body></html>"),
                nl: s("<br/>"),
                type_start: s("<ast-type>"),
                type_end: s("</ast-type>"),
                stmt_start: s("<ast-stmt>"),
                stmt_end: s("</ast-stmt>"),
                expr_start: s("<ast-expr>"),
                expr_end: s("</ast-expr>"),
                comment_start: s("<ast-comment>"),
                comment_end: s("</ast-comment>"),
                literal_start: s("<ast-expr class=lit>"),
                literal_end: s("</ast-expr>"),
                keyword_start: s("<ast-keyword>"),
                keyword_end: s("</ast-keyword>"),
                space: s("&nbsp;"),
            }
        } else {
            Style {
                render_html: false,
                space: s(" "),
                header: String::new(),
                footer: String::new(),
                nl: String::new(),
                type_start: String::new(),
                type_end: String::new(),
                stmt_start: String::new(),
                stmt_end: String::new(),
                expr_start: String::new(),
                expr_end: String::new(),
                comment_start: String::new(),
                comment_end: String::new(),
                keyword_start: String::new(),
                keyword_end: String::new(),
                literal_start: String::new(),
                literal_end: String::new(),
            }
        }
    }
}

/// `FormatVisitor`.
pub struct FormatVisitor<'a> {
    arena: &'a AstArena,
    cache: Option<&'a dyn FormatCacheHooks>,
    st: Style,
    indent: i32,
    /// Render type annotations (HTML mode).
    render_type: bool,
}

impl<'a> FormatVisitor<'a> {
    pub fn new(arena: &'a AstArena, html: bool, cache: Option<&'a dyn FormatCacheHooks>) -> Self {
        Self { arena, cache, st: Style::new(html), indent: 0, render_type: html }
    }

    /// `FormatVisitor::apply(stmt, cache, html)`.
    pub fn apply_stmt(arena: &AstArena, s: StmtId, cache: Option<&dyn FormatCacheHooks>, html: bool) -> String {
        let v = FormatVisitor::new(arena, html, cache);
        format!("{}{}{}", v.st.header, v.transform_stmt(Some(s), 0), v.st.footer)
    }
    /// `FormatVisitor::apply(expr, cache, html)`.
    pub fn apply_expr(arena: &AstArena, e: ExprId, cache: Option<&dyn FormatCacheHooks>, html: bool) -> String {
        let v = FormatVisitor::new(arena, html, cache);
        format!("{}{}{}", v.st.header, v.transform_expr(Some(e)), v.st.footer)
    }

    fn anchor_root(&self, s: &str) -> String {
        if self.st.render_html { format!("<a class=\".root\" name=\"{s}\">{s}</a>") } else { s.to_string() }
    }
    fn anchor(&self, s: &str) -> String {
        if self.st.render_html { format!("<a class=\".anchor\" href=\"#{s}\">{s}</a>") } else { s.to_string() }
    }
    fn pad(&self, indent: i32) -> String {
        self.st.space.repeat(((self.indent + indent) * 2).max(0) as usize)
    }
    fn newline(&self) -> String {
        format!("{}\n", self.st.nl)
    }
    fn keyword(&self, s: &str) -> String {
        format!("{}{}{}", self.st.keyword_start, s, self.st.keyword_end)
    }
    fn literal(&self, s: &str) -> String {
        format!("{}{}{}", self.st.literal_start, s, self.st.literal_end)
    }
    fn type_name(&self, t: TypeId) -> String {
        self.arena.type_realized_name(t)
    }
    fn render_expr(&self, id: ExprId, body: String) -> String {
        let t = match self.arena[id].ty {
            Some(t) if self.render_type || self.st.render_html => {
                format!("{}{}{}", self.st.type_start, self.anchor(&self.type_name(t)), self.st.type_end)
            }
            // PORT-NOTE: C++ renders the type whenever it is set; with empty markers in
            // text mode that still appends the bare type name. Kept for fidelity.
            Some(t) => self.anchor(&self.type_name(t)),
            None => String::new(),
        };
        format!("{}{}{}{}", self.st.expr_start, body, t, self.st.expr_end)
    }

    /// `transform(Expr*)`.
    pub fn transform_expr(&self, e: Option<ExprId>) -> String {
        match e {
            Some(e) => {
                let v = FormatVisitor { arena: self.arena, cache: self.cache, st: self.st.clone(), indent: 0, render_type: self.render_type };
                v.expr(e)
            }
            None => String::new(),
        }
    }

    /// `transform(Stmt*, indent)`.
    pub fn transform_stmt(&self, s: Option<StmtId>, indent: i32) -> String {
        let v = FormatVisitor {
            arena: self.arena,
            cache: self.cache,
            st: self.st.clone(),
            indent: self.indent + indent,
            render_type: self.render_type,
        };
        let Some(s) = s else { return String::new() };
        let r = v.stmt(s);
        if r.is_empty() {
            return String::new();
        }
        let is_suite = self.arena[s].kind.as_suite().is_some();
        format!(
            "{}{}{}{}{}",
            self.st.stmt_start,
            if is_suite { String::new() } else { v.pad(0) },
            r,
            self.st.stmt_end,
            self.newline()
        )
    }

    fn items(&self, v: &[ExprId]) -> String {
        v.iter().map(|&e| self.transform_expr(Some(e))).collect::<Vec<_>>().join(", ")
    }

    fn expr(&self, id: ExprId) -> String {
        let a = self.arena;
        let t = |e: Option<ExprId>| self.transform_expr(e);
        let body = match &a[id].kind {
            ExprKind::None(_) => "None".to_string(),
            ExprKind::Bool(b) => self.literal(if b.value { "True" } else { "False" }),
            ExprKind::Int(i) => format!("{}{}", self.literal(&i.value), i.suffix),
            ExprKind::Float(i) => format!("{}{}", self.literal(&i.value), i.suffix),
            ExprKind::String(s) => self.literal(&format!("\"{}\"", escape(&s.value()))),
            ExprKind::Id(i) => {
                let is_fn = a[id].ty.is_some_and(|t| a.type_fmt.as_ref().is_some_and(|f| f.is_func(t)));
                if is_fn { self.anchor(&i.value) } else { i.value.clone() }
            }
            ExprKind::Star(e) => format!("*{}", t(e.expr)),
            ExprKind::KeywordStar(e) => format!("**{}", t(e.expr)),
            ExprKind::Tuple(e) => format!("({})", self.items(&e.items)),
            ExprKind::List(e) => format!("[{}]", self.items(&e.items)),
            ExprKind::Instantiate(e) => format!("{}⟦{}⟧", t(e.expr), self.items(&e.items)),
            ExprKind::Set(e) => format!("{{{}}}", self.items(&e.items)),
            ExprKind::Dict(e) => {
                let s: Vec<String> = e
                    .items
                    .iter()
                    .filter_map(|&i| a[i].kind.as_tuple())
                    .map(|tp| format!("{}: {}", t(tp.items.first().copied()), t(tp.items.get(1).copied())))
                    .collect();
                format!("{{{}}}", s.join(", "))
            }
            ExprKind::Generator(_) => return "GENERATOR_IMPL".into(),
            ExprKind::If(e) => format!(
                "({} {} {} {} {})",
                t(e.ifexpr),
                self.keyword("if"),
                t(e.cond),
                self.keyword("else"),
                t(e.elsexpr)
            ),
            ExprKind::Unary(e) => format!("{}{}", e.op, t(e.expr)),
            ExprKind::Binary(e) => format!("({} {} {})", t(e.lexpr), e.op, t(e.rexpr)),
            ExprKind::ChainBinary(_) => {
                // PORT-NOTE: C++ FormatVisitor has no ChainBinaryExpr overload and asserts in
                // defaultVisit; we render the S-expression instead.
                a.expr_to_string(id, -1)
            }
            ExprKind::Pipe(e) => {
                let mut items: Vec<String> = vec![];
                for p in &e.items {
                    if items.is_empty() {
                        items.push(t(p.expr));
                    } else {
                        items.push(format!("{} {}", p.op, t(p.expr)));
                    }
                }
                format!("({})", items.join(" "))
            }
            ExprKind::Index(e) => format!("{}[{}]", t(e.expr), t(e.index)),
            ExprKind::Call(e) => {
                let args: Vec<String> = e
                    .items
                    .iter()
                    .map(|i| if i.name.is_empty() { t(i.value) } else { format!("{}={}", i.name, t(i.value)) })
                    .collect();
                format!("{}({})", t(e.expr), args.join(", "))
            }
            ExprKind::Dot(e) => format!("{}.{}", t(e.expr), e.member),
            ExprKind::Slice(e) => {
                let mut s = String::new();
                if e.start.is_some() {
                    s += &t(e.start);
                }
                s += ":";
                if e.stop.is_some() {
                    s += &t(e.stop);
                }
                s += ":";
                if e.step.is_some() {
                    s += &t(e.step);
                }
                s
            }
            ExprKind::Ellipsis(_) => "...".into(),
            ExprKind::Lambda(e) => {
                let s: Vec<&str> = e.items.iter().map(|p| p.name.as_str()).collect();
                format!("{} {}: {}", self.keyword("lambda"), s.join(", "), t(e.expr))
            }
            ExprKind::Yield(_) => format!("({})", self.keyword("yield")),
            ExprKind::Await(e) => return format!("{} {}", self.keyword("await"), t(e.expr)),
            ExprKind::Stmt(e) => {
                let mut s = String::new();
                for &i in &e.items {
                    s += &format!("{}{}", self.pad(2), self.transform_stmt(Some(i), 2));
                }
                format!("《{}{}{}{}{}》", self.newline(), s, self.newline(), self.pad(2), t(e.expr))
            }
            ExprKind::Assign(e) => format!("({} := {})", t(e.var), t(e.expr)),
            ExprKind::Range(_) => {
                // PORT-NOTE: no C++ overload (asserts); render the S-expression.
                a.expr_to_string(id, -1)
            }
        };
        self.render_expr(id, body)
    }

    fn stmt(&self, id: StmtId) -> String {
        let a = self.arena;
        let te = |e: Option<ExprId>| self.transform_expr(e);
        let ts = |s: Option<StmtId>, i: i32| self.transform_stmt(s, i);
        let kw = |s: &str| self.keyword(s);
        let nl = self.newline();
        match &a[id].kind {
            StmtKind::Suite(s) => s.items.iter().map(|&x| ts(Some(x), 0)).collect(),
            StmtKind::Break(_) => kw("break"),
            StmtKind::Continue(_) => kw("continue"),
            StmtKind::Expr(s) => te(s.expr),
            StmtKind::Assign(s) => {
                if s.type_expr.is_some() {
                    format!("{}: {} = {}", te(s.lhs), te(s.type_expr), te(s.rhs))
                } else {
                    format!("{} = {}", te(s.lhs), te(s.rhs))
                }
            }
            StmtKind::AssignMember(s) => format!("{}.{} = {}", te(s.lhs), s.member, te(s.rhs)),
            StmtKind::Del(s) => format!("{} {}", kw("del"), te(s.expr)),
            StmtKind::Print(s) => format!("{} {}", kw("print"), self.items(&s.items)),
            StmtKind::Return(s) => {
                format!("{}{}", kw("return"), if s.expr.is_some() { format!(" {}", te(s.expr)) } else { String::new() })
            }
            StmtKind::Yield(s) => {
                format!("{}{}", kw("yield"), if s.expr.is_some() { format!(" {}", te(s.expr)) } else { String::new() })
            }
            StmtKind::Assert(s) => format!("{} {}", kw("assert"), te(s.expr)),
            StmtKind::While(s) => format!("{} {}:{}{}", kw("while"), te(s.cond), nl, ts(s.suite, 1)),
            StmtKind::For(s) => {
                format!("{} {} {} {}:{}{}", kw("for"), te(s.var), kw("in"), te(s.iter), nl, ts(s.suite, 1))
            }
            StmtKind::If(s) => format!(
                "{} {}:{}{}{}",
                kw("if"),
                te(s.cond),
                nl,
                ts(s.if_suite, 1),
                if s.else_suite.is_some() { format!("{}:{}{}", kw("else"), nl, ts(s.else_suite, 1)) } else { String::new() }
            ),
            StmtKind::Match(s) => {
                let mut out = String::new();
                for c in &s.items {
                    out += &format!(
                        "{}{}{}{}:{}{}",
                        self.pad(1),
                        kw("case"),
                        te(c.pattern),
                        if c.guard.is_some() { format!(" {} {}", kw("case"), te(c.guard)) } else { String::new() },
                        nl,
                        ts(c.suite, 2)
                    );
                }
                format!("{} {}:{}{}", kw("match"), te(s.expr), nl, out)
            }
            StmtKind::Import(s) => {
                let r#as = if s.as_.is_empty() { String::new() } else { format!(" {} {} ", kw("as"), s.as_) };
                if s.what.is_none() {
                    format!("{} {}{}", kw("import"), te(s.from), r#as)
                } else {
                    format!("{} {} {} {}{}", kw("from"), te(s.from), kw("import"), te(s.what), r#as)
                }
            }
            StmtKind::Try(s) => {
                let catches: Vec<String> = s
                    .items
                    .iter()
                    .filter_map(|&c| a[c].kind.as_except())
                    .map(|c| {
                        format!(
                            "{} {}{}:{}{}",
                            kw("except"),
                            te(c.exc),
                            if c.var.is_empty() { String::new() } else { format!("{} {}", kw("as"), c.var) },
                            nl,
                            ts(c.suite, 1)
                        )
                    })
                    .collect();
                format!(
                    "{}:{}{}{}{}",
                    kw("try"),
                    nl,
                    ts(s.suite, 1),
                    catches.join(""),
                    if s.finally.is_some() { format!("{}:{}{}", kw("finally"), nl, ts(s.finally, 1)) } else { String::new() }
                )
            }
            StmtKind::Except(_) => String::new(),
            StmtKind::Global(s) => format!("{} {}", kw("global"), s.var),
            StmtKind::Throw(s) => format!(
                "{} {}{}",
                kw("raise"),
                te(s.expr),
                if s.from.is_some() { format!(" {} {}", kw("from"), te(s.from)) } else { String::new() }
            ),
            StmtKind::Function(f) => self.function(f),
            StmtKind::Class(c) => self.class(id, c),
            StmtKind::YieldFrom(s) => format!("{} {}", kw("yield from"), te(s.expr)),
            StmtKind::With(_) | StmtKind::Directive(_) => String::new(),
            StmtKind::Comment(s) => format!("{}# {}{}", self.st.comment_start, s.comment, self.st.comment_end),
        }
    }

    fn function(&self, f: &crate::ast::FunctionStmt) -> String {
        let a = self.arena;
        let Some(cache) = self.cache else { return String::new() };
        let Some(reals) = cache.function_realizations(&f.name) else { return String::new() };
        if reals.is_empty() {
            return String::new();
        }
        let nl = self.newline();
        let mut result = format!("<details><summary># {}</summary>", format!("{} {}", self.keyword("def"), f.name));
        for r in reals {
            let Some(fa_id) = r.ast else { continue };
            let Some(fa) = a[fa_id].kind.as_function() else { continue };
            let mut attrs: Vec<String> = fa.decorators.iter().map(|&d| format!("@{}", self.transform_expr(Some(d)))).collect();
            if let Some(m) = a[fa_id].attrs.get_str(Attr::Module).filter(|m| !m.is_empty()) {
                attrs.push(format!("@module:{m}"));
            }
            if let Some(m) = a[fa_id].attrs.get_str(Attr::ParentClass).filter(|m| !m.is_empty()) {
                attrs.push(format!("@parent:{m}"));
            }
            let mut args = vec![];
            let mut j = 0;
            for p in &fa.items {
                if p.is_value() {
                    let tn = r.arg_types.get(j).cloned().unwrap_or_default();
                    j += 1;
                    args.push(format!(
                        "{}: {}{}",
                        p.name,
                        self.anchor(&tn),
                        p.default_value.map(|d| format!("={}", self.transform_expr(Some(d)))).unwrap_or_default()
                    ));
                }
            }
            let body = self.transform_stmt(fa.suite, 1);
            let name = self.anchor_root(&fa.name);
            result += &format!(
                "{}{}{}{} {}({}){}:{}{}",
                nl,
                self.pad(0),
                if attrs.is_empty() {
                    String::new()
                } else {
                    format!("{}{}{}", attrs.join(&format!("{nl}{}", self.pad(0))), nl, self.pad(0))
                },
                self.keyword("def"),
                self.anchor_root(&name),
                args.join(", "),
                format!(" -> {}", self.anchor(&r.ret_type)),
                nl,
                if body.is_empty() { self.keyword("pass") } else { body }
            );
        }
        result + "</details>"
    }

    fn class(&self, id: StmtId, c: &crate::ast::ClassStmt) -> String {
        let a = self.arena;
        let Some(cache) = self.cache else { return String::new() };
        let Some(reals) = cache.class_realizations(&c.name) else { return String::new() };
        if reals.is_empty() {
            return String::new();
        }
        let nl = self.newline();
        let mut result = format!(
            "<details><summary># {}</summary>",
            format!(
                "{} {} {}",
                self.keyword("class"),
                c.name,
                if a[id].attrs.has(Attr::Extend) { " +@extend" } else { "" }
            )
        );
        for r in reals {
            let args: Vec<String> = r
                .fields
                .iter()
                .map(|(n, t)| format!("{}{}: {}{}", self.st.expr_start, n, self.anchor(t), self.st.expr_end))
                .collect();
            result += &format!(
                "{}{}{}{} {}",
                nl,
                self.pad(0),
                if a[id].attrs.has(Attr::Tuple) { format!("@tuple{}{}", nl, self.pad(0)) } else { String::new() },
                self.keyword("class"),
                self.anchor_root(&r.name)
            );
            if !args.is_empty() {
                result += &format!(":{}{}{}", nl, self.pad(self.indent + 1), args.join(&format!("{nl}{}", self.pad(self.indent + 1))));
            }
        }
        result + "</details>"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::*;
    use typhon_common::SrcInfo;

    #[test]
    fn format_text() {
        let mut a = AstArena::new();
        let s = SrcInfo::default();
        let x = a.new_expr(IdExpr::new("x"), s);
        let one = a.new_expr(IntExpr::new(1), s);
        let b = a.new_expr(BinaryExpr { op: "+".into(), lexpr: Some(x), rexpr: Some(one), in_place: false }, s);
        let y = a.new_expr(IdExpr::new("y"), s);
        let asg = a.new_stmt(AssignStmt { lhs: Some(y), rhs: Some(b), type_expr: None, update: UpdateMode::Assign }, s);
        let c = a.new_expr(BoolExpr { value: true }, s);
        let w = a.new_stmt(WhileStmt { cond: Some(c), suite: Some(asg), else_suite: None, goto_var: String::new() }, s);
        let out = FormatVisitor::apply_stmt(&a, w, None, false);
        assert_eq!(out, "while True:\n  y = (x + 1)\n\n\n");
    }
}
