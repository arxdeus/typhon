//! Port of codon/parser/peg/rules.h (`ParseContext`) plus the helper functions of the
//! grammar PREAMBLE (`setSI`, `chain`, `wrap_tuple`) and the hand-written cpp-peglib
//! pieces of codon/parser/peg/peg.cpp (`NLP` predicate, parens enter/leave hooks,
//! INDENT/SAMEDENT/DEDENT/EXTERNDENT predicates and actions, error reporting).

use std::cell::{Cell, RefCell};
use std::collections::{HashMap, HashSet};

use peg::RuleResult;
use typhon_common::{CompilerError, ErrorMessage, ParserErrors, SrcInfo};

use crate::ast::*;

/// Custom keywords registered by DSL plugins (C++ `cache->customBlockStmts` /
/// `cache->customExprStmts`). The cache lane owns the data and passes it in.
#[derive(Clone, Debug, Default)]
pub struct ParseOptions {
    /// keyword -> whether the block statement takes an expression (`kw expr:` vs `kw:`).
    pub custom_block_stmts: HashMap<String, bool>,
    pub custom_expr_stmts: HashSet<String>,
}

/// `ParseContext`: indentation stack, parens counter, offsets, plus the AST builder.
pub struct ParseState<'a> {
    arena: RefCell<&'a mut AstArena>,
    opts: &'a ParseOptions,
    pub file: String,
    pub line_offset: i32,
    pub col_offset: i32,
    pub indent: RefCell<Vec<usize>>,
    pub parens: Cell<i32>,
    /// cpp-peglib `error_info.message_pos` / `message` (predicate messages).
    pub message: RefCell<Option<(usize, String)>>,
    /// First exception thrown by an AST constructor inside an action (C++ throws
    /// `ParserException` straight out of the parser).
    pub fatal: RefCell<Option<CompilerError>>,
    line_starts: Vec<usize>,
    src: &'a str,
}

/// Fails without touching the error state (used for cpp-peglib rule predicates whose
/// failures do not contribute expected tokens).
pub fn cond<T>(v: Option<T>) -> impl FnOnce(&str, usize) -> RuleResult<T> {
    move |_, p| match v {
        Some(v) => RuleResult::Matched(p, v),
        None => RuleResult::Failed,
    }
}

impl<'a> ParseState<'a> {
    pub fn new(
        arena: &'a mut AstArena,
        opts: &'a ParseOptions,
        file: &str,
        src: &'a str,
        line_offset: i32,
        col_offset: i32,
    ) -> Self {
        let mut line_starts = vec![0];
        for (i, b) in src.bytes().enumerate() {
            if b == b'\n' {
                line_starts.push(i + 1);
            }
        }
        Self {
            arena: RefCell::new(arena),
            opts,
            file: file.to_string(),
            line_offset,
            col_offset,
            indent: RefCell::new(vec![]),
            parens: Cell::new(0),
            message: RefCell::new(None),
            fatal: RefCell::new(None),
            line_starts,
            src,
        }
    }

    /// Reset per-parse state (each cpp-peglib retry starts with a fresh `ParseContext`).
    pub fn reset(&self) {
        self.indent.borrow_mut().clear();
        self.parens.set(0);
        *self.message.borrow_mut() = None;
    }

    /// peglib `line_info`: 1-based line, 1-based column in code points.
    pub fn line_info(&self, pos: usize) -> (usize, usize) {
        let li = match self.line_starts.binary_search(&pos) {
            Ok(i) => i,
            Err(i) => i - 1,
        };
        let start = self.line_starts[li];
        let col = self.src[start..pos].chars().count() + 1;
        (li + 1, col)
    }

    /// C++ `LOC`.
    pub fn loc(&self, sp: (usize, usize)) -> SrcInfo {
        let (l, c) = self.line_info(sp.0);
        SrcInfo::new(
            &self.file,
            l as i32 + self.line_offset,
            c as i32 + self.col_offset,
            (sp.1 - sp.0) as i32,
        )
    }

    pub fn arena(&self) -> std::cell::RefMut<'_, &'a mut AstArena> {
        self.arena.borrow_mut()
    }

    pub fn e(&self, k: impl Into<ExprKind>, sp: (usize, usize)) -> ExprId {
        let src = self.loc(sp);
        self.arena.borrow_mut().new_expr(k, src)
    }
    pub fn s(&self, k: impl Into<StmtKind>, sp: (usize, usize)) -> StmtId {
        let src = self.loc(sp);
        self.arena.borrow_mut().new_stmt(k, src)
    }
    pub fn id(&self, name: &str, sp: (usize, usize)) -> ExprId {
        self.e(IdExpr::new(name), sp)
    }
    /// `asts(Suite, LOC, items)` followed by `flatten()`.
    pub fn suite(&self, items: Vec<StmtId>, sp: (usize, usize)) -> StmtId {
        let s = self.s(SuiteStmt { items }, sp);
        self.arena.borrow_mut().suite_flatten(s);
        s
    }
    pub fn wrap(&self, s: Option<StmtId>) -> Option<StmtId> {
        self.arena.borrow_mut().wrap_suite(s)
    }
    pub fn binary(&self, l: ExprId, op: &str, r: ExprId, in_place: bool, sp: (usize, usize)) -> ExprId {
        self.e(
            BinaryExpr { op: op.to_string(), lexpr: Some(l), rexpr: Some(r), in_place },
            sp,
        )
    }
    /// PREAMBLE `chain`.
    pub fn chain(&self, first: ExprId, rest: Vec<(&str, ExprId)>, sp: (usize, usize)) -> ExprId {
        let mut b = first;
        for (op, r) in rest {
            b = self.binary(b, op, r, false, sp);
        }
        b
    }
    /// PREAMBLE `wrap_tuple`.
    pub fn wrap_tuple(&self, (items, tok): (Vec<ExprId>, bool), sp: (usize, usize)) -> ExprId {
        if items.len() == 1 && !tok {
            return items[0];
        }
        self.e(TupleExpr { items }, sp)
    }
    /// `Param(LOC, name, type, default)` (status inferred).
    pub fn param(&self, sp: (usize, usize), name: &str, ty: Option<ExprId>, def: Option<ExprId>) -> Param {
        let src = self.loc(sp);
        self.arena.borrow().new_param(src, name, ty, def, 0, ParamKind::Normal)
    }
    /// `CallArg(Expr*)`: source copied from the value.
    pub fn arg(&self, v: ExprId) -> CallArg {
        self.arena.borrow().call_arg("", Some(v))
    }
    /// `CallArg(name, value)`: source copied from the value.
    pub fn narg(&self, name: &str, v: Option<ExprId>) -> CallArg {
        self.arena.borrow().call_arg(name, v)
    }
    /// `CallArg(LOC, name, value)`.
    pub fn arg_at(&self, sp: (usize, usize), name: &str, v: ExprId) -> CallArg {
        let mut a = CallArg::new(name, Some(v));
        a.src = self.loc(sp);
        a
    }
    pub fn generator(&self, kind: GeneratorKind, e: ExprId, loops: Vec<StmtId>, sp: (usize, usize)) -> ExprId {
        let src = self.loc(sp);
        self.arena.borrow_mut().new_generator(src, kind, e, loops)
    }
    pub fn dict_generator(&self, kv: ExprId, loops: Vec<StmtId>, sp: (usize, usize)) -> ExprId {
        let src = self.loc(sp);
        let mut a = self.arena.borrow_mut();
        let items = a[kv].kind.as_tuple().map(|t| t.items.clone()).unwrap_or_default();
        a.new_dict_generator(src, items[0], items[1], loops)
    }
    pub fn clone_expr(&self, e: ExprId) -> ExprId {
        self.arena.borrow_mut().clone_expr(e, false)
    }
    pub fn set_fatal(&self, e: CompilerError) {
        let mut f = self.fatal.borrow_mut();
        if f.is_none() {
            *f = Some(e);
        }
    }
    pub fn lambda(&self, params: Vec<Param>, body: ExprId, sp: (usize, usize)) -> ExprId {
        let src = self.loc(sp);
        let r = self.arena.borrow_mut().new_lambda(src, params, Some(body));
        match r {
            Ok(e) => e,
            Err(err) => {
                self.set_fatal(err);
                self.e(LambdaExpr { items: vec![], expr: Some(body) }, sp)
            }
        }
    }
    pub fn function(&self, name: String, ret: Option<ExprId>, params: Vec<Param>, sp: (usize, usize)) -> StmtId {
        let src = self.loc(sp);
        let r = self.arena.borrow_mut().new_function(src, name.clone(), ret, params, None, vec![], false);
        match r {
            Ok(s) => s,
            Err(err) => {
                self.set_fatal(err);
                let src = self.loc(sp);
                self.arena.borrow_mut().new_function(src, name, ret, vec![], None, vec![], false).unwrap()
            }
        }
    }

    // ----------------------------------------------------------------- predicates

    /// Records a predicate message (cpp-peglib: only if `message_pos < s`).
    pub fn set_message(&self, pos: usize, msg: &str) {
        let mut m = self.message.borrow_mut();
        if m.as_ref().is_none_or(|(p, _)| *p < pos) {
            *m = Some((pos, msg.to_string()));
        }
    }
    pub fn has_custom_stmt_keyword(&self, kwd: &str, has_expr: bool) -> bool {
        self.opts.custom_block_stmts.get(kwd).is_some_and(|&h| h == has_expr)
    }
    pub fn has_custom_expr_stmt(&self, kwd: &str) -> bool {
        self.opts.custom_expr_stmts.contains(kwd)
    }

    /// `NLP` user rule: `\` consumes one char; inside parentheses an empty match.
    pub fn nlp(&self) -> impl FnOnce(&str, usize) -> RuleResult<()> + '_ {
        move |i: &str, p: usize| {
            if i.as_bytes().get(p) == Some(&b'\\') {
                RuleResult::Matched(p + 1, ())
            } else if self.parens.get() != 0 {
                RuleResult::Matched(p, ())
            } else {
                RuleResult::Failed
            }
        }
    }
    pub fn inc_parens(&self) {
        self.parens.set(self.parens.get() + 1);
    }
    pub fn dec_parens(&self) {
        self.parens.set(self.parens.get() - 1);
    }

    /// `INDENT` predicate + action (push).
    pub fn indent_pred(&self, start: usize, n: usize) -> Option<()> {
        let mut st = self.indent.borrow_mut();
        if !(st.is_empty() && n > 0) && !(!st.is_empty() && n > *st.last().unwrap()) {
            drop(st);
            self.set_message(start, "unexpected indentation");
            return None;
        }
        st.push(n);
        Some(())
    }
    /// `SAMEDENT` predicate.
    pub fn samedent_pred(&self, n: usize) -> Option<()> {
        let st = self.indent.borrow();
        (!(st.is_empty() && n > 0) && !(!st.is_empty() && n != *st.last().unwrap())).then_some(())
    }
    /// `DEDENT` predicate + action (pop).
    pub fn dedent_pred(&self, start: usize, n: usize) -> Option<()> {
        let mut st = self.indent.borrow_mut();
        if !(!st.is_empty() && n < *st.last().unwrap()) {
            drop(st);
            self.set_message(start, "unexpected dedent");
            return None;
        }
        st.pop();
        Some(())
    }
    /// `EXTERNDENT` predicate.
    pub fn externdent_pred(&self, n: usize) -> Option<()> {
        let st = self.indent.borrow();
        (!(st.is_empty() && n > 0) && !(!st.is_empty() && n < *st.last().unwrap())).then_some(())
    }

    /// `STRING` action + predicate. `name` is the optional prefix, `raw` the STR body.
    pub fn string_part(&self, start: usize, name: Option<String>, raw: &str) -> Option<StringPart> {
        let prefix = name.unwrap_or_default();
        let lp = prefix.to_lowercase();
        let mut p = StringPart::new(raw, prefix, None);
        if lp == "rf" || lp == "fr" {
            p.prefix = "f".into();
        } else if lp != "r" && lp != "rb" && lp != "br" {
            match typhon_common::strings::unescape(raw, lp != "b") {
                // Bytes literals are stored byte-per-char (Latin-1) so the LLVM emitter can
                // recover the exact bytes (see `LLVMVisitor::bytes_const`).
                Ok(bytes) if lp == "b" => p.value = bytes.into_iter().map(|c| c as char).collect(),
                Ok(bytes) => p.value = bytes_to_string(bytes),
                Err(e) => {
                    self.set_message(start, &format!("invalid code in a string: {}", e.0));
                    return None;
                }
            }
        } else {
            p.prefix = if lp != "r" { "b".into() } else { String::new() };
        }
        Some(p)
    }

    /// Builds the cpp-peglib error report (`ErrorInfo::output_log` + the `log`
    /// lambda of `parseCode`).
    pub fn syntax_error(&self, err_pos: usize, file: &str) -> CompilerError {
        let (pos, msg) = match self.message.borrow().clone() {
            Some((p, m)) => {
                let tok = heuristic_error_token(self.src, p);
                (p, m.replace("%t", &tok))
            }
            None => {
                let tok = heuristic_error_token(self.src, err_pos);
                // PORT-NOTE: cpp-peglib appends ", expecting ..." which `parseCode` strips
                // whenever an unexpected token exists. At EOF there is no token and the
                // expected list (rule names) cannot be reproduced from rust-peg; we emit
                // "syntax error." there.
                if tok.is_empty() {
                    (err_pos, "syntax error.".to_string())
                } else {
                    (err_pos, format!("syntax error, unexpected '{tok}'"))
                }
            }
        };
        let (l, c) = self.line_info(pos);
        CompilerError::Parser(ParserErrors::from_message(ErrorMessage::new(
            msg,
            SrcInfo::new(file, l as i32, c as i32, 0),
        )))
    }
}

/// PORT-NOTE: C++ keeps unescaped (bytes) literals as raw `std::string` bytes. Rust
/// `String` must be UTF-8: valid UTF-8 is kept as is, other byte strings are mapped
/// byte-for-char (Latin-1) so no information is lost.
pub fn bytes_to_string(b: Vec<u8>) -> String {
    match String::from_utf8(b) {
        Ok(s) => s,
        Err(e) => e.into_bytes().into_iter().map(|c| c as char).collect(),
    }
}

/// cpp-peglib `ErrorInfo::heuristic_error_token` + `escape_characters`.
pub fn heuristic_error_token(s: &str, pos: usize) -> String {
    let b = s.as_bytes();
    if pos >= b.len() {
        return String::new();
    }
    let is_stop = |c: u8| c.is_ascii_punctuation() || c.is_ascii_whitespace() || c == 0x0b;
    let mut i = 1;
    if !is_stop(b[pos]) {
        while pos + i < b.len() && !is_stop(b[pos + i]) {
            i += 1;
        }
    }
    let mut end = pos + i;
    while end < b.len() && !s.is_char_boundary(end) {
        end += 1;
    }
    let tok: String = s[pos..end].chars().take(32).collect();
    let mut r = String::new();
    for c in tok.chars() {
        match c {
            '\x0c' => r.push_str("\\f"),
            '\n' => r.push_str("\\n"),
            '\r' => r.push_str("\\r"),
            '\t' => r.push_str("\\t"),
            '\x0b' => r.push_str("\\v"),
            c => r.push(c),
        }
    }
    r
}
