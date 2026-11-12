//! Note: the reference calls `ast::parseExpr(cache, code, offset)` (rule `fstring`)
//! directly. The grammar lane lives in `crate::peg`; to keep this module independent of
//! its landing order the parser is reached through a process-wide hook installed with
//! [`set_parse_expr_hook`] (the grammar lane / compiler driver installs
//! `crate::peg::parse_expr`). Without a hook, f-string interpolation reports an error.

use std::sync::OnceLock;

use typhon_common::srcinfo::SrcInfo;
use typhon_common::{CompilerError, ErrorKind, Result};

use super::ScopingVisitor;
use crate::ast::{CallArg, CallExpr, DotExpr, FormatSpec, StringExpr, StringPart};
use crate::cache::Cache;
use crate::ids::ExprId;

/// `parseExpr(cache, code, offset) -> (Expr*, FormatSpec)`.
pub type ParseExprFn = fn(&mut Cache, &str, &SrcInfo) -> Result<(ExprId, FormatSpec)>;

static PARSE_EXPR: OnceLock<ParseExprFn> = OnceLock::new();

/// Install the f-string expression parser . First call wins.
pub fn set_parse_expr_hook(f: ParseExprFn) {
    let _ = PARSE_EXPR.set(f);
}

fn parse_expr(cache: &mut Cache, code: &str, offset: &SrcInfo) -> Result<(ExprId, FormatSpec)> {
    match PARSE_EXPR.get() {
        Some(f) => f(cache, code.trim_start(), offset),
        None => Err(CompilerError::custom("f-string expression parser is not installed", offset.clone())),
    }
}

impl ScopingVisitor<'_> {
    /// `visit(StringExpr *)`.
    pub(crate) fn scope_string(&mut self, id: ExprId) {
        let parts = self.cache.ast[id].kind.as_string().map(|s| s.strings.clone()).unwrap_or_default();
        let mut exprs: Vec<StringPart> = vec![];
        for p in parts {
            if p.prefix == "f" || p.prefix == "F" {
                // Transform an F-string
                let fstr = self.unpack_fstring(&p.value);
                if !self.can_continue() {
                    return;
                }
                for pf in fstr {
                    if pf.prefix.is_empty() && exprs.last().is_some_and(|b| b.prefix.is_empty()) {
                        exprs.last_mut().unwrap().value += &pf.value;
                    } else {
                        exprs.push(pf);
                    }
                }
            } else if !p.prefix.is_empty() {
                exprs.push(p);
            } else if exprs.last().is_some_and(|b| b.prefix.is_empty()) {
                exprs.last_mut().unwrap().value += &p.value;
            } else {
                exprs.push(p);
            }
        }
        if let Some(s) = self.cache.ast[id].kind.as_string_mut() {
            s.strings = exprs;
        }
    }

    /// Split a Python-like f-string into a list:
    ///   `f"foo {x+1} bar"` -> `("foo ", str(x+1), " bar")`
    /// Supports "{x=}" specifier (that prints the raw expression as well):
    ///   `f"{x+1=}"` -> `("x+1=", str(x+1))`
    pub fn unpack_fstring(&mut self, value: &str) -> Vec<StringPart> {
        // Strings to be concatenated
        let v = value.as_bytes();
        let sub = |a: usize, b: usize| String::from_utf8_lossy(&v[a..b]).into_owned();
        let mut items: Vec<StringPart> = vec![];
        let (mut brace_count, mut brace_start) = (0i32, 0usize);
        let mut i = 0usize;
        while i < v.len() {
            if v[i] == b'{' {
                if brace_count == 0 && i + 1 < v.len() && v[i + 1] == b'{' {
                    if brace_start < i {
                        items.push(StringPart::new(sub(brace_start, i), "", None));
                    }
                    items.push(StringPart::new("{", "", None));
                    i += 1;
                    brace_start = i + 1;
                    i += 1;
                    continue;
                }
                if brace_count == 0 {
                    if brace_start < i {
                        items.push(StringPart::new(sub(brace_start, i), "", None));
                    }
                    brace_start = i + 1;
                }
                brace_count += 1;
            } else if v[i] == b'}' {
                if brace_count == 0 && i + 1 < v.len() && v[i + 1] == b'}' {
                    if brace_start < i {
                        items.push(StringPart::new(sub(brace_start, i), "", None));
                    }
                    items.push(StringPart::new("}", "", None));
                    i += 1;
                    brace_start = i + 1;
                    i += 1;
                    continue;
                }
                brace_count -= 1;
                if brace_count == 0 {
                    let code = sub(brace_start, i);
                    let mut offset = self.src.clone();
                    offset.col += i as i32;
                    let mut item = StringPart::new(code.clone(), "#f", None);
                    item.src = offset.clone();
                    items.push(item);
                    match parse_expr(self.cache, &code, &offset) {
                        Err(e) => self.add_compiler_error(e),
                        Ok((e, mut fs)) => {
                            if !self.finish_fstring_item(items.last_mut().unwrap(), &code, e, &mut fs, &offset) {
                                return items;
                            }
                        }
                    }
                    brace_start = i + 1;
                }
            }
            i += 1;
        }
        if brace_count > 0 {
            let s = self.src.clone();
            self.add_error(ErrorKind::STR_FSTRING_BALANCE_EXTRA, &s, &[]);
        } else if brace_count < 0 {
            let s = self.src.clone();
            self.add_error(ErrorKind::STR_FSTRING_BALANCE_MISSING, &s, &[]);
        }
        if brace_start != v.len() {
            items.push(StringPart::new(sub(brace_start, v.len()), "", None));
        }
        items
    }

    /// Body of the `{...}` branch of `unpackFString` after a successful parse.
    /// Returns false on the reference early `return items`.
    fn finish_fstring_item(
        &mut self,
        item: &mut StringPart,
        code: &str,
        e: ExprId,
        fs: &mut FormatSpec,
        offset: &SrcInfo,
    ) -> bool {
        item.expr = Some(e);
        item.value = std::mem::take(&mut fs.text);
        if !self.transform_e(item.expr) {
            return false;
        }
        item.format = fs.clone();

        if !item.value.is_empty() {
            let lw = code.bytes().take_while(|&c| c == b' ' || c == b'\t').count();
            if lw > 0 {
                item.value = format!("{}{}", &code[..lw], item.value);
            }
            if item.format.conversion.is_empty() && item.format.spec.is_empty() {
                item.format.conversion = "r".into();
            }
        }

        let spec = item.format.spec.clone();
        if spec.contains('{') {
            let mut format = String::from("{");
            if !item.format.conversion.is_empty() {
                format += &format!("!{}", item.format.conversion);
            }
            format += ":";
            let mut arguments = vec![item.expr.unwrap()];
            let sb = spec.as_bytes();
            let mut start = 0usize;
            let mut j = 0usize;
            while j < sb.len() {
                if sb[j] != b'{' {
                    j += 1;
                    continue;
                }
                format += &spec[start..j];
                format += "{";
                let mut depth = 1;
                j += 1;
                let expression_start = j;
                while j < sb.len() && depth > 0 {
                    if sb[j] == b'{' {
                        depth += 1;
                    } else if sb[j] == b'}' {
                        depth -= 1;
                    }
                    if depth > 0 {
                        j += 1;
                    }
                }
                let nested = parse_expr(self.cache, &spec[expression_start..j.min(sb.len())], offset);
                let (ne, nf) = match nested {
                    Err(err) => {
                        self.add_compiler_error(err);
                        return false;
                    }
                    Ok(x) => x,
                };
                if !self.transform_e(Some(ne)) {
                    return false;
                }
                arguments.push(ne);
                if !nf.conversion.is_empty() {
                    format += &format!("!{}", nf.conversion);
                }
                if !nf.spec.is_empty() {
                    format += &format!(":{}", nf.spec);
                }
                format += "}";
                start = j + 1;
                j += 1;
            }
            format += spec.get(start..).unwrap_or("");
            format += "}";
            let src = self.src.clone();
            let ast = &mut self.cache.ast;
            let s = ast.new_expr(StringExpr::new(format, ""), src.clone());
            let d = ast.new_expr(DotExpr { expr: Some(s), member: "format".into() }, src.clone());
            let items: Vec<CallArg> = arguments.iter().map(|a| ast.call_arg("", Some(*a))).collect();
            let c = ast.new_expr(CallExpr { expr: Some(d), items, ordered: false, partial: false }, src);
            item.expr = Some(c);
            item.format.conversion.clear();
            item.format.spec.clear();
        }
        true
    }
}
