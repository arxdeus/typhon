//! Every cpp-peglib rule maps to one rust-peg rule with the same name. Semantic
//! actions build the AST through [`ParseState`] (`ParseContext`). `LOC` (the
//! source span of the whole rule match) is taken with `p:position!() ... q:position!()`
//! at the start/end of every alternative.
//!
//! Note: cpp-peglib enables packrat parsing on `program`, memoizing all rules that
//! are not `__NOPACKRAT`/side-effecting. rust-peg's `#[cache]` keys only on the input
//! position; it is enabled on the pure expression rules (whose results only depend on
//! the position and on `parens`, which is constant for a given position in practice).
//! Indentation and suite rules are never cached (they mutate the indentation stack).

use peg::RuleResult;

use super::state::{ParseState as Ctx, cond};
use crate::ast::*;

/// Result of the `fstring` rule: expression + format spec.
pub type FStringResult = (ExprId, FormatSpec);

/// `from_id` value: (what, params, return type, is_function).
type FromId = (ExprId, Vec<Param>, Option<ExprId>, bool);

/// `primary_tail` value (`pair<int, any>`).
pub enum Tail {
    Dot(String),
    Gen(ExprId),
    Call(Vec<CallArg>),
    Index(ExprId),
}

peg::parser! {
pub grammar python(st: &Ctx<'_>) for str {

// ------------------------------------------------------------------ helpers

rule reset() = #{|_, p| { st.reset(); RuleResult::Matched(p, ()) }}
rule inc() = #{|_, p| { st.inc_parens(); RuleResult::Matched(p, ()) }}
rule dec() = #{|_, p| { st.dec_parens(); RuleResult::Matched(p, ()) }}
rule eof() = ![_]

/// Macro `list(c, e) <- e (_ c _ e)*`
rule list<T>(c: rule<()>, e: rule<T>) -> Vec<T>
    = x:e() xs:(_ c() _ y:e() {y})* { let mut v = vec![x]; v.extend(xs); v }
/// Macro `tlist(c, e) <- e (_ c _ e)* (_ <c>)?`; the bool is the trailing token.
rule tlist<T>(c: rule<()>, e: rule<T>) -> (Vec<T>, bool)
    = x:e() xs:(_ c() _ y:e() {y})* t:(_ c())? { let mut v = vec![x]; v.extend(xs); (v, t.is_some()) }

// ------------------------------------------------------------------ entry points

pub rule program() -> StmtId
    = reset() p:position!() s:(s:statements() (_ EOL())* {Some(s)} / (_ EOL())* {None}) q:position!() eof() {
        match s { Some(s) => s, None => st.s(SuiteStmt { items: vec![] }, (p, q)) }
    }

pub rule fstring() -> FStringResult
    = reset() e:fstring_prefix() t:fstring_tail()? _ eof() {
        let mut fs = FormatSpec { text: e.1, conversion: String::new(), spec: String::new() };
        if let Some((c, s)) = t { fs.conversion = c; fs.spec = s; }
        (e.0, fs)
    }
rule fstring_prefix() -> (ExprId, String)
    = p:position!() e:star_expressions() _ "=" [' ' | '\t']* q:position!() t:#{|i: &str, x| RuleResult::Matched(x, i[p..q].to_string())} { (e, t) }
    / e:star_expression() { (e, String::new()) }
rule fstring_tail() -> (String, String)
    = c:fstring_conversion() s:fstring_spec()? { (c, s.unwrap_or_default()) }
    / s:fstring_spec() { (String::new(), s) }
rule fstring_conversion() -> String = "!" s:$("s" / "r" / "a") { s.to_string() }
rule fstring_spec() -> String = ":" s:format_spec() { s }
rule format_spec() -> String = s:$([_]*) { s.to_string() }

// ------------------------------------------------------------------ statements

rule statements() -> StmtId
    = p:position!() v:((_ EOL())* s:statement() {s})+ q:position!() { st.suite(v, (p, q)) }
rule statement() -> StmtId = SAMEDENT() s:compound_stmt() {s} / SAMEDENT() s:simple_stmt() {s}
rule simple_stmt() -> StmtId
    = p:position!() v:tlist(<";">, <small_stmt()>) _ EOL() q:position!() { st.suite(v.0, (p, q)) }
rule stmt_end() = &(SPACE() / ";" / EOL())
rule small_stmt() -> StmtId
    = directive()
    / assignment()
    / p:position!() "pass" stmt_end() q:position!() { st.s(SuiteStmt { items: vec![] }, (p, q)) }
    / p:position!() "break" stmt_end() q:position!() { st.s(BreakStmt, (p, q)) }
    / p:position!() "continue" stmt_end() q:position!() { st.s(ContinueStmt, (p, q)) }
    / global_stmt()
    / nonlocal_stmt()
    / s:yield_stmt() stmt_end() {s}
    / assert_stmt()
    / del_stmt()
    / s:return_stmt() stmt_end() {s}
    / s:raise_stmt() stmt_end() {s}
    / print_stmt()
    / import_stmt()
    / p:position!() e:expressions() &(_ ";" / _ EOL()) q:position!() { st.s(ExprStmt { expr: Some(e) }, (p, q)) }

rule assignment() -> StmtId
    = p:position!() l:primary() _ ":" _ t:expression() r:(_ "=" _ r:star_expressions() {r})? q:position!() {
        st.s(AssignStmt { lhs: Some(l), rhs: r, type_expr: Some(t), update: UpdateMode::Assign }, (p, q))
    }
    / p:position!() v:(t:star_targets() _ (!"==" "=") _ {t})+ r:star_expressions() !(_ "=") q:position!() {
        let mut vs = v; vs.push(r);
        let mut stmts = vec![];
        for i in (0..vs.len() - 1).rev() {
            stmts.push(st.s(AssignStmt { lhs: Some(vs[i]), rhs: Some(vs[i + 1]), type_expr: None, update: UpdateMode::Assign }, (p, q)));
        }
        st.s(SuiteStmt { items: stmts }, (p, q))
    }
    / p:position!() l:star_expression() _ op:augassign() "=" _ r:star_expressions() q:position!() {
        let c = st.clone_expr(l);
        let b = st.binary(c, &op, r, true, (p, q));
        st.s(AssignStmt { lhs: Some(l), rhs: Some(b), type_expr: None, update: UpdateMode::Assign }, (p, q))
    }
rule augassign() -> String
    = s:$("+" / "-" / "**" / "*" / "@" / "//" / "/" / "%" / "&" / "|" / "^" / "<<" / ">>") { s.to_string() }
rule global_stmt() -> StmtId
    = p:position!() "global" SPACE() v:tlist(<",">, <NAME()>) q:position!() {
        let items = v.0.into_iter().map(|n| st.s(GlobalStmt { var: n, non_local: false }, (p, q))).collect();
        st.s(SuiteStmt { items }, (p, q))
    }
rule nonlocal_stmt() -> StmtId
    = p:position!() "nonlocal" SPACE() v:tlist(<",">, <NAME()>) q:position!() {
        let items = v.0.into_iter().map(|n| st.s(GlobalStmt { var: n, non_local: true }, (p, q))).collect();
        st.s(SuiteStmt { items }, (p, q))
    }
rule yield_stmt() -> StmtId
    = p:position!() "yield" SPACE() "from" SPACE() e:expression() q:position!() { st.s(YieldFromStmt { expr: Some(e) }, (p, q)) }
    / p:position!() "yield" e:(SPACE() e:expressions() {e})? q:position!() { st.s(YieldStmt { expr: e }, (p, q)) }
rule assert_stmt() -> StmtId
    = p:position!() "assert" SPACE() e:expression() m:(_ "," _ m:expression() {m})? q:position!() {
        st.s(AssertStmt { expr: Some(e), message: m }, (p, q))
    }
rule del_stmt() -> StmtId
    = p:position!() "del" SPACE() v:tlist(<",">, <expression()>) q:position!() {
        let items = v.0.into_iter().map(|e| st.s(DelStmt { expr: Some(e) }, (p, q))).collect();
        st.s(SuiteStmt { items }, (p, q))
    }
rule return_stmt() -> StmtId
    = p:position!() "return" e:(SPACE() e:expressions() {e})? q:position!() { st.s(ReturnStmt { expr: e }, (p, q)) }
rule raise_stmt() -> StmtId
    = p:position!() "raise" SPACE() e:expression() f:(SPACE() "from" SPACE() f:expression() {f})? q:position!() {
        st.s(ThrowStmt { expr: Some(e), from: f, transformed: false }, (p, q))
    }
    / p:position!() "raise" q:position!() { st.s(ThrowStmt { expr: None, from: None, transformed: false }, (p, q)) }
rule print_stmt() -> StmtId
    = p:position!() "print" SPACE() e:star_expression() es:(_ "," _ e:star_expression() {e})* t:(_ ",")? q:position!() {
        let mut items = vec![e]; items.extend(es);
        st.s(PrintStmt { items, no_newline: t.is_some() }, (p, q))
    }
    / p:position!() "print" _ &EOL() q:position!() { st.s(PrintStmt { items: vec![], no_newline: false }, (p, q)) }
rule import_stmt() -> StmtId = import_name() / import_from()
rule import_name() -> StmtId
    = p:position!() "import" SPACE() v:list(<",">, <as_name()>) q:position!() {
        let items = v.into_iter().map(|(e, a)| st.s(ImportStmt {
            from: Some(e), what: None, as_: a, dots: 0, args: vec![], ret: None, is_function: true,
        }, (p, q))).collect();
        st.s(SuiteStmt { items }, (p, q))
    }
rule as_name() -> (ExprId, String)
    = e:dot_name() a:(SPACE() "as" SPACE() n:NAME() {n})? { (e, a.unwrap_or_default()) }
rule import_from() -> StmtId
    = p:position!() "from" SPACE() d:(_ ".")* f:(_ f:dot_name() {f})? SPACE() "import" SPACE() "*" q:position!() {
        let what = st.id("*", (p, q));
        st.s(ImportStmt { from: f, what: Some(what), as_: String::new(), dots: d.len(), args: vec![], ret: None, is_function: true }, (p, q))
    }
    / p:position!() "from" SPACE() d:(_ ".")* f:(_ f:dot_name() {f})? SPACE() "import" SPACE()
      v:(from_as_parens() / from_as_items()) q:position!() {
        let items = v.into_iter().map(|((w, args, ret, isf), a)| st.s(ImportStmt {
            from: f, what: Some(w), as_: a, dots: d.len(), args, ret, is_function: isf,
        }, (p, q))).collect();
        st.s(SuiteStmt { items }, (p, q))
    }
rule from_as_parens() -> Vec<(FromId, String)>
    = inc() r:("(" _ v:tlist(<",">, <from_as()>) _ ")" {v.0})? dec() x:#{cond(r)} {x}
rule from_as_items() -> Vec<(FromId, String)> = list(<",">, <from_as()>)
rule from_as() -> (FromId, String)
    = f:from_id() a:(SPACE() "as" SPACE() n:NAME() {n})? { (f, a.unwrap_or_default()) }
rule from_id() -> FromId
    = d:dot_name() _ ":" _ e:expression() { (d, vec![], Some(e), false) }
    / p:position!() d:dot_name() _ ps:from_params() r:(_ "->" _ e:expression() {e})? q:position!() {
        let r = r.unwrap_or_else(|| st.id("NoneType", (p, q)));
        (d, ps, Some(r), true)
    }
    / d:dot_name() { (d, vec![], None, true) }
rule dot_name() -> ExprId
    = p:position!() i:id() ns:(_ "." _ n:NAME() {n})* q:position!() {
        let mut e = i;
        for n in ns { e = st.e(DotExpr { expr: Some(e), member: n }, (p, q)); }
        e
    }
rule from_params() -> Vec<Param>
    = inc() r:("(" _ v:tlist(<",">, <from_param()>)? _ ")" {v.map(|v| v.0).unwrap_or_default()})? dec() x:#{cond(r)} {x}
rule from_param() -> Param
    = p:position!() e:expression() q:position!() { st.param((p, q), "", Some(e), None) }

rule suite() -> StmtId
    = simple_stmt()
    / (_ EOL())+ &INDENT() s:statements() (_ EOL())* &DEDENT() {s}
rule compound_stmt() -> StmtId
    = function() / if_stmt() / class() / with_stmt_async() / for_() / try_stmt()
    / while_stmt() / match_stmt()
rule if_stmt() -> StmtId
    = p:position!() "if" SPACE() c:named_expression() _ ":" _ s:suite()
      el:(SAMEDENT() "elif" SPACE() c:named_expression() _ ":" _ s:suite() {(c, s)})*
      e:(SAMEDENT() "else" _ ":" _ s:suite() {s})? q:position!() {
        let mut v = vec![(c, s)]; v.extend(el);
        let mut last = st.wrap(e);
        for (c, s) in v.into_iter().rev() {
            let s = st.wrap(Some(s));
            last = Some(st.s(IfStmt { cond: Some(c), if_suite: s, else_suite: last }, (p, q)));
        }
        last.unwrap()
    }
rule while_stmt() -> StmtId
    = p:position!() "while" SPACE() c:named_expression() _ ":" _ s:suite()
      e:(SAMEDENT() "else" (SPACE() "not" SPACE() "break")* _ ":" _ s:suite() {s})? q:position!() {
        st.s(WhileStmt { cond: Some(c), suite: Some(s), else_suite: e, goto_var: String::new() }, (p, q))
    }
rule for_() -> StmtId
    = d:decorator()? s:for_stmt_async() {
        if let Some(d) = d { st.arena()[s].kind.as_for_mut().unwrap().decorator = Some(d); }
        s
    }
rule for_stmt_async() -> StmtId
    = "async" SPACE() s:for_stmt() { st.arena()[s].kind.as_for_mut().unwrap().async_ = true; s }
    / for_stmt()
rule for_stmt() -> StmtId
    = p:position!() "for" SPACE() v:star_targets()
      SPACE() "in" SPACE() i:star_expressions() _ ":" _ s:suite()
      e:(SAMEDENT() "else" (SPACE() "not" SPACE() "break")* _ ":" _ s:suite() {s})? q:position!() {
        st.s(ForStmt { var: Some(v), iter: Some(i), suite: Some(s), else_suite: e, ..Default::default() }, (p, q))
    }
rule with_stmt_async() -> StmtId
    = "async" SPACE() s:with_stmt() { st.arena()[s].kind.as_with_mut().unwrap().async_ = true; s }
    / with_stmt()
rule with_stmt() -> StmtId
    = p:position!() "with" SPACE() v:(with_parens_item() / with_item()) _ ":" _ s:suite() q:position!() {
        let src = st.loc((p, q));
        st.arena().new_with_pairs(src, v, Some(s), false)
    }
rule with_parens_item() -> Vec<(ExprId, Option<ExprId>)>
    = inc() r:("(" _ v:tlist(<",">, <as_item()>) _ ")" {v.0})? dec() x:#{cond(r)} {x}
rule with_item() -> Vec<(ExprId, Option<ExprId>)> = list(<",">, <as_item()>)
rule as_item() -> (ExprId, Option<ExprId>)
    = e:expression() SPACE() "as" SPACE() i:id() &(_ ("," / ")" / ":")) { (e, Some(i)) }
    / e:expression() { (e, None) }
rule try_stmt() -> StmtId
    = p:position!() "try" _ ":" _ s:suite() ex:excepts() ef:else_finally()? q:position!() {
        let (e, f) = ef.unwrap_or((None, None));
        st.s(TryStmt { items: ex, suite: Some(s), else_suite: e, finally: f }, (p, q))
    }
    / p:position!() "try" _ ":" _ s:suite() f:(SAMEDENT() "finally" _ ":" _ s:suite() {s})? q:position!() {
        st.s(TryStmt { items: vec![], suite: Some(s), else_suite: None, finally: f }, (p, q))
    }
rule else_finally() -> (Option<StmtId>, Option<StmtId>)
    = SAMEDENT() "else" _ ":" _ e:suite() SAMEDENT() "finally" _ ":" _ f:suite() { (Some(e), Some(f)) }
    / SAMEDENT() "else" _ ":" _ e:suite() { (Some(e), None) }
    / SAMEDENT() "finally" _ ":" _ f:suite() { (None, Some(f)) }
rule excepts() -> Vec<StmtId> = (SAMEDENT() e:except_block() {e})+
rule except_block() -> StmtId
    = p:position!() "except" SPACE() e:expression() n:(SPACE() "as" SPACE() n:NAME() {n})? _ ":" _ s:suite() q:position!() {
        st.s(ExceptStmt { var: n.unwrap_or_default(), exc: Some(e), suite: Some(s) }, (p, q))
    }
    / p:position!() "except" _ ":" _ s:suite() q:position!() {
        st.s(ExceptStmt { var: String::new(), exc: None, suite: Some(s) }, (p, q))
    }
rule function() -> StmtId
    = p:position!() d:extern_decorators() f:function_def_async() (_ EOL())+ &INDENT() x:extern_() (_ EOL())* &DEDENT() q:position!() {
        let x = x.trim_end_matches('\n').to_string();
        let se = st.e(StringExpr::new(x, ""), (p, q));
        let es = st.s(ExprStmt { expr: Some(se) }, (p, q));
        let w = st.wrap(Some(es));
        let mut a = st.arena();
        let fs = a[f].kind.as_function_mut().unwrap();
        fs.decorators = d; fs.suite = w;
        f
    }
    / d:decorators()? f:function_def_async() _ s:suite() {
        let w = st.wrap(Some(s));
        let mut a = st.arena();
        let fs = a[f].kind.as_function_mut().unwrap();
        if let Some(d) = d { fs.decorators = d; }
        fs.suite = w;
        f
    }
rule extern_() -> String
    = s:$((empty_line()* EXTERNDENT() [^ '\r' | '\n']* EOL() empty_line()*)+) { s.to_string() }
rule empty_line() = [' ' | '\t']* EOL()
rule function_def_async() -> StmtId
    = "async" SPACE() f:function_def() { st.arena()[f].kind.as_function_mut().unwrap().async_ = true; f }
    / function_def()
rule function_def() -> StmtId
    = p:position!() "def" SPACE() n:NAME() _ g:generics() _ ps:params() r:(_ "->" _ e:expression() {e})? _ ":" q:position!() {
        let f = st.function(n, r, ps, (p, q));
        let mut a = st.arena();
        let fs = a[f].kind.as_function_mut().unwrap();
        for x in g { fs.add_param(x); }
        f
    }
    / p:position!() "def" SPACE() n:NAME() _ ps:params() r:(_ "->" _ e:expression() {e})? _ ":" q:position!() {
        st.function(n, r, ps, (p, q))
    }
rule params() -> Vec<Param>
    = inc() r:("(" _ v:tlist(<",">, <function_param()>)? _ ")" {v.map(|v| v.0).unwrap_or_default()})? dec() x:#{cond(r)} {x}
rule function_param() -> Param = param() / param_separator()
rule param_separator() -> Param
    = p:position!() s:$("*" / "/") q:position!() { st.param((p, q), s, None, None) }
rule param() -> Param
    = p:position!() n:param_name() _ ":" _ t:expression() d:(_ "=" _ d:expression() {d})? q:position!() {
        st.param((p, q), &n, Some(t), d)
    }
    / p:position!() n:param_name() d:(_ "=" _ d:expression() {d})? q:position!() {
        st.param((p, q), &n, None, d)
    }
rule param_name() -> String
    = s:$("**" / "*")? _ n:NAME() { format!("{}{}", s.unwrap_or(""), n) }
rule generics() -> Vec<Param>
    = inc() r:(p:position!() "[" _ v:tlist(<",">, <param()>) _ "]" q:position!() {
        v.0.into_iter().map(|mut x| {
            x.status = ParamStatus::Generic;
            if x.ty.is_none() { x.ty = Some(st.id("type", (p, q))); }
            x
        }).collect::<Vec<_>>()
    })? dec() x:#{cond(r)} {x}
rule decorators() -> Vec<ExprId> = decorator()+
rule decorator() -> ExprId
    = "@" _ !(("llvm" / "python") _ EOL()) e:named_expression() _ EOL() SAMEDENT() {e}
rule extern_decorators() -> Vec<ExprId>
    = p:position!() a:decorators()? "@" _ k:$("llvm" / "python") _ EOL() SAMEDENT() b:decorators()? q:position!() {
        let mut v = vec![st.id(k, (p, q))];
        v.extend(a.unwrap_or_default());
        v.extend(b.unwrap_or_default());
        v
    }
rule class() -> StmtId
    = d:decorators()? c:class_def() {
        if let Some(d) = d { st.arena()[c].kind.as_class_mut().unwrap().decorators = d; }
        c
    }
rule base_class_args() -> Vec<ExprId>
    = "(" _ v:tlist(<",">, <expression()>)? _ ")" { v.map(|v| v.0).unwrap_or_default() }
rule class_args() -> (Vec<Param>, Vec<ExprId>)
    = g:generics() _ b:base_class_args() { (g, b) }
    / g:generics() { (g, vec![]) }
    / b:base_class_args() { (vec![], b) }
rule class_def() -> StmtId
    = p:position!() "class" SPACE() n:NAME() _ ca:class_args()? _ ":" _ s:suite() q:position!() {
        let (generics, bases) = ca.unwrap_or_default();
        let mut args = vec![];
        let suite = st.s(SuiteStmt { items: vec![] }, (p, q));
        {
            let mut a = st.arena();
            let items = a[s].kind.as_suite().map(|x| x.items.clone()).unwrap_or_else(|| vec![s]);
            for i in items {
                if let Some(asg) = a[i].kind.as_assign()
                    && let Some(l) = asg.lhs
                    && let Some(id) = a[l].kind.as_id()
                {
                    let src = a[i].src;
                    let (ty, rhs) = (asg.type_expr, asg.rhs);
                    let name = id.value.clone();
                    args.push(a.new_param(src, name, ty, rhs, 0, ParamKind::Normal));
                    continue;
                }
                a.suite_add_stmt(suite, Some(i));
            }
            a.suite_flatten(suite);
        }
        args.extend(generics);
        let src = st.loc((p, q));
        st.arena().new_class(src, n, args, Some(suite), vec![], bases)
    }
rule match_stmt() -> StmtId
    = p:position!() "match" SPACE() e:expression() _ ":" (_ EOL())+
      &INDENT() cs:(SAMEDENT() c:case() {c})+ (_ EOL())* &DEDENT() q:position!() {
        st.s(MatchStmt { expr: Some(e), items: cs }, (p, q))
    }
rule case() -> MatchCase
    = "case" SPACE() m:match_pattern() SPACE() "if" SPACE() g:pipe() _ ":" _ s:suite() {
        MatchCase { pattern: Some(m), guard: Some(g), suite: st.wrap(Some(s)) }
    }
    / "case" SPACE() m:match_pattern() _ ":" _ s:suite() {
        MatchCase { pattern: Some(m), guard: None, suite: st.wrap(Some(s)) }
    }
rule match_pattern() -> ExprId
    = p:position!() e:expression() n:(SPACE() "as" SPACE() n:NAME() {n})? q:position!() {
        match n {
            None => e,
            Some(n) => { let i = st.id(&n, (p, q)); st.e(AssignExpr { var: Some(i), expr: Some(e) }, (p, q)) }
        }
    }
rule directive() -> StmtId
    = p:position!() "##" _ "typhon:" _ n:NAME() _ "=" _ v:(INT() / NAME()) q:position!() {
        st.s(DirectiveStmt { key: n, value: v }, (p, q))
    }

// ------------------------------------------------------------------ expressions

rule expressions() -> ExprId
    = p:position!() v:tlist(<",">, <expression()>) q:position!() { st.wrap_tuple(v, (p, q)) }
#[cache]
rule expression() -> ExprId
    = lambdef()
    / p:position!() a:disjunction() SPACE()? "if" SPACE()? c:disjunction() SPACE()? "else" SPACE()? b:expression() q:position!() {
        st.e(IfExpr { cond: Some(c), ifexpr: Some(a), elsexpr: Some(b) }, (p, q))
    }
    / pipe()
rule lambdef() -> ExprId
    = p:position!() "lambda" SPACE() ps:lparams() _ ":" _ e:expression() q:position!() { st.lambda(ps, e, (p, q)) }
    / p:position!() "lambda" _ ":" _ e:expression() q:position!() { st.lambda(vec![], e, (p, q)) }
rule lparams() -> Vec<Param> = v:tlist(<",">, <lambda_param()>)? { v.map(|v| v.0).unwrap_or_default() }
rule lambda_param() -> Param = lparam() / param_separator()
rule lparam() -> Param
    = p:position!() n:param_name() d:(_ "=" _ d:expression() {d})? q:position!() { st.param((p, q), &n, None, d) }
rule pipe() -> ExprId
    = p:position!() x:disjunction() xs:(_ o:$("|>" / "||>") _ e:disjunction() {(o, e)})* q:position!() {
        if xs.is_empty() { return x; }
        let mut v = vec![Pipe { op: String::new(), expr: Some(x) }];
        for (o, e) in xs { v.push(Pipe { op: o.to_string(), expr: Some(e) }); }
        st.e(PipeExpr { items: v, in_types: vec![] }, (p, q))
    }
#[cache]
rule disjunction() -> ExprId
    = p:position!() x:conjunction() xs:(SPACE()? "or" SPACE()? e:conjunction() {e})* q:position!() {
        st.chain(x, xs.into_iter().map(|e| ("||", e)).collect(), (p, q))
    }
rule conjunction() -> ExprId
    = p:position!() x:inversion() xs:(SPACE()? "and" SPACE()? e:inversion() {e})* q:position!() {
        st.chain(x, xs.into_iter().map(|e| ("&&", e)).collect(), (p, q))
    }
rule inversion() -> ExprId
    = p:position!() "not" SPACE() e:inversion() q:position!() { st.e(UnaryExpr { op: "!".into(), expr: Some(e) }, (p, q)) }
    / comparison()
rule comparison() -> ExprId
    = p:position!() x:bitwise_or() xs:compare_op_bitwise_or()* q:position!() {
        match xs.len() {
            0 => x,
            1 => { let (o, e) = xs.into_iter().next().unwrap(); st.binary(x, &o, e, false, (p, q)) }
            _ => {
                let mut v = vec![(String::new(), x)]; v.extend(xs);
                st.e(ChainBinaryExpr { exprs: v }, (p, q))
            }
        }
    }
rule compare_op_bitwise_or() -> (String, ExprId)
    = SPACE() "not" SPACE() "in" SPACE() e:bitwise_or() { ("not in".into(), e) }
    / SPACE() "is" SPACE() "not" SPACE() e:bitwise_or() { ("is not".into(), e) }
    / SPACE() o:$("in" / "is") SPACE() e:bitwise_or() { (o.into(), e) }
    / _ o:$("==" / "!=" / "<=" / "<" / ">=" / ">") _ e:bitwise_or() { (o.into(), e) }
#[cache]
rule bitwise_or() -> ExprId
    = p:position!() x:bitwise_xor() xs:(_ o:$("|") _ e:bitwise_xor() {(o, e)})* q:position!() { st.chain(x, xs, (p, q)) }
rule bitwise_xor() -> ExprId
    = p:position!() x:bitwise_and() xs:(_ o:$("^") _ e:bitwise_and() {(o, e)})* q:position!() { st.chain(x, xs, (p, q)) }
rule bitwise_and() -> ExprId
    = p:position!() x:shift_expr() xs:(_ o:$("&") _ e:shift_expr() {(o, e)})* q:position!() { st.chain(x, xs, (p, q)) }
rule shift_expr() -> ExprId
    = p:position!() x:sum() xs:(_ o:$("<<" / ">>") _ e:sum() {(o, e)})* q:position!() { st.chain(x, xs, (p, q)) }
rule sum() -> ExprId
    = p:position!() x:term() xs:(_ o:$("+" / "-") _ e:term() {(o, e)})* q:position!() { st.chain(x, xs, (p, q)) }
rule term() -> ExprId
    = p:position!() x:factor() xs:(_ o:$("*" / "//" / "/" / "%" / "@") _ e:factor() {(o, e)})* q:position!() { st.chain(x, xs, (p, q)) }
rule factor() -> ExprId
    = p:position!() o:$("+" / "-" / "~") _ e:factor() q:position!() { st.e(UnaryExpr { op: o.into(), expr: Some(e) }, (p, q)) }
    / power()
rule power() -> ExprId
    = p:position!() x:await_primary() y:(_ "**" _ f:factor() {f})? q:position!() {
        match y { None => x, Some(y) => st.binary(x, "**", y, false, (p, q)) }
    }
rule await_primary() -> ExprId
    = p:position!() "await" SPACE() e:primary() q:position!() { st.e(AwaitExpr { expr: Some(e), transformed: false }, (p, q)) }
    / primary()
#[cache]
rule primary() -> ExprId
    = p:position!() a:atom() ts:(_ t:primary_tail() {t})* q:position!() {
        let mut e = a;
        for t in ts {
            e = match t {
                Tail::Dot(n) => st.e(DotExpr { expr: Some(e), member: n }, (p, q)),
                Tail::Gen(g) => { let arg = st.arg(g); st.e(CallExpr { expr: Some(e), items: vec![arg], ordered: false, partial: false }, (p, q)) }
                Tail::Call(v) => st.e(CallExpr { expr: Some(e), items: v, ordered: false, partial: false }, (p, q)),
                Tail::Index(i) => st.e(IndexExpr { expr: Some(e), index: Some(i) }, (p, q)),
            };
        }
        e
    }
rule primary_tail() -> Tail
    = "." _ n:NAME() { Tail::Dot(n) }
    / g:genexp() { Tail::Gen(g) }
    / a:arguments() { Tail::Call(a) }
    / s:slices() { Tail::Index(s) }
rule slices() -> ExprId
    = inc() r:(p:position!() "[" _ v:tlist(<",">, <slice()>) _ "]" q:position!() { st.wrap_tuple(v, (p, q)) })? dec() x:#{cond(r)} {x}
rule slice() -> ExprId
    = p:position!() a:slice_part() _ ":" _ b:slice_part() c:(_ ":" _ c:slice_part() {c})? q:position!() {
        st.e(SliceExpr { start: a, stop: b, step: c.flatten() }, (p, q))
    }
    / expression()
rule slice_part() -> Option<ExprId> = expression()?
rule atom() -> ExprId
    = p:position!() s:STRING() ss:(SPACE() s:STRING() {s})* q:position!() {
        let mut v = vec![s]; v.extend(ss);
        st.e(StringExpr::from_parts(v), (p, q))
    }
    / id()
    / p:position!() "True" q:position!() { st.e(BoolExpr { value: true }, (p, q)) }
    / p:position!() "False" q:position!() { st.e(BoolExpr { value: false }, (p, q)) }
    / p:position!() "None" q:position!() { st.e(NoneExpr, (p, q)) }
    / p:position!() a:INT() _ "..." _ b:INT() q:position!() {
        let a = st.e(IntExpr::parse(&a, ""), (p, q));
        let b = st.e(IntExpr::parse(&b, ""), (p, q));
        st.e(RangeExpr { start: Some(a), stop: Some(b) }, (p, q))
    }
    / p:position!() f:FLOAT() n:NAME()? q:position!() { st.e(FloatExpr::parse(&f, n.unwrap_or_default()), (p, q)) }
    / p:position!() i:INT() n:NAME()? q:position!() { st.e(IntExpr::parse(&i, n.unwrap_or_default()), (p, q)) }
    / parentheses()
    / p:position!() "..." q:position!() { st.e(EllipsisExpr { mode: EllipsisType::Standalone }, (p, q)) }
rule parentheses() -> ExprId
    = inc() r:(e:(tuple() / yield_() / named() / genexp() / listexpr() / listcomp() / dict() / set() / dictcomp() / setcomp()) {e})? dec() x:#{cond(r)} {x}
rule tuple() -> ExprId
    = p:position!() "(" _ ")" q:position!() { st.e(TupleExpr { items: vec![] }, (p, q)) }
    / p:position!() "(" _ v:tlist(<",">, <star_named_expression()>) _ ")" q:position!() { st.wrap_tuple(v, (p, q)) }
rule yield_() -> ExprId = p:position!() "(" _ "yield" _ ")" q:position!() { st.e(YieldExpr, (p, q)) }
rule named() -> ExprId = "(" _ e:named_expression() _ ")" {e}
rule genexp() -> ExprId
    = inc() r:(p:position!() "(" _ e:named_expression() SPACE() l:for_if_clauses() _ ")" q:position!() {
        st.generator(GeneratorKind::Generator, e, l, (p, q))
    })? dec() x:#{cond(r)} {x}
rule listexpr() -> ExprId
    = p:position!() "[" _ v:tlist(<",">, <star_named_expression()>)? _ "]" q:position!() {
        st.e(ListExpr { items: v.map(|v| v.0).unwrap_or_default() }, (p, q))
    }
rule listcomp() -> ExprId
    = p:position!() "[" _ e:named_expression() SPACE() l:for_if_clauses() _ "]" q:position!() {
        st.generator(GeneratorKind::ListGenerator, e, l, (p, q))
    }
rule set() -> ExprId
    = p:position!() "{" _ v:tlist(<",">, <star_named_expression()>) _ "}" q:position!() { st.e(SetExpr { items: v.0 }, (p, q)) }
rule setcomp() -> ExprId
    = p:position!() "{" _ e:named_expression() SPACE() l:for_if_clauses() _ "}" q:position!() {
        st.generator(GeneratorKind::SetGenerator, e, l, (p, q))
    }
rule dict() -> ExprId
    = p:position!() "{" _ v:tlist(<",">, <double_starred_kvpair()>)? _ "}" q:position!() {
        st.e(DictExpr { items: v.map(|v| v.0).unwrap_or_default() }, (p, q))
    }
rule dictcomp() -> ExprId
    = p:position!() "{" _ kv:kvpair() SPACE() l:for_if_clauses() _ "}" q:position!() { st.dict_generator(kv, l, (p, q)) }
rule double_starred_kvpair() -> ExprId
    = p:position!() "**" _ e:bitwise_or() q:position!() { st.e(KeywordStarExpr { expr: Some(e) }, (p, q)) }
    / kvpair()
rule kvpair() -> ExprId
    = p:position!() k:expression() _ ":" _ v:expression() q:position!() { st.e(TupleExpr { items: vec![k, v] }, (p, q)) }
rule for_if_clauses() -> Vec<StmtId>
    = x:for_if_clause_async() xs:(SPACE() c:for_if_clause_async() {c})* {
        let mut v = x; for t in xs { v.extend(t); } v
    }
rule for_if_clause_async() -> Vec<StmtId>
    = "async" SPACE() s:for_if_clause() {
        if let Some(&f) = s.first() && let Some(fs) = st.arena()[f].kind.as_for_mut() { fs.async_ = true; }
        s
    }
    / for_if_clause()
rule for_if_clause() -> Vec<StmtId>
    = p:position!() "for" SPACE() v:star_targets() SPACE() "in" SPACE() i:disjunction()
      cs:(SPACE()? "if" SPACE()? c:disjunction() {c})* q:position!() {
        let mut r = vec![st.s(ForStmt { var: Some(v), iter: Some(i), ..Default::default() }, (p, q))];
        for c in cs { r.push(st.s(IfStmt { cond: Some(c), if_suite: None, else_suite: None }, (p, q))); }
        r
    }
rule star_targets() -> ExprId
    = p:position!() v:tlist(<",">, <star_target()>) q:position!() { st.wrap_tuple(v, (p, q)) }
rule star_target() -> ExprId
    = p:position!() "*" _ !"*" e:star_target() q:position!() { st.e(StarExpr { expr: Some(e) }, (p, q)) }
    / star_parens()
    / primary()
rule star_parens() -> ExprId
    = inc() r:(
        p:position!() "(" _ v:tlist(<",">, <star_target()>) _ ")" q:position!() { st.wrap_tuple(v, (p, q)) }
        / p:position!() "[" _ v:tlist(<",">, <star_target()>) _ "]" q:position!() { st.e(ListExpr { items: v.0 }, (p, q)) }
    )? dec() x:#{cond(r)} {x}
rule star_expressions() -> ExprId
    = p:position!() v:tlist(<",">, <star_expression()>) q:position!() { st.wrap_tuple(v, (p, q)) }
rule star_expression() -> ExprId
    = p:position!() "*" _ e:bitwise_or() q:position!() { st.e(StarExpr { expr: Some(e) }, (p, q)) }
    / expression()
rule star_named_expression() -> ExprId
    = p:position!() "*" _ e:bitwise_or() q:position!() { st.e(StarExpr { expr: Some(e) }, (p, q)) }
    / named_expression()
rule named_expression() -> ExprId
    = p:position!() n:NAME() _ ":=" _ c:(e:expression() {Some(e)} / {None}) q:position!() x:#{cond(c)} {
        let i = st.id(&n, (p, q));
        st.e(AssignExpr { var: Some(i), expr: Some(x) }, (p, q))
    }
    / !(NAME() _ ":=") e:expression() !(_ ":=") {e}
rule arguments() -> Vec<CallArg>
    = inc() r:("(" _ v:tlist(<",">, <args()>)? _ ")" {
        let mut result: Vec<CallArg> = v.map(|v| v.0).unwrap_or_default().into_iter().flatten().collect();
        if let Some(last) = result.last_mut()
            && last.name.is_empty()
            && let Some(e) = last.value
        {
            let src = st.arena()[e].src;
            if st.arena()[e].kind.as_ellipsis().is_some() {
                let en = st.arena().new_expr(EllipsisExpr { mode: EllipsisType::Partial }, src);
                *last = CallArg::new("", Some(en));
                last.src = src;
            }
        }
        result
    })? dec() x:#{cond(r)} {x}
rule args() -> Vec<CallArg>
    = a:simple_args() k:(_ "," _ k:kwargs() {k})? { let mut a = a; a.extend(k.unwrap_or_default()); a }
    / kwargs()
rule simple_args() -> Vec<CallArg>
    = v:list(<",">, <(starred_expression() / e:named_expression() !(_ "=") {e})>) {
        v.into_iter().map(|e| st.arg(e)).collect()
    }
rule starred_expression() -> ExprId
    = p:position!() "*" _ e:expression() q:position!() { st.e(StarExpr { expr: Some(e) }, (p, q)) }
rule kwargs() -> Vec<CallArg>
    = a:list(<",">, <kwarg_or_starred()>) _ "," _ b:list(<",">, <kwarg_or_double_starred()>) { let mut a = a; a.extend(b); a }
    / list(<",">, <kwarg_or_starred()>)
    / list(<",">, <kwarg_or_double_starred()>)
rule kwarg_or_starred() -> CallArg
    = p:position!() n:NAME() _ "=" _ e:expression() q:position!() { st.arg_at((p, q), &n, e) }
    / e:starred_expression() { st.arg(e) }
rule kwarg_or_double_starred() -> CallArg
    = p:position!() n:NAME() _ "=" _ e:expression() q:position!() { st.arg_at((p, q), &n, e) }
    / p:position!() "**" _ e:expression() q:position!() { let k = st.e(KeywordStarExpr { expr: Some(e) }, (p, q)); st.arg(k) }
rule id() -> ExprId = p:position!() n:NAME() q:position!() { st.id(&n, (p, q)) }

// ------------------------------------------------------------------ lexical

rule INT() -> String = s:$(BININT() / HEXINT() / OCTINT() / DECINT()) { s.to_string() }
rule BININT() = "0" ['b' | 'B'] ['0'..='1'] ("_"* ['0'..='1'])*
rule OCTINT() = "0" ['o' | 'O'] ['0'..='7'] ("_"* ['0'..='7'])*
rule HEXINT() = "0" ['x' | 'X'] ['0'..='9' | 'a'..='f' | 'A'..='F'] ("_"? ['0'..='9' | 'a'..='f' | 'A'..='F'])*
rule DECINT() = ['0'..='9'] ("_"? ['0'..='9'])*
rule FLOAT() -> String = s:$(EXPFLOAT() / PTFLOAT()) { s.to_string() }
rule PTFLOAT() = DECINT()? "." DECINT() / DECINT() "."
rule EXPFLOAT() = (PTFLOAT() / DECINT()) ['e' | 'E'] ("+" / "-")? DECINT()
rule NAME() -> String
    = s:$(keyword() ['a'..='z' | 'A'..='Z' | '_' | '0'..='9']+) { s.to_string() }
    / !keyword() s:$(['a'..='z' | 'A'..='Z' | '_'] ['a'..='z' | 'A'..='Z' | '_' | '0'..='9']*) { s.to_string() }
rule STRING() -> StringPart
    = p:position!() n:NAME()? s:STR() x:#{cond(st.string_part(p, n.clone(), &s))} {x}
rule STR() -> String
    = "\"\"\"" s:$((!"\"\"\"" CHAR())*) "\"\"\"" { s.to_string() }
    / "'''" s:$((!"'''" CHAR())*) "'''" { s.to_string() }
    / "\"" s:$((!("\"" / EOL()) CHAR())*) "\"" { s.to_string() }
    / "'" s:$((!("'" / EOL()) CHAR())*) "'" { s.to_string() }
rule CHAR() = "\\" [_] / [_]
rule COMMENT() = !directive_lookahead() "#" [^ '\r' | '\n']*
/// Note: `!directive` would build AST nodes from a predicate; an equivalent pure
/// lookahead of the `directive` syntax is used instead.
rule directive_lookahead() = "##" _ "typhon:" _ NAME() _ "=" _ (INT() / NAME())
rule ws() -> usize = s:$([' ' | '\t']*) { s.len() }
rule INDENT() = p:position!() n:ws() x:#{cond(st.indent_pred(p, n))}
rule SAMEDENT() = n:ws() x:#{cond(st.samedent_pred(n))}
rule DEDENT() = p:position!() n:ws() x:#{cond(st.dedent_pred(p, n))}
rule EXTERNDENT() = n:ws() x:#{cond(st.externdent_pred(n))}
rule EOL() = quiet!{"\r\n" / ['\r' | '\n']}
rule NLP() = #{st.nlp()}
rule SPACE() = quiet!{([' ' | '\t']+ / COMMENT() / NLP() EOL())+}
rule _ = SPACE()?
rule keyword()
    = "False" / "else" / "import" / "pass" / "None" / "break" / "except" / "in" / "raise"
    / "True" / "class" / "finally" / "is" / "return" / "and" / "continue" / "for" / "as"
    / "lambda" / "try" / "def" / "from" / "while" / "assert" / "del" / "global" / "not"
    / "with" / "elif" / "if" / "or" / "yield" / "async" / "await"
}
}
