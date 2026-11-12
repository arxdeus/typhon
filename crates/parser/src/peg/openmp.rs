
use super::state::ParseState as Ctx;
use crate::ast::*;

peg::parser! {
pub grammar omp(st: &Ctx<'_>) for str {

pub rule pragma() -> Vec<CallArg>
    = "omp"? _ "parallel"? _ v:(c:clause() _ {c})* { v.into_iter().flatten().collect() }
rule clause() -> Vec<CallArg>
    = p:position!() "schedule" _ "(" _ k:schedule_kind() c:(_ "," _ i:int() {i})? _ ")" q:position!() {
        let s = st.e(StringExpr::new(k, ""), (p, q));
        let mut v = vec![st.narg("schedule", Some(s))];
        if let Some(c) = c { v.push(st.narg("chunk_size", Some(st.e(IntExpr::new(c), (p, q))))); }
        v
    }
    / p:position!() "num_threads" _ "(" _ i:int() _ ")" q:position!() {
        vec![st.narg("num_threads", Some(st.e(IntExpr::new(i), (p, q))))]
    }
    / p:position!() "ordered" q:position!() {
        vec![st.narg("ordered", Some(st.e(BoolExpr { value: true }, (p, q))))]
    }
    // Note: the reference implementation `collapse` action reads `V0` with no semantic value (would throw
    // `bad_any_cast`); we keep the grammar and fail the clause instead.
    / "collapse" x:#{|_, _| peg::RuleResult::<Vec<CallArg>>::Failed} {x}
    / p:position!() "gpu" q:position!() {
        vec![st.narg("gpu", Some(st.e(BoolExpr { value: true }, (p, q))))]
    }
rule schedule_kind() -> String
    = s:$("static" / "dynamic" / "guided" / "auto" / "runtime") { s.to_string() }
rule int() -> i64 = s:$(['1'..='9'] ['0'..='9']*) {? s.parse().or(Err("int")) }
rule SPACE() = [' ' | '\t']+
rule _ = SPACE()*
}
}
