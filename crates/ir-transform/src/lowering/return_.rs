
use std::collections::{BTreeSet, VecDeque};

use rustc_hash::{FxHashMap, FxHashSet};
use typhon_ir::util::irtools;
use typhon_ir::util::operator::{Operator, OperatorState};
use typhon_ir::dsl::CustomCategory;
use typhon_ir::{FuncId, Id, Module, NodeRef, TypeId, TypeKind, ValueId, ValueKind};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum ExitKind {
    Normal,
    Return,
    Throw,
    Break,
    Continue,
}

type Exit = (ExitKind, Id);
type Exits = BTreeSet<Exit>;
const NO_TARGET: Id = -1;

fn exit(kind: ExitKind) -> Exits {
    BTreeSet::from([(kind, NO_TARGET)])
}
fn has(e: &Exits, kind: ExitKind) -> bool {
    e.contains(&(kind, NO_TARGET))
}
fn merge(dst: &mut Exits, src: &Exits) {
    dst.extend(src.iter().copied());
}
fn sequence(mut first: Exits, second: &Exits) -> Exits {
    if first.remove(&(ExitKind::Normal, NO_TARGET)) {
        merge(&mut first, second);
    }
    first
}

fn is_nullable(m: &Module, t: TypeId, none: TypeId) -> bool {
    if m.type_is(t, none) {
        return true;
    }
    match m.type_kind(t) {
        TypeKind::Optional { .. } => true,
        TypeKind::Union { types } => types.iter().any(|x| is_nullable(m, *x, none)),
        _ => false,
    }
}

fn make_none(m: &mut Module, t: TypeId) -> ValueId {
    if let TypeKind::Union { types } = m.type_kind(t).clone() {
        let none = m.get_none_type();
        let idx = types.iter().position(|c| is_nullable(m, *c, none)).expect("union has no nullable member");
        let member = types[idx];
        let value = make_none(m, member);
        let it = m.get_int_type();
        let ctor = m
            .get_or_realize_method(t, "_make", &[it, member], &[typhon_ir::Generic::Type(t)])
            .expect("union constructor not found");
        let i = m.get_int(idx as i64);
        return irtools::call(m, ctor, vec![i, value]);
    }
    let ctor = m
        .get_or_realize_method(t, "__new__", &[], &[])
        .unwrap_or_else(|| panic!("null constructor not found for {}", m.type_name(t)));
    irtools::call(m, ctor, vec![])
}

struct ReturnAnalysis<'a> {
    non_returning: &'a FxHashSet<FuncId>,
}

impl ReturnAnalysis<'_> {
    fn branch(&self, m: &Module, cond: ValueId, t: Option<ValueId>, f: Option<ValueId>) -> Exits {
        let result = self.analyze(m, Some(cond));
        if !has(&result, ExitKind::Normal) {
            return result;
        }
        let mut alt = Exits::new();
        let c = m.as_bool_const(cond).map(|c| c.val);
        if c != Some(false) {
            merge(&mut alt, &self.analyze(m, t));
        }
        if c != Some(true) {
            merge(&mut alt, &self.analyze(m, f));
        }
        sequence(result, &alt)
    }

    fn loop_(&self, flow_id: Id, body: &Exits, may_be_empty: bool) -> Exits {
        let mut result = if may_be_empty { exit(ExitKind::Normal) } else { Exits::new() };
        for &(k, tgt) in body {
            let local = tgt == NO_TARGET || tgt == flow_id;
            if k == ExitKind::Normal || (k == ExitKind::Continue && local) {
                continue;
            }
            if k == ExitKind::Break && local {
                result.insert((ExitKind::Normal, NO_TARGET));
            } else {
                result.insert((k, tgt));
            }
        }
        result
    }

    fn analyze(&self, m: &Module, value: Option<ValueId>) -> Exits {
        let Some(v) = value else { return exit(ExitKind::Normal) };
        let v = m.actual_value(v);
        let uid = m.value_uid(v);
        match m.kind(v) {
            ValueKind::SeriesFlow(s) => {
                let mut result = exit(ExitKind::Normal);
                for c in &s.series {
                    if !has(&result, ExitKind::Normal) {
                        break;
                    }
                    result = sequence(result, &self.analyze(m, Some(*c)));
                }
                result
            }
            ValueKind::IfFlow(f) => self.branch(m, f.cond, Some(f.true_branch), f.false_branch),
            ValueKind::TernaryInstr(f) => self.branch(m, f.cond, Some(f.true_value), Some(f.false_value)),
            ValueKind::WhileFlow(w) => {
                let result = self.analyze(m, Some(w.cond));
                let c = m.as_bool_const(w.cond).map(|c| c.val);
                if c == Some(false) {
                    return result;
                }
                let body = self.analyze(m, Some(w.body));
                sequence(result, &self.loop_(uid, &body, c.is_none()))
            }
            ValueKind::ForFlow(f) => {
                let result = self.analyze(m, Some(f.iter));
                let body = self.analyze(m, Some(f.body));
                let mut out = self.loop_(uid, &body, true);
                out.insert((ExitKind::Throw, NO_TARGET));
                sequence(result, &out)
            }
            ValueKind::ImperativeForFlow(f) => {
                let result = sequence(self.analyze(m, Some(f.start)), &self.analyze(m, Some(f.end)));
                let s = m.as_int_const(f.start).map(|c| c.val);
                let e = m.as_int_const(f.end).map(|c| c.val);
                let nonempty = match (s, e) {
                    (Some(s), Some(e)) => (f.step > 0 && s < e) || (f.step < 0 && s > e),
                    _ => false,
                };
                let body = self.analyze(m, Some(f.body));
                let mut out = self.loop_(uid, &body, !nonempty);
                if f.is_parallel() {
                    out.insert((ExitKind::Throw, NO_TARGET));
                }
                if has(&body, ExitKind::Normal)
                    || has(&body, ExitKind::Continue)
                    || body.contains(&(ExitKind::Continue, uid))
                {
                    out.insert((ExitKind::Normal, NO_TARGET));
                }
                sequence(result, &out)
            }
            ValueKind::TryCatchFlow(t) => {
                let body = self.analyze(m, Some(t.body));
                let mut result = sequence(body.clone(), &self.analyze(m, t.else_));
                if has(&body, ExitKind::Throw) {
                    for c in &t.catches {
                        merge(&mut result, &self.analyze(m, Some(c.handler)));
                    }
                }
                if t.finally.is_some() && !result.is_empty() {
                    let mut cleanup = self.analyze(m, t.finally);
                    if !cleanup.remove(&(ExitKind::Normal, NO_TARGET)) {
                        result.clear();
                    }
                    merge(&mut result, &cleanup);
                }
                result
            }
            ValueKind::ReturnInstr(r) => sequence(self.analyze(m, r.value), &exit(ExitKind::Return)),
            ValueKind::ThrowInstr(r) => sequence(self.analyze(m, r.value), &exit(ExitKind::Throw)),
            ValueKind::BreakInstr(b) => {
                BTreeSet::from([(ExitKind::Break, b.loop_.map(|l| m.value_uid(l)).unwrap_or(NO_TARGET))])
            }
            ValueKind::ContinueInstr(b) => {
                BTreeSet::from([(ExitKind::Continue, b.loop_.map(|l| m.value_uid(l)).unwrap_or(NO_TARGET))])
            }
            ValueKind::YieldInstr(y) => {
                let mut out = exit(if y.final_ { ExitKind::Return } else { ExitKind::Normal });
                out.insert((ExitKind::Throw, NO_TARGET));
                sequence(self.analyze(m, y.value), &out)
            }
            k if k.is_flow() || matches!(k, ValueKind::Custom(c) if c.category == CustomCategory::Instr) => {
                [ExitKind::Normal, ExitKind::Return, ExitKind::Throw, ExitKind::Break, ExitKind::Continue]
                    .into_iter()
                    .map(|k| (k, NO_TARGET))
                    .collect()
            }
            k => {
                let mut result = exit(ExitKind::Normal);
                for op in m.used_values(v.into()) {
                    if !has(&result, ExitKind::Normal) {
                        break;
                    }
                    result = sequence(result, &self.analyze(m, Some(op)));
                }
                match k {
                    ValueKind::CallInstr(c) => {
                        let callee = irtools::get_func(m, c.callee).filter(|f| m.func(*f).unwrap().is_bodied());
                        let mut out = exit(ExitKind::Throw);
                        if !callee.is_some_and(|f| self.non_returning.contains(&f)) {
                            out.insert((ExitKind::Normal, NO_TARGET));
                        }
                        result = sequence(result, &out);
                    }
                    ValueKind::AwaitInstr(_) | ValueKind::YieldInInstr(_) => {
                        let out = BTreeSet::from([(ExitKind::Normal, NO_TARGET), (ExitKind::Throw, NO_TARGET)]);
                        result = sequence(result, &out);
                    }
                    _ => {}
                }
                result
            }
        }
    }
}

#[derive(Default)]
struct Calls {
    state: OperatorState,
    callers: FxHashMap<FuncId, Vec<Option<FuncId>>>,
}

impl Operator for Calls {
    super::operator_state!();
    fn handle_call_instr(&mut self, m: &mut Module, v: ValueId) {
        let callee = m.as_call_instr(v).unwrap().callee;
        if let Some(f) = irtools::get_func(m, callee).filter(|f| m.func(*f).unwrap().is_bodied()) {
            let p = self.parent_func(m).filter(|p| m.func(*p).is_some_and(|d| d.is_bodied()));
            let e = self.callers.entry(f).or_default();
            if !e.contains(&p) {
                e.push(p);
            }
        }
    }
}

/// Appends an explicit `return None`/`raise` to functions whose end is reachable but
/// which must return a value.
#[derive(Default)]
pub struct MissingReturnLowering;

impl MissingReturnLowering {
    pub const KEY: &'static str = "core-missing-return-lowering";
    pub fn new() -> Self {
        Self
    }

    pub fn run_module(&mut self, m: &mut Module) {
        loop {
            if !self.run_once(m) {
                break;
            }
        }
    }

    fn run_once(&mut self, m: &mut Module) -> bool {
        let is_bodied = |m: &Module, f: FuncId| m.func(f).is_some_and(|d| d.is_bodied());
        let mut functions = Vec::new();
        let main = m.main_func();
        if is_bodied(m, main) {
            functions.push(main);
        }
        for v in m.vars() {
            if m.is_var_alive(v) && is_bodied(m, v) {
                functions.push(v);
            }
        }

        let mut calls = Calls::default();
        typhon_ir::util::operator::run_operator(&mut calls, m);

        let mut non_returning = FxHashSet::default();
        let mut pending: VecDeque<FuncId> = functions.iter().copied().collect();
        let mut queued: FxHashSet<FuncId> = functions.iter().copied().collect();
        while let Some(f) = pending.pop_front() {
            queued.remove(&f);
            let fd = m.func(f).unwrap();
            if fd.generator || fd.is_async || fd.body().is_none() || non_returning.contains(&f) {
                continue;
            }
            let outcomes = ReturnAnalysis { non_returning: &non_returning }.analyze(m, fd.body());
            if has(&outcomes, ExitKind::Normal) || has(&outcomes, ExitKind::Return) {
                continue;
            }
            non_returning.insert(f);
            for c in calls.callers.get(&f).cloned().unwrap_or_default().into_iter().flatten() {
                if queued.insert(c) {
                    pending.push_back(c);
                }
            }
        }

        let analysis = ReturnAnalysis { non_returning: &non_returning };
        let mut helper = None;
        let mut changed = false;
        for f in functions {
            let fd = m.func(f).unwrap();
            let Some(body) = fd.body() else { continue };
            if fd.generator {
                continue;
            }
            let is_async = fd.is_async;
            let name = fd.unmangled_name.clone();
            let Some(mut result) = irtools::return_type(m, f) else { continue };
            if is_async {
                if let TypeKind::Generator { base } = m.type_kind(result) {
                    result = *base;
                }
            }
            let none = m.get_none_type();
            if m.type_is(result, none)
                || matches!(m.type_kind(result), TypeKind::Optional { .. })
                || !has(&analysis.analyze(m, Some(body)), ExitKind::Normal)
            {
                continue;
            }
            let info = m.src_info(f.into());
            let ending = if is_nullable(m, result, none) {
                let n = make_none(m, result);
                let r = m.nr_return(Some(n));
                m.set_src_info(r.into(), info.clone());
                r
            } else {
                let h = *helper.get_or_insert_with(|| {
                    let s = m.get_string_type();
                    let i = m.get_int_type();
                    m.get_or_realize_func("_missing_return_error", &[s, s, s, i, i], &[], "std.internal.types.error")
                });
                let h = h.expect("missing-return helper not found");
                let message = format!(
                    "function '{}' reached its end without returning a value ({}:{}:{})",
                    name, info.file, info.line, info.col
                );
                let args = vec![
                    m.get_string(message),
                    m.get_string(name.clone()),
                    m.get_string(info.file.to_string()),
                    m.get_int(info.line as i64),
                    m.get_int(info.col as i64),
                ];
                let exc = irtools::call(m, h, args);
                let t = m.nr_throw(Some(exc));
                m.set_src_info(t.into(), info.clone());
                t
            };
            let body = if m.as_series_flow(body).is_some() {
                body
            } else {
                let s = m.nr_series_flow();
                m.set_src_info(s.into(), info.clone());
                m.series_push(s, body);
                m.set_func_body(f, Some(s));
                s
            };
            m.series_push(body, ending);
            changed = true;
            typhon_common::log_flag!(IR, "missing-return: {} [{}]", m.name(NodeRef::Var(f)), info);
        }
        changed
    }
}

impl typhon_ir::pass::Pass for MissingReturnLowering {
    fn key(&self) -> String {
        Self::KEY.to_string()
    }
    fn run(&mut self, m: &mut Module, _ctx: &dyn typhon_ir::pass::PassContext) {
        self.run_module(m);
    }
}
