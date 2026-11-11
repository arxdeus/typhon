//! Port of codon/cir/transform/numpy/indexing.{h,cpp}

use rustc_hash::FxHashMap;
use typhon_common::mangle;
use typhon_ir::pass::{Pass, PassContext};
use typhon_ir::types::TypeKind;
use typhon_ir::util::cloning::CloneVisitor;
use typhon_ir::util::operator::Operator;
use typhon_ir::util::irtools::{call, get_func, is_call_of, is_call_of_types};
use typhon_ir::{Id, Module, NodeRef, ValueId, VarId, magic};
use typhon_ir_analyze::dataflow::cfg::SyntheticAssignKind;
use typhon_ir_analyze::dataflow::reaching::{RDInspector, RDResult};

use super::numpy::Walk;
use super::{FUSION_MODULE, numpy_should_run};

/// Port of the anonymous `Term`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TermKind {
    Int,
    Var,
    Len,
}

/// Linear term: `val`, `val*var` or `val*len(var)`. `var` is the `VarValue`.
#[derive(Clone, Copy, Debug)]
pub struct Term {
    pub kind: TermKind,
    pub val: i64,
    pub var: Option<ValueId>,
}

impl Term {
    pub fn val_term(v: i64) -> Self {
        Self { kind: TermKind::Int, val: v, var: None }
    }
    pub fn var_term(v: ValueId) -> Self {
        Self { kind: TermKind::Var, val: 1, var: Some(v) }
    }
    pub fn len_term(v: ValueId) -> Self {
        Self { kind: TermKind::Len, val: 1, var: Some(v) }
    }
    fn var_id(&self, m: &Module) -> Id {
        let v = m.as_var_value(self.var.unwrap()).unwrap().var;
        m.var_uid(v)
    }
    pub fn negate(&mut self) {
        self.val = -self.val;
    }
    pub fn multiply(&mut self, n: i64) {
        self.val *= n;
    }
    pub fn combine(&mut self, m: &Module, other: &Term) -> bool {
        if self.kind == other.kind && (self.kind == TermKind::Int || self.var_id(m) == other.var_id(m)) {
            self.val += other.val;
            return true;
        }
        false
    }
    pub fn zero(&self) -> bool {
        self.val == 0
    }
    pub fn str(&self, m: &Module) -> String {
        let name = |t: &Term| m.var_name(m.as_var_value(t.var.unwrap()).unwrap().var).to_string();
        let pre = if self.val == 1 { String::new() } else { format!("{}*", self.val) };
        match self.kind {
            TermKind::Int => self.val.to_string(),
            TermKind::Var => format!("{pre}{}", name(self)),
            TermKind::Len => format!("{pre}len({})", name(self)),
        }
    }
}

/// `t2s(terms)`.
pub fn t2s(m: &Module, terms: &[Term]) -> String {
    format!("[{}]", terms.iter().map(|t| t.str(m)).collect::<Vec<_>>().join(", "))
}

/// `simplify(terms)`.
pub fn simplify(m: &Module, terms: &mut Vec<Term>) {
    // Presumably the number of terms will be small,
    // so we use a simple quadratic algorithm.
    let mut i = 0;
    while i < terms.len() {
        let mut j = i + 1;
        while j < terms.len() {
            let t2 = terms[j];
            if terms[i].combine(m, &t2) {
                terms.remove(j);
            } else {
                j += 1;
            }
        }
        i += 1;
    }
    terms.retain(|t| !t.zero());
}

/// `checkTotal(terms, strict)`.
pub fn check_total(terms: &[Term], strict: bool) -> bool {
    let mut total = 0i64;
    for t in terms {
        match t.kind {
            TermKind::Int => total += t.val,
            TermKind::Var => {
                if t.val != 0 {
                    return false;
                }
            }
            // len is never negative
            TermKind::Len => {}
        }
    }
    if strict { total > 0 } else { total >= 0 }
}

fn less_check(m: &Module, t1: &[Term], t2: &[Term], strict: bool) -> bool {
    // Checking if: t1_0 + ... + t1_N < t2_0 + ... + t2_M
    let mut terms: Vec<Term> = t2.to_vec();
    terms.extend(t1.iter().map(|t| {
        let mut t = *t;
        t.negate();
        t
    }));
    simplify(m, &mut terms);
    check_total(&terms, strict)
}
pub fn less_than(m: &Module, t1: &[Term], t2: &[Term]) -> bool {
    less_check(m, t1, t2, true)
}
pub fn less_than_or_equal(m: &Module, t1: &[Term], t2: &[Term]) -> bool {
    less_check(m, t1, t2, false)
}

/// `replaceLoopVariable(terms, loopVar, replacement)`.
pub fn replace_loop_variable(m: &Module, terms: &[Term], loop_var: VarId, replacement: &[Term]) -> Vec<Term> {
    let lid = m.var_uid(loop_var);
    let mut ans = vec![];
    for t in terms {
        if t.kind == TermKind::Var && t.var_id(m) == lid {
            for r in replacement {
                let mut r = *r;
                r.multiply(t.val);
                ans.push(r);
            }
        } else {
            ans.push(*t);
        }
    }
    ans
}

fn is_array_type_dim(m: &mut Module, t: typhon_ir::TypeId, dim1: bool) -> bool {
    let mut r = matches!(m.type_kind(t), TypeKind::Record { .. })
        && m.type_name(t).starts_with(&format!("{}[", mangle::stdlib().ndarray));
    if r && dim1 {
        let g = m.type_generics(t);
        r = g.len() == 2 && g[0].is_type() && g[1].is_static() && g[1].static_value() == 1;
    }
    r
}

fn is_len(m: &Module, f: typhon_ir::FuncId) -> bool {
    m.var_name(f).starts_with(&format!("{}[", mangle::func("std.internal.builtin", "len")))
}

/// `parse(x, terms, negate)`.
pub fn parse_terms(m: &mut Module, x: ValueId, terms: &mut Vec<Term>, negate: bool) -> bool {
    let push = |terms: &mut Vec<Term>, mut t: Term| {
        if negate {
            t.negate();
        }
        terms.push(t);
    };
    if let Some(c) = m.as_int_const(x) {
        let v = c.val;
        push(terms, Term::val_term(v));
        return true;
    }
    if m.as_var_value(x).is_some() {
        push(terms, Term::var_term(x));
        return true;
    }
    let it = m.get_int_type();
    if let Some(c) = m.as_call_instr(x) {
        let args = c.args.clone();
        let callee = c.callee;
        if is_call_of_types(m, x, magic::ADD, &[Some(it), Some(it)], Some(it), true) {
            return parse_terms(m, args[0], terms, negate) && parse_terms(m, args[1], terms, negate);
        }
        if is_call_of_types(m, x, magic::SUB, &[Some(it), Some(it)], Some(it), true) {
            return parse_terms(m, args[0], terms, negate) && parse_terms(m, args[1], terms, !negate);
        }
        if args.len() == 1 {
            let at = m.value_type_mut(args[0]);
            if is_array_type_dim(m, at, false) && m.as_var_value(args[0]).is_some() {
                if is_call_of_types(m, x, "size", &[Some(at)], Some(it), true) {
                    push(terms, Term::len_term(args[0]));
                    return true;
                }
                if is_call_of_types(m, x, "len", &[Some(at)], Some(it), false) && get_func(m, callee).is_some_and(|f| is_len(m, f)) {
                    push(terms, Term::len_term(args[0]));
                    return true;
                }
            }
        }
    }
    false
}

#[derive(Clone, Copy, Debug)]
struct IndexInfo {
    orig: ValueId,
    arr: ValueId,
    idx: ValueId,
    item: Option<ValueId>,
}

fn find_array_indexes(m: &mut Module, body: ValueId) -> Vec<IndexInfo> {
    let mut indexes = vec![];
    Walk::children_first().run(m, body, &mut |_, m, v| {
        let Some(c) = m.as_call_instr(v) else { return };
        let args = c.args.clone();
        if args.is_empty() || m.as_var_value(args[0]).is_none() {
            return;
        }
        let arr_t = m.value_type_mut(args[0]);
        if !is_array_type_dim(m, arr_t, true) {
            return;
        }
        let it = m.get_int_type();
        if is_call_of_types(m, v, magic::GETITEM, &[Some(arr_t), Some(it)], None, true) {
            indexes.push(IndexInfo { orig: v, arr: args[0], idx: args[1], item: None });
        } else if args.len() == 3 {
            let bt = m.value_type_mut(args[2]);
            if is_call_of_types(m, v, magic::SETITEM, &[Some(arr_t), Some(it), Some(bt)], None, true) {
                indexes.push(IndexInfo { orig: v, arr: args[0], idx: args[1], item: Some(args[2]) });
            }
        }
    });
    let _ = is_call_of;
    indexes
}

fn elide_bounds_check(m: &mut Module, index: IndexInfo) {
    let mut cv = CloneVisitor::new(true);
    let arr_var = m.as_var_value(index.arr).unwrap().var;
    let at = m.value_type_mut(index.arr);
    let it = m.get_int_type();
    if let Some(item) = index.item {
        let bt = m.value_type_mut(item);
        let f = m
            .get_or_realize_func("_array1d_set_nocheck", &[at, it, bt], &[], FUSION_MODULE)
            .expect("setitem function not found");
        let a = m.nr_var_value(arr_var);
        let i = cv.clone_value(m, index.idx);
        let v = cv.clone_value(m, item);
        let c = call(m, f, vec![a, i, v]);
        m.replace_all_value(index.orig, c);
    } else {
        let f = m.get_or_realize_func("_array1d_get_nocheck", &[at, it], &[], FUSION_MODULE).expect("getitem function not found");
        let a = m.nr_var_value(arr_var);
        let i = cv.clone_value(m, index.idx);
        let c = call(m, f, vec![a, i]);
        m.replace_all_value(index.orig, c);
    }
}

fn is_original_loop_var(m: &Module, loc: ValueId, loop_: ValueId, rd: &RDInspector) -> bool {
    // The loop variable should have exactly two reaching definitions:
    //   - The initial assignment for the loop
    //   - The update assignment
    // Both are represented as `SyntheticAssignInstr` in the CFG.
    let loop_var = m.as_imperative_for_flow(loop_).unwrap().var;
    let defs = rd.get_reaching_definitions(m, loop_var, loc);
    if defs.len() != 2 {
        return false;
    }
    let cfg = rd.cfg();
    let (Some(s1), Some(s2)) = (cfg.as_synthetic_assign(defs[0].assignment), cfg.as_synthetic_assign(defs[1].assignment)) else {
        return false;
    };
    let (mut a, mut b) = (defs[0].assignment, defs[1].assignment);
    if s1.kind == SyntheticAssignKind::Add && s2.kind == SyntheticAssignKind::Known {
        std::mem::swap(&mut a, &mut b);
    } else if !(s1.kind == SyntheticAssignKind::Known && s2.kind == SyntheticAssignKind::Add) {
        return false;
    }
    let lid = m.value_uid(loop_);
    let src_ok = |x| match cfg.synth_source(x) {
        Some(NodeRef::Value(v)) => m.as_imperative_for_flow(v).is_some() && m.value_uid(v) == lid,
        _ => false,
    };
    src_ok(a) && src_ok(b)
}

fn is_alias_of_loop_var(m: &Module, v: ValueId, loop_: ValueId, rd: &RDInspector) -> Option<ValueId> {
    let var = m.as_var_value(v)?.var;
    let defs = rd.get_reaching_definitions(m, var, v);
    let loop_var = m.as_imperative_for_flow(loop_)?.var;
    if defs.len() != 1 || !defs[0].known() {
        return None;
    }
    let a = defs[0].assignee_value()?;
    let av = m.as_var_value(a)?;
    if m.var_uid(av.var) != m.var_uid(loop_var) || !is_original_loop_var(m, v, loop_, rd) {
        return None;
    }
    Some(a)
}

fn can_elide_bounds_check(
    m: &mut Module,
    loop_: ValueId,
    index: &IndexInfo,
    start: &[Term],
    stop: &[Term],
    rd: &RDInspector,
) -> bool {
    let (loop_var, step) = {
        let f = m.as_imperative_for_flow(loop_).unwrap();
        (f.var, f.step)
    };
    let mut idx_terms = vec![];
    if !parse_terms(m, index.idx, &mut idx_terms, false) {
        return false;
    }
    // First, check that all involved variables refer to a consistent
    // value. We do this by making sure there is just one reaching def
    // for all VarValues referring to the same Var.
    let mut reach: FxHashMap<Id, Id> = FxHashMap::default();
    let lid = m.var_uid(loop_var);
    let mut check = |m: &Module, v: ValueId| {
        let var = m.as_var_value(v).unwrap().var;
        let id = m.var_uid(var);
        if id == lid {
            is_original_loop_var(m, v, loop_, rd)
        } else {
            let defs = rd.get_reaching_definitions(m, var, v);
            if defs.len() != 1 {
                return false;
            }
            let rid = defs[0].id();
            *reach.entry(id).or_insert(rid) == rid
        }
    };
    if !check(m, index.arr) {
        return false;
    }
    for t in start.iter().chain(stop.iter()).chain(idx_terms.iter()) {
        if t.kind != TermKind::Int && !check(m, t.var.unwrap()) {
            return false;
        }
    }
    // Update vars that are aliases of the loop var. This can
    // happen in e.g. compound assignments which need to create
    // a temporary copy of the index.
    for t in idx_terms.iter_mut() {
        if t.kind != TermKind::Var {
            continue;
        }
        if let Some(lv) = is_alias_of_loop_var(m, t.var.unwrap(), loop_, rd) {
            t.var = Some(lv);
        }
    }
    // Next, see if we can prove that indexes are in range.
    let limit = [Term::len_term(index.arr)];
    let t1 = replace_loop_variable(m, &idx_terms, loop_var, start);
    let t2 = replace_loop_variable(m, &idx_terms, loop_var, stop);
    if step > 0 {
        less_than_or_equal(m, &[Term::val_term(0)], &t1) && less_than_or_equal(m, &t2, &limit)
    } else {
        less_than(m, &t1, &limit) && less_than_or_equal(m, &[Term::val_term(-1)], &t2)
    }
}

/// NumPy bounds check elision pass
pub struct NumPyBoundsCheckElisionPass {
    /// Key of the reaching definition analysis
    reaching_def_key: String,
}

impl NumPyBoundsCheckElisionPass {
    pub const KEY: &'static str = "core-numpy-bounds-check-elision";

    /// Constructs a NumPy bounds check elision pass.
    pub fn new(reaching_def_key: impl Into<String>) -> Self {
        Self { reaching_def_key: reaching_def_key.into() }
    }

    /// `visit(ImperativeForFlow*)`.
    fn visit_loop(&self, m: &mut Module, f: ValueId, parent: typhon_ir::FuncId, ctx: &dyn PassContext) {
        let Some(fl) = m.as_imperative_for_flow(f) else { return };
        let (step, var, body, start, end) = (fl.step, fl.var, fl.body, fl.start, fl.end);
        if step == 0 || m.is_global(var) {
            return;
        }
        let indexes = find_array_indexes(m, body);
        let mut start_terms = vec![];
        let mut stop_terms = vec![];
        if indexes.is_empty() || !parse_terms(m, start, &mut start_terms, false) || !parse_terms(m, end, &mut stop_terms, false) {
            return;
        }
        let Some(r) = ctx.get_analysis::<RDResult>(&self.reaching_def_key) else { return };
        let Some(rd) = r.results.get(&m.var_uid(parent)).cloned() else { return };
        for index in indexes {
            if can_elide_bounds_check(m, f, &index, &start_terms, &stop_terms, &rd) {
                elide_bounds_check(m, index);
            }
        }
    }
}

impl Pass for NumPyBoundsCheckElisionPass {
    fn key(&self) -> String {
        Self::KEY.to_string()
    }
    fn should_run(&self, m: &Module) -> bool {
        numpy_should_run(m)
    }
    fn run(&mut self, m: &mut Module, ctx: &dyn PassContext) {
        // Collect (loop, parent function) pairs first, then rewrite.
        // PORT-NOTE: C++ rewrites during the operator walk; replacements made here
        // only touch the loop body's index calls, so deferring is equivalent.
        let mut loops = vec![];
        Walk::new().run_module(m, &mut |w, m, v| {
            if m.as_imperative_for_flow(v).is_some()
                && let Some(p) = w.parent_func(m)
            {
                loops.push((v, p));
            }
        });
        for (l, p) in loops {
            self.visit_loop(m, l, p, ctx);
        }
    }
}
