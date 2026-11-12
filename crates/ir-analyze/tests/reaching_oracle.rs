//! Reaching definitions: the lazy bitset query (`RDInspector`) must agree with a
//! straightforward eager computation (per-block var -> set of incoming defs, built
//! from an independent set-based fixpoint) at every location of a few CFGs.

use std::collections::BTreeSet;
use std::rc::Rc;

use rustc_hash::FxHashMap;
use typhon_ir::{Id, Module, ValueId, VarId, reset_id};
use typhon_ir_analyze::dataflow::cfg::{CFGraph, CfValue, build_cf_graph};
use typhon_ir_analyze::dataflow::reaching::{RDInspector, get_generated, get_killed};

type Defs = BTreeSet<(Id, Option<Id>)>; // (assignment uid, assignee uid)

/// Eager oracle: in-sets of (var -> defs) per block via iterative set fixpoint.
fn eager_in_sets(cfg: &CFGraph, m: &Module) -> Vec<FxHashMap<Id, Defs>> {
    let n = cfg.size();
    let key = |v: CfValue| -> (Id, (Id, Option<Id>)) {
        let (var, d) = get_generated(cfg, m, v).unwrap();
        (var, (d.assignment_uid(), d.assignee_uid()))
    };
    let mut ins: Vec<FxHashMap<Id, Defs>> = vec![FxHashMap::default(); n];
    let mut outs: Vec<FxHashMap<Id, Defs>> = vec![FxHashMap::default(); n];
    loop {
        let mut changed = false;
        for b in cfg.block_ids() {
            let mut in_: FxHashMap<Id, Defs> = FxHashMap::default();
            for p in cfg.block(b).predecessors() {
                for (v, s) in &outs[p] {
                    in_.entry(*v).or_default().extend(s.iter().copied());
                }
            }
            let mut out = in_.clone();
            for &val in cfg.block(b).values() {
                if let CfValue::Real(v) = val
                    && m.as_pointer_value(v).is_some()
                {
                    continue;
                }
                if get_killed(cfg, m, val) != -1 {
                    let (var, d) = key(val);
                    out.insert(var, [d].into_iter().collect());
                }
            }
            if in_ != ins[b] || out != outs[b] {
                changed = true;
                ins[b] = in_;
                outs[b] = out;
            }
        }
        if !changed {
            return ins;
        }
    }
}

/// Compare the inspector against the oracle for every (var, value) pair in the CFG.
fn check(m: &Module, f: VarId, vars: &[VarId]) -> usize {
    let cfg = Rc::new(build_cf_graph(m, f));
    let mut rd = RDInspector::new(cfg.clone());
    rd.analyze(m);
    let ins = eager_in_sets(&cfg, m);
    let mut checked = 0;
    for b in cfg.block_ids() {
        for (pos, &loc) in cfg.block(b).values().iter().enumerate() {
            let CfValue::Real(loc_v) = loc else { continue };
            for &var in vars {
                let vuid = m.var_uid(var);
                // last local def strictly before or at `loc` in this block, else block in-set
                let local = cfg.block(b).values()[..=pos]
                    .iter()
                    .rev()
                    .find(|&&x| get_killed(&cfg, m, x) == vuid)
                    .map(|&x| get_generated(&cfg, m, x).unwrap().1);
                let expect: Defs = match local {
                    Some(d) => [(d.assignment_uid(), d.assignee_uid())].into_iter().collect(),
                    None if b == cfg.entry_block() => Defs::new(),
                    None => ins[b].get(&vuid).cloned().unwrap_or_default(),
                };
                let got: Defs = rd
                    .get_reaching_definitions(m, var, loc_v)
                    .iter()
                    .map(|d| (d.assignment_uid(), d.assignee_uid()))
                    .collect();
                assert_eq!(got, expect, "block {b} pos {pos} var {vuid}");
                checked += 1;
            }
        }
    }
    checked
}

fn func(m: &mut Module) -> (VarId, ValueId, VarId, VarId) {
    let f = m.nr_bodied_func("rd_f");
    let b = m.nr_series_flow();
    m.set_func_body(f, Some(b));
    let it = m.get_int_type();
    let x = m.nr_var(it, false, "x");
    let y = m.nr_var(it, false, "y");
    m.func_push_symbol(f, x);
    m.func_push_symbol(f, y);
    (f, b, x, y)
}

fn assign(m: &mut Module, s: ValueId, v: VarId, k: i64) {
    let c = m.get_int(k);
    let a = m.nr_assign(v, c);
    m.series_push(s, a);
}

fn use_(m: &mut Module, s: ValueId, v: VarId) {
    let u = m.nr_var_value(v);
    m.series_push(s, u);
}

#[test]
fn oracle_straight_line_and_if() {
    reset_id();
    let mut m = Module::new("t");
    let (f, b, x, y) = func(&mut m);
    assign(&mut m, b, x, 0);
    use_(&mut m, b, x);
    let (tb, fb) = (m.nr_series_flow(), m.nr_series_flow());
    let c = m.get_bool(true);
    let iff = m.nr_if_flow(c, tb, Some(fb));
    m.series_push(b, iff);
    assign(&mut m, tb, x, 1);
    assign(&mut m, tb, y, 2);
    use_(&mut m, tb, x);
    assign(&mut m, fb, y, 3);
    use_(&mut m, b, x);
    use_(&mut m, b, y);
    assign(&mut m, b, x, 4);
    use_(&mut m, b, x);
    assert!(check(&m, f, &[x, y]) > 0);
}

#[test]
fn oracle_nested_loops_break_continue() {
    reset_id();
    let mut m = Module::new("t");
    let (f, b, x, y) = func(&mut m);
    assign(&mut m, b, x, 0);
    assign(&mut m, b, y, 0);
    let outer_body = m.nr_series_flow();
    let c1 = m.get_bool(true);
    let outer = m.nr_while_flow(c1, outer_body);
    m.series_push(b, outer);
    use_(&mut m, outer_body, x);
    let inner_body = m.nr_series_flow();
    let c2 = m.get_bool(false);
    let inner = m.nr_while_flow(c2, inner_body);
    m.series_push(outer_body, inner);
    assign(&mut m, inner_body, y, 1);
    let (tb, fb) = (m.nr_series_flow(), m.nr_series_flow());
    let c3 = m.get_bool(true);
    let iff = m.nr_if_flow(c3, tb, Some(fb));
    m.series_push(inner_body, iff);
    assign(&mut m, tb, x, 2);
    let brk = m.nr_break(Some(inner));
    m.series_push(tb, brk);
    assign(&mut m, fb, y, 3);
    let cont = m.nr_continue(Some(inner));
    m.series_push(fb, cont);
    use_(&mut m, outer_body, y);
    assign(&mut m, outer_body, x, 5);
    use_(&mut m, b, x);
    use_(&mut m, b, y);
    assert!(check(&m, f, &[x, y]) > 0);
}

#[test]
fn oracle_try_catch_and_for() {
    reset_id();
    let mut m = Module::new("t");
    let (f, b, x, y) = func(&mut m);
    assign(&mut m, b, x, 0);
    let (body, exc, fin) = (m.nr_series_flow(), m.nr_series_flow(), m.nr_series_flow());
    let tc = m.nr_try_catch_flow(body, Some(fin), None);
    m.try_catch_push(tc, exc, None, None);
    m.series_push(b, tc);
    assign(&mut m, body, x, 1);
    use_(&mut m, body, x);
    assign(&mut m, body, y, 2);
    assign(&mut m, exc, x, 3);
    use_(&mut m, fin, x);
    use_(&mut m, fin, y);
    let lb = m.nr_series_flow();
    let (s, e) = (m.get_int(0), m.get_int(10));
    let fl = m.nr_imperative_for_flow(s, 1, e, lb, y);
    m.series_push(b, fl);
    use_(&mut m, lb, y);
    assign(&mut m, lb, x, 4);
    use_(&mut m, b, x);
    use_(&mut m, b, y);
    assert!(check(&m, f, &[x, y]) > 0);
}
