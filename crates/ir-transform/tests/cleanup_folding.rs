//! Tests for cleanup and folding passes on small hand-built modules.

use rustc_hash::FxHashMap;
use typhon_common::Options;
use typhon_common::mangle::func as mangled;
use typhon_ir::attribute::{Attribute, KeyValueAttribute};
use typhon_ir::pass::{Analysis, AnalysisResult, NullPassContext, Pass, PassContext};
use typhon_ir::util::irtools::call;
use typhon_ir::util::side_effect::SideEffectStatus;
use typhon_ir::{FuncId, Module, TypeId, ValueId, magic};
use typhon_ir_analyze::module::side_effect::SideEffectResult;
use typhon_ir_transform::cleanup::*;
use typhon_ir_transform::folding::*;
use typhon_ir_transform::manager::PassManager;

const SE_KEY: &str = "**pure**";

/// Marks every value pure (CanonicalizationKeepsUnchangedCalls fixture).
/// Marks only constants pure.
struct ConstPureAnalysis;
impl Analysis for ConstPureAnalysis {
    fn key(&self) -> String {
        SE_KEY.into()
    }
    fn run(&mut self, m: &Module, _ctx: &dyn PassContext) -> Box<dyn AnalysisResult> {
        let mut result = FxHashMap::default();
        for v in m.values() {
            if m.is_const(v) {
                result.insert(m.value_uid(v), SideEffectStatus::Pure);
            }
        }
        Box::new(SideEffectResult { result })
    }
}

struct PureAnalysis;
impl Analysis for PureAnalysis {
    fn key(&self) -> String {
        SE_KEY.into()
    }
    fn run(&mut self, m: &Module, _ctx: &dyn PassContext) -> Box<dyn AnalysisResult> {
        let mut result = FxHashMap::default();
        for v in m.values() {
            result.insert(m.value_uid(v), SideEffectStatus::Pure);
        }
        Box::new(SideEffectResult { result })
    }
}

fn method(m: &mut Module, name: &str, parent: TypeId, args: Vec<TypeId>, ret: TypeId) -> FuncId {
    let f = m.nr_bodied_func(format!("{}.{name}", m.type_name(parent)));
    {
        let fd = m.func_mut(f).unwrap();
        fd.unmangled_name = name.into();
        fd.parent_type = Some(parent);
    }
    let names: Vec<String> = (0..args.len()).map(|i| format!("a{i}")).collect();
    let ft = m.nr_func_type(format!("{name}.sig.{}", args.len()), ret, args, false);
    m.realize_func(f, ft, &names);
    f
}

fn set_main_body(m: &mut Module, vals: &[ValueId]) -> ValueId {
    let body = m.nr_series_flow();
    for v in vals {
        m.series_push(body, *v);
    }
    let main = m.main_func();
    m.set_func_body(main, Some(body));
    body
}

fn pm() -> PassManager {
    let mut o = Options::default();
    o.pmempty = true;
    PassManager::new(o)
}

#[test]
fn folds_int_constants() {
    let mut m = Module::new("t");
    let int = m.get_int_type();
    let add = method(&mut m, magic::ADD, int, vec![int, int], int);
    let mul = method(&mut m, magic::MUL, int, vec![int, int], int);
    let (a, b) = (m.get_int(2), m.get_int(3));
    let sum = call(&mut m, add, vec![a, b]);
    let c = m.get_int(4);
    let prod = call(&mut m, mul, vec![sum, c]);
    let ret = m.nr_return(Some(prod));
    set_main_body(&mut m, &[ret]);

    let mut fp = FoldingPass::new();
    fp.run(&mut m, &NullPassContext);
    let r = m.as_return_instr(ret).unwrap().value.unwrap();
    assert_eq!(m.as_int_const(r).map(|c| c.val), Some(20));
    assert_eq!(fp.num_replacements(), 2);
}

#[test]
fn folds_multiply_by_zero_keeping_operand() {
    let mut m = Module::new("t");
    let int = m.get_int_type();
    let mul = method(&mut m, magic::MUL, int, vec![int, int], int);
    let x = m.nr_var(int, false, "x");
    let xv = m.nr_var_value(x);
    let z = m.get_int(0);
    let prod = call(&mut m, mul, vec![z, xv]);
    let ret = m.nr_return(Some(prod));
    set_main_body(&mut m, &[ret]);
    FoldingPass::new().run(&mut m, &NullPassContext);
    let r = m.as_return_instr(ret).unwrap().value.unwrap();
    let fi = m.as_flow_instr(r).expect("flow instr");
    assert_eq!(m.as_int_const(fi.val).map(|c| c.val), Some(0));
}

#[test]
fn no_fold_division_by_zero() {
    let mut m = Module::new("t");
    let int = m.get_int_type();
    let div = method(&mut m, magic::FLOOR_DIV, int, vec![int, int], int);
    let (a, b) = (m.get_int(7), m.get_int(0));
    let d = call(&mut m, div, vec![a, b]);
    let ret = m.nr_return(Some(d));
    set_main_body(&mut m, &[ret]);
    let mut fp = FoldingPass::new();
    fp.run(&mut m, &NullPassContext);
    assert_eq!(fp.num_replacements(), 0);
}

#[test]
fn dce_folds_constant_if_and_removes_pure() {
    let mut m = Module::new("t");
    let int = m.get_int_type();
    let cond = m.get_bool(false);
    let t = m.nr_series_flow();
    let one = m.get_int(1);
    let r1 = m.nr_return(Some(one));
    m.series_push(t, r1);
    let f = m.nr_series_flow();
    let two = m.get_int(2);
    let r2 = m.nr_return(Some(two));
    m.series_push(f, r2);
    let iff = m.nr_if_flow(cond, t, Some(f));
    let dead = m.get_int(5);
    let _ = int;
    set_main_body(&mut m, &[dead, iff]);

    let mut p = pm();
    p.register_analysis(Box::new(ConstPureAnalysis), vec![]);
    p.register_pass(Box::new(DeadCodeCleanupPass::new(SE_KEY)), "", vec![SE_KEY.into()], vec![SE_KEY.into()]);
    p.run(&mut m);
    // the pure const is deleted; the if-flow is replaced by a clone of the false branch
    let main = m.main_func();
    let body = m.func_body(main).unwrap();
    assert_eq!(m.as_series_flow(body).unwrap().series, vec![iff]);
    let resolved = m.actual_value(iff);
    assert_ne!(resolved, f);
    let s = m.as_series_flow(resolved).expect("series");
    let ret = s.series[0];
    let v = m.as_return_instr(ret).unwrap().value.unwrap();
    assert_eq!(m.as_int_const(v).map(|c| c.val), Some(2));
}

#[test]
fn canonicalization_flattens_series_and_keeps_unchanged_calls() {
    let mut m = Module::new("t");
    let int = m.get_int_type();
    let combine = method(&mut m, "combine", int, vec![int, int], int);
    let (a, b) = (m.get_int(1), m.get_int(2));
    let c = call(&mut m, combine, vec![a, b]);
    let inner = m.nr_series_flow();
    m.series_push(inner, c);
    let body = set_main_body(&mut m, &[inner]);

    let mut p = pm();
    p.register_analysis(Box::new(PureAnalysis), vec![]);
    p.register_pass(Box::new(CanonicalizationPass::new(SE_KEY)), "", vec![SE_KEY.into()], vec![SE_KEY.into()]);
    let n = m.values().len();
    p.run(&mut m);
    assert_eq!(m.as_series_flow(body).unwrap().series, vec![c]);
    assert_eq!(m.values().len(), n);

    // right chain (1 `combine` (2 `combine` 3)) with associativity -> left chain
    let mut kv = KeyValueAttribute::default();
    kv.attributes.insert(mangled("std.internal.attributes", "associative"), String::new());
    m.set_attribute(combine.into(), Attribute::KeyValue(kv));
    let three = m.get_int(3);
    let inner_c = call(&mut m, combine, vec![b, three]);
    let right = call(&mut m, combine, vec![a, inner_c]);
    m.series_push(body, right);
    p.run(&mut m);
    let canon = m.actual_value(right);
    assert_ne!(canon, right);
    let args = m.as_call_instr(canon).unwrap().args.clone();
    let left = m.as_call_instr(args[0]).expect("left call");
    assert_eq!(left.args, vec![a, b]);
    assert_eq!(args[1], three);
}

#[test]
fn global_demotion() {
    let mut m = Module::new("t");
    let int = m.get_int_type();
    let g = m.nr_var(int, true, "g");
    let one = m.get_int(1);
    let asg = m.nr_assign(g, one);
    set_main_body(&mut m, &[asg]);
    let mut gd = GlobalDemotionPass::new();
    gd.run(&mut m, &NullPassContext);
    assert_eq!(gd.num_demotions(), 1);
    assert!(!m.is_global(g));
    let main = m.main_func();
    assert!(m.func(main).unwrap().symbols().contains(&g));
}

#[test]
fn folding_group_repeat_logic() {
    let g = FoldingPassGroup::new("se", "rd", "gv", 5, true);
    assert_eq!(g.key(), FoldingPassGroup::KEY);
    // nothing ran: no replacements, so no repeat
    assert!(!g.should_repeat(1));
}

#[test]
fn replacer_runs() {
    let mut m = Module::new("t");
    let a = m.get_int(1);
    let b = m.get_int(2);
    let r = m.nr_return(Some(a));
    set_main_body(&mut m, &[r]);
    m.replace_all_value(a, b);
    ReplaceCleanupPass::new().run(&mut m, &NullPassContext);
    assert_eq!(m.as_return_instr(r).unwrap().value.map(|v| m.actual_value(v)), Some(b));
}
