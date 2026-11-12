//! Unit tests for the pure parts of the NumPy optimizations (port of
//!
//! Note: parsing real ndarray expressions, fusion codegen, inlining and the
//! lifetime pass need a `TypeRealizer` (ndarray generics, `std.numpy.fusion`
//! helpers) and are validated only by the end-to-end suite (`test/numpy/**`).

use indexmap::IndexMap;
use typhon_ir::Module;
use typhon_ir::pass::{Analysis, NullPassContext, Pass};
use typhon_ir_transform::numpy::expr::{BroadcastInfo, ExprArena, Op};
use typhon_ir_transform::numpy::forward::{Forwarding, connected_components, get_forwarding_root, has_cycle};
use typhon_ir_transform::numpy::indexing::{Term, check_total, less_than, less_than_or_equal, simplify};
use typhon_ir_transform::numpy::*;

fn arr(dtype: TypeCode, ndim: i64) -> NumPyType {
    NumPyType::new(dtype, ndim)
}

#[test]
fn pass_keys() {
    assert_eq!(NumPyExpressionAnalysis::KEY, "core-numpy-expressions");
    assert_eq!(NumPyFusionPass::KEY, "core-numpy-fusion");
    assert_eq!(NumPyInlinePass::KEY, "core-numpy-inline");
    assert_eq!(NumPyLifetimePass::KEY, "core-numpy-lifetime");
    assert_eq!(NumPyBoundsCheckElisionPass::KEY, "core-numpy-bounds-check-elision");
    assert_eq!(NumPyFusionPass::new("e", "s").key(), "core-numpy-fusion");
    assert_eq!(FUSION_MODULE, "std.numpy.fusion");
}

#[test]
fn type_mapping_and_display() {
    let mut m = Module::new("t");
    let t = NumPyPrimitiveTypes::new(&mut m);
    assert_eq!(NumPyType::get(&mut m, t.f64, &t), NumPyType::scalar(TypeCode::F64));
    assert_eq!(NumPyType::get(&mut m, t.i64, &t), NumPyType::scalar(TypeCode::I64));
    assert_eq!(NumPyType::get(&mut m, t.u8, &t), NumPyType::scalar(TypeCode::U8));
    assert_eq!(NumPyType::get(&mut m, t.bool_, &t), NumPyType::scalar(TypeCode::Bool));
    let s = m.get_string_type();
    assert!(!NumPyType::get(&mut m, s, &t).is_some());
    assert_eq!(arr(TypeCode::ArrF32, 2).to_string(), "array[f32, 2]");
    assert_eq!(NumPyType::scalar(TypeCode::C128).to_string(), "c128");
    assert!(arr(TypeCode::ArrBool, 1).is_array());
    assert!(!NumPyType::scalar(TypeCode::C128).is_array());
    assert!(m.type_is(arr(TypeCode::ArrI32, 1).ir_base_type(&t), t.i32));
    // no ndarray types -> passes are skipped
    assert!(!numpy_should_run(&m));
    assert!(!NumPyFusionPass::new("e", "s").should_run(&m));
}

#[test]
fn opstrings_and_costs() {
    assert_eq!(Op::Min.opstring(), "minimum");
    assert_eq!(Op::None.opstring(), "a");
    assert_eq!(Op::ClipMax.opstring(), "clip_max");
    assert_eq!(Op::Arctan2.opcost(), 35);
    assert_eq!(Op::Matmul.opcost(), 20);
    assert_eq!(Op::Clip.opcost(), 2);
}

#[test]
fn cost_model() {
    let mut m = Module::new("t");
    let x = m.get_int(0);
    let two = m.get_int(2);
    let three = m.get_int(3);
    let it = m.get_int_type();
    let av = m.nr_var(it, false, "a");
    let x = { let _ = x; m.nr_var_value(av) };
    let mut a = ExprArena::new();
    let f64a = arr(TypeCode::ArrF64, 1);
    let l = a.leaf(f64a, x);
    let r = a.leaf(f64a, x);
    // add: 1
    let add = a.node(f64a, x, Op::Add, Some(l), Some(r), None);
    assert_eq!(a.cost(&m, add), 1);
    // div by constant -> 1, div by array -> 8
    let c2 = a.leaf(NumPyType::scalar(TypeCode::I64), two);
    let div = a.node(f64a, x, Op::TrueDiv, Some(l), Some(c2), None);
    assert_eq!(a.cost(&m, div), 1);
    let div2 = a.node(f64a, x, Op::TrueDiv, Some(l), Some(r), None);
    assert_eq!(a.cost(&m, div2), 8);
    // pow 2 -> 1, pow 3 -> 5
    let p2 = a.node(f64a, x, Op::Pow, Some(l), Some(c2), None);
    assert_eq!(a.cost(&m, p2), 1);
    let c3 = a.leaf(NumPyType::scalar(TypeCode::I64), three);
    let p3 = a.node(f64a, x, Op::Pow, Some(l), Some(c3), None);
    assert_eq!(a.cost(&m, p3), 5);
    // vectorized sin on f64: max(10*3, 64) = 64; on f32: max(60, 64) = 64; exp f32: 5*3*2=30 -> 64
    let sin = a.node(f64a, x, Op::Sin, Some(l), None, None);
    assert!(a.have_vectorized_loop(sin));
    assert_eq!(a.cost(&m, sin), 64);
    // arctan2 f64: 35*3 = 105
    let at2 = a.node(f64a, x, Op::Arctan2, Some(l), Some(r), None);
    assert_eq!(a.cost(&m, at2), 105);
    // tan has no vectorized loop
    let tan = a.node(f64a, x, Op::Tan, Some(l), None, None);
    assert!(!a.have_vectorized_loop(tan));
    assert_eq!(a.cost(&m, tan), 10);
    // abs(complex) -> 50 and vectorized: 150
    let c = a.leaf(arr(TypeCode::ArrC128, 1), x);
    let ab = a.node(f64a, x, Op::Abs, Some(c), None, None);
    assert_eq!(a.cost(&m, ab), 150);
    // nested: (l + r) * l -> 1 + 1
    let mul = a.node(f64a, x, Op::Mul, Some(add), Some(l), None);
    assert_eq!(a.cost(&m, mul), 2);
    assert_eq!(a.depth(mul), 3);
    assert_eq!(a.count_nodes(mul), 5);
    assert!(a.str(&m, mul).contains("mul"));
    // int array with sin is not vectorized
    let ia = a.leaf(arr(TypeCode::ArrI64, 1), x);
    let isin = a.node(f64a, x, Op::Sin, Some(ia), None, None);
    assert!(!a.have_vectorized_loop(isin));
}

#[test]
fn replace_and_owned_result() {
    let mut m = Module::new("t");
    let x = m.get_int(0);
    let mut a = ExprArena::new();
    let f = arr(TypeCode::ArrF64, 2);
    let l = a.leaf(f, x);
    let r = a.leaf(f, x);
    let add = a.node(f, x, Op::Add, Some(l), Some(r), None);
    assert!(has_owned_result(&a, add));
    assert!(!has_owned_result(&a, l));
    let tr = a.node(f, x, Op::Transpose, Some(l), None, None);
    assert!(!has_owned_result(&a, tr));
    let dst = a.leaf(f, x);
    a.replace(dst, add);
    assert_eq!(a[dst].op, Op::Add);
    assert_eq!(a[add].op, Op::None);
    assert!(a[add].val.is_none());
}

#[test]
fn broadcast_info() {
    let mut m = Module::new("t");
    let i = m.get_int_type();
    let v1 = m.nr_var(i, false, "a");
    let v2 = m.nr_var(i, false, "b");
    let a1 = m.nr_var_value(v1);
    let a1b = m.nr_var_value(v1);
    let b1 = m.nr_var_value(v2);
    let mut a = ExprArena::new();
    let f1 = arr(TypeCode::ArrF64, 1);
    let f2 = arr(TypeCode::ArrF64, 2);
    // same var twice -> NO
    let l = a.leaf(f1, a1);
    let r = a.leaf(f1, a1b);
    let e = a.node(f1, a1, Op::Add, Some(l), Some(r), None);
    assert_eq!(a.broadcast_info(&m, e), BroadcastInfo::No);
    // two vars -> MAYBE
    let r2 = a.leaf(f1, b1);
    let e2 = a.node(f1, a1, Op::Add, Some(l), Some(r2), None);
    assert_eq!(a.broadcast_info(&m, e2), BroadcastInfo::Maybe);
    // different ndim -> YES
    let r3 = a.leaf(f2, b1);
    let e3 = a.node(f2, a1, Op::Add, Some(l), Some(r3), None);
    assert_eq!(a.broadcast_info(&m, e3), BroadcastInfo::Yes);
}

fn fwd(dst: usize, src: usize) -> Forwarding {
    let mut m = Module::new("t");
    let i = m.get_int_type();
    let var = m.nr_var(i, false, "x");
    Forwarding { dst, src, var, dst_leaf: 0, dst_id: dst as i64, src_id: src as i64 }
}

#[test]
fn forwarding_dag_components_and_cycles() {
    // 0 <- 1 <- 2, 3 alone
    let mut dag: IndexMap<usize, Vec<Forwarding>> = IndexMap::new();
    dag.insert(0, vec![fwd(0, 1)]);
    dag.insert(1, vec![fwd(1, 2)]);
    dag.insert(2, vec![]);
    dag.insert(3, vec![]);
    let comps = connected_components(&dag, 4);
    assert_eq!(comps.len(), 2);
    let big = comps.iter().find(|c| c.len() == 3).unwrap();
    assert_eq!(get_forwarding_root(big), 0);
    assert!(!has_cycle(&dag, 4));
    // add a cycle 2 -> 0
    dag.insert(2, vec![fwd(2, 0)]);
    assert!(has_cycle(&dag, 4));
}

#[test]
fn bounds_terms() {
    let mut m = Module::new("t");
    let i = m.get_int_type();
    let n = m.nr_var(i, false, "n");
    let a = m.nr_var(i, false, "a");
    let nv = m.nr_var_value(n);
    let av = m.nr_var_value(a);
    // i in [0, len(a)) : 0 <= 0, (len(a) - 1) + 1 <= len(a)
    assert!(less_than_or_equal(&m, &[Term::val_term(0)], &[Term::val_term(0)]));
    assert!(!less_than(&m, &[Term::val_term(0)], &[Term::val_term(0)]));
    assert!(less_than_or_equal(&m, &[Term::len_term(av)], &[Term::len_term(av)]));
    assert!(less_than(&m, &[Term::len_term(av), Term::val_term(-1)], &[Term::len_term(av)]));
    // n vs len(a): unknown
    assert!(!less_than_or_equal(&m, &[Term::var_term(nv)], &[Term::len_term(av)]));
    // n - n cancels
    let mut neg = Term::var_term(nv);
    neg.negate();
    let mut ts = vec![Term::var_term(nv), neg, Term::val_term(3), Term::val_term(-1)];
    simplify(&m, &mut ts);
    assert_eq!(ts.len(), 1);
    assert_eq!(ts[0].val, 2);
    assert!(check_total(&ts, true));
}

#[test]
fn analysis_and_passes_run_on_empty_module() {
    let mut m = Module::new("t");
    let mut an = NumPyExpressionAnalysis::new("core-analyses-rd", "core-analyses-side-effect");
    let r = an.run(&m, &NullPassContext);
    assert!(r.as_any().downcast_ref::<NumPyExpressionResult>().is_some());
    // passes are no-ops without their analyses
    NumPyFusionPass::new("x", "y").run(&mut m, &NullPassContext);
    NumPyInlinePass::new("x").run(&mut m, &NullPassContext);
    NumPyLifetimePass::new("x").run(&mut m, &NullPassContext);
    NumPyBoundsCheckElisionPass::new("x").run(&mut m, &NullPassContext);
}

#[test]
fn options_defaults() {
    let o = numpy_options();
    assert_eq!(o.always_fuse_cost_threshold, 10);
    assert_eq!(o.never_fuse_cost_threshold, 50);
    assert!(o.fuse_matmul);
    assert_eq!(o.replacement_free_threshold, 512);
}
