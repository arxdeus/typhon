//! Tests for `typhon_ir_transform::pythonic`.
//! PORT-NOTE: test/cir/transform/{enumerate,format}.cpp compile full Codon programs;
//! without a TypeRealizer here we port the format-string parser checks and the
//! structural (realizer-free) behaviour; the end-to-end versions belong to typhon-testsuite.

use typhon_ir::pass::{NullPassContext, Pass};
use typhon_ir::{FuncId, Module, TypeId, ValueId};
use typhon_ir_transform::pythonic::format::{FieldAccess, parse_directive, parse_format};
use typhon_ir_transform::pythonic::*;

fn func(m: &mut Module, name: &str, ret: TypeId, args: Vec<TypeId>) -> FuncId {
    let f = m.nr_bodied_func(name);
    let ft = m.unsafe_get_func_type(&format!("{name}_t"), ret, args.clone(), false);
    let names: Vec<String> = (0..args.len()).map(|i| format!("a{i}")).collect();
    m.realize_func(f, ft, &names);
    m.func_mut(f).unwrap().unmangled_name = name.to_string();
    f
}

fn main_with(m: &mut Module, body: ValueId) {
    let main = m.main_func();
    let i = m.get_int_type();
    let ft = m.unsafe_get_func_type("main_t", i, vec![], false);
    m.realize_func(main, ft, &[]);
    m.set_func_body(main, Some(body));
}

#[test]
fn keys() {
    assert_eq!(DictArithmeticOptimization::KEY, "core-pythonic-dict-arithmetic-opt");
    assert_eq!(GeneratorArgumentOptimization::KEY, "core-pythonic-generator-argument-opt");
    assert_eq!(GeneratorLoopFusion::KEY, "core-pythonic-generator-loop-fusion");
    assert_eq!(EnumerateOptimization::KEY, "core-pythonic-enumerate-opt");
    assert_eq!(IOCatOptimization::KEY, "core-pythonic-io-cat-opt");
    assert_eq!(FormattingOptimization::KEY, "core-pythonic-formatting-opt");
    assert_eq!(ListAdditionOptimization::KEY, "core-pythonic-list-addition-opt");
    assert_eq!(StrAdditionOptimization::KEY, "core-pythonic-str-addition-opt");
}

#[test]
fn format_parser() {
    let p = parse_format("abc{:04d}xyz").unwrap();
    assert_eq!(p.len(), 3);
    assert_eq!(p[0].literal, "abc");
    assert!(p[1].is_field());
    assert_eq!(p[1].index, 0);
    assert_eq!(p[1].directive.width, 4);
    assert_eq!(p[1].directive.flags, 2);
    assert_eq!(p[1].directive.conversion, 'd' as i64);
    assert_eq!(p[2].literal, "xyz");

    let p = parse_format("{0.value}:{1[0]}:{2[key]}").unwrap();
    assert_eq!(p[0].accesses, vec![FieldAccess::Member("value".into())]);
    assert_eq!(p[2].accesses, vec![FieldAccess::IntegerElement(0)]);
    assert_eq!(p[4].accesses, vec![FieldAccess::StringElement("key".into())]);

    assert!(parse_format("{").is_none());
    assert!(parse_format("}").is_none());
    assert!(parse_format("{}{0}").is_none()); // mixed numbering
    assert_eq!(parse_format("{{}}").unwrap()[0].literal, "{}");
    assert!(parse_format("{!x}").is_none());
    assert_eq!(parse_format("{!r}").unwrap()[0].field_conversion, 'r' as i64);

    let d = parse_directive("*^+z#010,.3f").unwrap();
    assert_eq!(d.fill, '*' as i64);
    assert_eq!(d.alignment, '^' as i64);
    assert!(d.coerce_negative_zero);
    assert_eq!(d.flags, 16 | 1 | 2);
    assert_eq!(d.width, 10);
    assert_eq!(d.grouping, ',' as i64);
    assert_eq!(d.precision, 3);
    let d = parse_directive("€<5").unwrap();
    assert_eq!(d.fill, '€' as i64);
    assert!(parse_directive("n").unwrap().locale);
    assert!(parse_directive("q").is_none());
    assert!(parse_directive("dd").is_none());
    assert!(parse_directive("99999999999999999999").is_none());
}

#[test]
fn generator_loop_fusion_splices_producer() {
    let mut m = Module::new("");
    let i = m.get_int_type();
    let gt = m.get_generator_type(i);
    // def g(a0): yield a0
    let g = func(&mut m, "g", gt, vec![i]);
    m.func_mut(g).unwrap().generator = true;
    let a0 = m.func_args(g)[0];
    let gb = m.nr_series_flow();
    let av = m.nr_var_value(a0);
    let y = m.nr_yield(Some(av), false);
    m.series_push(gb, y);
    m.set_func_body(g, Some(gb));
    // def h(a0) -> int
    let h = func(&mut m, "h", i, vec![i]);
    // main: for x in g(5): h(x)
    let x = m.nr_var(i, false, "x");
    let main = m.main_func();
    let gv = m.nr_var_value(g);
    let five = m.get_int(5);
    let call = m.nr_call(gv, vec![five]);
    let lb = m.nr_series_flow();
    let hv = m.nr_var_value(h);
    let xv = m.nr_var_value(x);
    let hc = m.nr_call(hv, vec![xv]);
    m.series_push(lb, hc);
    let lp = m.nr_for_flow(call, lb, x);
    let body = m.nr_series_flow();
    m.series_push(body, lp);
    main_with(&mut m, body);
    m.func_push_symbol(main, x);

    GeneratorLoopFusion::new().run(&mut m, &NullPassContext);
    let setup = m.as_series_flow(lp).expect("loop replaced by setup series").series.clone();
    // a0 snapshot + exit wrapper loop
    assert_eq!(setup.len(), 2);
    assert!(m.as_assign_instr(setup[0]).is_some());
    let w = m.as_while_flow(setup[1]).expect("one-shot exit loop");
    let wrapper = m.as_series_flow(w.body).unwrap().series.clone();
    assert!(m.as_break_instr(*wrapper.last().unwrap()).is_some());
    let s = typhon_ir::util::format::format(&m, typhon_ir::NodeRef::Value(lp));
    assert!(!s.contains("yield"), "{s}");
    assert!(s.contains("call"), "{s}");
}

#[test]
fn generator_loop_fusion_rejects_continue() {
    let mut m = Module::new("");
    let i = m.get_int_type();
    let gt = m.get_generator_type(i);
    let g = func(&mut m, "g", gt, vec![]);
    m.func_mut(g).unwrap().generator = true;
    let gb = m.nr_series_flow();
    let one = m.get_int(1);
    let y = m.nr_yield(Some(one), false);
    m.series_push(gb, y);
    m.set_func_body(g, Some(gb));
    let x = m.nr_var(i, false, "x");
    let gv = m.nr_var_value(g);
    let call = m.nr_call(gv, vec![]);
    let lb = m.nr_series_flow();
    let c = m.nr_continue(None);
    m.series_push(lb, c);
    let lp = m.nr_for_flow(call, lb, x);
    let body = m.nr_series_flow();
    m.series_push(body, lp);
    main_with(&mut m, body);
    GeneratorLoopFusion::new().run(&mut m, &NullPassContext);
    assert!(m.as_for_flow(lp).is_some());
}

#[test]
fn enumerate_declines_non_builtin() {
    let mut m = Module::new("");
    let i = m.get_int_type();
    let gt = m.get_generator_type(i);
    let e = func(&mut m, "enumerate", gt, vec![i, i]);
    let x = m.nr_var(i, false, "x");
    let ev = m.nr_var_value(e);
    let a = m.get_int(1);
    let b = m.get_int(0);
    let call = m.nr_call(ev, vec![a, b]);
    let lb = m.nr_series_flow();
    let lp = m.nr_for_flow(call, lb, x);
    let body = m.nr_series_flow();
    m.series_push(body, lp);
    main_with(&mut m, body);
    EnumerateOptimization::new().run(&mut m, &NullPassContext);
    assert_eq!(m.as_for_flow(lp).unwrap().iter, call);
}

#[test]
fn reducer_and_addition_passes_are_noops_without_matches() {
    let mut m = Module::new("");
    let i = m.get_int_type();
    let add = func(&mut m, "__add__", i, vec![i, i]);
    let av = m.nr_var_value(add);
    let a = m.get_int(1);
    let b = m.get_int(2);
    let c = m.nr_call(av, vec![a, b]);
    let body = m.nr_series_flow();
    m.series_push(body, c);
    main_with(&mut m, body);
    let ctx = NullPassContext;
    StrAdditionOptimization::new().run(&mut m, &ctx);
    ListAdditionOptimization::new().run(&mut m, &ctx);
    DictArithmeticOptimization::new().run(&mut m, &ctx);
    IOCatOptimization::new().run(&mut m, &ctx);
    FormattingOptimization::new().run(&mut m, &ctx);
    GeneratorArgumentOptimization::new().run(&mut m, &ctx);
    assert_eq!(m.as_series_flow(body).unwrap().series, vec![c]);
    assert!(m.as_call_instr(c).is_some());
}
