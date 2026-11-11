//! Integration tests for the CIR -> LLVM IR visitor (port of llvisitor.cpp behaviour).
//! Small CIR modules are built with typhon-ir builders, lowered, verified and (where
//! possible) JIT-executed through `typhon_llvm::Engine` with the Rust runtime linked in.

use inkwell::context::Context;
use typhon_common::Options;
use typhon_ir::ids::{FuncId, TypeId, ValueId};
use typhon_ir::module::Module;
use typhon_ir::util::irtools;
use typhon_ir::var::FuncKind;
use typhon_llvm::{Engine, EngineConfig, LlvmVisitor, optimize};

fn opts() -> Options {
    let mut o = Options::default();
    o.debug = false;
    o.native = false;
    o.standalone = true;
    o
}

/// Exported bodied function `name(args...) -> ret`.
fn func(m: &mut Module, name: &str, ret: TypeId, args: &[TypeId], argnames: &[&str]) -> FuncId {
    let f = m.nr_bodied_func(name);
    let ft = m.unsafe_get_func_type(&format!("fn.{name}"), ret, args.to_vec(), false);
    let names: Vec<String> = argnames.iter().map(|s| s.to_string()).collect();
    m.realize_func(f, ft, &names);
    m.func_mut(f).unwrap().unmangled_name = name.to_string();
    irtools::set_export(m, f, true);
    f
}

/// `@llvm def name(a: int, b: int) -> int: <op> i64 %a, %b`.
fn llvm_binop(m: &mut Module, name: &str, op: &str, ret: TypeId) -> FuncId {
    let i = m.get_int_type();
    let f = m.nr_llvm_func(name);
    let ft = m.unsafe_get_func_type(&format!("fn.{name}"), ret, vec![i, i], false);
    m.realize_func(f, ft, &["a".to_string(), "b".to_string()]);
    let rt = if op.starts_with("icmp") { "%c = {} i64 %a, %b\n  %r = zext i1 %c to i8\n  ret i8 %r" } else { "%r = {} i64 %a, %b\n  ret i64 %r" };
    let body = format!("entry:\n  {}", rt.replacen("{}", op, 1));
    // `{}` in the body would be format placeholders; literals are escaped by the parser.
    let fd = m.func_mut(f).unwrap();
    fd.set_llvm_body(body.replace('{', "{{").replace('}', "}}"));
    f
}

fn call(m: &mut Module, f: FuncId, args: Vec<ValueId>) -> ValueId {
    let c = m.nr_var_value(f);
    m.nr_call(c, args)
}

fn series(m: &mut Module, items: &[ValueId]) -> ValueId {
    let s = m.nr_series_flow();
    for &i in items {
        m.series_push(s, i);
    }
    s
}

/// Lower `m`, verify, optimize and JIT; returns the address of `sym`.
fn jit<'c>(ctx: &'c Context, m: &Module) -> (Engine, inkwell::module::Module<'c>) {
    let mut v = LlvmVisitor::new(ctx, opts());
    v.visit_module(m).expect("codegen");
    let module = v.take_module(None, None).expect("take module");
    if let Err(e) = module.verify() {
        panic!("invalid module: {}\n{}", e.to_string(), module.print_to_string().to_string());
    }
    optimize(&module, &opts(), &[]).expect("optimize");
    let engine = Engine::new(&opts(), EngineConfig::default()).expect("engine");
    use codonrt as rt;
    let syms: Vec<(&str, u64)> = vec![
        ("seq_personality", rt::exc::seq_personality as *const () as usize as u64),
        ("seq_alloc_exc", rt::exc::seq_alloc_exc as *const () as usize as u64),
        ("seq_throw", rt::exc::seq_throw as *const () as usize as u64),
        ("seq_terminate", rt::exc::seq_terminate as *const () as usize as u64),
        ("seq_alloc", rt::lib_rt::seq_alloc as *const () as usize as u64),
        ("seq_alloc_atomic", rt::lib_rt::seq_alloc_atomic as *const () as usize as u64),
        ("seq_free", rt::lib_rt::seq_free as *const () as usize as u64),
        ("seq_init", rt::lib_rt::seq_init as *const () as usize as u64),
    ];
    engine.define_absolute(&syms).expect("define runtime symbols");
    engine.add_module(&module).expect("add module");
    (engine, module)
}

#[test]
fn codegen_and_jit_small_programs() {
    codonrt::lib_rt::seq_init(0); // GC + JIT unwind-section registration
    let mut m = Module::new("test");
    let i = m.get_int_type();
    let b = m.get_bool_type();
    let add = llvm_binop(&mut m, "add", "add", i);
    let lt = llvm_binop(&mut m, "lt", "icmp slt", b);

    // fn arith(a, b) = (a + b) + 1
    let f1 = func(&mut m, "arith", i, &[i, i], &["a", "b"]);
    let [a, bb] = m.func_args(f1)[..] else { unreachable!() };
    let (va, vb) = (m.nr_var_value(a), m.nr_var_value(bb));
    let s = call(&mut m, add, vec![va, vb]);
    let one = m.nr_int_const(1, i);
    let s2 = call(&mut m, add, vec![s, one]);
    let r = m.nr_return(Some(s2));
    let body = series(&mut m, &[r]);
    m.set_func_body(f1, Some(body));

    // fn loops(n): total = 0; i = 0; while i < n: total += i; i += 1
    //             for j in range(0, n, 1) (imperative): total += 1 ; if total < 0: total = 0 ; return total
    let f2 = func(&mut m, "loops", i, &[i], &["n"]);
    let n = m.func_args(f2)[0];
    let total = m.nr_var(i, false, "total");
    let k = m.nr_var(i, false, "k");
    let j = m.nr_var(i, false, "j");
    for v in [total, k, j] {
        m.func_push_symbol(f2, v);
    }
    let z = m.nr_int_const(0, i);
    let a0 = m.nr_assign(total, z);
    let z = m.nr_int_const(0, i);
    let a1 = m.nr_assign(k, z);
    let (vk, vn) = (m.nr_var_value(k), m.nr_var_value(n));
    let cond = call(&mut m, lt, vec![vk, vn]);
    let (vt, vk) = (m.nr_var_value(total), m.nr_var_value(k));
    let sum = call(&mut m, add, vec![vt, vk]);
    let as1 = m.nr_assign(total, sum);
    let (vk, one) = (m.nr_var_value(k), m.nr_int_const(1, i));
    let inc = call(&mut m, add, vec![vk, one]);
    let as2 = m.nr_assign(k, inc);
    let wbody = series(&mut m, &[as1, as2]);
    let w = m.nr_while_flow(cond, wbody);
    let (vt, one) = (m.nr_var_value(total), m.nr_int_const(1, i));
    let inc = call(&mut m, add, vec![vt, one]);
    let as3 = m.nr_assign(total, inc);
    let fbody = series(&mut m, &[as3]);
    let (start, end) = (m.nr_int_const(0, i), m.nr_var_value(n));
    let ifor = m.nr_imperative_for_flow(start, 1, end, fbody, j);
    let (vt, zero) = (m.nr_var_value(total), m.nr_int_const(0, i));
    let neg = call(&mut m, lt, vec![vt, zero]);
    let zero = m.nr_int_const(0, i);
    let reset = m.nr_assign(total, zero);
    let tb = series(&mut m, &[reset]);
    let iff = m.nr_if_flow(neg, tb, None);
    let vt = m.nr_var_value(total);
    let r = m.nr_return(Some(vt));
    let body = series(&mut m, &[a0, a1, w, ifor, iff, r]);
    m.set_func_body(f2, Some(body));

    // record: fn rec(a, b) -> a of (b, a) via extract on a ternary-built tuple
    let tup = m.nr_record_type("Pair", vec![i, i], vec!["x".into(), "y".into()]);
    let mk_sig = m.unsafe_get_func_type("fn.Pair.new", tup, vec![i, i], false);
    let mk = m.get_intrinsic(typhon_ir::var::Intrinsic::RecordNew, Some(tup), Some(mk_sig));
    let f3 = func(&mut m, "rec", i, &[i, i], &["a", "b"]);
    let [a, bb] = m.func_args(f3)[..] else { unreachable!() };
    let (va, vb) = (m.nr_var_value(a), m.nr_var_value(bb));
    let pair = call(&mut m, mk, vec![vb, va]);
    let ex = m.nr_extract(pair, "y");
    let (va, vb) = (m.nr_var_value(a), m.nr_var_value(bb));
    let c = call(&mut m, lt, vec![va, vb]);
    let big = m.nr_int_const(100, i);
    let tern = m.nr_ternary(c, ex, big);
    let r = m.nr_return(Some(tern));
    let body = series(&mut m, &[r]);
    m.set_func_body(f3, Some(body));

    // try/catch: fn exc() -> int: x = 1; try: raise <null obj> except: x = 42; return x
    let p = m.get_pointer_type(Some(i));
    let g = m.nr_var(p, true, "exc_obj");
    let f4 = func(&mut m, "exc", i, &[], &[]);
    let x = m.nr_var(i, false, "x");
    m.func_push_symbol(f4, x);
    let one = m.nr_int_const(1, i);
    let ax = m.nr_assign(x, one);
    let vg = m.nr_var_value(g);
    let th = m.nr_throw(Some(vg));
    let tbody = series(&mut m, &[th]);
    let fin = series(&mut m, &[]);
    let tc = m.nr_try_catch_flow(tbody, Some(fin), None);
    let c42 = m.nr_int_const(42, i);
    let ah = m.nr_assign(x, c42);
    let hbody = series(&mut m, &[ah]);
    m.try_catch_push(tc, hbody, None, None);
    let vx = m.nr_var_value(x);
    let r = m.nr_return(Some(vx));
    let body = series(&mut m, &[ax, tc, r]);
    m.set_func_body(f4, Some(body));

    // generator: gen() yields 1, 2, 3; fn gsum() = sum(gen())
    let gt = m.get_generator_type(i);
    let gen_f = m.nr_bodied_func("gen");
    let gft = m.unsafe_get_func_type("fn.gen", gt, vec![], false);
    m.realize_func(gen_f, gft, &[]);
    m.func_mut(gen_f).unwrap().generator = true;
    let mut ys = Vec::new();
    for k in 1..=3 {
        let c = m.nr_int_const(k, i);
        ys.push(m.nr_yield(Some(c), false));
    }
    let body = series(&mut m, &ys);
    m.set_func_body(gen_f, Some(body));
    let f5 = func(&mut m, "gsum", i, &[], &[]);
    let acc = m.nr_var(i, false, "acc");
    let it = m.nr_var(i, false, "it");
    m.func_push_symbol(f5, acc);
    m.func_push_symbol(f5, it);
    let z = m.nr_int_const(0, i);
    let a0 = m.nr_assign(acc, z);
    let gen_call = call(&mut m, gen_f, vec![]);
    let (va, vi) = (m.nr_var_value(acc), m.nr_var_value(it));
    let s = call(&mut m, add, vec![va, vi]);
    let as1 = m.nr_assign(acc, s);
    let lb = series(&mut m, &[as1]);
    let fl = m.nr_for_flow(gen_call, lb, it);
    let va = m.nr_var_value(acc);
    let r = m.nr_return(Some(va));
    let body = series(&mut m, &[a0, fl, r]);
    m.set_func_body(f5, Some(body));

    // a string constant somewhere in main to exercise StringConst lowering
    let st = m.get_string_type();
    let sv = m.nr_var(st, true, "greeting");
    let sc = m.nr_string_const("héllo ✓", st);
    let asg = m.nr_assign(sv, sc);
    let mainb = series(&mut m, &[asg]);
    let main = m.main_func();
    let none = m.get_none_type();
    let mt = m.unsafe_get_func_type("fn.main", none, vec![], false);
    m.realize_func(main, mt, &[]);
    m.set_func_body(main, Some(mainb));
    assert!(matches!(m.func(main).unwrap().kind, FuncKind::Bodied { .. }));

    let ctx = Context::create();
    let (engine, _module) = jit(&ctx, &m);
    unsafe {
        let arith: extern "C" fn(i64, i64) -> i64 = std::mem::transmute(engine.lookup("arith").unwrap() as usize);
        assert_eq!(arith(2, 3), 6);
        let loops: extern "C" fn(i64) -> i64 = std::mem::transmute(engine.lookup("loops").unwrap() as usize);
        assert_eq!(loops(5), 10 + 5);
        let rec: extern "C" fn(i64, i64) -> i64 = std::mem::transmute(engine.lookup("rec").unwrap() as usize);
        assert_eq!(rec(1, 2), 1);
        assert_eq!(rec(3, 2), 100);
        let gsum: extern "C" fn() -> i64 = std::mem::transmute(engine.lookup("gsum").unwrap() as usize);
        assert_eq!(gsum(), 6);
        // The try/catch function is always lowered and verified above. Executing it needs
        // the engine to register JIT EH frames with the system unwinder; RTDyld on
        // macOS/arm64 does not yet (see llvm-infra lane), so execution is opt-in.
        let exc_addr = engine.lookup("exc").unwrap();
        assert_ne!(exc_addr, 0);
        if std::env::var_os("TYPHON_TEST_JIT_EXCEPTIONS").is_some() {
            let exc: extern "C" fn() -> i64 = std::mem::transmute(exc_addr as usize);
            assert_eq!(exc(), 42);
        }
    }
}
