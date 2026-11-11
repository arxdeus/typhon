//! Translate lane tests: hand-built typechecked AST -> IR (checked via `util::format`).

use std::cell::RefCell;
use std::rc::Rc;

use typhon_ir::util::format::format;
use typhon_ir::{Module, NodeRef};
use typhon_parser::ast::*;
use typhon_parser::cache::{Cache, Class, ClassRealization};
use typhon_parser::visitors::translate::{self, TranslateItem, TranslateVisitor};

struct Fx {
    cache: Cache,
    int_t: typhon_parser::TypeId,
    bool_t: typhon_parser::TypeId,
}

fn setup() -> Fx {
    let mut cache = Cache::new(None);
    let m = Rc::new(RefCell::new(Module::new("test")));
    cache.module = Some(m.clone());
    let int_t = cache.types.new_class("int", vec![], vec![]);
    let bool_t = cache.types.new_class("bool", vec![], vec![]);
    let none_t = cache.types.new_class("NoneType", vec![], vec![]);
    let ctx = translate::codegen_ctx(&mut cache);
    for (n, t) in [("int", int_t), ("bool", bool_t), ("NoneType", none_t)] {
        let ir = {
            let mut mm = m.borrow_mut();
            match n {
                "int" => mm.get_int_type(),
                "bool" => mm.get_bool_type(),
                _ => mm.nr_record_type("NoneType", vec![], vec![]),
            }
        };
        let mut c = Class::default();
        c.realizations.insert(n.into(), Rc::new(RefCell::new(ClassRealization { ty: Some(t), ir: Some(ir), ..Default::default() })));
        cache.classes.insert(n.into(), c);
        ctx.borrow_mut().add(n, TranslateItem::Type(ir));
    }
    translate::install_hooks(&mut cache);
    Fx { cache, int_t, bool_t }
}

fn int(fx: &mut Fx, v: i64) -> ExprId {
    let e = fx.cache.ast.new_expr(IntExpr::new(v), Default::default());
    fx.cache.ast[e].ty = Some(fx.int_t);
    e
}
fn id(fx: &mut Fx, n: &str, t: typhon_parser::TypeId) -> ExprId {
    let e = fx.cache.ast.new_expr(IdExpr { value: n.into() }, Default::default());
    fx.cache.ast[e].ty = Some(t);
    e
}

#[test]
fn translates_main_with_locals_and_flows() {
    let mut fx = setup();
    let (it, bt) = (fx.int_t, fx.bool_t);
    // x = 1; while True: if x: break else: x = 2
    let one = int(&mut fx, 1);
    let x = id(&mut fx, "x", it);
    let a = fx.cache.ast.new_stmt(AssignStmt { lhs: Some(x), rhs: Some(one), ..Default::default() }, Default::default());
    let cond = fx.cache.ast.new_expr(BoolExpr { value: true }, Default::default());
    fx.cache.ast[cond].ty = Some(bt);
    let br = fx.cache.ast.new_stmt(BreakStmt::default(), Default::default());
    let two = int(&mut fx, 2);
    let x2 = id(&mut fx, "x", it);
    let upd = fx.cache.ast.new_stmt(AssignStmt { lhs: Some(x2), rhs: Some(two), update: UpdateMode::Update, ..Default::default() }, Default::default());
    let xc = id(&mut fx, "x", it);
    let ifs = fx.cache.ast.new_stmt(IfStmt { cond: Some(xc), if_suite: Some(br), else_suite: Some(upd) }, Default::default());
    let w = fx.cache.ast.new_stmt(WhileStmt { cond: Some(cond), suite: Some(ifs), ..Default::default() }, Default::default());
    let pre = fx.cache.ast.new_stmt(SuiteStmt::default(), Default::default());
    let top = fx.cache.ast.new_stmt(SuiteStmt { items: vec![pre, a, w] }, Default::default());

    let main = TranslateVisitor::apply(&mut fx.cache, top).expect("translate");
    let m = fx.cache.module.clone().unwrap();
    let s = format(&m.borrow(), NodeRef::Var(main));
    assert!(s.contains("while"), "{s}");
    assert!(s.contains("if"), "{s}");
    assert!(s.contains("break"), "{s}");
    assert!(s.contains("(assign (var '\"x."), "{s}");
    let ctx = translate::existing_codegen_ctx(&fx.cache).unwrap();
    assert!(ctx.borrow().main_user_start.is_some());
}

#[test]
fn list_literal_attribute_and_parsed_hook() {
    let mut fx = setup();
    let it = fx.int_t;
    let pre = fx.cache.ast.new_stmt(SuiteStmt::default(), Default::default());
    let top = fx.cache.ast.new_stmt(SuiteStmt { items: vec![pre] }, Default::default());
    TranslateVisitor::apply(&mut fx.cache, top).unwrap();
    // `translate_parsed` attaches a "parsed" block into main.
    let one = int(&mut fx, 7);
    fx.cache.ast[one].attrs.set(Attr::ExprList);
    let y = id(&mut fx, "y", it);
    let a = fx.cache.ast.new_stmt(AssignStmt { lhs: Some(y), rhs: Some(one), ..Default::default() }, Default::default());
    let h = fx.cache.hooks.translate_parsed.clone().unwrap();
    let block = h(&mut fx.cache, a).unwrap();
    let m = fx.cache.module.clone().unwrap();
    let mm = m.borrow();
    let ser = mm.as_series_flow(block).unwrap().series.clone();
    assert_eq!(ser.len(), 1);
    let rhs = mm.as_assign_instr(ser[0]).unwrap().rhs;
    assert!(mm.get_attribute(NodeRef::Value(rhs), typhon_ir::attr_id::LIST_LITERAL).is_some());
    let body = mm.func_body(mm.main_func()).unwrap();
    assert!(mm.as_series_flow(body).unwrap().series.contains(&block));
}
