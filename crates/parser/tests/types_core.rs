
use typhon_parser::ast::{AstArena, FunctionStmt, Param, ParamStatus};
use typhon_parser::cache::FunctionRealization;
use typhon_parser::types::{FuncAstInfo, Generic, LinkKind, LiteralKind, NoHost, TypeArena, Unification};

#[test]
fn new_function_realization_is_incomplete() {
    let r = FunctionRealization::default();
    assert!(r.ty.is_none());
    assert!(r.ast.is_none());
    assert!(r.ir.is_none());
    assert!(r.captures.is_empty());
}

#[test]
fn replacing_function_arguments() {
    let types = TypeArena::new();
    let mut ast = AstArena::new();
    let params = vec![
        Param::with_status("x", None, None, ParamStatus::Value),
        Param::with_status("y", None, None, ParamStatus::Value),
    ];
    let fs = ast.new_stmt(FunctionStmt { name: "replace_args".into(), items: params, ..Default::default() }, Default::default());
    types.register_func_ast(
        fs,
        FuncAstInfo {
            name: "replace_args".into(),
            params: vec![("x".into(), ParamStatus::Value), ("y".into(), ParamStatus::Value)],
            allow_pass_through: false,
        },
    );
    let before = types.new_class("Before", vec![], vec![]);
    let after = types.new_class("After", vec![], vec![]);
    let args = types.new_class(
        "Tuple",
        vec![Generic::new("T0", Some(before), 0, LiteralKind::Runtime), Generic::new("T1", Some(before), 1, LiteralKind::Runtime)],
        vec![],
    );
    let base = types.new_class(
        "Function",
        vec![Generic::new("T", Some(args), 2, LiteralKind::Runtime), Generic::new("TR", Some(before), 3, LiteralKind::Runtime)],
        vec![],
    );
    let function = types.new_func(base, fs, vec![], None);
    let sibling = types.new_func(base, fs, vec![], None);
    // Prime both names, then specialize without mutating a shared tuple.
    let old_args_name = types.realized_name(args);
    let old_function_name = types.class_realized_name(function);
    let old_signature = types.realized_name(function);
    types.replace_arg_types(function, &[Some(after), None]);
    assert_eq!(types.func_arg(function, 0), after);
    assert_eq!(types.func_arg(function, 1), before);
    assert_ne!(types.class_realized_name(function), old_function_name);
    assert_ne!(types.realized_name(function), old_signature);
    assert_eq!(types.realized_name(args), old_args_name);
    assert_eq!(types.func_arg(sibling, 0), before);
    assert_eq!(types.class_realized_name(sibling), old_function_name);
    types.replace_arg_types(function, &[Some(before), Some(after)]);
    assert_eq!(types.func_arg(function, 0), before);
    assert_eq!(types.func_arg(function, 1), after);
}

#[test]
fn checks_shared_type_graphs_and_restores_levels() {
    let types = TypeArena::new();
    let mut host = NoHost;
    let leaf = types.new_unbound(types.next_unbound_id(), 5, Default::default());
    let variable = types.new_unbound(types.next_unbound_id(), 1, Default::default());
    let mut shared = leaf;
    // Occurs checks must visit shared nodes once while still lowering levels.
    for _ in 0..24 {
        shared = types.new_class(
            "Shared",
            vec![Generic::new("left", Some(shared), 0, LiteralKind::Runtime), Generic::new("right", Some(shared), 1, LiteralKind::Runtime)],
            vec![],
        );
    }
    let mut changes = Unification::new();
    assert!(types.unify(variable, shared, Some(&mut changes), &mut host).unwrap() >= 0);
    assert_eq!(types.link(leaf).unwrap().level, 1);
    assert_eq!(changes.leveled.len(), 1);
    changes.undo(&types);
    assert_eq!(types.link(leaf).unwrap().level, 5);
    assert_eq!(types.link(variable).unwrap().kind, LinkKind::Unbound);

    let recursive = types.new_class(
        "Recursive",
        vec![Generic::new("shared", Some(shared), 0, LiteralKind::Runtime), Generic::new("cycle", Some(variable), 1, LiteralKind::Runtime)],
        vec![],
    );
    let mut rejected = Unification::new();
    assert!(types.unify(variable, recursive, Some(&mut rejected), &mut host).unwrap() < 0);
    rejected.undo(&types);
    assert_eq!(types.link(variable).unwrap().kind, LinkKind::Unbound);
    assert_eq!(types.link(leaf).unwrap().level, 5);
    assert!(types.unify(variable, shared, None, &mut host).unwrap() >= 0);
    assert_eq!(types.link(leaf).unwrap().level, 5);
}

#[test]
fn unify_generalize_instantiate_roundtrip() {
    let types = TypeArena::new();
    let mut host = NoHost;
    let u = types.new_unbound(types.next_unbound_id(), 1, Default::default());
    let list = types.new_class("List", vec![Generic::new("T", Some(u), 7, LiteralKind::Runtime)], vec![]);
    let g = types.generalize(list, 1);
    assert!(types.has_unbounds(g, true) && !types.has_unbounds(g, false));
    let i1 = types.instantiate(g, 2, None);
    let i2 = types.instantiate(g, 2, None);
    let int = types.new_class("int", vec![], vec![]);
    let s = types.new_class("str", vec![], vec![]);
    assert!(types.unify_or_undo(types.generic_type(i1, 0).unwrap(), int, &mut host).unwrap());
    assert!(types.unify_or_undo(types.generic_type(i2, 0).unwrap(), s, &mut host).unwrap());
    assert!(types.can_realize(i1));
    assert_eq!(types.realized_name(i1), "List[int]");
    assert_eq!(types.realized_name(i2), "List[str]");
    assert!(!types.unify_or_undo(i1, i2, &mut host).unwrap());
    let st = types.new_int_static(5);
    assert_eq!(types.debug_string(st, 2), "Literal[5]");
    assert_eq!(types.debug_string(st, 1), "5");
}
