
mod common;

use common::*;
use typhon_ir::util::irtools::{self, call, get_func, get_var, is_const_val, virtual_call};
use typhon_ir::util::matching::{match_values, match_vars};
use typhon_ir::util::operator::{Operator, OperatorState};
use typhon_ir::util::side_effect::SideEffectStatus as S;
use typhon_ir::*;

fn ift(m: &mut Module) -> TypeId {
    let i = m.get_int_type();
    m.unsafe_get_func_type("<internal_func_type>", i, vec![], false)
}


#[test]
fn function_export_attributes() {
    let (mut m, _) = setup();
    let f = m.nr_bodied_func("");
    assert!(!irtools::is_export(&m, f));
    assert!(!irtools::is_gpu_kernel(&m, f));
    irtools::set_export(&mut m, f, false);
    assert!(!m.has_attribute(f.into(), attr_id::KEY_VALUE));
    irtools::set_export(&mut m, f, true);
    assert!(irtools::is_export(&m, f));
    m.get_attribute_mut(f.into(), attr_id::KEY_VALUE)
        .unwrap()
        .as_key_value_mut()
        .unwrap()
        .attributes
        .insert("custom.attribute".into(), "preserved".into());
    irtools::set_export(&mut m, f, true);
    assert!(irtools::is_export(&m, f));
    irtools::set_export(&mut m, f, false);
    irtools::set_export(&mut m, f, false);
    assert!(!irtools::is_export(&m, f));
    assert!(!irtools::is_gpu_kernel(&m, f));
    let kv = m.get_attribute(f.into(), attr_id::KEY_VALUE).unwrap().as_key_value().unwrap();
    assert_eq!(kv.get("custom.attribute"), "preserved");
}

#[test]
fn func_realization_and_var_insertion_erase_and_iterators() {
    let (mut m, _) = setup();
    let f = m.nr_bodied_func("");
    let t = ift(&mut m);
    m.realize_func(f, t, &[]);
    let i = m.get_int_type();
    let ft = m.unsafe_get_func_type("**test_type**", i, vec![i], false);
    m.realize_func(f, ft, &names(&["foo"]));
    assert!(m.is_global(f));
    let args = m.func_args(f);
    assert_eq!(1, args.len());
    assert_eq!(Some(i), m.var_type(args[0]));
    assert_eq!(Some(args[0]), m.func_arg_var(f, "foo"));
    let var = m.nr_var(i, false, "hi");
    m.func_push_symbol(f, var);
    assert_eq!(1, m.func(f).unwrap().symbols().len());
    m.func_mut(f).unwrap().symbols_mut().unwrap().remove(0);
    assert_eq!(0, m.func(f).unwrap().symbols().len());
    m.func_mut(f).unwrap().symbols_mut().unwrap().insert(0, var);
    assert_eq!(Some(i), m.var_type(m.func(f).unwrap().symbols()[0]));
}

#[test]
fn bodied_func_query_and_replace() {
    let (mut m, _) = setup();
    let f = m.nr_bodied_func("");
    let t = ift(&mut m);
    m.realize_func(f, t, &[]);
    assert_eq!(None, m.func_body(f));
    assert_eq!(0, m.used_values(f.into()).len());
    let body = m.nr_series_flow();
    m.set_func_body(f, Some(body));
    assert_eq!(vec![body], m.used_values(f.into()));
    let b2 = m.nr_series_flow();
    assert_eq!(1, m.replace_used_value(f.into(), body, b2));
    let bad = m.nr_var_value(f);
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| m.replace_used_value(f.into(), b2, bad)));
    assert!(r.is_err());
    assert_ne!(Some(body), m.func_body(f));
}

#[test]
fn bodied_func_unmangled_name() {
    let (mut m, _) = setup();
    let f = m.nr_bodied_func("Int.foo");
    m.func_mut(f).unwrap().unmangled_name = "foo".into();
    let t = ift(&mut m);
    m.realize_func(f, t, &[]);
    assert_eq!("foo", m.func(f).unwrap().unmangled_name);
}

#[test]
fn bodied_func_cloning() {
    let (mut m, mut cv) = setup();
    let f = m.nr_bodied_func("fn");
    let t = ift(&mut m);
    m.realize_func(f, t, &[]);
    let b = m.nr_series_flow();
    m.set_func_body(f, Some(b));
    let c = cv.force_clone_var(&mut m, f);
    assert!(match_vars(&m, f, c));
}

#[test]
fn external_func_unmangled_name_and_cloning() {
    let (mut m, mut cv) = setup();
    let f = m.nr_external_func("fn");
    let t = ift(&mut m);
    m.realize_func(f, t, &[]);
    m.func_mut(f).unwrap().unmangled_name = "foo".into();
    assert_eq!("foo", m.func(f).unwrap().unmangled_name);
    let c = cv.force_clone_var(&mut m, f);
    assert!(match_vars(&m, f, c));
}

#[test]
fn internal_func_parent_type_unmangled_name_and_cloning() {
    let (mut m, _) = setup();
    let f = m.nr_internal_func("fn.1");
    m.func_mut(f).unwrap().unmangled_name = "fn".into();
    let t = ift(&mut m);
    m.realize_func(f, t, &[]);
    let i = m.get_int_type();
    let b = m.get_bool_type();
    m.func_mut(f).unwrap().parent_type = Some(i);
    assert_eq!(Some(i), m.func(f).unwrap().parent_type);
    let mut cv = util::cloning::CloneVisitor::new(true);
    let c0 = cv.force_clone_var(&mut m, f);
    assert!(match_vars(&m, f, c0));
    m.set_intrinsic(f, Intrinsic::Typeinfo, Some(b));
    let mut cv = util::cloning::CloneVisitor::new(true);
    let cloned = cv.force_clone_var(&mut m, f);
    assert_ne!(cloned, f);
    assert_eq!(Intrinsic::Typeinfo, m.func(cloned).unwrap().intrinsic());
    assert_eq!(Some(b), m.func(cloned).unwrap().intrinsic_type());
    assert!(match_vars(&m, f, cloned));
    m.set_intrinsic(cloned, Intrinsic::Allocate, Some(b));
    assert!(!match_vars(&m, f, cloned));
    m.set_intrinsic(cloned, Intrinsic::Typeinfo, Some(i));
    assert!(!match_vars(&m, f, cloned));
    assert_eq!(1, m.replace_used_type(f.into(), b, i));
    assert_eq!(Some(i), m.func(f).unwrap().intrinsic_type());
}

#[test]
fn type_info_method_lookup_contract() {
    let (mut m, mut cv) = setup();
    let descriptor = m.unsafe_get_membered_type("lookup.descriptor", true);
    let i = m.get_int_type();
    let pointer = m.unsafe_get_pointer_type(Some(i));
    let lookup = m.nr_internal_func("lookup.method");
    let sig = m.unsafe_get_func_type("lookup.signature", pointer, vec![descriptor, i], false);
    m.realize_func(lookup, sig, &names(&["info", "id"]));
    m.set_intrinsic(lookup, Intrinsic::TypeinfoMethodLookup, None);
    assert!(m.has_valid_intrinsic_signature(lookup));
    assert_eq!("typeinfo_method_lookup", m.func(lookup).unwrap().intrinsic().name());
    assert_eq!(S::Unknown, m.intrinsic_side_effect_status(lookup));
    let cloned = cv.force_clone_var(&mut m, lookup);
    assert!(m.has_valid_intrinsic_signature(cloned));
    assert!(match_vars(&m, lookup, cloned));
    let inv = m.unsafe_get_func_type("lookup.invalid", pointer, vec![i, i], false);
    m.realize_func(lookup, inv, &names(&["info", "id"]));
    assert!(!m.has_valid_intrinsic_signature(lookup));
}

#[test]
fn representation_cast_intrinsic_contracts() {
    let (mut m, mut cv) = setup();
    let source = m.unsafe_get_membered_type("cast.source", true);
    let target = m.unsafe_get_membered_type("cast.target", true);
    let f = m.nr_internal_func("cast.representation");
    let s = m.unsafe_get_func_type("cast.valid", target, vec![source], false);
    m.realize_func(f, s, &names(&["value"]));
    m.set_intrinsic(f, Intrinsic::RepresentationCast, Some(target));
    assert!(m.has_valid_intrinsic_signature(f));
    assert_eq!("representation_cast", m.func(f).unwrap().intrinsic().name());
    assert_eq!(S::Pure, m.intrinsic_side_effect_status(f));
    assert!(m.intrinsic_derives_from_arguments(f));
    let cloned = cv.force_clone_var(&mut m, f);
    assert!(match_vars(&m, f, cloned));
    let i = m.get_int_type();
    let s = m.unsafe_get_func_type("cast.structural", target, vec![i], false);
    m.realize_func(f, s, &names(&["value"]));
    assert!(m.has_valid_intrinsic_signature(f));
    m.set_intrinsic(f, Intrinsic::RepresentationCast, Some(source));
    assert!(!m.has_valid_intrinsic_signature(f));
    m.set_intrinsic(f, Intrinsic::RepresentationCast, Some(target));
    let s = m.unsafe_get_func_type("cast.empty", target, vec![], false);
    m.realize_func(f, s, &[]);
    assert!(!m.has_valid_intrinsic_signature(f));
    let s = m.unsafe_get_func_type("cast.extra", target, vec![source, source], false);
    m.realize_func(f, s, &names(&["first", "second"]));
    assert!(!m.has_valid_intrinsic_signature(f));
}

#[test]
fn runtime_metadata_dependencies() {
    let (mut m, _) = setup();
    let t = m.unsafe_get_membered_type("runtime.dependencies", true);
    let method = m.nr_bodied_func("runtime.method");
    let replacement = m.nr_bodied_func("runtime.replacement");
    let i = m.get_int_type();
    let b = m.get_bool_type();
    let mut info = RuntimeInfo { mro: vec![t, i], parameters: vec![b], ..Default::default() };
    info.methods.insert(1, method);
    m.set_runtime_info(t, info);
    let types = m.used_types(t.into());
    assert!(types.contains(&i));
    assert!(types.contains(&b));
    assert_eq!(vec![method], m.used_variables(t.into()));
    assert_eq!(1, m.replace_used_variable(t.into(), method, replacement));
    assert_eq!(replacement, m.runtime_info(t).unwrap().methods[&1]);
    assert_eq!(1, m.replace_used_variable(t.into(), replacement, None));
    assert!(m.used_variables(t.into()).is_empty());
}

#[test]
fn llvm_func_unmangled_name_query_and_replace() {
    let (mut m, _) = setup();
    let f = m.nr_llvm_func("fn");
    let t = ift(&mut m);
    m.realize_func(f, t, &[]);
    let i = m.get_int_type();
    let fl = m.get_float_type();
    let b = m.get_bool_type();
    {
        let fd = m.func_mut(f).unwrap();
        fd.set_llvm_body("body");
        fd.set_llvm_declarations("decl");
        fd.parent_type = Some(i);
        fd.set_llvm_literals(vec![Generic::Static(1), Generic::Type(i)]);
    }
    assert_eq!("body", m.func(f).unwrap().llvm_body());
    assert_eq!("decl", m.func(f).unwrap().llvm_declarations());
    let mut expected = vec![t, i, i];
    assert_eq!(expected, m.used_types(f.into()));
    assert_eq!(2, m.replace_used_type(f.into(), i, fl));
    assert_eq!(Some(fl), m.func(f).unwrap().parent_type);
    assert_eq!(Some(fl), m.func(f).unwrap().llvm_literals()[1].type_value());
    expected[1] = fl;
    expected[2] = fl;
    assert_eq!(expected, m.used_types(f.into()));

    let alias = m.nr_var(b, false, "");
    let forwarder = m.nr_var(i, false, "");
    m.replace_all_var(alias, forwarder);
    m.replace_all_var(forwarder, f);
    assert_eq!(expected, m.used_types(alias.into()));
    assert_eq!(2, m.replace_used_type(alias.into(), fl, b));
    assert_eq!(Some(b), m.func(f).unwrap().parent_type);
    expected[1] = b;
    expected[2] = b;
    assert_eq!(expected, m.used_types(alias.into()));
    let rt = m.unsafe_get_func_type("<replacement_func_type>", b, vec![], false);
    assert_eq!(1, m.replace_used_type(alias.into(), t, rt));
    assert_eq!(Some(rt), m.var_type(f));
    expected[0] = rt;
    assert_eq!(expected, m.used_types(alias.into()));
}


#[test]
fn assign_instr_query_replace_and_cloning() {
    let (mut m, mut cv) = setup();
    let i = m.get_int_type();
    let var = m.nr_var(i, false, "");
    let val = m.nr_int_const(1, i);
    let instr = m.nr_assign(var, val);
    assert_eq!(var, m.as_assign_instr(instr).unwrap().lhs);
    assert_eq!(vec![val], m.used_values(instr.into()));
    assert_eq!(vec![var], m.used_variables(instr.into()));
    let c = cv.clone_value(&mut m, instr);
    assert!(match_values(&m, instr, c));
    let v2 = m.nr_int_const(1, i);
    assert_eq!(1, m.replace_used_value(instr.into(), val, v2));
    let var2 = m.nr_var(i, false, "");
    assert_eq!(1, m.replace_used_variable(instr.into(), var, var2));
}

#[test]
fn extract_instr_query_replace_and_cloning() {
    let (mut m, mut cv) = setup();
    let t = m.unsafe_get_membered_type("**internal**", false);
    let i = m.get_int_type();
    m.realize_membered(t, vec![i], names(&["foo"]));
    let var = m.nr_var(t, false, "");
    let val = m.nr_var_value(var);
    let instr = m.nr_extract(val, "foo");
    assert_eq!("foo", m.as_extract_instr(instr).unwrap().field);
    assert_eq!(Some(m.value_type(instr)), m.member_type(t, "foo"));
    assert_eq!(vec![val], m.used_values(instr.into()));
    let c = cv.clone_value(&mut m, instr);
    assert!(match_values(&m, instr, c));
    let v2 = m.nr_var_value(var);
    assert_eq!(1, m.replace_used_value(instr.into(), val, v2));
}

#[test]
fn insert_instr_query_replace_and_cloning() {
    let (mut m, mut cv) = setup();
    let t = m.unsafe_get_membered_type("**internal**", true);
    let i = m.get_int_type();
    m.realize_membered(t, vec![i], names(&["foo"]));
    let var = m.nr_var(t, false, "");
    let lhs = m.nr_var_value(var);
    let rhs = m.nr_int_const(1, i);
    let instr = m.nr_insert(lhs, "foo", rhs);
    assert_eq!(t, m.value_type(instr));
    assert_eq!(2, m.used_values(instr.into()).len());
    let c = cv.clone_value(&mut m, instr);
    assert!(match_values(&m, instr, c));
    let l2 = m.nr_var_value(var);
    assert_eq!(1, m.replace_used_value(instr.into(), lhs, l2));
    let r2 = m.nr_int_const(1, i);
    assert_eq!(1, m.replace_used_value(instr.into(), rhs, r2));
}

struct Inspect {
    state: OperatorState,
    call: ValueId,
    first: ValueId,
    second: ValueId,
    replacement: ValueId,
    visited: Vec<NodeRef>,
}
impl Operator for Inspect {
    fn state(&self) -> &OperatorState {
        &self.state
    }
    fn state_mut(&mut self) -> &mut OperatorState {
        &mut self.state
    }
    fn pre_hook(&mut self, m: &mut Module, n: NodeRef) {
        self.visited.push(n);
        if n == NodeRef::Value(self.first) {
            m.replace_used_value(self.call.into(), self.second, self.replacement);
        }
    }
}

#[test]
fn call_operand_traversal_snapshot() {
    let (mut m, _) = setup();
    let i = m.get_int_type();
    let t = m.unsafe_get_func_type("snapshot", i, vec![], false);
    let fv = m.nr_var(t, false, "");
    let callee = m.nr_var_value(fv);
    let first = m.get_int(1);
    let second = m.get_int(2);
    let replacement = m.get_int(3);
    let call = m.nr_call(callee, vec![first, second, first]);
    let mut ins =
        Inspect { state: OperatorState::new(false), call, first, second, replacement, visited: Vec::new() };
    ins.process(&mut m, call.into());
    let mut expected: Vec<NodeRef> = vec![call.into(), first.into(), second.into(), callee.into()];
    assert_eq!(expected, ins.visited);
    ins.reset();
    ins.visited.clear();
    ins.process(&mut m, call.into());
    expected[2] = replacement.into();
    assert_eq!(expected, ins.visited);
}

#[test]
fn call_instr_query_replace_and_cloning() {
    let (mut m, mut cv) = setup();
    let t = ift(&mut m);
    let f = m.nr_bodied_func("");
    m.realize_func(f, t, &[]);
    let fv = m.nr_var_value(f);
    let instr = m.nr_call(fv, vec![]);
    assert_eq!(fv, m.as_call_instr(instr).unwrap().callee);
    assert_eq!(m.func_type_ret(t), Some(m.value_type(instr)));
    assert_eq!(1, m.used_values(instr.into()).len());
    let c = cv.clone_value(&mut m, instr);
    assert!(match_values(&m, instr, c));
    let fv2 = m.nr_var_value(f);
    assert_eq!(1, m.replace_used_value(instr.into(), fv, fv2));
}

#[test]
fn virtual_call_construction_and_cloning() {
    let (mut m, mut cv) = setup();
    let thunk_id = 7usize;
    let t = m.unsafe_get_membered_type("virtual.receiver", true);
    let i = m.get_int_type();
    m.realize_membered(t, vec![i], names(&["field"]));
    if let TypeKind::Ref { polymorphic, .. } = &mut m.ty_mut(t).kind {
        *polymorphic = true;
    }
    let rv = m.nr_var(t, false, "");
    let receiver = m.nr_var_value(rv);
    let argument = m.get_int(42);
    let method = m.nr_bodied_func("");
    let mt = m.unsafe_get_func_type("virtual.method", i, vec![t, i], false);
    m.realize_func(method, mt, &names(&["self", "value"]));
    let s = m.nr_series_flow();
    let mv = m.nr_var_value(method);
    let callee = m.nr_flow_instr(s, mv);
    let parent = m.nr_bodied_func("");
    let call_v = virtual_call(&mut m, callee, vec![receiver, argument], thunk_id, parent);
    m.set_name(call_v.into(), "virtual.call");
    m.set_src_info(call_v.into(), typhon_common::SrcInfo::new("virtual.py", 12, 3, 1));
    assert_eq!(i, m.value_type(call_v));
    let ci = m.as_call_instr(call_v).unwrap().clone();
    assert_eq!(argument, *ci.args.last().unwrap());
    let target = m.as_flow_instr(ci.callee).unwrap().clone();
    let prefix = m.as_series_flow(target.flow).unwrap().series.clone();
    assert_eq!(2, prefix.len());
    assert_eq!(callee, prefix[0]);
    let assignment = m.as_assign_instr(prefix[1]).unwrap().clone();
    assert_eq!(receiver, assignment.rhs);
    let temporary = assignment.lhs;
    assert_eq!(Some(temporary), get_var(&m, ci.args[0]));
    assert!(m.func(parent).unwrap().symbols().contains(&temporary));
    let lookup = m.as_call_instr(target.val).unwrap().clone();
    let intrinsic = get_func(&m, lookup.callee).unwrap();
    assert_eq!(Intrinsic::VirtualLookup, m.func(intrinsic).unwrap().intrinsic());
    assert_eq!(Some(temporary), get_var(&m, lookup.args[0]));
    assert!(is_const_val(&m, lookup.args[1], &(thunk_id as i64)));
    let other = m.get_int(thunk_id as i64 + 1);
    let alt = call(&mut m, intrinsic, vec![lookup.args[0], other]);
    assert!(!match_values(&m, target.val, alt));

    let cloned_parent = m.nr_bodied_func("");
    let cloned_call = cv.clone_value_into(&mut m, call_v, Some(cloned_parent), &Default::default());
    assert_eq!("virtual.call", m.value_name(cloned_call));
    assert_eq!(12, m.src_info(cloned_call.into()).line);
    let cc = m.as_call_instr(cloned_call).unwrap().clone();
    let cloned_temp = get_var(&m, cc.args[0]).unwrap();
    assert_ne!(temporary, cloned_temp);
    assert!(m.func(cloned_parent).unwrap().symbols().contains(&cloned_temp));
    let ct = m.as_flow_instr(cc.callee).unwrap().clone();
    let cl = m.as_call_instr(ct.val).unwrap().clone();
    assert_eq!(Some(cloned_temp), get_var(&m, cl.args[0]));
    assert!(is_const_val(&m, cl.args[1], &(thunk_id as i64)));
    let cp = m.as_series_flow(ct.flow).unwrap().series.clone();
    assert_eq!(cloned_temp, m.as_assign_instr(cp[1]).unwrap().lhs);
}

#[test]
fn polymorphic_instruction_metadata() {
    let (mut m, mut cv) = setup();
    let t = m.unsafe_get_membered_type("polymorphic", true);
    let i = m.get_int_type();
    m.realize_membered(t, vec![i], names(&["field"]));
    let rv = m.nr_var(t, false, "");
    let receiver = m.nr_var_value(rv);
    let value = m.get_int(42);
    let extract = m.nr_named(
        ValueKind::ExtractInstr(ExtractInstr { val: receiver, field: "field".into(), polymorphic: true }),
        "extract",
    );
    let insert = m.nr_named(
        ValueKind::InsertInstr(InsertInstr { lhs: receiver, field: "field".into(), rhs: value, polymorphic: true }),
        "insert",
    );
    let ec = cv.clone_value(&mut m, extract);
    let ic = cv.clone_value(&mut m, insert);
    assert!(m.as_extract_instr(ec).unwrap().polymorphic);
    assert!(m.as_insert_instr(ic).unwrap().polymorphic);
    assert_eq!("extract", m.value_name(ec));
    assert_eq!("insert", m.value_name(ic));
    assert!(match_values(&m, extract, ec));
    assert!(match_values(&m, insert, ic));
    let plain_e = m.nr_extract(receiver, "field");
    assert!(!match_values(&m, extract, plain_e));
    m.as_extract_instr_mut(extract).unwrap().polymorphic = false;
    assert!(!match_values(&m, extract, ec));
    let named_e = m.nr_named(
        ValueKind::ExtractInstr(ExtractInstr { val: receiver, field: "field".into(), polymorphic: false }),
        "extract",
    );
    assert!(match_values(&m, extract, named_e));
    let mut cv2 = util::cloning::CloneVisitor::new(true);
    let e2 = cv2.clone_value(&mut m, extract);
    assert!(!m.as_extract_instr(e2).unwrap().polymorphic);
}

#[test]
fn stack_alloc_and_yield_in() {
    let (mut m, mut cv) = setup();
    let i = m.get_int_type();
    let fl = m.get_float_type();
    let at = m.unsafe_get_pointer_type(Some(i));
    let instr = m.nr_stack_alloc(at, 1);
    assert_eq!(1, m.as_stack_alloc_instr(instr).unwrap().count);
    assert_eq!(at, m.value_type(instr));
    assert_eq!(vec![at], m.used_types(instr.into()));
    let c = cv.clone_value(&mut m, instr);
    assert!(match_values(&m, instr, c));
    let pf = m.unsafe_get_pointer_type(Some(fl));
    assert_eq!(1, m.replace_used_type(instr.into(), at, pf));

    let y = m.nr_yield_in(at, true);
    assert_eq!(at, m.value_type(y));
    assert_eq!(vec![at], m.used_types(y.into()));
    let c = cv.clone_value(&mut m, y);
    assert!(match_values(&m, y, c));
    assert_eq!(1, m.replace_used_type(y.into(), at, pf));
}

#[test]
fn intrinsic_query_call_types_and_replacement() {
    let (mut m, mut cv) = setup();
    let i = m.get_int_type();
    let t = m.unsafe_get_pointer_type(Some(i));
    let atomic = m.get_intrinsic(Intrinsic::IsAtomic, Some(t), None);
    let callee = m.nr_var_value(atomic);
    let instr = m.nr_call(callee, vec![]);
    let b = m.get_bool_type();
    assert_eq!(b, m.value_type(instr));
    assert_eq!(0, m.as_call_instr(instr).unwrap().num_args());
    let c = cv.clone_value(&mut m, instr);
    assert!(match_values(&m, instr, c));
    let size = m.get_intrinsic(Intrinsic::Sizeof, Some(t), None);
    let sv = m.nr_var_value(size);
    assert_eq!(1, m.replace_used_value(instr.into(), callee, sv));
    assert_eq!(i, m.value_type(instr));

    let cloned = cv.force_clone_var(&mut m, atomic);
    assert_ne!(atomic, cloned);
    assert!(match_vars(&m, atomic, cloned));
    assert!(m.has_valid_intrinsic_signature(cloned));
    assert!(m.used_types(cloned.into()).contains(&t));
    let fl = m.get_float_type();
    let rep = m.unsafe_get_pointer_type(Some(fl));
    assert_eq!(1, m.replace_used_type(cloned.into(), t, rep));
    assert_eq!(Some(rep), m.func(cloned).unwrap().intrinsic_type());
    assert!(!match_vars(&m, atomic, cloned));
    assert!(m.has_valid_intrinsic_signature(cloned));
}

#[test]
fn ternary_instr_query_replace_and_cloning() {
    let (mut m, mut cv) = setup();
    let t = m.get_bool(true);
    let f = m.get_bool(false);
    let c = m.get_bool(true);
    let instr = m.nr_ternary(c, t, f);
    assert_eq!(3, m.used_values(instr.into()).len());
    let cl = cv.clone_value(&mut m, instr);
    assert!(match_values(&m, instr, cl));
    for old in [c, t, f] {
        let n = m.get_bool(true);
        assert_eq!(1, m.replace_used_value(instr.into(), old, n));
    }
}

#[test]
fn control_flow_instrs() {
    let (mut m, mut cv) = setup();
    for instr in [m.nr_continue(None), m.nr_break(None)] {
        let c = cv.clone_value(&mut m, instr);
        assert!(match_values(&m, instr, c));
    }
    let i = m.get_int_type();
    for mk in [0, 1, 2] {
        let val = m.nr_int_const(1, i);
        let instr = match mk {
            0 => m.nr_return(Some(val)),
            1 => m.nr_yield(Some(val), false),
            _ => m.nr_throw(Some(val)),
        };
        assert_eq!(1, m.used_values(instr.into()).len());
        assert_eq!(1, m.replace_used_value(instr.into(), val, None));
        assert_eq!(0, m.used_values(instr.into()).len());
        let c = cv.clone_value(&mut m, instr);
        assert!(match_values(&m, instr, c));
    }
}

#[test]
fn throw_instr_fallback() {
    let (mut m, mut cv) = setup();
    let i = m.get_int_type();
    let val = m.nr_int_const(1, i);
    let instr = m.nr_throw(Some(val));
    assert_eq!(1, m.replace_used_value(instr.into(), val, None));
    let bare = m.nr_throw(None);
    m.set_throw_fallback(bare, Some(val));
    assert_eq!(1, m.used_values(bare.into()).len());
    assert!(!match_values(&m, instr, bare));
    let cloned = cv.clone_value(&mut m, bare);
    assert!(match_values(&m, bare, cloned));
    let fb = m.as_throw_instr(cloned).unwrap().fallback.unwrap();
    assert!(match_values(&m, val, fb));
    assert_eq!(1, m.replace_used_value(bare.into(), val, None));
    assert_eq!(None, m.as_throw_instr(bare).unwrap().fallback);
    assert!(match_values(&m, instr, bare));
}

#[test]
fn flow_instr_query_replace_and_cloning() {
    let (mut m, mut cv) = setup();
    let flow = m.nr_series_flow();
    let i = m.get_int_type();
    let val = m.nr_int_const(1, i);
    let instr = m.nr_flow_instr(flow, val);
    assert_eq!(i, m.value_type(instr));
    assert_eq!(2, m.used_values(instr.into()).len());
    let c = cv.clone_value(&mut m, instr);
    assert!(match_values(&m, instr, c));
    let v2 = m.nr_int_const(2, i);
    assert_eq!(1, m.replace_used_value(instr.into(), val, v2));
    let f2 = m.nr_series_flow();
    assert_eq!(1, m.replace_used_value(instr.into(), flow, f2));
}
