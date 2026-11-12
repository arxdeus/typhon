//! The IR module: owner of every node, with the behaviour of `Node`, `Value`, `Var`, `Func`
//! and `Type` expressed as `Module` methods over ids.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::rc::Rc;

use la_arena::Arena;
use rustc_hash::FxHashMap;
use typhon_common::{SrcInfo, log_flag};

use crate::attribute::{Attribute, AttributeMap, attr_id};
use crate::base::NodeBase;
use crate::ids::{FuncId, Id, NodeRef, TypeId, ValueId, VarId, next_id};
use crate::realize::{AstTypeRef, TypeRealizer};
use crate::types::*;
use crate::util::side_effect::SideEffectStatus;
use crate::value::*;
use crate::var::{FuncData, FuncKind, Intrinsic, VarData};

/// Magic method names (`Module::*_MAGIC_NAME`) and builtin type names.
pub mod magic {
    pub const BOOL_NAME: &str = "bool";
    pub const FLOAT_NAME: &str = "float";
    pub const FLOAT32_NAME: &str = "float32";
    pub const FLOAT16_NAME: &str = "float16";
    pub const BFLOAT16_NAME: &str = "bfloat16";
    pub const FLOAT128_NAME: &str = "float128";
    pub const STRING_NAME: &str = "str";
    pub const BYTES_NAME: &str = "bytes";

    pub const EQ: &str = "__eq__";
    pub const NE: &str = "__ne__";
    pub const LT: &str = "__lt__";
    pub const GT: &str = "__gt__";
    pub const LE: &str = "__le__";
    pub const GE: &str = "__ge__";
    pub const POS: &str = "__pos__";
    pub const NEG: &str = "__neg__";
    pub const INVERT: &str = "__invert__";
    pub const ABS: &str = "__abs__";
    pub const ADD: &str = "__add__";
    pub const SUB: &str = "__sub__";
    pub const MUL: &str = "__mul__";
    pub const MATMUL: &str = "__matmul__";
    pub const TRUE_DIV: &str = "__truediv__";
    pub const FLOOR_DIV: &str = "__floordiv__";
    pub const MOD: &str = "__mod__";
    pub const POW: &str = "__pow__";
    pub const LSHIFT: &str = "__lshift__";
    pub const RSHIFT: &str = "__rshift__";
    pub const AND: &str = "__and__";
    pub const OR: &str = "__or__";
    pub const XOR: &str = "__xor__";
    pub const IADD: &str = "__iadd__";
    pub const ISUB: &str = "__isub__";
    pub const IMUL: &str = "__imul__";
    pub const IMATMUL: &str = "__imatmul__";
    pub const ITRUE_DIV: &str = "__itruediv__";
    pub const IFLOOR_DIV: &str = "__ifloordiv__";
    pub const IMOD: &str = "__imod__";
    pub const IPOW: &str = "__ipow__";
    pub const ILSHIFT: &str = "__ilshift__";
    pub const IRSHIFT: &str = "__irshift__";
    pub const IAND: &str = "__iand__";
    pub const IOR: &str = "__ior__";
    pub const IXOR: &str = "__ixor__";
    pub const RADD: &str = "__radd__";
    pub const RSUB: &str = "__rsub__";
    pub const RMUL: &str = "__rmul__";
    pub const RMATMUL: &str = "__rmatmul__";
    pub const RTRUE_DIV: &str = "__rtruediv__";
    pub const RFLOOR_DIV: &str = "__rfloordiv__";
    pub const RMOD: &str = "__rmod__";
    pub const RPOW: &str = "__rpow__";
    pub const RLSHIFT: &str = "__rlshift__";
    pub const RRSHIFT: &str = "__rrshift__";
    pub const RAND: &str = "__rand__";
    pub const ROR: &str = "__ror__";
    pub const RXOR: &str = "__rxor__";
    pub const INT: &str = "__int__";
    pub const FLOAT: &str = "__float__";
    pub const BOOL: &str = "__bool__";
    pub const STR: &str = "__str__";
    pub const REPR: &str = "__repr__";
    pub const CALL: &str = "__call__";
    pub const GETITEM: &str = "__getitem__";
    pub const SETITEM: &str = "__setitem__";
    pub const ITER: &str = "__iter__";
    pub const LEN: &str = "__len__";
    pub const NEW: &str = "__new__";
    pub const INIT: &str = "__init__";
}

#[derive(Default, Debug)]
struct ArenaFrame {
    values: Vec<Id>,
    vars: Vec<Id>,
    types: Vec<String>,
}

/// owner of every IR node.
pub struct Module {
    base: NodeBase,
    values: Arena<ValueData>,
    vars: Arena<VarData>,
    types: Arena<TypeData>,
    // listed (i.e. part of the module lists) nodes; insertion order == arena order.
    value_listed: Vec<bool>,
    var_listed: Vec<bool>,
    type_listed: Vec<bool>,
    value_map: FxHashMap<Id, ValueId>,
    var_map: FxHashMap<Id, VarId>,
    type_map: FxHashMap<String, TypeId>,
    arenas: Vec<ArenaFrame>,
    intrinsics: BTreeMap<(Intrinsic, String, String), Id>,
    main_func: FuncId,
    argv_var: VarId,
    argc_var: VarId,
    none_type: Option<TypeId>,
    realizer: Option<Rc<RefCell<dyn TypeRealizer>>>,
}

impl std::fmt::Debug for Module {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Module")
            .field("name", &self.base.name)
            .field("values", &self.values.len())
            .field("vars", &self.vars.len())
            .field("types", &self.types.len())
            .finish()
    }
}

macro_rules! value_accessors {
    ($($as:ident, $as_mut:ident, $variant:ident, $ty:ty;)*) => {
        $(
            #[doc = concat!("`cast<", stringify!($variant), ">(v)`.")]
            pub fn $as(&self, v: ValueId) -> Option<&$ty> {
                match &self.values[self.actual_value(v)].kind {
                    ValueKind::$variant(x) => Some(x),
                    _ => None,
                }
            }
            pub fn $as_mut(&mut self, v: ValueId) -> Option<&mut $ty> {
                let a = self.actual_value(v);
                match &mut self.values[a].kind {
                    ValueKind::$variant(x) => Some(x),
                    _ => None,
                }
            }
        )*
    };
}

impl Module {
    // ------------------------------------------------------------------
    // construction
    // ------------------------------------------------------------------

    /// `Module(cache = nullptr, name)`.
    pub fn new(name: impl Into<String>) -> Self {
        let dummy_values: Arena<ValueData> = Arena::new();
        let _ = dummy_values;
        let mut vars: Arena<VarData> = Arena::new();
        // Placeholder ids, fixed up below.
        let placeholder = vars.alloc(VarData {
            base: NodeBase::new(""),
            ty: None,
            global: false,
            external: false,
            tls: false,
            func: None,
            any: false,
            uid: -1,
            replacement: None,
            alive: false,
        });
        let mut m = Module {
            base: NodeBase::new(name),
            values: Arena::new(),
            vars,
            types: Arena::new(),
            value_listed: Vec::new(),
            var_listed: vec![false],
            type_listed: Vec::new(),
            value_map: FxHashMap::default(),
            var_map: FxHashMap::default(),
            type_map: FxHashMap::default(),
            arenas: Vec::new(),
            intrinsics: BTreeMap::new(),
            main_func: placeholder,
            argv_var: placeholder,
            argc_var: placeholder,
            none_type: None,
            realizer: None,
        };
        // mainFunc = make_unique<BodiedFunc>("main") (not stored in the module lists).
        let main = m.alloc_var_raw(VarData {
            base: NodeBase::new("main"),
            ty: None,
            global: true,
            external: false,
            tls: false,
            func: Some(Box::new(FuncData::new(FuncKind::bodied()))),
            any: false,
            uid: next_id(),
            replacement: None,
            alive: true,
        });
        m.vars[main].base.replaceable = false;
        m.main_func = main;
        let s = m.get_string_type();
        let argv_ty = m.unsafe_get_pointer_type(Some(s));
        let argv = m.alloc_var_raw(VarData {
            base: NodeBase::new(".argv"),
            ty: Some(argv_ty),
            global: true,
            external: false,
            tls: false,
            func: None,
            any: false,
            uid: next_id(),
            replacement: None,
            alive: true,
        });
        m.vars[argv].base.replaceable = false;
        m.argv_var = argv;
        let i = m.get_int_type();
        let argc = m.alloc_var_raw(VarData {
            base: NodeBase::new(".argc"),
            ty: Some(i),
            global: true,
            external: false,
            tls: false,
            func: None,
            any: false,
            uid: next_id(),
            replacement: None,
            alive: true,
        });
        m.vars[argc].base.replaceable = false;
        m.argc_var = argc;
        // Note: the reference realizes NoneType lazily through the cache; here the structural
        // fallback is created eagerly so that `value_type(&self)` works on flows/instrs.
        // A parser-realized `NoneType` registered later under the same name supersedes it.
        m.get_none_type();
        m
    }

    fn alloc_var_raw(&mut self, d: VarData) -> VarId {
        let id = self.vars.alloc(d);
        self.var_listed.push(false);
        id
    }

    /// `setCache`: attach the parser cache.
    pub fn set_realizer(&mut self, r: Option<Rc<RefCell<dyn TypeRealizer>>>) {
        self.realizer = r;
        self.none_type = None;
    }
    /// `getCache`.
    pub fn realizer(&self) -> Option<Rc<RefCell<dyn TypeRealizer>>> {
        self.realizer.clone()
    }

    pub fn main_func(&self) -> FuncId {
        self.main_func
    }
    pub fn argv_var(&self) -> VarId {
        self.argv_var
    }
    pub fn argc_var(&self) -> VarId {
        self.argc_var
    }
    pub fn module_base(&self) -> &NodeBase {
        &self.base
    }

    // ------------------------------------------------------------------
    // storage
    // ------------------------------------------------------------------

    fn store_value(&mut self, data: ValueData) -> ValueId {
        let uid = data.uid;
        let id = self.values.alloc(data);
        self.value_listed.push(true);
        self.value_map.insert(uid, id);
        if let Some(a) = self.arenas.last_mut() {
            a.values.push(uid);
        }
        id
    }

    fn store_var(&mut self, data: VarData) -> VarId {
        let uid = data.uid;
        let id = self.vars.alloc(data);
        self.var_listed.push(true);
        self.var_map.insert(uid, id);
        if let Some(a) = self.arenas.last_mut() {
            a.vars.push(uid);
        }
        id
    }

    fn store_type(&mut self, data: TypeData) -> TypeId {
        let name = data.base.name.clone();
        let id = self.types.alloc(data);
        self.type_listed.push(true);
        if let Some(old) = self.type_map.insert(name.clone(), id) {
            // the reference keeps the old list entry; only the map is overwritten.
            let _ = old;
        }
        if let Some(a) = self.arenas.last_mut() {
            a.types.push(name);
        }
        id
    }

    /// `N<T>(src, ...)` for values.
    pub fn n(&mut self, src: SrcInfo, kind: ValueKind) -> ValueId {
        self.n_named(src, kind, "")
    }
    /// `N<T>(src, ..., name)` for values.
    pub fn n_named(&mut self, src: SrcInfo, kind: ValueKind, name: impl Into<String>) -> ValueId {
        let mut base = NodeBase::new(name);
        base.attributes.insert(attr_id::SRC_INFO, Attribute::SrcInfo(src));
        self.store_value(ValueData { base, kind, uid: next_id(), replacement: None, alive: true })
    }
    /// `Nr<T>(...)` for values.
    pub fn nr(&mut self, kind: ValueKind) -> ValueId {
        self.n(SrcInfo::default(), kind)
    }
    /// `Nr<T>(..., name)` for values.
    pub fn nr_named(&mut self, kind: ValueKind, name: impl Into<String>) -> ValueId {
        self.n_named(SrcInfo::default(), kind, name)
    }

    /// `N<Var>(src, type, global, external, tls, name)`.
    pub fn n_var(
        &mut self,
        src: SrcInfo,
        ty: Option<TypeId>,
        global: bool,
        external: bool,
        tls: bool,
        name: impl Into<String>,
    ) -> VarId {
        let mut base = NodeBase::new(name);
        base.attributes.insert(attr_id::SRC_INFO, Attribute::SrcInfo(src));
        self.store_var(VarData {
            base,
            ty,
            global,
            external,
            tls,
            func: None,
            any: false,
            uid: next_id(),
            replacement: None,
            alive: true,
        })
    }
    /// `Nr<Var>(type, global, name)`.
    pub fn nr_var(&mut self, ty: TypeId, global: bool, name: impl Into<String>) -> VarId {
        self.n_var(SrcInfo::default(), Some(ty), global, false, false, name)
    }
    /// `N<Func subclass>(src, name)`.
    pub fn n_func(&mut self, src: SrcInfo, name: impl Into<String>, kind: FuncKind) -> FuncId {
        let mut base = NodeBase::new(name);
        base.attributes.insert(attr_id::SRC_INFO, Attribute::SrcInfo(src));
        self.store_var(VarData {
            base,
            ty: None,
            global: true,
            external: false,
            tls: false,
            func: Some(Box::new(FuncData::new(kind))),
            any: false,
            uid: next_id(),
            replacement: None,
            alive: true,
        })
    }
    pub fn nr_func(&mut self, name: impl Into<String>, kind: FuncKind) -> FuncId {
        self.n_func(SrcInfo::default(), name, kind)
    }
    pub fn nr_bodied_func(&mut self, name: impl Into<String>) -> FuncId {
        self.nr_func(name, FuncKind::bodied())
    }
    pub fn nr_external_func(&mut self, name: impl Into<String>) -> FuncId {
        self.nr_func(name, FuncKind::External)
    }
    pub fn nr_internal_func(&mut self, name: impl Into<String>) -> FuncId {
        self.nr_func(name, FuncKind::internal())
    }
    pub fn nr_llvm_func(&mut self, name: impl Into<String>) -> FuncId {
        self.nr_func(name, FuncKind::llvm())
    }
    /// `util::AnyVar` (`func == false`) / `util::AnyFunc` (`func == true`).
    pub fn nr_any_var(&mut self, func: bool) -> VarId {
        let v = if func {
            let f = self.nr_bodied_func("");
            if let Some(fd) = self.func_mut(f) {
                fd.unmangled_name = "any".into();
            }
            f
        } else {
            self.n_var(SrcInfo::default(), None, false, false, false, "")
        };
        self.vars[v].any = true;
        v
    }

    /// `Nr<SomeType>(...)`: creates and stores a type (overwriting the name entry).
    pub fn add_type(&mut self, name: impl Into<String>, kind: TypeKind) -> TypeId {
        self.n_type(SrcInfo::default(), name, kind)
    }
    pub fn n_type(&mut self, src: SrcInfo, name: impl Into<String>, kind: TypeKind) -> TypeId {
        let mut d = TypeData::new(name, kind);
        d.base.attributes.insert(attr_id::SRC_INFO, Attribute::SrcInfo(src));
        self.store_type(d)
    }
    /// `Nr<RecordType>(name, types, names)`.
    pub fn nr_record_type(&mut self, name: impl Into<String>, types: Vec<TypeId>, names: Vec<String>) -> TypeId {
        let fields = types.into_iter().zip(names).map(|(t, n)| Field::new(n, t)).collect();
        self.add_type(name, TypeKind::Record { fields })
    }
    /// `Nr<RecordType>(name, types)`: fields named "1", "2", ...
    pub fn nr_record_type_unnamed(&mut self, name: impl Into<String>, types: Vec<TypeId>) -> TypeId {
        let fields = types.into_iter().enumerate().map(|(i, t)| Field::new((i + 1).to_string(), t)).collect();
        self.add_type(name, TypeKind::Record { fields })
    }
    /// `Nr<RefType>(name, contents, polymorphic)`.
    pub fn nr_ref_type(&mut self, name: impl Into<String>, contents: TypeId, polymorphic: bool) -> TypeId {
        self.add_type(name, TypeKind::Ref { contents, polymorphic })
    }
    /// `Nr<FuncType>(name, ret, args, variadic)`.
    pub fn nr_func_type(&mut self, name: impl Into<String>, ret: TypeId, args: Vec<TypeId>, variadic: bool) -> TypeId {
        self.add_type(name, TypeKind::Func { ret, args, variadic })
    }

    // ---- value builders ------------------------------------------------

    pub fn nr_int_const(&mut self, val: i64, ty: TypeId) -> ValueId {
        self.nr(ValueKind::IntConst(TemplatedConst { val, ty }))
    }
    pub fn nr_float_const(&mut self, val: f64, ty: TypeId) -> ValueId {
        self.nr(ValueKind::FloatConst(TemplatedConst { val, ty }))
    }
    pub fn nr_bool_const(&mut self, val: bool, ty: TypeId) -> ValueId {
        self.nr(ValueKind::BoolConst(TemplatedConst { val, ty }))
    }
    pub fn nr_string_const(&mut self, val: impl Into<String>, ty: TypeId) -> ValueId {
        self.nr(ValueKind::StringConst(TemplatedConst { val: val.into(), ty }))
    }
    pub fn nr_bytes_const(&mut self, val: impl Into<String>, ty: TypeId) -> ValueId {
        self.nr(ValueKind::BytesConst(TemplatedConst { val: val.into(), ty }))
    }
    pub fn nr_var_value(&mut self, var: VarId) -> ValueId {
        self.nr(ValueKind::VarValue(VarValue { var }))
    }
    pub fn nr_pointer_value(&mut self, var: VarId, fields: Vec<String>) -> ValueId {
        self.nr(ValueKind::PointerValue(PointerValue { var, fields }))
    }
    pub fn nr_series_flow(&mut self) -> ValueId {
        self.nr(ValueKind::SeriesFlow(SeriesFlow::default()))
    }
    pub fn nr_if_flow(&mut self, cond: ValueId, t: ValueId, f: Option<ValueId>) -> ValueId {
        self.nr(ValueKind::IfFlow(IfFlow { cond, true_branch: t, false_branch: f }))
    }
    pub fn nr_while_flow(&mut self, cond: ValueId, body: ValueId) -> ValueId {
        self.nr(ValueKind::WhileFlow(WhileFlow { cond, body }))
    }
    pub fn nr_for_flow(&mut self, iter: ValueId, body: ValueId, var: VarId) -> ValueId {
        self.nr(ValueKind::ForFlow(ForFlow { iter, body, var, schedule: None, is_async: false }))
    }
    pub fn nr_imperative_for_flow(
        &mut self,
        start: ValueId,
        step: i64,
        end: ValueId,
        body: ValueId,
        var: VarId,
    ) -> ValueId {
        self.nr(ValueKind::ImperativeForFlow(ImperativeForFlow { start, step, end, body, var, schedule: None }))
    }
    pub fn nr_try_catch_flow(&mut self, body: ValueId, finally: Option<ValueId>, else_: Option<ValueId>) -> ValueId {
        self.nr(ValueKind::TryCatchFlow(TryCatchFlow { catches: Vec::new(), body, else_, finally }))
    }
    pub fn nr_pipeline_flow(&mut self, stages: Vec<Stage>) -> ValueId {
        self.nr(ValueKind::PipelineFlow(PipelineFlow { stages }))
    }
    pub fn nr_assign(&mut self, lhs: VarId, rhs: ValueId) -> ValueId {
        self.nr(ValueKind::AssignInstr(AssignInstr { lhs, rhs }))
    }
    pub fn nr_extract(&mut self, val: ValueId, field: impl Into<String>) -> ValueId {
        self.nr(ValueKind::ExtractInstr(ExtractInstr { val, field: field.into(), polymorphic: false }))
    }
    pub fn nr_insert(&mut self, lhs: ValueId, field: impl Into<String>, rhs: ValueId) -> ValueId {
        self.nr(ValueKind::InsertInstr(InsertInstr { lhs, field: field.into(), rhs, polymorphic: false }))
    }
    pub fn nr_call(&mut self, callee: ValueId, args: Vec<ValueId>) -> ValueId {
        self.nr(ValueKind::CallInstr(CallInstr { callee, args }))
    }
    pub fn nr_stack_alloc(&mut self, ptr_type: TypeId, count: i64) -> ValueId {
        self.nr(ValueKind::StackAllocInstr(StackAllocInstr { ptr_type, count }))
    }
    pub fn nr_yield_in(&mut self, ty: TypeId, suspend: bool) -> ValueId {
        self.nr(ValueKind::YieldInInstr(YieldInInstr { ty, suspend }))
    }
    pub fn nr_ternary(&mut self, cond: ValueId, t: ValueId, f: ValueId) -> ValueId {
        self.nr(ValueKind::TernaryInstr(TernaryInstr { cond, true_value: t, false_value: f }))
    }
    pub fn nr_break(&mut self, loop_: Option<ValueId>) -> ValueId {
        self.nr(ValueKind::BreakInstr(BreakInstr { loop_ }))
    }
    pub fn nr_continue(&mut self, loop_: Option<ValueId>) -> ValueId {
        self.nr(ValueKind::ContinueInstr(ContinueInstr { loop_ }))
    }
    pub fn nr_return(&mut self, value: Option<ValueId>) -> ValueId {
        self.nr(ValueKind::ReturnInstr(ReturnInstr { value }))
    }
    pub fn nr_yield(&mut self, value: Option<ValueId>, final_: bool) -> ValueId {
        self.nr(ValueKind::YieldInstr(YieldInstr { value, final_ }))
    }
    pub fn nr_await(&mut self, value: ValueId, ty: TypeId, generator: bool) -> ValueId {
        self.nr(ValueKind::AwaitInstr(AwaitInstr { value, ty, generator }))
    }
    pub fn nr_throw(&mut self, value: Option<ValueId>) -> ValueId {
        self.nr(ValueKind::ThrowInstr(ThrowInstr { value, fallback: None }))
    }
    pub fn nr_flow_instr(&mut self, flow: ValueId, val: ValueId) -> ValueId {
        self.nr(ValueKind::FlowInstr(FlowInstr { flow, val }))
    }
    /// `util::AnyValue` / `util::AnyFlow`.
    pub fn nr_any_value(&mut self, flow: bool) -> ValueId {
        self.nr(ValueKind::Any { flow })
    }

    // ------------------------------------------------------------------
    // module lists
    // ------------------------------------------------------------------

    /// Iteration over module variables (`begin()/end()`), in insertion order.
    pub fn vars(&self) -> Vec<VarId> {
        self.vars.iter().filter(|(id, _)| self.var_listed[id.into_raw().into_u32() as usize]).map(|(id, _)| id).collect()
    }
    /// `values_begin()/values_end()`.
    pub fn values(&self) -> Vec<ValueId> {
        self.values
            .iter()
            .filter(|(id, _)| self.value_listed[id.into_raw().into_u32() as usize])
            .map(|(id, _)| id)
            .collect()
    }
    /// `types_begin()/types_end()`.
    pub fn types(&self) -> Vec<TypeId> {
        self.types
            .iter()
            .filter(|(id, _)| self.type_listed[id.into_raw().into_u32() as usize])
            .map(|(id, _)| id)
            .collect()
    }
    /// `getVar(id)`.
    pub fn get_var(&self, uid: Id) -> Option<VarId> {
        self.var_map.get(&uid).copied()
    }
    /// `getValue(id)`.
    pub fn get_value(&self, uid: Id) -> Option<ValueId> {
        self.value_map.get(&uid).copied()
    }
    /// `getType(name)`.
    pub fn get_type(&self, name: &str) -> Option<TypeId> {
        self.type_map.get(name).copied()
    }
    /// `remove(Var*)`.
    pub fn remove_var(&mut self, v: VarId) {
        let uid = self.var_uid(v);
        if let Some(id) = self.var_map.remove(&uid) {
            self.var_listed[id.into_raw().into_u32() as usize] = false;
            self.vars[id].alive = false;
        }
    }
    /// `remove(Value*)`.
    pub fn remove_value(&mut self, v: ValueId) {
        let uid = self.value_uid(v);
        if let Some(id) = self.value_map.remove(&uid) {
            self.value_listed[id.into_raw().into_u32() as usize] = false;
            self.values[id].alive = false;
        }
    }
    /// `remove(Type*)`.
    pub fn remove_type(&mut self, t: TypeId) {
        let name = self.type_name(t).to_string();
        if let Some(id) = self.type_map.remove(&name) {
            self.type_listed[id.into_raw().into_u32() as usize] = false;
            self.types[id].alive = false;
            if self.none_type == Some(id) {
                self.none_type = None;
            }
        }
    }
    /// Whether the node is still owned by the module lists.
    pub fn is_value_alive(&self, v: ValueId) -> bool {
        self.values[v].alive
    }
    pub fn is_var_alive(&self, v: VarId) -> bool {
        self.vars[v].alive
    }

    /// `pushArena()`.
    pub fn push_arena(&mut self) {
        self.arenas.push(ArenaFrame::default());
    }
    /// `popArena()`.
    pub fn pop_arena(&mut self) {
        let Some(frame) = self.arenas.pop() else { return };
        for uid in frame.values {
            if let Some(id) = self.value_map.remove(&uid) {
                self.value_listed[id.into_raw().into_u32() as usize] = false;
                self.values[id].alive = false;
            }
        }
        for uid in frame.vars {
            if let Some(id) = self.var_map.remove(&uid) {
                self.var_listed[id.into_raw().into_u32() as usize] = false;
                self.vars[id].alive = false;
            }
        }
        for name in frame.types {
            if let Some(id) = self.type_map.remove(&name) {
                self.type_listed[id.into_raw().into_u32() as usize] = false;
                self.types[id].alive = false;
                if self.none_type == Some(id) {
                    self.none_type = None;
                }
            }
        }
        let var_map = &self.var_map;
        self.intrinsics.retain(|_, uid| var_map.contains_key(uid));
    }

    // ------------------------------------------------------------------
    // raw access and replacement forwarding
    // ------------------------------------------------------------------

    /// `getActual()` for values.
    pub fn actual_value(&self, mut v: ValueId) -> ValueId {
        while let Some(r) = self.values[v].replacement {
            v = r;
        }
        v
    }
    pub fn actual_var(&self, mut v: VarId) -> VarId {
        while let Some(r) = self.vars[v].replacement {
            v = r;
        }
        v
    }
    pub fn actual_type(&self, mut t: TypeId) -> TypeId {
        while let Some(r) = self.types[t].replacement {
            t = r;
        }
        t
    }
    pub fn actual(&self, n: NodeRef) -> NodeRef {
        match n {
            NodeRef::Value(v) => NodeRef::Value(self.actual_value(v)),
            NodeRef::Var(v) => NodeRef::Var(self.actual_var(v)),
            NodeRef::Type(t) => NodeRef::Type(self.actual_type(t)),
            NodeRef::Module => NodeRef::Module,
        }
    }

    /// Value data (replacement resolved).
    pub fn value(&self, v: ValueId) -> &ValueData {
        &self.values[self.actual_value(v)]
    }
    pub fn value_mut(&mut self, v: ValueId) -> &mut ValueData {
        let a = self.actual_value(v);
        &mut self.values[a]
    }
    /// Raw value slot (no replacement resolution).
    pub fn value_raw(&self, v: ValueId) -> &ValueData {
        &self.values[v]
    }
    pub fn kind(&self, v: ValueId) -> &ValueKind {
        &self.value(v).kind
    }
    pub fn kind_mut(&mut self, v: ValueId) -> &mut ValueKind {
        &mut self.value_mut(v).kind
    }
    pub fn var(&self, v: VarId) -> &VarData {
        &self.vars[self.actual_var(v)]
    }
    pub fn var_mut(&mut self, v: VarId) -> &mut VarData {
        let a = self.actual_var(v);
        &mut self.vars[a]
    }
    pub fn var_raw(&self, v: VarId) -> &VarData {
        &self.vars[v]
    }
    pub fn ty(&self, t: TypeId) -> &TypeData {
        &self.types[self.actual_type(t)]
    }
    pub fn ty_mut(&mut self, t: TypeId) -> &mut TypeData {
        let a = self.actual_type(t);
        &mut self.types[a]
    }
    pub fn type_kind(&self, t: TypeId) -> &TypeKind {
        &self.ty(t).kind
    }

    /// `Value::getId()`.
    pub fn value_uid(&self, v: ValueId) -> Id {
        self.value(v).uid
    }
    /// `Var::getId()`.
    pub fn var_uid(&self, v: VarId) -> Id {
        self.var(v).uid
    }
    /// `getId()` for any node with an id (values and vars); types/modules return -1.
    pub fn node_uid(&self, n: NodeRef) -> Id {
        match n {
            NodeRef::Value(v) => self.value_uid(v),
            NodeRef::Var(v) => self.var_uid(v),
            _ => -1,
        }
    }

    /// `ReplaceableNodeBase<Value>::replaceAll`.
    pub fn replace_all_value(&mut self, v: ValueId, with: ValueId) {
        assert!(self.values[v].base.replaceable, "node {} not replaceable", self.reference_string(with.into()));
        self.values[v].replacement = Some(with);
    }
    pub fn replace_all_var(&mut self, v: VarId, with: VarId) {
        assert!(self.vars[v].base.replaceable, "node {} not replaceable", self.reference_string(with.into()));
        self.vars[v].replacement = Some(with);
    }
    pub fn replace_all_type(&mut self, t: TypeId, with: TypeId) {
        assert!(self.types[t].base.replaceable, "node {} not replaceable", self.reference_string(with.into()));
        self.types[t].replacement = Some(with);
    }
    pub fn has_replacement(&self, n: NodeRef) -> bool {
        match n {
            NodeRef::Value(v) => self.values[v].replacement.is_some(),
            NodeRef::Var(v) => self.vars[v].replacement.is_some(),
            NodeRef::Type(t) => self.types[t].replacement.is_some(),
            NodeRef::Module => false,
        }
    }
    pub fn is_replaceable(&self, n: NodeRef) -> bool {
        match n {
            NodeRef::Value(v) => self.values[v].base.replaceable,
            NodeRef::Var(v) => self.vars[v].base.replaceable,
            NodeRef::Type(t) => self.types[t].base.replaceable,
            NodeRef::Module => false,
        }
    }
    pub fn set_replaceable(&mut self, n: NodeRef, r: bool) {
        match n {
            NodeRef::Value(v) => self.values[v].base.replaceable = r,
            NodeRef::Var(v) => self.vars[v].base.replaceable = r,
            NodeRef::Type(t) => self.types[t].base.replaceable = r,
            NodeRef::Module => self.base.replaceable = r,
        }
    }

    // ------------------------------------------------------------------
    // generic node API (names, attributes)
    // ------------------------------------------------------------------

    pub fn node_base(&self, n: NodeRef) -> &NodeBase {
        match self.actual(n) {
            NodeRef::Value(v) => &self.values[v].base,
            NodeRef::Var(v) => &self.vars[v].base,
            NodeRef::Type(t) => &self.types[t].base,
            NodeRef::Module => &self.base,
        }
    }
    pub fn node_base_mut(&mut self, n: NodeRef) -> &mut NodeBase {
        match self.actual(n) {
            NodeRef::Value(v) => &mut self.values[v].base,
            NodeRef::Var(v) => &mut self.vars[v].base,
            NodeRef::Type(t) => &mut self.types[t].base,
            NodeRef::Module => &mut self.base,
        }
    }
    /// `getName()`.
    pub fn name(&self, n: NodeRef) -> &str {
        &self.node_base(n).name
    }
    /// `setName()`.
    pub fn set_name(&mut self, n: NodeRef, name: impl Into<String>) {
        self.node_base_mut(n).name = name.into();
    }
    pub fn value_name(&self, v: ValueId) -> &str {
        &self.value(v).base.name
    }
    pub fn var_name(&self, v: VarId) -> &str {
        &self.var(v).base.name
    }
    pub fn type_name(&self, t: TypeId) -> &str {
        &self.ty(t).base.name
    }
    pub fn attributes(&self, n: NodeRef) -> &AttributeMap {
        &self.node_base(n).attributes
    }
    pub fn attributes_mut(&mut self, n: NodeRef) -> &mut AttributeMap {
        &mut self.node_base_mut(n).attributes
    }
    pub fn has_attribute(&self, n: NodeRef, key: i32) -> bool {
        self.attributes(n).contains(key)
    }
    pub fn get_attribute(&self, n: NodeRef, key: i32) -> Option<&Attribute> {
        self.attributes(n).get(key)
    }
    pub fn get_attribute_mut(&mut self, n: NodeRef, key: i32) -> Option<&mut Attribute> {
        self.attributes_mut(n).get_mut(key)
    }
    /// `setAttribute(attr)` (key = `attr.id()`).
    pub fn set_attribute(&mut self, n: NodeRef, a: Attribute) {
        let k = a.id();
        self.attributes_mut(n).insert(k, a);
    }
    /// `setAttribute(attr, key)`.
    pub fn set_attribute_key(&mut self, n: NodeRef, a: Attribute, key: i32) {
        self.attributes_mut(n).insert(key, a);
    }
    /// `eraseAttribute(key)`. Note: the reference erases on the node itself (not the actual);
    /// here the actual node is used for consistency with the other attribute accessors.
    pub fn erase_attribute(&mut self, n: NodeRef, key: i32) {
        self.attributes_mut(n).erase(key);
    }
    /// `cloneAttributesFrom(n)`.
    pub fn clone_attributes_from(&mut self, to: NodeRef, from: NodeRef) {
        let attrs = self.attributes(from).clone();
        *self.attributes_mut(to) = attrs;
    }
    /// `getSrcInfo()`.
    pub fn src_info(&self, n: NodeRef) -> SrcInfo {
        self.get_attribute(n, attr_id::SRC_INFO).and_then(|a| a.as_src_info()).cloned().unwrap_or_default()
    }
    /// `setSrcInfo()`.
    pub fn set_src_info(&mut self, n: NodeRef, s: SrcInfo) {
        self.set_attribute(n, Attribute::SrcInfo(s));
    }
    /// `referenceString()`.
    pub fn reference_string(&self, n: NodeRef) -> String {
        match n {
            NodeRef::Value(v) => format!("{}.{}", self.value_name(v), self.value_uid(v)),
            NodeRef::Var(v) => format!("{}.{}", self.var_name(v), self.var_uid(v)),
            NodeRef::Type(t) => self.type_name(t).to_string(),
            NodeRef::Module => self.base.name.clone(),
        }
    }

    value_accessors! {
        as_int_const, as_int_const_mut, IntConst, IntConst;
        as_float_const, as_float_const_mut, FloatConst, FloatConst;
        as_bool_const, as_bool_const_mut, BoolConst, BoolConst;
        as_string_const, as_string_const_mut, StringConst, StringConst;
        as_bytes_const, as_bytes_const_mut, BytesConst, BytesConst;
        as_var_value, as_var_value_mut, VarValue, VarValue;
        as_pointer_value, as_pointer_value_mut, PointerValue, PointerValue;
        as_series_flow, as_series_flow_mut, SeriesFlow, SeriesFlow;
        as_while_flow, as_while_flow_mut, WhileFlow, WhileFlow;
        as_for_flow, as_for_flow_mut, ForFlow, ForFlow;
        as_imperative_for_flow, as_imperative_for_flow_mut, ImperativeForFlow, ImperativeForFlow;
        as_if_flow, as_if_flow_mut, IfFlow, IfFlow;
        as_try_catch_flow, as_try_catch_flow_mut, TryCatchFlow, TryCatchFlow;
        as_pipeline_flow, as_pipeline_flow_mut, PipelineFlow, PipelineFlow;
        as_assign_instr, as_assign_instr_mut, AssignInstr, AssignInstr;
        as_extract_instr, as_extract_instr_mut, ExtractInstr, ExtractInstr;
        as_insert_instr, as_insert_instr_mut, InsertInstr, InsertInstr;
        as_call_instr, as_call_instr_mut, CallInstr, CallInstr;
        as_stack_alloc_instr, as_stack_alloc_instr_mut, StackAllocInstr, StackAllocInstr;
        as_yield_in_instr, as_yield_in_instr_mut, YieldInInstr, YieldInInstr;
        as_ternary_instr, as_ternary_instr_mut, TernaryInstr, TernaryInstr;
        as_break_instr, as_break_instr_mut, BreakInstr, BreakInstr;
        as_continue_instr, as_continue_instr_mut, ContinueInstr, ContinueInstr;
        as_return_instr, as_return_instr_mut, ReturnInstr, ReturnInstr;
        as_yield_instr, as_yield_instr_mut, YieldInstr, YieldInstr;
        as_await_instr, as_await_instr_mut, AwaitInstr, AwaitInstr;
        as_throw_instr, as_throw_instr_mut, ThrowInstr, ThrowInstr;
        as_flow_instr, as_flow_instr_mut, FlowInstr, FlowInstr;
    }

    pub fn is_flow(&self, v: ValueId) -> bool {
        self.kind(v).is_flow()
    }
    pub fn is_const(&self, v: ValueId) -> bool {
        self.kind(v).is_const()
    }
    pub fn is_instr(&self, v: ValueId) -> bool {
        self.kind(v).is_instr()
    }

    /// `SeriesFlow::push_back`.
    pub fn series_push(&mut self, series: ValueId, v: ValueId) {
        self.as_series_flow_mut(series).expect("not a series flow").series.push(v);
    }
    /// `SeriesFlow::insert(begin() + idx, v)`.
    pub fn series_insert(&mut self, series: ValueId, idx: usize, v: ValueId) {
        self.as_series_flow_mut(series).expect("not a series flow").series.insert(idx, v);
    }
    /// `TryCatchFlow::emplace_back(handler, type, var)`.
    pub fn try_catch_push(&mut self, tc: ValueId, handler: ValueId, ty: Option<TypeId>, var: Option<VarId>) {
        self.as_try_catch_flow_mut(tc).expect("not a try-catch flow").catches.push(Catch { handler, ty, var });
    }
    /// `ThrowInstr::setFallback`.
    pub fn set_throw_fallback(&mut self, t: ValueId, fallback: Option<ValueId>) {
        self.as_throw_instr_mut(t).expect("not a throw instr").fallback = fallback;
    }

    // ------------------------------------------------------------------
    // vars and functions
    // ------------------------------------------------------------------

    pub fn is_func(&self, v: VarId) -> bool {
        self.var(v).func.is_some()
    }
    pub fn func(&self, f: FuncId) -> Option<&FuncData> {
        self.var(f).func.as_deref()
    }
    pub fn func_mut(&mut self, f: FuncId) -> Option<&mut FuncData> {
        self.var_mut(f).func.as_deref_mut()
    }
    /// `Var::getType()`.
    pub fn var_type(&self, v: VarId) -> Option<TypeId> {
        self.var(v).ty
    }
    /// `Var::setType()`.
    pub fn set_var_type(&mut self, v: VarId, t: TypeId) {
        self.var_mut(v).ty = Some(t);
    }
    pub fn is_global(&self, v: VarId) -> bool {
        self.var(v).global
    }
    pub fn set_global(&mut self, v: VarId, g: bool) {
        self.var_mut(v).global = g;
    }
    pub fn is_external(&self, v: VarId) -> bool {
        self.var(v).external
    }
    pub fn is_thread_local(&self, v: VarId) -> bool {
        self.var(v).tls
    }
    /// `BodiedFunc::push_back(var)`.
    pub fn func_push_symbol(&mut self, f: FuncId, v: VarId) {
        self.func_mut(f).and_then(|fd| fd.symbols_mut()).expect("not a bodied func").push(v);
    }
    /// `BodiedFunc::setBody`.
    pub fn set_func_body(&mut self, f: FuncId, body: Option<ValueId>) {
        self.func_mut(f).expect("not a func").set_body(body);
    }
    pub fn func_body(&self, f: FuncId) -> Option<ValueId> {
        self.func(f).and_then(|fd| fd.body())
    }
    pub fn func_args(&self, f: FuncId) -> Vec<VarId> {
        self.func(f).map(|fd| fd.args.clone()).unwrap_or_default()
    }

    /// `Func::realize(newType, names)`.
    pub fn realize_func(&mut self, f: FuncId, new_type: TypeId, names: &[String]) {
        let arg_types = match self.type_kind(new_type) {
            TypeKind::Func { args, .. } => args.clone(),
            _ => panic!("{} is not a function type", self.type_name(new_type)),
        };
        self.set_var_type(f, new_type);
        let mut args = Vec::with_capacity(arg_types.len());
        for (i, t) in arg_types.into_iter().enumerate() {
            let name = names.get(i).cloned().unwrap_or_default();
            args.push(self.n_var(SrcInfo::default(), Some(t), false, false, false, name));
        }
        self.func_mut(f).expect("not a func").args = args;
    }

    /// `Func::getArgVar(name)`.
    pub fn func_arg_var(&self, f: FuncId, name: &str) -> Option<VarId> {
        self.func(f)?.args.iter().copied().find(|a| self.var_name(*a) == name)
    }

    /// `ExternalFunc::isVariadic()` (or any func).
    pub fn is_variadic(&self, f: FuncId) -> bool {
        self.var_type(f).map(|t| matches!(self.type_kind(t), TypeKind::Func { variadic: true, .. })).unwrap_or(false)
    }

    /// `InternalFunc::setIntrinsic`.
    pub fn set_intrinsic(&mut self, f: FuncId, kind: Intrinsic, ty: Option<TypeId>) {
        self.func_mut(f).expect("not a func").set_intrinsic(kind, ty);
    }

    /// `InternalFunc::hasValidIntrinsicSignature()`.
    pub fn has_valid_intrinsic_signature(&mut self, f: FuncId) -> bool {
        let Some(fd) = self.func(f) else { return false };
        let (intrinsic, intrinsic_type, generator, is_async) =
            (fd.intrinsic(), fd.intrinsic_type(), fd.generator, fd.is_async);
        let Some(sig) = self.var_type(f) else { return false };
        let (result, arguments, variadic) = match self.type_kind(sig) {
            TypeKind::Func { ret, args, variadic } => (*ret, args.clone(), *variadic),
            _ => return false,
        };
        if variadic || generator || is_async {
            return false;
        }
        let result = self.actual_type(result);
        let arguments: Vec<TypeId> = arguments.into_iter().map(|t| self.actual_type(t)).collect();
        let intrinsic_type = intrinsic_type.map(|t| self.actual_type(t));
        let int_t = self.get_int_type();
        let bool_t = self.get_bool_type();
        let is_ref = |m: &Module, t: Option<TypeId>| t.map(|t| m.type_kind(t).is_ref()).unwrap_or(false);
        let is_ptr = |m: &Module, t: Option<TypeId>| t.map(|t| m.type_kind(t).is_pointer()).unwrap_or(false);
        match intrinsic {
            Intrinsic::None => false,
            Intrinsic::Sizeof => intrinsic_type.is_some() && arguments.is_empty() && result == int_t,
            Intrinsic::IsAtomic | Intrinsic::IsContentAtomic => {
                intrinsic_type.is_some() && arguments.is_empty() && result == bool_t
            }
            Intrinsic::Allocate => {
                is_ref(self, intrinsic_type) && arguments.is_empty() && Some(result) == intrinsic_type
            }
            Intrinsic::AllocationSize => is_ref(self, intrinsic_type) && arguments.is_empty() && result == int_t,
            Intrinsic::Typeinfo => {
                intrinsic_type.is_some()
                    && arguments.is_empty()
                    && (is_ref(self, Some(result)) || is_ptr(self, Some(result)))
            }
            Intrinsic::TypeinfoLookup => {
                intrinsic_type.is_none()
                    && arguments.len() == 1
                    && arguments[0] == int_t
                    && (is_ref(self, Some(result)) || is_ptr(self, Some(result)))
            }
            Intrinsic::TypeinfoMethodLookup => {
                intrinsic_type.is_none()
                    && arguments.len() == 2
                    && (is_ref(self, Some(arguments[0])) || is_ptr(self, Some(arguments[0])))
                    && arguments[1] == int_t
                    && is_ptr(self, Some(result))
            }
            Intrinsic::VirtualLookup => {
                is_ref(self, intrinsic_type)
                    && arguments.len() == 2
                    && Some(arguments[0]) == intrinsic_type
                    && arguments[1] == int_t
                    && self.type_kind(result).is_func()
            }
            Intrinsic::RecordNew => {
                let Some(rec) = intrinsic_type else { return false };
                let TypeKind::Record { fields } = self.type_kind(rec) else { return false };
                result == rec
                    && arguments.len() == fields.len()
                    && arguments.iter().zip(fields).all(|(a, f)| *a == self.actual_type(f.ty))
            }
            Intrinsic::RepresentationCast => {
                intrinsic_type.is_some() && Some(result) == intrinsic_type && arguments.len() == 1
            }
            Intrinsic::GeneratorPromise => {
                let Some(gen_t) = intrinsic_type else { return false };
                let TypeKind::Generator { base: gbase } = self.type_kind(gen_t) else { return false };
                let TypeKind::Pointer { base: pbase } = self.type_kind(result) else { return false };
                self.actual_type(*pbase) == self.actual_type(*gbase) && arguments.len() == 1 && arguments[0] == gen_t
            }
            Intrinsic::GeneratorExcSlot => {
                let Some(gen_t) = intrinsic_type else { return false };
                matches!(self.type_kind(gen_t), TypeKind::Generator { .. })
                    && is_ptr(self, Some(result))
                    && arguments.len() == 1
                    && arguments[0] == gen_t
            }
        }
    }

    /// `InternalFunc::getIntrinsicSideEffectStatus()`.
    pub fn intrinsic_side_effect_status(&self, f: FuncId) -> SideEffectStatus {
        self.func(f).map(|fd| fd.intrinsic().side_effect_status()).unwrap_or(SideEffectStatus::Unknown)
    }
    /// `InternalFunc::intrinsicDerivesFromArguments()`.
    pub fn intrinsic_derives_from_arguments(&self, f: FuncId) -> bool {
        self.func(f).map(|fd| fd.intrinsic().derives_from_arguments()).unwrap_or(false)
    }

    // ------------------------------------------------------------------
    // used values / types / variables
    // ------------------------------------------------------------------

    /// `getUsedValues()`.
    pub fn used_values(&self, n: NodeRef) -> Vec<ValueId> {
        match self.actual(n) {
            NodeRef::Value(v) => self.values[v].kind.used_values(),
            NodeRef::Var(v) => self.vars[v].func.as_ref().and_then(|f| f.body()).into_iter().collect(),
            _ => Vec::new(),
        }
    }

    /// `replaceUsedValue(Value *old, Value *new)`.
    pub fn replace_used_value(&mut self, n: NodeRef, old: ValueId, new: impl Into<Option<ValueId>>) -> i32 {
        let uid = self.value_uid(old);
        self.replace_used_value_uid(n, uid, new.into())
    }

    /// `replaceUsedValue(id_t id, Value *new)`.
    ///
    /// Panics (after leaving the node untouched for that slot) if a flow-typed slot
    /// would receive a non-flow value, like the reference `seqassert`.
    pub fn replace_used_value_uid(&mut self, n: NodeRef, id: Id, new: Option<ValueId>) -> i32 {
        let new_is_flow = new.map(|v| self.is_flow(v)).unwrap_or(false);
        let failed = std::cell::Cell::new(false);
        let count = match self.actual(n) {
            NodeRef::Value(v) => {
                let mut kind = std::mem::replace(&mut self.values[v].kind, ValueKind::Any { flow: false });
                let count = self.replace_value_in_kind(&mut kind, id, new, new_is_flow, &failed);
                self.values[v].kind = kind;
                count
            }
            NodeRef::Var(v) => {
                let body = self.vars[v].func.as_ref().and_then(|f| f.body());
                match body {
                    Some(b) if self.value_uid(b) == id => {
                        if new_is_flow {
                            self.vars[v].func.as_mut().unwrap().set_body(new);
                            1
                        } else {
                            failed.set(true);
                            0
                        }
                    }
                    _ => 0,
                }
            }
            _ => 0,
        };
        if failed.get() {
            let desc = new.map(|v| self.reference_string(v.into())).unwrap_or_else(|| "(null)".into());
            panic!("{desc} is not a flow");
        }
        count
    }

    fn replace_value_in_kind(
        &self,
        kind: &mut ValueKind,
        id: Id,
        new: Option<ValueId>,
        new_is_flow: bool,
        failed: &std::cell::Cell<bool>,
    ) -> i32 {
        let is = |x: ValueId| self.value_uid(x) == id;
        let req = |slot: &mut ValueId, flow: bool| -> i32 {
            if is(*slot) {
                if flow && !new_is_flow {
                    failed.set(true);
                    return 0;
                }
                match new {
                    Some(nv) => {
                        *slot = nv;
                        1
                    }
                    None => {
                        failed.set(true);
                        0
                    }
                }
            } else {
                0
            }
        };
        let opt = |slot: &mut Option<ValueId>, flow: bool| -> i32 {
            if let Some(s) = *slot {
                if is(s) {
                    if flow && new.is_some() && !new_is_flow {
                        failed.set(true);
                        return 0;
                    }
                    *slot = new;
                    return 1;
                }
            }
            0
        };
        match kind {
            ValueKind::SeriesFlow(f) => {
                let mut c = 0;
                for s in f.series.iter_mut() {
                    c += req(s, false);
                }
                c
            }
            ValueKind::WhileFlow(f) => req(&mut f.cond, false) + req(&mut f.body, true),
            ValueKind::ForFlow(f) => {
                let mut c = 0;
                if let Some(s) = f.schedule.as_mut() {
                    c += s.replace_used_value(self, id, new);
                }
                c += req(&mut f.iter, false);
                c += req(&mut f.body, true);
                c
            }
            ValueKind::ImperativeForFlow(f) => {
                let mut c = 0;
                if let Some(s) = f.schedule.as_mut() {
                    c += s.replace_used_value(self, id, new);
                }
                c += req(&mut f.body, true);
                c += req(&mut f.start, false);
                c += req(&mut f.end, false);
                c
            }
            ValueKind::IfFlow(f) => {
                req(&mut f.cond, false) + req(&mut f.true_branch, true) + opt(&mut f.false_branch, true)
            }
            ValueKind::TryCatchFlow(f) => {
                let mut c = req(&mut f.body, true);
                c += opt(&mut f.else_, true);
                c += opt(&mut f.finally, true);
                for cat in f.catches.iter_mut() {
                    c += req(&mut cat.handler, true);
                }
                c
            }
            ValueKind::PipelineFlow(f) => {
                let mut c = 0;
                for s in f.stages.iter_mut() {
                    c += req(&mut s.callee, false);
                    for a in s.args.iter_mut() {
                        c += opt(a, false);
                    }
                }
                c
            }
            ValueKind::AssignInstr(i) => req(&mut i.rhs, false),
            ValueKind::ExtractInstr(i) => req(&mut i.val, false),
            ValueKind::InsertInstr(i) => req(&mut i.lhs, false) + req(&mut i.rhs, false),
            ValueKind::CallInstr(i) => {
                let mut c = req(&mut i.callee, false);
                for a in i.args.iter_mut() {
                    c += req(a, false);
                }
                c
            }
            ValueKind::TernaryInstr(i) => {
                req(&mut i.cond, false) + req(&mut i.true_value, false) + req(&mut i.false_value, false)
            }
            ValueKind::BreakInstr(i) => opt(&mut i.loop_, true),
            ValueKind::ContinueInstr(i) => opt(&mut i.loop_, true),
            ValueKind::ReturnInstr(i) => opt(&mut i.value, false),
            ValueKind::YieldInstr(i) => opt(&mut i.value, false),
            ValueKind::AwaitInstr(i) => req(&mut i.value, false),
            ValueKind::ThrowInstr(i) => opt(&mut i.value, false) + opt(&mut i.fallback, false),
            ValueKind::FlowInstr(i) => req(&mut i.flow, true) + req(&mut i.val, false),
            _ => 0,
        }
    }

    /// `getUsedTypes()`.
    pub fn used_types(&self, n: NodeRef) -> Vec<TypeId> {
        match self.actual(n) {
            NodeRef::Value(v) => self.values[v].kind.used_types(),
            NodeRef::Var(v) => {
                let d = &self.vars[v];
                let mut ret: Vec<TypeId> = d.ty.into_iter().collect();
                if let Some(f) = &d.func {
                    ret.extend(f.parent_type);
                    match &f.kind {
                        FuncKind::Internal { intrinsic_type: Some(t), .. } => ret.push(*t),
                        FuncKind::Llvm { literals, .. } => ret.extend(literals.iter().filter_map(|l| l.type_value())),
                        _ => {}
                    }
                }
                ret
            }
            NodeRef::Type(t) => {
                let d = &self.types[t];
                let mut result: Vec<TypeId> = match &d.kind {
                    TypeKind::Record { fields } => fields.iter().map(|f| f.ty).collect(),
                    TypeKind::Ref { contents, .. } => vec![*contents],
                    TypeKind::Func { ret, args, .. } => {
                        let mut r = args.clone();
                        r.push(*ret);
                        r
                    }
                    TypeKind::Pointer { base } | TypeKind::Optional { base } | TypeKind::Generator { base } => {
                        vec![*base]
                    }
                    TypeKind::Union { types } => types.clone(),
                    _ => Vec::new(),
                };
                if let Some(info) = &d.runtime_info {
                    let mut append = |ty: Option<TypeId>| {
                        if let Some(ty) = ty {
                            if self.actual_type(ty) != t && !result.contains(&ty) {
                                result.push(ty);
                            }
                        }
                    };
                    for x in &info.mro {
                        append(Some(*x));
                    }
                    for x in &info.field_owners {
                        append(Some(*x));
                    }
                    for x in &info.parameters {
                        append(Some(*x));
                    }
                    append(info.exception_base);
                    append(info.system_exit_base);
                }
                result
            }
            NodeRef::Module => Vec::new(),
        }
    }

    /// `replaceUsedType(Type *old, Type *new)`.
    pub fn replace_used_type(&mut self, n: NodeRef, old: TypeId, new: TypeId) -> i32 {
        let name = self.type_name(old).to_string();
        self.replace_used_type_name(n, &name, new)
    }

    /// `replaceUsedType(const std::string &name, Type *new)`.
    pub fn replace_used_type_name(&mut self, n: NodeRef, name: &str, new: TypeId) -> i32 {
        let matches = |m: &Module, t: TypeId| m.type_name(t) == name;
        match self.actual(n) {
            NodeRef::Value(v) => {
                let mut kind = std::mem::replace(&mut self.values[v].kind, ValueKind::Any { flow: false });
                let rep = |slot: &mut TypeId| -> i32 {
                    if matches(self, *slot) {
                        *slot = new;
                        1
                    } else {
                        0
                    }
                };
                let c = match &mut kind {
                    ValueKind::IntConst(c) => rep(&mut c.ty),
                    ValueKind::FloatConst(c) => rep(&mut c.ty),
                    ValueKind::BoolConst(c) => rep(&mut c.ty),
                    ValueKind::StringConst(c) | ValueKind::BytesConst(c) => rep(&mut c.ty),
                    ValueKind::TryCatchFlow(f) => {
                        let mut c = 0;
                        for cat in f.catches.iter_mut() {
                            if let Some(t) = cat.ty.as_mut() {
                                c += rep(t);
                            }
                        }
                        c
                    }
                    ValueKind::StackAllocInstr(i) => rep(&mut i.ptr_type),
                    ValueKind::YieldInInstr(i) => rep(&mut i.ty),
                    ValueKind::AwaitInstr(i) => rep(&mut i.ty),
                    _ => 0,
                };
                self.values[v].kind = kind;
                c
            }
            NodeRef::Var(v) => {
                let ty = self.vars[v].ty;
                let mut count = 0;
                if let Some(t) = ty {
                    if matches(self, t) {
                        self.vars[v].ty = Some(new);
                        count += 1;
                    }
                }
                let Some(mut f) = self.vars[v].func.take() else { return count };
                if let Some(p) = f.parent_type {
                    if matches(self, p) {
                        f.parent_type = Some(new);
                        count += 1;
                    }
                }
                match &mut f.kind {
                    FuncKind::Internal { intrinsic_type: Some(t), .. } => {
                        if matches(self, *t) {
                            *t = new;
                            count += 1;
                        }
                    }
                    FuncKind::Llvm { literals, .. } => {
                        for l in literals.iter_mut() {
                            if let Generic::Type(t) = l {
                                if matches(self, *t) {
                                    *l = Generic::Type(new);
                                    count += 1;
                                }
                            }
                        }
                    }
                    _ => {}
                }
                self.vars[v].func = Some(f);
                count
            }
            NodeRef::Type(_) => panic!("types not replaceable"),
            NodeRef::Module => 0,
        }
    }

    /// `getUsedVariables()`.
    pub fn used_variables(&self, n: NodeRef) -> Vec<VarId> {
        match self.actual(n) {
            NodeRef::Value(v) => self.values[v].kind.used_variables(),
            NodeRef::Var(v) => match &self.vars[v].func {
                Some(f) => {
                    let mut ret = f.args.clone();
                    ret.extend_from_slice(f.symbols());
                    ret
                }
                None => Vec::new(),
            },
            NodeRef::Type(t) => self.types[t]
                .runtime_info
                .as_ref()
                .map(|i| i.methods.values().copied().collect())
                .unwrap_or_default(),
            NodeRef::Module => Vec::new(),
        }
    }

    /// `replaceUsedVariable(Var *old, Var *new)`.
    pub fn replace_used_variable(&mut self, n: NodeRef, old: VarId, new: impl Into<Option<VarId>>) -> i32 {
        let uid = self.var_uid(old);
        self.replace_used_variable_uid(n, uid, new.into())
    }

    /// `replaceUsedVariable(id_t id, Var *new)`.
    pub fn replace_used_variable_uid(&mut self, n: NodeRef, id: Id, new: Option<VarId>) -> i32 {
        let is = |m: &Module, x: VarId| m.var_uid(x) == id;
        match self.actual(n) {
            NodeRef::Value(v) => {
                let mut kind = std::mem::replace(&mut self.values[v].kind, ValueKind::Any { flow: false });
                let req = |m: &Module, slot: &mut VarId| -> i32 {
                    if is(m, *slot) {
                        *slot = new.expect("cannot replace a variable with null");
                        1
                    } else {
                        0
                    }
                };
                let c = match &mut kind {
                    ValueKind::VarValue(x) => req(self, &mut x.var),
                    ValueKind::PointerValue(x) => req(self, &mut x.var),
                    ValueKind::ForFlow(f) => req(self, &mut f.var),
                    ValueKind::ImperativeForFlow(f) => req(self, &mut f.var),
                    ValueKind::AssignInstr(i) => req(self, &mut i.lhs),
                    ValueKind::TryCatchFlow(f) => {
                        let mut c = 0;
                        for cat in f.catches.iter_mut() {
                            if let Some(x) = cat.var {
                                if is(self, x) {
                                    cat.var = new;
                                    c += 1;
                                }
                            }
                        }
                        c
                    }
                    _ => 0,
                };
                self.values[v].kind = kind;
                c
            }
            NodeRef::Var(v) => {
                let Some(mut f) = self.vars[v].func.take() else { return 0 };
                let mut c = 0;
                let mut rep_list = |m: &Module, list: &mut Vec<VarId>| {
                    for a in list.iter_mut() {
                        if is(m, *a) {
                            *a = new.expect("cannot replace a variable with null");
                            c += 1;
                        }
                    }
                };
                rep_list(self, &mut f.args);
                if let Some(s) = f.symbols_mut() {
                    rep_list(self, s);
                }
                self.vars[v].func = Some(f);
                c
            }
            NodeRef::Type(t) => {
                if let Some(nv) = new {
                    assert!(self.is_func(nv), "runtime method target must be a function");
                }
                let Some(mut info) = self.types[t].runtime_info.take() else { return 0 };
                let mut count = 0;
                let keys: Vec<usize> = info.methods.keys().copied().collect();
                for k in keys {
                    let f = info.methods[&k];
                    if self.var_uid(f) != id {
                        continue;
                    }
                    count += 1;
                    match new {
                        Some(nv) => {
                            info.methods.insert(k, nv);
                        }
                        None => {
                            info.methods.remove(&k);
                        }
                    }
                }
                self.types[t].runtime_info = Some(info);
                count
            }
            NodeRef::Module => 0,
        }
    }

    // ------------------------------------------------------------------
    // types of values
    // ------------------------------------------------------------------

    /// Cached none type, if already obtained through [`Module::get_none_type`].
    pub fn none_type_cached(&self) -> Option<TypeId> {
        self.none_type
    }

    fn try_value_type(&self, v: ValueId) -> Result<TypeId, MissingType> {
        let v = self.actual_value(v);
        let none = || {
            self.none_type
                .filter(|t| self.types[*t].alive)
                .or_else(|| self.get_type(&typhon_common::mangle::stdlib().none_type))
                .ok_or(MissingType::None)
        };
        Ok(match &self.values[v].kind {
            ValueKind::IntConst(c) => c.ty,
            ValueKind::FloatConst(c) => c.ty,
            ValueKind::BoolConst(c) => c.ty,
            ValueKind::StringConst(c) | ValueKind::BytesConst(c) => c.ty,
            ValueKind::VarValue(x) => self.var_type(x.var).expect("variable has no type"),
            ValueKind::PointerValue(x) => {
                let base = self.var_type(x.var).expect("variable has no type");
                let name = pointer_type_name(self.type_name(base));
                self.get_type(&name).ok_or(MissingType::Pointer(base))?
            }
            ValueKind::ExtractInstr(i) => {
                let t = self.try_value_type(i.val)?;
                assert!(self.type_kind(t).is_membered(), "{} is not a membered type", self.type_name(t));
                self.member_type(t, &i.field).expect("member not found")
            }
            ValueKind::InsertInstr(i) => self.try_value_type(i.lhs)?,
            ValueKind::CallInstr(i) => {
                let t = self.try_value_type(i.callee)?;
                match self.type_kind(t) {
                    TypeKind::Func { ret, .. } => *ret,
                    _ => panic!("{} is not a function type", self.type_name(t)),
                }
            }
            ValueKind::StackAllocInstr(i) => i.ptr_type,
            ValueKind::YieldInInstr(i) => i.ty,
            ValueKind::TernaryInstr(i) => self.try_value_type(i.true_value)?,
            ValueKind::AwaitInstr(i) => i.ty,
            ValueKind::FlowInstr(i) => self.try_value_type(i.val)?,
            ValueKind::Any { flow: false } => {
                self.get_type(&int_type_name(64, true)).ok_or(MissingType::Int)?
            }
            _ => none()?,
        })
    }

    /// `Value::getType()` on a shared module. Panics if the result type would have to be
    /// created (the none type before [`Module::get_none_type`] was called, or a pointer
    /// type that does not exist yet); use [`Module::value_type_mut`] while building IR.
    pub fn value_type(&self, v: ValueId) -> TypeId {
        match self.try_value_type(v) {
            Ok(t) => t,
            Err(e) => panic!("type of value {} is not yet available ({e:?})", self.reference_string(v.into())),
        }
    }

    /// `Value::getType()`: creates the none / pointer type if needed.
    pub fn value_type_mut(&mut self, v: ValueId) -> TypeId {
        loop {
            match self.try_value_type(v) {
                Ok(t) => return t,
                Err(MissingType::None) => {
                    self.get_none_type();
                }
                Err(MissingType::Pointer(b)) => {
                    self.get_pointer_type(Some(b));
                }
                Err(MissingType::Int) => {
                    self.get_int_type();
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // type behaviour
    // ------------------------------------------------------------------

    /// `Type::is(other)`: name equality.
    pub fn type_is(&self, a: TypeId, b: TypeId) -> bool {
        self.type_name(a) == self.type_name(b)
    }

    /// `Type::isAtomic()`.
    pub fn type_is_atomic(&self, t: TypeId) -> bool {
        match self.type_kind(t) {
            k if k.is_primitive() => true,
            TypeKind::Record { fields } => fields.iter().all(|f| self.type_is_atomic(f.ty)),
            TypeKind::Ref { .. } | TypeKind::Func { .. } | TypeKind::Pointer { .. } | TypeKind::Generator { .. } => {
                false
            }
            TypeKind::Optional { base } => self.type_is_atomic(*base),
            TypeKind::Union { types } => types.iter().all(|x| self.type_is_atomic(*x)),
            _ => false,
        }
    }

    /// `Type::isContentAtomic()`.
    pub fn type_is_content_atomic(&self, t: TypeId) -> bool {
        match self.type_kind(t) {
            TypeKind::Ref { contents, .. } => self.type_fields(*contents).iter().all(|f| self.type_is_atomic(f.ty)),
            _ => true,
        }
    }

    /// Fields of a membered type (`MemberedType::begin()/end()`).
    pub fn type_fields(&self, t: TypeId) -> Vec<Field> {
        match self.type_kind(t) {
            TypeKind::Record { fields } => fields.clone(),
            TypeKind::Ref { contents, .. } => self.type_fields(*contents),
            _ => Vec::new(),
        }
    }

    /// `RefType::getContents()`.
    pub fn ref_contents(&self, t: TypeId) -> Option<TypeId> {
        match self.type_kind(t) {
            TypeKind::Ref { contents, .. } => Some(*contents),
            _ => None,
        }
    }

    /// `MemberedType::getMemberType(name)`.
    pub fn member_type(&self, t: TypeId, name: &str) -> Option<TypeId> {
        match self.type_kind(t) {
            TypeKind::Record { fields } => fields.iter().find(|f| f.name == name).map(|f| f.ty),
            TypeKind::Ref { contents, .. } => self.member_type(*contents, name),
            _ => None,
        }
    }

    /// `MemberedType::getMemberIndex(name)` (-1 if absent).
    pub fn member_index(&self, t: TypeId, name: &str) -> i32 {
        match self.type_kind(t) {
            TypeKind::Record { fields } => fields.iter().position(|f| f.name == name).map(|i| i as i32).unwrap_or(-1),
            TypeKind::Ref { contents, .. } => self.member_index(*contents, name),
            _ => -1,
        }
    }

    /// `MemberedType::realize(types, names)`.
    pub fn realize_membered(&mut self, t: TypeId, types: Vec<TypeId>, names: Vec<String>) {
        let target = match self.type_kind(t) {
            TypeKind::Ref { contents, .. } => *contents,
            TypeKind::Record { .. } => t,
            _ => panic!("{} is not a membered type", self.type_name(t)),
        };
        let fields = types.into_iter().zip(names).map(|(t, n)| Field::new(n, t)).collect();
        if let TypeKind::Record { fields: f } = &mut self.ty_mut(target).kind {
            *f = fields;
        }
    }

    /// `FuncType::getReturnType()`.
    pub fn func_type_ret(&self, t: TypeId) -> Option<TypeId> {
        match self.type_kind(t) {
            TypeKind::Func { ret, .. } => Some(*ret),
            _ => None,
        }
    }
    /// `FuncType` argument types.
    pub fn func_type_args(&self, t: TypeId) -> Vec<TypeId> {
        match self.type_kind(t) {
            TypeKind::Func { args, .. } => args.clone(),
            _ => Vec::new(),
        }
    }
    /// `Type::setRuntimeInfo`.
    pub fn set_runtime_info(&mut self, t: TypeId, info: RuntimeInfo) {
        self.ty_mut(t).runtime_info = Some(Box::new(info));
    }
    /// `Type::getRuntimeInfo`.
    pub fn runtime_info(&self, t: TypeId) -> Option<&RuntimeInfo> {
        self.ty(t).runtime_info.as_deref()
    }
    /// `Type::getAstType`.
    pub fn ast_type(&self, t: TypeId) -> Option<AstTypeRef> {
        self.ty(t).ast_type.clone()
    }
    /// `Type::setAstType`.
    pub fn set_ast_type(&mut self, t: TypeId, a: Option<AstTypeRef>) {
        self.ty_mut(t).ast_type = a;
    }

    /// `Type::getGenerics()` (through the realizer).
    pub fn type_generics(&mut self, t: TypeId) -> Vec<Generic> {
        let Some(ast) = self.ast_type(t) else { return Vec::new() };
        let is_func = self.type_kind(t).is_func();
        let Some(r) = self.realizer.clone() else { return Vec::new() };
        let Ok(mut r) = r.try_borrow_mut() else { return Vec::new() };
        r.generics(self, &ast, is_func)
    }

    /// `Type::construct(args)` / `Type::operator()(args...)`.
    pub fn construct(&mut self, t: TypeId, args: Vec<ValueId>) -> Option<ValueId> {
        if self.type_kind(t).is_ref() {
            let args_tuple = crate::util::irtools::make_tuple(self, &args)?;
            let tt = self.value_type_mut(args_tuple);
            let f = self.get_or_realize_func("construct_ref", &[tt], &[Generic::Type(t)], "std.internal.gc")?;
            let fv = self.nr_var_value(f);
            return Some(self.nr_call(fv, vec![args_tuple]));
        }
        let arg_types: Vec<TypeId> = args.iter().map(|a| self.value_type_mut(*a)).collect();
        let f = self.get_or_realize_method(t, magic::NEW, &arg_types, &[])?;
        let fv = self.nr_var_value(f);
        Some(self.nr_call(fv, args))
    }

    // ------------------------------------------------------------------
    // realization through the parser cache
    // ------------------------------------------------------------------

    fn log_parser_error(what: &str, e: &typhon_common::CompilerError) {
        if let Some(errs) = e.parser_errors() {
            for trace in &errs.errors {
                for msg in &trace.trace {
                    log_flag!(IR, "{} parser error at {}: {}", what, msg.loc, msg.msg);
                }
            }
        } else {
            log_flag!(IR, "{} error: {}", what, e);
        }
    }

    /// `getOrRealizeMethod(parent, name, args, generics)`.
    pub fn get_or_realize_method(
        &mut self,
        parent: TypeId,
        method_name: &str,
        args: &[TypeId],
        generics: &[Generic],
    ) -> Option<FuncId> {
        let r = self.realizer.clone()?;
        let cls = self.ast_type(parent)?;
        let mut r = r.try_borrow_mut().ok()?;
        let method = r.find_method(self, &cls, method_name, args)?;
        match r.realize_function(self, &method, args, generics, Some(&cls)) {
            Ok(f) => f,
            Err(e) => {
                Self::log_parser_error("getOrRealizeMethod", &e);
                None
            }
        }
    }

    /// `getOrRealizeFunc(name, args, generics, module)`.
    pub fn get_or_realize_func(
        &mut self,
        func_name: &str,
        args: &[TypeId],
        generics: &[Generic],
        module: &str,
    ) -> Option<FuncId> {
        let r = self.realizer.clone()?;
        let fq = if module.is_empty() { func_name.to_string() } else { format!("{module}.{func_name}") };
        let mut r = r.try_borrow_mut().ok()?;
        let func = r.find_function(&fq).or_else(|| r.find_function(&format!("{fq}.0")))?;
        match r.realize_function(self, &func, args, generics, None) {
            Ok(f) => f,
            Err(e) => {
                Self::log_parser_error("getOrRealizeFunc", &e);
                None
            }
        }
    }

    /// `getOrRealizeType(name, generics)`.
    pub fn get_or_realize_type(&mut self, type_name: &str, generics: &[Generic]) -> Option<TypeId> {
        let r = self.realizer.clone()?;
        let mut r = r.try_borrow_mut().ok()?;
        let ty = r.find_class(type_name)?;
        match r.realize_type(self, &ty, generics) {
            Ok(t) => t,
            Err(e) => {
                Self::log_parser_error("getOrRealizeType", &e);
                None
            }
        }
    }

    /// `parseCode(code)`.
    pub fn parse_code(&mut self, code: &str) -> typhon_common::Result<()> {
        let Some(r) = self.realizer.clone() else {
            return Err(typhon_common::CompilerError::custom("no parser cache attached", SrcInfo::default()));
        };
        let mut r = r.try_borrow_mut().map_err(|_| {
            typhon_common::CompilerError::custom("parser cache already borrowed", SrcInfo::default())
        })?;
        r.parse_code(self, code)
    }

    // ------------------------------------------------------------------
    // type factories
    // ------------------------------------------------------------------

    fn get_or_add(&mut self, name: &str, kind: impl FnOnce() -> TypeKind) -> TypeId {
        if let Some(t) = self.get_type(name) {
            return t;
        }
        let k = kind();
        self.add_type(name, k)
    }

    pub fn get_bool_type(&mut self) -> TypeId {
        self.get_or_add(magic::BOOL_NAME, || TypeKind::Bool)
    }
    pub fn get_float_type(&mut self) -> TypeId {
        self.get_or_add(magic::FLOAT_NAME, || TypeKind::Float)
    }
    pub fn get_float32_type(&mut self) -> TypeId {
        self.get_or_add(magic::FLOAT32_NAME, || TypeKind::Float32)
    }
    pub fn get_float16_type(&mut self) -> TypeId {
        self.get_or_add(magic::FLOAT16_NAME, || TypeKind::Float16)
    }
    pub fn get_bfloat16_type(&mut self) -> TypeId {
        self.get_or_add(magic::BFLOAT16_NAME, || TypeKind::BFloat16)
    }
    pub fn get_float128_type(&mut self) -> TypeId {
        self.get_or_add(magic::FLOAT128_NAME, || TypeKind::Float128)
    }
    pub fn get_string_type(&mut self) -> TypeId {
        if let Some(t) = self.get_type(magic::STRING_NAME) {
            return t;
        }
        let p = self.unsafe_get_pointer_type(None);
        let i = self.unsafe_get_int_type(64, true);
        self.nr_record_type(magic::STRING_NAME, vec![p, i], vec!["_ptr".into(), "_meta".into()])
    }
    pub fn get_bytes_type(&mut self) -> TypeId {
        if let Some(t) = self.get_type(magic::BYTES_NAME) {
            return t;
        }
        let p = self.unsafe_get_pointer_type(None);
        let i = self.unsafe_get_int_type(64, true);
        self.nr_record_type(magic::BYTES_NAME, vec![p, i], vec!["_ptr".into(), "_len".into()])
    }
    /// `getIntType()` (64-bit signed).
    pub fn get_int_type(&mut self) -> TypeId {
        self.unsafe_get_int_type(64, true)
    }
    /// `getIntType(len, sign)`.
    pub fn get_int_type_n(&mut self, len: u32, sign: bool) -> TypeId {
        if len == 64 && sign {
            return self.unsafe_get_int_type(len, sign);
        }
        let name = if sign { &typhon_common::mangle::stdlib().int } else { &typhon_common::mangle::stdlib().uint };
        let name = name.clone();
        if self.realizer.is_some() {
            if let Some(t) = self.get_or_realize_type(&name, &[Generic::Static(len as i64)]) {
                return t;
            }
        }
        // Note: no cache attached (unit tests): use the structural type.
        self.unsafe_get_int_type(len, sign)
    }
    /// `getPointerType(base)` (base defaults to `UInt[8]`).
    pub fn get_pointer_type(&mut self, base: Option<TypeId>) -> TypeId {
        let base = match base {
            Some(b) => b,
            None => self.get_int_type_n(8, false),
        };
        let name = typhon_common::mangle::stdlib().ptr.clone();
        if self.realizer.is_some() {
            if let Some(t) = self.get_or_realize_type(&name, &[Generic::Type(base)]) {
                return t;
            }
        }
        self.unsafe_get_pointer_type(Some(base))
    }
    pub fn get_generator_type(&mut self, base: TypeId) -> TypeId {
        let name = typhon_common::mangle::stdlib().generator.clone();
        if self.realizer.is_some() {
            if let Some(t) = self.get_or_realize_type(&name, &[Generic::Type(base)]) {
                return t;
            }
        }
        self.unsafe_get_generator_type(base)
    }
    pub fn get_optional_type(&mut self, base: TypeId) -> TypeId {
        let name = typhon_common::mangle::stdlib().optional.clone();
        if self.realizer.is_some() {
            if let Some(t) = self.get_or_realize_type(&name, &[Generic::Type(base)]) {
                return t;
            }
        }
        self.unsafe_get_optional_type(base)
    }
    pub fn get_vector_type(&mut self, count: u32, base: TypeId) -> TypeId {
        let name = typhon_common::mangle::stdlib().vec.clone();
        if self.realizer.is_some() {
            if let Some(t) = self.get_or_realize_type(&name, &[Generic::Type(base), Generic::Static(count as i64)]) {
                return t;
            }
        }
        self.unsafe_get_vector_type(count, base)
    }
    /// `getNoneType()`.
    pub fn get_none_type(&mut self) -> TypeId {
        if let Some(t) = self.none_type {
            if self.types[t].alive {
                return t;
            }
        }
        let name = typhon_common::mangle::stdlib().none_type.clone();
        let t = if self.realizer.is_some() { self.get_or_realize_type(&name, &[]) } else { None };
        let t = match t {
            Some(t) => t,
            None => self.unsafe_get_membered_type(&name, false),
        };
        self.none_type = Some(t);
        t
    }
    /// `getFuncType(ret, args, variadic)`.
    pub fn get_func_type(&mut self, ret: TypeId, args: Vec<TypeId>, variadic: bool) -> TypeId {
        let made = match self.realizer.clone() {
            Some(r) => match r.try_borrow_mut() {
                Ok(mut r) => match r.make_function(self, ret, &args) {
                    Ok(t) => Some(t),
                    Err(e) => {
                        Self::log_parser_error("getFuncType", &e);
                        None
                    }
                },
                Err(_) => None,
            },
            None => None,
        };
        let result = match made {
            Some(t) => t,
            None => {
                let names: Vec<String> = args.iter().map(|a| self.type_name(*a).to_string()).collect();
                let name = format!("Function[[{}],{}]", names.join(","), self.type_name(ret));
                self.unsafe_get_func_type(&name, ret, args.clone(), false)
            }
        };
        if variadic {
            let name = format!("{}$variadic", self.type_name(result));
            let (r, a) = match self.type_kind(result) {
                TypeKind::Func { ret, args, .. } => (*ret, args.clone()),
                _ => (ret, args),
            };
            let ast = self.ast_type(result);
            let v = self.unsafe_get_func_type(&name, r, a, true);
            self.set_ast_type(v, ast);
            return v;
        }
        result
    }
    /// `getTupleType(types)`.
    pub fn get_tuple_type(&mut self, args: Vec<TypeId>) -> TypeId {
        if let Some(r) = self.realizer.clone() {
            if let Ok(mut r) = r.try_borrow_mut() {
                match r.make_tuple(self, &args) {
                    Ok(t) => return t,
                    Err(e) => Self::log_parser_error("getTupleType", &e),
                }
            }
        }
        // Note: structural fallback without a parser cache.
        let names: Vec<String> = args.iter().map(|a| self.type_name(*a).to_string()).collect();
        let name = format!("Tuple[{}]", names.join(","));
        if let Some(t) = self.get_type(&name) {
            return t;
        }
        let fields = args.iter().enumerate().map(|(i, _)| format!("item{}", i + 1)).collect();
        self.nr_record_type(name, args, fields)
    }
    /// `getUnionType(types)`.
    pub fn get_union_type(&mut self, types: Vec<TypeId>) -> TypeId {
        if let Some(r) = self.realizer.clone() {
            if let Ok(mut r) = r.try_borrow_mut() {
                match r.make_union(self, &types) {
                    Ok(t) => return t,
                    Err(e) => Self::log_parser_error("getUnionType", &e),
                }
            }
        }
        self.unsafe_get_union_type(&types)
    }

    /// `getIntrinsic(kind, type, signature)`.
    pub fn get_intrinsic(&mut self, kind: Intrinsic, ty: Option<TypeId>, signature: Option<TypeId>) -> FuncId {
        let signature = match signature {
            Some(s) => s,
            None => {
                assert!(kind.is_type_query(), "intrinsic requires an explicit signature");
                let result = if kind == Intrinsic::Sizeof { self.get_int_type() } else { self.get_bool_type() };
                let name = format!(".intrinsic.query.{}", self.type_name(result));
                self.unsafe_get_func_type(&name, result, vec![], false)
            }
        };
        let key = (
            kind,
            ty.map(|t| self.type_name(t).to_string()).unwrap_or_default(),
            self.type_name(signature).to_string(),
        );
        if let Some(uid) = self.intrinsics.get(&key).copied() {
            if let Some(f) = self.get_var(uid) {
                let ok = self.func(f).map(|fd| {
                    fd.intrinsic() == kind
                        && fd.intrinsic_type().map(|t| self.actual_type(t)) == ty.map(|t| self.actual_type(t))
                }) == Some(true)
                    && self.var_type(f).map(|t| self.actual_type(t)) == Some(self.actual_type(signature));
                if ok && self.has_valid_intrinsic_signature(f) {
                    return f;
                }
            }
        }
        let nargs = match self.type_kind(signature) {
            TypeKind::Func { args, .. } => args.len(),
            _ => panic!("intrinsic signature is not a function type"),
        };
        let mut name = format!(".intrinsic.{}", kind.name());
        for component in [&key.1, &key.2] {
            name += &format!(".{}:{}", component.len(), component);
        }
        let f = self.nr_internal_func(name);
        self.set_intrinsic(f, kind, ty);
        self.set_global(f, true);
        let names: Vec<String> = (0..nargs).map(|i| format!("arg{i}")).collect();
        self.realize_func(f, signature, &names);
        assert!(self.has_valid_intrinsic_signature(f), "invalid intrinsic signature: {}", kind.name());
        let uid = self.var_uid(f);
        self.intrinsics.insert(key, uid);
        f
    }

    pub fn unsafe_get_pointer_type(&mut self, base: Option<TypeId>) -> TypeId {
        let base = match base {
            Some(b) => b,
            None => self.unsafe_get_int_type(8, false),
        };
        let name = pointer_type_name(self.type_name(base));
        self.get_or_add(&name, || TypeKind::Pointer { base })
    }
    pub fn unsafe_get_generator_type(&mut self, base: TypeId) -> TypeId {
        let name = generator_type_name(self.type_name(base));
        self.get_or_add(&name, || TypeKind::Generator { base })
    }
    pub fn unsafe_get_optional_type(&mut self, base: TypeId) -> TypeId {
        let name = optional_type_name(self.type_name(base));
        self.get_or_add(&name, || TypeKind::Optional { base })
    }
    pub fn unsafe_get_func_type(&mut self, name: &str, ret: TypeId, args: Vec<TypeId>, variadic: bool) -> TypeId {
        self.get_or_add(name, || TypeKind::Func { ret, args, variadic })
    }
    pub fn unsafe_get_membered_type(&mut self, name: &str, is_ref: bool) -> TypeId {
        if let Some(t) = self.get_type(name) {
            return t;
        }
        if is_ref {
            let content_name = format!("{name}.contents");
            let record = match self.get_type(&content_name) {
                Some(r) => r,
                None => self.add_type(content_name, TypeKind::Record { fields: Vec::new() }),
            };
            self.add_type(name, TypeKind::Ref { contents: record, polymorphic: false })
        } else {
            self.add_type(name, TypeKind::Record { fields: Vec::new() })
        }
    }
    pub fn unsafe_get_int_type(&mut self, len: u32, sign: bool) -> TypeId {
        let name = int_type_name(len, sign);
        self.get_or_add(&name, || TypeKind::Int { len, sign })
    }
    pub fn unsafe_get_vector_type(&mut self, count: u32, base: TypeId) -> TypeId {
        assert!(self.type_kind(base).is_primitive(), "base type must be a primitive type");
        assert!(count > 0, "vector width must be positive");
        let name = vector_type_name(count, self.type_name(base));
        self.get_or_add(&name, || TypeKind::Vector { count, base })
    }
    pub fn unsafe_get_union_type(&mut self, types: &[TypeId]) -> TypeId {
        let names: Vec<String> = types.iter().map(|t| self.type_name(*t).to_string()).collect();
        let name = union_type_name(&names);
        let types = types.to_vec();
        self.get_or_add(&name, || TypeKind::Union { types })
    }

    // ---- constants -----------------------------------------------------

    pub fn get_int(&mut self, v: i64) -> ValueId {
        let t = self.get_int_type();
        self.nr_int_const(v, t)
    }
    pub fn get_float(&mut self, v: f64) -> ValueId {
        let t = self.get_float_type();
        self.nr_float_const(v, t)
    }
    pub fn get_bool(&mut self, v: bool) -> ValueId {
        let t = self.get_bool_type();
        self.nr_bool_const(v, t)
    }
    pub fn get_string(&mut self, v: impl Into<String>) -> ValueId {
        let t = self.get_string_type();
        self.nr_string_const(v, t)
    }
    pub fn get_bytes(&mut self, v: impl Into<String>) -> ValueId {
        let t = self.get_bytes_type();
        self.nr_bytes_const(v, t)
    }

    // ------------------------------------------------------------------
    // Value operators
    // ------------------------------------------------------------------

    /// `Value::doUnaryOp(name)`.
    pub fn unary_op(&mut self, v: ValueId, name: &str) -> Option<ValueId> {
        let t = self.value_type_mut(v);
        let f = self.get_or_realize_method(t, name, &[t], &[])?;
        let fv = self.nr_var_value(f);
        Some(self.nr_call(fv, vec![v]))
    }
    /// `Value::doBinaryOp(name, other)`.
    pub fn binary_op(&mut self, v: ValueId, name: &str, other: ValueId) -> Option<ValueId> {
        let t = self.value_type_mut(v);
        let o = self.value_type_mut(other);
        let f = self.get_or_realize_method(t, name, &[t, o], &[])?;
        let fv = self.nr_var_value(f);
        Some(self.nr_call(fv, vec![v, other]))
    }
    /// `Value::operator()(args...)`.
    pub fn call_value(&mut self, v: ValueId, args: Vec<ValueId>) -> ValueId {
        self.nr_call(v, args)
    }
    /// `Value::operator||`.
    pub fn op_or(&mut self, v: ValueId, other: ValueId) -> Option<ValueId> {
        let a = self.to_bool(v)?;
        let t = self.get_bool(true);
        let b = self.to_bool(other)?;
        Some(self.nr_ternary(a, t, b))
    }
    /// `Value::operator&&`.
    pub fn op_and(&mut self, v: ValueId, other: ValueId) -> Option<ValueId> {
        let a = self.to_bool(v)?;
        let b = self.to_bool(other)?;
        let f = self.get_bool(false);
        Some(self.nr_ternary(a, b, f))
    }
    pub fn to_int(&mut self, v: ValueId) -> Option<ValueId> {
        self.unary_op(v, magic::INT)
    }
    pub fn to_float(&mut self, v: ValueId) -> Option<ValueId> {
        self.unary_op(v, magic::FLOAT)
    }
    pub fn to_bool(&mut self, v: ValueId) -> Option<ValueId> {
        self.unary_op(v, magic::BOOL)
    }
    /// `Value::toStr()` (uses `__repr__`).
    pub fn to_str(&mut self, v: ValueId) -> Option<ValueId> {
        self.unary_op(v, magic::REPR)
    }
    pub fn len(&mut self, v: ValueId) -> Option<ValueId> {
        self.unary_op(v, magic::LEN)
    }
    pub fn iter(&mut self, v: ValueId) -> Option<ValueId> {
        self.unary_op(v, magic::ITER)
    }
    /// `Value::operator[]`.
    pub fn getitem(&mut self, v: ValueId, other: ValueId) -> Option<ValueId> {
        self.binary_op(v, magic::GETITEM, other)
    }
}

#[derive(Debug)]
enum MissingType {
    None,
    Pointer(TypeId),
    Int,
}
