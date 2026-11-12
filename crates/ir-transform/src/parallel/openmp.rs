//! `OpenMPPass` lowers parallel `ForFlow`s / `ImperativeForFlow`s (those carrying an
//! `OMPSched`) into outlined bodies plus calls into the `std.openmp` (and
//! `std.internal.gpu`) loop templates, wiring up shared/private captures, reductions,
//! schedules, chunking, collapse, task loops and GPU kernels.
//!
//! Note: full lowering requires a `TypeRealizer` (stdlib templates are realized
//! through the parser); it is validated by the e2e suite (`test/core/` parallel tests,
//! `test/stdlib/openmp*`). Unit tests cover reduction detection, initial values,
//! template selection and collapse preconditions on hand-built modules.

use std::cell::Cell;

use rustc_hash::{FxHashMap, FxHashSet};
use typhon_common::diag::{MessageGroupPos, compilation_warning};
use typhon_common::mangle;
use typhon_ir::ids::{FuncId, Id, NodeRef, TypeId, ValueId, VarId};
use typhon_ir::module::{Module, magic};
use typhon_ir::pass::{Pass, PassContext};
use typhon_ir::schedule::OMPSched;
use typhon_ir::types::{Generic, TypeKind};
use typhon_ir::util::cloning::CloneVisitor;
use typhon_ir::util::irtools as util;
use typhon_ir::util::operator::{Operator, OperatorState, run_operator};
use typhon_ir::value::ValueKind;
use typhon_ir::var::Intrinsic;
use typhon_ir_analyze::util::outlining::{ArgKind, OutlineResult, outline_series};

pub const OMP_MODULE: &str = "std.openmp";
pub const GPU_MODULE: &str = "std.internal.gpu";
pub const BUILTIN_MODULE: &str = "std.internal.builtin";

macro_rules! op_state {
    () => {
        fn state(&self) -> &OperatorState {
            &self.state
        }
        fn state_mut(&mut self) -> &mut OperatorState {
            &mut self.state
        }
    };
}

fn warn(m: &Module, msg: &str, v: ValueId) {
    let src = m.src_info(v.into());
    compilation_warning(msg, src.file.as_str(), src.line, src.col, 0, -1, false, MessageGroupPos::None);
}

struct OMPTypes {
    i64: TypeId,
    i32ptr: TypeId,
    i8ptr: TypeId,
}

impl OMPTypes {
    fn new(m: &mut Module) -> Self {
        let i64 = m.get_int_type();
        let i32 = m.get_int_type_n(32, true);
        let i8ptr = m.get_pointer_type(None);
        let i32ptr = m.get_pointer_type(Some(i32));
        Self { i64, i32ptr, i8ptr }
    }
}

fn args_of(m: &Module, call: ValueId) -> Vec<ValueId> {
    m.as_call_instr(call).expect("not a call").args.clone()
}

fn callee_func(m: &Module, call: ValueId) -> Option<FuncId> {
    util::get_func(m, m.as_call_instr(call)?.callee)
}

fn unmangled(m: &Module, f: FuncId) -> String {
    m.func(f).map(|f| f.unmangled_name.clone()).unwrap_or_default()
}

fn is_var_value(m: &Module, v: ValueId) -> bool {
    m.as_var_value(v).is_some()
}
fn is_pointer_value(m: &Module, v: ValueId) -> bool {
    m.as_pointer_value(v).is_some()
}

/// `getVarFromOutlinedArg`.
fn var_from_outlined_arg(m: &Module, arg: ValueId) -> VarId {
    if let Some(v) = m.as_var_value(arg) {
        m.actual_var(v.var)
    } else if let Some(v) = m.as_pointer_value(arg) {
        m.actual_var(v.var)
    } else {
        panic!("unknown outline var");
    }
}

fn pointer_base(m: &Module, t: TypeId) -> TypeId {
    match m.type_kind(t) {
        TypeKind::Pointer { base } => *base,
        _ => panic!("expected shared var to be of pointer type"),
    }
}

fn var_ty(m: &Module, v: VarId) -> TypeId {
    m.var_type(v).expect("var has no type")
}

/// `ptrFromFunc`.
fn ptr_from_func(m: &mut Module, f: FuncId) -> ValueId {
    let ft = var_ty(m, f);
    let raw = m.get_or_realize_method(ft, "__raw__", &[ft], &[]).expect("cannot find function __raw__ method");
    let fv = m.nr_var_value(f);
    util::call(m, raw, vec![fv])
}

fn bin(m: &mut Module, a: ValueId, op: &str, b: ValueId) -> ValueId {
    m.binary_op(a, op, b).unwrap_or_else(|| panic!("could not realize {op}"))
}

fn tuple(m: &mut Module, args: &[ValueId]) -> ValueId {
    util::make_tuple(m, args).expect("could not make tuple")
}

thread_local! {
    static LOCK_COUNTER: Cell<u64> = const { Cell::new(1) };
    static GPU_INSTANCE: Cell<i64> = const { Cell::new(0) };
}

/// Locks are created lazily to avoid them when not needed.
#[derive(Default)]
struct ReductionLocks {
    /// lock used in calls to `_reduce_no_wait` and `_end_reduce_no_wait`
    main_lock: Option<VarId>,
    /// lock used in reduction critical sections
    crit_lock: Option<VarId>,
}

impl ReductionLocks {
    fn create_lock(m: &mut Module) -> VarId {
        let lock_type = m.get_or_realize_type(&mangle::class(OMP_MODULE, "Lock"), &[]).expect("openmp.Lock type not found");
        let c = LOCK_COUNTER.with(|c| {
            let v = c.get();
            c.set(v + 1);
            v
        });
        let var = m.nr_var(lock_type, true, format!(".omp_lock.{c}"));
        // add it to main function so it doesn't get demoted by IR pass
        let series = m.func_body(m.main_func()).expect("main has no body");
        let init = m.construct(lock_type, vec![]).expect("could not initialize openmp.Lock");
        let a = m.nr_assign(var, init);
        m.series_insert(series, 0, a);
        var
    }
    fn main_lock(&mut self, m: &mut Module) -> VarId {
        *self.main_lock.get_or_insert_with(|| Self::create_lock(m))
    }
    fn crit_lock(&mut self, m: &mut Module) -> VarId {
        *self.crit_lock.get_or_insert_with(|| Self::create_lock(m))
    }
}

/// `Reduction::Kind`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ReductionKind {
    #[default]
    None,
    Add,
    Sub,
    Mul,
    And,
    Or,
    Xor,
    Min,
    Max,
}

#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct Reduction {
    pub kind: ReductionKind,
    pub shared: Option<VarId>,
}

impl Reduction {
    pub fn new(kind: ReductionKind, shared: VarId) -> Self {
        Self { kind, shared: Some(shared) }
    }
    /// `operator bool`.
    pub fn is_some(&self) -> bool {
        self.kind != ReductionKind::None
    }

    pub fn ty(&self, m: &Module) -> TypeId {
        pointer_base(m, var_ty(m, self.shared.expect("empty reduction")))
    }

    /// `getInitial()`.
    pub fn initial(&self, m: &mut Module) -> Option<ValueId> {
        use ReductionKind as K;
        if !self.is_some() {
            return None;
        }
        let ty = self.ty(m);
        if util::is_int(m, ty, 64, true) {
            let v = match self.kind {
                K::Add | K::Sub | K::Or | K::Xor => 0,
                K::Mul => 1,
                K::And => !0,
                K::Min => i64::MAX,
                K::Max => i64::MIN,
                K::None => return None,
            };
            return Some(m.get_int(v));
        }
        match m.type_kind(ty) {
            TypeKind::Float => {
                let v = match self.kind {
                    K::Add | K::Sub => 0.0,
                    K::Mul => 1.0,
                    K::Min => f64::MAX,
                    // Note: the reference uses numeric_limits<double>::min() (smallest positive); preserved.
                    K::Max => f64::MIN_POSITIVE,
                    _ => return None,
                };
                return Some(m.get_float(v));
            }
            TypeKind::Float32 => {
                let v: f32 = match self.kind {
                    K::Add | K::Sub => 0.0,
                    K::Mul => 1.0,
                    K::Min => f32::MAX,
                    K::Max => f32::MIN_POSITIVE,
                    _ => return None,
                };
                let f32t = m.get_or_realize_type("float32", &[])?;
                let fv = m.get_float(v as f64);
                return m.construct(f32t, vec![fv]);
            }
            _ => {}
        }
        let bt = m.get_bool_type();
        if m.type_is(ty, bt) {
            return match self.kind {
                K::And => Some(m.get_bool(true)),
                K::Or | K::Xor => Some(m.get_bool(false)),
                _ => None,
            };
        }
        let init = m.construct(ty, vec![])?;
        let it = m.value_type_mut(init);
        if !m.type_is(it, ty) {
            return None;
        }
        Some(init)
    }

    /// `generateNonAtomicReduction(ptr, arg)`.
    pub fn generate_non_atomic(&self, m: &mut Module, ptr: ValueId, arg: ValueId) -> Option<ValueId> {
        use ReductionKind as K;
        let lhs = util::ptr_load(m, ptr);
        let result = match self.kind {
            // Note: the reference uses `+` for SUB as well (partial sums are added); preserved.
            K::Add | K::Sub => bin(m, lhs, magic::ADD, arg),
            K::Mul => bin(m, lhs, magic::MUL, arg),
            K::And => bin(m, lhs, magic::AND, arg),
            K::Or => bin(m, lhs, magic::OR, arg),
            K::Xor => bin(m, lhs, magic::XOR, arg),
            K::Min | K::Max => {
                // signature is (tuple of args, key, default)
                let name = if self.kind == K::Min { "min" } else { "max" };
                let tup = tuple(m, &[lhs, arg]);
                let nt = m.get_none_type();
                let none = m.construct(nt, vec![]).expect("none construct");
                let tt = m.value_type_mut(tup);
                let nvt = m.value_type_mut(none);
                let f = m
                    .get_or_realize_func(name, &[tt, nvt, nvt], &[], BUILTIN_MODULE)
                    .unwrap_or_else(|| panic!("{name} function not found"));
                let none2 = m.construct(nt, vec![]).expect("none construct");
                util::call(m, f, vec![tup, none, none2])
            }
            K::None => return None,
        };
        Some(util::ptr_store(m, ptr, result))
    }

    /// `generateAtomicReduction(ptr, arg, loc, gtid, locks)`.
    fn generate_atomic(
        &self,
        m: &mut Module,
        ptr: ValueId,
        arg: ValueId,
        loc: Option<VarId>,
        gtid: Option<VarId>,
        locks: &mut ReductionLocks,
    ) -> Option<ValueId> {
        use ReductionKind as K;
        let ty = self.ty(m);
        let func = if util::is_int(m, ty, 64, true) {
            match self.kind {
                K::Add | K::Sub => "_atomic_int_add",
                K::Mul => "_atomic_int_mul",
                K::And => "_atomic_int_and",
                K::Or => "_atomic_int_or",
                K::Xor => "_atomic_int_xor",
                K::Min => "_atomic_int_min",
                K::Max => "_atomic_int_max",
                K::None => "",
            }
        } else {
            match (m.type_kind(ty), self.kind) {
                // Note: the reference has no float SUB atomic (falls through to method lookup); preserved.
                (TypeKind::Float, K::Add) => "_atomic_float_add",
                (TypeKind::Float, K::Mul) => "_atomic_float_mul",
                (TypeKind::Float, K::Min) => "_atomic_float_min",
                (TypeKind::Float, K::Max) => "_atomic_float_max",
                (TypeKind::Float32, K::Add | K::Sub) => "_atomic_float32_add",
                (TypeKind::Float32, K::Mul) => "_atomic_float32_mul",
                (TypeKind::Float32, K::Min) => "_atomic_float32_min",
                (TypeKind::Float32, K::Max) => "_atomic_float32_max",
                _ => "",
            }
        };
        let pt = m.value_type_mut(ptr);
        let at = m.value_type_mut(arg);
        if !func.is_empty() {
            let op = m
                .get_or_realize_func(func, &[pt, at], &[], OMP_MODULE)
                .unwrap_or_else(|| panic!("atomic op '{func}' not found"));
            return Some(util::call(m, op, vec![ptr, arg]));
        }
        let method = match self.kind {
            K::Add | K::Sub => "__atomic_add__",
            K::Mul => "__atomic_mul__",
            K::And => "__atomic_and__",
            K::Or => "__atomic_or__",
            K::Xor => "__atomic_xor__",
            K::Min => "__atomic_min__",
            K::Max => "__atomic_max__",
            K::None => "",
        };
        if !method.is_empty() {
            if let Some(op) = m.get_or_realize_method(at, method, &[pt, at], &[]) {
                return Some(util::call(m, op, vec![ptr, arg]));
            }
        }

        let (loc, gtid) = (loc.expect("loc is null"), gtid.expect("gtid is null"));
        let lck = locks.crit_lock(m);
        let lt = var_ty(m, lck);
        let lck_ptr_type = m.get_pointer_type(Some(lt));
        let (loct, gt) = (var_ty(m, loc), var_ty(m, gtid));
        let crit_begin = m
            .get_or_realize_func("_critical_begin", &[loct, gt, lck_ptr_type], &[], OMP_MODULE)
            .expect("critical begin function not found");
        let crit_end = m
            .get_or_realize_func("_critical_end", &[loct, gt, lck_ptr_type], &[], OMP_MODULE)
            .expect("critical end function not found");
        let a = vec![m.nr_var_value(loc), m.nr_var_value(gtid), m.nr_pointer_value(lck, vec![])];
        let enter = util::call(m, crit_begin, a);
        let operation = self.generate_non_atomic(m, ptr, arg)?;
        let b = vec![m.nr_var_value(loc), m.nr_var_value(gtid), m.nr_pointer_value(lck, vec![])];
        let exit = util::call(m, crit_end, b);
        // make sure the unlock is in a finally-block
        let body = util::series(m, &[operation]).unwrap();
        let fin = util::series(m, &[exit]).unwrap();
        let tc = m.nr_try_catch_flow(body, Some(fin), None);
        util::series(m, &[enter, tc])
    }
}

pub struct ReductionIdentifier {
    state: OperatorState,
    pub shareds: Vec<VarId>,
    pub loop_var_arg: Option<VarId>,
    pub reductions: FxHashMap<Id, Reduction>,
}

impl Default for ReductionIdentifier {
    fn default() -> Self {
        Self::new(Vec::new(), None)
    }
}

impl ReductionIdentifier {
    pub fn new(shareds: Vec<VarId>, loop_var_arg: Option<VarId>) -> Self {
        Self { state: OperatorState::new(false), shareds, loop_var_arg, reductions: FxHashMap::default() }
    }

    fn is_shared(&self, m: &Module, shared: VarId) -> bool {
        let id = m.var_uid(shared);
        if let Some(l) = self.loop_var_arg {
            if m.var_uid(l) == id {
                return false;
            }
        }
        self.shareds.iter().any(|v| m.var_uid(*v) == id)
    }

    fn is_shared_deref(m: &mut Module, shared: VarId, v: ValueId) -> bool {
        let ptr_type = var_ty(m, shared);
        let ty = pointer_base(m, ptr_type);
        let it = m.get_int_type();
        if util::is_call_of_types(m, v, magic::GETITEM, &[Some(ptr_type), Some(it)], Some(ty), true) {
            let args = args_of(m, v);
            let var = util::get_var(m, args[0]);
            return util::is_const_val::<i64>(m, *args.last().unwrap(), &0)
                && var.map(|x| m.var_uid(x) == m.var_uid(shared)).unwrap_or(false);
        }
        false
    }

    fn extract_associative_op_chain(m: &mut Module, v: ValueId, op: &str, ty: TypeId, result: &mut Vec<ValueId>) {
        if util::is_call_of_types(m, v, op, &[Some(ty), None], Some(ty), true)
            || util::is_call_of_types(m, v, op, &[None, Some(ty)], Some(ty), true)
        {
            let args = args_of(m, v);
            Self::extract_associative_op_chain(m, args[0], op, ty, result);
            Self::extract_associative_op_chain(m, *args.last().unwrap(), op, ty, result);
        } else {
            result.push(v);
        }
    }

    fn is_shared_deref_for_reduction_cond(m: &mut Module, shared: VarId, v: ValueId) -> bool {
        if Self::is_shared_deref(m, shared, v) {
            return true;
        }
        // Only unwrap the known lowering artifact:
        // flow(series(assign tmp = shared_deref), tmp)
        if let Some(flow) = m.as_flow_instr(v) {
            let (fl, val) = (flow.flow, flow.val);
            let Some(series) = m.as_series_flow(fl) else { return false };
            let Some(ret) = m.as_var_value(val) else { return false };
            if series.series.len() != 1 {
                return false;
            }
            let ret_var = ret.var;
            let Some(assign) = m.as_assign_instr(series.series[0]) else { return false };
            if m.var_uid(assign.lhs) != m.var_uid(ret_var) {
                return false;
            }
            let rhs = assign.rhs;
            return Self::is_shared_deref(m, shared, rhs);
        }
        false
    }

    fn condition_result_var(m: &Module, v: ValueId) -> Option<VarId> {
        if let Some(flow) = m.as_flow_instr(v) {
            if let Some(ret) = m.as_var_value(flow.val) {
                return Some(ret.var);
            }
        }
        m.as_var_value(v).map(|r| r.var)
    }

    fn is_condition_result(m: &Module, cond: ValueId, v: ValueId) -> bool {
        match (Self::condition_result_var(m, cond), m.as_var_value(v)) {
            (Some(c), Some(vv)) => m.var_uid(c) == m.var_uid(vv.var),
            _ => false,
        }
    }

    fn is_logical_false(m: &Module, cond: ValueId, v: ValueId) -> bool {
        util::is_const_val::<bool>(m, v, &false) || Self::is_condition_result(m, cond, v)
    }
    fn is_logical_true(m: &Module, cond: ValueId, v: ValueId) -> bool {
        util::is_const_val::<bool>(m, v, &true) || Self::is_condition_result(m, cond, v)
    }

    fn checked(m: &mut Module, kind: ReductionKind, shared: VarId) -> Reduction {
        let r = Reduction::new(kind, shared);
        if r.initial(m).is_some() { r } else { Reduction::default() }
    }

    fn reduction_from_ternary(m: &mut Module, shared: VarId, ty: TypeId, item: ValueId) -> Reduction {
        let bt = m.get_bool_type();
        if !m.type_is(ty, bt) {
            return Reduction::default();
        }
        let Some(t) = m.as_ternary_instr(item) else { return Reduction::default() };
        let (cond, tv, fv) = (t.cond, t.true_value, t.false_value);

        if Self::is_shared_deref_for_reduction_cond(m, shared, cond) && Self::is_logical_false(m, cond, fv) {
            let tt = m.value_type_mut(tv);
            if m.type_is(tt, ty) {
                return Self::checked(m, ReductionKind::And, shared);
            }
        }
        if Self::is_shared_deref_for_reduction_cond(m, shared, cond) && Self::is_logical_true(m, cond, tv) {
            let ft = m.value_type_mut(fv);
            if m.type_is(ft, ty) {
                return Self::checked(m, ReductionKind::Or, shared);
            }
        }
        if Self::is_shared_deref_for_reduction_cond(m, shared, tv) && Self::is_logical_false(m, cond, fv) {
            return Self::checked(m, ReductionKind::And, shared);
        }
        if Self::is_shared_deref_for_reduction_cond(m, shared, fv) && Self::is_logical_true(m, cond, tv) {
            return Self::checked(m, ReductionKind::Or, shared);
        }
        Reduction::default()
    }

    /// `getReductionFromCall`.
    pub fn reduction_from_call(&self, m: &mut Module, v: ValueId) -> Reduction {
        let none = Reduction::default();
        let Some(func) = callee_func(m, v) else { return none };
        let args = args_of(m, v);
        if args.len() != 3 || unmangled(m, func) != magic::SETITEM {
            return none;
        }
        let (self_, idx, item) = (args[0], args[1], args[2]);
        let Some(shared) = util::get_var(m, self_) else { return none };
        if !self.is_shared(m, shared) || !util::is_const_val::<i64>(m, idx, &0) {
            return none;
        }
        let ptr_type = var_ty(m, shared);
        let ty = pointer_base(m, ptr_type);
        let none_type = m.get_none_type();

        // double-check the call
        let (st, it, vt) = (m.value_type_mut(self_), m.value_type_mut(idx), m.value_type_mut(item));
        if !util::is_call_of_types(m, v, magic::SETITEM, &[Some(st), Some(it), Some(vt)], Some(none_type), true) {
            return none;
        }

        let funcs: [(&str, ReductionKind, bool); 8] = [
            (magic::ADD, ReductionKind::Add, true),
            (magic::SUB, ReductionKind::Sub, true),
            (magic::MUL, ReductionKind::Mul, true),
            (magic::AND, ReductionKind::And, true),
            (magic::OR, ReductionKind::Or, true),
            (magic::XOR, ReductionKind::Xor, true),
            ("min", ReductionKind::Min, false),
            ("max", ReductionKind::Max, false),
        ];

        for (name, kind, method) in funcs {
            if method {
                if !(util::is_call_of_types(m, item, name, &[Some(ty), None], Some(ty), true)
                    || util::is_call_of_types(m, item, name, &[None, Some(ty)], Some(ty), true))
                {
                    continue;
                }
            } else {
                let tt = m.get_tuple_type(vec![ty, ty]);
                if !util::is_call_of_types(m, item, name, &[Some(tt), Some(none_type), Some(none_type)], Some(ty), false) {
                    continue;
                }
            }

            let mut deref = None;
            if method {
                let mut chain = Vec::new();
                Self::extract_associative_op_chain(m, item, name, ty, &mut chain);
                if chain.len() < 2 {
                    continue;
                }
                deref = chain.into_iter().find(|val| Self::is_shared_deref(m, shared, *val));
            } else {
                // this will be Tuple.__new__
                let first = args_of(m, item)[0];
                if m.as_call_instr(first).is_none() {
                    continue;
                }
                for val in args_of(m, first) {
                    if Self::is_shared_deref(m, shared, val) {
                        deref = Some(val);
                        break;
                    }
                }
            }
            if deref.is_none() {
                return none;
            }
            return Self::checked(m, kind, shared);
        }

        Self::reduction_from_ternary(m, shared, ty, item)
    }

    /// `getReduction(shared)`.
    pub fn reduction(&self, m: &Module, shared: VarId) -> Reduction {
        self.reductions.get(&m.var_uid(shared)).copied().unwrap_or_default()
    }
}

impl Operator for ReductionIdentifier {
    op_state!();
    fn handle_call_instr(&mut self, m: &mut Module, v: ValueId) {
        let r = self.reduction_from_call(m, v);
        if r.is_some() {
            let id = m.var_uid(r.shared.unwrap());
            // if we've seen the var before, make sure it's consistent
            // otherwise mark as invalid via an empty reduction
            match self.reductions.get_mut(&id) {
                None => {
                    self.reductions.insert(id, r);
                }
                Some(e) if e.is_some() && e.kind != r.kind => *e = Reduction::default(),
                _ => {}
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct SharedInfo {
    /// member index in template's `extra` arg
    memb: usize,
    /// the local var we create to store current value
    local: VarId,
    /// the reduction we're performing, or empty if none
    reduction: Reduction,
}

/// Common state of `LoopTemplateReplacer` / `ParallelLoopTemplateReplacer`.
struct ParallelCore {
    parent: FuncId,
    replacement: Option<ValueId>,
    loop_var: VarId,
    shared_info: Vec<SharedInfo>,
    locks: ReductionLocks,
    loc_ref: Option<VarId>,
    reduction_loc_ref: Option<VarId>,
    gtid: Option<VarId>,
}

impl ParallelCore {
    fn new(parent: FuncId, replacement: ValueId, loop_var: VarId) -> Self {
        Self {
            parent,
            replacement: Some(replacement),
            loop_var,
            shared_info: Vec::new(),
            locks: ReductionLocks::default(),
            loc_ref: None,
            reduction_loc_ref: None,
            gtid: None,
        }
    }

    fn num_reductions(&self) -> usize {
        self.shared_info.iter().filter(|i| i.reduction.is_some()).count()
    }

    fn reduction_tuple(&self, m: &mut Module) -> ValueId {
        let elems: Vec<ValueId> = self
            .shared_info
            .iter()
            .filter(|i| i.reduction.is_some())
            .map(|i| i.local)
            .collect::<Vec<_>>()
            .into_iter()
            .map(|l| m.nr_pointer_value(l, vec![]))
            .collect();
        tuple(m, &elems)
    }

    fn make_reduction_func(&self, m: &mut Module) -> FuncId {
        let tv = self.reduction_tuple(m);
        let tuple_type = m.value_type_mut(tv);
        let arg_type = m.get_pointer_type(Some(tuple_type));
        let nt = m.get_none_type();
        let func_type = m.get_func_type(nt, vec![arg_type, arg_type], false);
        let name = util::derived_func_name(m, "__omp_reducer", self.parent, None);
        let reducer = m.nr_bodied_func(name);
        m.realize_func(reducer, func_type, &["lhs".into(), "rhs".into()]);
        let fargs = m.func_args(reducer);
        let (lhs_var, rhs_var) = (fargs[0], fargs[1]);
        let body = m.nr_series_flow();
        let mut next = 0;
        for info in self.shared_info.clone() {
            if info.reduction.is_some() {
                let lv = m.nr_var_value(lhs_var);
                let lhs = util::ptr_load(m, lv);
                let rv = m.nr_var_value(rhs_var);
                let rhs = util::ptr_load(m, rv);
                let le = util::tuple_get(m, lhs, next);
                let re = util::tuple_get(m, rhs, next);
                let rl = util::ptr_load(m, re);
                let r = info.reduction.generate_non_atomic(m, le, rl).unwrap();
                m.series_push(body, r);
                next += 1;
            }
        }
        m.set_func_body(reducer, Some(body));
        reducer
    }

    /// `ParallelLoopTemplateReplacer::handle(CallInstr*)`.
    fn handle_call(&mut self, m: &mut Module, v: ValueId) {
        let Some(func) = callee_func(m, v) else { return };
        let name = unmangled(m, func);

        if name == "_loop_loc_and_gtid" {
            let args = args_of(m, v);
            assert!(args.len() == 3 && args.iter().all(|a| is_var_value(m, *a)), "unexpected loop loc and gtid stub");
            self.loc_ref = util::get_var(m, args[0]);
            self.reduction_loc_ref = util::get_var(m, args[1]);
            self.gtid = util::get_var(m, args[2]);
        }

        if name == "_loop_reductions" {
            let (Some(red_loc), Some(gtid)) = (self.reduction_loc_ref, self.gtid) else {
                panic!("bad visit order in template")
            };
            let args = args_of(m, v);
            assert!(args.len() == 1 && is_var_value(m, args[0]), "unexpected shared updates stub");
            if self.num_reductions() == 0 {
                return;
            }
            let extras = util::get_var(m, args[0]).unwrap();
            let series = m.nr_series_flow();
            for i in 0..self.shared_info.len() {
                if self.shared_info[i].reduction.is_some() {
                    let vv = m.nr_var_value(self.shared_info[i].local);
                    self.shared_info[i].local = util::make_var(m, vv, series, Some(self.parent), false);
                }
            }
            let reduction_tuple = self.reduction_tuple(m);
            let reducer = self.make_reduction_func(m);
            let lck = self.locks.main_lock(m);
            let raw_reducer = ptr_from_func(m, reducer);

            let lt = var_ty(m, lck);
            let lck_ptr_type = m.get_pointer_type(Some(lt));
            let (rlt, gt) = (var_ty(m, red_loc), var_ty(m, gtid));
            let (rtt, rrt) = (m.value_type_mut(reduction_tuple), m.value_type_mut(raw_reducer));
            let reduce = m
                .get_or_realize_func("_reduce_nowait", &[rlt, gt, rtt, rrt, lck_ptr_type], &[], OMP_MODULE)
                .expect("reduce nowait function not found");
            let reduce_end = m
                .get_or_realize_func("_end_reduce_nowait", &[rlt, gt, lck_ptr_type], &[], OMP_MODULE)
                .expect("end reduce nowait function not found");

            let tuple_val = util::make_var(m, reduction_tuple, series, Some(self.parent), false);
            let a = vec![
                m.nr_var_value(red_loc),
                m.nr_var_value(gtid),
                m.nr_var_value(tuple_val),
                raw_reducer,
                m.nr_pointer_value(lck, vec![]),
            ];
            let reduce_code = util::call(m, reduce, a);
            let code_var = util::make_var(m, reduce_code, series, Some(self.parent), false);
            let it = m.get_int_type();
            assert!(m.type_is(var_ty(m, code_var), it), "wrong reduce code type");

            let non_atomic = m.nr_series_flow();
            let atomic = m.nr_series_flow();
            for info in self.shared_info.clone() {
                if info.reduction.is_some() {
                    let ev = m.nr_var_value(extras);
                    let ptr = util::tuple_get(m, ev, info.memb);
                    let arg = m.nr_var_value(info.local);
                    let r = info.reduction.generate_non_atomic(m, ptr, arg).unwrap();
                    m.series_push(non_atomic, r);
                }
            }
            let a = vec![m.nr_var_value(red_loc), m.nr_var_value(gtid), m.nr_pointer_value(lck, vec![])];
            let end = util::call(m, reduce_end, a);
            m.series_push(non_atomic, end);

            for info in self.shared_info.clone() {
                if info.reduction.is_some() {
                    let ev = m.nr_var_value(extras);
                    let ptr = util::tuple_get(m, ev, info.memb);
                    let arg = m.nr_var_value(info.local);
                    let r = info
                        .reduction
                        .generate_atomic(m, ptr, arg, self.loc_ref, self.gtid, &mut self.locks)
                        .unwrap();
                    m.series_push(atomic, r);
                }
            }

            // make: if code == 1 { sectionNonAtomic } elif code == 2 { sectionAtomic }
            let cv = m.nr_var_value(code_var);
            let one = m.get_int(1);
            let c1 = bin(m, cv, magic::EQ, one);
            let cv2 = m.nr_var_value(code_var);
            let two = m.get_int(2);
            let c2 = bin(m, cv2, magic::EQ, two);
            let inner = m.nr_if_flow(c2, atomic, None);
            let elif = util::series(m, &[inner]).unwrap();
            let sw = m.nr_if_flow(c1, non_atomic, Some(elif));
            m.series_push(series, sw);
            m.replace_all_value(v, series);
        }
    }
}

struct ImperativeLoopTemplateReplacer<'a> {
    state: OperatorState,
    core: ParallelCore,
    reds: &'a ReductionIdentifier,
    sched: OMPSched,
    step: i64,
}

impl Operator for ImperativeLoopTemplateReplacer<'_> {
    op_state!();
    fn handle_call_instr(&mut self, m: &mut Module, v: ValueId) {
        self.core.handle_call(m, v);
        let Some(func) = callee_func(m, v) else { return };
        let name = unmangled(m, func);
        let parent = self.core.parent;

        if name == "_loop_step" {
            let s = m.get_int(self.step);
            m.replace_all_value(v, s);
        }

        if name == "_loop_body_stub" {
            let replacement = self.core.replacement.expect("unexpected double replacement");
            let args = args_of(m, v);
            assert!(
                args.len() == 2 && is_var_value(m, args[0]) && is_var_value(m, args[1]),
                "unexpected loop body stub"
            );
            let outlined = callee_func(m, replacement).unwrap();
            // the template passes the new loop var and extra args to the body stub for convenience
            let new_loop_var = util::get_var(m, args[0]).unwrap();
            let extras = util::get_var(m, args[1]).unwrap();
            let outlined_args = m.func_args(outlined);
            let loop_uid = m.var_uid(self.core.loop_var);

            let mut new_args = Vec::new();
            let mut next = 0usize;
            for (k, arg) in args_of(m, replacement).into_iter().enumerate() {
                if m.var_uid(var_from_outlined_arg(m, arg)) != loop_uid {
                    let new_arg = if is_pointer_value(m, arg) {
                        let at = m.value_type_mut(arg);
                        let base = pointer_base(m, at);
                        // get extras again since we'll be inserting the new var before extras local
                        let last_arg = *m.func_args(parent).last().unwrap();
                        let lv = m.nr_var_value(last_arg);
                        let ld = util::ptr_load(m, lv);
                        let val = util::tuple_get(m, ld, 3);
                        let tg = util::tuple_get(m, val, next);
                        let mut init = util::ptr_load(m, tg);
                        let reduction = self.reds.reduction(m, outlined_args[k]);
                        if reduction.is_some() {
                            init = reduction.initial(m).expect("unknown reduction init value");
                            let it = m.value_type_mut(init);
                            assert!(m.type_is(it, base), "unknown reduction init value");
                        }
                        let body = m.func_body(parent).unwrap();
                        let new_var = util::make_var(m, init, body, Some(parent), true);
                        self.core.shared_info.push(SharedInfo { memb: next, local: new_var, reduction });
                        next += 1;
                        m.nr_pointer_value(new_var, vec![])
                    } else {
                        let ev = m.nr_var_value(extras);
                        let r = util::tuple_get(m, ev, next);
                        next += 1;
                        r
                    };
                    new_args.push(new_arg);
                } else if is_var_value(m, arg) {
                    new_args.push(m.nr_var_value(new_loop_var));
                } else if is_pointer_value(m, arg) {
                    new_args.push(m.nr_pointer_value(new_loop_var, vec![]));
                } else {
                    panic!("unknown outline var");
                }
            }
            let c = util::call(m, outlined, new_args);
            m.replace_all_value(v, c);
            self.core.replacement = None;
        }

        if name == "_loop_shared_updates" {
            // for all non-reduction shareds, set the final values
            // this will be similar to OpenMP's "lastprivate"
            let args = args_of(m, v);
            assert!(args.len() == 1 && is_var_value(m, args[0]), "unexpected shared updates stub");
            let extras = util::get_var(m, args[0]).unwrap();
            let series = m.nr_series_flow();
            for info in self.core.shared_info.clone() {
                if info.reduction.is_some() {
                    continue;
                }
                let fv = m.nr_var_value(info.local);
                let ev = m.nr_var_value(extras);
                let orig = util::tuple_get(m, ev, info.memb);
                let st = util::ptr_store(m, orig, fv);
                m.series_push(series, st);
            }
            m.replace_all_value(v, series);
        }

        if name == "_loop_schedule" {
            let c = m.get_int(self.sched.code as i64);
            m.replace_all_value(v, c);
        }
        if name == "_loop_ordered" {
            let c = m.get_bool(self.sched.ordered);
            m.replace_all_value(v, c);
        }
    }
}

struct TaskLoopReductionVarReplacer {
    state: OperatorState,
    reduction_remap: Vec<(VarId, VarId)>,
    parent: FuncId,
}

impl TaskLoopReductionVarReplacer {
    fn new(m: &mut Module, reduction_args: Vec<VarId>, parent: FuncId) -> Self {
        let reduction_remap = reduction_args
            .into_iter()
            .map(|var| {
                let t = var_ty(m, var);
                (var, m.nr_var(t, false, ""))
            })
            .collect();
        Self { state: OperatorState::new(false), reduction_remap, parent }
    }

    /// Needs to be a separate step since otherwise the old variable in the
    /// assignment would be replaced.
    fn finalize(&self, m: &mut Module) {
        let body = m.func_body(self.parent).unwrap();
        let gtid = *m.func_args(self.parent).last().unwrap();
        for &(old, new) in &self.reduction_remap {
            let it = m.get_int_type();
            let ot = var_ty(m, old);
            let trd = m.get_or_realize_func("_taskred_data", &[it, ot], &[], OMP_MODULE).expect("could not find '_taskred_data'");
            let a = vec![m.nr_var_value(gtid), m.nr_var_value(old)];
            let c = util::call(m, trd, a);
            let assign = m.nr_assign(new, c);
            m.series_insert(body, 0, assign);
            m.func_push_symbol(self.parent, new);
        }
    }
}

impl Operator for TaskLoopReductionVarReplacer {
    op_state!();
    fn pre_hook(&mut self, m: &mut Module, n: NodeRef) {
        for &(old, new) in &self.reduction_remap {
            let uid = m.var_uid(old);
            m.replace_used_variable_uid(n, uid, Some(new));
        }
    }
}

struct TaskLoopBodyStubReplacer {
    state: OperatorState,
    replacement: Option<ValueId>,
    reduce_args: Vec<bool>,
}

impl Operator for TaskLoopBodyStubReplacer {
    op_state!();
    fn handle_call_instr(&mut self, m: &mut Module, v: ValueId) {
        let Some(func) = callee_func(m, v) else { return };
        if unmangled(m, func) != "_task_loop_body_stub" {
            return;
        }
        let replacement = self.replacement.expect("unexpected double replacement");
        let args = args_of(m, v);
        assert!(
            args.len() == 3 && is_var_value(m, args[0]) && is_var_value(m, args[2]),
            "unexpected loop body stub"
        );
        // the template passes gtid, privs and shareds to the body stub for convenience
        let (gtid, privs, shareds) = (args[0], args[1], args[2]);
        let (mut pn, mut sn) = (0, 0);
        let mut new_args = Vec::new();
        let has_reductions = self.reduce_args.iter().any(|b| *b);
        let rargs = args_of(m, replacement);
        for &arg in &rargs {
            if is_var_value(m, arg) {
                new_args.push(util::tuple_get(m, privs, pn));
                pn += 1;
            } else if is_pointer_value(m, arg) {
                new_args.push(util::tuple_get(m, shareds, sn));
                sn += 1;
            } else {
                // make sure we're on the last arg, which should be gtid in case of reductions
                assert!(has_reductions && Some(&arg) == rargs.last(), "unknown outline var");
            }
        }
        let outlined = callee_func(m, replacement).unwrap();
        if has_reductions {
            new_args.push(gtid);
            let reduction_args: Vec<VarId> = m
                .func_args(outlined)
                .into_iter()
                .enumerate()
                .filter(|(i, _)| self.reduce_args[*i])
                .map(|(_, a)| a)
                .collect();
            let mut redrep = TaskLoopReductionVarReplacer::new(m, reduction_args, outlined);
            redrep.process(m, NodeRef::Var(outlined));
            redrep.finalize(m);
        }
        let c = util::call(m, outlined, new_args);
        m.replace_all_value(v, c);
        self.replacement = None;
    }
}

struct TaskLoopRoutineStubReplacer {
    state: OperatorState,
    core: ParallelCore,
    privates: Vec<ValueId>,
    shareds: Vec<ValueId>,
    /// task reduction input array
    array: Option<VarId>,
    /// task group identifier
    tskgrp: Option<VarId>,
}

impl TaskLoopRoutineStubReplacer {
    fn new(
        m: &mut Module,
        core: ParallelCore,
        privates: Vec<ValueId>,
        shareds: Vec<ValueId>,
        shared_redux: &[Reduction],
    ) -> Self {
        let mut s = Self { state: OperatorState::new(false), core, privates, shareds, array: None, tskgrp: None };
        let loop_uid = m.var_uid(s.core.loop_var);
        let parent = s.core.parent;
        for (next, val) in s.shareds.clone().into_iter().enumerate() {
            if m.var_uid(var_from_outlined_arg(m, val)) != loop_uid {
                let reduction = shared_redux[next];
                if reduction.is_some() {
                    let init = reduction.initial(m).expect("reduction initial");
                    let body = m.func_body(parent).unwrap();
                    let nv = util::make_var(m, init, body, Some(parent), true);
                    s.core.shared_info.push(SharedInfo { memb: next, local: nv, reduction });
                }
            }
        }
        s
    }

    fn red_func(&self, m: &mut Module, r: &Reduction, prefix: &str) -> (FuncId, VarId, VarId) {
        let t = r.ty(m);
        let at = m.get_pointer_type(Some(t));
        let nt = m.get_none_type();
        let ft = m.get_func_type(nt, vec![at, at], false);
        let name = util::derived_func_name(m, prefix, self.core.parent, r.shared.map(NodeRef::Var));
        let f = m.nr_bodied_func(name);
        m.realize_func(f, ft, &["lhs".into(), "rhs".into()]);
        let a = m.func_args(f);
        (f, a[0], a[1])
    }

    fn make_task_red_init_func(&self, m: &mut Module, r: &Reduction) -> FuncId {
        let (f, lhs, _) = self.red_func(m, r, "__red_init");
        let body = m.nr_series_flow();
        let lp = m.nr_var_value(lhs);
        let init = r.initial(m).unwrap();
        let st = util::ptr_store(m, lp, init);
        m.series_push(body, st);
        m.set_func_body(f, Some(body));
        f
    }

    fn make_task_red_comb_func(&self, m: &mut Module, r: &Reduction) -> FuncId {
        let (f, lhs, rhs) = self.red_func(m, r, "__red_comb");
        let body = m.nr_series_flow();
        let lp = m.nr_var_value(lhs);
        let rp = m.nr_var_value(rhs);
        let rl = util::ptr_load(m, rp);
        let x = r.generate_non_atomic(m, lp, rl).unwrap();
        m.series_push(body, x);
        m.set_func_body(f, Some(body));
        f
    }

    fn make_task_red_input(&self, m: &mut Module, r: &Reduction, shar: ValueId, orig: ValueId) -> ValueId {
        let t = r.ty(m);
        let so = m.get_intrinsic(Intrinsic::Sizeof, Some(t), None);
        let size = util::call(m, so, vec![]);
        let initf = self.make_task_red_init_func(m, r);
        let init = ptr_from_func(m, initf);
        let combf = self.make_task_red_comb_func(m, r);
        let comb = ptr_from_func(m, combf);
        let tri = m
            .get_or_realize_type(&mangle::class(OMP_MODULE, "TaskReductionInput"), &[])
            .expect("could not find 'TaskReductionInput' type");
        m.construct(tri, vec![shar, orig, size, init, comb]).expect("bad construction of 'TaskReductionInput' type")
    }

    fn is_reduction_shared_at(&self, info_next: usize, shareds_next: usize) -> bool {
        info_next < self.core.shared_info.len()
            && self.core.shared_info[info_next].memb == shareds_next
            && self.core.shared_info[info_next].reduction.is_some()
    }
}

impl Operator for TaskLoopRoutineStubReplacer {
    op_state!();

    fn handle_var_value(&mut self, m: &mut Module, v: ValueId) {
        let Some(func) = util::get_func(m, v) else { return };
        if unmangled(m, func) != "_routine_stub" {
            return;
        }
        let replacement = self.core.replacement.expect("no replacement");
        let rargs = args_of(m, replacement);
        let mut reduce_args = Vec::new();
        let (mut sn, mut inext) = (0, 0);
        for &arg in &rargs {
            if is_var_value(m, arg) {
                reduce_args.push(false);
            } else if is_pointer_value(m, arg) {
                if self.is_reduction_shared_at(inext, sn) {
                    reduce_args.push(true);
                    inext += 1;
                } else {
                    reduce_args.push(false);
                }
                sn += 1;
            } else {
                // make sure we're on the last arg, which should be gtid in case of reductions
                assert!(self.core.num_reductions() > 0 && Some(&arg) == rargs.last(), "unknown outline var");
                reduce_args.push(false);
            }
        }

        let mut cv = CloneVisitor::new(true);
        let new_routine = cv.force_clone_var(m, func);
        let name = util::derived_func_name(m, &m.var_name(func).to_string(), self.core.parent, None);
        m.set_name(new_routine.into(), name);
        let mut rep =
            TaskLoopBodyStubReplacer { state: OperatorState::new(false), replacement: Some(replacement), reduce_args };
        rep.process(m, NodeRef::Var(new_routine));
        m.as_var_value_mut(v).unwrap().var = new_routine;
    }

    fn handle_call_instr(&mut self, m: &mut Module, v: ValueId) {
        self.core.handle_call(m, v);
        let Some(func) = callee_func(m, v) else { return };
        let name = unmangled(m, func);
        let parent = self.core.parent;

        if name == "_taskred_setup" {
            let (Some(red_loc), Some(gtid)) = (self.core.reduction_loc_ref, self.core.gtid) else {
                panic!("bad visit order in template")
            };
            let args = args_of(m, v);
            assert!(args.len() == 1 && is_var_value(m, args[0]), "unexpected shared updates stub");
            let num_red = self.core.num_reductions();
            if num_red == 0 {
                return;
            }
            let extras = util::get_var(m, args[0]).unwrap();

            // add task reduction inputs
            let init_series = m.nr_series_flow();
            let tri = m
                .get_or_realize_type(&mangle::class(OMP_MODULE, "TaskReductionInput"), &[])
                .expect("could not find 'TaskReductionInput' type");
            let arr_type = m.get_pointer_type(Some(tri));
            let sa = m.nr_stack_alloc(arr_type, num_red as i64);
            let array = util::make_var(m, sa, init_series, Some(parent), false);
            self.array = Some(array);
            let it = m.get_int_type();
            let setitem = m
                .get_or_realize_method(arr_type, magic::SETITEM, &[arr_type, it, tri], &[])
                .expect("could not find 'TaskReductionInputArray.__setitem__' method");
            let mut i = 0;
            for info in self.core.shared_info.clone() {
                if info.reduction.is_some() {
                    let shar = m.nr_pointer_value(info.local, vec![]);
                    let ev = m.nr_var_value(extras);
                    let orig = util::tuple_get(m, ev, info.memb);
                    let input = self.make_task_red_input(m, &info.reduction, shar, orig);
                    let av = m.nr_var_value(array);
                    let iv = m.get_int(i);
                    i += 1;
                    let c = util::call(m, setitem, vec![av, iv, input]);
                    m.series_push(init_series, c);
                }
            }
            let (rlt, gt, at) = (var_ty(m, red_loc), var_ty(m, gtid), var_ty(m, array));
            let init_func = m
                .get_or_realize_func("_taskred_init", &[rlt, gt, it, at], &[], OMP_MODULE)
                .expect("task red init function not found");
            let a = vec![m.nr_var_value(red_loc), m.nr_var_value(gtid), m.get_int(num_red as i64), m.nr_var_value(array)];
            let c = util::call(m, init_func, a);
            self.tskgrp = Some(util::make_var(m, c, init_series, Some(parent), false));
            m.replace_all_value(v, init_series);
        }

        if name == "_fix_privates_and_shareds" {
            let args = args_of(m, v);
            assert!(args.len() == 3, "invalid _fix_privates_and_shareds call found");
            let num_red = self.core.num_reductions();
            let (new_loop_var, mut privs, mut shareds) = (args[0], args[1], args[2]);
            let loop_uid = m.var_uid(self.core.loop_var);
            let (mut need_p, mut need_s) = (false, false);
            let mut new_privates = Vec::new();
            let mut new_shareds = Vec::new();

            let last_priv = self.privates.last().copied();
            for (pn, val) in self.privates.clone().into_iter().enumerate() {
                if num_red > 0 && Some(val) == last_priv {
                    // i.e. task group identifier
                    let t = self.tskgrp.expect("tskgrp var not set");
                    new_privates.push(m.nr_var_value(t));
                    need_p = true;
                } else if m.var_uid(var_from_outlined_arg(m, val)) != loop_uid {
                    new_privates.push(util::tuple_get(m, privs, pn));
                } else {
                    new_privates.push(new_loop_var);
                    need_p = true;
                }
            }
            let mut inext = 0;
            for (sn, val) in self.shareds.clone().into_iter().enumerate() {
                if m.var_uid(var_from_outlined_arg(m, val)) != loop_uid {
                    if self.is_reduction_shared_at(inext, sn) {
                        let l = self.core.shared_info[inext].local;
                        new_shareds.push(m.nr_pointer_value(l, vec![]));
                        need_s = true;
                        inext += 1;
                    } else {
                        new_shareds.push(util::tuple_get(m, shareds, sn));
                    }
                } else {
                    let lv = util::get_var(m, new_loop_var).expect("loop var");
                    new_shareds.push(m.nr_pointer_value(lv, vec![]));
                    need_s = true;
                }
            }
            if need_p {
                privs = tuple(m, &new_privates);
            }
            if need_s {
                shareds = tuple(m, &new_shareds);
            }
            let result = tuple(m, &[privs, shareds]);
            m.replace_all_value(v, result);
        }

        if name == "_taskred_finish" {
            let (Some(red_loc), Some(gtid)) = (self.core.reduction_loc_ref, self.core.gtid) else {
                panic!("bad visit order in template")
            };
            if self.core.num_reductions() == 0 {
                return;
            }
            let (rlt, gt) = (var_ty(m, red_loc), var_ty(m, gtid));
            let fini = m
                .get_or_realize_func("_taskred_fini", &[rlt, gt], &[], OMP_MODULE)
                .expect("taskred finish function not found not found");
            let a = vec![m.nr_var_value(red_loc), m.nr_var_value(gtid)];
            let c = util::call(m, fini, a);
            m.replace_all_value(v, c);
        }
    }
}

struct GPULoopBodyStubReplacer {
    state: OperatorState,
    replacement: Option<ValueId>,
    loop_var: VarId,
    step: i64,
}

impl Operator for GPULoopBodyStubReplacer {
    op_state!();
    fn handle_call_instr(&mut self, m: &mut Module, v: ValueId) {
        let Some(func) = callee_func(m, v) else { return };
        let name = unmangled(m, func);
        if name == "_gpu_loop_body_stub" {
            let replacement = self.replacement.expect("unexpected double replacement");
            let args = args_of(m, v);
            assert!(args.len() == 2, "unexpected loop body stub");
            let (idx, targs) = (args[0], args[1]);
            let loop_uid = m.var_uid(self.loop_var);
            let mut next = 0;
            let mut new_args = Vec::new();
            for arg in args_of(m, replacement) {
                if m.var_uid(var_from_outlined_arg(m, arg)) == loop_uid {
                    new_args.push(idx);
                } else {
                    new_args.push(util::tuple_get(m, targs, next));
                    next += 1;
                }
            }
            let outlined = callee_func(m, replacement).unwrap();
            let c = util::call(m, outlined, new_args);
            m.replace_all_value(v, c);
            self.replacement = None;
        }
        if name == "_loop_step" {
            let s = m.get_int(self.step);
            m.replace_all_value(v, s);
        }
    }
}

struct GPULoopTemplateReplacer {
    state: OperatorState,
    step: i64,
}

impl Operator for GPULoopTemplateReplacer {
    op_state!();
    fn handle_call_instr(&mut self, m: &mut Module, v: ValueId) {
        let Some(func) = callee_func(m, v) else { return };
        if unmangled(m, func) == "_loop_step" {
            let s = m.get_int(self.step);
            m.replace_all_value(v, s);
        }
    }
}

struct OpenMPTransformData {
    outline: OutlineResult,
    shared_vars: Vec<VarId>,
    reds: ReductionIdentifier,
}

fn loop_parts(m: &Module, v: ValueId) -> (Option<&OMPSched>, ValueId, VarId) {
    match m.kind(v) {
        ValueKind::ForFlow(f) => (f.schedule.as_deref(), f.body, f.var),
        ValueKind::ImperativeForFlow(f) => (f.schedule.as_deref(), f.body, f.var),
        _ => panic!("not a loop"),
    }
}

fn unpar(m: &mut Module, v: ValueId) -> Option<OpenMPTransformData> {
    match m.kind_mut(v) {
        ValueKind::ForFlow(f) => f.set_parallel(false),
        ValueKind::ImperativeForFlow(f) => f.set_parallel(false),
        _ => {}
    }
    None
}

/// `setupOpenMPTransform`. `None` = loop was made sequential.
fn setup_openmp_transform(m: &mut Module, v: ValueId, parent: Option<FuncId>, gpu: bool) -> Option<OpenMPTransformData> {
    let (sched, body, loop_var) = loop_parts(m, v);
    if sched.is_none() {
        return unpar(m, v);
    }
    let parent = parent.filter(|p| m.func(*p).map(|f| f.is_bodied()).unwrap_or(false));
    let (Some(parent), true) = (parent, m.as_series_flow(body).is_some()) else { return unpar(m, v) };
    let outline = outline_series(m, parent, body, false, true, gpu);
    if !outline.is_some() {
        return unpar(m, v);
    }
    let call_args = args_of(m, outline.call.unwrap());
    let func_args = m.func_args(outline.func.unwrap());
    let loop_uid = m.var_uid(loop_var);
    let mut shared_vars = Vec::new();
    let mut loop_var_arg = None;
    for (i, a) in func_args.into_iter().enumerate() {
        // pick out loop variable to pass to reduction identifier, which will ensure we don't reduce over it
        if m.var_uid(var_from_outlined_arg(m, call_args[i])) == loop_uid {
            loop_var_arg = Some(a);
        }
        if outline.arg_kinds[i] == ArgKind::Modified {
            shared_vars.push(a);
        }
    }
    let mut reds = ReductionIdentifier::new(shared_vars.clone(), loop_var_arg);
    reds.process(m, NodeRef::Var(outline.func.unwrap()));
    Some(OpenMPTransformData { outline, shared_vars, reds })
}

struct ForkCallData {
    fork: ValueId,
    push_num_threads: Option<ValueId>,
}

fn create_fork_call(m: &mut Module, types: &OMPTypes, raw_template: ValueId, extra: &[ValueId], sched: &OMPSched) -> ForkCallData {
    let fe = tuple(m, extra);
    let ft = m.value_type_mut(fe);
    let fork_func = m.get_or_realize_func("_fork_call", &[types.i8ptr, ft], &[], OMP_MODULE).expect("fork call function not found");
    let fork = util::call(m, fork_func, vec![raw_template, fe]);
    let mut push_num_threads = None;
    if let Some(t) = sched.threads {
        let tt = m.value_type_mut(t);
        if m.type_is(tt, types.i64) {
            let f = m
                .get_or_realize_func("_push_num_threads", &[types.i64], &[], OMP_MODULE)
                .expect("push num threads func not found");
            push_num_threads = Some(util::call(m, f, vec![t]));
        }
    }
    ForkCallData { fork, push_num_threads }
}

/// `CollapseResult`.
#[derive(Debug, Default)]
pub struct CollapseResult {
    pub collapsed: Option<ValueId>,
    pub setup: Option<ValueId>,
    pub error: String,
}

struct LoopRange {
    loop_: ValueId,
    start: VarId,
    len: VarId,
    step: i64,
}

/// `collapseLoop(parent, v, levels)`.
pub fn collapse_loop(m: &mut Module, parent: FuncId, v: ValueId, levels: i64) -> CollapseResult {
    let fail = |e: &str| CollapseResult { error: e.into(), ..Default::default() };
    if levels < 1 {
        return fail("'collapse' must be at least 1");
    }
    let mut nests = vec![v];
    let mut curr = v;
    for _ in 0..levels - 1 {
        let body = m.as_imperative_for_flow(curr).unwrap().body;
        let s = m.as_series_flow(body).expect("unexpected loop body");
        if s.series.len() != 1 || m.as_imperative_for_flow(s.series[0]).is_none() {
            return fail("loop nest not collapsible");
        }
        curr = s.series[0];
        nests.push(curr);
    }

    let setup = m.nr_series_flow();
    let it = m.get_int_type();
    let len_calc = m.get_or_realize_func("_range_len", &[it, it, it], &[], OMP_MODULE).expect("range length calculation function not found");
    let mut ranges = Vec::new();
    for &l in &nests {
        let f = m.as_imperative_for_flow(l).unwrap().clone();
        let start = util::make_var(m, f.start, setup, Some(parent), false);
        let stop = util::make_var(m, f.end, setup, Some(parent), false);
        let a = vec![m.nr_var_value(start), m.nr_var_value(stop), m.get_int(f.step)];
        let c = util::call(m, len_calc, a);
        let len = util::make_var(m, c, setup, Some(parent), false);
        ranges.push(LoopRange { loop_: l, start, len, step: f.step });
    }

    let mut num_iters = m.get_int(1);
    for r in &ranges {
        let lv = m.nr_var_value(r.len);
        num_iters = bin(m, num_iters, magic::MUL, lv);
    }

    let collapsed_var = m.nr_var(it, false, "");
    m.func_push_symbol(parent, collapsed_var);
    let body = m.nr_series_flow();
    let mut sched = m.as_imperative_for_flow(v).unwrap().schedule.clone().expect("not parallel");
    sched.collapse = 0;
    let zero = m.get_int(0);
    let collapsed = m.nr_imperative_for_flow(zero, 1, num_iters, body, collapsed_var);
    m.as_imperative_for_flow_mut(collapsed).unwrap().schedule = Some(sched);

    // reconstruct indices by successive divmods
    let mut last_div: Option<VarId> = None;
    for r in ranges.iter().rev() {
        let k = last_div.unwrap_or(collapsed_var);
        let (kv, lv) = (m.nr_var_value(k), m.nr_var_value(r.len));
        let d = bin(m, kv, magic::FLOOR_DIV, lv);
        let div = util::make_var(m, d, body, Some(parent), false);
        let (kv, lv) = (m.nr_var_value(k), m.nr_var_value(r.len));
        let md = bin(m, kv, magic::MOD, lv);
        let mod_ = util::make_var(m, md, body, Some(parent), false);
        let mv = m.nr_var_value(mod_);
        let sv = m.get_int(r.step);
        let prod = bin(m, mv, magic::MUL, sv);
        let st = m.nr_var_value(r.start);
        let i = bin(m, st, magic::ADD, prod);
        let lvar = m.as_imperative_for_flow(r.loop_).unwrap().var;
        let a = m.nr_assign(lvar, i);
        m.series_push(body, a);
        last_div = Some(div);
    }

    let old_body = m.as_imperative_for_flow(*nests.last().unwrap()).unwrap().body;
    for x in m.as_series_flow(old_body).unwrap().series.clone() {
        m.series_push(body, x);
    }
    CollapseResult { collapsed: Some(collapsed), setup: Some(setup), error: String::new() }
}

/// Picks the loop template for an imperative parallel loop.
pub fn imperative_template_name(sched: &OMPSched) -> &'static str {
    if sched.gpu {
        "_gpu_loop_outline_template"
    } else if sched.dynamic {
        "_dynamic_loop_outline_template"
    } else if sched.chunk.is_some() {
        "_static_chunked_loop_outline_template"
    } else {
        "_static_loop_outline_template"
    }
}

pub struct OpenMPPass {
    state: OperatorState,
}

impl Default for OpenMPPass {
    fn default() -> Self {
        Self::new()
    }
}

impl OpenMPPass {
    pub const KEY: &'static str = "core-parallel-openmp";

    /// Constructs an OpenMP pass.
    pub fn new() -> Self {
        Self { state: OperatorState::new(true) }
    }

    fn bodied_parent(&self, m: &Module) -> Option<FuncId> {
        self.parent_func(m).filter(|f| m.func(*f).map(|d| d.is_bodied()).unwrap_or(false))
    }

    fn handle_for(&mut self, m: &mut Module, v: ValueId) {
        let parent = self.bodied_parent(m);
        let Some(data) = setup_openmp_transform(m, v, parent, false) else { return };
        let OpenMPTransformData { mut outline, shared_vars, reds } = data;
        let parent = parent.unwrap();
        let f = m.as_for_flow(v).unwrap().clone();
        let sched = *f.schedule.clone().unwrap();
        let loop_var = f.var;
        let types = OMPTypes::new(m);
        let ocall = outline.call.unwrap();

        // separate arguments into 'private' and 'shared'
        let mut shared_redux = Vec::new();
        let (mut privates, mut shareds) = (Vec::new(), Vec::new());
        let mut i = 0;
        for arg in args_of(m, ocall) {
            if is_var_value(m, arg) {
                privates.push(arg);
            } else {
                shareds.push(arg);
                shared_redux.push(reds.reduction(m, shared_vars[i]));
                i += 1;
            }
        }

        let mut cv = CloneVisitor::new(true);

        // We need to pass the task group identifier returned from
        // __kmpc_taskred_modifier_init to the task entry, so append it to private
        // data (initially as null void pointer). Also we add an argument to the end
        // of the outlined function for the gtid.
        if !reds.reductions.is_empty() {
            let null_ptr = m.construct(types.i8ptr, vec![]).expect("null pointer");
            privates.push(null_ptr);

            let of = outline.func.unwrap();
            let oft = var_ty(m, of);
            let mut arg_types = m.func_type_args(oft);
            arg_types.push(m.get_int_type());
            let ret = m.func_type_ret(oft).unwrap();
            let old_args = m.func_args(of);
            let mut names: Vec<String> = old_args.iter().map(|a| m.var_name(*a).to_string()).collect();
            names.push("gtid".into());
            let nname = util::derived_func_name(m, "__outlined_new", of, None);
            let nf = m.nr_bodied_func(nname);
            let nft = m.get_func_type(ret, arg_types, false);
            m.realize_func(nf, nft, &names);
            let new_args = m.func_args(nf);
            let remaps: FxHashMap<Id, VarId> =
                old_args.iter().zip(&new_args).map(|(o, n)| (m.var_uid(*o), *n)).collect();
            let ob = m.func_body(of).unwrap();
            let nb = cv.clone_value_into(m, ob, Some(nf), &remaps);
            m.set_func_body(nf, Some(nb));

            // update outline struct
            outline.func = Some(nf);
            let nfv = m.nr_var_value(nf);
            let zero = m.get_int(0);
            let c = m.as_call_instr_mut(ocall).unwrap();
            c.callee = nfv;
            c.args.push(zero);
            outline.arg_kinds.push(ArgKind::Constant);
        }

        let privs_tuple = tuple(m, &privates);
        let shareds_tuple = tuple(m, &shareds);

        // template call
        let itt = m.value_type_mut(f.iter);
        let ptt = m.value_type_mut(privs_tuple);
        let stt = m.value_type_mut(shareds_tuple);
        let tt = m.get_tuple_type(vec![itt, ptt, stt]);
        let ttp = m.get_pointer_type(Some(tt));
        let tf = m
            .get_or_realize_func("_task_loop_outline_template", &[types.i32ptr, types.i32ptr, ttp], &[], OMP_MODULE)
            .expect("task loop outline template not found");
        let tf = cv.force_clone_var(m, tf);
        let name = util::derived_func_name(m, &m.var_name(tf).to_string(), outline.func.unwrap(), None);
        m.set_name(tf.into(), name);
        let core = ParallelCore::new(tf, ocall, loop_var);
        let mut rep = TaskLoopRoutineStubReplacer::new(m, core, privates, shareds, &shared_redux);
        rep.process(m, NodeRef::Var(tf));
        let raw = ptr_from_func(m, tf);

        let fd = create_fork_call(m, &types, raw, &[f.iter, privs_tuple, shareds_tuple], &sched);
        if let Some(p) = fd.push_num_threads {
            self.insert_before(m, p);
        }
        let _ = parent;
        m.replace_all_value(v, fd.fork);
    }

    fn handle_imperative(&mut self, m: &mut Module, mut v: ValueId) {
        let parent = self.bodied_parent(m);

        let collapse = m.as_imperative_for_flow(v).unwrap().schedule.as_ref().map(|s| s.collapse).unwrap_or(0);
        if collapse != 0 {
            if let Some(p) = parent {
                let res = collapse_loop(m, p, v, collapse);
                if let Some(c) = res.collapsed {
                    m.replace_all_value(v, c);
                    v = c;
                    self.insert_before(m, res.setup.unwrap());
                } else if !res.error.is_empty() {
                    warn(m, &format!("could not collapse loop: {}", res.error), v);
                }
            }
        }

        let gpu = m.as_imperative_for_flow(v).unwrap().schedule.as_ref().map(|s| s.gpu).unwrap_or(false);
        let Some(data) = setup_openmp_transform(m, v, parent, gpu) else { return };
        let OpenMPTransformData { outline, shared_vars, reds } = data;
        let f = m.as_imperative_for_flow(v).unwrap().clone();
        let sched = *f.schedule.clone().unwrap();
        let loop_var = f.var;
        let types = OMPTypes::new(m);
        let ocall = outline.call.unwrap();
        let ofunc = outline.func.unwrap();

        // we disable shared vars for GPU loops
        assert!(!(sched.gpu && !shared_vars.is_empty()), "GPU-parallel loop had shared vars");

        // gather extra arguments
        let loop_uid = m.var_uid(loop_var);
        let mut extra_args = Vec::new();
        let mut extra_types = Vec::new();
        for arg in args_of(m, ocall) {
            if m.var_uid(var_from_outlined_arg(m, arg)) != loop_uid {
                extra_args.push(arg);
                extra_types.push(m.value_type_mut(arg));
            }
        }

        let template_name = imperative_template_name(&sched);

        if sched.gpu {
            let gpu_attr = mangle::func(GPU_MODULE, "kernel");
            let kernels_of = |m: &Module| -> Vec<FuncId> {
                m.vars()
                    .into_iter()
                    .filter(|x| {
                        m.func(*x).map(|d| d.is_bodied()).unwrap_or(false) && util::has_attribute(m, *x, &gpu_attr)
                    })
                    .collect()
            };
            let kernels: FxHashSet<Id> = kernels_of(m).into_iter().map(|k| m.var_uid(k)).collect();
            let ett = m.get_tuple_type(extra_types);
            let instance = GPU_INSTANCE.with(|c| {
                let v = c.get();
                c.set(v + 1);
                v
            });
            let tf = m.get_or_realize_func(template_name, &[types.i64, types.i64, ett], &[Generic::Static(instance)], GPU_MODULE);
            let Some(tf) = tf else {
                warn(m, "loop not compilable for GPU; ignoring", v);
                unpar(m, v);
                return;
            };

            let mut kernel = None;
            for k in kernels_of(m) {
                if !kernels.contains(&m.var_uid(k)) {
                    assert!(kernel.is_none(), "multiple new kernels found after instantiation");
                    kernel = Some(k);
                }
            }
            let kernel = kernel.expect("no new kernel found");
            let mut brep =
                GPULoopBodyStubReplacer { state: OperatorState::new(false), replacement: Some(ocall), loop_var, step: f.step };
            brep.process(m, NodeRef::Var(kernel));

            let mut cv = CloneVisitor::new(true);
            let tf = cv.force_clone_var(m, tf);
            let mut rep = GPULoopTemplateReplacer { state: OperatorState::new(false), step: f.step };
            rep.process(m, NodeRef::Var(tf));
            let et = tuple(m, &extra_args);
            let c = util::call(m, tf, vec![f.start, f.end, et]);
            m.replace_all_value(v, c);
        } else {
            let ett = m.get_tuple_type(extra_types);
            let tt = m.get_tuple_type(vec![types.i64, types.i64, types.i64, ett]);
            let ttp = m.get_pointer_type(Some(tt));
            let tf = m
                .get_or_realize_func(template_name, &[types.i32ptr, types.i32ptr, ttp], &[], OMP_MODULE)
                .expect("imperative loop outline template not found");
            let mut cv = CloneVisitor::new(true);
            let tf = cv.force_clone_var(m, tf);
            let name = util::derived_func_name(m, &m.var_name(tf).to_string(), ofunc, None);
            m.set_name(tf.into(), name);
            let mut rep = ImperativeLoopTemplateReplacer {
                state: OperatorState::new(false),
                core: ParallelCore::new(tf, ocall, loop_var),
                reds: &reds,
                sched: sched.clone(),
                step: f.step,
            };
            rep.process(m, NodeRef::Var(tf));
            let raw = ptr_from_func(m, tf);

            let chunk = match sched.chunk {
                Some(c) if {
                    let ct = m.value_type_mut(c);
                    m.type_is(ct, types.i64)
                } =>
                {
                    c
                }
                _ => m.get_int(1),
            };
            let mut fork_extra = vec![chunk, f.start, f.end];
            fork_extra.extend(extra_args);
            let fd = create_fork_call(m, &types, raw, &fork_extra, &sched);
            if let Some(p) = fd.push_num_threads {
                self.insert_before(m, p);
            }
            m.replace_all_value(v, fd.fork);
        }
    }
}

impl Operator for OpenMPPass {
    op_state!();
    fn handle_for_flow(&mut self, m: &mut Module, v: ValueId) {
        self.handle_for(m, v);
    }
    fn handle_imperative_for_flow(&mut self, m: &mut Module, v: ValueId) {
        self.handle_imperative(m, v);
    }
}

impl Pass for OpenMPPass {
    fn key(&self) -> String {
        Self::KEY.to_string()
    }
    fn run(&mut self, m: &mut Module, _ctx: &dyn PassContext) {
        run_operator(self, m);
    }
}

#[cfg(test)]
mod tests;
