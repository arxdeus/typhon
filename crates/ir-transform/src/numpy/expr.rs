//!
//! Note: the reference implementation `NumPyExpr` is a tree of `unique_ptr`s and maps are keyed by
//! `NumPyExpr*`. Here all nodes of one function live in an [`ExprArena`] and are
//! referenced by [`ExprId`] (stable identity, like the the reference implementation pointers). `replace`
//! copies the node fields (including child ids) and resets the source node.

use rustc_hash::FxHashMap;
use typhon_ir::util::cloning::CloneVisitor;
use typhon_ir::util::irtools::{call, derived_func_name, get_func, make_tuple, make_var, return_type, series, tuple_get};
use typhon_ir::{FuncId, Generic, Module, NodeRef, TypeId, ValueId, VarId, magic};

use super::{FUSION_MODULE, NumPyPrimitiveTypes, NumPyType, TypeCode, gstr, gty};

/// Index of a node in an [`ExprArena`] (`NumPyExpr*`).
pub type ExprId = usize;

macro_rules! ops {
    ($($v:ident => $s:literal, $c:expr;)*) => {
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        pub enum Op { $($v,)* }
        impl Op {
            /// `opstring()`.
            pub fn opstring(self) -> &'static str { match self { $(Op::$v => $s,)* } }
            /// `opcost()`.
            pub fn opcost(self) -> i64 { match self { $(Op::$v => $c,)* } }
        }
    };
}

ops! {
    None => "a", 0;
    Pos => "pos", 0;
    Neg => "neg", 0;
    Invert => "invert", 0;
    Abs => "abs", 1;
    Transpose => "transpose", 0;
    Add => "add", 1;
    Sub => "sub", 1;
    Mul => "mul", 1;
    Matmul => "matmul", 20;
    TrueDiv => "true_div", 8;
    FloorDiv => "floor_div", 8;
    Mod => "mod", 8;
    Fmod => "fmod", 8;
    Pow => "pow", 8;
    Lshift => "lshift", 1;
    Rshift => "rshift", 1;
    And => "and", 1;
    Or => "or", 1;
    Xor => "xor", 1;
    LogicalAnd => "logical_and", 1;
    LogicalOr => "logical_or", 1;
    LogicalXor => "logical_xor", 1;
    Eq => "eq", 1;
    Ne => "ne", 1;
    Lt => "lt", 1;
    Le => "le", 1;
    Gt => "gt", 1;
    Ge => "ge", 1;
    Min => "minimum", 3;
    Max => "maximum", 3;
    Fmin => "fmin", 3;
    Fmax => "fmax", 3;
    Sin => "sin", 10;
    Cos => "cos", 10;
    Tan => "tan", 10;
    Arcsin => "arcsin", 20;
    Arccos => "arccos", 20;
    Arctan => "arctan", 20;
    Arctan2 => "arctan2", 35;
    Hypot => "hypot", 5;
    Sinh => "sinh", 10;
    Cosh => "cosh", 10;
    Tanh => "tanh", 10;
    Arcsinh => "arcsinh", 10;
    Arccosh => "arccosh", 10;
    Arctanh => "arctanh", 10;
    Conj => "conj", 1;
    Exp => "exp", 5;
    Exp2 => "exp2", 5;
    Log => "log", 5;
    Log2 => "log2", 5;
    Log10 => "log10", 5;
    Expm1 => "expm1", 5;
    Log1p => "log1p", 5;
    Sqrt => "sqrt", 2;
    Square => "square", 1;
    Cbrt => "cbrt", 5;
    Logaddexp => "logaddexp", 10;
    Logaddexp2 => "logaddexp2", 10;
    Reciprocal => "reciprocal", 1;
    Rint => "rint", 1;
    Floor => "floor", 1;
    Ceil => "ceil", 1;
    Trunc => "trunc", 1;
    Isnan => "isnan", 1;
    Isinf => "isinf", 1;
    Isfinite => "isfinite", 1;
    Sign => "sign", 1;
    Signbit => "signbit", 1;
    Copysign => "copysign", 1;
    Spacing => "spacing", 1;
    Nextafter => "nextafter", 1;
    Deg2rad => "deg2rad", 2;
    Rad2deg => "rad2deg", 2;
    Heaviside => "heaviside", 3;
    Roll => "roll", 1;
    Cast => "cast", 1;
    ZerosLike => "zeros_like", 1;
    OnesLike => "ones_like", 1;
    Where => "where", 1;
    Clip => "clip", 2;
    ClipMin => "clip_min", 1;
    ClipMax => "clip_max", 1;
    Sum => "sum", 1;
    Mean => "mean", 1;
    Prod => "prod", 1;
    Any => "any", 1;
    All => "all", 1;
    Amin => "amin", 1;
    Amax => "amax", 1;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BroadcastInfo {
    Unknown,
    Yes,
    No,
    Maybe,
}

/// Port of `NumPyExpr` (node data; children are arena ids).
#[derive(Clone, Debug)]
pub struct NumPyExpr {
    pub ty: NumPyType,
    /// `val` (null after the node was moved out by `replace`).
    pub val: Option<ValueId>,
    pub op: Op,
    pub lhs: Option<ExprId>,
    pub rhs: Option<ExprId>,
    pub third: Option<ExprId>,
    /// A leaf owns storage that this consumer may reuse or release. Requires no
    /// remaining alias access, not just allocation provenance; repeated references
    /// must not independently acquire this permission for the same allocation.
    pub owned_last_use: bool,
}

impl NumPyExpr {
    pub fn is_clip(&self) -> bool {
        matches!(self.op, Op::Clip | Op::ClipMin | Op::ClipMax)
    }
    pub fn is_reduction(&self) -> bool {
        matches!(self.op, Op::Sum | Op::Mean | Op::Prod | Op::Any | Op::All | Op::Amin | Op::Amax)
    }
    pub fn is_leaf(&self) -> bool {
        self.lhs.is_none() && self.rhs.is_none() && self.third.is_none()
    }
    pub fn v(&self) -> ValueId {
        self.val.expect("NumPyExpr has no value")
    }
}

/// Arena owning the expression nodes of one function.
#[derive(Clone, Debug, Default)]
pub struct ExprArena {
    pub nodes: Vec<NumPyExpr>,
}

/// Anything that holds an arena (for generic helpers).
pub trait AsArena {
    fn arena(&self) -> &ExprArena;
}
impl AsArena for ExprArena {
    fn arena(&self) -> &ExprArena {
        self
    }
}

impl std::ops::Index<ExprId> for ExprArena {
    type Output = NumPyExpr;
    fn index(&self, i: ExprId) -> &NumPyExpr {
        &self.nodes[i]
    }
}
impl std::ops::IndexMut<ExprId> for ExprArena {
    fn index_mut(&mut self, i: ExprId) -> &mut NumPyExpr {
        &mut self.nodes[i]
    }
}

/// `apply(f)`: pre-order; children are read after `f` ran (so `f` may `replace`).
pub fn apply_in<F: FnMut(&mut ExprArena, ExprId)>(a: &mut ExprArena, e: ExprId, f: &mut F) {
    f(a, e);
    if let Some(l) = a[e].lhs {
        apply_in(a, l, f);
    }
    if let Some(r) = a[e].rhs {
        apply_in(a, r, f);
    }
    if let Some(t) = a[e].third {
        apply_in(a, t, f);
    }
}

fn binop(m: &mut Module, a: ValueId, name: &str, b: ValueId) -> ValueId {
    m.binary_op(a, name, b).unwrap_or_else(|| panic!("operator {name} not found"))
}

/// `getOrRealizeFunc(name, args, generics, FUSION_MODULE)` + `seqassertn`.
pub(crate) fn fusion_func(m: &mut Module, name: &str, args: &[TypeId], generics: &[Generic]) -> FuncId {
    m.get_or_realize_func(name, args, generics, FUSION_MODULE)
        .unwrap_or_else(|| panic!("{FUSION_MODULE}.{name} not found"))
}

pub(crate) fn vtype(m: &mut Module, v: ValueId) -> TypeId {
    m.value_type_mut(v)
}
pub(crate) fn var_ty(m: &Module, v: VarId) -> TypeId {
    m.var_type(v).expect("variable has no type")
}
pub(crate) fn tuple_of(m: &mut Module, args: &[ValueId]) -> ValueId {
    make_tuple(m, args).expect("tuple construction failed")
}
pub(crate) fn call_args(m: &Module, v: ValueId) -> Option<Vec<ValueId>> {
    m.as_call_instr(v).map(|c| c.args.clone())
}
pub(crate) fn callee_func(m: &Module, v: ValueId) -> Option<FuncId> {
    m.as_call_instr(v).and_then(|c| get_func(m, c.callee))
}
fn deref0(m: &mut Module, var: VarId) -> ValueId {
    let vv = m.nr_var_value(var);
    let z = m.get_int(0);
    m.getitem(vv, z).expect("pointer getitem not found")
}

impl ExprArena {
    pub fn new() -> Self {
        Self::default()
    }
    /// `NumPyExpr(type, val)`.
    pub fn leaf(&mut self, ty: NumPyType, val: ValueId) -> ExprId {
        self.node(ty, val, Op::None, None, None, None)
    }
    /// `NumPyExpr(type, val, op, lhs, rhs, third)`.
    pub fn node(
        &mut self,
        ty: NumPyType,
        val: ValueId,
        op: Op,
        lhs: Option<ExprId>,
        rhs: Option<ExprId>,
        third: Option<ExprId>,
    ) -> ExprId {
        self.nodes.push(NumPyExpr { ty, val: Some(val), op, lhs, rhs, third, owned_last_use: false });
        self.nodes.len() - 1
    }

    pub fn is_leaf(&self, e: ExprId) -> bool {
        self[e].is_leaf()
    }

    /// `replace(e)`: moves `src` into `dst` and resets `src`.
    pub fn replace(&mut self, dst: ExprId, src: ExprId) {
        if dst == src {
            return;
        }
        let s = std::mem::replace(
            &mut self.nodes[src],
            NumPyExpr { ty: NumPyType::default(), val: None, op: Op::None, lhs: None, rhs: None, third: None, owned_last_use: false },
        );
        self.nodes[dst] = s;
    }

    /// Pre-order list of the nodes of the tree (snapshot).
    pub fn preorder(&self, e: ExprId) -> Vec<ExprId> {
        let mut out = vec![];
        let mut stack = vec![e];
        while let Some(x) = stack.pop() {
            out.push(x);
            for c in [self[x].third, self[x].rhs, self[x].lhs].into_iter().flatten() {
                stack.push(c);
            }
        }
        out
    }

    pub fn depth(&self, e: ExprId) -> i64 {
        let n = &self[e];
        [n.lhs, n.rhs, n.third].into_iter().flatten().map(|c| self.depth(c)).max().unwrap_or(0) + 1
    }
    pub fn count_nodes(&self, e: ExprId) -> i64 {
        let n = &self[e];
        [n.lhs, n.rhs, n.third].into_iter().flatten().map(|c| self.count_nodes(c)).sum::<i64>() + 1
    }

    /// `haveVectorizedLoop()`.
    pub fn have_vectorized_loop(&self, e: ExprId) -> bool {
        let n = &self[e];
        let lt = n.lhs.map(|l| self[l].ty.dtype);
        let rt = n.rhs.map(|r| self[r].ty.dtype);
        if matches!(lt, Some(TypeCode::ArrC64 | TypeCode::ArrC128)) && n.op.opstring() == "abs" {
            return true;
        }
        let fl = |t: TypeCode| matches!(t, TypeCode::ArrF32 | TypeCode::ArrF64);
        if lt.is_some_and(|t| !fl(t)) || rt.is_some_and(|t| !fl(t)) {
            return false;
        }
        if let (Some(a), Some(b)) = (lt, rt)
            && a != b
        {
            return false;
        }
        // These are the loops available in the runtime library.
        const VEC_LOOPS: [&str; 19] = [
            "arccos", "arccosh", "arcsin", "arcsinh", "arctan", "arctanh", "arctan2", "cos", "exp", "exp2", "expm1", "log",
            "log10", "log1p", "log2", "sin", "sinh", "tanh", "hypot",
        ];
        VEC_LOOPS.contains(&n.op.opstring())
    }

    /// `cost()`.
    pub fn cost(&self, m: &Module, e: ExprId) -> i64 {
        let n = &self[e];
        let mut c = n.op.opcost();
        if c == -1 {
            return -1;
        }
        let lt = n.lhs.map(|l| self[l].ty.dtype);
        // Special-case for abs(complex)
        if n.op == Op::Abs && matches!(lt, Some(TypeCode::ArrC128 | TypeCode::ArrC64)) {
            c = 50;
        }
        // Account for the fact that the vectorized loops are much faster.
        if self.have_vectorized_loop(e) {
            c *= 3;
            if lt == Some(TypeCode::ArrF32) {
                c *= 2;
            }
            c = c.max(64);
        }
        let leaf_const = |x: Option<ExprId>, int: bool| {
            x.is_some_and(|x| {
                self[x].is_leaf()
                    && self[x].val.is_some_and(|v| if int { m.as_int_const(v).is_some() } else { m.as_float_const(v).is_some() })
            })
        };
        let lhs_int = leaf_const(n.lhs, true);
        let rhs_int = leaf_const(n.rhs, true);
        let lhs_const = lhs_int || leaf_const(n.lhs, false);
        let rhs_const = rhs_int || leaf_const(n.rhs, false);
        if rhs_const || lhs_const {
            match n.op {
                Op::TrueDiv | Op::FloorDiv | Op::Mod | Op::Fmod => c = 1,
                Op::Pow => {
                    if rhs_int {
                        c = if m.as_int_const(self[n.rhs.unwrap()].v()).unwrap().val == 2 { 1 } else { 5 };
                    }
                }
                _ => {}
            }
        }
        for child in [n.lhs, n.rhs, n.third].into_iter().flatten() {
            let cc = self.cost(m, child);
            if cc == -1 {
                return -1;
            }
            c += cc;
        }
        c
    }

    /// `rollAxis()`.
    pub fn roll_axis(&self, m: &Module, e: ExprId) -> i64 {
        let args = call_args(m, self[e].v()).expect("roll is not a call");
        let axis = m.as_int_const(*args.last().unwrap()).expect("roll axis is not constant").val;
        if axis < 0 { axis + self[e].ty.ndim } else { axis }
    }

    fn dump(&self, m: &Module, e: ExprId, os: &mut String, level: usize, leaf_id: &mut i32) {
        let n = &self[e];
        os.push_str(&"  ".repeat(level));
        if n.op == Op::None {
            os.push_str(&format!("\x1b[1;36m{}{}", n.op.opstring(), leaf_id));
            *leaf_id += 1;
        } else {
            os.push_str(&format!("\x1b[1;33m{}", n.op.opstring()));
        }
        os.push_str(&format!("\x1b[0m <{}>", n.ty));
        if n.op != Op::None {
            os.push_str(&format!(" \x1b[1;35m[cost={}]\x1b[0m", self.cost(m, e)));
        }
        os.push('\n');
        for c in [n.lhs, n.rhs, n.third].into_iter().flatten() {
            self.dump(m, c, os, level + 1, leaf_id);
        }
    }

    /// `str()` / `operator<<`.
    pub fn str(&self, m: &Module, e: ExprId) -> String {
        let mut s = String::new();
        let mut id = 0;
        self.dump(m, e, &mut s, 0, &mut id);
        s
    }

    /// `getBroadcastInfo()`.
    pub fn broadcast_info(&self, m: &Module, e: ExprId) -> BroadcastInfo {
        let mut arr_dim = -1i64;
        let mut var_leaf: Option<VarId> = None;
        let mut multiple_leaf_vars = false;
        let mut num_non_var_leaf_arrays = 0;
        let mut definitely = false;
        for x in self.preorder(e) {
            let n = &self[x];
            if n.is_leaf() && n.ty.is_array() {
                if arr_dim == -1 {
                    arr_dim = n.ty.ndim;
                } else if arr_dim != n.ty.ndim {
                    definitely = true;
                }
                if let Some(v) = n.val.and_then(|v| m.as_var_value(v)) {
                    let var = m.actual_var(v.var);
                    match var_leaf {
                        Some(vl) => {
                            if vl != var {
                                multiple_leaf_vars = true
                            }
                        }
                        None => var_leaf = Some(var),
                    }
                } else {
                    num_non_var_leaf_arrays += 1;
                }
            }
        }
        let might = num_non_var_leaf_arrays > 1 || multiple_leaf_vars || (num_non_var_leaf_arrays == 1 && var_leaf.is_some());
        if definitely {
            BroadcastInfo::Yes
        } else if might {
            BroadcastInfo::Maybe
        } else {
            BroadcastInfo::No
        }
    }
}

// ---------------------------------------------------------------------------
// type coercion (anonymous namespace)
// ---------------------------------------------------------------------------

fn coerce_scalar_array(scalar: NumPyType, array: NumPyType, t: &NumPyPrimitiveTypes) -> TypeId {
    use TypeCode::*;
    let xtype = scalar.dtype;
    let atype = array.dtype;
    let (mut a_int, mut a_float, mut a_complex) = (false, false, false);
    match atype {
        ArrBool => {}
        ArrI8 | ArrU8 | ArrI16 | ArrU16 | ArrI32 | ArrU32 | ArrI64 | ArrU64 => a_int = true,
        ArrF16 | ArrF32 | ArrF64 => a_float = true,
        ArrC64 | ArrC128 => a_complex = true,
        _ => panic!("unexpected type"),
    }
    let x_int = xtype == Bool || xtype == I64;
    let x_float = xtype == F64;
    let x_complex = xtype == C128;
    let should_cast = (x_int && (a_int || a_float || a_complex)) || (x_float && (a_float || a_complex)) || (x_complex && a_complex);
    if (atype == ArrF16 || atype == ArrF32) && xtype == C128 {
        t.c64.expect("complex64 type not available")
    } else if should_cast {
        array.ir_base_type(t)
    } else {
        scalar.ir_base_type(t)
    }
}

/// `isPythonScalar(t)`.
pub fn is_python_scalar(t: NumPyType) -> bool {
    !t.is_array() && matches!(t.dtype, TypeCode::Bool | TypeCode::I64 | TypeCode::F64 | TypeCode::C128)
}

/// `decideTypes(expr, lhs, rhs, T)`.
pub(crate) fn decide_types(m: &mut Module, op: Op, ty: NumPyType, lhs: NumPyType, rhs: NumPyType, t: &NumPyPrimitiveTypes) -> TypeId {
    // Special case(s)
    if op == Op::Copysign {
        return ty.ir_base_type(t);
    }
    if lhs.is_array() && is_python_scalar(rhs) {
        return coerce_scalar_array(rhs, lhs, t);
    }
    if is_python_scalar(lhs) && rhs.is_array() {
        return coerce_scalar_array(lhs, rhs, t);
    }
    let t1 = lhs.ir_base_type(t);
    let t2 = rhs.ir_base_type(t);
    let f = fusion_func(m, "_coerce", &[], &[gty(t1), gty(t2)]);
    return_type(m, f).expect("coerce func has no return type")
}

// ---------------------------------------------------------------------------
// codegen
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct CodegenContext {
    pub series: ValueId,
    pub func: FuncId,
    pub vars: FxHashMap<ExprId, VarId>,
    pub layouts: FxHashMap<ExprId, VarId>,
    pub types: NumPyPrimitiveTypes,
}

impl CodegenContext {
    pub fn new(series: ValueId, func: FuncId, types: NumPyPrimitiveTypes) -> Self {
        Self { series, func, vars: FxHashMap::default(), layouts: FxHashMap::default(), types }
    }
    fn mkvar(&self, m: &mut Module, v: ValueId) -> VarId {
        make_var(m, v, self.series, Some(self.func), false)
    }
}

fn new_scalar_func(m: &mut Module, val: Option<ValueId>, parent: FuncId, ty: TypeId, names: Vec<String>) -> FuncId {
    let name = derived_func_name(m, "__numpy_fusion_scalar_fn", parent, val.map(NodeRef::Value));
    let f = m.nr_bodied_func(name);
    if let Some(v) = val {
        let s = m.src_info(v.into());
        m.set_src_info(f.into(), s);
    }
    m.realize_func(f, ty, &names);
    f
}

impl ExprArena {
    /// `codegenBroadcasts(C)`.
    pub fn codegen_broadcasts(&mut self, m: &mut Module, c: &mut CodegenContext, e: ExprId) -> ValueId {
        let mut target: Option<ValueId> = None;
        let mut result: Option<ValueId> = None;
        for x in self.preorder(e) {
            if self[x].is_leaf() && self[x].ty.is_array() {
                let var = *c.vars.get(&x).expect("NumPyExpr not found in vars map (codegen broadcasts)");
                let vt = var_ty(m, var);
                let shape = fusion_func(m, "_shape", &[vt], &[]);
                let vv = m.nr_var_value(var);
                let leaf_shape = call(m, shape, vec![vv]);
                match target {
                    None => target = Some(leaf_shape),
                    Some(ts) => {
                        let diff = binop(m, ts, magic::NE, leaf_shape);
                        result = Some(match result {
                            Some(r) => binop(m, r, magic::OR, diff),
                            None => diff,
                        });
                    }
                }
            }
        }
        result.unwrap_or_else(|| m.get_bool(false))
    }

    /// `codegenLayout(C)`: plan shapes and strides without materializing producer
    /// arrays. Fusion needs these descriptors to preserve layout-dependent traversal
    /// and reduction order.
    pub fn codegen_layout(&mut self, m: &mut Module, c: &mut CodegenContext, e: ExprId) -> VarId {
        if self.is_leaf(e) {
            return *c.vars.get(&e).expect("NumPyExpr not found in vars map (layout)");
        }
        if let Some(v) = c.layouts.get(&e) {
            return *v;
        }
        let n = self[e].clone();
        let base = n.ty.ir_base_type(&c.types);
        let mut operands: Vec<ValueId> = vec![];
        if n.op == Op::Matmul || n.op == Op::Transpose {
            let l = self.codegen_layout(m, c, n.lhs.unwrap());
            operands.push(m.nr_var_value(l));
            if let Some(r) = n.rhs {
                let r = self.codegen_layout(m, c, r);
                operands.push(m.nr_var_value(r));
            }
            let tys: Vec<TypeId> = operands.iter().map(|o| vtype(m, *o)).collect();
            let f = if n.op == Op::Matmul {
                fusion_func(m, "_matmul_layout", &tys, &[gty(base)])
            } else {
                fusion_func(m, "_transpose", &tys, &[])
            };
            let cl = call(m, f, operands);
            let v = c.mkvar(m, cl);
            c.layouts.insert(e, v);
            return v;
        }
        if n.op == Op::Where {
            let mut tys = vec![];
            for o in [n.lhs, n.rhs, n.third].into_iter().flatten() {
                let l = self.codegen_layout(m, c, o);
                operands.push(m.nr_var_value(l));
                tys.push(var_ty(m, l));
            }
            let f = fusion_func(m, "_where_layout", &tys, &[gty(base)]);
            let cl = call(m, f, operands);
            let v = c.mkvar(m, cl);
            c.layouts.insert(e, v);
            return v;
        }
        for o in [n.lhs, n.rhs, n.third].into_iter().flatten() {
            if self[o].ty.is_array() {
                let l = self.codegen_layout(m, c, o);
                operands.push(m.nr_var_value(l));
            }
        }
        let f;
        if n.op == Op::Roll {
            operands.push(m.get_bool(true));
            let t0 = vtype(m, operands[0]);
            let bt = m.get_bool_type();
            f = fusion_func(m, "_producer_layout", &[t0, bt], &[gty(base), gstr(n.op.opstring()), gstr("K")]);
        } else if matches!(n.op, Op::Cast | Op::ZerosLike | Op::OnesLike) {
            let args = call_args(m, n.v()).expect("producer is not a call");
            let order = m.as_string_const(args[1]).expect("order is not a string constant").val.clone();
            let copy = if args.len() == 3 { m.as_bool_const(*args.last().unwrap()).map(|b| b.val) } else { None };
            operands.push(m.get_bool(copy.unwrap_or(true)));
            let t0 = vtype(m, operands[0]);
            let bt = m.get_bool_type();
            f = fusion_func(m, "_producer_layout", &[t0, bt], &[gty(base), gstr(n.op.opstring()), gstr(order)]);
        } else {
            let arrays = tuple_of(m, &operands);
            operands = vec![arrays];
            let at = vtype(m, arrays);
            f = fusion_func(m, if n.is_clip() { "_clip_layout" } else { "_layout" }, &[at], &[gty(base)]);
        }
        let cl = call(m, f, operands);
        let v = c.mkvar(m, cl);
        c.layouts.insert(e, v);
        v
    }

    /// `codegenFusedEval(C, destination)`: build a scalar callback for one element,
    /// then let a stdlib loop helper handle broadcasting, strides and allocation (or
    /// reduction / writing into a destination).
    pub fn codegen_fused_eval(&mut self, m: &mut Module, c: &mut CodegenContext, e: ExprId, destination: Option<VarId>) -> VarId {
        let me = self[e].clone();
        let reduction = me.is_reduction();
        let element = if reduction { me.lhs.unwrap() } else { e };
        let types = c.types;

        let mut roll_inputs: FxHashMap<ExprId, ExprId> = FxHashMap::default();
        let mut leaves: Vec<(ExprId, VarId)> = vec![];
        for x in self.preorder(element) {
            if self[x].op == Op::Roll {
                roll_inputs.insert(self[x].lhs.unwrap(), x);
            }
        }
        for x in self.preorder(element) {
            if self[x].is_leaf() {
                let var = *c.vars.get(&x).expect("NumPyExpr not found in vars map (fused eval)");
                leaves.push((x, var));
            }
        }

        let mut arrays = vec![];
        let mut shifts = vec![];
        let mut arg_names: Vec<String> = vec![];
        let mut arg_types: Vec<TypeId> = vec![];
        let mut arg_map: FxHashMap<ExprId, VarId> = FxHashMap::default();
        let mut extra = vec![];
        let mut extra_map: FxHashMap<ExprId, usize> = FxHashMap::default();

        let el_ty = self[element].ty;
        let base = el_ty.ir_base_type(&types);
        if !reduction {
            arg_names.push("out".into());
            arg_types.push(m.get_pointer_type(Some(base)));
        }
        let mut arg_idx = 0usize;
        let mut extra_idx = 0usize;
        for &(x, var) in &leaves {
            if self[x].ty.is_array() {
                arrays.push(m.nr_var_value(var));
                if !roll_inputs.is_empty() {
                    let mut dims: Vec<ValueId> = (0..el_ty.ndim).map(|_| m.get_int(0)).collect();
                    if let Some(&roll) = roll_inputs.get(&x) {
                        let axis = self.roll_axis(m, roll) + el_ty.ndim - self[roll].ty.ndim;
                        let rv = *c.vars.get(&self[roll].rhs.unwrap()).expect("roll shift not bound");
                        dims[axis as usize] = m.nr_var_value(rv);
                    }
                    shifts.push(tuple_of(m, &dims));
                }
                arg_names.push(format!("in{arg_idx}"));
                arg_idx += 1;
                let bt = self[x].ty.ir_base_type(&types);
                arg_types.push(m.get_pointer_type(Some(bt)));
            } else {
                extra.push(m.nr_var_value(var));
                extra_map.insert(x, extra_idx);
                extra_idx += 1;
            }
        }
        let extra_tuple = tuple_of(m, &extra);
        arg_names.push("extra".into());
        arg_types.push(vtype(m, extra_tuple));
        let ret = if reduction { base } else { m.get_none_type() };
        let fty = m.get_func_type(ret, arg_types.clone(), false);
        let scalar_func = new_scalar_func(m, me.val, c.func, fty, arg_names);
        let fargs = m.func_args(scalar_func);

        let mut ai = if reduction { 0 } else { 1 };
        for &(x, _) in &leaves {
            if self[x].ty.is_array() {
                arg_map.insert(x, fargs[ai]);
                ai += 1;
            }
        }
        let scalar_expr = self.codegen_scalar_expr(m, c, element, &arg_map, &extra_map, *fargs.last().unwrap());
        if reduction {
            let st = vtype(m, scalar_expr);
            let cast_f = fusion_func(m, "_cast", &[st], &[gty(base)]);
            let cv = call(m, cast_f, vec![scalar_expr]);
            let r = m.nr_return(Some(cv));
            let body = series(m, &[r]).unwrap();
            m.set_func_body(scalar_func, Some(body));
        } else {
            let ot = vtype(m, scalar_expr);
            let ptrset = fusion_func(m, "_ptrset", &[arg_types[0], base], &[]);
            let _ = ot;
            let o = m.nr_var_value(fargs[0]);
            let cl = call(m, ptrset, vec![o, scalar_expr]);
            let body = series(m, &[cl]).unwrap();
            m.set_func_body(scalar_func, Some(body));
        }

        let mut callback = m.nr_var_value(scalar_func);
        if reduction {
            let sft = var_ty(m, scalar_func);
            let fname = m.var_name(scalar_func).to_string();
            let wrapper = fusion_func(m, "_reduction_function", &[sft], &[gstr(fname)]);
            callback = call(m, wrapper, vec![callback]);
        }
        let arrays_tuple = tuple_of(m, &arrays);
        let mut loop_args = vec![arrays_tuple, callback, extra_tuple];
        let mut loop_types = vec![vtype(m, arrays_tuple), vtype(m, callback), vtype(m, extra_tuple)];
        let mut loop_generics = vec![gty(base)];
        if reduction {
            let initial = if let Some(r) = me.rhs {
                let v = *c.vars.get(&r).expect("initial not bound");
                m.nr_var_value(v)
            } else if me.op == Op::Mean {
                let it = m.get_int_type();
                let cast_f = fusion_func(m, "_cast", &[it], &[gty(me.ty.ir_base_type(&types))]);
                let z = m.get_int(0);
                call(m, cast_f, vec![z])
            } else {
                tuple_of(m, &[])
            };
            loop_args.push(initial);
            loop_types.push(vtype(m, initial));
            loop_generics = vec![gty(base), gty(me.ty.ir_base_type(&types)), gstr(me.op.opstring())];
        }
        let mut needs_layout = !roll_inputs.is_empty() || (destination.is_none() && c.layouts.contains_key(&element));
        for x in self.preorder(element) {
            let n = &self[x];
            if n.op == Op::Where || n.is_clip() || (n.ty.ndim > 1 && matches!(n.op, Op::Cast | Op::ZerosLike | Op::OnesLike)) {
                needs_layout = true;
            }
        }
        let mut loop_name = if reduction { "_loop_reduce".to_string() } else { "_loop_alloc".to_string() };
        let axis = if reduction { Some(call_args(m, me.v()).expect("reduction is not a call")[1]) } else { None };
        let axis_reduction = reduction && (me.ty.is_array() || {
            let at = vtype(m, axis.unwrap());
            !m.type_is(at, types.none)
        });
        if needs_layout || axis_reduction {
            let layout = self.codegen_layout(m, c, element);
            loop_args.push(m.nr_var_value(layout));
            loop_types.push(var_ty(m, layout));
            loop_name += "_layout";
        }
        if !roll_inputs.is_empty() {
            let st = tuple_of(m, &shifts);
            loop_args.push(st);
            loop_types.push(vtype(m, st));
            loop_name = "_loop_alloc_roll".into();
        }
        if axis_reduction {
            let axis = axis.unwrap();
            let cl = CloneVisitor::new(true).clone_value(m, axis);
            loop_args.push(cl);
            loop_types.push(vtype(m, axis));
            loop_generics.push(Generic::Static(me.ty.ndim));
            loop_name = "_loop_reduce_axis".into();
        }
        if let Some(d) = destination {
            loop_args.push(m.nr_var_value(d));
            loop_types.push(var_ty(m, d));
            loop_generics.clear();
            loop_name = "_loop_into".into();
        }
        let loop_func = fusion_func(m, &loop_name, &loop_types, &loop_generics);
        let cl = call(m, loop_func, loop_args);
        let result = c.mkvar(m, cl);

        // Release only after the entire fused loop. Ownership proofs must exclude
        // duplicate owners: this walks expression leaves, not unique data pointers.
        for x in self.preorder(e) {
            if self[x].is_leaf() && self[x].owned_last_use {
                let var = *c.vars.get(&x).expect("NumPyExpr not found in vars map (fused eval)");
                let mut free_args = vec![m.nr_var_value(var)];
                let mut free_types = vec![var_ty(m, var)];
                let copy_src = needs_layout && !reduction;
                if copy_src {
                    free_args.push(m.nr_var_value(result));
                    free_types.push(var_ty(m, result));
                }
                let f = fusion_func(m, if copy_src { "_free_copy_source" } else { "_free" }, &free_types, &[]);
                let cl = call(m, f, free_args);
                m.series_push(c.series, cl);
            }
        }
        result
    }

    fn free_array(m: &mut Module, arr: VarId) -> ValueId {
        let t = var_ty(m, arr);
        let f = fusion_func(m, "_free", &[t], &[]);
        let vv = m.nr_var_value(arr);
        call(m, f, vec![vv])
    }

    /// `codegenSequentialEval(C)`: evaluate whole-array operations one node at a time
    /// when fusion is not chosen. Owned temporaries can still be recycled if their
    /// dtype, rank and runtime shape fit.
    pub fn codegen_sequential_eval(&mut self, m: &mut Module, c: &mut CodegenContext, e: ExprId) -> VarId {
        if self.is_leaf(e) {
            return *c.vars.get(&e).expect("NumPyExpr not found in vars map (codegen sequential eval)");
        }
        let types = c.types;
        let n = self[e].clone();
        let lhs = n.lhs.unwrap();
        let lv = self.codegen_sequential_eval(m, c, lhs);
        let rv = n.rhs.map(|r| self.codegen_sequential_eval(m, c, r));
        let ln = self[lhs].clone();
        let rn = n.rhs.map(|r| self[r].clone());

        let lfreeable = ln.ty.is_array() && (ln.owned_last_use || !ln.is_leaf());
        let rfreeable = rn.as_ref().is_some_and(|r| r.ty.is_array() && (r.owned_last_use || !r.is_leaf()));
        let ltmp = lfreeable && ln.ty.dtype == n.ty.dtype && ln.ty.ndim == n.ty.ndim;
        let rtmp = rfreeable && rn.as_ref().is_some_and(|r| r.ty.dtype == n.ty.dtype && r.ty.ndim == n.ty.ndim);
        let series_v = c.series;
        let push = |m: &mut Module, v: ValueId| m.series_push(series_v, v);

        if n.op == Op::Roll {
            let cargs = call_args(m, n.v()).unwrap();
            let callee = callee_func(m, n.v()).unwrap();
            let a = m.nr_var_value(lv);
            let b = m.nr_var_value(rv.unwrap());
            let cl = call(m, callee, vec![a, b, *cargs.last().unwrap()]);
            let result = c.mkvar(m, cl);
            if lfreeable {
                let f = Self::free_array(m, lv);
                push(m, f);
            }
            return result;
        }

        if n.op == Op::Where || n.is_clip() {
            let third_value = n.third.map(|t| self.codegen_sequential_eval(m, c, t));
            let mut args = call_args(m, n.v()).unwrap();
            let callee = callee_func(m, n.v()).unwrap();
            args[0] = m.nr_var_value(lv);
            args[if n.op == Op::ClipMax { 2 } else { 1 }] = m.nr_var_value(rv.unwrap());
            if let Some(tv) = third_value {
                args[2] = m.nr_var_value(tv);
            }
            let cl = call(m, callee, args);
            let result = c.mkvar(m, cl);
            if lfreeable {
                let f = Self::free_array(m, lv);
                push(m, f);
            }
            if rfreeable {
                let f = Self::free_array(m, rv.unwrap());
                push(m, f);
            }
            if let Some(t) = n.third
                && self[t].ty.is_array()
                && (self[t].owned_last_use || !self[t].is_leaf())
            {
                let f = Self::free_array(m, third_value.unwrap());
                push(m, f);
            }
            return result;
        }

        if n.ty.ndim > 1 && matches!(n.op, Op::Cast | Op::ZerosLike | Op::OnesLike) {
            let mut args = call_args(m, n.v()).unwrap();
            let callee = callee_func(m, n.v()).unwrap();
            args[0] = m.nr_var_value(lv);
            let cl = call(m, callee, args);
            let result = c.mkvar(m, cl);
            if lfreeable {
                let (a, b) = (var_ty(m, lv), var_ty(m, result));
                let f = fusion_func(m, "_free_copy_source", &[a, b], &[]);
                let (x, y) = (m.nr_var_value(lv), m.nr_var_value(result));
                let cl = call(m, f, vec![x, y]);
                push(m, cl);
            }
            return result;
        }

        let like: VarId;
        let out_shape_val: ValueId;
        let shape_of = |m: &mut Module, v: VarId| {
            let t = var_ty(m, v);
            let f = fusion_func(m, "_shape", &[t], &[]);
            let vv = m.nr_var_value(v);
            call(m, f, vec![vv])
        };
        if let Some(rv) = rv {
            // Can't do anything special with matmul here...
            if n.op == Op::Matmul {
                let (a, b) = (var_ty(m, lv), var_ty(m, rv));
                let f = fusion_func(m, "_matmul", &[a, b], &[]);
                let (x, y) = (m.nr_var_value(lv), m.nr_var_value(rv));
                let cl = call(m, f, vec![x, y]);
                let ret = c.mkvar(m, cl);
                if lfreeable {
                    let f = Self::free_array(m, lv);
                    push(m, f);
                }
                if rfreeable {
                    let f = Self::free_array(m, rv);
                    push(m, f);
                }
                return ret;
            }
            let ls = shape_of(m, lv);
            let rs = shape_of(m, rv);
            let (lt, rt) = (vtype(m, ls), vtype(m, rs));
            let f = fusion_func(m, "_broadcast", &[lt, rt], &[]);
            like = if rn.as_ref().unwrap().ty.ndim > ln.ty.ndim { rv } else { lv };
            out_shape_val = call(m, f, vec![ls, rs]);
        } else {
            like = lv;
            out_shape_val = shape_of(m, lv);
        }
        let out_shape = c.mkvar(m, out_shape_val);
        let t = n.ty.ir_base_type(&types);
        let new_array = |m: &mut Module| {
            let (a, b) = (var_ty(m, like), var_ty(m, out_shape));
            let f = fusion_func(m, "_create", &[a, b], &[gty(t)]);
            let (x, y) = (m.nr_var_value(like), m.nr_var_value(out_shape));
            call(m, f, vec![x, y])
        };
        let cond = |m: &mut Module, c: &CodegenContext, v: VarId| {
            let s = shape_of(m, v);
            let o = m.nr_var_value(out_shape);
            let eq = binop(m, s, magic::EQ, o);
            c.mkvar(m, eq)
        };

        let mut free_left_static = false;
        let mut free_right_static = false;
        let mut lcond = None;
        let mut rcond = None;
        let result: VarId;
        if let Some(rv) = rv {
            let rnd = rn.as_ref().unwrap().ty.ndim;
            if ltmp && rnd == 0 {
                // We are adding lhs temp array to const or 0-dim array, so reuse lhs array.
                result = lv;
            } else if rtmp && ln.ty.ndim == 0 {
                // We are adding rhs temp array to const or 0-dim array, so reuse rhs array.
                result = rv;
            } else if !ltmp && !rtmp {
                // Neither operand is a temp array, so we must allocate a new array.
                let na = new_array(m);
                result = c.mkvar(m, na);
                free_left_static = lfreeable;
                free_right_static = rfreeable;
            } else if ltmp && rtmp {
                // We won't know until runtime if we can reuse the temp array(s) since they
                // might broadcast.
                let lc = cond(m, c, lv);
                let rc = cond(m, c, rv);
                lcond = Some(lc);
                rcond = Some(rc);
                let na = new_array(m);
                let (rcv, rvv) = (m.nr_var_value(rc), m.nr_var_value(rv));
                let inner = m.nr_ternary(rcv, rvv, na);
                let (lcv, lvv) = (m.nr_var_value(lc), m.nr_var_value(lv));
                let arr = m.nr_ternary(lcv, lvv, inner);
                result = c.mkvar(m, arr);
            } else if ltmp {
                let lc = cond(m, c, lv);
                lcond = Some(lc);
                let na = new_array(m);
                let (lcv, lvv) = (m.nr_var_value(lc), m.nr_var_value(lv));
                let arr = m.nr_ternary(lcv, lvv, na);
                result = c.mkvar(m, arr);
                free_right_static = rfreeable;
            } else {
                let rc = cond(m, c, rv);
                rcond = Some(rc);
                let na = new_array(m);
                let (rcv, rvv) = (m.nr_var_value(rc), m.nr_var_value(rv));
                let arr = m.nr_ternary(rcv, rvv, na);
                result = c.mkvar(m, arr);
                free_left_static = lfreeable;
            }
        } else if ltmp {
            result = lv;
        } else {
            let na = new_array(m);
            result = c.mkvar(m, na);
            free_left_static = lfreeable;
        }

        let opstr = n.op.opstring();
        if self.have_vectorized_loop(e) {
            // We have a vectorized loop available for this operations.
            if let Some(rv) = rv {
                let tys = [var_ty(m, lv), var_ty(m, rv), var_ty(m, result)];
                let f = fusion_func(m, "_apply_vectorized_loop_binary", &tys, &[gstr(opstr)]);
                let args = vec![m.nr_var_value(lv), m.nr_var_value(rv), m.nr_var_value(result)];
                let cl = call(m, f, args);
                push(m, cl);
            } else {
                let tys = [var_ty(m, lv), var_ty(m, result)];
                let f = fusion_func(m, "_apply_vectorized_loop_unary", &tys, &[gstr(opstr)]);
                let args = vec![m.nr_var_value(lv), m.nr_var_value(result)];
                let cl = call(m, f, args);
                push(m, cl);
            }
        } else {
            let mut arrays = vec![m.nr_var_value(result)];
            let mut names: Vec<String> = vec!["out".into()];
            let base = n.ty.ir_base_type(&types);
            let mut tys = vec![m.get_pointer_type(Some(base))];
            let mut extra = vec![];
            if ln.ty.is_array() {
                if result != lv {
                    names.push("in0".into());
                    tys.push(m.get_pointer_type(Some(ln.ty.ir_base_type(&types))));
                    arrays.push(m.nr_var_value(lv));
                }
            } else {
                extra.push(m.nr_var_value(lv));
            }
            if let Some(rv) = rv {
                let r = rn.as_ref().unwrap();
                if r.ty.is_array() {
                    if result != rv {
                        names.push("in1".into());
                        tys.push(m.get_pointer_type(Some(r.ty.ir_base_type(&types))));
                        arrays.push(m.nr_var_value(rv));
                    }
                } else {
                    extra.push(m.nr_var_value(rv));
                }
            }
            let extra_tuple = tuple_of(m, &extra);
            names.push("extra".into());
            tys.push(vtype(m, extra_tuple));
            let none = m.get_none_type();
            let fty = m.get_func_type(none, tys.clone(), false);
            let sf = new_scalar_func(m, n.val, c.func, fty, names);
            let fargs = m.func_args(sf);
            let body = m.nr_series_flow();
            let name = format!("_{opstr}");

            if let Some(rv) = rv {
                let r = rn.clone().unwrap();
                let (litem, ritem);
                if ln.ty.is_array() && r.ty.is_array() {
                    if result == lv {
                        litem = deref0(m, fargs[0]);
                        ritem = deref0(m, fargs[1]);
                    } else if result == rv {
                        litem = deref0(m, fargs[1]);
                        ritem = deref0(m, fargs[0]);
                    } else {
                        litem = deref0(m, fargs[1]);
                        ritem = deref0(m, fargs[2]);
                    }
                } else if ln.ty.is_array() {
                    litem = deref0(m, fargs[if result == lv { 0 } else { 1 }]);
                    let x = m.nr_var_value(*fargs.last().unwrap());
                    ritem = tuple_get(m, x, 0);
                } else if r.ty.is_array() {
                    ritem = deref0(m, fargs[if result == rv { 0 } else { 1 }]);
                    let x = m.nr_var_value(*fargs.last().unwrap());
                    litem = tuple_get(m, x, 0);
                } else {
                    panic!("both lhs are rhs are scalars");
                }
                let common = decide_types(m, n.op, n.ty, ln.ty, r.ty, &types);
                let lt = vtype(m, litem);
                let lc = fusion_func(m, "_cast", &[lt], &[gty(common)]);
                let litem = call(m, lc, vec![litem]);
                let rt = vtype(m, ritem);
                let rc = fusion_func(m, "_cast", &[rt], &[gty(common)]);
                let ritem = call(m, rc, vec![ritem]);
                let (a, b) = (vtype(m, litem), vtype(m, ritem));
                let opf = fusion_func(m, &name, &[a, b], &[]);
                let oitem = call(m, opf, vec![litem, ritem]);
                let ot = vtype(m, oitem);
                let ps = fusion_func(m, "_ptrset", &[tys[0], ot], &[]);
                let o = m.nr_var_value(fargs[0]);
                let cl = call(m, ps, vec![o, oitem]);
                m.series_push(body, cl);
            } else {
                let litem = deref0(m, fargs[if result == lv { 0 } else { 1 }]);
                let mut generics = vec![];
                if matches!(n.op, Op::Cast | Op::ZerosLike | Op::OnesLike) {
                    generics.push(gty(base));
                }
                let lt = vtype(m, litem);
                let opf = fusion_func(m, &name, &[lt], &generics);
                let oitem = call(m, opf, vec![litem]);
                let ot = vtype(m, oitem);
                let ps = fusion_func(m, "_ptrset", &[tys[0], ot], &[]);
                let o = m.nr_var_value(fargs[0]);
                let cl = call(m, ps, vec![o, oitem]);
                m.series_push(body, cl);
            }
            m.set_func_body(sf, Some(body));
            let arrays_tuple = tuple_of(m, &arrays);
            let lt = [vtype(m, arrays_tuple), var_ty(m, sf), vtype(m, extra_tuple)];
            let lf = fusion_func(m, "_loop_basic", &lt, &[]);
            let sfv = m.nr_var_value(sf);
            let cl = call(m, lf, vec![arrays_tuple, sfv, extra_tuple]);
            push(m, cl);
        }

        assert!(!(free_left_static && lcond.is_some()), "unexpected free conditions for left arg");
        assert!(!(free_right_static && rcond.is_some()), "unexpected free conditions for right arg");

        let empty = |m: &mut Module| m.nr_series_flow();
        if let (Some(lc), Some(rc)) = (lcond, rcond) {
            let rv = rv.unwrap();
            let fr1 = Self::free_array(m, rv);
            let s1 = series(m, &[fr1]).unwrap();
            let fl = Self::free_array(m, lv);
            let fr2 = Self::free_array(m, rv);
            let s3 = series(m, &[fr2]).unwrap();
            let rcv = m.nr_var_value(rc);
            let e1 = empty(m);
            let inner = m.nr_if_flow(rcv, e1, Some(s3));
            let s2 = series(m, &[fl, inner]).unwrap();
            let lcv = m.nr_var_value(lc);
            let iff = m.nr_if_flow(lcv, s1, Some(s2));
            push(m, iff);
        } else {
            if free_left_static {
                let f = Self::free_array(m, lv);
                push(m, f);
            } else if let Some(lc) = lcond {
                let f = Self::free_array(m, lv);
                let s = series(m, &[f]).unwrap();
                let cv = m.nr_var_value(lc);
                let e1 = empty(m);
                let iff = m.nr_if_flow(cv, e1, Some(s));
                push(m, iff);
            }
            if free_right_static {
                let f = Self::free_array(m, rv.unwrap());
                push(m, f);
            } else if let Some(rc) = rcond {
                let f = Self::free_array(m, rv.unwrap());
                let s = series(m, &[f]).unwrap();
                let cv = m.nr_var_value(rc);
                let e1 = empty(m);
                let iff = m.nr_if_flow(cv, e1, Some(s));
                push(m, iff);
            }
        }
        result
    }

    /// `codegenScalarExpr(C, args, scalarMap, scalars)`.
    pub fn codegen_scalar_expr(
        &mut self,
        m: &mut Module,
        c: &mut CodegenContext,
        e: ExprId,
        args: &FxHashMap<ExprId, VarId>,
        scalar_map: &FxHashMap<ExprId, usize>,
        scalars: VarId,
    ) -> ValueId {
        let n = self[e].clone();
        let types = c.types;
        if n.op == Op::Roll {
            return self.codegen_scalar_expr(m, c, n.lhs.unwrap(), args, scalar_map, scalars);
        }
        let lv = n.lhs.map(|l| self.codegen_scalar_expr(m, c, l, args, scalar_map, scalars));
        let rv = n.rhs.map(|r| self.codegen_scalar_expr(m, c, r, args, scalar_map, scalars));
        let name = format!("_{}", n.op.opstring());

        if n.op == Op::Where || n.is_clip() {
            let mut operands = vec![lv.unwrap(), rv.unwrap()];
            if let Some(t) = n.third {
                operands.push(self.codegen_scalar_expr(m, c, t, args, scalar_map, scalars));
            }
            let tys: Vec<TypeId> = operands.iter().map(|o| vtype(m, *o)).collect();
            let f = fusion_func(m, &name, &tys, &[gty(n.ty.ir_base_type(&types))]);
            return call(m, f, operands);
        }
        match (lv, rv) {
            (Some(lv), Some(rv)) => {
                let (lt, rt) = (self[n.lhs.unwrap()].ty, self[n.rhs.unwrap()].ty);
                let common = decide_types(m, n.op, n.ty, lt, rt, &types);
                let a = vtype(m, lv);
                let c1 = fusion_func(m, "_cast", &[a], &[gty(common)]);
                let b = vtype(m, rv);
                let c2 = fusion_func(m, "_cast", &[b], &[gty(common)]);
                let lv = call(m, c1, vec![lv]);
                let rv = call(m, c2, vec![rv]);
                let (a, b) = (vtype(m, lv), vtype(m, rv));
                let f = fusion_func(m, &name, &[a, b], &[]);
                call(m, f, vec![lv, rv])
            }
            (Some(lv), None) => {
                let t = n.ty.ir_base_type(&types);
                let mut generics = vec![];
                if matches!(n.op, Op::Cast | Op::ZerosLike | Op::OnesLike) {
                    generics.push(gty(t));
                }
                let a = vtype(m, lv);
                let f = fusion_func(m, &name, &[a], &generics);
                call(m, f, vec![lv])
            }
            _ => {
                if n.ty.is_array() {
                    let var = *args.get(&e).expect("NumPyExpr not found in args map (codegen expr)");
                    deref0(m, var)
                } else {
                    let idx = *scalar_map.get(&e).expect("NumPyExpr not found in scalar map (codegen expr)");
                    let sv = m.nr_var_value(scalars);
                    tuple_get(m, sv, idx)
                }
            }
        }
    }
}
