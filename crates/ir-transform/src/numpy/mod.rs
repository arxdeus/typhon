//!
//! File map:
//! - `mod.rs`      : numpy.h shared declarations (`NumPyType`, `NumPyPrimitiveTypes`,
//!                   `NumPyOptimizationUnit`, options, type predicates)
//! - `expr.rs`     : expr.cpp (`NumPyExpr` arena, cost model, codegen)
//! - `numpy.rs`    : numpy.cpp (parser, analysis, fusion/inline/lifetime passes)
//! - `forward.rs`  : forward.cpp (forwarding DAGs)
//! - `indexing.rs` : indexing.{h,cpp} (bounds check elision)

pub mod expr;
pub mod forward;
pub mod indexing;
#[allow(clippy::module_inception)]
pub mod numpy;

use std::cell::RefCell;
use std::fmt;

use rustc_hash::FxHashMap;
use typhon_common::mangle;
use typhon_ir::types::TypeKind;
use typhon_ir::{FuncId, Generic, Module, TypeId, ValueId};

pub use expr::{AsArena, BroadcastInfo, CodegenContext, ExprArena, ExprId, NumPyExpr, Op, apply_in};
pub use forward::{Forwarding, ForwardingDAG, do_forwarding, get_forwarding_dags};
pub use indexing::NumPyBoundsCheckElisionPass;
pub use numpy::{
    NumPyExpressionAnalysis, NumPyExpressionResult, NumPyFunctionExpressions, NumPyFusionPass, NumPyInlinePass,
    NumPyLifetimePass, has_owned_result, parse,
};

/// `FUSION_MODULE`.
pub const FUSION_MODULE: &str = "std.numpy.fusion";

/// The `llvm::cl` options of numpy.cpp (`-npfuse-always`, `-npfuse-never`,
/// `-npfuse-matmul`, `-npfree-threshold`, `-npfuse-verbose`).
///
/// Note: the reference implementation reads these from LLVM's global command line. Here they are a
/// thread-local configuration (defaults identical) that the driver may override with
/// [`set_numpy_options`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NumPyOptions {
    /// Expression cost below which (<=) to always fuse.
    pub always_fuse_cost_threshold: i64,
    /// Expression cost above which (>) to never fuse.
    pub never_fuse_cost_threshold: i64,
    /// Fuse compatible matmul additions and subtractions into BLAS.
    pub fuse_matmul: bool,
    /// Minimum NumPy loop-replacement buffer size in bytes to free (0 always frees).
    pub replacement_free_threshold: u32,
    /// Print information about fused expressions.
    pub verbose: bool,
}

impl Default for NumPyOptions {
    fn default() -> Self {
        Self {
            always_fuse_cost_threshold: 10,
            never_fuse_cost_threshold: 50,
            fuse_matmul: true,
            replacement_free_threshold: 512,
            verbose: false,
        }
    }
}

thread_local! {
    static OPTIONS: RefCell<NumPyOptions> = RefCell::new(NumPyOptions::default());
}

/// Current NumPy optimization options.
pub fn numpy_options() -> NumPyOptions {
    OPTIONS.with(|o| *o.borrow())
}
/// Overrides the NumPy optimization options (for this thread).
pub fn set_numpy_options(opts: NumPyOptions) {
    OPTIONS.with(|o| *o.borrow_mut() = opts);
}

macro_rules! xlog {
    ($($arg:tt)*) => {
        if $crate::numpy::numpy_options().verbose {
            eprintln!($($arg)*);
        }
    };
}
pub(crate) use xlog;

// ---------------------------------------------------------------------------
// Type predicates
// ---------------------------------------------------------------------------

/// `isArrayType(t)`: a realized `std.numpy.ndarray.ndarray[...]` record.
pub fn is_array_type(m: &Module, t: Option<TypeId>) -> bool {
    let Some(t) = t else { return false };
    matches!(m.type_kind(t), TypeKind::Record { .. }) && m.type_name(t).starts_with(&format!("{}[", mangle::stdlib().ndarray))
}

/// `isUFuncType(t)`.
pub fn is_ufunc_type(m: &Module, t: Option<TypeId>) -> bool {
    let Some(t) = t else { return false };
    let n = m.type_name(t);
    n.starts_with(&format!("{}[", mangle::class("std.numpy.ufunc", "UnaryUFunc")))
        || n.starts_with(&format!("{}[", mangle::class("std.numpy.ufunc", "BinaryUFunc")))
}

/// `isNoneType(t, T)`.
pub fn is_none_type(m: &Module, t: TypeId, types: &NumPyPrimitiveTypes) -> bool {
    m.type_is(t, types.none)
}

/// `NumPyPass::shouldRun(module)`: any ndarray type in the module.
pub fn numpy_should_run(m: &Module) -> bool {
    m.types().into_iter().any(|t| is_array_type(m, Some(t)))
}

/// Type equality against an optional primitive (complex types may be missing).
fn is_opt(m: &Module, t: TypeId, other: Option<TypeId>) -> bool {
    other.is_some_and(|o| m.type_is(t, o))
}

// ---------------------------------------------------------------------------
// NumPyPrimitiveTypes / NumPyType
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
pub struct NumPyPrimitiveTypes {
    pub none: TypeId,
    pub bool_: TypeId,
    pub i8: TypeId,
    pub u8: TypeId,
    pub i16: TypeId,
    pub u16: TypeId,
    pub i32: TypeId,
    pub u32: TypeId,
    pub i64: TypeId,
    pub u64: TypeId,
    pub f16: TypeId,
    pub f32: TypeId,
    pub f64: TypeId,
    /// Note: `M->getType(...)` may return null in the reference implementation too; kept optional.
    pub c64: Option<TypeId>,
    pub c128: Option<TypeId>,
}

impl NumPyPrimitiveTypes {
    pub fn new(m: &mut Module) -> Self {
        Self {
            none: m.get_none_type(),
            bool_: m.get_bool_type(),
            i8: m.get_int_type_n(8, true),
            u8: m.get_int_type_n(8, false),
            i16: m.get_int_type_n(16, true),
            u16: m.get_int_type_n(16, false),
            i32: m.get_int_type_n(32, true),
            u32: m.get_int_type_n(32, false),
            i64: m.get_int_type_n(64, true),
            u64: m.get_int_type_n(64, false),
            f16: m.get_float16_type(),
            f32: m.get_float32_type(),
            f64: m.get_float_type(),
            c64: m.get_type(&mangle::stdlib().complex64),
            c128: m.get_type(&mangle::stdlib().complex),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(i32)]
pub enum TypeCode {
    None = -1,
    Bool,
    I8,
    U8,
    I16,
    U16,
    I32,
    U32,
    I64,
    U64,
    F16,
    F32,
    F64,
    C64,
    C128,
    ScalarEnd,
    ArrBool,
    ArrI8,
    ArrU8,
    ArrI16,
    ArrU16,
    ArrI32,
    ArrU32,
    ArrI64,
    ArrU64,
    ArrF16,
    ArrF32,
    ArrF64,
    ArrC64,
    ArrC128,
}

const SCALARS: [TypeCode; 14] = [
    TypeCode::Bool,
    TypeCode::I8,
    TypeCode::U8,
    TypeCode::I16,
    TypeCode::U16,
    TypeCode::I32,
    TypeCode::U32,
    TypeCode::I64,
    TypeCode::U64,
    TypeCode::F16,
    TypeCode::F32,
    TypeCode::F64,
    TypeCode::C64,
    TypeCode::C128,
];
const ARRAYS: [TypeCode; 14] = [
    TypeCode::ArrBool,
    TypeCode::ArrI8,
    TypeCode::ArrU8,
    TypeCode::ArrI16,
    TypeCode::ArrU16,
    TypeCode::ArrI32,
    TypeCode::ArrU32,
    TypeCode::ArrI64,
    TypeCode::ArrU64,
    TypeCode::ArrF16,
    TypeCode::ArrF32,
    TypeCode::ArrF64,
    TypeCode::ArrC64,
    TypeCode::ArrC128,
];

impl TypeCode {
    /// Scalar index 0..14 of a scalar or array code.
    fn element_index(self) -> Option<usize> {
        SCALARS.iter().position(|c| *c == self).or_else(|| ARRAYS.iter().position(|c| *c == self))
    }
    fn name(self) -> &'static str {
        const NAMES: [&str; 14] =
            ["bool", "i8", "u8", "i16", "u16", "i32", "u32", "i64", "u64", "f16", "f32", "f64", "c64", "c128"];
        match self {
            TypeCode::None => "none",
            TypeCode::ScalarEnd => "",
            c => NAMES[c.element_index().unwrap()],
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct NumPyType {
    pub dtype: TypeCode,
    pub ndim: i64,
}

impl Default for NumPyType {
    fn default() -> Self {
        Self { dtype: TypeCode::None, ndim: 0 }
    }
}

impl NumPyType {
    pub fn new(dtype: TypeCode, ndim: i64) -> Self {
        assert!(ndim >= 0, "ndim must be non-negative");
        Self { dtype, ndim }
    }
    pub fn scalar(dtype: TypeCode) -> Self {
        Self::new(dtype, 0)
    }

    fn element_types(t: &NumPyPrimitiveTypes) -> [Option<TypeId>; 14] {
        [
            Some(t.bool_),
            Some(t.i8),
            Some(t.u8),
            Some(t.i16),
            Some(t.u16),
            Some(t.i32),
            Some(t.u32),
            Some(t.i64),
            Some(t.u64),
            Some(t.f16),
            Some(t.f32),
            Some(t.f64),
            t.c64,
            t.c128,
        ]
    }

    /// `NumPyType::get(t, T)`.
    ///
    /// Note: the reference implementation asserts on malformed ndarray generics; here such a type maps to
    /// the "none" NumPy type (not optimizable) since generics need a `TypeRealizer`.
    pub fn get(m: &mut Module, t: TypeId, types: &NumPyPrimitiveTypes) -> NumPyType {
        let elems = Self::element_types(types);
        for (i, e) in elems.iter().enumerate() {
            if is_opt(m, t, *e) {
                return NumPyType::scalar(SCALARS[i]);
            }
        }
        if is_array_type(m, Some(t)) {
            let generics = m.type_generics(t);
            if generics.len() != 2 || !generics[0].is_type() || !generics[1].is_static() {
                return NumPyType::default();
            }
            let dtype = generics[0].type_value().unwrap();
            let ndim = generics[1].static_value();
            for (i, e) in elems.iter().enumerate() {
                if is_opt(m, dtype, *e) {
                    return NumPyType::new(ARRAYS[i], ndim.max(0));
                }
            }
        }
        NumPyType::default()
    }

    /// `getIRBaseType(T)`.
    pub fn ir_base_type(&self, types: &NumPyPrimitiveTypes) -> TypeId {
        let idx = self.dtype.element_index().unwrap_or_else(|| panic!("unexpected type code ({:?})", self.dtype));
        Self::element_types(types)[idx].unwrap_or_else(|| panic!("complex type not available ({})", self.dtype.name()))
    }

    /// `operator bool`.
    pub fn is_some(&self) -> bool {
        self.dtype != TypeCode::None
    }
    pub fn is_array(&self) -> bool {
        self.dtype > TypeCode::ScalarEnd
    }
    pub fn str(&self) -> String {
        self.to_string()
    }
}

impl fmt::Display for NumPyType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_array() { write!(f, "array[{}, {}]", self.dtype.name(), self.ndim) } else { write!(f, "{}", self.dtype.name()) }
    }
}

// ---------------------------------------------------------------------------
// NumPyOptimizationUnit
// ---------------------------------------------------------------------------

/// Expression nodes live in the owning
/// [`ExprArena`] (shared by all units of one function so forwarding can move
/// subtrees between units).
#[derive(Clone, Debug)]
pub struct NumPyOptimizationUnit {
    /// Original IR value being corresponding to expression
    pub value: ValueId,
    /// Function in which the value exists
    pub func: FuncId,
    /// Root expression
    pub expr: ExprId,
    /// Leaves ordered by execution in original expression
    pub leaves: Vec<(ExprId, ValueId)>,
    /// AssignInstr in which RHS is represented by this expression, or null if none
    pub assign: Option<ValueId>,
    pub validation_order: Vec<(Option<ValueId>, ExprId)>,
    pub validation_flows: FxHashMap<ValueId, ValueId>,
}

impl NumPyOptimizationUnit {
    pub fn new(value: ValueId, func: FuncId, expr: ExprId, leaves: Vec<(ExprId, ValueId)>, assign: Option<ValueId>) -> Self {
        Self { value, func, expr, leaves, assign, validation_order: Vec::new(), validation_flows: FxHashMap::default() }
    }
}

/// Generic helpers.
pub(crate) fn gty(t: TypeId) -> Generic {
    Generic::Type(t)
}
pub(crate) fn gstr(s: impl Into<String>) -> Generic {
    Generic::StaticStr(s.into())
}
