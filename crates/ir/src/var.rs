//! (data part).

use crate::base::NodeBase;
use crate::ids::{Id, TypeId, ValueId, VarId};
use crate::types::Generic;
use crate::util::side_effect::SideEffectStatus;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Intrinsic {
    #[default]
    None,
    Allocate,
    AllocationSize,
    Typeinfo,
    TypeinfoLookup,
    TypeinfoMethodLookup,
    VirtualLookup,
    RecordNew,
    RepresentationCast,
    GeneratorPromise,
    Sizeof,
    IsAtomic,
    IsContentAtomic,
}

impl Intrinsic {
    /// `InternalFunc::getIntrinsicName(kind)`.
    pub fn name(self) -> &'static str {
        match self {
            Intrinsic::None => "none",
            Intrinsic::Allocate => "allocate",
            Intrinsic::AllocationSize => "allocation_size",
            Intrinsic::Typeinfo => "typeinfo",
            Intrinsic::TypeinfoLookup => "typeinfo_lookup",
            Intrinsic::TypeinfoMethodLookup => "typeinfo_method_lookup",
            Intrinsic::VirtualLookup => "virtual_lookup",
            Intrinsic::RecordNew => "record_new",
            Intrinsic::RepresentationCast => "representation_cast",
            Intrinsic::GeneratorPromise => "generator_promise",
            Intrinsic::Sizeof => "sizeof",
            Intrinsic::IsAtomic => "is_atomic",
            Intrinsic::IsContentAtomic => "is_content_atomic",
        }
    }
    /// `InternalFunc::isTypeQuery(kind)`.
    pub fn is_type_query(self) -> bool {
        matches!(self, Intrinsic::Sizeof | Intrinsic::IsAtomic | Intrinsic::IsContentAtomic)
    }
    /// `getIntrinsicSideEffectStatus()`.
    pub fn side_effect_status(self) -> SideEffectStatus {
        match self {
            Intrinsic::Sizeof
            | Intrinsic::IsAtomic
            | Intrinsic::IsContentAtomic
            | Intrinsic::AllocationSize
            | Intrinsic::Typeinfo
            | Intrinsic::RecordNew
            | Intrinsic::RepresentationCast
            | Intrinsic::GeneratorPromise => SideEffectStatus::Pure,
            Intrinsic::Allocate => SideEffectStatus::NoSideEffect,
            Intrinsic::TypeinfoLookup
            | Intrinsic::TypeinfoMethodLookup
            | Intrinsic::VirtualLookup
            | Intrinsic::None => SideEffectStatus::Unknown,
        }
    }
    /// `intrinsicDerivesFromArguments()`.
    pub fn derives_from_arguments(self) -> bool {
        matches!(self, Intrinsic::RecordNew | Intrinsic::RepresentationCast | Intrinsic::GeneratorPromise)
    }
}

/// Function subclass payload.
#[derive(Clone, Debug)]
pub enum FuncKind {
    /// `BodiedFunc`.
    Bodied { symbols: Vec<VarId>, body: Option<ValueId> },
    /// `ExternalFunc`.
    External,
    /// `InternalFunc`.
    Internal { intrinsic: Intrinsic, intrinsic_type: Option<TypeId> },
    /// `LLVMFunc`.
    Llvm { literals: Vec<Generic>, declarations: String, body: String },
}

impl FuncKind {
    pub fn bodied() -> Self {
        FuncKind::Bodied { symbols: Vec::new(), body: None }
    }
    pub fn internal() -> Self {
        FuncKind::Internal { intrinsic: Intrinsic::None, intrinsic_type: None }
    }
    pub fn llvm() -> Self {
        FuncKind::Llvm { literals: Vec::new(), declarations: String::new(), body: String::new() }
    }
}

#[derive(Clone, Debug)]
pub struct FuncData {
    pub unmangled_name: String,
    pub generator: bool,
    pub is_async: bool,
    pub parent_type: Option<TypeId>,
    pub args: Vec<VarId>,
    pub kind: FuncKind,
}

impl FuncData {
    pub fn new(kind: FuncKind) -> Self {
        Self {
            unmangled_name: String::new(),
            generator: false,
            is_async: false,
            parent_type: None,
            args: Vec::new(),
            kind,
        }
    }
    pub fn is_bodied(&self) -> bool {
        matches!(self.kind, FuncKind::Bodied { .. })
    }
    pub fn is_external(&self) -> bool {
        matches!(self.kind, FuncKind::External)
    }
    pub fn is_internal(&self) -> bool {
        matches!(self.kind, FuncKind::Internal { .. })
    }
    pub fn is_llvm(&self) -> bool {
        matches!(self.kind, FuncKind::Llvm { .. })
    }
    /// `BodiedFunc::getBody()`.
    pub fn body(&self) -> Option<ValueId> {
        match &self.kind {
            FuncKind::Bodied { body, .. } => *body,
            _ => None,
        }
    }
    /// `BodiedFunc` symbols (local variables).
    pub fn symbols(&self) -> &[VarId] {
        match &self.kind {
            FuncKind::Bodied { symbols, .. } => symbols,
            _ => &[],
        }
    }
    pub fn symbols_mut(&mut self) -> Option<&mut Vec<VarId>> {
        match &mut self.kind {
            FuncKind::Bodied { symbols, .. } => Some(symbols),
            _ => None,
        }
    }
    pub fn set_body(&mut self, b: Option<ValueId>) {
        if let FuncKind::Bodied { body, .. } = &mut self.kind {
            *body = b;
        }
    }
    pub fn intrinsic(&self) -> Intrinsic {
        match &self.kind {
            FuncKind::Internal { intrinsic, .. } => *intrinsic,
            _ => Intrinsic::None,
        }
    }
    pub fn intrinsic_type(&self) -> Option<TypeId> {
        match &self.kind {
            FuncKind::Internal { intrinsic_type, .. } => *intrinsic_type,
            _ => None,
        }
    }
    /// `InternalFunc::setIntrinsic(kind, type)`.
    pub fn set_intrinsic(&mut self, kind: Intrinsic, ty: Option<TypeId>) {
        if let FuncKind::Internal { intrinsic, intrinsic_type } = &mut self.kind {
            *intrinsic = kind;
            *intrinsic_type = ty;
        }
    }
    pub fn llvm_literals(&self) -> &[Generic] {
        match &self.kind {
            FuncKind::Llvm { literals, .. } => literals,
            _ => &[],
        }
    }
    pub fn llvm_declarations(&self) -> &str {
        match &self.kind {
            FuncKind::Llvm { declarations, .. } => declarations,
            _ => "",
        }
    }
    pub fn llvm_body(&self) -> &str {
        match &self.kind {
            FuncKind::Llvm { body, .. } => body,
            _ => "",
        }
    }
    pub fn set_llvm_literals(&mut self, v: Vec<Generic>) {
        if let FuncKind::Llvm { literals, .. } = &mut self.kind {
            *literals = v;
        }
    }
    pub fn set_llvm_declarations(&mut self, v: impl Into<String>) {
        if let FuncKind::Llvm { declarations, .. } = &mut self.kind {
            *declarations = v.into();
        }
    }
    pub fn set_llvm_body(&mut self, v: impl Into<String>) {
        if let FuncKind::Llvm { body, .. } = &mut self.kind {
            *body = v.into();
        }
    }
}

/// Storage of one variable (or function).
#[derive(Clone, Debug)]
pub struct VarData {
    pub base: NodeBase,
    pub ty: Option<TypeId>,
    pub global: bool,
    pub external: bool,
    pub tls: bool,
    pub func: Option<Box<FuncData>>,
    /// `util::AnyVar` / `util::AnyFunc` wildcard (matching only).
    pub any: bool,
    pub(crate) uid: Id,
    pub(crate) replacement: Option<VarId>,
    pub(crate) alive: bool,
}

impl VarData {
    pub fn is_func(&self) -> bool {
        self.func.is_some()
    }
    pub fn raw_uid(&self) -> Id {
        self.uid
    }
    pub fn replacement(&self) -> Option<VarId> {
        self.replacement
    }
    /// Short class name (`Var`, `BodiedFunc`, ...).
    pub fn kind_name(&self) -> &'static str {
        match &self.func {
            None => "Var",
            Some(f) => match f.kind {
                FuncKind::Bodied { .. } => "BodiedFunc",
                FuncKind::External => "ExternalFunc",
                FuncKind::Internal { .. } => "InternalFunc",
                FuncKind::Llvm { .. } => "LLVMFunc",
            },
        }
    }
}
