//! the reference uses `std::shared_ptr<Type>` with in-place mutation (binding of `LinkType`s,
//! `ClassType::generics[i].type = ...`, `_rn` caches). Here every type is a slot in a
//! [`TypeArena`] addressed by a `Copy` [`TypeId`]; slots are mutable through interior
//! mutability (`RefCell`) and never freed. All arena methods take `&self` and only hold
//! the internal borrow for the duration of a single slot access, so type operations may
//! freely re-enter the type checker (see [`TypeHost`]).
//!
//! Identity: the reference `shared_from_this()` == the same `TypeId`; `std::make_shared<X>(...)`
//! == a fresh slot (`TypeArena::new_*`).
//!
//! Module map: this file (Type, Unification, LiteralKind helpers), [`link`] (LinkType),
//! [`class`] (ClassType, Generic), [`function`] (FuncType), [`static_`] (Int/Str/BoolStaticType),
//! [`union`] (UnionType), [`traits`] (CallableTrait, TypeTrait).

pub mod class;
pub mod function;
pub mod link;
pub mod static_;
pub mod traits;
pub mod union;

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;
use typhon_common::{Result, SrcInfo};
use ustr::Ustr;

/// Dispatch summary of a type slot (see `TypeArena::shape`).
#[derive(Clone, Copy)]
enum Shape {
    Link(LinkKind, Option<TypeId>),
    Func,
    Static,
    /// Plain or union class; flags for the names with special `canRealize` rules.
    Class { type_: bool, unrealized: bool },
    Trait,
}

use crate::ast::ParamStatus;
pub use crate::ast::LiteralKind;
use crate::ids::StmtId;
pub use crate::ids::TypeId;

/// the reference `InstantiationCache` (generic id -> instantiated type).
pub type InstantiationCache = FxHashMap<i32, TypeId>;

/// `LinkType::Kind`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LinkKind {
    Unbound,
    Generic,
    Link,
}

/// an unbound type variable, a generic, or a link to another type.
#[derive(Clone, Debug)]
pub struct LinkType {
    /// Current state.
    pub kind: LinkKind,
    /// Unique identifier of an unbound or generic type.
    pub id: i32,
    /// Type-checking level of an unbound type.
    pub level: i32,
    /// Target of a `Link` (None for unbound/generic).
    pub ty: Option<TypeId>,
    /// Non-runtime if a static type (e.g. `N` in `Int[N: int]`).
    pub static_kind: LiteralKind,
    /// Optional trait that an unbound requires prior to unification.
    pub trait_: Option<TypeId>,
    /// Generic name (pretty-printing).
    pub generic_name: String,
    /// Type used if an unbound is not resolved.
    pub default_type: Option<TypeId>,
    /// Can be used unrealized as a function argument during function realization.
    pub pass_through: bool,
}

#[derive(Clone, Debug)]
pub struct Generic {
    /// Generic name (canonical).
    pub name: Ustr,
    /// Unique generic ID.
    pub id: i32,
    /// Realized type (or generic `LinkType`).
    pub ty: Option<TypeId>,
    /// Set if this is a static generic.
    pub static_kind: LiteralKind,
}

impl Generic {
    pub fn new(name: impl AsRef<str>, ty: Option<TypeId>, id: i32, static_kind: LiteralKind) -> Self {
        Self { name: Ustr::from(name.as_ref()), id, ty, static_kind }
    }
    /// `getType()` (panics if null, which the reference would dereference anyway).
    pub fn ty(&self) -> TypeId {
        self.ty.expect("generic without type")
    }
}

/// `FuncType`-specific data (the `ClassType` part realizes `Function[args, ret]`).
#[derive(Clone, Debug)]
pub struct FuncData {
    /// Canonical AST node (`FunctionStmt *ast`). Name/params are mirrored in
    /// [`TypeArena::func_ast`].
    pub ast: StmtId,
    /// Function generics (e.g. `T` in `def foo[T](...)`).
    pub func_generics: Vec<Generic>,
    /// Enclosing class or function.
    pub func_parent: Option<TypeId>,
}

/// The dynamic class of a [`ClassType`] (subclasses of `ClassType`).
#[derive(Clone, Debug)]
pub enum ClassKind {
    /// Plain `ClassType`.
    Plain,
    /// `FuncType`.
    Func(Box<FuncData>),
    /// `UnionType`.
    Union,
    /// `IntStaticType`.
    IntStatic(i64),
    /// `StrStaticType`.
    StrStatic(String),
    /// `BoolStaticType`.
    BoolStatic(bool),
}

#[derive(Clone, Debug)]
pub struct ClassType {
    /// Canonical type name.
    pub name: Ustr,
    pub generics: Vec<Generic>,
    pub hidden_generics: Vec<Generic>,
    pub is_tuple: bool,
    /// Fully generalized concrete types are reused.
    pub immutable: bool,
    /// Cached realized name (`_rn`).
    pub rn: String,
    pub kind: ClassKind,
}

impl ClassType {
    pub fn new(name: impl AsRef<str>, generics: Vec<Generic>, hidden_generics: Vec<Generic>) -> Self {
        Self {
            name: Ustr::from(name.as_ref()),
            generics,
            hidden_generics,
            is_tuple: false,
            immutable: false,
            rn: String::new(),
            kind: ClassKind::Plain,
        }
    }
    pub fn is_func(&self) -> bool {
        matches!(self.kind, ClassKind::Func(_))
    }
    pub fn is_static(&self) -> bool {
        matches!(self.kind, ClassKind::IntStatic(_) | ClassKind::StrStatic(_) | ClassKind::BoolStatic(_))
    }
    pub fn func(&self) -> Option<&FuncData> {
        match &self.kind {
            ClassKind::Func(f) => Some(f),
            _ => None,
        }
    }
    pub fn func_mut(&mut self) -> Option<&mut FuncData> {
        match &mut self.kind {
            ClassKind::Func(f) => Some(f),
            _ => None,
        }
    }
    /// `isRecord()`.
    pub fn is_record(&self) -> bool {
        self.is_tuple
    }
}

/// Traits (`CallableTrait`, `TypeTrait`).
#[derive(Clone, Debug)]
pub enum TraitKind {
    /// `CallableTrait`: `args = [Tuple[arg types], ret type]` (entries may be null).
    Callable { args: Vec<Option<TypeId>> },
    /// `TypeTrait`.
    Type { ty: TypeId },
}

/// Concrete kind of an arena slot.
#[derive(Clone, Debug)]
pub enum TypeKind {
    Link(LinkType),
    Class(ClassType),
    Trait(TraitKind),
}

/// One arena slot (`Type` object).
#[derive(Clone, Debug)]
pub struct TypeData {
    pub kind: TypeKind,
    pub src: SrcInfo,
}

/// Snapshot of the parts of a `FunctionStmt` that type printing / `canRealize` need.
#[derive(Clone, Debug, Default)]
pub struct FuncAstInfo {
    pub name: String,
    /// `(name, status)` for each parameter (`(*ast)[i]`).
    pub params: Vec<(String, ParamStatus)>,
    /// `ast->hasAttribute(Attr::AllowPassThrough)`.
    pub allow_pass_through: bool,
}

/// list of destructive unification steps that can be
/// undone later.
#[derive(Clone, Debug, Default)]
pub struct Unification {
    /// Unbound types that were linked.
    pub linked: Vec<TypeId>,
    /// Unbound types whose level changed (with the old level).
    pub leveled: Vec<(TypeId, i32)>,
    /// Unbound types that got a trait assigned.
    pub traits: Vec<TypeId>,
    /// Unbound types whose static status changed.
    pub statics: Vec<TypeId>,
}

impl Unification {
    pub fn new() -> Self {
        Self::default()
    }
    /// `Unification::undo()`.
    /// Note: the journal is drained (the reference keeps it; replaying it twice is a no-op there
    /// in practice).
    pub fn undo(&mut self, arena: &TypeArena) {
        for t in self.linked.drain(..).rev() {
            arena.with_link_mut(t, |l| {
                l.kind = LinkKind::Unbound;
                l.ty = None;
            });
        }
        for (t, lvl) in self.leveled.drain(..).rev() {
            arena.with_link_mut(t, |l| {
                typhon_common::seqassert!(l.kind == LinkKind::Unbound, "not unbound");
                l.level = lvl;
            });
        }
        for t in self.traits.drain(..) {
            arena.with_link_mut(t, |l| l.trait_ = None);
        }
        for t in self.statics.drain(..) {
            arena.with_link_mut(t, |l| l.static_kind = LiteralKind::Runtime);
        }
    }
}

/// Callbacks into the type checker that some type operations need (the reference types construct a
/// `TypecheckVisitor(cache->typeCtx)` internally).
pub trait TypeHost {
    /// `CallableTrait::unify` -> `TypecheckVisitor::matchCallable(typ, args, ret, us)`;
    /// `args` are the argument types (each becomes a typed dummy `NoneExpr`).
    fn match_callable(
        &mut self,
        typ: TypeId,
        args: &[TypeId],
        ret: Option<TypeId>,
        undo: Option<&mut Unification>,
    ) -> Result<i32>;
    /// `tv.instantiateType(tv.generateTuple(n), args)` (`args == None`: plain
    /// `instantiateType(generateTuple(n))`).
    fn instantiate_tuple(&mut self, n: usize, args: Option<&[TypeId]>) -> Result<TypeId>;
}

/// A host for contexts without a type checker (unit tests, printing): callable traits never
/// match and tuples cannot be generated.
pub struct NoHost;

impl TypeHost for NoHost {
    fn match_callable(&mut self, _: TypeId, _: &[TypeId], _: Option<TypeId>, _: Option<&mut Unification>) -> Result<i32> {
        Ok(-1)
    }
    fn instantiate_tuple(&mut self, n: usize, _: Option<&[TypeId]>) -> Result<TypeId> {
        Err(typhon_common::CompilerError::custom(
            format!("cannot generate Tuple[{n}] without a type checker"),
            SrcInfo::default(),
        ))
    }
}

/// Resolver for `cache->findClass(name)` used by `StaticType::getNonStaticType`.
pub type ClassResolver = Rc<dyn Fn(&str) -> Option<TypeId>>;

/// Owner of all type-checker types (`shared_ptr<Type>` heap) plus the bits of `Cache`
/// that type printing needs (reverse identifier lookup, generated tuple names, unbound
/// counter).
pub struct TypeArena {
    slots: RefCell<Vec<TypeData>>,
    /// `Cache::unboundCount` (starts at 256).
    pub unbound_count: Cell<i32>,
    /// `Cache::reverseIdentifierLookup` (canonical -> user name).
    pub reverse_lookup: RefCell<FxHashMap<String, String>>,
    /// `Cache::generatedTupleNames`.
    pub generated_tuple_names: RefCell<Vec<Vec<String>>>,
    /// Snapshot of function ASTs referenced by `FuncType`s.
    func_asts: RefCell<FxHashMap<StmtId, Rc<FuncAstInfo>>>,
    /// `cache->findClass` (installed by `Cache`), plus a memo of its answers.
    class_resolver: RefCell<Option<ClassResolver>>,
    non_static: RefCell<FxHashMap<String, TypeId>>,
}

impl Default for TypeArena {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for TypeArena {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "TypeArena {{ types: {} }}", self.len())
    }
}

/// `Type::literalFromString`.
pub fn literal_from_string(s: &str) -> LiteralKind {
    LiteralKind::from_str(s)
}

/// `Type::stringFromLiteral`.
pub fn string_from_literal(k: LiteralKind) -> &'static str {
    match k {
        LiteralKind::Int => "int",
        LiteralKind::String => "str",
        LiteralKind::Bool => "bool",
        LiteralKind::Runtime => "",
    }
}

/// `static_cast<int>(LiteralKind)`.
pub fn literal_index(k: LiteralKind) -> i32 {
    match k {
        LiteralKind::Runtime => 0,
        LiteralKind::Int => 1,
        LiteralKind::String => 2,
        LiteralKind::Bool => 3,
    }
}

pub(crate) fn join<S: AsRef<str>>(v: &[S], sep: &str) -> String {
    v.iter().map(|s| s.as_ref()).collect::<Vec<_>>().join(sep)
}

impl TypeArena {
    pub fn new() -> Self {
        Self {
            slots: RefCell::new(Vec::new()),
            unbound_count: Cell::new(256),
            reverse_lookup: RefCell::new(FxHashMap::default()),
            generated_tuple_names: RefCell::new(vec![vec![]]),
            func_asts: RefCell::new(FxHashMap::default()),
            class_resolver: RefCell::new(None),
            non_static: RefCell::new(FxHashMap::default()),
        }
    }

    pub fn len(&self) -> usize {
        self.slots.borrow().len()
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Next unbound id (`cache->unboundCount++`).
    pub fn next_unbound_id(&self) -> i32 {
        let v = self.unbound_count.get();
        self.unbound_count.set(v + 1);
        v
    }

    /// `cache->rev(s)` (empty string if unknown, the reference asserts).
    pub fn rev(&self, s: &str) -> String {
        match self.reverse_lookup.borrow().get(s) {
            Some(r) => r.clone(),
            None => {
                typhon_common::seqassert!(false, "'{}' has no non-canonical name", s);
                String::new()
            }
        }
    }
    /// `getUnmangledName`: reverse lookup or the input.
    pub fn unmangled(&self, s: &str) -> String {
        self.reverse_lookup.borrow().get(s).cloned().unwrap_or_else(|| s.to_string())
    }

    /// Install `cache->findClass` for `getNonStaticType`.
    pub fn set_class_resolver(&self, r: Option<ClassResolver>) {
        *self.class_resolver.borrow_mut() = r;
        self.non_static.borrow_mut().clear();
    }
    /// `cache->findClass(name)` through the resolver (memoized).
    pub fn find_class(&self, name: &str) -> Option<TypeId> {
        if let Some(t) = self.non_static.borrow().get(name) {
            return Some(*t);
        }
        let r = self.class_resolver.borrow().clone()?;
        let t = r(name)?;
        self.non_static.borrow_mut().insert(name.to_string(), t);
        Some(t)
    }

    /// Register / refresh the AST snapshot for a function (see [`FuncAstInfo`]).
    pub fn register_func_ast(&self, ast: StmtId, info: FuncAstInfo) {
        self.func_asts.borrow_mut().insert(ast, Rc::new(info));
    }
    /// AST snapshot of a function (empty info if not registered).
    pub fn func_ast(&self, ast: StmtId) -> Rc<FuncAstInfo> {
        self.func_asts.borrow().get(&ast).cloned().unwrap_or_default()
    }

    // ------------------------------------------------------------------
    // slot access
    // ------------------------------------------------------------------

    /// Allocate a new slot.
    pub fn alloc(&self, kind: TypeKind, src: SrcInfo) -> TypeId {
        let mut s = self.slots.borrow_mut();
        let id = TypeId(s.len() as u32);
        s.push(TypeData { kind, src });
        id
    }
    /// Read a slot. Do not call arena methods from `f`.
    pub fn with<R>(&self, id: TypeId, f: impl FnOnce(&TypeData) -> R) -> R {
        f(&self.slots.borrow()[id.0 as usize])
    }
    /// Mutate a slot. Do not call arena methods from `f`.
    pub fn with_mut<R>(&self, id: TypeId, f: impl FnOnce(&mut TypeData) -> R) -> R {
        f(&mut self.slots.borrow_mut()[id.0 as usize])
    }
    /// Clone of a slot.
    pub fn get(&self, id: TypeId) -> TypeData {
        self.with(id, |d| d.clone())
    }
    pub fn kind(&self, id: TypeId) -> TypeKind {
        self.with(id, |d| d.kind.clone())
    }
    /// `getSrcInfo()`.
    pub fn src(&self, id: TypeId) -> SrcInfo {
        self.with(id, |d| d.src)
    }
    /// `setSrcInfo()`.
    pub fn set_src(&self, id: TypeId, src: SrcInfo) {
        self.with_mut(id, |d| d.src = src)
    }
    /// Raw link data of a `LinkType` slot (no following).
    pub fn link(&self, id: TypeId) -> Option<LinkType> {
        self.with(id, |d| match &d.kind {
            TypeKind::Link(l) => Some(l.clone()),
            _ => None,
        })
    }
    /// Mutate a `LinkType` slot (no-op on other kinds).
    pub fn with_link_mut<R>(&self, id: TypeId, f: impl FnOnce(&mut LinkType) -> R) -> Option<R> {
        self.with_mut(id, |d| match &mut d.kind {
            TypeKind::Link(l) => Some(f(l)),
            _ => None,
        })
    }
    /// Raw class data of a class slot (no following, no canonicalization).
    pub fn class(&self, id: TypeId) -> Option<ClassType> {
        self.with(id, |d| match &d.kind {
            TypeKind::Class(c) => Some(c.clone()),
            _ => None,
        })
    }
    /// Read the class data of a class slot (no following).
    pub fn with_class<R>(&self, id: TypeId, f: impl FnOnce(&ClassType) -> R) -> Option<R> {
        self.with(id, |d| match &d.kind {
            TypeKind::Class(c) => Some(f(c)),
            _ => None,
        })
    }
    /// Mutate a class slot (no following).
    pub fn with_class_mut<R>(&self, id: TypeId, f: impl FnOnce(&mut ClassType) -> R) -> Option<R> {
        self.with_mut(id, |d| match &mut d.kind {
            TypeKind::Class(c) => Some(f(c)),
            _ => None,
        })
    }
    fn trait_kind(&self, id: TypeId) -> Option<TraitKind> {
        self.with(id, |d| match &d.kind {
            TypeKind::Trait(t) => Some(t.clone()),
            _ => None,
        })
    }
    pub fn is_trait(&self, id: TypeId) -> bool {
        self.with(id, |d| matches!(d.kind, TypeKind::Trait(_)))
    }

    // ------------------------------------------------------------------
    // constructors
    // ------------------------------------------------------------------

    /// `LinkType(cache, kind, id, level, type, staticKind, trait, defaultType, genericName,
    /// passThrough)`.
    pub fn new_link_type(&self, l: LinkType, src: SrcInfo) -> TypeId {
        typhon_common::seqassert!(
            (l.ty.is_some() && l.kind == LinkKind::Link) || (l.ty.is_none() && l.kind != LinkKind::Link),
            "inconsistent link state"
        );
        self.alloc(TypeKind::Link(l), src)
    }
    /// `make_shared<LinkType>(cache, Unbound, id, level)`.
    pub fn new_unbound(&self, id: i32, level: i32, src: SrcInfo) -> TypeId {
        self.new_link_type(
            LinkType {
                kind: LinkKind::Unbound,
                id,
                level,
                ty: None,
                static_kind: LiteralKind::Runtime,
                trait_: None,
                generic_name: String::new(),
                default_type: None,
                pass_through: false,
            },
            src,
        )
    }
    /// `make_shared<LinkType>(type)` (a `Link` to `t`; inherits the source location).
    pub fn new_link(&self, t: TypeId) -> TypeId {
        let src = self.src(t);
        self.new_link_type(
            LinkType {
                kind: LinkKind::Link,
                id: 0,
                level: 0,
                ty: Some(t),
                static_kind: LiteralKind::Runtime,
                trait_: None,
                generic_name: String::new(),
                default_type: None,
                pass_through: false,
            },
            src,
        )
    }
    /// `make_shared<ClassType>(cache, name, generics, hiddenGenerics)`.
    pub fn new_class(&self, name: impl AsRef<str>, generics: Vec<Generic>, hidden: Vec<Generic>) -> TypeId {
        self.alloc(TypeKind::Class(ClassType::new(name, generics, hidden)), SrcInfo::default())
    }
    /// Allocate a fully specified class.
    pub fn new_class_from(&self, c: ClassType, src: SrcInfo) -> TypeId {
        self.alloc(TypeKind::Class(c), src)
    }
    /// `make_shared<ClassType>(const ClassType *base)` (copy constructor: always yields a
    /// plain `ClassType`; `immutable` and `_rn` are not copied).
    pub fn copy_class(&self, base: TypeId) -> TypeId {
        let base = self.get_class(base).expect("copy of a non-class type");
        let (c, src) = self.with(base, |d| match &d.kind {
            TypeKind::Class(c) => {
                let mut n = ClassType::new(c.name, c.generics.clone(), c.hidden_generics.clone());
                n.is_tuple = c.is_tuple;
                (n, d.src)
            }
            _ => unreachable!(),
        });
        self.alloc(TypeKind::Class(c), src)
    }
    /// `make_shared<FuncType>(baseType, ast, funcGenerics, funcParent)`.
    pub fn new_func(&self, base: TypeId, ast: StmtId, func_generics: Vec<Generic>, func_parent: Option<TypeId>) -> TypeId {
        let base = self.get_class(base).expect("function base must be a class");
        let (mut c, src) = self.with(base, |d| match &d.kind {
            TypeKind::Class(c) => (c.clone(), d.src),
            _ => unreachable!(),
        });
        c.immutable = false;
        c.rn = String::new();
        c.kind = ClassKind::Func(Box::new(FuncData { ast, func_generics, func_parent }));
        self.alloc(TypeKind::Class(c), src)
    }
    /// `make_shared<IntStaticType>(cache, i)`.
    pub fn new_int_static(&self, v: i64) -> TypeId {
        let mut c = ClassType::new("Int", vec![], vec![]);
        c.rn = "Int[64]".into();
        c.kind = ClassKind::IntStatic(v);
        self.alloc(TypeKind::Class(c), SrcInfo::default())
    }
    /// `make_shared<StrStaticType>(cache, s)`.
    pub fn new_str_static(&self, v: impl Into<String>) -> TypeId {
        let mut c = ClassType::new("str", vec![], vec![]);
        c.kind = ClassKind::StrStatic(v.into());
        self.alloc(TypeKind::Class(c), SrcInfo::default())
    }
    /// `make_shared<BoolStaticType>(cache, b)`.
    pub fn new_bool_static(&self, v: bool) -> TypeId {
        let mut c = ClassType::new("bool", vec![], vec![]);
        c.kind = ClassKind::BoolStatic(v);
        self.alloc(TypeKind::Class(c), SrcInfo::default())
    }
    /// `make_shared<UnionType>(cache, generics)`.
    pub fn new_union(&self, generics: Vec<Generic>) -> TypeId {
        let mut c = ClassType::new(&typhon_common::mangle::stdlib().union, generics, vec![]);
        c.is_tuple = true;
        c.kind = ClassKind::Union;
        self.alloc(TypeKind::Class(c), SrcInfo::default())
    }
    /// `make_shared<CallableTrait>(cache, args)`.
    pub fn new_callable_trait(&self, args: Vec<Option<TypeId>>) -> TypeId {
        self.alloc(TypeKind::Trait(TraitKind::Callable { args }), SrcInfo::default())
    }
    /// `make_shared<TypeTrait>(type)`.
    pub fn new_type_trait(&self, t: TypeId) -> TypeId {
        let src = self.src(t);
        self.alloc(TypeKind::Trait(TraitKind::Type { ty: t }), src)
    }

    // ------------------------------------------------------------------
    // "virtual" getters (getLink/getClass/getFunc/...)
    // ------------------------------------------------------------------

    /// `follow()`: final type after following all links.
    pub fn follow(&self, mut id: TypeId) -> TypeId {
        loop {
            let next = self.with(id, |d| match &d.kind {
                TypeKind::Link(l) if l.kind == LinkKind::Link => l.ty,
                _ => None,
            });
            match next {
                Some(n) => id = n,
                None => return id,
            }
        }
    }
    /// `getLink()`: the slot itself if it is a `LinkType` (any state).
    pub fn get_link(&self, id: TypeId) -> Option<TypeId> {
        self.with(id, |d| matches!(d.kind, TypeKind::Link(_))).then_some(id)
    }
    /// `getUnbound()`: follows links; the unbound `LinkType`, if any.
    pub fn get_unbound(&self, id: TypeId) -> Option<TypeId> {
        let f = self.follow(id);
        self.with(f, |d| matches!(&d.kind, TypeKind::Link(l) if l.kind == LinkKind::Unbound)).then_some(f)
    }
    /// True for an (unfollowed or followed) unbound.
    pub fn is_unbound(&self, id: TypeId) -> bool {
        self.get_unbound(id).is_some()
    }
    /// `getClass()`: follows links, applies the canonical-optional rule.
    pub fn get_class(&self, id: TypeId) -> Option<TypeId> {
        let f = self.follow(id);
        if !self.with(f, |d| matches!(d.kind, TypeKind::Class(_))) {
            return None;
        }
        Some(self.canonical_optional(f).unwrap_or(f))
    }
    /// `getFunc()`.
    pub fn get_func(&self, id: TypeId) -> Option<TypeId> {
        let f = self.follow(id);
        self.with(f, |d| matches!(&d.kind, TypeKind::Class(c) if c.is_func())).then_some(f)
    }
    /// `getPartial()`.
    pub fn get_partial(&self, id: TypeId) -> Option<TypeId> {
        let f = self.follow(id);
        let is_p = self.with(f, |d| matches!(&d.kind, TypeKind::Class(c) if c.name == "Partial"));
        if is_p { self.get_class(f) } else { None }
    }
    /// `getStatic()`.
    pub fn get_static(&self, id: TypeId) -> Option<TypeId> {
        let f = self.follow(id);
        self.with(f, |d| matches!(&d.kind, TypeKind::Class(c) if c.is_static())).then_some(f)
    }
    /// `getIntStatic()` value.
    pub fn get_int_static(&self, id: TypeId) -> Option<i64> {
        let f = self.follow(id);
        self.with(f, |d| match &d.kind {
            TypeKind::Class(ClassType { kind: ClassKind::IntStatic(v), .. }) => Some(*v),
            _ => None,
        })
    }
    /// `getStrStatic()` value.
    pub fn get_str_static(&self, id: TypeId) -> Option<String> {
        let f = self.follow(id);
        self.with(f, |d| match &d.kind {
            TypeKind::Class(ClassType { kind: ClassKind::StrStatic(v), .. }) => Some(v.clone()),
            _ => None,
        })
    }
    /// `getBoolStatic()` value.
    pub fn get_bool_static(&self, id: TypeId) -> Option<bool> {
        let f = self.follow(id);
        self.with(f, |d| match &d.kind {
            TypeKind::Class(ClassType { kind: ClassKind::BoolStatic(v), .. }) => Some(*v),
            _ => None,
        })
    }
    /// `getUnion()`.
    pub fn get_union(&self, id: TypeId) -> Option<TypeId> {
        let f = self.follow(id);
        self.with(f, |d| matches!(&d.kind, TypeKind::Class(ClassType { kind: ClassKind::Union, .. }))).then_some(f)
    }
    /// `is(s)`: `getClass() && getClass()->name == s`.
    pub fn is(&self, id: TypeId, s: &str) -> bool {
        match self.get_class(id) {
            Some(c) => self.with_class(c, |c| c.name == s).unwrap_or(false),
            None => false,
        }
    }
    /// Name of `getClass()`.
    pub fn class_name(&self, id: TypeId) -> Option<Ustr> {
        self.get_class(id).and_then(|c| self.with_class(c, |c| c.name))
    }
    /// `getStaticKind()`.
    pub fn static_kind(&self, id: TypeId) -> LiteralKind {
        let f = self.follow(id);
        self.with(f, |d| match &d.kind {
            TypeKind::Class(c) => match c.kind {
                ClassKind::IntStatic(_) => LiteralKind::Int,
                ClassKind::StrStatic(_) => LiteralKind::String,
                ClassKind::BoolStatic(_) => LiteralKind::Bool,
                _ => LiteralKind::Runtime,
            },
            TypeKind::Link(l) => l.static_kind,
            TypeKind::Trait(_) => LiteralKind::Runtime,
        })
    }
    /// Generics of `getClass()` (cloned).
    pub fn generics(&self, id: TypeId) -> Vec<Generic> {
        self.get_class(id).and_then(|c| self.with_class(c, |c| c.generics.clone())).unwrap_or_default()
    }
    /// Hidden generics of `getClass()` (cloned).
    pub fn hidden_generics(&self, id: TypeId) -> Vec<Generic> {
        self.get_class(id).and_then(|c| self.with_class(c, |c| c.hidden_generics.clone())).unwrap_or_default()
    }
    /// `getClass()->generics[idx].type`.
    pub fn generic_type(&self, id: TypeId, idx: usize) -> Option<TypeId> {
        self.get_class(id).and_then(|c| self.with_class(c, |c| c.generics.get(idx).and_then(|g| g.ty)).flatten())
    }
    /// `getClass()->generics.size()`.
    pub fn generics_len(&self, id: TypeId) -> usize {
        self.get_class(id).and_then(|c| self.with_class(c, |c| c.generics.len())).unwrap_or(0)
    }
    /// `getClass()->isRecord()`.
    pub fn is_record(&self, id: TypeId) -> bool {
        self.get_class(id).and_then(|c| self.with_class(c, |c| c.is_tuple)).unwrap_or(false)
    }

    // ------------------------------------------------------------------
    // virtual dispatch
    // ------------------------------------------------------------------

    /// `Type::unify(typ, undo)`: score `-1` on failure, `>= 0` on success (higher is
    /// better). Destructive if `undo` is `Some` (record changes there).
    pub fn unify(&self, a: TypeId, b: TypeId, undo: Option<&mut Unification>, host: &mut dyn TypeHost) -> Result<i32> {
        enum K {
            Link,
            Class,
            Func,
            Union,
            Int,
            StrBool,
            Trait,
        }
        let k = self.with(a, |d| match &d.kind {
            TypeKind::Link(_) => K::Link,
            TypeKind::Trait(_) => K::Trait,
            TypeKind::Class(c) => match c.kind {
                ClassKind::Plain => K::Class,
                ClassKind::Func(_) => K::Func,
                ClassKind::Union => K::Union,
                ClassKind::IntStatic(_) => K::Int,
                ClassKind::StrStatic(_) | ClassKind::BoolStatic(_) => K::StrBool,
            },
        });
        match k {
            K::Link => self.link_unify(a, b, undo, host),
            K::Class => self.class_unify(a, b, undo, host),
            K::Func => self.func_unify(a, b, undo, host),
            K::Union => self.union_unify(a, b, undo, host),
            K::Int | K::StrBool => self.static_unify(a, b, undo, host),
            K::Trait => self.trait_unify(a, b, undo, host),
        }
    }

    /// `Type::operator<<`: unify; undo on failure. Returns success.
    pub fn unify_or_undo(&self, a: TypeId, b: TypeId, host: &mut dyn TypeHost) -> Result<bool> {
        let mut undo = Unification::new();
        let r = self.unify(a, b, Some(&mut undo), host);
        match r {
            Ok(s) if s >= 0 => Ok(true),
            Ok(_) => {
                undo.undo(self);
                Ok(false)
            }
            Err(e) => {
                undo.undo(self);
                Err(e)
            }
        }
    }

    /// `generalize(atLevel)`: replace unbounds at level >= `at_level` with generics.
    pub fn generalize(&self, id: TypeId, at_level: i32) -> TypeId {
        match self.kind(id) {
            TypeKind::Link(_) => self.link_generalize(id, at_level),
            TypeKind::Class(c) => match c.kind {
                ClassKind::Plain => self.class_generalize(id, at_level),
                ClassKind::Func(_) => self.func_generalize(id, at_level),
                ClassKind::Union => self.union_generalize(id, at_level),
                _ => id,
            },
            TypeKind::Trait(_) => self.trait_generalize(id, at_level),
        }
    }

    /// `instantiate(atLevel, unboundCount, cache)` (the arena counter is always used).
    pub fn instantiate(&self, id: TypeId, at_level: i32, cache: Option<&mut InstantiationCache>) -> TypeId {
        let mut local;
        let (c, has) = match cache {
            Some(c) => (c, true),
            None => {
                local = InstantiationCache::default();
                (&mut local, false)
            }
        };
        self.instantiate_impl(id, at_level, true, c, has)
    }

    /// Full `instantiate` (`use_counter == false` mirrors a null `unboundCount`;
    /// `has_cache == false` a null cache).
    pub(crate) fn instantiate_impl(
        &self,
        id: TypeId,
        at_level: i32,
        use_counter: bool,
        cache: &mut InstantiationCache,
        has_cache: bool,
    ) -> TypeId {
        match self.kind(id) {
            TypeKind::Link(_) => self.link_instantiate(id, at_level, use_counter, cache, has_cache),
            TypeKind::Class(c) => match c.kind {
                ClassKind::Plain => self.class_instantiate(id, at_level, use_counter, cache, has_cache),
                ClassKind::Func(_) => self.func_instantiate(id, at_level, use_counter, cache, has_cache),
                ClassKind::Union => self.union_instantiate(id, at_level, use_counter, cache, has_cache),
                _ => id,
            },
            TypeKind::Trait(_) => self.trait_instantiate(id, at_level, use_counter, cache, has_cache),
        }
    }

    /// `hasUnbounds(includeGenerics)`.
    pub fn has_unbounds(&self, id: TypeId, include_generics: bool) -> bool {
        match self.shape(id) {
            Shape::Link(LinkKind::Unbound, _) => true,
            Shape::Link(LinkKind::Generic, _) => include_generics,
            Shape::Link(LinkKind::Link, t) => self.has_unbounds(t.unwrap(), include_generics),
            Shape::Func => self.func_has_unbounds(id, include_generics),
            Shape::Class { unrealized: true, .. } => false,
            Shape::Class { .. } | Shape::Static => self
                .class_generic_tys(id)
                .into_iter()
                .any(|t| self.has_unbounds(t, include_generics)),
            Shape::Trait => false,
        }
    }

    /// `getUnbounds(includeGenerics)` (order: later generics first).
    pub fn get_unbounds(&self, id: TypeId, include_generics: bool) -> Vec<TypeId> {
        match self.kind(id) {
            TypeKind::Link(l) => match l.kind {
                LinkKind::Unbound => vec![id],
                LinkKind::Link => self.get_unbounds(l.ty.unwrap(), include_generics),
                LinkKind::Generic => {
                    if include_generics {
                        vec![id]
                    } else {
                        vec![]
                    }
                }
            },
            TypeKind::Class(c) => match c.kind {
                ClassKind::Func(_) => self.func_get_unbounds(id, include_generics),
                ClassKind::IntStatic(_) | ClassKind::StrStatic(_) | ClassKind::BoolStatic(_) => {
                    self.class_get_unbounds(&c, include_generics)
                }
                _ => self.class_get_unbounds(&c, include_generics),
            },
            TypeKind::Trait(_) => vec![],
        }
    }

    /// `canRealize()`.
    pub fn can_realize(&self, id: TypeId) -> bool {
        match self.shape(id) {
            Shape::Link(k, t) => k == LinkKind::Link && self.can_realize(t.unwrap()),
            Shape::Func => self.func_can_realize(id),
            Shape::Static => true,
            Shape::Class { type_: true, .. } if !self.has_unbounds(id, false) => true,
            Shape::Class { unrealized: true, .. } => {
                let g0 = self.with_class(id, |c| c.generics[0].ty()).unwrap();
                self.get_class(g0).is_some()
            }
            Shape::Class { .. } => self.class_generic_tys(id).into_iter().all(|t| self.can_realize(t)),
            Shape::Trait => false,
        }
    }

    /// Cheap classification of a slot (avoids cloning the slot's data).
    fn shape(&self, id: TypeId) -> Shape {
        self.with(id, |d| match &d.kind {
            TypeKind::Link(l) => Shape::Link(l.kind, l.ty),
            TypeKind::Class(c) => match c.kind {
                ClassKind::Func(_) => Shape::Func,
                ClassKind::IntStatic(_) | ClassKind::StrStatic(_) | ClassKind::BoolStatic(_) => Shape::Static,
                _ => {
                    let st = typhon_common::mangle::stdlib();
                    Shape::Class {
                        type_: c.name == st.type_.as_str(),
                        unrealized: c.name == st.unrealized_type.as_str(),
                    }
                }
            },
            TypeKind::Trait(_) => Shape::Trait,
        })
    }

    /// Types of a class slot's generics followed by its hidden generics (unset ones skipped).
    fn class_generic_tys(&self, id: TypeId) -> SmallVec<[TypeId; 8]> {
        self.with_class(id, |c| c.generics.iter().chain(c.hidden_generics.iter()).filter_map(|g| g.ty).collect())
            .unwrap_or_default()
    }

    /// `isInstantiated()`.
    pub fn is_instantiated(&self, id: TypeId) -> bool {
        self.is_instantiated_impl(id, None)
    }
    pub(crate) fn is_instantiated_impl(&self, id: TypeId, skip_parent_of: Option<TypeId>) -> bool {
        match self.shape(id) {
            Shape::Link(k, t) => k == LinkKind::Link && self.is_instantiated_impl(t.unwrap(), skip_parent_of),
            Shape::Func => self.func_is_instantiated(id, skip_parent_of),
            Shape::Static => true,
            Shape::Class { unrealized: true, .. } => {
                let g0 = self.with_class(id, |c| c.generics[0].ty()).unwrap();
                self.get_class(g0).is_some()
            }
            Shape::Class { .. } => {
                self.class_generic_tys(id).into_iter().all(|t| self.is_instantiated_impl(t, skip_parent_of))
            }
            Shape::Trait => false,
        }
    }

    /// `debugString(mode)`: 0 = pretty, 1 = llvm, 2 = debug.
    pub fn debug_string(&self, id: TypeId, mode: i32) -> String {
        match self.kind(id) {
            TypeKind::Link(l) => self.link_debug_string(&l, mode),
            TypeKind::Class(c) => match &c.kind {
                ClassKind::Func(_) => self.func_debug_string(id, mode),
                ClassKind::Union => self.union_debug_string(id, &c, mode),
                ClassKind::IntStatic(_) | ClassKind::StrStatic(_) | ClassKind::BoolStatic(_) => {
                    self.static_debug_string(&c, mode)
                }
                ClassKind::Plain => self.class_debug_string(id, mode),
            },
            TypeKind::Trait(t) => self.trait_debug_string(&t, mode),
        }
    }
    /// `toString()` (= `debugString(2)`).
    pub fn to_string(&self, id: TypeId) -> String {
        self.debug_string(id, 2)
    }
    /// `prettyString()` (= `debugString(0)`).
    pub fn pretty_string(&self, id: TypeId) -> String {
        self.debug_string(id, 0)
    }

    /// `realizedName()`.
    pub fn realized_name(&self, id: TypeId) -> String {
        match self.shape(id) {
            Shape::Link(LinkKind::Link, t) => self.realized_name(t.unwrap()),
            Shape::Link(..) => self.link(id).map(|l| format!("#{}", l.generic_name)).unwrap(),
            Shape::Func => self.func_realized_name(id),
            // StaticType::realizedName = debugString(0)
            Shape::Static => self.with_class(id, |c| c.clone()).map(|c| self.static_debug_string(&c, 0)).unwrap(),
            Shape::Class { .. } => self.class_realized_name(id),
            Shape::Trait => String::new(),
        }
    }

    /// `getNonStaticType()` of a static type (`cache->findClass`); `None` if the class
    /// is not known yet.
    pub fn non_static_type(&self, id: TypeId) -> Option<TypeId> {
        let s = self.get_static(id)?;
        let name = self.with_class(s, |c| match c.kind {
            ClassKind::IntStatic(_) => "Int[64]".to_string(),
            _ => c.name.to_string(),
        })?;
        self.find_class(&name)
    }
}

/// Pretty printer for AST dumps (`AstArena::type_fmt`).
impl crate::ast::TypeFormatter for TypeArena {
    fn debug_string(&self, ty: TypeId, mode: i32) -> String {
        TypeArena::debug_string(self, ty, mode)
    }
    fn realized_name(&self, ty: TypeId) -> String {
        TypeArena::realized_name(self, ty)
    }
    fn is_func(&self, ty: TypeId) -> bool {
        self.get_func(ty).is_some()
    }
}

/// Visited set helper.
pub(crate) type Visited = FxHashSet<TypeId>;
