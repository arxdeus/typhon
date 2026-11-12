//! `TypecheckItem`, `TypeContext`.
//!
//! Contexts are shared (`Rc<RefCell<TypeContext>>` = [`CtxRef`]); items are shared and
//! mutable (`Rc<RefCell<TypecheckItem>>` = [`ItemRef`]), exactly like shared pointers
//! `shared_ptr`s. Operations that in the reference reach `ctx->cache` (stdlib lookup, the root
//! `typeCtx`, reverse identifier lookup) are associated functions taking the cache and
//! the context handle explicitly (`TypeContext::find_in`, `add_always_visible`, ...).
//! Never hold a `RefCell` borrow of a context across a call that may touch other contexts.

use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::rc::Rc;

use rustc_hash::FxHashMap;
use typhon_common::fs::{ImportFile, ImportStatus};
use typhon_common::mangle::stdlib;
use typhon_common::SrcInfo;
use ustr::Ustr;

use crate::cache::{Cache, STDLIB_IMPORT};
use crate::ctx::Context;
use crate::ids::{StmtId, TypeId};
use crate::types::{LiteralKind, TypeArena};

/// Shared context handle.
pub type CtxRef = Rc<RefCell<TypeContext>>;
/// Shared item handle.
pub type ItemRef = Rc<RefCell<TypecheckItem>>;

/// Typecheck context identifier: a function, a class (type), or a variable.
#[derive(Clone, Debug)]
pub struct TypecheckItem {
    /// Unique identifier (canonical name).
    pub canonical_name: String,
    /// Base name (e.g., foo.bar.baz).
    pub base_name: Ustr,
    /// Full module name.
    pub module_name: Ustr,
    /// Type.
    pub ty: Option<TypeId>,
    /// Number of nested conditionals (blocks).
    pub block_level: i32,
    /// When the name was added to the context.
    pub time: i64,
    /// Set if an identifier is a class or a function generic.
    pub generic: bool,
    /// Alternative item (used within isinstance blocks).
    pub alternative: Option<ItemRef>,
    pub src: SrcInfo,
}

impl TypecheckItem {
    pub fn new(canonical_name: impl Into<String>, base_name: impl AsRef<str>, module_name: impl AsRef<str>, ty: Option<TypeId>, block_level: i32) -> Self {
        Self {
            canonical_name: canonical_name.into(),
            base_name: Ustr::from(base_name.as_ref()),
            module_name: Ustr::from(module_name.as_ref()),
            ty,
            block_level,
            time: 0,
            generic: false,
            alternative: None,
            src: SrcInfo::default(),
        }
    }
    pub fn into_ref(self) -> ItemRef {
        Rc::new(RefCell::new(self))
    }
    pub fn base_name(&self) -> &str {
        self.base_name.as_str()
    }
    pub fn module(&self) -> &str {
        self.module_name.as_str()
    }
    pub fn name(&self) -> &str {
        &self.canonical_name
    }
    pub fn ty(&self) -> TypeId {
        self.ty.expect("item without type")
    }
    /// `isFunc()`.
    pub fn is_func(&self, types: &TypeArena) -> bool {
        self.ty.is_some_and(|t| types.get_func(t).is_some())
    }
    /// `isType()`.
    pub fn is_type(&self, types: &TypeArena) -> bool {
        self.ty.is_some_and(|t| types.is(t, &stdlib().type_))
    }
    /// `isVar()`.
    pub fn is_var(&self, types: &TypeArena) -> bool {
        !self.generic && !self.is_func(types) && !self.is_type(types)
    }
    /// `isGlobal()`.
    pub fn is_global(&self) -> bool {
        self.block_level == 0 && self.base_name.is_empty()
    }
    /// `isConditional()`.
    pub fn is_conditional(&self) -> bool {
        self.block_level > 0
    }
    pub fn is_generic(&self) -> bool {
        self.generic
    }
    /// `getStaticKind()`.
    pub fn static_kind(&self, types: &TypeArena) -> LiteralKind {
        self.ty.map(|t| types.static_kind(t)).unwrap_or(LiteralKind::Runtime)
    }
}

/// `TypeContext::Base::Loop`.
#[derive(Clone, Debug)]
pub struct Loop {
    pub break_var: String,
    /// False if a loop has continue/break statement.
    pub flat: bool,
}

impl Loop {
    pub fn new(break_var: impl Into<String>) -> Self {
        Self { break_var: break_var.into(), flat: true }
    }
}

/// `TypeContext::Base::deduce`.
#[derive(Clone, Debug, Default)]
pub struct Deduce {
    /// Class fields in order of traversal (for `@deduce` classes).
    pub deduced_members: Option<Rc<RefCell<Vec<String>>>>,
    /// Canonical name of `self`.
    pub self_name: String,
}

/// A function or class block (`TypeContext::Base`).
#[derive(Clone, Debug, Default)]
pub struct Base {
    /// Canonical name of a function or a class that owns this base.
    pub name: Ustr,
    /// Function type.
    pub ty: Option<TypeId>,
    /// The return type of currently realized function.
    pub return_type: Option<TypeId>,
    /// Typechecking iteration.
    pub iteration: i32,
    /// Only set for functions.
    pub func: Option<StmtId>,
    pub suite: Option<StmtId>,
    /// Collect reached returns before committing to a result type.
    pub infer_return_type: bool,
    /// Reached returns and whether each was originally bare.
    pub return_statements: Vec<(StmtId, bool)>,
    /// Index of the parent base.
    pub parent: i32,
    pub deduce: Deduce,
    /// Stack of nested loops.
    pub loops: Vec<Loop>,
    /// Priority -> unbounds with defaults.
    pub pending_defaults: BTreeMap<i32, BTreeSet<TypeId>>,
    pub recursive_dependencies: BTreeSet<TypeId>,
}

impl Base {
    pub const NONE_ARGUMENT_DEFAULT_PRIORITY: i32 = -1;

    pub fn new(name: impl AsRef<str>, ty: Option<TypeId>, return_type: Option<TypeId>) -> Self {
        Self { name: Ustr::from(name.as_ref()), ty, return_type, ..Default::default() }
    }
    /// `getLoop()`.
    pub fn get_loop(&mut self) -> Option<&mut Loop> {
        self.loops.last_mut()
    }
    /// `isType()`.
    pub fn is_type(&self) -> bool {
        self.func.is_none()
    }
}

/// Context that tracks identifiers during typechecking.
#[derive(Clone)]
pub struct TypeContext {
    pub base: Context<ItemRef>,
    pub types: Rc<TypeArena>,
    /// Base stack (the last enclosing base is last).
    pub bases: Vec<Base>,
    module_info: ImportFile,
    module_identifier: Ustr,
    /// Set if the standard library is currently being loaded.
    pub is_stdlib_loading: bool,
    /// Current type-checking level (for instantiation and generalization).
    pub typecheck_level: i32,
    pub changed_nodes: i32,
    /// Number of nested realizations.
    pub realization_depth: i32,
    /// Number of nested blocks (0 for toplevel).
    pub block_level: i32,
    /// True if an early return is found.
    pub return_early: bool,
    /// Stack of static loop control variables.
    pub static_loops: Vec<String>,
    /// Current statement time.
    pub time: i64,
    /// Type to be expected upon completed typechecking.
    pub expected_type: Option<TypeId>,
    /// True if no side-effects allowed.
    pub simple_types: bool,
    pub global_shadows: FxHashMap<String, usize>,
    pub itime: FxHashMap<String, f64>,
}

/// State saved for a speculative realization (`saveInferenceState`).
pub struct SavedInferenceState(pub TypeContext);

impl TypeContext {
    /// Bootstrap context used before the cache exists.
    pub(crate) fn placeholder() -> Self {
        Self::raw(Rc::new(TypeArena::new()), "")
    }

    fn raw(types: Rc<TypeArena>, filename: &str) -> Self {
        Self {
            base: Context::new(filename),
            types,
            bases: vec![Base::default()],
            module_info: ImportFile { status: ImportStatus::Package, path: String::new(), module: String::new() },
            module_identifier: Ustr::from(""),
            is_stdlib_loading: false,
            typecheck_level: 0,
            changed_nodes: 0,
            realization_depth: 0,
            block_level: 0,
            return_early: false,
            static_loops: vec![],
            time: 0,
            expected_type: None,
            simple_types: false,
            global_shadows: FxHashMap::default(),
            itime: FxHashMap::default(),
        }
    }

    /// `TypeContext(cache, filename)`.
    pub fn new(cache: &mut Cache, filename: &str) -> Self {
        let mut c = Self::raw(cache.types.clone(), filename);
        let src = cache.generate_src_info();
        c.base.push_node(src); // Always have srcInfo() around
        c
    }

    /// Wrap into a [`CtxRef`].
    pub fn into_ref(self) -> CtxRef {
        Rc::new(RefCell::new(self))
    }

    // ------------------------------------------------------------------
    // checkpoints
    // ------------------------------------------------------------------

    /// `saveInferenceState()`: copy of everything but the bindings.
    pub fn save_inference_state(&mut self) -> SavedInferenceState {
        let bindings = std::mem::take(&mut self.base.map);
        let saved = self.clone();
        self.base.map = bindings;
        SavedInferenceState(saved)
    }
    /// `restoreInferenceState(saved)`: restore everything but the bindings.
    pub fn restore_inference_state(&mut self, saved: SavedInferenceState) {
        let mut saved = saved.0;
        std::mem::swap(&mut saved.base.map, &mut self.base.map);
        // keep active checkpoints of the live context
        let mut cps = vec![];
        while let Some(c) = self.base.pop_checkpoint() {
            cps.push(c);
        }
        *self = saved;
        // Note: checkpoints are owned by the context here; carry them over.
        for c in cps.into_iter().rev() {
            let i = self.base.push_checkpoint(c.wants_state);
            if let Some(slot) = self.base.checkpoint_mut(i) {
                *slot = c;
            }
        }
    }
    /// `prepareCheckpoint()`: prepare all checkpoints, snapshotting the inference state for
    /// those that asked for it.
    pub fn prepare_checkpoint(&mut self) {
        if !self.base.has_checkpoints() {
            return;
        }
        let need = self.base.prepare_checkpoint();
        if !need.is_empty() {
            let snap = self.save_inference_state();
            let mut snap = Some(snap);
            for (k, i) in need.iter().enumerate() {
                let s: Box<dyn std::any::Any> = if k + 1 == need.len() {
                    Box::new(snap.take().unwrap())
                } else {
                    Box::new(SavedInferenceState(snap.as_ref().unwrap().0.clone()))
                };
                if let Some(c) = self.base.checkpoint_mut(*i) {
                    c.state = Some(s);
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // bindings
    // ------------------------------------------------------------------

    /// `add(name, var)`.
    pub fn add(&mut self, name: &str, var: ItemRef) {
        self.prepare_checkpoint();
        self.base.add(name, var);
    }
    /// `remove(name)`.
    pub fn remove(&mut self, name: &str) {
        self.prepare_checkpoint();
        self.base.remove(name);
    }
    /// `addBlock()`.
    pub fn add_block(&mut self) {
        self.prepare_checkpoint();
        self.base.add_block();
    }
    /// `popBlock()`.
    pub fn pop_block(&mut self) {
        self.prepare_checkpoint();
        self.base.pop_block();
    }
    /// `removeFromMap(name)`.
    pub fn remove_from_map(&mut self, name: &str) {
        self.prepare_checkpoint();
        self.base.remove_from_map(name);
    }
    /// `addToplevel(name, item)`.
    pub fn add_toplevel(&mut self, name: &str, item: ItemRef) -> ItemRef {
        self.prepare_checkpoint();
        self.base.add_binding_only(name, item.clone());
        item
    }
    /// `Context::find(name)` (plain lookup, no base/time/stdlib logic).
    pub fn find_raw(&self, name: &str) -> Option<ItemRef> {
        self.base.find(name)
    }
    pub fn block(&self) -> &Vec<String> {
        self.base.block()
    }
    pub fn stack_size(&self) -> usize {
        self.base.stack_size()
    }
    pub fn filename(&self) -> String {
        self.base.filename().to_string()
    }
    pub fn set_filename(&mut self, f: &str) {
        self.base.set_filename(f)
    }
    pub fn src_info(&self) -> SrcInfo {
        self.base.src_info()
    }

    fn make_item(&self, canonical: &str, ty: Option<TypeId>, src: SrcInfo) -> ItemRef {
        let mut t = TypecheckItem::new(canonical, self.base_name(), self.module(), ty, self.block_level);
        t.src = src;
        t.into_ref()
    }

    /// `addVar(name, canonicalName, type, time, srcInfo)`.
    pub fn add_var(cache: &Cache, ctx: &CtxRef, name: &str, canonical: &str, ty: TypeId, time: i64, src: SrcInfo) -> ItemRef {
        typhon_common::seqassert!(!canonical.is_empty(), "empty canonical name for '{}'", name);
        let t = ctx.borrow().make_item(canonical, Some(ty), src);
        t.borrow_mut().time = time;
        ctx.borrow_mut().add(name, t.clone());
        Self::add_always_visible(cache, ctx, t.clone(), false);
        t
    }
    /// `addType(name, canonicalName, type, srcInfo)`.
    pub fn add_type(cache: &Cache, ctx: &CtxRef, name: &str, canonical: &str, ty: TypeId, src: SrcInfo) -> ItemRef {
        typhon_common::seqassert!(!canonical.is_empty(), "empty canonical name for '{}'", name);
        let t = ctx.borrow().make_item(canonical, Some(ty), src);
        ctx.borrow_mut().add(name, t.clone());
        Self::add_always_visible(cache, ctx, t.clone(), false);
        t
    }
    /// `addFunc(name, canonicalName, type, srcInfo)`.
    pub fn add_func(cache: &Cache, ctx: &CtxRef, name: &str, canonical: &str, ty: TypeId, src: SrcInfo) -> ItemRef {
        typhon_common::seqassert!(!canonical.is_empty(), "empty canonical name for '{}'", name);
        typhon_common::seqassert!(cache.types.get_func(ty).is_some(), "bad func");
        let t = ctx.borrow().make_item(canonical, Some(ty), src);
        ctx.borrow_mut().add(name, t.clone());
        Self::add_always_visible(cache, ctx, t.clone(), false);
        t
    }
    /// `addAlwaysVisible(item, pop)`: also register the item (by canonical name) in the
    /// root context so realizations are visible everywhere.
    pub fn add_always_visible(cache: &Cache, ctx: &CtxRef, item: ItemRef, pop: bool) -> ItemRef {
        let cname = item.borrow().canonical_name.clone();
        {
            let mut c = ctx.borrow_mut();
            c.add(&cname, item.clone());
            if pop {
                c.base.block_mut().pop(); // do not remove it later!
            }
        }
        let root = &cache.type_ctx;
        let exists = if Rc::ptr_eq(root, ctx) {
            true // just added
        } else {
            root.borrow().base.find(&cname).is_some()
        };
        if !exists {
            {
                let mut r = root.borrow_mut();
                r.add(&cname, item.clone());
                if pop {
                    r.base.block_mut().pop();
                }
            }
            // Realizations etc.
            if !cache.has_reverse(&cname) {
                cache.set_reverse(cname.clone(), cname.clone());
            }
        }
        item
    }

    /// `find(name, time, inBase)`: lookup with base/time domination rules, falling back to
    /// the standard library and the root context.
    pub fn find_in(cache: &Cache, ctx: &CtxRef, name: &str, time: i64, in_base: Option<&str>) -> Option<ItemRef> {
        {
            let c = ctx.borrow();
            if let Some(stack) = c.base.map.get(name) {
                let is_mangled = name.contains('.');
                if is_mangled {
                    return stack.first().cloned();
                }
                let cur_base = c.bases.last().map(|b| b.name.as_str().to_string()).unwrap_or_default();
                let base: &str = in_base.unwrap_or(&cur_base);
                let module = c.module_identifier;
                for i in stack.iter() {
                    let ib = i.borrow();
                    let same_base = if in_base.is_some() { ib.base_name.as_str() == base } else { ib.base_name.as_str() == cur_base };
                    if !same_base && !base.starts_with(ib.base_name.as_str()) {
                        continue; // avoid middle realizations
                    }
                    if !same_base || time == 0 || ib.module_name != module || ib.time <= time {
                        return Some(i.clone());
                    }
                }
            }
        }
        // Item is not found in the current module. Time to look in the standard library!
        let mut t = None;
        if let Some(stdlib) = cache.imports.get(STDLIB_IMPORT).and_then(|i| i.ctx.clone()) {
            if !Rc::ptr_eq(&stdlib, ctx) {
                t = stdlib.borrow().base.find(name);
            }
        }
        // Maybe we are looking for a canonical identifier?
        if t.is_none() && !Rc::ptr_eq(&cache.type_ctx, ctx) {
            t = cache.type_ctx.borrow().base.find(name);
        }
        t
    }

    /// `forceFind(name)`.
    pub fn force_find_in(cache: &Cache, ctx: &CtxRef, name: &str) -> ItemRef {
        match Self::find_in(cache, ctx, name, 0, None) {
            Some(f) => f,
            None => panic!("cannot find '{name}'"),
        }
    }

    // ------------------------------------------------------------------
    // getters
    // ------------------------------------------------------------------

    /// `getBaseName()`.
    pub fn base_name(&self) -> Ustr {
        self.bases.last().map(|b| b.name).unwrap_or_default()
    }
    /// `getModule()`.
    pub fn module(&self) -> Ustr {
        self.module_identifier
    }
    /// `setModule(module)`.
    pub fn set_module(&mut self, module: ImportFile) {
        let mut base = if module.status == ImportStatus::Stdlib { "std.".to_string() } else { String::new() };
        base += &module.module;
        if let Some(rest) = base.strip_prefix("__main__") {
            base = rest.to_string();
        }
        self.module_identifier = Ustr::from(&base);
        self.module_info = module;
    }
    pub fn module_info(&self) -> &ImportFile {
        &self.module_info
    }
    /// `getModulePath()`.
    pub fn module_path(&self) -> String {
        self.module_info.path.clone()
    }

    /// `generateCanonicalName(name, includeBase, noSuffix)`.
    pub fn generate_canonical_name(&self, cache: &mut Cache, name: &str, include_base: bool, no_suffix: bool) -> String {
        if name.contains('.') {
            return name.to_string();
        }
        let mut new_name = name.to_string();
        let include_base = include_base && !name.starts_with('%');
        let mut no_suffix = no_suffix;
        if include_base {
            let mut base = self.base_name().to_string();
            if base.is_empty() {
                base = self.module().to_string();
            }
            if base == "std.internal.core" {
                no_suffix = true;
                base = String::new();
            }
            new_name = if base.is_empty() { new_name } else { format!("{base}.{new_name}") };
        }
        let num = {
            let e = cache.identifier_count.entry(new_name.clone()).or_insert(0);
            let n = *e;
            *e += 1;
            n
        };
        if !no_suffix {
            new_name = format!("{new_name}.{num}");
        }
        if name != new_name {
            *cache.identifier_count.entry(new_name.clone()).or_insert(0) += 1;
        }
        cache.set_reverse(new_name.clone(), name);
        new_name
    }

    /// `isGlobal()`.
    pub fn is_global(&self) -> bool {
        self.bases.len() == 1
    }
    /// `isConditional()`.
    pub fn is_conditional(&self) -> bool {
        self.block_level > 0
    }
    /// `getBase()`.
    pub fn get_base(&mut self) -> &mut Base {
        self.bases.last_mut().expect("no base")
    }
    pub fn base_ref(&self) -> &Base {
        self.bases.last().expect("no base")
    }
    /// `inFunction()`.
    pub fn in_function(&self) -> bool {
        !self.is_global() && !self.base_ref().is_type()
    }
    /// `inClass()`.
    pub fn in_class(&self) -> bool {
        !self.is_global() && self.base_ref().is_type()
    }
    /// `isOuter(val)`.
    pub fn is_outer(&self, val: &ItemRef) -> bool {
        let v = val.borrow();
        self.base_name() != v.base_name || self.module() != v.module_name
    }
    /// `getClassBase()`.
    pub fn class_base(&mut self) -> Option<&mut Base> {
        let n = self.bases.len();
        if n >= 2 && self.bases[n - 2].is_type() { Some(&mut self.bases[n - 2]) } else { None }
    }
    /// `getRealizationDepth()`.
    pub fn realization_depth(&self) -> usize {
        self.bases.len()
    }
    /// `getRealizationStackName()`.
    pub fn realization_stack_name(&self) -> String {
        let s: Vec<String> = self.bases.iter().filter_map(|b| b.ty.map(|t| self.types.realized_name(t))).collect();
        s.join(":")
    }

    /// `dump()`.
    pub fn dump(&self) {
        let mut ordered: Vec<_> = self.base.map.iter().collect();
        ordered.sort_by(|a, b| a.0.cmp(b.0));
        typhon_common::logger::log_line(&format!("current module: {} ({})", self.module_info.module, self.module_info.path));
        typhon_common::logger::log_line(&format!("current base:   {} / {}", self.realization_stack_name(), self.base_name()));
        for (name, stack) in ordered {
            let Some(t) = stack.first() else { continue };
            let t = t.borrow();
            let types = &self.types;
            typhon_common::logger::log_line(&format!("{:.<25}", name));
            typhon_common::logger::log_line(&format!(
                "   ... kind:      {}",
                (t.is_type(types) as i32) * 100 + (t.is_func(types) as i32) * 10 + t.is_var(types) as i32
            ));
            typhon_common::logger::log_line(&format!("   ... canonical: {}", t.canonical_name));
            typhon_common::logger::log_line(&format!("   ... base:      {}", t.base_name));
            typhon_common::logger::log_line(&format!("   ... module:    {}", t.module_name));
            typhon_common::logger::log_line(&format!(
                "   ... type:      {}",
                t.ty.map(|x| types.debug_string(x, 2)).unwrap_or_else(|| "<null>".into())
            ));
            typhon_common::logger::log_line(&format!(
                "   ... gnrc/sttc: {} / {}",
                t.generic,
                crate::types::literal_index(t.static_kind(types))
            ));
        }
    }

    /// `debugInfo()`.
    pub fn debug_info(&self) -> String {
        format!("[{}:i{}@{:?}]", self.base_name(), self.base_ref().iteration, self.src_info())
    }
}
