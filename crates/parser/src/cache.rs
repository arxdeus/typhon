//! Data shared across front-end stages.
//!
//! Ownership model (see `TYPECHECK_API.md`):
//! * `Cache` is owned by the compiler (usually as `Rc<RefCell<Cache>>`, see
//!   [`CacheRealizer`]); visitors borrow it as `&mut Cache` for their whole lifetime and
//!   re-borrow it for nested visitors (`TypecheckVisitor v(ctx, ...)`).
//! * `Cache::ast` owns every AST node (`N<T>` / `_nodes`), `Cache::types` owns every
//!   type-checker type (`Rc` so it can be used while `&mut Cache` is lent out).
//! * Contexts are `Rc<RefCell<TypeContext>>` ([`CtxRef`]).
//! * The IR module is `Rc<RefCell<typhon_ir::Module>>`.

use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::rc::Rc;

use indexmap::IndexMap;
use rustc_hash::{FxHashMap, FxHashSet};

/// Insertion-ordered map with the fast (non-SipHash) hasher; iteration order is unaffected.
pub type FxIndexMap<K, V> = IndexMap<K, V, rustc_hash::FxBuildHasher>;
use typhon_common::fs::FileSystem;
use typhon_common::mangle::{self, stdlib};
use typhon_common::{CompilerError, Options, ParserErrors, Result, SrcInfo};
use typhon_ir::{AstTypeRef, FuncId, Module as IrModule, TypeRealizer};

use crate::ast::{AstArena, Attr};
use crate::ids::{ExprId, StmtId, TypeId};
use crate::typecheck::ctx::{CtxRef, TypeContext};
use crate::typecheck::TypecheckVisitor;
use crate::types::{TypeArena, Unification};

pub const FILE_GENERATED: &str = "<generated>";
pub const MODULE_MAIN: &str = "__main__";
pub const STDLIB_INTERNAL_MODULE: &str = "internal";
pub const MAIN_IMPORT: &str = "";
pub const STDLIB_IMPORT: &str = ":stdlib:";
pub const FN_DISPATCH_SUFFIX: &str = ":dispatch";
pub const FN_SETTER_SUFFIX: &str = ":set_";
pub const VAR_CLASS_TOPLEVEL: &str = ":toplevel";
pub const VAR_USED_SUFFIX: &str = ":used";
pub const MAX_ERRORS: usize = 5;
pub const MAX_TUPLE: usize = 2048;
pub const MAX_INT_WIDTH: i32 = 10000;
pub const MAX_REALIZATION_DEPTH: usize = 200;
pub const MAX_STATIC_ITER: usize = 1024;

/// `VAR_ARGV`.
pub fn var_argv() -> String {
    mangle::var("", "__argv__")
}
/// `VAR_ARGC`.
pub fn var_argc() -> String {
    mangle::var("", "__argc__")
}
/// `FN_OPTIONAL_UNWRAP`.
pub fn fn_optional_unwrap() -> String {
    mangle::func("std.internal.types.optional", "unwrap")
}

/// Holds module import data (`Cache::Module`).
#[derive(Clone, Default)]
pub struct ImportModule {
    /// Relative module name (e.g., `foo.bar`).
    pub name: String,
    /// Absolute filename of an import.
    pub filename: String,
    /// Import typechecking context.
    pub ctx: Option<CtxRef>,
    /// Unique import variable for checking already loaded imports.
    pub import_var: String,
    /// File content (line:col indexable).
    pub content: Vec<String>,
    /// Set if loaded at toplevel.
    pub loaded_at_toplevel: bool,
}

impl ImportModule {
    pub fn new() -> Self {
        Self { loaded_at_toplevel: true, ..Default::default() }
    }
    /// `update(name, filename, ctx)`.
    pub fn update(&mut self, name: &str, filename: &str, ctx: CtxRef) {
        self.name = name.to_string();
        self.filename = filename.to_string();
        self.ctx = Some(ctx);
    }
}

/// A class field (`Cache::Class::ClassField`).
#[derive(Clone, Debug)]
pub struct ClassField {
    /// Field name.
    pub name: String,
    /// A corresponding generic field type.
    pub ty: Option<TypeId>,
    /// Base class name (if available).
    pub base_class: String,
    pub type_expr: Option<ExprId>,
}

impl ClassField {
    pub fn new(name: impl Into<String>, ty: Option<TypeId>, base_class: impl Into<String>, type_expr: Option<ExprId>) -> Self {
        Self { name: name.into(), ty, base_class: base_class.into(), type_expr }
    }
}

/// A class realization (`Cache::Class::ClassRealization`).
#[derive(Clone, Debug, Default)]
pub struct ClassRealization {
    /// Realized class type.
    pub ty: Option<TypeId>,
    /// Field names and realized field types.
    pub fields: Vec<(String, TypeId)>,
    /// IR type.
    pub ir: Option<typhon_ir::TypeId>,
    /// Bases (in MRO order).
    pub bases: Vec<TypeId>,
    /// Realization vtable: method id -> thunk realization.
    pub vtable: BTreeMap<usize, TypeId>,
    /// Realization ID.
    pub id: usize,
}

pub type ClassRealizationRef = Rc<RefCell<ClassRealization>>;

/// Class data (`Cache::Class`).
#[derive(Clone, Debug, Default)]
pub struct Class {
    /// Module information.
    pub module: String,
    /// Generic (unrealized) class template AST.
    pub ast: Option<StmtId>,
    /// Non-simplified AST. Used for base class instantiation.
    pub original_ast: Option<StmtId>,
    /// Method lookup table: non-canonical name -> root function name.
    pub methods: FxHashMap<String, String>,
    /// Ordered list of fields.
    pub fields: Vec<ClassField>,
    /// Class variables: name -> canonical name.
    pub class_vars: FxHashMap<String, String>,
    /// Realized class name -> realization.
    pub realizations: FxHashMap<String, ClassRealizationRef>,
    /// Set if a class is polymorphic and has RTTI.
    pub rtti: bool,
    /// Virtual method names.
    pub virtuals: FxHashSet<String>,
    /// MRO (class types).
    pub mro: Vec<TypeId>,
    /// Classes whose base is this class.
    pub descendants: FxHashSet<String>,
}

impl Class {
    /// `hasRTTI()` (`ast_name`: canonical name of `ast`).
    pub fn has_rtti(&self, ast_name: &str) -> bool {
        self.mro.len() > 1 || ast_name == stdlib().object.as_str()
    }
}

/// A function realization (`Cache::Function::FunctionRealization`).
#[derive(Clone, Debug, Default)]
pub struct FunctionRealization {
    /// Realized function type.
    pub ty: Option<TypeId>,
    /// Realized function AST.
    pub ast: Option<StmtId>,
    /// IR function.
    pub ir: Option<FuncId>,
    /// Resolved captures.
    pub captures: Vec<String>,
    /// Whether this constructor requires an independent field-initialization check.
    pub check_initialization: bool,
    /// (AST, type) for which the field-initialization check has passed.
    pub initialization_checked: Option<(StmtId, TypeId)>,
}

pub type FunctionRealizationRef = Rc<RefCell<FunctionRealization>>;

/// Function data (`Cache::Function`).
#[derive(Clone, Debug, Default)]
pub struct Function {
    pub module: String,
    pub root_name: String,
    /// Generic (unrealized) function template AST.
    pub ast: Option<StmtId>,
    /// Unrealized function type.
    pub ty: Option<TypeId>,
    pub is_toplevel: bool,
    /// Realized name -> realization.
    pub realizations: FxHashMap<String, FunctionRealizationRef>,
    pub captures: BTreeSet<String>,
}

/// Hooks into other front-end lanes that the cache needs (parser, scoping, translation).
/// the reference calls these functions directly; Rust breaks the module cycle through hooks
/// installed by the compiler driver.
#[derive(Clone, Default)]
pub struct CacheHooks {
    /// `ast::parseCode(cache, file, code, startLine)`.
    pub parse_code: Option<Rc<dyn Fn(&mut Cache, &str, &str, i32) -> Result<StmtId>>>,
    /// `ast::parseFile(cache, file)`.
    pub parse_file: Option<Rc<dyn Fn(&mut Cache, &str) -> Result<StmtId>>>,
    /// `ScopingVisitor::apply(cache, stmt, &globalShadows)`; the `CtxRef` (if any) is the
    /// context whose `global_shadows` must be updated.
    pub scoping: Option<Rc<dyn Fn(&mut Cache, StmtId, Option<CtxRef>) -> Result<()>>>,
    /// `TranslateVisitor(codegenCtx).translateStmts(stmt)`.
    pub translate: Option<Rc<dyn Fn(&mut Cache, StmtId) -> Result<()>>>,
    /// `ScopingVisitor::apply(cache, n, &ictx->globalShadows, dominateAll)` for new imports.
    pub scoping_import: Option<Rc<dyn Fn(&mut Cache, StmtId, CtxRef, bool) -> Result<()>>>,
    /// `Cache::parseCode` IR part (translate the typechecked `node` into a fresh series
    /// flow attached to main); returns the series flow value.
    pub translate_parsed: Option<Rc<dyn Fn(&mut Cache, StmtId) -> Result<typhon_ir::ValueId>>>,
    /// `TranslateVisitor::apply(cache, stmts)` (whole-program translation; returns main).
    pub translate_apply: Option<Rc<dyn Fn(&mut Cache, StmtId) -> Result<typhon_ir::FuncId>>>,
}

/// Cache encapsulation that holds data structures shared across various transformation
/// stages (AST transformation, type checking etc.).
pub struct Cache {
    /// Filesystem object used for accessing files.
    pub fs: Option<Rc<RefCell<dyn FileSystem>>>,
    /// Count for each identifier (name) seen in the code.
    pub identifier_count: FxHashMap<String, i32>,
    /// Number of code-generated source code positions.
    pub generated_src_info_count: i32,
    /// Number of auto-generated variables so far (`getTemporaryVar`).
    pub var_count: i32,
    /// Scope counter. Each conditional block gets a new scope ID.
    pub block_count: i32,

    /// Compiler options (`compiler->getOptions()`).
    pub options: Option<Rc<Options>>,
    /// IR module.
    pub module: Option<Rc<RefCell<IrModule>>>,

    /// Imported files: absolute filename -> import (":stdlib:" = stdlib, "" = main).
    pub imports: IndexMap<String, ImportModule>,
    /// Canonical global identifiers -> IR variable.
    pub globals: BTreeMap<String, Option<typhon_ir::VarId>>,

    /// Canonical class name -> class.
    pub classes: FxIndexMap<String, Class>,
    pub class_realization_cnt: usize,

    pub thunk_ids: BTreeMap<String, usize>,
    pub super_calls: BTreeMap<usize, TypeId>,
    pub super_call_signatures: BTreeMap<usize, TypeId>,

    /// Canonical function name -> function.
    pub functions: FxIndexMap<String, Function>,
    /// Root name -> canonical overload names.
    pub overloads: FxHashMap<String, Vec<String>>,

    /// Root type-checking context (`typeCtx`).
    pub type_ctx: CtxRef,
    /// Translation context (`codegenCtx`), owned by the translate lane.
    pub codegen_ctx: Option<Rc<dyn std::any::Any>>,
    /// Function realizations that are to be translated to IR: (function, realization).
    pub pending_realizations: BTreeSet<(String, String)>,

    /// Union inference for unannotated empty collection literals.
    pub empty_literal_unions: BTreeMap<String, Vec<TypeId>>,
    /// Container types of empty literals in flight: type -> ((slot, key)s, type).
    pub empty_literal_slots: BTreeMap<TypeId, (Vec<(usize, String)>, TypeId)>,
    /// Realizations that must be restarted due to a widened slot.
    pub empty_literal_restarts: BTreeSet<String>,
    /// Python int->float variable widening: declaration sites (`owner\x01file:line:col`)
    /// of function locals bound to a bare int literal that are later assigned a float.
    /// On restart of the owner the literal is emitted as a float.
    pub int_widen_sites: BTreeSet<String>,
    /// Canonical variable name -> declaration site of locals bound to a bare int literal.
    pub int_literal_decls: BTreeMap<String, String>,


    pub generated_tuples: FxHashSet<usize>,
    pub generated_kw_tuples: FxHashMap<String, i32>,
    pub errors: ParserErrors,

    /// All AST nodes (`_nodes` / `N<T>`).
    pub ast: AstArena,
    /// All type-checker types.
    pub types: Rc<TypeArena>,
    /// Hooks into other lanes.
    pub hooks: CacheHooks,

    /// `_timings`.
    pub timings: FxHashMap<String, f64>,
}

impl Cache {
    /// `Cache(nodes, argv0, fs, nodeStorage)`.
    pub fn new(fs: Option<Rc<RefCell<dyn FileSystem>>>) -> Self {
        let types = Rc::new(TypeArena::new());
        let mut ast = AstArena::new();
        let fmt: Rc<dyn crate::ast::TypeFormatter> = types.clone();
        ast.type_fmt = Some(fmt);
        let mut c = Cache {
            fs,
            identifier_count: FxHashMap::default(),
            generated_src_info_count: 0,
            var_count: 0,
            block_count: 1,
            options: None,
            module: None,
            imports: IndexMap::new(),
            globals: BTreeMap::new(),
            classes: FxIndexMap::default(),
            class_realization_cnt: 0,
            thunk_ids: BTreeMap::new(),
            super_calls: BTreeMap::new(),
            super_call_signatures: BTreeMap::new(),
            functions: FxIndexMap::default(),
            overloads: FxHashMap::default(),
            type_ctx: Rc::new(RefCell::new(TypeContext::placeholder())),
            codegen_ctx: None,
            pending_realizations: BTreeSet::new(),
            empty_literal_unions: BTreeMap::new(),
            empty_literal_slots: BTreeMap::new(),
            empty_literal_restarts: BTreeSet::new(),
            int_widen_sites: BTreeSet::new(),
            int_literal_decls: BTreeMap::new(),
            generated_tuples: FxHashSet::default(),
            generated_kw_tuples: FxHashMap::default(),
            errors: ParserErrors::default(),
            ast,
            types,
            hooks: CacheHooks::default(),
            timings: FxHashMap::default(),
        };
        let root = TypeContext::new(&mut c, ".root");
        c.type_ctx = Rc::new(RefCell::new(root));
        // `cache->findClass(name)` for `StaticType::getNonStaticType` (types need it to
        // unify literals with their runtime classes, e.g. `Literal[64]` with `int`).
        // Classes and their realizations are always visible in the root context.
        let root = Rc::downgrade(&c.type_ctx);
        let arena = Rc::downgrade(&c.types);
        c.types.set_class_resolver(Some(Rc::new(move |name: &str| {
            let (root, types) = (root.upgrade()?, arena.upgrade()?);
            let f = root.try_borrow().ok()?.base.find(name)?;
            let t = f.try_borrow().ok()?.ty?;
            if !types.is(t, &stdlib().type_) {
                return None;
            }
            let g = types.generic_type(types.get_class(t)?, 0)?;
            types.get_class(g)
        })));
        c
    }

    /// Unbound counter (`unboundCount`).
    pub fn unbound_count(&self) -> i32 {
        self.types.unbound_count.get()
    }

    /// `getTemporaryVar(prefix, sigil)`: `"{sigil}_{prefix}{counter}"`.
    pub fn temporary_var(&mut self, prefix: &str, sigil: Option<char>) -> String {
        self.var_count += 1;
        match sigil {
            Some(s) => format!("{s}_{prefix}_{}", self.var_count),
            None => format!("{prefix}_{}", self.var_count),
        }
    }
    /// `getTemporaryVar(prefix)` with the default `%` sigil.
    pub fn get_temporary_var(&mut self, prefix: &str) -> String {
        self.temporary_var(prefix, Some('%'))
    }

    /// `rev(s)`: non-canonical version of a canonical name.
    pub fn rev(&self, s: &str) -> String {
        self.types.rev(s)
    }
    /// `reverseIdentifierLookup[k] = v`.
    pub fn set_reverse(&self, k: impl Into<String>, v: impl Into<String>) {
        self.types.reverse_lookup.borrow_mut().insert(k.into(), v.into());
    }
    /// `in(reverseIdentifierLookup, k)`.
    pub fn has_reverse(&self, k: &str) -> bool {
        self.types.reverse_lookup.borrow().contains_key(k)
    }

    /// `generateSrcInfo()`: unique SrcInfo for internally generated AST nodes.
    pub fn generate_src_info(&mut self) -> SrcInfo {
        let n = self.generated_src_info_count;
        self.generated_src_info_count += 1;
        SrcInfo::new(FILE_GENERATED, n, n, 0)
    }

    /// `getContent(info)`: file contents at the given location.
    pub fn content(&self, info: &SrcInfo) -> String {
        let Some(i) = self.imports.get(info.file.as_str()) else { return String::new() };
        let line = info.line - 1;
        if line < 0 || line as usize >= i.content.len() {
            return String::new();
        }
        let s = &i.content[line as usize];
        let col = info.col - 1;
        if col < 0 || col as usize >= s.len() {
            return String::new();
        }
        let end = (col as usize + info.len.max(0) as usize).min(s.len());
        s.get(col as usize..end).unwrap_or("").to_string()
    }

    /// `getClass(type)`.
    pub fn get_class_by_type(&self, t: TypeId) -> Option<&Class> {
        let name = self.types.class_name(t)?;
        self.classes.get(name.as_str())
    }
    pub fn get_class_by_type_mut(&mut self, t: TypeId) -> Option<&mut Class> {
        let name = self.types.class_name(t)?;
        self.classes.get_mut(name.as_str())
    }

    /// `getMethod(type, member)`.
    pub fn get_method(&self, t: TypeId, member: &str) -> String {
        if let Some(cls) = self.get_class_by_type(t) {
            if let Some(m) = cls.methods.get(member) {
                return m.clone();
            }
        }
        typhon_common::seqassert!(false, "cannot find '{}' in '{}'", member, self.types.debug_string(t, 2));
        String::new()
    }

    /// `findClass(name)`: _uninstantiated_ class type of a canonical class name.
    pub fn find_class(&self, name: &str) -> Option<TypeId> {
        let f = TypeContext::find_in(self, &self.type_ctx, name, 0, None)?;
        let t = f.borrow().ty?;
        if !self.types.is(t, &stdlib().type_) {
            return None;
        }
        let c = self.types.get_class(t)?;
        let g = self.types.generic_type(c, 0)?;
        self.types.get_class(g)
    }

    /// `findFunction(name)`.
    pub fn find_function(&self, name: &str) -> Option<TypeId> {
        for n in [name.to_string(), format!("{name}:0")] {
            if let Some(f) = TypeContext::find_in(self, &self.type_ctx, &n, 0, None) {
                if let Some(t) = f.borrow().ty {
                    if let Some(ft) = self.types.get_func(t) {
                        return Some(ft);
                    }
                }
            }
        }
        None
    }

    /// `findMethod(typ, member, args)`.
    pub fn find_method(&mut self, typ: TypeId, member: &str, args: &[TypeId]) -> Result<Option<TypeId>> {
        let ctx = self.type_ctx.clone();
        let mut tv = TypecheckVisitor::new(self, ctx, None, None);
        tv.find_best_method_types(typ, member, args)
    }

    /// `realizeType(type, generics)`.
    pub fn realize_type(&mut self, t: TypeId, generics: &[TypeId]) -> Result<Option<typhon_ir::TypeId>> {
        let ctx = self.type_ctx.clone();
        let mut tv = TypecheckVisitor::new(self, ctx, None, None);
        let inst = tv.instantiate_type_with(t, generics)?;
        if let Some(rtv) = tv.realize(inst)? {
            let types = tv.cache.types.clone();
            let c = types.get_class(rtv).unwrap();
            let name = types.class_name(c).unwrap();
            let rn = types.realized_name(c);
            return Ok(tv.cache.classes.get(name.as_str()).and_then(|c| c.realizations.get(&rn)).and_then(|r| r.borrow().ir));
        }
        Ok(None)
    }

    /// `realizeFunction(type, args, generics, parentClass)`; `args[0]` is the return type.
    pub fn realize_function(
        &mut self,
        ft: TypeId,
        args: &[TypeId],
        generics: &[TypeId],
        parent_class: Option<TypeId>,
    ) -> Result<Option<FuncId>> {
        let ctx = self.type_ctx.clone();
        let types = self.types.clone();
        let mut tv = TypecheckVisitor::new(self, ctx, None, None);
        let t = tv.instantiate_type(ft, parent_class)?;
        if args.len() != types.func_size(t) + 1 {
            return Ok(None);
        }
        let mut undo = Unification::new();
        if types.unify(types.func_ret(t), args[0], Some(&mut undo), &mut tv)? < 0 {
            undo.undo(&types);
            return Ok(None);
        }
        for gi in 1..args.len() {
            let mut undo = Unification::new();
            if types.unify(types.func_arg(t, gi - 1), args[gi], Some(&mut undo), &mut tv)? < 0 {
                undo.undo(&types);
                return Ok(None);
            }
        }
        if !generics.is_empty() {
            let fg = types.func_generics(t);
            if generics.len() != fg.len() {
                return Ok(None);
            }
            for (gi, g) in generics.iter().enumerate() {
                let mut undo = Unification::new();
                if types.unify(fg[gi].ty(), *g, Some(&mut undo), &mut tv)? < 0 {
                    undo.undo(&types);
                    return Ok(None);
                }
            }
        }
        let mut f = None;
        if let Some(rtv) = tv.realize(t)? {
            let fname = types.func_name(rtv);
            let rname = types.realized_name(rtv);
            let realization = tv.cache.functions.get(&fname).and_then(|f| f.realizations.get(&rname).cloned());
            let check = tv.get_unmangled_name(&fname) == "__init__";
            if let Some(r) = &realization {
                r.borrow_mut().check_initialization |= check;
            }
            if !tv.cache.pending_realizations.is_empty() || check {
                tv.prepare_vtables()?;
            }
            let pr = tv.cache.pending_realizations.clone(); // copy it as it might be modified
            for (key, _) in pr {
                tv.cache.translate_function(&key)?;
            }
            f = tv.cache.functions.get(&fname).and_then(|f| f.realizations.get(&rname)).and_then(|r| r.borrow().ir);
        }
        Ok(f)
    }

    /// `TranslateVisitor(codegenCtx).translateStmts(clone(functions[key].ast))`.
    pub fn translate_function(&mut self, key: &str) -> Result<()> {
        let Some(ast) = self.functions.get(key).and_then(|f| f.ast) else { return Ok(()) };
        let Some(h) = self.hooks.translate.clone() else { return Ok(()) };
        // Translation only reads the function statement (it emits the pending
        // realizations by name), so the reference's defensive clone is skipped.
        h(self, ast)
    }

    /// `makeTuple(types)`.
    pub fn make_tuple(&mut self, types: &[TypeId]) -> Result<Option<typhon_ir::TypeId>> {
        let ctx = self.type_ctx.clone();
        let mut tv = TypecheckVisitor::new(self, ctx, None, None);
        let tup = tv.generate_tuple(types.len(), true)?;
        let t = tv.instantiate_type_with(tup, types)?;
        let c = tv.cache.types.get_class(t).unwrap();
        self.realize_type(c, types)
    }

    /// `makeFunction([ret, args...])`.
    pub fn make_function(&mut self, types: &[TypeId]) -> Result<Option<typhon_ir::TypeId>> {
        typhon_common::seqassert!(!types.is_empty(), "types must have at least one argument");
        let ctx = self.type_ctx.clone();
        let mut tv = TypecheckVisitor::new(self, ctx, None, None);
        let tup = tv.generate_tuple(types.len() - 1, true)?;
        let arg_type = tv.instantiate_type_with(tup, &types[1..])?;
        let f = tv.get_stdlib_type("Function")?;
        let f = tv.cache.types.get_class(f).unwrap();
        self.realize_type(f, &[arg_type, types[0]])
    }

    /// `makeUnion(types)`.
    pub fn make_union(&mut self, types: &[TypeId]) -> Result<Option<typhon_ir::TypeId>> {
        let ctx = self.type_ctx.clone();
        let mut tv = TypecheckVisitor::new(self, ctx, None, None);
        let tup = tv.generate_tuple(types.len(), true)?;
        let arg_type = tv.instantiate_type_with(tup, types)?;
        let u = tv.get_stdlib_type(&stdlib().union)?;
        let u = tv.cache.types.get_class(u).unwrap();
        self.realize_type(u, &[arg_type])
    }

    /// `getRealizationId(type)`.
    pub fn realization_id(&mut self, t: TypeId) -> usize {
        let ctx = self.type_ctx.clone();
        let tv = TypecheckVisitor::new(self, ctx, None, None);
        tv.get_class_realization(t).map(|r| r.borrow().id).unwrap_or(0)
    }

    /// `getBaseRealizationIds(type)`.
    pub fn base_realization_ids(&mut self, t: TypeId) -> Vec<usize> {
        let ctx = self.type_ctx.clone();
        let tv = TypecheckVisitor::new(self, ctx, None, None);
        let bases = tv.get_class_realization(t).map(|r| r.borrow().bases.clone()).unwrap_or_default();
        bases.into_iter().map(|b| self.realization_id(b)).collect()
    }

    /// `getChildRealizationIds(type)`.
    pub fn child_realization_ids(&mut self, t: TypeId) -> Vec<usize> {
        let parent_id = self.realization_id(t);
        let mut all = vec![];
        for c in self.classes.values() {
            for r in c.realizations.values() {
                let r = r.borrow();
                all.push((r.id, r.bases.clone()));
            }
        }
        let mut child_ids = vec![];
        for (id, bases) in all {
            for b in bases {
                if self.realization_id(b) == parent_id {
                    child_ids.push(id);
                    break;
                }
            }
        }
        child_ids
    }

    /// `parseCode(code)`: parse, scope, typecheck and translate a code snippet into the
    /// current IR module (used by IR passes).
    pub fn parse_code(&mut self, code: &str) -> Result<Vec<typhon_ir::ValueId>> {
        let parse = self.hooks.parse_code.clone().ok_or_else(|| CompilerError::custom("no parser hook installed", SrcInfo::default()))?;
        let node = parse(self, "<internal>", code, 0)?;
        let sctx = self.imports.get(MAIN_IMPORT).and_then(|m| m.ctx.clone()).expect("main import not loaded");
        if let Some(sc) = self.hooks.scoping.clone() {
            sc(self, node, Some(sctx.clone()))?;
        }
        let node = TypecheckVisitor::apply_ctx(self, sctx.clone(), node, "<internal>", true)?;
        {
            let mut tv = TypecheckVisitor::new(self, sctx, None, None);
            tv.prepare_vtables()?;
        }
        let tr = self
            .hooks
            .translate_parsed
            .clone()
            .ok_or_else(|| CompilerError::custom("no translate hook installed", SrcInfo::default()))?;
        Ok(vec![tr(self, node)?])
    }

    /// `mergeC3(seqs)`: C3 linearization (Python MRO).
    pub fn merge_c3(types: &TypeArena, seqs: &mut [Vec<TypeId>]) -> Vec<TypeId> {
        let mut result = vec![];
        loop {
            let mut found = false;
            let mut cand: Option<TypeId> = None;
            for seq in seqs.iter() {
                if seq.is_empty() {
                    continue;
                }
                found = true;
                let mut nothead = false;
                for s in seqs.iter() {
                    if !s.is_empty() {
                        let mut in_ = false;
                        for j in 1..s.len() {
                            let n = types.class_name(s[j]).unwrap();
                            in_ |= types.is(seq[0], &n);
                            if in_ {
                                break;
                            }
                        }
                        if in_ {
                            nothead = true;
                            break;
                        }
                    }
                }
                if !nothead {
                    cand = types.get_class(seq[0]);
                    break;
                }
            }
            if !found {
                return result;
            }
            let Some(cand) = cand else { return vec![] };
            result.push(cand);
            for s in seqs.iter_mut() {
                if !s.is_empty() {
                    let n = types.class_name(s[0]).unwrap();
                    if types.is(cand, &n) {
                        s.remove(0);
                    }
                }
            }
        }
    }

    /// Convenience: `ast->hasAttribute(a)` for a statement.
    pub fn stmt_has_attr(&self, s: StmtId, a: Attr) -> bool {
        self.ast[s].attrs.has(a)
    }

    /// Run `f` with `self.module` temporarily pointing at `m` (used when the IR module
    /// calls back into the cache through [`TypeRealizer`] while it is itself borrowed).
    /// Note: the reference simply shares raw pointers; here the borrowed module is moved into a
    /// temporary `Rc<RefCell<..>>` for the duration of the call and moved back afterwards.
    pub fn with_module<R>(&mut self, m: &mut IrModule, f: impl FnOnce(&mut Cache) -> R) -> R {
        let placeholder = IrModule::new("");
        let holder = Rc::new(RefCell::new(std::mem::replace(m, placeholder)));
        let old = self.module.replace(holder.clone());
        let r = f(self);
        self.module = old;
        let inner = Rc::try_unwrap(holder).ok().map(|c| c.into_inner()).expect("module still borrowed");
        *m = inner;
        r
    }
}

/// `AstTypeRef` payload: a parser type handle (types never die, so the id is enough).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AstType(pub TypeId);

impl AstType {
    pub fn to_ref(t: TypeId) -> AstTypeRef {
        AstTypeRef::new(AstType(t))
    }
    pub fn from_ref(r: &AstTypeRef) -> Option<TypeId> {
        r.downcast_ref::<AstType>().map(|a| a.0)
    }
}

/// [`TypeRealizer`] implementation over a shared cache.
pub struct CacheRealizer(pub Rc<RefCell<Cache>>);

impl CacheRealizer {
    fn ast_types(m: &IrModule, args: &[typhon_ir::TypeId]) -> Option<Vec<TypeId>> {
        args.iter().map(|t| m.ty(*t).ast_type.as_ref().and_then(AstType::from_ref)).collect()
    }
    fn translate_generics(cache: &Cache, m: &IrModule, generics: &[typhon_ir::Generic]) -> Option<Vec<TypeId>> {
        let types = &cache.types;
        generics
            .iter()
            .map(|g| match g {
                typhon_ir::Generic::StaticStr(s) => Some(types.new_link(types.new_str_static(s.clone()))),
                typhon_ir::Generic::Static(i) => Some(types.new_link(types.new_int_static(*i))),
                typhon_ir::Generic::Type(t) => m.ty(*t).ast_type.as_ref().and_then(AstType::from_ref).map(|t| types.new_link(t)),
            })
            .collect()
    }
}

impl TypeRealizer for CacheRealizer {
    fn find_class(&mut self, name: &str) -> Option<AstTypeRef> {
        self.0.borrow().find_class(name).map(AstType::to_ref)
    }
    fn find_function(&mut self, name: &str) -> Option<AstTypeRef> {
        self.0.borrow().find_function(name).map(AstType::to_ref)
    }
    fn find_method(&mut self, m: &IrModule, cls: &AstTypeRef, method: &str, args: &[typhon_ir::TypeId]) -> Option<AstTypeRef> {
        let cls = AstType::from_ref(cls)?;
        let args = Self::ast_types(m, args)?;
        self.0.borrow_mut().find_method(cls, method, &args).ok().flatten().map(AstType::to_ref)
    }
    fn realize_type(
        &mut self,
        m: &mut IrModule,
        cls: &AstTypeRef,
        generics: &[typhon_ir::Generic],
    ) -> Result<Option<typhon_ir::TypeId>> {
        let Some(cls) = AstType::from_ref(cls) else { return Ok(None) };
        let mut c = self.0.borrow_mut();
        let Some(g) = Self::translate_generics(&c, m, generics) else { return Ok(None) };
        c.with_module(m, |c| c.realize_type(cls, &g))
    }
    fn realize_function(
        &mut self,
        m: &mut IrModule,
        func: &AstTypeRef,
        args: &[typhon_ir::TypeId],
        generics: &[typhon_ir::Generic],
        parent_cls: Option<&AstTypeRef>,
    ) -> Result<Option<FuncId>> {
        let Some(func) = AstType::from_ref(func) else { return Ok(None) };
        let Some(arg_types) = Self::ast_types(m, args) else { return Ok(None) };
        let mut c = self.0.borrow_mut();
        let Some(g) = Self::translate_generics(&c, m, generics) else { return Ok(None) };
        // translateArgs: the unbound return type first.
        let ret = c.types.new_unbound(0, 0, SrcInfo::default());
        let mut a = vec![ret];
        a.extend(arg_types);
        let parent = parent_cls.and_then(AstType::from_ref);
        c.with_module(m, |c| c.realize_function(func, &a, &g, parent))
    }
    fn make_function(&mut self, m: &mut IrModule, ret: typhon_ir::TypeId, args: &[typhon_ir::TypeId]) -> Result<typhon_ir::TypeId> {
        let mut all = vec![ret];
        all.extend_from_slice(args);
        let t = Self::ast_types(m, &all).ok_or_else(|| CompilerError::custom("type has no ast type", SrcInfo::default()))?;
        let mut c = self.0.borrow_mut();
        c.with_module(m, |c| c.make_function(&t))?
            .ok_or_else(|| CompilerError::custom("cannot realize function type", SrcInfo::default()))
    }
    fn make_tuple(&mut self, m: &mut IrModule, args: &[typhon_ir::TypeId]) -> Result<typhon_ir::TypeId> {
        let t = Self::ast_types(m, args).ok_or_else(|| CompilerError::custom("type has no ast type", SrcInfo::default()))?;
        let mut c = self.0.borrow_mut();
        c.with_module(m, |c| c.make_tuple(&t))?.ok_or_else(|| CompilerError::custom("cannot realize tuple", SrcInfo::default()))
    }
    fn make_union(&mut self, m: &mut IrModule, types: &[typhon_ir::TypeId]) -> Result<typhon_ir::TypeId> {
        let t = Self::ast_types(m, types).ok_or_else(|| CompilerError::custom("type has no ast type", SrcInfo::default()))?;
        let mut c = self.0.borrow_mut();
        c.with_module(m, |c| c.make_union(&t))?.ok_or_else(|| CompilerError::custom("cannot realize union", SrcInfo::default()))
    }
    fn parse_code(&mut self, m: &mut IrModule, code: &str) -> Result<()> {
        let mut c = self.0.borrow_mut();
        c.with_module(m, |c| c.parse_code(code)).map(|_| ())
    }
    fn generics(&mut self, m: &mut IrModule, ast: &AstTypeRef, func_generics: bool) -> Vec<typhon_ir::Generic> {
        let Some(t) = AstType::from_ref(ast) else { return vec![] };
        let mut c = self.0.borrow_mut();
        let types = c.types.clone();
        let gens: Vec<TypeId> = if func_generics {
            match types.get_func(t) {
                Some(f) => types.func_generics(f).iter().filter_map(|g| g.ty).collect(),
                None => return vec![],
            }
        } else {
            match types.get_class(t) {
                Some(cl) => types.generics(cl).iter().filter_map(|g| g.ty).collect(),
                None => return vec![],
            }
        };
        let mut ret = vec![];
        c.with_module(m, |c| {
            for g in gens {
                if let Some(v) = types.get_int_static(g) {
                    ret.push(typhon_ir::Generic::Static(v));
                } else if let Some(v) = types.get_bool_static(g) {
                    ret.push(typhon_ir::Generic::Static(v as i64));
                } else if let Some(v) = types.get_str_static(g) {
                    ret.push(typhon_ir::Generic::StaticStr(v));
                } else if let Some(ac) = types.get_class(g) {
                    let gs: Vec<TypeId> = types.generics(ac).iter().filter_map(|g| g.ty).collect();
                    if let Ok(Some(it)) = c.realize_type(ac, &gs) {
                        ret.push(typhon_ir::Generic::Type(it));
                    }
                } else {
                    typhon_common::seqassert!(false, "IR only supports int, bool or str statics");
                }
            }
        });
        ret
    }
}
