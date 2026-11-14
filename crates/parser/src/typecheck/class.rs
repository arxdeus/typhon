
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::rc::Rc;

use typhon_common::mangle::{self, stdlib};
use typhon_common::{ErrorKind, Result};

use super::{Base, TypeContext, TypecheckItem, TypecheckVisitor};
use crate::ast::{
    AssignStmt, AstArena, Attr, Child, ClassStmt, ExprKind, ExprStmt, IdExpr, IfStmt, IndexExpr, IntExpr,
    KeywordStarExpr, ListExpr, Param, ParamKind, ParamStatus, ReturnStmt, StarExpr, StmtKind, StringExpr, SuiteStmt,
    TryStmt, TupleExpr, UpdateMode, YieldFromStmt, is_id,
};
use crate::cache::{ClassField, ClassRealization, ClassRealizationRef, FN_SETTER_SUFFIX, MAX_TUPLE, STDLIB_IMPORT};
use crate::ids::{ExprId, StmtId, TypeId};
use crate::types::{Generic, LinkKind, LiteralKind};
use crate::visitors::visitor::{expr_children, stmt_children};

impl<'a> TypecheckVisitor<'a> {
    /// `checkFieldInitialization(fn, ast)`: reject fields not proven initialized at a
    /// realized `__init__`'s normal exits. Tuple classes & internal classes are exempt.
    pub fn check_field_initialization(&mut self, f: TypeId, ast: StmtId) -> Result<()> {
        let types = self.types();
        let nparams = self.cache.ast[ast].kind.as_function().map(|x| x.items.len()).unwrap_or(0);
        let Some(pt) = types.func_parent(f) else { return Ok(()) };
        if self.get_unmangled_name(&types.func_name(f)) != "__init__" || nparams == 0 {
            return Ok(());
        }
        let Some(parent) = types.get_class(pt) else { return Ok(()) };
        if types.with_class(parent, |c| c.is_tuple).unwrap_or(false) {
            return Ok(());
        }
        let Some(cls_ast) = self.cache.get_class_by_type(parent).and_then(|c| c.ast) else { return Ok(()) };
        if self.cache.ast[cls_ast].attrs.has(Attr::Internal) {
            return Ok(());
        }
        let receiver_type = types.get_class(self.extract_func_arg_type(f, 0));
        let receiver = self.get_class_realization(receiver_type.unwrap_or(parent));
        let initialized = {
            let mut v = FieldInitializationVisitor::new(self, receiver, Default::default(), Default::default());
            v.analyze(ast, 0)
        };
        if !initialized.reachable {
            return Ok(());
        }
        for field in self.get_class_fields(parent) {
            if !initialized.fields.contains(&field.name) {
                let src = self.cache.ast[ast].src;
                let pn = self.get_unmangled_name(&types.class_name(parent).unwrap());
                typhon_common::bail_e!(
                    ErrorKind::CUSTOM,
                    src,
                    "field '{}.{}' may be uninitialized when __init__ returns",
                    pn,
                    field.name
                );
            }
        }
        Ok(())
    }

    /// `isEnum(typ)`.
    pub fn is_enum(&self, typ: TypeId) -> bool {
        let e = mangle::class("std.enum", "Enum");
        let types = &self.cache.types;
        self.cache.get_class_by_type(typ).is_some_and(|c| c.mro.iter().any(|b| types.is(*b, &e)))
    }

    /// `getClassMethods(stmt)`: all statements within a class suite (recursively).
    /// Assumes that each statement is either a function, a class or a docstring.
    /// Note: returns `Result` (throws CLASS_BAD_ATTR); only class.rs calls it.
    pub fn get_class_methods(&self, s: StmtId) -> Result<Vec<StmtId>> {
        let a = &self.cache.ast;
        let mut v = vec![];
        match &a[s].kind {
            StmtKind::Suite(sp) => {
                for ss in sp.items.clone() {
                    v.extend(self.get_class_methods(ss)?);
                }
            }
            StmtKind::Function(_) | StmtKind::Class(_) => v.push(s),
            StmtKind::Expr(e) if e.expr.is_some_and(|x| matches!(a[x].kind, ExprKind::String(_))) => {}
            _ => typhon_common::bail_e!(ErrorKind::CLASS_BAD_ATTR, a[s].src),
        }
        Ok(v)
    }
    fn class_methods_opt(&self, s: Option<StmtId>) -> Result<Vec<StmtId>> {
        match s {
            Some(s) => self.get_class_methods(s),
            None => Ok(vec![]),
        }
    }

    /// `transformNestedClasses(stmt, clsStmts, varStmts, fnStmts)`: extract nested classes
    /// and transform them before the main class.
    pub fn transform_nested_classes(
        &mut self,
        s: StmtId,
        cls: &mut Vec<StmtId>,
        vars: &mut Vec<StmtId>,
        fns: &mut Vec<StmtId>,
    ) -> Result<()> {
        let (suite, parent_name) = {
            let c = self.cache.ast[s].kind.as_class().unwrap();
            (c.suite, c.name.clone())
        };
        for sp in self.class_methods_opt(suite)? {
            let Some(orig_name) = self.cache.ast[sp].kind.as_class().map(|c| c.name.clone()) else { continue };
            // If class B is nested within A, it's name is always A.B, never B itself.
            // Ensure that parent class name is appended
            self.cache.ast[sp].kind.as_class_mut().unwrap().name = format!("{parent_name}.{orig_name}");
            let tsp = self.tr_stmt(sp)?;
            if let Some(items) = self.cache.ast[tsp].kind.as_suite().map(|x| x.items.clone()) {
                let mut name = String::new();
                for i in items {
                    match &self.cache.ast[i].kind {
                        StmtKind::Class(c) => {
                            name = c.name.clone();
                            cls.push(i);
                        }
                        StmtKind::Assign(_) => vars.push(i),
                        _ => fns.push(i),
                    }
                }
                let v = self.force_find(&name);
                self.ctx.borrow_mut().add(&orig_name, v);
            }
        }
        Ok(())
    }

    /// `autoDeduceMembers(stmt, args)`: find the first `__init__` with self parameter and
    /// use it to deduce class members. Each deduced member will be treated as generic.
    pub fn auto_deduce_members(&mut self, s: StmtId, args: &mut Vec<Param>) -> Result<bool> {
        let st = stdlib();
        let suite = self.cache.ast[s].kind.as_class().unwrap().suite;
        // Python `__dict__` order: first assignment order (no sorting).
        let mut members: Vec<String> = Vec::new();
        for sp in self.class_methods_opt(suite)? {
            let a = &self.cache.ast[sp];
            if a.kind.as_function().is_some_and(|f| f.name == "__init__") {
                if let Some(b) = a.attrs.get_str_list(Attr::ClassDeduce) {
                    for m in b.iter() {
                        if !members.contains(m) {
                            members.push(m.clone());
                        }
                    }
                }
            }
        }
        if members.is_empty() {
            return Ok(false);
        }
        if let Some(aa) = self.cache.ast[s].attrs.get_str_list_mut(Attr::ClassMagic) {
            aa.retain(|m| m != "init");
        }
        for m in members {
            let generic_name = format!("T_{m}");
            let ty = self.id_n(&st.type_);
            let def = self.id_n(&st.none_type);
            args.push(self.cache.ast.new_param(Default::default(), generic_name.clone(), Some(ty), Some(def), 1, ParamKind::Normal));
            let gt = self.id_n(&generic_name);
            args.push(self.cache.ast.new_param(Default::default(), m, Some(gt), None, 0, ParamKind::Normal));
        }
        Ok(true)
    }

    /// `codegenMagic(op, typExpr, allArgs, isRecord)`: generate a magic method `__op__`.
    pub fn codegen_magic(&mut self, op: &str, typ: ExprId, all_args: &[Param], is_record: bool) -> Result<Option<StmtId>> {
        let st = stdlib();
        let mut ret: Option<ExprId> = None;
        let mut fargs: Vec<Param> = vec![];
        let mut stmts: Vec<StmtId> = vec![];
        let mut attrs = vec![Attr::AutoGenerated];
        let args: Vec<Param> = all_args.iter().map(|a| self.cache.ast.clone_param(a, false)).collect();

        macro_rules! i {
            ($s:expr) => {
                self.id_n(&$s)
            };
        }
        macro_rules! ns {
            ($x:expr) => {{
                let m = self.id_n("__magic__");
                self.dot_n(m, $x)
            }};
        }
        macro_rules! ty {
            () => {
                self.cache.ast.clone_expr(typ, false)
            };
        }
        // `Param(name, type, default)` (status inferred).
        macro_rules! p {
            ($n:expr, $t:expr) => {
                self.cache.ast.new_param(Default::default(), $n, $t, None, 0, ParamKind::Normal)
            };
            ($n:expr, $t:expr, $d:expr) => {
                self.cache.ast.new_param(Default::default(), $n, $t, $d, 0, ParamKind::Normal)
            };
        }
        macro_rules! ret_call {
            ($f:expr, [$($a:expr),*]) => {{
                let f = $f;
                let args = vec![$($a),*];
                let c = self.call_n(f, args);
                self.n_stmt(ReturnStmt { expr: Some(c) })
            }};
        }
        let ptr_u8 = |v: &mut Self| {
            let p = v.id_n(&st.ptr);
            let u = v.id_n("u8");
            v.n_expr(IndexExpr { expr: Some(p), index: Some(u) })
        };

        match op {
            "new" => {
                ret = Some(ty!());
                if is_record {
                    // Tuples: def __new__() -> T (internal)
                    for a in &args {
                        let t = self.cache.ast.clone_expr_opt(a.ty, false);
                        let d = self.cache.ast.clone_expr_opt(a.default_value, false);
                        fargs.push(p!(a.name.clone(), t, d));
                    }
                    attrs.push(Attr::Internal);
                } else {
                    // Classes: def __new__() -> T
                    let n = ns!(op);
                    let t = ty!();
                    stmts.push(ret_call!(n, [t]));
                }
            }
            "init" => {
                // Classes: def __init__(self: T, a1: T1, ..., aN: TN) -> None:
                //            self.aI = aI ...
                ret = Some(i!(st.none_type));
                let t = ty!();
                fargs.push(p!("self", Some(t)));
                let has_rtti = i!(mangle::func("std.internal.static", "has_rtti"));
                let t = ty!();
                let cond = self.call_n(has_rtti, vec![t]);
                let sup = i!("super");
                let sup = self.call_n(sup, vec![]);
                let sup_init = self.dot_n(sup, "__init__");
                let a1 = i!("_args");
                let a1 = self.n_expr(StarExpr { expr: Some(a1) });
                let a2 = i!("_kwargs");
                let a2 = self.n_expr(KeywordStarExpr { expr: Some(a2) });
                let call = self.call_n(sup_init, vec![a1, a2]);
                let es = self.n_stmt(ExprStmt { expr: Some(call) });
                stmts.push(self.n_stmt(IfStmt { cond: Some(cond), if_suite: Some(es), else_suite: None }));
                for a in &args {
                    let t = self.cache.ast.clone_expr_opt(a.ty, true);
                    let d = self.cache.ast.clone_expr_opt(a.default_value, false);
                    fargs.push(p!(a.name.clone(), t, d));
                    let s = i!("self");
                    let lhs = self.dot_n(s, &a.name);
                    let rhs = i!(a.name);
                    stmts.push(self.n_stmt(AssignStmt { lhs: Some(lhs), rhs: Some(rhs), type_expr: None, update: UpdateMode::Assign }));
                }
                fargs.push(p!("*_args", None));
                fargs.push(p!("**_kwargs", None));
                attrs.push(Attr::HasSelf);
            }
            "raw" | "dict" => {
                // Classes: def __raw__(self: T)
                let t = ty!();
                fargs.push(p!("self", Some(t)));
                let n = ns!(op);
                let s = i!("self");
                stmts.push(ret_call!(n, [s]));
            }
            "tuplesize" => {
                // def __tuplesize__() -> int
                ret = Some(i!("int"));
                let n = ns!(op);
                stmts.push(ret_call!(n, []));
            }
            "getitem" => {
                // Tuples: def __getitem__(self: T, index: int)
                let t = ty!();
                fargs.push(p!("self", Some(t)));
                let it = i!("int");
                fargs.push(p!("index", Some(it)));
                let n = ns!(op);
                let s = i!("self");
                let ix = i!("index");
                stmts.push(ret_call!(n, [s, ix]));
            }
            "iter" => {
                // Tuples: def __iter__(self: T)
                let t = ty!();
                fargs.push(p!("self", Some(t)));
                let n = ns!(op);
                let s = i!("self");
                let c = self.call_n(n, vec![s]);
                stmts.push(self.n_stmt(YieldFromStmt { expr: Some(c) }));
            }
            "contains" => {
                // Tuples: def __contains__(self: T, what) -> bool
                let t = ty!();
                fargs.push(p!("self", Some(t)));
                fargs.push(p!("what", None));
                ret = Some(i!(st.bool_));
                let n = ns!(op);
                let s = i!("self");
                let w = i!("what");
                stmts.push(ret_call!(n, [s, w]));
            }
            "eq" | "ne" | "lt" | "le" | "gt" | "ge" => {
                // def __op__(self: T, obj: T) -> bool
                let t = ty!();
                fargs.push(p!("self", Some(t)));
                let t = ty!();
                fargs.push(p!("obj", Some(t)));
                ret = Some(i!(st.bool_));
                let n = ns!(op);
                let s = i!("self");
                let o = i!("obj");
                stmts.push(ret_call!(n, [s, o]));
            }
            "hash" | "len" => {
                // def __hash__(self: T) -> int
                let t = ty!();
                fargs.push(p!("self", Some(t)));
                ret = Some(i!("int"));
                let n = ns!(op);
                let s = i!("self");
                stmts.push(ret_call!(n, [s]));
            }
            "pickle" => {
                // def __pickle__(self: T, dest: Ptr[u8])
                let t = ty!();
                fargs.push(p!("self", Some(t)));
                let pu = ptr_u8(self);
                fargs.push(p!("dest", Some(pu)));
                let n = ns!(op);
                let s = i!("self");
                let d = i!("dest");
                stmts.push(ret_call!(n, [s, d]));
            }
            "unpickle" => {
                // def __unpickle__(src: Ptr[u8]) -> T
                let pu = ptr_u8(self);
                fargs.push(p!("src", Some(pu)));
                ret = Some(ty!());
                let n = ns!(op);
                let s = i!("src");
                let t = ty!();
                stmts.push(ret_call!(n, [s, t]));
            }
            "to_gpu" => {
                // def __to_gpu__(self: T, cache) -> T
                let t = ty!();
                fargs.push(p!("self", Some(t)));
                fargs.push(p!("cache", None));
                ret = Some(ty!());
                let n = ns!(op);
                let s = i!("self");
                let c = i!("cache");
                stmts.push(ret_call!(n, [s, c]));
            }
            "from_gpu" => {
                // def __from_gpu__(self: T, other: T)
                let t = ty!();
                fargs.push(p!("self", Some(t)));
                let t = ty!();
                fargs.push(p!("other", Some(t)));
                let n = ns!(op);
                let s = i!("self");
                let o = i!("other");
                stmts.push(ret_call!(n, [s, o]));
            }
            "from_gpu_new" => {
                // def __from_gpu_new__(other: T) -> T
                let t = ty!();
                fargs.push(p!("other", Some(t)));
                ret = Some(ty!());
                let n = ns!(op);
                let o = i!("other");
                stmts.push(ret_call!(n, [o]));
            }
            "repr" | "repr_default" | "repr_dataclass" => {
                // def __repr__(self: T) -> str
                let t = ty!();
                fargs.push(p!("self", Some(t)));
                ret = Some(i!(st.string));
                let n = ns!(op);
                let s = i!("self");
                stmts.push(ret_call!(n, [s]));
            }
            "add" => {
                // def __add__(self, obj)
                let t = ty!();
                fargs.push(p!("self", Some(t)));
                fargs.push(p!("obj", None));
                let n = ns!(op);
                let s = i!("self");
                let o = i!("obj");
                stmts.push(ret_call!(n, [s, o]));
            }
            "mul" => {
                // def __mul__(self, i: Literal[int])
                let t = ty!();
                fargs.push(p!("self", Some(t)));
                let l = i!("Literal");
                let it = i!("int");
                let lt = self.n_expr(IndexExpr { expr: Some(l), index: Some(it) });
                fargs.push(p!("i", Some(lt)));
                let n = ns!(op);
                let s = i!("self");
                let iv = i!("i");
                stmts.push(ret_call!(n, [s, iv]));
            }
            _ => {
                typhon_common::seqassert!(false, "invalid magic {}", op);
                return Err(typhon_common::CompilerError::custom(format!("invalid magic {op}"), self.src));
            }
        }
        // NC<>: no source location
        let suite = self.cache.ast.new_stmt(SuiteStmt { items: stmts }, Default::default());
        let t = self.cache.ast.new_function(Default::default(), format!("__{}__", if op == "repr_dataclass" { "repr" } else { op }), ret, fargs, Some(suite), vec![], false)?;
        for a in attrs {
            self.cache.ast[t].attrs.set(a);
        }
        let src = self.cache.generate_src_info();
        self.cache.ast[t].src = src;
        Ok(Some(t))
    }

    /// `generateKwId(names)`: id of a keyword-name combination (0 for none).
    pub fn generate_kw_id(&mut self, names: &[String]) -> i32 {
        if names.is_empty() {
            return 0;
        }
        let key = names.join(";");
        if let Some(v) = self.cache.generated_kw_tuples.get(&key) {
            return *v;
        }
        self.cache.types.generated_tuple_names.borrow_mut().push(names.to_vec());
        let id = self.cache.generated_kw_tuples.len() as i32 + 1;
        self.cache.generated_kw_tuples.insert(key, id);
        id
    }

    /// `generateTuple(n, generateNew)`: the generic `Tuple` class with `n` generics.
    pub fn generate_tuple(&mut self, n: usize, generate_new: bool) -> Result<TypeId> {
        if n > MAX_TUPLE {
            typhon_common::bail_e!(ErrorKind::CUSTOM, self.src, "tuple too large ({})", n);
        }
        let st = stdlib();
        let types = self.types();
        let key = format!("{}.{}", st.tuple, n);
        let rctx = self.get_import(STDLIB_IMPORT).ctx.clone().expect("stdlib not loaded");
        let val = TypeContext::find_in(self.cache, &rctx, &key, 0, None);
        let val = match val {
            Some(v) => v,
            None => {
                let fields = self.cache.classes.get(st.tuple.as_str()).map(|c| c.fields.clone()).unwrap_or_default();
                typhon_common::seqassert!(n <= fields.len(), "tuple too large");
                let mut gens = vec![];
                for f in fields.iter().take(n) {
                    let gt = types.link(f.ty.unwrap()).map(|l| l.id).unwrap_or(0);
                    let name = f
                        .type_expr
                        .and_then(|e| self.cache.ast[e].kind.as_id().map(|i| i.value.clone()))
                        .unwrap_or_default();
                    gens.push(Generic::new(name, f.ty, gt, LiteralKind::Runtime));
                }
                let mut c = crate::types::ClassType::new(&st.tuple, gens, vec![]);
                c.is_tuple = true;
                let t = types.new_class_from(c, Default::default());
                TypeContext::add_type(self.cache, &rctx, &key, &key, t, Default::default())
            }
        };
        let t = types.get_class(val.borrow().ty()).unwrap();
        if generate_new && !self.cache.generated_tuples.contains(&n) {
            self.cache.generated_tuples.insert(n);
            let mut params = vec![];
            let mut type_args = vec![];
            for i in 0..n {
                let ti = self.n_expr(IdExpr::new(format!("T{}", i + 1)));
                params.push(Param::with_status(format!("item{}", i + 1), Some(ti), None, ParamStatus::Value));
                type_args.push(self.n_expr(IdExpr::new(format!("T{}", i + 1))));
            }
            for i in 0..n {
                let ty = self.n_expr(IdExpr::new(st.type_.clone()));
                params.push(Param::with_status(format!("T{}", i + 1), Some(ty), None, ParamStatus::Generic));
            }
            let tup = self.n_expr(IdExpr::new(st.tuple.clone()));
            let targs = self.n_expr(TupleExpr { items: type_args });
            let ret = self.n_expr(IndexExpr { expr: Some(tup), index: Some(targs) });
            let src = self.src;
            let f = self.cache.ast.new_function(src, "__new__".to_string(), Some(ret), params, None, vec![], false)?;
            self.cache.ast[f].attrs.set(Attr::Internal);
            let body = self.cache.ast.wrap_suite(Some(f));
            let ext = self.n_stmt(ClassStmt { name: st.tuple.clone(), items: vec![], suite: body, decorators: vec![], base_classes: vec![] });
            self.cache.ast[ext].attrs.set(Attr::Extend);
            self.cache.ast[ext].attrs.set(Attr::AutoGenerated);
            let ext = self.n_stmt(SuiteStmt { items: vec![ext] });
            if let Some(sc) = self.cache.hooks.scoping.clone() {
                sc(self.cache, ext, None)?;
            }
            let (old_bases, old_bl) = {
                let mut r = rctx.borrow_mut();
                r.prepare_checkpoint();
                let ob = std::mem::take(&mut r.bases);
                r.bases.push(ob[0].clone());
                let obl = std::mem::replace(&mut r.block_level, 0);
                (ob, obl)
            };
            let res = TypecheckVisitor::apply_ctx(self.cache, rctx.clone(), ext, "<internal>", false);
            {
                let mut r = rctx.borrow_mut();
                r.block_level = old_bl;
                r.bases = old_bases;
            }
            let ext = res?;
            let pre = self.preamble;
            self.cache.ast.suite_add_stmt(pre, Some(ext));
        }
        Ok(t)
    }
}

// ---------------------------------------------------------------------------------------
// Field initialization analysis (anonymous namespace )
// ---------------------------------------------------------------------------------------

/// Initialization facts for the tracked receiver on a modeled control-flow path.
#[derive(Clone, Debug)]
struct InitializedFields {
    /// Fields assigned on every reachable path merged into this state.
    fields: BTreeSet<String>,
    /// Local names known to refer to the receiver, including supported wrappers.
    aliases: BTreeSet<String>,
    /// Whether execution can continue along the current path.
    reachable: bool,
}

impl Default for InitializedFields {
    fn default() -> Self {
        Self { fields: BTreeSet::new(), aliases: BTreeSet::new(), reachable: true }
    }
}

impl InitializedFields {
    fn unreachable() -> Self {
        Self { fields: BTreeSet::new(), aliases: BTreeSet::new(), reachable: false }
    }
    /// Keep only facts shared by continuing paths; terminated paths impose no facts.
    fn intersect(&mut self, other: &InitializedFields) {
        if !other.reachable {
            return;
        }
        if !self.reachable {
            *self = other.clone();
            return;
        }
        self.fields.retain(|f| other.fields.contains(f));
        self.aliases.retain(|a| other.aliases.contains(a));
    }
}

/// `ReboundLocalsVisitor`: names (re)bound by assignments, loops and handlers,
/// not descending into nested functions/classes.
pub(crate) fn rebound_locals(a: &AstArena, s: StmtId, names: &mut BTreeSet<String>) {
    match &a[s].kind {
        StmtKind::Function(_) | StmtKind::Class(_) => return,
        StmtKind::Assign(x) => {
            if let Some(i) = x.lhs.and_then(|l| a[l].kind.as_id()) {
                names.insert(i.value.clone());
            }
        }
        StmtKind::For(x) => {
            if let Some(i) = x.var.and_then(|l| a[l].kind.as_id()) {
                names.insert(i.value.clone());
            }
        }
        StmtKind::Except(x) => {
            names.insert(x.var.clone());
        }
        _ => {}
    }
    for c in stmt_children(a, s) {
        match c {
            Child::Stmt(x) | Child::Suite(x) => rebound_locals(a, x, names),
            Child::Expr(e) => rebound_locals_expr(a, e, names),
        }
    }
}
/// Names used through the explicit Optional API (`x.__val__()`, `x.__has__()`) anywhere in
/// `s`. Such code already unwraps `x` itself, so None-narrowing must leave `x` Optional.
pub(crate) fn optional_api_names(a: &AstArena, s: StmtId, names: &mut BTreeSet<String>) {
    fn ex(a: &AstArena, e: ExprId, names: &mut BTreeSet<String>) {
        if let Some(d) = a[e].kind.as_dot() {
            if d.member == "__val__" || d.member == "__has__" {
                if let Some(i) = d.expr.and_then(|x| a[x].kind.as_id()) {
                    names.insert(i.value.clone());
                }
            }
        }
        for c in expr_children(a, e) {
            match c {
                Child::Stmt(x) | Child::Suite(x) => optional_api_names(a, x, names),
                Child::Expr(e) => ex(a, e, names),
            }
        }
    }
    for c in stmt_children(a, s) {
        match c {
            Child::Stmt(x) | Child::Suite(x) => optional_api_names(a, x, names),
            Child::Expr(e) => ex(a, e, names),
        }
    }
}

fn rebound_locals_expr(a: &AstArena, e: ExprId, names: &mut BTreeSet<String>) {
    for c in expr_children(a, e) {
        match c {
            Child::Stmt(x) | Child::Suite(x) => rebound_locals(a, x, names),
            Child::Expr(e) => rebound_locals_expr(a, e, names),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ExitKind {
    Return,
    Break,
    Continue,
}

struct Exit {
    kind: ExitKind,
    state: InitializedFields,
}

type SummaryKey = (StmtId, usize);

/// Conservatively track constructor field writes in a realized, typed AST.
/// Follow known helper bodies so delegated initialization is not spuriously rejected;
/// this analysis emits diagnostics only and does not insert runtime field guards.
struct FieldInitializationVisitor<'v, 'a> {
    tv: &'v TypecheckVisitor<'a>,
    /// Concrete receiver realization used to resolve cooperative super() dispatch.
    receiver: Option<ClassRealizationRef>,
    active: Rc<RefCell<BTreeSet<SummaryKey>>>,
    summaries: Rc<RefCell<BTreeMap<SummaryKey, InitializedFields>>>,
    /// Facts at the current point in the body being analyzed.
    state: InitializedFields,
    /// Pending abrupt exits, intercepted by enclosing loops and finally blocks.
    exits: Vec<Exit>,
    /// Facts shared by potential exceptions in each enclosing protected region.
    exceptions: Vec<InitializedFields>,
}

impl<'v, 'a> FieldInitializationVisitor<'v, 'a> {
    fn new(
        tv: &'v TypecheckVisitor<'a>,
        receiver: Option<ClassRealizationRef>,
        active: Rc<RefCell<BTreeSet<SummaryKey>>>,
        summaries: Rc<RefCell<BTreeMap<SummaryKey, InitializedFields>>>,
    ) -> Self {
        Self { tv, receiver, active, summaries, state: Default::default(), exits: vec![], exceptions: vec![] }
    }
    fn ast(&self) -> &AstArena {
        &self.tv.cache.ast
    }

    fn record_exception(&mut self) {
        if self.state.reachable && !self.exceptions.is_empty() {
            let st = self.state.clone();
            self.exceptions.last_mut().unwrap().intersect(&st);
        }
    }

    fn forget_rebound_aliases(&mut self, s: StmtId) {
        let mut names = BTreeSet::new();
        rebound_locals(self.ast(), s, &mut names);
        for n in names {
            self.state.aliases.remove(&n);
        }
    }

    fn analyze_loop_body(&mut self, body: Option<StmtId>) {
        let incoming = self.state.clone();
        let previous = self.exits.len();
        self.stmt(body);
        let pending: Vec<Exit> = self.exits.drain(previous..).collect();
        self.state = incoming;
        for exit in pending {
            if exit.kind == ExitKind::Break {
                self.state.intersect(&exit.state);
            } else if exit.kind == ExitKind::Return {
                self.exits.push(exit);
            }
        }
    }

    /// Recognize receiver aliases through known casts and lowered super() wrappers.
    fn is_instance(&self, e: Option<ExprId>) -> bool {
        let Some(e) = e else { return false };
        let a = self.ast();
        let types = &self.tv.cache.types;
        match &a[e].kind {
            ExprKind::Id(i) => return self.state.aliases.contains(&i.value),
            ExprKind::Dot(member) => {
                let inner = member.expr.unwrap();
                if member.member == "_obj" && a[inner].ty.is_some_and(|t| types.is(t, "Super")) {
                    return self.is_instance(Some(inner));
                }
                if let Some(arguments) = a[inner].kind.as_dot() {
                    let ai = arguments.expr.unwrap();
                    if member.member == "item1" && arguments.member == "args" && a[ai].ty.is_some_and(|t| types.is(t, "Partial")) {
                        return self.is_instance(Some(ai));
                    }
                }
            }
            ExprKind::Call(call) if !call.items.is_empty() => {
                let Some(f) = call.expr.and_then(|x| a[x].ty).and_then(|t| types.get_func(t)) else { return false };
                let name = types.func_name(f);
                if name.starts_with(&mangle::func("std.internal.types.rtti", "force_cast"))
                    || name.starts_with(&mangle::mangled_method("", "type", "_force_cast", 0, 0))
                    || name.starts_with(&mangle::mangled_method("", "Super", "__getattr__", 0, 0))
                {
                    return self.is_instance(call.items[0].value);
                }
                if name.starts_with(&mangle::mangled_method("", "Super", "__new__", 0, 0)) {
                    for arg in &call.items {
                        if self.is_instance(arg.value) {
                            return true;
                        }
                    }
                }
            }
            _ => {}
        }
        false
    }

    /// Summarize normal exits with the receiver bound to an AST parameter index.
    /// Recursive re-entry contributes no new facts rather than assuming a write.
    /// Publish a summary only after merging all exits, never from a partial traversal.
    fn analyze(&mut self, function: StmtId, argument: usize) -> InitializedFields {
        let f = self.ast()[function].kind.as_function().unwrap().clone();
        if f.async_ || self.ast()[function].attrs.has(Attr::IsGenerator) {
            return InitializedFields::default();
        }
        let key = (function, argument);
        if let Some(s) = self.summaries.borrow().get(&key) {
            return s.clone();
        }
        if !self.active.borrow_mut().insert(key) {
            return self.state.clone();
        }
        self.state.aliases.insert(f.items[argument].name.clone());
        self.stmt(f.suite);
        let exits: Vec<InitializedFields> =
            self.exits.iter().filter(|e| e.kind == ExitKind::Return).map(|e| e.state.clone()).collect();
        for e in exits {
            self.state.intersect(&e);
        }
        self.active.borrow_mut().remove(&key);
        self.summaries.borrow_mut().insert(key, self.state.clone());
        self.state.clone()
    }

    fn child(&mut self, c: Child) {
        match c {
            Child::Expr(e) => self.expr(Some(e)),
            Child::Stmt(s) | Child::Suite(s) => self.stmt(Some(s)),
        }
    }

    /// Ignore expression effects on paths that have already terminated.
    fn expr(&mut self, e: Option<ExprId>) {
        let Some(e) = e else { return };
        if !self.state.reachable {
            return;
        }
        match self.ast()[e].kind.clone() {
            ExprKind::Call(_) => self.visit_call(e),
            ExprKind::If(x) => {
                // Conditional expressions may invoke different initialization helpers.
                self.expr(x.cond);
                let incoming = self.state.clone();
                self.expr(x.ifexpr);
                let first = self.state.clone();
                self.state = incoming;
                self.expr(x.elsexpr);
                self.state.intersect(&first);
            }
            _ => {
                for c in expr_children(self.ast(), e) {
                    self.child(c);
                }
            }
        }
    }

    /// Incorporate guaranteed writes from known helpers receiving the receiver.
    /// Typed calls omit generic arguments, so match only value parameters in the AST.
    fn visit_call(&mut self, e: ExprId) {
        for c in expr_children(self.ast(), e) {
            self.child(c);
        }
        if !self.state.reachable {
            return;
        }
        self.record_exception();
        let types = self.tv.cache.types.clone();
        let call = self.ast()[e].kind.as_call().unwrap().clone();
        let Some(mut function) = call.expr.and_then(|x| self.ast()[x].ty).and_then(|t| types.get_func(t)) else { return };
        if let Some(dispatch) = self.ast()[e].attrs.get_int(Attr::ExprVirtual) {
            let mut thunk = 0usize;
            if dispatch >= 0 {
                thunk = dispatch as usize;
            } else if self.tv.is_polymorphic(function) {
                match self.tv.cache.thunk_ids.get(&self.tv.vtable_signature(function)) {
                    Some(k) => thunk = *k,
                    None => return,
                }
            }
            if thunk != 0 {
                if call.items.is_empty() || !self.is_instance(call.items[0].value) {
                    return;
                }
                let target = self.receiver.as_ref().and_then(|r| r.borrow().vtable.get(&thunk).copied());
                let Some(target) = target else { return };
                function = target;
            }
        }
        let Some(cached) = self.tv.cache.functions.get(&types.func_name(function)) else { return };
        let Some(realization) = cached.realizations.get(&types.realized_name(function)) else { return };
        let Some(body) = realization.borrow().ast else { return };
        let params = self.ast()[body].kind.as_function().unwrap().items.clone();
        let mut argument = 0;
        for (parameter, p) in params.iter().enumerate() {
            if !p.is_value() {
                continue;
            }
            if argument >= call.items.len() {
                break;
            }
            let v = call.items[argument].value;
            argument += 1;
            if !self.is_instance(v) {
                continue;
            }
            let mut helper = FieldInitializationVisitor::new(self.tv, self.receiver.clone(), self.active.clone(), self.summaries.clone());
            let initialized = helper.analyze(body, parameter);
            self.state.fields.extend(initialized.fields);
            self.state.reachable &= initialized.reachable;
        }
    }

    /// Stop traversing statements after an abrupt control-flow transfer.
    fn stmt(&mut self, s: Option<StmtId>) {
        let Some(s) = s else { return };
        if !self.state.reachable {
            return;
        }
        match self.ast()[s].kind.clone() {
            // Declaring a nested function / class does not execute its initialization helpers.
            StmtKind::Function(_) | StmtKind::Class(_) => {}
            StmtKind::Assign(x) => {
                // Track local receiver aliases, discarding them when reassigned to other values.
                self.expr(x.rhs);
                if let (Some(t), Some(_)) = (x.lhs.and_then(|l| self.ast()[l].kind.as_id().map(|i| i.value.clone())), x.rhs) {
                    if self.is_instance(x.rhs) {
                        self.state.aliases.insert(t);
                    } else {
                        self.state.aliases.remove(&t);
                    }
                }
            }
            StmtKind::AssignMember(x) => {
                // Credit writes to the receiver only after evaluating their operands.
                self.expr(x.lhs);
                self.expr(x.rhs);
                if self.is_instance(x.lhs) {
                    self.state.fields.insert(x.member.clone());
                }
            }
            StmtKind::If(x) => {
                // Merge branch facts without requiring initialization on terminated paths.
                self.expr(x.cond);
                let incoming = self.state.clone();
                self.stmt(x.if_suite);
                let first = self.state.clone();
                self.state = incoming;
                self.stmt(x.else_suite);
                self.state.intersect(&first);
            }
            StmtKind::While(x) => {
                // Collect body returns without assuming that the loop executes an iteration.
                self.forget_rebound_aliases(s);
                self.expr(x.cond);
                self.analyze_loop_body(x.suite);
            }
            StmtKind::For(x) => {
                // Collect body returns while allowing an empty iterable to skip all writes.
                self.expr(x.iter);
                self.forget_rebound_aliases(s);
                self.record_exception();
                self.analyze_loop_body(x.suite);
            }
            StmtKind::Break(_) => {
                self.exits.push(Exit { kind: ExitKind::Break, state: self.state.clone() });
                self.state.reachable = false;
            }
            StmtKind::Continue(_) => {
                self.exits.push(Exit { kind: ExitKind::Continue, state: self.state.clone() });
                self.state.reachable = false;
            }
            StmtKind::Return(x) => {
                // Save normal-return facts, including writes performed by the return expression.
                self.expr(x.expr);
                if self.state.reachable {
                    self.exits.push(Exit { kind: ExitKind::Return, state: self.state.clone() });
                }
                self.state.reachable = false;
            }
            StmtKind::Throw(x) => {
                // Exceptional exits do not require a fully initialized returned instance.
                self.expr(x.expr);
                self.record_exception();
                self.state.reachable = false;
            }
            StmtKind::Try(x) => self.visit_try(&x),
            _ => {
                for c in stmt_children(self.ast(), s) {
                    self.child(c);
                }
            }
        }
    }

    /// Merge exception-site facts and apply finally to every pending exit kind.
    fn visit_try(&mut self, x: &TryStmt) {
        let previous = self.exits.len();
        self.exceptions.push(InitializedFields::unreachable());
        self.stmt(x.suite);
        let thrown = self.exceptions.last().unwrap().clone();
        *self.exceptions.last_mut().unwrap() = InitializedFields::unreachable();
        self.stmt(x.else_suite);
        let mut normal = self.state.clone();
        let mut catches_all = false;
        for &h in &x.items {
            let handler = self.ast()[h].kind.as_except().unwrap().clone();
            self.state = thrown.clone();
            self.state.aliases.remove(&handler.var);
            self.stmt(handler.suite);
            let st = self.state.clone();
            normal.intersect(&st);
            let all = match handler.exc {
                None => true,
                Some(ex) => self.ast()[ex].ty.is_some_and(|t| {
                    let t = self.tv.extract_type(t);
                    self.tv.cache.types.is(t, &stdlib().base_exception)
                }),
            };
            if all {
                catches_all = true;
            }
        }
        if !catches_all {
            self.exceptions.last_mut().unwrap().intersect(&thrown);
        }
        let unhandled = self.exceptions.pop().unwrap();
        let pending: Vec<Exit> = self.exits.drain(previous..).collect();
        for exit in pending {
            self.state = exit.state;
            self.stmt(x.finally);
            if self.state.reachable {
                self.exits.push(Exit { kind: exit.kind, state: self.state.clone() });
            }
        }
        self.state = unhandled;
        self.stmt(x.finally);
        self.record_exception();
        self.state = normal;
        self.stmt(x.finally);
    }
}

/// Statement lists produced while processing a class body.
#[derive(Default)]
struct ClassParts {
    cls: Vec<StmtId>,
    vars: Vec<StmtId>,
    fns: Vec<StmtId>,
    enum_init: Vec<StmtId>,
    add_later: Vec<super::ItemRef>,
}

impl<'a> TypecheckVisitor<'a> {
    /// `argsToParse`: the class's own params, or (for `@extend`) the extended class's.
    fn class_args_to_parse(&self, s: StmtId, canonical: &str) -> Vec<Param> {
        if self.cache.ast[s].attrs.has(Attr::Extend) {
            let a = self.cache.classes.get(canonical).and_then(|c| c.ast).unwrap();
            self.cache.ast[a].kind.as_class().unwrap().items.clone()
        } else {
            self.cache.ast[s].kind.as_class().unwrap().items.clone()
        }
    }

    /// `visit(ClassStmt*)`: parse a class (type) declaration and add a (generic) type to
    /// the context.
    pub fn visit_class(&mut self, s: StmtId) -> Result<()> {
        let types = self.types();
        let st = stdlib();
        // Get root name
        let name = self.cache.ast[s].kind.as_class().unwrap().name.clone();
        let has = |v: &Self, a: Attr| v.cache.ast[s].attrs.has(a);

        // Generate/find class' canonical name (unique ID) and AST
        let canonical_name: String;
        // classItem will be added later when the scope is different
        let (module, block_level) = {
            let c = self.ctx.borrow();
            (c.module(), c.block_level)
        };
        let mut class_item = TypecheckItem::new("", "", module, None, block_level);
        class_item.src = self.cache.ast[s].src;
        let mut timed_item = None;
        let class_item_ref;
        let typ: TypeId;
        if !has(self, Attr::Extend) {
            let internal = has(self, Attr::Internal);
            canonical_name = self.gen_canonical(&name, !internal, internal);
            class_item.canonical_name = canonical_name.clone();
            let t = if canonical_name == st.union {
                types.new_union(vec![])
            } else {
                types.new_class(&canonical_name, vec![], vec![])
            };
            if self.cache.ast.class_is_record(s) {
                types.with_class_mut(t, |c| c.is_tuple = true);
            }
            types.set_src(t, self.cache.ast[s].src);
            typ = t;
            class_item.ty = Some(t);
            if canonical_name != st.type_ {
                class_item.ty = Some(self.instantiate_type_var(t)?);
            }
            let ci = class_item.into_ref();
            let ti = ci.borrow().clone().into_ref();
            // Reference types are added to the context here.
            // Tuple types are added after class contents are parsed to prevent
            // recursive record types (note: these are allowed for reference types)
            if !has(self, Attr::Tuple) {
                self.ctx.borrow_mut().add(&name, ti.clone());
                TypeContext::add_always_visible(self.cache, &self.ctx, ci.clone(), false);
            }
            timed_item = Some(ti);
            class_item_ref = Some(ci);
        } else {
            class_item_ref = None;
            // Find the canonical name and AST of the class that is to be extended
            {
                let c = self.ctx.borrow();
                if !c.is_global() || c.is_conditional() {
                    drop(c);
                    typhon_common::bail_e!(ErrorKind::EXPECTED_TOPLEVEL, self.src, "class extension");
                }
            }
            let t = self.get_time();
            let val = self.find(&name, t);
            let Some(val) = val.filter(|v| v.borrow().is_type(&types)) else {
                typhon_common::bail_e!(ErrorKind::CLASS_ID_NOT_FOUND, self.src, name);
            };
            let (vname, vty) = {
                let v = val.borrow();
                (v.canonical_name.clone(), v.ty())
            };
            typ = if vname == st.type_ { types.get_class(vty).unwrap() } else { self.extract_class_type(vty).unwrap() };
            let cast = self.cache.get_class_by_type(typ).and_then(|c| c.ast).unwrap();
            if self.cache.ast[cast].attrs.has(Attr::NoExtend)
                && !self.ctx.borrow().is_stdlib_loading
                && !has(self, Attr::AutoGenerated)
            {
                typhon_common::bail_e!(ErrorKind::CLASS_NO_EXTEND, self.src, name);
            }
            canonical_name = types.class_name(typ).unwrap().to_string();
        }
        self.cache.classes.entry(canonical_name.clone()).or_default();

        let mut parts = ClassParts::default();
        // Add the class base
        {
            let mut c = self.ctx.borrow_mut();
            c.bases.push(Base::new(&canonical_name, Some(typ), None));
            c.add_block();
        }
        let r = self.visit_class_body(s, &name, &canonical_name, typ, timed_item.clone(), class_item_ref.clone(), &mut parts);
        {
            let mut c = self.ctx.borrow_mut();
            c.bases.pop();
            c.pop_block();
        }
        if let Err(e) = r {
            if !has(self, Attr::Tuple) {
                self.ctx.borrow_mut().remove(&name);
            }
            self.cache.classes.shift_remove(&name);
            return Err(e);
        }
        for i in std::mem::take(&mut parts.add_later) {
            let n = self.get_unmangled_name(&i.borrow().canonical_name);
            self.ctx.borrow_mut().add(&n, i);
        }

        // Extensions are not needed as the cache is already populated
        if !has(self, Attr::Extend) {
            let c = self.cache.classes.get(&canonical_name).and_then(|c| c.ast);
            typhon_common::seqassert!(c.is_some(), "not a class AST for {}", canonical_name);
            let c = c.unwrap();
            self.set_done_stmt(c);
            parts.cls.push(c);

            if canonical_name == "Int" {
                // Auto-realize int64
                let rt = self.instantiate_type(typ, None)?;
                let rn = "Int[64]";
                let module = self.ctx.borrow().module();
                let val = TypecheckItem::new(rn, "", module, Some(rt), 0).into_ref();
                let vt = self.instantiate_type_var(rt)?;
                val.borrow_mut().ty = Some(vt);
                TypeContext::add_always_visible(self.cache, &self.ctx, val.clone(), true);
                self.cache.type_ctx.borrow_mut().add("int", val);

                let g = self.extract_class_generic(rt, 0);
                let s64 = types.new_int_static(64);
                self.unify(g, s64)?;

                self.cache.class_realization_cnt += 1;
                let real = Rc::new(RefCell::new(ClassRealization {
                    ty: Some(rt),
                    id: self.cache.class_realization_cnt,
                    ..Default::default()
                }));
                self.cache.classes.get_mut(&canonical_name).unwrap().realizations.insert(rn.to_string(), real);
                self.make_ir_type(rt)?;
            }
        }

        let mut cls_stmts = std::mem::take(&mut parts.cls);
        cls_stmts.extend(parts.fns.iter().copied());
        for a in std::mem::take(&mut parts.vars) {
            // Transform class variables here to allow self-references
            cls_stmts.push(self.tr_stmt(a)?);
        }
        // enum_test: allocate singletons only after the class layout and methods exist.
        for e in std::mem::take(&mut parts.enum_init) {
            cls_stmts.push(self.tr_stmt(e)?);
        }
        self.result_stmt = Some(self.n_stmt(SuiteStmt { items: cls_stmts }));
        Ok(())
    }

    /// Python fields annotated with a bare generic class (`dq: list`, `q: queue.Queue`,
    /// `q: queue.Queue | None = None`) do not fix the element type. In user code give each
    /// such field a hidden generic `T_<field>` (default `NoneType`) that is deduced from the
    /// constructor arguments, like an unannotated deduced member.
    fn generalize_bare_generic_fields(&mut self, args: &mut Vec<Param>) -> Result<bool> {
        let st = stdlib();
        if !self.is_user_module() {
            return Ok(false);
        }
        let mut new_generics = vec![];
        for i in 0..args.len() {
            let a = &args[i];
            if !a.is_value() || self.cache.ast.is_class_var(a) {
                continue;
            }
            let Some(t) = a.ty else { continue };
            // `X` or `X | None` / `Optional[X]` with `X` a bare name.
            let (bare, optional) = match &self.cache.ast[t].kind {
                ExprKind::Id(_) | ExprKind::Dot(_) => (t, false),
                ExprKind::Binary(b) if b.op == "|" => {
                    let is_none = |e: Option<ExprId>, v: &Self| e.is_some_and(|e| matches!(v.cache.ast[e].kind, ExprKind::None(_)) || is_id(&v.cache.ast, Some(e), "None"));
                    let pick = if is_none(b.rexpr, self) { b.lexpr } else if is_none(b.lexpr, self) { b.rexpr } else { None };
                    match pick {
                        Some(x) if matches!(self.cache.ast[x].kind, ExprKind::Id(_) | ExprKind::Dot(_)) => (x, true),
                        _ => continue,
                    }
                }
                ExprKind::Index(ix) if is_id(&self.cache.ast, ix.expr, "Optional") => match ix.index {
                    Some(x) if matches!(self.cache.ast[x].kind, ExprKind::Id(_) | ExprKind::Dot(_)) => (x, true),
                    _ => continue,
                },
                _ => continue,
            };
            let probe = self.cache.ast.clone_expr(bare, false);
            let Ok(Some(te)) = self.transform_type(Some(probe), true) else { continue };
            let Some(ct) = self.extract_class_type_expr(te) else { continue };
            let types = self.types();
            let ng = types.generics(types.get_class(ct).unwrap()).len();
            if ng == 0 || types.can_realize(ct) {
                continue;
            }
            // `list` -> `list[T_<field>_0]` (each class generic becomes a hidden class generic);
            // an unset optional field then still realizes (`Optional[List[NoneType]]`).
            let mut idx = vec![];
            for j in 0..ng {
                let g = format!("T_{}_{}", a.name, j);
                new_generics.push(g.clone());
                idx.push(self.id_n(&g));
            }
            let index = if idx.len() == 1 { idx[0] } else { self.n_expr(TupleExpr { items: idx }) };
            let base = self.cache.ast.clone_expr(bare, false);
            let gid = self.n_expr(IndexExpr { expr: Some(base), index: Some(index) });
            let nty = if optional {
                let o = self.id_n(&st.optional);
                self.n_expr(IndexExpr { expr: Some(o), index: Some(gid) })
            } else {
                gid
            };
            args[i].ty = Some(nty);
        }
        let any = !new_generics.is_empty();
        for g in new_generics.into_iter().rev() {
            let ty = self.id_n(&st.type_);
            let def = self.id_n(&st.none_type);
            let p = self.cache.ast.new_param(Default::default(), g, Some(ty), Some(def), 1, ParamKind::Normal);
            args.insert(0, p);
        }
        Ok(any)
    }

    /// A bare base class `A` whose members were auto-deduced from its `__init__` is
    /// implicitly generic (`A[T_c]`). Python code writes `class B(A)`, so give `B` the same
    /// hidden generics and rewrite the base to `A[T_c, ...]`; otherwise the base can never
    /// be realized ("cannot realize base").
    fn propagate_deduced_base_generics(&mut self, s: StmtId, args: &mut Vec<Param>) -> Result<()> {
        let st = stdlib();
        let mut bases = self.cache.ast[s].kind.as_class().unwrap().base_classes.clone();
        let mut changed = false;
        let mut raw_base: Option<ExprId> = None;
        for b in bases.iter_mut() {
            if self.cache.ast[*b].kind.as_id().is_none() {
                continue;
            }
            let probe = self.cache.ast.clone_expr(*b, false);
            let Ok(Some(t)) = self.transform_type(Some(probe), true) else { continue };
            let Some(ct) = self.extract_class_type_expr(t) else { continue };
            let Some(bast) = self.cache.get_class_by_type(ct).and_then(|c| c.ast) else { continue };
            if !self.cache.ast[bast].attrs.has(Attr::ClassDeduce) {
                continue;
            }
            let gens: Vec<String> = self.cache.ast[bast]
                .kind
                .as_class()
                .unwrap()
                .items
                .iter()
                .filter(|p| p.is_generic())
                .map(|p| self.get_unmangled_name(&p.name))
                .filter(|n| n.starts_with("T_"))
                .collect();
            if gens.is_empty() {
                continue;
            }
            let mut idx = vec![];
            for g in gens {
                if !args.iter().any(|p| p.is_generic() && self.get_unmangled_name(&p.name) == g) {
                    let ty = self.id_n(&st.type_);
                    let def = self.id_n(&st.none_type);
                    let p = self.cache.ast.new_param(Default::default(), g.clone(), Some(ty), Some(def), 1, ParamKind::Normal);
                    args.insert(0, p);
                }
                idx.push(self.id_n(&g));
            }
            let index = if idx.len() == 1 { idx[0] } else { self.n_expr(TupleExpr { items: idx }) };
            if raw_base.is_none() {
                raw_base = Some(self.cache.ast.clone_expr(*b, false));
            }
            *b = self.n_expr(IndexExpr { expr: Some(*b), index: Some(index) });
            changed = true;
        }
        if changed {
            self.cache.ast[s].kind.as_class_mut().unwrap().base_classes = bases;
            // The subclass `__init__` (which forwards to the base `__init__`) must defer the
            // inherited hidden generics just like the base's own deducing `__init__`.
            self.cache.ast[s].attrs.set(Attr::ClassDeduce);
            let suite = self.cache.ast[s].kind.as_class().unwrap().suite;
            for sp in self.class_methods_opt(suite)? {
                let a = &mut self.cache.ast[sp];
                if a.kind.as_function().is_some_and(|f| f.name == "__init__") && !a.attrs.has(Attr::ClassDeduce) {
                    a.attrs.set_value(Attr::ClassDeduce, crate::ast::AttrValue::StrList(vec![]));
                }
                if let Some(rb) = raw_base {
                    self.rewrite_super_init(sp, rb);
                }
            }
        }
        Ok(())
    }

    /// `super().__init__(args)` -> `Base.__init__(self, args)` at the top level of `__init__`,
    /// so the base's hidden generics are inferred from the forwarded arguments.
    fn rewrite_super_init(&mut self, f: StmtId, base: ExprId) {
        let Some(fd) = self.cache.ast[f].kind.as_function().cloned() else { return };
        if fd.name != "__init__" {
            return;
        }
        let Some(slf) = fd.items.first().map(|p| p.name.clone()) else { return };
        let Some(items) = fd.suite.and_then(|x| self.cache.ast[x].kind.as_suite()).map(|x| x.items.clone()) else { return };
        let super_name = mangle::func("", "super");
        for st in items {
            let Some(ex) = self.cache.ast[st].kind.as_expr().and_then(|x| x.expr) else { continue };
            let Some(call) = self.cache.ast[ex].kind.as_call().cloned() else { continue };
            let Some(dot) = call.expr.and_then(|d| self.cache.ast[d].kind.as_dot().cloned()) else { continue };
            if dot.member != "__init__" {
                continue;
            }
            let Some(inner) = dot.expr.and_then(|d| self.cache.ast[d].kind.as_call().cloned()) else { continue };
            let is_super = inner.items.is_empty()
                && inner.expr.and_then(|i| self.cache.ast[i].kind.as_id()).is_some_and(|i| i.value == super_name || i.value == "super");
            if !is_super {
                continue;
            }
            let b = self.cache.ast.clone_expr(base, false);
            let d = self.dot_n(b, "__init__");
            let sid = self.id_n(&slf);
            let mut args = vec![self.cache.ast.call_arg("", Some(sid))];
            args.extend(call.items.iter().cloned());
            let c = self.cache.ast[ex].kind.as_call_mut().unwrap();
            c.expr = Some(d);
            c.items = args;
        }
    }

    fn set_class_items(&mut self, s: StmtId, canonical: &str, items: Vec<Param>) {
        if self.cache.ast[s].attrs.has(Attr::Extend) {
            let a = self.cache.classes.get(canonical).and_then(|c| c.ast).unwrap();
            self.cache.ast[a].kind.as_class_mut().unwrap().items = items;
        } else {
            self.cache.ast[s].kind.as_class_mut().unwrap().items = items;
        }
    }

    /// The base-guarded `try` block of `visit(ClassStmt*)`.
    #[allow(clippy::too_many_arguments)]
    fn visit_class_body(
        &mut self,
        s: StmtId,
        name: &str,
        canonical_name: &str,
        typ: TypeId,
        timed_item: Option<super::ItemRef>,
        class_item: Option<super::ItemRef>,
        parts: &mut ClassParts,
    ) -> Result<()> {
        let types = self.types();
        let st = stdlib();
        let has = |v: &Self, a: Attr| v.cache.ast[s].attrs.has(a);
        let extend = has(self, Attr::Extend);

        // Parse and add class generics
        let mut args: Vec<Param> = vec![];
        let mut args_to_parse = self.class_args_to_parse(s, canonical_name);
        if extend {
            for a in &args_to_parse {
                if !a.is_generic() {
                    continue;
                }
                let val = self.force_find(&a.name);
                let (vt, vc) = {
                    let v = val.borrow();
                    (v.ty(), v.canonical_name.clone())
                };
                types.with_link_mut(vt, |l| l.kind = LinkKind::Unbound);
                let un = self.get_unmangled_name(&vc);
                self.ctx.borrow_mut().add(&un, val);
                args.push(Param::with_status(vc, None, None, a.status));
            }
        } else {
            if has(self, Attr::ClassDeduce) && !self.auto_deduce_members(s, &mut args_to_parse)? {
                self.cache.ast[s].attrs.erase(Attr::ClassDeduce);
            }
            if self.generalize_bare_generic_fields(&mut args_to_parse)? {
                // `__init__` assigning such a field must defer the hidden generics.
                self.cache.ast[s].attrs.set(Attr::ClassDeduce);
                let suite = self.cache.ast[s].kind.as_class().unwrap().suite;
                for sp in self.class_methods_opt(suite)? {
                    let a = &mut self.cache.ast[sp];
                    if a.kind.as_function().is_some_and(|f| f.name == "__init__") && !a.attrs.has(Attr::ClassDeduce) {
                        a.attrs.set_value(Attr::ClassDeduce, crate::ast::AttrValue::StrList(vec![]));
                    }
                }
            }
            self.propagate_deduced_base_generics(s, &mut args_to_parse)?;

            // Add all generics before parent classes, fields and methods
            for a in args_to_parse.iter_mut() {
                if !a.is_generic() {
                    continue;
                }
                let var_name = self.gen_canonical(&a.name, false, false);
                let generic = self.instantiate_unbound();
                let typ_id = types.link(generic).unwrap().id;
                types.with_link_mut(generic, |l| l.generic_name = var_name.clone());
                let dd = self.cache.ast.clone_expr_opt(a.default_value, false);
                let def_type = self.transform_type(dd, false)?;
                if let Some(dt) = def_type {
                    let t = self.extract_type_expr(dt);
                    types.with_link_mut(generic, |l| l.default_type = Some(t));
                }
                let sk = self.cache.ast.get_static_generic(a.ty);
                if sk != LiteralKind::Runtime {
                    // Note: the reference checks `st == LiteralKind::Runtime` inside `if (st)`,
                    // which is unreachable; kept as a no-op.
                    types.with_link_mut(generic, |l| l.static_kind = sk);
                    let val = TypeContext::add_var(self.cache, &self.ctx, &a.name, &var_name, generic, 0, Default::default());
                    val.borrow_mut().generic = true;
                } else {
                    if a.ty.is_some_and(|t| matches!(self.cache.ast[t].kind, ExprKind::Index(_))) {
                        // Parse TraitVar
                        a.ty = self.transform(a.ty)?;
                        let ti = a.ty.and_then(|t| self.cache.ast[t].kind.as_instantiate().cloned());
                        typhon_common::seqassert!(
                            ti.as_ref().is_some_and(|ti| is_id(&self.cache.ast, ti.expr, &st.type_trait)),
                            "not a TypeTrait instantiation"
                        );
                        let ti = ti.unwrap();
                        let l = self.extract_type_expr(ti.items[0]);
                        let tr = match types.link(l).and_then(|x| x.trait_) {
                            Some(tr) => tr,
                            None => types.new_type_trait(l),
                        };
                        types.with_link_mut(generic, |x| x.trait_ = Some(tr));
                    }
                    let v = TypeContext::add_type(self.cache, &self.ctx, &a.name, &var_name, generic, Default::default());
                    v.borrow_mut().generic = true;
                }
                let level = self.ctx.borrow().typecheck_level;
                let g = types.generalize(generic, level);
                types.with_class_mut(typ, |c| c.generics.push(Generic::new(&var_name, Some(g), typ_id, sk)));
                let mut p = Param::with_status(var_name, a.ty, def_type, a.status);
                p.kind = ParamKind::Normal;
                args.push(p);
            }
            self.set_class_items(s, canonical_name, args_to_parse.clone());
        }

        parts.enum_init = self.prepare_enum(s, typ, &mut parts.vars)?;
        let args_to_parse = self.class_args_to_parse(s, canonical_name);

        // Form class type node (e.g. `Foo`, or `Foo[T, U]` for generic classes)
        let mut transformed_type_ast: Option<ExprId> = None;
        if !extend {
            let mut t = self.id_n(canonical_name);
            let mut inst: Option<Vec<ExprId>> = None;
            for a in &args {
                if a.is_generic() {
                    let id = self.id_n(&a.name);
                    let e = self.transform_allow(Some(id), true)?.unwrap();
                    inst.get_or_insert_with(Vec::new).push(e);
                }
            }
            if let Some(items) = inst {
                let base = self.id_n(canonical_name);
                t = self.n_expr(crate::ast::InstantiateExpr { expr: Some(base), items });
            }
            transformed_type_ast = Some(t);
        }

        // A ClassStmt will be separated into class variable assignments, method-free
        // ClassStmts (that include nested classes) and method FunctionStmts
        self.transform_nested_classes(s, &mut parts.cls, &mut parts.vars, &mut parts.fns)?;

        // Collect class fields
        for a in &args_to_parse {
            if !a.is_value() {
                continue;
            }
            if self.cache.ast.is_class_var(a) {
                // Handle class variables. Transform them later to allow self-references
                let var_name = format!("{}.{}", canonical_name, a.name);
                let l = self.id_n(&var_name);
                let decl = self.n_stmt(AssignStmt { lhs: Some(l), rhs: None, type_expr: None, update: UpdateMode::Assign });
                let h = self.tr_stmt(decl)?;
                let pre = self.preamble;
                self.cache.ast.suite_add_stmt(pre, Some(h));
                let val = self.force_find(&var_name);
                let vc = {
                    let mut v = val.borrow_mut();
                    v.base_name = Default::default();
                    v.block_level = 0;
                    v.canonical_name.clone()
                };
                self.register_global(&vc);
                if a.default_value.is_some() {
                    let l = self.id_n(&var_name);
                    let rhs = self.transform(a.default_value)?;
                    let te = a.ty.and_then(|t| self.cache.ast[t].kind.as_index().and_then(|i| i.index));
                    let assign = self.n_stmt(AssignStmt { lhs: Some(l), rhs, type_expr: te, update: UpdateMode::Update });
                    parts.vars.push(assign);
                }
                self.cache.classes.get_mut(canonical_name).unwrap().class_vars.insert(a.name.clone(), var_name);
                self.ctx.borrow_mut().add(&a.name, val);
            } else if !extend {
                let var_name = a.name.clone();
                let tt = self.cache.ast.clone_expr_opt(a.ty, true);
                let ta = self.transform_type(tt, true)?;
                let mut dd = self.cache.ast.clone_expr_opt(a.default_value, false);
                if self.is_dataclass_stmt(s) {
                    dd = self.dataclass_field_default(dd);
                }
                let td = self.transform_allow(dd, true)?;
                args.push(self.cache.ast.new_param(Default::default(), var_name.clone(), ta, td, 0, ParamKind::Normal));
                self.cache.classes.get_mut(canonical_name).unwrap().fields.push(ClassField::new(var_name, None, canonical_name, None));
            }
        }

        // Handle RTTI inheritance. Must come after other arguments are handled.
        if !extend {
            let mut bases = self.cache.ast[s].kind.as_class().unwrap().base_classes.clone();
            let r = self.parse_base_classes(&mut bases, &mut args, s, canonical_name, typ);
            self.cache.ast[s].kind.as_class_mut().unwrap().base_classes = bases;
            r?;
        }

        // ASTs for member arguments to be used for populating magic methods
        let arg_bases: BTreeMap<String, String> = self.cache.classes[canonical_name]
            .fields
            .iter()
            .map(|f| (f.name.clone(), f.base_class.clone()))
            .collect();
        let mut member_args = vec![];
        for a in &args {
            if a.is_value() && arg_bases.get(&a.name).map(|x| x.as_str()).unwrap_or("") == canonical_name {
                member_args.push(self.cache.ast.clone_param(a, false));
            }
        }

        // Handle class members
        if !extend {
            self.ctx.borrow_mut().typecheck_level += 1; // to avoid unifying generics early
            let r = self.class_member_types(canonical_name, &args);
            self.ctx.borrow_mut().typecheck_level -= 1;
            r?;
        }

        // Parse class members (arguments) and methods
        let suite = self.cache.ast[s].kind.as_class().unwrap().suite;
        if !extend {
            // Now that we are done with arguments, add record type to the context
            if has(self, Attr::Tuple) {
                self.ctx.borrow_mut().add(name, timed_item.clone().unwrap());
                TypeContext::add_always_visible(self.cache, &self.ctx, class_item.clone().unwrap(), false);
            }
            // Create a cached AST.
            let (mstatus, mpath) = {
                let c = self.ctx.borrow();
                (c.module_info().status, c.module_info().path.clone())
            };
            let mod_attr = if mstatus == typhon_common::fs::ImportStatus::Stdlib { STDLIB_IMPORT.to_string() } else { mpath.clone() };
            self.cache.ast[s].attrs.set_str(Attr::Module, mod_attr);
            let empty = self.n_stmt(SuiteStmt::default());
            let cargs: Vec<Param> = args.clone();
            let cast = self.n_stmt(ClassStmt { name: canonical_name.to_string(), items: cargs, suite: Some(empty), decorators: vec![], base_classes: vec![] });
            self.cache.ast[cast].attrs = self.cache.ast[s].attrs.clone();
            let bc = self.cache.ast[s].kind.as_class().unwrap().base_classes.clone();
            self.cache.ast[cast].kind.as_class_mut().unwrap().base_classes = bc;
            {
                let cls = self.cache.classes.get_mut(canonical_name).unwrap();
                cls.ast = Some(cast);
                cls.module = mpath;
            }

            // Codegen default magic methods
            // __new__ must be the first
            let methods = self.class_methods_opt(suite)?;
            let custom_new = methods
                .iter()
                .any(|m| self.cache.ast[*m].kind.as_function().is_some_and(|f| f.name == "__new__"));
            let magic = self.cache.ast[s].attrs.get_str_list(Attr::ClassMagic).cloned();
            if let Some(magic) = magic {
                let is_record = self.cache.ast.class_is_record(s);
                let is_tuple = has(self, Attr::Tuple);
                for m in magic {
                    // class_custom_new_err: a custom allocator must not silently fall back
                    // to the generated zero-argument allocator on an invalid call.
                    if m == "new" && custom_new && !is_record {
                        continue;
                    }
                    if matches!(m.as_str(), "frozen" | "kw_only" | "dataclass") {
                        continue;
                    }
                    let op = if m == "repr" && self.is_dataclass_stmt(s) { "repr_dataclass" } else { m.as_str() };
                    let f = self.codegen_magic(op, transformed_type_ast.unwrap(), &member_args, is_tuple)?;
                    let f = self.dataclass_postprocess_magic(s, &m, f);
                    if let Some(f) = self.transform_stmt_opt(f)? {
                        parts.fns.push(f);
                    }
                }
            }
        }

        // Add class methods
        for sp in self.class_methods_opt(suite)? {
            let Some(fp) = self.cache.ast[sp].kind.as_function().cloned() else { continue };
            let nvalues = fp.items.iter().filter(|p| p.is_value()).count();
            for (di, dc) in fp.decorators.iter().enumerate() {
                // Handle @setter setters
                let is_setter = self.cache.ast[*dc].kind.as_dot().is_some_and(|d| {
                    d.member == "setter" && is_id(&self.cache.ast, d.expr, &fp.name)
                });
                if is_setter && nvalues == 2 {
                    let f = self.cache.ast[sp].kind.as_function_mut().unwrap();
                    f.name = format!("{}{}", FN_SETTER_SUFFIX, fp.name);
                    // Note: the reference nulls the decorator slot; Rust decorators are non-null ids.
                    f.decorators.remove(di);
                    break;
                }
            }
            // All tuple methods are marked with AutoGenerated for later convenience
            // (e.g. heterogenous tuple processing).
            if canonical_name == st.tuple {
                self.cache.ast[sp].attrs.set(Attr::AutoGenerated);
            }
            let t = self.tr_stmt(sp)?;
            parts.fns.push(t);
        }

        // After popping context block, record types and nested classes will disappear.
        // Store their references and re-add them to the context after popping
        for c in parts.cls.clone() {
            let n = self.cache.ast[c].kind.as_class().unwrap().name.clone();
            if let Some(i) = self.find(&n, 0) {
                parts.add_later.push(i);
            }
        }
        if has(self, Attr::Tuple) {
            parts.add_later.push(self.force_find(name));
        }

        // Mark functions as virtual
        let banned = ["__init__", "__new__", "__raw__", "__tuplesize__", "__repr_default__"];
        let (methods, mro) = {
            let cls = &self.cache.classes[canonical_name];
            let mut m: Vec<String> = cls.methods.keys().cloned().collect();
            m.sort();
            (m, cls.mro.clone())
        };
        for method in &methods {
            for b in mro.iter().skip(1) {
                // ... in the current class
                let bn = types.class_name(*b).unwrap().to_string();
                if self.get_class_name(&bn).is_some_and(|c| c.methods.contains_key(method)) && !banned.contains(&method.as_str()) {
                    self.cache.classes.get_mut(canonical_name).unwrap().virtuals.insert(method.clone());
                }
            }
            let mut virtuals: Vec<String> = self.cache.classes[canonical_name].virtuals.iter().cloned().collect();
            virtuals.sort();
            for v in &virtuals {
                for b in mro.iter().skip(1) {
                    // ... and in parent classes
                    let bn = types.class_name(*b).unwrap().to_string();
                    self.cache.classes.entry(bn).or_default().virtuals.insert(v.clone());
                }
            }
        }

        // Generalize generics and remove them from the context
        for g in &args {
            if g.is_value() {
                continue;
            }
            let generic = self.force_find(&g.name).borrow().ty;
            if g.status == ParamStatus::Generic {
                // Generalize generics. Hidden generics are linked to the class generics so
                // ignore them
                let gl = generic.and_then(|t| types.link(t));
                typhon_common::seqassert!(gl.as_ref().is_some_and(|l| l.kind != LinkKind::Link), "generic has been unified");
                if let Some(t) = generic {
                    types.with_link_mut(t, |l| l.kind = LinkKind::Generic);
                }
            }
            self.ctx.borrow_mut().remove(&g.name);
        }

        // Debug information
        if !canonical_name.starts_with("Tuple") {
            let cls = &self.cache.classes[canonical_name];
            typhon_common::log_flag!(REALIZE, "[class] {} -> {} / {}", canonical_name, types.debug_string(typ, 1), cls.fields.len());
            for m in &cls.fields {
                typhon_common::log_flag!(REALIZE, "       - member: {}: {}", m.name, m.ty.map(|t| types.debug_string(t, 1)).unwrap_or_default());
            }
            let mut ms: Vec<_> = cls.methods.iter().collect();
            ms.sort();
            for (k, v) in ms {
                typhon_common::log_flag!(REALIZE, "       - method: {}: {}", k, v);
            }
            for m in &cls.mro {
                typhon_common::log_flag!(REALIZE, "       - mro: {}", types.debug_string(*m, 1));
            }
        }
        Ok(())
    }

    /// Field types of a newly defined class (at `typecheckLevel + 1`).
    fn class_member_types(&mut self, canonical_name: &str, args: &[Param]) -> Result<()> {
        let types = self.types();
        let st = stdlib();
        if canonical_name == st.tuple {
            // Special tuple handling!
            for aj in 0..MAX_TUPLE {
                let gen_name = format!("T{}", aj + 1);
                let gen_can_name = self.gen_canonical(&gen_name, false, false);
                let generic = self.instantiate_unbound();
                types.with_link_mut(generic, |l| l.generic_name = gen_name.clone());
                let te = self.id_n(&gen_can_name);
                let level = self.ctx.borrow().typecheck_level;
                let g = types.generalize(generic, level);
                self.cache
                    .classes
                    .get_mut(canonical_name)
                    .unwrap()
                    .fields
                    .push(ClassField::new(format!("item{}", aj + 1), Some(g), "", Some(te)));
            }
        } else {
            let mut aj = 0;
            for a in args {
                if a.is_value() && !self.cache.ast.is_class_var(a) {
                    let te = self.cache.ast.clone_expr_opt(a.ty, true);
                    let t = self.extract_type_expr(a.ty.unwrap());
                    let level = self.ctx.borrow().typecheck_level;
                    let g = types.generalize(t, level - 1);
                    types.set_src(g, self.cache.ast[a.ty.unwrap()].src);
                    let f = &mut self.cache.classes.get_mut(canonical_name).unwrap().fields[aj];
                    f.type_expr = te;
                    f.ty = Some(g);
                    aj += 1;
                }
            }
        }
        Ok(())
    }
}

impl<'a> TypecheckVisitor<'a> {
    /// `prepareEnum(stmt, typ, values)`: lower Enum declarations to ordinary reference
    /// classes and singleton globals.
    pub fn prepare_enum(&mut self, s: StmtId, typ: TypeId, values: &mut Vec<StmtId>) -> Result<Vec<StmtId>> {
        let types = self.types();
        let st = stdlib();
        if self.cache.ast[s].attrs.has(Attr::Extend) {
            return Ok(vec![]);
        }
        let mut enumeration = false;
        let bases = self.cache.ast[s].kind.as_class().unwrap().base_classes.clone();
        for base in &bases {
            let c = self.cache.ast.clone_expr(*base, false);
            let Some(te) = self.transform_type(Some(c), true)? else { continue };
            let Some(t) = self.extract_class_type_expr(te) else { continue };
            if !self.is_enum(t) {
                continue;
            }
            // enum_extend_err: members already have ordinary classVars bindings;
            // enum_test: memberless bases may also have ordinary typed ClassVars;
            // only bindings holding instances of the enum are enumeration members.
            let tn = types.class_name(t).unwrap().to_string();
            let mut cvs: Vec<(String, String)> =
                self.cache.get_class_by_type(t).unwrap().class_vars.iter().map(|(a, b)| (a.clone(), b.clone())).collect();
            cvs.sort();
            for (name, var) in cvs {
                if !name.starts_with("__") && types.is(self.force_find(&var).borrow().ty(), &tn) {
                    let src = self.cache.ast[s].src;
                    typhon_common::bail_e!(ErrorKind::CUSTOM, src, "cannot extend an enumeration with members");
                }
            }
            enumeration = true;
        }
        if !enumeration {
            return Ok(vec![]);
        }
        let ssrc = self.cache.ast[s].src;
        if bases.len() != 1 || self.cache.ast[s].attrs.has(Attr::Tuple) {
            typhon_common::bail_e!(ErrorKind::CUSTOM, ssrc, "Enum requires a single enumeration base class");
        }
        let suite = self.cache.ast[s].kind.as_class().unwrap().suite;
        for m in self.class_methods_opt(suite)? {
            if self.cache.ast[m].kind.as_function().is_some_and(|f| f.name == "__init__" || f.name == "__new__") {
                let src = self.cache.ast[m].src;
                typhon_common::bail_e!(ErrorKind::CUSTOM, src, "custom enumeration constructors are not supported");
            }
        }

        let tname = types.class_name(typ).unwrap().to_string();
        let mut names: Vec<String> = vec![];
        let mut raw_values: Vec<ExprId> = vec![];
        let mut members: Vec<ExprId> = vec![];
        let mut init: Vec<StmtId> = vec![];
        let member_buffer = self.get_temporary_var("enum_members");
        let mut auto_maximum = String::new();
        let helper = |v: &mut Self, n: &str| v.id_n(&mangle::func("std.enum", n));
        let class_var = |v: &mut Self, tname: &str| {
            let cv = v.id_n("ClassVar");
            let sf = v.id_n(tname);
            v.n_expr(IndexExpr { expr: Some(cv), index: Some(sf) })
        };

        // enum_test: class-body references use raw values (B = A + 1), and each
        // initializer runs once. Bindings are replaced by members later in visit(ClassStmt).
        let mut items = self.cache.ast[s].kind.as_class().unwrap().items.clone();
        for a in items.iter_mut() {
            if !a.is_value() || !self.cache.ast.is_class_var(a) || a.default_value.is_none() || a.name.starts_with("__") {
                continue;
            }
            if ["_name_", "_value_", "name", "value"].contains(&a.name.as_str()) {
                typhon_common::bail_e!(ErrorKind::CUSTOM, a.src, "reserved enumeration member name '{}'", a.name);
            }
            let mut value = self.cache.ast.clone_expr(a.default_value.unwrap(), false);
            if let Some(call) = self.cache.ast[value].kind.as_call().cloned() {
                let hc = self.cache.ast.clone_expr(call.expr.unwrap(), false);
                let head = self.tr(hc)?;
                if Self::is_type_expr(self.cache, head)
                    && self.extract_class_type_expr(head).is_some_and(|t| types.is(t, &mangle::class("std.enum", "auto")))
                {
                    if !call.items.is_empty() {
                        let src = self.cache.ast[value].src;
                        typhon_common::bail_e!(ErrorKind::CUSTOM, src, "auto() does not accept arguments");
                    }
                    value = if raw_values.is_empty() {
                        self.n_expr(IntExpr::new(1))
                    } else {
                        let h = helper(self, "_next_value");
                        let am = self.id_n(&auto_maximum);
                        let rv = self.cache.ast.clone_expr(raw_values[0], false);
                        self.call_n(h, vec![am, rv])
                    };
                }
            }
            let raw = self.get_temporary_var("enum_value");
            let l = self.id_n(&raw);
            let asg = self.n_stmt(AssignStmt { lhs: Some(l), rhs: Some(value), type_expr: None, update: UpdateMode::Assign });
            values.push(self.tr_stmt(asg)?);
            let binding = self.force_find(&raw);
            let bc = {
                let mut b = binding.borrow_mut();
                b.base_name = Default::default();
                b.block_level = 0;
                b.canonical_name.clone()
            };
            self.register_global(&bc);
            self.ctx.borrow_mut().add(&a.name, binding);
            raw_values.push(self.id_n(&bc));
            let last = *raw_values.last().unwrap();
            if auto_maximum.is_empty() {
                auto_maximum = self.get_temporary_var("enum_maximum");
                let l = self.id_n(&auto_maximum);
                let h = helper(self, "_auto_maximum");
                let rv = self.cache.ast.clone_expr(last, false);
                let z = self.n_expr(IntExpr::new(0));
                let t = self.n_expr(crate::ast::BoolExpr { value: true });
                let c = self.call_n(h, vec![rv, z, t]);
                let asg = self.n_stmt(AssignStmt { lhs: Some(l), rhs: Some(c), type_expr: None, update: UpdateMode::Assign });
                values.push(self.tr_stmt(asg)?);
                let mb = self.force_find(&auto_maximum);
                let mc = {
                    let mut b = mb.borrow_mut();
                    b.base_name = Default::default();
                    b.block_level = 0;
                    b.canonical_name.clone()
                };
                self.register_global(&mc);
                auto_maximum = mc;
            } else {
                let l = self.id_n(&auto_maximum);
                let h = helper(self, "_auto_maximum");
                let rv = self.cache.ast.clone_expr(last, false);
                let am = self.id_n(&auto_maximum);
                let f = self.n_expr(crate::ast::BoolExpr { value: false });
                let c = self.call_n(h, vec![rv, am, f]);
                let asg = self.n_stmt(AssignStmt { lhs: Some(l), rhs: Some(c), type_expr: None, update: UpdateMode::Update });
                values.push(self.tr_stmt(asg)?);
            }
            let member = self.id_n(&format!("{}.{}", tname, a.name));
            let mc = self.cache.ast.clone_expr(member, false);
            let h = helper(self, "_make_member");
            let sf = self.id_n(&tname);
            let ns = self.n_expr(StringExpr::new(a.name.clone(), ""));
            let rv = self.cache.ast.clone_expr(last, false);
            let mbuf = self.id_n(&member_buffer);
            let idx = self.n_expr(IntExpr::new(members.len() as i64));
            let c = self.call_n(h, vec![sf, ns, rv, mbuf, idx]);
            let sf2 = self.id_n(&tname);
            init.push(self.n_stmt(AssignStmt { lhs: Some(mc), rhs: Some(c), type_expr: Some(sf2), update: UpdateMode::Update }));
            names.push(a.name.clone());
            members.push(member);
            a.ty = Some(class_var(self, &tname));
            a.default_value = None;
        }
        if !raw_values.is_empty() {
            let rv0 = self.cache.ast[raw_values[0]].kind.as_id().unwrap().value.clone();
            let value_type = self.force_find(&rv0).borrow().ty();
            for v in &raw_values {
                let vn = self.cache.ast[*v].kind.as_id().unwrap().value.clone();
                let t = self.force_find(&vn).borrow().ty();
                let a1 = self.get_underlying_static_type(value_type)?;
                let a2 = self.get_underlying_static_type(t)?;
                if types.unify(a1, a2, None, self)? < 0 {
                    typhon_common::bail_e!(ErrorKind::CUSTOM, ssrc, "enumeration values must have the same type");
                }
            }
            let ut = self.get_underlying_static_type(value_type)?;
            let Some(realized) = self.realize(ut)? else {
                typhon_common::bail_e!(ErrorKind::CUSTOM, ssrc, "cannot infer enumeration value type");
            };
            let rn = types.realized_name(realized);
            let vt = self.id_n(&rn);
            items.push(self.cache.ast.new_param(Default::default(), "_value_", Some(vt), None, 0, ParamKind::Normal));
            let vi = self.id_n(&mangle::class("std.enum", "_ValueIndex"));
            let r1 = self.id_n(&rn);
            let sf = self.id_n(&tname);
            let tup = self.n_expr(TupleExpr { items: vec![r1, sf] });
            let index_type = self.n_expr(IndexExpr { expr: Some(vi), index: Some(tup) });
            let cv = self.id_n("ClassVar");
            let cvt = self.n_expr(IndexExpr { expr: Some(cv), index: Some(index_type) });
            items.push(self.cache.ast.new_param(Default::default(), "__value_index__", Some(cvt), None, 0, ParamKind::Normal));
            let l = self.id_n(&member_buffer);
            let p = self.id_n("Ptr");
            let sf = self.id_n(&tname);
            let pt = self.n_expr(IndexExpr { expr: Some(p), index: Some(sf) });
            let n = self.n_expr(IntExpr::new(members.len() as i64));
            let c = self.call_n(pt, vec![n]);
            init.insert(0, self.n_stmt(AssignStmt { lhs: Some(l), rhs: Some(c), type_expr: None, update: UpdateMode::Assign }));
        }
        let mapping_type = {
            let od = self.id_n(&mangle::class("std.internal.types.collections.dict", "OrderedDict"));
            let s1 = self.id_n(&st.string);
            let sf = self.id_n(&tname);
            let tup = self.n_expr(TupleExpr { items: vec![s1, sf] });
            self.n_expr(IndexExpr { expr: Some(od), index: Some(tup) })
        };
        let cv = self.id_n("ClassVar");
        let mt = self.cache.ast.clone_expr(mapping_type, false);
        let cvt = self.n_expr(IndexExpr { expr: Some(cv), index: Some(mt) });
        items.push(self.cache.ast.new_param(Default::default(), "__members__", Some(cvt), None, 0, ParamKind::Normal));
        let mut entries = vec![];
        for (i, n) in names.iter().enumerate() {
            let ns = self.n_expr(StringExpr::new(n.clone(), ""));
            let m = self.cache.ast.clone_expr(members[i], false);
            entries.push(self.n_expr(TupleExpr { items: vec![ns, m] }));
        }
        let mapping = if names.is_empty() {
            self.call_n(mapping_type, vec![])
        } else {
            let l = self.n_expr(ListExpr { items: entries });
            let it = self.dot_n(l, "__iter__");
            let c = self.call_n(it, vec![]);
            self.call_n(mapping_type, vec![c])
        };
        let l = self.id_n(&format!("{tname}.__members__"));
        init.push(self.n_stmt(AssignStmt { lhs: Some(l), rhs: Some(mapping), type_expr: None, update: UpdateMode::Update }));
        if !raw_values.is_empty() {
            let l = self.id_n(&format!("{tname}.__value_index__"));
            let h = helper(self, "_make_value_index");
            let sf = self.id_n(&tname);
            let rv = self.cache.ast.clone_expr(raw_values[0], false);
            let c = self.call_n(h, vec![sf, rv]);
            init.push(self.n_stmt(AssignStmt { lhs: Some(l), rhs: Some(c), type_expr: None, update: UpdateMode::Update }));
        }
        self.cache.ast[s].kind.as_class_mut().unwrap().items = items;

        self.cache.ast[s].attrs.erase(Attr::ClassDeduce);
        if let Some(magic) = self.cache.ast[s].attrs.get_str_list_mut(Attr::ClassMagic) {
            magic.retain(|n| n != "init" && n != "new");
        }
        Ok(init)
    }
}

impl<'a> TypecheckVisitor<'a> {
    /// `parseBaseClasses(baseClasses, args, attr, canonicalName, typ)`: parse statically
    /// inherited classes. Returns their types (MRO order) and updates the class fields.
    pub fn parse_base_classes(
        &mut self,
        bases: &mut Vec<ExprId>,
        args: &mut Vec<Param>,
        attr: StmtId,
        canonical: &str,
        typ: TypeId,
    ) -> Result<Vec<TypeId>> {
        let types = self.types();
        let st = stdlib();
        let mut asts: Vec<TypeId> = vec![];
        let mut mro: Vec<Vec<TypeId>> = vec![vec![typ]];
        for cls in bases.iter_mut() {
            // Get the base class and generic replacements (e.g., if there is Bar[T],
            // Bar in Foo(Bar[int]) will have `T = int`)
            let orig = self.cache.ast.clone_expr(*cls, false);
            *cls = self.transform_type(Some(*cls), true)?.unwrap();
            if self.expr_class(*cls).is_none() {
                let n = crate::visitors::format::FormatVisitor::apply_expr(&self.cache.ast, orig, None, false);
                let n = self.get_unmangled_name(&n);
                typhon_common::bail_e!(ErrorKind::CUSTOM, self.src, "cannot use '{}' as a base class", n);
            }
            let cls_typ = self.extract_class_type_expr(*cls).unwrap();
            // `class C(Base)` where Base has `t: Literal[str] = "x"`: a bare base keeps
            // its static defaults (otherwise the inherited generic stays unbound).
            for bg in types.generics(cls_typ) {
                if bg.static_kind == LiteralKind::Runtime {
                    continue;
                }
                let gt = types.follow(bg.ty());
                if let Some(l) = types.link(gt) {
                    if matches!(l.kind, LinkKind::Unbound) {
                        if let Some(d) = l.default_type {
                            let mut undo = crate::types::Unification::new();
                            types.unify(gt, d, Some(&mut undo), self)?;
                        }
                    }
                }
            }
            let cls_name = types.class_name(cls_typ).unwrap().to_string();
            let Some(cached_ast) = self.cache.get_class_by_type(cls_typ).and_then(|c| c.ast) else {
                typhon_common::bail_e!(ErrorKind::CLASS_NO_INHERIT, self.src, "nested", "surrounding");
            };
            // Sanity checks
            let attr_tuple = self.cache.ast[attr].attrs.has(Attr::Tuple);
            if attr_tuple {
                typhon_common::bail_e!(ErrorKind::CLASS_NO_INHERIT, self.src, "tuple", "other");
            }
            if !attr_tuple && self.cache.ast[cached_ast].attrs.has(Attr::Tuple) {
                typhon_common::bail_e!(ErrorKind::CLASS_TUPLE_INHERIT, self.src);
            }
            if self.cache.ast[cached_ast].attrs.has(Attr::Internal) {
                typhon_common::bail_e!(ErrorKind::CLASS_NO_INHERIT, self.src, "internal", "other");
            }

            // Mark parent classes as polymorphic as well.
            let (has_rtti, mro_len) = {
                let c = self.cache.get_class_by_type(cls_typ).unwrap();
                let an = self.cache.ast[cached_ast].kind.as_class().unwrap().name.clone();
                (c.has_rtti(&an), c.mro.len())
            };
            if !has_rtti {
                let object = st.object.clone();
                if cls_name != object && mro_len == 1 {
                    let object_type = self.get_stdlib_type(&object)?;
                    self.cache.get_class_by_type_mut(cls_typ).unwrap().mro.push(object_type);
                    self.cache.get_class_by_type_mut(object_type).unwrap().descendants.insert(cls_name.clone());

                    // Fix IR stuff [TODO: maybe move this to translate?!]
                    self.realize(object_type)?;
                    let object_ir = self.make_ir_type(object_type)?;
                    let reals: Vec<ClassRealizationRef> =
                        self.cache.get_class_by_type(cls_typ).unwrap().realizations.values().cloned().collect();
                    for r in reals {
                        r.borrow_mut().bases.push(object_type);
                        let ir = r.borrow().ir;
                        if let (Some(ir), Some(m)) = (ir, self.cache.module.clone()) {
                            let mut m = m.borrow_mut();
                            if let typhon_ir::TypeKind::Ref { polymorphic, .. } = &mut m.ty_mut(ir).kind {
                                *polymorphic = true;
                                if let Some(info) = m.ty_mut(ir).runtime_info.as_mut() {
                                    info.mro.push(object_ir);
                                }
                            }
                        }
                    }
                }
            }

            let cached_mro = self.cache.get_class_by_type(cls_typ).unwrap().mro.clone();
            let mut root_mro = vec![];
            for t in cached_mro {
                root_mro.push(self.instantiate_type(t, Some(cls_typ))?);
            }
            mro.push(root_mro);
            asts.push(cls_typ);

            // Add hidden generics
            self.add_class_generics(cls_typ, false, false, false)?;
            let level = self.ctx.borrow().typecheck_level;
            let gens = types.generics(cls_typ);
            let hgens = types.hidden_generics(cls_typ);
            let mut new_gens = vec![];
            for mut g in gens {
                g.ty = Some(types.generalize(g.ty(), level));
                new_gens.push(g);
            }
            let mut new_hgens = vec![];
            for mut g in hgens {
                g.ty = Some(types.generalize(g.ty(), level));
                new_hgens.push(g);
            }
            types.with_class_mut(cls_typ, |c| {
                c.generics = new_gens.clone();
                c.hidden_generics = new_hgens.clone();
            });
            types.with_class_mut(typ, |c| {
                c.hidden_generics.extend(new_gens);
                c.hidden_generics.extend(new_hgens);
            });
        }

        if !asts.is_empty() || types.class_name(typ).is_some_and(|n| n == st.object.as_str()) {
            mro.push(asts.clone());
        }
        let merged = crate::cache::Cache::merge_c3(&types, &mut mro);
        self.cache.classes.get_mut(canonical).unwrap().mro = merged.clone();
        if merged.is_empty() {
            typhon_common::bail_e!(ErrorKind::CLASS_BAD_MRO, self.src);
        }
        let asts = merged; // use MRO order
        for m in &asts {
            let n = types.class_name(*m).unwrap().to_string();
            self.cache.classes.entry(n).or_default().descendants.insert(canonical.to_string());
        }

        // Resolve class names in C3 order so a sibling's override
        // also wins over an earlier base's inherited variable.
        for base in &asts {
            let bname = types.class_name(*base).unwrap().to_string();
            if bname == canonical {
                continue;
            }
            let mut cvs: Vec<(String, String)> =
                self.cache.classes[&bname].class_vars.iter().map(|(a, b)| (a.clone(), b.clone())).collect();
            cvs.sort();
            for (var_name, var_canonical) in cvs {
                if var_canonical != format!("{bname}.{var_name}")
                    || self.cache.classes[canonical].class_vars.contains_key(&var_name)
                {
                    continue;
                }
                let new_name = format!("{canonical}.{var_name}");
                let new_canonical = self.gen_canonical(&new_name, false, false);
                self.cache.classes.get_mut(canonical).unwrap().class_vars.insert(var_name.clone(), var_canonical.clone());
                let v = self.force_find(&var_canonical);
                self.ctx.borrow_mut().add(&new_name, v.clone());
                self.ctx.borrow_mut().add(&new_canonical, v);
            }
        }

        // Add normal fields
        for cls_typ in &asts {
            if types.is(*cls_typ, &types.class_name(typ).unwrap()) {
                continue;
            }
            let ct = types.get_class(*cls_typ).unwrap();
            let src = self.src;
            self.with_class_generics(ct, false, false, false, |v| {
                let mut ai = 0;
                let ast = v.cache.get_class_by_type(ct).and_then(|c| c.ast).unwrap();
                let (aname, items) = {
                    let c = v.cache.ast[ast].kind.as_class().unwrap();
                    (c.name.clone(), c.items.clone())
                };
                for a in &items {
                    if a.is_value() && !v.cache.ast.is_class_var(a) {
                        let (fbase, acls_ast_name) = {
                            let acls = &v.cache.classes[&aname];
                            let an = acls.ast.and_then(|x| v.cache.ast[x].kind.as_class().map(|c| c.name.clone())).unwrap_or_default();
                            (acls.fields[ai].base_class.clone(), an)
                        };
                        if fbase == acls_ast_name {
                            let name = a.name.clone();
                            if args.iter().any(|aa| aa.name == name) {
                                // A member auto-deduced from the subclass `__init__`
                                // (`self.axis = ...`) that is already an inherited field is
                                // just a re-assignment (Python semantics): keep the base field.
                                let deduced = v.cache.ast[attr].attrs.has(Attr::ClassDeduce);
                                if deduced {
                                    args.retain(|aa| aa.name != name);
                                    v.cache.classes.get_mut(canonical).unwrap().fields.retain(|f| f.name != name);
                                } else {
                                    let n = v.get_unmangled_name(&a.name);
                                    typhon_common::bail_e!(ErrorKind::CUSTOM, src, "field '{}' already defined", n);
                                }
                            }
                            let tt = v.cache.ast.clone_expr_opt(a.ty, true);
                            let ta = v.transform_type(tt, true)?;
                            let dd = v.cache.ast.clone_expr_opt(a.default_value, true);
                            let td = v.transform(dd)?;
                            args.push(v.cache.ast.new_param(Default::default(), name.clone(), ta, td, 0, ParamKind::Normal));
                            let ft = v.extract_type_expr(ta.unwrap());
                            v.cache.classes.get_mut(canonical).unwrap().fields.push(ClassField::new(name, Some(ft), fbase, None));
                        }
                        ai += 1;
                    }
                }
                Ok(true)
            })?;
        }
        Ok(asts)
    }
}
