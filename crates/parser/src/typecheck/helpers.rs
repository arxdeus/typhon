//! Part 2: helpers (getters,
//! extract*, instantiate*, find*, ...). Call matching / wrapping live in
//! [`super::helpers_call`]; cythonization in [`super::helpers_py`].

use std::path::{Path, PathBuf};

use typhon_common::fs::ImportFile;
use typhon_common::mangle::stdlib;
use typhon_common::{CompilerError, ErrorKind, ErrorMessage, ParserErrors, Result, SrcInfo};

use super::{ItemRef, TypeContext, TypecheckVisitor};
use crate::ast::{IdExpr, IndexExpr, ExprTag};
use crate::cache::{Cache, ClassField, ClassRealizationRef, Class, Function, ImportModule, FN_DISPATCH_SUFFIX, STDLIB_IMPORT};
use crate::ids::{ExprId, StmtId, TypeId};
use crate::types::{Generic, InstantiationCache, LinkKind, LiteralKind, string_from_literal};

/// `getImportFile(cache, what, relativeTo, forceStdlib)`.
pub fn get_import_file(cache: &mut Cache, what: &str, relative_to: &str, force_stdlib: bool) -> Option<ImportFile> {
    // the reference `path.replace_extension(ext)`: `Path::with_extension` is a no-op on paths ending
    // in `..` (`from .. import x`), which would then match the parent *directory*.
    fn replace_ext(p: PathBuf, ext: &str) -> PathBuf {
        if p.file_name().is_some() {
            p.with_extension(ext)
        } else {
            let mut s = p.into_os_string();
            s.push(format!(".{ext}"));
            PathBuf::from(s)
        }
    }
    let fs = cache.fs.clone()?;
    let mut paths: Vec<String> = vec![];
    let parent = Path::new(relative_to).parent().map(|p| p.to_path_buf()).unwrap_or_default();
    let gen_ = |p: PathBuf| p.to_string_lossy().replace('\\', "/");
    if !force_stdlib {
        let f = fs.borrow();
        for p in [
            replace_ext(parent.join(what), "py"),
            parent.join(what).join("__init__.py"),
        ] {
            if f.exists(&p) {
                paths.push(gen_(f.canonical(&p)));
            }
        }
    }
    let sps: Vec<PathBuf> = fs.borrow().search_paths().to_vec();
    // Python `sys.path[0]`: the main script's directory is a search root for every import made
    // from user code (also from modules in subpackages), before the stdlib.
    if !force_stdlib {
        let f = fs.borrow();
        let rel = relative_to.replace('\\', "/");
        let from_stdlib = sps.iter().any(|p| rel.starts_with(&*p.to_string_lossy().replace('\\', "/")));
        let m0 = f.module0().parent().map(|p| p.to_path_buf()).unwrap_or_default();
        if !from_stdlib && !m0.as_os_str().is_empty() && m0 != parent {
            for c in [replace_ext(m0.join(what), "py"), m0.join(what).join("__init__.py")] {
                if f.exists(&c) {
                    paths.push(gen_(f.canonical(&c)));
                }
            }
        }
    }
    for p in sps {
        {
            let f = fs.borrow();
            for c in [replace_ext(p.join(what), "py"), p.join(what).join("__init__.py")] {
                if f.exists(&c) {
                    paths.push(gen_(f.canonical(&c)));
                }
            }
        }
    }
    let first = paths.first()?;
    Some(fs.borrow().root_of(Path::new(first)))
}

impl<'a> TypecheckVisitor<'a> {
    /// `getTime()`.
    pub fn get_time(&self) -> i64 {
        self.ctx.borrow().time
    }
    /// `ctx->find(name, time)`.
    pub fn find(&self, name: &str, time: i64) -> Option<ItemRef> {
        TypeContext::find_in(self.cache, &self.ctx, name, time, None)
    }
    /// `ctx->find(name, time, inBase)`.
    pub fn find_in_base(&self, name: &str, time: i64, base: &str) -> Option<ItemRef> {
        TypeContext::find_in(self.cache, &self.ctx, name, time, Some(base))
    }
    /// `ctx->forceFind(name)`.
    pub fn force_find(&self, name: &str) -> ItemRef {
        TypeContext::force_find_in(self.cache, &self.ctx, name)
    }
    /// `getImport(s)`.
    pub fn get_import(&self, s: &str) -> &ImportModule {
        self.cache.imports.get(s).expect("bad import")
    }
    /// `getClass(string)`.
    pub fn get_class_name(&self, t: &str) -> Option<&Class> {
        self.cache.classes.get(t)
    }
    /// `getClass(Type*)` (asserts a class).
    pub fn get_class_type(&self, t: TypeId) -> &Class {
        self.cache.get_class_by_type(t).expect("bad class")
    }
    pub fn get_class_type_mut(&mut self, t: TypeId) -> &mut Class {
        self.cache.get_class_by_type_mut(t).expect("bad class")
    }
    /// `getFunction(string)`.
    pub fn get_function_name(&self, n: &str) -> Option<&Function> {
        self.cache.functions.get(n)
    }
    /// `getFunction(Type*)`.
    pub fn get_function_type(&self, t: TypeId) -> Option<&Function> {
        let f = self.cache.types.get_func(t).expect("bad function");
        let n = self.cache.types.func_name(f);
        self.cache.functions.get(&n)
    }
    /// `getClassRealization(t)`.
    pub fn get_class_realization(&self, t: TypeId) -> Option<ClassRealizationRef> {
        let types = &self.cache.types;
        typhon_common::seqassert!(types.can_realize(t), "bad class");
        let c = types.get_class(t)?;
        let rn = types.realized_name(c);
        self.cache.get_class_by_type(t)?.realizations.get(&rn).cloned()
    }
    /// `getRootName(fn)`.
    pub fn get_root_name(&self, f: TypeId) -> String {
        let n = self.cache.types.func_name(f);
        self.cache.functions.get(&n).map(|f| f.root_name.clone()).unwrap_or_default()
    }
    /// `hasOverloads(root)`.
    pub fn has_overloads(&self, root: &str) -> bool {
        self.cache.overloads.get(root).is_some_and(|v| v.len() > 1)
    }
    /// `getOverloads(root)`.
    pub fn get_overloads(&self, root: &str) -> Vec<String> {
        self.cache.overloads.get(root).cloned().expect("bad root")
    }
    /// `getUnmangledName(s)`.
    pub fn get_unmangled_name(&self, s: &str) -> String {
        self.cache.types.unmangled(s)
    }
    /// `getUserFacingName(s)`.
    pub fn get_user_facing_name(&self, s: &str) -> String {
        let n = self.get_unmangled_name(s);
        n.strip_prefix('$').map(|x| x.to_string()).unwrap_or(n)
    }
    /// `getTemporaryVar(s)`.
    pub fn get_temporary_var(&mut self, s: &str) -> String {
        self.cache.get_temporary_var(s)
    }
    /// `registerGlobal(name)`.
    pub fn register_global(&mut self, name: &str) {
        self.cache.globals.entry(name.to_string()).or_insert(None);
    }
    /// `isCanonicalName(name)`.
    pub fn is_canonical_name(name: &str) -> bool {
        name.contains('.')
    }
    /// `isImportFn(s)`.
    pub fn is_import_fn(s: &str) -> bool {
        s.starts_with("%_import_")
    }
    /// `isDispatch(string)`.
    pub fn is_dispatch_name(s: &str) -> bool {
        s.ends_with(FN_DISPATCH_SUFFIX)
    }
    /// `isDispatch(FunctionStmt*)`.
    pub fn is_dispatch_ast(&self, ast: Option<StmtId>) -> bool {
        ast.and_then(|a| self.cache.ast[a].kind.as_function().map(|f| Self::is_dispatch_name(&f.name))).unwrap_or(false)
    }
    /// `isDispatch(Type*)`.
    pub fn is_dispatch(&self, f: TypeId) -> bool {
        self.cache.types.get_func(f).is_some_and(|f| Self::is_dispatch_name(&self.cache.types.func_name(f)))
    }
    /// `isUnbound(type)`.
    pub fn is_unbound(&self, t: TypeId) -> bool {
        self.cache.types.get_unbound(t).is_some()
    }
    /// `isUnbound(expr)`.
    pub fn is_unbound_expr(&self, e: ExprId) -> bool {
        self.is_unbound(self.ety(e))
    }
    /// `isTypeExpr(e)`.
    pub fn is_type_expr(cache: &Cache, e: ExprId) -> bool {
        cache.ast[e].ty.is_some_and(|t| cache.types.is(t, &stdlib().type_))
    }
    /// `isFunctionExpr(e, name)`.
    pub fn is_function_expr(&self, e: Option<ExprId>, name: &str) -> bool {
        let Some(f) = e.and_then(|e| self.expr_type(e)).and_then(|t| self.cache.types.get_func(t)) else { return false };
        name.is_empty() || self.cache.types.func_name(f) == name
    }
    /// `isMetaclassType(type)`.
    pub fn is_metaclass_type(&self, t: TypeId) -> bool {
        self.cache.types.is(t, &stdlib().type_) && self.cache.types.generics_len(t) == 0
    }
    /// `getHeadExpr(e)`.
    pub fn get_head_expr(&self, mut e: ExprId) -> ExprId {
        while let Some(s) = self.cache.ast[e].kind.as_stmt() {
            e = s.expr.expect("StmtExpr without expr");
        }
        e
    }
    /// `hasSideEffect(e)`.
    pub fn has_side_effect(&self, e: ExprId) -> bool {
        let a = &self.cache.ast;
        match a[e].kind.tag() {
            ExprTag::Id | ExprTag::Instantiate | ExprTag::Bool | ExprTag::Int | ExprTag::String | ExprTag::None | ExprTag::Float
            | ExprTag::Ellipsis | ExprTag::Yield => false,
            ExprTag::Dot => !a[e].kind.as_dot().unwrap().expr.is_some_and(|x| a[x].kind.tag() == ExprTag::Id),
            _ => true,
        }
    }

    /// `extractClassGeneric(t, idx)`.
    pub fn extract_class_generic(&self, t: TypeId, idx: usize) -> TypeId {
        self.cache.types.generic_type(t, idx).expect("bad class")
    }
    /// `extractFuncGeneric(t, idx)`.
    pub fn extract_func_generic(&self, t: TypeId, idx: usize) -> TypeId {
        self.cache.types.func_generics(t)[idx].ty()
    }
    /// `extractFuncArgType(t, idx)`.
    pub fn extract_func_arg_type(&self, t: TypeId, idx: usize) -> TypeId {
        let g0 = self.extract_class_generic(t, 0);
        self.extract_class_generic(g0, idx)
    }
    /// `extractType(Type*)`: unwrap `type[T]` layers.
    pub fn extract_type(&self, mut t: TypeId) -> TypeId {
        while self.cache.types.is(t, &stdlib().type_) && !self.is_metaclass_type(t) {
            t = self.extract_class_generic(t, 0);
        }
        t
    }
    /// `extractType(Expr*)`.
    pub fn extract_type_expr(&self, e: ExprId) -> TypeId {
        let a = &self.cache.ast;
        let st = stdlib();
        if crate::ast::is_id(a, Some(e), &st.type_) {
            return self.ety(e);
        }
        if let Some(i) = a[e].kind.as_instantiate() {
            if crate::ast::is_id(a, i.expr, &st.type_) {
                return self.ety(e);
            }
        }
        self.extract_type(self.ety(e))
    }
    /// `extractType(string)`.
    pub fn extract_type_name(&self, s: &str) -> TypeId {
        let c = self.force_find(s).borrow().ty();
        if s == stdlib().type_ { c } else { self.extract_type(c) }
    }
    /// `extractClassType(Expr*)`.
    pub fn extract_class_type_expr(&self, e: ExprId) -> Option<TypeId> {
        self.cache.types.get_class(self.extract_type_expr(e))
    }
    /// `extractClassType(Type*)`.
    pub fn extract_class_type(&self, t: TypeId) -> Option<TypeId> {
        self.cache.types.get_class(self.extract_type(t))
    }
    /// `extractClassType(string)`.
    pub fn extract_class_type_name(&self, s: &str) -> Option<TypeId> {
        self.cache.types.get_class(self.extract_type_name(s))
    }
    /// `extractFunction(t)`: function of a FuncType or a Partial.
    pub fn extract_function(&self, t: TypeId) -> Option<TypeId> {
        let types = &self.cache.types;
        types.get_func(t).or_else(|| types.get_partial(t).map(|p| types.partial_func(p)))
    }

    /// `getStdLibType(name)`.
    pub fn get_stdlib_type(&self, ty: &str) -> Result<TypeId> {
        let ctx = self.get_import(STDLIB_IMPORT).ctx.clone().expect("stdlib not loaded");
        let t = TypeContext::force_find_in(self.cache, &ctx, ty).borrow().ty();
        if ty == stdlib().type_ {
            return Ok(self.cache.types.get_class(t).unwrap());
        }
        self.extract_class_type(t).ok_or_else(|| CompilerError::custom(format!("not a class: {ty}"), self.src))
    }
    /// `getMetaclassType()`: `type` without the `type[T]` generic.
    pub fn get_metaclass_type(&self) -> Result<TypeId> {
        let t = self.get_stdlib_type(&stdlib().type_)?;
        let m = self.cache.types.copy_class(t);
        self.cache.types.with_class_mut(m, |c| c.generics.clear());
        Ok(m)
    }
    /// `getUnderlyingStaticType(t)`.
    pub fn get_underlying_static_type(&self, t: TypeId) -> Result<TypeId> {
        let types = &self.cache.types;
        if types.get_static(t).is_some() {
            return types.non_static_type(t).ok_or_else(|| CompilerError::custom("no non-static type", self.src));
        }
        let k = types.static_kind(t);
        if k != LiteralKind::Runtime {
            return self.get_stdlib_type(string_from_literal(k));
        }
        Ok(t)
    }
    /// `getSelfType(t)`.
    pub fn get_self_type(&self, t: TypeId) -> Result<Option<TypeId>> {
        let u = self.get_underlying_static_type(t)?;
        let Some(r) = self.cache.types.get_class(u) else { return Ok(None) };
        if self.cache.types.get_func(r).is_some() {
            return Ok(Some(self.cache.types.copy_class(r)));
        }
        Ok(Some(r))
    }
    /// `getStrLiteral(t, pos)`.
    pub fn get_str_literal(&self, t: TypeId, pos: usize) -> String {
        let types = &self.cache.types;
        types.get_str_static(t).unwrap_or_else(|| types.get_str_static(self.extract_class_generic(t, pos)).expect("not a string literal"))
    }
    /// `getIntLiteral(t, pos)`.
    pub fn get_int_literal(&self, t: TypeId, pos: usize) -> i64 {
        let types = &self.cache.types;
        types.get_int_static(t).unwrap_or_else(|| types.get_int_static(self.extract_class_generic(t, pos)).expect("not a int literal"))
    }
    /// `getBoolLiteral(t, pos)`.
    pub fn get_bool_literal(&self, t: TypeId, pos: usize) -> bool {
        let types = &self.cache.types;
        types.get_bool_static(t).unwrap_or_else(|| types.get_bool_static(self.extract_class_generic(t, pos)).expect("not a bool literal"))
    }
    /// `getParamType(t)`: annotation expression for generic params.
    pub fn get_param_type(&mut self, t: Option<TypeId>) -> Option<ExprId> {
        let t = t?;
        if self.cache.types.is(t, &stdlib().type_) {
            return Some(self.n_expr(IdExpr::new(stdlib().type_.clone())));
        }
        let k = self.cache.types.static_kind(t);
        if k != LiteralKind::Runtime {
            let l = self.n_expr(IdExpr::new("Literal"));
            let s = self.n_expr(IdExpr::new(string_from_literal(k)));
            return Some(self.n_expr(IndexExpr { expr: Some(l), index: Some(s) }));
        }
        None
    }
    /// `getClassMethod(typ, member)`.
    pub fn get_class_method(&self, t: TypeId, member: &str) -> String {
        self.cache.get_method(t, member)
    }
    /// `getClassFields(t)` (Tuple: only the first `generics.len()` fields).
    pub fn get_class_fields(&self, t: TypeId) -> Vec<ClassField> {
        let types = &self.cache.types;
        let name = types.class_name(t).unwrap();
        let f = &self.cache.classes.get(name.as_str()).expect("bad class").fields;
        if name == stdlib().tuple.as_str() {
            return f[..types.generics_len(t).min(f.len())].to_vec();
        }
        f.clone()
    }
    /// `findMember(type, member)`.
    pub fn find_member(&self, t: TypeId, member: &str) -> Option<ClassField> {
        let types = &self.cache.types;
        let cls = self.cache.get_class_by_type(t)?;
        for pc in &cls.mro {
            let mc = self.cache.get_class_by_type(*pc)?;
            for (i, mm) in mc.fields.iter().enumerate() {
                if types.is(*pc, &stdlib().tuple) && i >= types.generics_len(t) {
                    break;
                }
                if mm.name == member {
                    return Some(mm.clone());
                }
            }
        }
        None
    }
    /// `isPolymorphic(fn)`.
    pub fn is_polymorphic(&self, f: TypeId) -> bool {
        let types = &self.cache.types;
        let ast = types.func_ast(types.func_ast_id(f));
        let Some(first) = ast.params.first() else { return false };
        if types.func_size(f) == 0 || self.get_unmangled_name(&first.0) != "self" {
            return false;
        }
        let slf = types.get_class(self.extract_func_arg_type(f, 0)).unwrap();
        let Some(cls) = self.cache.get_class_by_type(slf) else { return false };
        let member = self.get_unmangled_name(&ast.name);
        let mut virt = cls.virtuals.contains(&member);
        let aid = types.func_ast_id(f);
        let fattrs = &self.cache.ast[aid].attrs;
        if types.func_generics(f).iter().any(|g| self.get_unmangled_name(&g.name) == "Self") {
            if let Some(p) = fattrs.get_str(crate::ast::Attr::ParentClass) {
                virt |= self.cache.classes.get(p).is_some_and(|c| c.virtuals.contains(&member));
            }
        }
        let cname = cls.ast.and_then(|a| self.cache.ast[a].kind.as_class().map(|c| c.name.clone())).unwrap_or_default();
        cls.has_rtti(&cname)
            && virt
            && !self.is_dispatch(f)
            && !fattrs.has(crate::ast::Attr::StaticMethod)
            && !fattrs.has(crate::ast::Attr::Property)
    }

    /// `findTypecheckErrors(n)`: "cannot typecheck" errors for every not-done node.
    pub fn find_typecheck_errors(&self, n: StmtId) -> ParserErrors {
        use crate::ast::Child;
        use crate::visitors::visitor::{expr_children, stmt_children};
        let a = &self.cache.ast;
        let mut res: Vec<ExprId> = vec![];
        // SearchVisitor: descend into not-done statements; collect top-most not-done exprs.
        fn walk_stmt(a: &crate::ast::AstArena, s: StmtId, res: &mut Vec<ExprId>) {
            if a[s].done {
                return;
            }
            for c in stmt_children(a, s) {
                walk_child(a, c, res);
            }
        }
        fn walk_expr(a: &crate::ast::AstArena, e: ExprId, res: &mut Vec<ExprId>) {
            if !a[e].done {
                res.push(e);
                return;
            }
            for c in expr_children(a, e) {
                walk_child(a, c, res);
            }
        }
        fn walk_child(a: &crate::ast::AstArena, c: Child, res: &mut Vec<ExprId>) {
            match c {
                Child::Expr(e) => walk_expr(a, e, res),
                Child::Stmt(s) => walk_stmt(a, s, res),
                #[allow(unreachable_patterns)]
                _ => {}
            }
        }
        walk_stmt(a, n, &mut res);
        let errors = res
            .into_iter()
            .map(|e| {
                let src = a[e].src;
                let code = self.cache.content(&src);
                // Class instance with generics that nothing could deduce (e.g. `C(1.0)`
                // for `class C[F, N: Literal[int]]`): name them.
                let types = &self.cache.types;
                let hint = a[e]
                    .ty
                    .and_then(|t| types.get_class(t))
                    .filter(|c| types.get_func(*c).is_none())
                    .map(|c| {
                        let missing: Vec<String> = types
                            .generics(c)
                            .iter()
                            .filter(|g| g.ty.is_none_or(|t| types.get_unbound(t).is_some()))
                            .map(|g| format!("'{}'", self.get_user_facing_name(g.name.as_str())))
                            .collect();
                        (types.class_name(c).map(|n| self.get_user_facing_name(n.as_str())).unwrap_or_default(), missing)
                    })
                    .filter(|(_, m)| !m.is_empty())
                    .map(|(n, m)| format!(" (cannot deduce generic {} of '{n}'; pass it explicitly, e.g. {n}[...](...))", m.join(", ")))
                    .unwrap_or_default();
                if !code.is_empty() {
                    ErrorMessage::new(format!("cannot typecheck '{code}'{hint}"), src)
                } else {
                    ErrorMessage::new(format!("cannot typecheck the expression{hint}"), src)
                }
            })
            .collect();
        ParserErrors::from_messages(errors)
    }

    // ------------------------------------------------------------------
    // instantiation
    // ------------------------------------------------------------------

    /// `instantiateUnbound(src, level)`.
    pub fn instantiate_unbound_at(&self, src: SrcInfo, level: i32) -> TypeId {
        let types = &self.cache.types;
        types.new_unbound(types.next_unbound_id(), level, src)
    }
    /// `instantiateUnbound()` (visitor src, current level).
    pub fn instantiate_unbound(&self) -> TypeId {
        let l = self.ctx.borrow().typecheck_level;
        self.instantiate_unbound_at(self.src, l)
    }
    /// `instantiateTypeVar(t)`: `type[t]`.
    pub fn instantiate_type_var(&self, t: TypeId) -> Result<TypeId> {
        let ty = self.get_stdlib_type(&stdlib().type_)?;
        self.instantiate_type_with(ty, &[t])
    }

    /// `instantiateType(src, type, generics)`: instantiate with receiver generics bound.
    pub fn instantiate_type(&self, t: TypeId, generics: Option<TypeId>) -> Result<TypeId> {
        let types = self.cache.types.clone();
        let mut cache = InstantiationCache::default();
        if let Some(g) = generics {
            let g = types.get_class(self.get_underlying_static_type(g)?).unwrap();
            let not_generic = |t: TypeId| !types.link(t).is_some_and(|l| l.kind == LinkKind::Generic);
            for gg in types.hidden_generics(g).into_iter().chain(types.generics(g)) {
                if let Some(gt) = gg.ty {
                    if not_generic(gt) {
                        cache.insert(gg.id, gt);
                    }
                }
            }
            // Self binds to the static receiver.
            if let Some(f) = types.get_func(t) {
                for fg in types.func_generics(f) {
                    if self.get_unmangled_name(&fg.name) == "Self" {
                        if let Some(s) = self.get_self_type(g)? {
                            cache.insert(fg.id, s);
                        }
                    }
                }
            }
        }
        let level = self.ctx.borrow().typecheck_level;
        let r = types.instantiate(t, level, Some(&mut cache));
        for (_, val) in cache {
            if let Some(l) = types.link(val) {
                types.set_src(val, self.src);
                if l.default_type.is_some() {
                    self.ctx.borrow_mut().get_base().pending_defaults.entry(0).or_default().insert(val);
                }
            }
        }
        Ok(r)
    }
    /// `instantiateType(src, root, generics vector)`.
    pub fn instantiate_type_with(&self, root: TypeId, generics: &[TypeId]) -> Result<TypeId> {
        let types = &self.cache.types;
        let c = types.get_class(root).expect("root class is null");
        let cg = types.generics(c);
        if generics.len() != cg.len() {
            let n = self.get_user_facing_name(&types.class_name(c).unwrap());
            typhon_common::bail_e!(ErrorKind::GENERICS_MISMATCH, self.src, n, cg.len(), generics.len());
        }
        let mut g = vec![];
        for (i, cgi) in cg.iter().enumerate() {
            let mut t = generics[i];
            if cgi.static_kind == LiteralKind::Runtime && types.get_static(t).is_some() {
                t = types.non_static_type(t).unwrap_or(t);
            }
            g.push(Generic::new("", Some(t), cgi.id, cgi.static_kind));
        }
        let dummy = types.new_class("", g, vec![]);
        self.instantiate_type(root, Some(dummy))
    }
}
