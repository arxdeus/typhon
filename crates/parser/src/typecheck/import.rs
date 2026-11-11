//! Port of codon/parser/visitors/typecheck/import.cpp.

use typhon_common::fs::ImportFile;
use typhon_common::mangle::{self, stdlib};
use typhon_common::{CompilerError, ErrorKind, Result, SrcInfo};

use super::{TypeContext, TypecheckVisitor};
use crate::ast::{
    Attr, CommentStmt, ImportStmt, IndexExpr, ListExpr, Param, ParamKind, ReturnStmt, SuiteStmt, TupleExpr, UpdateMode,
};
use crate::cache::{MODULE_MAIN, STDLIB_IMPORT};
use crate::ids::{ExprId, StmtId};
use crate::visitors::format::FormatVisitor;

impl<'a> TypecheckVisitor<'a> {
    /// `visit(ImportStmt*)`: import and parse a new module into its own context; guard
    /// non-stdlib imports with a runtime `_import_N_call()`.
    pub fn visit_import(&mut self, s: StmtId) -> Result<()> {
        typhon_common::seqassert!(!self.ctx.borrow().in_class(), "imports within a class");
        if let Some(r) = self.transform_special_import(s)? {
            self.result_stmt = Some(r);
            return Ok(());
        }
        let st = self.cache.ast[s].kind.as_import().unwrap().clone();
        let ssrc = self.cache.ast[s].src;

        // Fetch the import
        let components = self.get_import_path(st.from, st.dots);
        let path = components.join("/");
        let relative_from = st.dots > 0 && st.from.is_none();
        let filename = self.ctx.borrow().filename();
        let file = super::helpers::get_import_file(self.cache, &path, &filename, false);
        let Some(file) = file else {
            if relative_from {
                let as_ = if st.as_.is_empty() { self.id_value(st.what) } else { st.as_.clone() };
                let n = self.n_stmt(ImportStmt { from: st.what, what: None, args: vec![], ret: None, as_, dots: st.dots, is_function: true });
                self.result_stmt = Some(self.tr_stmt(n)?);
                return Ok(());
            }
            let autopy = self.ctx.borrow().auto_python || self.cache.options.as_ref().is_some_and(|o| o.autopy);
            if st.dots == 0 && autopy {
                let from = st.from.expect("import without from");
                let mut new_str = FormatVisitor::apply_expr(&self.cache.ast, from, None, false);
                if let Some(w) = st.what {
                    new_str += &format!(".{}", FormatVisitor::apply_expr(&self.cache.ast, w, None, false));
                }
                typhon_common::diag::compilation_warning(
                    &format!("importing '{new_str}' from Python"),
                    ssrc.file.as_str(),
                    ssrc.line,
                    ssrc.col,
                    0,
                    0,
                    false,
                    Default::default(),
                );
                let fsrc = self.cache.ast[from].src;
                let e = self.parse_expr_via_hook(&new_str, fsrc)?;
                let py = self.xm_id("python");
                let n = self.n_stmt(ImportStmt { from: Some(py), what: Some(e), args: st.args.clone(), ret: st.ret, as_: st.as_.clone(), dots: 0, is_function: true });
                self.result_stmt = Some(self.tr_stmt(n)?);
                return Ok(());
            }

            let mut s2 = ".".repeat(st.dots);
            for c in &components {
                if c == ".." {
                    continue;
                } else if !s2.is_empty() && !s2.ends_with('.') {
                    s2 += &format!(".{c}");
                } else {
                    s2 += c;
                }
            }
            let all_dot = s2.chars().all(|c| c == '.');
            if all_dot && st.what.is_some_and(|w| self.cache.ast[w].kind.as_id().is_some()) {
                s2 = self.id_value(st.what);
            }
            typhon_common::bail_e!(ErrorKind::IMPORT_NO_MODULE, ssrc, s2);
        };

        // If the file has not been seen before, load it into cache
        let mut handled = true;
        if !self.cache.imports.contains_key(&file.path) {
            self.result_stmt = self.transform_new_import(&file)?;
            if self.result_stmt.is_none() {
                handled = false; // we need an import
            }
        }

        let (import_var, ictx, loaded_at_toplevel) = {
            let i = self.get_import(&file.path);
            (i.import_var.clone(), i.ctx.clone().expect("import without ctx"), i.loaded_at_toplevel)
        };
        // The main module has no runtime import variable.
        if ictx.borrow().module_info().module == MODULE_MAIN && !self.ctx.borrow().is_stdlib_loading {
            typhon_common::bail_e!(ErrorKind::CUSTOM, ssrc, "cannot import main module '{}'", file.module);
        }
        if !loaded_at_toplevel {
            handled = false;
        }

        if !handled {
            let f = self.xm_id(&mangle::mangled_func("", &format!("{import_var}_call"), 0, 0, true));
            let c = self.xm_call(f, &[]);
            self.result_stmt = Some(self.xm_expr_stmt(c));
            typhon_common::log_flag!(TYPECHECK, "[import] loading {}", import_var);
        }

        // Import requested identifiers from the import's scope to the current scope
        let what_id = st.what.and_then(|w| self.cache.ast[w].kind.as_id().map(|i| i.value.clone()));
        if st.what.is_none() {
            // Case: import foo
            let name = if st.as_.is_empty() { path.clone() } else { st.as_.clone() };
            let e = self.force_find(&import_var);
            self.ctx.borrow_mut().add(&name, e);
        } else if what_id.as_deref() == Some("*") {
            // Case: from foo import *
            typhon_common::seqassert!(st.as_.is_empty(), "renamed star-import");
            let stdlib_loading = self.ctx.borrow().is_stdlib_loading;
            let entries: Vec<_> = ictx.borrow().base.map.iter().filter_map(|(k, v)| v.first().map(|x| (k.clone(), x.clone()))).collect();
            for (i, ival) in entries {
                if !i.starts_with('_') || (stdlib_loading && i.starts_with("__")) {
                    let mut c = ival;
                    if c.borrow().is_conditional() && !i.contains('.') {
                        if let Some(x) = TypeContext::find_in(self.cache, &ictx, &i, 0, None) {
                            c = x;
                        }
                    }
                    // Imports should ignore noShadow property
                    self.ctx.borrow_mut().add(&i, c);
                }
            }
        } else {
            // Case 3: from foo import bar
            let Some(iv) = what_id else { panic!("not a valid import what expression") };
            let c = if relative_from {
                ictx.borrow().find_raw(&iv)
            } else {
                TypeContext::find_in(self.cache, &ictx, &iv, 0, None)
            };
            // Make sure that we are importing an existing global symbol
            if c.is_none() && relative_from {
                let p = if path.is_empty() { iv.clone() } else { format!("{path}/{iv}") };
                if super::helpers::get_import_file(self.cache, &p, &filename, false).is_some() {
                    // Support from . import int and similar commands
                    let as_ = if st.as_.is_empty() { iv.clone() } else { st.as_.clone() };
                    let ri = self.n_stmt(ImportStmt { from: st.what, what: None, args: vec![], ret: None, as_, dots: st.dots, is_function: true });
                    let su = self.xm_suite(&[self.result_stmt, Some(ri)]);
                    self.result_stmt = Some(self.tr_stmt(su)?);
                    return Ok(());
                }
            }
            let Some(mut c) = c else {
                let src = self.cache.ast[st.what.unwrap()].src;
                typhon_common::bail_e!(ErrorKind::IMPORT_NO_NAME, src, iv, file.module);
            };
            if c.borrow().is_conditional() {
                if let Some(x) = TypeContext::find_in(self.cache, &ictx, &iv, 0, None) {
                    c = x;
                }
            }
            let name = if st.as_.is_empty() { iv } else { st.as_.clone() };
            self.ctx.borrow_mut().add(&name, c);
        }
        let r = match self.result_stmt {
            Some(r) => r,
            None => self.n_stmt(SuiteStmt::default()), // erase it
        };
        self.result_stmt = Some(self.tr_stmt(r)?);
        Ok(())
    }

    fn id_value(&self, e: Option<ExprId>) -> String {
        e.and_then(|e| self.cache.ast[e].kind.as_id().map(|i| i.value.clone())).expect("expected IdExpr")
    }

    /// `parseExpr(cache, code, src)` through the `parse_code` hook.
    /// PORT-NOTE: `CacheHooks` has no expression parser; parse `code` as a one-line module
    /// and take the expression of its single `ExprStmt`.
    fn parse_expr_via_hook(&mut self, code: &str, src: SrcInfo) -> Result<ExprId> {
        let parse = self.cache.hooks.parse_code.clone().ok_or_else(|| CompilerError::custom("no parser hook installed", src))?;
        let file = src.file.as_str().to_string();
        let mut node = parse(self.cache, &file, code, src.line)?;
        loop {
            match &self.cache.ast[node].kind {
                crate::ast::StmtKind::Suite(su) if su.items.len() == 1 => node = su.items[0],
                crate::ast::StmtKind::Expr(e) if e.expr.is_some() => return Ok(e.expr.unwrap()),
                _ => return Err(CompilerError::custom(format!("cannot parse '{code}' as an expression"), src)),
            }
        }
    }
}

impl<'a> TypecheckVisitor<'a> {
    /// `transformSpecialImport(stmt)`: `from C import ...` and `from python import ...`.
    pub fn transform_special_import(&mut self, s: StmtId) -> Result<Option<StmtId>> {
        let st = self.cache.ast[s].kind.as_import().unwrap().clone();
        let Some(fi) = st.from.and_then(|f| self.cache.ast[f].kind.as_id().map(|i| i.value.clone())) else {
            return Ok(None);
        };
        if fi == "C" {
            let wi = st.what.and_then(|w| self.cache.ast[w].kind.as_id().map(|i| i.value.clone()));
            if let Some(wi) = &wi {
                if !st.is_cvar() {
                    // C function imports
                    return self.transform_c_import(wi, &st.args, st.ret, &st.as_);
                }
                // C variable imports
                return self.transform_c_var_import(wi, st.ret, &st.as_);
            } else if let Some(de) = st.what.and_then(|w| self.cache.ast[w].kind.as_dot().cloned()) {
                // dylib C imports
                return self.transform_cdll_import(de.expr.unwrap(), &de.member, &st.args, st.ret, &st.as_, !st.is_cvar());
            }
        } else if fi == "python" && st.what.is_some() {
            // Python imports
            return self.transform_python_import(st.what.unwrap(), &st.args, st.ret, &st.as_);
        }
        Ok(None)
    }

    /// `getImportPath(from, dots)`: `Dot(Dot(a, b), c)` -> `[a, b, c]` (+ `..` per extra dot).
    pub fn get_import_path(&self, from: Option<ExprId>, dots: usize) -> Vec<String> {
        let mut components = vec![];
        if let Some(mut f) = from {
            while let Some(d) = self.cache.ast[f].kind.as_dot() {
                components.push(d.member.clone());
                f = d.expr.expect("invalid import statement");
            }
            match self.cache.ast[f].kind.as_id() {
                Some(i) => components.push(i.value.clone()),
                None => panic!("invalid import statement"),
            }
        }
        // Handle dots (i.e., `..` in `from ..m import x`)
        for _ in 1..dots.max(1) {
            components.push("..".to_string());
        }
        components.reverse();
        components
    }

    /// `transformCImport(name, args, ret, alt)`: `from C import foo(int) -> float as f`
    /// -> `@.c def foo(a0: int) -> float: pass` (+ rename to `f`).
    pub fn transform_c_import(&mut self, name: &str, args: &[Param], ret: Option<ExprId>, alt: &str) -> Result<Option<StmtId>> {
        let mut fn_args = vec![];
        let mut has_var_args = false;
        for (ai, a) in args.iter().enumerate() {
            typhon_common::seqassert!(a.name.is_empty(), "unexpected argument name");
            typhon_common::seqassert!(a.default_value.is_none(), "unexpected default argument");
            let ty = a.ty.expect("missing type");
            if self.cache.ast[ty].kind.as_ellipsis().is_some() && ai + 1 == args.len() {
                // C VAR_ARGS support
                has_var_args = true;
                fn_args.push(self.cache.ast.new_param(self.src, "*args", None, None, 0, ParamKind::Normal));
            } else {
                let n = if a.name.is_empty() { format!("a{ai}") } else { a.name.clone() };
                let t = self.cache.ast.clone_expr(ty, false);
                fn_args.push(self.cache.ast.new_param(self.src, n, Some(t), None, 0, ParamKind::Normal));
            }
        }
        let _ = self.ctx.borrow().generate_canonical_name(self.cache, name, false, false); // avoid canonicalName == name
        let r = match ret {
            Some(r) => self.cache.ast.clone_expr(r, false),
            None => self.xm_id(&stdlib().none_type),
        };
        let f = self.cache.ast.new_function(self.src, name, Some(r), fn_args, None, vec![], false)?;
        let t = self.get_time();
        if t != 0 {
            self.cache.ast[f].attrs.set_int(Attr::ExprTime, t);
        }
        self.cache.ast[f].attrs.set(Attr::C);
        if has_var_args {
            self.cache.ast[f].attrs.set(Attr::CVarArg);
        }
        let f = self.tr_stmt(f)?; // Already in the preamble
        if !alt.is_empty() {
            let _v = self.find(alt, 0);
            let val = self.force_find(name);
            self.ctx.borrow_mut().add(alt, val);
            self.ctx.borrow_mut().remove(name);
        }
        Ok(Some(f))
    }

    /// `transformCVarImport(name, type, alt)`: `from C import foo: int as f` -> `f: int = "foo"`.
    pub fn transform_c_var_import(&mut self, name: &str, ty: Option<ExprId>, alt: &str) -> Result<Option<StmtId>> {
        let canonical = self.ctx.borrow().generate_canonical_name(self.cache, name, false, false);
        let t = ty.map(|t| self.cache.ast.clone_expr(t, false));
        let typ = self.transform_type(t, false)?.expect("missing C variable type");
        let cls = self.extract_class_type_expr(typ).expect("bad C variable type");
        let link = self.cache.types.new_link(cls);
        let time = self.get_time();
        let n = if alt.is_empty() { name } else { alt };
        let val = TypeContext::add_var(self.cache, &self.ctx, n, &canonical, link, time, SrcInfo::default());
        let l = self.xm_id(&canonical);
        let s = self.xm_assign(l, None, Some(typ), UpdateMode::Assign);
        self.cache.ast[l].attrs.set(Attr::ExprExternVar);
        let vt = val.borrow().ty();
        self.set_type(l, vt);
        self.set_done_expr(l);
        self.set_done_stmt(s);
        Ok(Some(s))
    }

    /// `transformCDLLImport(dylib, name, args, ret, alt, isFunction)`:
    /// `from C import lib.foo(int) -> float as f` -> `f = _dlsym_c(lib, "foo", Fn=Function[[int], float])`.
    pub fn transform_cdll_import(
        &mut self,
        dylib: ExprId,
        name: &str,
        args: &[Param],
        ret: Option<ExprId>,
        alt: &str,
        is_function: bool,
    ) -> Result<Option<StmtId>> {
        let ty = if is_function {
            let mut items = vec![];
            for a in args {
                typhon_common::seqassert!(a.name.is_empty(), "unexpected argument name");
                typhon_common::seqassert!(a.default_value.is_none(), "unexpected default argument");
                let t = a.ty.expect("missing type");
                items.push(self.cache.ast.clone_expr(t, false));
            }
            let l = self.n_expr(ListExpr { items });
            let r = match ret {
                Some(r) => self.cache.ast.clone_expr(r, false),
                None => self.xm_id(&stdlib().none_type),
            };
            let tu = self.n_expr(TupleExpr { items: vec![l, r] });
            let f = self.xm_id(&stdlib().function);
            Some(self.n_expr(IndexExpr { expr: Some(f), index: Some(tu) }))
        } else {
            ret.map(|r| self.cache.ast.clone_expr(r, false))
        };
        let c = self.cache.ast.clone_expr(dylib, false);
        let lhs = self.xm_id(if alt.is_empty() { name } else { alt });
        let f = self.xm_id(if is_function { "_dlsym_c" } else { "_dlsym" });
        let ns = self.xm_str(name);
        let items = vec![self.cache.ast.call_arg("", Some(c)), self.cache.ast.call_arg("", Some(ns)), self.cache.ast.call_arg("Fn", ty)];
        let call = self.xm_call_args(f, items);
        let a = self.xm_assign(lhs, Some(call), None, UpdateMode::Assign);
        Ok(Some(self.tr_stmt(a)?))
    }
}

impl<'a> TypecheckVisitor<'a> {
    /// `transformPythonImport(what, args, ret, alt)`:
    /// `from python import module as f` -> `f = pyobj._import("module")`;
    /// `from python import lib.foo(int) -> float as f` -> a wrapper function.
    pub fn transform_python_import(&mut self, what: ExprId, args: &[Param], ret: Option<ExprId>, alt: &str) -> Result<Option<StmtId>> {
        // Get a module name (e.g., os.path)
        let components = self.get_import_path(Some(what), 0);
        let last = components.last().cloned().unwrap_or_default();

        if ret.is_none() && args.is_empty() {
            // Simple import: `from python import foo.bar` -> `bar = pyobj._import("foo.bar")`
            let lhs = self.xm_id(if alt.is_empty() { &last } else { alt });
            let p = self.xm_id("pyobj");
            let d = self.xm_dot(p, "_import");
            let sname = self.xm_str(&components.join("."));
            let c = self.xm_call(d, &[sname]);
            let a = self.xm_assign(lhs, Some(c), None, UpdateMode::Assign);
            return Ok(Some(self.tr_stmt(a)?));
        }

        // f = pyobj._import("foo")._getattr("bar")
        let lhs = self.xm_id("f");
        let p = self.xm_id("pyobj");
        let d = self.xm_dot(p, "_import");
        let modname = components[..components.len().saturating_sub(1)].join(".");
        let ms = self.xm_str(&modname);
        let imp = self.xm_call(d, &[ms]);
        let ga = self.xm_dot(imp, "_getattr");
        let ls = self.xm_str(&last);
        let gac = self.xm_call(ga, &[ls]);
        let call = self.xm_assign(lhs, Some(gac), None, UpdateMode::Assign);
        // f(a1, ...)
        let mut params = vec![];
        let mut call_args = vec![];
        for (i, a) in args.iter().enumerate() {
            let t = a.ty.map(|t| self.cache.ast.clone_expr(t, false));
            params.push(self.cache.ast.new_param(self.src, format!("a{i}"), t, None, 0, ParamKind::Normal));
            call_args.push(self.xm_id(&format!("a{i}")));
        }
        // `return ret.__from_py__(f(a1, ...))`
        let ret_type = match ret {
            Some(r) if self.cache.ast[r].kind.as_none().is_none() => self.cache.ast.clone_expr(r, false),
            _ => self.xm_id(&stdlib().none_type),
        };
        let rt = self.cache.ast.clone_expr(ret_type, false);
        let fp = self.xm_dot(rt, "__from_py__");
        let f = self.xm_id("f");
        let fc = self.xm_call(f, &call_args);
        let fcp = self.xm_dot(fc, "p");
        let ret_expr = self.xm_call(fp, &[fcp]);
        let ret_stmt = self.n_stmt(ReturnStmt { expr: Some(ret_expr) });
        let body = self.xm_suite(&[Some(call), Some(ret_stmt)]);
        let name = if alt.is_empty() { last } else { alt.to_string() };
        let func = self.cache.ast.new_function(self.src, name, Some(ret_type), params, Some(body), vec![], false)?;
        let t = self.get_time();
        if t != 0 {
            self.cache.ast[func].attrs.set_int(Attr::ExprTime, t);
        }
        Ok(Some(self.tr_stmt(func)?))
    }

    /// `transformNewImport(file)`: parse a new file into its own context and wrap its
    /// top-level statements into `_import_N_call()` (except during stdlib loading).
    pub fn transform_new_import(&mut self, file: &ImportFile) -> Result<Option<StmtId>> {
        // Use a clean context to parse a new file
        let module_id = file.module.replace('.', "_");
        let ictx = {
            let mut c = TypeContext::new(self.cache, &file.path);
            c.is_stdlib_loading = self.ctx.borrow().is_stdlib_loading;
            c.set_module(file.clone());
            c.into_ref()
        };
        let cur_path = self.ctx.borrow().module_info().path.clone();
        let parent_toplevel = self.get_import(&cur_path).loaded_at_toplevel;
        let toplevel = {
            let c = self.ctx.borrow();
            parent_toplevel && (c.is_stdlib_loading || (c.is_global() && c.block_level == 0))
        };
        let import_var = self.get_temporary_var(&format!("import_{module_id}"));
        {
            let import = self.cache.imports.entry(file.path.clone()).or_insert_with(crate::cache::ImportModule::new);
            import.update(&file.module, &file.path, ictx.clone());
            import.loaded_at_toplevel = toplevel;
            import.import_var = import_var.clone();
        }
        typhon_common::log_flag!(REALIZE, "[import] initializing {} (location: {}, toplevel: {})", import_var, file.path, toplevel);

        // __name__ = [import name]
        let mut n = None;
        if file.module != "internal.core" {
            // str is not defined when loading internal.core; __name__ is not needed anyway
            let (mm, mp) = {
                let c = ictx.borrow();
                (c.module_info().module.clone(), c.module_info().path.clone())
            };
            let l = self.xm_id("__name__");
            let r = self.xm_str(&mm);
            let a1 = self.xm_assign(l, Some(r), None, UpdateMode::Assign);
            let l = self.xm_id("__file__");
            let r = self.xm_str(&mp);
            let a2 = self.xm_assign(l, Some(r), None, UpdateMode::Assign);
            n = Some(self.xm_suite(&[Some(a1), Some(a2)]));

            self.ctx.borrow_mut().add_block();
            let l = self.xm_id(&import_var);
            let f = self.xm_id("Import.__new__");
            let b = self.xm_bool(false);
            let ps = self.xm_str(&file.path);
            let ms = self.xm_str(&file.module);
            let c = self.xm_call(f, &[b, ps, ms]);
            let ty = self.xm_id("Import");
            let a = self.xm_assign(l, Some(c), Some(ty), UpdateMode::Assign);
            let a = self.tr_stmt(a);
            let a = match a {
                Ok(a) => a,
                Err(e) => {
                    self.ctx.borrow_mut().pop_block();
                    return Err(e);
                }
            };
            self.cache.ast.suite_add_stmt(self.preamble, Some(a));
            let val = self.force_find(&import_var);
            self.ctx.borrow_mut().pop_block();
            {
                let mut v = val.borrow_mut();
                v.block_level = 0;
                v.base_name = Default::default();
                v.module_name = ustr::Ustr::from(MODULE_MAIN);
                v.time = 0;
            }
            let sctx = self.get_import(STDLIB_IMPORT).ctx.clone().expect("stdlib not loaded");
            sctx.borrow_mut().add_toplevel(&import_var, val.clone());
            let vn = val.borrow().canonical_name.clone();
            self.register_global(&vn);
        }
        let parse = self.cache.hooks.parse_file.clone().ok_or_else(|| CompilerError::custom("no parser hook installed", self.src))?;
        let node = parse(self.cache, &file.path)?;
        let n = self.xm_suite(&[n, Some(node)]);
        if let Some(sc) = self.cache.hooks.scoping.clone() {
            // PORT-NOTE: `ScopingVisitor::apply(cache, n, &ictx->globalShadows, !isStdlibLoading)`;
            // the hook derives the last flag from the context it is given.
            sc(self.cache, n, Some(ictx.clone()))?;
        }
        if !self.cache.errors.is_empty() {
            return Err(CompilerError::Parser(self.cache.errors.clone()));
        }
        // Add comment to the top of import for easier dump inspection
        let comment = self.n_stmt(CommentStmt { comment: format!("import: {} at {}", file.module, file.path) });
        let suite = self.xm_suite(&[Some(comment), Some(n)]);

        let preamble = self.preamble;
        if self.ctx.borrow().is_stdlib_loading {
            // When loading the standard library, imports are not wrapped.
            let mut tv = TypecheckVisitor::new(self.cache, ictx, Some(preamble), None);
            return Ok(Some(tv.tr_stmt(suite)?));
        }
        // Generate import identifier
        let stmts = self.n_stmt(SuiteStmt::default());
        let ret = self.n_stmt(ReturnStmt { expr: None });
        self.cache.ast[ret].attrs.set(Attr::Internal); // do not trigger toplevel ReturnStmt error
        let iv = self.xm_id(&import_var);
        let ld = self.xm_dot(iv, "loaded");
        let i = self.xm_if(ld, Some(ret), None);
        self.cache.ast.suite_add_stmt(stmts, Some(i));
        let sl = self.xm_id("Import._set_loaded");
        let pf = self.xm_id("__ptr__");
        let iv = self.xm_id(&import_var);
        let pc = self.xm_call(pf, &[iv]);
        let slc = self.xm_call(sl, &[pc]);
        let es = self.xm_expr_stmt(slc);
        self.cache.ast.suite_add_stmt(stmts, Some(es));
        self.cache.ast.suite_add_stmt(stmts, Some(suite));

        // Wrap all imported top-level statements into a function.
        let fn_name = format!("{import_var}_call");
        let nt = self.xm_id(&stdlib().none_type);
        let f = self.cache.ast.new_function(self.src, fn_name.clone(), Some(nt), vec![], Some(stmts), vec![], false)?;
        let t = self.get_time();
        if t != 0 {
            self.cache.ast[f].attrs.set_int(Attr::ExprTime, t);
        }
        let f = {
            let mut tv = TypecheckVisitor::new(self.cache, ictx.clone(), Some(preamble), None);
            let f = tv.tr_stmt(f)?;
            let ft = tv.force_find(&fn_name).borrow().ty();
            tv.realize(ft)?;
            f
        };
        self.cache.ast.suite_add_stmt(preamble, Some(f));
        Ok(None)
    }
}
