//! Port of codon/parser/visitors/translate/{translate,translate_ctx}.{h,cpp} (translate lane).
//!
//! `TranslateVisitor` lowers the typechecked AST into `typhon_ir` nodes. Entry points:
//! * [`TranslateVisitor::apply`]: build `main` + globals from the fully typechecked program;
//! * [`TranslateVisitor::translate_stmts`]: translate pending realizations (`Cache` hooks);
//! * [`install_hooks`]: wire `Cache::hooks.translate` / `translate_parsed`.

pub mod ctx;
mod expr;
pub mod func;
mod stmt;

use std::cell::RefCell;
use std::rc::Rc;

use typhon_common::mangle::{self, stdlib};
use typhon_common::{CompilerError, Result, SrcInfo};
use typhon_ir::{FuncId, Module, ValueId, ValueKind};

use crate::ast::{Attr, ExprId, StmtId};
use crate::cache::{Cache, var_argc, var_argv};
use crate::ids::TypeId;
use crate::typecheck::ctx::TypeContext;
use crate::typecheck::TypecheckVisitor;

pub use func::rebody_forward;
pub use ctx::{TranslateContext, TranslateCtxRef, TranslateItem, codegen_ctx, existing_codegen_ctx};

/// `TranslateVisitor`.
pub struct TranslateVisitor<'a> {
    pub cache: &'a mut Cache,
    pub ctx: TranslateCtxRef,
    pub module: Rc<RefCell<Module>>,
    /// `getSrcInfo()` of the node being translated.
    pub src: SrcInfo,
}

/// Install the translation hooks into the cache (`Cache::hooks.translate*`).
pub fn install_hooks(cache: &mut Cache) {
    cache.hooks.translate = Some(Rc::new(|cache: &mut Cache, s: StmtId| {
        let mut v = TranslateVisitor::new(cache)?;
        v.translate_stmts(Some(s))
    }));
    cache.hooks.translate_parsed = Some(Rc::new(translate_parsed));
}

/// IR part of `Cache::parseCode`: translate `node` into a fresh `parsed` series attached to
/// main right before the first user statement.
pub fn translate_parsed(cache: &mut Cache, node: StmtId) -> Result<ValueId> {
    let mut v = TranslateVisitor::new(cache)?;
    let block = v.module.borrow_mut().nr_named(ValueKind::SeriesFlow(Default::default()), "parsed");
    let (old_series, old_bases, main) = {
        let c = v.ctx.borrow();
        (c.series.clone(), c.bases.clone(), c.main_func)
    };
    let main = main.ok_or_else(|| CompilerError::custom("main function not translated", SrcInfo::default()))?;
    {
        let mut c = v.ctx.borrow_mut();
        c.series = vec![block];
        c.bases = vec![main];
    }
    let r = v.translate_stmts(Some(node));
    {
        let mut c = v.ctx.borrow_mut();
        c.series = old_series;
        c.bases = old_bases;
    }
    r?;
    let start = v.ctx.borrow().main_user_start;
    let mut m = v.module.borrow_mut();
    if m.as_series_flow(block).is_some_and(|s| !s.series.is_empty()) {
        if let Some(body) = m.func_body(main) {
            let pos = m
                .as_series_flow(body)
                .map(|s| start.and_then(|st| s.series.iter().position(|x| *x == st)).unwrap_or(s.series.len()))
                .unwrap_or(0);
            m.series_insert(body, pos, block);
        }
    }
    Ok(block)
}

impl<'a> TranslateVisitor<'a> {
    pub fn new(cache: &'a mut Cache) -> Result<Self> {
        let module = cache.module.clone().ok_or_else(|| CompilerError::custom("cache's module is not set", SrcInfo::default()))?;
        let ctx = codegen_ctx(cache);
        Ok(Self { cache, ctx, module, src: SrcInfo::default() })
    }

    /// `TranslateVisitor::apply(cache, stmts)`: returns the main (or JIT cell) function.
    pub fn apply(cache: &mut Cache, stmts: StmtId) -> Result<FuncId> {
        let jit = cache.options.as_ref().is_some_and(|o| o.jit);
        let none = cache
            .classes
            .get("NoneType")
            .and_then(|c| c.realizations.get("NoneType"))
            .and_then(|r| r.borrow().ir)
            .ok_or_else(|| CompilerError::custom("NoneType is not realized", SrcInfo::default()))?;
        let module = cache.module.clone().ok_or_else(|| CompilerError::custom("cache's module is not set", SrcInfo::default()))?;
        let main = {
            let mut m = module.borrow_mut();
            if jit {
                let fn_name = format!("_jit_{}", cache.jit_cell);
                let main = m.nr_bodied_func(fn_name.clone());
                m.set_src_info(main.into(), SrcInfo::new("<jit>", 0, 0, 0));
                m.set_global(main, true);
                let t = m.unsafe_get_func_type(&fn_name, none, vec![], false);
                m.realize_func(main, t, &[]);
                if let Some(fd) = m.func_mut(main) {
                    if let typhon_ir::FuncKind::Bodied { jit, .. } = &mut fd.kind {
                        *jit = true;
                    }
                }
                main
            } else {
                let main = m.main_func();
                let t = m.unsafe_get_func_type("<internal_func_type>", none, vec![], false);
                m.realize_func(main, t, &[]);
                let path = cache.fs.as_ref().map(|fs| fs.borrow().module0().to_string_lossy().to_string()).unwrap_or_default();
                m.set_src_info(main.into(), SrcInfo::new(&path, 0, 0, 0));
                main
            }
        };
        let block = module.borrow_mut().nr_named(ValueKind::SeriesFlow(Default::default()), "body");
        module.borrow_mut().set_func_body(main, Some(block));

        let ctx = codegen_ctx(cache);
        {
            let mut c = ctx.borrow_mut();
            c.bases = vec![main];
            c.series = vec![block];
        }
        cache.populate_python_module()?;
        {
            let mut c = ctx.borrow_mut();
            c.main_func = Some(main);
            c.main_user_start = None;
        }
        let mut v = TranslateVisitor::new(cache)?;
        v.initialize_globals()?;
        let suite = v.cache.ast[stmts].kind.as_suite().map(|s| s.items.clone());
        let has_preamble = !jit && suite.as_ref().is_some_and(|s| !s.is_empty());
        let block_len = |v: &TranslateVisitor| v.module.borrow().as_series_flow(block).map(|s| s.series.len()).unwrap_or(0);
        if has_preamble {
            v.transform_stmt(suite.as_ref().unwrap()[0])?;
        }
        let before = block_len(&v);
        if has_preamble {
            for s in &suite.as_ref().unwrap()[1..] {
                v.transform_stmt(*s)?;
            }
        } else {
            v.transform_stmt(stmts)?;
        }
        if block_len(&v) > before {
            let first = v.module.borrow().as_series_flow(block).map(|s| s.series[before]);
            v.ctx.borrow_mut().main_user_start = first;
        }
        v.translate_all_functions()?;
        Ok(main)
    }

    /// Translate every function AST in the cache (only pending realizations emit code).
    fn translate_all_functions(&mut self) -> Result<()> {
        let asts: Vec<StmtId> = self.cache.functions.values().filter_map(|f| f.ast).collect();
        for a in asts {
            self.transform_stmt(a)?;
        }
        Ok(())
    }

    /// `translateStmts(stmts)`.
    pub fn translate_stmts(&mut self, stmts: Option<StmtId>) -> Result<()> {
        self.initialize_globals()?;
        if let Some(s) = stmts {
            self.transform_stmt(s)?;
        }
        self.translate_all_functions()
    }

    /// `initializeGlobals()`.
    pub fn initialize_globals(&mut self) -> Result<()> {
        let names: Vec<String> = self.cache.globals.iter().filter(|(_, v)| v.is_none()).map(|(k, _)| k.clone()).collect();
        let types = self.cache.types.clone();
        for name in names {
            let type_ctx = self.cache.type_ctx.clone();
            let mut vt = None;
            let t = TypeContext::find_in(self.cache, &type_ctx, &name, 0, None).and_then(|i| i.borrow().ty);
            if let Some(t) = t {
                if !types.is_instantiated(t) || types.is(t, &stdlib().type_) || types.get_func(t).is_some() {
                    continue;
                }
                vt = Some(self.get_type(t)?);
            }
            let ir = {
                let mut m = self.module.borrow_mut();
                if name == var_argv() {
                    m.argv_var()
                } else if name == var_argc() {
                    m.argc_var()
                } else {
                    m.n_var(SrcInfo::default(), vt, true, false, false, name.clone())
                }
            };
            self.cache.globals.insert(name.clone(), Some(ir));
            self.ctx.borrow_mut().add(&name, TranslateItem::Var(ir));
        }
        Ok(())
    }

    // ------------------------------------------------------------------
    // helpers
    // ------------------------------------------------------------------

    pub(crate) fn find(&mut self, name: &str) -> Option<TranslateItem> {
        ctx::find(self.cache, &self.ctx, name)
    }
    pub(crate) fn force_find(&mut self, name: &str) -> Result<TranslateItem> {
        self.find(name).ok_or_else(|| CompilerError::custom(format!("cannot find '{name}'"), self.src))
    }

    /// `getType(t)`: IR type of a realized class type.
    pub(crate) fn get_type(&mut self, t: TypeId) -> Result<typhon_ir::TypeId> {
        let types = self.cache.types.clone();
        let c = types
            .get_class(t)
            .ok_or_else(|| CompilerError::custom(format!("not a class: {}", types.debug_string(t, 2)), self.src))?;
        let name = types.class_realized_name(c);
        match self.find(&name).and_then(|i| i.ty()) {
            Some(t) => Ok(t),
            None => Err(CompilerError::custom(format!("type {} not IR-realized: {}", types.debug_string(t, 2), name), self.src)),
        }
    }

    pub(crate) fn extract_type(&mut self, t: TypeId) -> TypeId {
        let tc = self.cache.type_ctx.clone();
        TypecheckVisitor::new(self.cache, tc, None, None).extract_type(t)
    }

    pub(crate) fn ety(&self, e: ExprId) -> Result<TypeId> {
        self.cache.ast[e].ty.ok_or_else(|| CompilerError::custom("expression has no type", self.cache.ast[e].src))
    }

    /// `make<T>(node, ...)`.
    pub(crate) fn make(&self, src: SrcInfo, kind: ValueKind) -> ValueId {
        self.module.borrow_mut().n(src, kind)
    }
    pub(crate) fn make_named(&self, src: SrcInfo, kind: ValueKind, name: &str) -> ValueId {
        self.module.borrow_mut().n_named(src, kind, name)
    }
    pub(crate) fn new_series(&self, src: SrcInfo, name: &str) -> ValueId {
        self.make_named(src, ValueKind::SeriesFlow(Default::default()), name)
    }
    pub(crate) fn push_series(&self, v: ValueId) {
        let s = self.ctx.borrow().get_series();
        self.module.borrow_mut().series_push(s, v);
    }
    pub(crate) fn none_new_name() -> String {
        mangle::mangled_method("", "NoneType", "__new__", 0, 0)
    }
    pub(crate) fn has_attr_e(&self, e: ExprId, a: Attr) -> bool {
        self.cache.ast[e].attrs.has(a)
    }
}
