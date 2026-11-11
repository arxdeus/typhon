//! Port of codon/parser/visitors/translate/translate_ctx.{h,cpp}.

use std::cell::RefCell;
use std::rc::Rc;

use typhon_ir::{FuncId, TypeId as IrTypeId, ValueId, VarId};

use crate::cache::Cache;
use crate::ctx::Context;
use crate::typecheck::ctx::TypeContext;
use crate::typecheck::TypecheckVisitor;

/// `TranslateItem`: an identifier bound to an IR function, type or variable.
/// PORT-NOTE: the C++ `base` pointer is never read by the translator; it is dropped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TranslateItem {
    Func(FuncId),
    Type(IrTypeId),
    Var(VarId),
}

impl TranslateItem {
    pub fn func(&self) -> Option<FuncId> {
        if let TranslateItem::Func(f) = self { Some(*f) } else { None }
    }
    pub fn ty(&self) -> Option<IrTypeId> {
        if let TranslateItem::Type(t) = self { Some(*t) } else { None }
    }
    pub fn var(&self) -> Option<VarId> {
        if let TranslateItem::Var(v) = self { Some(*v) } else { None }
    }
}

/// `TranslateContext`. The cache is passed explicitly to lookups (C++ stores `Cache*`).
pub struct TranslateContext {
    pub base: Context<TranslateItem>,
    /// Stack of function bases.
    pub bases: Vec<FuncId>,
    /// Stack of IR series (blocks).
    pub series: Vec<ValueId>,
    /// Stack of sequence items for attribute initialization: (attr discriminant, value).
    pub seq_items: Vec<Vec<(crate::ast::Attr, ValueId)>>,
    /// Function holding top-level code.
    pub main_func: Option<FuncId>,
    /// First user statement after the runtime preamble.
    pub main_user_start: Option<ValueId>,
}

pub type TranslateCtxRef = Rc<RefCell<TranslateContext>>;

impl Default for TranslateContext {
    fn default() -> Self {
        Self::new()
    }
}

impl TranslateContext {
    pub fn new() -> Self {
        Self { base: Context::new(""), bases: vec![], series: vec![], seq_items: vec![], main_func: None, main_user_start: None }
    }
    pub fn add(&mut self, name: &str, item: TranslateItem) {
        self.base.add(name, item);
    }
    pub fn get_base(&self) -> FuncId {
        *self.bases.last().expect("no function base")
    }
    pub fn get_series(&self) -> ValueId {
        *self.series.last().expect("no series")
    }
}

/// Fetch (or lazily create) the translate context stored in `Cache::codegen_ctx`.
pub fn codegen_ctx(cache: &mut Cache) -> TranslateCtxRef {
    if let Some(c) = cache.codegen_ctx.clone() {
        if let Ok(r) = c.downcast::<RefCell<TranslateContext>>() {
            return r;
        }
    }
    let r: TranslateCtxRef = Rc::new(RefCell::new(TranslateContext::new()));
    cache.codegen_ctx = Some(r.clone());
    r
}

/// The translate context if one was installed.
pub fn existing_codegen_ctx(cache: &Cache) -> Option<TranslateCtxRef> {
    cache.codegen_ctx.clone().and_then(|c| c.downcast::<RefCell<TranslateContext>>().ok())
}

/// `TranslateContext::find(name)`: local bindings first, then realized types/functions
/// from the root type-checking context.
pub fn find(cache: &mut Cache, ctx: &TranslateCtxRef, name: &str) -> Option<TranslateItem> {
    if let Some(t) = ctx.borrow().base.find(name) {
        return Some(t);
    }
    let type_ctx = cache.type_ctx.clone();
    let tt = TypeContext::find_in(cache, &type_ctx, name, 0, None)?;
    let (is_type, ty) = {
        let b = tt.borrow();
        (b.is_type(&cache.types), b.ty)
    };
    let ty = ty?;
    let types = cache.types.clone();
    if is_type && types.can_realize(ty) {
        let mut t = ty;
        if name != types.realized_name(t) {
            let tv = TypecheckVisitor::new(cache, type_ctx, None, None);
            t = tv.extract_type(t);
        }
        let c = types.get_class(t)?;
        let n = types.class_name(c)?;
        let r = cache.classes.get(n.as_str())?.realizations.get(name)?.clone();
        let ir = r.borrow().ir?;
        Some(TranslateItem::Type(ir))
    } else if let Some(f) = types.get_func(ty).filter(|_| types.can_realize(ty)) {
        let fname = types.func_name(f);
        let r = cache.functions.get(&fname).and_then(|f| f.realizations.get(name)).cloned();
        typhon_common::seqassert!(r.is_some(), "cannot find function realization {}", name);
        let ir = r?.borrow().ir?;
        Some(TranslateItem::Func(ir))
    } else {
        None
    }
}
