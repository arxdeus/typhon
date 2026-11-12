//! `TypeRealizer`: the IR -> parser cache coupling (replaces `ast::Cache*` in
//! the reference).

use std::any::Any;
use std::fmt;
use std::rc::Rc;

use typhon_common::Result;

use crate::ids::{FuncId, TypeId};
use crate::module::Module;
use crate::types::Generic;

/// Opaque handle to a parser-side type or function (`ast::types::TypePtr`).
/// Only the parser interprets it (via `downcast_ref`).
#[derive(Clone)]
pub struct AstTypeRef(pub Rc<dyn Any>);

impl AstTypeRef {
    pub fn new<T: Any>(v: T) -> Self {
        AstTypeRef(Rc::new(v))
    }
    pub fn downcast_ref<T: Any>(&self) -> Option<&T> {
        self.0.downcast_ref()
    }
    pub fn ptr_eq(&self, other: &AstTypeRef) -> bool {
        Rc::ptr_eq(&self.0, &other.0)
    }
}

impl fmt::Debug for AstTypeRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "AstTypeRef({:p})", Rc::as_ptr(&self.0))
    }
}

/// Everything the IR module and types need from `ast::Cache`.
///
/// Errors returned as `Err` correspond to the reference `exc::ParserException`; the module logs
/// them (`LOG_IR`) and returns `None`, exactly like the reference `try/catch`.
pub trait TypeRealizer {
    /// `cache->findClass(name)`.
    fn find_class(&mut self, name: &str) -> Option<AstTypeRef>;
    /// `cache->findFunction(name)`.
    fn find_function(&mut self, name: &str) -> Option<AstTypeRef>;
    /// `cache->findMethod(cls, method, argTypes)`; `args` are IR types whose
    /// `ast_type` the parser may read through `m.ty(id).ast_type`.
    fn find_method(&mut self, m: &Module, cls: &AstTypeRef, method: &str, args: &[TypeId]) -> Option<AstTypeRef>;
    /// `cache->realizeType(cls, generics)`.
    fn realize_type(&mut self, m: &mut Module, cls: &AstTypeRef, generics: &[Generic]) -> Result<Option<TypeId>>;
    /// `cache->realizeFunction(func, translateArgs(args), translateGenerics(generics), parentCls)`.
    fn realize_function(
        &mut self,
        m: &mut Module,
        func: &AstTypeRef,
        args: &[TypeId],
        generics: &[Generic],
        parent_cls: Option<&AstTypeRef>,
    ) -> Result<Option<FuncId>>;
    /// `cache->makeFunction([ret, args...])`.
    fn make_function(&mut self, m: &mut Module, ret: TypeId, args: &[TypeId]) -> Result<TypeId>;
    /// `cache->makeTuple(types)`.
    fn make_tuple(&mut self, m: &mut Module, args: &[TypeId]) -> Result<TypeId>;
    /// `cache->makeUnion(types)`.
    fn make_union(&mut self, m: &mut Module, types: &[TypeId]) -> Result<TypeId>;
    /// `cache->parseCode(code)`.
    fn parse_code(&mut self, m: &mut Module, code: &str) -> Result<()>;
    /// Generics extraction of `Type::doGetGenerics` (`func_generics == false`, class
    /// generics of `ast`) and `FuncType::doGetGenerics` (`func_generics == true`).
    /// Class-typed generics must be realized to IR types (`cache->realizeType`).
    fn generics(&mut self, m: &mut Module, ast: &AstTypeRef, func_generics: bool) -> Vec<Generic>;
}
