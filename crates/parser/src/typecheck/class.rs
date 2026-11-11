//! Port of codon/parser/visitors/typecheck/class.cpp (phase-3 lane: replace STUB bodies).
//! `generateTuple`/`generateKwId` are ported (core needs them).

use typhon_common::{ErrorKind, Result};
use typhon_common::mangle::stdlib;

use super::{TypecheckVisitor, TypeContext, not_ported};
use crate::ast::{Attr, ClassStmt, IdExpr, IndexExpr, Param, ParamStatus, SuiteStmt, TupleExpr};
use crate::cache::{MAX_TUPLE, STDLIB_IMPORT};
use crate::ids::{ExprId, StmtId, TypeId};
use crate::types::{Generic, LiteralKind};

impl<'a> TypecheckVisitor<'a> {
    /// `visit(ClassStmt*)`.
    pub fn visit_class(&mut self, _s: StmtId) -> Result<()> {
        Err(not_ported("class::visit_class", self.src)) // STUB(class.cpp)
    }
    /// `parseBaseClasses(baseClasses, args, attr, canonicalName, typ)`.
    pub fn parse_base_classes(
        &mut self,
        _bases: &mut Vec<ExprId>,
        _args: &mut Vec<Param>,
        _attr: StmtId,
        _canonical: &str,
        _typ: TypeId,
    ) -> Result<Vec<TypeId>> {
        Err(not_ported("class::parse_base_classes", self.src)) // STUB(class.cpp)
    }
    /// `autoDeduceMembers(stmt, args)`.
    pub fn auto_deduce_members(&mut self, _s: StmtId, _args: &mut Vec<Param>) -> Result<bool> {
        Err(not_ported("class::auto_deduce_members", self.src)) // STUB(class.cpp)
    }
    /// `checkFieldInitialization(fn, ast)`.
    pub fn check_field_initialization(&mut self, _f: TypeId, _ast: StmtId) -> Result<()> {
        Err(not_ported("class::check_field_initialization", self.src)) // STUB(class.cpp)
    }
    /// `prepareEnum(stmt, typ, clsStmts)`.
    pub fn prepare_enum(&mut self, _s: StmtId, _typ: TypeId, _cls_stmts: &mut Vec<StmtId>) -> Result<Vec<StmtId>> {
        Err(not_ported("class::prepare_enum", self.src)) // STUB(class.cpp)
    }
    /// `isEnum(typ)`.
    pub fn is_enum(&self, _typ: TypeId) -> bool {
        false // STUB(class.cpp)
    }
    /// `getClassMethods(stmt)` (static in C++).
    pub fn get_class_methods(&self, _s: StmtId) -> Vec<StmtId> {
        vec![] // STUB(class.cpp)
    }
    /// `transformNestedClasses(stmt, clsStmts, varStmts, fnStmts)`.
    pub fn transform_nested_classes(
        &mut self,
        _s: StmtId,
        _cls: &mut Vec<StmtId>,
        _vars: &mut Vec<StmtId>,
        _fns: &mut Vec<StmtId>,
    ) -> Result<()> {
        Err(not_ported("class::transform_nested_classes", self.src)) // STUB(class.cpp)
    }
    /// `codegenMagic(op, typExpr, allArgs, isRecord)`.
    pub fn codegen_magic(&mut self, _op: &str, _typ: ExprId, _args: &[Param], _is_record: bool) -> Result<Option<StmtId>> {
        Err(not_ported("class::codegen_magic", self.src)) // STUB(class.cpp)
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
