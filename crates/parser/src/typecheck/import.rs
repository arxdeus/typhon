//! Port of codon/parser/visitors/typecheck/import.cpp (phase-3 lane: replace STUB bodies).

use typhon_common::Result;
use typhon_common::fs::ImportFile;

use super::{TypecheckVisitor, not_ported};
use crate::ast::Param;
use crate::ids::{ExprId, StmtId};

impl<'a> TypecheckVisitor<'a> {
    /// `visit(ImportStmt*)`.
    pub fn visit_import(&mut self, _s: StmtId) -> Result<()> {
        Err(not_ported("import::visit_import", self.src)) // STUB(import.cpp)
    }
    /// `transformSpecialImport(stmt)`.
    pub fn transform_special_import(&mut self, _s: StmtId) -> Result<Option<StmtId>> {
        Err(not_ported("import::transform_special_import", self.src)) // STUB(import.cpp)
    }
    /// `getImportPath(from, dots)`.
    pub fn get_import_path(&self, _from: Option<ExprId>, _dots: usize) -> Vec<String> {
        vec![] // STUB(import.cpp)
    }
    /// `transformCImport(name, args, ret, alt)`.
    pub fn transform_c_import(&mut self, _name: &str, _args: &[Param], _ret: Option<ExprId>, _alt: &str) -> Result<Option<StmtId>> {
        Err(not_ported("import::transform_c_import", self.src)) // STUB(import.cpp)
    }
    /// `transformCVarImport(name, type, alt)`.
    pub fn transform_c_var_import(&mut self, _name: &str, _ty: Option<ExprId>, _alt: &str) -> Result<Option<StmtId>> {
        Err(not_ported("import::transform_c_var_import", self.src)) // STUB(import.cpp)
    }
    /// `transformCDLLImport(dylib, name, args, ret, alt, isFunction)`.
    pub fn transform_cdll_import(
        &mut self,
        _dylib: ExprId,
        _name: &str,
        _args: &[Param],
        _ret: Option<ExprId>,
        _alt: &str,
        _is_function: bool,
    ) -> Result<Option<StmtId>> {
        Err(not_ported("import::transform_cdll_import", self.src)) // STUB(import.cpp)
    }
    /// `transformPythonImport(what, args, ret, alt)`.
    pub fn transform_python_import(&mut self, _what: ExprId, _args: &[Param], _ret: Option<ExprId>, _alt: &str) -> Result<Option<StmtId>> {
        Err(not_ported("import::transform_python_import", self.src)) // STUB(import.cpp)
    }
    /// `transformNewImport(file)`.
    pub fn transform_new_import(&mut self, _file: &ImportFile) -> Result<Option<StmtId>> {
        Err(not_ported("import::transform_new_import", self.src)) // STUB(import.cpp)
    }
}
