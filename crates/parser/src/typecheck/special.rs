//! Port of codon/parser/visitors/typecheck/special.cpp (phase-3 lane: replace STUB bodies).

use typhon_common::Result;

use super::{TypecheckVisitor, not_ported};
use crate::ids::{ExprId, StmtId, TypeId};

impl<'a> TypecheckVisitor<'a> {
    /// `prepareVTables()`: build virtual tables for all realized polymorphic classes.
    pub fn prepare_vtables(&mut self) -> Result<()> {
        Ok(()) // STUB(special.cpp)
    }
    /// `vTableSignature(fn)`.
    pub fn vtable_signature(&self, _f: TypeId) -> String {
        String::new() // STUB(special.cpp)
    }
    /// `valueMethodId(method)`.
    pub fn value_method_id(&mut self, _method: &str) -> usize {
        0 // STUB(special.cpp)
    }
    /// `generateSpecialAst(fn)`: body for compiler-generated functions (None if regular).
    pub fn generate_special_ast(&mut self, _f: TypeId) -> Result<Option<StmtId>> {
        Ok(None) // STUB(special.cpp)
    }
    /// `transform_named_tuple(call)` (special.cpp builtin handler).
    pub fn transform_named_tuple(&mut self, _call: ExprId) -> Result<Option<ExprId>> {
        Err(not_ported("special::transform_named_tuple", self.src)) // STUB(special.cpp)
    }
    /// `transform_functools_partial(call)` (special.cpp builtin handler).
    pub fn transform_functools_partial(&mut self, _call: ExprId) -> Result<Option<ExprId>> {
        Err(not_ported("special::transform_functools_partial", self.src)) // STUB(special.cpp)
    }
    /// `transform_super_f(call)` (special.cpp builtin handler).
    pub fn transform_super_f(&mut self, _call: ExprId) -> Result<Option<ExprId>> {
        Err(not_ported("special::transform_super_f", self.src)) // STUB(special.cpp)
    }
    /// `transform_ptr(call)` (special.cpp builtin handler).
    pub fn transform_ptr(&mut self, _call: ExprId) -> Result<Option<ExprId>> {
        Err(not_ported("special::transform_ptr", self.src)) // STUB(special.cpp)
    }
    /// `transform_array(call)` (special.cpp builtin handler).
    pub fn transform_array(&mut self, _call: ExprId) -> Result<Option<ExprId>> {
        Err(not_ported("special::transform_array", self.src)) // STUB(special.cpp)
    }
    /// `transform_is_instance(call)` (special.cpp builtin handler).
    pub fn transform_is_instance(&mut self, _call: ExprId) -> Result<Option<ExprId>> {
        Err(not_ported("special::transform_is_instance", self.src)) // STUB(special.cpp)
    }
    /// `transform_static_len(call)` (special.cpp builtin handler).
    pub fn transform_static_len(&mut self, _call: ExprId) -> Result<Option<ExprId>> {
        Err(not_ported("special::transform_static_len", self.src)) // STUB(special.cpp)
    }
    /// `transform_get_attr(call)` (special.cpp builtin handler).
    pub fn transform_get_attr(&mut self, _call: ExprId) -> Result<Option<ExprId>> {
        Err(not_ported("special::transform_get_attr", self.src)) // STUB(special.cpp)
    }
    /// `transform_set_attr(call)` (special.cpp builtin handler).
    pub fn transform_set_attr(&mut self, _call: ExprId) -> Result<Option<ExprId>> {
        Err(not_ported("special::transform_set_attr", self.src)) // STUB(special.cpp)
    }
    /// `transform_compile_error(call)` (special.cpp builtin handler).
    pub fn transform_compile_error(&mut self, _call: ExprId) -> Result<Option<ExprId>> {
        Err(not_ported("special::transform_compile_error", self.src)) // STUB(special.cpp)
    }
    /// `transform_tuple_fn(call)` (special.cpp builtin handler).
    pub fn transform_tuple_fn(&mut self, _call: ExprId) -> Result<Option<ExprId>> {
        Err(not_ported("special::transform_tuple_fn", self.src)) // STUB(special.cpp)
    }
    /// `transform_type_fn(call)` (special.cpp builtin handler).
    pub fn transform_type_fn(&mut self, _call: ExprId) -> Result<Option<ExprId>> {
        Err(not_ported("special::transform_type_fn", self.src)) // STUB(special.cpp)
    }
    /// `transform_realized_fn(call)` (special.cpp builtin handler).
    pub fn transform_realized_fn(&mut self, _call: ExprId) -> Result<Option<ExprId>> {
        Err(not_ported("special::transform_realized_fn", self.src)) // STUB(special.cpp)
    }
    /// `transform_static_print_fn(call)` (special.cpp builtin handler).
    pub fn transform_static_print_fn(&mut self, _call: ExprId) -> Result<Option<ExprId>> {
        Err(not_ported("special::transform_static_print_fn", self.src)) // STUB(special.cpp)
    }
    /// `transform_has_rtti_fn(call)` (special.cpp builtin handler).
    pub fn transform_has_rtti_fn(&mut self, _call: ExprId) -> Result<Option<ExprId>> {
        Err(not_ported("special::transform_has_rtti_fn", self.src)) // STUB(special.cpp)
    }
    /// `transform_static_fn_can_call(call)` (special.cpp builtin handler).
    pub fn transform_static_fn_can_call(&mut self, _call: ExprId) -> Result<Option<ExprId>> {
        Err(not_ported("special::transform_static_fn_can_call", self.src)) // STUB(special.cpp)
    }
    /// `transform_static_fn_arg_has_type(call)` (special.cpp builtin handler).
    pub fn transform_static_fn_arg_has_type(&mut self, _call: ExprId) -> Result<Option<ExprId>> {
        Err(not_ported("special::transform_static_fn_arg_has_type", self.src)) // STUB(special.cpp)
    }
    /// `transform_static_fn_arg_get_type(call)` (special.cpp builtin handler).
    pub fn transform_static_fn_arg_get_type(&mut self, _call: ExprId) -> Result<Option<ExprId>> {
        Err(not_ported("special::transform_static_fn_arg_get_type", self.src)) // STUB(special.cpp)
    }
    /// `transform_static_fn_args(call)` (special.cpp builtin handler).
    pub fn transform_static_fn_args(&mut self, _call: ExprId) -> Result<Option<ExprId>> {
        Err(not_ported("special::transform_static_fn_args", self.src)) // STUB(special.cpp)
    }
    /// `transform_static_fn_has_default(call)` (special.cpp builtin handler).
    pub fn transform_static_fn_has_default(&mut self, _call: ExprId) -> Result<Option<ExprId>> {
        Err(not_ported("special::transform_static_fn_has_default", self.src)) // STUB(special.cpp)
    }
    /// `transform_static_fn_get_default(call)` (special.cpp builtin handler).
    pub fn transform_static_fn_get_default(&mut self, _call: ExprId) -> Result<Option<ExprId>> {
        Err(not_ported("special::transform_static_fn_get_default", self.src)) // STUB(special.cpp)
    }
    /// `transform_static_fn_wrap_call_args(call)` (special.cpp builtin handler).
    pub fn transform_static_fn_wrap_call_args(&mut self, _call: ExprId) -> Result<Option<ExprId>> {
        Err(not_ported("special::transform_static_fn_wrap_call_args", self.src)) // STUB(special.cpp)
    }
    /// `transform_static_vars(call)` (special.cpp builtin handler).
    pub fn transform_static_vars(&mut self, _call: ExprId) -> Result<Option<ExprId>> {
        Err(not_ported("special::transform_static_vars", self.src)) // STUB(special.cpp)
    }
    /// `transform_static_children(call)` (special.cpp builtin handler).
    pub fn transform_static_children(&mut self, _call: ExprId) -> Result<Option<ExprId>> {
        Err(not_ported("special::transform_static_children", self.src)) // STUB(special.cpp)
    }
    /// `transform_static_tuple_type(call)` (special.cpp builtin handler).
    pub fn transform_static_tuple_type(&mut self, _call: ExprId) -> Result<Option<ExprId>> {
        Err(not_ported("special::transform_static_tuple_type", self.src)) // STUB(special.cpp)
    }
    /// `transform_static_format(call)` (special.cpp builtin handler).
    pub fn transform_static_format(&mut self, _call: ExprId) -> Result<Option<ExprId>> {
        Err(not_ported("special::transform_static_format", self.src)) // STUB(special.cpp)
    }
    /// `transform_static_int_to_str(call)` (special.cpp builtin handler).
    pub fn transform_static_int_to_str(&mut self, _call: ExprId) -> Result<Option<ExprId>> {
        Err(not_ported("special::transform_static_int_to_str", self.src)) // STUB(special.cpp)
    }
    /// `transform_static_platform(call)` (special.cpp builtin handler).
    pub fn transform_static_platform(&mut self, _call: ExprId) -> Result<Option<ExprId>> {
        Err(not_ported("special::transform_static_platform", self.src)) // STUB(special.cpp)
    }
    /// `transform_static_contains(call)` (special.cpp builtin handler).
    pub fn transform_static_contains(&mut self, _call: ExprId) -> Result<Option<ExprId>> {
        Err(not_ported("special::transform_static_contains", self.src)) // STUB(special.cpp)
    }
    /// `transformSuper()`.
    pub fn transform_super(&mut self) -> Result<Option<ExprId>> {
        Err(not_ported("special::transform_super", self.src)) // STUB(special.cpp)
    }
    /// `transformHasAttr(call, allowStatic)`.
    pub fn transform_has_attr(&mut self, _call: ExprId, _allow: bool) -> Result<Option<ExprId>> {
        Err(not_ported("special::transform_has_attr", self.src)) // STUB(special.cpp)
    }
    /// `generate_function_call_internal_ast(fn)`: generated suite for a special function.
    pub fn generate_function_call_internal_ast(&mut self, _f: TypeId) -> Result<Option<StmtId>> {
        Err(not_ported("special::generate_function_call_internal_ast", self.src)) // STUB(special.cpp)
    }
    /// `generate_union_new_ast(fn)`: generated suite for a special function.
    pub fn generate_union_new_ast(&mut self, _f: TypeId) -> Result<Option<StmtId>> {
        Err(not_ported("special::generate_union_new_ast", self.src)) // STUB(special.cpp)
    }
    /// `generate_union_tag_ast(fn)`: generated suite for a special function.
    pub fn generate_union_tag_ast(&mut self, _f: TypeId) -> Result<Option<StmtId>> {
        Err(not_ported("special::generate_union_tag_ast", self.src)) // STUB(special.cpp)
    }
    /// `generate_union_dispatch_ast(fn)`: generated suite for a special function.
    pub fn generate_union_dispatch_ast(&mut self, _f: TypeId) -> Result<Option<StmtId>> {
        Err(not_ported("special::generate_union_dispatch_ast", self.src)) // STUB(special.cpp)
    }
    /// `generate_union_iterator_ast(fn)`: generated suite for a special function.
    pub fn generate_union_iterator_ast(&mut self, _f: TypeId) -> Result<Option<StmtId>> {
        Err(not_ported("special::generate_union_iterator_ast", self.src)) // STUB(special.cpp)
    }
    /// `generate_union_binary_ast(fn)`: generated suite for a special function.
    pub fn generate_union_binary_ast(&mut self, _f: TypeId) -> Result<Option<StmtId>> {
        Err(not_ported("special::generate_union_binary_ast", self.src)) // STUB(special.cpp)
    }
    /// `generate_named_keys_ast(fn)`: generated suite for a special function.
    pub fn generate_named_keys_ast(&mut self, _f: TypeId) -> Result<Option<StmtId>> {
        Err(not_ported("special::generate_named_keys_ast", self.src)) // STUB(special.cpp)
    }
    /// `generate_tuple_mul_ast(fn)`: generated suite for a special function.
    pub fn generate_tuple_mul_ast(&mut self, _f: TypeId) -> Result<Option<StmtId>> {
        Err(not_ported("special::generate_tuple_mul_ast", self.src)) // STUB(special.cpp)
    }
    /// `generate_type_info_init_ast(fn)`: generated suite for a special function.
    pub fn generate_type_info_init_ast(&mut self, _f: TypeId) -> Result<Option<StmtId>> {
        Err(not_ported("special::generate_type_info_init_ast", self.src)) // STUB(special.cpp)
    }
    /// `generate_super_dispatch_ast(fn)`: generated suite for a special function.
    pub fn generate_super_dispatch_ast(&mut self, _f: TypeId) -> Result<Option<StmtId>> {
        Err(not_ported("special::generate_super_dispatch_ast", self.src)) // STUB(special.cpp)
    }
    /// `populate_static_tuple_loop(iter, vars)`: statements for each static loop iteration.
    pub fn populate_static_tuple_loop(&mut self, _iter: ExprId, _vars: &[String]) -> Result<Vec<StmtId>> {
        Err(not_ported("special::populate_static_tuple_loop", self.src)) // STUB(special.cpp)
    }
    /// `populate_simple_static_range_loop(iter, vars)`: statements for each static loop iteration.
    pub fn populate_simple_static_range_loop(&mut self, _iter: ExprId, _vars: &[String]) -> Result<Vec<StmtId>> {
        Err(not_ported("special::populate_simple_static_range_loop", self.src)) // STUB(special.cpp)
    }
    /// `populate_static_range_loop(iter, vars)`: statements for each static loop iteration.
    pub fn populate_static_range_loop(&mut self, _iter: ExprId, _vars: &[String]) -> Result<Vec<StmtId>> {
        Err(not_ported("special::populate_static_range_loop", self.src)) // STUB(special.cpp)
    }
    /// `populate_static_fn_overloads_loop(iter, vars)`: statements for each static loop iteration.
    pub fn populate_static_fn_overloads_loop(&mut self, _iter: ExprId, _vars: &[String]) -> Result<Vec<StmtId>> {
        Err(not_ported("special::populate_static_fn_overloads_loop", self.src)) // STUB(special.cpp)
    }
    /// `populate_static_enumerate_loop(iter, vars)`: statements for each static loop iteration.
    pub fn populate_static_enumerate_loop(&mut self, _iter: ExprId, _vars: &[String]) -> Result<Vec<StmtId>> {
        Err(not_ported("special::populate_static_enumerate_loop", self.src)) // STUB(special.cpp)
    }
    /// `populate_static_vars_loop(iter, vars)`: statements for each static loop iteration.
    pub fn populate_static_vars_loop(&mut self, _iter: ExprId, _vars: &[String]) -> Result<Vec<StmtId>> {
        Err(not_ported("special::populate_static_vars_loop", self.src)) // STUB(special.cpp)
    }
    /// `populate_static_methods_loop(iter, vars)`: statements for each static loop iteration.
    pub fn populate_static_methods_loop(&mut self, _iter: ExprId, _vars: &[String]) -> Result<Vec<StmtId>> {
        Err(not_ported("special::populate_static_methods_loop", self.src)) // STUB(special.cpp)
    }
    /// `populate_static_var_types_loop(iter, vars)`: statements for each static loop iteration.
    pub fn populate_static_var_types_loop(&mut self, _iter: ExprId, _vars: &[String]) -> Result<Vec<StmtId>> {
        Err(not_ported("special::populate_static_var_types_loop", self.src)) // STUB(special.cpp)
    }
    /// `populate_static_heterogenous_tuple_loop(iter, vars)`: statements for each static loop iteration.
    pub fn populate_static_heterogenous_tuple_loop(&mut self, _iter: ExprId, _vars: &[String]) -> Result<Vec<StmtId>> {
        Err(not_ported("special::populate_static_heterogenous_tuple_loop", self.src)) // STUB(special.cpp)
    }
}
