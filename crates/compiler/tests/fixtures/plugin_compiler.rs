//! Rust counterpart of `test/dsl/library.cpp` built with `CODON_TEST_COMPILER_PLUGIN`:
//! a compiler plugin with the default DSL that also exports `codon_test_compiler_value`.

#[derive(Default)]
pub struct TestDsl;
impl typhon_dsl::Dsl for TestDsl {}

typhon_dsl::export_plugin!(TestDsl);

#[unsafe(no_mangle)]
pub extern "C" fn codon_test_compiler_value() -> i64 {
    17
}
