//! Rust counterpart of `test/dsl/library.cpp` (runtime variant).

#[unsafe(no_mangle)]
pub extern "C" fn codon_test_runtime_value() -> i64 {
    42
}
