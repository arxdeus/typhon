//! Compiles the variadic `__kmpc_fork_call` shim (stable Rust cannot define C-variadic
//! functions). Linked whole-archive so the symbol survives into libcodonrt / the cdylib.
fn main() {
    println!("cargo:rerun-if-changed=csrc/fork_call.c");
    cc::Build::new()
        .file("csrc/fork_call.c")
        .warnings(true)
        .link_lib_modifier("+whole-archive")
        .compile("typhon_omp_shim");
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    // Export the C entry points from this crate's own cdylib (used as a drop-in
    // `libomp.dylib` for oracle runs). Rust #[no_mangle] items are exported anyway.
    for sym in ["__kmpc_fork_call", "typhon_omp_invoke_microtask"] {
        if target_os == "macos" {
            println!("cargo:rustc-cdylib-link-arg=-Wl,-exported_symbol,_{sym}");
        } else if target_os == "linux" {
            println!("cargo:rustc-cdylib-link-arg=-Wl,--export-dynamic-symbol={sym}");
        }
    }
}
