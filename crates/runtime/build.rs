//! Link-time configuration for libcodonrt (mirrors CMakeLists.txt codonrt target).
use std::env;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=CODON_LIBOMP_DIR");
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();

    if env::var_os("CARGO_FEATURE_OPENMP").is_some() {
        let dir = env::var("CODON_LIBOMP_DIR")
            .unwrap_or_else(|_| "/Users/mind/orca/toolchains/llvm-codon/lib".to_string());
        println!("cargo:rustc-link-search=native={dir}");
        println!("cargo:rustc-link-lib=dylib=omp");
        if target_os == "macos" || target_os == "linux" {
            println!("cargo:rustc-link-arg=-Wl,-rpath,{dir}");
        }
    }

    if env::var_os("CARGO_FEATURE_BLAS").is_some() && target_os == "macos" {
        println!("cargo:rustc-link-lib=framework=Accelerate");
    }

    // C libraries linked statically (bdwgc, liblzma) are not exported from a Rust
    // cdylib by default. The C++ libcodonrt exports them, so re-export by pattern.
    if target_os == "macos" {
        for pat in ["_GC_*", "_lzma_*"] {
            println!("cargo:rustc-cdylib-link-arg=-Wl,-exported_symbol,{pat}");
        }
    } else if target_os == "linux" {
        println!("cargo:rustc-cdylib-link-arg=-Wl,--export-dynamic-symbol=GC_*");
        println!("cargo:rustc-cdylib-link-arg=-Wl,--export-dynamic-symbol=lzma_*");
    }
}
