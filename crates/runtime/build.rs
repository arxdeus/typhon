//! Link-time configuration for libcodonrt (mirrors CMakeLists.txt codonrt target).
use std::env;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=vendor/bdwgc");
    println!("cargo:rerun-if-env-changed=CODON_LIBOMP_DIR");
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();

    build_bdwgc(&target_os);

    let omp_rust = env::var_os("CARGO_FEATURE_OMP_RUST").is_some();
    let omp_libomp = env::var_os("CARGO_FEATURE_OPENMP_LIBOMP").is_some();
    if omp_rust && omp_libomp {
        panic!("features `omp-rust` and `openmp-libomp` are mutually exclusive");
    }
    if omp_libomp {
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
    // cdylib by default, and only referenced archive members are linked. The C++
    // libcodonrt exports them in full: link whole archives and re-export by pattern.
    // (Search paths are emitted by bdwgc-alloc-sys / liblzma-sys build scripts.)
    println!("cargo:rustc-link-lib=static:+whole-archive=gc");
    println!("cargo:rustc-link-lib=static:+whole-archive=lzma");
    // The Rust OpenMP runtime (typhon-omp rlib; its C shim is bundled whole-archive
    // by typhon-omp's build script) is exported by pattern as well.
    if target_os == "macos" {
        let omp_pats: &[&str] = if omp_rust { &["___kmpc_*", "_omp_*"] } else { &[] };
        for pat in ["_GC_*", "_lzma_*"].iter().chain(omp_pats) {
            println!("cargo:rustc-cdylib-link-arg=-Wl,-exported_symbol,{pat}");
        }
    } else if target_os == "linux" {
        println!("cargo:rustc-cdylib-link-arg=-Wl,--export-dynamic-symbol=GC_*");
        println!("cargo:rustc-cdylib-link-arg=-Wl,--export-dynamic-symbol=lzma_*");
        if omp_rust {
            println!("cargo:rustc-cdylib-link-arg=-Wl,--export-dynamic-symbol=__kmpc_*");
            println!("cargo:rustc-cdylib-link-arg=-Wl,--export-dynamic-symbol=omp_*");
        }
    }
}

/// bdwgc v8.2.12 (vendor/bdwgc, same tag as C++ `cmake/deps.cmake`) built with the
/// same CMake options as the C++ libcodonrt. The `bdwgc-alloc-sys` crate vendors a
/// bdwgc 8.3 snapshot without `enable_large_config`, whose heap-growth policy
/// collects ~7x more often on allocation-heavy programs (float.py: 41 vs 6 GCs).
fn build_bdwgc(target_os: &str) {
    let mut cfg = cmake::Config::new("vendor/bdwgc");
    cfg.profile("Release")
        .define("CMAKE_POSITION_INDEPENDENT_CODE", "ON")
        .define("BUILD_SHARED_LIBS", "OFF")
        .define("build_cord", "OFF")
        .define("build_tests", "OFF")
        .define("enable_docs", "OFF")
        .define("install_headers", "ON")
        .define("enable_threads", "ON")
        .define("enable_large_config", "ON")
        .define("enable_thread_local_alloc", "ON");
    if target_os == "windows" {
        cfg.define("enable_parallel_mark", "OFF")
            .define("disable_handle_fork", "ON")
            .define("enable_single_obj_compilation", "ON")
            .cflag("-DGC_BUILTIN_ATOMIC");
    } else {
        cfg.define("enable_handle_fork", "ON");
    }
    let dst = cfg.build();
    println!("cargo:rustc-link-search=native={}", dst.join("lib").display());
    println!("cargo:include={}", dst.join("include").display());
}
