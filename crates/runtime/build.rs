//! Link-time configuration for libtyphonrt (mirrors CMakeLists.txt typhonrt target).
use std::env;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=TYPHON_LIBOMP_DIR");
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();

    let omp_rust = env::var_os("CARGO_FEATURE_OMP_RUST").is_some();
    let omp_libomp = env::var_os("CARGO_FEATURE_OPENMP_LIBOMP").is_some();
    if omp_rust && omp_libomp {
        panic!("features `omp-rust` and `openmp-libomp` are mutually exclusive");
    }
    if omp_libomp {
        let dir = env::var("TYPHON_LIBOMP_DIR")
            .unwrap_or_else(|_| "/Users/mind/orca/toolchains/llvm20/lib".to_string());
        println!("cargo:rustc-link-search=native={dir}");
        println!("cargo:rustc-link-lib=dylib=omp");
        if target_os == "macos" || target_os == "linux" {
            println!("cargo:rustc-link-arg=-Wl,-rpath,{dir}");
        }
    }

    if env::var_os("CARGO_FEATURE_BLAS").is_some() && (target_os == "macos" || target_os == "ios") {
        // Android has no system BLAS: numpy linalg (cblas_*/LAPACK) is unavailable there.
        println!("cargo:rustc-link-lib=framework=Accelerate");
    }

    // C libraries linked statically (bdwgc, liblzma): link whole archives so every
    // `GC_*` / `lzma_*` symbol the stdlib may reach via `from C import` is in
    // libtyphonrt.a. (Search paths come from bdwgc-alloc-sys / liblzma-sys.)
    println!("cargo:rustc-link-lib=static:+whole-archive=gc");
    println!("cargo:rustc-link-lib=static:+whole-archive=lzma");
    let _ = omp_rust;
}
