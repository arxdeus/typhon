//! Link-time configuration for libtyphonrt (mirrors CMakeLists.txt typhonrt target).
use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=TYPHON_LIBOMP_DIR");
    println!("cargo:rerun-if-env-changed=TYPHON_OPENBLAS_DIR");
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

    if env::var_os("CARGO_FEATURE_BLAS").is_some() {
        match target_os.as_str() {
            // Accelerate (ILP64 `$NEWLAPACK$ILP64` symbols, macOS 13.3+ / iOS 16.4+).
            "macos" | "ios" => println!("cargo:rustc-link-lib=framework=Accelerate"),
            // Windows links the prebuilt LP64 libopenblas (see llvisitor/output.rs).
            "windows" => {}
            // Android / Linux: OpenBLAS built from source, bundled into libtyphonrt.a.
            _ => openblas(),
        }
    }

    // C libraries linked statically (bdwgc, liblzma): link whole archives so every
    // `GC_*` / `lzma_*` symbol the stdlib may reach via `from C import` is in
    // libtyphonrt.a. (Search paths come from bdwgc-alloc-sys / liblzma-sys.)
    println!("cargo:rustc-link-lib=static:+whole-archive=gc");
    println!("cargo:rustc-link-lib=static:+whole-archive=lzma");
    let _ = omp_rust;
}

/// Build static OpenBLAS (ILP64, plain `cblas_*` / Fortran `xxxx_` names, C-translated
/// LAPACK so no Fortran compiler is needed) and bundle it whole into libtyphonrt.a,
/// since the stdlib reaches the symbols by name through `from C import`.
/// `TYPHON_OPENBLAS_DIR=<dir with libopenblas.a>` skips the build.
fn openblas() {
    if let Ok(dir) = env::var("TYPHON_OPENBLAS_DIR") {
        println!("cargo:rustc-link-search=native={dir}");
        println!("cargo:rustc-link-lib=static:+whole-archive=openblas");
        return;
    }
    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    let lib = out.join("openblas-omp");
    if !lib.join("libopenblas.a").exists() {
        let src = openblas_build::download(&out).expect("download OpenBLAS source");
        make_openblas(&src);
        std::fs::create_dir_all(&lib).unwrap();
        std::fs::copy(src.join("libopenblas.a"), lib.join("libopenblas.a")).expect("libopenblas.a");
    }
    println!("cargo:rustc-link-search=native={}", lib.display());
    println!("cargo:rustc-link-lib=static:+whole-archive=openblas");
}

fn make_openblas(src: &Path) {
    let arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let cross = env::var("TARGET").ok() != env::var("HOST").ok();
    let tool = cc::Build::new().get_compiler();
    let ar = cc::Build::new().get_archiver();
    let mut ccpath = tool.path().to_string_lossy().into_owned();
    // gcc's -fopenmp emits GOMP_* calls; typhon-omp implements only the libomp ABI.
    if os == "linux" && !tool.is_like_clang() {
        ccpath = "clang".into();
    }
    // OpenBLAS derives RANLIB from CC's name (`<triple>-ranlib`), which the NDK lacks.
    let arp = PathBuf::from(ar.get_program());
    let ranlib = match arp.file_name().and_then(|f| f.to_str()) {
        Some("llvm-ar") => arp.with_file_name("llvm-ranlib").to_string_lossy().into_owned(),
        _ => "ranlib".into(),
    };
    let mut args: Vec<String> = vec![
        "libs".into(),
        "netlib".into(),
        "BINARY=64".into(),
        "INTERFACE64=1".into(),
        "NOFORTRAN=1".into(), // implies C_LAPACK=1 (f2c-translated LAPACK)
        "NO_SHARED=1".into(),
        "NO_LAPACKE=1".into(),
        // Threads via OpenMP, resolved against typhon-omp (`__kmpc_*`/`omp_*` in this
        // archive): one shared pool with `@par`; BLAS inside a parallel region runs
        // serially (omp_in_parallel). Needs clang (`-fopenmp` -> libomp ABI, no GOMP_*).
        "USE_THREAD=1".into(),
        "USE_OPENMP=1".into(),
        "NUM_THREADS=64".into(),
        // Buffer sets for concurrent BLAS calls from independent initial threads
        // (e.g. several Flutter isolates calling a `-lib`).
        "NUM_PARALLEL=8".into(),
        "BUILD_TESTING=0".into(),
        format!("AR={}", ar.get_program().to_string_lossy()),
        format!("RANLIB={ranlib}"),
        format!("-j{}", env::var("NUM_JOBS").unwrap_or_else(|_| "8".into())),
    ];
    // Flags cc picked up (target, sysroot, -fPIC, ...) except optimisation/debug ones,
    // folded into CC: a command-line CFLAGS would replace OpenBLAS's own.
    let cflags: Vec<String> = tool
        .args()
        .iter()
        .map(|a| a.to_string_lossy().into_owned())
        .filter(|a| !a.starts_with("-O") && a != "-g" && !a.starts_with("-gdwarf"))
        .collect();
    args.push(format!("CC={} {}", ccpath, cflags.join(" ")).trim_end().to_string());
    match (arch.as_str(), os.as_str()) {
        ("aarch64", "android") => args.push("TARGET=ARMV8".into()),
        ("aarch64", _) => args.extend(["TARGET=ARMV8".into(), "DYNAMIC_ARCH=1".into()]),
        ("x86_64", _) => args.extend(["TARGET=PRESCOTT".into(), "DYNAMIC_ARCH=1".into()]),
        _ => {}
    }
    if cross {
        let host = cc::Build::new().target(&env::var("HOST").unwrap()).get_compiler();
        args.push(format!("HOSTCC={}", host.path().to_string_lossy()));
        args.push("CROSS=1".into());
    }
    let t0 = std::time::Instant::now();
    let log = std::fs::File::create(src.join("build.log")).unwrap();
    // A partial earlier build in this tree would leave a mismatched archive.
    let _ = Command::new("make").current_dir(src).arg("clean").env_remove("TARGET").output();
    let st = Command::new("make")
        .current_dir(src)
        .args(&args)
        .env_remove("TARGET") // OpenBLAS reads TARGET as a CPU name
        .env_remove("CFLAGS")
        .env_remove("MAKEFLAGS")
        .env_remove("MFLAGS")
        .stdout(log.try_clone().unwrap())
        .stderr(log)
        .status()
        .expect("run make for OpenBLAS");
    if !st.success() {
        let txt = std::fs::read_to_string(src.join("build.log")).unwrap_or_default();
        let tail: Vec<&str> = txt.lines().rev().take(40).collect();
        panic!("OpenBLAS build failed ({}):\n{}", args.join(" "), tail.into_iter().rev().collect::<Vec<_>>().join("\n"));
    }
    println!("cargo:warning=OpenBLAS built in {:.0}s", t0.elapsed().as_secs_f64());
}
