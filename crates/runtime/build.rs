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

    if env::var("CARGO_CFG_TARGET_ARCH").as_deref() == Ok("aarch64")
        && matches!(target_os.as_str(), "macos" | "ios" | "android" | "linux")
    {
        torch_kernels(&target_os);
    }
}

/// SLEEF (AdvSIMD expf/expm1f/logf) + NNPACK (conv output) at torch 2.15.0's pinned
/// commits, plus the C-ABI wrappers in src/torchk/torchk.c, for bit-identical torch CPU
/// results. Compiled with each library's own CMake flags (no cmake needed). aarch64 only;
/// other targets use the .py ports in stdlib/torch.
fn torch_kernels(target_os: &str) {
    let v = Path::new("vendor");
    println!("cargo:rerun-if-changed=vendor");
    println!("cargo:rerun-if-changed=src/torchk");
    let base = || {
        let mut b = cc::Build::new();
        b.cargo_metadata(false).warnings(false).opt_level(3).define("NDEBUG", None);
        b
    };
    // SLEEF: sleef's CMake flags for the ADVSIMD libm target (-ffp-contract=off matters).
    let s = v.join("sleef/src");
    base()
        .std("gnu99")
        .flag("-ffp-contract=off")
        .flag("-fno-math-errno")
        .flag("-fno-trapping-math")
        .define("DORENAME", "1")
        .define("ENABLE_ADVSIMD", "1")
        .define("ENABLE_BUILTIN_MATH", "1")
        .define("SLEEF_STATIC_LIBS", "1")
        .define("ALIAS_NO_EXT_SUFFIX", "\"alias_advsimd.h\"")
        .include(s.join("common"))
        .include(s.join("arch"))
        .include(v.join("sleef/generated"))
        .include(s.join("libm"))
        .file(s.join("libm/sleefsimdsp.c"))
        .file(s.join("libm/rempitab.c"))
        .compile("typhon_sleef");
    // pthreadpool + cpuinfo (NNPACK's deps, needed by nnp_initialize / its API).
    let inc = |b: &mut cc::Build| {
        for d in ["nnpack/include", "nnpack/src", "pthreadpool/include", "cpuinfo/include", "psimd/include", "fp16/include", "fxdiv/include"] {
            b.include(v.join(d));
        }
    };
    let mut pt = base();
    inc(&mut pt);
    pt.std("c11").define("FXDIV_USE_INLINE_ASSEMBLY", "0").define("PTHREADPOOL_USE_FASTPATH", "0");
    for f in ["legacy-api.c", "memory.c", "portable-api.c", "pthreads.c"] {
        pt.file(v.join("pthreadpool/src").join(f));
    }
    pt.compile("typhon_pthreadpool");
    let mut ci = base();
    ci.std("c99").include(v.join("cpuinfo/include")).include(v.join("cpuinfo/src"));
    ci.define("CPUINFO_LOG_LEVEL", "0").define("CPUINFO_LOG_TO_STDIO", "0");
    let mut cif = vec!["api.c", "init.c", "cache.c", "log.c", "arm/cache.c", "arm/uarch.c"];
    if target_os == "macos" || target_os == "ios" {
        cif.extend(["arm/mach/init.c", "mach/topology.c"]);
    } else {
        ci.define("_GNU_SOURCE", "1");
        cif.extend([
            "arm/linux/aarch64-isa.c", "arm/linux/chipset.c", "arm/linux/clusters.c", "arm/linux/cpuinfo.c",
            "arm/linux/hwcap.c", "arm/linux/init.c", "arm/linux/midr.c", "linux/cpulist.c",
            "linux/multiline.c", "linux/processors.c", "linux/smallfile.c",
        ]);
        if target_os == "android" {
            cif.push("arm/android/properties.c");
        }
    }
    for f in cif {
        ci.file(v.join("cpuinfo/src").join(f));
    }
    ci.compile("typhon_cpuinfo");
    // NNPACK, neon backend (NNPACK CMake: -O3 -std=gnu99; its ARM -mfpu flags are 32-bit only).
    let mut nn = base();
    inc(&mut nn);
    nn.std("gnu99")
        .define("CPUINFO_SUPPORTED_PLATFORM", "1")
        .define("FXDIV_USE_INLINE_ASSEMBLY", "0")
        .define("NNP_CONVOLUTION_ONLY", "0")
        .define("NNP_INFERENCE_ONLY", "0");
    let nsrc = v.join("nnpack/src");
    for e in std::fs::read_dir(&nsrc).unwrap().chain(std::fs::read_dir(nsrc.join("neon")).unwrap())
        .chain(std::fs::read_dir(nsrc.join("neon/blas")).unwrap())
        .chain(std::fs::read_dir(nsrc.join("psimd")).unwrap())
        .chain(std::fs::read_dir(nsrc.join("psimd/blas")).unwrap())
    {
        let p = e.unwrap().path();
        if p.extension().is_some_and(|x| x == "c") {
            nn.file(p);
        }
    }
    nn.file("src/torchk/torchk.c");
    nn.compile("typhon_nnpack");
    // Whole archives: the stdlib reaches typhon_* by name through `from C import`.
    let out = env::var("OUT_DIR").unwrap();
    println!("cargo:rustc-link-search=native={out}");
    for l in ["typhon_nnpack", "typhon_sleef", "typhon_pthreadpool", "typhon_cpuinfo"] {
        println!("cargo:rustc-link-lib=static:+whole-archive={l}");
    }
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
    // Directory name keys the cached archive by its make config (see make_openblas).
    let android = env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("android");
    let lib = out.join(if android { "openblas-omp-mt1024" } else { "openblas-omp" });
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
        // GEMM_MULTITHREAD_THRESHOLD: generic ARMV8 threads every GEMM with M*N*K > 65536*4
        // over omp_get_max_threads() (all 8 cores of a phone, little ones included), and
        // driver/level3/level3_thread.c spin-waits (YIELDING nops) on the slowest thread.
        // On a Pixel 9a fine-tune that spin was 45% of cycles and OMP 8 ran slower than 1
        // BLAS thread. 1024 keeps GEMMs up to 2^26 MNK on the caller thread. Bits do not
        // change: level-3 threading splits M/N only, never K (checked on device).
        ("aarch64", "android") => {
            args.push("TARGET=ARMV8".into());
            args.push("GEMM_MULTITHREAD_THRESHOLD=1024".into());
        }
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
