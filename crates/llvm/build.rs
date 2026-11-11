//! Builds `cpp/tm_shim.cpp` (TargetMachine options missing from the LLVM C API).
fn main() {
    let prefix = std::env::var("LLVM_SYS_201_PREFIX").unwrap_or_else(|_| "/Users/mind/orca/toolchains/llvm-codon".into());
    println!("cargo:rerun-if-changed=cpp/tm_shim.cpp");
    println!("cargo:rerun-if-env-changed=LLVM_SYS_201_PREFIX");
    cc::Build::new()
        .cpp(true)
        .file("cpp/tm_shim.cpp")
        .include(format!("{prefix}/include"))
        .flag_if_supported("-std=c++17")
        .flag_if_supported("-fno-rtti")
        .warnings(false)
        .compile("typhon_tm_shim");
}
