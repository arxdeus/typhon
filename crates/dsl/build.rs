//! Captures the rustc version for the plugin ABI tag.
fn main() {
    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".into());
    let v = std::process::Command::new(rustc)
        .arg("--version")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .unwrap_or_default();
    println!("cargo:rustc-env=TYPHON_DSL_RUSTC_VERSION={}", v.trim());
    println!("cargo:rerun-if-env-changed=RUSTC");
}
