//! Unpack the generated stdlib modules (Unicode tables, float dtoa tables) like the C++
//! CMake target `unicode_data` (cmake/UnicodeTables.cmake): runs
//! `scripts/unicode_tables.py extract` into `$OUT_DIR/stdlib`. The driver adds that
//! directory as an extra stdlib search root (`TYPHON_GENERATED_STDLIB`).

use std::path::PathBuf;
use std::process::Command;

fn main() {
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let root = manifest.join("../../..");
    let script = root.join("scripts/unicode_tables.py");
    let archive = root.join("scripts/unicode/tables-16.0.0.tar.gz");
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap()).join("stdlib");
    println!("cargo:rerun-if-changed={}", script.display());
    println!("cargo:rerun-if-changed={}", archive.display());
    println!("cargo:rerun-if-changed={}", root.join("scripts/unicode/manifest.json").display());
    println!("cargo:rustc-env=TYPHON_GENERATED_STDLIB={}", out.display());
    let marker = out.join("internal/numeric/generated/float_dtoa_tables.codon");
    if marker.exists() {
        return;
    }
    let python = std::env::var("PYTHON").unwrap_or_else(|_| "python3".into());
    let status = Command::new(&python)
        .arg(&script)
        .arg("extract")
        .arg("--archive")
        .arg(&archive)
        .arg("--output-dir")
        .arg(&out)
        .status();
    match status {
        Ok(s) if s.success() => {}
        Ok(s) => println!("cargo:warning=generated stdlib tables: {python} exited with {s}"),
        Err(e) => println!("cargo:warning=generated stdlib tables: could not run {python}: {e}"),
    }
}
