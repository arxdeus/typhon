//! Unpack the generated stdlib modules (Unicode tables, float dtoa tables) from the
//! bundled archive `tools/unicode/tables-*.tar.gz` into `$OUT_DIR/stdlib`, verifying
//! the SHA-256 hashes listed in `tools/unicode/manifest.json`. The driver adds that
//! directory as an extra stdlib search root (`TYPHON_GENERATED_STDLIB`).

use sha2::{Digest, Sha256};
use std::io::Read;
use std::path::{Path, PathBuf};

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn main() {
    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let tools = manifest_dir.join("../../tools/unicode");
    let manifest_path = tools.join("manifest.json");
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap()).join("stdlib");
    println!("cargo:rerun-if-changed={}", manifest_path.display());
    println!("cargo:rustc-env=TYPHON_GENERATED_STDLIB={}", out.display());

    let manifest: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&manifest_path).expect("read manifest.json"))
            .expect("parse manifest.json");
    let archive = tools.join(manifest["archive"].as_str().expect("manifest: archive"));
    println!("cargo:rerun-if-changed={}", archive.display());
    let data = std::fs::read(&archive).expect("read generated-tables archive");
    let want = manifest["sha256"].as_str().unwrap_or("");
    let got = hex(&Sha256::digest(&data));
    assert_eq!(got, want, "archive checksum mismatch: {}", archive.display());

    let files = manifest["files"].as_object().expect("manifest: files");
    let mut seen = 0usize;
    let mut ar = tar::Archive::new(flate2::read::GzDecoder::new(&data[..]));
    for entry in ar.entries().expect("read archive") {
        let mut entry = entry.expect("archive entry");
        if !entry.header().entry_type().is_file() {
            continue;
        }
        let name = entry.path().expect("entry path").to_string_lossy().into_owned();
        let meta = files.get(&name).unwrap_or_else(|| panic!("unexpected archive member {name}"));
        let mut buf = Vec::new();
        entry.read_to_end(&mut buf).expect("read entry");
        assert_eq!(hex(&Sha256::digest(&buf)), meta["sha256"].as_str().unwrap_or(""), "checksum mismatch: {name}");
        assert!(!Path::new(&name).is_absolute() && !name.contains(".."), "bad member path {name}");
        let dest = out.join(&name);
        if std::fs::read(&dest).ok().as_deref() != Some(&buf[..]) {
            std::fs::create_dir_all(dest.parent().unwrap()).unwrap();
            std::fs::write(&dest, &buf).unwrap();
        }
        seen += 1;
    }
    assert_eq!(seen, files.len(), "archive is missing files listed in manifest");
}
