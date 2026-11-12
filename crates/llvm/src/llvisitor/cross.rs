//! Cross-target link drivers: Android NDK clang and Xcode SDK clang.

use std::path::{Path, PathBuf};
use std::process::Command;

use typhon_common::target::{TargetInfo, TargetOs};
use typhon_common::{CompilerError, Result};

fn err(msg: String) -> CompilerError {
    CompilerError::Llvm(msg)
}

/// NDK root: `$ANDROID_NDK_HOME`, `$ANDROID_NDK_ROOT`, else the newest
/// `$ANDROID_HOME/ndk/*` (or `$ANDROID_SDK_ROOT`, `~/Library/Android/sdk`).
pub fn android_ndk() -> Result<PathBuf> {
    for v in ["ANDROID_NDK_HOME", "ANDROID_NDK_ROOT", "ANDROID_NDK"] {
        if let Ok(d) = std::env::var(v)
            && !d.is_empty()
            && Path::new(&d).is_dir()
        {
            return Ok(PathBuf::from(d));
        }
    }
    let mut sdks: Vec<PathBuf> = ["ANDROID_HOME", "ANDROID_SDK_ROOT"].iter().filter_map(|v| std::env::var(v).ok()).map(PathBuf::from).collect();
    if let Ok(h) = std::env::var("HOME") {
        sdks.push(PathBuf::from(&h).join("Library/Android/sdk"));
        sdks.push(PathBuf::from(h).join("Android/Sdk"));
    }
    for sdk in sdks {
        let Ok(rd) = std::fs::read_dir(sdk.join("ndk")) else { continue };
        let mut v: Vec<PathBuf> = rd.filter_map(|e| e.ok().map(|e| e.path())).filter(|p| p.is_dir()).collect();
        v.sort_by_key(|p| version_key(&p.file_name().unwrap_or_default().to_string_lossy()));
        // Newest complete NDK (partial installs lack the toolchain).
        if let Some(p) = v.into_iter().rev().find(|p| ndk_clang(p).is_ok()) {
            return Ok(p);
        }
    }
    Err(err("Android NDK not found: set ANDROID_NDK_HOME (or install an NDK under $ANDROID_HOME/ndk)".into()))
}

fn version_key(s: &str) -> Vec<u64> {
    s.split(|c: char| !c.is_ascii_digit()).filter(|x| !x.is_empty()).map(|x| x.parse().unwrap_or(0)).collect()
}

fn ndk_clang(ndk: &Path) -> Result<PathBuf> {
    let prebuilt = ndk.join("toolchains/llvm/prebuilt");
    for host in ["darwin-x86_64", "darwin-arm64", "linux-x86_64", "windows-x86_64"] {
        let c = prebuilt.join(host).join("bin/clang");
        if c.is_file() {
            return Ok(c);
        }
    }
    Err(err(format!("clang not found in {}", prebuilt.display())))
}

fn xcrun(args: &[&str]) -> Result<String> {
    let out = Command::new("xcrun").args(args).output().map_err(|e| err(format!("xcrun: {e}")))?;
    if !out.status.success() {
        return Err(err(format!("xcrun {} failed: {}", args.join(" "), String::from_utf8_lossy(&out.stderr).trim())));
    }
    Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

/// Driver + target/sysroot arguments for the cross link.
pub fn linker_command(target: &TargetInfo, triple: &str, sysroot: &str) -> Result<Vec<String>> {
    let override_linker = std::env::var("TYPHON_CROSS_LINKER").ok().filter(|s| !s.is_empty());
    match target.os {
        TargetOs::Android => {
            let clang = match override_linker {
                Some(l) => l,
                None => ndk_clang(&android_ndk()?)?.to_string_lossy().into_owned(),
            };
            let mut v = vec![clang, format!("--target={triple}")];
            if !sysroot.is_empty() {
                v.push(format!("--sysroot={sysroot}"));
            }
            Ok(v)
        }
        TargetOs::Ios => {
            let sdk = if target.simulator { "iphonesimulator" } else { "iphoneos" };
            let clang = match override_linker {
                Some(l) => l,
                None => xcrun(&["--sdk", sdk, "--find", "clang"])?,
            };
            let root = if sysroot.is_empty() { xcrun(&["--sdk", sdk, "--show-sdk-path"])? } else { sysroot.to_string() };
            Ok(vec![clang, "-target".into(), triple.into(), "-isysroot".into(), root])
        }
        _ => {
            let mut v = vec![override_linker.unwrap_or_else(|| "clang".into()), format!("--target={triple}")];
            if !sysroot.is_empty() {
                v.push(format!("--sysroot={sysroot}"));
            }
            Ok(v)
        }
    }
}

pub fn gc_flags(target: &TargetInfo) -> &'static [&'static str] {
    if target.is_apple() {
        &["-Wl,-dead_strip", "-Wl,-dead_strip_dylibs"]
    } else if target.os == TargetOs::Android {
        // --allow-multiple-definition: an `@export` in the program may override a runtime
        // hook (e.g. `seq_os_home`); the object comes first so it wins, as with ld64.
        &["-Wl,--gc-sections", "-Wl,--as-needed", "-Wl,-z,max-page-size=16384", "-Wl,-z,separate-loadable-segments", "-Wl,--allow-multiple-definition"]
    } else {
        &["-Wl,--gc-sections", "-Wl,--as-needed"]
    }
}

/// System libraries the runtime archive needs on the target.
pub fn native_libs(target: &TargetInfo) -> &'static [&'static str] {
    match target.os {
        TargetOs::Android => &["-ldl", "-llog", "-lunwind", "-lm", "-lc"],
        TargetOs::Ios | TargetOs::MacOs => &["-liconv", "-lc", "-lm", "-framework", "Accelerate"],
        TargetOs::Linux => &["-lgcc_s", "-lutil", "-lrt", "-lpthread", "-lm", "-ldl", "-lc"],
        _ => &[],
    }
}
