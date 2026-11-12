//! //! Host CPU / feature detection and the `ArchNativePass` that stamps
//! `target-cpu` / `target-features` / `frame-pointer` on every function.

use inkwell::module::Module;

use crate::{ffi, target};

/// Parsed pieces of an LLVM triple that the target tables need
/// (subset of `llvm::Triple`).
#[derive(Clone, Debug, Default)]
pub struct Triple {
    pub arch_name: String,
    pub vendor: String,
    pub os: String,
    pub env: String,
}

impl Triple {
    pub fn parse(s: &str) -> Self {
        let mut it = s.split('-');
        Triple {
            arch_name: it.next().unwrap_or("").to_string(),
            vendor: it.next().unwrap_or("").to_string(),
            os: it.next().unwrap_or("").to_string(),
            env: it.collect::<Vec<_>>().join("-"),
        }
    }
    pub fn os_name(&self) -> String {
        self.os.trim_end_matches(|c: char| c.is_ascii_digit() || c == '.').to_string()
    }
    pub fn is_aarch64(&self) -> bool {
        matches!(self.arch_name.as_str(), "aarch64" | "aarch64_be" | "aarch64_32" | "arm64" | "arm64_32" | "arm64e")
    }
    pub fn is_arm(&self) -> bool {
        let a = self.arch_name.as_str();
        (a.starts_with("arm") || a.starts_with("thumb")) && !self.is_aarch64()
    }
    pub fn is_x86(&self) -> bool {
        matches!(self.arch_name.as_str(), "i386" | "i486" | "i586" | "i686" | "x86" | "x86_64" | "x86_64h" | "amd64")
    }
    pub fn is_x86_64(&self) -> bool {
        matches!(self.arch_name.as_str(), "x86_64" | "x86_64h" | "amd64")
    }
    pub fn is_os_linux(&self) -> bool {
        self.os_name() == "linux"
    }
    pub fn is_os_darwin(&self) -> bool {
        matches!(self.os_name().as_str(), "darwin" | "macosx" | "macos" | "ios" | "tvos" | "watchos" | "xros" | "driverkit")
    }
    pub fn is_macosx(&self) -> bool {
        matches!(self.os_name().as_str(), "darwin" | "macosx" | "macos")
    }
    pub fn is_android(&self) -> bool {
        self.env.starts_with("android")
    }
    pub fn is_ohos(&self) -> bool {
        self.env.starts_with("ohos") || self.os_name() == "liteos"
    }
    /// `Triple::getOSVersion` major/minor for macOS (darwin N -> macOS 10.(N-4)).
    fn macos_version(&self) -> (u32, u32) {
        let digits: String = self.os.chars().skip_while(|c| !c.is_ascii_digit()).collect();
        let mut parts = digits.split('.').map(|p| p.parse::<u32>().unwrap_or(0));
        let major = parts.next().unwrap_or(0);
        let minor = parts.next().unwrap_or(0);
        if self.os_name() == "darwin" {
            if major == 0 {
                return (10, 4);
            }
            if major < 20 { (10, major.saturating_sub(4)) } else { (major - 9, 0) }
        } else if major == 0 {
            (10, 4)
        } else {
            (major, minor)
        }
    }
}

/// Host target description.
pub trait NativeTarget {
    fn cpu(&self, triple: &Triple) -> String;
    fn features(&self, triple: &Triple) -> String;
}

fn host_feature_list() -> Vec<String> {
    // `llvm::sys::getHostCPUFeatures()` as "+a,-b,..."
    target::host_cpu_features().split(',').filter(|s| !s.is_empty()).map(str::to_string).collect()
}

unsafe extern "C" {
    /// `cpp/tm_shim.cpp`: TargetParser default feature list for an AArch64 CPU.
    fn typhon_aarch64_cpu_default_features(cpu: *const libc::c_char, buf: *mut libc::c_char, cap: libc::c_ulong) -> libc::c_long;
}

/// `None` if LLVM does not know the CPU (mirrors `if (!cpuInfo) return "";`).
fn aarch64_cpu_default_features(cpu: &str) -> Option<Vec<String>> {
    let c = std::ffi::CString::new(cpu).ok()?;
    let mut buf = vec![0u8; 4096];
    loop {
        let n = unsafe { typhon_aarch64_cpu_default_features(c.as_ptr(), buf.as_mut_ptr().cast(), buf.len() as _) };
        if n < 0 {
            return None;
        }
        let n = n as usize;
        if n < buf.len() {
            let s = std::str::from_utf8(&buf[..n]).ok()?;
            return Some(s.split(',').filter(|s| !s.is_empty()).map(str::to_string).collect());
        }
        buf.resize(n + 1, 0);
    }
}

/// Port of `targets/aarch64.cpp`.
pub struct Aarch64;

impl NativeTarget for Aarch64 {
    fn cpu(&self, _triple: &Triple) -> String {
        target::host_cpu_name()
    }

    fn features(&self, triple: &Triple) -> String {
        let cpu = target::host_cpu_name();
        // `AArch64::parseCpu` + `addCPUDefaults/addArchDefaults` via cpp/tm_shim.cpp.
        // These defaults matter: on macOS `getHostCPUFeatures()` is empty, so without
        // them functions get no NEON/v8.x features and run several times slower.
        let Some(mut features) = aarch64_cpu_default_features(&cpu) else {
            return String::new();
        };
        features.extend(host_feature_list());
        if cpu == "cyclone" || cpu.starts_with("apple") {
            features.push("+zcm".into());
            features.push("+zcz".into());
        }
        if triple.is_android() || triple.is_ohos() {
            features.push("+fix-cortex-a53-835769".into());
        } else if triple.os_name() == "fuchsia" && (cpu.is_empty() || cpu == "generic" || cpu == "cortex-a53") {
            features.push("+fix-cortex-a53-835769".into());
        }
        if triple.os_name() == "openbsd" {
            features.push("+strict-align".into());
        }
        features.join(",")
    }
}

/// Port of `targets/x86.cpp`.
pub struct X86;

impl NativeTarget for X86 {
    fn cpu(&self, triple: &Triple) -> String {
        let cpu = target::host_cpu_name();
        if !cpu.is_empty() && cpu != "generic" {
            return cpu;
        }
        if !triple.is_x86() {
            return String::new();
        }
        let is64 = triple.is_x86_64();
        if triple.is_os_darwin() {
            if triple.arch_name == "x86_64h" {
                return "core-avx2".into();
            }
            if triple.is_macosx() && triple.macos_version() >= (10, 12) {
                return "penryn".into();
            }
            if triple.os_name() == "driverkit" {
                return "nehalem".into();
            }
            return if is64 { "core2" } else { "yonah" }.into();
        }
        if triple.os_name() == "ps4" {
            return "btver2".into();
        }
        if triple.os_name() == "ps5" {
            return "znver2".into();
        }
        if triple.is_android() {
            return if is64 { "x86-64" } else { "i686" }.into();
        }
        if is64 {
            return "x86-64".into();
        }
        match triple.os_name().as_str() {
            "netbsd" => "i486",
            "haiku" | "openbsd" => "i586",
            "freebsd" => "i686",
            _ => "pentium4",
        }
        .into()
    }

    fn features(&self, triple: &Triple) -> String {
        let mut features = host_feature_list();
        if triple.arch_name == "x86_64h" {
            for f in ["-rdrnd", "-aes", "-pclmul", "-rtm", "-fsgsbase"] {
                features.push(f.into());
            }
        }
        if triple.is_android() {
            if triple.is_x86_64() {
                for f in ["+sse4.2", "+popcnt", "+cx16"] {
                    features.push(f.into());
                }
            } else {
                features.push("+ssse3".into());
            }
        }
        features.join(",")
    }
}

/// Port of `targets/arm.cpp`.
pub struct Arm;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum FloatAbi {
    Invalid,
    Hard,
    Soft,
    SoftFp,
}

impl Arm {
    /// `ARM::parseArchVersion(triple.getArchName())`.
    fn sub_arch_version(triple: &Triple) -> u32 {
        let a = triple.arch_name.trim_start_matches("arm").trim_start_matches("thumb").trim_start_matches("eb");
        let a = a.trim_start_matches('v');
        a.chars().take_while(|c| c.is_ascii_digit()).collect::<String>().parse().unwrap_or(0)
    }

    fn arch_suffix(triple: &Triple) -> String {
        let a = triple.arch_name.as_str();
        let s = a.trim_start_matches("armeb").trim_start_matches("thumbeb").trim_start_matches("arm").trim_start_matches("thumb");
        s.trim_start_matches('v').to_string()
    }

    fn is_m_profile(triple: &Triple) -> bool {
        let s = Self::arch_suffix(triple);
        s.ends_with('m') || s.contains("m.") || s.contains("m-") || s.contains(".main") || s.contains(".base") || s == "6m" || s == "7em"
    }

    fn use_aapcs_for_macho(triple: &Triple) -> bool {
        matches!(triple.env.as_str(), "eabi" | "eabihf") || triple.os.is_empty() || triple.os == "unknown" || Self::is_m_profile(triple)
    }

    fn default_float_abi(triple: &Triple) -> FloatAbi {
        let sub = Self::sub_arch_version(triple);
        match triple.os_name().as_str() {
            "darwin" | "macosx" | "macos" | "ios" | "tvos" | "driverkit" | "xros" => {
                if triple.arch_name.ends_with("7k") {
                    FloatAbi::Hard
                } else if sub == 6 || sub == 7 {
                    FloatAbi::SoftFp
                } else {
                    FloatAbi::Soft
                }
            }
            "watchos" => FloatAbi::Hard,
            "win32" | "windows" => {
                if triple.vendor == "apple" && !Self::use_aapcs_for_macho(triple) {
                    FloatAbi::Soft
                } else {
                    FloatAbi::Hard
                }
            }
            "netbsd" => match triple.env.as_str() {
                "eabihf" | "gnueabihf" => FloatAbi::Hard,
                _ => FloatAbi::Soft,
            },
            "freebsd" => match triple.env.as_str() {
                "gnueabihf" => FloatAbi::Hard,
                _ => FloatAbi::Soft,
            },
            "haiku" | "openbsd" => FloatAbi::SoftFp,
            _ => {
                if triple.is_ohos() {
                    return FloatAbi::Soft;
                }
                match triple.env.as_str() {
                    "gnueabihf" | "gnueabihft64" | "musleabihf" | "eabihf" => FloatAbi::Hard,
                    "android" | "gnueabi" | "gnueabit64" | "musleabi" | "eabi" => FloatAbi::SoftFp,
                    e if e.starts_with("android") => FloatAbi::SoftFp,
                    _ => FloatAbi::Invalid,
                }
            }
        }
    }

    fn float_abi(triple: &Triple) -> FloatAbi {
        match Self::default_float_abi(triple) {
            FloatAbi::Invalid => {
                if triple.is_os_darwin() && Self::arch_suffix(triple) == "7em" {
                    FloatAbi::Hard
                } else {
                    FloatAbi::Soft
                }
            }
            abi => abi,
        }
    }

    fn has_integer_mve(features: &[String]) -> bool {
        let mve = features.iter().rposition(|f| f == "+mve");
        let no = features.iter().rposition(|f| f == "-mve");
        match (mve, no) {
            (Some(m), Some(n)) => m > n,
            (Some(_), None) => true,
            _ => false,
        }
    }
}

impl NativeTarget for Arm {
    fn cpu(&self, _triple: &Triple) -> String {
        target::host_cpu_name()
    }

    fn features(&self, triple: &Triple) -> String {
        let mut features: Vec<String> = Vec::new();
        let abi = Self::float_abi(triple);
        if abi == FloatAbi::Soft {
            features.push("+soft-float".into());
        }
        if abi != FloatAbi::Hard {
            features.push("+soft-float-abi".into());
        }
        features.extend(host_feature_list());

        // PORT-NOTE: `ARM::getDefaultFPU` / `ARM::getFPUFeatures` (TargetParser FPU
        // tables) are unavailable through the C API. The host feature list already
        // contains the FPU features (vfp*, neon, d32, fp16...), so the FPU kind is
        // derived from it: no "+vfp2"/"+fp-armv8"/"+neon"-style feature means FK_NONE.
        let fpu_none = !features.iter().any(|f| {
            f.starts_with("+vfp") || f.starts_with("+fp-armv8") || f == "+neon" || f.starts_with("+fpregs")
        });
        let fpu_fpv5_d16 = false;

        let ri = |fs: &Vec<String>, name: &str| fs.iter().rposition(|f| f == name);
        let no_full = ri(&features, "-fullfp16");
        let fml = ri(&features, "+fp16fml");
        // reverse-iterator ordering: "a < b" in C++ rbegin space means a is later.
        let later = |a: Option<usize>, b: Option<usize>| match (a, b) {
            (Some(x), Some(y)) => x > y,
            (Some(_), None) => true,
            _ => false,
        };
        let v84 = Self::arch_suffix(triple).starts_with("8.4");
        let mut fallthrough = !v84;
        if v84 {
            let full = ri(&features, "+fullfp16");
            if full.is_some() && later(full, no_full) && later(full, fml) {
                let after = full.map(|i| features[i + 1..].iter().any(|f| f == "-fp16fml")).unwrap_or(false);
                if !after {
                    features.push("+fp16fml".into());
                }
            } else {
                fallthrough = true;
            }
        }
        if fallthrough {
            if later(no_full, fml) {
                features.push("-fp16fml".into());
            } else if later(fml, no_full) {
                features.push("+fullfp16".into());
            }
        }

        let mut has_fp_regs = true;
        if abi == FloatAbi::Soft {
            for f in ["-vfp2", "-vfp3", "-vfp4", "-fp-armv8", "-fullfp16", "-neon", "-crypto"] {
                features.push(f.into());
            }
            for f in ["-dotprod", "-fp16fml", "-bf16", "-mve", "-mve.fp"] {
                features.push(f.into());
            }
            has_fp_regs = false;
        } else if fpu_none {
            for f in ["-dotprod", "-fp16fml", "-bf16", "-mve.fp"] {
                features.push(f.into());
            }
            has_fp_regs = Self::has_integer_mve(&features);
        }
        if !has_fp_regs {
            features.push("-fpregs".into());
        }
        if fpu_fpv5_d16 {
            features.push("-mve.fp".into());
        }

        let last = |fs: &Vec<String>, p: &dyn Fn(&str) -> bool| fs.iter().rev().find(|f| p(f)).cloned();
        let crypto = last(&features, &|f| f.contains("crypto"));
        let sha2 = last(&features, &|f| f.contains("crypto") || f.contains("sha2"));
        let aes = last(&features, &|f| f.contains("crypto") || f.contains("aes"));
        let has_sha2 = sha2.as_deref().is_some_and(|f| f.starts_with('+'));
        let has_aes = aes.as_deref().is_some_and(|f| f.starts_with('+'));
        if crypto.is_some() {
            features.push(if has_sha2 && has_aes { "+crypto" } else { "-crypto" }.into());
            features.push(if has_sha2 { "+sha2" } else { "-sha2" }.into());
            features.push(if has_aes { "+aes" } else { "-aes" }.into());
        }
        if has_sha2 || has_aes {
            let suffix = Self::arch_suffix(triple);
            let version = Self::sub_arch_version(triple);
            let profile_ok = !Self::is_m_profile(triple);
            if !(version >= 8 && profile_ok && !suffix.is_empty()) {
                features.push("-sha2".into());
                features.push("-aes".into());
            }
        }

        let version = Self::sub_arch_version(triple);
        let suffix = Self::arch_suffix(triple);
        if triple.is_os_darwin() || triple.os_name() == "netbsd" {
            if version < 6 || suffix == "6m" {
                features.push("+strict-align".into());
            }
        } else if triple.vendor == "apple" {
            features.push("+strict-align".into());
        } else if version < 7 || suffix == "6m" || suffix.starts_with("8m.base") {
            features.push("+strict-align".into());
        }
        features.join(",")
    }
}

/// Port of `getNativeTarget(triple)`.
pub fn native_target(triple: &Triple) -> Option<Box<dyn NativeTarget>> {
    if triple.is_aarch64() {
        Some(Box::new(Aarch64))
    } else if triple.is_arm() {
        Some(Box::new(Arm))
    } else if triple.is_x86() {
        Some(Box::new(X86))
    } else {
        None
    }
}

/// CPU and features for native compilation on the host (`addNativeLLVMPasses`
/// computes these from `EngineBuilder().selectTarget()->getTargetTriple()`, the host).
pub fn native_cpu_and_features(triple: &str) -> Option<(String, String)> {
    let t = Triple::parse(triple);
    let target = native_target(&t)?;
    Some((target.cpu(&t), target.features(&t)))
}

/// Port of `ArchNativePass::run` applied to every function in the module.
pub fn apply_native_attributes(module: &Module, cpu: &str, features: &str) {
    let m = module.as_mut_ptr();
    let triple = Triple::parse(&module.get_triple().as_str().to_string_lossy());
    let frame_pointer = if triple.is_aarch64() && triple.is_os_linux() { "non-leaf" } else { "none" };
    for f in ffi::functions(m) {
        if ffi::is_declaration(f) {
            // Function passes only visit definitions.
            continue;
        }
        if !cpu.is_empty() {
            ffi::add_fn_string_attr(f, "target-cpu", cpu);
        }
        if !features.is_empty() {
            ffi::add_fn_string_attr(f, "target-features", features);
        }
        ffi::add_fn_string_attr(f, "frame-pointer", frame_pointer);
    }
}

/// Port of `addNativeLLVMPasses`: host CPU/features applied to the module.
pub fn add_native_llvm_passes(module: &Module) {
    if let Some((cpu, features)) = native_cpu_and_features(&target::host_triple()) {
        apply_native_attributes(module, &cpu, &features);
    }
}
