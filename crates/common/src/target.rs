//! Cross-compilation target description: `-target` triple / alias parsing,
//! per-target compile-time defines and runtime archive naming.

/// Operating system family of a target triple.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TargetOs {
    MacOs,
    Ios,
    Android,
    Linux,
    Windows,
    Other,
}

/// A parsed compilation target.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TargetInfo {
    /// LLVM triple, e.g. `aarch64-linux-android24`.
    pub triple: String,
    pub arch: String,
    pub os: TargetOs,
    /// `-simulator` environment (iOS simulator).
    pub simulator: bool,
}

/// Friendly aliases accepted by `-target`.
pub const TARGET_ALIASES: &[(&str, &str)] = &[
    ("android-arm64", "aarch64-linux-android24"),
    ("android-x64", "x86_64-linux-android24"),
    ("ios-arm64", "arm64-apple-ios13.0"),
    ("ios-sim-arm64", "arm64-apple-ios13.0-simulator"),
];

/// Resolve an alias to an LLVM triple (non-aliases are returned unchanged).
pub fn resolve_target_alias(t: &str) -> String {
    let t = t.trim();
    TARGET_ALIASES.iter().find(|(a, _)| *a == t).map(|(_, v)| v.to_string()).unwrap_or_else(|| t.to_string())
}

impl TargetInfo {
    /// Parse an LLVM triple or alias.
    pub fn parse(t: &str) -> TargetInfo {
        let triple = resolve_target_alias(t);
        let parts: Vec<&str> = triple.split('-').collect();
        let arch = parts.first().copied().unwrap_or("").to_string();
        let has = |p: &str| parts.iter().skip(1).any(|x| x.starts_with(p));
        let simulator = parts.iter().skip(1).any(|x| *x == "simulator");
        let os = if has("android") {
            TargetOs::Android
        } else if has("ios") {
            TargetOs::Ios
        } else if has("macos") || has("darwin") || has("macosx") {
            TargetOs::MacOs
        } else if has("windows") || has("win32") || has("mingw") {
            TargetOs::Windows
        } else if has("linux") {
            TargetOs::Linux
        } else {
            TargetOs::Other
        };
        TargetInfo { triple, arch, os, simulator }
    }

    /// Target of the compiler itself.
    pub fn host() -> TargetInfo {
        let os = if cfg!(target_os = "macos") {
            TargetOs::MacOs
        } else if cfg!(target_os = "ios") {
            TargetOs::Ios
        } else if cfg!(target_os = "android") {
            TargetOs::Android
        } else if cfg!(target_os = "linux") {
            TargetOs::Linux
        } else if cfg!(windows) {
            TargetOs::Windows
        } else {
            TargetOs::Other
        };
        TargetInfo { triple: String::new(), arch: std::env::consts::ARCH.into(), os, simulator: false }
    }

    /// `options.target` empty means the host.
    pub fn from_option(t: &str) -> TargetInfo {
        if t.is_empty() { Self::host() } else { Self::parse(t) }
    }

    pub fn is_apple(&self) -> bool {
        matches!(self.os, TargetOs::MacOs | TargetOs::Ios)
    }

    pub fn is_aarch64(&self) -> bool {
        matches!(self.arch.as_str(), "aarch64" | "arm64" | "arm64e")
    }

    /// Compile-time defines `__apple__`, `__windows__`, `__linux__`, `__android__`,
    /// `__ios__`, `__macos__` (`__linux__` is true for Android too, like the C macro).
    pub fn defines(&self) -> Vec<(&'static str, bool)> {
        vec![
            ("__apple__", self.is_apple()),
            ("__windows__", self.os == TargetOs::Windows),
            ("__linux__", matches!(self.os, TargetOs::Linux | TargetOs::Android)),
            ("__android__", self.os == TargetOs::Android),
            ("__ios__", self.os == TargetOs::Ios),
            ("__macos__", self.os == TargetOs::MacOs),
        ]
    }

    /// Rust target triple naming the runtime archive directory.
    pub fn rust_triple(&self) -> Option<String> {
        let arch = if self.is_aarch64() { "aarch64" } else { self.arch.as_str() };
        Some(match self.os {
            TargetOs::Android => format!("{arch}-linux-android"),
            TargetOs::Ios if self.simulator && arch == "x86_64" => "x86_64-apple-ios".into(),
            TargetOs::Ios if self.simulator => format!("{arch}-apple-ios-sim"),
            TargetOs::Ios => format!("{arch}-apple-ios"),
            TargetOs::MacOs => format!("{arch}-apple-darwin"),
            TargetOs::Linux => format!("{arch}-unknown-linux-gnu"),
            _ => return None,
        })
    }

    /// Android API level from the triple (`...-android24` -> 24, default 24).
    pub fn android_api(&self) -> u32 {
        self.triple.rsplit('-').next().and_then(|e| e.strip_prefix("android")).and_then(|v| v.parse().ok()).unwrap_or(24)
    }

    /// Default CPU / features for a non-host target.
    pub fn default_cpu_features(&self) -> (String, Vec<String>) {
        match (self.os, self.is_aarch64()) {
            (TargetOs::Ios | TargetOs::MacOs, true) => ("apple-a12".into(), vec![]),
            (_, true) => ("generic".into(), vec!["+neon".into()]),
            (_, false) if self.arch == "x86_64" => ("x86-64".into(), vec!["+sse4.2".into(), "+popcnt".into()]),
            _ => ("generic".into(), vec![]),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aliases() {
        assert_eq!(TargetInfo::parse("android-arm64").triple, "aarch64-linux-android24");
        assert_eq!(TargetInfo::parse("android-x64").rust_triple().unwrap(), "x86_64-linux-android");
        let s = TargetInfo::parse("ios-sim-arm64");
        assert!(s.simulator && s.os == TargetOs::Ios);
        assert_eq!(s.rust_triple().unwrap(), "aarch64-apple-ios-sim");
        assert_eq!(TargetInfo::parse("arm64-apple-ios13.0").rust_triple().unwrap(), "aarch64-apple-ios");
        assert_eq!(TargetInfo::parse("aarch64-linux-android30").android_api(), 30);
    }

    #[test]
    fn defines() {
        let d = |t: &str| TargetInfo::parse(t).defines().into_iter().filter(|x| x.1).map(|x| x.0).collect::<Vec<_>>();
        assert_eq!(d("aarch64-linux-android24"), vec!["__linux__", "__android__"]);
        assert_eq!(d("arm64-apple-ios13.0"), vec!["__apple__", "__ios__"]);
        assert_eq!(d("arm64-apple-macosx14.0"), vec!["__apple__", "__macos__"]);
        assert_eq!(d("x86_64-unknown-linux-gnu"), vec!["__linux__"]);
        assert_eq!(d("x86_64-pc-windows-msvc"), vec!["__windows__"]);
    }
}
