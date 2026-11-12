
/// Compiler version (the workspace crate version).
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub const VERSION_MAJOR: u32 = parse_u32(env!("CARGO_PKG_VERSION_MAJOR"));
pub const VERSION_MINOR: u32 = parse_u32(env!("CARGO_PKG_VERSION_MINOR"));
pub const VERSION_PATCH: u32 = parse_u32(env!("CARGO_PKG_VERSION_PATCH"));

const fn parse_u32(s: &str) -> u32 {
    let b = s.as_bytes();
    let mut i = 0;
    let mut v = 0u32;
    while i < b.len() {
        v = v * 10 + (b[i] - b'0') as u32;
        i += 1;
    }
    v
}

/// Typhon's own crate version.
pub const TYPHON_VERSION: &str = env!("CARGO_PKG_VERSION");
