//! Typhon plugin C-ABI entry-point convention (see `PLUGINS.md`).
//!
//! Rust has no stable ABI for trait objects across dylibs, so a plugin dylib must be built
//! against the *same* `typhon-dsl` version with the *same* rustc. This is checked through an
//! exported ABI tag before any Rust-ABI value crosses the boundary.

use std::ffi::{CStr, c_char};

/// Bumped whenever the `Dsl` trait or the entry-point contract changes.
pub const ABI_VERSION: u32 = 1;

/// Symbol: `extern "C" fn typhon_plugin_abi_tag() -> *const c_char` (NUL-terminated, static).
pub const ABI_TAG_SYMBOL: &[u8] = b"typhon_plugin_abi_tag\0";
/// Symbol: `extern "C" fn typhon_plugin_load(abi_version: u32) -> *mut c_void`
/// returning `Box::into_raw(Box::new(Box<dyn Dsl>))`, or null on failure / ABI mismatch.
pub const LOAD_SYMBOL: &[u8] = b"typhon_plugin_load\0";

/// NUL-terminated ABI tag: typhon-dsl version, rustc version, ABI version.
pub const ABI_TAG_NUL: &str = concat!(
    "typhon-dsl ",
    env!("CARGO_PKG_VERSION"),
    "; ",
    env!("TYPHON_DSL_RUSTC_VERSION"),
    "; abi 1\0"
);

/// ABI tag without the trailing NUL.
pub fn abi_tag() -> &'static str {
    &ABI_TAG_NUL[..ABI_TAG_NUL.len() - 1]
}

/// Pointer for `typhon_plugin_abi_tag` implementations.
pub fn abi_tag_ptr() -> *const c_char {
    CStr::from_bytes_with_nul(ABI_TAG_NUL.as_bytes()).map(CStr::as_ptr).unwrap_or(std::ptr::null())
}

/// Exports the Typhon plugin entry points from a `cdylib` crate.
///
/// ```ignore
/// typhon_dsl::export_plugin!(MyDsl::default());
/// ```
#[macro_export]
macro_rules! export_plugin {
    ($ctor:expr) => {
        #[unsafe(no_mangle)]
        pub extern "C" fn typhon_plugin_abi_tag() -> *const ::std::ffi::c_char {
            $crate::abi::abi_tag_ptr()
        }

        #[unsafe(no_mangle)]
        pub extern "C" fn typhon_plugin_load(abi_version: u32) -> *mut ::std::ffi::c_void {
            if abi_version != $crate::abi::ABI_VERSION {
                return ::std::ptr::null_mut();
            }
            let dsl: ::std::boxed::Box<dyn $crate::Dsl> = ::std::boxed::Box::new($ctor);
            ::std::boxed::Box::into_raw(::std::boxed::Box::new(dsl)) as *mut ::std::ffi::c_void
        }
    };
}
