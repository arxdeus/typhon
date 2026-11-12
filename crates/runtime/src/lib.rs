//! Port of codon/runtime/** (libtyphonrt). Exports the same C ABI symbols as the C++
//! runtime so compiled Codon programs (and the stdlib's `from C import ...`) can link
//! against it.
//!
//! | C++ source                  | Rust module        |
//! |-----------------------------|--------------------|
//! | `lib.h`                     | [`types`]          |
//! | `lib.cpp`                   | [`lib_rt`]         |
//! | `exc.cpp`                   | [`exc`]            |
//! | `re.cpp`                    | [`re`]             |
//! | `rtti.h`                    | [`rtti`]           |
//! | `emutls.cpp`                | [`emutls`]         |
//! | `numpy/{loops,sort,zmath}`  | [`numpy`]          |
//!
//! Third-party C ABIs re-exported from the cdylib (as the C++ build does): bdwgc
//! (`GC_*`), zlib (`gz*`, `inflate*`, ... via `libz-rs-sys`), bzip2 (`BZ2_*` via
//! `libbz2-rs-sys`), liblzma (`lzma_*`), plus libomp / Accelerate linked dynamically.
#![allow(clippy::missing_safety_doc)]

pub mod emutls;
pub mod exc;
pub mod lib_rt;
pub mod numpy;
pub mod re;
pub mod rtti;
pub mod types;

pub mod gc;
pub use exc::{JitError, make_backtrace_frame_string, set_jit_error_callback};
pub use lib_rt::get_captured_output;

// Force the compression crates (pure-symbol providers) into the link.
pub use libbz2_rs_sys;
pub use liblzma_sys;
pub use libz_rs_sys;

#[cfg(feature = "openblas")]
extern crate openblas_src;
