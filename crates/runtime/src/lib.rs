//! libtyphonrt: exports the C ABI symbols that compiled Typhon programs (and the
//! stdlib's `from C import ...`) link against.
//!
//! | C++ source                  | Rust module        |
//! |-----------------------------|--------------------|
//! | `lib.h`                     | [`types`]          |
//! | `lib.cpp`                   | [`lib_rt`]         |
//! | `exc.cpp`                   | [`exc`]            |
//! | `re.cpp`                    | [`re`]             |
//! | `rtti.h`                    | [`rtti`]           |
//! | `numpy/{loops,sort,zmath}`  | [`numpy`]          |
//!
//! Third-party C ABIs bundled into the static archive: bdwgc
//! (`GC_*`), zlib (`gz*`, `inflate*`, ... via `libz-rs-sys`), bzip2 (`BZ2_*` via
//! `libbz2-rs-sys`), liblzma (`lzma_*`), plus libomp / Accelerate linked dynamically.
#![allow(clippy::missing_safety_doc)]

pub mod exc;
pub mod lib_rt;
pub mod numpy;
pub mod re;
pub mod rtti;
pub mod types;

pub mod gc;
pub use exc::make_backtrace_frame_string;

// Force the compression crates (pure-symbol providers) into the link.
pub use libbz2_rs_sys;
pub use liblzma_sys;
pub use libz_rs_sys;


