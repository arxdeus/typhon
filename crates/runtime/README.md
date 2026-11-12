# typhon-runtime (libcodonrt)

## Garbage collector

bdwgc comes from the crates.io crate `bdwgc-alloc-sys` 0.6.16 (bdwgc 8.3 snapshot,
`cmake` feature). C++ Codon builds bdwgc v8.2.12 with `enable_large_config=ON`.
Two differences made Typhon collect far more often, and both are compensated:

1. **LARGE_CONFIG** (MINHINCR 64 / MAXHINCR 4096 heap increments). The crate offers
   no CMake options, so `.cargo/config.toml` sets `CFLAGS_<host triple>=-DLARGE_CONFIG`
   (cc/cmake-rs pick it up; `TARGET_CFLAGS` is ignored for native builds). Verify with
   `grep LARGE_CONFIG target/release/build/bdwgc-alloc-sys-*/out/build/CMakeFiles/gc.dir/flags.make`
   (needs `cargo clean -p bdwgc-alloc-sys` after changing it).
2. **Collect-after-growth.** 8.3's `GC_should_collect()` returns true whenever the
   heap grew since the last GC, so it collects after every expansion. `gc_init`
   (`src/lib_rt.rs`) installs `GC_set_stop_func(gc_growth_stop_func)`, which declines
   allocation-triggered collections until 8.2's `min_bytes_allocd`
   (`2 * in_use / free_space_divisor`) bytes were allocated, so the heap expands instead.

Measured 2026-11-12 (M-series, AOT `-release`, 5 runs, median):

| benchmark | Codon GCs / time | Typhon before | Typhon (1 only) | Typhon (1+2) |
|---|---|---|---|---|
| float 10000000 | 6 / 0.51 s | 41 / ~2.4 s | 22 / 1.66 s | 3 / 0.35 s |
| binary_trees 17 | 42 / 0.32 s | 84 / 0.40 s | 72 / 0.34 s | 41 / 0.31 s |

Peak RSS is the same as Codon on float (492 MB) and slightly higher on
binary_trees (56 vs 45 MB).
