# typhon-runtime (libtyphonrt)

## Linking

The runtime is built **only as a static library** (`crate-type = ["rlib", "staticlib"]`;
`cargo build --release` puts `target/release/libtyphonrt.a` next to `typhon`). There is no
shared `libtyphonrt` and no JIT: every program is compiled ahead of time and the
archive is linked into it.

* `typhon build -exe` and `typhon run` (which builds into a temp dir and execs the binary)
  produce a single self-contained executable. `typhon build -lib` produces a shared
  library that embeds its own copy of the runtime (two such libraries in one process
  would mean two GCs).
* The archive is located like this: `TYPHON_LIB_DIR`, else the directory of the `typhon`
  binary, `../lib`, `../lib/typhon`. Install layout: `bin/typhon` + `lib/typhon/libtyphonrt.a`
  (or both in one directory).
* Link line (`crates/llvm/src/llvisitor/output.rs`, `RUNTIME_NATIVE_LIBS` and
  `LINK_GC_FLAGS`), after the program object and user `-l` libs, then `-linker-flags`:
  * macOS: `c++ prog.o -L<dirs> -l<user> -Wl,-dead_strip -Wl,-dead_strip_dylibs
    <dir>/libtyphonrt.a -liconv -lm -framework Accelerate -o prog`
  * Linux: `c++ prog.o -L<dirs> -l<user> -Wl,--gc-sections -Wl,--as-needed
    <dir>/libtyphonrt.a -lgcc_s -lutil -lrt -lpthread -lm -ldl -lc -o prog -no-pie`
  The system libs are what `cargo rustc -p typhon-runtime --release --crate-type staticlib
  -- --print native-static-libs` reports (macOS: `-liconv -lSystem -lc -lm -framework
  Accelerate`). zlib, bzip2, liblzma, bdwgc and OpenMP (`typhon-omp`) are inside the
  archive; the C++ runtime is not needed (`-dead_strip_dylibs` / `--as-needed` drop the
  `libc++` the `c++` driver adds). A stripped hello world is ~0.5 MB and depends only on
  libSystem (macOS).
* Dead stripping is safe because the compiled program references runtime functions
  directly (`from C import f(...)` without a library is a plain external symbol). What
  `dlsym(RTLD_DEFAULT, ...)` can reach at run time is only the program's *exported*
  symbols: `from C import "lib".f` / `internal.dlopen.dlsym` with an empty library name
  does not find runtime functions in an executable (it does find libc). Exporting the
  whole runtime ABI (`-exported_symbols_list`, ~730 symbols) would grow hello world
  from 0.5 to 1.2 MB; nothing in the stdlib needs it.
* `.cargo/config.toml` sets `MACOSX_DEPLOYMENT_TARGET=11.0` so the archive's C objects
  (bdwgc, liblzma, omp shim) match rustc's baseline and `ld` does not warn about objects
  "built for newer macOS".


## Garbage collector

bdwgc comes from the crates.io crate `bdwgc-alloc-sys` 0.6.16 (bdwgc 8.3 snapshot,
`cmake` feature). The C++ baseline builds bdwgc v8.2.12 with `enable_large_config=ON`.
Two differences made Typhon collect far more often, and both are compensated:

1. **LARGE_CONFIG** (MINHINCR 64 / MAXHINCR 4096 heap increments). The crate offers
   no CMake options, so `.cargo/config.toml` sets `CFLAGS_<host triple>=-DLARGE_CONFIG`
   (cc/cmake-rs pick it up; `TARGET_CFLAGS` is ignored for native builds). Verify with
   `grep LARGE_CONFIG target/release/build/bdwgc-alloc-sys-*/out/build/CMakeFiles/gc.dir/flags.make`
   (needs `cargo clean -p bdwgc-alloc-sys` after changing it).
2. **Collect-after-growth.** 8.3's `GC_should_collect()` returns true whenever the
   heap grew since the last GC, so it collects after every expansion. `gc_init`
   (`src/lib_rt.rs`) installs `GC_set_stop_func(gc_growth_stop_func)`, which declines
   only the forced post-growth collection (heap size changed since the last GC it
   allowed) and only while fewer than `in_use / (2 * free_space_divisor)` bytes were
   allocated since the last GC. Ordinary threshold GCs (8.3's own `min_bytes_allocd`,
   which counts atomic data at 1/4 like 8.2) are never declined. An earlier version
   declined every GC below `2 * in_use / divisor`, which kept the heap ~2x the baseline.

`TYPHON_GC_STATS=1` prints collection count and heap stats at exit. Parallel
marking, `GC_MARKERS`, initial heap size and unmapping made no measurable RSS
difference; `GC_FREE_SPACE_DIVISOR` still works as a runtime override.

### Memory policy result (2026-11-12, M-series, `-release`, median of 3-5)

| benchmark | Baseline RSS / time / GCs | Typhon before (1a148483) | Typhon now |
|---|---|---|---|
| binary_trees 17 | 45 MB / 0.31 s / 42 | 56 MB / 0.30 s / 41 | 36 MB / 0.34 s / 72 |
| binary_trees 20 | 264 MB / 3.71 s / 111 | 540 MB / 2.98 s | 263 MB / 3.66 s / 127 |
| chaos | 56 MB / 0.72 s / 416 | 143 MB / 0.63 s / 31 | 57 MB / 0.71 s / 357 |
| float 10000000 | 493 MB / 0.40 s / 6 | 492 MB / 0.34 s / 3 | 492 MB / 0.38 s / 9 |
| spectral_norm 40 | 13 MB / 0.09 s | 15 MB / 0.08 s | 11 MB / 0.08 s |
| fannkuch, go, mandelbrot, nbody, primes, set_partition | 7-13 MB | 0.5 MB below baseline | unchanged, 0.5 MB below baseline |

Runtimes now track the baseline (within noise); the old policy was faster on GC-heavy
programs only because it let the heap grow ~2x.

Measured 2026-11-12 (M-series, AOT `-release`, 5 runs, median):

| benchmark | Baseline GCs / time | Typhon before | Typhon (1 only) | Typhon (1+2) |
|---|---|---|---|---|
| float 10000000 | 6 / 0.51 s | 41 / ~2.4 s | 22 / 1.66 s | 3 / 0.35 s |
| binary_trees 17 | 42 / 0.32 s | 84 / 0.40 s | 72 / 0.34 s | 41 / 0.31 s |

Peak RSS is the same as the baseline on float (492 MB) and slightly higher on
binary_trees (56 vs 45 MB).
