# armeabi-v7a (armv7a-linux-androideabi24) feasibility

This was a 45-minute recon on 2026-11-14. No compiler or runtime changes were committed. Scratch work is in
`~/.jcode/scratch/armv7/`, with a separate `CARGO_TARGET_DIR` of `~/.jcode/scratch/armv7-target`.

**Short answer:** the target builds and links, but it is not correct yet. The whole BCI app
(`bci_preprocess.py` + `preprocessing.py`) compiled and linked into a 32-bit ARM `.so` (10 MB) after two
manual workarounds. That `.so` would still crash or give wrong results at runtime. The reasons are the
`size_t` ABI mismatches and the exception unwinding format, which is EHABI on ARM32 instead of DWARF.
None of this was executed: no AArch32 machine was available.

## Probe log (commands and first error)

1. `rustup target add armv7-linux-androideabi`: OK.
2. `CARGO_TARGET_DIR=... tools/build-runtime.sh armv7-linux-androideabi`
   * The script picked NDK 28.2. NDK 30.0 has no `toolchains/llvm/prebuilt/darwin-x86_64/bin`, so it is unusable.
   * First error, in the OpenBLAS build: `No rule to make target ../kernel/arm/amax.S`. `build.rs` passes no
     `TARGET=` for arm, and with `BINARY=64` OpenBLAS tries an armv8 build (`libopenblas_armv8p`).
   * With `TYPHON_OPENBLAS_DIR=<stub>`, all of the Rust code builds with zero errors: runtime, bdwgc, typhon-omp
     and numpy kernels. Output: `libtyphonrt.a`.
3. OpenBLAS by hand: `make libs netlib TARGET=ARMV7 BINARY=32 INTERFACE64=1 ARM_SOFTFP_ABI=1 NOFORTRAN=1 ...
   CC=armv7a-linux-androideabi24-clang` builds `libopenblas_armv7p` (14 MB, ILP64 interface, C_LAPACK) in under 1 minute.
   I tested `USE_OPENMP=0` only.
4. `typhon build -release -lib -target armv7a-linux-androideabi24 hello.py` with `TYPHON_LIB_DIR` set
   (an int, list and try/except test): OK, `ELF 32-bit ARM EABI5`, with `.ARM.exidx/.ARM.extab`. The target name
   becomes `armv7a-linux-android` in the runtime lookup path.
5. The numpy test (`arange/sin/exp/@`) and the real app both fail with:
   `LLVM ERROR: Cannot select: i32 = setcc (i64 build_pair ...), Constant:i64<0>` (util.py:793).
   * Cause: `llvm.maximum.f64` and `llvm.minimum.f64` are broken in the ARM backend. I reproduced it with
     a 3-line `.ll` and LLVM 22 `llc`.
   * Workaround: replace them with `maxnum`/`minnum` and run `llc -relocation-model=pic`. After that, the
     app object links against `libtyphonrt.a` plus ARMV7 OpenBLAS with `--no-undefined`.
   * All 10+ `bci_*` exports are present.

## What breaks

| Component | Status | Fix | Effort |
|---|---|---|---|
| Target parse / triple mapping | works (`armv7a-...androideabi24` -> runtime dir `armv7a-linux-android`, rust triple `armv7-linux-androideabi`) | add `android-arm` alias in `common/src/target.rs`; make build-runtime output dir match | 1 h |
| NDK 30.0 | broken install (no host bin dir) | use 28.2 / reinstall | 0 |
| Runtime Rust + bdwgc + omp | **builds** | none for compile; see ABI rows | 0 |
| OpenBLAS in `runtime/build.rs` | breaks (`kernel/arm/amax.S`) | add an arm match arm: `TARGET=ARMV7 BINARY=32 ARM_SOFTFP_ABI=1`, keep `INTERFACE64=1` (compiler passes i64 to cblas/LAPACK, so ILP64 is the right match); verify `USE_OPENMP=1` with typhon-omp | 2-4 h |
| `llvm.maximum/minimum.f64` | **LLVM crash** in ISel | in stdlib `numpy/util.py` `fmax64/fmin64`: use select+fcmp NaN logic (or maxnum + NaN check) when not 64-bit | 2 h |
| PIC | the `.o` from `-llvm` + llc needed `-relocation-model=pic` (typhon's own `-lib` path sets it; check) | confirm in the `-lib` path | 0.5 h |
| `size_t` ABI: runtime `seq_alloc*(usize)` | works by accident (i64 low half lands in r0) | change Rust sigs to `i64` (SeqInt) or emit `intptr` | 1 h |
| `seq_realloc(ptr, i64, i64)` vs Rust `(ptr, usize, usize)` | **breaks**: AAPCS puts the i64 in r2:r3, so the callee reads newsize from r1 (garbage) | Rust side uses `i64` args | 0.5 h |
| libc `strlen` declared `-> i64`, `memcmp/memchr(.., i64)` | **strlen breaks** (high word r1 is garbage); memcmp/memchr work by accident | compiler: declare libc size_t fns with target `intptr` type and zext/trunc. Grep all `@llvm`/`declare` in stdlib for `size_t` | 1-2 d |
| Other runtime externs | most use `SeqInt`=i64 on both sides (`seq_time`, `seq_pid`, `{ptr,i64}` str), so OK | audit: 93 runtime and 81 omp `extern "C"`. Only 7+5 have usize in the signature (5 omp ones are `*mut usize` locks, fine) | 2 h |
| Codegen pointer assumptions | IR uses datalayout `p:32`. `seq_exc_offset(ptr_size)` is already parameterised. 607 `ptrtoint/i64 8` sites in the app IR were not audited | audit `Ptr[byte]` arithmetic, `__ptrsize__`-like constants, `sizeof` in stdlib `.py` `@llvm` blocks; run testsuite under qemu | 2-4 d |
| **Exceptions (exc.rs)** | **will break at runtime** (cannot test) | see below | 2-3 d |
| numpy `hmath.rs` | only `cfg(aarch64 \| fma)` mul_add selection; falls back to `a*b+c`. Builds | none (maybe `+neon,+vfpv4` for FMA) | 0 |
| BLAS/LAPACK need | app IR references cblas `dcopy/ddot/dgemv/dsyrk/dgemm/zcopy` and LAPACK `dgeev/dgelsd/dgesdd/dsyevd/dsyevr/dsyevx/dsygvd/dsygvx` (from LDA fit and the numpy linalg in the closure). IIR sosfiltfilt alone does not need them | ship ARMV7 OpenBLAS (~3-5 MB stripped) | incl. above |
| Flutter build hook / gradle abiFilters | not checked | add `armeabi-v7a` to `hook/build.dart` + abiFilters | 2 h |
| Testing | none possible on hand | see below | 1-2 d setup |

### Exceptions: what must change (ARM EHABI)
* `UnwindException` must become `_Unwind_Control_Block`. It is 88 bytes and 8-aligned:
  `char exception_class[8]; cleanup; unwinder_cache[5]; barrier_cache{sp,bitpattern[5]};
  cleanup_cache[4]; pr_cache{fnstart,ehtp,additional}`.
  `exception_class` stays 8 bytes but is compared as bytes. Make the `align(16)` conditional and keep
  `seq_exc_offset` in llvisitor/mod.rs:76 consistent (header 12 B, aligned to 16 or 8).
* The `seq_personality` signature on ARM is `(_Unwind_State state, _Unwind_Control_Block*, _Unwind_Context*)`.
  You have to map the state (`US_VIRTUAL_UNWIND_FRAME` = search, `US_UNWIND_FRAME_STARTING` = cleanup,
  `US_UNWIND_FRAME_RESUME`) to actions.
  * Phase-1 handler found: set `barrier_cache.sp` (= `_Unwind_GetGR(ctx, 13)`).
  * Phase 2: compare against the saved barrier.
  * Before returning `_URC_CONTINUE_UNWIND`, you must call `__gnu_unwind_frame(ucbp, ctx)`.
  * The LSDA comes from `_Unwind_GetLanguageSpecificData`, which works on EHABI because LLVM emits the
    `.ARM.extab` personality+LSDA form. Rust std `personality/gcc.rs` (EHABI branch) is a direct template.
* `EH_DATA_REG` = [0, 1] is already correct for ARM (r0/r1). In the binary, `_Unwind_GetIP/SetGR/SetIP` resolve
  to the libunwind wrappers over `_Unwind_VRS_*`, so the extern decls can stay.
* llvisitor: the IR-side personality declaration `(i32, i32, i64, ptr, ptr)` does not matter for codegen,
  and landingpad lowering is unchanged.

## Testing options (Pixel 9a and Apple Silicon cannot run AArch32)
* **qemu-arm user mode on Linux** (CI x86_64 or a Linux VM): run a test executable built with
  `-target armv7a-linux-androideabi24`, linked against the NDK sysroot with bionic's `linker`. This is fiddly
  but free. A Debian armhf `qemu-user` loop with an `armv7-linux-gnueabihf` target is easier for the testsuite.
  Not installed here.
* **qemu-system-arm / Android armv7 emulator image**: Google ships only old (API ≤ 25-ish) armeabi-v7a
  system images, and they are very slow. Free, but the emulator does not run on Apple Silicon with arm32
  images in practice.
* **Firebase Test Lab**: has some armeabi-v7a physical devices. The free Spark tier gives a few device-runs
  per day. Paid is per device-minute. Best for an end-to-end Flutter smoke test.
* **An old 32-bit phone**, for example a used Android Go device. About $30-60 and the most reliable option
  for real perf/EH validation.

## Market (cited)
* Play requires a 64-bit build for every 32-bit ABI shipped. "for each native 32-bit architecture you support
  you must include the corresponding 64-bit architecture" (developer.android.com/google/play/requirements/64-bit).
* Play "will continue to deliver apps to existing 32-bit devices" (Android Developers Blog, Wear OS, 2026-04).
  So armeabi-v7a only *adds* reach to old or low-end 32-bit-only devices (Android Go class). arm64 is already
  required and covers everything else.

## Estimate and recommendation
* Minimal "builds + EH works + basic tests pass under qemu": **about 1.5-2 weeks** for one engineer.
  * ~1 d build plumbing (OpenBLAS, aliases, maximum intrinsic)
  * 2-3 d EHABI personality
  * 2-4 d size_t/pointer-width audit of compiler + stdlib `@llvm`
  * 1-2 d test infrastructure
* Production confidence on the BCI app (numerics vs arm64, threads/omp, Flutter packaging): +1 week.
* **Recommendation:** do not do it unless analytics show meaningful 32-bit-only users. If needed, do it in this
  order:
  1. qemu-user test harness first
  2. EHABI personality
  3. the size_t declare audit
  4. then plumbing

  Ship arm64-v8a only meanwhile. Alternative: compile the app with no exceptions crossing the FFI
  boundary and with `-disable-exceptions`, if that option exists. That would defer the EHABI work.
