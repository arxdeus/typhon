# WebAssembly (`wasm32-wasip1`): reconnaissance

Status as of 2026-11-12: **prototype only**. The compiler emits valid wasm32 objects today, and
with a small ABI fixup (prototyped as an IR rewrite) plus the real `libtyphonrt.a`
cross-built for `wasm32-wasip1`, both milestones ran under Node 26 (`node:wasi`) and
wasmtime 50:

| goal | result |
|---|---|
| (a) `print("hello")` exe | **runs** (Node + wasmtime), 2.0 MB `.wasm` with full runtime + bdwgc |
| (b) `@export add` / `fib` lib, called from JS | **runs**: `add(2,40)= 42n fib(25)= 75025n` |
| list/dict/str program, 100 k allocations | runs with `GC_DONT_GC=1` or a big initial heap; **crashes in the GC** with collection enabled (see B4) |
| `try/raise/except` | does **not** work (B5): raise aborts |

Committed (safe for every other target): the `wasm32-wasip1` target in `crates/common/src/target.rs`
(aliases `wasi`, `wasm32`; `TargetOs::Wasi`; defines `__wasm__`, `__wasi__`;
`sys.platform == "wasi"`; runtime archive dir `wasm32-wasip1`) and WASI `cfg` gates
in `crates/runtime/src/lib_rt.rs` (stat timestamps, `seq_os_home`, `set_errno`).
Prototype helpers live in `tools/wasm/` (not wired into the build).

## How it was run

Toolchain: Homebrew `llvm` 22 clang (has the WebAssembly backend), `/opt/homebrew/bin/wasm-ld`,
Homebrew `wasi-libc` (sysroot `/opt/homebrew/opt/wasi-libc/share/wasi-sysroot`), Rust target
`wasm32-wasip1` (rustup toolchain; note: Homebrew `rustc` on `PATH` shadows rustup and has no wasm std, so use
`~/.rustup/toolchains/stable-*/bin/{cargo,rustc}`), `wasmtime` (optional; Node is enough).

`libclang_rt.builtins.a` for wasm32 is missing from Homebrew llvm (it's in the
`wasi-runtimes` formula, which we did not manage to download). Workaround: repackage Rust's
`compiler_builtins` rlib as the builtins archive in a private resource dir:

```sh
RD=$PWD/resdir; mkdir -p $RD/lib/wasm32-unknown-wasip1; ln -s /opt/homebrew/opt/llvm/lib/clang/22/include $RD/include
CB=$(ls ~/.rustup/toolchains/stable-aarch64-apple-darwin/lib/rustlib/wasm32-wasip1/lib/libcompiler_builtins-*.rlib)
mkdir x && cd x && llvm-ar x $CB && rm -f lib.rmeta && llvm-ar rcs $RD/lib/wasm32-unknown-wasip1/libclang_rt.builtins.a *.o
# cc-wasi wrapper:
#   clang --target=wasm32-wasip1 --sysroot=$WASI_SYSROOT -resource-dir=$RD "$@"
```

Runtime (applied `tools/wasm/runtime-cargo-wasi.patch` to `crates/runtime/Cargo.toml`: liblzma
without the `parallel` feature on WASI):

```sh
export CC_wasm32_wasip1=cc-wasi AR_wasm32_wasip1=llvm-ar \
  CMAKE_TOOLCHAIN_FILE_wasm32_wasip1=tools/wasm/wasi.cmake TYPHON_WASI_CC=cc-wasi \
  BINDGEN_EXTRA_CLANG_ARGS_wasm32_wasip1="--sysroot=$WASI_SYSROOT -UGC_THREADS"
cargo build --release -p typhon-runtime --target wasm32-wasip1 --no-default-features
# -> target/wasm32-wasip1/release/libtyphonrt.a (27 MB)
```

Program (until the codegen fixes land, `-llvm` + `tools/wasm/fixup.py`):

```sh
typhon build -release -target wasi -llvm -o hello.ll hello.py
python3 tools/wasm/fixup.py hello.ll hello.fix.ll && cc-wasi -c -O2 hello.fix.ll -o hello.o
cc-wasi -O2 -c tools/wasm/gcshim.c tools/wasm/unw.c
L=$WASI_SYSROOT/lib/wasm32-wasip1
wasm-ld $L/crt1-command.o hello.o gcshim.o unw.o libtyphonrt.a -L$L -lc $RD/lib/wasm32-unknown-wasip1/libclang_rt.builtins.a -o hello.wasm
node tools/wasm/run.mjs hello.wasm     # -> hello
wasmtime hello.wasm                    # -> hello

typhon build -release -target wasi -llvm -o lib.ll lib.py     # @export add, fib
# same fixup/compile, then:
wasm-ld lib.o gcshim.o unw.o libtyphonrt.a -L$L -lc builtins.a --no-entry \
  --export=add --export=fib --export=seq_init -o lib.wasm      # 2.0 MB
node tools/wasm/run.mjs lib.wasm      # -> add(2,40)= 42n fib(25)= 75025n
```

`int` is `i64`, so JS passes and receives `BigInt`. `run.mjs` calls `wasi.initialize()` for
reactor modules (no `_start`). Without the runtime (pure arithmetic), `wasm-ld --no-entry
--export=add --export=fib` gives a 349-byte module.

## Blockers

| # | blocker | evidence | fix | effort |
|---|---|---|---|---|
| B1 | **No wasm link step** | `typhon build -target wasm32-wasip1 -exe` -> `LLVM error: unsupported cross target 'wasm32-wasip1'` (before the commit: `rust_triple()` was `None`). `-llvm`/`-obj` already work (datalayout `e-m:e-p:32:32...`, triple `wasm32-wasip1`). | `cross.rs`: `TargetOs::Wasi` => `wasm-ld` (or `clang --target=wasm32-wasip1 --sysroot`), libs `-lc` + builtins; exe: `crt1-command.o`; `-lib`: `--no-entry --export=<each @export> --export=seq_init` (or `crt1-reactor.o` + `_initialize`); `output.rs`: no `-lgcc_s/-lpthread`, output `.wasm`. Main must be `__main_argc_argv` (wasi-libc's `__main_void` calls it; with plain `main(i32,ptr)` you get `undefined_weak:main` -> `unreachable` trap). | S |
| B2 | **Pointer-width ABI** (size_t) | wasm-ld: `function signature mismatch: seq_alloc` / `seq_alloc_atomic` / `seq_realloc` / `strlen` / `memcmp` (`defined as (i32, i32, i64) -> i32 in hello.o` vs `(i32, i32, i32) -> i32 in libc.a(memcmp.c.obj)`); at run time the call becomes `signature_mismatch:<fn>` -> `unreachable`. Sources: `llvisitor/mod.rs` `seq_alloc*(i64)`, `seq_realloc(ptr,i64,i64)`, `strlen -> i64`, plus `memcmp/memcpy/...` sizes. Stdlib: 300 `from C import` lines, 65 pass `int`; on wasm the relevant ones are `read/write/lseek/getcwd/ftruncate/truncate/readlink/mmap/munmap/dlopen/exit` (plus omp/cnp ones, excluded on wasm). | Codegen: use an `intptr` type (`DataLayout` pointer size) for size_t params in runtime decls (~15 sites in `llvisitor/*.rs`), trunc/zext at the call. Stdlib: introduce `csize`/`cssize` aliases = `i64`/`i32` by `__wasm__` and use them in ~10 imports. `fixup.py` is the throwaway equivalent. | M |
| B3 | **Aggregate by-value C ABI** | `seq_print_full({ptr,i64}, ptr)` -> wasm-ld `defined as (i32, i64, i32) -> void in hello.o` vs `(i32, i32) -> void` in runtime: the wasm C ABI passes structs **indirectly** (by pointer), LLVM IR `{ptr,i64}` by value is flattened. Affects every runtime fn taking/returning `str` (`SeqStr`): 3 in exc.rs, 8 in lib_rt.rs, 7 in rtti.rs, 2 in types.rs. Also the `tail` marker must be dropped when passing an alloca (a call with a `tail` marker printed nothing until removed). | Either runtime-side: make these `extern "C"` fns take `*const SeqStr` on wasm (cfg), or codegen-side: an ABI lowering for wasm (alloca+pointer for aggregate args, sret for aggregate returns). Codegen side is cleaner (also covers `from C import` of structs). | M |
| B4 | **GC correctness**: bdwgc cannot scan wasm locals/value stack | Single-threaded bdwgc builds and initialises fine. 100 elems: OK. 5 000+: `Duplicate large block deallocation of 0x69000` -> abort. Same binary with `GC_DONT_GC=1` or `GC_INITIAL_HEAP_SIZE=256M`: correct output `100000 1000 0 1 2 3 4`. So live objects referenced only from wasm locals (not spilled to the linear-memory shadow stack) are freed. | M1: (i) GC only at safe points: `GC_disable()` in `seq_init`; `GC_enable(); GC_gcollect(); GC_disable()` between exported calls / on JS request, plus a large initial heap; (ii) Binaryen `--spill-pointers` pass (spills pointer locals to the shadow stack: makes conservative scanning correct, ~10-30% slower); (iii) later: LLVM statepoints / precise roots. Recommend (i) for M1, (ii) for M2. | M |
| B5 | **Exceptions** | Runtime references `_Unwind_RaiseException` (exc.rs): link error `undefined symbol: _Unwind_RaiseException`; stubbed with abort (`tools/wasm/unw.c`). Typhon emits Itanium `invoke`/`landingpad` + `seq_personality`; WebAssembly backend lowers them only via SjLj-like emulation or wasm EH, and `llc -wasm-enable-eh -exception-model=wasm -mattr=+exception-handling` **crashes** (LLVM 20 and 22, `WebAssembly Peephole` on `__main_argc_argv`) on this IR. `-disable-exceptions` removes `invoke/landingpad/seq_throw` from user code (0 left in `big.py`), but stdlib raises still exist as calls. Node 26 has `WebAssembly.Tag/Exception` (EH supported); wasmtime 50 supports legacy+exnref EH. | M1: `-disable-exceptions` (or treat raise as trap) + abort stub. M2: LLVM wasm EH: run `WasmEHPrepare` (needs funclet IR: `catchswitch/catchpad`; Typhon must emit Windows-style funclets or rely on `WasmEHPrepare`'s landingpad conversion with the C++-style `__gxx_wasm_personality_v0` + `_Unwind_CallPersonality` from libunwind wasm), a wasm `seq_personality` that doesn't parse LSDA the Itanium way, and libunwind for wasm (wasi-runtimes ships `libunwind.a` for EH builds). | L |
| B6 | **Runtime portability** (Rust/C) | After cfg gates committed: Rust compiles clean. C deps: liblzma `parallel` -> `wasm lacks signal support ... -D_WASI_EMULATED_SIGNAL` (77x, `pthread_sigmask`, `SIG_SETMASK`): fixed by dropping `parallel` on WASI (Cargo.toml patch). bdwgc bindgen: `'pthread.h' file not found` (bdwgc-alloc-sys forces `-DGC_THREADS`): `BINDGEN_EXTRA_CLANG_ARGS_wasm32_wasip1=-UGC_THREADS`. bdwgc cmake needs a WASI toolchain file (`CMAKE_SYSTEM_NAME WASI`, `enable_threads OFF`). Link: `undefined symbol: GC_get_prof_stats_unsafe`, `GC_allow_register_threads` (thread-only; shim in `tools/wasm/gcshim.c`, should be `#[cfg(not(target_os="wasi"))]` in `gc.rs`/`lib_rt.rs` gc_init, which is a hot area for the omp lanes). zlib/bz2 (Rust), regex, backtrace (no-op): OK. Needs `--no-default-features` (omp-rust, blas). | S |
| B7 | **Toolchain packaging** | Homebrew llvm lacks `lib/clang/22/lib/wasm32-unknown-wasip1/libclang_rt.builtins.a` (`wasm-ld: error: cannot open .../libclang_rt.builtins.a`). Worked around with Rust compiler_builtins. Clean options: `brew install wasi-runtimes` (pulls llvm 23 as a dep, large, slow download here), or wasi-sdk (~100 MB tarball, self-contained clang+sysroot+compiler-rt, recommended for CI). `tools/build-runtime.sh` needs a `wasm32-wasip1` case with the env above. | S |
| B8 | **Stdlib features without WASI support** | Not exercised. No threads (`openmp`, `@par`), no `dlopen`, no `mmap`, no signals, no subprocess, no sockets in p1. Gate with `__wasi__` in stdlib (`os`, `openmp`, `numpy` BLAS paths). | M |

### wasm64 (memory64) as a B2 workaround?

It would make `ptr` = `i64` and remove the size_t mismatches. But: Rust has only tier-3
`wasm64-unknown-unknown` (no `wasm64-wasip1` std, so `-Zbuild-std` on nightly), wasi-libc has
no `wasm64-wasip1` sysroot in the Homebrew bottle (only `wasm32-wasip1{,-threads}`,
`wasip2`, `wasip3`), and Node's WASI implements only 32-bit memory. Browsers only recently
shipped memory64, and it's slower (bounds checks). **Not recommended**: fixing B2/B3 in codegen is smaller.

## Milestones

* **M1 (S+M, ~1-2 weeks)**: single-threaded WASI exe + `-lib` reactor. B1, B2, B3, B6, B7;
  B5 via `-disable-exceptions`/abort; B4 via GC disabled except at explicit safe points
  between exported calls (`seq_gc_collect()` export) and a large initial heap. Testsuite subset
  under `node`/`wasmtime`.
* **M2 (L)**: exceptions via wasm EH (B5), Binaryen `--spill-pointers` or shadow-stack root
  spilling for an always-on GC (B4).
* **M3 (M)**: browser: either `wasm32-unknown-unknown` + a tiny JS WASI shim
  (`@bjorn3/browser_wasi_shim`), or Emscripten (`wasm32-unknown-emscripten`, which has EH
  and its own JS glue; bdwgc has an Emscripten port). Typed JS wrappers for `@export`
  (BigInt for `int`, string marshalling through `seq_alloc`).
* **M4 (L)**: `wasm32-wasip1-threads` (shared memory, bdwgc with threads, typhon-omp on
  wasi-threads), SIMD128, BLAS (no OpenBLAS kernel for wasm; use reference kernels).

## Installed during the recon

`brew install wasi-libc wasmtime` (small). This **upgraded `z3` 5.1.0 -> 5.4.1** as a side
effect, which broke Homebrew `llvm` 22 (`dyld: Library not loaded: .../libz3.5.1.dylib`). Repaired
by pointing `/opt/homebrew/opt/z3` back to `../Cellar/z3/5.1.0` (both kept). A `brew upgrade llvm`
(to 23) would also fix it. `rustup target add wasm32-wasip1` (stable toolchain). `wasi-runtimes` was
attempted but did not complete (it pulls llvm 23).
