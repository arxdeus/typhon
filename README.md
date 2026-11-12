# Typhon

Typhon is a Rust port of the [Codon](https://github.com/upstream/codon) compiler
(a Python-syntax ahead-of-time compiler built on LLVM). It lives next to the
original C++ sources in this repository and does not modify them. The Codon
standard library (`../stdlib/**/*.codon`) and the `.codon` test programs
(`../test/**`) are compiler *inputs* and are reused as-is.

See [`DESIGN.md`](DESIGN.md) for the architecture, memory model, crate graph and the
full C++ dependency -> Rust crate mapping.

## Scope

* Ported: the whole compiler (`codon/**`): PEG parser, scoping, type checker,
  AST -> IR translation, Codon IR with all analyses and optimization passes
  (cleanup, folding, lowering, pythonic, NumPy fusion, OpenMP/GPU), LLVM code
  generation, `run` via an in-process ORC JIT, `build` to exe/obj/lib/llvm/bc/asm/pyext,
  `doc`, the DSL/plugin system, the runtime library `libtyphonrt`, and the C++ tests.
* Not ported (by decision): the incremental JIT / REPL (`codon jit`), the Jupyter
  kernel and the Python `@codon.jit` bridge.
* Replaced with Rust: LLVM's `libomp` (OpenMP runtime) is reimplemented in
  `crates/omp`.

## Layout

| Crate | Port of | Notes |
|---|---|---|
| `typhon-common` | `codon/util`, `compiler/error`, `compiler/options`, parts of `parser/common` | errors, source info, logging, name mangling, filesystem |
| `typhon-parser` | `codon/parser/**` | AST arena, rust-peg grammar, scoping, type checker, translate, doc |
| `typhon-ir` | `codon/cir` core + `cir/util` | arena-based IR, visitors, operator, cloning, formatting, pass traits |
| `typhon-ir-analyze` | `codon/cir/analyze/**`, `cir/util/{inlining,outlining}` | CFG, dominators, reaching defs, capture, side effects |
| `typhon-ir-transform` | `codon/cir/transform/**` | pass manager and all IR passes |
| `typhon-llvm` | `codon/cir/llvm/**`, `compiler/{engine,memory_manager,debug_listener}` | inkwell + llvm-sys (LLVM 20), ORC JIT |
| `typhon-dsl` | `codon/dsl/**` | plugin manager, Rust cdylib plugin ABI |
| `typhon-compiler` | `codon/compiler/compiler.*` | driver wiring all stages |
| `typhon-cli` | `codon/app/main.cpp` | `typhon` binary, CLI-compatible with `codon` |
| `typhon-runtime` | `codon/runtime/**` | `libtyphonrt` (C ABI): GC, exceptions, I/O, regex, numpy kernels |
| `typhon-omp` | LLVM `libomp` | OpenMP `__kmpc_*` / `omp_*` ABI in Rust, linked into libtyphonrt |
| `typhon-testsuite` | `test/main.cpp` | data-driven e2e harness over `../test/**` |

## Building

Requirements: Rust stable (edition 2024), LLVM 20.1 static libraries
(configured in `.cargo/config.toml` via `LLVM_SYS_201_PREFIX`), a C/C++ toolchain
and CMake (for the vendored Boehm GC).

```sh
cd typhon
cargo build --release -p typhon-cli -p typhon-runtime
```

This produces `target/release/typhon` and `target/release/libtyphonrt.dylib`
(`.so` on Linux). The stdlib is found through `TYPHON_STDLIB_DIR`/`CODON_PATH`, next
to the executable, or at `../stdlib` in this repository.

## Usage

```sh
target/release/typhon run program.py                # JIT compile and run
target/release/typhon run -release program.py       # with optimizations
target/release/typhon build -exe -o prog program.py # native executable
target/release/typhon build -llvm program.py        # emit LLVM IR
target/release/typhon doc file.codon                # JSON docs
```

## Tests

```sh
cargo test --workspace                                   # unit + integration tests
TYPHON_BIN=$PWD/target/release/typhon \
  cargo test -p typhon-testsuite --test e2e -- core/     # end-to-end suites
```

See `crates/testsuite/README.md` for suite filters and environment variables.
