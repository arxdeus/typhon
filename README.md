# Typhon

Typhon is an ahead-of-time compiler for Python-syntax programs, written in Rust
on top of LLVM 20. It type-checks a statically typed Python dialect, lowers it to
an optimizing IR, and produces native executables. `typhon run` builds a
temporary executable and runs it (ahead-of-time, no JIT).

See [`DESIGN.md`](DESIGN.md) for the architecture.

## Layout

| Crate | Role |
|---|---|
| `typhon-common` | errors, source info, logging, name mangling, options |
| `typhon-parser` | PEG parser, scoping, type checker, AST to IR translation, docs |
| `typhon-ir` | arena-based IR, visitors, cloning, formatting, pass traits |
| `typhon-ir-analyze` | CFG, dominators, reaching definitions, capture and side-effect analyses |
| `typhon-ir-transform` | pass manager and IR passes (cleanup, folding, lowering, NumPy fusion, OpenMP/GPU) |
| `typhon-llvm` | LLVM code generation, optimization, object emission |
| `typhon-compiler` | driver wiring all stages |
| `typhon-cli` | the `typhon` binary |
| `typhon-runtime` | `libtyphonrt` (static): GC, exceptions, I/O, regex, NumPy kernels |
| `typhon-omp` | OpenMP runtime ABI in Rust, linked into `libtyphonrt` |
| `typhon-testsuite` | data-driven end-to-end harness over `test/` |

Other directories: `stdlib/` (the standard library, written in Typhon Python),
`test/` (test programs), `tools/` (generators), `bench/` (benchmarks, see
[`bench/README.md`](bench/README.md)), `docs/`.

## Building

Requirements: Rust stable (edition 2024), LLVM 20.1 static libraries (set
`LLVM_SYS_201_PREFIX` in `.cargo/config.toml`), a C/C++ toolchain and CMake
(for the vendored Boehm GC).

```sh
cargo build --release -p typhon-cli -p typhon-runtime
```

This produces `target/release/typhon`; the runtime is linked statically into every program.
The stdlib is found through `TYPHON_STDLIB_DIR` / `TYPHON_PATH`, next to the
executable, or at `./stdlib` in this repository.

## Usage

```sh
typhon run program.py                 # build to a temp executable and run
typhon run -release program.py        # with optimizations
typhon build -exe -o prog program.py  # native executable
typhon build -llvm program.py         # emit LLVM IR
typhon doc file.py                    # JSON docs
```

## Runtime

`libtyphonrt` is always linked statically; there is no shared runtime library.

## Standard library

`stdlib/` contains the builtin types and modules (`math`, `re`, `os`, `datetime`, ...),
plus native implementations of subsets of NumPy, SciPy, pandas, scikit-learn, torch,
MNE, braindecode, pyriemann, einops, joblib and safetensors.

## Tests

```sh
cargo test --workspace
TYPHON_BIN=$PWD/target/release/typhon \
  cargo test -p typhon-testsuite --test e2e -- core/
```

See [`crates/testsuite/README.md`](crates/testsuite/README.md) for filters and
environment variables.

## License

Apache-2.0, see [`LICENSE`](LICENSE). Attribution for the original work this
project derives from is in [`NOTICE`](NOTICE).
