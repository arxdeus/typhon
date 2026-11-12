# Typhon

Typhon is an ahead-of-time compiler for Python-syntax programs, written in Rust
on top of LLVM 20. It type-checks a statically typed Python dialect, lowers it to
an optimizing IR, and either runs it through an in-process ORC JIT or produces
native executables, libraries and CPython extensions.

See [`DESIGN.md`](DESIGN.md) for the architecture.

## Layout

| Crate | Role |
|---|---|
| `typhon-common` | errors, source info, logging, name mangling, options |
| `typhon-parser` | PEG parser, scoping, type checker, AST to IR translation, docs |
| `typhon-ir` | arena-based IR, visitors, cloning, formatting, pass traits |
| `typhon-ir-analyze` | CFG, dominators, reaching definitions, capture and side-effect analyses |
| `typhon-ir-transform` | pass manager and IR passes (cleanup, folding, lowering, NumPy fusion, OpenMP/GPU) |
| `typhon-llvm` | LLVM code generation, optimization, ORC JIT |
| `typhon-compiler` | driver wiring all stages |
| `typhon-cli` | the `typhon` binary |
| `typhon-runtime` | `libtyphonrt`: GC, exceptions, I/O, regex, NumPy kernels |
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

This produces `target/release/typhon` and `target/release/libtyphonrt.{dylib,so}`.
The stdlib is found through `TYPHON_STDLIB_DIR` / `TYPHON_PATH`, next to the
executable, or at `./stdlib` in this repository.

## Usage

```sh
typhon run program.py                 # JIT compile and run
typhon run -release program.py        # with optimizations
typhon build -exe -o prog program.py  # native executable
typhon build -lib -o libx program.py  # shared library
typhon build -pyext -o x program.py   # CPython extension
typhon build -llvm program.py         # emit LLVM IR
typhon doc file.py                    # JSON docs
```

## Runtime

`run`, `-lib` and `-pyext` use the shared `libtyphonrt`. `-exe` embeds the runtime
statically by default; pass `-dynamic-runtime` to link the shared library instead.

## Python interoperability

Python code can be called with `from python import module`, using the libpython
found automatically or given by `TYPHON_PYTHON`. Typhon programs can be built as
CPython extension modules with `-pyext`; see [`docs/pyext.md`](docs/pyext.md).

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
