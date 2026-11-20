<div align="center">

# Typhon

**Python syntax. Native binaries. Built with Rust and LLVM.**

[![Release workflow](https://github.com/arxdeus/typhon/actions/workflows/release.yml/badge.svg)](https://github.com/arxdeus/typhon/actions/workflows/release.yml)
[![GitHub release](https://img.shields.io/github/v/release/arxdeus/typhon?include_prereleases)](https://github.com/arxdeus/typhon/releases)
![LLVM 20.1](https://img.shields.io/badge/LLVM-20.1-262D3A?logo=llvm)

[Get started](#get-started) · [Usage](#usage) · [Scientific computing](#scientific-computing) · [Mobile libraries](#mobile-libraries) · [Documentation](#documentation)

</div>

Typhon is an ahead-of-time compiler for a statically typed Python dialect. It infers types, optimizes your code, and compiles it into native executables or libraries without a CPython interpreter. Use familiar Python syntax for numerical programs, signal processing, and native components you can embed in other applications.

```python
import numpy as np

samples = np.array([1.0, 2.0, 3.0, 4.0])
centered = samples - np.mean(samples)
print(centered)
```

```sh
typhon run -release signal.py
```

## Why Typhon?

- **Native ahead-of-time compilation.** Build executables, shared libraries, object files, assembly, or LLVM IR. `typhon run` compiles a temporary executable before running it, rather than using a JIT.
- **Python-style code, static types.** Type inference and generics with compile-time diagnostics.
- **Numerical code without Python bindings.** Native implementations of selected scientific Python APIs, NumPy fusion passes, and BLAS/LAPACK integration.
- **Embeddable computation.** Export C-callable functions for native applications, including Android and iOS apps using Flutter's `dart:ffi`.
- **A Rust compiler and runtime.** An arena-based IR, LLVM 20 backend, garbage collection, and a Rust implementation of the OpenMP runtime ABI.

> [!IMPORTANT]
> Typhon is not a drop-in replacement for CPython. Imports resolve to Typhon modules, not arbitrary packages from `pip`. Scientific APIs are implemented in subsets, and support for a name does not guarantee full upstream behavior. JIT, Jupyter, and CPython interoperability are outside the current scope.

## Get started

### Use a release archive

Check [GitHub Releases](https://github.com/arxdeus/typhon/releases) for available builds. The release workflow targets:

| Platform | Architecture | Archive |
| --- | --- | --- |
| macOS | Apple Silicon | `.tar.gz` |
| Linux | x86_64, ARM64 | `.tar.gz` |
| Windows | x86_64 (MSVC) | `.zip` |

Extract the archive and add its `bin` directory to your `PATH`. Keep the bundled `lib/typhon` directory alongside it: it contains the standard library and static runtime needed to compile programs. Windows packages also include OpenBLAS files.

A native linker toolchain is still required to build programs. On macOS, use the Xcode command-line tools. On Linux, install a C/C++ compiler toolchain. Windows requires the MSVC build tools and Clang/LLD.

> [!NOTE]
> The table describes the configured release matrix, not a guarantee that every platform has a published, validated build. Windows support is experimental. Check the workflow results and release assets for the version you use.

### Build from source

Requirements:

- Rust **stable**, with edition 2024 support.
- **LLVM 20.1** development libraries, including static libraries and the required target backends.
- A C/C++ toolchain, **CMake**, and **Make**. Linux's bundled OpenBLAS build uses Clang. macOS uses Accelerate for BLAS/LAPACK, requiring macOS 13.3+ for those APIs.
- Network access for Cargo dependencies and, on Linux, the OpenBLAS source download.

```sh
git clone https://github.com/arxdeus/typhon.git
cd typhon

# Point to your LLVM installation, not its bin directory.
export LLVM_SYS_201_PREFIX=/path/to/llvm20

# Linux only: override the macOS deployment flag in .cargo/config.toml.
# export CFLAGS_x86_64_unknown_linux_gnu="-DLARGE_CONFIG -DNO_BLACK_LISTING"
# export CFLAGS_aarch64_unknown_linux_gnu="-DLARGE_CONFIG -DNO_BLACK_LISTING"

cargo build --release -p typhon-cli -p typhon-runtime
export PATH="$PWD/target/release:$PATH"
```

On macOS with Homebrew LLVM 20, the prefix is `$(brew --prefix llvm@20)`. The environment variable overrides the developer-local default in [`.cargo/config.toml`](.cargo/config.toml).

On Linux, uncomment the two `CFLAGS` overrides **before** building. For platform-specific dependency setup, including Windows LLVM and OpenBLAS installation, see the [release workflow](.github/workflows/release.yml).

### Run your first program

Save the NumPy example above as `signal.py`, then run:

```sh
typhon run -release signal.py
# Displays the centered values: -1.5, -0.5, 0.5, 1.5
```

Or compile a reusable executable:

```sh
typhon build -release -exe -o signal signal.py
./signal
```

The runtime is linked into generated programs, so there is no separate Typhon runtime service or interpreter to launch. Platform system libraries and any explicitly linked external libraries may still be required.

## Usage

```sh
typhon run program.py                         # Compile and run
typhon run -release program.py                # Compile with optimizations
typhon build -release -exe -o app program.py  # Native executable
typhon build -obj -o program.o program.py     # Object file
typhon build -llvm program.py                # LLVM IR
typhon doc program.py                        # Documentation as JSON
typhon version                               # Compiler version
```

Typhon uses single-dash long options, such as `-release`, `-target`, and `-lib`.

| Environment variable | Purpose |
| --- | --- |
| `LLVM_SYS_201_PREFIX` | LLVM installation used to build the compiler |
| `TYPHON_STDLIB_DIR` | Override the standard-library search location |
| `TYPHON_PATH` | Additional module search paths |
| `TYPHON_LIB_DIR` | Directory containing the static runtime archive |

## Scientific computing

The [`stdlib/`](stdlib/) tree contains built-in types, Python-style standard-library modules, and native implementations of selected APIs from:

| Area | Modules |
| --- | --- |
| Arrays, numerics, and statistics | NumPy, SciPy, pandas |
| Machine learning and tensors | scikit-learn, torch |
| EEG and signal processing | MNE, braindecode, pyriemann, autoreject |
| Supporting utilities | einops, joblib, safetensors |

These are Typhon implementations, not embedded copies of the Python packages. Check the [scientific API inventory](docs/api_coverage.md), [pandas and pyriemann inventory](docs/api_coverage_pandas_pyriemann.md), and [standard-library inventory](docs/stdlib_coverage.md) alongside the relevant [tests](test/). The inventories are snapshots of public names, not behavioral compatibility guarantees.

## Mobile libraries

Export typed functions with `@export` to call them through a C ABI:

```python
@export
def add(a: int, b: int) -> int:
    return a + b
```

Save this as `native.py`. With the Android NDK installed, build the target runtime and library:

```sh
tools/build-runtime.sh aarch64-linux-android
typhon build -release -lib -target android-arm64 -o libnative.so native.py
```

For iOS, use Xcode on macOS:

```sh
tools/build-runtime.sh aarch64-apple-ios
typhon build -release -lib -target ios-arm64 -o libnative.dylib native.py
```

Aliases also include `android-x64` and `ios-sim-arm64`. Each target needs its own runtime build. The [mobile guide](docs/mobile.md) covers SDK discovery, ABI types, memory ownership, and linking. [ONNX Runtime integration](docs/onnx.md) documents calling an external inference engine through its C API.

## Development

```sh
# Build the debug compiler and runtime before running subprocess-based tests.
cargo build -p typhon-cli -p typhon-runtime
cargo test --workspace

# Run just the core end-to-end suite with the release compiler.
TYPHON_BIN="$PWD/target/release/typhon" \
  cargo test -p typhon-testsuite --test e2e -- core/
```

See the [testsuite guide](crates/testsuite/README.md) for filters, timeouts, platform-specific cases, and known gaps. Benchmark methodology and recorded results live in [`bench/`](bench/); some comparison backends require a separate checkout and additional tools.

### Project layout

| Path | Contents |
| --- | --- |
| `crates/parser`, `crates/common` | Parsing, type checking, diagnostics, and configuration |
| `crates/ir`, `crates/ir-analyze`, `crates/ir-transform` | Intermediate representation, analyses, and optimization passes |
| `crates/llvm`, `crates/compiler`, `crates/cli` | Code generation, compiler driver, and command-line interface |
| `crates/runtime`, `crates/omp` | Static runtime and OpenMP ABI |
| `stdlib`, `test`, `crates/testsuite` | Library implementations and regression tests |
| `tools`, `docs`, `bench` | Build utilities, detailed guides, and benchmarks |

Releases are created manually through **Actions → release → Run workflow** with a version and optional prerelease flag. The workflow updates the workspace version, creates a tag, builds the platform matrix, and publishes archives with SHA-256 checksums to GitHub Releases and workflow artifacts. Enable **build_only** to build and smoke-test artifacts without modifying versions, tags, or releases. In this mode, **platform** can select `windows`, `macos`, or `linux` for a targeted check. Publishing a release always builds the full matrix.

## Documentation

- [Compiler architecture](DESIGN.md): pipeline, crate graph, IR, and memory model.
- [Runtime internals](crates/runtime/README.md): allocation, garbage collection, and native dependencies.
- [Android and iOS](docs/mobile.md): cross-compilation and exported native functions.
- [Testsuite](crates/testsuite/README.md): running and extending the test harness.
- [Benchmarks](bench/README.md): comparison setup and measurement methodology.
- [WebAssembly research](docs/wasm.md): prototype status and current limitations, not a supported release target.
