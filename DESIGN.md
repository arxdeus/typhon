# Typhon design

Typhon is a self-contained Cargo workspace implementing a compiler for a statically
typed Python dialect.

## Pipeline

1. **Parse** (`typhon-parser`): PEG grammar (`peg`), scoping, type checking with
   unification and realization of generics.
2. **Translate**: typed AST to Typhon IR (`typhon-ir`).
3. **Analyze and transform** (`typhon-ir-analyze`, `typhon-ir-transform`): cleanup,
   constant folding, lowering, Pythonic rewrites, NumPy fusion, OpenMP and GPU passes.
4. **Generate** (`typhon-llvm`): LLVM IR, optimization, object emission and linking.
   `run` builds a temporary executable and runs it (AOT, no JIT).
5. **Run**: programs link against `libtyphonrt` (`typhon-runtime`), with the OpenMP
   ABI provided by `typhon-omp`.

## Crate graph

```
typhon-common
  ├── typhon-ir
  │     ├── typhon-ir-analyze
  │     │     └── typhon-ir-transform
  │     ├── typhon-llvm
  │     └── typhon-parser
  ├── typhon-compiler   (everything above)
  ├── typhon-cli        (binary `typhon`: run / build / doc)
  ├── typhon-runtime    (`libtyphonrt`, C ABI)
  ├── typhon-omp        (OpenMP runtime in Rust)
  └── typhon-testsuite
```

The parser and IR depend on each other conceptually, so the IR defines the
`typhon_ir::realize::TypeRealizer` trait, implemented by the parser's `Cache`. IR types
carry an opaque `AstTypeRef` handle that only the parser interprets. Pass traits
live in `typhon-ir::pass`; the `PassManager` lives in `typhon-ir-transform`.

## Memory model

* **Arenas and typed ids** (`la-arena`): `Copy` ids (`ValueId`, `VarId`, `TypeId`,
  `ExprId`, `StmtId`, ...). The owner (`ir::Module`, `parser::Cache`) owns the arenas.
* **Enums** for node kinds; `match` replaces runtime type tests.
* **Replacement forwarding**: each node slot has `replacement: Option<Id>`, and
  accessors resolve through `actual(id)`.
* **Visitors** are traits with one default method per node kind plus `walk_*` helpers.
* **Types** live in a `TypeArena` owned by `Cache`; links, unbound and generic
  bindings are mutable arena slots.

## Errors

`typhon_common::Result<T>` with `CompilerError`
(`Parser(ParserErrors) | Runtime{..} | Io | Llvm`), raised via `bail_e!`.

## Runtime

`libtyphonrt` provides the allocator (Boehm GC), exceptions and unwinding, I/O,
regex, compression, and NumPy kernels. It is linked statically only.
`typhon-omp` implements `__kmpc_*` / `omp_*` on `std::thread`.

## Dependencies

| Purpose | Crate |
|---|---|
| LLVM bindings | `inkwell` (`llvm20-1`), `llvm-sys` 201 |
| Grammar | `peg` |
| Embedded stdlib | `include_dir` |
| Serialization | `serde`, `postcard` |
| Interning, containers | `ustr`, `rustc-hash`, `indexmap`, `smallvec` |
| Graph analyses | `petgraph` |
| Allocator | `bdwgc-alloc`, optional `mimalloc` |
| Regex | `regex` |
| Backtraces, unwinding | `backtrace`, `gimli` |
| Compression | `libz-rs-sys`, `libbz2-rs-sys`, `liblzma-sys` |
| Float parsing | `fast-float2` |
| Sorting, SIMD, math | `glidesort`, `pulp`, `libm`, `num-complex` |
| Linear algebra | `openblas-src` / `accelerate-src`, `cblas-sys` |
| Threads | `parking_lot`, `crossbeam-utils`, `crossbeam-deque` |
| CLI | `clap` |
| Dynamic loading | `libloading` |
| Tests | `libtest-mimic`, `insta` |

## Conventions

* `unsafe` only for FFI (runtime, LLVM).
* No `todo!()` / `unimplemented!()` in committed code.
* Every crate passes `cargo check -p <crate>` and `cargo test -p <crate>`.

## Toolchain

Rust stable, edition 2024. LLVM 20.1 static libraries located through
`LLVM_SYS_201_PREFIX`. Stdlib directory: `TYPHON_STDLIB_DIR` (defaults to `./stdlib`).
