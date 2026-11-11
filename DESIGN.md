# Typhon: Rust port of the Codon compiler

`typhon/` is a self-contained Cargo workspace containing a Rust translation of the
Codon compiler sources (`../codon/**`, `../test/*.cpp`).

**Scope (user decision):** only the basic JIT used by `typhon run` (compile the whole
program into an ORC JIT and execute `main`, i.e. `LLVMVisitor::run` + `compiler/engine`)
is ported. NOT ported: the incremental JIT / REPL (`codon jit` mode, `compiler/jit.cpp`
`JIT::init/compile/execute`), the Jupyter kernel (`../jupyter`, `codon/util/jupyter.*`)
and the Python `@codon.jit` bridge (`../jit`, `jit_extern.h`, `JIT::executePython`).
Everything else (`run`, `build` to exe/obj/llvm/bc/lib/pyext, `doc`, plugins/DSL,
all IR passes, GPU, OpenMP, runtime) is in scope.
Nothing outside `typhon/` is modified. The Codon-language standard library
(`../stdlib/**/*.codon`) and the `.codon` test programs (`../test/**`) are *inputs*
of the compiler, not compiler source, so they are reused in place (read-only).

## Crate graph (no cycles)

```
typhon-common        codon/util, compiler/error, compiler/options, parser/common, config
  ├── typhon-ir             codon/cir/*.{h,cpp}, cir/util/*, cir/dsl/nodes, transform/pass.h,
  │     │                   analyze/analysis.h, transform/parallel/schedule.h, transform/rewrite.h
  │     ├── typhon-ir-analyze    codon/cir/analyze/{dataflow,module}/*
  │     │     └── typhon-ir-transform  codon/cir/transform/{cleanup,folding,lowering,pythonic,
  │     │                              numpy,parallel}/*, transform/manager.*
  │     ├── typhon-llvm         codon/cir/llvm/* (llvisitor, optimize, gpu, native/*),
  │     │                       compiler/{engine,memory_manager,debug_listener}
  │     └── typhon-parser       codon/parser/** (ast, types, peg grammar, visitors, cache, ctx,
  │                             match, capi)
  ├── typhon-dsl            codon/dsl/* (DSL trait, PluginManager)    -> parser, ir-transform, llvm
  ├── typhon-compiler       codon/compiler/{compiler,error}            -> everything above
  ├── typhon-cli  (bin `typhon`)   codon/app/main.cpp (run / build / doc modes; no jit REPL, no jupyter)
  ├── typhon-runtime        codon/runtime/** -> `libcodonrt` (staticlib + cdylib, C ABI)
  └── typhon-testsuite      test/main.cpp, test/types.cpp, test/cir/**, test/dsl/**
```

### Breaking the C++ include cycles

C++ freely includes in both directions (`cir/module.h` <-> `parser/cache.h`,
`llvisitor.h` -> `dsl/plugins.h` -> `parser`). In Rust the dependency is inverted through traits:

| C++ coupling | Rust replacement |
|---|---|
| `ir::Module` holds `ast::Cache*` and calls `cache->realizeType/realizeFunction/findMethod/makeTuple/...` | `typhon_ir::realize::TypeRealizer` trait (defined in `typhon-ir`, implemented by `typhon-parser::Cache`). `Module` stores `Option<Rc<RefCell<dyn TypeRealizer>>>`. |
| `ir::types::Type::astType` (`ast::types::TypePtr`) | opaque `AstTypeRef` handle (`u64` / `Rc<dyn Any>`) stored on IR types; only the parser interprets it. |
| `ast::ASTNode : ir::Node` (attributes) | AST owns its own attribute map (`AttrValue` enum). Translate converts to IR attributes. |
| `cir/util/side_effect.cpp`, transforms using `ast::getMangledFunc` / `ast::StdlibTypes` | `typhon_common::mangle` (`mangled_func`, `stdlib()`). |
| `cache->compiler->getOptions()` | `typhon_common::Options` passed in / stored in `Cache`. |
| parser loading plugins / DSL keywords (`PluginManager`) | `typhon_parser::PluginHost` trait, implemented in `typhon-dsl`. Keyword callback type is defined in the parser. |
| `optimize.h`/`llvisitor.h` -> `dsl/plugins.h` (`addLLVMPasses`, custom node codegen) | `typhon_llvm::LlvmHooks` trait + `typhon_ir::dsl` builder traits downcast in `typhon-llvm`. |
| `analyze::Analysis : transform::Pass` vs manager in transform | `Pass`, `PassGroup`, `Analysis`, `AnalysisResult` traits live in `typhon-ir::pass`; `PassManager` lives in `typhon-ir-transform`. |

## Memory model

C++ uses raw pointers into a node graph with LLVM-style RTTI (`isA<>`, `cast<>`,
`dynamicNodeId`) and a `replaceAll` forwarding mechanism. Rust uses:

* **Arenas + typed ids** (`la-arena`). `Copy` ids (`ValueId`, `VarId`, `TypeId`,
  `ExprId`, `StmtId`, ...) instead of pointers. The owner (`ir::Module`,
  `parser::Cache`) owns the arenas. No `Rc` cycles.
* **Enums instead of class hierarchies.** RTTI (`isA<IntConst>(v)`) becomes `match`
  on a kind enum; `cast<T>` becomes helper accessors returning `Option<&T>`.
* **Replacement forwarding** (`Node::replaceAll`, `getActual`) is kept: each node
  slot has `replacement: Option<Id>`; all accessors resolve through `actual(id)`.
* **Visitors**: C++ double dispatch (`accept(Visitor&)`) becomes a trait with one
  default method per node kind plus a `walk_*` helper, dispatched by `match`.
* **Types in the type checker** (`ast::types::*`, `LinkType` unification) live in a
  `TypeArena` owned by `Cache`, referenced by `TypeId`. Link/unbound/generic binding
  is a mutable slot in the arena (union-find style).
* Hold `&mut` borrows only for short scopes; copy ids out before recursing.

## Errors

C++ exceptions -> `typhon_common::Result<T>` with `CompilerError`
(`Parser(ParserErrors) | Runtime{..} | Plugin | Io | Llvm`). `E(Error::X, src, args...)`
-> `bail_e!(ErrorKind::X, src, args...)`. Speculative `try { ... } catch (ParserException&)`
blocks in the type checker become `match` on the `Result`.

## C++ dependency -> Rust crate mapping

| C++ / system dependency | Used for | Rust crate |
|---|---|---|
| LLVM 20 C++ API | IR building, passes, codegen, ORC JIT, DWARF | `inkwell` 0.11 (`llvm20-1`) + `llvm-sys` 201 (ORC LLJIT, `LLVMRunPasses`, DIBuilder bits) |
| upstream/cpp-peglib + `util/peg2cpp.cpp` | PEG grammar (`grammar.peg`, `openmp.peg`) | `peg` 0.8 (grammar compiled by proc-macro, no code generator needed) |
| fmtlib/fmt | formatting | `std::fmt` / `format!` |
| marzer/tomlplusplus | plugin manifests | `toml` |
| Neargye/semver | plugin version checks | `semver` |
| cmrc (embedded resources) | embedded stdlib | `include_dir` |
| util/tser.h, util/serialize.h | AST (de)serialization | `serde` + `postcard` |
| `InternedString` | file names | `ustr` |
| llvm::DenseMap / SmallVector / StringMap | containers | `rustc-hash`, `indexmap`, `smallvec` |
| llvm dominator tree / CFG utilities | dataflow analyses | `petgraph` (`algo::dominators`) |
| bdwgc (Boehm GC) | runtime allocator (`seq_alloc`, `GC_*`) | `bdwgc-alloc` (vendored bdwgc build) |
| google/re2 | `re` module runtime | `regex` (same linear-time semantics) |
| ianlancetaylor/libbacktrace | runtime backtraces | `backtrace` |
| libunwind / Itanium EH (`_Unwind_*`, LSDA) | runtime exceptions personality | `gimli` (LSDA/DWARF parsing) + system `_Unwind_*` |
| zlib-ng | `gzip` module | `libz-rs-sys` (C ABI `gz*` exported) |
| bzip2 | `bz2` module | `libbz2-rs-sys` (C ABI `BZ2_*` exported) |
| xz / liblzma | `lzma` module | `liblzma-sys` |
| fast_float | float parsing in runtime | `fast-float2` |
| google/highway (vqsort, SIMD math) | numpy sort / ufunc loops | `glidesort` + `pulp` + `libm` |
| OpenBLAS / Accelerate | numpy linalg | `openblas-src` / `accelerate-src` + `cblas-sys` |
| mimalloc | optional allocator | `mimalloc` |
| complex math (`std::complex`) | numpy zmath | `num-complex` |
| xeus / xeus-zmq | Jupyter kernel | out of scope (not ported) |
| Cython (`jit.pyx`) | Python `@codon.jit` bridge | out of scope (not ported) |
| llvm::cl options (CLI) | command line | `clap` |
| dlopen | plugins, JIT symbols | `libloading` |
| googletest | C++ tests | `libtest-mimic` (data-driven `.codon` tests) + `#[test]` + `insta` |

## Conventions for porters

1. One Rust module per C++ `.h/.cpp` pair, mirroring the path
   (`codon/cir/transform/folding/const_fold.cpp` -> `crates/ir-transform/src/folding/const_fold.rs`).
   First line: `//! Port of codon/<path>`.
2. Port the full behavior, not just signatures. No `todo!()` / `unimplemented!()` in
   delivered code. If something truly cannot be expressed, write the closest
   equivalent and mark it with `// PORT-NOTE: <why>`.
3. Names: `getFoo()` -> `foo()`, `setFoo()` -> `set_foo()`, `isFoo()` -> `is_foo()`,
   types keep CamelCase. Keep doc comments.
4. `unsafe` only for FFI (runtime, LLVM, plugins, JIT).
5. Every crate must pass `cargo check -p <crate>` and `cargo test -p <crate>`.
   Port the matching C++ unit tests (`test/cir/**`, `test/types.cpp`) as `#[test]`s.
6. Use `cargo check`, not `cargo build`, while iterating (disk is limited; the shared
   `target/` is locked by one cargo at a time, so expect short waits).
7. Only edit your own crate/module. API requests for another crate: DM its owner
   or list them in your final report.
8. Commit only your files: `git add typhon/crates/<yours>` then
   `git commit -m "typhon(<crate>): ..."`. Never `git add -A`, never touch files
   outside `typhon/`.

## Toolchain

* Rust stable (1.97 tested), edition 2024.
* LLVM 20.1 static libs at `/Users/mind/orca/toolchains/llvm-codon` (the same
  toolchain the C++ build uses). Set in `.cargo/config.toml` as `LLVM_SYS_201_PREFIX`.
* Stdlib directory: `TYPHON_STDLIB_DIR` (defaults to `../stdlib`).
