# Typhon plugin format

A plugin is a directory with `plugin.toml` (same schema as Codon):

```toml
[about]
name = "myplugin"
supported = ">=0.20.0"        # npm-style semver range; empty/missing = any
[library]
codon = "stdlib"              # plugin stdlib directory (relative to plugin.toml)
compiler = "build/libmy"      # compiler dylib stem (".dylib/.so/.dll" appended); "" disables
runtime = "build/libmyrt"     # runtime dylib stem; "" disables default linking
cpp = "build/libmy"           # legacy: default for both compiler and runtime
link = ["-lfoo", "{root}/x.a"] # or a string; `link = false` == runtime = ""
```

Stdlib-only plugins (no compiler library) are supported and get a no-op `Dsl`.

## Compiler-library ABI

Rust has no stable ABI for trait objects, so a compiler plugin must be a `cdylib` built
against the **same `typhon-dsl` version and the same rustc** as the host. It exports:

| symbol | signature |
|---|---|
| `typhon_plugin_abi_tag` | `extern "C" fn() -> *const c_char` returning `typhon_dsl::abi::ABI_TAG_NUL` |
| `typhon_plugin_load` | `extern "C" fn(abi_version: u32) -> *mut c_void` returning `Box::into_raw(Box::new(Box<dyn Dsl>))`, null on mismatch |

The host checks the tag string (`"typhon-dsl <ver>; <rustc --version>; abi N"`) before calling
`typhon_plugin_load`. Use the macro:

```rust
#[derive(Default)] struct MyDsl;
impl typhon_dsl::Dsl for MyDsl { /* hooks */ }
typhon_dsl::export_plugin!(MyDsl);
```

Libraries are never unloaded (like C++ `getPermanentLibrary`).

PORT-NOTE: C++ Codon plugins (exporting `std::unique_ptr<DSL> load()`) are not loadable; the
loader reports a missing `typhon_plugin_abi_tag` symbol. Runtime libraries are plain dylibs and
work as before. LLVM pass hooks are per-stage module callbacks / textual pipelines
(`LlvmPassHooks`) rather than `PassBuilder` callbacks.
