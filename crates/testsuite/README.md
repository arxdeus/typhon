# typhon-testsuite

Port of `test/main.cpp` (googletest) to a data-driven `libtest-mimic` harness, plus
scaffolding for the in-process unit-test ports.

## End-to-end suites (`tests/e2e.rs`)

Every case runs `<TYPHON_BIN> run [-debug|-release] -numerics=py|c <file>` as a subprocess.

| env | default | meaning |
|---|---|---|
| `TYPHON_BIN` | `<ws>/target/debug/typhon` | compiler CLI (flag compatible with `codon`) |
| `TYPHON_STDLIB_DIR` | `<ws>/stdlib` | exported as `TYPHON_PATH` |
| `TYPHON_TEST_DIR` | `<ws>/test` | `.py` test inputs |
| `TYPHON_TEST_TIMEOUT_SECS` | `1200` | per-case timeout |
| `CODON_PYTHON` | (inherit) | libpython for python-interop suites (see `test/python/find-python-library.py`) |

```
export TYPHON_BIN=/Users/mind/orca/codon/build/codon   # validate against C++
cargo test -p typhon-testsuite --test e2e -- core/            # one suite
cargo test -p typhon-testsuite --test e2e -- types/ --test-threads 8
cargo test -p typhon-testsuite --test e2e -- --list
```
Name filters (substring): `types/`, `core/`, `stdlib/`, `cnumerics/`, `opt/`, `gpu/`, `numpy/`,
`scipy/` (also pandas, mne, pyriemann), `braindecode/` (also torch, einops, sklearn, joblib).
Only one filter positional is accepted by libtest-mimic; use `--skip` for exclusion.

### Semantics ported
* `#%%` splitting (`types/`): header `#%% name[,barebones]`, `__ignore__` skipped, case name
  `<endline>_<name>`. Snippets are written to a temp dir under the original file name with
  `start_line` blank lines prepended (line numbers preserved); the source directory is copied
  alongside so `import a` resolves. `CODON_TEST_FLAGS = 1 + barebones` (honoured by the CLI,
  as in `app/main.cpp`).
* Expectations: `# EXPECT: `, `#: ` (exact line compare), `#! ` (expected compile error: exit
  code 1, prefix compare). Exit code must equal `isError`. Any `TEST FAILED` fails the case.
  Output line count must match.
* Compile errors: the C++ child printed bare messages on stdout; the CLI prints decorated
  diagnostics on stderr, so `error: ...` messages (including `├─`/`╰─` continuation lines) are
  extracted from stderr (ANSI stripped) and treated as output.
* Per-case cwd: tests use `build/...` scratch paths and `test/...` fixtures relative to the repo
  root, so each case runs in a fresh temp dir with an empty `build/` and a `test` symlink.
* `PYTHONPATH=<test>/python[:existing]`.
* `gpu/` is compile-only (`build -llvm`), `cnumerics/` uses `-numerics=c`.
* The Windows-only JIT tests and the 64 KiB stdout truncation are not ported.

### Known gaps vs C++ (need in-process compiler API)
`options->test` and `options->standalone` are set in-process by C++ (`test=true` keeps the full
debug pass pipeline off; `standalone` is bit 2 of `seq_flags`). The CLI cannot set them; hence
`types/import_c` (expects `seq_flags == 5`) fails against the CLI. The custom test passes
`TestOutliner`, `TestInliner`, `EscapeValidator` (needed by `opt/outlining`, `opt/inlining`,
`opt/escapes`) also need an in-process pass manager; ports belong in `tests/unit/`.
A suggested CLI hook for Typhon: honour env `TYPHON_TEST_MODE=1` to set `test`/`standalone` and
register those passes.

## In-process unit tests (`tests/unit/`)
Empty scaffolding: `cir.rs` (test/cir/**), `types.rs` (test/types.cpp), `dsl.rs` (test/dsl/**).
Add `#[test]` fns there once the corresponding crates exist; run `cargo test -p typhon-testsuite --test unit`.

## Adding cases
Add the path to the right table in `src/suites.rs` (or just a new `#%%` section / `# EXPECT:` in a
`.codon` file; no Rust change needed).
