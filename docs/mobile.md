# Android and iOS libraries

Typhon can cross-compile Python modules to native libraries that mobile apps
(for example Flutter via `dart:ffi`) load at run time.

## Targets

| `-target` alias  | LLVM triple                       | Runtime (Rust triple)   | Output    |
|------------------|-----------------------------------|-------------------------|-----------|
| `android-arm64`  | `aarch64-linux-android24`         | `aarch64-linux-android` | `.so`     |
| `android-x64`    | `x86_64-linux-android24`          | `x86_64-linux-android`  | `.so`     |
| `ios-arm64`      | `arm64-apple-ios16.4`             | `aarch64-apple-ios`     | `.dylib`  |
| `ios-sim-arm64`  | `arm64-apple-ios16.4-simulator`   | `aarch64-apple-ios-sim` | `.dylib`  |

You can also pass any full LLVM triple. With a cross target, code is always PIC and
uses generic per-target CPUs (`generic+neon` on Android arm64, `x86-64+sse4.2` on
Android x64, `apple-a12` on iOS), never the host CPU. You can override this with
`-mcpu`/`-mattr`. `-sysroot <dir>` overrides the NDK/SDK sysroot.

## 1. Build the runtime

```sh
tools/build-runtime.sh aarch64-linux-android x86_64-linux-android aarch64-apple-ios aarch64-apple-ios-sim
```

`typhon` looks for `libtyphonrt.a` in these places, in order:

1. `$TYPHON_LIB_DIR/<T>/libtyphonrt.a`
2. `<dir of typhon>/../<T>/release/libtyphonrt.a`
3. `<dir of typhon>/../lib/typhon/<T>/libtyphonrt.a`

## 2. Build a library

```sh
typhon build -release -lib -target android-arm64 -o libfoo.so foo.py
typhon build -release -lib -target ios-arm64     -o libfoo.dylib foo.py
typhon build -release -exe -target android-arm64 -o foo foo.py   # CLI program
```

`-obj` and `-llvm` also accept `-target`, which is useful for static linking on iOS.

Link details (set `TYPHON_LOG_COMMANDS=1` to print the command):

- **Android**: NDK clang (`$ANDROID_NDK_HOME`, else the newest complete
  `$ANDROID_HOME/ndk/*`) with `--target=<triple>`, `-shared -Wl,-soname,<file>`,
  `-Wl,--gc-sections -Wl,--as-needed -Wl,-z,max-page-size=16384` (Android 15 uses
  16 KB pages), and `-ldl -llog -lunwind -lm -lc`.
- **iOS**: `xcrun --sdk iphoneos|iphonesimulator clang -target <triple> -isysroot <sdk>`,
  install name `@rpath/<file>`, `-Wl,-dead_strip`, `-liconv -lc -lm -framework Accelerate`.
- In both cases, only `@export` functions are exported (a version script on Android,
  `-exported_symbols_list` on iOS). Runtime symbols stay hidden.
- `TYPHON_CROSS_LINKER` overrides the linker driver.

## Exported functions

```python
@export
def add(a: int, b: int) -> int: return a + b

@export
def strlen(s: cobj) -> int: return len(str.from_ptr(s))
```

| Typhon      | C                                |
|-------------|----------------------------------|
| `int`       | `int64_t`                        |
| `float`     | `double`                         |
| `bool`      | `bool` (`uint8_t`)               |
| `cobj`, `Ptr[byte]` | `void*` / `const char*` |

Strings and memory ownership:

- `str.from_ptr(p)` copies a NUL-terminated C string. The caller keeps ownership of `p`.
- To return a string, return a `cobj` to memory that the C side frees. Allocate it
  with libc `malloc`, never with the GC heap, and export a matching free function.
  GC memory may move out of reach once no Typhon reference remains.

## Threading

- Module top-level code runs in a library constructor when the library is loaded.
- Every `@export` function `f` is emitted as a thin public wrapper `f` that calls
  `seq_register_current_thread()` and then tail-calls the internal body
  `f.typhon.body` (local symbol, never exported). Internal calls, including
  recursion, go straight to the body, so registration is paid once per call from
  outside, not per recursive call. Registration is idempotent and registers the
  calling thread with the GC, so Dart isolates and other foreign threads can call
  exported functions. After the first call on a thread it is a thread-local flag
  check (no lock), so the per-call overhead is a few ns.
- Foreign threads may exit freely: a thread-local destructor unregisters each thread the
  library registered from the GC, including the thread that initialized the GC on the
  first call. A dead thread left registered makes the next collection signal a stale
  `pthread_t`, which bionic aborts on.
- Exceptions must not escape an exported function. Catch them and return an error code.

## Compile-time defines and platform

`__apple__` (macOS and iOS), `__macos__`, `__ios__`, `__android__`, `__linux__`
(Linux and Android) and `__windows__` follow the target. `sys.platform` is
`"android"` or `"ios"` on mobile (as in CPython 3.13+), and otherwise `"darwin"`,
`"linux"` or `"win32"`.

## Flutter

Bundle the `.so` files under `android/app/src/main/jniLibs/<abi>/` (`arm64-v8a`,
`x86_64`). On iOS, wrap the dylib in a framework or xcframework. Then open the
library with `DynamicLibrary.open('libfoo.so')` (Android) or by framework path (iOS),
and use `lookupFunction<Int64 Function(Int64, Int64), int Function(int, int)>('add')`.

## Tested on Android

The e2e suites run on an arm64 emulator (API 35) with
`TYPHON_TEST_TARGET=android-arm64` (see `crates/testsuite/README.md`). Every case is built on
the host, pushed with adb, and run in `/data/local/tmp`.

| suite | passed | failed | notes |
|---|---|---|---|
| `core/` | 38/38 | 0 | |
| `stdlib/` | 60/60 | 0 | on Android, tests copy files instead of hardlinking (SELinux denies `link(2)` in `/data/local/tmp`) and skip the locale grouping check (bionic `localeconv` is always C) |
| `types/` | 620/620 | 0 | |
| `opt/` | 26/26 | 0 | needed 64-byte TLS alignment (bionic loader requirement) |
| `cnumerics/` | 8/8 | 0 | |
| `numpy/` | 120/120 | 0 | includes linalg, dot/matmul, random legacy |
| `scipy/` | 38/38 | 0 | linalg (dsyevr, dsygvd, QR, ...) and signal (tf2sos) via OpenBLAS |

BLAS/LAPACK: OpenBLAS 0.3.32 on Android (ILP64, f2c LAPACK, threads via OpenMP on typhon-omp,
see `crates/runtime/README.md`). On iOS it is Accelerate, whose ILP64 LAPACK needs **iOS 16.4**
(the default deployment target of the `ios-*` aliases).

BLAS threading cost (emulator, 4 vCPU): small calls must not lose to 1 thread. Three fixes made
that hold: the runtime sets `GC_RETRY_SIGNALS=0` before `GC_init` (unless the user set it), because
bdwgc's retry path slept 3 ms per stop-the-world once idle workers existed. Parked typhon-omp
workers wait inside `GC_do_blocking`, so collections skip them. typhon-omp spins with
`spin_loop` instead of `sched_yield` and creates task deques lazily, which cuts fork/join from
4.9 to 2.4 us (libomp: 0.6 us). OpenBLAS `GEMM_MULTITHREAD_THRESHOLD` was left at its default
because raising it to 16 changed nothing.

Android fixes so far: `@thread_local` globals are aligned to 64 bytes, links pass
`--allow-multiple-definition` so a program `@export` can override a runtime hook (as on ld64),
and complex `log` avoids `clog`, which bionic only has from API 26.
