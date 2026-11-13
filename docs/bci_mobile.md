# BCI log-power + LDA library on Android and iOS

`test/bci/lib/logpower_lib.py` (log band-power features from two IIR band-pass filters, then LDA)
is compiled to a shared library and gives the same results as scikit-learn on the host, on an
Android arm64 emulator and in the iOS simulator.

## Build

```sh
tools/build-runtime.sh aarch64-linux-android x86_64-linux-android aarch64-apple-ios-sim aarch64-apple-ios
typhon build -release -lib -target android-arm64 -o liblogpower.so    test/bci/lib/logpower_lib.py
typhon build -release -lib -target android-x64   -o liblogpower_x64.so test/bci/lib/logpower_lib.py
typhon build -release -lib -target ios-sim-arm64 -o liblogpower.dylib test/bci/lib/logpower_lib.py   # simulator
typhon build -release -lib -target ios-arm64     -o liblogpower.dylib test/bci/lib/logpower_lib.py   # device
```

Android needs the NDK (`$ANDROID_HOME/ndk/*`), iOS needs Xcode (`xcrun`). See `mobile.md`.

## C API (all `int64`, trials are row-major `double[n][6][500]`, labels `int64`)

| function | meaning |
|---|---|
| `create() -> h` | new model handle |
| `feat(X, n, out[n*12]) -> rc` | log-power features |
| `fit(h, X, y, n) -> rc` | fit from raw trials |
| `add_trials(h, X, y, n)` / `add_features(h, F, y, n)` | append calibration data |
| `refit(h) -> rc` | refit on all stored data |
| `predict(h, X, n, labels[n], proba[n*n_classes]) -> n_classes` | predict |
| `save(h, path) -> rc` / `load(path) -> h` | text model file |
| `n_trials(h)` | stored trials |

## Check driver

`test/bci/lib/mobile/gen_ref.py` writes the fixed dataset (60 trials, 6x500) and sklearn reference
(features, labels, probabilities) as binary files. `driver.c` dlopens the library, fits, predicts,
compares, times, and does a save/load roundtrip. Build and run:

```sh
python gen_ref.py data/
$NDK/toolchains/llvm/prebuilt/darwin-x86_64/bin/clang --target=aarch64-linux-android24 -O2 driver.c -o driver -ldl -lm
adb push driver liblogpower.so data /data/local/tmp/b && adb shell 'cd /data/local/tmp/b && ./driver ./liblogpower.so data .'
xcrun --sdk iphonesimulator clang -target arm64-apple-ios16.4-simulator -O2 driver.c -o driver_ios
xcrun simctl spawn <udid> $PWD/driver_ios $PWD/liblogpower.dylib $PWD/data $PWD/tmp
```

## Measured (CPU time, release)

| platform | max abs diff proba vs sklearn | label mismatches | feature max abs diff | fit 60 trials | predict 1 trial | save/load diff |
|---|---|---|---|---|---|---|
| host (macOS arm64) | 0 | 0 | 7.1e-15 | 15.4 ms | 0.35 ms | 0 |
| Android arm64 emulator (API 35, on the host CPU) | 0 | 0 | 7.1e-15 | 13.2 ms | 0.26 ms | 0 |
| iOS simulator (arm64) | 0 | 0 | 7.1e-15 | 20.6 ms | 1.05 ms | 0 |

The emulator runs arm64 code on an arm64 host, so its numbers say little about a phone. The
simulator figures include cold caches (200 repeated calls, first-use overhead). The Android x86_64
library builds (`android-x64`) but was not run.

## Android loader note

bionic only runs `.init_array`, not `.ctors`. Module globals are initialized by a global
constructor, so before this was fixed a library loaded with `dlopen` on Android kept all module
globals zeroed (for example a constant tuple of strings looked empty and `filter_data` reported
"ftype must be in filter_dict"). The Android target now emits the constructor into `.init_array`
(`setup_global_ctor` in `crates/llvm/src/llvisitor/output.rs`).

## Flutter (`dart:ffi`)

```dart
import 'dart:ffi';
import 'dart:io';
import 'package:ffi/ffi.dart';

final lib = Platform.isAndroid
    ? DynamicLibrary.open('liblogpower.so')
    : DynamicLibrary.open('liblogpower.framework/liblogpower'); // dylib wrapped in a framework
final create = lib.lookupFunction<Int64 Function(), int Function()>('create');
final fit = lib.lookupFunction<Int64 Function(Int64, Pointer<Double>, Pointer<Int64>, Int64),
    int Function(int, Pointer<Double>, Pointer<Int64>, int)>('fit');
final predict = lib.lookupFunction<
    Int64 Function(Int64, Pointer<Double>, Int64, Pointer<Int64>, Pointer<Double>),
    int Function(int, Pointer<Double>, int, Pointer<Int64>, Pointer<Double>)>('predict');

int classify(int h, Float64List trial /* 6*500 */) {
  final x = malloc<Double>(3000)..asTypedList(3000).setAll(0, trial);
  final lab = malloc<Int64>(1), pr = malloc<Double>(8);
  predict(h, x, 1, lab, pr);
  final r = lab.value;
  malloc.free(x); malloc.free(lab); malloc.free(pr);
  return r;
}
```

Allocate `proba` with room for `n * n_classes` doubles. Exports register the calling thread with
the GC, so calling from a Dart isolate is fine.

## Gradient boosting on device

`HistGradientBoostingClassifier` and `GradientBoostingClassifier` (stdlib `sklearn.ensemble`) compile
and train inside a Typhon program or library, so on-device training works. Synthetic 3-class data,
600 trials, 50 iterations, host macOS arm64 release, single thread (wall time of a single-threaded
loop; reference is scikit-learn 1.9.1 with `OMP_NUM_THREADS=1`):

| data | model | fit Typhon / sklearn | predict 1 row Typhon / sklearn | max abs proba diff | label mismatches |
|---|---|---|---|---|---|
| 600x12 (log-power) | HistGB | 92 / 88 ms | 0.007 / 0.50 ms | 0 | 0 |
| 600x12 | GB | 215 / 182 ms | 0.003 / 0.23 ms | 0 | 0 |
| 600x21 (tangent space) | HistGB | 161 / 115 ms | 0.006 / 0.47 ms | 0 | 0 |
| 600x21 | GB | 369 / 286 ms | 0.004 / 0.23 ms | 4e-17 (sum, 1 ulp) | 0 |

Training is 1.0 to 1.4x the sklearn time, a single prediction is about 70x faster (no Python call
overhead). Library variant `test/bci/lib/gb_lib.py` (same exports as `logpower_lib.py`, model =
log-power features + `HistGradientBoostingClassifier(max_iter=50)`): host ctypes and Android arm64
emulator give proba diff 0 and 0 label mismatches against sklearn on the features; fit of 60 trials
is 24 ms on the emulator, a single predict 0.46 ms. `save` writes features and labels and `load`
refits (a tree ensemble has no compact parameter file, and the refit is deterministic, load diff 0).
Regression test: `test/bci/test_gb.py` (generator `gen_gb.py`).

XGBoost, LightGBM and CatBoost are not ported. A model trained offline with them can run on device
through ONNX Runtime from a Typhon library (see `docs/onnx.md`): convert with `onnxmltools`
(`convert_xgboost`, `convert_lightgbm`) or `skl2onnx` (sklearn models, including the Hist/GB ones)
to `.onnx`, then `ort_load`/`ort_run`. That gives inference only. Training on device stays with
the ported sklearn estimators (or ORT training artifacts, which do not cover tree models).
