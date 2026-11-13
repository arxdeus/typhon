# BCI model: compiled preprocessing + ONNX Runtime

The user's `log_power_lda_model/preprocessing.py` (mne IIR Butterworth 8-13 / 13-30 Hz, zero phase, log mean power) is compiled **unchanged** by Typhon into a C library. The trained head (`model.onnx`: `features` double [N,12] -> `label` int64 [N], MatMul/Add/ArgMax/Gather, opset 17) runs in ONNX Runtime through its C API (see `docs/onnx.md`).

```
raw uV (N,6,500) --[Typhon: preprocess()]--> features (N,12) --[ORT: model.onnx]--> label (N)
```

Sources: `test/bci/lib/ort/` (`bci_model.py` exports, `ort_api.py` ORT table bindings, `build.sh`, `check_parity.py`). `preprocessing.py` is not in the repo: `build.sh` copies it next to `bci_model.py` in a temp dir.

## C API

All return `long`; `rc == 0` success, `-2` bad input (ValueError: shape, NaN/Inf, N=0, bad handle), `-1` other errors (ORT). No exception escapes an export.

| function | |
|---|---|
| `long bci_create(const char* model_path)` | handle > 0, or -1 |
| `long bci_preprocess(const double* x, long n, double* out12)` | x: n*6*500 C order, out: n*12 |
| `long bci_preprocess_shape(const double* x, long n, long ch, long t, double* out)` | same, explicit shape |
| `long bci_predict(long h, const double* x, long n, int64_t* labels)` | preprocess + ORT |
| `long bci_predict_features(long h, const double* f, long n, int64_t* labels)` | ORT only |
| `long bci_set_lda(long h, const double* coef12x3, const double* intercept3)` | recalibration: new session with `AddInitializer` for `lda_coefficients_transposed` (= sklearn `coef_.T`, row-major 12x3) and `lda_intercept`; swapped in only on success |
| `long bci_last_error(char* buf, long cap)` | last message (UTF-8), returns its length |
| `long bci_destroy(long h)` | |

Classes: 0 Rest, 1 Foot, 2 Right Hand. Not thread-safe across handles creation/destroy (global table); `predict` on one handle from one thread at a time.

## Build

```
test/bci/lib/ort/build.sh <preprocessing.py> <ort_root> <out> [typhon args]   # TYPHON=path/to/typhon
```
* macOS: `build.sh .../preprocessing.py onnxruntime-osx-arm64-1.31.1 libbci_model.dylib` (links `-lonnxruntime`, rpath to `<ort_root>/lib`).
* iOS: link the `onnxruntime-c` xcframework: pass `<ort_root>` with `lib/` containing it, plus the iOS target args; ORT is resolved through the link (`from C import OrtGetApiBase`).
* Android: `build.sh ... - libbci_model.so <android target args>`: no link dependency, `dlopen("libonnxruntime.so")` at first use (put it into `jniLibs/<abi>/` from the `onnxruntime-android` AAR).

Requests ORT API version 30 (ORT >= 1.30); an older runtime gives `-1` from `bci_create`.

Mobile (validated by the mobile lane): Android arm64 emulator (ORT 1.31.1, dlopen) and iOS simulator (ORT 1.30.0 pod, linked): features max abs diff 7.1e-15, labels 0/60 mismatches vs Python; 1 trial 519 us (Android), 637 us (iOS).

## Results (macOS arm64, ORT 1.31.1)

* `preprocessing.py` compiled byte-identical: yes (no stdlib/compiler change needed).
* 120 random trials (seed 1234) and N=1 vs python (numpy 2.5.4, mne 1.13.2, onnxruntime 1.31.1): features max rel diff 3.0e-16 (not bitwise, last-ulp), labels identical; `predict_features` and `set_lda` (random head vs numpy argmax, then original head restored) identical.
* Errors: NaN, N=0, wrong shape, bad handle -> -2 with the original message; missing model -> -1.
* Single-trial preprocess+predict, CPU time: Typhon 0.34-0.36 ms vs Python 2.5-2.9 ms (about 7-8x).
* Build: about 18 s CPU (release). Regression (no ORT): `test/bci/test_preprocessing_ort.py`.
