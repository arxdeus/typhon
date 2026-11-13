# Calling ONNX Runtime from a Typhon library

A Typhon library (`typhon build -release -lib`, Android `.so`, iOS `.dylib`) can call ONNX Runtime (ORT) through its plain C API. Verified on macOS arm64 with ORT 1.31.1 (`ORT_API_VERSION` 31): a 2-layer MLP run from a Typhon `.dylib` driven by Python `ctypes` matches python `onnxruntime` bit for bit (max diff 0.0). Cost per `ort_run` call (batch 2x4, MLP) is about 8 us CPU, the same as the Python binding.

## The C API from Typhon

`OrtGetApiBase()->GetApi(ORT_API_VERSION)` returns an `OrtApi*`, a struct of function pointers. Read slot `i` as `Ptr[cobj](api)[i]` and call it through a `Function[[...], R]` cast. The slot index is the field order in `onnxruntime_c_api.h` and is fixed per API version (append-only). Get it with `offsetof(OrtApi, X) / 8`; for version 31: CreateEnv 3, CreateSession 7, Run 9, CreateSessionOptions 10, CreateTensorWithDataAsOrtValue 49, GetTensorMutableData 51, CreateCpuMemoryInfo 69, GetErrorMessage 2, ReleaseEnv 92, ReleaseStatus 93, ReleaseMemoryInfo 94, ReleaseSession 95, ReleaseValue 96, ReleaseSessionOptions 100. Request the API version you generated the table for. ORT returns a null table if the runtime is older.

Every call returns `OrtStatus*` (null is success). Check it, read `GetErrorMessage`, then `ReleaseStatus`.

```python
from C import OrtGetApiBase() -> cobj
base = OrtGetApiBase()                       # OrtApiBase { GetApi, GetVersionString }
api = Function[[u32], cobj](Ptr[cobj](base)[0])(u32(31))
create_env = Function[[i32, cobj, Ptr[cobj]], cobj](Ptr[cobj](api)[3])
```

Full working code: `ort_core.py` (bindings: load, run, free), `stateful.py` (exports) in the scratch lane dir `/Users/mind/.jcode/scratch/onnx-ffi/` (`exe.py` is the executable variant).

## Stateful exported library

```python
sessions = Dict[int, Session]()
@export
def ort_load(path: cobj) -> int: ...          # returns handle, -1 on error
@export
def ort_run(h: int, inp: Ptr[float32], rows: int, cols: int, out: Ptr[float32], m: int) -> int: ...
@export
def ort_free(h: int) -> int: ...
```

Globals (the session table) persist across calls from the host language. Build and call:

```
typhon build -release -lib -o libmodel.dylib stateful.py \
  "-linker-flags=-L<ort>/lib -lonnxruntime -Wl,-rpath,<ort>/lib"
```
```python
lib = ctypes.CDLL("./libmodel.dylib"); h = lib.ort_load(b"model.onnx")
lib.ort_run(h, x.ctypes.data, 2, 4, out.ctypes.data, 6)
```

## Link vs dlopen

* **Link**: `-linker-flags="-L<dir> -lonnxruntime"` and `from C import OrtGetApiBase() -> cobj`. The library records `@rpath/libonnxruntime.1.dylib` (macOS) or `libonnxruntime.so` (Android), so the loader must find ORT next to it. Tested: OK (exe and lib).
* **dlopen**: no link-time dependency. `from internal.dlopen import dlopen, dlsym`; `h = dlopen("/path/libonnxruntime.dylib")`; `dlsym(h, "OrtGetApiBase", Function[[], cobj])()`. The rest is identical. Tested: OK (`stateful_dl.py`, `otool -L` shows no onnxruntime dependency). Good when the ORT path is only known at runtime.

## Android

Take `libonnxruntime.so` from the `onnxruntime-android` AAR (`jni/<abi>/`), put it into `jniLibs/<abi>/` next to `libfoo.so`. Either link with `-L<dir> -lonnxruntime` (the loader resolves `DT_NEEDED` from the APK lib dir; Java must `System.loadLibrary("onnxruntime")` first on old API levels) or `dlopen("libonnxruntime.so")` by bare name, which resolves within the app's native library path. The AAR also ships the C headers (`headers/`). Not run here (no device test).

## iOS

dlopen of non-bundled dylibs is not allowed, and the App Store wants frameworks embedded and signed. Use the `onnxruntime-c` pod or xcframework and link it into the app (`-framework onnxruntime`, or `-L` + `-lonnxruntime` for the static variant). Resolve `OrtGetApiBase` through the link (`from C import`), or `dlopen` of the embedded `onnxruntime.framework/onnxruntime` by path inside the app bundle. Not run here.

## Training

Verified from the ORT install page: on-device training packages exist (`onnxruntime-training-android` AAR, `onnxruntime-training-c` pod, `onnxruntime-training-objc` pod, plus desktop `onnxruntime-training`). The regular inference packages and the release tgz used here do not contain the training API. Flow:

1. Offline, in Python: `onnxruntime.training.artifacts.generate_artifacts(model, requires_grad=..., loss=LossType.MSELoss, optimizer=OptimType.AdamW, artifact_directory=...)` yields `training_model.onnx`, `eval_model.onnx`, `optimizer_model.onnx`, `checkpoint`.
2. On device, with the training build of the library: `OrtGetApiBase()->GetApi(v)->GetTrainingApi(v)` returns an `OrtTrainingApi*` (another function pointer table, same slot technique, header `onnxruntime_training_c_api.h`): `CreateTrainingSession`, `TrainStep`, `OptimizerStep`, `LazyResetGrad`, `EvalStep`, `ExportModelForInferencing`, `SaveCheckpoint`.
It is the same mechanism as inference. This lane did not run training (needs the training build, not part of the downloaded package).

## Threading and memory

* `OrtSession::Run` is thread-safe. One session can be shared by several threads. Typhon exports register calling threads with the GC. Keep the session table access (the `Dict`) single-writer or guard it with a lock.
* Memory from `CreateTensorWithDataAsOrtValue` wraps the caller buffer without copying. Release the tensor before returning from the exported function. Never keep a pointer to GC memory inside ORT beyond a call: ORT allocates its own buffers, and the GC does not scan them.
* Output tensors created by `Run` (when you pass null outputs) belong to you: copy the data out (`GetTensorMutableData`) and `ReleaseValue`. Strings handed to ORT (`"x".c_str()`) need to stay alive during the call only.
* Release order: values, session, memory info, env.

## Retraining model.onnx on device

Model: `log_power_lda_model/artifacts/model.onnx` (IR 8, opset 17). Input `eeg_uV` (N,6,500) float64, outputs `label`, `probabilities`, `scores`, `features` (N,12). The Scan IIR filters and log power are fixed initializers. Only `lda_coefficients_transposed` (12,3) and `lda_intercept` (3) are the learnable head. Code: `/Users/mind/.jcode/scratch/onnx-ffi/lda_lib.py` (Typhon lib), `tl.py` (ctypes test), `lda/train.py` (training), data in `lda/`.

### A) Closed-form retrain (works, exact)

`lda_lib.py` exports `lda_load`, `lda_features`, `lda_proba`, `lda_retrain`, `lda_free`.
1. Run the session with output `features` on the calibration trials (90 synthetic trials, 3 classes).
2. Refit `sklearn.discriminant_analysis.LinearDiscriminantAnalysis()` (the Typhon stdlib port, solver svd) on `(features, y)` inside the Typhon lib.
3. Create a new session whose `SessionOptions` has `AddInitializer` (OrtApi slot 150) for `lda_coefficients_transposed` = `coef_.T` and `lda_intercept` = `intercept_`. No file is rewritten. The initializer tensors are copied by `CreateSession`, so the buffers can be released after it returns (they are kept alive here for simplicity). The override replaces the same-named initializer, with the same shape and dtype (float64).
4. Swap the session in the handle table, release the old one.

Result (host, ctypes): features equal python ORT exactly (0.0), probabilities after retrain equal `sklearn` LDA fit on the same features to 1.0e-14 (limit 1e-10). Accuracy on the calibration set 0.30 to 0.656 (the original head was trained on other data). Sessions are never mutated, a retrain is a new session (model load cost, ms).

Persisting an updated `model.onnx`: no protobuf writer exists in Typhon. Either keep only the two LDA arrays (24 bytes per float64 pair: 36+3 doubles) in your own file and call `AddInitializer` at every start, or write the updated file with Python `onnx` (replace the two initializers, `onnx.save`). Verified: the file rewritten that way gives probabilities equal to sklearn to 1.0e-14. On device, the first option is the practical one.

### B) ORT on-device training (gradient fine-tuning, not exact LDA)

Package availability, checked against the ORT install page and Maven Central: training packages exist for Android (`com.microsoft.onnxruntime:onnxruntime-training-android`, AAR with C headers and `libonnxruntime.so` in `jni/`) and iOS (CocoaPods `onnxruntime-training-c` and `-objc`). The page does not state a version, and the latest `onnxruntime-training-android` on Maven Central is 1.19.2 (Sep 2024), while the inference packages are at 1.31.1. So the training build lags the inference builds. Its API (`GetTrainingApi`) must match its own ORT version, so use its own headers.

Host check: `pip install onnxruntime-training` has no macOS wheel (Linux x86_64 only, 1.19.2). The macOS arm64 build is `onnxruntime-training-cpu` 1.19.2 from `https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT/pypi/simple/` (needs `torch`, `onnx<1.17`, `numpy<2`). With it, `lda/train.py` worked:
* Only the two LDA tensors are trainable: `generate_artifacts(model, requires_grad=[the 2], frozen_params=[all other initializers], loss=CrossEntropyLoss, optimizer=AdamW)`.
* Required model edits: ORT's training kernels do not support float64 gradient accumulators and the loss op mixes types, so the head is made float32 (features cast to float32 before MatMul, scores output as float32) and the model's outputs are reduced to `scores`. This applies to the training copy only. The Scan filters stay float64 and frozen.
* `Module` + `Optimizer`: 5 epochs of `train_step` + `optimizer.step` ran, loss 1.216, 1.164, 1.172, 1.186, 1.184 (default AdamW lr 1e-3, no tuning).

What it gives: softmax regression (cross entropy gradient descent) on top of the fixed features. It does not reproduce LDA (different objective, no closed form, depends on lr/epochs). From Typhon the calls are the same table technique: `GetTrainingApi(version)` then `CreateCheckpointState`/`LoadCheckpoint`, `CreateTrainingSession`, `TrainStep`, `OptimizerStep`, `LazyResetGrad`, `ExportModelForInferencing`. Not exercised from Typhon here (no macOS training C library was available to link, only the Python wheel).

Recommendation: for LDA the closed-form path (A) is exact, tiny, and works with the standard inference package. Use B only if a gradient-trained head is wanted.
