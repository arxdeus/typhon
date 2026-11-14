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

## Android / iOS

Verified on the Android arm64 emulator (API 35) and the iOS simulator (arm64) with the model `log_power_lda_model/artifacts/model.onnx` (input `features` double [N,12], output `label` int64 [N]). Minimal lib: `core.py` + `ort_android.py` / `ort_ios.py` in `/Users/mind/.jcode/scratch/ort-mobile/src/` (about 90 lines, session from a path, features to labels), driven by a C `dlopen` driver (`driver.c`). The OrtApi slot table is the same for API 30 and 31 for the slots used.

**Binaries**
* Android: Maven Central `com.microsoft.onnxruntime:onnxruntime-android:1.31.1` (AAR is a zip, 55 MB). `jni/<abi>/libonnxruntime.so` (arm64-v8a 34.2 MB, armeabi-v7a 24.2 MB, x86 and x86_64 also included) and `headers/` (API 31).
* iOS: CocoaPods `onnxruntime-c`. The newest pod is **1.30.0** (API 30), 1.31.x is not published for iOS yet. Zip: `https://download.onnxruntime.ai/pod-archive-onnxruntime-c-1.30.0.zip` containing `onnxruntime.xcframework` (slices `ios-arm64` 48 MB, `ios-arm64_x86_64-simulator` 98 MB fat, 48 MB per arch, plus macOS) and `Headers/`. The framework binary is a **static archive**, so it is linked into your image. Request API 30 there (`GetApi(30)`), it returns null for a newer version than the library has.

**Build**
```
typhon build -release -lib -target android-arm64 -o libort_android.so ort_android.py        # dlopen("libonnxruntime.so")
typhon build -release -lib -target ios-sim-arm64 -o libort_ios.dylib ort_ios.py \
  "-linker-flags=-F<pod>/onnxruntime.xcframework/ios-arm64_x86_64-simulator -framework onnxruntime -framework Foundation -framework CoreML -framework Network -framework Accelerate -framework UIKit -lc++ -lz"
```
For the device use `-target ios-arm64` with the `ios-arm64` slice.

**Android** uses dlopen: the Typhon `.so` (6.0 MB, no ORT dependency) and `libonnxruntime.so` go side by side into the APK `jniLibs/<abi>/`. `dlopen("libonnxruntime.so")` by bare name resolves in the app native library path (in the emulator test: `LD_LIBRARY_PATH` to the push dir). Linking with `-lonnxruntime` also works (DT_NEEDED).

**iOS** cannot dlopen a loose dylib in an app, and `onnxruntime.framework` is static, so link it (`from C import OrtGetApiBase() -> cobj`). The frameworks above are required by the CoreML and network bits in the archive (missing ones show as undefined `_OBJC_CLASS_$_UIDevice`, `_nw_path_*`, `_objc_retainBlock`). The resulting simulator dylib is 38 MB (ORT is inside it); in an app, link the archive into the app binary once instead. Package as an embedded framework or static link, per the Apple signing rules.

**Measured** (CPU time, 64 random feature rows scaled so that labels 0 and 2 both occur, 2000 calls each, labels compared with Python onnxruntime 1.31.1 on the same features):

| platform | ORT | label mismatches | session load | predict 1 row | predict 64 rows |
|---|---|---|---|---|---|
| Android arm64 emulator (host CPU) | 1.31.1 | 0 of 64 | 65 ms | 6.6 us | 6.9 us |
| iOS simulator arm64 | 1.30.0 | 0 of 64 | 87 ms | 8.4 us | 29.7 us |

Emulator and simulator run on the host CPU, so phone numbers will be slower. The iOS 64 row figure is higher than Android, probably because the simulator build uses the thread pool and CPU time counts all threads.

**Real library** (`test/bci/lib/ort/`, user preprocessing.py compiled + ORT predict, 60 random trials 6x500): features vs Python `preprocess` max abs 7.1e-15 (max rel 3e-16) on both platforms, labels 0 of 60 mismatches vs Python onnxruntime. The iOS build needs `ORT_API_VERSION = 30` in `ort_api.py` (pod is 1.30.0). Same build flags as above. Android lib 6.7 MB, iOS dylib 38.8 MB.

| platform | create | 1 trial (preprocess + ORT) | 60 trials |
|---|---|---|---|
| Android arm64 emulator | 88 ms | 519 us | 21.0 ms |
| iOS simulator arm64 | 97 ms | 637 us | 18.3 ms |

**Size.** The stock `libonnxruntime.so` is 34 MB (arm64) because it contains every operator kernel. A model that only needs MatMul, Add, ArgMax, Gather and similar can use an ORT **minimal build** (`--minimal_build` with `--include_ops_by_config required_operators.config` and `--disable_ml_ops`, `--disable_exceptions` optional), which needs the model converted to ORT format (`python -m onnxruntime.tools.convert_onnx_models_to_ort`, which also produces the operator config). Such builds are much smaller (size not measured here), but they load `.ort` files only and must be built from source for each ABI (not done here). The iOS static archive shrinks the same way, and the linker already strips unused objects when the archive is linked into an app.

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

## Retraining an ONNX model on device (no ORT training APIs)

ORT on-device training is being retired upstream (last prebuilt training packages: 1.19.2), so
a retrainable ONNX model is kept as a plain inference graph and its weights are replaced at
session creation: train in Typhon's `torch`/`braindecode` port (`bci_nn.md`), save the
`state_dict`, then create the session from the ORIGINAL `.onnx` with one
`SessionOptions AddInitializer(name, OrtValue)` per trained tensor (same C API call as the LDA
retrain in `test/bci/lib/ort/ort_api.py`). Check: `test/bci/lib/ort/check_retrain_nn.py`.

Export: `torch.onnx.export(..., opset_version=17, do_constant_folding=False,
keep_initializers_as_inputs=False, dynamo=False)`. Initializer names equal the `state_dict` keys
(EEGNet: 18 float32 tensors incl. BatchNorm `running_mean`/`running_var`;
`num_batches_tracked` is not in the graph). Trap: the exporter deduplicates initializers with
equal values, and a fresh model has `running_mean == bias == 0`, `running_var == weight == 1`,
`bnorm_2 == bnorm_1`, so 10 initializers serve 18 tensors and cannot be overridden separately.
Fix (in the check script): export with unique random values, map names by value, then write the
real weights back into the `.onnx`.

EEGNet (6 ch, 3 classes, 500 samples), 3 epochs x 240 trials Adam in Typhon (loss 1.1693 ->
1.1042, all 18 tensors changed), inference on 240 trials vs PyTorch with the same trained
weights, onnxruntime 1.32.0 CPU, M-series host:

| ORT graph optimization | max abs logit diff | max abs proba diff | labels equal | session create ms (plain / overrides) | run 1 trial ms |
|---|---|---|---|---|---|
| ORT_DISABLE_ALL | 5.6e-8 | 3.0e-8 | yes | 1.2 / 1.3 | 0.11 |
| ORT_ENABLE_BASIC | 3.5e-2 | 1.0e-2 | no | 3.9 | 0.09 |
| ORT_ENABLE_EXTENDED / ALL | 3.5e-2 | 1.0e-2 | no | 3.6 / 4.0 | 0.10 |

(Original weights vs trained: 7.4e-2, so at BASIC+ the overrides are partly lost.)

Caveats:
- With any optimization above `ORT_DISABLE_ALL`, overrides of tensors that ORT constant-folds or
  fuses (Conv+BatchNorm fusion, the weight-norm parametrization of `conv_spatial`, BN stats)
  are silently dropped: per tensor, only `conv_separable_depth` and the classifier survived.
  No error. Use `ORT_DISABLE_ALL` (costs nothing measurable at EEGNet size).
- dtype mismatch (float64 OrtValue for a float32 initializer), shape mismatch and unknown names
  are all accepted silently (wrong dtype = the original weight is used). Validate name, dtype
  and shape against the graph yourself before `AddInitializer`.
- `AddInitializer` does not copy: the OrtValue and its buffer must stay alive until the session
  is released (the Typhon wrapper keeps them on the session object).
- Session creation is ~1 ms, so recreating the session after each retrain is fine.

Recommendation: keep one exported `.onnx` (dedup-safe export as above) as the deployed
artifact, train in Typhon, store weights as safetensors, and open ORT with
`ORT_DISABLE_ALL` + `AddInitializer` for every trained tensor after checking names, dtypes and
shapes. Alternative when optimization is wanted: rewrite the initializers into the `.onnx`
bytes and load that (no override semantics involved).
