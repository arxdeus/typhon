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
