# ONNX Runtime C API (ORT_API_VERSION 31, ORT 1.31.x) via the OrtApi function-pointer table.
# Slot indices = field order in onnxruntime_c_api.h (append-only), see docs/onnx.md.
from internal.dlopen import dlopen, dlsym

ORT_API_VERSION = 31
I_GetErrorMessage = 2
I_CreateEnv = 3
I_CreateSession = 7
I_Run = 9
I_CreateSessionOptions = 10
I_CreateTensorWithDataAsOrtValue = 49
I_GetTensorMutableData = 51
I_CreateCpuMemoryInfo = 69
I_ReleaseEnv = 92
I_ReleaseStatus = 93
I_ReleaseMemoryInfo = 94
I_ReleaseSession = 95
I_ReleaseValue = 96
I_ReleaseSessionOptions = 100
I_AddInitializer = 150

ONNX_DOUBLE = 11
ONNX_INT64 = 7

if not __android__:
    from C import OrtGetApiBase() -> cobj

_lib = cobj()
_api = cobj()

def ort_api() -> cobj:
    # Android: dlopen by bare name (resolved in the APK native lib dir). Elsewhere: linked.
    global _lib, _api
    if _api != cobj():
        return _api
    base = cobj()
    if __android__:
        if _lib == cobj():
            _lib = dlopen("libonnxruntime.so")
        base = dlsym(_lib, "OrtGetApiBase", Function[[], cobj])()
    else:
        base = OrtGetApiBase()
    _api = Function[[u32], cobj](Ptr[cobj](base)[0])(u32(ORT_API_VERSION))
    if _api == cobj():
        raise RuntimeError("onnxruntime is older than API version 31")
    return _api

def fn(a: cobj, i: int) -> cobj:
    return Ptr[cobj](a)[i]

def check(a: cobj, st: cobj):
    # OrtStatus* null = success; otherwise raise with ORT's message
    if st == cobj():
        return
    msg = str.from_ptr(Ptr[byte](Function[[cobj], cobj](fn(a, I_GetErrorMessage))(st)))
    Function[[cobj], None](fn(a, I_ReleaseStatus))(st)
    raise RuntimeError("onnxruntime: " + msg)

def release(a: cobj, i: int, v: cobj):
    if v != cobj():
        Function[[cobj], None](fn(a, i))(v)

def tensor(a: cobj, mem: cobj, data: cobj, nbytes: int, dims: List[int], elem: int) -> cobj:
    # wraps caller memory without copying: must not outlive the call
    shape = Ptr[int](len(dims))
    for i in range(len(dims)):
        shape[i] = dims[i]
    v = cobj()
    check(a, Function[[cobj, cobj, int, Ptr[int], int, i32, Ptr[cobj]], cobj](fn(a, I_CreateTensorWithDataAsOrtValue))(
        mem, data, nbytes, shape, len(dims), i32(elem), __ptr__(v)))
    return v

class OrtModel:
    env: cobj
    sess: cobj
    mem: cobj

    def __init__(self, path: str, w: Ptr[float], b: Ptr[float]):
        # w: (12,3) lda_coefficients_transposed, b: (3,) lda_intercept; null = keep the model's own
        a = ort_api()
        self.env = cobj()
        self.sess = cobj()
        self.mem = cobj()
        check(a, Function[[i32, cobj, Ptr[cobj]], cobj](fn(a, I_CreateEnv))(i32(3), "bci_model".c_str(), __ptr__(self.env)))
        check(a, Function[[i32, i32, Ptr[cobj]], cobj](fn(a, I_CreateCpuMemoryInfo))(i32(1), i32(0), __ptr__(self.mem)))
        opts = cobj()
        check(a, Function[[Ptr[cobj]], cobj](fn(a, I_CreateSessionOptions))(__ptr__(opts)))
        vw = cobj()
        vb = cobj()
        try:
            if w != Ptr[float]():
                vw = tensor(a, self.mem, cobj(w.as_byte()), 36 * 8, [12, 3], ONNX_DOUBLE)
                vb = tensor(a, self.mem, cobj(b.as_byte()), 3 * 8, [3], ONNX_DOUBLE)
                add = Function[[cobj, cobj, cobj], cobj](fn(a, I_AddInitializer))
                check(a, add(opts, "lda_coefficients_transposed".c_str(), vw))
                check(a, add(opts, "lda_intercept".c_str(), vb))
            # CreateSession copies the initializers, buffers can go after it returns
            check(a, Function[[cobj, cobj, cobj, Ptr[cobj]], cobj](fn(a, I_CreateSession))(
                self.env, path.c_str(), opts, __ptr__(self.sess)))
        finally:
            release(a, I_ReleaseSessionOptions, opts)
            release(a, I_ReleaseValue, vw)
            release(a, I_ReleaseValue, vb)

    def predict(self, feats: Ptr[float], n: int, out: Ptr[int]):
        # input `features` double [n,12] -> output `label` int64 [n]
        a = ort_api()
        iv = tensor(a, self.mem, cobj(feats.as_byte()), n * 12 * 8, [n, 12], ONNX_DOUBLE)
        in_names = Ptr[cobj](1)
        in_names[0] = "features".c_str()
        out_names = Ptr[cobj](1)
        out_names[0] = "label".c_str()
        ins = Ptr[cobj](1)
        ins[0] = iv
        outs = Ptr[cobj](1)
        outs[0] = cobj()
        try:
            check(a, Function[[cobj, cobj, Ptr[cobj], Ptr[cobj], int, Ptr[cobj], int, Ptr[cobj]], cobj](fn(a, I_Run))(
                self.sess, cobj(), in_names, ins, 1, out_names, 1, outs))
            data = cobj()
            check(a, Function[[cobj, Ptr[cobj]], cobj](fn(a, I_GetTensorMutableData))(outs[0], __ptr__(data)))
            src = Ptr[int](data)
            for i in range(n):
                out[i] = src[i]
        finally:
            release(a, I_ReleaseValue, outs[0])
            release(a, I_ReleaseValue, iv)

    def close(self):
        a = ort_api()
        release(a, I_ReleaseSession, self.sess)
        release(a, I_ReleaseMemoryInfo, self.mem)
        release(a, I_ReleaseEnv, self.env)
        self.sess = cobj()
        self.mem = cobj()
        self.env = cobj()
