# BCI log-power LDA model as a C library: user's preprocessing.py compiled unchanged + ONNX Runtime.
# preprocessing.py is not part of this repo: build.sh copies it next to this file at build time.
import numpy as np
from preprocessing import preprocess
from ort_api import OrtModel

N_CH = 6
N_T = 500
NF = 12

class Handle:
    path: str
    model: OrtModel

handles = Dict[int, Handle]()
next_h = 1
last_msg = ""

def _fail(msg: str, rc: int) -> int:
    # rc -2: bad input (ValueError), -1: runtime/ORT error
    global last_msg
    last_msg = msg
    return rc

def _read(p: Ptr[float], n: int, c: int, t: int) -> np.ndarray[float, 3]:
    X = np.empty((n, c, t))
    q = X.data
    for k in range(n * c * t):
        q[k] = p[k]
    return X

def _features(x: Ptr[float], n: int, nch: int, nt: int, out: Ptr[float]):
    F = preprocess(_read(x, n, nch, nt))
    k = 0
    for i in range(n):
        for j in range(NF):
            out[k] = F[i, j]
            k += 1

def _get(h: int) -> Handle:
    if h not in handles:
        raise ValueError("invalid handle " + str(h))
    return handles[h]

@export
def bci_create(model_path: cobj) -> int:
    # returns handle > 0, or -1 (see bci_last_error)
    global next_h, last_msg
    try:
        path = str.from_ptr(Ptr[byte](model_path))
        m = OrtModel(path, Ptr[float](), Ptr[float]())
        h = next_h
        next_h += 1
        handles[h] = Handle(path, m)
        return h
    except ValueError as e:
        return _fail("ValueError: " + e.message, -2)
    except Exception as e:
        return _fail("Error: " + e.message, -1)

@export
def bci_preprocess_shape(x: Ptr[float], n: int, nch: int, nt: int, out: Ptr[float]) -> int:
    # generic shape entry (lets the host test the shape check); out: n*12 doubles
    try:
        _features(x, n, nch, nt, out)
        return 0
    except ValueError as e:
        return _fail("ValueError: " + e.message, -2)
    except Exception as e:
        return _fail("Error: " + e.message, -1)

@export
def bci_preprocess(x: Ptr[float], n: int, out: Ptr[float]) -> int:
    # x: n*6*500 doubles (uV, C order), out: n*12 doubles
    return bci_preprocess_shape(x, n, N_CH, N_T, out)

@export
def bci_predict_features(h: int, feats: Ptr[float], n: int, out_labels: Ptr[int]) -> int:
    try:
        if n <= 0:
            raise ValueError("n must be positive")
        _get(h).model.predict(feats, n, out_labels)
        return 0
    except ValueError as e:
        return _fail("ValueError: " + e.message, -2)
    except Exception as e:
        return _fail("Error: " + e.message, -1)

@export
def bci_predict(h: int, x: Ptr[float], n: int, out_labels: Ptr[int]) -> int:
    try:
        m = _get(h)
        f = Ptr[float](max(n, 1) * NF)
        _features(x, n, N_CH, N_T, f)
        m.model.predict(f, n, out_labels)
        return 0
    except ValueError as e:
        return _fail("ValueError: " + e.message, -2)
    except Exception as e:
        return _fail("Error: " + e.message, -1)

@export
def bci_set_lda(h: int, coef: Ptr[float], intercept: Ptr[float]) -> int:
    # coef: 12x3 row-major (= lda_coefficients_transposed, i.e. sklearn coef_.T), intercept: 3.
    # New session with AddInitializer overrides, swapped in only on success.
    try:
        m = _get(h)
        nm = OrtModel(m.path, coef, intercept)
        m.model.close()
        m.model = nm
        return 0
    except ValueError as e:
        return _fail("ValueError: " + e.message, -2)
    except Exception as e:
        return _fail("Error: " + e.message, -1)

@export
def bci_last_error(buf: Ptr[byte], cap: int) -> int:
    # copies the UTF-8 message (NUL-terminated, truncated to cap-1); returns full length
    p = Ptr[byte](last_msg.c_str())
    n = 0
    while p[n] != byte(0):
        n += 1
    if cap > 0:
        k = min(n, cap - 1)
        for i in range(k):
            buf[i] = p[i]
        buf[k] = byte(0)
    return n

@export
def bci_destroy(h: int) -> int:
    try:
        m = _get(h)
        m.model.close()
        del handles[h]
        return 0
    except ValueError as e:
        return _fail("ValueError: " + e.message, -2)
    except Exception as e:
        return _fail("Error: " + e.message, -1)
