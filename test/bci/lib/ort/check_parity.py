# Host parity check: python check_parity.py <model.onnx> <dir with preprocessing.py> <libbci_model.dylib>
# needs numpy, mne, onnx, onnxruntime (python).
import ctypes, sys, time, numpy as np, onnxruntime as ort
sys.path.insert(0, sys.argv[2])
from preprocessing import preprocess
M = sys.argv[1]
lib = ctypes.CDLL(sys.argv[3])
P = ctypes.c_void_p
for f in ("bci_preprocess", "bci_predict", "bci_predict_features"):
    getattr(lib, f).restype = ctypes.c_long
lib.bci_create.restype = ctypes.c_long
def err():
    b = ctypes.create_string_buffer(512); lib.bci_last_error(b, ctypes.c_long(512)); return b.value.decode()
h = lib.bci_create(M.encode()); assert h > 0, err()
sess = ort.InferenceSession(M)
rng = np.random.default_rng(1234)
X = np.ascontiguousarray(rng.normal(0, 20, (120, 6, 500)))
def ty(X):
    n = X.shape[0]; F = np.empty((n, 12)); L = np.empty(n, np.int64)
    X = np.ascontiguousarray(X, dtype=np.float64)
    rc1 = lib.bci_preprocess(P(X.ctypes.data), ctypes.c_long(n), P(F.ctypes.data))
    rc2 = lib.bci_predict(ctypes.c_long(h), P(X.ctypes.data), ctypes.c_long(n), P(L.ctypes.data))
    return rc1, rc2, F, L
for name, Xs in (("120", X), ("N=1", X[:1])):
    rc1, rc2, F, L = ty(Xs)
    Fr = preprocess(Xs); Lr = sess.run(["label"], {"features": Fr})[0]
    rel = np.max(np.abs(F - Fr) / np.abs(Fr))
    print(name, rc1, rc2, "bitwise", np.array_equal(F, Fr), "maxrel", rel, "labels eq", np.array_equal(L, Lr), np.bincount(Lr, minlength=3))
    L2 = np.empty(len(L), np.int64)
    assert lib.bci_predict_features(ctypes.c_long(h), P(Fr.ctypes.data), ctypes.c_long(len(L)), P(L2.ctypes.data)) == 0
    assert np.array_equal(L2, Lr)
# errors
Xb = X[:2].copy(); Xb[1, 3, 7] = np.nan
print("nan", ty(Xb)[:2], err())
Xw = np.zeros((1, 5, 500)); F = np.empty(12)
print("shape", lib.bci_preprocess_shape(P(Xw.ctypes.data), ctypes.c_long(1), ctypes.c_long(5), ctypes.c_long(500), P(F.ctypes.data)), err())
print("n=0", ty(X[:0])[:2], err())
print("bad handle", lib.bci_predict_features(ctypes.c_long(99), P(X.ctypes.data), ctypes.c_long(1), P(F.ctypes.data)), err())
print("bad model", lib.bci_create(b"/nonexistent.onnx"), err()[:80])
# set_lda: random head, compare with python ORT + numpy argmax
W = rng.normal(size=(12, 3)); B = rng.normal(size=3)
assert lib.bci_set_lda(ctypes.c_long(h), P(W.ctypes.data), P(B.ctypes.data)) == 0, err()
Fr = preprocess(X); L = np.empty(120, np.int64)
lib.bci_predict(ctypes.c_long(h), P(X.ctypes.data), ctypes.c_long(120), P(L.ctypes.data))
print("set_lda labels eq numpy", np.array_equal(L, np.argmax(Fr @ W + B, axis=1)))
# restore original head from model initializers
import onnx
init = {i.name: onnx.numpy_helper.to_array(i) for i in onnx.load(M).graph.initializer}
W0 = np.ascontiguousarray(init["lda_coefficients_transposed"]); B0 = np.ascontiguousarray(init["lda_intercept"])
lib.bci_set_lda(ctypes.c_long(h), P(W0.ctypes.data), P(B0.ctypes.data))
_, _, _, L = ty(X); print("restored eq", np.array_equal(L, sess.run(["label"], {"features": Fr})[0]))
# latency single trial
x1 = np.ascontiguousarray(X[:1]); l1 = np.empty(1, np.int64)
def bench(f, k=300):
    for _ in range(20): f()
    t = time.process_time()
    for _ in range(k): f()
    return (time.process_time() - t) / k * 1e6
tt = bench(lambda: lib.bci_predict(ctypes.c_long(h), P(x1.ctypes.data), ctypes.c_long(1), P(l1.ctypes.data)))
tp = bench(lambda: sess.run(["label"], {"features": preprocess(x1)}))
tpp = bench(lambda: preprocess(x1))
print(f"latency CPU us/trial: typhon preprocess+predict {tt:.1f}  python {tp:.1f} (preprocess alone {tpp:.1f})")
lib.bci_destroy(ctypes.c_long(h))
