"""check_lda_regression.py <lib.dylib> [<baseline lib.dylib>]

LDA (log-power + LDA, examples/bci_flutter variant B) regression check on assets/reference.json + 240
synthetic trials (same generator as bench_host/gen_data.py):
  features vs Python preprocessing.py (needs mne in the running python), labels vs reference.json,
  proba vs ORT model_full.onnx (if onnxruntime importable), calib fit vs sklearn LDA,
  and, given a baseline lib, bit-for-bit equality of features/labels/proba + timing ratio.
Exit 1 on any failure."""
import ctypes as C, json, os, sys, time, importlib.util
import numpy as np

H = os.path.dirname(os.path.abspath(__file__))
APP = os.path.join(H, "../../examples/bci_flutter")
PRE = os.environ.get("BCI_PREPROCESSING_PY",
    "/Users/mind/orca/codon/experiments/bci-research-lab/research/src/models/log_power_lda_model/preprocessing.py")
D = C.POINTER(C.c_double); I = C.POINTER(C.c_int64)
dp = lambda a: a.ctypes.data_as(D); ip = lambda a: a.ctypes.data_as(I)
fails = []

def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name + (" " + info if info else ""))
    if not ok: fails.append(name)

def load(path):
    L = C.CDLL(os.path.abspath(path))
    L.bci_preprocess.argtypes = [D, C.c_int64, D]
    L.bci_lda_set.argtypes = [D, D, I]
    L.bci_full_predict.argtypes = [D, C.c_int64, I, D]
    for n in ("bci_preprocess", "bci_lda_set", "bci_full_predict"): getattr(L, n).restype = C.c_int64
    return L

ref = json.load(open(os.path.join(APP, "assets/reference.json")))
Xr = np.array([t["x"] for t in ref["trials"]], dtype=np.float64).reshape(-1, 6, 500)
yr = np.array([t["label"] for t in ref["trials"]], dtype=np.int64)
rng = np.random.default_rng(0)
base = Xr[rng.integers(0, len(Xr), 240)]
X = np.ascontiguousarray(np.concatenate([Xr, base * rng.uniform(0.7, 1.4, (240, 6, 1)) + rng.normal(0, 5.0, (240, 6, 500))]))
N = len(X)
lda = json.load(open(os.path.join(APP, "assets/lda.json")))
def arr(*keys):
    for k in keys:
        if k in lda: return np.ascontiguousarray(np.array(lda[k], dtype=np.float64).ravel())
    raise KeyError(keys)
coef = arr("coef", "coef_"); icpt = arr("intercept", "intercept_")
cls = np.ascontiguousarray(np.array(lda.get("classes", lda.get("classes_", [0, 1, 2])), dtype=np.int64))

def run(L):
    F = np.zeros((N, 12)); lab = np.zeros(N, np.int64); pr = np.zeros((N, 3))
    assert L.bci_preprocess(dp(X), N, dp(F)) == 0
    assert L.bci_lda_set(dp(coef), dp(icpt), ip(cls)) == 0
    assert L.bci_full_predict(dp(X), N, ip(lab), dp(pr)) == 0
    return F, lab, pr

def timing(L, reps=200):
    one = np.ascontiguousarray(X[:1]); f = np.zeros(12); l = np.zeros(1, np.int64); p = np.zeros(3)
    ts = {}
    for name, fn in (("preprocess", lambda: L.bci_preprocess(dp(one), 1, dp(f))),
                     ("full_predict", lambda: L.bci_full_predict(dp(one), 1, ip(l), dp(p)))):
        for _ in range(10): fn()
        t0 = time.process_time()
        for _ in range(reps): fn()
        ts[name] = (time.process_time() - t0) / reps * 1e3
    return ts

L = load(sys.argv[1])
F, lab, pr = run(L)
check("labels reference 24/24", (lab[:24] == yr).all(), f"{int((lab[:24] == yr).sum())}/24")
try:
    spec = importlib.util.spec_from_file_location("preprocessing", PRE); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    Fp = np.asarray(m.preprocess(X.copy()), dtype=np.float64)
    d = float(np.abs(F - Fp).max()); check("features vs python", d < 1e-12, f"max abs {d:.2e}")
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    yc = np.ascontiguousarray(np.concatenate([yr, rng.integers(0, 3, N - 24)]).astype(np.int64))
    s = LDA().fit(F, yc)
    L.bci_calib_reset(); L.bci_calib_add_features.argtypes = [D, C.c_int64, I]; L.bci_calib_fit.argtypes = [D, D, I]
    assert L.bci_calib_add_features(dp(np.ascontiguousarray(F)), N, ip(yc)) == 0
    c2 = np.zeros(36); b2 = np.zeros(3); k2 = np.zeros(3, np.int64)
    assert L.bci_calib_fit(dp(c2), dp(b2), ip(k2)) == 0
    d = max(float(np.abs(c2 - s.coef_.ravel()).max() / np.abs(s.coef_).max()), float(np.abs(b2 - s.intercept_).max() / np.abs(s.intercept_).max()))
    check("calib fit vs sklearn LDA", d < 1e-9, f"max rel {d:.2e}")
except ImportError as e:
    print("SKIP python reference:", e)
try:
    import onnxruntime as ort
    sess = ort.InferenceSession(os.path.join(APP, "assets/model_full.onnx"))
    inp = sess.get_inputs()[0]
    x_in = X.astype(np.float32) if "float)" in inp.type else X
    outs = sess.run(None, {inp.name: x_in})
    po = [o for o in outs if np.asarray(o).ndim == 2 and np.asarray(o).shape[1] == 3]
    if po:
        d = float(np.abs(pr - np.asarray(po[0], dtype=np.float64)).max()); check("proba vs ORT model_full", d < 1e-6, f"max abs {d:.2e}")
except ImportError as e:
    print("SKIP ORT:", e)

if len(sys.argv) > 2:
    B = load(sys.argv[2])
    Fb, labb, prb = run(B)
    check("features bit-identical vs baseline", np.array_equal(F, Fb))
    check("labels identical vs baseline", np.array_equal(lab, labb))
    check("proba bit-identical vs baseline", np.array_equal(pr, prb), f"max abs {float(np.abs(pr - prb).max()):.1e}")
    tn, tb = [], []
    for _ in range(9):  # interleaved rounds, min CPU time (robust to background load)
        tn.append(timing(L)); tb.append(timing(B))
    for k in ("preprocess", "full_predict"):
        a = float(np.min([t[k] for t in tn])); b = float(np.min([t[k] for t in tb]))
        check(f"cpu {k} not >5% slower", a <= b * 1.05, f"new {a:.4f} ms vs base {b:.4f} ms ({(a / b - 1) * 100:+.1f}%)")
else:
    print("timing", timing(L))
sys.exit(1 if fails else 0)
