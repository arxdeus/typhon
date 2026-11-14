"""Retrain an ONNX neural model without ORT training APIs (host check).

1. export braindecode EEGNet to ONNX, every trainable tensor (and BN stats) kept as a named initializer
2. train it in Typhon (libeegnet, ctypes: train_step + save_weights)
3. run the ORIGINAL .onnx in plain inference ORT, overriding initializers via SessionOptions.add_initializer
4. compare with torch on the Typhon-trained weights; time session creation

usage: python check_retrain_nn.py <libeegnet.dylib> [scratch_dir]
needs: torch, braindecode, onnx, onnxruntime, safetensors
"""
import ctypes, sys, time, numpy as np, torch, onnx, onnxruntime as ort
from onnx import numpy_helper
from safetensors.torch import load_file
from braindecode.models import EEGNet

LIB = sys.argv[1]
D = sys.argv[2] if len(sys.argv) > 2 else "/Users/mind/.jcode/scratch/bci-nn/"
ONNX = D + "eegnet_retrain.onnx"
TRAINED = D + "eegnet_typhon_trained.safetensors"

# --- 1. export ---------------------------------------------------------------------------------
m = EEGNet(n_chans=6, n_outputs=3, n_times=500).eval()
m.load_state_dict(load_file(D + "eegnet.safetensors"))
sd = {k: v.clone() for k, v in m.state_dict().items()}
# The TorchScript exporter DEDUPLICATES initializers with equal values (fresh BatchNorm:
# running_mean == bias == 0, running_var == weight == 1, bnorm_2 == bnorm_1), which would make
# several trainable tensors share one ONNX name. Export from a copy with unique random values,
# match names by value, then write the real weights back into the graph.
torch.manual_seed(123)
with torch.no_grad():
    for k, v in m.state_dict().items():
        if v.is_floating_point(): v.copy_(torch.rand_like(v) + 0.5)
probe = {k: v.clone() for k, v in m.state_dict().items()}
torch.onnx.export(m, torch.zeros(1, 6, 500), ONNX, opset_version=17, input_names=["x"],
                  output_names=["logits"], dynamic_axes={"x": {0: "n"}, "logits": {0: "n"}},
                  do_constant_folding=False, keep_initializers_as_inputs=False, dynamo=False)
model = onnx.load(ONNX)
inits = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}
mapping = {}
for k, v in probe.items():                             # match by (unique random) value
    hit = [n for n, a in inits.items() if a.shape == tuple(v.shape) and np.array_equal(a, v.numpy())]
    if len(hit) == 1: mapping[k] = hit[0]
assert len(set(mapping.values())) == len(mapping), "two tensors share one initializer"
for i in model.graph.initializer:                      # restore the real (pre-training) weights
    k = next(k for k, n in mapping.items() if n == i.name)
    i.CopyFrom(numpy_helper.from_array(sd[k].numpy(), i.name))
onnx.save(model, ONNX)
m.load_state_dict(sd)
print(f"state_dict tensors {len(sd)}, onnx initializers {len(inits)}, mapped {len(mapping)}")
for k in sd:
    print(f"  {k:40s} -> {mapping.get(k, 'MISSING'):40s} {tuple(sd[k].shape)} {sd[k].dtype}")
extra = sorted(set(inits) - set(mapping.values()))
print("  extra initializers (constants, not trained):", extra)

# --- 2. train in Typhon -------------------------------------------------------------------------
P = ctypes.POINTER
L = ctypes.CDLL(LIB)
L.create.restype = ctypes.c_int64; L.create.argtypes = [ctypes.c_int64]
L.load_weights.argtypes = [ctypes.c_int64, ctypes.c_char_p]
L.save_weights.argtypes = [ctypes.c_int64, ctypes.c_char_p]
L.train_step.restype = ctypes.c_double
L.train_step.argtypes = [ctypes.c_int64, P(ctypes.c_double), P(ctypes.c_int64), ctypes.c_int64]
d = load_file(D + "data.safetensors")
X = d["X"].numpy().astype(np.float64); y = d["y"].numpy().astype(np.int64)
h = L.create(0); assert L.load_weights(h, (D + "eegnet.safetensors").encode()) == 0
losses = []
for ep in range(3):
    for i in range(0, 240, 64):
        xb = np.ascontiguousarray(X[i:i+64]); yb = np.ascontiguousarray(y[i:i+64])
        losses.append(L.train_step(h, xb.ctypes.data_as(P(ctypes.c_double)),
                                   yb.ctypes.data_as(P(ctypes.c_int64)), len(yb)))
assert L.save_weights(h, TRAINED.encode()) == 0
print(f"typhon train 3 epochs: loss {losses[0]:.6f} -> {losses[-1]:.6f}")
new = load_file(TRAINED)
changed = [k for k in mapping if not torch.equal(new[k], sd[k])]
print(f"tensors changed by training: {len(changed)}/{len(mapping)}")

# --- 3. inference parity ------------------------------------------------------------------------
Xf = X[:240].astype(np.float32)
mt = EEGNet(n_chans=6, n_outputs=3, n_times=500).eval(); mt.load_state_dict(new)
with torch.no_grad(): ref = mt(torch.from_numpy(Xf)).numpy()

def session(level, overrides):
    so = ort.SessionOptions(); so.graph_optimization_level = level
    keep = []                                          # OrtValues must outlive the session
    for k, onnx_name in (overrides or {}).items():
        v = ort.OrtValue.ortvalue_from_numpy(np.ascontiguousarray(new[k].numpy()))
        so.add_initializer(onnx_name, v); keep.append(v)
    s = ort.InferenceSession(ONNX, so, providers=["CPUExecutionProvider"])
    s._keep = keep
    return s

def softmax(z): e = np.exp(z - z.max(1, keepdims=True)); return e / e.sum(1, keepdims=True)

L_ = ort.GraphOptimizationLevel
print("\n| ORT opt level | max abs logit diff | max abs proba diff | labels equal | session ms (no overrides / overrides) | run 1 trial ms |")
for name, lvl in [("DISABLE_ALL", L_.ORT_DISABLE_ALL), ("ENABLE_BASIC", L_.ORT_ENABLE_BASIC),
                  ("ENABLE_EXTENDED", L_.ORT_ENABLE_EXTENDED), ("ENABLE_ALL", L_.ORT_ENABLE_ALL)]:
    def tm(ov):
        ts = []
        for _ in range(10):
            t = time.perf_counter(); s = session(lvl, ov); ts.append((time.perf_counter() - t) * 1e3)
        return s, float(np.median(ts))
    _, t0 = tm(None)
    s, t1 = tm(mapping)
    out = s.run(None, {"x": Xf})[0]
    t = time.perf_counter()
    for _ in range(200): s.run(None, {"x": Xf[:1]})
    tr = (time.perf_counter() - t) * 5
    print(f"| {name} | {np.abs(out - ref).max():.2e} | {np.abs(softmax(out) - softmax(ref)).max():.2e} "
          f"| {(out.argmax(1) == ref.argmax(1)).all()} | {t0:.1f} / {t1:.1f} | {tr:.3f} |")
base = session(L_.ORT_ENABLE_ALL, None).run(None, {"x": Xf})[0]
print("sanity, original weights vs trained torch, max abs logit diff:", f"{np.abs(base - ref).max():.2e}")

# --- 4. edge cases ------------------------------------------------------------------------------
so = ort.SessionOptions()
k = changed[0]
try:
    so.add_initializer(mapping[k], ort.OrtValue.ortvalue_from_numpy(new[k].numpy().astype(np.float64)))
    o = ort.InferenceSession(ONNX, so, providers=["CPUExecutionProvider"]).run(None, {"x": Xf})[0]
    print("dtype mismatch: accepted at session creation and run, max abs logit diff", f"{np.abs(o - ref).max():.2e}")
except Exception as e:
    print("dtype mismatch (float64 for float32):", type(e).__name__, str(e).splitlines()[0][:150])
so = ort.SessionOptions()
try:
    so.add_initializer(mapping[k], ort.OrtValue.ortvalue_from_numpy(np.zeros((1,) + tuple(new[k].shape), np.float32)))
    ort.InferenceSession(ONNX, so, providers=["CPUExecutionProvider"]).run(None, {"x": Xf[:1]})
    print("shape mismatch: accepted (!)")
except Exception as e:
    print("shape mismatch:", type(e).__name__, str(e).splitlines()[0][:150])
so = ort.SessionOptions()
try:
    so.add_initializer("no_such_tensor", ort.OrtValue.ortvalue_from_numpy(np.zeros(1, np.float32)))
    ort.InferenceSession(ONNX, so, providers=["CPUExecutionProvider"]); print("unknown name: silently ignored")
except Exception as e:
    print("unknown name:", type(e).__name__, str(e).splitlines()[0][:150])
