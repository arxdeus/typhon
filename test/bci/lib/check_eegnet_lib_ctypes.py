import ctypes, sys, time, numpy as np, torch
from safetensors.torch import load_file
D = "/Users/mind/.jcode/scratch/bci-nn/"
L = ctypes.CDLL(sys.argv[1])
P = ctypes.POINTER
L.create.restype = ctypes.c_int64; L.create.argtypes = [ctypes.c_int64]
L.load_weights.argtypes = [ctypes.c_int64, ctypes.c_char_p]
L.predict.argtypes = [ctypes.c_int64, P(ctypes.c_double), ctypes.c_int64, P(ctypes.c_double)]
L.train_step.restype = ctypes.c_double
L.train_step.argtypes = [ctypes.c_int64, P(ctypes.c_double), P(ctypes.c_int64), ctypes.c_int64]
d = load_file(D + "data.safetensors"); X = d["X"].numpy().astype(np.float64); y = d["y"].numpy().astype(np.int64)
ref = torch.softmax(load_file(D + "eegnet_out.safetensors")["o64"], 1).numpy()
h = L.create(0); assert L.load_weights(h, (D + "eegnet.safetensors").encode()) == 0
out = np.zeros((64, 3))
dp = lambda a: a.ctypes.data_as(P(ctypes.c_double))
L.predict(h, dp(X[:64]), 64, dp(out))
print("predict max abs diff vs torch:", np.abs(out - ref).max())
t = time.process_time()
for _ in range(100): L.predict(h, dp(X[:1]), 1, dp(out))
print("predict 1 trial ms:", (time.process_time() - t) * 10)
losses = []
t = time.process_time()
for ep in range(5):
    for i in range(0, 240, 64):
        xb = np.ascontiguousarray(X[i:i+64]); yb = np.ascontiguousarray(y[i:i+64])
        losses.append(L.train_step(h, dp(xb), yb.ctypes.data_as(P(ctypes.c_int64)), len(yb)))
print("5 epochs ms:", (time.process_time() - t) * 1e3, "loss first/last", losses[0], losses[-1])
