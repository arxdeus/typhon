import sys, time
import numpy as np

n = int(sys.argv[1]) if len(sys.argv) > 1 else 500
reps = int(sys.argv[2]) if len(sys.argv) > 2 else 4
t0 = time.time()
m = np.sin(np.arange(n * n, dtype=np.float64) * 0.37).reshape(n, n)
sym = m + m.T
se = 0.0
ss = 0.0
for r in range(reps):
    w, v = np.linalg.eigh(sym)
    se += float(w.sum()) + float(np.abs(w).max())
    s = np.linalg.svd(m, compute_uv=False)
    ss += float(s.sum())
    sym = sym + np.eye(n)
    m = m + 0.01 * np.eye(n)
print("eigh", se)
print("svd", ss)
print(time.time() - t0)
