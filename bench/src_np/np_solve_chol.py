import sys, time
import numpy as np

n = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
reps = int(sys.argv[2]) if len(sys.argv) > 2 else 5
t0 = time.time()
m = np.sin(np.arange(n * n, dtype=np.float64)).reshape(n, n)
a = m @ m.T + n * np.eye(n)
b = np.cos(np.arange(n, dtype=np.float64))
sx = 0.0
sl = 0.0
for r in range(reps):
    x = np.linalg.solve(a, b)
    sx += float(x.sum())
    l = np.linalg.cholesky(a)
    sl += float(np.trace(l))
    b = b + 0.5
print("solve", sx)
print("chol", sl)
print(time.time() - t0)
