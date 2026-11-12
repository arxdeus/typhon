import sys, time
import numpy as np

n = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
reps = int(sys.argv[2]) if len(sys.argv) > 2 else 3
t0 = time.time()
a = np.sin(np.arange(n * n, dtype=np.float64)).reshape(n, n)
b = np.cos(np.arange(n * n, dtype=np.float64)).reshape(n, n)
s = 0.0
for r in range(reps):
    c = a @ b
    s += float(c.sum())
    a = a + 1e-3 * c[0, 0]
print("sum", s)
print(time.time() - t0)
