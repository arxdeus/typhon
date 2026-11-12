import sys, time
import numpy as np

iters = int(sys.argv[1]) if len(sys.argv) > 1 else 50000
t0 = time.time()
a4 = np.sin(np.arange(16, dtype=np.float64)).reshape(4, 4) + 4.0 * np.eye(4)
a8 = np.cos(np.arange(64, dtype=np.float64)).reshape(8, 8) + 8.0 * np.eye(8)
b8 = np.arange(8, dtype=np.float64)
acc = 0.0
for i in range(iters):
    e = 1e-6 * (i % 100)
    p = a4 @ a4
    q = np.linalg.inv(a4 + e)
    x = np.linalg.solve(a8, b8 + e)
    d = np.dot(a8[0], x)
    acc += p[0, 0] + q[1, 1] + d
print("acc", acc)
print(time.time() - t0)
