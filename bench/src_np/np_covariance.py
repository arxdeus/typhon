# npbench covariance (reductions + per-column row-vector @ matrix)
import sys, time
import numpy as np

m = int(sys.argv[1]) if len(sys.argv) > 1 else 1600
n = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
t0 = time.time()
ii = np.arange(n, dtype=np.float64).reshape(n, 1)
jj = np.arange(m, dtype=np.float64).reshape(1, m)
data = (ii * jj) / m
float_n = float(n)
mean = np.mean(data, axis=0)
data -= mean
cov = np.zeros((m, m), dtype=np.float64)
for i in range(m):
    cov[i:m, i] = cov[i, i:m] = data[:, i] @ data[:, i:m] / (float_n - 1.0)
print("sum", float(cov.sum()), float(np.trace(cov)))
print(time.time() - t0)
