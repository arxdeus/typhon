# npbench jacobi_2d (slicing stencil)
import sys, time
import numpy as np

n = int(sys.argv[1]) if len(sys.argv) > 1 else 700
tsteps = int(sys.argv[2]) if len(sys.argv) > 2 else 200
t0 = time.time()
ii = np.arange(n, dtype=np.float64).reshape(n, 1)
jj = np.arange(n, dtype=np.float64).reshape(1, n)
A = ii * (jj + 2) / n
B = ii * (jj + 3) / n
for t in range(1, tsteps):
    B[1:-1, 1:-1] = 0.2 * (A[1:-1, 1:-1] + A[1:-1, :-2] + A[1:-1, 2:] + A[2:, 1:-1] + A[:-2, 1:-1])
    A[1:-1, 1:-1] = 0.2 * (B[1:-1, 1:-1] + B[1:-1, :-2] + B[1:-1, 2:] + B[2:, 1:-1] + B[:-2, 1:-1])
print("sum", float(A.sum()), float(B.sum()))
print(time.time() - t0)
