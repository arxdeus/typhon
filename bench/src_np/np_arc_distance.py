# npbench arc_distance (element-wise ufunc chain)
import sys, time
import numpy as np

n = int(sys.argv[1]) if len(sys.argv) > 1 else 10000000
reps = int(sys.argv[2]) if len(sys.argv) > 2 else 3
t0 = time.time()
i = np.arange(n, dtype=np.float64)
theta_1 = np.sin(i) * 0.5 + 0.5
phi_1 = np.cos(i * 1.3) * 0.5 + 0.5
theta_2 = np.sin(i * 0.7) * 0.5 + 0.5
phi_2 = np.cos(i * 0.9) * 0.5 + 0.5
s = 0.0
for r in range(reps):
    temp = np.sin((theta_2 - theta_1) / 2)**2 + np.cos(theta_1) * np.cos(theta_2) * np.sin((phi_2 - phi_1) / 2)**2
    d = 2 * (np.arctan2(np.sqrt(temp), np.sqrt(1 - temp)))
    s += float(d.sum())
print("sum", s)
print(time.time() - t0)
