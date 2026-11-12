# Expected joblib.hash digests for test/joblib/test_joblib.py (joblib 1.6.0, numpy 2.x).
import numpy as np, joblib
a = np.arange(24, dtype=float).reshape(2, 3, 4) / 7.0
cases = {
    "f3d": a, "f2d": a[0], "fT": a[0].T, "fslice": a[:, 1, ::2],
    "i1d": np.arange(10), "big": np.ones((70000, 2)), "sq": np.zeros((300, 300)),
    "b": np.array([True, False, True]), "f32": np.arange(5, dtype=np.float32),
    "i32": np.arange(5, dtype=np.int32), "scalar": np.array(3.5),
    "d4": np.zeros((2, 2, 2, 2)),
}
for k, v in cases.items():
    print(f'    "{k}": "{joblib.hash(v)}",')
