# Ported from numpy 2.5.4 (BSD-3-Clause): expected values from the reference.
import numpy as np

v = np.array([3,1,2,1,3,2,1,2,0,3,1,2,2,1,0,3,3,1,2,0,1,1,2,3,0,2])
f = v.astype(float)
x = np.linspace(-2.0, 2.0, 9)

QS = [19, 8, 24, 14, 1, 3, 21, 20, 6, 10, 17, 13, 12, 22, 18, 25, 11, 7, 5, 2, 16, 9, 4, 23, 15, 0]
assert list(np.argsort(v)) == QS
assert list(np.argsort(f)) == QS
assert list(np.argsort(v, kind='stable')) == [8, 14, 19, 24, 1, 3, 6, 10, 13, 17, 20, 21, 2, 5, 7, 11, 12, 18, 22, 25, 0, 4, 9, 15, 16, 23]
assert list(np.argsort(f[:10])) == [8, 1, 3, 6, 2, 5, 7, 0, 4, 9]
assert list(np.lexsort((v, v % 2))) == [8, 14, 19, 24, 2, 5, 7, 11, 12, 18, 22, 25, 1, 3, 6, 10, 13, 17, 20, 21, 0, 4, 9, 15, 16, 23]

assert list(np.piecewise(x, [x < 0, x >= 0], (lambda t: -t, 5.0))) == [2.0, 1.5, 1.0, 0.5, 5.0, 5.0, 5.0, 5.0, 5.0]
assert list(np.piecewise(x, [x < 1, x > -1], [1.0, 2.0])) == [1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
assert list(np.piecewise(x, [x < -1], [7.0])) == [7.0, 7.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
print("ok")
