# Ported from numpy 2.5.4 (BSD-3-Clause)
#
# numpy.matrix: 2-D wrapper (matrix multiply for *, ** as matrix power,
# .T, .I, .A, .H). ndarray is a value type so matrix wraps a 2-D array.

import numpy as np
from .linalg.linalg import inv, matrix_power

@tuple
class matrix[T]:
    a: T

    @property
    def A(self):
        return self.a

    @property
    def T(self):
        return matrix(self.a.T)

    @property
    def H(self):
        return matrix(np.conjugate(self.a).T)

    @property
    def I(self):
        return matrix(inv(self.a))

    @property
    def shape(self):
        return self.a.shape

    @property
    def ndim(self):
        return 2

    @property
    def size(self):
        return self.a.size

    def __mul__(self, other):
        if isinstance(other, matrix):
            return matrix(np.matmul(self.a, other.a))
        elif isinstance(other, np.ndarray):
            return matrix(np.matmul(self.a, other))
        else:
            return matrix(self.a * other)

    def __rmul__(self, other):
        if isinstance(other, np.ndarray):
            return matrix(np.matmul(other, self.a))
        return matrix(other * self.a)

    def __pow__(self, n: int):
        return matrix(matrix_power(self.a, n))

    def __add__(self, other):
        if isinstance(other, matrix):
            return matrix(self.a + other.a)
        return matrix(self.a + other)

    def __sub__(self, other):
        if isinstance(other, matrix):
            return matrix(self.a - other.a)
        return matrix(self.a - other)

    def __neg__(self):
        return matrix(-self.a)

    def __getitem__(self, idx):
        r = self.a[idx]
        return r

    def sum(self, axis: Optional[int] = None):
        if axis is None:
            return self.a.sum()
        return matrix(self.a.sum(axis=axis, keepdims=True))

    def tolist(self):
        return self.a.tolist()

    def __str__(self):
        return str(self.a)

    def __repr__(self):
        s = repr(self.a)
        return 'matrix(' + s[6:-1].replace('\n', '\n       ') + ')' if s.startswith('array(') else s

def asmatrix(data):
    x = np.asarray(data)
    if x.ndim == 1:
        return matrix(x.reshape((1, x.size)))
    return matrix(x)

def mat(data):
    return asmatrix(data)

def bmat(blocks):
    return asmatrix(np.block(blocks))
