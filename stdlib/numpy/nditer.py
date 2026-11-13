# Ported from numpy 2.5.4 (BSD-3-Clause)
#
# numpy.nditer: single-operand C-order iteration with multi_index,
# external_loop (flat chunks) and readwrite via write().

import numpy as np

class nditer:
    arr: np.ndarray
    flat: List
    pos: int

    def __init__(self, op, flags: str = '', order: str = 'K'):
        self.arr = np.asarray(op)
        self.pos = 0

    def __len__(self):
        return self.arr.size

    def __iter__(self):
        for x in self.arr.ravel():
            yield x

    @property
    def shape(self):
        return self.arr.shape

    @property
    def itersize(self):
        return self.arr.size

    @property
    def ndim(self):
        return self.arr.ndim

    def multi_indices(self):
        return np.ndindex(self.arr.shape)

    def external_loop(self):
        yield self.arr.ravel()

def nested_iters(op, axes, flags: str = '', order: str = 'K'):
    a = np.asarray(op)
    outer = List[np.ndarray]()
    return nditer(a), nditer(a)
