# Ported from torch 2.15.0 (BSD-3-Clause)
#
# Datasets. Typhon is statically typed, so datasets are duck-typed: any class
# with __len__ and __getitem__(int) works with DataLoader / Subset /
# random_split. `Dataset` is an optional base class.

import math
from torch._tensor import Tensor
from torch._random import Generator

class Dataset:
    _ds: int

    def __init__(self):
        self._ds = 0

    def __len__(self) -> int:
        raise NotImplementedError("Dataset.__len__")

def _index_all(tensors, i: int):
    if static.len(tensors) == 1:
        return (tensors[0][i],)
    elif static.len(tensors) == 2:
        return (tensors[0][i], tensors[1][i])
    elif static.len(tensors) == 3:
        return (tensors[0][i], tensors[1][i], tensors[2][i])
    elif static.len(tensors) == 4:
        return (tensors[0][i], tensors[1][i], tensors[2][i], tensors[3][i])
    else:
        compile_error("TensorDataset supports 1 to 4 tensors")

class _TensorDataset[T]:
    """TensorDataset(*tensors): item i is a tuple (t[i] for t in tensors)."""
    tensors: T

    def __init__(self, tensors: T):
        n = tensors[0].size(0)
        for t in tensors:
            if t.size(0) != n:
                raise ValueError("Size mismatch between tensors")
        self.tensors = tensors

    def __getitem__(self, index: int):
        return _index_all(self.tensors, index)

    def __len__(self) -> int:
        return self.tensors[0].size(0)

def TensorDataset(*tensors):
    return _TensorDataset(tensors)

class Subset[D]:
    dataset: D
    indices: List[int]

    def __init__(self, dataset: D, indices):
        self.dataset = dataset
        self.indices = [int(i) for i in indices]

    def __getitem__(self, idx: int):
        return self.dataset[self.indices[idx]]

    def __getitems__(self, indices: List[int]):
        return [self.dataset[self.indices[i]] for i in indices]

    def __len__(self) -> int:
        return len(self.indices)

class ConcatDataset[D]:
    datasets: List[D]
    cumulative_sizes: List[int]

    def __init__(self, datasets: List[D]):
        self.datasets = datasets
        if len(self.datasets) == 0:
            raise AssertionError("datasets should not be an empty iterable")
        self.cumulative_sizes = []
        s = 0
        for d in self.datasets:
            s += len(d)
            self.cumulative_sizes.append(s)

    def __len__(self) -> int:
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx: int):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        # bisect_right
        lo = 0
        hi = len(self.cumulative_sizes)
        while lo < hi:
            mid = (lo + hi) // 2
            if idx < self.cumulative_sizes[mid]:
                hi = mid
            else:
                lo = mid + 1
        di = lo
        si = idx if di == 0 else idx - self.cumulative_sizes[di - 1]
        return self.datasets[di][si]

def random_split(dataset, lengths, generator: Optional[Generator] = None):
    from torch import randperm
    n = len(dataset)
    lens = List[int]()
    is_frac = False
    total = 0.0
    for l in lengths:
        total += float(l)
    if isinstance(lengths, List[float]) or isinstance(lengths, Tuple[float, float]) or isinstance(lengths, Tuple[float, float, float]):
        if math.isclose(total, 1.0):
            is_frac = True
    if is_frac:
        for f in lengths:
            if float(f) < 0 or float(f) > 1:
                raise ValueError("Fraction is not between 0 and 1")
            lens.append(int(math.floor(n * float(f))))
        rem = n - sum(lens)
        for i in range(rem):
            lens[i % len(lens)] += 1
    else:
        for l in lengths:
            lens.append(int(l))
    if sum(lens) != n:
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")
    perm = randperm(n, generator=generator)
    indices = [int(perm[i].item()) for i in range(n)]
    out = []
    off = 0
    for l in lens:
        out.append(Subset(dataset, indices[off:off + l]))
        off += l
    return out
