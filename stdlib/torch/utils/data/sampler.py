# Ported from torch 2.15.0 (BSD-3-Clause)

from torch._tensor import Tensor
from torch.nn._util import _or
from torch._random import Generator

def _rand_int64_seed(generator: Optional[Generator]) -> int:
    # torch.empty((), dtype=torch.int64).random_(generator=g).item()
    from torch import _random_int64_full
    return _random_int64_full(generator)

def _perm_list(n: int, g: Generator) -> List[int]:
    from torch import randperm
    p = randperm(n, generator=g)
    return [int(p[i].item()) for i in range(n)]

class Sampler:
    _s: int
    def __init__(self):
        self._s = 0

class SequentialSampler:
    n: int
    def __init__(self, data_source):
        self.n = len(data_source)
    def indices(self, base_gen: Optional[Generator]) -> List[int]:
        return list(range(self.n))
    def __iter__(self):
        for i in range(self.n):
            yield i
    def __len__(self) -> int:
        return self.n

class RandomSampler:
    n: int
    replacement: bool
    _num_samples: Optional[int]
    generator: Optional[Generator]

    def __init__(self, data_source, replacement: bool = False, num_samples: Optional[int] = None, generator: Optional[Generator] = None):
        self.n = len(data_source)
        self.replacement = replacement
        self._num_samples = num_samples
        self.generator = generator

    @property
    def num_samples(self) -> int:
        return _or(self._num_samples, self.n)

    def indices(self) -> List[int]:
        from torch import randint
        g: Generator
        if self.generator is None:
            seed = _rand_int64_seed(None)
            g = Generator()
            g.manual_seed(seed)
        else:
            g = self.generator
        out = List[int]()
        ns = self.num_samples
        n = self.n
        if self.replacement:
            for _ in range(ns // 32):
                r = randint(0, n, [32], generator=g)
                out.extend([int(r[i].item()) for i in range(32)])
            k = ns % 32
            r = randint(0, n, [k], generator=g)
            out.extend([int(r[i].item()) for i in range(k)])
        else:
            for _ in range(ns // n):
                out.extend(_perm_list(n, g))
            out.extend(_perm_list(n, g)[:ns % n])
        return out

    def __iter__(self):
        for i in self.indices():
            yield i

    def __len__(self) -> int:
        return self.num_samples

class SubsetRandomSampler:
    idx: List[int]
    generator: Optional[Generator]

    def __init__(self, indices, generator: Optional[Generator] = None):
        self.idx = [int(i) for i in indices]
        self.generator = generator

    def indices(self) -> List[int]:
        from torch import randperm
        p = randperm(len(self.idx), generator=self.generator)
        return [self.idx[int(p[i].item())] for i in range(len(self.idx))]

    def __iter__(self):
        for i in self.indices():
            yield i

    def __len__(self) -> int:
        return len(self.idx)

class BatchSampler[S]:
    sampler: S
    batch_size: int
    drop_last: bool

    def __init__(self, sampler: S, batch_size: int, drop_last: bool):
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batch = List[int]()
        for i in self.sampler:
            batch.append(i)
            if len(batch) == self.batch_size:
                yield batch
                batch = List[int]()
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self) -> int:
        n = len(self.sampler)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size
