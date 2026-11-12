# Ported from torch 2.15.0 (BSD-3-Clause)
#
# Single-process DataLoader. RNG use matches torch exactly: each epoch
# (each iter()) first draws the iterator base seed from the loader generator
# (or the global generator), then the RandomSampler draws its own seed from
# the global generator if it has none, then randperm.

from torch._tensor import Tensor
from torch.nn._util import _or
from torch._random import Generator
from .sampler import _rand_int64_seed, RandomSampler, _perm_list

def _stack(items: List[Tensor]) -> Tensor:
    from torch import stack
    return stack(items, 0)

def _collate_list(items: List[T], T: type):
    if T is Tensor:
        return _stack(items)
    elif T is int:
        from torch import tensor, int64
        return tensor(items, dtype=int64)
    elif T is float:
        from torch import tensor, float64
        return tensor(items, dtype=float64)
    elif T is bool:
        from torch import tensor
        return tensor(items)
    elif T is str:
        return items
    elif isinstance(T, Tuple):
        return _collate_tuple(items)
    else:
        compile_error("default_collate: unsupported element type")

def _collate_tuple(items):
    first = items[0]
    if static.len(first) == 1:
        return (_collate_list([x[0] for x in items]),)
    elif static.len(first) == 2:
        return (_collate_list([x[0] for x in items]), _collate_list([x[1] for x in items]))
    elif static.len(first) == 3:
        return (_collate_list([x[0] for x in items]), _collate_list([x[1] for x in items]), _collate_list([x[2] for x in items]))
    elif static.len(first) == 4:
        return (_collate_list([x[0] for x in items]), _collate_list([x[1] for x in items]), _collate_list([x[2] for x in items]),
                _collate_list([x[3] for x in items]))
    else:
        compile_error("default_collate: tuples of 1 to 4 elements supported")

def default_collate(batch):
    return _collate_list(batch)

class DataLoader[D]:
    dataset: D
    batch_size: Optional[int]
    shuffle: bool
    drop_last: bool
    generator: Optional[Generator]
    num_workers: int

    def __init__(self, dataset: D, batch_size: Optional[int] = 1, shuffle: Optional[bool] = None, sampler = None,
                 batch_sampler = None, num_workers: int = 0, collate_fn = None, pin_memory: bool = False,
                 drop_last: bool = False, timeout: float = 0, worker_init_fn = None, multiprocessing_context = None,
                 generator: Optional[Generator] = None, prefetch_factor: Optional[int] = None,
                 persistent_workers: bool = False, pin_memory_device: str = "", in_order: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = _or(shuffle, False)
        self.drop_last = drop_last
        self.generator = generator
        self.num_workers = num_workers

    def _indices(self) -> List[int]:
        n = len(self.dataset)
        # iterator base seed (consumes one int64 draw like torch)
        _rand_int64_seed(self.generator)
        if self.shuffle:
            g: Generator
            if self.generator is None:
                seed = _rand_int64_seed(None)
                g = Generator()
                g.manual_seed(seed)
            else:
                g = self.generator
            if n == 0:
                return List[int]()
            perm = _perm_list(n, g)
            # RandomSampler.__iter__ also draws randperm(n)[:num_samples % n]
            # for the (empty) tail, advancing the generator once more
            _perm_list(n, g)
            return perm
        return list(range(n))

    def __iter__(self):
        idx = self._indices()
        if self.batch_size is None:
            for i in idx:
                yield self.dataset[i]
        else:
            bs = self.batch_size
            n = len(idx)
            s = 0
            while s < n:
                e = min(s + bs, n)
                if e - s < bs and self.drop_last:
                    break
                items = [self.dataset[idx[j]] for j in range(s, e)]
                yield default_collate(items)
                s = e

    def __len__(self) -> int:
        n = len(self.dataset)
        if self.batch_size is None:
            return n
        bs = self.batch_size
        if self.drop_last:
            return n // bs
        return (n + bs - 1) // bs
