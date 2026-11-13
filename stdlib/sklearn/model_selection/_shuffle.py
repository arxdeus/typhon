# Copyright (C) 2026 the Typhon authors
# Ported from scikit-learn 1.9.1 (BSD-3-Clause)

# ShuffleSplit / StratifiedShuffleSplit / GroupShuffleSplit and the repeated
# k-fold splitters. One numpy.random.RandomState(int) stream per split() call,
# exactly as sklearn's check_random_state(int).

import numpy as np
import math
from ._split import _approximate_mode, StratifiedKFold

def _rs(random_state: Optional[int], who: str):
    from ..utils import _LegacyRS
    if random_state is None:
        raise ValueError(f"{who} needs an integer random_state in Typhon")
    return _LegacyRS(random_state)

def _validate_shuffle_split(n: int, test_size, train_size, default_test: float):
    ts_f = -1.0; ts_i = -1; tr_f = -1.0; tr_i = -1
    if test_size is None and train_size is None:
        ts_f = default_test
    elif test_size is not None:
        if isinstance(test_size, float):
            ts_f = test_size
        else:
            ts_i = int(test_size)
    if train_size is not None:
        if isinstance(train_size, float):
            tr_f = train_size
        else:
            tr_i = int(train_size)
    if (ts_i >= 0 and (ts_i >= n or ts_i <= 0)) or (ts_f >= 0.0 and (ts_f <= 0.0 or ts_f >= 1.0)):
        raise ValueError(f"test_size={test_size} should be either positive and smaller than the number of samples {n} or a float in the (0, 1) range")
    if (tr_i >= 0 and (tr_i >= n or tr_i <= 0)) or (tr_f >= 0.0 and (tr_f <= 0.0 or tr_f >= 1.0)):
        raise ValueError(f"train_size={train_size} should be either positive and smaller than the number of samples {n} or a float in the (0, 1) range")
    if ts_f >= 0.0 and tr_f >= 0.0 and ts_f + tr_f > 1.0:
        raise ValueError(f"The sum of test_size and train_size = {ts_f + tr_f}, should be in the (0, 1) range. Reduce test_size and/or train_size.")
    have_te = ts_f >= 0.0 or ts_i >= 0
    have_tr = tr_f >= 0.0 or tr_i >= 0
    n_test = 0; n_train = 0
    if ts_f >= 0.0:
        n_test = int(math.ceil(ts_f * n))
    elif ts_i >= 0:
        n_test = ts_i
    if tr_f >= 0.0:
        n_train = int(math.floor(tr_f * n))
    elif tr_i >= 0:
        n_train = tr_i
    if not have_tr:
        n_train = n - n_test
    elif not have_te:
        n_test = n - n_train
    if n_train + n_test > n:
        raise ValueError(f"The sum of train_size and test_size = {n_train + n_test}, should be smaller than the number of samples {n}. Reduce test_size and/or train_size.")
    if n_train == 0:
        raise ValueError(f"With n_samples={n}, test_size={test_size} and train_size={train_size}, the resulting train set will be empty. Adjust any of the aforementioned parameters.")
    return n_train, n_test

class ShuffleSplit[TS = NoneType, TR = NoneType]:
    n_splits: int
    test_size: TS
    train_size: TR
    random_state: Optional[int]
    _default_test_size: float

    def __init__(self, n_splits: int = 10, *, test_size: TS = None, train_size: TR = None,
                 random_state: Optional[int] = None):
        self.n_splits = n_splits
        self.test_size = test_size
        self.train_size = train_size
        self.random_state = random_state
        self._default_test_size = 0.1

    def get_n_splits(self, X = None, y = None, groups = None) -> int:
        return self.n_splits

    def _iter_n(self, n: int):
        ntr, nte = _validate_shuffle_split(n, self.test_size, self.train_size, self._default_test_size)
        rng = _rs(self.random_state, "ShuffleSplit")
        for _ in range(self.n_splits):
            perm = rng.permutation(n)
            yield perm[nte:nte + ntr].copy(), perm[:nte].copy()

    def split(self, X, y = None, groups = None):
        return self._iter_n(np.asarray(X).shape[0])

class GroupShuffleSplit[TS = NoneType, TR = NoneType]:
    n_splits: int
    test_size: TS
    train_size: TR
    random_state: Optional[int]

    def __init__(self, n_splits: int = 5, *, test_size: TS = None, train_size: TR = None,
                 random_state: Optional[int] = None):
        self.n_splits = n_splits
        self.test_size = test_size
        self.train_size = train_size
        self.random_state = random_state

    def get_n_splits(self, X = None, y = None, groups = None) -> int:
        return self.n_splits

    def split(self, X, y = None, groups = None):
        if groups is None:
            raise ValueError("The 'groups' parameter should not be None.")
        u, gi = np.unique(np.asarray(groups), return_inverse=True)
        ng = u.shape[0]
        ntr, nte = _validate_shuffle_split(ng, self.test_size, self.train_size, 0.2)
        rng = _rs(self.random_state, "GroupShuffleSplit")
        n = gi.shape[0]
        for _ in range(self.n_splits):
            perm = rng.permutation(ng)
            role = np.zeros(ng, dtype=int)
            for k in range(nte):
                role[perm[k]] = 2
            for k in range(nte, nte + ntr):
                role[perm[k]] = 1
            tr = List[int](); te = List[int]()
            for i in range(n):
                r = role[gi[i]]
                if r == 1:
                    tr.append(i)
                elif r == 2:
                    te.append(i)
            yield np.array(tr, dtype=int), np.array(te, dtype=int)

class StratifiedShuffleSplit[TS = NoneType, TR = NoneType]:
    n_splits: int
    test_size: TS
    train_size: TR
    random_state: Optional[int]

    def __init__(self, n_splits: int = 10, *, test_size: TS = None, train_size: TR = None,
                 random_state: Optional[int] = None):
        self.n_splits = n_splits
        self.test_size = test_size
        self.train_size = train_size
        self.random_state = random_state

    def get_n_splits(self, X = None, y = None, groups = None) -> int:
        return self.n_splits

    def split(self, X, y, groups = None):
        ya = np.asarray(y)
        n = ya.shape[0]
        ntr, nte = _validate_shuffle_split(n, self.test_size, self.train_size, 0.1)
        classes, y_ind, counts = np.unique(ya, return_inverse=True, return_counts=True)
        nc = classes.shape[0]
        if int(np.min(counts)) < 2:
            raise ValueError("The least populated class in y has only 1 member, which is too few. The minimum number of groups for any class cannot be less than 2.")
        if ntr < nc:
            raise ValueError(f"The train_size = {ntr} should be greater or equal to the number of classes = {nc}")
        if nte < nc:
            raise ValueError(f"The test_size = {nte} should be greater or equal to the number of classes = {nc}")
        order = np.argsort(y_ind, kind="stable")
        rng = _rs(self.random_state, "StratifiedShuffleSplit")
        for _ in range(self.n_splits):
            n_i = _approximate_mode(counts, ntr, rng)
            t_i = _approximate_mode(counts - n_i, nte, rng)
            train = List[int](); test = List[int]()
            start = 0
            for i in range(nc):
                ci = int(counts[i])
                pm = rng.permutation(ci)
                for k in range(int(n_i[i])):
                    train.append(int(order[start + pm[k]]))
                for k in range(int(n_i[i]), int(n_i[i]) + int(t_i[i])):
                    test.append(int(order[start + pm[k]]))
                start += ci
            yield rng.permutation_of(np.array(train, dtype=int)), rng.permutation_of(np.array(test, dtype=int))

def _kfold_tests(n: int, n_splits: int, indices):
    out = List[np.ndarray[int, 1]]()
    current = 0
    for k in range(n_splits):
        size = n // n_splits + (1 if k < n % n_splits else 0)
        out.append(indices[current:current + size].copy())
        current += size
    return out

def _mask_pair(n: int, te):
    m = np.zeros(n, dtype=bool)
    m[te] = True
    return np.nonzero(~m)[0], np.nonzero(m)[0]

class RepeatedKFold:
    n_splits: int
    n_repeats: int
    random_state: Optional[int]

    def __init__(self, *, n_splits: int = 5, n_repeats: int = 10, random_state: Optional[int] = None):
        if n_repeats <= 0:
            raise ValueError("Number of repetitions must be greater than 0.")
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.random_state = random_state

    def get_n_splits(self, X = None, y = None, groups = None) -> int:
        return self.n_splits * self.n_repeats

    def split(self, X, y = None, groups = None):
        n = np.asarray(X).shape[0]
        if self.n_splits > n:
            raise ValueError(f"Cannot have number of splits n_splits={self.n_splits} greater than the number of samples: n_samples={n}.")
        rng = _rs(self.random_state, "RepeatedKFold")
        for _ in range(self.n_repeats):
            idx = rng.permutation(n)
            for te in _kfold_tests(n, self.n_splits, idx):
                yield _mask_pair(n, te)

class RepeatedStratifiedKFold:
    n_splits: int
    n_repeats: int
    random_state: Optional[int]

    def __init__(self, *, n_splits: int = 5, n_repeats: int = 10, random_state: Optional[int] = None):
        if n_repeats <= 0:
            raise ValueError("Number of repetitions must be greater than 0.")
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.random_state = random_state

    def get_n_splits(self, X = None, y = None, groups = None) -> int:
        return self.n_splits * self.n_repeats

    def split(self, X, y, groups = None):
        rng = _rs(self.random_state, "RepeatedStratifiedKFold")
        skf = StratifiedKFold(self.n_splits, shuffle=True, random_state=0)
        for _ in range(self.n_repeats):
            tf = skf._test_folds_rng(y, rng)
            for f in range(self.n_splits):
                yield np.nonzero(tf != f)[0], np.nonzero(tf == f)[0]
