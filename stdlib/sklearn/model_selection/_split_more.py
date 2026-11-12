# Copyright (C) 2026 the Typhon authors
# Ported from scikit-learn 1.9.1 (BSD-3-Clause)

# Group / time-series / leave-out / shuffle cross-validators. Shuffling uses
# numpy.random.RandomState(int) streams (random_state must be an int when
# shuffling; sklearn's None = global state is not reproducible anyway).

import numpy as np
import math

def _n_samples(X) -> int:
    return np.asarray(X).shape[0]

def _mask_split(n: int, test):
    m = np.zeros(n, dtype=bool)
    m[test] = True
    return np.nonzero(~m)[0], np.nonzero(m)[0]

def _rs(random_state: Optional[int], who: str):
    from ..utils import _LegacyRS
    if random_state is None:
        raise ValueError(f"{who} needs an integer random_state in Typhon")
    return _LegacyRS(random_state)

def _group_codes(groups):
    g = np.asarray(groups)
    u, inv = np.unique(g, return_inverse=True)
    return u, inv

class GroupKFold:
    n_splits: int
    shuffle: bool
    random_state: Optional[int]

    def __init__(self, n_splits: int = 5, *, shuffle: bool = False, random_state: Optional[int] = None):
        if n_splits <= 1:
            raise ValueError("k-fold cross-validation requires at least one train/test split by setting n_splits=2 or more")
        if not shuffle and random_state is not None:
            raise ValueError("Setting a random_state has no effect since shuffle is False. You should leave random_state to its default (None), or set shuffle=True.")
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def get_n_splits(self, X = None, y = None, groups = None) -> int:
        return self.n_splits

    def split(self, X, y = None, groups = None):
        if groups is None:
            raise ValueError("The 'groups' parameter should not be None.")
        n = _n_samples(X)
        if self.n_splits > n:
            raise ValueError(f"Cannot have number of splits n_splits={self.n_splits} greater than the number of samples: n_samples={n}.")
        u, gidx = _group_codes(groups)
        ng = u.shape[0]
        if self.n_splits > ng:
            raise ValueError(f"Cannot have number of splits n_splits={self.n_splits} greater than the number of groups: {ng}.")
        fold_of = np.zeros(ng, dtype=int)
        if self.shuffle:
            perm = _rs(self.random_state, "GroupKFold(shuffle=True)").permutation(ng)
            # np.array_split(permuted groups, n_splits)
            q = 0
            for f in range(self.n_splits):
                size = ng // self.n_splits + (1 if f < ng % self.n_splits else 0)
                for _ in range(size):
                    fold_of[perm[q]] = f
                    q += 1
        else:
            cnt = np.bincount(gidx)
            order = np.argsort(cnt, kind="stable")[::-1]
            per_fold = np.zeros(self.n_splits)
            for gi in range(ng):
                g = order[gi]
                lf = int(np.argmin(per_fold))
                per_fold[lf] += float(cnt[g])
                fold_of[g] = lf
        for f in range(self.n_splits):
            te = np.nonzero(fold_of[gidx] == f)[0]
            yield _mask_split(n, te)

class StratifiedGroupKFold:
    n_splits: int
    shuffle: bool
    random_state: Optional[int]

    def __init__(self, n_splits: int = 5, shuffle: bool = False, random_state: Optional[int] = None):
        if n_splits <= 1:
            raise ValueError("k-fold cross-validation requires at least one train/test split by setting n_splits=2 or more")
        if not shuffle and random_state is not None:
            raise ValueError("Setting a random_state has no effect since shuffle is False. You should leave random_state to its default (None), or set shuffle=True.")
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def get_n_splits(self, X = None, y = None, groups = None) -> int:
        return self.n_splits

    def _best_fold(self, ypf, y_cnt, gyc) -> int:
        best = -1
        min_eval = math.inf
        min_samples = math.inf
        for i in range(self.n_splits):
            ypf[i] += gyc
            spc = np.std(ypf / y_cnt.reshape(1, -1), axis=0)
            ypf[i] -= gyc
            fe = float(np.mean(spc))
            sif = float(np.sum(ypf[i]))
            better = fe < min_eval or (abs(fe - min_eval) <= 1e-8 + 1e-5 * abs(min_eval) and sif < min_samples)
            if better:
                min_eval = fe
                min_samples = sif
                best = i
        return best

    def split(self, X, y, groups = None):
        if groups is None:
            raise ValueError("The 'groups' parameter should not be None.")
        n = _n_samples(X)
        ya = np.asarray(y)
        _, y_inv, y_cnt = np.unique(ya, return_inverse=True, return_counts=True)
        if np.all(self.n_splits > y_cnt):
            raise ValueError(f"n_splits={self.n_splits} cannot be greater than the number of members in each class.")
        nc = y_cnt.shape[0]
        _, g_inv, g_cnt = np.unique(np.asarray(groups), return_inverse=True, return_counts=True)
        ng = g_cnt.shape[0]
        if self.n_splits > ng:
            raise ValueError(f"Cannot have number of splits n_splits={self.n_splits} greater than the number of groups: {ng}.")
        ypg = np.zeros((ng, nc))
        for i in range(n):
            ypg[g_inv[i], y_inv[i]] += 1.0
        ypf = np.zeros((self.n_splits, nc))
        if self.shuffle:
            perm = np.arange(ng)
            _rs(self.random_state, "StratifiedGroupKFold(shuffle=True)").shuffle(perm)
            ypg = ypg[perm]
            inv = np.empty(ng, dtype=int)
            inv[perm] = np.arange(ng)
            g_inv = inv[g_inv]
        order = np.argsort(-np.std(ypg, axis=1), kind="stable")
        fold_of = np.zeros(ng, dtype=int)
        ycf = y_cnt.astype(float)
        for gi in order:
            gyc = ypg[gi].copy()
            bf = self._best_fold(ypf, ycf, gyc)
            ypf[bf] += gyc
            fold_of[gi] = bf
        for f in range(self.n_splits):
            te = np.nonzero(fold_of[g_inv] == f)[0]
            yield _mask_split(n, te)

class TimeSeriesSplit:
    n_splits: int
    max_train_size: Optional[int]
    test_size: Optional[int]
    gap: int

    def __init__(self, n_splits: int = 5, *, max_train_size: Optional[int] = None,
                 test_size: Optional[int] = None, gap: int = 0):
        if n_splits <= 1:
            raise ValueError("k-fold cross-validation requires at least one train/test split by setting n_splits=2 or more")
        self.n_splits = n_splits
        self.max_train_size = max_train_size
        self.test_size = test_size
        self.gap = gap

    def get_n_splits(self, X = None, y = None, groups = None) -> int:
        return self.n_splits

    def split(self, X, y = None, groups = None):
        n = _n_samples(X)
        nf = self.n_splits + 1
        ts = n // nf if self.test_size is None else self.test_size
        if nf > n:
            raise ValueError(f"Cannot have number of folds={nf} greater than the number of samples={n}.")
        if n - self.gap - ts * self.n_splits <= 0:
            raise ValueError(f"Too many splits={self.n_splits} for number of samples={n} with test_size={ts} and gap={self.gap}.")
        idx = np.arange(n)
        for start in range(n - self.n_splits * ts, n, ts):
            te_ = start - self.gap
            if self.max_train_size is not None and self.max_train_size > 0 and self.max_train_size < te_:
                yield idx[te_ - self.max_train_size:te_].copy(), idx[start:start + ts].copy()
            else:
                yield idx[:te_].copy(), idx[start:start + ts].copy()

class LeaveOneOut:
    def __init__(self):
        pass

    def get_n_splits(self, X = None, y = None, groups = None) -> int:
        return _n_samples(X)

    def split(self, X, y = None, groups = None):
        n = _n_samples(X)
        if n <= 1:
            raise ValueError(f"Cannot perform LeaveOneOut with n_samples={n}.")
        for i in range(n):
            yield _mask_split(n, np.array([i]))

def _combinations(n: int, k: int):
    from ..preprocessing._polynomial import _combos
    return _combos(n, k, True)

class LeavePOut:
    p: int

    def __init__(self, p: int):
        self.p = p

    def get_n_splits(self, X = None, y = None, groups = None) -> int:
        return math.comb(_n_samples(X), self.p)

    def split(self, X, y = None, groups = None):
        n = _n_samples(X)
        if n <= self.p:
            raise ValueError(f"p={self.p} must be strictly less than the number of samples={n}")
        for c in _combinations(n, self.p):
            yield _mask_split(n, np.array(c))

class LeaveOneGroupOut:
    def __init__(self):
        pass

    def get_n_splits(self, X = None, y = None, groups = None) -> int:
        if groups is None:
            raise ValueError("The 'groups' parameter should not be None.")
        return np.unique(np.asarray(groups)).shape[0]

    def split(self, X, y = None, groups = None):
        if groups is None:
            raise ValueError("The 'groups' parameter should not be None.")
        u, inv = _group_codes(groups)
        if u.shape[0] <= 1:
            raise ValueError(f"The groups parameter contains fewer than 2 unique groups ({u}). LeaveOneGroupOut expects at least 2.")
        n = inv.shape[0]
        for g in range(u.shape[0]):
            yield _mask_split(n, np.nonzero(inv == g)[0])

class LeavePGroupsOut:
    n_groups: int

    def __init__(self, n_groups: int):
        self.n_groups = n_groups

    def get_n_splits(self, X = None, y = None, groups = None) -> int:
        if groups is None:
            raise ValueError("The 'groups' parameter should not be None.")
        return math.comb(np.unique(np.asarray(groups)).shape[0], self.n_groups)

    def split(self, X, y = None, groups = None):
        if groups is None:
            raise ValueError("The 'groups' parameter should not be None.")
        u, inv = _group_codes(groups)
        if self.n_groups >= u.shape[0]:
            raise ValueError(f"The groups parameter contains fewer than (or equal to) n_groups ({self.n_groups}) numbers of unique groups ({u}). LeavePGroupsOut expects that at least n_groups + 1 ({self.n_groups + 1}) unique groups be present")
        n = _n_samples(X)
        for c in _combinations(u.shape[0], self.n_groups):
            m = np.zeros(n, dtype=bool)
            for g in c:
                m |= inv == g
            yield np.nonzero(~m)[0], np.nonzero(m)[0]

class PredefinedSplit:
    test_fold: np.ndarray[int, 1]
    unique_folds: np.ndarray[int, 1]

    def __init__(self, test_fold):
        self.test_fold = np.array(test_fold, dtype=int).ravel()
        u = np.unique(self.test_fold)
        self.unique_folds = u[u != -1]

    def get_n_splits(self, X = None, y = None, groups = None) -> int:
        return self.unique_folds.shape[0]

    def split(self, X = None, y = None, groups = None):
        n = self.test_fold.shape[0]
        for f in self.unique_folds:
            yield _mask_split(n, np.nonzero(self.test_fold == f)[0])
