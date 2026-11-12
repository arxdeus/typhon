# Ported from scikit-learn 1.9.1 (BSD-3-Clause)

import numpy as np
from ..base import clone, _accuracy

def _n_samples(X) -> int:
    return np.asarray(X).shape[0]

def _legacy_permutation(n: int, seed: int):
    # numpy RandomState(seed).shuffle(arange(n)): MT19937 legacy seeding +
    # Fisher-Yates with masked rejection sampling (numpy's legacy bounded int).
    from ..utils import _LegacyMT
    g = _LegacyMT(seed)
    a = np.arange(n)
    i = n - 1
    while i > 0:
        mask = u32(i)
        mask |= mask >> u32(1); mask |= mask >> u32(2); mask |= mask >> u32(4)
        mask |= mask >> u32(8); mask |= mask >> u32(16)
        j = mask & u32(g.genrand_int32())
        while j > u32(i):
            j = mask & u32(g.genrand_int32())
        jj = int(j)
        t = a[i]; a[i] = a[jj]; a[jj] = t
        i -= 1
    return a

class KFold:
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

    def _test_folds(self, n: int) -> List[np.ndarray[int, 1]]:
        if self.n_splits > n:
            raise ValueError(f"Cannot have number of splits n_splits={self.n_splits} greater than the number of samples: n_samples={n}.")
        indices = np.arange(n)
        if self.shuffle:
            if self.random_state is None:
                raise ValueError("KFold(shuffle=True) needs an integer random_state in Typhon")
            indices = _legacy_permutation(n, self.random_state)
        out = List[np.ndarray[int, 1]]()
        current = 0
        for k in range(self.n_splits):
            size = n // self.n_splits + (1 if k < n % self.n_splits else 0)
            out.append(indices[current:current + size].copy())
            current += size
        return out

    def split(self, X, y = None, groups = None):
        n = _n_samples(X)
        for te in self._test_folds(n):
            mask = np.ones(n, dtype=bool)
            mask[te] = False
            # sklearn goes through boolean masks, so both index sets are sorted
            tr = np.nonzero(mask)[0]
            tem = np.zeros(n, dtype=bool)
            tem[te] = True
            yield (tr, np.nonzero(tem)[0])

    def __repr__(self) -> str:
        rs = "None" if self.random_state is None else str(self.random_state)
        return f"KFold(n_splits={self.n_splits}, random_state={rs}, shuffle={self.shuffle})"

class StratifiedKFold:
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

    def _make_test_folds(self, y):
        from ..utils import _LegacyRS
        ya = np.asarray(y)
        n = ya.shape[0]
        _, y_idx, y_inv = np.unique(ya, return_index=True, return_inverse=True)
        _, class_perm = np.unique(y_idx, return_inverse=True)
        y_enc = class_perm[y_inv]
        nc = y_idx.shape[0]
        counts = np.bincount(y_enc)
        if np.all(self.n_splits > counts):
            raise ValueError(f"n_splits={self.n_splits} cannot be greater than the number of members in each class.")
        y_order = np.sort(y_enc)
        alloc = np.zeros((self.n_splits, nc), dtype=int)
        for i in range(self.n_splits):
            j = i
            while j < n:
                alloc[i, y_order[j]] += 1
                j += self.n_splits
        rng = _LegacyRS(0 if self.random_state is None else self.random_state)
        if self.shuffle and self.random_state is None:
            raise ValueError("StratifiedKFold(shuffle=True) needs an integer random_state in Typhon")
        test_folds = np.empty(n, dtype=int)
        for k in range(nc):
            tot = int(np.sum(alloc[:, k]))
            ff = np.empty(tot, dtype=int)
            q = 0
            for f in range(self.n_splits):
                for _ in range(alloc[f, k]):
                    ff[q] = f
                    q += 1
            if self.shuffle:
                rng.shuffle(ff)
            q = 0
            for i in range(n):
                if y_enc[i] == k:
                    test_folds[i] = ff[q]
                    q += 1
        return test_folds

    def split(self, X, y, groups = None):
        tf = self._make_test_folds(y)
        n = tf.shape[0]
        for f in range(self.n_splits):
            yield (np.nonzero(tf != f)[0], np.nonzero(tf == f)[0])

def check_cv(cv = 5, y = None, *, classifier: bool = True):
    # sklearn: an int cv means StratifiedKFold for classifiers (all ported
    # estimators are classifiers)
    if isinstance(cv, int):
        return StratifiedKFold(cv)
    else:
        return cv

def _n_split(test_size, train_size, n: int):
    import math
    nt = 0
    ntr = 0
    if test_size is None and train_size is None:
        nt = int(math.ceil(0.25 * n))
        ntr = n - nt
    elif test_size is not None:
        if isinstance(test_size, float):
            nt = int(math.ceil(test_size * n))
        else:
            nt = int(test_size)
        if train_size is None:
            ntr = n - nt
        elif isinstance(train_size, float):
            ntr = int(math.floor(train_size * n))
        else:
            ntr = int(train_size)
    else:
        if isinstance(train_size, float):
            ntr = int(math.floor(train_size * n))
        else:
            ntr = int(train_size)
        nt = n - ntr
    return ntr, nt

def _approximate_mode(counts, n_draws: int, rng):
    tot = float(np.sum(counts))
    cont = counts.astype(float) / tot * float(n_draws)
    floored = np.floor(cont)
    need = int(n_draws - floored.sum())
    if need > 0:
        rem = cont - floored
        vals = np.sort(np.unique(rem))[::-1]
        for v in vals:
            inds = np.nonzero(rem == v)[0]
            add_now = min(inds.shape[0], need)
            ch = rng.choice_noreplace(inds, add_now)
            for i in ch:
                floored[i] += 1
            need -= add_now
            if need == 0:
                break
    return floored.astype(int)

def _tts_indices(n: int, test_size, train_size, random_state: Optional[int], shuffle: bool, stratify):
    from ..utils import _LegacyRS
    ntr, nt = _n_split(test_size, train_size, n)
    if not shuffle:
        return np.arange(ntr), np.arange(ntr, ntr + nt)
    if random_state is None:
        raise ValueError("train_test_split(shuffle=True) needs an integer random_state in Typhon")
    rng = _LegacyRS(random_state)
    if stratify is None:
        perm = rng.permutation(n)
        return perm[nt:nt + ntr].copy(), perm[:nt].copy()
    ys = np.asarray(stratify)
    classes, y_ind, counts = np.unique(ys, return_inverse=True, return_counts=True)
    nc = classes.shape[0]
    order = np.argsort(y_ind, kind="stable")
    n_i = _approximate_mode(counts, ntr, rng)
    t_i = _approximate_mode(counts - n_i, nt, rng)
    train = List[int]()
    test = List[int]()
    start = 0
    for i in range(nc):
        ci = int(counts[i])
        cls = order[start:start + ci]
        pm = rng.permutation(ci)
        pic = cls[pm]
        for k in range(int(n_i[i])):
            train.append(int(pic[k]))
        for k in range(int(n_i[i]), int(n_i[i]) + int(t_i[i])):
            test.append(int(pic[k]))
        start += ci
    return rng.permutation_of(np.array(train)), rng.permutation_of(np.array(test))

def train_test_split(*arrays, test_size = None, train_size = None,
                     random_state: Optional[int] = None, shuffle: bool = True, stratify = None):
    """Returns (a_train, a_test, b_train, b_test, ...) as a tuple."""
    n = np.asarray(arrays[0]).shape[0]
    tr, te = _tts_indices(n, test_size, train_size, random_state, shuffle, stratify)
    return _tts_take(arrays, tr, te)

def _tts_take(arrays, tr, te):
    if static.len(arrays) == 0:
        return ()
    else:
        a = np.asarray(arrays[0])
        return (a[tr], a[te]) + _tts_take(arrays[1:], tr, te)

def _take(X, idx):
    return np.asarray(X)[idx]

def _cvp_impl(estimator, X, y, cv):
    cvo = check_cv(5 if cv is None else cv)
    Xa = np.asarray(X)
    ya = np.asarray(y)
    n = Xa.shape[0]
    out = np.zeros(n, dtype=ya.dtype)
    for tr, te in cvo.split(Xa, ya):
        est = clone(estimator)
        est.fit(Xa[tr], ya[tr])
        out[te] = np.asarray(est.predict(Xa[te]))
    return out

def _cvs_impl(estimator, X, y, cv):
    cvo = check_cv(5 if cv is None else cv)
    Xa = np.asarray(X)
    ya = np.asarray(y)
    scores = List[float]()
    for tr, te in cvo.split(Xa, ya):
        est = clone(estimator)
        est.fit(Xa[tr], ya[tr])
        scores.append(_accuracy(ya[te], est.predict(Xa[te])))
    return np.array(scores)

def cross_val_predict(estimator, X, y = None, *, groups = None, cv = None,
                      n_jobs = None, verbose: int = 0, params = None,
                      pre_dispatch: str = "2*n_jobs", method: str = "predict"):
    """Out-of-fold predictions. Folds run sequentially (n_jobs ignored).
    `estimator` may be a Union (e.g. a value of a dict of different pipelines)."""
    if isinstance(estimator, Union):
        for tag, T in static.vars_types(estimator, with_index=True):
            if Union._get_tag(estimator) == tag:
                return _cvp_impl(Union._get_data(estimator, T), X, y, cv)
        raise TypeError("invalid union")
    else:
        return _cvp_impl(estimator, X, y, cv)

def cross_val_score(estimator, X, y = None, *, groups = None, scoring = None,
                    cv = None, n_jobs = None, verbose: int = 0, params = None,
                    pre_dispatch: str = "2*n_jobs", error_score = None):
    """Per-fold accuracy (classifier default scorer). Sequential."""
    if isinstance(estimator, Union):
        for tag, T in static.vars_types(estimator, with_index=True):
            if Union._get_tag(estimator) == tag:
                return _cvs_impl(Union._get_data(estimator, T), X, y, cv)
        raise TypeError("invalid union")
    else:
        return _cvs_impl(estimator, X, y, cv)

# ------------------------------------------------------------ grid search

def _set_param_rt(est, key: str, val: float) -> bool:
    """Set parameter `key` (runtime string, sklearn `step__param` nesting) to a
    numeric value. Returns False when no field matches."""
    if hasattr(est, "_sk_clone") and hasattr(est, "steps"):
        i = key.find("__")
        if i < 0:
            return False
        head = key[:i]
        rest = key[i + 2:]
        ok = False
        for st in est.steps:
            if st[0] == head:
                ok = _set_param_rt(st[1], rest, val)
        return ok
    else:
        ok = False
        for k, v in static.vars(est):
            if k == key:
                if isinstance(v, float):
                    setattr(est, k, float(val)); ok = True
                elif isinstance(v, int):
                    setattr(est, k, int(val)); ok = True
                elif isinstance(v, Optional[float]):
                    setattr(est, k, Optional[float](float(val))); ok = True
                elif isinstance(v, Optional[int]):
                    setattr(est, k, Optional[int](int(val))); ok = True
        return ok

def _param_grid(grid: Dict[str, List[float]]) -> List[List[Tuple[str, float]]]:
    keys = sorted(grid.keys())
    out = [List[Tuple[str, float]]()]
    for k in keys:
        nxt = List[List[Tuple[str, float]]]()
        for base in out:
            for v in grid[k]:
                c = base.copy()
                c.append((k, v))
                nxt.append(c)
        out = nxt
    return out

class ParameterGrid:
    grid: Dict[str, List[float]]
    def __init__(self, param_grid: Dict[str, List[float]]):
        self.grid = param_grid
    def __iter__(self):
        for c in _param_grid(self.grid):
            yield {k: v for k, v in c}
    def __len__(self) -> int:
        return len(_param_grid(self.grid))

class GridSearchCV[E, CV]:
    """Exhaustive search over a numeric parameter grid (dict name -> list of
    floats; ints are cast), accuracy scoring, refit on the full data."""
    estimator: E
    param_grid: Dict[str, List[float]]
    cv: CV
    refit: bool
    cv_results_: Dict[str, np.ndarray[float, 1]]
    params_: List[Dict[str, float]]
    best_index_: int
    best_score_: float
    best_params_: Dict[str, float]
    best_estimator_: E
    n_splits_: int

    def __init__(self, estimator: E, param_grid: Dict[str, List[float]], *, scoring = None,
                 n_jobs = None, refit: bool = True, cv: CV = 5, verbose: int = 0,
                 pre_dispatch = "2*n_jobs", error_score = None, return_train_score: bool = False):
        self.estimator = estimator
        self.param_grid = param_grid
        self.cv = cv
        self.refit = refit
        self.cv_results_ = {}
        self.params_ = []
        self.best_index_ = -1
        self.best_score_ = 0.0
        self.best_params_ = {}
        self.best_estimator_ = clone(estimator)
        self.n_splits_ = 0

    def _sk_clone(self):
        return GridSearchCV(clone(self.estimator), self.param_grid, refit=self.refit, cv=self.cv)

    def fit(self, X, y):
        cvo = check_cv(self.cv)
        Xa = np.asarray(X)
        ya = np.asarray(y)
        folds = [(tr, te) for tr, te in cvo.split(Xa, ya)]
        combos = _param_grid(self.param_grid)
        nk = len(combos)
        nf = len(folds)
        self.n_splits_ = nf
        means = np.empty(nk)
        stds = np.empty(nk)
        split_scores = [np.empty(nk) for _ in range(nf)]
        self.params_ = []
        for ci in range(nk):
            sc = np.empty(nf)
            for fi in range(nf):
                tr, te = folds[fi]
                est = clone(self.estimator)
                for kv in combos[ci]:
                    if not _set_param_rt(est, kv[0], kv[1]):
                        raise ValueError(f"Invalid parameter {kv[0]!r} for estimator")
                est.fit(Xa[tr], ya[tr])
                sc[fi] = _accuracy(ya[te], est.predict(Xa[te]))
                split_scores[fi][ci] = sc[fi]
            means[ci] = np.mean(sc)
            stds[ci] = np.std(sc)
            self.params_.append({kv[0]: kv[1] for kv in combos[ci]})
        # rank_test_score: rankdata(-mean, method="min"); best = first rank 1
        best = 0
        for ci in range(1, nk):
            if means[ci] > means[best]:
                best = ci
        rank = np.empty(nk)
        for ci in range(nk):
            r = 1
            for cj in range(nk):
                if means[cj] > means[ci]:
                    r += 1
            rank[ci] = float(r)
        res = Dict[str, np.ndarray[float, 1]]()
        res["mean_test_score"] = means
        res["std_test_score"] = stds
        res["rank_test_score"] = rank
        for fi in range(nf):
            res[f"split{fi}_test_score"] = split_scores[fi]
        self.cv_results_ = res
        self.best_index_ = best
        self.best_score_ = means[best]
        self.best_params_ = self.params_[best]
        if self.refit:
            est = clone(self.estimator)
            for kv in combos[best]:
                _set_param_rt(est, kv[0], kv[1])
            est.fit(Xa, ya)
            self.best_estimator_ = est
        return self

    def predict(self, X):
        return self.best_estimator_.predict(X)

    def score(self, X, y, sample_weight = None):
        return _accuracy(y, self.predict(X))
