# Copyright (C) 2026 the Typhon authors
# Ported from pyriemann 0.12 (BSD-3-Clause)

# Permutation tests: PermutationDistance (estimator=None) and PermutationModel.
# Labels are integers. The plot method is not provided.

import numpy as np
from numpy.random import RandomState
from sklearn.model_selection import cross_val_score
from .classification import MDM
from .geometry.distance import distance, pairwise_distance
from .geometry.mean import gmean

def _factorial(n: int) -> float:
    r = 1.0
    for i in range(2, n + 1):
        r *= float(i)
    return r

def multiset_perm_number(y) -> float:
    y = np.asarray(y, dtype=int)
    pr = 1.0
    for c in np.unique(y):
        pr *= _factorial(int(np.sum(y == c)))
    return _factorial(y.shape[0]) / pr

def _unique_perms(elements: List[int], acc: List[int], out: List[List[int]]):
    if len(elements) == 0:
        out.append(list(acc))
        return
    seen = Set[int]()
    cand = List[int]()
    for e in elements:
        if e not in seen:
            seen.add(e)
            cand.append(e)
    for first in cand:
        rest = List[int]()
        removed = False
        for e in elements:
            if e == first and not removed:
                removed = True
            else:
                rest.append(e)
        acc.append(first)
        _unique_perms(rest, acc, out)
        acc.pop()

def unique_permutations(elements) -> List[List[int]]:
    el = [int(v) for v in np.asarray(elements, dtype=int)]
    out = List[List[int]]()
    _unique_perms(el, List[int](), out)
    return out

def _shuffle(y, groups, rs: RandomState):
    if groups is None:
        indices = rs.permutation(y.shape[0])
        return y[indices]
    g = np.asarray(groups)
    indices = np.arange(g.shape[0])
    for grp in np.unique(g):
        this_mask = g == grp
        indices[this_mask] = rs.permutation(indices[this_mask])
    return y[indices]

def _perm_test(obj, X, y, groups, n_perms: int, random_state: int):
    y = np.asarray(y, dtype=int)
    Npe = multiset_perm_number(y)
    n_scores = min(n_perms, int(Npe))
    scores = np.zeros(n_scores)
    X = obj._initial_transform(X)
    scores[0] = obj.score(X, y, groups)
    if Npe <= float(n_perms):
        perms = unique_permutations(y)
        ii = 0
        for perm in perms:
            pa = np.array(perm)
            if not bool(np.all(pa == y)):
                scores[ii + 1] = obj.score(X, pa, groups)
                ii += 1
    else:
        rs = RandomState(random_state)
        for ii in range(n_perms - 1):
            perm = _shuffle(y, groups, rs)
            scores[ii + 1] = obj.score(X, perm, groups)
    p_value = float(np.mean(np.asarray(scores[0] <= scores, dtype=float)))
    return p_value, scores

class PermutationModel[M = MDM]:
    n_perms: int
    model: M
    cv: int
    random_state: int
    p_value_: float
    scores_: np.ndarray[float, 1]

    def __init__(self, n_perms: int = 100, model: M = MDM(), cv: int = 3,
                 scoring = None, n_jobs: int = 1, random_state: int = 42):
        self.n_perms = n_perms
        self.model = model
        self.cv = cv
        self.random_state = random_state
        self.p_value_ = 0.0
        self.scores_ = np.empty(0)

    def _initial_transform(self, X):
        return X

    def score(self, X, y, groups = None):
        s = cross_val_score(self.model, X, y, cv=self.cv)
        return float(np.mean(s))

    def test(self, X, y, groups = None, verbose: bool = True):
        p, s = _perm_test(self, X, y, groups, self.n_perms, self.random_state)
        self.p_value_ = p
        self.scores_ = s
        return p, s

class PermutationDistance:
    n_perms: int
    metric: str
    mode: str
    random_state: int
    p_value_: float
    scores_: np.ndarray[float, 1]
    global_mean: np.ndarray[float, 2]
    mdm: MDM

    def __init__(self, n_perms: int = 100, metric: str = "riemann",
                 mode: str = "pairwise", n_jobs: int = 1, random_state: int = 42):
        if mode != "pairwise" and mode != "ttest" and mode != "ftest":
            raise ValueError("mode must be 'pairwise', 'ttest' or 'ftest'")
        self.n_perms = n_perms
        self.metric = metric
        self.mode = mode
        self.random_state = random_state
        self.p_value_ = 0.0
        self.scores_ = np.empty(0)
        self.global_mean = np.empty((0, 0))
        self.mdm = MDM(metric, n_jobs)

    def _initial_transform(self, X):
        X = np.asarray(X, dtype=float)
        self.mdm = MDM(self.metric, 1)
        if self.mode == "ftest":
            self.global_mean = gmean(X, metric=self.metric)
            return X
        elif self.mode == "pairwise":
            return pairwise_distance(X, metric=self.metric, squared=True)
        return X

    def score(self, X, y, groups = None):
        y = np.asarray(y, dtype=int)
        if self.mode == "ttest":
            return self._score_ttest(X, y)
        elif self.mode == "ftest":
            return self._score_ftest(X, y)
        return self._score_pairwise(X, y)

    def _score_ftest(self, X, y):
        mdm = self.mdm.fit(X, y)
        covmeans = np.array(mdm.covmeans_)
        n_classes = covmeans.shape[0]
        between = 0.0
        for ix in range(n_classes):
            di = distance(covmeans[ix], self.global_mean, metric=self.metric, squared=True)
            between += float(np.sum(y == mdm.classes_[ix])) * float(di[0, 0])
        between /= n_classes - 1
        within = 0.0
        for ix in range(n_classes):
            within += float(np.sum(distance(X[y == mdm.classes_[ix]], covmeans[ix],
                                            metric=self.metric, squared=True)))
        within /= y.shape[0] - n_classes
        return between / within

    def _score_ttest(self, X, y):
        mdm = self.mdm.fit(X, y)
        covmeans = np.array(mdm.covmeans_)
        n_classes = covmeans.shape[0]
        pairs = pairwise_distance(covmeans, metric=self.metric)
        mean_dist = float(np.sum(np.triu(pairs)))
        mean_dist /= n_classes * (n_classes - 1) / 2.0
        dist = 0.0
        for ix in range(n_classes):
            di = float(np.mean(distance(X[y == mdm.classes_[ix]], covmeans[ix],
                                        metric=self.metric, squared=True)))
            dist += di / float(np.sum(y == mdm.classes_[ix]))
        return mean_dist / np.sqrt(dist)

    def _score_pairwise(self, X, y):
        classes = np.unique(y)
        n_classes = classes.shape[0]
        n_samples = y.shape[0]
        total_ss = float(np.sum(X)) / (2 * n_samples)
        pattern = np.zeros((n_samples, n_samples))
        for c in classes:
            ix = np.asarray(y == c, dtype=float)
            pattern += np.outer(ix, ix) / np.sum(ix)
        within_ss = float(np.sum(X * pattern)) / 2
        between_ss = total_ss - within_ss
        return between_ss / (n_classes - 1) / (within_ss / (n_samples - n_classes))

    def test(self, X, y, groups = None, verbose: bool = True):
        p, s = _perm_test(self, X, y, groups, self.n_perms, self.random_state)
        self.p_value_ = p
        self.scores_ = s
        return p, s
