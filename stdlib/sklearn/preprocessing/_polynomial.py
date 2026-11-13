# Copyright (C) 2026 the Typhon authors
# Ported from scikit-learn 1.9.1 (BSD-3-Clause)

# PolynomialFeatures (dense output).

from ..base import estimator_repr as _estimator_repr
import numpy as np

def _combos(n: int, deg: int, interaction_only: bool) -> List[List[int]]:
    # itertools.combinations(_with_replacement)(range(n), deg) in order
    out = List[List[int]]()
    if deg == 0:
        out.append(List[int]())
        return out
    cur = [0] * deg
    if interaction_only:
        if deg > n:
            return out
        for i in range(deg):
            cur[i] = i
    while True:
        out.append(cur.copy())
        i = deg - 1
        if interaction_only:
            while i >= 0 and cur[i] == n - deg + i:
                i -= 1
            if i < 0:
                return out
            cur[i] += 1
            for j in range(i + 1, deg):
                cur[j] = cur[j - 1] + 1
        else:
            while i >= 0 and cur[i] == n - 1:
                i -= 1
            if i < 0:
                return out
            v = cur[i] + 1
            for j in range(i, deg):
                cur[j] = v

class PolynomialFeatures:
    def __repr__(self) -> str:
        return _estimator_repr(self)

    degree: int
    min_degree: int
    interaction_only: bool
    include_bias: bool
    order: str
    n_features_in_: int
    n_output_features_: int
    _combs: List[List[int]]

    def __init__(self, degree = 2, *, interaction_only: bool = False, include_bias: bool = True,
                 order: str = "C"):
        if isinstance(degree, int):
            self.min_degree = 0
            self.degree = degree
        else:
            self.min_degree = int(degree[0])
            self.degree = int(degree[1])
        if self.degree < 0 or self.min_degree < 0 or self.min_degree > self.degree:
            raise ValueError(f"degree=({self.min_degree}, {self.degree}) must be non-negative integers with min_degree <= max_degree")
        self.interaction_only = interaction_only
        self.include_bias = include_bias
        self.order = order
        self.n_features_in_ = 0
        self.n_output_features_ = 0
        self._combs = []

    def fit(self, X, y = None):
        Xa = np.asarray(X, dtype=float)
        p = Xa.shape[1]
        self.n_features_in_ = p
        cs = List[List[int]]()
        if self.include_bias:
            cs.append(List[int]())
        for d in range(max(1, self.min_degree), self.degree + 1):
            for c in _combos(p, d, self.interaction_only):
                cs.append(c)
        if len(cs) == 0:
            raise ValueError("Setting degree to zero and include_bias to False would result in an empty output array.")
        self._combs = cs
        self.n_output_features_ = len(cs)
        return self

    @property
    def powers_(self):
        out = np.zeros((len(self._combs), self.n_features_in_), dtype=int)
        for i in range(len(self._combs)):
            for j in self._combs[i]:
                out[i, j] += 1
        return out

    def get_feature_names_out(self, input_features = None) -> List[str]:
        names = List[str]()
        for c in self._combs:
            if len(c) == 0:
                names.append("1")
                continue
            parts = List[str]()
            k = 0
            while k < len(c):
                m = k
                while m < len(c) and c[m] == c[k]:
                    m += 1
                nm = f"x{c[k]}" if input_features is None else str(input_features[c[k]])
                parts.append(nm if m - k == 1 else f"{nm}^{m - k}")
                k = m
            names.append(" ".join(parts))
        return names

    def transform(self, X):
        Xa = np.asarray(X, dtype=float)
        n = Xa.shape[0]
        if Xa.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {Xa.shape[1]} features, but PolynomialFeatures is expecting {self.n_features_in_} features as input.")
        out = np.empty((n, len(self._combs)))
        for j in range(len(self._combs)):
            c = self._combs[j]
            for i in range(n):
                if len(c) == 0:
                    out[i, j] = 1.0
                else:
                    v = Xa[i, c[0]]
                    for k in range(1, len(c)):
                        v = v * Xa[i, c[k]]
                    out[i, j] = v
        return out

    def fit_transform(self, X, y = None):
        return self.fit(X).transform(X)

