# Copyright (C) 2026 the Typhon authors
# Ported from scikit-learn 1.9.1 (BSD-3-Clause)

# OneHotEncoder / OrdinalEncoder over dense 2-D arrays. The category type T is
# a class parameter: OneHotEncoder() works on float columns,
# OneHotEncoder[str]() / OrdinalEncoder[int]() on string / int columns.
# Output is always dense (sparse_output is accepted but ignored); dropped or
# absent indices are encoded as -1 instead of None.

import numpy as np
from ._enc_base import _check_X, _enc_fit, _enc_transform, _is_nan, _infrequent_categories

def _min_freq(mf) -> Tuple[int, float]:
    if mf is None:
        return (0, 0.0)
    elif isinstance(mf, int):
        return (1, float(mf))
    else:
        return (2, float(mf))

def _cats_param(categories, T: type):
    out = List[np.ndarray[T, 1]]()
    if not isinstance(categories, str):
        for c in categories:
            out.append(np.asarray(c, dtype=T))
    return out

class OneHotEncoder[T = float]:
    categories: List[np.ndarray[T, 1]]
    _auto: bool
    drop: Optional[str]
    _drop_values: List[T]
    sparse_output: bool
    handle_unknown: str
    min_frequency: Optional[float]
    _min_freq_kind: int
    _min_freq: float
    max_categories: Optional[int]
    categories_: List[np.ndarray[T, 1]]
    n_features_in_: int
    _infrequent_indices: List[np.ndarray[int, 1]]
    _default_to_infrequent: List[np.ndarray[int, 1]]
    _drop_after: List[int]
    drop_idx_: List[int]
    _n_features_outs: List[int]

    def __init__(self, *, categories = "auto", drop = None, sparse_output: bool = True,
                 dtype: type = float, handle_unknown: str = "error", min_frequency = None,
                 max_categories: Optional[int] = None, feature_name_combiner: str = "concat"):
        self._auto = isinstance(categories, str)
        self.categories = _cats_param(categories, T)
        self._drop_values = []
        if drop is None:
            self.drop = None
        elif isinstance(drop, str):
            self.drop = drop
        else:
            self.drop = "array"
            for v in drop:
                self._drop_values.append(T(v))
        self.sparse_output = sparse_output
        self.handle_unknown = handle_unknown
        self.min_frequency = None if min_frequency is None else float(min_frequency)
        k, f = _min_freq(min_frequency)
        self._min_freq_kind = k
        self._min_freq = f
        self.max_categories = max_categories
        self.categories_ = []
        self.n_features_in_ = 0
        self._infrequent_indices = []
        self._default_to_infrequent = []
        self._drop_after = []
        self.drop_idx_ = []
        self._n_features_outs = []

    def _infrequent_enabled(self) -> bool:
        return (self.max_categories is not None and self.max_categories >= 1) or self._min_freq_kind != 0

    @property
    def infrequent_categories_(self):
        return [_infrequent_categories(self, i) for i in range(len(self.categories_))]

    def _map_drop_idx(self, i: int, d: int) -> int:
        if not self._infrequent_enabled():
            return d
        mp = self._default_to_infrequent[i]
        if mp.shape[0] == 0:
            return d
        for k in self._infrequent_indices[i]:
            if k == d:
                raise ValueError(f"Unable to drop category {self.categories_[i][d]!r} from feature {i} because it is infrequent")
        return mp[d]

    def _set_drop_idx(self):
        p = len(self.categories_)
        after = List[int]()
        if self.drop is None:
            pass
        elif self.drop == "first":
            after = [0] * p
        elif self.drop == "if_binary":
            for i in range(p):
                nf = self.categories_[i].shape[0]
                if self._infrequent_enabled() and self._infrequent_indices[i].shape[0] > 0:
                    nf -= self._infrequent_indices[i].shape[0] - 1
                after.append(0 if nf == 2 else -1)
        else:
            if len(self._drop_values) != p:
                raise ValueError(f"`drop` should have length equal to the number of features ({p}), got {len(self._drop_values)}")
            for i in range(p):
                dv = self._drop_values[i]
                cats = self.categories_[i]
                found = -1
                for k in range(cats.shape[0]):
                    if cats[k] == dv or (_is_nan(dv) and _is_nan(cats[k])):
                        found = k
                        break
                if found < 0:
                    raise ValueError(f"The following categories were supposed to be dropped, but were not found in the training data.\nCategory: {dv}, Feature: {i}")
                after.append(self._map_drop_idx(i, found))
        self._drop_after = after
        if not self._infrequent_enabled() or self.drop is None:
            self.drop_idx_ = after.copy()
        else:
            self.drop_idx_ = []
            for i in range(p):
                mp = self._default_to_infrequent[i]
                d = after[i]
                if d < 0 or mp.shape[0] == 0:
                    self.drop_idx_.append(d)
                else:
                    self.drop_idx_.append(int(np.flatnonzero(mp == d)[0]))

    def _compute_n_features_outs(self):
        out = [c.shape[0] for c in self.categories_]
        if self.drop is not None:
            for i in range(len(out)):
                if self._drop_after[i] >= 0:
                    out[i] -= 1
        if self._infrequent_enabled():
            for i in range(len(out)):
                ni = self._infrequent_indices[i].shape[0]
                if ni > 0:
                    out[i] -= ni - 1
        return out

    def fit(self, X, y = None):
        Xa = _check_X(X, T)
        _enc_fit(self, Xa, self.handle_unknown, False)
        self._set_drop_idx()
        self._n_features_outs = self._compute_n_features_outs()
        return self

    def transform(self, X):
        Xa = _check_X(X, T)
        hu = "infrequent_if_exist" if self.handle_unknown == "warn" else self.handle_unknown
        X_int, X_mask = _enc_transform(self, Xa, hu, [-1] * Xa.shape[1])
        n, p = X_int.shape
        if self.drop is not None:
            for i in range(p):
                d = self._drop_after[i]
                for r in range(n):
                    if d >= 0 and X_int[r, i] == d:
                        X_mask[r, i] = False
                    td = d if d >= 0 else self.categories_[i].shape[0]
                    if X_int[r, i] > td:
                        X_int[r, i] -= 1
        tot = 0
        offs = List[int]()
        for k in self._n_features_outs:
            offs.append(tot)
            tot += k
        out = np.zeros((n, tot))
        for r in range(n):
            for i in range(p):
                if X_mask[r, i]:
                    out[r, offs[i] + X_int[r, i]] = 1.0
        return out

    def fit_transform(self, X, y = None):
        return self.fit(X).transform(X)

    def _transformed_categories(self, i: int, remove_dropped: bool):
        # categories after infrequent grouping (the infrequent bucket is
        # represented by its first infrequent category)
        cats = self.categories_[i]
        if self._infrequent_enabled():
            mp = self._default_to_infrequent[i]
            if mp.shape[0] > 0:
                mx = int(np.max(mp))
                keep = List[T]()
                for k in range(mp.shape[0]):
                    if mp[k] < mx:
                        keep.append(cats[k])
                keep.append(cats[self._infrequent_indices[i][0]])
                cats = np.array(keep, dtype=T)
        if remove_dropped and self.drop is not None and self._drop_after[i] >= 0:
            cats = np.delete(cats, self._drop_after[i])
        return cats

    def inverse_transform(self, X):
        """Unknown (all-zero) rows decode to the first category with
        handle_unknown='ignore' (sklearn: None); the infrequent bucket decodes
        to its first infrequent category (sklearn: 'infrequent_sklearn')."""
        Xa = np.asarray(X, dtype=float)
        n = Xa.shape[0]
        p = len(self.categories_)
        nout = 0
        for k in self._n_features_outs:
            nout += k
        if Xa.shape[1] != nout:
            raise ValueError(f"Shape of the passed X data is not correct. Expected {nout} columns, got {Xa.shape[1]}.")
        out = np.empty((n, p), dtype=T)
        j = 0
        for i in range(p):
            full = self._transformed_categories(i, False)
            cw = self._transformed_categories(i, True)
            nc = cw.shape[0]
            d = self._drop_after[i] if self.drop is not None else -1
            for r in range(n):
                if nc == 0:
                    out[r, i] = self.categories_[i][d]
                    continue
                best = 0
                s = 0.0
                for c in range(nc):
                    s += Xa[r, j + c]
                    if Xa[r, j + c] > Xa[r, j + best]:
                        best = c
                if s == 0.0:
                    if d >= 0:
                        out[r, i] = full[d]
                    elif self.handle_unknown == "error":
                        raise ValueError(f"Samples [{r}] can not be inverted when drop=None and handle_unknown='error' because they contain all zeros")
                    else:
                        out[r, i] = cw[best]
                else:
                    out[r, i] = cw[best]
            j += nc
        return out
