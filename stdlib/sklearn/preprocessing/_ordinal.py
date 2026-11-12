# Copyright (C) 2026 the Typhon authors
# Ported from scikit-learn 1.9.1 (BSD-3-Clause)

import numpy as np
from ._enc_base import _check_X, _enc_fit, _enc_transform, _is_nan, _infrequent_categories
from ._encoders import _min_freq, _cats_param

class OrdinalEncoder[T = float]:
    """handle_unknown 'error' / 'use_encoded_value'; output dtype float.
    inverse_transform maps unknown_value rows to the first category
    (sklearn: None) and the infrequent bucket to its first infrequent
    category (sklearn: 'infrequent_sklearn')."""
    categories: List[np.ndarray[T, 1]]
    _auto: bool
    handle_unknown: str
    unknown_value: Optional[float]
    encoded_missing_value: float
    min_frequency: Optional[float]
    _min_freq_kind: int
    _min_freq: float
    max_categories: Optional[int]
    categories_: List[np.ndarray[T, 1]]
    n_features_in_: int
    _infrequent_indices: List[np.ndarray[int, 1]]
    _default_to_infrequent: List[np.ndarray[int, 1]]
    _missing_indices: List[int]

    def __init__(self, *, categories = "auto", dtype: type = float, handle_unknown: str = "error",
                 unknown_value = None, encoded_missing_value: float = np.nan,
                 min_frequency = None, max_categories: Optional[int] = None):
        self._auto = isinstance(categories, str)
        self.categories = _cats_param(categories, T)
        self.handle_unknown = handle_unknown
        self.unknown_value = None if unknown_value is None else float(unknown_value)
        self.encoded_missing_value = float(encoded_missing_value)
        self.min_frequency = None if min_frequency is None else float(min_frequency)
        k, f = _min_freq(min_frequency)
        self._min_freq_kind = k
        self._min_freq = f
        self.max_categories = max_categories
        self.categories_ = []
        self.n_features_in_ = 0
        self._infrequent_indices = []
        self._default_to_infrequent = []
        self._missing_indices = []

    def _infrequent_enabled(self) -> bool:
        return (self.max_categories is not None and self.max_categories >= 1) or self._min_freq_kind != 0

    @property
    def infrequent_categories_(self):
        return [_infrequent_categories(self, i) for i in range(len(self.categories_))]

    def fit(self, X, y = None):
        if self.handle_unknown == "use_encoded_value":
            if self.unknown_value is None:
                raise TypeError("unknown_value should be an integer or np.nan when handle_unknown is 'use_encoded_value', got None.")
        elif self.unknown_value is not None:
            raise TypeError(f"unknown_value should only be set when handle_unknown is 'use_encoded_value', got {self.unknown_value}.")
        Xa = _check_X(X, T)
        self._missing_indices = _enc_fit(self, Xa, self.handle_unknown, True)
        p = len(self.categories_)
        card = [c.shape[0] for c in self.categories_]
        for i in range(p):
            if self._infrequent_enabled():
                card[i] -= self._infrequent_indices[i].shape[0]
            if self._missing_indices[i] >= 0:
                card[i] -= 1
        if self.handle_unknown == "use_encoded_value":
            uv = 0.0 if self.unknown_value is None else float(self.unknown_value)
            if uv == uv:
                for c in card:
                    if 0.0 <= uv and uv < float(c):
                        raise ValueError(f"The used value for unknown_value {uv} is one of the values already used for encoding the seen categories.")
        emv = self.encoded_missing_value
        if emv == emv:
            for i in range(p):
                if self._missing_indices[i] >= 0 and 0.0 <= emv and emv < float(card[i]):
                    raise ValueError(f"encoded_missing_value ({emv}) is already used to encode a known category in features: [{i}]")
        return self

    def transform(self, X):
        Xa = _check_X(X, T)
        X_int, X_mask = _enc_transform(self, Xa, self.handle_unknown, self._missing_indices)
        n, p = X_int.shape
        out = X_int.astype(float)
        for i in range(p):
            mi = self._missing_indices[i]
            if mi >= 0:
                for r in range(n):
                    if X_int[r, i] == mi:
                        out[r, i] = self.encoded_missing_value
        if self.handle_unknown == "use_encoded_value":
            if self.unknown_value is not None:
                uv = float(self.unknown_value)
                for r in range(n):
                    for i in range(p):
                        if not X_mask[r, i]:
                            out[r, i] = uv
        return out

    def fit_transform(self, X, y = None):
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        Xa = np.asarray(X, dtype=float)
        n = Xa.shape[0]
        p = len(self.categories_)
        if Xa.shape[1] != p:
            raise ValueError(f"Shape of the passed X data is not correct. Expected {p} columns, got {Xa.shape[1]}.")
        out = np.empty((n, p), dtype=T)
        for i in range(p):
            cats = self.categories_[i]
            inf = self._infrequent_indices[i] if self._infrequent_enabled() else np.empty(0, dtype=int)
            freq = cats
            infcode = -1
            if inf.shape[0] > 0:
                infcode = cats.shape[0] - inf.shape[0]
                keep = np.ones(cats.shape[0], dtype=bool)
                for k in inf:
                    keep[k] = False
                freq = cats[keep]
            for r in range(n):
                v = Xa[r, i]
                mi = self._missing_indices[i]
                if mi >= 0 and (v == self.encoded_missing_value or (v != v and self.encoded_missing_value != self.encoded_missing_value)):
                    v = float(mi)
                uv = 0.0 if self.unknown_value is None else float(self.unknown_value)
                if self.handle_unknown == "use_encoded_value" and (v == uv or (v != v and uv != uv)):
                    out[r, i] = cats[0]
                elif infcode >= 0 and int(v) == infcode:
                    out[r, i] = cats[inf[0]]
                else:
                    out[r, i] = freq[int(v)]
        return out
