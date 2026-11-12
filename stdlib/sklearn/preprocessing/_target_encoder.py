# Copyright (C) 2026 the Typhon authors
# Ported from scikit-learn 1.9.1 (BSD-3-Clause)

# TargetEncoder with internal cross fitting. Targets are numeric (labels are
# converted to float); classes_ is a float array.

import numpy as np
from ._enc_base import _check_X, _enc_fit, _enc_transform
from ._encoders import _cats_param

def _fit_encoding(X_int, y, ncat: List[int], smooth_auto: bool, smooth: float, y_mean: float):
    n, p = X_int.shape
    out = List[np.ndarray[float, 1]]()
    if not smooth_auto:
        for f in range(p):
            sums = np.full(ncat[f], smooth * y_mean)
            counts = np.full(ncat[f], smooth)
            for i in range(n):
                k = X_int[i, f]
                if k == -1:
                    continue
                sums[k] += y[i]
                counts[k] += 1.0
            enc = np.empty(ncat[f])
            for c in range(ncat[f]):
                enc[c] = y_mean if counts[c] == 0 else sums[c] / counts[c]
            out.append(enc)
        return out
    yv = float(np.var(y))
    for f in range(p):
        means = np.zeros(ncat[f])
        counts = np.zeros(ncat[f], dtype=int)
        ssd = np.zeros(ncat[f])
        for i in range(n):
            k = X_int[i, f]
            if k == -1:
                continue
            counts[k] += 1
            means[k] += y[i]
        for c in range(ncat[f]):
            means[c] = means[c] / float(counts[c]) if counts[c] > 0 else np.nan
        for i in range(n):
            k = X_int[i, f]
            if k == -1:
                continue
            d = y[i] - means[k]
            ssd[k] += d * d
        enc = np.empty(ncat[f])
        for c in range(ncat[f]):
            cc = float(counts[c])
            den = yv * cc + (ssd[c] / cc if cc > 0 else np.nan)
            lam = yv * cc / den if den != 0.0 else np.nan
            enc[c] = y_mean if lam != lam else lam * means[c] + (1.0 - lam) * y_mean
        out.append(enc)
    return out

class TargetEncoder[T = float]:
    """cv: int (KFold / StratifiedKFold with shuffle=True) or a splitter.
    sklearn draws the default shuffle seed from global numpy state; here
    random_state=None means seed 0."""
    categories: List[np.ndarray[T, 1]]
    _auto: bool
    target_type: str
    smooth: str
    _smooth: float
    cv: int
    shuffle: bool
    random_state: Optional[int]
    categories_: List[np.ndarray[T, 1]]
    n_features_in_: int
    _infrequent_indices: List[np.ndarray[int, 1]]
    _default_to_infrequent: List[np.ndarray[int, 1]]
    max_categories: Optional[int]
    _min_freq_kind: int
    _min_freq: float
    target_type_: str
    classes_: np.ndarray[float, 1]
    target_mean_: np.ndarray[float, 1]
    encodings_: List[np.ndarray[float, 1]]

    def __init__(self, categories = "auto", target_type: str = "auto", smooth = "auto", cv: int = 5,
                 shuffle: bool = True, random_state: Optional[int] = None):
        self._auto = isinstance(categories, str)
        self.categories = _cats_param(categories, T)
        self.target_type = target_type
        if isinstance(smooth, str):
            self.smooth = smooth
            self._smooth = 0.0
        else:
            self.smooth = ""
            self._smooth = float(smooth)
        self.cv = cv
        self.shuffle = shuffle
        self.random_state = random_state
        self.categories_ = []
        self.n_features_in_ = 0
        self._infrequent_indices = []
        self._default_to_infrequent = []
        self.max_categories = None
        self._min_freq_kind = 0
        self._min_freq = 0.0
        self.target_type_ = ""
        self.classes_ = np.empty(0)
        self.target_mean_ = np.empty(0)
        self.encodings_ = []

    def _infrequent_enabled(self) -> bool:
        return False

    def _encode_y(self, y):
        ya = np.asarray(y, dtype=float).ravel()
        tt = self.target_type
        u = np.unique(ya)
        if tt == "auto":
            cont = False
            for v in ya:
                if v != float(int(v)):
                    cont = True
            tt = "continuous" if cont else ("binary" if u.shape[0] <= 2 else "multiclass")
        self.target_type_ = tt
        if tt == "binary":
            self.classes_ = u
            return np.searchsorted(u, ya).astype(float).reshape(ya.shape[0], 1)
        elif tt == "multiclass":
            self.classes_ = u
            Y = np.zeros((ya.shape[0], u.shape[0]))
            idx = np.searchsorted(u, ya)
            for i in range(ya.shape[0]):
                Y[i, idx[i]] = 1.0
            return Y
        self.classes_ = np.empty(0)
        return ya.reshape(ya.shape[0], 1)

    def _encodings(self, X_int, Y, ncat: List[int], mean):
        p = X_int.shape[1]
        nc = Y.shape[1]
        per = List[List[np.ndarray[float, 1]]]()
        for c in range(nc):
            per.append(_fit_encoding(X_int, np.ascontiguousarray(Y[:, c]), ncat, self.smooth == "auto", self._smooth, mean[c]))
        out = List[np.ndarray[float, 1]]()
        for f in range(p):
            for c in range(nc):
                out.append(per[c][f])
        return out

    def _apply(self, X_out, X_int, X_mask, rows, enc, mean):
        nc = mean.shape[0]
        for e in range(len(enc)):
            f = e // nc
            for r in rows:
                X_out[r, e] = enc[e][X_int[r, f]]
            for r in range(X_int.shape[0]):
                if not X_mask[r, f]:
                    X_out[r, e] = mean[e % nc]

    def fit(self, X, y):
        self._fit_all(X, y)
        return self

    def _fit_all(self, X, y):
        Xa = _check_X(X, T)
        _enc_fit(self, Xa, "ignore", False)
        Y = self._encode_y(y)
        if Y.shape[0] != Xa.shape[0]:
            raise ValueError("Found input variables with inconsistent numbers of samples")
        self.target_mean_ = np.mean(Y, axis=0)
        X_int, X_mask = _enc_transform(self, Xa, "ignore", [-1] * Xa.shape[1])
        for r in range(X_int.shape[0]):
            for f in range(X_int.shape[1]):
                if not X_mask[r, f]:
                    X_int[r, f] = -1
        ncat = [c.shape[0] for c in self.categories_]
        self.encodings_ = self._encodings(X_int, Y, ncat, self.target_mean_)
        return Xa, X_int, X_mask, Y, ncat

    def fit_transform(self, X, y):
        from ..model_selection import KFold, StratifiedKFold
        Xa, X_int, X_mask, Y, ncat = self._fit_all(X, y)
        n = Xa.shape[0]
        nc = self.target_mean_.shape[0]
        X_out = np.empty((n, X_int.shape[1] * nc))
        rs = 0 if self.random_state is None else self.random_state
        folds = List[Tuple[np.ndarray[int, 1], np.ndarray[int, 1]]]()
        if self.target_type_ == "continuous":
            for tr, te in KFold(self.cv, shuffle=self.shuffle, random_state=(rs if self.shuffle else None)).split(Xa):
                folds.append((tr, te))
        else:
            for tr, te in StratifiedKFold(self.cv, shuffle=self.shuffle, random_state=(rs if self.shuffle else None)).split(Xa, np.asarray(y, dtype=float).ravel()):
                folds.append((tr, te))
        for tr, te in folds:
            Ytr = Y[tr]
            m = np.mean(Ytr, axis=0)
            enc = self._encodings(np.ascontiguousarray(X_int[tr]), Ytr, ncat, m)
            self._apply(X_out, X_int, np.ones(X_mask.shape, dtype=bool), te, enc, m)
        for r in range(n):
            for f in range(X_int.shape[1]):
                if not X_mask[r, f]:
                    for c in range(nc):
                        X_out[r, f * nc + c] = self.target_mean_[c]
        return X_out

    def transform(self, X):
        Xa = _check_X(X, T)
        X_int, X_mask = _enc_transform(self, Xa, "ignore", [-1] * Xa.shape[1])
        nc = self.target_mean_.shape[0]
        X_out = np.empty((Xa.shape[0], Xa.shape[1] * nc))
        self._apply(X_out, X_int, X_mask, range(Xa.shape[0]), self.encodings_, self.target_mean_)
        return X_out
