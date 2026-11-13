# Copyright (C) 2026 the Typhon authors
# Ported from scikit-learn 1.9.1 (BSD-3-Clause)

# LabelEncoder / LabelBinarizer / label_binarize / MultiLabelBinarizer for 1-d
# label arrays (binary / multiclass targets). The label type T is a class
# parameter (int by default; LabelEncoder[str]() for strings).

from ..base import estimator_repr as _estimator_repr
import numpy as np

def _unique_sorted(y: np.ndarray[T, 1], T: type):
    l = List[T]()
    for v in y:
        l.append(v)
    l.sort()
    u = List[T]()
    for v in l:
        if len(u) == 0 or u[-1] != v:
            u.append(v)
    return np.array(u, dtype=T)

def _encode_check(classes, y, T: type):
    d = Dict[T, int]()
    for i in range(classes.shape[0]):
        d[classes[i]] = i
    out = np.empty(y.shape[0], dtype=int)
    bad = List[T]()
    for i in range(y.shape[0]):
        k = d.get(y[i], -1)
        if k < 0:
            bad.append(y[i])
        out[i] = k
    if len(bad) > 0:
        raise ValueError(f"y contains previously unseen labels: {_unique_sorted(np.array(bad, dtype=T))}")
    return out

class LabelEncoder[T = int]:
    def __repr__(self) -> str:
        return _estimator_repr(self)

    classes_: np.ndarray[T, 1]

    def __init__(self):
        self.classes_ = np.empty(0, dtype=T)

    def fit(self, y):
        self.classes_ = _unique_sorted(np.asarray(y, dtype=T).ravel())
        return self

    def fit_transform(self, y):
        return self.fit(y).transform(y)

    def transform(self, y):
        ya = np.asarray(y, dtype=T).ravel()
        return _encode_check(self.classes_, ya, T)

    def inverse_transform(self, y):
        ya = np.asarray(y, dtype=int).ravel()
        nc = self.classes_.shape[0]
        for v in ya:
            if v < 0 or v >= nc:
                raise ValueError(f"y contains previously unseen labels: [{v}]")
        return self.classes_[ya]

def _label_binarize_impl(ya, classes, neg_label: int, pos_label: int, y_type: str, T: type):
    if neg_label >= pos_label:
        raise ValueError(f"neg_label={neg_label} must be strictly less than pos_label={pos_label}.")
    n = ya.shape[0]
    nc = classes.shape[0]
    pos_switch = pos_label == 0
    pl = -neg_label if pos_switch else pos_label
    if y_type == "binary" and nc == 1:
        return np.full((n, 1), neg_label, dtype=int)
    yt = "multiclass" if (y_type == "binary" and nc >= 3) else y_type
    d = Dict[T, int]()
    for i in range(nc):
        d[classes[i]] = i
    Y = np.zeros((n, nc), dtype=int)
    for i in range(n):
        k = d.get(ya[i], -1)
        if k >= 0:
            Y[i, k] = pl
    if neg_label != 0:
        Y[Y == 0] = neg_label
    if pos_switch:
        Y[Y == pl] = 0
    if yt == "binary":
        return np.ascontiguousarray(Y[:, nc - 1:nc])
    return Y

def _type_of_target_1d(ya) -> str:
    u = _unique_sorted(ya)
    if isinstance(ya[0], float):
        for v in ya:
            if v != float(int(v)):
                raise ValueError("continuous target data is not supported with label binarization")
    return "binary" if u.shape[0] <= 2 else "multiclass"

def _lb_entry(ya: np.ndarray[T, 1], classes, neg_label: int, pos_label: int, T: type):
    ca = np.asarray(classes, dtype=T)
    return _label_binarize_impl(ya, ca, neg_label, pos_label, _type_of_target_1d(ya), T)

def label_binarize(y, *, classes, neg_label: int = 0, pos_label: int = 1, sparse_output: bool = False):
    """Dense output only; classes keep the caller's order (as sklearn)."""
    return _lb_entry(np.asarray(y).ravel(), classes, neg_label, pos_label)

class LabelBinarizer[T = int]:
    def __repr__(self) -> str:
        return _estimator_repr(self)

    neg_label: int
    pos_label: int
    sparse_output: bool
    classes_: np.ndarray[T, 1]
    y_type_: str
    sparse_input_: bool

    def __init__(self, *, neg_label: int = 0, pos_label: int = 1, sparse_output: bool = False):
        self.neg_label = neg_label
        self.pos_label = pos_label
        self.sparse_output = sparse_output
        self.classes_ = np.empty(0, dtype=T)
        self.y_type_ = ""
        self.sparse_input_ = False

    def fit(self, y):
        if self.neg_label >= self.pos_label:
            raise ValueError(f"neg_label={self.neg_label} must be strictly less than pos_label={self.pos_label}.")
        ya = np.asarray(y, dtype=T).ravel()
        if ya.shape[0] == 0:
            raise ValueError(f"y has 0 samples: {ya}")
        self.y_type_ = _type_of_target_1d(ya)
        self.classes_ = _unique_sorted(ya)
        return self

    def fit_transform(self, y):
        return self.fit(y).transform(y)

    def transform(self, y):
        ya = np.asarray(y, dtype=T).ravel()
        return _label_binarize_impl(ya, self.classes_, self.neg_label, self.pos_label, self.y_type_, T)

    def inverse_transform(self, Y, threshold: Optional[float] = None):
        Ya = np.asarray(Y, dtype=float)
        if static.len(Ya.shape) == 1:
            return self.inverse_transform(Ya.reshape(Ya.shape[0], 1), threshold)
        else:
            thr = (self.pos_label + self.neg_label) / 2.0 if threshold is None else threshold
            n = Ya.shape[0]
            nc = self.classes_.shape[0]
            out = np.empty(n, dtype=T)
            if self.y_type_ == "multiclass":
                idx = np.argmax(Ya, axis=1)
                for i in range(n):
                    out[i] = self.classes_[min(max(int(idx[i]), 0), nc - 1)]
                return out
            if Ya.shape[1] > 2:
                raise ValueError(f"output_type='binary', but y.shape = {Ya.shape}")
            for i in range(n):
                if nc == 1:
                    out[i] = self.classes_[0]
                else:
                    col = 1 if Ya.shape[1] == 2 else 0
                    out[i] = self.classes_[1 if Ya[i, col] > thr else 0]
            return out

class MultiLabelBinarizer[T = int]:
    def __repr__(self) -> str:
        return _estimator_repr(self)

    """y: iterable of iterables of labels; dense int indicator output."""
    classes: Optional[List[T]]
    sparse_output: bool
    classes_: np.ndarray[T, 1]

    def __init__(self, *, classes = None, sparse_output: bool = False):
        if classes is None:
            self.classes = None
        else:
            self.classes = [T(c) for c in classes]
        self.sparse_output = sparse_output
        self.classes_ = np.empty(0, dtype=T)

    def fit(self, y):
        if self.classes is None:
            s = List[T]()
            for row in y:
                for v in row:
                    s.append(T(v))
            self.classes_ = _unique_sorted(np.array(s, dtype=T))
        else:
            cl = List[T]()
            if self.classes is not None:
                cl = self.classes.copy()
            if len(set(cl)) < len(cl):
                raise ValueError("The classes argument contains duplicate classes. Remove these duplicates before passing them to MultiLabelBinarizer.")
            self.classes_ = np.array(cl, dtype=T)
        return self

    def fit_transform(self, y):
        return self.fit(y).transform(y)

    def transform(self, y):
        d = Dict[T, int]()
        for i in range(self.classes_.shape[0]):
            d[self.classes_[i]] = i
        rows = List[List[int]]()
        for row in y:
            r = List[int]()
            for v in row:
                k = d.get(T(v), -1)
                if k >= 0:
                    r.append(k)
            rows.append(r)
        out = np.zeros((len(rows), self.classes_.shape[0]), dtype=int)
        for i in range(len(rows)):
            for k in rows[i]:
                out[i, k] = 1
        return out

    def inverse_transform(self, yt):
        Ya = np.asarray(yt, dtype=int)
        if Ya.shape[1] != self.classes_.shape[0]:
            raise ValueError(f"Expected indicator for {self.classes_.shape[0]} classes, but got {Ya.shape[1]}")
        out = List[List[T]]()
        for i in range(Ya.shape[0]):
            r = List[T]()
            for k in range(Ya.shape[1]):
                if Ya[i, k] != 0 and Ya[i, k] != 1:
                    raise ValueError("Expected only 0s and 1s in label indicator.")
                if Ya[i, k] == 1:
                    r.append(self.classes_[k])
            out.append(r)
        return out
