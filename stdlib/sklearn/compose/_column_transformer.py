# Copyright (C) 2026 the Typhon authors
# Ported from scikit-learn 1.9.1 (BSD-3-Clause)

# ColumnTransformer over dense ndarrays. Column selectors: int, list/array of
# ints, boolean mask, or slice. transformers: static tuple of
# (name, transformer|'drop'|'passthrough', columns). Output is a dense float
# array (sklearn's sparse_threshold stacking is not used).

from ..base import estimator_repr as _estimator_repr
import numpy as np
from ..base import clone
from ..pipeline import _ft1, _hstack2, _NamedSteps

def _cols(sel, p: int) -> np.ndarray[int, 1]:
    if isinstance(sel, int):
        return np.array([sel if sel >= 0 else sel + p])
    elif isinstance(sel, slice):
        return np.arange(p)[sel]
    else:
        a = np.asarray(sel)
        if a.dtype is bool:
            if a.shape[0] != p:
                raise ValueError("boolean column mask has the wrong length")
            return np.nonzero(a)[0]
        else:
            out = a.astype(int)
            for i in range(out.shape[0]):
                if out[i] < 0:
                    out[i] += p
                if out[i] < 0 or out[i] >= p:
                    raise IndexError("column index out of range")
            return out

def _take(X, c):
    if c.shape[0] == 0:
        # (numpy port raises on X[:, empty_int_array])
        return np.empty((X.shape[0], 0), dtype=X.dtype)
    return np.ascontiguousarray(X[:, c])

def _clone_tr(tl):
    if static.len(tl) == 0:
        return ()
    else:
        e = tl[0][1]
        if isinstance(e, str):
            return ((tl[0][0], e, tl[0][2]),) + _clone_tr(tl[1:])
        else:
            return ((tl[0][0], clone(e), tl[0][2]),) + _clone_tr(tl[1:])

class ColumnTransformer[T, R = str]:
    def __repr__(self) -> str:
        return _estimator_repr(self, False)

    transformers: T
    remainder: R
    sparse_threshold: float
    n_jobs: Optional[int]
    verbose: bool
    n_features_in_: int
    _columns: List[np.ndarray[int, 1]]
    _remainder_cols: np.ndarray[int, 1]
    output_indices_: Dict[str, Tuple[int, int]]
    _widths: List[int]

    def __init__(self, transformers: T, *, remainder: R = "drop", sparse_threshold: float = 0.3,
                 n_jobs: Optional[int] = None, transformer_weights = None, verbose: bool = False,
                 verbose_feature_names_out: bool = True, force_int_remainder_cols = None):
        self.transformers = transformers
        self.remainder = remainder
        if isinstance(remainder, str):
            if remainder != "drop" and remainder != "passthrough":
                raise ValueError(f"The remainder keyword needs to be one of 'drop', 'passthrough', or estimator. '{remainder}' was passed instead")
        self.sparse_threshold = sparse_threshold
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.n_features_in_ = 0
        self._columns = []
        self._remainder_cols = np.empty(0, dtype=int)
        self.output_indices_ = {}
        self._widths = []

    def _sk_clone(self):
        r = self.remainder
        if isinstance(r, str):
            return ColumnTransformer(_clone_tr(self.transformers), remainder=r,
                                     sparse_threshold=self.sparse_threshold, n_jobs=self.n_jobs,
                                     verbose=self.verbose)
        else:
            return ColumnTransformer(_clone_tr(self.transformers), remainder=clone(r),
                                     sparse_threshold=self.sparse_threshold, n_jobs=self.n_jobs,
                                     verbose=self.verbose)

    @property
    def named_transformers_(self):
        return _NamedSteps(self.transformers)

    def _resolve(self, p: int):
        self._columns = []
        used = np.zeros(p, dtype=bool)
        for t in self.transformers:
            c = _cols(t[2], p)
            self._columns.append(c)
            used[c] = True
        self._remainder_cols = np.nonzero(~used)[0]

    def _block(self, est, X, c, fit: bool, y):
        Xs = _take(X, c)
        if c.shape[0] == 0 and not isinstance(est, str):
            # sklearn skips transformers on empty column selections
            return np.empty((X.shape[0], 0))
        if isinstance(est, str):
            if est == "drop":
                return np.empty((X.shape[0], 0))
            elif est == "passthrough":
                return Xs.astype(float)
            else:
                raise ValueError(f"unknown transformer string {est!r}")
                return np.empty((0, 0))
        elif fit:
            return np.asarray(_ft1(est, Xs, y), dtype=float)
        else:
            return np.asarray(est.transform(Xs), dtype=float)

    def _run(self, X, y, fit: bool):
        Xa = np.asarray(X)
        n, p = Xa.shape
        if fit:
            self.n_features_in_ = p
            self._resolve(p)
        elif p != self.n_features_in_:
            raise ValueError(f"X has {p} features, but ColumnTransformer is expecting {self.n_features_in_} features as input.")
        out = np.empty((n, 0))
        k = 0
        start = 0
        self._widths = []
        for t in self.transformers:
            b = self._block(t[1], Xa, self._columns[k], fit, y)
            if fit:
                self.output_indices_[t[0]] = (start, start + b.shape[1])
            start += b.shape[1]
            self._widths.append(b.shape[1])
            out = _hstack2(out, b)
            k += 1
        rb = self._block(self.remainder, Xa, self._remainder_cols, fit, y)
        if fit:
            self.output_indices_["remainder"] = (start, start + rb.shape[1])
        out = _hstack2(out, rb)
        return out

    def fit(self, X, y = None):
        self._run(X, y, True)
        return self

    def fit_transform(self, X, y = None):
        return self._run(X, y, True)

    def transform(self, X):
        return self._run(X, None, False)

def _mk(ts):
    if static.len(ts) == 0:
        return ()
    else:
        e = ts[0][0]
        nm = e if isinstance(e, str) else type(e).__name__.lower()
        return ((nm, e, ts[0][1]),) + _mk(ts[1:])

def make_column_transformer(*transformers, remainder = "drop", sparse_threshold: float = 0.3,
                            n_jobs: Optional[int] = None, verbose: bool = False,
                            verbose_feature_names_out: bool = True, force_int_remainder_cols = None):
    """transformers: (transformer, columns) pairs."""
    return ColumnTransformer(_mk(transformers), remainder=remainder,
                             sparse_threshold=sparse_threshold, n_jobs=n_jobs, verbose=verbose)
