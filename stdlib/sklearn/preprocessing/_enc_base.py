# Copyright (C) 2026 the Typhon authors
# Ported from scikit-learn 1.9.1 (BSD-3-Clause)

# Shared machinery of OneHotEncoder / OrdinalEncoder (_BaseEncoder). The
# category element type T is a class parameter (float by default; use
# OneHotEncoder[str]() for string columns). "None" sentinels from sklearn are
# represented by empty index arrays / -1.

import numpy as np

def _is_nan(v) -> bool:
    if isinstance(v, float):
        return v != v
    else:
        return False

def _unique_counts(col: np.ndarray[T, 1], T: type):
    """sklearn.utils._encode._unique with counts: sorted uniques, a single
    trailing NaN for float input."""
    n = col.shape[0]
    vals = List[T]()
    nn = 0
    for i in range(n):
        if _is_nan(col[i]):
            nn += 1
        else:
            vals.append(col[i])
    vals.sort()
    u = List[T]()
    c = List[int]()
    for v in vals:
        if len(u) > 0 and u[-1] == v:
            c[-1] += 1
        else:
            u.append(v)
            c.append(1)
    if nn > 0:
        u.append(col[0] if _is_nan(col[0]) else _nan_of(T))
        c.append(nn)
    return np.array(u, dtype=T), np.array(c, dtype=int)

def _nan_of(T: type) -> T:
    if T is float:
        return np.nan
    else:
        return T()

class _Lookup[T]:
    table: Dict[T, int]
    nan_idx: int

    def __init__(self, cats: np.ndarray[T, 1]):
        self.table = {}
        self.nan_idx = -1
        for i in range(cats.shape[0]):
            if _is_nan(cats[i]):
                self.nan_idx = i
            else:
                self.table[cats[i]] = i

    def find(self, v: T) -> int:
        if _is_nan(v):
            return self.nan_idx
        return self.table.get(v, -1)

def _check_X(X, T: type):
    Xa = np.asarray(X, dtype=T)
    if static.len(Xa.shape) != 2:
        compile_error("Expected 2D array")
    return Xa

def _enc_fit(enc, Xa, handle_unknown: str, ignore_missing_for_infrequent: bool):
    """_BaseEncoder._fit; sets categories_, n_features_in_ and the infrequent
    mapping. Returns the per-feature missing index (-1 when absent)."""
    n, p = Xa.shape
    enc.n_features_in_ = p
    if not enc._auto and len(enc.categories) != p:
        raise ValueError("Shape mismatch: if categories is an array, it has to be of shape (n_features,).")
    enc.categories_ = []
    counts_l = List[np.ndarray[int, 1]]()
    for i in range(p):
        col = np.ascontiguousarray(Xa[:, i])
        if enc._auto:
            u, c = _unique_counts(col)
            enc.categories_.append(u)
            counts_l.append(c)
        else:
            cats = enc.categories[i].copy()
            if cats.shape[0] != _unique_counts(cats)[0].shape[0]:
                raise ValueError(f"In column {i}, the predefined categories contain duplicate elements.")
            if isinstance(cats[0], float):
                for k in range(cats.shape[0] - 1):
                    if _is_nan(cats[k]) or (not _is_nan(cats[k + 1]) and cats[k + 1] < cats[k]):
                        raise ValueError("Unsorted categories are not supported for numerical categories")
            lk = _Lookup(cats)
            cnt = np.zeros(cats.shape[0], dtype=int)
            for r in range(n):
                k = lk.find(col[r])
                if k < 0:
                    if handle_unknown == "error":
                        raise ValueError(f"Found unknown categories ['{col[r]}'] in column {i} during fit")
                else:
                    cnt[k] += 1
            enc.categories_.append(cats)
            counts_l.append(cnt)
    missing = List[int]()
    for i in range(p):
        c = enc.categories_[i]
        missing.append(c.shape[0] - 1 if (ignore_missing_for_infrequent and c.shape[0] > 0 and _is_nan(c[c.shape[0] - 1])) else -1)
    enc._infrequent_indices = [np.empty(0, dtype=int) for _ in range(p)]
    enc._default_to_infrequent = [np.empty(0, dtype=int) for _ in range(p)]
    if enc._infrequent_enabled():
        for i in range(p):
            cc = counts_l[i]
            ncat = enc.categories_[i].shape[0]
            if missing[i] >= 0:
                cc = np.delete(cc, missing[i])
                ncat -= 1
            inf = _identify_infrequent(enc, cc, n)
            enc._infrequent_indices[i] = inf
            if inf.shape[0] > 0:
                mapping = np.empty(ncat, dtype=int)
                nfreq = ncat - inf.shape[0]
                isinf = np.zeros(ncat, dtype=bool)
                for k in inf:
                    isinf[k] = True
                q = 0
                for k in range(ncat):
                    if isinf[k]:
                        mapping[k] = nfreq
                    else:
                        mapping[k] = q
                        q += 1
                enc._default_to_infrequent[i] = mapping
    return missing

def _identify_infrequent(enc, cc: np.ndarray[int, 1], n: int):
    m = cc.shape[0]
    mask = np.zeros(m, dtype=bool)
    if enc._min_freq_kind == 1:
        for k in range(m):
            mask[k] = float(cc[k]) < enc._min_freq
    elif enc._min_freq_kind == 2:
        thr = float(n) * enc._min_freq
        for k in range(m):
            mask[k] = float(cc[k]) < thr
    ncur = m - int(np.sum(mask)) + 1
    if enc.max_categories is not None:
        mc = enc.max_categories
        if mc < ncur:
            fc = mc - 1
            if fc == 0:
                mask[:] = True
            else:
                order = np.argsort(cc, kind="mergesort")
                for k in range(m - fc):
                    mask[order[k]] = True
    return np.flatnonzero(mask)

def _enc_transform(enc, Xa, handle_unknown: str, ignore_missing):
    """_BaseEncoder._transform -> (X_int, X_mask)."""
    n, p = Xa.shape
    if p != enc.n_features_in_:
        raise ValueError(f"X has {p} features, but {enc.__class__.__name__} is expecting {enc.n_features_in_} features as input.")
    X_int = np.zeros((n, p), dtype=int)
    X_mask = np.ones((n, p), dtype=bool)
    for i in range(p):
        lk = _Lookup(enc.categories_[i])
        for r in range(n):
            k = lk.find(Xa[r, i])
            if k < 0:
                if handle_unknown == "error":
                    raise ValueError(f"Found unknown categories ['{Xa[r, i]}'] in column {i} during transform")
                X_mask[r, i] = False
                k = 0
            X_int[r, i] = k
    if enc._infrequent_enabled():
        for i in range(p):
            inf = enc._infrequent_indices[i]
            if inf.shape[0] == 0:
                continue
            for r in range(n):
                if not X_mask[r, i]:
                    X_int[r, i] = inf[0]
                    if handle_unknown == "infrequent_if_exist" or handle_unknown == "warn":
                        X_mask[r, i] = True
        for i in range(p):
            mp = enc._default_to_infrequent[i]
            if mp.shape[0] == 0:
                continue
            for r in range(n):
                if ignore_missing[i] >= 0 and X_int[r, i] == ignore_missing[i]:
                    continue
                X_int[r, i] = mp[X_int[r, i]]
    return X_int, X_mask

def _infrequent_categories(enc, i: int):
    inf = enc._infrequent_indices[i]
    return enc.categories_[i][inf]
