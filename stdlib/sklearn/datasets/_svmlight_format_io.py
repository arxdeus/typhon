# Copyright (C) 2026 the Typhon authors
# Ported from scikit-learn 1.9.1 (BSD-3-Clause)

# svmlight / libsvm text format reader and writer
# (sklearn/datasets/_svmlight_format_io.py + _svmlight_format_fast.pyx).

import numpy as np
from scipy.sparse import csr_matrix

class _Parsed:
    data: List[float]
    indices: List[int]
    indptr: List[int]
    labels: List[float]
    query: List[int]

    def __init__(self):
        self.data = List[float]()
        self.indices = List[int]()
        self.indptr = [0]
        self.labels = List[float]()
        self.query = List[int]()

def _parse_file(path: str, zero_based: bool) -> _Parsed:
    p = _Parsed()
    with open(path) as f:
        for raw in f:
            line = raw
            h = line.find("#")
            if h >= 0:
                line = line[:h]
            toks = line.split()
            if len(toks) == 0:
                continue
            p.labels.append(float(toks[0]))
            prev = -1
            for t in toks[1:]:
                c = t.find(":")
                if c < 0:
                    raise ValueError(f"expected 'idx:value' in {repr(t)}")
                key = t[:c]
                if key == "qid":
                    p.query.append(int(t[c + 1:]))
                    continue
                idx = int(key)
                if not zero_based:
                    idx -= 1
                if idx < 0:
                    raise ValueError("Feature indices in svmlight format must be positive unless zero_based=True")
                if idx <= prev:
                    raise ValueError("ill-formed file: feature indices must be strictly increasing")
                prev = idx
                p.indices.append(idx)
                p.data.append(float(t[c + 1:]))
            p.indptr.append(len(p.indices))
    return p

def _auto_zero_based(ps: List[_Parsed]) -> bool:
    # sklearn: zero_based="auto" -> one-based unless some file has index 0
    mn = -1
    first = True
    for p in ps:
        for i in p.indices:
            if first or i < mn:
                mn = i
                first = False
    return mn == 0 if not first else False

def _to_csr(p: _Parsed, n_features: int, shift: int):
    nnz = len(p.indices)
    data = np.empty(nnz)
    ind = np.empty(nnz, dtype=int)
    for k in range(nnz):
        data[k] = p.data[k]
        ind[k] = p.indices[k] + shift
    ptr = np.empty(len(p.indptr), dtype=int)
    for k in range(len(p.indptr)):
        ptr[k] = p.indptr[k]
    return csr_matrix((data, ind, ptr), shape=(len(p.labels), n_features))

def _read_all(paths: List[str], n_features: int, zero_based):
    # first pass: indices are parsed as one-based (shift -1) only if zero_based is False
    parsed = List[_Parsed]()
    zb_known = isinstance(zero_based, bool)
    zb = False
    if zb_known:
        zb = bool(zero_based)
    for pth in paths:
        parsed.append(_parse_file(pth, True))      # raw indices, shifted below
    if not zb_known:
        zb = _auto_zero_based(parsed)
    shift = 0 if zb else -1
    mx = -1
    for p in parsed:
        for i in p.indices:
            if i + shift > mx:
                mx = i + shift
            if i + shift < 0:
                raise ValueError("Feature indices in svmlight format must be positive unless zero_based=True")
    nf = n_features if n_features > 0 else mx + 1
    if n_features > 0 and mx + 1 > n_features:
        raise ValueError(f"n_features was set to {n_features}, but input file contains {mx + 1} features")
    return parsed, nf, shift

def _labels_arr(p: _Parsed):
    y = np.empty(len(p.labels))
    for k in range(len(p.labels)):
        y[k] = p.labels[k]
    return y

def load_svmlight_file(f: str, *, n_features: int = 0, dtype: type = float, multilabel: bool = False,
                       zero_based = "auto", query_id: Literal[bool] = False, offset: int = 0,
                       length: int = -1):
    """Load a svmlight file as (csr_matrix X, ndarray y) (plus query ids)."""
    parsed, nf, shift = _read_all([f], n_features, zero_based)
    X = _to_csr(parsed[0], nf, shift)
    y = _labels_arr(parsed[0])
    if query_id:
        q = np.empty(len(parsed[0].query), dtype=int)
        for k in range(len(q)):
            q[k] = parsed[0].query[k]
        return X, y, q
    return X, y

def load_svmlight_files(files: List[str], *, n_features: int = 0, dtype: type = float,
                        multilabel: bool = False, zero_based = "auto",
                        query_id: Literal[bool] = False, offset: int = 0, length: int = -1):
    """Load several files with a common number of features: list of
    X_0, y_0, X_1, y_1, ... in a flat tuple-like list of (X, y) pairs."""
    parsed, nf, shift = _read_all(files, n_features, zero_based)
    out = List[Tuple[type(_to_csr(parsed[0], nf, shift)), np.ndarray[float, 1]]]()
    for p in parsed:
        out.append((_to_csr(p, nf, shift), _labels_arr(p)))
    return out

def dump_svmlight_file(X, y, f: str, *, zero_based: bool = True, comment: Optional[str] = None,
                       query_id = None, multilabel: bool = False):
    """Write X (dense or sparse) and y in svmlight format."""
    Xs = csr_matrix(X)
    ya = np.asarray(y)
    if Xs.shape[0] != ya.shape[0]:
        raise ValueError(f"X.shape[0] and y.shape[0] should be the same, got: {Xs.shape[0]} and {ya.shape[0]}")
    off = 0 if zero_based else 1
    with open(f, "w") as out:
        if comment is not None:
            out.write("# Generated by dump_svmlight_file from scikit-learn\n")
            out.write("# " + str(comment) + "\n")
        for i in range(Xs.shape[0]):
            line = f"{float(ya[i])!r}" if False else _fmt(float(ya[i]))
            for k in range(Xs.indptr[i], Xs.indptr[i + 1]):
                line += f" {Xs.indices[k] + off}:{_fmt(float(Xs.data[k]))}"
            out.write(line + "\n")

def _fmt(v: float) -> str:
    # sklearn writes with "%.16g"-like shortest repr; integers print without ".0"
    if v == float(int(v)) and abs(v) < 1e15:
        return str(int(v))
    return repr(v)
