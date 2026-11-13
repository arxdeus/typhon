# Ported from scikit-learn 1.9.1 (BSD-3-Clause)

# FeatureHasher (signed 32-bit MurmurHash3 of the UTF-8 feature name, seed 0;
# input_type 'dict' (Dict[str, float|str]), 'pair' or 'string'). Returns a
# scipy.sparse CSR array with duplicates summed.

from ..base import estimator_repr as _estimator_repr
import numpy as np
from ..utils._indexing import _murmur_bytes

def _hash_index(name: str, n_features: int) -> Tuple[int, int]:
    bs = name.encode("utf-8")
    data = [int(bs[i]) & 0xff for i in range(len(bs))]
    h = int(i32(_murmur_bytes(data, 0)))
    idx = 0
    if h == -2147483648:
        idx = (2147483647 - (n_features - 1)) % n_features
    else:
        idx = abs(h) % n_features
    return idx, h

def _hash_rows(rows: List[List[Tuple[str, float]]], n_features: int, alternate_sign: bool):
    import scipy.sparse as sp
    n = len(rows)
    if n == 0:
        raise ValueError("Cannot vectorize empty sequence.")
    D = np.zeros((n, n_features))
    for i in range(n):
        for f, v in rows[i]:
            if v == 0:
                continue
            idx, h = _hash_index(f, n_features)
            val = v
            if alternate_sign:
                val *= float((1 if h >= 0 else 0) * 2 - 1)
            D[i, idx] += val
    return sp.csr_array(D)

class FeatureHasher:
    def __repr__(self) -> str:
        return _estimator_repr(self)

    n_features: int
    input_type: str
    alternate_sign: bool

    def __init__(self, n_features: int = 1048576, *, input_type: str = "dict", dtype: type = float,
                 alternate_sign: bool = True):
        if input_type not in ("dict", "pair", "string"):
            raise ValueError(f"The 'input_type' parameter of FeatureHasher must be a str among {{'dict', 'pair', 'string'}}. Got {input_type!r} instead.")
        self.n_features = n_features
        self.input_type = input_type
        self.alternate_sign = alternate_sign

    def fit(self, X = None, y = None):
        return self

    def transform(self, raw_X):
        rows = List[List[Tuple[str, float]]]()
        for x in raw_X:
            r = List[Tuple[str, float]]()
            if hasattr(x, "items"):
                if self.input_type != "dict":
                    raise ValueError("input_type='dict' expects mappings")
                for f, v in x.items():
                    if isinstance(v, str):
                        r.append((f"{f}={v}", 1.0))
                    else:
                        r.append((str(f), float(v)))
            elif isinstance(x, str):
                raise ValueError("Samples can not be a single string. The input must be an iterable over iterables of strings.")
            else:
                for e in x:
                    if isinstance(e, str):
                        r.append((e, 1.0))
                    else:
                        f, v = e
                        if isinstance(v, str):
                            r.append((f"{f}={v}", 1.0))
                        else:
                            r.append((str(f), float(v)))
            rows.append(r)
        return _hash_rows(rows, self.n_features, self.alternate_sign)

    def fit_transform(self, raw_X, y = None):
        return self.transform(raw_X)
