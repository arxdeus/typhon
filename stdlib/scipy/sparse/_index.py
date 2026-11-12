# Copyright (C) 2026 the Typhon authors
# Ported from scipy 1.18.1 (BSD-3-Clause)

# Index helpers shared by the LIL and DOK containers (scipy _index.py):
# normalisation of int / slice / sequence keys and broadcasting of the
# assigned value.

import numpy as np

def _idx1(k, n: int) -> List[int]:
    if isinstance(k, int):
        kk = k + n if k < 0 else k
        if kk < 0 or kk >= n:
            raise IndexError(f"index ({k}) out of range")
        return [kk]
    elif isinstance(k, slice):
        return list(range(n)[k])
    else:
        out = List[int]()
        for v in k:
            vv = int(v)
            if vv < 0:
                vv += n
            if vv < 0 or vv >= n:
                raise IndexError(f"index ({v}) out of range")
            out.append(vv)
        return out

def _is_seq(k) -> Literal[bool]:
    return not (isinstance(k, int) or isinstance(k, slice))

def _key2(key, shape: Tuple[int, int]):
    # -> (rows, cols, pairwise) ; pairwise: both indices are sequences
    # (numpy fancy indexing pairs them up instead of the outer product)
    m, n = shape
    if isinstance(key, Tuple):
        rows = _idx1(key[0], m)
        cols = _idx1(key[1], n)
        if _is_seq(key[0]) and _is_seq(key[1]):
            if len(rows) != len(cols):
                if len(rows) == 1:
                    rows = rows * len(cols)
                elif len(cols) == 1:
                    cols = cols * len(rows)
                else:
                    raise IndexError("shape mismatch: indexing arrays could not be broadcast together")
            return rows, cols, True
        return rows, cols, False
    else:
        return _idx1(key, m), list(range(n)), False

def _bcast_value(x, nr: int, nc: int, pairwise: bool, T: type):
    # value array V with V[a][b] for the outer case, V[0][q] for pairwise
    if hasattr(x, "_spfmt"):
        return _bcast_value(x.toarray(), nr, nc, pairwise, T)
    elif isinstance(x, np.ndarray) or isinstance(x, List):
        X = np.asarray(x).astype(T)
        if pairwise:
            Y = X.ravel()
            out = np.empty((1, nr), dtype=T)
            for q in range(nr):
                out[0, q] = Y[0 if len(Y) == 1 else q]
            return out
        if X.ndim == 1:
            if len(X) != nc and len(X) != 1 and not (nc == 1 and len(X) == nr):
                raise ValueError("shape mismatch in assignment")
            out = np.empty((nr, nc), dtype=T)
            for a in range(nr):
                for b in range(nc):
                    if len(X) == 1:
                        out[a, b] = X[0]
                    elif nc == 1 and len(X) == nr and nr != 1:
                        out[a, b] = X[a]
                    else:
                        out[a, b] = X[b]
            return out
        elif X.ndim == 2:
            r0 = X.shape[0]
            c0 = X.shape[1]
            if (r0 != nr and r0 != 1) or (c0 != nc and c0 != 1):
                raise ValueError("shape mismatch in assignment")
            out = np.empty((nr, nc), dtype=T)
            for a in range(nr):
                for b in range(nc):
                    out[a, b] = X[a if r0 != 1 else 0, b if c0 != 1 else 0]
            return out
        else:
            compile_error("assigned value must be at most 2-D")
    else:
        if pairwise:
            return np.full((1, nr), T(x), dtype=T)
        return np.full((nr, nc), T(x), dtype=T)
