# Copyright (C) 2026 the Typhon authors
# Ported from scipy 1.18.1 (BSD-3-Clause)

# Public constructors: csr/csc/coo array & matrix factories and the
# scipy.sparse._construct functions (eye, diags, kron, hstack, ...).
# DIA/BSR/LIL/DOK formats are not provided: functions whose scipy default is
# a DIA/BSR result (eye, diags, spdiags, kron with format=None) return CSR,
# which holds exactly the entries scipy's dia->csr conversion produces.

import numpy as np
from ._base import _compressed, _coo, _up, _is_cplx
from ._sparsetools import *

def issparse(x) -> bool:
    return isinstance(x, _compressed) or isinstance(x, _coo)

def isspmatrix(x) -> bool:
    if isinstance(x, _compressed) or isinstance(x, _coo):
        return x._is_matrix
    return False

def _is_shape(a) -> Literal[bool]:
    if isinstance(a, Tuple):
        if static.len(a) == 2:
            return isinstance(a[0], int) and isinstance(a[1], int)
    return False

def _idx(a):
    return np.asarray(a).astype(int)

def _data_as(a, dtype: type):
    x = np.asarray(a)
    if dtype is NoneType:
        return x.ravel().copy()
    else:
        return x.ravel().astype(dtype)

def _cast_sparse(a, dtype: type):
    if dtype is NoneType:
        return a
    else:
        return a.astype(dtype)

def _make_coo(arg1, shape, dtype: type, M: Static[int]):
    if isinstance(arg1, _compressed) or isinstance(arg1, _coo):
        c = arg1.tocoo(copy=True)
        if M == 1:
            return _cast_sparse(c._as_matrix(), dtype)
        else:
            return _cast_sparse(c._as_array(), dtype)
    elif _is_shape(arg1):
        D = type(float() if dtype is NoneType else dtype())
        return _coo[D, M](np.empty(0, dtype=D), np.empty(0, dtype=int),
                          np.empty(0, dtype=int), (arg1[0], arg1[1]), True)
    elif isinstance(arg1, Tuple):
        data = _data_as(arg1[0], dtype)
        row = _idx(arg1[1][0])
        col = _idx(arg1[1][1])
        if shape is None:
            if len(row) == 0 or len(col) == 0:
                raise ValueError("cannot infer dimensions from zero sized index arrays")
            shp = (int(row.max()) + 1, int(col.max()) + 1)
        else:
            shp = (shape[0], shape[1])
        return _coo[data.dtype, M](data, row, col, shp, False)
    else:
        x = np.asarray(arg1)
        if x.ndim == 1:
            X = x.reshape((1, len(x)))
        else:
            X = x
        if shape is not None:
            if (shape[0], shape[1]) != (X.shape[0], X.shape[1]):
                raise ValueError("inconsistent shapes")
        rs = List[int]()
        cs = List[int]()
        ds = List[X.dtype]()
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                if X[i, j] != X.dtype():
                    rs.append(i)
                    cs.append(j)
                    ds.append(X[i, j])
        d = np.array(ds, dtype=X.dtype)
        dd = d if dtype is NoneType else d.astype(dtype)
        return _coo[dd.dtype, M](dd, np.array(rs, dtype=int), np.array(cs, dtype=int),
                                 (X.shape[0], X.shape[1]), True)

def _make_compressed(arg1, shape, dtype: type, F: Static[int], M: Static[int]):
    if isinstance(arg1, _compressed) or isinstance(arg1, _coo):
        a = arg1.tocsr(copy=True) if F == 0 else arg1.tocsc(copy=True)
        if M == 1:
            return _cast_sparse(a._as_matrix(), dtype)
        else:
            return _cast_sparse(a._as_array(), dtype)
    elif _is_shape(arg1):
        D = type(float() if dtype is NoneType else dtype())
        nmaj = arg1[0] if F == 0 else arg1[1]
        return _compressed[D, F, M](np.empty(0, dtype=D), np.empty(0, dtype=int),
                                    np.zeros(nmaj + 1, dtype=int), (arg1[0], arg1[1]))
    elif isinstance(arg1, Tuple):
        if static.len(arg1) == 3:
            data = _data_as(arg1[0], dtype)
            indices = _idx(arg1[1]).copy()
            indptr = _idx(arg1[2]).copy()
            if shape is None:
                nmaj = len(indptr) - 1
                nmin = int(indices.max()) + 1 if len(indices) > 0 else 0
                shp = (nmaj, nmin) if F == 0 else (nmin, nmaj)
            else:
                shp = (shape[0], shape[1])
            nm = shp[0] if F == 0 else shp[1]
            if len(indptr) != nm + 1:
                raise ValueError(f"index pointer size {len(indptr)} should be {nm + 1}")
            return _compressed[data.dtype, F, M](data, indices, indptr, shp)
        else:
            c = _make_coo(arg1, shape, dtype, M)
            return c.tocsr() if F == 0 else c.tocsc()
    else:
        c = _make_coo(arg1, shape, dtype, M)
        return c.tocsr() if F == 0 else c.tocsc()

def csr_array(arg1, shape = None, dtype: type = NoneType, copy: bool = False):
    return _make_compressed(arg1, shape, dtype, 0, 0)

def csr_matrix(arg1, shape = None, dtype: type = NoneType, copy: bool = False):
    return _make_compressed(arg1, shape, dtype, 0, 1)

def csc_array(arg1, shape = None, dtype: type = NoneType, copy: bool = False):
    return _make_compressed(arg1, shape, dtype, 1, 0)

def csc_matrix(arg1, shape = None, dtype: type = NoneType, copy: bool = False):
    return _make_compressed(arg1, shape, dtype, 1, 1)

def coo_array(arg1, shape = None, dtype: type = NoneType, copy: bool = False):
    return _make_coo(arg1, shape, dtype, 0)

def coo_matrix(arg1, shape = None, dtype: type = NoneType, copy: bool = False):
    return _make_coo(arg1, shape, dtype, 1)

def _asformat(A, format: Literal[str]):
    if format == "":
        return A
    else:
        return A.asformat(format)

# --- diagonal constructors ---------------------------------------------------

def _diags_impl(diagonals, offsets, shape, dtype: type, M: Static[int]):
    # offsets: List[int]; diagonals: List[ndarray]
    if len(diagonals) != len(offsets):
        raise ValueError("Different number of diagonals and offsets.")
    if shape is None:
        m0 = len(diagonals[0]) + abs(offsets[0])
        m, n = m0, m0
    else:
        m, n = shape[0], shape[1]
    K = min(m, n)
    D = type(diagonals[0][0] if dtype is NoneType else dtype())
    # dense band data as in dia_array, then dia_tocsr (sorted offsets per row,
    # explicit zeros dropped; later duplicates of an offset overwrite)
    nd = len(offsets)
    Lmax = 0
    for off in offsets:
        Lmax = max(Lmax, min(m + off, n - off) + max(0, off))
    Lmax = max(0, Lmax)
    band = np.zeros((nd, Lmax), dtype=D)
    for j in range(nd):
        off = offsets[j]
        k = max(0, off)
        length = min(m + off, n - off, K)
        if length < 0:
            raise ValueError(f"Offset {off} (index {j}) out of bounds")
        dj = diagonals[j]
        if len(dj) != length and len(dj) != 1:
            raise ValueError(f"Diagonal length (index {j}: {len(dj)} at offset {off}) does not agree with array size ({m}, {n}).")
        for t in range(length):
            band[j, k + t] = D(dj[t if len(dj) != 1 else 0])
    order = sorted(range(nd), key=lambda q: offsets[q])
    Bp = np.zeros(m + 1, dtype=int)
    Bj = List[int]()
    Bx = List[D]()
    for i in range(m):
        for q in order:
            j = i + offsets[q]
            if 0 <= j < n and j < Lmax:
                v = band[q, j]
                if v != D():
                    Bj.append(j)
                    Bx.append(v)
        Bp[i + 1] = len(Bj)
    return _compressed[D, 0, M](np.array(Bx, dtype=D), np.array(Bj, dtype=int), Bp, (m, n))

def _norm_diags(diagonals, offsets):
    # returns (List[ndarray], List[int])
    if isinstance(offsets, int):
        if isinstance(diagonals, np.ndarray):
            if diagonals.ndim == 1:
                return [diagonals.astype(type(_up(diagonals.dtype, float)))], [offsets]
            else:
                return [diagonals[0].astype(type(_up(diagonals.dtype, float)))], [offsets]
        else:
            x = np.asarray(diagonals)
            return [x.astype(type(_up(x.dtype, float)))], [offsets]
    else:
        offs = [int(o) for o in offsets]
        if isinstance(diagonals, np.ndarray):
            X = diagonals.astype(type(_up(diagonals.dtype, float)))
            return [X[q] for q in range(X.shape[0])], offs
        else:
            T0 = type(np.asarray(diagonals[0]).dtype())
            R = type(_up(T0, float))
            return [np.asarray(d).astype(R) for d in diagonals], offs

def diags_array(diagonals, offsets = 0, shape = None, format: Literal[str] = "",
                dtype: type = NoneType):
    dg, offs = _norm_diags(diagonals, offsets)
    return _asformat(_diags_impl(dg, offs, shape, dtype, 0), format)

def diags(diagonals, offsets = 0, shape = None, format: Literal[str] = "",
          dtype: type = NoneType):
    dg, offs = _norm_diags(diagonals, offsets)
    return _asformat(_diags_impl(dg, offs, shape, dtype, 1), format)

def spdiags(data, diags, m = None, n = None, format: Literal[str] = ""):
    X = np.asarray(data)
    if X.ndim == 1:
        X2 = X.reshape((1, len(X)))
    else:
        X2 = X
    offs = [diags] if isinstance(diags, int) else [int(o) for o in diags]
    if m is None:
        mm = X2.shape[1]
        nn = X2.shape[1]
    elif n is None:
        mm = m[0]
        nn = m[1]
    else:
        mm = m
        nn = n
    D = X2.dtype
    Bp = np.zeros(mm + 1, dtype=int)
    Bj = List[int]()
    Bx = List[D]()
    order = sorted(range(len(offs)), key=lambda q: offs[q])
    for i in range(mm):
        for q in order:
            j = i + offs[q]
            if 0 <= j < nn and j < X2.shape[1]:
                v = X2[q, j]
                if v != D():
                    Bj.append(j)
                    Bx.append(v)
        Bp[i + 1] = len(Bj)
    A = _compressed[D, 0, 1](np.array(Bx, dtype=D), np.array(Bj, dtype=int), Bp, (mm, nn))
    return _asformat(A, format)

def _eye(m: int, n, k: int, dtype: type, format: Literal[str], M: Static[int]):
    nn = m if n is None else n
    L = max(0, min(m + k, nn - k, min(m, nn)))
    D = type(float() if dtype is NoneType else dtype())
    if L < 0:
        L = 0
    i0 = max(0, -k)
    Bp = np.zeros(m + 1, dtype=int)
    for i in range(m):
        Bp[i + 1] = Bp[i] + (1 if i0 <= i < i0 + L else 0)
    Bj = np.array([i0 + t + k for t in range(L)], dtype=int)
    A = _compressed[D, 0, M](np.ones(L, dtype=D), Bj, Bp, (m, nn))
    return _asformat(A, format)

def eye_array(m: int, n = None, k: int = 0, dtype: type = float,
              format: Literal[str] = ""):
    return _eye(m, n, k, dtype, format, 0)

def eye(m: int, n = None, k: int = 0, dtype: type = float, format: Literal[str] = ""):
    return _eye(m, n, k, dtype, format, 1)

def identity(n: int, dtype: type = float, format: Literal[str] = ""):
    return _eye(n, n, 0, dtype, format, 1)

# --- block constructors ----------------------------------------------------

def _any_matrix(blocks) -> bool:
    for b in blocks:
        if isinstance(b, _compressed) or isinstance(b, _coo):
            if not b._is_matrix:
                return False
    return True

def _block_dtype(blocks):
    # common data type of a homogeneous list / tuple of sparse blocks
    if isinstance(blocks, Tuple):
        R = type(blocks[0].data[0] if False else blocks[0].dtype())
        return R()
    else:
        return blocks[0].dtype()

def _stack(blocks, axis: int, format: Literal[str], dtype: type, M: Static[int]):
    # blocks: list of sparse arrays with a common dtype
    D0 = type(blocks[0].dtype())
    D = D0 if dtype is NoneType else dtype
    rs = List[int]()
    cs = List[int]()
    ds = List[D]()
    off = 0
    other = -1
    for b in blocks:
        c = b.tocoo()
        m, n = c.shape
        if axis == 0:
            if other >= 0 and n != other:
                raise ValueError(f"blocks[0,:] has incompatible column dimensions")
            other = n
        else:
            if other >= 0 and m != other:
                raise ValueError(f"blocks[:,0] has incompatible row dimensions")
            other = m
        for k in range(len(c.data)):
            if axis == 0:
                rs.append(c.row[k] + off)
                cs.append(c.col[k])
            else:
                rs.append(c.row[k])
                cs.append(c.col[k] + off)
            ds.append(D(c.data[k]))
        off += m if axis == 0 else n
    shp = (off, other) if axis == 0 else (other, off)
    C = _coo[D, M](np.array(ds, dtype=D), np.array(rs, dtype=int),
                   np.array(cs, dtype=int), shp, False)
    # scipy: vstack of CSR / hstack of CSC concatenates the compressed arrays
    # directly; the COO path below produces identical canonical results.
    if format == "":
        return C.tocsr() if axis == 0 else C.tocsr()
    else:
        return C.asformat(format)

def hstack(blocks, format: Literal[str] = "", dtype: type = NoneType):
    if blocks[0]._mstatic():
        return _stack(blocks, 1, format, dtype, 1)
    else:
        return _stack(blocks, 1, format, dtype, 0)

def vstack(blocks, format: Literal[str] = "", dtype: type = NoneType):
    if blocks[0]._mstatic():
        return _stack(blocks, 0, format, dtype, 1)
    else:
        return _stack(blocks, 0, format, dtype, 0)

def bmat(blocks, format: Literal[str] = "", dtype: type = NoneType):
    rows = [hstack(r, format="coo", dtype=dtype) for r in blocks]
    return vstack(rows, format=format if format != "" else "coo", dtype=dtype)

def block_diag(mats, format: Literal[str] = "", dtype: type = NoneType):
    D0 = type(mats[0].dtype())
    D = D0 if dtype is NoneType else dtype
    rs = List[int]()
    cs = List[int]()
    ds = List[D]()
    r_idx = 0
    c_idx = 0
    for a in mats:
        c = a.tocoo()
        for k in range(len(c.data)):
            rs.append(c.row[k] + r_idx)
            cs.append(c.col[k] + c_idx)
            ds.append(D(c.data[k]))
        r_idx += c.shape[0]
        c_idx += c.shape[1]
    if mats[0]._mstatic():
        C = _coo[D, 1](np.array(ds, dtype=D), np.array(rs, dtype=int),
                       np.array(cs, dtype=int), (r_idx, c_idx), False)
        return _asformat(C, format)
    else:
        C2 = _coo[D, 0](np.array(ds, dtype=D), np.array(rs, dtype=int),
                        np.array(cs, dtype=int), (r_idx, c_idx), False)
        return _asformat(C2, format)

def _kron_impl(A, B, M: Static[int]):
    R = type(_up(type(A.dtype()), type(B.dtype())))
    a = A.tocoo()
    b = B.tocoo()
    shp = (a.shape[0] * b.shape[0], a.shape[1] * b.shape[1])
    nb = b.nnz
    n = a.nnz * nb
    rs = np.empty(n, dtype=int)
    cs = np.empty(n, dtype=int)
    ds = np.empty(n, dtype=R)
    for i in range(a.nnz):
        for k in range(nb):
            q = i * nb + k
            rs[q] = a.row[i] * b.shape[0] + b.row[k]
            cs[q] = a.col[i] * b.shape[1] + b.col[k]
            ds[q] = R(a.data[i]) * R(b.data[k])
    return _coo[R, M](ds, rs, cs, shp, False)

def _as_sparse_like(A):
    if isinstance(A, _compressed) or isinstance(A, _coo):
        return A
    else:
        return coo_matrix(A)

def kron(A, B, format: Literal[str] = ""):
    a = _as_sparse_like(A)
    b = _as_sparse_like(B)
    if a._mstatic() and b._mstatic():
        C = _kron_impl(a, b, 1)
    else:
        C = _kron_impl(a, b, 0)
    # scipy's default here is BSR; return CSR instead
    if format == "":
        return C.tocsr()
    else:
        return C.asformat(format)

def kronsum(A, B, format: Literal[str] = ""):
    a = _as_sparse_like(A)
    b = _as_sparse_like(B)
    if a._mstatic():
        IA = eye(a.shape[0], dtype=type(a.dtype()))
        IB = eye(b.shape[0], dtype=type(b.dtype()))
    else:
        IA = eye_array(a.shape[0], dtype=type(a.dtype()))
        IB = eye_array(b.shape[0], dtype=type(b.dtype()))
    L = kron(IB, a, format="coo")
    Rr = kron(b, IA, format="coo")
    return _asformat((L + Rr), format)

# --- extraction ------------------------------------------------------------

def find(A):
    c = coo_array(A) if not (isinstance(A, _compressed) or isinstance(A, _coo)) else A.tocoo(copy=True)
    c = c.copy()
    c.sum_duplicates()
    T = type(c.dtype())
    rs = List[int]()
    cs = List[int]()
    ds = List[T]()
    for k in range(len(c.data)):
        if c.data[k] != T():
            rs.append(c.row[k])
            cs.append(c.col[k])
            ds.append(c.data[k])
    return np.array(rs, dtype=int), np.array(cs, dtype=int), np.array(ds, dtype=T)

def _tri(A, k: int, lower: bool, format: Literal[str]):
    c = _as_sparse_like(A).tocoo()
    T = type(c.dtype())
    rs = List[int]()
    cs = List[int]()
    ds = List[T]()
    for q in range(len(c.data)):
        keep = (c.row[q] + k >= c.col[q]) if lower else (c.row[q] + k <= c.col[q])
        if keep:
            rs.append(c.row[q])
            cs.append(c.col[q])
            ds.append(c.data[q])
    if c._mstatic():
        C = _coo[T, 1](np.array(ds, dtype=T), np.array(rs, dtype=int),
                       np.array(cs, dtype=int), c.shape, False)
        return _asformat(C, format)
    else:
        C2 = _coo[T, 0](np.array(ds, dtype=T), np.array(rs, dtype=int),
                        np.array(cs, dtype=int), c.shape, False)
        return _asformat(C2, format)

def tril(A, k: int = 0, format: Literal[str] = ""):
    return _tri(A, k, True, format)

def triu(A, k: int = 0, format: Literal[str] = ""):
    return _tri(A, k, False, format)

# --- random ----------------------------------------------------------------

def _random_impl(m: int, n: int, density: float, rng, dtype: type, M: Static[int]):
    # scipy: raveled = rng.choice(m*n, size=k, replace=False), unraveled in
    # Fortran order, then data = rng.uniform(size=k) (np.random.Generator).
    if density < 0 or density > 1:
        raise ValueError("density expected to be 0 <= density <= 1")
    tot = m * n
    k = int(round(density * tot))
    ind = rng.choice(tot, size=k, replace=False)
    vals = rng.uniform(size=k).astype(dtype)
    rs = np.empty(k, dtype=int)
    cs = np.empty(k, dtype=int)
    for q in range(k):
        # unravel_index(ind, (m, n), order='F')
        rs[q] = ind[q] % m
        cs[q] = ind[q] // m
    return _coo[dtype, M](vals, rs, cs, (m, n), False)

def random_array(shape: Tuple[int, int], density: float = 0.01,
                 format: Literal[str] = "coo", dtype: type = float, rng = None):
    g = np.random.default_rng(rng)
    return _asformat(_random_impl(shape[0], shape[1], density, g, dtype, 0), format)

def random(m: int, n: int, density: float = 0.01, format: Literal[str] = "coo",
           dtype: type = float, rng = None):
    g = np.random.default_rng(rng)
    return _asformat(_random_impl(m, n, density, g, dtype, 1), format)

def rand(m: int, n: int, density: float = 0.01, format: Literal[str] = "coo",
         dtype: type = float, rng = None):
    return random(m, n, density, format, dtype, rng)
