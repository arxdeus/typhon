# Copyright (C) 2026 the Typhon authors
# Ported from scipy 1.18.1 (BSD-3-Clause)

# Sparse containers. One generic class per storage scheme:
#   _compressed[T, F, M]  F = 0 (CSR) or 1 (CSC), M = 0 (sparray) / 1 (spmatrix)
#   _coo[T, M]
# The public names csr_array, csr_matrix, ... (in _construct) are factory
# functions building these. Index arrays are always int64. A CSC container
# stores the CSR arrays of its transpose, exactly as scipy does, so every
# kernel works on (major, minor) axes.

import numpy as np
from ._sparsetools import *
from ._sparsetools import _apply

def _dtype_name(T: type) -> str:
    if T is float:
        return "float64"
    elif T is float32:
        return "float32"
    elif T is int:
        return "int64"
    elif T is complex:
        return "complex128"
    elif T is bool:
        return "bool"
    elif T is i32:
        return "int32"
    else:
        return "object"

def _is_cplx(T: type) -> Literal[bool]:
    return T is complex or T is complex64

def _up(T: type, U: type):
    # numpy-style promotion for the scalar types used here
    if T is U:
        return T()
    elif _is_cplx(T) or _is_cplx(U):
        return complex()
    elif T is float or U is float:
        return float()
    elif T is bool:
        return U()
    elif U is bool:
        return T()
    elif T is float32 or U is float32:
        return float()
    else:
        return int()

def _sum_t(T: type):
    if T is bool:
        return int()
    else:
        return T()

def _swap2(F: Static[int], t):
    if F == 0:
        return t
    else:
        return (t[1], t[0])

class _compressed[T, F: Static[int], M: Static[int]]:
    data: np.ndarray[T, 1]
    indices: np.ndarray[int, 1]
    indptr: np.ndarray[int, 1]
    _shape: Tuple[int, int]

    def __init__(self, data: np.ndarray[T, 1], indices: np.ndarray[int, 1],
                 indptr: np.ndarray[int, 1], shape: Tuple[int, int]):
        self.data = data
        self.indices = indices
        self.indptr = indptr
        self._shape = shape

    # --- basic attributes ---
    @property
    def shape(self):
        return self._shape

    @property
    def ndim(self):
        return 2

    @property
    def nnz(self) -> int:
        return self.indptr[len(self.indptr) - 1]

    @property
    def size(self) -> int:
        return self.nnz

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def format(self) -> str:
        return "csr" if F == 0 else "csc"

    @property
    def _is_matrix(self) -> bool:
        return M == 1

    def _mstatic(self) -> Literal[bool]:
        return M == 1

    def _major_minor(self):
        return _swap2(F, self._shape)

    def _new(self, data: np.ndarray[U, 1], indices: np.ndarray[int, 1],
             indptr: np.ndarray[int, 1], U: type):
        return _compressed[U, F, M](data, indices, indptr, self._shape)

    def _with_data(self, data):
        return _compressed[data.dtype, F, M](
            data, self.indices.copy(), self.indptr.copy(), self._shape)

    def getnnz(self) -> int:
        return self.nnz

    def count_nonzero(self) -> int:
        c = 0
        for i in range(self.nnz):
            if self.data[i] != T():
                c += 1
        return c

    def copy(self):
        return _compressed[T, F, M](self.data.copy(), self.indices.copy(),
                                    self.indptr.copy(), self._shape)

    def astype(self, dt: type, copy: bool = True):
        return _compressed[dt, F, M](self.data.astype(dt), self.indices.copy(),
                                     self.indptr.copy(), self._shape)

    def __repr__(self):
        kind = "Compressed Sparse Row" if F == 0 else "Compressed Sparse Column"
        what = "matrix" if M == 1 else "array"
        return (f"<{kind} sparse {what} of dtype '{_dtype_name(T)}'\n\twith "
                f"{self.nnz} stored elements and shape {self._shape}>")

    def __str__(self):
        c = self.tocoo()
        s = self.__repr__() + "\n  Coords\tValues"
        for k in range(len(c.data)):
            s += f"\n  ({c.row[k]}, {c.col[k]})\t{c.data[k]}"
        return s

    def __len__(self):
        raise TypeError("sparse array length is ambiguous; use getnnz() or shape[0]")
        return 0

    # --- canonical format ---
    @property
    def has_sorted_indices(self) -> bool:
        return csr_has_sorted_indices(len(self.indptr) - 1, self.indptr, self.indices)

    @property
    def has_canonical_format(self) -> bool:
        return csr_has_canonical_format(len(self.indptr) - 1, self.indptr, self.indices)

    def sort_indices(self):
        csr_sort_indices(len(self.indptr) - 1, self.indptr, self.indices, self.data)

    def sorted_indices(self):
        A = self.copy()
        A.sort_indices()
        return A

    def _prune(self):
        nnz = self.nnz
        if len(self.indices) != nnz:
            self.indices = self.indices[:nnz].copy()
        if len(self.data) != nnz:
            self.data = self.data[:nnz].copy()

    def sum_duplicates(self):
        if self.has_canonical_format:
            return
        self.sort_indices()
        csr_sum_duplicates(len(self.indptr) - 1, self.indptr, self.indices, self.data)
        self._prune()

    def eliminate_zeros(self):
        csr_eliminate_zeros(len(self.indptr) - 1, self.indptr, self.indices, self.data)
        self._prune()

    def prune(self):
        self._prune()

    # --- conversions ---
    def toarray(self):
        Mj, Nm = self._major_minor()
        D = csr_todense(Mj, Nm, self.indptr, self.indices, self.data)
        if F == 0:
            return D
        else:
            return np.ascontiguousarray(D.T)

    def todense(self):
        return self.toarray()

    def tocsr(self, copy: bool = False):
        if F == 0:
            return self.copy() if copy else self
        else:
            m, n = self._shape
            Bp, Bj, Bx = csr_tocsc(n, m, self.indptr, self.indices, self.data)
            return _compressed[T, 0, M](Bx, Bj, Bp, self._shape)

    def tocsc(self, copy: bool = False):
        if F == 1:
            return self.copy() if copy else self
        else:
            m, n = self._shape
            Bp, Bj, Bx = csr_tocsc(m, n, self.indptr, self.indices, self.data)
            return _compressed[T, 1, M](Bx, Bj, Bp, self._shape)

    def tocoo(self, copy: bool = False):
        Mj, Nm = self._major_minor()
        major = expandptr(Mj, self.indptr)
        nnz = self.nnz
        minor = self.indices[:nnz].copy()
        if F == 0:
            return _coo[T, M](self.data[:nnz].copy(), major, minor, self._shape)
        else:
            return _coo[T, M](self.data[:nnz].copy(), minor, major, self._shape)

    def asformat(self, fmt: Literal[str]):
        if fmt == "csr":
            return self.tocsr()
        elif fmt == "csc":
            return self.tocsc()
        elif fmt == "coo":
            return self.tocoo()
        else:
            compile_error("unsupported sparse format")

    def _as_array(self):
        return _compressed[T, F, 0](self.data, self.indices, self.indptr, self._shape)

    def _as_matrix(self):
        return _compressed[T, F, 1](self.data, self.indices, self.indptr, self._shape)

    def transpose(self, axes = None, copy: bool = False):
        return _compressed[T, 1 - F, M](self.data.copy(), self.indices.copy(),
                                        self.indptr.copy(),
                                        (self._shape[1], self._shape[0]))

    @property
    def T(self):
        return self.transpose()

    def conj(self, copy: bool = True):
        if _is_cplx(T):
            return self._with_data(self.data.conj())
        else:
            return self.copy()

    def conjugate(self, copy: bool = True):
        return self.conj()

    @property
    def H(self):
        return self.conj().transpose()

    @property
    def real(self):
        if _is_cplx(T):
            return self._with_data(self.data.real.copy())
        else:
            return self.copy()

    @property
    def imag(self):
        if _is_cplx(T):
            return self._with_data(self.data.imag.copy())
        else:
            return self._with_data(np.zeros(len(self.data), dtype=T))

    # --- conversion of the other operand to this format ---
    def _same_fmt(self, other):
        if isinstance(other, _coo):
            o = other.tocsr() if F == 0 else other.tocsc()
            return o
        elif isinstance(other, _compressed):
            return other.tocsr() if F == 0 else other.tocsc()
        else:
            compile_error("expected a sparse operand")

    # --- unary ---
    def __neg__(self):
        return self._with_data(-self.data)

    def __abs__(self):
        return self._with_data(np.abs(self.data))

    def __pos__(self):
        return self.copy()

    def power(self, n, dtype: type = NoneType):
        return self._with_data(self.data ** n)

    def sqrt(self):
        return self._with_data(np.sqrt(self.data))

    def __round__(self, ndigits: int = 0):
        return self._with_data(np.round(self.data, ndigits))

    # --- binary sparse-sparse kernels (in major/minor space) ---
    def _binopt(self, other, op: Literal[int]):
        o = self._same_fmt(other)
        R = type(_up(T, o.data.dtype))
        if self._shape != o.shape:
            raise ValueError("inconsistent shapes")
        Mj, Nm = self._major_minor()
        Cp, Cj, Cx = csr_binop_csr(Mj, Nm, self.indptr, self.indices,
                                   self.data.astype(R), o.indptr, o.indices,
                                   o.data.astype(R), op)
        return _compressed[R, F, M](Cx, Cj, Cp, self._shape)

    def _scalar(self, s, op: Literal[int]):
        R = type(_up(T, type(s)))
        d = self.data.astype(R)
        v = R(s)
        for i in range(len(d)):
            d[i] = _apply(op, d[i], v)
        return _compressed[R, F, M](d, self.indices.copy(), self.indptr.copy(),
                                    self._shape)

    def _dense_add(self, other, sign: Literal[int]):
        D = np.asarray(other)
        A = self.toarray()
        if sign == 0:
            return A + D
        else:
            return A - D

    def __add__(self, other):
        if isinstance(other, _compressed) or isinstance(other, _coo):
            return self._binopt(other, 0)
        elif isinstance(other, np.ndarray) or isinstance(other, List):
            return self._dense_add(other, 0)
        else:
            if other == 0:
                return self.copy()
            raise NotImplementedError("adding a nonzero scalar to a sparse array is not supported")
            return self.copy()

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, _compressed) or isinstance(other, _coo):
            return self._binopt(other, 1)
        elif isinstance(other, np.ndarray) or isinstance(other, List):
            return self._dense_add(other, 1)
        else:
            if other == 0:
                return self.copy()
            raise NotImplementedError("subtracting a nonzero scalar from a sparse array is not supported")
            return self.copy()

    def __rsub__(self, other):
        if isinstance(other, np.ndarray) or isinstance(other, List):
            return np.asarray(other) - self.toarray()
        else:
            return (-self).__add__(other)

    def multiply(self, other):
        if isinstance(other, _compressed) or isinstance(other, _coo):
            return self._binopt(other, 2)
        elif isinstance(other, np.ndarray) or isinstance(other, List):
            D = np.asarray(other)
            R = type(_up(T, D.dtype))
            c = self.tocoo()
            c.sum_duplicates()
            m, n = self._shape
            if D.ndim == 2 and D.shape[0] == m and D.shape[1] == n:
                d = np.empty(len(c.data), dtype=R)
                for k in range(len(d)):
                    d[k] = R(c.data[k]) * R(D[c.row[k], c.col[k]])
                return _coo[R, M](d, c.row, c.col, self._shape)
            else:
                return self.toarray() * D
        else:
            return self._scalar(other, 2)

    def maximum(self, other):
        return self._binopt(other, 4)

    def minimum(self, other):
        return self._binopt(other, 5)

    def __mul__(self, other):
        if isinstance(other, _compressed) or isinstance(other, _coo):
            if M == 1:
                return self.__matmul__(other)
            else:
                return self._binopt(other, 2)
        elif isinstance(other, np.ndarray) or isinstance(other, List):
            if M == 1:
                return self.__matmul__(other)
            else:
                return self.multiply(other)
        else:
            return self._scalar(other, 2)

    def __rmul__(self, other):
        if isinstance(other, np.ndarray) or isinstance(other, List):
            if M == 1:
                return self.__rmatmul__(other)
            else:
                return self.multiply(other)
        else:
            return self._scalar(other, 2)

    def __truediv__(self, other):
        if isinstance(other, _compressed) or isinstance(other, _coo):
            return self.toarray() / other.toarray()
        elif isinstance(other, np.ndarray) or isinstance(other, List):
            return self.toarray() / np.asarray(other)
        else:
            R = type(_up(type(_up(T, type(other))), float))
            return self.astype(R)._scalar(R(other), 3)

    # --- products ---
    def _matvec(self, x: np.ndarray[X, 1], X: type):
        R = type(_up(T, X))
        m, n = self._shape
        if len(x) != n:
            raise ValueError("dimension mismatch")
        y = np.zeros(m, dtype=R)
        xr = x.astype(R)
        d = self.data.astype(R)
        if F == 0:
            csr_matvec(m, self.indptr, self.indices, d, xr, y)
        else:
            csc_matvec(n, self.indptr, self.indices, d, xr, y)
        return y

    def _matmat(self, X: np.ndarray[XT, 2], XT: type):
        R = type(_up(T, XT))
        m, n = self._shape
        if X.shape[0] != n:
            raise ValueError("dimension mismatch")
        k = X.shape[1]
        Y = np.zeros((m, k), dtype=R)
        Xr = X.astype(R)
        d = self.data.astype(R)
        if F == 0:
            csr_matvecs(m, k, self.indptr, self.indices, d, Xr, Y)
        else:
            csc_matvecs(n, k, self.indptr, self.indices, d, Xr, Y)
        return Y

    def _matmul_sparse(self, other):
        o = self._same_fmt(other)
        R = type(_up(T, o.data.dtype))
        m, k1 = self._shape
        k2, n = o.shape
        if k1 != k2:
            raise ValueError("dimension mismatch")
        if F == 0:
            Cp, Cj, Cx = csr_matmat(m, n, self.indptr, self.indices,
                                    self.data.astype(R), o.indptr, o.indices,
                                    o.data.astype(R))
        else:
            Cp, Cj, Cx = csr_matmat(n, m, o.indptr, o.indices, o.data.astype(R),
                                    self.indptr, self.indices, self.data.astype(R))
        return _compressed[R, F, M](Cx, Cj, Cp, (m, n))

    def __matmul__(self, other):
        if isinstance(other, _compressed) or isinstance(other, _coo):
            return self._matmul_sparse(other)
        else:
            x = np.asarray(other)
            if x.ndim == 1:
                return self._matvec(x)
            else:
                return self._matmat(x)

    def __rmatmul__(self, other):
        x = np.asarray(other)
        if x.ndim == 1:
            return self.transpose()._matvec(x)
        else:
            return self.transpose()._matmat(np.ascontiguousarray(x.T)).T

    def dot(self, other):
        return self.__matmul__(other)

    # --- reductions ---
    def sum(self, axis = None):
        S = type(_sum_t(T))
        m, n = self._shape
        if axis is None:
            s = S()
            for i in range(self.nnz):
                s += S(self.data[i])
            return s
        else:
            ax = axis if axis >= 0 else axis + 2
            c = self.tocoo()
            if ax == 0:
                r = np.zeros(n, dtype=S)
                for k in range(len(c.data)):
                    r[c.col[k]] += S(c.data[k])
            else:
                r = np.zeros(m, dtype=S)
                for k in range(len(c.data)):
                    r[c.row[k]] += S(c.data[k])
            return r

    def mean(self, axis = None):
        m, n = self._shape
        if axis is None:
            return self.sum() / float(m * n)
        else:
            ax = axis if axis >= 0 else axis + 2
            return self.sum(axis) / float(m if ax == 0 else n)

    def _maxmin(self, is_max: bool):
        A = self.copy()
        A.sum_duplicates()
        m, n = self._shape
        if m * n == 0:
            raise ValueError("zero-size array to reduction operation")
        z = T()
        have = False
        best = z
        for i in range(A.nnz):
            v = A.data[i]
            if not have or (v > best if is_max else v < best):
                best = v
                have = True
        if A.nnz < m * n:
            if not have or (z > best if is_max else z < best):
                best = z
        return best

    def max(self):
        return self._maxmin(True)

    def min(self):
        return self._maxmin(False)

    def diagonal(self, k: int = 0):
        Mj, Nm = self._major_minor()
        kk = k if F == 0 else -k
        if kk <= -Mj or kk >= Nm:
            return np.empty(0, dtype=T)
        return csr_diagonal(kk, Mj, Nm, self.indptr, self.indices, self.data)

    def trace(self, offset: int = 0):
        return self.diagonal(offset).sum()

    # --- indexing ---
    def _get(self, i: int, j: int) -> T:
        m, n = self._shape
        if i < 0:
            i += m
        if j < 0:
            j += n
        if not (0 <= i < m and 0 <= j < n):
            raise IndexError("index out of bounds")
        a, b = _swap2(F, (i, j))
        s = T()
        for jj in range(self.indptr[a], self.indptr[a + 1]):
            if self.indices[jj] == b:
                s += self.data[jj]
        return s

    def _select(self, rows: List[int], cols: List[int]):
        # submatrix A[rows][:, cols] in the format of self
        c = self.tocsr()
        m, n = self._shape
        cmap = List[List[int]]()
        for _ in range(n):
            cmap.append(List[int]())
        for q in range(len(cols)):
            cmap[cols[q]].append(q)
        Bp = np.zeros(len(rows) + 1, dtype=int)
        Bj = List[int]()
        Bx = List[T]()
        for r in range(len(rows)):
            i = rows[r]
            for jj in range(c.indptr[i], c.indptr[i + 1]):
                for q in cmap[c.indices[jj]]:
                    Bj.append(q)
                    Bx.append(c.data[jj])
            Bp[r + 1] = len(Bj)
        res = _compressed[T, 0, M](np.array(Bx, dtype=T), np.array(Bj, dtype=int),
                                   Bp, (len(rows), len(cols)))
        if F == 0:
            return res
        else:
            return res.tocsc()

    def _idx_list(self, k, n: int):
        if isinstance(k, int):
            return [k + n if k < 0 else k]
        elif isinstance(k, slice):
            return list(range(n)[k])
        else:
            out = List[int]()
            for v in k:
                out.append(v + n if v < 0 else v)
            return out

    def __getitem__(self, key):
        m, n = self._shape
        if isinstance(key, Tuple):
            if isinstance(key[0], int) and isinstance(key[1], int):
                return self._get(key[0], key[1])
            else:
                return self._select(self._idx_list(key[0], m), self._idx_list(key[1], n))
        else:
            return self._select(self._idx_list(key, m), list(range(n)))

    def getrow(self, i: int):
        return self._select([i + self._shape[0] if i < 0 else i], list(range(self._shape[1])))

    def getcol(self, j: int):
        return self._select(list(range(self._shape[0])), [j + self._shape[1] if j < 0 else j])

    def __setitem__(self, key: Tuple[int, int], v):
        # structural insertion is supported (slow path, as in scipy)
        i, j = key
        m, n = self._shape
        if i < 0:
            i += m
        if j < 0:
            j += n
        a, b = _swap2(F, (i, j))
        for jj in range(self.indptr[a], self.indptr[a + 1]):
            if self.indices[jj] == b:
                self.data[jj] = T(v)
                return
        pos = self.indptr[a + 1]
        self.indices = np.concatenate((self.indices[:pos], np.array([b]), self.indices[pos:]))
        self.data = np.concatenate((self.data[:pos], np.array([T(v)]), self.data[pos:]))
        for r in range(a + 1, len(self.indptr)):
            self.indptr[r] += 1

    def nonzero(self):
        c = self.tocoo()
        rs = List[int]()
        cs = List[int]()
        for k in range(len(c.data)):
            if c.data[k] != T():
                rs.append(c.row[k])
                cs.append(c.col[k])
        return np.array(rs, dtype=int), np.array(cs, dtype=int)

    def setdiag(self, values, k: int = 0):
        c = self.tocoo()
        c.setdiag(values, k)
        A = c.tocsr() if F == 0 else c.tocsc()
        self.data = A.data
        self.indices = A.indices
        self.indptr = A.indptr


class _coo[T, M: Static[int]]:
    data: np.ndarray[T, 1]
    row: np.ndarray[int, 1]
    col: np.ndarray[int, 1]
    _shape: Tuple[int, int]
    _canonical: bool

    def __init__(self, data: np.ndarray[T, 1], row: np.ndarray[int, 1],
                 col: np.ndarray[int, 1], shape: Tuple[int, int],
                 canonical: bool = False):
        if len(row) != len(col) or len(row) != len(data):
            raise ValueError("row, column, and data array must all be the same length")
        for k in range(len(row)):
            if row[k] < 0 or row[k] >= shape[0]:
                raise ValueError("row index exceeds matrix dimensions")
            if col[k] < 0 or col[k] >= shape[1]:
                raise ValueError("column index exceeds matrix dimensions")
        self.data = data
        self.row = row
        self.col = col
        self._shape = shape
        self._canonical = canonical

    @property
    def shape(self):
        return self._shape

    @property
    def ndim(self):
        return 2

    @property
    def nnz(self) -> int:
        return len(self.data)

    @property
    def size(self) -> int:
        return self.nnz

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def format(self) -> str:
        return "coo"

    @property
    def _is_matrix(self) -> bool:
        return M == 1

    def _mstatic(self) -> Literal[bool]:
        return M == 1

    @property
    def coords(self):
        return (self.row, self.col)

    @property
    def has_canonical_format(self) -> bool:
        return self._canonical

    def getnnz(self) -> int:
        return self.nnz

    def count_nonzero(self) -> int:
        A = self.copy()
        A.sum_duplicates()
        c = 0
        for v in A.data:
            if v != T():
                c += 1
        return c

    def copy(self):
        return _coo[T, M](self.data.copy(), self.row.copy(), self.col.copy(),
                          self._shape, self._canonical)

    def astype(self, dt: type, copy: bool = True):
        return _coo[dt, M](self.data.astype(dt), self.row.copy(), self.col.copy(),
                           self._shape, self._canonical)

    def _with_data(self, d):
        return _coo[d.dtype, M](d, self.row.copy(), self.col.copy(), self._shape,
                                self._canonical)

    def __repr__(self):
        what = "matrix" if M == 1 else "array"
        return (f"<COOrdinate sparse {what} of dtype '{_dtype_name(T)}'\n\twith "
                f"{self.nnz} stored elements and shape {self._shape}>")

    def __str__(self):
        s = self.__repr__() + "\n  Coords\tValues"
        for k in range(len(self.data)):
            s += f"\n  ({self.row[k]}, {self.col[k]})\t{self.data[k]}"
        return s

    def sum_duplicates(self):
        if self._canonical:
            return
        n = len(self.data)
        if n == 0:
            self._canonical = True
            return
        order = np.lexsort((self.col, self.row))
        r = self.row[order]
        c = self.col[order]
        d = self.data[order]
        nr = List[int]()
        nc = List[int]()
        nd = List[T]()
        for k in range(n):
            if k > 0 and r[k] == r[k - 1] and c[k] == c[k - 1]:
                nd[len(nd) - 1] += d[k]
            else:
                nr.append(r[k])
                nc.append(c[k])
                nd.append(d[k])
        self.row = np.array(nr, dtype=int)
        self.col = np.array(nc, dtype=int)
        self.data = np.array(nd, dtype=T)
        self._canonical = True

    def eliminate_zeros(self):
        nr = List[int]()
        nc = List[int]()
        nd = List[T]()
        for k in range(len(self.data)):
            if self.data[k] != T():
                nr.append(self.row[k])
                nc.append(self.col[k])
                nd.append(self.data[k])
        self.row = np.array(nr, dtype=int)
        self.col = np.array(nc, dtype=int)
        self.data = np.array(nd, dtype=T)

    def toarray(self):
        B = np.zeros(self._shape, dtype=T)
        for k in range(len(self.data)):
            B[self.row[k], self.col[k]] += self.data[k]
        return B

    def todense(self):
        return self.toarray()

    def tocsr(self, copy: bool = False):
        m, n = self._shape
        Bp, Bj, Bx = coo_tocsr(m, n, self.row, self.col, self.data)
        A = _compressed[T, 0, M](Bx, Bj, Bp, self._shape)
        if not self._canonical:
            A.sum_duplicates()
        return A

    def tocsc(self, copy: bool = False):
        m, n = self._shape
        Bp, Bj, Bx = coo_tocsr(n, m, self.col, self.row, self.data)
        A = _compressed[T, 1, M](Bx, Bj, Bp, self._shape)
        if not self._canonical:
            A.sum_duplicates()
        return A

    def tocoo(self, copy: bool = False):
        return self.copy() if copy else self

    def asformat(self, fmt: Literal[str]):
        if fmt == "csr":
            return self.tocsr()
        elif fmt == "csc":
            return self.tocsc()
        elif fmt == "coo":
            return self
        else:
            compile_error("unsupported sparse format")

    def _as_array(self):
        return _coo[T, 0](self.data, self.row, self.col, self._shape, self._canonical)

    def _as_matrix(self):
        return _coo[T, 1](self.data, self.row, self.col, self._shape, self._canonical)

    def transpose(self, axes = None, copy: bool = False):
        return _coo[T, M](self.data.copy(), self.col.copy(), self.row.copy(),
                          (self._shape[1], self._shape[0]))

    @property
    def T(self):
        return self.transpose()

    def conj(self, copy: bool = True):
        if _is_cplx(T):
            return self._with_data(self.data.conj())
        else:
            return self.copy()

    def conjugate(self, copy: bool = True):
        return self.conj()

    @property
    def H(self):
        return self.conj().transpose()

    def __neg__(self):
        return self._with_data(-self.data)

    def __abs__(self):
        return self._with_data(np.abs(self.data))

    def power(self, n, dtype: type = NoneType):
        return self._with_data(self.data ** n)

    def diagonal(self, k: int = 0):
        return self.tocsr().diagonal(k)

    def trace(self, offset: int = 0):
        return self.diagonal(offset).sum()

    def setdiag(self, values, k: int = 0):
        m, n = self._shape
        if k < 0:
            i0 = -k
            j0 = 0
        else:
            i0 = 0
            j0 = k
        L = max(0, min(m - i0, n - j0))
        if isinstance(values, np.ndarray) or isinstance(values, List):
            vals = np.asarray(values).astype(T)
            L = min(L, len(vals))
        else:
            vals = np.full(max(L, 0), T(values), dtype=T)
        nr = List[int]()
        nc = List[int]()
        nd = List[T]()
        for q in range(len(self.data)):
            d = self.col[q] - self.row[q]
            t = self.row[q] - i0
            if d == k and 0 <= t < L:
                continue
            nr.append(self.row[q])
            nc.append(self.col[q])
            nd.append(self.data[q])
        for t in range(L):
            nr.append(i0 + t)
            nc.append(j0 + t)
            nd.append(vals[t])
        self.row = np.array(nr, dtype=int)
        self.col = np.array(nc, dtype=int)
        self.data = np.array(nd, dtype=T)
        self._canonical = False

    def nonzero(self):
        A = self.copy()
        A.sum_duplicates()
        rs = List[int]()
        cs = List[int]()
        for k in range(len(A.data)):
            if A.data[k] != T():
                rs.append(A.row[k])
                cs.append(A.col[k])
        return np.array(rs, dtype=int), np.array(cs, dtype=int)

    # arithmetic is delegated to CSR (scipy does the same)
    def __add__(self, o):
        return self.tocsr().__add__(o)

    def __radd__(self, o):
        return self.tocsr().__radd__(o)

    def __sub__(self, o):
        return self.tocsr().__sub__(o)

    def __rsub__(self, o):
        return self.tocsr().__rsub__(o)

    def __mul__(self, o):
        if isinstance(o, _compressed) or isinstance(o, _coo) or isinstance(o, np.ndarray) or isinstance(o, List):
            return self.tocsr().__mul__(o)
        else:
            return self._with_data(self.data * o)

    def __rmul__(self, o):
        if isinstance(o, np.ndarray) or isinstance(o, List):
            return self.tocsr().__rmul__(o)
        else:
            return self._with_data(self.data * o)

    def __truediv__(self, o):
        return self.tocsr().__truediv__(o)

    def multiply(self, o):
        return self.tocsr().multiply(o)

    def maximum(self, o):
        return self.tocsr().maximum(o)

    def minimum(self, o):
        return self.tocsr().minimum(o)

    def __matmul__(self, o):
        return self.tocsr().__matmul__(o)

    def __rmatmul__(self, o):
        return self.tocsr().__rmatmul__(o)

    def dot(self, o):
        return self.tocsr().__matmul__(o)

    def sum(self, axis = None):
        return self.tocsr().sum(axis)

    def mean(self, axis = None):
        return self.tocsr().mean(axis)

    def max(self):
        return self.tocsr().max()

    def min(self):
        return self.tocsr().min()

    def __getitem__(self, key):
        return self.tocsr().__getitem__(key)

    def getrow(self, i: int):
        return self.tocsr().getrow(i)

    def getcol(self, j: int):
        return self.tocsr().getcol(j)
