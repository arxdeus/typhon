# Copyright (C) 2026 the Typhon authors
# Ported from scipy 1.18.1 (BSD-3-Clause)

# List of Lists format: per row, sorted column indices and values.
# Deviation: indexing that yields more than a scalar returns a 2-D LIL
# container (scipy's sparse arrays return 1-D COO for A[i] / A[:, j]).

import numpy as np
from ._base import _compressed, _coo, _up, _is_cplx, _issparse, _coo_str, \
    _shape2, _shape_args, _asfmt, _dtype_name
from ._index import _idx1, _key2, _bcast_value

def _bisect_left(a: List[int], x: int) -> int:
    lo = 0
    hi = len(a)
    while lo < hi:
        mid = (lo + hi) // 2
        if a[mid] < x:
            lo = mid + 1
        else:
            hi = mid
    return lo

class _lil[T, M: Static[int]]:
    rows: List[List[int]]
    data: List[List[T]]
    _shape: Tuple[int, int]

    def __init__(self, shape: Tuple[int, int]):
        self._shape = shape
        self.rows = [List[int]() for _ in range(shape[0])]
        self.data = [List[T]() for _ in range(shape[0])]

    @staticmethod
    def _spfmt() -> Literal[str]:
        return "lil"

    @property
    def shape(self):
        return self._shape

    @property
    def ndim(self):
        return 2

    @property
    def format(self) -> str:
        return "lil"

    @property
    def dtype(self):
        return T

    @property
    def _is_matrix(self) -> bool:
        return M == 1

    def _mstatic(self) -> Literal[bool]:
        return M == 1

    @property
    def nnz(self) -> int:
        s = 0
        for r in self.rows:
            s += len(r)
        return s

    @property
    def size(self) -> int:
        return self.nnz

    def getnnz(self, axis = None):
        if axis is None:
            return self.nnz
        else:
            ax = axis if axis >= 0 else axis + 2
            if ax == 0:
                out = np.zeros(self._shape[1], dtype=int)
                for r in self.rows:
                    for j in r:
                        out[j] += 1
                return out
            else:
                return np.array([len(r) for r in self.rows], dtype=int)

    def count_nonzero(self) -> int:
        c = 0
        for d in self.data:
            for v in d:
                if v != T():
                    c += 1
        return c

    def copy(self):
        B = _lil[T, M](self._shape)
        for i in range(self._shape[0]):
            B.rows[i] = self.rows[i].copy()
            B.data[i] = self.data[i].copy()
        return B

    def astype(self, dt: type, copy: bool = True):
        B = _lil[dt, M](self._shape)
        for i in range(self._shape[0]):
            B.rows[i] = self.rows[i].copy()
            B.data[i] = [dt(v) for v in self.data[i]]
        return B

    def _as_array(self):
        B = _lil[T, 0](self._shape)
        B.rows = self.rows
        B.data = self.data
        return B

    def _as_matrix(self):
        B = _lil[T, 1](self._shape)
        B.rows = self.rows
        B.data = self.data
        return B

    def __repr__(self):
        what = "matrix" if M == 1 else "array"
        return (f"<List of Lists sparse {what} of dtype '{_dtype_name(T)}'\n\twith "
                f"{self.nnz} stored elements and shape {self._shape}>")

    def __str__(self):
        return _coo_str(self.tocoo(), self.__repr__())

    def __len__(self):
        raise TypeError("sparse array length is ambiguous; use getnnz() or shape[0]")
        return 0

    # --- element access (scipy _csparsetools lil_* helpers) ---
    def _get1(self, i: int, j: int) -> T:
        r = self.rows[i]
        pos = _bisect_left(r, j)
        if pos < len(r) and r[pos] == j:
            return self.data[i][pos]
        return T()

    def _insert(self, i: int, j: int, x: T):
        # lil_insert: zero deletes the entry
        r = self.rows[i]
        d = self.data[i]
        pos = _bisect_left(r, j)
        if x == T():
            if pos < len(r) and r[pos] == j:
                r.pop(pos)
                d.pop(pos)
        else:
            if pos < len(r) and r[pos] == j:
                d[pos] = x
            else:
                r.insert(pos, j)
                d.insert(pos, x)

    def _chk(self, i: int, j: int):
        m, n = self._shape
        if i < -m or i >= m:
            raise IndexError(f"row index ({i}) out of range")
        if j < -n or j >= n:
            raise IndexError(f"column index ({j}) out of range")
        return (i + m if i < 0 else i, j + n if j < 0 else j)

    def __getitem__(self, key):
        if isinstance(key, Tuple):
            if isinstance(key[0], int) and isinstance(key[1], int):
                i, j = self._chk(key[0], key[1])
                return self._get1(i, j)
        rows, cols, pw = _key2(key, self._shape)
        if pw:
            B = _lil[T, M]((1, len(rows)))
            for q in range(len(rows)):
                v = self._get1(rows[q], cols[q])
                if v != T():
                    B.rows[0].append(q)
                    B.data[0].append(v)
            return B
        B = _lil[T, M]((len(rows), len(cols)))
        n = self._shape[1]
        # column map for arbitrary (possibly repeated / unsorted) columns
        cmap = List[List[int]]()
        for _ in range(n):
            cmap.append(List[int]())
        for q in range(len(cols)):
            cmap[cols[q]].append(q)
        for a in range(len(rows)):
            i = rows[a]
            pr = List[Tuple[int, T]]()
            for p in range(len(self.rows[i])):
                for q in cmap[self.rows[i][p]]:
                    pr.append((q, self.data[i][p]))
            pr.sort(key=lambda t: t[0])
            for t in pr:
                B.rows[a].append(t[0])
                B.data[a].append(t[1])
        return B

    def getrow(self, i: int):
        return self.__getitem__((i, slice(None, None, None)))

    def getcol(self, j: int):
        return self.__getitem__((slice(None, None, None), j))

    def __setitem__(self, key, x):
        if isinstance(key, Tuple):
            if isinstance(key[0], int) and isinstance(key[1], int):
                i, j = self._chk(key[0], key[1])
                if _issparse(x):
                    self._insert(i, j, T(x.toarray()[0, 0]))
                elif isinstance(x, np.ndarray):
                    if x.size != 1:
                        raise ValueError("Trying to assign a sequence to an item")
                    self._insert(i, j, T(x.ravel()[0]))
                else:
                    self._insert(i, j, T(x))
                return
        rows, cols, pw = _key2(key, self._shape)
        V = _bcast_value(x, len(rows), len(cols), pw, T)
        if pw:
            for q in range(len(rows)):
                self._insert(rows[q], cols[q], V[0, q])
        else:
            for a in range(len(rows)):
                for b in range(len(cols)):
                    self._insert(rows[a], cols[b], V[a, b])

    # --- conversions ---
    def tocsr(self, copy: bool = False):
        m, n = self._shape
        Bp = np.zeros(m + 1, dtype=int)
        for i in range(m):
            Bp[i + 1] = Bp[i] + len(self.rows[i])
        nnz = Bp[m]
        Bj = np.empty(nnz, dtype=int)
        Bx = np.empty(nnz, dtype=T)
        q = 0
        for i in range(m):
            for p in range(len(self.rows[i])):
                Bj[q] = self.rows[i][p]
                Bx[q] = self.data[i][p]
                q += 1
        return _compressed[T, 0, M](Bx, Bj, Bp, self._shape)

    def tocsc(self, copy: bool = False):
        return self.tocsr().tocsc()

    def tocoo(self, copy: bool = False):
        c = self.tocsr().tocoo()
        c._canonical = True
        return c

    def todia(self, copy: bool = False):
        return self.tocoo().todia()

    def tobsr(self, blocksize = None, copy: bool = False):
        return self.tocsr().tobsr(blocksize)

    def tolil(self, copy: bool = False):
        return self.copy() if copy else self

    def todok(self, copy: bool = False):
        return self.tocoo().todok()

    def asformat(self, fmt: Literal[str]):
        return _asfmt(self, fmt)

    def toarray(self):
        D = np.zeros(self._shape, dtype=T)
        for i in range(self._shape[0]):
            for p in range(len(self.rows[i])):
                D[i, self.rows[i][p]] = self.data[i][p]
        return D

    def todense(self):
        return self.toarray()

    def transpose(self, axes = None, copy: bool = False):
        return _csr_tolil(self.tocsr().transpose().tocsr())

    @property
    def T(self):
        return self.transpose()

    def conj(self, copy: bool = True):
        if _is_cplx(T):
            B = self.copy()
            for d in B.data:
                for p in range(len(d)):
                    d[p] = d[p].conjugate()
            return B
        else:
            return self.copy()

    def conjugate(self, copy: bool = True):
        return self.conj()

    @property
    def H(self):
        return self.conj().transpose()

    def reshape(self, shape, order: str = "C", copy: bool = False):
        nm, nn = _shape2(shape)
        m, n = self._shape
        if nm < 0:
            nm = (m * n) // nn
        if nn < 0:
            nn = (m * n) // nm
        if (nm, nn) == self._shape:
            return self.copy() if copy else self
        if nm * nn != m * n:
            raise ValueError(f"cannot reshape array of size {m * n} into shape ({nm}, {nn})")
        B = _lil[T, M]((nm, nn))
        for i in range(m):
            for p in range(len(self.rows[i])):
                j = self.rows[i][p]
                f = i * n + j if order == "C" else i + j * m
                if order == "C":
                    B._insert(f // nn, f % nn, self.data[i][p])
                else:
                    B._insert(f % nm, f // nm, self.data[i][p])
        return B

    def resize(self, *shape):
        nm, nn = _shape_args(shape)
        m, n = self._shape
        if nm < m:
            self.rows = self.rows[:nm]
            self.data = self.data[:nm]
        else:
            for _ in range(m, nm):
                self.rows.append(List[int]())
                self.data.append(List[T]())
        if nn < n:
            for i in range(len(self.rows)):
                t = _bisect_left(self.rows[i], nn)
                self.rows[i] = self.rows[i][:t]
                self.data[i] = self.data[i][:t]
        self._shape = (nm, nn)

    # --- arithmetic (scipy: LIL delegates to CSR except scalar * and /) ---
    def __neg__(self):
        return -self.tocsr()

    def __abs__(self):
        return abs(self.tocsr())

    def _scale(self, s):
        R = type(_up(T, type(s)))
        B = _lil[R, M](self._shape)
        if s == 0:
            return B
        for i in range(self._shape[0]):
            B.rows[i] = self.rows[i].copy()
            B.data[i] = [R(v) * R(s) for v in self.data[i]]
        return B

    def __add__(self, other):
        return self.tocsr().__add__(other)

    def __radd__(self, other):
        return self.tocsr().__radd__(other)

    def __sub__(self, other):
        return self.tocsr().__sub__(other)

    def __rsub__(self, other):
        return self.tocsr().__rsub__(other)

    def multiply(self, other):
        return self.tocsr().multiply(other)

    def __mul__(self, other):
        if _issparse(other) or isinstance(other, np.ndarray) or isinstance(other, List):
            return self.tocsr().__mul__(other)
        else:
            return self._scale(other)

    def __rmul__(self, other):
        if isinstance(other, np.ndarray) or isinstance(other, List):
            return self.tocsr().__rmul__(other)
        else:
            return self._scale(other)

    def __truediv__(self, other):
        if _issparse(other) or isinstance(other, np.ndarray) or isinstance(other, List):
            return self.tocsr().__truediv__(other)
        else:
            R = type(_up(type(_up(T, type(other))), float))
            B = _lil[R, M](self._shape)
            for i in range(self._shape[0]):
                B.rows[i] = self.rows[i].copy()
                B.data[i] = [R(v) / R(other) for v in self.data[i]]
            return B

    def power(self, n, dtype: type = NoneType):
        return self.tocsr().power(n)

    def maximum(self, other):
        return self.tocsr().maximum(other)

    def minimum(self, other):
        return self.tocsr().minimum(other)

    def __matmul__(self, other):
        return self.tocsr().__matmul__(other)

    def __rmatmul__(self, other):
        return self.tocsr().__rmatmul__(other)

    def dot(self, other):
        return self.tocsr().__matmul__(other)

    def sum(self, axis = None):
        return self.tocsr().sum(axis)

    def mean(self, axis = None):
        return self.tocsr().mean(axis)

    def max(self):
        return self.tocsr().max()

    def min(self):
        return self.tocsr().min()

    def diagonal(self, k: int = 0):
        m, n = self._shape
        if k <= -m or k >= n:
            return np.empty(0, dtype=T)
        i0 = -k if k < 0 else 0
        L = min(m - i0, n - (i0 + k))
        out = np.empty(L, dtype=T)
        for t in range(L):
            out[t] = self._get1(i0 + t, i0 + t + k)
        return out

    def trace(self, offset: int = 0):
        return self.diagonal(offset).sum()

    def setdiag(self, values, k: int = 0):
        # scipy _spbase._setdiag -> self[i, j] = values
        m, n = self._shape
        i0 = -k if k < 0 else 0
        L = max(0, min(m - i0, n - (i0 + k)))
        if isinstance(values, np.ndarray) or isinstance(values, List):
            vals = np.asarray(values).astype(T)
            L = min(L, len(vals))
            for t in range(L):
                self._insert(i0 + t, i0 + t + k, vals[t])
        else:
            for t in range(L):
                self._insert(i0 + t, i0 + t + k, T(values))

    def nonzero(self):
        return self.tocsr().nonzero()

def _csr_tolil(A):
    # scipy _csr.tolil (after sum_duplicates)
    T = type(A.dtype())
    C = A.copy()
    C.sum_duplicates()
    m = A.shape[0]
    if A._mstatic():
        L = _lil[T, 1](A.shape)
    else:
        L = _lil[T, 0](A.shape)
    for i in range(m):
        for p in range(C.indptr[i], C.indptr[i + 1]):
            L.rows[i].append(C.indices[p])
            L.data[i].append(C.data[p])
    return L
