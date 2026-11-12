# Copyright (C) 2026 the Typhon authors
# Ported from scipy 1.18.1 (BSD-3-Clause)

# Sparse DIAgonal format: data[k, j] holds A[j - offsets[k], j].

import numpy as np
from ._sparsetools import _arr
from ._sparsetools import dia_matvec, dia_matvecs, dia_tocsr, dia_matmat
from ._base import _compressed, _coo, _up, _is_cplx, _issparse, _fmt_name, \
    _coo_str, _shape2, _shape_args, _reshape_coo, _asfmt, _dtype_name, _sum_t

def _pymod(a: int, b: int) -> int:
    r = a % b
    return r + b if r < 0 else r

class _dia[T, M: Static[int]]:
    data: np.ndarray[T, 2]
    offsets: np.ndarray[int, 1]
    _shape: Tuple[int, int]

    def __init__(self, data: np.ndarray[T, 2], offsets: np.ndarray[int, 1],
                 shape: Tuple[int, int]):
        if data.shape[0] != len(offsets):
            raise ValueError(f"number of diagonals ({data.shape[0]}) does not match the number of offsets ({len(offsets)})")
        for a in range(len(offsets)):
            for b in range(a + 1, len(offsets)):
                if offsets[a] == offsets[b]:
                    raise ValueError("offset array contains duplicate values")
        self.data = data
        self.offsets = offsets
        self._shape = shape

    @staticmethod
    def _spfmt() -> Literal[str]:
        return "dia"

    @property
    def shape(self):
        return self._shape

    @property
    def ndim(self):
        return 2

    @property
    def format(self) -> str:
        return "dia"

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def _is_matrix(self) -> bool:
        return M == 1

    def _mstatic(self) -> Literal[bool]:
        return M == 1

    @property
    def nnz(self) -> int:
        m, n = self._shape
        L = min(self.data.shape[1], n)
        s = 0
        for k in range(len(self.offsets)):
            off = self.offsets[k]
            s += max(min(m + off, L) - max(off, 0), 0)
        return s

    @property
    def size(self) -> int:
        return self.nnz

    def getnnz(self) -> int:
        return self.nnz

    def _in_mask(self, k: int, j: int) -> bool:
        m, n = self._shape
        r = j - self.offsets[k]
        return r >= 0 and r < m and j < n

    def count_nonzero(self) -> int:
        c = 0
        for k in range(self.data.shape[0]):
            for j in range(self.data.shape[1]):
                if self._in_mask(k, j) and self.data[k, j] != T():
                    c += 1
        return c

    def copy(self):
        return _dia[T, M](self.data.copy(), self.offsets.copy(), self._shape)

    def astype(self, dt: type, copy: bool = True):
        return _dia[dt, M](self.data.astype(dt), self.offsets.copy(), self._shape)

    def _with_data(self, d):
        return _dia[d.dtype, M](d, self.offsets.copy(), self._shape)

    def _as_array(self):
        return _dia[T, 0](self.data, self.offsets, self._shape)

    def _as_matrix(self):
        return _dia[T, 1](self.data, self.offsets, self._shape)

    def __repr__(self):
        what = "matrix" if M == 1 else "array"
        return (f"<DIAgonal sparse {what} of dtype '{_dtype_name(T)}'\n\twith "
                f"{self.nnz} stored elements ({len(self.offsets)} diagonals) "
                f"and shape {self._shape}>")

    def __str__(self):
        return _coo_str(self.tocoo(), self.__repr__())

    def __len__(self):
        raise TypeError("sparse array length is ambiguous; use getnnz() or shape[0]")
        return 0

    # --- conversions ---
    def tocsr(self, copy: bool = False):
        m, n = self._shape
        if m == 0 or n == 0 or len(self.offsets) == 0:
            return _compressed[T, 0, M](np.empty(0, dtype=T), np.empty(0, dtype=int),
                                        np.zeros(m + 1, dtype=int), self._shape)
        order = np.argsort(self.offsets, kind="stable")
        Bp, Bj, Bx = dia_tocsr(m, n, self.data.shape[0], self.data.shape[1],
                               self.offsets, self.data, order)
        return _compressed[T, 0, M](Bx, Bj, Bp, self._shape)

    def tocsc(self, copy: bool = False):
        return self.tocsr().tocsc()

    def tocoo(self, copy: bool = False):
        c = self.tocsr().tocoo()
        c._canonical = True
        return c

    def todia(self, copy: bool = False):
        return self.copy() if copy else self

    def tobsr(self, blocksize = None, copy: bool = False):
        return self.tocsr().tobsr(blocksize)

    def tolil(self, copy: bool = False):
        return self.tocsr().tolil()

    def todok(self, copy: bool = False):
        return self.tocoo().todok()

    def asformat(self, fmt: Literal[str]):
        return _asfmt(self, fmt)

    def toarray(self):
        return self.tocoo().toarray()

    def todense(self):
        return self.toarray()

    def transpose(self, axes = None, copy: bool = False):
        m, n = self._shape
        max_dim = max(m, n)
        nd = len(self.offsets)
        offs = np.empty(nd, dtype=int)
        for k in range(nd):
            offs[k] = -self.offsets[k]
        W = max(self.data.shape[1], max_dim)
        data = np.zeros((nd, m), dtype=T)
        for k in range(nd):
            sh = _pymod(offs[k], max_dim)
            for t in range(m):
                c = t - sh
                if c < 0:
                    c += W
                if c < self.data.shape[1]:
                    data[k, t] = self.data[k, c]
        return _dia[T, M](data, offs, (n, m))

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

    def reshape(self, shape, order: str = "C", copy: bool = False):
        # deviation: always COO (scipy returns self when the shape is unchanged)
        return _reshape_coo(self.tocoo(), shape, order, M)

    def resize(self, *shape):
        nm, nn = _shape_args(shape)
        m = self._shape[0]
        w = min(self.data.shape[1], nn)
        d = np.zeros((self.data.shape[0], w), dtype=T)
        for k in range(self.data.shape[0]):
            for j in range(w):
                d[k, j] = self.data[k, j]
        if nm > m:
            for k in range(self.data.shape[0]):
                if self.offsets[k] + m < w:
                    for j in range(w):
                        if self.offsets[k] + m <= j:
                            d[k, j] = T()
        self.data = d
        self._shape = (nm, nn)

    # --- elementwise / arithmetic (scipy _dia / _data semantics) ---
    def __neg__(self):
        return self._with_data(-self.data)

    def __abs__(self):
        return self._with_data(np.abs(self.data))

    def __pos__(self):
        return self.copy()

    def power(self, n, dtype: type = NoneType):
        return self._with_data(self.data ** n)

    def _scale(self, s):
        R = type(_up(T, type(s)))
        return self._with_data(self.data.astype(R) * R(s))

    def _add_dia(self, other, sub: bool):
        # scipy _dia._add_sparse (general path; fast paths give equal values)
        R = type(_up(T, other.data.dtype))
        if self._shape != other.shape:
            raise ValueError("inconsistent shapes")
        offs = List[int]()
        for k in range(len(self.offsets)):
            offs.append(self.offsets[k])
        for k in range(len(other.offsets)):
            offs.append(other.offsets[k])
        uo = sorted(set(offs))
        new_off = _arr(uo, int)
        m, n = self._shape
        same = len(self.offsets) == len(other.offsets)
        if same:
            for k in range(len(self.offsets)):
                if self.offsets[k] != other.offsets[k]:
                    same = False
        if same and self.data.shape[1] == other.data.shape[1]:
            d0 = self.data.astype(R)
            d1 = other.data.astype(R)
            return _dia[R, M](d0 - d1 if sub else d0 + d1, self.offsets.copy(), self._shape)
        sd = self.data.shape[1]
        od = other.data.shape[1]
        if sd == od and (len(uo) == len(self.offsets) or len(uo) == len(other.offsets)):
            d = sd
        else:
            d = min(m + uo[len(uo) - 1], n)
        nd = np.zeros((len(uo), max(d, 0)), dtype=R)
        W = nd.shape[1]
        for k in range(len(self.offsets)):
            row = uo.index(self.offsets[k])
            for j in range(min(sd, W)):
                nd[row, j] += R(self.data[k, j])
        for k in range(len(other.offsets)):
            row = uo.index(other.offsets[k])
            for j in range(min(od, W)):
                if sub:
                    nd[row, j] -= R(other.data[k, j])
                else:
                    nd[row, j] += R(other.data[k, j])
        return _dia[R, M](nd, new_off, self._shape)

    def __add__(self, other):
        if _issparse(other):
            if other._spfmt() == "dia":
                return self._add_dia(other, False)
            else:
                return self.tocsr().__add__(other)
        else:
            return self.tocsr().__add__(other)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if _issparse(other):
            if other._spfmt() == "dia":
                return self._add_dia(other, True)
            else:
                return self.tocsr().__sub__(other)
        else:
            return self.tocsr().__sub__(other)

    def __rsub__(self, other):
        return self.tocsr().__rsub__(other)

    def multiply(self, other):
        if _issparse(other):
            if other._spfmt() == "dia":
                R = type(_up(T, other.data.dtype))
                if other.shape != self._shape:
                    return self.tocsr().multiply(other)
                offs = List[int]()
                ia = List[int]()
                ib = List[int]()
                for a in sorted(range(len(self.offsets)), key=lambda q: self.offsets[q]):
                    for b in range(len(other.offsets)):
                        if self.offsets[a] == other.offsets[b]:
                            offs.append(self.offsets[a])
                            ia.append(a)
                            ib.append(b)
                L = min(self.data.shape[1], other.data.shape[1])
                d = np.zeros((len(offs), L), dtype=R)
                for q in range(len(offs)):
                    for j in range(L):
                        d[q, j] = R(self.data[ia[q], j]) * R(other.data[ib[q], j])
                return _dia[R, M](d, _arr(offs, int), self._shape)
            else:
                return self.tocsr().multiply(other)
        elif isinstance(other, np.ndarray) or isinstance(other, List):
            return self.tocsr().multiply(other)
        else:
            return self._scale(other)

    def __mul__(self, other):
        if _issparse(other) or isinstance(other, np.ndarray) or isinstance(other, List):
            if M == 1:
                return self.__matmul__(other)
            else:
                return self.multiply(other)
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
            return self._with_data(self.data.astype(R) / R(other))

    def maximum(self, other):
        return self.tocsr().maximum(other)

    def minimum(self, other):
        return self.tocsr().minimum(other)

    # --- products ---
    def _matvec(self, x: np.ndarray[X, 1], X: type):
        R = type(_up(T, X))
        m, n = self._shape
        if len(x) != n:
            raise ValueError("dimension mismatch")
        y = np.zeros(m, dtype=R)
        dia_matvec(m, n, len(self.offsets), self.data.shape[1], self.offsets,
                   self.data.astype(R), x.astype(R), y)
        return y

    def _matmat(self, B: np.ndarray[XT, 2], XT: type):
        R = type(_up(T, XT))
        m, n = self._shape
        if B.shape[0] != n:
            raise ValueError("dimension mismatch")
        Y = np.zeros((m, B.shape[1]), dtype=R)
        dia_matvecs(m, n, self.data.shape[0], self.data.shape[1], self.offsets,
                    self.data.astype(R), B.shape[1], B.astype(R), Y)
        return Y

    def __matmul__(self, other):
        if _issparse(other):
            if other._spfmt() == "dia":
                R = type(_up(T, other.data.dtype))
                m, k1 = self._shape
                k2, n = other.shape
                if k1 != k2:
                    raise ValueError("dimension mismatch")
                if m == 0 or n == 0 or k1 == 0 or len(self.offsets) == 0 or len(other.offsets) == 0:
                    return _dia[R, M](np.zeros((0, 0), dtype=R), np.empty(0, dtype=int), (m, n))
                offs, d = dia_matmat(m, k1, self.data.shape[0], self.data.shape[1],
                                     self.offsets, self.data.astype(R), n,
                                     other.data.shape[0], other.data.shape[1],
                                     other.offsets, other.data.astype(R))
                return _dia[R, M](d, offs, (m, n))
            else:
                return self.tocsr().__matmul__(other)
        else:
            x = np.asarray(other)
            if x.ndim == 1:
                return self._matvec(x)
            else:
                return self._matmat(x)

    def __rmatmul__(self, other):
        return self.tocsr().__rmatmul__(other)

    def dot(self, other):
        return self.__matmul__(other)

    # --- reductions / diagonals ---
    def sum(self, axis = None):
        S = type(_sum_t(T))
        m, n = self._shape
        if axis is None:
            s = S()
            r = self.sum(1)
            for i in range(m):
                s += r[i]
            return s
        ax = axis if axis >= 0 else axis + 2
        if ax == 0:
            res = np.zeros(n, dtype=S)
            for j in range(min(n, self.data.shape[1])):
                for k in range(self.data.shape[0]):
                    if self._in_mask(k, j):
                        res[j] += S(self.data[k, j])
            return res
        else:
            res = np.zeros(m, dtype=S)
            dia_matvec(m, n, len(self.offsets), self.data.shape[1], self.offsets,
                       self.data.astype(S), np.ones(n, dtype=S), res)
            return res

    def mean(self, axis = None):
        m, n = self._shape
        if axis is None:
            return self.sum() / float(m * n)
        ax = axis if axis >= 0 else axis + 2
        return self.sum(axis) / float(m if ax == 0 else n)

    def max(self):
        return self.tocsr().max()

    def min(self):
        return self.tocsr().min()

    def diagonal(self, k: int = 0):
        rows, cols = self._shape
        if k <= -rows or k >= cols:
            return np.empty(0, dtype=T)
        first_col = max(0, k)
        last_col = min(rows + k, cols)
        res = np.zeros(last_col - first_col, dtype=T)
        for q in range(len(self.offsets)):
            if self.offsets[q] == k:
                for j in range(first_col, min(last_col, self.data.shape[1])):
                    res[j - first_col] = self.data[q, j]
        return res

    def trace(self, offset: int = 0):
        return self.diagonal(offset).sum()

    def setdiag(self, values, k: int = 0):
        # scipy _dia._setdiag
        m, n = self._shape
        if isinstance(values, np.ndarray) or isinstance(values, List):
            vals = np.asarray(values).astype(T)
            vn = len(vals)
        else:
            vals = np.full(1, T(values), dtype=T)
            vn = 1 << 62
        if k < 0:
            cnt = min(m + k, n, vn)
            lo = 0
            hi = cnt
        else:
            cnt = min(m, n - k, vn)
            lo = k
            hi = k + cnt
        rows, cols = self.data.shape[0], self.data.shape[1]
        where = -1
        for q in range(len(self.offsets)):
            if self.offsets[q] == k:
                where = q
        if where >= 0:
            if hi > cols:
                d = np.zeros((rows, hi), dtype=T)
                for a in range(rows):
                    for b in range(cols):
                        d[a, b] = self.data[a, b]
                self.data = d
            for j in range(lo, hi):
                self.data[where, j] = vals[j - lo] if vn != (1 << 62) else vals[0]
        else:
            no = np.empty(len(self.offsets) + 1, dtype=int)
            for q in range(len(self.offsets)):
                no[q] = self.offsets[q]
            no[len(self.offsets)] = k
            w = max(hi, cols)
            d = np.zeros((rows + 1, w), dtype=T)
            for a in range(rows):
                for b in range(cols):
                    d[a, b] = self.data[a, b]
            for j in range(lo, hi):
                d[rows, j] = vals[j - lo] if vn != (1 << 62) else vals[0]
            self.offsets = no
            self.data = d

    def nonzero(self):
        return self.tocsr().nonzero()

    def __getitem__(self, key):
        return self.tocsr().__getitem__(key)

    def getrow(self, i: int):
        return self.tocsr().getrow(i)

    def getcol(self, j: int):
        return self.tocsr().getcol(j)

def _coo_todia(c):
    # scipy _coo.todia
    T = type(c.dtype())
    A = c.copy()
    A.sum_duplicates()
    k = len(A.data)
    ks = List[int]()
    for q in range(k):
        ks.append(A.col[q] - A.row[q])
    diags = sorted(set(ks))
    if k == 0:
        data = np.zeros((0, 0), dtype=T)
    else:
        cmax = 0
        for q in range(k):
            cmax = max(cmax, A.col[q])
        data = np.zeros((len(diags), cmax + 1), dtype=T)
        for q in range(k):
            data[diags.index(ks[q]), A.col[q]] = A.data[q]
    if c._mstatic():
        return _dia[T, 1](data, _arr(diags, int), c.shape)
    else:
        return _dia[T, 0](data, _arr(diags, int), c.shape)
