# Copyright (C) 2026 the Typhon authors
# Ported from scipy 1.18.1 (BSD-3-Clause)

# Block Sparse Row format. data is (nblocks, R, C); kernels in _sparsetools
# work on the flat C-ordered block data.

import numpy as np
from ._sparsetools import _arr
from ._sparsetools import csr_count_blocks, csr_tobsr, bsr_tocsr, \
    bsr_transpose, bsr_sort_indices, bsr_matvec, bsr_matvecs, bsr_matmat, \
    bsr_binop_bsr, bsr_diagonal, csr_has_sorted_indices, \
    csr_has_canonical_format, csr_eliminate_zeros
from ._base import _compressed, _coo, _up, _is_cplx, _issparse, _coo_str, \
    _shape2, _shape_args, _reshape_coo, _asfmt, _dtype_name, _sum_t, \
    _resize_coo

def _flat(d):
    return d.reshape(d.shape[0] * d.shape[1] * d.shape[2])

class _bsr[T, M: Static[int]]:
    data: np.ndarray[T, 3]
    indices: np.ndarray[int, 1]
    indptr: np.ndarray[int, 1]
    _shape: Tuple[int, int]

    def __init__(self, data: np.ndarray[T, 3], indices: np.ndarray[int, 1],
                 indptr: np.ndarray[int, 1], shape: Tuple[int, int]):
        R = data.shape[1]
        C = data.shape[2]
        if shape[0] % R != 0 or shape[1] % C != 0:
            raise ValueError("shape must be multiple of blocksize")
        if len(indptr) != shape[0] // R + 1:
            raise ValueError(f"index pointer size ({len(indptr)}) should be ({shape[0] // R + 1})")
        self.data = data
        self.indices = indices
        self.indptr = indptr
        self._shape = shape

    @staticmethod
    def _spfmt() -> Literal[str]:
        return "bsr"

    @property
    def shape(self):
        return self._shape

    @property
    def ndim(self):
        return 2

    @property
    def format(self) -> str:
        return "bsr"

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def blocksize(self) -> Tuple[int, int]:
        return (self.data.shape[1], self.data.shape[2])

    @property
    def _is_matrix(self) -> bool:
        return M == 1

    def _mstatic(self) -> Literal[bool]:
        return M == 1

    @property
    def nnz(self) -> int:
        R, C = self.blocksize
        return self.indptr[len(self.indptr) - 1] * R * C

    @property
    def size(self) -> int:
        return self.nnz

    def getnnz(self) -> int:
        return self.nnz

    def count_nonzero(self) -> int:
        return self.tocsr().count_nonzero()

    def _nb(self):
        R, C = self.blocksize
        return self._shape[0] // R, self._shape[1] // C

    def copy(self):
        return _bsr[T, M](self.data.copy(), self.indices.copy(), self.indptr.copy(),
                          self._shape)

    def astype(self, dt: type, copy: bool = True):
        return _bsr[dt, M](self.data.astype(dt), self.indices.copy(),
                           self.indptr.copy(), self._shape)

    def _with_data(self, d):
        return _bsr[d.dtype, M](d, self.indices.copy(), self.indptr.copy(), self._shape)

    def _as_array(self):
        return _bsr[T, 0](self.data, self.indices, self.indptr, self._shape)

    def _as_matrix(self):
        return _bsr[T, 1](self.data, self.indices, self.indptr, self._shape)

    def __repr__(self):
        what = "matrix" if M == 1 else "array"
        R, C = self.blocksize
        return (f"<Block Sparse Row sparse {what} of dtype '{_dtype_name(T)}'\n\twith "
                f"{self.nnz} stored elements (blocksize={R}x{C}) and shape {self._shape}>")

    def __str__(self):
        return _coo_str(self.tocoo(), self.__repr__())

    def __len__(self):
        raise TypeError("sparse array length is ambiguous; use getnnz() or shape[0]")
        return 0

    # --- structure ---
    @property
    def has_sorted_indices(self) -> bool:
        return csr_has_sorted_indices(len(self.indptr) - 1, self.indptr, self.indices)

    @property
    def has_canonical_format(self) -> bool:
        return csr_has_canonical_format(len(self.indptr) - 1, self.indptr, self.indices)

    def sort_indices(self):
        if self.has_sorted_indices:
            return
        R, C = self.blocksize
        nbr, nbc = self._nb()
        fl = _flat(self.data).copy()
        bsr_sort_indices(nbr, nbc, R, C, self.indptr, self.indices, fl)
        self.data = fl.reshape((len(self.indices), R, C))

    def sorted_indices(self):
        A = self.copy()
        A.sort_indices()
        return A

    def prune(self):
        bnnz = self.indptr[len(self.indptr) - 1]
        self.data = self.data[:bnnz].copy()
        self.indices = self.indices[:bnnz].copy()

    def sum_duplicates(self):
        # scipy _bsr.sum_duplicates (python port of csr_sum_duplicates)
        if self.has_canonical_format:
            return
        self.sort_indices()
        n_row = len(self.indptr) - 1
        nnz = 0
        row_end = 0
        for i in range(n_row):
            jj = row_end
            row_end = self.indptr[i + 1]
            while jj < row_end:
                j = self.indices[jj]
                x = self.data[jj].copy()
                jj += 1
                while jj < row_end and self.indices[jj] == j:
                    x += self.data[jj]
                    jj += 1
                self.indices[nnz] = j
                self.data[nnz] = x
                nnz += 1
            self.indptr[i + 1] = nnz
        self.prune()

    def eliminate_zeros(self):
        R, C = self.blocksize
        nb = self.indptr[len(self.indptr) - 1]
        if nb == 0:
            return
        nbr, nbc = self._nb()
        mask = np.zeros(nb, dtype=int)
        keep = List[int]()
        for b in range(nb):
            nzb = False
            for r in range(R):
                for c in range(C):
                    if self.data[b, r, c] != T():
                        nzb = True
            if nzb:
                mask[b] = 1
                keep.append(b)
        nd = self.data[_arr(keep, int)].copy()
        csr_eliminate_zeros(nbr, self.indptr, self.indices, mask)
        self.data = nd
        self.prune()

    # --- conversions ---
    def tocsr(self, copy: bool = False):
        R, C = self.blocksize
        nbr, nbc = self._nb()
        Bp, Bj, Bx = bsr_tocsr(nbr, nbc, R, C, self.indptr, self.indices, _flat(self.data))
        return _compressed[T, 0, M](Bx, Bj, Bp, self._shape)

    def tocsc(self, copy: bool = False):
        return self.tocsr().tocsc()

    def tocoo(self, copy: bool = False):
        R, C = self.blocksize
        nbr, nbc = self._nb()
        nb = self.indptr[nbr]
        row = np.empty(nb * R * C, dtype=int)
        col = np.empty(nb * R * C, dtype=int)
        q = 0
        for br in range(nbr):
            for b in range(self.indptr[br], self.indptr[br + 1]):
                for r in range(R):
                    for c in range(C):
                        row[q] = R * br + r
                        col[q] = C * self.indices[b] + c
                        q += 1
        return _coo[T, M](_flat(self.data).copy(), row, col, self._shape, False)

    def tobsr(self, blocksize = None, copy: bool = False):
        if blocksize is None:
            return self.copy() if copy else self
        else:
            if (blocksize[0], blocksize[1]) == self.blocksize:
                return self.copy() if copy else self
            return _csr_tobsr(self.tocsr(), blocksize)

    def todia(self, copy: bool = False):
        return self.tocoo().todia()

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
        R, C = self.blocksize
        m, n = self._shape
        nbr, nbc = self._nb()
        nb = self.indptr[nbr]
        if nb == 0:
            return _bsr[T, M](np.zeros((0, C, R), dtype=T), np.empty(0, dtype=int),
                              np.zeros(n // C + 1, dtype=int), (n, m))
        Bp, Bj, Bx = bsr_transpose(nbr, nbc, R, C, self.indptr, self.indices,
                                   _flat(self.data))
        return _bsr[T, M](Bx.reshape((nb, C, R)), Bj, Bp, (n, m))

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
        # scipy: falls back to a csr resize then back to bsr (blocksize kept)
        nm, nn = _shape_args(shape)
        c = self.tocoo()
        _resize_coo(c, (nm, nn))
        B = _csr_tobsr(c.tocsr(), self.blocksize)
        self.data = B.data
        self.indices = B.indices
        self.indptr = B.indptr
        self._shape = (nm, nn)

    # --- arithmetic ---
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

    def _binopt(self, other, op: Literal[int]):
        # scipy _bsr._binopt: other converted to this blocksize
        o = _csr_tobsr(other.tocsr(), self.blocksize) if other._spfmt() != "bsr" \
            else other.tobsr(self.blocksize)
        R = type(_up(T, o.data.dtype))
        if self._shape != o.shape:
            raise ValueError("inconsistent shapes")
        a = self.astype(R)
        b = o.astype(R)
        a.sum_duplicates()
        b.sum_duplicates()
        Rb, Cb = self.blocksize
        nbr, nbc = self._nb()
        Cp, Cj, Cx = bsr_binop_bsr(nbr, nbc, Rb, Cb, a.indptr, a.indices, _flat(a.data),
                                   b.indptr, b.indices, _flat(b.data), op)
        return _bsr[R, M](Cx.reshape((len(Cj), Rb, Cb)), Cj, Cp, self._shape)

    def __add__(self, other):
        if _issparse(other):
            return self._binopt(other, 0)
        else:
            return self.tocsr().__add__(other)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if _issparse(other):
            return self._binopt(other, 1)
        else:
            return self.tocsr().__sub__(other)

    def __rsub__(self, other):
        return self.tocsr().__rsub__(other)

    def multiply(self, other):
        if _issparse(other):
            return self._binopt(other, 2)
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
            Rt = type(_up(type(_up(T, type(other))), float))
            return self._with_data(self.data.astype(Rt) / Rt(other))

    def maximum(self, other):
        return self.tocsr().maximum(other)

    def minimum(self, other):
        return self.tocsr().minimum(other)

    def _matvec(self, x: np.ndarray[X, 1], X: type):
        Rt = type(_up(T, X))
        m, n = self._shape
        if len(x) != n:
            raise ValueError("dimension mismatch")
        R, C = self.blocksize
        nbr, nbc = self._nb()
        y = np.zeros(m, dtype=Rt)
        bsr_matvec(nbr, nbc, R, C, self.indptr, self.indices,
                   _flat(self.data).astype(Rt), x.astype(Rt), y)
        return y

    def _matmat(self, B: np.ndarray[XT, 2], XT: type):
        Rt = type(_up(T, XT))
        m, n = self._shape
        if B.shape[0] != n:
            raise ValueError("dimension mismatch")
        R, C = self.blocksize
        nbr, nbc = self._nb()
        Y = np.zeros((m, B.shape[1]), dtype=Rt)
        bsr_matvecs(nbr, nbc, B.shape[1], R, C, self.indptr, self.indices,
                    _flat(self.data).astype(Rt), B.astype(Rt), Y)
        return Y

    def _matmul_sparse(self, other):
        # scipy _bsr._matmul_sparse: other -> bsr with blocksize (n, C)
        m, k1 = self._shape
        k2, n = other.shape
        if k1 != k2:
            raise ValueError("dimension mismatch")
        R, nb = self.blocksize
        if other._spfmt() == "bsr":
            C = other.blocksize[1]
        else:
            C = 1
        o = _csr_tobsr(other.tocsr(), (nb, C))
        Rt = type(_up(T, o.data.dtype))
        Cp, Cj, Cx = bsr_matmat(m // R, n // C, R, C, nb, self.indptr, self.indices,
                                _flat(self.data).astype(Rt), o.indptr, o.indices,
                                _flat(o.data).astype(Rt))
        return _bsr[Rt, M](Cx.reshape((len(Cj), R, C)), Cj, Cp, (m, n))

    def __matmul__(self, other):
        if _issparse(other):
            return self._matmul_sparse(other)
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

    def sum(self, axis = None):
        return self.tocsr().sum(axis)

    def mean(self, axis = None):
        return self.tocsr().mean(axis)

    def max(self):
        return self.tocsr().max()

    def min(self):
        return self.tocsr().min()

    def diagonal(self, k: int = 0):
        rows, cols = self._shape
        if k <= -rows or k >= cols:
            return np.empty(0, dtype=T)
        R, C = self.blocksize
        return bsr_diagonal(k, rows // R, cols // C, R, C, self.indptr, self.indices,
                            _flat(self.data))

    def trace(self, offset: int = 0):
        return self.diagonal(offset).sum()

    def setdiag(self, values, k: int = 0):
        # scipy: via csr (_spbase._setdiag -> self[i, j] = v), keeps blocksize
        A = self.tocsr()
        A.setdiag(values, k)
        B = _csr_tobsr(A, self.blocksize)
        self.data = B.data
        self.indices = B.indices
        self.indptr = B.indptr

    def nonzero(self):
        return self.tocsr().nonzero()

    def __getitem__(self, key):
        raise NotImplementedError("BSR does not support indexing")
        return self.tocsr().__getitem__(key)

def _estimate_blocksize(A, efficiency: float = 0.7) -> Tuple[int, int]:
    # scipy._spfuncs.estimate_blocksize (A: CSR)
    if A.nnz == 0:
        return (1, 1)
    high = (1.0 + efficiency) / 2.0
    nnz = float(A.nnz)
    M, N = A.shape
    e22 = 0.0
    e33 = 0.0
    if M % 2 == 0 and N % 2 == 0:
        e22 = nnz / (4 * csr_count_blocks(M, N, 2, 2, A.indptr, A.indices))
    if M % 3 == 0 and N % 3 == 0:
        e33 = nnz / (9 * csr_count_blocks(M, N, 3, 3, A.indptr, A.indices))
    if e22 > high and e33 > high:
        e66 = nnz / (36 * csr_count_blocks(M, N, 6, 6, A.indptr, A.indices))
        if e66 > efficiency:
            return (6, 6)
        else:
            return (3, 3)
    else:
        e44 = 0.0
        if M % 4 == 0 and N % 4 == 0:
            e44 = nnz / (16 * csr_count_blocks(M, N, 4, 4, A.indptr, A.indices))
        if e44 > efficiency:
            return (4, 4)
        elif e33 > efficiency:
            return (3, 3)
        elif e22 > efficiency:
            return (2, 2)
        else:
            return (1, 1)

def _csr_tobsr(A, blocksize):
    # scipy _csr.tobsr (A: CSR container)
    T = type(A.dtype())
    if blocksize is None:
        bs = _estimate_blocksize(A)
    else:
        bs = (int(blocksize[0]), int(blocksize[1]))
    R, C = bs
    m, n = A.shape
    if R < 1 or C < 1 or m % R != 0 or n % C != 0:
        raise ValueError(f"invalid blocksize {bs}")
    Bp, Bj, Bx = csr_tobsr(m, n, R, C, A.indptr, A.indices, A.data)
    data = Bx.reshape((len(Bj), R, C))
    if A._mstatic():
        return _bsr[T, 1](data, Bj, Bp, A.shape)
    else:
        return _bsr[T, 0](data, Bj, Bp, A.shape)
