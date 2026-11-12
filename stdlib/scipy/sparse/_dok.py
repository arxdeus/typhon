# Copyright (C) 2026 the Typhon authors
# Ported from scipy 1.18.1 (BSD-3-Clause)

# Dictionary Of Keys format: insertion-ordered Dict[(i, j)] -> value
# (tocoo() keeps the dict order and flags the result canonical, as scipy).

import numpy as np
from ._base import _compressed, _coo, _up, _is_cplx, _issparse, _coo_str, \
    _shape2, _shape_args, _asfmt, _dtype_name
from ._index import _idx1, _key2, _bcast_value

class _dok[T, M: Static[int]]:
    _dict: Dict[Tuple[int, int], T]
    _shape: Tuple[int, int]

    def __init__(self, shape: Tuple[int, int]):
        self._shape = shape
        self._dict = Dict[Tuple[int, int], T]()

    @staticmethod
    def _spfmt() -> Literal[str]:
        return "dok"

    @property
    def shape(self):
        return self._shape

    @property
    def ndim(self):
        return 2

    @property
    def format(self) -> str:
        return "dok"

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
        return len(self._dict)

    @property
    def size(self) -> int:
        return self.nnz

    def getnnz(self) -> int:
        return self.nnz

    def count_nonzero(self) -> int:
        c = 0
        for v in self._dict.values():
            if v != T():
                c += 1
        return c

    def __len__(self):
        raise TypeError("sparse array length is ambiguous; use getnnz() or shape[0]")
        return 0

    def __contains__(self, key: Tuple[int, int]) -> bool:
        return key in self._dict

    def keys(self):
        return self._dict.keys()

    def values(self):
        return self._dict.values()

    def items(self):
        return self._dict.items()

    def get(self, key: Tuple[int, int], default = 0.0):
        i, j = self._chk(key[0], key[1])
        if (i, j) in self._dict:
            return self._dict[(i, j)]
        return T(default)

    def __delitem__(self, key: Tuple[int, int]):
        del self._dict[key]

    def clear(self):
        self._dict.clear()

    def pop(self, key: Tuple[int, int], default = None):
        if key in self._dict:
            return self._dict.pop(key)
        if default is None:
            raise KeyError(str(key))
            return T()
        return T(default)

    def setdefault(self, key: Tuple[int, int], default = None):
        if key in self._dict:
            return self._dict[key]
        v = T() if default is None else T(default)
        if v != T():
            self._dict[key] = v
        return v

    def update(self, other):
        raise NotImplementedError("Direct update to DOK sparse format is not allowed.")

    def copy(self):
        B = _dok[T, M](self._shape)
        for k, v in self._dict.items():
            B._dict[k] = v
        return B

    def astype(self, dt: type, copy: bool = True):
        B = _dok[dt, M](self._shape)
        for k, v in self._dict.items():
            B._dict[k] = dt(v)
        return B

    def _as_array(self):
        B = _dok[T, 0](self._shape)
        B._dict = self._dict
        return B

    def _as_matrix(self):
        B = _dok[T, 1](self._shape)
        B._dict = self._dict
        return B

    def __repr__(self):
        what = "matrix" if M == 1 else "array"
        return (f"<Dictionary Of Keys sparse {what} of dtype '{_dtype_name(T)}'\n\twith "
                f"{self.nnz} stored elements and shape {self._shape}>")

    def __str__(self):
        return _coo_str(self.tocoo(), self.__repr__())

    def _chk(self, i: int, j: int):
        m, n = self._shape
        if i < -m or i >= m:
            raise IndexError(f"row index ({i}) out of range")
        if j < -n or j >= n:
            raise IndexError(f"column index ({j}) out of range")
        return (i + m if i < 0 else i, j + n if j < 0 else j)

    def _set1(self, i: int, j: int, x: T):
        key = (i, j)
        if x != T():
            self._dict[key] = x
        elif key in self._dict:
            del self._dict[key]

    def __getitem__(self, key):
        if isinstance(key, Tuple):
            if isinstance(key[0], int) and isinstance(key[1], int):
                i, j = self._chk(key[0], key[1])
                return self._dict.get((i, j), T())
        rows, cols, pw = _key2(key, self._shape)
        if pw:
            B = _dok[T, M]((1, len(rows)))
            for q in range(len(rows)):
                v = self._dict.get((rows[q], cols[q]), T())
                if v != T():
                    B._dict[(0, q)] = v
            return B
        B = _dok[T, M]((len(rows), len(cols)))
        for a in range(len(rows)):
            for b in range(len(cols)):
                v = self._dict.get((rows[a], cols[b]), T())
                if v != T():
                    B._dict[(a, b)] = v
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
                    self._set1(i, j, T(x.toarray()[0, 0]))
                elif isinstance(x, np.ndarray):
                    if x.size != 1:
                        raise ValueError("Trying to assign a sequence to an item")
                    self._set1(i, j, T(x.ravel()[0]))
                else:
                    self._set1(i, j, T(x))
                return
        rows, cols, pw = _key2(key, self._shape)
        V = _bcast_value(x, len(rows), len(cols), pw, T)
        if pw:
            for q in range(len(rows)):
                self._set1(rows[q], cols[q], V[0, q])
        else:
            for a in range(len(rows)):
                for b in range(len(cols)):
                    self._set1(rows[a], cols[b], V[a, b])

    # --- conversions ---
    def tocoo(self, copy: bool = False):
        k = len(self._dict)
        r = np.empty(k, dtype=int)
        c = np.empty(k, dtype=int)
        d = np.empty(k, dtype=T)
        q = 0
        for key, v in self._dict.items():
            r[q] = key[0]
            c[q] = key[1]
            d[q] = v
            q += 1
        return _coo[T, M](d, r, c, self._shape, True)

    def tocsr(self, copy: bool = False):
        return self.tocoo().tocsr()

    def tocsc(self, copy: bool = False):
        return self.tocoo().tocsc()

    def todia(self, copy: bool = False):
        return self.tocoo().todia()

    def tobsr(self, blocksize = None, copy: bool = False):
        return self.tocsr().tobsr(blocksize)

    def tolil(self, copy: bool = False):
        return self.tocsr().tolil()

    def todok(self, copy: bool = False):
        return self.copy() if copy else self

    def asformat(self, fmt: Literal[str]):
        return _asfmt(self, fmt)

    def toarray(self):
        D = np.zeros(self._shape, dtype=T)
        for key, v in self._dict.items():
            D[key[0], key[1]] = v
        return D

    def todense(self):
        return self.toarray()

    def transpose(self, axes = None, copy: bool = False):
        B = _dok[T, M]((self._shape[1], self._shape[0]))
        for key, v in self._dict.items():
            B._dict[(key[1], key[0])] = v
        return B

    @property
    def T(self):
        return self.transpose()

    def conj(self, copy: bool = True):
        if _is_cplx(T):
            B = _dok[T, M](self._shape)
            for key, v in self._dict.items():
                B._dict[key] = v.conjugate()
            return B
        else:
            return self.copy()

    def conjugate(self, copy: bool = True):
        return self.conj()

    @property
    def H(self):
        return self.conj().transpose()

    def reshape(self, shape, order: str = "C", copy: bool = False):
        return self.tocoo().reshape(shape, order)

    def resize(self, *shape):
        nm, nn = _shape_args(shape)
        m, n = self._shape
        if nm < m or nn < n:
            drop = List[Tuple[int, int]]()
            for key in self._dict.keys():
                if key[0] >= nm or key[1] >= nn:
                    drop.append(key)
            for key in drop:
                del self._dict[key]
        self._shape = (nm, nn)

    # --- arithmetic (scipy _dok semantics; others via CSR) ---
    def __neg__(self):
        B = _dok[T, M](self._shape)
        for key, v in self._dict.items():
            B._dict[key] = -v
        return B

    def __abs__(self):
        return abs(self.tocsr())

    def _scale(self, s):
        R = type(_up(T, type(s)))
        B = _dok[R, M](self._shape)
        for key, v in self._dict.items():
            B._dict[key] = R(v) * R(s)
        return B

    def __add__(self, other):
        if _issparse(other):
            if other.shape != self._shape:
                raise ValueError("Matrix dimensions are not equal.")
            R = type(_up(T, type(other.dtype())))
            B = _dok[R, M](self._shape)
            for key, v in self._dict.items():
                B._dict[key] = R(v)
            if other._spfmt() == "dok":
                for key, v in other._dict.items():
                    B._dict[key] = B._dict.get(key, R()) + R(v)
            else:
                c = other.tocoo()
                for q in range(len(c.data)):
                    key = (c.row[q], c.col[q])
                    B._dict[key] = B._dict.get(key, R()) + R(c.data[q])
            return B
        elif isinstance(other, np.ndarray) or isinstance(other, List):
            return self.toarray() + np.asarray(other)
        else:
            R = type(_up(T, type(other)))
            B = _dok[R, M](self._shape)
            for i in range(self._shape[0]):
                for j in range(self._shape[1]):
                    a = R(self._dict.get((i, j), T())) + R(other)
                    if a != R():
                        B._dict[(i, j)] = a
            return B

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if _issparse(other):
            return self.tocsr().__sub__(other)
        elif isinstance(other, np.ndarray) or isinstance(other, List):
            return self.toarray() - np.asarray(other)
        else:
            return self.__add__(-other)

    def __rsub__(self, other):
        return (-self).__add__(other)

    def multiply(self, other):
        return self.tocsr().multiply(other)

    def __mul__(self, other):
        if _issparse(other) or isinstance(other, np.ndarray) or isinstance(other, List):
            if M == 1:
                return self.__matmul__(other)
            else:
                return self.tocsr().multiply(other)
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
            B = _dok[R, M](self._shape)
            for key, v in self._dict.items():
                B._dict[key] = R(v) / R(other)
            return B

    def power(self, n, dtype: type = NoneType):
        return self.tocsr().power(n)

    def maximum(self, other):
        return self.tocsr().maximum(other)

    def minimum(self, other):
        return self.tocsr().minimum(other)

    def __matmul__(self, other):
        if _issparse(other):
            return self.tocsr().__matmul__(other)
        x = np.asarray(other)
        R = type(_up(T, x.dtype))
        if x.ndim == 1:
            if len(x) != self._shape[1]:
                raise ValueError("dimension mismatch")
            y = np.zeros(self._shape[0], dtype=R)
            for key, v in self._dict.items():
                y[key[0]] += R(v) * R(x[key[1]])
            return y
        else:
            if x.shape[0] != self._shape[1]:
                raise ValueError("dimension mismatch")
            Y = np.zeros((self._shape[0], x.shape[1]), dtype=R)
            for key, v in self._dict.items():
                for c in range(x.shape[1]):
                    Y[key[0], c] += R(v) * R(x[key[1], c])
            return Y

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
        return self.tocsr().diagonal(k)

    def trace(self, offset: int = 0):
        return self.diagonal(offset).sum()

    def setdiag(self, values, k: int = 0):
        m, n = self._shape
        i0 = -k if k < 0 else 0
        L = max(0, min(m - i0, n - (i0 + k)))
        if isinstance(values, np.ndarray) or isinstance(values, List):
            vals = np.asarray(values).astype(T)
            L = min(L, len(vals))
            for t in range(L):
                self._set1(i0 + t, i0 + t + k, vals[t])
        else:
            for t in range(L):
                self._set1(i0 + t, i0 + t + k, T(values))

    def nonzero(self):
        return self.tocsr().nonzero()

def _coo_todok(c):
    # scipy _coo.todok: sum_duplicates then dict(zip(coords, data))
    T = type(c.dtype())
    A = c.copy()
    A.sum_duplicates()
    if c._mstatic():
        B = _dok[T, 1](c.shape)
    else:
        B = _dok[T, 0](c.shape)
    for q in range(len(A.data)):
        B._dict[(A.row[q], A.col[q])] = A.data[q]
    return B
