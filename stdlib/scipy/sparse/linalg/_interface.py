# Copyright (C) 2026 the Typhon authors
# Ported from scipy 1.18.1 (BSD-3-Clause)

# LinearOperator: a generic matrix-free operator. MV / RMV / MM / RMM are the
# callable types (NoneType when not given); D is the scalar dtype.

import numpy as np
from .._base import _compressed, _coo, _up

def _is_sparse(A) -> Literal[bool]:
    return isinstance(A, _compressed) or isinstance(A, _coo)

# The operator algebra below is shared by every operator class through
# _OpMixin-style duplication (Typhon classes cannot inherit generically).

def _dot_any(A, x):
    if isinstance(x, np.ndarray) or isinstance(x, List):
        v = np.asarray(x)
        if v.ndim == 1:
            return A.matvec(v)
        else:
            return A.matmat(v)
    else:
        return _ProductOp(A, x)

def _default_matmat(A, X, adj: bool):
    R = type(_up(A.dtype, X.dtype))
    n = A.shape[1] if adj else A.shape[0]
    Y = np.empty((n, X.shape[1]), dtype=R)
    for j in range(X.shape[1]):
        col = np.ascontiguousarray(X[:, j])
        y = A.rmatvec(col) if adj else A.matvec(col)
        for i in range(n):
            Y[i, j] = R(y[i])
    return Y


def _scale(A, x):
    R = type(_up(A.dtype, type(x)))
    return _ScaledOp[type(A), R](A, R(x))

class _FuncOp[MV, RMV, MM, RMM, D]:
    shape: Tuple[int, int]
    _mv: MV
    _rmv: RMV
    _mm: MM
    _rmm: RMM

    def __init__(self, shape: Tuple[int, int], mv: MV, rmv: RMV, mm: MM, rmm: RMM):
        self.shape = shape
        self._mv = mv
        self._rmv = rmv
        self._mm = mm
        self._rmm = rmm

    @property
    def dtype(self):
        return D

    def matvec(self, x):
        v = np.asarray(x).ravel()
        if v.shape[0] != self.shape[1]:
            raise ValueError("dimension mismatch")
        return np.asarray(self._mv(v)).ravel()

    def rmatvec(self, x):
        v = np.asarray(x).ravel()
        if v.shape[0] != self.shape[0]:
            raise ValueError("dimension mismatch")
        if RMV is NoneType:
            if RMM is NoneType:
                raise NotImplementedError("rmatvec is not defined")
                return np.asarray(self._mv(v)).ravel()
            else:
                return np.asarray(self._rmm(v.reshape((len(v), 1)))).ravel()
        else:
            return np.asarray(self._rmv(v)).ravel()

    def matmat(self, X):
        A = np.asarray(X)
        if A.ndim != 2:
            raise ValueError("expected 2-d ndarray or matrix")
        if A.shape[0] != self.shape[1]:
            raise ValueError("dimension mismatch")
        if MM is NoneType:
            return _default_matmat(self, A, False)
        else:
            return np.asarray(self._mm(A))

    def rmatmat(self, X):
        A = np.asarray(X)
        if A.ndim != 2:
            raise ValueError("expected 2-d ndarray or matrix")
        if RMM is NoneType:
            return _default_matmat(self, A, True)
        else:
            return np.asarray(self._rmm(A))

    @property
    def ndim(self):
        return 2

    def dot(self, x):
        return _dot_any(self, x)

    def __matmul__(self, x):
        return _dot_any(self, x)

    def __call__(self, x):
        return _dot_any(self, x)

    def __mul__(self, x):
        if isinstance(x, np.ndarray) or _is_linop(x):
            return _dot_any(self, x)
        else:
            return _scale(self, x)

    def __rmul__(self, x):
        return _scale(self, x)

    def __neg__(self):
        return _scale(self, -1.0)

    def __add__(self, B):
        return _SumOp(self, B)

    def __sub__(self, B):
        return _SumOp(self, _scale(B, -1.0))

    def adjoint(self):
        return _AdjointOp(self)

    @property
    def H(self):
        return _AdjointOp(self)

    def transpose(self):
        return _TransposeOp(self)

    @property
    def T(self):
        return _TransposeOp(self)

    def todense(self):
        return self.matmat(np.eye(self.shape[1], dtype=self.dtype))

    def __repr__(self):
        return f"<{self.shape[0]}x{self.shape[1]} LinearOperator>"


def LinearOperator(shape, matvec, rmatvec = None, matmat = None, dtype: type = float,
                   rmatmat = None):
    shp = (int(shape[0]), int(shape[1]))
    return _FuncOp[type(matvec), type(rmatvec), type(matmat), type(rmatmat), dtype](
        shp, matvec, rmatvec, matmat, rmatmat)

class _MatrixOp[S, SH, D]:
    # wraps a dense ndarray or a sparse container
    shape: Tuple[int, int]
    A: S
    AH: SH

    def __init__(self, A: S, AH: SH):
        self.A = A
        self.AH = AH
        self.shape = A.shape

    @property
    def dtype(self):
        return D

    def matvec(self, x):
        v = np.asarray(x).ravel()
        if v.shape[0] != self.shape[1]:
            raise ValueError("dimension mismatch")
        return self.A @ v

    def rmatvec(self, x):
        v = np.asarray(x).ravel()
        if v.shape[0] != self.shape[0]:
            raise ValueError("dimension mismatch")
        return self.AH @ v

    def matmat(self, X):
        return self.A @ np.asarray(X)

    def rmatmat(self, X):
        return self.AH @ np.asarray(X)

    @property
    def ndim(self):
        return 2

    def dot(self, x):
        return _dot_any(self, x)

    def __matmul__(self, x):
        return _dot_any(self, x)

    def __call__(self, x):
        return _dot_any(self, x)

    def __mul__(self, x):
        if isinstance(x, np.ndarray) or _is_linop(x):
            return _dot_any(self, x)
        else:
            return _scale(self, x)

    def __rmul__(self, x):
        return _scale(self, x)

    def __neg__(self):
        return _scale(self, -1.0)

    def __add__(self, B):
        return _SumOp(self, B)

    def __sub__(self, B):
        return _SumOp(self, _scale(B, -1.0))

    def adjoint(self):
        return _AdjointOp(self)

    @property
    def H(self):
        return _AdjointOp(self)

    def transpose(self):
        return _TransposeOp(self)

    @property
    def T(self):
        return _TransposeOp(self)

    def todense(self):
        return self.matmat(np.eye(self.shape[1], dtype=self.dtype))

    def __repr__(self):
        return f"<{self.shape[0]}x{self.shape[1]} LinearOperator>"


class _ScaledOp[A, D]:
    shape: Tuple[int, int]
    op: A
    alpha: D

    def __init__(self, op: A, alpha: D):
        self.op = op
        self.shape = op.shape
        self.alpha = alpha

    @property
    def dtype(self):
        return D

    def matvec(self, x):
        return self.alpha * self.op.matvec(x)

    def rmatvec(self, x):
        a = self.alpha.conjugate() if D is complex else self.alpha
        return a * self.op.rmatvec(x)

    def matmat(self, X):
        return self.alpha * self.op.matmat(X)

    def rmatmat(self, X):
        a = self.alpha.conjugate() if D is complex else self.alpha
        return a * self.op.rmatmat(X)

    @property
    def ndim(self):
        return 2

    def dot(self, x):
        return _dot_any(self, x)

    def __matmul__(self, x):
        return _dot_any(self, x)

    def __call__(self, x):
        return _dot_any(self, x)

    def __mul__(self, x):
        if isinstance(x, np.ndarray) or _is_linop(x):
            return _dot_any(self, x)
        else:
            return _scale(self, x)

    def __rmul__(self, x):
        return _scale(self, x)

    def __neg__(self):
        return _scale(self, -1.0)

    def __add__(self, B):
        return _SumOp(self, B)

    def __sub__(self, B):
        return _SumOp(self, _scale(B, -1.0))

    def adjoint(self):
        return _AdjointOp(self)

    @property
    def H(self):
        return _AdjointOp(self)

    def transpose(self):
        return _TransposeOp(self)

    @property
    def T(self):
        return _TransposeOp(self)

    def todense(self):
        return self.matmat(np.eye(self.shape[1], dtype=self.dtype))

    def __repr__(self):
        return f"<{self.shape[0]}x{self.shape[1]} LinearOperator>"


class _SumOp[A, B]:
    shape: Tuple[int, int]
    a: A
    b: B

    def __init__(self, a: A, b: B):
        if a.shape != b.shape:
            raise ValueError("cannot add operators with different shapes")
        self.a = a
        self.b = b
        self.shape = a.shape

    @property
    def dtype(self):
        return type(_up(self.a.dtype, self.b.dtype))

    def matvec(self, x):
        return self.a.matvec(x) + self.b.matvec(x)

    def rmatvec(self, x):
        return self.a.rmatvec(x) + self.b.rmatvec(x)

    def matmat(self, X):
        return self.a.matmat(X) + self.b.matmat(X)

    def rmatmat(self, X):
        return self.a.rmatmat(X) + self.b.rmatmat(X)

    @property
    def ndim(self):
        return 2

    def dot(self, x):
        return _dot_any(self, x)

    def __matmul__(self, x):
        return _dot_any(self, x)

    def __call__(self, x):
        return _dot_any(self, x)

    def __mul__(self, x):
        if isinstance(x, np.ndarray) or _is_linop(x):
            return _dot_any(self, x)
        else:
            return _scale(self, x)

    def __rmul__(self, x):
        return _scale(self, x)

    def __neg__(self):
        return _scale(self, -1.0)

    def __add__(self, B):
        return _SumOp(self, B)

    def __sub__(self, B):
        return _SumOp(self, _scale(B, -1.0))

    def adjoint(self):
        return _AdjointOp(self)

    @property
    def H(self):
        return _AdjointOp(self)

    def transpose(self):
        return _TransposeOp(self)

    @property
    def T(self):
        return _TransposeOp(self)

    def todense(self):
        return self.matmat(np.eye(self.shape[1], dtype=self.dtype))

    def __repr__(self):
        return f"<{self.shape[0]}x{self.shape[1]} LinearOperator>"


class _ProductOp[A, B]:
    shape: Tuple[int, int]
    a: A
    b: B

    def __init__(self, a: A, b: B):
        if a.shape[1] != b.shape[0]:
            raise ValueError("cannot multiply operators with incompatible shapes")
        self.a = a
        self.b = b
        self.shape = (a.shape[0], b.shape[1])

    @property
    def dtype(self):
        return type(_up(self.a.dtype, self.b.dtype))

    def matvec(self, x):
        return self.a.matvec(self.b.matvec(x))

    def rmatvec(self, x):
        return self.b.rmatvec(self.a.rmatvec(x))

    def matmat(self, X):
        return self.a.matmat(self.b.matmat(X))

    def rmatmat(self, X):
        return self.b.rmatmat(self.a.rmatmat(X))

    @property
    def ndim(self):
        return 2

    def dot(self, x):
        return _dot_any(self, x)

    def __matmul__(self, x):
        return _dot_any(self, x)

    def __call__(self, x):
        return _dot_any(self, x)

    def __mul__(self, x):
        if isinstance(x, np.ndarray) or _is_linop(x):
            return _dot_any(self, x)
        else:
            return _scale(self, x)

    def __rmul__(self, x):
        return _scale(self, x)

    def __neg__(self):
        return _scale(self, -1.0)

    def __add__(self, B):
        return _SumOp(self, B)

    def __sub__(self, B):
        return _SumOp(self, _scale(B, -1.0))

    def adjoint(self):
        return _AdjointOp(self)

    @property
    def H(self):
        return _AdjointOp(self)

    def transpose(self):
        return _TransposeOp(self)

    @property
    def T(self):
        return _TransposeOp(self)

    def todense(self):
        return self.matmat(np.eye(self.shape[1], dtype=self.dtype))

    def __repr__(self):
        return f"<{self.shape[0]}x{self.shape[1]} LinearOperator>"


class _AdjointOp[A]:
    shape: Tuple[int, int]
    a: A

    def __init__(self, a: A):
        self.a = a
        self.shape = (a.shape[1], a.shape[0])

    @property
    def dtype(self):
        return self.a.dtype

    def matvec(self, x):
        return self.a.rmatvec(x)

    def rmatvec(self, x):
        return self.a.matvec(x)

    def matmat(self, X):
        return self.a.rmatmat(X)

    def rmatmat(self, X):
        return self.a.matmat(X)

    @property
    def ndim(self):
        return 2

    def dot(self, x):
        return _dot_any(self, x)

    def __matmul__(self, x):
        return _dot_any(self, x)

    def __call__(self, x):
        return _dot_any(self, x)

    def __mul__(self, x):
        if isinstance(x, np.ndarray) or _is_linop(x):
            return _dot_any(self, x)
        else:
            return _scale(self, x)

    def __rmul__(self, x):
        return _scale(self, x)

    def __neg__(self):
        return _scale(self, -1.0)

    def __add__(self, B):
        return _SumOp(self, B)

    def __sub__(self, B):
        return _SumOp(self, _scale(B, -1.0))

    def adjoint(self):
        return _AdjointOp(self)

    @property
    def H(self):
        return _AdjointOp(self)

    def transpose(self):
        return _TransposeOp(self)

    @property
    def T(self):
        return _TransposeOp(self)

    def todense(self):
        return self.matmat(np.eye(self.shape[1], dtype=self.dtype))

    def __repr__(self):
        return f"<{self.shape[0]}x{self.shape[1]} LinearOperator>"


class _TransposeOp[A]:
    shape: Tuple[int, int]
    a: A

    def __init__(self, a: A):
        self.a = a
        self.shape = (a.shape[1], a.shape[0])

    @property
    def dtype(self):
        return self.a.dtype

    def matvec(self, x):
        if self.a.dtype is complex:
            return np.conj(self.a.rmatvec(np.conj(np.asarray(x))))
        else:
            return self.a.rmatvec(x)

    def rmatvec(self, x):
        if self.a.dtype is complex:
            return np.conj(self.a.matvec(np.conj(np.asarray(x))))
        else:
            return self.a.matvec(x)

    def matmat(self, X):
        return _default_matmat(self, np.asarray(X), False)

    def rmatmat(self, X):
        return _default_matmat(self, np.asarray(X), True)

    @property
    def ndim(self):
        return 2

    def dot(self, x):
        return _dot_any(self, x)

    def __matmul__(self, x):
        return _dot_any(self, x)

    def __call__(self, x):
        return _dot_any(self, x)

    def __mul__(self, x):
        if isinstance(x, np.ndarray) or _is_linop(x):
            return _dot_any(self, x)
        else:
            return _scale(self, x)

    def __rmul__(self, x):
        return _scale(self, x)

    def __neg__(self):
        return _scale(self, -1.0)

    def __add__(self, B):
        return _SumOp(self, B)

    def __sub__(self, B):
        return _SumOp(self, _scale(B, -1.0))

    def adjoint(self):
        return _AdjointOp(self)

    @property
    def H(self):
        return _AdjointOp(self)

    def transpose(self):
        return _TransposeOp(self)

    @property
    def T(self):
        return _TransposeOp(self)

    def todense(self):
        return self.matmat(np.eye(self.shape[1], dtype=self.dtype))

    def __repr__(self):
        return f"<{self.shape[0]}x{self.shape[1]} LinearOperator>"


def _ident(x):
    return np.asarray(x).copy()

def IdentityOperator(shape: Tuple[int, int], dtype: type = float):
    return LinearOperator(shape, _ident, rmatvec=_ident, matmat=_ident,
                          dtype=dtype, rmatmat=_ident)

def _is_linop(A) -> Literal[bool]:
    return (isinstance(A, _FuncOp) or isinstance(A, _MatrixOp) or
            isinstance(A, _ScaledOp) or isinstance(A, _SumOp) or
            isinstance(A, _ProductOp) or isinstance(A, _AdjointOp) or
            isinstance(A, _TransposeOp))

def aslinearoperator(A):
    if _is_linop(A):
        return A
    elif _is_sparse(A):
        S = A.tocsr()
        SH = S.conj().transpose().tocsr()
        return _MatrixOp[type(S), type(SH), S.data.dtype](S, SH)
    else:
        M = np.asarray(A)
        if M.ndim != 2:
            raise ValueError("expected a 2-d array")
        if M.dtype is complex:
            MH = np.ascontiguousarray(np.conj(M.T))
        else:
            MH = np.ascontiguousarray(M.T)
        return _MatrixOp[type(M), type(MH), M.dtype](M, MH)
