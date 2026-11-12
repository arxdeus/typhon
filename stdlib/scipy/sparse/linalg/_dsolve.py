# Copyright (C) 2026 the Typhon authors
# Ported from scipy 1.18.1 (BSD-3-Clause)

# Sparse direct solvers. scipy wraps SuperLU; here splu is a left-looking
# (column-by-column) sparse LU with partial pivoting, the same threshold rule
# as SuperLU with diag_pivot_thresh=1 (largest magnitude in the column; ties
# go to the diagonal / smallest row). Column ordering: NATURAL for every
# permc_spec (COLAMD / MMD orderings are accepted but not applied), so the
# factors L, U, perm_r, perm_c differ from SuperLU's while solutions agree to
# rounding.

import numpy as np
from .._base import _compressed, _coo, _up, _is_cplx
from .._construct import csc_matrix, csc_array, csr_array, eye

class MatrixRankWarning:
    pass

class SuperLU[T]:
    n: int
    # L: unit lower triangular, U: upper triangular, stored per column in
    # pivoted row order (row index = pivot position)
    Lp: List[int]
    Li: List[int]
    Lx: List[T]
    Up: List[int]
    Ui: List[int]
    Ux: List[T]
    perm_r: np.ndarray[int, 1]
    perm_c: np.ndarray[int, 1]
    _piv_rows: List[int]

    def __init__(self, n: int):
        self._piv_rows = []
        self.n = n
        self.Lp = [0]
        self.Li = []
        self.Lx = []
        self.Up = [0]
        self.Ui = []
        self.Ux = []
        self.perm_r = np.zeros(n, dtype=int)
        self.perm_c = np.arange(n)

    @property
    def shape(self):
        return (self.n, self.n)

    @property
    def nnz(self) -> int:
        return len(self.Lx) + len(self.Ux)

    @property
    def L(self):
        n = self.n
        Bx = np.array(self.Lx, dtype=T)
        Bi = np.array(self.Li, dtype=int)
        return csc_array((Bx, Bi, np.array(self.Lp, dtype=int)), shape=(n, n))

    @property
    def U(self):
        n = self.n
        return csc_array((np.array(self.Ux, dtype=T), np.array(self.Ui, dtype=int),
                          np.array(self.Up, dtype=int)), shape=(n, n))

    def _solve1(self, b: np.ndarray[T, 1], trans: str):
        # Pr A Pc = L U   (Pr: row i of A goes to perm_r[i])
        n = self.n
        if trans == "N":
            # A x = b  ->  L U (Pc^T x) = Pr b
            y = np.empty(n, dtype=T)
            for i in range(n):
                y[self.perm_r[i]] = b[i]
            for j in range(n):
                yj = y[j]
                if yj != T():
                    for p in range(self.Lp[j], self.Lp[j + 1]):
                        y[self.Li[p]] -= self.Lx[p] * yj
            for j in range(n - 1, -1, -1):
                # diagonal is the last entry of column j of U
                last = self.Up[j + 1] - 1
                y[j] /= self.Ux[last]
                yj = y[j]
                if yj != T():
                    for p in range(self.Up[j], last):
                        y[self.Ui[p]] -= self.Ux[p] * yj
            x = np.empty(n, dtype=T)
            for j in range(n):
                x[self.perm_c[j]] = y[j]
            return x
        else:
            conj = trans == "H"
            # A^T x = b -> U^T L^T Pr x = Pc^T b
            y = np.empty(n, dtype=T)
            for j in range(n):
                y[j] = b[self.perm_c[j]]
            for j in range(n):
                s = y[j]
                last = self.Up[j + 1] - 1
                for p in range(self.Up[j], last):
                    u = self.Ux[p]
                    if _is_cplx(T) and conj:
                        u = u.conjugate()
                    s -= u * y[self.Ui[p]]
                d = self.Ux[last]
                if _is_cplx(T) and conj:
                    d = d.conjugate()
                y[j] = s / d
            for j in range(n - 1, -1, -1):
                s = y[j]
                for p in range(self.Lp[j], self.Lp[j + 1]):
                    l = self.Lx[p]
                    if _is_cplx(T) and conj:
                        l = l.conjugate()
                    s -= l * y[self.Li[p]]
                y[j] = s
            x = np.empty(n, dtype=T)
            for i in range(n):
                x[i] = y[self.perm_r[i]]
            return x

    def solve(self, rhs, trans: str = "N"):
        b = np.asarray(rhs)
        R = type(_up(T, b.dtype))
        if R is not T:
            compile_error("splu: right-hand side dtype must not be complex for a real factorization")
        if b.ndim == 1:
            if len(b) != self.n:
                raise ValueError("dimension mismatch")
            return self._solve1(b.astype(T), trans)
        else:
            X = np.empty((b.shape[0], b.shape[1]), dtype=T)
            for j in range(b.shape[1]):
                col = np.ascontiguousarray(b[:, j]).astype(T)
                xj = self._solve1(col, trans)
                for i in range(self.n):
                    X[i, j] = xj[i]
            return X

def _lu_factor(A, T: type, drop_tol: float = 0.0, fill_factor: float = 1e30):
    # A: CSC with sorted, summed entries
    m, n = A.shape
    if m != n:
        raise ValueError("can only factor square matrices")
    F = SuperLU[T](n)
    pinv = np.full(n, -1, dtype=int)   # row of A -> pivot position
    x = np.zeros(n, dtype=T)
    mark = np.zeros(n, dtype=bool)
    nz = List[int]()
    for j in range(n):
        nz.clear()
        for p in range(A.indptr[j], A.indptr[j + 1]):
            i = A.indices[p]
            x[i] += A.data[p]
            if not mark[i]:
                mark[i] = True
                nz.append(i)
        # left-looking update in pivot order 0..j-1 (a topological order:
        # column k of L only holds rows pivoted after step k). L row indices
        # are original rows during factorization, converted at the end.
        for k in range(j):
            r = F._piv_rows[k]
            xr = x[r]
            if xr != T():
                for p in range(F.Lp[k], F.Lp[k + 1]):
                    i = F.Li[p]
                    if not mark[i]:
                        mark[i] = True
                        nz.append(i)
                    x[i] -= F.Lx[p] * xr
        # U column: entries in pivoted rows
        for k in range(j):
            r = F._piv_rows[k]
            if mark[r] and x[r] != T():
                F.Ui.append(k)
                F.Ux.append(x[r])
        # pivot: largest magnitude among non-pivoted rows; ties -> diagonal
        piv = -1
        best = -1.0
        for i in nz:
            if pinv[i] < 0:
                a = abs(x[i])
                if a > best or (a == best and i == j):
                    best = a
                    piv = i
        if piv < 0 or best == 0.0:
            raise RuntimeError("Factor is exactly singular")
        d = x[piv]
        F.Ui.append(j)
        F.Ux.append(d)
        F.Up.append(len(F.Ux))
        pinv[piv] = j
        F._piv_rows.append(piv)
        for i in nz:
            if pinv[i] < 0 and x[i] != T():
                v = x[i] / d
                if drop_tol == 0.0 or abs(v) > drop_tol:
                    F.Li.append(i)
                    F.Lx.append(v)
        F.Lp.append(len(F.Lx))
        for i in nz:
            x[i] = T()
            mark[i] = False
    # convert L row indices to pivot positions
    for p in range(len(F.Li)):
        F.Li[p] = pinv[F.Li[p]]
    F.perm_r = pinv
    return F

def _as_csc(A):
    if isinstance(A, _compressed) or isinstance(A, _coo):
        C = A.tocsc(copy=True)
    else:
        C = csc_array(np.asarray(A))
    C.sum_duplicates()
    return C

def _is_mat(A) -> Literal[bool]:
    if isinstance(A, _compressed) or isinstance(A, _coo):
        return A._mstatic()
    else:
        return False

def _ftype(T: type):
    return _up(T, float)

def splu(A, permc_spec = None, diag_pivot_thresh = None, relax = None,
         panel_size = None, options = None):
    C = _as_csc(A)
    T = type(_ftype(C.data.dtype))
    return _lu_factor(C.astype(T), T)

def spilu(A, drop_tol = None, fill_factor = None, drop_rule = None,
          permc_spec = None, diag_pivot_thresh = None, relax = None,
          panel_size = None, options = None):
    C = _as_csc(A)
    T = type(_ftype(C.data.dtype))
    dt = 1e-4 if drop_tol is None else float(drop_tol)
    return _lu_factor(C.astype(T), T, dt)

def factorized(A):
    return splu(A)

def spsolve(A, b, permc_spec = None, use_umfpack: bool = True):
    C = _as_csc(A)
    m, n = C.shape
    if m != n:
        raise ValueError(f"matrix must be square (has shape {C.shape})")
    if isinstance(b, _compressed) or isinstance(b, _coo):
        # sparse right-hand side: sparse result (CSC for 2-D)
        B = b.tocsc()
        if B.shape[0] != n:
            raise ValueError("matrix - rhs dimension mismatch")
        T = type(_up(type(_ftype(C.data.dtype)), B.data.dtype))
        F = _lu_factor(C.astype(T), T)
        X = F.solve(B.toarray().astype(T))
        if A._mstatic():
            return csc_matrix(X)
        else:
            return csc_array(X)
    else:
        bv = np.asarray(b)
        T = type(_up(type(_ftype(C.data.dtype)), bv.dtype))
        if bv.shape[0] != n:
            raise ValueError(f"matrix - rhs dimension mismatch ({C.shape} - {bv.shape[0]})")
        F = _lu_factor(C.astype(T), T)
        if bv.ndim == 2 and bv.shape[1] == 1:
            return F.solve(bv.astype(T).ravel())
        return F.solve(bv.astype(T))

def spsolve_triangular(A, b, lower: bool = True, overwrite_A: bool = False,
                       overwrite_b: bool = False, unit_diagonal: bool = False):
    R = A.tocsr()
    R.sum_duplicates()
    bv = np.asarray(b)
    T = type(_up(type(_ftype(R.data.dtype)), bv.dtype))
    n = R.shape[0]
    if R.shape[0] != R.shape[1]:
        raise ValueError("A must be a square matrix")
    X = bv.astype(T).copy()
    two = X.ndim == 2
    k = X.shape[1] if two else 1
    order = list(range(n)) if lower else list(range(n - 1, -1, -1))
    for i in order:
        diag = T()
        for c in range(k):
            s = X[i, c] if two else X[i]
            for p in range(R.indptr[i], R.indptr[i + 1]):
                j = R.indices[p]
                if j == i:
                    if c == 0:
                        diag += T(R.data[p])
                elif (j < i) if lower else (j > i):
                    s -= T(R.data[p]) * (X[j, c] if two else X[j])
            if unit_diagonal:
                v = s
            else:
                if c == 0 and diag == T():
                    raise ValueError("A is singular: zero entry on diagonal.")
                v = s / diag
            if two:
                X[i, c] = v
            else:
                X[i] = v
    return X

def inv(A):
    C = _as_csc(A)
    n = C.shape[0]
    T = type(_ftype(C.data.dtype))
    F = _lu_factor(C.astype(T), T)
    X = F.solve(np.eye(n, dtype=T))
    if _is_mat(A):
        return csc_matrix(X)
    else:
        return csc_array(X)
