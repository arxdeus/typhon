# Copyright (C) 2026 the Typhon authors
# Ported from scipy 1.18.1 (BSD-3-Clause)

# Sparse direct solvers. scipy wraps SuperLU; here splu follows SuperLU's
# driver: column ordering by get_perm_c (COLAMD default, MMD_ATA,
# MMD_AT_PLUS_A, NATURAL; ports in _colamd.py / _mmd.py), column etree +
# postorder (sp_preorder, skipped in SymmetricMode / NATURAL), then a
# left-looking LU with SuperLU's threshold partial pivoting (dpivotL: largest
# magnitude, the diagonal of Pc A Pc' is kept when |a_diag| >= thresh * max).
# Deviations: no supernodes, so values agree with SuperLU to rounding, ties
# between equal off-diagonal pivot candidates may be broken differently, and
# L/U indices are sorted within each column (SuperLU's are in supernode
# order); `nnz` counts the stored nonzeros of L and U (SuperLU also counts
# structural zeros inside supernodes). spilu uses the same machinery with a
# plain magnitude drop rule (not SuperLU's ILUTP rules).

import numpy as np
from ._colamd import colamd
from .._sparsetools import _arr
from ._mmd import getata, at_plus_a, mmd_order
from .._base import _compressed, _coo, _up, _is_cplx, _issparse
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
        # stored entries of L (with its unit diagonal) and U
        return len(self.Lx) + self.n + len(self.Ux)

    @property
    def L(self):
        # scipy LU_to_csc: unit diagonal stored explicitly, rows = pivot
        # positions (sorted within each column here)
        n = self.n
        Bp = np.zeros(n + 1, dtype=int)
        Bi = List[int]()
        Bx = List[T]()
        for j in range(n):
            pr = List[Tuple[int, T]]()
            for p in range(self.Lp[j], self.Lp[j + 1]):
                pr.append((self.Li[p], self.Lx[p]))
            pr.sort(key=lambda t: t[0])
            Bi.append(j)
            Bx.append(T(1))
            for t in pr:
                Bi.append(t[0])
                Bx.append(t[1])
            Bp[j + 1] = len(Bi)
        return csc_array((_arr(Bx, T), _arr(Bi, int), Bp), shape=(n, n))

    @property
    def U(self):
        n = self.n
        Bp = np.zeros(n + 1, dtype=int)
        Bi = List[int]()
        Bx = List[T]()
        for j in range(n):
            pr = List[Tuple[int, T]]()
            for p in range(self.Up[j], self.Up[j + 1]):
                pr.append((self.Ui[p], self.Ux[p]))
            pr.sort(key=lambda t: t[0])
            for t in pr:
                Bi.append(t[0])
                Bx.append(t[1])
            Bp[j + 1] = len(Bi)
        return csc_array((_arr(Bx, T), _arr(Bi, int), Bp), shape=(n, n))

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
            for i in range(n):
                x[i] = y[self.perm_c[i]]
            return x
        else:
            conj = trans == "H"
            # A^T x = b -> U^T L^T Pr x = Pc^T b
            y = np.empty(n, dtype=T)
            for i in range(n):
                y[self.perm_c[i]] = b[i]
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

def _sp_coletree(colbeg: List[int], colend: List[int], rowind, nr: int,
                 nc: int) -> List[int]:
    # SuperLU sp_coletree: column elimination tree of A (cols in AC order)
    root = [0] * nc
    pp = [0] * nc
    firstcol = [nc] * nr
    for col in range(nc):
        for p in range(colbeg[col], colend[col]):
            row = rowind[p]
            firstcol[row] = min(firstcol[row], col)
    parent = [0] * nc
    for col in range(nc):
        pp[col] = col
        cset = col
        root[cset] = col
        parent[col] = nc
        for p in range(colbeg[col], colend[col]):
            row = firstcol[rowind[p]]
            if row >= col:
                continue
            # find with path halving
            i = row
            q = pp[i]
            gp = pp[q]
            while gp != q:
                pp[i] = gp
                i = gp
                q = pp[i]
                gp = pp[q]
            rset = q
            rroot = root[rset]
            if rroot != col:
                parent[rroot] = col
                pp[cset] = rset
                cset = rset
                root[cset] = col
    return parent

def _tree_postorder(n: int, parent: List[int]) -> List[int]:
    # SuperLU TreePostorder (non-recursive nr_etdfs)
    first_kid = [-1] * (n + 1)
    next_kid = [0] * (n + 1)
    post = [0] * (n + 1)
    for v in range(n - 1, -1, -1):
        dad = parent[v]
        next_kid[v] = first_kid[dad]
        first_kid[dad] = v
    postnum = 0
    current = n
    while postnum != n:
        first = first_kid[current]
        if first == -1:
            post[current] = postnum
            postnum += 1
            nxt = next_kid[current]
            while nxt == -1:
                current = parent[current]
                post[current] = postnum
                postnum += 1
                nxt = next_kid[current]
            if postnum == n + 1:
                return post
            current = nxt
        else:
            current = first
    return post

def _get_perm_c(spec: str, A) -> List[int]:
    m, n = A.shape
    if spec == "NATURAL":
        return list(range(n))
    elif spec == "COLAMD":
        return colamd(m, n, A.indptr, A.indices)
    elif spec == "MMD_ATA":
        bp, bi = getata(m, n, A.indptr, A.indices)
        return mmd_order(n, bp, bi)
    elif spec == "MMD_AT_PLUS_A":
        if m != n:
            raise ValueError("Matrix is not square")
        bp, bi = at_plus_a(n, A.indptr, A.indices)
        return mmd_order(n, bp, bi)
    else:
        raise ValueError(f"Invalid permc_spec: {spec}")
        return List[int]()

def _sp_preorder(A, perm_c: List[int], symmetric: bool) -> List[int]:
    # SuperLU sp_preorder: etree of A*Pc, then postorder it (not in
    # SymmetricMode); returns the final perm_c
    m, n = A.shape
    if symmetric:
        return perm_c
    colbeg = [0] * n
    colend = [0] * n
    for i in range(n):
        colbeg[perm_c[i]] = A.indptr[i]
        colend[perm_c[i]] = A.indptr[i + 1]
    etree = _sp_coletree(colbeg, colend, A.indices, m, n)
    post = _tree_postorder(n, etree)
    out = [0] * n
    for i in range(n):
        out[i] = post[perm_c[i]]
    return out

def _lu_factor(A, T: type, drop_tol: float = 0.0, fill_factor: float = 1e30,
               permc_spec: str = "COLAMD", thresh: float = 1.0,
               symmetric: bool = False):
    # A: CSC with sorted, summed entries. Factor Pr A Pc' = L U with the
    # column order of SuperLU (get_perm_c + sp_preorder).
    m, n = A.shape
    if m != n:
        raise ValueError("can only factor square matrices")
    sym = symmetric or permc_spec == "NATURAL"
    perm_c = _sp_preorder(A, _get_perm_c(permc_spec, A), sym)
    iperm_c = [0] * n
    for i in range(n):
        iperm_c[perm_c[i]] = i
    F = SuperLU[T](n)
    pinv = np.full(n, -1, dtype=int)   # row of A -> pivot position
    x = np.zeros(n, dtype=T)
    mark = np.zeros(n, dtype=bool)
    nz = List[int]()
    for j in range(n):
        nz.clear()
        oc = iperm_c[j]     # original column factored at step j
        for p in range(A.indptr[oc], A.indptr[oc + 1]):
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
        # dpivotL: largest magnitude (first in scan order), then prefer the
        # diagonal (row oc of A) when |a_diag| >= thresh * max
        piv = -1
        best = 0.0
        diag = -1
        for i in nz:
            if pinv[i] < 0:
                a = abs(x[i])
                if a > best:
                    best = a
                    piv = i
                if i == oc:
                    diag = i
        if piv < 0 or best == 0.0:
            raise RuntimeError("Factor is exactly singular")
        if diag >= 0:
            a = abs(x[diag])
            if a != 0.0 and a >= thresh * best:
                piv = diag
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
    F.perm_c = np.array(perm_c, dtype=int)
    return F

def _as_csc(A):
    if _issparse(A):
        C = A.tocsc(copy=True)
    else:
        C = csc_array(np.asarray(A))
    C.sum_duplicates()
    return C

def _is_mat(A) -> Literal[bool]:
    if _issparse(A):
        return A._mstatic()
    else:
        return False

def _ftype(T: type):
    return _up(T, float)

def _spec(permc_spec) -> str:
    if permc_spec is None:
        return "COLAMD"
    else:
        return str(permc_spec)

def _symmode(options) -> bool:
    if options is None:
        return False
    else:
        return bool(options.get("SymmetricMode", False))

def splu(A, permc_spec = None, diag_pivot_thresh = None, relax = None,
         panel_size = None, options = None):
    C = _as_csc(A)
    T = type(_ftype(C.data.dtype))
    th = 1.0 if diag_pivot_thresh is None else float(diag_pivot_thresh)
    return _lu_factor(C.astype(T), T, 0.0, 1e30, _spec(permc_spec), th,
                      _symmode(options))

def spilu(A, drop_tol = None, fill_factor = None, drop_rule = None,
          permc_spec = None, diag_pivot_thresh = None, relax = None,
          panel_size = None, options = None):
    C = _as_csc(A)
    T = type(_ftype(C.data.dtype))
    dt = 1e-4 if drop_tol is None else float(drop_tol)
    th = 0.1 if diag_pivot_thresh is None else float(diag_pivot_thresh)
    return _lu_factor(C.astype(T), T, dt, 1e30, _spec(permc_spec), th,
                      _symmode(options))

def factorized(A):
    return splu(A)

def spsolve(A, b, permc_spec = None, use_umfpack: bool = True):
    C = _as_csc(A)
    m, n = C.shape
    if m != n:
        raise ValueError(f"matrix must be square (has shape {C.shape})")
    if _issparse(b):
        # sparse right-hand side: sparse result (CSC for 2-D)
        B = b.tocsc()
        if B.shape[0] != n:
            raise ValueError("matrix - rhs dimension mismatch")
        T = type(_up(type(_ftype(C.data.dtype)), B.data.dtype))
        F = _lu_factor(C.astype(T), T, 0.0, 1e30, _spec(permc_spec))
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
        F = _lu_factor(C.astype(T), T, 0.0, 1e30, _spec(permc_spec))
        # note: scipy returns a 1-D result for b of shape (n, 1); the result
        # rank must be static here, so a 2-D b always gives a 2-D result
        return F.solve(bv.astype(T))

def _tri_solve1(R, x, lower: bool, unit_diagonal: bool, T: type):
    n = R.shape[0]
    for t in range(n):
        i = t if lower else n - 1 - t
        s = x[i]
        diag = T()
        for p in range(R.indptr[i], R.indptr[i + 1]):
            j = R.indices[p]
            if j == i:
                diag += T(R.data[p])
            elif (j < i) if lower else (j > i):
                s -= T(R.data[p]) * x[j]
        if unit_diagonal:
            x[i] = s
        else:
            if diag == T():
                raise ValueError("A is singular: zero entry on diagonal.")
            x[i] = s / diag

def spsolve_triangular(A, b, lower: bool = True, overwrite_A: bool = False,
                       overwrite_b: bool = False, unit_diagonal: bool = False):
    R = A.tocsr()
    R.sum_duplicates()
    bv = np.asarray(b)
    T = type(_up(type(_ftype(R.data.dtype)), bv.dtype))
    if R.shape[0] != R.shape[1]:
        raise ValueError("A must be a square matrix")
    if bv.ndim == 1:
        x = bv.astype(T).copy()
        _tri_solve1(R, x, lower, unit_diagonal, T)
        return x
    else:
        X = bv.astype(T).copy()
        for c in range(X.shape[1]):
            col = np.ascontiguousarray(X[:, c])
            _tri_solve1(R, col, lower, unit_diagonal, T)
            for i in range(X.shape[0]):
                X[i, c] = col[i]
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

def is_sptriangular(A):
    # scipy 1.18 linsolve.is_sptriangular -> (lower, upper)
    if A._spfmt() == "dia":
        lo = True
        up = True
        for k in range(len(A.offsets)):
            if A.offsets[k] > 0:
                lo = False
            if A.offsets[k] < 0:
                up = False
        return lo, up
    elif A._spfmt() != "bsr":
        c = A.tocoo()
        lo = True
        up = True
        for k in range(len(c.data)):
            if c.col[k] > c.row[k]:
                lo = False
            if c.col[k] < c.row[k]:
                up = False
        return lo, up
    else:
        return is_sptriangular(A.tocsr())

def spbandwidth(A):
    # scipy 1.18 linsolve.spbandwidth -> (below, above)
    if A._spfmt() == "dia":
        mn = 0
        mx = 0
        if len(A.offsets) > 0:
            mn = A.offsets[0]
            mx = A.offsets[0]
        for k in range(len(A.offsets)):
            mn = min(mn, A.offsets[k])
            mx = max(mx, A.offsets[k])
        return max(0, -mn), max(0, mx)
    c = A.tocoo()
    if len(c.data) == 0:
        if A._spfmt() == "dok":
            return 0, 0
        raise ValueError("zero-size array to reduction operation minimum which has no identity")
        return 0, 0
    gmin = c.col[0] - c.row[0]
    gmax = gmin
    for k in range(len(c.data)):
        g = c.col[k] - c.row[k]
        gmin = min(gmin, g)
        gmax = max(gmax, g)
    return max(-gmin, 0), max(gmax, 0)
