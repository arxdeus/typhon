# Copyright (C) 2026 the Typhon authors
# Ported from scipy 1.18.1 (BSD-3-Clause)

# Port of the scipy/sparse/sparsetools C++ kernels (csr.h, coo.h). Index
# arrays are int (int64); the visiting order of every kernel matches the C++
# code so that unsorted results (e.g. of A @ B or A + B on non-canonical
# inputs) have the same index order as scipy.

import numpy as np

def _zeros_like_t(n: int, T: type):
    return np.zeros(n, dtype=T)

def coo_tocsr(n_row: int, n_col: int, Ai: np.ndarray[int, 1],
              Aj: np.ndarray[int, 1], Ax: np.ndarray[T, 1], T: type):
    nnz = len(Ax)
    Bp = np.zeros(n_row + 1, dtype=int)
    Bj = np.empty(nnz, dtype=int)
    Bx = np.empty(nnz, dtype=T)
    for n in range(nnz):
        Bp[Ai[n]] += 1
    cumsum = 0
    for i in range(n_row):
        temp = Bp[i]
        Bp[i] = cumsum
        cumsum += temp
    Bp[n_row] = nnz
    for n in range(nnz):
        row = Ai[n]
        dest = Bp[row]
        Bj[dest] = Aj[n]
        Bx[dest] = Ax[n]
        Bp[row] += 1
    last = 0
    for i in range(n_row + 1):
        temp = Bp[i]
        Bp[i] = last
        last = temp
    return Bp, Bj, Bx

def csr_tocsc(n_row: int, n_col: int, Ap: np.ndarray[int, 1],
              Aj: np.ndarray[int, 1], Ax: np.ndarray[T, 1], T: type):
    nnz = Ap[n_row]
    Bp = np.zeros(n_col + 1, dtype=int)
    Bi = np.empty(nnz, dtype=int)
    Bx = np.empty(nnz, dtype=T)
    for n in range(nnz):
        Bp[Aj[n]] += 1
    cumsum = 0
    for col in range(n_col):
        temp = Bp[col]
        Bp[col] = cumsum
        cumsum += temp
    Bp[n_col] = nnz
    for row in range(n_row):
        for jj in range(Ap[row], Ap[row + 1]):
            col = Aj[jj]
            dest = Bp[col]
            Bi[dest] = row
            Bx[dest] = Ax[jj]
            Bp[col] += 1
    last = 0
    for col in range(n_col + 1):
        temp = Bp[col]
        Bp[col] = last
        last = temp
    return Bp, Bi, Bx

def expandptr(n_row: int, Ap: np.ndarray[int, 1]):
    Bi = np.empty(Ap[n_row], dtype=int)
    for i in range(n_row):
        for jj in range(Ap[i], Ap[i + 1]):
            Bi[jj] = i
    return Bi

def csr_has_sorted_indices(n_row: int, Ap: np.ndarray[int, 1],
                           Aj: np.ndarray[int, 1]) -> bool:
    for i in range(n_row):
        for jj in range(Ap[i], Ap[i + 1] - 1):
            if Aj[jj] > Aj[jj + 1]:
                return False
    return True

def csr_has_canonical_format(n_row: int, Ap: np.ndarray[int, 1],
                             Aj: np.ndarray[int, 1]) -> bool:
    for i in range(n_row):
        if Ap[i] > Ap[i + 1]:
            return False
        for jj in range(Ap[i] + 1, Ap[i + 1]):
            if not (Aj[jj - 1] < Aj[jj]):
                return False
    return True

def csr_sort_indices(n_row: int, Ap: np.ndarray[int, 1],
                     Aj: np.ndarray[int, 1], Ax: np.ndarray[T, 1], T: type):
    # in place; stable by column within each row
    for i in range(n_row):
        a = Ap[i]
        b = Ap[i + 1]
        # insertion sort (rows are short); stable
        for k in range(a + 1, b):
            j = Aj[k]
            x = Ax[k]
            m = k - 1
            while m >= a and Aj[m] > j:
                Aj[m + 1] = Aj[m]
                Ax[m + 1] = Ax[m]
                m -= 1
            Aj[m + 1] = j
            Ax[m + 1] = x

def csr_sum_duplicates(n_row: int, Ap: np.ndarray[int, 1],
                       Aj: np.ndarray[int, 1], Ax: np.ndarray[T, 1], T: type) -> int:
    nnz = 0
    row_end = 0
    for i in range(n_row):
        jj = row_end
        row_end = Ap[i + 1]
        while jj < row_end:
            j = Aj[jj]
            x = Ax[jj]
            jj += 1
            while jj < row_end and Aj[jj] == j:
                x += Ax[jj]
                jj += 1
            Aj[nnz] = j
            Ax[nnz] = x
            nnz += 1
        Ap[i + 1] = nnz
    return nnz

def csr_eliminate_zeros(n_row: int, Ap: np.ndarray[int, 1],
                        Aj: np.ndarray[int, 1], Ax: np.ndarray[T, 1], T: type) -> int:
    nnz = 0
    row_end = 0
    for i in range(n_row):
        jj = row_end
        row_end = Ap[i + 1]
        while jj < row_end:
            j = Aj[jj]
            x = Ax[jj]
            if x != T():
                Aj[nnz] = j
                Ax[nnz] = x
                nnz += 1
            jj += 1
        Ap[i + 1] = nnz
    return nnz

def _apply(op: Literal[int], a, b):
    if op == 0:
        return a + b
    elif op == 1:
        return a - b
    elif op == 2:
        return a * b
    elif op == 3:
        return a / b
    elif op == 4:
        return a if a > b else b
    else:
        return a if a < b else b

def csr_binop_csr(n_row: int, n_col: int,
                  Ap: np.ndarray[int, 1], Aj: np.ndarray[int, 1], Ax: np.ndarray[T, 1],
                  Bp: np.ndarray[int, 1], Bj: np.ndarray[int, 1], Bx: np.ndarray[T, 1],
                  op: Literal[int], T: type):
    maxnnz = len(Ax) + len(Bx)
    Cp = np.zeros(n_row + 1, dtype=int)
    Cj = np.empty(maxnnz, dtype=int)
    Cx = np.empty(maxnnz, dtype=T)
    zero = T()
    nnz = 0
    if (csr_has_canonical_format(n_row, Ap, Aj) and
            csr_has_canonical_format(n_row, Bp, Bj)):
        for i in range(n_row):
            A_pos = Ap[i]
            B_pos = Bp[i]
            A_end = Ap[i + 1]
            B_end = Bp[i + 1]
            while A_pos < A_end and B_pos < B_end:
                A_j = Aj[A_pos]
                B_j = Bj[B_pos]
                if A_j == B_j:
                    result = _apply(op, Ax[A_pos], Bx[B_pos])
                    if result != zero:
                        Cj[nnz] = A_j
                        Cx[nnz] = result
                        nnz += 1
                    A_pos += 1
                    B_pos += 1
                elif A_j < B_j:
                    result = _apply(op, Ax[A_pos], zero)
                    if result != zero:
                        Cj[nnz] = A_j
                        Cx[nnz] = result
                        nnz += 1
                    A_pos += 1
                else:
                    result = _apply(op, zero, Bx[B_pos])
                    if result != zero:
                        Cj[nnz] = B_j
                        Cx[nnz] = result
                        nnz += 1
                    B_pos += 1
            while A_pos < A_end:
                result = _apply(op, Ax[A_pos], zero)
                if result != zero:
                    Cj[nnz] = Aj[A_pos]
                    Cx[nnz] = result
                    nnz += 1
                A_pos += 1
            while B_pos < B_end:
                result = _apply(op, zero, Bx[B_pos])
                if result != zero:
                    Cj[nnz] = Bj[B_pos]
                    Cx[nnz] = result
                    nnz += 1
                B_pos += 1
            Cp[i + 1] = nnz
    else:
        nxt = np.full(n_col, -1, dtype=int)
        A_row = np.zeros(n_col, dtype=T)
        B_row = np.zeros(n_col, dtype=T)
        for i in range(n_row):
            head = -2
            length = 0
            for jj in range(Ap[i], Ap[i + 1]):
                j = Aj[jj]
                A_row[j] += Ax[jj]
                if nxt[j] == -1:
                    nxt[j] = head
                    head = j
                    length += 1
            for jj in range(Bp[i], Bp[i + 1]):
                j = Bj[jj]
                B_row[j] += Bx[jj]
                if nxt[j] == -1:
                    nxt[j] = head
                    head = j
                    length += 1
            for _ in range(length):
                result = _apply(op, A_row[head], B_row[head])
                if result != zero:
                    Cj[nnz] = head
                    Cx[nnz] = result
                    nnz += 1
                temp = head
                head = nxt[head]
                nxt[temp] = -1
                A_row[temp] = zero
                B_row[temp] = zero
            Cp[i + 1] = nnz
    return Cp, Cj[:nnz].copy(), Cx[:nnz].copy()

def csr_matmat(n_row: int, n_col: int,
               Ap: np.ndarray[int, 1], Aj: np.ndarray[int, 1], Ax: np.ndarray[T, 1],
               Bp: np.ndarray[int, 1], Bj: np.ndarray[int, 1], Bx: np.ndarray[T, 1],
               T: type):
    nxt = np.full(n_col, -1, dtype=int)
    sums = np.zeros(n_col, dtype=T)
    zero = T()
    Cp = np.zeros(n_row + 1, dtype=int)
    Cj = List[int]()
    Cx = List[T]()
    nnz = 0
    for i in range(n_row):
        head = -2
        length = 0
        for jj in range(Ap[i], Ap[i + 1]):
            j = Aj[jj]
            v = Ax[jj]
            for kk in range(Bp[j], Bp[j + 1]):
                k = Bj[kk]
                sums[k] += v * Bx[kk]
                if nxt[k] == -1:
                    nxt[k] = head
                    head = k
                    length += 1
        for _ in range(length):
            if sums[head] != zero:
                Cj.append(head)
                Cx.append(sums[head])
                nnz += 1
            temp = head
            head = nxt[head]
            nxt[temp] = -1
            sums[temp] = zero
        Cp[i + 1] = nnz
    return Cp, np.array(Cj, dtype=int), np.array(Cx, dtype=T)

def csr_matvec(n_row: int, Ap: np.ndarray[int, 1], Aj: np.ndarray[int, 1],
               Ax: np.ndarray[T, 1], x, y, T: type):
    for i in range(n_row):
        s = y[i]
        for jj in range(Ap[i], Ap[i + 1]):
            s += Ax[jj] * x[Aj[jj]]
        y[i] = s

def csc_matvec(n_col: int, Ap: np.ndarray[int, 1], Ai: np.ndarray[int, 1],
               Ax: np.ndarray[T, 1], x, y, T: type):
    for j in range(n_col):
        for ii in range(Ap[j], Ap[j + 1]):
            y[Ai[ii]] += Ax[ii] * x[j]

def csr_matvecs(n_row: int, n_vecs: int, Ap: np.ndarray[int, 1],
                Aj: np.ndarray[int, 1], Ax: np.ndarray[T, 1], X, Y, T: type):
    for i in range(n_row):
        for jj in range(Ap[i], Ap[i + 1]):
            j = Aj[jj]
            a = Ax[jj]
            for k in range(n_vecs):
                Y[i, k] += a * X[j, k]

def csc_matvecs(n_col: int, n_vecs: int, Ap: np.ndarray[int, 1],
                Ai: np.ndarray[int, 1], Ax: np.ndarray[T, 1], X, Y, T: type):
    for j in range(n_col):
        for ii in range(Ap[j], Ap[j + 1]):
            i = Ai[ii]
            a = Ax[ii]
            for k in range(n_vecs):
                Y[i, k] += a * X[j, k]

def csr_diagonal(k: int, n_row: int, n_col: int, Ap: np.ndarray[int, 1],
                 Aj: np.ndarray[int, 1], Ax: np.ndarray[T, 1], T: type):
    first_row = -k if k < 0 else 0
    first_col = k if k > 0 else 0
    N = min(n_row - first_row, n_col - first_col)
    Yx = np.zeros(max(N, 0), dtype=T)
    for i in range(N):
        row = first_row + i
        col = first_col + i
        s = T()
        for jj in range(Ap[row], Ap[row + 1]):
            if Aj[jj] == col:
                s += Ax[jj]
        Yx[i] = s
    return Yx

def csr_todense(n_row: int, n_col: int, Ap: np.ndarray[int, 1],
                Aj: np.ndarray[int, 1], Ax: np.ndarray[T, 1], T: type):
    B = np.zeros((n_row, n_col), dtype=T)
    for i in range(n_row):
        for jj in range(Ap[i], Ap[i + 1]):
            B[i, Aj[jj]] += Ax[jj]
    return B
