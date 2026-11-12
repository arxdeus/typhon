# Copyright (C) 2026 the Typhon authors
# Ported from scipy 1.18.1 (BSD-3-Clause)

# Port of the scipy/sparse/sparsetools C++ kernels (csr.h, coo.h). Index
# arrays are int (int64); the visiting order of every kernel matches the C++
# code so that unsorted results (e.g. of A @ B or A + B on non-canonical
# inputs) have the same index order as scipy.

import numpy as np

def _arr(lst, T: type):
    # _arr(list, T) that also handles an empty list
    a = np.empty(len(lst), dtype=T)
    for i in range(len(lst)):
        a[i] = T(lst[i])
    return a

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
    return Cp, _arr(Cj, int), _arr(Cx, T)

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

# --- BSR kernels (bsr.h, dense.h). Block data is passed flat (C order) ---

def csr_count_blocks(n_row: int, n_col: int, R: int, C: int,
                     Ap: np.ndarray[int, 1], Aj: np.ndarray[int, 1]) -> int:
    mask = np.full(n_col // C + 1, -1, dtype=int)
    n_blks = 0
    for i in range(n_row):
        bi = i // R
        for jj in range(Ap[i], Ap[i + 1]):
            bj = Aj[jj] // C
            if mask[bj] != bi:
                mask[bj] = bi
                n_blks += 1
    return n_blks

def csr_tobsr(n_row: int, n_col: int, R: int, C: int, Ap: np.ndarray[int, 1],
              Aj: np.ndarray[int, 1], Ax: np.ndarray[T, 1], T: type):
    nblks = csr_count_blocks(n_row, n_col, R, C, Ap, Aj)
    RC = R * C
    n_brow = n_row // R
    Bp = np.zeros(n_brow + 1, dtype=int)
    Bj = np.empty(nblks, dtype=int)
    Bx = np.zeros(nblks * RC, dtype=T)
    blocks = np.full(n_col // C + 1, -1, dtype=int)
    n_blks = 0
    for bi in range(n_brow):
        for r in range(R):
            i = R * bi + r
            for jj in range(Ap[i], Ap[i + 1]):
                j = Aj[jj]
                bj = j // C
                c = j % C
                if blocks[bj] < 0:
                    blocks[bj] = RC * n_blks
                    Bj[n_blks] = bj
                    n_blks += 1
                Bx[blocks[bj] + C * r + c] += Ax[jj]
        for jj in range(Ap[R * bi], Ap[R * (bi + 1)]):
            blocks[Aj[jj] // C] = -1
        Bp[bi + 1] = n_blks
    return Bp, Bj, Bx

def bsr_tocsr(n_brow: int, n_bcol: int, R: int, C: int, Ap: np.ndarray[int, 1],
              Aj: np.ndarray[int, 1], Ax: np.ndarray[T, 1], T: type):
    RC = R * C
    nnz = Ap[n_brow] * RC
    Bp = np.zeros(n_brow * R + 1, dtype=int)
    Bj = np.empty(nnz, dtype=int)
    Bx = np.empty(nnz, dtype=T)
    Bp[n_brow * R] = nnz
    for brow in range(n_brow):
        brow_size = Ap[brow + 1] - Ap[brow]
        row_size = C * brow_size
        for r in range(R):
            row = R * brow + r
            Bp[row] = RC * Ap[brow] + r * row_size
            for bjj in range(brow_size):
                b_ind = Ap[brow] + bjj
                bcol = Aj[b_ind]
                for c in range(C):
                    d = Bp[row] + bjj * C + c
                    Bj[d] = C * bcol + c
                    Bx[d] = Ax[RC * b_ind + C * r + c]
    return Bp, Bj, Bx

def bsr_transpose(n_brow: int, n_bcol: int, R: int, C: int,
                  Ap: np.ndarray[int, 1], Aj: np.ndarray[int, 1],
                  Ax: np.ndarray[T, 1], T: type):
    nblks = Ap[n_brow]
    RC = R * C
    perm_in = np.arange(nblks)
    Bp, Bj, perm_out = csr_tocsc(n_brow, n_bcol, Ap, Aj, perm_in)
    Bx = np.empty(nblks * RC, dtype=T)
    for i in range(nblks):
        a0 = RC * perm_out[i]
        b0 = RC * i
        for r in range(R):
            for c in range(C):
                Bx[b0 + c * R + r] = Ax[a0 + r * C + c]
    return Bp, Bj, Bx

def bsr_sort_indices(n_brow: int, n_bcol: int, R: int, C: int,
                     Ap: np.ndarray[int, 1], Aj: np.ndarray[int, 1],
                     Ax: np.ndarray[T, 1], T: type):
    # in place
    nblks = Ap[n_brow]
    RC = R * C
    perm = np.arange(nblks)
    csr_sort_indices(n_brow, Ap, Aj, perm)
    Ac = Ax.copy()
    for i in range(nblks):
        for q in range(RC):
            Ax[RC * i + q] = Ac[RC * perm[i] + q]

def bsr_matvec(n_brow: int, n_bcol: int, R: int, C: int, Ap: np.ndarray[int, 1],
               Aj: np.ndarray[int, 1], Ax: np.ndarray[T, 1], x, y, T: type):
    RC = R * C
    for i in range(n_brow):
        for jj in range(Ap[i], Ap[i + 1]):
            j = Aj[jj]
            for r in range(R):
                dot = y[R * i + r]
                for c in range(C):
                    dot += Ax[RC * jj + C * r + c] * x[C * j + c]
                y[R * i + r] = dot

def bsr_matvecs(n_brow: int, n_bcol: int, n_vecs: int, R: int, C: int,
                Ap: np.ndarray[int, 1], Aj: np.ndarray[int, 1],
                Ax: np.ndarray[T, 1], X, Y, T: type):
    # X: (n_bcol*C, n_vecs), Y: (n_brow*R, n_vecs) 2-D arrays
    RC = R * C
    for i in range(n_brow):
        for jj in range(Ap[i], Ap[i + 1]):
            j = Aj[jj]
            for r in range(R):
                for v in range(n_vecs):
                    dot = Y[R * i + r, v]
                    for c in range(C):
                        dot += Ax[RC * jj + C * r + c] * X[C * j + c, v]
                    Y[R * i + r, v] = dot

def bsr_matmat(n_brow: int, n_bcol: int, R: int, C: int, N: int,
               Ap: np.ndarray[int, 1], Aj: np.ndarray[int, 1], Ax: np.ndarray[T, 1],
               Bp: np.ndarray[int, 1], Bj: np.ndarray[int, 1], Bx: np.ndarray[T, 1],
               T: type):
    # A blocks R x N, B blocks N x C; result blocks R x C. Explicit zero
    # blocks are kept (as in scipy).
    RC = R * C
    RN = R * N
    NC = N * C
    nxt = np.full(n_bcol, -1, dtype=int)
    mats = np.zeros(n_bcol, dtype=int)
    Cp = np.zeros(n_brow + 1, dtype=int)
    Cj = List[int]()
    Cx = List[T]()
    nnz = 0
    for i in range(n_brow):
        head = -2
        length = 0
        for jj in range(Ap[i], Ap[i + 1]):
            j = Aj[jj]
            for kk in range(Bp[j], Bp[j + 1]):
                k = Bj[kk]
                if nxt[k] == -1:
                    nxt[k] = head
                    head = k
                    Cj.append(k)
                    mats[k] = RC * nnz
                    for _ in range(RC):
                        Cx.append(T())
                    nnz += 1
                    length += 1
                m0 = mats[k]
                for r in range(R):
                    for c in range(C):
                        dot = Cx[m0 + C * r + c]
                        for d in range(N):
                            dot += Ax[jj * RN + N * r + d] * Bx[kk * NC + C * d + c]
                        Cx[m0 + C * r + c] = dot
        for _ in range(length):
            temp = head
            head = nxt[head]
            nxt[temp] = -1
        Cp[i + 1] = nnz
    return Cp, _arr(Cj, int), _arr(Cx, T)

def _nonzero_block(Cx, start: int, RC: int, T: type) -> bool:
    for n in range(RC):
        if Cx[start + n] != T():
            return True
    return False

def bsr_binop_bsr(n_brow: int, n_bcol: int, R: int, C: int,
                  Ap: np.ndarray[int, 1], Aj: np.ndarray[int, 1], Ax: np.ndarray[T, 1],
                  Bp: np.ndarray[int, 1], Bj: np.ndarray[int, 1], Bx: np.ndarray[T, 1],
                  op: Literal[int], T: type):
    # canonical inputs required (bsr_binop_bsr_canonical); general inputs
    # are canonicalized by the caller (scipy's general path gives the same
    # values, in a different block order)
    RC = R * C
    Cp = np.zeros(n_brow + 1, dtype=int)
    Cj = List[int]()
    Cx = np.empty(RC * (len(Aj) + len(Bj)), dtype=T)
    z = T()
    nnz = 0
    for i in range(n_brow):
        A_pos = Ap[i]
        B_pos = Bp[i]
        A_end = Ap[i + 1]
        B_end = Bp[i + 1]
        while A_pos < A_end or B_pos < B_end:
            if A_pos < A_end and B_pos < B_end and Aj[A_pos] == Bj[B_pos]:
                for n in range(RC):
                    Cx[RC * nnz + n] = _apply(op, Ax[RC * A_pos + n], Bx[RC * B_pos + n])
                j = Aj[A_pos]
                A_pos += 1
                B_pos += 1
            elif A_pos < A_end and (B_pos >= B_end or Aj[A_pos] < Bj[B_pos]):
                for n in range(RC):
                    Cx[RC * nnz + n] = _apply(op, Ax[RC * A_pos + n], z)
                j = Aj[A_pos]
                A_pos += 1
            else:
                for n in range(RC):
                    Cx[RC * nnz + n] = _apply(op, z, Bx[RC * B_pos + n])
                j = Bj[B_pos]
                B_pos += 1
            if _nonzero_block(Cx, RC * nnz, RC, T):
                Cj.append(j)
                nnz += 1
        Cp[i + 1] = nnz
    return Cp, _arr(Cj, int), Cx[:RC * nnz].copy()

def bsr_diagonal(k: int, n_brow: int, n_bcol: int, R: int, C: int,
                 Ap: np.ndarray[int, 1], Aj: np.ndarray[int, 1],
                 Ax: np.ndarray[T, 1], T: type):
    RC = R * C
    rows = n_brow * R
    cols = n_bcol * C
    D = min(rows + min(k, 0), cols - max(k, 0))
    Yx = np.zeros(max(D, 0), dtype=T)
    if D <= 0:
        return Yx
    first_row = -k if k < 0 else 0
    first_brow = first_row // R
    last_brow = (first_row + D - 1) // R + 1
    for brow in range(first_brow, last_brow):
        first_bcol = (brow * R + k) // C
        last_bcol = ((brow + 1) * R + k - 1) // C + 1
        for jj in range(Ap[brow], Ap[brow + 1]):
            bcol = Aj[jj]
            if first_bcol <= bcol and bcol < last_bcol:
                block_k = brow * R + k - bcol * C
                block_D = min(R + min(block_k, 0), C - max(block_k, 0))
                block_first_row = -block_k if block_k < 0 else 0
                Y_idx = brow * R + block_first_row - first_row
                Ax_idx = RC * jj + (block_k if block_k >= 0 else -C * block_k)
                for kk in range(block_D):
                    Yx[Y_idx + kk] += Ax[Ax_idx + kk * (C + 1)]
    return Yx

# --- DIA kernels (dia.h). data is the 2-D (n_diags, L) array ---

def dia_matvec(n_row: int, n_col: int, n_diags: int, L: int,
               offsets: np.ndarray[int, 1], data: np.ndarray[T, 2], x, y, T: type):
    for i in range(n_diags):
        k = offsets[i]
        i_start = max(0, -k)
        j_start = max(0, k)
        j_end = min(min(n_row + k, n_col), L)
        for n in range(j_end - j_start):
            y[i_start + n] += data[i, j_start + n] * x[j_start + n]

def dia_matvecs(A_rows: int, A_cols: int, A_diags: int, A_L: int,
                offsets: np.ndarray[int, 1], data: np.ndarray[T, 2],
                B_cols: int, B, Y, T: type):
    k_end = min(A_cols, A_L)
    for i in range(A_rows):
        for n in range(A_diags):
            k = i + offsets[n]
            if k < 0 or k >= k_end:
                continue
            a = data[n, k]
            for j in range(B_cols):
                Y[i, j] += a * B[k, j]

def dia_tocsr(n_rows: int, n_cols: int, n_diags: int, L: int,
              offsets: np.ndarray[int, 1], data: np.ndarray[T, 2],
              order: np.ndarray[int, 1], T: type):
    j_end = min(L, n_cols)
    indptr = np.zeros(n_rows + 1, dtype=int)
    indices = List[int]()
    vals = List[T]()
    for i in range(n_rows):
        for k in range(n_diags):
            n = order[k]
            j = i + offsets[n]
            if j < 0 or j >= j_end:
                continue
            x = data[n, j]
            if x != T():
                indices.append(j)
                vals.append(x)
        indptr[i + 1] = len(indices)
    return indptr, _arr(indices, int), _arr(vals, T)

def dia_matmat(A_rows: int, A_cols: int, A_diags: int, A_L: int,
               A_offsets: np.ndarray[int, 1], A_data: np.ndarray[T, 2],
               B_cols: int, B_diags: int, B_L: int,
               B_offsets: np.ndarray[int, 1], B_data: np.ndarray[T, 2], T: type):
    B_rows = A_cols
    rows = A_rows
    cols = B_cols
    L = min(cols, B_L)
    amin = A_offsets[0]
    amax = A_offsets[0]
    for i in range(A_diags):
        amin = min(amin, A_offsets[i])
        amax = max(amax, A_offsets[i])
    bmin = B_offsets[0]
    bmax = B_offsets[0]
    for i in range(B_diags):
        bmin = min(bmin, B_offsets[i])
        bmax = max(bmax, B_offsets[i])
    min_map = amin + bmin
    max_map = amax + bmax
    min_ofs = max(min_map, 1 - rows)
    max_ofs = min(max_map, L - 1)
    ofs_map = np.zeros(max_map - min_map + 1, dtype=int)
    for i in range(A_diags):
        for j in range(B_diags):
            ofs_map[A_offsets[i] + B_offsets[j] - min_map] = 1
    offs = List[int]()
    for ofs in range(min_ofs, max_ofs + 1):
        if ofs_map[ofs - min_map] != 0:
            ofs_map[ofs - min_map] = len(offs)
            offs.append(ofs)
    N_ofs = len(offs)
    data = np.zeros((N_ofs, max(L, 0)), dtype=T)
    for B_i in range(B_diags):
        B_ofs = B_offsets[B_i]
        B_j_beg = max(-B_ofs, 0)
        B_j_end = min(min(B_cols, B_L) - B_ofs, B_rows)
        for A_i in range(A_diags):
            A_ofs = A_offsets[A_i]
            ofs = A_ofs + B_ofs
            if ofs < min_ofs or ofs > max_ofs:
                continue
            row = ofs_map[ofs - min_map]
            j_beg = max(B_j_beg, A_ofs)
            j_end = min(min(B_j_end, A_cols), min(A_L, A_rows + A_ofs))
            for j in range(j_beg, j_end):
                data[row, j + B_ofs] += A_data[A_i, j] * B_data[B_i, j + B_ofs]
    return _arr(offs, int), data
