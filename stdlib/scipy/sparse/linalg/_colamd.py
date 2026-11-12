# Copyright (C) 2026 the Typhon authors
# Ported from SuiteSparse COLAMD 3.x (BSD-3-Clause)

# COLAMD approximate minimum degree column ordering, as shipped with scipy's
# SuperLU (colamd.c, version 2.9.1) and called by get_perm_c(COLAMD) with
# the default knobs (dense row/col = 10, aggressive absorption). The C union
# fields of Colamd_Col / Colamd_Row are separate int arrays here:
#   Col: c_start, c_len, c_s1 (thickness|parent), c_s2 (score|order),
#        c_s3 (headhash|hash|prev), c_s4 (degree_next|hash_next)
#   Row: r_start, r_len, r_s1 (degree|p), r_s2 (mark|first_column)

import numpy as np

_EMPTY = -1
_INT_MAX = 2147483647

class _Colamd:
    n_row: int
    n_col: int
    A: List[int]
    c_start: List[int]
    c_len: List[int]
    c_s1: List[int]
    c_s2: List[int]
    c_s3: List[int]
    c_s4: List[int]
    r_start: List[int]
    r_len: List[int]
    r_s1: List[int]
    r_s2: List[int]

    def __init__(self, n_row: int, n_col: int, Alen: int):
        self.n_row = n_row
        self.n_col = n_col
        self.A = [0] * Alen
        self.c_start = [0] * (n_col + 1)
        self.c_len = [0] * (n_col + 1)
        self.c_s1 = [0] * (n_col + 1)
        self.c_s2 = [0] * (n_col + 1)
        self.c_s3 = [0] * (n_col + 1)
        self.c_s4 = [0] * (n_col + 1)
        self.r_start = [0] * (n_row + 1)
        self.r_len = [0] * (n_row + 1)
        self.r_s1 = [0] * (n_row + 1)
        self.r_s2 = [0] * (n_row + 1)

    def row_dead(self, r: int) -> bool:
        return self.r_s2[r] < 0

    def row_alive(self, r: int) -> bool:
        return self.r_s2[r] >= 0

    def col_dead(self, c: int) -> bool:
        return self.c_start[c] < 0

    def col_alive(self, c: int) -> bool:
        return self.c_start[c] >= 0

    def kill_row(self, r: int):
        self.r_s2[r] = -1

    def kill_principal_col(self, c: int):
        self.c_start[c] = -1

    def kill_non_principal_col(self, c: int):
        self.c_start[c] = -2

    def init_rows_cols(self, p: List[int]) -> bool:
        n_row = self.n_row
        n_col = self.n_col
        A = self.A
        jumbled = False
        for col in range(n_col):
            self.c_start[col] = p[col]
            self.c_len[col] = p[col + 1] - p[col]
            if self.c_len[col] < 0:
                return False
            self.c_s1[col] = 1
            self.c_s2[col] = 0
            self.c_s3[col] = _EMPTY
            self.c_s4[col] = _EMPTY
        for row in range(n_row):
            self.r_len[row] = 0
            self.r_s2[row] = -1
        for col in range(n_col):
            last_row = -1
            for q in range(p[col], p[col + 1]):
                row = A[q]
                if row < 0 or row >= n_row:
                    return False
                if row <= last_row or self.r_s2[row] == col:
                    jumbled = True
                if self.r_s2[row] != col:
                    self.r_len[row] += 1
                else:
                    self.c_len[col] -= 1
                self.r_s2[row] = col
                last_row = row
        if n_row > 0:
            self.r_start[0] = p[n_col]
            self.r_s1[0] = self.r_start[0]
            self.r_s2[0] = -1
        for row in range(1, n_row):
            self.r_start[row] = self.r_start[row - 1] + self.r_len[row - 1]
            self.r_s1[row] = self.r_start[row]
            self.r_s2[row] = -1
        if jumbled:
            for col in range(n_col):
                for q in range(p[col], p[col + 1]):
                    row = A[q]
                    if self.r_s2[row] != col:
                        A[self.r_s1[row]] = col
                        self.r_s1[row] += 1
                        self.r_s2[row] = col
        else:
            for col in range(n_col):
                for q in range(p[col], p[col + 1]):
                    row = A[q]
                    A[self.r_s1[row]] = col
                    self.r_s1[row] += 1
        for row in range(n_row):
            self.r_s2[row] = 0
            self.r_s1[row] = self.r_len[row]
        if jumbled:
            self.c_start[0] = 0
            p[0] = 0
            for col in range(1, n_col):
                self.c_start[col] = self.c_start[col - 1] + self.c_len[col - 1]
                p[col] = self.c_start[col]
            for row in range(n_row):
                rs = self.r_start[row]
                for q in range(rs, rs + self.r_len[row]):
                    A[p[A[q]]] = row
                    p[A[q]] += 1
        return True

    def init_scoring(self, head: List[int], dense_row_knob: float,
                     dense_col_knob: float):
        n_row = self.n_row
        n_col = self.n_col
        A = self.A
        if dense_row_knob < 0:
            dense_row_count = n_col - 1
        else:
            dense_row_count = int(max(16.0, dense_row_knob * float(n_col) ** 0.5))
        if dense_col_knob < 0:
            dense_col_count = n_row - 1
        else:
            dense_col_count = int(max(16.0, dense_col_knob * float(min(n_row, n_col)) ** 0.5))
        max_deg = 0
        n_col2 = n_col
        n_row2 = n_row
        for c in range(n_col - 1, -1, -1):
            if self.c_len[c] == 0:
                n_col2 -= 1
                self.c_s2[c] = n_col2
                self.kill_principal_col(c)
        for c in range(n_col - 1, -1, -1):
            if self.col_dead(c):
                continue
            deg = self.c_len[c]
            if deg > dense_col_count:
                n_col2 -= 1
                self.c_s2[c] = n_col2
                cs = self.c_start[c]
                for q in range(cs, cs + self.c_len[c]):
                    self.r_s1[A[q]] -= 1
                self.kill_principal_col(c)
        for r in range(n_row):
            deg = self.r_s1[r]
            if deg > dense_row_count or deg == 0:
                self.kill_row(r)
                n_row2 -= 1
            else:
                max_deg = max(max_deg, deg)
        for c in range(n_col - 1, -1, -1):
            if self.col_dead(c):
                continue
            score = 0
            cs = self.c_start[c]
            new_cp = cs
            for q in range(cs, cs + self.c_len[c]):
                row = A[q]
                if self.row_dead(row):
                    continue
                A[new_cp] = row
                new_cp += 1
                score += self.r_s1[row] - 1
                score = min(score, n_col)
            col_length = new_cp - cs
            if col_length == 0:
                n_col2 -= 1
                self.c_s2[c] = n_col2
                self.kill_principal_col(c)
            else:
                self.c_len[c] = col_length
                self.c_s2[c] = score
        for c in range(n_col + 1):
            head[c] = _EMPTY
        for c in range(n_col - 1, -1, -1):
            if self.col_alive(c):
                score = self.c_s2[c]
                next_col = head[score]
                self.c_s3[c] = _EMPTY
                self.c_s4[c] = next_col
                if next_col != _EMPTY:
                    self.c_s3[next_col] = c
                head[score] = c
        return n_row2, n_col2, max_deg

    def clear_mark(self, tag_mark: int, max_mark: int) -> int:
        if tag_mark <= 0 or tag_mark >= max_mark:
            for r in range(self.n_row):
                if self.row_alive(r):
                    self.r_s2[r] = 0
            tag_mark = 1
        return tag_mark

    def find_ordering(self, Alen: int, head: List[int], n_col2: int,
                      max_deg: int, pfree: int, aggressive: bool) -> int:
        n_row = self.n_row
        n_col = self.n_col
        A = self.A
        max_mark = _INT_MAX - n_col
        tag_mark = self.clear_mark(0, max_mark)
        min_score = 0
        ngarbage = 0
        k = 0
        while k < n_col2:
            while min_score < n_col and head[min_score] == _EMPTY:
                min_score += 1
            pivot_col = head[min_score]
            next_col = self.c_s4[pivot_col]
            head[min_score] = next_col
            if next_col != _EMPTY:
                self.c_s3[next_col] = _EMPTY
            pivot_col_score = self.c_s2[pivot_col]
            self.c_s2[pivot_col] = k
            pivot_col_thickness = self.c_s1[pivot_col]
            k += pivot_col_thickness
            needed_memory = min(pivot_col_score, n_col - k)
            if pfree + needed_memory >= Alen:
                pfree = self.garbage_collection(pfree)
                ngarbage += 1
                tag_mark = self.clear_mark(0, max_mark)
            pivot_row_start = pfree
            pivot_row_degree = 0
            self.c_s1[pivot_col] = -pivot_col_thickness
            cs = self.c_start[pivot_col]
            for q in range(cs, cs + self.c_len[pivot_col]):
                row = A[q]
                if self.row_alive(row):
                    rs = self.r_start[row]
                    for t in range(rs, rs + self.r_len[row]):
                        col = A[t]
                        col_thickness = self.c_s1[col]
                        if col_thickness > 0 and self.col_alive(col):
                            self.c_s1[col] = -col_thickness
                            A[pfree] = col
                            pfree += 1
                            pivot_row_degree += col_thickness
            self.c_s1[pivot_col] = pivot_col_thickness
            max_deg = max(max_deg, pivot_row_degree)
            for q in range(cs, cs + self.c_len[pivot_col]):
                self.kill_row(A[q])
            pivot_row_length = pfree - pivot_row_start
            if pivot_row_length > 0:
                pivot_row = A[self.c_start[pivot_col]]
            else:
                pivot_row = _EMPTY
            for t in range(pivot_row_start, pivot_row_start + pivot_row_length):
                col = A[t]
                col_thickness = -self.c_s1[col]
                self.c_s1[col] = col_thickness
                cur_score = self.c_s2[col]
                prev_col = self.c_s3[col]
                next_col = self.c_s4[col]
                if prev_col == _EMPTY:
                    head[cur_score] = next_col
                else:
                    self.c_s4[prev_col] = next_col
                if next_col != _EMPTY:
                    self.c_s3[next_col] = prev_col
                ccs = self.c_start[col]
                for q in range(ccs, ccs + self.c_len[col]):
                    row = A[q]
                    row_mark = self.r_s2[row]
                    if row_mark < 0:
                        continue
                    set_difference = row_mark - tag_mark
                    if set_difference < 0:
                        set_difference = self.r_s1[row]
                    set_difference -= col_thickness
                    if set_difference == 0 and aggressive:
                        self.kill_row(row)
                    else:
                        self.r_s2[row] = set_difference + tag_mark
            for t in range(pivot_row_start, pivot_row_start + pivot_row_length):
                col = A[t]
                hsh = 0
                cur_score = 0
                ccs = self.c_start[col]
                new_cp = ccs
                for q in range(ccs, ccs + self.c_len[col]):
                    row = A[q]
                    row_mark = self.r_s2[row]
                    if row_mark < 0:
                        continue
                    A[new_cp] = row
                    new_cp += 1
                    hsh += row
                    cur_score += row_mark - tag_mark
                    cur_score = min(cur_score, n_col)
                self.c_len[col] = new_cp - ccs
                if self.c_len[col] == 0:
                    self.kill_principal_col(col)
                    pivot_row_degree -= self.c_s1[col]
                    self.c_s2[col] = k
                    k += self.c_s1[col]
                else:
                    self.c_s2[col] = cur_score
                    hsh %= n_col + 1
                    head_column = head[hsh]
                    if head_column > _EMPTY:
                        first_col = self.c_s3[head_column]
                        self.c_s3[head_column] = col
                    else:
                        first_col = -(head_column + 2)
                        head[hsh] = -(col + 2)
                    self.c_s4[col] = first_col
                    self.c_s3[col] = hsh
            self.detect_super_cols(head, pivot_row_start, pivot_row_length)
            self.kill_principal_col(pivot_col)
            tag_mark = self.clear_mark(tag_mark + max_deg + 1, max_mark)
            new_rp = pivot_row_start
            for t in range(pivot_row_start, pivot_row_start + pivot_row_length):
                col = A[t]
                if self.col_dead(col):
                    continue
                A[new_rp] = col
                new_rp += 1
                A[self.c_start[col] + self.c_len[col]] = pivot_row
                self.c_len[col] += 1
                cur_score = self.c_s2[col] + pivot_row_degree
                max_score = n_col - k - self.c_s1[col]
                cur_score -= self.c_s1[col]
                cur_score = min(cur_score, max_score)
                self.c_s2[col] = cur_score
                next_col = head[cur_score]
                self.c_s4[col] = next_col
                self.c_s3[col] = _EMPTY
                if next_col != _EMPTY:
                    self.c_s3[next_col] = col
                head[cur_score] = col
                min_score = min(min_score, cur_score)
            if pivot_row_degree > 0:
                self.r_start[pivot_row] = pivot_row_start
                self.r_len[pivot_row] = new_rp - pivot_row_start
                self.r_s1[pivot_row] = pivot_row_degree
                self.r_s2[pivot_row] = 0
        return ngarbage

    def detect_super_cols(self, head: List[int], row_start: int, row_length: int):
        A = self.A
        for t in range(row_start, row_start + row_length):
            col = A[t]
            if self.col_dead(col):
                continue
            hsh = self.c_s3[col]
            head_column = head[hsh]
            if head_column > _EMPTY:
                first_col = self.c_s3[head_column]
            else:
                first_col = -(head_column + 2)
            super_c = first_col
            while super_c != _EMPTY:
                length = self.c_len[super_c]
                prev_c = super_c
                c = self.c_s4[super_c]
                while c != _EMPTY:
                    if self.c_len[c] != length or self.c_s2[c] != self.c_s2[super_c]:
                        prev_c = c
                        c = self.c_s4[c]
                        continue
                    p1 = self.c_start[super_c]
                    p2 = self.c_start[c]
                    i = 0
                    while i < length:
                        if A[p1 + i] != A[p2 + i]:
                            break
                        i += 1
                    if i != length:
                        prev_c = c
                        c = self.c_s4[c]
                        continue
                    self.c_s1[super_c] += self.c_s1[c]
                    self.c_s1[c] = super_c
                    self.kill_non_principal_col(c)
                    self.c_s2[c] = _EMPTY
                    self.c_s4[prev_c] = self.c_s4[c]
                    c = self.c_s4[c]
                super_c = self.c_s4[super_c]
            if head_column > _EMPTY:
                self.c_s3[head_column] = _EMPTY
            else:
                head[hsh] = _EMPTY

    def garbage_collection(self, pfree: int) -> int:
        A = self.A
        pdest = 0
        for c in range(self.n_col):
            if self.col_alive(c):
                psrc = self.c_start[c]
                self.c_start[c] = pdest
                length = self.c_len[c]
                for j in range(length):
                    r = A[psrc + j]
                    if self.row_alive(r):
                        A[pdest] = r
                        pdest += 1
                self.c_len[c] = pdest - self.c_start[c]
        for r in range(self.n_row):
            if self.row_dead(r) or self.r_len[r] == 0:
                self.kill_row(r)
            else:
                psrc = self.r_start[r]
                self.r_s2[r] = A[psrc]
                A[psrc] = -r - 1
        psrc = pdest
        while psrc < pfree:
            v = A[psrc]
            psrc += 1
            if v < 0:
                psrc -= 1
                r = -A[psrc] - 1
                A[psrc] = self.r_s2[r]
                self.r_start[r] = pdest
                length = self.r_len[r]
                for j in range(length):
                    c = A[psrc]
                    psrc += 1
                    if self.col_alive(c):
                        A[pdest] = c
                        pdest += 1
                self.r_len[r] = pdest - self.r_start[r]
        return pdest

    def order_children(self, p: List[int]):
        n_col = self.n_col
        for i in range(n_col):
            if self.c_start[i] != -1 and self.c_s2[i] == _EMPTY:
                parent = i
                while True:
                    parent = self.c_s1[parent]
                    if self.c_start[parent] == -1:
                        break
                c = i
                order = self.c_s2[parent]
                while True:
                    self.c_s2[c] = order
                    order += 1
                    self.c_s1[c] = parent
                    c = self.c_s1[c]
                    if self.c_s2[c] != _EMPTY:
                        break
                self.c_s2[parent] = order
        for c in range(n_col):
            p[self.c_s2[c]] = c

def colamd(n_row: int, n_col: int, colptr, rowind,
           dense_row: float = 10.0, dense_col: float = 10.0,
           aggressive: bool = True) -> List[int]:
    # SuperLU get_colamd: returns perm_c with perm_c[p[i]] = i
    nnz = colptr[n_col]
    col_size = 6 * (n_col + 1)
    row_size = 4 * (n_row + 1)
    Alen = 2 * nnz + col_size + row_size + n_col + nnz // 5
    Alen -= col_size + row_size
    S = _Colamd(n_row, n_col, Alen)
    p = [0] * (n_col + 1)
    for i in range(n_col + 1):
        p[i] = int(colptr[i])
    for i in range(nnz):
        S.A[i] = int(rowind[i])
    if not S.init_rows_cols(p):
        raise RuntimeError("COLAMD failed")
    # (colamd.c passes p itself as `head`; p is not read again after
    # init_rows_cols, so a separate array is equivalent)
    head = [0] * (n_col + 1)
    n_row2, n_col2, max_deg = S.init_scoring(head, dense_row, dense_col)
    S.find_ordering(Alen, head, n_col2, max_deg, 2 * nnz, aggressive)
    S.order_children(p)
    perm_c = [0] * n_col
    for i in range(n_col):
        perm_c[p[i]] = i
    return perm_c
