# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# Typed Cython port of bench/codon/fannkuch.py.
# Entry point: module-level code runs at import (see run.py); reads sys.argv[1].
from sys import argv
from time import time
from libc.stdlib cimport malloc, free


cdef long long fact(int n):
    cdef long long r = 1
    cdef int k
    for k in range(2, n + 1):
        r *= k
    return r


cdef void perm(int n, long long i, int *p):
    cdef int k, j
    cdef long long f
    for k in range(n):
        f = fact(n - 1 - k)
        p[k] = <int>(i // f)
        i = i % f

    for k in range(n - 1, -1, -1):
        for j in range(k - 1, -1, -1):
            if p[j] <= p[k]:
                p[k] += 1


cdef int run_fannkuch(int n):
    cdef int max_flips = 0
    cdef long long idx, total = fact(n)
    cdef int flips, k, i, j, tmp
    cdef int *p = <int *>malloc(n * sizeof(int))
    for idx in range(total):
        perm(n, idx, p)
        flips = 0
        k = p[0]
        while k:
            i = 0
            j = k
            while i < j:
                tmp = p[i]
                p[i] = p[j]
                p[j] = tmp
                i += 1
                j -= 1
            k = p[0]
            flips += 1
        if flips > max_flips:
            max_flips = flips
    free(p)
    return max_flips


n = int(argv[1])

t0 = time()
max_flips = run_fannkuch(n)
print(f'Pfannkuchen({n}) = {max_flips}')
t1 = time()
print(t1 - t0)
