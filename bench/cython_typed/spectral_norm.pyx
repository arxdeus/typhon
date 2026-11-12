# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# Typed Cython port of bench/codon/spectral_norm.py (no argv; 100 loops fixed).
# Entry point: module-level code runs at import (see run.py).
# Optional env SPECTRAL_LOOPS overrides the loop count for quick checks only.
# Same algorithm (dense A^T A u products, 10 power iterations per loop), but
# vectors are C double arrays instead of Python lists of floats.
import os
from time import time
from libc.stdlib cimport malloc, free

DEF DEFAULT_N = 260


cdef inline double eval_A(long i, long j) noexcept nogil:
    return 1.0 / ((i + j) * (i + j + 1) // 2 + i + 1)


cdef void part_A_times_u(const double *u, double *out) noexcept nogil:
    cdef long i, j
    cdef double partial_sum
    for i in range(DEFAULT_N):
        partial_sum = 0.
        for j in range(DEFAULT_N):
            partial_sum += eval_A(i, j) * u[j]
        out[i] = partial_sum


cdef void part_At_times_u(const double *u, double *out) noexcept nogil:
    cdef long i, j
    cdef double partial_sum
    for i in range(DEFAULT_N):
        partial_sum = 0.
        for j in range(DEFAULT_N):
            partial_sum += eval_A(j, i) * u[j]
        out[i] = partial_sum


cdef void eval_AtA_times_u(const double *u, double *out, double *tmp) noexcept nogil:
    part_A_times_u(u, tmp)
    part_At_times_u(tmp, out)


cdef double bench_spectral_norm(int loops):
    cdef double *u = <double *>malloc(DEFAULT_N * sizeof(double))
    cdef double *v = <double *>malloc(DEFAULT_N * sizeof(double))
    cdef double *tmp = <double *>malloc(DEFAULT_N * sizeof(double))
    cdef double total = 0., vBv, vv
    cdef int k, dummy, i
    for k in range(loops):
        for i in range(DEFAULT_N):
            u[i] = 1.
        for dummy in range(10):
            eval_AtA_times_u(u, v, tmp)
            eval_AtA_times_u(v, u, tmp)
        vBv = vv = 0.
        for i in range(DEFAULT_N):
            vBv += u[i] * v[i]
            vv += v[i] * v[i]
        total += vBv + vv
    free(u)
    free(v)
    free(tmp)
    return total


t0 = time()
print(bench_spectral_norm(int(os.environ.get("SPECTRAL_LOOPS", 100))))
t1 = time()
print(t1 - t0)
