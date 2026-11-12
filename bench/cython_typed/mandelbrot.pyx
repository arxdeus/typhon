# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# Typed Cython port of bench/codon/mandelbrot.py (no argv; N fixed).
# Entry point: module-level code runs at import (see run.py).
# Optional env MANDEL_N overrides N for quick correctness checks only.
import os
import time
from libc.stdlib cimport malloc, free
from libc.math cimport hypot, fma

cdef int MAX = 1000
cdef int N = int(os.environ.get("MANDEL_N", 4096))


cdef inline double scale(int x, double a, double b):
    return a + (<double>x / N) * (b - a)


cdef long long run():
    cdef int *pixels = <int *>malloc(<size_t>N * N * sizeof(int))
    cdef int i, j, iteration
    cdef double cr, ci, zr, zi, tr
    cdef long long total = 0
    cdef Py_ssize_t k
    for k in range(<Py_ssize_t>N * N):
        pixels[k] = 0
    for i in range(N):
        for j in range(N):
            cr = scale(j, -2.00, 0.47)
            ci = scale(i, -1.12, 1.12)
            zr = 0.0
            zi = 0.0
            iteration = 0
            # complex arithmetic spelled out on doubles exactly like CPython:
            # abs(z) == hypot(re, im); z**2 == c_prod(z, z), which the
            # reference CPython (Apple clang, arm64) compiles with FMA
            # contraction. fma() reproduces it bit-for-bit (build uses
            # -ffp-contract=off so nothing else gets fused).
            while hypot(zr, zi) <= 2 and iteration < MAX:
                tr = fma(zr, zr, -(zi * zi)) + cr
                zi = fma(zr, zi, zi * zr) + ci
                zr = tr
                iteration += 1
            pixels[i * N + j] = <int>(255 * iteration / <double>MAX)
    for k in range(<Py_ssize_t>N * N):
        total += pixels[k]
    free(pixels)
    return total


t0 = time.time()
print(run())
print(time.time() - t0)
