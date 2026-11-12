# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# Typed Cython port of bench/codon/primes.py.
# Entry point: module-level code runs at import (see run.py); reads sys.argv[1].
from sys import argv
from time import time


cdef bint is_prime(long long n):
    cdef long long factors = 0, i
    for i in range(2, n):
        if n % i == 0:
            factors += 1
    return factors == 0


cdef long long count(long long limit):
    cdef long long total = 0, i
    for i in range(2, limit):
        if is_prime(i):
            total += 1
    return total


limit = int(argv[1])

t0 = time()
total = count(limit)
t1 = time()

print(total)
print(t1 - t0)
