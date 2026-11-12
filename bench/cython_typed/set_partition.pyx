# cython: language_level=3, boundscheck=False, wraparound=False
# Typed Cython port of bench/codon/set_partition.py.
# Entry point: module-level code runs at import (see run.py); reads sys.argv[1].
# The recursive generator over nested Python lists is inherently dynamic;
# typing is limited to cdef helpers, typed ints and typed list locals.
import sys
import time


cdef bint conforms(list candidate, Py_ssize_t minsize, Py_ssize_t forgive):
    cdef Py_ssize_t deficit = 0, need
    cdef list p
    for p in candidate:
        need = minsize - len(p)
        if need > 0:
            deficit += need
    return deficit <= forgive


def partition_filtered(list collection, Py_ssize_t minsize=1, Py_ssize_t forgive=0):
    if len(collection) == 1:
        yield [collection]
        return

    first = collection[0]
    cdef list smaller, subset, candidate
    cdef Py_ssize_t n
    for smaller in partition_filtered(collection[1:], minsize, forgive + 1):
        for n in range(len(smaller)):
            subset = smaller[n]
            candidate = smaller[:n] + [[first] + subset] + smaller[n + 1:]
            if conforms(candidate, minsize, forgive):
                yield candidate

        candidate = [[first]] + smaller
        if conforms(candidate, minsize, forgive):
            yield candidate


def main():
    cdef list something, p
    cdef long long x = 0
    t = time.time()
    something = list(range(1, int(sys.argv[1])))
    v = partition_filtered(something, minsize=2)
    for p in v:
        p.sort()
        x += <long long>p[len(p) // 3][0]
    print(x)
    print(time.time() - t)


main()
