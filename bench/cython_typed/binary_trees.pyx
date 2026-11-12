# cython: language_level=3, boundscheck=False, wraparound=False
# Typed Cython port of bench/codon/binary_trees.py.
# Entry point: module-level code runs main(int(sys.argv[1])) at import (see run.py).
import sys
import time
cimport cython


cdef class Node:
    cdef Node left
    cdef Node right

    def __init__(self, Node left=None, Node right=None):
        self.left = left
        self.right = right


cdef Node make_tree(int d):
    if d > 0:
        return Node(make_tree(d - 1), make_tree(d - 1))
    return Node()


cdef long long check_tree(Node node):
    cdef Node l = node.left
    cdef Node r = node.right
    if l is None:
        return 1
    return 1 + check_tree(l) + check_tree(r)


cdef long long make_check(tuple itde):
    cdef int d = itde[1]
    return check_tree(make_tree(d))


def get_argchunks(long long i, int d, int chunksize=5000):
    assert chunksize % 2 == 0
    cdef list chunk = []
    cdef long long k
    for k in range(1, i + 1):
        chunk.append((k, d))
        if len(chunk) == chunksize:
            yield chunk
            chunk = []
    if len(chunk) > 0:
        yield chunk


def main(int n, int min_depth=4):
    cdef int max_depth = max(min_depth + 2, n)
    cdef int stretch_depth = max_depth + 1
    cdef int mmd, d
    cdef long long i, cs
    cdef list argchunk
    cdef tuple it
    print(f'stretch tree of depth {stretch_depth}\t check: {make_check((0, stretch_depth))}')

    long_lived_tree = make_tree(max_depth)

    mmd = max_depth + min_depth
    for d in range(min_depth, stretch_depth, 2):
        i = (<long long>1) << (mmd - d)
        cs = 0
        for argchunk in get_argchunks(i, d):
            for it in argchunk:
                cs += make_check(it)
        print(f'{i}\t trees of depth {d}\t check: {cs}')

    print(f'long lived tree of depth {max_depth}\t check: {check_tree(long_lived_tree)}')


t0 = time.time()
main(int(sys.argv[1]))
t1 = time.time()
print(t1 - t0)
