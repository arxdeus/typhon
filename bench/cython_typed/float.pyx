# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# Typed Cython port of bench/codon/float.py (no argv; POINTS fixed).
# Entry point: module-level code runs at import (see run.py).
# Optional env FLOAT_POINTS overrides POINTS for quick correctness checks only.
import os
from time import time
from libc.math cimport sin, cos, sqrt

POINTS = int(os.environ.get("FLOAT_POINTS", 10000000))


cdef class Point:
    cdef public double x, y, z

    def __init__(self, double i):
        cdef double x
        self.x = x = sin(i)
        self.y = cos(i) * 3
        self.z = (x * x) / 2

    def __repr__(self):
        return f"<Point: x={self.x}, y={self.y}, z={self.z}>"

    cdef void normalize(self):
        cdef double x = self.x, y = self.y, z = self.z
        cdef double norm = sqrt(x * x + y * y + z * z)
        self.x /= norm
        self.y /= norm
        self.z /= norm

    cdef Point maximize(self, Point other):
        self.x = self.x if self.x > other.x else other.x
        self.y = self.y if self.y > other.y else other.y
        self.z = self.z if self.z > other.z else other.z
        return self


cdef Point maximize(list points):
    cdef Point nxt = points[0]
    cdef Py_ssize_t k
    for k in range(1, len(points)):
        nxt = nxt.maximize(<Point>points[k])
    return nxt


def benchmark(Py_ssize_t n):
    cdef list points = [None] * n
    cdef Py_ssize_t i
    cdef Point p
    for i in range(n):
        points[i] = Point(i)
    for p in points:
        p.normalize()
    return maximize(points)


t0 = time()
print(benchmark(POINTS))
t1 = time()
print(t1 - t0)
