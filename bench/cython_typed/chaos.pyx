# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# Typed Cython port of bench/codon/chaos.py.
# Entry point: module-level code runs main() at import (see run.py);
# sys.argv[1] is the output PPM filename, exactly like the original.
# Optional env CHAOS_ITERS / CHAOS_SIZE override iterations / image size for
# quick correctness checks only. Python's `random` module is kept so the
# random sequence (and therefore the image) is identical to CPython.
import random
import sys
import os
import time
from libc.math cimport sqrt

DEFAULT_THICKNESS = 1.0
DEFAULT_WIDTH = int(os.environ.get("CHAOS_SIZE", 2048))
DEFAULT_HEIGHT = DEFAULT_WIDTH
DEFAULT_ITERATIONS = int(os.environ.get("CHAOS_ITERS", 1000000))
DEFAULT_RNG_SEED = 1234


cdef class GVector:
    cdef public double x, y, z

    def __init__(self, double x=0, double y=0, double z=0):
        self.x = x
        self.y = y
        self.z = z

    cdef double Mag(self):
        return sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

    cdef double dist(self, GVector other):
        return sqrt((self.x - other.x) ** 2
                    + (self.y - other.y) ** 2
                    + (self.z - other.z) ** 2)

    cdef GVector add(self, GVector other):
        return GVector(self.x + other.x, self.y + other.y, self.z + other.z)

    cdef GVector mul(self, double other):
        return GVector(self.x * other, self.y * other, self.z * other)

    cdef GVector sub(self, GVector other):
        # same as original: self + other * -1
        return self.add(other.mul(-1))

    cdef GVector linear_combination(self, GVector other, double l1, double l2):
        return GVector(self.x * l1 + other.x * l2,
                       self.y * l1 + other.y * l2,
                       self.z * l1 + other.z * l2)

    def __str__(self):
        return "<%f, %f, %f>" % (self.x, self.y, self.z)


cdef class Spline:
    cdef list points
    cdef int degree
    cdef double knots[16]
    cdef int nknots

    def __init__(self, list points, int degree, list knots):
        if len(points) > len(knots) - degree + 1:
            raise ValueError("too many control points")
        elif len(points) < len(knots) - degree + 1:
            raise ValueError("not enough control points")
        last = knots[0]
        for cur in knots[1:]:
            if cur < last:
                raise ValueError("knots not strictly increasing")
            last = cur
        assert len(knots) <= 16
        cdef int k
        self.nknots = len(knots)
        for k in range(self.nknots):
            self.knots[k] = knots[k]
        self.points = points
        self.degree = degree

    cdef inline double dom0(self):
        return self.knots[self.degree - 1]

    cdef inline double dom1(self):
        return self.knots[self.nknots - self.degree]

    cdef GVector call(self, double u):
        cdef double d0 = self.dom0(), d1 = self.dom1()
        if u < d0 or u > d1:
            raise ValueError("Function value not in domain")
        if u == d0:
            return self.points[0]
        if u == d1:
            return self.points[len(self.points) - 1]
        cdef int I = self.GetIndex(u)
        cdef int deg = self.degree, ii, ik, index
        cdef double ua, ub, co1, co2
        cdef list d = [self.points[I - deg + 1 + ii] for ii in range(deg + 1)]
        for ik in range(1, deg + 1):
            for ii in range(I - deg + ik + 1, I + 2):
                ua = self.knots[ii + deg - ik]
                ub = self.knots[ii - 1]
                co1 = (ua - u) / (ua - ub)
                co2 = (u - ub) / (ua - ub)
                index = ii - I + deg - ik - 1
                d[index] = (<GVector>d[index]).linear_combination(
                    <GVector>d[index + 1], co1, co2)
        return d[0]

    cdef int GetIndex(self, double u):
        cdef int ii
        for ii in range(self.degree - 1, self.nknots - self.degree):
            if u >= self.knots[ii] and u < self.knots[ii + 1]:
                return ii
        return <int>self.dom1() - 1


def write_ppm(im, filename):
    magic = 'P6\n'
    maxval = 255
    w = len(im)
    h = len(im[0])

    with open(filename, "w", encoding="latin1", newline='') as fp:
        fp.write(magic)
        fp.write('%i %i\n%i\n' % (w, h, maxval))
        for j in range(h):
            for i in range(w):
                val = im[i][j]
                c = val * 255
                fp.write('%c%c%c' % (c, c, c))


cdef class Chaosgame:
    cdef list splines
    cdef double thickness, minx, miny, maxx, maxy, height, width
    cdef list num_trafos
    cdef long num_total

    def __init__(self, list splines, double thickness=0.1):
        self.splines = splines
        self.thickness = thickness
        self.minx = min([p.x for spl in splines for p in (<Spline>spl).points])
        self.miny = min([p.y for spl in splines for p in (<Spline>spl).points])
        self.maxx = max([p.x for spl in splines for p in (<Spline>spl).points])
        self.maxy = max([p.y for spl in splines for p in (<Spline>spl).points])
        self.height = self.maxy - self.miny
        self.width = self.maxx - self.minx
        self.num_trafos = []
        cdef double maxlength = thickness * self.width / self.height
        cdef double length, t
        cdef GVector curr, last
        cdef Spline spl
        cdef int i
        for spl in splines:
            length = 0
            curr = spl.call(0)
            for i in range(1, 1000):
                last = curr
                t = 1.0 / 999 * i
                curr = spl.call(t)
                length += curr.dist(last)
            self.num_trafos.append(max(1, int(length / maxlength * 1.5)))
        self.num_total = sum(self.num_trafos)

    cdef tuple get_random_trafo(self):
        cdef long r = random.randrange(int(self.num_total) + 1)
        cdef long l = 0, nt
        cdef Py_ssize_t i
        for i in range(len(self.num_trafos)):
            nt = self.num_trafos[i]
            if r >= l and r < l + nt:
                return i, random.randrange(nt)
            l += nt
        return len(self.num_trafos) - 1, random.randrange(self.num_trafos[len(self.num_trafos) - 1])

    cdef GVector transform_point(self, GVector point):
        cdef double x = (point.x - self.minx) / self.width
        cdef double y = (point.y - self.miny) / self.height
        cdef tuple trafo = self.get_random_trafo()
        cdef int t0 = trafo[0]
        cdef long t1 = trafo[1]
        cdef Spline spl = self.splines[t0]
        cdef double start = spl.dom0(), end = spl.dom1()
        cdef double length = end - start
        cdef double seg_length = length / <long>self.num_trafos[t0]
        cdef double t = start + seg_length * t1 + seg_length * x
        cdef GVector basepoint = spl.call(t)
        cdef GVector neighbour, derivative
        if t + 1.0 / 50000 > end:
            neighbour = spl.call(t - 1.0 / 50000)
            derivative = neighbour.sub(basepoint)
        else:
            neighbour = spl.call(t + 1.0 / 50000)
            derivative = basepoint.sub(neighbour)
        if derivative.Mag() != 0:
            basepoint.x += derivative.y / derivative.Mag() * (y - 0.5) * \
                self.thickness
            basepoint.y += -derivative.x / derivative.Mag() * (y - 0.5) * \
                self.thickness
        else:
            print("r", end='')
        self.truncate(basepoint)
        return basepoint

    cdef void truncate(self, GVector point):
        if point.x >= self.maxx:
            point.x = self.maxx
        if point.y >= self.maxy:
            point.y = self.maxy
        if point.x < self.minx:
            point.x = self.minx
        if point.y < self.miny:
            point.y = self.miny

    def create_image_chaos(self, int w, int h, long iterations, filename, rng_seed):
        random.seed(rng_seed)
        cdef list im = [bytearray(b'\x01') * h for i in range(w)]
        cdef GVector point = GVector((self.maxx + self.minx) / 2,
                                     (self.maxy + self.miny) / 2, 0)
        cdef long _
        cdef int x, y
        cdef bytearray col
        for _ in range(iterations):
            point = self.transform_point(point)
            x = <int>((point.x - self.minx) / self.width * w)
            y = <int>((point.y - self.miny) / self.height * h)
            if x == w:
                x -= 1
            if y == h:
                y -= 1
            col = im[x]
            col[h - y - 1] = 0

        if filename:
            write_ppm(im, filename)


def main():
    splines = [
        Spline([
            GVector(1.597350, 3.304460, 0.000000),
            GVector(1.575810, 4.123260, 0.000000),
            GVector(1.313210, 5.288350, 0.000000),
            GVector(1.618900, 5.329910, 0.000000),
            GVector(2.889940, 5.502700, 0.000000),
            GVector(2.373060, 4.381830, 0.000000),
            GVector(1.662000, 4.360280, 0.000000)],
            3, [0, 0, 0, 1, 1, 1, 2, 2, 2]),
        Spline([
            GVector(2.804500, 4.017350, 0.000000),
            GVector(2.550500, 3.525230, 0.000000),
            GVector(1.979010, 2.620360, 0.000000),
            GVector(1.979010, 2.620360, 0.000000)],
            3, [0, 0, 0, 1, 1, 1]),
        Spline([
            GVector(2.001670, 4.011320, 0.000000),
            GVector(2.335040, 3.312830, 0.000000),
            GVector(2.366800, 3.233460, 0.000000),
            GVector(2.366800, 3.233460, 0.000000)],
            3, [0, 0, 0, 1, 1, 1])
    ]

    chaos = Chaosgame(splines, DEFAULT_THICKNESS)
    chaos.create_image_chaos(DEFAULT_WIDTH, DEFAULT_HEIGHT, DEFAULT_ITERATIONS, sys.argv[1], DEFAULT_RNG_SEED)


t0 = time.time()
main()
t1 = time.time()
print(t1 - t0)
