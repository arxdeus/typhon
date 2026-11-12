# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# Typed Cython port of bench/codon/nbody.py.
# Entry point: module-level code runs main(int(sys.argv[1])) at import (see run.py).
# Bodies are a C struct array (pos, vel, mass) in the original dict order
# (sun, jupiter, saturn, uranus, neptune); pairs iterated in the same order
# as combinations(SYSTEM). mag uses pow(d2, -1.5) like `** (-1.5)`.
from time import time
import sys
from libc.math cimport pow, sqrt

DEF NB = 5

cdef double PI = 3.14159265358979323
cdef double SOLAR_MASS = 4 * PI * PI
cdef double DAYS_PER_YEAR = 365.24

cdef struct Body:
    double x[3]
    double v[3]
    double m

cdef Body bodies[NB]


cdef void set_body(int k, double x, double y, double z,
                   double vx, double vy, double vz, double m):
    bodies[k].x[0] = x
    bodies[k].x[1] = y
    bodies[k].x[2] = z
    bodies[k].v[0] = vx
    bodies[k].v[1] = vy
    bodies[k].v[2] = vz
    bodies[k].m = m


set_body(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, SOLAR_MASS)
set_body(1, 4.84143144246472090e+00, -1.16032004402742839e+00,
         -1.03622044471123109e-01,
         1.66007664274403694e-03 * DAYS_PER_YEAR,
         7.69901118419740425e-03 * DAYS_PER_YEAR,
         -6.90460016972063023e-05 * DAYS_PER_YEAR,
         9.54791938424326609e-04 * SOLAR_MASS)
set_body(2, 8.34336671824457987e+00, 4.12479856412430479e+00,
         -4.03523417114321381e-01,
         -2.76742510726862411e-03 * DAYS_PER_YEAR,
         4.99852801234917238e-03 * DAYS_PER_YEAR,
         2.30417297573763929e-05 * DAYS_PER_YEAR,
         2.85885980666130812e-04 * SOLAR_MASS)
set_body(3, 1.28943695621391310e+01, -1.51111514016986312e+01,
         -2.23307578892655734e-01,
         2.96460137564761618e-03 * DAYS_PER_YEAR,
         2.37847173959480950e-03 * DAYS_PER_YEAR,
         -2.96589568540237556e-05 * DAYS_PER_YEAR,
         4.36624404335156298e-05 * SOLAR_MASS)
set_body(4, 1.53796971148509165e+01, -2.59193146099879641e+01,
         1.79258772950371181e-01,
         2.68067772490389322e-03 * DAYS_PER_YEAR,
         1.62824170038242295e-03 * DAYS_PER_YEAR,
         -9.51592254519715870e-05 * DAYS_PER_YEAR,
         5.15138902046611451e-05 * SOLAR_MASS)


cdef void advance(double dt, long n) noexcept nogil:
    cdef long i
    cdef int a, b
    cdef double dx, dy, dz, mag, b1m, b2m
    for i in range(n):
        for a in range(NB - 1):
            for b in range(a + 1, NB):
                dx = bodies[a].x[0] - bodies[b].x[0]
                dy = bodies[a].x[1] - bodies[b].x[1]
                dz = bodies[a].x[2] - bodies[b].x[2]
                mag = dt * pow(dx * dx + dy * dy + dz * dz, -1.5)
                b1m = bodies[a].m * mag
                b2m = bodies[b].m * mag
                bodies[a].v[0] -= dx * b2m
                bodies[a].v[1] -= dy * b2m
                bodies[a].v[2] -= dz * b2m
                bodies[b].v[0] += dx * b1m
                bodies[b].v[1] += dy * b1m
                bodies[b].v[2] += dz * b1m
        for a in range(NB):
            bodies[a].x[0] += dt * bodies[a].v[0]
            bodies[a].x[1] += dt * bodies[a].v[1]
            bodies[a].x[2] += dt * bodies[a].v[2]


cdef void report_energy():
    cdef double e = 0.0, dx, dy, dz, vx, vy, vz
    cdef int a, b
    for a in range(NB - 1):
        for b in range(a + 1, NB):
            dx = bodies[a].x[0] - bodies[b].x[0]
            dy = bodies[a].x[1] - bodies[b].x[1]
            dz = bodies[a].x[2] - bodies[b].x[2]
            e -= (bodies[a].m * bodies[b].m) / pow(dx * dx + dy * dy + dz * dz, 0.5)
    for a in range(NB):
        vx = bodies[a].v[0]
        vy = bodies[a].v[1]
        vz = bodies[a].v[2]
        e += bodies[a].m * (vx * vx + vy * vy + vz * vz) / 2.
    print(e)


cdef void offset_momentum(int ref):
    cdef double px = 0.0, py = 0.0, pz = 0.0, m
    cdef int a
    for a in range(NB):
        m = bodies[a].m
        px -= bodies[a].v[0] * m
        py -= bodies[a].v[1] * m
        pz -= bodies[a].v[2] * m
    m = bodies[ref].m
    bodies[ref].v[0] = px / m
    bodies[ref].v[1] = py / m
    bodies[ref].v[2] = pz / m


def main(long n):
    offset_momentum(0)  # 'sun'
    report_energy()
    advance(0.01, n)
    report_energy()


t0 = time()
main(int(sys.argv[1]))
t1 = time()
print(t1 - t0)
