# Ported from scipy 1.18.1 (BSD-3-Clause): scipy/spatial/_spherical_voronoi.py
# and _voronoi.pyx (sort_vertices_of_regions).  Vertex numbering follows the
# ConvexHull facet order, which differs from Qhull (see _qhull.py).

import numpy as np
from math import atan2, acos, pi
from ._qhull import ConvexHull
from ._kdtree import KDTree

def _solid_angle(a, b, c) -> float:
    num = (a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
           + a[2] * (b[0] * c[1] - b[1] * c[0]))
    den = 1 + (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]) + (b[0] * c[0] + b[1] * c[1] + b[2] * c[2]) + (c[0] * a[0] + c[1] * a[1] + c[2] * a[2])
    return abs(2 * atan2(num, den))

class SphericalVoronoi:
    def __init__(self, points, radius: float = 1.0, center = None, threshold: float = 1e-06):
        P = np.array(points, dtype=float)
        self.radius = float(radius)
        self.points = P
        self._dim = P.shape[1]
        self.center = np.zeros(self._dim) if center is None else np.array(center, dtype=float)
        rank = int(np.linalg.matrix_rank(P - P[0], tol=threshold * self.radius))
        if rank < self._dim:
            raise ValueError(f"Rank of input points must be at least {self._dim}")
        if len(KDTree(P).query_pairs(threshold * self.radius)) > 0:
            raise ValueError("Duplicate generators present.")
        md = 0.0
        for i in range(P.shape[0]):
            md = max(md, abs(float(np.linalg.norm(P[i] - self.center)) - self.radius))
        if md >= threshold * self.radius:
            raise ValueError("Radius inconsistent with generators.")
        conv = ConvexHull(P)
        D = self._dim
        nf = conv.simplices.shape[0]
        self.vertices = self.radius * conv.equations[:, :D] + self.center
        self._simplices = conv.simplices
        regions = [List[int]() for _ in range(P.shape[0])]
        for p in range(P.shape[0]):
            for s in range(nf):
                for k in range(D):
                    if conv.simplices[s, k] == p:
                        regions[p].append(s)
        self.regions = regions

    def sort_vertices_of_regions(self):
        if self._dim != 3:
            raise TypeError("Only supported for three-dimensional point sets")
        S = self._simplices
        for n in range(len(self.regions)):
            rem = self.regions[n]
            m = len(rem)
            out = [0] * m
            cs = rem[0]
            cv = 0
            for i in range(3):
                k = S[cs, i]
                if k != n:
                    cv = k
                    break
            for k in range(m):
                out[k] = cs
                for i in range(m):
                    if rem[i] == out[k]:
                        continue
                    s = rem[i]
                    for j in range(3):
                        if cv == S[s, j]:
                            cs = s
                for i in range(3):
                    s = S[cs, i]
                    if s != n and s != cv:
                        cv = s
                        break
            self.regions[n] = out

    def calculate_areas(self):
        if self._dim == 2:
            ns = self._simplices.shape[0]
            areas = np.empty(ns)
            for s in range(ns):
                a0 = self.points[self._simplices[s, 0]] - self.center
                a1 = self.points[self._simplices[s, 1]] - self.center
                d = float(np.sum((a1 - a0) ** 2))
                th = acos(1 - d / (2 * self.radius ** 2))
                areas[s] = self.radius * th
                if float(np.dot(a0, self.vertices[s] - self.center)) < 0:
                    areas[s] = 2 * pi * self.radius - areas[s]
            return areas
        elif self._dim == 3:
            self.sort_vertices_of_regions()
            n = len(self.regions)
            out = np.empty(n)
            for p in range(n):
                pn = (self.points[p] - self.center) / self.radius
                reg = self.regions[p]
                m = len(reg)
                tot = 0.0
                for i in range(m):
                    v1 = (self.vertices[reg[i]] - self.center) / self.radius
                    v2 = (self.vertices[reg[(i - 1) % m]] - self.center) / self.radius
                    tot += _solid_angle(pn, v1, v2)
                out[p] = tot * self.radius ** 2
            return out
        else:
            raise TypeError("Only supported for 2D and 3D point sets")
