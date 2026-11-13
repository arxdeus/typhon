# Ported from scipy 1.18.1 (BSD-3-Clause); geometry after Qhull 2019.1
# (Barber, Dobkin, Huhdanpaa: "The Quickhull algorithm for convex hulls").
#
# Incremental beneath-beyond convex hull in dimension D >= 2 with outside
# (conflict) sets, as in Qhull.  Facets are simplicial (Qhull option Qt).
# Facet i: verts[i] (D point ids), normal[i] (unit outward), offset[i]
# (normal . x + offset <= 0 inside), neigh[i][j] = facet opposite verts[i][j].
#
# Deviation from Qhull: facet/vertex ordering differs, and for degenerate
# (coplanar / cospherical) input a different but valid triangulation can be
# produced.

import numpy as np
from math import sqrt, fabs

class QhullError(RuntimeError):
    def __init__(self, msg: str = ''):
        super().__init__(msg)

def _det(M: List[List[float]], n: int) -> float:
    # determinant by Gaussian elimination with partial pivoting
    A = [[M[i][j] for j in range(n)] for i in range(n)]
    det = 1.0
    for c in range(n):
        p = c
        best = fabs(A[c][c])
        for r in range(c + 1, n):
            if fabs(A[r][c]) > best:
                best = fabs(A[r][c])
                p = r
        if best == 0.0:
            return 0.0
        if p != c:
            A[c], A[p] = A[p], A[c]
            det = -det
        piv = A[c][c]
        det *= piv
        for r in range(c + 1, n):
            f = A[r][c] / piv
            if f != 0.0:
                for k in range(c + 1, n):
                    A[r][k] -= f * A[c][k]
    return det

class Hull:
    D: int
    pts: np.ndarray[float, 2]
    verts: List[List[int]]
    normal: List[List[float]]
    offset: List[float]
    neigh: List[List[int]]
    alive: List[bool]
    outside: List[List[int]]
    interior: List[float]
    eps: float
    is_vertex: List[bool]

    def __init__(self, pts):
        self.pts = pts
        self.D = pts.shape[1]
        self.verts = List[List[int]]()
        self.normal = List[List[float]]()
        self.offset = List[float]()
        self.neigh = List[List[int]]()
        self.alive = List[bool]()
        self.outside = List[List[int]]()
        self.interior = [0.0] * self.D
        n = pts.shape[0]
        mx = 0.0
        for i in range(n):
            for j in range(self.D):
                mx = max(mx, fabs(pts[i, j]))
        if mx == 0.0:
            mx = 1.0
        # Qhull DISTround-like tolerance
        self.eps = mx * float(self.D) * 2.220446049250313e-16 * 10.0
        self.is_vertex = [False] * n

    def dist(self, f: int, p: int) -> float:
        nr = self.normal[f]
        s = self.offset[f]
        for j in range(self.D):
            s += nr[j] * self.pts[p, j]
        return s

    def _plane(self, vs: List[int]):
        # unit normal through D points (generalised cross product), oriented
        # away from the interior point
        D = self.D
        rows = List[List[float]]()
        p0 = vs[0]
        for i in range(1, D):
            rows.append([self.pts[vs[i], j] - self.pts[p0, j] for j in range(D)])
        nr = [0.0] * D
        for c in range(D):
            M = List[List[float]]()
            for r in range(D - 1):
                M.append([rows[r][j] for j in range(D) if j != c])
            d = _det(M, D - 1)
            nr[c] = d if c % 2 == 0 else -d
        nn = 0.0
        for j in range(D):
            nn += nr[j] * nr[j]
        nn = sqrt(nn)
        if nn == 0.0:
            raise QhullError("QH6154 Qhull precision error: initial simplex is flat (facet is degenerate)")
        for j in range(D):
            nr[j] /= nn
        off = 0.0
        for j in range(D):
            off -= nr[j] * self.pts[p0, j]
        si = off
        for j in range(D):
            si += nr[j] * self.interior[j]
        if si > 0:
            for j in range(D):
                nr[j] = -nr[j]
            off = -off
        return nr, off

    def _new_facet(self, vs: List[int]) -> int:
        nr, off = self._plane(vs)
        self.verts.append(vs)
        self.normal.append(nr)
        self.offset.append(off)
        self.neigh.append([-1] * self.D)
        self.alive.append(True)
        self.outside.append(List[int]())
        return len(self.verts) - 1

    def _initial_simplex(self) -> List[int]:
        D = self.D
        n = self.pts.shape[0]
        if n < D + 1:
            raise QhullError(f"QH6214 qhull input error: not enough points({n}) to construct initial simplex (need {D + 1})")
        imin = 0
        imax = 0
        for i in range(n):
            if self.pts[i, 0] < self.pts[imin, 0]:
                imin = i
            if self.pts[i, 0] > self.pts[imax, 0]:
                imax = i
        chosen = [imin]
        if imax != imin:
            chosen.append(imax)
        # orthonormal basis of the affine hull of the chosen points
        basis = List[List[float]]()
        while len(chosen) < D + 1:
            basis = List[List[float]]()
            o = chosen[0]
            for c in chosen[1:]:
                v = [self.pts[c, j] - self.pts[o, j] for j in range(D)]
                for b in basis:
                    d = 0.0
                    for j in range(D):
                        d += v[j] * b[j]
                    for j in range(D):
                        v[j] -= d * b[j]
                nv = 0.0
                for j in range(D):
                    nv += v[j] * v[j]
                nv = sqrt(nv)
                if nv > 0:
                    basis.append([v[j] / nv for j in range(D)])
            best = -1
            bd = 0.0
            for i in range(n):
                v = [self.pts[i, j] - self.pts[o, j] for j in range(D)]
                for b in basis:
                    d = 0.0
                    for j in range(D):
                        d += v[j] * b[j]
                    for j in range(D):
                        v[j] -= d * b[j]
                nv = 0.0
                for j in range(D):
                    nv += v[j] * v[j]
                nv = sqrt(nv)
                if nv > bd:
                    bd = nv
                    best = i
            if best < 0 or bd <= self.eps * 10.0:
                raise QhullError("QH6154 Qhull precision error: initial simplex is flat")
            chosen.append(best)
        return chosen

    def build(self, order: Optional[List[int]] = None):
        D = self.D
        n = self.pts.shape[0]
        simp = self._initial_simplex()
        for j in range(D):
            s = 0.0
            for v in simp:
                s += self.pts[v, j]
            self.interior[j] = s / float(D + 1)
        for v in simp:
            self.is_vertex[v] = True
        for k in range(D + 1):
            vs = [simp[i] for i in range(D + 1) if i != k]
            self._new_facet(vs)
        # facet k is opposite simplex vertex k; neighbours
        for k in range(D + 1):
            f = k
            vs = self.verts[f]
            for j in range(D):
                # facet sharing vs minus vs[j] is the one opposite vs[j]
                self.neigh[f][j] = simp.index(vs[j])
        # partition
        pts_order = order if order is not None else [i for i in range(n)]
        for i in pts_order:
            if self.is_vertex[i]:
                continue
            for f in range(D + 1):
                if self.dist(f, i) > self.eps:
                    self.outside[f].append(i)
                    break
        queue = [f for f in range(D + 1)]
        qi = 0
        while qi < len(queue):
            f = queue[qi]
            qi += 1
            if not self.alive[f] or len(self.outside[f]) == 0:
                continue
            # furthest point
            best = -1
            bd = -1.0
            for p in self.outside[f]:
                d = self.dist(f, p)
                if d > bd:
                    bd = d
                    best = p
            newf = self._add_point(best, f)
            for g in newf:
                if len(self.outside[g]) > 0:
                    queue.append(g)

    def _add_point(self, p: int, f0: int) -> List[int]:
        D = self.D
        self.is_vertex[p] = True
        visible = [f0]
        mark = Dict[int, bool]()
        mark[f0] = True
        qi = 0
        while qi < len(visible):
            f = visible[qi]
            qi += 1
            for g in self.neigh[f]:
                if g in mark:
                    continue
                if self.dist(g, p) > self.eps:
                    mark[g] = True
                    visible.append(g)
                else:
                    mark[g] = False
        newfs = List[int]()
        ridge_map = Dict[str, Tuple[int, int]]()
        for f in visible:
            for j in range(D):
                g = self.neigh[f][j]
                if mark[g]:
                    continue
                ridge = [self.verts[f][i] for i in range(D) if i != j]
                vs = ridge + [p]
                nf = self._new_facet(vs)
                newfs.append(nf)
                self.neigh[nf][D - 1] = g
                gi = self.neigh[g].index(f)
                self.neigh[g][gi] = nf
                for r in range(D - 1):
                    sub = sorted([ridge[i] for i in range(D - 1) if i != r])
                    key = ','.join([str(v) for v in sub])
                    if key in ridge_map:
                        of, oi = ridge_map[key]
                        self.neigh[nf][r] = of
                        self.neigh[of][oi] = nf
                    else:
                        ridge_map[key] = (nf, r)
        for f in visible:
            self.alive[f] = False
            for q in self.outside[f]:
                if q == p:
                    continue
                for g in newfs:
                    if self.dist(g, q) > self.eps:
                        self.outside[g].append(q)
                        break
            self.outside[f] = List[int]()
        return newfs

    def compact(self):
        # returns (ids of alive facets in order, map old->new)
        idmap = [-1] * len(self.verts)
        ids = List[int]()
        for f in range(len(self.verts)):
            if self.alive[f]:
                idmap[f] = len(ids)
                ids.append(f)
        return ids, idmap

    def insert_coplanar(self, p: int, f0: int):
        # Delaunay mode: insert a point lying on the plane of facet f0
        # (cocircular input); facets coplanar with p are replaced as well.
        D = self.D
        self.is_vertex[p] = True
        visible = [f0]
        mark = Dict[int, bool]()
        mark[f0] = True
        qi = 0
        while qi < len(visible):
            f = visible[qi]
            qi += 1
            for g in self.neigh[f]:
                if g in mark:
                    continue
                if self.dist(g, p) > -self.eps:
                    mark[g] = True
                    visible.append(g)
                else:
                    mark[g] = False
        newfs = List[int]()
        ridge_map = Dict[str, Tuple[int, int]]()
        for f in visible:
            for j in range(D):
                g = self.neigh[f][j]
                if mark[g]:
                    continue
                ridge = [self.verts[f][i] for i in range(D) if i != j]
                nf = self._new_facet(ridge + [p])
                newfs.append(nf)
                self.neigh[nf][D - 1] = g
                gi = self.neigh[g].index(f)
                self.neigh[g][gi] = nf
                for r in range(D - 1):
                    sub = sorted([ridge[i] for i in range(D - 1) if i != r])
                    key = ','.join([str(v) for v in sub])
                    if key in ridge_map:
                        of, oi = ridge_map[key]
                        self.neigh[nf][r] = of
                        self.neigh[of][oi] = nf
                    else:
                        ridge_map[key] = (nf, r)
        for f in visible:
            self.alive[f] = False

    def finish_delaunay(self) -> List[Tuple[int, int, int]]:
        # make every remaining non-vertex point a vertex (lifted points are in
        # convex position); exact duplicates become `coplanar` records
        # (point, facet, vertex) like Qhull's Qc/Qz handling.
        n = self.pts.shape[0]
        D = self.D
        cop = List[Tuple[int, int, int]]()
        for p in range(n):
            if self.is_vertex[p]:
                continue
            best = -1
            bd = -1e300
            for f in range(len(self.verts)):
                if not self.alive[f]:
                    continue
                d = self.dist(f, p)
                if d > bd:
                    bd = d
                    best = f
            dup = -1
            for v in self.verts[best]:
                same = True
                for j in range(D - 1):
                    if self.pts[v, j] != self.pts[p, j]:
                        same = False
                        break
                if same:
                    dup = v
                    break
            if dup >= 0:
                cop.append((p, best, dup))
                continue
            if bd > self.eps:
                self._add_point(p, best)
            else:
                self.insert_coplanar(p, best)
        return cop
