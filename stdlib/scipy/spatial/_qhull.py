# Ported from scipy 1.18.1 (BSD-3-Clause) scipy/spatial/_qhull.pyx
#
# ConvexHull, Delaunay, Voronoi, HalfspaceIntersection and tsearch on top of
# the beneath-beyond engine in _hull.py.  Delaunay uses the Qhull default
# "Qbb Qc Qz Q12" paraboloid lifting (scale = max|x| / (1.1 max z - min z)).
# Deviations from Qhull: the order of simplices/facets/Voronoi vertices and
# regions, the starting vertex of 2-D hull `vertices` (always CCW like
# scipy), and N-D (N >= 3) simplex orientation may differ; the sets of
# simplices, equations (per facet), volumes and areas are the same for
# input in general position.  No incremental mode.

import numpy as np
from math import sqrt, fabs
from _hull import Hull, QhullError, _det

def _fact(n: int) -> float:
    r = 1.0
    for i in range(2, n + 1):
        r *= float(i)
    return r

def _simplex_volume(P, vs: List[int], D: int) -> float:
    # volume of D-simplex (D+1 points in D dims)
    M = List[List[float]]()
    for i in range(1, D + 1):
        M.append([P[vs[i], j] - P[vs[0], j] for j in range(D)])
    return fabs(_det(M, D)) / _fact(D)

def _facet_area(P, vs: List[int], D: int) -> float:
    # (D-1)-volume of a facet with D vertices in D dims (Gram determinant)
    k = D - 1
    vecs = List[List[float]]()
    for i in range(1, D):
        vecs.append([P[vs[i], j] - P[vs[0], j] for j in range(D)])
    G = List[List[float]]()
    for a in range(k):
        row = List[float]()
        for b in range(k):
            s = 0.0
            for j in range(D):
                s += vecs[a][j] * vecs[b][j]
            row.append(s)
        G.append(row)
    g = _det(G, k)
    return sqrt(max(g, 0.0)) / _fact(k)

def _barycentric_transforms(points, simplices):
    ndim = points.shape[1]
    ns = simplices.shape[0]
    Tinvs = np.zeros((ns, ndim + 1, ndim))
    eps = 2.220446049250313e-16
    for s in range(ns):
        T = np.empty((ndim, ndim))
        for i in range(ndim):
            Tinvs[s, ndim, i] = points[simplices[s, ndim], i]
            for j in range(ndim):
                T[i, j] = points[simplices[s, j], i] - Tinvs[s, ndim, i]
        anorm = 0.0
        for j in range(ndim):
            cs = 0.0
            for i in range(ndim):
                cs += fabs(T[i, j])
            anorm = max(anorm, cs)
        ok = True
        Ti = np.zeros((ndim, ndim))
        try:
            Ti = np.linalg.inv(T)
        except:
            ok = False
        if ok:
            inorm = 0.0
            for j in range(ndim):
                cs = 0.0
                for i in range(ndim):
                    cs += fabs(Ti[i, j])
                inorm = max(inorm, cs)
            rcond = 1.0 / (anorm * inorm) if anorm * inorm > 0 else 0.0
            if not (rcond >= 1000 * eps):
                ok = False
        for i in range(ndim):
            for j in range(ndim):
                Tinvs[s, i, j] = Ti[i, j] if ok else np.nan
        if not ok:
            for j in range(ndim):
                Tinvs[s, ndim, j] = np.nan
    return Tinvs

def _check_points(P, mind: int):
    if P.ndim != 2:
        raise ValueError("Input points array must have 2 dimensions.")
    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            if not np.isfinite(P[i, j]):
                raise ValueError("Points cannot contain NaN")
    if P.shape[1] < 2:
        raise ValueError("Need at least 2-D data")

def _lift(P):
    n = P.shape[0]
    D = P.shape[1]
    z = np.empty(n)
    zmin = 1e300
    zmax = -1e300
    mx = 0.0
    for i in range(n):
        s = 0.0
        for j in range(D):
            s += P[i, j] * P[i, j]
            mx = max(mx, fabs(P[i, j]))
        z[i] = s
        zmin = min(zmin, s)
        zmax = max(zmax, s)
    den = 1.1 * zmax - zmin
    scale = mx / den if den > 0 else 1.0
    shift = -zmin * scale
    L = np.empty((n, D + 1))
    for i in range(n):
        for j in range(D):
            L[i, j] = P[i, j]
        L[i, D] = z[i] * scale + shift
    return L, scale, shift

class _Tri:
    # result of a Delaunay triangulation (shared by Delaunay and Voronoi)
    simplices: np.ndarray[int, 2]
    neighbors: np.ndarray[int, 2]
    equations: np.ndarray[float, 2]
    coplanar: np.ndarray[int, 2]
    scale: float
    shift: float
    lifted: np.ndarray[float, 2]

    def __init__(self, P):
        n = P.shape[0]
        D = P.shape[1]
        if n < D + 2:
            raise QhullError(f"QH6214 qhull input error: not enough points({n}) to construct initial simplex (need {D + 2})")
        L, sc, sh = _lift(P)
        self.scale = sc
        self.shift = sh
        self.lifted = L
        H = Hull(L)
        H.build()
        cop = H.finish_delaunay()
        ids = List[int]()
        idmap = [-1] * len(H.verts)
        for f in range(len(H.verts)):
            if H.alive[f] and H.normal[f][D] < -1e-12:
                idmap[f] = len(ids)
                ids.append(f)
        ns = len(ids)
        S = np.empty((ns, D + 1), dtype=int)
        N = np.empty((ns, D + 1), dtype=int)
        E = np.empty((ns, D + 2))
        for a in range(ns):
            f = ids[a]
            for j in range(D + 1):
                S[a, j] = H.verts[f][j]
                N[a, j] = idmap[H.neigh[f][j]]
            for j in range(D + 1):
                E[a, j] = H.normal[f][j]
            E[a, D + 1] = H.offset[f]
            if D == 2:
                p0 = S[a, 0]
                p1 = S[a, 1]
                p2 = S[a, 2]
                cr = (P[p1, 0] - P[p0, 0]) * (P[p2, 1] - P[p0, 1]) - (P[p1, 1] - P[p0, 1]) * (P[p2, 0] - P[p0, 0])
                if cr < 0:
                    S[a, 0] = p1
                    S[a, 1] = p0
                    t = N[a, 0]
                    N[a, 0] = N[a, 1]
                    N[a, 1] = t
        C = np.empty((len(cop), 3), dtype=int)
        for i in range(len(cop)):
            p, f, v = cop[i]
            C[i, 0] = p
            C[i, 1] = idmap[f]
            C[i, 2] = v
        self.simplices = S
        self.neighbors = N
        self.equations = E
        self.coplanar = C

class Delaunay:
    def __init__(self, points, furthest_site: bool = False, incremental: bool = False,
                 qhull_options: Optional[str] = None):
        P = np.ascontiguousarray(np.asarray(points, dtype=float))
        _check_points(P, 2)
        if furthest_site:
            raise NotImplementedError("Delaunay: furthest_site is not supported")
        tri = _Tri(P)
        self._points = P
        self.ndim = P.shape[1]
        self.npoints = P.shape[0]
        self.min_bound = np.min(P, axis=0)
        self.max_bound = np.max(P, axis=0)
        self.simplices = tri.simplices
        self.neighbors = tri.neighbors
        self.equations = tri.equations
        self.coplanar = tri.coplanar
        self.good = np.zeros(0, dtype=int)
        self.nsimplex = tri.simplices.shape[0]
        self.paraboloid_scale = tri.scale
        self.paraboloid_shift = tri.shift
        self.furthest_site = furthest_site
        self._transform = np.zeros((0, 0, 0))
        self._have_transform = False

    @property
    def points(self):
        return self._points

    @property
    def vertices(self):
        return self.simplices

    @property
    def transform(self):
        if not self._have_transform:
            self._transform = _barycentric_transforms(self._points, self.simplices)
            self._have_transform = True
        return self._transform

    @property
    def vertex_to_simplex(self):
        arr = np.full(self.npoints, -1, dtype=int)
        for i in range(self.coplanar.shape[0]):
            arr[self.coplanar[i, 0]] = self.coplanar[i, 2]
        for s in range(self.nsimplex):
            for k in range(self.ndim + 1):
                v = self.simplices[s, k]
                if arr[v] == -1:
                    arr[v] = s
        return arr

    @property
    def vertex_neighbor_vertices(self):
        sets = [List[int]() for _ in range(self.npoints)]
        for i in range(self.nsimplex):
            for j in range(self.ndim + 1):
                for k in range(self.ndim + 1):
                    a = self.simplices[i, j]
                    b = self.simplices[i, k]
                    if a != b and b not in sets[a]:
                        sets[a].append(b)
        indptr = np.zeros(self.npoints + 1, dtype=int)
        tot = 0
        for i in range(self.npoints):
            sets[i].sort()
            tot += len(sets[i])
            indptr[i + 1] = tot
        indices = np.empty(tot, dtype=int)
        q = 0
        for i in range(self.npoints):
            for v in sets[i]:
                indices[q] = v
                q += 1
        return indptr, indices

    @property
    def convex_hull(self):
        D = self.ndim
        rows = List[List[int]]()
        for s in range(self.nsimplex):
            for k in range(D + 1):
                if self.neighbors[s, k] == -1:
                    rows.append([int(self.simplices[s, j]) for j in range(D + 1) if j != k])
        out = np.empty((len(rows), D), dtype=int)
        for i in range(len(rows)):
            for j in range(D):
                out[i, j] = rows[i][j]
        return out

    def lift_points(self, x):
        xa = np.asarray(x, dtype=float)
        D = self.ndim
        X = np.ascontiguousarray(xa).reshape((-1, D))
        z = np.empty((X.shape[0], D + 1))
        for i in range(X.shape[0]):
            s = 0.0
            for j in range(D):
                z[i, j] = X[i, j]
                s += X[i, j] * X[i, j]
            z[i, D] = s * self.paraboloid_scale + self.paraboloid_shift
        return z.reshape(xa.shape[:-1] + (D + 1,))

    def _distplane(self, s: int, z) -> float:
        D = self.ndim
        d = self.equations[s, D + 1]
        for k in range(D + 1):
            d += self.equations[s, k] * z[k]
        return d

    def _bary_single(self, T, s: int, x, c, i: int):
        D = self.ndim
        if i == D:
            c[D] = 1.0
            for j in range(D):
                c[D] -= c[j]
        else:
            c[i] = 0.0
            for j in range(D):
                c[i] += T[s, i, j] * (x[j] - T[s, D, j])

    def _bary_all(self, T, s: int, x, c):
        D = self.ndim
        c[D] = 1.0
        for i in range(D):
            c[i] = 0.0
            for j in range(D):
                c[i] += T[s, i, j] * (x[j] - T[s, D, j])
            c[D] -= c[i]

    def _outside(self, x, eps: float) -> bool:
        for i in range(self.ndim):
            if x[i] < self.min_bound[i] - eps or x[i] > self.max_bound[i] + eps:
                return True
        return False

    def _find_bruteforce(self, T, c, x, eps: float, eps_broad: float) -> int:
        D = self.ndim
        if self._outside(x, eps):
            return -1
        for s in range(self.nsimplex):
            if T[s, 0, 0] == T[s, 0, 0]:
                self._bary_all(T, s, x, c)
                inside = True
                for i in range(D + 1):
                    if not (-eps <= c[i] and c[i] <= 1 + eps):
                        inside = False
                        break
                if inside:
                    return s
            else:
                for k in range(D + 1):
                    nb = self.neighbors[s, k]
                    if nb == -1:
                        continue
                    if T[nb, 0, 0] != T[nb, 0, 0]:
                        continue
                    self._bary_all(T, nb, x, c)
                    inside = True
                    for m in range(D + 1):
                        if self.neighbors[nb, m] == s:
                            if not (-eps_broad <= c[m] and c[m] <= 1 + eps):
                                inside = False
                                break
                        else:
                            if not (-eps <= c[m] and c[m] <= 1 + eps):
                                inside = False
                                break
                    if inside:
                        return nb
        return -1

    def _find_directed(self, T, c, x, start: int, eps: float, eps_broad: float):
        D = self.ndim
        s = start
        if s < 0 or s >= self.nsimplex:
            s = 0
        done = False
        for cyc in range(1 + self.nsimplex // 4):
            if s == -1:
                done = True
                break
            inside = 1
            for k in range(D + 1):
                self._bary_single(T, s, x, c, k)
                if c[k] < -eps:
                    m = self.neighbors[s, k]
                    if m == -1:
                        return -1, s
                    s = m
                    inside = -1
                    break
                elif c[k] <= 1 + eps:
                    pass
                else:
                    inside = 0
            if inside == -1:
                continue
            elif inside == 1:
                done = True
                break
            else:
                s = self._find_bruteforce(T, c, x, eps, eps_broad)
                done = True
                break
        if not done:
            s = self._find_bruteforce(T, c, x, eps, eps_broad)
        return s, s

    def _find(self, T, c, x, start: int, eps: float, eps_broad: float):
        D = self.ndim
        if self._outside(x, eps):
            return -1, start
        if self.nsimplex <= 0:
            return -1, start
        s = start
        if s < 0 or s >= self.nsimplex:
            s = 0
        z = [0.0] * (D + 1)
        zz = 0.0
        for i in range(D):
            z[i] = x[i]
            zz += x[i] * x[i]
        z[D] = zz * self.paraboloid_scale + self.paraboloid_shift
        best = self._distplane(s, z)
        changed = True
        while changed:
            if best > 0:
                break
            changed = False
            for k in range(D + 1):
                nb = self.neighbors[s, k]
                if nb == -1:
                    continue
                d = self._distplane(nb, z)
                if d > best + eps * (1 + fabs(best)):
                    s = nb
                    best = d
                    changed = True
        return self._find_directed(T, c, x, s, eps, eps_broad)

    def find_simplex(self, xi, bruteforce: bool = False, tol: Optional[float] = None):
        xa = np.asarray(xi, dtype=float)
        if xa.shape[xa.ndim - 1] != self.ndim:
            raise ValueError("wrong dimensionality in xi")
        X = np.ascontiguousarray(xa).reshape((-1, self.ndim))
        eps = 100 * 2.220446049250313e-16 if tol is None else tol
        eps_broad = sqrt(eps)
        T = self.transform
        c = np.zeros(self.ndim + 1)
        out = np.zeros(X.shape[0], dtype=int)
        start = 0
        for k in range(X.shape[0]):
            if bruteforce:
                out[k] = self._find_bruteforce(T, c, X[k], eps, eps_broad)
            else:
                r, start = self._find(T, c, X[k], start, eps, eps_broad)
                out[k] = r
        return out.reshape(xa.shape[:-1])

    def plane_distance(self, xi):
        xa = np.asarray(xi, dtype=float)
        X = np.ascontiguousarray(xa).reshape((-1, self.ndim))
        Z = self.lift_points(X)
        out = np.zeros((X.shape[0], self.nsimplex))
        for i in range(X.shape[0]):
            for j in range(self.nsimplex):
                out[i, j] = self._distplane(j, Z[i])
        return out.reshape(xa.shape[:-1] + (self.nsimplex,))

def tsearch(tri: Delaunay, xi):
    return tri.find_simplex(xi)

def _hull_of(P, interior = None):
    H = Hull(P)
    H.build()
    return H

class ConvexHull:
    def __init__(self, points, incremental: bool = False, qhull_options: Optional[str] = None):
        P = np.ascontiguousarray(np.asarray(points, dtype=float))
        _check_points(P, 2)
        D = P.shape[1]
        if P.shape[0] < D + 1:
            raise QhullError(f"QH6214 qhull input error: not enough points({P.shape[0]}) to construct initial simplex (need {D + 1})")
        H = _hull_of(P)
        ids, idmap = H.compact()
        nf = len(ids)
        S = np.empty((nf, D), dtype=int)
        N = np.empty((nf, D), dtype=int)
        E = np.empty((nf, D + 1))
        vol = 0.0
        area = 0.0
        for a in range(nf):
            f = ids[a]
            vs = H.verts[f]
            for j in range(D):
                S[a, j] = vs[j]
                N[a, j] = idmap[H.neigh[f][j]]
                E[a, j] = H.normal[f][j]
            E[a, D] = H.offset[f]
            area += _facet_area(P, vs, D)
            # pyramid from the interior point: |offset_c| * facetarea / D
            dc = H.offset[f]
            for j in range(D):
                dc += H.normal[f][j] * H.interior[j]
            vol += -dc * _facet_area(P, vs, D) / float(D)
        self._points = P
        self.ndim = D
        self.npoints = P.shape[0]
        self.min_bound = np.min(P, axis=0)
        self.max_bound = np.max(P, axis=0)
        self.simplices = S
        self.neighbors = N
        self.equations = E
        self.coplanar = np.zeros((0, 3), dtype=int)
        self.good = None
        self.nsimplex = nf
        self.volume = vol
        self.area = area
        if D == 2:
            self._vertices = self._extremes_2d(H, ids, idmap)
        else:
            vs = List[int]()
            for a in range(nf):
                for j in range(D):
                    vs.append(int(S[a, j]))
            self._vertices = np.unique(np.array(vs, dtype=int))

    def _extremes_2d(self, H: Hull, ids: List[int], idmap: List[int]):
        # counter-clockwise walk around the hull (scipy order), starting at
        # the first facet's first vertex
        P = self._points
        nf = len(ids)
        nxt = Dict[int, int]()
        for a in range(nf):
            f = ids[a]
            v0 = H.verts[f][0]
            v1 = H.verts[f][1]
            cr = (P[v0, 0] - H.interior[0]) * (P[v1, 1] - H.interior[1]) - (P[v0, 1] - H.interior[1]) * (P[v1, 0] - H.interior[0])
            if cr > 0:
                nxt[v0] = v1
            else:
                nxt[v1] = v0
        start = H.verts[ids[0]][0]
        out = [start]
        cur = nxt[start]
        while cur != start and len(out) <= nf:
            out.append(cur)
            cur = nxt[cur]
        return np.array(out, dtype=int)

    @property
    def points(self):
        return self._points

    @property
    def vertices(self):
        return self._vertices

class Voronoi:
    def __init__(self, points, furthest_site: bool = False, incremental: bool = False,
                 qhull_options: Optional[str] = None):
        P = np.ascontiguousarray(np.asarray(points, dtype=float))
        _check_points(P, 2)
        if furthest_site:
            raise NotImplementedError("Voronoi: furthest_site is not supported")
        D = P.shape[1]
        n = P.shape[0]
        tri = _Tri(P)
        S = tri.simplices
        ns = S.shape[0]
        # Voronoi vertices = circumcentres of the Delaunay simplices
        V = np.empty((ns, D))
        for s in range(ns):
            A = np.empty((D, D))
            b = np.empty(D)
            for i in range(D):
                bi = 0.0
                for j in range(D):
                    A[i, j] = 2.0 * (P[S[s, i + 1], j] - P[S[s, 0], j])
                    bi += P[S[s, i + 1], j] ** 2 - P[S[s, 0], j] ** 2
                b[i] = bi
            c = np.linalg.solve(A, b)
            for j in range(D):
                V[s, j] = c[j]
        # ridges: Delaunay edges (D=2) / faces between adjacent points
        ridge_pts = List[Tuple[int, int]]()
        ridge_vs = List[List[int]]()
        seen = Dict[str, int]()
        for s in range(ns):
            for a in range(D + 1):
                for b2 in range(a + 1, D + 1):
                    p = int(S[s, a])
                    q = int(S[s, b2])
                    lo = min(p, q)
                    hi = max(p, q)
                    key = f"{lo},{hi}"
                    if key in seen:
                        continue
                    seen[key] = 1
                    # simplices containing both p and q, ordered around the edge
                    ring = self._ring(S, tri.neighbors, s, p, q, D)
                    ridge_pts.append((p, q))
                    ridge_vs.append(ring)
        self.vertices = V
        RP = np.empty((len(ridge_pts), 2), dtype=int)
        for i in range(len(ridge_pts)):
            RP[i, 0] = ridge_pts[i][0]
            RP[i, 1] = ridge_pts[i][1]
        self.ridge_points = RP
        self.ridge_vertices = ridge_vs
        # regions: for each input point, simplices around it in cyclic order
        regions = List[List[int]]()
        point_region = np.full(n, -1, dtype=int)
        regions.append(List[int]())
        vts = List[List[int]]()
        for p in range(n):
            vts.append(List[int]())
        for s in range(ns):
            for a in range(D + 1):
                vts[int(S[s, a])].append(s)
        for p in range(n):
            if len(vts[p]) == 0:
                continue
            reg = self._region(S, tri.neighbors, vts[p], p, D)
            point_region[p] = len(regions)
            regions.append(reg)
        for i in range(tri.coplanar.shape[0]):
            point_region[tri.coplanar[i, 0]] = point_region[tri.coplanar[i, 2]]
        self.regions = regions
        self.point_region = point_region
        self._points = P
        self.ndim = D
        self.npoints = n
        self.min_bound = np.min(P, axis=0)
        self.max_bound = np.max(P, axis=0)
        self.furthest_site = furthest_site

    def _ring(self, S, N, s0: int, p: int, q: int, D: int) -> List[int]:
        # simplices sharing edge (p, q); -1 marks the infinite vertex
        out = List[int]()
        if D == 2:
            out.append(s0)
            # the neighbour opposite the third vertex shares edge p-q
            third = 0
            for a in range(3):
                if S[s0, a] != p and S[s0, a] != q:
                    third = a
            nb = int(N[s0, third])
            if nb == -1:
                return [-1, s0]
            out.append(nb)
            return out
        else:
            for s in range(S.shape[0]):
                hp = False
                hq = False
                for a in range(D + 1):
                    if S[s, a] == p:
                        hp = True
                    if S[s, a] == q:
                        hq = True
                if hp and hq:
                    out.append(s)
            bnd = False
            for s in out:
                cnt = 0
                for a in range(D + 1):
                    v = S[s, a]
                    if v != p and v != q and N[s, a] == -1:
                        bnd = True
            if bnd:
                out.insert(0, -1)
            return out

    def _region(self, S, N, simps: List[int], p: int, D: int) -> List[int]:
        if D != 2:
            res = List[int]()
            unbounded = False
            for s in simps:
                for a in range(D + 1):
                    if S[s, a] != p and N[s, a] == -1:
                        unbounded = True
                res.append(s)
            if unbounded:
                res.insert(0, -1)
            return res
        # walk around p through neighbouring triangles
        start = simps[0]
        # if p is on the hull, start from a boundary triangle
        for s in simps:
            ip = 0
            for a in range(3):
                if S[s, a] == p:
                    ip = a
            nb_next = int(N[s, (ip + 1) % 3])
            if nb_next == -1:
                start = s
        res = List[int]()
        cur = start
        unb = False
        for it in range(len(simps) + 1):
            res.append(cur)
            ip = 0
            for a in range(3):
                if S[cur, a] == p:
                    ip = a
            nb = int(N[cur, (ip + 2) % 3])
            if nb == -1:
                unb = True
                break
            if nb == start:
                break
            cur = nb
        if unb:
            res.append(-1)
        return res

    @property
    def points(self):
        return self._points

    @property
    def ridge_dict(self):
        d = Dict[Tuple[int, int], List[int]]()
        for i in range(self.ridge_points.shape[0]):
            d[(int(self.ridge_points[i, 0]), int(self.ridge_points[i, 1]))] = self.ridge_vertices[i]
        return d

class HalfspaceIntersection:
    def __init__(self, halfspaces, interior_point, incremental: bool = False,
                 qhull_options: Optional[str] = None):
        Hs = np.ascontiguousarray(np.asarray(halfspaces, dtype=float))
        ip = np.ascontiguousarray(np.asarray(interior_point, dtype=float))
        D = Hs.shape[1] - 1
        if ip.shape[0] != D:
            raise ValueError("Feasible point must be a (ndim-1,) array")
        m = Hs.shape[0]
        dual = np.empty((m, D))
        for i in range(m):
            off = Hs[i, D]
            for j in range(D):
                off += Hs[i, j] * ip[j]
            if not (off < 0):
                raise QhullError(f"QH6023 qhull input error: feasible point is not clearly inside halfspace")
            for j in range(D):
                dual[i, j] = Hs[i, j] / -off
        H = _hull_of(dual)
        ids, idmap = H.compact()
        nf = len(ids)
        facets = List[List[int]]()
        deq = np.empty((nf, D + 1))
        inter = np.empty((nf, D))
        vol = 0.0
        area = 0.0
        for a in range(nf):
            f = ids[a]
            facets.append([v for v in H.verts[f]])
            for j in range(D):
                deq[a, j] = H.normal[f][j]
            deq[a, D] = H.offset[f]
            for j in range(D):
                inter[a, j] = deq[a, j] / -deq[a, D] + ip[j]
            area += _facet_area(dual, H.verts[f], D)
            dc = H.offset[f]
            for j in range(D):
                dc += H.normal[f][j] * H.interior[j]
            vol += -dc * _facet_area(dual, H.verts[f], D) / float(D)
        self.halfspaces = Hs
        self.interior_point = ip
        self.dual_facets = facets
        self.dual_equations = deq
        self.dual_points = dual
        self.dual_volume = vol
        self.dual_area = area
        self.intersections = inter
        self.ndim = D
        self.nineq = m
        vs = List[int]()
        for fl in facets:
            for v in fl:
                vs.append(v)
        self.dual_vertices = np.unique(np.array(vs, dtype=int))
