# Ported from scipy 1.18.1 (BSD-3-Clause)

# Simplicial complex of shgo (scipy/optimize/_shgo_lib/_complex.py and
# _vertex.py), restricted to what shgo's 'simplicial' sampling uses
# (triangulate / refine / refine_all / refine_local_space / split_edge /
# vpool, scalar-field vertex cache).
#
# The order in which scipy refines the complex depends on the iteration
# order of Python `set`s of vertices (vertex neighbourhoods), which is
# determined by CPython's set hash table and the vertices' hashes
# (hash(tuple(x))). To reproduce scipy's sampling exactly, _PySet below
# emulates CPython 3.12's set implementation (open addressing, linear
# probes, perturbation, dummies, resizing and set copy/merge) and the
# vertex hash is computed with CPython's float and tuple hash algorithms.

import math

_MOD = u64(2305843009213693951)    # 2**61 - 1

def _py_hash_float(v: float) -> int:
    # CPython _Py_HashDouble
    if math.isinf(v):
        return 314159 if v > 0 else -314159
    if v != v:
        return 0
    m, e = math.frexp(v)
    sign = 1
    if m < 0:
        sign = -1
        m = -m
    x = u64(0)
    while m != 0.0:
        x = ((x << u64(28)) & _MOD) | (x >> u64(61 - 28))
        m *= 268435456.0
        e -= 28
        y = u64(int(m))
        m -= float(int(m))
        x += y
        if x >= _MOD:
            x -= _MOD
    if e >= 0:
        e = e % 61
    else:
        e = 61 - 1 - ((-1 - e) % 61)
    x = ((x << u64(e)) & _MOD) | (x >> u64(61 - e))
    r = int(x) * sign
    if r == -1:
        r = -2
    return r

def _py_hash_tuple(xs: List[float]) -> int:
    # CPython tuplehash (xxHash-based, 64-bit)
    P1 = u64(11400714785074694791)
    P2 = u64(14029467366897019727)
    P5 = u64(2870177450012600261)
    acc = P5
    for v in xs:
        lane = u64(_py_hash_float(v))
        acc += lane * P2
        acc = (acc << u64(31)) | (acc >> u64(33))
        acc *= P1
    acc += u64(len(xs)) ^ (P5 ^ u64(3527539))
    if acc == u64(18446744073709551615):
        return 1546275796
    return int(acc)

_EMPTY = -1
_DUMMY = -2

class _PySet:
    # CPython set of vertex ids (identity equality) with given hashes
    keys: List[int]
    hs: List[int]
    mask: int
    fill: int
    used: int

    def __init__(self):
        self.keys = [_EMPTY] * 8
        self.hs = [0] * 8
        self.mask = 7
        self.fill = 0
        self.used = 0

    def __len__(self) -> int:
        return self.used

    def items(self) -> List[int]:
        out = List[int](self.used)
        for k in self.keys:
            if k >= 0:
                out.append(k)
        return out

    def _insert_clean(self, key: int, h: int):
        mask = u64(self.mask)
        perturb = u64(h)
        i = u64(h) & mask
        while True:
            ii = int(i)
            if self.keys[ii] == _EMPTY:
                self.keys[ii] = key
                self.hs[ii] = h
                return
            if i + u64(9) <= mask:
                for j in range(1, 10):
                    if self.keys[ii + j] == _EMPTY:
                        self.keys[ii + j] = key
                        self.hs[ii + j] = h
                        return
            perturb >>= u64(5)
            i = (i * u64(5) + u64(1) + perturb) & mask

    def _resize(self, minused: int):
        newsize = 8
        while newsize <= minused:
            newsize <<= 1
        if newsize == 8 and self.mask == 7 and self.fill == self.used:
            return
        oldk = self.keys
        oldh = self.hs
        self.keys = [_EMPTY] * newsize
        self.hs = [0] * newsize
        self.mask = newsize - 1
        for t in range(len(oldk)):
            if oldk[t] >= 0:
                self._insert_clean(oldk[t], oldh[t])
        self.fill = self.used

    def _lookup(self, key: int, h: int) -> int:
        mask = u64(self.mask)
        perturb = u64(h)
        i = u64(h) & mask
        while True:
            ii = int(i)
            probes = 9 if i + u64(9) <= mask else 0
            for p in range(probes + 1):
                k = self.keys[ii + p]
                if k == _EMPTY:
                    return -1
                if k == key:
                    return ii + p
            perturb >>= u64(5)
            i = (i * u64(5) + u64(1) + perturb) & mask

    def contains(self, key: int, h: int) -> bool:
        return self._lookup(key, h) >= 0

    def add(self, key: int, h: int):
        mask = u64(self.mask)
        perturb = u64(h)
        i = u64(h) & mask
        freeslot = -1
        while True:
            ii = int(i)
            probes = 9 if i + u64(9) <= mask else 0
            for p in range(probes + 1):
                k = self.keys[ii + p]
                if k == _EMPTY:
                    if freeslot >= 0:
                        self.used += 1
                        self.keys[freeslot] = key
                        self.hs[freeslot] = h
                        return
                    self.fill += 1
                    self.used += 1
                    self.keys[ii + p] = key
                    self.hs[ii + p] = h
                    if self.fill * 5 < self.mask * 3:
                        return
                    self._resize(self.used * 2 if self.used > 50000 else self.used * 4)
                    return
                if k == key:
                    return
                if k == _DUMMY:
                    freeslot = ii + p
            perturb >>= u64(5)
            i = (i * u64(5) + u64(1) + perturb) & mask

    def discard(self, key: int, h: int) -> bool:
        pos = self._lookup(key, h)
        if pos < 0:
            return False
        self.keys[pos] = _DUMMY
        self.hs[pos] = -1
        self.used -= 1
        return True

    def update(self, other: _PySet):
        # set_merge
        if other.used == 0:
            return
        if (self.fill + other.used) * 5 >= self.mask * 3:
            self._resize((self.used + other.used) * 2)
        if self.fill == 0 and self.mask == other.mask and other.fill == other.used:
            for t in range(other.mask + 1):
                self.keys[t] = other.keys[t]
                self.hs[t] = other.hs[t] if other.keys[t] >= 0 else 0
                if other.keys[t] < 0:
                    self.keys[t] = _EMPTY
            self.fill = other.fill
            self.used = other.used
            return
        if self.fill == 0:
            self.fill = other.used
            self.used = other.used
            for t in range(other.mask + 1):
                k = other.keys[t]
                if k >= 0:
                    self._insert_clean(k, other.hs[t])
            return
        for t in range(other.mask + 1):
            k = other.keys[t]
            if k >= 0:
                self.add(k, other.hs[t])

    def copy(self) -> _PySet:
        s = _PySet()
        s.update(self)
        return s

def _xeq(a: List[float], b: List[float]) -> bool:
    if len(a) != len(b):
        return False
    for i in range(len(a)):
        if a[i] != b[i]:
            return False
    return True

class Complex:
    dim: int
    bounds: List[List[float]]
    # vertex storage (vertex id = creation index = scipy's v.index)
    vx: List[List[float]]
    vh: List[int]
    nn: List[_PySet]
    f: List[float]
    evaluated: List[bool]
    check_min: List[bool]
    is_min: List[bool]
    cp: Generator[int]
    has_cp: bool
    rls: Generator[int]
    has_rls: bool
    lookup: Dict[int, List[int]]
    split_memo: Dict[Tuple[int, int], int]
    nfev: int
    triangulated_vectors: List[Tuple[List[float], List[float]]]
    has_tv: bool
    origin: List[float]
    supremum: List[float]

    def __init__(self, dim: int, bounds: List[List[float]]):
        self.dim = dim
        self.bounds = bounds
        self.vx = List[List[float]]()
        self.vh = List[int]()
        self.nn = List[_PySet]()
        self.f = List[float]()
        self.evaluated = List[bool]()
        self.check_min = List[bool]()
        self.is_min = List[bool]()
        self.cp = self._nogen()
        self.has_cp = False
        self.rls = self._nogen()
        self.has_rls = False
        self.lookup = Dict[int, List[int]]()
        self.split_memo = Dict[Tuple[int, int], int]()
        self.nfev = 0
        self.triangulated_vectors = List[Tuple[List[float], List[float]]]()
        self.has_tv = False
        self.origin = [b[0] for b in bounds]
        self.supremum = [b[1] for b in bounds]

    def size(self) -> int:
        return len(self.vx)

    def find(self, x: List[float]) -> int:
        h = _py_hash_tuple(x)
        if h in self.lookup:
            for k in self.lookup[h]:
                if _xeq(self.vx[k], x):
                    return k
        return -1

    def V(self, x: List[float]) -> int:
        h = _py_hash_tuple(x)
        if h in self.lookup:
            for k in self.lookup[h]:
                if _xeq(self.vx[k], x):
                    return k
        else:
            self.lookup[h] = List[int]()
        k = len(self.vx)
        self.vx.append(list(x))
        self.vh.append(h)
        self.nn.append(_PySet())
        self.f.append(0.0)
        self.evaluated.append(False)
        self.check_min.append(True)
        self.is_min.append(False)
        self.lookup[h].append(k)
        return k

    def connect(self, a: int, b: int):
        if a != b and not self.nn[a].contains(b, self.vh[b]):
            self.nn[a].add(b, self.vh[b])
            self.nn[b].add(a, self.vh[a])
            self.check_min[a] = True
            self.check_min[b] = True

    def disconnect(self, a: int, b: int):
        if self.nn[a].contains(b, self.vh[b]):
            self.nn[a].discard(b, self.vh[b])
            self.nn[b].discard(a, self.vh[a])
            self.check_min[a] = True
            self.check_min[b] = True

    def star(self, a: int):
        # VertexBase.star(): adds the vertex to its own neighbourhood
        self.nn[a].add(a, self.vh[a])

    def _nogen(self):
        for i in range(0):
            yield 0

    def split_edge(self, v1: int, v2: int) -> int:
        # functools.cache(_split_edge): keyed on the (ordered) vertex pair
        if (v1, v2) in self.split_memo:
            return self.split_memo[(v1, v2)]
        self.disconnect(v1, v2)
        p = self.vx[v1]
        q = self.vx[v2]
        vct = [(q[i] - p[i]) / 2.0 + p[i] for i in range(len(p))]
        vc = self.V(vct)
        self.connect(vc, v1)
        self.connect(vc, v2)
        self.split_memo[(v1, v2)] = vc
        return vc

    def set_copy(self, a: int) -> _PySet:
        return self.nn[a].copy()

    def vpool(self, origin: List[float], supremum: List[float]) -> _PySet:
        vo = self.V(origin)
        vs = self.V(supremum)
        bl = list(origin)
        bu = list(supremum)
        for i in range(len(origin)):
            if bl[i] > supremum[i]:
                bl[i] = supremum[i]
            if bu[i] < origin[i]:
                bu[i] = origin[i]
        pool = _PySet()
        pool.update(self.nn[vo])
        pool.update(self.nn[vs])
        cp = pool.copy()
        for vn in cp.items():
            xn = self.vx[vn]
            for i in range(len(xn)):
                if not (bl[i] <= xn[i] <= bu[i]):
                    pool.discard(vn, self.vh[vn])
        return pool

    def minimiser(self, a: int) -> bool:
        # VertexScalarField.minimiser (result cached until the
        # neighbourhood changes through connect/disconnect)
        if self.check_min[a]:
            fa = self.f[a]
            r = True
            for b in self.nn[a].items():
                if not (fa < self.f[b]):
                    r = False
                    break
            self.is_min[a] = r
            self.check_min[a] = False
        return self.is_min[a]

    # ------------------------------------------------- initial triangulation
    def cyclic_product(self, centroid: bool):
        vot = self.origin
        vut = self.supremum
        n = len(vot)
        vo = self.V(vot)
        yield vo
        self.connect(self.V(vut), self.V(vot))
        yield 0
        C0x = [[vo]]
        a = list(vot)
        a[0] = vut[0]
        a_vo = self.V(a)
        self.connect(vo, a_vo)
        yield a_vo
        C1x = [[a_vo]]
        ab_C = List[Tuple[int, int]]()
        for i in range(n - 1):
            C0x.append(List[int]())
            C1x.append(List[int]())
            cC0x = [list(x) for x in C0x[:i + 1]]
            cC1x = [list(x) for x in C1x[:i + 1]]
            for j in range(len(cC0x)):
                VL = cC0x[j]
                VU = cC1x[j]
                for k in range(min(len(VL), len(VU))):
                    vl = VL[k]
                    vu = VU[k]
                    xl = list(self.vx[vl])
                    xu = list(self.vx[vu])
                    xl[i + 1] = vut[i + 1]
                    xu[i + 1] = vut[i + 1]
                    a_vl = self.V(xl)
                    self.connect(vl, a_vl)
                    yield a_vl
                    a_vu = self.V(xu)
                    self.connect(vu, a_vu)
                    self.connect(a_vl, a_vu)
                    self.connect(vl, a_vu)
                    ab_C.append((vl, a_vu))
                    C0x[i + 1].append(vl)
                    C0x[i + 1].append(vu)
                    C1x[i + 1].append(a_vl)
                    C1x[i + 1].append(a_vu)
                    C0x[j].append(a_vl)
                    C1x[j].append(a_vu)
                    yield a_vu
            ab_Cc = list(ab_C)
            for vp in ab_Cc:
                b = list(self.vx[vp[0]])
                ab = list(self.vx[vp[1]])
                b[i + 1] = vut[i + 1]
                ab[i + 1] = vut[i + 1]
                b_v = self.V(b)
                ab_v = self.V(ab)
                self.connect(vp[0], ab_v)
                self.connect(b_v, ab_v)
                ab_C.append((vp[0], ab_v))
                ab_C.append((b_v, ab_v))
        if centroid:
            vo = self.V(vot)
            vs = self.V(vut)
            self.disconnect(vo, vs)
            vc = self.split_edge(vo, vs)
            for v in self.nn[vo].items():
                self.connect(v, vc)
            yield vc
        else:
            yield 0

    def _add_tv(self, o: List[float], s: List[float]):
        self.triangulated_vectors.append((list(o), list(s)))
        self.has_tv = True

    def triangulate(self, n: int, centroid: bool = True):
        # n < 0 means None
        if n < 0:
            self.cp = self.cyclic_product(centroid)
            self.has_cp = True
            for _ in self.cp:
                pass
            self._add_tv(self.origin, self.supremum)
        else:
            if not self.has_cp:
                self.cp = self.cyclic_product(centroid)
                self.has_cp = True
            try:
                while self.size() < n:
                    next(self.cp)
            except StopIteration:
                self._add_tv(self.origin, self.supremum)

    def refine(self, n: int):
        if n < 0:
            if self.has_tv:
                self.refine_all()
            else:
                self.triangulate(-1)
            return
        nt = self.size() + n
        while self.size() < nt:
            if self.has_tv:
                ok = False
                if self.has_rls:
                    try:
                        next(self.rls)
                        ok = True
                    except StopIteration:
                        ok = False
                if not ok:
                    vp = self.triangulated_vectors[0]
                    self.rls = self.refine_local_space(list(vp[0]), list(vp[1]))
                    self.has_rls = True
                    next(self.rls)
            else:
                self.triangulate(nt)

    def refine_all(self, centroids: bool = True):
        if self.has_tv:
            tvs = list(self.triangulated_vectors)
            for vp in tvs:
                self.rls = self.refine_local_space(list(vp[0]), list(vp[1]))
                self.has_rls = True
                for _ in self.rls:
                    pass
        else:
            self.triangulate(-1, centroids)

    def _comb_split(self, comb: List[int]):
        for a in range(len(comb)):
            for b in range(a + 1, len(comb)):
                self.split_edge(comb[a], comb[b])

    def _setx(self, v: int, i: int, val: float) -> List[float]:
        x = list(self.vx[v])
        x[i] = val
        return x

    # ------------------------------------------------------- refinement
    def refine_local_space(self, origin: List[float], supremum: List[float]):
        # Python's function-scoped loop variables leak between the loops of
        # the original; they are kept as explicit locals here.
        origin_c = list(origin)
        supremum_c = list(supremum)
        n = len(origin)
        vl = -1
        vu = -1
        a_vu = -1
        vc = -1
        s_origin = list(origin)
        s_supremum = list(supremum)
        for i in range(n):
            if origin[i] > supremum[i]:
                s_origin[i] = supremum[i]
                s_supremum[i] = origin[i]
        vot = s_origin
        vut = s_supremum
        vo = self.V(vot)
        vs = self.V(vut)
        vco = self.split_edge(vo, vs)
        sup_set = self.nn[vco].copy()
        t = list(vot)
        t[0] = vut[0]
        a_vl = self.V(t)
        c_v = self.split_edge(vo, a_vl)
        self.connect(c_v, vco)
        yield 0
        Cox = [[vo]]
        Ccx = [[c_v]]
        Cux = [[a_vl]]
        ab_C = List[List[int]]()
        s_ab_C = List[List[int]]()
        for i in range(n - 1):
            ii = i + 1
            Cox.append(List[int]())
            Ccx.append(List[int]())
            Cux.append(List[int]())
            t_a_vl = list(vot)
            t_a_vl[ii] = vut[ii]
            cCox = [list(x) for x in Cox[:i + 1]]
            cCcx = [list(x) for x in Ccx[:i + 1]]
            cCux = [list(x) for x in Cux[:i + 1]]
            ab_Cc = list(ab_C)
            s_ab_Cc = list(s_ab_C)
            if self.find(t_a_vl) >= 0 and self.find(vut) >= 0:
                for vectors in s_ab_Cc:
                    bc_vc = self.V(self._setx(vectors[0], ii, vut[ii]))
                    self.connect(bc_vc, vco)
                    yield 0
                    d_bc_vc = self.split_edge(vectors[0], bc_vc)
                    self.connect(d_bc_vc, bc_vc)
                    self.connect(d_bc_vc, vectors[1])
                    self.connect(d_bc_vc, vectors[2])
                    self.connect(d_bc_vc, vectors[3])
                    yield 0
                    b_vl = self.V(self._setx(vectors[1], ii, vut[ii]))
                    self.connect(bc_vc, b_vl)
                    self.connect(d_bc_vc, b_vl)
                    yield 0
                    b_vu = self.V(self._setx(vectors[2], ii, vut[ii]))
                    self.connect(bc_vc, b_vu)
                    self.connect(d_bc_vc, b_vu)
                    b_vl_c = self.split_edge(b_vu, b_vl)
                    self.connect(bc_vc, b_vl_c)
                    yield 0
                    ba_vu = self.V(self._setx(vectors[3], ii, vut[ii]))
                    self.connect(bc_vc, ba_vu)
                    self.connect(d_bc_vc, ba_vu)
                    self.split_edge(vectors[1], ba_vu)
                    self.split_edge(b_vl, ba_vu)
                    b_vu_c = self.split_edge(b_vu, ba_vu)
                    self.connect(bc_vc, b_vu_c)
                    yield 0
                    yield 0
                    yield 0
                    d_bc_vc = self.split_edge(vectors[0], bc_vc)
                    self.connect(d_bc_vc, vco)
                    yield 0
                    d_b_vl = self.split_edge(vectors[1], b_vl)
                    self.connect(d_bc_vc, vco)
                    self.connect(d_bc_vc, d_b_vl)
                    yield 0
                    d_b_vu = self.split_edge(vectors[2], b_vu)
                    self.connect(d_bc_vc, vco)
                    self.connect(d_bc_vc, d_b_vu)
                    yield 0
                    d_ba_vu = self.split_edge(vectors[3], ba_vu)
                    self.connect(d_bc_vc, vco)
                    self.connect(d_bc_vc, d_ba_vu)
                    yield 0
                    self._comb_split([vl, vu, a_vu, b_vl, b_vu, ba_vu])
                    ab_C.append([d_bc_vc, vectors[1], b_vl, a_vu, ba_vu])
                    ab_C.append([d_bc_vc, vl, b_vl, a_vu, ba_vu])
                for vectors in ab_Cc:
                    bc_vc = self.V(self._setx(vectors[0], ii, vut[ii]))
                    self.connect(bc_vc, vco)
                    yield 0
                    d_bc_vc = self.split_edge(vectors[0], bc_vc)
                    self.connect(d_bc_vc, bc_vc)
                    self.connect(d_bc_vc, vectors[1])
                    self.connect(d_bc_vc, vectors[2])
                    self.connect(d_bc_vc, vectors[3])
                    self.connect(d_bc_vc, vectors[4])
                    yield 0
                    b_vl = self.V(self._setx(vectors[1], ii, vut[ii]))
                    self.connect(bc_vc, b_vl)
                    self.connect(d_bc_vc, b_vl)
                    yield 0
                    b_vu = self.V(self._setx(vectors[2], ii, vut[ii]))
                    self.connect(bc_vc, b_vu)
                    self.connect(d_bc_vc, b_vu)
                    yield 0
                    ba_vl = self.V(self._setx(vectors[3], ii, vut[ii]))
                    self.connect(bc_vc, ba_vl)
                    self.connect(d_bc_vc, ba_vl)
                    self.split_edge(b_vu, ba_vl)
                    yield 0
                    ba_vu = self.V(self._setx(vectors[4], ii, vut[ii]))
                    self.connect(bc_vc, ba_vu)
                    self.connect(d_bc_vc, ba_vu)
                    self.split_edge(vectors[1], ba_vu)
                    self.split_edge(b_vl, ba_vu)
                    yield 0
                    yield 0
                    yield 0
                    d_bc_vc = self.split_edge(vectors[0], bc_vc)
                    self.connect(d_bc_vc, vco)
                    yield 0
                    d_b_vl = self.split_edge(vectors[1], b_vl)
                    self.connect(d_bc_vc, vco)
                    self.connect(d_bc_vc, d_b_vl)
                    yield 0
                    d_b_vu = self.split_edge(vectors[2], b_vu)
                    self.connect(d_bc_vc, vco)
                    self.connect(d_bc_vc, d_b_vu)
                    yield 0
                    d_ba_vl = self.split_edge(vectors[3], ba_vl)
                    self.connect(d_bc_vc, vco)
                    self.connect(d_bc_vc, d_ba_vl)
                    yield 0
                    d_ba_vu = self.split_edge(vectors[4], ba_vu)
                    self.connect(d_bc_vc, vco)
                    self.connect(d_bc_vc, d_ba_vu)
                    yield 0
                    vl = vectors[1]
                    vu = vectors[2]
                    a_vl = vectors[3]
                    a_vu = vectors[4]
                    self._comb_split([vl, vu, a_vl, a_vu, b_vl, b_vu, ba_vl, ba_vu])
                    ab_C.append([bc_vc, b_vl, b_vu, ba_vl, ba_vu])
                    ab_C.append([d_bc_vc, d_b_vl, d_b_vu, d_ba_vl, d_ba_vu])
                    ab_C.append([d_bc_vc, vectors[1], b_vl, a_vu, ba_vu])
                    ab_C.append([d_bc_vc, vu, b_vu, a_vl, ba_vl])
                for j in range(len(cCox)):
                    VL = cCox[j]
                    VC = cCcx[j]
                    VU = cCux[j]
                    for k in range(min(len(VL), len(VC), len(VU))):
                        vl = VL[k]
                        vc = VC[k]
                        vu = VU[k]
                        a_vl = self.V(self._setx(vl, ii, vut[ii]))
                        a_vu = self.V(self._setx(vu, ii, vut[ii]))
                        c_vc = self.split_edge(vl, a_vu)
                        self.split_edge(vl, vu)
                        self.connect(c_vc, vco)
                        self.connect(c_vc, vc)
                        self.connect(c_vc, vl)
                        self.connect(c_vc, vu)
                        self.connect(c_vc, a_vl)
                        self.connect(c_vc, a_vu)
                        yield 0
                        c_vl = self.split_edge(vl, a_vl)
                        self.connect(c_vl, vco)
                        self.connect(c_vc, c_vl)
                        yield 0
                        c_vu = self.split_edge(vu, a_vu)
                        self.connect(c_vu, vco)
                        self.connect(c_vc, c_vu)
                        yield 0
                        a_vc = self.split_edge(a_vl, a_vu)
                        self.connect(a_vc, vco)
                        self.connect(a_vc, c_vc)
                        ab_C.append([c_vc, vl, vu, a_vl, a_vu])
                        Cox[ii].append(vl)
                        Cox[ii].append(vc)
                        Cox[ii].append(vu)
                        Ccx[ii].append(c_vl)
                        Ccx[ii].append(c_vc)
                        Ccx[ii].append(c_vu)
                        Cux[ii].append(a_vl)
                        Cux[ii].append(a_vc)
                        Cux[ii].append(a_vu)
                        Cox[j].append(c_vl)
                        Cox[j].append(a_vl)
                        Ccx[j].append(c_vc)
                        Ccx[j].append(a_vc)
                        Cux[j].append(c_vu)
                        Cux[j].append(a_vu)
                        yield 0
            else:
                for vectors in ab_Cc:
                    ba_vl_x = self._setx(vectors[3], ii, vut[ii])
                    ba_vu = self.V(self._setx(vectors[4], ii, vut[ii]))
                    yield 0
                    d_bc_vc = self.split_edge(vectors[1], ba_vu)
                    yield 0
                    self.connect(d_bc_vc, vectors[1])
                    self.connect(d_bc_vc, vectors[2])
                    self.connect(d_bc_vc, vectors[3])
                    self.connect(d_bc_vc, vectors[4])
                    yield 0
                    ba_vl = self.V(ba_vl_x)
                    yield 0
                    d_ba_vl = self.split_edge(vectors[3], ba_vl)
                    d_ba_vu = self.split_edge(vectors[4], ba_vu)
                    self.split_edge(d_ba_vl, d_ba_vu)
                    yield 0
                    yield 0
                    yield 0
                    vl = vectors[1]
                    vu = vectors[2]
                    a_vl = vectors[3]
                    a_vu = vectors[4]
                    self._comb_split([vl, vu, a_vl, a_vu, ba_vl, ba_vu])
                VL = Cox[i]
                VC = Ccx[i]
                VU = Cux[i]
                k = 0
                while k < len(VL) and k < len(VC) and k < len(VU):
                    # (zip over lists that grow while iterating)
                    vl = VL[k]
                    vc = VC[k]
                    vu = VU[k]
                    a_vu = self.V(self._setx(vu, ii, vut[ii]))
                    yield 0
                    c_vc = self.split_edge(vl, a_vu)
                    self.split_edge(vl, vu)
                    self.connect(c_vc, vco)
                    self.connect(c_vc, vc)
                    self.connect(c_vc, vl)
                    self.connect(c_vc, vu)
                    self.connect(c_vc, a_vu)
                    yield 0
                    c_vu = self.split_edge(vu, a_vu)
                    self.connect(c_vu, vco)
                    self.connect(c_vc, c_vu)
                    yield 0
                    Cox[ii].append(vu)
                    Ccx[ii].append(c_vu)
                    Cux[ii].append(a_vu)
                    s_ab_C.append([c_vc, vl, vu, a_vu])
                    yield 0
                    k += 1
        # bookkeeping of the sub-cubes still to be refined
        for t2 in range(len(self.triangulated_vectors)):
            tv = self.triangulated_vectors[t2]
            if _xeq(tv[0], origin_c) and _xeq(tv[1], supremum_c):
                self.triangulated_vectors.pop(t2)
                break
        sup = sup_set.items()
        for vs2 in sup:
            self._add_tv(self.vx[vco], self.vx[vs2])
        c_nn_lists = List[_PySet]()
        for vs2 in sup:
            c_nn_lists.append(self.vpool(self.vx[vco], self.vx[vs2]))
        for q in range(len(sup)):
            vcn = self.split_edge(vco, sup[q])
            for vnn in c_nn_lists[q].items():
                self.connect(vcn, vnn)
            yield 0
        yield 0

    # ---------------------------------------------------- field evaluation
    def process_pools(self, func, args):
        # VertexCacheField.process_pools (no constraints): evaluate all new
        # vertices, then refresh the cached minimiser flags of all vertices
        for k in range(self.size()):
            if not self.evaluated[k]:
                v = _eval_field(func, self.vx[k], args)
                self.nfev += 1
                if v != v:
                    v = math.inf
                self.f[k] = v
                self.evaluated[k] = True
        for k in range(self.size()):
            self.minimiser(k)

def _eval_field(func, x: List[float], args) -> float:
    import numpy as np
    from ._common import _call_scalar
    return _call_scalar(func, np.array(x, float), args)
