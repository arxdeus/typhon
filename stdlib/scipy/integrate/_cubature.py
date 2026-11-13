# Copyright (C) 2026 the Typhon authors
# Ported from scipy 1.18.1 (BSD-3-Clause)

# scipy.integrate.cubature: adaptive multidimensional cubature with
# Gauss-Kronrod product rules ("gk15", "gk21") and the Genz-Malik rule
# (scipy/integrate/_cubature.py and _rules/*). The integrand takes an array
# of nodes with shape (npoints, ndim) and returns a 1-D array of npoints
# values (scalar-valued integrands).

import math
import numpy as np
from heapq import heappush, heappop
from scipy.special import roots_legendre

_N21 = [0.995657163025808080735527280689003, 0.973906528517171720077964012084452, 0.930157491355708226001207180059508, 0.865063366688984510732096688423493, 0.780817726586416897063717578345042, 0.679409568299024406234327365114874, 0.562757134668604683339000099272694, 0.433395394129247190799265943165784, 0.294392862701460198131126603103866, 0.148874338981631210884826001129720, 0, -0.148874338981631210884826001129720, -0.294392862701460198131126603103866, -0.433395394129247190799265943165784, -0.562757134668604683339000099272694, -0.679409568299024406234327365114874, -0.780817726586416897063717578345042, -0.865063366688984510732096688423493, -0.930157491355708226001207180059508, -0.973906528517171720077964012084452, -0.995657163025808080735527280689003]
_W21 = [0.011694638867371874278064396062192, 0.032558162307964727478818972459390, 0.054755896574351996031381300244580, 0.075039674810919952767043140916190, 0.093125454583697605535065465083366, 0.109387158802297641899210590325805, 0.123491976262065851077958109831074, 0.134709217311473325928054001771707, 0.142775938577060080797094273138717, 0.147739104901338491374841515972068, 0.149445554002916905664936468389821, 0.147739104901338491374841515972068, 0.142775938577060080797094273138717, 0.134709217311473325928054001771707, 0.123491976262065851077958109831074, 0.109387158802297641899210590325805, 0.093125454583697605535065465083366, 0.075039674810919952767043140916190, 0.054755896574351996031381300244580, 0.032558162307964727478818972459390, 0.011694638867371874278064396062192]
_N15 = [0.991455371120812639206854697526329, 0.949107912342758524526189684047851, 0.864864423359769072789712788640926, 0.741531185599394439863864773280788, 0.586087235467691130294144838258730, 0.405845151377397166906606412076961, 0.207784955007898467600689403773245, 0.000000000000000000000000000000000, -0.207784955007898467600689403773245, -0.405845151377397166906606412076961, -0.586087235467691130294144838258730, -0.741531185599394439863864773280788, -0.864864423359769072789712788640926, -0.949107912342758524526189684047851, -0.991455371120812639206854697526329]
_W15 = [0.022935322010529224963732008058970, 0.063092092629978553290700663189204, 0.104790010322250183839876322541518, 0.140653259715525918745189590510238, 0.169004726639267902826583426598550, 0.190350578064785409913256402421014, 0.204432940075298892414161999234649, 0.209482141084727828012999174891714, 0.204432940075298892414161999234649, 0.190350578064785409913256402421014, 0.169004726639267902826583426598550, 0.140653259715525918745189590510238, 0.104790010322250183839876322541518, 0.063092092629978553290700663189204, 0.022935322010529224963732008058970]

class _Rule:
    """Nested fixed rule on [-1, 1]^ndim: nodes (n, ndim), weights (n,),
    lower-order nodes and weights for the error estimate."""
    ndim: int
    nodes: np.ndarray[float, 2]
    weights: np.ndarray[float, 1]
    lnodes: np.ndarray[float, 2]
    lweights: np.ndarray[float, 1]

    def __init__(self, ndim: int, nodes: np.ndarray[float, 2], weights: np.ndarray[float, 1],
                 lnodes: np.ndarray[float, 2], lweights: np.ndarray[float, 1]):
        self.ndim = ndim
        self.nodes = nodes
        self.weights = weights
        self.lnodes = lnodes
        self.lweights = lweights

def _apply_fixed_rule(f, a: np.ndarray[float, 1], b: np.ndarray[float, 1],
                      nodes: np.ndarray[float, 2], weights: np.ndarray[float, 1], args) -> float:
    nd = a.shape[0]
    if nodes.shape[1] != nd or b.shape[0] != nd:
        raise ValueError(f"rule and function are of incompatible dimension, nodes have ndim {nodes.shape[1]}, while limit of integration has ndim a_ndim={nd}, b_ndim={b.shape[0]}")
    n = nodes.shape[0]
    lengths = b - a
    pts = np.empty((n, nd))
    for i in range(n):
        for k in range(nd):
            pts[i, k] = (nodes[i, k] + 1) * (lengths[k] * 0.5) + a[k]
    prodl = np.prod(lengths)
    scale = prodl / float(2 ** nd)
    w = weights * scale
    fv = np.asarray(f(pts, *args), dtype=float)
    return float(np.sum(w * fv))

def _rule_estimate(r: _Rule, f, a: np.ndarray[float, 1], b: np.ndarray[float, 1], args) -> float:
    return _apply_fixed_rule(f, a, b, r.nodes, r.weights, args)

def _rule_error(r: _Rule, f, a: np.ndarray[float, 1], b: np.ndarray[float, 1], args) -> float:
    n1 = r.nodes.shape[0]
    n2 = r.lnodes.shape[0]
    nd = r.nodes.shape[1]
    nodes = np.empty((n1 + n2, nd))
    w = np.empty(n1 + n2)
    for i in range(n1):
        for k in range(nd):
            nodes[i, k] = r.nodes[i, k]
        w[i] = r.weights[i]
    for i in range(n2):
        for k in range(nd):
            nodes[n1 + i, k] = r.lnodes[i, k]
        w[n1 + i] = -r.lweights[i]
    return abs(_apply_fixed_rule(f, a, b, nodes, w, args))

def _cartesian(arrs: List[List[float]]):
    """Cartesian product (meshgrid 'ij' order) as rows."""
    nd = len(arrs)
    total = 1
    for x in arrs:
        total *= len(x)
    out = np.empty((total, nd))
    rep = total
    for k in range(nd):
        m = len(arrs[k])
        rep = rep // m
        for i in range(total):
            out[i, k] = arrs[k][(i // rep) % m]
    return out

def _gk_rule(npoints: int, ndim: int) -> _Rule:
    nodes1 = _N21 if npoints == 21 else _N15
    w1 = _W21 if npoints == 21 else _W15
    gx, gw = roots_legendre(npoints // 2)
    ln = [float(v) for v in gx]
    lw = [float(v) for v in gw]
    nodes = _cartesian([nodes1 for _ in range(ndim)])
    wmat = _cartesian([w1 for _ in range(ndim)])
    weights = np.empty(wmat.shape[0])
    for i in range(wmat.shape[0]):
        p = wmat[i, 0]
        for k in range(1, ndim):
            p = p * wmat[i, k]
        weights[i] = p
    lnodes = _cartesian([ln for _ in range(ndim)])
    lmat = _cartesian([lw for _ in range(ndim)])
    lweights = np.empty(lmat.shape[0])
    for i in range(lmat.shape[0]):
        p = lmat[i, 0]
        for k in range(1, ndim):
            p = p * lmat[i, k]
        lweights[i] = p
    return _Rule(ndim, nodes, weights, lnodes, lweights)

def _distinct_permutations(items_in: List[float]) -> List[List[float]]:
    items = sorted(items_in)
    size = len(items)
    out = List[List[float]]()
    while True:
        out.append(list(items))
        i = size - 2
        found = False
        while i >= 0:
            if items[i] < items[i + 1]:
                found = True
                break
            i -= 1
        if not found:
            return out
        j = size - 1
        while j > i:
            if items[i] < items[j]:
                break
            j -= 1
        t = items[i]
        items[i] = items[j]
        items[j] = t
        tail = items[i + 1:]
        tail.reverse()
        for q in range(len(tail)):
            items[i + 1 + q] = tail[q]

def _gm_points(ndim: int, with_l5: bool) -> List[List[float]]:
    l2 = math.sqrt(9 / 70)
    l3 = math.sqrt(9 / 10)
    l4 = math.sqrt(9 / 10)
    l5 = math.sqrt(9 / 19)
    pts = List[List[float]]()
    pts.append([0.0] * ndim)
    base1 = [l2] + [0.0] * (ndim - 1)
    for p in _distinct_permutations(base1):
        pts.append(p)
    base1 = [-l2] + [0.0] * (ndim - 1)
    for p in _distinct_permutations(base1):
        pts.append(p)
    base1 = [l3] + [0.0] * (ndim - 1)
    for p in _distinct_permutations(base1):
        pts.append(p)
    base1 = [-l3] + [0.0] * (ndim - 1)
    for p in _distinct_permutations(base1):
        pts.append(p)
    base2 = [l4, l4] + [0.0] * (ndim - 2)
    for p in _distinct_permutations(base2):
        pts.append(p)
    base2 = [l4, -l4] + [0.0] * (ndim - 2)
    for p in _distinct_permutations(base2):
        pts.append(p)
    base2 = [-l4, -l4] + [0.0] * (ndim - 2)
    for p in _distinct_permutations(base2):
        pts.append(p)
    if with_l5:
        total = 1 << ndim
        for m in range(total):
            row = [0.0] * ndim
            for k in range(ndim):
                bit = (m >> (ndim - 1 - k)) & 1
                row[k] = l5 if bit == 0 else -l5
            pts.append(row)
    return pts

def _gm_rule(ndim: int) -> _Rule:
    if ndim < 2:
        raise ValueError("Genz-Malik cubature is only defined for ndim >= 2")
    hp = _gm_points(ndim, True)
    lp = _gm_points(ndim, False)
    nodes = np.empty((len(hp), ndim))
    for i in range(len(hp)):
        for k in range(ndim):
            nodes[i, k] = hp[i][k]
    lnodes = np.empty((len(lp), ndim))
    for i in range(len(lp)):
        for k in range(ndim):
            lnodes[i, k] = lp[i][k]
    n = float(ndim)
    p2 = float(2 ** ndim)
    w1 = p2 * (12824 - 9120 * n + (400 * n * n)) / 19683
    w2 = p2 * 980 / 6561
    w3 = p2 * (1820 - 400 * n) / 19683
    w4 = p2 * (200 / 19683)
    w5 = 6859 / 19683
    weights = np.empty(len(hp))
    c = 0
    weights[c] = w1
    c += 1
    for _ in range(2 * ndim):
        weights[c] = w2
        c += 1
    for _ in range(2 * ndim):
        weights[c] = w3
        c += 1
    for _ in range(2 * (ndim - 1) * ndim):
        weights[c] = w4
        c += 1
    for _ in range(2 ** ndim):
        weights[c] = w5
        c += 1
    lw1 = p2 * (729 - 950 * n + 50 * n * n) / 729
    lw2 = p2 * (245 / 486)
    lw3 = p2 * (265 - 100 * n) / 1458
    lw4 = p2 * (25 / 729)
    lweights = np.empty(len(lp))
    c = 0
    lweights[c] = lw1
    c += 1
    for _ in range(2 * ndim):
        lweights[c] = lw2
        c += 1
    for _ in range(2 * ndim):
        lweights[c] = lw3
        c += 1
    for _ in range(2 * (ndim - 1) * ndim):
        lweights[c] = lw4
        c += 1
    return _Rule(ndim, nodes, weights, lnodes, lweights)

def _split_subregion(a: np.ndarray[float, 1], b: np.ndarray[float, 1],
                     split_at: np.ndarray[float, 1]):
    nd = a.shape[0]
    total = 1 << nd
    out = List[Tuple[np.ndarray[float, 1], np.ndarray[float, 1]]]()
    for m in range(total):
        sa = np.empty(nd)
        sb = np.empty(nd)
        for k in range(nd):
            bit = (m >> (nd - 1 - k)) & 1
            if bit == 0:
                sa[k] = a[k]
                sb[k] = split_at[k]
            else:
                sa[k] = split_at[k]
                sb[k] = b[k]
        out.append((sa, sb))
    return out

def _midpoint(a: np.ndarray[float, 1], b: np.ndarray[float, 1]) -> np.ndarray[float, 1]:
    return (a + b) / 2

class CubatureRegion:
    estimate: float
    error: float
    a: np.ndarray[float, 1]
    b: np.ndarray[float, 1]

    def __init__(self, estimate: float, error: float, a: np.ndarray[float, 1],
                 b: np.ndarray[float, 1]):
        self.estimate = estimate
        self.error = error
        self.a = a
        self.b = b

    def __lt__(self, other) -> bool:
        return abs(self.error) > abs(other.error)

class CubatureResult:
    estimate: float
    error: float
    status: str
    regions: List[CubatureRegion]
    subdivisions: int
    atol: float
    rtol: float

    def __init__(self, estimate: float, error: float, status: str, regions: List[CubatureRegion],
                 subdivisions: int, atol: float, rtol: float):
        self.estimate = estimate
        self.error = error
        self.status = status
        self.regions = regions
        self.subdivisions = subdivisions
        self.atol = atol
        self.rtol = rtol

class _InfiniteLimitsTransform[F]:
    f: F
    orig_a: np.ndarray[float, 1]
    orig_b: np.ndarray[float, 1]
    double_inf: List[bool]
    semi_inf: List[bool]
    num_inf: int

    def __init__(self, f: F, a: np.ndarray[float, 1], b: np.ndarray[float, 1]):
        self.f = f
        nd = a.shape[0]
        self.orig_a = a
        self.orig_b = b
        self.double_inf = [False] * nd
        self.semi_inf = [False] * nd
        ninf = 0
        for k in range(nd):
            dbl = a[k] == -math.inf and b[k] == math.inf
            start_inf = a[k] != -math.inf and b[k] == math.inf
            end_inf = a[k] == -math.inf and b[k] != math.inf
            self.double_inf[k] = dbl
            self.semi_inf[k] = start_inf or end_inf
            if end_inf:
                ta = -b[k]
                tb = -a[k]
                self.orig_a[k] = ta
                self.orig_b[k] = tb
            if dbl or start_inf or end_inf:
                ninf += 1
        self.num_inf = ninf

    def transformed_limits(self):
        nd = self.orig_a.shape[0]
        a = self.orig_a.copy()
        b = self.orig_b.copy()
        for k in range(nd):
            if self.double_inf[k]:
                a[k] = -1.0
                b[k] = 1.0
            if self.semi_inf[k]:
                a[k] = 0.0
                b[k] = 1.0
        return a, b

    def points(self) -> List[np.ndarray[float, 1]]:
        if self.num_inf != 0:
            return [np.zeros(self.orig_a.shape[0])]
        return List[np.ndarray[float, 1]]()

    def inv(self, x: np.ndarray[float, 1]) -> np.ndarray[float, 1]:
        t = x.copy()
        for k in range(x.shape[0]):
            if self.double_inf[k]:
                if x[k] == 0:
                    t[k] = math.inf
                else:
                    sg = 1.0 if x[k] > 0 else -1.0
                    t[k] = 1 / (x[k] + sg)
            if self.semi_inf[k]:
                t[k] = 1 / (x[k] - self.orig_a[k] + 1)
        return t

    def __call__(self, t: np.ndarray[float, 2], *args):
        npts = t.shape[0]
        nd = t.shape[1]
        x = t.copy()
        jac = np.ones(npts)
        for i in range(npts):
            p = 1.0
            first = True
            for k in range(nd):
                if self.double_inf[k]:
                    x[i, k] = (1 - abs(t[i, k])) / t[i, k]
                if self.semi_inf[k]:
                    x[i, k] = self.orig_a[k] + (1 - t[i, k]) / t[i, k]
                if self.double_inf[k] or self.semi_inf[k]:
                    q = t[i, k] ** 2
                    if first:
                        p = q
                        first = False
                    else:
                        p = p * q
            jac[i] = 1 / p if not first else 1.0
        fx = np.asarray(self.f(x, *args), dtype=float)
        return fx * jac

class _Plain[F]:
    f: F

    def __init__(self, f: F):
        self.f = f

    def __call__(self, x: np.ndarray[float, 2], *args):
        return self.f(x, *args)

def _is_strictly_in_region(a: np.ndarray[float, 1], b: np.ndarray[float, 1],
                           point: np.ndarray[float, 1]) -> bool:
    eqa = True
    eqb = True
    inside = True
    for k in range(a.shape[0]):
        if point[k] != a[k]:
            eqa = False
        if point[k] != b[k]:
            eqb = False
        if not (a[k] <= point[k] and point[k] <= b[k]):
            inside = False
    if eqa or eqb:
        return False
    return inside

def _split_region_at_points(a: np.ndarray[float, 1], b: np.ndarray[float, 1],
                            points: List[np.ndarray[float, 1]]):
    regions = [(a, b)]
    for point in points:
        has_inf = False
        for k in range(point.shape[0]):
            if math.isinf(point[k]):
                has_inf = True
        if has_inf:
            continue
        new_sub = List[Tuple[np.ndarray[float, 1], np.ndarray[float, 1]]]()
        for ak, bk in regions:
            if _is_strictly_in_region(ak, bk, point):
                subs = _split_subregion(ak, bk, point)
                for left, right in subs:
                    eq = False
                    for k in range(left.shape[0]):
                        if left[k] == right[k]:
                            eq = True
                    if eq:
                        continue
                    new_sub.append((left, right))
                # scipy's trailing extend(subregions) is a no-op: the generator is already consumed
            else:
                new_sub.append((ak, bk))
        regions = new_sub
    return regions

def _cubature_run(f, rule: _Rule, a_in: np.ndarray[float, 1], b_in: np.ndarray[float, 1],
                  rtol: float, atol: float, max_subdivisions: int, args,
                  points_in: List[np.ndarray[float, 1]]) -> CubatureResult:
    nd = a_in.shape[0]
    nflip = 0
    for k in range(nd):
        if a_in[k] > b_in[k]:
            nflip += 1
    sign = -1.0 if nflip % 2 == 1 else 1.0
    a = np.minimum(a_in, b_in)
    b = np.maximum(a_in, b_in)
    regions = List[CubatureRegion]()
    est = 0.0
    err = 0.0
    if len(points_in) == 0:
        initial = [(a, b)]
    else:
        initial = _split_region_at_points(a, b, points_in)
    for ak, bk in initial:
        est_k = _rule_estimate(rule, f, ak, bk, args)
        err_k = _rule_error(rule, f, ak, bk, args)
        heap_item = CubatureRegion(est_k, err_k, ak, bk)
        regions.append(heap_item)
        est += est_k
        err += err_k
    # scipy keeps `regions` as a plain list until the first heappop; heappop assumes a heap
    subdivisions = 0
    success = True
    while err > atol + rtol * abs(est):
        region_k = heappop(regions)
        est_k = region_k.estimate
        err_k = region_k.error
        est -= est_k
        err -= err_k
        mid = _midpoint(region_k.a, region_k.b)
        for sa, sb in _split_subregion(region_k.a, region_k.b, mid):
            e_sub = _rule_estimate(rule, f, sa, sb, args)
            r_sub = _rule_error(rule, f, sa, sb, args)
            est += e_sub
            err += r_sub
            heappush(regions, CubatureRegion(e_sub, r_sub, sa, sb))
        subdivisions += 1
        if subdivisions >= max_subdivisions:
            success = False
            break
    return CubatureResult(sign * est, err, "converged" if success else "not_converged",
                          regions, subdivisions, atol, rtol)

def _make_rule(rule, ndim: int) -> _Rule:
    name = str(rule)
    if name == "genz-malik":
        return _gm_rule(ndim)
    elif name == "gauss-kronrod" or name == "gk21":
        return _gk_rule(21, ndim)
    elif name == "gk15":
        return _gk_rule(15, ndim)
    raise ValueError(f"unknown rule {name}")

def cubature(f, a, b, *, rule = "gk21", rtol: float = 1e-8, atol: float = 0.0,
             max_subdivisions: int = 10000, args = (), workers: int = 1, points = None):
    """Adaptive cubature of a scalar-valued f over the box [a, b] (infinite
    limits supported). f(x, *args) maps nodes (n, ndim) to values (n,)."""
    aa = np.atleast_1d(np.asarray(a, dtype=float)).copy()
    bb = np.atleast_1d(np.asarray(b, dtype=float)).copy()
    if aa.shape[0] == 0 or bb.shape[0] == 0:
        raise ValueError("`a` and `b` must be nonempty")
    pts = List[np.ndarray[float, 1]]()
    if points is not None:
        for p in points:
            pts.append(np.atleast_1d(np.asarray(p, dtype=float)).copy())
    r = _make_rule(rule, aa.shape[0])
    has_inf = False
    for k in range(aa.shape[0]):
        if math.isinf(aa[k]) or math.isinf(bb[k]):
            has_inf = True
    # normalise ordering before the transform (scipy flips first)
    nflip = 0
    lo = np.minimum(aa, bb)
    hi = np.maximum(aa, bb)
    if has_inf:
        tr = _InfiniteLimitsTransform(f, lo, hi)
        ta, tb = tr.transformed_limits()
        tpts = List[np.ndarray[float, 1]]()
        for p in pts:
            tpts.append(tr.inv(p))
        for p in tr.points():
            tpts.append(p)
        res = _cubature_run(tr, r, ta, tb, rtol, atol, max_subdivisions, args, tpts)
    else:
        res = _cubature_run(_Plain(f), r, lo, hi, rtol, atol, max_subdivisions, args, pts)
    # sign from the original ordering
    n_flip = 0
    for k in range(aa.shape[0]):
        if aa[k] > bb[k]:
            n_flip += 1
    if n_flip % 2 == 1:
        res.estimate = -res.estimate
    return res
