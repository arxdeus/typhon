# Ported from scipy 1.18.1 (BSD-3-Clause)

# Labelling and region measurements (scipy.ndimage._measurements).
# label() numbers regions in raster order of their first pixel, which is
# the numbering produced by scipy's merge-table labeller.

import numpy as np
import internal.static as static
from math import sqrt
from ._ni_support import _shape, _strides, _prod, _next_coord, _fbuf
from ._filters import _bool_buf

def _uf(parent: List[int], x: int) -> int:
    r = x
    while parent[r] != r:
        r = parent[r]
    while parent[x] != r:
        nx = parent[x]
        parent[x] = r
        x = nx
    return r

def _label_buf(I: List[bool], sh: List[int], ssh: List[int], sm: List[bool]) -> Tuple[List[int], int]:
    nd = len(sh)
    n = len(I)
    st = _strides(sh)
    # neighbour offsets that precede the centre in C order
    rel = List[List[int]]()
    idx = [0] * nd
    half = len(sm) // 2
    for k in range(half):
        if sm[k]:
            rel.append([idx[d] - 1 for d in range(nd)])
        _next_coord(idx, ssh)
    parent = list(range(n))
    coord = [0] * nd
    for p in range(n):
        if I[p]:
            for r in rel:
                ok = True
                q = 0
                for d in range(nd):
                    c = coord[d] + r[d]
                    if c < 0 or c >= sh[d]:
                        ok = False
                        break
                    q += c * st[d]
                if ok and I[q]:
                    a = _uf(parent, p)
                    b = _uf(parent, q)
                    if a != b:
                        if a < b:
                            parent[b] = a
                        else:
                            parent[a] = b
        _next_coord(coord, sh)
    lab = [0] * n
    rootlab = Dict[int, int]()
    nl = 0
    for p in range(n):
        if I[p]:
            r = _uf(parent, p)
            if r in rootlab:
                lab[p] = rootlab[r]
            else:
                nl += 1
                rootlab[r] = nl
                lab[p] = nl
    return (lab, nl)

def label(input, structure = None, output = None):
    a = np.asarray(input)
    sh = _shape(a)
    nd = len(sh)
    ssh = List[int]()
    sm = List[bool]()
    if structure is None:
        ssh = [3] * nd
        sm = List[bool]()
        idx = [0] * nd
        for k in range(_prod(ssh)):
            s = 0
            for d in range(nd):
                s += abs(idx[d] - 1)
            sm.append(s <= 1)
            _next_coord(idx, ssh)
    else:
        S = np.asarray(structure)
        ssh = _shape(S)
        sm = _bool_buf(S)
        if len(ssh) != nd:
            raise RuntimeError('structure and input must have equal rank')
        for s in ssh:
            if s != 3:
                raise ValueError('structure dimensions must be equal to 3')
    r = _label_buf(_bool_buf(a), sh, ssh, sm)
    if output is None:
        out = np.empty(a.shape, dtype=np.int32)
        for p in range(len(r[0])):
            out.data[p] = np.int32(r[0][p])
        return (out, r[1])
    elif isinstance(output, type):
        out = np.empty(a.shape, dtype=output)
        for p in range(len(r[0])):
            out.data[p] = output(r[0][p])
        return (out, r[1])
    else:
        for p in range(len(r[0])):
            output.data[p] = type(output.data[0])(r[0][p])
        return r[1]

def find_objects(input, max_label: int = 0):
    a = np.asarray(input)
    sh = _shape(a)
    nd = len(sh)
    A = np.ascontiguousarray(a)
    n = A.size
    ml = max_label
    if ml < 1:
        for p in range(n):
            if int(A.data[p]) > ml:
                ml = int(A.data[p])
    reg = [-1] * (2 * nd * ml)
    coord = [0] * nd
    for p in range(n):
        s = int(A.data[p]) - 1
        if s >= 0 and s < ml:
            b = s * 2 * nd
            if reg[b] < 0:
                for k in range(nd):
                    reg[b + k] = coord[k]
                    reg[b + k + nd] = coord[k] + 1
            else:
                for k in range(nd):
                    if coord[k] < reg[b + k]:
                        reg[b + k] = coord[k]
                    if coord[k] + 1 > reg[b + k + nd]:
                        reg[b + k + nd] = coord[k] + 1
        _next_coord(coord, sh)
    out = List[Optional[type(_slices_t(a.shape))]]()
    for i in range(ml):
        b = i * 2 * nd
        if reg[b] >= 0:
            out.append(_mk_slices(reg, b, nd, a.shape))
        else:
            out.append(None)
    return out

def _slices_t(shape):
    return tuple(slice(0, 0) for _ in shape)

def _mk_slices(reg: List[int], b: int, nd: int, shape):
    t = _slices_t(shape)
    p = Ptr[type(slice(0, 0))](__ptr__(t).as_byte())
    for k in range(nd):
        p[k] = slice(reg[b + k], reg[b + k + nd])
    return t

# ---- per-label statistics -------------------------------------------------------

def _groups(input, labels, index) -> Tuple[List[float], List[List[int]]]:
    # returns flat values and, per requested group, the flat positions
    a = np.asarray(input)
    vals = _fbuf(a)
    n = len(vals)
    groups = List[List[int]]()
    if labels is None:
        groups.append(list(range(n)))
        return (vals, groups)
    else:
        La = np.ascontiguousarray(np.broadcast_to(np.asarray(labels), a.shape))
        lab = [int(La.data[i]) for i in range(n)]
        if index is None:
            g = List[int]()
            for i in range(n):
                if lab[i] > 0:
                    g.append(i)
            groups.append(g)
        elif isinstance(index, int):
            g = List[int]()
            for i in range(n):
                if lab[i] == index:
                    g.append(i)
            groups.append(g)
        else:
            idx = [int(v) for v in np.atleast_1d(np.asarray(index)).flat]
            pos = Dict[int, int]()
            for k in range(len(idx)):
                groups.append(List[int]())
                if idx[k] not in pos:
                    pos[idx[k]] = k
            for i in range(n):
                if lab[i] in pos:
                    groups[pos[lab[i]]].append(i)
            for k in range(len(idx)):
                if pos[idx[k]] != k:
                    groups[k] = groups[pos[idx[k]]]
        return (vals, groups)

def _vec_index(labels, index) -> Literal[bool]:
    # whether per-label results are returned as a sequence (decided statically)
    if isinstance(labels, NoneType):
        return False
    elif isinstance(index, NoneType):
        return False
    elif isinstance(index, int):
        return False
    else:
        return True

def _fout(x: List[float], labels, index):
    if not _vec_index(labels, index):
        return x[0]
    else:
        return np.array(x)

def _vtype(input):
    return np.asarray(input).dtype

def sum_labels(input, labels = None, index = None):
    g = _groups(input, labels, index)
    out = List[float]()
    for grp in g[1]:
        s = 0.0
        for i in grp:
            s += g[0][i]
        out.append(s)
    return _fout(out, labels, index)

def sum(input, labels = None, index = None):
    return sum_labels(input, labels, index)

def mean(input, labels = None, index = None):
    g = _groups(input, labels, index)
    out = List[float]()
    for grp in g[1]:
        s = 0.0
        for i in grp:
            s += g[0][i]
        out.append(s / float(len(grp)) if len(grp) > 0 else float('nan'))
    return _fout(out, labels, index)

def variance(input, labels = None, index = None):
    g = _groups(input, labels, index)
    out = List[float]()
    for grp in g[1]:
        if len(grp) == 0:
            out.append(float('nan'))
            continue
        s = 0.0
        for i in grp:
            s += g[0][i]
        m = s / len(grp)
        c = 0.0
        for i in grp:
            d = g[0][i] - m
            c += d * d
        out.append(c / float(len(grp)))
    return _fout(out, labels, index)

def standard_deviation(input, labels = None, index = None):
    v = variance(input, labels, index)
    return np.sqrt(v)

def _sel(input, labels, index, mx: bool):
    g = _groups(input, labels, index)
    # scipy's sort-and-scatter path (sequence index) reports the last tied
    # position of a maximum; the single-group path reports the first one
    last_tie = mx and _vec_index(labels, index)
    out = List[float]()
    pos = List[int]()
    for grp in g[1]:
        if len(grp) == 0:
            out.append(0.0)
            pos.append(0)
            continue
        b = g[0][grp[0]]
        bp = grp[0]
        for i in grp:
            v = g[0][i]
            if (mx and (v > b or (last_tie and v == b))) or ((not mx) and v < b):
                b = v
                bp = i
        out.append(b)
        pos.append(bp)
    return (out, pos)

def _typed_out(x: List[float], input, labels, index):
    T = type(np.asarray(input).dtype())
    if not _vec_index(labels, index):
        return T(x[0]) if T is not bool else x[0] != 0.0
    else:
        arr = np.empty(len(x), dtype=T)
        for i in range(len(x)):
            arr[i] = T(x[i])
        return arr

def minimum(input, labels = None, index = None):
    return _typed_out(_sel(input, labels, index, False)[0], input, labels, index)

def maximum(input, labels = None, index = None):
    return _typed_out(_sel(input, labels, index, True)[0], input, labels, index)

def median(input, labels = None, index = None):
    g = _groups(input, labels, index)
    out = List[float]()
    for grp in g[1]:
        v = sorted([g[0][i] for i in grp])
        m = len(v)
        if m == 0:
            out.append(0.0)
        else:
            out.append((v[(m - 1) // 2] + v[m // 2]) / 2.0)
    return _fout(out, labels, index)

def _unravel(p: int, sh: List[int], like):
    st = _strides(sh)
    t = like
    q = Ptr[int](__ptr__(t).as_byte())
    for d in range(len(sh)):
        q[d] = p // st[d] % sh[d]
    return t

def _pos_out(pos: List[int], input, labels, index):
    a = np.asarray(input)
    sh = _shape(a)
    if not _vec_index(labels, index):
        return _unravel(pos[0], sh, a.shape)
    else:
        return [_unravel(p, sh, a.shape) for p in pos]

def minimum_position(input, labels = None, index = None):
    return _pos_out(_sel(input, labels, index, False)[1], input, labels, index)

def maximum_position(input, labels = None, index = None):
    return _pos_out(_sel(input, labels, index, True)[1], input, labels, index)

def extrema(input, labels = None, index = None):
    mn = _sel(input, labels, index, False)
    mx = _sel(input, labels, index, True)
    return (_typed_out(mn[0], input, labels, index), _typed_out(mx[0], input, labels, index),
            _pos_out(mn[1], input, labels, index), _pos_out(mx[1], input, labels, index))

def center_of_mass(input, labels = None, index = None):
    a = np.asarray(input)
    sh = _shape(a)
    nd = len(sh)
    st = _strides(sh)
    g = _groups(a, labels, index)
    res = List[List[float]]()
    for grp in g[1]:
        tot = 0.0
        for i in grp:
            tot += g[0][i]
        c = [0.0] * nd
        for d in range(nd):
            s = 0.0
            for i in grp:
                s += g[0][i] * float(i // st[d] % sh[d])
            c[d] = s / tot
        res.append(c)
    if not _vec_index(labels, index):
        return _ftuple(res[0], a.shape)
    else:
        return [_ftuple(r, a.shape) for r in res]

def _ftuple(c: List[float], like):
    t = tuple(0.0 for _ in like)
    q = Ptr[float](__ptr__(t).as_byte())
    for d in range(len(c)):
        q[d] = c[d]
    return t

def histogram(input, min: float, max: float, bins: int, labels = None, index = None):
    g = _groups(input, labels, index)
    edges = np.linspace(min, max, bins + 1)
    res = List[type(np.zeros(1, dtype=int))]()
    for grp in g[1]:
        v = np.array([g[0][i] for i in grp]) if len(grp) > 0 else np.zeros(0)
        res.append(np.histogram(v, edges)[0])
    if not _vec_index(labels, index):
        return res[0]
    else:
        return res

def labeled_comprehension(input, labels, index, func, out_dtype: type, default,
                          pass_positions: Literal[bool] = False):
    g = _groups(input, labels, index)
    out = List[out_dtype]()
    for grp in g[1]:
        if len(grp) == 0:
            out.append(out_dtype(default))
            continue
        v = np.array([g[0][i] for i in grp])
        if pass_positions:
            pp = np.array(grp)
            out.append(out_dtype(func(v, pp)))
        else:
            out.append(out_dtype(func(v)))
    if not _vec_index(labels, index):
        return out[0]
    else:
        arr = np.empty(len(out), dtype=out_dtype)
        for i in range(len(out)):
            arr[i] = out[i]
        return arr

# --- value_indices ---------------------------------------------------------

def value_indices(arr, *, ignore_value = None):
    """Dict mapping each distinct value to the flat-order index arrays (one
    per axis, as a tuple list) of its occurrences."""
    a = np.ascontiguousarray(np.asarray(arr))
    sh = _shape(a)
    nd = len(sh)
    groups = Dict[type(a.data[0]), List[int]]()
    order = List[type(a.data[0])]()
    for p in range(a.size):
        v = a.data[p]
        if ignore_value is not None and v == type(v)(ignore_value):
            continue
        if v not in groups:
            groups[v] = List[int]()
            order.append(v)
        groups[v].append(p)
    out = Dict[type(a.data[0]), List[np.ndarray[int, 1]]]()
    st = _strides(sh)
    for v in order:
        flat = groups[v]
        cols = List[np.ndarray[int, 1]]()
        for k in range(nd):
            c = np.empty(len(flat), dtype=int)
            for i in range(len(flat)):
                c[i] = (flat[i] // st[k]) % sh[k]
            cols.append(c)
        out[v] = cols
    return out

# --- watershed_ift (NI_WatershedIFT) ---------------------------------------

def watershed_ift(input, markers, structure = None, output = None):
    a = np.ascontiguousarray(np.asarray(input))
    if a.dtype is not u8 and a.dtype is not u16:
        raise TypeError('only 8 and 16 unsigned inputs are supported')
    sh = _shape(a)
    nd = len(sh)
    if nd > 7:
        raise RuntimeError('too many dimensions')
    if structure is None:
        from ._morphology import _bstruct
        ps = _bstruct(nd, 1)[1]
    else:
        S = np.asarray(structure, dtype=bool)
        if len(S.shape) != nd:
            raise RuntimeError('structure and input must have equal rank')
        for d in S.shape:
            if d != 3:
                raise RuntimeError('structure dimensions must be equal to 3')
        ps = _bool_buf(S)
    M = np.ascontiguousarray(np.asarray(markers))
    if M.shape != a.shape:
        raise RuntimeError('input and markers must have equal shape')
    size = a.size
    maxval = 0
    for j in range(size):
        v = int(a.data[j])
        if v > maxval:
            maxval = v
    cost = [0] * size
    nxt = [-1] * size
    prv = [-1] * size
    done = [False] * size
    first = [-1] * (maxval + 1)
    last = [-1] * (maxval + 1)
    lab = [0] * size
    for j in range(size):
        label = int(M.data[j])
        lab[j] = label
        if label != 0:
            cost[j] = 0
            if first[0] < 0:
                first[0] = j
                nxt[j] = -1
                prv[j] = -1
                last[0] = j
            else:
                if label > 0:
                    nxt[j] = first[0]
                    prv[j] = -1
                    prv[first[0]] = j
                    first[0] = j
                else:
                    nxt[j] = -1
                    prv[j] = last[0]
                    nxt[last[0]] = j
                    last[0] = j
        else:
            cost[j] = maxval + 1
    st = _strides(sh)
    ssize = len(ps)
    nstr = List[int]()
    coords = [-1] * nd
    for kk in range(ssize):
        if ps[kk] and kk != ssize // 2:
            off = 0
            for l in range(nd):
                off += coords[l] * st[l]
            if off != 0:
                nstr.append(off)
        l = nd - 1
        while l >= 0:
            if coords[l] < 1:
                coords[l] += 1
                break
            else:
                coords[l] = -1
            l -= 1
    nneigh = len(nstr)
    for jj in range(maxval + 1):
        while first[jj] >= 0:
            v = first[jj]
            first[jj] = nxt[v]
            if first[jj] >= 0:
                prv[first[jj]] = -1
            prv[v] = -1
            nxt[v] = -1
            done[v] = True
            for hh in range(nneigh):
                p = v + nstr[hh]
                if p < 0 or p >= size:
                    continue
                if done[p]:
                    continue
                wvp = int(a.data[p]) - int(a.data[v])
                if wvp < 0:
                    wvp = -wvp
                pcost = cost[p]
                mx = cost[v] if cost[v] > wvp else wvp
                if mx < pcost:
                    cost[p] = mx
                    label = lab[v]
                    lab[p] = label
                    if nxt[p] >= 0 or prv[p] >= 0:
                        pv = prv[p]
                        nx = nxt[p]
                        if first[pcost] == p:
                            first[pcost] = nx
                        if last[pcost] == p:
                            last[pcost] = pv
                        if pv >= 0:
                            nxt[pv] = nx
                        if nx >= 0:
                            prv[nx] = pv
                    if label < 0:
                        prv[p] = last[mx]
                        nxt[p] = -1
                        if last[mx] >= 0:
                            nxt[last[mx]] = p
                        last[mx] = p
                        if first[mx] < 0:
                            first[mx] = p
                    else:
                        nxt[p] = first[mx]
                        prv[p] = -1
                        if first[mx] >= 0:
                            prv[first[mx]] = p
                        first[mx] = p
                        if last[mx] < 0:
                            last[mx] = p
    out = np.empty(M.shape, dtype=M.dtype)
    for j in range(size):
        out.data[j] = type(out.data[0])(lab[j])
    return out
