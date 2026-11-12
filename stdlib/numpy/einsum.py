# np.einsum: general Einstein summation.
#
# The subscripts string must be a string literal so that the output rank is
# known at compile time. Supports explicit and implicit output, repeated
# indices (traces/diagonals), '...' broadcasting and any number of operands.
# Two-operand contractions without repeated indices are dispatched to matmul
# (BLAS for float/complex types).

import util
import internal.static as static

from .ndarray import ndarray
from .routines import asarray, empty, zeros, ascontiguousarray
from .linalg_sym import matmul

# ---- compile-time parsing (output rank) ----
# All helpers return tuples whose static length encodes the answer, and only
# use nested static conditions so the compiler can fold every branch.

def _st_letter(c: Literal[str]):
    # () if c is not a subscript letter, (0,) otherwise
    if c == ",":
        return ()
    else:
        if c == ".":
            return ()
        else:
            if c == " ":
                return ()
            else:
                if c == "-":
                    return ()
                else:
                    if c == ">":
                        return ()
                    else:
                        return (0,)

def _st_arrow(s: Literal[str], i: Literal[int]):
    # tuple of length = index of '-' (or len(s) if absent)
    if i == static.len(s):
        return (0,) * i
    else:
        if s[i] == "-":
            return (0,) * i
        else:
            return _st_arrow(s, i + 1)

def _st_term(s: Literal[str], i: Literal[int], term: Literal[int], k: Literal[int]):
    # substring of the k-th comma separated term (returned as a static tuple of letters count)
    if i == static.len(s):
        return ()
    else:
        if s[i] == ",":
            return _st_term(s, i + 1, term + 1, k)
        else:
            if term == k:
                return _st_letter(s[i]) + _st_term(s, i + 1, term, k)
            else:
                return _st_term(s, i + 1, term, k)

def _st_term_dots(s: Literal[str], i: Literal[int], term: Literal[int], k: Literal[int]):
    if i == static.len(s):
        return ()
    else:
        if s[i] == ",":
            return _st_term_dots(s, i + 1, term + 1, k)
        else:
            if term == k:
                if s[i] == ".":
                    return (0,)
                else:
                    return _st_term_dots(s, i + 1, term, k)
            else:
                return _st_term_dots(s, i + 1, term, k)

def _st_maxell(lhs: Literal[str], ops, k: Literal[int], acc):
    if k == static.len(ops):
        return acc
    else:
        if static.len(_st_term_dots(lhs, 0, 0, k)) == 1:
            n: Literal[int] = ops[k].ndim - static.len(_st_term(lhs, 0, 0, k))
            if n > static.len(acc):
                return _st_maxell(lhs, ops, k + 1, (0,) * n)
            else:
                return _st_maxell(lhs, ops, k + 1, acc)
        else:
            return _st_maxell(lhs, ops, k + 1, acc)

def _st_cnt(s: Literal[str], c: Literal[str], i: Literal[int]):
    if i == static.len(s):
        return ()
    else:
        if s[i] == c:
            return (0,) + _st_cnt(s, c, i + 1)
        else:
            return _st_cnt(s, c, i + 1)

def _st_once(s: Literal[str], i: Literal[int]):
    if i == static.len(s):
        return ()
    else:
        if static.len(_st_letter(s[i])) == 1:
            if static.len(_st_cnt(s, s[i], 0)) == 1:
                return (0,) + _st_once(s, i + 1)
            else:
                return _st_once(s, i + 1)
        else:
            return _st_once(s, i + 1)

def _st_letters(s: Literal[str], i: Literal[int]):
    if i == static.len(s):
        return ()
    else:
        return _st_letter(s[i]) + _st_letters(s, i + 1)

def _st_has_dot(s: Literal[str], i: Literal[int]):
    if i == static.len(s):
        return ()
    else:
        if s[i] == ".":
            return (0,)
        else:
            return _st_has_dot(s, i + 1)

def _st_outnd(subscripts: Literal[str], ops):
    a: Literal[int] = static.len(_st_arrow(subscripts, 0))
    lhs: Literal[str] = subscripts[:a]
    ell = _st_maxell(lhs, ops, 0, ())
    if a == static.len(subscripts):
        return ell + _st_once(lhs, 0)
    else:
        rhs: Literal[str] = subscripts[a + 2:]
        if static.len(_st_has_dot(rhs, 0)) == 1:
            return ell + _st_letters(rhs, 0)
        else:
            return _st_letters(rhs, 0)

# ---- runtime parsing ----

_ELL_BASE: Literal[int] = 100000

def _parse_term(t: str) -> List[int]:
    res = List[int]()
    i = 0
    n = len(t)
    while i < n:
        c = t[i]
        if c == ".":
            if i + 2 >= n or t[i + 1] != "." or t[i + 2] != ".":
                raise ValueError("einsum(): invalid subscript '.' is not part of an ellipsis")
            res.append(-1)
            i += 3
            continue
        if c == " ":
            i += 1
            continue
        if not c.isalpha():
            raise ValueError(f"einsum(): invalid subscript '{c}' in einstein sum subscripts string")
        res.append(ord(c))
        i += 1
    return res

def _expand(term: List[int], ndim: int, maxell: int) -> List[int]:
    res = List[int]()
    nlet = 0
    for x in term:
        if x >= 0:
            nlet += 1
    for x in term:
        if x == -1:
            nell = ndim - nlet
            if nell < 0:
                raise ValueError("einsum(): operand has more dimensions than subscripts given")
            for t in range(nell):
                res.append(_ELL_BASE + maxell - nell + t)
        else:
            res.append(x)
    if len(res) != ndim:
        raise ValueError(f"einsum(): operand has {ndim} dimensions but {len(res)} subscripts were given")
    return res

def _parse(subscripts: str, ndims: List[int]):
    s = subscripts
    if "->" in s:
        k = s.index("->")
        lhs = s[:k]
        rhs = s[k + 2:]
        explicit = True
    else:
        lhs = s
        rhs = ""
        explicit = False
    terms = lhs.split(",")
    if len(terms) != len(ndims):
        raise ValueError("einsum(): more operands were provided than specified in the subscripts string"
                         if len(ndims) > len(terms) else
                         "einsum(): fewer operands provided than specified in the subscripts string")
    raw = [_parse_term(t) for t in terms]
    maxell = 0
    for k in range(len(raw)):
        if -1 in raw[k]:
            nlet = 0
            for x in raw[k]:
                if x >= 0:
                    nlet += 1
            maxell = max(maxell, ndims[k] - nlet)
    ins = [_expand(raw[k], ndims[k], maxell) for k in range(len(raw))]
    out = List[int]()
    if explicit:
        r = _parse_term(rhs)
        for x in r:
            if x == -1:
                for t in range(maxell):
                    out.append(_ELL_BASE + t)
            else:
                if x in out:
                    raise ValueError(f"einsum(): output subscript '{chr(x)}' appears more than once")
                found = False
                for term in ins:
                    if x in term:
                        found = True
                if not found:
                    raise ValueError(f"einsum(): output subscript '{chr(x)}' does not appear in the input")
                out.append(x)
    else:
        for t in range(maxell):
            out.append(_ELL_BASE + t)
        counts = Dict[int, int]()
        for term in ins:
            for x in term:
                if x < _ELL_BASE:
                    counts[x] = counts.get(x, 0) + 1
        letters = sorted([x for x, c in counts.items() if c == 1])
        out.extend(letters)
    return ins, out

# ---- evaluation ----

def _result_dtype(ops, k: Literal[int], acc: type):
    if k == static.len(ops):
        return acc()
    else:
        x = util.coerce(acc, ops[k].dtype)
        return _result_dtype(ops, k + 1, type(x))

def _generic(ops, ins: List[List[int]], out: List[int], R: type, outnd: Literal[int]):
    nops: Literal[int] = static.len(ops)
    # label sizes
    sizes = Dict[int, int]()
    for k in static.range(nops):
        shape = ops[k].shape
        term = ins[k]
        for d in range(len(term)):
            lab = term[d]
            n = shape[d]
            if lab in sizes:
                m = sizes[lab]
                if m != n:
                    if m == 1:
                        sizes[lab] = n
                    elif n != 1:
                        raise ValueError(f"einsum(): operands could not be broadcast together "
                                         f"(dimension of size {n} vs {m})")
            else:
                sizes[lab] = n
    labels = List[int](out)
    sumlabs = sorted([lab for lab in sizes.keys() if lab not in out])
    labels.extend(sumlabs)
    nl = len(labels)
    no = len(out)
    # per-operand byte strides per label
    strides = Ptr[int](nops * max(nl, 1))
    for i in range(nops * max(nl, 1)):
        strides[i] = 0
    for k in static.range(nops):
        shape = ops[k].shape
        st = ops[k].strides
        term = ins[k]
        for d in range(len(term)):
            if shape[d] != 1:
                j = labels.index(term[d])
                strides[k * nl + j] += st[d]
    lsize = Ptr[int](max(nl, 1))
    for j in range(nl):
        lsize[j] = sizes[labels[j]]

    oshape = (0,) * outnd
    poshape = Ptr[int](__ptr__(oshape).as_byte())
    for j in static.range(outnd):
        poshape[j] = lsize[j]
    result = zeros(oshape, dtype=R)

    total_out = 1
    for j in range(no):
        total_out *= lsize[j]
    total_sum = 1
    for j in range(no, nl):
        total_sum *= lsize[j]
    if total_out == 0:
        return result

    bases = Ptr[int](nops)
    offs = Ptr[int](nops)
    for k in range(nops):
        bases[k] = 0
    oidx = Ptr[int](max(nl, 1))
    for j in range(nl):
        oidx[j] = 0
    rdata = result.data

    for o in range(total_out):
        acc = util.zero(R)
        for k in range(nops):
            offs[k] = bases[k]
        if total_sum > 0:
            for j in range(no, nl):
                oidx[j] = 0
            for t in range(total_sum):
                prod = util.cast(Ptr[ops[0].dtype](ops[0].data.as_byte() + offs[0])[0], R)
                for k in static.range(1, nops):
                    prod = prod * util.cast(Ptr[ops[k].dtype](ops[k].data.as_byte() + offs[k])[0], R)
                acc += prod
                # advance summation odometer
                j = nl - 1
                while j >= no:
                    oidx[j] += 1
                    for k in range(nops):
                        offs[k] += strides[k * nl + j]
                    if oidx[j] < lsize[j]:
                        break
                    for k in range(nops):
                        offs[k] -= strides[k * nl + j] * lsize[j]
                    oidx[j] = 0
                    j -= 1
        rdata[o] = acc
        # advance output odometer
        j = no - 1
        while j >= 0:
            oidx[j] += 1
            for k in range(nops):
                bases[k] += strides[k * nl + j]
            if oidx[j] < lsize[j]:
                break
            for k in range(nops):
                bases[k] -= strides[k * nl + j] * lsize[j]
            oidx[j] = 0
            j -= 1
    return result

def _same_sizes(sa, sb, ins: List[List[int]]) -> bool:
    # matmul path cannot broadcast size-1 dimensions
    a = ins[0]
    b = ins[1]
    for i in range(len(a)):
        if a[i] in b:
            if sa[i] != sb[b.index(a[i])]:
                return False
    return True

def _can_matmul(ins: List[List[int]], out: List[int]) -> bool:
    a = ins[0]
    b = ins[1]
    for term in ins:
        for i in range(len(term)):
            for j in range(i + 1, len(term)):
                if term[i] == term[j]:
                    return False
    for x in a:
        if x not in out and x not in b:
            return False
    for x in b:
        if x not in out and x not in a:
            return False
    return True

def _matmul_path(A, B, ins: List[List[int]], out: List[int], R: type, outnd: Literal[int]):
    a = ins[0]
    b = ins[1]
    ash = A.shape
    bsh = B.shape
    batch = [x for x in out if x in a and x in b]
    akeep = [x for x in out if x in a and x not in b]
    bkeep = [x for x in out if x in b and x not in a]
    contr = [x for x in a if x not in out]
    for x in contr + batch:
        if ash[a.index(x)] != bsh[b.index(x)]:
            raise ValueError("einsum(): operands could not be broadcast together")
    Bt = 1
    M = 1
    N = 1
    K = 1
    for x in batch:
        Bt *= ash[a.index(x)]
    for x in akeep:
        M *= ash[a.index(x)]
    for x in bkeep:
        N *= bsh[b.index(x)]
    for x in contr:
        K *= ash[a.index(x)]
    perm_a = [a.index(x) for x in batch + akeep + contr]
    perm_b = [b.index(x) for x in batch + contr + bkeep]
    At = ascontiguousarray(A.transpose(perm_a), dtype=R).reshape(Bt, M, K)
    Bm = ascontiguousarray(B.transpose(perm_b), dtype=R).reshape(Bt, K, N)
    res = matmul(At, Bm)
    order = batch + akeep + bkeep
    shp = (0,) * outnd
    pshp = Ptr[int](__ptr__(shp).as_byte())
    for i in range(len(order)):
        x = order[i]
        pshp[i] = ash[a.index(x)] if x in a else bsh[b.index(x)]
    r = res.reshape(shp)
    perm = [order.index(x) for x in out]
    return ascontiguousarray(r.transpose(perm))

def einsum(subscripts: Literal[str], *operands, out = None, dtype: type = NoneType,
           order: str = 'K', casting: str = 'safe', optimize = False):
    ops = tuple(asarray(op) for op in operands)
    if static.len(ops) == 0:
        compile_error("einsum(): must specify at least one operand")
    outnd: Literal[int] = static.len(_st_outnd(subscripts, ops))
    ndims = [op.ndim for op in ops]
    ins, outl = _parse(subscripts, ndims)
    if len(outl) != outnd:
        raise ValueError("einsum(): internal error determining output rank")
    if dtype is NoneType:
        R = type(_result_dtype(ops, 1, ops[0].dtype))
    else:
        R = dtype

    use_mm = False
    if static.len(ops) == 2:
        if R is float or R is float32 or R is complex or R is complex64:
            use_mm = _can_matmul(ins, outl) and _same_sizes(ops[0].shape, ops[1].shape, ins)
    if static.len(ops) == 2:
        if use_mm:
            res = _matmul_path(ops[0], ops[1], ins, outl, R, outnd)
        else:
            res = _generic(ops, ins, outl, R, outnd)
    else:
        res = _generic(ops, ins, outl, R, outnd)

    if out is not None:
        out[...] = res
        return out
    else:
        if outnd == 0:
            return res.item()
        else:
            return res

def einsum_path(subscripts: str, *operands, optimize = 'greedy', einsum_call: bool = False):
    # Simplified: always returns the trivial left-to-right contraction path.
    n = len(operands)
    path = ['einsum_path']
    for i in range(n - 1):
        path.append('(0, 1)')
    return path, "Complete contraction:  " + subscripts + "\n"
