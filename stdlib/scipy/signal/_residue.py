# Ported from scipy 1.18.1 (BSD-3-Clause)

# Partial fraction expansion: unique_roots, residue, residuez, invres, invresz.
# Deviations: poles and residues are always complex arrays (numpy returns real
# arrays when all roots are real) and the polynomial part `k` is a float
# array; real coefficients only.

import math
import numpy as np
from ._filter_design import _roots, _carr, _flist

def _trim_f(x: List[float]) -> List[float]:
    i = 0
    while i < len(x) and x[i] == 0.0:
        i += 1
    return x[i:]

def _trim_b(x: List[float]) -> List[float]:
    j = len(x)
    while j > 0 and x[j - 1] == 0.0:
        j -= 1
    return x[:j]

def _pmul(a: List[complex], b: List[complex]) -> List[complex]:
    out = [complex(0.0)] * (len(a) + len(b) - 1)
    for i in range(len(a)):
        for j in range(len(b)):
            out[i + j] = out[i + j] + a[i] * b[j]
    return out

def _padd(a: List[complex], b: List[complex]) -> List[complex]:
    n = max(len(a), len(b))
    out = [complex(0.0)] * n
    for i in range(len(a)):
        out[n - len(a) + i] = out[n - len(a) + i] + a[i]
    for i in range(len(b)):
        out[n - len(b) + i] = out[n - len(b) + i] + b[i]
    return out

def _psub(a: List[complex], b: List[complex]) -> List[complex]:
    n = max(len(a), len(b))
    out = [complex(0.0)] * n
    for i in range(len(a)):
        out[n - len(a) + i] = out[n - len(a) + i] + a[i]
    for i in range(len(b)):
        out[n - len(b) + i] = out[n - len(b) + i] - b[i]
    return out

def _pval(p: List[complex], x: complex) -> complex:
    y = complex(0.0)
    for c in p:
        y = y * x + c
    return y

def _pdiv(u: List[complex], v: List[complex]):
    # numpy.polydiv (remainder trimmed with allclose(r[0], 0, rtol=1e-14))
    m = len(u) - 1
    n = len(v) - 1
    scale = complex(1.0) / v[0]
    q = [complex(0.0)] * max(m - n + 1, 1)
    r = [c for c in u]
    for k in range(0, m - n + 1):
        d = scale * r[k]
        q[k] = d
        for j in range(n + 1):
            r[k + j] = r[k + j] - d * v[j]
    while abs(r[0]) <= 1e-8 and len(r) > 1:
        r = r[1:]
    return q, r

def _lex_less(a: complex, b: complex) -> bool:
    if a.real != b.real:
        return a.real < b.real
    return a.imag < b.imag

def _reduce(vals: List[complex], rtype: str) -> complex:
    if rtype == 'max' or rtype == 'maximum':
        m = vals[0]
        for v in vals:
            if _lex_less(m, v):
                m = v
        return m
    elif rtype == 'min' or rtype == 'minimum':
        m = vals[0]
        for v in vals:
            if _lex_less(v, m):
                m = v
        return m
    elif rtype == 'avg' or rtype == 'mean':
        s = complex(0.0)
        for v in vals:
            s = s + v
        return s / float(len(vals))
    raise ValueError("`rtype` must be one of {'max', 'maximum', 'min', 'minimum', 'avg', 'mean'}")

def _unique_roots(p: List[complex], tol: float, rtype: str):
    uniq: List[complex] = []
    mult: List[int] = []
    n = len(p)
    used = [False] * n
    for i in range(n):
        if used[i]:
            continue
        grp: List[int] = []
        for j in range(n):
            if not used[j] and abs(p[j] - p[i]) <= tol:
                grp.append(j)
        vals = [p[j] for j in grp]
        uniq.append(_reduce(vals, rtype))
        mult.append(len(grp))
        for j in grp:
            used[j] = True
    return uniq, mult

def unique_roots(p, tol: float = 1e-3, rtype: str = 'min'):
    pp = _carr(np.asarray(p))
    pl = [pp[i] for i in range(pp.size)]
    u, m = _unique_roots(pl, tol, rtype)
    return _carr(u), np.array(m)

def _cmplx_order(p: List[complex]) -> List[int]:
    idx = [i for i in range(len(p))]
    # stable sort by magnitude
    return sorted(idx, key=lambda i: abs(p[i]))

def _group_poles(poles: List[complex], tol: float, rtype: str):
    uniq: List[complex] = []
    mult: List[int] = []
    pole = poles[0]
    block = [pole]
    for i in range(1, len(poles)):
        if abs(poles[i] - pole) <= tol:
            block.append(pole)
        else:
            uniq.append(_reduce(block, rtype))
            mult.append(len(block))
            pole = poles[i]
            block = [pole]
    uniq.append(_reduce(block, rtype))
    mult.append(len(block))
    return uniq, mult

def _compute_factors(roots: List[complex], multiplicity: List[int], include_powers: bool):
    current = [complex(1.0)]
    suffixes = [current]
    for i in range(len(roots) - 1, 0, -1):
        monomial = [complex(1.0), -roots[i]]
        for _ in range(multiplicity[i]):
            current = _pmul(current, monomial)
        suffixes.append(current)
    suffixes = suffixes[::-1]
    factors: List[List[complex]] = []
    current = [complex(1.0)]
    for t in range(len(roots)):
        monomial = [complex(1.0), -roots[t]]
        block: List[List[complex]] = []
        for i in range(multiplicity[t]):
            if i == 0 or include_powers:
                block.append(_pmul(current, suffixes[t]))
            current = _pmul(current, monomial)
        for b in reversed(block):
            factors.append(b)
    return factors, current

def _compute_residues(poles: List[complex], multiplicity: List[int], numerator: List[complex]):
    factors, _ = _compute_factors(poles, multiplicity, False)
    residues: List[complex] = []
    for t in range(len(poles)):
        pole = poles[t]
        mult = multiplicity[t]
        factor = factors[t]
        if mult == 1:
            residues.append(_pval(numerator, pole) / _pval(factor, pole))
        else:
            numer = [c for c in numerator]
            monomial = [complex(1.0), -pole]
            factor, d = _pdiv(factor, monomial)
            block: List[complex] = []
            for _ in range(mult):
                numer, nn = _pdiv(numer, monomial)
                r = nn[0] / d[0]
                numer = _psub(numer, [r * c for c in factor])
                block.append(r)
            for b in reversed(block):
                residues.append(b)
    return residues

def _rl(x) -> List[complex]:
    a = np.atleast_1d(np.asarray(x, float))
    return [complex(float(a[i])) for i in range(a.size)]

def residue(b, a, tol: float = 1e-3, rtype: str = 'avg'):
    bl = _trim_f(_flist(b))
    al = _trim_f(_flist(a))
    if len(al) == 0:
        raise ValueError('Denominator `a` is zero.')
    poles = _roots(al)
    bc = [complex(v) for v in bl]
    if len(bc) == 0:
        order0 = _cmplx_order(poles)
        return np.zeros(len(poles), dtype=complex), _carr([poles[i] for i in order0]), np.zeros(0)
    k: List[complex] = []
    if len(bl) >= len(al):
        k, bc = _pdiv(bc, [complex(v) for v in al])
    uniq, mult = _unique_roots(poles, tol, rtype)
    order = _cmplx_order(uniq)
    uniq = [uniq[i] for i in order]
    mult = [mult[i] for i in order]
    res = _compute_residues(uniq, mult, bc)
    index = 0
    for t in range(len(uniq)):
        for j in range(mult[t]):
            poles[index + j] = uniq[t]
        index += mult[t]
    rr = np.empty(len(res), dtype=complex)
    for i in range(len(res)):
        rr[i] = res[i] / al[0]
    kk = np.empty(len(k))
    for i in range(len(k)):
        kk[i] = k[i].real
    return rr, _carr(poles), kk

def residuez(b, a, tol: float = 1e-3, rtype: str = 'avg'):
    bl = _trim_b(_flist(b))
    al = _trim_b(_flist(a))
    if len(al) == 0:
        raise ValueError('Denominator `a` is zero.')
    elif al[0] == 0:
        raise ValueError('First coefficient of determinant `a` must be non-zero.')
    poles = _roots(al)
    if len(bl) == 0:
        order0 = _cmplx_order(poles)
        return np.zeros(len(poles), dtype=complex), _carr([poles[i] for i in order0]), np.zeros(0)
    b_rev = [complex(v) for v in bl[::-1]]
    a_rev = [complex(v) for v in al[::-1]]
    k_rev: List[complex] = []
    if len(b_rev) >= len(a_rev):
        k_rev, b_rev = _pdiv(b_rev, a_rev)
    uniq, mult = _unique_roots(poles, tol, rtype)
    order = _cmplx_order(uniq)
    uniq = [uniq[i] for i in order]
    mult = [mult[i] for i in order]
    inv_poles = [complex(1.0) / u for u in uniq]
    res = _compute_residues(inv_poles, mult, b_rev)
    index = 0
    powers = [0] * len(res)
    for t in range(len(uniq)):
        for j in range(mult[t]):
            poles[index + j] = uniq[t]
            powers[index + j] = 1 + j
        index += mult[t]
    rr = np.empty(len(res), dtype=complex)
    for i in range(len(res)):
        f = (-poles[i]) ** powers[i]
        rr[i] = res[i] * f / a_rev[0]
    kk = np.empty(len(k_rev))
    for i in range(len(k_rev)):
        kk[len(k_rev) - 1 - i] = k_rev[i].real
    return rr, _carr(poles), kk

def invres(r, p, k, tol: float = 1e-3, rtype: str = 'avg'):
    rl = _carr(np.atleast_1d(np.asarray(r)))
    pl = _carr(np.atleast_1d(np.asarray(p)))
    kl = _trim_f(_flist(k))
    uniq, mult = _group_poles([pl[i] for i in range(pl.size)], tol, rtype)
    factors, denominator = _compute_factors(uniq, mult, True)
    numerator: List[complex] = [complex(0.0)]
    if len(kl) > 0:
        numerator = _pmul([complex(v) for v in kl], denominator)
    for i in range(min(rl.size, len(factors))):
        numerator = _padd(numerator, [rl[i] * c for c in factors[i]])
    return _carr(numerator), _carr(denominator)

def invresz(r, p, k, tol: float = 1e-3, rtype: str = 'avg'):
    rl = _carr(np.atleast_1d(np.asarray(r)))
    pl = _carr(np.atleast_1d(np.asarray(p)))
    kl = _trim_b(_flist(k))
    uniq, mult = _group_poles([pl[i] for i in range(pl.size)], tol, rtype)
    factors, denominator = _compute_factors(uniq, mult, True)
    numerator: List[complex] = [complex(0.0)]
    if len(kl) > 0:
        numerator = _pmul([complex(v) for v in kl[::-1]], denominator[::-1])
    for i in range(min(rl.size, len(factors))):
        numerator = _padd(numerator, [rl[i] * c for c in factors[i][::-1]])
    return _carr(numerator[::-1]), _carr(denominator)
