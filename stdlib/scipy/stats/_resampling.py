# Copyright (C) 2026 the Typhon authors
# Ported from scipy 1.18.1 (BSD-3-Clause)

# Resampling methods: bootstrap (percentile / basic / BCa, paired),
# permutation_test (independent / samples / pairings, exact and randomized),
# monte_carlo_test.  ``data`` is a tuple of samples; ``statistic`` may accept an
# ``axis`` keyword (vectorized) or take 1-D samples.  ``rng`` is an int seed,
# None or a numpy Generator (as in scipy 1.18, normalized by default_rng).

import numpy as np
import internal.static as static
import scipy.special._kernels as K
from ._common import _f64

NAN = K.NAN
INF = K.INF

# ------------------------------------------------------------------ results

class _CI[L, H]:
    low: L
    high: H
    def __init__(self, low: L, high: H):
        self.low = low
        self.high = high
    def __getitem__(self, i: Literal[int]):
        if i == 0 or i == -2:
            return self.low
        elif i == 1 or i == -1:
            return self.high
        else:
            compile_error("index out of range")
    def __len__(self):
        return 2
    def __iter__(self):
        yield self.low
        yield self.high

class BootstrapResult[C, D, S]:
    confidence_interval: C
    bootstrap_distribution: D
    standard_error: S
    def __init__(self, confidence_interval: C, bootstrap_distribution: D, standard_error: S):
        self.confidence_interval = confidence_interval
        self.bootstrap_distribution = bootstrap_distribution
        self.standard_error = standard_error

class MonteCarloTestResult[S, P, N]:
    statistic: S
    pvalue: P
    null_distribution: N
    def __init__(self, statistic: S, pvalue: P, null_distribution: N):
        self.statistic = statistic
        self.pvalue = pvalue
        self.null_distribution = null_distribution
    def __getitem__(self, i: Literal[int]):
        if i == 0 or i == -3:
            return self.statistic
        elif i == 1 or i == -2:
            return self.pvalue
        elif i == 2 or i == -1:
            return self.null_distribution
        else:
            compile_error("index out of range")
    def __len__(self):
        return 3
    def __iter__(self):
        yield self.statistic
        yield self.pvalue
        yield self.null_distribution

class PermutationTestResult[S, P, N](MonteCarloTestResult[S, P, N]):
    def __init__(self, statistic: S, pvalue: P, null_distribution: N):
        super().__init__(statistic, pvalue, null_distribution)

# ------------------------------------------------------------------ helpers

def _tmap(f, d):
    if static.len(d) == 0:
        return ()
    else:
        return (f(d[0]),) + _tmap(f, d[1:])

def _vec(vectorized) -> bool:
    if isinstance(vectorized, bool):
        return vectorized
    else:
        return True

def _asf(x):
    return np.asarray(x).astype(float)

class _MoveLast:
    axis: int
    def __call__(self, a):
        return np.moveaxis(_f64(a), self.axis, -1)

class _LeadShape:
    def __call__(self, a):
        return a.shape[:-1]

class _Bc[T]:
    lead: T
    def __call__(self, a):
        return np.broadcast_to(a, self.lead + (a.shape[-1],))

def _bc_all(data):
    shapes = _tmap(_LeadShape(), data)
    return _tmap(_Bc(np.broadcast_shapes(*shapes)), data)

def _row(a, idx):
    if static.len(idx) == 0:
        return a
    else:
        return np.ascontiguousarray(a[idx])

def _rows(data, idx):
    if static.len(data) == 0:
        return ()
    else:
        return (_row(data[0], idx),) + _rows(data[1:], idx)

def _loop_eval(f, data):
    shp = data[0].shape[:-1]
    if static.len(shp) == 0:
        return np.asarray(float(f(*data)))
    else:
        out = np.empty(shp, dtype=float)
        for idx in np.ndindex(shp):
            out[idx] = float(f(*_rows(data, idx)))
        return out

def _eval(f, vec: bool, data):
    if static.function.can_call(f, *data, axis=-1):
        if vec:
            return _asf(f(*data, axis=-1))
        else:
            return _loop_eval(f, _bc_all(data))
    else:
        return _loop_eval(f, _bc_all(data))

class _Eval[F]:
    f: F
    vec: bool
    def __call__(self, data):
        return _eval(self.f, self.vec, data)

class _Gather[I]:
    idx: I
    def __call__(self, a):
        return a[..., self.idx]

class _PairedEval[E, D]:
    ev: E
    data: D
    def __call__(self, idxs):
        return self.ev(_tmap(_Gather(idxs[0]), self.data))

class _Resampler[G]:
    rng: G
    R: int
    def __call__(self, a):
        n = a.shape[-1]
        idx = self.rng.integers(0, n, size=(self.R, n))
        return a[..., idx]

def _nres(n) -> int:
    if isinstance(n, float):
        if n == INF:
            return 4611686018427387904
        return int(n)
    else:
        return int(n)

def _unwrap(x):
    if x.ndim == 0:
        return float(x)
    else:
        return x

def _prep(data, axis: int):
    if static.len(data) == 0:
        raise ValueError("`data` must contain at least one sample.")
    return _bc_all(_tmap(_MoveLast(axis), tuple(data)))

class _CheckLen:
    def __call__(self, a):
        if a.shape[-1] <= 1:
            raise ValueError("each sample in `data` must contain two or more observations along `axis`.")
        return a

def _ndtri(p: float) -> float:
    return K.ndtri(p)

def _lerp(a: float, b: float, t: float) -> float:
    d = b - a
    if t >= 0.5:
        return b - d * (1.0 - t)
    else:
        return a + d * t

def _quantile_sorted(s, p: float) -> float:
    n = s.shape[0]
    if p != p:
        return NAN
    pos = p * (n - 1)
    lo = int(pos)
    if lo < 0:
        lo = 0
    if lo >= n - 1:
        return s[n - 1]
    return _lerp(s[lo], s[lo + 1], pos - lo)

def _size(lead) -> int:
    n = 1
    for v in lead:
        n *= v
    return n

# ------------------------------------------------------------------ bootstrap

def _pct_of_score(row, score: float) -> float:
    B = row.shape[0]
    c = 0
    for i in range(B):
        if row[i] < score:
            c += 1
        if row[i] <= score:
            c += 1
    return c / (2.0 * B)

def _jack_idx(n: int):
    idx = np.empty((n, n - 1), dtype=int)
    for r in range(n):
        c = 0
        for k in range(n):
            if k != r:
                idx[r, c] = k
                c += 1
    return idx

def _jk_replace(exp, jk, j: Literal[int], k: Literal[int] = 0):
    if static.len(exp) == 0:
        return ()
    else:
        head = jk if k == j else exp[0]
        return (head,) + _jk_replace(exp[1:], jk, j, k + 1)

class _Expand:
    def __call__(self, a):
        return np.expand_dims(a, -2)

def _bca_ahat(ev, data, L: int):
    exp = _tmap(_Expand(), data)
    nums = np.zeros(L)
    dens = np.zeros(L)
    for j in static.range(static.len(data)):
        sample = data[j]
        n = sample.shape[-1]
        jk = sample[..., _jack_idx(n)]
        args = _bc_all(_jk_replace(exp, jk, j))
        th = ev(args)
        t2 = np.reshape(th, (L, n))
        for l in range(L):
            m = 0.0
            for i in range(n):
                m += t2[l, i]
            m /= n
            s3 = 0.0
            s2 = 0.0
            for i in range(n):
                u = (n - 1) * (m - t2[l, i])
                s3 += u * u * u
                s2 += u * u
            nums[l] += s3 / (float(n) ** 3)
            dens[l] += s2 / (float(n) ** 2)
    out = np.empty(L)
    for l in range(L):
        out[l] = 1.0 / 6.0 * nums[l] / (dens[l] ** 1.5)
    return out

def _boot_core(data, ev, n_resamples: int, batch: int, alternative: str, method: str,
               confidence_level: float, rng, prev):
    theta_hat = ev(data)
    lead = theta_hat.shape
    L = _size(lead)
    parts = []
    if not isinstance(prev, NoneType):
        parts.append(prev.bootstrap_distribution)
    if batch <= 0:
        nominal = n_resamples
    else:
        nominal = batch
    if nominal < 1:
        nominal = 1
    k = 0
    while k < n_resamples:
        ba = min(nominal, n_resamples - k)
        res = _tmap(_Resampler(rng, ba), data)
        parts.append(ev(res))
        k += nominal
    theta_b = np.concatenate(parts, axis=-1)
    B = theta_b.shape[-1]
    tb = np.reshape(theta_b, (L, B))
    alpha = (1.0 - confidence_level) / 2.0
    if alternative != 'two-sided':
        alpha = 1.0 - confidence_level
    a1 = np.empty(L)
    a2 = np.empty(L)
    if method == 'bca':
        th = np.reshape(theta_hat, (L,))
        ah = _bca_ahat(ev, data, L)
        za = _ndtri(alpha)
        for l in range(L):
            z0 = _ndtri(_pct_of_score(tb[l], th[l]))
            num1 = z0 + za
            a1[l] = K.ndtr(z0 + num1 / (1.0 - ah[l] * num1))
            num2 = z0 - za
            a2[l] = K.ndtr(z0 + num2 / (1.0 - ah[l] * num2))
    else:
        for l in range(L):
            a1[l] = alpha
            a2[l] = 1.0 - alpha
    lo = np.empty(L)
    hi = np.empty(L)
    se = np.empty(L)
    thf = np.reshape(theta_hat, (L,))
    for l in range(L):
        s = np.sort(tb[l])
        lo[l] = _quantile_sorted(s, a1[l])
        hi[l] = _quantile_sorted(s, a2[l])
        if method == 'basic':
            nl = 2 * thf[l] - hi[l]
            nh = 2 * thf[l] - lo[l]
            lo[l] = nl
            hi[l] = nh
        if alternative == 'less':
            lo[l] = -INF
        elif alternative == 'greater':
            hi[l] = INF
        m = 0.0
        for i in range(B):
            m += tb[l, i]
        m /= B
        v = 0.0
        for i in range(B):
            v += (tb[l, i] - m) * (tb[l, i] - m)
        se[l] = K.sqrt(v / (B - 1)) if B > 1 else NAN
    return BootstrapResult(_CI(_unwrap(np.reshape(lo, lead)), _unwrap(np.reshape(hi, lead))),
                           theta_b, _unwrap(np.reshape(se, lead)))

def bootstrap(data, statistic, *, n_resamples = 9999, batch = None, vectorized = None,
              paired: bool = False, axis: int = 0, confidence_level: float = 0.95,
              alternative: str = 'two-sided', method: str = 'BCa',
              bootstrap_result = None, rng = None):
    alt = alternative.lower()
    if alt != 'two-sided' and alt != 'less' and alt != 'greater':
        raise ValueError("`alternative` must be one of {'two-sided', 'less', 'greater'}")
    meth = method.lower()
    if meth != 'percentile' and meth != 'basic' and meth != 'bca':
        raise ValueError("`method` must be in {'percentile', 'basic', 'bca'}")
    nr = _nres(n_resamples)
    if nr < 0:
        raise ValueError("`n_resamples` must be a non-negative integer.")
    b = 0
    if not isinstance(batch, NoneType):
        b = int(batch)
        if b <= 0:
            raise ValueError("`batch` must be a positive integer or None.")
    if isinstance(bootstrap_result, NoneType):
        if nr == 0:
            raise ValueError("Either `bootstrap_result.bootstrap_distribution.size` or `n_resamples` must be positive.")
    g = np.random.default_rng(rng)
    d0 = _tmap(_CheckLen(), _prep(data, axis))
    ev0 = _Eval(statistic, _vec(vectorized))
    if paired:
        n = d0[0].shape[-1]
        if not _all_len(d0, n):
            raise ValueError("When `paired is True`, all samples must have the same length along `axis`")
        pev = _PairedEval(ev0, d0)
        return _boot_core((np.arange(n),), pev, nr, b, alt, meth, float(confidence_level), g, bootstrap_result)
    else:
        return _boot_core(d0, ev0, nr, b, alt, meth, float(confidence_level), g, bootstrap_result)

def _all_len(d, n: int) -> bool:
    if static.len(d) == 0:
        return True
    else:
        return d[0].shape[-1] == n and _all_len(d[1:], n)

# ------------------------------------------------------------------ p-values

def _pvalues(null, observed, alternative: str, adjustment: int, n_res: int):
    # null: (P,) + lead ; observed: lead
    obs = observed
    lead = obs.shape
    L = _size(lead)
    P = null.shape[0]
    nl = np.reshape(null, (P, L))
    ob = np.reshape(obs, (L,))
    out = np.empty(L)
    for l in range(L):
        o = ob[l]
        if o != o:
            out[l] = NAN
            continue
        gamma = abs(2.220446049250313e-16 * 100 * o)
        cl = 0
        cg = 0
        for p in range(P):
            v = nl[p, l]
            if v <= o + gamma:
                cl += 1
            if v >= o - gamma:
                cg += 1
        pl = (cl + adjustment) / (n_res + adjustment)
        pg = (cg + adjustment) / (n_res + adjustment)
        r = pl
        if alternative == 'greater':
            r = pg
        elif alternative == 'two-sided':
            r = min(pl, pg) * 2
        out[l] = min(max(r, 0.0), 1.0)
    return np.reshape(out, lead)

# ------------------------------------------------------------------ permutation test

def _perm_kth(n: int, k: int):
    # k-th permutation of range(n) in lexicographic order
    fact = 1
    for i in range(2, n):
        fact *= i
    avail = [i for i in range(n)]
    out = np.empty(n, dtype=int)
    kk = k
    for i in range(n):
        d = kk // fact if fact > 0 else 0
        kk = kk - d * fact
        out[i] = avail[d]
        avail.pop(d)
        if n - 1 - i > 0:
            fact = fact // (n - 1 - i)
    return out

def _fact_sat(n: int) -> float:
    f = 1.0
    for i in range(2, n + 1):
        f *= i
    return f

def _partitions(ns: List[int]):
    # all ways to split range(sum(ns)) into groups of sizes ns (concatenated rows)
    total = 0
    for v in ns:
        total += v
    rows = List[List[int]]()
    cur = List[int]()
    avail = [i for i in range(total)]
    _part_rec(ns, 0, avail, cur, rows)
    out = np.empty((len(rows), total), dtype=int)
    for r in range(len(rows)):
        for c in range(total):
            out[r, c] = rows[r][c]
    return out

def _choose(avail: List[int], k: int, start: int, pick: List[int], res: List[List[int]]):
    if len(pick) == k:
        res.append(pick.copy())
        return
    need = k - len(pick)
    for i in range(start, len(avail) - need + 1):
        pick.append(avail[i])
        _choose(avail, k, i + 1, pick, res)
        pick.pop()

def _part_rec(ns: List[int], gi: int, avail: List[int], cur: List[int], rows: List[List[int]]):
    if gi == len(ns):
        rows.append(cur.copy())
        return
    if gi == len(ns) - 1:
        c0 = len(cur)
        for v in avail:
            cur.append(v)
        rows.append(cur.copy())
        for _ in range(len(avail)):
            cur.pop()
        return
    combos = List[List[int]]()
    _choose(avail, ns[gi], 0, List[int](), combos)
    for c in combos:
        rest = List[int]()
        ci = 0
        for v in avail:
            if ci < len(c) and c[ci] == v:
                ci += 1
            else:
                rest.append(v)
        for v in c:
            cur.append(v)
        _part_rec(ns, gi + 1, rest, cur, rows)
        for _ in range(len(c)):
            cur.pop()

def _cat_data(data):
    return np.concatenate(data, axis=-1)

def _split(data, cat, off: int):
    if static.len(data) == 0:
        return ()
    else:
        n = data[0].shape[-1]
        return (cat[..., off:off + n],) + _split(data[1:], cat, off + n)

def _n_total(data) -> int:
    if static.len(data) == 0:
        return 0
    else:
        return data[0].shape[-1] + _n_total(data[1:])

def _comb_f(n: int, k: int) -> float:
    r = 1.0
    for i in range(1, k + 1):
        r = r * (n - k + i) / i
    return r

def _n_max_indep(data) -> float:
    ns = _ns(data)
    prod = 1.0
    cum = 0
    for v in ns:
        cum += v
        prod *= _comb_f(cum, v)
    return prod

def _ns(data):
    if static.len(data) == 0:
        return List[int]()
    else:
        r = [data[0].shape[-1]]
        for v in _ns(data[1:]):
            r.append(v)
        return r

def _run_batches(ev, make_batch, P: int, batch: int):
    # make_batch(k0, k1) -> tuple of data with leading perm axis
    parts = []
    nom = batch if batch > 0 else P
    if nom < 1:
        nom = 1
    k = 0
    while k < P:
        k1 = min(P, k + nom)
        parts.append(ev(make_batch.build(k, k1)))
        k = k1
    return np.concatenate(parts, axis=0)

class _IndepBatch[D, C]:
    data: D
    cat: C
    idx: np.ndarray[int, 2]
    def build(self, k0: int, k1: int):
        ind = self.idx[k0:k1]
        gb = np.moveaxis(self.cat[..., ind], -2, 0)
        return _split(self.data, gb, 0)

def _permute_indep(data, ev, nperm: int, batch: int, rng):
    nmax = _n_max_indep(data)
    ntot = _n_total(data)
    exact = float(nperm) >= nmax
    idx = np.empty((0, ntot), dtype=int)
    P = nperm
    if exact:
        idx = _partitions(_ns(data))
        P = idx.shape[0]
    else:
        idx = np.empty((nperm, ntot), dtype=int)
        for i in range(nperm):
            idx[i] = rng.permutation(ntot)
    cat = _cat_data(data)
    nd = _run_batches(ev, _IndepBatch(data, cat, idx), P, batch)
    return nd, P, exact

class _PairBatch[D]:
    data: D
    idx: np.ndarray[int, 3]
    def build(self, k0: int, k1: int):
        return _pair_tuple(self.data, self.idx, k0, k1, 0)

def _pair_tuple(data, idx, k0: int, k1: int, s: int):
    if static.len(data) == 0:
        return ()
    else:
        a = data[0]
        sub = idx[k0:k1, s, :]
        g = np.moveaxis(a[..., sub], -2, 0)
        return (g,) + _pair_tuple(data[1:], idx, k0, k1, s + 1)

def _permute_pairings(data, ev, nperm: int, batch: int, rng):
    S = static.len(data)
    n = data[0].shape[-1]
    nmax = _fact_sat(n) ** S
    exact = float(nperm) >= nmax
    P = nperm
    if exact:
        P = int(nmax)
    idx = np.empty((P, S, n), dtype=int)
    if exact:
        nf = int(_fact_sat(n))
        for r in range(P):
            rem = r
            for s in range(S - 1, -1, -1):
                d = rem % nf
                rem = rem // nf
                idx[r, s] = _perm_kth(n, d)
    else:
        base = np.tile(np.arange(n), (P, S, 1))
        idx = rng.permuted(base, axis=-1)
    nd = _run_batches(ev, _PairBatch(data, idx), P, batch)
    return nd, P, exact

class _SampBatch[D, XT]:
    data: D
    xx: XT
    idx: np.ndarray[int, 3]
    S: int
    def build(self, k0: int, k1: int):
        return _samp_tuple(self.data, self.xx, self.idx, k0, k1, 0)

def _samp_tuple(data, X, idx, k0: int, k1: int, s: Literal[int]):
    if static.len(data) == 0:
        return ()
    else:
        S = X.shape[-2]
        n = X.shape[-1]
        B = k1 - k0
        lead = X.shape[:-2]
        out = np.empty((B,) + lead + (n,), dtype=float)
        nl = _size(lead)
        Xf = np.reshape(X, (nl, S, n))
        of = np.reshape(out, (B, nl, n))
        for b in range(B):
            for l in range(nl):
                for j in range(n):
                    of[b, l, j] = Xf[l, idx[k0 + b, j, s], j]
        return (out,) + _samp_tuple(data[1:], X, idx, k0, k1, s + 1)

def _permute_samples(data, ev, nperm: int, batch: int, rng):
    if static.len(data) == 1:
        data2 = (data[0], -data[0])
        return _permute_samples_core(data2, data, ev, nperm, batch, rng)
    else:
        return _permute_samples_core(data, data, ev, nperm, batch, rng)

class _TakeFirst[E]:
    ev: E
    def __call__(self, d):
        return self.ev((d[0],))

def _permute_samples_core(data2, data0, ev, nperm: int, batch: int, rng):
    S = static.len(data2)
    n = data2[0].shape[-1]
    nmax = _fact_sat(S) ** n
    exact = float(nperm) >= nmax
    P = nperm
    if exact:
        P = int(nmax)
    idx = np.empty((P, n, S), dtype=int)
    if exact:
        nf = int(_fact_sat(S))
        for r in range(P):
            rem = r
            for j in range(n - 1, -1, -1):
                d = rem % nf
                rem = rem // nf
                idx[r, j] = _perm_kth(S, d)
    else:
        base = np.tile(np.arange(S), (P, n, 1))
        idx = rng.permuted(base, axis=-1)
    X = np.stack(data2, axis=-2)
    if static.len(data0) == 1:
        nd = _run_batches(_TakeFirst(ev), _SampBatch(data2, X, idx, S), P, batch)
    else:
        nd = _run_batches(ev, _SampBatch(data2, X, idx, S), P, batch)
    return nd, P, exact

def permutation_test(data, statistic, *, permutation_type: str = 'independent',
                     vectorized = None, n_resamples = 9999, batch = None,
                     alternative: str = 'two-sided', axis: int = 0, rng = None):
    pt = permutation_type.lower()
    if pt != 'independent' and pt != 'samples' and pt != 'pairings':
        raise ValueError("`permutation_type` must be in {'samples', 'pairings', 'independent'}.")
    alt = alternative.lower()
    if alt != 'two-sided' and alt != 'less' and alt != 'greater':
        raise ValueError("`alternative` must be in {'two-sided', 'greater', 'less'}")
    if static.len(data) < 2 and pt == 'independent':
        raise ValueError("`data` must be a tuple containing at least two samples")
    nr = _nres(n_resamples)
    if nr <= 0:
        raise ValueError("`n_resamples` must be a positive integer.")
    b = 0
    if not isinstance(batch, NoneType):
        b = int(batch)
        if b <= 0:
            raise ValueError("`batch` must be a positive integer or None.")
    g = np.random.default_rng(rng)
    d = _tmap(_CheckLen(), _prep(data, axis))
    ev = _Eval(statistic, _vec(vectorized))
    observed = ev(d)
    if pt == 'independent':
        r = _permute_indep(d, ev, nr, b, g)
        return _perm_finish(observed, r[0], r[1], r[2], alt)
    elif pt == 'pairings':
        r = _permute_pairings(d, ev, nr, b, g)
        return _perm_finish(observed, r[0], r[1], r[2], alt)
    else:
        r = _permute_samples(d, ev, nr, b, g)
        return _perm_finish(observed, r[0], r[1], r[2], alt)

def _perm_finish(observed, null, P: int, exact: bool, alt: str):
    adj = 0 if exact else 1
    pv = _pvalues(null, observed, alt, adj, P)
    return PermutationTestResult(_unwrap(observed), _unwrap(pv), null)

# ------------------------------------------------------------------ monte carlo test

def _mc_draw(rvs, data, B: int):
    if static.len(data) == 0:
        return ()
    else:
        n = data[0].shape[-1]
        return (_asf(rvs[0](size=(B, n))),) + _mc_draw(rvs[1:], data[1:], B)

def monte_carlo_test(data, rvs, statistic, *, vectorized = None, n_resamples = 9999,
                     batch = None, alternative: str = 'two-sided', axis: int = 0):
    alt = alternative.lower()
    if alt != 'two-sided' and alt != 'less' and alt != 'greater':
        raise ValueError("`alternative` must be in {'two-sided', 'greater', 'less'}")
    nr = _nres(n_resamples)
    if nr <= 0:
        raise ValueError("`n_resamples` must be a positive integer.")
    b = 0
    if not isinstance(batch, NoneType):
        b = int(batch)
        if b <= 0:
            raise ValueError("`batch` must be a positive integer or None.")
    if isinstance(data, Tuple):
        return _mc_run(_prep(data, axis), rvs, statistic, vectorized, nr, b, alt)
    else:
        return _mc_run(_prep((data,), axis), (rvs,), statistic, vectorized, nr, b, alt)

def _mc_run(d, rvs, statistic, vectorized, nr: int, b: int, alt: str):
    if static.len(rvs) != static.len(d):
        raise ValueError("If `rvs` is a sequence, `len(rvs)` must equal `len(data)`.")
    ev = _Eval(statistic, _vec(vectorized))
    observed = ev(d)
    parts = []
    nom = b if b > 0 else nr
    k = 0
    while k < nr:
        ba = min(nom, nr - k)
        parts.append(ev(_mc_draw(rvs, d, ba)))
        k += nom
    null = np.concatenate(parts, axis=0)
    pv = _pvalues(null, observed, alt, 1, nr)
    return MonteCarloTestResult(_unwrap(observed), _unwrap(pv), null)
