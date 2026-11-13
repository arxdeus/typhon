# Copyright (C) 2026 the Typhon authors
# Shared checker for test_dists_*.py (expected values from gen_dists_expected.py).
import numpy as np

NAN = float('nan')
INF = float('inf')

def close(name: str, what: str, x, ref: List[float], rtol: float) -> bool:
    a = np.asarray(x, dtype=float).ravel()
    if len(a) != len(ref):
        print(name, what, 'length', len(a), len(ref))
        return False
    ok = True
    for i in range(len(ref)):
        u = float(a[i])
        v = ref[i]
        if v != v:
            good = u != u
        elif v == INF or v == -INF:
            good = u == v
        else:
            good = abs(u - v) <= 1e-300 + rtol * abs(v) + (rtol * 1e-3 if v == 0 else 0.0)
        if not good:
            print(name, what, i, 'got', u, 'expected', v)
            ok = False
    return ok

def check(name: str, d, args, xs: List[float], qs: List[float],
          pdf: List[float], logpdf: List[float], cdf: List[float], sf: List[float],
          logcdf: List[float], logsf: List[float], ppf: List[float], isf: List[float],
          mvsk: List[float], ent: float, rvs: List[float], rvg: List[float], rtol: float) -> bool:
    fr = d(*args, loc=0.5, scale=1.5)
    x = np.array(xs)
    q = np.array(qs)
    ok = close(name, 'pdf', fr.pdf(x), pdf, rtol)
    ok &= close(name, 'logpdf', fr.logpdf(x), logpdf, rtol)
    ok &= close(name, 'cdf', fr.cdf(x), cdf, rtol)
    ok &= close(name, 'sf', fr.sf(x), sf, rtol)
    ok &= close(name, 'logcdf', fr.logcdf(x), logcdf, rtol)
    ok &= close(name, 'logsf', fr.logsf(x), logsf, rtol)
    ok &= close(name, 'ppf', fr.ppf(q), ppf, rtol)
    ok &= close(name, 'isf', fr.isf(q), isf, rtol)
    m, v, s, k = fr.stats(moments='mvsk')
    ok &= close(name, 'mvsk', [m, v, s, k], mvsk, max(rtol, 1e-9))
    ok &= close(name, 'entropy', [fr.entropy()], [ent], max(rtol, 1e-9))
    ok &= close(name, 'rvs', fr.rvs(size=5, random_state=np.random.RandomState(1234)), rvs, max(rtol, 1e-11))
    ok &= close(name, 'rvs-gen', fr.rvs(size=5, random_state=np.random.default_rng(42)), rvg, max(rtol, 1e-11))
    # scalar path and the unfrozen API
    ok &= close(name, 'cdf-scalar', [d.cdf(xs[2], *args, loc=0.5, scale=1.5)], [cdf[2]], rtol)
    return ok

def dcheck(name: str, d, args, ks: List[float], qs: List[float],
           pmf: List[float], logpmf: List[float], cdf: List[float], sf: List[float],
           logcdf: List[float], logsf: List[float], ppf: List[float], isf: List[float],
           mvsk: List[float], ent: float, rvs: List[float], rvg: List[float], rtol: float) -> bool:
    fr = d(*args, loc=1.0)
    k = np.array(ks)
    q = np.array(qs)
    ok = close(name, 'pmf', fr.pmf(k), pmf, rtol)
    ok &= close(name, 'logpmf', fr.logpmf(k), logpmf, rtol)
    ok &= close(name, 'cdf', fr.cdf(k), cdf, rtol)
    ok &= close(name, 'sf', fr.sf(k), sf, rtol)
    ok &= close(name, 'logcdf', fr.logcdf(k), logcdf, rtol)
    ok &= close(name, 'logsf', fr.logsf(k), logsf, rtol)
    ok &= close(name, 'ppf', fr.ppf(q), ppf, 0.0)
    ok &= close(name, 'isf', fr.isf(q), isf, 0.0)
    m, v, s, kk = fr.stats(moments='mvsk')
    ok &= close(name, 'mvsk', [m, v, s, kk], mvsk, max(rtol, 1e-9))
    ok &= close(name, 'entropy', [fr.entropy()], [ent], max(rtol, 1e-9))
    ok &= close(name, 'rvs', fr.rvs(size=6, random_state=np.random.RandomState(1234)), rvs, 0.0)
    ok &= close(name, 'rvs-gen', fr.rvs(size=6, random_state=np.random.default_rng(42)), rvg, 0.0)
    ok &= close(name, 'pmf-unfrozen', [d.pmf(ks[2], *args, loc=1.0)], [pmf[2]], rtol)
    return ok
