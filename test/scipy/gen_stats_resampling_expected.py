# Generates expected values for test/scipy/test_stats_resampling.py (scipy 1.18.1).
# Usage: python gen_stats_resampling_expected.py  (prints `name = value` lines)
import numpy as np
from scipy.stats import bootstrap, permutation_test, monte_carlo_test

def emit(name, v):
    a = np.asarray(v, dtype=float)
    print(f"{name} = {a.tolist()!r}")

x = np.array([2.1, 3.4, 1.9, 5.6, 4.2, 3.3, 2.8, 6.1, 4.4, 3.9, 2.2, 5.0])
y = np.array([1.8, 2.9, 3.1, 4.7, 5.2, 2.6, 3.0, 5.9, 3.5, 4.1, 2.5, 4.4])
X2 = np.cos(np.arange(30.)).reshape(3, 10) * 3 + 4

def md(a, b, axis=-1):
    return np.mean(a, axis=axis) - np.mean(b, axis=axis)

def corr(a, b, axis=-1):
    am = a - np.mean(a, axis=axis, keepdims=True)
    bm = b - np.mean(b, axis=axis, keepdims=True)
    return np.sum(am*bm, axis=axis) / np.sqrt(np.sum(am*am, axis=axis)*np.sum(bm*bm, axis=axis))

def ci(r):
    return [np.asarray(r.confidence_interval.low), np.asarray(r.confidence_interval.high), np.asarray(r.standard_error)]

out = []
r = bootstrap((x,), np.mean, n_resamples=500, rng=12); out += ci(r)
emit('b_bca', out)
out = []
r = bootstrap((x, y), md, n_resamples=500, rng=12, method='basic'); out += ci(r)
r = bootstrap((x, y), md, n_resamples=500, rng=12, method='percentile', paired=True); out += ci(r)
r = bootstrap((x, y), md, n_resamples=300, rng=5, method='bca', confidence_level=0.9); out += ci(r)
r = bootstrap((x,), np.median, n_resamples=300, rng=5, method='percentile', alternative='less', vectorized=True); out += ci(r)
emit('b_misc', out)
r = bootstrap((X2,), np.mean, axis=1, n_resamples=200, rng=9, method='bca')
emit('b_ax', [r.confidence_interval.low, r.confidence_interval.high, r.standard_error])
r2 = bootstrap((x,), np.mean, n_resamples=100, rng=4, bootstrap_result=r if False else None, method='percentile')
emit('b_dist', [r2.bootstrap_distribution.shape[0], np.sum(r2.bootstrap_distribution)])

out = []
p = permutation_test((x, y), md, n_resamples=999, rng=3); out += [p.statistic, p.pvalue, p.null_distribution.sum()]
p = permutation_test((x[:5], y[:4]), md, n_resamples=999, rng=3); out += [p.statistic, p.pvalue, p.null_distribution.shape[0], p.null_distribution.sum()]
p = permutation_test((x[:6], y[:6]), md, permutation_type='samples', n_resamples=999, rng=3); out += [p.statistic, p.pvalue, p.null_distribution.shape[0]]
p = permutation_test((x, y), md, permutation_type='samples', n_resamples=500, rng=3, alternative='greater'); out += [p.statistic, p.pvalue]
p = permutation_test((x,), np.mean, permutation_type='samples', n_resamples=500, rng=3); out += [p.statistic, p.pvalue]
p = permutation_test((x, y), corr, permutation_type='pairings', n_resamples=500, rng=3, alternative='less'); out += [p.statistic, p.pvalue]
p = permutation_test((x[:5], y[:5]), corr, permutation_type='pairings', n_resamples=500, rng=3); out += [p.statistic, p.pvalue, p.null_distribution.shape[0]]
p = permutation_test((X2[:, :5], X2[:, 5:]), md, n_resamples=200, rng=8, axis=1); out += list(p.statistic) + list(p.pvalue)
emit('perm', out)

g = np.random.default_rng(7)
m = monte_carlo_test(x, lambda size: g.standard_normal(size), np.mean, n_resamples=500, vectorized=True)
emit('mc', [m.statistic, m.pvalue, m.null_distribution.sum()])
