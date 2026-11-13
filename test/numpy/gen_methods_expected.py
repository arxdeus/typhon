# Generates test_ndarray_methods.py: ndarray methods / module functions vs numpy.
# Run with the reference venv: python gen_methods_expected.py > test_ndarray_methods.py
import numpy as np

PRE = '''
a = np.array([[3.0, 1.0, 2.0, 9.0], [7.0, 5.0, 4.0, 8.0], [6.0, 0.5, 2.5, 1.5]])
v = np.array([5.0, 3.0, 8.0, 1.0, 9.0, 2.0, 7.0])
w = np.array([1.0, 2.0, 1.0, 0.5, 3.0, 1.0, 2.0])
iv = np.array([4, 1, 3, 1, 5, 9, 2, 6])
'''
CASES = [
    ("partition_sorted_kth", "np.partition(v, 3)[3]"),
    ("partition_ax", "np.sort(np.partition(a, 1, axis=1)[:, :2], axis=1)"),
    ("argpartition_kth", "v[np.argpartition(v, 2)[2]]"),
    ("choose", "np.choose(np.array([0, 1, 0, 1]), [np.array([1.0, 2.0, 3.0, 4.0]), np.array([10.0, 20.0, 30.0, 40.0])])"),
    ("compress", "np.compress(np.array([True, False, True]), a, axis=0)"),
    ("compress_flat", "np.compress(v > 4, v)"),
    ("searchsorted_l", "np.searchsorted(np.sort(v), np.array([0.0, 5.0, 10.0]))"),
    ("searchsorted_r", "np.searchsorted(np.sort(v), np.array([0.0, 5.0, 10.0]), side='right')"),
    ("searchsorted_sorter", "np.searchsorted(v, np.array([0.0, 5.0, 10.0]), sorter=np.argsort(v))"),
    ("ptp", "np.ptp(a, axis=1)"),
    ("ptp_all", "np.ptp(a)"),
    ("resize", "np.resize(v, (3, 4))"),
    ("piecewise", "np.piecewise(v, [v < 4, v >= 4], [-1.0, 100.0])"),
    ("select", "np.select([v < 3, v > 7], [v, v * 10.0], default=-1.0)"),
    ("apply_along_axis", "np.apply_along_axis(lambda r: r.sum(), 1, a)"),
    ("lexsort", "np.lexsort((np.array([3, 1, 2, 1, 3, 2, 1, 2]), iv))"),
    ("packbits", "np.packbits(np.array([1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0], dtype=np.uint8))"),
    ("unpackbits", "np.unpackbits(np.array([177, 224], dtype=np.uint8))"),
    ("pad_const", "np.pad(a, ((1, 2), (0, 1)), constant_values=7.0)"),
    ("pad_edge", "np.pad(a, 2, mode='edge')"),
    ("pad_reflect", "np.pad(a, ((1, 1), (2, 2)), mode='reflect')"),
    ("pad_symmetric", "np.pad(a, ((1, 1), (2, 2)), mode='symmetric')"),
    ("pad_wrap", "np.pad(v, (3, 4), mode='wrap')"),
    ("pad_linear_ramp", "np.pad(v, (3, 2), mode='linear_ramp', end_values=(10.0, -4.0))"),
    ("pad_maximum", "np.pad(v, (2, 2), mode='maximum')"),
    ("pad_mean", "np.pad(v, (2, 3), mode='mean')"),
    ("pad_minimum", "np.pad(v, (2, 2), mode='minimum', stat_length=3)"),
    ("pad_median", "np.pad(v, (2, 2), mode='median')"),
    ("pad_odd", "np.pad(v, (3, 2), mode='reflect', reflect_type='odd')"),
    ("pq_linear", "np.percentile(v, [10.0, 50.0, 90.0])"),
    ("pq_lower", "np.percentile(v, [10.0, 50.0, 90.0], method='lower')"),
    ("pq_higher", "np.percentile(v, [10.0, 50.0, 90.0], method='higher')"),
    ("pq_nearest", "np.percentile(v, [10.0, 50.0, 90.0], method='nearest')"),
    ("pq_midpoint", "np.percentile(v, [10.0, 50.0, 90.0], method='midpoint')"),
    ("pq_inverted_cdf", "np.percentile(v, [10.0, 50.0, 90.0], method='inverted_cdf')"),
    ("pq_averaged_inverted_cdf", "np.percentile(v, [10.0, 50.0, 90.0], method='averaged_inverted_cdf')"),
    ("pq_closest_observation", "np.percentile(v, [10.0, 50.0, 90.0], method='closest_observation')"),
    ("pq_interp_inv_cdf", "np.percentile(v, [10.0, 50.0, 90.0], method='interpolated_inverted_cdf')"),
    ("pq_hazen", "np.percentile(v, [10.0, 50.0, 90.0], method='hazen')"),
    ("pq_weibull", "np.percentile(v, [10.0, 50.0, 90.0], method='weibull')"),
    ("pq_median_unbiased", "np.percentile(v, [10.0, 50.0, 90.0], method='median_unbiased')"),
    ("pq_normal_unbiased", "np.percentile(v, [10.0, 50.0, 90.0], method='normal_unbiased')"),
    ("quantile_ax", "np.quantile(a, 0.3, axis=1)"),
    ("hbe_auto", "np.histogram_bin_edges(v, bins='auto')"),
    ("hbe_fd", "np.histogram_bin_edges(v, bins='fd')"),
    ("hbe_doane", "np.histogram_bin_edges(v, bins='doane')"),
    ("hbe_scott", "np.histogram_bin_edges(v, bins='scott')"),
    ("hbe_stone", "np.histogram_bin_edges(v, bins='stone')"),
    ("hbe_rice", "np.histogram_bin_edges(v, bins='rice')"),
    ("hbe_sturges", "np.histogram_bin_edges(v, bins='sturges')"),
    ("hbe_sqrt", "np.histogram_bin_edges(v, bins='sqrt')"),
    ("cov_fw", "np.cov(a, fweights=np.array([1, 2, 1, 3]))"),
    ("cov_aw", "np.cov(a, aweights=np.array([0.5, 1.0, 2.0, 1.0]))"),
    ("cov_both", "np.cov(a, fweights=np.array([1, 2, 1, 3]), aweights=np.array([0.5, 1.0, 2.0, 1.0]), ddof=0)"),
    ("corrcoef", "np.corrcoef(a)"),
    ("gradient_eo2", "np.gradient(v, edge_order=2)"),
    ("gradient_x", "np.gradient(v, np.array([0.0, 1.0, 3.0, 4.0, 7.0, 8.0, 10.0]))"),
    ("gradient_dx", "np.gradient(v, 2.5)"),
    ("gradient_2d", "np.gradient(a, 0.5, 2.0)[1]"),
    ("histogramdd3", "np.histogramdd(np.stack([v, w, v * w], axis=1), bins=3, ndim=3)[0]"),
    ("mask_indices", "np.mask_indices(4, np.triu)[1]"),
    ("put", "_put()"),
    ("putmask", "_putmask()"),
    ("place", "_place()"),
    ("byteswap", "np.array([1, 256, 3], dtype=np.int16).byteswap()"),
    ("tobytes", "np.frombuffer(np.array([1.0, 2.0]).tobytes(), dtype=np.float64)"),
    ("apply_over_axes", "np.apply_over_axes(np.sum, a, [0, 1])"),
    ("nanpercentile", "np.nanpercentile(np.array([1.0, np.nan, 3.0, 4.0]), 40.0)"),
]

HELP = '''
def _put():
    x = np.zeros(6)
    np.put(x, [1, 3, 8], [10.0, 20.0, 30.0], mode='wrap')
    return x

def _putmask():
    x = np.arange(8.0)
    np.putmask(x, x > 4, np.array([-1.0, -2.0]))
    return x

def _place():
    x = np.arange(8.0)
    np.place(x, x % 2 == 0, np.array([100.0, 200.0]))
    return x
'''

def L(v):
    return [float(t) for t in np.asarray(v, dtype=np.float64).ravel()]

g = {"np": np}
exec(PRE, g)
exec(HELP, g)
print("# Generated by test/numpy/gen_methods_expected.py; expected values from numpy %s." % np.__version__)
print("import numpy as np")
print(PRE)
print(HELP)
print("def _same(got, exp: List[float]) -> bool:")
print("    g = [float(t) for t in np.asarray(got, dtype=float).ravel()]")
print("    if len(g) != len(exp):")
print("        return False")
print("    for i in range(len(g)):")
print("        d = g[i] - exp[i]")
print("        if d < 0.0:")
print("            d = -d")
print("        e = exp[i] if exp[i] >= 0.0 else -exp[i]")
print("        if d > 1e-9 * (1.0 + e):")
print("            return False")
print("    return True")
print()
for label, expr in CASES:
    exp = eval(expr.replace(', ndim=3', ''), g)
    print("@test")
    print("def test_%s():" % label)
    print("    assert _same(%s, %r)" % (expr, L(exp)))
    print("test_%s()" % label)
    print()
