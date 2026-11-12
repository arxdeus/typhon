# Generates expected values for test_ma.py (run with the reference venv).
import numpy as np
import numpy.ma as ma

def L(v):
    return [float(t) for t in np.asarray(v, dtype=float).ravel()]

def M(v):
    return [bool(t) for t in np.asarray(ma.getmaskarray(v)).ravel()]

x = ma.array([[1., 2., 3., 4.], [5., 6., 7., 8.], [9., 10., 11., 12.]],
             mask=[[0, 1, 0, 0], [0, 0, 0, 1], [1, 1, 1, 1]])
y = ma.array([2., 0., -1., 4.], mask=[0, 0, 0, 1])
out = []
out.append(("sum0", L(x.sum(axis=0).filled(0)), M(x.sum(axis=0))))
out.append(("sum1", L(x.sum(axis=1).filled(0)), M(x.sum(axis=1))))
out.append(("mean0", L(x.mean(axis=0).filled(0)), M(x.mean(axis=0))))
out.append(("var1", L(x.var(axis=1).filled(0)), M(x.var(axis=1))))
out.append(("std0d1", L(x.std(axis=0, ddof=1).filled(0)), M(x.std(axis=0, ddof=1))))
out.append(("min1", L(x.min(axis=1).filled(0)), M(x.min(axis=1))))
out.append(("max0", L(x.max(axis=0).filled(0)), M(x.max(axis=0))))
out.append(("prod1", L(x.prod(axis=1).filled(0)), M(x.prod(axis=1))))
out.append(("argmin1", L(x.argmin(axis=1)), []))
out.append(("argmax0", L(x.argmax(axis=0)), []))
out.append(("count0", L(x.count(axis=0)), []))
out.append(("scal", [float(x.sum()), float(x.mean()), float(x.var()), float(x.std()), float(x.min()), float(x.max()), float(x.count()), float(x.argmin()), float(x.argmax())], []))
out.append(("div", L((x[0] / y).filled(0)), M(x[0] / y)))
out.append(("add", L((x[1] + y).filled(0)), M(x[1] + y)))
out.append(("pow", L((y ** 2).filled(0)), M(y ** 2)))
out.append(("sqrt", L(ma.sqrt(y).filled(0)), M(ma.sqrt(y))))
out.append(("log", L(ma.log(y).filled(0)), M(ma.log(y))))
out.append(("gt", L((y > 1.).filled(False)), M(y > 1.)))
out.append(("median0", L(ma.median(x, axis=0).filled(0)), M(ma.median(x, axis=0))))
out.append(("median_all", [float(ma.median(x))], []))
w = [1., 2., 3., 4.]
out.append(("average1", L(ma.average(x, axis=1, weights=w).filled(0)), M(ma.average(x, axis=1, weights=w))))
out.append(("average_all", [float(ma.average(x))], []))
c = ma.cov(x[:2])
out.append(("cov", L(c.filled(0)), M(c)))
r = ma.corrcoef(x[:2])
out.append(("corrcoef", L(r.filled(0)), M(r)))
z = np.array([1., np.nan, 3., np.inf, 5.])
out.append(("invalid", L(ma.masked_invalid(z).filled(0)), M(ma.masked_invalid(z))))
v = np.arange(8.)
out.append(("inside", [], M(ma.masked_inside(v, 2, 4))))
out.append(("outside", [], M(ma.masked_outside(v, 2, 4))))
out.append(("values", [], M(ma.masked_values(np.array([1., 1.1, 1.0000001]), 1.0))))
cc = ma.concatenate([x[0], y])
out.append(("concat", L(cc.filled(0)), M(cc)))
wh = ma.where(y > 0., y, -y)
out.append(("where", L(wh.filled(0)), M(wh)))
for name, d, m in out:
    print(f"    ({name!r}, {d!r}, {m!r}),")
print(repr(x[:2, :2]))
print(str(y), repr(y))
print([(s.start, s.stop) for s in ma.clump_masked(ma.masked_greater(v, 5) + ma.masked_less(v, 2))])
