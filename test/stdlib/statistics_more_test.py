# Ported from CPython 3.14 statistics (PSF); exact values from the reference interpreter
import statistics as S

@test
def test_exact():
    d = [2.5, 3.1, 4.7, 5.0, 5.0, 6.2, 9.9, 1.3]
    x = [1.0, 2.0, 3.0, 4.0, 5.5]
    y = [2.1, 3.9, 6.2, 7.8, 11.1]
    w = [1, 2, 3, 4, 5, 6, 7, 8]
    assert S.harmonic_mean(d, w) == 3.1824279175903847
    assert S.fmean(d, w) == 5.130555555555556
    assert S.pvariance(d) == 6.05359375
    assert S.variance(d) == 6.918392857142858
    assert S.pstdev(d) == 2.4604052003684274
    assert S.stdev(d) == 2.630283797833013
    assert S.pvariance(d, 4.0) == 6.56125
    assert S.variance([1e5, 1e5 + 1e-3, 1e5 + 2e-3]) == 9.99999993131496e-07
    assert S.linear_regression(x, y).slope == 1.9950819672131148
    assert S.linear_regression(x, y).intercept == 0.03524590163934427
    assert S.correlation(x, y) == 0.9988928600593368
    assert S.median_low(d) == 4.7 and S.median_high(d) == 5.0
    assert S.multimode(d) == [5.0]
    assert S.quantiles(d, n=4) == [2.65, 4.85, 5.9]

test_exact()
