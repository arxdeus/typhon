# Copyright (C) 2026 the Typhon authors
# Expected values from scipy 1.18.1 (fit with differential_evolution rng=1; goodness_of_fit rng=5).
import numpy as np
import scipy.stats as st

def near(a: float, b: float, rtol: float = 1e-6) -> bool:
    ok = abs(a - b) <= 1e-9 + rtol * abs(b)
    if not ok:
        print('mismatch', a, b)
    return ok

y = np.array([2.1, 3.4, 1.9, 5.6, 4.2, 3.3, 2.8, 6.1, 4.4, 3.9, 2.2, 5.0])
x = np.array([10.159757848963785, 4.263299051565745, 2.3495227203650284, 0.5190801351653268,
              1.6895124885640687, 3.1, 2.2, 5.9, 0.8, 1.1, 2.7, 3.3, 6.2, 1.4, 2.0])

@test
def test_gof():
    ad = st.goodness_of_fit(st.norm, y, statistic='ad', n_mc_samples=499, rng=5)
    assert near(ad.statistic, 0.19324201838101374) and near(ad.pvalue, 0.888)
    assert near(ad.fit_result.params[0], 3.7416666666666667) and near(ad.fit_result.params[1], 1.3793663455193892)
    ks = st.goodness_of_fit(st.norm, y, statistic='ks', n_mc_samples=499, rng=5)
    assert near(ks.statistic, 0.11814451510611668) and near(ks.pvalue, 0.92)
    cv = st.goodness_of_fit(st.norm, y, statistic='cvm', n_mc_samples=499, rng=5)
    assert near(cv.statistic, 0.02311917036648323) and near(cv.pvalue, 0.948)
    fl = st.goodness_of_fit(st.norm, y, statistic='filliben', n_mc_samples=499, rng=5)
    assert near(fl.statistic, 0.9868934821658963) and near(fl.pvalue, 0.876)
    kn = st.goodness_of_fit(st.norm, y, known_params={'loc': 3.5}, statistic='ks', n_mc_samples=499, rng=5)
    assert near(kn.statistic, 0.11805806815431164) and near(kn.pvalue, 0.972)

@test
def test_fit():
    r = st.fit(st.norm, y, {'loc': (-5.0, 10.0), 'scale': (0.1, 10.0)}, rng=1)
    assert r.success
    assert near(r.params[0], float(np.mean(y)), 1e-3) and near(r.params[1], float(np.std(y)), 1e-3)
    assert near(r.nllf(), 0.5 * 12 * (1.0 + 1.8378770664093453) + 12 * float(np.log(np.std(y))), 1e-5)
    g = st.fit(st.gamma, x, [(0.5, 10.0), (-1.0, 1.0), (0.1, 10.0)], rng=1)
    assert g.success and g.params[0] > 0.5

test_gof()
test_fit()
