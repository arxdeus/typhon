# Copyright (C) 2026 the Typhon authors
# Expected values from scipy 1.18.1 barnard_exact / boschloo_exact (see gen_stats_exact2x2 inline in the file header of the commit).
import numpy as np
import scipy.stats as st

def near(a: float, b: float, rtol: float = 2e-3) -> bool:
    ok = abs(a - b) <= 1e-9 + rtol * abs(b)
    if not ok:
        print('mismatch', a, b)
    return ok

# (table, alternative, barnard stat, barnard p, boschloo stat, boschloo p)
@test
def test_exact():
    t1 = [[7, 17], [15, 5]]
    t2 = [[74, 31], [43, 32]]
    t3 = [[2, 7], [8, 2]]
    b = st.barnard_exact(t1)
    assert near(b.statistic, -3.0276503540974917) and near(b.pvalue, 0.0027668133596095543)
    c = st.boschloo_exact(t1)
    assert near(c.statistic, 0.0028841933752349743) and near(c.pvalue, 0.0030282813335134255)
    b = st.barnard_exact(t1, alternative='less')
    assert near(b.pvalue, 0.001383406679804777)
    c = st.boschloo_exact(t1, alternative='greater')
    assert near(c.pvalue, 0.9989766371198303)
    b = st.barnard_exact(t2, alternative='greater')
    assert near(b.statistic, 1.8225863590257527) and near(b.pvalue, 0.04691595404276845)
    c = st.boschloo_exact(t2, alternative='greater')
    assert near(c.statistic, 0.048312210086912236) and near(c.pvalue, 0.03556406430154655)
    b = st.barnard_exact(t3)
    assert near(b.pvalue, 0.01921081542968749)
    c = st.boschloo_exact(t3, alternative='less')
    assert near(c.pvalue, 0.009886140844640604)
    b = st.barnard_exact(t1, pooled=False)
    assert near(b.statistic, -3.4027852368936022) and near(b.pvalue, 0.0027668133596095543)

test_exact()
