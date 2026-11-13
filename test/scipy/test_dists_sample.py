# Copyright (C) 2026 the Typhon authors
# rv_histogram and rv_discrete(values=...) vs scipy 1.18.1 reference values.
import numpy as np
import scipy.stats as st
from dists_check import close

ok = True
h = st.rv_histogram(([1., 3, 2, 4], [0., 1, 2, 3, 4]))
ok &= close('hist', 'pdf', h.pdf(np.array([0.5, 1.5, 3.9])), [0.1, 0.3, 0.4], 1e-14)
ok &= close('hist', 'cdf', h.cdf(np.array([0.5, 1.5, 3.9])), [0.05, 0.25, 0.96], 1e-14)
ok &= close('hist', 'ppf', h.ppf(np.array([0.1, 0.5, 0.9])), [1.0, 2.5, 3.75], 1e-14)
m, v, s, k = h.stats(moments='mvsk')
ok &= close('hist', 'mvsk', [m, v, s, k],
            [2.4000000000000004, 1.173333333333332, -0.29269198281165243, -1.0800619834710645], 1e-12)
ok &= close('hist', 'entropy', [h.entropy()], [1.2798542258336676], 1e-13)
ok &= close('hist', 'rvs', h.rvs(size=3, random_state=np.random.RandomState(1)),
            [2.08511002e+00, 3.30081123e+00, 1.14374817e-03], 1e-8)

s = st.rv_discrete(values=([1, 4, 2], [0.2, 0.5, 0.3]))
ok &= close('sample', 'pmf', s.pmf(np.array([1., 2, 3, 4])), [0.2, 0.3, 0.0, 0.5], 1e-15)
ok &= close('sample', 'cdf', s.cdf(np.array([1., 2, 3, 4])), [0.2, 0.5, 0.5, 1.0], 1e-15)
ok &= close('sample', 'ppf', s.ppf(np.array([0.1, 0.5, 0.9])), [1.0, 2.0, 4.0], 0.0)
m, v, sk, ku = s.stats(moments='mvsk')
ok &= close('sample', 'mvsk', [m, v, sk, ku],
            [2.8, 1.5600000000000014, -0.23403345556128247, -1.6607495069033584], 1e-12)
ok &= close('sample', 'entropy', [s.entropy()], [1.0296530140645737], 1e-13)
ok &= close('sample', 'rvs', s.rvs(size=3, random_state=np.random.RandomState(1)), [2.0, 4.0, 1.0], 0.0)
ok &= close('sample', 'sf', s.sf(np.array([1., 2])), [0.8, 0.5], 1e-15)
ok &= close('sample', 'isf', s.isf(np.array([0.1, 0.6])), [4.0, 2.0], 0.0)
ok &= close('sample', 'expect', [s.mean(), s.expect(lambda x: x * x)], [2.8, 9.4], 1e-13)

# gumbel_r/l closed-form MLE (scipy: brentq on scale + logsumexp loc)
x = np.array([2.1, 3.4, 1.9, 5.6, 4.2, 3.3, 2.8, 6.1, 4.4, 3.9, 2.2, 5.0])
l, sc = st.gumbel_r.fit(x)
ok &= close('gumbel_r', 'fit', [l, sc], [3.0998081336820302, 1.1282447922121968], 1e-13)
l, sc = st.gumbel_l.fit(x)
ok &= close('gumbel_l', 'fit', [l, sc], [4.414395978072881, 1.2747291981872053], 1e-13)
l, sc = st.gumbel_r.fit(x, floc=1.0)
ok &= close('gumbel_r', 'fit-floc', [l, sc], [1.0, 2.0896317847772057], 1e-13)
l, sc = st.gumbel_r.fit(x, fscale=1.5)
ok &= close('gumbel_r', 'fit-fscale', [l, sc], [3.229974070982352, 1.5], 1e-13)
print('ALL OK' if ok else 'FAILED')
assert ok
