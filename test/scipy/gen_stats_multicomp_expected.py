# Generates expected values for test/scipy/test_stats_multicomp.py (scipy 1.18.1).
import numpy as np
import scipy.stats as st
from scipy.stats import multivariate_t

def emit(name, v):
    print(f"{name} = {np.asarray(v, dtype=float).tolist()!r}")

g1 = [24.5, 23.5, 26.4, 27.1, 29.9]
g2 = [28.4, 34.2, 29.5, 32.2, 30.1]
g3 = [26.1, 28.3, 24.3, 26.2, 27.8, 25.9]
r = st.tukey_hsd(g1, g2, g3)
emit('th_s', r.statistic); emit('th_p', r.pvalue)
ci = r.confidence_interval(0.9); emit('th_lo', ci.low); emit('th_hi', ci.high)
r = st.tukey_hsd(g1, g2, g3, equal_var=False)
emit('thw_p', r.pvalue)
ci = r.confidence_interval(); emit('thw_lo', ci.low)
emit('sr', [st.studentized_range.cdf(3.5, 4, 10), st.studentized_range.sf(3.5, 4, 10),
            st.studentized_range.pdf(3.5, 4, 10), st.studentized_range.cdf(2.0, 3, 200000),
            st.studentized_range.ppf(0.9, 4, 10)])
ctl = [24.1, 23.9, 25.0, 26.2, 24.4, 25.5]
out = []
for alt in ['two-sided', 'less', 'greater']:
    r = st.dunnett(g1, g2, g3, control=ctl, alternative=alt, rng=7)
    out += list(r.statistic) + list(r.pvalue)
emit('dn', out)
r = st.dunnett(g1, g2, g3, control=ctl, rng=7)
ci = r.confidence_interval(); emit('dn_lo', ci.low); emit('dn_hi', ci.high)
rho = np.array([[1, .5, .2], [.5, 1, .3], [.2, .3, 1.]])
emit('mvt', [multivariate_t(shape=rho, df=5, seed=3).cdf([1.0, 0.5, 2.0]),
             multivariate_t(shape=rho, df=5, seed=3).cdf([1.0, 0.5, 2.0], lower_limit=[-1., -2, -0.5])])
